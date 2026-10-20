#!/usr/bin/env python3
"""reproduce/lib/submit_twin.py — Slurm submission for the front door, with the TWIN-SUBMIT start-grant protocol.
stdlib + PyYAML only; the ONLY writer of GRANT / DENY.

  submit   --model K --unit U --lwid L --script S --name N --args "…" [--clusters a,b] [--twin] [--env "K=V …"]
           [--time T] [--array A] [--flags "--gres=gpu:N …"] [--kind target|attacker|fit|other] [--registers registry|none]
           [--group G] [--wave-sha SHA] [--foreground] [--resubmit] [--dry-run]
           --registers: the unit's REGISTRATION CONTRACT, recorded on the attempt:
           `registry` (the default for target / attacker / fit) — the job registers runs or objects (a pending file, a spool record,
           a registry row), so a dead job's --resubmit needs the fenced proof below; `none` (the default for kind `other`; the serving
           probe, the memprobe, the HPO recon declare it) — the job writes only its own record dir, so the proof that it never
           registered is `sacct` saying TERMINAL and nothing more (the fenced proof's ingest watermark is absent on a host no
           `registry ingest --pull` drives, and would strand every such unit).
           --flags: extra sbatch flags appended verbatim after the cluster's CLUSTER_SBATCH_EXTRA (the fit path's data-parallel
           allocation: 04_fit_objects.sh passes lib/common.sh repro_fit_width_flags — --gres=gpu:N --cpus-per-task=16N
           --mem=128N G — for a config whose fit_width is N; empty at width 1). Never --export= (refused).
           IDEMPOTENT: a unit already `submitted`/`registered` in the gate ledger prints `skipped …` and nothing is minted
           (a re-run after a partial wave submits only what is missing); --resubmit re-mints ONLY after the fenced proof
           that the bound job is terminal and never registered.
           Composes the wave shell's own line — `ssh <CLUSTER_SSH> 'cd <checkout> && VAR=value … sbatch --qos=… --partition=…
           [--account=…] [--time=…] --output=<logdir>/%x_%j.out -J <name> <CLUSTER_SBATCH_EXTRA> [--array=…]
           [--comment=sha:<sha> for scripts/sbatch/*] <script> <positionals>'` (never --export=), prints
           `submitted <cluster> <jobid> <unit>`, appends submissions.jsonl + attempts.jsonl, binds the unit `submitted`
           in the gate ledger, launches lib/watch_job.sh. Under --twin the unit is ONE election: the primary is
           submitted with SISTEER_ATTEMPT_ID / SISTEER_ATTEMPTS_ROOT in its env prefix (the in-body hook then waits for a
           grant), the election runs (foreground, or detached), and a twin goes to the other cluster only when the primary
           is still PENDING after SISTEER_DUAL_SUBMIT_THRESHOLD_MIN with fewer than SISTEER_DUAL_SUBMIT_MAX_JOBS units
           left to place. `--twin` REFUSES until the mirrored sbatch at the wave sha carries the hook marker.
  elect    --model K --lwid L [--poll-s 30] [--max-wait-s N]     the election loop (resumable: replays the ledger)
  reconcile --model K            replay every open election against the ledger, the remote sentinels, sacct and the
                                 registry; fail CLOSED on any inconsistency (two GRANTs, a GRANT for an uncommitted
                                 attempt, a run from a non-winner); a registered-then-dead winner is ATTACHED, never
                                 re-granted; a re-grant needs the four-way POSITIVE fenced proof (below)
  cancel-attempt --model K --attempt-id A     DENY + scancel THAT job id only (never $WAVE_ROOT/STOP)
  teardown-pool  --model K --group G          `attackers/<jobid>.STOP` for each member THAT group owns (never attackers/STOP); a member
                                              that already wrote <jobid>.SELF_EXIT (31b) is recorded torn_down, not stopped again; the writes
                                              run under the group's lock and are REFUSED while attackers/PLACED.transition is fresh (a pass is
                                              re-placing the group onto these members) or when attackers/PLACED is no
                                              longer the record this invocation read (a placement committed in between)
  mark-placed    --model K --group G --targets lw,lw…   the group's placement is COMPLETE: every named target bound -> attackers/PLACED
                                              {targets: [{job_id, logical_work_id}]} on the group's cluster + ledger `placement_complete`;
                                              a member nothing bound may then leave on its own (attacker_pool_lifecycle.sh); a member whose
                                              attackers/<job>.SELF_EXIT stands under the group's lock is EXCLUDED from the assignment, named
                                              in the answer and the record; with no member left it refuses rc 4
                                              (`no_live_member`, nothing written) — re-run the same 06 --resubmit pass once sacct shows the
                                              exited job(s) terminal: its walk re-mints the member and that pass's mark-placed commits
  placement-transition open|close --model K --group G --cluster L [--reason R] [--lease-s N]   the group's placement is IN TRANSITION:
                                              attackers/PLACED.transition {from_version, reason, lease_s, epoch} on the group's cluster, written
                                              by a pass BEFORE it submits replacements, so a re-minted member never leaves on the placement that
                                              predates it; mark-placed clears it when a placement commits, `close`
                                              clears it by hand, and a member ignores it once its lease lapsed (default 2 h)
  wait-pool      --model K --group G [--poll-s 60] [--max-hours 48]   the group-level backstop: poll sacct for the group's targets,
                                              enforce ONE member per three live targets (the surplus marked RETIRING, 31d), and
                                              teardown-pool the poll after the last target is terminal (31b); both HOLD while the group's
                                              attackers/PLACED.transition is fresh — checked under the group's lock at the write — and the
                                              waiter polls on until a placement commits or the marker's lease lapses;
                                              each write also carries the PLACED version the poll decided on, and a placement that committed
                                              after that read answers `stale` under the lock — nothing written, the next poll reads it
  retire-member  --model K --group G --job J     mark ONE member RETIRING (attackers/<job>.RETIRE + ledger `retiring`): it withdraws its
                                              record, drains, and self-exits; its drivers re-bind to a survivor (31d); refused while the
                                              group's placement is in transition or when attackers/PLACED moved since it was read
  job-env        --cluster L [--export]       READ-ONLY: the job environment every sbatch line this funnel composes for cluster L
                                              carries (Coordinator.job_env, the ONE list): the cluster block's roots and labels, the
                                              registry spool, push helper, campaign and fit venv name, and the site's W&B entity and
                                              project when the site names an entity. Printed as the line's own prefix (`K=V …`,
                                              shell-quoted) or, with --export, one `export K=V` line each for a job submitted by hand.
                                              lib/slurm.sh slurm_job_env prints this, so a dry run shows exactly what a live job carries.
                                              The site is read as a dry run reads it (nothing is created); --model is not needed
  job-env        --wandb [--export]           READ-ONLY: only the site's W&B destination from that list (site_wandb_env; empty when the
                                              site names no entity), for a fit job a launcher script submits outside this funnel: it
                                              takes the W&B keys and none of the registry or root values (scripts/stage_llama_fits.sh)

The election is CRASH-SAFE: the winner is committed to the grading host's ledger (append + fsync) BEFORE any remote write;
GRANT before any DENY; each sentinel is tmp + mv (atomic). A restart replays the ledger. A re-grant after a winner died
needs POSITIVE proof it never registered, all four read-only at the WINNING cluster: (i) sacct TERMINAL; (ii) the
attempt's pending file (<pending dir as submitted>/<jobid>.json) ABSENT; (iii) no record under the attempt's PRODUCER
spool (SISTEER_REGISTRY_SPOOL as submitted) naming the job id or the attempt id — a missing or failed traversal is
UNKNOWN, never absence; (iv) no `run` in the registry with that attempt id / job id AND the cluster's inbox ingested
past the job's end. Any of the four unreachable, unknown or positive -> HALT (named), cancel nothing. Before a
replacement is granted, the dead winner's GRANT is tombstoned (GRANT -> GRANT.retired.<utc>), so one unit never
carries two live GRANT sentinels.
The fit family (`--kind fit`: 02 memprobe, 03 recon / trials / confirm-pick, 04 fit cells / derived builds / the refusal direction):
every job env carries SDFBASE_FIT_ENV_NAME = the TARGET cluster's prebuilt fit venv name
(CLUSTER_FIT_ENV_NAME > SISTEER_FIT_ENV_NAME > sdfbase-fit — the sbatches' own reader, so a job on the other cluster runs on
THAT cluster's venv), and the attempt record carries the venv's identity read on the target cluster BEFORE the identity is
minted: `$VENV/BUILD-COMPLETE`'s second field (the sha16 of the build's freeze.txt — sdfbase_fit_env.sbatch step 6) and the
sha256 of `$VENV.build/requirements.txt`. A venv without BUILD-COMPLETE refuses by name (the job would die at $VENV/bin/python);
a cross-cluster pair (a twin, a resubmit on the other cluster) whose identities differ refuses by name — two clusters never
fit one unit on two stacks.
Test seam: CLUSTER_SSH=local:<dir> executes the remote command with `bash -c` on this host with FAKE_SLURM_DIR=<dir>
(fake sbatch/squeue/sacct/scancel on PATH); SISTEER_TWIN_CRASH_AT=<point>[,…] exits 97 at that point (fault injection).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import sqlite3
import subprocess
import sys
import time
from typing import Dict, List, Mapping, Optional, Sequence, Set, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import clusters as C  # noqa: E402
import stage as ST  # noqa: E402
from grant_ledger import Ledger, utc_now  # noqa: E402

#: the job env variable carrying the submission's own UTC clock (ISO-8601 `Z`), exported into EVERY unit's sbatch env by the funnel and
#: recorded on the attempt (`submitted_utc`): the native standup's staging-stamp fence reads it
SUBMITTED_UTC_ENV = "SISTEER_UNIT_SUBMITTED_UTC"

HOOK_MARKER = "sisteer-attempt-wait-hook"
#: the recorded submission spec (`attempt.spec` in attempts.jsonl) — what a twin or a re-submit re-issues verbatim
SPEC_KEYS = ("unit", "lwid", "script", "name", "args", "env", "sha", "time_limit", "array", "kind", "group", "clusters", "flags", "depends_on", "provenance")
#: the JOB-side keys a resume may carry outside the frozen workload spec (31f-fix): the attacker pin and its wait
#: bound — operational facts about the pool the resumed job binds, never a knob that alters the drivers' work (a seed, a decode policy, a
#: serving path, a registry route: those are the frozen spec's and ride --env, where the freeze guard sees them). Anything else, and any key
#: the frozen spec already names, refuses by name.
RESUME_JOB_ENV_ALLOWED = ("F22_ATTACKER_JOBS", "F22_ATTACKER_WAIT_S")
#: closer-fix-2: an eval job whose store closer REFUSED leaves with this exit code (common.sh f22_result) and a recipe
#: beside its pending file; the funnel's proof turns it into `reclose` — a CPU-only reclose_stores.sbatch at the recipe's sha re-runs the
#: closer under the driver venv — never a re-run of the eval. Capped per job; the same refusal surviving a reclose stops the automation.
CLOSER_REFUSED_EXIT = 75
RECLOSE_SCRIPT = "scripts/sdfbase/full22/reclose_stores.sbatch"
RECLOSE_MAX = 3
#: the paper frame's registry campaign — the LAST fallback of Site.registry_campaign (the same word 04's CAMPAIGN= line and common.sh
#: repro_registry_campaign end in); a site names another through SISTEER_REGISTRY_CAMPAIGN or REGISTRY_CAMPAIGN
REGISTRY_CAMPAIGN_DEFAULT = "sdfbase_full22"


_UNIT_LOCK_FD = None


def unit_lock_path(ledger_path: str, lwid: str) -> str:
    return os.path.join(ST.bindings_dir(ledger_path), ".locks", f"{ST.safe_id(lwid)}.unit.lock")   # outside the `<unit>.*` names readers glob as bindings


def acquire_unit_lock(ledger_path: str, lwid: str):
    """The per-unit lock (fcntl, exclusive) a `submit` or `reconcile-token` holds for its whole invocation. Bounded wait
    (SISTEER_UNIT_LOCK_TIMEOUT_S, default 600 s, polled); a lock still held then refuses by name — nothing reserved, launched or cleared."""
    import fcntl
    p = unit_lock_path(ledger_path, lwid); os.makedirs(os.path.dirname(p), exist_ok=True)
    fd = open(p, "a+")
    deadline = time.time() + float(os.environ.get("SISTEER_UNIT_LOCK_TIMEOUT_S", "600"))
    while True:
        try:
            fcntl.flock(fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return fd
        except OSError:
            if time.time() >= deadline:
                fd.close()
                raise Refused(f"{lwid}: the unit lock {p} is held by another funnel process (a submit or a reconciliation of this unit) — "
                              f"nothing reserved, launched or cleared; retry when it is released", 3)
            time.sleep(0.5)


def comment_with_nonce(flags: List[str], nonce: Optional[str]) -> List[str]:
    """The reservation's nonce on the sbatch line as `--comment=…nonce:<n>` — merged into an EXISTING comment (the fit path's `--comment=sha:<sha>`
    becomes `sha:<sha>,nonce:<n>`: one Slurm comment carries both; `,` because the funnel joins its flags into one unquoted shell line, where `;`
    would end the command), appended otherwise (without the merge, a path that already carries a comment suppresses the
    nonce, leaving its jobs permanently unverifiable)."""
    if not nonce:
        return list(flags)
    out = list(flags)
    for i, f in enumerate(out):
        if f.startswith("--comment="):
            out[i] = f + f",nonce:{nonce}"
            return out
    out.append(f"--comment=nonce:{nonce}")
    return out


def recipe_identity(recipe: dict) -> str:
    """closer-fix-6: the reclose recipe's content identity — every key but its receipts (`recloses`), the operator's `retries` and the funnel's
    own `path` — canonical JSON, sha256. The reclose job's preflight receipt records the same hash (reclose_stores.sbatch, its twin): a recipe
    repaired after a preflight refusal hashes differently, and that difference is the new evidence one more reclose needs."""
    import hashlib
    body = {k: v for k, v in recipe.items() if k not in ("recloses", "retries", "path")}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def nonce_of_comment(comment: str) -> str:
    """The nonce a job's comment carries (`nonce:<hex>` alone or beside other fields, `,`- or `;`-separated), '' when none."""
    m = re.search(r"(?:^|[,;])nonce:([0-9a-f]{6,})(?:[,;]|$)", str(comment or "").strip())
    return m.group(1) if m else ""
ACTIVE = ("PENDING", "RUNNING", "REQUEUED", "RESIZING", "SUSPENDED", "COMPLETING", "CONFIGURING")


class Refused(SystemExit):
    def __init__(self, msg: str, rc: int = 2):
        print(f"REFUSED: {msg}", file=sys.stderr)
        super().__init__(rc)


class Unreachable(RuntimeError):
    pass


class AccountingUnavailable(RuntimeError):
    """sacct or squeue answered with an ERROR (slurmdbd down, a database timeout, the controller unreachable) — an EMPTY answer under it
    proves nothing about a job's existence, so every reader of absence fails closed (31e-fix-3)."""


def _crash(point: str) -> None:
    pts = {p for p in os.environ.get("SISTEER_TWIN_CRASH_AT", "").split(",") if p}
    if point in pts:
        print(f"[twin] CRASH INJECTED at {point}", file=sys.stderr, flush=True)
        os._exit(97)


def log(msg: str) -> None:
    print(f"[twin {utc_now()}] {msg}", flush=True)


# ------------------------------------------------------------------------------------------------------------ site
class Site:
    def __init__(self, config: str, dry: bool):
        self.config_path = os.path.abspath(config)
        self.site_dir = os.path.dirname(self.config_path)
        # THE config file this run was given (--config; config.env by default) is what the site's scalars — the registry campaign, the
        # primary cluster, the dual-submit knobs, the fit venv name — and its cluster LIST are read from; never <site dir>/config.env
        # beside it. Reading that neighbour instead strands a run: `04 --config <dir>/experiment.env` naming campaign B has the
        # stage read done-ness under B — repro_load_config sources the file it is given — while every sbatch this funnel composed
        # states config.env's A, so a completed fit is filed where the pins writer never looks. The blocks stay
        # <site dir>/clusters/<label>.env, for the labels the given file names — the same two reads common.sh repro_load_config
        # makes (clusters.py --config "$SISTEER_CONFIG").
        self.cfg = C.site_config_file(self.config_path)
        self.clusters = C.load_all(self.site_dir, config=self.cfg)
        self.dry = dry
        # EVERY scalar below is read through `given` — the file laid over the environment, the stage's own view: the two
        # ROOTS read the environment FIRST (`os.environ.get(k) or cfg.get(k)`), so a direct `submit_twin.py --config
        # <file> … submit` from a shell carrying SISTEER_STATE_DIR / SISTEER_REGISTRY_ROOT wrote its ledger outside the state root the
        # file names and `--resubmit` read an unrelated registry for its fenced proof, while a stage over the same file (which sources the
        # file OVER its environment) read the file's; and the other scalars read the FILE ALONE, so a key the environment carried and the
        # file did not name was the stage's value and not the funnel's. One rule for all of them now, the campaign's.
        self.primary = self.given("SISTEER_PRIMARY_CLUSTER") or next(iter(self.clusters))
        if self.primary not in self.clusters:          # the shell refuses the same way (repro_load_config) — never a label with no block
            raise C.ClusterConfigError(f"SISTEER_PRIMARY_CLUSTER={self.primary!r} is not one of the clusters {self.config_path} names ({list(self.clusters)})")
        self.threshold_s = 60 * int(self.given("SISTEER_DUAL_SUBMIT_THRESHOLD_MIN") or 10)
        self.max_jobs = int(self.given("SISTEER_DUAL_SUBMIT_MAX_JOBS") or 16)
        self.grant_timeout_min = int(self.given("SISTEER_GRANT_TIMEOUT_MIN") or 10)
        self.state_root = os.path.expandvars(os.path.expanduser(self.given("SISTEER_STATE_DIR") or "~/.local/state/sisteer-reproduce"))
        self.registry_root = os.path.expanduser(self.given("SISTEER_REGISTRY_ROOT") or "~/registry")
        self.reproduce_root_tmpl = self.given("SISTEER_REPRODUCE_ROOT") or "$SHARED/exp/reproduce"

    def given(self, key: str) -> Optional[str]:
        """A site key READ THE WAY A STAGE READS IT: the config file this run was GIVEN laid over the process environment. A key the
        file names is the file's — empty included (`KEY=` under `set -a; source` sets it empty, and `${KEY:-default}` then reads it as
        unset, as a caller's `or default` does here); a key the file does not name is the environment's (None when neither names it).
        This is exactly repro_load_config (`set -a; source "$SISTEER_CONFIG"`) followed by the shell's `${KEY:-…}` reads, so the
        funnel and the stage that launches it resolve ONE value for every site key whether or not a stage sourced the file first
        (the direct `--config <file>` CLI, the README's cancel-attempt / teardown-pool lines). Every reader of a site scalar in this
        module goes through it (the roots, the primary, the dual-submit knobs, the fit venv name, the spool, the models table, the
        mirror-status path, the campaign); an environment-first or a file-only read is the bug class
        tests/test_site_config_over_environment.py sweeps for."""
        return self.cfg[key] if key in self.cfg else os.environ.get(key)

    def state_dir(self, model: str) -> str:
        d = os.path.join(self.state_root, model)
        if not self.dry:                       # a dry run computes paths and creates NOTHING
            os.makedirs(d, exist_ok=True)
        return d

    def cluster_root(self, label: str, model: str) -> str:
        shared = self.clusters[label]["CLUSTER_SHARED_ROOT"]
        return self.reproduce_root_tmpl.replace("$SHARED", shared) + "/" + model

    def other(self, label: str) -> Optional[str]:
        for lab in self.clusters:
            if lab != label:
                return lab
        return None

    def fit_env_name(self, label: str) -> str:
        """The prebuilt fit venv NAME on `label`: the cluster block's CLUSTER_FIT_ENV_NAME, else the site's SISTEER_FIT_ENV_NAME, else
        `sdfbase-fit` — the same resolution as common.sh cluster_fit_env_name, the site's key read as the stage reads it (`given`)."""
        return self.clusters[label].get("CLUSTER_FIT_ENV_NAME") or self.given("SISTEER_FIT_ENV_NAME") or "sdfbase-fit"

    def fit_env_dir(self, label: str) -> str:
        return f"{self.clusters[label]['CLUSTER_ENVS_DIR']}/{self.fit_env_name(label)}"

    def registry_campaign(self) -> str:
        """The registry campaign every job of this site STATES (REGISTRY_CAMPAIGN in its env) — the ONE resolution 04's own CAMPAIGN= line,
        05's, the pins writer (make_full22_configs.registry_campaign_default) and common.sh repro_registry_campaign share, read the way
        a stage reads it: the config file this run was GIVEN laid over the environment for BOTH spellings (repro_load_config `set -a;
        source`s that file over the stage's environment, so a key the file names — empty included, which `${KEY:-}` then reads as
        unset — is the file's and a key it does not name is the environment's; never config.env beside a differently named --config
        file), then SISTEER_REGISTRY_CAMPAIGN, else the job-env spelling REGISTRY_CAMPAIGN, else the paper
        frame's sdfbase_full22. Reading the site's key alone and defaulting breaks a site that names REGISTRY_CAMPAIGN: it fits and
        reads done-ness under its campaign while every fit job files under sdfbase_full22, and the pins writer, reading under the
        site's campaign, finds no row and writes no pin. Consulting the given file for SISTEER_REGISTRY_CAMPAIGN alone and the
        environment for REGISTRY_CAMPAIGN breaks it too: a direct `submit_twin.py --config <file> … submit` — no stage having
        sourced the file into the environment — with REGISTRY_CAMPAIGN alone in that file states sdfbase_full22 on the sbatch line
        while the stage's resolver answers the file's campaign over the same file, so a job submitted that way files where 04 and
        the pins writer never read. `given` (the file over the environment) is the read every site scalar of this module makes."""
        return self.given("SISTEER_REGISTRY_CAMPAIGN") or self.given("REGISTRY_CAMPAIGN") or REGISTRY_CAMPAIGN_DEFAULT


class Remote:
    """One cluster's login host. Every command is ONE ssh string (BatchMode); `local:<dir>` runs it on this host."""

    def __init__(self, site: Site, label: str):
        self.site = site; self.label = label; self.block = site.clusters[label]
        self.host = self.block["CLUSTER_SSH"]; self.shared = self.block["CLUSTER_SHARED_ROOT"]
        self.local_dir = self.host[len("local:"):] if self.host.startswith("local:") else None

    def sh(self, cmd: str, timeout: int = 180, allow_fail: bool = False) -> Tuple[int, str, str]:
        if self.site.dry:
            print(f"DRY-RUN: ssh {self.host} {shlex.quote(cmd)}")
            return 0, "", ""
        # NEVER the caller's stdin (`ssh -n` + stdin=/dev/null on both branches — the rule common.sh repro_ssh always had): the
        # stages call this funnel INSIDE `while read` loops fed by a here-string (06's unit plan), and an ssh that inherits that
        # stdin forwards the remaining lines to the remote, so the loop ends after its first unit — the u0 wave of 2026-09-30
        # placed 1 of 32 units and reported `06 done` (the dry run, which runs no ssh, had placed all 37)
        if self.local_dir is not None:
            env = dict(os.environ); env["FAKE_SLURM_DIR"] = self.local_dir
            p = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, timeout=timeout, env=env, stdin=subprocess.DEVNULL)
        else:
            p = subprocess.run(["ssh", "-n", "-o", "BatchMode=yes", "-o", "ConnectTimeout=30", self.host, cmd], capture_output=True, text=True,
                               timeout=timeout, stdin=subprocess.DEVNULL)
        if p.returncode == 255:      # ssh's own failure code; a fake in local mode exits 255 to emulate an outage
            raise Unreachable(f"ssh {self.host}: {p.stderr.strip()[:200]}")
        if p.returncode != 0 and not allow_fail:
            raise RuntimeError(f"[{self.label}] `{cmd[:160]}` rc={p.returncode}: {p.stderr.strip()[:400]}")
        return p.returncode, p.stdout, p.stderr

    def sbatch(self, checkout: str, env_prefix: str, flags: List[str], script: str, args: List[str]) -> str:
        line = f"cd {shlex.quote(checkout)} && {env_prefix}{' ' if env_prefix else ''}sbatch {' '.join(flags)} {script} {' '.join(shlex.quote(a) for a in args)}".strip()
        rc, out, err = self.sh(line)
        if self.site.dry:
            return "DRY"
        m = re.search(r"Submitted batch job (\d+)", out + err)
        if not m:
            raise RuntimeError(f"[{self.label}] sbatch printed no job id: {out[-200:]} {err[-200:]}")
        return m.group(1)

    def squeue_state(self, job_id: str) -> Optional[str]:
        rc, out, err = self.sh(f"squeue -h -j {job_id} -o %T 2>&1", allow_fail=True)
        if rc != 0:
            if "Invalid job id" in out + err:
                return None
            raise Unreachable(f"[{self.label}] squeue -j {job_id}: {(out + err).strip()[:200]}")
        s = out.strip().splitlines()
        return s[0].strip() if s else None

    def sacct(self, job_id: str) -> Optional[dict]:
        rc, out, _ = self.sh(f"sacct -X -n -P -o JobID,JobName,State,End -j {job_id}")
        rows = [ln.split("|") for ln in out.strip().splitlines() if ln.strip()]
        if not rows:
            return None
        jid, name, state, end = rows[0][:4]
        return {"job_id": jid, "job_name": name, "state": state.split()[0], "end": end, "terminal": state.split()[0] not in ACTIVE}

    def sacct_tasks(self, job_id: str) -> List[dict]:
        """EVERY task of `job_id` as sacct lists it — `[{raw, task, state, terminal, end}]`: an array's children under their RAW ids
        (`sacct -o JobIDRaw,JobID,State`: raw | <array>_<k> | state), a plain job as its one task (task None). The end time is read per
        raw id (a second, per-task sacct) so a watermark can be compared with each task's own end — reading the PARENT id alone
        misses children that registered under their raw ids."""
        rc, out, _ = self.sh(f"sacct -X -n -P -o JobIDRaw,JobID,State,ExitCode -j {shlex.quote(str(job_id))}")
        rows = []
        for ln in out.strip().splitlines():
            parts = ln.split("|")
            if len(parts) < 3 or not parts[0].strip():
                continue
            raw, jid, state = parts[0].strip(), parts[1].strip(), parts[2].split()[0] if parts[2].strip() else ""
            exit_code = parts[3].strip() if len(parts) > 3 and parts[3].strip() else None     # "75:0" = the closer refused (closer-fix-2)
            task = int(jid.rsplit("_", 1)[1]) if "_" in jid and jid.rsplit("_", 1)[1].isdigit() else None
            end = None
            if state not in ACTIVE:
                sac = self.sacct(raw)
                end = sac.get("end") if sac else None
            rows.append({"raw": raw, "task": task, "state": state, "terminal": state not in ACTIVE, "end": end, "exit_code": exit_code})
        return rows

    def spool_files_mentioning(self, needles: List[str], spool: Optional[str]) -> Optional[List[dict]]:
        """Every record file under the attempt's PRODUCER spool that names a needle (a task's raw job id), each with its path RELATIVE to
        the spool root, its sha256 and its record count — the facts the harvester's `record_ingest` row must match for the file to count
        as FOLDED (hole 1). None when the spool is missing or the traversal failed (never proof of absence)."""
        if not spool:
            return None
        pats = " ".join(f"-e {shlex.quote(n)}" for n in needles)
        # the files that NAME the job id locate the job's run DIRECTORIES; every record file in those directories is the job's (a
        # run_patch record carries no job id yet belongs to the run — 12 of a static job's 36 files), so all of them must be folded
        rc, out, err = self.sh(f"d={shlex.quote(spool)}; [ -d \"$d\" ] || {{ echo NOSPOOL; exit 0; }}; "
                               f"grep -rl {pats} \"$d\" 2>/dev/null | xargs -r -n1 dirname | sort -u | while IFS= read -r rd; do "
                               f"for f in \"$rd\"/*.jsonl; do [ -f \"$f\" ] || continue; "
                               f"printf '%s|%s|%s\\n' \"${{f#$d/}}\" \"$(sha256sum \"$f\" | cut -c1-64)\" \"$(grep -c . \"$f\")\"; done; done; echo END", allow_fail=True)
        if rc != 0:
            return None
        lines = (out or "").strip().splitlines()
        if not lines or lines[0].strip() == "NOSPOOL" or lines[-1].strip() != "END":
            return None
        rows = []
        for ln in lines[:-1]:
            parts = ln.split("|")
            if len(parts) == 3 and parts[0]:
                rows.append({"rel": parts[0], "sha256": parts[1], "n_records": int(parts[2] or 0)})
        return rows

    def sacct_by_name(self, name: str, since_utc: str) -> List[dict]:
        """Every (parent) job of `name` submitted since `since_utc` on this cluster — `[{job_id, submit, state}]` (sacct -X --name -S;
        the transition token's reconciliation: a job the funnel launched but never recorded is found by its NAME and SUBMIT time, never
        by a comment — sacct does not retain --comment on these clusters)."""
        rc, out, err = self.sh(f"sacct -X -n -P --name={shlex.quote(name)} -S {shlex.quote(since_utc)} -o JobID,Submit,State", allow_fail=True)
        if rc != 0:
            # 31e-fix-3: the exit code was discarded, so a slurmdbd failure read as "no job of that name" and the
            # reconciliation cleared a reservation whose job was running — an EMPTY answer under an error proves nothing
            raise AccountingUnavailable(f"[{self.label}] sacct --name={name}: rc={rc}: {(err or out or '').strip()[:200]}")
        rows = []
        for ln in (out or "").strip().splitlines():
            parts = ln.split("|")
            if len(parts) >= 3 and parts[0].strip():
                rows.append({"job_id": parts[0].strip().split("_")[0], "submit": parts[1].strip(), "state": parts[2].split()[0] if parts[2].strip() else ""})
        return rows

    def squeue_by_name(self, name: str) -> List[dict]:
        """The LIVE jobs of `name` on this cluster as the controller sees them — `[{job_id, state, submit}]` (31e-fix-3): read to corroborate
        an empty sacct answer, because accounting lags the controller by seconds to minutes after sbatch accepts. rc != 0 is
        AccountingUnavailable — squeue not answering proves nothing either."""
        # squeue -n answers across ALL users — this user's jobs only (-u "$USER", expanded on the cluster), with
        # the owner and the submit time in the row so the caller can hold the hit to the reservation's window
        # %k is the job's --comment — the per-attempt nonce the funnel's sbatch line carries (sacct does not retain comments on these
        # clusters, so it identifies LIVE jobs only; the reservation's snapshot carries the rest)
        rc, out, err = self.sh(f"squeue -h -u \"$USER\" -n {shlex.quote(name)} -o '%i|%T|%V|%u|%k' 2>&1", allow_fail=True)
        if rc != 0:
            raise AccountingUnavailable(f"[{self.label}] squeue -n {name}: rc={rc}: {(out + err).strip()[:200]}")
        rows = []
        for ln in (out or "").strip().splitlines():
            parts = ln.split("|")
            if len(parts) >= 2 and parts[0].strip():
                cm = parts[4].strip() if len(parts) > 4 else ""
                rows.append({"job_id": parts[0].strip().split("_")[0], "state": parts[1].strip(), "submit": parts[2].strip() if len(parts) > 2 else "",
                             "user": parts[3].strip() if len(parts) > 3 else "", "comment": "" if cm in ("(null)", "None") else cm})
        return rows

    def exists(self, path: str) -> bool:
        rc, out, _ = self.sh(f"[ -e {shlex.quote(path)} ] && echo yes || echo no")
        return out.strip() == "yes"

    def write_sentinel(self, path: str, text: str) -> None:
        d = os.path.dirname(path)
        self.sh(f"mkdir -p {shlex.quote(d)} && printf '%s\\n' {shlex.quote(text)} > {shlex.quote(path)}.tmp.$$ && mv -f {shlex.quote(path)}.tmp.$$ {shlex.quote(path)}")

    def scancel(self, job_id: str) -> None:
        self.sh(f"scancel {job_id}", allow_fail=True)

    def spool_mentions(self, needles: List[str], spool: Optional[str]) -> Optional[bool]:
        """True when a record under the attempt's PRODUCER spool (`SISTEER_REGISTRY_SPOOL` exactly as submitted — never a generic
        `$SHARED/registry` guess) names a needle; False ONLY when the spool exists and a complete traversal found nothing; None
        (UNKNOWN) when no spool was recorded, the directory is missing, or grep reported an error (unreadable entries) — a failed
        or missing traversal is never proof of absence."""
        if not spool:
            return None
        pats = " ".join(f"-e {shlex.quote(n)}" for n in needles)
        rc, out, err = self.sh(f"d={shlex.quote(spool)}; [ -d \"$d\" ] || {{ echo NOSPOOL; exit 0; }}; grep -rl {pats} \"$d\"; rc=$?; "
                               f"if [ $rc -le 1 ]; then echo \"END $rc\"; else echo \"ERR $rc\"; fi", allow_fail=True)
        if rc != 0:
            return None
        lines = out.strip().splitlines()
        if not lines or lines[-1] == "NOSPOOL" or not lines[-1].startswith("END"):
            return None
        return len(lines) > 1


# ---------------------------------------------------------------------------------------------------------- units
def placement_move_origin(replaced, label: str, pin_cluster: Optional[str]) -> Optional[str]:
    """move-knob-4: the cluster a --resubmit's placement receipt moves the unit FROM, or None when the unit did not move: the replaced
    attempt's cluster when it differs from the label (the cross-cluster replacement, 24d), else the manifest's PIN when it differs from the
    label (a group declared moved before anything ran on its origin, first placed on the destination by a plain submission: the pin still
    holds the origin, the dead attempt already sits on the destination, and the resubmit lands beside it owing the pin its receipt)."""
    if replaced is None:
        return None
    ob = replaced[0] if isinstance(replaced, (tuple, list)) else replaced
    oc = str((ob or {}).get("cluster") or "")
    if oc and oc != label:
        return oc
    if pin_cluster and str(pin_cluster) != label:
        return str(pin_cluster)
    return None


def manifest_for(state_dir: str, lwid: str) -> Optional[dict]:
    """The view's placement manifest (`gate/expected_units.placement.<view>.json`) as a dict, None when absent or unreadable — read-only."""
    try:
        return ST.read_placement(ST.placement_path(os.path.join(state_dir, "attempts.jsonl"), lwid.split("/")[1]))
    except Exception:
        return None


def placement_receipt_origin(state_dir: str, lwid: str, replaced, label: str, group: Optional[str]) -> Optional[str]:
    """move-knob-5: the cluster a --resubmit's placement receipt moves `lwid` FROM, decided BEFORE anything is submitted, or None when
    the unit does not move. (1) The replaced attempt on ANOTHER cluster: the classic cross-cluster replacement (24d) — its cluster. (2) The
    replaced attempt on THIS cluster while the manifest's EFFECTIVE pin is elsewhere: ONLY when the manifest DECLARES the group's move to
    `label` (ST.declared_destination_binding: no receipt yet, the pin at the move's origin or an earlier declared destination) AND the replaced
    attempt sits at that declared destination in the destination's own group (`mt-<view>-<label>`) — the pin; every other shape is REFUSED by
    name with no new job: an undeclared wrong-cluster placement (a plain submission on a cluster the pin never named, registered and dead)
    must not be laundered into a durable receipt through the rerun branch. (3) Pin equal to the label, or no pin:
    None."""
    if replaced is None:
        return None
    ob = replaced[0] if isinstance(replaced, (tuple, list)) else replaced
    oc = str((ob or {}).get("cluster") or "")
    if oc and oc != label:
        return oc
    pin = manifest_pin_cluster(state_dir, lwid)
    if not pin or str(pin) == label:
        return None
    man = manifest_for(state_dir, lwid) or {}
    pin_rec = ((man.get("units") or {}).get(lwid) or {})
    view = lwid.split("/")[1] if lwid.count("/") >= 1 else ""
    declared = ST.declared_destination_binding(man, lwid, pin_rec, label)
    at_destination = (oc == label) and (not group or str(group) == f"mt-{view}-{label}") and str((ob or {}).get("group") or f"mt-{view}-{label}") == f"mt-{view}-{label}"
    if declared and at_destination:
        return str(pin)
    raise Refused(f"{lwid}: --resubmit on {label} would replace attempt {(ob or {}).get('attempt_id')} (job {(ob or {}).get('job_id')}, "
                  f"{replaced[2] if isinstance(replaced, (tuple, list)) and len(replaced) > 2 else '?'}) on {oc or '?'} while the placement manifest pins the "
                  f"unit on {pin}" + ("" if declared else f" and declares no move to {label}") + (" (the replaced attempt is not at the declared destination's group)" if declared else "")
                  + " — an unreceipted move; NOTHING submitted (declare the move with `placement-manifest ensure --move-group <group>=<label> --move-reason …` and "
                  "resubmit under it, or resubmit on the pinned cluster; move-knob-5)", 3)


def placement_receipt_record(lwid: str, moved_from: str, dest: str, group: Optional[str], old_binding: Mapping[str, object], replaced_state: str,
                             why: str, aid: str, job_id: str) -> dict:
    """THE placement receipt of a cross-cluster replacement — ONE composer for the launcher (written after submit_one returns) and for
    an adoption from a transition token (Coordinator.receipt_after_adoption: the launcher died between sbatch and this write), so the two
    paths record the same event. `via_pin`: the move is measured against the manifest's PIN when the replaced attempt
    already sat on the destination."""
    via_pin = moved_from != old_binding.get("cluster")
    return {"event": "placement_receipt", "logical_work_id": lwid, "from_cluster": moved_from, "to_cluster": dest,
            "to_group": group, "replaced_job_id": str(old_binding.get("job_id")), "replaced_attempt_id": old_binding.get("attempt_id"),
            "replaced_state": replaced_state, "attempt_id": aid, "job_id": str(job_id),
            "reason": (f"fenced dead-job replacement under a declared move (submit_twin --resubmit; move-knob-4/5: the manifest declares the group's move "
                       f"{moved_from} -> {dest}): the pin held the move's origin {moved_from}, nothing ran there, the replaced attempt already sat on "
                       f"{old_binding.get('cluster')}: {why[:300]}" if via_pin
                       else f"fenced dead-job replacement on another cluster (submit_twin --resubmit): {why[:300]}")}


def manifest_pin_cluster(state_dir: str, lwid: str) -> Optional[str]:
    """The placement manifest's EFFECTIVE pinned cluster of `lwid`: the pin in the view's `gate/expected_units.placement.<view>.json`
    advanced under the LATEST placement receipt the ledgers already carry for the unit (a receipted move's pin is stale until the next
    `ensure`; the receipt's `to_cluster` is where the unit is). None when there is no manifest or no pin — read-only; a malformed manifest
    reads as no pin (the ensure/check refuse it by name on their own)."""
    try:
        view = lwid.split("/")[1]
        ledgers = [os.path.join(state_dir, n) for n in ("attempts.jsonl", "submissions.jsonl") if os.path.isfile(os.path.join(state_dir, n))]
        man = ST.read_placement(ST.placement_path(os.path.join(state_dir, "attempts.jsonl"), view))
        pin = str((((man or {}).get("units") or {}).get(lwid) or {}).get("cluster") or "") or None
        if pin is None:
            return None
        last = None
        for led in ledgers:
            for rc in ST.placement_receipts(led).get(lwid) or []:
                if rc.get("to_cluster") and (last is None or str(rc.get("utc") or "") >= str(last.get("utc") or "")):
                    last = rc
        return str(last["to_cluster"]) if last is not None else pin
    except Exception:
        return None


def attempt_id(lwid: str, label: str, n: int) -> str:
    return f"{lwid}@{label}@{n}"


REGISTERS_REGISTRY, REGISTERS_NONE = "registry", "none"
#: kinds whose units register nothing unless told otherwise: `other` is the serving probe's kind (its own record dir, no registry row)
NO_REGISTRATION_KINDS = ("other",)
#: registry lifecycle states PAST `running`: a run the dead job left there is a RESULT, and a same-identity re-run would overwrite it
PAST_RUNNING_STATES = ("generated", "judging", "judged", "ingested")
#: THE ALLOWLIST of script (+ mode) combinations known to register NOTHING — the only ones a `none` contract may name (without it,
#: any kind would be accepted on `--registers none` and the terminal Slurm state alone). The value is the job-env condition
#: the mode rides on (None = the script never registers): the memprobe and the serving probe write only their own record dirs; the HPO
#: array registers nothing in its RECON mode and emits fit_trial in its trial mode, so only `HPO_JOB_MODE=recon` is `none`.
NO_REGISTRATION_SCRIPTS: Dict[str, Optional[Tuple[str, str]]] = {
    "scripts/sbatch/memprobe_fit.sbatch": None,
    "scripts/sbatch/serving_probe.sbatch": None,
    "scripts/sbatch/hpo_trial.sbatch": ("HPO_JOB_MODE", "recon"),
    # the multi-turn attacker pool SERVES its group's targets and registers nothing (no pending file, spool record or registry row in the
    # script; the plan's attacker rows say registers=False) — it used to derive `registry` and a torn-down pool could never be re-minted
    "scripts/sdfbase/full22/attacker_pool.sbatch": None,
    # the dose ladder's CPU bank-view build (09 --step views): make_bank_view.py build + --check per view under the cluster's bank —
    # symlinks, sha256 and VIEW_MANIFEST.json, no f22_init, no pending file, no spool record, no registry row — without this entry it
    # derives `registry` as an unknown script, so a dead build's --resubmit asks for proof of a producer spool the job never names
    # and the dose sequence strands at --step guard; sacct-terminal is its whole proof
    "scripts/sdfbase/full22/strength/build_views.sbatch": None,
}


#: petri-completion (2026-10-13): the ONE script whose dead job's runs PAST `running` may still be re-minted — a Petri proxy whose drive filed
#: some legs before the proxy died (the demo's 1010177: leg 9 COMPLETE, its 7 runs `generated`, 63 `running`, TIMEOUT at 04:00:00). The
#: proof is the DRIVE'S OWN done-ness reading of those legs (`scripts/petri_run.py legs-status` over the stores the registry names:
#: results.json bound to its manifest and reconciled with the store — the predicate the resumed drive's already-done path runs, at the
#: same paths, for the same identity), never a heuristic: a leg the resumed drive would REGENERATE keeps the refusal. Any other script's
#: past-running run refuses by name as before (a result, never re-run).
PETRI_PROXY_SCRIPT = "scripts/sdfbase/full22/petri_proxy.sbatch"
PETRI_RUN_ENV = "SISTEER_PETRI_RUN"          # a TEST seam (like SISTEER_TWIN_CRASH_AT / SISTEER_NO_WATCHER): the legs-status script; production never sets it


def _script_matches(script: Optional[str], known: str) -> bool:
    key = str(script or "").lstrip("./")
    return bool(key) and (key == known or key.endswith("/" + known))


def resume_proof_for(script: Optional[str]):
    """The script-keyed RESUME PROOF a past-running run may be re-minted under, or None (the existing refusal)."""
    for known, proof in RESUME_PROOFS.items():
        if _script_matches(script, known):
            return proof
    return None


def petri_resume_proof(co: "Coordinator", lwid: str, att: Mapping[str, object], job_id: str, past: Mapping[str, Sequence[str]]) -> Tuple[str, dict]:
    """The Petri proxy's resume proof over `past` ({raw task id: [run ids past running]}): every such run must sit on a leg the resumed
    drive reads as DONE. Returns (the note the RERUN line prints, the proof record for the ledger) or raises Refused (rc 3) by name:
    the stores live on the grading host (SISTEER_GRADING_HOST must be `local` — the proof reads them locally, never over an ssh guess);
    every run names a store_uri in the registry; the stores share ONE out root (`<OUT>/cells/<petri_run_key>`) carrying the frozen
    enumeration `cells.json`; the audit's manifest is on this tree (`configs/capeval/manifests/petri_full22_<key>[_<view>].yaml` from
    the attempt's `spec.args[0:2]`, the drive's and the daemon's rule); `legs-status` runs under the site's interpreter (SISTEER_PYTHON,
    else this one) and reads the record; each run's store is a leg of the frozen enumeration, `done` and not `refused`."""
    spec = dict(att.get("spec") or {})
    # the two site keys this proof reads go through Site.given -- the given --config file over the environment, the way a stage reads them
    # (integration round 3a: the lane predates the round-3 Site.given rule; tests/test_site_config_over_environment.py sweeps for cfg.get)
    host = str(co.site.given("SISTEER_GRADING_HOST") or "local")
    if host != "local":
        raise Refused(f"--resubmit refused: {lwid}: petri resume proof: the Petri stores live on the grading host {host!r} and the proof reads them locally — "
                      "run 06 on the grading host, or re-run after a ruling", 3)
    run_ids = sorted({str(r) for rids in past.values() for r in rids})
    stores = co.registry_run_stores(run_ids)
    for rid in run_ids:
        if not stores.get(rid):
            raise Refused(f"--resubmit refused: {lwid}'s bound job {job_id} left run {rid} past `running` in the registry, but it names no store_uri — its leg cannot be "
                          "located; a result that cannot be located is never re-run", 3)
    roots = sorted({os.path.dirname(os.path.dirname(os.path.abspath(stores[rid]))) for rid in run_ids})
    if len(roots) != 1:
        raise Refused(f"--resubmit refused: {lwid}: petri resume proof: the past-running runs' stores sit under {len(roots)} out roots ({roots}) — one frozen "
                      "enumeration cannot read them all; nothing re-minted", 3)
    out_root = roots[0]
    if not os.path.isfile(os.path.join(out_root, "cells.json")):
        raise Refused(f"--resubmit refused: {lwid}: petri resume proof: no frozen enumeration at {out_root}/cells.json — the resumed drive would re-enumerate; "
                      "nothing is provably done", 3)
    args = [str(x) for x in (spec.get("args") or [])]
    if len(args) < 2:
        raise Refused(f"--resubmit refused: {lwid}: petri resume proof: the attempt record's spec.args {args} name no <key> <view> — the audit's manifest cannot be located", 3)
    key, view = args[0], args[1]
    tag = key if view == "main" else f"{key}_{view}"
    repo = os.path.dirname(os.path.dirname(HERE))
    manifest = os.path.join(repo, "configs", "capeval", "manifests", f"petri_full22_{tag}.yaml")
    if not os.path.isfile(manifest):
        raise Refused(f"--resubmit refused: {lwid}: petri resume proof: no Petri manifest at {manifest} on this tree — the drive's done-ness reading needs the audit's manifest", 3)
    python = str(co.site.given("SISTEER_PYTHON") or "") or sys.executable
    script = os.environ.get(PETRI_RUN_ENV) or os.path.join(repo, "scripts", "petri_run.py")
    cmd = [python, script, "--manifest", manifest, "legs-status", "--out-root", out_root]
    env = dict(os.environ, PYTHONPATH=os.path.join(repo, "src") + (os.pathsep + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else ""))
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=1800)
    except (OSError, subprocess.SubprocessError) as exc:
        raise Refused(f"--resubmit refused: {lwid}: petri resume proof: legs-status could not run ({type(exc).__name__}: {exc}) — the interpreter must import the petri "
                      "dependency group (08's contract)", 3)
    if r.returncode != 0:
        tail = (r.stderr or r.stdout or "").strip().splitlines()[-3:]
        raise Refused(f"--resubmit refused: {lwid}: petri resume proof: legs-status failed (rc {r.returncode}: {' | '.join(tail)[:600]}) — the interpreter must import the "
                      "petri dependency group (08's contract); nothing re-minted", 3)
    try:
        doc = json.loads(r.stdout)
        legs = list(doc["legs"])
    except (ValueError, KeyError, TypeError):
        raise Refused(f"--resubmit refused: {lwid}: petri resume proof: legs-status printed no JSON document ({(r.stdout or '')[:200]!r}); nothing re-minted", 3)
    if doc.get("record_refused"):
        raise Refused(f"--resubmit refused: {lwid}: petri resume proof: the resumed drive would refuse every leg ({doc['record_refused']}); nothing re-minted", 3)
    by_key = {str(l.get("petri_run_key")): l for l in legs}
    done_legs: Dict[str, dict] = {}
    for rid in run_ids:
        rk = os.path.basename(os.path.normpath(stores[rid]))
        leg = by_key.get(rk)
        if leg is None:
            raise Refused(f"--resubmit refused: {lwid}: petri resume proof: run {rid}'s store {stores[rid]} is not a leg of the frozen enumeration at {out_root} — a stale or "
                          "foreign store; nothing re-minted", 3)
        if leg.get("refused") or not leg.get("done"):
            raise Refused(f"--resubmit refused: {lwid}'s bound job {job_id} left run(s) past `running` in the registry ({dict(past)}) — those are results; attach/resume them, "
                          f"never re-run — a resumed drive would REGENERATE leg {leg.get('idx')} {leg.get('cell_id')} ({leg.get('reason') or ('no results.json' if not leg.get('results_present') else 'not done')})", 3)
        done_legs.setdefault(rk, {"idx": leg.get("idx"), "cell_id": leg.get("cell_id"), "petri_run_key": rk, "run_ids": []})["run_ids"].append(rid)
    names = ", ".join(f"{d['idx']}:{d['cell_id']}" for d in sorted(done_legs.values(), key=lambda d: int(d["idx"])))
    note = (f"{len(run_ids)} run(s) past running sit on leg(s) [{names}] the resumed drive reads as done (petri_run.py legs-status over {out_root}: results.json "
            "bound to its manifest and reconciled with the store — the predicate run's already-done path runs); never regenerated")
    record = {"out_root": out_root, "manifest": manifest, "done_legs": sorted(done_legs.values(), key=lambda d: int(d["idx"])),
              "legs_status_sha256": hashlib.sha256(r.stdout.encode("utf-8")).hexdigest()}
    return note, record


RESUME_PROOFS = {PETRI_PROXY_SCRIPT: petri_resume_proof}


def script_registers_nothing(script: Optional[str], env: Optional[Mapping[str, str]]) -> Optional[bool]:
    """True when `script` (+ the job env's mode) is a known non-registering combination, False when the script is known or assumed to
    register (anything outside the allowlist, or the allowlisted array outside its non-registering mode), None when no script is known
    (an attempt record older than the field)."""
    if not script:
        return None
    key = str(script).lstrip("./")
    for known, rule in NO_REGISTRATION_SCRIPTS.items():
        if key == known or key.endswith("/" + known):
            if rule is None:
                return True
            var, val = rule
            return str((env or {}).get(var, "")) == val
    return False


def registration_contract(kind: Optional[str], registers: Optional[str], *, script: Optional[str] = None,
                          env: Optional[Mapping[str, str]] = None) -> str:
    """The unit's registration contract. `none` is BOUND to the allowlist: an explicit `none` for a script that registers is refused by
    name (a misdeclared contract would authorise a re-mint over results); an explicit `registry` always stands; unspecified derives from
    the script + mode when known, else from the kind (`other` -> none; a record written before the field)."""
    known = script_registers_nothing(script, env)
    if registers == REGISTERS_NONE:
        if known is False:
            raise Refused(f"--registers none contradicts {script!r}{' under ' + ' '.join(f'{k}={v}' for k, v in sorted((env or {}).items()) if k == 'HPO_JOB_MODE') if env and 'HPO_JOB_MODE' in env else ''}: "
                          f"that job registers (a pending file / spool record / registry row), so its dead attempt needs the fenced proof — `none` is for "
                          f"{', '.join(NO_REGISTRATION_SCRIPTS)} (the HPO array in HPO_JOB_MODE=recon only)")
        return REGISTERS_NONE
    if registers == REGISTERS_REGISTRY:
        return REGISTERS_REGISTRY
    if registers not in (None, ""):
        raise Refused(f"--registers must be {REGISTERS_REGISTRY} | {REGISTERS_NONE}, got {registers!r}")
    if known is True:
        return REGISTERS_NONE
    if known is False:
        return REGISTERS_REGISTRY
    return REGISTERS_NONE if (kind or "target") in NO_REGISTRATION_KINDS else REGISTERS_REGISTRY


class Coordinator:
    def __init__(self, site: Site, model: str, overrides: Optional[str] = None):
        self.site = site; self.model = model
        self.state = site.state_dir(model)
        self.ledger = Ledger(os.path.join(self.state, "attempts.jsonl"))
        self.subs = Ledger(os.path.join(self.state, "submissions.jsonl"))
        self.overrides = overrides
        self.remotes: Dict[str, Remote] = {lab: Remote(site, lab) for lab in site.clusters}

    # ---- composing one sbatch -------------------------------------------------------------------------------------
    def checkout(self, label: str, sha: str) -> str:
        return f"{self.site.clusters[label]['CLUSTER_EVAL_REPO_ROOT']}/{sha}"

    def job_env(self, label: str) -> Dict[str, str]:
        b = self.site.clusters[label]
        env = {"SISTEER_SHARED_ROOT": b["CLUSTER_SHARED_ROOT"], "SISTEER_CLUSTER_LABEL": b["CLUSTER_LABEL"],
               "SISTEER_EVAL_REPO_ROOT": b["CLUSTER_EVAL_REPO_ROOT"], "SISTEER_HF_HOME": b["CLUSTER_HF_HOME"],
               # the cluster's directory roots cluster job scripts read and refuse by name without (clusters.py JOB_PATHS;
               # lib/slurm.sh slurm_job_env carries the same four)
               "SISTEER_PROJECT_ROOT": b["CLUSTER_PROJECT_ROOT"], "SISTEER_PETRI_REPO_ROOT": b["CLUSTER_PETRI_REPO_ROOT"],
               "SISTEER_REPO_MIRROR": b["CLUSTER_REPO_MIRROR"],
               "SISTEER_REGISTRY_SUBSTRATE": b["CLUSTER_SUBSTRATE"], "SISTEER_REGISTRY_CLUSTER_LABEL": b["CLUSTER_LABEL"],
               "REGISTRY_SHARED_ROOT": b["CLUSTER_SHARED_ROOT"], "REGISTRY_HOST_TAG": b["CLUSTER_LABEL"],
               # the producer spool: the site's key read as the stage reads it (Site.given), else the emitter's default on that cluster —
               # slurm_job_env's `${SISTEER_REGISTRY_SPOOL:-$shared/registry/records}` after repro_load_config
               "SISTEER_REGISTRY_SPOOL": self.site.given("SISTEER_REGISTRY_SPOOL") or f"{b['CLUSTER_SHARED_ROOT']}/registry/records",
               "SISTEER_REGISTRY_PUSH": f"{b['CLUSTER_SHARED_ROOT']}/scripts/registry/push_records.sh",
               "SISTEER_CLUSTER_QOS": b["CLUSTER_QOS"],       # the fit path's escalation resubmits at the site's QoS (sdfbase_fit_cell.sbatch)
               "SDFBASE_FIT_ENV_NAME": self.site.fit_env_name(label),   # the fit family's prebuilt venv: THIS cluster's name
               # the records' campaign, STATED to every job (the native emitters refuse without it since 2026-10-02 — their silent
               # `two_si` default filed this model's fit objects outside the paper frame's sdfbase_full22): the site's ONE resolution
               # (Site.registry_campaign — 04's own CAMPAIGN= line's, the pins writer's, lib/slurm.sh's and 08's), never the site's key alone
               "REGISTRY_CAMPAIGN": self.site.registry_campaign()}
        # the site's W&B destination (site_wandb_env): a fit job logs under the site's own entity (sdfbase_fit_cell.sbatch turns W&B
        # off without one) and `$WANDB_PROJECT` in a committed config expands on the cluster. This method is the ONE job environment:
        # lib/slurm.sh slurm_job_env prints it through `job-env`, so a dry run never shows a value that a live job does not carry
        env.update(site_wandb_env(self.site))
        return env

    # ---- the fit venv's identity on a cluster (kind fit) -------------------------------------------------------------
    def fit_env_identity(self, label: str) -> Dict[str, Optional[str]]:
        """Read, on `label`, the prebuilt fit venv's certification: BUILD-COMPLETE's two fields (`<iso utc> <sha16 of freeze.txt>`,
        sdfbase_fit_env.sbatch step 6 — the field the sbatches' preflight demands present) and the sha256 of the build dir's
        requirements.txt (the recipe). ONE ssh round trip. Refused BY NAME when BUILD-COMPLETE is absent or malformed: the job's own
        preflight would die at `$VENV/BUILD-COMPLETE` / `$VENV/bin/python` after an identity was minted for it."""
        name = self.site.fit_env_name(label); venv = self.site.fit_env_dir(label)
        if self.site.dry:
            return {"name": name, "venv": venv, "build_complete_utc": None, "build_complete_sha": None, "requirements_sha256": None, "dry_run": "1"}
        cmd = (f"f={shlex.quote(venv + '/BUILD-COMPLETE')}; r={shlex.quote(venv + '.build/requirements.txt')}; "
               f"if [ -s \"$f\" ]; then echo \"BC $(head -1 \"$f\")\"; else echo BC_MISSING; fi; "
               f"if [ -s \"$r\" ]; then echo \"REQ $(sha256sum \"$r\" | cut -d' ' -f1)\"; else echo REQ_MISSING; fi")
        rc, out, err = self.remotes[label].sh(cmd, allow_fail=True)
        if rc != 0:
            raise Refused(f"fit venv {name} on {label}: could not read {venv}/BUILD-COMPLETE (rc={rc}: {(out + err).strip()[:200]}) — nothing submitted")
        bc = next((ln[3:].split() for ln in out.splitlines() if ln.startswith("BC ")), None)
        req = next((ln[4:].strip() for ln in out.splitlines() if ln.startswith("REQ ")), None)
        if not bc or len(bc) < 2 or not re.fullmatch(r"[0-9a-f]{8,64}", bc[1]):
            raise Refused(f"fit venv {name} on {label} is not certified: {venv}/BUILD-COMPLETE is "
                          f"{'absent or empty' if not bc else 'malformed (' + ' '.join(bc)[:80] + ')'} — the fit job would die at $VENV/bin/python; "
                          f"build it (scripts/sbatch/sdfbase_fit_env.sbatch {name}) or point CLUSTER_FIT_ENV_NAME / SISTEER_FIT_ENV_NAME at a "
                          f"BUILD-COMPLETE venv under {self.site.clusters[label]['CLUSTER_ENVS_DIR']}; nothing submitted")
        return {"name": name, "venv": venv, "build_complete_utc": bc[0], "build_complete_sha": bc[1],
                "requirements_sha256": req if req and re.fullmatch(r"[0-9a-f]{64}", req) else None}

    @staticmethod
    def fit_env_mismatch(a: Dict[str, Optional[str]], b: Dict[str, Optional[str]]) -> Optional[str]:
        """Why two clusters' fit venv identities are NOT one stack (None when they agree): the freeze sha (BUILD-COMPLETE) must be equal;
        the requirements sha256 must be equal when BOTH are known (a missing build dir on one side is reported, never treated as equal)."""
        if a.get("build_complete_sha") != b.get("build_complete_sha"):
            return f"BUILD-COMPLETE freeze sha {a.get('build_complete_sha')} != {b.get('build_complete_sha')}"
        ra, rb = a.get("requirements_sha256"), b.get("requirements_sha256")
        if ra and rb and ra != rb:
            return f"requirements.txt sha256 {ra[:16]}… != {rb[:16]}…"
        return None

    def check_fit_env_pair(self, lwid: str, label: str, fid: Dict[str, Optional[str]]) -> None:
        """A second attempt of `lwid` on ANOTHER cluster must run on the same stack as every recorded attempt: refuse by name otherwise."""
        st = self.ledger.state().get(lwid) or {"attempts": {}}
        for aid, a in st["attempts"].items():
            other = a.get("fit_env")
            if not other or a.get("cluster") == label:
                continue
            why = self.fit_env_mismatch(other, fid)
            if why:
                raise Refused(f"{lwid}: the fit venv on {label} ({fid['name']}: {fid['venv']}) is not the stack attempt {aid} runs on {a.get('cluster')} "
                              f"({other.get('name')}: {other.get('venv')}) — {why}; a twin / resubmit across clusters needs ONE certified stack "
                              f"(D70: sdfbase-fit-tf515 on both, identical requirements + freeze shas); nothing submitted")

    def overrides_env(self, label: str, sha: str) -> Dict[str, str]:
        if not self.overrides:
            return {}
        base = os.path.basename(self.overrides)
        # the job side reads the SAME file from the checkout it runs (the overrides live under reproduce/overrides/)
        return {"SISTEER_RUN_OVERRIDES": f"{self.checkout(label, sha)}/reproduce/overrides/{base}"}

    def sbatch_flags(self, label: str, name: str, time_limit: Optional[str], array: Optional[str], script: str, sha: str, extra_flags: str = "") -> List[str]:
        b = self.site.clusters[label]
        logdir = b["CLUSTER_LOG_DIR"]        # clusters.load_cluster fills the default
        flags = [f"--qos={b['CLUSTER_QOS']}", f"--partition={b['CLUSTER_PARTITION']}"]
        if b.get("CLUSTER_ACCOUNT"):
            flags.append(f"--account={b['CLUSTER_ACCOUNT']}")
        if time_limit:
            flags.append(f"--time={time_limit}")
        flags += [f"--output={logdir}/%x_%j.out", f"-J {name}"]
        if b.get("CLUSTER_SBATCH_EXTRA"):
            flags.append(b["CLUSTER_SBATCH_EXTRA"])
        if array:
            flags.append(f"--array={array}")
        if extra_flags:
            flags.extend(shlex.split(extra_flags))
        if script.startswith("scripts/sbatch/"):     # the fit path pins by --comment=sha: (the wave shell by its positional)
            flags.append(f"--comment=sha:{sha}")
        return flags

    def hook_present(self, label: str, sha: str, script: str) -> bool:
        r = self.remotes[label]
        if self.site.dry:
            return True
        rc, out, _ = r.sh(f"grep -q {shlex.quote(HOOK_MARKER)} {shlex.quote(self.checkout(label, sha) + '/' + script)} && echo yes || echo no", allow_fail=True)
        return out.strip() == "yes"

    def n_attempts(self, lwid: str, label: str) -> int:
        st = self.ledger.state().get(lwid, {"attempts": {}})
        return sum(1 for a in st["attempts"].values() if a.get("cluster") == label)

    def units_left_to_place(self) -> int:
        """attempts the coordinator submitted that are not yet observed running / granted / closed (the twin cap input)."""
        n = 0
        for lw, st in self.ledger.state().items():
            if st["closed"] or st["winner"]:
                continue
            if any("observed_utc" not in a for a in st["attempts"].values()):
                n += 1
        return n

    # ---- per-unit sha provenance ------------------------------------------------------------------------------
    def inventory_unit(self, lwid: str) -> Optional[dict]:
        return ST.inventory_unit(ST.load_inventory(self.subs.path), lwid)

    def contract_at_this_tree(self, unit: dict) -> dict:
        """stage.contract_at_tree over THIS checkout's registry seeds (the tree the funnel runs from — 06 runs from the clean checkout at
        the wave's tip); a planner that mints no instrument for an eval refuses by name."""
        seeds = os.path.join(os.path.dirname(HERE), "..", "configs", "registry")
        try:
            return ST.contract_at_tree(unit, seeds if os.path.isdir(seeds) else None)
        except ValueError as exc:
            raise Refused(f"{exc} — the unit's contract cannot be verified here; nothing submitted", 3)

    def verify_unit_contract(self, lwid: str, sha: str, spec: Optional[Mapping[str, object]] = None) -> Optional[dict]:
        """Before a unit of the frozen inventory is minted at `sha`: when `sha` is not the unit's recorded sha (a first attempt at another
        tree, or a re-run answering a death), the unit's eval contract at THIS tree must be IDENTICAL to the frozen one — instrument,
        decode-policy ref, item set, evals, shards — else refused by name (tree identity is provenance; contract identity is the
        admission criterion; a contract change is a new wave, never a silent re-freeze). Returns the provenance facts for record_attempt
        (None for a unit outside the inventory: attacker pools, fits, probes)."""
        unit = self.inventory_unit(lwid)
        if unit is None:
            return None
        prov = ST.read_provenance(self.subs.path)
        amend = ST.read_amendments(self.subs.path)
        eff = ST.effective_contract(amend, unit)          # the frozen contract or its latest receipted amendment
        eff_hash = ST.contract_dict_sha256(eff)
        recorded = ST.unit_recorded_sha(prov, unit)
        entries = prov.get(lwid, [])
        # an attempt at a sha other than the unit's recorded one is an ADVANCE whether or not earlier entries exist (attempts that predate
        # the provenance file have no entry, but the unit's frozen sha is its recorded sha); at the recorded sha: first entry or a rerun
        kind = "advance" if sha != recorded else ("first" if not entries else "rerun")
        # (High) the WORKLOAD spec — the script, its args (sha token normalised), every env knob that reaches the job,
        # the array and group — as 06 composed it for THIS submission, against the unit's FIRST attempt's; compared on EVERY submit, same
        # sha included (an env knob such as the multi-turn seed base or a positional window changes the workload under an unchanged
        # registry contract). The one tolerated change is a decode-policy ref env knob re-spelled as the same row's versioned id.
        # the baseline is the FREEZE-TIME COMPOSITION — the unit plan re-derived from the frozen inventory at its code_sha over the
        # site's ordered clusters — and the unit's first attempt AT THE FROZEN TREE must equal it (a consistency check that names both);
        # attempts that predate the freeze are provisional history, never the baseline (the actorattack targets' 11:13Z pass lacked the
        # window positional a fix added before the freeze; every frozen-tree attempt carried it — 2026-09-30 19:54Z). Units outside the
        # inventory (pools, fits) keep "what they first ran as".
        if spec is not None:
            try:
                first, consistency = ST.workload_baseline(self.subs.path, lwid, table=self.site.given("SISTEER_MODELS_TABLE") or None, labels=list(self.site.clusters))
            except ValueError as exc:      # the freeze-time composition cannot be derived (a legacy inventory whose frozen tree the repo does not hold, …): a refusal by name
                raise Refused(f"{lwid}: the freeze-time workload baseline cannot be established — {exc}; nothing submitted (D306, commit 26)", 3)
            if consistency:
                raise Refused(f"{lwid}: " + "; ".join(consistency) + " — the frozen-tree attempt and the composition disagree; the wave's baseline "
                              "cannot be established from either alone; nothing submitted (D306)", 3)
            if first is not None:
                sdiff = ST.spec_diff(first, ST.spec_identity(spec))
                if sdiff:
                    raise Refused(f"{lwid}: this submission's WORKLOAD spec differs from the baseline's: " + "; ".join(sdiff[:6])
                                  + (f"; … {len(sdiff) - 6} more" if len(sdiff) > 6 else "")
                                  + f" — baseline = {ST.describe_baseline(first)}; the baseline unit is the freeze-time composition (script, args, env knobs, "
                                  "array, group); a changed workload is a new wave, not a re-run; nothing submitted (the review on 185949f55; D306)", 3)
        # the REGISTRY contract at this tree (instruments, decode-policy refs, item sets) against the effective one — on every submit too
        diff = ST.contract_diff(eff, self.contract_at_this_tree(unit))
        if diff:
                raise Refused(f"{lwid}: the unit's eval CONTRACT at tree {sha[:12]} differs from the effective one ({recorded[:12]}"
                              f"{', amended ' + str(len(amend.get(lwid, []))) + 'x' if amend.get(lwid) else ''}): "
                              + "; ".join(diff[:6]) + (f"; … {len(diff) - 6} more" if len(diff) > 6 else "")
                              + " — a contract change is a new wave, not a re-run (a seed-row correction goes through `stage.py amend-contract`, D217); nothing submitted (D200)", 3)
        if sha != recorded:
            log(f"PROVENANCE {lwid}: sha {recorded[:12]} -> {sha[:12]} ({kind}); the eval contract and the workload spec are identical at the new tree ({eff_hash[:12]}"
                f"{'; amended ' + str(len(amend.get(lwid, []))) + 'x' if amend.get(lwid) else ''})")
        return {"kind": kind, "from_sha": recorded, "contract_sha256": eff_hash}

    def fix_commit_of(self, sha: str) -> Optional[str]:
        r = subprocess.run(["git", "log", "-1", "--format=%H %s", sha], capture_output=True, text=True, cwd=os.path.dirname(HERE))
        return r.stdout.strip() if r.returncode == 0 and r.stdout.strip() else None

    def dependency_target(self, lwid: str, depends_on: str) -> dict:
        """The binding of the unit `lwid` must run after: refused by name when that unit was never submitted or is mid re-run
        (a dependency on a job that does not exist yet cannot be expressed to Slurm)."""
        b = ST.binding_state(self.subs.path, depends_on)
        if b["state"] == "unbound":
            raise Refused(f"{lwid} runs after {depends_on}, which was never submitted (no binding) — submit that unit first "
                          f"(the BoN climb escalates the 512 store the static -3 job writes; D186)", 3)
        if b["state"] == "transition":
            tok = b.get("transition") or {}
            raise Refused(f"{lwid} runs after {depends_on}, which is mid re-run (transition token since {tok.get('utc')}) — "
                          f"let `submit --resubmit` reconcile it first", 3)
        return b

    def dependency_flags(self, label: str, lwid: str, depends_on: str) -> Tuple[List[str], str]:
        """Slurm `--dependency=afterok:<job>` on the bound job of `depends_on` while that job is live or pending; NO flag once every task
        of it COMPLETED (the unit is complete: the dependency is already satisfied); refused by name when it failed (afterok would leave
        this job DependencyNeverSatisfied — re-run the depended unit instead) or when it is bound on another cluster (the store it writes
        is on that cluster's shared root). `afterok` fails CLOSED: a store that fails after this job is queued leaves it
        DependencyNeverSatisfied, which lib/watch_job.sh reports by name; nobody hand-releases it."""
        if self.site.dry:
            b = ST.binding_state(self.subs.path, depends_on)
            if b["state"] == "unbound":
                return [], f"(dry run) --dependency=afterok:<{depends_on}'s bound job> — that unit is not submitted yet; resolved at submit time"
            return [f"--dependency=afterok:{b.get('job_id')}"], f"afterok:{b.get('job_id')} ({depends_on}; dry run: its state not read)"
        b = self.dependency_target(lwid, depends_on)
        dep_job = str(b.get("job_id"))
        if b.get("cluster") != label:
            raise Refused(f"{lwid} runs after {depends_on}, which is bound on {b.get('cluster')} (job {dep_job}), not on {label} — "
                          f"a dependent unit is placed on the cluster its dependency runs on", 3)
        try:
            tasks = self.remotes[label].sacct_tasks(dep_job)
        except Unreachable as exc:
            raise Refused(f"{lwid}: sacct on {label} unreachable while reading {depends_on}'s job {dep_job} ({exc}) — nothing submitted", 3)
        if not tasks:
            raise Refused(f"{lwid}: {depends_on}'s bound job {dep_job} on {label} is unknown to sacct — its state cannot be read; nothing submitted", 3)
        states = sorted({t["state"] for t in tasks})
        if any(not t["terminal"] for t in tasks):
            return [f"--dependency=afterok:{dep_job}"], f"afterok:{dep_job} ({depends_on} is {','.join(states)}: Slurm starts this job after it completes; a failed store leaves it DependencyNeverSatisfied)"
        if all(t["state"] == "COMPLETED" for t in tasks):
            return [], f"none ({depends_on}'s job {dep_job} COMPLETED: the dependency is satisfied)"
        raise Refused(f"{lwid} runs after {depends_on}, whose bound job {dep_job} on {label} is {','.join(states)} — re-run that unit first "
                      f"(an afterok on a failed job would never be satisfied)", 3)

    def submit_one(self, label: str, *, unit: str, lwid: str, script: str, name: str, args: List[str], env: Dict[str, str],
                   sha: str, time_limit: Optional[str], array: Optional[str], kind: str, group: Optional[str], twin: bool,
                   clusters: Optional[List[str]] = None, force: bool = False, flags: str = "", registers: Optional[str] = None,
                   depends_on: Optional[str] = None, provenance: Optional[dict] = None, bind: bool = True,
                   job_env: Optional[Dict[str, str]] = None, resume_of: Optional[dict] = None, comment: Optional[str] = None) -> Tuple[str, str]:
        registers = registration_contract(kind, registers, script=script, env=env)
        # IDEMPOTENT at the funnel: a unit already `submitted` or `registered` in the gate ledger is SKIPPED — a re-run after a
        # partial wave submits only what is missing and never mints a second `job_name@job_id` identity. `force` is the two
        # legitimate second-attempt paths: the twin of an open election (elect) and reconcile's RESUBMIT of a provably dead,
        # unregistered winner; the CLI's --resubmit reaches it only after the same fenced proof.
        if not force:
            b = ST.binding_state(self.subs.path, lwid)
            if b["state"] == "transition" and not (comment and str((b.get("transition") or {}).get("nonce") or "") == str(comment)):
                # a token carrying OUR nonce is the reservation this very submission wrote a moment ago (every submit reserves first) — not a
                # mid re-run to skip; any other token is
                tok = b.get("transition") or {}
                print(f"skipped {b.get('cluster')} {b.get('job_id')} {unit} (mid re-run since {tok.get('utc')}: transition token for attempt {tok.get('intended_attempt')} — "
                      f"`submit --resubmit` reconciles it against Slurm first)", flush=True)
                return str(b.get("attempt_id") or ""), str(b.get("job_id") or "")
            if b["state"] in ("submitted", "registered"):
                print(f"skipped {b.get('cluster')} {b.get('job_id')} {unit} (already {b['state']}: attempt {b.get('attempt_id')}"
                      f"{'; runs ' + ','.join(b.get('run_ids') or []) if b.get('run_ids') else ''} — pass --resubmit to re-mint after a provably dead, unregistered job)", flush=True)
                return str(b.get("attempt_id") or ""), str(b.get("job_id") or "")
        r = self.remotes[label]
        n = self.n_attempts(lwid, label) + 1
        aid = attempt_id(lwid, label, n)
        fit_env = None
        if kind == "fit":
            # the fit venv's certification on THIS cluster, read before any identity is minted; a second cluster must be the same stack
            fit_env = self.fit_env_identity(label)
            if not self.site.dry:
                self.check_fit_env_pair(lwid, label, fit_env)
        full_env = dict(self.job_env(label)); full_env.update(self.overrides_env(label, sha)); full_env.update(env)
        if job_env:
            full_env.update(job_env)      # JOB-side facts outside the workload spec (a resume's F22_RESUME_FROM_JOB): never part of spec_identity
        if kind == "target" and "/multiturn/" in lwid:
            # a multi-turn TARGET always states its serving jobs at job start — the sibling shard groups' LIVE jobs when known,
            # and in every case its OWN id (the CLI adds it): the marker makes the register block pass the flag
            # even when the sibling list is empty (the first group of a pass), so no group registers with the field missing
            full_env["F22_REGISTRY_MULTITURN_TARGET"] = "1"
            if "F22_REGISTRY_SERVING_JOBS" not in full_env:
                sibs = sibling_serving_jobs(self.subs.path, lwid)      # the sibling shard groups' LIVE jobs, for the job-start registration
                full_env["F22_REGISTRY_SERVING_JOBS"] = ",".join(sibs)
        # the submission's own clock, into the job's env and the attempt record: the native standup's cross-cluster staging-stamp fence
        # (`stamp_not_before`) refuses a geometry stamp staged BEFORE this submission on a cluster that cannot reach the probe record —
        # without the export the fence never applied. A per-submission stamp, not a workload knob: the
        # workload-spec identity drops it (stage.spec_identity), so a re-run never differs from its first attempt by this value.
        submitted_utc = utc_now()
        full_env[SUBMITTED_UTC_ENV] = submitted_utc
        if twin:
            full_env["SISTEER_ATTEMPT_ID"] = aid
            full_env["SISTEER_ATTEMPTS_ROOT"] = f"{self.site.cluster_root(label, self.model)}/attempts"
            full_env["SISTEER_GRANT_TIMEOUT_MIN"] = str(self.site.grant_timeout_min)
        if group:
            full_env["F22_WAVE_ROOT"] = f"{self.site.cluster_root(label, self.model)}/waves/{group}"   # per-group pool root (§3.9)
        prefix = " ".join(f"{k}={shlex.quote(v)}" for k, v in full_env.items())
        sflags = self.sbatch_flags(label, name, time_limit, array, script, sha, extra_flags=flags or "")
        sflags = comment_with_nonce(sflags, comment)      # 31e-fix-5/6: the reservation's nonce, read back by squeue %k; merged into the fit path's sha comment
        if depends_on:
            dep_flags, dep_note = self.dependency_flags(label, lwid, depends_on)
            sflags += dep_flags
            print(f"dependency {label} {unit}: {dep_note}", flush=True)
        job_id = r.sbatch(self.checkout(label, sha), prefix, sflags, script, args)
        print(f"submitted {label} {job_id} {unit}", flush=True)
        if self.site.dry:
            return aid, job_id
        _crash("submit_after_sbatch")     # fault injection: the job exists, nothing records it yet (the transition token's case)
        spec = {"unit": unit, "lwid": lwid, "script": script, "name": name, "args": args, "env": env, "sha": sha,
                "time_limit": time_limit, "array": array, "kind": kind, "group": group, "clusters": list(clusters or [label]),
                "flags": flags or "", "registers": registers, "depends_on": depends_on, "provenance": provenance,
                "submitted_utc": submitted_utc, "resume_of": resume_of, "job_env": dict(job_env) if job_env else None}
        self.record_attempt(label, aid, job_id, spec, fit_env=fit_env, twin=twin, spool=full_env["SISTEER_REGISTRY_SPOOL"], bind=bind, nonce=comment, identified_by="launch")
        return aid, job_id

    def record_attempt(self, label: str, aid: str, job_id: str, spec: dict, *, fit_env: Optional[dict] = None, twin: bool = False,
                       spool: Optional[str] = None, bind: bool = True, nonce: Optional[str] = None, identified_by: str = "launch") -> None:
        """EVERYTHING the funnel records for a launched job, in one place: the ledger's attempt (the FULL spec + candidate clusters a detached
        `elect` / `reconcile` re-issues verbatim, and the producer paths AS SUBMITTED, which the fenced proof reads),
        the submissions row, the pool membership, the `submitted` binding (06 binds `registered` from the job's own pending file — every
        array task's, so the --array spec rides it) and the durable watcher. `reconcile_transition` calls it to ADOPT a job the
        funnel launched but a crash left unrecorded."""
        unit, lwid, kind, group, name = spec["unit"], spec["lwid"], spec.get("kind", "target"), spec.get("group"), spec["name"]
        rec = {"event": "attempt", "logical_work_id": lwid, "attempt_id": aid, "cluster": label, "job_id": job_id, "job_name": name,
               "kind": kind, "registers": spec.get("registers"), "script": spec["script"], "args": spec["args"], "sha": spec["sha"], "twin": twin,
               "unit": unit, "fit_env": fit_env, "submitted_utc": spec.get("submitted_utc"),   # = the job's SISTEER_UNIT_SUBMITTED_UTC
               "spool": spool if spool is not None else self.job_env(label).get("SISTEER_REGISTRY_SPOOL"),
               "pending_dir": f"{self.site.clusters[label]['CLUSTER_SHARED_ROOT']}/registry/pending", "spec": spec,
               # the reservation nonce the job's sbatch line carried and HOW its id was obtained — `launch` (sbatch's stdout at launch)
               # or `nonce` (adopted after a crash by the nonce read back from the job; never by elimination)
               "nonce": nonce, "identified_by": identified_by}
        self.ledger.append(rec)
        _crash("submit_after_ledger")     # fault injection: the attempt is in the ledger, nothing binds or watches it yet
        self._attempt_side_effects(label, aid, job_id, spec, bind=bind, only_missing=False)

    def _attempt_side_effects(self, label: str, aid: str, job_id: str, spec: dict, *, bind: bool, only_missing: bool) -> List[str]:
        """EVERY side effect of a recorded attempt beyond its ledger line (the completion of a half-recorded launch must replay all
        of them, not three): the submissions row; the pool_member event of an attacker-pool attempt (pool_members / mark-placed
        read it); the per-unit sha_provenance SIDECAR + its ledger event for a frozen-inventory unit (admission reads them); the
        `submitted` binding (unless a resume); the watcher. `only_missing` (the completion path) checks each against the durable record first
        and does only what is absent — idempotent; record_attempt runs them all. Returns the steps done."""
        unit, lwid, kind, group, name = spec["unit"], spec["lwid"], spec.get("kind", "target"), spec.get("group"), spec["name"]
        done: List[str] = []
        subs_rows = self.subs.records() if only_missing else []
        if not only_missing or not any(r.get("event") == "submitted" and str(r.get("job_id")) == str(job_id) and r.get("logical_work_id") == lwid for r in subs_rows):
            self.subs.append({"event": "submitted", "cluster": label, "job_id": job_id, "unit": unit, "model": self.model,
                              "logical_work_id": lwid, "attempt_id": aid, "job_name": name, "extra": {"kind": kind, "group": group, "sha": spec["sha"]}})
            done.append("submissions row")
        events = self.ledger.records() if only_missing else []
        if kind == "attacker" and group:
            if not only_missing or not any(e.get("event") == "pool_member" and e.get("logical_work_id") == lwid and e.get("attempt_id") == aid for e in events):
                self.ledger.append({"event": "pool_member", "logical_work_id": lwid, "attempt_id": aid, "cluster": label, "job_id": job_id, "group": group})
                done.append("pool_member")
        pv = spec.get("provenance")
        if pv is not None:      # a unit of the frozen inventory: its per-unit sha table advances with this attempt
            entry = {"sha": spec["sha"], "kind": pv.get("kind"), "attempt_id": aid, "job_id": job_id, "cluster": label, "answers": pv.get("answers"),
                     "fix_commit": pv.get("fix_commit"), "contract_sha256": pv.get("contract_sha256")}
            have_side = only_missing and any(str(e.get("attempt_id")) == str(aid) for e in (ST.read_provenance_doc(self.subs.path).get("units") or {}).get(lwid, []))
            if not have_side:
                ST.record_sha_provenance(self.subs.path, lwid, entry); done.append("sha_provenance sidecar")
            if not only_missing or not any(e.get("event") == "sha_provenance" and e.get("logical_work_id") == lwid and e.get("attempt_id") == aid for e in events):
                self.ledger.append({"event": "sha_provenance", "logical_work_id": lwid, "attempt_id": aid, **{k: v for k, v in entry.items() if k != "attempt_id"},
                                    "from_sha": pv.get("from_sha")})
                done.append("sha_provenance event")
        if bind:
            bpath = os.path.join(ST.bindings_dir(self.subs.path), f"{ST.safe_id(lwid)}.submitted.json")
            have_b = False
            if only_missing and os.path.isfile(bpath):
                try:
                    have_b = str(json.load(open(bpath, encoding="utf-8")).get("job_id")) == str(job_id)
                except (OSError, json.JSONDecodeError):
                    have_b = False
            if not have_b:
                ST.write_binding(self.subs.path, ST.submitted_record(lwid, job_id, label, aid, name, array=spec.get("array"))); done.append("submitted binding")
        elif not only_missing:
            # a RESUME attempt: the unit's REGISTERED binding stands — the resume job reuses its registration identity and its run ids;
            # the attempt is the ledger's, the submissions row's and the watcher's, never the binding's
            log(f"resume {lwid}: attempt {aid} job {job_id} recorded without a binding — the registered binding on job {(spec.get('resume_of') or {}).get('job_id')} stands")
        self.launch_watcher(label, job_id, unit); done.append("watcher")
        return done

    def finish_attempt_transaction(self, label: str, aid: str, job_id: str, att: dict, tok: dict) -> List[str]:
        """The launching process recorded the ledger attempt and died before the rest of record_attempt. Finish EVERYTHING it
        would have done — the submissions row, an attacker attempt's pool_member, a frozen unit's sha_provenance sidecar + event, the
        `submitted` binding (unless a resume), the watcher — idempotently, driven from the token's intended attempt SPEC (the ledger line's
        spec is the same object record_attempt was given); nothing is inferred. Returns the steps done."""
        spec = dict(att.get("spec") or tok.get("spec") or {})
        spec.setdefault("unit", tok.get("unit")); spec.setdefault("lwid", tok["logical_work_id"]); spec.setdefault("name", tok.get("job_name"))
        spec.setdefault("kind", att.get("kind", "target")); spec.setdefault("sha", att.get("sha"))
        return self._attempt_side_effects(label, aid, job_id, spec, bind=not bool(tok.get("resume")), only_missing=True)

    def reservation_snapshot(self, label: str, name: str) -> List[str]:
        """31e-fix-5: every job of the reserved NAME the cluster shows for this user at reservation time — squeue (live) and
        sacct (submitted within the skew window) — recorded in the transition token: none of them can be the job this reservation launches, so
        the reconciliation excludes them positively. Fails closed: accounting unavailable → nothing reserved."""
        try:
            import datetime as _dt
            since = (_dt.datetime.now(_dt.timezone.utc).replace(tzinfo=None) - _dt.timedelta(seconds=self.TRANSITION_SKEW_S)).strftime("%Y-%m-%dT%H:%M:%S")
            ids = {str(r["job_id"]) for r in self.remotes[label].squeue_by_name(name)} | {str(r["job_id"]) for r in self.remotes[label].sacct_by_name(name, since)}
        except AccountingUnavailable as exc:
            raise Refused(f"{name}: accounting unavailable while taking the reservation's snapshot ({exc}) — nothing reserved, nothing launched; retry when it answers", 3)
        return sorted(ids)

    @staticmethod
    def nonce() -> str:
        return os.urandom(6).hex()

    def identify_reserved_job(self, label: str, name: str, since: str, snap: Set[str], nonce: str, known: Set[str], *, what: str, token_path: str
                              ) -> Tuple[Optional[str], Dict[str, dict], Set[str], List[Tuple[str, str]]]:
        """THE reservation primitive (31e-fix-5; shared with the reclose reservation, closer-fix-4): which same-name job did a reservation written at
        `since` launch? Excluded: `known` ids (the ledger's); every id in the reservation's SNAPSHOT (it existed before the reservation — none can be
        ours); a job submitted before the reservation's own clock; a live job carrying ANOTHER reservation's nonce. A live job carrying THIS nonce
        (the sbatch line's --comment, squeue %k) is ours beyond doubt; otherwise exactly one candidate is; more than one fails CLOSED; none with an
        older same-name job the snapshot did not see fails CLOSED too. Accounting unavailable → refused (an empty answer proves nothing).
        Returns (pick | None, candidates, positive ids, excluded [(id, why)]); every refusal names `what` and the token to clear."""
        try:
            import datetime as _dt
            t0 = _dt.datetime.strptime(since, "%Y-%m-%dT%H:%M:%SZ") - _dt.timedelta(seconds=self.TRANSITION_SKEW_S)
            since_q = t0.strftime("%Y-%m-%dT%H:%M:%S")
        except ValueError:
            since_q = since
        since_s = since.replace("Z", "")[:19]
        try:
            rows = self.remotes[label].sacct_by_name(name, since_q)
        except AccountingUnavailable as exc:
            raise Refused(f"RECONCILE {what}: accounting unavailable ({exc}) — an empty answer under it would prove nothing, so the reservation "
                          f"STANDS: nothing adopted, nothing cleared, nothing launched; retry when sacct answers", 3)
        try:
            live = self.remotes[label].squeue_by_name(name)
        except AccountingUnavailable as exc:
            raise Refused(f"RECONCILE {what}: the controller cannot be asked ({exc}) — absence is not proven and a candidate cannot be identified, so the "
                          f"reservation STANDS: nothing adopted, nothing cleared, nothing launched; retry when squeue answers", 3)
        cands: Dict[str, dict] = {}; positive: Set[str] = set(); foreign_older: List[Tuple[str, str]] = []; excluded: List[Tuple[str, str]] = []
        nonce_of = {str(r_["job_id"]): nonce_of_comment(str(r_.get("comment") or "")) for r_ in live}      # squeue's comment identifies the JOB, whichever source lists it
        known_after: List[Tuple[str, str]] = []
        for src_, r_ in [("sacct", r) for r in rows] + [("squeue", r) for r in live]:
            jid = str(r_["job_id"])
            if jid in known:
                sub_k = str(r_.get("submit") or "").strip().replace("Z", "")[:19]
                if jid not in snap and sub_k and sub_k >= since_s:
                    known_after.append((jid, sub_k))      # This unit's own ledger job, NOT in the snapshot and submitted at/after the reservation — a launch, never a non-launch
                continue
            cm = nonce_of.get(jid, "")
            if jid in snap:
                excluded.append((jid, f"in the reservation's snapshot ({src_}): it existed before the reservation")); continue
            if nonce and cm and cm != nonce:
                excluded.append((jid, f"another attempt's nonce nonce:{cm} ({src_})")); continue
            sub = str(r_.get("submit") or "").strip().replace("Z", "")[:19]
            if not sub or sub < since_s:
                foreign_older.append((jid, f"{r_.get('state')} submitted {sub or 'at an unknown time'} ({src_})")); continue
            cands.setdefault(jid, dict(r_, source=src_))
            if nonce and cm == nonce:
                positive.add(jid)
        if len(positive) > 1:
            raise Refused(f"RECONCILE {what}: {len(positive)} live jobs of name {name} on {label} carry this reservation's nonce — {sorted(positive)} — impossible for one sbatch; "
                          f"the reservation STANDS; a human decides", 3)
        # 31e-fix-6: adoption requires the nonce READ BACK from the job itself — a lone candidate that merely qualifies
        # by time and owner may be another session's same-name job submitted after the reservation, and adopting it by elimination would record and
        # watch the wrong job while the intended launch never happens. A job no longer live shows no comment (sacct keeps none): it is refused too;
        # an operator records the right one by hand.
        pick = next(iter(positive)) if positive else None
        if pick is None and known_after:
            raise Refused(f"RECONCILE {what}: this unit's ledger already names job(s) {sorted(set(j for j, _ in known_after))} of name {name} on {label} submitted at/after "
                          f"the reservation — a launch the ledger recorded, never a non-launch to clear; complete that attempt's binding (the reconciliation does so when "
                          f"the token's intended attempt is the ledger's), or record it by hand, then clear {token_path}; the reservation STANDS", 3)
        if pick is None and cands:
            n_ = len(cands)
            raise Refused(f"RECONCILE {what}: {n_} qualifying candidate{'s' if n_ != 1 else ''} of name {name} on {label} — "
                          f"{[(j, c.get('state'), c.get('submit'), c['source'], ('nonce:' + nonce_of[j]) if nonce_of.get(j) else 'no nonce read back') for j, c in sorted(cands.items())]} — "
                          f"none carries this reservation's nonce read back from the job (a job no longer live shows none), and a reservation's job is never adopted by "
                          f"elimination: fail CLOSED (record the right one by hand, then clear {token_path}); the reservation STANDS", 3)
        if pick is None and foreign_older:
            # an older same-name job of ours that the snapshot did NOT see (it appeared in the instant between the snapshot and the token, or a clock
            # ran backwards) is still a refusal by name (31e-fix-4) — nothing launched beside it; snapshot and other-nonce jobs are known not ours
            raise Refused(f"RECONCILE {what}: a live job of the reserved name {name} on {label} is OLDER than this reservation or undated and NOT in its snapshot — "
                          f"{foreign_older} (the reservation's clock is {since}) — not adopted (it is not the job this reservation launched) and nothing launched beside it; "
                          f"the reservation STANDS; a human reads it", 3)
        return pick, cands, positive, excluded

    # ---- the reclose reservation (closer-fix-4): the same primitive, for a job's reclose ----------------------------
    def reclose_token_path(self, lwid: str) -> str:
        return os.path.join(ST.bindings_dir(self.subs.path), f"{ST.safe_id(lwid)}.reclose.json")

    def read_reclose_token(self, lwid: str) -> Optional[dict]:
        p = self.reclose_token_path(lwid)
        if not os.path.isfile(p):
            return None
        try:
            return json.load(open(p, encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            raise Refused(f"{lwid}: the reclose reservation {p} is unreadable — a human clears it; nothing submitted", 3)

    def clear_reclose_token(self, lwid: str) -> None:
        p = self.reclose_token_path(lwid)
        if os.path.isfile(p):
            os.replace(p, p + ".done." + utc_now().replace(":", ""))

    #: the reconciliation's tolerance between the token's clock (this host) and sacct's Submit (the cluster's): a job submitted within
    #: this many seconds BEFORE the token was written is still the token's (the token is written first, so a true successor is later)
    TRANSITION_SKEW_S = 300
    #: non-launch is declared only on authoritative absence (31e-fix-3): sacct rc 0 AND empty, squeue empty by the reserved job name, AND the
    #: token at least this old — accounting shows a job seconds to minutes after sbatch accepted it; a younger token with nothing visible
    #: is "not yet", never "never"
    TRANSITION_MIN_AGE_S = 120

    def reconcile_transition(self, lwid: str, tok: dict) -> str:
        """A transition token found before any action on `lwid` (hole 3): what happened after it was written?
        Slurm is asked by the intended job NAME on the token's cluster for jobs submitted since the token (minus the clock skew) that are
        not already the ledger's: exactly one → ADOPT it as the intended attempt (ledger, submissions, `submitted` binding, watcher — the
        crash fell after sbatch accepted); none → the sbatch never happened: the retired registered binding is RESTORED and the token
        cleared (the proof runs again below); several → refused by name (a human decides which is the attempt). Returns adopted|cleared."""
        label, name, aid = tok["cluster"], tok["job_name"], tok["intended_attempt"]
        st_u = self.ledger.state().get(lwid) or {"attempts": {}}
        known = {str(a.get("job_id")) for a in st_u["attempts"].values()}
        since = tok["utc"]
        nonce0 = str(tok.get("nonce") or "")
        rec_att = st_u["attempts"].get(aid)
        if rec_att and rec_att.get("job_id"):
            # the launching process recorded the token's intended attempt in the ledger and died BEFORE the binding
            # and the watcher. That job is OURS by the ledger's own record (written under this reservation); it must never be excluded as "known"
            # and then cleared as a non-launch (the next submit would launch a second array on the same stores). Verify, then FINISH the
            # transaction: live -> the nonce read back from the job must be this reservation's; not live (sacct only, no %k) -> the launch
            # record's own nonce must be; either mismatch refuses by name with the token standing.
            jid = str(rec_att["job_id"])
            try:
                live = {str(r_["job_id"]): nonce_of_comment(str(r_.get("comment") or "")) for r_ in self.remotes[label].squeue_by_name(name)}
            except AccountingUnavailable as exc:
                raise Refused(f"RECONCILE {lwid}: the ledger names job {jid} as the token's attempt {aid} but the controller cannot be asked ({exc}) — "
                              f"its nonce cannot be read back; the token STANDS: nothing completed, nothing launched; retry when squeue answers", 3)
            if jid in live:
                if not nonce0 or live[jid] != nonce0:
                    raise Refused(f"RECONCILE {lwid}: the ledger's attempt {aid} names job {jid}, live with comment nonce {live[jid]!r}, not this reservation's "
                                  f"{nonce0!r} — not this reservation's launch; the token STANDS; a human decides (then clear {ST.transition_path(self.subs.path, lwid)})", 3)
                method = "nonce"
            else:
                if not nonce0 or str(rec_att.get("nonce") or "") != nonce0:
                    raise Refused(f"RECONCILE {lwid}: the ledger's attempt {aid} names job {jid} (no longer live) recorded under nonce {rec_att.get('nonce')!r}, not this "
                                  f"reservation's {nonce0!r} — not this reservation's launch; the token STANDS; a human decides", 3)
                method = "ledger"
            done = self.finish_attempt_transaction(label, aid, jid, rec_att, tok)
            log(f"RECONCILE {lwid}: the token's attempt {aid} was already in the ledger as job {jid} on {label} — the launcher died before the binding; "
                f"verified by {'the nonce read back from the live job' if method == 'nonce' else 'the launch record under this reservation (job no longer live)'}; "
                f"finished: {', '.join(done) or 'nothing missing'}; nothing launched twice")
            self._reconciled_method = method
            self.receipt_after_adoption(lwid, tok, aid, jid)
            ST.clear_transition(self.subs.path, lwid)
            return "adopted"
        self._reconciled_method = "nonce"
        try:
            import datetime as _dt
            t0 = _dt.datetime.strptime(since, "%Y-%m-%dT%H:%M:%SZ") - _dt.timedelta(seconds=self.TRANSITION_SKEW_S)
            since_q = t0.strftime("%Y-%m-%dT%H:%M:%S")
        except ValueError:
            since_q = since
        # 31e-fix-5: the candidate is identified POSITIVELY. Excluded: the ledger's known ids; every job in the token's
        # SNAPSHOT (same-name jobs that existed at reservation time — none can be ours); a job submitted before the token's own clock (not the
        # skewed query window); a live job carrying ANOTHER attempt's nonce. A live job carrying THIS token's nonce (the sbatch line's --comment,
        # squeue %k) is ours beyond doubt. Otherwise exactly one candidate adopts; more than one fails CLOSED (named, the token standing, an
        # operator decides); none falls through to the non-launch proof below. 31e-fix-3's accounting rules stand (an error anywhere refuses).
        snap = {str(x) for x in (tok.get("snapshot_job_ids") or [])}; nonce = str(tok.get("nonce") or "")
        pick, cands, positive, excluded = self.identify_reserved_job(label, name, since, snap, nonce, known, what=f"{lwid}: the transition token (attempt {aid}, since {since})",
                                                                     token_path=ST.transition_path(self.subs.path, lwid))
        if pick is not None and pick not in positive:
            # identify_reserved_job returns a pick only from `positive` (the nonce read back); this is the adoption's own guard — a job
            # is never adopted into the ledger, bound and watched by elimination, whatever the primitive returns
            raise Refused(f"RECONCILE {lwid}: candidate {pick} of name {name} on {label} did not carry this reservation's nonce read back from the job — not adopted "
                          f"(a reservation's job is never adopted by elimination); the transition token (attempt {aid}) STANDS; record the right one by hand, then clear "
                          f"{ST.transition_path(self.subs.path, lwid)}", 3)
        if pick is not None:
            c = cands[pick]
            how = "positively identified by its nonce"
            lag = "; accounting has no row yet, the controller shows it" if c["source"] == "squeue" else ""
            log(f"RECONCILE {lwid}: the transition token's job exists — {name} {pick} on {label} ({c.get('state')}, submitted {c.get('submit')}, via {c['source']}{lag}) — ADOPTED as {aid}: {how} "
                f"(the crash fell after sbatch accepted; nothing is launched twice)" + (f"; excluded {excluded}" if excluded else ""))
            # a RESUME reservation's job is adopted without a binding (31e-fix-2): the unit's registered binding stands, the run ids are its
            self.record_attempt(label, aid, pick, tok["spec"], fit_env=tok.get("fit_env"), twin=False, spool=tok.get("spool"), bind=not bool(tok.get("resume")),
                                nonce=nonce, identified_by="nonce")
            self.receipt_after_adoption(lwid, tok, aid, pick)
            ST.clear_transition(self.subs.path, lwid)
            return "adopted"
        if excluded:
            log(f"RECONCILE {lwid}: {len(excluded)} same-name job(s) are known not to be this reservation's — {excluded}")
        try:
            import datetime as _dt
            age_s = (_dt.datetime.now(_dt.timezone.utc).replace(tzinfo=None) - _dt.datetime.strptime(since, "%Y-%m-%dT%H:%M:%SZ")).total_seconds()
        except ValueError:
            age_s = -1.0
        if age_s < self.TRANSITION_MIN_AGE_S:
            raise Refused(f"RECONCILE {lwid}: no job of name {name} on {label} in sacct or squeue, but the transition token (attempt {aid}) is "
                          f"{'unparseable' if age_s < 0 else f'{age_s:.0f} s old'} — under the {self.TRANSITION_MIN_AGE_S} s accounting-visibility floor that is "
                          f"'not yet', not 'never'; the token STANDS: nothing cleared, nothing launched; retry after the floor", 3)
        restored = ST.restore_registered_binding(self.subs.path, lwid, tok.get("retired"))
        log(f"RECONCILE {lwid}: no job of name {name} on {label} since the transition token ({since}; sacct rc 0 and empty, squeue empty by name, "
            f"token {age_s:.0f} s old) — the sbatch never happened; {'the retired registered binding is restored; ' if restored else ''}token cleared, the proof decides again")
        ST.clear_transition(self.subs.path, lwid)
        return "cleared"

    def receipt_after_adoption(self, lwid: str, tok: dict, aid: str, job_id: str) -> Optional[dict]:
        """A job ADOPTED from a --resubmit transition token that recorded a cross-cluster replacement (`moved_from`)
        gets the placement receipt (24d) its launcher could not write — the launcher appends it AFTER submit_one returns, so a crash between
        sbatch's acceptance and that write left the move unreceipted and `placement-manifest check` refusing it by name. Composed by the one
        composer the launcher uses (placement_receipt_record); written once per adopted job; None when the token records no move (a
        same-cluster re-run, a first submission, a token written before this field existed)."""
        moved_from = tok.get("moved_from")
        if not moved_from:
            return None
        if any(e.get("event") == "placement_receipt" and e.get("logical_work_id") == lwid and str(e.get("job_id")) == str(job_id) for e in self.ledger.records()):
            return None
        ob = tok.get("old_binding") or {}
        rec = placement_receipt_record(lwid, str(moved_from), str(tok["cluster"]), (tok.get("spec") or {}).get("group"), ob, str(tok.get("replaced_state") or ""),
                                       str(tok.get("why") or ""), aid, job_id)
        self.ledger.append(rec)
        log(f"PLACEMENT RECEIPT {lwid}: {moved_from} (replaced job {ob.get('job_id')} on {ob.get('cluster')}, {tok.get('replaced_state')}) -> {tok['cluster']} "
            f"(attempt {aid}, job {job_id}) — written at the adoption: the launcher died before it; the placement manifest advances under it")
        return rec

    def launch_watcher(self, label: str, job_id: str, unit: str) -> None:
        if self.site.dry or self.remotes[label].local_dir is not None or os.environ.get("SISTEER_NO_WATCHER") == "1":
            return
        r = self.remotes[label]
        events = os.path.join(self.state, "watch", f"{label}_{job_id}.jsonl"); os.makedirs(os.path.dirname(events), exist_ok=True)
        cmd = [os.path.join(HERE, "watch_job.sh"), "--slurm", job_id, "--host", r.host, "--shared-root", r.shared, "--events", events,
               "--remote-events", f"{self.site.cluster_root(label, self.model)}/watch/{job_id}.jsonl", "--unit", unit]
        # a detached watcher never holds the submitter's stdin: its own ssh would drain the plan the stage is still reading
        subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)

    # ---- the election -----------------------------------------------------------------------------------------------
    def attempt_dir(self, label: str, aid: str) -> str:
        return f"{self.site.cluster_root(label, self.model)}/attempts/{aid}"

    def observe(self, label: str, aid: str, job_id: str) -> Tuple[bool, Optional[str]]:
        """(waiting on its grant?, queue state): the WAITING marker under the attempt dir, else ONE squeue read for both answers."""
        r = self.remotes[label]
        rc, out, _ = r.sh(f"ls {shlex.quote(self.attempt_dir(label, aid))}/WAITING.* >/dev/null 2>&1 && echo yes || echo no", allow_fail=True)
        if out.strip() == "yes":
            return True, "RUNNING"
        state = r.squeue_state(job_id)
        return state == "RUNNING", state

    def retire_grant(self, lwid: str, aid: str, a: dict) -> None:
        """Tombstone a dead, PROVABLY unregistered winner's GRANT (`mv GRANT GRANT.retired.<utc>`, atomic) BEFORE its replacement is
        granted, so the remote GRANT set never holds two live sentinels for one unit and the next reconcile does not halt the
        unit whose replacement is running. Safe: the job is terminal (nothing can still read
        the sentinel); idempotent (no GRANT -> nothing to move); the ledger records it AFTER the move so a crash between the two
        replays to the same end state (the proof is re-derived, the move finds nothing)."""
        p = self.attempt_dir(a["cluster"], aid) + "/GRANT"
        rc, out, _ = self.remotes[a["cluster"]].sh(f"if [ -e {shlex.quote(p)} ]; then mv -f {shlex.quote(p)} {shlex.quote(p)}.retired.{utc_now().replace(':', '')} && echo moved; else echo none; fi")
        _crash("after_retire")
        moved = out.strip() == "moved"
        self.ledger.append({"event": "grant_retired", "logical_work_id": lwid, "attempt_id": aid, "sentinel_moved": moved})
        log(f"RETIRED the GRANT of dead, unregistered winner {aid} (job {a['job_id']} on {a['cluster']}){'' if moved else ' — no GRANT sentinel was on disk'}")

    def regrant_after_dead_winner(self, lwid: str, dead: str, dead_att: dict, replacement: str, why: str) -> None:
        """THE ONE retire -> commit -> grant transition after a committed winner provably died unregistered (the four-way fenced
        proof): the dead winner's GRANT is tombstoned BEFORE the replacement is committed and granted, and the replacement
        becomes the ledger's winner so a later election / reconcile grants IT — reconcile's RESUBMIT / REGRANT path and the CLI's
        `submit --resubmit --twin` both go through here. A manual path that created the new attempt without retiring or
        committing would leave `elect` reading the OLD committed winner, calling grant(old) and writing DENY for the very
        attempt it had just launched."""
        self.retire_grant(lwid, dead, dead_att)
        self.ledger.commit_winner(lwid, replacement, reason=f"re-grant after {dead} died unregistered: {why}")
        _crash("after_regrant_commit")
        self.grant(lwid, replacement)
        log(f"REGRANT {lwid}: {replacement}")

    def grant(self, lwid: str, winner: str) -> None:
        st = self.ledger.state()[lwid]
        att = st["attempts"][winner]
        if winner not in st["granted"]:
            self.remotes[att["cluster"]].write_sentinel(self.attempt_dir(att["cluster"], winner) + "/GRANT", json.dumps({"attempt_id": winner, "utc": utc_now(), "by": "submit_twin"}))
            _crash("after_grant")
            self.ledger.append({"event": "granted", "logical_work_id": lwid, "attempt_id": winner})
            log(f"GRANT {winner} (job {att['job_id']} on {att['cluster']})")
        for aid, a in st["attempts"].items():
            if aid == winner or aid in st["denied"]:
                continue
            self.remotes[a["cluster"]].write_sentinel(self.attempt_dir(a["cluster"], aid) + "/DENY", json.dumps({"attempt_id": aid, "winner": winner, "utc": utc_now()}))
            _crash(f"after_deny:{aid}")
            self.remotes[a["cluster"]].scancel(a["job_id"])       # ATTEMPT-scoped: this job id only, never $WAVE_ROOT/STOP
            self.ledger.append({"event": "denied", "logical_work_id": lwid, "attempt_id": aid})
            log(f"DENY {aid} (job {a['job_id']} on {a['cluster']}, scancelled)")

    def elect(self, lwid: str, poll_s: float, max_wait_s: float, twin_spec: Optional[dict] = None) -> str:
        """Run one unit's election to a decision. Returns granted|halted|dead|timeout. `twin_spec` (the primary's submission spec +
        its candidate `clusters`) is reloaded from the ledger when the caller passes none — the DEFAULT detached `elect` subprocess
        and a `reconcile` replay therefore launch the twin exactly as a --foreground election does; an election whose primary
        was not submitted under --twin never launches one."""
        t0 = time.time(); first_pending_seen = None; twin_submitted = False
        if twin_spec is None:
            st0 = self.ledger.state().get(lwid) or {"attempts": {}}
            for aid0, a0 in sorted(st0["attempts"].items(), key=lambda kv: kv[1].get("utc") or ""):
                if a0.get("twin") and a0.get("spec") and len(a0["spec"].get("clusters") or []) >= 2:
                    twin_spec = dict(a0["spec"]); break
        while True:
            st = self.ledger.state().get(lwid)
            if st is None:
                raise Refused(f"no attempts in the ledger for {lwid}")
            if st["halted"]:
                log(f"HALTED {lwid}: {st['halted']}"); return "halted"
            if st["winner"]:
                self.grant(lwid, st["winner"]); return "granted"
            live = {}
            for aid, a in st["attempts"].items():
                try:
                    waiting, state = self.observe(a["cluster"], aid, a["job_id"])
                    if waiting:
                        self.ledger.append({"event": "observed", "logical_work_id": lwid, "attempt_id": aid})
                        _crash("after_observed")
                        self.ledger.commit_winner(lwid, aid, cluster=a["cluster"], job_id=a["job_id"])     # the transaction point
                        _crash("after_ledger_commit")
                        self.grant(lwid, aid)
                        return "granted"
                    live[aid] = state
                except Unreachable as exc:
                    log(f"unreachable while polling {aid}: {exc}; retrying")
                    live[aid] = "UNKNOWN"
            if all(s is None for s in live.values()):
                # nobody queued or running: every attempt ended before a grant (a fast-fail) — say so, cancel nothing
                self.ledger.append({"event": "halted", "logical_work_id": lwid, "reason": "every attempt left the queue before a grant was written (fast-fail? see the watch events)"})
                return "dead"
            # the twin trigger (FD 20): primary PENDING past the threshold, few units left to place, no twin yet
            if twin_spec and not twin_submitted and len(st["attempts"]) == 1:
                (aid0, a0), = st["attempts"].items()
                if live.get(aid0) == "PENDING":
                    first_pending_seen = first_pending_seen or time.time()
                    if time.time() - first_pending_seen >= self.site.threshold_s and self.units_left_to_place() < self.site.max_jobs:
                        other = self.site.other(a0["cluster"])
                        if other and other in twin_spec["clusters"]:
                            log(f"primary {aid0} PENDING > {self.site.threshold_s}s — submitting the twin on {other}")
                            self.submit_one(other, twin=True, force=True, **{k: twin_spec.get(k) for k in SPEC_KEYS})
                            twin_submitted = True
            if time.time() - t0 > max_wait_s:
                log(f"election for {lwid} still open after {int(max_wait_s)}s (states {live}); leaving it to reconcile")
                return "timeout"
            time.sleep(poll_s)

    # ---- reconcile ---------------------------------------------------------------------------------------------------
    GENERIC_SUBSTRATES = frozenset({"", "slurm", "none", "unknown"})

    @staticmethod
    def _payload_cluster(p: dict) -> Optional[str]:
        """The cluster a run payload names, if any — the most CONCRETE name first: the real emitter requires the generic
        `slurm` substrate to carry the site's label as `extra.cluster_label` (emit.check_substrate, migration 0018), which the reducer persists
        under `_extra` in row_history and in the live row's extra_json; so `cluster_label` anywhere wins over `cluster`, which wins over the
        top-level `substrate` — read top-level, under `fields`, and in the `_extra` / `extra` metadata."""
        dicts = [d for d in (p, p.get("fields") or {}, p.get("_extra") or {}, p.get("extra") or {}, (p.get("fields") or {}).get("extra") or {})
                 if isinstance(d, dict)]
        for k in ("cluster_label", "cluster", "substrate"):
            for d in dicts:
                v = d.get(k)
                if isinstance(v, str) and v.strip():
                    return v
        return None

    @staticmethod
    def _payload_attempt(p: dict) -> Optional[str]:
        """The attempt id a run payload names: the reducer persists record extra under `_extra` (Effect.history_payload); older spellings are read too."""
        for d in (p.get("_extra") or {}, p.get("extra") or {}, (p.get("fields") or {}).get("extra") or {}):
            if isinstance(d, dict) and d.get("attempt_id"):
                return str(d["attempt_id"])
        return None

    @classmethod
    def _cluster_compatible(cls, named: Optional[str], cluster: Optional[str]) -> bool:
        """A hit is this cluster's unless it names a DIFFERENT, non-generic cluster (Slurm job numbers are allocated per cluster but
        the registry is shared — an old run that once had job J on the other cluster must not refuse a re-mint of job J here).
        An unknown or generic substrate is kept: absence is not evidence of another cluster (fail closed)."""
        if not cluster or named is None or named.strip().lower() in cls.GENERIC_SUBSTRATES:
            return True
        return named == cluster

    @classmethod
    def _history_runs(cls, conn, job_id: str, aid: Optional[str] = None, cluster: Optional[str] = None) -> List[str]:
        """The run ids whose registration HISTORY names the job (payload cluster_job_id, top-level or under `fields`) or the attempt
        (`_extra.attempt_id`, the reducer's spelling): the live row's cluster_job_id is the LAST writer's, so a run first registered under this
        job and since re-emitted under another spelling is only in row_history (reproduce/lib/stage.py's 32c pattern). row_history has NO run_id
        column — its key is natural_key_json. Hits naming another cluster are excluded (_cluster_compatible). Empty when
        the table is absent."""
        if not conn.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'row_history'").fetchone():
            return []
        q = ("SELECT json_extract(natural_key_json, '$.run_id'), payload_json FROM row_history WHERE table_name = 'run' AND record_type IN ('run', 'run_patch') "
             "AND (json_extract(payload_json, '$.cluster_job_id') = ? OR json_extract(payload_json, '$.fields.cluster_job_id') = ?")
        params: tuple = (job_id, job_id)
        if aid:
            q += " OR json_extract(payload_json, '$._extra.attempt_id') = ? OR json_extract(payload_json, '$.extra.attempt_id') = ? OR json_extract(payload_json, '$.fields.extra.attempt_id') = ?"
            params += (aid, aid, aid)
        q += ")"
        out: List[str] = []
        for rid, pj in conn.execute(q, params).fetchall():
            if not rid:
                continue
            try:
                p = json.loads(pj) if pj else {}
            except ValueError:
                p = {}
            by_attempt = bool(aid) and cls._payload_attempt(p) == aid
            if (by_attempt or cls._cluster_compatible(cls._payload_cluster(p), cluster)) and rid not in out:
                out.append(rid)
        return out

    @classmethod
    def _live_compatible(cls, extra_json: Optional[str], cluster: Optional[str]) -> bool:
        try:
            ex = json.loads(extra_json) if extra_json else {}
        except ValueError:
            ex = {}
        named = None
        if isinstance(ex, dict):      # the concrete site label (generic `slurm` substrates; emit.CLUSTER_LABEL_KEY) before the `cluster` spelling
            for k in ("cluster_label", "cluster"):
                v = ex.get(k)
                if isinstance(v, str) and v.strip():
                    named = v
                    break
        return cls._cluster_compatible(named, cluster)

    def registry_runs_for(self, job_id: str, aid: str, cluster: Optional[str] = None) -> Optional[List[str]]:
        db = os.path.join(self.site.registry_root, "registry.sqlite")
        if not os.path.isfile(db):
            return None
        conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        try:
            rows = conn.execute("SELECT run_id, extra_json, json_extract(extra_json, '$.attempt_id') FROM run WHERE cluster_job_id = ? OR json_extract(extra_json, '$.attempt_id') = ?",
                                (job_id, aid)).fetchall()
            found = [r[0] for r in rows if (aid and r[2] == aid) or Coordinator._live_compatible(r[1], cluster)]
            found += [r for r in Coordinator._history_runs(conn, job_id, aid, cluster) if r not in found]
            return found
        finally:
            conn.close()

    def registry_available(self) -> bool:
        return os.path.isfile(os.path.join(self.site.registry_root, "registry.sqlite"))

    def registry_runs_by_job(self, raw_job_id: str, cluster: Optional[str] = None, aid: Optional[str] = None) -> List[str]:
        """The registry's run ids filed under ONE scheduler job id (a task's RAW id for an array child) — the live row AND the registration
        history (the per-task evidence the fenced proof reads must see a run whose live row another writer has since re-keyed),
        restricted to this cluster's (job numbers are per cluster, the registry is shared)."""
        db = os.path.join(self.site.registry_root, "registry.sqlite")
        conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        try:
            rows = conn.execute("SELECT run_id, extra_json FROM run WHERE cluster_job_id = ?", (str(raw_job_id),)).fetchall()
            found = [r[0] for r in rows if Coordinator._live_compatible(r[1], cluster)]
            found += [r for r in Coordinator._history_runs(conn, str(raw_job_id), aid, cluster) if r not in found]
            return found
        finally:
            conn.close()

    def registry_run_states(self, run_ids: Sequence[str]) -> Dict[str, str]:
        """run_id -> lifecycle_state for the ids the registry carries (a missing id is simply absent)."""
        ids = [r for r in dict.fromkeys(run_ids) if r]
        if not ids:
            return {}
        db = os.path.join(self.site.registry_root, "registry.sqlite")
        conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        try:
            q = ",".join("?" * len(ids))
            return {r[0]: r[1] for r in conn.execute(f"SELECT run_id, lifecycle_state FROM run WHERE run_id IN ({q})", ids).fetchall()}
        finally:
            conn.close()

    def registry_run_stores(self, run_ids: Sequence[str]) -> Dict[str, Optional[str]]:
        """run_id -> store_uri for the ids the registry carries (read-only; a missing id is absent, a run without one maps to None)."""
        ids = [r for r in dict.fromkeys(run_ids) if r]
        if not ids:
            return {}
        db = os.path.join(self.site.registry_root, "registry.sqlite")
        conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        try:
            q = ",".join("?" * len(ids))
            return {r[0]: r[1] for r in conn.execute(f"SELECT run_id, store_uri FROM run WHERE run_id IN ({q})", ids).fetchall()}
        finally:
            conn.close()

    def records_folded(self, files: Sequence[Mapping[str, object]]) -> List[dict]:
        """The dead task's record files the registry has NOT folded to a durable cursor (hole 1): a file is folded
        when `record_ingest` carries it (its root-relative uri `records/<rel>`) with the SAME sha256 and n_records as the file on the
        cluster and no `record_cursor` row (a cursor = a fold interrupted mid-file). Anything else — absent (in transit), a different
        sha256 or count (the producer wrote after the fold), a cursor — is returned with its reason, and the re-run is refused: a
        same-identity re-run would upsert into the new attempt's identities while the old records are still landing."""
        db = os.path.join(self.site.registry_root, "registry.sqlite")
        conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        out = []
        try:
            for f in files:
                rel = str(f["rel"]); uris = (f"records/{rel}", rel)
                row = conn.execute("SELECT file_uri, file_sha256, n_records FROM record_ingest WHERE file_uri IN (?, ?)", uris).fetchone()
                if row is None:
                    out.append({"rel": rel, "why": "not in record_ingest (not pulled / not folded yet — in transit)"}); continue
                cur = conn.execute("SELECT records_done FROM record_cursor WHERE file_uri IN (?, ?)", uris).fetchone()
                if cur is not None:
                    out.append({"rel": rel, "why": f"partially folded (record_cursor at {cur[0]} of {f['n_records']} records)"}); continue
                if str(row[1]) != str(f["sha256"]):
                    out.append({"rel": rel, "why": f"sha256 differs (registry {str(row[1])[:12]}, cluster {str(f['sha256'])[:12]}: the producer wrote after the fold)"}); continue
                if int(row[2] or 0) != int(f["n_records"]):
                    out.append({"rel": rel, "why": f"n_records differ (registry {row[2]}, cluster {f['n_records']})"}); continue
        finally:
            conn.close()
        return out

    def pending_run_ids(self, label: str, pend_path: str) -> Optional[List[str]]:
        """The run ids a job's own pending file names (every run it registered), None when the file is absent."""
        r = self.remotes[label]
        rc, out, _ = r.sh(f"[ -f {shlex.quote(pend_path)} ] && cat {shlex.quote(pend_path)} || echo __ABSENT__", allow_fail=True)
        if rc != 0 or out.strip() == "__ABSENT__":
            return None
        try:
            d = json.loads(out)
        except json.JSONDecodeError:
            return []
        runs = d.get("runs") or ([{"run_id": d["run_id"]}] if d.get("run_id") else [])
        return [str(x.get("run_id")) for x in runs if x.get("run_id")]

    def task_evidence(self, att: dict, task: dict) -> dict:
        """What ONE task (a raw scheduler job id) left behind, read at its cluster: its pending file (+ the run ids it names), the
        producer spool's mention of it, the registry's rows filed under its raw id, and the lifecycle of every run it names. Never a
        verdict — the decision reads the whole unit's tasks together."""
        label = att["cluster"]; r = self.remotes[label]; raw = str(task["raw"])
        pend = f"{att.get('pending_dir') or r.shared + '/registry/pending'}/{raw}.json"
        pend_runs = self.pending_run_ids(label, pend)
        spool = r.spool_mentions([f'"{raw}"', f"@{raw}"], att.get("spool")) if att.get("spool") else None
        reg_runs = self.registry_runs_by_job(raw, cluster=label, aid=att.get("attempt_id")) if self.registry_available() else []
        run_ids = list(dict.fromkeys((pend_runs or []) + reg_runs))
        states = self.registry_run_states(run_ids) if (run_ids and self.registry_available()) else {}
        registered = pend_runs is not None or spool is True or bool(reg_runs)
        return {"raw": raw, "task": task.get("task"), "state": task["state"], "end": task.get("end"), "pending_file": pend, "pending_present": pend_runs is not None,
                "spool": spool, "registry_runs": reg_runs, "run_ids": run_ids, "run_states": states, "registered": registered,
                "past_running": sorted(rid for rid, st in states.items() if st in PAST_RUNNING_STATES)}

    def closer_recipe(self, label: str, job_id: str) -> Optional[dict]:
        """The reclose RECIPE an eval job left when its closer refused (common.sh f22_closer_recipe_write): <shared>/registry/pending/<job>.closer_refused.json
        — the closer's arguments captured from the job's own env, plus every reclose's outcome. None when the job left none."""
        r = self.remotes[label]; pend_dir = f"{r.shared}/registry/pending"; path = f"{pend_dir}/{job_id}.closer_refused.json"
        # closer-fix-7: ONE remote read of the recipe, of every per-job RECEIPT file the reclose jobs wrote
        # (`<job>.reclose.<reclose job>.receipt.json` — each written once by its own job, never replaced) and of the controller's append-only retry
        # notes (`<job>.reclose_retries.jsonl`). The recipe is immutable after the eval job writes it, so no file ever has two writers: before, the
        # controller's retry-note writer and a failing job's receipt writer both read/replaced the recipe, and the loser's write erased the other's
        # record (a lost `preflight` marker made the cap count a failure that counts toward nothing).
        script = ("import glob, json, os, sys\nd = sys.argv[1]; j = sys.argv[2]\nout = {'recipe': None, 'receipts': [], 'retries': [], 'bad': []}\n"
                  "p = os.path.join(d, j + '.closer_refused.json')\n"
                  "if os.path.isfile(p):\n    out['recipe'] = open(p, encoding='utf-8').read()\n"
                  "for f in sorted(glob.glob(os.path.join(d, j + '.reclose.*.receipt.json'))):\n"
                  "    try:\n        out['receipts'].append(json.load(open(f, encoding='utf-8')))\n    except ValueError:\n        out['bad'].append(f)\n"
                  "q = os.path.join(d, j + '.reclose_retries.jsonl')\n"
                  "if os.path.isfile(q):\n    for ln in open(q, encoding='utf-8'):\n        ln = ln.strip()\n        if ln:\n"
                  "            try:\n                out['retries'].append(json.loads(ln))\n            except ValueError:\n                out['bad'].append(q)\n"
                  "print(json.dumps(out))\n")
        rc, out, err = r.sh("python3 -S -c " + shlex.quote(script) + " " + shlex.quote(pend_dir) + " " + shlex.quote(str(job_id)), allow_fail=True)
        if rc != 0:
            raise Refused(f"{label}: cannot read the reclose recipe and receipts of job {job_id} under {pend_dir} (rc {rc}: {(err or out or '')[:200]}); nothing submitted", 3)
        try:
            bundle = json.loads(out or "{}")
        except ValueError:
            raise Refused(f"{label}: the reclose record read of job {job_id} returned no JSON ({(out or '')[:120]!r}); nothing submitted", 3)
        if not (bundle.get("recipe") or "").strip():
            return None
        try:
            d = json.loads(bundle["recipe"])
        except ValueError:
            raise Refused(f"{label}: the reclose recipe {path} is not JSON — re-run the closer by hand under the driver venv; nothing submitted", 3)
        if bundle.get("bad"):
            raise Refused(f"{label}: unreadable reclose receipt / retry record(s) of job {job_id}: {bundle['bad']} — a human reads them; nothing submitted", 3)
        by_job = {str(x.get("job_id")): x for x in (d.get("recloses") or []) if isinstance(x, dict)}            # legacy: receipts appended onto the recipe before closer-fix-7
        by_job.update({str(x.get("job_id")): x for x in (bundle.get("receipts") or []) if isinstance(x, dict)})   # the per-job file is the job's own word: it wins
        d["recloses"] = sorted(by_job.values(), key=lambda x: (str(x.get("utc") or ""), str(x.get("job_id"))))
        d["retries"] = list(d.get("retries") or []) + list(bundle.get("retries") or [])
        d["path"] = path
        return d

    def reclose_input(self, label: str, recipe: dict, job_id: str) -> dict:
        """closer-fix-6: the reclose's INPUT as the funnel sees it now — the recipe's content identity, the pending file's sha256 (or `absent`), the
        checkout's presence at the recipe's sha — the same three the reclose job's preflight receipt records (reclose_stores.sbatch)."""
        r = self.remotes[label]; pend = str(recipe.get("pending") or ""); co = self.checkout(label, str(recipe.get("code_sha") or ""))
        rc, out, err = r.sh(f"{{ [ -f {shlex.quote(pend)} ] && sha256sum {shlex.quote(pend)} | cut -c1-64 || echo absent; }}; "
                            f"{{ [ -d {shlex.quote(co + '/src/sisteer')} ] && echo present || echo absent; }}", allow_fail=True)
        lines = (out or "").split()
        if rc != 0 or len(lines) != 2 or lines[1] not in ("present", "absent"):
            raise Refused(f"{label}: cannot read the reclose input of job {job_id} (pending {pend}, checkout {co}) — rc {rc}: {(err or out or '')[:200]}; nothing submitted", 3)
        return {"recipe_sha256": recipe_identity(recipe), "pending": lines[0], "checkout": lines[1]}

    def recipe_append_retry(self, label: str, job_id: str, note: dict) -> None:
        """closer-fix-6/7: the operator's deliberate retry, recorded BESIDE the receipts in the controller's own append-only file
        (`<job>.reclose_retries.jsonl` — one JSON line per retry; never a receipt, never a deletion, never a rewrite of the recipe, which stays
        immutable after the eval job writes it). The ledger event carries the same note, so a failed write here loses no record."""
        if self.site.dry:
            return
        path = f"{self.remotes[label].shared}/registry/pending/{job_id}.reclose_retries.jsonl"
        rc, out, err = self.remotes[label].sh("printf '%s\\n' " + shlex.quote(json.dumps(note, sort_keys=True)) + " >> " + shlex.quote(path), allow_fail=True)
        if rc != 0:
            log(f"RECLOSE: could not record the retry in {path} (rc {rc}: {(err or out or '')[:200]}) — the ledger event carries it")

    #: closer-fix-3: the markers the tree the closer runs from must carry — closer-fix-2's `gate_binding` in the closer
    #: (26a27aac3, the first tree with it: a gate record decides only when bound) and reclose-at-tree's positional 3 in the reclose script (the
    #: funnel submits that script from the same tree)
    CLOSER_FIX2_SHA = "26a27aac30fc50398604aa450970f91d8373e85d"
    CLOSER_FIX2_MARKER = "def gate_binding("
    RECLOSE_TREE_MARKER = 'CLOSER_SHA="${3:-}"'
    # closer-fix-4: the closer-fix-3 commit — the first tree carrying all three repairs (the shard-local response tally,
    # PENDING_DIR bound on --emit-expected-goal-ids, validated gate metadata). A reclose runs only from a checkout whose ACTUAL commit descends from
    # it: the markers above also sit in 26a27aac3, the closer with all three defects, so the markers alone are no guard
    CLOSER_MIN_SHA = "b36e48a2ffb2e9878b648d1ebc64f49b4ea831b3"
    #: the publish step's root marker (content `squashed`, committed in the publication's root commit; sisteer.registry.snapshot.HISTORY_SQUASHED_MARKER
    #: is the same name): a repository published without its development history SAYS so. Read here only to NAME the shape of a clone that does not
    #: carry CLOSER_MIN_SHA in its refusal — the reclose is refused on every clone without the development history, published or not
    PUBLISHED_MARKER = ".history-squashed"

    def closer_tree_check(self, label: str, lwid: str, run_sha: str, *, explicit: bool) -> str:
        """closer-fix-3 + closer-fix-4: the checkout the closer RUNS from (`--tree`'s,
        else the job's own), refused by name unless (a) it carries the closer + src/sisteer, (b) it is a git checkout whose ACTUAL commit — `git
        rev-parse HEAD` at the tree path, never the directory's name — descends from CLOSER_MIN_SHA (`git merge-base --is-ancestor`), (c) it carries
        closer-fix-2's gate binding and (d) a reclose script with the closer-tree positional — the funnel submits reclose_stores.sbatch FROM this
        tree, so a script that drops positional 3 would make --tree inert (nine recloses by hand). Before (b) the marker greps alone let
        the real 26a27aac3 tree — the closer with all three defects closer-fix-3 repairs — pass, and so did a stub carrying the markers.
        A clone that does not carry CLOSER_MIN_SHA at all (`git cat-file -e` fails) cannot answer (b) and is refused by name, whatever its files
        say: the reclose is an INTERNAL recovery path — reclose_stores.sbatch re-runs the closer against the original job's private pending
        recipe, spool, driver environment and registry push — so it needs the development history and the private registry pipeline, and a
        repository published without its development history (the publish step's root marker, borne out: not shallow, PUBLISHED_MARKER reading
        `squashed`, the one root commit carrying it) or a shallow / partial clone has neither; the refusal names which shape it found. Content
        decides nothing here: the modules the closer imports out of the same tree cannot be pinned without going stale on every change to them,
        and an unpinned package dropped beside them shadows a pinned import. Where the clone carries the commit, (b) is read for real — presence
        alone never stands in for the lineage. ONE remote read. Returns the checkout path."""
        co = self.checkout(label, run_sha)
        closer = f"{co}/scripts/sdfbase/full22/registry_close_stores.py"; script = f"{co}/{RECLOSE_SCRIPT}"; pkg = f"{co}/src/sisteer"
        mk = self.PUBLISHED_MARKER
        cmd = (f"t={shlex.quote(co)}; c={shlex.quote(closer)}; s={shlex.quote(script)}; d={shlex.quote(pkg)}; m={shlex.quote(self.CLOSER_MIN_SHA)}; "
               f'{{ [ -f "$c" ] && [ -d "$d" ] && echo closer=ok || echo closer=absent; }}; '
               f'{{ h=$(git -C "$t" rev-parse --verify HEAD 2>/dev/null) && echo "head=$h" || echo head=absent; }}; '
               f'{{ git -C "$t" merge-base --is-ancestor "$m" HEAD 2>/dev/null && echo ancestry=ok || echo ancestry=no; }}; '
               f'{{ git -C "$t" cat-file -e "$m^{{commit}}" 2>/dev/null && echo min=present || echo min=absent; }}; '
               # the published shape, for the refusal's wording: not shallow, the root marker reading `squashed`, HEAD's one root commit carrying it.
               # `trim` strips SURROUNDING whitespace only (Python's str.strip(), the rule every other marker reader applies): `squa shed` is not the marker
               'trim(){ v=$(cat); v="${v#"${v%%[![:space:]]*}"}"; printf %s "${v%"${v##*[![:space:]]}"}"; }; '
               f'{{ pub=no; if [ "$(git -C "$t" rev-parse --is-shallow-repository 2>/dev/null)" = false ] && [ -f "$t/{mk}" ] '
               f'&& [ "$(trim < "$t/{mk}")" = squashed ]; then r=$(git -C "$t" rev-list --max-parents=0 HEAD 2>/dev/null | tr "\\n" " "); set -- $r; '
               f'if [ $# -eq 1 ] && [ "$(git -C "$t" cat-file -p "$1:{mk}" 2>/dev/null | trim)" = squashed ]; then pub=yes; fi; fi; echo published=$pub; }}; '
               f'{{ [ -f "$c" ] && grep -qF -- {shlex.quote(self.CLOSER_FIX2_MARKER)} "$c" && echo fix2=ok || echo fix2=absent; }}; '
               f'{{ [ -f "$s" ] && grep -qF -- {shlex.quote(self.RECLOSE_TREE_MARKER)} "$s" && echo script=ok || echo script=absent; }}')
        rc_t, out_t, err_t = self.remotes[label].sh(cmd, allow_fail=True)
        got = dict(tok.split("=", 1) for tok in (out_t or "").split() if "=" in tok)
        who = f"--tree {run_sha[:12]}" if explicit else f"the job's tree {run_sha[:12]}"
        hint = ("land and mirror a tree that carries it" if explicit
                else f"pass --tree <a landed tree at or past {self.CLOSER_MIN_SHA[:9]}> (reclose-at-tree: the job's sha stays the runs' code sha)")
        if rc_t != 0 or got.get("closer") != "ok":
            raise Refused(f"{lwid}: {who}: no checkout with the closer at {co} on {label} (rc {rc_t}: {(err_t or out_t or '')[:160]}) — land and mirror it first; "
                          f"nothing submitted", 3)
        head = got.get("head") or "absent"
        if head == "absent":
            raise Refused(f"{lwid}: {who}: {co} on {label} is not a git checkout (no HEAD to read) — the funnel reads the ACTUAL commit the closer runs "
                          f"from, never the directory's name (D1529); {hint}; nothing submitted", 3)
        if got.get("ancestry") != "ok":
            if got.get("min") == "absent":
                # the clone does not carry closer-fix-3 at all: no lineage to read, and no reclose — the recovery path is internal
                shape = ("a repository published without its development history (the publish step's root marker, borne out)" if got.get("published") == "yes"
                         else "a shallow or partial clone, not a publication")
                raise Refused(f"{lwid}: {who}: the closer at {co} is commit {head[:12]}; this clone does not carry closer-fix-3 {self.CLOSER_MIN_SHA[:12]} ({shape}) — "
                              f"the reclose is an internal recovery path; it needs the development history and the private registry pipeline (the original job's "
                              f"pending recipe, spool, driver environment and registry push), and this clone has neither; nothing submitted", 3)
            # the clone carries closer-fix-3, so the lineage was READ and HEAD does not descend from it
            raise Refused(f"{lwid}: {who}: the closer at {co} is commit {head[:12]}, which does not descend from closer-fix-3 {self.CLOSER_MIN_SHA[:12]} — "
                          f"a reclose runs only a closer carrying the shard-local response tally, the PENDING_DIR binding on --emit-expected-goal-ids and "
                          f"the validated gate metadata (D1529; {self.CLOSER_FIX2_SHA[:9]} carries the markers and none of the three); {hint}; nothing submitted", 3)
        if got.get("fix2") != "ok":
            raise Refused(f"{lwid}: {who} lacks closer-fix-2 (no `gate_binding` in {closer}; {self.CLOSER_MIN_SHA[:9]} is the minimum tree) — a reclose runs "
                          f"only a closer that binds the gate record before it decides; {hint}; nothing submitted", 3)
        if got.get("script") != "ok":
            raise Refused(f"{lwid}: the reclose script at {script} predates reclose-at-tree (no CLOSER_SHA positional 3) — the funnel submits {RECLOSE_SCRIPT} from the "
                          f"tree the closer runs from, so a script that drops the positional makes --tree inert (D1437); {hint}; nothing submitted", 3)
        return co

    def reclose(self, lwid: str, unit: str, b: dict, recipe: Optional[dict], *, retry_why: str = "", closer_sha: str = "", closer_args: Sequence[str] = ()) -> str:
        """closer-fix-2: job `b` ended with exit CLOSER_REFUSED_EXIT — its stores are complete, its closer refused. Submit reclose_stores.sbatch
        (CPU-only, at the recipe's sha; it confirms the driver venv by f22_driver_venv and re-runs the closer from the recipe) ONCE: refused
        without a recipe; `closed` when the last reclose filed (rc 0); refused past RECLOSE_MAX recloses or when the same refusal survived a
        reclose (the diagnosis is printed, a human decides); skipped while a reclose of the job is live (squeue/sacct by its name, fail closed).
        Recorded in the ledger (`reclose`) and watched. Returns submitted | skip | closed."""
        label = b["cluster"]; job_id = str(b.get("job_id"))
        if recipe is None:
            raise Refused(f"{lwid}: job {job_id} on {label} ended with exit {CLOSER_REFUSED_EXIT} (its store closer refused) but left no recipe at "
                          f"{self.remotes[label].shared}/registry/pending/{job_id}.closer_refused.json — re-run the closer by hand under the driver venv; nothing submitted", 3)
        why0 = str(recipe.get("why") or "")[:300]
        name = f"f22-reclose-{job_id}"
        # closer-fix-5: a recipe is usable only when it carries an absolute wave root (the STOP fence below reads
        # it; the closer takes --wave-root from it) and an absolute pending path — the writer used to read both from os.environ, where f22_init's
        # plain shell variables never were, so every real recipe carried wave_root='' and pending='/registry/pending/<job>.json': the fence was
        # skipped and the reclose job died on the bogus path before its trap. Fail CLOSED by name; a human fixes the recipe or closes by hand.
        wave_root = str(recipe.get("wave_root") or ""); pend_path = str(recipe.get("pending") or "")
        pend_expected = f"{self.remotes[label].shared}/registry/pending/{job_id}.json"       # the job's own pending file under ITS cluster's shared root — nothing else
        bad = ([f"wave_root ({wave_root or 'empty'}) is not an absolute path"] if not wave_root.startswith("/") else [])             + ([f"pending ({pend_path or 'empty'}) is not this job's pending file {pend_expected}"] if pend_path != pend_expected else [])
        if bad:
            raise Refused(f"{lwid}: the reclose recipe of job {job_id} on {label} is unusable — {'; '.join(bad)} "
                          f"(a closer whose shell variables never reached the recipe writer, before closer-fix-5); the wave's STOP cannot be checked and the reclose job "
                          f"would die before its trap; nothing submitted: fix the recipe by hand ({recipe.get('path')}) or close by hand under the driver venv", 3)
        # closer-fix-4: a RESERVATION written before sbatch — reconciled here, by the same primitive the resume path uses,
        # before any further reclose: a launcher that died after sbatch accepted and before the ledger append left no ledger line and (if that reclose
        # then died in its own pre-trap window) no receipt — invisible to a live-only check, so the cap permitted one more real reclose than it says
        tok = self.read_reclose_token(lwid)
        if tok is not None:
            known_r = {str(e.get("reclose_job_id")) for e in self.ledger.records() if e.get("event") == "reclose" and str(e.get("job_id")) == job_id}
            pick, cands, positive, excluded = self.identify_reserved_job(label, name, str(tok.get("utc")), {str(x) for x in (tok.get("snapshot_job_ids") or [])},
                                                                         str(tok.get("nonce") or ""), known_r, what=f"{lwid}: the reclose reservation of job {job_id} (written {tok.get('utc')})",
                                                                         token_path=self.reclose_token_path(lwid))
            if pick is not None:
                c = cands[pick]
                self.ledger.append({"event": "reclose", "logical_work_id": lwid, "unit": unit, "job_id": job_id, "cluster": label, "reclose_job_id": str(pick), "job_name": name,
                                    "attempt": int(tok.get("attempt") or 0), "why": str(tok.get("why") or why0), "code_sha": tok.get("code_sha"), "utc": utc_now(), "adopted": True})
                log(f"RECLOSE {lwid}: the reclose reservation's job exists — {name} {pick} on {label} ({c.get('state')}, submitted {c.get('submit')}, via {c['source']}) — ADOPTED "
                    f"(the launcher died after sbatch accepted, before the ledger line; {'positively identified by its nonce' if pick in positive else 'the one qualifying candidate'})")
                self.clear_reclose_token(lwid)
            else:
                try:
                    import datetime as _dt
                    age_s = (_dt.datetime.now(_dt.timezone.utc).replace(tzinfo=None) - _dt.datetime.strptime(str(tok.get("utc")), "%Y-%m-%dT%H:%M:%SZ")).total_seconds()
                except ValueError:
                    age_s = -1.0
                if age_s < self.TRANSITION_MIN_AGE_S:
                    raise Refused(f"{lwid}: a reclose reservation of job {job_id} stands ({tok.get('utc')}, {'unparseable' if age_s < 0 else f'{age_s:.0f} s old'}) and no job of name {name} "
                                  f"answers for it yet — under the {self.TRANSITION_MIN_AGE_S} s accounting-visibility floor that is 'not yet', not 'never'; nothing submitted; retry after the floor", 3)
                log(f"RECLOSE {lwid}: the reclose reservation of job {job_id} ({tok.get('utc')}, {age_s:.0f} s old) launched nothing (sacct and squeue by name, aged past the floor) — cleared"
                    + (f"; known not ours: {excluded}" if excluded else ""))
                self.clear_reclose_token(lwid)
        # closer-fix-4: a reclose under the wave's STOP would leave `stopped` (closing nothing) — refuse to submit while it stands
        if self.remotes[label].exists(f"{wave_root}/STOP"):      # closer-fix-5: unconditional — the recipe's wave_root was checked absolute above
            raise Refused(f"{lwid}: the wave's STOP stands at {wave_root}/STOP — a reclose of job {job_id} would leave as `stopped` and close nothing; clear the STOP first; nothing submitted", 3)
        recloses_all = list(recipe.get("recloses") or [])
        stopped_ids = {str(r_.get("job_id")) for r_ in recloses_all if r_.get("stopped")}          # left under STOP: the cap does not count them, nor do the closed / same-refusal rules
        preflight_ids = {str(r_.get("job_id")) for r_ in recloses_all if r_.get("preflight")}      # refused before the closer ran: not a closer outcome, not an attempt (closer-fix-6)
        recloses = [r_ for r_ in recloses_all if not r_.get("stopped") and not r_.get("preflight")]   # the closer's OUTCOMES: the closed / same-refusal / cap rules read these
        prior = [e for e in self.ledger.records() if e.get("event") == "reclose" and str(e.get("job_id")) == job_id and str(e.get("cluster")) == str(label)]
        after_preflight: Optional[dict] = None
        # closer-fix-5 / closer-fix-6: a reclose job that refused in PREFLIGHT (the recipe or the environment, before the
        # closer ran — recorded by its trap with the INPUT it saw) is not a closer outcome and counts toward nothing. It stops the automation only while
        # that input is UNCHANGED: a repaired input (the pending file restored, the checkout restored, the recipe fixed — the receipt's recorded identity
        # differs from what the funnel sees now) is new evidence for one more reclose, logged and recorded on the ledger event; otherwise the operator
        # records a deliberate retry (`reclose --job J --retry-why …`, kept beside the receipt in the recipe's `retries`). Never by deleting a receipt.
        # A retry already submitted after the receipt (a ledger line newer than it) is judged by the live / outcome rules below, not here.
        if recloses_all and recloses_all[-1].get("preflight"):
            pf = recloses_all[-1]; pf_job = str(pf.get("job_id")); pf_why = str(pf.get("why") or "")[:200]
            # closer-fix-7: a LATER retry is a DIFFERENT reclose job submitted after this one — Slurm ids are monotone per
            # cluster — never a ledger line newer than the receipt: an adopted reservation job is appended at reconciliation time, possibly after
            # its own receipt, and would otherwise pass for a retry of itself, skipping the unchanged-input fence
            def _after(a: str, b: str) -> bool:
                try:
                    return int(a) > int(b)
                except ValueError:
                    return a > b
            retried_after = [rj for rj in (str(e.get("reclose_job_id")) for e in prior) if rj != pf_job and _after(rj, pf_job)]
            # closer-fix-7: the receipt's own job still ACTIVE (its preflight receipt lands the instant it refuses; the controller may walk in
            # that window) is in flight — judged by the live rules when it ends, not fenced as "unchanged" now
            try:
                pf_tasks = self.remotes[label].sacct_tasks(pf_job)
            except Unreachable as exc:
                raise Refused(f"{lwid}: reclose job {pf_job} of job {job_id} cannot be read on sacct ({exc}) — not submitting another reclose blind; nothing submitted", 3)
            if pf_tasks and any(not t.get("terminal") for t in pf_tasks):
                print(f"skipped {label} {job_id} {unit} (reclose {name} {pf_job} is {','.join(sorted({str(t.get('state')) for t in pf_tasks}))} — in flight; its receipt is read when it ends)", flush=True)
                return "skip"
            import getpass as _gp, socket as _so
            who = f"{_gp.getuser()}@{_so.gethostname()}:{os.getpid()}"          # closer-fix-7: the caller's identity on every retry record
            if not retried_after:
                then_in = dict(pf.get("input") or {}); now_in = self.reclose_input(label, recipe, job_id)
                changed = sorted(k for k in set(now_in) | set(then_in) if now_in.get(k) != then_in.get(k)) if then_in else []
                if changed:
                    log(f"RECLOSE {lwid}: reclose job {pf_job} of job {job_id} refused in preflight ({pf_why}); the input it saw has since changed — "
                        f"{', '.join(f'{k}: {then_in.get(k)!s:.16} -> {now_in.get(k)!s:.16}' for k in changed)} — the repaired input is new evidence; one more reclose")
                    after_preflight = {"job_id": pf_job, "changed": changed, "input_then": then_in, "input_now": now_in, "who": who}
                elif retry_why:
                    log(f"RECLOSE {lwid}: reclose job {pf_job} of job {job_id} refused in preflight ({pf_why}); the input is unchanged; the operator records a deliberate retry: {retry_why}")
                    after_preflight = {"job_id": pf_job, "retry_why": retry_why, "input": now_in, "who": who}
                else:
                    raise Refused(f"{lwid}: reclose job {pf_job} of job {job_id} refused in PREFLIGHT ({pf_why}) and the input it saw is "
                                  f"{'unchanged (recipe, pending file and checkout identical)' if then_in else 'unknown (the receipt names no input)'} — the recipe or the environment is wrong, "
                                  f"not the closer, and another reclose would fail the same way; nothing submitted. Recovery: repair the input (the pending file "
                                  f"{recipe.get('pending')}, the checkout at {self.checkout(label, str(recipe.get('code_sha') or ''))}, or the recipe {recipe.get('path')}) and the next walk "
                                  f"proceeds on its own, or record a deliberate retry: `submit_twin.py reclose --job {job_id} --retry-why '<what you repaired>'` — never by deleting the "
                                  f"receipt (the history stands; preflight refusals count toward nothing)", 3)
        if recloses and int(recloses[-1].get("rc") or 0) == 0:
            log(f"RECLOSE {lwid}: job {job_id}'s stores were closed by reclose job {recloses[-1].get('job_id')} ({recloses[-1].get('utc')}) — its runs advanced; the gate reads them; nothing submitted")
            return "closed"
        if recloses and str(recloses[-1].get("why") or "")[:300] == why0:
            raise Refused(f"{lwid}: job {job_id}'s closer refusal SURVIVED reclose job {recloses[-1].get('job_id')} unchanged ({why0 or 'no REFUSED line'}) — "
                          f"not resubmitting the same recovery blind; {len(recloses)} reclose(s) so far; a human reads the refusal (recipe {recipe.get('path')})", 3)
        # closer-fix-3: the cap counts the DURABLE record — every `reclose` submission the ledger holds for this job, each read
        # on sacct — not the recipe's receipts: a reclose that died before its trap (f22_init / f22_driver_venv refusing, a node failure, walltime)
        # leaves a terminal sacct row and the ledger line but no receipt, and counting receipts alone launched another identical job every walk,
        # past the cap, with the failure hidden behind a fresh launch. A live prior reclose is in flight; a terminal one counts, receipted or not;
        # a receipt-less one is NAMED with its sacct state; at the cap the refusal names them all.
        receipted = {str(r_.get("job_id")) for r_ in recloses}
        terminal: List[str] = []; receiptless: List[Tuple[str, List[str]]] = []
        for e in prior:
            rj = str(e.get("reclose_job_id"))
            if rj in stopped_ids or rj in preflight_ids:
                continue                                                                        # left under STOP, or refused in preflight: not an attempt (closer-fix-6)
            try:
                tasks = self.remotes[label].sacct_tasks(rj)
            except Unreachable as exc:
                raise Refused(f"{lwid}: reclose job {rj} of job {job_id} cannot be read on sacct ({exc}) — not submitting another reclose blind; nothing submitted", 3)
            states = sorted({str(t.get("state")) for t in tasks})
            if not tasks or any(not t.get("terminal") for t in tasks):
                print(f"skipped {label} {job_id} {unit} (reclose {name} {rj} is {','.join(states) or 'not in sacct yet'} — in flight; the gate reads the runs when it files)", flush=True)
                return "skip"
            terminal.append(rj)
            if rj not in receipted:
                receiptless.append((rj, states))
        # the cap counts every reclose job known from EITHER durable record — the ledger's submissions (the funnel's) or the recipe's receipts (the job's):
        # a receipt proves a submission even where the ledger line is missing, a ledger line proves one even where the receipt is
        n_total = len(set(terminal) | receipted)
        if n_total >= RECLOSE_MAX:
            raise Refused(f"{lwid}: job {job_id} already had {n_total} reclose(s) (cap {RECLOSE_MAX}) — {len(terminal)} durable submission(s) in the ledger, {len(receipted)} receipt(s) on the recipe; "
                          f"receipt-less — died before their trap wrote one — {receiptless}; last receipt: {('rc ' + str(recloses[-1].get('rc')) + ' ' + str(recloses[-1].get('why') or '')[:160]) if recloses else 'none'} — "
                          f"a human reads them (recipe {recipe.get('path')}; the ledger's reclose events name the jobs)", 3)
        if receiptless:
            log(f"RECLOSE {lwid}: {len(receiptless)} earlier reclose(s) of job {job_id} left no receipt ({receiptless}) — they died before their trap; counted toward the cap of {RECLOSE_MAX}")
        try:
            live = self.remotes[label].squeue_by_name(name)
            rows = self.remotes[label].sacct_by_name(name, str(recipe.get("utc") or "2000-01-01T00:00:00Z").replace("Z", ""))
        except AccountingUnavailable as exc:
            raise Refused(f"{lwid}: accounting unavailable ({exc}) — cannot tell whether a reclose of job {job_id} is already live; nothing submitted", 3)
        active = live + [r_ for r_ in rows if r_["state"] in ACTIVE]
        if active:
            print(f"skipped {label} {job_id} {unit} (reclose {name} {active[0]['job_id']} is {active[0]['state']} — in flight; the gate reads the runs when it files)", flush=True)
            return "skip"
        recloses = recloses if len(recloses) >= n_total else recloses + [{}] * (n_total - len(recloses))     # the attempt number counts every known reclose
        sha = str(recipe.get("code_sha") or "")
        if not re.fullmatch(r"[0-9a-f]{40}", sha):
            raise Refused(f"{lwid}: the reclose recipe names no 40-hex code sha ({sha!r}) — the closer runs on the tree the job ran; nothing submitted", 3)
        closer_sha = str(closer_sha or "")
        closer_args = [str(x) for x in (closer_args or []) if str(x)]
        for x in closer_args:
            if any(ch in x for ch in "\n\r;|&`$"):
                raise Refused(f"{lwid}: --closer-arg {x!r} carries shell metacharacters; the closer's flags are plain tokens; nothing submitted", 3)
        if closer_sha:
            # reclose-at-tree: the closer of a LANDED fix, the job's tree untouched — the operator's explicit decision, recorded everywhere
            if not re.fullmatch(r"[0-9a-f]{40}", closer_sha):
                raise Refused(f"{lwid}: --tree names no 40-hex sha ({closer_sha!r}); nothing submitted", 3)
            if closer_sha == sha:
                raise Refused(f"{lwid}: --tree {closer_sha[:12]} is the job's own tree — a plain reclose already runs there; nothing submitted", 3)
        # closer-fix-3: the tree the closer RUNS from — --tree's, else the job's own — must carry the closer
        # WITH closer-fix-2 and a reclose_stores.sbatch that takes the closer's tree on positional 3, because the sbatch is submitted FROM that
        # tree: the funnel used to submit the JOB tree's script, which predated reclose-at-tree and dropped positional 3, so --tree was inert
        run_sha = closer_sha or sha
        co_run = self.closer_tree_check(label, lwid, run_sha, explicit=bool(closer_sha))
        full_env = dict(self.job_env(label)); full_env.update(self.overrides_env(label, sha)); full_env[SUBMITTED_UTC_ENV] = utc_now()
        prefix = " ".join(f"{k}={shlex.quote(v)}" for k, v in full_env.items())
        n = len(recloses) + 1
        # the RESERVATION (closer-fix-4): snapshot + nonce + the clock, written before sbatch — the next call reconciles it by positive identity
        nonce_c = self.nonce(); snap_c = self.reservation_snapshot(label, name)
        ST.write_atomic(self.reclose_token_path(lwid), {"logical_work_id": lwid, "unit": unit, "job_id": job_id, "cluster": label, "job_name": name, "utc": utc_now(),
                                                        "nonce": nonce_c, "snapshot_job_ids": snap_c, "attempt": n, "why": why0, "code_sha": sha, "script_tree": run_sha,
                                                        **({"closer_sha": closer_sha} if closer_sha else {}), **({"closer_args": closer_args} if closer_args else {})})
        _crash("reclose_after_reservation")
        flags = comment_with_nonce(self.sbatch_flags(label, name, "00:45:00", None, RECLOSE_SCRIPT, sha, extra_flags="--cpus-per-task=4 --mem=16G"), nonce_c)
        positionals = [job_id, sha]
        if closer_sha or closer_args:
            positionals.append(closer_sha)                                  # positional 3: the closer's tree ("" = the job's own)
        if closer_args:
            positionals.append(shlex.join(closer_args))                     # positional 4: the closer's extra flags, one shell-quoted string
        new = self.remotes[label].sbatch(co_run, prefix, flags, RECLOSE_SCRIPT, positionals)     # closer-fix-3 (4): the script from the tree the closer runs from
        _crash("reclose_after_sbatch")
        self.ledger.append({"event": "reclose", "logical_work_id": lwid, "unit": unit, "job_id": job_id, "cluster": label, "reclose_job_id": str(new), "job_name": name,
                            "attempt": n, "why": why0, "code_sha": sha, "script_tree": run_sha, "utc": utc_now(), **({"after_preflight": after_preflight} if after_preflight else {}),
                            **({"closer_sha": closer_sha} if closer_sha else {}), **({"closer_args": closer_args} if closer_args else {})})
        if after_preflight and after_preflight.get("retry_why"):
            self.recipe_append_retry(label, job_id, {"after_job": after_preflight["job_id"], "why": after_preflight["retry_why"], "reclose_job_id": str(new),
                                                     "utc": utc_now(), "input": after_preflight["input"], "who": after_preflight["who"]})
        self.clear_reclose_token(lwid)
        print(f"RECLOSE {label} {new} {unit}: job {job_id} ended with exit {CLOSER_REFUSED_EXIT} — its store closer refused ({why0 or 'no REFUSED line recorded'}); "
              f"{RECLOSE_SCRIPT} (the script from tree {run_sha[:12]}) re-runs the closer under the driver venv at {sha[:12]}"
              + (f" with the closer of tree {closer_sha[:12]} (reclose-at-tree: the job's sha stays {sha[:12]})" if closer_sha else "")
              + (f" with closer args {shlex.join(closer_args)}" if closer_args else "")
              + f" (reclose {n}/{RECLOSE_MAX}; no GPU, no re-run of the eval)", flush=True)
        self.launch_watcher(label, str(new), unit)
        return "submitted"

    def resubmit_decision(self, lwid: str, unit: str, b: dict, kind: str, registers: Optional[str], *, walk: bool = False) -> Tuple[str, str, dict]:
        """`--resubmit` on a bound unit -> (action, why, evidence) with action `skip` | `resubmit` | `rerun`, or Refused (rc 3). READ-ONLY: the
        decision writes nothing — `prove-resubmit` runs it as the documented read-only proof (03's design archive acts on it) — so the
        `torn_down` traces it DERIVES for a pool member (a dead one, a self-exited one, an untraced predecessor) ride the evidence as
        `teardowns`, a list of ledger events, and the authorized writer (`submit --resubmit`, Coordinator.record_teardowns) appends them;
        a Refused decision derives none (otherwise the proof would move pool_members by merely running). The FENCED
        PROOF over EVERY task of the bound job:
        * any task still active -> `skip` (06 --resubmit walks the family; a live unit is never the wave's failure).
        * a contract-`none` pool member that SELF-EXITED (attackers/<job>.SELF_EXIT) while another member of its group is live -> `skip`
          (a finished pool slot is never re-placed beside the member the targets are bound to); with no live member left it is
          fenced like a torn-down one and re-minted.
        * every task of a registering unit COMPLETED and `walk` (06's family pass) -> `skip` (the results stand; the pass walks on); otherwise a
          COMPLETED task -> refused: its runs are the unit's results (a partial re-run is a human's decision: the gate reads one binding) —
          except a contract-`none` pool member the ledger records as TORN DOWN (teardown-pool wrote its <jobid>.STOP and the pool exits 0
          on it): that COMPLETED is the teardown, not a result, and the job is fenced like a dead one.
        * contract `none` (kind other / declared): every task terminal-not-COMPLETED = the whole proof (`resubmit`); a DEAD pool member
          (kind attacker, no SELF_EXIT, never torn down) owes a `torn_down {dead: true, state}` ledger trace on that path, once, so
          pool_members / live_members drop it before the group's placement is written; an attacker unit's EARLIER pool jobs (re-minted
          before a trace existed) that are dead or self-exited owe the same trace, once each.
        * contract `registry`: the registry database must be reachable (fail CLOSED), the producer spool must be traversable, and EVERY
          record file the dead tasks left in it must be FOLDED by the harvester to a durable cursor (record_ingest: same sha256 and record
          count, no record_cursor) — else records are still landing and a same-identity re-run would upsert into the new identities (the
          per-cluster ingest watermark is informational: it is the last arrival of any run, not this job's fold — hole 1); per task the
          pending file, the spool and the registry decide `registered`; a run the registry carries past `running` refuses by name —
          except a Petri proxy whose past-running runs sit on legs the resumed drive reads as done (RESUME_PROOFS / petri_resume_proof:
          `petri_run.py legs-status` over the stores the registry names; a leg it would regenerate keeps the refusal).
        * a fit whose task registered -> refused (its successor attaches by name); every task never registered -> `resubmit`.
        * an eval unit whose tasks are dead and each registered or never registered -> `rerun` under the SAME identities (deterministic
          run_id; the registry upserts) when any task registered, else `resubmit`.
        A `registered` binding (06 --gate bound it) is decided the same way; the caller retires it before the new attempt is bound."""
        teardowns: List[dict] = []
        action, why, evidence = self._resubmit_decision(lwid, unit, b, kind, registers, teardowns, walk=walk)
        return action, why, {**evidence, "teardowns": teardowns}

    def record_teardowns(self, evidence: Mapping[str, object]) -> List[dict]:
        """The authorized write of a decision's `teardowns` (resubmit_decision): `submit --resubmit` appends each torn_down trace the read-only
        decision derived — once the decision has decided and, on a re-mint, once the preflight it runs under has passed, so a refusal leaves the
        ledger exactly as it was — and names each on stdout. `prove-resubmit` never calls it. Returns the events appended."""
        out = []
        for ev in evidence.get("teardowns") or []:
            self.ledger.append(dict(ev))
            how = "self_exit" if ev.get("self_exit") else "dead"
            which = "earlier pool member" if ev.get("superseded") else "job"
            state = f" ({ev.get('state')})" if ev.get("state") else ""
            log(f"{ev.get('logical_work_id')}: recorded torn_down ({how}) for {which} {ev.get('job_id')} on {ev.get('cluster')}{state} — pool_members / live_members "
                f"drop it; the group's placement never assigns a target to it")
            out.append(dict(ev))
        return out

    def _resubmit_decision(self, lwid: str, unit: str, b: dict, kind: str, registers: Optional[str], teardowns: List[dict], *, walk: bool = False) -> Tuple[str, str, dict]:
        """resubmit_decision's body: READ-ONLY; the torn_down traces it derives go into `teardowns` (the caller's list), never into the ledger."""
        label = b["cluster"]; job_id = str(b.get("job_id"))
        st_u = self.ledger.state().get(lwid) or {"attempts": {}, "winner": None}
        att = st_u["attempts"].get(b.get("attempt_id")) or {"cluster": label, "job_id": job_id, "job_name": b.get("sbatch_name")}
        att.setdefault("cluster", label); att.setdefault("job_id", job_id)
        spec = att.get("spec") or {}
        contract = registration_contract(att.get("kind", kind), att.get("registers") or spec.get("registers") or registers,
                                         script=att.get("script") or spec.get("script"), env=spec.get("env"))
        r = self.remotes[label]
        # a pool member the ledger records as TORN DOWN (teardown-pool wrote ITS <jobid>.STOP; attacker_pool.sbatch exits 0 on it) ended by
        # design: under contract `none` its COMPLETED is the teardown, not a result, so the job is fenced like a dead one (its evidence is
        # still read below). Only a torn_down record naming THIS job on THIS cluster counts; a registering unit's COMPLETED is always a result.
        torn_jobs = {str(j.get("job_id")) for j in (st_u.get("torn_down_jobs") or []) if j.get("cluster") in (None, label)}
        if att.get("kind") == "attacker":
            # The already-superseded case: the unit's EARLIER pool jobs, re-minted before this tree recorded a trace, stay in pool_members
            # forever and the next placement names them beside the member this pass mints (the demo smoke's 981273, cancelled and re-minted
            # into 983283 on the old tree: PLACED v2 would have carried both). The decision DERIVES their teardown here, once each, from
            # sacct and the shared root — terminal-not-COMPLETED -> `dead`; COMPLETED with attackers/<job>.SELF_EXIT -> `self_exit`; a
            # COMPLETED predecessor with no sentinel is left alone (ambiguous), as is a live one — into `teardowns`; the submit command
            # records them (record_teardowns), the read-only proof only names them.
            for m in st_u.get("pool_jobs") or []:
                mj = str(m.get("job_id"))
                if m.get("cluster") not in (None, label) or mj == str(job_id) or mj in torn_jobs:
                    continue
                try:
                    mt = r.sacct_tasks(mj)
                except Unreachable:
                    break
                if not mt or not all(t.get("terminal") for t in mt):
                    continue
                m_states = ",".join(sorted({t["state"] for t in mt}))
                if all(t["state"] == "COMPLETED" for t in mt):
                    if not self.member_self_exited(label, spec.get("group"), mj):
                        continue
                    teardowns.append({"event": "torn_down", "logical_work_id": lwid, "job_id": mj, "cluster": label, "self_exit": True, "superseded": True})
                    log(f"{lwid}: earlier pool member {mj} on {label} SELF-EXITED (attackers/{mj}.SELF_EXIT) and has no ledger trace — a torn_down (self_exit) trace is due")
                elif not any(t["state"] == "COMPLETED" for t in mt):
                    teardowns.append({"event": "torn_down", "logical_work_id": lwid, "job_id": mj, "cluster": label, "dead": True, "superseded": True, "state": m_states})
                    log(f"{lwid}: earlier pool member {mj} on {label} is DEAD on sacct ({m_states}) and has no ledger trace — a torn_down (dead) trace is due so the group's placement never assigns a target to it")
                else:
                    continue
                torn_jobs.add(mj)
        try:
            tasks = r.sacct_tasks(job_id)
        except Unreachable as exc:
            raise Refused(f"--resubmit refused: {lwid}: sacct on {label} unreachable ({exc}) — nothing decided", 3)
        if not tasks:
            raise Refused(f"--resubmit refused: {lwid}'s bound job {job_id} on {label} is unknown to sacct — its terminal state cannot be read", 3)
        active = [t for t in tasks if not t["terminal"]]
        if active:
            names = ", ".join(f"task {t['task']} ({t['raw']}) {t['state']}" if t["task"] is not None else t["state"] for t in active)
            return "skip", f"still {names}", {"tasks": tasks}
        completed = [t for t in tasks if t["state"] == "COMPLETED"]
        dead = [t for t in tasks if t["state"] != "COMPLETED"]
        teardown_note = ""
        self_exited = bool(completed and contract == REGISTERS_NONE and att.get("kind") == "attacker" and self.member_self_exited(label, spec.get("group"), job_id))
        if completed and contract == REGISTERS_NONE and (str(job_id) in torn_jobs or self_exited):
            if not self_exited:
                teardown_note = f"job {job_id} was TORN DOWN (ledger torn_down: its {job_id}.STOP was written) — its COMPLETED is the teardown, not a result; "
            else:
                # 31b: the member closed ITSELF (its last consumer ended, or the group was placed and nothing bound it) and wrote
                # attackers/<job>.SELF_EXIT — a teardown by design, read from the shared root, owed to the ledger on FIRST sight (once; the
                # submit command records it from `teardowns`, the read-only proof does not)
                if str(job_id) not in torn_jobs:
                    teardowns.append({"event": "torn_down", "logical_work_id": lwid, "job_id": str(job_id), "cluster": label, "self_exit": True})
                teardown_note = f"job {job_id} SELF-EXITED (attackers/{job_id}.SELF_EXIT: its last consumer ended, or the group was placed and nothing bound it) — its COMPLETED is the teardown, not a result; "
            # a torn-down / self-exited member is a FINISHED pool slot while another member of its group is live — the walk
            # re-placed attackers/2 of the s1 wave (880002) beside the member every remaining target was bound to, and the new pool stood up to
            # find no consumer. Skip by name while another member serves (every pass, not only the first: the ledger's torn_down record is what the
            # next pass reads); re-mint only when the pool is gone (no live member left).
            if att.get("kind") == "attacker":
                live = [m for m in self.live_members(str(spec.get("group"))) if str(m["job_id"]) != str(job_id)]
                if live:
                    return ("skip", f"{teardown_note}the group's remaining targets are served by live member(s) {', '.join(m['job_id'] for m in live)} — "
                                    f"a finished pool slot is never re-placed (D1713)", {"tasks": tasks, "live_members": live})
            dead, completed = list(tasks), []
        def arr(ts):
            return ",".join(str(t["task"]) for t in ts if t["task"] is not None)
        # closer-fix-2: every dead task FAILED with the closer's exit code — the eval's work finished, its store closer
        # refused: the answer is `reclose` (re-run the closer under the driver venv from the job's recipe), never a GPU re-run of the eval
        if dead and not completed and contract != REGISTERS_NONE and all(t["state"] == "FAILED" and str(t.get("exit_code") or "").startswith(f"{CLOSER_REFUSED_EXIT}:") for t in dead):
            recipe = self.closer_recipe(label, job_id)
            why_c = str((recipe or {}).get("why") or "no REFUSED line recorded")[:300]
            return "reclose", f"exit {CLOSER_REFUSED_EXIT}: the store closer refused on job {job_id} ({why_c}) — the stores are complete; re-run the closer, not the eval", {"tasks": tasks, "recipe": recipe}
        if completed and not dead and walk and contract != REGISTERS_NONE:
            # EVERY task of a REGISTERING unit COMPLETED and the caller is a family pass (06 --resubmit, `--walk`): the unit's results stand
            # (the gate reads them) and the pass walks on — it is not the wave's failure (06 --resubmit --family multiturn stopped at a
            # COMPLETED crescendo target and never reached the actorattack rows, 2026-09-30 21:07Z). A direct re-run request of a done
            # unit, and a contract-`none` pool that COMPLETED without a teardown, stay refused below.
            return "skip", f"COMPLETED: results stand (job {job_id}, task(s) {arr(completed) or job_id}; the gate reads them — never re-minted)", {"tasks": tasks}
        if completed:
            hint = f"; the dead tasks would be --array={arr(dead)} — a partial re-run is a human's decision (the gate reads one binding per unit)" if dead and any(t["task"] is not None for t in dead) else ""
            raise Refused(f"--resubmit refused: {lwid}'s bound job {job_id} on {label} has COMPLETED task(s) {arr(completed) or job_id}: their runs are the unit's results — "
                          f"a same-identity re-run needs every task DEAD (terminal, not COMPLETED){hint}", 3)
        ends = [t["end"] for t in dead if t.get("end") and t["end"] not in ("Unknown", "None")]
        if contract == REGISTERS_NONE:
            # the contract says nothing was registered — but the evidence that CAN be read is still read, and any of it refuses: a
            # misdeclared contract must never authorise a re-mint over results (hole 2). Absent means to check
            # (no spool recorded, no database) is not evidence either way under `none`.
            ev = [self.task_evidence(att, t) for t in dead]
            contradicted = [e for e in ev if e["pending_present"] or e["spool"] is True or e["registry_runs"]]
            if contradicted:
                what = "; ".join(f"task {e['task'] if e['task'] is not None else e['raw']}: "
                                 + ", ".join(x for x in (("pending file" if e["pending_present"] else ""), ("spool record" if e["spool"] is True else ""),
                                                        (f"registry run(s) {e['registry_runs']}" if e["registry_runs"] else "")) if x) for e in contradicted)
                raise Refused(f"--resubmit refused: {lwid} declares registers=none but its dead job left registration evidence ({what}) — a misdeclared contract; "
                              f"nothing re-minted (declare `registry`, or fix the script's contract)", 3)
            states = ",".join(sorted({t["state"] for t in dead}))
            # A DEAD pool member — kind attacker, CANCELLED / FAILED / TIMEOUT on sacct, no SELF_EXIT, never torn down — used to be re-minted with
            # NO ledger trace, so pool_members kept the dead job forever and the next mark-placed carried it beside its replacement: the demo
            # smoke's PLACED v1 assigned every target to the cancelled 981273, the re-minted member found nothing assigned to it and self-exited
            # (exit_unbound), and a target failed on its refused endpoint.
            # Derive the teardown Slurm performed — ONCE, guarded like the self-exit branch above (a torn_down naming this job already
            # stands, or the self-exit branch just derived this very decision's: nothing owed; a self-exited or torn-down member is a
            # teardown by design, never `dead`) — into `teardowns`: the submit command records it (record_teardowns) so pool_members /
            # live_members drop the job before any placement is written; the read-only proof names it and writes nothing.
            if (att.get("kind") or kind) == "attacker" and not self_exited and str(job_id) not in torn_jobs:
                teardowns.append({"event": "torn_down", "logical_work_id": lwid, "job_id": str(job_id), "cluster": label, "dead": True, "state": states})
                teardown_note = (f"{teardown_note}job {job_id} is DEAD on sacct ({states}; no SELF_EXIT, never torn down) — a torn_down (dead) trace is due so the "
                                 f"group's placement never assigns a target to it; ")
            why = (f"{teardown_note}registers={contract} (kind {att.get('kind', kind)}): the unit has no registration contract, sacct says "
                   f"{states} (terminal) for every task and no task left a pending file, spool record or registry row — check 1 is the whole proof")
            return "resubmit", why, {"tasks": tasks, "evidence": ev, "contract": contract}
        # the registry contract: fail CLOSED on anything the proof cannot read
        if not self.registry_available():
            raise Refused(f"--resubmit refused: {lwid}: no registry database under {self.site.registry_root} — the proof cannot read what the dead job filed (fail closed)", 3)
        latest = max(ends) if ends else ""
        # the per-cluster ingest WATERMARK (scripts/registry/mirror_status.py) is read for the record only: it is the last arrival of ANY
        # run on the substrate, not a completed fold of THIS job's records (hole 1) — the proof below is per file
        ing = self.ingested_past(label, latest.replace(" ", "T")) if latest else None
        if not att.get("spool"):
            raise Refused(f"--resubmit refused: {lwid}: the attempt's ledger record names no producer spool (SISTEER_REGISTRY_SPOOL as submitted) — cannot prove absence", 3)
        ev = [self.task_evidence(att, t) for t in dead]
        unknown = [e for e in ev if e["spool"] is None]
        if unknown:
            raise Refused(f"--resubmit refused: {lwid}: the producer spool {att.get('spool')} is missing or could not be traversed on {label} for task(s) "
                          f"{','.join(str(e['task'] if e['task'] is not None else e['raw']) for e in unknown)} — not proof of absence", 3)
        # THE FOLD PROOF (hole 1): every record file the dead tasks left in the producer spool must be folded by the harvester to a durable
        # cursor — present in record_ingest with the same sha256 and record count, no record_cursor — before the same identities are re-run
        unfolded = []
        for e in ev:
            files = r.spool_files_mentioning([f'"{e["raw"]}"', f"@{e['raw']}"], att.get("spool"))
            if files is None:
                raise Refused(f"--resubmit refused: {lwid}: the producer spool {att.get('spool')} could not be listed on {label} for task {e['task'] if e['task'] is not None else e['raw']} — the fold of its records cannot be proven", 3)
            e["spool_files"] = files
            for u in self.records_folded(files):
                unfolded.append({"task": e["task"] if e["task"] is not None else e["raw"], **u})
        if unfolded:
            what = "; ".join(f"task {u['task']}: {u['rel']} — {u['why']}" for u in unfolded[:8]) + (f"; … {len(unfolded) - 8} more" if len(unfolded) > 8 else "")
            raise Refused(f"--resubmit refused: {lwid}: {len(unfolded)} record file(s) of the dead job are not folded into the registry ({what}) — records still landing "
                          f"(or partially folded); a same-identity re-run would upsert into the new attempt's identities; let the harvester fold them (`registry ingest --pull`), then retry", 3)
        past = {e["raw"]: e["past_running"] for e in ev if e["past_running"]}
        why_tail = ""; resume_proof: Optional[dict] = None
        if past:
            # petri-completion (2026-10-13): a run past `running` is a RESULT and refuses by name — except under a script-keyed RESUME PROOF
            # (RESUME_PROOFS: the Petri proxy), which must show every such run sits on a leg the resumed drive reads as done (the drive's own
            # predicate, at the same paths, for the same identity); a leg it would regenerate keeps the refusal, by name
            proof = resume_proof_for(att.get("script") or spec.get("script"))
            if proof is None:
                raise Refused(f"--resubmit refused: {lwid}'s bound job {job_id} left run(s) past `running` in the registry ({past}) — those are results; attach/resume them, never re-run", 3)
            done_note, resume_proof = proof(self, lwid, att, job_id, past)
            why_tail = f"; {done_note}"
        registered = [e for e in ev if e["registered"]]
        n_files = sum(len(e.get("spool_files") or []) for e in ev)
        if not registered:
            why = f"all checks negative for every task (pending absent, spool traversed empty, no registry row; watermark {ing}, end {latest})"
            return "resubmit", why, {"tasks": tasks, "evidence": ev, "contract": contract}
        if (att.get("kind", kind) or "target") == "fit":
            raise Refused(f"--resubmit refused: the bound job {job_id} on {label} is not provably dead-and-unregistered (registered: task(s) "
                          f"{','.join(str(e['task'] if e['task'] is not None else e['raw']) for e in registered)} filed {sum(len(e['run_ids']) for e in registered)} run(s)) — a fit's successor attaches by name, never a re-mint", 3)
        why = (f"REGISTERED before dying (task(s) {','.join(str(e['task'] if e['task'] is not None else e['raw']) for e in registered)}: "
               f"{sum(len(e['run_ids']) for e in registered)} run id(s), {'none past `running`' if not past else str(sum(len(v) for v in past.values())) + ' past `running` under the resume proof'}; "
               f"{n_files} record file(s) folded to a durable cursor; watermark {ing}, end {latest}){why_tail}")
        return "rerun", why, {"tasks": tasks, "evidence": ev, "contract": contract, "resume_proof": resume_proof}

    def mirror_status_path(self) -> str:
        """`scripts/registry/mirror_status.py`'s output (C): SISTEER_REGISTRY_MIRROR_STATUS read as a stage reads a site key (the given
        file over the environment, Site.given — it read the environment first, the roots' class), else <registry
        root>/state/mirror_status.json."""
        return self.site.given("SISTEER_REGISTRY_MIRROR_STATUS") or os.path.join(self.site.registry_root, "state", "mirror_status.json")

    def ingested_past(self, label: str, end_utc: str) -> Optional[bool]:
        p = self.mirror_status_path()
        if not os.path.isfile(p) or not end_utc or end_utc in ("Unknown", "None"):
            return None
        try:
            doc = json.load(open(p, encoding="utf-8"))
            sub = self.site.clusters[label]["CLUSTER_SUBSTRATE"]; lab = self.site.clusters[label]["CLUSTER_LABEL"]
            ent = doc.get(lab) or doc.get(sub) or {}
            through = ent.get("ingested_through_utc")
            return bool(through) and through >= end_utc.replace(" ", "T")
        except Exception:
            return None

    def fenced_proof_never_registered(self, lwid: str, aid: str, a: dict, sac: Optional[dict] = None) -> Tuple[str, str]:
        """The close-round rule, read-only at the winning cluster -> (verdict, why): `never_registered` only when ALL FOUR checks
        are negative; `registered` on POSITIVE evidence (pending file, spool record, registry run); `unknown` for anything
        unreachable, unreadable or not yet settled. The caller branches on the verdict, never on the text."""
        r = self.remotes[a["cluster"]]
        if sac is None:
            try:
                sac = r.sacct(a["job_id"])
            except Unreachable as exc:
                return "unknown", f"check 1 (sacct terminal) unknown: {exc}"
        if sac is None or not sac["terminal"]:
            return "unknown", f"check 1: job {a['job_id']} is not terminal ({sac['state'] if sac else 'no sacct record'})"
        # checks 2 + 3 read the producer paths the attempt was SUBMITTED with (its ledger record) — never a generic root
        pend = f"{a.get('pending_dir') or r.shared + '/registry/pending'}/{a['job_id']}.json"
        try:
            if r.exists(pend):
                return "registered", f"check 2: pending file {pend} PRESENT — the winner registered"
        except Unreachable as exc:
            return "unknown", f"check 2 (pending file) unknown: {exc}"
        if not a.get("spool"):
            return "unknown", "check 3: the attempt's ledger record names no producer spool (SISTEER_REGISTRY_SPOOL as submitted) — cannot prove absence"
        try:
            sp = r.spool_mentions([f'"{a["job_id"]}"', aid, f"{a.get('job_name') or ''}@{a['job_id']}"], a["spool"])
        except Unreachable as exc:
            return "unknown", f"check 3 (spool) unknown: {exc}"
        if sp is None:
            return "unknown", f"check 3: the producer spool {a['spool']} is missing or could not be traversed on {a['cluster']} — not proof of absence"
        if sp:
            return "registered", "check 3: a spool record names the job / attempt — the winner registered locally"
        runs = self.registry_runs_for(a["job_id"], aid, cluster=a.get("cluster"))
        if runs is None:
            return "unknown", "check 4: no registry database to query"
        if runs:
            return "registered", f"check 4: registry has run(s) {runs} for the attempt — the winner registered"
        ing = self.ingested_past(a["cluster"], sac.get("end", ""))
        if ing is None:
            return "unknown", f"check 4: the cluster's ingest watermark is unknown ({self.mirror_status_path()} absent or unreadable; scripts/registry/mirror_status.py writes it)"
        if not ing:
            return "unknown", f"check 4: the cluster's inbox is not yet ingested past the job's end {sac.get('end')}"
        return "never_registered", "all four checks negative"

    def reconcile(self) -> int:
        rc = 0; n_units = 0; n_plain = 0; n_elections = 0
        for lwid, st in self.ledger.state().items():
            n_units += 1
            if st["closed"]:
                continue
            if st["halted"]:
                log(f"HALTED {lwid}: {st['halted']} (a human clears it: `attempts.jsonl` += unhalted)"); rc = 3; continue
            # A PLAIN submission (never --twin): no election, no sentinels, ONE attempt whose registered runs are its own — the watcher and
            # reconcile.sh's sacct pass own its lifecycle. Found by tests/test_front_door_wave_integration.py: the "run from a NON-winner"
            # rule below would otherwise HALT every registered plain unit, its winner being None.
            if st["winner"] is None and not any(a.get("twin") for a in st["attempts"].values()):
                n_plain += 1; continue
            n_elections += 1
            # consistency of the remote sentinels vs the ledger
            grants = []
            for aid, a in st["attempts"].items():
                try:
                    if self.remotes[a["cluster"]].exists(self.attempt_dir(a["cluster"], aid) + "/GRANT"):
                        grants.append(aid)
                except Unreachable as exc:
                    log(f"{lwid}: cannot read sentinels on {a['cluster']} ({exc}); leaving the election open"); rc = 3; grants = None; break
            if grants is None:
                continue
            if len(grants) > 1 or (grants and grants[0] != st["winner"]):
                reason = f"remote GRANT set {grants} disagrees with the committed winner {st['winner']!r} — two grants or a grant for an uncommitted attempt"
                self.ledger.append({"event": "halted", "logical_work_id": lwid, "reason": reason}); log(f"HALT {lwid}: {reason}"); rc = 3; continue
            # a run from a non-winner?
            for aid, a in st["attempts"].items():
                if aid != st["winner"]:
                    runs = self.registry_runs_for(a["job_id"], aid) or []
                    if runs:
                        reason = f"registry has run(s) {runs} from NON-winner attempt {aid}"
                        self.ledger.append({"event": "halted", "logical_work_id": lwid, "reason": reason}); log(f"HALT {lwid}: {reason}"); rc = 3; break
            else:
                if st["winner"] is None:
                    log(f"{lwid}: no winner committed yet — run `elect --lwid {lwid}`"); continue
                w = st["winner"]; a = st["attempts"][w]
                try:
                    sac = self.remotes[a["cluster"]].sacct(a["job_id"])
                except Unreachable as exc:
                    log(f"{lwid}: winner's cluster unreachable ({exc})"); rc = 3; continue
                if sac is None or not sac["terminal"]:
                    if w not in st["granted"]:
                        self.grant(lwid, w)          # replay: committed but the GRANT (and the DENYs) never landed
                    else:
                        log(f"{lwid}: winner {w} alive ({sac['state'] if sac else 'queued'}) — nothing to do")
                    continue
                if sac["state"] == "COMPLETED":
                    if w not in st["granted"]:
                        self.grant(lwid, w)          # it completed: its siblings still owe a DENY
                    self.ledger.append({"event": "closed", "logical_work_id": lwid}); log(f"{lwid}: winner {w} COMPLETED — closed"); continue
                # the winner is TERMINAL and not COMPLETED: did it register?
                winner_runs = self.registry_runs_for(a["job_id"], w) or []
                if st["registered"].get(w) or winner_runs:
                    verdict, why = "registered", "ledger/registry say registered"
                else:
                    verdict, why = self.fenced_proof_never_registered(lwid, w, a, sac)
                if verdict == "registered":
                    # a pending file, a spool record or a registry run at the WINNING cluster = it registered: attach, never re-grant
                    log(f"ATTACH {lwid}: winner {w} (job {a['job_id']}, {sac['state']}) REGISTERED before dying ({why}) — resume/attach on THAT identity (same-name resubmit; a fit's successor is bound by name), never re-grant")
                    self.ledger.append({"event": "registered", "logical_work_id": lwid, "attempt_id": w, "run_ids": winner_runs, "evidence": why})
                    if w not in st["granted"]:
                        self.grant(lwid, w)          # its siblings must still be denied (the identity is taken)
                    continue
                if verdict != "never_registered":
                    reason = f"winner {w} is {sac['state']} but a re-grant is not provably safe: {why}"
                    self.ledger.append({"event": "halted", "logical_work_id": lwid, "reason": reason}); log(f"HALT {lwid}: {reason} — nothing cancelled"); rc = 3; continue
                # provably died BEFORE registering: a LIVE sibling not yet denied is granted (ledger first); else a NEW attempt from the spec
                sib = None
                for aid, b in st["attempts"].items():
                    if aid == w or aid in st["denied"]:
                        continue
                    try:
                        s_ = self.remotes[b["cluster"]].squeue_state(b["job_id"])
                    except Unreachable:
                        s_ = None
                    if s_ in ("PENDING", "RUNNING", "CONFIGURING"):
                        sib = aid; break
                if sib is None:
                    spec = a.get("spec")
                    if not spec:
                        log(f"REGRANT-NEEDED {lwid}: winner {w} died before registering ({why}), no live sibling and no recorded spec — resubmit the unit by hand"); rc = 3; continue
                    log(f"RESUBMIT {lwid}: winner {w} died before registering ({why}); a NEW attempt on {a['cluster']}")
                    sib, _ = self.submit_one(a["cluster"], twin=True, force=True, **{k: spec.get(k) for k in SPEC_KEYS})
                self.regrant_after_dead_winner(lwid, w, a, sib, why)
        log(f"reconcile: {n_units} unit(s) in the ledger — {n_plain} plain submission(s) (no election), {n_elections} election(s) examined, rc={rc}")
        return rc

    def cancel_attempt(self, aid: str) -> None:
        for lwid, st in self.ledger.state().items():
            if aid in st["attempts"]:
                a = st["attempts"][aid]
                self.remotes[a["cluster"]].write_sentinel(self.attempt_dir(a["cluster"], aid) + "/DENY", json.dumps({"attempt_id": aid, "cancelled": True, "utc": utc_now()}))
                self.remotes[a["cluster"]].scancel(a["job_id"])
                self.ledger.append({"event": "denied", "logical_work_id": lwid, "attempt_id": aid, "cancelled": True})
                log(f"cancelled attempt {aid} (job {a['job_id']} on {a['cluster']})"); return
        raise Refused(f"attempt {aid} is not in the ledger")

    def pool_root(self, label: str, group: str) -> str:
        return f"{self.site.cluster_root(label, self.model)}/waves/{group}/attackers"

    def member_self_exited(self, label: str, group: Optional[str], job_id: str) -> bool:
        """The member wrote attackers/<job>.SELF_EXIT on its group's shared root (attacker_pool_lifecycle.sh, 31b)."""
        if not group:
            return False
        try:
            return self.remotes[label].exists(f"{self.pool_root(label, group)}/{job_id}.SELF_EXIT")
        except Exception:      # noqa: BLE001 — unreachable: not proven, the caller's other proofs decide
            return False

    def teardown_pool(self, group: str, *, placed: Optional[dict]) -> dict:
        """`attackers/<job>.STOP` for every member the ledger records for the group (never attackers/STOP); a member that wrote <job>.SELF_EXIT
        is recorded torn_down, not stopped again (31b). The STOP writes are ONE program on the group's cluster under `flock <pool_root>/PLACED.lock`
        that first reads attackers/PLACED.transition (the backstop's teardown used to bypass the member fence — a --resubmit pass had opened the
        marker and re-minted a member, the old targets were all terminal, and the waiter's next poll wrote STOP for the replacement; the
        supersession then assigned targets to a server that was gone). A FRESH marker HOLDS the whole
        teardown — {result: held, marker, left_s}, nothing written — since the pass that opened it is about to re-place the group onto exactly
        these members; the check and the writes share the lock the pass's `open` takes, so an `open` lands either before this reads (held) or
        after it wrote (its replacement is not yet in the ledger, so none of these STOPs is the replacement's). A lapsed or unreadable marker is
        ignored and named (the lease is the fence's bound).
        `placed` is the attackers/PLACED record the CALLER read and decided on (None: it read none). Its identity — the version and the (unit,
        job) pairs — travels in the intent, and the program compares it with the record under the lock BEFORE its first write (a whole pass —
        open, re-mint, mark-placed — could land between the waiter's read of v1 and its teardown; the commit had cleared the marker, so the lock
        found nothing to hold on, and the STOPs went to the members v2 had just been assigned). A record that moved answers {result: stale,
        expected_version, current_version, added, removed} with nothing written; the caller reads the current placement and decides again (the
        waiter's next poll). Returns {result: stopped | held | stale | dry, stopped: [job], self_exited: [job], marker_ignored, …}."""
        members = []
        for lwid, st in self.ledger.state().items():
            for m in st["pool_jobs"]:
                if lwid.endswith(f"/{group}/attackers") or f"/{group}/" in lwid:
                    members.append((lwid, m))
        if not members:
            raise Refused(f"no attacker members recorded for group {group}")
        if self.site.dry:
            for lwid, m in members:
                print(f"DRY-RUN: teardown-pool {group}: {self.pool_root(m['cluster'], group)}/{m['job_id']}.STOP (unless the member self-exited, the placement is in transition, "
                      f"or attackers/PLACED is no longer the record read)")
            return {"result": "dry", "stopped": [], "self_exited": []}
        by_cluster: Dict[str, List[Tuple[str, dict]]] = {}
        for lwid, m in members:
            by_cluster.setdefault(m["cluster"], []).append((lwid, m))
        out = {"result": "stopped", "stopped": [], "self_exited": [], "marker_ignored": None}
        expect = self._placed_expect(placed)
        for cluster, ms in sorted(by_cluster.items()):        # a group lives on one cluster (mark-placed enforces it); the loop is the ledger's shape
            root = self.pool_root(cluster, group)
            intent = json.dumps({"group": group, "jobs": [str(m["job_id"]) for _, m in ms], "expect": expect, "utc": utc_now()}, sort_keys=True)
            cmd = (f"mkdir -p {shlex.quote(root)} && flock -w 300 {shlex.quote(root + '/PLACED.lock')} python3 -c {shlex.quote(TEARDOWN_TX_PY)} "
                   f"{shlex.quote(root)} {shlex.quote(intent)}")
            rc, o, err = self.remotes[cluster].sh(cmd, timeout=400, allow_fail=True)
            if rc != 0 or not (o or "").strip():
                raise Refused(f"teardown {group}: the teardown transaction on {cluster} failed (rc {rc}): {(err or o or '').strip()[-300:]}; nothing recorded", 3)
            try:
                res = json.loads(o.strip().splitlines()[-1])
            except ValueError:
                raise Refused(f"teardown {group}: the teardown transaction on {cluster} answered unreadably: {o.strip()[-300:]}", 3)
            if res.get("result") == "held":
                marker = res.get("marker") or {}
                log(f"teardown {group}: HELD — the group's placement is in TRANSITION ({self._transition_phrase(root, marker, res.get('left_s'))}): a pass is re-placing "
                    f"the group and the next PLACED may assign targets to a member this would stop; nothing written on {cluster} "
                    f"(it proceeds once a placement commits, the lease lapses, or `placement-transition close --group {group} --cluster {cluster}` releases it)")
                return {"result": "held", "marker": marker, "left_s": res.get("left_s"), "cluster": cluster, "stopped": out["stopped"], "self_exited": out["self_exited"]}
            if res.get("result") == "stale":
                # placement-fix round 5: the record under the lock is not the one this teardown decided on — a placement committed (and cleared
                # its marker) after the caller read PLACED; the current record may assign targets to the very members this would stop
                log(f"teardown {group}: STALE — attackers/PLACED on {cluster} is version {res.get('current_version')}, not the version {res.get('expected_version')} this teardown "
                    f"decided on (a placement committed in between: {len(res.get('added') or [])} target pair(s) added, {len(res.get('removed') or [])} removed); the current "
                    f"placement may assign targets to members this would have stopped — nothing written on {cluster}")
                return {"result": "stale", "expected_version": res.get("expected_version"), "current_version": res.get("current_version"), "added": res.get("added") or [],
                        "removed": res.get("removed") or [], "cluster": cluster, "stopped": out["stopped"], "self_exited": out["self_exited"]}
            if res.get("result") != "stopped":
                raise Refused(f"teardown {group}: the teardown transaction on {cluster} answered {res.get('result')!r}", 3)
            exited = {str(j) for j in (res.get("self_exited") or [])}
            for lwid, m in ms:
                if str(m["job_id"]) in exited:
                    # 31b: the member already left on its own; record the teardown it performed, write no STOP over it
                    self.ledger.append({"event": "torn_down", "logical_work_id": lwid, "job_id": m["job_id"], "cluster": cluster, "self_exit": True})
                    log(f"teardown {group}: {root}/{m['job_id']}.SELF_EXIT already there (the member closed itself) — recorded, no STOP written")
                    out["self_exited"].append(str(m["job_id"])); continue
                self.ledger.append({"event": "torn_down", "logical_work_id": lwid, "job_id": m["job_id"], "cluster": cluster})
                log(f"teardown {group}: {root}/{m['job_id']}.STOP"); out["stopped"].append(str(m["job_id"]))
            if res.get("marker_ignored"):
                out["marker_ignored"] = res["marker_ignored"]
                log(f"teardown {group}: a placement-transition marker on {cluster} was ignored ({res['marker_ignored']}: {self._transition_phrase(root, res.get('marker') or {}, res.get('left_s'))})")
        return out

    @staticmethod
    def _placed_expect(placed: Optional[dict]) -> dict:
        """The identity of the attackers/PLACED record a caller read — {version, targets: [[unit, job], …]} — as the destructive transactions
        compare it under the group's lock (PLACED_IDENTITY_PY, placement-fix round 5). None (the caller read no record) is version 0 with no
        targets — what placed_record answers None for — so a record that appears under a teardown decided without one is a change too."""
        if not isinstance(placed, dict):
            return {"version": 0, "targets": []}
        pairs = sorted({(str(t.get("logical_work_id")), str(t.get("job_id"))) for t in (placed.get("targets") or []) if isinstance(t, dict) and t.get("job_id")})
        return {"version": int(placed.get("version") or 1), "targets": [list(p) for p in pairs]}

    @staticmethod
    def _transition_phrase(root: str, marker: Mapping[str, object], left_s: Optional[float]) -> str:
        """How a log line names attackers/PLACED.transition: `<root>/PLACED.transition since <utc> for <lease> s (<left> s left | lapsed <n> s ago): <reason>`."""
        if marker.get("error"):
            return f"{root}/PLACED.transition unreadable ({marker.get('error')})"
        when = ""
        if left_s is not None:
            when = f" ({int(left_s)} s left)" if float(left_s) > 0 else f" (lapsed {int(-float(left_s))} s ago)"
        return f"{root}/PLACED.transition since {marker.get('utc')} for {marker.get('lease_s')} s{when}: {marker.get('reason') or 'no reason'}"

    def live_members(self, group: str) -> List[dict]:
        """The group's pool members the ledger records that are neither torn down / retiring in the ledger nor terminal on sacct:
        [{lwid, job_id, cluster}]."""
        out = []
        for lwid, st in self.ledger.state().items():
            if not (lwid.endswith(f"/{group}/attackers") or f"/{group}/" in lwid):
                continue
            gone = {str(j.get("job_id")) for j in (st.get("torn_down_jobs") or [])} | set(st.get("retiring_jobs") or [])
            for m in st["pool_jobs"]:
                if str(m["job_id"]) in gone:
                    continue
                try:
                    tasks = self.remotes[m["cluster"]].sacct_tasks(str(m["job_id"]))
                except Unreachable:
                    raise
                if tasks and all(t.get("terminal") for t in tasks):
                    continue
                out.append({"lwid": lwid, "job_id": str(m["job_id"]), "cluster": m["cluster"]})
        return out

    def member_consumer_jobs(self, group: str, member: dict) -> Optional[List[str]]:
        """The TARGET job ids the member serves: the targets whose binding on the group's shared root names its URL UNION the targets PLACED
        assigned it (31b-fix), each once; None when the shared root could not be read."""
        root = self.pool_root(member["cluster"], group); serving = os.path.dirname(root) + "/serving"
        script = (
            "import json, glob, os, sys\n"
            "root, serving, job = sys.argv[1:4]\n"
            "rec = os.path.join(root, job + '.json')\n"
            "url = json.load(open(rec)).get('url') if os.path.isfile(rec) else None\n"
            "ids = set()\n"
            "if url:\n"
            "    for b in glob.glob(os.path.join(serving, 'attacks_*', 'attacker_binding.json')):\n"
            "        try:\n"
            "            if url in (json.load(open(b)).get('urls') or []): ids.add(os.path.basename(os.path.dirname(b)).rsplit('_', 1)[-1])\n"
            "        except Exception: pass\n"
            "p = os.path.join(root, 'PLACED')\n"
            "if os.path.isfile(p):\n"
            "    for t in (json.load(open(p)).get('targets') or []):\n"
            "        if str(t.get('assigned_member_job_id') or '') == job and t.get('job_id'): ids.add(str(t['job_id']))\n"
            "print('\\n'.join(sorted(ids)))\n"
        )
        rc, out, _ = self.remotes[member["cluster"]].sh(f"python3 -c {shlex.quote(script)} {shlex.quote(root)} {shlex.quote(serving)} {shlex.quote(str(member['job_id']))}", allow_fail=True)
        if rc != 0:
            return None
        return [ln.strip() for ln in (out or "").splitlines() if ln.strip()]

    def member_live_consumers(self, group: str, member: dict) -> Optional[int]:
        """How many of the member's consumers are still LIVE on sacct (31d-fix): a historical assignment to a finished
        target is not a consumer; a job sacct does not know yet, or a cluster that does not answer, counts as live. None when unreadable."""
        jobs = self.member_consumer_jobs(group, member)
        if jobs is None:
            return None
        live = 0
        for j in jobs:
            try:
                tasks = self.remotes[member["cluster"]].sacct_tasks(j)
            except Unreachable:
                live += 1; continue
            if not tasks or any(not t.get("terminal") for t in tasks):
                live += 1
        return live

    def retire_member(self, group: str, lwid: str, job_id: str, cluster: str, why: str, *, n_live_targets: int, placed: Optional[dict], live_jobs: Optional[Set[str]] = None) -> str:
        """Mark ONE member RETIRING — attackers/<job>.RETIRE on the group's shared root + ledger `retiring` (31d): the member withdraws its
        record, drains and self-exits; its drivers re-bind to a survivor. The check and the write run as ONE program under the group's lock
        (PLACED.lock; 31d-fix): concurrent retirers see each other's RETIRE files, and the group's LAST serving member
        (a record in discovery, neither retiring nor self-exited) is never withdrawn while `n_live_targets` > 0. The transaction also reads
        attackers/PLACED.transition under that lock: a FRESH marker HOLDS the retirement — `held_transition`, nothing
        written — since a pass is re-placing the group and the next PLACED may assign this member targets (the backstop's ratio used to retire a
        replacement that v1 assigned nothing); a lapsed or unreadable marker is ignored and named. `placed` is the attackers/PLACED record the
        caller counted this member's consumers on (None: none); a record that moved by the time the lock is held answers `stale_placement` with
        nothing written (the consumer count was the old placement's, and the current one may assign the member targets).
        Returns retired | unchanged | refused_last | held_transition | stale_placement."""
        root = self.pool_root(cluster, group)
        # 31d-fix-2: a surviving SERVER is a `.json` discovery record (written only once vLLM answered its health check)
        # whose job is LIVE on sacct — a `.pending` member (booting, ~22 min) or a record whose job ended never counts
        if live_jobs is None:
            live_jobs = {str(m["job_id"]) for m in self.live_members(group)}
        intent = json.dumps({"group": group, "job_id": str(job_id), "why": why, "n_live_targets": int(n_live_targets), "live_jobs": sorted(live_jobs),
                             "expect": self._placed_expect(placed), "utc": utc_now()}, sort_keys=True)
        cmd = (f"mkdir -p {shlex.quote(root)} && flock -w 300 {shlex.quote(root + '/PLACED.lock')} python3 -c {shlex.quote(RETIRE_TX_PY)} "
               f"{shlex.quote(root)} {shlex.quote(intent)}")
        rc, out, err = self.remotes[cluster].sh(cmd, timeout=400, allow_fail=True)
        if rc != 0 or not (out or "").strip():
            raise Refused(f"retire {group}: the retirement transaction on {cluster} failed (rc {rc}): {(err or out or '').strip()[-300:]}; nothing recorded", 3)
        try:
            res = json.loads(out.strip().splitlines()[-1])
        except ValueError:
            raise Refused(f"retire {group}: the retirement transaction on {cluster} answered unreadably: {out.strip()[-300:]}", 3)
        result = res.get("result")
        ignored = (f" (a placement-transition marker was ignored — {res['marker_ignored']}: {self._transition_phrase(root, res.get('marker') or {}, res.get('left_s'))})"
                   if res.get("marker_ignored") else "")
        if result == "retired":
            self.ledger.append({"event": "retiring", "logical_work_id": lwid, "job_id": str(job_id), "cluster": cluster, "group": group, "why": why, "survivors": res.get("survivors")})
            log(f"retire {group}: member job {job_id} on {cluster} marked RETIRING ({len(res.get('survivors') or [])} member(s) still serving) — {why}{ignored}")
        elif result == "held_transition":
            log(f"retire {group}: member job {job_id} on {cluster} — HELD: the group's placement is in TRANSITION ({self._transition_phrase(root, res.get('marker') or {}, res.get('left_s'))}): "
                f"a pass is re-placing the group and the next PLACED may assign it targets; nothing written (it proceeds once a placement commits, the lease lapses, "
                f"or `placement-transition close --group {group} --cluster {cluster}` releases it) — would have retired it: {why}")
        elif result == "stale_placement":
            log(f"retire {group}: member job {job_id} on {cluster} — STALE: attackers/PLACED is version {res.get('current_version')}, not the version {res.get('expected_version')} "
                f"this retirement decided on (a placement committed in between; the current one may assign it targets); nothing written — would have retired it: {why}")
        elif result == "unchanged":
            log(f"retire {group}: member job {job_id} on {cluster} is already RETIRING — nothing written")
        elif result == "refused_last":
            log(f"retire {group}: member job {job_id} on {cluster} is the group's LAST serving member while {n_live_targets} target(s) are live — NOT retired "
                f"(no healthy replacement: serving {res.get('serving')}, pending/not yet serving {res.get('pending')}, not live on sacct {res.get('dead')}, "
                f"no fresh ok health lease {res.get('unfit')}, withdrew as unhealthy {res.get('unhealthy')})")
        else:
            raise Refused(f"retire {group}: the retirement transaction on {cluster} answered {result!r}", 3)
        return str(result)

    def enforce_pool_ratio(self, group: str, n_live_targets: int, *, placed: Optional[dict]) -> List[str]:
        """Members allowed = stage.attacker_pool_size(live targets); the surplus is retired from the members with NO live consumer (bound or
        assigned, non-terminal on sacct), highest job id first; a member serving a live target is never retired, however the ratio stands
        (31d-fix: a historical assignment to a finished target ranked a live-serving member for retirement).
        `placed` is the attackers/PLACED record the caller's poll read: every retirement carries its identity, and one that moved by the write
        holds the rest of this call (the live-target count and the consumer counts are the old placement's).
        Returns the job ids retired this call."""
        members = self.live_members(group)
        allowed = ST.attacker_pool_size(max(n_live_targets, 1))
        surplus = len(members) - allowed
        if surplus <= 0:
            return []
        live_cons = {str(m["job_id"]): self.member_live_consumers(group, m) for m in members}
        live_jobs = {str(m["job_id"]) for m in members}
        idle = sorted((m for m in members if live_cons[str(m["job_id"])] == 0), key=lambda m: -int(m["job_id"]) if str(m["job_id"]).isdigit() else 0)
        retired = []
        for m in idle[:surplus]:
            res = self.retire_member(group, m["lwid"], m["job_id"], m["cluster"],
                                     f"{len(members)} live member(s) for {n_live_targets} live target(s); {allowed} allowed (one per three); the member serves no live target",
                                     n_live_targets=n_live_targets, placed=placed, live_jobs=live_jobs)
            if res == "retired":
                retired.append(m["job_id"])
            elif res == "held_transition":
                # placement-fix round 4: the ratio is enforced over a COMMITTED placement — while the group's is in transition its live-target
                # count and member list are both about to change (the pass is re-minting them), so the rest of this poll's surplus is held too
                log(f"ratio {group}: enforcement held this poll — the placement is in transition ({len(idle[:surplus])} idle member(s) above one-per-three not retired)")
                break
            elif res == "stale_placement":
                # placement-fix round 5: a placement committed after this poll read PLACED — the counts above are the old placement's; the next
                # poll reads the current one and counts again
                log(f"ratio {group}: enforcement held this poll — attackers/PLACED moved since this poll read it ({len(idle[:surplus])} idle member(s) above one-per-three not retired)")
                break
        if surplus > len(idle):
            busy = sorted(str(m["job_id"]) for m in members if live_cons[str(m["job_id"])] != 0)
            log(f"ratio {group}: {surplus - len(idle)} member(s) above one-per-three still serve live targets ({busy}) — not retired")
        return retired

    def mark_placed(self, group: str, targets: List[str], *, supersede: bool = False, reason: Optional[str] = None) -> dict:
        """The group's placement is COMPLETE: every named target unit is bound (submitted or registered) -> attackers/PLACED on the group's
        cluster {group, targets: [{job_id, logical_work_id, state, assigned_member_job_id}], members, version, utc} + ledger
        `placement_complete` (31b). A target still unbound refuses by name: a member must not read "placed" while a target may still come and
        bind it. A placement is IMMUTABLE: a repeat with the same roster and members returns the record unchanged
        (nothing written, no event); a different roster refuses unless `supersede` with a `reason` — then the previous record is archived
        (PLACED.v<n>), every unchanged (unit, job) pair keeps its member, only the new pairs are assigned, and the transition is recorded
        (version n+1, `supersedes`, `reason`, `changes`; ledger `placement_superseded`)."""
        if not targets:
            raise Refused(f"mark-placed {group}: no targets named (--targets lw,lw…)")
        rows = []; label = None
        for lw in targets:
            b = ST.binding_state(self.subs.path, lw)
            if b.get("state") not in ("submitted", "registered") or not b.get("job_id"):
                raise Refused(f"mark-placed {group}: target {lw} is {b.get('state')} — the group is not fully placed; nothing written", 3)
            rows.append({"logical_work_id": lw, "job_id": str(b["job_id"]), "cluster": b.get("cluster"), "state": b["state"]})
            label = label or b.get("cluster")
        clusters = {r["cluster"] for r in rows}
        if len(clusters) != 1:
            raise Refused(f"mark-placed {group}: the targets are bound on {sorted(clusters)} — a group lives on one cluster; nothing written", 3)
        # 31b-fix: an IMMUTABLE per-member assignment at PLACEMENT — target i -> member i mod n over the group's
        # recorded pool members (ledger pool_jobs, torn-down ones excluded), so a member knows its consumers before any target starts
        # (a target's binding is written at ITS runtime) and the target job binds its assigned member first (target_attacks.sbatch)
        members = self.pool_members(group)
        if not members:
            raise Refused(f"mark-placed {group}: no attacker members recorded for the group — the placement assigns each target a member; submit the pool first; nothing written", 3)
        root = self.pool_root(label, group)
        # 31b-fix-3: read/compare/version/archive/write is ONE remote step under the group's lock (flock on
        # <pool_root>/PLACED.lock) — two concurrent mark-placed invocations serialize there; the second reads what the first published
        res = self._placement_transaction(label, root, group, rows, members, supersede=supersede, reason=reason)
        _crash("placement_after_remote")     # fault injection: PLACED is live on the cluster, the ledger has no event yet (the 31b-fix-4 window)
        result, doc = res.get("result"), res.get("doc") or {}
        prev_version = int(res.get("prev_version") or doc.get("version") or 1)
        # The placement half of the fix: the transaction dropped every member whose attackers/<job>.SELF_EXIT stood under
        # the group's lock — members this invocation handed as live because the walk read their jobs as active on Slurm (a job in its exit
        # path is still RUNNING/COMPLETING there). Every answer below names them; the record and its ledger event carry them too.
        excluded = [str(m) for m in (res.get("excluded_self_exited") or [])]
        live_members = [m for m in members if m not in excluded]
        excl_note = (f" — {len(excluded)} member(s) EXCLUDED under the group's lock: attackers/<job>.SELF_EXIT stood on {label} for {excluded} (left, or leaving: the walk read "
                     f"their jobs as active on Slurm, and a job in its exit path is still RUNNING/COMPLETING there); the placement assigns them nothing" if excluded else "")
        # 31b-fix-5: every EARLIER version the cluster holds — the archived PLACED.v<n> records, and on a refusal
        # the live record the refusal stands on — has its ledger event reconciled FIRST, check-and-append under the ledger lock. A mark-placed
        # that died after committing v1 and before its append, followed by a roster change, used to leave v1's placement_complete missing for
        # good: `refused` repaired nothing and --supersede appended only v2, so the append-only ledger lacked the supersession's baseline.
        earlier = [v for v in (res.get("versions") or []) if isinstance(v, dict)] + ([doc] if doc and result in ("refused", "needs_reason", "no_live_member") else [])
        n_rec = sum(1 for v in earlier if self._ensure_placement_event(label, group, v))
        if n_rec:
            log(f"placed {group}: reconciled {n_rec} missing ledger event(s) for earlier placement version(s) {[int(v.get('version') or 1) for v in earlier]} "
                f"(an earlier invocation died after its remote commit)")
        if result == "no_live_member":
            # round 6: every member the ledger records for the group has left (or is leaving) — a placement cannot assign targets to servers
            # that are gone, and this pass minted no replacement for any of them (its walk read their jobs as active). Nothing written; the marker stands as
            # every refusal leaves it, bounded by its lease. The remedy is the SAME pass again once sacct reports the exited job(s) terminal:
            # its --walk then reads the SELF_EXIT, records the teardown and re-mints the member, and this mark-placed commits the supersession.
            # Not retried here: the re-mint needs sacct to report the job terminal, which lags the tombstone by the job's exit and the
            # accounting's write — a wait on a timer, not an event this invocation can read; the next pass reads it as a fact. rc 4 so 06 can
            # tell this refusal from the immutable-placement fence (rc 3) and print the remedy, not the sanctioned supersession.
            marker, left = res.get("marker") or {}, res.get("left_s")
            if res.get("marker_state") == "fresh" and isinstance(marker, dict):
                marker_note = (f"the pass's PLACED.transition marker stands as before (since {marker.get('utc')}, {marker.get('lease_s')} s lease, {int(left or 0)} s left: "
                               f"{marker.get('reason')!r}) — it holds no exit of a member that has already left, and its lease bounds it; the re-run refreshes it and its "
                               f"commit clears it, or release it with `placement-transition close --group {group} --cluster {label}`")
            elif res.get("marker_state") in ("lapsed", "unreadable"):
                marker_note = f"the pass's PLACED.transition marker is {res.get('marker_state')} (ignored by every reader; the re-run refreshes it)"
            else:
                marker_note = "no PLACED.transition marker stands"
            standing = f"attackers/PLACED stands at version {prev_version} ({doc.get('n_targets')} target(s), members {doc.get('members')})" if doc else "no attackers/PLACED record exists"
            raise Refused(f"mark-placed {group}: NO LIVE MEMBER — every member the ledger records for the group ({members}) has attackers/<job>.SELF_EXIT on {label} "
                          f"under the group's lock (excluded {excluded}); this pass's --walk minted no replacement for any of them (it read their jobs as active on Slurm, "
                          f"and a job in its exit path is still RUNNING/COMPLETING there), so a placement would assign {len(rows)} target(s) to servers that are gone — "
                          f"nothing written: {standing}, no archive, no ledger event. REMEDY: once sacct reports job(s) {excluded} terminal, run the same `06 --family "
                          f"multiturn --resubmit` pass again, with the flags that pass ran with (06 prints it as `followup: REMEDY`, its effective flags shell-quoted) — "
                          f"its --walk then reads the SELF_EXIT, records the teardown (torn_down self_exit) and re-mints the member, "
                          f"and that pass's mark-placed commits the supersession onto the replacement; {marker_note}", 4)
        if result == "unchanged":
            # 31b-fix-4: the record may have been committed by an invocation that died before its ledger append —
            # the ledger event is reconciled idempotently from the record itself (placed or superseded, by its version), under the ledger lock
            repaired = self._ensure_placement_event(label, group, doc)
            log(f"placed {group}: already placed, unchanged ({len(rows)} target(s), members {live_members}, version {doc.get('version', 1)}) — "
                + ("the ledger lacked its event; appended it (an earlier invocation died after the remote commit)" if repaired else "nothing written")
                + (" (the pass's PLACED.transition marker cleared: the placement stands as recorded)" if res.get("transition_cleared") else "") + excl_note)
            return doc
        if result == "refused":
            raise Refused(f"mark-placed {group}: attackers/PLACED exists (version {prev_version}: {doc.get('n_targets')} target(s), members {res.get('prev_members')}) with a different "
                          f"roster or members — added {res.get('added')}, removed {res.get('removed')}, members now {live_members}; a placement is immutable: a genuinely "
                          f"resubmitted target goes through `mark-placed --supersede --reason …` (a versioned transition with provenance); nothing written{excl_note}", 3)
        if result == "needs_reason":
            raise Refused(f"mark-placed {group}: --supersede needs --reason (the provenance of the re-placement); nothing written{excl_note}", 2)
        if result == "superseded":
            changes = doc.get("changes") or {}
            self._ensure_placement_event(label, group, doc)
            log(f"placed {group}: SUPERSEDED version {prev_version} -> {doc.get('version')} ({len(changes.get('added') or [])} added, {len(changes.get('removed') or [])} removed, "
                f"{changes.get('kept')} kept their member"
                + (f", {len(changes['reassigned'])} unchanged pair(s) re-assigned off the excluded member(s) to {sorted({str(x[3]) for x in changes['reassigned']})}" if changes.get("reassigned") else "")
                + f") — {reason}"
                + (" (the pass's PLACED.transition marker cleared: the transition committed)" if res.get("transition_cleared") else "") + excl_note)
            return doc
        if result != "placed":
            raise Refused(f"mark-placed {group}: the placement transaction on {label} answered {result!r} — nothing recorded", 3)
        # a GROUP's record, not a unit's: no logical_work_id, so the ledger's unit view (grant_ledger.state) and reconcile's unit count never
        # mint a unit for it (the wave integration test counted 35 units for 33 when it carried `<model>/<group>` as one)
        self._ensure_placement_event(label, group, doc)
        log(f"placed {group}: {len(rows)} target(s) bound on {label} -> {root}/PLACED (an unbound member may now leave on its own)"
            + (" (the pass's PLACED.transition marker cleared: the placement committed)" if res.get("transition_cleared") else "") + excl_note)
        return doc

    @staticmethod
    def _placement_event_of(label: str, group: str, doc: dict) -> dict:
        """The ledger event a PLACED record stands for — `placement_complete` for version 1, `placement_superseded` for a later one — built from
        the RECORD (the durable cross-host truth), never from the caller's intent, so a reconciliation and a first append write the same event."""
        version = int(doc.get("version") or 1)
        base = {"logical_work_id": None, "scope": "group", "group": group, "cluster": label, "targets": doc.get("targets"), "members": [str(m) for m in (doc.get("members") or [])]}
        if doc.get("excluded_self_exited"):
            # placement-fix round 6: the members the transaction dropped under the lock (their SELF_EXIT stood) ride the event as the record carries them
            base["excluded_self_exited"] = [str(m) for m in doc["excluded_self_exited"]]
        if version <= 1:
            return {"event": "placement_complete", **base, "version": 1}
        sup = doc.get("supersedes") or {}
        return {"event": "placement_superseded", **base, "from_version": int(sup.get("version") or version - 1), "to_version": version,
                "reason": doc.get("reason"), "changes": doc.get("changes") or {}}

    @staticmethod
    def _placement_identity(ev: Mapping[str, object]) -> Tuple[str, str, int, Tuple[Tuple[str, str, str], ...]]:
        """(group, cluster, version, the (unit, job, member) set) — the placement's identity in the ledger."""
        version = int(ev.get("to_version") or ev.get("version") or 1)
        trip = tuple(sorted((str(t.get("logical_work_id")), str(t.get("job_id")), str(t.get("assigned_member_job_id") or "")) for t in (ev.get("targets") or []) if isinstance(t, dict)))
        return str(ev.get("group")), str(ev.get("cluster")), version, trip

    def _ensure_placement_event(self, label: str, group: str, doc: dict) -> bool:
        """Check-and-append the placement's ledger event under the ledger's lock (31b-fix-4): True when it was missing and is appended now, False
        when the ledger already carried it. Idempotent: the identity is the record's (group, cluster, version, target/job/member set)."""
        import fcntl
        want = self._placement_event_of(label, group, doc)
        key = self._placement_identity(want)
        lock_path = self.ledger.path + ".lock"
        os.makedirs(os.path.dirname(os.path.abspath(lock_path)) or ".", exist_ok=True)
        with open(lock_path, "a+") as lk:
            fcntl.flock(lk.fileno(), fcntl.LOCK_EX)
            try:
                for r in self.ledger.records():
                    if r.get("event") in ("placement_complete", "placement_superseded") and self._placement_identity(r) == key:
                        return False
                self.ledger.append(want)
                return True
            finally:
                fcntl.flock(lk.fileno(), fcntl.LOCK_UN)

    def _placement_transaction(self, label: str, root: str, group: str, rows: List[dict], members: List[str], *, supersede: bool, reason: Optional[str]) -> dict:
        """The placement's read/compare/version/archive/write as ONE program on the group's cluster under `flock <root>/PLACED.lock` (31b-fix-3):
        {result: placed | unchanged | refused | needs_reason | superseded | no_live_member, doc, prev_version, prev_members, added, removed, versions
        (31b-fix-5: every archived PLACED.v<n> record, oldest first), excluded_self_exited (round 6: the handed members whose SELF_EXIT stood under
        the lock — present only when there were any)}. The intent (rows, members, supersede, reason, utc) travels as one JSON argument; the
        program never reads the grading host's view of the record, and reads the pool's SELF_EXIT stems under the lock, never this host's."""
        intent = json.dumps({"group": group, "rows": rows, "members": members, "supersede": bool(supersede), "reason": reason, "utc": utc_now()}, sort_keys=True)
        cmd = (f"mkdir -p {shlex.quote(root)} && flock -w 300 {shlex.quote(root + '/PLACED.lock')} python3 -c {shlex.quote(PLACEMENT_TX_PY)} "
               f"{shlex.quote(root)} {shlex.quote(intent)}")
        rc, out, err = self.remotes[label].sh(cmd, timeout=400, allow_fail=True)
        if rc != 0 or not (out or "").strip():
            raise Refused(f"mark-placed {group}: the placement transaction on {label} failed (rc {rc}): {(err or out or '').strip()[-400:]}; nothing recorded", 3)
        try:
            return json.loads(out.strip().splitlines()[-1])
        except ValueError:
            raise Refused(f"mark-placed {group}: the placement transaction on {label} answered unreadably: {out.strip()[-300:]}", 3)

    def pool_members(self, group: str) -> List[str]:
        """The group's recorded pool member job ids (ledger pool_jobs of the group's attacker units; torn-down ones excluded), numerically ordered."""
        out = set()
        for lwid, st in self.ledger.state().items():
            if f"/{group}/" not in lwid:
                continue
            gone = {str(j.get("job_id")) for j in (st.get("torn_down_jobs") or [])}
            out.update(str(m["job_id"]) for m in st["pool_jobs"] if str(m["job_id"]) not in gone)
        return sorted(out, key=lambda j: (len(j), j))

    def pool_label(self, group: str) -> Optional[str]:
        for lwid, st in self.ledger.state().items():
            if f"/{group}/" in lwid and st["pool_jobs"]:
                return st["pool_jobs"][0]["cluster"]
        return None

    def placed_record(self, label: str, group: str) -> Optional[dict]:
        """attackers/PLACED on the group's shared root (the funnel's own mark-placed), or None when absent or unreadable."""
        rc, out, _ = self.remotes[label].sh(f"cat {shlex.quote(self.pool_root(label, group))}/PLACED 2>/dev/null", allow_fail=True)
        if rc != 0 or not (out or "").strip():
            return None
        try:
            doc = json.loads(out)
        except ValueError:
            return None
        return doc if isinstance(doc, dict) else None

    #: how long a placement-transition marker fences the group's members (seconds) when the pass that opened it never commits a placement —
    #: a mark-placed that failed and was never retried, a pass that died mid-submission with nobody re-running it. Long enough for a
    #: re-minted member to come up (~20 min) and an operator to run the sanctioned supersession a refusal printed; short enough that an
    #: unused member does not idle on a GPU for a day (its walltime is the only other bound).
    PLACEMENT_TRANSITION_LEASE_S = 7200

    def placement_transition(self, op: str, group: str, cluster: str, *, reason: Optional[str] = None, lease_s: Optional[int] = None, by: str = "placement-transition") -> dict:
        """OPEN or CLOSE the group's placement-transition marker, attackers/PLACED.transition on the group's cluster, under the group's lock:
        the --resubmit pass submits every replacement job BEFORE it supersedes PLACED, so a replacement member that came up while the pass was
        still submitting, or while a failed mark-placed awaited its retry, read the record that predates it, found nothing assigned to it, and
        left exit_unbound; the supersession then assigned targets to a server that was gone. The pass opens the marker before its first
        submission; the placement transaction clears it when a placement COMMITS (placed / unchanged / superseded — a refusal leaves it, the
        sanctioned supersession being that pass's continuation); the member's lifecycle holds every exit while it is fresh and ignores it once
        its lease lapsed (the bound). `cluster` is the group's label — the ledger's pool label, when it knows one, must agree. Returns the
        transaction's answer {result, doc, prev}."""
        if op not in ("open", "close"):
            raise Refused(f"placement-transition {group}: op {op!r} is not open | close", 2)
        known = self.pool_label(group)
        if known and known != cluster:
            raise Refused(f"placement-transition {group}: the ledger places the group's pool on {known}, not {cluster}; nothing written", 2)
        if op == "open" and not reason:
            raise Refused(f"placement-transition open {group}: --reason is the marker's provenance (which pass is re-placing the group); nothing written", 2)
        root = self.pool_root(cluster, group)
        intent = json.dumps({"op": op, "group": group, "reason": reason, "lease_s": int(lease_s or self.PLACEMENT_TRANSITION_LEASE_S), "by": by}, sort_keys=True)
        cmd = (f"mkdir -p {shlex.quote(root)} && flock -w 300 {shlex.quote(root + '/PLACED.lock')} python3 -c {shlex.quote(PLACEMENT_TRANSITION_PY)} "
               f"{shlex.quote(root)} {shlex.quote(intent)}")
        rc, out, err = self.remotes[cluster].sh(cmd, timeout=400, allow_fail=True)
        if rc != 0 or not (out or "").strip():
            raise Refused(f"placement-transition {op} {group}: the marker transaction on {cluster} failed (rc {rc}): {(err or out or '').strip()[-400:]}; nothing written", 3)
        try:
            res = json.loads(out.strip().splitlines()[-1])
        except ValueError:
            raise Refused(f"placement-transition {op} {group}: the marker transaction on {cluster} answered unreadably: {out.strip()[-300:]}", 3)
        result, doc, prev = res.get("result"), res.get("doc") or {}, res.get("prev")
        if result in ("opened", "refreshed"):
            log(f"placement-transition {group}: {root}/PLACED.transition {result} on {cluster} — PLACED version {doc.get('from_version')} is in transition for "
                f"{doc.get('lease_s')} s (since {doc.get('utc')}): no member of the group leaves until a placement commits or the lease lapses — {reason}"
                + (f" (an earlier marker since {prev.get('utc')} stood: {prev.get('reason')!r})" if isinstance(prev, dict) and prev.get("utc") else ""))
        elif result == "closed":
            log(f"placement-transition {group}: {root}/PLACED.transition closed on {cluster} (it stood since {(prev or {}).get('utc')}: {(prev or {}).get('reason')!r}) — "
                f"the members decide on PLACED and their bindings alone again")
        elif result == "absent":
            log(f"placement-transition {group}: no PLACED.transition on {cluster} — nothing to close")
        else:
            raise Refused(f"placement-transition {op} {group}: the marker transaction on {cluster} answered {result!r}", 3)
        return res

    def placement_changes(self, group: str, targets: List[str]) -> Tuple[List[str], List[str], Optional[str], Optional[dict]]:
        """READ-ONLY: (units, dropped, unaccounted, record) — how the group's current bindings and pool members differ from the attackers/PLACED
        record the cluster holds, each difference ACCOUNTED FOR by the ledger, named for the supersession's reason: `units`, the units a
        --resubmit pass re-minted SINCE the placement (the member whose job is in pool_members but not in the record's members, every target
        whose binding is not the record's (unit, job) pair — each by the attempt that minted it; the unit's name is the attempt's), and
        `dropped`, every member the record names that LEFT the pool with no replacement — its torn_down is in the ledger (pool_members excludes
        exactly those) and its unit was not re-minted — as `<unit> (job <id>: self-exited | dead on sacct <state> | torn down by its STOP)`
        (a group of two members, one self-exited beside the live one, is skipped by the walk and its teardown recorded, so the member list
        changes with NO new job; naming re-mints alone derived nothing for it, 06 passed no --supersede, and the fence refused the changed
        roster on every pass). A dropped member whose unit WAS re-minted is covered by `re-minted <unit>` alone (the
        unit's old job leaving and its new one arriving are one change). The baseline is the RECORD (placed_record — the one the placement
        transaction compares against), never this invocation's own appends, so a pass whose mark-placed failed after the re-mint names the
        same changes when it runs again (a line-count watermark named `none` on the retry and superseded the placement with that as its
        provenance). A difference the ledger cannot account for — a target the record names that the roster no longer
        carries (a plan change or a move, never a re-mint), a bound job or a member no attempt records, a member the record names the
        ledger never knew — comes back as `unaccounted`, and the caller passes no --supersede: the immutable-placement fence decides. No
        record on the cluster -> ([], [], None, None): a first placement, nothing to supersede. Nothing is written."""
        rows: Dict[str, str] = {}
        for lw in targets:
            b = ST.binding_state(self.subs.path, lw)
            if b.get("state") in ("submitted", "registered") and b.get("job_id"):
                rows[lw] = str(b["job_id"])      # an unbound target is mark-placed's own refusal ("is unbound"); it is no change of the roster
        label = self.pool_label(group)
        if not label:
            return [], [], None, None
        prev = self.placed_record(label, group)
        if not prev:
            return [], [], None, None
        version = int(prev.get("version") or 1)
        prev_rows = {str(t.get("logical_work_id")): str(t.get("job_id")) for t in (prev.get("targets") or []) if isinstance(t, dict) and t.get("job_id")}
        prev_members = [str(m) for m in (prev.get("members") or [])]
        st = self.ledger.state()
        group_units = {lw: s for lw, s in st.items() if f"/{group}/" in lw}
        # the unit's NAME is on the raw attempt record (`unit`, e.g. multiturn.attacker1 — the name 06's plan and the reason speak in); the
        # state view keeps an attempt's cluster / job / name only. HOW a member left is on its torn_down record (self_exit / dead + state /
        # neither = its STOP was written); the state view keeps the job alone.
        unit_by_job: Dict[Tuple[str, str], str] = {}; left_how: Dict[str, str] = {}
        for rec in self.ledger.records():
            if rec.get("event") == "attempt" and rec.get("job_id") is not None:
                unit_by_job[(str(rec.get("logical_work_id")), str(rec.get("job_id")))] = str(rec.get("unit") or rec.get("logical_work_id"))
            elif rec.get("event") == "torn_down" and rec.get("job_id") is not None and str(rec.get("logical_work_id")) in group_units:
                how = "self-exited" if rec.get("self_exit") else (f"dead on sacct {str(rec.get('state') or '').replace(',', '/')}".rstrip() if rec.get("dead") else "torn down by its STOP")
                left_how.setdefault(str(rec.get("job_id")), how)
        units: List[str] = []; dropped: List[str] = []; unaccounted: List[str] = []
        members = self.pool_members(group)
        for m in members:
            if m in prev_members:
                continue
            hit = next((unit_by_job[(lw, m)] for lw in group_units if (lw, m) in unit_by_job), None)
            if hit is None:
                unaccounted.append(f"member job {m} is recorded by no ledger attempt")
            elif hit not in units:
                units.append(hit)
        known = {str(m.get("job_id")) for s in group_units.values() for m in (s.get("pool_jobs") or [])}
        for m in prev_members:
            if m in members:
                continue
            if m not in known:
                unaccounted.append(f"member job {m} named by version {version} is unknown to the ledger"); continue
            # in the ledger's pool_jobs and out of pool_members: torn down (pool_members excludes exactly those) — accounted for by that trace
            hit = next((unit_by_job[(lw, m)] for lw in group_units if (lw, m) in unit_by_job), None)
            if hit is None:
                unaccounted.append(f"member job {m} named by version {version} left the pool but no ledger attempt names its unit"); continue
            if hit in units:
                continue       # the unit was re-minted above: `re-minted <unit>` is the whole account of its old job leaving
            dropped.append(f"{hit} (job {m}: {left_how.get(m) or 'torn down'})")
        for lw, job in rows.items():
            if prev_rows.get(lw) == job:
                continue
            hit = unit_by_job.get((lw, job))
            if hit is None:
                unaccounted.append(f"target {lw} is bound to job {job}, which no ledger attempt records")
            elif hit not in units:
                units.append(hit)
        for lw, job in prev_rows.items():
            if lw not in targets:
                unaccounted.append(f"target {lw} (job {job}) named by version {version} is not in this roster — a plan change or a move, not a re-mint")
        return units, dropped, ("; ".join(unaccounted) or None), prev

    def group_targets(self, group: str) -> List[Tuple[str, dict]]:
        """(lwid, current binding) for every TARGET unit of `group` the ledger knows (an attempt whose spec.group is the group, kind target)
        that is bound `submitted` or `registered` right now."""
        lwids = set()
        for lwid, st in self.ledger.state().items():
            for a in st["attempts"].values():
                spec = a.get("spec") or {}
                if spec.get("group") == group and str(a.get("kind") or spec.get("kind") or "target") == "target":
                    lwids.add(lwid)
        out = []
        for lwid in sorted(lwids):
            b = ST.binding_state(self.subs.path, lwid)
            if b.get("state") in ("submitted", "registered") and b.get("job_id"):
                out.append((lwid, b))
        return out

    def placement_record(self, group: str) -> Tuple[str, str, Optional[dict]]:
        """READ-ONLY: IS the group placed? -> (verdict, detail, record) — the authoritative answer a caller reads before it records a pass as
        placed, never 06's output (06 notes a mark-placed refusal other than NO LIVE MEMBER —
        the placement transaction failed on the cluster, an ssh outage, a PLACED.lock it could not take, the immutable-placement fence, rc 3 —
        and goes on to write WAVE.<view>.json and exit 0, so its exit code does not say whether attackers/PLACED was committed). The group as
        the LEDGER knows it (group_targets: every target bound submitted/registered now; pool_members: every recorded member not torn down)
        against the attackers/PLACED record on the group's cluster (placed_record), the record's differences accounted for exactly as
        placement-changes accounts for them. `placed`: a record whose version names every bound target at its bound job with a recorded member
        assigned, every recorded member, and no change it predates. `empty`: the ledger binds no target and records no member for the group —
        nothing to place (06's own "no target of the group went out this call"). `none`: targets or members recorded, no record on the cluster
        — mark-placed never committed, or never ran. `stale`: a record that predates a re-mint or a member that left, a target it does not
        name or assigns to no recorded member, or a difference the ledger cannot account for. Unreachable propagates: the caller decides.
        Nothing is written."""
        targets = self.group_targets(group)
        members = self.pool_members(group)
        if not targets and not members:
            return "empty", "the ledger binds no target and records no pool member for the group — nothing to place", None
        label = self.pool_label(group) or (targets[0][1].get("cluster") if targets else None)
        if not label:
            return "none", f"{len(targets)} bound target(s) and no pool member recorded for the group — no cluster to read attackers/PLACED on", None
        prev = self.placed_record(label, group)
        if not prev:
            return "none", f"no attackers/PLACED on {label} ({len(targets)} bound target(s), {len(members)} recorded member(s)): the placement was never committed", None
        version = int(prev.get("version") or 1)
        units, dropped, unaccounted, _ = self.placement_changes(group, [lw for lw, _ in targets])
        diffs: List[str] = []
        if units:
            diffs.append(f"re-minted since version {version}: {','.join(units)}")
        if dropped:
            diffs.append(f"dropped since version {version}: {','.join(dropped)}")
        if unaccounted:
            diffs.append(unaccounted)
        rows = {str(t.get("logical_work_id")): t for t in (prev.get("targets") or []) if isinstance(t, dict)}
        for lw, b in targets:
            row = rows.get(lw)
            if row is None:
                diffs.append(f"target {lw} (job {b.get('job_id')}) is not named by version {version}")
            elif str(row.get("job_id")) == str(b.get("job_id")) and str(row.get("assigned_member_job_id") or "") not in members:
                diffs.append(f"target {lw} is assigned member {row.get('assigned_member_job_id')!r}, not a recorded live member of the pool ({members})")
        if diffs:
            return "stale", f"attackers/PLACED version {version} on {label} is not the group's current placement: " + "; ".join(diffs), prev
        return ("placed", f"attackers/PLACED version {version} on {label}: {len(targets)} bound target(s) each at its bound job and assigned a recorded member; "
                f"members {members}", prev)

    def wait_pool(self, group: str, poll_s: float = 60.0, max_hours: float = 48.0) -> int:
        """The group-level backstop, in the repo (31b): poll SACCT — never squeue absence — every `poll_s` (≤ 60 s) for the group's TARGET
        jobs and, the poll after the LAST one is terminal, `teardown_pool`; a target sacct does not know yet, or a cluster that does not
        answer, counts as live. The idle window is bounded by one poll (2.7–3.4 min of idle members were measured behind a 180 s
        squeue-absence waiter, 2026-10-01; the members' own lifecycle, attacker_pool_lifecycle.sh, closes them first when it can).
        The teardown and the ratio's retirements are each checked against attackers/
        PLACED.transition UNDER the group's lock at the destructive write (teardown_pool / retire_member) — a fresh marker HOLDS them, the
        waiter keeps polling, and the next poll reads the roster the pass re-placed; a transition nobody commits lapses (the lease) and the
        teardown proceeds; at the bound the waiter refuses by name instead of stopping a group whose placement is still changing.
        The teardown and every retirement also carry the PLACED record THIS poll read (its version and
        target pairs); a placement that committed after that read — the whole pass landing between the poll's read and its write, its marker
        already cleared by the commit — answers `stale` under the lock with nothing written, this poll's ratio is skipped (its counts are the
        old placement's), and the next poll reads the current record and the ledger's new bindings."""
        poll_s = min(float(poll_s), 60.0)
        t0 = time.time(); said_incomplete = False; said_ratio_waits = False; last = None
        while True:
            targets = self.group_targets(group)
            if not targets:
                raise Refused(f"wait-pool {group}: the ledger binds no target of the group — nothing to wait for", 2)
            label = self.pool_label(group) or targets[0][1].get("cluster")
            # The roster the backstop waits for is the COMPLETE placement (attackers/PLACED, the funnel's own
            # mark-placed) UNION the ledger's bound targets — never the currently bound entries alone: a partial group (a --units subset, a
            # refused submission) must not stop the pool when its subset ends. No PLACED -> no teardown, however idle the members.
            try:
                placed = self.placed_record(label, group)
            except Unreachable:
                placed = None
            roster = {(lwid, str(b["job_id"]), str(b["cluster"])) for lwid, b in targets}
            for t in (placed or {}).get("targets") or []:
                if t.get("job_id"):
                    roster.add((str(t.get("logical_work_id")), str(t["job_id"]), str(t.get("cluster") or label)))
            live = []
            for lwid, job_id, cl in sorted(roster):
                try:
                    tasks = self.remotes[cl].sacct_tasks(job_id)
                except Unreachable as exc:
                    live.append((lwid, job_id, f"sacct unreachable ({exc})")); continue
                if not tasks or any(not t.get("terminal") for t in tasks):
                    live.append((lwid, job_id, ",".join(sorted({str(t.get("state")) for t in tasks})) or "not in sacct yet"))
            stale_poll = False
            if not live and placed is None:
                if not said_incomplete:
                    log(f"wait-pool {group}: every bound target terminal but the placement is NOT complete (no attackers/PLACED on {label}) — not tearing down; "
                        f"mark-placed marks the group once its last target is submitted")
                    said_incomplete = True
            elif not live:
                log(f"wait-pool {group}: placement complete ({len(placed.get('targets') or [])} placed target(s)) and every target terminal on sacct ({len(roster)} target(s)) — tearing the pool down")
                res = self.teardown_pool(group, placed=placed)
                if res.get("result") == "stopped":
                    return 0
                last = res
                if res.get("result") == "held":
                    # placement-fix round 4: a --resubmit pass opened the marker and is re-minting the group's member and targets; its old targets are
                    # all terminal but the roster this poll read predates the pass — the next PLACED (and the ledger's new bindings) names live ones
                    log(f"wait-pool {group}: teardown HELD this poll — the group's placement is in transition; polling on (the next PLACED re-places the group; the marker's lease is the bound)")
                else:
                    # placement-fix round 5: the record this poll judged terminal is not the record any more — a pass committed a supersession (and
                    # cleared its marker) after this poll read PLACED, so its new targets are not in this poll's roster; nothing was written
                    stale_poll = True
                    log(f"wait-pool {group}: teardown STALE this poll — attackers/PLACED moved from version {res.get('expected_version')} to {res.get('current_version')} after this poll "
                        f"read it (a pass re-placed the group); nothing written; polling on (the next poll reads the current placement and the ledger's new bindings)")
            # 31d: the pool holds no more than ONE member per three LIVE targets (stage.attacker_pool_size); the surplus — the members with
            # NO live consumer, highest job id first — are marked RETIRING, once each. Not before the placement is complete (31d-fix):
            # a partial group's finished subset must not collapse the pool before the remaining targets are submitted
            if placed is None:
                if not said_ratio_waits:
                    log(f"wait-pool {group}: ratio enforcement waits for the placement (no attackers/PLACED on {label})"); said_ratio_waits = True
            elif stale_poll:
                log(f"wait-pool {group}: ratio check skipped this poll (the placement this poll read is superseded — its live-target count is stale)")
            else:
                try:
                    self.enforce_pool_ratio(group, n_live_targets=len(live), placed=placed)
                except Unreachable as exc:
                    log(f"wait-pool {group}: ratio check skipped this poll (cluster unreachable: {exc})")
            if time.time() - t0 > max_hours * 3600:
                if live:
                    raise Refused(f"wait-pool {group}: {len(live)} target(s) still live after {max_hours} h ({'; '.join(f'{j} {s}' for _, j, s in live)}) — not torn down", 3)
                if placed is not None and last is not None and last.get("result") == "held":
                    raise Refused(f"wait-pool {group}: every target terminal but the group's placement stayed in TRANSITION within {max_hours} h "
                                  f"({self._transition_phrase(self.pool_root(last.get('cluster') or label, group), last.get('marker') or {}, last.get('left_s'))}) — not torn down; "
                                  f"complete the placement (mark-placed), let the marker's lease lapse, or release it with `placement-transition close --group {group} --cluster {last.get('cluster') or label}`", 3)
                if placed is not None and last is not None and last.get("result") == "stale":
                    raise Refused(f"wait-pool {group}: every target this waiter knew terminal but attackers/PLACED moved under its teardown (version {last.get('expected_version')} -> "
                                  f"{last.get('current_version')}) at {max_hours} h — not torn down; the group was re-placed: run wait-pool again against the current placement", 3)
                raise Refused(f"wait-pool {group}: every bound target terminal but the placement never completed within {max_hours} h (no attackers/PLACED on {label}) — not torn down; run mark-placed", 3)
            time.sleep(poll_s)


#: the placement-transition marker as the destructive transactions read it (placement-fix round 4), prepended to RETIRE_TX_PY and
#: TEARDOWN_TX_PY: transition_state(root) -> (fresh | absent | lapsed | unreadable, doc, seconds left). The marker's epoch is THIS cluster's
#: clock (the open transaction wrote it here), compared with the same clock — never the grading host's. Read under the same `flock PLACED.lock` the
#: open takes, so an open lands wholly before or wholly after the write it fences.
TRANSITION_READ_PY = r"""
import json as _json, os as _os, time as _time
def transition_state(root):
    p = _os.path.join(root, "PLACED.transition")
    if not _os.path.isfile(p):
        return "absent", None, None
    try:
        d = _json.load(open(p)); left = float(d["epoch"]) + float(d["lease_s"]) - _time.time()
    except Exception as e:
        return "unreadable", {"error": str(e)}, None
    return ("fresh" if left > 0 else "lapsed"), d, left
"""

#: the placement's IDENTITY as the destructive transactions compare it, prepended after
#: TRANSITION_READ_PY to RETIRE_TX_PY and TEARDOWN_TX_PY: placed_identity(root) -> (version, {(logical_work_id, job_id)}) of attackers/PLACED —
#: (0, empty) when absent or unreadable, exactly what the grading host's placed_record answers None for — and stale_answer(result, expect, root) -> the
#: {result, expected_version, current_version, added, removed} line when the caller's `expect` (the record it judged terminal, or counted a
#: member's consumers on: Coordinator._placed_expect) is not the record under the lock now, else None. A supersession bumps the version under
#: this same lock, so a commit that landed between the caller's read and its write is seen here and the write is refused with nothing written.
PLACED_IDENTITY_PY = r"""
def placed_identity(root):
    p = _os.path.join(root, "PLACED")
    if not _os.path.isfile(p):
        return 0, set()
    try:
        d = _json.load(open(p))
    except Exception:
        return 0, set()
    if not isinstance(d, dict):
        return 0, set()
    return int(d.get("version") or 1), {(str(t.get("logical_work_id")), str(t.get("job_id"))) for t in (d.get("targets") or []) if isinstance(t, dict) and t.get("job_id")}
def stale_answer(result, expect, root):
    cur_v, cur_pairs = placed_identity(root)
    exp_v, exp_pairs = int(expect["version"]), {tuple(x) for x in expect["targets"]}
    if cur_v == exp_v and cur_pairs == exp_pairs:
        return None
    return {"result": result, "expected_version": exp_v, "current_version": cur_v, "added": sorted(cur_pairs - exp_pairs), "removed": sorted(exp_pairs - cur_pairs)}
"""

#: the retirement transaction (31d-fix): runs ON the group's cluster under the SAME lock as the placement — argv[1] = the pool root, argv[2] =
#: the intent JSON {group, job_id, why, n_live_targets, expect, utc}; prints ONE JSON line {result: retired | unchanged | refused_last |
#: held_transition | stale_placement, ...}. The serving set is read under the lock (records in discovery minus RETIRE/SELF_EXIT tombstones), so
#: two retirers cannot both withdraw the last. A FRESH PLACED.transition marker (read under the same lock) holds the retirement with nothing
#: written (placement-fix round 4): the pass that opened it is re-placing the group, and the next PLACED may assign this member the targets it
#: re-mints. An attackers/PLACED that is not the record `expect` names (round 5) answers stale_placement with nothing written: the caller's
#: consumer count was the old placement's.
RETIRE_TX_PY = TRANSITION_READ_PY + PLACED_IDENTITY_PY + r"""
import calendar, json, os, sys, time
root, intent = sys.argv[1], json.loads(sys.argv[2])
job, n_live = str(intent["job_id"]), int(intent.get("n_live_targets") or 0)
live_jobs = {str(j) for j in (intent.get("live_jobs") or [])}
tr_state, tr_doc, tr_left = transition_state(root)
marker_ignored = tr_state if tr_state in ("lapsed", "unreadable") else None
def stems(suffix):
    return {f[:-len(suffix)] for f in os.listdir(root) if f.endswith(suffix)} if os.path.isdir(root) else set()
def fresh_ok(j):
    # 31d-fix-3 (a HIGH review finding on 609b0ea20): a surviving server PROVES its health with the lease it wrote itself — fresh (within its own lease_s) and ok;
    # none, stale, unreadable or ok=false is no server (fail closed): a vLLM pid can live while its endpoint answers nothing
    try:
        with open(os.path.join(root, j + ".health")) as fh:
            h = json.load(fh)
    except (OSError, ValueError):
        return False, "no health lease"
    if not h.get("ok"):
        return False, "health probe failed (lease ok=false)"
    try:
        age = time.time() - calendar.timegm(time.strptime(str(h.get("utc") or ""), "%Y-%m-%dT%H:%M:%SZ")); lease = float(h.get("lease_s") or 0)
    except (ValueError, TypeError):
        return False, "health lease unreadable"
    if age > lease:
        return False, f"health lease stale ({age:.0f}s old, lease {lease:.0f}s)"
    return True, "fresh"
retired, exited, unhealthy = stems(".RETIRE"), stems(".SELF_EXIT"), stems(".UNHEALTHY")
ready = stems(".json") - {"PLACED"}                      # a record written once vLLM answered its health check: a usable URL
pending = stems(".pending") - ready                       # booting: no server yet (31d-fix-2)
dead = sorted(j for j in ready if j not in live_jobs)      # a record whose job is no longer live on sacct: no server either
health = {j: fresh_ok(j) for j in ready if j in live_jobs}
unfit = sorted(f"{j}: {health[j][1]}" for j in health if not health[j][0] and j not in retired and j not in exited)
serving = {j for j in health if health[j][0]} - retired - exited - unhealthy
if job in retired:
    print(json.dumps({"result": "unchanged", "serving": sorted(serving)})); sys.exit(0)
if tr_state == "fresh":
    # placement-fix round 4: the group's placement is in transition — nothing written; the caller holds and decides again on the next poll
    print(json.dumps({"result": "held_transition", "marker": tr_doc, "left_s": tr_left, "serving": sorted(serving)})); sys.exit(0)
stale = stale_answer("stale_placement", intent["expect"], root)
if stale is not None:
    # placement-fix round 5: a placement committed after the caller read PLACED — the member's consumers were counted on the old one; nothing written
    print(json.dumps(dict(stale, marker=tr_doc, left_s=tr_left, serving=sorted(serving)))); sys.exit(0)
survivors = sorted(serving - {job})
if n_live > 0 and not survivors:
    print(json.dumps({"result": "refused_last", "serving": sorted(serving), "pending": sorted(pending - retired - exited), "dead": dead, "unfit": unfit,
                      "unhealthy": sorted(unhealthy), "n_live_targets": n_live, "marker_ignored": marker_ignored, "marker": tr_doc, "left_s": tr_left})); sys.exit(0)
path = os.path.join(root, job + ".RETIRE"); tmp = path + ".tmp." + str(os.getpid())
with open(tmp, "w") as fh:
    json.dump({"group": intent["group"], "why": intent["why"], "utc": intent["utc"], "survivors": survivors}, fh, sort_keys=True); fh.write("\n")
os.replace(tmp, path)
print(json.dumps({"result": "retired", "survivors": survivors, "marker_ignored": marker_ignored, "marker": tr_doc, "left_s": tr_left}))
"""

#: the teardown transaction (placement-fix round 4): runs ON the group's cluster under `flock <pool_root>/PLACED.lock` — argv[1] = the pool
#: root, argv[2] = the intent JSON {group, jobs, expect, utc}; prints ONE JSON line {result: held | stale | stopped, ...}. A FRESH
#: PLACED.transition marker (read under the lock) holds the WHOLE teardown — {result: held, marker, left_s}, nothing written. An attackers/PLACED
#: that is not the record `expect` names — the one the caller judged terminal (round 5: a supersession committed, and cleared its marker, after
#: that read) — answers {result: stale, expected_version, current_version, added, removed}, nothing written. Otherwise each job gets attackers/
#: <job>.STOP {group, utc} (tmp + os.replace, per MEMBER, never attackers/STOP) unless its <job>.SELF_EXIT is there (31b: the member closed
#: itself; recorded, not stopped again); a lapsed or unreadable marker is ignored and named in `marker_ignored`.
TEARDOWN_TX_PY = TRANSITION_READ_PY + PLACED_IDENTITY_PY + r"""
import json, os, sys
root, intent = sys.argv[1], json.loads(sys.argv[2])
tr_state, tr_doc, tr_left = transition_state(root)
if tr_state == "fresh":
    print(json.dumps({"result": "held", "marker": tr_doc, "left_s": tr_left})); sys.exit(0)
stale = stale_answer("stale", intent["expect"], root)
if stale is not None:
    # placement-fix round 5: the record under the lock is not the one the caller judged terminal — the current placement may assign targets to
    # the very members this would stop; nothing written, the caller reads the current record and decides again
    print(json.dumps(dict(stale, marker=tr_doc, left_s=tr_left))); sys.exit(0)
stopped, exited = [], []
for job in [str(j) for j in intent["jobs"]]:
    if os.path.isfile(os.path.join(root, job + ".SELF_EXIT")):
        exited.append(job); continue
    path = os.path.join(root, job + ".STOP"); tmp = path + ".tmp." + str(os.getpid())
    with open(tmp, "w") as fh:
        json.dump({"group": intent["group"], "utc": intent["utc"]}, fh, sort_keys=True); fh.write("\n")
    os.replace(tmp, path); stopped.append(job)
print(json.dumps({"result": "stopped", "stopped": stopped, "self_exited": exited, "marker_ignored": tr_state if tr_state in ("lapsed", "unreadable") else None,
                  "marker": tr_doc, "left_s": tr_left}))
"""

#: the placement transaction (31b-fix-3): runs ON the group's cluster under `flock <pool_root>/PLACED.lock` — argv[1] = the pool root,
#: argv[2] = the intent JSON {group, rows, members, supersede, reason, utc}; prints ONE JSON line {result, doc, ...}. The compare/version/
#: archive/write never leaves the lock, so two invocations cannot both publish a first record or the same version. A COMMIT (placed,
#: unchanged, superseded) also clears the group's PLACED.transition marker (below) in the same locked step —
#: `transition_cleared` says whether one stood; a refusal leaves it, since the sanctioned supersession is that pass's continuation.
#: The PLACEMENT half of the interleaving: a member whose attackers/<job>.SELF_EXIT stands under
#: this lock is EXCLUDED from `members` before any decision — the caller's walk read its job as active on Slurm (a job in its exit path is
#: still RUNNING/COMPLETING there) and handed it as live, but the member has left, or is leaving, and its EXIT trap removes the record a
#: target needs to bind it. The exclusion rides every answer and record as `excluded_self_exited` ONLY when it happened, so a pass with no
#: SELF_EXIT answers and writes byte for byte as before; with no member left the transaction answers `no_live_member` (nothing written, the
#: marker left as every refusal leaves it — its lease is the bound — and named with what is left of it) so the pass re-mints instead of
#: committing an assignment to a server that is gone. The member-side half (serialising the exit decision with the marker open) is NOT
#: here: a member that decided to leave but has not yet written its tombstone at commit time is still named, by design (the named residual
#: tests/test_attacker_pool_lifecycle.py pins stays loud).
PLACEMENT_TX_PY = TRANSITION_READ_PY + r"""
import json, os, sys
root, intent = sys.argv[1], json.loads(sys.argv[2])
path = os.path.join(root, "PLACED")
prev = None
if os.path.isfile(path):
    try:
        prev = json.load(open(path))
    except Exception:
        prev = None
rows, handed = intent["rows"], [str(m) for m in intent["members"]]
# round 6: the SELF_EXIT stems of the pool root, read UNDER THIS LOCK (the same flock the retirement transaction reads its serving set under,
# never the caller's view) — a member named there is dropped from the members this transaction assigns, whatever Slurm says of its job
exited = {f[:-len(".SELF_EXIT")] for f in os.listdir(root) if f.endswith(".SELF_EXIT")} if os.path.isdir(root) else set()
excluded = [m for m in handed if m in exited]
members = [m for m in handed if m not in exited]
def excl():
    # the exclusion is named on every answer and record it happened on, and NOWHERE when it did not (byte-identical records and answers)
    return {"excluded_self_exited": excluded} if excluded else {}
def write(p, doc):
    tmp = p + ".tmp." + str(os.getpid())
    with open(tmp, "w") as fh:
        json.dump(doc, fh, sort_keys=True); fh.write("\n")
    os.replace(tmp, p)
def clear_transition():
    # the placement committed (or stands as recorded): the pass's in-transition marker is moot — under this same lock, so no member reads a
    # committed record beside a marker that says it is still changing
    p = os.path.join(root, "PLACED.transition")
    if not os.path.isfile(p):
        return False
    try:
        os.remove(p)
    except FileNotFoundError:
        return False
    return True
def archived():
    # 31b-fix-5: every earlier version this cluster archived (PLACED.v<n>), oldest first — the controller reconciles each one's ledger event
    out = []
    for name in os.listdir(root):
        if name.startswith("PLACED.v"):
            try:
                d = json.load(open(os.path.join(root, name)))
            except Exception:
                continue
            if isinstance(d, dict):
                out.append(d)
    return sorted(out, key=lambda d: int(d.get("version") or 0))
if not members:
    # round 6: every member the caller handed has left — nothing to assign a target to. Nothing written: no record, no archive, and the
    # marker exactly as every refusal leaves it (the pass that opened it re-mints on its re-run and its mark-placed then commits); its
    # lease bounds it, named here with what is left of it
    tr_state, tr_doc, tr_left = transition_state(root)
    print(json.dumps({"result": "no_live_member", "excluded_self_exited": excluded, "handed": handed, "doc": prev if isinstance(prev, dict) else None,
                      "prev_version": int(prev.get("version") or 1) if isinstance(prev, dict) else 0, "versions": archived(),
                      "marker_state": tr_state, "marker": tr_doc, "left_s": tr_left})); sys.exit(0)
if not isinstance(prev, dict):
    for i, r in enumerate(rows):
        r["assigned_member_job_id"] = members[i % len(members)]
    doc = {"group": intent["group"], "targets": rows, "members": members, "n_targets": len(rows), "version": 1, "utc": intent["utc"], **excl()}
    write(path, doc); print(json.dumps({"result": "placed", "doc": doc, "transition_cleared": clear_transition(), **excl()})); sys.exit(0)
prev_pairs = {(str(t.get("logical_work_id")), str(t.get("job_id"))): str(t.get("assigned_member_job_id") or "") for t in (prev.get("targets") or []) if t.get("job_id")}
prev_members = [str(m) for m in (prev.get("members") or [])]
new_pairs = {(str(r["logical_work_id"]), str(r["job_id"])) for r in rows}
prev_version = int(prev.get("version") or 1)
# `members` is the LIVE list (round 6): a record that names an excluded member differs from it, so it is never answered `unchanged` and
# left assigning that member its targets — it is refused, or superseded onto the members that remain
if set(prev_pairs) == new_pairs and prev_members == members:
    print(json.dumps({"result": "unchanged", "doc": prev, "prev_version": prev_version, "versions": archived(), "transition_cleared": clear_transition(), **excl()})); sys.exit(0)
added = sorted(new_pairs - set(prev_pairs)); removed = sorted(set(prev_pairs) - new_pairs)
if not intent.get("supersede"):
    print(json.dumps({"result": "refused", "doc": prev, "prev_version": prev_version, "prev_members": prev_members, "versions": archived(),
                      "added": [list(x) for x in added], "removed": [list(x) for x in removed], **excl()})); sys.exit(0)
if not intent.get("reason"):
    print(json.dumps({"result": "needs_reason", "doc": prev, "prev_version": prev_version, "versions": archived(), **excl()})); sys.exit(0)
kept = {k: v for k, v in prev_pairs.items() if k in new_pairs and v in members}
counter = 0; reassigned = []
for r in rows:
    k = (str(r["logical_work_id"]), str(r["job_id"]))
    if k in kept:
        r["assigned_member_job_id"] = kept[k]
    else:
        # the new pairs — and an unchanged pair whose member was EXCLUDED (round 6) — round-robin over the members that remain
        r["assigned_member_job_id"] = members[counter % len(members)]; counter += 1
        if k in prev_pairs and prev_pairs[k] in excluded:
            reassigned.append([k[0], k[1], prev_pairs[k], r["assigned_member_job_id"]])
write(os.path.join(root, "PLACED.v%d" % prev_version), prev)
changes = {"added": [list(x) for x in added], "removed": [list(x) for x in removed], "kept": len(kept), "members_before": prev_members}
if excluded:
    changes["excluded_self_exited"] = excluded; changes["reassigned"] = reassigned     # [unit, job, from (excluded), to]
doc = {"group": intent["group"], "targets": rows, "members": members, "n_targets": len(rows), "version": prev_version + 1,
       "supersedes": {"version": prev_version, "utc": prev.get("utc"), "n_targets": len(prev_pairs)}, "reason": intent["reason"], "changes": changes, "utc": intent["utc"], **excl()}
write(path, doc); print(json.dumps({"result": "superseded", "doc": doc, "prev_version": prev_version, "versions": archived(), "transition_cleared": clear_transition(), **excl()}))
"""

#: the placement-transition marker (placement-fix round 3): runs ON the group's cluster under the same
#: `flock <pool_root>/PLACED.lock` — argv[1] = the pool root, argv[2] = the intent JSON {op: open | close, group, reason, lease_s}; prints ONE
#: JSON line {result: opened | refreshed | closed | absent, doc, prev}. `open` writes attackers/PLACED.transition {group, from_version (the
#: PLACED version it will supersede; 0 without a record), reason, lease_s, epoch (the CLUSTER's clock — the member compares against its own
#: node's, never the grading host's), utc, by}; a marker that already stands (an earlier pass died mid-way) is REFRESHED, since this pass is its
#: continuation. The member's lifecycle holds every exit while the marker is fresh (epoch + lease_s ahead of now) and ignores a lapsed one —
#: the lease is the fence's BOUND: a pass that never re-places the group (nobody runs the sanctioned supersession) cannot pin a member for
#: ever. `close` removes it by hand (the operator decided not to re-place).
PLACEMENT_TRANSITION_PY = r"""
import json, os, sys, time
root, intent = sys.argv[1], json.loads(sys.argv[2])
path = os.path.join(root, "PLACED.transition")
prev = None
if os.path.isfile(path):
    try:
        prev = json.load(open(path))
    except Exception:
        prev = {"unreadable": True}
if intent["op"] == "close":
    if prev is None:
        print(json.dumps({"result": "absent", "prev": None})); sys.exit(0)
    os.remove(path); print(json.dumps({"result": "closed", "prev": prev})); sys.exit(0)
placed = None
pp = os.path.join(root, "PLACED")
if os.path.isfile(pp):
    try:
        placed = json.load(open(pp))
    except Exception:
        placed = None
now = time.time()
doc = {"group": intent["group"], "from_version": int(placed.get("version") or 1) if isinstance(placed, dict) else 0, "reason": intent["reason"],
       "lease_s": int(intent["lease_s"]), "epoch": int(now), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)), "by": intent.get("by")}
tmp = path + ".tmp." + str(os.getpid())
with open(tmp, "w") as fh:
    json.dump(doc, fh, sort_keys=True); fh.write("\n")
os.replace(tmp, path)
print(json.dumps({"result": "refreshed" if prev is not None else "opened", "doc": doc, "prev": prev}))
"""

# ------------------------------------------------------------------------------------------------------------ cli
def sibling_serving_jobs(subs_ledger: str, lwid: str) -> List[str]:
    """For a multi-turn TARGET unit `<model>/<view>/multiturn/<attack>/<group>`, the job ids of the SIBLING
    shard groups' LIVE bindings — the units under the same `<model>/<view>/multiturn/<attack>/` prefix with another group whose binding (the
    submissions ledger's bindings dir, ST.binding_state) is `submitted` or `registered`; a transition, an unbound or a torn sibling states
    nothing; sorted, de-duplicated. The registering job adds its own id (emit_cli register --serving-jobs); siblings unknown at this
    submission (the first group of a pass) are filed later by the SERVINGJOBS backfill pass. Never a store's routing tables."""
    parts = lwid.split("/")
    if len(parts) != 5 or parts[2] != "multiturn":
        return []
    prefix = "/".join(parts[:4]) + "/"
    d = ST.bindings_dir(subs_ledger)
    if not os.path.isdir(d):
        return []
    out = set()
    for fn in sorted(os.listdir(d)):
        if not (fn.endswith(".submitted.json") or fn.endswith(".registered.json")):
            continue
        try:
            rec = json.load(open(os.path.join(d, fn), encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        other = str(rec.get("logical_work_id") or "")
        if not other.startswith(prefix) or other == lwid:
            continue
        b = ST.binding_state(subs_ledger, other)
        if b.get("state") in ("submitted", "registered") and str(b.get("job_id") or "").isdigit():
            out.add(str(b["job_id"]))
    return sorted(out)


def _env_pairs(s: str) -> Dict[str, str]:
    out = {}
    for tok in shlex.split(s or ""):
        if "=" not in tok:
            raise Refused(f"--env wants K=V tokens, got {tok!r}")
        k, v = tok.split("=", 1); out[k] = v
    return out


def site_wandb_env(site: Site) -> Dict[str, str]:
    """The site's W&B destination as job environment, read as the stage reads it (Site.given): when the site names an entity,
    SISTEER_WANDB_ENTITY and SISTEER_WANDB_PROJECT (the project `$WANDB_PROJECT` in a committed config expands to); with no entity,
    nothing. The ONE place these keys are composed: Coordinator.job_env takes them for every funnel job, and `job-env --wandb`
    prints them alone for a fit job a launcher script submits outside the funnel (scripts/stage_llama_fits.sh, and
    scripts/two_si_overnight.py imports this function), so every fit job names the same destination."""
    entity = site.given("SISTEER_WANDB_ENTITY")
    if not entity:
        return {}
    return {"SISTEER_WANDB_ENTITY": entity, "SISTEER_WANDB_PROJECT": site.given("SISTEER_WANDB_PROJECT") or ""}


def print_job_env(config: str, label: Optional[str], *, export: bool = False, wandb_only: bool = False) -> int:
    """`job-env`: Coordinator.job_env for cluster `label`, printed as the sbatch line's own prefix (submit_one's composition) or, with `export`,
    as one `export K=V` line each (a job submitted by hand evals them before sbatch, whose implicit export carries them). With
    `wandb_only`, the W&B part of that environment alone (site_wandb_env, the same function job_env calls; no cluster needed).
    The site is read as a dry run reads it, so nothing is created; job_env reads the site alone, so no model is consulted."""
    site = Site(config, True)
    if wandb_only:
        env = site_wandb_env(site)
    elif label not in site.clusters:
        print(f"REFUSED: job-env: {label!r} is not one of the clusters {site.config_path} names ({' '.join(site.clusters)})",
              file=sys.stderr)
        return 3
    else:
        env = Coordinator(site, "").job_env(label)
    print("\n".join(f"export {k}={shlex.quote(v)}" for k, v in env.items()) if export
          else " ".join(f"{k}={shlex.quote(v)}" for k, v in env.items()))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", required=True); ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--model", default=None, help="the model key (every command but job-env needs it)")
    ap.add_argument("--overrides", default=None)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("submit")
    p.add_argument("--unit", required=True); p.add_argument("--lwid", required=True); p.add_argument("--script", required=True)
    p.add_argument("--name", required=True); p.add_argument("--args", default=""); p.add_argument("--clusters", default="")
    p.add_argument("--twin", action="store_true"); p.add_argument("--env", default=""); p.add_argument("--time", default=None)
    p.add_argument("--array", default=None); p.add_argument("--kind", default="target", choices=["target", "attacker", "fit", "other"])
    p.add_argument("--flags", default="", help="extra sbatch flags, appended verbatim (D96: the fit path's --gres/--cpus-per-task/--mem for a width-N fit)")
    p.add_argument("--registers", default=None, choices=[REGISTERS_REGISTRY, REGISTERS_NONE],
                   help="the unit's registration contract (default: none for kind other, registry otherwise); `none` makes --resubmit's proof sacct-terminal alone")
    p.add_argument("--group", default=None); p.add_argument("--wave-sha", default=None)
    p.add_argument("--depends-on", default=None, metavar="LWID",
                   help="the logical_work_id this unit's job runs AFTER: Slurm --dependency=afterok on its bound job while live/pending, none once complete; placed on that unit's cluster (D186)")
    p.add_argument("--foreground", action="store_true", help="run the election inline (tests / small waves)")
    p.add_argument("--walk", action="store_true", help="a family pass (06 --resubmit): a registering unit whose every task COMPLETED is SKIPPED (its results stand) instead of refused")
    p.add_argument("--resume-from-job", default=None, help="D590: RESUME a REGISTERED unit's drivers under the SAME canonical runs — a new attempt whose job reuses the named "
                                                            "(terminal) job's registration identity (target_attacks.sbatch copies pending/<job>.json to its own; no new runs) and resumes the "
                                                            "stores (an errored goal is not covered); the registered binding stands")
    p.add_argument("--job-env", default="", help="with --resume-from-job: JOB-side K=V pairs OUTSIDE the workload spec (the attacker pin F22_ATTACKER_JOBS=<job> when the group "
                                                 "still carries dead members' records, a wait bound) — recorded in the attempt as job_env, never part of spec_identity")
    p.add_argument("--resubmit", action="store_true", help="re-mint an identity for a unit whose bound job is provably TERMINAL and never registered (the four-way fenced proof); refused otherwise")
    p.add_argument("--poll-s", type=float, default=30.0); p.add_argument("--max-wait-s", type=float, default=6 * 3600)
    p = sub.add_parser("elect"); p.add_argument("--lwid", required=True); p.add_argument("--poll-s", type=float, default=30.0); p.add_argument("--max-wait-s", type=float, default=6 * 3600)
    sub.add_parser("reconcile")
    p = sub.add_parser("cancel-attempt"); p.add_argument("--attempt-id", required=True)
    p = sub.add_parser("teardown-pool"); p.add_argument("--group", required=True)
    p = sub.add_parser("mark-placed"); p.add_argument("--group", required=True); p.add_argument("--targets", required=True, help="comma-separated logical work ids of the group's targets")
    p.add_argument("--supersede", action="store_true", help="a placement is immutable: re-place a CHANGED roster as a versioned transition (archives PLACED.v<n>; unchanged pairs keep their member); needs --reason")
    p.add_argument("--reason", default=None, help="the provenance of a --supersede (why the roster changed: the resubmitted target, its ruling)")
    p = sub.add_parser("placement-transition", help="the group's placement is IN TRANSITION (placement-fix round 3): `open` writes attackers/PLACED.transition on the "
                       "group's cluster BEFORE a pass submits replacements, so a re-minted member never leaves on the placement that predates it; mark-placed "
                       "clears it when a placement commits; `close` clears it by hand; a member ignores it once its lease lapsed (the bound)")
    p.add_argument("op", choices=("open", "close")); p.add_argument("--group", required=True); p.add_argument("--cluster", required=True, help="the group's cluster label")
    p.add_argument("--reason", default=None, help="open: which pass is re-placing the group (e.g. `06 --resubmit pass on <sha>`)")
    p.add_argument("--lease-s", type=int, default=None, help=f"open: how long the marker fences the members (default {Coordinator.PLACEMENT_TRANSITION_LEASE_S} s)")
    p = sub.add_parser("placement-changes", help="READ-ONLY: how the group's bindings and pool members differ from attackers/PLACED on the cluster, each "
                       "difference accounted for by the ledger — as ONE line on stdout, the change clause of 06's --supersede reason: `re-minted <units>` "
                       "(the units a --resubmit pass re-minted since the placement), `dropped <unit> (job <id>: self-exited | dead on sacct <state> | torn "
                       "down by its STOP)` (a member the record names that left with no replacement — its torn_down is in the ledger), or both joined by "
                       "`; `; EMPTY when nothing changed, when no record exists, or when a difference no ledger attempt or teardown accounts for (named on "
                       "stderr: the immutable-placement fence then decides). Nothing written")
    p.add_argument("--group", required=True); p.add_argument("--targets", required=True, help="comma-separated logical work ids of the group's targets (mark-placed's roster)")
    p = sub.add_parser("placement-record", help="READ-ONLY: is the group PLACED? attackers/PLACED on the group's cluster against the LEDGER's view of the group — every "
                       "target bound now at its bound job with a recorded member assigned, every recorded member named, no change the record predates "
                       "(placement-changes' accounting). ONE tab-separated line on stdout, `placement-record <group> <verdict> <detail>`: `placed` and `empty` "
                       "(the ledger binds no target and records no member for the group: nothing to place) exit 0; `none` (no record on the cluster: the "
                       "placement never committed) and `stale` exit 3, as does a cluster that does not answer. A caller reads it before it records a pass as "
                       "placed — 06 notes a mark-placed refusal other than NO LIVE MEMBER and exits 0 (stage 09's multiturn legs). Nothing written")
    p.add_argument("--group", required=True)
    p = sub.add_parser("reclose", help="re-run a finished eval job's store closer under the driver venv (closer-fix-2): the job left with exit 75 and a recipe beside its pending file; "
                                       "submits scripts/sdfbase/full22/reclose_stores.sbatch at the recipe's sha (CPU-only; cap 3 per job; the same refusal twice stops it)")
    p.add_argument("--job", required=True, help="the eval job whose closer refused (its id on the cluster)")
    p.add_argument("--tree", default=None, metavar="SHA", help="reclose-at-tree (D1120 (3a)): run the CLOSER from this landed tree (a closer fix that post-dates the job) — the job's own tree stays the recipe's identity; the receipt, token and ledger event record both shas")
    p.add_argument("--closer-arg", action="append", default=[], metavar="ARG", help="an extra token for registry_close_stores.py (repeatable; a value that starts with -- needs the = form: --closer-arg=--ruled-out-shards --closer-arg=5,6,7,8 --closer-arg=--ruled-out-objects --closer-arg=none --closer-arg=--ruling-evidence-uri --closer-arg=<uri>, D1168) — recorded on the token, the event and the receipt")
    p.add_argument("--retry-why", default="", help="closer-fix-6: after a reclose refused in PREFLIGHT and the input it saw is unchanged, record WHY one more reclose is deliberate "
                                                   "(what was repaired); the receipt stays — the ledger event and the recipe's `retries` carry the note")
    p = sub.add_parser("retire-member", help="mark one pool member RETIRING (attackers/<job>.RETIRE + ledger retiring); it drains and self-exits, its drivers re-bind (31d)")
    p.add_argument("--group", required=True); p.add_argument("--job", required=True); p.add_argument("--why", default="retired by the operator")
    p = sub.add_parser("reconcile-token", help="the funnel's own transition-token reconciliation for ONE unit (Coordinator.reconcile_transition): a token "
                       "left by a submit that died after its sbatch is resolved against Slurm — the launched job identified positively by its nonce (squeue %k), "
                       "ADOPTED into the ledger, the submissions, the `submitted` binding and the watcher; none -> cleared; ambiguity -> refused by name with "
                       "the token standing. Prints `token\tnone|adopted|cleared` (exit 0) or `token\trefused\t<reason>` (exit 3); then the `binding` line. "
                       "The review on 50841cf14 (D753): a stage reconciles a lost launch THROUGH this, never by closing its own receipt on an operator's word")
    p.add_argument("--lwid", required=True)
    p = sub.add_parser("binding", help="READ-ONLY: the unit's current binding — prints `binding\t<state>\t<cluster>\t<job id>\t<token yes|no>` "
                       "(state unbound|submitted|registered|transition; `token yes` = the funnel's own re-run transition token is in flight: a launch it has "
                       "not recorded yet, reconciled by the funnel before anything else). A stage's reconciliation reads it as the durable submission "
                       "evidence (the review on e11e596e1: absence from squeue alone never proves that nothing was submitted)")
    p.add_argument("--lwid", required=True)
    p = sub.add_parser("prove-resubmit", help="READ-ONLY: the fenced --resubmit proof over a bound unit (Coordinator.resubmit_decision, the same "
                       "decision `submit --resubmit` takes) — exit 0 and print `bound_job\t<cluster>\t<job id>` only when a re-mint is AUTHORIZED; "
                       "nothing minted, nothing moved (the review on 84d2732b1: a stage that archives a failed design before this proof acts before evidence)")
    p.add_argument("--unit", required=True); p.add_argument("--lwid", required=True)
    p.add_argument("--kind", default="fit", choices=["target", "attacker", "fit", "other"]); p.add_argument("--registers", default=None, choices=[REGISTERS_REGISTRY, REGISTERS_NONE])
    p = sub.add_parser("wait-pool", help="the group-level backstop: poll sacct (≤ 60 s) for the group's targets, enforce one member per three live targets, and teardown-pool the poll after the last is terminal (31b/31d)")
    p.add_argument("--group", required=True); p.add_argument("--poll-s", type=float, default=60.0); p.add_argument("--max-hours", type=float, default=48.0)
    p = sub.add_parser("job-env", help="READ-ONLY: the job environment every sbatch line for --cluster carries (Coordinator.job_env, the ONE "
                       "list; lib/slurm.sh slurm_job_env prints it): `K=V …` shell-quoted, or with --export one `export K=V` line each for a "
                       "job submitted by hand. Nothing is created; --model is not needed")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--cluster", help="a site label the config names")
    g.add_argument("--wandb", action="store_true", help="only the site's W&B destination (site_wandb_env; empty when the site names no "
                   "entity), for a fit job a launcher script submits outside the funnel")
    p.add_argument("--export", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "job-env":
        return print_job_env(a.config, a.cluster, export=a.export, wandb_only=a.wandb)
    if not a.model:
        ap.error(f"--model is required for {a.cmd}")
    site = Site(a.config, a.dry_run)
    co = Coordinator(site, a.model, a.overrides)
    if a.cmd in ("submit", "reconcile-token") and not site.dry and getattr(a, "lwid", None):
        # ONE per-unit lock serializes reservation -> sbatch -> ledger/binding -> clear with any reconciliation of
        # the same unit — held for the whole invocation (released at exit), bounded wait, refused by name when another funnel process holds it
        global _UNIT_LOCK_FD
        _UNIT_LOCK_FD = acquire_unit_lock(co.subs.path, a.lwid)
    if a.cmd in ("binding", "reconcile-token"):
        if a.cmd == "reconcile-token":
            tok = ST.read_transition(co.subs.path, a.lwid)
            if not tok:
                print("token\tnone", flush=True)
            elif site.dry:
                print(f"DRY-RUN: reconcile-token {a.lwid}: a transition token stands (attempt {tok.get('intended_attempt')} since {tok.get('utc')}); nothing reconciled", flush=True)
                print("token\tstanding", flush=True)
            else:
                try:
                    outcome = co.reconcile_transition(a.lwid, tok)
                except Refused as exc:
                    print(f"token\trefused\t{str(exc).replace(chr(9), ' ').replace(chr(10), ' ')}", flush=True)
                    return 3
                how_line = ""
                if outcome == "adopted":
                    b1 = ST.binding_state(co.subs.path, a.lwid); st1 = co.ledger.state().get(a.lwid) or {"attempts": {}}
                    att1 = next((x for x in st1["attempts"].values() if str(x.get("job_id")) == str(b1.get("job_id"))), {})
                    # the RECONCILIATION's method: `nonce` (read back from the live job) or `ledger` (the launch record under this reservation,
                    # job no longer live — a half-recorded launch); the attempt's own identified_by (launch | nonce) is on the binding line
                    how_line = f"\t{getattr(co, '_reconciled_method', None) or att1.get('identified_by') or 'unknown'}\t{att1.get('nonce') or ''}"
                print(f"token\t{outcome}{how_line}", flush=True)
        b = ST.binding_state(co.subs.path, a.lwid)
        tok = b.get("transition") if b.get("state") == "transition" else None
        print("binding\t%s\t%s\t%s\t%s" % (b.get("state"), b.get("cluster") or "-", b.get("job_id") or "none", "yes" if tok else "no"), flush=True)
        st_u = co.ledger.state().get(a.lwid) or {"attempts": {}}
        jobs = sorted({str(x.get("job_id")) for x in (st_u.get("attempts") or {}).values() if x.get("job_id")} | ({str(b["job_id"])} if b.get("job_id") else set()))
        print("attempts\t%s" % ",".join(jobs), flush=True)       # EVERY job id the durable ledger knows for the unit: all excluded by the reconcile
        batt = next(((aid_, x) for aid_, x in (st_u.get("attempts") or {}).items() if b.get("job_id") and str(x.get("job_id")) == str(b.get("job_id"))), None)
        if batt:
            print(f"bound_attempt\t{batt[0]}\t{batt[1].get('identified_by') or 'unknown'}\t{batt[1].get('nonce') or ''}", flush=True)   # How the bound job's id was obtained
        if tok:
            print(f"transition token: attempt {tok.get('intended_attempt')} since {tok.get('utc')} (the funnel reconciles it by job name + submit time)", flush=True)
        return 0
    if a.cmd == "prove-resubmit":
        b = ST.binding_state(co.subs.path, a.lwid)
        if b["state"] not in ("submitted", "registered"):
            print(f"prove-resubmit: {a.lwid} has no bound job (binding state {b['state']}) — nothing to relaunch, nothing authorized", flush=True)
            return 3
        try:
            action, why, evidence = co.resubmit_decision(a.lwid, a.unit, b, a.kind, a.registers)
        except Refused as exc:
            print(f"prove-resubmit: REFUSED {a.lwid}: {exc}", flush=True)
            return 3
        for ev in evidence.get("teardowns") or []:
            # READ-ONLY (placement-fix round 1): the torn_down traces the decision derived are NAMED here and written by nobody —
            # `submit --resubmit` records them (record_teardowns); this proof leaves pool_members exactly as it found them
            print(f"prove-resubmit: would record torn_down ({'self_exit' if ev.get('self_exit') else 'dead'}) for job {ev.get('job_id')} on {ev.get('cluster')}"
                  f"{' (' + str(ev.get('state')) + ')' if ev.get('state') else ''} — READ-ONLY: nothing written; `submit --resubmit` records it", flush=True)
        if action != "resubmit":
            print(f"prove-resubmit: NOT authorized {a.lwid}: {action}: {why}", flush=True)
            return 3
        tasks_txt = ",".join(f"{t['task'] if t['task'] is not None else t['raw']}:{t['state']}" for t in evidence.get("tasks", []))
        print(f"prove-resubmit: AUTHORIZED {a.lwid}: bound job {b.get('job_id')} on {b.get('cluster')} [{tasks_txt}] — {why}", flush=True)
        print(f"bound_job\t{b.get('cluster')}\t{b.get('job_id')}", flush=True)
        return 0
    if a.cmd == "submit":
        sha = a.wave_sha or subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=os.path.dirname(HERE)).stdout.strip()
        if not re.fullmatch(r"[0-9a-f]{40}", sha or ""):
            raise Refused(f"--wave-sha must be 40 hex (got {sha!r})")
        labels = [c for c in a.clusters.split(",") if c] or ([site.primary] + [l for l in site.clusters if l != site.primary])
        for lab in labels:
            if lab not in site.clusters:
                raise Refused(f"cluster {lab!r} is not configured ({' '.join(site.clusters)})")
        args = shlex.split(a.args)
        if "--export" in (a.flags or ""):
            raise Refused("--flags may not carry --export= (cluster hard rule 1: it crashes slurmd); pass job-side values as --env K=V")
        if "--mem=0" in (a.flags or "").replace(" ", ""):
            raise Refused("--flags may not carry --mem=0 (cluster hard rule 2: it reserves the whole node)")
        if a.script.startswith("scripts/sdfbase/full22/static_padding/") and any("," in x for x in args[3:4]):
            raise Refused("the padded static scripts take ONE padding level per job (five submissions per view) — a comma list of pads would register only the first level (close round E P1)")
        if a.twin and len(labels) < 2:
            raise Refused("--twin needs two clusters (SISTEER_CLUSTERS or --clusters a,b)")
        if a.twin and not co.hook_present(labels[0], sha, a.script):
            raise Refused(f"--twin refused: the mirrored {a.script} at {sha[:12]} does not carry the in-body grant hook ({HOOK_MARKER}); a twin without it would register before any grant (PR7b lands the hook)")
        if a.twin and a.kind == "fit" and not site.dry:
            # every candidate cluster's fit venv is ONE certified stack, checked BEFORE the primary's identity is minted
            ids = {lab: co.fit_env_identity(lab) for lab in labels}
            for lab in labels[1:]:
                why = co.fit_env_mismatch(ids[labels[0]], ids[lab])
                if why:
                    raise Refused(f"--twin refused for {a.lwid}: the fit venv on {labels[0]} ({ids[labels[0]]['venv']}) and on {lab} ({ids[lab]['venv']}) "
                                  f"are not one stack — {why}; a twin would fit the unit on two stacks (D70: sdfbase-fit-tf515 on both, identical "
                                  f"requirements + freeze shas); nothing submitted")
        if a.depends_on and not (site.dry and ST.binding_state(co.subs.path, a.depends_on)["state"] == "unbound"):
            # a dependent unit is placed on the cluster its dependency is bound on (the store it escalates lives on that shared root);
            # a dry run against a world where the dependency is not submitted yet only says what it would do (submit_one prints it)
            dep = co.dependency_target(a.lwid, a.depends_on)
            if dep.get("cluster") not in labels:
                raise Refused(f"{a.lwid} runs after {a.depends_on}, which is bound on {dep.get('cluster')} — not one of --clusters {','.join(labels)}", 3)
            if a.twin:
                raise Refused(f"--twin refused for {a.lwid}: it runs after {a.depends_on} on {dep.get('cluster')}; a dependent unit has one cluster", 2)
            labels = [dep["cluster"]]
        spec = dict(unit=a.unit, lwid=a.lwid, script=a.script, name=a.name, args=args, env=_env_pairs(a.env), sha=sha,
                    time_limit=a.time, array=a.array, kind=a.kind, group=a.group, clusters=labels, flags=a.flags or "",
                    registers=registration_contract(a.kind, a.registers, script=a.script, env=_env_pairs(a.env)), depends_on=a.depends_on)
        if not site.dry:
            pv = co.verify_unit_contract(a.lwid, sha, spec)    # the registry contract AND the workload spec, on every submit
            if pv is not None:
                pv["fix_commit"] = co.fix_commit_of(sha) if pv["kind"] == "advance" else None
                spec["provenance"] = pv
        if a.job_env and not a.resume_from_job:
            raise Refused("--job-env is accepted with --resume-from-job only: a workload knob rides --env (and the spec identity); only a resume carries job-side facts outside it", 2)
        if a.resume_from_job:
            # RESUME a REGISTERED unit's drivers under the SAME canonical runs. A new attempt (its own job, sha advance, watcher) whose job
            # reuses the original job's registration identity — F22_RESUME_FROM_JOB in the JOB env, not the workload spec: target_attacks.sbatch
            # copies pending/<from>.json to its own before f22_registry_register runs, so no new runs are minted — and resumes the stores
            # (their path is job-independent; an errored goal is not covered). The registered binding is NOT retired or replaced.
            if a.twin or a.resubmit or a.depends_on or a.array:
                raise Refused("--resume-from-job takes one cluster and no --twin / --resubmit / --depends-on / --array", 2)
            # 31e-fix-2: the duplicate-resume check, the reservation and the launch serialize per unit under a local lock —
            # two concurrent callers cannot both see "no prior resume" and launch two jobs on the same stores and run ids
            import fcntl
            _slug = a.lwid.replace("/", "__"); _bdir = os.path.join(os.path.dirname(os.path.abspath(co.subs.path)), "bindings"); os.makedirs(_bdir, exist_ok=True)
            with open(os.path.join(_bdir, f"{_slug}.resume.lock"), "a+") as _lk:
                fcntl.flock(_lk.fileno(), fcntl.LOCK_EX)
                if not site.dry:
                    # a reservation left by a resume that died after its sbatch: reconciled against Slurm by job name FIRST (adopted as the resume
                    # attempt — no binding written — or cleared when no job of the name exists), so a retry never duplicates a live resume
                    tok0 = ST.read_transition(co.subs.path, a.lwid)
                    if tok0 is not None:
                        if not tok0.get("resume"):
                            raise Refused(f"--resume-from-job {a.resume_from_job}: {a.lwid} carries a transition token of a plain re-run (attempt {tok0.get('intended_attempt')}) — "
                                          f"`submit --resubmit` reconciles it first; nothing submitted", 3)
                        outcome = co.reconcile_transition(a.lwid, tok0)
                        log(f"resume {a.lwid}: an earlier resume's reservation (attempt {tok0.get('intended_attempt')}) was {outcome} — reconciled against Slurm by job name before any new launch")
                b = ST.binding_state(co.subs.path, a.lwid)
                if b.get("state") != "registered" or str(b.get("job_id")) != str(a.resume_from_job):
                    raise Refused(f"--resume-from-job {a.resume_from_job}: {a.lwid} is {b.get('state')} on job {b.get('job_id')} — a resume follows the unit's REGISTERED job only; nothing submitted", 3)
                if b.get("cluster") not in labels:
                    raise Refused(f"--resume-from-job {a.resume_from_job}: the registered job ran on {b.get('cluster')}, not one of --clusters {','.join(labels)} — the stores live there; nothing submitted", 3)
                if not site.dry:
                    tasks = co.remotes[b["cluster"]].sacct_tasks(str(a.resume_from_job))
                    if not tasks or any(not t.get("terminal") for t in tasks):
                        raise Refused(f"--resume-from-job {a.resume_from_job}: not terminal on sacct ({sorted({str(t.get('state')) for t in tasks}) or 'not in sacct'}) — a resume runs only after "
                                      f"the job it resumes ended; nothing submitted", 3)
                # 31e-fix: the registered binding stays on J, so the guard above alone let a second invocation submit
                # a third job beside a live resume (two jobs driving the same stores and run ids) or after a completed one (re-running a closed
                # run). The ledger's resume attempts are the durable in-flight record: every prior resume of the unit is checked on sacct —
                # live or unknown -> refused; COMPLETED -> refused (the stores were closed by it); terminal-not-completed -> a crashed resume,
                # reconciled by recording it in this attempt.
                prior_resumes = []
                for rec_ in co.ledger.records():
                    if rec_.get("event") == "attempt" and rec_.get("logical_work_id") == a.lwid and isinstance(rec_.get("spec"), dict) and rec_["spec"].get("resume_of"):
                        prior_resumes.append(rec_)
                crashed = []
                for rec_ in sorted(prior_resumes, key=lambda r_: str(r_.get("utc") or "")):
                    if site.dry:
                        continue
                    try:
                        tasks = co.remotes[rec_["cluster"]].sacct_tasks(str(rec_["job_id"]))
                    except Unreachable as exc:
                        raise Refused(f"--resume-from-job {a.resume_from_job}: the prior resume attempt {rec_.get('attempt_id')} job {rec_.get('job_id')} cannot be read on sacct ({exc}) — "
                                      f"not submitting a second resume blind; nothing submitted", 3)
                    states = sorted({str(t.get("state")) for t in tasks})
                    if not tasks or any(not t.get("terminal") for t in tasks):
                        raise Refused(f"--resume-from-job {a.resume_from_job}: a resume of {a.lwid} is already IN FLIGHT — attempt {rec_.get('attempt_id')} job {rec_.get('job_id')} "
                                      f"({','.join(states) or 'not in sacct yet'}); two jobs must never drive the same stores and run ids; nothing submitted", 3)
                    if any(str(t.get("state")).startswith("COMPLETED") for t in tasks):
                        raise Refused(f"--resume-from-job {a.resume_from_job}: the resume attempt {rec_.get('attempt_id')} job {rec_.get('job_id')} COMPLETED — the unit's stores were closed "
                                      f"by it; a further resume would re-run a closed run (read the stores / 06 --gate first); nothing submitted", 3)
                    crashed.append({"attempt_id": rec_.get("attempt_id"), "job_id": str(rec_.get("job_id")), "states": states})
                resume_of = {"job_id": str(a.resume_from_job), "attempt_id": b.get("attempt_id"), "cluster": b.get("cluster"), "run_ids": list(b.get("run_ids") or []),
                             "prior_resumes": crashed}
                extra_env = _env_pairs(a.job_env)
                bad = sorted(k for k in extra_env if k not in RESUME_JOB_ENV_ALLOWED)
                if bad:
                    raise Refused(f"--job-env {bad}: not on the resume allowlist {list(RESUME_JOB_ENV_ALLOWED)} — a job-side fact is the attacker pin or its wait bound; a knob that "
                                  f"alters the drivers' work (a seed, a decode policy, a serving path, a registry route) is the frozen workload's and rides --env, where the freeze "
                                  f"guard sees it; nothing submitted", 2)
                collide = sorted(k for k in extra_env if k in (spec.get("env") or {}))
                if collide:
                    raise Refused(f"--job-env {collide}: the frozen workload spec already names these keys — a resume cannot override them; nothing submitted", 2)
                job_env = {"F22_RESUME_FROM_JOB": str(a.resume_from_job)}
                job_env.update(extra_env)
                # the RESERVATION (31e-fix-2): a transition token naming the intended attempt and the job name, written BEFORE
                # sbatch under the unit's lock — a caller interrupted between sbatch and the ledger append leaves it for the next caller, which
                # reconciles it against Slurm by job name (adopt the launched job as the resume attempt, or clear) before anything new launches
                label_r = b["cluster"]; intended = attempt_id(a.lwid, label_r, co.n_attempts(a.lwid, label_r) + 1)
                nonce_r = co.nonce() if not site.dry else None
                if not site.dry:
                    # 31e-fix-5: the snapshot of every same-name job BEFORE the token (none can be ours) and the attempt's nonce, both in the token
                    snap_r = co.reservation_snapshot(label_r, a.name)
                    ST.write_transition(co.subs.path, {"logical_work_id": a.lwid, "unit": a.unit, "cluster": label_r, "job_name": a.name, "intended_attempt": intended, "utc": utc_now(),
                                                       "old_binding": {k: b.get(k) for k in ("state", "job_id", "attempt_id", "run_ids", "array")},
                                                       "spec": dict(spec, resume_of=resume_of, job_env=job_env), "retired": None,
                                                       "spool": co.job_env(label_r).get("SISTEER_REGISTRY_SPOOL"), "why": f"resume of job {a.resume_from_job} (D590)", "resume": True,
                                                       "nonce": nonce_r, "snapshot_job_ids": snap_r})
                _crash("resume_after_reservation")
                aid, job = co.submit_one(label_r, twin=False, force=True, bind=False, job_env=job_env, resume_of=resume_of, comment=nonce_r,
                                         **{k: spec.get(k) for k in SPEC_KEYS})
                if not site.dry:
                    ST.clear_transition(co.subs.path, a.lwid)
                log(f"resume {a.lwid}: attempt {aid} job {job} on {b['cluster']} resumes job {a.resume_from_job} under its {len(resume_of['run_ids'])} registered run(s); the registered binding stands"
                    + (f"; {len(crashed)} crashed resume(s) before it reconciled: {[c['job_id'] for c in crashed]}" if crashed else ""))
                return 0
        dead_winner = None      # (attempt id, its record, why) when --resubmit replaces a COMMITTED election winner
        replaced = None         # (old binding, why, sacct states) when --resubmit replaces a bound job: a CROSS-CLUSTER replacement writes a placement receipt (24d)
        moved_from = None       # move-knob-6: the receipt's origin, decided as a PREFLIGHT right after the fenced resubmit decision (below)
        if not site.dry:
            tok = ST.read_transition(co.subs.path, a.lwid)
            if tok:
                # a re-run was interrupted mid-flight: reconcile the token against Slurm BEFORE anything else (hole 3)
                co.reconcile_transition(a.lwid, tok)
        transition_open = False
        if a.resubmit and not site.dry:
            # --resubmit: the fenced proof over EVERY task of the bound job (Coordinator.resubmit_decision); a `registered` binding
            # (06 --gate) is decided the same way and RETIRED before the new attempt is bound
            b = ST.binding_state(co.subs.path, a.lwid)
            if b["state"] in ("submitted", "registered"):
                action, why, evidence = co.resubmit_decision(a.lwid, a.unit, b, a.kind, a.registers, walk=bool(getattr(a, "walk", False)))
                if action == "skip":
                    # the decision is READ-ONLY: the torn_down traces it derived — a self-exited member skipped beside a live
                    # one (the ledger's trace is what the next pass reads), an untraced dead predecessor of a live member —
                    # are recorded HERE, by the submit command, before the pass walks on
                    co.record_teardowns(evidence)
                    tail = "" if why.startswith("COMPLETED") else " — --resubmit re-runs only a terminal job"
                    print(f"skipped {b.get('cluster')} {b.get('job_id')} {a.unit} ({why}: attempt {b.get('attempt_id')}{tail})", flush=True)
                    return 0
                if action == "reclose":
                    # closer-fix-2: the bound job's stores are complete and its closer refused — the recovery is a reclose, submitted here (06's
                    # --resubmit walk reaches this line for every such unit), never a re-run of the eval; the binding stands
                    co.record_teardowns(evidence)
                    log(f"RECLOSE {a.lwid}: bound job {b.get('job_id')} on {b.get('cluster')} — {why}")
                    co.reclose(a.lwid, a.unit, b, evidence.get("recipe"))
                    return 0
                st_u = co.ledger.state().get(a.lwid) or {"attempts": {}, "winner": None}
                att = st_u["attempts"].get(b.get("attempt_id")) or {"cluster": b.get("cluster"), "job_id": b.get("job_id"), "job_name": b.get("sbatch_name")}
                tasks_txt = ",".join(f"{t['task'] if t['task'] is not None else t['raw']}:{t['state']}" for t in evidence.get("tasks", []))
                if spec.get("provenance") is not None:      # the death this attempt answers, on the unit's sha table
                    spec["provenance"]["answers"] = {"job_id": b.get("job_id"), "cluster": b.get("cluster"), "attempt_id": b.get("attempt_id"),
                                                     "state": ",".join(sorted({t["state"] for t in evidence.get("tasks", [])})), "why": why[:400]}
                # move-knob-6: the declared-move gate is a PREFLIGHT — the receipt's origin is decided HERE, after the
                # fenced decision and BEFORE any supersession, transition reservation or binding retirement, so a refusal (rc 3, nothing
                # submitted) leaves the old binding, its filed runs and the ledger exactly as they were; the value is retained for the
                # post-submit receipt. move-knob-5 decided it after those writes, and the refusal rolled none of them back.
                moved_from = placement_receipt_origin(co.state, a.lwid, (b, why, ",".join(sorted({t["state"] for t in evidence.get("tasks", [])}))),
                                                      labels[0], spec.get("group"))
                # the torn_down traces the READ-ONLY decision derived (the dead member being re-minted, its untraced
                # predecessors, a self-exited one with no live peer) are recorded HERE — after the preflight above, so a
                # refusal (rc 3, nothing submitted) leaves the ledger exactly as it was, and before the transition token,
                # so pool_members / live_members drop the dead job before the pass's mark-placed reads them
                co.record_teardowns(evidence)
                if action == "rerun":
                    log(f"RERUN {a.lwid}: bound job {b.get('job_id')} on {b.get('cluster')} [{tasks_txt}] {why} — a new attempt re-runs the SAME run identities "
                        "(deterministic run_id; the registry upserts)")
                    co.ledger.append({"event": "superseded_attempt", "logical_work_id": a.lwid, "attempt_id": b.get("attempt_id"), "job_id": b.get("job_id"),
                                      "cluster": b.get("cluster"), "state": ",".join(sorted({t["state"] for t in evidence.get("tasks", [])})),
                                      "binding_state": b["state"], "tasks": evidence.get("tasks"), "why": why,
                                      "run_ids": sorted({rid for e in evidence.get("evidence", []) for rid in e.get("run_ids", [])}),
                                      "resume_proof": evidence.get("resume_proof")})
                else:
                    log(f"RESUBMIT {a.lwid}: bound job {b.get('job_id')} on {b.get('cluster')} [{tasks_txt}] is terminal and never registered ({why}) — a new attempt")
                # THE TRANSITION TOKEN (hole 3): written BEFORE the old binding is retired or sbatch is called — (unit, old binding, intended
                # attempt, the full spec) — so a crash anywhere up to the new attempt's binding leaves a durable record the next
                # invocation reconciles by job name + submit time; binding_state reads `transition` meanwhile, so 06 --gate cannot bind the
                # superseded generation and a plain `submit` skips the unit.
                # The RESERVATION is taken on the DESTINATION cluster — `labels[0]`, the one `submit_one` launches on — never on the replaced
                # binding's (petri-completion): a CROSS-CLUSTER replacement (24d; the overflow proxy moved back to the
                # first pass's cluster) reserved on the OLD cluster recorded an intended attempt, a same-name snapshot and a spool that all named
                # the wrong cluster, so a crash after sbatch accepted was reconciled by searching the old cluster, found nothing there, cleared
                # the token as a non-launch, and the retry launched a SECOND job against the same stores and run ids. The old binding stays on
                # the token (with its cluster) as the replacement's origin, with the receipt facts the adoption writes (receipt_after_adoption).
                dest = labels[0]
                intended = attempt_id(a.lwid, dest, co.n_attempts(a.lwid, dest) + 1)
                snap = co.reservation_snapshot(dest, a.name)      # 31e-fix-5: before the token — none of these can be the job it launches
                tok = {"logical_work_id": a.lwid, "unit": a.unit, "cluster": dest, "job_name": a.name, "intended_attempt": intended, "utc": utc_now(),
                       "old_binding": {k: b.get(k) for k in ("state", "job_id", "attempt_id", "run_ids", "array", "cluster")}, "spec": spec, "retired": None,
                       "spool": (att.get("spool") if str(b.get("cluster") or "") == dest else None) or co.job_env(dest).get("SISTEER_REGISTRY_SPOOL"),
                       "why": why, "nonce": co.nonce(), "snapshot_job_ids": snap,
                       "moved_from": moved_from, "replaced_state": ",".join(sorted({t["state"] for t in evidence.get("tasks", [])}))}
                ST.write_transition(co.subs.path, tok); transition_open = True
                _crash("resubmit_after_token")
                if b["state"] == "registered":
                    retired = ST.retire_registered_binding(co.subs.path, a.lwid, b.get("attempt_id"))
                    tok["retired"] = retired; ST.write_transition(co.subs.path, tok)
                    log(f"retired the registered binding of {a.lwid} (attempt {b.get('attempt_id')}, runs {b.get('run_ids')}) -> {retired}; the new attempt binds `submitted` and 06 --gate re-binds it")
                _crash("resubmit_after_retire")
                if st_u.get("winner") == b.get("attempt_id"):
                    dead_winner = (b["attempt_id"], att, why)
                replaced = (b, why, ",".join(sorted({t["state"] for t in evidence.get("tasks", [])})))
        elif a.resubmit:
            # a dry run decides nothing (no sacct is read): say what the flag would do, so a stage's dry run shows --resubmit reached the funnel
            # for this unit (09 --step views --resubmit) and under which contract the proof would run
            b = ST.binding_state(co.subs.path, a.lwid)
            if b["state"] in ("submitted", "registered", "transition"):
                print(f"DRY-RUN: --resubmit {a.lwid}: bound {b['state']} to job {b.get('job_id')} on {b.get('cluster')} — the fenced proof over sacct would decide "
                      f"(skip while any task is live; re-mint a terminal job under registration contract {spec['registers']}"
                      + ("; a petri unit's past-running runs would be checked against petri_run.py legs-status over the stores they name" if resume_proof_for(a.script) else "")
                      + "); nothing decided in a dry run", flush=True)
            else:
                print(f"DRY-RUN: --resubmit {a.lwid}: unbound — nothing to re-mint, a first submission (registration contract {spec['registers']})", flush=True)
        if not transition_open and not site.dry and ST.binding_state(co.subs.path, a.lwid)["state"] == "unbound":
            # a FIRST submission (the unit unbound after the pre-check above) is reserved exactly like a re-run — the token (unit, intended attempt,
            # the full spec, a fresh nonce, the snapshot of same-name jobs that pre-exist it) is durable BEFORE sbatch, the nonce rides the sbatch
            # line (submit_one merges it into the fit path's sha comment), and a crash between sbatch's acceptance and the ledger append — the
            # window in which the job's id lived only in sbatch's stdout — is reconciled by the next invocation: the job identified by its nonce
            # read back is adopted; nothing is launched twice; a lone nonce-less same-name job is never adopted by elimination
            intended0 = attempt_id(a.lwid, labels[0], co.n_attempts(a.lwid, labels[0]) + 1)
            snap0 = co.reservation_snapshot(labels[0], a.name)
            tok = {"logical_work_id": a.lwid, "unit": a.unit, "cluster": labels[0], "job_name": a.name, "intended_attempt": intended0, "utc": utc_now(),
                   "old_binding": None, "spec": spec, "retired": None, "spool": co.job_env(labels[0]).get("SISTEER_REGISTRY_SPOOL"),
                   "why": "first submission: reserved with a nonce before its sbatch (D775)", "nonce": co.nonce(), "snapshot_job_ids": snap0, "first": True}
            ST.write_transition(co.subs.path, tok); transition_open = True
            _crash("submit_after_reservation")
        # move-knob-5 / move-knob-6: `moved_from` was decided right after the fenced resubmit decision (above), before any durable write; the
        # ledger's receipt write follows the submission
        new_aid, new_job = co.submit_one(labels[0], twin=a.twin, force=a.resubmit, comment=(tok.get("nonce") if transition_open else None), **spec)
        if transition_open:
            # clear ONLY the reservation this process wrote — its nonce must still be the one on disk (under the unit lock another writer
            # cannot interleave; this is the belt to that brace). A first submission's reservation leaves no trail (the ledger's attempt record
            # carries the nonce); a re-run's token keeps its `.done` trail as before
            _now = ST.read_transition(co.subs.path, a.lwid)
            if _now is None:
                log(f"{a.lwid}: the reservation this invocation wrote is already gone (nonce {tok.get('nonce')}) — nothing cleared")
            elif str(_now.get("nonce") or "") != str(tok.get("nonce") or ""):
                raise Refused(f"{a.lwid}: the reservation on disk carries nonce {_now.get('nonce')!r}, not this invocation's {tok.get('nonce')!r} — another writer's; NOT cleared "
                              f"(this attempt {new_aid} / job {new_job} is bound; reconcile the other reservation with `reconcile-token`)", 3)
            elif tok.get("first"):
                os.remove(ST.transition_path(co.subs.path, a.lwid))
            else:
                ST.clear_transition(co.subs.path, a.lwid)       # the new attempt is bound: the transition is complete
        if moved_from is not None and not site.dry:
            # the unit MOVED: a fenced, provenance-recorded re-run of a dead job on another cluster is the only sanctioned way a placement
            # changes, and it leaves a RECEIPT the placement manifest advances under — an unreceipted move stays a refusal by name in
            # `stage.py placement-manifest ensure|check`. The move is measured against the replaced attempt's cluster OR the manifest's
            # PIN — a group declared moved before anything ran on its origin and first placed on the destination by a plain submission is
            # pinned at the origin with its dead attempt already on the destination; its resubmit lands on the same cluster as that
            # attempt and still owes the pin its receipt (from_cluster = the pin; nothing ran there).
            ob = replaced[0]; via_pin = moved_from != ob.get("cluster")
            co.ledger.append(placement_receipt_record(a.lwid, moved_from, labels[0], spec.get("group"), ob, replaced[2], replaced[1], new_aid, new_job))
            where = (f"the pin; replaced job {ob.get('job_id')} on {ob.get('cluster')}" if via_pin else f"job {ob.get('job_id')}")
            log(f"PLACEMENT RECEIPT {a.lwid}: {moved_from} ({where}, {replaced[2]}) -> {labels[0]} (attempt {new_aid}, job {new_job}); "
                f"the placement manifest advances under it")
        if dead_winner is not None and not site.dry:
            # the bound job was the COMMITTED winner of an election: the manual recovery takes reconcile's retire -> commit -> grant
            # transition (one path), so the election below finds the NEW attempt as the winner instead of re-granting the dead one
            # and denying its replacement
            co.regrant_after_dead_winner(a.lwid, dead_winner[0], dead_winner[1], new_aid, dead_winner[2])
        if a.twin and not site.dry:
            if a.foreground:
                res = co.elect(a.lwid, a.poll_s, a.max_wait_s, twin_spec={**spec, "clusters": labels})
                print(f"election {a.lwid}: {res}")
                return 0 if res == "granted" else 3
            cmd = [sys.executable, os.path.abspath(__file__), "--config", a.config, "--model", a.model, "elect", "--lwid", a.lwid, "--poll-s", str(a.poll_s), "--max-wait-s", str(a.max_wait_s)]
            subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=open(os.path.join(co.state, "elect.log"), "a"), stderr=subprocess.STDOUT, start_new_session=True)
            print(f"election {a.lwid}: detached (log {co.state}/elect.log)")
        elif a.twin and site.dry:
            other = site.other(labels[0])
            print(f"DRY-RUN: election {a.lwid}: primary on {labels[0]}; twin to {other} after {site.threshold_s}s PENDING with < {site.max_jobs} units left; GRANT/DENY under {site.cluster_root(labels[0], a.model)}/attempts/")
        return 0
    if a.cmd == "elect":
        res = co.elect(a.lwid, a.poll_s, a.max_wait_s)
        print(f"election {a.lwid}: {res}")
        return 0 if res == "granted" else 3
    if a.cmd == "reconcile":
        return co.reconcile()
    if a.cmd == "cancel-attempt":
        co.cancel_attempt(a.attempt_id); return 0
    if a.cmd == "teardown-pool":
        # placement-fix round 5: the attackers/PLACED record THIS invocation decides on is read first and travels into the locked transaction,
        # which refuses (`stale`) when a placement commits in between — the operator reads the current one and decides again
        label = co.pool_label(a.group); placed = None
        if label and not site.dry:
            try:
                placed = co.placed_record(label, a.group)
            except Unreachable as exc:
                raise Refused(f"teardown-pool {a.group}: {label} did not answer while reading attackers/PLACED ({exc}) — nothing written; run this again", 3)
        res = co.teardown_pool(a.group, placed=placed)
        if res.get("result") == "held":
            # placement-fix round 4: an operator's teardown during a transition is refused by name — the sanctioned release is the placement's
            # commit, the lease, or `placement-transition close`; never a STOP written beside a marker that says the group is being re-placed
            cl = res.get("cluster") or label or ""
            raise Refused(f"teardown-pool {a.group}: the group's placement is in TRANSITION ({co._transition_phrase(co.pool_root(cl, a.group), res.get('marker') or {}, res.get('left_s'))}) "
                          f"— refused, nothing written: a pass is re-placing the group onto these members; complete the placement (mark-placed), let the lease lapse, "
                          f"or release it with `placement-transition close --group {a.group} --cluster {cl}` and run this again", 3)
        if res.get("result") == "stale":
            cl = res.get("cluster") or label or ""
            raise Refused(f"teardown-pool {a.group}: attackers/PLACED on {cl} moved under this teardown (version {res.get('expected_version')} when it was read, version "
                          f"{res.get('current_version')} under the lock; target pairs added {res.get('added')}, removed {res.get('removed')}) — refused, nothing written: a pass "
                          f"re-placed the group, and the current placement may assign targets to members this would have stopped; run this again against the current placement", 3)
        return 0
    if a.cmd == "mark-placed":
        if site.dry:
            print(f"DRY-RUN: mark-placed {a.group}: {len([t for t in a.targets.split(',') if t])} target(s) -> attackers/PLACED"); return 0
        co.mark_placed(a.group, [t for t in a.targets.split(",") if t], supersede=a.supersede, reason=a.reason); return 0
    if a.cmd == "placement-transition":
        if site.dry:
            print(f"DRY-RUN: placement-transition {a.op} {a.group}: attackers/PLACED.transition on {a.cluster} — nothing written"); return 0
        co.placement_transition(a.op, a.group, a.cluster, reason=a.reason, lease_s=a.lease_s, by=f"submit_twin.py placement-transition {a.op}"); return 0
    if a.cmd == "placement-changes":
        # stdout is the ANSWER alone (06 captures it into the reason); every note goes to stderr. A dry run reads no record and derives nothing.
        if site.dry:
            print(f"DRY-RUN: placement-changes {a.group}: no record read, nothing derived", file=sys.stderr, flush=True); print("", flush=True); return 0
        units, dropped, unaccounted, prev = co.placement_changes(a.group, [t for t in a.targets.split(",") if t])
        if unaccounted:
            print(f"placement-changes {a.group}: attackers/PLACED version {int((prev or {}).get('version') or 1)} differs from the roster in a way no ledger attempt "
                  f"or teardown accounts for ({unaccounted}) — no supersession derived; the immutable-placement fence decides (mark-placed --supersede --reason "
                  f"… by hand, with the provenance, is the one sanctioned transition)", file=sys.stderr, flush=True)
            print("", flush=True); return 0
        # the answer is the reason's CHANGE clause, composed here (never by 06): `re-minted <units>`, `dropped <member units>` (round 2: a member
        # that left beside a live one, nothing re-minted), or both joined by `; ` — 06 prefixes `06 --resubmit pass on <sha>: `
        clause = "; ".join(c for c in ((f"re-minted {','.join(units)}" if units else ""), (f"dropped {','.join(dropped)}" if dropped else "")) if c)
        if prev is None:
            print(f"placement-changes {a.group}: no attackers/PLACED record (or no pool recorded) — a first placement, nothing to supersede", file=sys.stderr, flush=True)
        elif not clause:
            print(f"placement-changes {a.group}: every bound target and member is the record's (version {int(prev.get('version') or 1)}) — unchanged", file=sys.stderr, flush=True)
        else:
            if units:
                print(f"placement-changes {a.group}: re-minted since version {int(prev.get('version') or 1)}: {','.join(units)}", file=sys.stderr, flush=True)
            if dropped:
                print(f"placement-changes {a.group}: dropped since version {int(prev.get('version') or 1)}: {','.join(dropped)}", file=sys.stderr, flush=True)
        print(clause, flush=True); return 0
    if a.cmd == "placement-record":
        if site.dry:
            print(f"DRY-RUN: placement-record {a.group}: attackers/PLACED on the group's cluster against the ledger's bound targets and recorded members — nothing read, nothing written"); return 0
        try:
            verdict, detail, _ = co.placement_record(a.group)
        except Unreachable as exc:
            raise Refused(f"placement-record {a.group}: the group's cluster did not answer while reading attackers/PLACED ({exc}) — not verified, nothing written; run this again", 3)
        print(f"placement-record\t{a.group}\t{verdict}\t{detail}", flush=True)
        if verdict in ("none", "stale"):
            raise Refused(f"placement-record {a.group}: {verdict.upper()} — {detail}; an unplaced group's targets bind round-robin instead of their recorded assignment "
                          f"and wait-pool refuses to tear the pool down without attackers/PLACED; the sanctioned remedy is the pass that places the group (06's mark-placed: "
                          f"the same 06 pass again, every unit already submitted skipped; on the immutable-placement fence the supersession 06 printed) — never a "
                          f"hand-edited PLACED, marker or ledger", 3)
        return 0
    if a.cmd == "reclose":
        hits = [(lwid, aid, att) for lwid, st in co.ledger.state().items() for aid, att in st["attempts"].items() if str(att.get("job_id")) == str(a.job)]
        if not hits:
            raise Refused(f"reclose: no attempt with job {a.job} in the ledger", 2)
        lwid, aid, att = hits[0]
        b = {"cluster": att["cluster"], "job_id": str(a.job), "attempt_id": aid}
        if site.dry:
            print(f"DRY-RUN: reclose: read {co.remotes[att['cluster']].shared}/registry/pending/{a.job}.closer_refused.json, submit {RECLOSE_SCRIPT} {a.job} <recipe sha> as f22-reclose-{a.job}"); return 0
        co.reclose(lwid, att.get("unit") or lwid.split("/", 2)[-1], b, co.closer_recipe(att["cluster"], str(a.job)), retry_why=str(a.retry_why or ""),
                   closer_sha=str(getattr(a, "tree", None) or ""), closer_args=list(getattr(a, "closer_arg", None) or []))
        return 0
    if a.cmd == "retire-member":
        hit = [(lwid, m) for lwid, st in co.ledger.state().items() if f"/{a.group}/" in lwid for m in st["pool_jobs"] if str(m["job_id"]) == str(a.job)]
        if not hit:
            raise Refused(f"retire-member {a.group}: no member with job {a.job} in the ledger", 2)
        if site.dry:
            print(f"DRY-RUN: retire-member {a.group}: attackers/{a.job}.RETIRE"); return 0
        # placement-fix round 5: the attackers/PLACED record this invocation decides on is read BEFORE the live-target count and travels into the
        # locked transaction, which refuses (`stale_placement`) when a placement commits in between
        try:
            placed = co.placed_record(hit[0][1]["cluster"], a.group)
        except Unreachable as exc:
            raise Refused(f"retire-member {a.group}: {hit[0][1]['cluster']} did not answer while reading attackers/PLACED ({exc}) — nothing written; run this again", 3)
        n_live = 0
        for lwid_t, bt in co.group_targets(a.group):
            try:
                tasks = co.remotes[bt["cluster"]].sacct_tasks(str(bt["job_id"]))
            except Unreachable:
                n_live += 1; continue
            if not tasks or any(not t.get("terminal") for t in tasks):
                n_live += 1
        res = co.retire_member(a.group, hit[0][0], str(a.job), hit[0][1]["cluster"], a.why, n_live_targets=n_live, placed=placed)
        if res == "refused_last":
            raise Refused(f"retire-member {a.group}: job {a.job} is the group's LAST serving member while {n_live} target(s) are live — refused (submit a replacement first)", 3)
        if res == "held_transition":
            raise Refused(f"retire-member {a.group}: the group's placement is in TRANSITION (attackers/PLACED.transition on {hit[0][1]['cluster']}, above) — refused, nothing written: "
                          f"the next PLACED may assign job {a.job} targets; complete the placement (mark-placed), let the lease lapse, or release it with "
                          f"`placement-transition close --group {a.group} --cluster {hit[0][1]['cluster']}` and run this again", 3)
        if res == "stale_placement":
            raise Refused(f"retire-member {a.group}: attackers/PLACED on {hit[0][1]['cluster']} moved under this retirement (above) — refused, nothing written: a pass re-placed "
                          f"the group, and the current placement may assign job {a.job} targets; run this again against the current placement", 3)
        return 0
    if a.cmd == "wait-pool":
        if site.dry:
            print(f"DRY-RUN: wait-pool {a.group}: poll sacct every {min(a.poll_s, 60.0):g}s for the group's targets, teardown-pool the poll after the last is terminal"); return 0
        return co.wait_pool(a.group, poll_s=a.poll_s, max_hours=a.max_hours)
    return 2


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Refused as exc:
        sys.exit(exc.code)
