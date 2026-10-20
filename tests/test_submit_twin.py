"""reproduce/lib/submit_twin.py — the twin-submit START-GRANT protocol under fake sbatch / squeue / sacct / scancel
(state per cluster under $FAKE_SLURM_DIR; CLUSTER_SSH=local:<dir> runs
each "remote" command on this host). Exactly ONE GRANT per logical unit ever; the loser gets DENY + an attempt-scoped
scancel; $WAVE_ROOT/STOP and attackers/STOP are never written; the winner is committed to the ledger BEFORE any remote
write; a crash at every injected point replays to the same end state; --twin is refused when the mirrored sbatch lacks the
hook marker; a registered-then-dead winner is ATTACHED, never re-granted; a re-grant needs the four-way fenced proof."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from front_door_testlib import LIB, job_ids, make_registry, set_job_state, write_fake_slurm, write_site  # noqa: E402

from sisteer.registry import synth  # noqa: E402

SHA = synth.code_sha("c")   # a derived 40-hex wave sha, never a placeholder: the real emitter's job-start registration refuses forty identical hex digits; "c" kept as the tag
SCRIPT = "scripts/sdfbase/full22/target_attacks.sbatch"


class World:
    def __init__(self, tmp: Path, hook: bool = True, config_extra: str = ""):
        self.tmp = tmp
        self.fake = {lab: tmp / f"fake_{lab}" for lab in ("cluster_a", "cluster_b")}
        for d in self.fake.values():
            d.mkdir()
        self.bin = write_fake_slurm(tmp / "bin")
        self.site = write_site(tmp, ssh={lab: f"local:{d}" for lab, d in self.fake.items()}, extra_config=config_extra)
        self.shared = {lab: tmp / f"shared_{lab}" for lab in self.fake}
        for lab, sh in self.shared.items():        # the mirrored checkout at the wave sha, with (or without) the hook marker
            co = sh / "sisteer-eval-repo" / SHA / "scripts/sdfbase/full22"; co.mkdir(parents=True)
            (co / "target_attacks.sbatch").write_text("#!/bin/bash\n" + ("f22_attempt_wait   # sisteer-attempt-wait-hook\n" if hook else "") + "f22_init\n")
        make_registry(tmp / "registry")
        self.state = tmp / "state" / "m"

    #: the site keys the operator's shell of this world carries NONE of, so the world's site file decides them (integration round 2, review
    #: r3: the funnel reads every site key the way a stage does — the given file over the environment — so a key the file does not name is
    #: the SHELL's; the suite's session fixture puts a hermetic SISTEER_REGISTRY_SPOOL into os.environ for every test, and a world that
    #: inherited it recorded the suite's spool on its attempts and found a traversable spool where its proofs expect none)
    SITE_KEYS_THE_SHELL_LEAVES_TO_THE_FILE = ("SISTEER_REGISTRY_SPOOL", "SISTEER_FIT_ENV_NAME", "SISTEER_PRIMARY_CLUSTER", "SISTEER_DUAL_SUBMIT_THRESHOLD_MIN",
                                              "SISTEER_DUAL_SUBMIT_MAX_JOBS", "SISTEER_GRANT_TIMEOUT_MIN", "SISTEER_REPRODUCE_ROOT", "SISTEER_MODELS_TABLE",
                                              "SISTEER_REGISTRY_MIRROR_STATUS", "SISTEER_REGISTRY_CAMPAIGN", "REGISTRY_CAMPAIGN", "SISTEER_SITE_DIR", "SISTEER_CONFIG")

    def env(self, **extra):
        e = dict(os.environ, PATH=f"{self.bin}:{os.environ['PATH']}", SISTEER_STATE_DIR=str(self.tmp / "state"), SISTEER_NO_WATCHER="1",
                 SISTEER_REGISTRY_ROOT=str(self.tmp / "registry"))
        e.pop("SISTEER_TWIN_CRASH_AT", None)
        for k in self.SITE_KEYS_THE_SHELL_LEAVES_TO_THE_FILE:      # a test that names one passes it through `extra`, which stands
            e.pop(k, None)
        e.update(extra)
        return e

    def run(self, *args, **extra):
        return subprocess.run([sys.executable, str(LIB / "submit_twin.py"), "--config", str(self.site / "config.env"), "--model", "m", *args],
                              capture_output=True, text=True, env=self.env(**extra), timeout=120)

    def submit_twin(self, unit="multiturn.crescendo.g012", **extra):
        return self.run("submit", "--unit", unit, "--lwid", f"m/u0/multiturn/{unit}", "--script", SCRIPT, "--name", "f22-att-m-g012-crescendo",
                        "--args", f"m u0 crescendo {SHA} 9 0,1,2", "--twin", "--foreground", "--poll-s", "0.2", "--max-wait-s", "20", "--wave-sha", SHA, **extra)

    def ledger(self):
        p = self.state / "attempts.jsonl"
        return [json.loads(l) for l in p.read_text().splitlines()] if p.exists() else []

    def attempts_root(self, lab):
        return self.shared[lab] / "exp/reproduce/m/attempts"

    def sentinels(self, kind):
        return sorted(str(p.relative_to(self.tmp)) for lab in self.shared for p in self.attempts_root(lab).rglob(kind))

    def scancels(self, lab):
        p = self.fake[lab] / "scancel.log"
        return p.read_text().split() if p.exists() else []

    def no_stop_anywhere(self):
        bad = [str(p) for lab in self.shared for p in self.shared[lab].rglob("STOP")]
        assert not bad, bad

    def spool(self, lab, path=None) -> Path:
        """The producer spool the job env names (default `<shared>/registry/records`): the fenced proof's check 3 needs it to EXIST
        and be traversable before an empty result counts as absence."""
        p = Path(path) if path else self.shared[lab] / "registry/records"
        p.mkdir(parents=True, exist_ok=True); return p

    def ingested(self, lab, through="2026-01-01T00:20:00"):
        st = self.tmp / "registry" / "state"; st.mkdir(exist_ok=True)
        (st / "mirror_status.json").write_text(json.dumps({lab: {"ingested_through_utc": through}}))

    def retired(self):
        return sorted(str(p.relative_to(self.tmp)) for lab in self.shared for p in self.attempts_root(lab).rglob("GRANT.retired.*"))


def test_twin_refused_without_the_hook_marker(tmp_path):
    w = World(tmp_path, hook=False)
    r = w.submit_twin()
    assert r.returncode == 2 and "does not carry the in-body grant hook" in r.stderr, r.stderr
    assert job_ids(w.fake["cluster_a"]) == []          # nothing was submitted


def test_primary_pending_twin_wins_one_grant_loser_denied_and_scancelled(tmp_path):
    w = World(tmp_path)
    (w.fake["cluster_b"] / "policy").write_text("start_on_submit\n")       # b starts at once; a stays PENDING
    r = w.submit_twin()
    assert r.returncode == 0, r.stdout + r.stderr
    subs = [ln for ln in r.stdout.splitlines() if ln.startswith("submitted ")]
    assert [ln.split()[1] for ln in subs] == ["cluster_a", "cluster_b"], subs
    assert len(w.sentinels("GRANT")) == 1 and "shared_cluster_b" in w.sentinels("GRANT")[0]
    assert len(w.sentinels("DENY")) == 1 and "shared_cluster_a" in w.sentinels("DENY")[0]
    assert w.scancels("cluster_a") == [str(job_ids(w.fake["cluster_a"])[0])] and w.scancels("cluster_b") == []
    w.no_stop_anywhere()
    ev = [(e["event"], e.get("attempt_id", "")) for e in w.ledger()]
    order = [e[0] for e in ev]
    assert order.index("winner") < order.index("granted") < order.index("denied")
    # the job-side env prefix carried the attempt identity + the portability/registry env
    envA = (w.fake["cluster_a"] / "jobs" / f"{job_ids(w.fake['cluster_a'])[0]}.env").read_text()
    assert "SISTEER_ATTEMPT_ID=m/u0/multiturn/multiturn.crescendo.g012@cluster_a@1" in envA and "SISTEER_ATTEMPTS_ROOT=" in envA
    assert "SISTEER_REGISTRY_SUBSTRATE=slurm" in envA and "SISTEER_REGISTRY_CLUSTER_LABEL=cluster_a" in envA and "SISTEER_SHARED_ROOT=" in envA
    assert "SISTEER_CLUSTER_QOS=qos_cluster_a" in envA                                   # the site's QoS reaches the job (fit resubmit)
    argvA = (w.fake["cluster_a"] / "jobs" / f"{job_ids(w.fake['cluster_a'])[0]}.argv").read_text()
    assert "--qos=qos_cluster_a --partition=gpu_cluster_a --account=acct_cluster_a" in argvA and "--export" not in argvA
    # the attempt record carries the FULL spec + candidate clusters and the producer paths as submitted (findings 2 and 3)
    att = [e for e in w.ledger() if e["event"] == "attempt"][0]
    assert att["spec"]["clusters"] == ["cluster_a", "cluster_b"] and att["spool"] == f"{w.shared['cluster_a']}/registry/records"
    assert att["pending_dir"] == f"{w.shared['cluster_a']}/registry/pending"


def test_default_detached_submit_launches_the_twin(tmp_path):
    """Finding 2: the DEFAULT path (no --foreground) detaches `elect`, which reloads the spec + candidate clusters from the ledger and
    launches the twin on the other cluster when the primary stays PENDING."""
    w = World(tmp_path)
    (w.fake["cluster_b"] / "policy").write_text("start_on_submit\n")
    r = w.run("submit", "--unit", "multiturn.crescendo.g012", "--lwid", "m/u0/multiturn/multiturn.crescendo.g012", "--script", SCRIPT, "--name", "f22-att-m-g012-crescendo",
              "--args", f"m u0 crescendo {SHA} 9 0,1,2", "--twin", "--poll-s", "0.2", "--max-wait-s", "30", "--wave-sha", SHA)
    assert r.returncode == 0 and "detached" in r.stdout, r.stdout + r.stderr
    assert job_ids(w.fake["cluster_b"]) == []                                           # nothing on b yet: the detached election decides
    for _ in range(200):                                                                # the election runs on its own; wait for its decision
        if len(w.sentinels("GRANT")) == 1 and w.scancels("cluster_a"):
            break
        time.sleep(0.2)
    assert len(job_ids(w.fake["cluster_b"])) == 1, (w.state / "elect.log").read_text() if (w.state / "elect.log").exists() else "no elect.log"
    assert len(w.sentinels("GRANT")) == 1 and "shared_cluster_b" in w.sentinels("GRANT")[0]
    assert len(w.sentinels("DENY")) == 1 and w.scancels("cluster_a") == [str(job_ids(w.fake["cluster_a"])[0])]
    assert sum(1 for e in w.ledger() if e["event"] == "winner") == 1
    w.no_stop_anywhere()


def test_simultaneous_starts_yield_exactly_one_grant(tmp_path):
    w = World(tmp_path)
    for lab in w.fake:
        (w.fake[lab] / "policy").write_text("start_on_submit\n")
    r = w.submit_twin()
    # the primary starts at once: it is observed waiting before any twin is needed
    assert r.returncode == 0, r.stdout + r.stderr
    assert len(w.sentinels("GRANT")) == 1 and "shared_cluster_a" in w.sentinels("GRANT")[0]
    assert sum(1 for e in w.ledger() if e["event"] == "winner") == 1
    assert job_ids(w.fake["cluster_b"]) == []           # no twin was ever needed
    w.no_stop_anywhere()


@pytest.mark.parametrize("point", ["after_observed", "after_ledger_commit", "after_grant"])
def test_crash_at_every_point_replays_to_one_grant(tmp_path, point):
    w = World(tmp_path)
    (w.fake["cluster_b"] / "policy").write_text("start_on_submit\n")
    r = w.submit_twin(SISTEER_TWIN_CRASH_AT=point)
    assert r.returncode == 97, r.stdout + r.stderr
    events = [e["event"] for e in w.ledger()]
    if point == "after_observed":
        assert "winner" not in events and w.sentinels("GRANT") == []
    if point == "after_ledger_commit":
        assert "winner" in events and w.sentinels("GRANT") == [] and w.sentinels("DENY") == []     # committed, nothing remote yet
    if point == "after_grant":
        assert len(w.sentinels("GRANT")) == 1 and "granted" not in events and w.sentinels("DENY") == []
    # restart: replay the ledger to the decided end state
    r2 = w.run("elect", "--lwid", "m/u0/multiturn/multiturn.crescendo.g012", "--poll-s", "0.2", "--max-wait-s", "20")
    assert r2.returncode == 0, r2.stdout + r2.stderr
    assert len(w.sentinels("GRANT")) == 1 and "shared_cluster_b" in w.sentinels("GRANT")[0]
    assert len(w.sentinels("DENY")) == 1
    assert sum(1 for e in w.ledger() if e["event"] == "winner") == 1
    assert w.scancels("cluster_a") == [str(job_ids(w.fake["cluster_a"])[0])]
    w.no_stop_anywhere()


def test_crash_after_deny_replays_idempotently(tmp_path):
    w = World(tmp_path)
    (w.fake["cluster_b"] / "policy").write_text("start_on_submit\n")
    r = w.submit_twin(SISTEER_TWIN_CRASH_AT="after_deny:m/u0/multiturn/multiturn.crescendo.g012@cluster_a@1")
    assert r.returncode == 97
    assert len(w.sentinels("DENY")) == 1 and "denied" not in [e["event"] for e in w.ledger()]
    r2 = w.run("reconcile")
    assert r2.returncode == 0, r2.stdout + r2.stderr
    assert len(w.sentinels("GRANT")) == 1 and len(w.sentinels("DENY")) == 1
    assert sum(1 for e in w.ledger() if e["event"] == "winner") == 1


def test_registered_then_dead_winner_is_attached_never_regranted(tmp_path):
    w = World(tmp_path)
    (w.fake["cluster_b"] / "policy").write_text("start_on_submit\n")
    assert w.submit_twin().returncode == 0
    jb = job_ids(w.fake["cluster_b"])[0]
    # the winner registered (the job's own pending file exists on ITS cluster) and then died
    pend = w.shared["cluster_b"] / "registry/pending"; pend.mkdir(parents=True)
    (pend / f"{jb}.json").write_text(json.dumps({"runs": [{"run_id": "run-x"}]}))
    set_job_state(w.fake["cluster_b"], str(jb), "FAILED")
    r = w.run("reconcile")
    assert "ATTACH" in r.stdout and "REGISTERED before dying" in r.stdout, r.stdout + r.stderr
    assert sum(1 for e in w.ledger() if e["event"] == "winner") == 1
    assert len(w.sentinels("GRANT")) == 1


def test_dead_unregistered_winner_halts_without_the_fenced_proof(tmp_path):
    w = World(tmp_path)
    (w.fake["cluster_b"] / "policy").write_text("start_on_submit\n")
    assert w.submit_twin(SISTEER_TWIN_CRASH_AT="after_ledger_commit").returncode == 97    # winner committed, sibling a still PENDING
    jb = job_ids(w.fake["cluster_b"])[0]
    set_job_state(w.fake["cluster_b"], str(jb), "NODE_FAIL", end="2026-01-01T00:10:00")
    w.spool("cluster_b")                                  # the spool exists and is empty (check 3 negative) — the ingest watermark is UNKNOWN: fail closed
    # no pending file, no spool record, registry empty — but the ingest watermark is UNKNOWN: fail closed
    r = w.run("reconcile")
    assert r.returncode == 3 and "HALT" in r.stdout and "check 4" in r.stdout and "ingest watermark is unknown" in r.stdout, r.stdout + r.stderr
    assert sum(1 for e in w.ledger() if e["event"] == "winner") == 1 and w.sentinels("GRANT") == []        # a dead winner is never granted; no re-grant
    assert w.sentinels("DENY") == [] and w.scancels("cluster_a") == []                                     # HALT cancels nothing
    assert [e for e in w.ledger() if e["event"] == "halted"]


def test_dead_unregistered_winner_regrants_a_live_sibling_with_positive_proof(tmp_path):
    w = World(tmp_path)
    (w.fake["cluster_b"] / "policy").write_text("start_on_submit\n")
    assert w.submit_twin(SISTEER_TWIN_CRASH_AT="after_ledger_commit").returncode == 97
    jb = job_ids(w.fake["cluster_b"])[0]
    set_job_state(w.fake["cluster_b"], str(jb), "NODE_FAIL", end="2026-01-01T00:10:00")
    w.spool("cluster_b"); w.ingested("cluster_b")
    # the crashed coordinator never wrote b's GRANT; on replay b is found dead BEFORE registering (all four checks negative) and the
    # still-PENDING sibling a is granted under its own attempt id, ledger first
    r = w.run("reconcile")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "REGRANT" in r.stdout
    winners = [e["attempt_id"] for e in w.ledger() if e["event"] == "winner"]
    assert len(winners) == 2 and winners[1].endswith("@cluster_a@1")
    grants = w.sentinels("GRANT")
    assert len(grants) == 1 and "shared_cluster_a" in grants[0]                 # the dead winner is never granted
    assert w.scancels("cluster_a") == []                                          # the re-granted sibling is never cancelled
    w.no_stop_anywhere()


def test_dead_unregistered_winner_with_no_live_sibling_resubmits_a_new_attempt(tmp_path):
    w = World(tmp_path)
    (w.fake["cluster_b"] / "policy").write_text("start_on_submit\n")
    assert w.submit_twin().returncode == 0                        # b won; a was denied + scancelled; b's GRANT is on disk
    jb = job_ids(w.fake["cluster_b"])[0]
    set_job_state(w.fake["cluster_b"], str(jb), "NODE_FAIL", end="2026-01-01T00:10:00")   # died AFTER its GRANT, BEFORE registering
    w.spool("cluster_b"); w.ingested("cluster_b")
    r = w.run("reconcile")
    assert r.returncode == 0 and "RESUBMIT" in r.stdout and "RETIRED" in r.stdout and "REGRANT" in r.stdout, r.stdout + r.stderr
    assert len(job_ids(w.fake["cluster_b"])) == 2                 # a NEW attempt of the same unit on the winner's cluster
    winners = [e["attempt_id"] for e in w.ledger() if e["event"] == "winner"]
    assert winners[-1].endswith("@cluster_b@2")
    # the dead original's GRANT was tombstoned BEFORE the replacement's was written — ONE live GRANT, one retired
    assert w.sentinels("GRANT") == [g for g in w.sentinels("GRANT") if "@cluster_b@2" in g] and len(w.sentinels("GRANT")) == 1
    assert len(w.retired()) == 1 and "@cluster_b@1" in w.retired()[0]
    assert [e["attempt_id"] for e in w.ledger() if e["event"] == "grant_retired"] == [winners[0]]
    # ... so the NEXT reconcile does not halt the unit whose replacement is running
    r2 = w.run("reconcile")
    assert r2.returncode == 0 and "HALT" not in r2.stdout and "two grants" not in r2.stdout, r2.stdout + r2.stderr
    assert not [e for e in w.ledger() if e["event"] == "halted"]


def test_manual_resubmit_twin_of_a_dead_winner_grants_the_replacement(tmp_path):
    """`submit --resubmit --twin` for a unit whose bound job was the COMMITTED election
    winner and died unregistered goes through reconcile's retire -> commit -> grant transition — the NEW attempt is the winner and is
    GRANTED; before, the election re-read the old committed winner, granted it and wrote DENY (+ scancel) for the very job it had just
    launched, while reporting `granted`."""
    w = World(tmp_path)
    (w.fake["cluster_b"] / "policy").write_text("start_on_submit\n")
    assert w.submit_twin().returncode == 0                        # b won; a was denied + scancelled; b's GRANT is on disk
    jb = job_ids(w.fake["cluster_b"])[0]
    set_job_state(w.fake["cluster_b"], str(jb), "NODE_FAIL", end="2026-01-01T00:10:00")   # died AFTER its GRANT, BEFORE registering
    w.spool("cluster_b"); w.ingested("cluster_b")
    (w.fake["cluster_a"] / "policy").write_text("start_on_submit\n")
    unit = "multiturn.crescendo.g012"                             # the manual recovery: the SAME submit line + --resubmit (a new attempt on the primary, then the election)
    r = w.run("submit", "--unit", unit, "--lwid", f"m/u0/multiturn/{unit}", "--script", SCRIPT, "--name", "f22-att-m-g012-crescendo",
              "--args", f"m u0 crescendo {SHA} 9 0,1,2", "--twin", "--foreground", "--poll-s", "0.2", "--max-wait-s", "20", "--wave-sha", SHA, "--resubmit")
    assert r.returncode == 0 and "RESUBMIT" in r.stdout and "RETIRED" in r.stdout and "REGRANT" in r.stdout and "granted" in r.stdout, r.stdout + r.stderr
    winners = [e["attempt_id"] for e in w.ledger() if e["event"] == "winner"]
    new = winners[-1]
    assert new.endswith("@cluster_a@2") and new not in [e["attempt_id"] for e in w.ledger() if e["event"] == "denied"]
    assert w.sentinels("GRANT") == [g for g in w.sentinels("GRANT") if "@cluster_a@2" in g] and len(w.sentinels("GRANT")) == 1   # ONE live GRANT: the replacement's
    assert len(w.retired()) == 1 and "@cluster_b@1" in w.retired()[0]                          # the dead winner's tombstoned
    assert not [d for d in w.sentinels("DENY") if "@cluster_a@2" in d]                         # never denied its own replacement
    new_job = str(job_ids(w.fake["cluster_a"])[-1])
    assert new_job not in w.scancels("cluster_a")                                              # ... nor scancelled it
    r2 = w.run("reconcile")
    assert r2.returncode == 0 and "HALT" not in r2.stdout, r2.stdout + r2.stderr             # the next reconcile is clean


def test_crash_after_retire_replays_to_one_grant(tmp_path):
    w = World(tmp_path)
    (w.fake["cluster_b"] / "policy").write_text("start_on_submit\n")
    assert w.submit_twin().returncode == 0
    jb = job_ids(w.fake["cluster_b"])[0]
    set_job_state(w.fake["cluster_b"], str(jb), "NODE_FAIL", end="2026-01-01T00:10:00")
    w.spool("cluster_b"); w.ingested("cluster_b")
    r = w.run("reconcile", SISTEER_TWIN_CRASH_AT="after_retire")
    assert r.returncode == 97 and w.sentinels("GRANT") == [] and len(w.retired()) == 1       # the move landed, nothing committed yet
    r2 = w.run("reconcile")
    assert r2.returncode == 0 and "REGRANT" in r2.stdout, r2.stdout + r2.stderr
    assert len(w.sentinels("GRANT")) == 1 and "@cluster_b@2" in w.sentinels("GRANT")[0] and len(w.retired()) == 1
    assert [e["attempt_id"] for e in w.ledger() if e["event"] == "winner"][-1].endswith("@cluster_b@2")


def test_custom_spool_with_a_failed_push_attaches_and_never_regrants(tmp_path):
    """Finding 3: the proof reads the attempt's PRODUCER spool (SISTEER_REGISTRY_SPOOL as submitted) — a winner that registered locally,
    failed its push and died with no pending file is ATTACHED, not replaced, even though the generic spool root is empty."""
    custom = tmp_path / "custom_spool"
    w = World(tmp_path, config_extra=f"SISTEER_REGISTRY_SPOOL={custom}")
    (w.fake["cluster_b"] / "policy").write_text("start_on_submit\n")
    assert w.submit_twin(SISTEER_TWIN_CRASH_AT="after_ledger_commit").returncode == 97
    jb = job_ids(w.fake["cluster_b"])[0]
    envB = (w.fake["cluster_b"] / "jobs" / f"{jb}.env").read_text()
    assert f"SISTEER_REGISTRY_SPOOL={custom}" in envB                                    # the job wrote THERE
    set_job_state(w.fake["cluster_b"], str(jb), "FAILED", end="2026-01-01T00:10:00")
    w.spool("cluster_b")                                                                  # the generic root exists and is EMPTY
    (custom / "run-dir").mkdir(parents=True); (custom / "run-dir" / "rec.jsonl").write_text(json.dumps({"record_type": "run", "cluster_job_id": str(jb)}) + "\n")
    w.ingested("cluster_b")
    r = w.run("reconcile")
    assert "ATTACH" in r.stdout and "REGISTERED before dying" in r.stdout and "REGRANT" not in r.stdout, r.stdout + r.stderr
    assert sum(1 for e in w.ledger() if e["event"] == "winner") == 1 and len(w.sentinels("GRANT")) == 1 and "shared_cluster_b" in w.sentinels("GRANT")[0]


def test_missing_spool_is_unknown_never_proof_of_absence(tmp_path):
    w = World(tmp_path)
    (w.fake["cluster_b"] / "policy").write_text("start_on_submit\n")
    assert w.submit_twin(SISTEER_TWIN_CRASH_AT="after_ledger_commit").returncode == 97
    jb = job_ids(w.fake["cluster_b"])[0]
    set_job_state(w.fake["cluster_b"], str(jb), "NODE_FAIL", end="2026-01-01T00:10:00")
    w.ingested("cluster_b")                                                               # checks 1, 2, 4 negative — but NO spool directory at all
    r = w.run("reconcile")
    assert r.returncode == 3 and "HALT" in r.stdout and "check 3" in r.stdout and "missing or could not be traversed" in r.stdout, r.stdout + r.stderr
    assert w.sentinels("GRANT") == [] and sum(1 for e in w.ledger() if e["event"] == "winner") == 1 and w.scancels("cluster_a") == []


def test_submit_is_idempotent_and_resubmit_needs_the_fenced_proof(tmp_path):
    """Ruling 11: a unit already bound is SKIPPED (no second identity); --resubmit re-mints only after the proof that the bound job is
    terminal and never registered; a REGISTERED unit is never re-minted."""
    w = World(tmp_path)
    (w.fake["cluster_a"] / "policy").write_text("start_on_submit\n")
    sub = ["submit", "--unit", "mmlu", "--lwid", "m/u0/mmlu/mmlu", "--script", "scripts/sdfbase/full22/mmlu_full.sbatch", "--name", "mmluF-m-u0", "--args", f"m u0 {SHA}",
           "--clusters", "cluster_a", "--wave-sha", SHA]
    r1 = w.run(*sub)
    assert r1.returncode == 0 and "submitted cluster_a 1001 mmlu" in r1.stdout, r1.stdout + r1.stderr
    r2 = w.run(*sub)
    assert r2.returncode == 0 and "skipped cluster_a 1001 mmlu (already submitted" in r2.stdout and "submitted " not in r2.stdout, r2.stdout
    assert job_ids(w.fake["cluster_a"]) == [1001] and sum(1 for e in w.ledger() if e["event"] == "attempt") == 1
    r3 = w.run(*sub, "--resubmit")                                                        # the bound job is RUNNING: skipped (06 walks on), nothing minted
    assert r3.returncode == 0 and "skipped cluster_a 1001 mmlu (still RUNNING" in r3.stdout and "submitted " not in r3.stdout and job_ids(w.fake["cluster_a"]) == [1001], r3.stdout + r3.stderr
    set_job_state(w.fake["cluster_a"], "1001", "NODE_FAIL", end="2026-01-01T00:10:00")
    w.spool("cluster_a"); w.ingested("cluster_a")
    r4 = w.run(*sub, "--resubmit")
    assert r4.returncode == 0 and "submitted cluster_a 1002 mmlu" in r4.stdout, r4.stdout + r4.stderr
    assert [e["attempt_id"] for e in w.ledger() if e["event"] == "attempt"] == ["m/u0/mmlu/mmlu@cluster_a@1", "m/u0/mmlu/mmlu@cluster_a@2"]
    bind = json.load(open(w.state / "bindings" / "m__u0__mmlu__mmlu.submitted.json"))
    assert bind["job_id"] == "1002"
    # a registered unit whose job is LIVE: skipped like any live unit (a registered binding is decided by the same proof)
    (w.state / "bindings" / "m__u0__mmlu__mmlu.registered.json").write_text(json.dumps({"logical_work_id": "m/u0/mmlu/mmlu", "state": "registered", "run_ids": ["run-x"], "job_id": "1002", "cluster": "cluster_a"}))
    r5 = w.run(*sub, "--resubmit")
    assert r5.returncode == 0 and "skipped cluster_a 1002 mmlu (still RUNNING" in r5.stdout and job_ids(w.fake["cluster_a"]) == [1001, 1002], r5.stdout + r5.stderr


def test_resubmit_reruns_a_registered_then_dead_eval_unit_under_the_same_identities(tmp_path):
    """paper-companion-repo stage E (2026-09-30): every eval job registers natively at start, so an eval job that dies after its first
    minute is "registered" and ruling 11's four-way proof alone would strand it. --resubmit re-runs such a unit as a new attempt of the
    SAME identities (deterministic run_id; the registry upserts) when its bound job is TERMINAL and not COMPLETED; a COMPLETED job and a
    fit (whose successor attaches by name) are still refused."""
    w = World(tmp_path)
    (w.fake["cluster_a"] / "policy").write_text("start_on_submit\n")
    sub = ["submit", "--unit", "static.t1", "--lwid", "m/u0/static/t1", "--script", "scripts/sdfbase/full22/sampled/target_sampled.sbatch",
           "--name", "f22s-m-u0-static_base_t1", "--args", f"m static_base_t1 {SHA} 9 '' u0", "--clusters", "cluster_a", "--wave-sha", SHA]
    r1 = w.run(*sub)
    assert r1.returncode == 0 and "submitted cluster_a 1001 static.t1" in r1.stdout, r1.stdout + r1.stderr
    # the job registered (its own pending file on its cluster) and then died at standup; the spool is traversable and the registry
    # has ingested past the job's end (the proof fails closed without either)
    pend = w.shared["cluster_a"] / "registry/pending"; pend.mkdir(parents=True, exist_ok=True)
    (pend / "1001.json").write_text(json.dumps({"job_id": "1001", "runs": [{"run_id": "run-a"}, {"run_id": "run-b"}]}))
    set_job_state(w.fake["cluster_a"], "1001", "FAILED", end="2026-01-01T00:10:00")
    w.spool("cluster_a"); w.ingested("cluster_a", through="2026-01-01T06:00:00")
    r2 = w.run(*sub, "--resubmit")
    assert r2.returncode == 0 and "submitted cluster_a 1002 static.t1" in r2.stdout, r2.stdout + r2.stderr
    assert "RERUN m/u0/static/t1" in r2.stdout + r2.stderr and "REGISTERED before dying" in r2.stdout + r2.stderr and "2 run id(s)" in r2.stdout + r2.stderr
    sup = [e for e in w.ledger() if e["event"] == "superseded_attempt"]
    assert [ (e["attempt_id"], e["job_id"], e["state"]) for e in sup ] == [("m/u0/static/t1@cluster_a@1", "1001", "FAILED")]
    assert [e["attempt_id"] for e in w.ledger() if e["event"] == "attempt"] == ["m/u0/static/t1@cluster_a@1", "m/u0/static/t1@cluster_a@2"]
    assert json.load(open(w.state / "bindings" / "m__u0__static__t1.submitted.json"))["job_id"] == "1002"
    # the re-run registered and COMPLETED: a re-run of a finished job is refused (its runs are the unit's results)
    (pend / "1002.json").write_text(json.dumps({"job_id": "1002", "runs": [{"run_id": "run-a"}, {"run_id": "run-b"}]}))
    set_job_state(w.fake["cluster_a"], "1002", "COMPLETED", end="2026-01-01T02:00:00")
    r3 = w.run(*sub, "--resubmit")
    assert r3.returncode == 3 and "DEAD (terminal, not COMPLETED)" in r3.stderr and "COMPLETED task" in r3.stderr and job_ids(w.fake["cluster_a"]) == [1001, 1002], r3.stderr
    # a FIT that registered then died keeps ruling 11's path: never re-run (its successor is bound by name). The fake cluster's
    # fit venv is certified the way sdfbase_fit_env.sbatch certifies it (BUILD-COMPLETE + the recipe), or the funnel refuses earlier.
    venv = w.shared["cluster_a"] / "envs" / "sdfbase-fit"; venv.mkdir(parents=True, exist_ok=True)
    (venv / "BUILD-COMPLETE").write_text("2026-09-16T05:09:17+00:00 40b49ecf00359e44\n")
    (w.shared["cluster_a"] / "envs" / "sdfbase-fit.build").mkdir(exist_ok=True); (w.shared["cluster_a"] / "envs" / "sdfbase-fit.build" / "requirements.txt").write_text("torch==2.11.0\n")
    fit = ["submit", "--unit", "fit.relu", "--lwid", "m/04_fit/relu/x", "--script", "scripts/sdfbase/sdfbase_fit_cell.sbatch", "--name", "fit-m-relu-x",
           "--args", f"m relu x {SHA}", "--clusters", "cluster_a", "--wave-sha", SHA, "--kind", "fit"]
    r4 = w.run(*fit)
    assert r4.returncode == 0 and "submitted cluster_a 1003 fit.relu" in r4.stdout, r4.stdout + r4.stderr
    (pend / "1003.json").write_text(json.dumps({"job_id": "1003", "runs": [{"run_id": "run-fit"}]}))
    set_job_state(w.fake["cluster_a"], "1003", "FAILED", end="2026-01-01T03:00:00")
    w.ingested("cluster_a", through="2026-01-01T06:00:00")
    r5 = w.run(*fit, "--resubmit")
    assert r5.returncode == 3 and "not provably dead-and-unregistered (registered" in r5.stderr and job_ids(w.fake["cluster_a"]) == [1001, 1002, 1003], r5.stderr


def test_spool_record_counts_as_registered(tmp_path):
    w = World(tmp_path)
    (w.fake["cluster_b"] / "policy").write_text("start_on_submit\n")
    assert w.submit_twin(SISTEER_TWIN_CRASH_AT="after_ledger_commit").returncode == 97
    jb = job_ids(w.fake["cluster_b"])[0]
    set_job_state(w.fake["cluster_b"], str(jb), "FAILED")
    spool = w.shared["cluster_b"] / "registry/records/run-dir"; spool.mkdir(parents=True)
    (spool / "rec.jsonl").write_text(json.dumps({"record_type": "run", "cluster_job_id": str(jb)}) + "\n")
    r = w.run("reconcile")
    assert "ATTACH" in r.stdout and sum(1 for e in w.ledger() if e["event"] == "winner") == 1


def test_unreachable_cluster_halts(tmp_path):
    w = World(tmp_path)
    (w.fake["cluster_b"] / "policy").write_text("start_on_submit\n")
    assert w.submit_twin(SISTEER_TWIN_CRASH_AT="after_ledger_commit").returncode == 97
    (w.fake["cluster_b"] / "outage").write_text("")     # every squeue/sacct on b exits 255
    r = w.run("reconcile")
    assert r.returncode == 3 and "unreachable" in (r.stdout + r.stderr).lower()
    assert sum(1 for e in w.ledger() if e["event"] == "winner") == 1


def test_teardown_writes_per_member_stop_never_the_pool_stop(tmp_path):
    w = World(tmp_path)
    (w.fake["cluster_a"] / "policy").write_text("start_on_submit\n")
    for i in (1, 2):
        r = w.run("submit", "--unit", f"multiturn.attacker{i}", "--lwid", f"m/u0/multiturn/mt-u0-cluster_a/attackers/{i}", "--script", "scripts/sdfbase/full22/attacker_pool.sbatch",
                  "--name", f"f22-atk-m-{i}", "--args", SHA, "--clusters", "cluster_a", "--kind", "attacker", "--group", "mt-u0-cluster_a", "--wave-sha", SHA)
        assert r.returncode == 0, r.stdout + r.stderr
        envA = (w.fake["cluster_a"] / "jobs" / f"{job_ids(w.fake['cluster_a'])[-1]}.env").read_text()
        assert "F22_WAVE_ROOT=" in envA and "/waves/mt-u0-cluster_a" in envA           # the group's PRIVATE pool root
    r = w.run("teardown-pool", "--group", "mt-u0-cluster_a")
    assert r.returncode == 0, r.stdout + r.stderr
    names = sorted(p.name for p in (w.shared["cluster_a"] / "exp/reproduce/m/waves/mt-u0-cluster_a/attackers").glob("*"))
    # the pool root holds EXACTLY the per-member STOPs plus the group's lock the teardown transaction ran under (`flock <pool_root>/PLACED.lock`,
    # placement-fix round 4, 4d27178fe: the STOP writes and the transition / placement checks share the lock the pass's `open` takes) — a lock
    # file, never a pool-wide STOP and never a sentinel for a member the ledger does not record
    assert names == sorted([f"{j}.STOP" for j in job_ids(w.fake["cluster_a"])] + ["PLACED.lock"])
    assert "STOP" not in names                                                             # never attackers/STOP


def test_padded_scripts_refuse_a_comma_list_of_levels(tmp_path):
    w = World(tmp_path)
    r = w.run("submit", "--unit", "pad", "--lwid", "m/u0/padding/x", "--script", "scripts/sdfbase/full22/static_padding/target_static_padded.sbatch",
              "--name", "padstat-m", "--args", f"m {SHA} 98304 16384,65536 grok420 3", "--clusters", "cluster_a", "--wave-sha", SHA)
    assert r.returncode == 2 and "ONE padding level per job" in r.stderr


def test_dry_run_prints_the_line_and_writes_nothing(tmp_path):
    w = World(tmp_path)
    r = w.run("--dry-run", "submit", "--unit", "u", "--lwid", "m/u0/static/pasttense", "--script", SCRIPT, "--name", "n", "--args", f"m u0 crescendo {SHA} 9 0,1,2", "--twin", "--wave-sha", SHA)
    assert r.returncode == 0 and "DRY-RUN: ssh local:" in r.stdout and "sbatch --qos=qos_cluster_a --partition=gpu_cluster_a" in r.stdout, r.stdout + r.stderr
    assert job_ids(w.fake["cluster_a"]) == [] and not (w.state / "attempts.jsonl").exists()
