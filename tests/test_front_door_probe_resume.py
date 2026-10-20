"""`reproduce/03_hpo.sh --phase probe` OWNS the resume of an INCOMPLETE probe record and reads the record's state from
`hpo_probe_status.py --record` alone. Executed, not string-matched: the REAL stage runs against a `local:` site whose
`ssh` executes the remote string on this host (the status reader travels on its stdin, as live), whose `rsync` copies
locally, and whose `sbatch`/`squeue` are the front-door fakes — so the sbatch argv, the env prefix, the ledger line and
the exit codes are the production ones.

Pinned:
* INCOMPLETE_STALLED -> ONE sbatch through lib/slurm.sh: the original carrier, the cluster block's qos/partition/account/time,
  `--comment=sha:<HEAD>`, the record's positionals 4-6 from probe_plan.json, attempt+1 and the cap from probe_attempt.json, the
  record's width as `--gres/--cpus-per-task/--mem` (or --probe-width), `parallel` auto, SERVING_PROBE_OUT=<record> as an env
  prefix (never --export=, never --dependency), the attempt-scoped unit `hpo.lora.probe.a<N>` in the ledger and on the printed
  submission line.
* a queued/running probe job -> nothing submitted, exit 0 (RUNNING / INCOMPLETE_RUNNING).
* a menu present but the checkpoint incomplete (MENU_STALE) -> not fetched, not marked, exit 2.
* READY -> fetched, the DONE marker records the result generation, a second run skips; a NEWER local menu is never overwritten.
* the cap reached / a record with no attempt record / an unrecorded width without --probe-width -> refused by name, no sbatch.
* an attempt number the ledger already binds is skipped; a fresh NOT_STARTED probe takes --probe-width onto the command line.
* the carrier's attempt-record block, run under bash, writes the JSON the resume reads.
* `--reopen` (before it was wired the verb was defined and never called, so a READY `--phase probe --reopen` took the fetch):
  on a READY (or DONE-and-current) record the site copy and every cluster's staged overlay are revoked (REVOKED markers), the
  revocation receipt is written INTO the record with the cluster acknowledged, ONE sbatch goes out with the receipt's 32-hex
  nonce as positional 10, nothing is fetched and nothing skipped; a held record, an incomplete record refuse by name with
  nothing revoked; a dry run describes the revocation and the token.
Synthetic everything (placeholder shas, TESTHOST, the neutral site labels).
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from front_door_testlib import REPO, REPRO, VENV_PY, assert_no_tokens, job_ids, write_fake_slurm, write_site  # noqa: E402

sys.path.insert(0, str(REPO / "reproduce" / "lib"))
import stage as ST  # noqa: E402

MODEL = "qwen35_4b_demo"                                                # an onboarded model whose LoRA pick is pending (FD 6)
SRV_ID = "Qwen/Qwen3.5-4B"                                    # its served checkpoint (reproduce/models.yaml)
OVERRIDES = REPO / "tests" / "fixtures" / "front_door" / "fd6_overrides.env"   # the overrides the LoRA probe applies: a synthetic overrides file
SETS = ["wo", "qkvo", "attn_mlp", "mlp", "down"]
NAME = f"hpo-{MODEL}-lora-probe"
LWID = f"{MODEL}/03_hpo/lora/hpo.lora.probe"
HEAD = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO).stdout.strip()
REAL_RSYNC = shutil.which("rsync")

FAKE_SSH = r'''#!/bin/bash
# fake ssh: drop the options, take the host, run the remote string on this host; stdin passes through (the status reader
# travels on it, exactly as live); a `local:<dir>` host names the fake Slurm state dir
host=""; while [ $# -gt 0 ]; do case "$1" in -o) shift 2;; -*) shift;; *) host="$1"; shift; break;; esac; done
case "$host" in local:*) export FAKE_SLURM_DIR="${host#local:}";; esac
printf '%s\t%s\n' "$host" "$*" >> "${FAKE_SSH_LOG:?}"
exec bash -c "$*"
'''
FAKE_RSYNC = '''#!/bin/bash
# fake rsync: the source is `<host>:<path>/` — drop the host and copy locally with the real rsync and the SAME filters
args=("$@"); n=${#args[@]}; src="${args[n-2]}"; dst="${args[n-1]}"; src="${src##*:}"
unset 'args[n-1]' 'args[n-2]'
exec %s "${args[@]}" "$src" "$dst"
'''
FAKE_SQUEUE = r'''#!/bin/bash
# fake squeue: `-u USER -h -o %j` lists the NAMES of the fake state's active jobs (03's cluster_job_active); `-j ID` as the
# front-door fake
[ -z "${FAKE_SQUEUE_FAIL:-}" ] || { echo "squeue: error: slurm_load_jobs error: Unable to contact slurm controller (connect failure)" >&2; exit 1; }
d="${FAKE_SLURM_DIR:?}"; ids=""; fmt=""; user=""; prev=""
for a in "$@"; do [ "$prev" = "-j" ] && ids="$a"; [ "$prev" = "-o" ] && fmt="$a"; [ "$prev" = "-u" ] && user="$a"; prev="$a"; done
if [ -n "$user" ] && [ -z "$ids" ]; then
  for f in "$d"/jobs/*.state; do [ -f "$f" ] || continue; st=$(cat "$f"); id=$(basename "$f" .state)
    case "$st" in PENDING|RUNNING|CONFIGURING|COMPLETING) if [ "$fmt" = "%j" ]; then cat "$d/jobs/$id.name"; else echo "$id $(cat "$d/jobs/$id.name") $st 0:00"; fi;; esac
  done; exit 0
fi
any=0
for id in ${ids//,/ }; do [ -f "$d/jobs/$id.state" ] || continue; any=1; st=$(cat "$d/jobs/$id.state")
  case "$st" in PENDING|RUNNING|CONFIGURING|COMPLETING) if [ "$fmt" = "%T" ]; then echo "$st"; else echo "$id $(cat "$d/jobs/$id.name") $st 0:00"; fi;; esac
done
[ "$any" = 1 ] || { echo "slurm_load_jobs error: Invalid job id specified" >&2; exit 1; }
'''


def _plan() -> dict:
    launches = [{"id": f"L0{i}", "kind": "servability" if i < 5 else "memo", "L": 4096, "cap": 8, "set": "wo", "rank": 4,
                 "manifest": "wo_r4", "topology": "single", "admitted_S": 8, "priority": 10} for i in range(1, 8)]
    return {"version": 1, "written_utc": "2026-09-30T00:00:00Z", "model_key": MODEL, "model_id": SRV_ID, "served_hf_id": SRV_ID,
            "revision": "0" * 40, "sets": SETS, "ranks": [4, 16, 64], "tp": 1, "n_launches": 7, "launches": launches}


def _progress(done: int, complete: bool) -> dict:
    ids = [f"L0{i}" for i in range(1, 8)]
    return {"version": 1, "written_utc": "2026-09-30T01:00:00Z", "model_key": MODEL, "plan_launches": 7, "n_done": done,
            "done": ids[:done], "outstanding": [{"id": i, "kind": "servability", "L": 4096, "cap": 8} for i in ids[done:]],
            "decision_outstanding": ids[done:], "replan_required": [], "complete": complete, "stopped_reason": None if complete else "time_budget"}


def _attempt(attempt: int = 2, cap: int = 4, gpus: int = 4) -> dict:
    return {"version": 1, "written_utc": "2026-09-30T00:30:00Z", "attempt": attempt, "max_attempts": cap, "gpus": gpus, "parallel": "auto",
            "mode": "tp1", "tp": 1, "sets": ",".join(SETS), "ranks": "4,16,64", "model_key": MODEL, "served_hf_id": SRV_ID,
            "revision": "0" * 40, "job_id": "11111111", "job_name": NAME, "sha": "a" * 40, "qos": "qos_cluster_a", "time_limit": "04:00:00",
            "cpus_per_task": "64", "mem_mb": "524288", "host": "TESTHOST"}


class World:
    def __init__(self, tmp_path: Path):
        self.tmp = tmp_path
        self.fake = tmp_path / "fake_a"; self.fake.mkdir(parents=True)
        self.shared = tmp_path / "shared_a"
        self.site = write_site(tmp_path, clusters=("cluster_a",), ssh={"cluster_a": f"local:{self.fake}"}, shared={"cluster_a": str(self.shared)},
                               extra_config=f"SISTEER_RUN_OVERRIDES={OVERRIDES}\n")
        self.bin = tmp_path / "bin"; write_fake_slurm(self.bin)
        for name, body in (("ssh", FAKE_SSH), ("rsync", FAKE_RSYNC % REAL_RSYNC), ("squeue", FAKE_SQUEUE)):
            p = self.bin / name; p.write_text(body); p.chmod(0o755)
        self.ssh_log = tmp_path / "ssh.log"
        self.record = self.shared / "exp" / "reproduce" / MODEL / "03_hpo" / "lora"
        self.record.mkdir(parents=True)
        (self.shared / "sisteer-eval-repo" / HEAD).mkdir(parents=True)     # the pinned checkout the sbatch `cd`s into
        self.state = tmp_path / "state" / MODEL / "03_hpo" / "lora"
        self.ledger = tmp_path / "state" / MODEL / "submissions.jsonl"

    # --- the record on the "cluster"
    def lay(self, *, done=3, complete=False, attempt: dict | None = _attempt(), menu: bool = False, verdict: bool = False,
            legacy_head: str | None = None):
        (self.record / "probe_plan.json").write_text(json.dumps(_plan()))
        (self.record / "probe_progress.json").write_text(json.dumps(_progress(done, complete)))
        launches = self.record / "launches"; launches.mkdir(exist_ok=True)
        for i in range(1, done + 1):
            (launches / f"L0{i}.json").write_text(json.dumps({"launch_id": f"L0{i}", "status": "passed", "host": "TESTHOST"}))
        if attempt is not None:
            (self.record / "probe_attempt.json").write_text(json.dumps(attempt))
        if verdict:
            (self.record / "probe_verdict.json").write_text(json.dumps({
                "version": 1, "written_utc": "2026-09-30T02:00:00Z", "model_key": MODEL, "probe_job_id": "11111111", "status": "complete",
                "certifying": True, "launches": [{"id": f"L0{i}", "status": "passed" if i <= done else "not_run"} for i in range(1, 8)]}))
        if menu:
            (self.record / "allowed_pairs.json").write_text(json.dumps({
                "model_id": SRV_ID, "served_hf_id": SRV_ID, "revision": "0" * 40, "model_key": MODEL, "probe_job_id": "11111111",
                "written_utc": "2026-09-30T02:00:00Z", "status": "complete", "certifying": True, "pairs": [{"rank": 4, "set": "wo"}]}))
        if legacy_head is not None:
            figs = self.shared / "exp" / "figs" / "serving-probe" / "lora"; figs.mkdir(parents=True, exist_ok=True)
            (figs / "RESULT").write_text(legacy_head + "\nverdict=none\n")

    def env(self, **extra_env) -> dict:
        # the mirror step of submit() (72c2392f4) pushes the whole repo to the cluster mirror; its own test covers it — here the
        # invocation is told the pinned sha is already mirrored, so the funnel proceeds to the sbatch. 03 reads HEAD at RUN time
        # (repro_head_sha), so the sha is read HERE, per call, never the import-time HEAD: on the shared worktree peers commit under a
        # long run, the import-time HEAD fell behind, every later test performed a REAL ~6 GB mirror into the fake shared root and the
        # grading host root filled in 30 minutes (2026-09-30). SISTEER_TEST_NO_MIRROR=1 makes lib/mirror_repo.sh refuse by name if it is
        # ever reached from a test world all the same.
        head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO).stdout.strip() or HEAD
        (self.shared / "sisteer-eval-repo" / head).mkdir(parents=True, exist_ok=True)     # the pinned checkout the sbatch `cd`s into
        env = {"PATH": f"{self.bin}:{os.environ['PATH']}", "HOME": str(self.tmp), "USER": "nobody-testhost", "FAKE_SSH_LOG": str(self.ssh_log),
               "SISTEER_STATE_DIR": str(self.tmp / "state"), "SISTEER_NO_WATCHER": "1",
               "MIRRORED_SHA": head, "SISTEER_TEST_NO_MIRROR": "1"}
        env.update(extra_env); return env

    def argv(self, *extra: str) -> list:
        return ["bash", str(REPRO / "03_hpo.sh"), "--model", MODEL, "--object-class", "lora", "--phase", "probe", "--config", str(self.site / "config.env"), *extra]

    def run(self, *extra: str, **extra_env) -> subprocess.CompletedProcess:
        return subprocess.run(self.argv(*extra), capture_output=True, text=True, env=self.env(**extra_env), cwd=str(REPO), timeout=600)

    def popen(self, *extra: str, stdout, stderr, **extra_env) -> subprocess.Popen:
        return subprocess.Popen(self.argv(*extra), stdout=stdout, stderr=stderr, text=True, env=self.env(**extra_env), cwd=str(REPO))

    def sbatch_calls(self) -> list[dict]:
        out = []
        for jid in job_ids(self.fake):
            argv = (self.fake / "jobs" / f"{jid}.argv").read_text().split()
            envs = dict(ln.split("=", 1) for ln in (self.fake / "jobs" / f"{jid}.env").read_text().splitlines() if "=" in ln)
            out.append({"id": jid, "argv": argv[1:], "env": envs, "name": (self.fake / "jobs" / f"{jid}.name").read_text().strip()})
        return out


def _scan(w: World, text: str) -> None:
    scan = text.replace(str(REPO), "<repo>").replace(VENV_PY, "<python>").replace(str(w.tmp), "<tmp>").replace(os.path.expanduser("~"), "<home>")
    assert_no_tokens(scan, "the probe's output")


@pytest.fixture
def world(tmp_path):
    if REAL_RSYNC is None:
        pytest.skip("no rsync on this host")
    return World(tmp_path)


def test_incomplete_stalled_resumes_the_same_record_through_the_funnel(world):
    """One sbatch, the production line: attempt 3/4 of the record, positionals 4-6 from the plan, the record's width on the
    command line, SERVING_PROBE_OUT as an env prefix, the attempt-scoped unit in the ledger and on the printed submission line."""
    world.lay()
    r = world.run()
    out = r.stdout + r.stderr
    assert r.returncode == 0, out
    calls = world.sbatch_calls()
    assert len(calls) == 1, out
    c = calls[0]; argv = c["argv"]
    assert c["name"] == NAME
    i = argv.index("scripts/sbatch/serving_probe.sbatch")
    assert argv[i + 1:] == [MODEL, SRV_ID] + [argv[i + 3]] + [",".join(SETS), "4,16,64", "tp1", "3", "4", "auto"], argv[i:]
    assert len(argv[i + 3]) == 40                                              # the served revision from models.yaml
    flags = argv[:i]
    assert "--qos=qos_cluster_a" in flags and "--partition=gpu_cluster_a" in flags and "--account=acct_cluster_a" in flags
    assert "--time=04:00:00" in flags and "-J" in flags and flags[flags.index("-J") + 1] == NAME
    # the comment carries the pin AND the reservation's nonce as `,`-separated key:value fields (comment_with_nonce; the sbatch readers parse
    # the sha field by name, pin-fix) — the resumed job stays adoptable by its nonce (squeue %k), the pin still names HEAD
    cmt = [f for f in flags if f.startswith("--comment=")]
    assert len(cmt) == 1, flags
    fields = dict(kv.split(":", 1) for kv in cmt[0][len("--comment="):].split(","))
    assert fields["sha"] == HEAD and re.fullmatch(r"[0-9a-f]{12}", fields.get("nonce", "")), cmt
    assert "--gres=gpu:4" in flags and "--cpus-per-task=64" in flags and "--mem=512G" in flags     # the record's last width
    assert not any(f.startswith("--export") or f.startswith("--dependency") for f in flags), flags
    # the job-side env: the site's prefix (the fake sbatch records the SISTEER_* family) and SERVING_PROBE_OUT=<record> as the
    # funnel's own --env record on the attempt (the fake keeps no SERVING_* variable; the funnel composes both as VAR=value
    # prefixes on the one ssh string — never --export=, refused by submit_twin)
    assert c["env"]["SISTEER_SHARED_ROOT"] == str(world.shared) and c["env"]["SISTEER_CLUSTER_QOS"] == "qos_cluster_a"
    assert c["env"]["SISTEER_RUN_OVERRIDES"].endswith("/reproduce/overrides/" + OVERRIDES.name)
    # the ledger line + the printed submission line, under the attempt-scoped unit; the fit venv was NOT read (kind other)
    assert f"submitted cluster_a {c['id']} hpo.lora.probe.a3" in out, out
    subs = [json.loads(ln) for ln in world.ledger.read_text().splitlines() if ln.strip()]
    assert [s for s in subs if s.get("event") == "submitted" and s["unit"] == "hpo.lora.probe.a3" and s["job_id"] == str(c["id"])]
    assert ST.binding_state(str(world.ledger), f"{LWID}.a3")["state"] == "submitted"
    attempts = [json.loads(ln) for ln in (world.tmp / "state" / MODEL / "attempts.jsonl").read_text().splitlines() if ln.strip()]
    att = [a for a in attempts if a.get("event") == "attempt"]
    assert len(att) == 1 and att[0]["logical_work_id"] == f"{LWID}.a3" and att[0]["kind"] == "other" and att[0]["args"][6:9] == ["3", "4", "auto"]
    assert att[0]["spec"]["env"] == {"SERVING_PROBE_OUT": str(world.record)} and att[0]["script"] == "scripts/sbatch/serving_probe.sbatch"
    assert att[0]["sha"] == HEAD and att[0]["job_name"] == NAME
    assert "counter and cap from probe_attempt.json" in out and "resuming" in out and "INCOMPLETE on cluster_a with no job" in out
    # the status reader ran on the "cluster" over the record, on stdin
    ssh = world.ssh_log.read_text()
    assert f"python3 - --record '{world.record}' --job-active 0" in ssh
    _scan(world, out)
    # the SAME invocation again, the job now PENDING under the probe's name: nothing submitted, exit 0
    r2 = world.run()
    out2 = r2.stdout + r2.stderr
    assert r2.returncode == 0 and len(world.sbatch_calls()) == 1, out2
    assert "INCOMPLETE_RUNNING" in out2 and "nothing submitted" in out2 and "job-active 1" in world.ssh_log.read_text()


def test_probe_width_overrides_the_records_width_and_auto_takes_the_header(world):
    world.lay()
    r = world.run("--probe-width", "2")
    assert r.returncode == 0, r.stdout + r.stderr
    flags = world.sbatch_calls()[0]["argv"]
    assert "--gres=gpu:2" in flags and "--cpus-per-task=32" in flags and "--mem=256G" in flags
    w2 = World(world.tmp / "w2"); w2.lay(attempt=_attempt(attempt=1))
    r = w2.run("--probe-width", "auto")
    assert r.returncode == 0, r.stdout + r.stderr
    argv = w2.sbatch_calls()[0]["argv"]
    assert not any(f.startswith("--gres") or f.startswith("--cpus-per-task") or f.startswith("--mem") for f in argv), argv
    assert argv[argv.index("tp1") + 1:] == ["2", "4", "auto"]
    r = w2.run("--probe-width", "eight")
    assert r.returncode == 2 and "--probe-width must be auto or a positive integer" in r.stderr


def test_the_cap_a_missing_attempt_record_and_an_unrecorded_width_refuse_by_name_and_submit_nothing(world):
    world.lay(attempt=_attempt(attempt=4, cap=4))
    r = world.run()
    assert r.returncode == 2 and "attempt cap 4 is reached" in r.stderr and "NOT submitting a fresh probe" in r.stderr, r.stdout + r.stderr
    assert world.sbatch_calls() == []
    # a record written before the attempt record existed and no carrier RESULT head: refused (never a re-typed counter)
    (world.record / "probe_attempt.json").unlink()
    r = world.run()
    assert r.returncode == 2 and "no probe_attempt.json" in r.stderr and world.sbatch_calls() == []
    # the carrier's per-exit head supplies the counter, but no width is recorded: --probe-width is required …
    world.lay(attempt=None, legacy_head="state=INTERRUPTED exit=143 attempt=1/4 elapsed=7585 host=TESTHOST model=qwen35_4b_demo mode=tp1 job=22222222")
    r = world.run()
    assert r.returncode == 2 and "no recorded width" in r.stderr and "--probe-width" in r.stderr and world.sbatch_calls() == []
    # … and with it the resume is attempt 2/4 at the given width, the note naming the legacy source
    r = world.run("--probe-width", "4")
    out = r.stdout + r.stderr
    assert r.returncode == 0, out
    argv = world.sbatch_calls()[0]["argv"]
    assert argv[argv.index("tp1") + 1:] == ["2", "4", "auto"] and "--gres=gpu:4" in argv
    assert "RESULT head" in out and "attempt=1/4 job=22222222" in out and "hpo.lora.probe.a2" in out


def test_an_attempt_number_the_ledger_already_binds_is_skipped(world):
    """The front door submitted attempt 3 earlier and that job left the queue before its carrier wrote the attempt record: the
    funnel would skip `…probe.a3` as already submitted — the derivation takes a4 and says why."""
    world.lay()
    world.ledger.parent.mkdir(parents=True, exist_ok=True); world.ledger.write_text("")
    ST.write_binding(str(world.ledger), ST.submitted_record(f"{LWID}.a3", "99999999", "cluster_a", f"{LWID}.a3@cluster_a@1", NAME))
    r = world.run()
    out = r.stdout + r.stderr
    assert r.returncode == 0, out
    calls = world.sbatch_calls()
    assert len(calls) == 1 and calls[0]["argv"][calls[0]["argv"].index("tp1") + 1:] == ["4", "4", "auto"]
    assert "already binds attempt 3" in out and "hpo.lora.probe.a4" in out and "skipped" not in out


def test_a_menu_over_an_incomplete_checkpoint_is_stale_never_fetched_never_marked_and_resumes(world):
    """MENU_STALE is handled like INCOMPLETE_*: never fetched, never marked; with no job the record is resumed (the carrier
    rewrites the verdict + menu from the current results); with a job queued/running nothing is submitted and the exit is 0."""
    world.lay(done=3, complete=False, menu=True, verdict=True)
    r = world.run()
    out = r.stdout + r.stderr
    assert r.returncode == 0, out
    assert "STALE" in out and "NOT fetched, NOT marked" in out and "resuming the record" in out
    assert not (world.state / "allowed_pairs.json").exists() and not (world.tmp / "state" / MODEL / "03_hpo" / "DONE.lora.probe").exists()
    calls = world.sbatch_calls()
    assert len(calls) == 1 and calls[0]["argv"][calls[0]["argv"].index("tp1") + 1:] == ["3", "4", "auto"]     # the resume, attempt 3/4
    assert f"submitted cluster_a {calls[0]['id']} hpo.lora.probe.a3" in out
    assert not any("rsync" in ln for ln in world.ssh_log.read_text().splitlines())
    # the resume is now PENDING under the probe's name: MENU_STALE with a job -> said, nothing fetched, nothing submitted, exit 0
    r2 = world.run()
    out2 = r2.stdout + r2.stderr
    assert r2.returncode == 0 and len(world.sbatch_calls()) == 1, out2
    assert "MENU_STALE" in out2 and "NOT fetched, nothing submitted" in out2 and not (world.state / "allowed_pairs.json").exists()


def _with_parity(rec: dict, pairs=((4, "wo"),)) -> dict:
    """A geometry record says BOTH facts — geometry certification AND adapter parity — per set and at record level;
    the tests build the record with the venv's `sisteer` (the main checkout, which predates the rule) and add the block by hand
    in the parser's shape (required keys status / basis / per_pair; per_pair exactly the set's allowed pairs)."""
    block = {"status": "verified", "basis": "hf_peft", "per_pair": [{"rank": r, "set": s, "status": "verified"} for r, s in pairs]}
    for sid in rec["sets"]:
        rec["sets"][sid]["adapter_parity"] = dict(block)
    rec["adapter_parity"] = dict(block)
    return rec


def _stamp(world: World) -> str:
    """Stamp the record's verdict + menu with its CURRENT result generation (the real reader's own algorithm) -> the generation."""
    sys.path.insert(0, str(REPO / "scripts" / "sdfbase"))
    import hpo_probe_status as hps  # noqa: E402
    gen = hps.record_generation(world.record)["generation"]
    for name in ("probe_verdict.json", "allowed_pairs.json"):
        d = json.loads((world.record / name).read_text()); d[hps.GENERATION_KEY] = gen
        (world.record / name).write_text(json.dumps(d))
    return gen


def test_ready_stamped_fetches_the_geometry_record_of_the_same_generation_and_refuses_a_superseded_fold(world):
    """A STAMPED READY record (verdict + menu carry the record's result_generation — 170c98a29's stamp): fetched, marked with
    that generation; its geometry record travels to the site only when ITS stamp is the same generation — a fold of a
    superseded result set refuses by name and nothing is marked."""
    world.lay(done=7, complete=True, menu=True, verdict=True)
    gen = _stamp(world)
    try:
        from sisteer.capeval.serving import geometry as g
    except ImportError:                                                     # pragma: no cover — the geometry package is the site copy's validator
        pytest.skip("sisteer.capeval.serving.geometry not importable")
    budget = g.ModelBudget(model_key=MODEL, batched_token_budget=8 * 4096, certified=True, source=g.MEASURED_BUDGET_SOURCE,
                           basis=g.BudgetBasis(method=f"{g.SITE_GEOMETRY_METHOD}: test", max_model_len=4096, max_num_seqs=8, gpu_memory_utilization=0.9,
                                               kv_tokens_reported=None, probe_job_id="11111111", measured_on="2026-09-30", source="test"))
    sets = {"wo": {"lora_modules": ["o_proj"], "lora_max_module_dim": 4096, "budget": budget, "allowed_pairs": [[4, "wo"]],
                   "certified_by": ["L01"], "module_dim_launches": {"L01": 4096}, "budget_launches": {"L01": {"S": 8, "L": 4096, "batched_tokens": 8 * 4096}}}}
    stale = _with_parity(g.site_geometry_record(model_key=MODEL, hf_id=SRV_ID, revision="0" * 40, written_utc="2026-09-30T02:00:00Z", sets=sets,
                                                provenance={"probe_job_id": "11111111", "probe_record": str(world.record), "host": "TESTHOST", "result_generation": "f" * 64,
                                                            "attestation_sha256": {"L01": {"file": None, "sha256": "ab" * 32}}}))
    (world.record / "geometry_record.json").write_text(json.dumps(stale))
    r = world.run()
    out = r.stdout + r.stderr
    assert r.returncode == 2 and "is stamped for result generation ffffffffffff" in r.stderr and "superseded result set" in r.stderr, out
    assert not (world.site / "geometry" / f"{MODEL}.json").exists() and not (world.tmp / "state" / MODEL / "03_hpo" / "DONE.lora.probe").exists()
    current = _with_parity(g.site_geometry_record(model_key=MODEL, hf_id=SRV_ID, revision="0" * 40, written_utc="2026-09-30T02:00:00Z", sets=sets,
                                                  provenance={"probe_job_id": "11111111", "probe_record": str(world.record), "host": "TESTHOST", "result_generation": gen,
                                                              "attestation_sha256": {"L01": {"file": None, "sha256": "ab" * 32}}}))
    (world.record / "geometry_record.json").write_text(json.dumps(current))
    r = world.run()
    out = r.stdout + r.stderr
    assert r.returncode == 0, out
    assert "stamped_match" in out and f"generation={gen}" in (world.tmp / "state" / MODEL / "03_hpo" / "DONE.lora.probe").read_text()
    site_copy = world.site / "geometry" / f"{MODEL}.json"
    assert site_copy.is_file() and json.loads(site_copy.read_text())["provenance"]["result_generation"] == gen
    assert "carries no result_generation stamp" not in out


def test_ready_fetches_marks_with_the_generation_and_never_overwrites_a_newer_mirror(world):
    world.lay(done=7, complete=True, menu=True, verdict=True)              # UNSTAMPED verdict + menu: verified by recomputation
    r = world.run()
    out = r.stdout + r.stderr
    assert r.returncode == 0, out
    assert (world.state / "allowed_pairs.json").is_file() and (world.state / "probe_verdict.json").is_file()
    assert (world.state / "launches" / "L07.json").is_file()
    marker = (world.tmp / "state" / MODEL / "03_hpo" / "DONE.lora.probe").read_text()
    fetched = json.loads((world.state / "probe_fetch.json").read_text())
    assert f"generation={fetched['generation']}" in marker and len(fetched["generation"]) == 64 and fetched["n_results"] == 7
    assert "written_utc=2026-09-30T02:00:00Z" in marker and "probe_job_id=11111111" in marker
    assert "READY on cluster_a (result generation" in out and "recomputed_match" not in r.stderr
    assert world.sbatch_calls() == [] and "no geometry_record" in out          # the synthetic record folded no geometry: a warning, not a refusal
    # a second run: the DONE marker is REVALIDATED (one status read of the record) and stands; nothing re-fetched
    n_ssh = len(world.ssh_log.read_text().splitlines()); before = (world.state / "allowed_pairs.json").read_bytes()
    r2 = world.run()
    assert r2.returncode == 0 and "DONE marker present and the local menu is the record's current result generation" in r2.stdout, r2.stdout + r2.stderr
    new_ssh = world.ssh_log.read_text().splitlines()[n_ssh:]
    assert len([ln for ln in new_ssh if "--record" in ln]) == 1 and "recomputed_match" in out
    assert (world.state / "allowed_pairs.json").read_bytes() == before and not (world.state / "superseded").exists()
    # a NEWER mirror is never overwritten: plant a later menu locally, drop the marker, re-run -> refused, the local file intact
    (world.tmp / "state" / MODEL / "03_hpo" / "DONE.lora.probe").unlink()
    newer = json.loads((world.state / "allowed_pairs.json").read_text()); newer["written_utc"] = "2026-10-01T00:00:00Z"
    (world.state / "allowed_pairs.json").write_text(json.dumps(newer))
    r3 = world.run()
    assert r3.returncode == 2 and "is NEWER than cluster_a's menu" in r3.stderr, r3.stdout + r3.stderr
    assert json.loads((world.state / "allowed_pairs.json").read_text())["written_utc"] == "2026-10-01T00:00:00Z"
    _scan(world, out + r3.stderr)


def test_not_started_submits_a_fresh_probe_and_probe_width_rides_the_line(world):
    r = world.run()
    out = r.stdout + r.stderr
    assert r.returncode == 0, out
    calls = world.sbatch_calls()
    assert len(calls) == 1 and f"submitted cluster_a {calls[0]['id']} hpo.lora.probe" in out
    argv = calls[0]["argv"]; i = argv.index("scripts/sbatch/serving_probe.sbatch")
    assert argv[i + 1:] == [MODEL, SRV_ID, argv[i + 3]] and not any(f.startswith("--gres") for f in argv)     # the header's width
    w2 = World(world.tmp / "w2")
    r = w2.run("--probe-width", "4")
    assert r.returncode == 0, r.stdout + r.stderr
    argv = w2.sbatch_calls()[0]["argv"]
    assert "--gres=gpu:4" in argv and "--cpus-per-task=64" in argv and "--mem=512G" in argv and argv[-3:] == [MODEL, SRV_ID, argv[-1]]


def test_the_carriers_attempt_record_block_writes_the_json_the_resume_reads(tmp_path):
    """RUN the carrier's attempt-record fragment under bash: one JSON object with the counter, cap, width and positionals; a
    non-integer counter refuses; the fragment sits AFTER the job's log line (outside the width block the parallel tests run)."""
    text = (REPO / "scripts/sbatch/serving_probe.sbatch").read_text()
    frag = text[text.index("# --- THE ATTEMPT RECORD"):text.index('log "attempt record:')]
    assert text.index("# --- THE ATTEMPT RECORD") > text.index('log "model=$MODEL_KEY')
    assert "probe_attempt.json" in text[text.index("for f in probe_plan.json"):text.index("mirror-figs.sh")]
    out = tmp_path / "rec"; out.mkdir()

    def run(attempt, cap):
        script = (f'set -u; OUT={out}; ATTEMPT={attempt}; MAX_ATTEMPTS={cap}; NGPU=4; PARALLEL=auto; MODE=tp1; TP=1; SETS=wo,down; RANKS=4,16; '
                  f'MODEL_KEY=qwen35_4b_demo; SERVED_ID=org/toy; REVISION={"0" * 40}; SLURM_JOB_ID=4242; SLURM_JOB_NAME=hpo-m-lora-probe; '
                  f'HEAD_SHA={"a" * 40}; JOB_QOS=high; TIMELIMIT=04:00:00; SLURM_CPUS_PER_TASK=64; SLURM_MEM_PER_NODE=524288; FINAL_RC=; '
                  'log(){ echo "[log] $*"; }\n' + frag)
        return subprocess.run(["bash", "-c", script], capture_output=True, text=True, env={"PATH": os.environ["PATH"], "HOME": str(tmp_path)}, timeout=60)

    r = run(2, 4)
    assert r.returncode == 0, r.stdout + r.stderr
    rec = json.loads((out / "probe_attempt.json").read_text())
    assert rec["attempt"] == 2 and rec["max_attempts"] == 4 and rec["gpus"] == 4 and rec["parallel"] == "auto" and rec["mode"] == "tp1"
    assert rec["sets"] == "wo,down" and rec["ranks"] == "4,16" and rec["job_id"] == "4242" and rec["sha"] == "a" * 40 and rec["qos"] == "high"
    assert not list(out.glob("*.tmp.*"))
    sys.path.insert(0, str(REPO / "scripts" / "sdfbase"))
    import hpo_probe_resume as hpr  # noqa: E402
    spec = hpr.derive({"sets": ["wo", "down"], "ranks": [4, 16], "tp": 1}, attempt_json=json.dumps(rec), legacy_line="", width="record",
                      ledger=None, lwid_prefix=None)
    assert spec["attempt"] == 3 and spec["width"] == "4" and spec["mode"] == "tp1"
    r = run("two", 4)
    assert r.returncode == 2 and "positionals 7/8 attempt/max_attempts must be positive integers" in r.stdout


def test_a_reopened_record_archives_the_local_menu_and_the_site_copy_then_the_consuming_phase_refuses(world):
    """After a READY fetch the record is REOPENED on the cluster (a result superseded, the checkpoint incomplete).
    `--phase design` revalidates first: the local menu / verdict / fetch record / DONE marker are archived under
    superseded/<utc>/ (nothing deleted), the site's geometry copy moves to geometry/superseded/<utc>/, and design refuses by name;
    `--phase probe` then archives nothing twice, sees MENU_STALE with no job and RESUMES the record."""
    try:
        from sisteer.capeval.serving import geometry as g
    except ImportError:                                                     # pragma: no cover
        pytest.skip("sisteer.capeval.serving.geometry not importable")
    world.lay(done=7, complete=True, menu=True, verdict=True)
    gen = _stamp(world)
    budget = g.ModelBudget(model_key=MODEL, batched_token_budget=8 * 4096, certified=True, source=g.MEASURED_BUDGET_SOURCE,
                           basis=g.BudgetBasis(method=f"{g.SITE_GEOMETRY_METHOD}: test", max_model_len=4096, max_num_seqs=8, gpu_memory_utilization=0.9,
                                               kv_tokens_reported=None, probe_job_id="11111111", measured_on="2026-09-30", source="test"))
    sets = {"wo": {"lora_modules": ["o_proj"], "lora_max_module_dim": 4096, "budget": budget, "allowed_pairs": [[4, "wo"]],
                   "certified_by": ["L01"], "module_dim_launches": {"L01": 4096}, "budget_launches": {"L01": {"S": 8, "L": 4096, "batched_tokens": 8 * 4096}}}}
    (world.record / "geometry_record.json").write_text(json.dumps(_with_parity(g.site_geometry_record(
        model_key=MODEL, hf_id=SRV_ID, revision="0" * 40, written_utc="2026-09-30T02:00:00Z", sets=sets,
        provenance={"probe_job_id": "11111111", "probe_record": str(world.record), "host": "TESTHOST", "result_generation": gen,
                    "attestation_sha256": {"L01": {"file": None, "sha256": "ab" * 32}}}))))
    r = world.run()
    assert r.returncode == 0, r.stdout + r.stderr
    marker = world.tmp / "state" / MODEL / "03_hpo" / "DONE.lora.probe"
    site_copy = world.site / "geometry" / f"{MODEL}.json"
    assert marker.is_file() and site_copy.is_file() and (world.state / "allowed_pairs.json").is_file()
    # the record is CURRENT: design revalidates, proceeds, and fails on ITS OWN input (no recon.json yet)
    r = world.run("--phase", "design")
    out = r.stdout + r.stderr
    assert r.returncode == 2 and "local menu is the record's current result generation" in out and "design proceeds" in out, out
    assert "design needs" in r.stderr and "recon.json" in r.stderr and marker.is_file() and site_copy.is_file()
    # REOPEN the record on the cluster: one result superseded (moved out of launches/), the checkpoint incomplete again
    (world.record / "launches" / "superseded").mkdir()
    (world.record / "launches" / "L07.json").rename(world.record / "launches" / "superseded" / "L07.json")
    (world.record / "probe_progress.json").write_text(json.dumps(_progress(6, False)))
    r = world.run("--phase", "design")
    out = r.stdout + r.stderr
    assert r.returncode == 2, out
    assert f"design: the local menu for {MODEL} was NOT the record's current result generation" in r.stderr
    assert "record MENU_STALE" in r.stderr and "archived under" in out and "nothing deleted" in out and "DONE marker is cleared" in out
    assert not marker.exists() and not (world.state / "allowed_pairs.json").exists() and not (world.state / "probe_verdict.json").exists()
    arch = sorted((world.state / "superseded").iterdir())
    assert len(arch) == 1 and {p.name for p in arch[0].iterdir()} >= {"allowed_pairs.json", "probe_verdict.json", "probe_fetch.json", "geometry_record.json", "DONE.lora.probe"}
    assert json.loads((arch[0] / "allowed_pairs.json").read_text())["probe_job_id"] == "11111111"        # archived, byte for byte
    assert not site_copy.exists()
    moved = sorted((world.site / "geometry" / "superseded").iterdir())
    assert len(moved) == 1 and (moved[0] / f"{MODEL}.json").is_file() and json.loads((moved[0] / f"{MODEL}.json").read_text())["provenance"]["result_generation"] == gen
    assert world.sbatch_calls() == []                                                 # a consuming phase never submits the probe
    # --phase probe on the reopened record: nothing local left to archive; MENU_STALE + no job -> the resume goes out
    r = world.run()
    out = r.stdout + r.stderr
    assert r.returncode == 0, out
    assert "STALE" in out and "resuming the record" in out and len(world.sbatch_calls()) == 1 and not marker.exists()
    assert len(sorted((world.state / "superseded").iterdir())) == 1                  # no second archive dir for nothing
    _scan(world, out)


def test_a_done_marker_is_revalidated_on_the_probe_phase_and_a_new_generation_is_refetched_in_place(world):
    """`--phase probe` with a DONE marker reads the record anyway: a record re-verdicted at a NEW generation (READY, different
    generation) archives the local copies and is re-fetched in the same invocation; the marker then carries the new generation."""
    world.lay(done=7, complete=True, menu=True, verdict=True)
    r = world.run()
    assert r.returncode == 0, r.stdout + r.stderr
    marker = world.tmp / "state" / MODEL / "03_hpo" / "DONE.lora.probe"
    gen1 = json.loads((world.state / "probe_fetch.json").read_text())["generation"]
    # the record moves on: a launch re-run (its result file changes), the verdict + menu rewritten and STAMPED for the new set
    (world.record / "launches" / "L07.json").write_text(json.dumps({"launch_id": "L07", "status": "passed", "host": "TESTHOST", "rerun": True}))
    for name in ("probe_verdict.json", "allowed_pairs.json"):
        d = json.loads((world.record / name).read_text()); d["written_utc"] = "2026-09-30T03:00:00Z"; (world.record / name).write_text(json.dumps(d))
    gen2 = _stamp(world)
    assert gen2 != gen1
    r = world.run()
    out = r.stdout + r.stderr
    assert r.returncode == 0, out
    assert "archived under" in out and "READY on cluster_a" in out and f"generation={gen2}" in marker.read_text()
    assert json.loads((world.state / "probe_fetch.json").read_text())["generation"] == gen2
    assert json.loads((world.state / "allowed_pairs.json").read_text())["written_utc"] == "2026-09-30T03:00:00Z"
    arch = sorted((world.state / "superseded").iterdir())
    assert len(arch) == 1 and json.loads((arch[0] / "probe_fetch.json").read_text())["generation"] == gen1
    # and a plain re-run with nothing changed: the marker is revalidated (one status read) and stands
    n_ssh = len(world.ssh_log.read_text().splitlines())
    r = world.run()
    assert r.returncode == 0 and "DONE marker present and the local menu is the record's current result generation" in r.stdout
    assert len([ln for ln in world.ssh_log.read_text().splitlines()[n_ssh:] if "--record" in ln]) == 1 and world.sbatch_calls() == []


# --------------------------------------------------------------------------------------------- --reopen (item 1)


def _reopen_world(world: World) -> Path:
    """A READY record (complete checkpoint, unstamped verdict + menu: verified by recomputation) with a staged overlay on the
    cluster and a site copy — what a reopen must withdraw before the record is touched."""
    world.lay(done=7, complete=True, menu=True, verdict=True)
    geo = world.shared / "exp" / "reproduce" / "geometry"; geo.mkdir(parents=True, exist_ok=True)
    (geo / f"{MODEL}.json").write_text("{}")
    (world.site / "geometry").mkdir(exist_ok=True); (world.site / "geometry" / f"{MODEL}.json").write_text("{}")
    return geo


def _the_one_resume_with_a_receipt(world: World, out: str) -> str:
    """ONE sbatch — the resume (attempt 3/4 of the record, `parallel` auto) — with the receipt's nonce as positional 10; the receipt
    stands in the record for the record's generation, naming that nonce and the acknowledged cluster; returns the nonce."""
    calls = world.sbatch_calls()
    assert len(calls) == 1, out
    argv = calls[0]["argv"]; tail = argv[argv.index("tp1") + 1:]
    assert tail[:3] == ["3", "4", "auto"] and len(tail) == 4 and re.fullmatch(r"[0-9a-f]{32}", tail[3]), tail
    assert "--resume" not in argv                                                            # the carrier derives the resume itself
    # SERVING_PROBE_OUT=<record> rides as the funnel's --env prefix (the fake sbatch keeps only the SISTEER_* family): the attempt record
    attempts = [json.loads(ln) for ln in (world.tmp / "state" / MODEL / "attempts.jsonl").read_text().splitlines() if ln.strip()]
    att = [a for a in attempts if a.get("event") == "attempt"]
    assert len(att) == 1 and att[0]["logical_work_id"] == f"{LWID}.a3" and att[0]["spec"]["env"] == {"SERVING_PROBE_OUT": str(world.record)}
    assert att[0]["args"][6:] == ["3", "4", "auto", tail[3]], att[0]["args"]
    nonce = tail[3]
    receipts = sorted((world.record / "reopen_receipts").glob("*.json"))
    assert len(receipts) == 1 and receipts[0].name.endswith(f".{nonce}.json"), receipts
    doc = json.loads(receipts[0].read_text()); gen = receipts[0].name.split(".")[0]
    assert doc["nonce"] == nonce and doc["generation"] == gen and len(gen) == 64 and doc["record"] == str(world.record)
    assert [a["cluster"] for a in doc["acknowledged"]] == ["cluster_a"] and doc["probe_cluster"] == "cluster_a"
    assert f"submitted cluster_a {calls[0]['id']} hpo.lora.probe.a3" in out
    return nonce


def test_reopen_of_a_ready_record_revokes_receipts_and_resumes_and_neither_fetches_nor_skips(world):
    """The production line, executed: `--phase probe --reopen` on a READY record revokes the site copy and the cluster's staged overlay
    (archived under superseded/, REVOKED markers written), writes the revocation receipt into the record, submits ONE resume carrying
    the nonce, and neither fetches the menu (no local copy, no DONE marker) nor skips. While that resume is queued, a second `--reopen`
    refuses by name: the record is held — nothing more revoked, nothing more submitted, the one receipt still stands."""
    geo = _reopen_world(world)
    r = world.run("--reopen")
    out = r.stdout + r.stderr
    assert r.returncode == 0, out
    assert "--reopen: reopening the READY record" in out and "revocation receipt written into the record" in out and "positional 10" in out
    assert "READY on cluster_a (result generation" not in out and "fetched to" not in out and "skipping" not in out, out
    assert not (world.state / "allowed_pairs.json").exists() and not (world.state / "probe_verdict.json").exists()
    assert not (world.tmp / "state" / MODEL / "03_hpo" / "DONE.lora.probe").exists()
    assert not any("rsync" in ln for ln in world.ssh_log.read_text().splitlines())                    # nothing fetched
    nonce = _the_one_resume_with_a_receipt(world, out)
    # revoked: the cluster's copy archived + marked, the site copy archived + marked, both for the record's generation
    assert (geo / f"{MODEL}.REVOKED.json").is_file() and not (geo / f"{MODEL}.json").exists() and list(geo.glob(f"superseded/*/{MODEL}.json"))
    site_geo = world.site / "geometry"
    assert (site_geo / f"{MODEL}.REVOKED.json").is_file() and not (site_geo / f"{MODEL}.json").exists() and list(site_geo.glob(f"superseded/*/{MODEL}.json"))
    marker = json.loads((geo / f"{MODEL}.REVOKED.json").read_text())
    assert marker["generation"] == sorted((world.record / "reopen_receipts").glob("*.json"))[0].name.split(".")[0] and "--phase probe" in marker["reason"]
    assert nonce not in marker["reason"]
    # the resume is PENDING under the probe's name: a second --reopen refuses by name — nothing revoked again, nothing submitted
    r2 = world.run("--reopen")
    out2 = r2.stdout + r2.stderr
    assert r2.returncode == 2 and "--reopen" in r2.stderr and "nothing revoked, nothing submitted" in r2.stderr, out2
    assert len(world.sbatch_calls()) == 1 and len(list((world.record / "reopen_receipts").glob("*.json"))) == 1
    assert len(list(geo.glob("superseded/*"))) == 1 and len(list(site_geo.glob("superseded/*"))) == 1
    _scan(world, out + out2)


def test_reopen_of_a_done_and_current_record_archives_the_local_copies_and_resumes(world):
    """A record already fetched (DONE, the local menu current): `--reopen` archives the local menu / verdict / fetch record + the DONE
    marker under the mirror's superseded/, revokes, receipts and resumes — the same one sbatch with the nonce; nothing is skipped."""
    geo = _reopen_world(world)
    r0 = world.run()
    assert r0.returncode == 0 and (world.state / "allowed_pairs.json").is_file() and (world.tmp / "state" / MODEL / "03_hpo" / "DONE.lora.probe").is_file(), r0.stdout + r0.stderr
    assert world.sbatch_calls() == []
    r = world.run("--reopen")
    out = r.stdout + r.stderr
    assert r.returncode == 0, out
    assert "DONE marker present and the local menu is the record's current result generation" in out     # the revalidation ran first …
    assert "reopening the READY record" in out and "skipping" in out.split("reopening")[0]                 # … then the verb, not the skip
    assert not (world.state / "allowed_pairs.json").exists() and not (world.tmp / "state" / MODEL / "03_hpo" / "DONE.lora.probe").exists()
    arch = sorted(p for p in (world.state / "superseded").iterdir() if p.is_dir())
    assert len(arch) == 1 and {p.name for p in arch[0].iterdir()} >= {"allowed_pairs.json", "probe_verdict.json", "probe_fetch.json", "DONE.lora.probe"}
    _the_one_resume_with_a_receipt(world, out)
    assert (geo / f"{MODEL}.REVOKED.json").is_file() and (world.site / "geometry" / f"{MODEL}.REVOKED.json").is_file()
    _scan(world, out)


def test_reopen_of_an_incomplete_record_resumes_it_with_nothing_revoked(world):
    """`--reopen` on an INCOMPLETE_STALLED record (launches outstanding, no job) RESUMES it exactly as the plain probe
    phase does — attempt+1 through the funnel, no receipt, no revocation (the record is not complete, so no overlay is withdrawn); it
    used to refuse by name, which left a PARTIAL record at its attempt cap with no front-door path to a widened resume."""
    world.lay()                                                                          # done 3/7, no job: INCOMPLETE_STALLED
    geo = world.shared / "exp" / "reproduce" / "geometry"; geo.mkdir(parents=True); (geo / f"{MODEL}.json").write_text("{}")
    r = world.run("--reopen")
    out = r.stdout + r.stderr
    assert r.returncode == 0 and "INCOMPLETE_STALLED — launches outstanding, no job: resuming it (no receipt" in out and "as attempt 3/4" in out, out
    calls = world.sbatch_calls(); assert len(calls) == 1 and not (world.record / "reopen_receipts").exists()
    assert (geo / f"{MODEL}.json").read_text() == "{}" and not (geo / f"{MODEL}.REVOKED.json").exists() and not (world.site / "geometry" / f"{MODEL}.REVOKED.json").exists()
    _scan(world, out)


def test_reopen_dry_run_describes_the_revocation_the_receipt_and_the_resume_and_touches_nothing(world):
    """A dry run reads no record (03 says so: PSTATE is NOT_STARTED by construction), so the verb DESCRIBES the reopen — the site copy and
    each cluster's overlay it would revoke, the receipt, the resume with the nonce as positional 10 — refuses nothing and touches nothing."""
    geo = _reopen_world(world)
    r = world.run("--reopen", "--dry-run")
    out = r.stdout + r.stderr
    assert r.returncode == 0, out
    assert "DRY-RUN: --reopen: the record's state is not read under a dry run" in out and "REFUSED" not in out
    assert "DRY-RUN: geometry revoke" in out and f"{MODEL}.json on cluster_a" in out and "DRY-RUN: geometry revoke the site copy" in out
    assert "DRY-RUN: the revocation receipt" in out and "positional 10" in out and "serving_probe.sbatch" in out and "0" * 32 in out
    assert world.sbatch_calls() == [] and not (world.record / "reopen_receipts").exists()
    assert (geo / f"{MODEL}.json").is_file() and not (geo / f"{MODEL}.REVOKED.json").exists()
    assert (world.site / "geometry" / f"{MODEL}.json").is_file() and not (world.site / "geometry" / f"{MODEL}.REVOKED.json").exists()
    assert not (world.state / "allowed_pairs.json").exists()
    _scan(world, out)


def test_a_test_world_can_never_perform_a_real_repo_mirror(tmp_path):
    """The disk bomb of 2026-09-30: a front-door test world reached lib/mirror_repo.sh for real once per test (a ~6 GB checkout of the repo
    into the fake shared root) because its import-time MIRRORED_SHA fell behind a HEAD peers moved during the run. The World now reads
    HEAD per call and exports SISTEER_TEST_NO_MIRROR=1; mirror_repo.sh refuses by name under it, writing nothing."""
    world = World(tmp_path)
    env = world.env()
    assert env["SISTEER_TEST_NO_MIRROR"] == "1" and env["MIRRORED_SHA"] == subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO).stdout.strip()
    assert (world.shared / "sisteer-eval-repo" / env["MIRRORED_SHA"]).is_dir()
    mirror = tmp_path / "mirror"; eval_root = tmp_path / "eval"
    r = subprocess.run(["bash", str(REPRO / "lib" / "mirror_repo.sh"), "--ssh", f"local:{tmp_path / 'fake'}", "--mirror", str(mirror), "--eval-root", str(eval_root), "--sha", env["MIRRORED_SHA"]],
                       capture_output=True, text=True, env={**os.environ, "SISTEER_TEST_NO_MIRROR": "1"}, cwd=str(REPO), timeout=120)
    assert r.returncode == 3 and "REFUSED: mirror_repo.sh under SISTEER_TEST_NO_MIRROR=1" in r.stderr and "a test world never mirrors the real repo" in r.stderr
    assert not mirror.exists() and not eval_root.exists()                                       # nothing written
    # a dry run is still described (the dry-run tests read it)
    r = subprocess.run(["bash", str(REPRO / "lib" / "mirror_repo.sh"), "--ssh", f"local:{tmp_path / 'fake'}", "--mirror", str(mirror), "--eval-root", str(eval_root), "--sha", env["MIRRORED_SHA"], "--dry-run"],
                       capture_output=True, text=True, env={**os.environ, "SISTEER_TEST_NO_MIRROR": "1"}, cwd=str(REPO), timeout=120)
    assert r.returncode == 0 and not mirror.exists(), r.stdout + r.stderr
