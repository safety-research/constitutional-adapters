"""f22_live_attackers waits for the pool's DECLARED membership instead of snapshotting it.

THE DEFECT (review finding F6, 2026-09-18). The old body globbed `$WAVE_ROOT/attackers/*.json`
ONCE and bound the driver round-robin to whatever happened to be serving at that instant. A target
that reached the line while the pool was still loading weights saw ONE member and pinned EVERY
driver of the wave to it, with no rebalancing afterwards. At the measured 70-100 drivers/member
knee a 36-driver job on one member queues for hours beside two idle GPUs, and nothing in the run
record said so. The snapshot is not the bug; taking it before the pool is up is.

THE MECHANISM. attacker_pool.sbatch writes `<jobid>.pending` at start and its cleanup trap removes
both markers at exit, so the union of `*.json` and `*.pending` job ids IS the declared
pool. f22_live_attackers waits for every declared member to serve, drops only members Slurm says
are definitively gone, records the binding to SERVE_DIR/attacker_binding.json, and warns loudly
when it proceeds under strength.

THE FAIL-SAFE THAT MUST HOLD. `squeue` exits 1 for BOTH "that job is over" and "I could not reach
the controller" (measured on cluster A 2026-09-18: a finished job and a broken SLURM_CONF both give
rc=1). Only the definitive "Invalid job id specified" may drop a member. Any other failure is
undecidable, and an undecidable member is KEPT -- otherwise one slurmctld hiccup shrinks a healthy
pool to nothing and refuses a job whose attackers are all serving fine.

Every case here is a KNOWN-BAD input: the test asserts the new code CHANGES the outcome, so it
would fail against the old snapshot body.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
COMMON = REPO / "scripts/sdfbase/full22/common.sh"
MODEL = "huihui-ai/Huihui-Qwen3.6-27B-abliterated"


def _functions() -> str:
    """The REAL shipped text of the two functions, not a copy of them."""
    src = COMMON.read_text()
    out = []
    for name in ("f22_declared_attackers", "f22_live_attackers"):
        m = re.search(rf"(?m)^{name}\(\)\{{.*?^\}}\n", src, re.S)
        assert m, f"{name} not found in {COMMON} -- the test must track the shipped function"
        out.append(m.group(0))
    return "\n".join(out)


def _harness(tmp_path: Path, *, declared, serving, squeue_mode: dict, env=None, manifests=None) -> tuple[int, str, Path]:
    """Run f22_live_attackers over a synthetic wave root.

    declared: {jobid: "json"|"pending"}   the markers on disk
    serving:  set of jobids whose url answers /v1/models with the attacker model
    squeue_mode: {jobid: "RUNNING"|"gone"|"outage"}
    """
    wave = tmp_path / "wave"; (wave / "attackers").mkdir(parents=True)
    serve = tmp_path / "serve"; serve.mkdir()
    for jid, kind in declared.items():
        rec = wave / "attackers" / f"{jid}.{kind}"
        body = {"url": f"http://gpu-{jid}:8000", "job_id": str(jid)}
        man = (manifests or {}).get(jid)
        if man is not None:
            body["serving_manifest"] = str(man)
        rec.write_text(json.dumps(body))
    # a .STOP sentinel must never be read as a member
    (wave / "attackers" / "999.STOP").write_text("")

    bindir = tmp_path / "bin"; bindir.mkdir()
    squeue = bindir / "squeue"
    squeue.write_text(
        "#!/bin/bash\n"
        "for a in \"$@\"; do case \"$prev\" in -j) id=$a;; esac; prev=$a; done\n"
        + "".join(
            f'if [ "$id" = "{jid}" ]; then '
            + {"RUNNING": 'echo RUNNING; exit 0',
               "gone": 'echo "slurm_load_jobs error: Invalid job id specified" >&2; exit 1',
               "outage": 'echo "squeue: error: s_p_parse_file: cannot stat file: No such file" >&2; exit 1',
               }[mode]
            + "; fi\n"
            for jid, mode in squeue_mode.items()
        )
        + 'echo "slurm_load_jobs error: Invalid job id specified" >&2; exit 1\n'
    )
    squeue.chmod(0o755)
    curl = bindir / "curl"
    ok = " ".join(f"http://gpu-{j}:8000/v1/models" for j in serving)
    curl.write_text(
        "#!/bin/bash\nurl=\"${@: -1}\"\n"
        f"for u in {ok}; do [ \"$url\" = \"$u\" ] && {{ echo '{{\"data\":[{{\"id\":\"{MODEL}\"}}]}}'; exit 0; }}; done\n"
        "exit 0\n"
    )
    curl.chmod(0o755)

    cfg = tmp_path / "cfg.yaml"; cfg.write_text(f"attacker:\n  model: {MODEL}\n")
    script = tmp_path / "run.sh"
    script.write_text(
        "set -u\n"
        "f22_log(){ echo \"LOG $*\"; }\n"
        "f22_fatal(){ echo \"FATAL $*\"; exit 9; }\n"
        f"MPY={sys.executable!r}\nCFG={str(cfg)!r}\nWAVE_ROOT={str(wave)!r}\nSERVE_DIR={str(serve)!r}\n"
        "F22_SERVING_PATH=lora_native\n"
        + _functions()
        + "\nf22_live_attackers\necho \"BOUND ${ATTACKERS[*]}\"\n"
        + 'echo "MANIFESTS ${ATTACKER_MANIFESTS[@]+${ATTACKER_MANIFESTS[@]}}"\n'
    )
    e = dict(os.environ, PATH=f"{bindir}:{os.environ['PATH']}")
    e.update(env or {})
    p = subprocess.run(["bash", str(script)], capture_output=True, text=True, env=e, timeout=120)
    return p.returncode, p.stdout + p.stderr, serve


def test_under_strength_pool_is_recorded_and_warned_not_silently_bound(tmp_path):
    """THE DEFECT: 3 members declared, 1 serving. The old body bound the 1 and said nothing."""
    rc, out, serve = _harness(
        tmp_path,
        declared={101: "json", 102: "pending", 103: "pending"},
        serving={101},
        squeue_mode={101: "RUNNING", 102: "RUNNING", 103: "RUNNING"},
        env={"F22_ATTACKER_WAIT_S": "0"},
    )
    assert rc == 0, out
    assert "UNDER STRENGTH" in out, out
    rec = json.loads((serve / "attacker_binding.json").read_text())
    assert rec["declared"] == 3 and rec["alive"] == 3 and rec["serving"] == 1, rec
    assert rec["urls"] == ["http://gpu-101:8000"], rec


def test_full_pool_binds_every_member(tmp_path):
    rc, out, serve = _harness(
        tmp_path,
        declared={101: "json", 102: "json", 103: "json"},
        serving={101, 102, 103},
        squeue_mode={101: "RUNNING", 102: "RUNNING", 103: "RUNNING"},
    )
    assert rc == 0, out
    assert "UNDER STRENGTH" not in out, out
    rec = json.loads((serve / "attacker_binding.json").read_text())
    assert rec["serving"] == 3 and len(rec["urls"]) == 3, rec


def test_squeue_outage_must_not_shrink_a_healthy_pool(tmp_path):
    """KNOWN-BAD: squeue rc=1 with a CONTROLLER error, not 'Invalid job id'.

    The undecidable member is kept, so the pool is still counted at 2 and the run proceeds.
    Dropping it would have left declared=1 and hidden that a member was unaccounted for.
    """
    rc, out, serve = _harness(
        tmp_path,
        declared={101: "json", 102: "json"},
        serving={101, 102},
        squeue_mode={101: "RUNNING", 102: "outage"},
        env={"F22_ATTACKER_WAIT_S": "0"},
    )
    assert rc == 0, out
    assert "could not answer" in out, out
    rec = json.loads((serve / "attacker_binding.json").read_text())
    assert rec["alive"] == 2 and rec["serving"] == 2, rec


def test_stale_marker_is_dropped_so_the_wave_does_not_wait_forever(tmp_path):
    """KNOWN-BAD: a .pending left behind by a member Slurm has finished (killed trap).

    It must be dropped -- otherwise every wave waits the full deadline on a dead marker -- and the
    run must proceed on the survivor WITHOUT an under-strength warning, since nothing is missing.
    """
    rc, out, serve = _harness(
        tmp_path,
        declared={101: "json", 102: "pending"},
        serving={101},
        squeue_mode={101: "RUNNING", 102: "gone"},
    )
    assert rc == 0, out
    assert "has left the queue" in out, out
    assert "UNDER STRENGTH" not in out, out
    rec = json.loads((serve / "attacker_binding.json").read_text())
    assert rec["declared"] == 2 and rec["alive"] == 1 and rec["serving"] == 1, rec


def test_require_full_refuses_rather_than_running_under_strength(tmp_path):
    rc, out, _ = _harness(
        tmp_path,
        declared={101: "json", 102: "pending"},
        serving={101},
        squeue_mode={101: "RUNNING", 102: "RUNNING"},
        env={"F22_ATTACKER_WAIT_S": "0", "F22_ATTACKER_REQUIRE_FULL": "1"},
    )
    assert rc == 9, out
    assert "REQUIRE_FULL" in out, out


def test_no_serving_member_still_refuses_with_the_serving_path_hint(tmp_path):
    rc, out, _ = _harness(
        tmp_path,
        declared={101: "pending"},
        serving=set(),
        squeue_mode={101: "RUNNING"},
        env={"F22_ATTACKER_WAIT_S": "0"},
    )
    assert rc == 9, out
    assert "no live attacker" in out and "F22_SERVING_PATH=lora_native" in out, out


def test_stop_sentinel_is_not_a_member(tmp_path):
    """A `<jobid>.STOP` file sits in the same directory and must never be counted as a member."""
    rc, out, serve = _harness(
        tmp_path,
        declared={101: "json"},
        serving={101},
        squeue_mode={101: "RUNNING"},
    )
    assert rc == 0, out
    rec = json.loads((serve / "attacker_binding.json").read_text())
    assert rec["declared"] == 1, rec


# --- the serving manifest, without which a store cannot be resumed --------------------------------
#
# THE DEFECT (2026-09-18, found the hard way). The launcher passed only --attacker-endpoint, so the
# driver called attacker_serving.driver_fields(None) and wrote `attacker_quantization: 'unrecorded'`
# into every store. The engine-identity join then REFUSES to resume such a store under any identity
# ("the identity that produced that work cannot be verified"), so a timeout is not a resume but a
# full restart of the rung -- and 754 rows were lost to exactly this. The pool record has carried the
# engine-verified `serving_manifest` path all along; nothing was reading it.

def test_manifests_are_index_parallel_with_the_urls(tmp_path):
    """A driver binds by index, so a mismatch would hand one member's identity to another's traffic."""
    (tmp_path / "m101.json").write_text("{}"); (tmp_path / "m102.json").write_text("{}")
    rc, out, _ = _harness(
        tmp_path,
        declared={101: "json", 102: "json"},
        serving={101, 102},
        squeue_mode={101: "RUNNING", 102: "RUNNING"},
        manifests={101: tmp_path / "m101.json", 102: tmp_path / "m102.json"},
    )
    assert rc == 0, out
    bound = [l for l in out.splitlines() if l.startswith("BOUND ")][0].split()[1:]
    mans = [l for l in out.splitlines() if l.startswith("MANIFESTS ")][0].split()[1:]
    assert len(bound) == len(mans) == 2, out
    for u, m in zip(bound, mans):
        assert u.split("//gpu-")[1].split(":")[0] in m, f"{u} bound to {m}"


def test_a_record_without_a_manifest_yields_an_empty_element_not_a_shifted_list(tmp_path):
    """An older record must not shift the parallel array — that would mis-bind every later driver."""
    m = tmp_path / "m102.json"; m.write_text("{}")
    rc, out, _ = _harness(
        tmp_path,
        declared={101: "json", 102: "json"},
        serving={101, 102},
        squeue_mode={101: "RUNNING", 102: "RUNNING"},
        manifests={102: m},          # 101 deliberately has none
    )
    assert rc == 0, out
    assert "only 1 of 2 attacker members carry a serving manifest" in out, out
    assert "CANNOT be resumed" in out, out


def test_a_manifest_path_that_does_not_exist_is_dropped_loudly(tmp_path):
    """A stale path must not be handed to a driver as if it were identity."""
    rc, out, _ = _harness(
        tmp_path,
        declared={101: "json"},
        serving={101},
        squeue_mode={101: "RUNNING"},
        manifests={101: tmp_path / "gone.json"},
    )
    assert rc == 0, out
    assert "does not exist" in out and "UNRESUMABLE" in out, out


def test_the_driver_passes_the_manifest_of_the_member_it_binds_to():
    """The sbatch bind line must index BOTH arrays with the same expression."""
    import re
    src = (REPO / "scripts/sdfbase/full22/target_attacks.sbatch").read_text()
    # 3efdfdbf7: each bind line branches on ASSIGNED_LIVE — the assigned member (`[0]`) or round-robin over the live members — so the
    # index expressions are compared BRANCH BY BRANCH (non-greedy, every occurrence), the manifests' `:-` default aside
    url = re.findall(r'url="\$\{ATTACKERS\[(.+?)\]\}"', src)
    man = re.findall(r'aman="\$\{ATTACKER_MANIFESTS\[(.+?)\]:-\}"', src)
    assert url and man and len(url) == len(man), f"bind lines not found or unpaired: {url!r} vs {man!r}"
    assert url == man, f"index mismatch: {url!r} vs {man!r}"
    assert url == ["0", "$((i % ${#ATTACKERS[@]}))"], url            # the assigned member, else round-robin over the live members
    assert '${ASERV[@]+"${ASERV[@]}"}' in src, "the --attacker-serving flag is not on the driver argv"


# --------------------------------------------------------------------------------------------
# F22_ATTACKER_JOBS: a consumer binds NAMED members instead of round-robining the whole pool.
#
# Why it matters (2026-09-22): a driver binds its URL at task start and never rediscovers, so
# round-robin is over "whatever happened to be live then" and submission ORDER becomes
# load-bearing. Pinning lets targets go out piecemeal as members come online. Three properties
# are load-bearing and each is a KNOWN-BAD input below:
#   1. an unknown pinned id REFUSES -- it must never fall back to the rest of the pool;
#   2. the refusal must reach the JOB, not die in a substitution subshell;
#   3. a non-Slurm member (a Modal pool publishes `<pool-name>.json`) is still a member.
# --------------------------------------------------------------------------------------------

GOALSET = REPO / "scripts/sdfbase/full22/target_attacks_goalset.sbatch"


def test_pin_selects_only_the_named_members(tmp_path):
    rc, out, serve = _harness(
        tmp_path,
        declared={101: "json", 102: "json", 103: "json"},
        serving={101, 102, 103},
        squeue_mode={101: "RUNNING", 102: "RUNNING", 103: "RUNNING"},
        env={"F22_ATTACKER_JOBS": "101,103"},
    )
    assert rc == 0, out
    rec = json.loads((serve / "attacker_binding.json").read_text())
    assert rec["declared"] == 2 and rec["serving"] == 2, rec
    assert sorted(rec["urls"]) == ["http://gpu-101:8000", "http://gpu-103:8000"], rec


def test_unpinned_still_enumerates_every_member(tmp_path):
    """The default must be byte-identical to the pre-pin behaviour."""
    rc, out, serve = _harness(
        tmp_path,
        declared={101: "json", 102: "json", 103: "json"},
        serving={101, 102, 103},
        squeue_mode={101: "RUNNING", 102: "RUNNING", 103: "RUNNING"},
    )
    assert rc == 0, out
    assert json.loads((serve / "attacker_binding.json").read_text())["declared"] == 3


def test_pinned_id_with_no_record_refuses_and_does_not_bind_the_rest(tmp_path):
    """KNOWN-BAD: pin 101,102 with only 101 on disk.

    THE DEFECT this pins down: f22_fatal inside f22_declared_attackers only exited the
    substitution subshell, and f22_log writes to STDOUT, so the refusal text was captured AS A
    MEMBER ID -- declared=(101 "[...] FATAL: ... 102 ...") -- and the job proceeded, bound to 101
    alone. That is the silent bind to an unnamed member the mechanism exists to prevent.
    """
    rc, out, serve = _harness(
        tmp_path,
        declared={101: "json"},
        serving={101},
        squeue_mode={101: "RUNNING"},
        env={"F22_ATTACKER_JOBS": "101,102", "F22_ATTACKER_WAIT_S": "0"},
    )
    assert rc == 9, f"the pin refusal must END the job, not be swallowed: {out}"
    assert "102" in out and "no record" in out, out
    assert "BOUND" not in out, f"refused, so nothing may bind: {out}"
    assert not (serve / "attacker_binding.json").exists(), "a refused job must record no binding"


def test_refusal_goes_to_stderr_so_stdout_carries_only_member_ids(tmp_path):
    """KNOWN-BAD, at the unit: f22_declared_attackers' stdout IS the member list.

    f22_log writes to stdout, so the old refusal (f22_fatal, i.e. f22_log + exit) put its text on
    the very channel the caller reads, and the caller parsed it as a member id. Streams are kept
    SEPARATE here -- the merged view in _harness cannot see this -- so the assertion is exactly
    "no diagnostic on the data channel", and rc must carry the refusal.
    """
    wave = tmp_path / "wave"; (wave / "attackers").mkdir(parents=True)
    (wave / "attackers" / "101.json").write_text(json.dumps({"url": "http://gpu-101:8000"}))
    script = tmp_path / "decl.sh"
    script.write_text(
        "set -u\n"
        "f22_log(){ echo \"[$(date -Is)] $*\"; }\n"          # the REAL f22_log: stdout
        "f22_fatal(){ f22_log \"FATAL: $*\"; exit 1; }\n"
        f"WAVE_ROOT={str(wave)!r}\n"
        + _functions().split("f22_live_attackers()")[0]
        + "\nf22_declared_attackers\n"
    )
    p = subprocess.run(["bash", str(script)], capture_output=True, text=True,
                       env=dict(os.environ, F22_ATTACKER_JOBS="101,102"), timeout=60)
    assert p.returncode == 1, f"the refusal must be carried by the return status: {p.stdout}{p.stderr}"
    assert p.stdout.strip() == "101", f"stdout must be member ids ONLY, got: {p.stdout!r}"
    assert "FATAL" not in p.stdout, f"the refusal leaked onto the data channel: {p.stdout!r}"
    assert "102" in p.stderr and "no record" in p.stderr, p.stderr


def test_non_slurm_member_is_declarable_and_never_probed_with_squeue(tmp_path):
    """KNOWN-BAD: a Modal pool record, whose stem is a pool name, not a job id.

    The `*[!0-9]*` filter dropped it silently, so a wave whose only member was Modal-hosted saw an
    EMPTY declared pool and fatalled -- for a pool serving perfectly well. squeue must not be asked
    about it either: `squeue -j <non-numeric>` answers "Invalid job id specified", the one reply
    that drops a member.
    """
    mid = "modal_relu-paired-atk-1"
    rc, out, serve = _harness(
        tmp_path,
        declared={mid: "json"},
        serving={mid},
        squeue_mode={},
        env={"F22_ATTACKER_WAIT_S": "0"},
    )
    assert rc == 0, out
    assert "is not a Slurm job id" in out, out
    rec = json.loads((serve / "attacker_binding.json").read_text())
    assert rec["declared"] == 1 and rec["serving"] == 1, rec
    assert rec["urls"] == [f"http://gpu-{mid}:8000"], rec


def test_pinned_non_slurm_id_with_no_record_refuses_too(tmp_path):
    """The requirement: the refusal is the point, and it must not have a non-numeric hole."""
    rc, out, _ = _harness(
        tmp_path,
        declared={"modal_atk_a": "json"},
        serving={"modal_atk_a"},
        squeue_mode={},
        env={"F22_ATTACKER_JOBS": "modal_atk_a,modal_atk_b", "F22_ATTACKER_WAIT_S": "0"},
    )
    assert rc == 9, f"an unknown pinned NON-NUMERIC id must refuse exactly as a numeric one does: {out}"
    assert "modal_atk_b" in out and "no record" in out, out
    # It must refuse for the RIGHT reason. Under the old numeric filter both modal ids were dropped,
    # the declared pool came out empty, and the job died the generic "no live attacker" death after
    # waiting -- a refusal by accident, with the wrong diagnosis and the wrong latency.
    assert "no live attacker" not in out, f"refused for the wrong reason: {out}"
    assert "BOUND" not in out, out


# --------------------------------------------------------------------------------------------
# The goalset target's PRE-WAIT, driven from the real shipped script text.
# --------------------------------------------------------------------------------------------

def _prewait(tmp_path, *, declared, serving, env=None, wait_passes=3) -> tuple[int, str]:
    """Drive target_attacks_goalset.sbatch's pre-wait over a synthetic wave root.

    Extracts the REAL atk_declare helper and wait loop from the shipped script, so a future edit
    that reintroduces the raw glob fails here.
    """
    src = GOALSET.read_text()
    m_fn = re.search(r"(?m)^atk_declare\(\)\{.*?^\}\n", src, re.S)
    assert m_fn, "atk_declare not found in the shipped goalset script"
    m_loop = re.search(r"(?m)^for \(\( w=0; w<ATK_WAIT_MIN; w\+\+ \)\); do.*?^done\n", src, re.S)
    assert m_loop, "the pre-wait loop not found in the shipped goalset script"

    wave = tmp_path / "wave"; (wave / "attackers").mkdir(parents=True)
    for jid, kind in declared.items():
        (wave / "attackers" / f"{jid}.{kind}").write_text(
            json.dumps({"url": f"http://gpu-{jid}:8000", "job_id": str(jid)}))

    bindir = tmp_path / "bin"; bindir.mkdir()
    curl = bindir / "curl"
    ok = " ".join(f"http://gpu-{j}:8000/v1/models" for j in serving)
    curl.write_text("#!/bin/bash\nurl=\"${@: -1}\"\n"
                    + f"for u in {ok}; do [ \"$url\" = \"$u\" ] && {{ echo '{{\"data\":[{{\"id\":\"{MODEL}\"}}]}}'; exit 0; }}; done\n"
                    + "exit 0\n")
    curl.chmod(0o755)

    script = tmp_path / "prewait.sh"
    script.write_text(
        "set -u\n"
        "f22_log(){ echo \"LOG $*\"; }\n"
        "f22_fatal(){ echo \"FATAL $*\"; exit 9; }\n"
        "f22_stop_requested(){ return 1; }\n"
        "sleep(){ echo \"SLEPT $1\"; }\n"
        f"MPY={sys.executable!r}\nWAVE_ROOT={str(wave)!r}\nserved_name={MODEL!r}\n"
        f"ATK_WAIT_MIN={wait_passes}\nRC=0\n"
        + _functions().split("f22_live_attackers()")[0]   # f22_declared_attackers only
        + m_fn.group(0) + m_loop.group(0)
        + 'echo "BROKE_AT n_live=$n_live"\n'
    )
    e = dict(os.environ, PATH=f"{bindir}:{os.environ['PATH']}")
    e.update(env or {})
    p = subprocess.run(["bash", str(script)], capture_output=True, text=True, env=e, timeout=120)
    return p.returncode, p.stdout + p.stderr


def test_prewait_does_not_break_early_on_an_unpinned_live_member(tmp_path):
    """KNOWN-BAD: my pinned member is still booting while ANOTHER lane's member is serving.

    The old loop counted every record and broke at n_live>=1, so the target spent none of its
    ATK_WAIT_MIN budget on its OWN member and handed the wait to f22_live_attackers' much shorter
    window. Here the pinned member never serves, so the pre-wait must use its whole budget.
    """
    rc, out = _prewait(
        tmp_path,
        declared={101: "json", 202: "json"},
        serving={101},                      # 101 is NOT pinned; 202 is pinned and not yet up
        env={"F22_ATTACKER_JOBS": "202"},
    )
    assert rc == 0, out
    assert "BROKE_AT n_live=0" in out, f"must not have been satisfied by member 101: {out}"
    assert out.count("SLEPT") == 3, f"the full budget must be spent on the pinned member: {out}"
    assert "pinned: 202" in out, out


def test_prewait_breaks_as_soon_as_the_pinned_member_serves(tmp_path):
    rc, out = _prewait(
        tmp_path,
        declared={101: "json", 202: "json"},
        serving={202},
        env={"F22_ATTACKER_JOBS": "202"},
    )
    assert rc == 0, out
    assert "BROKE_AT n_live=1" in out, out
    assert "SLEPT" not in out, f"it must break on the first pass, not sleep: {out}"


def test_prewait_unpinned_is_satisfied_by_any_member(tmp_path):
    """The default path is unchanged: any live member ends the wait."""
    rc, out = _prewait(tmp_path, declared={101: "json", 202: "json"}, serving={101})
    assert rc == 0, out
    assert "BROKE_AT n_live=1" in out and "SLEPT" not in out, out


def test_prewait_refuses_a_pinned_member_with_no_record(tmp_path):
    rc, out = _prewait(
        tmp_path,
        declared={101: "json"},
        serving={101},
        env={"F22_ATTACKER_JOBS": "404"},
    )
    assert rc == 9, f"must refuse rather than idle the GPUs: {out}"
    assert "404" in out, out
