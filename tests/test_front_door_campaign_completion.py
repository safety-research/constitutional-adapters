"""04_fit_objects.sh's completion predicates are campaign- and contract-aware: `registry_done` asks the registry
for the unit's object UNDER THIS RUN'S CAMPAIGN; `emission_state` reads `emitted` evidence as `stale` when it was filed under another
campaign or an older emission contract, and `published_unfolded` then schedules a REPLAY (resubmit; the job validates the kept object
and never retrains or rebuilds) instead of waiting for a fold that will never come. The shell is EXECUTED with the cluster stubbed by a
directory. Synthetic throughout."""
import json
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FRONT = REPO / "reproduce" / "04_fit_objects.sh"


def _fn(src: str, name: str) -> str:
    start = src.index(f"{name}(){{")
    depth, i = 0, start
    while True:
        c = src[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return src[start:i + 1]
        i += 1


def _run(tmp_path: Path, evidence: dict | None, *, campaign="sdfbase_full22", active=False):
    src = FRONT.read_text()
    d = tmp_path / "cell"; d.mkdir(exist_ok=True)
    for f in d.iterdir():
        f.unlink()
    if evidence is not None:
        (d / "registry_emission.json").write_text(json.dumps(evidence))
    script = f"""
CAMPAIGN={campaign}; EMISSION_CONTRACT=2; CLUSTER=c; REPRO_DRY_RUN=0
repro_python(){{ echo python3; }}
cluster_has(){{ [ -s "$1" ]; }}
cluster_cat(){{ cat "$1"; }}
cluster_job_active(){{ [ "{int(active)}" = 1 ]; }}
repro_note(){{ echo "note: $*"; }}
repro_fatal(){{ echo "FATAL: $*"; exit 7; }}
{_fn(src, "emission_state")}
{_fn(src, "published_unfolded")}
emission_state {d}
published_unfolded unit {d} jobname; echo "PU_RC=$?"
"""
    return subprocess.run(["bash", "-c", script], capture_output=True, text=True)


def test_registry_done_asks_for_the_units_object_under_this_runs_campaign():
    src = FRONT.read_text()
    fn = _fn(src, "registry_done")
    assert "AND campaign='$CAMPAIGN'" in fn and "registry_object_ids \"($2) AND campaign" in fn
    assert 'CAMPAIGN="${SISTEER_REGISTRY_CAMPAIGN:-${REGISTRY_CAMPAIGN:-sdfbase_full22}}"' in src
    assert 'EMISSION_CONTRACT="$(sed -n' in src and "registry_emission.py" in src


def test_stale_evidence_schedules_a_replay_and_current_evidence_waits_for_the_fold(tmp_path):
    base = {"object_id": "a" * 64, "status": "emitted", "records": ["/s/1.jsonl", "/s/2.jsonl"]}
    # emitted under another campaign (the emitters' old default) -> stale -> resubmit (rc 1), never done, never a fatal
    r = _run(tmp_path, dict(base, campaign="two_si", contract="2"))
    assert r.returncode == 0, r.stdout + r.stderr
    assert r.stdout.splitlines()[0].startswith("stale\temitted under campaign 'two_si', this run states 'sdfbase_full22'")
    assert "PU_RC=1" in r.stdout and "REPLAY the record under campaign sdfbase_full22" in r.stdout and "does not retrain" in r.stdout
    # emitted under an older contract -> stale -> resubmit
    r = _run(tmp_path, dict(base, campaign="sdfbase_full22"))
    assert r.stdout.splitlines()[0].startswith("stale\temitted under emission contract 1, current 2") and "PU_RC=1" in r.stdout
    # stale but the replay job is already live -> wait (rc 0)
    r = _run(tmp_path, dict(base, campaign="two_si", contract="2"), active=True)
    assert "PU_RC=0" in r.stdout and "still queued/running" in r.stdout
    # emitted under THIS campaign and the current contract -> waiting for the fold (rc 0)
    r = _run(tmp_path, dict(base, campaign="sdfbase_full22", contract="2"))
    assert r.stdout.splitlines()[0].startswith("emitted\tobject aaaaaaaaaaaaaaaa, 2 record file(s)") and "PU_RC=0" in r.stdout and "waiting for the fold" in r.stdout
    # failed / never emitted still refuse by name; no evidence resubmits
    r = _run(tmp_path, dict(base, status="failed", error="boom"))
    assert r.returncode == 7 and "FAILED" in r.stdout
    r = _run(tmp_path, None)
    assert "unverified\tno emission evidence beside COMPLETE" in r.stdout and "PU_RC=1" in r.stdout
