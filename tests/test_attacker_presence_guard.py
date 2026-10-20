"""A multi-turn target must not idle its GPUs waiting for an attacker that was never submitted.

THE INCIDENT (2026-09-17). A launch sequence submitted the target arrays and was then interrupted — the
process running it restarted — before the step that deploys the attacker member. That step never ran. Seven
targets came up healthy, passed their positive controls, and sat idle printing "waiting for a LIVE attacker
(N/90 min; the target is up and idle)" against a 90-minute driver budget. Nothing in the wave recorded that a
stage had been skipped: the ledger held no attacker event, and an empty ``attackers/`` directory is exactly
what a pool that is still booting also looks like. Seven H200s burned ~70 minutes on a pool that did not
exist. An absence is invisible unless something asserts presence.

THE FIX, in two halves, both pinned here:

1. ``attacker_pool.sbatch`` writes ``<jobid>.pending`` AT JOB START — not when it first serves ~22 minutes
   later — and removes it in the same cleanup trap that removes the live record. So "a member exists and is
   booting" becomes an observable fact rather than an inference from silence.

2. ``target_attacks_goalset.sbatch`` refuses IMMEDIATELY when the attackers directory holds neither a
   ``.json`` (serving) nor a ``.pending`` (booting). The bounded 90-minute wait is RIGHT when a pool is
   booting — the 27B attacker's standup is slower than the target's — so the guard does not shorten it; it
   only declines to enter it when nothing exists to wait for.

``.STOP`` files are deliberately NOT evidence. They are tombstones of members that have exited, and a
directory holding only tombstones is precisely the state the incident left behind — the discriminating case,
and the one a naive "is the directory non-empty?" check would get wrong.
"""

import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
POOL = REPO / "scripts" / "sdfbase" / "full22" / "attacker_pool.sbatch"
TARGET = REPO / "scripts" / "sdfbase" / "full22" / "target_attacks_goalset.sbatch"

COMMON = REPO / "scripts" / "sdfbase" / "full22" / "common.sh"


def _predicate() -> str:
    """The REAL shipped predicate, assembled from the two files that implement it.

    Since 2026-09-22 the evidence check no longer globs `attackers/*` itself: it asks
    f22_declared_attackers, the single enumeration point, which is also where F22_ATTACKER_JOBS
    applies. So a PINNED target asks "do MY members exist?" rather than "does anyone's?". Driving
    the shipped text -- rather than the copy that used to live here -- is what makes this file
    incapable of passing against a predicate that has drifted.
    """
    common = COMMON.read_text()
    m_decl = re.search(r"(?m)^f22_declared_attackers\(\)\{.*?^\}\n", common, re.S)
    assert m_decl, "f22_declared_attackers not found in common.sh"
    target = TARGET.read_text()
    m_fn = re.search(r"(?m)^atk_declare\(\)\{.*?^\}\n", target, re.S)
    assert m_fn, "atk_declare not found in the target sbatch"
    m_if = re.search(r"(?m)^atk_declare\nif \[ \"\$\{#ATK_DECLARED\[@\]\}\".*?^fi\n", target, re.S)
    assert m_if, "the evidence check not found in the target sbatch"
    return (
        'WAVE_ROOT="$1"\n'
        'F22_SERVING_PATH=lora_native\n'
        'f22_log(){ :; }\n'
        'f22_fatal(){ echo REFUSE; exit 2; }\n'
        + m_decl.group(0) + m_fn.group(0) + m_if.group(0)
        + "echo PROCEED\n"
    )


def _run(wave_root: Path, env_extra=None):
    env = {"PATH": "/usr/bin:/bin"}
    env.update(env_extra or {})
    p = subprocess.run(["bash", "-c", _predicate(), "_", str(wave_root)],
                       capture_output=True, text=True, env=env)
    return p.stdout.strip()


def _wave(tmp_path, *names, make_dir=True):
    root = tmp_path / "wave"
    if make_dir:
        (root / "attackers").mkdir(parents=True)
    else:
        root.mkdir(parents=True)
    for n in names:
        (root / "attackers" / n).write_text("{}\n")
    return root


def test_the_predicate_is_taken_from_the_sbatch_not_a_copy():
    """Guard the guard. This used to pin a hand-copied predicate; it now extracts the shipped one, so
    the drift it was written to catch cannot happen. What still needs pinning is that the evidence
    comes from the single enumeration point, and that the escape hatch survives."""
    body = TARGET.read_text()
    assert "atk_declare" in body and "f22_declared_attackers" in body, \
        "the evidence check must ask the single enumeration point, not re-glob attackers/*"
    assert 'if [ "${#ATK_DECLARED[@]}" -eq 0 ] && [ -z "${F22_ATTACKER_WAIT_ANYWAY:-}" ]; then' in body
    assert '"$WAVE_ROOT"/attackers/*.json' not in body, \
        "a raw evidence glob here would ignore F22_ATTACKER_JOBS and be satisfied by another lane's member"
    _predicate()   # the extraction itself must still resolve


def test_evidence_is_scoped_to_a_pinned_member(tmp_path):
    """The pin's whole point: another lane's member is not evidence that MINE was submitted."""
    assert _run(_wave(tmp_path, "548100.json"), {"F22_ATTACKER_JOBS": "548100"}) == "PROCEED"


def test_a_pinned_member_with_no_record_refuses_even_beside_a_live_one(tmp_path):
    """KNOWN-BAD: member 548100 is up, but this target was pinned to 777 and must not ride 548100."""
    assert _run(_wave(tmp_path, "548100.json"), {"F22_ATTACKER_JOBS": "777"}) == "REFUSE"


def test_only_STOP_tombstones_refuses_immediately(tmp_path):
    """THE INCIDENT'S EXACT SIGNATURE. The directory exists and is non-empty, but every entry is the
    tombstone of a member that has exited. A non-emptiness check would proceed and idle for 90 minutes."""
    assert _run(_wave(tmp_path, "519217.STOP", "536128.STOP")) == "REFUSE"


def test_empty_attackers_directory_refuses_immediately(tmp_path):
    assert _run(_wave(tmp_path)) == "REFUSE"


def test_missing_attackers_directory_refuses_without_crashing(tmp_path):
    """No directory at all — nullglob must make this a clean refusal, not an unmatched-glob error."""
    assert _run(_wave(tmp_path, make_dir=False)) == "REFUSE"


def test_a_booting_member_is_waited_for(tmp_path):
    """The LEGITIMATE wait: a pending marker means a member has started and is loading. The 90-minute
    bound exists for exactly this and the guard must not shorten or pre-empt it."""
    assert _run(_wave(tmp_path, "548100.pending")) == "PROCEED"


def test_a_serving_member_is_waited_for(tmp_path):
    assert _run(_wave(tmp_path, "548100.json")) == "PROCEED"


def test_evidence_beside_tombstones_still_waits(tmp_path):
    """A pool that lost a member and started another: tombstones must not veto live evidence."""
    assert _run(_wave(tmp_path, "9.STOP", "548100.pending")) == "PROCEED"


def test_the_escape_hatch_admits_a_pool_that_publishes_no_marker(tmp_path):
    """A Modal pool writes its endpoint elsewhere, so the operator can opt out deliberately — and the
    refusal message must name the variable, or the hatch is undiscoverable."""
    assert _run(_wave(tmp_path), {"F22_ATTACKER_WAIT_ANYWAY": "1"}) == "PROCEED"
    assert "F22_ATTACKER_WAIT_ANYWAY=1" in TARGET.read_text()


def test_the_refusal_names_the_remedy_and_the_cost():
    """A refusal that does not say what to do next just moves the stall from the GPU to the operator."""
    body = TARGET.read_text()
    assert "attacker_pool.sbatch" in body and "never submitted" in body
    assert "idle its GPUs" in body


def test_the_pool_announces_itself_at_start_and_cleans_up_on_exit():
    """Half 2: the marker is written BEFORE the ~22-minute vLLM boot (so it is visible during it) and is
    removed by the same trap that removes the live record (so a dead member leaves no false evidence)."""
    body = POOL.read_text()
    assert 'PENDING="$POOL_DIR/${SLURM_JOB_ID}.pending"' in body
    assert 'rm -f "$REC" "$PENDING"' in body, "the cleanup trap must remove the marker as well as the record"
    # written before the serve loop, not after it
    assert body.index("PENDING=") < body.index("attacker vLLM attempt"), \
        "the pending marker must be written BEFORE the vLLM boot, or it cannot cover the boot window"
    # and before the trap is installed, so the trap can see it
    assert body.index("PENDING=") < body.index("trap cleanup EXIT")


@pytest.mark.parametrize("script", [POOL, TARGET])
def test_scripts_parse(script):
    assert subprocess.run(["bash", "-n", str(script)]).returncode == 0
