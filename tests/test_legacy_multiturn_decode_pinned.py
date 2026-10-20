"""Every LEGACY multi-turn consumer pins its target decode, and a new one cannot be missed.

On 2026-09-11 the multi-turn driver's DEFAULT target decode became "sample at the served model's
own generation config" (`jailbreak_run._target_temperature` -> None; the canonical 4,096-cap
instrument), and on 2026-09-16 the served model's VENDOR PRESET (`attacks.target_decode:
vendor_preset`, `jailbreak_run._resolve_multiturn_target_decode`; the user's instruction). Either is
the right default for the campaign's own lanes — and either is silently WRONG for anything replaying
a filed wave, because the run still succeeds and still writes plausible rows while the target has
moved underneath a published comparison. (Under the vendor-preset default an HF-path replay without
a pin now REFUSES rather than sampling, which is louder but still not the filed greedy.)

The named victim: `scripts/modal/grokmt_served.py` replays the published HeldOut wave of
2026-09-01 as an ATTACKER-ONLY contrast (grok-4.6 vs the campaign's production attacker, same
served target, same goals, legacy config `configs/two_si_gemma4e4b_reduced.yaml`). Its contrast is
clean only if everything but the attacker is fixed.

So: every consumer that drives a MULTI-TURN attack against a LEGACY config carries
`--target-greedy`, and this file both pins the known ones and DISCOVERS new ones — a consumer that
grows a multi-turn attack on a legacy config and no pin fails `test_no_unpinned_legacy_consumer`,
which is the check that survives someone forgetting this file exists.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

#: config families whose stores were filed under the pre-2026-09-11 greedy target
LEGACY_CONFIG_RE = re.compile(r"configs/(two_si[\w./-]*|stage2_jailbreak[\w./-]*|sdfbase_transfer[\w./-]*)\.yaml"
                              r"|two_si_\w+\.yaml|stage2_jailbreak\.yaml|sdfbase_transfer_")
#: the attacks that reach the multi-turn target (bon / gcg / the static families do not)
MULTITURN_RE = re.compile(r"\bcrescendo\b|\bactorattack\b|[\"'(\s]pair[\"')\s,]|\btap\b")
#: the driver invocation
DRIVER_RE = re.compile(r"jailbreak_run(?:\.py)?[\"'\s,]+.{0,40}generate|jailbreak_run\.py generate")

#: PINNED: legacy config + multi-turn + drives the CLI -> must carry --target-greedy
PINNED = {
    "scripts/modal/grokmt_served.py",          # the published 2026-09-01 HeldOut attacker-only contrast
    "scripts/modal/fst_multiturn.py",          # fit-support-transfer exp 2; its identity record binds target_temperature 0.0
    "scripts/sdfbase/mt/attack_wave.sbatch",   # the mt1 waves (sdfbase_transfer_* configs), filed greedy
    "scripts/sbatch/jailbreak_attacks.sbatch", # the stage-2 jailbreak wave (pair/tap/crescendo/actorattack)
    "scripts/sbatch/jailbreak_vllm_measure.sbatch",  # attacker-engine A/B: the target must be held fixed
}
#: CHECKED and deliberately NOT pinned, each with the reason it cannot drift
EXEMPT = {
    # RETIRED 2026-09-04 (its cluster no longer exists; exits 64 before any cluster code)
    "scripts/sbatch/attacker_pool_canary.sbatch": "retired launcher, exits 64",
    # no multi-turn attack: FAMS = bon pasttense pap encoding aim gcg_replay
    "scripts/run_dsr_eval.sh": "single-shot + BoN families only",
    # passes no --attack: the parser default is `bon`
    "scripts/sbatch/jailbreak.sbatch": "no --attack; the default tier is bon",
    "scripts/sbatch/jailbreak_gcg.sbatch": "gcg only",
}
#: the campaign's OWN lanes: they take the new default on purpose and must NOT be pinned
CANONICAL_UNPINNED = (
    "scripts/sdfbase/full22/target_attacks.sbatch",
    "scripts/sdfbase/full22/target_attacks_goalset.sbatch",
    "scripts/sdfbase/full22/smoke_full22.sbatch",
)


def _text(rel: str) -> str:
    return (REPO / rel).read_text()


@pytest.mark.parametrize("rel", sorted(PINNED))
def test_every_legacy_multiturn_consumer_pins_greedy(rel):
    s = _text(rel)
    assert "--target-greedy" in s, f"{rel} drives a multi-turn target on a legacy config unpinned"
    # the pin must be ON the driver invocation, not merely mentioned in prose
    drv = s[s.index("jailbreak_run"):]
    assert "--target-greedy" in drv, f"{rel}: the flag is not on the driver command"
    # and it must say WHY, at the place someone would edit
    why = s.lower()
    assert "pinned" in why or "frozen" in why, f"{rel}: the pin carries no reason"


def test_the_named_victim_is_the_published_attacker_only_contrast():
    s = _text("scripts/modal/grokmt_served.py")
    assert 'CFG_REL = "configs/two_si_gemma4e4b_reduced.yaml"' in s
    assert "2026-09-01" in s and "--target-greedy" in s
    # both sites: the real driver command AND the preflight argv the app validates
    assert s.count("--target-greedy") >= 2


def test_fst_identity_record_and_driver_agree_on_greedy():
    """The record binds target_temperature 0.0; the driver must actually send greedy, or the
    record is a lie rather than a refusal."""
    s = _text("scripts/modal/fst_multiturn.py")
    assert '"target_temperature": 0.0' in s and "--target-greedy" in s


def test_the_in_process_callers_state_greedy_rather_than_omitting_it():
    """two_si_run and jailbreak_run's canary call _run_gated_attack directly. They were greedy only
    because the kwarg was not threaded — an omission a mechanical edit could flip."""
    for rel in ("scripts/two_si_run.py", "scripts/jailbreak_run.py"):
        s = _text(rel)
        assert "target_temperature=0.0" in s, rel
        assert "GREEDY, STATED" in s, rel


def test_the_campaign_lanes_are_deliberately_unpinned():
    """The instrument's own lanes take the new default; pinning one would silently restore 512-era
    greedy to the very wave the ruling is about."""
    for rel in CANONICAL_UNPINNED:
        assert "--target-greedy" not in _text(rel), rel


def test_no_unpinned_legacy_consumer():
    """DISCOVERY, so the next one cannot be missed: any tracked script that names a legacy config,
    names a multi-turn attack and invokes the driver must be in PINNED or EXEMPT."""
    import subprocess
    tracked = subprocess.check_output(["git", "ls-files", "scripts"], cwd=REPO, text=True).split()
    found = set()
    for rel in tracked:
        if not rel.endswith((".py", ".sh", ".sbatch")):
            continue
        s = (REPO / rel).read_text(errors="replace")
        if LEGACY_CONFIG_RE.search(s) and MULTITURN_RE.search(s) and DRIVER_RE.search(s):
            found.add(rel)
    unknown = sorted(found - PINNED - set(EXEMPT))
    assert not unknown, (
        "these scripts drive a multi-turn attack on a legacy config and are neither pinned with "
        f"--target-greedy nor listed as exempt: {unknown}. The driver's default target decode SAMPLES "
        "(the served model's generation config since 2026-09-11, its vendor preset since 2026-09-16); "
        "a replay of a filed wave must pin greedy or it silently stops reproducing.")
    # the registry must not rot either: every listed path still exists
    for rel in sorted(PINNED | set(EXEMPT)):
        assert (REPO / rel).is_file(), f"{rel} is in the registry but not in the tree"
