"""Placement after a `--resubmit` pass: 06_eval_waves.sh passed mark-placed `--supersede` only for a DECLARED move-group, never on a
`--resubmit` pass — so the pass that re-minted the demo smoke's cancelled member (981273 -> 983283) and its three targets ran into the
immutable-placement fence ("attackers/PLACED exists (version 1 …) with a different roster or members"), PLACED v1 kept assigning every
target to the dead member, the new member found nothing assigned to it and self-exited (exit_unbound), and a target failed on its refused
endpoint. Now a --resubmit pass with no declared move passes `--supersede --reason '06 --resubmit pass on <sha>: re-minted <units>'` —
inert when nothing changed (the placement transaction answers `unchanged` before it reads supersede) — and a refusal prints the exact
sanctioned command. A member that LEFT beside a live one with nothing re-minted (the walk's skip + its recorded teardown) rides the same
path as `dropped <unit> (job <id>: self-exited)`, so a member-only change is superseded with a truthful reason instead of refused on every
pass. Through the REAL stage scripts on the wave-integration World (fake Slurm, a tmp repo at the wave sha; the demo smoke overrides —
one member and the three smoke targets partitioned over cluster_a / cluster_b — for the re-mint cases, the paper's plan — two members per
label — for the dropped-member one)."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from front_door_testlib import REPO, REPRO, set_job_state  # noqa: E402
import test_front_door_wave_integration as WI  # noqa: E402

sys.path.insert(0, str(REPRO / "lib"))
import models as M  # noqa: E402

MODEL = WI.MODEL
VIEW = M.baseline_view(MODEL, str(REPRO / "models.yaml"))      # the single-phase recipe's one wave, never a literal
DOC_REL = "reproduce/overrides/demo_smoke.env"                   # the smoke's knob: one shard group -> three targets, one member per label
LIFE = REPO / "scripts" / "sdfbase" / "full22" / "attacker_pool_lifecycle.sh"


def _placed(w: WI.World, lab: str):
    hits = list(w.shared[lab].glob(f"exp/reproduce/{MODEL}/waves/mt-{VIEW}-{lab}/attackers/PLACED"))
    assert len(hits) == 1, hits
    return hits[0], json.loads(hits[0].read_text())


def _member_verdict(w: WI.World, lab: str, pool: Path, member: str):
    """The REAL member lifecycle (attacker_pool_lifecycle.sh f22_pool_member_verdict) run as the member `member` of the group whose pool dir is
    `pool`, over the World's shared root and fake Slurm: (verdict, the reason the sbatch logs)."""
    env = dict(os.environ, PATH=f"{w.bin}:{os.environ['PATH']}", FAKE_SLURM_DIR=str(w.fake[lab]), PY_LIFECYCLE=sys.executable)
    r = subprocess.run(["bash", "-c", 'source "$1"; shift; "$@"', "_", str(LIFE), "f22_pool_member_verdict", str(pool.parent), "http://gpu-9:8700", member],
                       capture_output=True, text=True, env=env, timeout=120)
    assert r.returncode == 0 and len(r.stdout.split()) == 1, r.stdout + r.stderr
    return r.stdout.strip(), r.stderr.strip().splitlines()[-1]


def _submitted(r) -> list:
    """[(label, job id, unit)] of the `submitted <cluster> <jobid> <unit>` lines."""
    return [(ln.split()[1], ln.split()[2], ln.split()[3]) for ln in r.stdout.splitlines() if ln.startswith("submitted ")]


def _stage06(w: WI.World, *args: str):
    return w.stage("06_eval_waves.sh", "--model", MODEL, "--view", VIEW, "--family", "multiturn", "--overrides", DOC_REL, *args, SISTEER_RUN_OVERRIDES="")


def test_the_resubmit_pass_supersedes_the_placement_with_the_re_minted_member_and_targets_and_is_inert_elsewhere(tmp_path):
    w = WI.World(tmp_path)
    r = w.stage("05_stage_roster.sh", "--model", MODEL, "--step", "freeze", "--overrides", DOC_REL, SISTEER_RUN_OVERRIDES="")
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    for lab in w.fake:
        (w.shared[lab] / "registry" / "records").mkdir(parents=True, exist_ok=True)       # the producer spool the targets' fenced proof traverses (empty = absence)
    # ---- a PARTIAL first call (the members alone): mark-placed refuses by name (a target is unbound) and 06 prints the sanctioned transition
    #      verbatim beside the "do not arm" note — with a placeholder reason, since nothing was re-minted and no move is declared
    r = _stage06(w, "--units", "multiturn.attacker1")
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    assert [u for _, _, u in _submitted(r)] == ["multiturn.attacker1", "multiturn.attacker1"] and "followup: arm the group-level backstop now" not in r.stdout
    assert "is unbound — the group is not fully placed" in r.stderr and "do NOT arm wait-pool before it succeeds" in r.stdout
    cmd_lines = [ln for ln in r.stdout.splitlines() if "the sanctioned transition is exactly:" in ln]
    assert len(cmd_lines) == 2 and all(f"mark-placed --group mt-{VIEW}-" in ln and "--supersede --reason" in ln and "never hand-edit PLACED, a marker or a ledger" in ln for ln in cmd_lines), cmd_lines
    assert all(f"--model {MODEL} --overrides {DOC_REL} mark-placed --group" in ln for ln in cmd_lines), cmd_lines   # the overrides the pass ran under ride the command
    assert "PLACED.transition" not in r.stdout + r.stderr                                          # round 3: a pass that is not --resubmit opens no transition marker
    members = {lab: j for lab, j, u in _submitted(r)}
    assert set(members) == {"cluster_a", "cluster_b"}
    # ---- the full first pass: the members are skipped (bound, live), the three smoke targets go out, each label's group is PLACED (v1)
    r = _stage06(w)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    subs = _submitted(r)
    assert len(subs) == 3 and all(".attacker" not in u for _, _, u in subs) and r.stdout.count("followup: arm the group-level backstop now") == 2, r.stdout[-3000:]
    v1 = {}; paths = {}
    for lab in ("cluster_a", "cluster_b"):
        paths[lab], v1[lab] = _placed(w, lab)
        assert v1[lab]["version"] == 1 and v1[lab]["members"] == [members[lab]] and all(t["assigned_member_job_id"] == members[lab] for t in v1[lab]["targets"])
        assert sorted(t["job_id"] for t in v1[lab]["targets"]) == sorted(j for l, j, _ in subs if l == lab) and v1[lab]["n_targets"] >= 1
    assert len([e for e in w.attempts() if e["event"] == "placement_complete"]) == 2 and not [e for e in w.attempts() if e["event"] == "placement_superseded"]
    lab = max(("cluster_a", "cluster_b"), key=lambda l: v1[l]["n_targets"]); other = "cluster_b" if lab == "cluster_a" else "cluster_a"
    old_member = members[lab]; old_targets = sorted(t["job_id"] for t in v1[lab]["targets"])
    # ---- the member is CANCELLED (an operator's scancel: no SELF_EXIT, no STOP, no ledger trace) and its targets die on the refused endpoint;
    #      the other label's group stands untouched
    set_job_state(w.fake[lab], old_member, "CANCELLED", end="2026-01-01T01:00:00")
    for j in old_targets:
        set_job_state(w.fake[lab], j, "CANCELLED", end="2026-01-01T01:05:00")
    before_other = paths[other].read_bytes()
    # ---- the --resubmit pass: the walk records the dead member's torn_down, re-mints the member + its targets, and mark-placed SUPERSEDES
    #      PLACED v1 -> v2 with the pass's provenance; the other label's placement is unchanged (the supersede is inert there)
    r = _stage06(w, "--resubmit")
    out = r.stdout + r.stderr
    assert r.returncode == 0, out[-4000:]
    group = f"mt-{VIEW}-{lab}"
    assert f"RESUBMIT {MODEL}/{VIEW}/multiturn/{group}/attackers/1" in out and f"job {old_member} is DEAD on sacct (CANCELLED" in out and "recorded torn_down (dead)" in out, out[-4000:]
    subs2 = _submitted(r)
    assert len(subs2) == 1 + len(old_targets) and all(l == lab for l, _, _ in subs2), subs2
    new_member = next(j for _, j, u in subs2 if ".attacker" in u); new_targets = sorted(j for _, j, u in subs2 if ".attacker" not in u)
    assert new_member != old_member and not set(new_targets) & set(old_targets)
    assert sum(1 for ln in r.stdout.splitlines() if ln.startswith("skipped ") and f" {other} " in ln) == 1 + v1[other]["n_targets"]   # the other label: all live, all skipped
    torn = [e for e in w.attempts() if e["event"] == "torn_down" and e["job_id"] == old_member]
    assert len(torn) == 1 and torn[0]["dead"] is True and torn[0]["state"] == "CANCELLED" and torn[0]["cluster"] == lab, torn
    path, v2 = _placed(w, lab)
    assert v2["version"] == 2 and v2["members"] == [new_member] and sorted(t["job_id"] for t in v2["targets"]) == new_targets, v2
    assert all(t["assigned_member_job_id"] == new_member for t in v2["targets"]) and v2["changes"]["kept"] == 0 and v2["changes"]["members_before"] == [old_member]
    assert v2["supersedes"]["version"] == 1 and v2["reason"].startswith(f"06 --resubmit pass on {w.sha}: re-minted "), v2["reason"]
    reminted = v2["reason"].split("re-minted ", 1)[1].split(",")
    assert "multiturn.attacker1" in reminted and len(reminted) == 1 + len(old_targets) and all(u.startswith("multiturn.") for u in reminted), reminted
    assert sorted(x[0] for x in v2["changes"]["removed"]) == sorted(t["logical_work_id"] for t in v1[lab]["targets"]) and len(v2["changes"]["added"]) == len(old_targets)
    assert json.loads((path.parent / "PLACED.v1").read_text()) == v1[lab]
    assert "SUPERSEDED version 1 -> 2" in out and r.stdout.count("followup: arm the group-level backstop now") == 2 and "the sanctioned transition is exactly:" not in r.stdout
    sup = [e for e in w.attempts() if e["event"] == "placement_superseded"]
    assert len(sup) == 1 and sup[0]["group"] == group and sup[0]["from_version"] == 1 and sup[0]["to_version"] == 2 and sup[0]["members"] == [new_member] \
        and sup[0]["reason"] == v2["reason"] and sup[0]["changes"]["kept"] == 0, sup
    assert len([e for e in w.attempts() if e["event"] == "placement_complete"]) == 2
    assert paths[other].read_bytes() == before_other and not (paths[other].parent / "PLACED.v1").exists() and "already placed, unchanged" in out
    # placement-fix round 3: the pass marked BOTH groups' placement IN TRANSITION before its first submission (the
    # marker opened on each label, from version 1), and each mark-placed cleared it in the locked step that committed — the supersession on this
    # label, `unchanged` on the other — so no marker outlives the pass: the re-minted member was fenced from leaving exactly while it mattered
    assert r.stdout.count("/PLACED.transition opened on ") == 2 and r.stdout.index("/PLACED.transition opened on ") < r.stdout.index("\nsubmitted "), r.stdout[-6000:]
    assert all(f"placement-transition mt-{VIEW}-{l}: " in r.stdout and f"PLACED.transition opened on {l} — PLACED version 1 is in transition for 7200 s" in r.stdout for l in (lab, other)), r.stdout[-6000:]
    assert "(the pass's PLACED.transition marker cleared: the transition committed)" in out and "(the pass's PLACED.transition marker cleared: the placement stands as recorded)" in out, out[-4000:]
    assert not (path.parent / "PLACED.transition").exists() and not (paths[other].parent / "PLACED.transition").exists()
    # ---- a further --resubmit pass with nothing dead: every unit skipped, both placements unchanged, still ONE supersession (inert); the marker
    #      opened and cleared again on both labels (`unchanged` is a commit), none left behind
    r = _stage06(w, "--resubmit")
    out = r.stdout + r.stderr
    assert r.returncode == 0 and not _submitted(r) and out.count("already placed, unchanged") == 2, out[-3000:]
    assert json.loads(path.read_text()) == v2 and paths[other].read_bytes() == before_other
    assert len([e for e in w.attempts() if e["event"] == "placement_superseded"]) == 1 and len([e for e in w.attempts() if e["event"] == "torn_down"]) == 1
    assert r.stdout.count("/PLACED.transition opened on ") == 2 and out.count("(the pass's PLACED.transition marker cleared: the placement stands as recorded)") == 2, out[-4000:]
    assert not (path.parent / "PLACED.transition").exists() and not (paths[other].parent / "PLACED.transition").exists()


def test_a_retry_after_a_transiently_failed_mark_placed_supersedes_with_the_units_actually_re_minted_and_never_none(tmp_path):
    """The pass re-mints the member and targets and mark-placed fails transiently; on the retry every new job is live and skipped, so a reason
    derived from THIS call's appends read `re-minted none` — and --supersede still rode it, so the retry wrote PLACED v2 with that as the
    provenance of an immutable-placement transition. Now the units come from the funnel's READ-ONLY placement-changes (the current bindings
    and pool members against the record on the cluster), so the retry names exactly what the earlier pass re-minted, the refusal's printed
    command carries them too, and with nothing named no --supersede is passed at all."""
    w = WI.World(tmp_path)
    r = w.stage("05_stage_roster.sh", "--model", MODEL, "--step", "freeze", "--overrides", DOC_REL, SISTEER_RUN_OVERRIDES="")
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    for lab in w.fake:
        (w.shared[lab] / "registry" / "records").mkdir(parents=True, exist_ok=True)
    r = _stage06(w); assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    members = {lab: j for lab, j, u in _submitted(r) if ".attacker" in u}; assert set(members) == {"cluster_a", "cluster_b"}
    v1 = {}; paths = {}
    for lab in ("cluster_a", "cluster_b"):
        paths[lab], v1[lab] = _placed(w, lab); assert v1[lab]["version"] == 1 and v1[lab]["members"] == [members[lab]]
    lab = max(("cluster_a", "cluster_b"), key=lambda l: v1[l]["n_targets"]); other = "cluster_b" if lab == "cluster_a" else "cluster_a"
    old_member = members[lab]; old_targets = sorted(t["job_id"] for t in v1[lab]["targets"]); group = f"mt-{VIEW}-{lab}"
    set_job_state(w.fake[lab], old_member, "CANCELLED", end="2026-01-01T01:00:00")
    for j in old_targets:
        set_job_state(w.fake[lab], j, "CANCELLED", end="2026-01-01T01:05:00")
    before_other = paths[other].read_bytes()
    # ---- the pass re-mints, and its mark-placed FAILS transiently: the archive slot PLACED.v1 is occupied by a directory for the length of the
    #      pass, so the transaction cannot archive v1 (round 3 moved the injection off a read-only attackers directory: the pass now writes its
    #      transition marker there BEFORE submitting, and a directory that refuses writes is fatal before the first re-mint, by design) — nothing
    #      committed, no event, the wave not failed (the note + the sanctioned command print)
    pool = paths[lab].parent; (pool / "PLACED.v1").mkdir()
    r = _stage06(w, "--resubmit"); out = r.stdout + r.stderr
    assert r.returncode == 0, out[-4000:]
    subs2 = _submitted(r); assert len(subs2) == 1 + len(old_targets) and all(l == lab for l, _, _ in subs2), subs2
    new_member = next(j for _, j, u in subs2 if ".attacker" in u); new_targets = sorted(j for _, j, u in subs2 if ".attacker" not in u)
    assert f"the placement transaction on {lab} failed" in out and "do NOT arm wait-pool before it succeeds" in r.stdout, out[-4000:]
    assert json.loads(paths[lab].read_text()) == v1[lab] and (pool / "PLACED.v1").is_dir() and not [e for e in w.attempts() if e["event"] == "placement_superseded"]
    assert f"placement-changes {group}: re-minted since version 1: " in r.stderr, r.stderr[-3000:]
    cmd = [ln for ln in r.stdout.splitlines() if "the sanctioned transition is exactly:" in ln and f"--group {group} " in ln]
    assert len(cmd) == 1 and "re-minted" in cmd[0] and "multiturn.attacker1" in cmd[0] and "re-minted\\ none" not in cmd[0] and "re-minted none" not in cmd[0], cmd
    assert r.stdout.count("followup: arm the group-level backstop now") == 1       # the other label alone
    # ---- placement-fix round 3: the FENCE. The pass opened attackers/PLACED.transition on both labels before its first
    #      submission; the failed mark-placed left this label's standing (from version 1, the pass's reason, a finite lease) while the other label's
    #      `unchanged` cleared it; the note says the members hold and names the release. The REAL member lifecycle, run as the re-minted member
    #      over this pool (v1 assigns it nothing; v1's targets are all CANCELLED; the new targets have no binding yet), would have answered
    #      exit_unbound — and holds (`serve`) while the marker is fresh
    marker = pool / "PLACED.transition"; assert marker.exists() and not (paths[other].parent / "PLACED.transition").exists()
    mdoc = json.loads(marker.read_text())
    assert mdoc["group"] == group and mdoc["from_version"] == 1 and mdoc["reason"] == f"06 --resubmit pass on {w.sha}" and mdoc["lease_s"] == 7200 and mdoc["by"] == "submit_twin.py placement-transition open", mdoc
    assert isinstance(mdoc["epoch"], int) and mdoc["utc"].endswith("Z") and r.stdout.count("/PLACED.transition opened on ") == 2, mdoc
    note = [ln for ln in r.stdout.splitlines() if f"group {group} is not fully placed yet" in ln]
    assert len(note) == 1 and "hold every exit while this pass's attackers/PLACED.transition marker is fresh" in note[0] and f"placement-transition close --group {group} --cluster {lab}" in note[0], note
    v, why = _member_verdict(w, lab, pool, new_member)
    assert v == "serve" and "the group's placement is in TRANSITION (attackers/PLACED.transition since " in why and f"06 --resubmit pass on {w.sha}" in why and "would have answered exit_unbound" in why, (v, why)
    # the BOUND: the same marker past its lease is ignored (named in the reason) — the member decides on PLACED and its bindings alone, as before
    # the fence existed; a pass nobody completes cannot pin the member for ever
    marker.write_text(json.dumps(dict(mdoc, epoch=mdoc["epoch"] - mdoc["lease_s"] - 1)))
    v, why = _member_verdict(w, lab, pool, new_member)
    assert v == "exit_unbound" and why.startswith(f"placement complete ({len(old_targets)} target(s))") and "lapsed its lease: ignored" in why, (v, why)
    marker.write_text(json.dumps(mdoc))                                           # fresh again: the retry below clears it by COMMITTING, not by its lapse
    # ---- the RETRY: every new job is live and skipped — nothing re-minted THIS call — and the supersession names what the earlier pass re-minted;
    #      the archive slot is free again (the transient cause gone)
    (pool / "PLACED.v1").rmdir()
    for stray in pool.glob("PLACED.v1.tmp.*"):
        stray.unlink()                                                             # the failed transaction's temp file beside the occupied slot
    r = _stage06(w, "--resubmit"); out = r.stdout + r.stderr
    assert r.returncode == 0 and not _submitted(r) and "SUPERSEDED version 1 -> 2" in out, out[-4000:]
    path, v2 = _placed(w, lab)
    assert v2["version"] == 2 and v2["members"] == [new_member] and sorted(t["job_id"] for t in v2["targets"]) == new_targets, v2
    assert all(t["assigned_member_job_id"] == new_member for t in v2["targets"]) and v2["changes"]["kept"] == 0 and v2["changes"]["members_before"] == [old_member]
    assert v2["reason"].startswith(f"06 --resubmit pass on {w.sha}: re-minted "), v2["reason"]
    reminted = v2["reason"].split("re-minted ", 1)[1].split(",")
    assert reminted[0] == "multiturn.attacker1" and len(reminted) == 1 + len(old_targets) and all(u.startswith("multiturn.") for u in reminted) and "none" not in reminted, reminted
    sup = [e for e in w.attempts() if e["event"] == "placement_superseded"]
    assert len(sup) == 1 and sup[0]["group"] == group and sup[0]["reason"] == v2["reason"] and sup[0]["members"] == [new_member], sup
    assert json.loads((pool / "PLACED.v1").read_text()) == v1[lab] and paths[other].read_bytes() == before_other
    assert r.stdout.count("followup: arm the group-level backstop now") == 2 and "the sanctioned transition is exactly:" not in r.stdout
    # RECONCILED: the retry refreshed the marker (the earlier one stood) and the supersession's commit cleared it in the same locked step; the member
    # now serves on v2's own grounds — its assigned targets, live and not yet bound — with no marker in play
    assert f"/PLACED.transition refreshed on {lab}" in r.stdout and "(an earlier marker since " in r.stdout and "(the pass's PLACED.transition marker cleared: the transition committed)" in out, r.stdout[-5000:]
    assert not marker.exists() and not (paths[other].parent / "PLACED.transition").exists()
    v, why = _member_verdict(w, lab, pool, new_member)
    assert v == "serve" and "TRANSITION" not in why and all(j in why for j in new_targets) and ("neither bound nor ended" in why or "consumer(s) active (bound or assigned)" in why), (v, why)
    # ---- a further pass: nothing derived on either label (no --supersede at all), both `unchanged`, still ONE supersession
    r = _stage06(w, "--resubmit"); out = r.stdout + r.stderr
    assert r.returncode == 0 and not _submitted(r) and out.count("already placed, unchanged") == 2, out[-3000:]
    assert r.stderr.count("is the record's (version") == 2 and "re-minted" not in r.stderr, r.stderr[-3000:]
    assert json.loads(path.read_text()) == v2 and paths[other].read_bytes() == before_other and len([e for e in w.attempts() if e["event"] == "placement_superseded"]) == 1


def _stage06_paper(w: WI.World, *args: str):
    """The paper's plan (no overrides document): twelve multi-turn targets over the two labels, SIX per label -> TWO members per label (one per three)."""
    return w.stage("06_eval_waves.sh", "--model", MODEL, "--view", VIEW, "--family", "multiturn", *args, SISTEER_RUN_OVERRIDES="")


def test_a_member_that_left_beside_a_live_one_supersedes_the_placement_as_a_dropped_member_with_nothing_re_minted(tmp_path):
    """Round 2 (on 8b9a7b5c7): a group of TWO placed members; one self-exits (its last consumer ended) while the other serves on. The --resubmit
    pass skips the exited slot and records its teardown, so the member list changes with NO new job — and the pass passed no
    --supersede (placement-changes named re-minted units alone), mark-placed refused the changed roster on every pass, 06 noted "runs again on
    the next pass" and never printed the backstop's arm line again. Now the reason names the sha and the dropped unit, PLACED v2 carries the
    live member alone with every target assigned to it (the pairs it already served KEPT), the arm line prints, and a further pass is inert."""
    w = WI.World(tmp_path)
    r = w.stage("05_stage_roster.sh", "--model", MODEL, "--step", "freeze", SISTEER_RUN_OVERRIDES="")
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    for lab in w.fake:
        (w.shared[lab] / "registry" / "records").mkdir(parents=True, exist_ok=True)
    r = _stage06_paper(w); assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    subs = _submitted(r)
    by_unit = {lab: {u: j for l, j, u in subs if l == lab and ".attacker" in u} for lab in ("cluster_a", "cluster_b")}
    assert all(sorted(by_unit[lab]) == ["multiturn.attacker1", "multiturn.attacker2"] for lab in by_unit) and r.stdout.count("followup: arm the group-level backstop now") == 2, (by_unit, r.stdout[-3000:])
    v1 = {}; paths = {}
    for lab in ("cluster_a", "cluster_b"):
        paths[lab], v1[lab] = _placed(w, lab)
        assert v1[lab]["version"] == 1 and sorted(v1[lab]["members"]) == sorted(by_unit[lab].values()) and {t["assigned_member_job_id"] for t in v1[lab]["targets"]} == set(by_unit[lab].values())
        assert v1[lab]["n_targets"] == len([1 for l, _, u in subs if l == lab and ".attacker" not in u]) >= 4           # two members: more than three targets
    assert len([e for e in w.attempts() if e["event"] == "placement_complete"]) == 2
    lab, other = "cluster_a", "cluster_b"; group = f"mt-{VIEW}-{lab}"
    gone, live = by_unit[lab]["multiturn.attacker1"], by_unit[lab]["multiturn.attacker2"]
    kept_n = sum(1 for t in v1[lab]["targets"] if t["assigned_member_job_id"] == live); assert kept_n >= 1
    # member 1 closed itself (attackers/<job>.SELF_EXIT) and COMPLETED; member 2 serves on; every target runs on (the ones assigned the exited
    # member bound the live one at their runtime — a target never reads PLACED for discovery); the other label's group stands untouched
    pool = paths[lab].parent
    (pool / f"{gone}.SELF_EXIT").write_text(json.dumps({"job_id": gone, "reason": "exit_last_consumer", "consumers": "", "why": "placement complete and every consumer terminal on sacct (bound or assigned)", "utc": "2026-01-01T02:00:00Z"}))
    set_job_state(w.fake[lab], gone, "COMPLETED", end="2026-01-01T02:00:00")
    before_other = paths[other].read_bytes()
    # ---- the --resubmit pass: nothing re-minted (every other unit live and skipped), the exited slot's teardown recorded, and mark-placed
    #      SUPERSEDES v1 -> v2 with a reason naming the sha and the dropped unit; the other label is `unchanged`
    r = _stage06_paper(w, "--resubmit"); out = r.stdout + r.stderr
    assert r.returncode == 0 and not _submitted(r), out[-4000:]
    assert "SELF-EXITED" in out and "never re-placed (D1713)" in out and f"recorded torn_down (self_exit) for job {gone} on {lab}" in out, out[-4000:]
    torn = [e for e in w.attempts() if e["event"] == "torn_down"]
    assert len(torn) == 1 and torn[0]["job_id"] == gone and torn[0]["self_exit"] is True and torn[0]["cluster"] == lab, torn
    assert f"placement-changes {group}: dropped since version 1: multiturn.attacker1 (job {gone}: self-exited)" in r.stderr and "re-minted" not in r.stderr, r.stderr[-3000:]
    path, v2 = _placed(w, lab)
    reason = f"06 --resubmit pass on {w.sha}: dropped multiturn.attacker1 (job {gone}: self-exited)"
    assert v2["version"] == 2 and v2["members"] == [live] and v2["reason"] == reason and v2["supersedes"]["version"] == 1, v2
    assert sorted(t["job_id"] for t in v2["targets"]) == sorted(t["job_id"] for t in v1[lab]["targets"]) and all(t["assigned_member_job_id"] == live for t in v2["targets"]), v2
    assert v2["changes"] == {"added": [], "removed": [], "kept": kept_n, "members_before": v1[lab]["members"]}, v2["changes"]
    assert json.loads((pool / "PLACED.v1").read_text()) == v1[lab]
    assert f"SUPERSEDED version 1 -> 2 (0 added, 0 removed, {kept_n} kept their member)" in out and r.stdout.count("followup: arm the group-level backstop now") == 2, out[-4000:]
    assert "the sanctioned transition is exactly:" not in r.stdout and "is not fully placed yet" not in out
    sup = [e for e in w.attempts() if e["event"] == "placement_superseded"]
    assert len(sup) == 1 and sup[0]["group"] == group and sup[0]["members"] == [live] and sup[0]["reason"] == reason and sup[0]["changes"]["kept"] == kept_n, sup
    assert paths[other].read_bytes() == before_other and not (paths[other].parent / "PLACED.v1").exists() and out.count("already placed, unchanged") == 1
    assert len([e for e in w.attempts() if e["event"] == "placement_complete"]) == 2
    # ---- a further --resubmit pass: nothing derived on either label (no --supersede at all), both `unchanged`, still ONE supersession, ONE teardown
    r = _stage06_paper(w, "--resubmit"); out = r.stdout + r.stderr
    assert r.returncode == 0 and not _submitted(r) and out.count("already placed, unchanged") == 2, out[-3000:]
    assert r.stderr.count("is the record's (version") == 2 and "dropped since version" not in r.stderr and "re-minted" not in r.stderr, r.stderr[-3000:]
    assert json.loads(path.read_text()) == v2 and paths[other].read_bytes() == before_other
    assert len([e for e in w.attempts() if e["event"] == "placement_superseded"]) == 1 and len([e for e in w.attempts() if e["event"] == "torn_down"]) == 1


def test_06_derives_the_supersede_from_the_record_through_placement_changes_without_a_declared_move_and_prints_the_sanctioned_command_on_a_refusal():
    path = REPO / "reproduce" / "06_eval_waves.sh"; sh = path.read_text()
    assert subprocess.run(["bash", "-n", str(path)], capture_output=True).returncode == 0
    i_loop = sh.index("while IFS=$'\\t' read -r sf fam unit lwid cl kind grp script name array tkey twin_ok args env depends; do")
    i_mv = sh.index('MARK_EXTRA=(--supersede --reason "declared move-group $mv_g=$l (from $mv_from): $mv_reason")')
    i_rs = sh.index('elif [ "$RESUBMIT" = 1 ]; then')
    i_pc = sh.index('placement-changes --group "$g" --targets "${GROUP_TARGETS[$g]}")" || changes=""')
    i_re = sh.index('if [ -n "$changes" ]; then MARK_EXTRA=(--supersede --reason "06 --resubmit pass on $WAVE_SHA: $changes"); fi')
    i_mp = sh.index('mark-placed --group "$g" --targets "${GROUP_TARGETS[$g]}" ${MARK_EXTRA[@]+"${MARK_EXTRA[@]}"}')
    i_note = sh.index("do NOT arm wait-pool before it succeeds"); i_cmd = sh.index("the sanctioned transition is exactly:")
    assert i_loop < i_mv < i_rs < i_pc < i_re < i_mp < i_note < i_cmd     # a declared move's reason wins; the pass's reason only without one, from the record; the command on the refusal branch
    assert "--supersede --reason $(printf '%q' \"$sanctioned_reason\")" in sh and "never hand-edit PLACED, a marker or a ledger" in sh
    # never a line-count watermark over this call's own appends, never `none` as a supersession's provenance
    assert "LEDGER_N0" not in sh and "${changes:-none}" not in sh and "${reminted" not in sh and "re-minted none" not in sh
    # round 2: the reason's change clause is the funnel's whole answer (re-minted and/or dropped members) — 06 composes nothing of its own past
    # the `06 --resubmit pass on <sha>: ` prefix, so a member-only change (nothing re-minted) rides the same --supersede
    assert ": re-minted $" not in sh and '--reason "06 --resubmit pass on $WAVE_SHA: re-minted' not in sh and sh.count('--reason "06 --resubmit pass on $WAVE_SHA: $changes"') == 1
    assert sh.count("placement-changes --group") == 1 and sh[i_pc - 400:i_pc].count("$(repro_dry_flag)") == 1      # the read-only derivation, the dry flag forwarded
    assert 'elif [ "$RESUBMIT" = 1 ]; then sanctioned_reason="06 --resubmit pass on $WAVE_SHA: <the change the ledger accounts for — re-minted <units> and/or dropped <member units> — ' in sh   # the refusal's command names the pass when nothing was derived
    # round 3: a --resubmit pass marks every group's placement IN TRANSITION through the funnel BEFORE the submission
    # loop (one open per label, the dry flag forwarded, fatal with nothing submitted when it cannot be written), never on a plain pass; the refusal
    # note on a --resubmit pass says the members hold and names the sanctioned release
    i_open = sh.index('placement-transition open --group "mt-$VIEW-$l" --cluster "$l" --reason "06 --resubmit pass on $WAVE_SHA"')
    assert i_open < i_loop and sh.count("placement-transition open") == 1 and sh[i_open - 300:i_open].count("$(repro_dry_flag)") == 1
    assert 'if [ "$RESUBMIT" = 1 ]; then' in sh[i_open - 400:i_open] and 'case ",$FAMILIES," in *,multiturn,*)' in sh[i_open - 600:i_open]
    assert 'could not be opened on $l (above) — nothing submitted' in sh[i_open:i_open + 800]
    i_hold = sh.index("hold every exit while this pass's attackers/PLACED.transition marker is fresh")
    assert i_mp < i_hold < i_note < i_cmd and '[ "$RESUBMIT" = 1 ] && hold_note=' in sh[i_hold - 200:i_hold] and "placement-transition close --group $g --cluster $l" in sh[i_hold:i_hold + 600]
    assert sh[i_note:i_note + 400].count("$hold_note") == 1                       # the note carries it on a --resubmit pass alone
    assert sh.count('echo "followup: arm the group-level backstop now:') == 1
