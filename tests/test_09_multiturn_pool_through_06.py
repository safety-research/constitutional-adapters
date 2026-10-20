"""gap-09-attacker-pool (reproduce/README.md's Known gap: "Stage 09's multiturn legs bind to a live attacker pool this stage does not start"):
stage 09's `multiturn` eval is now the dose view's multiturn FAMILY run through 06_eval_waves.sh --view <dose view> --family multiturn, per
selected view — the ONE planner (lib/stage.py unit_plan: a DOSE view plans DOSE_LADDER_MT_ATTACKS, crescendo + actorattack, the Appendix D.2
legs; never pair) sizes the pool members from the view's targets and partitions the targets across every configured cluster, the ONE funnel
(lib/submit_twin.py) submits them, marks each group PLACED, prints the backstop's arm line and handles NO LIVE MEMBER exactly as the battery
wave's (06 exit 4 -> 09 exit 4, nothing duplicated: 06's text is byte-identical to the base tree and its placement tests stand unchanged). A
group 06 leaves UNPLACED past any other mark-placed refusal (its note-and-continue: the placement transaction failed, the immutable fence —
exit 0, WAVE written) is refused by 09 on the funnel's READ-ONLY placement-record, the authoritative record against the ledger:
exit 3, LADDER.<view>.partial.json, no LADDER.<view>.json, the same call the remedy.
Before, 09 composed the targets by hand onto the FIRST selected cluster, carried no F22_MT_TARGET_POLICY_REF / seed base, and started no
pool — the targets waited for a member nobody submitted. Dry runs against the NEUTRAL site example (labels cluster_a / cluster_b); the real
run on the wave-integration World (fake Slurm, a tmp repo at the wave sha, the demo smoke overrides: one shard group -> two D.2 targets, one
per label -> one member per label); the REAL member lifecycle for the verdict. Nothing here names a real host, user, path or session."""
from __future__ import annotations

import json
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from front_door_testlib import LIB, REPRO, VENV_PY, run_stage, set_job_state  # noqa: E402
import test_front_door_wave_integration as WI  # noqa: E402
from test_06_resubmit_pass_supersedes_placement import _member_verdict, _submitted  # noqa: E402  (the REAL lifecycle as the member's tick runs it)

sys.path.insert(0, str(LIB))
import stage as ST  # noqa: E402
import models as M  # noqa: E402

KEY = "qwen35_4b_demo"                                          # the front-door model of reproduce/models.yaml; its dose rungs are d0p75 / d0p75_relu
VIEW = "d0p75"
DOC_REL = "reproduce/overrides/demo_smoke.env"                  # the smoke knob: one shard group -> two D.2 targets, one per label, one member each
AB = ["cluster_a", "cluster_b"]
GOLDEN = Path(__file__).resolve().parent / "fixtures" / "unit_plan_qwen35_4b_demo_battery_base.json"   # unit_plan(battery) at the base tree b83e25368
KNOBS = ("SISTEER_SHARD_GROUPS", "SISTEER_MT_ATTACKER_GPUS", "SISTEER_MT_SEED_BASE", "SISTEER_MODELS_TABLE")
#: the lines of a 06 multiturn pass that ARE the pool, the targets, the placement and the backstop — what 09's pass must print identically
COMPOSED = ("submitted ", "DRY-RUN: mark-placed ", "followup: arm the group-level backstop now", "DRY-RUN: pin the placement manifest", "DRY-RUN: no placement manifest")


@pytest.fixture(scope="module")
def site(tmp_path_factory):
    root = tmp_path_factory.mktemp("g09")
    site = root / "site"
    shutil.copytree(REPRO / "site.example", site)
    with open(site / "config.env", "a") as fh:
        fh.write(f"\nSISTEER_STATE_DIR={root / 'state'}\nSISTEER_PYTHON={VENV_PY}\nSISTEER_REGISTRY_ROOT={root / 'registry'}\n")
    return site


def _dry09(site, *extra):
    r = run_stage("09_dose_ladder.sh", "--dry-run", "--alpha", "0.75", "--model", KEY, *extra, site=site, timeout=900)
    return r, r.stdout + r.stderr


def _dry06(site, *extra):
    r = run_stage("06_eval_waves.sh", "--dry-run", "--model", KEY, *extra, site=site, timeout=900)
    return r, r.stdout + r.stderr


def _composed(out: str) -> list:
    """The composed lines of a pass, in order: the `submitted` lines, the mark-placed / arm / placement-manifest dry lines, and every sbatch command.
    The one token that differs between two passes of the same plan is the submission clock the funnel stamps into every job's environment
    (SISTEER_UNIT_SUBMITTED_UTC); it is normalised so the comparison reads the composed command, not the second it was composed."""
    return [re.sub(r"SISTEER_UNIT_SUBMITTED_UTC=\S+", "SISTEER_UNIT_SUBMITTED_UTC=<utc>", ln) for ln in out.splitlines()
            if ln.startswith(COMPOSED) or (ln.startswith("DRY-RUN: ssh ") and " sbatch " in ln)]


def _units(out: str) -> list:
    return [ln.split()[3] for ln in out.splitlines() if ln.startswith("submitted ")]


def _state_files(site: Path) -> set:
    st = site.parent / "state"
    return {str(p) for p in st.rglob("*")} if st.exists() else set()


def test_09_dry_run_on_the_dose_view_with_evals_multiturn_composes_the_pool_the_targets_the_placement_and_the_backstop_like_06(site):
    before = _state_files(site)
    r, out = _dry09(site, "--step", "submit", "--evals", "multiturn", "--objects", "difference_lora")
    assert r.returncode == 0, out[-3000:]
    subs = [(ln.split()[1], ln.split()[3]) for ln in r.stdout.splitlines() if ln.startswith("submitted ")]
    # per label ONE group: the pool SIZED from the label's targets (four D.2 targets -> two members, one per three) + the targets PARTITIONED
    # across the two configured clusters (crescendo + actorattack x the four default shard groups, each once; never pair, never the first cluster alone)
    for lab in AB:
        mine = [u for l, u in subs if l == lab]
        assert sorted(u for u in mine if ".attacker" in u) == ["multiturn.attacker1", "multiturn.attacker2"], (lab, mine)
        assert sum(1 for u in mine if u.startswith("multiturn.") and ".attacker" not in u) == 4, (lab, mine)
    targets = sorted(u for _, u in subs if u.startswith("multiturn.") and ".attacker" not in u)
    assert targets == sorted(f"multiturn.{atk}.{g}" for atk in ("crescendo", "actorattack") for g in ("g012", "g34", "g56", "g78")), targets
    assert not any(".pair." in u for _, u in subs) and not any(u.startswith("ladder.") for _, u in subs)
    # the placement and the backstop, per label, exactly as 06 prints them for its own wave; the group is the dose view's
    for lab in AB:
        assert f"DRY-RUN: mark-placed mt-{VIEW}-{lab}: 4 target(s) -> attackers/PLACED" in out, out[-3000:]
        assert any(ln.startswith("followup: arm the group-level backstop now:") and f" wait-pool --group mt-{VIEW}-{lab}  (" in ln for ln in r.stdout.splitlines()), out[-3000:]
    assert r.stdout.count("followup: arm the group-level backstop now:") == 2 and f"expected_units.placement.{VIEW}.json" in out
    # ... and 09's own read of the authoritative record per selected group before its ladder record (the funnel's placement-record; a dry run reads nothing)
    assert all(f"DRY-RUN: placement-record mt-{VIEW}-{lab}: attackers/PLACED on the group's cluster against the ledger's bound targets and recorded members — nothing read, nothing written" in r.stdout for lab in AB)
    assert r.stdout.count("DRY-RUN: placement-record mt-") == 2 and r.stdout.index("DRY-RUN: placement-record mt-") > r.stdout.index("06 done")
    # the jobs: the pool member's sbatch and the targets' under the GROUP's wave root, the targets carrying the planner's decode-policy ref and seed
    # base (the hand-composed legs carried neither), the view on the positional contract; never --export=
    sbatch = [ln for ln in r.stdout.splitlines() if ln.startswith("DRY-RUN: ssh ") and " sbatch " in ln]
    assert len(sbatch) == 12 and sum(1 for ln in sbatch if "attacker_pool.sbatch" in ln) == 4 and sum(1 for ln in sbatch if "target_attacks.sbatch" in ln) == 8, len(sbatch)
    assert all(f"F22_WAVE_ROOT=" in ln and f"/waves/mt-{VIEW}-cluster_" in ln for ln in sbatch), sbatch[0][:600]
    assert all("F22_MT_TARGET_POLICY_REF=" in ln and "F22_MT_TARGET_SEED_BASE=" in ln for ln in sbatch if "target_attacks.sbatch" in ln)
    assert f"target_attacks.sbatch {KEY} {VIEW} crescendo " in out and f"target_attacks.sbatch {KEY} {VIEW} actorattack " in out and " 9 0,1,2" in out
    assert "--export=" not in out and "REFUSED" not in out and "Traceback" not in out
    # 09 says what it runs and records the view; 06's wave record and 09's ladder record are both composed (the dry run writes neither)
    assert f"multiturn legs of {VIEW}: the attacker pool + its targets are the dose view's multiturn family through 06" in out
    assert f"06_eval_waves.sh --model {KEY} --view {VIEW} --family multiturn --clusters cluster_a,cluster_b --nshards 9 --wave-sha " in out
    assert f"WAVE.{VIEW}.json" in out and out.count(f"LADDER.{VIEW}.json") >= 1 and "LADDER.d0p75_relu.json" not in out
    assert f"--view {VIEW} --family multiturn" in out and "--twin" not in [w for ln in out.splitlines() if "06_eval_waves.sh --model" in ln for w in ln.split()]
    # the retry's grammar is 06's --resubmit pass on the dose view, spelled out with this pass's flags
    note = [ln for ln in r.stdout.splitlines() if ln.startswith(f"note: multiturn legs of {VIEW}: a dead pool member or target is re-run through 06's fenced pass")]
    assert len(note) == 1 and f"--view {VIEW} --family multiturn --clusters cluster_a,cluster_b --nshards 9 --wave-sha " in note[0] and note[0].rstrip().split("  (")[0].endswith(" --resubmit"), note
    assert _state_files(site) == before
    # LIKE 06: the composed lines are 06's own for the dose view, byte for byte (the same planner, the same funnel, the same shell)
    r6, out6 = _dry06(site, "--view", VIEW, "--family", "multiturn")
    assert r6.returncode == 0, out6[-3000:]
    assert _composed(r.stdout) == _composed(r6.stdout) and len(_composed(r6.stdout)) == 12 + 12 + 2 + 2 + 2, (_composed(r.stdout)[:3], _composed(r6.stdout)[:3])


def test_09s_default_dry_run_plans_the_d2_legs_with_the_pool_for_both_rungs_and_twin_elects_the_single_gpu_legs_only(site):
    """The DEFAULT `--evals` (static,multiturn,controls — the D.2 legs) now composes the pool beside the legs for BOTH dose
    views; `--twin` elects the single-GPU legs and is NOT forwarded to the multi-turn family (06 refuses --twin for a group)."""
    r, out = _dry09(site, "--step", "submit")
    assert r.returncode == 0, out[-3000:]
    units = _units(out)
    for view in ("d0p75", "d0p75_relu"):
        assert sorted(u for u in units if u.startswith(f"ladder.{view}.")) == sorted([f"ladder.{view}.static_t0", f"ladder.{view}.static_t1", f"ladder.{view}.controls_t0", f"ladder.{view}.controls_t1"]), units
        assert f"DRY-RUN: mark-placed mt-{view}-cluster_a: 4 target(s) -> attackers/PLACED" in out and f"wait-pool --group mt-{view}-cluster_b" in out
        assert f"LADDER.{view}.json" in out and f"WAVE.{view}.json" in out
    mt = [u for u in units if u.startswith("multiturn.")]
    assert len(mt) == 2 * (4 + 8) and sum(1 for u in mt if ".attacker" in u) == 8 and not any(".pair." in u for u in mt) and not any(".gsm8k." in u for u in units), mt
    assert r.stdout.count("followup: arm the group-level backstop now:") == 4 and "--export=" not in out
    r, out = _dry09(site, "--step", "submit", "--evals", "static,multiturn", "--twin", "--objects", "difference_lora")
    assert r.returncode == 0 and "DRY-RUN: election" in out and "not implemented for multi-turn" not in out, out[-2500:]
    assert f"DRY-RUN: mark-placed mt-{VIEW}-cluster_a: 4 target(s) -> attackers/PLACED" in out and sum(1 for u in _units(out) if ".attacker" in u) == 4
    # the views-step retry stays the views step's; the refusal names where the multi-turn legs' retry lives
    r, out = _dry09(site, "--step", "submit", "--resubmit")
    assert r.returncode == 2 and "--resubmit is the views step's fenced retry" in out and "pass --step views" in out and "submitted " not in out
    assert f"06_eval_waves.sh --model {KEY} --view d0p75|d0p75_relu --family multiturn" in out and "--resubmit" in out.split("--family multiturn", 1)[1]


def test_the_dose_views_plan_the_d2_attacks_alone_and_06s_battery_plan_is_byte_identical_to_the_base_tree(monkeypatch):
    """The ONE planner: a DOSE view (a `views` entry declaring `dose:`) plans crescendo + actorattack for its multiturn family and sizes the pool from
    them; the wave (battery) plans the three attacks exactly as the base tree did — pinned against a golden of unit_plan(battery) generated at
    b83e25368 (tests/fixtures/unit_plan_qwen35_4b_demo_battery_base.json; regenerate it only with a deliberate change of the battery's plan)."""
    for k in KNOBS:
        monkeypatch.delenv(k, raising=False)
    rows = ST.unit_plan(KEY, "battery", clusters=AB, sha="{sha}", nshards=9)
    assert json.loads(json.dumps(rows, sort_keys=True)) == json.loads(GOLDEN.read_text()), "06's battery plan moved off the base tree's"
    assert sorted({r["unit"].split("/")[0] for r in rows if r["family"] == "multiturn" and r["kind"] == "target"}) == sorted(ST.MT_ATTACKS)
    m = M.model(KEY)
    assert ST.is_dose_view_row(m, "d0p75") and ST.is_dose_view_row(m, "d0p75_relu") and not ST.is_dose_view_row(m, "battery") and not ST.is_dose_view_row(m, "nosuch")
    assert ST.DOSE_LADDER_MT_ATTACKS == ("crescendo", "actorattack") and set(ST.DOSE_LADDER_MT_ATTACKS) < set(ST.MT_ATTACKS)
    for view in ("d0p75", "d0p75_relu"):
        d = ST.unit_plan(KEY, view, families=["multiturn"], clusters=AB, sha="{sha}", nshards=9)
        tg = [r for r in d if r["kind"] == "target"]; mem = [r for r in d if r["kind"] == "attacker"]
        assert sorted({r["unit"].split("/")[0] for r in tg}) == ["actorattack", "crescendo"] and len(tg) == 8 and len(mem) == 4, [r["unit"] for r in d]
        for lab in AB:
            n_t = sum(1 for r in tg if r["cluster"] == lab)
            assert n_t == 4 and sum(1 for r in mem if r["cluster"] == lab) == ST.attacker_pool_size(n_t) == 2 and all(r["group"] == f"mt-{view}-{lab}" for r in d if r["cluster"] == lab)
        assert {r["logical_work_id"] for r in tg} == {f"{KEY}/{view}/multiturn/{atk}/{g}" for atk in ("crescendo", "actorattack") for g in ("g012", "g34", "g56", "g78")}
        assert all(r["env"]["F22_MT_TARGET_POLICY_REF"] and r["env"]["F22_MT_TARGET_SEED_BASE"] == "20260909" for r in tg)
        assert {r["logical_work_id"] for r in mem} == {f"{KEY}/{view}/multiturn/mt-{view}-{lab}/attackers/{i}" for lab in AB for i in (1, 2)}
    # the single-GPU D.2 legs are the planner's too (09 composes them itself, in the sampled form) — unchanged by the attack rule
    s = ST.unit_plan(KEY, "d0p75", families=["static", "controls"], clusters=AB, sha="{sha}", nshards=9)
    assert sorted(r["unit"] for r in s) == ["store512", "t0", "t0", "t1", "t1"]


def _placed(w: WI.World, lab: str):
    hits = list(w.shared[lab].glob(f"exp/reproduce/{KEY}/waves/mt-{VIEW}-{lab}/attackers/PLACED"))
    assert len(hits) == 1, hits
    return hits[0], json.loads(hits[0].read_text())


def _world_with_the_dose_view(tmp_path):
    """The wave-integration World plus the dose view's wave config (05 --step configs renders it; 06 refuses the view without it), committed —
    the wave sha is the tree the stages pin."""
    w = WI.World(tmp_path)
    (w.repo / "configs" / f"sdfbase_full22_{KEY}_{VIEW}.yaml").write_text(
        f"private_run: sdfbase_full22_{KEY}_{VIEW}\nexplicit_cells: [none, constv2-mix-full__lora_r4_ce_minus_ctl]\nbon:\n  n_max: 75\n  goal_subset: jbb\n")
    g = ["git", "-c", "user.name=t", "-c", "user.email=t@TESTHOST"]
    subprocess.run([*g, "add", "-A"], cwd=w.repo, check=True); subprocess.run([*g, "commit", "-q", "-m", "the dose view's config"], cwd=w.repo, check=True)
    w.sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=w.repo, capture_output=True, text=True, check=True).stdout.strip()
    return w


def _stage09(w: WI.World, lw: str, *args: str):
    return w.stage("09_dose_ladder.sh", "--model", KEY, "--alpha", "0.75", "--objects", "difference_lora", "--step", "submit", "--evals", "multiturn",
                   "--force-gate", f"{lw}=smoke", "--overrides", DOC_REL, *args, SISTEER_RUN_OVERRIDES="")


def _frozen_world(tmp_path):
    """The World with the dose view, the battery frozen under the smoke document (05 --step freeze), the inventory cut to its one mmlu unit —
    FORCED by every 09 call here, so the dose view's admission admits (as test_front_door_dry_run's forced 06 test does) — and the producer
    spool the targets' fenced proof traverses on every label. Returns (world, the forced unit's logical work id)."""
    w = _world_with_the_dose_view(tmp_path)
    r = w.stage("05_stage_roster.sh", "--model", KEY, "--step", "freeze", "--overrides", DOC_REL, SISTEER_RUN_OVERRIDES="")
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    inv = w.inventory(); units = [u for u in inv["units"] if u["logical_work_id"].endswith("/mmlu/mmlu")]
    assert len(units) == 1 and inv["code_sha"] == w.sha and inv["view"] == M.baseline_view(KEY, str(w.repo / "reproduce" / "models.yaml"))
    lw = units[0]["logical_work_id"]
    ST.write_atomic(str(w.state / "gate" / "expected_units.json"), {**inv, "units": units, "n_units": 1})
    for lab in w.fake:
        (w.shared[lab] / "registry" / "records").mkdir(parents=True, exist_ok=True)       # the producer spool the targets' fenced proof traverses
    return w, lw


def test_09_submit_evals_multiturn_in_a_fake_world_starts_the_pool_places_the_targets_the_member_serves_and_06s_pass_re_mints_a_dead_member(tmp_path):
    """END TO END on the World: 05 freezes the battery under the smoke document; the inventory is cut to its one mmlu unit and FORCED, so the dose
    view's admission admits (as test_front_door_dry_run's forced 06 test does); `09 --step submit --evals multiturn` then runs 06's multiturn
    family for d0p75 against the fake Slurm — one member per label (the smoke's two D.2 targets, one per label), each group PLACED (v1), the
    arm line printed, 06's WAVE record and 09's LADDER record written — and the REAL member lifecycle answers `serve` on the committed placement.
    A member and its target die; the 06 pass 09's note spells (--resubmit on the dose view) re-mints both and SUPERSEDES the group's PLACED
    (v2) with the pass's provenance, the other label untouched; a further 09 pass skips every live unit and leaves both placements unchanged."""
    w, lw = _frozen_world(tmp_path)
    # ---- the pass: admission forced, 06's multiturn family for the dose view, the pool + the targets, PLACED per label, both records
    r = _stage09(w, lw); out = r.stdout + r.stderr
    assert r.returncode == 0, out[-4000:]
    assert f"ok   {lw}: FORCED (smoke)" in out and "ADMITTED" in out and f"06 eval_waves: model={KEY} view={VIEW}" in out and "09 done" in out, out[-3000:]
    assert f"dose view {VIEW}: no committed copy" in out and "the build/stage path applies" in out      # the World's tree carries no objects/: a note, nothing refused
    subs = _submitted(r)
    assert len(subs) == 4 and all(l in AB for l, _, _ in subs), subs
    members = {l: j for l, j, u in subs if u == "multiturn.attacker1"}; targets = {l: (j, u) for l, j, u in subs if ".attacker" not in u}
    assert set(members) == set(targets) == set(AB) and sorted(u for _, u in targets.values()) == ["multiturn.actorattack.g0", "multiturn.crescendo.g0"], subs
    assert r.stdout.count("followup: arm the group-level backstop now:") == 2 and all(f"wait-pool --group mt-{VIEW}-{l}  (" in r.stdout for l in AB)
    v1 = {}; paths = {}
    for lab in AB:
        paths[lab], v1[lab] = _placed(w, lab)
        assert v1[lab]["version"] == 1 and v1[lab]["members"] == [members[lab]] and v1[lab]["group"] == f"mt-{VIEW}-{lab}", v1[lab]
        assert [t["job_id"] for t in v1[lab]["targets"]] == [targets[lab][0]] and v1[lab]["targets"][0]["assigned_member_job_id"] == members[lab]
        assert v1[lab]["targets"][0]["logical_work_id"] == f"{KEY}/{VIEW}/multiturn/{targets[lab][1].split('.')[1]}/g0"
        # the jobs as the fake sbatch recorded them: the member runs attacker_pool.sbatch, the target target_attacks.sbatch under the group's wave
        # root with the planner's policy ref and seed base in its environment
        assert "attacker_pool.sbatch" in (w.fake[lab] / "jobs" / f"{members[lab]}.argv").read_text()
        tenv = (w.fake[lab] / "jobs" / f"{targets[lab][0]}.env").read_text()
        assert f"F22_WAVE_ROOT={w.shared[lab]}/exp/reproduce/{KEY}/waves/mt-{VIEW}-{lab}" in tenv and "F22_MT_TARGET_POLICY_REF=" in tenv and "F22_MT_TARGET_SEED_BASE=20260909" in tenv, tenv
        assert f"target_attacks.sbatch {KEY} {VIEW} " in (w.fake[lab] / "jobs" / f"{targets[lab][0]}.argv").read_text()
        # the REAL member lifecycle on the committed placement: the target it was assigned is live and has not bound yet -> every member stays
        v, why = _member_verdict(w, lab, paths[lab].parent, members[lab])
        assert v == "serve" and targets[lab][0] in why and ("neither bound nor ended" in why or "consumer(s) active (bound or assigned)" in why), (v, why)
    events = w.attempts()
    assert len([e for e in events if e["event"] == "placement_complete"]) == 2 and not [e for e in events if e["event"] == "placement_superseded"]
    assert not any((p.parent / "PLACED.transition").exists() for p in paths.values())    # a plain pass opens no transition marker
    wave = json.loads((w.state / f"WAVE.{VIEW}.json").read_text()); ladder = json.loads((w.state / f"LADDER.{VIEW}.json").read_text())
    assert wave["view"] == VIEW and wave["families"] == "multiturn" and wave["n_units"] == 4 and wave["forced"] == [{"logical_work_id": lw, "reason": "smoke"}] and wave["overrides"] == "demo_smoke"
    assert ladder["view"] == VIEW and ladder["evals"] == "multiturn" and ladder["forced"] == wave["forced"] and ladder["wave_sha"] == wave["wave_sha"] == w.sha and ladder["overrides"] == "demo_smoke"
    assert not (w.state / f"WAVE.{VIEW}.partial.json").exists() and not (w.state / "LADDER.d0p75_relu.json").exists()
    # ---- a member and its target die (an operator's scancel); the retry is the 06 pass 09's note spells — runnable as printed
    lab, other = "cluster_a", "cluster_b"
    set_job_state(w.fake[lab], members[lab], "CANCELLED", end="2026-01-01T01:00:00"); set_job_state(w.fake[lab], targets[lab][0], "CANCELLED", end="2026-01-01T01:05:00")
    before_other = paths[other].read_bytes()
    note = [ln for ln in r.stdout.splitlines() if ln.startswith(f"note: multiturn legs of {VIEW}: a dead pool member or target is re-run through 06's fenced pass")]
    assert len(note) == 1, r.stdout[-3000:]
    argv = shlex.split(note[0].split("(the views step's): ", 1)[1].split("  (", 1)[0])
    assert Path(argv[0]).resolve() == (w.repo / "reproduce" / "06_eval_waves.sh").resolve() and argv[-1] == "--resubmit" and f"{lw}=smoke" in argv and "--twin" not in argv, argv
    assert Path(argv[argv.index("--overrides") + 1]).resolve() == (w.repo / "reproduce" / "overrides" / "demo_smoke.env").resolve() and argv[argv.index("--wave-sha") + 1] == w.sha
    assert argv[argv.index("--config") + 1] == str(w.site / "config.env") and argv[argv.index("--view") + 1] == VIEW and argv[argv.index("--family") + 1] == "multiturn"
    r2 = subprocess.run(["bash", *argv], capture_output=True, text=True, env=w.env(SISTEER_RUN_OVERRIDES=""), cwd=str(w.repo), timeout=900); out2 = r2.stdout + r2.stderr
    assert r2.returncode == 0, out2[-4000:]
    subs2 = _submitted(r2)
    assert len(subs2) == 2 and all(l == lab for l, _, _ in subs2) and sorted(u for _, _, u in subs2) == ["multiturn.attacker1", targets[lab][1]], subs2
    new_member = next(j for _, j, u in subs2 if ".attacker" in u); new_target = next(j for _, j, u in subs2 if ".attacker" not in u)
    assert f"RESUBMIT {KEY}/{VIEW}/multiturn/mt-{VIEW}-{lab}/attackers/1" in out2 and "recorded torn_down (dead)" in out2 and "SUPERSEDED version 1 -> 2" in out2, out2[-4000:]
    path, v2 = _placed(w, lab)
    assert v2["version"] == 2 and v2["members"] == [new_member] and [t["job_id"] for t in v2["targets"]] == [new_target] and v2["targets"][0]["assigned_member_job_id"] == new_member, v2
    assert v2["reason"].startswith(f"06 --resubmit pass on {w.sha}: re-minted multiturn.attacker1,") and v2["supersedes"]["version"] == 1 and json.loads((path.parent / "PLACED.v1").read_text()) == v1[lab]
    assert paths[other].read_bytes() == before_other and "already placed, unchanged" in out2 and r2.stdout.count("followup: arm the group-level backstop now:") == 2
    assert not (path.parent / "PLACED.transition").exists() and not (paths[other].parent / "PLACED.transition").exists()
    v, why = _member_verdict(w, lab, path.parent, new_member)
    assert v == "serve" and new_target in why, (v, why)
    # ---- a further 09 pass: every unit live and skipped, both placements unchanged, the ladder record rewritten, still one supersession
    r3 = _stage09(w, lw); out3 = r3.stdout + r3.stderr
    assert r3.returncode == 0 and not _submitted(r3) and sum(1 for ln in r3.stdout.splitlines() if ln.startswith("skipped ")) == 4 and out3.count("already placed, unchanged") == 2, out3[-3000:]
    assert json.loads(path.read_text()) == v2 and paths[other].read_bytes() == before_other and "09 done" in out3
    assert len([e for e in w.attempts() if e["event"] == "placement_superseded"]) == 1 and json.loads((w.state / f"LADDER.{VIEW}.json").read_text())["utc"] >= ladder["utc"]


def test_09_refuses_the_ladder_record_when_06_ends_its_pass_past_a_failed_placement_transaction_and_the_same_call_then_places_the_group(tmp_path):
    """Every submission of the pass succeeds, then the placement transaction on cluster_a FAILS (the
    group's pool root is read-only, so `flock` cannot take PLACED.lock — the funnel's rc 3, nothing recorded; the code an ssh outage ends in too).
    06 notes the group "not fully placed yet — mark-placed runs again on the next pass", writes WAVE.d0p75.json, prints `06 done` and exits 0 —
    and 09 used to write LADDER.d0p75.json and print `09 done` over the unplaced group, whose targets bind round-robin and whose pool wait-pool
    cannot tear down. Now 09 reads the authoritative record per selected group (the funnel's READ-ONLY placement-record): cluster_b `placed`,
    cluster_a `none` -> exit 3 naming the group and the remedy, LADDER.d0p75.partial.json written, LADDER.d0p75.json NOT, no `09 done`;
    cluster_b's placement stands untouched. The remedy is the same call: with the pool root writable again every unit is skipped, 06's
    mark-placed commits cluster_a's v1, both records read `placed`, the ladder record lands, `09 done`, exit 0."""
    w, lw = _frozen_world(tmp_path)
    root_a = w.shared["cluster_a"] / "exp" / "reproduce" / KEY / "waves" / f"mt-{VIEW}-cluster_a" / "attackers"
    root_a.mkdir(parents=True); root_a.chmod(0o555)
    try:
        r = _stage09(w, lw); out = r.stdout + r.stderr
        assert r.returncode == 3, out[-4000:]
        subs = _submitted(r)
        assert len(subs) == 4 and sorted(l for l, _, _ in subs) == sorted(AB * 2), subs           # every submission went out before the placement
        members = {l: j for l, j, u in subs if u == "multiturn.attacker1"}; targets = {l: j for l, j, u in subs if ".attacker" not in u}
        ma, mb, ta, tb = members["cluster_a"], members["cluster_b"], targets["cluster_a"], targets["cluster_b"]
        # 06's pass ended as before (the finding's premise): the transaction's refusal, the note, WAVE written, `06 done` — and no record on cluster_a
        assert re.search(rf"REFUSED: mark-placed mt-{VIEW}-cluster_a: the placement transaction on cluster_a failed \(rc \d+\): .*PLACED\.lock", out), out[-4000:]
        assert f"note: group mt-{VIEW}-cluster_a is not fully placed yet (above) — mark-placed runs again on the next pass" in out
        assert "06 done: 4 unit(s) placed" in out and (w.state / f"WAVE.{VIEW}.json").exists() and not (w.state / f"WAVE.{VIEW}.partial.json").exists()
        assert not (root_a / "PLACED").exists() and not (root_a / "PLACED.lock").exists()
        path_b, vb = _placed(w, "cluster_b")
        assert vb["version"] == 1 and vb["members"] == [mb] and [t["job_id"] for t in vb["targets"]] == [tb] and vb["targets"][0]["assigned_member_job_id"] == mb
        assert r.stdout.count("followup: arm the group-level backstop now:") == 1 and f"wait-pool --group mt-{VIEW}-cluster_b  (" in r.stdout
        # 09: the authoritative read per selected group, the refusal, the partial record, no ladder record, no `09 done`
        assert f"placement-record\tmt-{VIEW}-cluster_b\tplaced\tattackers/PLACED version 1 on cluster_b: 1 bound target(s) each at its bound job and assigned a recorded member; members ['{mb}']" in r.stdout
        assert f"placement-record\tmt-{VIEW}-cluster_a\tnone\tno attackers/PLACED on cluster_a (1 bound target(s), 1 recorded member(s)): the placement was never committed" in r.stdout
        assert f"REFUSED: placement-record mt-{VIEW}-cluster_a: NONE — no attackers/PLACED on cluster_a" in r.stderr and "never a hand-edited PLACED" in r.stderr
        fatal = [ln for ln in r.stderr.splitlines() if ln.startswith(f"REFUSED: multiturn legs of {VIEW}: group(s) mt-{VIEW}-cluster_a are NOT placed (placement-record above)")]
        assert len(fatal) == 1 and f"{w.state}/LADDER.{VIEW}.json is NOT written" in fatal[0] and "run this same call again" in fatal[0] and "sanctioned supersession" in fatal[0], r.stderr[-3000:]
        assert "09 done" not in out and not (w.state / f"LADDER.{VIEW}.json").exists()
        partial = json.loads((w.state / f"LADDER.{VIEW}.partial.json").read_text())
        assert partial["unplaced_groups"] == [f"mt-{VIEW}-cluster_a"] and partial["view"] == VIEW and partial["evals"] == "multiturn" and partial["wave_sha"] == w.sha
        assert partial["forced"] == [{"logical_work_id": lw, "reason": "smoke"}] and partial["overrides"] == "demo_smoke" and partial["clusters"] == "cluster_a,cluster_b"
        assert len([e for e in w.attempts() if e["event"] == "placement_complete"]) == 1
    finally:
        root_a.chmod(0o755)
    # ---- the remedy: the same call. Every unit is live and skipped; 06's mark-placed commits cluster_a's v1; both records read placed; the ladder record lands
    r2 = _stage09(w, lw); out2 = r2.stdout + r2.stderr
    assert r2.returncode == 0, out2[-4000:]
    assert not _submitted(r2) and sum(1 for ln in r2.stdout.splitlines() if ln.startswith("skipped ")) == 4 and "09 done" in out2
    assert not [ln for ln in r2.stderr.splitlines() if ln.startswith("REFUSED:")], r2.stderr[-2000:]
    _, va = _placed(w, "cluster_a")
    assert va["version"] == 1 and va["members"] == [ma] and [t["job_id"] for t in va["targets"]] == [ta] and va["targets"][0]["assigned_member_job_id"] == ma, va
    assert json.loads(path_b.read_text()) == vb and "already placed, unchanged" in out2 and r2.stdout.count("followup: arm the group-level backstop now:") == 2
    assert r2.stdout.count("\tplaced\t") == 2 and f"placement-record\tmt-{VIEW}-cluster_a\tplaced\tattackers/PLACED version 1 on cluster_a: 1 bound target(s)" in r2.stdout
    ladder = json.loads((w.state / f"LADDER.{VIEW}.json").read_text())
    assert ladder["view"] == VIEW and ladder["wave_sha"] == w.sha and ladder["forced"] == partial["forced"] and ladder["evals"] == "multiturn"
    assert len([e for e in w.attempts() if e["event"] == "placement_complete"]) == 2


def test_placement_record_verdicts_the_ledgers_group_against_the_record_on_the_cluster(monkeypatch):
    """The funnel's READ-ONLY placement-record, the authoritative answer 09 reads before its ladder record: the group
    as the LEDGER knows it (bound targets, recorded members) against attackers/PLACED on the group's cluster, the differences accounted for as
    placement-changes accounts for them. The verdict table, on a bare Coordinator whose five reads are stubbed (nothing else is touched)."""
    import submit_twin as TW
    G = f"mt-{VIEW}-cluster_a"; T1 = f"{KEY}/{VIEW}/multiturn/crescendo/g0"; T2 = f"{KEY}/{VIEW}/multiturn/actorattack/g0"
    co = TW.Coordinator.__new__(TW.Coordinator)
    world = {"targets": [], "members": [], "label": None, "prev": None, "changes": ([], [], None, None), "asked": []}
    monkeypatch.setattr(co, "group_targets", lambda g: world["targets"])
    monkeypatch.setattr(co, "pool_members", lambda g: world["members"])
    monkeypatch.setattr(co, "pool_label", lambda g: world["label"])
    monkeypatch.setattr(co, "placed_record", lambda lab, g: world["asked"].append(("record", lab, g)) or world["prev"])
    monkeypatch.setattr(co, "placement_changes", lambda g, tg: world["asked"].append(("changes", g, tuple(tg))) or world["changes"])
    bind = lambda job: {"state": "submitted", "job_id": job, "cluster": "cluster_a"}   # noqa: E731
    rec = lambda rows, members, version=1: {"group": G, "version": version, "members": members, "n_targets": len(rows),   # noqa: E731
                                            "targets": [{"logical_work_id": lw, "job_id": j, "assigned_member_job_id": m} for lw, j, m in rows]}
    # empty: nothing bound, nothing recorded -> nothing to place (06's "no target of the group went out this call"); no cluster is read
    v, d, p = co.placement_record(G)
    assert (v, p) == ("empty", None) and "nothing to place" in d and not world["asked"]
    # none: a target bound, no pool recorded -> the record is read on the target's cluster; there is none
    world.update(targets=[(T1, bind("11"))])
    v, d, p = co.placement_record(G)
    assert (v, p) == ("none", None) and d.startswith("no attackers/PLACED on cluster_a (1 bound target(s), 0 recorded member(s))") and world["asked"] == [("record", "cluster_a", G)]
    world["asked"].clear()
    # none: the pool is recorded on cluster_a, no attackers/PLACED there -> the placement was never committed (the finding's scenario)
    world.update(members=["7"], label="cluster_a")
    v, d, p = co.placement_record(G)
    assert (v, p) == ("none", None) and d.startswith("no attackers/PLACED on cluster_a (1 bound target(s), 1 recorded member(s)): the placement was never committed")
    assert world["asked"] == [("record", "cluster_a", G)]; world["asked"].clear()
    # placed: the record names every bound target at its bound job with a recorded member assigned, every member, no change since
    world.update(prev=rec([(T1, "11", "7")], ["7"]), changes=([], [], None, None))
    v, d, p = co.placement_record(G)
    assert v == "placed" and p is world["prev"] and d.startswith("attackers/PLACED version 1 on cluster_a: 1 bound target(s) each at its bound job and assigned a recorded member") and "members ['7']" in d
    assert world["asked"] == [("record", "cluster_a", G), ("changes", G, (T1,))]; world["asked"].clear()
    # stale: a re-mint the record predates (placement-changes names the unit)
    world.update(targets=[(T1, bind("12"))], changes=(["multiturn.crescendo.g0"], [], None, world["prev"]))
    v, d, p = co.placement_record(G)
    assert v == "stale" and p is world["prev"] and d.startswith("attackers/PLACED version 1 on cluster_a is not the group's current placement: ") and "re-minted since version 1: multiturn.crescendo.g0" in d
    # stale: a bound target the record does not name
    world.update(targets=[(T1, bind("11")), (T2, bind("13"))], changes=(["multiturn.actorattack.g0"], [], None, world["prev"]))
    v, d, _ = co.placement_record(G)
    assert v == "stale" and f"target {T2} (job 13) is not named by version 1" in d and world["asked"][-1] == ("changes", G, (T1, T2))
    # stale: the assigned member left (dropped) and a replacement stands -> the target's assignment names no recorded live member
    world.update(targets=[(T1, bind("11"))], members=["8"], changes=([], ["multiturn.attacker1 (job 7: self-exited)"], None, world["prev"]))
    v, d, _ = co.placement_record(G)
    assert v == "stale" and "dropped since version 1: multiturn.attacker1 (job 7: self-exited)" in d and f"target {T1} is assigned member '7', not a recorded live member of the pool (['8'])" in d
    # stale: a difference the ledger cannot account for
    world.update(members=["7"], changes=([], [], f"target {T2} (job 13) named by version 1 is not in this roster — a plan change or a move, not a re-mint", world["prev"]))
    v, d, _ = co.placement_record(G)
    assert v == "stale" and "not in this roster" in d
    # the CLI: ONE tab-separated line, exit 0 on placed / empty and 3 (Refused) on none / stale or a cluster that does not answer; a dry run reads nothing
    src = (LIB / "submit_twin.py").read_text()
    assert 'sub.add_parser("placement-record"' in src and src.count("def placement_record(self, group: str)") == 1
    i = src.index('if a.cmd == "placement-record":'); blk = src[i:src.index('if a.cmd == "reclose":', i)]
    assert 'print(f"placement-record\\t{a.group}\\t{verdict}\\t{detail}", flush=True)' in blk and 'if verdict in ("none", "stale"):' in blk
    assert "except Unreachable as exc:" in blk and blk.count(", 3)") == 2 and "if site.dry:" in blk and "DRY-RUN: placement-record" in blk


def test_09_runs_06s_multiturn_family_for_the_dose_view_without_twin_or_resubmit_and_propagates_its_exit_code():
    """The static shape: 09 composes NO multi-turn job of its own (no target_attacks / attacker_pool sbatch, no shard-group loop, no
    `--group` of its own), runs 06 with this pass's effective flags (--clusters, --nshards, --wave-sha, every --force-gate, the resolved overrides
    document, the site config, the dry flag) and never --twin or --resubmit, and ends the pass on 06's exit code BEFORE the ladder record
    (4 -> 4 with the partial record named; any other refusal -> 2). 06 itself is untouched: its multiturn text pins (the placement tests) stand."""
    path = REPRO / "09_dose_ladder.sh"; src = path.read_text()
    assert subprocess.run(["bash", "-n", str(path)], capture_output=True).returncode == 0
    assert "target_attacks.sbatch" not in src and "attacker_pool.sbatch" not in src and "SISTEER_SHARD_GROUPS" not in src
    assert src.count('--group "mt-') == 1 and 'placement-record --group "mt-$view-$l"' in src      # the ONE group of its own is the read-only query, never a job
    i_arm = src.index("        multiturn) #"); i_next = src.index("        controls)", i_arm); blk = src[i_arm:i_next]
    i_mt = blk.index('mt=("$HERE/06_eval_waves.sh" --model "$K" --view "$view" --family multiturn --clusters "$CLUSTERS" --nshards "$NSHARDS" --wave-sha "$SHA")')
    i_force = blk.index('for f in ${FORCE[@]+"${FORCE[@]}"}; do mt+=(--force-gate "$f"); done')
    i_doc = blk.index('mt+=(--overrides "$SISTEER_RUN_OVERRIDES" --config "$SISTEER_CONFIG")'); i_dry = blk.index('[ "$REPRO_DRY_RUN" = 1 ] && mt+=(--dry-run)')
    i_run = blk.index('"${mt[@]}"; mt_rc=$?'); i_four = blk.index('if [ "$mt_rc" = 4 ]; then'); i_fatal4 = blk.index('REPRO_REFUSE_RC=4 repro_fatal "multiturn legs of $view: placement INCOMPLETE')
    i_else = blk.index('elif [ "$mt_rc" != 0 ]; then'); i_note = blk.index('repro_note "multiturn legs of $view: a dead pool member or target is re-run through 06\'s fenced pass')
    # after 06's exit code, the AUTHORITATIVE record per selected group decides (the funnel's read-only placement-record,
    # the one dry flag the arm forwards) BEFORE the retry note and the ladder record — a group 06 left unplaced past a mark-placed refusal it notes
    # and goes on from: the partial record, exit 3 (the funnel's placement-refusal code; 4 stays NO LIVE MEMBER's, 2 a refusal's)
    i_rec = blk.index('$(repro_dry_flag) placement-record --group "mt-$view-$l"'); i_partial = blk.index('repro_write "$STATE/LADDER.$view.partial.json"')
    i_fatal3 = blk.index('REPRO_REFUSE_RC=3 repro_fatal "multiturn legs of $view: group(s) $unplaced are NOT placed (placement-record above)')
    assert i_mt < i_force < i_doc < i_dry < i_run < i_four < i_fatal4 < i_else < i_rec < i_partial < i_fatal3 < i_note
    code = "\n".join(ln for ln in blk.splitlines() if not ln.strip().startswith("#"))      # the arm's CODE lines (its comments explain why --twin stays out)
    assert "--twin" not in code and "mt+=(--resubmit" not in code and "slurm_twin_flag" not in code and "slurm_submit" not in code
    assert code.count("$(repro_dry_flag)") == 1 and code.count("$(repro_dry_flag) placement-record --group") == 1 and 'for l in ${CLUSTERS//,/ }; do' in code[code.index('unplaced=""'):]
    assert f"WAVE.$view.partial.json" in blk[i_fatal4:i_else] and "LADDER.$view.json is NOT written" in blk[i_fatal4:i_else] and "LADDER.$view.json is NOT written" in blk[i_else:i_rec]
    assert "LADDER.$view.json is NOT written" in blk[i_fatal3:i_note] and "LADDER.$view.partial.json" in blk[i_fatal3:i_note] and "run this same call again" in blk[i_fatal3:i_note]
    assert blk.rstrip().endswith("--resubmit  (the placement transition opens, the walk re-mints what sacct proves dead, mark-placed supersedes the group's PLACED with the pass's provenance; arm wait-pool only after the arm line above)\";;")
    assert src.index('repro_write "$STATE/LADDER.$view.json"') > i_arm + i_note                   # the record follows a placed multi-turn family
    assert src.count('"${mt[@]}"; mt_rc=$?') == 1 and src.count("REPRO_REFUSE_RC=4") == 1 and src.count("REPRO_REFUSE_RC=3") == 1 and "the opt-in gsm8k" in src and "EVALS=static,multiturn,controls;" in src
    # the planner's rule the stage relies on, named where 09 says what it runs; the funnel's read-only query it reads before the record
    assert "DOSE_LADDER_MT_ATTACKS" in src and "DOSE_LADDER_MT_ATTACKS = (\"crescendo\", \"actorattack\")" in (LIB / "stage.py").read_text()
    assert 'sub.add_parser("placement-record"' in (LIB / "submit_twin.py").read_text() and "placement-record" in src[:src.index("set -uo pipefail")]
    # the usage prints the whole header (the header grew with the multi-turn paragraph): its last line precedes `set -uo pipefail`
    lines = src.splitlines()
    n = next(i for i, ln in enumerate(lines) if ln.startswith("set -uo pipefail"))
    assert f"usage(){{ sed -n '2,{n}p' \"$0\"; repro_common_help; }}" in src and all(ln.startswith("#") for ln in lines[1:n]), (n, lines[n - 1][:80])
