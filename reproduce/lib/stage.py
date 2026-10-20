#!/usr/bin/env python3
"""reproduce/lib/stage.py — the stage scripts' Python helper for the ONE unit plan, the gate's frozen inventory, the
unit->job->run BINDING ledger and the sanity verdicts. stdlib + PyYAML; the registry's own
planner (sisteer.registry.jobplan, under <repo>/src) is invoked as a subprocess to MINT the ids a job registers.

  plan              --model K --view V [--families F,…] [--nshards 9] [--sha SHA] [--clusters a,b] [--only-clusters a] [--models-table T] [--json]
                    [--overrides F]
                    The submission units of a view, ONE row per sbatch: 06_eval_waves.sh
                    iterates these rows and freeze-inventory freezes the same rows (view u0), so the frozen inventory and the
                    submissions cannot diverge — every unit carries the EXACT logical_work_id its submission binds, the evals
                    its job registers (F22_REGISTRY_EVALS is passed explicitly, never left to a job-name table) and the shards
                    each eval must cover. Multi-turn: per named cluster ONE group (its attacker pool + the target units placed
                    on it); the 3 attacks x 4 shard-group target units are partitioned DISJOINTLY across the clusters
                    (round-robin, deterministic), never the same logical unit on two clusters. The model's overrides
                    document (--overrides, default $SISTEER_RUN_OVERRIDES) may set the ONE smoke knob, OVERRIDE_SMOKE_GOALS=<lo>-<hi>/N
                    (reproduce/overrides/demo_smoke.env): the multi-turn targets and the per-view padding levels run shards lo..hi of
                    N, every other row is the paper's; a plan knob the frozen inventory pins (plan_knobs), so --inventory refuses a
                    plan that drops or changes it.
  freeze-inventory  --model K --roster R --models-table T --code-sha S --out gate/expected_units.json [--filler PG19] [--seeds DIR]
                    [--goals-jsonl F] [--petri-bundle D]
                    Freezes the LOGICAL units of the unsteered (`u0`) battery = plan(u0)'s registering units — never a run_id (a
                    run id derives from `job_name@job_id`, which sbatch assigns). Per eval: `instrument_id` (the 64-hex the
                    registry's planner mints — jobplan.canonical_instrument; a padded level through `padded-instrument-id`),
                    `decode_policy_ref`, `expected_shards`, and `coverage` = the family's AUTHORITATIVE artifact layout
                    (lib/coverage_probe.py) plus the EXPECTED WORK IDENTITIES: the goal ids per shard the producer's own recipe
                    yields (two_si_run._eval_goals(confirmatory) sliced goals[s::nshards] — the same call the drivers make), the
                    agentic scenario ids (misalign.runner.agentic_items), the item set's declared size / ids sha256 (registry seeds),
                    the Petri `<behavior>/<scenario>` ids from the FROZEN scenario bundle (petri-scenarios.lock.yaml's content hash;
                    the bundle's `<behavior>/scenarios/seeds/*` stems — scenarios.scenario_ids' rule; $SISTEER_PETRI_BUNDLE_DIR, else the
                    grading host's store `~/.local/state/petri/scenarios/<hash>`, the lock's location, or the copy this checkout publishes at
                    `data/petri/scenarios/<hash>`). With no copy anywhere and no family named, the freeze SKIPS the Petri family by name
                    (the inventory's `skipped_families`; 06 --family all drops it) instead of refusing the whole inventory.
                    Never an items-per-shard line count. `--goals-jsonl` (or
                    $SISTEER_GOALS_JSONL) is the offline fixture seam grade_static_sampled.py already has (rows {goal_id, dataset,…}).
                    Idempotent: a changed inventory is written beside the old one (expected_units.<utc>.json) and the diff printed.
  rerun-units       --out gate/expected_units.json --cells C,… --source "ruling" --code-sha S [--models-table T] [--roster R] [--seeds DIR]
                    [--goals-jsonl F] [--wave-config F] [--clusters a,b] [--overrides F] [--supersede-receipt] [--dry-run]
                    STAMP an existing, started inventory with the BoN RERUN pair of the named cells (`bon/store512_<slug>` ->
                    `bon/climb_<slug>`, afterok) — the funnel's door for a COMPLETED, registered 512 store whose cells a ruling re-runs (the
                    constitution-SI cell served at 4,864: every slot refused prompt_over_window, the closer's no-fit certificate filed for a
                    mis-served cell). A re-freeze cannot add them (the inventory is frozen once attempts exist) and `--resubmit`
                    refuses a COMPLETED registering unit, so the pair is appended the way `hold-out` stamps: the ledger receipt (`rerun_units`)
                    first, then the inventory; idempotent; the original units untouched. A receipt covers a stamp on its WHOLE declaration
                    + tree (never the pair's name alone): a retry after an interrupted apply under another tree / ruling is refused by name
                    until it either completes the receipt's declaration or passes `--supersede-receipt`, which appends a receipt naming the
                    orphan superseded by its unique `receipt_id` and its whole reference (never by its second-resolution write time + tree, which
                    two pairs' receipts written in one second share) — the ledger never attributes a rerun to a tree it did not run at, and
                    a conflicting receipt stays visible until the receipt that retires it names IT. `plan --inventory` (06 passes it for a started wave)
                    reads the stamped declarations and plans the pair for THEIR view, so `06 --family static --units bon.store512_<slug>` and
                    `06 --family bon --units bon.climb_<slug>` submit them against the same inventory check every other unit passes. The pair
                    serves exactly the ruling's cells (its cells positional + F22_REGISTRY_CELLS) under its OWN wave root
                    (`<campaign root>/rerun/<slug>`: F22_BON_RERUN -> bon_long_si.sh f22_bon_rerun_root — transcripts_results, bon_climb, the
                    closer's --wave-root and the STOP sentinel, as the per-group pool roots own theirs, §3.9) and registers under its own
                    job-independent producer key (`sdfbase_full22-stat-3-<tag>-<slug>`: F22_REGISTRY_KEY_SEGMENTS = the slug), so the old
                    certificate's rows and the rerun's never share a store, a file or a run id.
  bind              --ledger F --logical-work-id L --state submitted --job-id J --cluster C --attempt-id A [--sbatch-name N] [--array A]
                    --ledger F --logical-work-id L --state registered --run-ids R1,R2 [--pending-file P] [--runs-json '[…]']
                    One record per transition, written tmp+rename into <ledger dir>/bindings/<lwid>.<state>.json, so a unit
                    is `unbound | submitted | registered` and never ambiguous (a torn record reads as the previous state).
  amend-contract    --ledger F --logical-work-id L --reason "…" --commit SHA [--tip SHA] --registry DB [--config F] [--repo D] [--old-export FILE]
                    [--dry-run] | --recheck
                    The RECEIPTED contract amendment: a seed-row correction that changed instrument ids is admitted for a unit only when
                    (1) the OLD decode-policy export is DERIVED from the unit's recorded tree (`git show <recorded sha>:configs/registry/
                    decode_policy.json`; a supplied --old-export is accepted only byte-identical to it), (2) --commit is an ancestor-or-equal of
                    --tip (the tree the re-run submits at; default HEAD of --repo), the NEW export and the seeds are read from git at --commit, and
                    the row diff is confined to the three keys' EXACT neutral transitions (min_p null->0.0, repetition_penalty null->1.0,
                    presence_penalty null->0.0) plus documentation fields, (3) every old job of the unit is TERMINAL on sacct (the funnel's reader
                    over --config) and every old run — the ledger's plus the registry's runs naming those jobs — is non-completed: a `running` run only with its job proven terminal-not-COMPLETED, recorded
                    per run — and the registry is re-read after the proofs, right before the ledger event is written. The receipt (a ledger event
                    first, then the provenance sidecar) names every old job and run and why each is excluded. --recheck re-runs the three proofs
                    over every recorded receipt of the unit, writing nothing.
                    --kind petri-role-fallback | code-follow-up: the Petri unit's receipted sha ADVANCES — the per-request refusal fallback's two
                    manifest keys, and a CODE-ONLY follow-up to a receipted advance (a strict descendant of the amended sha;
                    manifest, roster, contract, export, old run set and carried set unchanged; --note names the findings it carries).
  array-tasks       --sacct TEXT   the array's tasks from `sacct -X -n -P -o JobIDRaw,JobID,State -j <array id>`: [{raw, task, state}]
                    (the pending file of an array TASK is pending/<raw id>.json — every task registers itself)
  probe-plan        --state-root D --logical-work-id L --runs-json '[{run_id, eval_id, object, array_task, store_uri}]' --shared ROOT
                    --probe PATH   one TSV line per bound run: run_id, host (cluster|grading_host), the coverage_probe.py argv
  coverage          --state-root D --logical-work-id L --runs-json '[{run_id, eval_id, object, store_uri, probe: {…}}]'
                    gate/coverage/<unit>.json — per BOUND run, the probe's completed / errored / expected ids; recomputed on every --gate.
  verdict           --state-root D --family F (--headline X --range LO,HI | --headline-record @F) --run-ids ... [--verdict pass|fail] [--note ...]
                    gate/<family>.json {headline, plausible_range, verdict, ruled_utc, run_ids}; from a --headline-record (06 --gate --verdict:
                    lib/gate_headline.py read) also {headline_rule, expected, headline_legs, headline_held_out}. `run_ids` = the runs the pass RULED
                    (06 --gate hands it family_bound_run_ids: the union of the family's registered bindings); admission refuses the
                    family when that set differs from the family's bindings at check time (a verdict is bound to its runs).
  status            --state-root D   the inventory x bindings x verdicts table (what the admission gate would see)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import os
import secrets
import shlex
import shutil
import subprocess
import sys
from typing import Dict, List, Mapping, Optional, Sequence, Set, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import models as M  # noqa: E402
# the gate's reader lib/coverage_probe.py (stdlib-only) declares the coverage contract's DONE_PREDICATES; this module reads them
# through this_tree_done_predicates() -- the file's bytes, never a `coverage_probe` import that Python's bytecode cache may serve stale (a
# same-length edit within the second of the last compile; 2026-10-14) -- so the freeze, rerun-units and the gate read ONE value
from grant_ledger import utc_now  # noqa: E402

REPO = os.path.dirname(os.path.dirname(HERE))


def arg_doc(value: Optional[str]) -> Optional[str]:
    """A CLI flag's DOCUMENT: `@<path>` is read from that file, `@-` from stdin, anything else is the literal value. Every stage.py flag that
    takes JSON or a text capture as its value goes through this (bind / probe-plan / coverage --runs-json, array-tasks --sacct, pending-runs
    --tasks-json, coverage --probes-json, verdict --contract): 06 --gate hands them FILES — the pending file of a 40-run static unit is
    149,440 bytes and Linux caps ONE argv string at MAX_ARG_STRLEN = 131,072 bytes (`Argument list too long`, which the old shell read as
    "the pending file names no runs"; 2026-10-13). Literals keep working for small values and the tests."""
    if value is None:
        return None
    if value == "@-":
        return sys.stdin.read()
    if value.startswith("@"):
        with open(value[1:], encoding="utf-8") as fh:
            return fh.read()
    return value
#: the paper's five padding levels (each its own instrument identity, migration 0012) and the padded window — the frozen
#: inventory and the submissions share this ONE table
PADDING_LEVELS = (0, 8192, 16384, 32768, 65536)
PADDING_WINDOW = 98304
#: the three multi-turn attacks and the DEFAULT four disjoint shard groups over the 9 shards (§3.9; SISTEER_SHARD_GROUPS
#: overrides them — validated as a disjoint cover of 0..nshards-1, refused otherwise)
MT_ATTACKS = ("crescendo", "pair", "actorattack")
#: the Appendix D.2 dose ladder's multi-turn legs: the ladder's
#: default set is "sampled statics, crescendo + actorattack, controls" — PAIR is a leg of the WAVE, never of a ladder rung. A DOSE view (a
#: table view declaring `dose:`, models.py's rule for a rung) plans these two for its multiturn family; every other view plans MT_ATTACKS.
#: The ONE planner is what stage 09 runs for its multi-turn legs (through 06's multiturn family for the dose view: the pool members sized
#: from these targets, the placement, the backstop), so the rung's attack set lives here and nowhere else; the
#: placement manifest, its check and the funnel's proofs all plan the view through this function and agree by construction.
DOSE_LADDER_MT_ATTACKS = ("crescendo", "actorattack")
#: attacker pool members per concurrently live target on a cluster (commit 31): one per three, minimum one
MT_TARGETS_PER_ATTACKER = 3


def is_dose_view_row(m: Mapping[str, object], view: str) -> bool:
    """Whether `view` is a DOSE view of the model row `m` — a `views` entry declaring `dose:` (the table's own rule for a ladder rung:
    models.py staged_id_of / declared_wave_problems read the same key). Name grammar alone (`d<tok>`) is not consulted: the row declares."""
    return "dose" in (((m.get("views") or {}).get(view)) or {})


def attacker_pool_size(n_targets: int, cap: Optional[int] = None) -> int:
    """The pool size for `n_targets` concurrently live multi-turn targets on one cluster: ceil(n / MT_TARGETS_PER_ATTACKER), at least one; a
    `cap` (SISTEER_MT_ATTACKER_GPUS) can only LOWER it (never below one)."""
    import math as _m
    n = max(1, int(_m.ceil(max(0, int(n_targets)) / MT_TARGETS_PER_ATTACKER)))
    if cap is not None:
        n = max(1, min(n, int(cap)))
    return n
#: ActorAttack has NO measured 4,096-cap serving window (geometry.MULTITURN_CAP4096_WINDOWS names crescendo and pair only) and
#: target_attacks.sbatch refuses it unless the window rides positional 8 ("pass max_model_len explicitly and record it"). The front
#: door passes the one ActorAttack window the paper ever pinned — 49,152, the sdfbase-llama8b roster caps table's
#: "ActorAttack's 49,152 x 9" (geometry.multiturn_window's docstring), which is also the crescendo window — and the standup records
#: WINDOW_SOURCE=positional. 2026-09-30: the first u0 wave's two actorattack targets died in their
#: first minute at `no measured 4096-cap serving window for ['actorattack']`; crescendo and pair keep the measured table.
MT_ACTORATTACK_WINDOW = 49152
DEFAULT_SHARD_GROUPS = ("0,1,2", "3,4", "5,6", "7,8")
STATIC_EVALS = ("pasttense", "codeattack", "renellm", "grok420")
#: what the Petri proxy job registers (scripts/sdfbase/full22/common.sh f22_registry_evals_for_job `f22-petri-*`; pinned equal by
#: tests/test_front_door_wave_integration.py so the two tables cannot drift)
PETRI_EVALS = ("petri_pooled", "petri_cyber_bio_chem_capabilities", "petri_self_preservation", "petri_delusion_sycophancy",
               "petri_defer_to_users", "petri_reasoning_unfaithfulness", "petri_instruction_evasion")
AGENTIC_EVALS = ("agentic_leaking", "agentic_murder")
CONTROL_EVALS = ("xstest", "or_bench_hard")
#: the Slurm eval families of the GLOBAL gate — the --family vocabulary of 06_eval_waves.sh, in submission order
FAMILIES = ("static", "padding", "bon", "multiturn", "petri", "agentic", "controls", "mmlu", "gsm8k")
SERVING_PATH = "lora_native"          # FD 19: the native path, never lens
#: the preset the recipe's two best-of-N units declare (target_static.sbatch and bon_climb.sbatch forward F22_BON_DECODE_PRESET to the
#: drivers' --bon-decode-preset): `canonical` = the registry label whose sampler reproduces THE CANONICAL best-of-N instrument the registry's
#: planner binds for the served model -- the instrument the unit's start-of-job registration binds -- resolved by the driver at launch
#: (bon_served.canonical_bon_label; refused by name when no label does), never a number or a label chosen here. The worked example declares
#: no preset of its own: its canonical instrument is the paper's T=1.0 climb (its rows mirror qwen35_4b_it's), which the sbatch default
#: `vendor_preset` (0.7 / 0.8 / 20 / 1.5 on Qwen) would NOT draw, so the store would have attested the T=1.0 instrument for vendor draws.
#: The pipeline-test models resolve to vendor_preset under the same rule.
BON_DECODE_PRESET_ENV = "F22_BON_DECODE_PRESET"
BON_DECODE_PRESET = "canonical"
#: the BoN RERUN pair's job-side declaration (bon_long_si.sh f22_bon_rerun_root): the slug on the two rerun rows' env — the job
#: re-roots its WAVE_ROOT to `<campaign root>/rerun/<slug>` (transcripts_results, bon_climb, serving, the closer's --wave-root and the STOP
#: sentinel all follow, the per-group pool roots' rule) and exports F22_REGISTRY_KEY_SEGMENTS=<slug>, so the store's job-independent producer
#: key is `sdfbase_full22-stat-3-<tag>-<slug>` (common.sh f22_registry_producer_key), never the superseded store's
BON_RERUN_ENV = "F22_BON_RERUN"
RERUN_ROOT_REL = "rerun"
PLAN_COLUMNS = ("stage_family", "family", "unit", "logical_work_id", "cluster", "kind", "group", "script", "name", "array",
                "time_key", "twin_ok", "args", "env", "depends")
#: `depends` (trailing, `-` = none): the logical_work_id of a unit THIS unit's job must run after — the funnel submits it with
#: `--dependency=afterok:<that unit's bound job>` while that job is live or pending, none once the unit is complete, and refuses
#: by name when the unit was never submitted or its job failed (the BoN climb ran while the 512 store was being written)
# ---- the family's AUTHORITATIVE coverage artifact ----------------------------------------------------------------------
#: the wave root every front-door job writes under, relative to the cluster's shared root: common.sh f22_init's rule with the
#: native serving path (FD 19) — `$SHARED/exp/sdfbase_full22_native`; F22_WAVE_ROOT is set only for the grouped multi-turn units
WAVE_ROOT_REL = "exp/sdfbase_full22_native"
#: jailbreak_run.py's gated stores: `<PRIVATE_ROOT>/<private_run>/<attack>/<cell>/shardNN/{progress,responses}.jsonl`, PRIVATE_ROOT =
#: `$WAVE_ROOT/transcripts_results` (target_static / target_attacks / sampled/target_sampled.sbatch); the padded statics under
#: `$WAVE_ROOT/static_padding/transcripts_results/pad_static_<key>_p<pad>/...` (static_padding/target_static_padded.sbatch)
PRIVATE_ROOT_REL = "transcripts_results"
PADDING_PRIVATE_ROOT_REL = "static_padding/transcripts_results"
#: the tinycap CellStore tasks the capability rows carry (cell_controls_run.JUDGED_TASK_OF; tiers/parity.PARITY_TASK_MMLU;
#: contracts.GSM8K_STANDARD_TASK) — the probe counts unique item ids OF THE TASK, so a store holding two controls never pools
CAPABILITY_TASK = {"xstest": "judged/xstest_v2", "or_bench_hard": "judged/or_bench_hard_1k", "mmlu": "s2parity/mmlu_letter",
                   "gsm8k": "inspect/gsm8k_paired200"}
#: misalign_proxy.sbatch's agentic family: `--n 10 --behaviors leaking murder` (AGENTIC_ARGS), stores `$SHARED/exp/misalign/<TAG>/agentic/<cell>/responses.jsonl`
AGENTIC_N_SAMPLES = 10
AGENTIC_BEHAVIOR = {"agentic_leaking": "leaking", "agentic_murder": "murder"}
#: the Petri proxy's per-cell store on the GRADING host (petri_drive_full22.sh drives petri_run.py there; the petri_stats adapter records
#: store_uri `~/.local/state/petri/petri_full22_<view>/cells/<run key>`): coverage_probe.py's `petri_cellstore` layout over
#: `rows.jsonl` + the runner's failure / refusal ledgers (derived from the real stores' artifacts).
#: `petri_pooled` covers every behavior (the whole bundle); `petri_<behavior>` covers that behavior's scenarios
PETRI_BEHAVIORS = ("cyber_bio_chem_capabilities", "self_preservation", "delusion_sycophancy", "defer_to_users", "reasoning_unfaithfulness",
                   "instruction_evasion")                     # sisteer.capeval.manifest.PETRI_BEHAVIORS (pinned equal by tests/test_admission.py)
PETRI_BEHAVIOR_OF_EVAL: Dict[str, Optional[str]] = {"petri_pooled": None, **{f"petri_{b}": b for b in PETRI_BEHAVIORS}}
PETRI_LOCK_REL = os.path.join("configs", "capeval", "petri-scenarios.lock.yaml")
PETRI_BUNDLE_DIR_ENV = "SISTEER_PETRI_BUNDLE_DIR"           # the frozen bundle's local materialization (tests: a synthetic bundle)
PETRI_REPO_BUNDLE_REL = os.path.join("data", "petri", "scenarios")   # + /<content hash>: the published copy (scenarios.REPO_BUNDLE_ROOT_REL)


class PetriBundleAbsent(RuntimeError):
    """No copy of the frozen Petri scenario bundle anywhere the default lookup looks (nothing was named): the one condition the freeze
    answers by skipping the Petri family by name rather than refusing. A copy that is there and wrong stays a plain RuntimeError."""


#: the coverage layout per eval (lib/coverage_probe.py LAYOUTS); the BoN units and the padding levels override the private run / root
COVERAGE_LAYOUT = {**{e: "jailbreak_cellstore" for e in (*STATIC_EVALS, *MT_ATTACKS)}, "strongreject_bon": "bon_store512",
                   **{e: "capability_cellstore" for e in CAPABILITY_TASK}, **{e: "misalign_responses" for e in AGENTIC_EVALS},
                   **{e: "petri_cellstore" for e in PETRI_EVALS}}
#: how the gate BINDS a unit to its run ids: `native` — the job registers itself at job start (common.sh f22_registry_register:
#: `emit_cli register --from-config` under F22_REGISTRY_EVALS, the registry's own planner minting the canonical instrument per
#: model) and the binder reads pending/<job id>.json (every array task's). EVERY registering unit of the front door is native
#: (the spec: registry native at every step, never a post-hoc ingest) — the mmlu_full job included
REGISTRATION_KINDS = ("native",)


def write_atomic(path: str, obj) -> None:
    d = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(d, exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, sort_keys=True)
        fh.flush(); os.fsync(fh.fileno())
    os.replace(tmp, path)
    fsync_dir(d)      # the rename itself is durable only once the directory is (31e-fix-3 secondary / 30-fix MEDIUM)


def fsync_dir(d: str) -> None:
    """fsync a directory so a rename or an append inside it survives a crash (best effort: a filesystem that refuses is not an error)."""
    try:
        fd = os.open(d, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def append_fsync(path: str, line: str) -> None:
    """Append one line and fsync the file AND its directory: an append-only receipt that is on disk when this returns."""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(line if line.endswith("\n") else line + "\n"); fh.flush(); os.fsync(fh.fileno())
    fsync_dir(os.path.dirname(os.path.abspath(path)) or ".")


def _crash(point: str) -> None:
    """Fault injection for the recoverability tests: SISTEER_STAGE_CRASH_AT=<point>[,…] exits 97 at that point (as the funnel's)."""
    pts = {x for x in os.environ.get("SISTEER_STAGE_CRASH_AT", "").split(",") if x}
    if point in pts:
        print(f"[stage] CRASH INJECTED at {point}", file=sys.stderr, flush=True)
        os._exit(97)


def logical_work_id(model: str, view: str, family: str, unit: str) -> str:
    return f"{model}/{view}/{family}/{unit}"


def shards(n: int) -> List[str]:
    return [f"shard{i:02d}" for i in range(n)]


def shard_groups(nshards: int, spec: Optional[str] = None, smoke: Optional[Mapping[str, object]] = None) -> List[Tuple[int, ...]]:
    """The multi-turn shard groups (`SISTEER_SHARD_GROUPS`, default DEFAULT_SHARD_GROUPS): a DISJOINT cover of 0..nshards-1 with
    no empty group, refused otherwise (§3.9, review E P2). Under the SMOKE goal-shard knob (`smoke`, the overrides document's
    `OVERRIDE_SMOKE_GOALS=<lo>-<hi>/N` as :func:`smoke_goals` reads it) the groups are the smoke's ONE range lo..hi of N — the cover rule is
    relaxed HERE ONLY, while the knob is set, and the freeze pins the knob in the inventory header so `plan --inventory` refuses a plan
    that drops or changes it (demo-import M10; Q9 ruled); `SISTEER_SHARD_GROUPS` does not apply to a smoke's targets."""
    if smoke is not None:
        if int(smoke["nshards"]) != int(nshards):
            raise ValueError(f"shard_groups: nshards {nshards} is not the smoke knob's N {smoke['nshards']} ({smoke.get('spec')}) — the caller passes the smoke's N")
        return [tuple(range(int(smoke["lo"]), int(smoke["hi"]) + 1))]
    spec = spec if spec is not None else os.environ.get("SISTEER_SHARD_GROUPS") or " ".join(DEFAULT_SHARD_GROUPS)
    groups = [tuple(int(x) for x in g.split(",") if x != "") for g in spec.split()]
    flat = [s for g in groups for s in g]
    if not groups or any(len(g) == 0 for g in groups):
        raise ValueError(f"SISTEER_SHARD_GROUPS {spec!r}: an empty group")
    if sorted(flat) != list(range(nshards)):
        raise ValueError(f"SISTEER_SHARD_GROUPS {spec!r} is not a disjoint cover of 0..{nshards - 1} (got {sorted(flat)})")
    return groups


def coverage_spec(eval_id: str, *, private_run: Optional[str] = None, root_rel: Optional[str] = None, host: str = "cluster",
                  layout: Optional[str] = None, **more) -> dict:
    """The eval's coverage spec in a plan row / frozen unit: `layout` (coverage_probe.py), `host` (where the artifact is read:
    `cluster` = the unit's cluster over ssh, from the mirrored checkout's probe; `grading_host` = this host, for a store the
    registry's harvester mirrored), and the artifact's location: `store_uri` (the run row's recorded store — the tinycap CellStore
    hook records `cells/<key>`) or a `root_rel` under the cluster's shared root with `<cell>` for the run's object. The EXPECTED
    identities are attached by expected_units (the freeze); the plan carries the shape only."""
    spec = {"layout": layout or COVERAGE_LAYOUT[eval_id], "host": host, "path": {"from": "store_uri"} if root_rel is None else {"root_rel": root_rel},
            "private_run": private_run, "expected": None, "expected_source": None}
    if spec["layout"] == "capability_cellstore":
        spec["task"] = CAPABILITY_TASK[eval_id]
    if spec["layout"] == "misalign_responses":
        spec.update(behavior=AGENTIC_BEHAVIOR[eval_id], n_samples=AGENTIC_N_SAMPLES)
    if spec["layout"] == "petri_cellstore":
        spec["behavior"] = PETRI_BEHAVIOR_OF_EVAL[eval_id]          # None = the pooled eval: every behavior of the bundle
    spec.update(more)
    return spec


def _row(stage_family: str, family: str, unit: str, model: str, view: str, evals: List[str], nshards_by_eval: Dict[str, int], *,
         script: str, name: str, args: List[str], env: Optional[Dict[str, str]] = None, array: Optional[str] = None,
         time_key: str = "eval", kind: str = "target", group: Optional[str] = None, cluster: str = "*", twin_ok: bool = True,
         padding_tokens: Optional[int] = None, registers: bool = True, lwid: Optional[str] = None, registration: str = "native",
         coverage: Optional[Dict[str, dict]] = None, depends_on: Optional[str] = None) -> dict:
    env = dict(env or {})
    if registration not in REGISTRATION_KINDS:
        raise ValueError(f"unknown registration kind {registration!r}")
    if registers:
        env.setdefault("F22_REGISTRY_EVALS", ",".join(evals))      # the job registers EXACTLY the unit's evals (never a job-name guess)
    return {"stage_family": stage_family, "family": family, "unit": unit, "logical_work_id": lwid or logical_work_id(model, view, family, unit),
            "private_run": None, "evals": list(evals), "expected_shards": {e: shards(nshards_by_eval[e]) for e in evals},
            "coverage": dict(coverage or {}), "registration": registration if registers else None,
            "script": script, "name": name, "args": [str(a) for a in args], "env": env, "array": array, "time_key": time_key,
            "kind": kind, "group": group, "cluster": cluster, "twin_ok": twin_ok, "padding_tokens": padding_tokens, "registers": registers,
            "depends_on": depends_on}


#: the Petri proxy's sbatch, the walltime key its rows carry (06: `petri` -> the site's SISTEER_TIME_PETRI, a day by default, as the paper
#: ran every proxy; every other eval row's `eval` -> SISTEER_TIME_EVAL — the demo's 04:00:00 eval walltime cut a full drive at 521 of 660
#: scenarios, 2026-10-13), and the two PASSES a 06 petri call plans: `first` (the proxy at the certified window — the row the freeze
#: inventories) and `overflow` (the CONTEXT-OVERFLOW RE-RUN PASS: a second proxy at petri_runner.rerun_serving_geometry()'s doubled window
#: for the legs the first pass left on overflow ledgers — `petri_proxy.sbatch … overflow-rerun <cells>`, driven by the grading daemon under
#: F22_OVERFLOW_RERUN=1; a follow-up unit, never inventoried, PINNED to the cluster the first pass is bound on — the object bank and the bank
#: view the proxy's standup requires are staged per cluster (05 --clusters), so a `*` row expanded to every label and taken by the funnel's
#: first label ran where nothing was staged)
PETRI_PROXY_SCRIPT = "scripts/sdfbase/full22/petri_proxy.sbatch"
PETRI_TIME_KEY = "petri"
PETRI_PASSES = ("first", "overflow")
PETRI_OVERFLOW_UNIT = "overflow"
PETRI_OVERFLOW_PASS_ARG = "overflow-rerun"

#: the jobs that take a CELLS positional (cells-fix): script -> the 0-based index of that positional in the row's args
CELLS_POSITIONAL = {"scripts/sdfbase/full22/target_attacks.sbatch": 6,            # <key> <view> <attack> <sha> [nshards] [shards] [cells] [window]
                    "scripts/sdfbase/full22/sampled/target_sampled.sbatch": 4,    # <key> <phase> <sha> [nshards] [cells] [view]
                    "scripts/sdfbase/full22/target_static.sbatch": 5,    # <key> <view> <families> <sha> [nshards] [cells] — cells-fix-2: 4 is NSHARDS there
                    "scripts/sdfbase/full22/petri_proxy.sbatch": 4,      # <key> <view> <sha> [pass] [cells] — petri-cells: the pass stays positional 4
                    "scripts/sdfbase/full22/bon_climb.sbatch": 4}        # <key> <view> <sha> <nshards> [cells] [cap] — the rerun climb serves its ruling's cells alone
CELLS_ENV = "F22_REGISTRY_CELLS"


def cells_selection(raw: Optional[str]) -> Optional[List[str]]:
    """The CLI's `--cells` value as the selection (cells-fix-2): None when the flag is ABSENT (the view's full enumeration); a given
    value is split on commas and every token must name a cell — '' , ',' and 'a,,b' refuse by name instead of collapsing to "no selection"
    (which launched the full config, `none` included, from an empty generated selector). Validation against the view's explicit_cells is
    :func:`apply_cells_selection`'s."""
    if raw is None:
        return None
    toks = [t.strip() for t in str(raw).split(",")]
    if any(not t for t in toks):
        raise ValueError(f"--cells {raw!r}: an empty selection or a blank token — name every cell, or omit --cells for the view's full enumeration")
    return toks


def view_cells_of(model_key: str, view: str, *, roster_path: Optional[str] = None, table: Optional[str] = None) -> List[str]:
    """The view config's `explicit_cells` ("the registry is EXACTLY this"), found through the roster
    (`targets.<key>.views.<view>.config`; $F22_ROSTER or configs/sdfbase_full22_roster.yaml) — the ONE list a `--cells` selection is held to."""
    import yaml
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    rp = roster_path or os.environ.get("F22_ROSTER") or os.path.join(repo, "configs", "sdfbase_full22_roster.yaml")
    roster = yaml.safe_load(open(rp, encoding="utf-8"))
    try:
        cfg_rel = roster["targets"][model_key]["views"][view]["config"]
    except (KeyError, TypeError):
        raise ValueError(f"--cells: the roster {rp} names no config for {model_key}/{view}")
    cfg_path = cfg_rel if os.path.isabs(cfg_rel) else os.path.join(repo, cfg_rel)
    cfg = yaml.safe_load(open(cfg_path, encoding="utf-8")) or {}
    cells = [str(c) for c in (cfg.get("explicit_cells") or [])]
    if not cells:
        raise ValueError(f"--cells: {cfg_path} enumerates no explicit_cells")
    if M.steered_wave_view(view, model_key, table):   # Target-aware — a front-door battery / s2 / dose view is a steered wave by its target, not by the `s<N>r` name
        bad = [c for c in cells if c in M.UNSTEERED_LEGS]
        if bad:
            # the generated config of a STEERED-WAVE view carries an unsteered leg — a plan over it would serve, register and
            # score a `none` run, which the rule forbids; refused BY NAME here, where default planning (06, no --cells) reads the view
            raise ValueError(f"{model_key}/{view}: the generated config {cfg_rel} lists the unsteered leg(s) {bad} on a steered-wave view — a steered-wave "
                             "view measures no `none` (D1152 / D1154); regenerate the wave (make_full22_configs.py) and its dependents, never launch over it")
    return cells


def refuse_unsteered_legs_on_steered_wave_view(model_key: str, view: str, *, roster_path: Optional[str] = None, table: Optional[str] = None) -> None:
    """Default stage-06 planning of a STEERED-WAVE view (s<N>r) refuses by name when any of its GENERATED artefacts carries
    an unsteered leg — the wave config (``view_cells_of``), the Petri manifest's ``cells`` and the front door's GSM8K matrix rows of the view
    (`none` / `none__lora0` / `none__lens0`). The artefacts the launchers read are the witnesses: a plan that passed here cannot launch, register
    or score an unsteered run on the view. An ordinary view is untouched."""
    if not M.steered_wave_view(view, model_key, table):
        return
    import yaml
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    rp = roster_path or os.environ.get("F22_ROSTER") or os.path.join(repo, "configs", "sdfbase_full22_roster.yaml")
    roster = yaml.safe_load(open(rp, encoding="utf-8"))
    carried = (((roster.get("targets") or {}).get(model_key) or {}).get("views") or {}).get(view) or {}
    if not carried.get("config"):
        # the roster does not carry the view yet (05 --step roster / configs come first) — no generated artefact exists to witness, so none
        # can carry an unsteered leg; 06's dry run composes its lines regardless (its own note), a real launch refuses on the missing config by name
        return
    view_cells_of(model_key, view, roster_path=roster_path, table=table)         # the wave config (refuses by name inside)
    man_rel = roster["targets"][model_key]["views"][view].get("manifest")
    if man_rel:
        man_path = man_rel if os.path.isabs(man_rel) else os.path.join(repo, man_rel)
        if os.path.isfile(man_path):
            man = yaml.safe_load(open(man_path, encoding="utf-8")) or {}
            mcells = [str(c) for c in ((man.get("petri") or {}).get("cells") or [])]
            bad = [c for c in mcells if c in M.UNSTEERED_LEGS]
            if bad:
                raise ValueError(f"{model_key}/{view}: the generated Petri manifest {man_rel} lists the unsteered leg(s) {bad} on a steered-wave view "
                                 "(D1152 / D1154) — regenerate the wave, never drive over it")
    mx = os.path.join(repo, "configs", "sdfbase", "gsm8k", f"matrix_frontdoor_{model_key}.json")
    if os.path.isfile(mx):
        rows = (json.load(open(mx, encoding="utf-8")) or {}).get("rows") or []
        bad_rows = [r.get("cell") for r in rows if r.get("view") == view and str(r.get("cell")) in M.UNSTEERED_LEGS]
        if bad_rows:
            raise ValueError(f"{model_key}/{view}: the GSM8K matrix {os.path.relpath(mx, repo)} carries the unsteered row(s) {bad_rows} on a steered-wave "
                             "view (D1152 / D1154) — regenerate the matrix (05 --step configs), never launch over it")


def apply_cells_selection(rows: List[dict], cells: Sequence[str], model_key: str, view: str, *, roster_path: Optional[str] = None,
                          table: Optional[str] = None) -> List[dict]:
    """cells-fix: the ONE selection, validated against the view's explicit_cells (non-empty, no repeats, every cell enumerated), written into
    every registering TARGET row whose job takes a cells positional (:data:`CELLS_POSITIONAL`) and into its env as :data:`CELLS_ENV`, so the
    job serves and registers exactly these cells; pool members carry no cells; a row whose job has no cells positional REFUSES by name (a
    selection it cannot honour would launch the full config, `none` included). Ids, groups, clusters and receipts are untouched."""
    sel = [str(c) for c in cells]
    if not sel:
        raise ValueError("--cells: an empty selection (name at least one cell, or omit --cells for the view's full enumeration)")
    if view == M.baseline_view(model_key, table) and any(CELLS_POSITIONAL.get(r["script"]) is not None and r["script"].endswith("petri_proxy.sbatch") for r in rows if r.get("registers", True)):
        # the BASELINE wave's audit legs are the WHOLE baseline (its frozen expected_cells
        # advertises the manifest; u0, or the single-phase recipe's one wave), so its Petri proxy is never stood up with a partial --cell —
        # the drive would refuse the declaration; refuse at the plan
        raise ValueError(f"--cells on {model_key}/{view}'s petri unit: the baseline wave's audit legs are the whole baseline (every canonical leg, none included) — "
                         f"a partial declaration narrows it; launch the {view} petri proxy without --cells")
    if len(set(sel)) != len(sel):
        raise ValueError(f"--cells: a cell is named twice ({sel})")
    have = view_cells_of(model_key, view, roster_path=roster_path, table=table)
    unknown = [c for c in sel if c not in have]
    if unknown:
        raise ValueError(f"--cells: {unknown} are not in {model_key}/{view}'s explicit_cells {have}")
    for r in rows:
        if r.get("kind") == "attacker" or not r.get("registers", True):
            continue
        if r.get("rerun_of"):
            # a rerun row serves exactly the cells its stamped declaration names (written by _rerun_rows): a --cells that differs
            # would re-aim a ruling's rerun — refused by name; the same selection is a no-op
            if list(r.get("cells") or []) != sel:
                raise ValueError(f"--cells {sel} on the rerun unit {r['unit']}: its cells are its stamped declaration's {list(r.get('cells') or [])} "
                                 f"({r.get('rerun_source')}) — a rerun serves exactly the cells its ruling names; launch it without --cells, or name exactly those")
            continue
        if CELLS_POSITIONAL.get(r["script"]) is None:
            raise ValueError(f"--cells is not plumbed for family {r['family']!r} (unit {r['unit']}: {r['script']} takes no cells positional) — "
                             f"launch that family without --cells or extend the selector; nothing planned")
        write_cells_into(r, sel)
    return rows


def write_cells_into(r: dict, sel: Sequence[str]) -> dict:
    """The ONE writer of a row's cells: the CSV on the script's cells positional (:data:`CELLS_POSITIONAL`, padded), :data:`CELLS_ENV` in its
    env, ``cells`` on the row — apply_cells_selection's selection and the steered-wave Petri proxy's default write the same shape."""
    sel = [str(c) for c in sel]
    if not sel:
        raise ValueError(f"{r.get('unit')}: an empty cells list for {r['script']}")
    idx = CELLS_POSITIONAL[r["script"]]
    csv = ",".join(sel)
    args = list(r["args"])
    while len(args) <= idx:
        args.append("")
    args[idx] = csv
    r["args"] = args
    r["env"] = {**dict(r.get("env") or {}), CELLS_ENV: csv}
    r["cells"] = list(sel)
    return r


# --------------------------------------------------------------------------------------------- the BoN RERUN pair of a started wave
#: The demo smoke's 512 store (job 983259, view battery) served prompted__si_constitution at 4,864 tokens; every one of its 7,500 slots was
#: refused prompt_over_window and the closer filed its designed no-fit certificate for a cell that was mis-served (the paper's u0 store
#: 808185 before it). The ruling re-runs the cell under a window that fits and names the door: NEW funnel units
#: `store512_constitution_rerun -> climb_constitution_rerun` (afterok), the old run superseded by ruling and untouched. Why new units:
#: `--resubmit` refuses a COMPLETED registering unit (its runs are the unit's results), `--cells` keeps the logical work id, and a same-config
#: job would resume the SAME store dir (jailbreak_run._bon_defense_dir: <private root>/<private_run>/<cell>, where bon_errors.jsonl holds
#: the 7,500 refusals under a `complete: true` marker) and register the SAME job-independent run id (`sdfbase_full22-stat-3-<tag>`)
#: — old refusals and new responses sharing one store and one identity. The pair therefore carries THREE distinctions, each on the plan row:
#: its own logical work id (the inventory's unit), its own wave root (BON_RERUN_ENV -> `<campaign root>/rerun/<slug>`, where the job's
#: transcripts_results, bon_climb, serving dirs, the closer's --wave-root and the STOP sentinel live — the per-group pool roots' rule, §3.9;
#: the coverage root_rel spells the same place), and its own producer key (F22_REGISTRY_KEY_SEGMENTS = the slug, exported by the same
#: helper). It serves exactly the ruling's cells (the cells positional + F22_REGISTRY_CELLS, write_cells_into). A plan carries the pair only
#: for a declaration the frozen inventory was STAMPED with (`rerun-units`; `plan --inventory` reads `reruns`): the paper's and the demo's
#: default rows are byte-identical, a fresh wave plans no rerun (it serves the cell at the corrected window itself), and 06's base-view check
#: (frozen units == planned units, per family) holds because the stamp and the plan read one declaration.
BON_RERUN_OF = ("store512", "climb")
RERUN_SLUG_RE = re.compile(r"^[A-Za-z0-9_]+$")


def rerun_slug(cells: Sequence[str]) -> str:
    """The pair's slug from its cells: each cell's short name (`prompted__si_<si>` -> `<si>`, `prompted__<p>` -> `<p>`, every other run of
    non-alphanumerics -> `_`), joined by `_`, + `_rerun` — `constitution_rerun` for prompted__si_constitution. One token of
    [A-Za-z0-9_] (a unit name, a directory name, a producer-key segment and an env value at once)."""
    shorts = []
    for c in cells:
        s = str(c)
        for pre in ("prompted__si_", "prompted__"):
            if s.startswith(pre):
                s = s[len(pre):]; break
        s = re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_")
        if not s:
            raise ValueError(f"rerun slug: cell {c!r} leaves no name")
        shorts.append(s)
    slug = "_".join(shorts) + "_rerun"
    if not RERUN_SLUG_RE.match(slug):
        raise ValueError(f"rerun slug {slug!r} is not one token of [A-Za-z0-9_]")
    return slug


def bon_rerun_declaration(cells: Sequence[str], source: str, model_key: str, view: str, *, roster_path: Optional[str] = None,
                          table: Optional[str] = None) -> dict:
    """The validated declaration a stamp records and a plan reads: {slug, cells, source, model, view, of: {store512: lwid, climb: lwid},
    units: {store512: rerun lwid, climb: rerun lwid}}. Refuses by name an empty / repeated selection, a cell outside the view's
    explicit_cells (view_cells_of), and a missing source — the ruling (who, when) is the provenance, not an option."""
    sel = [str(c) for c in cells]
    if not sel:
        raise ValueError("rerun: an empty cell selection (name the cell(s) the ruling re-runs)")
    if len(set(sel)) != len(sel):
        raise ValueError(f"rerun: a cell is named twice ({sel})")
    src = str(source or "").strip()
    if not src:
        raise ValueError("rerun: no --source — a rerun carries the ruling it came from (who, when; e.g. `D913 (ii)`), or it is not stamped")
    have = view_cells_of(model_key, view, roster_path=roster_path, table=table)
    unknown = [c for c in sel if c not in have]
    if unknown:
        raise ValueError(f"rerun: {unknown} are not in {model_key}/{view}'s explicit_cells {have}")
    slug = rerun_slug(sel)
    return {"slug": slug, "cells": sel, "source": src, "model": model_key, "view": view,
            "of": {u: logical_work_id(model_key, view, "bon", u) for u in BON_RERUN_OF},
            "units": {u: logical_work_id(model_key, view, "bon", f"{u}_{slug}") for u in BON_RERUN_OF}}


def inventory_reruns(inv: Optional[Mapping[str, object]], view: str) -> Optional[List[dict]]:
    """The rerun declarations a frozen inventory was stamped with FOR `view` (`reruns`, written by `rerun-units`), or None — a plan of
    another view under the same inventory (06 passes the baseline inventory for every view) carries no rerun rows."""
    decls = [dict(d) for d in ((inv or {}).get("reruns") or []) if isinstance(d, Mapping) and str(d.get("view")) == str(view)]
    return decls or None


def _rerun_rows(decl: Mapping[str, object], model_key: str, view: str, tag: str, sha: str, nshards: int, wave_private_run: str,
                one: Dict[str, int], *, store: bool, climb: bool) -> List[dict]:
    """The pair's plan rows for one declaration: the store under `--family static` (the recipe's store512 row's sibling: target_static.sbatch,
    FAMILIES=bon, a `f22-stat-<tag>-<slug>-3` name — the `-3` suffix is what common.sh's job tables key the strongreject_bon eval and the
    `stat-3` producer-key family on) and the climb under `--family bon` (bon_climb.sbatch, array 0-3, afterok on the rerun STORE). Both carry
    BON_RERUN_ENV, the ruling's cells on their cells positional + F22_REGISTRY_CELLS, `rerun_of` and `rerun_source`; their coverage reads the
    rerun's own wave root (`<WAVE_ROOT_REL>/rerun/<slug>/...`), exactly where the re-rooted job writes."""
    slug, cells, source = str(decl["slug"]), [str(c) for c in decl["cells"]], str(decl["source"])
    root = f"{WAVE_ROOT_REL}/{RERUN_ROOT_REL}/{slug}"
    env = {BON_DECODE_PRESET_ENV: BON_DECODE_PRESET, BON_RERUN_ENV: slug}
    rows: List[dict] = []
    if store:
        rows.append(_row("static", "bon", f"store512_{slug}", model_key, view, ["strongreject_bon"], one,
                         script="scripts/sdfbase/full22/target_static.sbatch", name=f"f22-stat-{tag}-{slug}-3", args=[model_key, view, "bon", sha, nshards],
                         env=dict(env),
                         coverage={"strongreject_bon": coverage_spec("strongreject_bon", private_run=wave_private_run,
                                                                     root_rel=f"{root}/{PRIVATE_ROOT_REL}/{wave_private_run}/<cell>")}))
        rows[-1]["rerun_of"] = logical_work_id(model_key, view, "bon", "store512")
    if climb:
        rows.append(_row("bon", "bon", f"climb_{slug}", model_key, view, ["strongreject_bon"], {"strongreject_bon": 4},
                         script="scripts/sdfbase/full22/bon_climb.sbatch", name=f"f22-bonclimb-{tag}-{slug}", args=[model_key, view, sha, 4], array="0-3",
                         env=dict(env),
                         depends_on=logical_work_id(model_key, view, "bon", f"store512_{slug}"),
                         coverage={"strongreject_bon": coverage_spec("strongreject_bon", layout="bon_climb_flat", private_run=wave_private_run,
                                                                     root_rel=f"{root}/bon_climb/{tag}", nshards=4,
                                                                     expected_source="run-time: plan/climb_manifest.jsonl (bon_climb_plan.py, written by array task 0)")}))
        rows[-1]["rerun_of"] = logical_work_id(model_key, view, "bon", "climb")
    for r in rows:
        write_cells_into(r, cells)
        r["rerun_source"] = source
    return rows


def unit_plan(model_key: str, view: str, *, table: Optional[str] = None, families: Optional[List[str]] = None, nshards: int = 9,
              sha: str = "{sha}", clusters: Optional[List[str]] = None, shard_groups_spec: Optional[str] = None,
              mt_attacker_gpus: Optional[int] = None, mt_seed_base: Optional[str] = None,
              only_clusters: Optional[List[str]] = None, move_groups: Optional[Mapping[str, str]] = None,
              cells: Optional[Sequence[str]] = None, smoke_goals: Optional[Mapping[str, object]] = None,
              reruns: Optional[Sequence[Mapping[str, object]]] = None, pass_: str = "first",
              overflow_cells: Optional[Sequence[str]] = None, overflow_cluster: Optional[str] = None) -> List[dict]:
    """The submission units of (model, view) — the ONE source both freeze-inventory and 06_eval_waves.sh read. `clusters` is the
    PLACEMENT universe: the multi-turn groups are partitioned over it (None = placement-free, the freeze's view); every other unit
    carries cluster `*` = the stage's list. `only_clusters` SELECTS which placed units the caller submits this call (a placed unit
    on another label is dropped; `*` units stay) — it never re-partitions. The distinction is load-bearing: 06 used to hand its
    `--clusters` subset in as the universe, so a call naming the FIRST configured label alone re-partitioned all twelve targets onto
    it and moved crescendo/g34's group from the second label's pool to the first's; the funnel's workload-spec proof refused it
    (2026-09-30 18:17Z). The universe is every configured label, the same call to call, so a unit's placement is a
    function of the site. `smoke_goals` (the overrides document's goal-shard knob as :func:`smoke_goals` reads it; None = the paper's
    rows, byte for byte) narrows the multi-turn targets and the per-view padding levels to shards lo..hi of N and touches no other row.
    `reruns` (the frozen inventory's stamped BoN rerun declarations for THIS view — :func:`inventory_reruns`; None = no rerun rows) adds one
    `bon/store512_<slug>` row under `static` and one `bon/climb_<slug>` row under `bon` per declaration (:func:`_rerun_rows`)."""
    m = M.model(model_key, table)
    model_id = m["serve"]["checkpoint"]["model_id"]
    tag = model_key if view == "main" else f"{model_key}_{view}"
    if M.steered_wave_view(view, model_key, table) and not os.environ.get("SISTEER_PLAN_SKIP_ARTEFACT_CHECK"):   # The guard runs for every front-door non-u0 view
        # s3r-fix: default planning of a steered-wave view reads the view's generated artefacts and refuses an unsteered leg by name
        # (the knob is for tests that plan a view whose artefacts are not on their synthetic tree — never set in production)
        refuse_unsteered_legs_on_steered_wave_view(model_key, view, table=table)
    fams = list(families) if families else list(FAMILIES)
    for f in fams:
        if f not in FAMILIES:
            raise ValueError(f"unknown family {f!r} ({' '.join(FAMILIES)})")
    if pass_ not in PETRI_PASSES:
        raise ValueError(f"--pass must be one of {', '.join(PETRI_PASSES)}, got {pass_!r}")
    if pass_ == "overflow":
        # the Petri OVERFLOW RE-RUN PASS (petri-completion, 2026-10-13): ONE follow-up unit beside the inventoried first pass — the
        # rerun proxy for the cells 06 derived from the daemon's record of the first pass (lib/petri_overflow.py); never another family's
        # row, never a --cells selection (the cells ARE the ledgered legs')
        if fams != ["petri"]:
            raise ValueError("--pass overflow is the Petri overflow re-run pass: --family petri, no other family")
        if cells is not None:
            raise ValueError("--pass overflow takes no --cells: the re-run proxy serves the cells of the legs the first pass left on context-overflow "
                             "ledgers (derived from the drive's record), never a selection")
        # THE CLUSTER: the rerun proxy runs on the FIRST PASS's bound cluster — 05 --clusters stages the object bank and builds
        # the bank view per cluster, and the proxy's standup requires both (petri_proxy.sbatch f22_bank_view); a `*` row expanded to every
        # configured label and the funnel's first label took it. The caller names it (06, from lib/petri_overflow.py's first_cluster; the row
        # is pinned there, never twinned); a universe or a selection (--only-clusters) that excludes it refuses by name, never re-places the proxy
        if overflow_cluster and overflow_cluster != "*":      # `*` is the stage's list, not a label: overflow_rows refuses it by name below
            if clusters is not None and overflow_cluster not in clusters:
                raise ValueError(f"--pass overflow: the first pass's cluster {overflow_cluster!r} is not in the placement universe {clusters} — the re-run proxy "
                                 "runs where the first pass ran (its object bank and bank view are there); restore the label in the site's list")
            if only_clusters is not None and overflow_cluster not in only_clusters:
                raise ValueError(f"--pass overflow: the re-run proxy follows the first pass's cluster {overflow_cluster!r}, but --only-clusters selects "
                                 f"{only_clusters} — a selection that excludes it is refused; drop --clusters or name {overflow_cluster}")
        return overflow_rows(model_key, view, list(overflow_cells or []), cluster=overflow_cluster, sha=sha, table=table)
    n_all = {e: nshards for e in (*STATIC_EVALS, *MT_ATTACKS)}
    # the shard vocabulary is the PRODUCER's: the tinycap CellStore families (controls, mmlu, gsm8k) and Petri have no shards at
    # all (one store per cell — the coverage is the item set); the BoN 512 store is ONE append-only file per cell (`shard00` names
    # it); the agentic runner ONE responses.jsonl per cell; the climb's `shardNN` is the array task index
    none = {e: 0 for e in (*PETRI_EVALS, *CONTROL_EVALS, "mmlu", "gsm8k")}
    one = {e: 1 for e in (*AGENTIC_EVALS, "strongreject_bon")}
    wave_private_run = f"sdfbase_full22_{tag}"      # make_full22_configs.render_config: `private_run: sdfbase_full22_<tag>` (the wave config)
    jb = lambda e, run, root=PRIVATE_ROOT_REL: coverage_spec(e, private_run=run, root_rel=f"{WAVE_ROOT_REL}/{root}/{run}/{e}/<cell>")  # noqa: E731
    rows: List[dict] = []
    if "static" in fams:
        # The four statics run in the SAMPLED lane's form: `target_sampled.sbatch static_base_t<T>` for the
        # two seeded trials, at the paper's canonical sampled decode (T = 1.0) from the sampled config
        # `configs/sdfbase_full22_<TAG>_sampled_t<T>.yaml` — NOT the wave form `target_static.sbatch` with `sdfbase_full22_<TAG>.yaml`:
        # statics are graded through sampled/grade_static_sampled.py, whose RUN_RE admits only `sdfbase_full22_<key>_sampled_t<T>`
        # private runs, so a wave-form static store would be refused as "not a sampled private_run" and never graded. The unit's
        # private_run is that identity. The BoN 512 store keeps submit_full22.sh's third job (f22-stat-*-3, its own grader).
        for t in (0, 1):
            run = f"sdfbase_full22_{tag}_sampled_t{t}"
            rows.append(_row("static", "static", f"t{t}", model_key, view, list(STATIC_EVALS), n_all,
                             script="scripts/sdfbase/full22/sampled/target_sampled.sbatch", name=f"f22s-{model_key}-{tag}-static_base_t{t}",
                             args=[model_key, f"static_base_t{t}", sha, nshards, "", view], coverage={e: jb(e, run) for e in STATIC_EVALS}))
            rows[-1]["private_run"] = run
        # the 512 store: bon_served_run.py writes ONE `bon_responses.jsonl` per cell under `<private run>/<cell>` (jailbreak_run._bon_defense_dir),
        # n_max draws per goal of the config's `bon.goal_subset` (jbb; two_si_run._eval_goals(attack="bon"))
        rows.append(_row("static", "bon", "store512", model_key, view, ["strongreject_bon"], one,
                         script="scripts/sdfbase/full22/target_static.sbatch", name=f"f22-stat-{tag}-3", args=[model_key, view, "bon", sha, nshards],
                         env={BON_DECODE_PRESET_ENV: BON_DECODE_PRESET},
                         coverage={"strongreject_bon": coverage_spec("strongreject_bon", private_run=wave_private_run,
                                                                     root_rel=f"{WAVE_ROOT_REL}/{PRIVATE_ROOT_REL}/{wave_private_run}/<cell>")}))
        # the rerun STORE(s) of a stamped declaration ride `--family static` as the store512 row does (its stage family)
        for decl in (reruns or []):
            rows.extend(_rerun_rows(decl, model_key, view, tag, sha, nshards, wave_private_run, one, store=True, climb=False))
    excl = ((m.get("views") or {}).get(view) or {}).get("exclude_families") or []
    if "bon" in excl:
        # a held BoN family via the table's exclude_families: the 512 store rides `--family static` as its store512 row — dropped here
        # for an excluding view, so the exclusion holds in the PLAN and not only at a launch's --units; naming `bon` outright is refused
        if "bon" in fams:
            raise ValueError(f"{model_key}/{view}: the bon family is EXCLUDED for this view by ruling "
                             f"({((m.get('views') or {}).get(view) or {}).get('exclude_reason')}) — not planned; submit the other families")
        rows = [r for r in rows if r.get("family") != "bon"]
    if "padding" in fams and "padding" in excl:
        # the models table EXCLUDES the padding family from this view by ruling — refused BY NAME, never planned
        # (06's `--family all` drops it with a note; naming it is a mistake the plan says so about)
        raise ValueError(f"{model_key}/{view}: the padding family is EXCLUDED for this view by ruling "
                         f"({((m.get('views') or {}).get(view) or {}).get('exclude_reason')}) — not planned; submit the other families")
    if "padding" in fams:  # FIVE jobs per view, ONE level each (close round E P1): the same 98,304 window / manifest / cap for every level
        pad_n = eval_nshards("padding", nshards, smoke_goals)                 # the shard TOTAL: the filed rungs' 3, or the smoke's N
        # the smoke's range on the launcher's `shards` positional (`<lo>-<hi>/N`: this job owns shards lo..hi of N, the store comes out as
        # shard<lo>..shard<hi>); the paper's rows keep the single-job `3` byte for byte
        pad_shards_arg = smoke_goals["spec"] if smoke_goals else PADDING_NSHARDS
        for pad in PADDING_LEVELS:
            if view == "u0":
                if smoke_goals:
                    # the u0 launcher (target_static_padded.sbatch) takes a plain shard COUNT (`nshards=3`) and refuses anything else, so the
                    # knob cannot reach it: refused by name at the plan, never a job that dies on its positional (the smoke is the single-phase
                    # worked example's; a phased table's u0 padding runs whole or not at all)
                    raise ValueError(f"{model_key}/{view}: {SMOKE_GOALS_KEY}={smoke_goals['spec']} does not reach the u0 padding launcher "
                                     f"(target_static_padded.sbatch takes a shard count, not a range) — the smoke goal-shard knob narrows the per-view "
                                     f"padding levels (target_static_padded_obj.sbatch) only; plan u0's padding without the knob, or leave the family out")
                prun = f"pad_static_{model_key}_p{pad}"
                rows.append(_row("padding", "padding", f"p{pad}", model_key, view, list(STATIC_EVALS), {e: 3 for e in STATIC_EVALS},
                                 script="scripts/sdfbase/full22/static_padding/target_static_padded.sbatch", name=f"padstat-{model_key}-p{pad}",
                                 args=[model_key, sha, PADDING_WINDOW, pad, ",".join(STATIC_EVALS), 3], padding_tokens=pad,
                                 coverage={e: jb(e, prun, PADDING_PRIVATE_ROOT_REL) for e in STATIC_EVALS}))
            else:
                # a STEERED view pads through the sbatch's config-family positional: the front door's PER-VIEW steered family, NAMED
                # after the view it pads (make_padding_configs.front_door_view_families — the view's own roster
                # cells, the production bank view, no lane roster, no cross-bank pin), so every row here has a generated config
                # `sdfbase_full22_<key>.padstatic_<view>_p<pad>.yaml` (`obj` is qwen35_4b's family, never a
                # front-door key's)
                # the store identity the obj launcher files (target_static_padded_obj.sbatch RUN=pad_static_<key>_<config family>_p<pad>; the
                # config family IS the view's name) — the coverage the freeze binds when this wave is the baseline (the single-phase recipe)
                prun = f"pad_static_{model_key}_{view}_p{pad}"
                rows.append(_row("padding", "padding", f"p{pad}", model_key, view, ["grok420"], {"grok420": pad_n},
                                 script="scripts/sdfbase/full22/static_padding/target_static_padded_obj.sbatch", name=f"padobj-{tag}-p{pad}",
                                 args=[model_key, sha, PADDING_WINDOW, pad, "grok420", pad_shards_arg, "all", view], padding_tokens=pad,
                                 coverage={"grok420": jb("grok420", prun, PADDING_PRIVATE_ROOT_REL)}))
                if smoke_goals:
                    rows[-1]["expected_shards"] = {"grok420": list(smoke_goals["shards"])}      # this job owns the smoke's shards of N, nothing else
    if "bon" in fams:      # the 4,096 climb of the cap-coherent at-cap draws (array; shard 0 writes the plan) — after the static -3 job's 512 store
        # ONE array task = ONE shard = FLAT `$WAVE_ROOT/bon_climb/<TAG>/out/<cell>/shard<k>.jsonl` (bon_climb.sbatch; the producer is not
        # changed): `shardNN` here is the task index k; every task registers its own runs (pending/<raw task id>.json), so the
        # binder enumerates the array's tasks; the expected draws are the run-time climb manifest's, read by the probe
        # the climb ESCALATES the 512 store the static -3 job writes: it runs after that unit (Slurm --dependency=afterok on its bound
        # job, resolved by the funnel at submit time), so no reader ever meets a store being written
        rows.append(_row("bon", "bon", "climb", model_key, view, ["strongreject_bon"], {"strongreject_bon": 4},
                         script="scripts/sdfbase/full22/bon_climb.sbatch", name=f"f22-bonclimb-{tag}", args=[model_key, view, sha, 4], array="0-3",
                         env={BON_DECODE_PRESET_ENV: BON_DECODE_PRESET},
                         depends_on=logical_work_id(model_key, view, "bon", "store512"),
                         coverage={"strongreject_bon": coverage_spec("strongreject_bon", layout="bon_climb_flat", private_run=wave_private_run,
                                                                     root_rel=f"{WAVE_ROOT_REL}/bon_climb/{tag}", nshards=4,
                                                                     expected_source="run-time: plan/climb_manifest.jsonl (bon_climb_plan.py, written by array task 0)")}))
        # the rerun CLIMB(s): after the rerun STORE (afterok on it), the rerun's own climb root, the ruling's cells alone
        for decl in (reruns or []):
            rows.extend(_rerun_rows(decl, model_key, view, tag, sha, nshards, wave_private_run, one, store=False, climb=True))
    if "multiturn" in fams:   # per named cluster ONE GROUP (attacker members + the target units placed there); targets partitioned disjointly
        # the shard TOTAL the targets slice goals[s::N] over, and the groups: the plan's nshards under SISTEER_SHARD_GROUPS' disjoint cover, or
        # — under the smoke goal-shard knob (M10) — the smoke's N and its ONE range lo..hi (shard_groups relaxes the cover rule there only)
        mt_n = eval_nshards("multiturn", nshards, smoke_goals)
        groups = shard_groups(mt_n, shard_groups_spec, smoke=smoke_goals)
        labels = list(clusters) if clusters else ["*"]
        # the pool is SIZED from the targets placed on each label — one member per three concurrent targets, minimum one (commit 31; measured
        # 2026-10-01 00:10Z: four 27B replicas per cluster served 1–4 targets at ~37 % busy, under one request in flight each);
        # `mt_attacker_gpus` / SISTEER_MT_ATTACKER_GPUS is a CAP that can only lower the count; a caller forcing MORE than the ratio allows
        # is refused by name
        cap_raw = mt_attacker_gpus if mt_attacker_gpus is not None else os.environ.get("SISTEER_MT_ATTACKER_GPUS")
        cap = int(cap_raw) if cap_raw not in (None, "") else None
        forced = mt_attacker_gpus is not None
        seed_base = mt_seed_base or os.environ.get("SISTEER_MT_SEED_BASE", "20260909")
        # THE served model's multi_turn vendor row, spelled by the driver's OWN composer (target_decode.vendor_preset_policy_ref over the
        # export's model-id mapping): `<model_id>|multi_turn|off|vendor_preset@v<N>`. A ref composed here by hand dropped the version and
        # every multi-turn driver refused it against the registry row (`… is not the served model's multi_turn vendor row …@v1`), so no
        # multi-turn target of the u0 wave generated a single goal (2026-09-30 16:01Z; the front door's onboarding rule: the eval's
        # decode-policy ref is RESOLVED from the registry, never typed by a planner)
        from sisteer.eval.attacks.target_decode import vendor_preset_policy_ref
        from sisteer.registry import decode_policy as _decode_policy
        mt_ref = vendor_preset_policy_ref(_decode_policy.load().model_id(model_key), "multi_turn")
        mt_env = {"F22_MT_TARGET_POLICY_REF": mt_ref, "F22_MT_TARGET_SEED_BASE": str(seed_base)}
        # a DOSE view's multi-turn legs are the Appendix D.2 ladder's (DOSE_LADDER_MT_ATTACKS: crescendo + actorattack, never pair); the wave's are
        # the three — the pool below is sized from whichever targets the view plans, so a rung's pool is sized from the rung's legs
        mt_attacks = DOSE_LADDER_MT_ATTACKS if is_dose_view_row(m, view) else MT_ATTACKS
        targets = [(atk, g) for atk in mt_attacks for g in groups]
        # a DECLARED move (`--move-group <shard group>=<label>`, 2026-10-02): the named group's targets are placed on that label instead of
        # their round-robin one, and the label's pool is sized from the MOVED count — a cluster that cannot stand the serving venv up hands
        # its group to one that can, with provenance (the placement manifest's `declared_moves`, the resubmit's 24d receipt)
        moves = _validated_group_moves(move_groups, [_gname(g) for g in groups], labels)
        placed_labels = [moves.get(_gname(g), labels[i % len(labels)]) for i, (atk, g) in enumerate(targets)]
        n_targets_by_label = {lab: sum(1 for pl in placed_labels if pl == lab) for lab in labels}
        for lab in labels:
            if lab == "*":
                continue                                   # placement-free (the freeze): attacker pools register nothing and are not units
            grp = f"mt-{view}-{lab}"
            n_t = n_targets_by_label[lab]
            if n_t == 0:
                continue                                   # every group moved off this label: it serves no target, so no pool is sized for it
            n_atk = attacker_pool_size(n_t, cap=cap)
            if forced and cap is not None and cap > attacker_pool_size(n_t):
                raise ValueError(f"attacker pool of {cap} for {n_t} target(s) on {lab} exceeds one member per three concurrent targets "
                                 f"(allowed {attacker_pool_size(n_t)}) — shared GPU services are sized from measured need; refused")
            for i in range(1, n_atk + 1):
                rows.append(_row("multiturn", "multiturn", f"attacker{i}", model_key, view, [], {}, script="scripts/sdfbase/full22/attacker_pool.sbatch",
                                 name=f"f22-atk-{tag}-{i}", args=[sha], kind="attacker", group=grp, cluster=lab, twin_ok=False, registers=False,
                                 lwid=logical_work_id(model_key, view, "multiturn", f"{grp}/attackers/{i}")))
        for i, (atk, g) in enumerate(targets):
            lab = placed_labels[i]                         # DISJOINT partition across the named clusters, deterministic — or the group's DECLARED move
            gname = "g" + "".join(str(s) for s in g)
            rows.append(_row("multiturn", "multiturn", f"{atk}/{gname}", model_key, view, [atk], {atk: mt_n},
                             script="scripts/sdfbase/full22/target_attacks.sbatch", name=f"f22-att-{tag}-{gname}-{atk}",
                             args=[model_key, view, atk, sha, mt_n, ",".join(str(s) for s in g)]
                                  + (["", str(MT_ACTORATTACK_WINDOW)] if atk == "actorattack" else []),   # positional 7 cells=all, 8 = the window
                             env=mt_env, kind="target",
                             group=None if lab == "*" else f"mt-{view}-{lab}", cluster=lab, twin_ok=False, coverage={atk: jb(atk, wave_private_run)}))
            rows[-1]["expected_shards"] = {atk: [f"shard{s:02d}" for s in g]}      # this unit covers ITS shard group only
            if gname in moves:
                rows[-1]["moved_from"] = labels[i % len(labels)]                    # the round-robin label the declared move took it off
    if "petri" in fams:
        rows.append(_row("petri", "petri", "proxy", model_key, view, list(PETRI_EVALS), none, script=PETRI_PROXY_SCRIPT,
                         name=f"f22-petri-{tag}", args=[model_key, view, sha], time_key=PETRI_TIME_KEY,
                         coverage={e: coverage_spec(e, host="grading_host") for e in PETRI_EVALS}))
        if M.steered_wave_view(view, model_key, table):   # A front-door battery / dose proxy declares its cells too, never the whole enumeration
            # a steered-wave audit drives ONLY the cells the proxy DECLARES — petri_run._frozen_enumeration refuses an s<N>r
            # record whose declaration is empty (a whole-enumeration drive would audit `none` and the controls twin, which is refused) — so the
            # PLAN names them: the view's measured cells (the wave config's explicit_cells, view_cells_of — refuses an unsteered leg by name) land on
            # the proxy's cells positional and in F22_REGISTRY_CELLS by default, never `none`; a --cells selection (apply_cells_selection below)
            # narrows them to a validated subset. A u0 / ordinary view's proxy keeps the whole baseline (no positional: its legs ARE the manifest).
            write_cells_into(rows[-1], view_cells_of(model_key, view, table=table))
    if "agentic" in fams:   # misalign_proxy.sbatch <target=tag> <sha> full: MIS_ROOT=$SHARED/exp/misalign/<tag>, the runner's `agentic/<cell>/responses.jsonl`
        rows.append(_row("agentic", "agentic", "proxy", model_key, view, list(AGENTIC_EVALS), one, script="scripts/sdfbase/full22/misalign/misalign_proxy.sbatch",
                         name=f"f22-misalign-{tag}", args=[tag, sha, "full"],
                         coverage={e: coverage_spec(e, root_rel=f"exp/misalign/{tag}/agentic") for e in AGENTIC_EVALS}))
    if "controls" in fams:   # XSTest + OR-Bench-Hard: the sampled ladder, two seeded trials, decode from configs/sdfbase_sampled_decode.yaml (E5)
        for t in (0, 1):
            run = f"sdfbase_full22_{tag}_sampled_t{t}"
            rows.append(_row("controls", "controls", f"t{t}", model_key, view, list(CONTROL_EVALS), none, script="scripts/sdfbase/full22/sampled/target_sampled.sbatch",
                             name=f"f22s-{model_key}-{tag}-controls_t{t}", args=[model_key, f"controls_t{t}", sha, nshards, "", view],
                             coverage={e: coverage_spec(e, private_run=run) for e in CONTROL_EVALS}))
            rows[-1]["private_run"] = run
    if "mmlu" in fams:
        # mmlu_full.sbatch: the 1,531-item read IS the canonical `mmlu` instrument the registry's planner mints for the front-door models
        # (verified against configs/registry/instruments.yaml, jobplan.canonical_instrument and v_paper_current). A FRONT-DOOR key
        # registers it NATIVELY: F22_REGISTRY_EVALS=mmlu rides the job env
        # like every other unit's, f22_registry_register mints the run under `mmlu_letter_prefill_full_1531` / item set `mmlu_1531`, the
        # sbatch REFUSES any other registered identity and files the terminal run_state itself; the paper's own tags (no
        # F22_REGISTRY_EVALS) keep the sbatch's no-registration guard byte for byte. Coverage: the tinycap store the run's row records
        # (`cells/<key>` under $WAVE_ROOT/mmlu_full/<TAG>_t0), its manifest attesting the 1,531 item ids against the seeds' sha256.
        rows.append(_row("mmlu", "mmlu", "mmlu", model_key, view, ["mmlu"], none, script="scripts/sdfbase/full22/mmlu_full.sbatch",
                         name=f"mmluF-{model_key}-{view}", args=[model_key, view, sha], coverage={"mmlu": coverage_spec("mmlu")}))
    if "gsm8k" in fams:
        # one row per cell of the view (inspect gsm8k_paired200, thinking ON, 8,192 rung) on the FRONT DOOR's matrix `frontdoor_<key>`
        # (g8k_cells.front_door_matrices — rendered by 05 --step configs from the roster's block; gsm8k_thinking.sbatch's second `case`
        # arm admits the name), the cell spelled as the ROSTER's config cell id (models.roster_cell_id: what `g8k_cells.py row --key
        # --view --cell` matches — `constv2ctl2-full__psr_ce_all`, `arditi_allpos`, never the table's short name); the UNIT keeps the
        # table's short name (`default` + the table names matched no row)
        # the target-aware predicate: a STEERED-WAVE view plans no unsteered leg — the table's `none` (the u0 anchor) is dropped from
        # its rows; the generated artefacts are guarded above, the rows built from the table here — the rule holds for every non-u0 view
        g8k_cells = [c for c in (m.get("views") or {}).get(view, {}).get("cells", []) if not (M.steered_wave_view(view, model_key, table) and c in M.UNSTEERED_LEGS)]
        for cell in g8k_cells:
            fitted = 0 if (cell == "none" or cell.startswith("prompted__")) else 1
            rows.append(_row("gsm8k", "gsm8k", cell, model_key, view, ["gsm8k"], none, script="scripts/sdfbase/gsm8k/gsm8k_thinking.sbatch",
                             name=f"g8k-{model_key}-{view}-{cell[:24]}",
                             args=[model_key, view, M.roster_cell_id(m, cell), 8192, sha, fitted, 0, f"frontdoor_{model_key}"], time_key="gsm8k",
                             coverage={"gsm8k": coverage_spec("gsm8k")}))
    if only_clusters is not None:
        unknown = [c for c in only_clusters if clusters is not None and c not in clusters]
        if unknown:
            raise ValueError(f"only_clusters names {', '.join(map(repr, unknown))}, not in the placement universe {clusters} — a unit is submitted "
                             f"where the universe placed it; widen `clusters` (the site's list), do not select outside it")
        rows = [r for r in rows if r["cluster"] == "*" or r["cluster"] in only_clusters]
    if cells is not None:
        apply_cells_selection(rows, cells, model_key, view, table=table)     # cells-fix: served AND registered cells, one validated selection
    return rows


def overflow_rows(model_key: str, view: str, cells: Sequence[str], *, cluster: Optional[str] = None, sha: str = "{sha}", table: Optional[str] = None) -> List[dict]:
    """The Petri OVERFLOW RE-RUN PASS's one row: `petri_proxy.sbatch <key> <view> <sha> overflow-rerun <cells>`
    — the second proxy at the doubled window, named `f22-petri-<tag>-rerun` (the paper's rerun proxies' name), unit `petri/overflow`
    (logical work id `<key>/<view>/petri/overflow`), the petri walltime key, the seven petri evals registered under the
    `…-pass-overflow-rerun` producer key (the sbatch's default segment for the pass), PINNED to `cluster` — the cluster the first pass is
    bound on (lib/petri_overflow.py's first_cluster), where 05 staged the object bank and built the bank view the proxy's standup requires;
    required, never `*` (the stage's list, whose first label the funnel would take), and never twinned. `cells` are the CONFIG cells (the roster spelling the
    sbatch's positional 5 and F22_REGISTRY_CELLS carry) of the legs the first pass left on context-overflow ledgers, plus `none` when a control
    leg is among them (lib/petri_overflow.py derives them from the daemon's record) — validated against the view's explicit_cells and written
    through :func:`write_cells_into`, the ONE writer of a row's cells. This bypasses :func:`apply_cells_selection`'s baseline refusal on
    purpose: an overflow pass serves a SUBSET of the frozen baseline by design (petri_run.py admits it under --overflow-rerun only),
    and the row carries `pass: overflow` + `pass_of` (the first-pass unit) so 06's inventory check and the freeze never
    see it as a wave unit."""
    sel = [str(c) for c in cells]
    if not sel:
        raise ValueError("--pass overflow: no cells — the overflow pass serves the cells of the ledgered legs; nothing planned")
    if len(set(sel)) != len(sel):
        raise ValueError(f"--pass overflow: a cell is named twice ({sel})")
    have = view_cells_of(model_key, view, table=table)
    unknown = [c for c in sel if c not in have]
    if unknown:
        raise ValueError(f"--pass overflow: {unknown} are not in {model_key}/{view}'s explicit_cells {have}")
    if not cluster or str(cluster) == "*":
        raise ValueError("--pass overflow needs --overflow-cluster <label>: the re-run proxy runs on the cluster the first pass is bound on "
                         "(lib/petri_overflow.py's first_cluster — its staged object bank and bank view are there), never on the stage's list")
    tag = model_key if view == "main" else f"{model_key}_{view}"
    none = {e: 0 for e in PETRI_EVALS}
    row = _row("petri", "petri", PETRI_OVERFLOW_UNIT, model_key, view, list(PETRI_EVALS), none, script=PETRI_PROXY_SCRIPT,
               name=f"f22-petri-{tag}-rerun", args=[model_key, view, sha, PETRI_OVERFLOW_PASS_ARG], time_key=PETRI_TIME_KEY,
               cluster=str(cluster), twin_ok=False, coverage={e: coverage_spec(e, host="grading_host") for e in PETRI_EVALS})
    write_cells_into(row, sel)
    row["pass"] = "overflow"
    row["pass_of"] = logical_work_id(model_key, view, "petri", "proxy")
    return [row]


def plan_tsv(rows: List[dict]) -> str:
    """One tab-separated line per unit for the shell (PLAN_COLUMNS order; `-` = empty; args / env shell-quoted)."""
    out = []
    for r in rows:
        vals = {"stage_family": r["stage_family"], "family": r["family"], "unit": r["unit"], "logical_work_id": r["logical_work_id"],
                "cluster": r["cluster"], "kind": r["kind"], "group": r["group"] or "-", "script": r["script"], "name": r["name"],
                "array": r["array"] or "-", "time_key": r["time_key"], "twin_ok": "1" if r["twin_ok"] else "0",
                "args": " ".join(shlex.quote(a) for a in r["args"]) or "-",
                "env": " ".join(f"{k}={shlex.quote(v)}" for k, v in r["env"].items()) or "-",
                "depends": r.get("depends_on") or "-"}
        out.append("\t".join(vals[c] for c in PLAN_COLUMNS))
    return "\n".join(out) + ("\n" if out else "")


# ------------------------------------------------------------------------------------------------- the registry's ids
def _planner_env() -> dict:
    src = os.environ.get("SISTEER_SRC_DIR") or os.path.join(REPO, "src")
    return dict(os.environ, PYTHONPATH=src + os.pathsep + os.environ.get("PYTHONPATH", ""))


def default_seeds_dir() -> str:
    return os.environ.get("SISTEER_REGISTRY_SEEDS") or os.path.join(REPO, "configs", "registry")


_CANONICAL_SCRIPT = r'''
import json, sys
from sisteer.registry import jobplan as JP
seeds, model_id, sp, evals = sys.argv[1], sys.argv[2], sys.argv[3], [e for e in sys.argv[4].split(",") if e]
idx = JP.load_seed_index(seeds)
out = {}
for e in evals:
    try:
        payload, extra = JP.canonical_instrument(idx, e, model_id, sp)
    except LookupError as exc:
        out[e] = {"error": str(exc)}; continue
    refs = payload.get("decode_policy_refs") or extra.get("decode_policy_refs") or []
    if isinstance(refs, str):
        try: refs = json.loads(refs)
        except Exception: refs = [refs]
    ref = next((str(r) for r in refs if str(r).startswith(model_id + "|")), None) or \
          next((str(r).replace("*|", model_id + "|", 1) for r in refs if str(r).startswith("*|")), None)
    iset = idx.item_sets.get(str(payload.get("item_set_id"))) or {}
    n = iset.get("n_items"); sha = iset.get("items_sha256")
    out[e] = {"instrument_id": payload["instrument_id"], "seed_instrument_id": extra.get("seed_instrument_id"), "decode_policy_ref": ref,
              "item_set_id": payload.get("item_set_id"), "n_items": int(n) if isinstance(n, int) or (isinstance(n, str) and n.isdigit()) else None,
              "items_sha256": sha if isinstance(sha, str) and len(sha) == 64 else None}
print(json.dumps(out))
'''

#: the producer's OWN goal recipe, run under the venv: two_si_run._eval_goals(cfg, confirmatory=True, attack=...) — jailbreak_run's
#: recipe (HarmBench standard test + the stratified JBB confirmatory subset) plus the attack-scoped `bon.goal_subset` restriction —
#: the exact call jailbreak_run.py generate / bon_served_run.py / bon_climb_run.py make, so the frozen identities ARE the drivers'.
#: `--goals-jsonl` (grade_static_sampled.py's offline fixture seam: rows {goal_id, dataset, …}) replaces the datasets; the BoN
#: subset rule is then applied from the config's `bon.goal_subset` when the config states one, else `jbb` (every full22 config's).
_GOALS_SCRIPT = r'''
import json, sys
repo, config, attacks, fixture = sys.argv[1], sys.argv[2], [a for a in sys.argv[3].split(",") if a], sys.argv[4] or None
sys.path.insert(0, repo + "/scripts"); sys.argv = ["stage.goal_pool"]
import yaml
cfg = yaml.safe_load(open(config)) if config else {}
out = {}
if fixture:
    rows = [json.loads(l) for l in open(fixture, encoding="utf-8") if l.strip()]
    subset = ((cfg.get("bon") or {}).get("goal_subset")) or "jbb"
    for a in attacks:
        keep = [r for r in rows if a != "bon" or subset is None or r.get("dataset") == subset]
        out[a] = {"goal_ids": [r["goal_id"] for r in keep], "source": f"fixture {fixture}" + (f" (bon: dataset == {subset})" if a == "bon" else "")}
else:
    import two_si_run as tsr
    for a in attacks:
        goals = tsr._eval_goals(cfg, confirmatory=True, attack=(a if a == "bon" else None))
        out[a] = {"goal_ids": [g.goal_id for g in goals], "source": f"two_si_run._eval_goals({config}, confirmatory=True, attack={a if a == 'bon' else None!r})"}
print(json.dumps(out))
'''

_AGENTIC_SCRIPT = r'''
import json, sys
from sisteer.capeval.misalign.runner import agentic_items
behaviors = [b for b in sys.argv[1].split(",") if b]
print(json.dumps({b: sorted(i["item_id"] for i in agentic_items([b])) for b in behaviors}))
'''


def canonical_instruments(model_id: str, eval_ids: List[str], seeds_dir: Optional[str] = None, serving_path: str = SERVING_PATH) -> Dict[str, dict]:
    """{eval_id: {instrument_id (64-hex), seed_instrument_id, decode_policy_ref} | {error}} minted by the registry's OWN planner
    (sisteer.registry.jobplan.canonical_instrument over load_seed_index) — the id a job registers is a sha256 over the canonical
    instrument fields (identity.instrument_id), so a hand-written seed NAME could never match a run row. Raises RuntimeError
    when the planner cannot run at all."""
    seeds = seeds_dir or default_seeds_dir()
    try:
        p = subprocess.run([sys.executable, "-c", _CANONICAL_SCRIPT, seeds, model_id, serving_path, ",".join(eval_ids)],
                           capture_output=True, text=True, env=_planner_env(), timeout=600)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"the registry planner timed out minting the canonical instrument ids: {exc}")
    if p.returncode != 0:
        raise RuntimeError(f"the registry planner (sisteer.registry.jobplan over {seeds}) failed: {p.stderr.strip()[-600:]}")
    return json.loads(p.stdout)


def goal_pool(config_path: Optional[str], attacks: List[str], goals_jsonl: Optional[str] = None) -> Dict[str, dict]:
    """{attack: {goal_ids (ORDERED, the drivers' order), source}} — the producer's recipe (two_si_run._eval_goals) under the venv,
    or the `--goals-jsonl` fixture. `attack` is `bon` (the goal_subset restriction) or a static / multi-turn family (the full
    pool). Raises RuntimeError when the pool cannot be loaded — the freeze then REFUSES (strict): a frozen null identity set
    would silently weaken the gate."""
    fixture = goals_jsonl or os.environ.get("SISTEER_GOALS_JSONL") or ""
    if not fixture and not (config_path and os.path.isfile(config_path)):
        raise RuntimeError(f"no wave config at {config_path} (05_stage_roster.sh --step configs generates it) and no --goals-jsonl fixture — the goal identities cannot be frozen")
    try:
        p = subprocess.run([sys.executable, "-c", _GOALS_SCRIPT, REPO, config_path if (config_path and os.path.isfile(config_path)) else "", ",".join(attacks), fixture],
                           capture_output=True, text=True, env=_planner_env(), timeout=900)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"the goal recipe timed out: {exc}")
    if p.returncode != 0:
        raise RuntimeError(f"the goal recipe (two_si_run._eval_goals over {config_path or fixture}) failed: {p.stderr.strip()[-600:]}")
    out = json.loads(p.stdout)
    for a, blk in out.items():
        if not blk.get("goal_ids"):
            raise RuntimeError(f"the goal recipe yielded NO goals for {a!r} ({blk.get('source')})")
    return out


def agentic_item_ids(behaviors: List[str]) -> Dict[str, List[str]]:
    """{behavior: [scenario ids]} from sisteer.capeval.misalign.runner.agentic_items — the runner's own enumeration (item_id ==
    scenario id); RuntimeError when it cannot run (the freeze refuses)."""
    try:
        p = subprocess.run([sys.executable, "-c", _AGENTIC_SCRIPT, ",".join(behaviors)], capture_output=True, text=True, env=_planner_env(), timeout=600)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"misalign.runner.agentic_items timed out: {exc}")
    if p.returncode != 0:
        raise RuntimeError(f"misalign.runner.agentic_items failed: {p.stderr.strip()[-600:]}")
    out = json.loads(p.stdout)
    for b, ids in out.items():
        if not ids:
            raise RuntimeError(f"misalign.runner.agentic_items yielded NO scenarios for behavior {b!r}")
    return out


def shard_goal_ids(goal_ids: List[str], shard_names: List[str], nshards: int) -> Dict[str, List[str]]:
    """The drivers' slicing, byte for byte: `goals[s::nshards]` (jailbreak_run.py generate; two_si_run.py) — shard `shardNN` is
    index NN of `nshards`."""
    out = {}
    for s in shard_names:
        i = int(s[len("shard"):])
        if i >= nshards:
            raise ValueError(f"shard {s} is not one of 0..{nshards - 1}")
        out[s] = list(goal_ids[i::nshards])
    return out


def padded_instrument_id(model_key: str, model_id: str, eval_id: str, pad: int, filler: Optional[str], window: int = PADDING_WINDOW,
                         serving_path: str = SERVING_PATH, seeds_dir: Optional[str] = None) -> Tuple[Optional[str], str]:
    """The per-level padding instrument id is a HASH minted by the registry (migration 0012 identity), never a literal: derived
    through `python -m sisteer.registry.jobplan padding-block --config <level config> --filler F --window W` then
    `padded-instrument-id --seeds configs/registry --model-id … --eval … --padding-json …` — the same inputs the job uses.
    Returns (id, source); (None, reason) when the level's config or the filler is not at hand (the gate then compares the
    registry row's id only once 06 --gate binds it)."""
    cfg = os.path.join(REPO, "configs", "sdfbase", "static_padding", f"sdfbase_full22_{model_key}.padstatic_p{pad}.yaml")
    if not os.path.isfile(cfg):
        return None, f"derive at gate: {os.path.relpath(cfg, REPO)} not generated yet (jobplan padded-instrument-id)"
    if not filler or not os.path.isfile(filler):
        return None, "derive at gate: no PG-19 filler at hand on this host (pass --filler; jobplan padded-instrument-id)"
    env = _planner_env()
    py = sys.executable
    try:
        blk = subprocess.run([py, "-m", "sisteer.registry.jobplan", "padding-block", "--config", cfg, "--filler", filler, "--window", str(window)],
                             capture_output=True, text=True, env=env, check=True, timeout=300).stdout.strip()
        out = subprocess.run([py, "-m", "sisteer.registry.jobplan", "padded-instrument-id", "--seeds", seeds_dir or default_seeds_dir(),
                              "--model-id", model_id, "--eval", eval_id, "--serving-path", serving_path, "--padding-json", blk],
                             capture_output=True, text=True, env=env, check=True, timeout=300).stdout
        return json.loads(out)["instrument_id"], "jobplan padded-instrument-id"
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, json.JSONDecodeError, KeyError) as exc:
        return None, f"derive at gate: jobplan padded-instrument-id failed here ({str(exc)[:160]})"


def petri_lock(path: Optional[str] = None) -> dict:
    """configs/capeval/petri-scenarios.lock.yaml (hashes only, committed): the bundle's content hash + location, the behaviors' hashes,
    num_scenarios_per_behavior. RuntimeError when absent or malformed (the freeze refuses)."""
    p = path or os.path.join(REPO, PETRI_LOCK_REL)
    if not os.path.isfile(p):
        raise RuntimeError(f"no Petri scenario lock at {p} — the Petri scenario identities cannot be frozen")
    import yaml  # noqa: WPS433
    doc = yaml.safe_load(open(p, encoding="utf-8")) or {}
    bundle = doc.get("bundle") if isinstance(doc, dict) else None
    if not isinstance(bundle, dict) or not isinstance(bundle.get("content_hash"), str) or doc.get("status") != "frozen":
        raise RuntimeError(f"{p}: not a frozen lock with a bundle.content_hash (status {doc.get('status') if isinstance(doc, dict) else None!r})")
    return doc


_PETRI_REPO_VERIFY_SCRIPT = r'''
import sys
from sisteer.capeval.petri import scenarios as S
try:
    b = S.verify_bundle_content(sys.argv[1], expected_hash=sys.argv[2])
except S.ScenarioError as exc:
    print(exc, file=sys.stderr); sys.exit(2)
print(sum(len(m) for m in b.full_file_maps.values()))
'''


def petri_repo_bundle_dir(h: str) -> str:
    """`<this checkout>/data/petri/scenarios/<h>`: the copy of the frozen bundle the repository publishes (it may be absent)."""
    return os.path.join(REPO, PETRI_REPO_BUNDLE_REL, h)


def verify_petri_repo_bundle(d: str, h: str) -> None:
    """The published copy, checked by the loader's own rule (sisteer.capeval.petri.scenarios.verify_bundle_content: every file against
    the bundle manifest, the content hash recomputed and equal to the lock's; a checkout keeps no modes, so those are not checked). Run
    under this interpreter with the checkout's src first (_planner_env). RuntimeError, by name, when it does not verify."""
    try:
        p = subprocess.run([sys.executable, "-c", _PETRI_REPO_VERIFY_SCRIPT, d, h], capture_output=True, text=True, env=_planner_env(), timeout=300)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"the published Petri scenario bundle {d} could not be verified: {exc}")
    if p.returncode != 0:
        raise RuntimeError(f"the published Petri scenario bundle {d} does not verify as bundle {h}: {(p.stderr or p.stdout).strip()[-600:]} "
                           "— refusing to enumerate its scenarios (restore the copy from the repository)")


def petri_bundle_dir(lock: dict, explicit: Optional[str] = None) -> str:
    """The frozen bundle's directory on THIS host: `explicit` / $SISTEER_PETRI_BUNDLE_DIR, else the grading host's store
    `~/.local/state/petri/scenarios/<content hash>` (petri_drive_full22.sh's PETRI_BUNDLE_DIR), else the lock's recorded location when it
    is mounted here, else the copy this checkout publishes at `data/petri/scenarios/<content hash>` (the bundle ships with the
    repository for exact reproduction). A store or a named dir is checked by its `bundle_manifest.json` carrying the lock's content
    hash; the published copy is verified file by file (verify_petri_repo_bundle). A named dir that is absent, or a copy that is there
    and wrong, is a RuntimeError by name; no copy anywhere the default lookup looks is PetriBundleAbsent."""
    h = lock["bundle"]["content_hash"]
    named = explicit or os.environ.get(PETRI_BUNDLE_DIR_ENV)
    if named and not os.path.isdir(named):
        # a bundle dir NAMED by the caller / the environment must exist: never fall through to another materialization silently
        raise RuntimeError(f"the Petri scenario bundle dir {named} ({'--petri-bundle' if explicit else PETRI_BUNDLE_DIR_ENV}) is not a directory — "
                           "the Petri scenario identities cannot be frozen from it")
    loc, shown = lock["bundle"].get("location"), []
    if loc and not named and "$" in str(loc):
        # the lock records the bundle's location in placeholder form ($SHARED/...): expanded through src/sisteer/site.py; a
        # placeholder with no value here leaves that candidate out and names it in the refusal, never probed as a literal path
        import clusters as _C  # noqa: E402 — the site parser, loaded by path
        S = _C.site_module()
        try:
            loc = os.path.expanduser(S.expand(str(loc), key="petri-scenarios.lock bundle.location"))
        except S.PlaceholderError as exc:
            shown, loc = [f"{loc} (unresolved: {exc})"], None
    published = None if named else petri_repo_bundle_dir(h)
    cands = [c for c in ((named,) if named else (os.path.join(os.path.expanduser("~"), ".local", "state", "petri", "scenarios", h),
                                                 loc)) if c]
    for c in cands:
        if os.path.isdir(c):
            mp = os.path.join(c, "bundle_manifest.json")
            if not os.path.isfile(mp):
                raise RuntimeError(f"Petri bundle dir {c} has no bundle_manifest.json — not a frozen bundle (scenarios.freeze_scenarios writes it)")
            got = (json.load(open(mp, encoding="utf-8")) or {}).get("content_hash")
            if got != h:
                raise RuntimeError(f"Petri bundle dir {c} is bundle {got!r}, the lock names {h!r} — another freeze; refusing to enumerate its scenarios")
            return c
    if published and os.path.isdir(published):
        verify_petri_repo_bundle(published, h)
        return published
    tried = cands + ([published] if published else []) + shown
    raise PetriBundleAbsent(f"the frozen Petri scenario bundle {h} is not materialized here (tried: {', '.join(tried)}) — set {PETRI_BUNDLE_DIR_ENV} to its "
                            "directory (bundle_manifest.json + <behavior>/scenarios/seeds/*); the Petri scenario identities cannot be frozen without it")


def petri_scenario_ids(bundle_dir: Optional[str] = None, lock_path: Optional[str] = None) -> Tuple[Dict[str, List[str]], str]:
    """({behavior: [scenario ids]}, source) — scenarios.scenario_ids' rule (the sorted stems of `<behavior>/scenarios/seeds/*`; Bloom names the
    seed file after the scenario) over the frozen bundle; every behavior of PETRI_BEHAVIORS must hold exactly the lock's
    num_scenarios_per_behavior (RuntimeError otherwise — a short set would silently weaken the gate). Names only: never a seed's text."""
    lock = petri_lock(lock_path)
    d = petri_bundle_dir(lock, bundle_dir)
    want_n = lock.get("num_scenarios_per_behavior")
    out: Dict[str, List[str]] = {}
    for b in PETRI_BEHAVIORS:
        seeds = os.path.join(d, b, "scenarios", "seeds")
        if not os.path.isdir(seeds):
            raise RuntimeError(f"Petri bundle {d}: behavior {b!r} has no scenarios/seeds/ dir")
        stems = sorted({os.path.splitext(f)[0] for f in os.listdir(seeds) if not f.startswith(".") and os.path.isfile(os.path.join(seeds, f))})
        if not stems:
            raise RuntimeError(f"Petri bundle {d}: behavior {b!r} has an empty scenarios/seeds/")
        if isinstance(want_n, int) and len(stems) != want_n:
            raise RuntimeError(f"Petri bundle {d}: behavior {b!r} holds {len(stems)} scenario(s), the lock says num_scenarios_per_behavior {want_n}")
        out[b] = stems
    return out, f"Petri scenario bundle {lock['bundle']['content_hash']} at {d} (<behavior>/scenarios/seeds stems; {PETRI_LOCK_REL})"


def _wave_config(model_key: str, view: Optional[str] = None, path: Optional[str] = None, table: Optional[str] = None) -> Tuple[Optional[str], Optional[dict]]:
    """(path, parsed) of the (model, view) wave config — `configs/sdfbase_full22_<tag>.yaml` (05 --step configs renders it), or an
    explicit `path` (the tests' seam); (path, None) when absent. `view` None = the model's baseline wave (M.baseline_view)."""
    if view is None:
        view = M.baseline_view(model_key, table)
    tag = model_key if view == "main" else f"{model_key}_{view}"
    p = path or os.path.join(REPO, "configs", f"sdfbase_full22_{tag}.yaml")
    if not os.path.isfile(p):
        return p, None
    try:
        import yaml  # noqa: WPS433
        return p, (yaml.safe_load(open(p, encoding="utf-8")) or {})
    except Exception as exc:  # noqa: BLE001 — a config that does not parse is reported by the caller
        return p, {"_error": str(exc)}


def attach_expected(units: List[dict], model_key: str, *, nshards: int, goals_jsonl: Optional[str] = None, strict: bool = True,
                    wave_config: Optional[str] = None, petri_bundle: Optional[str] = None, table: Optional[str] = None,
                    smoke_goals: Optional[Mapping[str, object]] = None) -> None:
    """Fill every eval's `coverage.expected` (the frozen WORK IDENTITIES) in place: the goal ids per shard from the producer's
    recipe for the jailbreak layouts and the BoN 512 store (+ `n_max` from the wave config's `bon.n_max`), the scenario ids per
    behavior for the agentic runner, the item set's declared size / ids sha256 for the tinycap stores (the planner's item set); the
    Petri `<behavior>/<scenario>` ids from the frozen scenario bundle (petri_scenario_ids); the climb's expected set is run-time (the
    manifest) and says so. `strict` refuses (RuntimeError)
    when an identity set cannot be minted. The jailbreak layouts slice goals[s::N] with the SAME N the plan composed the unit's positionals
    with (`eval_nshards`: the smoke knob's N for the multi-turn targets and the padding levels when set — the frozen expectation and the
    submission stay one plan)."""
    cfg_path, cfg = _wave_config(model_key, path=wave_config, table=table)
    if cfg and cfg.get("_error"):
        raise RuntimeError(f"{cfg_path} does not parse: {cfg['_error']}")
    need_pool = sorted({"bon" if ev["coverage"]["layout"] == "bon_store512" else e for u in units for e, ev in u["evals"].items()
                        if ev["coverage"]["layout"] in ("jailbreak_cellstore", "bon_store512")})
    pools: Dict[str, dict] = {}
    if need_pool:
        try:
            pools = goal_pool(cfg_path, need_pool, goals_jsonl)
        except RuntimeError:
            if strict:
                raise
    behaviors = sorted({ev["coverage"]["behavior"] for u in units for ev in u["evals"].values() if ev["coverage"]["layout"] == "misalign_responses"})
    agentic: Dict[str, List[str]] = {}
    if behaviors:
        try:
            agentic = agentic_item_ids(behaviors)
        except RuntimeError:
            if strict:
                raise
    n_max = ((cfg or {}).get("bon") or {}).get("n_max")
    petri_ids: Dict[str, List[str]] = {}; petri_src = ""
    if any(ev["coverage"]["layout"] == "petri_cellstore" for u in units for ev in u["evals"].values()):
        try:
            petri_ids, petri_src = petri_scenario_ids(petri_bundle)
        except RuntimeError:
            if strict:
                raise
    for u in units:
        for e, ev in u["evals"].items():
            cov = ev["coverage"]; lay = cov["layout"]
            if lay == "jailbreak_cellstore":
                pool = pools.get(e)
                if pool:
                    n = eval_nshards(u["family"], nshards, smoke_goals)
                    cov["expected"] = shard_goal_ids(pool["goal_ids"], ev["expected_shards"], n)
                    cov["expected_source"] = f"{pool['source']}; goals[s::{n}] (jailbreak_run.py generate)"
                else:
                    cov["expected_source"] = "unresolved: the goal pool could not be loaded at freeze time"
            elif lay == "bon_store512":
                pool = pools.get("bon")
                if pool:
                    cov["expected"] = {"shard00": list(pool["goal_ids"])}
                    cov["expected_source"] = f"{pool['source']} (bon.goal_subset); n_max draws per goal"
                else:
                    cov["expected_source"] = "unresolved: the BoN goal pool could not be loaded at freeze time"
                if isinstance(n_max, int) and n_max > 0:
                    cov["n_max"] = n_max; cov["n_max_source"] = f"{cfg_path} bon.n_max"
                elif strict:
                    raise RuntimeError(f"{u['logical_work_id']}: no `bon.n_max` in {cfg_path} — the BoN 512 store's draws-per-goal cannot be frozen")
            elif lay == "misalign_responses":
                ids = agentic.get(cov["behavior"])
                if ids:
                    cov["expected"] = {"shard00": list(ids)}; cov["expected_source"] = "sisteer.capeval.misalign.runner.agentic_items"
                else:
                    cov["expected_source"] = "unresolved: misalign.runner.agentic_items could not run at freeze time"
            elif lay == "petri_cellstore":
                behs = [cov["behavior"]] if cov.get("behavior") else list(PETRI_BEHAVIORS)
                if petri_ids:
                    cov["expected"] = {"shard00": [f"{b}/{sid}" for b in behs for sid in petri_ids[b]]}
                    cov["expected_source"] = petri_src
                else:
                    cov["expected_source"] = "unresolved: the frozen Petri scenario bundle could not be enumerated at freeze time"
            elif lay == "capability_cellstore":
                cov["expected_source"] = f"registry seeds item set {cov.get('item_set_id')} (n_items{'; items_sha256' if cov.get('items_sha256') else ''})"
                if not cov.get("n_items") and strict:
                    raise RuntimeError(f"{u['logical_work_id']}: the registry seeds declare no n_items for eval {e!r}'s item set {cov.get('item_set_id')!r} — the item coverage cannot be frozen")


# ------------------------------------------------------------------------------ the model-scoped HOLD-OUT of evals
#: `OVERRIDE_HOLD_OUT_EVALS=codeattack,agentic_murder` + `OVERRIDE_HOLD_OUT_SOURCE='ruling, 2026-09-30 23:23Z'` in the MODEL's
#: overrides file (reproduce/overrides/<model>_*.env — already model-scoped, and stamped on every row the model produces) hold those evals
#: OUT of the model's battery: the inventory's units keep their evals, their workload and their contract (nothing is deleted, the jobs run
#: what they run), the units that carry a held-out eval are stamped `held_out: {eval: {source, override_id, overrides_path,
#: overrides_sha256}}`, and admission stops REQUIRING those evals, saying `excluded_by_ruling (<source>)`. A hold-out without a source is
#: refused by name: the ruling (who, when) is the provenance, not an option.
HOLD_OUT_EVALS_KEY = "OVERRIDE_HOLD_OUT_EVALS"
HOLD_OUT_SOURCE_KEY = "OVERRIDE_HOLD_OUT_SOURCE"
#: 30-fix: a hold-out is ONE model's ruling, and "the model's file" was the only scoping — a runner retaining
#: SISTEER_RUN_OVERRIDES, or a file passed for another model, stamped the Qwen-only ruling onto that model and admission then skipped its
#: codeattack / agentic_murder runs. The document names its model key; the freeze, the live apply and the stage shell refuse a mismatch by name.
HOLD_OUT_MODEL_KEY = "OVERRIDE_MODEL"
RUN_OVERRIDES_ENV = "SISTEER_RUN_OVERRIDES"
#: every eval id the plan can place (the plan's own vocabulary; a hold-out of anything else is a typo, refused by name)
ALL_EVAL_IDS = tuple(sorted(set(COVERAGE_LAYOUT) | set(CONTROL_EVALS)))


def parse_env_file(path: str) -> Dict[str, str]:
    """`KEY=value` lines of a shell `.env` (comments and blanks skipped; a leading `export`, an inline ` # comment` after an unquoted
    value, and surrounding quotes stripped) — the same reading the job side gives the overrides file (sisteer.registry.emit)."""
    out: Dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            if line.startswith("export "):
                line = line[len("export "):].strip()
            k, _, v = line.partition("=")
            k = k.strip(); v = v.strip()
            if len(v) >= 2 and v[0] in "'\"" and v[0] in v[1:]:
                v = v[1:v.index(v[0], 1)]
            elif " #" in v:
                v = v.split(" #", 1)[0].rstrip()
            if k:
                out[k] = v
    return out


def hold_out_evals(overrides: Optional[str], model_key: Optional[str] = None) -> Dict[str, dict]:
    """{eval_id: {source, override_id, overrides_path, overrides_sha256, model}} — the evals the model's overrides file holds OUT of its battery
    (`OVERRIDE_HOLD_OUT_EVALS`, comma-separated; `OVERRIDE_HOLD_OUT_SOURCE` names the ruling; `OVERRIDE_MODEL` names the ONE model it is for).
    {} for no file or no key (the paper's full battery). Refuses by name: a missing file, a hold-out without a source, a hold-out without
    its model, a model other than `model_key` when the caller names one, an eval id the plan does not know (30-fix: the scoping is the
    document's own, never "the file we happened to be handed")."""
    p = (overrides or "").strip()
    if not p:
        return {}
    p = os.path.expanduser(p)
    if not os.path.isfile(p):
        raise ValueError(f"overrides file {p} does not exist — its hold-outs cannot be read")
    kv = parse_env_file(p)
    raw = kv.get(HOLD_OUT_EVALS_KEY, "").strip()
    if not raw:
        return {}
    evals = [e.strip() for e in raw.split(",") if e.strip()]
    source = kv.get(HOLD_OUT_SOURCE_KEY, "").strip()
    if not source:
        raise ValueError(f"{p}: {HOLD_OUT_EVALS_KEY}={raw!r} holds evals out but {HOLD_OUT_SOURCE_KEY} is empty — a hold-out carries the ruling it came from "
                         f"(who, when), or it is not applied")
    unknown = [e for e in evals if e not in ALL_EVAL_IDS]
    if unknown:
        raise ValueError(f"{p}: {HOLD_OUT_EVALS_KEY} names eval(s) the plan does not know: {unknown} (known: {list(ALL_EVAL_IDS)})")
    model = kv.get(HOLD_OUT_MODEL_KEY, "").strip()
    if not model:
        raise ValueError(f"{p}: {HOLD_OUT_EVALS_KEY}={raw!r} holds evals out but {HOLD_OUT_MODEL_KEY} is empty — a hold-out is ONE model's ruling; the document names its "
                         f"model key, or it is not applied (30-fix)")
    if model_key and model != model_key:
        raise ValueError(f"{p}: {HOLD_OUT_MODEL_KEY}={model!r} but this inventory / stage runs model {model_key!r} — a model's ruling never travels to another model; "
                         f"nothing stamped")
    with open(p, "rb") as fh:
        digest = hashlib.sha256(fh.read()).hexdigest()
    ident = kv.get("OVERRIDE_ID") or os.path.splitext(os.path.basename(p))[0]
    return {e: {"source": source, "override_id": ident, "overrides_path": os.path.basename(p), "overrides_sha256": digest, "model": model} for e in evals}


# ------------------------------------------------------------------------------ the SMOKE goal-shard knob (demo-import M10; Q9 ruled)
#: `OVERRIDE_SMOKE_GOALS=<lo>-<hi>/N` in the MODEL's overrides document (reproduce/overrides/demo_smoke.env) is the ONE knob a smoke
#: turns, and the production planner itself emits it: the multi-turn targets run shards lo..hi of N (target_attacks.sbatch `[nshards]
#: [shards]` — the driver slices goals[s::N], so `0-0/99` is goals [0::99], 3 of 259 per attack; the producer key bakes
#: `sh<shards>of<nshards>`, so a smoke and a confirmatory run are different data by construction) and the per-view padding levels run
#: the same range (target_static_padded_obj.sbatch `shards=<lo>-<hi>/N`); static, controls, mmlu, gsm8k, the BoN store and climb,
#: agentic and Petri run whole. It is a PLAN KNOB: `freeze_plan_knobs` pins it in the inventory header, `plan --inventory` refuses a
#: differing live value and `refreeze_refusal` refuses a silent change once attempts exist. Its VALUE is read from the DOCUMENT
#: (`parse_env_file` + the 30-fix OVERRIDE_MODEL scoping), never the bare environment, so the registry's run_overrides stamp (the job
#: env's SISTEER_RUN_OVERRIDES) marks every row the smoke produces with the document's id. No document, or no key -> the plan's rows
#: are byte-identical to the paper's.
SMOKE_GOALS_KEY = "OVERRIDE_SMOKE_GOALS"
#: the document's OVERRIDE_ID must say it is a smoke: a goal subset under a confirmatory id would file partial data as a run of record
SMOKE_OVERRIDE_ID_MARK = "smoke"
#: N <= 99: target_static_padded_obj.sbatch names the store's shards shard00..shard99 (two digits; the grader sorts by name) — the
#: planner refuses here what the job would refuse in its first minute, so `0-0/259` is refused and the one shared value is `0-0/99`
SMOKE_MAX_NSHARDS = 99
#: the per-view padding rows' shard TOTAL when no smoke knob is set (the filed rungs'; the launcher's `shards=3` positional)
PADDING_NSHARDS = 3
SMOKE_GOALS_RE = re.compile(r"^(?P<lo>[0-9]+)-(?P<hi>[0-9]+)/(?P<n>[0-9]+)$")


def parse_smoke_goals(raw: str, where: str = SMOKE_GOALS_KEY) -> Tuple[int, int, int]:
    """(lo, hi, N) of `<lo>-<hi>/N` — the padding launcher's own grammar (target_static_padded_obj.sbatch, `shards: begin`), mirrored at
    PLAN time: a malformed spec, N < 1, N > SMOKE_MAX_NSHARDS, lo > hi or hi >= N refuses by name here, never in the job's first minute.
    The bare `N` form (every shard) is not a smoke and is refused too."""
    spec = str(raw or "").strip()
    m = SMOKE_GOALS_RE.match(spec)
    if not m:
        raise ValueError(f"{where}={spec!r} is not `<lo>-<hi>/N` (shards lo..hi, inclusive and 0-based, of N; e.g. 0-0/99) — the smoke goal-shard "
                         f"knob is a shard RANGE of a shard total, nothing else")
    lo, hi, n = int(m.group("lo")), int(m.group("hi")), int(m.group("n"))
    if n < 1:
        raise ValueError(f"{where}={spec!r}: N must be >= 1")
    if n > SMOKE_MAX_NSHARDS:
        raise ValueError(f"{where}={spec!r}: N={n} > {SMOKE_MAX_NSHARDS} — the padding store names shards shard00..shard{SMOKE_MAX_NSHARDS} "
                         f"(target_static_padded_obj.sbatch), so one value shared by multi-turn and padding is at most /{SMOKE_MAX_NSHARDS}")
    if lo > hi:
        raise ValueError(f"{where}={spec!r}: lo {lo} > hi {hi} (inverted range)")
    if hi >= n:
        raise ValueError(f"{where}={spec!r}: hi {hi} is out of range for N={n} (the shards are 0..{n - 1}) — a shard index >= N would slice an "
                         f"empty goal list and write an empty shard, silently")
    return lo, hi, n


def smoke_goals(overrides: Optional[str], model_key: Optional[str] = None) -> Optional[dict]:
    """The smoke goal-shard knob the model's overrides DOCUMENT sets, or None (no document, or no `OVERRIDE_SMOKE_GOALS` key: the paper's
    rows). {spec (normalised `<lo>-<hi>/N`), lo, hi, nshards, shards (`shardNN` names lo..hi), override_id, overrides_path (basename),
    overrides_sha256, model}. Refuses by name: a missing file; a document whose OVERRIDE_ID does not contain `smoke`; an absent
    OVERRIDE_MODEL, or one naming another model than `model_key` when the caller names one (the 30-fix scoping: the document's own,
    never "the file we happened to be handed"); and every malformation `parse_smoke_goals` names."""
    p = (overrides or "").strip()
    if not p:
        return None
    p = os.path.expanduser(p)
    if not os.path.isfile(p):
        raise ValueError(f"overrides file {p} does not exist — its smoke goal-shard knob cannot be read")
    kv = parse_env_file(p)
    raw = kv.get(SMOKE_GOALS_KEY, "").strip()
    if not raw:
        return None
    ident = kv.get("OVERRIDE_ID", "").strip()
    if SMOKE_OVERRIDE_ID_MARK not in ident.lower():
        raise ValueError(f"{p}: {SMOKE_GOALS_KEY}={raw!r} narrows the goal pool but OVERRIDE_ID={ident or '(unset)'!r} names no smoke (the id must contain "
                         f"`{SMOKE_OVERRIDE_ID_MARK}`) — a goal subset is a smoke's, never a run of record's; nothing planned")
    model = kv.get(HOLD_OUT_MODEL_KEY, "").strip()
    if not model:
        raise ValueError(f"{p}: {SMOKE_GOALS_KEY}={raw!r} but {HOLD_OUT_MODEL_KEY} is empty — a smoke is ONE model's recipe; the document names its model key, "
                         f"or the knob is not applied (30-fix)")
    if model_key and model != model_key:
        raise ValueError(f"{p}: {HOLD_OUT_MODEL_KEY}={model!r} but this plan / inventory / stage runs model {model_key!r} — a model's smoke recipe never travels "
                         f"to another model; nothing planned (30-fix)")
    lo, hi, n = parse_smoke_goals(raw, f"{p}: {SMOKE_GOALS_KEY}")
    with open(p, "rb") as fh:
        digest = hashlib.sha256(fh.read()).hexdigest()
    return {"spec": f"{lo}-{hi}/{n}", "lo": lo, "hi": hi, "nshards": n, "shards": [f"shard{s:02d}" for s in range(lo, hi + 1)],
            "override_id": ident, "overrides_path": os.path.basename(p), "overrides_sha256": digest, "model": model}


def eval_nshards(family: str, nshards: int, smoke: Optional[Mapping[str, object]] = None) -> int:
    """The shard TOTAL an eval's goals are sliced over (`goals[s::N]`, jailbreak_run.py generate) for a plan row of `family`: the
    multi-turn targets' is the plan's `nshards`, or the smoke's N; the per-view padding rows' is PADDING_NSHARDS (the filed rungs'), or
    the smoke's N; every other family's (static) is the plan's `nshards` — static never takes the knob (grade_static_sampled.py requires
    shard00..shard<N-1>). The ONE rule `unit_plan` composes the positionals with and `attach_expected` freezes the identities with."""
    if family == "padding":
        return int(smoke["nshards"]) if smoke else PADDING_NSHARDS
    if family == "multiturn":
        return int(smoke["nshards"]) if smoke else nshards
    return nshards


def hold_out_event(model: Optional[str], stamped: Mapping[str, List[str]], src: Mapping[str, object], at: str) -> dict:
    """The append-only RECEIPT of a hold-out stamp (`hold_out` in the attempts ledger): the model, {eval: [stamped lwids]}, the ruling and the
    overrides document's identity (id + sha256), and where it was applied (`freeze` | `hold-out`). Admission requires a matching receipt for
    every stamp it reads (30-fix MEDIUM: a stamp without its receipt admits nothing)."""
    return {"event": "hold_out", "model": model, "evals": {e: sorted(lws) for e, lws in stamped.items()}, "source": src["source"], "override_id": src["override_id"],
            "overrides_path": src["overrides_path"], "overrides_sha256": src["overrides_sha256"], "at": at, "utc": utc_now()}


def hold_out_receipts(ledger: str) -> List[dict]:
    """Every `hold_out` receipt in the attempts ledger beside `ledger` (the submissions ledger or the attempts file itself)."""
    p = ledger if os.path.basename(ledger) == "attempts.jsonl" else os.path.join(os.path.dirname(os.path.abspath(ledger)), "attempts.jsonl")
    out: List[dict] = []
    if os.path.isfile(p):
        for ln in open(p, encoding="utf-8"):
            try:
                r = json.loads(ln)
            except json.JSONDecodeError:
                continue
            if r.get("event") == "hold_out":
                out.append(r)
    return out


def hold_out_receipt_matches(rec: Mapping[str, object], stamp: Mapping[str, object], eval_id: str, lwid: str) -> bool:
    """A receipt covers a unit's stamp when it names the same overrides document (id + sha256) and lists the unit under that eval."""
    return str(rec.get("override_id")) == str(stamp.get("override_id")) and str(rec.get("overrides_sha256")) == str(stamp.get("overrides_sha256")) \
        and lwid in ((rec.get("evals") or {}).get(eval_id) or [])


def apply_hold_outs(units: List[dict], held: Mapping[str, dict]) -> Dict[str, List[str]]:
    """Stamp `held_out` ({eval: provenance}) on every unit that carries a held-out eval — its `evals`, workload and contract untouched
    (nothing deleted); an existing stamp of another eval is kept (a hold-out is never lifted here). {eval: [stamped lwids]}."""
    out: Dict[str, List[str]] = {}
    for u in units:
        mine = {e: dict(held[e]) for e in (u.get("evals") or {}) if e in held}
        if mine:
            u["held_out"] = {**(u.get("held_out") or {}), **mine}
            for e in mine:
                out.setdefault(e, []).append(u["logical_work_id"])
    return out


def expected_units(model_key: str, code_sha: str, table: Optional[str] = None, families: Optional[List[str]] = None,
                   filler: Optional[str] = None, nshards: int = 9, seeds_dir: Optional[str] = None, strict: bool = True,
                   goals_jsonl: Optional[str] = None, wave_config: Optional[str] = None, petri_bundle: Optional[str] = None,
                   labels: Optional[List[str]] = None, overrides: Optional[str] = None,
                   reruns: Optional[Sequence[Mapping[str, object]]] = None) -> List[dict]:
    """The frozen LOGICAL units = plan(<baseline view>)'s registering units (M.baseline_view: u0 when the table declares it, else the
    single-phase recipe's one wave), each eval bound to the ids the registry's planner mints and to the WORK IDENTITIES its coverage is
    judged on (attach_expected). `strict` refuses (RuntimeError) when the planner cannot mint an unpadded id or an identity set cannot
    be loaded — a frozen null would silently weaken the gate. `reruns` (the `rerun-units` stamp's declarations; never the freeze's) adds
    the rerun pair's units, each carrying `rerun_of`, `rerun_source` and `cells` beside the fields every unit has."""
    m = M.model(model_key, table)
    model_id = m["serve"]["checkpoint"]["model_id"]
    thinking = (m.get("thinking") or {})
    base_view = M.baseline_view(model_key, table)
    # M10: the smoke goal-shard knob the model's overrides DOCUMENT sets (None = the paper's rows) narrows the plan, the frozen identities and
    # the persisted workload alike — read once here from the same document the hold-outs are read from, under the same model scoping
    smoke = smoke_goals(overrides, model_key)
    rows = [r for r in unit_plan(model_key, base_view, table=table, families=families, nshards=nshards, sha=code_sha, clusters=None, smoke_goals=smoke,
                                 reruns=reruns)
            if r["registers"]]
    all_evals = sorted({e for r in rows for e in r["evals"]})
    try:
        canon = canonical_instruments(model_id, all_evals, seeds_dir) if all_evals else {}
    except RuntimeError as exc:
        if strict:
            raise
        canon = {e: {"error": str(exc)} for e in all_evals}
    units: List[dict] = []
    for r in rows:
        evals = {}
        for e in r["evals"]:
            mode = "on" if (e == "gsm8k" and thinking.get("gsm8k") == "on") else "off"
            c = canon.get(e) or {}
            ref = c.get("decode_policy_ref")
            if r["family"] == "padding":
                inst, src = padded_instrument_id(model_key, model_id, e, r["padding_tokens"], filler, seeds_dir=seeds_dir)
            elif "error" in c or not c.get("instrument_id"):
                if strict:
                    raise RuntimeError(f"{r['logical_work_id']}: the registry planner mints no canonical instrument for eval {e!r} on {model_id}: {c.get('error', 'no id')}")
                inst, src = None, f"unresolved: {c.get('error', 'no id')}"
            else:
                inst, src = c["instrument_id"], f"jobplan canonical_instrument ({c.get('seed_instrument_id')})"
            cov = dict(r["coverage"][e])
            if cov["layout"] == "capability_cellstore":
                cov.update(item_set_id=c.get("item_set_id"), n_items=c.get("n_items"), items_sha256=c.get("items_sha256"))
            evals[e] = {"eval_id": e, "instrument_id": inst, "instrument_id_source": src, "decode_policy_ref": ref, "thinking_mode": mode,
                        "expected_shards": r["expected_shards"][e], "coverage": cov}
        units.append({"logical_work_id": r["logical_work_id"], "family": r["family"], "stage_family": r["stage_family"], "unit": r["unit"],
                      "evals": evals, "padding_tokens": r["padding_tokens"], "script": r["script"], "sbatch_name": r["name"], "array": r["array"],
                      "registration": r["registration"], "private_run": r.get("private_run"), "model_id": model_id, "code_sha": code_sha,
                      # the rerun pair's provenance (the stamp's declaration): which frozen unit it re-runs, the ruling, its cells
                      **({"rerun_of": r["rerun_of"], "rerun_source": r.get("rerun_source"), "cells": list(r.get("cells") or [])} if r.get("rerun_of") else {})})
    attach_expected(units, model_key, nshards=nshards, goals_jsonl=goals_jsonl, strict=strict, wave_config=wave_config, petri_bundle=petri_bundle, table=table,
                    smoke_goals=smoke)
    if labels:
        # commit 26: the composed WORKLOAD identity of every unit over the pinned cluster order (the multi-turn targets' group is a function
        # of that order) — persisted so a later baseline never re-derives it from a newer working tree
        placed = {r["logical_work_id"]: r for r in unit_plan(model_key, base_view, table=table, families=families, nshards=nshards, sha=code_sha, clusters=list(labels),
                                                             smoke_goals=smoke, reruns=reruns) if r["registers"]}
        for u in units:
            row = placed.get(u["logical_work_id"])
            if row is None:
                raise RuntimeError(f"{u['logical_work_id']}: the plan over the pinned clusters {list(labels)} has no row for a frozen unit — the workload identity cannot be frozen")
            u["workload"] = workload_identity_of_row(row, code_sha)
    apply_hold_outs(units, hold_out_evals(overrides, model_key))      # Stamped, never deleted (commit 30); the document's model must be THIS model (30-fix)
    return units


def cmd_plan(a) -> int:
    fams = [f for f in a.families.split(",") if f] if a.families else None
    clusters = [c for c in a.clusters.split(",") if c] if a.clusters else None
    only = [c for c in a.only_clusters.split(",") if c] if a.only_clusters else None
    try:
        # M10: the smoke goal-shard knob is read from the model's overrides DOCUMENT (`--overrides`, default $SISTEER_RUN_OVERRIDES — the
        # absolute path every stage exports after sourcing it), never from the sourced environment; its refusals are the planner's
        smoke = smoke_goals(getattr(a, "overrides", None), a.model)
    except ValueError as exc:
        print(f"REFUSED: plan: {exc}", file=sys.stderr); return 2
    inv_path = getattr(a, "inventory", None)
    reruns = None
    if inv_path and os.path.isfile(inv_path):
        # 26d: the live plan runs with the knobs the frozen inventory pinned, or not at all — a changed attacker cap would size extra pool
        # members for the running wave without a new wave; a smoke's goal subset dropped or changed after the freeze is the same refusal (M10)
        try:
            inv_doc = json.load(open(inv_path, encoding="utf-8")) or {}
        except (OSError, json.JSONDecodeError) as exc:
            print(f"REFUSED: plan: the frozen inventory {inv_path} is unreadable ({exc})", file=sys.stderr); return 2
        pinned = inv_doc.get("plan_knobs")
        # the BoN rerun pair(s) the inventory was STAMPED with for THIS view (`rerun-units`) are planned beside the frozen rows — the
        # one source 06's base-view check (frozen units == planned units) and the submission read; another view's plan carries none
        reruns = inventory_reruns(inv_doc, a.view)
        kdiff = plan_knob_diff(pinned, freeze_plan_knobs(smoke))
        if kdiff:
            print(f"REFUSED: plan: the live plan's knobs differ from the frozen inventory's ({inv_path}): " + ", ".join(f"{k} pinned {p!r}, live {l!r}" for k, (p, l) in kdiff.items())
                  + f" — restore the knob (SISTEER_MT_SEED_BASE / SISTEER_MT_ATTACKER_GPUS / SISTEER_SHARD_GROUPS in the environment; {SMOKE_GOALS_KEY} in the "
                  f"overrides document the stage runs under) or freeze a NEW wave; nothing planned", file=sys.stderr)
            return 2
    try:
        declared: Dict[str, str] = {}
        ledger = getattr(a, "ledger", None)
        if ledger:
            pp = placement_path(ledger, a.view)
            man = read_placement(pp)
            _check_manifest_view(man, a.view, pp)
            declared = declared_move_map(man)
        cli_moves = parse_move_groups(getattr(a, "move_group", None) or [])
        re_move = bool(getattr(a, "re_move", False))
        if re_move:
            # the same validation `placement-manifest ensure --re-move` applies (move-knob-2): a re-point names a DECLARED group
            undeclared = sorted(gn for gn in cli_moves if gn not in declared)
            if undeclared or not cli_moves:
                raise ValueError(f"--re-move names {undeclared or 'no group'} with no declaration to re-point — a first move is `--move-group` alone")
        moves = merge_move_groups(declared, cli_moves, re_move=re_move)
        rows = unit_plan(a.model, a.view, table=a.models_table, families=fams, nshards=a.nshards, sha=a.sha or "{sha}", clusters=clusters,
                         only_clusters=only, move_groups=moves or None,
                         cells=cells_selection(getattr(a, "cells", None)),      # cells-fix-2: absent -> None; '' / ',' / blank tokens refuse by name
                         smoke_goals=smoke, reruns=reruns, pass_=getattr(a, "pass_", None) or "first",
                         overflow_cells=cells_selection(getattr(a, "overflow_cells", None)), overflow_cluster=getattr(a, "overflow_cluster", None) or None)
    except ValueError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr); return 2
    if a.json:
        print(json.dumps(rows, indent=2))
    else:
        sys.stdout.write(plan_tsv(rows))
    return 0


def freeze_hold_out_receipts(model: Optional[str], units: List[dict], ledger: str, *, repair: bool = False) -> int:
    """30-fix / 30-fix-2: one append-only hold-out receipt per overrides document for the stamps `units` carry
    (admission requires a matching receipt per stamp), appended + fsynced when missing — BEFORE the stamped inventory is published (the proof
    before the publish: a death between the two leaves a receipt without an inventory, which the next freeze publishes, never an inventory
    without its receipt, which admission refuses) — and again on an unchanged re-freeze (`repair`: an inventory a dying freeze published
    without its receipt gets it back without a separate `hold-out` invocation). Returns how many receipts were appended."""
    stamped_by_src: Dict[Tuple[str, str], Tuple[dict, Dict[str, List[str]]]] = {}
    for u in units:
        for e, h in (u.get("held_out") or {}).items():
            k = (str(h.get("override_id")), str(h.get("overrides_sha256")))
            stamped_by_src.setdefault(k, (h, {}))[1].setdefault(e, []).append(u["logical_work_id"])
    if not stamped_by_src:
        return 0
    have = hold_out_receipts(ledger); n = 0
    for h, stamped in stamped_by_src.values():
        # 30-fix-3: coverage is per stamped (eval, unit) — admission requires a receipt for EACH pair — so a document
        # already receipted for one family's stamps (the families were frozen one at a time) still receipts the pairs a later, wider freeze adds
        # under the same override id and sha; exactly the missing set is receipted (a document-level `any` left the new family's stamps bare)
        missing = {e: [lw for lw in lws if not any(hold_out_receipt_matches(r, h, e, lw) for r in have)] for e, lws in stamped.items()}
        missing = {e: lws for e, lws in missing.items() if lws}
        if not missing:
            continue
        append_fsync(ledger, json.dumps(hold_out_event(model, missing, h, "freeze"), sort_keys=True)); n += 1
        print(f"hold-out receipt {'REPAIRED (missing for the published inventory) in' if repair else 'appended to'} {ledger} ({h.get('source')}: "
              + ", ".join(f"{e} on {len(lws)} unit(s)" for e, lws in sorted(missing.items())) + ")")
    return n


def petri_skip(families: Optional[List[str]], petri_bundle: Optional[str]) -> Tuple[Optional[List[str]], Dict[str, str]]:
    """(families, skipped) for a freeze: with the Petri family in play and NO copy of the frozen scenario bundle anywhere the lookup
    looks (PetriBundleAbsent: nothing named, no store, no lock location mounted, no published copy in this checkout), the family is
    SKIPPED by name — `skipped` = {"petri": why} and the families are every other one — rather than refusing the whole inventory (the
    worked example's other families do not depend on it). Petri named in --families is refused instead (RuntimeError): asked for, it
    cannot be frozen. Any other trouble (a named dir that is absent, a copy that is there and wrong, no lock) is left to
    expected_units, which refuses it under strict."""
    if families is not None and "petri" not in families:
        return families, {}
    try:
        petri_bundle_dir(petri_lock(), petri_bundle)
    except PetriBundleAbsent as exc:
        if families is not None:
            raise RuntimeError(f"--families names petri, and {exc}")
        return [f for f in FAMILIES if f != "petri"], {"petri": str(exc)}
    except RuntimeError:
        pass
    return families, {}


def cmd_freeze(a) -> int:
    fams = a.families.split(",") if a.families else None
    try:
        # the Petri family is skipped BY NAME when its frozen scenario bundle is genuinely absent (petri_skip); recorded in the header
        fams, skipped = petri_skip(fams, a.petri_bundle)
        # M10: the smoke goal-shard knob of the model's overrides document (None = the paper's rows) — expected_units reads the same document
        # for the rows and the identities; the header pins the knob (`plan_knobs`) and records the document it came from (`smoke_goals`)
        smoke = smoke_goals(a.overrides, a.model)
        units = expected_units(a.model, a.code_sha, a.models_table, fams, a.filler, a.nshards, a.seeds, strict=not a.dry_run, goals_jsonl=a.goals_jsonl,
                               wave_config=a.wave_config, petri_bundle=a.petri_bundle, labels=[c for c in (a.clusters or "").split(",") if c] or None,
                               overrides=a.overrides)
        groups = [list(g) for g in shard_groups(int(smoke["nshards"]) if smoke else a.nshards, smoke=smoke)]     # the multi-turn targets' groups as planned
    except (RuntimeError, ValueError) as exc:
        print(f"REFUSED: freeze-inventory: {exc}", file=sys.stderr); return 2
    for f, why in sorted(skipped.items()):
        print(f"SKIPPED: freeze-inventory: the {f} family is left out of this inventory: {why}. 06 --family all drops it and the gate has "
              f"no {f} unit to rule; restore the bundle (data/petri/scenarios in the repository) and run the freeze again to include it", file=sys.stderr)
    roster_sha = None
    if a.roster and os.path.isfile(a.roster):
        roster_sha = hashlib.sha256(open(a.roster, "rb").read()).hexdigest()
    doc = {"schema": 2, "model": a.model, "view": M.baseline_view(a.model, a.models_table), "frozen_utc": utc_now(), "code_sha": a.code_sha, "roster_sha256": roster_sha,
           "nshards": a.nshards, "shard_groups": groups, "plan_knobs": freeze_plan_knobs(smoke), "smoke_goals": smoke_goals_header(smoke),
           "n_units": len(units), "units": units}
    if skipped:
        doc["skipped_families"] = dict(skipped)   # read by 06 (--family all drops these; naming one refuses)
    # the COVERAGE CONTRACT — the unit set, every eval's identity keys and the probe's per-layout done predicates — recorded with
    # the inventory at freeze, ONCE; `06 --gate` compares the current probe's contract against it structurally before reading a started wave
    doc["coverage_contract"] = coverage_contract_record(units, this_tree_done_predicates(), source="frozen", frozen_utc=doc["frozen_utc"], inventory_code_sha=a.code_sha)
    why = refreeze_refusal(a.out, units, a.code_sha, doc["plan_knobs"])
    if why:
        print(f"REFUSED: freeze-inventory: {why}", file=sys.stderr); return 2
    ledger = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(a.out))), "attempts.jsonl")
    if os.path.isfile(a.out):
        old = json.load(open(a.out, encoding="utf-8"))
        if old.get("units") == units and old.get("code_sha") == a.code_sha and old.get("roster_sha256") == roster_sha and old.get("plan_knobs") == doc["plan_knobs"]:
            print(f"inventory unchanged: {a.out} ({len(units)} units)")
            if not a.dry_run:
                freeze_hold_out_receipts(a.model, units, ledger, repair=True)     # 30-fix-2: a published inventory whose receipt a dying freeze never wrote
            return 0
        keep = a.out.replace(".json", f".{old.get('frozen_utc', 'old').replace(':', '')}.json")
        if not a.dry_run:
            os.replace(a.out, keep)
        old_ids = {u["logical_work_id"] for u in old.get("units", [])}; new_ids = {u["logical_work_id"] for u in units}
        print(f"inventory CHANGED: previous kept at {keep}; +{sorted(new_ids - old_ids)} -{sorted(old_ids - new_ids)}"
              f"{' code_sha ' + str(old.get('code_sha')) + ' -> ' + a.code_sha if old.get('code_sha') != a.code_sha else ''}")
    held = {}
    for u in units:
        for e, h in (u.get("held_out") or {}).items():
            held.setdefault((e, h["source"]), []).append(u["logical_work_id"].split("/", 2)[2])
    for (e, src), lws in sorted(held.items()):
        print(f"held out by ruling: {e} on {', '.join(lws)} — {src} (the units keep the eval and its workload; admission stops requiring it)")
    unresolved = [(u["logical_work_id"], e, ev["instrument_id_source"]) for u in units for e, ev in u["evals"].items() if not ev["instrument_id"]]
    if unresolved:
        print(f"note: {len(unresolved)} eval(s) carry instrument_id null (first: {unresolved[0][1]} of {unresolved[0][0]}: {unresolved[0][2]})")
    n_ids = sum(len(ids) for u in units for ev in u["evals"].values() for ids in (ev["coverage"].get("expected") or {}).values())
    unverified = sorted({u["logical_work_id"] for u in units for ev in u["evals"].values() if ev["coverage"]["layout"] == "unverified"})
    print(f"coverage identities frozen: {n_ids} work ids over {sum(1 for u in units for ev in u['evals'].values() if ev['coverage'].get('expected'))} eval(s); "
          f"item-set sizes for {sum(1 for u in units for ev in u['evals'].values() if ev['coverage'].get('n_items'))}; "
          f"run-time manifests for {sum(1 for u in units for ev in u['evals'].values() if ev['coverage']['layout'] == 'bon_climb_flat')}"
          + (f"; UNVERIFIED layout (the gate refuses by name): {unverified}" if unverified else ""))
    if a.dry_run:
        print(f"DRY-RUN: write {a.out} ({len(units)} units)")
        return 0
    # 30-fix-2: the PROOF before the publish — the stamps' receipts are appended and fsynced BEFORE the stamped
    # inventory is written (30-fix wrote them after it: a death between the two left an inventory admission refused and an unchanged re-freeze
    # could not repair)
    freeze_hold_out_receipts(a.model, units, ledger)
    _crash("freeze_after_receipt")
    write_atomic(a.out, doc)
    print(f"frozen {len(units)} logical units -> {a.out}")
    labels = [c for c in (a.clusters or "").split(",") if c]
    if labels and (fams is None or "multiturn" in fams):    # The ORDERED placement universe is pinned with the wave, once
        base_view = M.baseline_view(a.model, a.models_table)
        man = placement_from_plan(a.model, base_view, labels, table=a.models_table, nshards=a.nshards, sha=a.code_sha, smoke_goals=smoke)
        pp = placement_path(a.out, base_view)     # u0: the historic file; the single-phase recipe's one wave: ITS per-view file, the one 06 reads for it
        old_man = read_placement(pp)
        if old_man is not None and (old_man["labels"] != man["labels"] or old_man["units"] != man["units"]):
            print(f"REFUSED: freeze-inventory: the placement manifest at {pp} pins {old_man['labels']} and would change to {man['labels']} / "
                  f"{sum(1 for k in man['units'] if man['units'][k] != old_man['units'].get(k))} moved unit(s) — a wave's placement is pinned once", file=sys.stderr)
            return 2
        if old_man is None:
            write_atomic(pp, man)
            print(f"placement manifest pinned from the plan: {len(man['units'])} multi-turn units over {labels} -> {pp}")
    return 0


def cmd_hold_out(a) -> int:
    """`hold-out --out <inventory> --overrides F [--dry-run]`: stamp an EXISTING frozen inventory's units with the model overrides file's
    hold-outs (the live u0 wave was frozen before the ruling) and append a `hold_out` event to the attempts ledger beside it.
    Idempotent (an identical stamp is 'unchanged'); a unit already held out from ANOTHER source refuses by name; nothing is deleted and
    no contract hash moves (unit_contract ignores the stamp)."""
    if not os.path.isfile(a.out):
        print(f"REFUSED: hold-out: no inventory at {a.out}", file=sys.stderr); return 2
    inv = json.load(open(a.out, encoding="utf-8"))
    try:
        held = hold_out_evals(a.overrides, str(inv.get("model") or "") or None)      # 30-fix: the document's model is the inventory's, or nothing
    except ValueError as exc:
        print(f"REFUSED: hold-out: {exc}", file=sys.stderr); return 2
    if not held:
        print(f"hold-out: {a.overrides} holds nothing out — inventory unchanged"); return 0
    conflicts = [f"{u['logical_work_id']}: eval {e} is already held out by {cur.get('source')!r}, not {held[e]['source']!r}"
                 for u in inv.get("units", []) for e, cur in (u.get("held_out") or {}).items() if e in held and cur.get("source") != held[e]["source"]]
    if conflicts:
        print("REFUSED: hold-out: " + "; ".join(conflicts) + " — a hold-out is never re-attributed; lift it by its own ruling first", file=sys.stderr); return 2
    before = json.dumps(inv.get("units", []), sort_keys=True)
    stamped = apply_hold_outs(inv.get("units", []), held)
    src = next(iter(held.values()))
    ledger = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(a.out))), "attempts.jsonl")
    # the receipt this document's stamps need: {eval: [every unit of the inventory that carries the eval]} — the same set whether the stamp is
    # being made now or was made by an interrupted apply / an older code path that wrote the inventory first (30-fix MEDIUM)
    want = {e: sorted(u["logical_work_id"] for u in inv.get("units", []) if e in (u.get("evals") or {})) for e in held}
    want = {e: lws for e, lws in want.items() if lws}
    receipts = hold_out_receipts(ledger)
    def receipted() -> bool:
        return all(any(hold_out_receipt_matches(r, src, e, lw) for r in receipts) for e, lws in want.items() for lw in lws)
    if json.dumps(inv.get("units", []), sort_keys=True) == before:
        if receipted() or a.dry_run:
            print("hold-out: inventory unchanged (already stamped" + ("" if receipted() else "; DRY-RUN: the receipt is missing and would be appended") + ")"); return 0
        # stamped by an interrupted apply (or by the pre-30-fix order: inventory first, event second): the receipt is REPAIRED, never left missing
        append_fsync(ledger, json.dumps(hold_out_event(inv.get("model"), want, src, "hold-out"), sort_keys=True))
        print(f"hold-out: inventory already stamped; its receipt was MISSING from {ledger} — appended now ({src['source']})"); return 0
    summary = ", ".join(f"{e} on {len(lws)} unit(s)" for e, lws in sorted(stamped.items()))
    if a.dry_run:
        print(f"DRY-RUN: hold-out would stamp {summary} -> {a.out} ({src['source']})"); return 0
    # EVENT FIRST (30-fix MEDIUM): the append-only receipt is on disk (file + directory fsync'd) before the inventory is replaced, so a crash
    # between the two leaves a receipt without a stamp — the retry stamps and appends nothing — never a stamp without its receipt
    if not receipted():
        append_fsync(ledger, json.dumps(hold_out_event(inv.get("model"), want, src, "hold-out"), sort_keys=True))
        _crash("hold_out_after_event")
        note = "ledger event appended"
    else:
        note = "receipt already on the ledger (an interrupted apply wrote it first)"
    write_atomic(a.out, inv)
    print(f"hold-out: stamped {summary} -> {a.out}; {note} ({src['source']})")
    return 0


#: the fields a rerun RECEIPT must equal the stamp's declaration on to cover it (rerun_receipt_matches), beside the tree (`code_sha`)
RERUN_RECEIPT_KEYS = ("model", "view", "slug", "cells", "source", "of", "units")
#: how a receipt is NAMED (rerun_receipt_ref): its unique durable id first, then every field a refusal spells — the write time, the tree, the
#: pair (model, view, slug), the ruling and the cells. A `supersedes` entry retires a receipt only when it equals this WHOLE reference.
RERUN_RECEIPT_REF_KEYS = ("receipt_id", "utc", "code_sha", "model", "view", "slug", "source", "cells")


def rerun_event(decl: Mapping[str, object], code_sha: str, at: str, *, supersedes: Optional[Sequence[Mapping[str, object]]] = None) -> dict:
    """The append-only RECEIPT of a rerun stamp (`rerun_units` in the attempts ledger): a unique durable `receipt_id` (32 hex, minted here and
    written nowhere but the receipt — the geometry revocation's nonce, slurm_revoke_geometry), the declaration (model, view, slug, cells,
    source, the superseded units and the pair), the tree the pair runs at, and where it was applied. Written BEFORE the inventory (the
    hold-out's order): a death between the two leaves a receipt without a stamp, which the retry completes — never a stamp without its
    receipt. `supersedes` names the earlier receipt(s) of the SAME pair under another declaration / tree this one retires (an interrupted
    apply's receipt the operator chose not to complete — `--supersede-receipt`; or the receipt a stamp landed beside without its own), each
    by its whole reference (rerun_receipt_ref): the ledger then reads what was attempted, what was retired, and which receipt the stamp is."""
    ev = {"event": "rerun_units", "receipt_id": secrets.token_hex(16), "model": decl["model"], "view": decl["view"], "slug": decl["slug"], "cells": list(decl["cells"]),
          "source": decl["source"], "of": dict(decl["of"]), "units": dict(decl["units"]), "code_sha": code_sha, "at": at, "utc": utc_now()}
    if supersedes:
        ev["supersedes"] = [rerun_receipt_ref(r) for r in supersedes]
    return ev


def rerun_receipts(ledger: str) -> List[dict]:
    """Every `rerun_units` receipt in the attempts ledger beside `ledger`."""
    return [r for r in _ledger_events(ledger) if r.get("event") == "rerun_units"]


def rerun_receipt_ref(rec: Mapping[str, object]) -> dict:
    """How a receipt is NAMED by another (a `supersedes` entry) and in a refusal: its `receipt_id` and its whole reference
    (RERUN_RECEIPT_REF_KEYS — write time, tree, pair, ruling, cells). Never the write time + tree alone: `utc_now` has one-second precision,
    so two receipts written on one tree in one second — two pairs' applies in a row, or one pair's under two rulings — share both, and naming
    by those two retired the wrong receipt (an unrelated pair's supersession hid this pair's orphan; a receipt superseding a same-second
    orphan retired itself). A receipt written before the id existed (this branch's 9ddea1ad2 only; no durable ledger
    carries one) is named with `receipt_id` null and the rest of its reference."""
    return {k: (list(rec.get(k) or []) if k == "cells" else rec.get(k)) for k in RERUN_RECEIPT_REF_KEYS}


def rerun_receipt_matches(rec: Mapping[str, object], decl: Mapping[str, object], code_sha: str) -> bool:
    """A receipt COVERS a rerun stamp when its WHOLE declaration (model, view, slug, cells, source, the superseded units and the pair) and
    the tree it names equal the stamp's — never the pair's name alone. After a crash between the receipt and the inventory write, a retry
    on another tree or under a corrected ruling is a DIFFERENT stamp: recovering it under the old receipt would stamp the pair at the new
    tree / ruling while the durable ledger attributes the rerun to the old one."""
    return all(rec.get(k) == decl[k] for k in RERUN_RECEIPT_KEYS) and str(rec.get("code_sha") or "") == str(code_sha or "")


def rerun_receipt_superseded(rec: Mapping[str, object], receipts: Sequence[Mapping[str, object]]) -> bool:
    """Whether a receipt on the ledger names `rec` superseded: a `supersedes` entry EQUAL to rec's whole reference (rerun_receipt_ref) — the
    id and every field. A partial reference (the retired (utc, code_sha) shape), another receipt's id, or the right id beside a drifted field
    names nothing: the receipt stays visible and `rerun-units` keeps refusing by its name, the safe direction."""
    ref = rerun_receipt_ref(rec)
    return any(s == ref for r in receipts for s in (r.get("supersedes") or []))


def rerun_receipt_conflicts(receipts: Sequence[Mapping[str, object]], decl: Mapping[str, object], code_sha: str) -> List[dict]:
    """The receipts of the SAME pair (model, view, slug) that do NOT cover this stamp and that no receipt on the ledger has superseded
    (rerun_receipt_superseded: by the receipt's own id and whole reference): the orphans of an interrupted apply under another declaration /
    tree, which a stamp may not silently inherit — `rerun-units` refuses them by name, or retires them under `--supersede-receipt`."""
    return [dict(r) for r in receipts
            if all(r.get(k) == decl[k] for k in ("model", "view", "slug")) and not rerun_receipt_matches(r, decl, code_sha)
            and not rerun_receipt_superseded(r, receipts)]


def _rerun_receipt_words(recs: Sequence[Mapping[str, object]]) -> str:
    return "; ".join(f"receipt {r.get('receipt_id') or '(no id)'} of {r.get('utc')}: tree {str(r.get('code_sha') or '?')[:12]}, source {r.get('source')!r}, cells {list(r.get('cells') or [])}"
                     for r in recs)


def cmd_rerun_units(a) -> int:
    """`rerun-units --out <inventory> --cells C,… --source "ruling" --code-sha S [the freeze's inputs] [--dry-run]`: STAMP an existing, started
    inventory with the BoN rerun pair of the named cells — `bon/store512_<slug>` (target_static.sbatch, FAMILIES=bon, the ruling's
    cells alone) -> `bon/climb_<slug>` (bon_climb.sbatch, afterok) — frozen exactly as the freeze freezes a unit (the planner's canonical
    instrument, the coverage identities, the workload identity over the pinned clusters), each carrying `rerun_of` / `rerun_source` / `cells`.
    Refuses by name: no inventory; a cell outside the view's explicit_cells; no source; a view whose inventory does not carry the original
    store512 / climb (an excluding view, a partial freeze); an original store512 with NO recorded attempt (a wave that has not run its store
    serves the cell at the corrected window itself — nothing to re-run); a pair already stamped under a DIFFERENT declaration or contract (a
    rerun is stamped once; an identical stamp is `unchanged`). The ledger receipt (`rerun_units`) is appended and fsync'd FIRST, then the
    inventory is replaced (the hold-out's order); `plan --inventory` then plans the pair for the declaration's view. A receipt covers a stamp
    only when its WHOLE declaration and tree equal the stamp's (rerun_receipt_matches): a retry after an interrupted apply that names another
    tree or ruling finds the orphan receipt CONFLICTING and is refused by name — complete the apply under the receipt's declaration, or pass
    `--supersede-receipt` to append a receipt that names the orphan superseded — by the orphan's own `receipt_id` and whole reference, never
    by its second-resolution write time + tree, which two pairs' same-second receipts share — and stamp this declaration (the ledger then
    records both and which one the stamp is). An inventory already stamped beside a receipt that does not cover it (a stamp without its own receipt) gets its
    receipt REPAIRED, naming the foreign one superseded — the inventory is the stamp, and there is no second outcome to choose."""
    if not os.path.isfile(a.out):
        print(f"REFUSED: rerun-units: no inventory at {a.out} (05_stage_roster.sh --step freeze first; a rerun stamps a STARTED wave's inventory)", file=sys.stderr); return 2
    inv = json.load(open(a.out, encoding="utf-8"))
    model, view = str(inv.get("model") or ""), str(inv.get("view") or "")
    if not model or not view:
        print(f"REFUSED: rerun-units: {a.out} names no model / view (schema {inv.get('schema')!r}) — not a frozen inventory", file=sys.stderr); return 2
    if getattr(a, "model", None) and a.model != model:
        print(f"REFUSED: rerun-units: --model {a.model} but the inventory {a.out} is {model}'s — a rerun never travels to another model", file=sys.stderr); return 2
    state = os.path.dirname(os.path.dirname(os.path.abspath(a.out)))
    ledger = os.path.join(state, "attempts.jsonl")
    try:
        sel = cells_selection(a.cells)
        if sel is None:
            raise ValueError("rerun: --cells is required (the cell(s) the ruling re-runs)")
        decl = bon_rerun_declaration(sel, a.source, model, view, roster_path=a.roster, table=a.models_table)
    except ValueError as exc:
        print(f"REFUSED: rerun-units: {exc}", file=sys.stderr); return 2
    missing = [lw for lw in decl["of"].values() if inventory_unit(inv, lw) is None]
    if missing:
        print(f"REFUSED: rerun-units: {missing} are not units of the inventory {a.out} (an excluding view, or a freeze without the bon family) — "
              f"a rerun re-runs a FROZEN unit", file=sys.stderr); return 2
    attempted = {str(r.get("logical_work_id")) for r in _ledger_events(ledger) if r.get("event") == "attempt"}
    if decl["of"]["store512"] not in attempted:
        print(f"REFUSED: rerun-units: {decl['of']['store512']} has no recorded attempt in {ledger} — nothing to re-run: a wave that has not run its 512 store "
              f"serves {decl['cells']} at the corrected window itself (bon_long_si.sh); a rerun is of a RUN unit (D913 (ii))", file=sys.stderr); return 2
    try:
        units = expected_units(model, a.code_sha, a.models_table, ["static", "bon"], a.filler, int(inv.get("nshards") or a.nshards), a.seeds, strict=not a.dry_run,
                               goals_jsonl=a.goals_jsonl, wave_config=a.wave_config, petri_bundle=a.petri_bundle,
                               labels=[c for c in (a.clusters or "").split(",") if c] or None, overrides=a.overrides, reruns=[decl])
    except (RuntimeError, ValueError) as exc:
        print(f"REFUSED: rerun-units: {exc}", file=sys.stderr); return 2
    new = [u for u in units if u.get("rerun_of")]
    want = set(decl["units"].values())
    if {u["logical_work_id"] for u in new} != want:
        print(f"REFUSED: rerun-units: the plan composed {sorted(u['logical_work_id'] for u in new)} for the pair, not {sorted(want)} — the declaration and the plan disagree", file=sys.stderr); return 2
    existing = {u["logical_work_id"]: u for u in inv.get("units", [])}
    receipts = rerun_receipts(ledger)
    # a receipt covers THIS stamp only on its whole declaration + tree (the pair's name alone let a retry after an
    # interrupted apply stamp a newer tree / ruling under the old receipt); the same pair's other, unsuperseded receipts CONFLICT
    covering = [r for r in receipts if rerun_receipt_matches(r, decl, a.code_sha)]
    conflicting = rerun_receipt_conflicts(receipts, decl, a.code_sha)
    supersede = bool(getattr(a, "supersede_receipt", False))
    if all(lw in existing for lw in want):
        same = all(existing[u["logical_work_id"]] == u for u in new)
        if not same:
            diff = [u["logical_work_id"] for u in new if existing[u["logical_work_id"]] != u]
            print(f"REFUSED: rerun-units: {sorted(want)} are already stamped on {a.out} with a different contract / provenance ({diff}) — a rerun is stamped ONCE; "
                  f"a second re-run is a new declaration (other cells), never a re-stamp", file=sys.stderr); return 2
        if (covering and not conflicting) or a.dry_run:
            print("rerun-units: inventory unchanged (already stamped" + ("" if covering and not conflicting else "; DRY-RUN: the stamp's receipt is "
                  + ("missing" if not covering else "on the ledger") + (f" and the ledger carries a receipt of this pair under another declaration / tree ({_rerun_receipt_words(conflicting)}) that the stamp's receipt would name superseded" if conflicting else "")
                  + " — it would be appended") + ")"); return 0
        # the stamp IS the inventory's: its receipt is repaired, never left missing (an interrupted apply before this order, or a stamp that
        # landed beside another declaration's receipt under the pair's name) — a foreign receipt of the pair is named superseded, not inherited
        append_fsync(ledger, json.dumps(rerun_event(decl, a.code_sha, "rerun-units", supersedes=conflicting), sort_keys=True))
        print(f"rerun-units: inventory already stamped; its receipt was MISSING from {ledger} — appended now ({decl['source']}; tree {a.code_sha[:12]})"
              + (f"; it names superseded the receipt(s) the ledger carried for this pair under another declaration / tree, which never covered this stamp: {_rerun_receipt_words(conflicting)}" if conflicting else "")); return 0
    if any(lw in existing for lw in want):
        print(f"REFUSED: rerun-units: {sorted(lw for lw in want if lw in existing)} are already units of {a.out} while {sorted(lw for lw in want if lw not in existing)} are not — "
              f"a half-stamped pair; the inventory was edited by hand", file=sys.stderr); return 2
    summary = " -> ".join(decl["units"][u] for u in BON_RERUN_OF)
    if conflicting and not supersede:
        print(f"REFUSED: rerun-units: {ledger} already carries a `rerun_units` receipt for {summary} under another declaration / tree, and {a.out} carries no stamp of the pair "
              f"(an interrupted apply): {_rerun_receipt_words(conflicting)} — this stamp names tree {a.code_sha[:12]}, source {decl['source']!r}, cells {decl['cells']}; "
              f"stamping it under that receipt would attribute the rerun to the receipt's tree / ruling. Complete the interrupted apply with the receipt's exact "
              f"--code-sha / --source / --cells, or pass --supersede-receipt to append a receipt that names it superseded and stamp THIS declaration", file=sys.stderr); return 2
    if supersede and not conflicting:
        print(f"REFUSED: rerun-units: --supersede-receipt, but {ledger} carries no receipt for {summary} under another declaration / tree — nothing to supersede; drop the flag"
              + (" (this stamp's own receipt is already on the ledger: an interrupted apply wrote it first)" if covering else ""), file=sys.stderr); return 2
    # a stamp AMENDS the inventory's coverage contract (its unit set grows by the receipted pair) but never rewrites its record — and
    # it refuses, BEFORE the receipt, when this tree's probe would change WHAT counts as done: a predicate change is a ruling, not a stamp
    try:
        cc_rec, cc_src, _cc_where = frozen_coverage_contract(state)
    except ValueError as exc:
        print(f"REFUSED: rerun-units: {exc}", file=sys.stderr); return 2
    cc_amend = None
    if cc_rec is not None:
        # the record is read through the same check the gate reads it through (version, the three parts, the digest): a record of another contract
        # version, or one edited by hand, is refused by name BEFORE the receipt — a stamp over a record this tree cannot compare amends nothing
        cc_body, cc_why = coverage_contract_record_check(cc_rec, cc_src, _cc_where)
        if cc_why:
            print(f"REFUSED: rerun-units: {cc_why}; nothing stamped, no receipt written", file=sys.stderr); return 2
        pred_diff = [ln for ln in coverage_contract_diff(cc_body, {"done_predicates": dict(this_tree_done_predicates())}, units=False) if ln.startswith("done predicate ")]
        if pred_diff:
            print(f"REFUSED: rerun-units: this tree's coverage done predicates differ from the inventory's {cc_src} contract ({cc_rec.get('sha256', '')[:12]}): "
                  f"{'; '.join(pred_diff)} — a predicate change is a ruling (D3003 (2)), not a stamp; nothing stamped, no receipt written", file=sys.stderr); return 2
        after = coverage_contract(list(inv.get("units", [])) + new, this_tree_done_predicates())
        cc_amend = {"from_sha256": cc_rec.get("sha256"), "to_sha256": after["sha256"], "units_added": sorted(want), "record_source": cc_src,
                    "note": "informational: 06 --gate derives the amendment from the stamp's rerun_units receipt (D3003 (2)); the record itself is never rewritten"}
    if a.dry_run:
        print(f"DRY-RUN: rerun-units would stamp {summary} (cells {decl['cells']}; {decl['source']}; tree {a.code_sha[:12]}) -> {a.out} and append the `rerun_units` receipt to {ledger}"
              + (f", naming superseded: {_rerun_receipt_words(conflicting)}" if conflicting else "")
              + (f"; coverage contract: amends the {cc_src} record {str(cc_rec.get('sha256'))[:12]} -> {cc_amend['to_sha256'][:12]} (+{sorted(want)})" if cc_amend
                 else "; coverage contract: the inventory carries no record yet (frozen before the field existed; 06 --gate admits the stamped set ONCE, D3003 (2))"))
        return 0
    if not covering or conflicting:
        append_fsync(ledger, json.dumps(rerun_event(decl, a.code_sha, "rerun-units", supersedes=conflicting), sort_keys=True))
        _crash("rerun_after_event")
        note = "ledger event appended" + (f" (it names superseded: {_rerun_receipt_words(conflicting)})" if conflicting else "")
    else:
        note = "receipt already on the ledger (an interrupted apply wrote it first)"
    inv["units"] = list(inv.get("units", [])) + new
    inv["n_units"] = len(inv["units"])
    inv["reruns"] = list(inv.get("reruns") or []) + [{**{k: decl[k] for k in ("slug", "cells", "source", "model", "view", "of", "units")}, "code_sha": a.code_sha, "stamped_utc": utc_now(),
                                                      **({"coverage_contract_amendment": cc_amend} if cc_amend else {})}]
    write_atomic(a.out, inv)
    print(f"rerun-units: stamped {summary} (cells {decl['cells']}; {decl['source']}; tree {a.code_sha[:12]}) -> {a.out} ({inv['n_units']} units); {note}")
    print(f"next: 06_eval_waves.sh --model {model} --view {view} --family static --units bon.store512_{decl['slug']}   then   "
          f"06_eval_waves.sh --model {model} --view {view} --family bon --units bon.climb_{decl['slug']}   (each under the wave's overrides document; the climb waits afterok on the store)")
    return 0


# ---------------------------------------------------------------------------------------- the placement manifest
#: the ORDERED placement universe pinned with the wave: the site's cluster labels in config order + every multi-turn unit's assigned label
#: and group. Multi-turn placement is round-robin over the ordered labels (unit_plan), so a site config that reorders, removes or adds a
#: label between per-cluster submissions would silently move the remaining targets (Medium). The manifest is written
#: at freeze (05) beside the inventory, or — for a wave that predates it — derived ONCE from the recorded attempts (where each unit
#: actually ran) and pinned; every later per-cluster call refuses a cluster list that differs from it (order and membership) or a planned
#: assignment that differs from the pinned one, by name.
PLACEMENT_FILE = "expected_units.placement.json"


def placement_file(view: Optional[str]) -> str:
    """The per-VIEW placement namespace: u0 keeps the historic filename (the freeze writes it; the legacy composition reads
    it); every other view has its own file, so an s2r declaration never lands in — or is read from — the u0 manifest."""
    if view in (None, "u0"):
        return PLACEMENT_FILE
    if not re.fullmatch(r"[A-Za-z0-9_]+", str(view)):
        raise ValueError(f"placement namespace: view {view!r} is not a plain token")
    return f"expected_units.placement.{view}.json"


def placement_path(ledger_or_inventory: str, view: Optional[str] = None) -> str:
    d = os.path.dirname(os.path.abspath(ledger_or_inventory))
    return os.path.join(d if os.path.basename(d) == "gate" else os.path.join(d, "gate"), placement_file(view))


def _check_manifest_view(man: Optional[Mapping[str, object]], view: str, path: str) -> None:
    """A manifest read for `view` must be that view's (a u0 manifest read through an s2r call, or the reverse, refuses by name)."""
    if man is not None and man.get("view") not in (None, view):
        raise ValueError(f"the placement manifest at {path} is view {man.get('view')!r}, not {view!r} — each view has its own placement namespace")


#: the canonical pool-member id: `<model>/<view>/multiturn/mt-<view>-<cluster>/attackers/<positive integer>` (unit_plan's `logical_work_id(model, view,
#: "multiturn", f"{grp}/attackers/{i}")` with grp = `mt-<view>-<label>`); nothing else is a member, whatever a row calls itself
POOL_MEMBER_ID_RE = re.compile(r"^[^/]+/(?P<view>[^/]+)/multiturn/mt-(?P=view)-[^/]+/attackers/[1-9][0-9]*$")


def is_pool_member(lwid: str, unit: Optional[Mapping[str, object]] = None) -> bool:
    """A shared attacker pool member: the CANONICAL id (`POOL_MEMBER_ID_RE`) and, when a row or pinned unit is given with a `kind`, kind
    `attacker` — BOTH, never either (a target row mislabeled attacker in a supplied --plan-json, or a pinned
    target whose kind drifted, would otherwise ride the membership skip and a wave could omit a target's coverage). An id/kind
    disagreement RAISES by name (fail closed); the placement readers turn it into a refusal. A member's presence in the plan is pool
    SIZING (one member per three targets), not placement — the manifest RECORDS the members the wave ran."""
    by_id = bool(POOL_MEMBER_ID_RE.match(str(lwid)))
    kind = None if unit is None else unit.get("kind")
    if kind is not None:
        by_kind = str(kind) == "attacker"
        if by_id != by_kind:
            raise ValueError(f"{lwid}: the id {'is' if by_id else 'is not'} a pool member's (…/multiturn/mt-<view>-<cluster>/attackers/<n>) but the row's kind is "
                             f"{kind!r} — a unit whose id and kind disagree is refused, never skipped as a member (fail closed)")
    return by_id


def _gname(g) -> str:
    """The shard group's name as the units spell it (`g5678` for shards 5,6,7,8)."""
    return "g" + "".join(str(s) for s in g)


def _group_of_target(lwid: str) -> Optional[str]:
    """`<model>/<view>/multiturn/<attack>/<gname>` -> gname; None for a pool member or anything else."""
    parts = str(lwid).split("/")
    if len(parts) == 5 and parts[2] == "multiturn" and parts[3] in MT_ATTACKS and re.fullmatch(r"g[0-9]+", parts[4]):
        return parts[4]
    return None


def parse_move_groups(items: Optional[Sequence[str]]) -> Dict[str, str]:
    """`--move-group g5678=cluster_a` tokens -> {gname: label}; malformed or a group named twice with different labels refuses by name."""
    out: Dict[str, str] = {}
    for tok in items or []:
        if "=" not in str(tok):
            raise ValueError(f"--move-group {tok!r}: expected <shard group>=<label> (e.g. g5678=cluster_a)")
        gn, lab = str(tok).split("=", 1)
        gn, lab = gn.strip(), lab.strip()
        if not re.fullmatch(r"g[0-9]+", gn) or not lab:
            raise ValueError(f"--move-group {tok!r}: expected <shard group>=<label> (e.g. g5678=cluster_a)")
        if gn in out and out[gn] != lab:
            raise ValueError(f"--move-group {gn}: named twice with different labels ({out[gn]!r} and {lab!r})")
        out[gn] = lab
    return out


def _validated_group_moves(spec: Optional[Mapping[str, str]], gnames: List[str], labels: List[str]) -> Dict[str, str]:
    """The declared moves this plan honours: every named group must be one of the plan's shard groups and every label one of the placement
    universe (a group moves WITHIN the universe, never off it; a placement-free plan — the freeze's `*` — has nothing to move to)."""
    moves: Dict[str, str] = {}
    for gn, lab in dict(spec or {}).items():
        if gn not in gnames:
            raise ValueError(f"--move-group {gn}={lab}: {gn!r} is not a shard group of this plan ({gnames})")
        if labels == ["*"] or lab not in labels:
            raise ValueError(f"--move-group {gn}={lab}: {lab!r} is not a label of the placement universe {labels} — a group moves within the "
                             f"universe, never off it (a placement-free plan has no destination)")
        moves[gn] = lab
    return moves


def declared_move_map(man: Optional[Mapping[str, object]]) -> Dict[str, str]:
    """{gname: to-label} from a placement manifest's `declared_moves` (absent -> {})."""
    out: Dict[str, str] = {}
    for gn, d in dict((man or {}).get("declared_moves") or {}).items():       # type: ignore[union-attr]
        if isinstance(d, Mapping) and isinstance(d.get("to"), str):
            out[str(gn)] = str(d["to"])
    return out


def merge_move_groups(declared: Mapping[str, str], cli: Mapping[str, str], *, re_move: bool = False) -> Dict[str, str]:
    """The moves in force: the manifest's declared ones plus this call's. A group declared for one label and named for another is a RE-MOVE:
    refused by name unless the caller says so (`--re-move`, with a new reason) — a second destination is a decision with its own provenance,
    never a stray flag."""
    out = dict(declared)
    for gn, lab in cli.items():
        if gn in out and out[gn] != lab and not re_move:
            raise ValueError(f"--move-group {gn}={lab}: the placement manifest already declares {gn} -> {out[gn]!r}; re-pointing a declared move "
                             f"is its own decision: pass --re-move with a new --move-reason (the earlier declaration is kept in the history)")
        out[gn] = lab
    return out


def declare_moves(model_key: str, view: str, labels: List[str], moves_new: Mapping[str, str], reason: str, *, table: Optional[str] = None,
                  nshards: int = 9, sha: str = "{sha}", previous: Optional[Mapping[str, Mapping[str, object]]] = None,
                  smoke_goals: Optional[Mapping[str, object]] = None) -> Dict[str, dict]:
    """The `declared_moves` entries for NEW moves: {gname: {to, from, reason, declared_utc, units (the group's target ids), history}} — the
    provenance a later reader needs to see why a group sits off its round-robin label before its receipt lands. A RE-MOVE (`previous`
    carries the group's current declaration) records `from` = the previous destination and keeps every earlier declaration in `history`,
    oldest first; a first move's `from` is the round-robin label."""
    if not str(reason or "").strip():
        raise ValueError("--move-group needs --move-reason (the provenance of the move: why the group leaves its cluster)")
    plain = unit_plan(model_key, view, table=table, families=["multiturn"], nshards=nshards, sha=sha, clusters=list(labels), smoke_goals=smoke_goals)
    out: Dict[str, dict] = {}
    for gn, lab in moves_new.items():
        units = sorted(r["logical_work_id"] for r in plain if _group_of_target(r["logical_work_id"]) == gn)
        if not units:
            raise ValueError(f"--move-group {gn}={lab}: {gn!r} is not a shard group of this plan")
        origin = sorted({r["cluster"] for r in plain if _group_of_target(r["logical_work_id"]) == gn})
        prev = dict((previous or {}).get(gn) or {})
        history = [dict(h) for h in (prev.get("history") or [])] + ([{k: v for k, v in prev.items() if k != "history"}] if prev else [])
        out[gn] = {"to": lab, "from": (prev.get("to") if prev else (origin[0] if len(origin) == 1 else origin)), "reason": str(reason).strip(),
                   "declared_utc": utc_now(), "units": units, "history": history, **({"re_move_of": prev.get("to")} if prev else {})}
    return out


def declared_moves_tsv(man: Optional[Mapping[str, object]]) -> str:
    """The view's CURRENT declarations as TSV `group<TAB>to<TAB>from<TAB>reason<TAB>declared_utc<TAB>n_history`, one per line (empty when none) —
    what 06 reads to derive mark-placed's --supersede + --reason for a grown group on a flag-free pass."""
    rows = []
    for gn, d in sorted(dict((man or {}).get("declared_moves") or {}).items()):      # type: ignore[union-attr]
        if isinstance(d, Mapping) and isinstance(d.get("to"), str):
            rows.append("\t".join([str(gn), str(d["to"]), str(d.get("from")), str(d.get("reason") or "").replace("\t", " ").replace("\n", " "),
                                   str(d.get("declared_utc") or ""), str(len(d.get("history") or []))]))
    return "\n".join(rows)


def placement_from_plan(model_key: str, view: str, labels: List[str], *, table: Optional[str] = None, nshards: int = 9, sha: str = "{sha}",
                        move_groups: Optional[Mapping[str, str]] = None, smoke_goals: Optional[Mapping[str, object]] = None) -> dict:
    """The manifest the plan implies over the ORDERED `labels`: every multi-turn row's (label, group) by logical_work_id (a declared move
    places its group at the destination). `smoke_goals` is the plan's (M10): the manifest pins the smoke's targets, not the paper's twelve."""
    rows = unit_plan(model_key, view, table=table, families=["multiturn"], nshards=nshards, sha=sha, clusters=list(labels), move_groups=move_groups,
                     smoke_goals=smoke_goals)
    return {"schema": 1, "model": model_key, "view": view, "labels": list(labels), "pinned_utc": utc_now(), "source": "plan",
            "units": {r["logical_work_id"]: {"cluster": r["cluster"], "group": r["group"], "kind": r.get("kind", "target")} for r in rows}}


def _frozen_tree_attempts(ledger: str, model_key: str, view: str, frozen_sha: Optional[str]) -> Dict[str, List[dict]]:
    """Each multi-turn unit's attempts AT A FROZEN TREE (sha == the inventory's code_sha, or a sha the funnel ADVANCED the unit to via its
    `sha_provenance` chain), in ledger order, with their attempt ids: {lwid: [{cluster, group, job_id, utc, sha, attempt_id}, …]}."""
    out: Dict[str, List[dict]] = {}
    prefix = f"{model_key}/{view}/multiturn/"
    prov = ledger_sha_provenance(ledger)
    def _trees(lw: str) -> set:
        return ({frozen_sha} if frozen_sha else set()) | {str(e.get("sha")) for e in prov.get(lw, []) if e.get("sha")}
    for r in _ledger_events(ledger):
        if r.get("event") != "attempt" or not str(r.get("logical_work_id") or "").startswith(prefix):
            continue
        lw = r["logical_work_id"]
        spec = r.get("spec") if isinstance(r.get("spec"), dict) else {}
        sha = str(r.get("sha") or spec.get("sha") or "")
        if not sha or sha not in _trees(lw):
            continue
        out.setdefault(lw, []).append({"cluster": r.get("cluster"), "group": spec.get("group") or r.get("group"), "job_id": str(r.get("job_id")),
                                       "utc": str(r.get("utc") or ""), "sha": sha, "attempt_id": str(r.get("attempt_id") or ""),
                                       "kind": (r.get("kind") or spec.get("kind"))})
    return out


def effective_attempts(ledger: str, model_key: str, view: str, frozen_sha: Optional[str], frozen_utc: Optional[str] = None) -> Dict[str, dict]:
    """Each multi-turn unit's EFFECTIVE attempts, selected exactly as the workload baseline is selected: the BASELINE is the unit's first
    attempt at a FROZEN TREE — sha == the inventory's code_sha, or a sha the funnel ADVANCED the unit to (its `sha_provenance` chain);
    `later` lists every subsequent attempt at such a tree, in order. Attempts that predate the freeze, or sit at a tree the unit was never
    frozen or advanced to, are provisional history and decide nothing (a non-frozen-sha post-freeze attempt must not win the
    selection, and a LATER re-run on another cluster must not be invisible).
    {lwid: {"baseline": {cluster, group, job_id, utc, sha}, "later": [{…}, …]}}. `frozen_utc` is accepted for callers that still pass it; the
    selection is by tree, never by clock. SUBMISSIONS, not results: `effective_results` filters these by evidence."""
    out: Dict[str, dict] = {}
    for lw, rows in _frozen_tree_attempts(ledger, model_key, view, frozen_sha).items():
        strip = [{k: v for k, v in a.items() if k not in ("attempt_id", "kind")} for a in rows]
        out[lw] = {"baseline": strip[0], "later": strip[1:]}
    return out


#: ledger events that EXCLUDE an attempt from being a result (24d): the election denied it (a twin's loser, or
#: `cancel-attempt`'s record, which carries `cancelled`), a same-identity re-run superseded it, or its grant was retired (a dead,
#: unregistered winner)
ATTEMPT_EXCLUDING_EVENTS = ("denied", "superseded_attempt", "grant_retired")
#: evidence that an attempt IS the unit's live generation, strongest first: a registration (ledger event or the unit's `registered`
#: binding naming it), the election's grant, the election's observation of it running, the unit's `submitted` binding naming it, or —
#: weakest — a plain attempt the ledger says nothing against (the funnel's live generation until a record says otherwise)
RESULT_EVIDENCE_ORDER = ("registered", "granted", "observed", "bound", "submitted")


def excluded_attempts(ledger: str, lwid: Optional[str] = None) -> Dict[str, Dict[str, str]]:
    """{lwid: {attempt_id: reason}} — every attempt the ledger records as denied (cancelled), superseded or grant-retired."""
    out: Dict[str, Dict[str, str]] = {}
    why = {"denied": "denied by the election", "superseded_attempt": "superseded by a same-identity re-run", "grant_retired": "its grant was retired (a dead, unregistered winner)"}
    for r in _ledger_events(ledger):
        ev, lw, aid = r.get("event"), r.get("logical_work_id"), r.get("attempt_id")
        if ev not in ATTEMPT_EXCLUDING_EVENTS or not lw or not aid or (lwid is not None and lw != lwid):
            continue
        out.setdefault(lw, {})[str(aid)] = "cancelled (cancel-attempt)" if (ev == "denied" and r.get("cancelled")) else why[ev]
    return out


def attempt_evidence_events(ledger: str) -> Dict[str, Dict[str, str]]:
    """{lwid: {attempt_id: strongest evidence}} from the ledger's `registered` / `granted` / `observed` events."""
    rank = {e: i for i, e in enumerate(RESULT_EVIDENCE_ORDER)}
    out: Dict[str, Dict[str, str]] = {}
    for r in _ledger_events(ledger):
        ev, lw, aid = r.get("event"), r.get("logical_work_id"), r.get("attempt_id")
        if ev in ("registered", "granted", "observed") and lw and aid:
            cur = out.setdefault(lw, {}).get(str(aid))
            if cur is None or rank[ev] < rank[cur]:
                out[lw][str(aid)] = ev
    return out


def read_binding_strict(ledger: str, lwid: str) -> Optional[dict]:
    """The unit's latest binding ({cluster, job_id, state, attempt_id}; `registered` over `submitted`) read FAIL-CLOSED: a torn, malformed
    or non-object binding file RAISES ValueError by name instead of reading as "no binding" (24d). None when no binding
    file exists."""
    d = bindings_dir(ledger); safe = safe_id(lwid); found = None
    for state in ("submitted", "registered"):
        p = os.path.join(d, f"{safe}.{state}.json")
        if not os.path.isfile(p):
            continue
        try:
            rec = json.load(open(p, encoding="utf-8"))
        except (json.JSONDecodeError, OSError, UnicodeDecodeError) as exc:
            raise ValueError(f"binding {p} is unreadable ({type(exc).__name__}: {str(exc)[:120]}) — fail closed: a torn or malformed binding is never read as 'no binding'") from exc
        if not isinstance(rec, dict):
            raise ValueError(f"binding {p} is unreadable (not a JSON object) — fail closed: a malformed binding is never read as 'no binding'")
        found = {"cluster": rec.get("cluster"), "job_id": str(rec.get("job_id")), "state": state, "attempt_id": rec.get("attempt_id")}
    return found


def latest_binding_cluster(ledger: str, lwid: str) -> Optional[dict]:
    """The unit's LATEST binding (submitted or registered) beside the ledger: {cluster, job_id, state, attempt_id} or None. An unreadable
    binding RAISES (ValueError, by name) — fail closed; before 24d it read as None."""
    return read_binding_strict(ledger, lwid)


def placement_receipts(ledger: str) -> Dict[str, List[dict]]:
    """{lwid: [receipt, …]} — the `placement_receipt` events the funnel's --resubmit CROSS-CLUSTER replacement writes (the unit, from, to,
    the replaced job with its sacct state, the replacing attempt and job), in ledger order. The only sanctioned way a unit moves."""
    out: Dict[str, List[dict]] = {}
    for r in _ledger_events(ledger):
        if r.get("event") == "placement_receipt" and r.get("logical_work_id"):
            out.setdefault(r["logical_work_id"], []).append({k: r.get(k) for k in ("from_cluster", "to_cluster", "to_group", "replaced_job_id", "replaced_state",
                                                                                   "attempt_id", "job_id", "utc", "reason")})
    return out


def effective_results(ledger: str, model_key: str, view: str, frozen_sha: Optional[str], frozen_utc: Optional[str] = None) -> Dict[str, dict]:
    """Each multi-turn unit's EFFECTIVE RESULT, derived from its frozen-tree attempts with ELECTION / REGISTRATION / BINDING evidence —
    never from submissions alone (24d): an attempt the ledger records as denied (a twin's loser, `cancel-attempt`),
    superseded by a same-identity re-run or grant-retired is EXCLUDED; a surviving attempt's evidence is the strongest of
    `RESULT_EVIDENCE_ORDER` (a `registered` ledger event or binding naming it, `granted`, `observed`, the `submitted` binding naming it
    (`bound`), else `submitted`). {lwid: {cluster, group, job_id, attempt_id, utc, sha, evidence, results: [every surviving attempt, in
    order], excluded: [{…, reason}, …]}} — the top-level fields are the LATEST surviving attempt's; a unit with none is absent. An
    unreadable binding is not evidence here; the placement reads refuse it by name themselves."""
    excl = excluded_attempts(ledger); evid = attempt_evidence_events(ledger)
    rank = {e: i for i, e in enumerate(RESULT_EVIDENCE_ORDER)}
    out: Dict[str, dict] = {}
    for lw, rows in _frozen_tree_attempts(ledger, model_key, view, frozen_sha).items():
        try:
            b = read_binding_strict(ledger, lw)
        except ValueError:
            b = None
        results: List[dict] = []; excluded: List[dict] = []
        for a in rows:
            aid = a["attempt_id"]
            if aid in excl.get(lw, {}):
                excluded.append({**a, "reason": excl[lw][aid]}); continue
            ev = evid.get(lw, {}).get(aid)
            if b is not None and str(b.get("attempt_id")) == aid:
                bev = "registered" if b["state"] == "registered" else "bound"
                ev = bev if ev is None or rank[bev] < rank[ev] else ev
            results.append({**a, "evidence": ev or "submitted"})
        if results:
            out[lw] = {**results[-1], "results": results, "excluded": excluded}
    return out


def _reconcile_results(seed_units: Mapping[str, dict], results: Mapping[str, dict], labels: List[str], ledger: Optional[str],
                       receipts: Mapping[str, List[dict]], man: Optional[Mapping[str, object]] = None) -> Tuple[List[str], Dict[str, dict]]:
    """(problems, advanced): every unit's SURVIVING results (`effective_results`) against its pinned assignment, each unit's assignment
    advancing under a placement RECEIPT that names one of its attempts (the funnel's --resubmit cross-cluster replacement): an
    attempt the receipt names must sit on the receipt's cluster; any other attempt must sit on the assignment then in force, else it is
    an UNRECEIPTED move, by name. With a ledger, the unit's latest binding (read fail-closed)
    must sit on its current assignment unless the ledger excludes that binding's attempt (a denied twin, a cancelled never-started
    attempt). `advanced` = {lwid: {cluster, group, receipt, receipted_utc, replaced_job_id, replaced_state}} for the units a receipt moved."""
    probs: List[str] = []; advanced: Dict[str, dict] = {}
    for lw, e in sorted(results.items()):
        p = seed_units.get(lw)
        rows = list(e.get("results") or [e])
        if p is None:
            base = rows[0]
            try:
                member = is_pool_member(lw, {"kind": base.get("kind")} if base.get("kind") is not None else None)
            except ValueError as exc:
                probs.append(str(exc)); continue
            if member:
                # pool SIZING, not placement (31c): the member ran on its group's cluster or it is a problem; the plan's count is not
                for a in rows:
                    if a["cluster"] not in labels:
                        probs.append(f"{lw} ran on {a['cluster']} (job {a['job_id']}, {a.get('utc')}), not one of the site's clusters {list(labels)}")
                    elif a.get("group") and not str(a["group"]).endswith(f"-{a['cluster']}"):
                        probs.append(f"{lw}: pool member of group {a.get('group')} ran on {a['cluster']} (job {a['job_id']}, {a.get('utc')}) — a member serves its group's cluster")
                continue
            probs.append(f"{lw} ran on {base['cluster']} (job {base['job_id']}) but the plan over {list(labels)} places no such unit")
            continue
        cur = {"cluster": p["cluster"], "group": p["group"]}
        by_attempt = {str(r.get("attempt_id")): r for r in receipts.get(lw, []) if r.get("attempt_id")}
        for i, a in enumerate(rows):
            tag = "baseline" if i == 0 else "later"
            rc = by_attempt.get(str(a.get("attempt_id")))
            if rc is not None:
                if rc.get("to_cluster") != a["cluster"]:
                    probs.append(f"{lw}: the placement receipt for attempt {a.get('attempt_id')} names {rc.get('to_cluster')} but the attempt ran on {a['cluster']} "
                                 f"(job {a['job_id']}, {a.get('utc')}) — the receipt is not this attempt's")
                    continue
                cur = {"cluster": rc.get("to_cluster"), "group": rc.get("to_group") or a.get("group") or cur["group"]}
                advanced[lw] = {**cur, "receipt": a.get("attempt_id"), "receipted_utc": rc.get("utc"), "replaced_job_id": rc.get("replaced_job_id"), "replaced_state": rc.get("replaced_state")}
                continue
            if a["cluster"] not in labels:
                probs.append(f"{lw} ran on {a['cluster']} (job {a['job_id']}, {a.get('utc')}), not one of the site's clusters {list(labels)}")
            elif declared_destination_binding(man, lw, cur, a["cluster"]) and (a.get("group") or f"mt-{(man or {}).get('view')}-{a['cluster']}") == f"mt-{(man or {}).get('view')}-{a['cluster']}":
                continue        # move-knob-4: an attempt at the group's DECLARED destination while the pin holds the origin, no receipt yet — the
                                # pin stays (it owes the results a receipt; the next fenced --resubmit writes it from the pin), nothing else is tolerated
            elif (a["cluster"], a.get("group") or cur["group"]) != (cur["cluster"], cur["group"]):
                probs.append(f"{lw}: pinned on {cur['cluster']} (group {cur['group']}) but its {tag} attempt job {a['job_id']} on {a['cluster']} (group {a.get('group')}, "
                             f"{a.get('utc')}) placed it elsewhere — the placement moved without a receipt (an unreceipted move: only the funnel's --resubmit "
                             f"cross-cluster replacement writes a placement_receipt)")
    if ledger:
        excl = excluded_attempts(ledger)
        for lw, p in sorted(seed_units.items()):
            pin = advanced.get(lw) or p
            try:
                b = latest_binding_cluster(ledger, lw)
            except ValueError as exc:
                probs.append(f"{lw}: its binding is unreadable — {exc}"); continue
            if b and b["cluster"] and b["cluster"] != pin["cluster"] and str(b.get("attempt_id")) not in excl.get(lw, {}):
                if declared_destination_binding(man, lw, pin, b["cluster"]):
                    continue        # move-knob-4: the declared destination's binding before its receipt (see check_placement)
                probs.append(f"{lw}: pinned on {pin['cluster']} but its latest binding (job {b['job_id']}, {b['state']}) is bound on {b['cluster']} — the results are not where the pin says")
    return probs, advanced


def _reconcile_placement(seed_units: Mapping[str, dict], attempts: Mapping[str, dict], labels: List[str], ledger: Optional[str] = None) -> List[str]:
    """The refusals, by name, of `effective_attempts`-shaped input (baseline + later) against the pinned assignments — kept for callers
    that hold attempts rather than results; receipt-free. `_reconcile_results` is the 24d reader."""
    results = {}
    for lw, e in attempts.items():
        rows = ([e["baseline"]] + list(e.get("later") or [])) if isinstance(e, dict) and "baseline" in e else [e]
        results[lw] = {**rows[-1], "results": rows, "excluded": []}
    return _reconcile_results(seed_units, results, labels, ledger, {})[0]

def placement_from_ledger(ledger: str, model_key: str, view: str, labels: List[str], *, table: Optional[str] = None, nshards: int = 9,
                          sha: str = "{sha}", smoke_goals: Optional[Mapping[str, object]] = None) -> Tuple[Optional[dict], List[str]]:
    """(manifest, problems): the wave's placement for a wave that predates the manifest — SEEDED from the FULL planned universe (every
    multi-turn unit the plan places over the ordered `labels`), then every unit's EFFECTIVE attempt (`effective_attempts`: at or after the
    freeze, never pre-freeze history) reconciled against the seed: a unit that ran on a cluster other than its pinned one, or under
    another group, is a problem by name, and nothing is persisted while one stands.
    None when the ledger records no effective multi-turn attempt of this view (a fresh wave pins from the plan)."""
    seed = placement_from_plan(model_key, view, labels, table=table, nshards=nshards, sha=sha, smoke_goals=smoke_goals)
    inv = load_inventory(ledger) if os.path.isfile(inventory_path(ledger)) else None
    sha_f, utc_f = (inv or {}).get("code_sha"), (inv or {}).get("frozen_utc")
    atts = effective_attempts(ledger, model_key, view, sha_f, utc_f)
    if not atts:
        return None, []
    # RESULTS, not submissions (24d): denied / cancelled / superseded attempts decide nothing; a receipted cross-cluster replacement
    # advances the unit's pin under its receipt
    probs, advanced = _reconcile_results(seed["units"], effective_results(ledger, model_key, view, sha_f, utc_f), labels, ledger, placement_receipts(ledger), seed)
    units = {lw: dict(u) for lw, u in seed["units"].items()}
    for lw, adv in advanced.items():
        units[lw] = {**units[lw], **adv}
    man = {**seed, "units": units, "source": "ledger", "reconciled_attempts": len(atts), "unattempted": sorted(set(seed["units"]) - set(atts))}
    return man, probs


def _prior_destinations(man: Optional[Mapping[str, object]], gn: str, origin: Optional[str]) -> set:
    """The labels a pinned target of shard group `gn` may still hold under a declared move: the plan's origin and every destination the
    group's declaration history recorded — a re-pointed group's pins hold the PREVIOUS destination until a placement receipt advances them
    (move-knob-2). A pin already at the new destination is simply equal to the plan's and never reaches this check."""
    d = ((man or {}).get("declared_moves") or {}).get(gn) or {}      # type: ignore[union-attr]
    out = {origin, d.get("from"), d.get("to")} | {h.get("to") for h in (d.get("history") or []) if isinstance(h, Mapping)}
    return {x for x in out if x}


def declared_destination_binding(man: Optional[Mapping[str, object]], lw: str, pin: Mapping[str, object], bound_cluster: Optional[str]) -> bool:
    """move-knob-4: the OTHER pre-receipt shape of a declared move — the unit's pin still holds the move's ORIGIN (or an earlier declared
    destination) while its latest binding sits at the DECLARED destination and no placement receipt has advanced the pin yet. It arises when a
    group is declared moved before anything ran on the origin and the first destination placement is a plain submission (never a cross-cluster
    --resubmit, the only writer of a receipt): the results ARE where the declaration says they go, the pin owes them a receipt, and the next
    fenced --resubmit on the destination writes it (submit_twin: the replaced attempt's cluster OR the pin differing from the label). Nothing but
    this group's declared destination is tolerated; an undeclared cross-cluster binding still refuses by name."""
    gn = _group_of_target(lw)
    if gn is None or pin.get("receipt") or not bound_cluster:
        return False
    to = declared_move_map(man).get(gn)
    return bool(to) and bound_cluster == to and str(pin.get("cluster")) != to and str(pin.get("cluster")) in _prior_destinations(man, gn, None)


def hold_bound_origin(ledger: Optional[str], units: Dict[str, dict], moves: Mapping[str, str], seed_plain_units: Mapping[str, Mapping[str, object]],
                      man: Optional[Mapping[str, object]], view: str) -> Tuple[List[str], List[str]]:
    """move-knob-3: a DECLARED move's targets pinned at the move's DESTINATION while their LATEST BINDING still sits on the origin (or an
    earlier declared destination) and no placement receipt has moved them — the pin RETREATS to the binding's cluster (its group follows:
    `mt-<view>-<cluster>`), because a pin says where the results ARE and the designed cross-cluster flow moves a unit only through the fenced
    resubmit's receipt (24d). Returns (retreated unit ids, problems); a torn or malformed binding is a problem by name (fail closed), never
    read as "no binding". A group nothing was ever bound to keeps its destination pin (a fresh wave owes no receipt)."""
    retreated: List[str] = []; probs: List[str] = []
    if not ledger or not moves:
        return retreated, probs
    for lw, u in sorted(units.items()):
        gn = _group_of_target(lw)
        if gn is None or gn not in moves or u.get("receipt") or u.get("cluster") != moves[gn]:
            continue
        try:
            b = latest_binding_cluster(ledger, lw)
        except ValueError as exc:            # fail CLOSED (24d): a torn or malformed binding is refused by name, never read as "no binding"
            probs.append(f"{lw}: its binding is unreadable — {exc}"); continue
        origin = (seed_plain_units.get(lw) or {}).get("cluster")
        if b and b.get("cluster") and b["cluster"] != moves[gn] and b["cluster"] in _prior_destinations(man, gn, origin):
            units[lw] = {**u, "cluster": b["cluster"], "group": f"mt-{view}-{b['cluster']}"}
            retreated.append(lw)
    return retreated, probs


def expand_placement(man: dict, ledger: str, model_key: str, view: str, labels: List[str], *, table: Optional[str] = None, nshards: int = 9,
                     sha: str = "{sha}", move_groups: Optional[Mapping[str, str]] = None, re_move: bool = False,
                     smoke_goals: Optional[Mapping[str, object]] = None) -> Tuple[dict, List[str], List[str]]:
    """VALIDATE an existing manifest against the full frozen plan and the effective attempts, and safely EXPAND it: (manifest, added units,
    problems) — the returned manifest also carries every assignment a placement RECEIPT advanced, so the caller persists on ANY difference
    from the stored one, never only on added units. The label list must be the manifest's; every planned unit missing from the
    manifest is added with the plan's assignment;
    a unit already pinned keeps its pin when the plan and its effective attempt agree and is a problem by name when either disagrees
    (a partial manifest persisted from one cluster's attempts refused the other cluster's group forever)."""
    probs: List[str] = []
    if list(labels) != list(man.get("labels") or []):
        probs.append(f"the site's cluster list {list(labels)} is not the wave's pinned placement universe {list(man.get('labels') or [])} (order and membership "
                     f"both bind: the multi-turn targets are placed round-robin over it) — restore the site config or re-freeze the wave; the manifest is never expanded under another order")
        return man, [], probs
    # `re_move`: the caller's explicit re-point (`ensure --re-move`) — the merge here is the same decision, never a second refusal
    moves = merge_move_groups(declared_move_map(man), dict(move_groups or {}), re_move=re_move)
    seed = placement_from_plan(model_key, view, labels, table=table, nshards=nshards, sha=sha, move_groups=moves, smoke_goals=smoke_goals)
    seed_plain = placement_from_plan(model_key, view, labels, table=table, nshards=nshards, sha=sha, smoke_goals=smoke_goals) if moves else seed
    inv = load_inventory(ledger) if (ledger and os.path.isfile(inventory_path(ledger))) else None
    sha_f, utc_f = (inv or {}).get("code_sha"), (inv or {}).get("frozen_utc")
    results = effective_results(ledger, model_key, view, sha_f, utc_f) if ledger else {}       # no attempts file -> {}
    receipts = placement_receipts(ledger) if ledger else {}
    units = {lw: dict(u) for lw, u in (man.get("units") or {}).items()}
    for lw, u in sorted(units.items()):
        try:
            is_pool_member(lw, u)       # a pinned unit whose id and kind disagree is refused by name, wherever the plan places it (31b, fail closed)
        except ValueError as exc:
            probs.append(str(exc))
    added: List[str] = []
    for lw, p in seed["units"].items():
        if lw not in units:
            units[lw] = dict(p); added.append(lw)
        elif units[lw].get("receipt"):
            pass        # a RECEIPTED pin is the receipt's assignment, not the plan's (24d): its presence in the plan is all the plan owes it
        elif (units[lw].get("cluster"), units[lw].get("group")) != (p["cluster"], p["group"]):
            gn = _group_of_target(lw)
            if gn is not None and moves.get(gn) == p["cluster"] and units[lw].get("cluster") in _prior_destinations(man, gn, seed_plain["units"].get(lw, {}).get("cluster")):
                continue        # the DECLARED move (or its re-point): the plan places the group at its destination while the pin still holds the
                                # origin or an EARLIER declared destination, awaiting the receipt (a re-moved group's pins hold the previous one)
            probs.append(f"{lw}: the manifest pins {units[lw].get('cluster')} (group {units[lw].get('group')}) but the plan over {list(labels)} places it on {p['cluster']} "
                         f"(group {p['group']}) — the pin is not this plan's")
    # move-knob-3: a moved group's pins at the destination while the units are still BOUND on the origin retreat to the origin (the receipt
    # moves them); the caller persists the difference
    retreated, rp0 = hold_bound_origin(ledger, units, moves, seed_plain["units"], man, view)
    probs += rp0
    for lw in sorted(set(units) - set(seed["units"])):
        try:
            if is_pool_member(lw, units[lw]):
                continue        # a recorded pool member today's sizing does not place: kept as the wave's record, never a refusal (31c)
        except ValueError as exc:
            probs.append(str(exc)); continue
        probs.append(f"{lw}: pinned but the plan over {list(labels)} places no such unit")
    # every unit's timeline starts at the PLAN's assignment (the original pin) and advances under the ledger's receipts; a receipt the
    # manifest carries that the ledger does not record is refused by name
    start = {lw: (seed_plain["units"].get(lw) or {"cluster": u.get("cluster"), "group": u.get("group")}) for lw, u in units.items()}
    rp, advanced = _reconcile_results(start, results, labels, ledger, receipts, man)
    probs += rp
    for lw, adv in advanced.items():
        units[lw] = {**units[lw], **adv}
    for lw, u in sorted(units.items()):
        if u.get("receipt") and lw not in advanced:
            probs.append(f"{lw}: the manifest pins a receipted assignment ({u['receipt']} -> {u.get('cluster')}) the attempts ledger does not record — the manifest is not this wave's")
    out = dict(man); out["units"] = units
    if added:
        out["expanded_utc"] = utc_now(); out["added_units"] = sorted(added); out["source"] = f"{man.get('source')}+expanded"
    if retreated:
        out["retreated_utc"] = utc_now(); out["retreated_units"] = sorted(retreated)
    return out, sorted(added), probs


def read_placement(path: str) -> Optional[dict]:
    if not os.path.isfile(path):
        return None
    doc = json.load(open(path, encoding="utf-8"))
    if not isinstance(doc, dict) or doc.get("schema") != 1 or not isinstance(doc.get("labels"), list) or not isinstance(doc.get("units"), dict):
        raise ValueError(f"{path}: not a placement manifest (schema 1 with `labels` and `units`)")
    return doc


def check_placement(manifest: dict, labels_now: List[str], rows: List[dict], ledger: Optional[str] = None) -> List[str]:
    """The refusals, by name: the site's cluster list differs from the pinned universe (order AND membership); a planned multi-turn row's
    label or group differs from the pinned one; a pinned multi-turn unit the plan no longer carries; and — with a ledger — a pinned unit
    whose LATEST binding sits on another cluster ("placement OK" means the results ARE where the pin says) — the
    binding read fail-closed (an unreadable one refuses by name) and skipped when the ledger EXCLUDES its attempt (denied, cancelled,
    superseded); a pin carrying a placement RECEIPT is the receipt's, not the plan's. Pool members (`attackers/<i>`)
    are RECORDED, not pinned against today's sizing: a member the plan adds or no longer sizes is neither grown nor lost."""
    probs: List[str] = []
    if ledger:
        excl = excluded_attempts(ledger)
        for lw, p in sorted((manifest.get("units") or {}).items()):
            try:
                b = latest_binding_cluster(ledger, lw)
            except ValueError as exc:            # fail CLOSED (24d): a torn or malformed binding is refused by name, never read as "no binding"
                probs.append(f"{lw}: its binding is unreadable — {exc}"); continue
            if b and b["cluster"] and b["cluster"] != p["cluster"] and str(b.get("attempt_id")) not in excl.get(lw, {}):
                gn = _group_of_target(lw)
                if (gn is not None and not p.get("receipt") and declared_move_map(manifest).get(gn) == p["cluster"]
                        and b["cluster"] in _prior_destinations(manifest, gn, None)):
                    continue        # move-knob-3: the DECLARED move before its receipt — bound on the move's origin (or an earlier declared
                                    # destination), pinned at the destination; ONLY this group's declared origins, every other cross-cluster binding refuses
                if declared_destination_binding(manifest, lw, p, b["cluster"]):
                    continue        # move-knob-4: pinned at the origin, bound at the DECLARED destination, no receipt yet — the resubmit writes it
                probs.append(f"{lw}: pinned on {p['cluster']} but its latest binding (job {b['job_id']}, {b['state']}) is bound on {b['cluster']} — the results are not where the pin says")
    if list(labels_now) != list(manifest["labels"]):
        probs.append(f"the site's cluster list {list(labels_now)} is not the wave's pinned placement universe {list(manifest['labels'])} (order and membership "
                     f"both bind: the multi-turn targets are placed round-robin over it) — restore the site config or re-freeze the wave")
    pinned = manifest["units"]
    planned = {r["logical_work_id"]: r for r in rows if r.get("family") == "multiturn"}
    for lw in sorted(set(pinned) | set(planned)):
        p, r = pinned.get(lw), planned.get(lw)
        try:
            member = is_pool_member(lw, p or r)
            if p is not None and r is not None:
                is_pool_member(lw, r)       # the PLAN row's kind must agree with the id too, not only the pin's (a member row calling itself a target)
        except ValueError as exc:
            probs.append(str(exc)); continue
        if (p is None or r is None) and member:
            continue        # pool SIZING, not placement (31c): a member the plan adds or no longer sizes is not a lost or grown unit
        if p is None:
            probs.append(f"{lw}: planned but not in the pinned placement (the plan grew a multi-turn unit the wave never placed)")
        elif r is None:
            probs.append(f"{lw}: pinned on {p['cluster']} but not planned (the plan lost a multi-turn unit of the wave)")
        elif p.get("receipt"):
            continue        # a RECEIPTED pin is the receipt's assignment, not the plan's (24d); the plan owes it only the unit's existence
        elif (r["cluster"], r["group"]) != (p["cluster"], p["group"]):
            gn = _group_of_target(lw)
            if gn is not None and declared_move_map(manifest).get(gn) == r["cluster"] and r["group"] == f"mt-{manifest.get('view')}-{r['cluster']}":
                continue        # the DECLARED move (manifest `declared_moves`): planned at the destination, pinned at the origin until the resubmit's receipt
            probs.append(f"{lw}: pinned on {p['cluster']} (group {p['group']}) but planned on {r['cluster']} (group {r['group']}) — the placement moved "
                         f"(a move is declared with `placement-manifest ensure --move-group {gn or '<group>'}=<label> --move-reason …`, never implied)")
    return probs


def cmd_placement(a) -> int:
    """`placement-manifest ensure|check`: ensure = read the manifest, else pin it (from the recorded attempts when the ledger has any of
    this view's multi-turn units, else from the plan over the ordered labels) and write it; check = refuse a cluster list or a planned
    assignment that differs from the pinned one, by name. `--dry-run` never writes."""
    labels = [c for c in a.clusters.split(",") if c]
    try:
        path = a.manifest or placement_path(a.ledger, a.view)
        man = read_placement(path)
        _check_manifest_view(man, a.view, path)
        # M10: the plan the manifest is pinned from / checked against carries the smoke goal-shard knob of the model's overrides document
        # (`--overrides`, default $SISTEER_RUN_OVERRIDES), the same reading `plan` and `freeze-inventory` make
        smoke = smoke_goals(getattr(a, "overrides", None), a.model)
    except ValueError as exc:
        print(f"REFUSED: placement-manifest: {exc}", file=sys.stderr); return 2
    if a.cmd_placement == "moves":
        tsv = declared_moves_tsv(man)
        if tsv:
            print(tsv)
        return 0
    try:
        cli_moves = parse_move_groups(getattr(a, "move_group", None) or [])
        declared_now = declared_move_map(man)
        re_move = bool(getattr(a, "re_move", False))
        if re_move:
            undeclared = sorted(gn for gn in cli_moves if gn not in declared_now)
            if undeclared or not cli_moves:
                raise ValueError(f"--re-move names {undeclared or 'no group'} with no declaration to re-point — a first move is `--move-group` alone")
        eff_moves = merge_move_groups(declared_now, cli_moves, re_move=re_move)
        new_moves = {gn: lab for gn, lab in cli_moves.items() if declared_now.get(gn) != lab}
        previous = {gn: dict((man or {}).get("declared_moves", {}).get(gn) or {}) for gn in new_moves if gn in declared_now}   # type: ignore[union-attr]
        new_decl = declare_moves(a.model, a.view, labels, new_moves, getattr(a, "move_reason", None) or "", table=a.models_table, nshards=a.nshards,
                                 sha=a.sha or "{sha}", previous=previous, smoke_goals=smoke) if new_moves else {}
    except ValueError as exc:
        print(f"REFUSED: placement-manifest: {exc}", file=sys.stderr); return 2
    if new_decl and a.cmd_placement != "ensure":
        print("REFUSED: placement-manifest check: a move is DECLARED with `ensure --move-group … --move-reason …`; check only reads it", file=sys.stderr); return 2
    if a.cmd_placement == "ensure":
        if man is not None:
            # an existing manifest is VALIDATED against the full frozen plan + the effective attempts and EXPANDED with any planned unit it
            # lacks (a partial manifest persisted from one cluster's attempts must not refuse the other cluster's group)
            new_man, added, probs = expand_placement(man, a.ledger or path, a.model, a.view, labels, table=a.models_table, nshards=a.nshards, sha=a.sha or "{sha}",
                                                     move_groups=eff_moves or None, re_move=re_move, smoke_goals=smoke)
            if probs:
                print("REFUSED: placement-manifest: " + "; ".join(probs), file=sys.stderr); return 2
            if new_decl:
                new_man["declared_moves"] = {**dict(new_man.get("declared_moves") or {}), **new_decl}
            # persist on ANY difference: units the plan added AND units a placement RECEIPT advanced (a frozen full manifest + a fenced
            # cross-cluster --resubmit computed the receipted pin in memory, discarded it because nothing was added, and the following
            # `check` reloaded the stale pin and refused the valid binding — recovery blocked until a hand repair)
            advanced = sorted(lw for lw, u in new_man["units"].items() if lw in (man.get("units") or {}) and
                              {k: u.get(k) for k in ("cluster", "group", "receipt")} != {k: (man["units"].get(lw) or {}).get(k) for k in ("cluster", "group", "receipt")})
            if added or advanced or new_decl or new_man.get("units") != man.get("units"):
                retreated_u = list(new_man.get("retreated_units") or [])
                what = "; ".join(x for x in ((f"{len(added)} planned unit(s) added from the plan" if added else ""),
                                              (f"{len(retreated_u)} moved unit(s) RETREATED to their origin pin (bound there; the resubmit's receipt moves them)" if retreated_u else ""),
                                              (f"{len(new_decl)} group move(s) DECLARED ({', '.join(f'{gn} -> {d['to']}' for gn, d in sorted(new_decl.items()))}; reason: {getattr(a, 'move_reason', '')})" if new_decl else ""),
                                              (f"{len(advanced)} unit(s) advanced under a placement receipt ({', '.join(lw.split('/', 2)[2] + ' -> ' + str(new_man['units'][lw].get('cluster')) for lw in advanced)})" if advanced else "")) if x) \
                       or "unit assignments changed"
                if a.dry_run:
                    print(f"DRY-RUN: rewrite the placement manifest: {what} -> {path}"); return 0
                if advanced:
                    new_man["advanced_utc"] = utc_now(); new_man["advanced_units"] = advanced; new_man["source"] = f"{new_man.get('source')}+receipted"
                if new_decl:
                    new_man["moves_declared_utc"] = utc_now(); new_man["source"] = f"{new_man.get('source')}+moved"
                write_atomic(path, new_man)
                print(f"placement manifest {'EXPANDED' if added else ('MOVED' if new_decl and not advanced else 'ADVANCED')}: {what} ({len(new_man['units'])} multi-turn units over {labels}) -> {path}")
                return 0
        if man is None:
            if a.ledger and os.path.isfile(a.ledger):
                man, probs = placement_from_ledger(a.ledger, a.model, a.view, labels, table=a.models_table, nshards=a.nshards, sha=a.sha or "{sha}", smoke_goals=smoke)
                if probs:
                    print("REFUSED: placement-manifest: " + "; ".join(probs), file=sys.stderr); return 2
                if man is not None and new_decl:
                    # move-knob-3: a first manifest seeded from the ledger records the declaration too — the pins are the
                    # attempts' (the origin), the move stands for every later plan, the receipt moves the units
                    man["declared_moves"] = dict(new_decl); man["moves_declared_utc"] = utc_now()
            if man is None:
                # a fresh pin WITH a declared move pins the group straight at its destination (nothing ran yet, so no receipt is owed) and still
                # records the declaration — the reason travels with the manifest
                man = placement_from_plan(a.model, a.view, labels, table=a.models_table, nshards=a.nshards, sha=a.sha or "{sha}", move_groups=eff_moves or None,
                                          smoke_goals=smoke)
                if new_decl:
                    man["declared_moves"] = dict(new_decl); man["moves_declared_utc"] = utc_now()
                if eff_moves and a.ledger and os.path.isfile(a.ledger):
                    # move-knob-3: "nothing ran yet" is a claim the BINDINGS decide, not effective_attempts (which selects by the frozen tree
                    # and sees no attempt of a view that was never frozen): a moved group whose units are bound on the origin holds the
                    # origin pin until the fenced resubmit's receipt moves it
                    plain = placement_from_plan(a.model, a.view, labels, table=a.models_table, nshards=a.nshards, sha=a.sha or "{sha}", smoke_goals=smoke)
                    held, hp = hold_bound_origin(a.ledger, man["units"], eff_moves, plain["units"], man, a.view)
                    if hp:
                        print("REFUSED: placement-manifest: " + "; ".join(hp), file=sys.stderr); return 2
                    if held:
                        man["held_origin_units"] = sorted(held)
            if a.dry_run:
                print(f"DRY-RUN: pin the placement manifest ({man['source']}: {len(man['units'])} multi-turn units over {labels}) -> {path}"); return 0
            write_atomic(path, man)
            extra = f" ({man.get('reconciled_attempts')} attempted unit(s) reconciled; {len(man.get('unattempted') or [])} not yet attempted)" if man["source"] == "ledger" else ""
            held_note = (f"; {len(man['held_origin_units'])} moved unit(s) hold their ORIGIN pin, bound there, until the resubmit's receipt"
                         if man.get("held_origin_units") else "")
            print(f"placement manifest pinned from the {man['source']}: {len(man['units'])} multi-turn units over {labels}{extra}{held_note} -> {path}")
        else:
            print(f"placement manifest: {len(man['units'])} multi-turn units over {man['labels']} ({man.get('source')}, pinned {man.get('pinned_utc')}) at {path}")
        return 0
    if man is None:
        if a.dry_run:
            print(f"DRY-RUN: no placement manifest at {path} yet — the real run pins it (ensure) before this check"); return 0
        print(f"REFUSED: placement-manifest check: no manifest at {path} (run `ensure` first)", file=sys.stderr); return 2
    rows = json.load(open(a.plan_json)) if a.plan_json else unit_plan(a.model, a.view, table=a.models_table, families=["multiturn"], nshards=a.nshards,
                                                                       sha=a.sha or "{sha}", clusters=labels, move_groups=eff_moves or None, smoke_goals=smoke)
    probs = check_placement(man, labels, rows, ledger=a.ledger)
    for p in probs:
        print(f"REFUSED: placement: {p}", file=sys.stderr)
    if not probs:
        print(f"placement OK: the site's clusters {labels} are the pinned universe; {sum(1 for r in rows if r.get('family') == 'multiturn')} planned multi-turn rows match")
    return 2 if probs else 0


# ---------------------------------------------------------------------------------------------------- the bindings
def bindings_dir(ledger: str) -> str:
    return os.path.join(os.path.dirname(os.path.abspath(ledger)), "bindings")


# ------------------------------------------------------------------------------------ per-unit sha provenance
# The inventory is frozen ONCE; its identity is the unit set and each unit's EVAL CONTRACT (the instrument, the decode-policy ref, the
# item set, the evals declared, the shards, the objects/windows the row carries). A tree sha is PROVENANCE, not identity: a unit's
# recorded sha is the sha of its first accepted attempt and advances ONLY through the funnel's re-run path, which records (unit, new sha,
# the death it answers, the fix commit) here and verifies the contract is identical at the new tree — a contract change is a refusal by
# name, never a silent re-freeze. Admission compares a unit's registered runs' code_sha to THAT unit's recorded sha. Completed work is
# canonical: units complete at an older sha are never re-run because a tree changed.
PROVENANCE_FILE = "expected_units.provenance.json"


def inventory_path(ledger: str) -> str:
    return os.path.join(os.path.dirname(os.path.abspath(ledger)), "gate", "expected_units.json")


def provenance_path(ledger: str) -> str:
    return os.path.join(os.path.dirname(os.path.abspath(ledger)), "gate", PROVENANCE_FILE)


def load_inventory(ledger: str) -> Optional[dict]:
    p = inventory_path(ledger)
    if not os.path.isfile(p):
        return None
    try:
        return json.load(open(p, encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def inventory_unit(inv: Optional[dict], lwid: str) -> Optional[dict]:
    for u in (inv or {}).get("units", []):
        if u.get("logical_work_id") == lwid:
            return u
    return None


def unit_contract(u: dict) -> dict:
    """The unit's EVAL CONTRACT — what admission identifies a unit by across trees: per eval the instrument, the decode-policy ref, the
    thinking mode, the shards, the coverage layout and item set; the model, the family, the script/array/padding the row carries.
    Never the code sha, the sbatch name, the freeze time, the instrument_id_source note — nor the `held_out` stamp (a held-out eval
    stays in the contract; the gate stops requiring it)."""
    evals = {}
    for e, ev in sorted((u.get("evals") or {}).items()):
        cov = ev.get("coverage") or {}
        evals[e] = {"instrument_id": ev.get("instrument_id"), "decode_policy_ref": ev.get("decode_policy_ref"), "thinking_mode": ev.get("thinking_mode"),
                    "expected_shards": list(ev.get("expected_shards") or []), "layout": cov.get("layout"), "item_set_id": cov.get("item_set_id"),
                    "items_sha256": cov.get("items_sha256"), "n_items": cov.get("n_items")}
    return {"logical_work_id": u.get("logical_work_id"), "model_id": u.get("model_id"), "family": u.get("family"), "script": u.get("script"),
            "array": u.get("array"), "padding_tokens": u.get("padding_tokens"), "private_run": u.get("private_run"), "evals": evals}


def contract_sha256(u: dict) -> str:
    return hashlib.sha256(json.dumps(unit_contract(u), sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def contract_diff(frozen: dict, current: dict) -> List[str]:
    """The contract fields that differ between two contract dicts, by name (`evals.<e>.<field>: frozen -> current`)."""
    out: List[str] = []
    for k in sorted(set(frozen) | set(current)):
        if k == "evals":
            fe, ce = frozen.get("evals") or {}, current.get("evals") or {}
            for e in sorted(set(fe) | set(ce)):
                if e not in fe or e not in ce:
                    out.append(f"evals.{e}: {'present' if e in fe else 'absent'} -> {'present' if e in ce else 'absent'}"); continue
                for f in sorted(set(fe[e]) | set(ce[e])):
                    if fe[e].get(f) != ce[e].get(f):
                        out.append(f"evals.{e}.{f}: {fe[e].get(f)!r} -> {ce[e].get(f)!r}")
        elif frozen.get(k) != current.get(k):
            out.append(f"{k}: {frozen.get(k)!r} -> {current.get(k)!r}")
    return out


PROVENANCE_RULE = ("D200: per-unit sha provenance — a unit's sha advances only through the funnel's re-run path; admission compares its runs' "
                   "code_sha to the unit's recorded sha and its contract hash to the frozen contract (or to the latest RECEIPTED AMENDMENT of it, D217)")


def read_provenance_doc(ledger: str) -> dict:
    p = provenance_path(ledger)
    if not os.path.isfile(p):
        return {"schema": 1, "rule": PROVENANCE_RULE, "units": {}, "amendments": {}}
    try:
        d = json.load(open(p, encoding="utf-8"))
    except json.JSONDecodeError:
        return {"schema": 1, "rule": PROVENANCE_RULE, "units": {}, "amendments": {}}
    d.setdefault("units", {}); d.setdefault("amendments", {}); d["rule"] = PROVENANCE_RULE
    return d


def read_provenance(ledger: str) -> Dict[str, List[dict]]:
    return {k: list(v) for k, v in (read_provenance_doc(ledger).get("units") or {}).items()}


def read_amendments(ledger: str) -> Dict[str, List[dict]]:
    return {k: list(v) for k, v in (read_provenance_doc(ledger).get("amendments") or {}).items()}


def record_sha_provenance(ledger: str, lwid: str, entry: dict) -> dict:
    """Append one provenance entry for `lwid` — {sha, utc, kind first|advance|rerun, attempt_id, job_id, cluster, answers, fix_commit,
    contract_sha256} — atomically; the file is the wave's per-unit sha table (every other section of the file is kept)."""
    doc = read_provenance_doc(ledger)
    rec = {"sha": entry["sha"], "utc": entry.get("utc") or utc_now(), "kind": entry.get("kind"), "attempt_id": entry.get("attempt_id"),
           "job_id": entry.get("job_id"), "cluster": entry.get("cluster"), "answers": entry.get("answers"), "fix_commit": entry.get("fix_commit"),
           "contract_sha256": entry.get("contract_sha256")}
    doc["units"].setdefault(lwid, []).append(rec)
    write_atomic(provenance_path(ledger), doc)
    return rec


# ------------------------------------------------------------------------------ the receipted CONTRACT AMENDMENT
#: the ONLY decode-policy keys an amendment may change, each with its ONE admissible transition: null / omitted -> the value the engine
#: resolves an omitted key to (vLLM's neutral min_p 0.0 and repetition_penalty 1.0; the OpenAI-protocol presence_penalty 0.0 —
#: contracts.GREEDY_REFERENCE_RESOLVED / check_protocol_effective). A row that stated these as "not sent" and now sends them explicitly
#: changes nothing about the sampling distribution; any OTHER value, or any other old value, is a new sampler and a new wave (a key-name
#: whitelist let `min_p: null -> 0.05` pass as an amendment). Every other row field (temperature, top_p,
#: top_k, max_tokens, the seed rule, the ref, the mode) is the contract itself
AMEND_NEUTRAL_TRANSITIONS: Dict[str, float] = {"min_p": 0.0, "repetition_penalty": 1.0, "presence_penalty": 0.0}
AMEND_ALLOWED_KEYS = tuple(AMEND_NEUTRAL_TRANSITIONS)
#: documentation fields of a decode-policy row: their change is recorded, never a refusal
AMEND_DOC_FIELDS = ("notes", "ruling_ref", "ruling_id", "same_values_as", "source_citation", "effective_from_utc", "effective_from_confidence",
                    "decode_source", "seed_decode_source")
#: a run in any of these states COMPLETED under its contract: the contract can no longer be amended for that unit
COMPLETED_RUN_STATES = ("generated", "judging", "judged", "ingested")
#: a registry run in one of these states is DEFINITIVELY not completed (sisteer.registry.vocab.LIFECYCLE_STATES' terminal, non-completed
#: states): it needs no sacct proof of its own — the job-level proof still covers the job that registered it. Every other non-completed
#: state (`queued`, `running`, anything unknown) is proven through its job (a `running` run passed the check and could reach
#: `generated` between the registry read and the ledger write)
TERMINAL_NON_COMPLETED_RUN_STATES = ("failed", "cancelled", "superseded", "withdrawn")
#: ------------------------------------------ the Petri per-request refusal fallback amendment
#: `amend-contract --kind`: `decode-neutral` is the contract amendment above (instrument ids alone, explained by one decode-policy row's
#: exact neutral transitions; every old job terminal); `petri-role-fallback` is the Petri unit's receipted sha ADVANCE for the per-request
#: refusal fallback: the drive's manifest changed by EXACTLY the two added keys below, the contract unchanged (or Petri instrument ids
#: alone), no run completed under the old contract — the serving job may be ALIVE (the drive restarts against it on the grading host), so
#: it is recorded, not required terminal — and its `running` runs are CARRIED (same run ids, same instrument: the resumed drive fills
#: them), never superseded; the receipt advances the unit's sha to the amending commit (sha_provenance kind `amend`) and the
#: wave's per-unit sha table follows (WAVE.<view>.json `unit_shas`)
#: `code-follow-up`: a CODE-ONLY follow-up to a receipted sha advance of the Petri unit — the commit is a strict descendant of
#: the unit's amended sha; the drive manifest and the roster (F22_ROSTER_REL) are unchanged; the contract and the decode-policy export are
#: unchanged; the old run set and the carried set are the prior receipt's exactly (no new old-job run) — recorded as a SECOND receipt
#: (role_diff [], `follow_up_of` the prior receipt, the --note naming the findings it carries) that advances the unit's sha again
AMEND_KINDS = ("decode-neutral", "petri-role-fallback", "code-follow-up")
#: the roster petri_drive_full22.sh hands the bank-view check (F22_ROSTER); unchanged across a code-only follow-up
F22_ROSTER_REL = "configs/sdfbase_full22_roster.yaml"
#: the drive's manifest (petri_drive_full22.sh: MANIFEST=configs/capeval/manifests/petri_full22_${TAG}.yaml, TAG = <model>_<view>)
PETRI_MANIFEST_REL_TMPL = "configs/capeval/manifests/petri_full22_{tag}.yaml"
#: the ONLY manifest changes the kind admits, under the top-level `petri:` block, each with its ONE transition (before -> after; None =
#: absent): two ADDED keys routing to Inspect's GenerateConfig.fallback_models (the Anthropic API's server-side per-request refusal
#: fallback, first-party models). The existing `auditor_fallback_models` chain is a harness-hash input (manifest.py role_models ->
#: harness_config_hash -> petri_run_key -> the cell dirs, the Inspect log_dir and the registered instrument) and must NOT move: the new
#: keys are additive, not hash inputs, so the Petri instrument id and every petri_run_key are identical at the amending commit and no
#: completed sample is lost. Anything else in the diff is a new wave, refused by name.
#: Each key admits a TUPLE of transitions — its INITIAL introduction (absent -> the Claude
#: leg; the two keys enter together, the whole per-request fallback) and, for the judge, the APPENDED TERMINAL non-Claude leg over OpenRouter
#: ([opus-4-8] -> [opus-4-8, grok-4.6-20260810]) — a JUDGE-ONLY later amendment (the auditor stands as it is), admitted by name; the Grok id is
#: the IMMUTABLE dated slug (the bare `grok-4.6` alias is mutable). Still additive, not a harness-hash input: no cell re-keys.
PETRI_ROLE_FALLBACK_TRANSITIONS: Dict[str, Tuple[Tuple[object, object], ...]] = {
    "auditor_refusal_fallback_models": ((None, ["anthropic/claude-sonnet-4-6"]),),
    "judge_refusal_fallback_models": ((None, ["anthropic/claude-opus-4-8"]),
                                      (["anthropic/claude-opus-4-8"], ["anthropic/claude-opus-4-8", "openrouter/x-ai/grok-4.6-20260810"])),
}
#: the admitted transitions that INTRODUCE a key (before = absent): these enter together or not at all
PETRI_ROLE_FALLBACK_INITIAL: Dict[str, Tuple[object, object]] = {k: next(t for t in trs if t[0] is None) for k, trs in PETRI_ROLE_FALLBACK_TRANSITIONS.items()}
#: Slurm's non-terminal states — submit_twin.ACTIVE, pinned equal by tests/test_amend_contract_enforceable.py so the two cannot drift
SACCT_ACTIVE = ("PENDING", "RUNNING", "REQUEUED", "RESIZING", "SUSPENDED", "COMPLETING", "CONFIGURING")
#: the decode-policy export and the registry seeds, relative to the repo root — read from GIT at a sha, never from a file handed in
DECODE_EXPORT_REL = "configs/registry/decode_policy.json"
SEEDS_REL = "configs/registry"


def contract_dict_sha256(c: dict) -> str:
    return hashlib.sha256(json.dumps(c, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def contract_at_tree(unit: dict, seeds_dir: Optional[str] = None) -> dict:
    """The unit's eval contract as THIS checkout's registry seeds mint it: the frozen contract with each eval's instrument id, decode-policy
    ref and (for a capability cellstore) item set replaced by the canonical instrument the seeds resolve for the unit's model. Padded
    units keep their frozen (derived) instrument ids. Raises ValueError, by name, when the planner mints no instrument for an eval."""
    cur = unit_contract(unit)
    evals = sorted(cur["evals"])
    if not evals or unit.get("family") == "padding":
        return cur
    canon = canonical_instruments(unit["model_id"], evals, seeds_dir)
    for e in evals:
        c = canon.get(e) or {}
        if "error" in c:
            raise ValueError(f"{unit['logical_work_id']}: the registry planner at this tree mints no canonical instrument for eval {e!r} on "
                             f"{unit['model_id']}: {c['error']}")
        cur["evals"][e]["instrument_id"] = c.get("instrument_id")
        cur["evals"][e]["decode_policy_ref"] = c.get("decode_policy_ref")
        if cur["evals"][e].get("layout") == "capability_cellstore":
            cur["evals"][e]["item_set_id"] = c.get("item_set_id"); cur["evals"][e]["n_items"] = c.get("n_items"); cur["evals"][e]["items_sha256"] = c.get("items_sha256")
    return cur


def effective_contract(amendments: Dict[str, List[dict]], unit: dict) -> dict:
    """The contract admission and the funnel compare against: the latest receipted amendment's contract, else the frozen one."""
    entries = amendments.get(unit.get("logical_work_id") or "", [])
    return dict(entries[-1]["contract"]) if entries and isinstance(entries[-1].get("contract"), dict) else unit_contract(unit)


def _flat(x, prefix: str = "") -> Dict[str, object]:
    out: Dict[str, object] = {}
    if isinstance(x, dict):
        for k, v in x.items():
            out.update(_flat(v, f"{prefix}.{k}" if prefix else str(k)))
    elif isinstance(x, list):
        out[prefix] = json.dumps(x, sort_keys=True)
    else:
        out[prefix] = x
    return out


def decode_row(export: dict, ref: str) -> Optional[dict]:
    """The current (non-superseded) decode-policy row of `ref` in an export document (its `policy_ref` or `policy_id`)."""
    rows = export.get("policies") or export.get("rows") or []
    cands = [r for r in rows if r.get("policy_id") == ref or r.get("policy_ref") == ref]
    live = [r for r in cands if not r.get("superseded_by_policy_id")]
    return (live or cands or [None])[-1]


def _is_neutral(value, want: float) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and float(value) == float(want)


def decode_row_diff(old_row: dict, new_row: dict) -> Tuple[List[dict], List[str]]:
    """(allowed changes [{path, old, new}], violations [path: old -> new …]) between two decode-policy rows. A change is allowed ONLY when it
    is (a) one of the three keys' EXACT neutral transition — the top-level `min_p` / `repetition_penalty` / `presence_penalty` from null or
    omitted to 0.0 / 1.0 / 0.0 (AMEND_NEUTRAL_TRANSITIONS); any other old or new value is a violation naming the field and both values —
    (b) `params_sent` whose set difference lies within those keys, (c) the same keys' `params_not_sent_json.<key>` notes (the note of a key
    now sent), or (d) a documentation field (AMEND_DOC_FIELDS). Anything else is a violation by name (the key-name whitelist this
    replaces let any numeric change to the three keys pass)."""
    fo, fn = _flat(old_row), _flat(new_row)
    allowed: List[dict] = []; violations: List[str] = []
    for path in sorted(set(fo) | set(fn)):
        if fo.get(path) == fn.get(path):
            continue
        old, new = fo.get(path), fn.get(path)
        head, last = path.split(".")[0], path.split(".")[-1]
        if path in AMEND_NEUTRAL_TRANSITIONS:
            want = AMEND_NEUTRAL_TRANSITIONS[path]
            if old is None and _is_neutral(new, want):
                allowed.append({"path": path, "old": old, "new": new}); continue
            violations.append(f"{path}: {old!r} -> {new!r} (the only admissible transition is null/omitted -> {want!r}, the value the engine "
                              f"resolves an omitted {path} to)"); continue
        if (head == "params_not_sent_json" and last in AMEND_NEUTRAL_TRANSITIONS) or head in AMEND_DOC_FIELDS:
            allowed.append({"path": path, "old": old, "new": new}); continue
        if path == "params_sent":
            try:
                so, sn = set(json.loads(old or "[]")), set(json.loads(new or "[]"))
            except (TypeError, ValueError):
                so, sn = set(), {"?"}
            if (so ^ sn) <= set(AMEND_ALLOWED_KEYS):
                allowed.append({"path": path, "old": old, "new": new}); continue
        violations.append(f"{path}: {old!r} -> {new!r}")
    return allowed, violations


# ---- git: the old export, the new export and the seeds are read from TREES, never from files handed in (High) ----------
def _git(repo: str, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", repo, *args], capture_output=True)


def git_rev(repo: str, ref: str) -> str:
    """The full sha `ref` names in `repo`, or ValueError by name."""
    r = _git(repo, "rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}")
    if r.returncode != 0 or not r.stdout.strip():
        raise ValueError(f"{ref!r} is not a commit of {repo}: {r.stderr.decode('utf-8', 'replace').strip()[:200]}")
    return r.stdout.decode("utf-8").strip()


def git_is_ancestor(repo: str, commit: str, tip: str) -> bool:
    r = _git(repo, "merge-base", "--is-ancestor", commit, tip)
    if r.returncode == 0:
        return True
    if r.returncode == 1:
        return False
    raise ValueError(f"git merge-base --is-ancestor {commit} {tip} failed in {repo}: {r.stderr.decode('utf-8', 'replace').strip()[:200]}")


def git_show_bytes(repo: str, sha: str, rel: str) -> bytes:
    r = _git(repo, "show", f"{sha}:{rel}")
    if r.returncode != 0:
        raise ValueError(f"cannot read {rel} at {sha} from {repo}: {r.stderr.decode('utf-8', 'replace').strip()[:200]}")
    return r.stdout


def seeds_at_tree(repo: str, sha: str) -> str:
    """`configs/registry` as committed at `sha`, exported (git archive) into a fresh temp dir — the seeds the NEW contract is minted from,
    so the contract the receipt carries is the amending commit's, not the working tree's. The caller removes the dir."""
    import io
    import tarfile
    import tempfile
    r = _git(repo, "archive", "--format=tar", sha, "--", SEEDS_REL)
    if r.returncode != 0 or not r.stdout:
        raise ValueError(f"cannot export {SEEDS_REL} at {sha} from {repo}: {r.stderr.decode('utf-8', 'replace').strip()[:200]}")
    tmp = tempfile.mkdtemp(prefix="amend-seeds-")
    with tarfile.open(fileobj=io.BytesIO(r.stdout)) as tf:
        tf.extractall(tmp, filter="data")
    seeds = os.path.join(tmp, SEEDS_REL)
    if not os.path.isdir(seeds):
        raise ValueError(f"{SEEDS_REL} is not a directory at {sha} in {repo}")
    return tmp


def contract_at_commit(unit: dict, repo: str, sha: str) -> dict:
    """contract_at_tree over the seeds committed at `sha` (seeds_at_tree); the export dir is removed afterwards."""
    import shutil
    tmp = seeds_at_tree(repo, sha)
    try:
        return contract_at_tree(unit, os.path.join(tmp, SEEDS_REL))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def amendment_old_sha(ledger: str, unit: dict, prior: List[dict], before_utc: Optional[str] = None) -> Tuple[str, str]:
    """(sha, source): the tree the unit's EFFECTIVE contract was minted at — the latest receipted amendment's commit when there is one, else
    the unit's recorded sha from the append-only ledger (its latest `sha_provenance` event, those before `before_utc` when given), else the
    freeze sha. Never typed: the old decode-policy export is `git show <this sha>:configs/registry/decode_policy.json`."""
    if prior:
        return str(prior[-1].get("commit_resolved") or prior[-1].get("commit")), f"amendment #{len(prior)}'s commit"
    lw = unit.get("logical_work_id") or ""
    # an amendment's OWN advance (kind `amend`) is never the old tree of an amendment: the next amendment's old tree is the prior
    # receipt's commit, which the `prior` branch above returns (a recheck must not read a receipt's advance as the tree before it)
    entries = [e for e in ledger_sha_provenance(ledger).get(lw, []) if e.get("sha") and e.get("kind") != "amend" and (before_utc is None or (e.get("utc") or "") <= before_utc)]
    if entries:
        return str(entries[-1]["sha"]), "the unit's recorded sha (ledger sha_provenance)"
    if not unit.get("code_sha"):
        raise ValueError(f"{lw}: the unit records no sha (no sha_provenance event, no code_sha) — the old decode-policy export cannot be derived")
    return str(unit["code_sha"]), "the freeze sha (no attempt recorded a sha)"


# ---- the old jobs and the old runs: every job TERMINAL on sacct, every run definitively non-completed (High) -----------
def _at_or_before(rec: dict, before_utc: Optional[str]) -> bool:
    """The cut-off a --recheck applies: a ledger event or binding stamped at or before the receipt's second is the receipt's world (an event
    with no stamp counts as old); the mint applies no cut-off."""
    return before_utc is None or (rec.get("utc") or "") <= before_utc


def unit_binding_records(ledger: str, lwid: str) -> List[dict]:
    """Every binding record of the unit beside the ledger — the current `submitted` / `registered` ones and every RETIRED registered binding
    (`<stem>.registered.retired.*.json`, the old attempt's binding a re-run moved aside) — each tagged `binding`."""
    d = bindings_dir(ledger); safe = safe_id(lwid)
    out: List[dict] = []
    if not os.path.isdir(d):
        return out
    for fn in sorted(os.listdir(d)):
        if not (fn == f"{safe}.submitted.json" or fn == f"{safe}.registered.json" or fn.startswith(f"{safe}.registered.retired.")) or not fn.endswith(".json"):
            continue
        try:
            rec = json.load(open(os.path.join(d, fn), encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if isinstance(rec, dict):
            rec = dict(rec); rec["binding"] = fn[len(safe) + 1:-len(".json")]
            out.append(rec)
    return out


def _old_bindings(ledger: str, lwid: str, before_utc: Optional[str], old_job_ids: set) -> List[dict]:
    """The unit's binding records that belong to the OLD world: all of them for a mint (no cut-off); for a --recheck those stamped at or
    before the receipt, or a retired binding, or one whose job is among the ledger's old jobs (a binding written by 06 --gate carries the
    job it bound; a job id match is unambiguous evidence)."""
    recs = unit_binding_records(ledger, lwid)
    if before_utc is None:
        return recs
    return [r for r in recs if (r.get("utc") and r["utc"] <= before_utc) or str(r.get("binding", "")).startswith("registered.retired.")
            or (r.get("job_id") is not None and str(r.get("job_id")) in old_job_ids)]


def unit_old_run_ids(ledger: str, lwid: str, before_utc: Optional[str] = None) -> List[str]:
    """Every run id the unit's attempts registered so far: the ledger's `registered` / `superseded_attempt` events and the unit's bindings
    (current and retired) — at or before `before_utc` when given (a --recheck), so a re-run under an amended contract, which registers NEW
    ids, is not the receipt's to exclude; these are the old ones the artifact must exclude."""
    ids = set()
    for r in _ledger_events(ledger):
        if r.get("logical_work_id") == lwid and r.get("run_ids") and _at_or_before(r, before_utc):
            ids.update(str(x) for x in r["run_ids"])
    old_job_ids = {j["job_id"] for j in unit_old_jobs(ledger, lwid, before_utc)} if before_utc is not None else set()
    for b in _old_bindings(ledger, lwid, before_utc, old_job_ids):
        ids.update(str(x) for x in (b.get("run_ids") or []))
    return sorted(ids)


def unit_old_jobs(ledger: str, lwid: str, before_utc: Optional[str] = None) -> List[dict]:
    """Every scheduler job the unit's attempts ever bound — [{cluster, job_id, attempt_id, utc, source}], one per (cluster, job) — from the
    append-only ledger's `attempt`, `superseded_attempt` and `sha_provenance` events and the unit's bindings (current and retired), at or
    before `before_utc` when given. Each must be proven TERMINAL on sacct before the contract is amended."""
    out: Dict[Tuple[str, str], dict] = {}
    def add(cluster, job_id, attempt_id, utc, source):
        if not job_id or not cluster:
            return
        key = (str(cluster), str(job_id))
        if key not in out:
            out[key] = {"cluster": str(cluster), "job_id": str(job_id), "attempt_id": attempt_id, "utc": utc, "source": source}
    for r in _ledger_events(ledger):
        if r.get("logical_work_id") != lwid or r.get("event") not in ("attempt", "superseded_attempt", "sha_provenance") or not _at_or_before(r, before_utc):
            continue
        add(r.get("cluster"), r.get("job_id"), r.get("attempt_id"), r.get("utc"), r["event"])
    for b in _old_bindings(ledger, lwid, before_utc, {k[1] for k in out}):
        add(b.get("cluster"), b.get("job_id"), b.get("attempt_id"), b.get("utc"), f"{b.get('binding')} binding")
    return sorted(out.values(), key=lambda j: (j["utc"] or "", j["cluster"], j["job_id"]))


def unit_run_jobs(ledger: str, lwid: str, before_utc: Optional[str] = None) -> Dict[str, dict]:
    """{run_id: {cluster, job_id}} as the ledger binds runs to jobs: `superseded_attempt` events (job + run ids), `registered` events (the
    attempt's job) and the unit's registered bindings (current and retired), at or before `before_utc` when given."""
    attempts: Dict[str, dict] = {}
    out: Dict[str, dict] = {}
    for r in _ledger_events(ledger):
        if r.get("logical_work_id") != lwid or not _at_or_before(r, before_utc):
            continue
        if r.get("event") == "attempt" and r.get("attempt_id"):
            attempts[r["attempt_id"]] = {"cluster": r.get("cluster"), "job_id": r.get("job_id")}
        elif r.get("event") == "superseded_attempt" and r.get("job_id"):
            for rid in r.get("run_ids") or []:
                out.setdefault(str(rid), {"cluster": r.get("cluster"), "job_id": str(r["job_id"])})
        elif r.get("event") == "registered" and r.get("attempt_id") in attempts and attempts[r["attempt_id"]].get("job_id"):
            a = attempts[r["attempt_id"]]
            for rid in r.get("run_ids") or []:
                out.setdefault(str(rid), {"cluster": a["cluster"], "job_id": str(a["job_id"])})
    old_job_ids = {j["job_id"] for j in unit_old_jobs(ledger, lwid, before_utc)} if before_utc is not None else set()
    for b in _old_bindings(ledger, lwid, before_utc, old_job_ids):
        if str(b.get("binding", "")).startswith("registered") and b.get("job_id"):
            for rid in b.get("run_ids") or []:
                out.setdefault(str(rid), {"cluster": b.get("cluster"), "job_id": str(b["job_id"])})
    return out


def registry_run_rows(registry_db: Optional[str], run_ids: List[str]) -> Dict[str, Optional[dict]]:
    """{run_id: {lifecycle_state, cluster_job_id} | None (not in the registry)} from a FRESH read-only connection — the caller reads once
    before the sacct proofs and again right before the ledger write (the fence). A missing database is a ValueError (fail closed)."""
    if not registry_db or registry_db == "none" or not os.path.isfile(registry_db):
        raise ValueError(f"registry database {registry_db!r} does not exist — the completed-run check cannot run (fail closed)")
    import sqlite3
    conn = sqlite3.connect(f"file:{registry_db}?mode=ro", uri=True)
    try:
        out: Dict[str, Optional[dict]] = {r: None for r in run_ids}
        for chunk in (run_ids[i:i + 400] for i in range(0, len(run_ids), 400)):
            if not chunk:
                continue
            marks = ",".join("?" for _ in chunk)
            try:
                rows = list(conn.execute(f"select run_id, lifecycle_state, cluster_job_id, code_sha from run where run_id in ({marks})", chunk))
            except sqlite3.OperationalError:
                try:
                    rows = [(rid, st, cj, None) for rid, st, cj in conn.execute(f"select run_id, lifecycle_state, cluster_job_id from run where run_id in ({marks})", chunk)]
                except sqlite3.OperationalError:
                    rows = [(rid, st, None, None) for rid, st in conn.execute(f"select run_id, lifecycle_state from run where run_id in ({marks})", chunk)]
            for rid, st, cj, sha in rows:
                # code_sha (32c): the run row's recorded producer tree — a --recheck tells an amended drive's own registration under an old
                # job from an old-contract run by it
                out[str(rid)] = {"lifecycle_state": st, "cluster_job_id": (str(cj) if cj not in (None, "") else None),
                                 "code_sha": (str(sha) if sha not in (None, "") else None)}
        return out
    finally:
        conn.close()


def registry_runs_of_jobs(registry_db: Optional[str], job_ids: List[str]) -> Dict[str, dict]:
    """{run_id: {lifecycle_state, cluster_job_id}} — every registry run whose OWN record names one of `job_ids` as its cluster job (the
    run_patch a job files at start). These are the old runs no binding or ledger event captured: an attempt that registered at job start
    and died before 06 --gate bound it leaves its runs only here (the second attempt of each amended multi-turn unit of 2026-09-30 left
    18 such runs in `running`, absent from the receipts). A registry without the column yields nothing; a missing database fails closed."""
    if not registry_db or registry_db == "none" or not os.path.isfile(registry_db):
        raise ValueError(f"registry database {registry_db!r} does not exist — the completed-run check cannot run (fail closed)")
    ids = sorted({str(j) for j in job_ids if j})
    if not ids:
        return {}
    import sqlite3
    conn = sqlite3.connect(f"file:{registry_db}?mode=ro", uri=True)
    try:
        out: Dict[str, dict] = {}
        for chunk in (ids[i:i + 400] for i in range(0, len(ids), 400)):
            marks = ",".join("?" for _ in chunk)
            try:
                rows = list(conn.execute(f"select run_id, lifecycle_state, cluster_job_id from run where cluster_job_id in ({marks})", chunk))
            except sqlite3.OperationalError:
                return {}
            for rid, st, cj in rows:
                out[str(rid)] = {"lifecycle_state": st, "cluster_job_id": str(cj)}
        return out
    finally:
        conn.close()


def _job_id_forms(old_jobs: List[dict], proofs: Dict[Tuple[str, str], dict]) -> List[str]:
    """Every spelling a run's `cluster_job_id` may carry for the unit's old jobs: the parent id, and per proven task its raw id and `<parent>_<k>`."""
    forms = {j["job_id"] for j in old_jobs}
    for p in proofs.values():
        forms.add(p["job_id"])
        for t in p["tasks"]:
            forms.add(t["raw"])
            if t["task"] is not None:
                forms.add(f"{p['job_id']}_{t['task']}")
    return sorted(forms)


def registry_run_states(registry_db: Optional[str], run_ids: List[str]) -> Dict[str, Optional[str]]:
    """{run_id: lifecycle_state | None (not in the registry)}; a missing database is a ValueError (fail closed)."""
    return {r: (row or {}).get("lifecycle_state") for r, row in registry_run_rows(registry_db, run_ids).items()}


def sacct_reader(config_path: Optional[str] = None):
    """The funnel's OWN sacct reader (submit_twin.Remote.sacct_tasks) over the site config's clusters: (cluster label, job id) -> the job's
    tasks [{raw, task, state, terminal, end}]. Built lazily on first use, so a unit with no attempts needs no site config; `config_path`
    defaults to $SISTEER_CONFIG, else <default site dir>/config.env. Tests inject a callable of the same shape."""
    box: Dict[str, object] = {}

    def read(label: str, job_id: str) -> List[dict]:
        if "site" not in box:
            import clusters as C
            import submit_twin as TW
            cfg = config_path or os.environ.get("SISTEER_CONFIG") or os.path.join(C.default_site_dir(), "config.env")
            if not os.path.isfile(cfg):
                raise ValueError(f"no site config at {cfg} — the sacct proof cannot reach cluster {label!r} (pass --config)")
            box["site"] = TW.Site(cfg, False); box["TW"] = TW; box["remotes"] = {}
        site = box["site"]
        if label not in site.clusters:      # type: ignore[attr-defined]
            raise ValueError(f"cluster {label!r} is not one of the site's clusters {sorted(site.clusters)} — job {job_id} cannot be read")  # type: ignore[attr-defined]
        remotes = box["remotes"]            # type: ignore[assignment]
        if label not in remotes:            # type: ignore[operator]
            remotes[label] = box["TW"].Remote(site, label)   # type: ignore[index, attr-defined]
        return remotes[label].sacct_tasks(str(job_id))       # type: ignore[index]
    return read


def prove_old_jobs_terminal(old_jobs: List[dict], sacct_tasks) -> Dict[Tuple[str, str], dict]:
    """{(cluster, job_id): {cluster, job_id, tasks: [{raw, task, state}], states: [..], utc}} — EVERY task of EVERY old job terminal on sacct
    (the reader is the funnel's, injected), else ValueError by name: a live task, a job unknown to sacct, an unreachable cluster."""
    out: Dict[Tuple[str, str], dict] = {}
    for j in old_jobs:
        cluster, job_id = j["cluster"], j["job_id"]
        try:
            tasks = list(sacct_tasks(cluster, job_id) or [])
        except Exception as exc:     # Unreachable / RuntimeError from the funnel's reader: nothing can be proven
            raise ValueError(f"job {job_id} on {cluster} ({j.get('source')}): sacct could not be read ({exc}) — its terminal state is unproven; refused")
        if not tasks:
            raise ValueError(f"job {job_id} on {cluster} ({j.get('source')}) is unknown to sacct — its terminal state cannot be read; refused")
        rows = [{"raw": str(t.get("raw")), "task": t.get("task"), "state": str(t.get("state") or "")} for t in tasks]
        live = [t for t in rows if not t["state"] or t["state"] in SACCT_ACTIVE]
        if live:
            what = ", ".join(f"task {t['task']} ({t['raw']}) {t['state'] or '?'}" if t["task"] is not None else (t["state"] or "?") for t in live)
            raise ValueError(f"job {job_id} on {cluster} ({j.get('source')}) is still {what} on sacct — an old-contract job that is alive can still "
                             f"register or generate; refused")
        out[(cluster, job_id)] = {"cluster": cluster, "job_id": job_id, "tasks": rows, "states": sorted({t["state"] for t in rows}), "utc": utc_now()}
    return out


def _run_job_states(run_id: str, row: Optional[dict], run_jobs: Dict[str, dict], proofs: Dict[Tuple[str, str], dict]) -> Tuple[Optional[dict], Optional[str]]:
    """(the proof of the run's job with the task states that are the run's, or None; a problem) — the ledger's binding first, else the
    registry's own `cluster_job_id` matched against the proven jobs (a raw task id, `<array>_<k>` or the parent id)."""
    bound = run_jobs.get(run_id)
    if bound and bound.get("job_id"):
        key = (str(bound.get("cluster")), str(bound["job_id"]))
        p = proofs.get(key)
        if p is None:
            return None, f"run {run_id} is bound to job {key[1]} on {key[0]}, which is not among the unit's proven jobs"
        return {**p, "run_states": p["states"]}, None
    cj = (row or {}).get("cluster_job_id")
    if not cj:
        return None, None
    hits = []
    for p in proofs.values():
        if p["job_id"] == cj:
            hits.append((p, p["states"]))
        else:
            mine = [t["state"] for t in p["tasks"] if t["raw"] == cj or (t["task"] is not None and f"{p['job_id']}_{t['task']}" == cj)]
            if mine:
                hits.append((p, sorted(set(mine))))
    if not hits:
        return None, f"run {run_id} names cluster job {cj}, which is not among the unit's recorded attempts — its terminal state cannot be proven"
    if len(hits) > 1:
        return None, f"run {run_id} names cluster job {cj}, which matches {len(hits)} of the unit's jobs — ambiguous; refused"
    p, states = hits[0]
    return {**p, "run_states": states}, None


def old_run_verdicts(old_runs: List[str], rows: Dict[str, Optional[dict]], run_jobs: Dict[str, dict], proofs: Dict[Tuple[str, str], dict]) -> Tuple[List[dict], List[str]]:
    """Per old run the receipt's line — {run_id, lifecycle_state, cluster, job_id, sacct_state, proof_utc, why} — and the refusals by name:
    a COMPLETED run; a `running` (or `queued`, or unknown-state) run whose job is not proven terminal-not-COMPLETED; a run bound to no job."""
    lines: List[dict] = []; problems: List[str] = []
    for rid in old_runs:
        row = rows.get(rid); st = (row or {}).get("lifecycle_state")
        proof, prob = _run_job_states(rid, row, run_jobs, proofs)
        line = {"run_id": rid, "lifecycle_state": st, "cluster": (proof or {}).get("cluster"), "job_id": (proof or {}).get("job_id"),
                "sacct_state": ",".join((proof or {}).get("run_states") or []) or None, "proof_utc": (proof or {}).get("utc"), "why": None}
        if row is None:
            line["why"] = "never registered in the registry (no run row)"
        elif st in COMPLETED_RUN_STATES:
            problems.append(f"run {rid} is `{st}` — it COMPLETED under the old contract")
            line["why"] = f"registry `{st}`: COMPLETED under the old contract — refused"
        elif st in TERMINAL_NON_COMPLETED_RUN_STATES:
            line["why"] = f"registry `{st}` (terminal, not completed)" + (f"; job {proof['job_id']} on {proof['cluster']} {line['sacct_state']} on sacct" if proof else "")
        else:
            if prob:
                problems.append(f"{prob} (registry `{st}`)"); line["why"] = prob
            elif proof is None:
                problems.append(f"run {rid} is `{st}` in the registry and no job of the unit's attempts is bound to it — its terminal state cannot be proven; refused")
                line["why"] = f"registry `{st}`, no job bound — unproven"
            elif "COMPLETED" in proof["run_states"]:
                problems.append(f"run {rid} is `{st}` in the registry but its job {proof['job_id']} on {proof['cluster']} COMPLETED on sacct — the run may have "
                                f"generated; refused")
                line["why"] = f"registry `{st}`; job {proof['job_id']} COMPLETED on sacct — refused"
            elif any(s in SACCT_ACTIVE for s in proof["run_states"]):
                problems.append(f"run {rid} is `{st}` and its job {proof['job_id']} on {proof['cluster']} is still {line['sacct_state']} on sacct; refused")
                line["why"] = f"registry `{st}`; job {proof['job_id']} {line['sacct_state']} — alive; refused"
            else:
                line["why"] = f"registry `{st}`; job {proof['job_id']} on {proof['cluster']} {line['sacct_state']} on sacct at {proof['utc']} (terminal, not COMPLETED)"
        lines.append(line)
    return lines, problems


def _amendment_row_diff(unit: dict, old_c: dict, new_c: dict, old_export: dict, new_export: dict) -> Tuple[str, dict, List[dict], dict]:
    """The instrument-ids-only contract diff and, behind it, the ONE decode-policy row's exact neutral transitions -> (ref, changed
    instruments, allowed row diff, new row); ValueError by name otherwise."""
    lwid = unit["logical_work_id"]
    diff = contract_diff(old_c, new_c)
    if not diff:
        raise ValueError(f"{lwid}: nothing to amend — the contract at this tree is identical to the effective one ({contract_dict_sha256(old_c)[:12]})")
    other = [d for d in diff if not re.match(r"evals\.[^.]+\.instrument_id: ", d)]
    if other:
        raise ValueError(f"{lwid}: the amendment would change more than instrument ids — refused by name: " + "; ".join(other))
    changed = {}
    for e in sorted(old_c["evals"]):
        if old_c["evals"][e].get("instrument_id") != new_c["evals"][e].get("instrument_id"):
            changed[e] = {"old": old_c["evals"][e].get("instrument_id"), "new": new_c["evals"][e].get("instrument_id")}
    refs = sorted({old_c["evals"][e].get("decode_policy_ref") for e in changed})
    if len(refs) != 1 or not refs[0]:
        raise ValueError(f"{lwid}: the changed evals name {len(refs)} decode-policy ref(s) ({refs}) — one row diff per amendment")
    ref = refs[0]
    old_row, new_row = decode_row(old_export, ref), decode_row(new_export, ref)
    if old_row is None or new_row is None:
        raise ValueError(f"{lwid}: decode-policy row {ref!r} is absent from the {'old' if old_row is None else 'new'} export — the row diff cannot be read")
    allowed, violations = decode_row_diff(old_row, new_row)
    if violations:
        raise ValueError(f"{lwid}: the decode-policy row {ref!r} changed beyond the exact neutral transitions of {list(AMEND_ALLOWED_KEYS)} — refused by name: "
                         + "; ".join(violations))
    if not any(c["path"] in AMEND_NEUTRAL_TRANSITIONS or c["path"] == "params_sent" for c in allowed):
        raise ValueError(f"{lwid}: the decode-policy row {ref!r} did not change in {list(AMEND_ALLOWED_KEYS)} — nothing explains the instrument change; refused")
    return ref, changed, allowed, new_row


def _derive_exports(repo: str, old_sha: str, commit_full: str, old_export_path: Optional[str]) -> Tuple[dict, dict, dict]:
    """The old and new decode-policy exports from GIT (`git show <sha>:configs/registry/decode_policy.json` at the old tree and at the
    amending commit) -> (old export, new export, {old_export_sha256, new_export_sha256, old_export_supplied}). A supplied old-export file
    is accepted ONLY byte-identical (sha256) to the derivation; the refusal names both hashes."""
    old_b = git_show_bytes(repo, old_sha, DECODE_EXPORT_REL)
    new_b = git_show_bytes(repo, commit_full, DECODE_EXPORT_REL)
    old_h, new_h = hashlib.sha256(old_b).hexdigest(), hashlib.sha256(new_b).hexdigest()
    supplied = None
    if old_export_path:
        sup_h = hashlib.sha256(open(old_export_path, "rb").read()).hexdigest()
        if sup_h != old_h:
            raise ValueError(f"the supplied --old-export {old_export_path} (sha256 {sup_h}) is not the decode-policy export at the unit's recorded tree "
                             f"{old_sha} (git show {old_sha[:12]}:{DECODE_EXPORT_REL}, sha256 {old_h}) — the old row is DERIVED, never typed; refused")
        supplied = {"path": os.path.abspath(old_export_path), "sha256": sup_h}
    try:
        old_export, new_export = json.loads(old_b.decode("utf-8")), json.loads(new_b.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"the decode-policy export at {old_sha[:12]} / {commit_full[:12]} is not JSON: {exc}")
    return old_export, new_export, {"old_export_sha256": old_h, "new_export_sha256": new_h, "old_export_supplied": supplied}


def petri_manifest_rel(unit: dict) -> str:
    """The drive manifest of the unit's wave: configs/capeval/manifests/petri_full22_<model>_<view>.yaml (the lwid's first two segments)."""
    parts = str(unit.get("logical_work_id") or "").split("/")
    if len(parts) < 2 or not parts[0] or not parts[1]:
        raise ValueError(f"{unit.get('logical_work_id')!r}: the Petri manifest path needs <model>/<view>/… in the logical work id")
    return PETRI_MANIFEST_REL_TMPL.format(tag=f"{parts[0]}_{parts[1]}")


def _yaml_doc_at(repo: str, sha: str, rel: str) -> Tuple[dict, str]:
    """(the YAML document at `sha`, its sha256) — read from GIT, never from a file handed in."""
    import yaml
    b = git_show_bytes(repo, sha, rel)
    try:
        doc = yaml.safe_load(b.decode("utf-8")) or {}
    except Exception as exc:        # noqa: BLE001 — any parse failure is the same refusal
        raise ValueError(f"{rel} at {sha[:12]} is not YAML: {exc}")
    if not isinstance(doc, dict):
        raise ValueError(f"{rel} at {sha[:12]} is not a mapping")
    return doc, hashlib.sha256(b).hexdigest()


def petri_role_diff(repo: str, old_sha: str, commit: str, rel: str) -> Tuple[List[dict], List[str], Dict[str, str]]:
    """The drive manifest's diff between the unit's recorded tree and the amending commit — [{field, before, after}] — and the refusals by
    name: any top-level key outside `petri:` that moved; any `petri.<key>` change that is not one of PETRI_ROLE_FALLBACK_TRANSITIONS with
    exactly its before -> after; an admitted transition that is absent (the amendment carries the whole per-request fallback, never a part).
    Also {manifest_rel, old_manifest_sha256, new_manifest_sha256} for the receipt."""
    old_doc, old_h = _yaml_doc_at(repo, old_sha, rel)
    new_doc, new_h = _yaml_doc_at(repo, commit, rel)
    diff: List[dict] = []; problems: List[str] = []
    for k in sorted(set(old_doc) | set(new_doc)):
        if k == "petri" or old_doc.get(k) == new_doc.get(k):
            continue
        diff.append({"field": k, "before": old_doc.get(k), "after": new_doc.get(k)})
        problems.append(f"{rel}: top-level `{k}` changed — outside the `petri:` block, not part of the per-request fallback")
    old_p = old_doc.get("petri") or {}; new_p = new_doc.get("petri") or {}
    for k in sorted(set(old_p) | set(new_p)):
        if old_p.get(k) == new_p.get(k):
            continue
        diff.append({"field": f"petri.{k}", "before": old_p.get(k), "after": new_p.get(k)})
        want = PETRI_ROLE_FALLBACK_TRANSITIONS.get(k)
        if want is None:
            problems.append(f"{rel}: petri.{k} changed ({old_p.get(k)!r} -> {new_p.get(k)!r}) — not one of the admitted fields {sorted(PETRI_ROLE_FALLBACK_TRANSITIONS)}; "
                            f"a harness-hash input (the existing fallback chains, the models, the bundle) or anything else re-keys the cells: a new wave, not this amendment")
        elif (old_p.get(k), new_p.get(k)) not in want:
            # one of the key's admitted transitions, exactly — the initial Claude leg, or (the judge) the appended terminal dated Grok
            # leg; a replaced leg, a reordered chain, a second non-Claude leg or the mutable alias is none of them
            problems.append(f"{rel}: petri.{k} is {old_p.get(k)!r} -> {new_p.get(k)!r}, not the admitted transition{'s' if len(want) > 1 else ''} "
                            + " / ".join(f"{b!r} -> {a!r}" for b, a in want))
    present = {d["field"]: d for d in diff}
    admitted_present = [k for k in PETRI_ROLE_FALLBACK_TRANSITIONS if f"petri.{k}" in present]
    # the INITIAL introduction carries the whole per-request fallback: when any admitted key enters from absent, every admitted key
    # must; a LATER transition of one key alone (the judge chain's appended terminal leg) is admitted by itself — the other key stands
    initial = [k for k in admitted_present if present[f"petri.{k}"]["before"] is None]
    if initial:
        missing = [k for k in PETRI_ROLE_FALLBACK_TRANSITIONS if k not in admitted_present]
        if missing:
            problems.append(f"{rel}: the admitted transition(s) of {missing} are absent from the diff {old_sha[:12]} -> {commit[:12]} — the amendment carries the whole "
                            f"per-request fallback (D354/D355), not a part of it")
    elif not admitted_present:
        problems.append(f"{rel}: the admitted transition(s) of {sorted(PETRI_ROLE_FALLBACK_TRANSITIONS)} are absent from the diff {old_sha[:12]} -> {commit[:12]} — nothing to amend")
    return diff, problems, {"manifest_rel": rel, "old_manifest_sha256": old_h, "new_manifest_sha256": new_h}


#: registry seed files a code-only follow-up MAY change, by exact repo-relative path — none today: the frozen-wave gate holds the whole tree
CODE_FOLLOW_UP_SEED_EXEMPT: Tuple[str, ...] = ()
SEED_TREE_REL = "configs/registry"


def seed_tree_digest(repo: str, sha: str, exempt: Tuple[str, ...] = CODE_FOLLOW_UP_SEED_EXEMPT) -> Tuple[str, List[str]]:
    """(sha256 of the committed configs/registry TREE at `sha` — every blob's mode, type, blob sha and path from `git ls-tree -r`, the exempt
    paths removed —, the paths digested). The whole seed tree, not this unit's projection of it: a seed change outside
    the Petri unit's instrument/contract is still a seed change the frozen-wave gate refuses."""
    r = _git(repo, "ls-tree", "-r", sha, "--", SEED_TREE_REL)
    if r.returncode != 0:
        raise ValueError(f"cannot list {SEED_TREE_REL} at {sha[:12]} in {repo}: {r.stderr.decode('utf-8', 'replace').strip()[:200]}")
    kept: List[str] = []
    for ln in r.stdout.decode("utf-8", "replace").splitlines():
        if not ln.strip() or "\t" not in ln:
            continue
        meta, path = ln.split("\t", 1)
        if path in exempt:
            continue
        kept.append(f"{meta}\t{path}")
    kept.sort()
    return hashlib.sha256("\n".join(kept).encode("utf-8")).hexdigest(), [k.split("\t", 1)[1] for k in kept]


def seed_tree_changes(repo: str, old_sha: str, new_sha: str, exempt: Tuple[str, ...] = CODE_FOLLOW_UP_SEED_EXEMPT) -> List[str]:
    """`<status> <path>` for every configs/registry path that differs between the trees (exempt paths removed)."""
    r = _git(repo, "diff", "--name-status", old_sha, new_sha, "--", SEED_TREE_REL)
    if r.returncode != 0:
        raise ValueError(f"cannot diff {SEED_TREE_REL} {old_sha[:12]}..{new_sha[:12]} in {repo}: {r.stderr.decode('utf-8', 'replace').strip()[:200]}")
    out: List[str] = []
    for ln in r.stdout.decode("utf-8", "replace").splitlines():
        parts = ln.split("\t")
        if len(parts) >= 2 and parts[-1] not in exempt:
            out.append(f"{parts[0]} {parts[-1]}")
    return out


def code_follow_up_diff(repo: str, old_sha: str, commit: str, rel: str) -> Tuple[List[str], Dict[str, str]]:
    """The code-only follow-up's proof over the trees: the drive manifest `rel` and the roster F22_ROSTER_REL are UNCHANGED between the
    unit's amended sha and the follow-up commit (documents compared, bytes hashed for the receipt); a file unreadable at either tree is a
    refusal by name. Returns (problems, {manifest_rel, old/new_manifest_sha256, roster_rel, old/new_roster_sha256})."""
    probs: List[str] = []; out: Dict[str, str] = {"manifest_rel": rel, "roster_rel": F22_ROSTER_REL}
    for label, path in (("manifest", rel), ("roster", F22_ROSTER_REL)):
        try:
            old_doc, old_h = _yaml_doc_at(repo, old_sha, path); new_doc, new_h = _yaml_doc_at(repo, commit, path)
        except ValueError as exc:
            probs.append(f"{path}: unreadable at one of the trees — {exc}"); continue
        out[f"old_{label}_sha256"] = old_h; out[f"new_{label}_sha256"] = new_h
        if old_doc != new_doc:
            fo, fn = _flat(old_doc), _flat(new_doc)
            moved = sorted(k for k in set(fo) | set(fn) if fo.get(k) != fn.get(k))
            probs.append(f"{path}: changed between {old_sha[:12]} and {commit[:12]} ({moved[:6]}{' …' if len(moved) > 6 else ''}) — a code-only follow-up "
                         f"carries no {label} change; refused by name")
    return probs, out


def read_old_jobs_states(old_jobs: List[dict], sacct_tasks) -> Dict[Tuple[str, str], dict]:
    """`prove_old_jobs_terminal`'s read without its refusal of a LIVE job: {(cluster, job_id): {cluster, job_id, tasks, states, utc, live}} —
    the petri-role-fallback kind records the serving job's state (the drive restarts against it); a job unknown to sacct or an unreachable
    cluster still refuses by name."""
    out: Dict[Tuple[str, str], dict] = {}
    for j in old_jobs:
        cluster, job_id = j["cluster"], j["job_id"]
        try:
            tasks = list(sacct_tasks(cluster, job_id) or [])
        except Exception as exc:     # Unreachable / RuntimeError from the funnel's reader: nothing can be read
            raise ValueError(f"job {job_id} on {cluster} ({j.get('source')}): sacct could not be read ({exc}) — its state is unknown; refused")
        if not tasks:
            raise ValueError(f"job {job_id} on {cluster} ({j.get('source')}) is unknown to sacct — its state cannot be read; refused")
        rows = [{"raw": str(t.get("raw")), "task": t.get("task"), "state": str(t.get("state") or "")} for t in tasks]
        out[(cluster, job_id)] = {"cluster": cluster, "job_id": job_id, "tasks": rows, "states": sorted({t["state"] for t in rows}), "utc": utc_now(),
                                  "live": any(not t["state"] or t["state"] in SACCT_ACTIVE for t in rows)}
    return out


def _is_amended_sha(sha: Optional[str], amended_shas: Optional[Set[str]]) -> bool:
    """The row's recorded code_sha names one of the amendment commits: equal, or an abbreviation of at least 12 hex characters of one
    (the sha12 the ledgers and logs carry); None or anything shorter never matches (fail closed)."""
    if not sha or not amended_shas:
        return False
    s = str(sha).strip().lower()
    if len(s) < 12 or any(c not in "0123456789abcdef" for c in s):
        return False
    return any(str(a).lower() == s or str(a).lower().startswith(s) for a in amended_shas)


def registry_run_first_registration(registry_db: Optional[str], run_ids: List[str]) -> Dict[str, Optional[dict]]:
    """{run_id: {written_utc, record_sha256, record_uri, code_sha, cluster_job_id} | None} — the run's ORIGINAL registration: the FIRST `run`
    record of the key in the registry's row_history (immutable). The live row's `code_sha` is the fold's LAST writer: the amended drive
    imports an identity the old proxy registered and re-emits a full `run` record for it with its own checkout sha over the old
    cluster_job_id, so only the first record can say who registered the run and when. A registry without row_history
    (or without the key's history) yields None for the id — the caller refuses (fail closed); a missing database raises."""
    if not registry_db or registry_db == "none" or not os.path.isfile(registry_db):
        raise ValueError(f"registry database {registry_db!r} does not exist — the registration history cannot be read (fail closed)")
    import sqlite3
    out: Dict[str, Optional[dict]] = {r: None for r in run_ids}
    if not run_ids:
        return out
    conn = sqlite3.connect(f"file:{registry_db}?mode=ro", uri=True)
    try:
        for rid in run_ids:
            try:
                rows = conn.execute("SELECT written_utc, record_sha256, record_uri, payload_json FROM row_history WHERE table_name = 'run' AND record_type = 'run' "
                                    "AND json_extract(natural_key_json, '$.run_id') = ? ORDER BY written_utc, record_sha256 LIMIT 1", (rid,)).fetchall()
            except sqlite3.OperationalError:
                return out
            if rows:
                try:
                    p = json.loads(rows[0][3]) if rows[0][3] else {}
                except ValueError:
                    p = {}
                job = str(p["cluster_job_id"]) if p.get("cluster_job_id") not in (None, "") else None
                if job is None:
                    # the first `run` record named no job (a producer that files it in a run_patch at job start): the EARLIEST record of the
                    # key that carries one is the registration's job (32c-d)
                    try:
                        jr = conn.execute("SELECT COALESCE(json_extract(payload_json, '$.cluster_job_id'), json_extract(payload_json, '$.fields.cluster_job_id')) FROM row_history "
                                          "WHERE table_name = 'run' AND record_type IN ('run', 'run_patch') AND json_extract(natural_key_json, '$.run_id') = ? "
                                          "AND COALESCE(json_extract(payload_json, '$.cluster_job_id'), json_extract(payload_json, '$.fields.cluster_job_id')) IS NOT NULL "
                                          "ORDER BY written_utc, record_sha256 LIMIT 1", (rid,)).fetchone()
                    except sqlite3.OperationalError:
                        jr = None
                    job = str(jr[0]) if jr and jr[0] not in (None, "") else None
                out[rid] = {"written_utc": rows[0][0], "record_sha256": rows[0][1], "record_uri": rows[0][2],
                            "code_sha": p.get("code_sha"), "cluster_job_id": job}
        return out
    finally:
        conn.close()


def registry_runs_of_jobs_history(registry_db: Optional[str], job_ids: List[str]) -> Dict[str, dict]:
    """{run_id: {written_utc, cluster_job_id, code_sha, record_type}} — every run whose registration HISTORY (row_history: `run` and `run_patch`
    records, immutable) ever named one of `job_ids` as its cluster job, keyed by the earliest such record. The live row's cluster_job_id is
    the fold's LAST writer: a late old-job registration re-emitted under a fresh job vanishes from `registry_runs_of_jobs`
    while its first record still names the old job — this is the inventory that cannot forget it. A registry without row_history yields
    nothing; a missing database fails closed."""
    if not registry_db or registry_db == "none" or not os.path.isfile(registry_db):
        raise ValueError(f"registry database {registry_db!r} does not exist — the registration history cannot be read (fail closed)")
    ids = sorted({str(j) for j in job_ids if j})
    if not ids:
        return {}
    import sqlite3
    conn = sqlite3.connect(f"file:{registry_db}?mode=ro", uri=True)
    try:
        out: Dict[str, dict] = {}
        for chunk in (ids[i:i + 400] for i in range(0, len(ids), 400)):
            marks = ",".join("?" for _ in chunk)
            try:
                rows = list(conn.execute(
                    "SELECT json_extract(natural_key_json, '$.run_id'), written_utc, record_type, "
                    "COALESCE(json_extract(payload_json, '$.cluster_job_id'), json_extract(payload_json, '$.fields.cluster_job_id')), "
                    "json_extract(payload_json, '$.code_sha') FROM row_history WHERE table_name = 'run' AND record_type IN ('run', 'run_patch') "
                    f"AND COALESCE(json_extract(payload_json, '$.cluster_job_id'), json_extract(payload_json, '$.fields.cluster_job_id')) IN ({marks}) "
                    "ORDER BY written_utc, record_sha256", chunk))
            except sqlite3.OperationalError:
                return {}
            for rid, utc, rtype, cj, sha in rows:
                if rid and str(rid) not in out:
                    out[str(rid)] = {"written_utc": utc, "cluster_job_id": str(cj), "code_sha": sha, "record_type": rtype}
        return out
    finally:
        conn.close()


def _amended_registration(first: Optional[dict], receipt_utc: Optional[str], amended_shas: Optional[Set[str]],
                          old_job_ids: Optional[Set[str]] = None) -> Tuple[bool, str]:
    """(the amended drive's OWN registration?, why) — from the run's original registration (registry_run_first_registration) and the receipt's
    stamp: yes only when the FIRST `run` record was written AFTER the receipt (to the second) AND by an amendment's commit. A first record at
    or before the receipt is an old-contract registration the receipt never listed (a late-ingested old-proxy registration, however the live
    row's code_sha reads); a later first record by a non-amended sha is the old proxy registering late; no history is unclassifiable — all
    refused. And the job (32c-c): the amended drive runs with SLURM_JOB_ID set to the ORIGINAL proxy's
    job and its recorder writes that as cluster_job_id, so an unlisted identity it registers after the mint under an OLD job carries an
    amended sha and a post-mint stamp yet never belonged to the receipt — its legitimate registrations are exactly the CARRIED ids. An
    unlisted id is admitted only with explicit fresh job provenance: a first record under a job the receipt does NOT name (`old_job_ids`,
    every spelling); a first record under an old job, or with no job, or when the receipt's old jobs are unknown here, is refused."""
    if not first:
        return False, "no registration history for the run in the registry (row_history): unclassifiable, treated as an old-contract registration"
    utc = str(first.get("written_utc") or ""); sha = first.get("code_sha"); job = first.get("cluster_job_id")
    sha12 = str(sha)[:12] if sha else None
    if old_job_ids is None:
        return False, f"the receipt's old jobs are unknown to this check — first registered {utc or '?'} by code_sha {sha12} under job {job}: not admitted without them"
    if job is None or str(job).strip() == "":
        return False, f"first registered {utc or '?'} by code_sha {sha12} with NO cluster job recorded: no fresh amended-job provenance — an unlisted identity is admitted only under a job the receipt does not name"
    if str(job).strip() in {str(j).strip() for j in old_job_ids}:
        return False, (f"first registered {utc or '?'} under the receipt's OLD job {job} (code_sha {sha12}) — an identity the receipt never listed under an old job is never "
                       f"the amended drive's own registration (its legitimate registrations are the carried ids; the drive runs under the original proxy's job id)")
    if not receipt_utc or not utc or utc[:19] <= str(receipt_utc)[:19]:
        return False, (f"first registered {utc or '?'} — at or before the receipt {receipt_utc} — by code_sha {sha12} under job {job}: an old-contract "
                       f"registration the receipt never listed")
    if not _is_amended_sha(sha, amended_shas):
        return False, f"first registered {utc} after the receipt but by code_sha {sha12} (not an amendment's commit) under job {job}: the old proxy registering late"
    return True, f"first registered {utc} after the receipt {receipt_utc} by code_sha {sha12} (an amendment's commit) under the fresh job {job} (not one the receipt names; record {first.get('record_uri')})"


def petri_run_verdicts(old_runs: List[str], rows: Dict[str, Optional[dict]], carried: bool = True,
                       carried_ids: Optional[Set[str]] = None, amended_shas: Optional[Set[str]] = None,
                       known_ids: Optional[Set[str]] = None, first_regs: Optional[Dict[str, Optional[dict]]] = None,
                       receipt_utc: Optional[str] = None, old_job_ids: Optional[Set[str]] = None) -> Tuple[List[dict], List[str]]:
    """Per old run the receipt's line for the petri-role-fallback kind: a COMPLETED run refuses; a terminal non-completed run is recorded;
    every other state (`running`, `queued`, unknown) is recorded as CARRIED under the amended contract when the instrument is unchanged
    (the expected case: same run id, same instrument — the resumed drive fills this very registration, its completed samples carried by
    provenance; a supersede would orphan them), else as superseded (the instrument moved: the drive re-registers under new
    run ids; the old row's terminal registry record is the registry's to file). `carried_ids` (a --recheck, from the receipt's
    `carried_run_ids`) evaluates the old-contract completion predicate at RECEIPT time: a carried id that completed afterwards is the
    AMENDED drive's work, never an old-contract completion; an uncarried old id completing stays a failure.
    In that mode membership in `carried_ids` comes BEFORE any carried attribution: an old-job run id the
    receipt did not carry — one a still-live old job registered AFTER the mint, found by registry_runs_of_jobs at recheck — is the amended
    drive's own registration only when its recorded provenance says so (`amended_shas`: the row's code_sha is the amendment's commit or a
    later receipt's); otherwise it is refused by name, never silently added to the carried set. `known_ids` (the receipt's `old_run_ids`,
    carried or superseded) are the ids the receipt already ruled on: a superseded id (the instrument moved) stays superseded, never
    "newly discovered"; a receipt's carried id stays carried whatever `carried` says for the others."""
    lines: List[dict] = []; problems: List[str] = []
    for rid in old_runs:
        row = rows.get(rid); st = (row or {}).get("lifecycle_state"); sha = (row or {}).get("code_sha")
        line = {"run_id": rid, "lifecycle_state": st, "cluster": None, "job_id": None, "sacct_state": None, "proof_utc": None, "why": None}
        # a --recheck (carried_ids given) and the receipt neither carried this id nor listed it at all: newly discovered since the mint
        absent = carried_ids is not None and rid not in carried_ids and not (known_ids and rid in known_ids)
        if row is None:
            line["why"] = "never registered in the registry (no run row)"
        elif absent and st not in TERMINAL_NON_COMPLETED_RUN_STATES and _amended_registration((first_regs or {}).get(rid), receipt_utc, amended_shas, old_job_ids)[0]:
            # 32c-b/32c-c: classified by the run's ORIGINAL registration (row_history's first `run` record: stamp, sha AND job), never by the
            # live row's code_sha; a first record under one of the receipt's old jobs is never the amended drive's own
            line["why"] = (f"registry `{st}`: registered AFTER the receipt by the AMENDED drive under a fresh job — "
                           f"{_amended_registration((first_regs or {}).get(rid), receipt_utc, amended_shas, old_job_ids)[1]}; its own registration, not an old-contract run; not carried")
            line["carried"] = False; line["amended_registration"] = True
        elif st in COMPLETED_RUN_STATES and carried_ids is not None and rid in carried_ids:
            line["why"] = f"registry `{st}`: completed under the AMENDED contract — a carried run id the resumed drive filled (D447-A), not an old-contract completion"
            line["carried"] = True
        elif st in COMPLETED_RUN_STATES:
            problems.append(f"run {rid} is `{st}` — it COMPLETED under the old contract")
            line["why"] = f"registry `{st}`: COMPLETED under the old contract — refused"
        elif st in TERMINAL_NON_COMPLETED_RUN_STATES:
            line["why"] = f"registry `{st}` (terminal, not completed)"
        elif absent:
            _, why = _amended_registration((first_regs or {}).get(rid), receipt_utc, amended_shas, old_job_ids)
            problems.append(f"run {rid} is `{st}` and ABSENT from the receipt's carried_run_ids — an old-job run the receipt never listed; {why}; not receipted, "
                            f"not the amended drive's own registration; refused by name (the live row's code_sha {sha!r} is the fold's last writer, not provenance)")
            line["why"] = f"registry `{st}`: absent from the receipt's carried set — {why} — refused"
            line["carried"] = False
        elif carried_ids is not None and rid in carried_ids:
            line["why"] = f"registry `{st}`: the receipt's carried id — carried under the amended contract; the resumed drive fills this registration (D447-A)"
            line["carried"] = True
        elif carried:
            line["why"] = f"registry `{st}`: carried under the amended contract — same run id, same instrument; the resumed drive fills this registration (D447-A)"
            line["carried"] = True
        else:
            line["why"] = f"registry `{st}`: superseded by the amendment — the instrument moved; the drive re-registers under new run ids; this row's terminal registry record is the registry's to file"
            line["carried"] = False
        lines.append(line)
    return lines, problems


def refresh_wave_unit_shas(ledger: str) -> List[str]:
    """Re-derive `unit_shas` (= sha_table) in every WAVE.<view>.json beside the ledger — the per-unit sha table 06 wrote at the wave's
    close; a receipted advance moves it. Returns the files rewritten (every other key kept; `unit_shas_refreshed_utc` stamped)."""
    state = os.path.dirname(os.path.abspath(ledger)); out: List[str] = []
    table = sha_table(ledger)
    for fn in sorted(os.listdir(state)) if os.path.isdir(state) else []:
        if not (fn.startswith("WAVE.") and fn.endswith(".json")) or ".partial." in fn:
            continue
        path = os.path.join(state, fn)
        try:
            doc = json.load(open(path, encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(doc, dict) or "unit_shas" not in doc:
            continue
        doc["unit_shas"] = table; doc["unit_shas_refreshed_utc"] = utc_now()
        write_atomic(path, doc); out.append(path)
    return out


def amend_contract(ledger: str, lwid: str, *, reason: str, commit: str, tip: str, repo: str, registry_db: Optional[str], sacct_tasks,
                   old_export_path: Optional[str] = None, dry_run: bool = False, kind: str = "decode-neutral", manifest_rel: Optional[str] = None,
                   note: Optional[str] = None) -> dict:
    """A RECEIPTED amendment of one unit's frozen eval contract, admissible ONLY when
    (1) the OLD decode-policy export is DERIVED from the unit's recorded tree (amendment_old_sha -> `git show <sha>:configs/registry/
        decode_policy.json`), never typed — a supplied old-export file is accepted only byte-identical to it; `commit` is a commit of `repo`
        that is an ancestor-or-equal of `tip` (the tree the unit's re-run will be submitted at) and the NEW export and the seeds the new
        contract is minted from are read from git at `commit`,
    (2) the contract changes are instrument ids alone AND the underlying decode-policy row diff is confined to the three keys' EXACT neutral
        transitions (min_p null->0.0, repetition_penalty null->1.0, presence_penalty null->0.0) plus documentation fields — any other field,
        value or transition is a refusal by name,
    (3) EVERY old job of the unit is TERMINAL on sacct (the funnel's reader, `sacct_tasks(cluster, job_id)`) and EVERY old run — the
        ledger's and bindings' run ids UNION the registry's runs whose own record names one of those jobs (registry_runs_of_jobs; a job
        that registered at start and died before 06 --gate bound it leaves its runs only there) — is definitively non-completed: `failed`/`cancelled`-class states pass; a `running` run passes only when its job is sacct-proven
        terminal-not-COMPLETED, recorded per run (run id, lifecycle state, job id, sacct state, the proof's utc); the registry is read
        BEFORE the proofs and AGAIN right before the ledger event is written (the fence), and any run that completed meanwhile refuses,
    (4) the entry carries the reason, the amending commit (as given and resolved), the tip, the old sha, both export
        hashes, old -> new contract hashes, the old jobs and runs with why each is excluded, and the amended contract itself, which the
        funnel and admission then compare against (effective_contract). The inventory file stays frozen. The receipt is a ledger event
        FIRST, then the sidecar entry. Returns the entry; raises ValueError by name on any refusal."""
    inv = load_inventory(ledger)
    unit = inventory_unit(inv, lwid)
    if unit is None:
        raise ValueError(f"{lwid}: not a unit of the frozen inventory at {inventory_path(ledger)}")
    if not reason or not re.search(r"\bD\d+\b", reason):
        raise ValueError("an amendment needs a --reason carrying its ruling's D-number")
    if not re.fullmatch(r"[0-9a-f]{7,40}", commit or ""):
        raise ValueError(f"--commit must be the amending commit's sha, got {commit!r}")
    probs = [p for p in reconcile_provenance(ledger) if p.startswith(f"{lwid}:")]
    if probs:
        raise ValueError(f"{lwid}: the provenance sidecar disagrees with the attempts ledger — {probs[0]}")
    # (1) the trees: the amending commit is bound to the tip the re-run submits at; the old export is derived from the recorded tree
    commit_full = git_rev(repo, commit)
    tip_full = git_rev(repo, tip)
    if not git_is_ancestor(repo, commit_full, tip_full):
        raise ValueError(f"--commit {commit} ({commit_full[:12]}) is not an ancestor-or-equal of the tip {tip_full[:12]} the unit will run at — the "
                         f"amending commit must be IN the tree the re-run submits; refused")
    if kind not in AMEND_KINDS:
        raise ValueError(f"--kind {kind!r} is not one of {list(AMEND_KINDS)}")
    amendments = read_amendments(ledger)
    prior = amendments.get(lwid, [])
    old_sha, old_sha_source = amendment_old_sha(ledger, unit, prior)
    old_export, new_export, hashes = _derive_exports(repo, old_sha, commit_full, old_export_path)
    old_c = effective_contract(amendments, unit)
    new_c = contract_at_commit(unit, repo, commit_full)
    role_diff: Optional[List[dict]] = None
    follow_up: Optional[dict] = None
    if kind == "decode-neutral":
        # (2) instrument ids alone, explained by the one row's exact neutral transitions
        ref, changed, allowed, new_row = _amendment_row_diff(unit, old_c, new_c, old_export, new_export)
    elif kind == "code-follow-up":
        # (2'') a CODE-ONLY follow-up to a receipted sha advance — the commit a strict descendant of the amended sha; the drive
        # manifest, the roster, the contract and the decode-policy export unchanged; the --note names the findings it carries; the old run
        # set and the carried set are checked against the prior receipt below
        if unit.get("family") != "petri":
            raise ValueError(f"{lwid}: --kind code-follow-up applies to the Petri unit only (this unit's family is {unit.get('family')!r})")
        if not prior or str(prior[-1].get("kind")) not in ("petri-role-fallback", "code-follow-up"):
            raise ValueError(f"{lwid}: --kind code-follow-up needs a prior receipted sha advance (petri-role-fallback or code-follow-up) to follow — none recorded")
        if not note:
            raise ValueError(f"{lwid}: --kind code-follow-up needs --note naming the findings the follow-up carries (D518)")
        if commit_full == old_sha or not git_is_ancestor(repo, old_sha, commit_full):
            raise ValueError(f"{lwid}: --commit {commit_full[:12]} is not a strict descendant of the unit's amended sha {old_sha[:12]} — a follow-up advances forward only; refused")
        rel = petri_manifest_rel(unit)
        if manifest_rel and os.path.normpath(manifest_rel) != os.path.normpath(rel):
            raise ValueError(f"{lwid}: --manifest-rel {manifest_rel!r} is not the unit's drive manifest {rel!r}; refused")
        fprobs, fh = code_follow_up_diff(repo, old_sha, commit_full, rel)
        hashes.update(fh)
        # 32c-e: the committed seed TREE itself, not this unit's projection of it
        old_seed, _ = seed_tree_digest(repo, old_sha); new_seed, _ = seed_tree_digest(repo, commit_full)
        hashes.update({"old_seed_tree_sha256": old_seed, "new_seed_tree_sha256": new_seed})
        if old_seed != new_seed:
            moved = seed_tree_changes(repo, old_sha, commit_full)
            fprobs.append(f"{SEED_TREE_REL}: the committed seed TREE changed ({moved[:8]}{' …' if len(moved) > 8 else ''}) — not code-only; a seed change outside this unit's "
                          f"projected contract is still a seed change the frozen-wave gate refuses")
        if hashes["old_export_sha256"] != hashes["new_export_sha256"]:
            fprobs.append(f"configs/registry/decode_policy.json: the export moved ({hashes['old_export_sha256'][:12]} -> {hashes['new_export_sha256'][:12]}) — not code-only")
        if contract_dict_sha256(old_c) != contract_dict_sha256(new_c):
            fprobs.append(f"the contract moved ({contract_dict_sha256(old_c)[:12]} -> {contract_dict_sha256(new_c)[:12]}) — not code-only")
        if fprobs:
            raise ValueError(f"{lwid}: the follow-up {old_sha[:12]} -> {commit_full[:12]} is not code-only — refused by name: " + "; ".join(fprobs))
        changed = {}; role_diff = []; ref, allowed, new_row = None, [], {}
        follow_up = {"utc": prior[-1].get("utc"), "kind": str(prior[-1].get("kind")), "commit": old_sha,
                     "carried_run_ids": sorted(str(x) for x in (prior[-1].get("carried_run_ids") or [])),
                     "old_run_ids": sorted({str(x.get("run_id")) for x in (prior[-1].get("old_run_ids") or []) if x.get("run_id")})}
    else:
        # (2') petri-role-fallback: the drive's manifest changed by EXACTLY the admitted added keys; the contract unchanged (the keys are
        # not harness-hash inputs — the expected case) or Petri instrument ids alone; anything else refuses by name
        if unit.get("family") != "petri":
            raise ValueError(f"{lwid}: --kind petri-role-fallback applies to the Petri unit only (this unit's family is {unit.get('family')!r})")
        rel = petri_manifest_rel(unit)
        if manifest_rel and os.path.normpath(manifest_rel) != os.path.normpath(rel):
            # another manifest (another view's) carrying the two additions proves nothing about THIS unit's
            # drive, which reads configs/capeval/manifests/petri_full22_<model>_<view>.yaml (petri_drive_full22.sh) — refused by name
            raise ValueError(f"{lwid}: --manifest-rel {manifest_rel!r} is not the unit's drive manifest {rel!r} (petri_drive_full22.sh reads "
                             f"petri_full22_<model>_<view>.yaml for the unit's model and view); an override that names another manifest is refused")
        role_diff, rprobs, mh = petri_role_diff(repo, old_sha, commit_full, rel)
        hashes.update(mh)
        if rprobs:
            raise ValueError(f"{lwid}: the Petri manifest diff {old_sha[:12]} -> {commit_full[:12]} is not the per-request refusal fallback alone — refused by name: "
                             + "; ".join(rprobs))
        diff = contract_diff(old_c, new_c)
        other = [d for d in diff if not re.match(r"evals\.[^.]+\.instrument_id: ", d)]
        if other:
            raise ValueError(f"{lwid}: the amendment would change more than instrument ids — refused by name: " + "; ".join(other))
        changed = {e: {"old": old_c["evals"][e].get("instrument_id"), "new": new_c["evals"][e].get("instrument_id")}
                   for e in sorted(old_c["evals"]) if old_c["evals"][e].get("instrument_id") != new_c["evals"][e].get("instrument_id")}
        foreign = [e for e in changed if e not in PETRI_EVALS]
        if foreign:
            raise ValueError(f"{lwid}: the amendment moves the instrument of non-Petri eval(s) {foreign} — refused by name")
        ref, allowed, new_row = None, [], {}
        # A LATER role-fallback receipt (the judge chain's appended terminal leg) FOLLOWS the prior receipted advance the way a
        # code-only follow-up does — the commit a strict descendant of the amended sha, the old run set the prior receipt's exactly, the carried
        # set carried FORWARD: a carried id that completed since completed under the AMENDED contract (the prior receipt's chain, as served and
        # recorded per sample) and keeps its recorded provenance — listed under `carried_completed`, never refused as an old-contract completion,
        # never re-run under the new leg. The FIRST receipt (no prior) keeps its rule: no run may have completed under the frozen contract.
        if prior and str(prior[-1].get("kind")) in ("petri-role-fallback", "code-follow-up"):
            if commit_full == old_sha or not git_is_ancestor(repo, old_sha, commit_full):
                raise ValueError(f"{lwid}: --commit {commit_full[:12]} is not a strict descendant of the unit's amended sha {old_sha[:12]} — a later role-fallback "
                                 f"receipt advances forward only; refused")
            follow_up = {"utc": prior[-1].get("utc"), "kind": str(prior[-1].get("kind")), "commit": old_sha,
                         "carried_run_ids": sorted(str(x) for x in (prior[-1].get("carried_run_ids") or [])),
                         "old_run_ids": sorted({str(x.get("run_id")) for x in (prior[-1].get("old_run_ids") or []) if x.get("run_id")})}
    # (3) the old jobs and runs: registry read #1, the sacct read (terminal required for decode-neutral; RECORDED for the Petri kind,
    # whose serving job stays up for the restarted drive), registry read #2
    ledger_runs = unit_old_run_ids(ledger, lwid)
    old_jobs = unit_old_jobs(ledger, lwid)
    pre_utc = utc_now()
    by_job = registry_runs_of_jobs(registry_db, _job_id_forms(old_jobs, {})) if old_jobs else {}
    old_runs = sorted(set(ledger_runs) | set(by_job))
    rows1 = registry_run_rows(registry_db, old_runs) if old_runs else {}
    # a follow-up's CARRIED ids (the prior receipt's) completing is the amended drive's work, never an old-contract completion
    prior_carried: Set[str] = set(follow_up["carried_run_ids"]) if follow_up else set()
    completed = sorted(r for r, row in rows1.items() if (row or {}).get("lifecycle_state") in COMPLETED_RUN_STATES and r not in prior_carried)
    if completed:
        raise ValueError(f"{lwid}: {len(completed)} run(s) COMPLETED under the old contract ({completed[:4]}{' …' if len(completed) > 4 else ''}) — "
                         f"an amendment is admissible only when none did; this is a new wave, not an amendment")
    try:
        proofs = prove_old_jobs_terminal(old_jobs, sacct_tasks) if kind == "decode-neutral" else read_old_jobs_states(old_jobs, sacct_tasks)
    except ValueError as exc:
        raise ValueError(f"{lwid}: {exc}")
    run_jobs = unit_run_jobs(ledger, lwid)
    if proofs:      # the proofs name every task's raw id: the registry is asked again for runs filed under any of those spellings
        by_job = registry_runs_of_jobs(registry_db, _job_id_forms(old_jobs, proofs))
        old_runs = sorted(set(old_runs) | set(by_job))
    post_utc = utc_now()
    rows2 = registry_run_rows(registry_db, old_runs) if old_runs else {}
    completed2 = sorted(r for r, row in rows2.items() if (row or {}).get("lifecycle_state") in COMPLETED_RUN_STATES and r not in prior_carried)
    if completed2:
        raise ValueError(f"{lwid}: {len(completed2)} run(s) COMPLETED under the old contract AFTER the sacct proofs were gathered ({completed2[:4]}) — the "
                         f"registry re-read at {post_utc} refuses the amendment; nothing written")
    if kind == "decode-neutral":
        lines, problems = old_run_verdicts(old_runs, rows2, run_jobs, proofs)
    elif follow_up:
        # the old run set must be the prior receipt's exactly — a run an old job registered since is a new fact a code-only follow-up cannot
        # carry — and the carried set stays the receipt's (a carried id that completed is the amended drive's work; one that failed
        # or left the set is a change the follow-up refuses by name)
        new_ids = sorted(set(old_runs) - set(follow_up["old_run_ids"])); gone = sorted(set(follow_up["old_run_ids"]) - set(old_runs))
        if new_ids or gone:
            raise ValueError(f"{lwid}: the old run set moved since the prior receipt ({follow_up['utc']}): new {new_ids[:4]}{' …' if len(new_ids) > 4 else ''}, "
                             f"gone {gone[:4]}{' …' if len(gone) > 4 else ''} — a code-only follow-up carries the receipt's set unchanged; refused by name")
        lines, problems = petri_run_verdicts(old_runs, rows2, carried=False, carried_ids=prior_carried, amended_shas={old_sha}, known_ids=set(follow_up["old_run_ids"]))
        carried_now = sorted(x["run_id"] for x in lines if x.get("carried"))
        if carried_now != follow_up["carried_run_ids"]:
            problems.append(f"the carried set moved since the prior receipt: now {len(carried_now)} id(s), the receipt carried {len(follow_up['carried_run_ids'])} "
                            f"(left: {sorted(set(follow_up['carried_run_ids']) - set(carried_now))[:4]}; joined: {sorted(set(carried_now) - set(follow_up['carried_run_ids']))[:4]})")
    else:
        lines, problems = petri_run_verdicts(old_runs, rows2, carried=not changed)
    if problems:
        raise ValueError(f"{lwid}: {len(problems)} old run(s) are not definitively non-completed — " + "; ".join(problems))
    live_jobs = [{"cluster": p["cluster"], "job_id": p["job_id"], "sacct_states": p["states"], "proof_utc": p["utc"]} for p in proofs.values() if p.get("live")]
    entry = {"utc": utc_now(), "kind": kind, "reason": reason, "commit": commit, "commit_resolved": commit_full, "tip": tip_full, "old_sha": old_sha,
             "old_sha_source": old_sha_source, **hashes,
             "old_contract_sha256": contract_dict_sha256(old_c), "new_contract_sha256": contract_dict_sha256(new_c),
             "changed_instruments": changed, "decode_policy_ref": ref, "policy_id": new_row.get("policy_id"), "row_diff": allowed,
             "role_diff": role_diff, "live_jobs": live_jobs, "note": note,
             "follow_up_of": ({k: follow_up[k] for k in ("utc", "kind", "commit")} if follow_up else None),
             "carried_run_ids": sorted(x["run_id"] for x in lines if x.get("carried")),
             # the carried runs already COMPLETED under the prior receipt's contract — carried forward with the provenance the registry
             # recorded (their code_sha; their per-sample chain is the store's), under the receipt they completed under; not new runs of this leg
             "carried_completed": [{"run_id": x["run_id"], "lifecycle_state": x.get("lifecycle_state"), "code_sha": (rows2.get(x["run_id"]) or {}).get("code_sha"),
                                    "under": (follow_up or {}).get("commit")}
                                   for x in lines if x.get("carried") and x.get("lifecycle_state") in COMPLETED_RUN_STATES],
             "sha_advance": ({"from": old_sha, "to": commit_full} if kind != "decode-neutral" else None),
             "old_jobs": [{"cluster": p["cluster"], "job_id": p["job_id"], "sacct_states": p["states"], "tasks": p["tasks"], "proof_utc": p["utc"]}
                          for p in proofs.values()],
             "old_run_ids": lines,
             "completed_run_check": {"registry": registry_db, "n_old_runs": len(old_runs), "n_old_jobs": len(old_jobs), "completed": [],
                                     "n_runs_from_ledger": len(ledger_runs), "n_runs_from_registry_by_job": len(set(by_job) - set(ledger_runs)),
                                     "pre_fence_utc": pre_utc, "post_fence_utc": post_utc},
             "contract": new_c}
    if not dry_run:
        import grant_ledger as GL   # the append-only attempts ledger is the truth; the sidecar is its readable receipt
        GL.Ledger(os.path.join(os.path.dirname(os.path.abspath(ledger)), "attempts.jsonl")).append(
            {"event": "contract_amendment", "logical_work_id": lwid, "kind": kind, "commit": commit, "commit_resolved": commit_full, "tip": tip_full, "old_sha": old_sha,
             "old_export_sha256": hashes["old_export_sha256"], "new_export_sha256": hashes["new_export_sha256"],
             "manifest_rel": hashes.get("manifest_rel"), "old_manifest_sha256": hashes.get("old_manifest_sha256"), "new_manifest_sha256": hashes.get("new_manifest_sha256"),
             "roster_rel": hashes.get("roster_rel"), "old_roster_sha256": hashes.get("old_roster_sha256"), "new_roster_sha256": hashes.get("new_roster_sha256"),
             "old_seed_tree_sha256": hashes.get("old_seed_tree_sha256"), "new_seed_tree_sha256": hashes.get("new_seed_tree_sha256"),
             "old_contract_sha256": entry["old_contract_sha256"], "new_contract_sha256": entry["new_contract_sha256"], "reason": reason,
             "changed_instruments": changed, "row_diff": allowed, "role_diff": role_diff, "live_jobs": live_jobs, "note": note, "sha_advance": entry["sha_advance"],
             "follow_up_of": entry["follow_up_of"],
             "carried_run_ids": entry["carried_run_ids"], "carried_completed": entry["carried_completed"],
             "old_jobs": entry["old_jobs"], "old_run_ids": lines, "completed_run_check": entry["completed_run_check"], "utc": entry["utc"]})
        doc = read_provenance_doc(ledger)
        doc["amendments"].setdefault(lwid, []).append(entry)
        write_atomic(provenance_path(ledger), doc)
        if kind != "decode-neutral":
            # the receipted sha ADVANCE: no funnel attempt is minted — the drive restarts at the amending commit against the same
            # serving job — so the receipt is the one path that moves the unit's sha; the ledger event first, the sidecar entry, then the
            # wave's per-unit sha table
            prov = {"sha": commit_full, "kind": "amend", "attempt_id": None, "job_id": None, "cluster": None,
                    "answers": {"amendment_utc": entry["utc"], "kind": kind, "commit": commit_full}, "fix_commit": commit_full,
                    "contract_sha256": entry["new_contract_sha256"], "utc": entry["utc"]}
            GL.Ledger(os.path.join(os.path.dirname(os.path.abspath(ledger)), "attempts.jsonl")).append(
                {"event": "sha_provenance", "logical_work_id": lwid, **prov, "from_sha": old_sha})
            record_sha_provenance(ledger, lwid, prov)
            entry["wave_files_refreshed"] = refresh_wave_unit_shas(ledger)
    return entry


def recheck_amendments(ledger: str, lwid: str, *, repo: str, tip: str, registry_db: Optional[str], sacct_tasks) -> List[dict]:
    """Re-run the three proofs over EVERY recorded receipt of `lwid`, writing nothing: per receipt {index, commit, utc, verdict PASS|FAIL,
    problems, …}. The old tree is the receipt's `old_sha` when it recorded one, else derived as the mint would have (the previous receipt's
    commit; the latest sha_provenance at or before the receipt; the freeze sha) — so receipts minted before the tool was enforceable are
    re-validated under it. The old jobs and runs are the ledger's events BEFORE the receipt plus what the receipt listed; the registry is
    read now, so an old run that has since completed FAILS the receipt."""
    inv = load_inventory(ledger)
    unit = inventory_unit(inv, lwid)
    if unit is None:
        raise ValueError(f"{lwid}: not a unit of the frozen inventory at {inventory_path(ledger)}")
    entries = read_amendments(ledger).get(lwid, [])
    ledger_probs = [p for p in reconcile_provenance(ledger) if p.startswith(f"{lwid}:")]
    tip_full = git_rev(repo, tip)
    out: List[dict] = []
    for k, e in enumerate(entries):
        problems: List[str] = list(ledger_probs)
        res = {"logical_work_id": lwid, "index": k, "commit": e.get("commit"), "utc": e.get("utc"), "tip": tip_full}
        try:
            commit_full = git_rev(repo, str(e.get("commit_resolved") or e.get("commit") or ""))
            res["commit_resolved"] = commit_full
            if not git_is_ancestor(repo, commit_full, tip_full):
                problems.append(f"commit {commit_full[:12]} is not an ancestor-or-equal of the tip {tip_full[:12]}")
            # the old tree is DERIVED for every receipt — from the ledger's recorded sha / the prior receipt / the freeze sha, never taken
            # from the sidecar: a repaired or copied receipt carrying a wrong old_sha with a matching export hash would otherwise pass
            # while omitting a real non-neutral old -> new change (Medium)
            old_sha, old_src = amendment_old_sha(ledger, unit, entries[:k], before_utc=e.get("utc"))
            res["old_sha"] = old_sha; res["old_sha_source"] = old_src
            if e.get("old_sha") and str(e["old_sha"]) != old_sha:
                problems.append(f"the receipt's old sha {str(e['old_sha'])[:12]} is not the derivation {old_sha[:12]} ({old_src}) — the ledger is the truth; "
                                f"the receipt is not this unit's")
            old_export, new_export, hashes = _derive_exports(repo, old_sha, commit_full, None)
            res.update(hashes)
            for hk in ("old_export_sha256", "new_export_sha256"):
                if e.get(hk) and e[hk] != hashes[hk]:
                    problems.append(f"the receipt's {hk} {e[hk][:12]} is not the export at {(old_sha if hk.startswith('old') else commit_full)[:12]} ({hashes[hk][:12]})")
            old_c = dict(entries[k - 1]["contract"]) if k > 0 and isinstance(entries[k - 1].get("contract"), dict) else unit_contract(unit)
            if contract_dict_sha256(old_c) != e.get("old_contract_sha256"):
                problems.append(f"the receipt's old contract hash {str(e.get('old_contract_sha256'))[:12]} is not the effective contract before it ({contract_dict_sha256(old_c)[:12]})")
            new_c = contract_at_commit(unit, repo, commit_full)
            if contract_dict_sha256(new_c) != e.get("new_contract_sha256"):
                problems.append(f"the receipt's new contract hash {str(e.get('new_contract_sha256'))[:12]} is not the contract the seeds at {commit_full[:12]} mint ({contract_dict_sha256(new_c)[:12]})")
            kind = str(e.get("kind") or "decode-neutral")
            res["kind"] = kind
            try:
                if kind == "petri-role-fallback":
                    rel = str(e.get("manifest_rel") or petri_manifest_rel(unit))
                    role_diff, rprobs, mh = petri_role_diff(repo, old_sha, commit_full, rel)
                    res["role_diff"] = [d["field"] for d in role_diff]; res.update(mh)
                    problems.extend(rprobs)
                    for hk in ("old_manifest_sha256", "new_manifest_sha256"):
                        if e.get(hk) and e[hk] != mh[hk]:
                            problems.append(f"the receipt's {hk} {str(e[hk])[:12]} is not the manifest at {(old_sha if hk.startswith('old') else commit_full)[:12]} ({mh[hk][:12]})")
                    changed = {ev: {"old": old_c["evals"][ev].get("instrument_id"), "new": new_c["evals"][ev].get("instrument_id")}
                               for ev in sorted(old_c["evals"]) if old_c["evals"][ev].get("instrument_id") != new_c["evals"][ev].get("instrument_id")}
                elif kind == "code-follow-up":
                    rel = str(e.get("manifest_rel") or petri_manifest_rel(unit))
                    fprobs, fh = code_follow_up_diff(repo, old_sha, commit_full, rel)
                    res.update(fh); res["role_diff"] = []
                    old_seed, _ = seed_tree_digest(repo, old_sha); new_seed, _ = seed_tree_digest(repo, commit_full)
                    fh = dict(fh, old_seed_tree_sha256=old_seed, new_seed_tree_sha256=new_seed); res.update({"old_seed_tree_sha256": old_seed, "new_seed_tree_sha256": new_seed})
                    if old_seed != new_seed:
                        fprobs.append(f"{SEED_TREE_REL}: the committed seed tree moved across the follow-up ({seed_tree_changes(repo, old_sha, commit_full)[:8]}) — not code-only")
                    if commit_full == old_sha or not git_is_ancestor(repo, old_sha, commit_full):
                        fprobs.append(f"commit {commit_full[:12]} is not a strict descendant of the amended sha {old_sha[:12]} it follows")
                    if hashes["old_export_sha256"] != hashes["new_export_sha256"]:
                        fprobs.append("the decode-policy export moved across the follow-up — not code-only")
                    if contract_dict_sha256(old_c) != contract_dict_sha256(new_c):
                        fprobs.append("the contract moved across the follow-up — not code-only")
                    if k == 0 or str(entries[k - 1].get("kind")) not in ("petri-role-fallback", "code-follow-up"):
                        fprobs.append("a code-follow-up receipt with no prior receipted sha advance to follow")
                    problems.extend(fprobs)
                    for hk in ("old_manifest_sha256", "new_manifest_sha256", "old_roster_sha256", "new_roster_sha256", "old_seed_tree_sha256", "new_seed_tree_sha256"):
                        if e.get(hk) and fh.get(hk) and e[hk] != fh[hk]:
                            problems.append(f"the receipt's {hk} {str(e[hk])[:12]} is not the tree's ({fh[hk][:12]})")
                    changed = {}
                else:
                    ref, changed, allowed, _ = _amendment_row_diff(unit, old_c, new_c, old_export, new_export)
                    res["decode_policy_ref"] = ref; res["row_diff"] = [c["path"] for c in allowed]
                if changed != (e.get("changed_instruments") or {}):
                    problems.append(f"the receipt's changed instruments {e.get('changed_instruments')} are not the trees' {changed}")
            except ValueError as exc:
                problems.append(str(exc))
            before = e.get("utc")
            jobs = {(j["cluster"], j["job_id"]): j for j in unit_old_jobs(ledger, lwid, before_utc=before)}
            for j in e.get("old_jobs") or []:
                jobs.setdefault((str(j.get("cluster")), str(j.get("job_id"))), {"cluster": str(j.get("cluster")), "job_id": str(j.get("job_id")), "source": "receipt"})
            runs = set(unit_old_run_ids(ledger, lwid, before_utc=before)) | {str(r.get("run_id")) for r in (e.get("old_run_ids") or []) if r.get("run_id")}
            try:
                proofs = prove_old_jobs_terminal(list(jobs.values()), sacct_tasks) if kind == "decode-neutral" else read_old_jobs_states(list(jobs.values()), sacct_tasks)
            except ValueError as exc:
                problems.append(str(exc)); proofs = {}
            old_job_forms = _job_id_forms(list(jobs.values()), proofs) if jobs else []
            by_job = registry_runs_of_jobs(registry_db, old_job_forms) if jobs else {}
            res["n_runs_from_registry_by_job"] = len(set(by_job) - runs)
            # 32c-d: the live cluster_job_id is the fold's last writer — a late old-job registration re-emitted under a fresh
            # job is invisible to the live read; the immutable registration history under every old-job spelling joins the inventory
            by_history = registry_runs_of_jobs_history(registry_db, old_job_forms) if jobs else {}
            res["n_runs_from_history_by_job"] = len(set(by_history) - runs - set(by_job))
            runs = sorted(runs | set(by_job) | set(by_history))
            res["n_old_jobs"] = len(jobs); res["n_old_runs"] = len(runs)
            res["registry_read_utc"] = utc_now()
            rows = registry_run_rows(registry_db, runs) if runs else {}
            if kind == "decode-neutral":
                lines, run_probs = old_run_verdicts(runs, rows, unit_run_jobs(ledger, lwid, before_utc=before), proofs)
            else:
                amended = {commit_full} | {str(x.get("commit_resolved")) for x in entries if x.get("commit_resolved")}
                carried_set = {str(x) for x in (e.get("carried_run_ids") or [])}
                known_set = {str(x.get("run_id")) for x in (e.get("old_run_ids") or []) if x.get("run_id")}
                absent_ids = [r for r in runs if r not in carried_set and r not in known_set]
                # 32c-b: a run the receipt never listed is classified by its ORIGINAL registration (row_history), against the receipt's stamp
                first_regs = registry_run_first_registration(registry_db, absent_ids) if absent_ids else {}
                res["n_runs_absent_from_receipt"] = len(absent_ids)
                lines, run_probs = petri_run_verdicts(runs, rows, carried=not changed, carried_ids=carried_set, amended_shas=amended, known_ids=known_set,
                                                      first_regs=first_regs, receipt_utc=e.get("utc"), old_job_ids=set(old_job_forms))
            res["old_run_ids"] = lines; res["live_jobs"] = [f"{p['job_id']}@{p['cluster']}:{','.join(p['states'])}" for p in proofs.values() if p.get("live")]
            if proofs or not jobs:
                problems.extend(run_probs)
            else:   # the job proofs failed: every non-terminal run is unproven by construction; report the completed ones by name only
                problems.extend(p for p in run_probs if "COMPLETED under the old contract" in p)
        except ValueError as exc:
            problems.append(str(exc))
        res["problems"] = problems; res["verdict"] = "PASS" if not problems else "FAIL"
        out.append(res)
    return out


def _ledger_events(ledger: str) -> List[dict]:
    att = os.path.join(os.path.dirname(os.path.abspath(ledger)), "attempts.jsonl")
    out: List[dict] = []
    if not os.path.isfile(att):
        return out
    for ln in open(att, encoding="utf-8"):
        try:
            out.append(json.loads(ln))
        except json.JSONDecodeError:
            continue
    return out


def ledger_sha_provenance(ledger: str) -> Dict[str, List[dict]]:
    """The per-unit sha provenance as the APPEND-ONLY attempts ledger records it (`sha_provenance` events, one per launched attempt of an
    inventory unit) — the truth the sidecar file must agree with (Medium)."""
    out: Dict[str, List[dict]] = {}
    for r in _ledger_events(ledger):
        if r.get("event") == "sha_provenance" and r.get("logical_work_id"):
            out.setdefault(r["logical_work_id"], []).append({"sha": r.get("sha"), "attempt_id": r.get("attempt_id"), "job_id": r.get("job_id"),
                                                             "cluster": r.get("cluster"), "kind": r.get("kind"), "contract_sha256": r.get("contract_sha256"), "utc": r.get("utc")})
    return out


def ledger_amendments(ledger: str) -> Dict[str, List[dict]]:
    out: Dict[str, List[dict]] = {}
    for r in _ledger_events(ledger):
        if r.get("event") == "contract_amendment" and r.get("logical_work_id"):
            # every proof field the event carries rides along (the reconciler compares the sidecar against ALL of them); a legacy
            # event (pre-18b) carries only the commit and the two contract hashes
            out.setdefault(r["logical_work_id"], []).append({**{k: r[k] for k in AMENDMENT_PROOF_FIELDS if k in r}, "utc": r.get("utc")})
    return out


def reconcile_provenance(ledger: str) -> List[str]:
    """Every entry of the provenance SIDECAR must be the ledger's: the same (sha, attempt, job) sequence per unit and the same amendment
    (commit, old -> new hash) sequence — a repaired or copied sidecar with a foreign sha or a foreign receipt is refused by name."""
    probs: List[str] = []
    side, truth = read_provenance(ledger), ledger_sha_provenance(ledger)
    for lw in sorted(set(side) | set(truth)):
        a = [(e.get("sha"), e.get("attempt_id"), str(e.get("job_id"))) for e in side.get(lw, [])]
        b = [(e.get("sha"), e.get("attempt_id"), str(e.get("job_id"))) for e in truth.get(lw, [])]
        if a != b:
            probs.append(f"{lw}: the provenance sidecar's sha entries {a[-3:]} disagree with the attempts ledger's {b[-3:]} — the ledger is the truth; the sidecar is not this wave's")
    sa, ta = read_amendments(ledger), ledger_amendments(ledger)
    for lw in sorted(set(sa) | set(ta)):
        side_e, truth_e = sa.get(lw, []), ta.get(lw, [])
        if len(side_e) != len(truth_e):
            probs.append(f"{lw}: the sidecar records {len(side_e)} contract amendment(s), the attempts ledger {len(truth_e)} — a receipt is a ledger event first; "
                         f"the sidecar is not this wave's")
            continue
        bad = []
        for i, (x, y) in enumerate(zip(side_e, truth_e)):
            # the LEDGER is the truth: every proof field its event carries must read the same in the sidecar (a receipt minted before a
            # field existed carries it on neither side; a sidecar field the ledger never recorded is verified by --recheck's derivation)
            diff = {k: (x.get(k), y.get(k)) for k in AMENDMENT_PROOF_FIELDS if k in y and _pf(x.get(k)) != _pf(y.get(k))}
            if diff:
                bad.append((i, diff))
        if bad:
            probs.append(f"{lw}: the sidecar's contract amendments disagree with the attempts ledger's at {bad} — a receipt is a ledger event first; "
                         f"the sidecar is not this wave's")
    return probs


#: every proof field a receipt records (the mint writes them to the ledger event FIRST, then the sidecar): the sidecar must agree with the
#: ledger on ALL of them, not only the (commit, old, new) contract triple — a receipt with a wrong old_sha and a matching export hash is
#: refused here and by --recheck's derivation (Medium). Fields absent on both sides compare equal (pre-fix receipts).
AMENDMENT_PROOF_FIELDS = ("commit", "commit_resolved", "old_sha", "old_export_sha256", "new_export_sha256", "old_contract_sha256", "new_contract_sha256",
                          "row_diff", "changed_instruments", "kind", "manifest_rel", "old_manifest_sha256", "new_manifest_sha256",
                          "roster_rel", "old_roster_sha256", "new_roster_sha256", "old_seed_tree_sha256", "new_seed_tree_sha256")


def _pf(v):
    return json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else v


#: workload-defining spec fields: what the plan composed for the job, beyond the registry contract — the script, its positional args (the
#: wave sha token normalised), every env knob that reaches the job (seed base, windows, policy ref, registry evals), the array and group
SPEC_IDENTITY_KEYS = ("script", "args", "env", "array", "group", "kind")
#: env keys whose change between attempts is the RECEIPTED spelling of the same decode-policy row: unversioned ref -> `…@vN`
SPEC_REF_ENV_KEYS = ("F22_MT_TARGET_POLICY_REF",)


#: per-submission stamps that ride the job env but are NOT workload knobs: dropped from the spec identity, so a re-run never
#: differs from its first attempt by them (the funnel exports SISTEER_UNIT_SUBMITTED_UTC on every sbatch)
SPEC_ENV_STAMPS = ("SISTEER_UNIT_SUBMITTED_UTC",)


def spec_identity(spec: Mapping[str, object]) -> dict:
    sha = str(spec.get("sha") or "")
    args = [("{sha}" if (sha and str(a) == sha) else str(a)) for a in (spec.get("args") or [])]
    env = {k: v for k, v in dict(spec.get("env") or {}).items() if k not in SPEC_ENV_STAMPS}
    return {"script": spec.get("script"), "args": args, "env": env, "array": spec.get("array"), "group": spec.get("group"), "kind": spec.get("kind", "target")}


def unit_first_spec(ledger: str, lwid: str) -> Optional[dict]:
    """The workload spec of the unit's FIRST recorded attempt of any tree (units OUTSIDE the frozen inventory — pools, fits — have no
    freeze-time composition, so their baseline is what they first ran as), or None. Inventory units use `workload_baseline`."""
    for r in _ledger_events(ledger):
        if r.get("event") == "attempt" and r.get("logical_work_id") == lwid and isinstance(r.get("spec"), dict):
            return spec_identity(r["spec"])
    return None


def unit_first_frozen_attempt(ledger: str, lwid: str, frozen_sha: str) -> Optional[dict]:
    """The unit's first recorded attempt AT THE FROZEN TREE (spec.sha == the inventory's code_sha) as {identity, job_id, cluster, utc, sha},
    or None. Attempts that predate the freeze are provisional history (proving passes at an earlier tree), never a baseline."""
    for r in _ledger_events(ledger):
        if r.get("event") == "attempt" and r.get("logical_work_id") == lwid and isinstance(r.get("spec"), dict) and (r.get("sha") or r["spec"].get("sha")) == frozen_sha:
            return {"identity": spec_identity(r["spec"]), "job_id": str(r.get("job_id")), "cluster": r.get("cluster"), "utc": r.get("utc"), "sha": frozen_sha}
    return None


#: the unit's composed WORKLOAD identity the freeze persists on every inventory unit (commit 26): script,
#: args with the sha token, the plan's env knobs, array, group, kind — read back by frozen_composition, never re-derived from today's plan
WORKLOAD_FIELDS = ("script", "args", "env", "array", "group", "kind")


def workload_identity_of_row(row: Mapping[str, object], sha: str) -> dict:
    """A plan row's workload identity: `spec_identity` over its script / args (the sha spelled as the `{sha}` token) / env / array / group / kind."""
    return spec_identity({"script": row["script"], "args": [str(a) for a in (row.get("args") or [])], "env": dict(row.get("env") or {}), "sha": sha,
                          "array": row.get("array"), "group": row.get("group"), "kind": row.get("kind", "target")})


def repo_root_of_lib() -> str:
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def frozen_tree_dir(ledger: str, code_sha: str) -> str:
    return os.path.join(os.path.dirname(os.path.abspath(inventory_path(ledger))), f"frozen_tree_{code_sha[:12]}")


def materialize_frozen_tree(ledger: str, code_sha: str, repo: str) -> str:
    """`git archive <code_sha>` of the repo into `<gate>/frozen_tree_<sha12>/` ONCE (a marker file names the full sha); refuses, by name, a
    sha the repo does not hold. The frozen tree is what a legacy inventory's composition is derived from — never the working tree."""
    d = frozen_tree_dir(ledger, code_sha)
    marker = os.path.join(d, ".frozen_tree_sha")
    if os.path.isfile(marker) and open(marker).read().strip() == code_sha and os.path.isfile(os.path.join(d, "reproduce", "lib", "stage.py")):
        return d
    chk = subprocess.run(["git", "-C", repo, "cat-file", "-e", f"{code_sha}^{{commit}}"], capture_output=True, text=True)
    if chk.returncode != 0:
        raise ValueError(f"the frozen tree {code_sha[:12]} is not a commit of {repo} — the freeze-time composition cannot be derived (the working tree "
                         f"is never the fallback)")
    tmp = d + ".tmp"
    shutil.rmtree(tmp, ignore_errors=True); os.makedirs(tmp)
    ar = subprocess.Popen(["git", "-C", repo, "archive", "--format=tar", code_sha], stdout=subprocess.PIPE)
    un = subprocess.run(["tar", "-x", "-C", tmp], stdin=ar.stdout, capture_output=True, text=True)
    ar.stdout.close(); ar.wait()
    if ar.returncode != 0 or un.returncode != 0:
        shutil.rmtree(tmp, ignore_errors=True)
        raise ValueError(f"git archive {code_sha[:12]} failed (rc {ar.returncode}/{un.returncode}): {un.stderr.strip()[:200]}")
    with open(os.path.join(tmp, ".frozen_tree_sha"), "w") as fh:
        fh.write(code_sha + "\n")
    shutil.rmtree(d, ignore_errors=True); os.replace(tmp, d)
    return d


#: the one label a legacy unit is planned over when the wave's placement is unknown: a single-GPU (`*`) unit composes the same under any
#: label, and a cluster-placed unit then refuses by name (commit 26b) — the CALLER's cluster list is never the source
UNPLACED_LABEL = "unplaced"


def pinned_placement_labels(ledger: str, lwid: str, code_sha: str, view: Optional[str] = None) -> Tuple[Optional[List[str]], str, Optional[dict]]:
    """The ordered cluster labels a LEGACY inventory unit's freeze-time composition is planned over — from the WAVE, never the caller
    (commit 26b): the pinned placement manifest's `labels` — the INVENTORY'S VIEW's file, the one the freeze pins
    (`placement_path(ledger, view)`: `gate/expected_units.placement.json` for u0, the per-view `expected_units.placement.<view>.json` for
    the single-phase recipe's one wave `s<N>`; a malformed one refuses by name) — or, failing that, the unit's first attempt AT THE FROZEN
    TREE (that attempt's cluster; the group it ran under is verified against the plan afterwards). (labels | None, where they came from,
    the attempt when it was the source). `view` None = the u0 historic file (a caller without an inventory)."""
    p = placement_path(ledger, view)
    man = read_placement(p)
    if man is not None and man.get("labels"):
        return [str(x) for x in man["labels"]], f"the pinned placement manifest {p}", None
    att = unit_first_frozen_attempt(ledger, lwid, code_sha)
    if att is not None and att.get("cluster"):
        return [str(att["cluster"])], f"attempt evidence (job {att['job_id']} on {att['cluster']}, {att['utc']})", att
    return None, f"neither a pinned placement manifest nor attempt evidence at the frozen tree {code_sha[:12]} (looked for {p})", None


#: the plan knobs a composition depends on that a pre-26c inventory does NOT pin, each with the plan-row env key it leaves behind — the
#: evidence a recorded attempt carries. A frozen tree's DEFAULT for one of these is never asserted as the freeze's setting: a wave frozen
#: with SISTEER_MT_SEED_BASE=20261234 would have had its unchanged retry refused against the invented default. Since 26c the freeze pins
#: them in the inventory header (`plan_knobs`); before that, the first frozen-tree attempt is the evidence; with neither, a unit whose row
#: carries the key is unknowable, refused by name. (The models table is NOT in this list: no plan row field depends on it —
#: tests/test_frozen_composition_inputs.py proves a mutated served id leaves the composition byte-identical.)
#: The smoke goal-shard knob (OVERRIDE_SMOKE_GOALS, M10) has NO entry here on purpose: it was born after 26c, so every inventory that
#: carries it pins it in `plan_knobs` (and persists each unit's `workload`); no pre-26c inventory can carry it, and its value is the
#: overrides document's, never an environment key an attempt's spec env could stand in for.
PLAN_KNOB_ENV_KEYS: Dict[str, str] = {"SISTEER_MT_SEED_BASE": "F22_MT_TARGET_SEED_BASE"}


def freeze_plan_knobs(smoke: Optional[Mapping[str, object]] = None) -> Dict[str, Optional[str]]:
    """The plan knobs THIS freeze runs with, as the plan reads them (unit_plan's env defaults; the smoke goal-shard knob as the model's
    overrides document sets it — `smoke`, :func:`smoke_goals`' reading, None when unset: exactly None, as SISTEER_MT_ATTACKER_GPUS is, so an
    inventory header frozen before the knob existed still plans under `plan --inventory`), pinned into the inventory header."""
    return {"SISTEER_MT_SEED_BASE": os.environ.get("SISTEER_MT_SEED_BASE", "20260909"),
            "SISTEER_MT_ATTACKER_GPUS": os.environ.get("SISTEER_MT_ATTACKER_GPUS") or None,
            "SISTEER_SHARD_GROUPS": os.environ.get("SISTEER_SHARD_GROUPS") or " ".join(DEFAULT_SHARD_GROUPS),
            SMOKE_GOALS_KEY: (str(smoke["spec"]) if smoke else None)}


def smoke_goals_header(smoke: Optional[Mapping[str, object]]) -> Optional[dict]:
    """The inventory header's record of the smoke goal-shard knob's PROVENANCE beside its pinned value: the range, the shard total, the
    shard names, and the overrides document it was read from (id, basename, sha256, the model it names) — None when no knob is set."""
    if not smoke:
        return None
    return {k: (list(smoke[k]) if k == "shards" else smoke[k]) for k in ("spec", "lo", "hi", "nshards", "shards", "override_id", "overrides_path", "overrides_sha256", "model")}


def recovered_plan_knobs(inv: dict, att: Optional[dict]) -> Tuple[Dict[str, str], str]:
    """(the knobs to plan the frozen tree with, where they came from): the inventory's pinned `plan_knobs` (a 26c+ freeze), else each knob
    the unit's FIRST frozen-tree attempt recorded (its spec env's PLAN_KNOB_ENV_KEYS value), else none — the caller refuses a row that
    depends on a knob no evidence names."""
    pinned = inv.get("plan_knobs")
    if isinstance(pinned, dict) and any(k in pinned for k in PLAN_KNOB_ENV_KEYS):
        return {k: str(pinned[k]) for k in PLAN_KNOB_ENV_KEYS if pinned.get(k) not in (None, "")}, "the inventory's pinned plan_knobs"
    if att is not None:
        env = (att.get("identity") or {}).get("env") or {}
        found = {k: str(env[ek]) for k, ek in PLAN_KNOB_ENV_KEYS.items() if env.get(ek) not in (None, "")}
        if found:
            return found, f"the unit's first frozen-tree attempt (job {att.get('job_id')} on {att.get('cluster')}, {att.get('utc')})"
    return {}, "no evidence"


def frozen_tree_plan_row(tree: str, model_key: str, view: str, fam: Optional[str], nshards: int, code_sha: str, labels: List[str],
                         table: Optional[str], lwid: str, *, shard_groups: Optional[List[List[int]]] = None, knobs: Optional[Mapping[str, str]] = None) -> Optional[dict]:
    """The unit's plan row as THE FROZEN TREE's own `stage.py plan --json` composes it, from the frozen wave's inputs ONLY (commit 26b):
    the tree's OWN models table (`reproduce/models.yaml` inside the archive — `table` is accepted for the signature and never used), the
    pinned cluster order, the inventory's pinned shard groups, and the tree's own defaults for every other plan knob — the inherited
    `SISTEER_*` environment (today's seed base, shard groups, attacker cap, models-table override) is scrubbed from the subprocess."""
    frozen_table = os.path.join(tree, "reproduce", "models.yaml")
    if not os.path.isfile(frozen_table):
        raise ValueError(f"the frozen tree {code_sha[:12]} carries no reproduce/models.yaml — its composition cannot be derived")
    cmd = [sys.executable, os.path.join(tree, "reproduce", "lib", "stage.py"), "plan", "--model", model_key, "--view", view, "--nshards", str(nshards),
           "--sha", code_sha, "--clusters", ",".join(labels), "--models-table", frozen_table, "--json"]
    if fam:
        cmd += ["--families", fam]
    env = {k: v for k, v in os.environ.items() if not k.startswith("SISTEER_")}
    env["PYTHONPATH"] = os.path.join(tree, "src") + (os.pathsep + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else "")
    if shard_groups:
        env["SISTEER_SHARD_GROUPS"] = " ".join(",".join(str(i) for i in g) for g in shard_groups)       # the inventory's PINNED groups, not today's
    for k, v in (knobs or {}).items():
        env[k] = str(v)                                                                                 # the RECOVERED freeze-time knobs (26c), never today's
    r = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=tree)
    if r.returncode != 0:
        raise ValueError(f"the frozen tree {code_sha[:12]}'s plan refused for {lwid}: {r.stderr.strip()[-400:]}")
    rows = json.loads(r.stdout)
    return next((x for x in rows if x.get("logical_work_id") == lwid), None)


def frozen_composition(ledger: str, lwid: str, *, table: Optional[str], labels: List[str], repo: Optional[str] = None) -> Optional[dict]:
    """The FREEZE-TIME COMPOSITION of an inventory unit's workload, from the FROZEN wave — never today's working tree:
    the `workload` identity the freeze persisted on the unit (commit 26) when present, else — a legacy inventory — the unit's
    plan row as the FROZEN git tree's own `stage.py plan --json` composes it (`git archive <code_sha>` materialised once beside the
    inventory) from the frozen wave's inputs only (commit 26b): the tree's own models table (a model it does not
    know REFUSES by name — never None), the cluster order of the pinned placement manifest or, failing that, of the unit's first
    frozen-tree attempt, and the inventory's pinned shard groups; `table` and `labels` (today's) are never a source. A cluster-placed unit
    whose placement neither a manifest nor an attempt records refuses by name; a single-GPU (`*`) unit composes regardless. None when the
    unit is not in the inventory (pools, fits) or the plan has no row for it. The identity's env carries the PLAN's knobs only; the site
    plumbing the stage adds (`slurm_base_env`) is not part of the composition. `baseline.source` names which: `inventory` or
    `frozen-tree <sha12>` (then `baseline.labels` / `labels_source` say where the cluster order came from)."""
    if not os.path.isfile(inventory_path(ledger)):
        return None
    inv = load_inventory(ledger)
    unit = inventory_unit(inv, lwid)
    if unit is None or not inv.get("code_sha"):
        return None
    code_sha = str(inv["code_sha"])
    wl = unit.get("workload")
    if isinstance(wl, dict) and all(k in wl for k in WORKLOAD_FIELDS):
        ident = {k: (dict(wl[k]) if k == "env" else (list(wl[k]) if k == "args" else wl[k])) for k in WORKLOAD_FIELDS}
        ident["baseline"] = {"kind": "composition", "sha": code_sha, "source": "inventory"}
        return ident
    tree = materialize_frozen_tree(ledger, code_sha, repo or repo_root_of_lib())
    frozen_table = os.path.join(tree, "reproduce", "models.yaml")
    try:
        M.model(str(inv["model"]), frozen_table)
    except KeyError as exc:
        raise ValueError(f"{inv['model']} is not in the frozen tree's models table ({code_sha[:12]}: reproduce/models.yaml) — the freeze-time composition of "
                         f"{lwid} cannot be derived; a model the wave's own tree does not know is a refusal, never 'no baseline'") from exc
    except (ValueError, OSError) as exc:
        raise ValueError(f"the frozen tree {code_sha[:12]}'s models table (reproduce/models.yaml) is unreadable: {exc}") from exc
    fam = unit.get("stage_family") or unit.get("family")
    view = str(inv.get("view") or "u0")
    # the universe comes from the manifest the FREEZE pinned, which is the inventory's view's file (cmd_freeze: `placement_path(out,
    # baseline view)` — u0's historic name, or the single-phase wave's per-view file); reading the u0 name for a battery-baseline wave found
    # nothing and refused every placed unit as unknowable (tests/test_frozen_composition_inputs.py pins the namespace)
    labels_used, labels_src, att = pinned_placement_labels(ledger, lwid, code_sha, view)
    first_att = att or unit_first_frozen_attempt(ledger, lwid, code_sha)
    knobs, knobs_src = recovered_plan_knobs(inv, first_att)
    row = frozen_tree_plan_row(tree, str(inv["model"]), view, fam, int(inv.get("nshards") or 9), code_sha,
                               labels_used or [UNPLACED_LABEL], None, lwid, shard_groups=inv.get("shard_groups"), knobs=knobs)
    if row is None:
        return None
    # a row that depends on a knob NO evidence names is unknowable: the tree's default is not the freeze's setting (26c)
    unknowable = [(k, ek) for k, ek in PLAN_KNOB_ENV_KEYS.items() if ek in (row.get("env") or {}) and k not in knobs]
    if unknowable:
        raise ValueError(f"{lwid}: the freeze-time plan knob(s) {[k for k, _ in unknowable]} are unknowable — the inventory (frozen before 26c) does not pin them and no "
                         f"attempt at the frozen tree {code_sha[:12]} recorded {[ek for _, ek in unknowable]}; the frozen tree's default is never asserted as the freeze's "
                         f"setting")
    if row.get("cluster") != "*":
        if labels_used is None:
            raise ValueError(f"{lwid}: the placement universe is unknowable — {labels_src} names the wave's clusters, and the caller's cluster list "
                             f"{list(labels)} is never inherited (a placed unit's group follows the pinned order; `stage.py placement-manifest ensure` pins it)")
        if att is not None and row.get("group") != att["identity"].get("group"):
            raise ValueError(f"{lwid}: planned over the attempt's cluster {labels_used[0]} the frozen tree composes group {row.get('group')}, but the attempt "
                             f"(job {att['job_id']}, {att['utc']}) ran under group {att['identity'].get('group')} — the placement cannot be reconstructed from "
                             f"attempt evidence; pin the placement manifest")
    ident = workload_identity_of_row(row, code_sha)
    ident["baseline"] = {"kind": "composition", "sha": code_sha, "source": f"frozen-tree {code_sha[:12]}", "labels": labels_used, "labels_source": labels_src,
                         "models_table": f"frozen-tree {code_sha[:12]} reproduce/models.yaml", "shard_groups": inv.get("shard_groups"),
                         "plan_knobs": dict(knobs), "plan_knobs_source": knobs_src}
    return ident

def workload_baseline(ledger: str, lwid: str, *, table: Optional[str], labels: List[str], repo: Optional[str] = None) -> Tuple[Optional[dict], List[str]]:
    """(baseline identity, consistency problems) for a submission's workload-spec proof. For an inventory unit the baseline is the
    freeze-time composition; its first frozen-tree attempt, when one exists, must EQUAL the composition on the plan's fields (script, args,
    array, group, kind, the plan's env knobs — the one tolerated difference is the receipted `@vN` re-spelling of a decode-policy ref), and
    supplies the site-plumbing env keys the composition does not carry. A unit with no frozen-tree attempt is compared against the
    composition alone. Pre-freeze attempts are never consulted. A unit outside the inventory falls back to its first attempt of any tree."""
    comp = frozen_composition(ledger, lwid, table=table, labels=labels, repo=repo)
    if comp is None:
        return unit_first_spec(ledger, lwid), []
    problems: List[str] = []
    att = unit_first_frozen_attempt(ledger, lwid, comp["baseline"]["sha"])
    base = {k: comp[k] for k in ("script", "args", "env", "array", "group", "kind")}
    if att is not None:
        a = att["identity"]
        plan_view = dict(a); plan_view["env"] = {k: v for k, v in (a.get("env") or {}).items() if k in (comp.get("env") or {})}
        d = spec_diff(base, plan_view)
        if d:
            problems.append(f"the unit's first frozen-tree attempt (job {att['job_id']} on {att['cluster']}, {att['utc']}, sha {att['sha'][:12]}) is not the freeze-time "
                            f"composition at {comp['baseline']['sha'][:12]}: " + "; ".join(d[:6]))
        base["env"] = {**{k: v for k, v in (a.get("env") or {}).items() if k not in base["env"]}, **base["env"]}
        base["baseline"] = {"kind": "composition+attempt", "sha": comp["baseline"]["sha"], "job_id": att["job_id"], "cluster": att["cluster"], "utc": att["utc"],
                            "source": comp["baseline"].get("source")}
    else:
        # no frozen-tree attempt yet: the composition alone binds — its own env knobs exactly; the site plumbing the stage adds
        # (slurm_base_env: campaign, roster, serving path) is outside the freeze and unconstrained until an attempt records it
        base["env_open"] = True
        base["baseline"] = {"kind": "composition", "sha": comp["baseline"]["sha"], "job_id": None, "source": comp["baseline"].get("source")}
    return base, problems


def describe_baseline(b: Optional[dict]) -> str:
    """One clause naming the baseline a workload refusal compared against."""
    if not b or not isinstance(b.get("baseline"), dict):
        return "the unit's first attempt"
    bb = b["baseline"]
    if bb.get("kind") == "composition+attempt":
        return f"the freeze-time composition at {str(bb.get('sha'))[:12]} (its first frozen-tree attempt = job {bb.get('job_id')} on {bb.get('cluster')}, {bb.get('utc')})"
    if bb.get("kind") == "composition":
        return f"the freeze-time composition at {str(bb.get('sha'))[:12]} (no attempt at the frozen tree yet)"
    return "the unit's first attempt"


def _same_policy_row(old: object, new: object) -> bool:
    o, n = str(old or ""), str(new or "")
    return bool(o) and bool(n) and o.split("@v", 1)[0] == n.split("@v", 1)[0]


def spec_diff(frozen: dict, current: dict) -> List[str]:
    """The workload-spec fields that differ between the unit's first attempt and this submission, by name. The one tolerated change:
    a decode-policy ref env knob re-spelled as the same row's versioned id."""
    out: List[str] = []
    for k in ("script", "array", "group", "kind"):
        if frozen.get(k) != current.get(k):
            out.append(f"{k}: {frozen.get(k)!r} -> {current.get(k)!r}")
    if list(frozen.get("args") or []) != list(current.get("args") or []):
        out.append(f"args: {frozen.get('args')} -> {current.get('args')}")
    fe, ce = frozen.get("env") or {}, current.get("env") or {}
    for k in sorted(set(fe) | set(ce)):
        if fe.get(k) == ce.get(k):
            continue
        if frozen.get("env_open") and k not in fe:
            continue                              # a composition-only baseline binds its own knobs; keys it does not carry are the stage's plumbing
        if k in SPEC_REF_ENV_KEYS and _same_policy_row(fe.get(k), ce.get(k)):
            continue
        out.append(f"env.{k}: {fe.get(k)!r} -> {ce.get(k)!r}")
    return out


def unit_recorded_sha(prov: Dict[str, List[dict]], unit: dict) -> Optional[str]:
    """The unit's current sha: the latest provenance entry's, else the sha it was frozen at."""
    entries = prov.get(unit.get("logical_work_id") or "", [])
    return (entries[-1].get("sha") if entries else None) or unit.get("code_sha")


def sha_table(ledger: str) -> Dict[str, dict]:
    """{lwid: {sha, frozen_sha, n_attempts_recorded, advanced}} — the wave artifact's per-unit sha table."""
    inv = load_inventory(ledger) or {}
    prov = read_provenance(ledger); amend = read_amendments(ledger)
    out = {}
    for u in inv.get("units", []):
        lw = u["logical_work_id"]; entries = prov.get(lw, [])
        out[lw] = {"sha": unit_recorded_sha(prov, u), "frozen_sha": u.get("code_sha"), "n_recorded": len(entries),
                   "advanced": any(e.get("kind") == "advance" for e in entries), "amended": len(amend.get(lw, []))}
    return out


def plan_knob_diff(pinned: Optional[Mapping[str, object]], live: Optional[Mapping[str, object]]) -> Dict[str, Tuple[object, object]]:
    """{knob: (pinned, live)} for every plan knob whose pinned value (a 26c+ inventory header) differs from the live one; {} when either side
    is absent (a pre-26c inventory pins nothing to compare)."""
    if not isinstance(pinned, dict) or not isinstance(live, dict):
        return {}
    return {k: (pinned.get(k), live.get(k)) for k in sorted(set(pinned) | set(live)) if pinned.get(k) != live.get(k)}


def refreeze_refusal(out_path: str, new_units: List[dict], new_sha: str, new_knobs: Optional[Mapping[str, object]] = None) -> Optional[str]:
    """The inventory is frozen ONCE. With an existing inventory whose units have recorded attempts in the state's ledger, a freeze
    that would change it (units, contracts, the tree, or — 26d — a pinned plan knob: the attacker cap is header
    metadata that registers no unit yet sizes the pools a later 06 pass submits) is refused by name — a unit's sha advances only through
    the funnel's re-run path. An identical inventory at the same sha and knobs is a no-op (the caller says `unchanged`); no attempts yet ->
    the freeze may change."""
    if not os.path.isfile(out_path):
        return None
    try:
        old = json.load(open(out_path, encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    state = os.path.dirname(os.path.dirname(os.path.abspath(out_path)))
    ledger = os.path.join(state, "attempts.jsonl")
    if not os.path.isfile(ledger):
        return None
    old_ids = {u["logical_work_id"] for u in old.get("units", [])}
    # 26d-fix: the wave has STARTED once ANY of its attempts exists — a unit's, or an attacker-pool member's (the pool
    # rows register no unit and are absent from the inventory, yet the plan knobs size them): a modular 06 pass that launched the capped pool
    # first, then a cap change + refreeze, used to take `not attempted` and rewrite the pinned cap, and `plan --inventory` then submitted
    # additional members for the same started wave. A pool member's lwid is `<model>/<view>/multiturn/<group>/attackers/<i>`.
    view = str(old.get("view") or "u0"); model = str(old.get("model") or "")
    pool_prefix = f"{model}/{view}/multiturn/" if model else f"/{view}/multiturn/"
    attempted = set()
    for ln in open(ledger, encoding="utf-8"):
        try:
            r = json.loads(ln)
        except json.JSONDecodeError:
            continue
        lw = str(r.get("logical_work_id") or "")
        if r.get("event") == "attempt" and (lw in old_ids or (("/attackers/" in lw) and (lw.startswith(pool_prefix) if model else pool_prefix in lw))):
            attempted.add(lw)
    if not attempted:
        return None
    same_units = old.get("units") == new_units
    kdiff = plan_knob_diff(old.get("plan_knobs"), new_knobs) if new_knobs is not None else {}
    if same_units and old.get("code_sha") == new_sha and not kdiff:
        return None
    what = []
    if kdiff:
        pool_only = sorted(lw for lw in attempted if lw not in old_ids)
        what.append(f"{len(kdiff)} plan knob(s) would change: " + ", ".join(f"{k} {p!r} -> {l!r}" for k, (p, l) in kdiff.items())
                    + " — a knob sizes or seeds the wave's execution (the attacker cap, the seed base, the shard grouping, a smoke's goal subset); a change is a new wave (26d)"
                    + (f"; the wave's attacker pool has already launched ({len(pool_only)} member attempt(s): {pool_only[:3]}{' …' if len(pool_only) > 3 else ''}) — "
                       f"a re-sized pool would add members with no consumers (26d-fix)" if pool_only and not (attempted & old_ids) else ""))
    if not same_units:
        oc = {u["logical_work_id"]: contract_sha256(u) for u in old.get("units", [])}
        nc = {u["logical_work_id"]: contract_sha256(u) for u in new_units}
        changed = sorted(lw for lw in set(oc) | set(nc) if oc.get(lw) != nc.get(lw))
        what.append(f"{len(changed)} unit contract(s) would change: {changed[:6]}{' …' if len(changed) > 6 else ''}")
    if old.get("code_sha") != new_sha:
        what.append(f"the tree would move {old.get('code_sha')} -> {new_sha}")
    return (f"the inventory is frozen ONCE (D200): {len(attempted)} of its units already have recorded attempts, and this freeze would "
            f"change it ({'; '.join(what)}). A unit's sha advances only through the funnel's re-run path (`submit --resubmit … --wave-sha <new>`, "
            f"which verifies the unit's contract at the new tree and records the provenance); a contract change is a new wave, not a re-freeze")


def safe_id(lwid: str) -> str:
    """A logical work id as a file-name stem (`/` -> `__`) — the bindings, coverage and gate files are keyed by it."""
    return lwid.replace("/", "__")


def submitted_record(lwid: str, job_id: str, cluster: str, attempt_id: str, sbatch_name: Optional[str] = None, array: Optional[str] = None) -> dict:
    """The `submitted` binding (unit -> scheduler attempt) — the ONE shape submit_twin.py and `bind --state submitted` write. `array`
    is the sbatch --array spec of an array unit: the binder then reads EVERY task's pending file."""
    return {"logical_work_id": lwid, "state": "submitted", "job_id": job_id, "cluster": cluster, "attempt_id": attempt_id,
            "sbatch_name": sbatch_name, "array": array, "utc": utc_now()}


def array_task_indices(spec: str) -> List[int]:
    """The task indices of an sbatch --array spec (`0-3`, `0,2,5`, `0-7:2`, with an optional `%N` throttle)."""
    out: List[int] = []
    for part in spec.split("%")[0].split(","):
        part = part.strip()
        if not part:
            continue
        step = 1
        if ":" in part:
            part, st = part.split(":", 1); step = int(st)
        if "-" in part:
            lo, hi = part.split("-", 1); out.extend(range(int(lo), int(hi) + 1, step))
        else:
            out.append(int(part))
    return sorted(set(out))


def parse_array_tasks(sacct_text: str) -> List[dict]:
    """`sacct -X -n -P -o JobIDRaw,JobID,State -j <array id>` -> [{raw, task, state}]: the RAW job id of every task (its pending
    file is pending/<raw>.json — SLURM_JOB_ID inside an array task) and its index from `JobID` (`<array>_<k>`); a non-array job
    yields its one row with task None."""
    out = []
    for ln in sacct_text.splitlines():
        p = ln.strip().split("|")
        if len(p) < 3 or not p[0]:
            continue
        raw, jid, state = p[0], p[1], p[2].split()[0] if p[2] else ""
        task = None
        if "_" in jid:
            t = jid.split("_", 1)[1]
            if t.startswith("[") or not t.isdigit():
                continue                                     # a pending `<array>_[2-3]` summary row: the tasks have no raw id yet
            task = int(t)
        out.append({"raw": raw, "task": task, "state": state})
    return out


def cmd_array_tasks(a) -> int:
    print(json.dumps(parse_array_tasks(arg_doc(a.sacct) or "")))
    return 0


def write_binding(ledger: str, rec: dict) -> str:
    """Write a `<state>` binding beside the ledger (tmp + rename); returns its path."""
    p = os.path.join(bindings_dir(ledger), f"{safe_id(rec['logical_work_id'])}.{rec['state']}.json")
    write_atomic(p, rec)
    return p


def retire_registered_binding(ledger: str, lwid: str, attempt_id: Optional[str]) -> Optional[str]:
    """Move the unit's `registered` binding aside — `<stem>.registered.retired.<attempt>.json` — so a NEW `submitted` binding is the one
    `binding_state` reads (a stale registered binding would otherwise shadow every later attempt). Returns the
    retired path, None when there was none. The record is kept, never deleted: it names the dead attempt's runs."""
    d = bindings_dir(ledger); safe = safe_id(lwid)
    p = os.path.join(d, f"{safe}.registered.json")
    if not os.path.isfile(p):
        return None
    tag = safe_id(str(attempt_id or "unknown")).replace("@", "_at_")
    dst = os.path.join(d, f"{safe}.registered.retired.{tag}.json")
    n = 1
    while os.path.exists(dst):
        n += 1; dst = os.path.join(d, f"{safe}.registered.retired.{tag}.{n}.json")
    os.replace(p, dst)
    return dst


def transition_path(ledger: str, lwid: str) -> str:
    return os.path.join(bindings_dir(ledger), f"{safe_id(lwid)}.transition.json")


def read_transition(ledger: str, lwid: str) -> Optional[dict]:
    """The unit's RE-RUN TRANSITION TOKEN when one is in flight (hole 3): `submit --resubmit` writes it BEFORE it
    retires the old binding or calls sbatch and clears it after the new attempt is bound, so an interruption anywhere in between leaves a
    durable record of what was intended; the next invocation reconciles it against Slurm by job name + submit time."""
    p = transition_path(ledger, lwid)
    if not os.path.isfile(p):
        return None
    try:
        return json.load(open(p, encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def write_transition(ledger: str, rec: dict) -> str:
    p = transition_path(ledger, rec["logical_work_id"])
    write_atomic(p, rec)
    return p


def clear_transition(ledger: str, lwid: str) -> None:
    p = transition_path(ledger, lwid)
    if os.path.isfile(p):
        os.replace(p, p + ".done." + utc_now().replace(":", ""))     # kept beside the bindings: the trail of what was intended and done


def restore_registered_binding(ledger: str, lwid: str, retired_path: Optional[str]) -> bool:
    """Undo `retire_registered_binding` when the re-run it made room for never launched (a crash before sbatch): the retired record
    moves back to `<stem>.registered.json` — unless a registered binding exists again (a later attempt bound), which stands."""
    dst = os.path.join(bindings_dir(ledger), f"{safe_id(lwid)}.registered.json")
    if not retired_path or not os.path.isfile(retired_path) or os.path.isfile(dst):
        return False
    os.replace(retired_path, dst)
    return True


def binding_state(ledger: str, lwid: str) -> dict:
    """The unit's current binding: {state: unbound|submitted|registered|transition, ...} — a torn tmp file never counts. `transition`
    (with the token under `transition`) while a re-run is mid-flight: the gate must not bind the superseded generation and the funnel
    must reconcile the token before anything else."""
    d = bindings_dir(ledger)
    safe = safe_id(lwid)
    st = {"state": "unbound", "logical_work_id": lwid}
    for state in ("submitted", "registered"):
        p = os.path.join(d, f"{safe}.{state}.json")
        if os.path.isfile(p):
            try:
                rec = json.load(open(p, encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            st.update(rec); st["state"] = state
    tok = read_transition(ledger, lwid)
    if tok:
        st["state"] = "transition"; st["transition"] = tok
    return st


def family_bound_run_ids(ledger: str, units, family: str) -> List[str]:
    """The run ids a FAMILY's units are bound to NOW — the ordered-unique union, in the inventory's unit order, of every unit's
    `binding_state(...)['run_ids']` (the registered binding's; a submitted or unbound unit contributes none). THE ONE RULE both sides
    read: `06 --gate <family> --headline … --range …` records this list as the verdict's `run_ids` (the runs
    the pass RULED), and lib/admission.py refuses the family when the standing verdict's `run_ids` differ AS A SET from this list at check time
    (a resubmission re-bound a unit since the ruling — the pass vouches for the runs it ruled, never for runs nobody ruled). A verdict is bound
    to RUNS, not to a coverage generation: a coverage refresh, a new probe sha or a re-read over the SAME runs changes nothing here."""
    out: List[str] = []
    for u in units:
        if u.get("family") != family:
            continue
        for rid in binding_state(ledger, u["logical_work_id"]).get("run_ids") or []:
            if str(rid) not in out:
                out.append(str(rid))
    return out


# ------------------------------------------------------------------- terminal registry records for superseded attempts (29)
#: A dead attempt's registered runs that NO later generation continues stay `running` in the registry forever (336 orphans of the u0
#: wave's 48 superseded attempts on 2026-10-01). They are terminal in the registry too — through `run_state` records into the producers'
#: spool: state `cancelled` when sacct says CANCELLED (a STOP sentinel / scancel), else
#: `failed` (FAILED, TIMEOUT, NODE_FAIL, OUT_OF_MEMORY, …, and a COMPLETED job whose run never filed); `event_utc` = sacct's End; evidence
#: kind `scheduler_state` with sacct's state string, the ledger line as the evidence uri and the ledger event as the note. The fold is
#: monotonic (a row already past `running` keeps its state), so the records are safe to re-emit; a local receipt beside the ledger keeps
#: them from being written twice. Never `superseded` — that lifecycle state is a `supersede` RULING's or the producer's own
#: `duplicate_registration` record for a twin registration of ONE store (the full22 closer's BoN climb-by-store512 and multi-turn
#: two-group rules; lib/admission.py `supersession` reads both) — a dead attempt's run is neither.
RECONCILED_FILE = "reconciled_registrations.jsonl"


def _utc_ms(stamp: Optional[str]) -> Optional[str]:
    """sacct's `2026-10-01T01:19:53` (or an ISO stamp with/without Z) -> the registry's `2026-10-01T01:19:53.000Z`; None when unknown."""
    if not stamp or str(stamp) in ("Unknown", "None", ""):
        return None
    t = str(stamp).strip().replace(" ", "T")
    if t.endswith("Z"):
        t = t[:-1]
    if "." in t:
        head, frac = t.split(".", 1); t = f"{head}.{(frac + '000')[:3]}"
    else:
        t = f"{t}.000"
    return t + "Z"


def reconciled_path(ledger: str) -> str:
    return os.path.join(os.path.dirname(os.path.abspath(ledger)), "gate", RECONCILED_FILE)


def registry_refused_receipts(registry_db: Optional[str], done: Dict[str, dict], spool: Optional[str]) -> Dict[str, dict]:
    """{run_id: {record_sha256, record_uri, record_type, field, reason, refused_utc}} for every local receipt whose record the registry REFUSED
    (`record_refusal`, matched by the receipt's record_sha256 or its spool-relative uri) — the one fact that voids a receipt. A registry
    without the table (a fixture) refuses nothing; a receipt that names no record stands."""
    if not registry_db or not done or not os.path.isfile(registry_db):
        return {}
    import sqlite3
    by_sha: Dict[str, str] = {}; by_uri: Dict[str, str] = {}
    for rid, r in done.items():
        if r.get("record_sha256"):
            by_sha[str(r["record_sha256"])] = rid
        uri = str(r.get("record_uri") or "")                       # 29-fix-2: the root-relative `records/...` uri the harvester's refusal carries
        if not uri:
            path = str(r.get("record_path") or "")
            if path and spool:
                try:
                    rel = os.path.relpath(path, spool); uri = "records/" + (rel[:-3] if rel.endswith(".gz") else rel)
                except ValueError:
                    uri = ""
        if uri:
            by_uri[uri] = rid
    if not by_sha and not by_uri:
        return {}
    conn = sqlite3.connect(f"file:{registry_db}?mode=ro", uri=True)
    try:
        if not conn.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'record_refusal'").fetchone():
            return {}
        q = (f"SELECT record_sha256, record_uri, record_type, field, reason, refused_utc FROM record_refusal "
             f"WHERE record_sha256 IN ({','.join('?' * len(by_sha)) or 'NULL'}) OR record_uri IN ({','.join('?' * len(by_uri)) or 'NULL'})"
             + "".join(" OR record_uri LIKE ? || '#%'" for _ in by_uri))                 # a multi-record file's refusal names its line: `<uri>#<line>`
        out: Dict[str, dict] = {}
        for sha, uri, rtype, field, reason, utc in conn.execute(q, list(by_sha) + list(by_uri) + list(by_uri)):
            rid = by_sha.get(str(sha)) or by_uri.get(str(uri).split("#", 1)[0])
            if rid:
                out[rid] = {"record_sha256": sha, "record_uri": uri, "record_type": rtype, "field": field, "reason": reason, "refused_utc": utc}
        return out
    finally:
        conn.close()


def read_reconciled(ledger: str) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    p = reconciled_path(ledger)
    if os.path.isfile(p):
        for ln in open(p, encoding="utf-8"):
            try:
                r = json.loads(ln)
            except json.JSONDecodeError:
                continue
            if r.get("run_id"):
                out[str(r["run_id"])] = r
    return out


def retired_run_tasks(ledger: str, lwid: str, attempt_id: Optional[str]) -> Optional[Dict[str, Optional[int]]]:
    """{run_id: array_task | None} from the RETIRED registered binding of `attempt_id` (`<stem>.registered.retired.<attempt>[.n].json`, the
    record 06 --gate wrote from the job's pending file: `runs: [{run_id, eval_id, object, array_task}]`) — the per-run task binding the
    reconciliation decides each run on (29-fix). None when no retired binding of that attempt carries `runs`."""
    d = bindings_dir(ledger); safe = safe_id(lwid); tag = safe_id(str(attempt_id or "unknown")).replace("@", "_at_")
    found: Dict[str, Optional[int]] = {}; seen = False
    for name in sorted(os.listdir(d)) if os.path.isdir(d) else []:
        if not (name.startswith(f"{safe}.registered.retired.{tag}") and name.endswith(".json")):
            continue
        try:
            rec = json.load(open(os.path.join(d, name), encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        runs = rec.get("runs")
        if not isinstance(runs, list):
            continue
        seen = True
        for r in runs:
            if isinstance(r, dict) and r.get("run_id"):
                found[str(r["run_id"])] = r.get("array_task") if isinstance(r.get("array_task"), int) else None
    return found if seen else None


def retired_binding_info(ledger: str, lwid: str, attempt_id: Optional[str]) -> Tuple[Optional[Dict[str, Optional[int]]], bool, Optional[str]]:
    """(run tasks, array known, array spec) from the RETIRED registered binding of `attempt_id` — the per-run `array_task` map (None when no
    retired binding carries `runs`), whether the binding records its sbatch `array` spec (the registered record copies it from the submitted one:
    None for a plain job), and that spec. 29-fix-2: array-ness comes from HERE, never from the rows sacct returns — when
    every child row is purged, sacct shows only the aggregate parent and an array looks plain."""
    d = bindings_dir(ledger); safe = safe_id(lwid); tag = safe_id(str(attempt_id or "unknown")).replace("@", "_at_")
    found: Dict[str, Optional[int]] = {}; seen_runs = False; arr_known = False; arr_spec: Optional[str] = None
    for name in sorted(os.listdir(d)) if os.path.isdir(d) else []:
        if not (name.startswith(f"{safe}.registered.retired.{tag}") and name.endswith(".json")):
            continue
        try:
            rec = json.load(open(os.path.join(d, name), encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if "array" in rec:
            arr_known = True; arr_spec = rec.get("array") if rec.get("array") not in ("", "-") else None
        runs = rec.get("runs")
        if isinstance(runs, list):
            seen_runs = True
            for r in runs:
                if isinstance(r, dict) and r.get("run_id"):
                    found[str(r["run_id"])] = r.get("array_task") if isinstance(r.get("array_task"), int) else None
    return (found if seen_runs else None), arr_known, arr_spec


def superseded_run_orphans(ledger: str, units: Optional[List[str]] = None) -> Tuple[List[dict], Dict[str, str]]:
    """(orphans, pending): per `superseded_attempt` event the dead attempt's run ids that NO later generation continues — not in a LATER
    `superseded_attempt`'s run ids (the next generation registered them too) and not in the unit's CURRENT registered binding — each as
    {logical_work_id, attempt_id, job_id, cluster, utc, run_id, state (sacct states the funnel saw), why}. A unit whose current binding is
    `submitted` (a live attempt that has not registered yet) is `pending`: its generation may still re-register the same ids."""
    events = [r for r in _ledger_events(ledger) if r.get("event") == "superseded_attempt" and r.get("logical_work_id") and (units is None or r["logical_work_id"] in units)]
    by_unit: Dict[str, List[dict]] = {}
    for r in events:
        by_unit.setdefault(r["logical_work_id"], []).append(r)
    orphans: List[dict] = []; pending: Dict[str, str] = {}
    for lw, evs in sorted(by_unit.items()):
        try:
            b = binding_state(ledger, lw)
        except Exception as exc:    # noqa: BLE001 — an unreadable binding: say so, decide nothing for the unit
            pending[lw] = f"binding unreadable ({exc})"; continue
        if b.get("state") in ("submitted", "transition"):
            pending[lw] = f"its current attempt (job {b.get('job_id')} on {b.get('cluster')}) is {b.get('state')} — not registered yet; its generation may re-register the same run ids"
            continue
        current = {str(x) for x in (b.get("run_ids") or [])} if b.get("state") == "registered" else set()
        for i, e in enumerate(evs):
            later = {str(x) for later_e in evs[i + 1:] for x in (later_e.get("run_ids") or [])}
            for rid in [str(x) for x in (e.get("run_ids") or [])]:
                if rid in later or rid in current:
                    continue
                orphans.append({"logical_work_id": lw, "attempt_id": e.get("attempt_id"), "job_id": str(e.get("job_id")), "cluster": e.get("cluster"),
                                "utc": e.get("utc"), "run_id": rid, "state": e.get("state"), "why": (e.get("why") or "")[:200]})
    return orphans, pending


def registry_run_campaigns(registry_db: Optional[str], run_ids: List[str]) -> Dict[str, Optional[dict]]:
    """{run_id: {lifecycle_state, campaign} | None} from a fresh read-only connection; a registry without a `campaign` column (a synthetic
    one) derives it from the run id's first segment (identity.run_id spells `<campaign>:<eval>:…`)."""
    if not registry_db or registry_db == "none" or not os.path.isfile(registry_db):
        raise ValueError(f"registry database {registry_db!r} does not exist — the orphan rows cannot be read (fail closed)")
    import sqlite3
    conn = sqlite3.connect(f"file:{registry_db}?mode=ro", uri=True)
    try:
        out: Dict[str, Optional[dict]] = {r: None for r in run_ids}
        for chunk in (run_ids[i:i + 400] for i in range(0, len(run_ids), 400)):
            if not chunk:
                continue
            marks = ",".join("?" for _ in chunk)
            try:
                rows = [(rid, st, camp) for rid, st, camp in conn.execute(f"select run_id, lifecycle_state, campaign from run where run_id in ({marks})", chunk)]
            except sqlite3.OperationalError:
                rows = [(rid, st, None) for rid, st in conn.execute(f"select run_id, lifecycle_state from run where run_id in ({marks})", chunk)]
            for rid, st, camp in rows:
                out[str(rid)] = {"lifecycle_state": st, "campaign": str(camp) if camp else str(rid).split(":", 1)[0]}
        return out
    finally:
        conn.close()


def reconcile_registrations(ledger: str, registry_db: Optional[str], sacct_tasks, *, spool: Optional[str], units: Optional[List[str]] = None,
                            write: bool = False, producer: Optional[dict] = None, code_sha: Optional[str] = None) -> dict:
    """The orphaned runs of superseded attempts -> terminal `run_state` records in the producers' spool (29). Per orphan: the registry row
    must exist and still be `queued`/`running` (anything else keeps its state: the fold is monotonic); the dead job's sacct tasks must all
    be terminal (a live job decides nothing); state = `cancelled` when any task CANCELLED else `failed`; `event_utc` = the latest task End
    (else the ledger event's utc). With `write`, one record per orphan through `sisteer.registry.emit.RecordWriter.emit_state` and a line
    in gate/reconciled_registrations.jsonl so a second pass writes nothing; without, the plan. Returns the report."""
    orphans, pending = superseded_run_orphans(ledger, units)
    done = read_reconciled(ledger)
    rows = registry_run_campaigns(registry_db, sorted({o["run_id"] for o in orphans})) if orphans else {}
    rep: Dict[str, object] = {"ledger": os.path.abspath(ledger), "registry": registry_db, "spool": spool, "write": write, "n_superseded_runs": len(orphans),
                              "pending_units": pending, "skipped": {"already_reconciled": [], "not_in_registry": [], "past_running": [], "job_alive": [], "job_unproven": [],
                                                                    "task_completed": [], "task_unbound": []},
                              "retried_receipts": [], "plan": [], "written": []}
    refused = registry_refused_receipts(registry_db, done, spool)       # 29-fix: the receipts the registry REFUSED — the one thing that voids one
    proofs: Dict[Tuple[str, str], Tuple[List[dict], Optional[str]]] = {}
    task_maps: Dict[Tuple[str, Optional[str]], Tuple[Optional[Dict[str, Optional[int]]], bool, Optional[str]]] = {}
    for o in orphans:
        rid = o["run_id"]
        row = rows.get(rid)
        if rid in done:
            # 29-fix: the local receipt stands unless the registry REFUSED the record it names — a `record_refusal`
            # row for the receipt's record (by its sha256, or its spool-relative uri) is the one thing that voids a receipt, and the retry is
            # named with the reason. An emission the harvester has not folded yet (the row still running; a `blocked_records` row awaiting its
            # parent) is ordinary lag: re-emitting on lag would duplicate run_state records every walk — the second `--write` is a no-op by contract.
            ref = refused.get(rid)
            if ref is None or (row is not None and row["lifecycle_state"] not in ("queued", "running")):
                rep["skipped"]["already_reconciled"].append(rid); continue           # no refusal, or the effect is already there: the receipt stands
            rep["retried_receipts"].append({"run_id": rid, "receipt": done[rid].get("written_utc"), "record": done[rid].get("record_path"), "refusal": ref})
        if row is None:
            rep["skipped"]["not_in_registry"].append(rid); continue
        if row["lifecycle_state"] not in ("queued", "running"):
            rep["skipped"]["past_running"].append({"run_id": rid, "lifecycle_state": row["lifecycle_state"]}); continue
        key = (str(o["cluster"]), str(o["job_id"]))
        if key not in proofs:
            try:
                tasks = [{"raw": str(t.get("raw")), "task": t.get("task"), "state": str(t.get("state") or ""), "end": t.get("end")} for t in (sacct_tasks(key[0], key[1]) or [])]
                proofs[key] = (tasks, None)
            except Exception as exc:    # noqa: BLE001 — the funnel's reader could not answer: unproven, decide nothing
                proofs[key] = ([], f"sacct could not be read for job {key[1]} on {key[0]}: {exc}")
        tasks, err = proofs[key]
        if err or not tasks:
            rep["skipped"]["job_unproven"].append({"run_id": rid, "job_id": key[1], "cluster": key[0], "why": err or "job unknown to sacct"}); continue
        # 29-fix: a run is decided on ITS OWN task, never on its siblings — on a mixed array (one task FAILED, another
        # COMPLETED with results awaiting harvest) the aggregate wrote `failed` over valid work. The run's task is the retired registered
        # binding's `array_task`; an array run without one is not inferred ("task_unbound"); a task that COMPLETED is never `failed` from here
        # ("task_completed": the store is the closer's / the harvester's to read).
        # 29-fix-2: array-ness is READ from the attempt's retired registered binding — the run's own `array_task`, or the
        # binding's sbatch `array` spec — never inferred from the rows sacct returns: when every child row is purged, sacct shows only the aggregate
        # parent (task None) and an array looks plain, and the parent's state would mark EVERY orphan run — a completed child's included.
        mk = (o["logical_work_id"], o.get("attempt_id"))
        if mk not in task_maps:
            task_maps[mk] = retired_binding_info(ledger, o["logical_work_id"], o.get("attempt_id"))
        tmap, arr_known, arr_spec = task_maps[mk]
        run_task = tmap.get(rid) if (tmap is not None and rid in tmap) else None
        if run_task is not None or (arr_known and arr_spec):
            is_array = True
        elif (arr_known and not arr_spec) or (tmap is not None and rid in tmap):
            is_array = False                                                                 # the binding declares no array, or names the run with no task
        else:
            rep["skipped"]["job_unproven"].append({"run_id": rid, "job_id": key[1], "cluster": key[0],
                                                   "why": "array-ness unknowable: no retired registered binding of this attempt names the run or records its array spec — "
                                                          "never inferred from sacct's rows; record the binding by hand"}); continue
        if is_array:
            if run_task is None:
                rep["skipped"]["task_unbound"].append({"run_id": rid, "job_id": key[1], "cluster": key[0],
                                                       "why": "array job: no per-run array_task in the attempt's retired registered binding — not inferred from sibling tasks"}); continue
            mine = [t for t in tasks if t.get("task") == run_task]
            if not mine:
                rep["skipped"]["job_unproven"].append({"run_id": rid, "job_id": key[1], "cluster": key[0],
                                                       "why": f"array task {run_task} of job {key[1]} is not in sacct's answer (purged, or not there yet) — never decided from the aggregate parent row"}); continue
        else:
            if any(t.get("task") is not None for t in tasks):
                rep["skipped"]["job_unproven"].append({"run_id": rid, "job_id": key[1], "cluster": key[0],
                                                       "why": f"the binding declares job {key[1]} a plain job but sacct lists array task rows — a contradiction; not decided"}); continue
            mine = [t for t in tasks if t.get("task") is None]
            if not mine:
                rep["skipped"]["job_unproven"].append({"run_id": rid, "job_id": key[1], "cluster": key[0], "why": f"job {key[1]} has no plain row in sacct's answer"}); continue
        if any(not t["state"] or t["state"] in SACCT_ACTIVE for t in mine):
            rep["skipped"]["job_alive"].append({"run_id": rid, "job_id": key[1], "cluster": key[0], "states": sorted({t["state"] for t in mine})}); continue
        if any(t["state"].startswith("COMPLETED") for t in mine):
            rep["skipped"]["task_completed"].append({"run_id": rid, "job_id": key[1], "cluster": key[0], "task": mine[0].get("task"),
                                                     "why": "the run's own task COMPLETED: its store may hold generated work awaiting the closer / harvester — never `failed` from here"}); continue
        states = sorted({t["state"] for t in mine})
        state = "cancelled" if any(t["state"].startswith("CANCELLED") for t in mine) else "failed"
        ends = [_utc_ms(t.get("end")) for t in mine if _utc_ms(t.get("end"))]
        event_utc = max(ends) if ends else (_utc_ms(o.get("utc")) or utc_now())
        line = {"run_id": rid, "logical_work_id": o["logical_work_id"], "attempt_id": o["attempt_id"], "job_id": key[1], "cluster": key[0], "campaign": row["campaign"],
                "registry_state": row["lifecycle_state"], "sacct_states": states, "state": state, "event_utc": event_utc, "array_task": mine[0].get("task") if is_array else None,
                "retry_of_receipt": bool(rid in done),
                "evidence_uri": f"{os.path.join(os.path.dirname(os.path.abspath(ledger)), 'attempts.jsonl')}#{o.get('utc')}",
                "note": json.dumps({"event": "superseded_attempt", "logical_work_id": o["logical_work_id"], "attempt_id": o["attempt_id"], "job_id": key[1], "cluster": key[0],
                                    "utc": o.get("utc"), "why": o.get("why"), "array_task": mine[0].get("task") if is_array else None,
                                    **({"retry_of_receipt": done[rid].get("written_utc")} if rid in done else {})}, sort_keys=True)}
        rep["plan"].append(line)
    if write and rep["plan"]:
        from sisteer.registry import emit as E     # the producers' own writer: one record file per run under <spool>/<campaign>/<run dir>/
        if not spool:
            raise ValueError("no spool: pass --spool or set SISTEER_REGISTRY_SPOOL (the registry owner names the producers' spool)")
        prod = dict(producer or E.producer_block("reproduce/lib/stage.py reconcile-registrations", code_sha=code_sha))
        os.makedirs(os.path.dirname(reconciled_path(ledger)), exist_ok=True)
        with open(reconciled_path(ledger), "a", encoding="utf-8") as fh:
            for line in rep["plan"]:
                w = E.RecordWriter(spool, line["campaign"], line["run_id"], prod, disabled=False)
                path = w.emit_state(line["state"], "scheduler_state", evidence_uri=line["evidence_uri"], event_utc=line["event_utc"],
                                    scheduler_state=",".join(line["sacct_states"]), note=line["note"])
                # 29-fix-2: the receipt names the record by the REGISTRY'S identity — records.record_sha of the canonical JSON (not
                # the file's bytes) and the root-relative `records/<campaign>/<run>/<file>` uri the harvester's record_refusal carries — so a real
                # refusal of this record is what voids the receipt
                from sisteer.registry import records as _R
                with open(path, encoding="utf-8") as _fh:
                    _obj = json.loads(_fh.readline())
                _rel = os.path.relpath(path, spool)
                rec = {**{k: line[k] for k in ("run_id", "logical_work_id", "attempt_id", "job_id", "cluster", "campaign", "state", "event_utc", "sacct_states")},
                       "record_path": path, "record_sha256": _R.record_sha(_obj), "record_uri": "records/" + (_rel[:-3] if _rel.endswith(".gz") else _rel), "written_utc": utc_now()}
                fh.write(json.dumps(rec, sort_keys=True) + "\n")
                rep["written"].append(rec)
    return rep


def cmd_reconcile_registrations(a) -> int:
    units = [u for u in (a.units or "").split(",") if u] or None
    spool = a.spool or os.environ.get("SISTEER_REGISTRY_SPOOL") or None
    if spool is None:
        try:
            from sisteer.registry import emit as E
            spool = E.spool_root()
        except Exception:     # noqa: BLE001 — no registry package on the path: the dry run still plans; --write refuses below
            spool = None
    try:
        reader = sacct_reader(a.config)
        rep = reconcile_registrations(a.ledger, a.registry, reader, spool=spool, units=units, write=a.write, code_sha=git_rev(a.repo, "HEAD") if a.repo else None)
    except (ValueError, OSError) as exc:
        print(f"REFUSED: reconcile-registrations: {exc}", file=sys.stderr); return 2
    if a.out:
        write_atomic(a.out, rep)
    sk = rep["skipped"]
    for line in rep["plan"]:
        print(f"{'wrote' if a.write else 'would write'} run_state {line['state']} for {line['run_id']} ({line['logical_work_id']}: job {line['job_id']} on {line['cluster']} "
              f"{'/'.join(line['sacct_states'])}, registry {line['registry_state']}; event_utc {line['event_utc']})")
    for lw, why in sorted(rep["pending_units"].items()):
        print(f"pending {lw}: {why}")
    print(f"reconcile-registrations: {rep['n_superseded_runs']} superseded run(s); {len(rep['plan'])} {'written' if a.write else 'to write'}"
          f"{' (' + str(len(rep['written'])) + ' record file(s) under ' + str(spool) + ')' if a.write else ''}; skipped: "
          + ", ".join(f"{k} {len(v)}" for k, v in sk.items()) + f"; pending units {len(rep['pending_units'])}"
          + (f"; receipts retried (record REFUSED by the registry) {len(rep['retried_receipts'])}" if rep.get("retried_receipts") else "")
          + (f"; report -> {a.out}" if a.out else ""))
    return 0


def cmd_sha_table(a) -> int:
    print(json.dumps(sha_table(a.ledger), indent=2, sort_keys=True)); return 0


def cmd_amend_contract(a) -> int:
    try:
        tip = a.tip or git_rev(a.repo, "HEAD")
        reader = sacct_reader(a.config)
        if a.recheck:
            results = recheck_amendments(a.ledger, a.logical_work_id, repo=a.repo, tip=tip, registry_db=a.registry, sacct_tasks=reader)
            if not results:
                print(f"REFUSED: amend-contract --recheck: {a.logical_work_id} has no receipted amendment in {provenance_path(a.ledger)}", file=sys.stderr); return 2
            for r in results:
                head = f"RECHECK {r['logical_work_id']} #{r['index']} commit {r.get('commit')} ({r.get('utc')}) against tip {r['tip'][:12]}: {r['verdict']}"
                if r["verdict"] == "PASS":
                    print(f"{head} — old export derived from {r['old_sha'][:12]} (sha256 {r['old_export_sha256'][:12]}), row {r.get('decode_policy_ref')} diff at "
                          f"{r.get('row_diff')}; {r['n_old_jobs']} old job(s) terminal on sacct; {r['n_old_runs']} old run(s) none completed (registry read {r['registry_read_utc']})")
                else:
                    print(f"{head} — " + "; ".join(r["problems"]))
            return 0 if all(r["verdict"] == "PASS" for r in results) else 3
        if not a.reason or not a.commit:
            print("REFUSED: amend-contract needs --reason and --commit (or --recheck)", file=sys.stderr); return 2
        entry = amend_contract(a.ledger, a.logical_work_id, reason=a.reason, commit=a.commit, tip=tip, repo=a.repo, registry_db=a.registry,
                               sacct_tasks=reader, old_export_path=a.old_export, dry_run=a.dry_run, kind=a.kind, manifest_rel=a.manifest_rel, note=a.note)
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        print(f"REFUSED: amend-contract: {exc}", file=sys.stderr); return 3
    if entry.get("kind") == "code-follow-up":
        fu = entry.get("follow_up_of") or {}
        print(f"{'DRY-RUN: would amend' if a.dry_run else 'amended'} {a.logical_work_id} (code-follow-up of the {fu.get('kind')} receipt at {str(fu.get('commit'))[:12]}, "
              f"{fu.get('utc')}): unit sha {entry['old_sha'][:12]} -> {entry['commit_resolved'][:12]} (sha table `amended`"
              f"{'; WAVE unit_shas refreshed: ' + str(len(entry.get('wave_files_refreshed') or [])) if not a.dry_run else ''}); manifest {entry.get('manifest_rel')} unchanged "
              f"({str(entry.get('new_manifest_sha256'))[:12]}), roster {entry.get('roster_rel')} unchanged ({str(entry.get('new_roster_sha256'))[:12]}), contract "
              f"{entry['new_contract_sha256'][:12]} unchanged, export unchanged; {len(entry['old_jobs'])} old job(s); {len(entry['old_run_ids'])} old run(s) the receipt's, "
              f"{len(entry.get('carried_run_ids') or [])} carried unchanged (registry re-read {entry['completed_run_check']['post_fence_utc']}); note: {entry.get('note')} "
              f"({entry['reason'][:80]}…)")
        return 0
    if entry.get("kind") == "petri-role-fallback":
        same = entry["old_contract_sha256"] == entry["new_contract_sha256"]
        roles = ", ".join(f"{d['field']} {d['before']!r} -> {d['after']!r}" for d in (entry.get("role_diff") or []))
        live = ", ".join(f"{j['job_id']} on {j['cluster']} {'/'.join(j['sacct_states'])}" for j in entry.get("live_jobs") or [])
        print(f"{'DRY-RUN: would amend' if a.dry_run else 'amended'} {a.logical_work_id} (petri-role-fallback): manifest {entry['manifest_rel']} {entry['old_sha'][:12]} -> "
              f"{entry['commit_resolved'][:12]}: {roles}; contract {entry['old_contract_sha256'][:12]} -> {entry['new_contract_sha256'][:12]} "
              f"({'unchanged' if same else 'instruments changed for ' + str(sorted(entry['changed_instruments']))}); unit sha {entry['old_sha'][:12]} -> {entry['commit_resolved'][:12]} "
              f"(sha table `amended`{'; WAVE unit_shas refreshed: ' + str(len(entry.get('wave_files_refreshed') or [])) if not a.dry_run else ''}); "
              f"{len(entry['old_jobs'])} old job(s){' (' + live + ' ALIVE: the drive restarts against it)' if live else ' terminal'}; "
              f"{len(entry['old_run_ids'])} old run(s), none completed, {len(entry.get('carried_run_ids') or [])} carried under the amended contract (registry re-read {entry['completed_run_check']['post_fence_utc']})"
              f"{'; note: ' + entry['note'] if entry.get('note') else ''} ({entry['reason'][:80]}…)")
        return 0
    print(f"{'DRY-RUN: would amend' if a.dry_run else 'amended'} {a.logical_work_id}: contract {entry['old_contract_sha256'][:12]} -> {entry['new_contract_sha256'][:12]}; "
          f"instruments changed for {sorted(entry['changed_instruments'])}; row {entry['policy_id']} diff at {[c['path'] for c in entry['row_diff']]} "
          f"(old export derived from {entry['old_sha'][:12]}, sha256 {entry['old_export_sha256'][:12]}; commit {entry['commit_resolved'][:12]} in tip {entry['tip'][:12]}); "
          f"{len(entry['old_jobs'])} old job(s) terminal on sacct; {len(entry['old_run_ids'])} old run(s) listed, none completed (registry re-read "
          f"{entry['completed_run_check']['post_fence_utc']}) ({entry['reason'][:80]}…)")
    return 0


def cmd_bind(a) -> int:
    if a.state == "submitted":
        if not (a.job_id and a.cluster and a.attempt_id):
            print("REFUSED: --state submitted needs --job-id --cluster --attempt-id", file=sys.stderr); return 2
        rec = submitted_record(a.logical_work_id, a.job_id, a.cluster, a.attempt_id, a.sbatch_name, a.array)
    elif a.state == "registered":
        rec = {"logical_work_id": a.logical_work_id, "state": a.state, "utc": utc_now()}
        if not a.run_ids:
            print("REFUSED: --state registered needs --run-ids", file=sys.stderr); return 2
        prev = binding_state(a.ledger, a.logical_work_id)
        if prev["state"] == "unbound":
            print(f"REFUSED: {a.logical_work_id} was never submitted (no submitted binding) — cannot register", file=sys.stderr); return 2
        if prev["state"] == "transition":
            tok = prev.get("transition") or {}
            print(f"REFUSED: {a.logical_work_id} is mid re-run (transition token since {tok.get('utc')}: attempt {tok.get('intended_attempt')} replacing "
                  f"{(tok.get('old_binding') or {}).get('attempt_id')}) — the gate must not bind the superseded generation; `submit --resubmit` reconciles it first", file=sys.stderr); return 2
        rec.update({k: prev.get(k) for k in ("job_id", "cluster", "attempt_id", "array")})
        # The climb array's four tasks register ONE run per cell, so 06 --gate's per-task
        # enumeration hands this `--run-ids` the same id once per task — `run_ids` is the SET of registry runs the unit binds (first
        # occurrence's order), while `runs` keeps every per-task row (run_id, array_task …): the task identities are preserved there
        rec.update({"run_ids": list(dict.fromkeys(r for r in a.run_ids.split(",") if r)), "pending_file": a.pending_file,
                    "runs": json.loads(arg_doc(a.runs_json) or "null") if a.runs_json else None})
    else:
        print(f"REFUSED: unknown state {a.state}", file=sys.stderr); return 2
    if a.dry_run:
        print(f"DRY-RUN: bind {a.logical_work_id} -> {a.state}")
        return 0
    write_binding(a.ledger, rec)
    print(f"bound {a.logical_work_id}: {a.state}")
    return 0


def coverage_path(state_root: str, lwid: str) -> str:
    return os.path.join(state_root, "gate", "coverage", safe_id(lwid) + ".json")


def frozen_unit(state_root: str, lwid: str) -> dict:
    inv = json.load(open(os.path.join(state_root, "gate", "expected_units.json"), encoding="utf-8"))
    for u in inv["units"]:
        if u["logical_work_id"] == lwid:
            return u
    raise KeyError(f"{lwid} is not a unit of the frozen inventory")


# ------------------------------------------------------------------ the authoritative store, the coverage contract, and both shas
STORE_SOURCE_REGISTRY = "registry.run.store_uri"
#: run.extra_json.climb_root — the BoN climb's FLAT root (`$WAVE_ROOT/bon_climb/<TAG>`: plan/climb_manifest.jsonl + out/<cell>/shard<k>.jsonl),
#: filed by the producer's own closer on every climb run it closes (scripts/sdfbase/full22/registry_close_stores.py, beside the store facts).
#: The registry's `store_uri` of a climb run is the 512 STORE it climbed (bon_served_manifest.json certifies the 512 draws, not the escalations),
#: so the climb's artifacts have no record but this key; a climb run without it reads "not covered, no store on record" by name, never the
#: producer convention recomputed.
CLIMB_ROOT_KEY = "climb_root"


def probe_store(unit: dict, run: dict) -> Tuple[Optional[str], str, Optional[str]]:
    """(store path, store_source, reason-when-none) for one BOUND run: the location the AUTHORITATIVE record names — the registry row's
    `store_uri` (the job-side closer's filing, or the tinycap recorder's / the Petri hand-off's), a ruled successor's `store_uri` (06 --gate
    writes it onto the run), the climb's `extra_json.climb_root` — NEVER the frozen `coverage.path` (`root_rel` + the run's object, a
    recomputation the ruling forbids: it read the paper's ungrouped root for the grouped multi-turn targets and 0/30 present). No record ->
    `not covered, no store on record`, by name. A run the PRODUCER retired as a duplicate registration of one store (admission's
    producer_supersession, `ruling_id` None) reads its OWN record, as a live run does (the work never moved)."""
    ev = unit["evals"].get(run.get("eval_id"))
    rid, e = run.get("run_id"), run.get("eval_id")
    if ev is None:
        return None, "none", f"run {rid} is eval {e!r}, not one of the unit's"
    lay = ev["coverage"]["layout"]
    succ = run.get("successor") or {}
    if succ and succ.get("ruling_id"):
        # a superseded run is probed at its ruled SUCCESSOR's recorded store (the consolidated store the ruling names)
        path = run.get("store_uri")
        if not path:
            return None, "none", (f"run {rid} ({e}) is superseded by {succ.get('run_id')}, which records no store (store_uri null) — nothing to read coverage from")
        return str(path), f"{STORE_SOURCE_REGISTRY} of the ruled successor {succ.get('run_id')} ({succ.get('ruling_id')})", None
    if lay == "bon_climb_flat":
        extra = run.get("extra") if isinstance(run.get("extra"), dict) else {}
        root = extra.get(CLIMB_ROOT_KEY)
        if isinstance(root, str) and root:
            return root, f"registry.run.extra_json.{CLIMB_ROOT_KEY}", None
        s512 = f"the 512 store {run.get('store_uri')}" if run.get("store_uri") else "no store (store_uri null)"
        return None, "none", (f"run {rid} ({e}, climb task {run.get('array_task')}): not covered, no store on record for the climb's flat root — the registry records "
                              f"{s512} (bon_served_manifest.json certifies the 512 draws, not the climb's escalations) and no extra_json.{CLIMB_ROOT_KEY}; the funnel "
                              f"binding records no wave root; the climb's plan/ and out/ live under a producer convention (bon_climb.sbatch ROOT=$WAVE_ROOT/bon_climb/<TAG>) "
                              f"that no record names — D3003 (1): never guessed from a recomputed root. The producer's closer files extra_json.{CLIMB_ROOT_KEY} on every "
                              f"climb run it closes (registry_close_stores.py); a climb closed before that is patched by the registry owner, never by this gate")
    path = run.get("store_uri")
    if not path:
        return None, "none", (f"run {rid} ({e}): not covered, no store on record — the registry row carries store_uri null (the producer's closer has not filed "
                              f"the store; D3003 (1): never guessed from a recomputed root)")
    return str(path), STORE_SOURCE_REGISTRY, None


COVERAGE_CONTRACT_VERSION = "cc1"
COVERAGE_CONTRACT_SIDECAR = "coverage_contract.json"       # <state>/gate/: the admit-once record of an inventory frozen before the header field existed
COVERAGE_CONTRACT_RULING = "D3003 (2)"
#: the per-eval coverage keys that are IDENTITY (WHAT is measured) beside unit_contract's — never `path` / `host` / `expected_source` (WHERE it is read)
COVERAGE_IDENTITY_KEYS = ("n_max", "nshards", "n_samples", "behavior", "task")


def _ids_sha256(ids) -> str:
    return hashlib.sha256("\n".join(str(x) for x in ids).encode("utf-8")).hexdigest()


def coverage_identity(u: dict) -> dict:
    """One unit's COVERAGE IDENTITY: unit_contract (every eval's instrument_id, decode_policy_ref, thinking_mode, expected_shards, layout,
    item_set_id, items_sha256, n_items — what admission compares) plus the per-eval coverage identity keys (n_max,
    nshards, n_samples, behavior, task), the frozen expected-id sets as per-shard sha256s with their counts, and the unit's object pins.
    Never the reader's locations (`coverage.path`, `host`, `expected_source`), the tree, the sbatch name, the freeze time, a hold-out /
    rerun stamp: a reader or layout fix changes nothing here; a changed identity is a new wave or a ruling."""
    c = unit_contract(u)
    for e, ev in sorted((u.get("evals") or {}).items()):
        cov = ev.get("coverage") or {}
        exp = cov.get("expected") if isinstance(cov.get("expected"), dict) else None
        c["evals"][e].update({k: cov.get(k) for k in COVERAGE_IDENTITY_KEYS})
        c["evals"][e]["expected_ids_sha256"] = {s: _ids_sha256(ids) for s, ids in sorted(exp.items())} if exp is not None else None
        c["evals"][e]["n_expected"] = {s: len(ids) for s, ids in sorted(exp.items())} if exp is not None else None
    objs = u.get("objects")
    c["objects"] = dict(sorted(objs.items())) if isinstance(objs, dict) and objs else None
    return c


COVERAGE_CONTRACT_BODY_KEYS = ("unit_set", "identities", "done_predicates")


def coverage_contract_sha256(version: str, body: Mapping[str, object]) -> str:
    """The contract's digest: over the canonical JSON of {version, unit_set, identities, done_predicates} — the VERSION is inside the digest,
    so one sha names one (rule, content) pair: a record of another version never shares a sha with
    this tree's reading of the same body, and admission's sha tie (inventory_contract_sha) refuses it without reading the record."""
    return contract_dict_sha256({"version": version, **{k: body[k] for k in COVERAGE_CONTRACT_BODY_KEYS}})


def coverage_contract(units: Sequence[dict], done_predicates: Mapping[str, str]) -> dict:
    """The COVERAGE CONTRACT over a unit list: {version, sha256, unit_set, identities, done_predicates} — the sha over the canonical
    JSON of the version and the three parts (coverage_contract_sha256); the done predicates are coverage_probe.DONE_PREDICATES of the probe in
    question (this tree's at freeze / stamp time, the current probe's at gate time)."""
    ids = {str(u["logical_work_id"]): coverage_identity(u) for u in units}
    body = {"unit_set": sorted(ids), "identities": ids, "done_predicates": dict(sorted((done_predicates or {}).items()))}
    return {"version": COVERAGE_CONTRACT_VERSION, "sha256": coverage_contract_sha256(COVERAGE_CONTRACT_VERSION, body), **body}


def coverage_contract_record_check(rec: Mapping[str, object], src: Optional[str], where: Optional[str]) -> Tuple[Optional[dict], Optional[str]]:
    """(the record's contract body, None) when the record is one THIS tree can compare against, else (None, the refusal by name) — read before
    any structural comparison: comparing bodies alone passes a record of version cc2 over the same unit set, identities and done
    predicates, and a cc1 verdict is then written over it. Three checks, in order: (1) `version` equals
    COVERAGE_CONTRACT_VERSION — a differing or absent version is a contract whose RULE (what its unit set, identities and done predicates
    comprise, how they are read) this tree does not know, so its body cannot be compared structurally: a coverage-contract version change is a
    ruling; (2) `contract` carries the three parts; (3) `sha256` equals the digest of the record's own (version,
    body) — else the body or the digest was changed by hand (the tools write a record whole, once; a stale digest over an edited body is a
    tamper signal, refused as such rather than compared)."""
    rv = rec.get("version")
    name = f"the {src or 'recorded'} coverage contract" + (f" at {where}" if where else "")
    if rv != COVERAGE_CONTRACT_VERSION:
        return None, (f"{name} is version {rv!r}, this tree's coverage contract is version {COVERAGE_CONTRACT_VERSION!r} — the contract's rule (what its unit set, "
                      f"identities and done predicates comprise) differs, so the record and the current contract cannot be compared structurally; a "
                      f"coverage-contract version change is a ruling: the operator must rule ({COVERAGE_CONTRACT_RULING}) before this tree reads the started wave")
    body = rec.get("contract") if isinstance(rec.get("contract"), dict) else None
    if not body or not all(k in body for k in COVERAGE_CONTRACT_BODY_KEYS):
        return None, f"{name} is malformed (no contract.unit_set / identities / done_predicates) — a hand edit"
    want = coverage_contract_sha256(str(rv), body)
    if rec.get("sha256") != want:
        return None, (f"{name} records sha256 {str(rec.get('sha256'))[:12]}… but its body digests to {want[:12]}… under version {rv} — the record was edited by hand "
                      f"(its body or its digest); the tools write it whole, once ({COVERAGE_CONTRACT_RULING}); nothing of it is compared")
    return dict(body), None


def coverage_contract_record(units: Sequence[dict], done_predicates: Mapping[str, str], *, source: str, **more) -> dict:
    """The IMMUTABLE record of a contract: the inventory header's `coverage_contract` (source `frozen`, written once at freeze) or the gate's
    admit-once sidecar (source `admitted_once`)."""
    c = coverage_contract(units, done_predicates)
    return {"version": c["version"], "sha256": c["sha256"], "source": source, "ruling": COVERAGE_CONTRACT_RULING,
            "contract": {k: c[k] for k in ("unit_set", "identities", "done_predicates")}, **more}


def coverage_contract_diff(rec: Mapping[str, object], cur: Mapping[str, object], *, units: bool = True) -> List[str]:
    """Named lines between a record's contract body and a current one: `done predicate <layout>: a -> b`; with `units`, `identity <lwid>/<field>:
    a -> b` for the units both carry and `unit set: +[…] -[…]`."""
    out: List[str] = []
    rp, cp = dict(rec.get("done_predicates") or {}), dict(cur.get("done_predicates") or {})    # type: ignore[arg-type]
    for lay in sorted(set(rp) | set(cp)):
        if rp.get(lay) != cp.get(lay):
            out.append(f"done predicate {lay}: {rp.get(lay)!r} -> {cp.get(lay)!r}")
    if units:
        ri, ci = dict(rec.get("identities") or {}), dict(cur.get("identities") or {})         # type: ignore[arg-type]
        for lw in sorted(set(ri) & set(ci)):
            out.extend(f"identity {lw}/{ln}" for ln in contract_diff(ri[lw], ci[lw]))
        rs, cs = set(rec.get("unit_set") or []), set(cur.get("unit_set") or [])              # type: ignore[arg-type]
        if rs != cs:
            out.append(f"unit set: +{sorted(cs - rs)} -{sorted(rs - cs)}")
    return out


def coverage_contract_sidecar(state_root: str) -> str:
    return os.path.join(state_root, "gate", COVERAGE_CONTRACT_SIDECAR)


def frozen_coverage_contract(state_root: str) -> Tuple[Optional[dict], Optional[str], Optional[str]]:
    """(record, source, where): the inventory header's `coverage_contract` (`frozen`), else the admit-once sidecar (`admitted_once`), else
    (None, None, None). Both present -> ValueError by name: the tools never create both (a freeze writes the header once; the gate writes the
    sidecar only when the header has none) — it is a hand edit."""
    inv_p = os.path.join(state_root, "gate", "expected_units.json")
    header = None
    if os.path.isfile(inv_p):
        try:
            header = (json.load(open(inv_p, encoding="utf-8")) or {}).get("coverage_contract")
        except json.JSONDecodeError:
            header = None
    side_p = coverage_contract_sidecar(state_root)
    side = None
    if os.path.isfile(side_p):
        try:
            side = json.load(open(side_p, encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{side_p} does not parse ({exc}) — the admitted-once coverage contract is unreadable (a hand edit; the gate wrote it whole)")
    if header and side:
        raise ValueError(f"the inventory {inv_p} carries a frozen coverage_contract AND the admit-once sidecar {side_p} exists — two authorities; the tools never "
                         f"create both (a hand edit): remove the sidecar when the header is the freeze's record")
    if header:
        return header, "frozen", inv_p
    if side:
        return side, "admitted_once", side_p
    return None, None, None


def this_tree_done_predicates() -> Dict[str, str]:
    """This tree's lib/coverage_probe.py DONE_PREDICATES, read from the file's bytes (probe_done_predicates): the freeze records them, rerun-units
    and `06 --gate` compare against them -- one reader, no bytecode cache between an edit of the probe and the next read."""
    return probe_done_predicates(os.path.join(HERE, "coverage_probe.py"))


def probe_done_predicates(probe_path: str) -> Dict[str, str]:
    """The DONE_PREDICATES a coverage_probe.py file declares (this tree's by default; a committed alternative in the tests) -- read from the
    file's BYTES AS THEY ARE, never through Python's bytecode cache: `importlib`'s source loader trusts a `__pycache__` entry whose recorded
    source mtime (whole seconds) and size match the file's, so a same-length edit made within the same second as the compile it followed --
    `capability_cellstore:v1` -> `:v2` -> `:v1` inside one second, the gate's own end-to-end test on a fast node (integration round 3a,
    2026-10-14) -- read the PREVIOUS predicate set and refused a gate whose probe had been reverted. The source is compiled and executed
    here directly (the module's `__file__` / `__name__` set as a loader would); nothing is written to the cache either."""
    import types
    try:
        with open(probe_path, "rb") as fh:
            source = fh.read()
    except OSError as exc:
        raise ValueError(f"{probe_path} cannot be read as a module: {exc}") from exc
    mod = types.ModuleType("coverage_probe_under_gate"); mod.__file__ = probe_path
    exec(compile(source, probe_path, "exec"), mod.__dict__)
    preds = getattr(mod, "DONE_PREDICATES", None)
    if not isinstance(preds, dict) or not preds:
        raise ValueError(f"{probe_path} declares no DONE_PREDICATES")
    return dict(preds)


def cmd_gate_contract(a) -> int:
    """`gate-contract --state-root S --probe <coverage_probe.py> [--admit-once] [--write <block>]`: run by `06 --gate` once per
    invocation BEFORE anything is bound. The CURRENT probe's contract (coverage_contract over the inventory's units with the probe's
    DONE_PREDICATES) is compared STRUCTURALLY with the inventory's immutable record — the header's (frozen) or the admit-once sidecar's: (1) the
    done predicates equal; (2) the identities of the record's units unchanged; (3) the unit set = the record's plus every RECEIPTED `rerun-units`
    stamp's units (the stamp's own rule, rerun_receipt_matches) — anything else refuses by name, naming the ruling needed. An inventory with no
    record is admitted at the current contract ONCE under --admit-once (the sidecar is written, never the inventory); the block written by
    --write (version, sha256, record_sha256, source, admitted_utc, amendments, both shas) is what `coverage --contract` and `verdict --contract`
    record. Exit 2 on a refusal; nothing bound, no coverage written."""
    inv_p = os.path.join(a.state_root, "gate", "expected_units.json")
    if not os.path.isfile(inv_p):
        print(f"REFUSED: gate-contract: no frozen inventory at {inv_p}", file=sys.stderr); return 2
    inv = json.load(open(inv_p, encoding="utf-8"))
    try:
        preds = probe_done_predicates(a.probe)
        rec, src, where = frozen_coverage_contract(a.state_root)
    except (ValueError, OSError, SyntaxError) as exc:
        print(f"REFUSED: gate-contract: {exc}", file=sys.stderr); return 2
    cur = coverage_contract(inv.get("units") or [], preds)
    ledger = os.path.join(a.state_root, "submissions.jsonl")
    block: Dict[str, object] = {"version": cur["version"], "sha256": cur["sha256"], "record_sha256": None, "record_version": None, "source": None, "admitted_utc": None, "amendments": [],
                                "probe_code_sha": a.probe_sha, "probe_sha256": a.probe_sha256, "inventory_code_sha": inv.get("code_sha"), "ruling": COVERAGE_CONTRACT_RULING}
    if rec is None:
        if not a.admit_once:
            print(f"REFUSED: gate-contract: the inventory {inv_p} ({str(inv.get('code_sha'))[:12]}, frozen {inv.get('frozen_utc')}) carries no coverage contract — frozen "
                  f"before the field existed; `06 --gate` admits it at the current contract ONCE (--admit-once; D3003 (2))", file=sys.stderr); return 2
        side_p = coverage_contract_sidecar(a.state_root)
        record = coverage_contract_record(inv.get("units") or [], preds, source="admitted_once", admitted_utc=utc_now(), inventory_code_sha=inv.get("code_sha"),
                                          inventory_frozen_utc=inv.get("frozen_utc"), probe_code_sha=a.probe_sha, probe_sha256=a.probe_sha256,
                                          note="D3003 (2): an inventory frozen before the field existed is admitted at the current contract once; the admitted set is "
                                               "the inventory's units as stamped at this moment (the receipted rerun pairs included)")
        block.update(record_sha256=record["sha256"], record_version=record["version"], source="admitted_once", admitted_utc=record["admitted_utc"])
        if a.dry_run:
            print(f"DRY-RUN: coverage contract {cur['version']} {cur['sha256'][:12]}: would be ADMITTED ONCE over {len(cur['unit_set'])} unit(s) — the inventory was frozen "
                  f"before the field existed (D3003 (2)) -> {side_p}")
        else:
            write_atomic(side_p, record)
            print(f"coverage contract {cur['version']} {cur['sha256'][:12]}: ADMITTED ONCE at {record['admitted_utc']} over {len(cur['unit_set'])} unit(s) — the inventory "
                  f"({str(inv.get('code_sha'))[:12]}, frozen {inv.get('frozen_utc')}) was frozen before the field existed (D3003 (2)); written into every verdict; "
                  f"record -> {side_p}")
    else:
        # the record's VERSION, shape and DIGEST first (coverage_contract_record_check): a record of another contract version cannot be compared
        # structurally (its rule differs — a ruling), and a record whose digest does not cover its body was edited by hand; either refuses before
        # a single predicate or identity is read (a cc2 record over the same body passed as cc1 here)
        body, why = coverage_contract_record_check(rec, src, where)
        if why:
            print(f"REFUSED: 06 --gate: {why}; nothing bound, no coverage written", file=sys.stderr); return 2
        when = (f"frozen with the inventory at {rec.get('frozen_utc') or inv.get('frozen_utc')}" if src == "frozen" else f"admitted once on {rec.get('admitted_utc')}")
        pred = coverage_contract_diff(body, cur, units=False)
        if pred:
            print(f"REFUSED: 06 --gate: the current probe's done predicates differ from the ones {when}: {'; '.join(pred)} — a changed predicate is a new wave or a "
                  f"ruling; the operator must rule (D3003 condition 2) before this gate reads the started wave; nothing bound, no coverage written", file=sys.stderr); return 2
        ident = [ln for ln in coverage_contract_diff(body, cur) if ln.startswith("identity ")]
        if ident:
            print(f"REFUSED: 06 --gate: the inventory's unit identities differ from the coverage contract {when}: {'; '.join(ident[:8])}{' …' if len(ident) > 8 else ''} — "
                  f"the inventory was edited; a changed identity is a new wave or a ruling (D3003 condition 2); nothing bound, no coverage written", file=sys.stderr); return 2
        receipts = rerun_receipts(ledger)
        rec_set = set(body["unit_set"]); allowed = set(rec_set); amendments: List[dict] = []
        for rr in inv.get("reruns") or []:
            added = sorted(lw for lw in (rr.get("units") or {}).values() if lw not in rec_set)
            if not added:
                continue
            decl = {k: rr.get(k) for k in RERUN_RECEIPT_KEYS}
            cov_r = [r for r in receipts if rerun_receipt_matches(r, decl, str(rr.get("code_sha") or ""))]
            if not cov_r:
                print(f"REFUSED: 06 --gate: unit(s) {added} were stamped on the inventory by rerun {rr.get('slug')!r} ({rr.get('source')}) but the attempts ledger carries no "
                      f"matching `rerun_units` receipt — a stamp without its receipt amends nothing (D3003 (2)); nothing bound, no coverage written", file=sys.stderr); return 2
            amendments.append({"slug": rr.get("slug"), "units": added, "receipt_id": cov_r[-1].get("receipt_id"), "receipt_utc": cov_r[-1].get("utc"),
                               "stamped_utc": rr.get("stamped_utc"), "source": rr.get("source"), "code_sha": rr.get("code_sha")})
            allowed.update(added)
        cur_set = set(cur["unit_set"])
        extra = sorted(cur_set - allowed)
        if extra:
            print(f"REFUSED: 06 --gate: unit(s) {extra} are not in the {src} coverage contract's set and no rerun receipt adds them — the inventory was edited by hand "
                  f"(D3003 (2)); nothing bound, no coverage written", file=sys.stderr); return 2
        gone = sorted(rec_set - cur_set)
        if gone:
            print(f"REFUSED: 06 --gate: unit(s) {gone} of the {src} coverage contract are missing from the inventory — the inventory was edited by hand (D3003 (2)); "
                  f"nothing bound, no coverage written", file=sys.stderr); return 2
        block.update(record_sha256=rec.get("sha256"), record_version=rec.get("version"), source=src, admitted_utc=rec.get("admitted_utc"), amendments=amendments)
        print(f"coverage contract {cur['version']} {cur['sha256'][:12]}: {when}, record {rec.get('version')} {str(rec.get('sha256'))[:12]} + {len(amendments)} receipted amendment(s)"
              + "".join(f" ({am['slug']}: +{am['units']}, receipt {am['receipt_id']})" for am in amendments) + " — ok")
    if a.write and not a.dry_run:
        write_atomic(a.write, block)
    elif a.write:
        print(f"DRY-RUN: write {a.write}")
    return 0


def parse_pending_capture(text: str, tasks: Optional[Dict[str, Optional[int]]]) -> Tuple[List[dict], List[str], List[str]]:
    """(runs, missing paths, notes) of 06 --gate's framed capture — `PENDING <file>` followed by the file's text, or `MISSING <file>`, per
    pending file (the ONE framing both the plain job and the array use). Every run the job registered: run_id + the eval it is (plan.eval_id)
    + the object it serves (route.object) + the array task (from `tasks`, raw job id -> index; None for a plain job). A file that does not
    parse raises ValueError NAMING it — never an empty list: the old shell read the empty substitution of an `Argument list too long` as
    "the pending file names no runs"."""
    out: List[dict] = []; missing: List[str] = []; notes: List[str] = []
    cur: Optional[str] = None; buf: List[str] = []

    def flush() -> None:
        if cur is None:
            return
        base = os.path.basename(cur)
        raw = base[:-5] if base.endswith(".json") else base
        body = "".join(buf)
        try:
            d = json.loads(body)
        except json.JSONDecodeError as exc:
            raise ValueError(f"pending file {cur}: {exc} ({len(body)} bytes read)")
        if not isinstance(d, dict):
            raise ValueError(f"pending file {cur}: not a JSON object")
        rs = d.get("runs") or ([{"run_id": d["run_id"], "plan": d.get("plan") or {}}] if d.get("run_id") else [])
        sk = d.get("skipped") or []
        if not rs:
            notes.append(f"pending file {cur} names no run (keys: {sorted(d)}; skipped: {len(sk) if isinstance(sk, list) else sk!r})")
        for r in rs:
            if not isinstance(r, dict) or not r.get("run_id"):
                raise ValueError(f"pending file {cur}: a runs[] entry carries no run_id")
            out.append({"run_id": r["run_id"], "eval_id": (r.get("plan") or {}).get("eval_id"), "object": (r.get("route") or {}).get("object"),
                        "array_task": (tasks or {}).get(raw) if tasks is not None else None, "store_uri": None})
    for ln in text.splitlines(keepends=True):
        if ln.startswith("PENDING "):
            flush(); cur = ln[8:].strip(); buf = []
        elif ln.startswith("MISSING "):
            flush(); cur = None; missing.append(ln[8:].strip())
        elif cur is not None:
            buf.append(ln)
    flush()
    return out, missing, notes


def cmd_pending_runs(a) -> int:
    """stdout = the runs list JSON and NOTHING else (06 redirects it to a file and loads it as one document); the MISSING paths go to
    --missing-out (always written, `[]` when none); a capture that cannot be parsed REFUSES (exit 2) by the file's name."""
    try:
        with open(a.capture, encoding="utf-8") as fh:
            text = fh.read()
    except OSError as exc:
        print(f"REFUSED: pending-runs: the capture {a.capture} cannot be read ({exc})", file=sys.stderr); return 2
    tasks: Optional[Dict[str, Optional[int]]] = None
    if a.tasks_json is not None:
        tasks = {str(t["raw"]): t.get("task") for t in json.loads(arg_doc(a.tasks_json) or "[]") if t.get("task") is not None}
    if not text.strip():
        print(f"REFUSED: pending-runs: the capture {a.capture} is empty — the remote command printed neither PENDING nor MISSING; nothing bound", file=sys.stderr); return 2
    try:
        runs, missing, notes = parse_pending_capture(text, tasks)
    except ValueError as exc:
        print(f"REFUSED: pending-runs: {exc} — the registration could not be read; nothing bound", file=sys.stderr); return 2
    with open(a.missing_out, "w", encoding="utf-8") as fh:
        json.dump(missing, fh)
    for n in notes:
        print(f"note: {n}", file=sys.stderr)
    print(json.dumps(runs))
    return 0


def cmd_probe_append(a) -> int:
    """ONE JSONL line per probe call: {run_id, array_task, probe, probe_error, probe_rc, probe_stderr_head, store_source}. `probe_error` is
    --why (a `none` host: the probe never ran), else `the probe exited <rc>: <first stderr line>` when rc != 0 (ssh's 255 named), else `printed
    no JSON` only when rc == 0 and stdout is not a JSON object — an exec failure never reads as "no JSON" (a `2>/dev/null` around the
    call hides every traceback). Nothing is accumulated in a shell variable: the results FILE is the record (MAX_ARG_STRLEN).
    The STATUS says whether a document was READ, after the line is written either way: rc 0 = a document recorded, or `--why` named a coverage
    FACT (no store on record: the probe never ran and nothing is wrong with the reader); rc 3 = recorded, and NO document was read — the call
    exited non-zero, or exited 0 and printed no JSON object (a login shell's banner before the document, a probe that printed prose). The second
    is a READER failure 06 --gate marks on the unit's family exactly like an ssh rc: before this an rc-0 call with unparseable stdout was recorded
    as probe_error and returned 0, so it entered no failure watermark and the pass wrote a fresh `pass` verdict over it.
    """
    def _read(p: str) -> str:
        try:
            with open(p, encoding="utf-8", errors="replace") as fh:
                return fh.read()
        except OSError:
            return ""
    out, err = _read(a.out_file), _read(a.err_file)
    rc = int(a.rc)
    doc = None; why: Optional[str] = None
    s = out.strip()
    if a.why:
        why = a.why
    elif rc != 0:
        head = next((ln for ln in err.splitlines() if ln.strip()), "") or next((ln for ln in out.splitlines() if ln.strip()), "")
        why = f"the probe exited {rc}" + (" (ssh could not reach the host or the command was refused)" if rc == 255 else "") + (f": {head[:300]}" if head else " (no stderr)")
    else:
        try:
            d = json.loads(s) if s else None
        except json.JSONDecodeError:
            d = None
        if isinstance(d, dict):
            doc = d
        else:
            why = f"the probe printed no JSON document (rc 0; stdout {len(out)} bytes: {s[:120]!r})"
    line = {"run_id": a.run_id, "array_task": (int(a.array_task) if str(a.array_task or "") != "" else None), "probe": doc, "probe_error": why, "probe_rc": rc,
            "probe_stderr_head": ("\n".join(err.splitlines()[:5])[:600] or None), "store_source": a.store_source or None}
    os.makedirs(os.path.dirname(os.path.abspath(a.results)) or ".", exist_ok=True)
    with open(a.results, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(line, sort_keys=True) + "\n")
    return 0 if (doc is not None or a.why) else 3


def bound_array_tasks(state_root: str, lwid: str) -> Dict[str, List[int]]:
    """{run_id: sorted array task indices} the unit's REGISTERED binding names for each run — the per-task rows 06 --gate bound (`runs`:
    one per task's pending file, each with its `array_task`). The climb array's four tasks
    register ONE run per cell, so one run id is bound to four tasks; 06's coverage pass keys its run list by run id (one row, the LAST
    task's index), so the probe plan and the coverage record read the task identities from the binding itself, never from that row.
    {} without a registered binding (the per-run `array_task` is then the only index)."""
    b = binding_state(os.path.join(state_root, "submissions.jsonl"), lwid)
    out: Dict[str, set] = {}
    for r in b.get("runs") or []:
        if r.get("run_id") and r.get("array_task") is not None:
            out.setdefault(str(r["run_id"]), set()).add(int(r["array_task"]))
    return {rid: sorted(s) for rid, s in out.items()}


def probe_calls(unit: dict, run: dict, shared_root: str, tasks: Optional[List[int]] = None) -> List[Tuple[Optional[int], Optional[str], Optional[List[str]], Optional[str]]]:
    """[(array_task, host, argv, reason-when-none)]: EVERY coverage_probe.py call one BOUND run of a frozen unit needs. One call for every
    layout but the climb's; for `bon_climb_flat` ONE CALL PER ARRAY TASK the binding names for the run (`tasks`, bound_array_tasks) — the same
    argv per task, `--shard-index k`, byte for byte — so the shared-run shape (one run id bound to the cell's four tasks since 2026-10-11)
    is read over every one of its shards by the coverage_probe.py at the inventory's FROZEN sha, which reads ONE
    `--shard-index` per call (it takes an int and a repeated flag keeps the last: every probe before 2026-10-11, the demo inventory's pin among
    them). 06 --gate runs each call and `stage.py coverage` folds a run's documents into one (fold_probe_results: `shards` per task,
    `shard_indices` = the tasks asked, the per-call provenance under `fold`). A plan of one call with the flag repeated read the LAST
    task's shard alone from the frozen probe, and admission then refused a whole climb by name.
    No task named (`tasks` empty) -> the run's own `array_task`, one call, as before."""
    ev = unit["evals"].get(run.get("eval_id"))
    if ev is not None and ev["coverage"]["layout"] == "bon_climb_flat":
        idx = sorted({int(t) for t in (tasks or []) if t is not None}) or ([int(run["array_task"])] if run.get("array_task") is not None else [])
        if idx:
            return [(k, *probe_argv(unit, {**run, "array_task": k}, shared_root, tasks=[k])) for k in idx]
    return [(None, *probe_argv(unit, run, shared_root))]


def probe_argv(unit: dict, run: dict, shared_root: str, tasks: Optional[List[int]] = None) -> Tuple[Optional[str], Optional[List[str]], Optional[str]]:
    """(host, coverage_probe.py argv, reason-when-none) for ONE call on one BOUND run of a frozen unit: the eval's coverage spec names the
    layout, the host and the artifact path (`store_uri` = the run row's recorded store; `root_rel` = under the cluster's shared root, `<cell>`
    = the run's object); an unverified layout or a run without a recorded store yields no argv and the reason. A climb run is probed at ONE
    shard per call (`--shard-index k`: `tasks`' one task, else the run's own `array_task`); a run bound to SEVERAL tasks has several calls
    (probe_calls, one per task) and is refused here by name — one call reads one shard at the inventory's frozen sha, never the cell."""
    ev = unit["evals"].get(run.get("eval_id"))
    if ev is None:
        return None, None, f"run {run.get('run_id')} is eval {run.get('eval_id')!r}, not one of the unit's"
    cov = ev["coverage"]; lay = cov["layout"]
    if lay == "unverified":
        return None, None, cov.get("reason") or "unverified layout"
    # the store from the AUTHORITATIVE record, never the frozen `coverage.path` (root_rel + the run's object — a recomputation that
    # read "0/30" where the grouped multi-turn targets wrote under the reproduce root; `shared_root` is kept for the CLI's shape and unused)
    path, _src, why = probe_store(unit, run)
    if why:
        return None, None, why
    argv = ["probe", "--layout", lay, "--path", path]
    if lay == "jailbreak_cellstore":
        argv += ["--shards", ",".join(ev["expected_shards"])]
    elif lay == "bon_store512":
        argv += ["--n-max", str(cov.get("n_max") or 0)]
    elif lay == "bon_climb_flat":
        # The shards this run's coverage is read from are the array tasks the BINDING names for it —
        # one call per task, `--shard-index k`, byte for byte the same argv (every task registered its own run), whether the task is the run's
        # own or one of the four the shared-run shape binds to one run id (probe_calls fans those out; `stage.py coverage` folds the documents).
        # Never the flag repeated in ONE call — the coverage_probe.py at the inventory's frozen sha takes one int (the
        # last flag wins), so that call read the last task's shard alone from the mirrored checkout and admission refused a whole climb by name.
        idx = sorted({int(t) for t in (tasks or []) if t is not None}) or ([int(run["array_task"])] if run.get("array_task") is not None else [])
        if not idx:
            return None, None, f"run {run.get('run_id')} carries no array task index — the climb's shard cannot be located"
        if len(idx) > 1:
            return None, None, (f"run {run.get('run_id')} is bound to array tasks {idx}: the climb is probed once per task (probe_calls), never in one call — "
                                f"the coverage_probe.py at the inventory's frozen sha reads one --shard-index per call")
        argv += ["--cell", str(run.get("object")), "--shard-index", str(idx[0]), "--nshards", str(cov.get("nshards") or len(ev["expected_shards"]))]
    elif lay == "capability_cellstore":
        argv += ["--task", cov["task"]]
    elif lay == "misalign_responses":
        argv += ["--cell", str(run.get("object")), "--behavior", cov["behavior"], "--n-samples", str(cov["n_samples"])]
    elif lay == "petri_cellstore":
        if cov.get("behavior"):
            argv += ["--behavior", cov["behavior"]]
    return cov["host"], argv, None


def cmd_probe_plan(a) -> int:
    unit = frozen_unit(a.state_root, a.logical_work_id)
    # `--probe-map label=path;…` names each cluster's probe (its mirrored checkout at the frozen sha); a superseded run's
    # probe is planned on its successor's cluster (the FIFTH column names it; empty = the unit's own cluster) with that cluster's probe
    probe_map = dict(kv.split("=", 1) for kv in (getattr(a, "probe_map", "") or "").split(";") if "=" in kv)
    tasks_by_rid = bound_array_tasks(a.state_root, a.logical_work_id)      # the task identities, from the binding (06 keys its rows by run id)
    # one TSV line PER CALL (probe_calls): run id, host, argv, reason, the successor's cluster, and the SIXTH column — the array task the call
    # reads ('' for a layout probed in one call) — so 06 --gate can hand each document back to `coverage --probes-json` under its task and the
    # fold can tie the call to what it read. A climb run bound to several tasks is several lines with the same run id.
    # the SEVENTH column names where the store came from (probe_store: `registry.run.store_uri`, `registry.run.extra_json.climb_root`, …, `none`
    # for a run with no store on record); 06 --gate hands it to probe-append and the coverage record keeps it per run
    for r in json.loads(arg_doc(a.runs_json) or "[]"):
        pcl = str(((r.get("successor") or {}).get("cluster") or "")) if r.get("successor") else ""
        _path, src, _why = probe_store(unit, r)
        for task, host, argv, why in probe_calls(unit, r, a.shared, tasks=tasks_by_rid.get(str(r.get("run_id")))):
            col6 = "" if task is None else str(task)
            if argv is None:
                sys.stdout.write("\t".join([r["run_id"], "none", "", why or "", pcl, col6, src or "none"]) + "\n")
            else:
                sys.stdout.write("\t".join([r["run_id"], host, " ".join(shlex.quote(x) for x in [probe_map.get(pcl, a.probe) if pcl else a.probe, *argv]), "", pcl, col6, src or "none"]) + "\n")
    return 0


def fold_probe_results(rid: str, tasks: Optional[List[int]], results: List[dict]) -> Tuple[Optional[dict], Optional[str]]:
    """(probe document, probe_error) for one bound run from the results of ITS probe calls — 06 --gate's `{run_id, array_task, probe, probe_error}`
    per call. One call planned and one result: the document as the probe printed it, byte for byte (every layout but the climb's; the per-task
    climb shape). A climb run bound to several `tasks` (the shared-run shape, probe_calls): the per-task documents FOLDED into one —
    `shards` = each call's shard block as printed; `shard_indices` = the tasks asked, in order; `fold.calls` = what each call read; the top-level
    notes concatenated — and refused by name (probe_error set, the fold of the good calls kept beside it for the audit) when a task has no result
    or two, a call failed, or a call's document does not read exactly its task's shard (`shard{k:02d}`; a `shard_indices` of its own, when the
    probe prints one, must name that task alone): the call and its document must agree, or the gate has read nothing it can vouch for."""
    want = sorted({int(t) for t in (tasks or []) if t is not None})
    if len(want) <= 1 and len(results) == 1:
        r = results[0]
        return r.get("probe"), r.get("probe_error")
    if not results:
        return None, f"no probe result for run {rid} — re-run 06 --gate"
    by_task: Dict[Optional[int], List[dict]] = {}
    for r in results:
        t = r.get("array_task")
        by_task.setdefault(None if t is None else int(t), []).append(r)
    asked = want or sorted(t for t in by_task if t is not None)
    errors: List[str] = []
    shards: Dict[str, dict] = {}; calls: List[dict] = []; notes: List[str] = []; paths: List[str] = []; exists = True; layout = None
    if None in by_task:
        errors.append(f"{len(by_task[None])} probe result(s) name no array task (a per-task climb call names the task it read)")
    for extra in sorted(t for t in by_task if t is not None and t not in asked):
        errors.append(f"array task {extra}: a probe result for a task the binding does not name for the run")
    for k in asked:
        rs = by_task.get(k) or []
        name = f"shard{k:02d}"
        if not rs:
            errors.append(f"array task {k}: no probe result (its call did not run) — re-run 06 --gate"); continue
        if len(rs) > 1:
            errors.append(f"array task {k}: {len(rs)} probe results for one call"); continue
        r = rs[0]; pr = r.get("probe")
        if r.get("probe_error") or not isinstance(pr, dict):
            errors.append(f"array task {k}: {r.get('probe_error') or 'no probe document'}"); continue
        got = sorted(pr.get("shards") or {})
        if got != [name] or ("shard_indices" in pr and [int(x) for x in pr["shard_indices"]] != [k]):
            errors.append(f"array task {k}: the probe document reads shard(s) {got}{' (shard_indices ' + repr(pr['shard_indices']) + ')' if 'shard_indices' in pr else ''}, "
                          f"not {name} alone — the call and its document disagree"); continue
        layout = layout or pr.get("layout")
        if pr.get("layout") != layout:
            errors.append(f"array task {k}: the probe document reads layout {pr.get('layout')!r}, the other calls {layout!r}"); continue
        if pr.get("path") not in paths:
            paths.append(pr.get("path"))
        exists = exists and bool(pr.get("exists"))
        shards[name] = pr["shards"][name]
        notes.extend(pr.get("notes") or [])
        calls.append({"array_task": k, "shards": [name], "shard_indices": pr.get("shard_indices"), "exists": bool(pr.get("exists"))})
    if len(paths) > 1:
        errors.append(f"the calls read different paths: {paths}")
    folded = {"layout": layout or "bon_climb_flat", "path": paths[0] if paths else None, "exists": exists, "shards": shards, "shard_indices": list(asked), "notes": notes,
              "fold": {"note": "one coverage_probe.py call per bound array task (the probe at the inventory's frozen sha reads one --shard-index per call); "
                               "folded by stage.py coverage from the per-call documents", "calls": calls}}
    return folded, ("; ".join(errors) or None)


FROZEN_PATH_SPEC_NOTE = ("ignored (D3003 (1)): every store was read at the location the registry recorded for the run (per run `store_source`), never at the "
                         "inventory's frozen coverage.path (kept in the inventory as history; nothing reads it)")


def cmd_coverage(a) -> int:
    runs = json.loads(arg_doc(a.runs_json) or "[]")
    tasks_by_rid = bound_array_tasks(a.state_root, a.logical_work_id)      # every task the binding names per run id, kept beside the row's one
    results: List[dict] = []
    if getattr(a, "probes_json", None):
        results += json.loads(arg_doc(a.probes_json) or "[]")
    if getattr(a, "probes_jsonl", None):
        # 06 --gate's results FILE (probe-append, one line per call): never a shell variable — mmlu's 1,531-id documents overflowed MAX_ARG_STRLEN
        # after the fifth run and `coverage` wrote probe null for all ten
        with open(a.probes_jsonl, encoding="utf-8") as fh:
            for ln in fh:
                if ln.strip():
                    results.append(json.loads(ln))
    if getattr(a, "probes_json", None) or getattr(a, "probes_jsonl", None):
        # per-CALL results (probe-plan's lines run one by one): a run's documents folded into its one record (fold_probe_results) — the shared-run
        # climb's four single-shard documents become one over the cell's shards, naming what was asked; a run NO result line names is said so, loudly
        by_rid: Dict[str, List[dict]] = {}
        for res in results:
            by_rid.setdefault(str(res.get("run_id")), []).append(res)
        for r in runs:
            mine = by_rid.get(str(r["run_id"]), [])
            r["probe"], r["probe_error"] = fold_probe_results(r["run_id"], tasks_by_rid.get(str(r.get("run_id"))), mine)
            if not mine:
                r["probe_error"] = f"no probe result recorded for run {r['run_id']} (the gate loop never reached it) — re-run 06 --gate"
            r["probe_rc"] = ([m.get("probe_rc") for m in mine] if len(mine) > 1 else (mine[0].get("probe_rc") if mine else None))
            r["probe_stderr_head"] = ([m.get("probe_stderr_head") for m in mine] if len(mine) > 1 else (mine[0].get("probe_stderr_head") if mine else None))
            r["store_source"] = next((m.get("store_source") for m in mine if m.get("store_source")), r.get("store_source"))
    contract = json.loads(arg_doc(a.contract)) if getattr(a, "contract", None) else None
    doc = {"logical_work_id": a.logical_work_id, "run_ids": [r["run_id"] for r in runs], "collected_utc": utc_now(),
           # both shas on every coverage record, beside the coverage contract the gate read under
           "inventory_code_sha": getattr(a, "inventory_sha", None), "probe_code_sha": getattr(a, "probe_sha", None), "probe_sha256": getattr(a, "probe_sha256", None),
           "coverage_contract": contract, "frozen_path_spec": FROZEN_PATH_SPEC_NOTE,
           "runs": {r["run_id"]: {"eval_id": r.get("eval_id"), "object": r.get("object"), "array_task": r.get("array_task"), "store_uri": r.get("store_uri"),
                                  "array_tasks": tasks_by_rid.get(str(r.get("run_id"))),
                                  "store_source": r.get("store_source"),                 # which record named the store (`none` = no store on record)
                                  "probe": r.get("probe"), "probe_error": r.get("probe_error"),
                                  "probe_rc": r.get("probe_rc"), "probe_stderr_head": r.get("probe_stderr_head"),
                                  "successor": r.get("successor")} for r in runs}}       # the successor the probe read
    p = coverage_path(a.state_root, a.logical_work_id)
    if a.dry_run:
        print(f"DRY-RUN: write {p} ({len(runs)} run(s))"); return 0
    write_atomic(p, doc)
    print(f"coverage {a.logical_work_id}: {len(runs)} run(s) -> {p}")
    return 0


def verdict_needs_provenance(state_root: str, family: str) -> bool:
    """Whether `family` is a family of the frozen 06 inventory under `state_root` — such a verdict records the inventory's and the probe's shas and
    the coverage contract; stage 07's `bcb_bfcl/<view>` verdicts (their own two-unit inventory under gate/bcb_bfcl/) have no 06
    inventory and keep their shape."""
    inv_p = os.path.join(state_root, "gate", "expected_units.json")
    if not os.path.isfile(inv_p):
        return False
    try:
        fams = {u.get("family") for u in (json.load(open(inv_p, encoding="utf-8")).get("units") or [])}
    except (json.JSONDecodeError, OSError, AttributeError):
        return False
    return family in fams


def parse_verdict_args(headline, range_s) -> Tuple[float, float, float]:
    """(headline, lo, hi) from `--headline <value>` / `--range <lo>,<hi>`, or a ValueError naming the flag and the value it got. `06 --gate` checks
    the pair with this BEFORE it binds or covers anything, and cmd_verdict refuses by name (rc 2, nothing written) instead of raising: a malformed
    --range used to reach `stage.py verdict` only AFTER the whole family was bound and covered, its traceback's rc 1 was the same status a persisted
    `fail` returned, and 06 swallowed both — the pass exited 0 with the OLD verdict (a `pass`) standing over the coverage it had just refreshed, which
    admission reads as admissible (Major 1)."""
    try:
        h = float(str(headline).strip())
    except (TypeError, ValueError):
        raise ValueError(f"--headline must be a number (got {headline!r})")
    parts = [x.strip() for x in str(range_s).split(",")]
    if len(parts) != 2 or not all(parts):
        raise ValueError(f"--range must be <lo>,<hi> — two numbers separated by one comma (got {range_s!r})")
    try:
        lo, hi = float(parts[0]), float(parts[1])
    except ValueError:
        raise ValueError(f"--range must be <lo>,<hi> — two numbers (got {range_s!r})")
    if any(math.isnan(x) or math.isinf(x) for x in (h, lo, hi)):
        raise ValueError(f"--headline and --range must be finite numbers (got --headline {headline!r} --range {range_s!r})")
    if lo > hi:
        raise ValueError(f"--range must be <lo>,<hi> with lo <= hi (got {range_s!r})")
    return h, lo, hi


def headline_record_problems(rec, family: str, run_ids: Sequence[str]) -> List[str]:
    """Why a `--headline-record` (lib/gate_headline.py read's record) cannot be ruled on as THIS verdict: another family's, a skipped read, no
    headline, no leg, a read taken while a unit of the family was not registered (a family is ruled whole), or a read of other runs than the
    verdict binds (the record's run_ids must be the verdict's --run-ids, as a set)."""
    if not isinstance(rec, dict):
        return ["the record is not a JSON object"]
    out = []
    if rec.get("family") != family:
        out.append(f"it reads family {rec.get('family')!r}, not {family!r}")
    if rec.get("skipped"):
        out.append(f"it was skipped, not ready to rule: {'; '.join(rec['skipped'])}")
    elif not isinstance(rec.get("headline"), (int, float)) or isinstance(rec.get("headline"), bool):
        out.append(f"it carries no headline (got {rec.get('headline')!r})")
    if rec.get("units_not_registered"):
        out.append(f"it was read while unit(s) {', '.join(map(str, rec['units_not_registered']))} were not registered")
    if not rec.get("skipped") and not (isinstance(rec.get("legs"), list) and rec["legs"]):
        out.append("it carries no leg")
    rng = rec.get("range")
    if not (isinstance(rng, list) and len(rng) == 2):
        out.append(f"it carries no range (got {rng!r})")
    if set(rec.get("run_ids") or []) != set(run_ids):
        out.append(f"it read the runs {sorted(rec.get('run_ids') or [])[:6]}…, not the {len(run_ids)} run(s) this verdict binds (--run-ids)")
    return out


def cmd_verdict(a) -> int:
    """rc 0 = the verdict is RECORDED — `pass` or `fail`, said on the line printed and in gate/<family>.json: a persisted `fail` is a successful
    ruling, not an error (returning 1 would be the same status as a traceback, and 06 could not then tell a ruled `fail` from a verdict that was
    never written); 2 = refused by name, nothing written (a malformed --headline / --range; a 06 family without its provenance;
    --headline-record beside --headline / --range, or a record that does not fit this verdict); anything else is an error that propagates as one
    (a write that failed: no verdict was written). A caller reads the verdict from the file or the line, never from the status.
    The headline comes from ONE of: `--headline X --range LO,HI` (by hand), or `--headline-record @<file>` (06 --gate --verdict: the record
    lib/gate_headline.py read wrote, headline and range read from it, every leg recorded under `headline_legs`)."""
    run_ids = [r for r in a.run_ids.split(",") if r]
    rec = None
    if a.headline_record is not None:
        if a.headline is not None or a.range is not None:
            print("REFUSED: verdict: --headline-record carries the headline and the range; pass it alone, or --headline and --range by hand — nothing written",
                  file=sys.stderr); return 2
        try:
            rec = json.loads(arg_doc(a.headline_record))
        except (OSError, json.JSONDecodeError) as e:
            print(f"REFUSED: verdict: --headline-record {a.headline_record!r} could not be read ({e}) — nothing written", file=sys.stderr); return 2
        why = headline_record_problems(rec, a.family, run_ids)
        if why:
            print(f"REFUSED: verdict: --headline-record does not fit this verdict: {'; '.join(why)} — nothing written", file=sys.stderr); return 2
        headline_s, range_s = repr(float(rec["headline"])), f"{rec['range'][0]!r},{rec['range'][1]!r}"
    elif a.headline is None or a.range is None:
        print("REFUSED: verdict: pass --headline <value> and --range <lo,hi>, or --headline-record @<file> — nothing written", file=sys.stderr); return 2
    else:
        headline_s, range_s = a.headline, a.range
    try:
        h, lo, hi = parse_verdict_args(headline_s, range_s)
    except ValueError as e:
        print(f"REFUSED: verdict: {e} — nothing written", file=sys.stderr); return 2
    v = a.verdict or ("pass" if lo <= h <= hi else "fail")
    doc = {"family": a.family, "headline": h, "plausible_range": [lo, hi], "verdict": v, "ruled_utc": utc_now(),
           "run_ids": run_ids, "note": a.note or ""}
    if rec is not None:
        doc.update(headline_rule=rec.get("rule"), expected=rec.get("expected"), headline_legs=rec.get("legs") or [],
                   headline_held_out=rec.get("held_out") or [])
    inv_sha, probe_sha, probe_sha256, contract_arg = (getattr(a, k, None) for k in ("inventory_sha", "probe_sha", "probe_sha256", "contract"))
    if verdict_needs_provenance(a.state_root, a.family):
        missing = [f for f, val in (("--inventory-sha", inv_sha), ("--probe-sha", probe_sha), ("--contract", contract_arg)) if not val]
        if missing:
            print(f"REFUSED: verdict: family {a.family!r} is a family of the frozen 06 inventory, whose verdicts record the inventory's and the probe's commits and the "
                  f"coverage contract they were ruled under (D3003 (2)/(3)): missing {missing} — rule it through `06_eval_waves.sh --gate {a.family} --headline … "
                  f"--range …`, which supplies them (stage 07's bcb_bfcl/<view> verdicts have no 06 inventory and need none)", file=sys.stderr); return 2
    if inv_sha or probe_sha or probe_sha256 or contract_arg:
        doc.update({"inventory_code_sha": inv_sha, "probe_code_sha": probe_sha, "probe_sha256": probe_sha256,
                    "coverage_contract": json.loads(arg_doc(contract_arg)) if contract_arg else None})
    p = os.path.join(a.state_root, "gate", f"{a.family}.json")
    if a.dry_run:
        print(f"DRY-RUN: write {p} verdict={v}")
        return 0
    write_atomic(p, doc)
    print(f"gate verdict {a.family}: {v} (headline {headline_s} in [{lo}, {hi}]) -> {p}")
    return 0


def cmd_status(a) -> int:
    inv = os.path.join(a.state_root, "gate", "expected_units.json")
    if not os.path.isfile(inv):
        print(f"no frozen inventory at {inv}"); return 1
    doc = json.load(open(inv, encoding="utf-8"))
    ledger = os.path.join(a.state_root, "submissions.jsonl")
    for u in doc["units"]:
        b = binding_state(ledger, u["logical_work_id"])
        vp = os.path.join(a.state_root, "gate", f"{u['family']}.json")
        v = json.load(open(vp))["verdict"] if os.path.isfile(vp) else "-"
        print(f"{u['logical_work_id']:60s} {b['state']:10s} job={b.get('job_id','-')} runs={','.join(b.get('run_ids', []) or ['-'])} verdict={v}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan"); p.add_argument("--model", required=True); p.add_argument("--view", required=True)
    p.add_argument("--families", default=""); p.add_argument("--nshards", type=int, default=9); p.add_argument("--sha", default=None)
    p.add_argument("--clusters", default="", help="the PLACEMENT universe (every configured label): the multi-turn groups are partitioned over it")
    p.add_argument("--only-clusters", default="", help="SELECT the placed units on these labels (a subset of --clusters) for this call; `*` units stay; never re-partitions")
    p.add_argument("--models-table", default=None); p.add_argument("--json", action="store_true")
    p.add_argument("--inventory", default=None, help="26d: the frozen inventory whose pinned plan_knobs the live plan must match (a differing knob refuses); absent file -> no check")
    p.add_argument("--move-group", action="append", default=[], metavar="GROUP=LABEL",
                   help="place the named shard group's targets on LABEL instead of their round-robin cluster and size LABEL's pool from the moved count (repeatable)")
    p.add_argument("--ledger", default=None, help="the wave's attempts ledger: the moves its placement manifest DECLARES apply to this plan (sticky across calls)")
    p.add_argument("--re-move", action="store_true", help="re-point an ALREADY declared group to another label for this plan (06 passes it with the same --move-group it hands `placement-manifest ensure --re-move`)")
    p.add_argument("--cells", default=None, metavar="CELL,…", help="cells-fix (D1152): serve AND register only these cells of the view (validated against the view config's explicit_cells; written into every target job's cells positional and F22_REGISTRY_CELLS)")
    p.add_argument("--pass", dest="pass_", default="first", choices=list(PETRI_PASSES),
                   help="the Petri pass this plan is for: `first` (the wave's rows, the inventory's) or `overflow` (the ONE context-overflow re-run proxy row for --families petri, with --overflow-cells; petri-completion 2026-10-13)")
    p.add_argument("--overflow-cells", default=None, metavar="CELL,…",
                   help="with --pass overflow: the config cells the re-run proxy serves and registers (the ledgered legs' cells + `none` for a control leg; lib/petri_overflow.py derives them from the daemon's record)")
    p.add_argument("--overflow-cluster", default=None, metavar="LABEL",
                   help="with --pass overflow (required): the cluster the first pass is bound on (lib/petri_overflow.py's first_cluster) — the re-run proxy's row is pinned there (its object bank and bank view are there); a --clusters universe or --only-clusters selection that excludes it refuses")
    p.add_argument("--overrides", default=os.environ.get(RUN_OVERRIDES_ENV) or None,
                   help=f"the model's overrides file (default $SISTEER_RUN_OVERRIDES, the stage's sourced document): its {SMOKE_GOALS_KEY}=<lo>-<hi>/N narrows the multi-turn targets and "
                        "the per-view padding levels to that shard range (M10; a plan knob the frozen inventory pins); absent or without the key -> the paper's rows")
    p = sub.add_parser("placement-manifest", help="the ORDERED placement universe pinned with the wave (D296)")
    p.add_argument("cmd_placement", choices=["ensure", "check", "moves"]); p.add_argument("--model", required=True); p.add_argument("--view", required=True)
    p.add_argument("--clusters", required=True, help="the site's cluster labels, in config order"); p.add_argument("--ledger", default=None)
    p.add_argument("--overrides", default=os.environ.get(RUN_OVERRIDES_ENV) or None,
                   help=f"the model's overrides file (default $SISTEER_RUN_OVERRIDES): the plan the manifest is pinned from / checked against carries its {SMOKE_GOALS_KEY} (M10)")
    p.add_argument("--manifest", default=None); p.add_argument("--plan-json", default=None); p.add_argument("--models-table", default=None)
    p.add_argument("--nshards", type=int, default=9); p.add_argument("--sha", default=None); p.add_argument("--dry-run", action="store_true")
    p.add_argument("--move-group", action="append", default=[], metavar="GROUP=LABEL", help="ensure: DECLARE a group move (recorded with --move-reason; the resubmit's receipt is its proof)")
    p.add_argument("--move-reason", default=None, help="the provenance of a declared move (why the group leaves its cluster)")
    p.add_argument("--re-move", action="store_true", help="ensure: re-point an ALREADY declared group to another label (a new --move-reason; the earlier declaration stays in the history)")
    p = sub.add_parser("freeze-inventory"); p.add_argument("--model", required=True); p.add_argument("--roster", default=None)
    p.add_argument("--overrides", default=os.environ.get(RUN_OVERRIDES_ENV) or None,
                   help="the model's overrides file (default $SISTEER_RUN_OVERRIDES): its OVERRIDE_HOLD_OUT_EVALS are stamped `held_out` on the units that carry them (D384); "
                        f"its {SMOKE_GOALS_KEY}=<lo>-<hi>/N narrows the multi-turn targets and the padding levels the inventory freezes and is pinned in the header's plan_knobs (M10)")
    p.add_argument("--models-table", default=None); p.add_argument("--code-sha", required=True); p.add_argument("--out", required=True)
    p.add_argument("--families", default=""); p.add_argument("--nshards", type=int, default=9); p.add_argument("--dry-run", action="store_true")
    p.add_argument("--clusters", default="", help="the site's cluster labels in config order: the placement manifest is pinned beside the inventory (D296)")
    p.add_argument("--seeds", default=None, help="the registry seeds (default configs/registry, or $SISTEER_REGISTRY_SEEDS)")
    p.add_argument("--filler", default=os.environ.get("SISTEER_PG19_FILLER"), help="the PG-19 filler (padding_identity_sha256) so the five padding units get their DERIVED instrument ids at freeze time")
    p.add_argument("--goals-jsonl", default=None, help="offline goal fixture (rows {goal_id, dataset, …}) instead of the producer's dataset recipe; $SISTEER_GOALS_JSONL")
    p.add_argument("--wave-config", default=None, help="the (model, baseline view) wave config to read bon.n_max / bon.goal_subset from (default configs/sdfbase_full22_<key>_<baseline view>.yaml: u0, or the single-phase recipe's one wave)")
    p.add_argument("--petri-bundle", default=None, help=f"the frozen Petri scenario bundle dir (default ${PETRI_BUNDLE_DIR_ENV}, else ~/.local/state/petri/scenarios/<lock hash>, the lock's location, or this checkout's {PETRI_REPO_BUNDLE_REL}/<lock hash>)")
    p = sub.add_parser("hold-out", help="stamp an existing frozen inventory with the model overrides file's hold-outs (D384); ledger event; idempotent")
    p.add_argument("--out", required=True, help="the inventory (gate/expected_units.json)"); p.add_argument("--overrides", required=True); p.add_argument("--dry-run", action="store_true")
    p = sub.add_parser("rerun-units", help="stamp an existing STARTED inventory with the D913 (ii) BoN rerun pair (bon/store512_<slug> -> bon/climb_<slug>) of the named cells; ledger receipt first; idempotent")
    p.add_argument("--out", required=True, help="the inventory (gate/expected_units.json) of the wave whose 512 store the ruling re-runs")
    p.add_argument("--cells", required=True, metavar="CELL,…", help="the cell(s) the ruling re-runs (validated against the view config's explicit_cells); the pair serves and registers exactly these")
    p.add_argument("--source", required=True, help="the ruling the rerun comes from (who, when; e.g. `D913 (ii)`) — recorded on the pair, the receipt and the inventory header")
    p.add_argument("--code-sha", required=True, help="the tree the pair runs at (the corrected sbatch files): the pair's units carry it as their code_sha")
    p.add_argument("--model", default=None, help="optional cross-check: the inventory's model")
    p.add_argument("--models-table", default=None); p.add_argument("--roster", default=None, help="the wave roster view_cells_of reads (default $F22_ROSTER, else configs/sdfbase_full22_roster.yaml)")
    p.add_argument("--filler", default=os.environ.get("SISTEER_PG19_FILLER")); p.add_argument("--seeds", default=None, help="the registry seeds (default configs/registry, or $SISTEER_REGISTRY_SEEDS)")
    p.add_argument("--goals-jsonl", default=None, help="offline goal fixture (rows {goal_id, dataset, …}); $SISTEER_GOALS_JSONL")
    p.add_argument("--wave-config", default=None, help="the (model, baseline view) wave config to read bon.n_max / bon.goal_subset from (the freeze's default)")
    p.add_argument("--petri-bundle", default=None); p.add_argument("--nshards", type=int, default=9, help="the freeze's shard total (the inventory's own `nshards` wins when it carries one)")
    p.add_argument("--clusters", default="", help="the site's cluster labels in config order: the pair's workload identity is composed over them, as the freeze composes every unit's")
    p.add_argument("--overrides", default=os.environ.get(RUN_OVERRIDES_ENV) or None, help="the model's overrides document the wave runs under (the freeze's; its hold-outs stamp the pair as they stamp every unit)")
    p.add_argument("--supersede-receipt", action="store_true",
                   help="when an interrupted apply left a `rerun_units` receipt for this pair under ANOTHER tree / ruling / cells (the inventory unstamped), append a receipt "
                        "that names it superseded and stamp THIS declaration; without the flag such a conflict refuses by name (a receipt covers a stamp on its whole "
                        "declaration + tree, never the pair's name alone); refused when there is nothing to supersede")
    p.add_argument("--dry-run", action="store_true")
    p = sub.add_parser("reconcile-registrations", help="terminal run_state records (cancelled|failed per sacct) for the runs of superseded attempts no later generation continues (29); dry-run unless --write")
    p.add_argument("--ledger", required=True); p.add_argument("--registry", required=True, help="registry.sqlite (read-only; fail closed when absent)")
    p.add_argument("--config", default=None, help="the site config whose clusters sacct is read through (default $SISTEER_CONFIG)")
    p.add_argument("--spool", default=None, help="the producers' spool (default $SISTEER_REGISTRY_SPOOL, else the registry's probe); records go to <spool>/<campaign>/<run dir>/")
    p.add_argument("--units", default="", help="comma-separated logical work ids (default: every unit with a superseded attempt)")
    p.add_argument("--write", action="store_true"); p.add_argument("--out", default=None, help="write the report JSON here")
    p.add_argument("--repo", default=os.path.dirname(os.path.dirname(HERE)), help="the checkout whose HEAD stamps the producer's code_sha")
    p = sub.add_parser("sha-table"); p.add_argument("--ledger", required=True)     # the wave artifact's per-unit sha table
    p = sub.add_parser("amend-contract")                                              # the receipted contract amendment
    p.add_argument("--ledger", required=True); p.add_argument("--logical-work-id", required=True)
    p.add_argument("--reason", default=None, help="the ruling, with its D-number (required to mint)")
    p.add_argument("--commit", default=None, help="the amending commit: must be an ancestor-or-equal of --tip; the NEW decode-policy export and the seeds the new "
                                                  "contract is minted from are read from git at this sha (required to mint)")
    p.add_argument("--tip", default=None, help="the tree the unit's re-run will be submitted at (default: HEAD of --repo, the checkout this CLI reads git from)")
    p.add_argument("--registry", required=True, help="registry.sqlite (fail closed when absent); read before the sacct proofs and again right before the ledger event")
    p.add_argument("--config", default=None, help="the site config.env whose clusters the sacct proofs ssh through (default $SISTEER_CONFIG, else the default site dir's); "
                                                  "the funnel's own reader (submit_twin.Remote.sacct_tasks)")
    p.add_argument("--old-export", default=None, help="OPTIONAL cross-check only: the old decode-policy export is DERIVED from the unit's recorded tree (git show "
                                                      "<recorded sha>:configs/registry/decode_policy.json); a file given here is accepted only byte-identical to it")
    p.add_argument("--repo", default=os.path.dirname(os.path.dirname(HERE)), help="the git checkout the shas are read from")
    p.add_argument("--kind", choices=list(AMEND_KINDS), default="decode-neutral",
                   help="decode-neutral (D217: instrument ids explained by one decode-policy row's exact neutral transitions; every old job terminal) | "
                        "petri-role-fallback (D354/D355/D442/D447: the Petri drive manifest's two added per-request refusal-fallback keys, contract unchanged "
                        "or Petri instruments alone, the serving job may be alive; the receipt ADVANCES the unit's sha to --commit) | "
                        "code-follow-up (D518: a CODE-ONLY follow-up to a receipted sha advance — a strict descendant of the amended sha; manifest, roster, "
                        "contract, export, old run set and carried set unchanged; --note names the findings it carries; advances the sha again)")
    p.add_argument("--manifest-rel", default=None, help="petri-role-fallback: the drive manifest's repo-relative path (default configs/capeval/manifests/petri_full22_<model>_<view>.yaml)")
    p.add_argument("--note", default=None, help="recorded verbatim in the receipt (petri-role-fallback: the attempt's completed-sample count; code-follow-up: REQUIRED, the findings the follow-up carries)")
    p.add_argument("--recheck", action="store_true", help="re-run the three proofs over every recorded receipt of the unit against --tip and the registry as they stand now; writes nothing")
    p.add_argument("--dry-run", action="store_true")
    p = sub.add_parser("bind"); p.add_argument("--ledger", required=True); p.add_argument("--logical-work-id", required=True)
    p.add_argument("--state", required=True); p.add_argument("--job-id"); p.add_argument("--cluster"); p.add_argument("--attempt-id")
    p.add_argument("--sbatch-name"); p.add_argument("--array", default=None); p.add_argument("--run-ids"); p.add_argument("--pending-file"); p.add_argument("--runs-json", default=None)
    p.add_argument("--dry-run", action="store_true")
    p = sub.add_parser("array-tasks"); p.add_argument("--sacct", required=True, help="sacct's JobIDRaw|JobID|State rows, or @<file> (06 --gate hands a file)")
    p = sub.add_parser("pending-runs", help="06 --gate's reader of the PENDING/MISSING framed capture of a job's pending file(s); stdout = the runs list JSON, nothing else")
    p.add_argument("--capture", required=True, help="the framed capture file (`PENDING <f>` + the file's text, or `MISSING <f>`, per file)")
    p.add_argument("--missing-out", required=True, help="REQUIRED: the JSON list of MISSING paths is written here ([] when none) — never a second stdout line")
    p.add_argument("--tasks-json", default=None, help="array-tasks' output ([{raw, task, state}], or @<file>): raw job id -> array task index; absent = a plain job")
    p = sub.add_parser("probe-plan"); p.add_argument("--state-root", required=True); p.add_argument("--logical-work-id", required=True)
    p.add_argument("--runs-json", required=True, help="the bound runs with their registry rows (store_uri, extra, successor), or @<file>"); p.add_argument("--shared", required=True); p.add_argument("--probe", required=True)
    p.add_argument("--probe-map", default="", help="label=path;… — each cluster's coverage_probe.py (a superseded run is probed on its successor's cluster)")
    p = sub.add_parser("probe-append", help="06 --gate's per-call recorder: ONE JSONL line {run_id, array_task, probe, probe_error, probe_rc, probe_stderr_head, store_source} appended to --results")
    p.add_argument("--results", required=True); p.add_argument("--run-id", required=True); p.add_argument("--array-task", default="")
    p.add_argument("--out-file", required=True, help="the probe call's captured stdout"); p.add_argument("--err-file", required=True, help="the probe call's captured stderr")
    p.add_argument("--rc", type=int, required=True, help="the probe call's exit status (ssh's own 255 for a lost connection)")
    p.add_argument("--why", default=None, help="a `none` host line's reason: the probe never ran, this names why"); p.add_argument("--store-source", default="")
    p = sub.add_parser("coverage"); p.add_argument("--state-root", required=True); p.add_argument("--logical-work-id", required=True)
    p.add_argument("--runs-json", required=True, help="the bound runs (@<file> or a literal)"); p.add_argument("--dry-run", action="store_true")
    p.add_argument("--probes-json", default=None, help="06 --gate's per-call results [{run_id, array_task, probe, probe_error}] (@<file> or a literal): a run's documents are "
                                                        "folded into its record (fold_probe_results); without it each --runs-json row's own probe/probe_error is recorded")
    p.add_argument("--probes-jsonl", default=None, help="the same per-call results as probe-append writes them: one JSON object per line (06 --gate's results file)")
    p.add_argument("--probe-sha", default=None, help="D3003 (3): the commit of the lib/coverage_probe.py that read the stores (recorded on the coverage record)")
    p.add_argument("--probe-sha256", default=None, help="the sha256 of the probe file's bytes as shipped and verified on the cluster")
    p.add_argument("--inventory-sha", default=None, help="the inventory's frozen code_sha (recorded beside the probe's)")
    p.add_argument("--contract", default=None, help="the coverage-contract block `gate-contract --write` produced (@<file> or a literal), recorded on the record")
    p = sub.add_parser("gate-contract", help="D3003 (2): compare the CURRENT probe's coverage contract (unit set, identities, done predicates) with the inventory's record, structurally; refuse by name")
    p.add_argument("--state-root", required=True); p.add_argument("--probe", default=os.path.join(HERE, "coverage_probe.py"), help="the lib/coverage_probe.py whose DONE_PREDICATES are the current ones (default: this tree's)")
    p.add_argument("--admit-once", action="store_true", help="an inventory frozen before the field existed is admitted at the current contract ONCE (the sidecar gate/coverage_contract.json is written); without the flag such an inventory refuses")
    p.add_argument("--probe-sha", default=None); p.add_argument("--probe-sha256", default=None)
    p.add_argument("--write", default=None, help="write the contract block (version, sha256, record_sha256, source, admitted_utc, amendments) here for `coverage --contract` / `verdict --contract`")
    p.add_argument("--dry-run", action="store_true", help="compare and say; never write the admit-once sidecar")
    p = sub.add_parser("verdict"); p.add_argument("--state-root", required=True); p.add_argument("--family", required=True)
    p.add_argument("--headline", default=None); p.add_argument("--range", default=None); p.add_argument("--run-ids", default="")
    p.add_argument("--headline-record", default=None, help="@<file>: lib/gate_headline.py read's record (06 --gate --verdict) — the headline, the range and every leg, "
                                                             "in place of --headline / --range")
    p.add_argument("--verdict", choices=["pass", "fail"], default=None); p.add_argument("--note", default=""); p.add_argument("--dry-run", action="store_true")
    p.add_argument("--inventory-sha", default=None, help="D3003 (3): the inventory's frozen code_sha — REQUIRED for a family of the frozen 06 inventory (06 --gate passes it); stage 07's bcb_bfcl/<view> verdicts have no 06 inventory and need none")
    p.add_argument("--probe-sha", default=None, help="D3003 (3): the commit of the lib/coverage_probe.py the gate read with — REQUIRED for a 06 family")
    p.add_argument("--probe-sha256", default=None, help="the shipped probe file's sha256")
    p.add_argument("--contract", default=None, help="the coverage-contract block from gate-contract --write (@<file> or a literal) — REQUIRED for a 06 family: the verdict records the contract it was ruled under (admitted once, or frozen)")
    p = sub.add_parser("status"); p.add_argument("--state-root", required=True)
    a = ap.parse_args(argv)
    return {"plan": cmd_plan, "placement-manifest": cmd_placement, "freeze-inventory": cmd_freeze, "reconcile-registrations": cmd_reconcile_registrations, "hold-out": cmd_hold_out, "rerun-units": cmd_rerun_units, "bind": cmd_bind, "array-tasks": cmd_array_tasks, "pending-runs": cmd_pending_runs, "probe-plan": cmd_probe_plan, "probe-append": cmd_probe_append, "sha-table": cmd_sha_table,
            "amend-contract": cmd_amend_contract, "gate-contract": cmd_gate_contract,
            "coverage": cmd_coverage, "verdict": cmd_verdict, "status": cmd_status}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
