#!/usr/bin/env python3
"""registry_close_stores.py — close a completed full22 job's REGISTERED runs from the stores it wrote, against the registration's PLAN
(completeness is judged against the EXPECTED coverage, never the directories present).

A full22 job registers one run per (declared eval x declared object) at job start (common.sh f22_registry_register -> emit_cli register
--from-config: the pending file $SHARED/registry/pending/<jobid>.json, each run's plan carrying `n_items_planned` and `item_set_id`) and
emits `running`; the transition to `generated` — with the run's `store_uri` and completion evidence, which the grading daemon needs to
locate and dispatch the run — is filed by the HARNESS STORE's default recorder (sisteer.registry.hooks on_results), which only the capeval
CellStore carries (Petri, the capability evals). The stores `scripts/jailbreak_run.py generate` and `served_capability_run` write carry no
recorder and the job-end `f22_registry_result` only pushes, so every such run stayed `running` with an empty store_uri while its store sat
complete on the cluster (static t0/t1, bon.store512, controls t0/t1 of the first front-door wave, 2026-09-30). This is the job-side closer for
those families, called by `f22_registry_result` on a CLEAN exit before the final push, and by hand for a job that ended before it existed.

Per registered run the layout follows the run's INSTRUMENT (plan.instrument_id / extra.seed_instrument_id):

  static_*        <wave root>/transcripts_results/sdfbase_full22_<tag>_sampled_t<trial>/<eval>/<object>/shard00..shard<N-1>/
                  {progress.jsonl, responses.jsonl, served_target.json} — EVERY one of the job's N shards (--nshards; the sbatch's NSHARDS)
                  present with all three files, every progress row `outcome: done`, the distinct goals == the item set's goals
                  (`…_<goals>x<variants>`), and responses == n_items_planned exactly. n_items_generated = n_items_planned.
  bon_*           <wave root>/transcripts_results/sdfbase_full22_<tag>/<object>/{bon_responses.jsonl, bon_errors.jsonl, bon_served_run.json}
                  — every planned slot DECIDED: a response row, or a no-fit refusal row in bon_errors (a slot the served window could not fit
                  is decided, not missing — the 7500/7500 `prompt_over_window` cell); decided slots >= bon_served_run.n_target_forwards and
                  decided goals == n_goals_selected == n_items_planned. n_items_generated = goals with at least one response.
  overrefusal_*   <wave root>/sampled/controls_cap<C>/<tag>_t<trial>/cells/<object cell>/ at the FINAL rung: the highest cap whose
                  cap_ladder.json shows the run's control (xstest -> `xstest`, or_bench_hard -> `or_bench_hard_1k`) with `escalate: false`,
                  its generation spool private/gens_<control>*.jsonl holding at least the control's `n` rows and at least n_items_planned
                  (xstest's registered item set is the SAFE half: 250 of the 450 generated). n_items_generated = n_items_planned.

  multiturn_*     <wave root>/transcripts_results/<private_run>/<attack>/<cell>/shard00..shard<N-1>/ (jailbreak_run._gated_defense_root; the
                  private_run is the REGISTERED config's `private_run`, else sdfbase_full22_<tag>) — EVERY one of the N shards present with
                  {progress.jsonl, responses.jsonl, served_target.json}, no progress outcome `error:*` / `refused:*` (the grader's
                  exclude_errored_goals rule: an errored goal makes the aggregate refuse), every goal in exactly ONE shard, at least one
                  response row per goal, and the goal ids EQUAL AS A SET the registered config's eval goal set (jailbreak_run._eval_goals
                  over extra.config, confirmatory — the grader's own expected ids: a count of 259 admitted a
                  substituted id, which the grader then scored as a silent miss), whose size is n_items_planned. The ids are ALWAYS
                  derived in-process by the active checkout's resolver (a file can be stale or edited); an
                  interpreter that cannot import it (the cluster's bare python3 -S) is an UNTRUSTED refusal by name — run the closer where
                  it imports (the job's driver venv; by hand, that venv on a compute node through srun). --expected-goal-ids, a file
                  --emit-expected-goal-ids wrote (config path + sha256, the resolver's implementation sha256, the dataset revisions, the
                  ids and their sha256), is a CROSS-CHECK only: it must agree with the recomputed set and the active resolver, else
                  refused. n_items_generated = n_items_planned. A cell
                  whose shards were written by several jobs (four shard-group jobs per attack, half on each cluster)
                  is one store: the job that registered the CANONICAL run closes it once the shards are all under one root; a
                  `consolidation.json` beside the shards (the one-time copy's provenance) rides the facts when present.

  strongreject_bon, TWICE: the 512 store is REGISTERED by two job kinds — the store512 job
                  (`f22-stat-<tag>-3`) and the 4,096 climb array (`f22-bonclimb-<tag>`; stage.py plans both over the same (eval, object) and
                  the bon layout above points both at the 512 store) — so one store carried five current registrations (the smoke's ten BoN
                  stores, 2026-10-11). The store512 registration SURVIVES; a climb registration covering the SAME store (same store_uri,
                  identical store_completion_sha256 and n_items_generated — FILED values) is a duplicate of it and is filed `superseded` by it
                  (reason duplicate_registration, extra.superseded_by_run_id: the producer's run_state record, whose fold is the census
                  effect) WHICHEVER CLOSES LAST — a climb closing after the store512's filing supersedes its own run and
                  files no `generated`; a store512 closing after a climb's `generated` files its own and retires the twin. A climb whose bytes
                  or item count differ is NOT a duplicate: refused / held by name, no supersede, a human rules. A rerun pair's jobs carry the
                  rerun-units slug (`f22-stat-<tag>-<slug>-3` / `f22-bonclimb-<tag>-<slug>`) and are twins when BOTH spell it; and a
                  registration at ANOTHER RUNG of the store's ladder (token_cap_tokens or pass_label differ on the registration records —
                  the 512 store beside its 4,096 climb) is NEVER a twin, whatever the store facts: noted, never held, never superseded.
                  RACE-FREE: the two closers are separate jobs and can close in the same second, each seeing no peer, so the decision
                  and its filing happen under an exclusive flock on the store's close lock — BESIDE the store, `<private run>/.<cell>.close.lock`
                  (bon_close_lock_path), a path the lock taker creates, so a run's lifecycle writers take the SAME lock before the store
                  exists — AND the rule is re-evaluated on every close — an
                  already-generated run of either kind still reads its twins, so a reclose or a later sweep files the supersede a race, or
                  an older tree, left missing. Siblings are the pending files beside this job's (this cluster; a twin on another cluster
                  stays until filed by hand).

Anything short — a missing shard, a row not done, a count off, an undecided slot, no final rung, an eval with no layout here — is REFUSED by
name (exit 2 at the end) and the run stays `running` for a human; nothing partial is ever filed as generated. A run whose record directory
already holds `run_state generated` is left alone (idempotent; the capeval / mmlu emitters' runs read as such). Filing goes through the real
emitter (sisteer.registry.emit.RecordWriter into the job's spool): run_patch {store_uri, store_completion_artifact_uri,
store_completion_sha256, n_items_generated, generated_utc} + run_state generated (evidence completion_artifact) + push --final. Stdlib +
sisteer.registry only, so it runs under the cluster's bare python3 -S from the eval-repo checkout. Never names a host or a user.

    registry_close_stores.py --pending <pending json> --wave-root <dir> --tag <model tag> [--job-name <slurm name> | --trial N]
                             [--nshards N] [--spool <dir>] [--push-script <sh>] [--code-sha <sha>] [--evals a,b] [--dry-run]
    registry_close_stores.py --pending <pending json> --private-root <dir> --private-run <name> [--nshards N] ...   (static only, explicit)
    registry_close_stores.py --pending <pending json> --emit-expected-goal-ids <out.json>      (multiturn: the eval goal ids + their provenance)
    registry_close_stores.py --pending <pending json> --wave-root <dir> --tag <tag> [--expected-goal-ids <out.json>] ...   (cross-check only)
"""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import glob
import hashlib
import json
import os
import re
import sys
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "src"))
from sisteer.eval import smoke_goals as SG  # noqa: E402  (demo-import M10: the SMOKE goal-shard knob the job's overrides document stamps on its runs)
from sisteer.registry import emit, identity  # noqa: E402
from sisteer.registry import decode_evidence as DE  # noqa: E402  (decode-recording: the request's observed sampler, filed beside the store facts)
from sisteer.registry import emit_cli as EC  # noqa: E402  (an array task's `running`, filed per run through the state emitter's own hardware capture)
from sisteer.registry import records as R  # noqa: E402
from sisteer.registry.records import utc_now  # noqa: E402

sys.path.insert(0, os.path.join(HERE, "sampled"))
import at_cap_controls as acc  # noqa: E402  (the sampled pipeline's controls store reader: the spool's texts and the store digest a gate record binds)

SCRIPT = "sdfbase.full22.registry_close_stores"
NSHARDS_DEFAULT = 9                                   # target_sampled.sbatch's NSHARDS default
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
FAMILY_OF_INSTRUMENT = (("static_", "static"), ("bon_", "bon"), ("overrefusal_", "controls"), ("multiturn_", "multiturn"))
FAMILY_OF_EVAL = {"strongreject_bon": "bon", "xstest": "controls", "or_bench_hard": "controls",
                  "pasttense": "static", "codeattack": "static", "renellm": "static", "grok420": "static",
                  "crescendo": "multiturn", "pair": "multiturn", "actorattack": "multiturn"}
CONTROL_KEY = {"xstest": "xstest", "or_bench_hard": "or_bench_hard_1k"}
#: static-reg-fix: a per-rung static registration's producer key IS the rung's private run (`sdfbase_full22_<tag>_sampled_t<T>[_cap<N>]`)
STATIC_RUN_KEY_RE = re.compile(r"^sdfbase_full22_(?P<tag>.+?)_sampled_t(?P<trial>\d)(?:_cap(?P<cap>\d+))?$")
#: evals ANOTHER emitter files are NOT this closer's — the Petri drive's registry adapter (petri_*: sisteer.capeval.petri registry_adapter, the
#: drive's own hand-off) and the capability evals filed by their own harness (mmlu_full.sbatch's registry_end; gsm8k / bigcodebench / bfcl through
#: the capeval CellStore's default recorder, sisteer.registry.hooks on_results). Their runs are skipped with one informational line and never
#: counted as refused — a refusal here is the job's exit (75) and a reclose, which cannot help a run this closer does not file.
#: An eval in neither this closer's layouts nor this declared set is still REFUSED by name: nobody closes it, and that must be loud.
OTHER_EMITTER_EVAL_PREFIXES = ("petri",)
OTHER_EMITTER_EVALS = ("mmlu", "gsm8k", "bigcodebench", "bfcl")


def other_emitter_of(eval_id: str) -> Optional[str]:
    """Who files `eval_id`'s runs when it is not this closer: a name for the skip line, None when the eval is this closer's or nobody's."""
    e = str(eval_id or "")
    if e.startswith(OTHER_EMITTER_EVAL_PREFIXES):
        return "the Petri drive's registry adapter (sisteer.capeval.petri)"
    if e in OTHER_EMITTER_EVALS or any(e.startswith(x + "_") for x in OTHER_EMITTER_EVALS):
        return "the capability eval's own harness (mmlu_full.sbatch registry_end / the capeval CellStore's default recorder)"
    return None
ITEM_SET_RE = re.compile(r"_(\d+)x(\d+)$")
TRIAL_RE = re.compile(r"_t(\d+)(?:\b|_|$)")
STATIC_FILES = ("progress.jsonl", "responses.jsonl", "served_target.json")
MULTITURN_FILES = STATIC_FILES                        # the multi-turn shards carry the same three files (plus attacker_engine.json and the call logs)
MULTITURN_CONSOLIDATION = "consolidation.json"        # the one-time cross-cluster copy's provenance, beside the shards
MULTITURN_NOT_GRADED_PREFIXES = ("error", "refused")  # jailbreak_artifacts.exclude_errored_goals: `error:*` refuses the aggregate, `refused:*` is excluded


def _say(msg: str) -> None:
    print(msg, flush=True)


def _rows(path: str) -> List[dict]:
    out: List[dict] = []
    with open(path, encoding="utf-8") as fh:
        for ln in fh:
            ln = ln.strip()
            if not ln:
                continue
            try:
                out.append(json.loads(ln))
            except ValueError:
                out.append({"_unparsable": True})
    return out


def _count_lines(path: str) -> int:
    n = 0
    with open(path, "rb") as fh:
        for ln in fh:
            if ln.strip():
                n += 1
    return n


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _json(path: str) -> Optional[dict]:
    try:
        with open(path, encoding="utf-8") as fh:
            d = json.load(fh)
        return d if isinstance(d, dict) else None
    except (OSError, ValueError):
        return None


def family_of(run: dict) -> Optional[str]:
    """The run's store family from its instrument's SEED spelling (`extra.seed_instrument_id`, e.g. static_sampled_vendor_preset_4096_8192;
    `plan.instrument_id` is the seed's hash), else from the eval id."""
    plan = run.get("plan") or {}
    if pad_level_of(run) is not None:
        return "padding"                                    # the padded static ladder: a STATIC instrument seed under its own lane root — before the prefix table
    for inst in (str((plan.get("extra") or {}).get("seed_instrument_id") or ""), str(plan.get("instrument_id") or "")):
        for prefix, fam in FAMILY_OF_INSTRUMENT:
            if inst.startswith(prefix):
                return fam
    eval_id = str((run.get("route") or {}).get("eval_id") or "")
    if eval_id.startswith("agentic_"):
        return "misalign"                                   # the misalign runner's agentic family (one cell file, one run per behaviour)
    return FAMILY_OF_EVAL.get(eval_id)


PAD_JOB_RE = re.compile(r"^pad(?:stat|obj)-.*-p(\d+)$")     # stage.py: padstat-<key>-p<pad> (base ladder) / padobj-<tag>-p<pad> (steered objects)


def _padding_block(run: dict) -> Optional[dict]:
    extra = ((run.get("plan") or {}).get("extra") or {})
    pad = extra.get("padding")
    if isinstance(pad, str) and pad.strip().startswith("{"):
        try:
            import ast
            pad = ast.literal_eval(pad)
        except (ValueError, SyntaxError):
            pad = None
    return pad if isinstance(pad, dict) else None


def pad_level_of(run: dict) -> Optional[int]:
    """The padding level of a padded static run: the registration's `extra.padding.tokens` (jobplan.padding_stamp: the level + the contract
    keys; older stamps spelt it `padding_tokens`), else the job name `padstat-<key>-p<pad>` / `padobj-<tag>-p<pad>`. None for any other run."""
    plan = run.get("plan") or {}; extra = plan.get("extra") or {}
    pad = _padding_block(run)
    if pad is not None:
        for k in ("tokens", "padding_tokens"):
            if isinstance(pad.get(k), int):
                return int(pad[k])
    if pad is not None or extra.get("padding_identity_sha256") or PAD_JOB_RE.match(str(plan.get("job_name") or "")):
        m = PAD_JOB_RE.match(str(plan.get("job_name") or "")); return int(m.group(1)) if m else None
    return None


PENDING_DIR: Optional[str] = None   # set by close_job: the directory of the pending file — the writer's home for `<job>.config.yaml` (cells-fix)


def _bind_pending_dir(pending_path: str) -> str:
    """PENDING_DIR is bound from the pending file on EVERY entry — close_job and --emit-expected-goal-ids
    alike. The ids emitter used to resolve the registered config against the checkout alone, so a narrowed registration (the
    `<job>.config.yaml` beside the pending file) refused `--emit-expected-goal-ids` while the closer itself would have found the file."""
    global PENDING_DIR
    PENDING_DIR = os.path.dirname(os.path.abspath(pending_path))
    return PENDING_DIR


def registered_config_path(run: dict, *, repo_root: str) -> Tuple[Optional[str], str, str]:
    """(refusal or None, the registered config's path on disk, its stamped spelling). An absolute stamp is itself; a relative one is tried
    against the checkout (`repo_root`, the wave configs) and then BESIDE THE PENDING FILE (`PENDING_DIR/<basename>`: cells-fix's narrowed copy
    `$SHARED/registry/pending/<job>.config.yaml`, which jobplan once stamped as `pending/<job>.config.yaml` — 837557, 2026-10-02). Refuses by
    name, naming every path tried, when none exists."""
    cfg_rel = str(((run.get("plan") or {}).get("extra") or {}).get("config") or "")
    if not cfg_rel:
        return "the registration names no config (extra.config)", "", ""
    tried = []
    if os.path.isabs(cfg_rel):
        tried.append(cfg_rel)
    else:
        tried.append(os.path.join(repo_root, cfg_rel))
        if PENDING_DIR:
            tried.append(os.path.join(PENDING_DIR, os.path.basename(cfg_rel)))
    for p in tried:
        if os.path.isfile(p):
            return None, p, cfg_rel
    return f"the registered config {cfg_rel} is not on disk (tried {', '.join(tried)})", tried[0], cfg_rel


def pad_private_run(run: dict, *, repo_root: str) -> Tuple[Optional[str], Optional[str]]:
    """The padded store's private_run — the REGISTERED config's `private_run` (make_padding_configs writes `pad_static_<key>_p<pad>` for the
    base ladder and `pad_static_<key>_obj_p<pad>` for the steered objects; the registry model id is not the key: qwen35_4b_it vs qwen35_4b).
    Refuses by name when the registration names no config or the config cannot be read."""
    why_p, path, cfg_rel = registered_config_path(run, repo_root=repo_root)
    if why_p:
        return f"{why_p}: the padded store's private_run cannot be resolved", None
    try:
        import yaml
        cfg = yaml.safe_load(open(path, encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return f"the registered config {cfg_rel} could not be read ({type(exc).__name__}: {exc})", None
    prun = str((cfg or {}).get("private_run") or "")
    if not prun.startswith("pad_static_"):
        return f"the registered config {cfg_rel} names no padded private_run (got {prun!r})", None
    return None, prun


def trial_of(job_name: Optional[str], explicit: Optional[int]) -> Optional[int]:
    if explicit is not None:
        return explicit
    m = TRIAL_RE.search(job_name or "")
    return int(m.group(1)) if m else None


def planned_of(run: dict) -> Tuple[Optional[int], str, Optional[int], Optional[int]]:
    """(n_items_planned, item_set_id, goals, variants) from the registration's plan; goals x variants parsed from `…_<g>x<v>`."""
    plan = run.get("plan") or {}
    n = plan.get("n_items_planned"); iset = str(plan.get("item_set_id") or "")
    m = ITEM_SET_RE.search(iset)
    return (int(n) if isinstance(n, int) else None), iset, (int(m.group(1)) if m else None), (int(m.group(2)) if m else None)


# ------------------------------------------------------------------------------------------------------------ the three layouts
def inspect_static(store: str, *, planned: int, goals: Optional[int], nshards: int,
                   shard_names: Optional[Sequence[str]] = None) -> Tuple[Optional[str], Dict[str, object], str]:
    """(refusal or None, facts, the completion artifact). Complete = every expected shard present with its three files, every progress row
    `done`, distinct goals == the item set's goals (when known), responses == n_items_planned exactly. `shard_names` (demo-import M10): the
    EXACT shards the store holds under the smoke goal-shard knob (shard<lo>..shard<hi> of N; the caller passes the subset's goal count and
    planned responses) in place of shard00..shard<N-1>; a shard outside them is a store laid out under another plan, refused."""
    facts: Dict[str, object] = {"layout": "static", "nshards_expected": nshards}
    if not os.path.isdir(store):
        return f"no store directory at {store}", facts, ""
    expected = list(shard_names) if shard_names else [f"shard{i:02d}" for i in range(nshards)]
    if shard_names:
        facts["shards_expected"] = list(expected)
    missing = [s for s in expected if not all(os.path.isfile(os.path.join(store, s, f)) for f in STATIC_FILES)]
    present = sorted(os.path.basename(d) for d in glob.glob(os.path.join(store, "shard*")) if os.path.isdir(d))
    unexpected = sorted(set(present) - set(expected))
    facts.update({"shards_present": present, "shards_missing": missing, "shards_unexpected": unexpected})
    if missing:
        return f"{store}: {len(missing)} of {len(expected)} expected shard(s) missing or incomplete: {', '.join(missing)}", facts, ""
    if unexpected:
        return f"{store}: shard directories outside the job's plan of {nshards}: {', '.join(unexpected)}", facts, ""
    done = 0; others: Dict[str, int] = {}; not_done_goals: List[str] = []; goal_ids: set = set(); responses = 0
    for s in expected:
        for r in _rows(os.path.join(store, s, "progress.jsonl")):
            oc = str(r.get("outcome") or ("unparsable" if r.get("_unparsable") else "missing"))
            goal_ids.add(str(r.get("goal_id")))
            if oc == "done":
                done += 1
            else:
                others[oc] = others.get(oc, 0) + 1; not_done_goals.append(f"{s}:{r.get('goal_id')}={oc}")
        responses += _count_lines(os.path.join(store, s, "responses.jsonl"))
    facts.update({"goals_done": done, "goals_not_done": sum(others.values()), "outcomes_not_done": others, "goals_distinct": len(goal_ids),
                  "goals_planned": goals, "responses": responses, "responses_planned": planned})
    if others:
        return f"{store}: {sum(others.values())} progress row(s) not done ({others}; e.g. {', '.join(not_done_goals[:4])})", facts, ""
    if goals is not None and len(goal_ids) != goals:
        return f"{store}: {len(goal_ids)} distinct goal(s) across the shards, the item set plans {goals}", facts, ""
    if responses != planned:
        return f"{store}: {responses} response row(s), the registration plans {planned}", facts, ""
    return None, facts, os.path.join(store, expected[0], "progress.jsonl")


def bon_plan_universe(store: str, served: dict, *, repo_root: str) -> Tuple[Optional[str], List[Tuple[str, int, int]], Dict[str, object]]:
    """The EXACT planned slot universe of a BoN cell — the store job's own resolver (two_si_run._eval_goals over the wave config the marker
    names, `bon.n_max`, trial 0), the same symbol the climb planner uses (bon_climb_plan.py). Refuses by name when
    the resolver cannot be imported here (its data loaders need the job's interpreter: pandas, the goal datasets) — never a silent pass."""
    facts: Dict[str, object] = {}
    cfg_rel = str(served.get("config") or "")
    if not cfg_rel:
        return f"{store}: {os.path.basename(store)}'s marker names no `config`: the plan universe cannot be resolved", [], facts
    try:
        sys.path.insert(0, os.path.join(repo_root, "scripts"))
        import jailbreak_run as jr  # noqa: E402
        import two_si_run as tsr  # noqa: E402
        cfg = jr._load_cfg(cfg_rel if os.path.isabs(cfg_rel) else os.path.join(repo_root, cfg_rel))
        confirmatory = served.get("confirmatory"); confirmatory = True if confirmatory is None else bool(confirmatory)
        goals = tsr._eval_goals(cfg, confirmatory=confirmatory, attack="bon")
        n_max = int(cfg["bon"]["n_max"])
    except Exception as exc:  # noqa: BLE001 — an import or a loader failure is a refusal with the reason, not a pass
        return f"{store}: the plan universe could not be resolved ({type(exc).__name__}: {exc}) — run the closer with the job's interpreter", [], facts
    facts.update({"plan_config": cfg_rel, "plan_goals": len(goals), "plan_n_max": n_max, "plan_confirmatory": confirmatory})
    return None, [(g.goal_id, i, 0) for g in goals for i in range(n_max)], facts


def inspect_bon(store: str, *, planned_goals: int, repo_root: str) -> Tuple[Optional[str], Dict[str, object], str]:
    """A BoN cell is complete when (a) its DURABLE manifest says every planned slot is DECIDED — `bon_climb.store_completion` over
    `bon_served_manifest.json`'s `units` with the per-slot classification (only typed `nofit_proven` rows are decided no-fits) — AND (b)
    the LIVE ledgers agree with it: `bon_climb.store_planned_gaps` over the exact planned goal x sample x trial universe (bon_plan_universe)
    finds no gap, and its covered count equals the manifest's n_done (a response ledger truncated or resumed after
    a complete manifest + marker were written used to read complete). `bon_served_run.json`'s n_target_forwards is the LAST invocation's
    call counter and is never read for completeness. The served selection must be the registration's plan and the resolver's."""
    from sisteer.eval.attacks import bon_climb as BC
    facts: Dict[str, object] = {"store": store}
    marker = os.path.join(store, BC.BON_STORE_MANIFEST)
    if not os.path.isdir(store):
        return f"{store}: no store directory", facts, ""
    if not os.path.isfile(marker):
        return f"{store}: no end-of-run marker {BC.BON_STORE_MANIFEST} (the unit has not finished writing)", facts, ""
    served = _json(marker) or {}
    n_goals = served.get("n_goals_selected")
    facts.update({"n_goals_selected": n_goals, "goals_planned": planned_goals, "n_target_forwards_last_invocation": served.get("n_target_forwards")})
    if not isinstance(n_goals, int) or n_goals <= 0:
        return f"{store}: {BC.BON_STORE_MANIFEST} names no n_goals_selected", facts, ""
    if n_goals != planned_goals:
        return f"{store}: the served run selected {n_goals} goal(s), the registration plans {planned_goals}", facts, ""
    units = BC.store_units(store)
    classify = getattr(BC, "store_slot_classification", None)
    classification = None
    if classify is not None:
        try:
            classification = classify(store)
        except Exception as exc:  # noqa: BLE001 — a classifier that cannot read the ledger is a refusal, never a silent pass
            return f"{store}: slot classification failed ({type(exc).__name__}: {exc})", facts, ""
    decided, why, counts = BC.store_completion(units, classification) if classification is not None else BC.store_completion(units)
    facts.update({"units": dict(units or {}), "completion": counts, "completion_why": why, "classified": classification is not None})
    if not decided:
        return f"{store}: {why}", facts, ""
    # (b) the live ledgers against the exact plan universe
    why_u, planned_keys, ufacts = bon_plan_universe(store, served, repo_root=repo_root)
    facts.update(ufacts)
    if why_u:
        return why_u, facts, ""
    if ufacts.get("plan_goals") != n_goals or (isinstance(served.get("n_max"), int) and served.get("n_max") != ufacts.get("plan_n_max")):
        return f"{store}: the resolver's plan ({ufacts.get('plan_goals')} goals x {ufacts.get('plan_n_max')}) is not the served run's ({n_goals} x {served.get('n_max')})", facts, ""
    gaps = BC.store_planned_gaps(store, planned_keys)
    facts["ledger"] = {k: gaps.get(k) for k in ("n_planned", "n_covered", "n_nofit_proven", "n_unparsable_rows")}
    facts["ledger"]["n_gaps"] = len(gaps.get("gaps") or []); facts["ledger"]["n_foreign_keys"] = len(gaps.get("foreign_keys") or [])
    if gaps.get("gaps"):
        g0 = gaps["gaps"][0]
        return f"{store}: {len(gaps['gaps'])} planned slot(s) have neither a response row nor a proven no-fit in the live ledgers (first {g0.get('goal_id')}/{g0.get('sample_idx')}: {g0.get('kind')}) — the manifest's units are stale", facts, ""
    n_done = int((counts or {}).get("n_done") or 0)
    if gaps["n_covered"] != n_done or gaps["n_covered"] + gaps["n_nofit_proven"] != gaps["n_planned"]:
        return f"{store}: the live ledgers cover {gaps['n_covered']} + {gaps['n_nofit_proven']} no-fit of {gaps['n_planned']} planned slot(s), the manifest says n_done={n_done} — the manifest is stale", facts, ""
    facts["_planned_keys"] = planned_keys                    # for the certificate's per-slot audit; popped before the note
    return None, facts, os.path.join(store, BC.BON_STORE_MANIFEST_UNITS)


def multiturn_private_run(run: dict, tag: Optional[str], *, repo_root: str) -> Tuple[Optional[str], Optional[str], str]:
    """(refusal or None, private_run, source). The store's private run is the REGISTERED config's `private_run` (extra.config, read with
    PyYAML when present, else the top-level `private_run:` line — the closer runs under the cluster's bare python3 -S); when the
    registration names no config, sdfbase_full22_<tag> (target_attacks.sbatch's PRIVATE_RUN is the config's, and the generator names the
    config after it). Refuses by name when neither resolves."""
    cfg_rel = str(((run.get("plan") or {}).get("extra") or {}).get("config") or "")
    if cfg_rel:
        why_p, path, _rel = registered_config_path(run, repo_root=repo_root)
        if why_p:
            return f"{why_p}", None, cfg_rel
        try:
            text = open(path, encoding="utf-8").read()
        except OSError as exc:
            return f"the registered config {cfg_rel} could not be read ({type(exc).__name__}: {exc})", None, cfg_rel
        prun = None
        try:
            import yaml
            prun = (yaml.safe_load(text) or {}).get("private_run")
        except ImportError:
            m = re.search(r"^private_run:\s*['\"]?([A-Za-z0-9_.-]+)['\"]?\s*(?:#.*)?$", text, re.M)
            prun = m.group(1) if m else None
        except ValueError as exc:
            return f"the registered config {cfg_rel} is not valid YAML ({exc})", None, cfg_rel
        if not prun:
            return f"the registered config {cfg_rel} names no private_run", None, cfg_rel
        return None, str(prun), f"config {cfg_rel}"
    if tag:
        return None, f"sdfbase_full22_{tag}", "sdfbase_full22_<tag>"
    return "the registration names no config (extra.config) and no --tag was given: the private run cannot be resolved", None, ""


MULTITURN_GOAL_RESOLVER = "jailbreak_run._eval_goals(confirmatory=True)"   # the grader's own expected ids (grade_full22_native.py)
#: the resolver's IMPLEMENTATION: the two modules whose content decides the eval goal set (jailbreak_run._eval_goals over
#: sisteer.eval.jailbreak_data's loaders and stratification); their sha256 binds an ids artifact to the code that produced it
MULTITURN_RESOLVER_FILES = ("scripts/jailbreak_run.py", "src/sisteer/eval/jailbreak_data.py")


def resolver_identity(repo_root: str) -> Dict[str, str]:
    """{resolver, resolver_sha256}: the resolver label and the sha256 over the resolver modules' bytes in this checkout (MULTITURN_RESOLVER_FILES,
    in order; a missing module hashes as its name + 'ABSENT')."""
    h = hashlib.sha256()
    for rel in MULTITURN_RESOLVER_FILES:
        path = os.path.join(repo_root, rel)
        h.update(rel.encode("utf-8"))
        try:
            with open(path, "rb") as fh:
                h.update(fh.read())
        except OSError:
            h.update(b"ABSENT")
    return {"resolver": MULTITURN_GOAL_RESOLVER, "resolver_sha256": h.hexdigest()}


def _ids_sha256(ids: Sequence[str]) -> str:
    return hashlib.sha256(json.dumps([str(g) for g in ids], separators=(",", ":")).encode("utf-8")).hexdigest()


def _registered_config_path(run: dict, *, repo_root: str) -> Tuple[Optional[str], str, str]:
    """(refusal or None, the registered config's path on disk, its stamped spelling) — :func:`registered_config_path`, the goal-set wording."""
    why_p, path, cfg_rel = registered_config_path(run, repo_root=repo_root)
    if why_p and not cfg_rel:
        return f"{why_p}: the eval goal set cannot be resolved", "", ""
    return why_p, path, cfg_rel


def multiturn_expected_goal_ids(run: dict, *, repo_root: str, ids_file: Optional[str] = None) -> Tuple[Optional[str], Optional[List[str]], Dict[str, object]]:
    """(refusal or None, the registered config's eval goal ids in resolver order, provenance). ALWAYS recomputed in-process by the ACTIVE
    checkout's resolver (scripts/jailbreak_run._eval_goals over the registered config, confirmatory — the ids the grader scores); an
    interpreter that cannot import it is an UNTRUSTED refusal by name, whatever file is offered (an ids file
    emitted by an older checkout, or edited, would otherwise file a substituted set the grader refuses). `ids_file`, when given, is a
    CROSS-CHECK: it must name the same config (path + sha256), the same resolver implementation (resolver_sha256 of the active checkout),
    carry ids whose sha256 matches, and equal the recomputed set — any disagreement refuses by name."""
    why, path, cfg_rel = _registered_config_path(run, repo_root=repo_root)
    if why:
        return why, None, {}
    cfg_sha = _sha256(path)
    ident = resolver_identity(repo_root)
    try:
        sys.path.insert(0, os.path.join(repo_root, "scripts"))
        import jailbreak_run as jr  # noqa: E402
        cfg = jr._load_cfg(path)
        goals = jr._eval_goals(cfg, confirmatory=True)
        ids = [str(g.goal_id) for g in goals]
    except Exception as exc:  # noqa: BLE001 — an import or loader failure is a refusal with the reason, not a pass
        return (f"UNTRUSTED: the eval goal set cannot be recomputed by this interpreter from {cfg_rel} ({type(exc).__name__}: {exc}); "
                f"{'an ids file (' + ids_file + ') cannot stand in for the resolver — ' if ids_file else ''}run the closer where the resolver imports "
                f"(the job's driver venv; by hand, that venv on a compute node through srun)"), None, {}
    if len(set(ids)) != len(ids):
        return f"the resolver returned duplicate goal ids for {cfg_rel}", None, {}
    prov: Dict[str, object] = {"source": "resolver in-process", "config": cfg_rel, "config_sha256": cfg_sha, **ident,
                               "datasets": {"harmbench_revision": str(((cfg.get("datasets") or {}).get("harmbench") or {}).get("revision") or ""),
                                            "jbb_revision": str(((cfg.get("datasets") or {}).get("jbb") or {}).get("revision") or "")}}
    if ids_file:
        doc = _json(ids_file)
        if not doc:
            return f"--expected-goal-ids {ids_file}: not a readable JSON object", None, {}
        fids = doc.get("goal_ids")
        if not isinstance(fids, list) or not fids:
            return f"--expected-goal-ids {ids_file}: goal_ids is empty or not a list", None, {}
        fids = [str(g) for g in fids]
        # the same config path, and its bytes equal or the release's declared rewrite of them (a certified scrub-map row)
        from sisteer import scrub_map as SM
        sha_check = SM.check(str(doc.get("config_sha256") or ""), cfg_sha, path=cfg_rel) if str(doc.get("config") or "") == cfg_rel else None
        if sha_check is None:
            return (f"--expected-goal-ids {ids_file} certifies config {doc.get('config')!r} @ {str(doc.get('config_sha256') or '')[:12]}, the registration's is "
                    f"{cfg_rel!r} @ {cfg_sha[:12]} — not this run's goal set"), None, {}
        if str(doc.get("resolver") or "") != ident["resolver"] or str(doc.get("resolver_sha256") or "") != ident["resolver_sha256"]:
            return (f"--expected-goal-ids {ids_file} was written by resolver {doc.get('resolver')!r} @ {str(doc.get('resolver_sha256') or '')[:12]}, the active checkout's is "
                    f"{ident['resolver']!r} @ {ident['resolver_sha256'][:12]} — a stale or foreign emitter; refusing the cross-check"), None, {}
        if str(doc.get("goal_ids_sha256") or "") != _ids_sha256(fids):
            return f"--expected-goal-ids {ids_file}: goal_ids do not hash to the file's goal_ids_sha256 — an edited list; refusing", None, {}
        if set(fids) != set(ids):
            return (f"--expected-goal-ids {ids_file} disagrees with the recomputed eval set ({len(set(ids) - set(fids))} missing, "
                    f"{len(set(fids) - set(ids))} foreign: {sorted(set(fids) - set(ids))[:3]}) — refusing"), None, {}
        prov["cross_check"] = {"file": ids_file, "agrees": True}
        if sha_check == SM.VIA_MAP:
            prov["cross_check"]["config_sha_check"] = SM.VIA_MAP
    return None, ids, prov


def write_expected_goal_ids(run: dict, out_path: str, *, repo_root: str) -> Optional[str]:
    """--emit-expected-goal-ids: resolve the registered config's eval goal ids HERE and write the self-certified file the cluster-side
    closer reads (config, config_sha256, resolver, n, goal_ids). Returns a refusal or None."""
    why, ids, prov = multiturn_expected_goal_ids(run, repo_root=repo_root)
    if why:
        return why
    doc = {"config": prov["config"], "config_sha256": prov["config_sha256"], "resolver": prov["resolver"], "resolver_sha256": prov["resolver_sha256"],
           "resolver_files": list(MULTITURN_RESOLVER_FILES), "datasets": prov.get("datasets"), "n": len(ids or []), "goal_ids": list(ids or []),
           "goal_ids_sha256": _ids_sha256(ids or []), "written_utc": utc_now(), "writer": SCRIPT,
           "note": "a CROSS-CHECK artifact: the closer always recomputes the eval set with the active checkout's resolver and refuses when this file disagrees or "
                   "when the resolver cannot import"}
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=1, sort_keys=True)
    _say(f"wrote {out_path}: {len(ids or [])} eval goal ids for {prov['config']} @ {prov['config_sha256'][:12]} ({prov['resolver']} @ {prov['resolver_sha256'][:12]})")
    return None


def inspect_multiturn(store: str, *, planned: int, nshards: int, expected_ids: Optional[Sequence[str]] = None,
                      tolerated_missing: Optional[Sequence[str]] = None,
                      subset: Optional[SG.SmokeGoals] = None) -> Tuple[Optional[str], Dict[str, object], str]:
    """(refusal or None, facts, the completion artifact). Complete = every expected shard present with its three files, no progress row
    whose outcome the grader would not grade (`error:*` / `refused:*`), every goal in exactly one shard with at least one response row, the
    goal ids EQUAL AS A SET to `expected_ids` (the registered config's eval goal set; a count alone admitted a substituted id), and
    distinct goals == n_items_planned. The facts carry the per-shard goal counts, the outcome tally and, when the cell was consolidated
    across clusters, the copy's provenance summary. `subset`: the smoke goal-shard knob the job's overrides document stamped on this run —
    the store holds EXACTLY shards lo..hi of N (`subset.nshards` is the caller's `nshards`), each held to its own dealing goals[s::N], and the
    goal set expected is the union of those deals; a shard outside the range refuses (a store laid out under another plan). Unlike a partial
    close this IS the plan: the caller files `generated` over `goals_expected`."""
    facts: Dict[str, object] = {"layout": "multiturn", "nshards_expected": nshards}
    if not os.path.isdir(store):
        return f"no store directory at {store}", facts, ""
    if subset is not None and int(subset.nshards) != int(nshards):
        return f"{store}: the smoke knob's N={subset.nshards} is not the job's shard plan {nshards} — one store, one layout", facts, ""
    if subset is not None and tolerated_missing:
        return f"{store}: a D1168 ruling (shards {', '.join(tolerated_missing)} ruled out) and a smoke goal-shard subset ({SG.KEY}={subset.spec}) name one store — nothing filed", facts, ""
    if subset is not None and expected_ids is None:
        return f"{store}: the smoke goal-shard subset ({SG.KEY}={subset.spec}) needs the registered goal set (the config's eval goals) to hold each shard to its deal", facts, ""
    expected = list(subset.shards) if subset is not None else [f"shard{i:02d}" for i in range(nshards)]
    if subset is not None:
        facts["smoke_goals"] = subset.record()
    missing = [sh for sh in expected if not all(os.path.isfile(os.path.join(store, sh, f)) for f in MULTITURN_FILES)]
    present = sorted(os.path.basename(d) for d in glob.glob(os.path.join(store, "shard*")) if os.path.isdir(d))
    unexpected = sorted(set(present) - set(expected))
    facts.update({"shards_present": present, "shards_missing": missing, "shards_unexpected": unexpected})
    cons = _json(os.path.join(store, MULTITURN_CONSOLIDATION))
    if cons:                                              # compact: the note is capped at 2,000 chars; the file beside the shards holds the rest
        facts["consolidation"] = {"copied_utc": cons.get("copied_utc"),
                                  "shards": {sh: f"{(src or {}).get('cluster')} job {(src or {}).get('job_id')}" for sh, src in sorted((cons.get("shards") or {}).items())}}
    shards = list(expected)
    if missing and tolerated_missing and set(missing) <= set(tolerated_missing):
        # the missing shards are RULED OUT — every PRESENT shard is still inspected in full below, each
        # against its OWN goal assignment (jailbreak_run: goals[shard::nshards]); the partial close itself (`cancelled`) is the caller's
        if expected_ids is None:
            return (f"{store}: a partial close over {', '.join(sh for sh in expected if sh not in missing)} needs the registered goal set "
                    f"(the config's eval goals) to validate each present shard's assignment"), facts, ""
        shards = [sh for sh in expected if sh not in missing]
        facts["partial"] = {"shards_missing_ruled_out": list(missing), "shards_closed": list(shards)}
    elif missing:
        return f"{store}: {len(missing)} of {len(expected)} expected shard(s) missing or incomplete: {', '.join(missing)}", facts, ""
    if unexpected:
        return (f"{store}: shard directories outside the job's plan of {nshards}"
                + (f" (the smoke subset {SG.KEY}={subset.spec} holds {', '.join(expected)} only)" if subset is not None else "")
                + f": {', '.join(unexpected)}"), facts, ""
    # the grader's reading (jailbreak_run._load_progress): the LATEST row of a goal is its outcome (a resumed goal appends a new row); torn
    # lines are skipped there, refused here (a store still being written is not complete)
    outcomes: Dict[str, int] = {}; not_graded: List[str] = []; shard_of: Dict[str, str] = {}; dup: List[str] = []; per_shard: Dict[str, int] = {}
    unparsable = 0; responses_of: Dict[str, int] = {}; resumed = 0; goals_in: Dict[str, set] = {}; resp_in: Dict[str, Dict[str, int]] = {}
    for sh in shards:
        latest: Dict[str, str] = {}
        for r in _rows(os.path.join(store, sh, "progress.jsonl")):
            if r.get("_unparsable"):
                unparsable += 1; continue
            gid = str(r.get("goal_id"))
            if gid in latest:
                resumed += 1
            latest[gid] = str(r.get("outcome") or "missing")
        for gid, oc in latest.items():
            outcomes[oc] = outcomes.get(oc, 0) + 1
            if oc.lower().startswith(MULTITURN_NOT_GRADED_PREFIXES):
                not_graded.append(f"{sh}:{gid}={oc}")
            if gid in shard_of and shard_of[gid] != sh:
                dup.append(f"{gid} in {shard_of[gid]} and {sh}")
            shard_of.setdefault(gid, sh)
        per_shard[sh] = len(latest); goals_in[sh] = set(latest)
        mine: Dict[str, int] = {}
        for r in _rows(os.path.join(store, sh, "responses.jsonl")):
            if r.get("_unparsable"):
                unparsable += 1; continue
            gid = str(r.get("goal_id")); responses_of[gid] = responses_of.get(gid, 0) + 1; mine[gid] = mine.get(gid, 0) + 1
        resp_in[sh] = mine
    # a response row counts for a goal only in the goal's OWN shard — jailbreak_run's drivers write a shard's
    # progress and responses side by side and the grader reads each shard's pair — so a row filed under another shard (or for a goal no shard
    # holds) is named as an orphan, never counted as coverage; before, one tally across the shards let such a row stand in for a missing reply
    orphan = [(sh, gid, n_) for sh in shards for gid, n_ in sorted(resp_in.get(sh, {}).items()) if gid not in goals_in.get(sh, set())]
    without = sorted(g for g, sh in shard_of.items() if not resp_in.get(sh, {}).get(g))
    facts.pop("shards_present", None)                     # complete here: the plan's shards, in order (the note is capped at 2,000 chars)
    facts.update({"goals_per_shard": [per_shard[sh] for sh in shards], "goals_distinct": len(shard_of), "goals_planned": planned, "outcomes": outcomes,
                  "goals_not_graded": len(not_graded), "goals_in_two_shards": len(dup), "goals_without_response": len(without),
                  "responses": sum(responses_of.values()), "responses_per_shard": [sum(resp_in.get(sh, {}).values()) for sh in shards],
                  "response_rows_outside_their_shards_progress": sum(n_ for _s, _g, n_ in orphan),
                  "progress_rows_superseded_by_a_later_row": resumed, "unparsable_rows": unparsable})
    if unparsable:
        return f"{store}: {unparsable} unparsable line(s) across the shards' progress / responses", facts, ""
    if not_graded:
        return f"{store}: {len(not_graded)} progress row(s) the grader would not grade ({', '.join(not_graded[:4])}{'…' if len(not_graded) > 4 else ''})", facts, ""
    if dup:
        return f"{store}: {len(dup)} goal(s) appear in two shards (e.g. {', '.join(dup[:3])})", facts, ""
    if orphan:
        where = [f"{sh}:{gid}" + (f" (progress in {shard_of[gid]})" if gid in shard_of else " (no progress row in any shard)") for sh, gid, _n in orphan[:3]]
        return (f"{store}: {sum(n_ for _s, _g, n_ in orphan)} response row(s) for {len(orphan)} goal(s) sit in a shard whose progress holds no row for their goal "
                f"(e.g. {', '.join(where)}{'…' if len(orphan) > 3 else ''}) — a response counts only in its goal's own shard"), facts, ""
    if without:
        return f"{store}: {len(without)} goal(s) carry no response row (e.g. {', '.join(without[:3])})", facts, ""
    want = planned
    if expected_ids is not None:
        ids = [str(g) for g in expected_ids]; exp_all = set(ids)
        # each shard against ITS assignment — jailbreak_run deals goals[shard::nshards] (the i-th goal to shard i % nshards)
        assign = {f"shard{k:02d}": {ids[i] for i in range(len(ids)) if i % nshards == k} for k in range(nshards)}
        narrowed = bool(facts.get("partial")) or subset is not None
        if narrowed:
            # ONLY when the set is narrowed by construction (a partial close, or the smoke goal-shard subset): the registry's unit
            # is the GOAL SET and a complete store is held to it below whatever its shards' dealing; a narrowed store has no whole set
            # to compare, so each present shard is held to its own deal
            for sh in shards:
                extra = sorted(goals_in.get(sh, set()) - assign[sh]); short = sorted(assign[sh] - goals_in.get(sh, set()))
                if extra or short:
                    k = int(sh[5:])
                    return (f"{store}: {sh} does not hold its assignment goals[{k}::{nshards}] ({len(short)} missing: {', '.join(short[:3])}"
                            f"{'…' if len(short) > 3 else ''}; {len(extra)} outside it: {', '.join(extra[:3])}{'…' if len(extra) > 3 else ''})"), facts, ""
        exp = set().union(*(assign[sh] for sh in shards)) if narrowed else exp_all
        have = set(shard_of); missing_g = sorted(exp - have); foreign = sorted(have - exp)
        facts.update({"goals_expected": len(exp), "goals_missing": len(missing_g), "goals_foreign": len(foreign)})
        if missing_g or foreign:
            return (f"{store}: the goal set differs from the registered config's eval set"
                    + (f" dealt over the smoke subset's shards {', '.join(shards)} of {nshards} ({SG.KEY}={subset.spec})" if subset is not None else "")
                    + f" ({len(missing_g)} missing: {', '.join(missing_g[:3])}"
                    f"{'…' if len(missing_g) > 3 else ''}; {len(foreign)} foreign: {', '.join(foreign[:3])}{'…' if len(foreign) > 3 else ''})"), facts, ""
        if len(exp_all) != planned:
            return f"{store}: the registered config resolves {len(exp_all)} eval goal(s), the registration plans {planned}", facts, ""
        if narrowed:
            want = len(exp)
    if len(shard_of) != want:
        return (f"{store}: {len(shard_of)} distinct goal(s) across the shards, " + (f"the present shards' assignment holds {want} (the registration plans {planned})"
                if facts.get("partial") else f"the registration plans {planned}")), facts, ""
    return None, facts, os.path.join(store, shards[0], "progress.jsonl")


GATE_EXITS = ("exit_le_1pct", "exit_ceiling")


def gate_path(wave_root: str, tag: str, trial: int, obj: str, key: str, cap: int) -> str:
    return os.path.join(wave_root, "sampled", "ladders", f"{tag}_t{trial}", "gates", f"cap{int(cap)}", f"{obj}__{key}.decision.json")


def gate_decision(wave_root: str, tag: str, trial: int, obj: str, key: str, cap: int) -> Optional[dict]:
    """The ladder gate's record for (cell, control) at `cap` — `<wave_root>/sampled/ladders/<tag>_t<trial>/gates/cap<cap>/<cell>__<control>.decision.json`
    (sampled/ladder_gate.py: the ONE BOUND DECISION of the controls loop; `decision` is exit_le_1pct | exit_ceiling | escalate). None when the
    rung has no gate record (a pre-gate store: the driver's cap_ladder.json is then the only reading). A record that exists decides only once
    :func:`gate_binding` has bound it to the selected store and generation."""
    p = gate_path(wave_root, tag, trial, obj, key, cap)
    return _json(p) if os.path.isfile(p) else None


def control_spool(cell: str, key: str) -> Optional[str]:
    """The control's generation spool in a controls cell: private/gens_<key>*.jsonl (not the .caps. / .meta. sidecars)."""
    gens = sorted(p for p in glob.glob(os.path.join(cell, "private", f"gens_{key}*.jsonl")) if ".caps." not in p and ".meta." not in p)
    return gens[0] if gens else None


def _iso_utc(s: object) -> Optional[float]:
    try:
        import datetime as _dt
        d = _dt.datetime.fromisoformat(str(s).replace("Z", "+00:00"))
        if d.tzinfo is None:
            d = d.replace(tzinfo=_dt.timezone.utc)
        return d.timestamp()
    except (TypeError, ValueError):
        return None


def gate_binding(gate: dict, path: str, *, cell: str, obj: str, key: str, cap: int) -> Tuple[Optional[str], Dict[str, object], Optional[Dict[str, str]]]:
    """(refusal or None, facts, the spool's {item_id: text}). A gate record decides only when it is BOUND to the
    selected store and its CURRENT generation: kind `controls`; identity.cell / control / store naming this cell dir; cap = the rung; counts.n_responses
    = the spool's items; and identity.store_digest (written by a current ladder_decide) = the spool's content digest — or, for a record without one, a
    tool.written_at that postdates the spool's last write (a regeneration rewrites the spool). Anything else is refused BY NAME (stale / wrong store /
    missing identity): the driver's flag never quietly replaces an unbound record, and the legitimate driver-vs-gate disagreement stands only where
    the record is bound. The store it binds to must VALIDATE (gen_stats.json[control] + the meta
    sidecar, at_cap_controls.read_control — a store missing a sidecar or whose sidecars disagree binds nothing), and a bound record is then
    RE-DERIVED from that store under the sibling state it declares (:func:`gate_rederivation`); a record the store does not reproduce is refused."""
    facts: Dict[str, object] = {"gate_record": path}
    ident = gate.get("identity") if isinstance(gate.get("identity"), dict) else None
    if gate.get("kind") != "controls" or not ident or not all(ident.get(k) for k in ("store", "cell", "control")):
        return f"gate record {path} carries no controls identity (kind={gate.get('kind')!r}; identity needs store/cell/control) — unbound, it decides nothing", facts, None
    if str(ident.get("cell")) != obj or str(ident.get("control")) != key:
        return f"gate record {path} names cell {ident.get('cell')!r} / control {ident.get('control')!r}, the selected store is {obj!r} / {key!r} — wrong record", facts, None
    if os.path.normpath(str(ident.get("store"))) != os.path.normpath(cell):
        return f"gate record {path} judged store {ident.get('store')}, the selected cell is {cell} — wrong store", facts, None
    try:
        if int(gate.get("cap")) != int(cap):
            return f"gate record {path} is for cap {gate.get('cap')}, the rung is {cap}", facts, None
    except (TypeError, ValueError):
        return f"gate record {path} carries no integer cap ({gate.get('cap')!r})", facts, None
    spool = control_spool(cell, key)
    if not spool:
        return f"{cell}: no generation spool private/gens_{key}*.jsonl to bind the gate record to", facts, None
    try:
        from pathlib import Path as _P
        texts = acc._spool_texts(_P(spool))
    except acc.Refusal as exc:
        return f"{spool}: the gens spool cannot be read as a complete store ({exc}) — the gate record cannot bind", facts, None
    # the record binds to a VALIDATED store — gen_stats.json[control] (n_total = the spooled items, n_at_cap and
    # at_cap_item_ids = the meta sidecar's tallies) and the meta sidecar bound row by row to the spooled texts: at_cap_controls.read_control, the
    # producer's own reading. A store missing a sidecar, or whose sidecars disagree, binds nothing
    try:
        gs_rec = acc.read_control(_P(cell), key)
        meta = acc._meta_bound(_P(spool).with_name(os.path.basename(spool)[:-len(".jsonl")] + acc.META_SUFFIX), texts)
    except acc.Refusal as exc:
        return f"{cell}: the store's sidecars do not validate ({exc}) — the gate record cannot bind to an unvalidated store", facts, texts
    gs_spool = str((((_json(os.path.join(cell, "gen_stats.json")) or {}).get(key) or {}).get("spool")) or "")
    if os.path.basename(gs_spool) != os.path.basename(spool):
        return (f"{cell}: gen_stats.json names spool {os.path.basename(gs_spool)!r} for {key}, the cell's generation spool is {os.path.basename(spool)!r} — "
                f"two spools; the gate record cannot bind"), facts, texts
    if int(gs_rec["cap"]) != int(cap):
        return f"{cell}: gen_stats.json records cap {gs_rec['cap']} for {key}, the rung is {cap} — the store is not this rung's", facts, texts
    facts["sidecars"] = {"n_total": int(gs_rec["n"]), "n_at_cap": int(gs_rec["n_at_cap"])}
    counts = gate.get("counts") if isinstance(gate.get("counts"), dict) else {}
    n_resp = counts.get("n_responses")
    if n_resp is None or int(n_resp) != len(texts):
        return f"gate record {path} counts {n_resp!r} response(s), the spool holds {len(texts)} item(s) — stale or wrong generation", facts, texts
    n_items = ident.get("n_items")
    if n_items is not None and int(n_items) != len(texts):
        return f"gate record {path} judged {n_items} item(s), the spool holds {len(texts)} — stale generation", facts, texts
    digest = acc.store_digest(texts); facts["spool_digest"] = digest[:16]
    if ident.get("store_digest"):
        if str(ident["store_digest"]) != digest:
            return (f"gate record {path} is STALE: it judged generation {str(ident['store_digest'])[:16]}…, the spool now holds {digest[:16]}… "
                    f"(a replaced generation; re-run the ladder gate at this rung)"), facts, texts
        facts["gate_binding"] = "digest"
    else:
        written = _iso_utc((gate.get("tool") or {}).get("written_at")) if isinstance(gate.get("tool"), dict) else None
        if written is None:
            return f"gate record {path} carries neither a store digest nor a tool.written_at — unbound to any generation", facts, texts
        mtime = os.path.getmtime(spool)
        if written < mtime:
            return (f"gate record {path} is STALE: written {(gate.get('tool') or {}).get('written_at')} predates the spool's last write "
                    f"({utc_now() if False else _dt_iso(mtime)}) — a regenerated store; re-run the ladder gate at this rung"), facts, texts
        facts["gate_binding"] = "identity+count+order"        # a pre-digest record: bound by identity, count and the write order
    # a bound record is held to the store — re-derived under the sibling state it declares
    why_r, rfacts = gate_rederivation(gate, path, obj=obj, key=key, cap=cap, texts=texts, meta=meta)
    facts.update(rfacts)
    return why_r, facts, texts


#: the blocks a ladder_decide record carries that the closer re-derives its decision from
GATE_RECORD_BLOCKS = ("rules", "ladder", "rep_ratio", "degeneracy", "sibling", "counts")
GATE_THRESHOLDS = ("trigger_share", "loop_rep_ratio_threshold", "repetitive_rep_ratio_min", "tail_words", "boundary_fraction")
GATE_COUNT_KEYS = ("n_responses", "n_at_cap", "n_loop", "n_repetitive", "n_boundary", "n_boundary_any", "n_truncated", "n_sibling_unreadable")
SIBLING_STATES = ("used", "unavailable", "not_given")
#: the classifier sources a record stamps (path + sha256), and where this checkout keeps the same modules
CLASSIFIER_REPO_RELATIVE = {"rep_ratio": "src/sisteer/capeval/tiers/cap_ladder.py", "degeneracy": "src/sisteer/eval/degeneracy.py"}


def _classifier_source(block: object, name: str) -> Tuple[Optional[str], str, str]:
    """(refusal or None, the path to load, how it was found). The record's stamped source when it is on disk with the stamped sha256; else this
    checkout's copy of the same module when ITS sha256 is the stamped one; else refused by name — a decision re-derived with another
    classifier would be another decision."""
    b = block if isinstance(block, dict) else {}
    src = str(b.get("source") or ""); want = str(b.get("source_sha256") or "")
    if not src or not re.fullmatch(r"[0-9a-f]{64}", want):
        return f"the record's {name} block names no source path + sha256", "", ""
    tried: List[str] = []
    for cand, how in ((src, "the record's stamped path"), (os.path.join(REPO_ROOT, CLASSIFIER_REPO_RELATIVE[name]), "this checkout's copy")):
        if os.path.isfile(cand):
            got = _sha256(cand)
            if got == want:
                return None, cand, how
            tried.append(f"{cand} @ {got[:12]}")
        else:
            tried.append(f"{cand} absent")
    return (f"the {name} source the record was decided with ({want[:12]}…) is not readable here (tried {'; '.join(tried)}) — the decision cannot be "
            f"re-derived"), "", ""


def gate_rederivation(gate: dict, path: str, *, obj: str, key: str, cap: int, texts: Dict[str, str], meta: Dict[str, dict]) -> Tuple[Optional[str], Dict[str, object]]:
    """(refusal or None, facts). A BOUND gate record is held to the store it binds: the closer
    re-derives the decision from the validated store — the spooled texts, the meta sidecar's at-cap flags — with the record's own thresholds,
    ladder and classifier sources (by sha256), UNDER THE SIBLING STATE THE RECORD DECLARES. `sibling.status` used: the sibling store the record
    names is read now, and its item set, cap and (when the record carries one) its `store_digest` must still be
    what the gate paired — a regenerated or vanished sibling is refused by name; unavailable / not_given: no boundary calls, as the gate ruled
    (the three t0 count drifts seen were records written while the t1 sibling was still absent — not a finding, and never a reason to re-read
    a sibling the record did not use). The re-derived decision and every count must equal the record's; anything else is refused by name. The
    record is never rewritten here; the recovery is the ladder gate re-run at the rung."""
    facts: Dict[str, object] = {}
    for k in GATE_RECORD_BLOCKS:
        v = gate.get(k)
        if not isinstance(v, (dict, list)) or not v:
            return f"gate record {path} carries no {k} block — not a ladder_decide record; its decision cannot be re-derived", facts
    rules = gate["rules"]
    missing = [t for t in GATE_THRESHOLDS if t not in rules]
    if missing:
        return f"gate record {path}: rules carry no {', '.join(missing)} — the decision cannot be re-derived under the record's thresholds", facts
    try:
        th = {"trigger_share": float(rules["trigger_share"]), "loop_rep_ratio_threshold": float(rules["loop_rep_ratio_threshold"]),
              "repetitive_rep_ratio_min": float(rules["repetitive_rep_ratio_min"]), "tail_words": int(rules["tail_words"]),
              "boundary_fraction": float(rules["boundary_fraction"])}
        ladder = tuple(int(x) for x in gate["ladder"])
    except (TypeError, ValueError) as exc:
        return f"gate record {path}: rules / ladder are not numeric ({exc})", facts
    if int(cap) not in ladder:
        return f"gate record {path}: cap {cap} is not on the record's ladder {list(ladder)}", facts
    sib = gate["sibling"]; state = str(sib.get("status") or "")
    if state not in SIBLING_STATES:
        return (f"gate record {path} declares no sibling state (sibling.status={sib.get('status')!r}; one of {', '.join(SIBLING_STATES)}) — the boundary "
                f"calls cannot be re-derived"), facts
    try:
        import ladder_decide as ld  # noqa: E402  (the sampled pipeline's gate: the classify_store / decide the record was written with)
    except ImportError as exc:
        return f"this checkout's sampled/ladder_decide.py cannot be imported ({exc}) — the decision cannot be re-derived", facts
    why_s, rep_path, rep_how = _classifier_source(gate.get("rep_ratio"), "rep_ratio")
    if why_s:
        return f"gate record {path}: {why_s}", facts
    why_s, deg_path, deg_how = _classifier_source(gate.get("degeneracy"), "degeneracy")
    if why_s:
        return f"gate record {path}: {why_s}", facts
    from pathlib import Path as _P
    try:
        rep_fn, _ri = ld.extract_rep_ratio(_P(rep_path)); deg_fn, _di = ld.load_degeneracy(_P(deg_path))
    except ld.Refusal as exc:
        return f"gate record {path}: the classifier sources do not load ({exc})", facts
    facts["classifier_sources"] = {"rep_ratio": rep_how, "degeneracy": deg_how}
    ids = sorted(texts); sib_rows: Optional[Dict[str, dict]] = None
    facts["sibling_state"] = state
    if state == "used":
        out_root = str(sib.get("out_root") or "")
        if not out_root:
            return f"gate record {path} declares the sibling used but names no sibling out_root", facts
        try:
            sst = ld.read_controls_store(_P(out_root), obj, key)
        except ld.Refusal as exc:
            return (f"gate record {path} declares the sibling store under {out_root} USED, but it now reads: {exc} — the sibling changed since the "
                    f"gate ruled; re-run the ladder gate at this rung"), facts
        if sib.get("store") and os.path.normpath(str(sst["cell_dir"])) != os.path.normpath(str(sib["store"])):
            return f"gate record {path}: the sibling root {out_root} now resolves {obj} to {sst['cell_dir']}, the record paired {sib['store']} — another sibling store", facts
        if set(sst["meta"]) != set(ids):
            return f"gate record {path}: the sibling store {sst['cell_dir']} holds {len(sst['meta'])} item(s), this store {len(ids)} — not the item set the gate paired", facts
        if int(sst["cap"]) != int(cap):
            return f"gate record {path}: the sibling store {sst['cell_dir']} is at cap {sst['cap']}, the rung is {cap}", facts
        sdig = acc.store_digest(sst["texts"]); facts["sibling_digest"] = sdig[:16]
        if sib.get("store_digest"):
            if str(sib["store_digest"]) != sdig:
                return (f"gate record {path} is STALE on its sibling: it paired generation {str(sib['store_digest'])[:16]}… of {sst['cell_dir']}, which now "
                        f"holds {sdig[:16]}… (a regenerated sibling; re-run the ladder gate at this rung)"), facts
            facts["sibling_binding"] = "digest"
        else:
            facts["sibling_binding"] = "declared-state"      # a record before closer-fix-3: the re-derivation below is what holds it to the sibling
        sib_rows = {i: {"finish_reason": m.get("finish_reason"), "completion_tokens": m.get("completion_tokens"), "source_cap": m.get("source_cap")}
                    for i, m in sst["meta"].items()}
    items = [(i, texts[i], bool(meta[i].get("at_cap"))) for i in ids]
    _per_item, counts = ld.classify_store(items, sib_rows, cap=int(cap), rep_ratio=rep_fn, degeneracy=deg_fn, thresholds=th)
    verdict = ld.decide(counts, cap=int(cap), ladder=ladder, trigger=th["trigger_share"])
    rec_counts = gate["counts"]; diff: Dict[str, object] = {}
    # closer-fix-4: a count key the record lacks is a refusal by name — it used to be skipped, so a counts block of
    # another closer generation (a classifier's count missing) could read as "reproduced" on the keys it happened to carry
    missing = [k for k in GATE_COUNT_KEYS if k not in rec_counts]
    if missing:
        return (f"gate record {path} counts block lacks {', '.join(missing)} (GATE_COUNT_KEYS) — a record of another closer generation cannot be "
                f"re-derived key for key; re-run the ladder gate at this rung"), facts
    for k in GATE_COUNT_KEYS:
        try:
            if int(rec_counts[k]) != int(counts[k]):
                diff[k] = (rec_counts[k], counts[k])
        except (TypeError, ValueError):
            diff[k] = (rec_counts[k], counts[k])
    facts["rederived"] = {"decision": verdict["decision"], **{k: counts[k] for k in GATE_COUNT_KEYS}}
    if diff or verdict["decision"] != str(gate.get("decision")):
        return (f"gate record {path} is not reproduced from the store under its declared sibling state ({state}): record {gate.get('decision')}, "
                f"re-derived {verdict['decision']}; counts differing (record, re-derived): {diff or 'none'} — a record of another generation or sibling; "
                f"re-run the ladder gate at this rung"), facts
    facts["gate_rederived"] = "reproduced"
    return None, facts


def _dt_iso(ts: float) -> str:
    import datetime as _dt
    return _dt.datetime.fromtimestamp(ts, tz=_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def inspect_controls(wave_root: str, tag: str, trial: int, obj: str, eval_id: str, *, planned: int) -> Tuple[Optional[str], Dict[str, object], str, str]:
    """(refusal or None, facts, store dir, artifact). The store is the object's cell at the FINAL rung — the highest controls_cap<C> where the
    LADDER GATE's decision for the run's control is an exit (`gate_decision`: exit_le_1pct | exit_ceiling — the gate is the ONE BOUND
    DECISION; the driver's cap_ladder.json `escalate` flag counts every at-cap item, boundary noise included, and decides nothing where a
    gate record exists), else, for a rung without a gate record, where cap_ladder.json shows escalate: false — and its generation spool
    holds the control's n rows (>= planned). closer-gate (2026-10-02): mix-psr x xstest / or_bench_hard exited the gate at 2048 and the
    driver's flag said escalate; the closer refused two complete stores by reading the wrong record."""
    facts: Dict[str, object] = {"layout": "controls"}
    key = CONTROL_KEY.get(eval_id)
    if not key:
        return f"eval {eval_id!r}: no controls key", facts, "", ""
    prefix = f"{obj.replace('__', '--')}__controls__"
    caps = []
    for d in glob.glob(os.path.join(wave_root, "sampled", "controls_cap*", f"{tag}_t{trial}", "cells")):
        m = re.search(r"controls_cap(\d+)", d)
        if m:
            caps.append((int(m.group(1)), d))
    facts["rungs_seen"] = sorted(c for c, _d in caps)
    if not caps:
        return f"no controls rungs under {wave_root}/sampled/controls_cap*/{tag}_t{trial}/cells", facts, "", ""
    ladders: Dict[int, dict] = {}
    for cap, cells in sorted(caps, reverse=True):
        cell_dirs = sorted(d for d in glob.glob(os.path.join(cells, prefix + "*")) if os.path.isdir(d))
        if not cell_dirs:
            continue
        cell = cell_dirs[-1]; ladder = _json(os.path.join(cell, "cap_ladder.json")) or {}
        ctl = (ladder.get("controls") or {}).get(key) or {}
        gate = gate_decision(wave_root, tag, trial, obj, key, cap); texts: Optional[Dict[str, str]] = None; bind: Dict[str, object] = {}
        if gate is not None:                                   # closer-fix-2: an existing record must BIND before it decides anything
            why_b, bind, texts = gate_binding(gate, gate_path(wave_root, tag, trial, obj, key, cap), cell=cell, obj=obj, key=key, cap=cap)
            if why_b:
                facts.update({"ladders": ladders, "gate_rung": cap, **bind})
                return f"{cell}: {why_b}", facts, cell, ""
        gate_exit = bool(gate) and str(gate.get("decision")) in GATE_EXITS
        ladders[cap] = {"escalate": ctl.get("escalate"), "exit": ctl.get("exit"), "n": ctl.get("n"),
                        "gate": (str(gate.get("decision")) if gate else None)}
        final = gate_exit if gate is not None else bool(ctl and ctl.get("escalate") is False)      # the gate decides where it ruled
        if final:
            if not ctl:
                return f"{cell}: the gate exited {key} at {cap} but the cell carries no cap_ladder.json controls.{key} record", facts, cell, ""
            n_ctl = int(ctl.get("n") or 0)
            gens = [control_spool(cell, key)] if control_spool(cell, key) else []
            rows = len(texts) if texts is not None else (_count_lines(gens[0]) if gens else 0)     # bound record: the spool's items; else the line count
            facts.update({"final_rung": cap, "cell": os.path.basename(cell), "control": key, "exit": (gate.get("decision") if gate else ctl.get("exit")),
                          "decided_by": ("ladder_gate" if gate is not None else "cap_ladder"), "n_control": n_ctl,
                          "gens_rows": rows, "planned": planned, "ladders": ladders, **bind})
            if not gens:
                return f"{cell}: no generation spool private/gens_{key}*.jsonl", facts, cell, ""
            if rows < n_ctl or rows < planned:
                return f"{cell}: {rows} generated row(s), the ladder records {n_ctl} and the registration plans {planned}", facts, cell, ""
            return None, facts, cell, os.path.join(cell, "cap_ladder.json")
    facts["ladders"] = ladders
    return (f"no rung of {tag}_t{trial} holds {obj}'s cell with {key} exited (the gate's decision where one exists, else cap_ladder.json "
            f"escalate=false) (ladders: {ladders})"), facts, "", ""


# ------------------------------------------------------------------------------------------------------------ the closing
def misalign_cell_of(obj: str) -> str:
    """The misalign runner writes its cells under the Petri HARNESS cell id (`prompted__safety`), the registration names the wave object
    (`prompted__safety_refusal`): registry_adapter.harness_cell_id is the one translation (the petri hand-off's `harness_object`)."""
    from sisteer.capeval.petri import registry_adapter as ra
    return ra.harness_cell_id(obj)


def inspect_misalign(misalign_root: str, obj: str, eval_id: str, *, planned: Optional[int]) -> Tuple[Optional[str], Dict[str, object], str, str]:
    """The misalign runner's agentic layout: `<root>/agentic/<harness cell>/responses.jsonl` (one row per (item, sample_idx), `behavior` on
    every row; both behaviours of the family share one cell file) + `meta.json` (`items`, `item_ids`, `n_samples`, `present`, `finished`).
    The run is (eval `agentic_<behaviour>`, object): complete when the cell FINISHED (`meta.finished`), every row parses, the cell holds
    exactly items x n_samples distinct (item, sample) rows (== `present`), and this behaviour's items — the rows' item ids, all of them in
    `meta.item_ids` — each carry n_samples rows. The plan is the store's frozen one (`meta`) when the registration names no
    n_items_planned (the agentic item sets record no size); a registration count that disagrees refuses. Returns (why, facts, store, artifact)."""
    facts: Dict[str, object] = {}
    try:
        cell = misalign_cell_of(obj)
    except Exception as exc:  # noqa: BLE001
        return f"object {obj!r} has no Petri harness cell id ({type(exc).__name__}: {exc})", facts, "", ""
    m = re.match(r"^agentic_(.+)$", eval_id)
    if not m:
        return f"eval {eval_id!r} is not an agentic_<behaviour> eval", facts, "", ""
    behaviour = m.group(1)
    store = os.path.join(misalign_root, "agentic", cell)
    rp, mp = os.path.join(store, "responses.jsonl"), os.path.join(store, "meta.json")
    facts.update({"store": store, "cell": cell, "behaviour": behaviour})
    if not os.path.isdir(store):
        return f"{store}: no cell directory", facts, store, ""
    meta = _json(mp)
    if not meta:
        return f"{store}: no readable meta.json", facts, store, ""
    if not meta.get("finished"):
        return f"{store}: meta.json says the cell has not finished", facts, store, ""
    n_items, n_samples, present = meta.get("items"), meta.get("n_samples"), meta.get("present")
    item_ids = set(str(i) for i in (meta.get("item_ids") or []))
    facts.update({"items": n_items, "n_samples": n_samples, "present": present})
    if not isinstance(n_items, int) or not isinstance(n_samples, int) or n_items <= 0 or n_samples <= 0:
        return f"{store}: meta.json names no items / n_samples", facts, store, ""
    if not os.path.isfile(rp):
        return f"{store}: no responses.jsonl", facts, store, ""
    rows = _rows(rp)
    bad = sum(1 for r in rows if r.get("_unparsable"))
    if bad:
        return f"{store}: {bad} unparsable line(s) in responses.jsonl", facts, store, ""
    slots = {(str(r.get("item_id")), r.get("sample_idx")) for r in rows}
    facts.update({"rows": len(rows), "slots_distinct": len(slots)})
    if len(rows) != n_items * n_samples or len(slots) != len(rows) or present != len(rows):
        return f"{store}: {len(rows)} row(s) / {len(slots)} distinct slot(s) / present {present} vs items {n_items} x n_samples {n_samples} = {n_items * n_samples}", facts, store, ""
    mine = [r for r in rows if str(r.get("behavior")) == behaviour]
    my_items = {str(r.get("item_id")) for r in mine}
    facts.update({"behaviour_rows": len(mine), "behaviour_items": len(my_items)})
    if not mine:
        return f"{store}: no row carries behavior {behaviour!r}", facts, store, ""
    if not my_items <= item_ids:
        return f"{store}: {len(my_items - item_ids)} item id(s) of behaviour {behaviour!r} are not in meta.item_ids", facts, store, ""
    per = {}
    for r in mine:
        per[str(r.get("item_id"))] = per.get(str(r.get("item_id")), 0) + 1
    short = {k: v for k, v in per.items() if v != n_samples}
    if short:
        return f"{store}: {len(short)} item(s) of behaviour {behaviour!r} do not carry n_samples={n_samples} rows (e.g. {sorted(short.items())[:3]})", facts, store, ""
    if planned is not None and planned != len(mine):
        return f"{store}: the registration plans {planned} item(s), the finished cell holds {len(mine)} for behaviour {behaviour!r}", facts, store, ""
    facts["plan_source"] = "registration" if planned is not None else "store meta (the registration names no n_items_planned)"
    return None, facts, store, mp


NOFIT_SOURCES = ("tokenize_endpoint", "error_message")
#: the prompt-FIT geometry (window + prompt-length inputs identical): the exact prompt length and the three window
#: readings. NOT `requested_max_tokens` — the completion budget the caller asked for, which the 512 store and the 4096 climb spell differently for
#: the SAME prompt against the SAME window (the live constitution cell: 6,946 slots carried both a 512- and a 4096-budget refusal, each a proof)
GEOMETRY_KEYS = ("prompt_tokens", "effective_window", "server_window", "window")
#: the keys served_window.PromptOverWindow.record() (+ the BoN caller's slot / serving keys) always write: a row missing any of them is a TORN
#: duplicate (two writers' appends interleaved, keys misspelt) — malformed, never a competing geometry stamp
NOFIT_CANONICAL_KEYS = ("goal_id", "sample_idx", "trial", "error", "prompt_tokens", "prompt_tokens_source", "prompt_floor", "completion_tokens", "window",
                        "server_window", "effective_window", "window_reason")


def nofit_slot_audit(store: str, planned_keys) -> Dict[str, object]:
    """The certificate's own per-slot audit over the RAW error ledger, never "first proof wins": a planned slot is
    CERTIFIED only when every VALID row on it is a no-fit classification (bon_climb.classify_over_window == nofit_proven) carrying one
    geometry stamp (prompt_tokens, effective_window, server_window, window, requested_max_tokens identical across the slot's rows — a resumed
    store whose serving geometry changed leaves differently-stamped rows and an old proof does not establish that the CURRENT attempt cannot
    fit). A valid row of another kind, or a differently-stamped valid row, makes the slot UNDECIDED (listed by name). A row is VALID when it
    parses, carries a well-formed slot key and the canonical no-fit fields (`error == prompt_over_window`, int `prompt_tokens`,
    `prompt_tokens_source` in NOFIT_SOURCES, int `effective_window`); anything else on the slot — a torn duplicate with a misspelt field, a
    malformed line — is counted (`n_malformed_rows`) and decides nothing. Returns {n_planned, n_certified, undecided: [...], n_no_row,
    n_malformed_rows, n_foreign_valid_rows, geometry: {windows, prompt_tokens_min/max}}."""
    from sisteer.eval.attacks import bon_climb as BC
    planned = {tuple(k) for k in planned_keys}
    valid: Dict[tuple, list] = {}; malformed = 0; foreign = 0
    for r in _rows(os.path.join(store, "bon_errors.jsonl")):
        if r.get("_unparsable"):
            malformed += 1; continue
        key = (r.get("goal_id"), r.get("sample_idx"), r.get("trial", 0))
        well_formed = (all(k in r for k in NOFIT_CANONICAL_KEYS) and isinstance(key[0], str) and key[0] and isinstance(key[1], int) and isinstance(key[2], int)
                       and r.get("error") == BC.NOFIT_ERROR and isinstance(r.get("prompt_tokens"), int) and r.get("prompt_tokens_source") in NOFIT_SOURCES
                       and isinstance(r.get("effective_window"), int))
        if not well_formed:
            malformed += 1; continue
        if key not in planned:
            foreign += 1; continue
        valid.setdefault(key, []).append(r)
    certified = 0; undecided = []; no_row = 0; wins = set(); pts = []
    for key in sorted(planned):
        rows = valid.get(key)
        if not rows:
            no_row += 1; undecided.append({"goal_id": key[0], "sample_idx": key[1], "trial": key[2], "why": "no valid no-fit row"}); continue
        kinds = {BC.classify_over_window(r) for r in rows}
        stamps = {tuple(r.get(k) for k in GEOMETRY_KEYS) for r in rows}
        if kinds != {BC.NOFIT_PROVEN}:
            undecided.append({"goal_id": key[0], "sample_idx": key[1], "trial": key[2], "why": f"row kinds {sorted(kinds)}"}); continue
        if len(stamps) != 1:
            undecided.append({"goal_id": key[0], "sample_idx": key[1], "trial": key[2], "why": f"{len(stamps)} geometry stamps {sorted(stamps, key=str)[:2]}"}); continue
        certified += 1; wins.add(rows[0].get("effective_window")); pts.append(int(rows[0]["prompt_tokens"]))
    return {"n_planned": len(planned), "n_certified": certified, "n_undecided": len(undecided), "undecided": undecided[:8], "n_no_row": no_row,
            "n_malformed_rows": malformed, "n_foreign_valid_rows": foreign,
            "geometry": {"windows": sorted(wins), "prompt_tokens_min": (min(pts) if pts else None), "prompt_tokens_max": (max(pts) if pts else None)}}


def bon_nofit_certificate(facts: Dict[str, object], store: str, planned_keys) -> Tuple[Optional[str], Optional[Dict[str, object]]]:
    """A NO-FIT-BY-DESIGN cell: no response row at all, and EVERY planned slot certified by
    nofit_slot_audit (every valid row a proven no-fit under one geometry stamp). Returns (refusal, None) when the cell has no responses but
    some planned slot is undecided — a store that must NOT be filed generated either (its emptiness is uncertified); (None, None) when the
    cell is not a no-fit candidate (it has responses); (None, certificate) when certified."""
    from sisteer.eval.attacks import bon_climb as BC
    completion = facts.get("completion") or {}
    ledger = facts.get("ledger") or {}
    if int(completion.get("n_done") or 0) != 0 or int(ledger.get("n_covered") or 0) != 0:
        return None, None                            # responses exist: not a no-fit cell (the grader grades them)
    audit = nofit_slot_audit(store, planned_keys)
    facts["nofit_audit"] = audit
    if audit["n_undecided"]:
        u0 = audit["undecided"][0]
        return (f"{store}: 0 responses and {audit['n_undecided']} of {audit['n_planned']} planned slot(s) UNDECIDED under the per-slot rule "
                f"(first {u0['goal_id']}/{u0['sample_idx']}: {u0['why']}) — an empty store without a certificate is a genuine failure"), None
    cert = {"kind": "nofit_by_design", "n_slots": audit["n_certified"], "n_planned": audit["n_planned"], "windows": audit["geometry"]["windows"],
            "prompt_tokens_min": audit["geometry"]["prompt_tokens_min"], "prompt_tokens_max": audit["geometry"]["prompt_tokens_max"],
            "min_completion": int(BC.MIN_CLAMPED_COMPLETION), "prompt_tokens_sources": list(NOFIT_SOURCES), "geometry_keys": list(GEOMETRY_KEYS),
            "n_malformed_rows": audit["n_malformed_rows"], "n_foreign_valid_rows": audit["n_foreign_valid_rows"],
            "rule": "every valid row on every planned slot is nofit_proven under one geometry stamp (registry_close_stores.nofit_slot_audit)"}
    return None, cert


def instrument_judge(run: dict, *, repo_root: str) -> Tuple[Optional[str], Optional[Dict[str, object]]]:
    """The run's INSTRUMENT judge as the registry identifies it: the registration's seed instrument (`extra.seed_instrument_id`) ->
    configs/registry/instruments.yaml `judge_id` (the seed's readable id) -> configs/registry/judges.yaml row -> identity.judge_id over its
    four identity fields (what v_pending_judgment.required_judge_id holds). Refuses by name when any link is missing."""
    import yaml
    plan = run.get("plan") or {}
    seed_inst = str((plan.get("extra") or {}).get("seed_instrument_id") or "")
    if not seed_inst:
        return "the registration names no seed instrument (extra.seed_instrument_id)", None
    try:
        insts = yaml.safe_load(open(os.path.join(repo_root, "configs", "registry", "instruments.yaml"), encoding="utf-8"))
        judges = yaml.safe_load(open(os.path.join(repo_root, "configs", "registry", "judges.yaml"), encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return f"the registry seeds could not be read ({type(exc).__name__}: {exc})", None
    rows = insts.get("instruments") if isinstance(insts, dict) else insts
    inst = next((r for r in (rows or []) if str(r.get("instrument_id")) == seed_inst), None)
    if not inst or not inst.get("judge_id"):
        return f"seed instrument {seed_inst!r} names no judge_id in instruments.yaml", None
    jrows = judges.get("judges") if isinstance(judges, dict) else judges
    jrow = next((r for r in (jrows or []) if str(r.get("judge_id")) == str(inst["judge_id"])), None)
    if not jrow:
        return f"judge seed {inst['judge_id']!r} is not in judges.yaml", None
    return None, {"judge_id": identity.judge_id(jrow), "seed_judge_id": str(jrow.get("judge_id")), "judge_model": jrow.get("judge_model"), "rubric_sha256": jrow.get("rubric_sha256")}


def already_certified(spool: str, campaign: str, run_id: str) -> bool:
    """A complete judgment from this closer already sits in the run's spool directory (idempotence of the no-fit filing)."""
    d = os.path.join(spool, campaign, identity.run_dir_name(run_id))
    if not os.path.isdir(d):
        return False
    for f in sorted(os.listdir(d)):
        if not f.endswith(".jsonl"):
            continue
        for r in _rows(os.path.join(d, f)):
            if r.get("record_type") == "judgment" and (r.get("payload") or {}).get("status") == "complete" and (r.get("producer") or {}).get("script") == SCRIPT:
                return True
    return False


def file_nofit_certificate(*, spool: str, push_script: Optional[str], campaign: str, rid: str, store: str, judge: Dict[str, object], cert: Dict[str, object],
                           producer: Dict[str, object], dry_run: bool) -> None:
    """The terminal row of a no-fit-by-design cell: a COMPLETE zero-item judgment for the instrument's judge (required coverage is the
    0 items generated) + run_state judged, carrying the certificate's provenance in the run's notes / extra_json — graded once, never
    dispatched (the grader would refuse an empty store as incomplete every tick)."""
    from sisteer.registry import hooks as HK
    note = (f"no-fit-by-design cell: every one of {cert['n_slots']} planned slots is a typed prompt_over_window refusal under one geometry stamp — prompt "
            f"{cert['prompt_tokens_min']}-{cert['prompt_tokens_max']} tokens ({'/'.join(cert['prompt_tokens_sources'])}) + {cert['min_completion']} minimum completion "
            f"> effective window {cert['windows']}; 0 candidates to judge; {cert['rule']}; filed by {SCRIPT}")
    if dry_run:
        _say(f"DRY-RUN nofit certificate {rid[:40]}…: would file a complete zero-item judgment for judge {judge['seed_judge_id']} ({judge['judge_id'][:12]}…) + run_state judged: {note[:160]}"); return
    w = emit.RecordWriter(spool, campaign, rid, producer, push_script=push_script)
    w.emit_patch(fields={"notes": note[:2000], "extra_json": {"nofit_certificate": cert}})
    w.push()
    scope = HK.judgment_scope(SCRIPT, {"run_id": rid, "campaign": campaign}, {"judge_id": judge["judge_id"], "rubric_sha256": judge.get("rubric_sha256")},
                              spool=spool, push_script=push_script, registry_emit=True, store_uri=store)
    if scope is None:
        raise RuntimeError("the registry emitter is disabled here; the no-fit certificate cannot be filed")
    scope.claim()
    scope.items([]); scope.metrics([])
    scope.complete("complete", n_items_judged=0, n_judge_refusals=0)
    _say(f"judged {rid[:40]}… no-fit-by-design ({cert['n_slots']} slots, window {cert['windows']}, prompt {cert['prompt_tokens_min']}-{cert['prompt_tokens_max']} tok) judge {judge['seed_judge_id']}")


def already_generated(spool: str, campaign: str, run_id: str, states: Sequence[str] = ("generated",)) -> bool:
    """A terminal run_state this closer files (`generated`; `cancelled` for a ruled-out partial store) already sits in the spool."""
    d = os.path.join(spool, campaign, identity.run_dir_name(run_id))
    if not os.path.isdir(d):
        return False
    for fn in sorted(os.listdir(d)):
        m = R.RECORD_FILE_RE.match(fn)
        if not m or m.group("type") != "run_state":
            continue
        try:
            for _i, rec, _sha in R.read_record_file(os.path.join(d, fn), compute_sha=False):
                p = rec.get("payload") if isinstance(rec, dict) else None
                if isinstance(p, dict) and p.get("run_id") == run_id and p.get("state") in states:
                    return True
        except Exception:  # noqa: BLE001 — an unreadable record is not evidence
            continue
    return False


def ruled_out_spec(shards: Optional[str], objects: Optional[str], evidence_uri: Optional[str], note: Optional[str]) -> Tuple[Optional[str], Optional[dict]]:
    """The ruled-out selection the closer may close `cancelled` against — (refusal, spec). `shards` = comma list (`5,6,7,8` or
    `shard05,…`), `objects` = the cells whose runs the ruling excludes (`none`), `evidence_uri` = the ruling's record (REQUIRED: a run is never
    cancelled on a bare flag). None, None when no selection is given."""
    if not shards and not objects and not evidence_uri:
        return None, None
    if not evidence_uri:
        return "--ruled-out-shards / --ruled-out-objects need --ruling-evidence-uri (the ruling that excludes them): a run is never cancelled on a bare flag", None
    if not shards or not objects:
        return "--ruling-evidence-uri needs both --ruled-out-shards (the shards the ruling excludes) and --ruled-out-objects (the cells whose runs it excludes)", None
    sh = set()
    for tok in [t.strip() for t in shards.split(",") if t.strip()]:
        try:
            sh.add(f"shard{int(tok.replace('shard', '')):02d}")
        except ValueError:
            return f"--ruled-out-shards: {tok!r} is not a shard index or shardNN", None
    objs = {t.strip() for t in objects.split(",") if t.strip()}
    if not sh or not objs:
        return "--ruled-out-shards / --ruled-out-objects are empty", None
    return None, {"shards": sh, "objects": objs, "evidence_uri": evidence_uri, "note": note or ""}


SUPERSEDED_RULE = ("the g01234 group's run survives — the registration whose shard group holds shard 00 (shards 00-04); the other group's registration "
                   "is retired whichever group closes first (D1816: a survivor by shard group is predictable by construction, as in s2r / s3r); a "
                   "two-group store registers once per group until the job-independent producer key makes it one run")
PENDING_FILE_RE = re.compile(r"^(\d+)\.json$")
GROUP_RE = re.compile(r"-g(\d+)-(crescendo|pair|actorattack)$")   # the multi-turn target job's shard group: f22-att-<tag>-g01234-<attack>


def shard_group_of(job_name: Optional[str]) -> Optional[str]:
    """284b: the multi-turn target job's shard-group digits (`f22-att-<tag>-g01234-crescendo` -> "01234"), None when the job name
    carries no group segment. Two registrations of one store are TWINS only when their groups differ (complementary groups of one store);
    the same group under another job id is a RETRY (since 282 the producer key is job-independent, so it is the same run id), never a twin."""
    m = GROUP_RE.search(str(job_name or ""))
    return m.group(1) if m else None


def canonical_group(job_name: Optional[str]) -> Optional[bool]:
    """True when the job's shard group holds shard 00 (the g01234 group: the survivor), False for another group, None when the job
    name carries no group segment (a single-job store: no twin rule applies)."""
    m = GROUP_RE.search(str(job_name or ""))
    if not m:
        return None
    return "0" in m.group(1)   # a job's registration; never its `.config.yaml` / `.closer_refused.json` / `.cluster.json` sidecars


def sibling_multiturn_runs(pending_dir: Optional[str], this_job: str, eval_id: str, obj: str, plan: dict, tag: Optional[str], prun: str,
                           *, repo_root: str, this_rid: str = "", this_group: Optional[str] = None) -> List[Tuple[str, str, str, str]]:
    """The OTHER jobs' registrations of the same multi-turn store on THIS cluster — the pending files beside this job's whose run names
    the same (eval, object) under the same measurement (model_id, view_legacy, item_set_id) and resolves to the same private run (the same
    store). Returns [(job id, run id, campaign, job name)]. A sibling group that ran on another cluster has no pending file here and is not found: its
    twin stays until filed by hand — the closer says so when the consolidation record names another job.
    A sibling is a registration by a DISTINCT, complementary shard group. The same run id under another
    job id (a retry of THIS group: the attack key is job-independent and carries the group) and a registration whose job name
    spells THIS group are never twins — before this, a retried g01234 attempt found the other attempt and superseded its OWN canonical run."""
    out: List[Tuple[str, str, str, str]] = []
    if not pending_dir or not os.path.isdir(pending_dir):
        return out
    want = tuple(str(plan.get(k) or "") for k in ("model_id", "view_legacy", "item_set_id"))
    for fn in sorted(os.listdir(pending_dir)):
        m = PENDING_FILE_RE.match(fn)
        if not m or m.group(1) == this_job:
            continue
        doc2 = _json(os.path.join(pending_dir, fn))
        runs2 = doc2.get("runs") if isinstance(doc2, dict) else None
        if not isinstance(runs2, list):
            continue
        job2 = str(doc2.get("job_id") or m.group(1))
        name2 = str(doc2.get("job_name") or "")
        if this_group is not None and shard_group_of(name2) == this_group:
            continue                                       # 284b: the same shard group under another job id is a RETRY of this group, not a twin
        for r2 in runs2:
            route2 = r2.get("route") or {}; plan2 = r2.get("plan") or {}
            if str(route2.get("eval_id") or plan2.get("eval_id") or "") != eval_id or str(route2.get("object") or "") != obj:
                continue
            if family_of(r2) != "multiturn" or tuple(str(plan2.get(k) or "") for k in ("model_id", "view_legacy", "item_set_id")) != want:
                continue
            why2, prun2, _src2 = multiturn_private_run(r2, tag, repo_root=repo_root)
            if why2 or str(prun2) != prun:
                continue
            rid2 = str(r2.get("run_id") or "")
            if rid2 and this_rid and rid2 == this_rid:
                continue                                   # 284b: the same run id (the job-independent key) is THIS run registered again, never its own twin
            if rid2:
                out.append((job2, rid2, str(plan2.get("campaign") or identity.parse_run_id(rid2)["campaign"]), name2 or str(plan2.get("job_name") or "")))
    return out


def file_superseded(*, spool: str, push_script: Optional[str], campaign: str, rid: str, by_rid: str, artifact: str, producer: Dict[str, object],
                    sibling_job: str, eval_id: str, obj: str, dry_run: bool, decision: str = "D1816", rule: str = SUPERSEDED_RULE,
                    kind: str = "a twin registration of one store", extra: Optional[Dict[str, object]] = None) -> None:
    """run_state superseded for a twin registration of one store, the survivor named in the note and in extra
    (the run table's superseded_by_run_id column is identity, not patchable; n_items_generated stays on both rows). The multi-turn two-group
    rule by default (the `decision` default and SUPERSEDED_RULE); the BoN store512 / climb rule passes its own decision, rule, kind and
    evidence (`extra.duplicate_of` = the matched store facts) — reason `duplicate_registration` and the record's shape are one."""
    now = utc_now()
    payload = {"run_id": rid, "state": "superseded", "event_utc": now, "evidence_kind": "completion_artifact", "evidence_uri": artifact,
               "note": (f"duplicate registration of one store by two jobs; the store's measurement is carried by {by_rid} — {rule}; "
                        f"filed by the job-side closer ({SCRIPT})")[:2000],
               "extra": {"superseded_by_run_id": by_rid, "reason": "duplicate_registration", "decision": decision, "sibling_job_id": sibling_job, **(extra or {})}}
    if dry_run:
        _say(f"DRY-RUN {eval_id} {obj} {rid[:40]}…: would file run_state superseded (by {by_rid[:40]}…; {kind}, {decision})"); return
    w = emit.RecordWriter(spool, campaign, rid, producer, push_script=push_script)
    w.write("run_state", payload)
    w.push(final=True)
    _say(f"superseded {eval_id} {obj} {rid[:40]}… by {by_rid[:40]}… ({kind}; {decision})")


# ------------------------------------------------------------ the BoN store's two registrations
BON_SUPERSEDED_DECISION = "D2599"
BON_STORE512_JOB_RE = re.compile(r"^f22-stat-.+-3$")        # the 512 store's job (stage.py: f22-stat-<tag>-3; common.sh's strongreject_bon kind)
BON_CLIMB_JOB_RE = re.compile(r"^f22-bonclimb-")            # the 4,096 climb array over the SAME store (stage.py: f22-bonclimb-<tag>, --array=0-3)
BON_DUPLICATE_FIELDS = ("store_uri", "store_completion_sha256", "n_items_generated")
BON_DUPLICATE_FACT_FIELDS = ("store_completion_sha256", "n_items_generated")   # compared once two filings cover the SAME store (store_uri, normalised)
BON_SUPERSEDED_RULE = ("the store512 registration survives — the f22-stat-<tag>-3 job's run; a f22-bonclimb-<tag> registration covering the SAME store "
                       "(same store_uri, identical store_completion_sha256 and n_items_generated) is a duplicate of it and is retired whichever "
                       "closes last (WrangleOpus D456, decision D2599); a climb whose bytes or item count differ is not a duplicate and is held by name; "
                       "a registration whose filing covers another store is no candidate at all; a registration at another rung of the store's ladder "
                       "(token_cap_tokens or pass_label differ) is never a twin (D654; the registry owner, 2026-10-13)")
BON_SUPERSEDED_KIND = "a climb registration duplicating the store512's"
#: a rerun pair's jobs carry the rerun-units slug (reproduce/lib/stage.py rerun_slug: one token of
#: [A-Za-z0-9_] ending `_rerun`, e.g. constitution_rerun) — the store job `f22-stat-<tag>-<slug>-3`, the climb `f22-bonclimb-<tag>-<slug>`; a
#: twin must spell the SAME slug (another slug is another rerun root: another store by construction, as another tag is)
BON_RERUN_SLUG_RE = re.compile(r"^[A-Za-z0-9_]+_rerun$")
#: the two registration-record fields that name a BoN run's rung of its cap ladder (the DRAWN cap, base_cap / ladder_rung)
BON_RUNG_FIELDS = ("token_cap_tokens", "pass_label")
BON_RUNG_RULE = ("a different token_cap_tokens or pass_label is another rung of the store's ladder, never a twin — a larger cap never supersedes "
                 "a smaller one (D654; the registry owner, 2026-10-13)")


def bon_rerun_slug_of(job_name: Optional[str], tag: Optional[str]) -> Optional[str]:
    """The rerun-units slug a BoN job name carries under ``tag`` — `f22-stat-<tag>-<slug>-3` or `f22-bonclimb-<tag>-<slug>` with ``slug``
    one token of [A-Za-z0-9_] ending `_rerun` (stage.rerun_slug's grammar) — else None: a plain name, another tag, a suffix that is not a rerun
    slug (never guessed)."""
    n = str(job_name or "")
    if not tag:
        return None
    for head, tail in ((f"f22-stat-{tag}-", "-3"), (f"f22-bonclimb-{tag}-", "")):
        if n.startswith(head) and n.endswith(tail) and len(n) > len(head) + len(tail):
            slug = n[len(head):len(n) - len(tail)] if tail else n[len(head):]
            if BON_RERUN_SLUG_RE.match(slug):
                return slug
    return None
BON_CLOSE_LOCK_SUFFIX = ".close.lock"                       # <private run>/.<cell>.close.lock — the exclusive flock every BoN close decides under (req. 1)
BON_CLOSE_LOCK_WAIT_S = 900.0
BON_CLOSE_LOCK_POLL_S = 0.5


def bon_role_of(job_name: Optional[str], tag: Optional[str] = None, *, rerun: Optional[str] = None) -> Optional[str]:
    """`store512` for the 512 store's job (f22-stat-<tag>-3), `climb` for the climb array (f22-bonclimb-<tag>), None for any other job
    name — a kind the rule does not cover (f22-stat-<tag>-all registers the eval too; no survivor is ruled between it and a climb). With
    ``tag``, the role only when the name spells THAT tag exactly: both kinds close the store `<wave root>/transcripts_results/sdfbase_full22_<tag>/<obj>`,
    so another tag's job registers another store, never a twin of this one. With ``rerun`` (the closing
    job's own rerun-units slug, bon_rerun_slug_of) the names are the rerun pair's — `f22-stat-<tag>-<slug>-3` / `f22-bonclimb-<tag>-<slug>` —
    and only THAT slug: a plain name beside a rerun job, or another slug, is another store by construction (the rerun's own wave root)."""
    n = str(job_name or "")
    if tag is not None:
        sfx = f"-{rerun}" if rerun else ""
        if n == f"f22-stat-{tag}{sfx}-3":
            return "store512"
        if n == f"f22-bonclimb-{tag}{sfx}":
            return "climb"
        return None
    if BON_STORE512_JOB_RE.match(n):
        return "store512"
    if BON_CLIMB_JOB_RE.match(n):
        return "climb"
    return None


def bon_same_store(mine: Dict[str, object], theirs: Dict[str, object]) -> bool:
    """Two filings cover the SAME store when their store_uri agree, compared normalised (one wave root, two spellings is
    one store). A filing over another store is not a duplicate and not a hold — it is no candidate (a climb's
    close used to read a store512 filing over another store as a `store_uri` mismatch and refuse by name)."""
    a, b = mine.get("store_uri"), theirs.get("store_uri")
    return bool(a) and bool(b) and os.path.normpath(str(a)) == os.path.normpath(str(b))


def filed_store_facts(spool: str, campaign: str, run_id: str) -> Optional[Dict[str, object]]:
    """The completion facts a closer FILED for the run — the LAST run_patch carrying store_uri: {store_uri, store_completion_artifact_uri,
    store_completion_sha256, n_items_generated}; None when no such patch sits in the spool (the run has not closed `generated` here)."""
    out: Optional[Dict[str, object]] = None
    for t, p in _records_of(spool, campaign, run_id):
        f = (p.get("fields") or {}) if t == "run_patch" else {}
        if f.get("store_uri"):
            out = {k: f.get(k) for k in ("store_uri", "store_completion_artifact_uri", *BON_DUPLICATE_FIELDS)}
    return out


def bon_rung_facts(spool: str, campaign: str, run_id: str) -> Optional[Dict[str, object]]:
    """The rung a BoN run was REGISTERED at — ``token_cap_tokens`` and ``pass_label`` (BON_RUNG_FIELDS) as the last registration record
    (`run`) in the run's spool directory states them (jobplan stamps the drawn cap and, for a rung registration, the pass label; neither is
    patchable, so no run_patch is read); an absent field is None. None when no registration record sits in this spool."""
    out: Optional[Dict[str, object]] = None
    for t, p in _records_of(spool, campaign, run_id):
        if t == "run":
            out = {k: p.get(k) for k in BON_RUNG_FIELDS}
    return out


def bon_rung_mismatch(mine: Optional[Dict[str, object]], theirs: Optional[Dict[str, object]]) -> Optional[str]:
    """The rung guard: None when two registrations name the SAME rung — equal ``token_cap_tokens`` and ``pass_label`` (an integer-looking cap
    compared as an integer; two absent halves agree, as the pre-ladder registrations of one store agree) — else the differing fields by name.
    A side whose registration record is not on record cannot be compared: named as such, never a twin."""
    if mine is None or theirs is None:
        return "the rung facts are not on record (no registration record in this spool) for " + ("this run" if mine is None else "the twin")
    diffs: List[str] = []
    for k in BON_RUNG_FIELDS:
        a, b = mine.get(k), theirs.get(k)
        if k == "token_cap_tokens":
            try:
                a, b = int(a), int(b)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                pass
        if a != b:
            diffs.append(f"{k} {a} vs {b}")
    return "; ".join(diffs) if diffs else None


def bon_duplicate_mismatch(mine: Dict[str, object], theirs: Dict[str, object]) -> Optional[str]:
    """None when two filings OF ONE STORE (bon_same_store, the caller's check) are duplicates (identical store_completion_sha256 and
    n_items_generated), else the differing fields by name — the bytes-or-count hold. The sha and the count are compared exactly."""
    diffs: List[str] = []
    for k in BON_DUPLICATE_FACT_FIELDS:
        a, b = mine.get(k), theirs.get(k)
        if k == "n_items_generated":
            try:
                a, b = int(a), int(b)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                pass
        if a != b:
            fa = str(a)[:12] + "…" if k == "store_completion_sha256" and a else str(a)
            fb = str(b)[:12] + "…" if k == "store_completion_sha256" and b else str(b)
            diffs.append(f"{k} {fa} vs {fb}")
    return "; ".join(diffs) if diffs else None


def pending_docs_beside(pending_dir: Optional[str], this_job: str) -> List[Tuple[str, str, list]]:
    """Every OTHER job's pending file beside this one, read once per close: [(job id, job name, runs)]."""
    out: List[Tuple[str, str, list]] = []
    if not pending_dir or not os.path.isdir(pending_dir):
        return out
    for fn in sorted(os.listdir(pending_dir)):
        m = PENDING_FILE_RE.match(fn)
        if not m or m.group(1) == this_job:
            continue
        doc2 = _json(os.path.join(pending_dir, fn))
        runs2 = doc2.get("runs") if isinstance(doc2, dict) else None
        if not isinstance(runs2, list):
            continue
        out.append((str(doc2.get("job_id") or m.group(1)), str(doc2.get("job_name") or ""), runs2))
    return out


def sibling_bon_runs(docs: Sequence[Tuple[str, str, list]], eval_id: str, obj: str, plan: dict, *, this_rid: str, tag: str, campaign: str,
                     rerun: Optional[str] = None
                     ) -> Tuple[List[Tuple[str, str, str, str]], List[Tuple[str, str, str, str]], List[Tuple[str, str, str, str]]]:
    """The OTHER jobs' registrations of the same BoN store on THIS cluster — the runs in the pending files beside this job's naming the
    same (eval, object) under the same measurement (model_id, view_legacy, item_set_id) and campaign in the bon family. Three lists of
    (job id, run id, campaign, job name): the twins by a job of a role the rule covers (bon_role_of under THIS tag and THIS job's rerun slug:
    f22-stat-<tag>[-<slug>]-3 / f22-bonclimb-<tag>[-<slug>], the two kinds that close this tag's store directory), the registrations
    by a job of a covered kind under ANOTHER tag or another slug (another store by construction: no candidate; the caller says so once), and
    the registrations by any other kind (no rule applies; the caller names them). The same run id under another job id — the climb array's
    other tasks since the job-independent key, a re-attempt — is THIS run registered again, never a twin."""
    want = tuple(str(plan.get(k) or "") for k in ("model_id", "view_legacy", "item_set_id"))
    twins: List[Tuple[str, str, str, str]] = []; other_store: List[Tuple[str, str, str, str]] = []; unruled: List[Tuple[str, str, str, str]] = []
    for job2, name2, runs2 in docs:
        for r2 in runs2:
            route2 = r2.get("route") or {}; plan2 = r2.get("plan") or {}
            if str(route2.get("eval_id") or plan2.get("eval_id") or "") != eval_id or str(route2.get("object") or "") != obj:
                continue
            if family_of(r2) != "bon" or tuple(str(plan2.get(k) or "") for k in ("model_id", "view_legacy", "item_set_id")) != want:
                continue
            rid2 = str(r2.get("run_id") or "")
            if not rid2 or rid2 == this_rid:
                continue
            campaign2 = str(plan2.get("campaign") or identity.parse_run_id(rid2)["campaign"])
            if campaign2 != campaign:
                continue                                       # another campaign's run: another spool, another registry row, never this store's twin
            entry = (job2, rid2, campaign2, name2 or str(plan2.get("job_name") or ""))
            if bon_role_of(entry[3], tag, rerun=rerun):
                twins.append(entry)
            elif bon_role_of(entry[3]):
                other_store.append(entry)
            else:
                unruled.append(entry)
    return twins, other_store, unruled


class StoreCloseLockBusy(RuntimeError):
    """The store's close lock could not be taken (held past the wait, or no lock service on this filesystem): the run is REFUSED by name and
    stays as it is — the reclose takes the decision later. Never a decision without the lock."""


def bon_close_lock_path(store: str) -> str:
    """The store's close lock: ``<private run>/.<cell>.close.lock`` — BESIDE the store directory, never inside it (a lifecycle writer for a
    store that did not exist yet had no lock to take, and `_run_lock` decided its `running` unlocked on the strength of the directory's
    absence — read once, written seconds later, with the state emitter's hardware capture in between — while a store created, completed and
    closed in that interval could land its `generated` / `superseded` first and be reversed). A sibling path exists before the store does, its
    directory is the lock taker's to create, and it is a pure function of the store path (normalised, as bon_same_store compares store_uri), so
    every close and every lifecycle writer of a BoN run — the closes over the inspected path or a FILED store_uri, an array task's `started` /
    `failed` / `cancelled`, the array's reconciliation, the refusal note — take ONE lock whether or not the store exists. A dotfile beside the
    cell directories: the store's own readers list `shard*` and cell directories, never a dotfile; the climb planner's cell stays ABSENT
    (bon_climb.store_state reads the store directory alone)."""
    p = os.path.normpath(str(store))
    return os.path.join(os.path.dirname(p), "." + os.path.basename(p) + BON_CLOSE_LOCK_SUFFIX)


class StoreCloseLock:
    """The store512 and climb closers are separate jobs and can close in the same second, each seeing no peer. Every decision about a BoN
    store's duplicate registrations — reading the twins' filings, filing `generated`, filing a `superseded` — and every lifecycle event a task
    files for its run is taken under an exclusive ``flock`` on the store's close lock, :func:`bon_close_lock_path` (``<private
    run>/.<cell>.close.lock``, beside the store so it can be taken before the store exists; the PublishLock shape of sisteer.registry.stores:
    VAST is nfs4.1 with server-side locks, so it holds across nodes; the kernel releases it when the holder dies). The lock's directory is
    created on entry when it is missing (the private run directory, before any store of it); a directory that cannot be created is a lock that
    cannot be opened. A waiter blocks, polling, up to ``wait_s`` (saying once who holds it), then :class:`StoreCloseLockBusy`. The lock file
    names the holder's job id (or pid) and the time; never a host or a user."""

    def __init__(self, store: str, wait_s: float = BON_CLOSE_LOCK_WAIT_S, poll_s: float = BON_CLOSE_LOCK_POLL_S, *, lock_path: Optional[str] = None):
        # `lock_path`: the same exclusive flock over another file — the array's outcome directory lock (array_task_event), one holder shape
        self.path = lock_path or bon_close_lock_path(store)
        self.wait_s, self.poll_s = float(wait_s), float(poll_s)
        self.fh = None
        self.waited_for: Optional[str] = None

    def _holder_line(self) -> str:
        try:
            with open(self.path, "r", encoding="utf-8") as fh:
                return fh.read().strip()[:200]
        except OSError:
            return ""

    def __enter__(self) -> "StoreCloseLock":
        try:
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)   # the lock's directory, before the store's: a writer may be first here
            fh = open(self.path, "a+")
        except OSError as exc:
            raise StoreCloseLockBusy(f"the store close lock {self.path} cannot be opened ({type(exc).__name__}: {exc})")
        t0 = time.monotonic()
        while True:
            try:
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                pass
            except OSError as exc:                   # ENOLCK and friends: no lock service here — never decide unlocked
                fh.close()
                raise StoreCloseLockBusy(f"the store close lock {self.path} cannot be taken ({type(exc).__name__}: {exc})")
            waited = time.monotonic() - t0
            if self.waited_for is None:
                self.waited_for = self._holder_line() or "another closer"
                _say(f"waiting for the store close lock {self.path} held by {self.waited_for} (up to {self.wait_s:.0f} s)")
            if waited >= self.wait_s:
                fh.close()
                raise StoreCloseLockBusy(f"the store close lock {self.path} is held by {self.waited_for} after {waited:.0f} s")
            time.sleep(max(0.01, min(self.poll_s, self.wait_s - waited)))
        self.fh = fh
        fh.seek(0); fh.truncate()
        fh.write(f"job {os.environ.get('SLURM_JOB_ID') or 'pid ' + str(os.getpid())} {utc_now()}\n"); fh.flush()
        return self

    def __exit__(self, *exc) -> None:
        if self.fh is not None:
            try:
                fcntl.flock(self.fh.fileno(), fcntl.LOCK_UN)
            except OSError:
                pass
            self.fh.close(); self.fh = None


def bon_close_lock(store: str, dry_run: bool):
    """The store's close lock, or nothing under --dry-run (a dry run writes nothing, so there is nothing to serialise)."""
    return contextlib.nullcontext() if dry_run else StoreCloseLock(store)


def bon_settle_duplicates(*, role: str, rid: str, campaign: str, mine: Dict[str, object], twins: List[Tuple[str, str, str, str]], spool: str,
                          push_script: Optional[str], producer: Dict[str, object], this_job: str, eval_id: str, obj: str, dry_run: bool,
                          out: Dict[str, int], file_generated: Optional[Callable[[], None]] = None,
                          file_store_facts: Optional[Callable[[], None]] = None) -> Tuple[Optional[str], bool]:
    """UNDER THE STORE LOCK (the caller's): the duplicate-registration rule over THIS run's completion facts ``mine`` (its filing, or the one
    it is about to make) and its twins' FILED facts. ``file_generated`` files this run's `generated` when it is closing now (None: already
    generated — the re-evaluation); ``file_store_facts`` files the run_patch carrying ``mine`` — the inspected store facts, no generated_utc —
    for a climb closing now that this call retires instead (a climb the store512 closed before never reached `generated`, so no run_patch
    carried its store to the registry row, and admission's producer_supersession, which verifies the record's claim by the ROWS — store_uri
    recorded on both and equal — refused it as having an unrecorded store; the facts are filed under this lock BEFORE the `superseded`, so the
    row records the store the record claims). A twin whose filing covers ANOTHER store (bon_same_store: the store_uri differ) is no candidate
    — noted, never held, never refused; the bytes-or-count rule applies to the filings of THIS store alone.
    Returns (refusal, superseded_self):
      store512 — this run's `generated` is filed first (the survivor stands before its twins are retired); then every climb twin whose filed
                 facts equal ``mine`` is filed superseded by this run (one already retired is left), a twin whose facts differ is HELD by name
                 (out['held']; no supersede; both stay current until a human rules), a twin with no filing yet is left to its own close.
      climb    — a store512 twin's filing covers this store with ``mine``'s facts: this run is filed superseded by it, nothing else
                 -> (None, True) — among several such filings (job-keyed store512 registrations of one store, from the earlier key) the one
                 whose facts match, the earliest job first when several match; every filing of this store differs -> (the refusal, False),
                 nothing filed; no store512 filing of this store yet -> this run closes `generated` (the store512 retires it when it closes
                 last) -> (None, False); no store512 registration beside it -> `generated` as for any lone registration.
    The climb array's tasks share one run id, so THIS run's `superseded` is re-read here, under the lock, before it is filed: a peer
    close that filed it first leaves nothing to file -> (None, True) with nothing written."""
    if role == "store512":
        if file_generated is not None:
            file_generated()
        for t_job, t_rid, t_campaign, t_name in twins:
            if bon_role_of(t_name) != "climb":
                continue
            theirs = filed_store_facts(spool, t_campaign, t_rid)
            if theirs is None:
                _say(f"note {eval_id} {obj} {t_rid[:40]}…: the climb registration by job {t_job} has not closed `generated` — it supersedes itself at its own "
                     f"close against this filing (D456)")
                continue
            if already_generated(spool, t_campaign, t_rid, states=("cancelled", "superseded")):
                continue
            if not bon_same_store(mine, theirs):
                _say(f"note {eval_id} {obj} {t_rid[:40]}… (climb job {t_job}): its filing covers another store ({theirs.get('store_uri')}) — not a duplicate of this "
                     f"store512 registration ({mine.get('store_uri')}; D456: same store_uri), no candidate; left as it is")
                continue
            rung = bon_rung_mismatch(bon_rung_facts(spool, campaign, rid), bon_rung_facts(spool, t_campaign, t_rid))
            if rung:
                # the climb at another rung of this store's ladder (the smoke's 512 store beside its 4,096 climb, identical store facts)
                # is the ladder's design, not a duplicate — never held, never superseded, whatever the bytes and the count say
                _say(f"note {eval_id} {obj} {t_rid[:40]}… (climb job {t_job}): another rung of this store's ladder — {rung} — not a duplicate of this "
                     f"store512 registration ({BON_RUNG_RULE}), no candidate; left as it is")
                continue
            diff = bon_duplicate_mismatch(mine, theirs)
            if diff:
                _say(f"HELD {eval_id} {obj} {t_rid[:40]}… (climb job {t_job}): not a duplicate of this store512 registration — {diff} (D456: bytes or item count "
                     f"differ); no supersede filed, both registrations stay current until a human rules")
                out["held"] += 1
                continue
            file_superseded(spool=spool, push_script=push_script, campaign=t_campaign, rid=t_rid, by_rid=rid, artifact=str(mine.get("store_completion_artifact_uri") or ""),
                            producer=producer, sibling_job=this_job, eval_id=eval_id, obj=obj, dry_run=dry_run, decision=BON_SUPERSEDED_DECISION,
                            rule=BON_SUPERSEDED_RULE, kind=BON_SUPERSEDED_KIND,
                            extra={"duplicate_of": {k: mine.get(k) for k in BON_DUPLICATE_FIELDS}, "superseded_role": "climb", "survivor_role": "store512"})
            out["superseded_bon"] += 1
        return None, False
    # a climb registration: the survivor is the store512 run — judged against its FILING, never by construction
    s512 = [(t, filed_store_facts(spool, t[2], t[1])) for t in twins if bon_role_of(t[3]) == "store512"]
    if not s512:
        if file_generated is not None:
            file_generated()
        return None, False
    filed = [(t, f) for t, f in s512 if f is not None]
    same: List[Tuple[Tuple[str, str, str, str], Dict[str, object]]] = []
    my_rung = bon_rung_facts(spool, campaign, rid)
    for t, f in filed:
        rung = bon_rung_mismatch(my_rung, bon_rung_facts(spool, t[2], t[1])) if bon_same_store(mine, f) else None
        if not bon_same_store(mine, f):
            _say(f"note {eval_id} {obj} {rid[:40]}…: the store512 registration {t[1][:40]}… (job {t[0]}) filed another store ({f.get('store_uri')}) — not a "
                 f"duplicate of this one ({mine.get('store_uri')}; D456: same store_uri), no candidate")
        elif rung:
            # a store512 registration at another rung of this store's ladder is the ladder's design, not this climb's survivor
            _say(f"note {eval_id} {obj} {rid[:40]}…: the store512 registration {t[1][:40]}… (job {t[0]}) of this store is another rung of its ladder — {rung} — "
                 f"not a duplicate of this one ({BON_RUNG_RULE}), no candidate")
        elif already_generated(spool, t[2], t[1], states=("cancelled", "superseded")):
            _say(f"note {eval_id} {obj} {rid[:40]}…: the store512 registration {t[1][:40]}… (job {t[0]}) of this store is itself retired — not a survivor")
        else:
            same.append((t, f))
    if not same:
        unfiled = [t for t, f in s512 if f is None]
        tail = f"this climb run {'closes generated' if file_generated is not None else 'stays generated'}; the store512 retires it when it closes last (D456)"
        if unfiled:
            _say(f"note {eval_id} {obj} {rid[:40]}…: the store512 registration {unfiled[0][1][:40]}… (job {unfiled[0][0]}) has not closed `generated`"
                 + (" over this store" if filed else "") + f" — {tail}")
        else:
            _say(f"note {eval_id} {obj} {rid[:40]}…: no store512 filing covers this store (the {len(filed)} filing(s) beside this run cover another store or are retired) — {tail}")
        if file_generated is not None:
            file_generated()
        return None, False
    same.sort(key=lambda tf: tf[0][0])                         # the earliest job first: deterministic when several store512 filings of one store match
    exact = [(t, f) for t, f in same if bon_duplicate_mismatch(mine, f) is None]
    if not exact:
        (t_job, t_rid, _t_campaign, _t_name), theirs = same[0]
        more = f"; and {len(same) - 1} more store512 filing(s) of this store, none a duplicate" if len(same) > 1 else ""
        return f"not a duplicate of the store512 registration {t_rid[:40]}… (job {t_job}): {bon_duplicate_mismatch(mine, theirs)}{more}", False
    (t_job, t_rid, _t_campaign, _t_name), theirs = exact[0]
    if len(exact) > 1:
        _say(f"note {eval_id} {obj} {rid[:40]}…: {len(exact)} store512 registrations of this store filed identical facts (jobs {', '.join(t[0] for t, _f in exact)}); "
             f"D456 ranks none of them against each other — superseded by the earliest, job {t_job}")
    if already_generated(spool, campaign, rid, states=("superseded",)):
        _say(f"skip {eval_id} {obj} {rid[:40]}… already superseded under the lock by another close of this run ({BON_SUPERSEDED_KIND}; {BON_SUPERSEDED_DECISION}) — nothing filed twice")
        return None, True
    if file_store_facts is not None:
        file_store_facts()                                     # the inspected store facts reach the row FIRST, under this lock; then the retirement
    file_superseded(spool=spool, push_script=push_script, campaign=campaign, rid=rid, by_rid=t_rid, artifact=str(mine.get("store_completion_artifact_uri") or ""),
                    producer=producer, sibling_job=t_job, eval_id=eval_id, obj=obj, dry_run=dry_run, decision=BON_SUPERSEDED_DECISION, rule=BON_SUPERSEDED_RULE,
                    kind=BON_SUPERSEDED_KIND, extra={"duplicate_of": {k: theirs.get(k) for k in BON_DUPLICATE_FIELDS}, "superseded_role": "climb", "survivor_role": "store512"})
    out["superseded_bon"] += 1
    return None, True


def close_job(pending_path: str, *, spool: str, push_script: Optional[str], code_sha: Optional[str], wave_root: Optional[str], tag: Optional[str],
              job_name: Optional[str], trial: Optional[int], nshards: int, private_root: Optional[str], private_run: Optional[str],
              evals: Optional[List[str]] = None, dry_run: bool = False, misalign_root: Optional[str] = None, pad_root: Optional[str] = None,
              expected_goal_ids: Optional[str] = None, ruled_out: Optional[dict] = None) -> Dict[str, int]:
    _bind_pending_dir(pending_path)
    with open(pending_path, encoding="utf-8") as fh:
        doc = json.load(fh)
    runs = doc.get("runs") if isinstance(doc, dict) else None
    if not isinstance(runs, list) or not runs:
        _say(f"REFUSED: {pending_path} names no `runs` (the job registered nothing)"); return {"filed": 0, "skipped": 0, "refused": 1, "registered": 0, "other_emitter": 0}
    job_name = job_name or str(doc.get("job_name") or "")
    producer = emit.producer_block(SCRIPT, code_sha=code_sha)
    out = {"filed": 0, "skipped": 0, "refused": 0, "registered": len(runs), "other_emitter": 0, "cancelled": 0, "superseded": 0, "superseded_bon": 0, "held": 0}
    refused_ids: List[str] = []        # the FULL ids of the runs this close refused — the exit path states `running` for these alone
    pending_docs: Optional[List[Tuple[str, str, list]]] = None    # The other jobs' pending files beside this one, read once, on the first BoN run
    this_job = str(doc.get("job_id") or os.path.basename(pending_path)[:-5])
    this_canon = canonical_group(str(doc.get("job_name") or ""))     # True = the g01234 group (survives), False = another group, None = no group
    this_group = shard_group_of(str(doc.get("job_name") or ""))     # 284b: the group itself — a sibling must be a DISTINCT group (a retry is the same run)
    # demo-import M10: the SMOKE goal-shard knob the job's overrides document sets ($SISTEER_RUN_OVERRIDES — the document the registration
    # stamped on every run of this job through emit.RecordWriter, read here through the same reader); None = the paper's plan. It narrows the
    # multi-turn targets and the padding levels (SG.NARROWED_FAMILIES) to shards lo..hi of N — exactly what the planner emitted and froze —
    # and touches no other family's close. A document that names the knob but cannot be honoured refuses those families' runs by name.
    smoke: Optional[SG.SmokeGoals] = None; smoke_why: Optional[str] = None
    try:
        smoke = SG.from_env()
    except SG.SmokeGoalsRefused as exc:
        smoke_why = str(exc)
    if smoke is not None:
        _say(f"smoke goal-shard knob: {SG.KEY}={smoke.spec} (overrides {smoke.override_id}) — the multi-turn targets and the padding levels close over "
             f"shards {', '.join(smoke.shards)} of {smoke.nshards}; every other family closes whole")
    for r in runs:
        cancel = None; twins: List[Tuple[str, str, str, str]] = []
        rid = str(r.get("run_id") or ""); route = r.get("route") or {}; plan = r.get("plan") or {}
        eval_id = str(route.get("eval_id") or plan.get("eval_id") or ""); obj = str(route.get("object") or "")
        if not rid or not eval_id or not obj:
            _say(f"REFUSED: a pending entry lacks run_id / route.eval_id / route.object: {json.dumps(r)[:160]}"); out["refused"] += 1; continue
        if evals and eval_id not in evals:
            out["skipped"] += 1; continue
        campaign = str(plan.get("campaign") or identity.parse_run_id(rid)["campaign"])
        planned, iset, goals, _variants = planned_of(r)
        fam = "static" if private_root else family_of(r)
        if fam is None and other_emitter_of(eval_id):
            # closer-fix-2: not a refusal — another emitter files this eval's runs; the run stays as that emitter leaves it
            _say(f"skip {eval_id} {obj} {rid[:40]}…: not this closer's layout — {other_emitter_of(eval_id)} files it; the run stays as its emitter leaves it")
            out["other_emitter"] += 1; continue
        if already_generated(spool, campaign, rid, states=("cancelled",)):
            _say(f"skip {eval_id} {obj} {rid[:40]}… already cancelled (shards ruled out; D1168)"); out["skipped"] += 1; continue
        if already_generated(spool, campaign, rid, states=("superseded",)):
            _say(f"skip {eval_id} {obj} {rid[:40]}… already superseded ({BON_SUPERSEDED_KIND + '; ' + BON_SUPERSEDED_DECISION if fam == 'bon' else 'the other shard group' + chr(39) + 's registration of one store; D1816'})")
            out["skipped"] += 1; continue
        was_generated = already_generated(spool, campaign, rid)
        # a BoN store's OTHER registrations beside this pending file — read for every bon run of a job kind the rule covers (the store512
        # job, the climb array), an already-generated one included: the rule is RE-EVALUATED on every close (the idempotent half), so a
        # reclose or a later sweep files the supersede a same-second race, or an older tree, left missing
        bon_role = bon_role_of(job_name) if fam == "bon" else None
        bon_twins: List[Tuple[str, str, str, str]] = []
        if bon_role and wave_root and tag:
            if pending_docs is None:
                pending_docs = pending_docs_beside(PENDING_DIR, this_job)
            bon_twins, bon_other_store, bon_unruled = sibling_bon_runs(pending_docs, eval_id, obj, plan, this_rid=rid, tag=tag, campaign=campaign,
                                                                       rerun=bon_rerun_slug_of(job_name, tag))      # This job's rerun slug, if any
            if bon_other_store:
                _say(f"note {eval_id} {obj} {rid[:40]}…: {len(bon_other_store)} registration(s) of this measurement by job(s) {', '.join(sorted(t[0] for t in bon_other_store))} "
                     f"({', '.join(sorted(set(t[3] for t in bon_other_store)))}) name another tag or another rerun slug — another store under this wave root, "
                     f"never a twin of this one (D456: same store_uri; D471)")
            for u_job, u_rid, _u_campaign, u_name in bon_unruled:
                _say(f"note {eval_id} {obj} {rid[:40]}…: registration {u_rid[:40]}… of this store by job {u_job} ({u_name!r}) is neither the store512 job's nor the "
                     f"climb array's — no survivor is ruled between them (D456 covers f22-stat-<tag>-3 vs f22-bonclimb-<tag>); file by hand")
        if bon_twins and was_generated:
            # the re-evaluation: this run's FILED facts against its twins' filings, under the store lock; nothing is inspected again
            mine = filed_store_facts(spool, campaign, rid)
            if mine is None:
                _say(f"note {eval_id} {obj} {rid[:40]}…: generated, but no run_patch with the store facts sits in this spool — its twins cannot be judged "
                     f"duplicates (D456 compares store_uri, store_completion_sha256 and n_items_generated); file by hand")
            else:
                try:
                    with bon_close_lock(str(mine["store_uri"]), dry_run):
                        why_b, self_superseded = bon_settle_duplicates(role=bon_role, rid=rid, campaign=campaign, mine=mine, twins=bon_twins, spool=spool,
                                                                       push_script=push_script, producer=producer, this_job=this_job, eval_id=eval_id,
                                                                       obj=obj, dry_run=dry_run, out=out)
                except StoreCloseLockBusy as exc:
                    _say(f"REFUSED {eval_id} {obj} {rid[:40]}…: {exc} — the run stays generated; its twins are judged at the next close"); out["refused"] += 1; continue
                if why_b:
                    _say(f"HELD {eval_id} {obj} {rid[:40]}…: {why_b} (D456: bytes or item count differ) — the run stays generated, no supersede filed; a human rules")
                    out["held"] += 1
                if self_superseded:
                    continue                                   # this climb's `generated` stands corrected by the supersede just filed; nothing more here
        # a multi-turn run that is already generated still goes through store inspection and sibling discovery — the branch below that retires
        # its twins (an already-generated g01234 run retires a g5678 `generated` an older closer left) was unreachable from here; it files no
        # second `generated` (the already-generated branch skips the filing)
        if was_generated and fam != "multiturn" and not (fam == "bon" and wave_root and tag and planned and not already_certified(spool, campaign, rid)):
            _say(f"skip {eval_id} {obj} {rid[:40]}… already generated"); out["skipped"] += 1; continue
        if not planned and fam != "misalign":
            _say(f"REFUSED {eval_id} {obj} {rid[:40]}…: the registration names no n_items_planned — nothing to judge completeness against"); out["refused"] += 1; refused_ids.append(rid); continue
        why = None; facts: Dict[str, object] = {}; store = ""; artifact = ""; n_items = 0
        if fam == "static":
            pk = str((r.get("plan") or {}).get("producer_run_key_legacy") or ""); m_pk = STATIC_RUN_KEY_RE.match(pk)
            shards = nshards; rung_cap = None; prun_s = None
            if private_root and private_run:
                store = os.path.join(private_root, private_run, eval_id, obj)
            elif m_pk and wave_root:
                # static-reg-fix: the run's key IS its private run (one run per rung) — the store is located from the run's own identity,
                # never from a hard-coded base spelling + the job name's _t<N>; a RUNG store is one shard (escalate_static: shard00, every row
                # of the pair re-drawn or copied forward, so the plan's count still holds)
                prun_s = m_pk.group(0); rung_cap = int(m_pk.group("cap")) if m_pk.group("cap") else None
                store = os.path.join(wave_root, "transcripts_results", prun_s, eval_id, obj)
                if rung_cap is not None:
                    shards = 1
            else:
                t = trial_of(job_name, trial)
                if not wave_root or not tag or t is None:
                    why = "static layout needs --wave-root, --tag and a trial (--trial or a job name carrying _t<N>)"
                else:
                    store = os.path.join(wave_root, "transcripts_results", f"sdfbase_full22_{tag}_sampled_t{t}", eval_id, obj)
            if not why:
                why, facts, artifact = inspect_static(store, planned=planned, goals=goals, nshards=shards); n_items = planned
                if prun_s:
                    facts.update({"private_run": prun_s, "rung_cap": rung_cap, "located_by": "producer_run_key"})
        elif fam == "bon":
            if not wave_root or not tag:
                why = "bon layout needs --wave-root and --tag"
            else:
                store = bon_store_path(wave_root, tag, obj)
                why, facts, artifact = inspect_bon(store, planned_goals=planned, repo_root=REPO_ROOT); n_items = int(((facts.get("completion") or {}).get("n_done")) or 0)
        elif fam == "controls":
            t = trial_of(job_name, trial)
            if not wave_root or not tag or t is None:
                why = "controls layout needs --wave-root, --tag and a trial (--trial or a job name carrying _t<N>)"
            else:
                why, facts, store, artifact = inspect_controls(wave_root, tag, t, obj, eval_id, planned=planned); n_items = planned
        elif fam == "padding":
            pad = pad_level_of(r)
            why_p, prun = pad_private_run(r, repo_root=REPO_ROOT)
            if not wave_root or pad is None:
                why = "padding layout needs --wave-root and a registration naming the pad level (extra.padding.tokens / job name pad(stat|obj)-…-p<pad>)"
            elif why_p:
                why = why_p
            elif smoke_why:
                why = f"the run-overrides document's smoke goal-shard knob cannot be honoured: {smoke_why}"
            else:
                root = pad_root or os.path.join(wave_root, "static_padding", "transcripts_results")
                store = os.path.join(root, prun, eval_id, obj)
                if smoke is None:
                    why, facts, artifact = inspect_static(store, planned=planned, goals=goals, nshards=nshards); n_items = planned
                else:
                    # M10: this level ran shards lo..hi of N (target_static_padded_obj.sbatch `shards=<lo>-<hi>/N`, the planner's positional); the
                    # registration plans the WHOLE item set (`…_<g>x<v>`), so the store is held to the subset's goals (goals[s::N] over lo..hi) and
                    # their responses (goals x variants) — the same dealing the frozen inventory's coverage.expected was composed with
                    why = (SG.nshards_disagreement(nshards, smoke) or SG.key_disagreement(plan.get("producer_run_key_legacy"), smoke, "padding"))
                    if not why and (goals is None or _variants is None or goals * _variants != planned):
                        why = (f"the registration's item set {iset!r} does not spell `<goals>x<variants>` matching n_items_planned {planned} — the smoke "
                               f"subset's planned responses cannot be derived")
                    if not why:
                        sub_goals = smoke.n_goals(goals); sub_planned = sub_goals * _variants
                        why, facts, artifact = inspect_static(store, planned=sub_planned, goals=sub_goals, nshards=nshards, shard_names=smoke.shards)
                        n_items = sub_planned
                        facts["smoke_goals"] = {**smoke.record(), "goals_expected": sub_goals, "responses_expected": sub_planned,
                                                "item_set_goals": goals, "item_set_variants": _variants}
                facts.update({"layout": "padding", "padding_tokens": pad, "private_run": prun})
        elif fam == "multiturn":
            why_m, prun, prun_src = multiturn_private_run(r, tag, repo_root=REPO_ROOT)
            why_g, exp_ids, exp_prov = multiturn_expected_goal_ids(r, repo_root=REPO_ROOT, ids_file=expected_goal_ids)
            if not wave_root:
                why = "multiturn layout needs --wave-root (the attacks wave root; its transcripts_results holds the private run)"
            elif why_m:
                why = why_m
            elif why_g:
                why = why_g
            elif smoke_why:
                why = f"the run-overrides document's smoke goal-shard knob cannot be honoured: {smoke_why}"
            else:
                store = os.path.join(wave_root, "transcripts_results", str(prun), eval_id, obj)
                # when the ruling names this object, its ruled-out shards may be ABSENT — every present shard is still
                # inspected in full (its own goal assignment, graded outcomes, responses, no torn lines); an incomplete present shard, or
                # shards merely not yet run, keep "stays running"
                tolerated = sorted(ruled_out["shards"]) if (ruled_out and obj in ruled_out["objects"] and not was_generated) else None
                # M10: under the smoke goal-shard knob this target ran shards lo..hi of N (target_attacks.sbatch `[nshards]=N [shards]=<lo..hi>`, the
                # planner's positionals; the producer key bakes `sh<shards>of<N>`): the store is held to exactly those shards, each to its deal
                # goals[s::N], and `generated` is filed over the subset's goals — the frozen inventory's coverage.expected, by construction
                why = ((SG.nshards_disagreement(nshards, smoke) or SG.key_disagreement(plan.get("producer_run_key_legacy"), smoke, "multiturn"))
                       if smoke is not None else None)
                if not why:
                    why, facts, artifact = inspect_multiturn(store, planned=planned, nshards=nshards, expected_ids=exp_ids, tolerated_missing=tolerated,
                                                             subset=smoke)
                    n_items = int(facts.get("goals_expected") or 0) if (smoke is not None and not why) else planned
                facts.update({"private_run": prun, "private_run_source": prun_src, "goal_set": {k: (v[:12] if k == "config_sha256" else v) for k, v in exp_prov.items() if k != "file"}})
                part = facts.get("partial") if not why else None
                if not why and part is None:                   # The whole store — the other group's registration(s) of it, on this cluster
                    twins = sibling_multiturn_runs(PENDING_DIR, this_job, eval_id, obj, plan, tag, str(prun), repo_root=REPO_ROOT, this_rid=rid, this_group=this_group)
                    cons_jobs = {str(src.get("job_id")) for src in ((facts.get("consolidation") or {}).get("shards") or {}).values() if isinstance(src, dict)} \
                        if isinstance((facts.get("consolidation") or {}).get("shards"), dict) else set()
                    if not twins and cons_jobs - {this_job, "None", ""}:
                        _say(f"note {eval_id} {obj} {rid[:40]}…: the store's shards were written by job(s) {', '.join(sorted(cons_jobs - {this_job}))} too, but no "
                             f"registration of theirs sits beside this pending file — a twin on another cluster stays until filed by hand (D1521)")
                if part:                                       # the ruled-out shards are missing and the present ones passed: closes `cancelled`
                    present = list(part["shards_closed"]); missing = list(part["shards_missing_ruled_out"])
                    counts = {"goals": int(facts.get("goals_distinct") or 0), "responses": int(facts.get("responses") or 0)}
                    cancel = {"store": store, "present": present, "missing": missing, "counts": counts,
                              "note": (f"shards {', '.join(missing)} ruled out by design ({ruled_out['evidence_uri']}); closed with shards "
                                       f"{', '.join(present)} present: {counts['goals']} goal(s), {counts['responses']} response(s)"
                                       + (f"; {ruled_out['note']}" if ruled_out.get("note") else ""))}
        elif fam == "misalign":
            root = misalign_root or (os.path.join(os.path.dirname(wave_root.rstrip("/")), "misalign", tag) if wave_root and tag else None)
            if not root:
                why = "misalign layout needs --misalign-root, or --wave-root + --tag (default <shared exp root>/misalign/<tag>)"
            else:
                why, facts, store, artifact = inspect_misalign(root, obj, eval_id, planned=planned); n_items = int(facts.get("behaviour_rows") or 0)
                planned = planned or n_items
        else:
            why = f"no store layout here for eval {eval_id!r} (instrument {plan.get('instrument_id') or (plan.get('extra') or {}).get('seed_instrument_id')!r}): another emitter's family, or one this closer does not know — left running"
        if why:
            _say(f"REFUSED {eval_id} {obj} {rid[:40]}…: {why} — the run stays running"); out["refused"] += 1; refused_ids.append(rid); continue
        if cancel is not None:
            now = utc_now()
            fields = {"store_uri": cancel["store"], "n_items_generated": int(cancel["counts"]["goals"])}
            if dry_run:
                _say(f"DRY-RUN {eval_id} {obj} {rid[:40]}…: would file run_patch {json.dumps(fields, sort_keys=True)} + run_state cancelled (ruling {ruled_out['evidence_uri']}): {cancel['note']}")
            else:
                w = emit.RecordWriter(spool, campaign, rid, producer, push_script=push_script)
                w.emit_patch(fields=fields)
                w.emit_state("cancelled", "ruling", ruled_out["evidence_uri"], event_utc=now,
                             note=(f"filed by the job-side closer ({SCRIPT}) against the registration's plan ({planned} planned): {cancel['note']}; "
                                   f"{json.dumps({k: v for k, v in facts.items() if k in ('layout', 'partial', 'goals_per_shard', 'outcomes', 'private_run', 'consolidation')}, sort_keys=True)}")[:2000])
                w.push(final=True)
                _say(f"cancelled {eval_id} {obj} {rid[:40]}… ({fam}) store_uri={cancel['store']} n_items={cancel['counts']['goals']}/{planned}: {cancel['note'][:160]}")
            out["cancelled"] += 1; continue
        cert = None
        if fam == "bon":
            planned_keys = facts.pop("_planned_keys", [])
            why_c, cert = bon_nofit_certificate(facts, store, planned_keys)
            if why_c:
                _say(f"REFUSED {eval_id} {obj} {rid[:40]}…: {why_c} — the run stays {'generated' if was_generated else 'running'}"); out["refused"] += 1
                if not was_generated:
                    refused_ids.append(rid)        # the exit path states `running` for the runs left RUNNING alone — a generated run is not listed
                continue
        # round-3 Major on e62b57124: a run THIS close retires as a duplicate registration of one store — the BoN climb a store512 filing
        # matches, the multi-turn group's own run beside the g01234 survivor — never reaches `generated`, so no run_patch carried its
        # store to the registry row, and admission's producer_supersession, which verifies the record's claim by the ROWS (store_uri recorded
        # on both and equal), refused the run as having an unrecorded store. The inspected facts — the four store facts, no generated_utc
        # (the run never reaches that state; the `cancelled` filing's shape) — are filed FIRST, before the `superseded` (under the store lock
        # for a BoN run), so the row records the store the record claims; `generated` files the same facts plus its clock.
        store_fields = {"store_uri": store, "store_completion_artifact_uri": artifact, "store_completion_sha256": _sha256(artifact), "n_items_generated": n_items}
        if fam == "bon" and bon_role == "climb" and wave_root and tag:
            # a climb run's `store_uri` is the 512 STORE it climbed (the twin rule compares it to
            # the store512's), so the climb's own artifacts — `$WAVE_ROOT/bon_climb/<TAG>/{plan/climb_manifest.jsonl, out/<cell>/shard<k>.jsonl}`,
            # bon_climb.sbatch's ROOT — had NO record and the gate could only guess them from the producer convention. The closer, the ONE filer of
            # store locations, records the flat root beside the store facts as `extra_json.climb_root` (the fold's additive merge keeps it
            # under a later patch); 06 --gate reads the climb there (stage.py probe_store) and reads "no store on record" by name without it
            store_fields["extra_json"] = {"climb_root": os.path.join(wave_root, "bon_climb", tag)}
        # decode-recording (the registry owner, 2026-10-13: decode_policy_id 'unrecorded' is a PRODUCER gap): the request's OBSERVED sampler, read from
        # the store this close inspected (bon: the manifest's `decode` + every recorded row; multiturn: the inspected shards' progress rows' target_decode,
        # corroborated by served_target.json; static / padding: every shard's served_target.json#decode + every response row; controls: the final-rung
        # cell's manifest read for the run's OWN control (xstest / or_bench_hard_1k), corroborated by gen_stats.json; misalign: meta.json#sampling) --
        # filed in the SAME run_patch as the store facts, by file_store_facts_now and file_generated_now alike: decode_sent_json + the typed *_sent
        # columns + the per-column confidence entries + extra_json.decode_evidence (stamp_run_rows re-derives decode_policy_id / _match from them every
        # cycle). A contradiction (two samplers in one store, a producer disagreeing with itself) or a shape problem REFUSES the run by name, like the
        # closer's other inconsistent-store refusals; an absence files the facts as before plus the `unverified` entry naming why.
        try:
            decode_ev = DE.for_family(fam, store, spelling=(CONTROL_KEY.get(eval_id) if fam == "controls" else None))
        except DE.DecodeEvidenceProblem as exc:
            _say(f"REFUSED {eval_id} {obj} {rid[:40]}…: decode evidence: {exc} — the run stays {'generated' if was_generated else 'running'}"); out["refused"] += 1
            if not was_generated:
                refused_ids.append(rid)
            continue
        # Both lanes file under the SAME run_patch key: gate-fixes' `extra_json.climb_root` (above) and decode-recording's
        # `extra_json.decode_evidence` (patch_fields, when the sampler was recorded or resolved). One dict per patch, so the two blocks are
        # MERGED here — never one assignment replacing the other (a climb whose BoN manifest records its sampler files both); a run with
        # neither files no extra_json at all, as before (integration round 3a, the merge of the two lanes).
        decode_fields = decode_ev.patch_fields()
        extra_merged = {**(store_fields.get("extra_json") or {}), **(decode_fields.pop("extra_json", None) or {})}
        store_fields.update(decode_fields)
        if extra_merged:
            store_fields["extra_json"] = extra_merged

        def file_store_facts_now() -> None:
            if dry_run:
                _say(f"DRY-RUN {eval_id} {obj} {rid[:40]}…: would file run_patch {json.dumps(store_fields, sort_keys=True)} (the inspected store facts of a registration this close retires)")
                return
            w = emit.RecordWriter(spool, campaign, rid, producer, push_script=push_script)
            w.emit_patch(fields=store_fields)
            w.push()
            _say(f"store facts {eval_id} {obj} {rid[:40]}…: run_patch store_uri={store} sha {store_fields['store_completion_sha256'][:12]}… n_items={n_items} filed before its "
                 f"retirement (the row records the store the supersede claims; admission verifies by the rows)")

        canon_sibs = [t for t in twins if canonical_group(t[3]) is True]

        def supersede_twins() -> None:
            # this run is the g01234 group's — every other group's registration of this store is retired, a `generated` one included
            # (a close from an earlier tree, or a reclose, may have filed it; the correction is this record, where earlier ones were filed by hand)
            for s_job, s_rid, s_campaign, _s_name in twins:
                if already_generated(spool, s_campaign, s_rid, states=("cancelled", "superseded")):
                    continue
                file_superseded(spool=spool, push_script=push_script, campaign=s_campaign, rid=s_rid, by_rid=rid, artifact=artifact, producer=producer,
                                sibling_job=this_job, eval_id=eval_id, obj=obj, dry_run=dry_run)
                out["superseded"] += 1

        if twins and this_canon is False and canon_sibs:
            # this registration is the other group's — retired in favour of the g01234 group's run, whatever closed first (its `generated`,
            # if an older closer filed one, stands corrected by this record)
            s_job, s_rid, _s_campaign, _s_name = canon_sibs[0]
            file_store_facts_now()                                 # the whole store this close inspected, on this run's row before its retirement
            file_superseded(spool=spool, push_script=push_script, campaign=campaign, rid=rid, by_rid=s_rid, artifact=artifact, producer=producer,
                            sibling_job=s_job, eval_id=eval_id, obj=obj, dry_run=dry_run)
            out["superseded"] += 1; continue
        if twins and this_canon is not True:
            _say(f"note {eval_id} {obj} {rid[:40]}…: sibling registration(s) of this store by job(s) {', '.join(sorted(t[0] for t in twins))}, but no shard-00 "
                 f"group can be told from the job names ({doc.get('job_name')!r} vs {sorted(set(t[3] for t in twins))}) — no twin filed; file by hand (D1521)")
            twins = []
        if was_generated and cert is None:
            _say(f"skip {eval_id} {obj} {rid[:40]}… already generated"); out["skipped"] += 1
            if twins:
                supersede_twins()                                      # An already-generated g01234 run still retires its twins
            continue
        now = utc_now()
        fields = {**store_fields, "generated_utc": now}
        note = f"filed by the job-side closer ({SCRIPT}) against the registration's plan ({planned} planned, item set {iset}) from {store}: {json.dumps(facts, sort_keys=True)}"

        def file_generated_now() -> None:
            if dry_run:
                _say(f"DRY-RUN {eval_id} {obj} {rid[:40]}…: would file run_patch {json.dumps(fields, sort_keys=True)} + run_state generated ({fam})")
            else:
                w = emit.RecordWriter(spool, campaign, rid, producer, push_script=push_script)
                w.emit_patch(fields=fields)
                w.emit_state("generated", "completion_artifact", artifact, event_utc=now, note=note[:2000])
                w.push(final=True)
                _say(f"generated {eval_id} {obj} {rid[:40]}… ({fam}) store_uri={store} n_items={n_items}/{planned}")
            out["filed"] += 1

        if fam == "bon":
            # EVERY write for a BoN run — its `generated` (and the duplicate-registration decision beside it), its no-fit certificate — is
            # taken under the store's lock, and the run's state is RE-READ there: the climb array's four tasks share one run id since
            # the job-independent key, so a peer close of THIS run may have filed `generated` / `superseded` / the certificate between the
            # unlocked reads at the top of the loop and this lock. One transition and one certificate per run, whichever task gets there first;
            # the later one re-evaluates its twins over the FILED facts and writes nothing twice.
            # The duplicate rule, this run closing now: a store512 files `generated` then retires the climb twins whose filing matches; a climb
            # files its OWN run superseded when a store512 filing of this store matches, `generated` when the store512 has not closed yet, and
            # is REFUSED by name (held; nothing filed) when the bytes or the count differ.
            mine_now = {k: fields[k] for k in ("store_uri", "store_completion_artifact_uri", *BON_DUPLICATE_FIELDS)}
            try:
                with bon_close_lock(store, dry_run):
                    if not was_generated and already_generated(spool, campaign, rid, states=("superseded",)):
                        _say(f"skip {eval_id} {obj} {rid[:40]}… already superseded under the lock by another close of this run ({BON_SUPERSEDED_KIND}; "
                             f"{BON_SUPERSEDED_DECISION}) — nothing filed")
                        out["skipped"] += 1; continue
                    if not was_generated and already_generated(spool, campaign, rid):
                        # a peer close of this run filed `generated` while this one inspected the store: no second filing; this close is the
                        # re-evaluation over the FILED facts (the same path a reclose takes), and the certificate below is re-read the same way
                        _say(f"skip {eval_id} {obj} {rid[:40]}… already generated under the lock by another close of this run (the climb array's peer task) — "
                             f"no second filing; its twins are judged over the filed facts")
                        was_generated = True; out["skipped"] += 1
                        mine_f = filed_store_facts(spool, campaign, rid)
                        if bon_role and bon_twins and mine_f is not None:
                            why_b, self_superseded = bon_settle_duplicates(role=bon_role, rid=rid, campaign=campaign, mine=mine_f, twins=bon_twins, spool=spool,
                                                                           push_script=push_script, producer=producer, this_job=this_job, eval_id=eval_id, obj=obj,
                                                                           dry_run=dry_run, out=out)
                            if why_b:
                                _say(f"HELD {eval_id} {obj} {rid[:40]}…: {why_b} (D456: bytes or item count differ) — the run stays generated, no supersede filed; a human rules")
                                out["held"] += 1
                            if self_superseded:
                                continue
                    elif not was_generated:
                        if bon_role:
                            why_b, self_superseded = bon_settle_duplicates(role=bon_role, rid=rid, campaign=campaign, mine=mine_now, twins=bon_twins, spool=spool,
                                                                           push_script=push_script, producer=producer, this_job=this_job, eval_id=eval_id, obj=obj,
                                                                           dry_run=dry_run, out=out, file_generated=file_generated_now, file_store_facts=file_store_facts_now)
                            if why_b:
                                _say(f"REFUSED {eval_id} {obj} {rid[:40]}…: {why_b} — HELD by name (D456: bytes or item count differ), no supersede filed; the run stays running; file by hand")
                                out["refused"] += 1; refused_ids.append(rid); continue
                            if self_superseded:
                                continue                           # filed superseded by the store512 run under the lock; no `generated`, no certificate
                        else:
                            file_generated_now()                   # a kind the rule does not cover (f22-stat-<tag>-all, …): `generated` as for any lone registration
                    if cert is not None:
                        if already_certified(spool, campaign, rid):
                            _say(f"skip {eval_id} {obj} {rid[:40]}… the no-fit certificate is already filed (an earlier close, or another close of this run under the lock)")
                            continue
                        why_j, judge = instrument_judge(r, repo_root=REPO_ROOT)
                        if why_j or not judge:
                            # the run IS generated (filed above, or by an earlier close) — never listed for the exit path's `running` statement
                            _say(f"REFUSED nofit certificate {eval_id} {obj} {rid[:40]}…: {why_j} — the run stays generated (a grader would refuse its empty store)"); out["refused"] += 1; continue
                        file_nofit_certificate(spool=spool, push_script=push_script, campaign=campaign, rid=rid, store=store, judge=judge, cert=cert, producer=producer, dry_run=dry_run)
                        out["certified"] = int(out.get("certified", 0)) + 1
            except StoreCloseLockBusy as exc:
                _say(f"REFUSED {eval_id} {obj} {rid[:40]}…: {exc} — the run stays {'generated' if was_generated else 'running'}"); out["refused"] += 1
                if not was_generated:
                    refused_ids.append(rid)
                continue
            continue
        if not was_generated:
            file_generated_now()
            if twins:
                supersede_twins()                                      # The g01234 group's run filed; the other group's registration is its twin
    _say((f"close (DRY RUN — nothing written): would file {out['filed']}, skip {out['skipped']}, refuse {out['refused']} of {out['registered']} registered run(s)" if dry_run
          else f"close: {out['filed']} filed, {out['skipped']} skipped, {out['refused']} refused of {out['registered']} registered run(s)")
         + (f", {out['cancelled']} cancelled (shards ruled out; D1168)" if out.get("cancelled") else "")
         + (f", {out['superseded']} superseded (a twin registration of one store; D1816)" if out.get("superseded") else "")
         + (f", {out['superseded_bon']} superseded ({BON_SUPERSEDED_KIND}; {BON_SUPERSEDED_DECISION})" if out.get("superseded_bon") else "")
         + (f", {out['held']} held (a BoN registration whose bytes or item count differ from its twin's; D456 — no supersede, a human rules)" if out.get("held") else "")
         + (f", {out['other_emitter']} another emitter's (not a refusal)" if out.get("other_emitter") else "")
         + (f", {out['certified']} no-fit-by-design certificate(s)" if out.get("certified") else "") + f"; spool {spool}")
    if not dry_run:
        write_refused_run_ids(pending_path, this_job, refused_ids)
    return out


def refused_run_ids_path(pending_path: str) -> str:
    """The sidecar beside the job's pending file naming the FULL run ids its closer refused (the exit path's `running` targets)."""
    return pending_path + ".refused_run_ids.json"


def write_refused_run_ids(pending_path: str, job_id: str, refused_ids: List[str]) -> Optional[str]:
    """Write the refused-ids sidecar (atomic), or remove a stale one when this close refused nothing; returns the path written or None."""
    p = refused_run_ids_path(pending_path)
    ids = sorted(set(refused_ids))
    if not ids:
        if os.path.isfile(p):
            os.remove(p); _say(f"refused-run list {p} removed (this close refused nothing)")
        return None
    tmp = f"{p}.tmp.{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump({"job_id": job_id, "refused_run_ids": ids, "written_utc": utc_now(), "decision": "D2100"}, fh, indent=1, sort_keys=True); fh.write("\n")
    os.replace(tmp, p)
    _say(f"refused-run list: {len(ids)} run id(s) -> {p} (the exit path states `running` for these alone; D2100 (b))")
    return p


# ------------------------------------------------------------------------------------- the successor relabel (U0STATICSUCC)
SUCCESSOR_REASON = "token_cap_identity_relabel"      # the registry's vocabulary keeps the r-word out of everything but token_cap_rung_index
SUCCESSOR_DECISION = "D1900"
STATIC_LADDER = (4096, 8192, 16384)                  # common_sampled.sh F22S_STATIC_RUNGS_EXT — the ladder every static registration indexes
ORPHAN_REASON = "attempt_failed_before_store"


def _records_of(spool: str, campaign: str, run_id: str) -> List[Tuple[str, dict]]:
    """Every (record_type, payload) in the run's spool directory, in file order ([] when it has none)."""
    d = os.path.join(spool, campaign, identity.run_dir_name(run_id)); out: List[Tuple[str, dict]] = []
    if not os.path.isdir(d):
        return out
    for fn in sorted(os.listdir(d)):
        m = R.RECORD_FILE_RE.match(fn)
        if not m:
            continue
        try:
            for _i, rec, _sha in R.read_record_file(os.path.join(d, fn), compute_sha=False):
                p = rec.get("payload") if isinstance(rec, dict) else None
                if isinstance(p, dict) and p.get("run_id") == run_id:
                    out.append((str(m.group("type")), p))
        except Exception:  # noqa: BLE001 — an unreadable record is not evidence
            continue
    return out


def already_registered(spool: str, campaign: str, run_id: str) -> bool:
    return any(t == "run" for t, _p in _records_of(spool, campaign, run_id))


def filed_store_uri(spool: str, campaign: str, run_id: str) -> Tuple[str, str]:
    """(store_uri, store_completion_artifact_uri) the closer filed for the run — the last run_patch carrying them, else ('', '')."""
    store, art = "", ""
    for t, p in _records_of(spool, campaign, run_id):
        f = (p.get("fields") or {}) if t == "run_patch" else {}
        if f.get("store_uri"):
            store = str(f["store_uri"]); art = str(f.get("store_completion_artifact_uri") or art)
    return store, art


def successor_pending_path(pending_path: str) -> str:
    """The successor registrations' pending file: `<pending dir>/succ/<original>.json` — our key (the harvester folds payloads,
    not spool keys); the payloads carry the PRODUCING job's id as cluster_job_id, never an invented one."""
    d, b = os.path.split(pending_path)
    return os.path.join(d, "succ", b)


def producing_code_sha(spool: str, campaign: str, run_id: str) -> Optional[str]:
    """The code_sha the PRODUCING job stated on the retired run's `run` record (the last one), else None."""
    sha = None
    for t, p in _records_of(spool, campaign, run_id):
        if t == "run" and p.get("code_sha"):
            sha = str(p["code_sha"])
    return sha


def register_successors(pending_path: str, *, spool: str, push_script: Optional[str], code_sha: Optional[str], tag: Optional[str], job_name: Optional[str],
                        trial: Optional[int], token_cap: int, pass_label: str, config: Optional[str], evals: Optional[List[str]], dry_run: bool,
                        producer: Dict[str, object], objects: Optional[List[str]] = None, wave_root: Optional[str] = None,
                        nshards: int = 9) -> Tuple[Optional[str], str, List[dict]]:
    """Re-register every STATIC run of the original pending file at the DRAWN-rung identity through the registry's own planner: the key
    `sdfbase_full22_<tag>_sampled_t<T>` (so the id is the one a post-fix static_base job mints — a second call is the same id), token_cap_tokens =
    the drawn cap with its ladder index, pass_label, trial, cluster_job_id / job_name = the ORIGINAL (producing) job's, extra.successor_of = the
    retired run id. Writes the successors' `run` records (lifecycle running) and the successor pending file; idempotent (an existing successor
    pending file is read, a registered run is not re-written). Returns (refusal, successor pending path, pairs)."""
    from sisteer.registry import jobplan as JP
    if not objects:
        # the relabel never plans a view's whole cell set — a steered view's legacy `none` rows cannot be
        # planned (its configs enumerate the steered cells) and are NOT relabelled (no unsteered leg on a steered wave); the source set is named
        return f"{pending_path}: name --objects explicitly (the source set is stated, never the pending file's whole cell set; D2084 (c))", "", []
    if not wave_root:
        return f"{pending_path}: the successor registration needs --wave-root (every retired run's filed store is verified against the unit's store before anything is written; D2084 (a)/(b))", "", []
    with open(pending_path, encoding="utf-8") as fh:
        doc = json.load(fh)
    runs = [r for r in (doc.get("runs") or []) if family_of(r) == "static" and (not evals or str((r.get("route") or {}).get("eval_id")) in evals)
            and (not objects or str((r.get("route") or {}).get("object")) in objects)]
    if not runs:
        return f"{pending_path}: no static run to relabel (evals {evals}, objects {objects})", "", []
    orig_job = str(doc.get("job_id") or os.path.splitext(os.path.basename(pending_path))[0])
    if not orig_job.isdigit():
        return f"{pending_path}: the original job id {orig_job!r} is not a scheduler id — a successor names the PRODUCING job and never an invented one", "", []
    job_name = job_name or str(doc.get("job_name") or "")
    t = trial_of(job_name, trial)
    if t is None:
        return f"{pending_path}: no trial (--trial or a job name carrying _t<N>); the drawn-rung key needs it", "", []
    plan0 = runs[0].get("plan") or {}
    tag = tag or str(plan0.get("view_legacy") or "")
    view = str((plan0.get("extra") or {}).get("view_legacy") or "")
    if not tag or not view:
        return f"{pending_path}: the registration names no view_legacy / extra.view_legacy — the successor's tag and roster selection cannot be formed", "", []
    if token_cap not in STATIC_LADDER:
        return f"--successor-token-cap {token_cap} is not on the static ladder {list(STATIC_LADDER)}", "", []
    key = f"sdfbase_full22_{tag}_sampled_t{t}"
    cfg = config or os.path.join(REPO_ROOT, "configs", f"sdfbase_full22_{tag}_sampled_t{t}.yaml")
    if not os.path.isfile(cfg):
        return f"the successor's config {cfg} is not on disk (the sampled trial config a post-fix static job registers from)", "", []
    root = JP._checkout_root(cfg)
    seeds = os.path.join(root, "configs", "registry") if os.path.isdir(os.path.join(root, "configs", "registry")) else os.path.join(REPO_ROOT, "configs", "registry")
    roster = os.path.join(REPO_ROOT, "configs", "sdfbase_full22_roster.yaml"); pins = os.path.join(REPO_ROOT, "configs", "sdfbase_full22_pins.json")
    succ_pend = successor_pending_path(pending_path)
    pairs: List[dict] = []
    if os.path.isfile(succ_pend):
        with open(succ_pend, encoding="utf-8") as fh:
            sdoc = json.load(fh)
        by_pair = {(str(r["route"]["eval_id"]), str(r["route"]["object"])): r for r in sdoc.get("runs") or []}
        for r in runs:
            ev, obj = str(r["route"]["eval_id"]), str(r["route"]["object"]); s = by_pair.get((ev, obj))
            if s is None:
                return f"{succ_pend}: an earlier successor registration exists but carries no run for ({ev}, {obj}) — remove nothing; relabel by hand", "", []
            pairs.append({"old": str(r["run_id"]), "new": str(s["run_id"]), "eval_id": ev, "object": obj, "tag": tag, "trial": t, "job": orig_job, "key": key})
        _say(f"successor: {succ_pend} already registers {len(pairs)} run(s) for job {orig_job} — reused (idempotent)")
        return None, succ_pend, pairs
    # FIRST every retired run is checked and every successor planned; nothing is emitted until all of
    # them pass — a refusal names the run and the fact, and the spool is exactly as it was.
    unit_store_root = os.path.join(wave_root, "transcripts_results", f"sdfbase_full22_{tag}_sampled_t{t}")
    planned_runs: List[Tuple[dict, str, str, str, object, str, str]] = []
    for r in runs:
        ev, obj, old = str(r["route"]["eval_id"]), str(r["route"]["object"]), str(r["run_id"])
        plan = r.get("plan") or {}
        campaign_old = str(plan.get("campaign") or identity.parse_run_id(old)["campaign"])
        # (a) the retired run FILED the store being reused: store_uri + completion artifact, and that store is the unit's planned store
        store_old, art_old = filed_store_uri(spool, campaign_old, old)
        if not store_old or not art_old:
            return (f"retired {old[:40]}… ({ev}, {obj}, job {orig_job}): it filed no store_uri / completion artifact — a FAILED attempt's registration never "
                    f"produced the store and cannot seed a successor (D2084 (a)); nothing registered"), "", []
        want_store = os.path.join(unit_store_root, ev, obj)
        if os.path.normpath(store_old) != os.path.normpath(want_store):
            return (f"retired {old[:40]}… ({ev}, {obj}): its filed store_uri {store_old} is not the unit's store {want_store} — not this unit's run "
                    f"(D2084 (a)); nothing registered"), "", []
        # (b) the store the successor will be closed from is COMPLETE now, by the closer's own static inspection — before any emission
        planned_n, _iset, goals, _variants = planned_of(r)
        why_store, _facts, _art = inspect_static(want_store, planned=int(planned_n or 0), goals=goals, nshards=nshards)
        if why_store:
            return f"retired {old[:40]}… ({ev}, {obj}): the store {want_store} is not complete — {why_store} (D2084 (b)); nothing registered", "", []
        # (d) the successor states the PRODUCING code_sha, read from the retired run's record — never this checkout's HEAD
        prod_sha = producing_code_sha(spool, campaign_old, old)
        if not prod_sha:
            return f"retired {old[:40]}… ({ev}, {obj}): its `run` record states no code_sha — the successor cannot name the producing tree (D2084 (d))", "", []
        try:
            bundle = JP.build_plans(config_path=cfg, model_key=tag, evals=[ev], serving_path=str(plan.get("serving_path") or "lora_native"),
                                    campaign=campaign_old, job_id=orig_job, job_name=job_name,
                                    roster_selection=view, seeds_dir=seeds, roster_path=roster, pins_path=pins, code_sha=prod_sha,
                                    substrate=producer.get("substrate") if isinstance(producer.get("substrate"), str) else None, lifecycle_state="running",
                                    extra={"successor_of": old, "successor_reason": SUCCESSOR_REASON, "successor_decision": SUCCESSOR_DECISION, "producing_job_id": orig_job,
                                           "producing_code_sha": prod_sha},
                                    producer_key=key, token_cap=token_cap, token_cap_rung_index=STATIC_LADDER.index(token_cap), ladder=list(STATIC_LADDER),
                                    pass_label=pass_label, trial_index=t, pairs=[f"{ev}:{obj}"])
        except (JP.JobPlanRefused, OSError, ValueError) as exc:
            return f"successor of {old[:40]}… ({ev}, {obj}): the planner refused: {type(exc).__name__}: {exc}", "", []
        if len(bundle.runs) != 1:
            return f"successor of {old[:40]}… ({ev}, {obj}): the planner yielded {len(bundle.runs)} run(s), not one — " + "; ".join(f"{w}: {why}" for w, why in bundle.skipped), "", []
        pr = bundle.runs[0]
        if str(pr.plan.get("cluster_job_id")) != orig_job:
            return f"successor of {old[:40]}…: the plan's cluster_job_id {pr.plan.get('cluster_job_id')!r} is not the producing job {orig_job}", "", []
        if str(pr.plan.get("code_sha") or "") != prod_sha:
            return f"successor of {old[:40]}…: the plan's code_sha {pr.plan.get('code_sha')!r} is not the producing sha {prod_sha} (D2084 (d))", "", []
        planned_runs.append((r, ev, obj, old, pr, bundle, prod_sha))
    # every source run checked, every successor planned: now the emission (dry: described)
    pdoc: Optional[dict] = None
    for r, ev, obj, old, pr, bundle, prod_sha in planned_runs:
        if dry_run:
            _say(f"DRY-RUN successor {ev} {obj} {old[:40]}… -> {pr.run_id[:40]}… (key {key}, cap {token_cap} idx {STATIC_LADDER.index(token_cap)} pass {pass_label} trial {t}, job {orig_job}, code_sha {prod_sha[:12]} = the producing tree)")
        elif already_registered(spool, str(pr.plan["campaign"]), pr.run_id):
            _say(f"successor {ev} {obj} {old[:40]}… -> {pr.run_id[:40]}… already registered (the same key, the same id)")
        else:
            _ids, run_payload, _state = emit.planned_run_records(pr.plan, producer)
            run_payload["lifecycle_state"] = "running"; run_payload.setdefault("started_utc", run_payload["queued_utc"])
            run_payload["code_sha"] = prod_sha                                            # (d): the producing tree, never the relabel checkout's
            w = emit.RecordWriter(spool, str(pr.plan["campaign"]), pr.run_id, producer, push_script=push_script)
            w.emit_run(run_payload); w.push(final=True)
            _say(f"successor registered {ev} {obj} {old[:40]}… -> {pr.run_id[:40]}… (key {key}, cap {token_cap} pass {pass_label} trial {t}, job {orig_job}, code_sha {prod_sha[:12]})")
        one = bundle.as_pending(orig_job, job_name)
        if pdoc is None:
            pdoc = one
        else:
            pdoc["runs"].extend(one["runs"])
        pairs.append({"old": old, "new": pr.run_id, "eval_id": ev, "object": obj, "tag": tag, "trial": t, "job": orig_job, "key": key})
    assert pdoc is not None
    pdoc["successor_of_pending"] = pending_path; pdoc["successor_decision"] = SUCCESSOR_DECISION; pdoc["successor_reason"] = SUCCESSOR_REASON
    if dry_run:
        import tempfile
        tmpd = tempfile.mkdtemp(prefix="succ_dry_"); succ_pend = os.path.join(tmpd, os.path.basename(pending_path))
    os.makedirs(os.path.dirname(succ_pend), exist_ok=True)
    with open(succ_pend + ".tmp", "w", encoding="utf-8") as fh:
        json.dump(pdoc, fh, indent=1, sort_keys=True)
    os.replace(succ_pend + ".tmp", succ_pend)
    return None, succ_pend, pairs


def file_successor_superseded(*, spool: str, push_script: Optional[str], campaign: str, old: str, new: str, artifact: str, producer: Dict[str, object],
                              eval_id: str, obj: str, job: str, dry_run: bool) -> bool:
    """The twin pattern: run_state superseded on the pre-fix row, the successor named in extra and the note."""
    if already_generated(spool, campaign, old, states=("superseded",)):
        _say(f"skip {eval_id} {obj} {old[:40]}… already superseded"); return False
    now = utc_now()
    payload = {"run_id": old, "state": "superseded", "event_utc": now, "evidence_kind": "completion_artifact", "evidence_uri": artifact,
               "note": (f"superseded by {new} — the same store, re-registered at the DRAWN-rung identity (token_cap_tokens {STATIC_LADDER[0]}, pass_label base_cap; "
                        f"static-reg-fix 158242e5d) by the job-side closer ({SCRIPT}); this row's declared cap / pass predate the fix ({SUCCESSOR_DECISION}); "
                        f"produced by job {job}"),
               "extra": {"superseded_by_run_id": new, "reason": SUCCESSOR_REASON, "decision": SUCCESSOR_DECISION, "producing_job_id": job}}
    if dry_run:
        _say(f"DRY-RUN {eval_id} {obj} {old[:40]}…: would file run_state superseded (by {new[:40]}…; {SUCCESSOR_REASON}, {SUCCESSOR_DECISION})"); return True
    w = emit.RecordWriter(spool, campaign, old, producer, push_script=push_script)
    w.write("run_state", payload); w.push(final=True)
    _say(f"superseded {eval_id} {obj} {old[:40]}… by {new[:40]}… ({SUCCESSOR_REASON}; {SUCCESSOR_DECISION})")
    return True


def cancel_orphans(specs: List[str], *, spool: str, push_script: Optional[str], producer: Dict[str, object], pairs: List[dict], evals: Optional[List[str]],
                   dry_run: bool, objects: Optional[List[str]] = None) -> Dict[str, int]:
    """The store-less, judgment-less registrations of FAILED attempts (`PENDING=EVIDENCE`, the scheduler's terminal state as the record
    can cite it) are filed `cancelled` — evidence_kind scheduler_state — with a note naming the attempt that produced the store and the
    successor; a run that filed a store_uri, or is generated / superseded / cancelled already, is never touched (said by name)."""
    out = {"cancelled": 0, "skipped": 0, "refused": 0, "ids": []}
    succ = {(p["eval_id"], p["object"]): p for p in pairs}
    for spec in specs:
        if "=" not in spec:
            _say(f"REFUSED --cancel-orphan {spec!r}: PENDING_PATH=EVIDENCE"); out["refused"] += 1; continue
        path, evidence = spec.split("=", 1)
        with open(path, encoding="utf-8") as fh:
            doc = json.load(fh)
        job = str(doc.get("job_id") or os.path.splitext(os.path.basename(path))[0]); job_name = str(doc.get("job_name") or "")
        for r in doc.get("runs") or []:
            if family_of(r) != "static":
                continue
            ev, obj, rid = str(r["route"]["eval_id"]), str(r["route"]["object"]), str(r["run_id"])
            if (evals and ev not in evals) or (objects and obj not in objects):
                continue
            campaign = str((r.get("plan") or {}).get("campaign") or identity.parse_run_id(rid)["campaign"])
            store, _art = filed_store_uri(spool, campaign, rid)
            if store or already_generated(spool, campaign, rid, states=("generated", "judged", "superseded")):
                _say(f"REFUSED orphan {ev} {obj} {rid[:40]}… (job {job}): it filed a store_uri ({store or 'a terminal state'}) — not an orphan; nothing cancelled"); out["refused"] += 1; continue
            if already_generated(spool, campaign, rid, states=("cancelled",)):
                _say(f"skip orphan {ev} {obj} {rid[:40]}… (job {job}): already cancelled"); out["skipped"] += 1; continue
            s = succ.get((ev, obj))
            note = (f"attempt {job} ({job_name}) ended {evidence} before writing its store; the store was produced by attempt "
                    + (f"{s['job']} and its run is registered at the drawn rung as {s['new']} ({SUCCESSOR_DECISION})" if s else f"a later attempt ({SUCCESSOR_DECISION})"))
            if dry_run:
                _say(f"DRY-RUN orphan {ev} {obj} {rid[:40]}… (job {job}): would file run_state cancelled (scheduler_state: {evidence})"); out["cancelled"] += 1; out["ids"].append(rid); continue
            w = emit.RecordWriter(spool, campaign, rid, producer, push_script=push_script)
            w.write("run_state", {"run_id": rid, "state": "cancelled", "event_utc": utc_now(), "evidence_kind": "scheduler_state", "evidence_uri": evidence, "note": note,
                                  "extra": {"reason": ORPHAN_REASON, "decision": SUCCESSOR_DECISION, "produced_by_job_id": s["job"] if s else None, "successor_run_id": s["new"] if s else None}})
            w.push(final=True)
            _say(f"cancelled orphan {ev} {obj} {rid[:40]}… (job {job}; {evidence})"); out["cancelled"] += 1; out["ids"].append(rid)
    return out


def successor_relabel(pending_path: str, *, spool: str, push_script: Optional[str], code_sha: Optional[str], wave_root: Optional[str], tag: Optional[str],
                      job_name: Optional[str], trial: Optional[int], nshards: int, token_cap: int, pass_label: str, config: Optional[str],
                      pairs_out: Optional[str], cancel: List[str], evals: Optional[List[str]], dry_run: bool, objects: Optional[List[str]] = None) -> int:
    """register the successors -> close them from the EXISTING store (the static branch, nothing re-run) -> supersede the old rows -> pairs file ->
    cancel the orphans. Exit 2 on any refusal; nothing partial is hidden (every step says what it did by name)."""
    if not wave_root:
        _say("REFUSED: the successor relabel needs --wave-root (the store it closes from)"); return 2
    producer = emit.producer_block(SCRIPT, code_sha=code_sha)
    why, succ_pend, pairs = register_successors(pending_path, spool=spool, push_script=push_script, code_sha=code_sha, tag=tag, job_name=job_name, trial=trial,
                                                token_cap=token_cap, pass_label=pass_label, config=config, evals=evals, dry_run=dry_run, producer=producer, objects=objects,
                                                wave_root=wave_root, nshards=nshards)
    if why:
        _say(f"REFUSED: {why}"); return 2
    tag = tag or pairs[0]["tag"]
    out = close_job(succ_pend, spool=spool, push_script=push_script, code_sha=code_sha, wave_root=wave_root, tag=tag, job_name=job_name, trial=trial, nshards=nshards,
                    private_root=None, private_run=None, evals=evals, dry_run=dry_run)
    if out.get("refused"):
        _say(f"REFUSED: {out['refused']} successor run(s) could not be closed from the existing store — nothing superseded"); return 2
    n_sup = 0
    for p in pairs:
        campaign = identity.parse_run_id(p["new"])["campaign"]
        _store, art = filed_store_uri(spool, campaign, p["new"])
        if not art and not dry_run:
            _say(f"REFUSED: the successor {p['new'][:40]}… filed no completion artifact — not superseding {p['old'][:40]}…"); return 2
        if file_successor_superseded(spool=spool, push_script=push_script, campaign=campaign, old=p["old"], new=p["new"], artifact=art, producer=producer,
                                     eval_id=p["eval_id"], obj=p["object"], job=p["job"], dry_run=dry_run):
            n_sup += 1
    orphans = cancel_orphans(cancel, spool=spool, push_script=push_script, producer=producer, pairs=pairs, evals=evals, dry_run=dry_run, objects=objects) if cancel else {"cancelled": 0, "refused": 0, "ids": []}
    if pairs_out and not dry_run:
        os.makedirs(os.path.dirname(os.path.abspath(pairs_out)), exist_ok=True)
        have = set()
        if os.path.isfile(pairs_out):
            with open(pairs_out, encoding="utf-8") as fh:
                have = {ln.split("\t")[0] for ln in fh if ln.strip() and not ln.startswith("#")}
        with open(pairs_out, "a", encoding="utf-8") as fh:
            if not have:
                fh.write("# retired_run_id\tsuccessor_run_id\teval_id\tobject\ttag\ttrial\tproducing_job\tkey\t(cancelled orphans: `cancelled` rows below)\n")
            for p in pairs:
                if p["old"] not in have:
                    fh.write("\t".join([p["old"], p["new"], p["eval_id"], p["object"], p["tag"], str(p["trial"]), p["job"], p["key"]]) + "\n")
            for rid in orphans["ids"]:
                if f"cancelled\t{rid}" not in have:
                    fh.write(f"cancelled\t{rid}\n")
    _say(f"successor relabel: {len(pairs)} pair(s), {n_sup} superseded now, {out['filed']} successor(s) closed now, {orphans['cancelled']} orphan(s) cancelled"
         + (f", {orphans['refused']} orphan refusal(s)" if orphans.get("refused") else "") + (" (DRY RUN)" if dry_run else ""))
    return 2 if orphans.get("refused") else 0


# ------------------------------------------------------------ a SHARED run's array tasks
# Since the job-independent companion key, the climb array's tasks register ONE run per cell, but each task still ends in its own EXIT trap,
# and common.sh's f22_registry_result filed that TASK's outcome as the RUN's lifecycle: a task exiting non-zero after a peer task's close had
# filed `generated` — or after the store512's close had filed the run `superseded` (the duplicate-registration retirement, and its effect on
# the census) — filed `run_state failed` on the shared id, and the reducer applies a lateral/terminal state event whatever stands before it
# (reducer.lifecycle_regresses ranks only the progression), so the retirement disappeared and admission read `failed`; a job-start `running`
# of a task that started after a peer's close reversed it the same way. The rule now: a TASK's event never reverses a CLOSE of the shared
# run. Each task's outcome is recorded ON THE TASK (<pending dir>/<array job>.tasks/<task>.json, one record per task, under the array's
# lock); a task's `failed` / `cancelled` is filed on a run only when no close has decided it (and not twice); a task's `running` only when no
# close stands; and the LAST task to exit reconciles the array under the store lock: a close's terminal stands as it is, a run no close
# decided and some task failed is filed failed once (the first failing task's scheduler state), a run no close decided and no task failed
# stays as its close left it (a refused close's `running` is the reclose's recovery). A job that is not an array is untouched: common.sh
# routes only a task with SLURM_ARRAY_TASK_ID here.
ARRAY_TASK_EVENTS = ("started", "completed", "failed", "cancelled")
ARRAY_TASK_EXITS = ("completed", "failed", "cancelled")            # the events that end a task; `started` is its job-start `running`
ARRAY_TASK_FAILURES = ("failed", "cancelled")
CLOSE_TERMINAL_STATES = ("generated", "superseded", "cancelled")   # the lifecycle states a CLOSE files (close_job's `generated`, bon_settle_duplicates /
                                                                   # supersede_twins' `superseded`, the ruled-out `cancelled`)
ARRAY_TASKS_DIR_SUFFIX = ".tasks"                                  # <pending dir>/<array job id>.tasks/<task>.json — one outcome record per task
ARRAY_TASKS_LOCK_FILE = ".lock"
ARRAY_LOCK_WAIT_S = 300.0
ARRAY_RULE = "a task's event never reverses a close of the shared run (round-2 Major on a9e75c210)"


def parse_task_set(spec: str) -> List[int]:
    """Slurm's array task set as sbatch spells it — `0-3`, `0-7:2`, `1,3,5`, `0-3,7`, with an optional `%<throttle>` — as sorted ints.
    ValueError, by name, for an empty or malformed spec: the array's membership is never guessed."""
    s = str(spec or "").strip()
    if "%" in s:
        s = s.split("%", 1)[0]
    out = set()
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        step = 1
        if ":" in part:
            part, st = part.rsplit(":", 1)
            step = int(st)
        if "-" in part:
            lo, hi = (int(x) for x in part.split("-", 1))
            if hi < lo or step < 1:
                raise ValueError(f"array task set {spec!r}: {part!r} is not an ascending range with a positive step")
            out.update(range(lo, hi + 1, step))
        else:
            out.add(int(part))
    if not out:
        raise ValueError(f"array task set {spec!r} names no task")
    return sorted(out)


def array_tasks_dir(pending_path: str, array_job: str) -> str:
    """The array's outcome directory beside its tasks' pending files: <pending dir>/<array job id>.tasks."""
    return os.path.join(os.path.dirname(os.path.abspath(pending_path)), f"{array_job}{ARRAY_TASKS_DIR_SUFFIX}")


def task_outcomes(d: str) -> Tuple[Dict[int, dict], List[str]]:
    """({task: its exit record} from the array's outcome directory, the files that could not be read as one) — an unreadable file is named,
    never counted as an exited task."""
    out: Dict[int, dict] = {}; bad: List[str] = []
    if not os.path.isdir(d):
        return out, bad
    for fn in sorted(os.listdir(d)):
        m = re.match(r"^(\d+)\.json$", fn)
        if not m:
            continue
        try:
            with open(os.path.join(d, fn), encoding="utf-8") as fh:
                doc = json.load(fh)
        except (OSError, ValueError):
            bad.append(fn); continue
        if isinstance(doc, dict) and doc.get("event") in ARRAY_TASK_EXITS:
            out[int(m.group(1))] = doc
        else:
            bad.append(fn)
    return out, bad


def last_run_state(spool: str, campaign: str, run_id: str) -> Optional[str]:
    """The run's LAST run_state in its spool's record order; None when it has none."""
    st = None
    for t, p in _records_of(spool, campaign, run_id):
        if t == "run_state" and p.get("state"):
            st = str(p["state"])
    return st


def close_terminal_of(spool: str, campaign: str, run_id: str) -> Optional[str]:
    """The lifecycle state a CLOSE filed for the run — `generated`, `superseded` or `cancelled` (CLOSE_TERMINAL_STATES; the latest such record,
    whichever job's close wrote it into the run's spool directory); None when no close has decided the run."""
    st = None
    for t, p in _records_of(spool, campaign, run_id):
        if t == "run_state" and p.get("state") in CLOSE_TERMINAL_STATES:
            st = str(p["state"])
    return st


def bon_store_path(wave_root: str, tag: str, obj: str) -> str:
    """The BoN store both registrations of a cell close: <wave root>/transcripts_results/sdfbase_full22_<tag>/<obj> (close_job's bon branch)."""
    return os.path.join(wave_root, "transcripts_results", f"sdfbase_full22_{tag}", obj)


def _run_lock(fam: Optional[str], obj: str, *, wave_root: Optional[str], tag: Optional[str], dry_run: bool) -> Tuple[Optional[object], str]:
    """(the lock a run's lifecycle events are decided under, how it was chosen): a BoN run's store close lock when the wave root and tag are
    known — the lock its closes take, so a close in flight and this decision serialise, WHETHER OR NOT THE STORE EXISTS YET (an absent store
    used to be decided unlocked, "no close of it can be in flight" — but the absence was read once and the write came after the state
    emitter's hardware capture, an interval in which a store created, completed and closed could land its `superseded` first; the lock lives
    beside the store now, bon_close_lock_path, so there is no unlocked branch); (None, why) for a BoN run whose store lock CANNOT be named
    (no --wave-root/--tag, no object) — the callers file nothing for it, by name (the fence IS the lock, so a BoN run's lifecycle is never
    written outside it); a null context, said by name, for every other family."""
    if fam == "bon":
        if wave_root and tag and obj:
            return bon_close_lock(bon_store_path(wave_root, tag, obj), dry_run), "under the store close lock"
        return None, "a BoN run's store close lock cannot be named (" + ("no --wave-root/--tag" if not (wave_root and tag) else "no object") + ")"
    return contextlib.nullcontext(), "without a store lock (" + (f"family {fam}" if fam else "no family") + ("" if wave_root and tag else "; no --wave-root/--tag") + ")"


LOCK_RULE = "a BoN run's lifecycle is decided under its store close lock alone"


def _file_task_terminal(*, spool: str, push_script: Optional[str], campaign: str, rid: str, event: str, scheduler_state: Optional[str],
                        producer: Dict[str, object], note: str) -> None:
    """The run_state a task's failure files when no close has decided the run — emit_cli state's shape (evidence scheduler_state), the note
    naming the task, pushed final."""
    w = emit.RecordWriter(spool, campaign, rid, producer, push_script=push_script)
    w.emit_state(event, "scheduler_state", "", scheduler_state=scheduler_state, note=note[:2000])
    w.push(final=True)


def array_task_event(pending_path: str, *, event: str, array_job: str, task: int, tasks_spec: str, spool: str, push_script: Optional[str],
                     code_sha: Optional[str], wave_root: Optional[str], tag: Optional[str], scheduler_state: Optional[str] = None,
                     rc: Optional[int] = None, node: Optional[str] = None, dry_run: bool = False) -> int:
    """One array task's lifecycle event for the runs its pending file registered (the block comment above): `started` (its job-start
    `running`), `completed` (a clean exit, after its close ran), `failed` / `cancelled` (the RESULT hook's terminal). Every decision under
    the array's lock (<array job>.tasks/.lock) and, per run, the store close lock; the task's exit is recorded on the task; the last task to
    exit reconciles the array. `tasks_spec` is the scheduler's task set (SLURM_ARRAY_TASK_MIN-MAX[:STEP]). Returns 0, or 2 on a refusal
    (nothing filed for what it refused). --dry-run says what it would do and writes nothing — no record, no outcome file."""
    if event not in ARRAY_TASK_EVENTS:
        _say(f"REFUSED: array task event {event!r} is not one of {', '.join(ARRAY_TASK_EVENTS)}"); return 2
    if not str(array_job or "").strip():
        _say("REFUSED: --array-job is empty — the array's outcome directory cannot be named; nothing filed, nothing recorded"); return 2
    try:
        tasks = parse_task_set(tasks_spec)
    except ValueError as exc:
        _say(f"REFUSED: {exc} — the array's membership comes from the scheduler (SLURM_ARRAY_TASK_MIN-MAX[:STEP]); nothing filed, nothing recorded"); return 2
    if task not in tasks:
        _say(f"REFUSED: array task {task} is not in the array's task set {tasks_spec!r} ({tasks}); nothing filed, nothing recorded"); return 2
    try:                                                  # no registered-config resolution here, so no PENDING_DIR binding (main binds it before every entry)
        with open(pending_path, encoding="utf-8") as fh:
            doc = json.load(fh)
    except (OSError, ValueError) as exc:
        _say(f"REFUSED: {pending_path} is unreadable ({type(exc).__name__}: {exc}); nothing filed, nothing recorded"); return 2
    runs = doc.get("runs") if isinstance(doc, dict) else None
    if not isinstance(runs, list) or not runs:
        _say(f"REFUSED: {pending_path} names no `runs` (the task registered nothing); nothing filed, nothing recorded"); return 2
    this_job = str(doc.get("job_id") or os.path.basename(pending_path)[:-5]); job_name = str(doc.get("job_name") or "")
    if rc is None and scheduler_state:
        m_rc = re.search(r"\brc=(\d+)\b", scheduler_state)
        rc = int(m_rc.group(1)) if m_rc else None
    producer = emit.producer_block(SCRIPT, code_sha=code_sha)
    who = f"array {array_job} task {task} (job {this_job}{', ' + job_name if job_name else ''})"
    mine: List[Dict[str, object]] = []
    for r in runs:
        rid = str(r.get("run_id") or ""); route = r.get("route") or {}; plan = r.get("plan") or {}
        if not rid:
            continue
        mine.append({"run_id": rid, "eval_id": str(route.get("eval_id") or plan.get("eval_id") or ""), "object": str(route.get("object") or ""),
                     "campaign": str(plan.get("campaign") or identity.parse_run_id(rid)["campaign"]), "family": family_of(r)})
    d = array_tasks_dir(pending_path, array_job)
    worst = 0

    def settle(m: Dict[str, object], ev: str, sched: Optional[str], note: str) -> int:
        """A run's lifecycle under its store lock: never over a close; `started` -> running, a failure -> once; says what it did."""
        rid, campaign, ev_id, obj = str(m["run_id"]), str(m["campaign"]), str(m["eval_id"]), str(m["object"])
        lock, how = _run_lock(m.get("family"), obj, wave_root=wave_root, tag=tag, dry_run=dry_run)   # type: ignore[arg-type]
        head = f"{ev_id} {obj} {rid[:40]}…"
        if lock is None:
            if ev == "completed":                                   # a note, no write: the run's last word is read as it stands
                lock = contextlib.nullcontext()
            else:
                _say(f"REFUSED {head}: `{'running' if ev == 'started' else ev}` for {who} — {how}; {LOCK_RULE}; nothing filed"); return 2
        try:
            with lock:
                ct = close_terminal_of(spool, campaign, rid)
                if ev == "started":
                    if ct:
                        _say(f"not filed {head}: `running` for {who} — a close of this run stands (`{ct}`); {ARRAY_RULE}; the task's start is on the job log alone ({how})"); return 0
                    if dry_run:
                        _say(f"DRY-RUN {head}: would file run_state running for {who} ({how})"); return 0
                    EC.state("running", run_id=rid, job=this_job, node=node, scheduler_state=sched, spool=spool, script=SCRIPT)
                    _say(f"running {head}: filed for {who} — no close has decided this run ({how})"); return 0
                if ev == "completed":
                    _say(f"note {head}: {who} exited clean after its close ran; the run reads `{last_run_state(spool, campaign, rid) or 'unfiled'}`; the exit is recorded on the task"); return 0
                if ct:
                    _say(f"not filed {head}: `{ev}` for {who} — a close of this run stands (`{ct}`); {ARRAY_RULE}; the task's {ev} is recorded on the task "
                         f"({d}/{task}.json) and the gate reads each task's shard through its coverage probe ({how})"); return 0
                last = last_run_state(spool, campaign, rid)
                if last in ARRAY_TASK_FAILURES:
                    _say(f"skip {head}: already `{last}` (a task's exit); nothing filed twice for {who}"); return 0
                if dry_run:
                    _say(f"DRY-RUN {head}: would file run_state {ev} ({sched}) for {who} — no close has decided this run ({how})"); return 0
                _file_task_terminal(spool=spool, push_script=push_script, campaign=campaign, rid=rid, event=ev, scheduler_state=sched, producer=producer, note=note)
                _say(f"{ev} {head}: filed for {who} ({sched}; {how}) — no close has decided this run"); return 0
        except StoreCloseLockBusy as exc:
            _say(f"REFUSED {head}: {exc}; nothing filed for {who}"); return 2

    if not dry_run:
        os.makedirs(d, exist_ok=True)
    alock = contextlib.nullcontext() if dry_run else StoreCloseLock(d, wait_s=ARRAY_LOCK_WAIT_S, lock_path=os.path.join(d, ARRAY_TASKS_LOCK_FILE))
    try:
        with alock:
            note = (f"{who} exited {event}" + (f" rc={rc}" if rc is not None else "") + (f" ({scheduler_state})" if scheduler_state else "")
                    + f"; the task's exit, filed on the shared run by {SCRIPT} because no close had decided it; {ARRAY_RULE}")
            for m in mine:
                worst = max(worst, settle(m, event, scheduler_state, note))
            if event not in ARRAY_TASK_EXITS:
                return worst
            rec = {"array_job_id": str(array_job), "job_id": this_job, "job_name": job_name, "task": task, "event": event, "scheduler_state": scheduler_state,
                   "rc": rc, "utc": utc_now(), "pending": os.path.abspath(pending_path), "code_sha": code_sha,
                   "runs": [{k: m[k] for k in ("run_id", "eval_id", "object", "campaign", "family")} for m in mine]}
            p = os.path.join(d, f"{task}.json")
            if dry_run:
                _say(f"DRY-RUN {who}: would record the task's exit at {p}: {json.dumps({k: rec[k] for k in ('event', 'scheduler_state', 'rc')}, sort_keys=True)}")
            else:
                prior = os.path.isfile(p)
                tmp = f"{p}.tmp.{os.getpid()}"
                with open(tmp, "w", encoding="utf-8") as fh:
                    json.dump(rec, fh, indent=1, sort_keys=True); fh.write("\n"); fh.flush(); os.fsync(fh.fileno())
                os.replace(tmp, p)
                _say(f"recorded {who}: exit {event}" + (f" rc={rc}" if rc is not None else "") + f" at {p}" + (" (replacing an earlier record of this task — a requeued attempt)" if prior else ""))
            have, bad = task_outcomes(d)
            if dry_run:
                have[task] = rec
            missing = [t for t in tasks if t not in have]
            if bad:
                _say(f"array {array_job}: {len(bad)} outcome file(s) unreadable or not an exit ({', '.join(bad)}) — not counted as exited")
            if missing:
                _say(f"array {array_job}: {len(have)} of {len(tasks)} task(s) have exited; waiting on {', '.join(str(t) for t in missing)} — the last task to exit reconciles the shared runs")
                return worst
            # every task has exited: reconcile the array's runs — every run any task registered, with the tasks that failed
            by_run: Dict[str, Dict[str, object]] = {}
            for t in sorted(have):
                o = have[t]
                for ru in (o.get("runs") or []):
                    rid = str((ru or {}).get("run_id") or "")
                    if not rid:
                        continue
                    e = by_run.setdefault(rid, {"meta": dict(ru), "tasks": [], "failed": []})
                    e["tasks"].append(t)                                        # type: ignore[attr-defined]
                    if o.get("event") in ARRAY_TASK_FAILURES:
                        e["failed"].append((t, str(o.get("event")), str(o.get("scheduler_state") or str(o.get("event")).upper()), o.get("rc")))   # type: ignore[attr-defined]
            n_failed = sum(1 for t in have if have[t].get("event") in ARRAY_TASK_FAILURES)
            _say(f"array {array_job}: every task has exited ({len(tasks)} task(s): {', '.join(str(t) for t in tasks)}; {n_failed} failed) — reconciling {len(by_run)} shared run(s)"
                 + (" (DRY-RUN: nothing filed)" if dry_run else ""))
            for rid in sorted(by_run):
                e = by_run[rid]; meta = dict(e["meta"]); failed = list(e["failed"])  # type: ignore[arg-type]
                m = {"run_id": rid, "eval_id": str(meta.get("eval_id") or ""), "object": str(meta.get("object") or ""),
                     "campaign": str(meta.get("campaign") or identity.parse_run_id(rid)["campaign"]), "family": meta.get("family")}
                head = f"{m['eval_id']} {m['object']} {rid[:40]}…"
                lock, how = _run_lock(m.get("family"), str(m["object"]), wave_root=wave_root, tag=tag, dry_run=dry_run)   # type: ignore[arg-type]
                if lock is None:
                    _say(f"REFUSED {head}: the array's reconciliation — {how}; {LOCK_RULE}; nothing filed for this run"); worst = 2; continue
                try:
                    with lock:
                        ct = close_terminal_of(spool, str(m["campaign"]), rid)
                        if ct:
                            tail = (f"{len(failed)} task(s) failed ({', '.join(str(f[0]) for f in failed)}) — recorded on the tasks; the lifecycle is the close's"
                                    if failed else "no task failed")
                            _say(f"stands {head}: `{ct}` by a close; {tail} ({how})"); continue
                        last = last_run_state(spool, str(m["campaign"]), rid)
                        if not failed:
                            _say(f"stays {head}: no close decided this run and no task failed — it reads `{last or 'unfiled'}` as its close left it "
                                 f"(a refused close's `running` is the reclose's recovery) ({how})"); continue
                        if last in ARRAY_TASK_FAILURES:
                            _say(f"stands {head}: `{last}` (a task's exit) — {len(failed)} task(s) failed ({', '.join(str(f[0]) for f in failed)}); nothing filed twice ({how})"); continue
                        t0, ev0, sched0, rc0 = failed[0]
                        note = (f"array {array_job}: every task exited; {len(failed)} of {len(tasks)} task(s) failed — "
                                + ", ".join(f"task {f[0]} {f[1]}" + (f" rc={f[3]}" if f[3] is not None else "") for f in failed)
                                + f"; no close decided this run, so its lifecycle is the array's reconciliation (the first failing task's scheduler state), filed by {SCRIPT} ({ARRAY_RULE})")
                        if dry_run:
                            _say(f"DRY-RUN {head}: would file run_state {ev0} ({sched0}) — {len(failed)} task(s) failed, no close decided this run ({how})"); continue
                        _file_task_terminal(spool=spool, push_script=push_script, campaign=str(m["campaign"]), rid=rid, event=ev0, scheduler_state=sched0, producer=producer, note=note)
                        _say(f"{ev0} {head}: filed by the array's reconciliation ({sched0}; task {t0}" + (f" rc={rc0}" if rc0 is not None else "") + f"; {len(failed)} failed task(s)) — "
                             f"no close decided this run; it read `{last or 'unfiled'}` ({how})")
                except StoreCloseLockBusy as exc:
                    _say(f"REFUSED {head}: {exc}; the array's reconciliation filed nothing for this run"); worst = 2
            return worst
    except StoreCloseLockBusy as exc:
        _say(f"REFUSED: {who}: the array lock could not be taken ({exc}) — nothing filed, nothing recorded"); return 2


# ----------------------------------------------------------------------- the RESULT hook's refusal note, under the fence
REFUSAL_RULE = "a refused close's `running` note never reverses a close of the run, nor a task's exit (round-3 Major on cb9ad5267)"


def refusal_notes(pending_path: str, *, note: str, spool: str, wave_root: Optional[str], tag: Optional[str], node: Optional[str] = None,
                  dry_run: bool = False) -> int:
    """Under the fence: the RESULT hook's `running` note for the runs a REFUSED close listed in
    <pending>.refused_run_ids.json (write_refused_run_ids). Before, common.sh's f22_registry_closer_refused filed it through the bare state
    emitter, per run id, AFTER the closer had released the store lock — so a climb task whose close refused an incomplete store (the inspection
    at the top of close_job's loop runs outside the lock), a peer task then closing the completed store and the store512 retiring the shared
    run, and this task's hook stating `running` last, folded the retired run `running` and CURRENT again: reducer.lifecycle_regresses ranks the
    progression alone, a lateral `superseded` has no rank, so `running` applied over it, and v_run_current reads the lifecycle (the run table's
    superseded_by_run_id is identity, never set by the record) — the duplicate-registration retirement undone by the task that closed nothing.
    Now every listed run is decided under the lock its closes take (_run_lock: a BoN run's store close lock — the same decision a task's
    `started` takes): the note is filed only when NO close has decided the run (close_terminal_of: generated / superseded / cancelled,
    whichever job's close wrote it) and the run's last word is not a task's exit (`failed` / `cancelled`, which the array's reconciliation
    owns); a run the pending file does not register is never stated for (the list is this job's); a BoN run whose lock cannot be named is
    refused by name. The note itself is the hook's (`closer REFUSED rc=… under …: <the first REFUSED line>`), filed through the state emitter
    as before (its own hardware capture, evidence job_log). Returns 0, or 2 on a refusal (nothing filed for what it refused). --dry-run says
    and writes nothing."""
    try:
        with open(pending_path, encoding="utf-8") as fh:
            doc = json.load(fh)
    except (OSError, ValueError) as exc:
        _say(f"REFUSED: {pending_path} is unreadable ({type(exc).__name__}: {exc}); no refusal note filed"); return 2
    runs = doc.get("runs") if isinstance(doc, dict) else None
    if not isinstance(runs, list) or not runs:
        _say(f"REFUSED: {pending_path} names no `runs` (the job registered nothing); no refusal note filed"); return 2
    this_job = str(doc.get("job_id") or os.path.basename(pending_path)[:-5])
    side = refused_run_ids_path(pending_path)
    if not os.path.isfile(side):
        _say(f"no refused-run list at {side} — stating nothing (D2100 (b): never the job's whole run set)"); return 0
    try:
        with open(side, encoding="utf-8") as fh:
            sdoc = json.load(fh)
        ids = [r for r in ((sdoc or {}).get("refused_run_ids") or []) if isinstance(r, str) and r]
    except (OSError, ValueError, AttributeError) as exc:
        _say(f"REFUSED: the refused-run list {side} is unreadable ({type(exc).__name__}: {exc}); no refusal note filed"); return 2
    if str((sdoc or {}).get("job_id") or "") not in ("", this_job):
        _say(f"REFUSED: the refused-run list {side} names job {sdoc.get('job_id')!r}, not this pending file's {this_job!r}; no refusal note filed"); return 2
    mine: Dict[str, Dict[str, object]] = {}
    for r in runs:
        rid = str(r.get("run_id") or ""); route = r.get("route") or {}; plan = r.get("plan") or {}
        if rid:
            mine[rid] = {"eval_id": str(route.get("eval_id") or plan.get("eval_id") or ""), "object": str(route.get("object") or ""),
                         "campaign": str(plan.get("campaign") or identity.parse_run_id(rid)["campaign"]), "family": family_of(r)}
    stated = held = foreign = 0; worst = 0
    for rid in ids:
        m = mine.get(rid)
        if m is None:
            _say(f"not filed {rid[:40]}…: the refused-run list names a run {pending_path} does not register — stated for this job's registered runs alone (D2100 (b))")
            foreign += 1; continue
        campaign, ev_id, obj = str(m["campaign"]), str(m["eval_id"]), str(m["object"])
        head = f"{ev_id} {obj} {rid[:40]}…"
        lock, how = _run_lock(m.get("family"), obj, wave_root=wave_root, tag=tag, dry_run=dry_run)   # type: ignore[arg-type]
        if lock is None:
            _say(f"REFUSED {head}: the closer's refusal note — {how}; {LOCK_RULE}; nothing filed"); worst = 2; continue
        try:
            with lock:
                ct = close_terminal_of(spool, campaign, rid)
                if ct:
                    _say(f"not filed {head}: `running` (the closer's refusal note) — a close of this run stands (`{ct}`); {REFUSAL_RULE}; the refusal is on the job log, "
                         f"the RESULT surface and the reclose recipe ({how})"); held += 1; continue
                last = last_run_state(spool, campaign, rid)
                if last in ARRAY_TASK_FAILURES:
                    _say(f"not filed {head}: `running` (the closer's refusal note) — the run reads `{last}` (a task's exit); {REFUSAL_RULE} ({how})"); held += 1; continue
                if dry_run:
                    _say(f"DRY-RUN {head}: would file run_state running with the closer's refusal note — no close has decided this run ({how})"); stated += 1; continue
                EC.state("running", run_id=rid, job=this_job, node=node, evidence_kind="job_log", note=note, spool=spool, script=SCRIPT)
                _say(f"running {head}: the closer's refusal note filed — no close has decided this run ({how})"); stated += 1
        except StoreCloseLockBusy as exc:
            _say(f"REFUSED {head}: {exc}; the refusal note was not filed for it"); worst = 2
    _say(f"state running: stated for {stated} refused run(s) of job {this_job} ({len(ids)} listed"
         + (f"; {held} not filed — a close or a task's exit stands" if held else "") + (f"; {foreign} not registered by this pending file" if foreign else "")
         + "; the refused-run list; D2100 (b))" + (" (DRY-RUN: nothing filed)" if dry_run else ""))
    return worst


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pending", required=True, help="the job's pending registration ($SHARED/registry/pending/<jobid>.json)")
    ap.add_argument("--wave-root", default=None, help="the wave root ($WAVE_ROOT: <shared>/exp/sdfbase_full22_native)")
    ap.add_argument("--tag", default=None, help="the model tag (<key>[_<view>]) the job served")
    ap.add_argument("--job-name", default=None, help="the Slurm job name (its _t<N> names the trial); default: the pending file's job_name")
    ap.add_argument("--trial", type=int, default=None, help="the sampled trial (overrides the job name)")
    ap.add_argument("--nshards", type=int, default=NSHARDS_DEFAULT, help=f"the static job's shard plan (target_sampled.sbatch NSHARDS; default {NSHARDS_DEFAULT})")
    ap.add_argument("--private-root", default=None, help="static only, explicit: jailbreak_run's --private-root")
    ap.add_argument("--private-run", default=None, help="static only, explicit: the config's private_run")
    ap.add_argument("--spool", default=None, help="the record spool (default $SISTEER_REGISTRY_SPOOL, else the host's resolved spool)")
    ap.add_argument("--push-script", default=None, help="the push script (default $SISTEER_REGISTRY_PUSH)")
    ap.add_argument("--code-sha", default=None)
    ap.add_argument("--evals", default=None, help="comma list: close only these evals' runs")
    ap.add_argument("--misalign-root", default=None, help="the misalign runner's root for this wave (default <dirname of --wave-root>/misalign/<tag>)")
    ap.add_argument("--pad-root", default=None, help="the padded static ladder's private root (default <wave-root>/static_padding/transcripts_results; "
                                                     "stores are pad_static_<model>_p<pad>/<eval>/<object>/shardNN)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--successor-token-cap", type=int, default=None, metavar="CAP", help="D1900 / D1904 (U0STATICSUCC): re-register the pending file's STATIC runs from their EXISTING store at this drawn cap "
                                                                                           "(the D1242 key, pass label below, cluster_job_id = the producing job), close the successors from the store, supersede the old rows")
    ap.add_argument("--successor-pass-label", default="base_cap", help="the successor's pass_label (base_cap for the base rung)")
    ap.add_argument("--successor-config", default=None, metavar="YAML", help="the config the successor registers from (default configs/sdfbase_full22_<tag>_sampled_t<T>.yaml of this checkout)")
    ap.add_argument("--pairs-out", default=None, metavar="TSV", help="append the retired -> successor id pairs (and the cancelled orphan ids) here")
    ap.add_argument("--cancel-orphan", action="append", default=[], metavar="PENDING=EVIDENCE", help="D1904 Q3: a FAILED attempt's store-less pending file and the scheduler's terminal state to cite; its static runs are filed cancelled")
    ap.add_argument("--objects", default=None, help="successor mode: relabel / cancel only these objects (comma list of config cell ids; default every static run of the pending file)")
    ap.add_argument("--ruled-out-shards", default=None, metavar="N,…", help="D1168: the shards a RULING excluded from this job's cells (e.g. 5,6,7,8 — the other group's); with --ruled-out-objects and "
                                                                           "--ruling-evidence-uri, a multi-turn run whose missing shards are exactly these closes `cancelled` with the present shards' counts")
    ap.add_argument("--ruled-out-objects", default=None, metavar="CELL,…", help="D1168: the cells whose runs the ruling excludes (e.g. none); a run of any other object with missing shards stays running")
    ap.add_argument("--ruling-evidence-uri", default=None, metavar="URI", help="D1168: the ruling record (rulings/added/<file>.jsonl) — REQUIRED with the two flags above; a run is never cancelled on a bare flag")
    ap.add_argument("--ruling-note", default=None, help="D1168: an optional note appended to the cancelled state's note")
    ap.add_argument("--audit-nofit", default=None, metavar="STORE", help="read-only: run nofit_slot_audit over one BoN store against its plan universe and print it")
    ap.add_argument("--expected-goal-ids", default=None, metavar="JSON", help="multiturn, CROSS-CHECK only: an ids file --emit-expected-goal-ids wrote; the eval set is always "
                                                                           "recomputed by the active checkout's resolver (an interpreter that cannot import it refuses, file or no file)")
    ap.add_argument("--emit-expected-goal-ids", default=None, metavar="OUT", help="multiturn: resolve the pending file's registered config's eval goal ids HERE and write the "
                                                                                 "cross-check artifact (config + resolver identity + ids + hashes); nothing else is done")
    ap.add_argument("--array-task-event", default=None, choices=ARRAY_TASK_EVENTS, metavar="EVENT",
                    help="an ARRAY TASK's lifecycle event for the shared runs its pending file registered (started | completed | failed | cancelled): recorded on the "
                         "task, never filed over a close of the run; the last task to exit reconciles the array (common.sh f22_registry_state / f22_registry_result)")
    ap.add_argument("--array-job", default=None, help="array mode: the array's job id (SLURM_ARRAY_JOB_ID) — names <pending dir>/<id>.tasks/")
    ap.add_argument("--array-task", type=int, default=None, help="array mode: this task's index (SLURM_ARRAY_TASK_ID)")
    ap.add_argument("--array-tasks", default=None, metavar="SET", help="array mode: the array's task set as sbatch spells it (SLURM_ARRAY_TASK_MIN-MAX[:STEP]; `0-3`)")
    ap.add_argument("--scheduler-state", default=None, help="array mode: the scheduler state the event cites (FAILED:rc=2, PREEMPTED, RUNNING)")
    ap.add_argument("--rc", type=int, default=None, help="array mode: the task's exit code (default: the `rc=N` in --scheduler-state, if any)")
    ap.add_argument("--node", default=None, help="array mode / refusal note: the job's node (SLURMD_NODENAME), for the `running` patch")
    ap.add_argument("--refusal-note", default=None, metavar="NOTE",
                    help="the RESULT hook's refusal note (common.sh f22_registry_closer_refused): state `running` with NOTE for the runs <pending>.refused_run_ids.json names, "
                         "each under its store close lock and never over a close of the run or a task's exit (round-3 Major on cb9ad5267); nothing else is done")
    a = ap.parse_args(argv)
    if bool(a.private_root) != bool(a.private_run):
        _say("REFUSED: --private-root and --private-run go together"); return 2
    spool = a.spool or os.environ.get(emit.SPOOL_ENV) or emit.spool_root()
    if not spool:
        _say("REFUSED: no record spool resolves (pass --spool or set the registry spool variable)"); return 2
    evals = [x.strip() for x in a.evals.split(",") if x.strip()] if a.evals else None
    why_r, ruled_out = ruled_out_spec(a.ruled_out_shards, a.ruled_out_objects, a.ruling_evidence_uri, a.ruling_note)
    if why_r:
        _say(f"REFUSED: {why_r}"); return 2
    _bind_pending_dir(a.pending)                        # closer-fix-3 (3): the ids emitter resolves the registered config beside the pending file too
    if a.refusal_note is not None:
        return refusal_notes(a.pending, note=a.refusal_note, spool=spool, wave_root=a.wave_root, tag=a.tag, node=a.node, dry_run=a.dry_run)
    if a.array_task_event is not None:
        if a.array_task is None or a.array_job is None or a.array_tasks is None:
            _say("REFUSED: --array-task-event needs --array-job, --array-task and --array-tasks (the scheduler's SLURM_ARRAY_JOB_ID / _TASK_ID / _TASK_MIN-MAX); nothing filed, nothing recorded")
            return 2
        return array_task_event(a.pending, event=a.array_task_event, array_job=a.array_job, task=a.array_task, tasks_spec=a.array_tasks, spool=spool,
                                push_script=a.push_script or os.environ.get(emit.PUSH_SCRIPT_ENV), code_sha=a.code_sha, wave_root=a.wave_root, tag=a.tag,
                                scheduler_state=a.scheduler_state, rc=a.rc, node=a.node, dry_run=a.dry_run)
    if a.successor_token_cap is not None:
        return successor_relabel(a.pending, spool=spool, push_script=a.push_script or os.environ.get(emit.PUSH_SCRIPT_ENV), code_sha=a.code_sha, wave_root=a.wave_root,
                                 tag=a.tag, job_name=a.job_name, trial=a.trial, nshards=a.nshards, token_cap=a.successor_token_cap, pass_label=a.successor_pass_label,
                                 config=a.successor_config, pairs_out=a.pairs_out, cancel=list(a.cancel_orphan), evals=evals, dry_run=a.dry_run,
                                 objects=[x.strip() for x in a.objects.split(",") if x.strip()] if a.objects else None)
    if a.emit_expected_goal_ids:
        with open(a.pending, encoding="utf-8") as fh:
            runs = (json.load(fh) or {}).get("runs") or []
        if not runs:
            _say(f"REFUSED: {a.pending} names no runs"); return 2
        why = write_expected_goal_ids(runs[0], a.emit_expected_goal_ids, repo_root=REPO_ROOT)
        if why:
            _say(f"REFUSED: {why}"); return 2
        return 0
    if a.audit_nofit:
        served = _json(os.path.join(a.audit_nofit, "bon_served_run.json")) or {}
        why, keys, ufacts = bon_plan_universe(a.audit_nofit, served, repo_root=REPO_ROOT)
        if why:
            _say(f"REFUSED: {why}"); return 2
        print(json.dumps({"store": a.audit_nofit, "plan": ufacts, "audit": nofit_slot_audit(a.audit_nofit, keys)}, indent=1, sort_keys=True, default=str)); return 0
    out = close_job(a.pending, spool=spool, push_script=a.push_script or os.environ.get(emit.PUSH_SCRIPT_ENV), code_sha=a.code_sha, wave_root=a.wave_root,
                    tag=a.tag, job_name=a.job_name, trial=a.trial, nshards=a.nshards, private_root=a.private_root, private_run=a.private_run,
                    evals=evals, dry_run=a.dry_run, misalign_root=a.misalign_root, pad_root=a.pad_root, expected_goal_ids=a.expected_goal_ids,
                    ruled_out=ruled_out)
    return 2 if out["refused"] else 0


if __name__ == "__main__":
    sys.exit(main())
