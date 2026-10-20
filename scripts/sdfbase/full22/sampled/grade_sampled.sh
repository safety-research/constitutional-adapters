#!/bin/bash
# full22 SAMPLED re-run: the detached post-pipeline on the grading host — grade the mirrored sampled stores.
#
#   grade_sampled.sh <static|controls-pass1|controls-pass2|all> [options]
#     --dest DIR            analysis root (default $F22S_DEST, else the script's built-in analysis path)
#     --worktree DIR        the checkout whose scripts run (default: the one this script lives in)
#     --python BIN          interpreter (default: <worktree>/.venv/bin/python, else <worktree>/../full22/.venv/bin/python)
#     --configs-dir DIR     where configs/sdfbase_full22_<key>_sampled_t<T>[_cap8192].yaml live (default <worktree>/configs)
#     --keys k1,k2          (default the four main-view targets; every other TAG of make_sampled_configs.SAMPLED_VIEWS is
#                           accepted too — the instruct-direct gemma4e4b_it / qwen35_4b_it and the strength frontier's 24
#                           per-dose TAGs <key>_d<dose> / <key>_it_d<dose> (2026-09-10): their private_runs
#                           sdfbase_full22_<TAG>_sampled_t<T>, their 512 roots controls_cap512/<TAG>_t<T> and the cell list of
#                           configs/sdfbase_full22_<TAG>_sampled_t<T>.yaml follow the TAG)      --trials 0,1
#     --parallel N          units in flight per phase (default 4)
#     --concurrency N       haiku grader in-flight ceiling (default 150)   --judge-concurrency N  sonnet judge gate (default 100)
#     --allow-over-ceiling "<reason>"  LANE CEILING override: before anything is planned or launched the script
#                           computes the configured judge ceiling = parallel x judge-concurrency for --org and REFUSES (exit 2, also
#                           under --plan) when it exceeds 500 on lowprio or 100 on highprio, or when --parallel > 1 on highprio (one
#                           unit at a time) — the hard in-flight limits. The override must name a reason; that reason is
#                           recorded in the plan JSON and in GRADING_RUN.json for that run. Every run prints one line
#                           `ceiling: org=<org> parallel=<p> judge_concurrency=<j> configured=<p*j> limit=<500|100>`. A static-only
#                           launch (haiku grader, no judge) prints the line and is not gated by it.
#     --cell-parallel N     judge cells at a time (default 6)
#     --seed-root DIR       greedy grade roots to seed the haiku caches from (default <dest>/..; --no-seed disables)
#     --grader-arg ARG      appended to every grade_static_sampled.py call (repeatable; e.g. --grader-arg --nshards=9)
#     --cells c1,c2         controls-pass2 only: judge ONLY these cells (cell_id) of each tag — the 512 flip root per cell via --match,
#                           (the rung steps are per (cell, CONTROL): each control at ITS final-rung store, one step per (cell, rung))
#                           and only their final-rung stores (the four instruct-direct constitution cells are held for a
#                           batch-64 re-fit; the it lanes' pass 2 runs on none + prompted__safety_refusal alone)
#     --org lowprio|highprio  the Anthropic pool EVERY unit of this launch uses: the key item the secret-manager CLI fetches inside
#                           the unit AND the --org the judge/grader records (default lowprio; highprio only when it is explicitly
#                           allotted — one pass-2 unit moved there so Low Prio stays at 500 in flight)
#     --judge-arg ARG       appended to every served_capability_judge.py call, pass 1 and pass 2 (repeatable; e.g.
#                           --judge-arg --no-resume-ok to re-judge blocks already filed ok / refused-ceiling — the
#                           store-level resume still skips every item already filed, so no row is filed twice)
#     --allow-partial       run the READY jobs of a phase and skip the refused ones (default: a phase with any refusal refuses)
#     --stamp S             unit-name stamp (default now, UTC)       --unit-prefix P (default f22s-grade)
#     --give-up-s N         orchestrator wait bound per phase (default 172800)
#     --plan                print the prechecks + the exact commands; launch nothing, write nothing
#     --plan-out DIR        with --plan: also keep the plan JSON (unit scripts included) per phase under DIR
#     --detach              re-exec this orchestrator as a systemd-run --user unit (log under <dest>/logs/)
#     --no-final-rung       skip the post-phase refresh of <dest>/report/analysis/final_rung.json (default: after every phase that
#                           launched units, `analysis_sampled.py final-rung --configs-dir <the phase's roster> --allow-pending` rewrites the index the paper reader keys to —
#                           it never fails the phase)   --final-rung-targets LIST (default the paper's ten: the six + the four llama8b views)
#
# Phases (each REFUSES to start when its inputs are incomplete, naming the exact missing pieces):
#   static          grade_static_sampled.py per (key, trial) BASE root, then per cap8192 root that exists (seeded from the
#                   base trial's cache + the greedy caches for the key); one systemd-run --user unit per root, --parallel at once.
#   controls-pass1  per (TAG, trial) 512 root: seed_private_cache.py (a seeded private copy of the shared sonnet cache, foreign
#                   rows scrubbed) -> served_capability_judge.py on controls_sampled/controls_cap512 with CAPEVAL_JUDGE_CACHE=<copy>
#                   -> merge_cache_back.py. Refuses unless the pristine flip copy of the 512 root already exists (pass 1 appends
#                   label rows to the mirror; the flip copy is pass 2's live re-ask and must predate pass 1).
#   controls-pass2  per (TAG, trial): ONE FRESH EMPTY sqlite cache PER ROOT — flip_pass/cap512/<TAG>_t<T> and each
#                   flip_pass/cap<R>/<TAG>_t<T> — the judge invocations back-to-back in one unit (cap512 first, then every
#                   final-rung store with its own --match/--controls), --run-started-at recorded per invocation, and NO merge
#                   back to the shared cache. A pre-existing pass-2 cache beside an unjudged root is refused (a stale cache would
#                   serve verdicts; the flip floor must be live); beside a partly judged root it is a resume.
#   all             static -> controls-pass1 -> controls-pass2, stopping at the first refusal / failure.
# GRADING_RUN.json under --dest records, per phase: unit names, start/end, exit codes, the cache paths, the org (lowprio).
#
# Keys: the sonnet judge units export ANTHROPIC_API_KEY from `$SISTEER_SECRET_CMD lowprio` INSIDE the unit (never in an env file, argv,
# a log or on disk); the haiku grader fetches its own key in-process the same way. Low Prio only; the graders throttle down on
# 429s and never switch org. served_capability_judge.py's --targets takes any subdir name under the controls root and records it
# as `target` (it validates nothing about the name), so the sampled `<TAG>_t<T>` dirs are addressed as-is.
set -u
PHASE=${1:-}; shift || true
case "$PHASE" in static|controls-pass1|controls-pass2|all) ;; -h|--help|"") sed -n '2,40p' "$0"; exit 0;; *) echo "unknown phase $PHASE" >&2; exit 2;; esac
HERE=$(cd "$(dirname "$0")" && pwd)
WT=$(cd "$HERE/../../../.." && pwd)
# env-mode secrets (docs/secrets.md): secrets_unit_flags adds NAME-ONLY --setenv imports to every
# systemd-run below under SISTEER_SECRETS_MODE=env (under op mode, the secret-manager CLI's path), and the generated unit scripts check the ONE
# ANTHROPIC_API_KEY instead of fetching the org's vault item. SISTEER_SECRETS_MODE is exported so the planner sees it.
# wif mode: the flags carry `--setenv=SISTEER_SECRETS_MODE=wif` into every unit (a detached unit inherits
# nothing; its vault reads the mode off its own environment) and the generated unit scripts re-assert the mode and only SCRUB the
# inherited ANTHROPIC_* — no key exists, the secret-manager CLI is never consulted. A non-op mode with the library unreadable is refused.
[ -r "$WT/reproduce/lib/secrets.sh" ] && source "$WT/reproduce/lib/secrets.sh"
if [ "${SISTEER_SECRETS_MODE:-op}" != op ] && ! type secrets_unit_flags >/dev/null 2>&1; then
  echo "REFUSED: SISTEER_SECRETS_MODE=$SISTEER_SECRETS_MODE but $WT/reproduce/lib/secrets.sh is not readable: the detached units inherit nothing and the library is what carries the mode (and, under env, the key) into them" >&2; exit 2
fi
type secrets_unit_flags >/dev/null 2>&1 || secrets_unit_flags(){ :; }
type secrets_unit_flags_cleanup >/dev/null 2>&1 || secrets_unit_flags_cleanup(){ :; }   # unlinks the systemd < 247 EnvironmentFile after start
export SISTEER_SECRETS_MODE="${SISTEER_SECRETS_MODE:-op}"
DEST=${F22S_DEST:-}          # default after the options: the run-state tree under the site state root ($STATE)
PY=""; CONFIGS=""; KEYS=gemma4e4b,qwen35_4b,nemotron3nano4b,glm46vflash; TRIALS=0,1
PARALLEL=4; CONC=150; JCONC=100; CELLPAR=6; SEED_ROOT=""; NOSEED=0; GRADER_ARGS=(); JUDGE_ARGS=(); PARTIAL=0; ALLOW_OVER=""
STAMP=$(date -u +%Y%m%d-%H%M%S); PREFIX=f22s-grade; GIVE_UP=172800; PLAN=0; PLAN_OUT=""; DETACH=0; ORG=lowprio; CELLS_FILTER=""
FINAL_RUNG=1; FR_TARGETS=gemma4e4b,qwen35_4b,nemotron3nano4b,glm46vflash,gemma4e4b_it,qwen35_4b_it,llama8b,llama8b_it,llama8b_shortconst,llama8b_fullconst   # the paper's ten lanes (the four llama8b views: 2026-09-13)
ORIG_ARGS=("$@")
while [ $# -gt 0 ]; do
  case "$1" in
    --dest) DEST=$2; shift 2;;            --worktree) WT=$(cd "$2" && pwd); shift 2;;
    --python) PY=$2; shift 2;;            --configs-dir) CONFIGS=$2; shift 2;;
    --keys) KEYS=$2; shift 2;;            --trials) TRIALS=$2; shift 2;;
    --parallel) PARALLEL=$2; shift 2;;    --concurrency) CONC=$2; shift 2;;
    --judge-concurrency) JCONC=$2; shift 2;; --cell-parallel) CELLPAR=$2; shift 2;;
    --seed-root) SEED_ROOT=$2; shift 2;;  --no-seed) NOSEED=1; shift;;
    --grader-arg) GRADER_ARGS+=("$2"); shift 2;;
    --judge-arg) JUDGE_ARGS+=("$2"); shift 2;;
    --org) ORG=$2; shift 2;;
    --cells) CELLS_FILTER=$2; shift 2;;
    --allow-partial) PARTIAL=1; shift;;   --stamp) STAMP=$2; shift 2;;
    --allow-over-ceiling) ALLOW_OVER=$2; shift 2;;
    --unit-prefix) PREFIX=$2; shift 2;;   --give-up-s) GIVE_UP=$2; shift 2;;
    --plan) PLAN=1; shift;;               --plan-out) PLAN_OUT=$2; shift 2;;
    --detach) DETACH=1; shift;;
    --no-final-rung) FINAL_RUNG=0; shift;;  --final-rung-targets) FR_TARGETS=$2; shift 2;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done
# the default analysis root: the wave's run-state tree under the site state root ($STATE, expanded by sisteer.site from the
# environment, else the site config), refused by name when neither --dest, F22S_DEST nor the site gives one
if [ -z "$DEST" ]; then
  DEST=$(PYTHONPATH="$WT/src${PYTHONPATH:+:$PYTHONPATH}" python3 -m sisteer.site expand '$STATE/romp/candidate-si-refit/run-state/full22/sampled' 2>/dev/null)
  DEST="${DEST/#\~/$HOME}"
  [ -n "$DEST" ] || { echo "grade_sampled.sh: no analysis root: pass --dest, set F22S_DEST, or name the state root in the site config (SISTEER_STATE_ROOT)" >&2; exit 2; }
fi
case "$DEST" in /*) ;; *) DEST=$PWD/$DEST;; esac   # absolute: a --detach unit starts in the worktree, not in this shell's directory
case "$ORG" in lowprio|highprio) ;; *) echo "--org must be lowprio or highprio, got: $ORG" >&2; exit 2;; esac
[ -z "$PY" ] && { if [ -x "$WT/.venv/bin/python" ]; then PY=$WT/.venv/bin/python; else PY=$WT/../full22/.venv/bin/python; fi; }
[ -x "$PY" ] || { echo "python not executable: $PY (pass --python)" >&2; exit 2; }
[ -z "$CONFIGS" ] && CONFIGS=$WT/configs
[ -z "$SEED_ROOT" ] && SEED_ROOT=$(cd "$DEST/.." 2>/dev/null && pwd || echo "$DEST/..")
[ $NOSEED -eq 1 ] && SEED_ROOT=""
say() { echo "[grade-sampled $(date -u +%H:%M:%SZ)] $*"; }

refresh_final_rung() {   # <phase> — the paper reader keys to <dest>/report/analysis/final_rung.json and never globs the cell dirs
  # (the constitution cells were judged for a day while the index still named only the safety cells), so the
  # index is rewritten after EVERY phase that launched units. --allow-pending: other lanes' cells may still be generating. It never
  # changes the phase's exit status; a failure is said and the log named, and the by-hand command is the same line.
  local phase=$1 out=$DEST/report/analysis/final_rung.json log=$DEST/logs/$phase/final_rung-$STAMP.log
  mkdir -p "$(dirname "$out")" "$DEST/logs/$phase"
  # the refresh reads the SAME cell roster the phase graded under (--configs-dir; the served-configs snapshot): without it, the
  # index is regenerated under the worktree's default configs and every cell the roster added since (the gemma4e4b_it itdoc64
  # pair, 2026-09-14) silently drops out of targets + lane_summary until the next roster-aware run
  if (cd "$WT" && PYTHONPATH="$WT/src:$WT/scripts" "$PY" scripts/sdfbase/full22/sampled/analysis_sampled.py final-rung --dest "$DEST" --configs-dir "$CONFIGS" --targets "$FR_TARGETS" --out "$out" --allow-pending > "$log" 2>&1); then
    say "post-phase $phase: final-rung index refreshed -> $out ($(tail -1 "$log" | cut -c1-140))"
  else
    say "post-phase $phase: final-rung index refresh FAILED (log $log); the phase result stands — re-run: analysis_sampled.py final-rung --dest $DEST --configs-dir $CONFIGS --targets $FR_TARGETS --out $out --allow-pending"
  fi
}

# --------------------------------------------------------------------------- the LANE CEILING: refuse BEFORE anything is planned or launched
# The hard limits: <= 500 in flight on Low Prio, at most 100 on High Prio and one unit at a time there. The supplementary
# pass 2 of 2026-09-11 ran 5 units x the then-default 200 = a configured 1,000 on Low Prio because the launch omitted an explicit
# --judge-concurrency; a norm without a guard. Computed here, printed on every run, enforced for the judge phases (the haiku grader's
# --concurrency is not this gate), overridable only with a recorded reason.
case "$ORG" in lowprio) LIMIT=500;; highprio) LIMIT=100;; esac
CONFIGURED=$((PARALLEL * JCONC))
CEIL_LINE="ceiling: org=$ORG parallel=$PARALLEL judge_concurrency=$JCONC configured=$CONFIGURED limit=$LIMIT"
[ -n "$ALLOW_OVER" ] && CEIL_LINE="$CEIL_LINE OVERRIDE: $ALLOW_OVER"
echo "$CEIL_LINE"
JUDGE_PHASE=0; case "$PHASE" in controls-pass1|controls-pass2|all) JUDGE_PHASE=1;; esac
if [ $JUDGE_PHASE -eq 1 ] && [ -z "$ALLOW_OVER" ]; then
  refuse=0
  if [ "$ORG" = highprio ] && [ "$PARALLEL" -gt 1 ]; then echo "REFUSED: parallel $PARALLEL > 1 on highprio (one unit at a time; S-D22 / S-D27)"; refuse=1; fi
  if [ "$CONFIGURED" -gt "$LIMIT" ]; then echo "REFUSED: configured ceiling $CONFIGURED > $LIMIT on $ORG (parallel $PARALLEL x judge-concurrency $JCONC; lower one, or pass --allow-over-ceiling \"<reason>\" on the user's word; S-D27)"; refuse=1; fi
  if [ $refuse -eq 1 ]; then echo "nothing planned, nothing launched"; exit 2; fi
fi

if [ $DETACH -eq 1 ]; then
  mkdir -p "$DEST/logs"
  UNIT=$PREFIX-orch-$PHASE-$STAMP; LOG=$DEST/logs/orchestrator-$PHASE-$STAMP.log
  args=(); for a in ${ORIG_ARGS[@]+"${ORIG_ARGS[@]}"}; do [ "$a" = --detach ] || args+=("$a"); done
  # a user-manager unit inherits none of this shell's variables (F22S_DEST, SISTEER_STATE_ROOT, SISTEER_CONFIG): the root
  # resolved above rides in as the last --dest, so the unit grades the tree this shell resolved
  args+=(--dest "$DEST")
  # the unit gets $HOME/.local/bin AHEAD of this shell's PATH (--setenv): the user manager's own PATH is the bare system one
  # (2026-09-15: forwarding "$PATH" alone is not enough when THIS shell is itself a systemd unit launched by another unit without
  # PATH); the secret-manager CLI itself rides in by absolute path (secrets_unit_flags: --setenv=SISTEER_SECRET_CMD=<path>)
  uflags="$(secrets_unit_flags)"
  # shellcheck disable=SC2086
  if systemd-run --user --unit="$UNIT" --collect --quiet --setenv=PATH="$HOME/.local/bin:$PATH" -p WorkingDirectory="$WT" $uflags \
    /bin/bash -c 'exec "$0" "$@" >> "'"$LOG"'" 2>&1' "$0" "$PHASE" ${args[@]+"${args[@]}"} --stamp "$STAMP"; then sr=0; else sr=$?; fi
  secrets_unit_flags_cleanup "$uflags"        # the systemd < 247 EnvironmentFile is unlinked once the unit has started
  [ "$sr" -eq 0 ] || exit 1
  say "detached: unit=$UNIT log=$LOG"; exit 0
fi

# --------------------------------------------------------------------------- the planner (prechecks + exact commands)
GA_JSON=$("$PY" -c 'import json,sys; print(json.dumps(sys.argv[1:]))' ${GRADER_ARGS[@]+"${GRADER_ARGS[@]}"})
JA_JSON=$("$PY" -c 'import json,sys; print(json.dumps(sys.argv[1:]))' ${JUDGE_ARGS[@]+"${JUDGE_ARGS[@]}"})
plan_phase() {   # plan_phase <phase> <out json>  -> prints the human-readable plan; the JSON carries the unit scripts
  "$PY" - "$1" "$2" "$DEST" "$WT" "$PY" "$CONFIGS" "$KEYS" "$TRIALS" "$CONC" "$JCONC" "$CELLPAR" "$SEED_ROOT" "$GA_JSON" "$STAMP" "$PREFIX" "$JA_JSON" "$ORG" "$CELLS_FILTER" \
          "$PARALLEL" "$CONFIGURED" "$LIMIT" "$ALLOW_OVER" <<'PY'
import json, os, re, shlex, subprocess, sys
from pathlib import Path

(phase, out_json, dest, wt, py, configs, keys, trials, conc, jconc, cellpar, seed_root, ga_json, stamp, prefix, ja_json, org, cells_filter) = sys.argv[1:19]
(parallel, ceil_configured, ceil_limit, allow_over) = sys.argv[19:23]     # the lane ceiling (S-D27): recorded in the plan JSON, then GRADING_RUN.json
CEILING = {"org": org, "parallel": int(parallel), "judge_concurrency": int(jconc), "configured": int(ceil_configured), "limit": int(ceil_limit),
           "allow_over_ceiling": allow_over or None}
CELLS_FILTER = [c for c in cells_filter.split(",") if c] if cells_filter else None   # controls-pass2: judge only these cell_ids
DEST, WT, CONFIGS = Path(dest), Path(wt), Path(configs)
KEYS = [k for k in keys.split(",") if k]; TRIALS = [int(t) for t in trials.split(",") if t != ""]
GRADER_ARGS = json.loads(ga_json)
JUDGE_ARGS = json.loads(ja_json)      # --judge-arg passthrough: every served_capability_judge.py call, pass 1 and pass 2
SAMPLED = "scripts/sdfbase/full22/sampled"; JUDGE = "scripts/sdfbase/full22/judge"
sys.path.insert(0, str(WT / SAMPLED))
import make_sampled_configs as msc   # the ONE scope list (SAMPLED_VIEWS / split_tag): which TAGs exist, and a TAG's key + view
PINNED_N = {"xstest": 450, "or_bench_hard_1k": 1319}      # XSTest v2 (250 safe + 200 unsafe); OR-Bench-Hard-1k
XSTEST_SAFE_N = 250   # a store whose manifest says xstest_subset: safe was generated over the SAFE half ONLY (the sdfbase-llama8b
                      # views, 2026-09-13): the over-refusal denominator is those 250 either way (served_capability_judge.py: on such a
                      # store 'unsafe' is empty by construction and 'all' == 'safe'), so 250 is the complete count for it
CONTROLS = ("xstest", "or_bench_hard_1k")
CACHE_NAME = "live_two_si_grade_cache.claude-haiku-4-5.jsonl"
GREEDY_LANES = {"gemma4e4b": ["grades", "grades_rest"], "qwen35_4b": ["grades", "grades_rest"],
                "nemotron3nano4b": ["grades_nemotron_complete", "grades_rest"], "glm46vflash": ["grades_glm"],
                # the instruct-direct views (lane F): none + the two itdoc cells were graded by the `rest` lane, the prompted
                # baseline by the prompted lane — both caches seed a TAG's sampled grades (a seed only ever serves a
                # byte-identical (goal, response) pair under the same judge, so an extra source is never a wrong verdict)
                "gemma4e4b_it": ["grades_rest", "grades_prompted"], "qwen35_4b_it": ["grades_rest", "grades_prompted"]}
bad_keys = [k for k in KEYS if k not in msc.sampled_tags()]
if bad_keys:
    sys.exit(f"--keys {bad_keys}: not TAGs of the sampled scope (make_sampled_configs.SAMPLED_VIEWS; `--tags` prints it)")


def seed_sources(tag, T, seed_root):
    """The haiku caches that may seed a TAG's trial-T grades (a seed serves only a byte-identical (goal, response) pair under the
    same judge, so every source below is safe and an extra one is never a wrong verdict): (1) the GREEDY lanes' caches — the six
    campaign TAGs' own entries; a per-dose view of the strength frontier has no greedy pass of its own, so it takes its lane's
    (`it_d<dose>` the `<key>_it` entry, `d<dose>` the `<key>` entry) read under the greedy TAG's directory; (2) the SAMPLED caches
    of every sibling TAG of the same key already graded at this trial — the `none` cell is the same served model, the same
    prompts and the same cell-independent seeds on every view of a key, so its draws largely repeat across views."""
    key, view = msc.split_tag(tag)
    greedy_tag = tag if tag in GREEDY_LANES else (f"{key}_it" if view.startswith("it_") else key)
    out = []
    if seed_root:
        for lane in GREEDY_LANES.get(greedy_tag, []):
            p = Path(seed_root) / lane / greedy_tag / CACHE_NAME
            if p.exists():
                out.append(str(p))
    for sib in (msc.tag_of(key, v) for v in msc.scope_views(key)):   # the key's views IN SCOPE (a front-door key's are the roster's; the fix-1 owner follow-up 5)
        p = DEST / "grades_sampled" / f"sdfbase_full22_{sib}_sampled_t{T}" / CACHE_NAME
        if sib != tag and p.exists():
            out.append(str(p))
    return out


LOGS = DEST / "logs" / phase
q = shlex.quote


def read_yaml(p):
    import yaml
    return yaml.safe_load(Path(p).read_text())


def count_lines(p):
    return sum(1 for line in Path(p).read_text().splitlines() if line.strip())


def store_dirs(cells_root: Path):
    """cell_id -> store dir (manifest.json's cell_id; a store without a manifest is reported under its dir name)."""
    out, problems = {}, []
    if not cells_root.is_dir():
        return out, [f"{cells_root}: no cells/ dir"]
    for d in sorted(p for p in cells_root.iterdir() if p.is_dir()):
        mp = d / "manifest.json"
        if not mp.exists():
            problems.append(f"{d.name}: manifest.json missing"); continue
        cid = json.loads(mp.read_text()).get("cell_id")
        if cid in out:
            problems.append(f"two stores for cell {cid}: {out[cid].name}, {d.name}")
        out[cid] = d
    return out, problems


def check_controls_store(d: Path, controls=CONTROLS):
    """The completeness a controls store must show before a judge pass: gen_stats for every control with the pinned n,
    finish_reasons summing to n, the private spool (header + n rows) and its meta sidecar (n rows)."""
    problems = []
    m = json.loads((d / "manifest.json").read_text())
    have = list(m.get("controls") or [])
    missing_ctl = [c for c in controls if c not in have]
    if missing_ctl:
        problems.append(f"{d.name}: manifest controls {have} lack {missing_ctl}")
    gsp = d / "gen_stats.json"
    if not gsp.exists():
        return problems + [f"{d.name}: gen_stats.json missing (generation not finished)"]
    gs = json.loads(gsp.read_text())
    for c in controls:
        if c not in have:
            continue
        blk = gs.get(c)
        if not isinstance(blk, dict):
            problems.append(f"{d.name}/{c}: no gen_stats block"); continue
        n = int(blk.get("n_total", -1))
        want, why = PINNED_N[c], ""
        if c == "xstest" and (m.get("resolved_config") or {}).get("xstest_subset", m.get("xstest_subset")) == "safe":   # the producer records it under resolved_config
            want, why = XSTEST_SAFE_N, " (manifest xstest_subset: safe — the SAFE half only)"
        if n != want:
            problems.append(f"{d.name}/{c}: gen_stats n_total {n} != pinned {want}{why}")
        fr = blk.get("finish_reasons") or {}
        if sum(int(v) for v in fr.values()) != n:
            problems.append(f"{d.name}/{c}: finish_reasons {fr} do not sum to n_total {n}")
        spool = Path(blk["spool"]).name if blk.get("spool") else ("gens_xstest_v2.jsonl" if c == "xstest" else f"gens_{c}.jsonl")
        sp = d / "private" / spool
        if not sp.exists():
            problems.append(f"{d.name}/{c}: spool {spool} missing"); continue
        rows = count_lines(sp) - 1
        if rows != n:
            problems.append(f"{d.name}/{c}: spool {spool} short: {rows} rows != n_total {n}")
        mp = d / "private" / spool.replace(".jsonl", ".meta.jsonl")
        if not mp.exists():
            problems.append(f"{d.name}/{c}: meta sidecar {mp.name} missing")
        elif count_lines(mp) != n:
            problems.append(f"{d.name}/{c}: meta sidecar {mp.name} has {count_lines(mp)} rows != n_total {n}")
    return problems


def judge_state(root: Path):
    """pristine | judged(n cells) for a controls root: judge_summary.json / judge_items.jsonl / label rows in rows.jsonl."""
    n = 0
    for d in sorted(p for p in (root / "cells").iterdir() if p.is_dir()) if (root / "cells").is_dir() else []:
        hit = (d / "judge_summary.json").exists() or (d / "judge_items.jsonl").exists()
        rp = d / "rows.jsonl"
        if not hit and rp.exists():
            for line in rp.read_text().splitlines():
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if str(r.get("metric", "")).startswith(("xst_", "orb_")) or str(r.get("task", "")).startswith("judged/") or r.get("tier") == 3:
                    hit = True; break
        n += int(hit)
    return "pristine" if n == 0 else f"judged({n} cells)"


def unit_script(name: str, steps: list, *, needs_key: bool) -> str:
    """The unit's bash: every step timed into steps.jsonl, EXIT written last; the key (if any) resolved from the vault
    inside the unit and never printed. Stops at the first failing step."""
    log = LOGS / f"{name}.log"; steps_p = LOGS / f"{name}.steps.jsonl"; exit_p = LOGS / f"EXIT-{name}"
    lines = ["#!/bin/bash", "set -u", f"cd {q(str(WT))}", "export PYTHONPATH=src:scripts HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1",
             f"LOG={q(str(log))}; STEPS={q(str(steps_p))}; EXITF={q(str(exit_p))}",
             'run_step() { local name=$1; shift; local s; s=$(date -u +%FT%TZ); echo "[unit] START $name $s"; "$@"; local rc=$?; '
             'local e; e=$(date -u +%FT%TZ); echo "[unit] END $name rc=$rc $e"; '
             'printf \'{"step":"%s","started":"%s","ended":"%s","exit":%d}\\n\' "$name" "$s" "$e" "$rc" >> "$STEPS"; return $rc; }',
             "{", f'echo "[unit] {name} start $(date -u +%FT%TZ) org={org}"', "rc=0"]
    if needs_key:
        mode = os.environ.get("SISTEER_SECRETS_MODE", "op")
        if mode == "env":
            # env mode (D24): EVERY org role resolves to the single documented ANTHROPIC_API_KEY, imported into the unit by
            # name (secrets_unit_flags); required here, never fetched, never printed — the *_LOWPRIO name is op-mode only
            lines.append(': "${ANTHROPIC_API_KEY:?SISTEER_SECRETS_MODE=env: ANTHROPIC_API_KEY must be set in the unit environment (one key serves every org role)}"; unset ANTHROPIC_BASE_URL ANTHROPIC_AUTH_TOKEN')
        elif mode == "wif":
            # wif (round 2 on 4574e103c): NO key exists — sisteer.vault federates inside the unit under the MODE, which the unit's
            # environment carries (secrets_unit_flags) and this line re-asserts; EVERY inherited ANTHROPIC_* name is dropped so none
            # shadows federation (bash's ${!PREFIX@}: the names, no value read -- round 10: the SDK reads ANTHROPIC_CUSTOM_HEADERS
            # whatever credential it is given, so the rule is the prefix, sisteer.vault.WIF_SHELL_SCRUB); the secret-manager CLI is never
            # consulted (the op line below used to render here for every non-env mode)
            lines.append('export SISTEER_SECRETS_MODE=wif; unset "${!ANTHROPIC_@}"')
        else:
            # the org's OWN variable (sisteer.vault.ANTHROPIC_ORGS: lowprio -> ANTHROPIC_API_KEY_LOWPRIO, highprio ->
            # ANTHROPIC_API_KEY), which the judge's sanctioned reader vault.anthropic_client(org) consumes, fetched through the
            # secret-manager CLI the unit's environment names (secrets_unit_flags; unset there is a refusal by name)
            env_var = "ANTHROPIC_API_KEY_LOWPRIO" if org == "lowprio" else "ANTHROPIC_API_KEY"
            lines.append(f'export {env_var}="$("${{SISTEER_SECRET_CMD:?SISTEER_SECRET_CMD must name the secret-manager CLI in the unit environment}}" {org})"; unset ANTHROPIC_BASE_URL ANTHROPIC_AUTH_TOKEN')
    for st in steps:
        env = " ".join(f"{k}={q(v)}" for k, v in (st.get("env") or {}).items())
        argv = " ".join(q(a) if a != "$(date -u +%FT%TZ)" else '"$(date -u +%FT%TZ)"' for a in st["argv"])
        cmd = (f"env {env} " if env else "") + argv
        if st.get("skip_if_exists") and Path(st["skip_if_exists"]).exists():
            lines.append(f'echo "[unit] SKIP {st["name"]}: {q(st["skip_if_exists"])} exists (resume)"')
        else:
            lines.append(f"run_step {q(st['name'])} {cmd} || {{ rc=$?; echo \"[unit] STOP at {st['name']} rc=$rc\"; }}")
            lines.append('[ "$rc" -eq 0 ] || { echo "$rc" > "$EXITF"; exit "$rc"; }')
    lines += ['echo "[unit] done $(date -u +%FT%TZ)"', 'echo 0 > "$EXITF"', "} >> \"$LOG\" 2>&1"]
    return "\n".join(lines) + "\n"


def render_cmd(st):
    env = " ".join(f"{k}={v}" for k, v in (st.get("env") or {}).items())
    argv = " ".join(a if a != "$(date -u +%FT%TZ)" else "<UTC now, at launch>" for a in st["argv"])
    return (env + " " if env else "") + argv


jobs = []          # {"id", "unit", "ready", "problems", "notes", "caches", "steps", "script", "needs_key"}
notes_global = []

# --------------------------------------------------------------------------- static
if phase == "static":
    for key in KEYS:
        for T in TRIALS:
            # every rung the static ladder can climb (ladder_extend: 4096 -> 8192 -> 16384); a rung root that does not exist
            # is simply "nothing escalated that far", never a problem. Before 2026-09-16 the list stopped at cap8192 and a
            # 16384 rung store sat ungraded while the record named it pending (llama8b_d1p5 codeattack, D243).
            for kind in ("base", "cap8192", "cap16384"):
                run = f"sdfbase_full22_{key}_sampled_t{T}" + ("" if kind == "base" else f"_{kind}")
                cfg = CONFIGS / f"{run}.yaml"
                root = DEST / "transcripts_sampled" / run
                jid = f"static/{key}/t{T}/{kind}"
                if kind != "base" and not root.is_dir():
                    notes_global.append(f"NONE {jid}: no root at {root} (nothing escalated yet, or not mirrored)"); continue
                problems, seeds, notes = [], [], []
                if not cfg.exists():
                    problems.append(f"config absent: {cfg}")
                if not root.is_dir():
                    problems.append(f"transcripts root absent: {root} (run mirror_sampled.sh)")
                seeds += seed_sources(key, T, seed_root)
                if seed_root:
                    if kind != "base":
                        base_cache = DEST / "grades_sampled" / f"sdfbase_full22_{key}_sampled_t{T}" / CACHE_NAME
                        if base_cache.exists():
                            seeds.append(str(base_cache))
                        else:
                            problems.append(f"the base trial's cache {base_cache} is absent — grade the base trial first (its verdicts seed the rung's copied-forward rows)")
                    if kind == "cap16384":
                        # the 8192 rung's verdicts seed the rows the 16384 rung copies forward (the same seeding the 8192 rung takes from the base)
                        prev_cache = DEST / "grades_sampled" / f"sdfbase_full22_{key}_sampled_t{T}_cap8192" / CACHE_NAME
                        if prev_cache.exists():
                            seeds.append(str(prev_cache))
                        else:
                            notes.append(f"no cap8192 cache at {prev_cache}: the 16384 rung seeds from the base trial only")
                if not seeds and seed_root:
                    notes.append("no seed caches found for this key under " + seed_root)
                argv = [py, f"{SAMPLED}/grade_static_sampled.py", "--target", key, "--config", str(cfg),
                        "--private-root", str(DEST / "transcripts_sampled"), "--grades-root", str(DEST / "grades_sampled"),
                        "--attacks", "pasttense,codeattack,renellm,grok420", "--concurrency", conc, "--org", org]
                if kind != "base":
                    # a rung root holds ONLY the (family, cell) pairs the static ladder escalated; the ladder record
                    # (the writer's) says which, so the grader grades exactly those and refuses any other shape.
                    # Without the record (not mirrored) the rung is held to every (attack, cell) pair, as before.
                    lrec = DEST / "ladders" / f"{key}_t{T}" / f"static_ladder_t{T}.json"
                    if lrec.is_file():
                        argv += ["--ladder-record", str(lrec)]
                    else:
                        notes.append(f"no static ladder record at {lrec}: the rung is held to every (attack, cell) pair of the config")
                for s in seeds:
                    argv += ["--seed-cache", s]
                argv += GRADER_ARGS
                if not problems:
                    chk = subprocess.run([a for a in argv if a not in ("--seed-cache",) and a not in seeds] + ["--check-only"],
                                         cwd=str(WT), capture_output=True, text=True,
                                         env={**os.environ, "PYTHONPATH": f"{WT}/src:{WT}/scripts", "HF_HUB_OFFLINE": "1"})
                    if chk.returncode != 0:
                        msg = [l for l in chk.stderr.splitlines() if l.startswith("REFUSING")] or chk.stderr.strip().splitlines()[-3:] or [f"exit {chk.returncode}"]
                        problems += msg
                    else:
                        ok = [l for l in chk.stdout.splitlines() if "completeness OK" in l]
                        notes.append(ok[-1].split("] ", 1)[-1] if ok else "check-only ok")
                # the unit (and its log / EXIT file) is named by the KIND: two rung jobs of one (key, trial) must never share a
                # name — a cap16384 unit launched under the cap8192 name while that unit still ran failed to start, and the
                # orchestrator then read the cap8192 unit's exit as the 16384 rung's (2026-09-16 03:39Z, D243)
                unit = f"{prefix}-static-{key}-t{T}{'' if kind == 'base' else '-' + kind}-{stamp}"
                steps = [{"name": "grade", "argv": argv}]
                jobs.append({"id": jid, "unit": unit, "ready": not problems, "problems": problems, "notes": notes,
                             "caches": [{"role": "haiku ScoreCache (append-only; seeded)", "path": str(DEST / "grades_sampled" / run / CACHE_NAME),
                                         "seeds": seeds}],
                             "steps": steps, "needs_key": False, "script": unit_script(unit, steps, needs_key=False)})

# --------------------------------------------------------------------------- controls (shared prechecks)
def explicit_cells(key, T):
    cfg = CONFIGS / f"sdfbase_full22_{key}_sampled_t{T}.yaml"
    if not cfg.exists():
        return None, [f"config absent: {cfg}"]
    return list(read_yaml(cfg)["explicit_cells"]), []


def check_root_complete(root: Path, cells_expected, controls=CONTROLS):
    problems = []
    if not (root / "cells").is_dir():
        return [f"{root}: absent (run mirror_sampled.sh)"]
    stores, p = store_dirs(root / "cells"); problems += p
    for c in cells_expected:
        if c not in stores:
            problems.append(f"{root.name}: no store for cell {c}"); continue
        problems += check_controls_store(stores[c], controls)
    return problems


def ladder_final_rungs(tag):
    """cell -> {control -> final rung} (512 = that control left the ladder at 512) from ladders/<tag>/ladder.json — PER CONTROL: the two
    controls of one cell exit at different rungs and each control's final-rung text lives only in that rung's store (the readers' one rule,
    compare_greedy_sampled.final_rung_store; S-D18 / 80938539). Problems for incomplete entries."""
    p = DEST / "ladders" / tag / "ladder.json"
    if not p.exists():
        return None, [f"ladders/{tag}/ladder.json absent (mirror the ladders; the job may not have written its ladder yet)"]
    lad = json.loads(p.read_text())
    out, problems = {}, []
    for cell, by in (lad.get("cells") or {}).items():
        out[cell] = {}
        for ctl, rec in by.items():
            v = str(rec.get("reached_zero_at"))
            m = re.match(r"^(\d+)", v)
            if v == "incomplete" or not m:
                problems.append(f"{tag}/{cell}/{ctl}: ladder {v!r} — the job did not finish its ladder (resubmit, never infer)"); continue
            out[cell][ctl] = max(512, int(m.group(1)))
    return out, problems


def judge_argv(controls_root: Path, tag: str, cache: Path, sentinel: Path, note: str, match=None, controls=None):
    argv = [py, "scripts/served_capability_judge.py", "--controls-root", str(controls_root), "--targets", tag, "--org", org,
            "--concurrency", jconc, "--cell-parallel", cellpar, "--sentinel-dir", str(sentinel), "--cache-note", note,
            "--run-started-at", "$(date -u +%FT%TZ)"]
    if match:
        argv += ["--match", match]
    if controls:
        argv += ["--controls", ",".join(controls)]
    argv += JUDGE_ARGS
    return {"argv": argv, "env": {"CAPEVAL_JUDGE_CACHE": str(cache)}}


if phase == "controls-pass1":
    for key in KEYS:
        for T in TRIALS:
            tag = f"{key}_t{T}"; jid = f"controls-pass1/{tag}"
            root = DEST / "controls_sampled" / "controls_cap512" / tag
            flip = DEST / "controls_sampled" / "flip_pass" / "cap512" / tag
            cells, problems = explicit_cells(key, T)
            notes = []
            if not root.is_dir():
                notes_global.append(f"NONE {jid}: no 512 root at {root} (not generated / not mirrored yet)"); continue
            if cells is not None:
                problems += check_root_complete(root, cells)
                if not (flip / "cells").is_dir():
                    problems.append(f"pristine flip copy absent: {flip} — run mirror_sampled.sh BEFORE pass 1 (pass 1 appends label rows to the mirror)")
                else:
                    fs = judge_state(flip)
                    if fs != "pristine":
                        # the flip copy is only ever judged by PASS 2 (pass 1 writes to the MIRROR, and mirror_sampled.sh refuses
                        # a fresh copy that carries label rows), so a judged copy means pass 2 already ran on it — a pass-1 RE-RUN
                        # on the mirror (2026-09-10: the last incomplete blocks after the 529 storm) cannot touch it. Noted, not refused.
                        notes.append(f"flip copy {flip} is {fs} — pass 2 already judged it; this pass-1 run writes only the mirror")
                    fstores, fp = store_dirs(flip / "cells"); problems += fp
                    mstores, _ = store_dirs(root / "cells")
                    if set(fstores) != set(mstores):
                        problems.append(f"flip copy cells {sorted(fstores)} != mirror cells {sorted(mstores)}")
            cache = DEST / "caches" / "pass1" / f"{tag}.sqlite"
            state = "resume (seeded copy exists)" if cache.exists() else "fresh seeded private copy of the shared cache"
            ms = judge_state(root) if (root / "cells").is_dir() else "absent"
            notes.append(f"mirror 512 root is {ms}; pass-1 cache {state}")
            sentinel = LOGS / tag
            steps = [{"name": "mkdir", "argv": ["mkdir", "-p", str(cache.parent)]},   # VACUUM INTO cannot create the directory (every pass-1 unit died here on 2026-09-10)
                     {"name": "seed", "argv": [py, f"{JUDGE}/seed_private_cache.py", "--controls-root", str(root.parent), "--targets", tag,
                                               "--out", str(cache)], "skip_if_exists": str(cache)},
                     {"name": "judge", **judge_argv(root.parent, tag, cache, sentinel,
                                                    "pass-1 (primary labels): seeded private copy of the shared cache via seed_private_cache.py, foreign rows scrubbed; merged back after the run")},
                     {"name": "merge", "argv": [py, f"{JUDGE}/merge_cache_back.py", "--private", str(cache)]}]
            unit = f"{prefix}-pass1-{tag}-{stamp}"
            jobs.append({"id": jid, "unit": unit, "ready": not problems, "problems": problems, "notes": notes,
                         "caches": [{"role": "pass-1 seeded private copy (merged back)", "path": str(cache), "state": state}],
                         "steps": steps, "needs_key": True, "script": unit_script(unit, steps, needs_key=True)})

if phase == "controls-pass2":
    for key in KEYS:
        for T in TRIALS:
            tag = f"{key}_t{T}"; jid = f"controls-pass2/{tag}"
            F = DEST / "controls_sampled" / "flip_pass"
            f512 = F / "cap512" / tag
            cells, problems = explicit_cells(key, T)
            notes, caches, steps = [], [], []
            if not (DEST / "controls_sampled" / "controls_cap512" / tag).is_dir():
                notes_global.append(f"NONE {jid}: no 512 root mirrored for {tag}"); continue
            if cells is not None:
                problems += check_root_complete(f512, cells) if (f512 / "cells").is_dir() else [f"flip copy absent: {f512} (run mirror_sampled.sh)"]
            finals, lp = ladder_final_rungs(tag)

            def in_scope(c):
                """--cells: the SELECTED cells are this plan's business; a held cell's ladder entry (incomplete, or a flip store the
                ladder no longer names) is reported as a note and never blocks judging the selected cells (2026-09-14: a controls job
                for two late cells re-folded the trial's ladder and re-marked the legacy count-only pairs 'incomplete'). A problem
                that names no cell (ladder absent, cross-check unavailable) is always in scope."""
                return CELLS_FILTER is None or c is None or c in CELLS_FILTER

            def cell_of(problem):   # ladder problems read "<tag>/<cell>/<ctl>: …"
                m = re.match(re.escape(tag) + r"/([^/]+)/", problem)
                return m.group(1) if m else None

            held_lp = [p for p in lp if not in_scope(cell_of(p))]
            problems += [p for p in lp if in_scope(cell_of(p))]
            if held_lp:
                notes.append("--cells: ladder entries of HELD cells are incomplete and do not block this plan: " + "; ".join(held_lp))

            def cache_rule(root: Path, cache: Path, label: str):
                st = judge_state(root) if (root / "cells").is_dir() else "absent"
                if cache.exists() and st == "pristine":
                    problems.append(f"{label}: cache {cache} exists but the root is unjudged — a stale cache would serve verdicts and the flip floor "
                                    "must be live; remove or rename it deliberately")
                    return "REFUSED"
                if cache.exists():
                    return f"resume ({st})"
                return "FRESH (absent; the judge creates an empty sqlite)"

            c512 = DEST / "caches" / "pass2" / f"cap512_{tag}.sqlite"
            caches.append({"role": "pass-2 fresh cache for flip_pass/cap512", "root": str(f512), "path": str(c512), "state": cache_rule(f512, c512, f"cap512/{tag}")})
            steps.append({"name": "mkdir", "argv": ["mkdir", "-p", str(c512.parent)]})   # fresh pass-2 caches need their directory too
            if CELLS_FILTER is None:
                steps.append({"name": "judge_cap512", **judge_argv(F / "cap512", tag, c512, LOGS / tag / "cap512",
                                                                   "pass-2 flip: FRESH empty private cache for THIS root only; every member a live re-ask; no merge back")})
            else:
                f512_stores, fp512 = store_dirs(f512 / "cells") if (f512 / "cells").is_dir() else ({}, [])
                problems += fp512
                for c in CELLS_FILTER:
                    if c not in f512_stores:
                        problems.append(f"--cells: {tag} has no 512 flip store for cell {c!r} (cells present: {sorted(f512_stores)})"); continue
                    steps.append({"name": f"judge_cap512_{c}", **judge_argv(F / "cap512", tag, c512, LOGS / tag / "cap512" / c,
                                                                            f"pass-2 flip: FRESH empty private cache for THIS root only; cell {c} only (--cells); no merge back",
                                                                            match=f512_stores[c].name)})
                notes.append(f"--cells: judging only {CELLS_FILTER} (the other cells of {tag} are held; 2026-09-11)")
            if finals is not None:
                # the ONE rule (S-D18 / 80938539): each control's own final rung — the ladder's per-control exit, cross-checked against the
                # mirrored rung stores through the readers' final_rung_store (a store holding a control ABOVE the ladder's rung refuses)
                try:
                    sys.path.insert(0, str(WT / SAMPLED)); import final_rung as fr        # stdlib only: the readers' rule, importable under any --python
                    for c, by in sorted(finals.items()):
                        if not in_scope(c):
                            continue
                        for ctl, r in sorted(by.items()):
                            got = fr.final_rung_store(DEST / "controls_sampled", key, T, c, ctl)[0]
                            if got is not None and got > r:
                                problems.append(f"{tag}/{c}/{ctl}: the ladder says final rung {r} but the mirrored rung stores hold {ctl} up to {got} (final_rung_store) — the ladder and the stores disagree")
                            elif got is not None and got < r:
                                notes.append(f"{tag}/{c}/{ctl}: the ladder says final rung {r}; the mirrored rung stores hold {ctl} only up to {got} (run mirror_sampled.sh)")
                except Exception as exc:  # noqa: BLE001
                    problems.append(f"final_rung_store cross-check unavailable: {type(exc).__name__}: {exc}")
                want_all_by_R = {}                     # R -> {cell: [controls whose final rung is R]} — the ladder's view, filter-independent
                for c, by in finals.items():
                    for ctl, r in by.items():
                        if r > 512:
                            want_all_by_R.setdefault(r, {}).setdefault(c, []).append(ctl)
                for R in (2048, 8192, 16384):     # every rung above 512 a control can reach (ceiling 16,384)
                    want_all = want_all_by_R.get(R, {})
                    want = {c: v for c, v in want_all.items() if CELLS_FILTER is None or c in CELLS_FILTER}
                    fR = F / f"cap{R}" / tag
                    if not want_all and (fR / "cells").is_dir():
                        msg = f"flip_pass/cap{R}/{tag} exists but the ladder names no (cell, control) whose final rung is {R}"
                        if CELLS_FILTER is None:
                            problems.append(msg)
                        else:
                            notes.append(msg + " (held cells' stores; not this plan's business under --cells)")
                    if not want:
                        continue
                    stores, sp = store_dirs(fR / "cells") if (fR / "cells").is_dir() else ({}, [f"flip_pass/cap{R}/{tag}: absent — the ladder says {sorted(want)} reached {R}; run mirror_sampled.sh"])
                    problems += sp
                    cR = DEST / "caches" / "pass2" / f"cap{R}_{tag}.sqlite"
                    if stores:
                        caches.append({"role": f"pass-2 fresh cache for flip_pass/cap{R}", "root": str(fR), "path": str(cR), "state": cache_rule(fR, cR, f"cap{R}/{tag}")})
                    for c in sorted(want):
                        ctls = sorted(want[c], key=CONTROLS.index)
                        if c not in stores:
                            problems.append(f"flip_pass/cap{R}/{tag}: no store for cell {c} (ladder final rung {R} for {ctls}); run mirror_sampled.sh"); continue
                        d = stores[c]
                        have = [x for x in (json.loads((d / "manifest.json").read_text()).get("controls") or []) if x in CONTROLS]
                        lack = [x for x in ctls if x not in have]
                        if lack:
                            problems.append(f"flip_pass/cap{R}/{tag}: store {d.name} lacks {lack}, whose final rung the ladder says is {R} (manifest controls {have})")
                        problems += check_controls_store(d, [x for x in ctls if x in have])
                        # ONE judge step per (cell, R): every control whose final rung is THIS store rides it — both controls when both exit here, never a duplicate
                        steps.append({"name": f"judge_cap{R}_{c}", **judge_argv(fR.parent, tag, cR, LOGS / tag / f"cap{R}" / c,
                                                                                 f"pass-2 flip: FRESH empty private cache for the cap{R} root only; final-rung store of {c}; no merge back",
                                                                                 match=d.name, controls=ctls)})
                    extra = sorted(set(stores) - set(want_all))       # against the LADDER, not the --cells selection …
                    extra_in = [c for c in extra if in_scope(c)]       # … but only a SELECTED cell's stray store blocks under --cells
                    extra_held = [c for c in extra if not in_scope(c)]
                    if extra_in:
                        problems.append(f"flip_pass/cap{R}/{tag} holds stores for {extra_in} whose ladder final rung is not {R}")
                    if extra_held:
                        notes.append(f"flip_pass/cap{R}/{tag} holds stores for HELD cells {extra_held} whose ladder final rung is not {R} (not this plan's business under --cells)")
                cells_no_rung = sorted(c for c, by in finals.items() if by and all(r == 512 for r in by.values()))
                if cells_no_rung:
                    notes.append(f"left the ladder at 512 (512 members re-judged, no rung member): {cells_no_rung}")
                split = [f"{c}: " + ", ".join(f"{ctl} -> {r}" for ctl, r in sorted(by.items(), key=lambda kv: CONTROLS.index(kv[0]) if kv[0] in CONTROLS else 9))
                         for c, by in sorted(finals.items()) if len(set(by.values())) > 1]
                if split:
                    notes.append("controls of one cell exit at different rungs — each judged at its OWN final-rung store (S-D18): " + "; ".join(split))
                if cells is not None:
                    unl = sorted(c for c in set(cells) - set(finals) if in_scope(c))
                    if unl:
                        problems.append(f"{tag}: ladder has no entry for cells {unl} (the controls job did not run / record them)")
            paths = [c["path"] for c in caches]
            if len(set(paths)) != len(paths):
                problems.append(f"pass-2 caches are not distinct: {paths}")
            notes.append("no merge back to the shared cache; --run-started-at is stamped at each invocation, judged_at per control by the judge")
            unit = f"{prefix}-pass2-{tag}-{stamp}"
            jobs.append({"id": jid, "unit": unit, "ready": not problems, "problems": problems, "notes": notes, "caches": caches,
                         "steps": steps, "needs_key": True, "script": unit_script(unit, steps, needs_key=True)})

# --------------------------------------------------------------------------- print + write
print(f"== phase {phase} == dest={DEST} worktree={WT} python={py} org={org} stamp={stamp}")
for n in notes_global:
    print(n)
for j in jobs:
    print(f"job {j['id']}: {'READY' if j['ready'] else 'REFUSED'}  unit={j['unit']}")
    for c in j["caches"]:
        extra = f" [{c['state']}]" if c.get("state") else ""
        seeds = f" seeds={len(c['seeds'])}" if "seeds" in c else ""
        print(f"  cache {c['role']}: {c['path']}{extra}{seeds}")
        for s in c.get("seeds", []):
            print(f"    seed <- {s}")
    for n in j["notes"]:
        print(f"  note: {n}")
    for p in j["problems"]:
        print(f"  MISSING: {p}")
    for st in j["steps"]:
        print(f"  $ {render_cmd(st)}")
n_ready = sum(j["ready"] for j in jobs); n_ref = len(jobs) - n_ready
print(f"plan {phase}: {n_ready} ready, {n_ref} refused, {len(notes_global)} none")
Path(out_json).write_text(json.dumps({"phase": phase, "stamp": stamp, "org": org, "ceiling": CEILING, "jobs": jobs, "none": notes_global}, indent=1))
sys.exit(0 if n_ref == 0 else 2)
PY
}

# --------------------------------------------------------------------------- run a phase's units
record_phase() {   # record_phase <phase> <plan json> <started> <ended> <launched units json>
  "$PY" - "$DEST/GRADING_RUN.json" "$1" "$2" "$3" "$4" "$5" "$DEST/logs/$1" <<'PY'
import json, sys
from pathlib import Path
out, phase, plan_p, started, ended, launched_p, logs = sys.argv[1:8]
plan = json.loads(Path(plan_p).read_text()); launched = json.loads(Path(launched_p).read_text())
rec = json.loads(Path(out).read_text()) if Path(out).exists() else {"runs": []}
units = []
for j in plan["jobs"]:
    L = launched.get(j["unit"])
    exit_p = Path(logs) / f"EXIT-{j['unit']}"
    steps_p = Path(logs) / f"{j['unit']}.steps.jsonl"
    steps = [json.loads(l) for l in steps_p.read_text().splitlines() if l.strip()] if steps_p.exists() else []
    units.append({"id": j["id"], "unit": j["unit"], "ready": j["ready"], "problems": j["problems"], "launched": bool(L),
                  "started": (L or {}).get("started"), "ended": (L or {}).get("ended"),
                  "exit": int(exit_p.read_text().strip()) if exit_p.exists() else None,
                  "log": str(Path(logs) / f"{j['unit']}.log"), "script": str(Path(logs) / f"{j['unit']}.sh"),
                  "caches": j["caches"], "steps": steps,
                  "commands": [" ".join(st["argv"]) for st in j["steps"]]})
rec["runs"].append({"phase": phase, "stamp": plan["stamp"], "started": started, "ended": ended, "org": plan.get("org", "lowprio"),
                    "key_source": f"the secret-manager CLI's {plan.get('org', 'lowprio')} item inside each unit (never on disk)", "units": units, "none": plan.get("none", []),
                    "ceiling": plan.get("ceiling")})     # S-D27: parallel x judge-concurrency, the limit for the org, and the override reason if one was given
rec.setdefault("phases", {})[phase] = rec["runs"][-1]
tmp = Path(out + ".tmp"); tmp.write_text(json.dumps(rec, indent=1, sort_keys=True) + "\n"); tmp.replace(out)
print(f"[grade-sampled] GRADING_RUN.json updated: phase {phase} ({sum(1 for u in units if u['launched'])} units)")
PY
}

run_phase() {   # run_phase <phase> -> 0 ok, 2 refused, 1 a unit failed
  local phase=$1
  mkdir -p "$DEST/logs/$phase"
  local plan_json; plan_json=$(mktemp "$DEST/logs/$phase/plan-$STAMP.XXXX.json")
  local plan_out; plan_out=$(plan_phase "$phase" "$plan_json"); local prc=$?
  echo "$plan_out"
  if [ $prc -ne 0 ] && [ $PARTIAL -eq 0 ]; then say "phase $phase REFUSED (incomplete inputs above); nothing launched"; return 2; fi
  local started; started=$(date -u +%FT%TZ)
  local launched="{}"
  local units; units=$("$PY" -c 'import json,sys; [print(j["unit"]) for j in json.load(open(sys.argv[1]))["jobs"] if j["ready"]]' "$plan_json")
  [ -z "$units" ] && { say "phase $phase: nothing to run"; echo "$launched" > "$plan_json.launched"; record_phase "$phase" "$plan_json" "$started" "$(date -u +%FT%TZ)" "$plan_json.launched"; return 0; }
  local inflight=0 rc=0
  for u in $units; do
    while :; do
      inflight=0; for v in $(echo "$launched" | "$PY" -c 'import json,sys; print(" ".join(json.load(sys.stdin)))'); do [ -f "$DEST/logs/$phase/EXIT-$v" ] || inflight=$((inflight+1)); done
      [ $inflight -lt "$PARALLEL" ] && break; sleep 20
    done
    "$PY" -c 'import json,sys; j=[x for x in json.load(open(sys.argv[1]))["jobs"] if x["unit"]==sys.argv[2]][0]; open(sys.argv[3],"w").write(j["script"])' "$plan_json" "$u" "$DEST/logs/$phase/$u.sh"
    chmod 700 "$DEST/logs/$phase/$u.sh"; rm -f "$DEST/logs/$phase/EXIT-$u"
    uflags="$(secrets_unit_flags)"
    # shellcheck disable=SC2086
    if systemd-run --user --unit="$u" --collect --quiet --setenv=PATH="$HOME/.local/bin:$PATH" -p WorkingDirectory="$WT" $uflags /bin/bash "$DEST/logs/$phase/$u.sh"; then
      say "launched unit $u (log $DEST/logs/$phase/$u.log)"
      launched=$(echo "$launched" | "$PY" -c 'import json,sys,time; d=json.load(sys.stdin); d[sys.argv[1]]={"started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}; print(json.dumps(d))' "$u")
    else
      say "FAILED to launch unit $u"; echo 1 > "$DEST/logs/$phase/EXIT-$u"; rc=1
    fi
    secrets_unit_flags_cleanup "$uflags"      # the systemd < 247 EnvironmentFile is unlinked once the unit has started
  done
  local t0; t0=$(date +%s)
  for u in $units; do
    until [ -f "$DEST/logs/$phase/EXIT-$u" ]; do
      [ $(( $(date +%s) - t0 )) -ge "$GIVE_UP" ] && { say "gave up waiting for $u after ${GIVE_UP}s (the unit keeps running; re-run the phase to resume)"; rc=1; break; }
      sleep 20
    done
    local code; code=$(cat "$DEST/logs/$phase/EXIT-$u" 2>/dev/null || echo "?")
    [ "$code" = "0" ] || rc=1
    launched=$(echo "$launched" | "$PY" -c 'import json,sys,time; d=json.load(sys.stdin); d.setdefault(sys.argv[1],{})["ended"]=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()); print(json.dumps(d))' "$u")
    say "unit $u exit=$code"
    grep -E "DONE |REFUSED|FAILED|REFUSING|STALLED|JUDGE_|GREP GATE" "$DEST/logs/$phase/$u.log" 2>/dev/null | cut -c1-220 | tail -12
  done
  echo "$launched" > "$plan_json.launched"
  record_phase "$phase" "$plan_json" "$started" "$(date -u +%FT%TZ)" "$plan_json.launched"
  [ $FINAL_RUNG -eq 1 ] && refresh_final_rung "$phase"
  return $rc
}

if [ $PLAN -eq 1 ]; then
  if [ -n "$PLAN_OUT" ]; then mkdir -p "$PLAN_OUT"; tmpd=$PLAN_OUT; else tmpd=$(mktemp -d); fi
  overall=0
  case "$PHASE" in all) phases="static controls-pass1 controls-pass2";; *) phases=$PHASE;; esac
  for ph in $phases; do plan_phase "$ph" "$tmpd/plan-$ph.json" || overall=2; done
  [ $FINAL_RUNG -eq 1 ] && echo "post-phase: final-rung index refresh -> $DEST/report/analysis/final_rung.json (analysis_sampled.py final-rung --targets $FR_TARGETS --allow-pending; after every phase that launched units; never fails the phase; --no-final-rung skips)"
  [ -n "$PLAN_OUT" ] || rm -rf "$tmpd"
  exit $overall
fi
case "$PHASE" in
  all) for ph in static controls-pass1 controls-pass2; do run_phase "$ph"; prc=$?; [ $prc -eq 0 ] || { say "stopping at phase $ph (exit $prc)"; exit $prc; }; done; say "all phases done";;
  *) run_phase "$PHASE";;
esac
