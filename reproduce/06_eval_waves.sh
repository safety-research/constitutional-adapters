#!/bin/bash
# reproduce/06_eval_waves.sh — stage 06: the evaluation waves on Slurm.
#
#   reproduce/06_eval_waves.sh --model K --view u0|s2|battery|d0p75|d0p75_relu --family <list>|all [--clusters a,b] [--twin] [--resubmit]
#                              [--nshards 9] [--wave-sha SHA] [--pass first|overflow] [--config F] [--overrides F] [--dry-run] [--help]
#   reproduce/06_eval_waves.sh --model K --gate <family>|all [--verdict | --headline X --range LO,HI] [--force-gate LWID=REASON]...
#   reproduce/06_eval_waves.sh --model K --view <baseline view> --family petri --pass overflow   # the Petri context-overflow re-run pass (--pass below)
#
#   families   static (2 static pairs + the BoN 512 store), padding (the paper's FIVE levels 0/8192/16384/32768/65536 — exactly
#              ONE level per sbatch job, five submissions per view), bon (the 4,096 climb array — submitted with a Slurm
#              --dependency=afterok on the static call's 512-store job while that job is live, so `--family static` goes first), multiturn (per named cluster
#              ONE group: its attacker-pool members + the target units placed on it; the 3 attacks x 4 disjoint shard groups
#              0,1,2 / 3,4 / 5,6 / 7,8 are PARTITIONED across EVERY configured cluster (SISTEER_CLUSTERS, the same every call —
#              a unit's placement is a function of the site, never of this call's --clusters) — never the same logical unit twice), petri (the
#              proxy; the drive runs on the grading host once endpoint.json lands — printed as `followup:`), agentic,
#              controls (XSTest + OR-Bench, 2 seeded trials), mmlu, gsm8k
#   The units come from ONE plan (lib/stage.py plan): the same rows 05_stage_roster.sh --step freeze froze as the gate's
#   inventory, so a submission binds EXACTLY its frozen logical_work_id. A u0 wave refuses
#   when the frozen inventory is missing or disagrees with the plan (units, shards, or the wave sha).
#   --view u0  the unsteered-first wave (none + the two prompted SIs) — NOT gated. A table that declares NO u0 view runs the
#              SINGLE-PHASE recipe (lib/models.py single_phase): its one steered wave carries its own unsteered legs and is
#              the baseline (lib/models.py baseline-view) — not gated either, only its compute plan is required.
#   --view != the baseline  passes the GLOBAL admission gate first (lib/admission.py: every baseline unit registered + complete under its
#              frozen bindings + covered per shard + verdicted) and needs the compute replan (tools/replan_compute.py)
#   --gate F   binds the family's u0 units to their run ids from the job's OWN pending file (every unit registers natively at job
#              start — the mmlu_full job included; EVERY array task's file: a task registers under its own raw job id),
#              then recomputes COVERAGE on EVERY call with THIS tree's lib/coverage_probe.py (shipped to the cluster by
#              content, digest-verified; its commit is recorded) at each run's RECORDED store (the registry's store_uri, the climb's
#              extra.climb_root, never a recomputed root) on the family's authoritative artifact (unique completed goal / item
#              ids, error outcomes apart; never a line count); refuses by name when the probe's coverage contract differs
#              from the inventory's; records gate/<family>.json {headline, plausible_range, verdict, both shas, the contract}: --verdict
#              READS each headline from the registry (lib/gate_headline.py), --headline/--range give one by hand. `--gate all` does
#              every family of the frozen inventory (one with no unit there, or under --verdict not ready to rule, is said and skipped). Exit 1 =
#              a reader / transport failure, or a verdict that could not be recorded (named; gate/<family>.json then stands as it was); exit 2 =
#              a refusal BEFORE anything is bound (--verdict beside --headline/--range, a malformed or half-given pair, a family not in the
#              inventory, the contract); exit 3 = --verdict skipped a family or ruled one `fail`.
#   --clusters SELECTS which placed units this call submits (default: every configured label); the multi-turn placement itself is
#              partitioned over every configured label regardless, so naming one label submits that label's group and nothing
#              else — it never re-partitions (a subset handed in as the universe moved crescendo/g34's group and the funnel's
#              workload-spec proof refused it). Single-GPU units (`*`) go to the selected labels.
#   --cells    the unsteered model is not re-run: serve AND register only these cells of the view (comma
#              list of config cell ids, validated against the view config's explicit_cells) — the plan writes them into every target job's
#              cells positional and F22_REGISTRY_CELLS; a family whose job takes no cells positional refuses by name. Omit = the view's
#              full enumeration (`none` included).
#   --units    SELECT units of the planned family by name (comma list of `<family>.<unit>` as the `submitted` line spells them, e.g.
#              gsm8k.none): the plan, the inventory check and the placement are unchanged; only the named units are submitted this
#              call and the wave file is NOT written (a selected call is a deliberate partial: re-run without --units to complete
#              the wave — placed units are skipped). The canary order's "one cell first" (the units are at the cap).
#   --twin     the two-cluster start-grant election for single-GPU units (§3.9); REFUSED with --family multiturn (a group
#              election is not implemented), never a silent double submission.
#   --move-group <shard group>=<label> [repeatable] + --move-reason "…": place that multi-turn group's three targets on <label> instead of
#              their round-robin cluster and size <label>'s pool from the moved count (2026-10-02: a cluster that cannot stand the
#              serving venv up hands its group to one that can). The placement manifest records the DECLARED move with its reason; the
#              --resubmit of the dead targets lands on <label> and writes the 24d placement receipt (the move's proof); later calls read
#              the manifest's declared moves (sticky). The grown group's PLACED record is superseded with the reason (mark-placed, derived
#              from the persisted declaration on every pass). --re-move re-points an already declared group (a new --move-reason; the
#              earlier declaration stays in the manifest's history). Each view has its own placement manifest (u0: the historic file).
#   --resubmit re-mint an identity for a unit whose bound job is provably TERMINAL and unregistered (submit_twin.py checks);
#              without it a re-run SKIPS every unit already submitted/bound and submits only what is missing. A Petri
#              proxy whose dead drive already filed some legs is re-minted too, after the funnel reads those legs' stores as done
#              (petri_run.py legs-status: the drive's own predicate); a filed leg a resumed drive would regenerate keeps the refusal.
#   --pass     first (default: the wave's rows) | overflow — the Petri CONTEXT-OVERFLOW RE-RUN PASS, with --family petri alone: ONE
#              second proxy (`petri_proxy.sbatch … overflow-rerun <cells>`, name f22-petri-<tag>-rerun, unit petri.overflow) at the
#              doubled window for the legs the daemon recorded INCOMPLETE on context-overflow ledgers (grade/petri_drive.<tag>.json:
#              the first-pass attempt `incomplete`, verdict overflow_pending); the cells are those legs' (+ `none` for the controls
#              twin), read through lib/petri_overflow.py and the drive's own done-ness reading — refused by name until the daemon has
#              recorded that verdict, when nothing is ledgered any more, or with --units / --cells / --twin. The proxy is placed on the
#              cluster the FIRST PASS is bound on (05 staged its object bank and bank view there): --clusters may name that cluster, alone
#              or among others, and is refused by name when it excludes it. The daemon drives the re-run proxy it submits (legs=ledgered,
#              F22_OVERFLOW_RERUN=1). No WAVE file is written (a follow-up unit, never inventoried). --resubmit on a bound overflow unit
#              carries its declaration VERBATIM (a partial pass whose proxy died re-mints the same runs). A submit of the pass interrupted
#              between its reservation and the new attempt's binding leaves the unit's transition token: a real call reconciles it through
#              the funnel (submit_twin.py reconcile-token) BEFORE the preflight reads the declaration — the launched job adopted, or the
#              token cleared — so the same command recovers; a dry run says so and composes nothing.
# Every submission goes through lib/slurm.sh slurm_submit (the ORIGINAL sbatch file, the wave shell's positional
# contract, the cluster block's --qos/--partition/--account/--time on the command line, job-side env as VAR=value
# prefixes — never --export=) and prints `submitted <cluster> <jobid> <unit>`; lib/watch_job.sh is launched per job. The
# pinned sha is mirrored/verified on every target cluster BEFORE the first submission. A failed submission
# FAILS the wave: the partial list is recorded in WAVE.<view>.partial.json and WAVE.<view>.json is NOT written. A multi-turn
# group whose mark-placed answers NO LIVE MEMBER (every member self-exited; the pass minted no replacement) ends the pass INCOMPLETE the
# same way: the group(s) and the REMEDY (the same pass again, printed) are recorded in WAVE.<view>.partial.json, WAVE.<view>.json is NOT
# written, no `06 done`, exit 4.
# Exit codes: 0 `06 done` (also --help, --dry-run and --gate; --gate's own 1 and 3 above); 2 a refusal — an argument, the plan, the frozen inventory, admission, a failed
#             submission (the partial record written, no wave file); 4 placement INCOMPLETE — a multi-turn group with NO LIVE MEMBER
#             (the partial record written with the group(s) and the printed REMEDY, no wave file; mark-placed's own code for the refusal, never
#             retried here). Every OTHER mark-placed refusal of a group (the immutable-placement fence, its rc 3) is a note: the group is placed on
#             the next pass or by the sanctioned command printed, the pass goes on, and it exits 0 — its targets have a live member to bind.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/lib/common.sh"; source "$HERE/lib/slurm.sh"
usage(){ sed -n '2,36p' "$0"; repro_common_help; }
# the words of a command line as the shell reads them back: a value of plain characters as is, anything else (a space, a quote, a `$`, an
# empty value) through `printf %q` — for the remedies this script prints to be run verbatim (placement-fix round 6; `%q` alone escapes commas)
shell_words(){ local v; for v in "$@"; do case "$v" in ""|*[!A-Za-z0-9_./:,=+@%-]*) printf '%q ' "$v";; *) printf '%s ' "$v";; esac; done; }
VIEW=""; FAMILIES=""; CLUSTERS=""; TWIN=0; RESUBMIT=0; NSHARDS=9; WAVE_SHA=""; GATE=""; HEADLINE=""; RANGE=""; VERDICT_AUTO=0; FORCE=(); MOVE=(); MOVE_REASON=""; MOVE_REMOVE=""; CELLS_SEL=""; CELLS_GIVEN=0
repro_parse_common_args "$@"; set -- "${REPRO_ARGS[@]+"${REPRO_ARGS[@]}"}"
while [ $# -gt 0 ]; do
  case "$1" in
    --view) VIEW="$2"; shift 2;; --family) FAMILIES="$2"; shift 2;; --clusters) CLUSTERS="$2"; shift 2;; --units) UNITS="$2"; shift 2;;
    --twin) TWIN=1; shift;; --resubmit) RESUBMIT=1; shift;; --nshards) NSHARDS="$2"; shift 2;; --wave-sha) WAVE_SHA="$2"; shift 2;;
    --gate) GATE="$2"; shift 2;; --headline) HEADLINE="$2"; shift 2;; --range) RANGE="$2"; shift 2;; --verdict) VERDICT_AUTO=1; shift;;
    --force-gate) FORCE+=("$2"); shift 2;;
    --move-group) MOVE+=("$2"); shift 2;; --move-reason) MOVE_REASON="$2"; shift 2;; --re-move) MOVE_REMOVE=1; shift;;
    --cells) CELLS_SEL="$2"; CELLS_GIVEN=1; shift 2;;
    --pass) PASS="$2"; shift 2;;
    *) echo "REFUSED: unknown argument $1" >&2; usage >&2; exit 2;;
  esac
done
PASS="${PASS:-first}"   # the Petri pass (petri-completion): first = the wave's rows; overflow = the context-overflow re-run proxy (--pass, above)
[ "$REPRO_HELP" = 1 ] && { usage; exit 0; }
[ "$VERDICT_AUTO" = 1 ] && [ -z "$GATE" ] && { echo "REFUSED: --verdict records the gate's verdicts and needs --gate <family>|all" >&2; exit 2; }
# cells-fix-2: an explicitly EMPTY selection (--cells '' / ',' / a blank token) refuses here, before any plan — it used to be dropped
# (`${CELLS_SEL:+…}`) and the plan then launched the FULL config, none included; absent --cells is still the view's full enumeration
if [ "$CELLS_GIVEN" = 1 ]; then _c="${CELLS_SEL// /}"; case ",$_c," in *,,*) echo "REFUSED: --cells '$CELLS_SEL' is an empty selection or names a blank cell — name every cell, or omit --cells for the view's full enumeration" >&2; exit 2;; esac; fi
repro_load_config; repro_require_model
K="$REPRO_MODEL"; PY="$(repro_python)"; STATE="$(repro_state_dir "$K")"
CLUSTERS_GIVEN="$CLUSTERS"   # the caller's EXPLICIT --clusters (empty = the default, every configured label): the overflow pass refuses an explicit selection that excludes the first pass's cluster
repro_resolve_clusters
[ -n "$WAVE_SHA" ] || WAVE_SHA="$(repro_head_sha)"
ALL_FAMILIES="static padding bon multiturn petri agentic controls mmlu gsm8k"
REGISTRY_DB="$(repro_registry_root)/registry.sqlite"
STAGE_PY=("$PY" "$HERE/lib/stage.py")
INV="$STATE/gate/expected_units.json"

# =============================================================================== --gate: bind, cover, verdict (per family)
if [ -n "$GATE" ]; then
  # --verdict reads every family's headline from the registry and --headline/--range give one family's by hand: one or the other, refused by
  # name before the inventory is read (the call itself is malformed, whatever the state)
  if [ "$VERDICT_AUTO" = 1 ] && [ -n "$HEADLINE$RANGE" ]; then
    repro_fatal "06 --gate: --verdict reads each family's headline from the registry and --headline/--range give one family's by hand — pass one or the other (got --verdict with --headline '$HEADLINE' --range '$RANGE'); nothing bound, no coverage written"
  fi
  [ -f "$INV" ] || repro_fatal "no frozen inventory at $INV — run 05_stage_roster.sh --step freeze first"
  IFS=$'\t' read -r INV_SHA INV_UTC < <("$PY" -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d.get("code_sha") or "", d.get("frozen_utc") or "", sep="\t")' "$INV")
  # the gate's scratch dir: EVERY JSON / text hand-off between the shell and stage.py goes through a file here, never an argv string — Linux caps
  # one argv string at MAX_ARG_STRLEN = 131,072 B, and a 40-run static pending file is 149,440 B, a climb's four task files ~146 KB, mmlu's ten
  # 1,531-id probe documents far more; `Argument list too long` read as "the pending file names no runs" / `probe: null`
  GATE_TMP="$(mktemp -d "${TMPDIR:-/tmp}/sisteer-gate.XXXXXX")"; trap 'rm -rf "$GATE_TMP"' EXIT
  # the verdict's arguments are checked HERE, before anything is bound or covered (Major 1): a malformed --range used to reach
  # `stage.py verdict` only after the whole family was bound and covered, and its traceback's rc 1 — the status a persisted `fail` also returned —
  # was swallowed (`|| true`), so the pass exited 0 with the OLD verdict (a `pass`) standing over the coverage it had just refreshed, which admission
  # reads as admissible. Both flags or neither; stage.py parse_verdict_args is the one rule (cmd_verdict refuses by the same words).
  if [ -n "$HEADLINE$RANGE" ]; then
    { [ -n "$HEADLINE" ] && [ -n "$RANGE" ]; } || repro_fatal "06 --gate: pass BOTH --headline <value> and --range <lo,hi> (the verdict pass) or neither (bind + cover only) — got --headline '$HEADLINE' --range '$RANGE'; nothing bound, no coverage written"
    why="$("$PY" -c 'import sys; sys.path.insert(0, sys.argv[1]); from stage import parse_verdict_args
try: parse_verdict_args(sys.argv[2], sys.argv[3])
except ValueError as e: print(e); sys.exit(1)' "$HERE/lib" "$HEADLINE" "$RANGE")" \
      || repro_fatal "06 --gate: $why — the verdict's arguments are checked before anything is bound (stage.py verdict would refuse them after the whole family was covered); nothing bound, no coverage written"
  fi
  [ "$VERDICT_AUTO" = 1 ] && [ ! -f "$REGISTRY_DB" ] \
    && repro_fatal "06 --gate --verdict: no registry at $REGISTRY_DB to read the headlines from (SISTEER_REGISTRY_ROOT); nothing bound, no coverage written"
  # the families and each one's unit list, from the inventory, into FILES whose producer's status is READ: a refused producer inside a process
  # substitution leaves an empty loop and loses its status (the same shape as the probe plan: a schema refusal, or a
  # family with no unit, read as "no units", and the verdict step then RULED over nothing read). `all` = every family of the frozen inventory, in
  # ALL_FAMILIES order (a family with no unit there is said and skipped, never ruled); a named family must be one of the inventory's.
  "$PY" - "$INV" "$GATE" "$ALL_FAMILIES" "$GATE_TMP" <<'PYEOF' || exit 2
import json, os, sys
inv_p, gate, all_fams, tmp = sys.argv[1:5]
inv = json.load(open(inv_p))
if inv.get("schema") != 2:
    sys.exit(f"REFUSED: 06 --gate: {inv_p} has schema {inv.get('schema')!r}; this stage reads schema 2 — re-freeze (05_stage_roster.sh --step freeze); nothing bound, no coverage written")
present = list(dict.fromkeys(u.get("family") for u in (inv.get("units") or [])))
if gate == "all":
    fams = [f for f in all_fams.split() if f in present]
    for f in all_fams.split():
        if f not in present:
            print(f"note: gate all: {f} has no unit in the frozen inventory — skipped (nothing to bind, cover or rule)")
else:
    fams = list(dict.fromkeys(f for f in gate.split(",") if f))
    bad = [f for f in fams if f not in present]
    if bad:
        sys.exit(f"REFUSED: 06 --gate: {bad} not a family of the frozen inventory (its families: {' '.join(present)}) — nothing bound, no coverage written")
for fam in fams:
    with open(os.path.join(tmp, f"units.{fam}.tsv"), "w", encoding="utf-8") as fh:
        for u in inv["units"]:
            if u["family"] == fam:
                fh.write("\t".join([u["logical_work_id"], u.get("registration") or "native", u.get("array") or "-"]) + "\n")
with open(os.path.join(tmp, "fams.txt"), "w", encoding="utf-8") as fh:
    fh.write(" ".join(fams))
PYEOF
  fams="$(cat "$GATE_TMP/fams.txt")"
  # a STARTED wave is read with the CURRENT tree's lib/coverage_probe.py (a reader, not treatment — the pinned probe at the
  # inventory's sha could not find the grouped multi-turn / sampled-controls / agentic stores and read "0/30"); the probe's commit is recorded
  # in every coverage record and verdict beside the inventory's, so the file must be committed; it travels to each cluster as ONE file by content
  # (repro_ssh_put: <shared>/scripts/registry/coverage_probe.<sha256-16>.py, digest-verified; never a ~6 GB mirror of the tree)
  PROBE_LOCAL="$HERE/lib/coverage_probe.py"
  if [ "$REPRO_DRY_RUN" != 1 ]; then
    git -C "$REPRO_REPO" diff --quiet HEAD -- reproduce/lib/coverage_probe.py 2>/dev/null \
      || repro_fatal "06 --gate: reproduce/lib/coverage_probe.py has uncommitted changes — the probe's commit is recorded in every verdict (D3003 (3)); commit it first"
  fi
  PROBE_SHA="$(repro_head_sha)"; PROBE_FILE_SHA256="$(sha256sum "$PROBE_LOCAL" | cut -d' ' -f1)"
  PROBE_REMOTE_REL="scripts/registry/coverage_probe.${PROBE_FILE_SHA256:0:16}.py"
  CONTRACT_JSON="$GATE_TMP/contract.json"
  # the coverage CONTRACT (the unit set, every eval's identity keys, the probe's per-layout done predicates) compared STRUCTURALLY with
  # the inventory's record BEFORE anything is bound — a changed predicate or identity refuses by name, naming the ruling; an inventory frozen before
  # the field existed is admitted at the current contract ONCE (gate/coverage_contract.json; the fact rides every verdict); a receipted rerun
  # stamp is an amendment, never a second authority
  "${STAGE_PY[@]}" gate-contract --state-root "$STATE" --probe "$PROBE_LOCAL" --admit-once --probe-sha "$PROBE_SHA" --probe-sha256 "$PROBE_FILE_SHA256" \
    --write "$CONTRACT_JSON" $(repro_dry_flag) || exit 2
  CONTRACT_LINE="(dry run)"
  # (%-formatting, no quote escaped: inside a single-quoted -c string a `\"` reaches Python as a literal backslash, a SyntaxError on stderr and an
  # empty line here — the shape the e2e pins: `cc1 <sha12> (admitted_once, admitted <utc>, 0 amendment(s))`)
  [ -f "$CONTRACT_JSON" ] && CONTRACT_LINE="$("$PY" -c 'import json,sys; d=json.load(open(sys.argv[1])); print("%s %s (%s%s, %d amendment(s))" % (d["version"], d["sha256"][:12], d["source"], (", admitted " + d["admitted_utc"]) if d.get("admitted_utc") else "", len(d.get("amendments") or [])))' "$CONTRACT_JSON")"
  GATE_FAILED=()                 # the units that hit a MECHANICAL failure (a transport error, a reader refusal, an empty registration): exit 1 after the loop
  GATE_VERDICT_FAILED=()         # the families whose verdict step did not record a verdict (stage.py verdict rc != 0; the ledger read failed): exit 1 too
  GATE_PASSED=(); GATE_RULED_FAIL=(); GATE_SKIPPED=()   # --verdict: the families ruled pass, ruled fail, and skipped as not ready to rule (exit 3 for the last two)
  gate_list(){ if [ $# -gt 0 ]; then echo "$*"; else echo none; fi; }
  [ "$VERDICT_AUTO" = 1 ] && [ "$REPRO_DRY_RUN" = 1 ] && repro_note "gate --verdict (dry run): each headline is read over the runs bound NOW; a real pass binds the submitted units first"
  declare -A PROBE_AT=()         # label -> the shipped probe's path on that cluster (shipped once per invocation, on first use)
  gate_prior_verdict(){   # <family> -> what gate/<family>.json holds NOW, for a note that says this pass wrote nothing over it: a verdict read from the
    # registry (it records its legs; admission re-reads them and refuses it while one reads differently) or one recorded by hand (bound to its runs alone)
    "$PY" -c 'import json,os,sys; p=sys.argv[1]; d=json.load(open(p)) if os.path.isfile(p) else None; print("verdict %r ruled %s %s" % (d["verdict"], d.get("ruled_utc"), "on the registry'"'"'s grades, which admission re-reads" if "headline_legs" in d else "by hand") if d else "absent")' "$STATE/gate/$1.json" 2>/dev/null || echo "unreadable"
  }
  probe_shipped(){   # <label> -> PROBE_PATH; rc 0 when the current probe's bytes are at <shared>/$PROBE_REMOTE_REL on that cluster (255 host down, 1 digest)
    local l="$1"; PROBE_PATH="${PROBE_AT[$l]:-}"; [ -n "$PROBE_PATH" ] && return 0
    local dst; dst="$(cluster_var "$l" CLUSTER_SHARED_ROOT)/$PROBE_REMOTE_REL"
    repro_ssh_put "$l" "$PROBE_LOCAL" "$dst" "$PROBE_FILE_SHA256" || return $?
    PROBE_AT[$l]="$dst"; PROBE_PATH="$dst"; return 0
  }
  n=0
  for fam in $fams; do
    repro_say "gate $fam: inventory ${INV_SHA:0:12} (frozen $INV_UTC), probe ${PROBE_SHA:0:12} (coverage_probe.py sha256 ${PROBE_FILE_SHA256:0:12}), coverage contract $CONTRACT_LINE: bind units -> run ids (the job's own pending file, every array task's), coverage by work identity at each run's RECORDED store (D3003), headline -> verdict"
    fam_failed_from=${#GATE_FAILED[@]}   # this family's watermark on GATE_FAILED: the verdict below is written only when NO unit of THIS family failed in THIS pass
    while IFS=$'\t' read -r lwid reg array; do
      [ -n "$lwid" ] || continue
      n=$((n+1)); T="$GATE_TMP/u$n"
      # the unit's binding (stage.py binding_state): state, and for a submitted one its job id + cluster — one read, to a FILE whose status is read
      # (a reader that raised inside a process substitution left `st` empty, and an empty state fell through the case below with no cluster)
      "$PY" -c 'import sys; sys.path.insert(0, sys.argv[1]); from stage import binding_state; b = binding_state(sys.argv[2], sys.argv[3]); print(b["state"], b.get("job_id") or "", b.get("cluster") or "", ",".join(b.get("run_ids") or []), sep="\t")' "$HERE/lib" "$STATE/submissions.jsonl" "$lwid" > "$T.binding.tsv" \
        || { repro_note "gate $fam: $lwid: its binding could not be read from $STATE/submissions.jsonl (above) — nothing bound, no coverage recorded"; GATE_FAILED+=("$lwid"); continue; }
      IFS=$'\t' read -r st job cl runs_csv < "$T.binding.tsv"
      case "$st" in
        unbound) repro_note "gate $fam: $lwid was never submitted (unbound) — the gate will refuse until it runs"; continue;;
        transition) repro_note "gate $fam: $lwid is mid re-run (a transition token: submit --resubmit was interrupted and reconciles it on its next call) — not bound this pass, the superseded generation is never read"; continue;;
        submitted)
          shared="$(cluster_var "$cl" CLUSTER_SHARED_ROOT)"; pend_dir="$shared/registry/pending"
          [ "$reg" = native ] || repro_fatal "unit $lwid: unknown registration kind '$reg' in the frozen inventory (every front-door unit registers natively — D65); re-freeze"
          tasks_arg=()
          # every task of an ARRAY unit registers under its OWN raw job id (SLURM_JOB_ID inside the task): pending/<raw>.json per task;
          # a plain job is its one "task" — ONE framed capture for both (`PENDING <f>` + the
          # file's text, or `MISSING <f>`, exit 0 either way), so only an ssh rc != 0 is a TRANSPORT failure, never an absent file
          if [ "$array" != - ] && [ -n "$array" ]; then
            if [ "$REPRO_DRY_RUN" = 1 ]; then echo "DRY-RUN: ssh $(cluster_var "$cl" CLUSTER_SSH) 'sacct -X -n -P -o JobIDRaw,JobID,State -j $job' -> every task's $pend_dir/<raw>.json -> stage.py bind --state registered --runs-json <run_id, eval_id, object, array_task per run>"; continue; fi
            repro_ssh "$cl" "sacct -X -n -P -o JobIDRaw,JobID,State -j $job" > "$T.sacct.txt"; rc=$?
            [ "$rc" = 0 ] || { repro_note "gate $fam: $lwid array $job: sacct could not be read over ssh (rc $rc) — a transport failure, not a task that has not started; nothing bound, still submitted"; GATE_FAILED+=("$lwid"); continue; }
            "${STAGE_PY[@]}" array-tasks --sacct "@$T.sacct.txt" > "$T.tasks.json" || { repro_note "gate $fam: $lwid array $job: sacct's rows could not be parsed (above) — nothing bound, still submitted"; GATE_FAILED+=("$lwid"); continue; }
            # every small reader's status is read: an empty `want_n`/`have_n` compared equal and read as "every task started"
            want_n="$("$PY" -c 'import sys; sys.path.insert(0, sys.argv[1]); from stage import array_task_indices; print(len(array_task_indices(sys.argv[2])))' "$HERE/lib" "$array")" \
              || { repro_note "gate $fam: $lwid array $job: the inventory's --array $array could not be read (above) — nothing bound, still submitted"; GATE_FAILED+=("$lwid"); continue; }
            have_n="$("$PY" -c 'import json,sys; print(sum(1 for t in json.load(open(sys.argv[1])) if t.get("task") is not None))' "$T.tasks.json")" \
              || { repro_note "gate $fam: $lwid array $job: the task list could not be read back (above) — nothing bound, still submitted"; GATE_FAILED+=("$lwid"); continue; }
            [ "$have_n" = "$want_n" ] || { repro_note "gate $fam: $lwid array $job: sacct lists $have_n of $want_n tasks (--array $array) — still submitted (a task not yet started has no raw id)"; continue; }
            pend="$pend_dir/<raw task id>.json"
            files="$("$PY" -c 'import json,shlex,sys; print(" ".join(shlex.quote(sys.argv[2] + "/" + t["raw"] + ".json") for t in json.load(open(sys.argv[1])) if t.get("task") is not None))' "$T.tasks.json" "$pend_dir")" \
              || { repro_note "gate $fam: $lwid array $job: the task files' names could not be formed (above) — nothing bound, still submitted"; GATE_FAILED+=("$lwid"); continue; }
            [ ${#files} -lt 100000 ] || repro_fatal "gate $fam: $lwid array $job: the task list is too long for one ssh command (${#files} bytes, $have_n tasks); split the array"
            tasks_arg=(--tasks-json "@$T.tasks.json")
          else
            pend="$pend_dir/$job.json"
            if [ "$REPRO_DRY_RUN" = 1 ]; then echo "DRY-RUN: ssh $(cluster_var "$cl" CLUSTER_SSH) 'cat $pend' -> stage.py bind --state registered --run-ids <from the pending file> --runs-json <run_id, eval_id, object per run>"; continue; fi
            files="$("$PY" -c 'import shlex,sys; print(shlex.quote(sys.argv[1]))' "$pend")"
          fi
          repro_ssh "$cl" "for f in $files; do if [ -f \"\$f\" ]; then echo \"PENDING \$f\"; cat \"\$f\"; echo; else echo \"MISSING \$f\"; fi; done" > "$T.pending.txt"; rc=$?
          [ "$rc" = 0 ] || { repro_note "gate $fam: $lwid job $job: the pending file(s) could not be read over ssh (rc $rc) — a transport failure, not a registration that has not happened; nothing bound, still submitted"; GATE_FAILED+=("$lwid"); continue; }
          # the reader (stage.py pending-runs): stdout = the runs list and nothing else, the MISSING paths in their own file; a file that does not
          # parse REFUSES by name — an exec failure never reads as "no runs" (`Argument list too long` did)
          "${STAGE_PY[@]}" pending-runs --capture "$T.pending.txt" --missing-out "$T.missing.json" ${tasks_arg[@]+"${tasks_arg[@]}"} > "$T.runs.json" \
            || { repro_note "gate $fam: $lwid job $job: the pending file could not be parsed (above) — nothing bound, still submitted"; GATE_FAILED+=("$lwid"); continue; }
          miss="$("$PY" -c 'import json,sys; print(" ".join(json.load(open(sys.argv[1]))))' "$T.missing.json")" \
            || { repro_note "gate $fam: $lwid job $job: the MISSING list could not be read back (above) — nothing bound, still submitted"; GATE_FAILED+=("$lwid"); continue; }
          if [ -n "$miss" ]; then
            if [ "$array" != - ] && [ -n "$array" ]; then repro_note "gate $fam: $lwid array $job: task pending file(s) not written yet: $miss — still submitted"
            else repro_note "gate $fam: $lwid job $job has no pending file yet ($pend) — still submitted"; fi
            continue
          fi
          runs_csv="$("$PY" -c 'import json,sys; print(",".join(r["run_id"] for r in json.load(open(sys.argv[1]))))' "$T.runs.json")" \
            || { repro_note "gate $fam: $lwid job $job: the runs list could not be read back (above) — nothing bound, still submitted"; GATE_FAILED+=("$lwid"); continue; }
          [ -n "$runs_csv" ] || { repro_note "gate $fam: $lwid job $job: the registration names no run (the pending file parsed; its runs list is empty — see the note above) — the producer's registration decides this, not the gate; nothing bound, still submitted"; GATE_FAILED+=("$lwid"); continue; }
          "${STAGE_PY[@]}" bind --ledger "$STATE/submissions.jsonl" --logical-work-id "$lwid" --state registered --run-ids "$runs_csv" --pending-file "$pend" --runs-json "@$T.runs.json" \
            || { repro_note "gate $fam: $lwid job $job: bind refused (above) — still submitted"; GATE_FAILED+=("$lwid"); continue; }
          for r in ${runs_csv//,/ }; do repro_registry_id run "$r"; done
          # 29: the runs this unit's EARLIER attempts registered that this registration does not continue are dead in the registry too —
          # terminal `run_state` records (cancelled|failed per sacct) into the producers' spool, never the database; a refusal is noted,
          # not fatal (the one-time `stage.py reconcile-registrations --write` catches up)
          PYTHONPATH="$REPRO_REPO/src${PYTHONPATH:+:$PYTHONPATH}" "${STAGE_PY[@]}" reconcile-registrations --ledger "$STATE/submissions.jsonl" --registry "$REGISTRY_DB" \
            --config "$SISTEER_CONFIG" --units "$lwid" --write \
            || repro_note "gate $fam: $lwid: terminal run_state records for its superseded attempts were not written (above) — run stage.py reconcile-registrations --write later"
          ;;
        registered) repro_say "gate $fam: $lwid already registered (runs $runs_csv) — coverage recomputed";;
        *) repro_note "gate $fam: $lwid: binding state '$st' is not one this gate reads (unbound | transition | submitted | registered) — nothing bound, no coverage recorded"; GATE_FAILED+=("$lwid"); continue;;
      esac
      # COVERAGE by WORK IDENTITY from the family's authoritative artifact, recomputed on every gate check:
      # THIS tree's lib/coverage_probe.py (shipped to the unit's cluster by content; or run on this
      # host for a store the harvester mirrored) at the store the REGISTRY RECORDED for the run — store_uri, a ruled successor's, the climb's
      # extra_json.climb_root — never the frozen coverage.path recomputed; never a scan of shardNN dirs, never a line count
      [ "$REPRO_DRY_RUN" = 1 ] && { echo "DRY-RUN: ship lib/coverage_probe.py@${PROBE_SHA:0:12} -> $(cluster_var "$cl" CLUSTER_SSH):$(cluster_var "$cl" CLUSTER_SHARED_ROOT)/$PROBE_REMOTE_REL (digest-verified); registry(ro) run rows (store_uri, extra_json) -> stage.py probe-plan -> the shipped probe per run at the run's RECORDED store (D3003), one call per bound array task of a climb run, folded by stage.py coverage (ssh $(cluster_var "$cl" CLUSTER_SSH) / local) -> stage.py coverage --logical-work-id $lwid"; continue; }
      "$PY" - "$REGISTRY_DB" "$STATE/submissions.jsonl" "$HERE/lib" "$lwid" > "$T.runs_reg.json" <<'PYEOF' || { repro_note "gate $fam: $lwid: the registry rows could not be read (above) — no coverage recorded"; GATE_FAILED+=("$lwid"); continue; }
import json, os, sqlite3, sys
db, ledger, lib, lwid = sys.argv[1:5]
sys.path.insert(0, lib); from stage import binding_state
from admission import bound_run_owner, supersession
state_root = os.path.dirname(os.path.abspath(ledger))
b = binding_state(ledger, lwid)
runs = {r["run_id"]: dict(r) for r in (b.get("runs") or [])}
for rid in b.get("run_ids") or []:
    runs.setdefault(rid, {"run_id": rid})
conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True) if os.path.isfile(db) else None
for rid, r in runs.items():
    row = conn.execute("SELECT eval_id, object_id, store_uri, lifecycle_state, extra_json FROM run WHERE run_id = ?", (rid,)).fetchone() if conn else None
    if row:
        r.setdefault("eval_id", row[0]); r["eval_id"] = r.get("eval_id") or row[0]; r["object"] = r.get("object") or row[1]; r["store_uri"] = row[2]
        # D3003 (1): the row's extra_json rides along — the climb's flat root is recorded there (extra.climb_root, filed by the producer's closer)
        try:
            r["extra"] = json.loads(row[4] or "{}") if row[4] else {}
        except json.JSONDecodeError:
            r["extra"] = {}
        # admission-fix (D878(i)): a run a `supersede` ruling retired is probed at its SUCCESSOR's recorded store (the consolidated store the
        # ruling names) for THIS unit's shards, on the cluster of the unit that binds the successor; the record carries the successor so
        # admission can tie the probe to the registry's ruling. No successor named: the run keeps its own (null) store and refuses by name there
        if row[3] == "superseded":
            sup = supersession(conn, rid) or {}
            if sup.get("successor"):
                srow = conn.execute("SELECT store_uri FROM run WHERE run_id = ?", (sup["successor"],)).fetchone()
                own = bound_run_owner(state_root, sup["successor"])
                r["successor"] = {"run_id": sup["successor"], "ruling_id": sup.get("ruling_id"), "store_uri": srow[0] if srow else None,
                                  "cluster": (own[1].get("cluster") if own else None), "owner": (own[0]["logical_work_id"] if own else None)}
                r["store_uri"] = srow[0] if srow else None
    r.setdefault("array_task", None); r.setdefault("store_uri", None); r.setdefault("successor", None); r.setdefault("extra", {})
print(json.dumps(list(runs.values())))
PYEOF
      shared="$(cluster_var "$cl" CLUSTER_SHARED_ROOT)"
      probe_shipped "$cl" || { repro_note "gate $fam: $lwid: the current coverage_probe.py could not be shipped to $cl (rc $?: 255 = the host could not be reached, 1 = the digest would not verify) — no coverage recorded"; GATE_FAILED+=("$lwid"); continue; }
      probe_remote="$PROBE_PATH"
      # a superseded run's probe runs on its SUCCESSOR's cluster — with the same shipped probe there
      probe_map=""; for pl in $(cluster_labels); do probe_shipped "$pl" || continue; probe_map="$probe_map${probe_map:+;}$pl=$PROBE_PATH"; done
      # ONE probe-plan line PER CALL (lib/stage.py probe_calls): a climb run bound to several array tasks (the shared-run shape and its companion
      # key) is several lines, `--shard-index k` each; the SIXTH column names the task, the SEVENTH where the store came from (store_source), and
      # `stage.py coverage --probes-jsonl` folds a run's documents into its one record. The plan goes to a FILE whose exit status is read (a
      # refused producer inside a process substitution left an empty loop and lost its status). The tabs are turned into unit
      # separators before `read`: a tab is IFS whitespace, so a run of tabs is ONE delimiter and an empty column would shift every column after it.
      "${STAGE_PY[@]}" probe-plan --state-root "$STATE" --logical-work-id "$lwid" --runs-json "@$T.runs_reg.json" --shared "$shared" --probe "$probe_remote" --probe-map "$probe_map" > "$T.plan.tsv" \
        || { repro_note "gate $fam: $lwid: probe-plan refused (above) — no coverage recorded"; GATE_FAILED+=("$lwid"); continue; }
      : > "$T.results.jsonl"; j=0
      while IFS=$'\037' read -r rid host argv why pcl task src; do
        [ -n "$rid" ] || continue
        j=$((j+1)); tcl="${pcl:-$cl}"; rc=0; whyarg=(); pout="$T.probe.$j.out"; perr="$T.probe.$j.err"
        # the call's stdout AND stderr are captured to files and handed to probe-append: a traceback, an ssh refusal or `Argument list too long`
        # is recorded by name (an earlier `2>/dev/null` made every one read as "the probe printed no JSON"); an rc != 0 is a MECHANICAL failure
        case "$host" in
          cluster)      repro_ssh "$tcl" "python3 $argv" > "$pout" 2> "$perr"; rc=$?;;
          grading_host) eval "set -- ${argv#* }"; "$PY" "$HERE/lib/coverage_probe.py" "$@" > "$pout" 2> "$perr"; rc=$?;;
          none)         : > "$pout"; : > "$perr"; whyarg=(--why "$why");;
          *)            : > "$pout"; : > "$perr"; whyarg=(--why "unknown probe host $host");;
        esac
        # the call's own status first, then probe-append's: rc 3 = the line IS recorded and NO document was read — the call exited non-zero, or
        # exited 0 and printed no JSON object (a login shell's banner on stdout before the document; a probe that printed prose) — a READER failure
        # marked on the unit's family exactly like an ssh rc. Before this the rc-0 no-JSON case was recorded as probe_error and
        # returned 0, entered no failure watermark, and the pass wrote a fresh `pass` verdict over a record that names a failure per run.
        [ "$rc" = 0 ] || GATE_FAILED+=("$lwid")
        "${STAGE_PY[@]}" probe-append --results "$T.results.jsonl" --run-id "$rid" --array-task "$task" --out-file "$pout" --err-file "$perr" --rc "$rc" --store-source "$src" ${whyarg[@]+"${whyarg[@]}"} \
          || { parc=$?; GATE_FAILED+=("$lwid"); how="exited $rc"; [ "$rc" = 0 ] && how="exited 0 and printed no JSON document on stdout (a login banner before the document? prose?)"
               if [ "$parc" = 3 ]; then repro_note "gate $fam: $lwid run $rid${task:+ (array task $task)}: no probe document was read — the call $how; the coverage record names it per run; a reader failure, never a coverage fact"
               else repro_note "gate $fam: $lwid run $rid: probe-append exited $parc (above) — the call's result was not recorded"; fi; }
      done < <(tr '\t' '\037' < "$T.plan.tsv")
      "${STAGE_PY[@]}" coverage --state-root "$STATE" --logical-work-id "$lwid" --runs-json "@$T.runs_reg.json" --probes-jsonl "$T.results.jsonl" \
        --probe-sha "$PROBE_SHA" --probe-sha256 "$PROBE_FILE_SHA256" --inventory-sha "$INV_SHA" --contract "@$CONTRACT_JSON" \
        || { repro_note "gate $fam: $lwid: coverage refused (above) — no coverage recorded"; GATE_FAILED+=("$lwid"); }
    done < "$GATE_TMP/units.$fam.tsv"
    # the sanity verdict: the headline from the registry (run_metric of the family's headline metric) or from the flags — recorded with BOTH shas
    # and the coverage contract it was ruled under; a `--run-ids` list is bounded by the inventory (<= 300 ids x ~50 B).
    # A FAILED pass writes NO verdict: a verdict claims "ruled now, under this probe, over the coverage this pass read" —
    # and a unit that hit a transport / reader failure above kept its EARLIER coverage record (or a record naming the failure per run), so a
    # verdict written here would stamp a fresh `pass` with this pass's provenance over coverage this pass never read; admission ties a record to
    # its run ids and the contract, never to the verdict's pass, and would have admitted the family on the stale record (a `fail` or absent
    # verdict newly turned `pass`; a store rewritten since the record was collected). Per FAMILY: a family whose every unit read is ruled even
    # when another family's unit failed (its own coverage is fresh); gate/<fam>.json is left exactly as it was, said by name, and the exit is 1.
    # And the verdict step's OWN status is read: `stage.py verdict` returns 0 for a RECORDED verdict, pass or fail, 2 for
    # a refusal by name, else an error (a write that failed) — a `|| true` here once swallowed a verdict that was never written (its traceback's
    # rc 1 was a persisted `fail`'s status too), so the pass exited 0 with the OLD verdict standing over the coverage it had just refreshed.
    fam_failed_n=$(( ${#GATE_FAILED[@]} - fam_failed_from ))
    RULING=0; { [ "$VERDICT_AUTO" = 1 ] || { [ -n "$HEADLINE" ] && [ -n "$RANGE" ]; }; } && RULING=1
    if [ "$VERDICT_AUTO" = 1 ]; then rerun="06 --gate $fam --verdict"; else rerun="06 --gate $fam --headline $HEADLINE --range $RANGE"; fi
    if [ "$RULING" = 1 ] && [ "$fam_failed_n" -gt 0 ]; then
      fam_failed_list="$(printf '%s\n' "${GATE_FAILED[@]:$fam_failed_from}" | awk '!seen[$0]++' | paste -sd' ' -)"
      fam_failed_uniq="$(printf '%s\n' "${GATE_FAILED[@]:$fam_failed_from}" | awk '!seen[$0]++' | wc -l | tr -d ' ')"
      repro_note "gate $fam: NO verdict written — $fam_failed_uniq unit(s) of the family hit a reader / transport failure in this pass ($fam_failed_list): a verdict records the coverage this pass read under the probe it names, and this pass read none for those units (their earlier coverage records stand, or name the failure per run); gate/$fam.json is left as it was ($(gate_prior_verdict "$fam")); fix the cause and re-run $rerun"
    elif [ "$RULING" = 1 ]; then
      # the verdict's `run_ids` = the runs this pass RULES: the family's registered bindings NOW (stage.py family_bound_run_ids — the ONE rule
      # admission reads too). A pass verdict is bound to the RUNS it ruled, not to a coverage generation —
      # admission refuses the family when the standing verdict's set differs from the family's bindings at check time (a resubmission re-bound
      # a unit since, and no verdict was written over the new ids), so a bind-only re-gate or a verdict step that failed after a resubmission
      # leaves a pass that admits nothing new; a coverage refresh or a new probe sha over the SAME runs is a re-read and still admits
      if ! runs_csv="$("$PY" -c 'import json,sys; sys.path.insert(0, sys.argv[1]); from stage import family_bound_run_ids; print(",".join(family_bound_run_ids(sys.argv[3], json.load(open(sys.argv[2]))["units"], sys.argv[4])))' "$HERE/lib" "$INV" "$STATE/submissions.jsonl" "$fam")"; then
        repro_note "gate $fam: NO verdict written — the family's bound run ids could not be read from the ledger (above); gate/$fam.json is left as it was ($(gate_prior_verdict "$fam")); fix the cause and re-run $rerun"
        GATE_VERDICT_FAILED+=("$fam")
      else
        verdict_args=(--headline "$HEADLINE" --range "$RANGE"); vnote="recorded by 06_eval_waves.sh --gate; mirror into decisions.md"; hrec=""; hrc=0
        if [ "$VERDICT_AUTO" = 1 ]; then
          # --verdict: the headline READ from the registry (read-only) over the SAME runs the verdict binds — every unsteered leg, the binding
          # one for the family's expected direction, the range from the family's row of lib/gate_headline.py's direction table; the record goes
          # to stage.py verdict as a FILE, so no number passes through this shell. rc 3 = not ready to rule (a unit not registered yet, an eval with
          # no unsteered leg, a leg not graded yet): the family is skipped, never ruled on part of it
          hrec="$GATE_TMP/headline.$fam.json"
          "$PY" "$HERE/lib/gate_headline.py" read --registry "$REGISTRY_DB" --state-root "$STATE" --family "$fam" --out "$hrec"; hrc=$?
          verdict_args=(--headline-record "@$hrec"); vnote="recorded by 06_eval_waves.sh --gate --verdict: the headline read from the registry"
        fi
        if [ "$hrc" = 3 ]; then
          repro_note "gate $fam: SKIPPED, no verdict written — a unit is not registered yet or an unsteered leg is not graded yet (the reasons above); gate/$fam.json is left as it was ($(gate_prior_verdict "$fam")); re-run $rerun once its units are registered and graded"
          GATE_SKIPPED+=("$fam")
        elif [ "$hrc" != 0 ]; then
          repro_note "gate $fam: NO verdict written — its headline could not be read from the registry (lib/gate_headline.py exited $hrc, above); gate/$fam.json is left as it was ($(gate_prior_verdict "$fam")); fix the cause and re-run $rerun"
          GATE_VERDICT_FAILED+=("$fam")
        else
          repro_run "${STAGE_PY[@]}" verdict --state-root "$STATE" --family "$fam" "${verdict_args[@]}" --run-ids "$runs_csv" --note "$vnote" \
            --inventory-sha "$INV_SHA" --probe-sha "$PROBE_SHA" --probe-sha256 "$PROBE_FILE_SHA256" --contract "@$CONTRACT_JSON" $(repro_dry_flag); vrc=$?
          if [ "$vrc" != 0 ]; then
            repro_note "gate $fam: NO verdict written — stage.py verdict exited $vrc (its reason is above: 2 = refused by name, anything else an error such as a write that failed); gate/$fam.json is left as it was ($(gate_prior_verdict "$fam")); fix the cause and re-run $rerun"
            GATE_VERDICT_FAILED+=("$fam")
          elif [ "$VERDICT_AUTO" = 1 ]; then
            # the verdict this pass ruled, read back from the file it wrote (a dry run: the one the record implies, by the same rule)
            vsrc="$STATE/gate/$fam.json"; [ "$REPRO_DRY_RUN" = 1 ] && vsrc="$hrec"
            ruled="$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("verdict") or "")' "$vsrc")"
            case "$ruled" in
              pass) GATE_PASSED+=("$fam");;
              fail) GATE_RULED_FAIL+=("$fam");;
              *) repro_note "gate $fam: the verdict could not be read back from $vsrc (got '$ruled')"; GATE_VERDICT_FAILED+=("$fam");;
            esac
          fi
        fi
      fi
    else
      repro_note "gate $fam: pass --verdict (each family's headline read from the registry) or --headline <value> --range <lo,hi> (one family's by hand) to write gate/$fam.json; the daemon (08_grade.sh) must have graded the runs first"
    fi
  done
  if [ "$VERDICT_AUTO" = 1 ]; then
    repro_say "gate verdicts: pass: $(gate_list ${GATE_PASSED[@]+"${GATE_PASSED[@]}"}); fail: $(gate_list ${GATE_RULED_FAIL[@]+"${GATE_RULED_FAIL[@]}"}); skipped, not ready to rule: $(gate_list ${GATE_SKIPPED[@]+"${GATE_SKIPPED[@]}"})"
  fi
  if [ ${#GATE_FAILED[@]} -gt 0 ] || [ ${#GATE_VERDICT_FAILED[@]} -gt 0 ]; then
    msg="REFUSED: 06 --gate:"
    if [ ${#GATE_FAILED[@]} -gt 0 ]; then
      failed_list="$(printf '%s\n' "${GATE_FAILED[@]}" | awk '!seen[$0]++' | paste -sd' ' -)"
      msg="$msg $(printf '%s\n' "${GATE_FAILED[@]}" | awk '!seen[$0]++' | wc -l | tr -d ' ') unit(s) hit a reader / transport failure (the notes above): $failed_list — a unit named here was not bound, or its coverage record names the failure per run, and NO verdict was written for its family (gate/<family>.json stands as it was);"
    fi
    if [ ${#GATE_VERDICT_FAILED[@]} -gt 0 ]; then
      vfailed_list="$(printf '%s\n' "${GATE_VERDICT_FAILED[@]}" | awk '!seen[$0]++' | paste -sd' ' -)"
      msg="$msg $(printf '%s\n' "${GATE_VERDICT_FAILED[@]}" | awk '!seen[$0]++' | wc -l | tr -d ' ') family verdict(s) could not be recorded (the notes above): $vfailed_list — NO verdict was written for the family (gate/<family>.json stands as it was);"
    fi
    echo "$msg fix the cause and re-run 06 --gate (exit 1 is mechanical, never 'not yet': a unit whose pending file has not landed exits 0)" >&2
    exit 1
  fi
  if [ "$VERDICT_AUTO" = 1 ] && [ $(( ${#GATE_SKIPPED[@]} + ${#GATE_RULED_FAIL[@]} )) -gt 0 ]; then
    echo "06 --gate --verdict: skipped, not ready to rule: $(gate_list ${GATE_SKIPPED[@]+"${GATE_SKIPPED[@]}"}); ruled fail: $(gate_list ${GATE_RULED_FAIL[@]+"${GATE_RULED_FAIL[@]}"}) — exit 3: re-run once the skipped families' units are registered and graded; a family ruled fail admits nothing" >&2
    exit 3
  fi
  exit 0
fi

# =============================================================================== the waves
[ -n "$VIEW" ] || repro_fatal "--view is required (a phase view of the model: $("$PY" "$HERE/lib/models.py" --table "$(repro_models_table)" views "$K" | tr '\n' ' ')) or --gate <family>"
repro_require_view "$K" "$VIEW"
BASE_VIEW="$("$PY" "$HERE/lib/models.py" --table "$(repro_models_table)" baseline-view "$K")" || exit 2   # u0, or the single-phase recipe's one wave
[ -n "$FAMILIES" ] || repro_fatal "--family <list>|all is required"
FAMILIES_ALL=0; [ "$FAMILIES" = all ] && FAMILIES_ALL=1
if [ "$FAMILIES" = all ]; then
  FAMILIES="${ALL_FAMILIES// /,}"
  # a family the models table EXCLUDES from this view (lib/models.py excluded-families; no padding on s2r) is dropped from
  # `all` and said so — naming it explicitly still refuses in the plan (lib/stage.py unit_plan)
  while IFS=$'\t' read -r xf xwhy; do
    [ -n "$xf" ] || continue
    FAMILIES="$(printf '%s' ",$FAMILIES," | sed "s/,$xf,/,/" | sed 's/^,//; s/,$//')"
    repro_note "family $xf is excluded for view $VIEW by ruling ($xwhy) — dropped from --family all"
  done < <("$PY" "$HERE/lib/models.py" --table "$(repro_models_table)" excluded-families "$K" "$VIEW")
fi
# a family the FREEZE skipped by name (the inventory's skipped_families: the Petri family when no copy of its frozen scenario bundle was
# found — lib/stage.py petri_skip) is dropped from `all` and said so; naming it refuses, since its units were never frozen
SKIPPED_TSV=""
if [ -f "$INV" ]; then
  SKIPPED_TSV="$("$PY" -c 'import json, sys
for f, why in sorted((json.load(open(sys.argv[1])).get("skipped_families") or {}).items()):
    print(f + "\t" + " ".join(str(why).split()))' "$INV")" || repro_fatal "the frozen inventory $INV does not parse"
fi
while IFS=$'\t' read -r sf swhy; do
  [ -n "$sf" ] || continue
  if [ "$FAMILIES_ALL" = 1 ]; then
    FAMILIES="$(printf '%s' ",$FAMILIES," | sed "s/,$sf,/,/" | sed 's/^,//; s/,$//')"
    repro_note "family $sf was skipped at the freeze ($swhy) — dropped from --family all; restore it and re-run 05_stage_roster.sh --step freeze to include it"
  else
    case ",$FAMILIES," in *",$sf,"*) repro_fatal "family $sf was skipped at the freeze ($swhy) — it has no frozen unit to submit; restore it and re-run 05_stage_roster.sh --step freeze";; esac
  fi
done <<< "$SKIPPED_TSV"
for f in ${FAMILIES//,/ }; do case " $ALL_FAMILIES " in *" $f "*) ;; *) repro_fatal "unknown family '$f' ($ALL_FAMILIES)";; esac; done
# the Petri OVERFLOW RE-RUN PASS (petri-completion, 2026-10-13): one unit, the petri family alone, no selection beside the ledgered legs'
case "$PASS" in first|overflow) ;; *) repro_fatal "--pass must be first or overflow, got '$PASS'";; esac
if [ "$PASS" = overflow ]; then
  [ "$FAMILIES" = petri ] || repro_fatal "--pass overflow is the Petri overflow re-run pass: --family petri, no other family (got --family $FAMILIES)"
  [ -z "${UNITS:-}" ] || repro_fatal "--pass overflow takes no --units: the pass is the one unit petri.overflow"
  [ "$CELLS_GIVEN" = 0 ] || repro_fatal "--pass overflow takes no --cells: the re-run proxy serves the cells of the legs the first pass left on context-overflow ledgers (read from the daemon's record), never a selection"
  [ "$TWIN" = 0 ] || repro_fatal "--pass overflow takes no --twin: the re-run proxy follows the first pass's cluster"
  # the pass is the ONE selected unit: the `--units` path below submits it alone and writes no WAVE file (a follow-up unit beside the wave; the
  # daemon's discovery reads the wave file + the petri.overflow row)
  UNITS=petri.overflow
fi
# a multi-turn GROUP (attacker pool + its targets) has no two-cluster election in this version: --twin is REFUSED for it rather
# than printed past and double-submitted; the groups are partitioned across every configured cluster instead (--clusters selects)
[ "$TWIN" = 1 ] && case ",$FAMILIES," in *,multiturn,*) repro_fatal "--twin is not implemented for multi-turn groups (attacker pool + its targets): submit --family multiturn WITHOUT --twin (the units are partitioned across every configured cluster, one group per cluster; --clusters selects the groups this call submits) and twin the other families in a separate call";; esac
repro_require_runnable "$K"
TAG="$K$( [ "$VIEW" = main ] || echo "_$VIEW" )"
CFG="configs/sdfbase_full22_${TAG}.yaml"
[ -f "$REPRO_REPO/$CFG" ] || { [ "$REPRO_DRY_RUN" = 1 ] && repro_note "$CFG is not generated yet (05_stage_roster.sh --step configs after the roster carries the view); the dry run composes the lines anyway" || repro_fatal "no $CFG — run 05_stage_roster.sh first"; }
if [ "$VIEW" != "$BASE_VIEW" ] || [ "$BASE_VIEW" != u0 ]; then
  if [ "$VIEW" = "$BASE_VIEW" ]; then
    repro_say "single-phase recipe: view $VIEW is the ONE wave — every arm beside its own unsteered legs (none + the prompted instructions run inside it); the unsteered-first gate (D15) has nothing to admit and is not applied; the compute replan (§6) applies to its steered legs"
  else
    repro_say "view $VIEW is STEERED: the global admission gate (D15) and the compute replan (§6) apply"
  fi
  args=(); for f in ${FORCE[@]+"${FORCE[@]}"}; do args+=(--force-gate "$f"); done
  # the steered waves are PRICED on the plan's serving rows (multi-turn, padding), so they need a COMPLETE plan, not the file's
  # existence (lib/common.sh repro_require_compute_plan's `complete` mode names the pending rows). The dry
  # run reads an on-disk plan too (read-only) and refuses an incomplete one by name; with no plan yet it only says what it needs.
  if [ "$REPRO_DRY_RUN" = 1 ]; then
    [ "$VIEW" = "$BASE_VIEW" ] || echo "DRY-RUN: $PY $HERE/lib/admission.py check --state-root $STATE --registry $REGISTRY_DB ${args[*]:-}"
    if [ -f "$STATE/compute_plan.json" ]; then repro_require_compute_plan "$K" complete; echo "DRY-RUN: compute plan $STATE/compute_plan.json is complete (tools/replan_compute.py)"
    else echo "DRY-RUN: require a COMPLETE $STATE/compute_plan.json (tools/replan_compute.py --probe the certified probe_verdict.json)"; fi
  else
    [ "$VIEW" = "$BASE_VIEW" ] || "$PY" "$HERE/lib/admission.py" check --state-root "$STATE" --registry "$REGISTRY_DB" "${args[@]+"${args[@]}"}" || repro_fatal "the unsteered battery does not admit view $VIEW yet (the refusals above name the units)"
    repro_require_compute_plan "$K" complete
  fi
fi
repro_require_clean_tree
# ONE plan for the freeze and the submissions (lib/stage.py unit_plan): the TSV rows the loop below iterates. The PLACEMENT universe is
# every configured label (the multi-turn groups are partitioned over it, the same every call); --clusters only SELECTS the placed
# units this call submits (`plan --only-clusters`). The inventory check below reads the UNFILTERED plan: the family's whole unit set.
ALL_CLUSTERS="$(cluster_labels | tr ' ' ',')"
PLAN_ARGS=(--model "$K" --view "$VIEW" --families "$FAMILIES" --nshards "$NSHARDS" --sha "$WAVE_SHA" --clusters "$ALL_CLUSTERS" --models-table "$(repro_models_table)")
# a DECLARED group move (2026-10-02: a cluster that cannot stand the serving venv up hands its multi-turn group to one that can): the plan
# places the group at the destination and sizes that label's pool from the moved count; the manifest records the declaration with its reason;
# the plan reads the manifest's declared moves on every later call (sticky), so nothing re-partitions the group back
if [ ${#MOVE[@]} -gt 0 ] && [ -z "$MOVE_REASON" ]; then repro_fatal "--move-group needs --move-reason (the provenance of the move: why the group leaves its cluster)"; fi
MOVE_ARGS=(); for m in ${MOVE[@]+"${MOVE[@]}"}; do MOVE_ARGS+=(--move-group "$m"); done
PLAN_ARGS+=(--ledger "$STATE/attempts.jsonl" ${MOVE_ARGS[@]+"${MOVE_ARGS[@]}"} ${MOVE_REMOVE:+--re-move} ${CELLS_SEL:+--cells "$CELLS_SEL"})   # move-knob-3: the plan runs FIRST and re-points under the same flag ensure records; cells-fix: the ONE cells selection (served + registered)
# 26d: the live plan runs with the knobs the frozen inventory pinned (the attacker cap, the seed base, the shard grouping), or refuses
[ -f "$STATE/gate/expected_units.json" ] && PLAN_ARGS+=(--inventory "$STATE/gate/expected_units.json")
if [ "$PASS" = overflow ]; then
  # THE EVENT the pass keys on: the daemon's record of the FIRST pass holds, for its bound job, an attempt settled `incomplete` whose result reads
  # `overflow_pending` (lib/petri_overflow.py reads it, the drive's result and the drive's own done-ness reading — petri_run.py legs-status — and
  # derives the re-run proxy's cells, or the bound overflow attempt's declaration verbatim; every refusal is by name). A dry run on a site the
  # daemon never wrote to says so and composes nothing (the README-sequence's bare site): there is no declaration to print.
  OVF_REC="$STATE/grade/petri_drive.$TAG.json"
  if [ ! -f "$OVF_REC" ] && [ "$REPRO_DRY_RUN" = 1 ]; then
    echo "DRY-RUN: --pass overflow reads the daemon's record $OVF_REC (08_grade.sh --daemon writes it): the first-pass attempt must be settled incomplete with verdict overflow_pending; a real call refuses petri_overflow_no_record until then, and then submits petri_proxy.sbatch $K $VIEW $WAVE_SHA overflow-rerun <the ledgered legs' cells> as unit petri.overflow (name f22-petri-$TAG-rerun, --time=$SISTEER_TIME_PETRI) on the cluster the first pass is bound on; nothing composed"
    repro_say "06 dry run ends here: 0 unit(s) placed (overflow pass, dry run without the daemon's record)"; exit 0   # not the pass's closing line: that one closes a REAL pass, after the wave write (a placement test pins its place in this source)
  fi
  # THE TOKEN: a `--pass overflow` submit interrupted between its reservation and the new attempt's binding — a first
  # submission's or a --resubmit's (hole 3) — leaves the overflow unit's TRANSITION TOKEN, which lib/petri_overflow.py refuses by name
  # (petri_overflow_transition). This preflight runs BEFORE any funnel call, so the funnel's own reconciliation (`submit` reconciles a standing
  # token first) was never reached and the same command refused forever. A real call reconciles the token THROUGH the funnel first
  # (submit_twin.py reconcile-token: the launched job identified by its nonce read back and ADOPTED — ledger, submissions row, `submitted`
  # binding, watcher, the 24d receipt of a cross-cluster move — or the token cleared as a non-launch; an ambiguous one STANDS, refused by
  # name, a human decides) and only then reads the binding, the declaration and the ledgers, all fresh. A dry run writes nothing: it says so.
  OVF_LWID="$("$PY" -c 'import sys; sys.path.insert(0, sys.argv[1]); import stage as ST; print(ST.logical_work_id(sys.argv[2], sys.argv[3], "petri", ST.PETRI_OVERFLOW_UNIT))' "$HERE/lib" "$K" "$VIEW")"
  OVF_BIND="$("$PY" -c 'import sys; sys.path.insert(0, sys.argv[1]); from stage import binding_state; print(binding_state(sys.argv[2], sys.argv[3])["state"])' "$HERE/lib" "$STATE/submissions.jsonl" "$OVF_LWID")"
  if [ "$OVF_BIND" = transition ]; then
    if [ "$REPRO_DRY_RUN" = 1 ]; then
      echo "DRY-RUN: the overflow unit $OVF_LWID is mid re-run (a transition token: a submit was interrupted between its reservation and the new attempt's binding); a real call reconciles it through the funnel first ($PY $HERE/lib/submit_twin.py --config $SISTEER_CONFIG --model $K reconcile-token --lwid $OVF_LWID: the launched job adopted by its nonce read back, or the token cleared as a non-launch; an ambiguous one stands) and then reads the declaration; nothing decided in a dry run"
      repro_say "06 dry run ends here: 0 unit(s) placed (overflow pass, dry run with the overflow unit's transition token standing)"; exit 0
    fi
    repro_say "overflow pass: the overflow unit $OVF_LWID is mid re-run (a transition token) — reconciled through the funnel before the declaration is read (submit_twin.py reconcile-token)"
    slurm_reconcile_token --lwid "$OVF_LWID" \
      || repro_fatal "petri_overflow_transition: the overflow unit $OVF_LWID's transition token STANDS (the funnel's reconciliation refused, above: nothing adopted, nothing cleared, nothing launched) — resolve it as the refusal says and re-run the same command"
  fi
  OVF_JSON="$("$PY" "$HERE/lib/petri_overflow.py" --state-root "$STATE" --model "$K" --view "$VIEW" --python "$PY" --repo "$REPRO_REPO" --models-table "$(repro_models_table)")" \
    || repro_fatal "the overflow pass is not ready (above)"
  OVF_CELLS="$("$PY" -c 'import json,sys; print(",".join(json.loads(sys.argv[1])["cells"]))' "$OVF_JSON")"
  # THE CLUSTER: the re-run proxy runs on the cluster the FIRST PASS is bound on — 05 --clusters stages the object bank and
  # builds the bank view per cluster, and the proxy's standup requires both (petri_proxy.sbatch f22_bank_view). The row used to carry `*`, which
  # the loop below expands to every configured label and the funnel's first label takes, so a first pass on a non-primary cluster got its re-run
  # proxy where nothing was staged. The first pass's binding names the cluster (lib/petri_overflow.py first_cluster); the plan's row is pinned to
  # it (--overflow-cluster); an explicit --clusters that excludes it is refused by name; the pass's mirror, geometry staging and submission go to
  # that ONE cluster. A bound overflow unit placed elsewhere before this rule is moved back by the funnel's cross-cluster replacement (24d).
  OVF_CLUSTER="$("$PY" -c 'import json,sys; print(json.loads(sys.argv[1])["first_cluster"])' "$OVF_JSON")"
  OVF_JOB="$("$PY" -c 'import json,sys; print(json.loads(sys.argv[1])["first_job_id"])' "$OVF_JSON")"
  case " $(cluster_labels) " in *" $OVF_CLUSTER "*) ;; *) repro_fatal "petri_overflow_cluster_unconfigured: the first pass (job $OVF_JOB) is bound on cluster '$OVF_CLUSTER', not one of SISTEER_CLUSTERS ($(cluster_labels)) — the re-run proxy runs where the first pass ran (its object bank and bank view are there); restore the label in the site config";; esac
  if [ -n "$CLUSTERS_GIVEN" ]; then case ",$CLUSTERS," in *",$OVF_CLUSTER,"*) ;; *) repro_fatal "petri_overflow_cluster_excluded: --clusters $CLUSTERS excludes '$OVF_CLUSTER', the cluster the first pass (job $OVF_JOB) is bound on — the re-run proxy runs there (its object bank and bank view are there), never on a selection; drop --clusters or name $OVF_CLUSTER";; esac; fi
  CLUSTERS="$OVF_CLUSTER"
  repro_say "overflow pass: $("$PY" -c 'import json,sys; d=json.loads(sys.argv[1]); print("first pass job %s attempt %s overflow_pending on %s; still ledgered: %s; done since: %s; declaration (%s): %s%s" % (d["first_job_id"], d["first_attempt"], d["first_cluster"], ", ".join("%s:%s" % (l["idx"], l["cell_id"]) for l in d["still_ledgered"]), ", ".join("%s:%s" % (l["idx"], l["cell_id"]) for l in d["done_legs"]) or "none", d["declaration_source"], ",".join(d["cells"]), ("; the bound overflow proxy (job %s) sits on %s and moves back to %s" % (d["overflow_job_id"], d["overflow_cluster"], d["first_cluster"])) if d.get("overflow_cluster") and d["overflow_cluster"] != d["first_cluster"] else ""))' "$OVF_JSON")"
  PLAN_ARGS+=(--pass overflow --overflow-cells "$OVF_CELLS" --overflow-cluster "$OVF_CLUSTER")
fi
PLAN_TSV="$("${STAGE_PY[@]}" plan "${PLAN_ARGS[@]}" --only-clusters "$CLUSTERS")" || repro_fatal "the unit plan refused (above)"
if [ "$VIEW" = "$BASE_VIEW" ] && [ "$PASS" = first ]; then   # the baseline wave (u0, or the single-phase recipe's one wave) IS the gate's inventory: it must be the frozen plan, at the frozen sha (the overflow pass is a follow-up unit the freeze never inventories)
  if [ -f "$INV" ]; then
    PLAN_JSON="$("${STAGE_PY[@]}" plan "${PLAN_ARGS[@]}" --json)" || repro_fatal "the unit plan refused (above)"
    "$PY" - "$INV" "$WAVE_SHA" "$FAMILIES" <(printf '%s' "$PLAN_JSON") <<'PYEOF' || repro_fatal "the submission plan disagrees with the frozen inventory $INV (above) — re-freeze (05_stage_roster.sh --step freeze) at this sha, or pass --wave-sha <frozen sha>"
import json, sys
plan, inv, sha, fams = json.load(open(sys.argv[4])), json.load(open(sys.argv[1])), sys.argv[2], sys.argv[3].split(",")
bad = []
if inv.get("schema") != 2: bad.append(f"inventory schema {inv.get('schema')!r} != 2 (re-freeze)")
# D200: the tree is PROVENANCE, not the inventory's identity — a wave sha other than the freeze sha is admitted here; the funnel verifies
# each unit's eval CONTRACT at this tree before minting and records the per-unit sha (stage.py provenance; admission compares per unit)
if inv.get("code_sha") != sha: print(f"note: wave sha {sha[:12]} is not the frozen tree {str(inv.get('code_sha'))[:12]}: per-unit sha provenance applies (D200) — each unit's contract is verified at this tree by the funnel", file=sys.stderr)
# the SUBMISSION family (`stage_family`: the family whose 06 call places the unit) decides membership, not the EVAL family
# (`family`: the grader/gate's). The BoN 512 store is eval family `bon` but is placed by `--family static`, so a `--family bon`
# call used to refuse it as "frozen but not planned" and a per-family re-run of the wave was impossible (D158 pass, 2026-09-30).
want = {u["logical_work_id"]: {e: ev["expected_shards"] for e, ev in u["evals"].items()} for u in inv["units"] if (u.get("stage_family") or u["family"]) in fams}
have = {r["logical_work_id"]: r["expected_shards"] for r in plan if r["registers"] and (r.get("pass") or "first") == "first"}   # a pass row (petri.overflow) is never inventoried
for lw in sorted(set(want) ^ set(have)): bad.append(f"unit {lw}: {'frozen but not planned' if lw in want else 'planned but not frozen'}")
for lw in sorted(set(want) & set(have)):
    if want[lw] != have[lw]: bad.append(f"unit {lw}: frozen shards {want[lw]} != planned {have[lw]} (--nshards?)")
for b in bad: print(f"REFUSED: {b}", file=sys.stderr)
sys.exit(1 if bad else 0)
PYEOF
    repro_say "u0 plan == frozen inventory ($INV) at sha $WAVE_SHA"
  elif [ "$REPRO_DRY_RUN" = 1 ]; then repro_note "no frozen inventory at $INV yet (05_stage_roster.sh --step freeze); a real $VIEW wave refuses without it"
  else repro_fatal "no frozen inventory at $INV — run 05_stage_roster.sh --step freeze first (the $VIEW wave IS the gate's baseline and must match it)"
  fi
fi
# the ORDERED placement universe is pinned with the wave (gate/expected_units.placement.json for u0 — written at freeze, or derived ONCE
# from the recorded attempts for a wave that predates it — and expected_units.placement.<view>.json for every other view: each view its own
# namespace); a cluster list that differs from it (order or membership) or a planned multi-turn assignment that moved
# refuses by name — a reordered site config between per-cluster calls would otherwise re-place the remaining targets and leave the wave
# silently incomplete. For EVERY view: inside the u0 gate alone, an s2r declaration was never persisted or checked.
DECLARED_MOVES_TSV=""
case ",$FAMILIES," in *,multiturn,*)
  [ -n "${PLAN_JSON:-}" ] || PLAN_JSON="$("${STAGE_PY[@]}" plan "${PLAN_ARGS[@]}" --json)" || repro_fatal "the unit plan refused (above)"
  "${STAGE_PY[@]}" placement-manifest ensure --model "$K" --view "$VIEW" --clusters "$ALL_CLUSTERS" --ledger "$STATE/attempts.jsonl" \
    --models-table "$(repro_models_table)" --nshards "$NSHARDS" --sha "$WAVE_SHA" ${MOVE_ARGS[@]+"${MOVE_ARGS[@]}"} ${MOVE_REASON:+--move-reason "$MOVE_REASON"} \
    ${MOVE_REMOVE:+--re-move} $(repro_dry_flag) || repro_fatal "the placement manifest refused (above)"
  "${STAGE_PY[@]}" placement-manifest check --model "$K" --view "$VIEW" --clusters "$ALL_CLUSTERS" --ledger "$STATE/attempts.jsonl" \
    --plan-json <(printf '%s' "$PLAN_JSON") $(repro_dry_flag) || repro_fatal "the placement check refused (above): this call's cluster list or the plan's multi-turn placement is not the wave's pinned one"
  # the PERSISTED declarations (read-only TSV: group, to, from, reason, …): mark-placed derives its --supersede + --reason from them on EVERY pass,
  # flagged or not (without them, a flag-free completion of a partially launched moved group could not establish the grown PLACED record)
  DECLARED_MOVES_TSV="$("${STAGE_PY[@]}" placement-manifest moves --model "$K" --view "$VIEW" --clusters "$ALL_CLUSTERS" --ledger "$STATE/attempts.jsonl" 2>/dev/null || true)"
  ;;
esac
# the pinned sha on EVERY target cluster BEFORE the first submission: mirror + clean detached checkout
slurm_mirror_clusters "$WAVE_SHA" "$CLUSTERS"
BASE_ENV="$(slurm_base_env)"
repro_say "06 eval_waves: model=$K view=$VIEW tag=$TAG families=$FAMILIES clusters=$CLUSTERS (placement over $ALL_CLUSTERS) twin=$TWIN resubmit=$RESUBMIT sha=$WAVE_SHA overrides=$OVERRIDE_ID"

# placement-fix round 3: a --resubmit pass submits EVERY replacement job below before it supersedes PLACED at the
# end, so a re-minted member that comes up while the pass is still submitting targets — or while a failed mark-placed awaits its retry — reads
# the record that predates it, finds nothing assigned to it and leaves exit_unbound; the supersession then assigns targets to a server that is
# gone. So BEFORE the first submission the pass marks each group's placement IN TRANSITION (attackers/PLACED.transition on the group's
# cluster, through the funnel): a member holds every exit while the marker is fresh; mark-placed clears it in the same locked step that
# commits a placement (placed / unchanged / superseded — a refusal leaves it, the sanctioned supersession being this pass's continuation);
# its lease (2 h by default) is the bound, so a pass nobody completes cannot pin an unused member for ever. A marker that cannot be written is
# fatal HERE, with nothing submitted: a replacement minted without it is exactly the member that leaves early.
case ",$FAMILIES," in *,multiturn,*)
  if [ "$RESUBMIT" = 1 ]; then
    for l in ${CLUSTERS//,/ }; do
      "$PY" "$HERE/lib/submit_twin.py" --config "$SISTEER_CONFIG" --model "$K" ${REPRO_OVERRIDES:+--overrides "$REPRO_OVERRIDES"} $(repro_dry_flag) placement-transition open --group "mt-$VIEW-$l" --cluster "$l" --reason "06 --resubmit pass on $WAVE_SHA" \
        || repro_fatal "the placement-transition marker for mt-$VIEW-$l could not be opened on $l (above) — nothing submitted: a replacement member minted without it could leave on the placement that predates it before this pass re-places the group"
    done
  fi;;
esac

SUBMITTED_NOW=(); declare -A GROUP_TARGETS=(); declare -A GROUP_MEMBERS=(); UNPLACED=()   # UNPLACED: `<group>\t<label>\t<remedy>` per NO LIVE MEMBER refusal (the pass ends INCOMPLETE below)
while IFS=$'\t' read -r sf fam unit lwid cl kind grp script name array tkey twin_ok args env depends; do
  [ -n "$lwid" ] || continue
  [ "$grp" != - ] && [ "$kind" = target ] && GROUP_TARGETS[$grp]="${GROUP_TARGETS[$grp]:-}${GROUP_TARGETS[$grp]:+,}$lwid"   # the group's targets, for mark-placed (31b)
  u="$fam.${unit//\//.}"
  # the group's member UNITS (`multiturn.attacker<i>`), recorded from the whole plan before the --units filter: the no-live-member remedy below
  # widens a --units selection by them, since the walk re-mints only the units it visits (placement-fix round 6)
  [ "$grp" != - ] && [ "$kind" = attacker ] && GROUP_MEMBERS[$grp]="${GROUP_MEMBERS[$grp]:-}${GROUP_MEMBERS[$grp]:+,}$u"
  if [ -n "${UNITS:-}" ]; then case ",$UNITS," in *",$u,"*) ;; *) continue;; esac; fi   # --units: only the named units go out this call
  [ "$cl" = "*" ] && cl="$CLUSTERS"
  # the row's walltime by its time_key (lib/stage.py): the Petri proxy's is the site's SISTEER_TIME_PETRI (a day by default — the paper's;
  # the proxy self-releases when the drive ends), never SISTEER_TIME_EVAL, which cut the demo's full drive at 04:00:00 (2026-10-13)
  case "$tkey" in gsm8k) t="0-01:30:00";; petri) t="$SISTEER_TIME_PETRI";; *) t="$SISTEER_TIME_EVAL";; esac
  extra=(); [ "$array" != - ] && extra+=(--array "$array"); [ "$grp" != - ] && extra+=(--group "$grp"); [ "$kind" != target ] && extra+=(--kind "$kind")
  [ "${depends:--}" != - ] && extra+=(--depends-on "$depends")   # runs after that unit's bound job (Slurm afterok; the funnel resolves it)
  [ "$RESUBMIT" = 1 ] && extra+=(--resubmit --walk)   # a family pass: a unit whose every task COMPLETED is skipped (results stand), never the wave's failure
  [ "$TWIN" = 1 ] && [ "$twin_ok" = 1 ] && extra+=(--twin)
  env_str="$BASE_ENV"; [ "$env" != - ] && env_str="$BASE_ENV $env"
  [ "$args" = - ] && args=""
  if ! slurm_submit --unit "$u" --lwid "$lwid" --script "$script" --name "$name" --args "$args" --clusters "$cl" --env "$env_str" --time "$t" --wave-sha "$WAVE_SHA" ${extra[@]+"${extra[@]}"}; then
    # a failed submission fails the WAVE — the partial list is recorded, the launch marker is NOT written
    partial="{\"model\":\"$K\",\"view\":\"$VIEW\",\"families\":\"$FAMILIES\",\"clusters\":\"$CLUSTERS\",\"wave_sha\":\"$WAVE_SHA\",\"failed_unit\":\"$u\",\"failed_logical_work_id\":\"$lwid\",\"placed_before_failure\":$(printf '%s\n' ${SUBMITTED_NOW[@]+"${SUBMITTED_NOW[@]}"} | "$PY" -c 'import json,sys; print(json.dumps([l for l in sys.stdin.read().splitlines() if l]))'),\"utc\":\"$(date -u +%FT%TZ)\"}"
    repro_write "$STATE/WAVE.$VIEW.partial.json" "$partial"
    repro_fatal "submission of $u ($lwid) FAILED — the wave is INCOMPLETE: ${#SUBMITTED_NOW[@]} unit(s) submitted in this run are listed in $STATE/WAVE.$VIEW.partial.json (every submission so far is in $STATE/submissions.jsonl); WAVE.$VIEW.json is NOT written. Fix the cause and re-run the same command: units already submitted are skipped, only the missing ones go out"
  fi
  SUBMITTED_NOW+=("$u")
done <<< "$PLAN_TSV"
case ",$FAMILIES," in *,multiturn,*)
  for l in ${CLUSTERS//,/ }; do
    # 31b: the group's placement is COMPLETE once every target is bound — attackers/PLACED lets a member nothing bound leave on its own;
    # a target still unbound (a --units subset, a refused submission) is a note: the next pass marks it
    g="mt-$VIEW-$l"
    if [ -n "${GROUP_TARGETS[$g]:-}" ]; then
      # the backstop is armed ONLY after the placement is complete — wait-pool refuses to tear down
      # without attackers/PLACED anyway, but an arm line printed on a partial group (a --units subset) invited stopping the pool early
      # a group GROWN by a declared move (its PLACED record names fewer targets) is re-placed as a versioned transition with the move's reason —
      # derived from the PERSISTED declaration whose destination is this label (flag-free passes included)
      MARK_EXTRA=(); mv_line="$(printf '%s\n' "$DECLARED_MOVES_TSV" | awk -F'\t' -v L="$l" '$2 == L { print; exit }')"
      if [ -n "$mv_line" ]; then
        mv_g="$(printf '%s' "$mv_line" | cut -f1)"; mv_from="$(printf '%s' "$mv_line" | cut -f3)"; mv_reason="$(printf '%s' "$mv_line" | cut -f4)"
        MARK_EXTRA=(--supersede --reason "declared move-group $mv_g=$l (from $mv_from): $mv_reason")
      elif [ "$RESUBMIT" = 1 ]; then
        # a --resubmit pass re-mints the group's dead member and targets, so the
        # roster's (unit, job) pairs and the member list CHANGE — and a placement is immutable. Without --supersede the funnel refused
        # ("attackers/PLACED exists ... with a different roster or members"), PLACED v1 kept assigning every target to the cancelled member, the
        # re-minted member found nothing assigned to it and self-exited (exit_unbound), and a target failed on its refused endpoint (the demo
        # smoke, 981273 -> 983283). The pass IS the provenance: --supersede --reason names the sha and the CHANGE SINCE THE RECORD — the
        # funnel's READ-ONLY placement-changes diffs the current bindings and pool members against attackers/PLACED on the cluster (the record
        # the placement transaction compares against) and answers the reason's change clause, each change accounted for by the ledger:
        # `re-minted <units>` (by the attempt that minted each), `dropped <unit> (job <id>: self-exited | dead on sacct <state> | torn down by its
        # STOP)` (a member the record names that LEFT with no replacement — one of two members self-exited beside the live one, the
        # walk skipped it and recorded its teardown, so the member list changed with no new job and a reason naming re-mints alone
        # derived nothing: no --supersede, the fence refused on every pass, this note said "runs again on the next pass" and the arm line never
        # printed), or both joined by `; ` — never this call's own appends (a line-count watermark named `none` on the retry
        # after a mark-placed that failed transiently, and superseded the placement with that as its provenance). No change, or a difference
        # the ledger cannot account for (a target dropped from the plan, a job no attempt records): NO --supersede — an unchanged placement
        # answers `unchanged`, a changed one is REFUSED by the immutable-placement fence and the sanctioned command below carries the decision
        # to the operator; the reason never says `none`. A declared move's reason (above) takes precedence over it.
        changes="$("$PY" "$HERE/lib/submit_twin.py" --config "$SISTEER_CONFIG" --model "$K" ${REPRO_OVERRIDES:+--overrides "$REPRO_OVERRIDES"} $(repro_dry_flag) placement-changes --group "$g" --targets "${GROUP_TARGETS[$g]}")" || changes=""
        if [ -n "$changes" ]; then MARK_EXTRA=(--supersede --reason "06 --resubmit pass on $WAVE_SHA: $changes"); fi
      fi
      if "$PY" "$HERE/lib/submit_twin.py" --config "$SISTEER_CONFIG" --model "$K" ${REPRO_OVERRIDES:+--overrides "$REPRO_OVERRIDES"} $(repro_dry_flag) mark-placed --group "$g" --targets "${GROUP_TARGETS[$g]}" ${MARK_EXTRA[@]+"${MARK_EXTRA[@]}"}; then
    echo "followup: arm the group-level backstop now: $PY $HERE/lib/submit_twin.py --config $SISTEER_CONFIG --model $K wait-pool --group mt-$VIEW-$l  (polls sacct every 60 s; teardown-pool the poll after the last target is terminal; per-member <jobid>.STOP, never attackers/STOP; a member also closes itself at its last consumer — 31b)"
      else
        mp_rc=$?   # the funnel's exit code: 4 is mark-placed's `no_live_member` (placement-fix round 6, below); every other refusal is handled as before
        hold_note=""; [ "$RESUBMIT" = 1 ] && hold_note="; the group's members hold every exit while this pass's attackers/PLACED.transition marker is fresh (its lease, 2 h by default), and an armed wait-pool holds its teardown and ratio under the same marker — complete the placement (the next pass, or the sanctioned command below) before it lapses, or release it deliberately with \`submit_twin.py placement-transition close --group $g --cluster $l\`"
        if [ "$mp_rc" != 4 ]; then
        repro_note "group $g is not fully placed yet (above) — mark-placed runs again on the next pass; do NOT arm wait-pool before it succeeds (it would not tear down without attackers/PLACED, but the members would idle)$hold_note"
        # placement-fix: a refusal by the immutable-placement fence has exactly ONE sanctioned answer — the versioned supersession with its
        # provenance, printed here verbatim so nobody reaches for PLACED, a marker or a ledger by hand (never sanctioned)
        if [ ${#MARK_EXTRA[@]} -gt 0 ]; then sanctioned_reason="${MARK_EXTRA[2]}"
        elif [ "$RESUBMIT" = 1 ]; then sanctioned_reason="06 --resubmit pass on $WAVE_SHA: <the change the ledger accounts for — re-minted <units> and/or dropped <member units> — placement-changes named none; the refusal above names the difference>"
        else sanctioned_reason="<why the roster changed: the re-minted units, or the ruling>"; fi
        echo "followup: if the refusal above is the immutable-placement fence (attackers/PLACED exists with a different roster or members), the sanctioned transition is exactly: $PY $HERE/lib/submit_twin.py --config $SISTEER_CONFIG --model $K${REPRO_OVERRIDES:+ --overrides $REPRO_OVERRIDES} mark-placed --group $g --targets ${GROUP_TARGETS[$g]} --supersede --reason $(printf '%q' "$sanctioned_reason")  (archives PLACED.v<n>, writes v<n+1> with provenance, appends placement_superseded; never hand-edit PLACED, a marker or a ledger)"
        else
        # mark-placed answered `no_live_member` (rc 4) — every
        # member the ledger records for the group has attackers/<job>.SELF_EXIT on the cluster, read under the group's lock, and THIS pass's --walk
        # minted no replacement for any of them: it read their jobs as active on Slurm (a job in its exit path is still RUNNING/COMPLETING there), so it kept them
        # as live and handed them to mark-placed, which refused to assign the re-minted targets to servers that are gone. Nothing was written (no
        # PLACED change, no archive, no event; the marker stands, bounded by its lease). The remedy is NOT retried here: the walk's re-mint needs
        # sacct to report the exited job terminal, which lags the SELF_EXIT by the job's exit and the accounting's write — a wait on a clock, not an
        # event this pass can read — so an in-pass retry would either race it and refuse again or poll on a timer; the next pass reads the terminal
        # state as a fact, records the teardown, re-mints the member, and its mark-placed commits the supersession. Printed verbatim, the same pass.
        repro_note "group $g: NO LIVE MEMBER (above) — every member of the group has attackers/<job>.SELF_EXIT on $l (read under the group's lock) and this pass minted no replacement for any of them (their jobs were still active on Slurm when its --walk read them); the re-minted targets were NOT placed onto a server that is gone: nothing written, attackers/PLACED is as it was; do NOT arm wait-pool on it$hold_note"
        # The printed pass is THIS pass's EFFECTIVE invocation, not a shorter spelling of it (a remedy that dropped
        # --nshards, --wave-sha, --cells and --units was told to be run verbatim; at a non-default shard count the plan, the inventory check or
        # the placement-manifest check refused it before the member could be re-minted, and without --units it submitted other units): --nshards
        # shapes the multi-turn plan (the shard groups name the target units and size the pool); --wave-sha pins the tree (the default is HEAD,
        # which may have moved by the re-run); --cells the served + registered cells; --force-gate the admissions a steered view re-checks on
        # every pass (gate/forced.jsonl is append-only and nothing reads it back); --overrides the resolved document (an environment default here
        # is a flag there). A --units selection is WIDENED by the group's member unit(s) it left out — the walk re-mints only the units it visits,
        # so the selection as given would be refused NO LIVE MEMBER again. Declared moves are not re-passed (the manifest persists them and every
        # pass reads them), --family is multiturn alone (the remedy is the placement's; the other families' units are untouched by it), and
        # every value is shell-quoted. The plan knobs the frozen inventory pins (SISTEER_SHARD_GROUPS, SISTEER_MT_ATTACKER_GPUS, SISTEER_MT_SEED_BASE)
        # travel in the environment, not on the command line, so the re-run is in the same environment.
        remedy=("$HERE/06_eval_waves.sh" --model "$K" --view "$VIEW" --family multiturn --clusters "$CLUSTERS" --nshards "$NSHARDS" --wave-sha "$WAVE_SHA" --resubmit)
        [ -n "$CELLS_SEL" ] && remedy+=(--cells "$CELLS_SEL")
        if [ -n "${UNITS:-}" ]; then
          remedy_units="$UNITS"
          for mu in ${GROUP_MEMBERS[$g]:+${GROUP_MEMBERS[$g]//,/ }}; do case ",$remedy_units," in *",$mu,"*) ;; *) remedy_units="$remedy_units,$mu";; esac; done
          remedy+=(--units "$remedy_units")
        fi
        for f in ${FORCE[@]+"${FORCE[@]}"}; do remedy+=(--force-gate "$f"); done
        remedy+=(--overrides "$SISTEER_RUN_OVERRIDES" --config "$SISTEER_CONFIG")
        remedy_cmd="$(shell_words "${remedy[@]}")"; remedy_cmd="${remedy_cmd% }"
        echo "followup: REMEDY for group $g — once sacct on $l reports the exited member job(s) named above terminal (a job in its exit path is RUNNING/COMPLETING for a moment after it writes SELF_EXIT), run exactly this pass again, in this environment (its effective flags; a --units selection widened by the group's member unit(s)): $remedy_cmd  (its --walk then reads the SELF_EXIT, records the teardown and re-mints the member; the re-minted targets are live and skipped; that pass's mark-placed commits the supersession onto the replacement and clears the marker; never hand-edit PLACED, a marker or a ledger)"
        UNPLACED+=("$g"$'\t'"$l"$'\t'"$remedy_cmd")   # the pass ends INCOMPLETE below, naming this group and this remedy
        fi
      fi
    else
      repro_note "group $g: no target of the group went out this call — nothing to place; arm wait-pool only after mark-placed has succeeded"
    fi
  done;;
esac
case ",$FAMILIES," in *,petri,*)
  if [ "$PASS" = overflow ]; then
    echo "followup: on the grading host once <wave root>/petri/$TAG/serving_rerun/endpoint.json lands, 08_grade.sh --daemon drives the overflow pass ONCE (legs=ledgered, F22_OVERFLOW_RERUN=1) and records it in <state>/grade/petri_drive.$TAG.overflow.json — after the first pass's record for its bound job reads incomplete; by hand it is: F22_OVERFLOW_RERUN=1 scripts/sdfbase/full22/petri_drive_full22.sh $K $VIEW <cluster label> $WAVE_SHA ledgered"
  else
  echo "followup: on the grading host once <wave root>/petri/$TAG/serving/endpoint.json lands, 08_grade.sh --daemon starts it ONCE and records the start in <state>/grade/petri_drive.$TAG.json (D121(B)); by hand it is: SISTEER_SECRETS_MODE=$SISTEER_SECRETS_MODE scripts/sdfbase/full22/petri_drive_full22.sh $K $VIEW <cluster label> $WAVE_SHA  (the drive imports the proxy job's registered identities from <shared>/registry/pending/<jobid>.json and files every run's store_uri + terminal state into this host's registry spool — petri_registry_handoff.py, D83; 06 --gate petri then reads the stores it names)"
  fi;;
esac
SHA_TABLE="$("${STAGE_PY[@]}" sha-table --ledger "$STATE/submissions.jsonl" 2>/dev/null || echo '{}')"   # the per-unit sha table
# the forced admissions THIS call carried (--force-gate LWID=REASON): printed and recorded in the wave
# record, so a forced steered submission never rests on gate/forced.jsonl alone (append-only; nothing reads it back — the provenance is
# the run_overrides stamp + this record). Empty on the baseline wave, where admission is skipped and the flag is moot (:275).
FORCED_JSON="[]"; FORCED_LINE="none"
if [ ${#FORCE[@]} -gt 0 ]; then
  if [ "$VIEW" = "$BASE_VIEW" ]; then
    repro_note "--force-gate is moot on the baseline wave $VIEW (admission is not applied there) — recorded as none"
  else
    FORCED_JSON="$("$PY" -c 'import json,sys; print(json.dumps([dict(zip(("logical_work_id","reason"), a.split("=",1))) for a in sys.argv[1:]]))' "${FORCE[@]}")"
    FORCED_LINE="$("$PY" -c 'import sys; print(", ".join("%s (%s)" % tuple(a.split("=",1)) for a in sys.argv[1:]))' "${FORCE[@]}")"
  fi
fi
# a pass whose mark-placed answered NO LIVE MEMBER for a group ends INCOMPLETE. It used to
# go on to write WAVE.<view>.json, print `06 done` and exit 0 — a success an unattended caller had no reason to doubt — while the targets it
# re-minted for that group had no member to bind: a target reads its assignment by its OWN job id (attacker_pool_lifecycle.sh
# f22_pool_assigned_member_url), so the re-minted jobs are unnamed in the standing record and fall to the round-robin over the live members,
# of which there are none; f22_live_attackers waits F22_ATTACKER_WAIT_S for one and the job ends fatal. The pass's product stands only once
# the printed REMEDY has run, so the pass says so the way a failed submission does: the groups, their labels and their remedies
# (verbatim, as printed) are recorded in WAVE.<view>.partial.json — the daemon's wave discovery and the sha-table refresh ignore `.partial.` —
# WAVE.<view>.json is NOT written (an earlier pass's record stands as it was: this pass changed nothing about the wave's launch, and every
# submission is in the ledger), the `06 done` line does not print, and the stage exits 4, mark-placed's own code for the refusal, distinct
# from a refusal by the plan (2). Every other group's mark-placed / arm line above stands; the other refusals (the immutable fence, rc 3)
# keep their note-and-continue handling, unchanged. A --units call is INCOMPLETE the same way (it writes no wave file either way). Not
# retried here, as the remedy's own note says: the re-mint needs sacct to report the exited job terminal, a wait on a clock.
if [ ${#UNPLACED[@]} -gt 0 ]; then
  partial="$(printf '%s\n' ${SUBMITTED_NOW[@]+"${SUBMITTED_NOW[@]}"} | "$PY" -c '
import json, sys
model, view, families, clusters, sha, forced, utc, *rows = sys.argv[1:]
groups = [dict(zip(("group", "cluster", "remedy"), r.split("\t", 2))) for r in rows]
print(json.dumps({"model": model, "view": view, "families": families, "clusters": clusters, "wave_sha": sha, "failed_unit": None,
                  "refusal": "no_live_member", "unplaced_groups": groups, "submitted": [l for l in sys.stdin.read().splitlines() if l],
                  "forced": json.loads(forced), "utc": utc}))' "$K" "$VIEW" "$FAMILIES" "$CLUSTERS" "$WAVE_SHA" "$FORCED_JSON" "$(date -u +%FT%TZ)" "${UNPLACED[@]}")"
  repro_write "$STATE/WAVE.$VIEW.partial.json" "$partial"
  unplaced_line=""; remedy_refs=""
  for e in "${UNPLACED[@]}"; do IFS=$'\t' read -r ug ul _ <<< "$e"; unplaced_line="$unplaced_line$ug (on $ul), "; remedy_refs="$remedy_refs\`followup: REMEDY for group $ug\`, "; done
  unplaced_line="${unplaced_line%, }"; remedy_refs="${remedy_refs%, }"
  REPRO_REFUSE_RC=4 repro_fatal "placement INCOMPLETE — group(s) $unplaced_line have NO LIVE MEMBER (above): the targets this pass re-minted for them have no member to bind (a target waits F22_ATTACKER_WAIT_S for a live member, then ends fatal), so this pass stands only once the exited member job(s) are terminal on sacct and the same pass has re-minted the member — run the printed REMEDY ($remedy_refs) then; recorded in $STATE/WAVE.$VIEW.partial.json (the groups and their remedies, verbatim); WAVE.$VIEW.json is NOT written (an earlier pass's record stands as it was); ${#SUBMITTED_NOW[@]} unit(s) of this pass went out or were skipped and every submission is in $STATE/submissions.jsonl; every other group's placement above stands"
fi
if [ -n "${UNITS:-}" ]; then
  repro_say "units=$UNITS selected: the wave file is NOT written (a selected call is a deliberate partial; re-run without --units to complete the wave — placed units are skipped)"
  [ "$FORCED_LINE" = none ] || repro_note "decisions.md: forced admissions for $K/$VIEW: $FORCED_LINE — a --units call writes no WAVE record, so record them in decisions.md (gate/forced.jsonl is append-only and nothing reads it back)"
else
repro_write "$STATE/WAVE.$VIEW.json" "{\"model\":\"$K\",\"view\":\"$VIEW\",\"families\":\"$FAMILIES\",\"clusters\":\"$CLUSTERS\",\"wave_sha\":\"$WAVE_SHA\",\"unit_shas\":$SHA_TABLE,\"overrides\":\"$OVERRIDE_ID\",\"forced\":$FORCED_JSON,\"n_units\":${#SUBMITTED_NOW[@]},\"utc\":\"$(date -u +%FT%TZ)\"}"
fi
repro_say "06 done: ${#SUBMITTED_NOW[@]} unit(s) placed; forced: $FORCED_LINE; every submission is in $STATE/submissions.jsonl; arm a Monitor per \`submitted\` line (D17); the daemon grades as runs complete"
