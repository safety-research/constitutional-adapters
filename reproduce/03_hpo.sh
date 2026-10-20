#!/bin/bash
# reproduce/03_hpo.sh — stage 03: the hyperparameter search of one model's steering objects.
#
#   reproduce/03_hpo.sh --model K --object-class relu|lora --phase probe|recon|design|trials|analyze|refine|table|pick|addendum|all
#                       [--rank R --set S] [--pairs R:S,…] [--probe-width N|auto] [--probe-time [D-]HH:MM:SS] [--parity-pass [--parity-scales default|S=x,…]] [--reopen [--reference-only [--parity-record-reopen] | --reverdict-only [--parity-record DIR]] [--certify-v2 all|R:S,…] [--probe-max-attempts N [--probe-cap-reason TEXT]]] [--resubmit] [--rehome] [--cluster <label>] [--fits-config configs/sdfbase_<key>_fits.yaml] [--config F] [--overrides F] [--dry-run] [--help]
#
#   probe    lora only: the serving probe (scripts/sbatch/serving_probe.sbatch <model_key> <served hf id> <revision>) — the
#            engine's servable (rank, module set) cells -> <hpo root>/allowed_pairs.json, the LoRA search's design space P.
#            Under the paper's protocol (searched clip, two warm-up levels: rank 4 / o_proj fixed, FD 18) the search consumes no
#            menu, so --phase all notes the probe and goes on; --phase probe runs the SAME state machine with the carrier's
#            positionals 4/5 = the ONE declared pair — the fits config's lora_r4_ce fit_selection (rank + module set, the set id
#            via lora_module_sets.MODULE_SETS, cross-checked against the row's lora_architecture), e.g. `… wo 4` — so the probe
#            certifies that pair alone: its verdict prices the compute plan's serving rows (tools/replan_compute.py --probe) and its
#            staged geometry record admits the served model's steered / padded standups. It registers nothing (kind other).
#            The record's state is READ through scripts/sdfbase/hpo_probe_status.py --record (never from file presence): READY
#            fetches the menu + verdict + geometry record and marks the phase done with the record's result generation; an
#            INCOMPLETE record with no job is RESUMED here — attempt+1 of the SAME record through lib/slurm.sh (ledger line +
#            watcher; the carrier's probe_attempt.json supplies the counter and the cap, the plan positionals 4-6);
#            --probe-width N|auto sizes the allocation (default: the record's last width on a resume, the sbatch header's on a fresh probe).
#            --reopen (with --phase probe) reopens a READY record deliberately: every staged geometry overlay is revoked, a revocation
#            receipt is written into the record and the resume carries its nonce — the only supported reopen.
#            --reopen on a REOPEN_IN_PROGRESS record (hpo_probe_status: an in-progress receipt beside a COMPLETE checkpoint — a reverdict
#            consumed the receipt and died before the certifying rewrite) routes the recovery — the carrier in phase=reverdict,
#            `reverdict --resume-interrupted` under THAT receipt on this cluster; never a fresh revocation. --reopen on a PARITY_STALE record
#            revokes, mints the receipt and routes `reverdict --parity-record <dir>` the same way (the parity record must be complete) —
#            unless the menu excludes pairs for SERVING reasons (launches to run): then it RESUMES the record like READY.
#            --certify-v2 all|R:S,… (with --reopen) carries the parity certificate v2 request to the resume: the named pairs'
#            current servers are planned at the ladder window and their v1-only launches re-run under the certificate's prompts.
#            --probe-time [D-]HH:MM:SS sets the carrier's --time for this invocation (config.env's SISTEER_TIME_PROBE otherwise).
#            --reopen on a VERDICT_REFUSED record whose verdict is uncertified for want of reference evidence and whose parity record is at
#            rest routes the receipted `reverdict --parity-record <dir>` (CPU-only, no attempt consumed).
#            --reopen --reference-only --parity-record-reopen re-makes the PARITY-PASS record's HF+PEFT judgements: that record (the
#            plan's parity_record_dir) must be READY — or VERDICT_REFUSED with an uncertified verdict, the fenced recovery of a FAILED
#            reference phase — with no job and no reopen in progress; the overlays are revoked, the receipt minted into
#            it, the carrier runs phase=reference on its own unit / name under its own attempt counter (--probe-max-attempts widens it).
#            --reopen --reverdict-only re-folds a READY (or PARITY_STALE) record's certification against its remembered parity record
#            through the receipted reverdict carrier — no resume, no launches, no attempt consumed.
#            --probe-max-attempts N [--probe-cap-reason TEXT] (with --reopen) is an operator cap widen for the resume: refused
#            below the next attempt, recorded in probe_attempt.json (the carrier) and in the reopen receipt's reason; --reopen on an
#            INCOMPLETE_STALLED / MENU_STALE record (launches outstanding, no job) resumes it the way the plain phase does.
#            --reopen --reference-only re-runs ONLY the HF+PEFT reference phase over the complete record (carrier positional 11
#            `reference`, one GPU, no engine: the reference-side fix's re-run) — the launches' servability stands as measured.
#            On a VERDICT_REFUSED record (a complete checkpoint whose verdict is uncertified) it is the re-run of a harness-filed reference
#            status: `reference_missing` filed by the identity guard is STICKY (the re-gate keeps a harness-filed status), so a fixed guard
#            is read only by judging the adapters again (probe 979954, the content-identity fix of 2026-10-10).
#            --parity-pass (with --phase probe, the paper protocol) runs the PARITY PASS of the one declared pair on ITS OWN record,
#            <hpo root>_parity (carrier phase=parity: `plan --parity-pass`, the per-set benign scales — wo 1.0 — the natural launches at
#            the ladder window, one adapter dir per pair, no throughput leg; one GPU by default). The main record's scale-100
#            adapters are the ladder's servability subjects, not assertion 4's: the default flow certifies adapter parity from the pass.
#            The verb is the same state machine over the parity record: NOT_STARTED submits, a running job waits, a stalled record
#            resumes (its own unit hpo.lora.probe.parity, job name <probe>-parity), READY names the re-verdict below, VERDICT_REFUSED
#            refuses by name (the pass's own parity_weak / parity_chaotic gates: `--parity-scales set=scale,…` re-plans it on a fresh
#            record after the refused one is moved aside by hand — the scale table is the pass's provenance, never rewritten in place).
#            --reopen --reverdict-only --parity-record DIR binds the main record to a parity-pass record for the first time (or to another
#            one): the receipted `reverdict --parity-record DIR` on a READY (or VERDICT_REFUSED / PARITY_STALE) main record; without the
#            flag the remembered pass is used (the verdict's stamp, else probe_plan.json parity_record_dir), else the stage's own
#            <hpo root>_parity when it is READY. The pass must be at rest. NEVER marks adapter parity verified by itself: the verdict
#            reads the pass's launches per adapter — a fixed guard without a pass leaves the pair as its own references judge it.
#            FAILED_START (the carrier wrote probe_attempt.json, the driver never wrote probe_progress.json, no job): with a plan the
#            record is resumed like INCOMPLETE; with no plan the phase refuses by name and, when re-run with --resubmit, PERFORMS the
#            recovery: every top-level file of the plan-less record (probe_attempt.json, the carrier's RESULT, leftovers) moves to
#            superseded/<utc>/ on the record (never deleted), the status is re-read, and the fresh submission goes to the funnel with
#            --resubmit — lib/slurm.sh re-mints the bound unit after ITS proof that the dead job registered nothing. NOT_STARTED with the
#            fresh unit already bound to a dead job (a carrier that died before any record file) refuses by name unless --resubmit.
#            The fetch exposes NOTHING before its record: the record is rsynced into <mirror>/.fetch.staging,
#            verified there, probe_fetch.json (the record's HOME — cluster + dir — and the verdict identity) is written first, then the
#            files move into place, the DONE marker last. Local probe evidence (menu, verdict, geometry record, DONE marker) whose fetch
#            record is absent, empty, malformed or names no home REFUSES every phase by name (nothing archived, nothing submitted) —
#            --cluster's path is the home only when nothing local exists. --rehome (with --phase probe) re-derives the home from the
#            mirror's own probe_verdict.json (provenance.probe_cluster / probe_record), verifies that cluster holds the record, and writes
#            the fetch record with the mirror's ACTUAL identity (and the marker when the mirror IS the record); the ordinary currency
#            check then decides — identical stays, rewritten is archived and re-fetched.
#            probe_status pipes the COMMITTED stdlib hpo_probe_status.py over the ssh's stdin (`python3 - --record …`) on purpose: the
#            record is verified in place on the cluster with no dependency on a mirrored sha for a read (keep it so).
#   recon    the gradient-norm reconnaissance (scripts/sbatch/hpo_trial.sbatch under HPO_JOB_MODE=recon -> `tune_arm.py recon`):
#            150-step trainings at an effectively unclipped 1e6 across a 5-point lr ladder spanning the box; the clip is the
#            OVERRIDE_HPO_CLIP_PERCENTILE-th percentile of the pooled pre-clip norms (<hpo root>/recon.json). LoRA: at the WIDEST
#            allowed pair (the largest object sets the most permissive clip, so clipping is rare for every pair).
#   design   `tune_arm.py emit-trials`: lora -> --design sobol16_coverage over allowed_pairs.json (16 scrambled-Sobol points in
#            u = log10(lr) x the allowed pairs, coverage-repaired; 32 only if 16 cannot cover); relu -> the 1-D LR-only design
#            (--clip-const from the recon, ONE warm-up level, 16 trials). Opens the registry fit_search (`registry: fit_search <id>`).
#   trials   the sbatch array (scripts/sbatch/hpo_trial.sbatch, 1 GPU per trial, --array=0-<N-1>, --comment=sha: pinned).
#   analyze  fetch the summaries and run `tune_arm.py analyze --design v2` (the additive model over u with rank / set effects and
#            the adequacy report; the plain quadratic in u for relu) -> r1_fit.json.
#   refine   the 3 + 1 split, ONE step per invocation from the cluster's state: emit + submit the three refine
#            trials (round 2) -> pair-only refit (`analyze --rounds 1,2 --pair-only`) -> the FOURTH trial: an exact confirmation
#            (round 9) when the refit is PD with an interior argmin, else an exploration replicate (round 3; never a confirmation).
#   table    scripts/sdfbase/hpo_table.py -> table.{csv,md,html} (one row per trial + the rank x set pivot over the allowed pairs +
#            the alarms verbatim), pushed to $SISTEER_FIG_DIR/hpo_<model>_<class>/ on the cluster (mirrored from there).
#   pick     lora only; REFUSES without --rank and --set (FD 6a: the operator picks from the table, no default in their absence):
#            scripts/sdfbase/hpo_pick.py records the phase-A ruling EITHER WAY, then — when the picked pair's learning rate has
#            measured clip-activation evidence below the 25 % limit — promotes it (models.yaml lora_architecture, hpo_pick.json,
#            hpo_fold.json = the --fold-hpo input); an OFF-WINNER pair with no evidence at its inferred lr is NOT promoted and
#            writes a confirm-pick request instead; a pair whose trial measured at/over the limit is refused by name (pick again).
#   confirm-pick  lora only; submits the ONE trial the request asks for (round 8 at the picked pair, tune_arm.py emit-trials
#            --confirm-pick-from, scripts/sbatch/hpo_trial.sbatch as every round) under the identity the request reserved
#            (r8_tNN: its own design file and job name; a replacement is refused while an earlier confirm trial is queued or
#            running); when its summary lands, re-run --phase pick.
#   addendum lora only, EXPLICIT (never part of --phase all): REQUIRES --pairs R:S,… — the pairs a
#            re-verdict certified AFTER the round-1 design was drawn (the local menu must be the record's CURRENT verdict, i.e. the widened
#            one, fetched by --phase probe). Refits the round-1 analyze over the widened menu (`analyze --design v2 --rounds 1 --allowed-pairs
#            <menu> --extends-menu` -> r1_fit_menu.json; the round-1 artefact r1_fit.json the pick was made from is restored from the record),
#            emits ONE trial per pair at that pair's predicted lr — else the set's best round-1 trial lr, stated — as round 4 / trial_role
#            addendum (`emit-trials --addendum-from`, provenance: addendum_of, the widened menu's sha + result generation, per-trial
#            lr_source) -> r4_trials.json, pushed to the record and submitted as ONE hpo_trial.sbatch array through the funnel.
#            `--phase table` then REBUILDS (its DONE marker is keyed on the trial set + menu, so new rows or a widened menu rebuild it) and
#            pivots over the widened menu with the addendum rows, predictions from the derived refit r1_fit_menu.json, the round-1
#            selection / provenance from r1_fit.json. The round-1 table is NOT held for it.
#            A FAILED array's relaunch (SISTEER_RESUBMIT=1) moves the record's design + trial dirs to superseded/<utc>/ under the funnel's
#            prove-resubmit, a per-record lock held to the close, and a durable transition receipt written before the move; every later
#            failure rolls back from the receipt and the next run reconciles an open one. An abandoned lock refuses by name: confirm its
#            owner is dead, `rm -r <record>/.resubmit.lock`, re-run.
#   all      every phase above in order (addendum excluded); a phase that must wait on a running job says so and the stage exits 0.
# Two protocols, chosen by the run-overrides file (FD 18: the paper's defaults unless a named file says otherwise):
#   paper  OVERRIDE_HPO_CLIP_MODE=searched + OVERRIDE_HPO_WARMUP_FRAC empty (+ OVERRIDE_LORA_ARCH_SEARCH=0): the paper's own
#          protocol — the legacy lr x clip Sobol over the two warm-up levels (scripts/tune_arm.py emit-trials --design legacy),
#          `analyze` (the 9-term quadratic), recentre / confirm through scripts/sdfbase/hpo_next_round.py; rank 4 / o_proj fixed,
#          so probe / recon / pick do not apply and say so.
#   fd6    OVERRIDE_HPO_CLIP_MODE=reconnaissance + OVERRIDE_HPO_WARMUP_FRAC set (+ OVERRIDE_LORA_ARCH_SEARCH=1 for lora): the
#          protocol above (a NAMED overrides file under reproduce/overrides/ that sets those keys). A mixed combination refuses by name.
# Idempotent by registry state (a fit_search of this campaign / model / object kind in the grading host's database) then by the
# DONE.<phase> markers under the grading host's state dir. Submissions go through lib/slurm.sh (absent: --dry-run prints the exact
# `sbatch --comment=sha:<sha>` line, live REFUSES). Every id produced prints `registry: <record_type> <id>`.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/lib/common.sh"
HAVE_SLURM_LIB=0
if [ -f "$HERE/lib/slurm.sh" ]; then source "$HERE/lib/slurm.sh"; HAVE_SLURM_LIB=1; fi
usage(){ sed -n '2,55p' "$0"; repro_common_help; }
CLASS=""; PHASE=all; PICK_RANK=""; PICK_SET=""; ADDENDUM_PAIRS=""; MENU_OVERRIDE=""; RECONCILE_DECISION=""; DECISION_WHY=""; CLUSTER=""; PROBE_WIDTH=""; PROBE_RESUBMIT=0; PROBE_REOPEN=0; PROBE_REHOME=0; PROBE_REF_ONLY=0
PROBE_CERTIFY_V2=""; PROBE_TIME=""; PROBE_MAX_ATTEMPTS=""; PROBE_CAP_REASON=""; PROBE_PARITY_REOPEN=0; PROBE_REVERDICT_ONLY=0; FITS_CONFIG=""
PROBE_PARITY_PASS=0; PROBE_PARITY_SCALES=""; PROBE_PARITY_RECORD=""
repro_parse_common_args "$@"
set -- "${REPRO_ARGS[@]+"${REPRO_ARGS[@]}"}"
while [ $# -gt 0 ]; do
  case "$1" in
    --object-class) CLASS="${2:?--object-class needs relu|lora}"; shift 2;; --object-class=*) CLASS="${1#--object-class=}"; shift;;
    --phase) PHASE="${2:?--phase needs a value}"; shift 2;; --phase=*) PHASE="${1#--phase=}"; shift;;
    --rank) PICK_RANK="${2:?--rank needs an integer}"; shift 2;; --rank=*) PICK_RANK="${1#--rank=}"; shift;;
    --set) PICK_SET="${2:?--set needs a module set id}"; shift 2;; --set=*) PICK_SET="${1#--set=}"; shift;;
    --pairs) ADDENDUM_PAIRS="${2:?--pairs needs rank:set,…}"; shift 2;; --pairs=*) ADDENDUM_PAIRS="${1#--pairs=}"; shift;;   # --phase addendum
    --menu-override) MENU_OVERRIDE="${2:?--menu-override needs the ruling id (its record is reproduce/menu_overrides/<id>.json)}"; shift 2;; --menu-override=*) MENU_OVERRIDE="${1#--menu-override=}"; shift;;
    --reconcile-transition) RECONCILE_DECISION="${2:?--reconcile-transition needs rollback}"; shift 2;; --reconcile-transition=*) RECONCILE_DECISION="${1#--reconcile-transition=}"; shift;;   # the operator's decision over an UNDECIDABLE relaunch transition
    --decision-why) DECISION_WHY="${2:?--decision-why needs the reason, recorded in the receipt}"; shift 2;; --decision-why=*) DECISION_WHY="${1#--decision-why=}"; shift;;   # the operator overrides the menu gate for EXACTLY --pairs
    --cluster) CLUSTER="${2:?--cluster needs a label}"; shift 2;; --cluster=*) CLUSTER="${1#--cluster=}"; shift;;
    --fits-config) FITS_CONFIG="${2:?--fits-config needs a path}"; shift 2;; --fits-config=*) FITS_CONFIG="${1#--fits-config=}"; shift;;   # the canonical config is the default (lib/common.sh repro_fits_config_for)
    --probe-width) PROBE_WIDTH="${2:?--probe-width needs N|auto}"; shift 2;; --probe-width=*) PROBE_WIDTH="${1#--probe-width=}"; shift;;
    --reopen) PROBE_REOPEN=1; shift;;     # --phase probe --reopen: the deliberate reopen of a READY record (revoke -> receipt -> resume)
    --reference-only) PROBE_REF_ONLY=1; shift;;   # with --reopen: re-run ONLY the HF+PEFT reference phase of the complete record on 1 GPU
    --parity-record-reopen) PROBE_PARITY_REOPEN=1; shift;;
    --reverdict-only) PROBE_REVERDICT_ONLY=1; shift;;   # with --reopen on a READY record — the receipted re-verdict against the remembered parity record, no resume, no attempt consumed   # with --reopen --reference-only — the record's PARITY-PASS record is the one reopened (its HF+PEFT judgements re-made)
    --parity-pass) PROBE_PARITY_PASS=1; shift;;     # --phase probe --parity-pass: the PARITY PASS of the one declared pair on its own record <hpo root>_parity (carrier phase=parity)
    --parity-scales) PROBE_PARITY_SCALES="${2:?--parity-scales needs default or set=scale,…}"; shift 2;; --parity-scales=*) PROBE_PARITY_SCALES="${1#--parity-scales=}"; shift;;   # with --parity-pass: the pass's per-set scales (SERVING_PROBE_PARITY_SCALES on the carrier)
    --parity-record) PROBE_PARITY_RECORD="${2:?--parity-record needs the parity-pass record dir (absolute, on the cluster)}"; shift 2;; --parity-record=*) PROBE_PARITY_RECORD="${1#--parity-record=}"; shift;;   # with --reopen --reverdict-only: the pass the re-verdict binds (first binding, or another pass)
    --certify-v2) PROBE_CERTIFY_V2="${2:?--certify-v2 needs all or rank:set,…}"; shift 2;; --certify-v2=*) PROBE_CERTIFY_V2="${1#--certify-v2=}"; shift;;   # The parity certificate v2 for named pairs, through the resume (SERVING_PROBE_CERTIFY_V2 on the carrier)
    --probe-time) PROBE_TIME="${2:?--probe-time needs [D-]HH:MM:SS}"; shift 2;; --probe-time=*) PROBE_TIME="${1#--probe-time=}"; shift;;   # the probe carrier's --time for THIS invocation (config.env's SISTEER_TIME_PROBE otherwise; the env never wins over the sourced config)
    --probe-max-attempts) PROBE_MAX_ATTEMPTS="${2:?--probe-max-attempts needs N}"; shift 2;; --probe-max-attempts=*) PROBE_MAX_ATTEMPTS="${1#--probe-max-attempts=}"; shift;;   # An OPERATOR cap widen for the resume — through the front door, never a by-hand sbatch
    --probe-cap-reason) PROBE_CAP_REASON="${2:?--probe-cap-reason needs text}"; shift 2;; --probe-cap-reason=*) PROBE_CAP_REASON="${1#--probe-cap-reason=}"; shift;;   # recorded with the widen (probe_attempt.json, the receipt)
    --resubmit) PROBE_RESUBMIT=1; shift;;
    --rehome) PROBE_REHOME=1; shift;;     # --phase probe --rehome: a mirror that lost its fetch record gets its home back from its own verdict
    *) echo "REFUSED: unknown argument $1" >&2; usage >&2; exit 2;;
  esac
done
[ "$REPRO_HELP" = 1 ] && { usage; exit 0; }
case "$CLASS" in relu|lora) ;; "") repro_fatal "--object-class relu|lora is required";; *) repro_fatal "--object-class must be relu or lora, got '$CLASS'";; esac
case "$PHASE" in probe|recon|design|trials|analyze|refine|table|pick|confirm-pick|addendum|all) ;; *) repro_fatal "--phase must be probe|recon|design|trials|analyze|refine|table|pick|confirm-pick|addendum|all";; esac
[ -z "$MENU_OVERRIDE" ] || { [ "$PHASE" = addendum ] || [ "$PHASE" = table ] || [ "$PHASE" = pick ]; } || repro_fatal "--menu-override applies to --phase addendum (with --pairs) --phase table and --phase pick only (got --phase $PHASE)"
[ -z "$MENU_OVERRIDE" ] || [ "$PHASE" != addendum ] || [ -n "$ADDENDUM_PAIRS" ] || repro_fatal "--menu-override needs --pairs rank:set,… — the override covers EXACTLY those pairs"
[ "$PHASE" != addendum ] || [ -n "$ADDENDUM_PAIRS" ] || repro_fatal "--phase addendum needs --pairs rank:set,… (the pairs the re-verdict certified after the round-1 design; D409)"
case "$PROBE_WIDTH" in ""|auto|[1-9]|[1-9][0-9]) ;; *) repro_fatal "--probe-width must be auto or a positive integer (GPUs on one node), got '$PROBE_WIDTH'";; esac
if [ -n "$PROBE_CERTIFY_V2" ]; then
  [ "$PHASE" = probe ] && [ "$PROBE_REOPEN" = 1 ] || repro_fatal "--certify-v2 re-runs launches of an EXISTING probe record under the certificate's prompts: pass it with --phase probe --reopen (the resume carries it to the driver as --certify-v2 with --resume --replan)"
  [ "$PROBE_REF_ONLY" != 1 ] || repro_fatal "--certify-v2 re-runs launches; --reference-only runs none — pass one of them"
  case "$PROBE_CERTIFY_V2" in all) ;; *) printf '%s' "$PROBE_CERTIFY_V2" | grep -Eq '^[0-9]+:[a-z_]+(,[0-9]+:[a-z_]+)*$' || repro_fatal "--certify-v2 must be all or rank:set,… (e.g. 16:attn_mlp,16:down), got '$PROBE_CERTIFY_V2'";; esac
fi
if [ -n "$PROBE_TIME" ]; then
  [ "$PHASE" = probe ] || repro_fatal "--probe-time sizes the serving probe's carrier alone: pass it with --phase probe"
  printf '%s' "$PROBE_TIME" | grep -Eq '^([0-9]+-)?[0-9]{1,2}:[0-9]{2}:[0-9]{2}$' || repro_fatal "--probe-time must be [D-]HH:MM:SS (Slurm's --time), got '$PROBE_TIME'"
fi
if [ -n "$PROBE_MAX_ATTEMPTS" ]; then
  [ "$PHASE" = probe ] && [ "$PROBE_REOPEN" = 1 ] || repro_fatal "--probe-max-attempts widens an EXISTING probe record's attempt cap for its resume: pass it with --phase probe --reopen (D549; the by-hand sbatch resume is not a path)"
  case "$PROBE_MAX_ATTEMPTS" in [1-9]|[1-9][0-9]) ;; *) repro_fatal "--probe-max-attempts must be a positive integer, got '$PROBE_MAX_ATTEMPTS'";; esac
  # a cap widen is an operator's decision about more GPU attempts — the reason is required up front (before anything is
  # read, archived or minted), and hpo_probe_resume.py refuses a cap that changes without one as the second belt
  [ -n "$(printf '%s' "$PROBE_CAP_REASON" | tr -d '[:space:]')" ] || repro_fatal "--probe-max-attempts $PROBE_MAX_ATTEMPTS needs --probe-cap-reason TEXT (who raises the cap and why is recorded on the record's probe_attempt.json cap_history and in the reopen receipt); nothing read, nothing submitted"
fi
[ -z "$PROBE_CAP_REASON" ] || [ -n "$PROBE_MAX_ATTEMPTS" ] || repro_fatal "--probe-cap-reason records the reason of a cap widen: pass it with --probe-max-attempts N"
if [ "$PROBE_REVERDICT_ONLY" = 1 ]; then
  [ "$PHASE" = probe ] && [ "$PROBE_REOPEN" = 1 ] || repro_fatal "--reverdict-only re-folds a READY probe record's certification against its remembered parity record: pass it with --phase probe --reopen"
  [ "$PROBE_REF_ONLY" = 0 ] && [ "$PROBE_PARITY_REOPEN" = 0 ] && [ -z "$PROBE_CERTIFY_V2" ] && [ -z "$PROBE_MAX_ATTEMPTS" ] \
    || repro_fatal "--reverdict-only runs no launches and no reference phase and consumes no attempt — it takes none of --reference-only / --parity-record-reopen / --certify-v2 / --probe-max-attempts"
fi
if [ -n "$PROBE_PARITY_RECORD" ]; then
  # the parity-pass record the re-verdict BINDS: an absolute dir on the cluster (the carrier reads it there: positional 12 `parity-record=/…`)
  [ "$PHASE" = probe ] && [ "$PROBE_REOPEN" = 1 ] && [ "$PROBE_REVERDICT_ONLY" = 1 ] \
    || repro_fatal "--parity-record names the parity-pass record a re-verdict reads assertion 4 from: pass it with --phase probe --reopen --reverdict-only (the first binding of a pass, or another pass; D163)"
  case "$PROBE_PARITY_RECORD" in /*) ;; *) repro_fatal "--parity-record must be the parity-pass record's ABSOLUTE dir on the cluster (the carrier reads it there), got '$PROBE_PARITY_RECORD'";; esac
fi
if [ "$PROBE_PARITY_PASS" = 1 ]; then
  [ "$PHASE" = probe ] || repro_fatal "--parity-pass runs the serving probe's PARITY PASS: pass it with --phase probe"
  [ "$PROBE_REOPEN" = 0 ] && [ "$PROBE_REF_ONLY" = 0 ] && [ "$PROBE_PARITY_REOPEN" = 0 ] && [ "$PROBE_REVERDICT_ONLY" = 0 ] && [ "$PROBE_REHOME" = 0 ] \
    && [ -z "$PROBE_CERTIFY_V2" ] && [ -z "$PROBE_MAX_ATTEMPTS" ] && [ -z "$PROBE_PARITY_RECORD" ] \
    || repro_fatal "--parity-pass is its own verb over the parity record <hpo root>_parity — it takes none of --reopen / --reference-only / --parity-record-reopen / --reverdict-only / --rehome / --certify-v2 / --probe-max-attempts / --parity-record (the main record's reopen verbs; the pass's re-verdict is --reopen --reverdict-only [--parity-record DIR])"
fi
if [ -n "$PROBE_PARITY_SCALES" ]; then
  [ "$PROBE_PARITY_PASS" = 1 ] || repro_fatal "--parity-scales sets the PARITY PASS's per-set scales: pass it with --phase probe --parity-pass"
  case "$PROBE_PARITY_SCALES" in
    default) ;;
    *) printf '%s' "$PROBE_PARITY_SCALES" | grep -Eq '^[a-z_]+=[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?(,[a-z_]+=[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)*$' \
         || repro_fatal "--parity-scales must be default or set=scale,… (e.g. wo=0.5,mlp=0.05; the driver's table is wo 1.0, qkvo 0.3, attn_mlp / mlp / down 0.05), got '$PROBE_PARITY_SCALES'";;
  esac
fi
if [ "$PROBE_PARITY_REOPEN" = 1 ]; then
  [ "$PHASE" = probe ] && [ "$PROBE_REOPEN" = 1 ] && [ "$PROBE_REF_ONLY" = 1 ] \
    || repro_fatal "--parity-record-reopen re-makes the PARITY-PASS record's HF+PEFT judgements: pass it with --phase probe --reopen --reference-only (D641; the parity record runs no launches here)"
  [ -z "$PROBE_CERTIFY_V2" ] || repro_fatal "--parity-record-reopen runs the reference phase alone; --certify-v2 re-runs launches — pass one of them"
fi
[ "$PROBE_REHOME" = 0 ] || [ "$PHASE" = probe ] || repro_fatal "--rehome needs --phase probe (the verb re-homes the mirror's fetch record and ends there; got --phase $PHASE)"
repro_load_config
repro_require_model
[ -n "$CLUSTER" ] || CLUSTER="$SISTEER_PRIMARY_CLUSTER"
cluster_require "$CLUSTER"
MODEL="$REPRO_MODEL"; STAGE=03_hpo; SHA="$(repro_head_sha)"
want(){ [ "$PHASE" = all ] || [ "$PHASE" = "$1" ]; }
case "$CLASS" in lora) ARM=lora_r4_ce; OBJECT_KIND=lora;; relu) ARM=psr_ce_all; OBJECT_KIND=relu_steer;; esac

# --- the overrides layer: the paper's defaults unless the run file says otherwise -------------------------------
WARMUP="${OVERRIDE_HPO_WARMUP_FRAC:-}"
CLIP_MODE="${OVERRIDE_HPO_CLIP_MODE:-searched}"
CLIP_PCT="${OVERRIDE_HPO_CLIP_PERCENTILE:-95}"
INIT_SCHEME="${OVERRIDE_INIT_SCHEME:-zero}"
ARCH_SEARCH="${OVERRIDE_LORA_ARCH_SEARCH:-0}"
N_TRIALS="${OVERRIDE_HPO_TRIALS:-16}"
N_RECENTRE="${OVERRIDE_HPO_RECENTRE_TRIALS:-4}"
TRIAL_EPOCHS="${SISTEER_HPO_TRIAL_EPOCHS:-3}"
HPO_SI="${SISTEER_HPO_POOL_CELL:-constv2-mix-10m}"
# registry campaign: the funnel's `two_si` default put this model's ReLU parent rows (and their epoch rows) under
# campaign two_si while the paper frame and this model's runs are sdfbase_full22 — object_id alone keys the table, so a two_si record upserts
# an adapter-filed row's campaign OUT of the paper frame. The fit_search row, the pick and the trial / fit emissions ride ONE value: the
# site's SISTEER_REGISTRY_CAMPAIGN, else REGISTRY_CAMPAIGN, else sdfbase_full22 — two_si only when a caller names it. job_env passes the
# resolved value into the job on BOTH branches (the Slurm branch carried none, so the job's emitter fell to its own two_si default).
CAMPAIGN="${SISTEER_REGISTRY_CAMPAIGN:-${REGISTRY_CAMPAIGN:-sdfbase_full22}}"
# the reconnaissance (paper-companion-repo stage E, 2026-09-30): the ladder runs as ONE ARRAY, one rung per task, one GPU
# each — five independent trainings never serialised. SISTEER_TIME_RECON is the walltime of ONE RUNG (one array task): 150
# optimizer steps at the FITS' folded effective batch (tune_arm bench cycles the pool when one pass is shorter than the budget —
# 4,096 rows / effective batch 64 = 64 steps per pass, so 150 steps ≈ 9,600 rows ≈ 23 M tokens — 5–8 h for a large model at
# ~1,000 tok/s; the paper models' rungs took minutes). No resume inside a rung (a TIMEOUT loses that rung only).
# 03:00:00 is the default; a site sizes it from its model's pace. SISTEER_HPO_RECON_CELL is the cell the rungs train on — DEFAULT
# constv2ctl2-full, the PAPER's reconnaissance cell (17,073 rows = 266 optimizer steps per pass at effective batch 64, so the
# recipe's 150 steps fit in ONE pass; the trial pool, 4,102 rows = 65 steps, cannot certify a rung). A cell too short for 150 steps in a
# pass REFUSES by name unless SISTEER_HPO_RECON_ALLOW_CYCLING=1 (recon_passes_for_cell below): repeating data lowers late-step norms
# and biases the p95 clip, so cycling is an opt-in recorded in the spec (cycled_passes), never silent. Held here, not in
# lib/common.sh's SISTEER_TIME_* block: these are the recon's alone.
RECON_TIME="${SISTEER_TIME_RECON:-03:00:00}"
RECON_SI="${SISTEER_HPO_RECON_CELL:-constv2ctl2-full}"
RECON_STEPS=150                                          # the recipe's budget (scripts/sdfbase/recon_clip.py RECON_STEPS; the reader accepts exactly this)
EFFECTIVE_BATCH="${OVERRIDE_EFFECTIVE_BATCH:-64}"        # the fits' effective batch every rung trains at (the memprobe fold realises it); CROSS-CHECKED against the resolved fits config below (CFG_EFFECTIVE_BATCH)
# the protocol (header): paper = searched clip + two-level warm-up + rank 4 / o_proj; fd6 = recon clip + one warm-up level
# (+ the rank x module-set search for lora). Anything in between is a misconfigured overrides file and refuses by name.
case "$CLIP_MODE:${WARMUP:+set}" in
  searched:) PROTOCOL=paper;;
  reconnaissance:set) PROTOCOL=fd6;;
  *) repro_fatal "overrides ${OVERRIDE_ID:-?}: OVERRIDE_HPO_CLIP_MODE=$CLIP_MODE with OVERRIDE_HPO_WARMUP_FRAC='${WARMUP}' — the paper's protocol is searched + empty (two levels), the FD 6 protocol reconnaissance + one level; mixed settings name no design";;
esac
if [ "$CLASS" = lora ]; then
  case "$PROTOCOL:$ARCH_SEARCH" in
    paper:0|fd6:1) ;;
    paper:1) repro_fatal "overrides ${OVERRIDE_ID:-?}: OVERRIDE_LORA_ARCH_SEARCH=1 needs the FD 6 protocol (reconnaissance clip + one warm-up level)";;
    fd6:0) repro_fatal "overrides ${OVERRIDE_ID:-?}: the FD 6 protocol with OVERRIDE_LORA_ARCH_SEARCH=0 is not a design this stage defines (the LoRA search IS the rank x module-set search)";;
    *) repro_fatal "OVERRIDE_LORA_ARCH_SEARCH must be 0 or 1, got '$ARCH_SEARCH'";;
  esac
fi
[ "$PROTOCOL" = paper ] || [ "$N_RECENTRE" = 4 ] || repro_fatal "OVERRIDE_HPO_RECENTRE_TRIALS=$N_RECENTRE: the FD 6 protocol is exactly 3 refine + 1 (plan §4 phase 6)"

# --- the model's facts (reproduce/models.yaml) ---------------------------------------------------------------------------
FIT_REPO="$(repro_model "$MODEL" fit.checkpoint.hf_repo)" || exit 2
FIT_REV="$(repro_model "$MODEL" fit.checkpoint.revision)" || exit 2
FIT_ID="$(repro_model "$MODEL" fit.checkpoint.model_id)" || exit 2
SRV_REPO="$(repro_model "$MODEL" serve.checkpoint.hf_repo)" || exit 2
SRV_REV="$(repro_model "$MODEL" serve.checkpoint.revision)" || exit 2
# the fits config that pins the fit checkpoint: lib/common.sh repro_fits_config_for (the ONE canonical configs/sdfbase_*_fits.yaml, seed
# replicates excluded, or the --fits-config named explicitly; the generator key may differ from the models table's model_id)
FIT_CFG="$(repro_fits_config_for "$FIT_REPO" "$FIT_REV" "$FITS_CONFIG" "$FIT_ID")" \
  || repro_fatal "no canonical fits config for $FIT_REPO@$FIT_REV (reason above; run scripts/sdfbase/make_sdfbase_configs.py, or name one with --fits-config <path>)"
FIT_KEY="$(basename "$FIT_CFG" _fits.yaml)"; FIT_KEY="${FIT_KEY#sdfbase_}"
# The fits config's top-level fit_width sizes the FITS (04); every unit here — the recon, the trials, the confirm-pick —
# trains in-process on ONE GPU (hpo_trial.sbatch gpu:1, tune_arm.py calls the trainer directly with no process group), so the
# width is stated and NOT applied: the search's per-trial cost is a one-GPU number whatever the fits' width
FIT_WIDTH_CFG="$(repro_fit_width_of_config "$REPRO_REPO/$FIT_CFG")"
[ "$FIT_WIDTH_CFG" -gt 1 ] 2>/dev/null && repro_note "fit_width $FIT_WIDTH_CFG in $(basename "$FIT_CFG") applies to 04's fits only — HPO trials, the recon and the confirm-pick stay width 1 (one GPU each)"
POOL_KEY="$(sed -n 's#^ *heldout_pool: /vol/pool/\([^/][^/]*\)/pairs_constv2-heldout\.jsonl *$#\1#p' "$REPRO_REPO/$FIT_CFG" | head -1)"
[ -n "$POOL_KEY" ] || repro_fatal "$FIT_CFG names no /vol/pool/<key>/ held-out pool"

# --- paths ---------------------------------------------------------------------------------------------------------------
SHARED_ROOT="$(cluster_var "$CLUSTER" CLUSTER_SHARED_ROOT)"
CL_HOST="$(cluster_var "$CLUSTER" CLUSTER_SSH)"
CHECKOUT="$(repro_expand_shared "$CLUSTER" "$(cluster_var "$CLUSTER" CLUSTER_EVAL_REPO_ROOT)")/$SHA"
CL_ROOT="$(repro_cluster_state_root "$CLUSTER" "$MODEL")/$STAGE/$CLASS"          # the hpo root on the cluster
LOCAL_ROOT="$(repro_state_dir "$MODEL")/$STAGE/$CLASS"                             # its grading host mirror (summaries, fits, table)
PAIRS="$SHARED_ROOT/vol/pool/$POOL_KEY/pairs_${HPO_SI}.jsonl"
RECON_PAIRS="$SHARED_ROOT/vol/pool/$POOL_KEY/pairs_${RECON_SI}.jsonl"   # the recon rungs' cell (SISTEER_HPO_RECON_CELL; default the trial pool)
CL_CONFIG="$CHECKOUT/$FIT_CFG"
FIG_DIR="$(repro_expand_shared "$CLUSTER" "$SISTEER_FIG_DIR")/hpo_${MODEL}_${CLASS}"
PY="$(repro_python)"
TUNE="$REPRO_REPO/scripts/tune_arm.py"
LOCAL_CFG="$REPRO_REPO/$FIT_CFG"
# the effective batch, cross-checked (demo-import M4): OVERRIDE_EFFECTIVE_BATCH is this stage's ARITHMETIC — recon_passes_for_cell
# and the recon spec's `effective_batch` — while every rung and trial TRAINS at the fits config's own effective batch (tune_arm
# config_effective_batch: batch_size_by_arm x accum_steps once the memprobe is folded, else the PENDING_MEMPROBE_EFF<n>_ACCUM
# sentinel's n). The two must agree or the recorded batch lies about what ran (the paper's qwen35_4b_base config realises 8; the demo
# lane and the seed replicates 64, the paper defaults' value). Refused by name on every invocation, before any phase, the way a
# mixed protocol is above; a disagreement is corrected in the overrides file (OVERRIDE_EFFECTIVE_BATCH), never here.
[ "$EFFECTIVE_BATCH" -gt 0 ] 2>/dev/null || repro_fatal "overrides ${OVERRIDE_ID:-?}: OVERRIDE_EFFECTIVE_BATCH='$EFFECTIVE_BATCH' is not a positive integer"
CFG_EFFECTIVE_BATCH="$("$PY" - "$LOCAL_CFG" <<'PY'
import re, sys, yaml
path = sys.argv[1]
cfg = yaml.safe_load(open(path, encoding="utf-8")) or {}
bs_by_arm = cfg.get("batch_size_by_arm") or {}
tks = cfg.get("train_kwargs") or {}
sentinel = re.compile(r"PENDING_MEMPROBE_EFF(\d+)_ACCUM")
is_int = lambda v: isinstance(v, int) and not isinstance(v, bool)
eff = {}
for arm in ("psr_ce_all", "lora_r4_ce"):
    bs = bs_by_arm.get(arm); acc = (tks.get(arm) or {}).get("accum_steps")
    if bs is None and acc is None:
        sys.exit(f"{path}: no batch_size_by_arm.{arm} / train_kwargs.{arm}.accum_steps — the config states no effective batch for {arm}")
    if is_int(bs) and is_int(acc):
        eff[arm] = bs * acc
        continue
    m = sentinel.fullmatch(str(acc)) if isinstance(acc, str) else None
    if not m:
        sys.exit(f"{path}: train_kwargs.{arm}.accum_steps={acc!r} with batch_size_by_arm.{arm}={bs!r} names no effective batch (neither two ints nor a PENDING_MEMPROBE_EFF<n>_ACCUM sentinel)")
    eff[arm] = int(m.group(1))
if len(set(eff.values())) != 1:
    sys.exit(f"{path}: the two CE arms realise different effective batches {eff} — one lane, one batch")
print(next(iter(eff.values())))
PY
)" || repro_fatal "fits config $FIT_CFG states no usable effective batch (its reason is above) — regenerate it (scripts/sdfbase/make_sdfbase_configs.py)"
[ "$CFG_EFFECTIVE_BATCH" = "$EFFECTIVE_BATCH" ] || repro_fatal "overrides ${OVERRIDE_ID:-?}: OVERRIDE_EFFECTIVE_BATCH=$EFFECTIVE_BATCH, but the fits config $FIT_CFG realises effective batch $CFG_EFFECTIVE_BATCH (batch_size_by_arm x accum_steps, or its PENDING_MEMPROBE_EFF<n>_ACCUM sentinel) — the recon rungs and the trials train at the config's batch (tune_arm config_effective_batch), so the override must say the same: set OVERRIDE_EFFECTIVE_BATCH=$CFG_EFFECTIVE_BATCH in the overrides file, or name a fits config at $EFFECTIVE_BATCH with --fits-config"
ALLOWED="$LOCAL_ROOT/allowed_pairs.json"; RECON="$LOCAL_ROOT/recon.json"
OVERRIDE_MENU=""; [ -z "$MENU_OVERRIDE" ] || OVERRIDE_MENU="$LOCAL_ROOT/allowed_pairs.override.$MENU_OVERRIDE.json"
MENU_OVERRIDES_DIR="${SISTEER_MENU_OVERRIDES_DIR:-$REPRO_REPO/reproduce/menu_overrides}"   # the committed authorization records (tests point it elsewhere)
MENU_OVERRIDE_RECORD=""
menu_override_authorize(){   # <pairs rank:set,…>: the override is bound to the EXACT approved set — the ruling must have a committed
  # record $MENU_OVERRIDES_DIR/<ruling>.json whose model / class are this invocation's, and every requested pair must be one of its pairs;
  # anything else is refused by name BEFORE any menu is built or anything submitted. Sets MENU_OVERRIDE_RECORD (the record's path).
  MENU_OVERRIDE_RECORD="$MENU_OVERRIDES_DIR/$MENU_OVERRIDE.json"
  [ -s "$MENU_OVERRIDE_RECORD" ] || repro_fatal "menu override $MENU_OVERRIDE: no authorization record at $MENU_OVERRIDE_RECORD — a ruling string without a committed record authorizes nothing"
  "$PY" - "$MENU_OVERRIDE_RECORD" "$MENU_OVERRIDE" "$MODEL" "$CLASS" "$1" <<'PYZ' || repro_fatal "menu override $MENU_OVERRIDE refused (its reason is above)"
import json, sys
rec = json.load(open(sys.argv[1]))
if str(rec.get("ruling")) != sys.argv[2]: sys.exit(f"the record names ruling {rec.get('ruling')!r}, not {sys.argv[2]!r}")
if str(rec.get("model")) != sys.argv[3] or str(rec.get("class")) != sys.argv[4]: sys.exit(f"the record is for {rec.get('model')}/{rec.get('class')}, this invocation is {sys.argv[3]}/{sys.argv[4]}")
approved = {(int(a), str(b)) for a, b in (rec.get("pairs") or [])}
asked = [(int(r), s.strip()) for r, s in (x.split(":", 1) for x in sys.argv[5].split(",") if x.strip())]
bad = [k for k in asked if k not in approved]
if bad: sys.exit(f"pair(s) {bad} are NOT in the approved set of {sys.argv[2]} {sorted(approved)} — refused; nothing built, nothing submitted")
if not asked: sys.exit("no pairs requested")
print(f"menu override {sys.argv[2]}: {len(asked)} requested pair(s) all within the approved set ({len(approved)}; record {sys.argv[1]})")
PYZ
}
# --- the FAILED-array relaunch's TRANSITION --------------------------------------------------------------------------
# A relaunch MOVES the record's canonical round-4 design and trial dirs to superseded/<utc>/ and only later re-derives, uploads and re-submits
# a replacement. Everything from the move to the submission runs under ONE per-record lock ($RESUB_LOCK: mkdir on the record host, HELD by the
# mover, released at the close or after the rollback) and behind a DURABLE TRANSITION RECEIPT ($RESUB_RECEIPT at the record's root — what moved
# where, when, by whom, under which bound job and pinned design sha, the replacement's sha once uploaded; state moving|moved|replaced|submitted)
# written on the record host BEFORE the first mv. EVERY failure after the move — a fetch or inventory refusal, a refit that leaves no recognized
# non-PD artefact, an emit or upload failure, the funnel's refusal, a crash — rolls the record back FROM THE RECEIPT (addendum_relaunch_exit, the
# phase's EXIT trap), and every live addendum phase RECONCILES an open receipt FIRST (addendum_reconcile_transition: rolled back when nothing
# was submitted, completed when the submission landed and only the close was lost) — a fresh design is never built over an open transition. The
# receipt closes into superseded/<utc>/transition.json (state complete | restored). An abandoned lock (its owner died holding it) fails CLOSED:
# the phase refuses by name with the owner line; the operator confirms that process is dead, removes <record>/$RESUB_LOCK, and re-runs — the open
# receipt then reconciles the record before anything else happens.
RESUB_LOCK=.resubmit.lock; RESUB_RECEIPT=r4_resubmit.transition.json
RESUB_OPEN4=0; RESUB_LOCKED4=0; RESUB_UTC4=""; UNATTRIBUTED_JOBS=""; RESUB_DECISION_JSON=""
resub_ssh(){ ssh -o BatchMode=yes -o ConnectTimeout=30 "$CL_HOST" "$@"; }
resub_who(){ echo "$(id -un 2>/dev/null || echo "${USER:-?}")@$(hostname -s 2>/dev/null || echo localhost) pid $$"; }
addendum_lock_acquire(){   # <utc>: the per-record relaunch lock, HELD until addendum_lock_release (atomic mkdir on the record host; names its owner)
  resub_ssh "set -u; cd '$CL_ROOT' || exit 3; if ! mkdir '$RESUB_LOCK' 2>/dev/null; then echo \"LOCKED: another relaunch holds $CL_HOST:$CL_ROOT/$RESUB_LOCK (owner: \$(cat '$RESUB_LOCK/owner' 2>/dev/null || echo unknown); age \$(( \$(date +%s) - \$(stat -c %Y '$RESUB_LOCK') )) s) — nothing moved; if that process is dead: rm -r '$CL_ROOT/$RESUB_LOCK' on $CL_HOST and re-run (an open transition receipt reconciles first)\"; exit 6; fi; printf '%s\\n' '$(resub_who) $1' > '$RESUB_LOCK/owner'" || return $?
  RESUB_LOCKED4=1
}
addendum_lock_release(){ [ "${RESUB_LOCKED4:-0}" = 1 ] || return 0; resub_ssh "rm -rf '$CL_ROOT/$RESUB_LOCK'" && RESUB_LOCKED4=0 || repro_warn "addendum: the relaunch lock $CL_HOST:$CL_ROOT/$RESUB_LOCK could not be released — remove it by hand once no relaunch is live"; }
transition_read(){   # -> the record's receipt, parsed: line 1 "state|utc|by|replacement_sha256|bound_job" ('|'-separated: an EMPTY field must survive the
  # split — IFS=tab collapses consecutive tabs and shifted bound_job into an empty replacement_sha256), then one moved path per line; non-zero when unreadable
  cluster_cat "$CL_ROOT/$RESUB_RECEIPT" | "$PY" -c '
import json, sys
r = json.load(sys.stdin)
if r.get("receipt") != "hpo-addendum-resubmit": sys.exit("not a relaunch transition receipt (receipt=%r)" % (r.get("receipt"),))
print("|".join(str(r.get(k) or "") for k in ("state", "utc", "by", "replacement_sha256", "bound_job")))
for p in r.get("moved") or []: print(p)
'
}
transition_write(){   # <state> [replacement sha256]: the receipt's state advances on the record (read-modify-write, atomic rename; the lock is held); rc only
  [ "$REPRO_DRY_RUN" = 1 ] && return 0
  cluster_cat "$CL_ROOT/$RESUB_RECEIPT" | "$PY" -c '
import datetime, json, sys
r = json.load(sys.stdin); r["state"] = sys.argv[1]; r["updated_utc"] = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
if len(sys.argv) > 2 and sys.argv[2]: r["replacement_sha256"] = sys.argv[2]
json.dump(r, sys.stdout, indent=1, sort_keys=True); print()
' "$1" "${2:-}" | resub_ssh "cd '$CL_ROOT' && cat > '$RESUB_RECEIPT.tmp' && mv -f '$RESUB_RECEIPT.tmp' '$RESUB_RECEIPT'"
}
transition_close(){   # <complete|restored> <utc> [<decision json>]: the receipt leaves the record's root for superseded/<utc>/transition.json in its final
  # state (+ the operator's decision block when one decided an undecidable transition: who, why, utc, the accepted job); rc only
  [ "$REPRO_DRY_RUN" = 1 ] && return 0
  cluster_cat "$CL_ROOT/$RESUB_RECEIPT" | "$PY" -c '
import datetime, json, sys
r = json.load(sys.stdin); r["state"] = sys.argv[1]; r["closed_utc"] = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
if len(sys.argv) > 2 and sys.argv[2]: r["decision"] = json.loads(sys.argv[2])
json.dump(r, sys.stdout, indent=1, sort_keys=True); print()
' "$1" "${3:-}" | resub_ssh "set -u; cd '$CL_ROOT' && mkdir -p 'superseded/$2' && cat > 'superseded/$2/transition.json.tmp' && mv -f 'superseded/$2/transition.json.tmp' 'superseded/$2/transition.json' && rm -f '$RESUB_RECEIPT'"
}
addendum_restore_superseded(){   # <utc>: roll the record back FROM ITS RECEIPT — every path the receipt lists returns to its canonical place from
  # superseded/<utc>/ (nothing deleted; an uploaded replacement design, recognized by the receipt's replacement sha, steps aside as
  # superseded/<utc>/replacement.r4_trials.json). Two passes on the record host: every path is classified first and a path at both places that is
  # not the recorded replacement, or at neither, refuses by name with NOTHING moved and the receipt left OPEN (the next run refuses until an
  # operator reconciles); only a clean classification moves. The mirror's copies follow. Closes the receipt as `restored`.
  local utc="$1" meta state rsha moved=()
  meta="$(transition_read)" || { repro_warn "addendum: no readable transition receipt at $CL_HOST:$CL_ROOT/$RESUB_RECEIPT — the record's state after the move at superseded/$utc/ is unknown; reconcile it by hand (nothing deleted)"; return 3; }
  IFS='|' read -r state _ _ rsha _ <<< "$(printf '%s\n' "$meta" | head -1)"
  mapfile -t moved < <(printf '%s\n' "$meta" | tail -n +2 | grep .)
  local margs=""; [ "${#moved[@]}" -gt 0 ] && margs="$(printf ' %q' "${moved[@]}")"
  if resub_ssh "bash -s -- '$CL_ROOT' '$utc' '$state' '$rsha'$margs" <<'REMOTE_RESTORE'
set -u
root="$1"; utc="$2"; state="$3"; rsha="$4"; shift 4
cd "$root" || exit 3
sup="superseded/$utc"; mkdir -p "$sup" || exit 3
plan=(); bad=0
for p in "$@"; do
  if [ -e "$p" ] && [ -e "$sup/$p" ]; then
    if [ "$p" = r4_trials.json ] && [ -n "$rsha" ] && [ "$(sha256sum "$p" | cut -d' ' -f1)" = "$rsha" ]; then plan+=("aside:$p")
    else echo "CONFLICT: $p is at its canonical path AND under $sup/ and is not the receipt's replacement (state $state) — nothing moved; reconcile by hand"; bad=1; fi
  elif [ -e "$sup/$p" ]; then plan+=("restore:$p")
  elif [ -e "$p" ]; then plan+=("inplace:$p")
  else echo "MISSING: $p is at neither its canonical path nor under $sup/ — nothing moved; reconcile by hand"; bad=1; fi
done
[ "$bad" = 0 ] || exit 9
for step in "${plan[@]+"${plan[@]}"}"; do
  p="${step#*:}"
  case "$step" in
    aside:*) mv -n "$p" "$sup/replacement.r4_trials.json" || exit 9; echo "the replacement r4_trials.json (sha ${rsha:0:12}) stepped aside as $sup/replacement.r4_trials.json"; mv -n "$sup/$p" . || exit 9; echo "restored $p";;
    restore:*) mv -n "$sup/$p" . || exit 9; echo "restored $p";;
    inplace:*) echo "in place $p";;
  esac
done
REMOTE_RESTORE
  then
    rm -f "$R4"; for d in "$LOCAL_ROOT"/superseded.r4.$utc.*/; do [ -d "$d" ] && mv -n "$d" "$LOCAL_ROOT/$(basename "$d" | sed "s/^superseded\.r4\.$utc\.//")" 2>/dev/null; done
    transition_close restored "$utc" "${RESUB_DECISION_JSON:-}" || { repro_warn "addendum: the record is restored but its receipt could not be closed — $CL_HOST:$CL_ROOT/$RESUB_RECEIPT stays open (state $state); reconcile by hand"; return 3; }
    repro_say "addendum: RESTORED the previous round-4 design and trial dirs on $CLUSTER from superseded/$utc/ (receipt superseded/$utc/transition.json, state restored; a replacement, if uploaded, is superseded/$utc/replacement.r4_trials.json)"
  else
    repro_warn "addendum: the rollback from superseded/$utc/ on $CL_HOST:$CL_ROOT did not complete (its reason is above) — the transition receipt stays OPEN; reconcile the record by hand (nothing deleted)"
    return 3
  fi
}
relaunch_submission_evidence(){   # <receipt utc> <the proof's dead job>: was ANYTHING submitted under this transition? 0 = yes (prints the evidence) |
  # 1 = provably nothing (prints what was read) | 2 = undecidable (prints why; UNATTRIBUTED_JOBS names the jobs an operator may decide over).
  # A crash between the funnel's accepted sbatch and the receipt's `submitted` leaves the receipt at `replaced` while the array
  # is bound in the funnel's ledger; once that array left the queue, an EMPTY QUEUE proves nothing — the durable evidence decides: (1) the
  # funnel's binding for the unit — a job other than the proof's dead one is a binding made after the move; (2) the funnel's own re-run
  # transition token (written BEFORE its sbatch, cleared only after the attempt is bound) is undecidable here, the funnel reconciles it first.
  # (1)+(2) are the complete evidence: with no token and an unchanged binding no job of THIS transition can exist. The queue and sacct are
  # GUARDS, never attributions — sacct does not retain --comment on these clusters and the job name is load-bearing, so
  # no per-transition identity rides a job: (3) the live queue is read BY JOB ID (fails closed): a live id the ledger knows for this unit is this
  # unit's submission; a live same-name job the ledger does NOT know (another record's or session's) is UNDECIDABLE — never "submitted"; (4) in
  # sacct every job id the ledger knows is excluded; an unknown same-name job with Submit >= receipt utc - RESUB_CLOCK_SKEW_S (both clocks UTC)
  # is UNDECIDABLE; an older same-name job is pre-existing and ignored. An unreadable queue / binding / sacct is undecidable, never "nothing".
  local utc="$1" bound="$2" jname; jname="$(name r4)"
  local known="$bound"
  if [ "$HAVE_SLURM_LIB" = 1 ]; then
    local bl bst bcl bjob btok batt
    bl="$(slurm_binding_state --lwid "$MODEL/$STAGE/$CLASS/$(unit r4)" 2>&1)" || { echo "the funnel's binding for $(unit r4) could not be read: $(printf %s "$bl" | tr '\n' ' ' | cut -c1-200)"; return 2; }
    IFS=$'\t' read -r _ bst bcl bjob btok <<< "$(printf '%s\n' "$bl" | awk -F'\t' '$1 == "binding"' | tail -1)"
    batt="$(printf '%s\n' "$bl" | awk -F'\t' '$1 == "attempts" {print $2}' | tail -1)"
    if [ "${btok:-}" = yes ]; then
      # the lost-launch case (sbatch accepted, the funnel died before its ledger append) is the FUNNEL's to reconcile —
      # its token reconciliation identifies the launched job positively by its nonce (squeue %k) and ADOPTS it (ledger, submissions, the
      # `submitted` binding, the watcher); only that new binding completes the receipt. None -> cleared (nothing launched; the guards below
      # decide); ambiguity -> the funnel refuses by name and the token stands: undecidable here, the funnel's message names the ledger repair.
      local rt rline rout
      rt="$(slurm_reconcile_token --lwid "$MODEL/$STAGE/$CLASS/$(unit r4)" 2>&1)"; rout=$?
      rline="$(printf '%s\n' "$rt" | awk -F'\t' '$1 == "token"' | tail -1)"
      case "${rline#token	}" in
        adopted*)
          local how; how="$(printf '%s\n' "$rline" | awk -F'\t' '{print $3}')"
          # an adoption counts here ONLY when the funnel identified the job by its nonce read back (squeue %k) — anything else (an older
          # funnel's elimination, an unknown method) is not this transition's proof; the receipt stays open for a manual verified binding
          # `ledger` = the launching process had recorded the attempt under this reservation and died before the binding; the funnel
          # verified the launch record's nonce (the job no longer live) and finished the binding — this reservation's launch too
          case "$how" in nonce|ledger) ;; *) echo "the funnel reports the token's job adopted but identified by '${how:-unknown}', not by the nonce read back from the job (nor by its launch record under this reservation) — not accepted as this transition's launch; the receipt stays open for a manual verified binding; $(printf '%s\n' "$rt" | tr '\n' ' ' | cut -c1-300)"; return 2;; esac
          IFS=$'\t' read -r _ bst bcl bjob btok <<< "$(printf '%s\n' "$rt" | awk -F'\t' '$1 == "binding"' | tail -1)"
          [ -n "${bjob:-}" ] && [ "$bjob" != none ] && [ "$bjob" != "$bound" ] \
            || { echo "the funnel reported the token's job adopted, yet its binding for $(unit r4) still names ${bjob:-nothing} (the proof's dead job is $bound) — inconsistent; $(printf '%s\n' "$rt" | tr '\n' ' ' | cut -c1-300)"; return 2; }
          echo "the funnel's transition token was reconciled: the launched job $bjob on ${bcl:-?} is ADOPTED and bound (${bst:-?}; identified by its nonce read back from the job, $(printf '%s\n' "$rline" | awk -F'\t' '{print $4}')) — a submission under this transition"; return 0;;
        cleared*)
          echo "the funnel's transition token was reconciled: no launch happened under it (cleared) — the guards below decide"
          IFS=$'\t' read -r _ bst bcl bjob btok <<< "$(printf '%s\n' "$rt" | awk -F'\t' '$1 == "binding"' | tail -1)"
          batt="$(printf '%s\n' "$rt" | awk -F'\t' '$1 == "attempts" {print $2}' | tail -1)";;
        *)
          echo "the funnel holds a re-run transition token for $(unit r4) and could not reconcile it (rc $rout): $(printf '%s\n' "$rt" | grep -v '^binding\|^attempts' | tr '\n' ' ' | cut -c1-400) — the funnel's ledger repair comes first (its message names it); nothing decided here"; return 2;;
      esac
    fi
    if [ -n "${bjob:-}" ] && [ "$bjob" != none ] && [ "$bjob" != "$bound" ]; then echo "the funnel's ledger binds $(unit r4) to job $bjob on ${bcl:-?} (${bst:-?}), not the proof's dead job $bound — a submission under this transition"; return 0; fi
    [ -z "${batt:-}" ] || known="$known,$batt"
  fi
  # (3) the live queue BY ID
  local q qrc live_known live_unknown
  q="$(ssh -o BatchMode=yes -o ConnectTimeout=30 "$CL_HOST" "squeue -h -u \"\${USER:-\$(id -un)}\" -n '$jname' -o %i 2>&1")"; qrc=$?
  [ "$qrc" = 0 ] || { echo "squeue on $CL_HOST did not answer cleanly (rc $qrc: $(printf %s "$q" | tr '\n' ' ' | cut -c1-160)) — the queue is not provably empty"; return 2; }
  live_known="$(printf '%s\n' "$q" | grep . | awk -F'|' -v known=",$known," '{sub(/_.*/, "", $1)} index(known, "," $1 ",") > 0 {print $1}')"
  live_unknown="$(printf '%s\n' "$q" | grep . | awk -F'|' -v known=",$known," '{sub(/_.*/, "", $1)} index(known, "," $1 ",") == 0 {print $1}')"
  [ -z "$live_known" ] || { echo "a job named $jname the funnel's ledger knows for $(unit r4) is queued/running ($(printf '%s\n' "$live_known" | tr '\n' ' ')) — this unit's submission"; return 0; }
  if [ -n "$live_unknown" ]; then
    echo "UNATTRIBUTED=$(printf '%s\n' "$live_unknown" | tr '\n' ',' | sed 's/,$//')"
    echo "a job named $jname is queued/running that the funnel's ledger does NOT know for $(unit r4) ($(printf '%s\n' "$live_unknown" | tr '\n' ' ')) — not attributable to this transition (another record's or session's same-name job on $CLUSTER)"; return 2
  fi
  # (4) sacct since the receipt
  local since_s since rows rc unknown
  since_s="$(date -u -d "${utc:0:4}-${utc:4:2}-${utc:6:2} ${utc:9:2}:${utc:11:2}:${utc:13:2} UTC" +%s 2>/dev/null)" || { echo "the receipt's utc '$utc' is unreadable"; return 2; }
  since_s=$(( since_s - ${RESUB_CLOCK_SKEW_S:-300} )); since="$(date -u -d "@$since_s" +%Y-%m-%dT%H:%M:%S)"
  rows="$(ssh -o BatchMode=yes -o ConnectTimeout=30 "$CL_HOST" "sacct -n -X -P --name='$jname' -S '$since' -o JobID,Submit,State 2>&1")"; rc=$?
  [ "$rc" = 0 ] || { echo "sacct on $CL_HOST did not answer cleanly (rc $rc: $(printf %s "$rows" | tr '\n' ' ' | cut -c1-160)) — the accounting is not provably empty"; return 2; }
  unknown="$(printf '%s\n' "$rows" | grep . | awk -F'|' -v known=",$known," -v since="$since" 'index(known, "," $1 ",") == 0 && $2 >= since')"
  if [ -n "$unknown" ]; then
    echo "UNATTRIBUTED=$(printf '%s\n' "$unknown" | cut -d'|' -f1 | tr '\n' ',' | sed 's/,$//')"
    echo "sacct names a job of $jname submitted since $since that the funnel's ledger does not know for $(unit r4): $(printf '%s\n' "$unknown" | head -3 | tr '\n' ' ')— not attributable to this transition (another record's same-name job on $CLUSTER, or a launch the funnel lost)"; return 2
  fi
  echo "no job named $jname queued/running or bound after $bound in the funnel's ledger (known attempts: $known), no unknown job of that name in sacct since $since — nothing was submitted under this transition"; return 1
}
addendum_reconcile_transition(){   # at the start of every LIVE addendum phase: an OPEN receipt on the record is reconciled before anything is read, built or submitted
  [ "$REPRO_DRY_RUN" = 1 ] && return 0
  cluster_has "$CL_ROOT/$RESUB_RECEIPT" || return 0
  local meta state utc by rsha bound
  meta="$(transition_read)" || repro_fatal "addendum: the record carries an unreadable relaunch transition receipt $CL_HOST:$CL_ROOT/$RESUB_RECEIPT (its reason is above) — reconcile it by hand; nothing read, built or submitted"
  IFS='|' read -r state utc by rsha bound <<< "$(printf '%s\n' "$meta" | head -1)"
  repro_warn "addendum: an OPEN relaunch transition on the record $CL_HOST:$CL_ROOT (state $state since $utc by $by; bound job ${bound:-?}) — the previous relaunch did not close; reconciling it FIRST, nothing is built over it"
  addendum_lock_acquire "$utc" || repro_fatal "addendum: the record's relaunch lock is held while a transition is open (above) — nothing reconciled, nothing built"
  case "$state" in
    moving|moved|replaced|submitted)
      # the tie-breaker is the DURABLE submission evidence (relaunch_submission_evidence: the queue by name, the funnel's binding,
      # its transition token, sacct by name since the receipt) — a submission under this transition completes it (never roll a submitted design
      # out from under its array, running or already terminal); provably nothing submitted rolls back; undecidable evidence refuses by name
      local ev erc
      if [ "$state" = submitted ]; then ev="the receipt records the submission"; erc=0
      else ev="$(relaunch_submission_evidence "$utc" "$bound")"; erc=$?; fi
      UNATTRIBUTED_JOBS="$(printf '%s\n' "$ev" | sed -n 's/^UNATTRIBUTED=//p' | tail -1)"; ev="$(printf '%s\n' "$ev" | grep -v '^UNATTRIBUTED=')"   # the evidence runs in a subshell: its unattributed ids travel on this line
      case "$erc" in
        0) [ -z "$RECONCILE_DECISION" ] || repro_note "addendum: --reconcile-transition $RECONCILE_DECISION not needed — the evidence decides ($ev)"
           transition_close complete "$utc" && repro_say "addendum: the transition $utc had been SUBMITTED ($ev; only the receipt's advance/close was lost) — completed: receipt at superseded/$utc/transition.json; the replacement design stands" \
             || repro_fatal "addendum: the submitted transition $utc could not be closed (above)";;
        1) [ -z "$RECONCILE_DECISION" ] || repro_note "addendum: --reconcile-transition $RECONCILE_DECISION not needed — the evidence decides"
           repro_say "addendum: $ev — rolling back"; addendum_restore_superseded "$utc" || repro_fatal "addendum: the open transition $utc could not be rolled back (above) — the receipt stays open; reconcile by hand, then re-run";;
        *) addendum_reconcile_undecidable "$utc" "$ev";;
      esac;;
    *) repro_fatal "addendum: the transition receipt names an unknown state '$state' — reconcile by hand";;
  esac
  addendum_lock_release
}
addendum_reconcile_undecidable(){   # <utc> <why>: an UNDECIDABLE open transition — without the operator's decision it
  # refuses by name and names the recovery. The ONLY operator decision is `--reconcile-transition rollback --decision-why '<why>'` (the
  # unattributed same-name job is NOT this transition's: the previous design and trial dirs return; the decision — who, why, utc, the evidence it
  # overrode — is recorded in the closed receipt). A LOST LAUNCH is never closed on an operator's word: the funnel's transition token identifies
  # and ADOPTS the launched job itself (slurm_reconcile_token, nonce first) and its new binding completes the receipt on the next run; when the
  # funnel refuses (ambiguous candidates) its message names the ledger repair (`stage.py bind` the right job, clear the token) — then re-run.
  local utc="$1" why="$2" dec="$RECONCILE_DECISION" who; who="$(resub_who)"
  local recovery="recovery: inspect $CL_HOST:$CL_ROOT/$RESUB_RECEIPT and $CL_ROOT/superseded/$utc/ (the archived design + trial dirs; the canonical r4_trials.json is the uploaded replacement when state is replaced) and the unattributed job(s) ${UNATTRIBUTED_JOBS:-?} in sacct / the queue. If the funnel holds a transition token, resolve THAT first (the funnel adopts its own launch by nonce on the next run; if it refused, repair its ledger as its message says) — never close this receipt over it. If the job is not this transition's, re-run THIS command with --reconcile-transition rollback --decision-why '<why it is not>' (the previous design and trial dirs return; the decision is recorded in the closed receipt)"
  [ -n "$dec" ] || repro_fatal "addendum: the open transition $utc cannot be decided — $why; nothing moved, the receipt stays open. $recovery"
  [ -n "$DECISION_WHY" ] || repro_fatal "addendum: --reconcile-transition $dec needs --decision-why '<text>' (recorded in the receipt with who decided) — nothing moved"
  case "$dec" in
    rollback)
      case "$why" in *"transition token"*) repro_fatal "addendum: --reconcile-transition rollback refused while the funnel holds a transition token for $(unit r4) — a launch may have happened and its array reads the canonical design; resolve the funnel's token first ($why); nothing moved";; esac
      RESUB_DECISION_JSON="$("$PY" -c 'import json,sys; print(json.dumps({"kind": "rollback", "by": sys.argv[1], "why": sys.argv[2], "evidence": sys.argv[3]}))' "$who" "$DECISION_WHY" "$why")"
      repro_say "addendum: the transition $utc is ROLLED BACK by the operator's decision ($who: $DECISION_WHY) — $why"
      addendum_restore_superseded "$utc" || repro_fatal "addendum: the open transition $utc could not be rolled back (above) — the receipt stays open; reconcile by hand, then re-run";;
    accept-binding:*)
      repro_fatal "addendum: --reconcile-transition $dec is not a decision this record can take (the review on 50841cf14, D753): closing the receipt would not bind the job in the funnel — a lost launch is adopted by the funnel's own transition-token reconciliation (nonce first) and its binding completes the receipt on the next run; if the funnel refused, repair its ledger as its message says; nothing moved. $recovery";;
    *) repro_fatal "addendum: --reconcile-transition '$dec' is not `rollback` — nothing moved";;
  esac
}
relaunch_rollback_if_safe(){   # <utc> <context>: roll the record back from its receipt ONLY when the funnel provably launched nothing under this
  # transition — its binding readable and still the proof's dead job, no transition token. A token or a new binding means
  # a launch may have happened and its array reads the canonical design: the record stays, the receipt stays OPEN, and the next --phase addendum
  # reconciles it through the funnel (the token's job adopted by its nonce and bound, then the receipt completes). An unreadable funnel is treated
  # the same way — never a rollback on absence of evidence. 0 = rolled back | 1 = left open (said why) | 2 = the rollback itself did not complete
  local utc="$1" ctx="$2" bl bjob btok verdict="rollback"
  if [ "$HAVE_SLURM_LIB" = 1 ] && [ "$REPRO_DRY_RUN" != 1 ]; then
    if bl="$(slurm_binding_state --lwid "$MODEL/$STAGE/$CLASS/$(unit r4)" 2>&1)"; then
      IFS=$'\t' read -r _ _ _ bjob btok <<< "$(printf '%s\n' "$bl" | awk -F'\t' '$1 == "binding"' | tail -1)"
      [ "${btok:-}" != yes ] || verdict="token"
      [ -z "${bjob:-}" ] || [ "$bjob" = none ] || [ "$bjob" = "${BOUND4:-}" ] || verdict="bound:$bjob"
    else verdict="unreadable"; fi
  fi
  if [ "$verdict" = rollback ]; then
    repro_warn "addendum: $ctx and the funnel launched nothing under it (binding unchanged, no token) — rolling the record back from the transition receipt (superseded/$utc/)"
    addendum_restore_superseded "$utc" && return 0
    repro_warn "addendum: the rollback did not complete — the receipt stays OPEN on the record; the next --phase addendum reconciles it, or an operator does"; return 2
  fi
  repro_warn "addendum: $ctx but the funnel may have LAUNCHED under it ($verdict) — NOT rolling back (the array reads the canonical design); the receipt stays OPEN at $CL_HOST:$CL_ROOT/$RESUB_RECEIPT and the next --phase addendum reconciles it through the funnel (the token's job is adopted by its nonce and bound, then the receipt completes)"
  return 1
}
addendum_relaunch_exit(){   # the addendum phase's EXIT trap: an open relaunch (moved, not closed) rolls back from the receipt; a held lock is released
  local rc=$?
  if [ "${RESUB_OPEN4:-0}" = 1 ]; then
    RESUB_OPEN4=0; relaunch_rollback_if_safe "$RESUB_UTC4" "--resubmit did not complete (exit $rc)" || true
  fi
  addendum_lock_release
}
override_recon_menu(){   # The recon revalidation recomputes the ladder's widest allowed pair from THE MENU THE RECONNAISSANCE RAN ON — the probe's
  # menu file may be archived by a reopen, so under the override that menu is the archived original whose sha256 is the one the round-1 design
  # recorded (allowed_pairs_source.sha256 — a positive identity, newest copy first), else a menu built from the design's own pair details; prints
  # "<kind>\t<path>" (kind: archived | design-detail). Only the recon check reads it; the probe gate itself stays overridden.
  "$PY" - "$LOCAL_ROOT/r1_trials.json" "$LOCAL_ROOT" "$LOCAL_ROOT/allowed_pairs.design.json" <<'PYR'
import glob, hashlib, json, os, sys
d = json.load(open(sys.argv[1])); src = d.get("allowed_pairs_source") or {}; want = src.get("sha256")
for p in sorted(glob.glob(os.path.join(sys.argv[2], "superseded", "*", "allowed_pairs.json")), reverse=True):
    if want and hashlib.sha256(open(p, "rb").read()).hexdigest() == want:
        print("archived\t" + p); sys.exit(0)
detail = d.get("allowed_pairs_detail") or []
if not detail: sys.exit(f"{sys.argv[1]} carries no allowed_pairs_detail and no archived menu matches sha {str(want)[:16]} — the recon menu cannot be reconstructed")
# The review on a52d126b2 (MEDIUM): the reconstruction must BE the design's menu — exact normalized set equality with the design's recorded allowed_pairs,
# no duplicates, valid details — or the recon check would see an artificial widest pair and archive a valid reconnaissance (the D225 shape)
def key(p):
    try: return (int(p["rank"]), str(p["set"]))
    except Exception: return None
keys = [key(p) for p in detail]
bad = [i for i, (p, k) in enumerate(zip(detail, keys)) if k is None or not isinstance(p.get("lora_modules"), list) or not p.get("lora_modules")
       or not isinstance(p.get("trainable_params"), int) or isinstance(p.get("trainable_params"), bool) or p["trainable_params"] <= 0]
if bad: sys.exit(f"allowed_pairs_detail entries {bad} are not valid pair entries (rank/set/lora_modules/trainable_params) — the recon menu cannot be reconstructed; nothing moved")
if len(set(keys)) != len(keys): sys.exit(f"allowed_pairs_detail carries duplicate pairs {sorted(k for k in set(keys) if keys.count(k) > 1)} — nothing moved")
design = sorted((int(a), str(b)) for a, b in (d.get("allowed_pairs") or []))
if sorted(keys) != design: sys.exit(f"allowed_pairs_detail {sorted(keys)} is not the design's recorded menu {design} (incomplete or extra entries) — the recon menu cannot be reconstructed; nothing moved")
doc = {"model_id": src.get("model_id"), "probe_job_id": src.get("probe_job_id"), "verdict_uri": src.get("verdict_uri"), "pairs": [dict(p) for p in detail], "excluded": [],
       "status": "design-detail", "reconstructed_from": {"design": sys.argv[1], "search_id": d.get("search_id"), "probe_menu_sha256": want}}
open(sys.argv[3], "w").write(json.dumps(doc, indent=1) + "\n"); print("design-detail\t" + sys.argv[3])
PYR
}
build_override_menu(){   # the menu gate is overridden for EXACTLY --pairs. The extended menu is the round-1 DESIGN's own pair details
  # (r1_trials.json allowed_pairs_detail — the probe's entries, verbatim; the probe's menu file may be archived by a reopen) plus the override pairs,
  # each synthesized from the design's entries (modules of the set; per-expert ranks by FD 6's rule when the design records them, else null; serving_certified
  # false; trainable_params unknown until the trial runs) and stamped menu_override {ruling, pairs, utc}. hpo_table / emit-trials / analyze read it
  # as the menu, so its sha is the menu generation every binding sees.
  "$PY" - "$LOCAL_ROOT/r1_trials.json" "$ADDENDUM_PAIRS" "$MENU_OVERRIDE" "$OVERRIDE_MENU" "$REPRO_REPO/scripts" "$MENU_OVERRIDE_RECORD" "${R4:-}" \
    "$(repro_done "$MODEL" "$STAGE" "$CLASS.addendum" && echo 1 || echo 0)" <<'PYO' || repro_fatal "addendum: the override menu could not be built (its reason is above)"
import json, sys, datetime, hashlib, os
sys.path.insert(0, sys.argv[5]); import tune_arm
d = json.load(open(sys.argv[1])); detail = d.get("allowed_pairs_detail") or []
if not detail: sys.exit(f"{sys.argv[1]} carries no allowed_pairs_detail — the round-1 design did not record the probe's pair entries")
pairs = [(int(r), s.strip()) for r, s in (x.split(":", 1) for x in sys.argv[2].split(",") if x.strip())]
have = {(int(p["rank"]), str(p["set"])): p for p in detail}
out = [dict(p) for p in detail]
# per_expert_rank on a synthesized pair: a design whose entries record per-expert ranks (a mixture-of-experts fit — no model in the menu
# has routed experts today, and the serving probe records no per-expert fields any more) gets FD 6's rule (reproduce/lib/models.py
# per_expert_rank: shared = rank, routed = max(1, rank // active_experts)) with a PROVABLE active-expert count — the design's recorded rule
# "max(1,r//A)" — else null and the object factory applies the rule itself; NEVER a value scaled from a lower-rank entry (the rank-4 entries
# carry the clamped floor max(1, 4 // A) = 1, and scaling it gave routed 4 / 16 at ranks 16 / 64 where the rule gives 2 / 8; every such task
# was refused by the factory). A design with no per-expert fields — every current menu model's — carries null.
import re as _re
_m = _re.fullmatch(r"max\(1,r//(\d+)\)", str(d.get("per_expert_rank_rule") or "").replace(" ", ""))
active = int(_m.group(1)) if _m else None
for rank, sid in pairs:
    if (rank, sid) in have: sys.exit(f"pair {(rank, sid)} is in the round-1 design — an addendum covers NEW pairs only")
    same_set = [p for p in detail if p["set"] == sid]
    if not same_set: sys.exit(f"no design entry of set {sid!r} to synthesize {(rank, sid)} from (sets in the design: {sorted({p['set'] for p in detail})})")
    base = min(same_set, key=lambda p: int(p["rank"]))
    expert_set = bool(base.get("per_expert_rank"))
    if expert_set and active:
        per = {"shared_expert": int(rank), "routed_expert": max(1, int(rank) // active)}; per_src = f"FD 6 rule max(1, r // {active}) (the design's recorded active_experts)"
    elif expert_set:
        per = None; per_src = "null: the object factory applies FD 6's rule with the model's active_experts (not recorded by the design)"
    else:
        per = None; per_src = "null: no routed experts in this set (no menu model has any; for a mixture-of-experts fit the object factory applies FD 6's rule itself)"
    out.append({"rank": rank, "set": sid, "lora_modules": list(base["lora_modules"]), "per_expert_rank": per, "trainable_params": None,
                "serving_certified": False, "topology": base.get("topology"), "menu_override": sys.argv[3],
                "synthesized_from": {"rank": base["rank"], "set": base["set"], "per_expert_rank_source": per_src}})
src = d.get("allowed_pairs_source") or {}
doc = {"model_id": src.get("model_id"), "probe_job_id": src.get("probe_job_id"), "verdict_uri": src.get("verdict_uri"), "result_generation": None,
       "pairs": out, "excluded": [], "status": "override",
       "menu_override": {"ruling": sys.argv[3], "pairs": [[r, s] for r, s in pairs], "utc": None,
                         "authorization": {"record": None, "sha256": hashlib.sha256(open(sys.argv[6], "rb").read()).hexdigest()},
                         "derived_from": {"design": sys.argv[1], "search_id": d.get("search_id"), "probe_menu_sha256": src.get("sha256")},
                         "note": "the user's override of the serving-probe menu gate for exactly these pairs; the probe's menu file is not consulted"}}
# IDEMPOTENT BY CONTENT (live, 2026-10-01 08:16Z: the completion run REBUILT this menu with a new build time, its sha no longer matched the identity
# the round-4 design and the refit had recorded, and the table refused by name). The build time and the authorization record's PATH are incidental
# — the authorization is its sha256 — so: (1) when the mirror's round-4 design (sys.argv[7]) names an override menu by sha and carries the build
# time + record path it was built with, the menu is re-rendered with THOSE values and taken only if it reproduces that sha EXACTLY (the menu the
# design was emitted over, byte-identical); (2) else an existing file equal to the new content modulo those two fields is kept as it is; (3) only
# NEW content writes a fresh menu (a new build time).
def render(utc, record):
    doc["menu_override"]["utc"] = utc; doc["menu_override"]["authorization"]["record"] = record
    return (json.dumps(doc, indent=1) + "\n").encode()
def strip(o):
    o = json.loads(json.dumps(o)); mo = o.get("menu_override") or {}; mo.pop("utc", None); (mo.get("authorization") or {}).pop("record", None); return o
existing = open(sys.argv[4], "rb").read() if os.path.exists(sys.argv[4]) else None
design4 = json.load(open(sys.argv[7])) if len(sys.argv) > 7 and sys.argv[7] and os.path.isfile(sys.argv[7]) else None
chosen = None; how = ""
done = len(sys.argv) > 8 and sys.argv[8] == "1"
# The review on 10fd62087 (MEDIUM): a COMPLETED addendum's menu is PINNED to the identity its round-4 design recorded — a candidate that cannot reproduce
# that sha (the authorization record's content changed, other pairs requested) must not overwrite it under the DONE marker (the table would then
# refuse on the stale sha and the completed addendum be unusable without a rederive): refuse by name, menu and DONE untouched; a changed ruling
# is a NEW generation under a new ruling id.
if design4 and (design4.get("menu_override") or {}).get("ruling") == sys.argv[3]:
    mo4 = design4["menu_override"]; want = (design4.get("allowed_pairs_source") or {}).get("sha256")
    cand = render(mo4.get("utc"), (mo4.get("authorization") or {}).get("record"))
    if want and hashlib.sha256(cand).hexdigest() == want:
        chosen, how = cand, f"the menu the round-4 design was emitted over, byte-identical (built {mo4.get('utc')})"
    else:
        sys.exit(f"the round-4 design {sys.argv[7]} was emitted over override menu sha {str(want)[:16]} (ruling {sys.argv[3]}, built {mo4.get('utc')}) and the "
                 f"current authorization record + requested pairs render a DIFFERENT menu (sha {hashlib.sha256(cand).hexdigest()[:16]}) — a completed addendum's "
                 f"menu is pinned: nothing written (menu and DONE marker untouched). A changed ruling is a NEW generation: approve it under a new ruling id "
                 f"and run --phase addendum --menu-override <new id> --pairs <its pairs>")
if chosen is None and existing is not None:
    try: ex = json.loads(existing)
    except ValueError: ex = None
    if isinstance(ex, dict) and strip(ex) == strip(doc):
        chosen, how = existing, f"unchanged (built {(ex.get('menu_override') or {}).get('utc')})"
    elif done:
        sys.exit(f"the addendum is DONE and its override menu {sys.argv[4]} is pinned (sha {hashlib.sha256(existing).hexdigest()[:16]}); the current authorization "
                 f"record + requested pairs render a DIFFERENT menu — nothing written (menu and DONE marker untouched). A changed ruling is a NEW generation under a new ruling id")
if chosen is None:
    chosen, how = render(datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), sys.argv[6]), ("REBUILT: its content changed" if existing is not None else "built now")
if existing != chosen:
    open(sys.argv[4], "wb").write(chosen)
print(f"addendum: override menu {sys.argv[4]} ({how}; sha {hashlib.sha256(chosen).hexdigest()[:16]}): the design's {len(detail)} pairs + {len(pairs)} override pairs ({sys.argv[3]}): {pairs}")
PYO
}
R1="$LOCAL_ROOT/r1_trials.json"; R2="$LOCAL_ROOT/r2_trials.json"; R9="$LOCAL_ROOT/r9_trials.json"; R3="$LOCAL_ROOT/r3_trials.json"
PICK_REQ="$LOCAL_ROOT/hpo_pick_confirm_request.json"   # the CURRENT confirm-pick request (hpo_pick.CONFIRM_PICK_REQUEST_NAME); its design path and
                                                        # job name come from the request's reserved identity, never a fixed r8_trials.json
FIT1="$LOCAL_ROOT/r1_fit.json"; FIT12="$LOCAL_ROOT/r1+2_fit_pair.json"
FIT1M="$LOCAL_ROOT/r1_fit_menu.json"; R4="$LOCAL_ROOT/r4_trials.json"   # The round-1 refit over the WIDENED menu and the addendum design
PICK_HOLD="$LOCAL_ROOT/hpo_pick.HOLD.json"      # A completed pick whose basis changed is HELD here; every consumer refuses by name while it stands
lwid(){ echo "$MODEL/$STAGE/$CLASS/$1"; }
unit(){ echo "hpo.$CLASS.$1"; }
name(){ echo "hpo-$MODEL-$CLASS-$1"; }
AP=(); [ "$CLASS" = lora ] && [ "$PROTOCOL" = fd6 ] && AP=(--allowed-pairs "$ALLOWED")   # the menu exists only under the FD 6 LoRA search

# --- helpers -------------------------------------------------------------------------------------------------------------
cluster_has(){ [ "$REPRO_DRY_RUN" = 1 ] && return 1; ssh -o BatchMode=yes -o ConnectTimeout=30 "$CL_HOST" "[ -s '$1' ]" 2>/dev/null; }
# a small cluster file's bytes on stdout (probe_progress.json, a pushed design) — 04_fit_objects.sh's reader; it was CALLED here
# (probe, confirm-pick) but never defined, so both live paths died with "command not found"
cluster_cat(){ [ "$REPRO_DRY_RUN" = 1 ] && return 1; ssh -o BatchMode=yes -o ConnectTimeout=30 "$CL_HOST" "cat '$1'" 2>/dev/null; }
# the queue is read ONCE per invocation (nothing this invocation submits is checked later in the same run)
cluster_job_active(){   # <job name> -> 0 active | 1 not active; an UNREADABLE queue refuses by name (fail CLOSED — an ssh /
  # squeue failure used to read as an EMPTY queue, cached for the invocation, so a live carrier's record could be archived under it)
  [ "$REPRO_DRY_RUN" = 1 ] && return 1
  if [ -z "${QUEUE_NAMES+x}" ]; then
    local out rc; out="$(ssh -o BatchMode=yes -o ConnectTimeout=30 "$CL_HOST" "squeue -h -u \$USER -o %j && echo __QUEUE_OK__" 2>&1)"; rc=$?
    case "$rc:$out" in
      0:*__QUEUE_OK__*) QUEUE_NAMES="${out%__QUEUE_OK__*}";;
      *) repro_fatal "the queue on $CLUSTER could not be read (ssh/squeue rc=$rc${out:+: ${out:0:200}}) — every job's state is UNKNOWN, never 'not active'; nothing archived, nothing submitted (re-run when the login host answers)";;
    esac
  fi
  grep -qxF -- "$1" <<<"$QUEUE_NAMES"
}
cluster_count(){ [ "$REPRO_DRY_RUN" = 1 ] && { echo 0; return; }; ssh -o BatchMode=yes -o ConnectTimeout=30 "$CL_HOST" "ls -d $1 2>/dev/null | wc -l" 2>/dev/null || echo 0; }
FETCHED=""
PROBE_OWNED_ROOT_FILES="allowed_pairs.json probe_verdict.json probe_fetch.json probe_plan.json probe_progress.json probe_attempt.json"   # + geometry_record*.json, launches/
fetch(){   # <remote path or glob dir> <local dir> [with-probe]: the cluster's evidence -> the grading host's mirror (summaries + metrics + fits);
           # one rsync per (remote, local) pair per invocation — nothing on the cluster changes between two calls of the same run.
           # A SEARCH-evidence fetch never delivers the PROBE-OWNED root files (menu, verdict, geometry, probe bookkeeping,
           # launches): the search record may carry an EARLIER probe's residue at the same layout (both records use this layout; no --delete),
           # and a consuming phase's fetch replaced the menu it had just revalidated against the probe record's home with that residue. Only the
           # probe phase's own READY fetch into its staging dir says `with-probe`; every consuming phase takes trial evidence only.
  local -a px=()
  [ "${3:-}" = with-probe ] || px=(--exclude=/allowed_pairs.json --exclude=/probe_verdict.json --exclude=/probe_fetch.json --exclude=/probe_plan.json --exclude=/probe_progress.json --exclude=/probe_attempt.json '--exclude=/geometry_record*.json' '--exclude=/launches/***')
  if [ "$REPRO_DRY_RUN" = 1 ]; then echo "DRY-RUN: rsync -a ${px[*]+${px[*]}} --include='*/' --include='summary.json' --include='metrics.jsonl' --include='*.json' --exclude='ckpt/**' --exclude='*' $CL_HOST:$1/ $(disp "$2")/"; return 0; fi
  case " $FETCHED " in *" $1>$2 "*) return 0;; esac
  mkdir -p "$2"
  rsync -a ${px[@]+"${px[@]}"} --include='*/' --include='summary.json' --include='metrics.jsonl' --include='*.json' --exclude='ckpt/**' --exclude='*' "$CL_HOST:$1/" "$2/" && FETCHED="$FETCHED $1>$2"
}
fetch_invalidate(){   # <remote> <local>: forget that this pair was fetched in this invocation — an archival of the local copies makes the mirror
  # stale by construction, and a later fetch of the same pair must rsync again (a LoRA --phase all run had fetched the HPO root
  # for the probe menu; the recon's digest check archived the mirror; the cached fetch then skipped, the admission found no local verdict and
  # misread the record's CURRENT verdict as foreign — archiving its rung results on the cluster and re-issuing the GPU array)
  local key=" $1>$2 "; FETCHED=" $FETCHED "; FETCHED="${FETCHED//"$key"/ }"; FETCHED="${FETCHED# }"; FETCHED="${FETCHED% }"
}
# CONTRACT: the range from the geometry_to_site function through the push function is sourced by tests under set -u with only
# MODEL/PY/CL_HOST/CL_ROOT defined (they slice the file by the first occurrence of each function header — never spell those headers above
# this line): functions and literal assignments only; no top-level line may expand any other variable (PICK_HOLD lived here, 11 reds)
geometry_to_site(){   # <local hpo root>: the probe's per-set serving-geometry record -> the SITE's overlay dir. The probe's
  # verdict writes geometry_record.json beside allowed_pairs.json (the fetch above brings it to the grading host's
  # mirror); geometry.site_geometry_dir() resolves $SISTEER_SITE_DIR/geometry/<model_key>.json for every front-door
  # process, and lib/slurm.sh's mirror step stages that dir to EVERY target cluster's $SHARED/exp/reproduce/geometry before a
  # submission — so a padded unit twinned to the other cluster reads the same record. The file is validated before it lands:
  # it must be the probe's record for THIS model (a record naming another model refuses by name); nothing is copied under
  # --dry-run. The record travels as <model_key>.json (the overlay's filename rule), never under the probe's own name.
  local src dst="$SISTEER_SITE_DIR/geometry/$MODEL.json" n=0
  for src in "$1"/geometry_record*.json; do
    [ -f "$src" ] || continue
    n=$((n + 1))
    if [ "$REPRO_DRY_RUN" = 1 ]; then echo "DRY-RUN: geometry record $(disp "$src") -> $(disp "$dst") (validated as $MODEL's before the copy)"; continue; fi
    # validated by geometry.py's OWN parser (every field the resolvers read) and written by its ONE serializer, so the site's copy
    # is byte-identical to the probe's record and to the overlay the probe wrote on its cluster
    "$PY" - "$src" "$MODEL" "$dst" "$REPRO_REPO" <<'PY' || repro_fatal "probe: $src is not $MODEL's serving-geometry record (its reason is above) — not copied to the site"
import json, sys
sys.path.insert(0, sys.argv[4] + "/src")
from sisteer.capeval.serving import geometry as g
doc = json.load(open(sys.argv[1]))
site = g.parse_site_geometry(doc, path=sys.argv[1], model_key=sys.argv[2])     # refuses another model's, a version-1, an inconsistent record
g.write_site_geometry_file(doc, sys.argv[3])
print(f"geometry record: model {site.model_key} sets {sorted(site.sets)} version {doc.get('version')} -> {sys.argv[3]} "
      f"(content sha256 {g.canonical_site_geometry_sha256(sys.argv[3])[:16]})")
PY
    repro_say "probe: serving-geometry record -> $dst; lib/slurm.sh stages the site's geometry dir to every target cluster before a submission"
  done
  [ "$n" -gt 0 ] || repro_warn "probe: no geometry_record*.json under $1 — the verdict certified no pair (or predates the fold); steered padding will refuse this model by name until a probe re-folds"
}
geometry_site_recover(){   # <local hpo root>: the recovery edge — a DONE probe whose SITE record is missing (a fresh
  # site dir, a lost copy) re-copies it from the grading host mirror through geometry_to_site; never re-fetches, never re-submits
  local dst="$SISTEER_SITE_DIR/geometry/$MODEL.json"
  if [ -f "$dst" ]; then return 0; fi
  if ls "$1"/geometry_record*.json >/dev/null 2>&1; then
    # the mirror must BE the record — a re-verdict rewrites geometry_record.json without moving the result generation, and a copy
    # from an older mirror would put a stale fold on the site while the front door says DONE; one ssh read of the record's hashes
    local rid mgeo rgeo; rid="$(probe_remote_identity)" || repro_fatal "probe: the record ${PROBE_CL_ROOT:-$CL_ROOT} on ${PROBE_CLUSTER:-$CLUSTER} could not be read for its geometry record's hash — the site copy is not recovered from an unverified mirror"
    rgeo="${rid##* }"; mgeo="$( [ -f "$1/geometry_record.json" ] && sha256sum "$1/geometry_record.json" | cut -d' ' -f1 || echo -)"
    [ "$rgeo" != - ] || repro_fatal "probe: the record ${PROBE_CL_ROOT:-$CL_ROOT} holds no geometry_record.json but the mirror $1 does — the mirror predates the record; fetch first (re-run --phase probe), nothing copied to the site"
    [ "$mgeo" = "$rgeo" ] || repro_fatal "probe: the mirror's geometry record predates the record's (mirror ${mgeo:0:12}, record ${rgeo:0:12} on ${PROBE_CLUSTER:-$CLUSTER}) — fetch first (re-run --phase probe; the stale mirror is archived and re-fetched), nothing copied to the site"
    repro_say "probe: DONE but the site's serving-geometry record $dst is missing — re-copying it from the mirror $1 (its geometry record IS the record's, ${rgeo:0:12})"
    geometry_to_site "$1"
  else
    repro_warn "probe: DONE but neither the site ($dst) nor the mirror ($1) holds geometry_record.json — re-fetch the record: rsync $CL_HOST:$CL_ROOT/geometry_record.json $1/ then re-run --phase probe"
  fi
}
# ---- the record inventory (lifted above every phase): every phase that fetches search evidence proves the mirror IS the
# record it names before reading it — table, addendum, analyze, pick ----
TRIAL_FILES_FIND='find r*_t*/ -type f \( -name "*.json" -o -name metrics.jsonl \) -not -path "*/ckpt/*" 2>/dev/null | sort'   # exactly what fetch delivers under a trial DIR (the
                                                                                                                                  # trailing slash keeps the root file r1_trials.json out)
record_inventory(){   # EVERY file the record at $CL_HOST:$CL_ROOT holds under its trial directories that a fetch would deliver — summary.json, the
  # metrics.jsonl sidecar the table's clip fractions and the addendum refit's promotability read, fit_trial_record.json and any other json —
  # hashed THERE: "<sha256>  rN_tNN/<file>" per line, sorted; exit 3 when the record directory is absent. A POSITIVE read of the record, never the
  # mirror's word for what the record holds. (Summaries alone let a rehomed record that lacked metrics be read through the previous
  # source's sidecars, and a metrics backfill never moved the table key.)
  ssh -o BatchMode=yes -o ConnectTimeout=30 "$CL_HOST" "cd '$CL_ROOT' || exit 3; $TRIAL_FILES_FIND | xargs -r sha256sum"
}
probe_owned_fingerprint(){   # the mirror's probe-owned files (menu, verdict, fetch record, geometry) BY CONTENT — taken after the revalidation and
  # before a search fetch, checked after it: a fetch that changed any of them delivered probe residue from the search record
  ( cd "$LOCAL_ROOT" 2>/dev/null && for f in allowed_pairs.json probe_verdict.json probe_fetch.json geometry_record*.json; do [ -f "$f" ] && { printf '%s ' "$f"; sha256sum "$f" | cut -d' ' -f1; }; done | sha256sum | cut -c1-16 ) 2>/dev/null
}
verify_fits(){   # <phase> -> 0 when every fit (design, pair refit, widened-menu refit) stands over the CURRENT trial evidence (rows, objective values,
  # clip evidence from the sidecars) and the menu / source bindings hold; nonzero with the reason printed (hpo_table's bindings, nothing rendered)
  local_run "$PY" "$REPRO_REPO/scripts/sdfbase/hpo_table.py" --search-dir "$LOCAL_ROOT" "${AP[@]}" --source-record "$CLUSTER:$CL_ROOT" --verify-only
}
INV=""
inventory_key(){   # the record evidence's identity (source + the verified inventory, sidecars included) — the key a DONE marker records so that the
  # phase re-runs, rather than skips, when the evidence it derived from has changed (a metrics backfill moved the table key while
  # the analyze marker still skipped the fit that predated the metrics)
  printf 'source=%s:%s\n%s\n' "$CLUSTER" "$CL_ROOT" "$INV" | sha256sum | cut -c1-16
}
source_inventory_check(){   # <phase> <fetch rc> [probe-owned fingerprint before the fetch] -> sets INV (the record inventory) or refuses. The model/class mirror is shared
  # across --cluster and fetch has no --delete, so a failed or partial fetch leaves ANOTHER record's evidence under this source's identity, and the
  # key / table / refit would certify it as this record's. The fetch must have succeeded; every summary the record holds must be on the mirror byte
  # for byte; every summary on the mirror must be on the record — anything else is refused BY NAME (nothing is deleted here: a residual is moved
  # aside by hand, "absence is never evidence" cuts both ways). The fit artefacts are local derivations and are bound to the summaries by hpo_table.
  local phase="$1" rc="$2" bad="" h f
  INV=""
  if [ "$REPRO_DRY_RUN" = 1 ]; then echo "DRY-RUN: ssh $CL_HOST 'cd $CL_ROOT && sha256sum r*_t*/summary.json' (the record inventory, checked against the mirror)"; return 0; fi
  [ "$rc" = 0 ] || repro_fatal "$phase: fetch of $CL_HOST:$CL_ROOT failed (rc=$rc) — the mirror $LOCAL_ROOT may hold another record's evidence; nothing read, derived, published or marked"
  [ -z "${3:-}" ] || [ "$(probe_owned_fingerprint)" = "$3" ] \
    || repro_fatal "$phase: the fetch of the search record $CLUSTER:$CL_ROOT CHANGED the probe-owned files on the mirror $LOCAL_ROOT (menu / verdict / geometry / fetch record) — the menu is no longer the one revalidated against the probe record's home; the search record carries probe residue a search fetch must never deliver; re-run --phase probe on the probe record's home to restore the mirror; nothing read, derived, published or marked"
  INV="$(record_inventory)" || repro_fatal "$phase: the record at $CL_HOST:$CL_ROOT cannot be read (absent, or ssh failed) — the mirror's files are not its evidence; nothing read, derived, published or marked"
  grep -q '/summary.json$' <<<"$INV" || repro_fatal "$phase: the record at $CL_HOST:$CL_ROOT holds no trial summary — nothing to work from there (the mirror's summaries are another source's); nothing read, derived, published or marked"
  while read -r h f; do                                    # every file the record holds under its trial dirs: on the mirror, byte for byte
    [ -n "$f" ] || continue
    [ -f "$LOCAL_ROOT/$f" ] && [ "$(sha256sum "$LOCAL_ROOT/$f" | cut -d' ' -f1)" = "$h" ] || bad="$bad $f(mirror differs from the record)"
  done <<<"$INV"
  for f in $(cd "$LOCAL_ROOT" 2>/dev/null && eval "$TRIAL_FILES_FIND"); do   # every such file on the mirror: on the record (a sidecar the record lacks is residue)
    awk '{print $2}' <<<"$INV" | grep -qxF -- "$f" || bad="$bad $f(on the mirror, not on the record)"
  done
  [ -z "$bad" ] || { INV=""; repro_fatal "$phase: the mirror $LOCAL_ROOT is not the record $CLUSTER:$CL_ROOT —$bad — residual evidence from another source or an incomplete fetch; move the residuals aside (nothing deleted here); nothing read, derived, published or marked"; }
  local ns nf; nf="$(printf '%s\n' "$INV" | grep -c .)"; ns="$(grep -c '/summary.json$' <<<"$INV")"
  repro_say "$phase: the mirror is the record $CLUSTER:$CL_ROOT — $ns trial summaries + $((nf - ns)) sidecars verified against the record's own hashes"
}
push(){   # <local file> <remote path>: a design file to the cluster's hpo root, one round-trip
  if [ "$REPRO_DRY_RUN" = 1 ]; then echo "DRY-RUN: ssh $CL_HOST 'mkdir -p $(dirname "$2") && cat > $2' < $(disp "$1")"; return 0; fi
  ssh -o BatchMode=yes "$CL_HOST" "mkdir -p $(dirname "$2") && cat > '$2'" < "$1"
}
job_env(){ if [ "$HAVE_SLURM_LIB" = 1 ]; then printf '%s REGISTRY_CAMPAIGN=%q' "$(slurm_job_env "$CLUSTER")" "$CAMPAIGN"; else printf 'SISTEER_SHARED_ROOT=%q SISTEER_CLUSTER_LABEL=%q REGISTRY_CAMPAIGN=%q SDFBASE_FIT_ENV_NAME=%q' "$SHARED_ROOT" "$(cluster_var "$CLUSTER" CLUSTER_LABEL)" "$CAMPAIGN" "$(cluster_fit_env_name "$CLUSTER")"; fi; }   # SDFBASE_FIT_ENV_NAME: the cluster's fit venv
submit(){   # <unit> <name> <script> "<args>" <time> [--array A] [--env "K=V …"] [--flags "--gres=gpu:N …"] [--kind fit|other] [--registers registry|none]
  # --registers: the unit's REGISTRATION CONTRACT on the funnel line (item (iv)): `none` for a unit that
  # writes only its own record dir — the RECON rungs (hpo_trial.sbatch under HPO_JOB_MODE=recon; the same sbatch's trial mode emits
  # fit_trial and stays `registry`); the serving probe is kind `other` and derives `none` from its script. The funnel binds `none` to its
  # allowlist of non-registering script+mode combinations and refuses a contradictory declaration, so a dead recon array is re-minted at
  # sacct-terminal alone (no registry database / ingest watermark needed) and a registering unit never is.
  # --kind: fit (default) reads the fit venv's certification on the cluster before minting an identity (the recon, the
  # trials, the confirm-pick run on it); `other` for the serving probe, which builds its OWN serving venv and must not be refused
  # for a fit venv it never uses. --flags: extra sbatch flags on the command line (the probe's --gres/--cpus-per-task/--mem width).
  local unit="$1" name="$2" script="$3" args="$4" tlim="$5"; shift 5
  local kind=fit registers="" rest=()
  while [ $# -gt 0 ]; do case "$1" in --kind) kind="${2:?--kind needs fit|other}"; shift 2;; --registers) registers="${2:?--registers needs registry|none}"; shift 2;; *) rest+=("$1"); shift;; esac; done
  set -- "${rest[@]+"${rest[@]}"}"
  if [ "$HAVE_SLURM_LIB" = 1 ] && [ "$REPRO_DRY_RUN" != 1 ]; then
    # the pinned sha on the target cluster BEFORE this invocation's first live submission (06/09's own step):
    # $SHA is HEAD read at run time, and a cluster checkout that lags a fresh commit fails the job's `cd <checkout>` at sbatch
    # time with nothing submitted (stage E, 2026-09-30: twice in one morning on a shared tree where peers commit). Idempotent
    # (an existing clean checkout is verified), once per invocation, and it also stages the site's geometry records.
    [ "${MIRRORED_SHA:-}" = "$SHA" ] || { slurm_mirror_clusters "$SHA" "$CLUSTER"; MIRRORED_SHA="$SHA"; }
    # SISTEER_RESUBMIT=1: the operator asks the funnel to RE-MINT a unit whose bound job died before registering (a cancelled recon);
    # the funnel's own four-way fenced proof (sacct terminal, no pending file, no spool record, the registry ingested past the job's
    # end — SISTEER_REGISTRY_MIRROR_STATUS names the watermark) still decides; without the opt-in a bound unit is skipped as before
    local resub=(); [ "${SISTEER_RESUBMIT:-0}" = 1 ] && resub=(--resubmit)
    slurm_submit --unit "$unit" --lwid "$(lwid "$unit")" --script "$script" --name "$name" --args "$args" \
                 --clusters "$CLUSTER" --time "$tlim" --kind "$kind" ${registers:+--registers "$registers"} --wave-sha "$SHA" "${resub[@]}" "$@"
    return
  fi
  # --dry-run composes the lib's own line here (lib/submit_twin.py's shape: the job-side env prefix, never --export=) so a
  # dry run touches nothing — the lib's Site() creates <state>/<model> even under --dry-run; a live submission goes through it
  local arr="" envp="" flags=""
  while [ $# -gt 0 ]; do case "$1" in --array) arr=" --array=$2"; shift 2;; --env) envp="$2 "; shift 2;; --flags) flags="${2:+ $2}"; shift 2;; *) shift;; esac; done
  local extra acct ovr=""; extra="$(cluster_var "$CLUSTER" CLUSTER_SBATCH_EXTRA)"; acct="$(cluster_var "$CLUSTER" CLUSTER_ACCOUNT)"
  [ -z "${SISTEER_RUN_OVERRIDES:-}" ] || ovr="SISTEER_RUN_OVERRIDES=$CHECKOUT/reproduce/overrides/$(basename "$SISTEER_RUN_OVERRIDES") "
  local line="cd $CHECKOUT && $(job_env) ${ovr}${envp}sbatch --qos=$(cluster_var "$CLUSTER" CLUSTER_QOS) --partition=$(cluster_var "$CLUSTER" CLUSTER_PARTITION)${acct:+ --account=$acct} --time=$tlim --output=$(cluster_var "$CLUSTER" CLUSTER_LOG_DIR)/%x_%j.out -J $name${extra:+ $extra}$arr$flags --comment=sha:$SHA $script $args"
  if [ "$REPRO_DRY_RUN" = 1 ]; then
    echo "registration contract: $unit registers=${registers:-derived} (kind $kind, $script)"   # what the funnel line will carry (a test reads it)
    echo "DRY-RUN: ssh $CL_HOST '$line'"; echo "submitted $CLUSTER DRY $unit"; return 0
  fi
  repro_fatal "reproduce/lib/slurm.sh not present — cannot submit $unit (the line would be: $line)"
}
# registry state: the fit_search ids of this campaign / model / object kind (empty when the database is absent)
registry_search_ids(){
  local root; root="$(eval echo "$SISTEER_REGISTRY_ROOT")"
  [ -f "$root/registry.sqlite" ] || return 0
  "$PY" -m sisteer.registry.cli --root "$root" query --json \
      "SELECT search_id FROM fit_search WHERE campaign='$CAMPAIGN' AND model_id='$FIT_ID' AND object_kind='$OBJECT_KIND' ORDER BY started_utc" 2>/dev/null \
    | "$PY" -c '
import json, re, sys
txt = sys.stdin.read(); ids = []
try:
    rows = json.loads(txt)
    for r in (rows if isinstance(rows, list) else []):
        v = r[0] if isinstance(r, (list, tuple)) else (r.get("search_id") if isinstance(r, dict) else r)
        if isinstance(v, str): ids.append(v)
except Exception:
    ids = re.findall(r"hpo-native:[0-9a-f]{16}", txt)
print("\n".join(dict.fromkeys(ids)))' 2>/dev/null
}
disp(){   # the printed form of a grading-host-local path: repo-relative, ~ for $HOME (the dry run prints the command, never a home path)
  local v="$1"; v="${v#"$REPRO_REPO"/}"
  case "$v" in "$SISTEER_STATE_DIR"/*) v="\$SISTEER_STATE_DIR${v#"$SISTEER_STATE_DIR"}";; "$HOME"/*) v="~${v#"$HOME"}";; esac
  printf '%s' "$v"
}
local_run(){   # <argv...>: a grading-host-local command (the interpreter + this checkout's scripts); dry-run prints it repo-relative
  if [ "$REPRO_DRY_RUN" = 1 ]; then printf 'DRY-RUN:'; local a; for a in "$@"; do printf ' %q' "$(disp "$a")"; done; echo; return 0; fi
  "$@"
}
done_or_skip(){ repro_done "$MODEL" "$STAGE" "$CLASS.$1" && { repro_say "$1: DONE marker present — skipping"; return 0; }; return 1; }
mark(){ repro_mark_done "$MODEL" "$STAGE" "$CLASS.$1" "${2:-}"; }
n_summaries(){ cluster_count "$CL_ROOT/r$1_t*/summary.json"; }
lr_ladder(){   # 5 points spanning the class's log10 lr box (tune_arm's seed boxes; the same constants the design reads)
  "$PY" - "$TUNE" "$CLASS" <<'PY'
import importlib.util, sys
spec = importlib.util.spec_from_file_location("tune_arm", sys.argv[1]); m = importlib.util.module_from_spec(spec); sys.modules["tune_arm"] = m; spec.loader.exec_module(m)
lo, hi = m.LORA_ARCH_LR_LOG10_BOX if sys.argv[2] == "lora" else m.RELU_LR_LOG10_BOX
print(",".join(f"{10 ** (lo + i * (hi - lo) / 4):.4e}" for i in range(5)))
PY
}
recon_passes_for_cell(){   # <rows> <effective batch> <recipe steps> -> the passes a rung takes (1 when one pass covers the budget);
  # a cell too short for the budget in one pass REFUSES by name (rows, steps per pass, the budget) unless the operator opted into
  # cycling (SISTEER_HPO_RECON_ALLOW_CYCLING=1), which is then RECORDED as cycled_passes > 1. Pure arithmetic: testable alone.
  local rows="$1" eff="$2" steps="$3" spp passes
  [ "$rows" -gt 0 ] 2>/dev/null || repro_fatal "recon: cannot count the rows of the recon cell ($rows) — the pairs file is missing or empty on the cluster"
  spp=$(( (rows + eff - 1) / eff ))
  if [ "$spp" -ge "$steps" ]; then echo 1; return 0; fi
  passes=$(( (steps + spp - 1) / spp ))
  if [ "${SISTEER_HPO_RECON_ALLOW_CYCLING:-0}" = 1 ]; then
    repro_warn "recon: cell $RECON_SI has $rows rows = $spp optimizer steps per pass at effective batch $eff < the recipe's $steps — CYCLING $passes passes (SISTEER_HPO_RECON_ALLOW_CYCLING=1; a methodological change, recorded as cycled_passes=$passes in the spec and recon.json)"
    echo "$passes"; return 0
  fi
  repro_fatal "recon: cell $RECON_SI has $rows rows = $spp optimizer steps per pass at effective batch $eff, below the recipe's $steps steps — a rung would stop at $spp and the recon reader files it incomplete (no clip). Name a cell with >= $((steps * eff)) rows (the paper's constv2ctl2-full; SISTEER_HPO_RECON_CELL) or opt into cycling the cell $passes times with SISTEER_HPO_RECON_ALLOW_CYCLING=1 (recorded, never silent)"
}
recon_archive_record(){   # <cluster record root> <local mirror>: every recon_*.json / recon_*.stamp.json / recon.json ALREADY on the record
  # (an earlier attempt's rungs — another cell, spec or model) moves to superseded/<utc>/ on BOTH the cluster and the grading host mirror before a
  # new array is submitted, so a changed spec starts from an EMPTY record and `--finalize` can never pool a stale rung. mv, never rm.
  # <reason> (3rd arg, optional) names the transition; the directory is UNIQUE per archival — utc to the second + nanoseconds + reason —
  # and mv -n refuses to overwrite: two archivals in the same second (a foreign local marker AND a foreign cluster verdict in one
  # invocation) once landed in the SAME superseded/<utc>/ and the second mv replaced the first
  local remote="$1" local_dir="$2" reason="${3:-archived}" stamp; stamp="$(date -u +%Y%m%dT%H%M%SZ).$(date +%N).${reason//[^A-Za-z0-9_-]/_}"
  local body='set -u; cd "$1" 2>/dev/null || { echo none; exit 0; }; found=0; for f in recon_[0-9]*.json recon_[0-9]*.stamp.json recon.json; do [ -e "$f" ] && found=1; done; if [ "$found" = 1 ]; then [ -e "superseded/$2" ] && { echo "COLLISION superseded/$2 exists"; exit 3; }; mkdir -p "superseded/$2" && for f in recon_[0-9]*.json recon_[0-9]*.stamp.json recon.json; do [ -e "$f" ] || continue; mv -n "$f" "superseded/$2/" && [ ! -e "$f" ] || { echo "COLLISION $f not moved"; exit 3; }; done; echo "archived -> $1/superseded/$2"; else echo none; fi'
  if [ "$REPRO_DRY_RUN" = 1 ]; then echo "DRY-RUN: archive any recon_*.json / recon.json under $remote and $(disp "$local_dir") to superseded/UTC/ (mv, never rm)"; return 0; fi
  local r; r="$(ssh -o BatchMode=yes -o ConnectTimeout=30 "$CL_HOST" "bash -c $(printf '%q' "$body") _ $(printf '%q' "$remote") $stamp" 2>&1)" || repro_fatal "recon: could not archive the cluster record $remote ($r) — nothing submitted, nothing overwritten"
  [ "$r" = none ] || repro_say "recon: cluster record $r"
  r="$(bash -c "$body" _ "$local_dir" "$stamp" 2>&1)" || repro_fatal "recon: could not archive the grading host mirror $local_dir ($r) — nothing overwritten"
  [ "$r" = none ] || repro_say "recon: the grading host mirror $r"
  fetch_invalidate "$remote" "$local_dir"
}
widest_pair(){   # <allowed_pairs.json> -> "rank modules" of the widest allowed pair (largest trainable_params; ties -> highest rank);
  # an EMPTY menu (every pair excluded — the live menu after the harness defect) REFUSES by name instead of `max()` of nothing
  "$PY" -c '
import json, sys
d = json.load(open(sys.argv[1])); prs = d.get("pairs") or []
if not prs:
    sys.exit(f"REFUSED: {sys.argv[1]} certifies no pair — the probe'"'"'s verdict excluded every (rank, set); re-judge the reference (--phase probe --reopen ...) before the LoRA recon")
w = max(prs, key=lambda p: (p.get("trainable_params") or 0, p["rank"]))
print(w["rank"], ",".join(w.get("lora_modules") or []) or w["set"])' "$1"
}
box_top_from_recon(){   # the certified lr top (an unstable rung lowers the box top to the highest stable one)
  "$PY" -c 'import json,sys,math; r=json.load(open(sys.argv[1])); v=r.get("certified_lr_log10_max"); print("" if v is None else f"{v:.6g}")' "$1"
}
submit_round(){   # <round N> <array spec>: the round's design (already pushed to $CL_ROOT/rN_trials.json) as one hpo_trial.sbatch array.
  # The funnel's refusal (rc 3: an uncertified fit venv, a bound unit, a failed proof) is FATAL here — a caller that only reads the next
  # line used to print "submitted" and exit 0 over a refusal (found by the real-funnel relaunch test, 2026-10-01)
  submit_design "r$1" "$CL_ROOT/r$1_trials.json" "$2" || repro_fatal "the funnel refused the round-$1 submission (its reason is above) — nothing submitted"
}
submit_design(){   # <unit/name suffix, e.g. r2 or r8-t01> <remote design path> <array spec>: ONE hpo_trial.sbatch array over that design
  submit "$(unit "$1")" "$(name "$1")" scripts/sbatch/hpo_trial.sbatch "$2 $CL_ROOT $ARM $CL_CONFIG $PAIRS $TRIAL_EPOCHS" "$SISTEER_TIME_HPO" --array "$3"
}
confirm_pick_identity(){   # <request json> -> "index request_file design job_suffix trial_id" (the identity hpo_pick reserved; refuses a request without it)
  "$PY" -c '
import json, sys
r = json.load(open(sys.argv[1]))
need = ("index", "request", "design", "job_suffix", "trial_id")
missing = [k for k in need if r.get(k) in (None, "")]
why = ", ".join(missing) if missing else ("index not an int" if not isinstance(r["index"], int) else "")
if why:
    sys.exit(f"the confirm-pick request {sys.argv[1]} reserves no identity ({why}) — re-run --phase pick (hpo_pick.py writes it)")
print(r["index"], r["request"], r["design"], r["job_suffix"], r["trial_id"])' "$1"
}
emit_from(){   # <--refine-from|--confirm-from|--explore-from> <fit artefact> <out design>: a follow-up design off an analyze artefact
  local_run "$PY" "$TUNE" emit-trials "$1" "$2" --out "$3" --arm "$ARM" --config "$LOCAL_CFG" --si "$HPO_SI"
}
refit(){   # [extra flags]: the pair-only refit over the design + refine rounds
  local_run "$PY" "$TUNE" analyze --design v2 --out "$LOCAL_ROOT" --rounds 1,2 --pair-only "${AP[@]}" "$@"
}

repro_say "03_hpo: model=$MODEL class=$CLASS (arm $ARM) fit=$FIT_ID ($FIT_REPO@${FIT_REV:0:12}) cfg=$FIT_CFG pool=$POOL_KEY/$HPO_SI cluster=$CLUSTER sha=${SHA:0:12} phase=$PHASE overrides=${OVERRIDE_ID:-?} protocol=$PROTOCOL (warmup ${WARMUP:-0.05,0.15}, clip ${CLIP_MODE}${CLIP_PCT:+ p$CLIP_PCT}, init $INIT_SCHEME, trials $N_TRIALS, recentre $( [ "$PROTOCOL" = fd6 ] && echo 3+1 || echo "hpo_next_round"))"
SEARCH_IDS="$(registry_search_ids)"
[ -z "$SEARCH_IDS" ] || repro_say "registry: fit_search $(echo "$SEARCH_IDS" | tail -1) already open for ($CAMPAIGN, $FIT_ID, $OBJECT_KIND) — phases reuse it"

# --- the probe's readers and writers ----------------------------------------------------------------------------------
json_field(){   # <json text> <key> -> the value (a string plain, anything else as JSON; null / absent / no text -> empty)
  # the document travels on STDIN, never as an argv: the live fetch fed a 174 KB allowed_pairs.json to python as ONE argument, over the
  # 128 KiB per-argument limit (`Argument list too long` x4), and written_utc / probe_job_id read as empty;
  # printf is a bash builtin, so the variable's size is not an exec limit
  printf '%s' "$1" | "$PY" -c 'import json,sys; raw=sys.stdin.read(); d=json.loads(raw) if raw.strip() else {}; v=d.get(sys.argv[1]) if isinstance(d, dict) else None; print("" if v is None else (v if isinstance(v, str) else json.dumps(v)))' "$2"
}
# THE PROBE RECORD'S HOME (the LoRA HPO lane, 2026-09-30): the menu is fetched from the cluster the serving probe RAN ON, but the recon /
# design / trials of the same class may run on ANOTHER cluster (the budget's call — the paper-companion run probed on the primary
# cluster and searched on the overflow one). Every phase that CONSUMES the menu revalidates it against the record WHERE IT LIVES — the
# fetch record this stage wrote at READY names that cluster and record dir (probe_fetch.json `cluster` / `record`) — never against
# $CLUSTER's copy of the path: on the other cluster that directory is empty, hpo_probe_status reads it NOT_STARTED, and the
# revalidation would have archived the current menu, the DONE marker and the site's geometry copy, then refused. The probe PHASE itself
# (a fresh probe, a resume, a reopen) still acts on $CLUSTER's record: with a fetch record naming another cluster it refuses by name
# instead of reading — and possibly re-probing — the wrong record. Without a fetch record (nothing fetched yet) the home is $CLUSTER's.
PROBE_CLUSTER=""; PROBE_CL_HOST=""; PROBE_CL_ROOT=""; PROBE_HOME_SOURCE=""
probe_local_evidence(){   # -> the local probe evidence that exists (menu, verdict, geometry record, DONE marker), space-separated; empty when none
  local ev="" marker; marker="$(repro_state_dir "$MODEL")/$STAGE/DONE.$CLASS.probe"
  [ -s "$ALLOWED" ] && ev="$ev allowed_pairs.json"
  [ -s "$LOCAL_ROOT/probe_verdict.json" ] && ev="$ev probe_verdict.json"
  ls "$LOCAL_ROOT"/geometry_record*.json >/dev/null 2>&1 && ev="$ev geometry_record.json"
  [ -f "$marker" ] && ev="$ev DONE.$CLASS.probe"
  printf '%s' "${ev# }"
}
probe_home(){   # sets PROBE_CLUSTER / PROBE_CL_HOST / PROBE_CL_ROOT (+ PROBE_HOME_SOURCE) from the local fetch record; --cluster's home ONLY when
  # nothing local exists: local probe evidence whose fetch record is absent / empty / malformed / names no home or an
  # unknown cluster REFUSES by name — on another cluster --cluster's path is empty, reads NOT_STARTED, and the revalidation would archive the
  # good menu, the DONE marker and the site copy (or the probe phase would submit a fresh probe over it). The remedy is --rehome.
  PROBE_CLUSTER="$CLUSTER"; PROBE_CL_HOST="$CL_HOST"; PROBE_CL_ROOT="$CL_ROOT"; PROBE_HOME_SOURCE="--cluster"
  local f="$LOCAL_ROOT/probe_fetch.json" defect="" doc c="" r=""
  if [ ! -e "$f" ]; then defect="absent"
  elif [ ! -s "$f" ]; then defect="empty"
  else
    doc="$(cat "$f")"
    if ! printf '%s' "$doc" | "$PY" -c 'import json,sys; d=json.load(sys.stdin); sys.exit(0 if isinstance(d, dict) else 1)' 2>/dev/null; then defect="malformed (not a JSON object)"
    else
      c="$(json_field "$doc" cluster)"; r="$(json_field "$doc" record)"
      if [ -z "$c" ] && [ -z "$r" ]; then defect="from before the home fields (it names no cluster and no record)"
      elif [ -z "$c" ]; then defect="naming no cluster"
      elif [ -z "$r" ]; then defect="naming no record"
      else case " $SISTEER_CLUSTERS " in *" $c "*) ;; *) defect="naming cluster '$c', not one of SISTEER_CLUSTERS ($SISTEER_CLUSTERS)";; esac
      fi
    fi
  fi
  if [ -z "$defect" ]; then PROBE_CLUSTER="$c"; PROBE_CL_HOST="$(cluster_var "$c" CLUSTER_SSH)"; PROBE_CL_ROOT="$r"; PROBE_HOME_SOURCE="probe_fetch.json"; return 0; fi
  local ev; ev="$(probe_local_evidence)"
  [ -n "$ev" ] || return 0                                   # genuinely nothing local (a first probe; a mirror archived to superseded/): --cluster's home stands
  repro_fatal "probe: the local probe evidence under $LOCAL_ROOT ($ev) has no fetch record naming its home — $f is $defect. Which record it came from cannot be told from --cluster's path ($CLUSTER:$CL_ROOT — on another cluster that path is empty, reads NOT_STARTED, and would archive this evidence or take a fresh probe over it). Nothing read, nothing archived, nothing submitted. Remedy: --phase probe --rehome re-derives the home from the mirror's own probe_verdict.json (provenance.probe_cluster / probe_record), verifies that cluster holds the record and writes the fetch record; evidence you no longer want moves under $LOCAL_ROOT/superseded/<utc>/ by hand (never deleted) before a fresh --phase probe --cluster <label>"
}
probe_job_active(){   # <job name> -> 0 active | 1 not — on the PROBE cluster's queue (the same fail-CLOSED read as cluster_job_active; one read
  # per invocation per host); the very same read when the probe cluster IS $CLUSTER
  [ -n "$PROBE_CL_HOST" ] || probe_home
  [ "$PROBE_CL_HOST" != "$CL_HOST" ] || { cluster_job_active "$1"; return; }
  [ "$REPRO_DRY_RUN" = 1 ] && return 1
  if [ -z "${PROBE_QUEUE_NAMES+x}" ]; then
    local out rc; out="$(ssh -o BatchMode=yes -o ConnectTimeout=30 "$PROBE_CL_HOST" "squeue -h -u \$USER -o %j && echo __QUEUE_OK__" 2>&1)"; rc=$?
    case "$rc:$out" in
      0:*__QUEUE_OK__*) PROBE_QUEUE_NAMES="${out%__QUEUE_OK__*}";;
      *) repro_fatal "the queue on $PROBE_CLUSTER could not be read (ssh/squeue rc=$rc${out:+: ${out:0:200}}) — the probe job's state is UNKNOWN, never 'not active'; nothing archived, nothing submitted (re-run when the login host answers)";;
    esac
  fi
  grep -qxF -- "$1" <<<"$PROBE_QUEUE_NAMES"
}
PROBE_REMOTE_IDS=""
probe_remote_identity(){   # -> "verdict_sha256 menu_sha256 geometry_sha256" of the record's files ON the cluster ('-' for an absent file); one ssh,
  # read ONCE per invocation (a re-verdict rewrites these files without moving the result generation — they, not the generation, are the
  # verdict's identity). Read at the record's HOME when a phase has resolved one (PROBE_CL_HOST / PROBE_CL_ROOT, the fetch record's cluster —
  # the LoRA flow's probe_home), else --cluster's record
  if [ -z "$PROBE_REMOTE_IDS" ]; then
    PROBE_REMOTE_IDS="$(ssh -o BatchMode=yes -o ConnectTimeout=30 "${PROBE_CL_HOST:-$CL_HOST}" "cd '${PROBE_CL_ROOT:-$CL_ROOT}' 2>/dev/null || exit 3; for f in probe_verdict.json allowed_pairs.json geometry_record.json; do if [ -f \"\$f\" ]; then sha256sum \"\$f\" | cut -d' ' -f1; else echo -; fi; done" | tr '\n' ' ')" || return 3
    PROBE_REMOTE_IDS="${PROBE_REMOTE_IDS% }"
    [ "$(wc -w <<<"$PROBE_REMOTE_IDS")" = 3 ] || { PROBE_REMOTE_IDS=""; return 3; }
  fi
  printf '%s' "$PROBE_REMOTE_IDS"
}
probe_local_identity(){   # <local root> -> the same three hashes of the MIRROR's files ('-' when absent)
  local f out=""; for f in probe_verdict.json allowed_pairs.json geometry_record.json; do
    if [ -f "$1/$f" ]; then out="$out$(sha256sum "$1/$f" | cut -d' ' -f1) "; else out="$out- "; fi
  done; printf '%s' "${out% }"
}
probe_fetch_fields(){   # <status json> <identity "v m g"> <written_utc> <probe_job_id>: FR_GEN FR_N FR_CHK FR_PCHK FR_PGEN FR_V FR_M FR_G — the fields the fetch record and the marker share
  FR_GEN="$(json_field "$1" generation)"; FR_N="$(json_field "$1" n_results)"; FR_CHK="$(json_field "$1" generation_check)"
  FR_PCHK="$(json_field "$1" parity_check)"; FR_PGEN="$(json_field "$(json_field "$1" parity_record)" stamped_generation)"
  read -r FR_V FR_M FR_G <<<"$2"
}
probe_write_fetch_record(){   # <status json> <identity "v m g"> <written_utc> <probe_job_id>: probe_fetch.json — the record's HOME (cluster + dir, as the
  # phase resolved it) and the VERDICT identity actually fetched — written ATOMICALLY (repro_write: temp + rename) and, in probe_fetch_ready,
  # BEFORE any menu / verdict / geometry file is exposed in the mirror (evidence never exists without a home). The DONE
  # marker is probe_mark_fetched, written LAST.
  probe_fetch_fields "$1" "$2" "$3" "$4"
  repro_write "$LOCAL_ROOT/probe_fetch.json" "{\"generation\": \"$FR_GEN\", \"n_results\": ${FR_N:-0}, \"generation_check\": \"$FR_CHK\", \"verdict_sha256\": \"$FR_V\", \"menu_sha256\": \"$FR_M\", \"geometry_sha256\": \"$FR_G\", \"written_utc\": \"$3\", \"probe_job_id\": \"$4\", \"parity_check\": \"${FR_PCHK:-none}\", \"parity_stamped_generation\": \"$FR_PGEN\", \"cluster\": \"${PROBE_CLUSTER:-$CLUSTER}\", \"record\": \"${PROBE_CL_ROOT:-$CL_ROOT}\", \"fetched_utc\": \"$(date -u +%FT%TZ)\"}"
}
probe_mark_fetched(){   # <the same four>: the DONE marker — after the site copy — with the same tokens as the fetch record
  probe_fetch_fields "$1" "$2" "$3" "$4"
  mark probe "${PROBE_CL_ROOT:-$CL_ROOT}/allowed_pairs.json generation=$FR_GEN n_results=$FR_N verdict_sha256=$FR_V menu_sha256=$FR_M geometry_sha256=$FR_G written_utc=$3 probe_job_id=$4 parity_check=${FR_PCHK:-none} parity_stamped_generation=$FR_PGEN"
}
probe_expose_staged(){   # <staging dir>: the staged record -> the mirror. Directories merge, every FILE is a rename into place (os.replace: atomic per
  # file, same filesystem), and the evidence files — allowed_pairs.json, probe_verdict.json, geometry_record*.json — go LAST, so the fetch record
  # (already written) precedes every byte of evidence; the staging dir is removed after
  "$PY" - "$1" "$LOCAL_ROOT" <<'PY' || repro_fatal "probe: could not move the staged record $1 into the mirror $LOCAL_ROOT (its reason is above) — the fetch record already names the record; re-run --phase probe (the mirror is re-fetched)"
import os, shutil, sys
stage, root = sys.argv[1], sys.argv[2]
late = []
for dirpath, _dirnames, filenames in os.walk(stage):
    rel = os.path.relpath(dirpath, stage)
    dst_dir = root if rel == "." else os.path.join(root, rel)
    os.makedirs(dst_dir, exist_ok=True)
    for name in filenames:
        src, dst = os.path.join(dirpath, name), os.path.join(dst_dir, name)
        if rel == "." and (name in ("allowed_pairs.json", "probe_verdict.json") or (name.startswith("geometry_record") and name.endswith(".json"))):
            late.append((src, dst)); continue
        os.replace(src, dst)
for src, dst in late:
    os.replace(src, dst)
shutil.rmtree(stage)
PY
}
probe_status(){   # <job-active 0|1> -> "STATE<TAB>detail<TAB>json": scripts/sdfbase/hpo_probe_status.py --record, run ON the cluster over the record dir
  # THE RECORD IS THE AUTHORITY: READY needs a complete checkpoint AND a verdict + menu tied to the record's CURRENT result generation
  # (a record reopened by `run --resume --replan` after its menu was written is MENU_STALE, never fetched). The module is stdlib-only
  # by contract and travels on the ssh's stdin, so the grading host's committed reader runs whatever the cluster checkout holds. Under
  # --dry-run nothing is read and the state is NOT_STARTED, so the fresh submission line prints. The record read is the PROBE record's
  # home (probe_home): $CLUSTER's own for the probe phase, the fetch record's cluster for every consuming phase.
  [ -n "$PROBE_CL_HOST" ] || probe_home
  local root="${2:-$PROBE_CL_ROOT}"      # The parity-pass record's dir when the reopen is of THAT record; the probe record's home otherwise
  if [ "$REPRO_DRY_RUN" = 1 ]; then printf 'NOT_STARTED\tdry run: the record is not read (live: hpo_probe_status.py --record %s on %s)\t{}\n' "$root" "$PROBE_CLUSTER"; return 0; fi
  local out rc
  out="$(ssh -o BatchMode=yes -o ConnectTimeout=30 "$PROBE_CL_HOST" "python3 - --record '$root' --job-active $1" < "$REPRO_REPO/scripts/sdfbase/hpo_probe_status.py")"; rc=$?
  [ "$rc" = 0 ] && [ -n "$out" ] || repro_fatal "probe: hpo_probe_status.py --record $root gave no verdict on $PROBE_CLUSTER (rc=$rc${out:+: ${out:0:300}}) — the record's state is unknown; nothing fetched, nothing submitted"
  printf '%s\n' "$out"
}
probe_fetch_ready(){   # <status json>: READY -> the menu, the verdict and the record's JSON to the grading host mirror, the geometry record to the site, DONE with the generation
  local gen n chk remote_menu remote_utc remote_job local_utc got_utc got_job
  gen="$(json_field "$1" generation)"; n="$(json_field "$1" n_results)"; chk="$(json_field "$1" generation_check)"
  [ -n "$gen" ] || repro_fatal "probe: READY without a result generation in the status record ($1) — refusing to fetch"
  remote_menu="$(cluster_cat "$CL_ROOT/allowed_pairs.json")" || repro_fatal "probe: could not read $CL_ROOT/allowed_pairs.json on $CLUSTER"
  remote_utc="$(json_field "$remote_menu" written_utc)"; remote_job="$(json_field "$remote_menu" probe_job_id)"
  # a fetch never overwrites a NEWER mirror: a local menu written AFTER the cluster's names a record that moved on (or a cluster
  # holding a stale copy) — refuse by name, never rsync over it
  if [ -s "$ALLOWED" ]; then
    local_utc="$(json_field "$(cat "$ALLOWED")" written_utc)"
    if [ -n "$local_utc" ] && [[ "$local_utc" > "$remote_utc" ]]; then
      repro_fatal "probe: the grading host mirror $ALLOWED (written $local_utc) is NEWER than $CLUSTER's menu (written $remote_utc, probe job $remote_job) — not overwriting it; reconcile the record by hand before re-running --phase probe"
    fi
  fi
  local rid lid stage; rid="$(probe_remote_identity)" || repro_fatal "probe: the record $CL_ROOT on $CLUSTER could not be read for its verdict identity — nothing fetched, nothing marked"
  # ATOMIC exposure: the record is rsynced into a STAGING dir inside the mirror and verified THERE (the identity the
  # status saw, the menu it saw, the geometry stamps); the fetch record — the HOME and the verdict identity — is written into the mirror FIRST
  # (temp + rename); only then are the staged files moved into place, the evidence files last. Menu evidence therefore never exists in the
  # mirror without a fetch record naming its home: an interrupted fetch used to leave a menu with no record, and a later `--phase recon
  # --cluster B` read B's empty path as NOT_STARTED and archived the good menu, the DONE marker and the site copy. A staging dir left by an
  # interrupted fetch is removed first; SISTEER_PROBE_FETCH_DIE_FILE (tests only) kills the process between the rsync and the record write.
  stage="$LOCAL_ROOT/.fetch.staging"; rm -rf "$stage"; mkdir -p "$stage"
  fetch "$CL_ROOT" "$stage" with-probe || { rm -rf "$stage"; repro_fatal "probe: the fetch from $CLUSTER:$CL_ROOT failed — nothing exposed, nothing marked"; }
  [ -z "${SISTEER_PROBE_FETCH_AFTER_RSYNC_HOOK:-}" ] || bash -c "$SISTEER_PROBE_FETCH_AFTER_RSYNC_HOOK"      # tests only: mutate the record between the rsync and the re-read
  # the record's identity is read AGAIN after the rsync — a re-verdict landing between the status read and the rsync
  # (or during it) would otherwise be stamped DONE from the OLD files; remote-before == remote-after == staged, or nothing is exposed
  local rid2; PROBE_REMOTE_IDS=""; rid2="$(probe_remote_identity)" || { rm -rf "$stage"; repro_fatal "probe: the record $CL_ROOT on $CLUSTER could not be re-read for its verdict identity after the rsync — nothing exposed, nothing marked"; }
  [ "$rid2" = "$rid" ] || { rm -rf "$stage"; repro_fatal "probe: the record $CL_ROOT on $CLUSTER changed DURING the fetch (verdict / menu / geometry $rid before the rsync, $rid2 after) — nothing exposed, nothing marked; re-run --phase probe"; }
  lid="$(probe_local_identity "$stage")"
  [ "$lid" = "$rid" ] || { rm -rf "$stage"; repro_fatal "probe: the fetched mirror's verdict / menu / geometry ($lid) is not what the status verified ($rid) — the record changed during the fetch; re-run --phase probe (nothing exposed, nothing marked)"; }
  # the fetched menu must be the one the status verified (a record that moved during the fetch is not exposed)
  got_utc="$(json_field "$(cat "$stage/allowed_pairs.json")" written_utc)"; got_job="$(json_field "$(cat "$stage/allowed_pairs.json")" probe_job_id)"
  [ "$got_utc" = "$remote_utc" ] && [ "$got_job" = "$remote_job" ] || { rm -rf "$stage"; repro_fatal "probe: the fetched allowed_pairs.json (written $got_utc, job $got_job) is not the menu the status verified (written $remote_utc, job $remote_job) — the record changed during the fetch; nothing exposed, nothing marked"; }
  # the geometry record travels ONLY when its fold is of the record's CURRENT result generation (170c98a29 stamps
  # provenance.result_generation; a fold of a superseded result set refuses by name — and is never exposed; a fold written before the
  # stamp existed is noted and copied on the strength of READY, the way geometry.check_overlay_generation reads an unstamped overlay)
  local grec gstamp
  for grec in "$stage"/geometry_record*.json; do
    [ -f "$grec" ] || continue
    gstamp="$("$PY" -c 'import json,sys; d=json.load(open(sys.argv[1])); print(((d.get("provenance") or {}).get("result_generation")) or "")' "$grec")"
    if [ -n "$gstamp" ] && [ "$gstamp" != "$gen" ]; then
      rm -rf "$stage"; repro_fatal "probe: $(basename "$grec") is stamped for result generation ${gstamp:0:12}, the record's is ${gen:0:12} — a fold of a superseded result set; not exposed, not copied to the site, nothing marked"
    fi
    [ -n "$gstamp" ] || repro_note "probe: $(basename "$grec") carries no result_generation stamp (a fold written before the stamp existed) — copied on the strength of READY"
  done
  # a mirror already holding OTHER evidence (a stale verdict re-homed by --rehome, an earlier fetch interrupted after its exposure) is
  # ARCHIVED before the re-fetch exposes the record — probe_archive_local (superseded/<utc>/, never deleted), the move the DONE
  # revalidation makes; identical evidence (the same bytes) is simply replaced. A fetch record ALONE — an archival that stopped at its
  # last move (probe_archive_local moves the fetch record last, so whatever stands keeps its home), a fetch that died between its
  # record and its exposure — is archived the same way, never written over by the new record
  local ev; ev="$(probe_local_evidence)"
  [ -n "$ev" ] || [ ! -s "$LOCAL_ROOT/probe_fetch.json" ] || ev="probe_fetch.json alone — a fetch record with no evidence beside it"
  if [ -n "$ev" ] && [ "$(probe_local_identity "$LOCAL_ROOT")" != "$rid" ]; then
    probe_archive_local probe READY "the mirror ($ev) is not the record's current verdict — archived before the re-fetch" "$1"
  fi
  if [ -n "${SISTEER_PROBE_FETCH_DIE_FILE:-}" ] && [ -e "$SISTEER_PROBE_FETCH_DIE_FILE" ]; then echo "probe: test hook — dying between the rsync and the fetch record" >&2; kill -KILL $$; fi
  probe_write_fetch_record "$1" "$lid" "$got_utc" "$got_job"     # the fetch record FIRST: the home + the identity ACTUALLY fetched — the fetched menu's own fields, the three hashes, the parity stamp
  probe_expose_staged "$stage"                                    # then the files, the evidence files last
  FETCHED="$FETCHED $CL_ROOT>$LOCAL_ROOT"                         # the mirror IS the record now: a later fetch of this pair in this invocation (the recon's) may skip
  repro_say "probe: allowed_pairs.json READY on $CLUSTER (result generation ${gen:0:12} — ${chk:-?}, ${n:-?} result files; written $remote_utc by probe job $remote_job) — fetched to $ALLOWED"
  geometry_to_site "$LOCAL_ROOT"     # the per-set serving-geometry record travels with the front door
  probe_mark_fetched "$1" "$lid" "$got_utc" "$got_job"           # the DONE marker LAST: it certifies the site copy too
}
probe_fresh_binding(){   # [unit] -> "<state> <job_id> <cluster>" of the FRESH probe unit's durable binding (lib/stage.py; unbound -> "unbound  ");
  # the parity pass passes its own unit ($(unit probe).parity)
  "$PY" - "$REPRO_LIB_DIR" "$(repro_state_dir "$MODEL")/submissions.jsonl" "$(lwid "${1:-$(unit probe)}")" <<'PY2'
import sys; sys.path.insert(0, sys.argv[1]); import stage as ST
b = ST.binding_state(sys.argv[2], sys.argv[3]); print(b.get("state", "unbound"), b.get("job_id") or "", b.get("cluster") or "")
PY2
}
probe_prove_dead(){   # <job id from the attempt record>: the bound job of the FRESH unit — else that job — must be TERMINAL per sacct through the
  # funnel's OWN proof (submit_twin.Coordinator.resubmit_decision, read-only; lib/submit_twin.py is the funnel's — nothing here edits it) BEFORE
  # the record is touched: a live carrier still bootstrapping (attempt record written, no checkpoint yet) reads FAILED_START, and
  # archiving its attempt file would lose the running job's record. `skip` (a task still active), a Refused proof, an unknown job -> refuse.
  local out rc
  out="$("$PY" - "$REPRO_LIB_DIR" "$SISTEER_CONFIG" "$MODEL" "$(lwid "$(unit probe)")" "$(unit probe)" "$CLUSTER" "${1:-}" <<'PY3'
import sys
lib, config, model, lwid, unit, label, job = sys.argv[1:8]
sys.path.insert(0, lib)
import submit_twin as SW, stage as ST
try:
    site = SW.Site(config, False); co = SW.Coordinator(site, model)
    b = ST.binding_state(co.subs.path, lwid)
    if b.get("state") == "transition":
        # the funnel's re-run of this unit is MID-FLIGHT (23eed2e32: the token is written BEFORE its sbatch and cleared only after the new
        # attempt is bound; the funnel reconciles it by the intended job NAME since the token's submit time). The old attempt's job id proves
        # nothing about the replacement the funnel may already have launched — a REFUSAL until reconciled, never a fall-through.
        tok = ST.read_transition(co.subs.path, lwid) or {}
        intended = tok.get("intended") if isinstance(tok.get("intended"), dict) else {}
        print("refuse\x1f\x1fthe fresh unit carries a TRANSITION token " + ST.transition_path(co.subs.path, lwid)
              + f" (written {tok.get('utc') or tok.get('written_utc') or '?'}, intended attempt {intended.get('attempt_id') or tok.get('attempt_id') or '?'}, job name "
              + f"{intended.get('job_name') or tok.get('job_name') or '?'}) — the funnel's re-run is mid-flight; its reconciliation must run first "
              "(a plain lib/slurm.sh submit of the unit reconciles the token against sacct by the intended job name since the token's submit time: one "
              "unknown job → adopted, none → restored, several → refused); the old attempt's job id is NOT the job to prove"); sys.exit(0)
    if b.get("state") in ("submitted", "registered"):
        action, why, _ev = co.resubmit_decision(lwid, unit, b, "other", None)
        print(f"{action}\x1f{b.get('job_id')}\x1fbound job {b.get('job_id')} on {b.get('cluster')}: {why}")
    else:
        if not job:
            print("refuse\x1f\x1fthe fresh unit is unbound and the attempt record names no job — nothing to prove terminal"); sys.exit(0)
        tasks = co.remotes[label].sacct_tasks(str(job))
        if not tasks:
            print(f"refuse\x1f\x1fjob {job} is unknown to sacct on {label} — its terminal state cannot be read"); sys.exit(0)
        active = [t for t in tasks if not t["terminal"]]
        if active:
            print("skip\x1f" + str(job) + "\x1fjob " + str(job) + " still " + ", ".join(str(t["state"]) for t in active) + " per sacct"); sys.exit(0)
        print(f"resubmit\x1f{job}\x1fjob {job}: sacct says {','.join(sorted({str(t['state']) for t in tasks}))} (terminal) for every task")
except SW.Refused as e:
    print(f"REFUSED: {e}", file=sys.stderr); sys.exit(3)
except SW.Unreachable as e:
    print(f"REFUSED: sacct on {label} unreachable ({e})", file=sys.stderr); sys.exit(3)
PY3
)"; rc=$?
  [ "$rc" = 0 ] && [ -n "$out" ] || repro_fatal "probe: the funnel's proof did not clear the dead job of $CL_ROOT (rc=$rc; its reason is above) — nothing archived, nothing submitted"
  local action why; IFS=$'\x1f' read -r action PROVED_JOB why <<<"$out"   # \x1f: a non-whitespace separator keeps an EMPTY job field (TABs would collapse)
  case "$action" in
    resubmit|rerun) repro_say "probe: the funnel's proof: $action — $why";;
    *) repro_fatal "probe: the job that wrote $CL_ROOT is NOT proven dead — $action: $why; nothing archived, nothing submitted";;
  esac
  # the proof covered ONE job; the attempt record on the cluster must name that job, else the record is another generation's
  [ -z "${1:-}" ] || [ -z "${PROVED_JOB:-}" ] || [ "$1" = "$PROVED_JOB" ] \
    || repro_fatal "probe: the attempt record on $CL_ROOT names job $1 but the funnel's proof covered job $PROVED_JOB — a different attempt generation; a terminal proof of another generation never authorises moving the record; nothing archived, nothing submitted"
}
probe_recover_planless(){   # <status json> <proved job id>: the guarded recovery of a FAILED_START record with NO plan — every
  # top-level file of the record (probe_attempt.json, the carrier's RESULT written by its EXIT trap, any leftover) and an adapters/ dir move
  # to superseded/<utc>/ ON THE RECORD, never deleted, so hpo_probe_status stops reading FAILED_START; the caller then re-reads the status
  # and hands the FRESH submission to the funnel with --resubmit (its fenced proof decides). Only under --resubmit; refused otherwise.
  # The archive directory on the record is RESERVED (a bare mkdir; the next `-NN` suffix on a collision — the rule of lib/common.sh
  # repro_superseded_dir, inlined: an ssh string cannot call a grading-host shell function; the stamp is the grading host's clock to the
  # microsecond, repro_superseded_stamp's one format) and reported as `ARCHIVE_DIR=<dir>` on its own line, which the message below names.
  # Until 2026-10-09 it was `mkdir -p superseded/<second>` + `mv -f` — the stage-02 / geometry-chain defect: a same-second archival
  # written over. Here the fence below makes a second archival of ONE record in one instant unreachable (probe_attempt.json is gone
  # once moved); the reservation holds the rule where the fence does not reach (a directory an operator or an earlier run left).
  local utc moved arch_rel expect="${2:-}"; utc="$(repro_superseded_stamp)"
  [ "$(json_field "$1" plan_present)" = true ] && repro_fatal "probe: internal — probe_recover_planless called on a record with a plan"
  [ -n "$expect" ] || repro_fatal "probe: internal — probe_recover_planless needs the job id the terminal proof covered; nothing archived"
  # test hook: pause here (proof done, nothing moved) so a replacement attempt can be planted before the archive runs
  if [ -n "${SISTEER_PROBE_PAUSE_BEFORE_ARCHIVE_FILE:-}" ]; then while [ -e "$SISTEER_PROBE_PAUSE_BEFORE_ARCHIVE_FILE" ]; do sleep 0.2; done; fi
  # THE GENERATION FENCE: inside the SAME remote critical section as the mv, probe_attempt.json is re-read and must
  # still name the job the terminal proof covered — a funnel retry that accepted a replacement after the status read wrote a NEW attempt
  # record into this directory, and moving it would take the LIVE replacement's provenance; a vanished record refuses too
  moved="$(ssh -o BatchMode=yes -o ConnectTimeout=30 "$CL_HOST" "set -u; cd '$CL_ROOT' || exit 3; [ -e probe_plan.json ] || [ -e probe_progress.json ] || [ -d launches ] && { echo 'REFUSED: the record holds a plan, a checkpoint or launches' >&2; exit 3; }; [ -f probe_attempt.json ] || { echo 'REFUSED: probe_attempt.json is gone since the terminal proof — the record changed; nothing moved' >&2; exit 4; }; cur=\$(python3 -c 'import json,sys; print(str(json.load(open(sys.argv[1])).get(\"job_id\") or \"\"))' probe_attempt.json 2>/dev/null || echo UNREADABLE); [ \"\$cur\" = '$expect' ] || { echo \"REFUSED: probe_attempt.json now names job \$cur, the terminal proof covered job $expect — a replacement attempt wrote the record between the proof and the archive; nothing moved\" >&2; exit 4; }; mkdir -p superseded || { echo 'REFUSED: superseded/ under the record could not be made — nothing moved' >&2; exit 5; }; A='superseded/$utc'; k=1; until mkdir \"\$A\" 2>/dev/null; do [ -e \"\$A\" ] || { echo 'REFUSED: no archive directory could be reserved under superseded/ (not a writable directory) — nothing moved' >&2; exit 5; }; k=\$((k + 1)); A=\"superseded/$utc-\$(printf '%02d' \$k)\"; done; n=0; for f in * .[!.]*; do [ -f \"\$f\" ] || { [ -d \"\$f\" ] && [ \"\$f\" = adapters ]; } || continue; mv -f -- \"\$f\" \"\$A/\" && { printf '%s ' \"\$f\"; n=\$((n + 1)); }; done; [ \"\$n\" -gt 0 ] || rmdir \"\$A\" 2>/dev/null || true; echo; echo \"ARCHIVE_DIR=\$A\"")" \
    || repro_fatal "probe: the recovery of the plan-less record $CL_ROOT on $CLUSTER failed (its reason is above) — nothing submitted"
  arch_rel="$(printf '%s\n' "$moved" | sed -n 's/^ARCHIVE_DIR=//p' | tail -1)"
  moved="$(printf '%s\n' "$moved" | sed '/^ARCHIVE_DIR=/d' | tr '\n' ' ' | sed 's/^ *//; s/ *$//')"
  repro_say "probe: recovered the plan-less record $CL_ROOT — moved to ${arch_rel:-superseded/$utc}/ (never deleted): ${moved:-nothing}"
}
probe_width_flags(){   # <N|auto> -> the carrier's documented narrower-allocation flags (16 CPUs + 128G per GPU); nothing for auto (the header's width)
  case "$1" in auto|"") return 0;; esac
  printf -- '--gres=gpu:%d --cpus-per-task=%d --mem=%dG' "$1" "$((16 * $1))" "$((128 * $1))"
}
probe_paper_pair(){   # -> "<set id> <rank>" on stdout: the ONE pair the paper's protocol declares (demo-import M6; Q3 ruled (a)) — serving_probe.sbatch's
  # positionals 4/5 on a FRESH submission, so the carrier's menu is that pair alone rather than its default 15-pair menu (--probe-width narrows
  # the allocation only). THE SOURCE is the fits config's train_kwargs.lora_r4_ce.fit_selection (lora_rank + lora_modules; the sorted modules
  # mapped to the lora_module_sets.MODULE_SETS id whose tuple equals them: [o_proj] -> wo), cross-checked against the row's lora_architecture in
  # models.yaml; absent, malformed, unmatched or disagreeing REFUSES by name (never a guessed pair; never the --rank / --set pick switches, which
  # record a search's winner, and never a flag of its own — the pair is declared in exactly one place). A resume / re-verdict keeps the
  # record's own probe_plan.json sets / ranks (probe_resume, probe_submit_reverdict) and does not call this.
  "$PY" - "$REPRO_REPO" "$LOCAL_CFG" "$REPRO_LIB_DIR" "$(repro_models_table)" "$MODEL" <<'PY'
import sys, yaml
repo, cfg_path, lib, table, key = sys.argv[1:6]
sys.path.insert(0, repo + "/src"); sys.path.insert(0, lib)
from sisteer.extraction.lora_module_sets import MODULE_SETS
import models as M                                   # the duplicate-key-rejecting loader (review A9), never a bare safe_load of the table
cfg = yaml.safe_load(open(cfg_path, encoding="utf-8")) or {}
sel = ((cfg.get("train_kwargs") or {}).get("lora_r4_ce") or {}).get("fit_selection")
if not isinstance(sel, dict):
    sys.exit(f"{cfg_path}: no train_kwargs.lora_r4_ce.fit_selection — the paper protocol's probe pair is READ from it (lora_rank + lora_modules); "
             "regenerate the config (scripts/sdfbase/make_sdfbase_configs.py)")
rank, mods = sel.get("lora_rank"), sel.get("lora_modules")
if isinstance(rank, bool) or not isinstance(rank, int) or rank < 1:
    sys.exit(f"{cfg_path}: fit_selection.lora_rank={rank!r} is not a positive integer — the probe pair's rank is the config's, never typed")
if not isinstance(mods, list) or not mods or not all(isinstance(m, str) and m for m in mods):
    sys.exit(f"{cfg_path}: fit_selection.lora_modules={mods!r} is not a non-empty list of module names")
want = tuple(sorted(set(mods)))
ids = [sid for sid, tup in MODULE_SETS.items() if tuple(sorted(tup)) == want]
if not ids:
    sys.exit(f"{cfg_path}: fit_selection.lora_modules {sorted(set(mods))} match no module set of lora_module_sets.MODULE_SETS "
             f"({ {k: list(v) for k, v in MODULE_SETS.items()} }) — the carrier names sets by id; nothing submitted")
set_id = ids[0]
la = M.model(key, table).get("lora_architecture") or {}
la_rank, la_mods = la.get("rank"), la.get("modules")
if la_rank != rank or tuple(sorted(set(la_mods or []))) != want:
    sys.exit(f"the fits config {cfg_path} declares rank {rank} / {sorted(set(mods))} (set {set_id}) but {table} {key}.lora_architecture says "
             f"rank {la_rank!r} / {la_mods!r} — one declared pair, two sources that disagree; fix the table or the config; nothing submitted")
print(f"{set_id} {rank}")
PY
}
probe_reopen_token(){   # -> the reopen token (the revocation receipt's NONCE) on stdout; empty when the record is not complete
  # THE ONE ENTRY POINT that reopens a COMPLETE record: lib/slurm.sh's slurm_revoke_geometry, under the per-model lock, archives the
  # site copy and EVERY configured cluster's staged overlay with a REVOKED marker (an unreachable cluster refuses the reopen, fail
  # closed), then writes the revocation receipt INTO the record on this cluster ($CL_ROOT/reopen_receipts/<generation>.<nonce>.json)
  # and returns the nonce — the token the carrier verifies against the receipt and consumes. The generation itself is never the token
  # (a typed digest reopens nothing). Progress goes to stderr; stdout is the nonce alone.
  local complete gen
  complete="$(json_field "${PINFO:-{\}}" complete)"; gen="$(json_field "${PINFO:-{\}}" generation)"
  [ "$complete" = "true" ] || { printf ''; return 0; }
  [ -n "$gen" ] || { echo "probe: the record is complete but its status carries no result generation — cannot mint a reopen receipt; nothing submitted" >&2; return 1; }
  command -v slurm_revoke_geometry >/dev/null 2>&1 || { echo "probe: reproduce/lib/slurm.sh is not loaded — a complete record cannot be reopened without its revocation; nothing submitted" >&2; return 1; }
  SLURM_REVOKE_NONCE=""
  slurm_revoke_geometry "$MODEL" "$gen" "reopen of $CL_ROOT (03_hpo.sh --phase probe, $(date -u +%FT%TZ))${PROBE_CAP_NOTE:+; $PROBE_CAP_NOTE}" "$CLUSTER" "$CL_ROOT" 1>&2 || return 1
  if [ "${REPRO_DRY_RUN:-0}" = 1 ]; then
    # a dry run revokes nothing and writes no receipt (slurm_revoke_geometry said what it would do): describe the token instead
    echo "DRY-RUN: the revocation receipt $CL_ROOT/reopen_receipts/${gen:0:16}….<nonce>.json would be written into the record and its 32-hex nonce ride the carrier's positional 10 (shown as zeros)" >&2
    printf '%032d' 0; return 0
  fi
  [ -n "$SLURM_REVOKE_NONCE" ] || { echo "probe: the revocation wrote no receipt into $CL_ROOT — no reopen token; nothing submitted" >&2; return 1; }
  printf '%s' "$SLURM_REVOKE_NONCE"
}
probe_submit_reverdict(){   # <resume-interrupted | parity-record=<dir>> <receipt nonce | ''> <why>: the carrier in phase=reverdict on THIS cluster's record —
  # the CPU-only certifying rewrite under the serving venv on a compute node (never over ssh on a login host; the venv lives on node-local
  # disk): one GPU, tp1, the plan's sets / ranks, the record's attempt counter as it stands (the carrier writes no attempt record in this
  # phase), a fresh logical unit hpo.lora.probe.rv<utc> so the funnel's idempotence never skips it; SERVING_PROBE_OUT points it at the record
  local plan att pos sets ranks a m token="$2" stamp
  plan="$(cluster_cat "$CL_ROOT/probe_plan.json")" && [ -n "$plan" ] || repro_fatal "probe: $CL_ROOT holds no probe_plan.json on $CLUSTER — nothing to re-verdict, nothing submitted"
  pos="$(printf '%s' "$plan" | "$PY" -c 'import json,sys; d=json.load(sys.stdin); s=d.get("sets") or []; r=d.get("ranks") or []
print(",".join(map(str, s)) if isinstance(s, list) else str(s), ",".join(str(int(x)) for x in r) if isinstance(r, list) else str(r))')" \
    || repro_fatal "probe: the plan under $CL_ROOT could not be read for its sets / ranks — nothing submitted"
  read -r sets ranks <<<"$pos"
  [ -n "$sets" ] && [ -n "$ranks" ] || repro_fatal "probe: the plan under $CL_ROOT names no sets / ranks ($pos) — nothing submitted"
  att="$(cluster_cat "$CL_ROOT/probe_attempt.json")" || att=""
  read -r a m <<<"$(printf '%s' "$att" | "$PY" -c 'import json,sys; raw=sys.stdin.read(); d=json.loads(raw) if raw.strip() else {}; print(int(d.get("attempt") or 1), int(d.get("max_attempts") or 1))')"
  stamp="$(date -u +%Y%m%dT%H%M%SZ)"
  repro_say "probe: $3 — the carrier goes out in phase=reverdict ($1${token:+, receipt $token}) on $CLUSTER for $CL_ROOT: one GPU, tp1, sets $sets, ranks $ranks; the record's certification is rewritten there, never over ssh"
  submit "$(unit probe).rv$stamp" "$(name probe)" scripts/sbatch/serving_probe.sbatch "$MODEL $SRV_REPO $SRV_REV $sets $ranks tp1 $a $m 1 ${token:--} reverdict $1" "$SISTEER_TIME_PROBE" --kind other --flags "$(probe_width_flags 1)" --env "SERVING_PROBE_OUT=$CL_ROOT"
}
probe_parity_record_reopen(){   # --phase probe --reopen --reference-only --parity-record-reopen: the PARITY-PASS record's HF+PEFT judgements re-made
  # The certificates' serving venv / engine identity come from the per-adapter judgements of the reference phase, and assertion 4 is read
  # from the parity-pass record — a parity record judged before that requirement existed carries none, and a re-verdict over its stored
  # records cannot add it (820880, 2026-10-01). This verb reopens THAT record (named by the probe record's plan, parity_record_dir): it
  # must be READY (complete, its own verdict + menu at its current generation) with no job on it and no reopen in progress; the model's
  # staged overlays are revoked and the receipt minted INTO the parity record; the carrier goes out in phase=reference (1 GPU, no engine)
  # on the parity record's own unit / name, under its own attempt counter (--probe-max-attempts widens it, recorded in cap_history and the
  # receipt). The probe record then reads PARITY_STALE and --reopen routes the receipted re-verdict against the re-judged pass.
  local pdir pname pline PST PDT PIN
  pdir="$(json_field "$(cluster_cat "$CL_ROOT/probe_plan.json")" parity_record_dir)"
  [ -n "$pdir" ] || repro_fatal "probe: --parity-record-reopen: the probe record $CL_ROOT remembers no parity-pass record (probe_plan.json parity_record_dir) — nothing to re-judge; nothing revoked, nothing submitted"
  pname="$(name probe)-parity"
  if cluster_job_active "$pname"; then
    repro_fatal "probe: --parity-record-reopen: job $pname is queued/running on $CLUSTER — the parity record $pdir is held; nothing revoked, nothing submitted"
  fi
  pline="$(probe_status 0 "$pdir")" || exit $?
  IFS=$'\t' read -r PST PDT PIN <<<"$pline"
  case "$PST" in
    READY) ;;
    VERDICT_REFUSED)
      # a reference phase that FAILED (an HF+PEFT crash, peft missing) leaves the parity record complete with
      # reference_check.json status=error and an uncertified verdict (no menu) — VERDICT_REFUSED — and every dependent certificate stranded
      # after the overlays were revoked. The fenced recovery is the same re-run of its reference phase: a fresh receipt into THIS record,
      # its own attempt counter (--probe-max-attempts widens it), nothing resumed on the probe record. Only an UNCERTIFIED verdict qualifies.
      [ "$(json_field "$PIN" verdict_status)" = uncertified ] \
        || repro_fatal "probe: --parity-record-reopen: the parity record $pdir is VERDICT_REFUSED with verdict status '$(json_field "$PIN" verdict_status)' — only an uncertified verdict (a failed or evidence-less reference phase) is re-judged here; read $pdir/probe_verdict.json; nothing revoked, nothing submitted"
      # `uncertified` means SOME pair lacks reference evidence; a record that ALSO excludes a pair for a serving
      # reason (a failed / refused / unrun engine launch — the verdict's own exclusions, hpo_probe_status.serving_excluded) is MIXED: a
      # reference-only re-run processes passed / failed launches alone and can never re-run the engine, so it would spend the record's
      # attempt and leave that pair's certificate stuck. Refused by name; the remedy that re-runs launches is named.
      local psxn; psxn="$(json_field "$PIN" serving_excluded_n)"
      if [ "${psxn:-0}" -gt 0 ] 2>/dev/null; then
        repro_fatal "probe: --parity-record-reopen --reference-only: the parity record $pdir is uncertified AND excludes $psxn pair(s) for SERVING reasons — $(json_field "$PIN" serving_excluded) — a reference-only re-run cannot re-run an engine launch, so it would spend the record's attempt and leave those pairs stuck; re-run its launches instead (the launch-capable path: \`--phase probe --reopen --parity-record-reopen --certify-v2 all\` — the D654 fallback, to be built when needed — or a targeted re-run of the failed launches through the parity carrier); nothing revoked, nothing submitted"
      fi
      repro_say "probe: --parity-record-reopen: the parity record $pdir is VERDICT_REFUSED (uncertified — its reference phase failed or produced no evidence; unreferenced $(json_field "$PIN" unreferenced_pairs); no serving-reason exclusion): the fenced recovery re-runs its reference phase under a fresh receipt"
      ;;
    REOPEN_IN_PROGRESS) repro_fatal "probe: --parity-record-reopen: a reopen of the parity record $pdir is IN PROGRESS ($PDT) — recover it first; never a second revocation over it; nothing revoked, nothing submitted";;
    *) repro_fatal "probe: --parity-record-reopen: the parity record $pdir is $PST — $PDT; it must be READY (complete, its own verdict + menu at its current generation, no job) or VERDICT_REFUSED with an uncertified verdict before its judgements are re-made; nothing revoked, nothing submitted";;
  esac
  [ "$(json_field "$PIN" complete)" = true ] || repro_fatal "probe: --parity-record-reopen: the parity record $pdir reads READY but complete=$(json_field "$PIN" complete) — not at rest; nothing revoked, nothing submitted"
  # at rest = no driver's attempt / hold on its checkpoint (READY says its artefacts are current; a resumed or reference-only run in flight
  # marks the checkpoint and would race the re-judgement)
  [ -z "$(json_field "$PIN" attempt)" ] && [ -z "$(json_field "$PIN" hold)" ] \
    || repro_fatal "probe: --parity-record-reopen: the parity record $pdir is not at rest (attempt=$(json_field "$PIN" attempt), hold=$(json_field "$PIN" hold)) — a run holds its checkpoint; nothing revoked, nothing submitted"
  repro_say "probe: --reopen --parity-record-reopen --reference-only: re-making the HF+PEFT judgements of the parity-pass record $pdir ($PST at $(json_field "$PIN" generation | cut -c1-12)) in the serving venv selected by key — every certificate then carries the venv and engine identity the reference measured (D641); the probe record $CL_ROOT will read PARITY_STALE until its --reopen re-verdict"
  CL_ROOT="$pdir"; PINFO="$PIN"; PSTATE="$PST"; PDETAIL="$PDT"; PROBE_REF_ONLY=1; PROBE_UNIT_SUFFIX=".parity"; PROBE_NAME_SUFFIX="-parity"
  probe_resume "reopen of the parity-pass record (--parity-record-reopen --reference-only): $PDT"
  PSTATE=REOPENED
  [ "$PHASE" = all ] && { repro_say "waiting on the parity record's reference re-run — then --phase probe --reopen re-verdicts the probe record"; exit 0; }
  return 0
}
probe_parity_record_at_rest(){   # <parity record dir> <verb>: the pass's OWN state, read on the cluster (hpo_probe_status.py --record): READY with a complete
  # checkpoint and no driver's attempt / hold, or the verb refuses by name with nothing minted — the shape probe_parity_record_reopen holds
  # a pass to; used when the main record's status did not vouch for the pass (a first binding through --reverdict-only)
  local line st detail info
  line="$(probe_status 0 "$1")" || exit $?
  IFS=$'\t' read -r st detail info <<<"$line"
  case "$st" in
    READY) ;;
    NOT_STARTED) repro_fatal "probe: $2: the parity record $1 does not exist on $CLUSTER (NOT_STARTED) — run the pass first (--phase probe --parity-pass); nothing revoked, nothing submitted";;
    *) repro_fatal "probe: $2: the parity record $1 is $st — $detail; a re-verdict reads a READY pass at rest only (its own verdict + menu at its current generation, complete, no job); nothing revoked, nothing submitted";;
  esac
  [ "$(json_field "$info" complete)" = true ] || repro_fatal "probe: $2: the parity record $1 reads READY but complete=$(json_field "$info" complete) — not at rest; nothing revoked, nothing submitted"
  [ -z "$(json_field "$info" attempt)" ] || [ "$(json_field "$info" attempt)" = null ] && { [ -z "$(json_field "$info" hold)" ] || [ "$(json_field "$info" hold)" = null ]; } \
    || repro_fatal "probe: $2: the parity record $1 is not at rest (attempt=$(json_field "$info" attempt), hold=$(json_field "$info" hold)) — a run holds its checkpoint; nothing revoked, nothing submitted"
  local pplan; pplan="$(cluster_cat "$1/probe_plan.json")" || pplan=""
  [ -z "$pplan" ] || [ "$(json_field "$pplan" parity_pass)" = true ] || repro_fatal "probe: $2: $1 is not a PARITY-PASS record (probe_plan.json parity_pass=$(json_field "$pplan" parity_pass)) — assertion 4 reads a pass written with \`plan --parity-pass\` only; nothing revoked, nothing submitted"
  repro_say "probe: $2: the parity record $1 is READY and at rest ($detail)"
}
probe_parity_pass_verb(){   # --phase probe --parity-pass: the PARITY PASS of the one declared pair on ITS OWN record <hpo root>_parity.
  # The main record's scale-100 synthetic adapters are the ladder's servability subjects: at that scale
  # control's v2 certificate fails by measurement (a top-1 flip within the base discrepancy), which is not a LoRA parity defect — the
  # default flow certifies assertion 4 from a pass at the per-set BENIGN scale (wo 1.0), the natural launches at the ladder window alone
  # (one adapter dir per pair, no rank-cap ladder, no throughput leg), and the main record's receipted `reverdict --parity-record` reads it.
  # The carrier runs phase=parity (positional 11; positionals 6-10 are spelled so 11 is reached: tp1, attempt 1, the carrier's default cap,
  # parallel auto, no token); SERVING_PROBE_OUT points it at the pass's record; one GPU unless --probe-width says otherwise. The same
  # state machine as the probe over THAT record: NOT_STARTED submits (the fresh unit hpo.lora.probe.parity, job name <probe>-parity),
  # a running job waits, a stalled record resumes through probe_resume under the pass's own unit / name in phase=parity, READY names the
  # re-verdict (the binding is --reopen --reverdict-only [--parity-record DIR] on the MAIN record — never made here), VERDICT_REFUSED
  # refuses by name (the pass's own gates). The scale table is the pass's provenance: --parity-scales rides a FRESH submission only.
  local proot="${CL_ROOT}_parity" pname="$(name probe)-parity" punit="$(unit probe).parity" tlim="${PROBE_TIME:-$SISTEER_TIME_PROBE}"
  [ -n "$PROBE_PAIR" ] || repro_fatal "probe: --parity-pass runs the parity pass of the paper protocol's ONE declared pair (the fits config's lora_r4_ce fit_selection); under the FD 6 protocol the search's menu is the carrier's default — its menu-wide pass is the carrier's phase=parity by hand (scripts/sbatch/serving_probe.sbatch positional 11) until the front door carries it; nothing submitted"
  local active=0; cluster_job_active "$pname" && active=1
  local line st detail info; line="$(probe_status "$active" "$proot")" || exit $?
  IFS=$'\t' read -r st detail info <<<"$line"
  case "$st" in
    NOT_STARTED)
      [ -z "$PROBE_PARITY_SCALES" ] || [ "$PROBE_PARITY_SCALES" = default ] || repro_note "probe: --parity-scales $PROBE_PARITY_SCALES overrides the driver's table for this pass (SERVING_PROBE_PARITY_SCALES on the carrier; recorded in the pass's plan as synthetic_scale_by_set)"
      local B_STATE B_JOB B_CLUSTER; read -r B_STATE B_JOB B_CLUSTER <<<"$(probe_fresh_binding "$punit")"
      if [ "$B_STATE" != unbound ] && [ "$active" != 1 ] && [ "$PROBE_RESUBMIT" != 1 ]; then
        repro_fatal "probe: --parity-pass: the parity record $proot is EMPTY (NOT_STARTED) but the unit $punit is already $B_STATE in the ledger to job ${B_JOB:-?} on ${B_CLUSTER:-$CLUSTER}, and no $pname job is active: that job died before writing any record file — read its log $(cluster_var "$CLUSTER" CLUSTER_LOG_DIR)/${pname}_${B_JOB:-<job>}.out, then re-run --phase probe --parity-pass --resubmit (lib/slurm.sh proves the job dead before re-minting); nothing submitted"
      fi
      local RESUB=(); [ "$PROBE_RESUBMIT" = 1 ] && { RESUB=(--resubmit); repro_note "probe: --resubmit forwarded to lib/slurm.sh for $punit (the funnel's proof decides)"; }
      local flags; flags="$(probe_width_flags "${PROBE_WIDTH:-1}")"
      repro_say "probe: --parity-pass: the PARITY PASS of set ${PROBE_PAIR% *} / rank ${PROBE_PAIR#* } on $CLUSTER -> $proot (carrier phase=parity: plan --parity-pass --parity-scales ${PROBE_PARITY_SCALES:-default}, the natural launches at the ladder window, one adapter dir per pair, no throughput leg; width ${PROBE_WIDTH:-1}${flags:+ ($flags)}, time $tlim${PROBE_TIME:+ (--probe-time)}); the main record $CL_ROOT is not touched — bind the pass to it when both are DONE: --phase probe --reopen --reverdict-only"
      submit "$punit" "$pname" scripts/sbatch/serving_probe.sbatch "$MODEL $SRV_REPO $SRV_REV ${PROBE_PAIR% *} ${PROBE_PAIR#* } tp1 1 4 auto - parity" "$tlim" --kind other --flags "$flags" "${RESUB[@]+"${RESUB[@]}"}" --env "SERVING_PROBE_OUT=$proot${PROBE_PARITY_SCALES:+ SERVING_PROBE_PARITY_SCALES=$PROBE_PARITY_SCALES}"
      ;;
    RUNNING|INCOMPLETE_RUNNING)
      repro_note "probe: --parity-pass: $st — $detail; job $pname is queued/running on $CLUSTER — nothing submitted; re-run when its watcher reports DONE";;
    INCOMPLETE_STALLED|MENU_STALE|FAILED_START)
      if [ "$st" = FAILED_START ] && [ "$(json_field "$info" plan_present)" != true ]; then
        repro_fatal "probe: --parity-pass: FAILED_START on $CLUSTER — $detail. No plan exists on $proot, so there is nothing to resume: read the failed attempt's log, move the plan-less record aside by hand (never delete: $proot -> $proot/superseded/<utc>/ of a fresh dir) and re-run --phase probe --parity-pass --resubmit; nothing submitted"
      fi
      [ -z "$PROBE_PARITY_SCALES" ] || repro_fatal "probe: --parity-pass: $proot is $st (a resume) — --parity-scales rides a FRESH pass only: the scale table is the pass's provenance (probe_plan.json synthetic_scale_by_set) and a resume keeps the record's own plan; drop the flag to resume, or move the record aside by hand (never delete) for a fresh pass at other scales; nothing submitted"
      repro_say "probe: --parity-pass: the parity record $proot is $st on $CLUSTER with no job — resuming it under its own unit / name in phase=parity"
      CL_ROOT="$proot"; PINFO="$info"; PSTATE="$st"; PDETAIL="$detail"; PROBE_UNIT_SUFFIX=".parity"; PROBE_NAME_SUFFIX="-parity"; PROBE_RESUME_PHASE=parity
      probe_resume "the parity pass (--parity-pass) on $st: $detail";;
    READY)
      repro_say "probe: --parity-pass: the parity record $proot is READY on $CLUSTER ($detail) — nothing submitted. Next: bind it to the main record's certification with --phase probe --reopen --reverdict-only (the receipted reverdict --parity-record $proot; adapter parity is whatever the pass's launches judge per adapter), then --phase probe fetches the re-verdicted menu";;
    VERDICT_REFUSED)
      repro_fatal "probe: --parity-pass: the parity record $proot is VERDICT_REFUSED on $CLUSTER — $detail (verdict status '$(json_field "$info" verdict_status)', unreferenced $(json_field "$info" unreferenced_pairs)). Read $proot/probe_verdict.json: the pass's own gates — parity_weak (raise the set's scale) / parity_chaotic (lower it) — are answered by a FRESH pass at other scales (move $proot aside by hand, never delete, then --phase probe --parity-pass --parity-scales set=scale,…); a failed reference phase is re-run through the main record once it remembers the pass (--reopen --reference-only --parity-record-reopen, D641); nothing submitted";;
    REOPEN_IN_PROGRESS)
      repro_fatal "probe: --parity-pass: a reopen of the parity record $proot is IN PROGRESS ($detail) — recover it first; nothing submitted";;
    PARITY_STALE)
      repro_fatal "probe: --parity-pass: the parity record $proot reads PARITY_STALE ($detail) — a pass reads no pass of its own; the record is not what this verb wrote; nothing submitted";;
    *) repro_fatal "probe: --parity-pass: hpo_probe_status.py returned an unknown state '$st' for $proot ($detail) — nothing submitted";;
  esac
}
probe_reopen_verb(){   # --phase probe --reopen on a READY (or DONE-and-current) record: archive the local copies + DONE marker, revoke, receipt, resume
  # THE DELIBERATE REOPEN: the only supported way to reopen a record that has a verdict — e.g. a completed false-OOM record — so
  # nobody hand-types a resume. Called right before `case "$PSTATE" in` in the probe branch, after the status read and the DONE
  # revalidation; sets PSTATE=REOPENED so the case does nothing more (--phase all then waits). A job holding the record, or any
  # state but READY / CURRENT, refuses by name.
  if [ "${REPRO_DRY_RUN:-0}" = 1 ]; then
    # a dry run reads no record (PSTATE is NOT_STARTED by construction), so nothing here can be judged: describe the reopen — what
    # slurm_revoke_geometry would withdraw (its own DRY-RUN lines), the receipt, and the resume the live path derives from the record
    repro_say "DRY-RUN: --reopen: the record's state is not read under a dry run; live, a READY (or DONE-and-current) $CL_ROOT would be reopened: the local menu / verdict / geometry copies + the DONE marker archived, every staged overlay of $MODEL revoked, the revocation receipt written into the record, and the record resumed"
    slurm_revoke_geometry "$MODEL" "$(printf '%064d' 0)" "reopen of $CL_ROOT (03_hpo.sh --phase probe --reopen, dry run)" "$CLUSTER" "$CL_ROOT" 1>&2 || return 1
    if [ "${PROBE_REVERDICT_ONLY:-0}" = 1 ]; then
      echo "DRY-RUN: --reverdict-only: the revocation receipt $CL_ROOT/reopen_receipts/<generation>.<nonce>.json would be written into the record and the carrier submitted in phase=reverdict with positional 12 parity-record=${PROBE_PARITY_RECORD:-<the remembered pass, else ${CL_ROOT}_parity when READY>} and the receipt's 32-hex nonce as positional 10 (shown as $(printf '%032d' 0)): one GPU, no launches, no attempt consumed; a live run reads the main record's state (READY, PARITY_STALE, or an uncertified complete record) and holds the pass to AT REST first" >&2
    else
      echo "DRY-RUN: the revocation receipt $CL_ROOT/reopen_receipts/<generation>.<nonce>.json would be written into the record; probe_resume would then submit scripts/sbatch/serving_probe.sbatch for the NEXT attempt of the SAME record (attempt counter and cap from probe_attempt.json, the record's width and sets/ranks/mode from probe_plan.json) with the receipt's 32-hex nonce as positional 10 (shown as $(printf '%032d' 0))${PROBE_REF_ONLY:+ and positional 11 'reference' (--reference-only: the HF+PEFT reference phase alone, width 1, D146; on an uncertified complete record the re-run of a harness-filed reference status)}; a live run derives every value from the record" >&2
    fi
    PSTATE=REOPENED
    [ "$PHASE" = all ] && { repro_say "waiting on the reopened probe — re-run --phase all when it is DONE"; exit 0; }
    return 0
  fi
  if [ "${PROBE_ACTIVE:-0}" = 1 ]; then
    repro_fatal "probe: --reopen: job $(name probe) is queued/running on $CLUSTER — the record $CL_ROOT is held (its state reads $PSTATE); nothing revoked, nothing submitted; re-run --reopen when the watcher reports the job DONE"
  fi
  if [ "${PROBE_PARITY_REOPEN:-0}" = 1 ]; then probe_parity_record_reopen; return 0; fi
  if [ "${PROBE_REVERDICT_ONLY:-0}" = 1 ]; then
    # a READY record whose certificates must be RE-FOLDED under a fixed writer (the serving side's
    # identity, the two-sided block) — the receipted re-verdict against the remembered parity record, the PARITY_STALE route taken on
    # purpose: no resume (a READY record's --reopen resumes, which a record at its attempt cap cannot), no launches, no attempt consumed.
    # The parity record must be at rest and no reopen of this record in progress; anything else refuses by name.
    case "$PSTATE" in
      READY|CURRENT|VERDICT_REFUSED|PARITY_STALE)
        # VERDICT_REFUSED is the first binding of a pass: a complete record whose own references left the pair
        # uncertified (reference_missing, or a scale-100 mismatch judged on the main record) reads assertion 4 from the pass once it is
        # bound — the receipted re-verdict, no launches, no attempt; the pass is held to AT REST below like every other route
        [ "$PSTATE" != VERDICT_REFUSED ] || [ "$(json_field "${PINFO:-{\}}" complete)" = true ] || repro_fatal "probe: --reverdict-only on VERDICT_REFUSED: $CL_ROOT reads complete=$(json_field "${PINFO:-{\}}" complete) — a re-verdict rewrites a COMPLETE record only; nothing revoked, nothing submitted";;
      REOPEN_IN_PROGRESS) repro_fatal "probe: --reverdict-only: a reopen of $CL_ROOT is IN PROGRESS ($PDETAIL) — recover it first (--reopen without --reverdict-only routes `reverdict --resume-interrupted`); nothing revoked, nothing submitted";;
      *) repro_fatal "probe: --reverdict-only needs a READY, PARITY_STALE or VERDICT_REFUSED record; $CL_ROOT is $PSTATE — $PDETAIL; nothing revoked, nothing submitted";;
    esac
    # THE PASS THE RE-VERDICT READS, in this order: --parity-record (an operator's binding: the first, or another pass), the verdict's
    # stamp (the pass the standing verdict read), probe_plan.json's parity_record_dir (the pass a reopen remembered), else the stage's
    # OWN parity record <hpo root>_parity (--parity-pass wrote it there) when it is READY — never a guess beyond the stage's own layout
    local rprec rpdir rsrc; rprec="$(json_field "${PINFO:-{\}}" parity_record)"; rpdir="$(json_field "$rprec" dir)"; rsrc="the verdict's stamp"
    if [ -n "$PROBE_PARITY_RECORD" ]; then
      [ -z "$rpdir" ] || [ "$rpdir" = "$PROBE_PARITY_RECORD" ] || repro_say "probe: --parity-record $PROBE_PARITY_RECORD replaces the pass the verdict read ($rpdir) for this re-verdict — the record remembers the new one afterwards (probe_plan.json parity_record_dir)"
      [ "$rpdir" = "$PROBE_PARITY_RECORD" ] || rprec=""
      rpdir="$PROBE_PARITY_RECORD"; rsrc="--parity-record"
    fi
    [ -n "$rpdir" ] || { rpdir="$(json_field "$(cluster_cat "$CL_ROOT/probe_plan.json")" parity_record_dir)"; rsrc="probe_plan.json parity_record_dir"; }
    if [ -z "$rpdir" ]; then
      local own="${CL_ROOT}_parity" oline ost odetail oinfo
      oline="$(probe_status 0 "$own")" || exit $?
      IFS=$'\t' read -r ost odetail oinfo <<<"$oline"
      case "$ost" in
        READY) rpdir="$own"; rsrc="the stage's own parity-pass record (--parity-pass)"; repro_say "probe: --reverdict-only: $CL_ROOT remembers no parity-pass record; the stage's own pass $own is READY ($odetail) — binding it (plan.parity_record_dir remembers it afterwards)";;
        NOT_STARTED) repro_fatal "probe: --reverdict-only: $CL_ROOT remembers no parity-pass record (neither its verdict nor its plan names one) and the stage's own pass $own does not exist on $CLUSTER — run the pass first (--phase probe --parity-pass), or name another with --parity-record DIR; nothing revoked, nothing submitted";;
        *) repro_fatal "probe: --reverdict-only: $CL_ROOT remembers no parity-pass record and the stage's own pass $own is $ost — $odetail; a re-verdict reads a READY pass at rest only (re-run --phase probe --parity-pass to resume it, or read $own/probe_verdict.json); nothing revoked, nothing submitted";;
      esac
    fi
    if [ -n "$rprec" ] && [ "$rprec" != null ]; then
      [ "$(json_field "$rprec" at_rest)" = true ] || repro_fatal "probe: --reverdict-only: the parity record $rpdir is not AT REST (at_rest=$(json_field "$rprec" at_rest): complete=$(json_field "$rprec" complete), attempt=$(json_field "$rprec" attempt), hold=$(json_field "$rprec" hold), checkpoint_generation_matches_live=$(json_field "$rprec" checkpoint_generation_matches_live), own_verdict_present=$(json_field "$rprec" own_verdict_present), own_reference_check_present=$(json_field "$rprec" own_reference_check_present)) — the re-verdict would read a moving or headless pass; nothing archived, nothing revoked, nothing submitted"
    else
      probe_parity_record_at_rest "$rpdir" "--reverdict-only"      # a pass the status did not vouch for (a first binding): its own state, read on the cluster
    fi
    repro_say "probe: --reopen --reverdict-only: re-folding the certification of $CL_ROOT ($PSTATE) against the parity record $rpdir ($rsrc) — revoking every staged overlay of $MODEL, minting the receipt, routing reverdict --parity-record on $CLUSTER (no launches, no attempt consumed; adapter parity is whatever the pass's launches judge per adapter — never verified by the binding itself)"
    probe_archive_local probe "$PSTATE" "reopen for the re-verdict (--reopen --reverdict-only)" "${PINFO:-{\}}" deliberate
    local rtoken; rtoken="$(probe_reopen_token)" || repro_fatal "probe: the reopen of $CL_ROOT was REFUSED (its reason is above) — nothing submitted"
    [ -n "$rtoken" ] || repro_fatal "probe: --reverdict-only: no receipt was minted for $CL_ROOT (its status reads complete=$(json_field "${PINFO:-{\}}" complete)) — nothing submitted"
    probe_submit_reverdict "parity-record=$rpdir" "$rtoken" "the re-verdict of $CL_ROOT (--reverdict-only)"
    PSTATE=REOPENED
    [ "$PHASE" = all ] && { repro_say "waiting on the reverdict — re-run --phase all when it is DONE"; exit 0; }
    return 0
  fi
  case "$PSTATE" in
    READY|CURRENT)
      repro_say "probe: --reopen: reopening the READY record $CL_ROOT on $CLUSTER ($([ "${PROBE_REF_ONLY:-0}" = 1 ] && echo "the HF+PEFT reference phase alone, 1 GPU — D146" || echo "re-plan + re-run under the current estimator"); the menu, verdict and geometry are withdrawn first)"
      probe_archive_local probe "$PSTATE" "reopen requested (--reopen)" "${PINFO:-{\}}" deliberate
      probe_resume "reopen requested (--reopen): $PDETAIL"
      PSTATE=REOPENED
      [ "$PHASE" = all ] && { repro_say "waiting on the reopened probe — re-run --phase all when it is DONE"; exit 0; }
      return 0;;
    VERDICT_REFUSED)
      # the verdict REFUSED certification for want of reference evidence (`uncertified`, unreferenced pairs) on a COMPLETE record
      # whose remembered parity record is AT REST (seen 2026-10-01, job 820268) — the remedy is the receipted re-verdict against that
      # record (revoke -> receipt -> `reverdict --parity-record <dir>`, the CPU-only carrier phase that writes no attempt record and
      # consumes no attempt), the same route as PARITY_STALE; a verdict refused for any other reason is not this remedy and refuses by name
      local vst unref prec pdir; vst="$(json_field "${PINFO:-{\}}" verdict_status)"; unref="$(json_field "${PINFO:-{\}}" unreferenced_pairs)"
      if [ "${PROBE_REF_ONLY:-0}" = 1 ]; then
        # the RE-RUN of a harness-filed reference status (probe 979954): the identity guard filed the pair's
        # adapters `reference_missing` on a complete record; that status is STICKY (the re-gate keeps a harness-filed status), so a fixed
        # guard is read only by judging every adapter again — the HF+PEFT reference phase alone, one GPU, the launches' servability as
        # measured; the record is complete, so the receipt is minted and its nonce rides positional 10, positional 11 `reference`
        [ "$(json_field "${PINFO:-{\}}" complete)" = true ] || repro_fatal "probe: --reopen --reference-only on VERDICT_REFUSED: $CL_ROOT reads complete=$(json_field "${PINFO:-{\}}" complete) — the reference re-run needs a COMPLETE record (resume its launches first); nothing revoked, nothing submitted"
        repro_say "probe: --reopen --reference-only: VERDICT_REFUSED — the verdict is '${vst:-?}' (unreferenced ${unref:-none}); re-judging every adapter of $CL_ROOT with the HF+PEFT reference alone (1 GPU, no engine; D146 — the re-run of a harness-filed reference status): the launches' servability stands as measured; a judged mismatch stays a mismatch (a parity pass certifies assertion 4 at the benign scale: --phase probe --parity-pass, then --reopen --reverdict-only)"
        probe_archive_local probe "$PSTATE" "reopen for the reference re-run (--reopen --reference-only on VERDICT_REFUSED)" "${PINFO:-{\}}" deliberate
        probe_resume "reopen requested (--reopen --reference-only) on VERDICT_REFUSED: $PDETAIL"
        PSTATE=REOPENED
        [ "$PHASE" = all ] && { repro_say "waiting on the reference re-run — re-run --phase all when it is DONE"; exit 0; }
        return 0
      fi
      [ "$vst" = uncertified ] && [ -n "$unref" ] && [ "$unref" != "[]" ] \
        || repro_fatal "probe: --reopen on VERDICT_REFUSED: the verdict's status is '${vst:-?}' with unreferenced pairs '${unref:-none}' — the reference re-verdict remedies an uncertified verdict with pairs lacking reference evidence; read $CL_ROOT/probe_verdict.json; nothing revoked, nothing submitted"
      prec="$(json_field "${PINFO:-{\}}" parity_record)"; pdir="$(json_field "$prec" dir)"
      [ -n "$pdir" ] || repro_fatal "probe: --reopen on VERDICT_REFUSED: the verdict read no parity record (unreferenced $unref) — a re-verdict has nothing to re-read; the pairs need a parity pass; nothing revoked, nothing submitted"
      [ "$(json_field "$prec" at_rest)" = true ] || repro_fatal "probe: --reopen on VERDICT_REFUSED: the parity record $pdir is not AT REST (at_rest=$(json_field "$prec" at_rest): complete=$(json_field "$prec" complete), attempt=$(json_field "$prec" attempt), hold=$(json_field "$prec" hold), checkpoint_generation_matches_live=$(json_field "$prec" checkpoint_generation_matches_live), own_verdict_present=$(json_field "$prec" own_verdict_present), own_reference_check_present=$(json_field "$prec" own_reference_check_present)) — the re-verdict would read a moving or headless pass; nothing archived, nothing revoked, nothing submitted"
      repro_say "probe: --reopen: VERDICT_REFUSED — the verdict is uncertified for want of reference evidence on $unref; revoking every staged overlay of $MODEL, minting the receipt into $CL_ROOT, and routing reverdict --parity-record $pdir on $CLUSTER (D623: the parity record's judgements are read per adapter across its launches)"
      probe_archive_local probe "$PSTATE" "reopen for the reference re-verdict (--reopen on VERDICT_REFUSED)" "${PINFO:-{\}}" deliberate
      local token; token="$(probe_reopen_token)" || repro_fatal "probe: the reopen of $CL_ROOT was REFUSED (its reason is above) — nothing submitted"
      [ -n "$token" ] || repro_fatal "probe: --reopen on VERDICT_REFUSED: no receipt was minted for $CL_ROOT (its status reads complete=$(json_field "${PINFO:-{\}}" complete)) — nothing submitted"
      probe_submit_reverdict "parity-record=$pdir" "$token" "the reference re-verdict of $CL_ROOT (unreferenced $unref)"
      PSTATE=REOPENED
      [ "$PHASE" = all ] && { repro_say "waiting on the reverdict — re-run --phase all when it is DONE"; exit 0; }
      return 0;;
    INCOMPLETE_STALLED|MENU_STALE)
      # a record with launches outstanding and no job (a driver stopped by its time budget at the attempt cap — PARTIAL) is resumed
      # under --reopen like the plain probe phase does: attempt+1 through the funnel (with --probe-max-attempts when the cap is reached), the
      # same width / time / certificate flags. An INCOMPLETE record mints no receipt; a MENU_STALE record whose checkpoint is COMPLETE (a
      # menu stamped for another result generation) is complete, so probe_resume mints the receipt and the nonce rides positional 10
      local rc_complete; rc_complete="$(json_field "${PINFO:-{\}}" complete)"
      repro_say "probe: --reopen: $CL_ROOT is $PSTATE — $([ "$rc_complete" = true ] && echo "a stale menu on a COMPLETE record, no job: resuming it (the record is complete: a revocation receipt is minted and its nonce rides positional 10)" || echo "launches outstanding, no job: resuming it (no receipt: the record is not complete)")"
      probe_resume "reopen requested (--reopen) on $PSTATE: $PDETAIL"
      PSTATE=REOPENED
      [ "$PHASE" = all ] && { repro_say "waiting on the reopened probe — re-run --phase all when it is DONE"; exit 0; }
      return 0;;
    REOPEN_IN_PROGRESS)
      # the probe phase's contract: a consumed receipt is IN PROGRESS until the certifying rewrite completes; a reverdict
      # that died after archiving the derived artifacts leaves an in-progress receipt beside a COMPLETE checkpoint, and hpo_probe_status
      # names the record REOPEN_IN_PROGRESS (its detail carries the recovery, info.interrupted_reopen the consumer). The recovery is
      # `reverdict --resume-interrupted` under THAT receipt, on this cluster — routed through the carrier; never a fresh mint (a second
      # revocation would strand the record, and slurm_revoke_geometry refuses it anyway). A record whose checkpoint is NOT complete stays
      # INCOMPLETE_* and its ordinary resume claims and finishes the receipt — no reverdict phase.
      local ir who=""; ir="$(json_field "${PINFO:-{\}}" interrupted_reopen)"
      if [ -n "$ir" ] && [ "$ir" != null ]; then who=" — $(json_field "$ir" what) consumed the receipt at $(json_field "$ir" started_utc) (pid $(json_field "$ir" pid) on $(json_field "$ir" host), job $(json_field "$ir" job_id)$([ "$(json_field "$ir" resumed)" = 0 ] || [ -z "$(json_field "$ir" resumed)" ] || echo ", resumed $(json_field "$ir" resumed)x"))"; fi
      repro_say "probe: --reopen: a reopen of $CL_ROOT is IN PROGRESS / INTERRUPTED$who; routing its recovery: reverdict --resume-interrupted under that receipt — never a fresh revocation"
      probe_archive_local probe "$PSTATE" "a reopen is in progress — the local copies predate its rewrite (--reopen)" "${PINFO:-{\}}" deliberate
      probe_submit_reverdict resume-interrupted "" "the interrupted reopen of $CL_ROOT"
      PSTATE=REOPENED
      [ "$PHASE" = all ] && { repro_say "waiting on the reverdict — re-run --phase all when it is DONE"; exit 0; }
      return 0;;
    PARITY_STALE)
      # the parity-pass record the verdict read moved / was re-planned; the remedy is the receipted re-verdict — revoke every staged
      # overlay, mint the receipt, and route `reverdict --parity-record <dir>` through the carrier; the parity record must be complete and at rest
      local prec pdir; prec="$(json_field "${PINFO:-{\}}" parity_record)"; pdir="$(json_field "$prec" dir)"
      [ -n "$pdir" ] || repro_fatal "probe: --reopen on PARITY_STALE: the status names no parity record dir ($PDETAIL) — nothing revoked, nothing submitted"
      # `complete` is written on the last RESULT before the reference / verdict phases — BOTH remedies (the parity re-verdict, and the
      # resume whose own verdict re-reads the parity record) need the status's AT-REST fact (complete, no attempt / hold, checkpoint
      # generation == live, the parity record's OWN verdict and reference artefacts present; hpo_probe_status.verify_parity_record)
      # BEFORE anything is archived or revoked — a pass still writing or re-planning is never read as evidence by either route
      [ "$(json_field "$prec" at_rest)" = true ] || repro_fatal "probe: --reopen on PARITY_STALE: the parity record $pdir is not AT REST (at_rest=$(json_field "$prec" at_rest): complete=$(json_field "$prec" complete), attempt=$(json_field "$prec" attempt), hold=$(json_field "$prec" hold), checkpoint_generation_matches_live=$(json_field "$prec" checkpoint_generation_matches_live), own_verdict_present=$(json_field "$prec" own_verdict_present), own_reference_check_present=$(json_field "$prec" own_reference_check_present)) — the re-verdict, or the resume's own verdict, would read a moving or headless pass; nothing archived, nothing revoked, nothing submitted"
      # a PARITY_STALE record WITH launches to run — pairs the menu excludes for a SERVING reason (a start-time OOM, a
      # KV shortfall, an unrun ladder role) or leaves undecided (hpo_probe_status.serving_excluded) — RESUMES on --reopen: the resume's
      # own verdict re-reads the parity record at its current generation, so the parity re-verdict alone would only cost a second round
      # (the main record read PARITY_STALE after 819493 moved the parity record's generation while 32 extension launches waited)
      local sxn; sxn="$(json_field "${PINFO:-{\}}" serving_excluded_n)"
      if [ "${sxn:-0}" -gt 0 ] 2>/dev/null && [ "${PROBE_REF_ONLY:-0}" != 1 ]; then
        repro_say "probe: --reopen: PARITY_STALE ($(json_field "${PINFO:-{\}}" parity_why)) AND $sxn pair(s) excluded for serving reasons — $(json_field "${PINFO:-{\}}" serving_excluded) — launches to run: resuming the record (its verdict re-reads the parity record $pdir at its current generation; D548), not the parity re-verdict alone"
        probe_archive_local probe "$PSTATE" "reopen with launches to run (--reopen on PARITY_STALE with serving exclusions)" "${PINFO:-{\}}" deliberate
        probe_resume "reopen requested (--reopen) on PARITY_STALE with $sxn serving exclusion(s): $PDETAIL"
        PSTATE=REOPENED
        [ "$PHASE" = all ] && { repro_say "waiting on the reopened probe — re-run --phase all when it is DONE"; exit 0; }
        return 0
      fi
      repro_say "probe: --reopen: PARITY_STALE — $(json_field "${PINFO:-{\}}" parity_why); revoking every staged overlay of $MODEL, minting the receipt into $CL_ROOT, and routing reverdict --parity-record $pdir on $CLUSTER"
      probe_archive_local probe "$PSTATE" "reopen for the parity re-verdict (--reopen)" "${PINFO:-{\}}" deliberate
      local token; token="$(probe_reopen_token)" || repro_fatal "probe: the reopen of $CL_ROOT was REFUSED (its reason is above) — nothing submitted"
      [ -n "$token" ] || repro_fatal "probe: --reopen on PARITY_STALE: no receipt was minted for $CL_ROOT (its status reads complete=$(json_field "${PINFO:-{\}}" complete)) — nothing submitted"
      probe_submit_reverdict "parity-record=$pdir" "$token" "the parity re-verdict of $CL_ROOT"
      PSTATE=REOPENED
      [ "$PHASE" = all ] && { repro_say "waiting on the reverdict — re-run --phase all when it is DONE"; exit 0; }
      return 0;;
    *) repro_fatal "probe: --reopen needs a READY record (or a DONE one whose menu is current); $CL_ROOT is $PSTATE — $PDETAIL; nothing revoked, nothing submitted";;
  esac
}
probe_resume(){   # <detail>: INCOMPLETE_STALLED -> the NEXT attempt of the SAME record through lib/slurm.sh (ledger line + watcher)
  # Never a fresh probe over a partial record. The record's probe_plan.json supplies positionals 4-6 (the driver refuses a
  # disagreement on a resume anyway); the carrier's probe_attempt.json — else its per-exit RESULT head in the figure dir, the
  # form before that record existed — supplies the attempt counter and the cap; --probe-width the width (default: the record's
  # last); hpo_probe_resume.py derives them (never re-typed) and refuses at the cap. The attempt-scoped unit hpo.lora.probe.a<N>
  # is a NEW logical work id, so the funnel's idempotence (a unit already submitted is skipped — what stranded attempt 2)
  # never skips a resume, and an attempt number the funnel already issued is skipped by the derivation. Everything else — the
  # cluster block's qos/partition/time on the command line, --comment=sha: of THIS checkout (the carrier refuses a HEAD mismatch),
  # env as VAR=value prefixes, never --export=, never --dependency — is lib/slurm.sh's one line.
  local plan att legacy spec key val
  local R_ATTEMPT="" R_MAX="" R_SETS="" R_RANKS="" R_MODE="" R_WIDTH="" R_PARALLEL="" R_SOURCE=""
  repro_say "probe: INCOMPLETE on $CLUSTER with no job — $1"
  plan="$(cluster_cat "$CL_ROOT/probe_plan.json")" && [ -n "$plan" ] || repro_fatal "probe: $CL_ROOT holds no probe_plan.json on $CLUSTER — nothing to resume"
  att="$(cluster_cat "$CL_ROOT/probe_attempt.json")" || att=""
  legacy=""; [ -n "$att" ] || legacy="$(cluster_cat "$SHARED_ROOT/exp/figs/serving-probe/$(basename "$CL_ROOT")/RESULT" | head -1)"
  # --probe-max-attempts is an OPERATOR cap widen, derived (and refused below the next attempt) by hpo_probe_resume.py — never a
  # by-hand positional 8; the widen rides to the carrier as SERVING_PROBE_CAP_WIDEN (recorded in probe_attempt.json) and into the reopen
  # receipt's reason (PROBE_CAP_NOTE, read by probe_reopen_token below) when the record is complete and one is minted
  local MA=(); [ -z "$PROBE_MAX_ATTEMPTS" ] || MA=(--max-attempts "$PROBE_MAX_ATTEMPTS" --cap-reason "$PROBE_CAP_REASON")
  local punit="$(unit probe)${PROBE_UNIT_SUFFIX:-}" pname="$(name probe)${PROBE_NAME_SUFFIX:-}"      # The parity record's own unit / job name
  spec="$(printf '%s' "$plan" | "$PY" "$REPRO_REPO/scripts/sdfbase/hpo_probe_resume.py" --plan - --attempt-json "$att" --legacy-result "$legacy" \
            --width "${PROBE_WIDTH:-record}" --ledger "$(repro_state_dir "$MODEL")/submissions.jsonl" --lwid-prefix "$(lwid "$punit")" ${MA[@]+"${MA[@]}"})" \
    || repro_fatal "probe: the resume of $CL_ROOT cannot be derived from the record (its reason is above); NOT submitting a fresh probe over the partial record"
  local R_CAP_WIDEN=""
  while IFS='=' read -r key val; do
    case "$key" in
      attempt) R_ATTEMPT="$val";; max_attempts) R_MAX="$val";; sets) R_SETS="$val";; ranks) R_RANKS="$val";; mode) R_MODE="$val";;
      width) R_WIDTH="$val";; parallel) R_PARALLEL="$val";; source) R_SOURCE="$val";; cap_widen) R_CAP_WIDEN="$val";; note) repro_note "probe: $val";;
    esac
  done <<<"$spec"
  PROBE_CAP_NOTE=""
  if [ -n "$R_CAP_WIDEN" ]; then
    # one line, no quotes or backslashes: it is written into JSON by the carrier's printf and into the receipt's reason by the revoke
    PROBE_CAP_NOTE="$(printf '%s' "cap widened $R_CAP_WIDEN by the operator (03_hpo.sh --probe-max-attempts $PROBE_MAX_ATTEMPTS, $(date -u +%FT%TZ), $(hostname)): ${PROBE_CAP_REASON:-no reason given}" | tr -d "\"\\\\'" | tr '\n' ' ')"
    repro_say "probe: $PROBE_CAP_NOTE"
  fi
  [ -n "$R_ATTEMPT" ] && [ -n "$R_MAX" ] && [ -n "$R_SETS" ] && [ -n "$R_RANKS" ] && [ -n "$R_MODE" ] && [ -n "$R_WIDTH" ] && [ -n "$R_PARALLEL" ] \
    || repro_fatal "probe: hpo_probe_resume.py printed an incomplete resume spec ($spec)"
  # --reference-only: the HF+PEFT reference phase alone, on ONE GPU — the width is 1 whatever the record's last was
  local phase=""; [ "${PROBE_REF_ONLY:-0}" = 1 ] && { phase=reference; R_WIDTH=1; R_PARALLEL=1; }
  # a PARITY-PASS record's resume (probe_parity_pass_verb) keeps the carrier in phase=parity (positional 11): the plan flags of the
  # pass — `--parity-pass --parity-scales` — ride the resume as they rode the fresh submission (the driver keeps the record's own plan)
  [ "${PROBE_RESUME_PHASE:-}" != parity ] || phase=parity
  local flags; flags="$(probe_width_flags "$R_WIDTH")"
  # revoke-before-reopen (re-check 2): a COMPLETE record is reopened only with the token minted after every staged overlay is
  # revoked — positional 10 of the carrier; an incomplete record (the usual stalled resume) needs none and gets none
  local token; token="$(probe_reopen_token)" || repro_fatal "probe: the reopen of $CL_ROOT was REFUSED (its reason is above) — nothing submitted"
  [ -z "$token" ] || repro_say "probe: the record is COMPLETE — every staged geometry overlay of $MODEL is revoked (REVOKED markers written) and the revocation receipt is in the record; the resume carries the receipt's nonce as positional 10"
  # positional 11 `reference` rides only behind a token: a reference-only re-run is a reopen of a COMPLETE record by construction;
  # positional 11 `parity` (the pass's own resume) rides behind the carrier's `-` (= no token) when the pass's record is incomplete
  [ "$phase" != reference ] || [ -n "$token" ] || repro_fatal "probe: --reference-only re-judges a COMPLETE record; $CL_ROOT is not complete — resume its launches first"
  [ -z "$phase" ] || [ -n "$token" ] || token=-
  local how="--resume --replan (D108)"; case "$phase" in reference) how="--resume --reference-only (D146)";; parity) how="--resume --replan under phase=parity (the pass's plan flags; D163)";; esac
  # --certify-v2 rides the carrier's env (SERVING_PROBE_CERTIFY_V2 -> the driver's --certify-v2 beside --resume --replan: the named
  # pairs' current servers planned at the ladder window when absent, their v1-only results superseded and re-run under the certificate's
  # prompts); a reference-only re-run has no launches to certify with
  [ -z "$PROBE_CERTIFY_V2" ] || [ -z "$phase" ] || repro_fatal "probe: --certify-v2 re-runs launches; --reference-only runs none — pass one of them"
  local tlim="${PROBE_TIME:-$SISTEER_TIME_PROBE}"
  repro_say "probe: resuming $CL_ROOT as attempt $R_ATTEMPT/$R_MAX (counter and cap from $R_SOURCE${R_CAP_WIDEN:+, cap widened $R_CAP_WIDEN}) — sets $R_SETS, ranks $R_RANKS, mode $R_MODE, width $R_WIDTH${flags:+ ($flags)}, parallel $R_PARALLEL, time $tlim${PROBE_TIME:+ (--probe-time)}${PROBE_CERTIFY_V2:+, certificate v2 for $PROBE_CERTIFY_V2 (SERVING_PROBE_CERTIFY_V2)}; the carrier passes $how and refuses a checkout that is not ${SHA:0:12}"
  submit "$punit.a$R_ATTEMPT" "$pname" scripts/sbatch/serving_probe.sbatch "$MODEL $SRV_REPO $SRV_REV $R_SETS $R_RANKS $R_MODE $R_ATTEMPT $R_MAX $R_PARALLEL${token:+ $token}${phase:+ $phase}" "$tlim" --kind other --flags "$flags" --env "SERVING_PROBE_OUT=$CL_ROOT${PROBE_CERTIFY_V2:+ SERVING_PROBE_CERTIFY_V2=$PROBE_CERTIFY_V2}${PROBE_CAP_NOTE:+ SERVING_PROBE_CAP_WIDEN='$PROBE_CAP_NOTE'}${PROBE_PARITY_SCALES:+ SERVING_PROBE_PARITY_SCALES=$PROBE_PARITY_SCALES}"
}
PROBE_STATUS_LINE=""
probe_status_cached(){   # <job-active 0|1> -> the record's status, read ONCE per invocation (nothing this invocation does changes the record)
  [ -n "$PROBE_STATUS_LINE" ] || PROBE_STATUS_LINE="$(probe_status "$1")" || exit $?
  printf '%s\n' "$PROBE_STATUS_LINE"
}
probe_local_generation(){   # -> the result generation the LOCAL menu was fetched at (probe_fetch.json, else the DONE marker's token; empty when unrecorded)
  local g="" marker; marker="$(repro_state_dir "$MODEL")/$STAGE/DONE.$CLASS.probe"
  [ -s "$LOCAL_ROOT/probe_fetch.json" ] && g="$(json_field "$(cat "$LOCAL_ROOT/probe_fetch.json")" generation)"
  [ -n "$g" ] || g="$(sed -n 's/.*generation=\([0-9a-f]\{64\}\).*/\1/p' "$marker" 2>/dev/null | head -1)"
  printf '%s' "$g"
}
probe_local_currency(){   # <state> <status json> -> current | refresh | stale <why>  (currency is the VERDICT's identity, never the generation alone —
  # a `reverdict --parity-record` keeps the result generation, since it hashes the launch results, but rewrites probe_verdict / allowed_pairs /
  # geometry_record; the local mirror kept a 0-pair verdict while the cluster held 9 pairs, and the front door said DONE)
  local gen rid lid fid f v m g pchk pgen
  [ "$1" = READY ] || { echo "stale the record is $1, not READY"; return 0; }
  # a MISSING DONE marker alone is never evidence — a fetch that died between the exposure and the
  # mirror leaves the CURRENT menu / verdict / geometry and a valid home receipt; the marker is judged LAST, after READY / generation / identity /
  # receipt, and when all of those agree the marker (and the site copy) is RECOVERED in place — archived only on a POSITIVE mismatch
  local marker; marker="$(repro_state_dir "$MODEL")/$STAGE/DONE.$CLASS.probe"
  gen="$(json_field "$2" generation)"
  [ -n "$gen" ] && [ "$gen" = "$(probe_local_generation)" ] || { echo "stale the local menu was fetched at generation $(probe_local_generation | cut -c1-12), the record is at ${gen:0:12}"; return 0; }
  rid="$(probe_remote_identity)" || { echo "stale the record's verdict identity could not be read on ${PROBE_CLUSTER:-$CLUSTER}"; return 0; }
  lid="$(probe_local_identity "$LOCAL_ROOT")"
  [ "$lid" = "$rid" ] || { echo "stale the mirror's verdict / menu / geometry ($lid) is not the record's ($rid) — the verdict was rewritten (a re-verdict keeps the generation)"; return 0; }
  pchk="$(json_field "$2" parity_check)"; pgen="$(json_field "$(json_field "$2" parity_record)" stamped_generation)"
  f="$LOCAL_ROOT/probe_fetch.json"; fid=""
  [ -s "$f" ] && fid="$(json_field "$(cat "$f")" verdict_sha256) $(json_field "$(cat "$f")" menu_sha256) $(json_field "$(cat "$f")" geometry_sha256)"
  if [ "$fid" = "$rid" ] && [ "$(json_field "$(cat "$f")" parity_check)" = "${pchk:-none}" ] && [ "$(json_field "$(cat "$f")" parity_stamped_generation)" = "$pgen" ]; then
    [ -f "$marker" ] && { echo current; return 0; }
    echo "recover the mirror IS the record ($rid) and its fetch record names it, but the DONE marker $marker is missing (a fetch interrupted between the exposure and the mark) — the site copy and the marker are recovered in place, nothing archived"; return 0
  fi
  echo "refresh the mirror IS the record ($rid) but probe_fetch.json records ${fid:-no verdict identity} / parity $(json_field "$(cat "$f" 2>/dev/null)" parity_check) — refreshing the fetch record in place"
}
probe_local_is_current(){   # <state> <status json>: current, or refreshed in place (the mirror IS the record; only the fetch record lacked the identity)
  local verdict why; read -r verdict why <<<"$(probe_local_currency "$1" "$2")"
  case "$verdict" in
    current) return 0;;
    recover) repro_say "probe: $why"; geometry_site_recover "$LOCAL_ROOT"
             probe_mark_fetched "$2" "$(probe_local_identity "$LOCAL_ROOT")" "$(json_field "$(cat "$ALLOWED")" written_utc)" "$(json_field "$(cat "$ALLOWED")" probe_job_id)"; return 0;;
    refresh) repro_say "probe: $why"; probe_write_fetch_record "$2" "$(probe_local_identity "$LOCAL_ROOT")" "$(json_field "$(cat "$ALLOWED")" written_utc)" "$(json_field "$(cat "$ALLOWED")" probe_job_id)"
             probe_mark_fetched "$2" "$(probe_local_identity "$LOCAL_ROOT")" "$(json_field "$(cat "$ALLOWED")" written_utc)" "$(json_field "$(cat "$ALLOWED")" probe_job_id)"; return 0;;
    *) repro_say "probe: the local menu is NOT current — $why"; return 1;;
  esac
}
probe_archive_local(){   # <phase> <state> <detail> <status json> [deliberate]: the local menu / verdict / fetch record / geometry record + the DONE marker ->
  # $LOCAL_ROOT/superseded/<utc>/ (NEVER deleted), the site's geometry copy -> $SISTEER_SITE_DIR/geometry/superseded/<utc>/ so the
  # wave stager cannot pick up a fold of a superseded result set; says what moved. It exits (repro_fatal, by name) ONLY
  # when an archival it has decided to make cannot be completed — a place that cannot be reserved, a file that cannot be moved: that is
  # a REFUSAL of the operation $1, never a warning the caller proceeds past (a caller that went on would fetch over the
  # unarchived mirror — probe_fetch_ready's fetch record, probe_expose_staged's os.replace, geometry_to_site over the site copy — or reopen
  # the record past it); a path that decides to move nothing returns 0 as before.
  # ARCHIVAL NEEDS POSITIVE EVIDENCE FROM A REAL READ (a dry run of 2026-09-30 21:13Z archived the live
  # mirror on the dry run's SYNTHETIC NOT_STARTED): under --dry-run this prints what WOULD move and moves nothing; a status with no result
  # generation (an empty, unreadable or synthetic read) moves nothing; a record whose generation AND identity equal the mirror's moves
  # nothing (no mismatch to act on). Only the reopen verbs, which make the mirror stale by construction, pass `deliberate`.
  # ONE archival at TWO places under ONE stamp (lib/common.sh repro_superseded_stamp: the grading host's clock, UTC to the microsecond, the
  # one format every superseded/ directory is named in): each place RESERVES its directory through repro_superseded_dir with that stamp —
  # a bare mkdir, the next `-NN` suffix only where THAT place collides — so two archivals in one instant (a frozen clock, two
  # invocations) land beside each other and the message names the directory each place took. Until 2026-10-09 the stamp was to the
  # second and the directory `mkdir -p`'d, so the second `mv -f` wrote the first archive over: the stage-02 pools-marker defect
  # and the geometry chain's (lib/slurm.sh), in this function too (it archives into the same geometry/superseded/ base as
  # that chain). A place with nothing to archive reserves nothing — no empty directory, no second one "for nothing".
  local utc arch f site_copy site_arch="" lgen marker; utc="$(repro_superseded_stamp)"; arch="$LOCAL_ROOT/superseded/$utc"
  marker="$(repro_state_dir "$MODEL")/$STAGE/DONE.$CLASS.probe"; lgen="$(probe_local_generation)"
  site_copy="$SISTEER_SITE_DIR/geometry/$MODEL.json"
  if [ "${REPRO_DRY_RUN:-0}" = 1 ]; then
    local would=""; for f in "$LOCAL_ROOT"/allowed_pairs.json "$LOCAL_ROOT"/probe_verdict.json "$LOCAL_ROOT"/probe_fetch.json "$LOCAL_ROOT"/geometry_record*.json "$marker"; do [ -f "$f" ] && would="$would $(basename "$f")"; done
    [ -f "$site_copy" ] && would="$would site:geometry/$MODEL.json"
    echo "DRY-RUN: probe ($1): would archive${would:- nothing} under $(disp "$LOCAL_ROOT")/superseded/<utc>/ (the site copy under geometry/superseded/<utc>/) — $3; NOTHING moved under a dry run: the record was not read, and a synthetic state is never evidence"
    return 0
  fi
  if [ "${5:-}" != deliberate ]; then
    local rgen rid; rgen="$(json_field "$4" generation)"
    if [ -z "$rgen" ]; then
      repro_warn "probe ($1): the record at ${PROBE_CLUSTER:-$CLUSTER}:${PROBE_CL_ROOT:-$CL_ROOT} reads $2 with NO result generation — an empty, unreadable or synthetic read is never evidence against the local copies (D225): NOTHING moved. $3"; return 0
    fi
    rid="$(probe_remote_identity 2>/dev/null)" || { repro_warn "probe ($1): the record's verdict identity could not be read at ${PROBE_CLUSTER:-$CLUSTER} — no positive mismatch: NOTHING moved. $3"; return 0; }
    if [ "$rgen" = "$lgen" ] && [ "$rid" = "$(probe_local_identity "$LOCAL_ROOT")" ]; then
      repro_warn "probe ($1): the record at ${PROBE_CLUSTER:-$CLUSTER}:${PROBE_CL_ROOT:-$CL_ROOT} IS the mirror (generation ${rgen:0:12}, identity equal; state $2) — no positive mismatch: NOTHING moved. $3"; return 0
    fi
  fi
  # EVERY PLACE IS RESERVED BEFORE ANYTHING MOVES, AND EVERY MOVE MUST LAND (a reservation that failed was a warning and
  # a `return 0`, so probe_fetch_ready went on to write the fetch record, os.replace the menu / verdict / geometry and write the site copy
  # over — the old evidence lost, not archived; the mirror's superseded/ left unwritable while the mirror itself was writable is enough).
  # A place that cannot be reserved REFUSES $1 with nothing moved (a second place that fails gives the first, still empty, back — no
  # directory for nothing). Every refusal names what stands, what moved, and the remedy; nothing is ever deleted.
  # THE MOVES GO IN THE REVERSE OF THE ORDER THE FETCH WRITES, AND THE SEQUENCE STOPS AT THE FIRST THAT FAILS:
  # probe_fetch_ready writes the fetch record first, exposes the files with the evidence last, copies the geometry record to the site and
  # marks DONE last — so this archival moves the DONE marker first, the site's geometry copy, the evidence (geometry record, verdict,
  # menu) and the fetch record LAST, and a move that fails ends the sequence there. What stands after ANY failure is therefore a prefix
  # of the fetch's own write order: a state an interrupted fetch leaves, which the next run already recovers by itself — a standing DONE
  # marker means nothing moved (it is revalidated and the archival made again); evidence or a site copy standing beside the fetch record,
  # no marker, has its home (probe_home) and the probe phase fetches, archiving what stands on the way (probe_fetch_ready's own archival;
  # identical bytes are replaced); a fetch record alone is archived by that same archival, never written over. Until a1db16018 the marker
  # moved LAST and the loop went on past a failed move: a state directory made read-only left the marker standing with every mirror file
  # archived — probe_home refused (evidence with no fetch record naming its home), --rehome refused (no verdict to read the home from),
  # and the probe could not retry without a hand's move.
  # BEFORE ANYTHING IS RESERVED, every directory a file leaves is checked writable and searchable (a rename unlinks its source from that
  # directory): the failures that happen in practice — a state directory, a mirror or a site made read-only — are refused with NOTHING
  # moved and nothing reserved; the order above covers what no check can see (a race between the check and the move, a device full under
  # a cross-device superseded/, an immutable file).
  local moves=() has_site=0 n_local
  [ ! -f "$marker" ] || moves+=("$marker")
  [ ! -f "$site_copy" ] || { moves+=("$site_copy"); has_site=1; }
  for f in "$LOCAL_ROOT"/geometry_record*.json "$LOCAL_ROOT"/probe_verdict.json "$LOCAL_ROOT"/allowed_pairs.json "$LOCAL_ROOT"/probe_fetch.json; do [ -f "$f" ] && moves+=("$f"); done
  n_local=$(( ${#moves[@]} - has_site ))
  local stands=""; [ ! -f "$marker" ] || stands="; the DONE marker stands (it is revalidated again on the next run)"
  local d bad="" names=""
  for f in ${moves[@]+"${moves[@]}"}; do
    if [ "$f" = "$site_copy" ]; then names="$names site:geometry/$MODEL.json"; else names="$names $(basename "$f")"; fi
    d="$(dirname "$f")"; case " $bad " in *" $d "*) continue;; esac
    [ -w "$d" ] && [ -x "$d" ] || bad="$bad $d"
  done
  [ -z "$bad" ] || repro_fatal "probe ($1): NOTHING moved and nothing reserved — a file cannot leave a directory that is not writable:$bad; the archival would move$names$stands; $1 is refused rather than fetch or reopen over unarchived evidence: make the directory writable and re-run (the archival is then made whole, under one stamp). $3"
  if [ "$n_local" -gt 0 ]; then
    arch="$(repro_superseded_dir "$LOCAL_ROOT/superseded" "$utc")" \
      || repro_fatal "probe ($1): no archive directory could be reserved under $LOCAL_ROOT/superseded/ for the $n_local local file(s) (the mirror's superseded/ is not a writable directory) — NOTHING moved$stands; $1 is refused rather than fetch or reopen over unarchived evidence: make $LOCAL_ROOT/superseded/ writable (or move it aside by hand, never delete) and re-run. $3"
  fi
  if [ "$has_site" = 1 ]; then
    site_arch="$(repro_superseded_dir "$SISTEER_SITE_DIR/geometry/superseded" "$utc")" \
      || { [ "$n_local" -eq 0 ] || rmdir "$arch" 2>/dev/null || true
           repro_fatal "probe ($1): no archive directory could be reserved under $SISTEER_SITE_DIR/geometry/superseded/ for the site's geometry copy $site_copy (not a writable directory) — NOTHING moved$stands$([ "$n_local" -eq 0 ] || echo " (the mirror's own reservation under $LOCAL_ROOT/superseded/ given back)"); $1 is refused rather than fetch or reopen over an unarchived fold: make $SISTEER_SITE_DIR/geometry/superseded/ writable (or move the copy aside by hand, never delete) and re-run. $3"; }
  fi
  local i j dst rest was
  for ((i = 0; i < ${#moves[@]}; i++)); do
    f="${moves[i]}"; if [ "$f" = "$site_copy" ]; then dst="$site_arch"; else dst="$arch"; fi
    if mv -f "$f" "$dst/" && [ -f "$dst/$(basename "$f")" ] && [ ! -e "$f" ]; then
      [ "$f" != "$site_copy" ] || repro_say "probe: the site's geometry copy $site_copy -> $site_arch/ (a fold of a superseded result set must not be staged to a cluster)"
      continue
    fi
    # the first move that fails ENDS the sequence (what stands is a prefix of the fetch's write order, above); a reserved directory still
    # empty is given back (rmdir refuses one that holds an archive)
    [ -z "$site_arch" ] || rmdir "$site_arch" 2>/dev/null || true; [ "$n_local" -eq 0 ] || rmdir "$arch" 2>/dev/null || true
    rest="${moves[*]:i+1}"; was=""
    for ((j = 0; j < i; j++)); do if [ "${moves[j]}" = "$site_copy" ]; then was="$was $site_arch/$(basename "${moves[j]}")"; else was="$was $arch/$(basename "${moves[j]}")"; fi; done
    if [ -n "$was" ]; then was="moved before it, now at:$was"; else was="nothing moved before it"; fi
    repro_fatal "probe ($1): $f could NOT be moved into $dst/ — it stands, and so does everything after it in the archival's order${rest:+ ($rest)}; $was (nothing deleted). What stands is what a fetch interrupted here leaves (the marker goes first, the fetch record last) and the next run recovers it by itself — a standing DONE marker is revalidated and the archival made again; evidence or a site copy beside the fetch record is archived by the fetch itself, a fetch record alone too; $1 is refused rather than fetch or reopen over unarchived evidence: fix what refused the move ($(dirname "$f") writable, $dst on the same device) and re-run. $3"
  done
  fetch_invalidate "$CL_ROOT" "$LOCAL_ROOT"      # the mirror is stale by construction now: a later fetch of this pair must rsync again
  local said; if [ "$n_local" -gt 0 ]; then said="$n_local local file(s) archived under $arch"; else said="no local file to archive (no directory reserved under $LOCAL_ROOT/superseded/)"; fi
  [ -z "$site_arch" ] || said="$said; the site's geometry copy under $site_arch"
  repro_warn "probe ($1): the LOCAL menu is NOT the record's current verdict — record $2 (generation $(json_field "$4" generation | cut -c1-12)), local ${lgen:0:12}: $3. $said (nothing deleted); the DONE marker is cleared"
}
probe_take_lock(){   # ONE invocation at a time reads the record, archives and submits: a NON-BLOCKING flock on this host — every
  # front-door lane runs here — keyed by the record's local mirror; a concurrent invocation refuses by name (it would otherwise read FAILED_START
  # for a carrier the first one just started and archive its pre-checkpoint attempt file). Held to the end of the phase. SISTEER_PROBE_PAUSE_FILE
  # (tests only) holds the lock while the file exists. A dry run touches nothing and takes no lock.
  [ "$REPRO_DRY_RUN" != 1 ] || return 0
  PROBE_LOCK="$(repro_state_dir "$MODEL")/$STAGE/.probe.$CLASS.recover.lock"; mkdir -p "$(dirname "$PROBE_LOCK")"
  exec {PROBE_LK}>"$PROBE_LOCK" || repro_fatal "probe: cannot open the record's recovery lock $PROBE_LOCK — nothing read, nothing submitted"
  flock -n "$PROBE_LK" || repro_fatal "probe: another 03 --phase probe invocation holds the recovery lock $PROBE_LOCK for $CL_ROOT — one at a time reads the record, archives and submits; re-run when it finishes (nothing archived, nothing submitted)"
  repro_say "probe: recovery lock $PROBE_LOCK taken for $CL_ROOT"
  if [ -n "${SISTEER_PROBE_PAUSE_FILE:-}" ]; then while [ -e "$SISTEER_PROBE_PAUSE_FILE" ]; do sleep 0.2; done; fi
}
probe_rehome_verb(){   # --phase probe --rehome: a mirror whose fetch record is gone or names no home gets it back from
  # its OWN verdict — provenance.probe_cluster + provenance.probe_record — VERIFIED against that cluster before anything is trusted: the cluster
  # is one of SISTEER_CLUSTERS and hpo_probe_status reads a record at that path (NOT_STARTED = no record there). The fetch record is written with
  # the mirror's ACTUAL identity and the record's parity fields; the DONE marker too when the mirror IS the record (its hashes are the
  # record's) — else the ordinary currency check archives and re-fetches on the next run. Nothing on any cluster is touched.
  local v="$LOCAL_ROOT/probe_verdict.json" ev prov c r line st detail info rid lid mgen
  ev="$(probe_local_evidence)"
  [ -n "$ev" ] || repro_fatal "probe --rehome: no local probe evidence under $LOCAL_ROOT — nothing to re-home (a fresh --phase probe --cluster <label> fetches)"
  [ -s "$v" ] || repro_fatal "probe --rehome: the mirror holds $ev but no probe_verdict.json to read the home from — move the evidence under $LOCAL_ROOT/superseded/<utc>/ by hand (never delete) and re-run --phase probe --cluster <label>"
  prov="$(json_field "$(cat "$v")" provenance)"; c="$(json_field "$prov" probe_cluster)"; r="$(json_field "$prov" probe_record)"
  [ -n "$c" ] && [ -n "$r" ] || repro_fatal "probe --rehome: $v carries no provenance.probe_cluster / probe_record (cluster '$c', record '$r') — the home cannot be re-derived; move the evidence under $LOCAL_ROOT/superseded/<utc>/ by hand (never delete) and re-run --phase probe --cluster <label>; nothing written"
  case " $SISTEER_CLUSTERS " in *" $c "*) ;; *) repro_fatal "probe --rehome: $v names cluster '$c', not one of SISTEER_CLUSTERS ($SISTEER_CLUSTERS) — nothing written";; esac
  PROBE_CLUSTER="$c"; PROBE_CL_HOST="$(cluster_var "$c" CLUSTER_SSH)"; PROBE_CL_ROOT="$r"; PROBE_HOME_SOURCE="probe_verdict.json"
  if [ "$REPRO_DRY_RUN" = 1 ]; then echo "DRY-RUN: rehome $(disp "$LOCAL_ROOT") -> $c:$r (verify hpo_probe_status.py --record $r on $c, then write probe_fetch.json)"; return 0; fi
  local active=0; probe_job_active "$(name probe)" && active=1
  line="$(probe_status "$active")" || exit $?
  IFS=$'\t' read -r st detail info <<<"$line"
  [ "$st" != NOT_STARTED ] || repro_fatal "probe --rehome: $c:$r holds no probe record (hpo_probe_status: $detail) — the verdict's home is not a record; nothing written"
  rid="$(probe_remote_identity)" || repro_fatal "probe --rehome: the record $r on $c could not be read for its verdict identity — nothing written"
  lid="$(probe_local_identity "$LOCAL_ROOT")"
  mgen="$(json_field "$(cat "$v")" result_generation)"          # the generation the MIRROR's verdict carries, else the record's current one
  [ -z "$mgen" ] || info="$(printf '%s' "$info" | "$PY" -c 'import json,sys; d=json.load(sys.stdin); d["generation"]=sys.argv[1]; print(json.dumps(d))' "$mgen")"
  probe_write_fetch_record "$info" "$lid" "$(json_field "$(cat "$ALLOWED" 2>/dev/null)" written_utc)" "$(json_field "$(cat "$ALLOWED" 2>/dev/null)" probe_job_id)"
  if [ "$lid" = "$rid" ] && [ "$st" = READY ]; then
    probe_mark_fetched "$info" "$lid" "$(json_field "$(cat "$ALLOWED")" written_utc)" "$(json_field "$(cat "$ALLOWED")" probe_job_id)"
    repro_say "probe --rehome: $LOCAL_ROOT is re-homed to $c:$r — the mirror IS the record (verdict / menu / geometry ${lid:0:12}…, state $st); probe_fetch.json and the DONE marker written; the consuming phases proceed, --phase probe --cluster $c revalidates as usual"
  else
    repro_say "probe --rehome: $LOCAL_ROOT is re-homed to $c:$r (record state $st) — the mirror is NOT the record's current verdict (mirror $lid, record $rid); probe_fetch.json written with the mirror's own identity, no DONE marker: the next --phase probe --cluster $c archives the mirror and re-fetches"
  fi
}
probe_revalidate_local(){   # <phase>: a phase that CONSUMES the menu (recon for lora, design, trials, refine, pick, confirm-pick) first
  # revalidates what it holds against the record: the local menu must be the record's CURRENT result generation —
  # a reopened record (checkpoint not complete), a stale menu or a different generation archives the local copies, clears the
  # marker and REFUSES the phase by name until --phase probe re-fetches a READY record; an unreachable record refuses (probe_status)
  [ "$CLASS" = lora ] && [ "$PROTOCOL" = fd6 ] || return 0                   # only the FD 6 LoRA search has a menu
  local marker; marker="$(repro_state_dir "$MODEL")/$STAGE/DONE.$CLASS.probe"
  [ -s "$ALLOWED" ] || [ -f "$marker" ] || [ -s "$LOCAL_ROOT/probe_verdict.json" ] || return 0   # nothing local: the phase's own "needs $ALLOWED" stands
  probe_home                                                                   # the record WHERE IT LIVES (probe_fetch.json), not $CLUSTER's path
  [ "$PROBE_HOME_SOURCE" != probe_fetch.json ] || [ "$PROBE_CLUSTER" = "$CLUSTER" ] \
    || repro_note "probe: the menu's record lives on $PROBE_CLUSTER:$PROBE_CL_ROOT (probe_fetch.json); $1 runs on $CLUSTER and revalidates against that record"
  if [ "$REPRO_DRY_RUN" = 1 ]; then repro_note "probe: dry run — the local menu is not revalidated against the record (live: hpo_probe_status.py --record $PROBE_CL_ROOT on $PROBE_CLUSTER before $1)"; return 0; fi
  local active=0 st detail info line; probe_job_active "$(name probe)" && active=1
  line="$(probe_status_cached "$active")" || exit $?
  IFS=$'\t' read -r st detail info <<<"$line"
  if probe_local_is_current "$st" "$info"; then repro_say "probe: the local menu is the record's current result generation ($(probe_local_generation | cut -c1-12); record $PROBE_CLUSTER:$PROBE_CL_ROOT) — $1 proceeds"; return 0; fi
  probe_archive_local "$1" "$st" "$detail" "$info"
  repro_fatal "$1: the local menu for $MODEL was NOT the record's current result generation (archived above; record $PROBE_CLUSTER:$PROBE_CL_ROOT state $st) — refusing until --phase probe --cluster $PROBE_CLUSTER re-fetches a READY record"
}

# ================================================================================================================ probe
if want probe; then
  if [ "$CLASS" != lora ]; then
    [ "$PHASE" = probe ] && repro_fatal "--phase probe is the LoRA serving probe; the ReLU search has no module-set menu"
  elif [ "$PROTOCOL" = paper ] && [ "$PHASE" != probe ]; then
    # demo-import M6 (Q3 ruled (a)): the paper's protocol fixes the architecture (FD 18), so the SEARCH consumes no module-set menu and
    # --phase all does not wait on the probe (design / trials read none) — but the serving probe is still RUN, by --phase probe, for the
    # ONE declared pair: its verdict prices the compute plan's serving rows (tools/replan_compute.py --probe; 06 / 09 refuse an incomplete
    # plan) and its staged geometry record admits every steered / padded standup of the served model (lib/slurm.sh slurm_stage_geometry;
    # geometry.standup_adapter_parity_gate). Nothing is read, nothing submitted here.
    repro_note "probe: the paper's protocol fixes the architecture (FD 18) — the search consumes no module-set menu, so --phase all does not wait on the probe; the serving probe certifies the ONE declared pair (the fits config's lora_r4_ce fit_selection: rank + module set) for the compute plan's serving rows and the steered standups of $MODEL — run --phase probe [--probe-width 1] before 06"
  else
    # serving_probe.sbatch's positionals 4/5 on a FRESH submission: EMPTY under FD 6 (the carrier's default menu — the rank x module-set
    # search's design space), the ONE declared pair under the paper's protocol (probe_paper_pair; the same state machine either way — the
    # record, the lock, the status reader's verdict, the fetch and the DONE marker are protocol-blind; a resume keeps the record's plan)
    PROBE_PAIR=""
    if [ "$PROTOCOL" = paper ]; then
      PROBE_PAIR="$(probe_paper_pair)" || repro_fatal "probe: the paper protocol's one declared pair cannot be derived from the fits config $FIT_CFG (its reason is above) — nothing read, nothing submitted"
      repro_note "probe: the paper's protocol fixes the architecture (FD 18) — the probe certifies the ONE declared pair, set ${PROBE_PAIR% *} / rank ${PROBE_PAIR#* } (the fits config's lora_r4_ce fit_selection, $FIT_CFG; the row's lora_architecture agrees): serving_probe.sbatch positionals 4/5 = '$PROBE_PAIR', not the carrier's default menu; a resume keeps the record's own probe_plan.json sets / ranks"
    fi
    # the probe PHASE acts on $CLUSTER's record (a fresh probe, a resume, a reopen). A menu already fetched from ANOTHER cluster's record
    # (probe_fetch.json) means this record is not the one the search holds: reading $CLUSTER's empty path would say NOT_STARTED, archive the
    # current menu and submit a fresh probe there — refused by name instead; the consuming phases revalidate against the record's home.
    # --rehome: a mirror whose fetch record is gone or names no home gets it back from its own verdict — BEFORE
    # probe_home, which refuses such a mirror by name; the verb takes the lock, verifies the home against that cluster's record, writes the
    # fetch record (and the DONE marker when the mirror IS the record) and ends the phase
    if [ "${PROBE_REHOME:-0}" = 1 ]; then
      [ "$PHASE" = probe ] || repro_fatal "--rehome needs --phase probe (the verb re-homes the mirror and ends there)"
      probe_take_lock; probe_rehome_verb; repro_say "03_hpo done (model=$MODEL class=$CLASS phase=$PHASE --rehome)"; exit 0
    fi
    probe_home
    if [ "$PROBE_HOME_SOURCE" = probe_fetch.json ] && [ "$PROBE_CLUSTER" != "$CLUSTER" ]; then
      repro_fatal "probe: the fetched menu's record lives on $PROBE_CLUSTER:$PROBE_CL_ROOT ($LOCAL_ROOT/probe_fetch.json), but --cluster is $CLUSTER — the probe phase runs where its record lives: re-run --phase probe --cluster $PROBE_CLUSTER (the consuming phases — recon, design, trials, refine, pick, confirm-pick — may run on $CLUSTER; they revalidate the menu against $PROBE_CLUSTER's record); nothing read, nothing archived, nothing submitted"
    fi
    probe_take_lock                          # one invocation at a time reads the record, archives and submits (held to the end of the phase)
    if [ "${PROBE_PARITY_PASS:-0}" = 1 ]; then
      # the PARITY PASS verb acts on the pass's own record (<hpo root>_parity) under the same lock and ends there (the main record is
      # neither read nor touched: the binding is the main record's --reopen --reverdict-only)
      probe_parity_pass_verb; repro_say "03_hpo done (model=$MODEL class=$CLASS phase=$PHASE --parity-pass)"; exit 0
    fi
    # THE RECORD IS THE AUTHORITY, read through hpo_probe_status.py --record ON the cluster: nothing here derives "done" from the
    # presence of allowed_pairs.json / geometry_record.json, on the cluster or in the mirror — the reader's verdict alone decides, and
    # exactly one state submits a FRESH probe (NOT_STARTED), exactly one resumes (INCOMPLETE_STALLED), exactly one fetches (READY). A
    # missing allowed_pairs.json is NOT "not started": the probe checkpoints its launch list in probe_progress.json, exits 4
    # (INCOMPLETE) when its time budget stops it, self-resubmits with --resume --replan on the SAME record and writes the menu only
    # when complete.
    PROBE_ACTIVE=0; cluster_job_active "$(name probe)" && PROBE_ACTIVE=1
    PROBE_STATUS_LINE="$(probe_status_cached "$PROBE_ACTIVE")" || exit $?
    IFS=$'\t' read -r PSTATE PDETAIL PINFO <<<"$PROBE_STATUS_LINE"
    if repro_done "$MODEL" "$STAGE" "$CLASS.probe"; then
      # a DONE marker is REVALIDATED on every invocation, never trusted: the local menu must be the record's
      # CURRENT result generation — else the local copies are archived, the marker cleared, and the record's state decides below
      # (a READY record is re-fetched in this same invocation); a DONE probe still re-copies a MISSING site record
      if [ "$REPRO_DRY_RUN" = 1 ]; then
        # the dry run's NOT_STARTED is SYNTHETIC (probe_status reads nothing) — it is not a verdict on the marker: nothing archived, nothing decided
        repro_say "DRY-RUN: probe: DONE marker present — live, the local menu is revalidated against the record (hpo_probe_status.py --record $PROBE_CL_ROOT on $PROBE_CLUSTER: current → skipped; a positive mismatch → archived and re-fetched); nothing read, nothing archived under a dry run"
        PSTATE=CURRENT
      elif probe_local_is_current "$PSTATE" "$PINFO"; then
        repro_say "probe: DONE marker present and the local menu is the record's current result generation ($(probe_local_generation | cut -c1-12)) — skipping"
        geometry_site_recover "$LOCAL_ROOT"; PSTATE=CURRENT
      else
        probe_archive_local probe "$PSTATE" "$PDETAIL" "$PINFO"
      fi
    fi
    # --reopen: the DELIBERATE reopen of a READY (or DONE-and-current) record — revoke every staged
    # overlay, mint the receipt into the record, resume with its nonce; the verb sets PSTATE=REOPENED so the case below neither fetches
    # nor skips; any other state (a job holding the record included) refuses by name with nothing revoked and nothing submitted
    [ "${PROBE_REOPEN:-0}" = 1 ] && probe_reopen_verb
    case "$PSTATE" in
      CURRENT|REOPENED) ;;
      READY)
        probe_fetch_ready "${PINFO:-{\}}";;
      NOT_STARTED)
        # the fresh unit may already be BOUND in the ledger to a job that died before creating any record file (a shared-root /
        # SERVING_PROBE_OUT failure before `mkdir $OUT`, a Slurm start failure): a plain submit would be
        # skipped by the funnel as already submitted (rc 0) and `--phase all` would wait forever. Bound + no job active -> refuse by
        # name with the dead job and its log, unless --resubmit forwards the funnel's fenced proof
        read -r B_STATE B_JOB B_CLUSTER <<<"$(probe_fresh_binding)"
        if [ "$B_STATE" != unbound ] && [ "$PROBE_ACTIVE" != 1 ] && [ "$PROBE_RESUBMIT" != 1 ]; then
          repro_fatal "probe: the record $CL_ROOT is EMPTY (NOT_STARTED) but the fresh unit $(unit probe) is already $B_STATE in the ledger to job ${B_JOB:-?} on ${B_CLUSTER:-$CLUSTER}, and no probe job is active: that job died before writing any record file — read its log $(cluster_var "$CLUSTER" CLUSTER_LOG_DIR)/$(name probe)_${B_JOB:-<job>}.out, then re-run --phase probe --resubmit (lib/slurm.sh re-mints $(unit probe) after ITS proof that job ${B_JOB:-?} is terminal and registered nothing; a plain re-run would be skipped as already submitted)"
        fi
        repro_note "probe: scripts/sbatch/serving_probe.sbatch (agent D6's) writes probe_verdict.json + allowed_pairs.json under its certification root; SERVING_PROBE_OUT=$CL_ROOT points it at this search's root"
        # the width: by default the sbatch header's (8 GPUs, `parallel` auto = 8 concurrent one-GPU standups) — the cluster
        # block has no per-job GPU knob (CLUSTER_HIGH_BUDGET_GPUS is the budget, CLUSTER_SBATCH_EXTRA is every submission's);
        # --probe-width N puts the carrier's documented narrower allocation on the command line and the driver derives the
        # concurrency; the positionals stay at their defaults either way (a fresh record starts at attempt 1 of the carrier's cap)
        PROBE_FLAGS="$(probe_width_flags "${PROBE_WIDTH:-auto}")"
        [ -z "$PROBE_FLAGS" ] || repro_note "probe: --probe-width $PROBE_WIDTH -> $PROBE_FLAGS (parallel auto)"
        # --resubmit: the fresh unit is already bound to a dead job (a FAILED_START whose attempt record was moved aside): the
        # funnel re-mints it only after ITS proof that the job is terminal and registered nothing — forwarded, never bypassed
        RESUB=(); [ "$PROBE_RESUBMIT" = 1 ] && { RESUB=(--resubmit); repro_note "probe: --resubmit forwarded to lib/slurm.sh for $(unit probe) (the funnel's proof decides)"; }
        # positionals 4/5 ride only under the paper's protocol (PROBE_PAIR, one word appended inside the args string); the FD 6 line is byte-identical
        submit "$(unit probe)" "$(name probe)" scripts/sbatch/serving_probe.sbatch "$MODEL $SRV_REPO $SRV_REV"${PROBE_PAIR:+" $PROBE_PAIR"} "$SISTEER_TIME_PROBE" --kind other --flags "$PROBE_FLAGS" "${RESUB[@]+"${RESUB[@]}"}" --env "SERVING_PROBE_OUT=$CL_ROOT"
        [ "$PHASE" = all ] && { repro_say "waiting on the probe — re-run --phase all when it is DONE"; exit 0; };;
      RUNNING|INCOMPLETE_RUNNING)
        repro_note "probe: $PSTATE — $PDETAIL; job $(name probe) is queued/running on $CLUSTER — nothing submitted; re-run when its watcher reports DONE"
        [ "$PHASE" = all ] && { repro_say "waiting on the probe — re-run --phase all when it is DONE"; exit 0; };;
      INCOMPLETE_STALLED)
        probe_resume "$PDETAIL"
        [ "$PHASE" = all ] && { repro_say "waiting on the probe's resumed attempt — re-run --phase all when it is DONE"; exit 0; };;
      MENU_STALE)
        # a menu that is not the record's current result generation (reopened by `run --resume --replan`, or an incomplete
        # checkpoint): NEVER fetched, NEVER marked; handled like INCOMPLETE_* — a running job is the re-probe, else the resume
        # re-runs what is outstanding and the carrier rewrites the verdict + menu from the current results (170c98a29)
        if [ "$PROBE_ACTIVE" = 1 ]; then
          repro_note "probe: MENU_STALE — $PDETAIL; job $(name probe) is queued/running on $CLUSTER (the re-probe) — NOT fetched, nothing submitted; re-run when its watcher reports DONE"
        else
          repro_say "probe: the menu on $CLUSTER is STALE and no job holds the record — NOT fetched, NOT marked; resuming the record so its verdict and menu are rewritten from the current results"
          probe_resume "$PDETAIL"
        fi
        [ "$PHASE" = all ] && { repro_say "waiting on the probe — re-run --phase all when it is DONE"; exit 0; };;
      FAILED_START)
        # the carrier wrote probe_attempt.json but died before the driver's first checkpoint (venv bootstrap / import / sha pin) and
        # no job holds the record. With a plan the record RESUMES like INCOMPLETE (attempt+1
        # from the attempt record; the carrier passes --resume --replan). With no plan nothing was planned: a fresh start is
        # needed, but the fresh unit $(unit probe) is already bound in the ledger to the dead job (the funnel would skip it, rc 0 —
        # the stuck-forever case), so the phase REFUSES by name with the recovery steps; nothing is re-typed, nothing deleted.
        if [ "$(json_field "${PINFO:-{\}}" plan_present)" = true ]; then
          repro_say "probe: FAILED_START on $CLUSTER — $PDETAIL; the record holds a plan — resuming it"
          probe_resume "$PDETAIL"
          [ "$PHASE" = all ] && { repro_say "waiting on the probe's resumed attempt — re-run --phase all when it is DONE"; exit 0; }
        else
          A_JOB="$(json_field "$(json_field "${PINFO:-{\}}" attempt)" job_id)"
          if [ "$PROBE_RESUBMIT" != 1 ]; then
            repro_fatal "probe: FAILED_START on $CLUSTER — $PDETAIL. No plan exists on $CL_ROOT, so there is nothing to resume, and the fresh unit $(unit probe) is already bound in the ledger to the dead job ${A_JOB:-?} (a plain re-submit would be skipped). Read the failed attempt's log $(cluster_var "$CLUSTER" CLUSTER_LOG_DIR)/$(name probe)_${A_JOB:-<job>}.out, fix the cause, then re-run --phase probe --resubmit: this phase will move every top-level file of the plan-less record (probe_attempt.json, the carrier's RESULT, leftovers) to $CL_ROOT/superseded/<utc>/ (never deleted), re-read the status, and hand the fresh submission to lib/slurm.sh with --resubmit — it re-mints $(unit probe) after ITS proof that job ${A_JOB:-?} is terminal and registered nothing"
          fi
          repro_say "probe: FAILED_START with no plan on $CL_ROOT (dead job ${A_JOB:-?}) — --resubmit given: proving the job dead through the funnel, then the guarded recovery"
          PROVED_JOB=""; probe_prove_dead "${A_JOB:-}"   # the record is untouched until the funnel's sacct proof says TERMINAL; sets PROVED_JOB
          probe_recover_planless "${PINFO:-{\}}" "${PROVED_JOB:-$A_JOB}"   # the mv is fenced to THAT attempt generation
          PROBE_STATUS_LINE=""; PROBE_STATUS_LINE="$(probe_status_cached "$PROBE_ACTIVE")" || exit $?
          IFS=$'\t' read -r PSTATE PDETAIL PINFO <<<"$PROBE_STATUS_LINE"
          [ "$PSTATE" = NOT_STARTED ] || repro_fatal "probe: after the recovery the record $CL_ROOT still reads $PSTATE ($PDETAIL) — not FAILED_START-free; nothing submitted"
          repro_say "probe: the record reads NOT_STARTED — handing the fresh submission of $(unit probe) to lib/slurm.sh with --resubmit (its fenced proof decides)"
          RESUB=(--resubmit)
          submit "$(unit probe)" "$(name probe)" scripts/sbatch/serving_probe.sbatch "$MODEL $SRV_REPO $SRV_REV"${PROBE_PAIR:+" $PROBE_PAIR"} "$SISTEER_TIME_PROBE" --kind other --flags "$(probe_width_flags "${PROBE_WIDTH:-auto}")" "${RESUB[@]}" --env "SERVING_PROBE_OUT=$CL_ROOT"
          [ "$PHASE" = all ] && { repro_say "waiting on the probe — re-run --phase all when it is DONE"; exit 0; }
        fi;;
      PARITY_STALE)
        # the menu IS the record's current result generation, but the PARITY-PASS record it read adapter parity from is not
        # what was stamped — moved (a reference redo, a re-run), incomplete, re-planned or absent. NEVER fetched, and NEVER probe_resume:
        # a complete main record's resume is a reopen the driver refuses without a receipt, and a plain `run --resume` would write a
        # verdict WITHOUT the parity record and revoke the standing menu. The remedy is the deliberate reopen once the parity record is
        # complete and at rest: --phase probe --reopen (revoke → receipt → `reverdict --parity-record <dir>`).
        P_WHY="$(json_field "${PINFO:-{\}}" parity_why)"; P_REC="$(json_field "${PINFO:-{\}}" parity_record)"
        repro_fatal "probe: PARITY_STALE on $CLUSTER — ${P_WHY:-$PDETAIL} [parity record $(json_field "$P_REC" dir): stamped generation $(json_field "$P_REC" stamped_generation | cut -c1-12) ($(json_field "$P_REC" stamped_n_results) results), live $(json_field "$P_REC" live_generation | cut -c1-12) ($(json_field "$P_REC" live_n_results) results), complete=$(json_field "$P_REC" complete)]. NOT fetched, NOT marked, NOT resumed — a complete record's resume is a reopen the driver refuses without a receipt, and a plain run --resume would write a verdict WITHOUT the parity record and revoke the standing menu. Remedy, once the parity record is complete and at rest: --phase probe --reopen (revoke → receipt → reverdict --parity-record $(json_field "$P_REC" dir))";;
      REOPEN_IN_PROGRESS)
        repro_fatal "probe: $PDETAIL — re-run --phase probe --reopen to route the recovery: the carrier runs reverdict --resume-interrupted under that receipt on $CLUSTER (never a fresh revocation); nothing fetched, nothing submitted"
        ;;
      VERDICT_REFUSED)
        repro_fatal "probe: $PDETAIL ($CL_ROOT/probe_verdict.json) — the design phase cannot start without a certified allowed_pairs.json$([ "$(json_field "${PINFO:-{\}}" verdict_status)" = uncertified ] && [ -n "$(json_field "${PINFO:-{\}}" parity_record)" ] && echo "; the verdict is uncertified for want of reference evidence on $(json_field "${PINFO:-{\}}" unreferenced_pairs) and read the parity record $(json_field "$(json_field "${PINFO:-{\}}" parity_record)" dir): --phase probe --reopen routes the receipted re-verdict against it (D623)")$([ "$(json_field "${PINFO:-{\}}" verdict_status)" = uncertified ] && [ -z "$(json_field "${PINFO:-{\}}" parity_record)" ] && echo "; the verdict is uncertified (unreferenced $(json_field "${PINFO:-{\}}" unreferenced_pairs)) and read no parity record: a harness-filed reference status is re-judged by --phase probe --reopen --reference-only (D146); assertion 4 at the benign scale is the parity pass — --phase probe --parity-pass, then --reopen --reverdict-only binds it (D163)")";;
      *) repro_fatal "probe: hpo_probe_status.py returned an unknown state '$PSTATE' ($PDETAIL) — nothing fetched, nothing submitted";;
    esac
  fi
fi

# ================================================================================================================ recon
# --- the CURRENT reconnaissance: every recon-phase decision is made against the spec THIS invocation would submit ------------
recon_compose_spec(){   # sets LADDER, SPEC (+ CYCLED, ROWS, EXTRA_JSON): the recon spec of the current config, cell and inputs; live reads the
  # cell's rows from the cluster and refuses a short cell (recon_passes_for_cell); a dry run states the check and composes without the cell fields
  LADDER="$(lr_ladder)"; EXTRA_JSON=""; CYCLED=1; ROWS=""; POOL_SHA=""; SIDECAR_SHA=""; CLUSTER_RECORD_DIGEST=""
  [ -n "$RECON_CL_HOST" ] || repro_fatal "recon: internal — recon_compose_spec before recon_home (the record's home is unresolved); nothing read, nothing moved"
  if [ "$CLASS" = lora ]; then
    if [ -s "$ALLOWED" ]; then
      WP="$(widest_pair "$ALLOWED")" || repro_fatal "recon (lora): $ALLOWED certifies no pair — the probe's verdict excluded every (rank, set); re-judge the reference (--phase probe --reopen ...) before the LoRA recon"; read -r WR WM <<<"$WP"
      EXTRA_JSON=", \"lora_rank\": $WR, \"lora_modules\": $("$PY" -c 'import json,sys; print(json.dumps(sys.argv[1].split(",")))' "$WM")"
      repro_note "recon: at the WIDEST allowed pair (rank $WR, modules $WM) — the largest object sets the most permissive clip"
    else
      [ "$REPRO_DRY_RUN" = 1 ] || repro_fatal "recon (lora) needs $ALLOWED (run --phase probe first): the ladder trains the widest allowed pair"
      repro_note "recon: allowed_pairs.json absent (dry run) — the live path trains the widest allowed pair"
    fi
  fi
  # the cell must cover the recipe's budget in ONE pass — rows read from the cluster's pairs file, live; a dry run states it
  local cell_json=""
  if [ "$REPRO_DRY_RUN" = 1 ]; then
    echo "DRY-RUN: rows of $RECON_PAIRS_HOME read live (ssh wc -l on $RECON_CLUSTER) -> recon_passes_for_cell ROWS $EFFECTIVE_BATCH $RECON_STEPS (refuses a cell short of $RECON_STEPS steps per pass unless SISTEER_HPO_RECON_ALLOW_CYCLING=1)"
  else
    # the CURRENT pool's content identity, read on the cluster in the same round trip as the rows: recorded in the
    # spec (pool_sha256 / sidecar_sha256 — recorded, not fingerprint keys) and passed to --finalize / --verify, which require every rung's
    # stamp to name exactly this copy — a pool rebuilt with other content at the same path never admits an old verdict
    # the SAME round trip also reads the record's current digest — sha256 of the sorted sha256sum list of every recon_<i>.json + recon.json
    # as they are on the cluster NOW (a rung re-run or replaced on the cluster behind a complete mirror leaves the mirror,
    # the marker and the local verdict mutually consistent, so a local --verify cannot see it; the digest can) — empty when the record holds none
    local ident; ident="$(ssh -o BatchMode=yes -o ConnectTimeout=30 "$RECON_CL_HOST" "wc -l < '$RECON_PAIRS_HOME'; sha256sum '$RECON_PAIRS_HOME' '$RECON_PAIRS_HOME.tokcache.npz' 2>/dev/null | cut -d' ' -f1; echo DIGEST; cd '$RECON_CL_ROOT' 2>/dev/null && ls recon_[0-9]*.json recon.json >/dev/null 2>&1 && sha256sum recon_[0-9]*.json recon.json 2>/dev/null | sort | sha256sum | cut -d' ' -f1" 2>/dev/null)"
    ROWS="$(sed -n 1p <<<"$ident" | tr -dc 0-9)"; POOL_SHA="$(sed -n 2p <<<"$ident" | tr -dc 0-9a-f)"; SIDECAR_SHA="$(sed -n '3{/^DIGEST$/!p}' <<<"$ident" | tr -dc 0-9a-f)"
    CLUSTER_RECORD_DIGEST="$(sed -n '/^DIGEST$/{n;p}' <<<"$ident" | tr -dc 0-9a-f)"
    [ ${#POOL_SHA} = 64 ] || repro_fatal "recon: cannot read the sha256 of the recon cell's pairs file $RECON_PAIRS_HOME on $RECON_CLUSTER (got '${POOL_SHA:-}') — the pool is missing or unreadable"
    CYCLED="$(recon_passes_for_cell "${ROWS:-0}" "$EFFECTIVE_BATCH" "$RECON_STEPS")" || exit 2
    cell_json=", \"rows\": $ROWS, \"effective_batch\": $EFFECTIVE_BATCH, \"steps_per_pass\": $(( (ROWS + EFFECTIVE_BATCH - 1) / EFFECTIVE_BATCH )), \"cycled_passes\": $CYCLED, \"pool_sha256\": \"$POOL_SHA\"${SIDECAR_SHA:+, \"sidecar_sha256\": \"$SIDECAR_SHA\"}"
    repro_say "recon: cell $RECON_SI — $ROWS rows, $(( (ROWS + EFFECTIVE_BATCH - 1) / EFFECTIVE_BATCH )) optimizer steps per pass at effective batch $EFFECTIVE_BATCH, $RECON_STEPS-step rungs in $CYCLED pass(es)"
  fi
  SPEC="{\"kind\": \"recon_spec\", \"lr_ladder\": [${LADDER//,/, }], \"steps\": $RECON_STEPS, \"clip_percentile\": $CLIP_PCT, \"warmup_frac\": $WARMUP, \"init_scheme\": \"$INIT_SCHEME\", \"si\": \"$RECON_SI\"$cell_json$EXTRA_JSON}"
}
# THE RECON RECORD'S HOME (the LoRA HPO lane 2026-09-30 — the recon twin of probe_home): the verdict's rung files + recon.json live on
# the cluster the ladder RAN ON, while design / trials of the same class may run on another (the budget's call). Every phase that CONSUMES the
# verdict revalidates it against the record WHERE IT LIVES — recon_fetch.json beside the mirrored verdict names {cluster, record, pairs, the
# record's digest at mark} — never against --cluster's copy of the path: on the other cluster that directory is absent, the digest reads EMPTY,
# and the revalidation used to archive the mirrored verdict, the rung files and the DONE marker, then refuse. Two rules, both fail-closed:
#  - an ABSENT or EMPTY cluster record is NEVER evidence of anything (`recon_mirror_is_the_record` false on an EMPTY digest is a refusal by name
#    with nothing moved); only a POSITIVE digest mismatch against the record at its home archives local state;
#  - a consuming phase with local recon evidence and no (or an incomplete) fetch record refuses by name — the recon PHASE is the bootstrap: it
#    runs where its record lives (--cluster), refuses a --cluster other than a recorded home, and writes the fetch record when it marks (idempotent
#    on the marker path for a recon marked before this record existed).
RECON_CLUSTER=""; RECON_CL_HOST=""; RECON_CL_ROOT=""; RECON_PAIRS_HOME=""; RECON_HOME_SOURCE=""
recon_fetch_record(){ echo "${RECON_FETCH:-$LOCAL_ROOT/recon_fetch.json}"; }
recon_home(){   # <phase>: sets RECON_CLUSTER / RECON_CL_HOST / RECON_CL_ROOT / RECON_PAIRS_HOME (+ RECON_HOME_SOURCE) — from the fetch record, else --cluster's
  local phase="${1:?recon_home needs the phase}" f c r pth
  RECON_CLUSTER="$CLUSTER"; RECON_CL_HOST="$CL_HOST"; RECON_CL_ROOT="$CL_ROOT"; RECON_PAIRS_HOME="$RECON_PAIRS"; RECON_HOME_SOURCE="--cluster"
  f="$(recon_fetch_record)"
  if [ ! -s "$f" ]; then
    # no fetch record: the recon phase is the bootstrap (it runs where its record lives and writes the record when it marks); a CONSUMING phase
    # holding local recon evidence has nothing to revalidate against and refuses — never a default to --cluster's copy of the path
    if [ "$phase" != recon ] && { [ -s "$RECON" ] || [ -f "$(recon_marker)" ]; }; then
      repro_fatal "$phase: the grading host holds a recon verdict / DONE marker but no $f names the record it came from — run \`--phase recon --cluster <the cluster the ladder ran on>\` once (it re-verifies the verdict against its record and writes the fetch record) before $phase; nothing moved"
    fi
    return 0
  fi
  read -r c r pth <<<"$("$PY" -c 'import json,sys; d=json.load(open(sys.argv[1])); print(" ".join(str(d.get(k) or "-") for k in ("cluster","record","pairs")))' "$f" 2>/dev/null || echo "- - -")"
  [ "$c" != - ] && [ "$r" != - ] && [ "$pth" != - ] || repro_fatal "$phase: $f is incomplete (needs cluster, record, pairs) — the verdict's home cannot be resolved; fix or remove the fetch record (then \`--phase recon --cluster <home>\` rewrites it); nothing moved"
  case " ${SISTEER_CLUSTERS:-$CLUSTER} " in *" $c "*) ;; *) repro_fatal "$phase: $f names its record on cluster '$c', not one of SISTEER_CLUSTERS (${SISTEER_CLUSTERS:-$CLUSTER}) — the verdict cannot be revalidated against its record; nothing moved";; esac
  if [ "$phase" = recon ] && [ "$c" != "$CLUSTER" ]; then
    repro_fatal "recon: the verdict's record lives on $c:$r ($f), but --cluster is $CLUSTER — the recon phase runs where its record lives: re-run --phase recon --cluster $c (design / trials may run on $CLUSTER; they revalidate the verdict against $c's record); nothing read, nothing archived, nothing submitted"
  fi
  # the home IS --cluster: this invocation's resolved host; another cluster: its block's CLUSTER_SSH (the one reader of a cluster block)
  RECON_CLUSTER="$c"; RECON_CL_ROOT="$r"; RECON_PAIRS_HOME="$pth"; RECON_HOME_SOURCE="recon_fetch.json"
  if [ "$c" = "$CLUSTER" ]; then RECON_CL_HOST="$CL_HOST"; else RECON_CL_HOST="$(cluster_var "$c" CLUSTER_SSH)"; fi
  [ -n "$RECON_CL_HOST" ] || repro_fatal "$phase: cluster '$c' (the verdict's home per $f) has no CLUSTER_SSH in its block — the record cannot be reached; nothing moved"
  [ "$RECON_CLUSTER" = "$CLUSTER" ] || repro_note "recon: the verdict's record lives on $RECON_CLUSTER:$RECON_CL_ROOT ($(basename "$f")); $phase runs on $CLUSTER and revalidates against that record"
}
recon_write_home(){   # the fetch record beside the mirrored verdict, written ATOMICALLY when the recon phase marks (and on the verified-marker path when absent):
  # {cluster, record, pairs, record_digest (the record's rung files + verdict as they are NOW), spec_fingerprint, written_utc}
  [ "$REPRO_DRY_RUN" = 1 ] && return 0
  local f d; f="$(recon_fetch_record)"
  d="$(ssh -o BatchMode=yes -o ConnectTimeout=30 "$RECON_CL_HOST" "cd '$RECON_CL_ROOT' 2>/dev/null && ls recon_[0-9]*.json recon.json >/dev/null 2>&1 && sha256sum recon_[0-9]*.json recon.json 2>/dev/null | sort | sha256sum | cut -d' ' -f1" 2>/dev/null | tr -dc 0-9a-f)"
  mkdir -p "$(dirname "$f")"
  printf '{"cluster": "%s", "record": "%s", "pairs": "%s", "record_digest": "%s", "spec_fingerprint": "%s", "written_utc": "%s"}\n' \
    "$RECON_CLUSTER" "$RECON_CL_ROOT" "$RECON_PAIRS_HOME" "$d" "${CUR_FP:-}" "$(date -u +%FT%TZ)" > "$f.tmp.$$" && mv -f "$f.tmp.$$" "$f" \
    || repro_fatal "recon: could not write the fetch record $f (nothing else changed)"
  repro_say "recon: the verdict's home recorded in $(basename "$f") — $RECON_CLUSTER:$RECON_CL_ROOT (record digest ${d:0:16}, pairs $(basename "$RECON_PAIRS_HOME"))"
}
recon_record_absent_refusal(){   # <phase>: an EMPTY record digest = no recon_<i>.json / recon.json at the home — never evidence; refuse with nothing moved
  repro_fatal "$1: no recon record at $RECON_CLUSTER:$RECON_CL_ROOT (no recon_<i>.json / recon.json there — the directory is absent or empty) while the grading host holds a verdict / DONE marker — an absent record is never evidence of a changed one; nothing moved. If the ladder ran elsewhere, $(recon_fetch_record) must name that cluster (\`--phase recon --cluster <home>\` rewrites it); if the record was removed on the cluster, that needs the coordinator"
}
recon_tune_args(){   # the recon CLI arguments every CPU helper (--fingerprint / --verify / --finalize) shares with the rungs — plus the CURRENT
  # pool identity (rows, pool + sidecar sha256 as read on the cluster) that --verify / --finalize bind admission to
  local extra=(); [ "$CLASS" != lora ] || [ ! -s "$ALLOWED" ] || { WP="$(widest_pair "$ALLOWED")" || repro_fatal "recon (lora): $ALLOWED certifies no pair — the probe's verdict excluded every (rank, set); re-judge the reference (--phase probe --reopen ...) before the LoRA recon"; read -r WR WM <<<"$WP"; extra=(--lora-rank "$WR" --lora-modules "$WM"); }
  [ -z "${POOL_SHA:-}" ] || extra+=(--pool-sha256 "$POOL_SHA" --rows "$ROWS"); [ -z "${SIDECAR_SHA:-}" ] || extra+=(--sidecar-sha256 "$SIDECAR_SHA")
  printf '%s\n' --config "$LOCAL_CFG" --pairs "$RECON_PAIRS_HOME" --si "$RECON_SI" --arm "$ARM" --batch-size 1 --out "$RECON" --lr-ladder "$LADDER" \
      --steps "$RECON_STEPS" --clip-percentile "$CLIP_PCT" --warmup-frac "$WARMUP" --init-scheme "$INIT_SCHEME" --passes "$CYCLED" "${extra[@]}"
}
recon_current_fp(){   # the canonical fingerprint of $SPEC — computed by tune_arm's own recon_spec_fingerprint (one implementation on both sides)
  local -a a; mapfile -t a < <(recon_tune_args)
  printf '%s' "$SPEC" | "$PY" "$TUNE" recon --fingerprint --spec - "${a[@]}"
}
recon_marker(){ echo "$(repro_state_dir "$MODEL")/$STAGE/DONE.$CLASS.recon"; }
recon_reset_local(){   # <why> [reason]: the local verdict, rung files and the DONE marker move to the mirror's superseded/<unique>/ (mv -n, never rm,
  # never over an earlier archival — one directory per transition) and the phase says so
  local reason="${2:-reset}" stamp; stamp="$(date -u +%Y%m%dT%H%M%SZ).$(date +%N).${reason//[^A-Za-z0-9_-]/_}"
  if [ "$REPRO_DRY_RUN" = 1 ]; then echo "DRY-RUN: archive the local recon verdict + DONE marker to $(disp "$LOCAL_ROOT")/superseded/UNIQUE/ ($1)"; return 0; fi
  [ -e "$LOCAL_ROOT/superseded/$stamp" ] && repro_fatal "recon: archival directory $LOCAL_ROOT/superseded/$stamp already exists — refusing to overwrite an earlier archival"
  mkdir -p "$LOCAL_ROOT/superseded/$stamp"
  local f moved=0; for f in "$RECON" "$LOCAL_ROOT"/recon_[0-9]*.json "$LOCAL_ROOT"/recon_[0-9]*.stamp.json "$(recon_marker)"; do
    [ -e "$f" ] || continue
    mv -n "$f" "$LOCAL_ROOT/superseded/$stamp/" && [ ! -e "$f" ] || repro_fatal "recon: $f was not moved into $LOCAL_ROOT/superseded/$stamp (a collision) — nothing overwritten, refusing"
    moved=1
  done
  [ "$moved" = 1 ] && repro_warn "recon: $1 — the local verdict / rung files / DONE marker moved to $LOCAL_ROOT/superseded/$stamp (nothing deleted)"
  fetch_invalidate "$CL_ROOT" "$LOCAL_ROOT"
  return 0
}
recon_local_record_digest(){   # the mirror's digest, computed exactly as the cluster's (sorted sha256sum list of recon_<i>.json + recon.json, hashed)
  ( cd "$LOCAL_ROOT" 2>/dev/null && ls recon_[0-9]*.json recon.json >/dev/null 2>&1 && sha256sum recon_[0-9]*.json recon.json 2>/dev/null | sort | sha256sum | cut -d' ' -f1 ) 2>/dev/null
}
recon_mirror_is_the_record(){   # 0 when the grading host mirror holds exactly the record's current rung files + verdict; a mirror that
  # lacks a file, holds a stale copy of one, or a record that changed since the fetch (a rung re-run or replaced on the cluster under the same spec
  # and pool) differs — and then the DONE marker and every local copy cache a SUPERSEDED verdict, whatever their internal consistency
  [ -n "$CLUSTER_RECORD_DIGEST" ] && [ "$(recon_local_record_digest)" = "$CLUSTER_RECORD_DIGEST" ]
}
recon_done_current(){   # the DONE marker is a CACHE of a VERIFIED verdict (hole 24): it counts only when it records the CURRENT
  # spec fingerprint AND the local verdict still verifies (spec fingerprint, every rung's stamp + result hash, the current pool identity, the
  # decision fields re-derived); a verdict certifying no clip is a fatal (never re-run the same ladder); anything else archives the local copies
  # and the phase re-derives from the cluster record
  local m; m="$(recon_marker)"
  [ -f "$m" ] || return 1
  # with a marker present, the record at the home must EXIST before the marker's fingerprint is even compared — an
  # empty digest is never evidence, whatever the marker says; a foreign-marker archive here used to precede this check, so a legacy recon (no
  # fetch record) re-run on another cluster with a reachable pool and a changed spec archived its valid rungs and went on to submit a ladder
  if [ "$REPRO_DRY_RUN" != 1 ] && [ -z "$CLUSTER_RECORD_DIGEST" ]; then recon_record_absent_refusal recon; fi
  if ! grep -qF -- "spec_fingerprint=$CUR_FP" "$m"; then
    recon_reset_local "the DONE marker records another reconnaissance ($(cut -c1-160 "$m" 2>/dev/null)), not the current spec ${CUR_FP:0:16}" foreign-marker
    return 1
  fi
  if [ "$REPRO_DRY_RUN" != 1 ] && ! recon_mirror_is_the_record; then
    recon_reset_local "the cluster record changed under the DONE marker (record digest ${CLUSTER_RECORD_DIGEST:0:16}, mirror $(recon_local_record_digest | cut -c1-16)) — a rung re-run or replaced since the last fetch; the marker caches a superseded verdict" stale-marker
    return 1                                                      # re-derive: the branch fetches the record's verdict and verifies THAT
  fi
  local vrc; recon_verify_verdict >/dev/null; vrc=$?
  case "$vrc" in
    0) repro_say "recon: DONE marker present and the local verdict re-verified as the current reconnaissance's (spec ${CUR_FP:0:16}) — skipping"
       [ -s "$(recon_fetch_record)" ] || recon_write_home          # a recon marked before the fetch record existed records its home now (idempotent)
       return 0;;
    3) repro_fatal "recon: the verdict the DONE marker caches is the current reconnaissance's but certifies NO clip — the ladder or the cell needs the coordinator; nothing resubmitted";;
    *) recon_reset_local "the DONE marker's verdict no longer verifies (tune_arm recon --verify rc $vrc: a replaced rung record, another pool copy, or an edited verdict)" stale-marker
       return 1;;
  esac
}
recon_verify_verdict(){   # the fetched $RECON against the current spec + the rung stamps beside it (tune_arm recon --verify) -> 0 verified
  local -a a; mapfile -t a < <(recon_tune_args)
  printf '%s' "$SPEC" | "$PY" "$TUNE" recon --verify --spec - "${a[@]}"
}
recon_revalidate_local(){   # <phase>: design / trials CONSUME the verdict — first revalidate what they hold against the CURRENT reconnaissance
  # (the probe phase's pattern): the marker records the current spec fingerprint and the local recon.json carries it; otherwise the local
  # copies are archived and the phase REFUSES by name until --phase recon re-derives (fetch + verify, or a new array)
  [ "$PROTOCOL" = fd6 ] || return 0
  [ -s "$RECON" ] || [ -f "$(recon_marker)" ] || return 0        # nothing local: the phase's own "needs $RECON" stands
  recon_home "$1"                                                             # the record WHERE IT LIVES (recon_fetch.json), never --cluster's copy of the path
  if [ "$REPRO_DRY_RUN" = 1 ]; then repro_note "recon: dry run — the local verdict is not revalidated against the current spec (live: tune_arm recon --fingerprint / --verify against $RECON_CLUSTER:$RECON_CL_ROOT before $1)"; return 0; fi
  recon_compose_spec; CUR_FP="$(recon_current_fp)" || repro_fatal "$1: cannot fingerprint the current recon spec"
  [ -n "$CLUSTER_RECORD_DIGEST" ] || recon_record_absent_refusal "$1"          # An absent / empty record is never evidence; nothing moved
  # the marker is a cache of a verified verdict: re-verified here on every consuming phase (hole 24) — the spec fingerprint, every rung's stamp
  # and result hash, the CURRENT pool identity, the decision fields re-derived from the rung records
  if ! recon_mirror_is_the_record; then
    recon_reset_local "$1 found the cluster record changed under the DONE marker (record digest ${CLUSTER_RECORD_DIGEST:0:16}, mirror $(recon_local_record_digest | cut -c1-16)) — a rung re-run or replaced since the last fetch" "stale-marker-at-$1"
    repro_fatal "$1: the recon verdict on record changed after it was fetched and marked (archived above) — re-run --phase recon before $1 (it re-fetches and re-verifies the record's current verdict)"
  fi
  local vrc=1; grep -qF -- "spec_fingerprint=$CUR_FP" "$(recon_marker)" 2>/dev/null && { recon_verify_verdict >/dev/null; vrc=$?; }
  [ "$vrc" != 3 ] || repro_fatal "$1: the recon verdict on record is the current reconnaissance's but certifies NO clip — no stable rung; the ladder or the cell needs the coordinator"
  if [ "$vrc" = 0 ]; then
    if [ "$1" = trials ] && [ -s "$R1" ]; then
      "$PY" - "$R1" "$RECON" <<'PYV' || repro_fatal "trials: the design $R1 was drawn from another recon verdict than $RECON (its recon.sha256 differs) — re-run --phase design"
import hashlib, json, sys
d = json.load(open(sys.argv[1])); r = (d.get("recon") or {})
want = r.get("sha256")
have = hashlib.sha256(open(sys.argv[2], "rb").read()).hexdigest()
sys.exit(0 if not want or want == have else 1)
PYV
    fi
    repro_say "recon: the local verdict re-verified as the current reconnaissance's (spec ${CUR_FP:0:16}) — $1 proceeds"; return 0
  fi
  recon_reset_local "$1 found a verdict that does not verify as the current reconnaissance's (another spec, a replaced rung record, another pool copy or an edited verdict; current spec ${CUR_FP:0:16})" "foreign-verdict-at-$1"
  repro_fatal "$1: the recon verdict on record does not verify as the current reconnaissance's (archived above) — re-run --phase recon before $1"
}

if want recon && [ "$PROTOCOL" = paper ]; then
  repro_note "recon: not applicable under the paper's protocol (the clip is a SEARCHED axis of the legacy lr x clip design)"
elif want recon; then
  probe_revalidate_local recon
  recon_home recon                                        # the recon phase runs where its record lives (a recorded home elsewhere refuses by name)
  recon_compose_spec                                       # the CURRENT reconnaissance: LADDER, SPEC, CYCLED (rows read live)
  CUR_FP="$(recon_current_fp)" || repro_fatal "recon: cannot fingerprint the current recon spec (output above)"
  # local recon EVIDENCE of any kind — the marker, a mirrored verdict, mirrored rung files — with NO recon record at the
  # home refuses by name before anything is compared, archived or submitted; when the home came from --cluster unproven (no fetch record), the ladder
  # may have run elsewhere and this cluster's empty directory proves nothing. A fresh reconnaissance has no local evidence and proceeds as before.
  if [ "$REPRO_DRY_RUN" != 1 ] && [ -z "$CLUSTER_RECORD_DIGEST" ] && { [ -f "$(recon_marker)" ] || [ -s "$RECON" ] || ls "$LOCAL_ROOT"/recon_[0-9]*.json >/dev/null 2>&1; }; then
    recon_record_absent_refusal recon
  fi
  if ! recon_done_current; then
    RECON_STATE=absent
    if cluster_has "$CL_ROOT/recon.json"; then
      # an existing verdict is accepted ONLY as THIS reconnaissance's: spec fingerprint + every rung's stamp + rung_fingerprints;
      # anything else is archived on both sides (never reused, never deleted) and the phase proceeds to (re)derive
      fetch "$CL_ROOT" "$LOCAL_ROOT"
      # a verdict is judged ONLY from a mirror that IS the record: the freshly fetched mirror's digest must equal the record's;
      # a mirror that is not the record (a skipped or failed fetch, a record that changed during this invocation) is not evidence of anything —
      # nothing is judged, nothing archived on the cluster, the phase says so and the next invocation re-composes
      recon_mirror_is_the_record || repro_fatal "recon: after the fetch the mirror is NOT the record (record digest ${CLUSTER_RECORD_DIGEST:0:16}, mirror $(recon_local_record_digest | cut -c1-16)) — the record changed during this invocation or the fetch did not run; nothing judged, nothing archived — re-run --phase recon"
      recon_verify_verdict; vrc=$?
      case "$vrc" in
        0)
          mark recon "$CL_ROOT/recon.json spec_fingerprint=$CUR_FP"; recon_write_home
          repro_say "recon: recon.json present on $CLUSTER and VERIFIED as the current reconnaissance's (spec ${CUR_FP:0:16}) — fetched to $RECON (clip $("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("clip"))' "$RECON" 2>/dev/null))"
          RECON_STATE=done;;
        3)
          # THIS reconnaissance's verdict, and it certifies no clip (every rung unstable): not a foreign verdict to archive and not a
          # reason to re-run the same ladder — the ladder or the cell needs the operator's decision (finalize's own exit status, preserved by --verify)
          repro_fatal "recon: the verdict on $CLUSTER is the current reconnaissance's but certifies NO clip — no stable rung on the ladder ($RECON records every rung's class); the ladder or the cell needs the coordinator; nothing marked, nothing resubmitted";;
        *)
          repro_warn "recon: the verdict on $CLUSTER ($CL_ROOT/recon.json) was made under ANOTHER reconnaissance (tune_arm recon --verify refused above, rc $vrc) — archiving it, never reusing it"
          recon_archive_record "$CL_ROOT" "$LOCAL_ROOT" foreign-verdict
          recon_reset_local "a verdict of another reconnaissance was on the record" foreign-verdict;;
      esac
    fi
    if [ "$RECON_STATE" = absent ]; then
      if cluster_job_active "$(name recon)"; then
        repro_note "recon: array $(name recon) is queued/running on $CLUSTER ($(cluster_count "$CL_ROOT/recon_[0-9]*.json")/$(echo "$LADDER" | tr ',' '\n' | wc -l) rung records so far) — re-run when its watcher reports DONE"
      elif [ "$(cluster_count "$CL_ROOT/recon_[0-9]*.json")" -gt 0 ]; then
        # the array is done (no job queued, no recon.json): the REDUCE — every rung's record to the local mirror, then
        # `tune_arm.py recon --finalize` (pure arithmetic over the records, no model, no GPU; every rung verified against the current
        # spec first) writes recon.json, pushed back beside the rung records so the cluster's record is whole and the branch
        # above (fetch + verify -> mark) holds on every re-run; a missing rung record refuses by index (re-run that task alone)
        N_RUNGS="$(echo "$LADDER" | tr ',' '\n' | wc -l)"
        fetch "$CL_ROOT" "$LOCAL_ROOT"
        mapfile -t TA < <(recon_tune_args)
        local_run "$PY" "$TUNE" recon --finalize --spec "$LOCAL_ROOT/recon_spec.json" "${TA[@]}"
        rc=$?
        [ -s "$RECON" ] || repro_fatal "recon: --finalize wrote no $RECON (rc $rc; a missing or foreign rung record is named above — re-run that array task under the current spec)"
        [ "$rc" = 0 ] || repro_fatal "recon: no stable rung on the ladder (rc $rc; $RECON records every rung's class) — the design cannot set a clip; the ladder or the cell needs the coordinator"
        push "$RECON" "$CL_ROOT/recon.json" && mark recon "$CL_ROOT/recon.json (reduced on the grading host from $N_RUNGS rung records) spec_fingerprint=$CUR_FP" && recon_write_home
        repro_say "recon: reduced $N_RUNGS rung records -> $RECON (clip $("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("clip"))' "$RECON" 2>/dev/null)); pushed to $CLUSTER:$CL_ROOT/recon.json"
      else
        if [ "$REPRO_DRY_RUN" = 1 ]; then echo "DRY-RUN: write $(disp "$LOCAL_ROOT")/recon_spec.json: $SPEC"; else repro_write "$LOCAL_ROOT/recon_spec.json" "$SPEC"; fi
        push "$LOCAL_ROOT/recon_spec.json" "$CL_ROOT/recon_spec.json"
        recon_archive_record "$CL_ROOT" "$LOCAL_ROOT" before-submit      # an earlier attempt's rung files never pool into this spec's clip
        # ONE array, one rung per task, one GPU each; --time is ONE rung's walltime (SISTEER_TIME_RECON)
        N_RUNGS="$(echo "$LADDER" | tr ',' '\n' | wc -l)"
        submit "$(unit recon)" "$(name recon)" scripts/sbatch/hpo_trial.sbatch "$CL_ROOT/recon_spec.json $CL_ROOT $ARM $CL_CONFIG $RECON_PAIRS" "$RECON_TIME" --registers none --array "0-$((N_RUNGS - 1))" --env "HPO_JOB_MODE=recon"
        [ "$PHASE" = all ] && { repro_say "waiting on the reconnaissance — re-run --phase all when it is DONE"; exit 0; }
      fi
    fi
  fi
fi

# ================================================================================================================ design
if want design && ! done_or_skip design; then
  probe_revalidate_local design
  recon_revalidate_local design
  SID_FLAG=(--campaign "$CAMPAIGN" --model-id "$FIT_ID")
  if [ "$PROTOCOL" = paper ]; then
    # the paper's own design: lr x clip Sobol (the seed box's wide clip range) over the two warm-up levels, 16 trials
    local_run "$PY" "$TUNE" emit-trials --design legacy --out "$R1" --arm "$ARM" --config "$LOCAL_CFG" --position all --si "$HPO_SI" \
        --warmup-levels 0.05,0.15 --n "$N_TRIALS" --trial-epochs "$TRIAL_EPOCHS" "${SID_FLAG[@]}"
  else
    if [ "$REPRO_DRY_RUN" != 1 ]; then
      [ -s "$RECON" ] || repro_fatal "design needs $RECON (run --phase recon first)"
      [ "$CLASS" != lora ] || [ -s "$ALLOWED" ] || repro_fatal "design (lora) needs $ALLOWED (run --phase probe first)"
    fi
    TOP="$( [ -s "$RECON" ] && box_top_from_recon "$RECON" )"
    [ -z "$TOP" ] || repro_note "design: the recon certified log10(lr) <= $TOP — the box top is lowered to it (an unstable rung never enters the search)"
    common=(--out "$R1" --arm "$ARM" --config "$LOCAL_CFG" --position all --si "$HPO_SI" --warmup-levels "$WARMUP" --init-scheme "$INIT_SCHEME" --trial-epochs "$TRIAL_EPOCHS" "${SID_FLAG[@]}")
    [ -z "$TOP" ] || common+=(--lr-log10-max "$TOP")
    if [ "$CLASS" = lora ]; then
      local_run "$PY" "$TUNE" emit-trials --design sobol16_coverage --allowed-pairs "$ALLOWED" --recon "$RECON" --n "$N_TRIALS" "${common[@]}"
    else
      local_run "$PY" "$TUNE" emit-trials --design legacy --recon "$RECON" --n "$N_TRIALS" "${common[@]}"
    fi
  fi
  rc=$?
  if [ "$rc" = 0 ] && [ "$REPRO_DRY_RUN" != 1 ]; then
    SID="$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("search_id") or "")' "$R1")"
    [ -z "$SID" ] || repro_registry_id fit_search "$SID"
    push "$R1" "$CL_ROOT/r1_trials.json" && mark design "$R1 search_id=$SID"
  elif [ "$REPRO_DRY_RUN" = 1 ]; then
    push "$R1" "$CL_ROOT/r1_trials.json"
  else
    repro_fatal "design: emit-trials failed (output above)"
  fi
fi

# ================================================================================================================ trials
if want trials && ! done_or_skip trials; then
  probe_revalidate_local trials
  recon_revalidate_local trials
  N="$( [ -s "$R1" ] && "$PY" -c 'import json,sys; print(len(json.load(open(sys.argv[1]))["trials"]))' "$R1" || echo "$N_TRIALS" )"
  have="$(n_summaries 1)"
  if [ "$have" -ge "$N" ] 2>/dev/null; then
    repro_say "trials: $have/$N round-1 summaries on $CLUSTER — complete"; mark trials "$have summaries"
  elif cluster_job_active "$(name r1)"; then
    repro_note "trials: array $(name r1) is queued/running on $CLUSTER ($have/$N summaries so far) — re-run when its watcher reports DONE"
    [ "$PHASE" = all ] && exit 0
  else
    [ "$REPRO_DRY_RUN" = 1 ] || [ -s "$R1" ] || repro_fatal "trials needs $R1 (run --phase design first)"
    submit_round 1 "0-$((N - 1))"
    [ "$PHASE" = all ] && { repro_say "waiting on the $N trials — re-run --phase all when the array is DONE"; exit 0; }
  fi
fi

# ================================================================================================================ analyze
if want analyze; then
  fetch "$CL_ROOT" "$LOCAL_ROOT"; frc=$?
  source_inventory_check analyze "$frc"                     # the fit is derived from the mirror — it must BE this record's evidence
  AKEY="$(inventory_key)"; AMARK="$(repro_state_dir "$MODEL")/$STAGE/DONE.$CLASS.analyze"
  # the DONE marker is keyed on the evidence the fit was derived from (summaries AND sidecars): a retried trial or a metrics backfill re-derives
  # the fit instead of skipping it; a dry run keeps the plain marker semantics (the record is not read)
  if repro_done "$MODEL" "$STAGE" "$CLASS.analyze" && { [ "$REPRO_DRY_RUN" = 1 ] || grep -qF -- "set=$AKEY" "$AMARK" 2>/dev/null; }; then
    repro_say "analyze: DONE marker present for this evidence (set=$AKEY) — skipping"
  else
    [ "$REPRO_DRY_RUN" = 1 ] || ! repro_done "$MODEL" "$STAGE" "$CLASS.analyze" \
      || repro_say "analyze: the DONE marker records other evidence than the record holds now (a retried trial or a metrics backfill; set=$AKEY) — the fit is re-derived"
    if [ "$PROTOCOL" = paper ]; then
      local_run "$PY" "$TUNE" analyze --out "$LOCAL_ROOT" --rounds 1
    else
      local_run "$PY" "$TUNE" analyze --design v2 --out "$LOCAL_ROOT" --rounds 1 "${AP[@]}"
    fi
    rc=$?
    if [ "$REPRO_DRY_RUN" != 1 ]; then
      [ -s "$FIT1" ] || repro_fatal "analyze wrote no $FIT1 (output above)"
      [ "$rc" = 0 ] || repro_note "analyze: the design fit has no interior fitted minimum (rc $rc) — the fall-through (best measured promotable trial) is filed in $FIT1; the refine round centres on it"
      mark analyze "$FIT1 set=$AKEY"
    fi
  fi
fi

# ================================================================================================================ refine (3 + 1)
if want refine && [ "$PROTOCOL" = paper ] && ! done_or_skip refine; then
  # the paper's recentre / confirm policy: scripts/sdfbase/hpo_next_round.py (recenter round 2 / confirmation round 9 / stop)
  NR="$REPRO_REPO/scripts/sdfbase/hpo_next_round.py"
  DRY=(); [ "$REPRO_DRY_RUN" = 1 ] && DRY=(--dry-run)
  local_run "$PY" "$NR" --fit "$FIT1" --config "$LOCAL_CFG" --model-key "$FIT_KEY" --arm "$ARM" --si "$HPO_SI" --out-dir "$LOCAL_ROOT" "${DRY[@]}"
  repro_note "refine (paper): the emitted design under $LOCAL_ROOT (r2_*.json / r9_*.json) is submitted with scripts/sbatch/hpo_trial.sbatch exactly as round 1 (push + submit --array 0-<n-1>); re-run --phase analyze --rounds as hpo_next_round directs"
elif want refine; then
  if repro_done "$MODEL" "$STAGE" "$CLASS.refine"; then
    # COMPLETED: the pair refit r1+2_fit_pair.json must stand over the CURRENT evidence — a retried trial or a metrics backfill
    # since (re-derived by --phase analyze into r1_fit.json) would otherwise strand it, and table / pick refuse it forever. Refit ONLY: nothing
    # emitted, pushed or submitted; the marker is re-stamped.
    probe_revalidate_local refine
    PF0="$(probe_owned_fingerprint)"; fetch "$CL_ROOT" "$LOCAL_ROOT"; frc=$?; source_inventory_check refine "$frc" "$PF0"
    VOUT="$(verify_fits refine 2>&1)"; vrc=$?; printf '%s\n' "$VOUT"
    if [ "$REPRO_DRY_RUN" = 1 ] || [ "$vrc" = 0 ]; then
      repro_say "refine: DONE marker present and every fit stands over the current evidence — skipping"
    elif grep -q '^\[hpo-table\] r1_fit.json ' <<<"$VOUT"; then
      repro_fatal "refine: the DESIGN fit $FIT1 itself predates the current evidence (its reason is above) — re-run --phase analyze first (it re-derives the fit over the current evidence), then --phase refine refreshes the pair refit"
    else
      repro_say "refine: the pair refit $FIT12 predates the current evidence (its reason is above) — re-deriving it over rounds 1+2 (refit only; nothing resubmitted)"
      CLOSE=(); [ "$CLASS" = relu ] && CLOSE=(--close-search)
      refit "${CLOSE[@]}"; [ -s "$FIT12" ] || repro_fatal "refine: the pair-only refit wrote no $FIT12"
      verify_fits refine || repro_fatal "refine: the re-derived pair refit still does not stand over the current evidence (its reason is above) — re-run --phase analyze first, then --phase refine"
      mark refine "refreshed over the current evidence; $FIT12"
    fi
  else
    probe_revalidate_local refine
    if [ "$REPRO_DRY_RUN" = 1 ]; then
      repro_note "refine advances ONE step per live invocation from the cluster's state; the chain is:"
      emit_from --refine-from "$FIT1" "$R2"
      push "$R2" "$CL_ROOT/r2_trials.json"
      submit_round 2 0-2
      refit
      emit_from --confirm-from "$FIT12" "$R9"
      repro_note "  (or, on the fallback path: emit-trials --explore-from $FIT12 --out $R3 — the exploration replicate, round 3, never a confirmation)"
      submit_round 9 0-0
      CLOSE=(); [ "$CLASS" = relu ] && CLOSE=(--close-search)
      refit "${CLOSE[@]}"
    else
      [ -s "$FIT1" ] || repro_fatal "refine needs $FIT1 (run --phase analyze first)"
      if ! cluster_has "$CL_ROOT/r2_trials.json"; then
        emit_from --refine-from "$FIT1" "$R2" || repro_fatal "refine: emit-trials --refine-from failed"
        push "$R2" "$CL_ROOT/r2_trials.json"
        submit_round 2 0-2
        repro_say "refine: the three refine trials are submitted — re-run when the array is DONE"; exit 0
      fi
      have2="$(n_summaries 2)"
      if [ "$have2" -lt 3 ]; then
        cluster_job_active "$(name r2)" && { repro_note "refine: array $(name r2) still running ($have2/3 summaries)"; exit 0; }
        repro_fatal "refine: only $have2/3 round-2 summaries on $CLUSTER and no job queued — inspect the array's logs"
      fi
      fetch "$CL_ROOT" "$LOCAL_ROOT"
      FOURTH=""; [ -s "$R9" ] && FOURTH=9; [ -s "$R3" ] && FOURTH=3
      if [ -z "$FOURTH" ]; then
        refit; rc=$?
        [ -s "$FIT12" ] || repro_fatal "refine: the pair-only refit wrote no $FIT12"
        ACTION="$("$PY" -c 'import json,sys; print((json.load(open(sys.argv[1])).get("next") or {}).get("action"))' "$FIT12")"
        case "$ACTION" in
          confirm) emit_from --confirm-from "$FIT12" "$R9" || repro_fatal "refine: --confirm-from failed"
                   push "$R9" "$CL_ROOT/r9_trials.json"; submit_round 9 0-0
                   repro_say "refine: PD interior refit — the exact CONFIRMATION (round 9) is submitted; re-run when DONE";;
          explore) emit_from --explore-from "$FIT12" "$R3" || repro_fatal "refine: --explore-from failed"
                   push "$R3" "$CL_ROOT/r3_trials.json"; submit_round 3 0-0
                   repro_say "refine: FALLBACK path (the refit is not PD or its argmin left the refined box) — the EXPLORATION REPLICATE (round 3) is submitted; re-run when DONE";;
          *) repro_fatal "refine: the refit's next.action is '$ACTION' (expected confirm|explore)";;
        esac
        exit 0
      fi
      if [ "$(n_summaries "$FOURTH")" -lt 1 ]; then
        cluster_job_active "$(name "r$FOURTH")" && { repro_note "refine: the fourth trial (round $FOURTH) is still running"; exit 0; }
        repro_fatal "refine: no round-$FOURTH summary on $CLUSTER and no job queued — inspect the job's log"
      fi
      fetch "$CL_ROOT" "$LOCAL_ROOT"
      CLOSE=(); [ "$CLASS" = relu ] && CLOSE=(--close-search)
      refit "${CLOSE[@]}"
      SID="$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("search_id") or "")' "$FIT12")"
      [ -z "$SID" ] || [ "$CLASS" != relu ] || repro_registry_id fit_search "$SID"
      mark refine "fourth trial round $FOURTH; $FIT12"
      [ "$CLASS" != relu ] || repro_note "relu: the search is closed with the refit's winner (fit_lr_selected_by = search); make_sdfbase_configs.py --fold-hpo folds it"
    fi
  fi
fi

# ================================================================================================================ table
table_set_key(){   # <record inventory>: the table's GENERATION: sha256 over the SOURCE RECORD identity
  # ($CLUSTER:$CL_ROOT), the menu's sha + result generation, every trial summary's PATH AND CONTENT as hashed ON THE RECORD HOST (the inventory
  # source_inventory_check verified the mirror against — a retried trial replacing rN_tNN/summary.json in place changes it), and the mirror's fit /
  # refit artefacts' bytes — the same filenames fetched from another cluster never read as "already published"; empty when nothing is there
  ( cd "$LOCAL_ROOT" 2>/dev/null && {
      echo "source=$CLUSTER:$CL_ROOT"
      [ -s "$ALLOWED" ] && { sha256sum "$ALLOWED" | cut -d' ' -f1; "$PY" -c 'import json,sys; print("gen=" + str(json.load(open(sys.argv[1])).get("result_generation")))' "$ALLOWED" 2>/dev/null; }
      printf '%s\n' "${1:-}"
      for f in r1_fit.json r1+2_fit_pair.json r1_fit_menu.json hpo_pick.json hpo_pick.HOLD.json; do [ -s "$f" ] && { printf '%s ' "$f"; sha256sum "$f" | cut -d' ' -f1; }; done   # the pick's state is table content (the HOLD banner)
    } | sha256sum | cut -c1-16 ) 2>/dev/null
}
if want table; then
  if [ -n "$MENU_OVERRIDE" ] && [ -s "$OVERRIDE_MENU" ]; then
    menu_override_authorize "$("$PY" -c 'import json,sys; print(",".join(f"{a}:{b}" for a, b in json.load(open(sys.argv[1]))["menu_override"]["pairs"]))' "$OVERRIDE_MENU")"
    AP=(--allowed-pairs "$OVERRIDE_MENU"); repro_say "table: MENU OVERRIDE $MENU_OVERRIDE — the table's menu is $OVERRIDE_MENU (authorized by $MENU_OVERRIDE_RECORD); the probe record is not revalidated here"
  elif [ -n "$MENU_OVERRIDE" ]; then repro_fatal "table: --menu-override $MENU_OVERRIDE names no override menu at $OVERRIDE_MENU (run --phase addendum --menu-override first)"
  else probe_revalidate_local table; fi                              # the menu the table pivots over (and keys on) must be the probe record's
                                                            # CURRENT verdict, read at the record's HOME (probe_fetch.json) — a re-verdicted probe on the
                                                            # other cluster refuses the table until --phase probe re-fetches; before any fetch / key / render
  PF0="$(probe_owned_fingerprint)"                        # the revalidated menu / verdict / geometry / fetch record: the search fetch must leave them untouched
  fetch "$CL_ROOT" "$LOCAL_ROOT"; frc=$?
  source_inventory_check table "$frc" "$PF0"                # sets INV; refuses a failed / partial fetch, a mirror that is not this record's, a changed menu
  TKEY="$(table_set_key "$INV")"
  if [ "$REPRO_DRY_RUN" != 1 ] && repro_done "$MODEL" "$STAGE" "$CLASS.table" && grep -qF -- "set=$TKEY" "$(repro_state_dir "$MODEL")/$STAGE/DONE.$CLASS.table" 2>/dev/null; then
    repro_say "table: DONE marker present for this trial set + menu (set=$TKEY) — skipping"
  else
    [ "$REPRO_DRY_RUN" = 1 ] || ! repro_done "$MODEL" "$STAGE" "$CLASS.table" || repro_say "table: the DONE marker records another trial set / menu than the mirror holds now (set=$TKEY) — rebuilding"
    local_run "$PY" "$REPRO_REPO/scripts/sdfbase/hpo_table.py" --search-dir "$LOCAL_ROOT" "${AP[@]}" --source-record "$CLUSTER:$CL_ROOT" --out "$LOCAL_ROOT/table"
    rc=$?
    if [ "$REPRO_DRY_RUN" = 1 ]; then
      echo "DRY-RUN: ssh $CL_HOST 'mkdir -p $FIG_DIR' && rsync -a $(disp "$LOCAL_ROOT")/table/ $CL_HOST:$FIG_DIR/"
    elif [ "$rc" = 0 ]; then
      ssh -o BatchMode=yes "$CL_HOST" "mkdir -p $FIG_DIR" && rsync -a "$LOCAL_ROOT/table/" "$CL_HOST:$FIG_DIR/" \
        && repro_say "table: $LOCAL_ROOT/table/table.{csv,md,html} -> $CLUSTER:$FIG_DIR (mirrored to the figure server from there)" \
        && mark table "$FIG_DIR set=$TKEY"
    else
      repro_fatal "table: hpo_table.py failed (output above)"
    fi
  fi
fi

# ================================================================================================================ pick
if want pick; then
  PICK_MENU="$ALLOWED"
  if [ -n "$MENU_OVERRIDE" ] && [ -s "$OVERRIDE_MENU" ]; then
    # the pick over the override menu (the addendum pairs) — the override doc bound by --phase addendum IS the menu the pick, its
    # fit verification and hpo_pick's menu-fit binding read; the probe record is not revalidated here (as for the table under the override).
    # The override is RE-AUTHORIZED here against the committed record on the menu's recorded pair set, exactly as
    # the table does — a record removed or revised after the addendum bound the menu, or a checkout without it, refuses by name before any fetch
    # or promotion (a pick under an unauthorized override would bypass the probe-menu gate and write models.yaml / hpo_pick.json / a ruling)
    menu_override_authorize "$("$PY" -c 'import json,sys; print(",".join(f"{a}:{b}" for a, b in json.load(open(sys.argv[1]))["menu_override"]["pairs"]))' "$OVERRIDE_MENU")"
    PICK_MENU="$OVERRIDE_MENU"; AP=(--allowed-pairs "$OVERRIDE_MENU")
    repro_say "pick: MENU OVERRIDE $MENU_OVERRIDE — the pick's menu is $OVERRIDE_MENU (authorized by $MENU_OVERRIDE_RECORD); the probe record is not revalidated here"
  elif [ -n "$MENU_OVERRIDE" ]; then repro_fatal "pick: --menu-override $MENU_OVERRIDE names no override menu at $OVERRIDE_MENU (run --phase addendum --menu-override first — it binds the menu)"
  else probe_revalidate_local pick
  fi
  if [ "$CLASS" != lora ]; then
    [ "$PHASE" = pick ] && repro_fatal "--phase pick is the LoRA architecture pick; the ReLU search selects its lr itself (fit_lr_selected_by = search)"
  elif [ "$PROTOCOL" = paper ]; then
    repro_note "pick: not applicable under the paper's protocol — rank 4 / o_proj is the paper's architecture (FD 18); the searched lr is folded by make_sdfbase_configs.py --fold-hpo"
  else
    # EVIDENCE FIRST: the pick reads the trial directories and the fits over them — the record is fetched and proven
    # to be the mirror's, and a COMPLETED pick is held to the current evidence before its DONE marker may skip anything
    PF0="$(probe_owned_fingerprint)"                      # the revalidated menu must survive the search fetch untouched
    fetch "$CL_ROOT" "$LOCAL_ROOT"; frc=$?
    source_inventory_check pick "$frc" "$PF0"             # the pick reads the trial directories — they must BE this record's
    if repro_done "$MODEL" "$STAGE" "$CLASS.pick" && [ ! -e "$PICK_HOLD" ]; then
      # the completed PROMOTION is re-evaluated against the current evidence — every fit bound AND the recorded
      # (pair, lr) still selected AND the clip gate at the promoted coordinates still promotable over every round's trials (a backfilled
      # confirming sidecar would otherwise leave the fold in force)
      VOUT="$(verify_fits pick 2>&1)" && VPOUT="$(local_run "$PY" "$REPRO_REPO/scripts/sdfbase/hpo_pick.py" --search-dir "$LOCAL_ROOT" "${AP[@]}" --source-record "$CLUSTER:$CL_ROOT" --verify-promotion 2>&1)" && VPOK=1 || VPOK=0
      printf '%s\n' "${VOUT:-}" "${VPOUT:-}" | grep . || true
      if [ "$VPOK" = 1 ]; then
        repro_say "pick: DONE marker present, every fit stands over the current evidence and the recorded promotion still stands — skipping"
      else
        HOLD_WHY="$(printf '%s\n' "${VOUT:-}" "${VPOUT:-}" | grep . | grep -v '^\[hpo-table\] [0-9]* trial rows' | tail -3 | tr '\n' ' ' | cut -c1-900)"
        # the PROMOTED pick's basis changed (a retried trial or a metrics backfill after the promotion) — the promotion is HELD, visibly:
        # the hold file beside hpo_pick.json makes every consumer refuse by name (04's LoRA fits, make_sdfbase_configs --fold-hpo, the table);
        # there is no un-fold, so nothing is reverted here; only a fresh --phase pick that re-derives a promotion from the current evidence clears it
        local_run "$PY" "$REPRO_REPO/scripts/sdfbase/hpo_pick.py" --search-dir "$LOCAL_ROOT" --write-hold "the promotion no longer stands over the current evidence (verified $(date -u +%FT%TZ)): ${HOLD_WHY:-a fit no longer stands over the current trial evidence (hpo_table --verify-only: see the pick phase output)}; re-run --phase analyze (and --phase refine / --phase addendum), then --phase pick --rank R --set S"
        repro_fatal "pick: the PROMOTED pick's basis has CHANGED — HOLD written at $PICK_HOLD; every consumer refuses by name while it stands (04_fit_objects.sh LoRA fits, make_sdfbase_configs.py --fold-hpo, the table's promotion); re-run --phase analyze (and --phase refine / --phase addendum, which refresh their fits), then --phase pick --rank R --set S to re-derive the promotion from the current evidence"
      fi
    elif [ -z "$PICK_RANK" ] || [ -z "$PICK_SET" ]; then
      msg="pick needs --rank R --set S: the LoRA architecture is the USER'S choice from the published table (FD 6a) — no default is applied in their absence"
      [ ! -e "$PICK_HOLD" ] || msg="pick is HELD ($PICK_HOLD) — a fresh pick from the current evidence needs --rank R --set S (the hold clears only on a promotion)"
      if [ "$PHASE" = all ]; then repro_note "$msg; table at $LOCAL_ROOT/table and $CLUSTER:$FIG_DIR"; else repro_fatal "$msg"; fi
    else
      if [ -e "$PICK_HOLD" ]; then      # the held re-pick passes the COMPLETE fit verification (the widened-menu refit included) before anything
        repro_say "pick: HELD ($PICK_HOLD) — this pick re-derives from the current evidence; a promotion clears the hold, anything else leaves it standing"
        verify_fits pick || repro_fatal "pick: HELD, and a fit does not stand over the current evidence (its reason is above) — re-run --phase analyze (and --phase refine / --phase addendum) first; the hold stands"
      fi
      PICK_ARGS=(--search-dir "$LOCAL_ROOT" --rank "$PICK_RANK" --set "$PICK_SET" --allowed-pairs "$PICK_MENU" --source-record "$CLUSTER:$CL_ROOT"
                 --models-table "$(repro_models_table)" --model-key "$MODEL" --fit-key "$FIT_KEY" --fit-config "$LOCAL_CFG" --table-url "$FIG_DIR/table.html")
      if [ "$REPRO_DRY_RUN" = 1 ]; then
        # the PREVIEW: hpo_pick runs READ-ONLY (no --execute — nothing written, no registry call, models.yaml untouched) over the mirror as it
        # stands, so the operator sees its decision (promoted through which confirmation / confirm-pick requested / refused) before the live run
        echo "DRY-RUN: hpo_pick.py runs read-only below (no --execute): its decision is the preview; nothing is written"
        "$PY" "$REPRO_REPO/scripts/sdfbase/hpo_pick.py" "${PICK_ARGS[@]}" || repro_note "pick (dry-run): hpo_pick.py refused (its reason is above) — the live run would refuse the same way"
      else
        local_run "$PY" "$REPRO_REPO/scripts/sdfbase/hpo_pick.py" "${PICK_ARGS[@]}" --execute || repro_fatal "pick: hpo_pick.py refused (its reason is above)"
      fi
      if [ "$REPRO_DRY_RUN" != 1 ]; then
        # promoted or not is hpo_pick.json's word: only a promoted pick folds the config and marks the phase DONE; a pick
        # with no clip evidence at its pair is recorded but waits for its confirm-pick trial
        PROMOTED="$("$PY" -c 'import json,sys; print(1 if json.load(open(sys.argv[1])).get("promoted") else 0)' "$LOCAL_ROOT/hpo_pick.json" 2>/dev/null || echo 0)"
        if [ "$PROMOTED" = 1 ]; then
          if local_run "$PY" "$REPRO_REPO/scripts/sdfbase/make_sdfbase_configs.py" --models "$FIT_KEY" --fold-hpo "$LOCAL_ROOT/hpo_fold.json"; then
            mark pick "rank $PICK_RANK set $PICK_SET"
            repro_note "commit reproduce/models.yaml + configs/sdfbase_folds.json + configs/sdfbase_${FIT_KEY}_fits.yaml; 04_fit_objects.sh --object *_lora is unblocked"
          else
            # the promotion cleared the hold, and the fold then failed — the folded config is not in a trusted state; the HOLD is
            # RESTORED (every consumer refuses again) and the phase fails by name
            local_run "$PY" "$REPRO_REPO/scripts/sdfbase/hpo_pick.py" --search-dir "$LOCAL_ROOT" --write-hold "the fold (make_sdfbase_configs.py --fold-hpo) FAILED after the promotion of rank $PICK_RANK / set $PICK_SET cleared the previous hold — the folded config is not trusted; fix the fold's refusal (above) and re-run --phase pick --rank $PICK_RANK --set $PICK_SET"
            repro_fatal "pick: the fold failed after the promotion (its reason is above) — HOLD restored at $PICK_HOLD; every consumer refuses by name until a fresh --phase pick re-derives the promotion and its fold succeeds"
          fi
        else
          repro_note "pick: rank $PICK_RANK set $PICK_SET is RECORDED (the ruling) but NOT promoted — no measured clip evidence at that pair's learning rate; its confirm-pick trial is requested ($PICK_REQ): --phase confirm-pick submits it, then re-run --phase pick --rank $PICK_RANK --set $PICK_SET when its summary lands"
        fi
      fi
    fi
  fi
fi

# ================================================================================================================ addendum (D409)
# ONE trial per (rank, set) pair the serving probe certified AFTER the round-1 design was drawn — the rule of 2026-09-30: the
# round-1 table is published first; these rows are an ADDENDUM at the lr the design's own refit predicts for each new pair. Explicit only.
if [ "$PHASE" = addendum ]; then
addendum_menu_refit(){   # the widened-menu refit: r1_fit_menu.json re-derived from the mirror's round-1 evidence over the
  # CURRENT menu; the canonical r1_fit.json is sha-guarded and restored on any change; only the recognized non-PD verdict is an accepted nonzero.
  # Shared by the first submission and by the completed phase's refit-only refresh.
  FIT1_SHA_BEFORE="$( [ -s "$FIT1" ] && sha256sum "$FIT1" | cut -d' ' -f1 )"
  [ -n "$FIT1_SHA_BEFORE" ] || [ "$REPRO_DRY_RUN" = 1 ] || repro_fatal "addendum: the canonical round-1 artefact $FIT1 is missing or empty — nothing refit"
  [ "$REPRO_DRY_RUN" = 1 ] || [ ! -e "$FIT1M" ] || { mv -n "$FIT1M" "$FIT1M.superseded.$(date -u +%Y%m%dT%H%M%SZ).json" && repro_say "addendum: an earlier widened refit moved aside (never reused)"; }
  local_run "$PY" "$TUNE" analyze --design v2 --out "$LOCAL_ROOT" --rounds 1 --allowed-pairs "$ADDENDUM_MENU" --extends-menu --fit-name "$(basename "$FIT1M")" --source-record "$CLUSTER:$CL_ROOT"
  arc=$?
  if [ "$REPRO_DRY_RUN" != 1 ]; then
    # finally-style: the canonical fit first, whatever the refit did
    FIT1_SHA_AFTER="$( [ -s "$FIT1" ] && sha256sum "$FIT1" | cut -d' ' -f1 )"
    if [ "$FIT1_SHA_AFTER" != "$FIT1_SHA_BEFORE" ]; then
      repro_warn "addendum: the canonical round-1 artefact $FIT1 CHANGED under the refit (sha ${FIT1_SHA_BEFORE:0:12} -> ${FIT1_SHA_AFTER:0:12}) — restoring it from the record before anything else"
      rsync -a "$CL_HOST:$CL_ROOT/r1_fit.json" "$FIT1" && [ "$(sha256sum "$FIT1" | cut -d' ' -f1)" = "$FIT1_SHA_BEFORE" ] \
        || repro_fatal "addendum: the canonical round-1 artefact could not be restored to its sha ${FIT1_SHA_BEFORE:0:12} from $CL_HOST:$CL_ROOT/r1_fit.json — the pick's basis is compromised; nothing submitted"
    fi
    if [ "$arc" != 0 ]; then
      "$PY" - "$FIT1M" "$ADDENDUM_MENU" "$FIT1" <<'PYB' || { [ -e "$FIT1M" ] && mv -f "$FIT1M" "$FIT1M.failed.$(date -u +%Y%m%dT%H%M%SZ).json"; repro_fatal "addendum: the widened-menu refit exited $arc and left no recognized non-PD artefact (its reason is above) — a failure, not the fallback; nothing submitted"; }
import json, sys
try:
    art = json.load(open(sys.argv[1]))
except Exception as e:
    sys.exit(f"no readable artefact at {sys.argv[1]}: {e}")
menu = json.load(open(sys.argv[2])); design = json.load(open(sys.argv[3]))
want = sorted((int(p["rank"]), str(p["set"])) for p in menu.get("pairs") or [])
have = sorted((int(a), str(b)) for a, b in (art.get("allowed_pairs") or []))
drawn = sorted((int(a), str(b)) for a, b in (design.get("allowed_pairs") or []))
ext_from = sorted((int(a), str(b)) for a, b in (art.get("menu_extended_from") or []))
fit = art.get("fit") or {}
for cond, why in ((art.get("design") == "v2", f"design {art.get('design')!r} is not v2"),
              (have == want, f"the artefact's menu {have} is not the widened menu {want}"),
              (ext_from == drawn, f"menu_extended_from {ext_from} is not the design's menu {drawn}"),
              (fit.get("positive_definite") is False, f"fit.positive_definite is {fit.get('positive_definite')!r}, not the recognized non-PD verdict"),
              (bool(art.get("rows")), "no rows"),
              ((art.get("selection") or {}).get("selected_from") == "best_raw_trial", f"selection.selected_from is {(art.get('selection') or {}).get('selected_from')!r}")):
    if not cond: sys.exit(why)
print(f"addendum: the refit is the recognized NON-positive-definite verdict over the widened menu (alarms {(art.get('selection') or {}).get('alarms')}) — the fallback state: each new pair runs at its set's best round-1 trial lr")
PYB
    fi
    [ -s "$FIT1M" ] || repro_fatal "addendum: the refit wrote no artefact at $FIT1M"
    "$PY" -c 'import json,sys; a=json.load(open(sys.argv[1])); sys.exit(0 if a.get("menu_extended_from") else "the refit records no menu extension — --extends-menu did not take; is the local menu the widened one?")' "$FIT1M" \
      || repro_fatal "addendum: $FIT1M is not a widened-menu refit (its reason is above); nothing submitted"
  else
    echo "DRY-RUN: the refit writes $(disp "$FIT1M") (never $(disp "$FIT1")); $(disp "$FIT1")'s sha is checked before/after and restored from $CL_HOST:$CL_ROOT on any change; a nonzero refit is accepted ONLY as the recognized non-PD artefact (the fallback state)"
  fi
}
  [ "$CLASS" = lora ] && [ "$PROTOCOL" = fd6 ] || repro_fatal "--phase addendum belongs to the FD 6 LoRA search (class=$CLASS protocol=$PROTOCOL)"
  if [ -n "$MENU_OVERRIDE" ]; then
    menu_override_authorize "$ADDENDUM_PAIRS"            # the exact approved set, or nothing
    repro_say "addendum: MENU OVERRIDE $MENU_OVERRIDE — the serving-probe menu gate is overridden by the user for exactly --pairs $ADDENDUM_PAIRS (authorized by $MENU_OVERRIDE_RECORD); the probe record is not revalidated here"
    [ -s "$LOCAL_ROOT/r1_trials.json" ] || repro_fatal "addendum: --menu-override needs the round-1 design $LOCAL_ROOT/r1_trials.json on the mirror (its pair details reconstruct the menus)"
    RM="$(override_recon_menu)" || repro_fatal "addendum: the menu the reconnaissance ran on cannot be resolved (its reason is above)"
    RECON_MENU="${RM#*$'\t'}"; repro_say "addendum: the recon revalidation reads the menu the reconnaissance ran on — ${RM%%$'\t'*} copy $RECON_MENU"
    ALLOWED_SAVE="$ALLOWED"; ALLOWED="$RECON_MENU"; recon_revalidate_local addendum; ALLOWED="$ALLOWED_SAVE"
    # the override menu is (re)bound on EVERY addendum invocation, the completion reads of a DONE addendum included: the builder is idempotent
    # by content and restores the menu the round-4 design records, and --phase table reads the file as it stands (live 2026-10-01 08:16Z: the
    # completion read had rebuilt it with a new build time; a later re-run skipped at the DONE marker before the builder could restore it)
    [ "$REPRO_DRY_RUN" = 1 ] || build_override_menu
    [ ! -s "$OVERRIDE_MENU" ] || AP=(--allowed-pairs "$OVERRIDE_MENU")   # an existing override doc is the menu every binding sees (verify_fits)
  else
    probe_revalidate_local addendum                    # the local menu must be the record's CURRENT (widened) verdict
    recon_revalidate_local addendum
  fi
  ADDENDUM_MENU="$ALLOWED"
  if repro_done "$MODEL" "$STAGE" "$CLASS.addendum"; then
    # COMPLETED: the widened-menu refit must stand over the CURRENT evidence (a retried trial or a metrics backfill since would
    # strand it, and the table refuses it forever) — refit ONLY: nothing emitted, pushed or submitted; the marker is re-stamped
    [ -z "$MENU_OVERRIDE" ] || { ADDENDUM_MENU="$OVERRIDE_MENU"; AP=(--allowed-pairs "$OVERRIDE_MENU"); }   # The override doc (bound above, before the DONE check) IS the menu generation every binding sees
    PF0="$(probe_owned_fingerprint)"; fetch "$CL_ROOT" "$LOCAL_ROOT"; frc=$?; source_inventory_check addendum "$frc" "$PF0"
    VOUT="$(verify_fits addendum 2>&1)"; vrc=$?; printf '%s\n' "$VOUT"
    if [ "$REPRO_DRY_RUN" = 1 ] || [ "$vrc" = 0 ]; then
      repro_say "addendum: DONE marker present and every fit stands over the current evidence — skipping"
    elif grep -q '^\[hpo-table\] r1_fit.json ' <<<"$VOUT"; then
      repro_fatal "addendum: the DESIGN fit $FIT1 itself predates the current evidence (its reason is above) — re-run --phase analyze first (it re-derives the fit over the current evidence), then --phase addendum refreshes the widened-menu refit"
    else
      repro_say "addendum: the widened-menu refit $FIT1M predates the current evidence (its reason is above) — re-deriving it (refit only; nothing resubmitted)"
      addendum_menu_refit
      verify_fits addendum || repro_fatal "addendum: the re-derived refit still does not stand over the current evidence (its reason is above) — re-run --phase analyze first, then --phase addendum"
      mark addendum "refreshed refit over the current evidence; $FIT1M"
    fi
  else
    if [ "$REPRO_DRY_RUN" != 1 ]; then
      trap addendum_relaunch_exit EXIT                     # every exit after a relaunch's move rolls the record back from its receipt
      addendum_reconcile_transition                        # an open relaunch transition on the record is reconciled before anything is read or built
      [ -s "$FIT1" ] || repro_fatal "addendum needs the round-1 analyze artefact $FIT1 (run --phase analyze first)"
      if [ -n "$MENU_OVERRIDE" ]; then ADDENDUM_MENU="$OVERRIDE_MENU"      # built above, before the DONE check
      else [ -s "$ALLOWED" ] || repro_fatal "addendum needs $ALLOWED (run --phase probe first — the widened menu)"; fi
      # every requested pair must be in the menu (the probe's, or the run's override for exactly these pairs) and new to the round-1 design
      "$PY" - "$ADDENDUM_MENU" "$FIT1" "$ADDENDUM_PAIRS" <<'PYA' || repro_fatal "addendum: --pairs refused (its reason is above)"
import json, sys
menu = json.load(open(sys.argv[1])); art = json.load(open(sys.argv[2]))
allowed = {(int(p["rank"]), str(p["set"])) for p in menu.get("pairs") or []}
drawn = {(int(a), str(b)) for a, b in (art.get("allowed_pairs") or [])}
for item in [x.strip() for x in sys.argv[3].split(",") if x.strip()]:
    r, sid = item.split(":", 1); key = (int(r), sid.strip())
    if key not in allowed: sys.exit(f"pair {key} is not in the current menu {sorted(allowed)}")
    if key in drawn: sys.exit(f"pair {key} was in the round-1 design's menu {sorted(drawn)} — an addendum covers pairs certified AFTER the design")
print("addendum pairs admitted:", sys.argv[3])
PYA
    fi
    N4="$( [ -s "$R4" ] && "$PY" -c 'import json,sys; print(len(json.load(open(sys.argv[1]))["trials"]))' "$R4" || echo "$ADDENDUM_PAIRS" | tr ',' '\n' | grep -c . )"
    have4="$(n_summaries 4)"
    # the FAILED-array relaunch (array 165623: every task refused by the object factory before a summary): with SISTEER_RESUBMIT=1 the record's
    # round-4 design and its trial dirs move to superseded/<utc>/ (nothing deleted), the design is RE-DERIVED below (a corrected menu, the refit,
    # the emit) and the funnel re-mints the unit under its own proof (--resubmit). Without the flag the state is refused by name.
    if [ "${SISTEER_RESUBMIT:-0}" = 1 ] && [ "$REPRO_DRY_RUN" != 1 ] && cluster_has "$CL_ROOT/r4_trials.json" && ! cluster_job_active "$(name r4)" && [ "${have4:-0}" -lt "${N4:-1}" ] 2>/dev/null; then
      # PROOF FIRST, nothing moved — the funnel's fenced decision over the bound job (READ-ONLY prove-resubmit: sacct terminal, every task
      # dead, pending absent, spool empty, no registry row incl. row_history); only an AUTHORIZED verdict names the bound job we act on
      [ "$HAVE_SLURM_LIB" = 1 ] || repro_fatal "addendum: --resubmit needs the funnel (lib/slurm.sh) to prove the bound job dead and unregistered — nothing moved"
      PROOF="$(slurm_prove_resubmit --unit "$(unit r4)" --lwid "$MODEL/$STAGE/$CLASS/$(unit r4)" --kind fit --registers registry 2>&1)" \
        || { printf '%s\n' "$PROOF"; repro_fatal "addendum: --resubmit REFUSED by the funnel's proof (above) — the bound round-4 job is not provably dead and unregistered; NOTHING moved, nothing re-derived, nothing submitted"; }
      printf '%s\n' "$PROOF" | grep -v '^bound_job'
      BOUND4="$(printf '%s\n' "$PROOF" | awk -F'\t' '$1 == "bound_job" {print $3}' | tail -1)"
      [ -n "$BOUND4" ] || repro_fatal "addendum: the proof authorized a re-mint but named no bound job — nothing moved"
      # pin the record's design, then move it with an ATOMIC recheck on the record host: the design is still the pinned bytes and the job name is
      # not in the queue at the instant of the move — otherwise nothing moves (the proof read a world that has since changed)
      PIN4="$(cluster_cat "$CL_ROOT/r4_trials.json" | sha256sum | cut -d' ' -f1)"
      utc4="$(date -u +%Y%m%dT%H%M%SZ)"
      # the per-record LOCK is taken first and HELD across the whole replacement (released at the close or after the rollback); the recheck +
      # receipt + move run under it on the record host — the design is still the pinned bytes, the job name is not in the queue (squeue must
      # exit 0: an error never reads as empty), no transition is open — and the receipt is durable BEFORE the first mv, so a death at any
      # later point rolls back from it (here, by the EXIT trap; on a later run, by the reconciliation)
      addendum_lock_acquire "$utc4" || repro_fatal "addendum: --resubmit: the record's relaunch lock could not be taken (its reason is above) — NOTHING moved; re-run once it is free"
      if ! resub_ssh "bash -s -- '$CL_ROOT' '$PIN4' '$(name r4)' '$utc4' '$BOUND4' '$(resub_who)' '$RESUB_RECEIPT' '$RESUB_LOCK'" <<'REMOTE_MOVE'
set -u
root="$1"; pin="$2"; jobname="$3"; utc="$4"; bound="$5"; by="$6"; receipt="$7"; lock="$8"
cd "$root" || exit 3
[ -d "$lock" ] || { echo "the relaunch lock $root/$lock is not held — nothing moved"; exit 6; }
[ ! -e "$receipt" ] || { echo "OPEN TRANSITION: $root/$receipt exists — it is reconciled first (re-run --phase addendum); nothing moved"; exit 7; }
[ "$(sha256sum r4_trials.json | cut -d' ' -f1)" = "$pin" ] || { echo "CHANGED: r4_trials.json is not the pinned design"; exit 4; }
q="$(squeue -h -u "${USER:-$(id -un)}" -n "$jobname" -o %i 2>&1)"; qrc=$?
[ "$qrc" = 0 ] || { echo "UNKNOWN: squeue did not answer cleanly (rc $qrc: $(printf %s "$q" | tr '\n' ' ' | cut -c1-160)) — the queue is not provably empty; nothing moved"; exit 5; }
[ -z "$(printf %s "$q" | tr -d '[:space:]')" ] || { echo "CHANGED: a job named $jobname is in the queue ($q)"; exit 4; }
list=(r4_trials.json); for d in r4_t*; do [ -d "$d" ] && list+=("$d"); done
receipt_json(){   # <state>: the durable transition receipt — what moves where, when, by whom, under which proof
  printf '{"receipt":"hpo-addendum-resubmit","state":"%s","utc":"%s","by":"%s","bound_job":"%s","job_name":"%s","pinned_sha256":"%s","superseded":"superseded/%s","replacement_sha256":null,"moved":[' "$1" "$utc" "$by" "$bound" "$jobname" "$pin" "$utc"
  local sep="" p; for p in "${list[@]}"; do printf '%s"%s"' "$sep" "$p"; sep=","; done; printf ']}\n'
}
receipt_json moving > "$receipt.tmp" && mv -f "$receipt.tmp" "$receipt" || { rm -f "$receipt.tmp"; echo "the transition receipt $root/$receipt could not be written — nothing moved"; exit 8; }
mkdir -p "superseded/$utc" || { echo "superseded/$utc could not be created — nothing moved (receipt open, state moving)"; exit 8; }
for p in "${list[@]}"; do mv -n "$p" "superseded/$utc/" || { echo "MOVE FAILED at $p — the receipt (state moving) stays open; rolled back from it"; exit 8; }; done
receipt_json moved > "$receipt.tmp" && mv -f "$receipt.tmp" "$receipt"
ls "superseded/$utc"
REMOTE_MOVE
      then
        RESUB_UTC4="$utc4"
        if cluster_has "$CL_ROOT/$RESUB_RECEIPT"; then RESUB_OPEN4=1; repro_fatal "addendum: --resubmit: the move on $CL_HOST:$CL_ROOT did not complete (its reason is above) — rolled back from the transition receipt (superseded/$utc4/); re-run to prove again"; fi
        repro_fatal "addendum: --resubmit: the record at $CL_HOST:$CL_ROOT could not be moved — changed between the proof and the move, the queue not provably empty, an open transition, or another relaunch holds the record's lock (its reason is above) — NOTHING moved; re-run to prove again"
      fi
      RESUB_UTC4="$utc4"; RESUB_OPEN4=1                  # from here every exit before the close rolls the record back (addendum_relaunch_exit)
      repro_say "addendum: RESUBMIT authorized by the funnel's proof (bound job $BOUND4 dead and unregistered) — the failed round-4 design (sha ${PIN4:0:12}) and its $have4-summary trial dirs on $CLUSTER moved to $CL_ROOT/superseded/$utc4/ (nothing deleted; transition receipt $RESUB_RECEIPT, lock held); the design is re-derived and re-submitted through the funnel's --resubmit proof"
      rm -f "$R4"; for d in "$LOCAL_ROOT"/r4_t*/; do [ -d "$d" ] && mv -n "$d" "$LOCAL_ROOT/superseded.r4.$utc4.$(basename "$d")" 2>/dev/null; done   # the mirror's copies step aside too (the fetch has no --delete)
      have4=0
    fi
    if [ "$N4" -gt 0 ] 2>/dev/null && [ "$have4" -ge "$N4" ] 2>/dev/null; then
      repro_say "addendum: $have4/$N4 round-4 summaries on $CLUSTER — complete"; mark addendum "$have4 summaries; pairs $ADDENDUM_PAIRS"
    elif cluster_job_active "$(name r4)"; then
      repro_note "addendum: array $(name r4) is queued/running on $CLUSTER ($have4/$N4 summaries so far) — re-run when its watcher reports DONE"
    elif cluster_has "$CL_ROOT/r4_trials.json"; then
      repro_fatal "addendum: $CL_ROOT/r4_trials.json is on the record with $have4/$N4 summaries and no job named $(name r4) queued — inspect the array's logs; to resubmit the SAME design re-run with SISTEER_RESUBMIT=1 (the funnel's proof decides)"
    else
      # the refit over the WIDENED menu writes to a DISTINCT path from the start (`--fit-name r1_fit_menu.json`) — the canonical r1_fit.json the
      # round-1 selection and the pick were made from is never its output; its sha is taken before and checked after, and on ANY change it is
      # restored from the record before anything else happens (the non-PD verdict overwrote it and exited nonzero). The ONLY nonzero refit that
      # is accepted is the recognized non-positive-definite verdict — an artefact at $FIT1M that validates as a v2 fit over exactly the widened
      # menu, extended from the design's, PD false, rows present, selection = the best raw trial — which is the fallback state
      # build_addendum_trials handles (each pair at its set's best round-1 lr); every other nonzero stays a failure.
      PF0="$(probe_owned_fingerprint)"                      # the revalidated menu must survive the search fetch untouched
      fetch "$CL_ROOT" "$LOCAL_ROOT"; frc=$?                # the round-1 summaries the refit pools live on the record (memoised per invocation)
      source_inventory_check addendum "$frc" "$PF0"         # the refit pools the mirror: it must BE this record's evidence
      addendum_menu_refit
      local_run "$PY" "$TUNE" emit-trials --addendum-from "$FIT1M" --allowed-pairs "$ADDENDUM_MENU" --pairs "$ADDENDUM_PAIRS" --out "$R4" --arm "$ARM" --config "$LOCAL_CFG" --si "$HPO_SI" \
        || [ "$REPRO_DRY_RUN" = 1 ] || repro_fatal "addendum: emit-trials --addendum-from failed (its reason is above)"
      push "$R4" "$CL_ROOT/r4_trials.json" || repro_fatal "addendum: the design could not be uploaded to $CL_HOST:$CL_ROOT/r4_trials.json; nothing submitted"
      [ -z "${RESUB_UTC4:-}" ] || transition_write replaced "$(sha256sum "$R4" | cut -d' ' -f1)" || repro_fatal "addendum: --resubmit: the transition receipt could not record the uploaded replacement (above) — rolled back from the receipt; nothing submitted"
      N4="$( [ -s "$R4" ] && "$PY" -c 'import json,sys; print(len(json.load(open(sys.argv[1]))["trials"]))' "$R4" || echo "$N4" )"
      if [ -n "${RESUB_UTC4:-}" ]; then
        # a refusal by the funnel's second proof AFTER the move must not leave a replacement design with no job and the previous one only under
        # superseded/ — the previous design and trial dirs return to their canonical paths; the replacement steps aside.
        # an ACCEPTED submission closes the transition FIRST (the array now runs the replacement; a lost close completes on the next run —
        # never a rollback), then the lock is released
        if submit_design r4 "$CL_ROOT/r4_trials.json" "0-$((N4 - 1))"; then
          RESUB_OPEN4=0
          transition_write submitted || repro_warn "addendum: --resubmit: the receipt could not advance to submitted (the array IS submitted; the next --phase addendum completes the transition)"
          transition_close complete "$RESUB_UTC4" && repro_say "addendum: --resubmit: transition $RESUB_UTC4 CLOSED — receipt superseded/$RESUB_UTC4/transition.json (state complete)"             || repro_warn "addendum: --resubmit: the receipt could not close (the array IS submitted; the next --phase addendum completes the transition)"
          addendum_lock_release
        else
          src=$?; RESUB_OPEN4=0
          if relaunch_rollback_if_safe "$RESUB_UTC4" "--resubmit: the funnel's re-submission did not complete AFTER the move (rc $src; its reason is above)"; then
            addendum_lock_release
            repro_fatal "addendum: --resubmit: the funnel refused the re-submission AFTER the move (its reason is above) — the previous round-4 design and its trial dirs are RESTORED to their canonical paths on $CLUSTER (the replacement is at superseded/$RESUB_UTC4/replacement.r4_trials.json); nothing submitted"
          fi
          addendum_lock_release
          repro_fatal "addendum: --resubmit: the funnel's re-submission did not complete (rc $src) and a launch may have happened under it — the record is left as it stands with its receipt OPEN; re-run --phase addendum to reconcile it through the funnel (nothing moved back)"
        fi
      else
        submit_round 4 "0-$((N4 - 1))"
      fi
      repro_say "addendum: $N4 trial(s) at pairs $ADDENDUM_PAIRS submitted as round 4 — re-run --phase addendum when the array is DONE, then --phase table"
    fi
  fi
fi

# ================================================================================================================ confirm-pick
# ONE trial at the picked pair at its inferred learning rate (hpo_pick's request), round 8, the same hpo_trial.sbatch as
# every round; its measured clip-activation rate is what --phase pick re-reads.
# The request RESERVES its identity: index -> its own immutable request file (the ledger),
# its own design path $CL_ROOT/r8_tNN_trials.json and its own job name $(name r8-tNN) — a queued trial's design input is never
# rewritten by a later pick, two picks never share a trial dir, and a replacement is REFUSED while any earlier confirm-pick job
# of this search is still queued/running (the queue is read here, where it can be; hpo_pick only reserves).
if want confirm-pick; then
  probe_revalidate_local confirm-pick
  if [ "$CLASS" != lora ] || [ "$PROTOCOL" = paper ]; then
    [ "$PHASE" = confirm-pick ] && repro_fatal "--phase confirm-pick belongs to the FD 6 LoRA search (an off-winner pick's confirmation trial); not applicable to class=$CLASS protocol=$PROTOCOL"
  elif done_or_skip pick; then :
  elif [ ! -s "$PICK_REQ" ]; then
    [ "$PHASE" = confirm-pick ] && repro_fatal "confirm-pick: no request at $PICK_REQ — run --phase pick --rank R --set S first (a pick whose pair has no clip evidence writes it)"
  else
    read -r IDX REQ_FILE DESIGN JOB_SUFFIX TRIAL_ID < <(confirm_pick_identity "$PICK_REQ") || repro_fatal "confirm-pick: $(confirm_pick_identity "$PICK_REQ" 2>&1)"
    R8_REQ="$LOCAL_ROOT/$REQ_FILE"; R8="$LOCAL_ROOT/$DESIGN"; CL_R8="$CL_ROOT/$DESIGN"; R8_NAME="$(name "$JOB_SUFFIX")"
    [ -s "$R8_REQ" ] || repro_fatal "confirm-pick: the pointer $PICK_REQ names the reserved request $R8_REQ, which is missing — the ledger and the pointer disagree; re-run --phase pick --rank R --set S"
    if [ "$REPRO_DRY_RUN" = 1 ]; then
      emit_from --confirm-pick-from "$R8_REQ" "$R8"; push "$R8" "$CL_R8"; submit_design "$JOB_SUFFIX" "$CL_R8" 0-0
    else
      # the design is emitted from the IMMUTABLE per-index request, never from the (movable) pointer
      emit_from --confirm-pick-from "$R8_REQ" "$R8" || repro_fatal "confirm-pick: emit-trials --confirm-pick-from failed (its reason is above)"
      if cluster_has "$CL_R8"; then
        LOCAL_SHA="$(sha256sum "$R8" | cut -c1-64)"; REMOTE_SHA="$(cluster_cat "$CL_R8" | sha256sum | cut -c1-64)"
        [ "$LOCAL_SHA" = "$REMOTE_SHA" ] || repro_fatal "confirm-pick: $CLUSTER:$CL_R8 differs from the design emitted from $R8_REQ — a reserved design is immutable and is never overwritten under a running job; inspect both (a hand edit?) and pick again (a new pick reserves a new index)"
        if cluster_has "$CL_ROOT/$TRIAL_ID/summary.json"; then
          fetch "$CL_ROOT" "$LOCAL_ROOT"
          repro_say "confirm-pick: $TRIAL_ID is DONE on $CLUSTER and fetched — re-run --phase pick --rank R --set S: the gate now reads its measured clip-activation rate (below 25% promotes; at/over refuses by name)"
        elif cluster_job_active "$R8_NAME"; then
          repro_note "confirm-pick: trial $R8_NAME ($TRIAL_ID) is still queued/running on $CLUSTER"; exit 0
        else
          repro_fatal "confirm-pick: no summary at $CL_ROOT/$TRIAL_ID on $CLUSTER and no job named $R8_NAME queued — inspect the job's log; to resubmit the SAME request delete the cluster's $CL_R8 and re-run --phase confirm-pick"
        fi
      else
        # not yet submitted: REFUSE while any EARLIER confirm-pick trial of this search (the ledger's lower indices) is active —
        # its result stays valid evidence at the earlier pick; wait for it, or cancel it by name (hpo_trial.sbatch never resubmits itself)
        for f in "$LOCAL_ROOT"/hpo_pick_confirm_request_t*.json; do
          [ -s "$f" ] || continue
          read -r P_IDX _ _ P_SUFFIX P_TRIAL < <(confirm_pick_identity "$f") || continue
          [ "$P_IDX" -lt "$IDX" ] || continue
          cluster_job_active "$(name "$P_SUFFIX")" && repro_fatal "confirm-pick: REFUSING to submit $R8_NAME ($TRIAL_ID) while its predecessor $(name "$P_SUFFIX") ($P_TRIAL, the earlier pick's confirm trial, request $(basename "$f")) is queued/running on $CLUSTER — wait for it (re-run when its watcher reports DONE; its trial stays valid clip evidence at that pick), or cancel it by name on $CLUSTER first: scancel -u \$USER --name=$(name "$P_SUFFIX")"
        done
        push "$R8" "$CL_R8"
        submit_design "$JOB_SUFFIX" "$CL_R8" 0-0
        repro_say "confirm-pick: $R8_NAME ($TRIAL_ID, design $DESIGN) is submitted — re-run --phase pick --rank R --set S when it is DONE"; exit 0
      fi
    fi
  fi
fi
repro_say "03_hpo done (model=$MODEL class=$CLASS phase=$PHASE)"
