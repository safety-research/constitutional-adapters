#!/bin/bash
#SBATCH --job-name=evals_cell_controls  # SESSION-UNIQUE evals_* name (capeval-controls §6);
#                                       # cancel by JOB ID only.
#SBATCH --partition=general
#SBATCH --qos=high                    # cluster A default; 1 of the 16-GPU high budget.
#                                     # #SBATCH headers cannot branch per cluster, so on
#                                     # cluster B (its own QoS name) submit through
#                                     # submit_evals_cell_controls.sh, which derives the
#                                     # QoS + --output at submit time and overrides both
#SBATCH --gres=gpu:1                  # one H200: ONE pinned model load serves all five
#                                     # controls per cell (driver design)
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G                    # ~128G per GPU; NEVER --mem=0 (hard rule 2)
#SBATCH --time=0-08:00:00             # §6: ~4h estimated for the 5-control cell, ~2x padding
#SBATCH --output=/workspace-vast/%u/exp/logs/%x_%j.out  # cluster A default; Slurm never
#                                     # creates log parents and a nonexistent path means
#                                     # the job NEVER STARTS — the submit wrapper mkdirs
#                                     # the cluster-correct dir and overrides this
#SBATCH --open-mode=append            # retries append, never clobber
#SBATCH --no-requeue                  # stage2_common self-escalates
#SBATCH --signal=B:TERM@120

# Submit (the sha pin is mandatory — s2_init refuses an unpinned submission) from a shell that carries the
# job environment (submit_evals_cell_controls.sh's header says how):
#   sbatch --comment=sha:$(git -C ${SISTEER_PROJECT_ROOT}-capeval-repo rev-parse HEAD) \
#          scripts/sbatch/evals_cell_controls.sbatch
#
# The cell controls (`evals_cell_controls`):
# the three wired controls (mmlu, xstest, or_bench_hard_1k;
# gsm8k is the served standard's since PW3 — cell_controls_run refuses it)
# through a REAL compiled object cell — plus the prompted cell
# as additional SI-as-system coverage — via scripts/cell_controls_run.py
# generate. Needs BOTH lanes' code, so it runs from the INTEGRATION
# checkout with the capeval lane's venv discipline (capeval_common.sh:
# CAPEVAL_REPO_DIR, EXPECTED_SHA inside the checkout, 3.12 --group capeval).
#
# §4.1 EQUIVALENCE GATE (r1 #2), run FIRST: engine-served must equal
# lane-served (identical greedy generations + resolved object/coefficient)
# on the real compiled cell BEFORE any control number is generated — the
# job fails loudly on a mismatch and generates nothing.
#
# Env seams (BODY-set or via a wrapper body — never sbatch --export,
# hard rule 1):
#   CELLS         space-separated DefenseCell ids run in ONE job
#                 (default: the compiled cell + the prompted
#                 coverage cell)
#   CONTROLS      csv for --controls (the retired wmdp / alpaca_eval refuse as unknown)
#   DEFENSES_DIR  the commit-independent compiled-objects dir TrainTest
#                 hands over (fit_report.json is never consulted; artifact
#                 PRESENCE is the truth)
#   OUT_ROOT      the controls store root — on SHARED storage, deliberately
#                 NOT inside the checkout (survives re-checkouts; the
#                 controls_summary.json under it is what two_si_run.py
#                 aggregate fills from after the grading host judge phase)
#   EQUIV_CELL    the compiled cell the equivalence gate exercises
#   LIMIT         per-control item cap for smoke runs (0 = full)
#   JUDGE_MODEL   judge model id passed to generate as --judge-model (empty
#                 = the capeval judging default). The judge model binds at
#                 GENERATE time (resolved config -> cell-store key), so a
#                 battery judging with a different model MUST set it here,
#                 and the grading host judge phase must pass the same value.
#   COEFF         serving-coefficient (alpha) override for dose-response
#                 sweeps: --coeff on generate + SISTEER_EQUIV_SCALE for the
#                 gate (empty = the config's pre-registered scale)
#   STEER_POSITIONS  which forwards run inside the arm ctx: empty/"all"
#                 (the existing serving) or "generated" (prefill outside
#                 the ctx — prompt positions served by the plain base
#                 model, decode forwards steered; method-generic, gated
#                 at ctx enter/exit). NOTE the §4.1 equivalence gate
#                 still runs at the DEFAULT all-positions serving — it
#                 proves engine==lane on the serving stack; the
#                 generated-only phase gating is pinned by its own unit
#                 tests (tests/test_steer_generated_only.py).
#   CAPEVAL_REPO_DIR  the integration checkout (capeval_common.sh gates on
#                 ITS EXPECTED_SHA; peers' checkouts never touched)
#
# Preemption tolerance: the CellStore rows, the judged GenSpools, and the
# judged spools all resume per item; stage2_common provides the
# SIGTERM drain, QoS escalation, STOP sentinel, and $FIGS_DIR mirror.
#
# Submit (after syncing the integration checkout + recording its sha) —
# through the wrapper, which takes the cluster's QoS,
# creates the log dir (Slurm never creates log parents), and overrides
# --output, so the SAME script submits correctly on BOTH clusters:
#   scripts/sbatch/submit_evals_cell_controls.sh                     # full
#   # smoke: LIMIT is body-env — use a wrapper body, e.g.
#   #   printf '%s\n' '#!/bin/bash' 'LIMIT=20' \
#   #     'source scripts/sbatch/evals_cell_controls.sbatch' | sbatch ...
# The JUDGE phase runs on the grading host afterwards (Low-Prio key + judge
# cache live there):
#   uv run python scripts/cell_controls_run.py judge --cell <cell> \
#     --out-root <OUT_ROOT> --defenses-dir <DEFENSES_DIR>

set -uo pipefail
# The shared root is SISTEER_SHARED_ROOT, which the front door exports from the cluster's site block; a job
# without it refuses by name (there is no built-in root). Needed HERE (capeval_common.sh reads it too)
# because the source path below is itself $SHARED-relative.
SHARED="${SISTEER_SHARED_ROOT:?set SISTEER_SHARED_ROOT, the shared root of the cluster (the front door exports it from the site config)}"
# our OWN integration checkout (capeval-controls §6) — capeval_common.sh
# pins S2_REPO_DIR to CAPEVAL_REPO_DIR and gates on ITS EXPECTED_SHA
export CAPEVAL_REPO_DIR="${CAPEVAL_REPO_DIR:-${SISTEER_PROJECT_ROOT:?set SISTEER_PROJECT_ROOT, the cluster project directory (the front door exports it from the site config)}-evals-repo}"
source "$CAPEVAL_REPO_DIR/scripts/sbatch/capeval_common.sh"
s2_init "$@"

# cluster NCCL requirements (inert on 1 GPU; uniform header per the rules)
export NCCL_NVLS_ENABLE=0

CELLS="${CELLS:-safety_refusal__vector_all prompted__safety_refusal}"
CONTROLS="${CONTROLS:-mmlu,xstest,or_bench_hard_1k}"
DEFENSES_DIR="${DEFENSES_DIR:-$SHARED/exp/jailbreak/two_si_defenses/cells}"
OUT_ROOT="${OUT_ROOT:-$SHARED/exp/training/evals_cell_controls/out}"
EQUIV_CELL="${EQUIV_CELL:-safety_refusal__vector_all}"
LIMIT="${LIMIT:-0}"
LIMIT_ARGS=""
if [ "$LIMIT" -gt 0 ] 2>/dev/null; then
  LIMIT_ARGS="--limit $LIMIT"
fi
JUDGE_MODEL="${JUDGE_MODEL:-}"
JUDGE_ARGS=""
if [ -n "$JUDGE_MODEL" ]; then
  JUDGE_ARGS="--judge-model $JUDGE_MODEL"
fi
# COEFF: serving-coefficient (alpha) override for dose-response sweeps —
# passed to generate as --coeff AND exported as SISTEER_EQUIV_SCALE so the
# equivalence gate compares engine vs lane at the SAME alpha (a
# non-default-alpha job would otherwise fail its own gate). Empty = the
# config's pre-registered scale.
COEFF="${COEFF:-}"
COEFF_ARGS=""
if [ -n "$COEFF" ]; then
  COEFF_ARGS="--coeff $COEFF"
  export SISTEER_EQUIV_SCALE="$COEFF"
fi
# STEER_POSITIONS: engine-phase steering policy (empty = all, the existing
# serving). The grading host judge phase must pass the same value — judge
# refuses a store generated under a different policy.
STEER_POSITIONS="${STEER_POSITIONS:-}"
STEER_ARGS=""
if [ -n "$STEER_POSITIONS" ] && [ "$STEER_POSITIONS" != "all" ]; then
  STEER_ARGS="--steer-positions $STEER_POSITIONS"
fi

# The compiled artifact must EXIST at the handed-over dir before anything
# runs (an empty per-commit checkout dir is a known failure mode; the
# equivalence test would SKIP without it, and a skipped gate
# must never look like a passed one).
if ! ls "$DEFENSES_DIR/$EQUIV_CELL"/object.* >/dev/null 2>&1; then
  s2_log "FATAL: no compiled artifact under $DEFENSES_DIR/$EQUIV_CELL -- point DEFENSES_DIR at the dir TrainTest handed over (criterion 2 needs a REAL object, D16)."
  exit 1
fi

# --- §4.1 equivalence gate (r1 #2): BEFORE any control generation ------------
export SISTEER_EQUIV_CELL="$EQUIV_CELL"
export SISTEER_EQUIV_DEFENSES_DIR="$DEFENSES_DIR"
s2_log "equivalence gate: engine-served vs lane-served on $EQUIV_CELL @ $DEFENSES_DIR"
if ! s2_run uv run --no-sync pytest tests/test_cell_controls_integration.py -q -rs; then
  s2_log "FATAL: the §4.1 equivalence test FAILED -- engine-served != lane-served, so no control number can be trusted; generating nothing."
  exit 1
fi

# --- generate: each cell in turn, all five controls per model load -----------
for CELL in $CELLS; do
  s2_log "cell-controls generate: cell=$CELL controls=$CONTROLS -> $OUT_ROOT"
  # shellcheck disable=SC2086  # deliberate word-splitting on the *_ARGS seams
  s2_run uv run --no-sync python scripts/cell_controls_run.py generate \
    --config configs/two_si.yaml \
    --cell "$CELL" --controls "$CONTROLS" \
    --out-root "$OUT_ROOT" --defenses-dir "$DEFENSES_DIR" \
    $LIMIT_ARGS $JUDGE_ARGS $COEFF_ARGS $STEER_ARGS \
    || exit $?
done

s2_log "generation complete for: $CELLS. Judge on the grading host next: uv run --no-sync python scripts/cell_controls_run.py judge --cell <cell> --out-root $OUT_ROOT --defenses-dir $DEFENSES_DIR"
exit 0
