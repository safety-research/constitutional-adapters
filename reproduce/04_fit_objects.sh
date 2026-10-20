#!/bin/bash
# reproduce/04_fit_objects.sh — stage 04: fit the steering objects of one model.
#
#   reproduce/04_fit_objects.sh --model K [--object control_lora|constitution_lora|control_relu|constitution_relu|
#                               difference_lora|difference_relu|refusal_direction|all] [--cluster <label>]
#                               [--refit <tag>] [--config F] [--overrides F] [--dry-run] [--help]
#
#   control_lora / constitution_lora   sbatch scripts/sbatch/sdfbase_fit_cell.sbatch <fit slug> <si> lora_r4_ce full22
#   control_relu / constitution_relu   the same with psr_ce_all      (si: control -> constv2ctl2-full, constitution -> constv2-mix-full;
#                                      fit slug = models.yaml models.<K>.fit.checkpoint.model_id, e.g. qwen35_4b_base; bank $SISTEER_FIT_BANK, full22)
#                                      DATA-PARALLEL: the fits config's top-level `fit_width: N` (absent = 1) sizes the
#                                      allocation — the sbatch line carries `--gres=gpu:N --cpus-per-task=16N --mem=128N G`
#                                      (lib/common.sh repro_fit_width_flags) and SDFBASE_FIT_WIDTH=N in the job env; in the job
#                                      slurm_fit_cell.py runs the trainer under torchrun --nproc_per_node=N (effective batch unchanged)
#                                      the trainer publishes the object record natively (fit_selection.json from the generated
#                                      config's train_kwargs.<arm>.fit_selection block, hashed into COMPLETE)
#   difference_lora                    scripts/sdfbase/lora_subtract.py on the cluster's login host (CPU tensor bookkeeping; inside
#                                      $SISTEER_CPU_SRUN when set) — needs both LoRA parents COMPLETE
#   difference_relu                    the DIRECT paired difference V = +P_mix - P_ctl: scripts/sbatch/sdfbase_derived_build.sbatch
#                                      `prep` (compose-v + verify-v + V's liveness + the native registry emission; kind composed,
#                                      label difference_relu) — ONE job; the psr_residual calibrate/assess/finish distillation is
#                                      LEGACY and never composed here — needs both ReLU parents COMPLETE
#   refusal_direction                  sbatch scripts/sbatch/fit_refusal_direction.sbatch <serve.checkpoint.model_id> <serve hf repo> <revision> <out_root>
#                                      (the SERVED checkpoint's registry id is the row's fit_model_id AND the publication directory
#                                      <out_root>/<served id>/<staged id>, shared by every key that serves it; done-ness = the row at this
#                                      rung AND that publication, its identity sidecar listed in COMPLETE — refusal_published_ready;
#                                      the job name is the identity's too, refusal-direction-<served id>-<staged id>, so a second key
#                                      serving the checkpoint sees the one fit queued and another coefficient's fit is not suppressed)
#                                      (c = the table's objects.refusal_direction cell, ar_c<c>__arditi_allpos — the fitter's spelling, the one identity
#                                      04 publishes, the pins writer pins and 05 stages; SISTEER_REFUSAL_COEFFICIENT may only restate it — spec FD 5, 0.3;
#                                      sets arditi:harmful_train / arditi:harmless_train: the paper fit's own training draw, the rows the fit behind
#                                      the committed refusal-direction objects started from (before its fit-time refusal filter), re-created from
#                                      andyrdt/refusal_direction's pinned splits and held to data/arditi/fit_prompts.manifest.json, read offline
#                                      from the copy 00_setup.sh --what third_party installs; SISTEER_REFUSAL_HARMFUL / _HARMLESS name other sets)
#   (the worked example qwen35_4b_demo declares `committed_objects_of` in reproduce/models.yaml: stage 05 IMPORTS the paper's committed
#   Qwen3.5-4B objects — objects/DEMO.json, through reproduce/tools/import_objects.py — in place of every unit above; this stage is the
#   fit-your-own path, which the README's second sequence runs after the committed one)
#   all                                every object above in that order; a unit whose parents are not yet published is reported, not forced
#   --refit <tag>                      a REFIT of a ReLU unit beside its published canonical cell (the random-init ReLU steers next to
#                                      the canonical zero-init pair). The tag is a REFIT-SIDE identity only — it writes NOTHING into the
#                                      registry record (the artefact's extra.init_scheme and the new object ids are the discriminator;
#                                      fit_replicate_label stays NULL):
#                                      the bank is DERIVED as full22_<tag> (SISTEER_FIT_BANK may only restate it; a foreign bank without a
#                                      tag refuses), the funnel unit / lwid is <unit>/<tag>, the job name fit-<model>-<unit>-<tag>
#                                      (derived-<model>-<tag>-prep; fresh STOP dirs), the grading host's marker DONE.<unit>.<tag>. Registry
#                                      done-ness is scoped by the PINNED fit config's non-zero psr init scheme (the blob at the run's sha —
#                                      the one the jobs run under; a dirty working-tree yaml refuses; a non-zero override must agree) AND an artifact location under the bank layout
#                                      …/exp/objects/sdfbase/full22_<tag>/ (a derived object through its first parent); the canonical
#                                      units exclude exactly those rows. Only control_relu / constitution_relu / difference_relu refit.
#                                      The sbatch scripts and the trainer are unchanged.
# The LoRA fits REFUSE until models.yaml models.<K>.lora_architecture.selection_ruling_id is set (FD 6a: the phase-A pick,
# under --dry-run the refusal is a note and the submission is still printed).
# Idempotent by registry state first (`registry query` on the grading host's database: an object row of the unit's kind / role /
# fit model), then the cluster's COMPLETE markers READ WITH their emission evidence (the emitter leaves registry_emission.json + a
# REGISTRY_EMIT_FAILED marker beside COMPLETE) — emitted: waiting for the fold (neither resubmitted nor marked done); failed or never
# emitted: REFUSED by name; no evidence: resubmitted, the job validates COMPLETE and replays the record without retraining — then the
# unit's DONE.<object> under the grading host's state dir. Prints `registry: object <id>` per id found or produced and
# `submitted <cluster> <jobid> <unit>` per sbatch (through lib/slurm.sh submit; absent: --dry-run prints the sbatch line, live REFUSES).
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/lib/common.sh"
HAVE_SLURM_LIB=0
if [ -f "$HERE/lib/slurm.sh" ]; then source "$HERE/lib/slurm.sh"; HAVE_SLURM_LIB=1; fi
usage(){ sed -n '2,39p' "$0"; repro_common_help; }
OBJECT=all; CLUSTER=""; REFIT="${SISTEER_FIT_REFIT:-}"
repro_parse_common_args "$@"
set -- "${REPRO_ARGS[@]+"${REPRO_ARGS[@]}"}"
while [ $# -gt 0 ]; do
  case "$1" in
    --object) OBJECT="${2:?--object needs a value}"; shift 2;; --object=*) OBJECT="${1#--object=}"; shift;;
    --cluster) CLUSTER="${2:?--cluster needs a label}"; shift 2;; --cluster=*) CLUSTER="${1#--cluster=}"; shift;;
    --refit) REFIT="${2:?--refit needs a tag}"; shift 2;; --refit=*) REFIT="${1#--refit=}"; shift;;
    *) echo "REFUSED: unknown argument $1" >&2; usage >&2; exit 2;;
  esac
done
[ "$REPRO_HELP" = 1 ] && { usage; exit 0; }
OBJECTS_ALL="control_lora constitution_lora control_relu constitution_relu difference_lora difference_relu refusal_direction"
case "$OBJECT" in
  all) UNITS="$OBJECTS_ALL";;
  control_lora|constitution_lora|control_relu|constitution_relu|difference_lora|difference_relu|refusal_direction) UNITS="$OBJECT";;
  *) repro_fatal "--object must be one of: $OBJECTS_ALL all";;
esac
repro_load_config
repro_require_model
[ -n "$CLUSTER" ] || CLUSTER="$SISTEER_PRIMARY_CLUSTER"
cluster_require "$CLUSTER"
STAGE=04_fit_objects
SHA="$(repro_head_sha)"
DEFAULT_BANK=full22                        # the paper's objects bank: <root>/exp/objects/sdfbase/<bank>/<fit slug>/cells_all/
BANK="${SISTEER_FIT_BANK:-$DEFAULT_BANK}"
MIX_SI=constv2-mix-full; CTL_SI=constv2ctl2-full
# --- the model's facts (reproduce/models.yaml; a missing path refuses by name) ----------------------------------------------
FIT_ID="$(repro_model "$REPRO_MODEL" fit.checkpoint.model_id)" || exit 2
FIT_REPO="$(repro_model "$REPRO_MODEL" fit.checkpoint.hf_repo)" || exit 2
FIT_REV="$(repro_model "$REPRO_MODEL" fit.checkpoint.revision)" || exit 2
SERVE_ID="$(repro_model "$REPRO_MODEL" serve.checkpoint.model_id)" || exit 2
SERVE_REPO="$(repro_model "$REPRO_MODEL" serve.checkpoint.hf_repo)" || exit 2
SERVE_REV="$(repro_model "$REPRO_MODEL" serve.checkpoint.revision)" || exit 2
# gap-refusal-pin: a unit whose cell the table IMPORTS from the paper's committed objects
# (`committed_objects_of` on the row or on the cell — reproduce/lib/models.py committed_objects_of; the worked example's row declares it) is the
# COMMITTED sequence's (05 --from-committed through reproduce/tools/import_objects.py), never fitted here: a fit under such a key registers a
# SECOND object per cell beside the committed pin, which the pins writer refuses by name (make_full22_configs.py: a committed pin's identity
# never changes by a fit), so the sequence could never reach 05 — refused BEFORE any registry or cluster read, before a GPU hour is spent.
# Under --dry-run a note, and the plan still prints. Fit your own objects under a key of your own (reproduce/README.md, Adding a model:
# copy the row WITHOUT committed_objects_of).
COMMITTED_UNITS="$("$(repro_python)" - "$REPRO_LIB_DIR" "$(repro_models_table)" "$REPRO_MODEL" $UNITS <<'PYEOF'
import sys
sys.path.insert(0, sys.argv[1]); import models as M
m = M.model(sys.argv[3], sys.argv[2])
print(" ".join(u for u in sys.argv[4:] if M.committed_objects_of(m, u)))
PYEOF
)" || repro_fatal "04: lib/models.py could not read $(repro_models_table) for $REPRO_MODEL's committed_objects_of (above)"
if [ -n "$COMMITTED_UNITS" ]; then
  COMMITTED_MSG="$REPRO_MODEL: [$COMMITTED_UNITS] are the paper's COMMITTED objects under this key (committed_objects_of in $(repro_models_table), on the row or the cell): 05 --from-committed imports them (reproduce/tools/import_objects.py), nothing fits them here — a fit under this key would register a second object per cell beside the committed pin, which the pins writer refuses (a committed pin's identity never changes by a fit), so the sequence could never reach 05; fit your own objects under a key of your own (reproduce/README.md, Adding a model: copy the row WITHOUT committed_objects_of)"
  [ "$REPRO_DRY_RUN" = 1 ] || repro_fatal "$COMMITTED_MSG"
  repro_note "would REFUSE live: $COMMITTED_MSG — printing the submissions it would make"
fi
# FD 6a: the LoRA architecture ruling — a missing key is tolerated under --dry-run only
if RULING="$(repro_model "$REPRO_MODEL" lora_architecture.selection_ruling_id 2>/dev/null)"; then :; else
  RULING=""
  [ "$REPRO_DRY_RUN" = 1 ] || repro_fatal "models.yaml models.$REPRO_MODEL.lora_architecture.selection_ruling_id is missing (FD 6a)"
  repro_note "models.yaml models.$REPRO_MODEL.lora_architecture has no selection_ruling_id key (tolerated under --dry-run only)"
fi

# --- the refit identity (RELU-REFIT lane, D1795 / D1813 / D1816): the tag IS the lane-side identity and the bank is DERIVED from it ------
# nothing of it reaches the registry record (fit_replicate_label is free text that v_seed_spread counts as a seed replicate; the
# designed discriminator is the artefact's extra.init_scheme plus the new object ids). The bank <DEFAULT_BANK>_<tag> is ONE knob (a
# free bank beside a free tag let a mistyped tag train into the right bank under the wrong name); SISTEER_FIT_BANK may only restate
# it; a foreign bank without a tag is refused. The scheme that tells a refit row from a canonical one is read from the COMMITTED fits
# config the jobs ran under (train_kwargs.psr_ce_all.init_scheme — the overrides file is not what the job read; a site on the paper
# defaults would otherwise get a falsely idempotent route), must be non-zero, and must agree with the run override when that names
# one. Only the ReLU units refit.
REFIT_UNITS="control_relu constitution_relu difference_relu"
# the fits config the jobs RUN UNDER is the PINNED blob at $SHA (the sbatch --comment pin; the job refuses any other checkout), never
# the working-tree file (a locally edited yaml would have been advertised and used for done-ness while the fit trained the pinned
# scheme). The scheme is parsed from `git show $SHA:<path>`; under --refit a working-tree file that differs from the blob refuses by
# name (04 pins what the cluster runs; stamp_init_scheme's own dirty check runs only for a non-zero override, so the paper-default
# `zero` would otherwise let a dirty yaml through).
REFIT_CFG="configs/sdfbase_${FIT_ID}_fits.yaml"                # repo-relative: the pinned blob's path (FIT_CFG_LOCAL below is the working-tree copy)
REFIT_CFG_BLOB="$(git -C "$REPRO_REPO" show "$SHA:$REFIT_CFG" 2>/dev/null)" || REFIT_CFG_BLOB=""
if [ -n "$REFIT_CFG_BLOB" ]; then
  REFIT_SCHEME="$(printf '%s\n' "$REFIT_CFG_BLOB" | "$(repro_python)" -c '
import sys, yaml
cfg = yaml.safe_load(sys.stdin) or {}
print(((cfg.get("train_kwargs") or {}).get("psr_ce_all") or {}).get("init_scheme") or "zero")' 2>/dev/null || echo unreadable)"
else
  REFIT_SCHEME=unreadable                                       # no pinned blob: no git HEAD, or the config is not tracked at $SHA
fi
if [ -n "$REFIT" ]; then
  [[ "$REFIT" =~ ^[a-z][a-z0-9_]{0,31}$ ]] || repro_fatal "--refit must match ^[a-z][a-z0-9_]{0,31}\\$ (got '$REFIT')"
  REFIT_BANK="${DEFAULT_BANK}_${REFIT}"
  [ -z "${SISTEER_FIT_BANK:-}" ] || [ "$SISTEER_FIT_BANK" = "$REFIT_BANK" ] || repro_fatal "--refit $REFIT binds the bank to $REFIT_BANK; SISTEER_FIT_BANK=$SISTEER_FIT_BANK contradicts it — unset it, or spell the derived bank"
  BANK="$REFIT_BANK"
  [ "$SHA" != unknown ] && [ -n "$REFIT_CFG_BLOB" ] || repro_fatal "--refit $REFIT: no pinned fits config — $REPRO_REPO has no git HEAD, or $REFIT_CFG is not tracked at ${SHA:0:12}; the cluster job runs the tree pinned by that sha, so a refit cannot be composed from an unpinned file (D1853)"
  git -C "$REPRO_REPO" diff --quiet "$SHA" -- "$REFIT_CFG" 2>/dev/null || repro_fatal "--refit $REFIT: the working-tree $REFIT_CFG differs from the blob pinned at ${SHA:0:12} — the cluster job runs the PINNED tree (--comment=sha) while this run would read the local file for its done-ness scope; commit + mirror first (D1853)"
  [ -n "$REFIT_SCHEME" ] && [ "$REFIT_SCHEME" != zero ] && [ "$REFIT_SCHEME" != unreadable ] || repro_fatal "--refit $REFIT: $(basename "$REFIT_CFG") at ${SHA:0:12} declares train_kwargs.psr_ce_all.init_scheme '$REFIT_SCHEME' — a refit needs a NON-ZERO psr init scheme in the committed fit config: the artefact's extra.init_scheme is what tells a refit row from a canonical one in the registry (D1816)"
  # the COMMITTED config is authoritative; the paper default `zero` means "nothing to stamp" to this front door (stamp_init_scheme returns
  # early on it), so only a NON-ZERO override that names ANOTHER scheme contradicts the config (a test site on the paper defaults, or the
  # README's fresh-site command, must not refuse a correctly stamped config)
  [ -z "${OVERRIDE_INIT_SCHEME:-}" ] || [ "$OVERRIDE_INIT_SCHEME" = zero ] || [ "$OVERRIDE_INIT_SCHEME" = "$REFIT_SCHEME" ] || repro_fatal "--refit $REFIT: the run override says OVERRIDE_INIT_SCHEME=$OVERRIDE_INIT_SCHEME but the committed $(basename "$REFIT_CFG") declares psr init_scheme $REFIT_SCHEME — stamp and commit the config first (stamp_init_scheme refuses a dirty one)"
  for _u in $UNITS; do
    case " $REFIT_UNITS " in *" $_u "*) ;; *) repro_fatal "--refit $REFIT: $_u is not a refit unit (refit units: $REFIT_UNITS; --object all is refused under --refit)";; esac
  done
elif [ "$BANK" != "$DEFAULT_BANK" ]; then
  repro_fatal "SISTEER_FIT_BANK=$BANK without --refit: a non-default bank is a refit identity and needs its tag (--refit <tag> derives the bank ${DEFAULT_BANK}_<tag>); the paper's bank is $DEFAULT_BANK"
fi
# --- cluster paths -----------------------------------------------------------------------------------------------------------
SHARED_ROOT="$(cluster_var "$CLUSTER" CLUSTER_SHARED_ROOT)"
CL_HOST="$(cluster_var "$CLUSTER" CLUSTER_SSH)"
CHECKOUT="$(cluster_var "$CLUSTER" CLUSTER_EVAL_REPO_ROOT)/$SHA"
BANK_ROOT="$SHARED_ROOT/exp/objects/sdfbase/$BANK"
CELLS="$BANK_ROOT/$FIT_ID/cells_all"
POOLS="$SHARED_ROOT/vol/pool/$FIT_ID"
DERIVED="$BANK_ROOT/derived"
LORA_DIFF="$DERIVED/lora_subtraction/$FIT_ID/${MIX_SI}__lora_r4_ce_minus_ctl"
# the LEGACY psr_residual distillation tree (never composed here; the table's difference_relu is the composed paired write under
# psr_paired_write) — probed only to report it as evidence; the worked example's difference_relu is the paper's committed composed object, imported by 05
RESID="$DERIVED/psr_residual/$FIT_ID/${MIX_SI}__psr_ce_all_minus_ctl_distilled"
REFUSAL_ROOT="$(repro_expand_shared "$CLUSTER" "${SISTEER_REFUSAL_OUT_ROOT:-\$SHARED/exp/objects/refusal_direction}")"
# the refusal direction's done-ness read carries NO SQL rung predicate (the identity has no coefficient column, and a query without the rung
# read ANY rung of the served checkpoint as this unit): the rung is read by the pins writer's OWN Python (make_full22_configs.py
# --arditi-registered-ids, through refusal_registry_ids below), never by an SQL twin of it.
REFUSAL_GENERATOR="$REPRO_REPO/scripts/sdfbase/make_full22_configs.py"   # the pins writer: the ONE reading of a row's rung (arditi_row_rung) AND of the cell
# the refusal direction's IDENTITY — its coefficient and its staged id — is the models table's declaration, objects.refusal_direction.cell_id
# (`ar_c<c>__arditi_allpos`, the fitter's spelling), read through the pins writer's own cell reading (make_full22_configs.py --arditi-cell:
# arditi_declared_cell / arditi_cell_coefficient, the function arditi_pins_from_registry picks the pin's rung with), never from a knob of its own
# (a COEF taken from SISTEER_REFUSAL_COEFFICIENT alone let a site that set 0.5 over the copied example's `ar_c0p3` cell fit and register
# a 0.5 the pins writer — selecting the rung from the cell — never pinned, and 05 refused to stage after the GPU hour; a numerically
# equal cell spelled `ar_c0p30` was pinned at 0.3 while 04 published under `ar_c0p3` and 05 read the literal `ar_c0p30` directory).
# One reading, three consumers: the cell IS the directory 04 publishes (<refusal root>/<served id>/<cell>), the directory 05 stages from, and the
# pin's key (`arditi__<cell>`), so the reader refuses a cell that is not the fitter's spelling of its own coefficient, and
# SISTEER_REFUSAL_COEFFICIENT may only RESTATE the cell's (a disagreement refuses by name, both values printed, before any registry or cluster
# read and before any submission; dry run included — a misdeclared identity has no plan). Read only when the unit is requested (every other unit
# is untouched by it; `--object all` requests it). To fit another rung, change the CELL, so every stage reads the same identity.
COEF=""; REFUSAL_STAGED=""
case " $UNITS " in *" refusal_direction "*)
  REFUSAL_KNOB=(); [ -z "${SISTEER_REFUSAL_COEFFICIENT:-}" ] || REFUSAL_KNOB=(--coefficient "$SISTEER_REFUSAL_COEFFICIENT")
  REFUSAL_CELL_OUT="$("$(repro_python)" "$REFUSAL_GENERATOR" --arditi-cell --models-table "$(repro_models_table)" --model-key "$REPRO_MODEL" "${REFUSAL_KNOB[@]+"${REFUSAL_KNOB[@]}"}")" \
    || repro_fatal "04: the refusal direction's identity could not be formed from $(repro_models_table) models.$REPRO_MODEL.objects.refusal_direction (above): the cell spells the coefficient 04 fits at and the directory it publishes, the pins writer pins and 05 stages — align the cell (the fitter's spelling, ar_c<c>__arditi_allpos) and SISTEER_REFUSAL_COEFFICIENT (which may only restate it, or stay unset), then re-run; nothing is submitted"
  COEF="${REFUSAL_CELL_OUT%% *}"; REFUSAL_STAGED="${REFUSAL_CELL_OUT#* }"
  [ -n "$COEF" ] && [ -n "$REFUSAL_STAGED" ] && [ "$COEF" != "$REFUSAL_STAGED" ] || repro_fatal "04: make_full22_configs.py --arditi-cell printed '$REFUSAL_CELL_OUT', not '<coefficient> <cell id>' — the refusal direction's identity cannot be formed; nothing is submitted";;
esac
# the refusal direction's publication directory is the SERVED checkpoint's — <refusal root>/<serve.checkpoint.model_id>/<staged id> — never the
# table key's: its registry identity (campaign, fit_model_id = the served id, vector_all / arditi_allpos / baseline_published, the rung) names
# the directory, as the fit bank's cells are named by the FIT checkpoint's id, so every key that serves the checkpoint reads ONE publication,
# ONE row, ONE pin (keyed by the table key, a second key serving the same checkpoint read the row done, published nothing of its own, and
# 05 pulled from an empty directory). 05's cells_json spells the same path — the table's cell, literally, which is why the cell must be
# the fitter's spelling (above).
REFUSAL_DIR="$REFUSAL_ROOT/$SERVE_ID/$REFUSAL_STAGED"
FIT_VENV_PY="$(cluster_var "$CLUSTER" CLUSTER_ENVS_DIR)/$(cluster_fit_env_name "$CLUSTER")/bin/python"   # that cluster's prebuilt fit venv
FIT_CFG_LOCAL="$REPRO_REPO/configs/sdfbase_${FIT_ID}_fits.yaml"
FIT_WIDTH="$(repro_fit_width_of_config "$FIT_CFG_LOCAL")"     # The config's top-level fit_width (absent = 1) — THE width of every fitted cell
# the records' campaign: the site's SISTEER_REGISTRY_CAMPAIGN, else REGISTRY_CAMPAIGN, else the paper frame's sdfbase_full22 — the old
# `two_si` default filed this model's fit objects outside the frame (2026-10-02). This stage reads every unit's done-ness under it and
# the no-lib fallback job env states it; the Slurm branch's job env (lib/slurm.sh slurm_job_env, submit_twin.Site.registry_campaign), the
# pins writer and 08 resolve the SAME expression (common.sh repro_registry_campaign), so the fit files where this stage and 05 look
CAMPAIGN="${SISTEER_REGISTRY_CAMPAIGN:-${REGISTRY_CAMPAIGN:-sdfbase_full22}}"
# the emission contract this front door expects evidence to attest to (scripts/sdfbase/registry_emission.py EVIDENCE_CONTRACT)
EMISSION_CONTRACT="$(sed -n 's/^EVIDENCE_CONTRACT = "\([^"]*\)"$/\1/p' "$REPRO_REPO/scripts/sdfbase/registry_emission.py" | head -1)"
[ -n "$EMISSION_CONTRACT" ] || repro_fatal "cannot read EVIDENCE_CONTRACT from $REPRO_REPO/scripts/sdfbase/registry_emission.py"

# --- helpers -----------------------------------------------------------------------------------------------------------------
lwid(){ echo "$REPRO_MODEL/fit/objects/$1"; }
# the refit identity on every name the front door mints (RELU-REFIT) is spelled INLINE where it is used — the funnel unit /
# lwid tail <unit>${REFIT:+/$REFIT}, the grading-host marker DONE.<unit>${REFIT:+.$REFIT}, the job-name suffix ${REFIT:+-$REFIT} — all
# empty without --refit, so the canonical names are byte-identical; inline (no helper) because the tests execute fit_unit /
# difference_relu / the dispatcher loop as extracted fragments with their collaborators stubbed.
# the registry's refit scope (RELU-REFIT): a refit row is told from a canonical one by the ARTEFACT's init scheme
# (extra.init_scheme, the registry's designed home for it; REFIT_SCHEME = the committed config's) AND an artifact location under the bank
# layout …/exp/objects/sdfbase/<bank>/ — never the adapter's staging spelling $STAGE/sdfbase/full22_objects/… (a
# bare '*/sdfbase/full22_*/*' matched it and dropped the random-init canonical LoRA parents and their subtraction); the location list is
# read in both shapes it takes (bare paths from the native hook, {uri, bytes} objects from the adapter). A DERIVED object carries the
# scope through its first parent (V's identity is its parents). Under --refit <tag>: rows of bank <DEFAULT_BANK>_<tag> with that scheme;
# without: everything that is not such a row of ANY refit bank. Applied by registry_object_ids (the ONE query); registry_done threads
# its 4th positional into it; the test harnesses stub registry_object_ids, so no extracted fragment references this helper. No column
# is written for it.
refit_scope_sql(){   # self | parent -> a SQL predicate over the outer `object` row
  local scheme="${REFIT_SCHEME:-}"
  if [ -z "$scheme" ] || [ "$scheme" = zero ] || [ "$scheme" = unreadable ]; then
    # the committed config fits zero-init: no refit row can exist for this lane (a refit run refused above) — the scope is a no-op
    case "$1" in self|parent) echo "1 = 1";; *) return 2;; esac
    return 0
  fi
  _loc(){   # <alias> <bank glob> -> EXISTS predicate: an artifact location under that bank's REAL layout
    echo "EXISTS (SELECT 1 FROM json_each($1.artifact_locations_json) l WHERE (CASE l.type WHEN 'object' THEN json_extract(l.value, '\$.uri') ELSE l.value END) GLOB '*/exp/objects/sdfbase/$2/*')"
  }
  _refit_row(){   # <alias> <bank glob> -> that row is a refit row of that bank
    echo "(json_extract($1.extra_json, '\$.init_scheme') = '$scheme' AND $(_loc "$1" "$2"))"
  }
  _self(){   # <alias> -> this run's scope on that alias's row
    if [ -n "${REFIT:-}" ]; then _refit_row "$1" "${DEFAULT_BANK}_${REFIT}"; else echo "NOT $(_refit_row "$1" "${DEFAULT_BANK}_*")"; fi
  }
  # a DERIVED row's scope is its PARENTS' — EVERY parent, not the first (a composed V has parents [mix, control]; a second parent with
  # unverifiable provenance was invisible). Refit: at least one parent and every parent a proper refit row of this bank (a parent id
  # with no object row counts as failing, and so does a parent whose init scheme is NULL — COALESCE, since `NOT NULL` is NULL in SQL
  # and would hide exactly the orphan parent). Canonical: every parent RESOLVES to an object row and none sits at a refit-bank
  # location — a parent whose record has not folded yet, or an empty list, is in NEITHER scope (the INNER JOIN read an unfolded second
  # parent as absent and the V as canonical; derived_integrity_sql below then names such a row, so it is refused, never rebuilt over).
  case "$1" in
    self) _self object;;
    parent) if [ -n "${REFIT:-}" ]; then echo "(json_type(object.parent_object_ids_json) = 'array' AND json_array_length(object.parent_object_ids_json) > 0 AND NOT EXISTS (SELECT 1 FROM json_each(object.parent_object_ids_json) pe LEFT JOIN object p ON p.object_id = pe.value WHERE p.object_id IS NULL OR NOT COALESCE($(_self p), 0)))"
            else echo "(NOT COALESCE(json_type(object.parent_object_ids_json) = 'array' AND json_array_length(object.parent_object_ids_json) = 0, 0) AND NOT EXISTS (SELECT 1 FROM json_each(object.parent_object_ids_json) pe LEFT JOIN object p ON p.object_id = pe.value WHERE p.object_id IS NULL OR $(_loc p "${DEFAULT_BANK}_*")))"; fi;;
    *) return 2;;
  esac
}
# the rows NEITHER scope can place (refit-fix-1): a row at a refit-bank location whose extra.init_scheme is NULL makes the
# refit predicate NULL and the canonical `NOT (...)` NULL too — SQL drops it from both. Such a row is named, never silently lost.
refit_orphans_sql(){   # self | parent -> a SQL predicate over the outer `object` row, or "0 = 1" when no refit can exist for this lane
  # a FITTED row is an orphan when its own extra.init_scheme is NULL at a refit-bank location; a DERIVED row (the composed V) carries no
  # init scheme of its own — it is an orphan when ANY of its parents is one (the self rule named the freshly folded refit V, a false
  # positive; a first-parent-only rule missed an orphaned SECOND parent)
  local scheme="${REFIT_SCHEME:-}"
  if [ -z "$scheme" ] || [ "$scheme" = zero ] || [ "$scheme" = unreadable ]; then echo "0 = 1"; return 0; fi
  _orphan_row(){   # <alias> -> that row's own init scheme is NULL at a refit-bank location
    echo "(json_extract($1.extra_json, '\$.init_scheme') IS NULL AND EXISTS (SELECT 1 FROM json_each($1.artifact_locations_json) l WHERE (CASE l.type WHEN 'object' THEN json_extract(l.value, '\$.uri') ELSE l.value END) GLOB '*/exp/objects/sdfbase/${DEFAULT_BANK}_*/*'))"
  }
  case "${1:-self}" in
    self) _orphan_row object;;
    parent) echo "EXISTS (SELECT 1 FROM json_each(object.parent_object_ids_json) pe JOIN object p ON p.object_id = pe.value WHERE $(_orphan_row p))";;
    *) return 2;;
  esac
}
# the rows a REFIT run cannot place because their recorded scheme CONTRADICTS the committed config at this run's bank: a control row
# fitted zero-init into full22_<tag> is wanted by neither scope — the refit scope needs the scheme, the canonical scope refuses the
# location — and `_orphan_row` tests IS NULL only, so the unit used to continue silently (a re-fit of an 11-hour cell for a fitted
# unit; for the composed unit a published V waiting forever on a parent it had excluded, or the prep/republish branch).
# Scoped to THIS bank/tag: a row at another tag's bank carries that tag's legitimate scheme and is simply not this run's.
refit_mismatch_sql(){   # self | parent -> a SQL predicate over the outer `object` row, or "0 = 1" outside a refit run
  local scheme="${REFIT_SCHEME:-}"
  if [ -z "${REFIT:-}" ] || [ -z "$scheme" ] || [ "$scheme" = zero ] || [ "$scheme" = unreadable ]; then echo "0 = 1"; return 0; fi
  _mismatch_row(){   # <alias> -> that row records an init scheme other than the committed one at THIS run's bank
    echo "(json_extract($1.extra_json, '\$.init_scheme') IS NOT NULL AND json_extract($1.extra_json, '\$.init_scheme') <> '$scheme' AND EXISTS (SELECT 1 FROM json_each($1.artifact_locations_json) l WHERE (CASE l.type WHEN 'object' THEN json_extract(l.value, '\$.uri') ELSE l.value END) GLOB '*/exp/objects/sdfbase/${DEFAULT_BANK}_${REFIT}/*'))"
  }
  case "${1:-self}" in
    self) _mismatch_row object;;
    parent) echo "EXISTS (SELECT 1 FROM json_each(object.parent_object_ids_json) pe JOIN object p ON p.object_id = pe.value WHERE $(_mismatch_row p))";;
    *) return 2;;
  esac
}
# a DIRECT paired composition's parents ARE its identity (V = +P_mix - P_ctl): exactly two ids, resolving to one constitution and one
# control relu_steer row of the same fit model. A list that is empty, short, long or duplicated, or carries an id with no object row yet
# (a parent whose record has not folded), is malformed for done-ness in EVERY run, refit or canonical (the canonical predicate's INNER
# JOIN read such a parent as absent and the V as canonical, and registry_done marked DONE on provenance it could not check; the
# registry's CHECK verifies JSON syntax and non-NULL only). Keyed on derivation `composition`: dose_scale rows carry ONE parent (a
# composed V) by design and the live registry holds approved ones in this unit's candidate set — the rule does not touch them, nor
# lora_subtraction rows.
derived_integrity_sql(){   # self | parent -> a SQL predicate naming the MALFORMED rows (self: a fitted row has no parents to check)
  _parents_of_role(){   # <role> -> how many of this row's parents resolve to a relu_steer row of that role and this row's fit model
    echo "(SELECT count(*) FROM json_each(object.parent_object_ids_json) pe JOIN object p ON p.object_id = pe.value WHERE p.object_kind = 'relu_steer' AND p.object_role = '$1' AND p.fit_model_id = object.fit_model_id)"
  }
  case "${1:-self}" in
    self) echo "0 = 1";;
    parent) echo "(object.derivation = 'composition' AND NOT COALESCE(json_type(object.parent_object_ids_json) = 'array' AND json_array_length(object.parent_object_ids_json) = 2 AND $(_parents_of_role constitution) = 1 AND $(_parents_of_role control) = 1, 0))";;
    *) return 2;;
  esac
}
# … and their ORDER is part of that identity (refit-fix-6): the producer emits [parent_pos, parent_neg]
# (psr_residual_distill._derived_hook_call; the sidecar adapter keeps the order), and V = u_pos - u_neg = +constitution - control. A
# composed artifact built the other way round — both parents the right model, bank and scheme — passed the arity/role count and could
# be marked DONE once a liveness receipt matched its bytes, certifying the OPPOSITE steering direction. Index 0 must resolve to the
# constitution relu_steer row and index 1 to the control relu_steer row of the fit model — and, when this run names the corpora
# (MIX_SI / CTL_SI, as every live parent row carries fit_corpus_id), to those corpora. Keyed on derivation `composition` like the
# integrity rule; every live 2-parent composed row (10 of 10, read-only 2026-10-04) is in this order.
derived_order_sql(){   # self | parent -> a SQL predicate naming the REVERSED (or otherwise mis-positioned) rows; self: "0 = 1"
  _parent_at(){   # <index> <role> <corpus or ''> -> that position resolves to a relu_steer row of that role (and corpus) and this row's fit model
    local corpus=""; [ -z "$3" ] || corpus=" AND p.fit_corpus_id = '$3'"
    echo "EXISTS (SELECT 1 FROM object p WHERE p.object_id = json_extract(object.parent_object_ids_json, '\$[$1]') AND p.object_kind = 'relu_steer' AND p.object_role = '$2' AND p.fit_model_id = object.fit_model_id$corpus)"
  }
  local mix="" ctl=""; if [ -n "${MIX_SI:-}" ] && [ -n "${CTL_SI:-}" ]; then mix="$MIX_SI"; ctl="$CTL_SI"; fi
  case "${1:-self}" in
    self) echo "0 = 1";;
    parent) echo "(object.derivation = 'composition' AND NOT COALESCE($(_parent_at 0 constitution "$mix") AND $(_parent_at 1 control "$ctl"), 0))";;
    *) return 2;;
  esac
}
# registry state: object ids matching a WHERE clause, one per line; empty when the database is absent or unreachable. The second
# positional (self | parent, default self) is the refit scope (refit_scope_sql) every query carries. Rows the scope cannot place (refit_orphans_sql) REFUSE the decision by name on stderr (stdout is the ids the caller reads).
registry_object_ids(){   # <where> [self|parent]
  local root scope q_orph q_ids gated orphans mism malf rev; root="$(eval echo "$SISTEER_REGISTRY_ROOT")"
  if [ ! -r "$root/registry.sqlite" ]; then
    # refit-fix-3: a MISSING or unreadable registry.sqlite used to return success with zero ids, so registry_done read
    # registered work as unregistered and a live run could submit a second costly fit or derived prep that no registry can fold
    # (repro_load_config checks only that the root string is non-empty). A LIVE run fails closed here; a DRY run — which submits
    # nothing — prints the same line and proceeds as unregistered, so the dry-run harnesses without a registry still print their plans
    # (the repro_require_clean_tree carve-out).
    echo "registry_unavailable: $root/registry.sqlite missing or unreadable (SISTEER_REGISTRY_ROOT=$SISTEER_REGISTRY_ROOT)" >&2
    if [ "${REPRO_DRY_RUN:-0}" = 1 ]; then repro_note "registry_unavailable: a dry run proceeds as if nothing were registered; a LIVE run refuses here (D1885)" >&2; return 0; fi
    repro_fatal "registry_object_ids: the registry is unavailable — the idempotence / orphan decision cannot be made (a missing or unreadable database is never zero rows); mount or restore $root/registry.sqlite, then re-run (D1885)"
  fi
  scope="$(refit_scope_sql "${2:-self}")" || repro_fatal "registry_object_ids: unknown refit scope '${2:-}' (self | parent)"
  # refit-fix-2: every CLI call runs into a file with ITS OWN exit status — never through a pipeline whose emptiness is
  # then read as "no rows"; its stderr is surfaced; a non-zero exit is `registry_unavailable` and REFUSES the idempotence / orphan
  # decision (the script is `set -uo pipefail` without -e, so a locked or unreadable registry, or a json_extract failure on a malformed
  # historical row, used to read as zero orphans and zero ids, and the operator proceeded to another expensive fit)
  _registry_query(){   # <sql> <out-file>: the CLI's stdout into out-file; stderr surfaced; a failed CLI refuses by name
    local sql="$1" out="$2" err rc; err="$(mktemp)"
    "$(repro_python)" -m sisteer.registry.cli --root "$root" query --json "$sql" >"$out" 2>"$err"; rc=$?
    [ -s "$err" ] && sed 's/^/registry stderr: /' "$err" >&2
    if [ "$rc" -ne 0 ]; then
      echo "registry_unavailable: $(tr '\n' ' ' < "$err" | cut -c1-600)" >&2
      rm -f "$err" "$out"
      repro_fatal "registry_object_ids: the registry CLI exited $rc for root $root — the idempotence / orphan decision cannot be made (a failed read is never zero rows); fix the registry read, then re-run (D1862)"
    fi
    rm -f "$err"
  }
  q_orph="$(mktemp)"; q_ids="$(mktemp)"
  # the rows NEITHER scope can place — named on stderr and REFUSED, never lost, never read from a failed query, never rebuilt over:
  #   init_scheme_missing (EVERY parent): a row at a refit-bank location with no recorded
  #     extra.init_scheme — a fitted row, or a derived row's parent.
  #   init_scheme_mismatch: a row at THIS run's bank whose recorded scheme is not the committed config's.
  #   parents_malformed: a direct paired composition whose parent list is not exactly two ids resolving to one
  #     constitution and one control relu_steer row of its model (a parent not folded yet, a missing/extra/duplicated id, an empty list).
  #   parents_reversed: a direct paired composition whose index 0 is not the constitution row or whose index 1
  #     is not the control row (of the fit model and, when named, the corpora) — a V composed the other way round steers the other way.
  # ONE query over the unit's candidate set, each row with its reason. A fitted row with no recorded init scheme at a refit-bank location,
  # or a derived row with such a parent, is neither done nor rebuilt: the operator repairs the row or waits for the fold (never re-fits an
  # 11-hour cell, never recomposes a V over a parent whose identity cannot be told).
  _registry_query "SELECT object_id || ' ' || CASE WHEN $(refit_orphans_sql "${2:-self}") THEN 'init_scheme_missing' WHEN $(refit_mismatch_sql "${2:-self}") THEN 'init_scheme_mismatch' WHEN $(derived_integrity_sql "${2:-self}") THEN 'parents_malformed' ELSE 'parents_reversed' END FROM object WHERE $1 AND (($(refit_orphans_sql "${2:-self}")) OR ($(refit_mismatch_sql "${2:-self}")) OR ($(derived_integrity_sql "${2:-self}")) OR ($(derived_order_sql "${2:-self}"))) AND status IN ('reference','approved')" "$q_orph"
  gated="$(grep -oE '\b[0-9a-f]{64} (init_scheme_missing|init_scheme_mismatch|parents_malformed|parents_reversed)\b' "$q_orph" | sort -u)"
  orphans="$(awk '$2 == "init_scheme_missing" {print $1}' <<<"$gated" | tr '\n' ' ')"
  mism="$(awk '$2 == "init_scheme_mismatch" {print $1}' <<<"$gated" | tr '\n' ' ')"
  malf="$(awk '$2 == "parents_malformed" {print $1}' <<<"$gated" | tr '\n' ' ')"
  rev="$(awk '$2 == "parents_reversed" {print $1}' <<<"$gated" | tr '\n' ' ')"
  if [ -n "$gated" ]; then
    [ -z "$orphans" ] || echo "registry: $(echo "$orphans" | wc -w) object row(s) in this unit's candidate set carry NO extra.init_scheme at a refit-bank location (a fitted row, or a derived row's parent) and fall in NEITHER the canonical nor a refit scope: $orphans" >&2
    [ -z "$mism" ] || echo "registry: $(echo "$mism" | wc -w) object row(s) in this unit's candidate set sit at bank ${DEFAULT_BANK}_${REFIT:-} with a recorded extra.init_scheme that is NOT the committed config's '${REFIT_SCHEME:-}' (a fitted row, or a derived row's parent) — an inconsistent record (D2116): $mism" >&2
    [ -z "$malf" ] || echo "registry: $(echo "$malf" | wc -w) direct paired-composition row(s) (derivation composition) in this unit's candidate set do not carry exactly two parent ids resolving to one constitution and one control relu_steer row of their fit model — a parent not folded yet, a missing/extra id, or an empty list (D2116): $malf" >&2
    [ -z "$rev" ] || echo "registry: $(echo "$rev" | wc -w) direct paired-composition row(s) in this unit's candidate set carry their two parents in the WRONG ORDER — index 0 must be the constitution relu_steer row${MIX_SI:+ (corpus $MIX_SI)} and index 1 the control row${CTL_SI:+ (corpus $CTL_SI)} of their fit model; the producer emits [parent_pos, parent_neg] and V = +constitution -control, so a reversed composition steers the OTHER way (D2126): $rev" >&2
    rm -f "$q_orph" "$q_ids"
    repro_fatal "registry_object_ids: unverifiable provenance in the registry for this unit (the rows named above) — the done-ness decision cannot be made and nothing is rebuilt over them; repair the row(s) or wait for the fold, then re-run (D2067 / D2116 / D2126)"
  fi
  _registry_query "SELECT object_id FROM object WHERE $1 AND $scope AND status IN ('reference','approved')" "$q_ids"
  "$(repro_python)" -c '
import json, re, sys
txt = open(sys.argv[1]).read()
ids = []
try:
    rows = json.loads(txt)
    for r in (rows if isinstance(rows, list) else []):
        v = r[0] if isinstance(r, (list, tuple)) else (r.get("object_id") if isinstance(r, dict) else r)
        if isinstance(v, str): ids.append(v)
except Exception:
    ids = re.findall(r"\b[0-9a-f]{64}\b", txt)
print("\n".join(dict.fromkeys(ids)))' "$q_ids"
  rm -f "$q_orph" "$q_ids"
}
# the cluster's view: a path present? a job of this name queued/running? (unknown under --dry-run: treated as absent)
cluster_has(){ [ "$REPRO_DRY_RUN" = 1 ] && return 1; ssh -o BatchMode=yes -o ConnectTimeout=30 "$CL_HOST" "[ -s '$1' ]" 2>/dev/null; }
# the queue is read ONCE per invocation (no unit checks a job name an earlier unit of the same run submitted)
cluster_job_active(){   # <job name> -> 0 active | 1 not active; an UNREADABLE queue REFUSES by name — fail CLOSED (03_hpo.sh's shape).
  # The ssh / squeue exit status was dropped and an error read as an EMPTY queue, cached for the invocation, so with the probe down every
  # unit read its job idle and submitted: for the refusal direction, a second key whose queue probe timed out while the first key's fit of
  # the same identity and rung was queued submitted a second fit (two objects at one identity and rung at the fold, which the pins writer
  # refuses), and the sidecar-repair branch resubmitted over a running repair. Every submission path of this stage (fit_unit,
  # difference_relu, refusal_direction and its repair, published_unfolded's wait) reads the queue here before it submits, so nothing is
  # submitted over an unknown queue. The sentinel tells "squeue answered nothing" (rc 0, the sentinel printed) from "nothing answered"
  # (ssh 255, squeue non-zero); a dry run reads no queue and answers not-active, as before.
  [ "$REPRO_DRY_RUN" = 1 ] && return 1
  if [ -z "${QUEUE_NAMES+x}" ]; then
    local out rc; out="$(ssh -o BatchMode=yes -o ConnectTimeout=30 "$CL_HOST" "squeue -h -u \$USER -o %j && echo __QUEUE_OK__" 2>&1)"; rc=$?
    case "$rc:$out" in
      0:*__QUEUE_OK__*) QUEUE_NAMES="${out%__QUEUE_OK__*}";;
      *) repro_fatal "the queue on $CLUSTER could not be read (ssh/squeue rc=$rc${out:+: ${out:0:200}}) — whether a job named $1 is queued or running is UNKNOWN, never 'not active': nothing is submitted (a submission over an unknown queue is the second fit of an identity whose first is queued); re-run when $CL_HOST answers";;
    esac
  fi
  grep -qxF -- "$1" <<<"$QUEUE_NAMES"
}
cluster_cat(){ [ "$REPRO_DRY_RUN" = 1 ] && return 1; ssh -o BatchMode=yes -o ConnectTimeout=30 "$CL_HOST" "cat '$1'" 2>/dev/null; }
# the emission evidence beside a published dir: the emitter writes registry_emission.json
# ({object_id, status: emitted|not_emitted|failed, records|reason|error}) after every attempt and a bare REGISTRY_EMIT_FAILED
# marker on failure (train_two_si_objects.REGISTRY_EMISSION_NAME / fit_refusal_direction.EMISSION_NAME — one spelling)
emission_state(){   # <published dir> -> "<status>\t<detail>": emitted | not_emitted | failed | unverified (no evidence at all)
  local d="$1" txt
  if cluster_has "$d/REGISTRY_EMIT_FAILED"; then
    txt="$(cluster_cat "$d/REGISTRY_EMIT_FAILED" | "$(repro_python)" -c '
import json, sys
try:
    e = json.load(sys.stdin); print(e.get("error") or e.get("reason") or "no reason recorded")
except Exception:
    print("unparseable marker")')"
    printf 'failed\t%s\n' "$txt"; return 0
  fi
  if cluster_has "$d/registry_emission.json"; then
    # `emitted` is terminal only for THIS run's campaign under the CURRENT emission contract: evidence under
    # another campaign (the emitters' old `two_si` default) or an older contract (a record the registry may have refused) is
    # `stale` — the unit is resubmitted to REPLAY its record (the job validates the kept object, never retrains or rebuilds)
    cluster_cat "$d/registry_emission.json" | "$(repro_python)" -c '
import json, sys
want_campaign, want_contract = sys.argv[1], sys.argv[2]
try:
    e = json.load(sys.stdin); st = e.get("status") or "unverified"
    det = e.get("reason") or e.get("error") or "object %s, %d record file(s)" % ((e.get("object_id") or "?")[:16], len(e.get("records") or []))
    if st == "emitted":
        got_campaign, got_contract = e.get("campaign"), str(e.get("contract") or "1")
        if got_campaign != want_campaign:
            st, det = "stale", "emitted under campaign %r, this run states %r" % (got_campaign, want_campaign)
        elif got_contract != want_contract:
            st, det = "stale", "emitted under emission contract %s, current %s (its record may be inadmissible)" % (got_contract, want_contract)
except Exception:
    st, det = "unverified", "unparseable registry_emission.json"
print("%s\t%s" % (st, det))' "$CAMPAIGN" "$EMISSION_CONTRACT"
    return 0
  fi
  printf 'unverified\tno emission evidence beside COMPLETE\n'
}
# a unit whose artefact is COMPLETE on the cluster but whose object row is not in the registry: returns 0 when nothing is to be
# done this invocation (emitted: waiting for the fold; or its job is still active), 1 when the caller should resubmit (NO evidence:
# the job ended between COMPLETE and its record, or predates verified emission — the resubmitted job validates COMPLETE and
# replays the record without retraining); a FAILED or NEVER-emitted record REFUSES by name (non-zero): COMPLETE alone is never
# read as "published"
published_unfolded(){   # <unit> <published dir> <job name>
  local unit="$1" d="$2" name="$3" st det
  IFS=$'\t' read -r st det < <(emission_state "$d")
  case "$st" in
    emitted) repro_note "$unit: $d is published on $CLUSTER and its object record was emitted ($det) — waiting for the fold; not marking done, not resubmitting"; return 0;;
    stale) if cluster_job_active "$name"; then repro_note "$unit: $d is published on $CLUSTER with stale emission evidence ($det) and job $name is still queued/running — waiting"; return 0; fi
           repro_note "$unit: $d is published on $CLUSTER but its emission evidence is STALE ($det) — resubmitting to REPLAY the record under campaign $CAMPAIGN; the job validates the kept object and does not retrain or rebuild (review P2 on f680858bc)"; return 1;;
    failed) repro_fatal "$unit: $d is published on $CLUSTER but its registry emission FAILED ($d/REGISTRY_EMIT_FAILED: $det) — fix the spool / push on $CLUSTER and resubmit the job: it validates COMPLETE and replays the record without retraining";;
    not_emitted) repro_fatal "$unit: $d is published on $CLUSTER but its object record was never emitted ($det) — registry emission was off for that job; enable it (SISTEER_REGISTRY_SPOOL / REGISTRY_EMIT) and resubmit: the job validates COMPLETE and replays the record without retraining";;
    *) if cluster_job_active "$name"; then repro_note "$unit: $d is published on $CLUSTER with no emission evidence yet and job $name is still queued/running — waiting"; return 0; fi
       repro_note "$unit: $d is published on $CLUSTER but carries no emission evidence ($det: the job ended between COMPLETE and its record, or predates verified emission) — resubmitting to replay the record; the job validates COMPLETE and does not retrain"; return 1;;
  esac
}
mark_done(){   # <unit> <ids...>: DONE.<unit> + one `registry: object <id>` line per id
  local unit="$1"; shift
  local id; for id in "$@"; do repro_registry_id object "$id"; done
  repro_mark_done "$REPRO_MODEL" "$STAGE" "$unit" "$*"
}
REGISTRY_UNREADY_IDS=""   # set by registry_done when the object IS registered but its readiness function refused (the caller's branch)
registry_done(){   # <unit> <where | ids> [<ready fn>] [self|parent|ids]: 0 (and marked) when the registry already holds the unit's object UNDER THIS RUN'S CAMPAIGN
  # the campaign is part of done-ness: an object present only under another campaign (the emitters' old `two_si` default) is not this
  # run's object — the unit falls through to the cluster read, whose stale evidence schedules the replay.
  # A readiness function, when named, must accept the ids (the composed difference row used to be filed by compose-v long before V's
  # liveness probe, so the row alone is NOT done-ness — v_liveness_ready reads the probe's receipt bound to V's bytes on the cluster);
  # a registered-but-unready object leaves its ids in REGISTRY_UNREADY_IDS and returns 1
  # Scope `ids`: the second positional IS the ids (one per line), read already by the unit's OWN reader — the refusal direction's
  # refusal_registry_ids, the pins writer's Python classification of a row's rung, which no WHERE clause restates without a second
  # reading that can disagree; the decision below is the one every unit's is, in ONE function.
  REGISTRY_UNREADY_IDS=""
  local ids   # declared apart from the assignment so `$?` below is the query's, not `local`'s
  if [ "${4:-self}" = ids ]; then ids="$2"; else
    ids="$(registry_object_ids "($2) AND campaign='$CAMPAIGN'" "${4:-self}")" || repro_fatal "$1: registry_object_ids refused (registry_unavailable above) — the done-ness of $1 cannot be decided, and an undecided unit is never submitted (D1862)"   # the refit scope: self (fitted) | parent (derived)
  fi
  [ -n "$ids" ] || return 1
  if [ -n "${3:-}" ] && ! "$3" "$ids"; then
    REGISTRY_UNREADY_IDS="$ids"
    repro_note "$1: in the registry under campaign $CAMPAIGN but not admission-ready on $CLUSTER ($3 refused$( [ "$REPRO_DRY_RUN" = 1 ] && echo "; cluster state is unknown under --dry-run" )) — the unit runs"
    return 1
  fi
  # shellcheck disable=SC2086
  mark_done "$1" $ids
  repro_say "$1: already in the registry under campaign $CAMPAIGN — nothing to do"
}
# the admission-ready liveness bound to V's bytes (seedrep/derived_admission.py's psr_paired_write branch refuses a V without
# gate_liveness.json, and refuses one with ANY layer at or below the dead floor): the cluster's CURRENT files are hashed on the cluster
# (never assumed from the receipt alone) and must agree with BOTH receipt fields (object_sha256 == sha256(object.pt) == one of the
# registry's ids for the unit; gate_liveness_sha256 == sha256(gate_liveness.json)), the receipt must attest not-fully-dead and the probe
# map itself must have no layer <= the dead floor (the admission's criterion). Unknown under --dry-run (cluster reads are absent there),
# so a dry run always prints the prep plan.
read -r -d '' LIVENESS_STATE_PY <<'PYLS' || true
import hashlib, json, os, sys
d = sys.argv[1]
def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest() if os.path.isfile(p) and os.path.getsize(p) > 0 else None
def doc(p):
    try:
        return json.load(open(p)) if os.path.isfile(p) else None
    except ValueError:
        return {"_unreadable": True}
print(json.dumps({"object_sha256": sha(os.path.join(d, "object.pt")), "gate_liveness_sha256": sha(os.path.join(d, "gate_liveness.json")),
                  "gate_liveness": doc(os.path.join(d, "gate_liveness.json")), "receipt": doc(os.path.join(d, "gate_liveness_receipt.json"))}))
PYLS
cluster_liveness_state(){   # <paired dir> -> one JSON line of the cluster's CURRENT hashes + files, or non-zero when unknown (--dry-run / unreachable)
  [ "$REPRO_DRY_RUN" = 1 ] && return 1
  ssh -o BatchMode=yes -o ConnectTimeout=30 "$CL_HOST" "python3 - '$1'" <<<"$LIVENESS_STATE_PY" 2>/dev/null
}
v_liveness_ready(){   # <object ids, one per line>
  local ids="$1" state why rc
  state="$(cluster_liveness_state "$PAIRED")" || return 1
  why="$("$(repro_python)" -c '
import json, sys
st = json.loads(sys.argv[1]); ids = set(sys.argv[2].split())
rc, gl = st.get("receipt") or {}, st.get("gate_liveness")
problems = []
if not st.get("object_sha256"): problems.append("no object.pt on the cluster")
if not st.get("gate_liveness_sha256"): problems.append("gate_liveness.json absent or empty")
if not rc: problems.append("gate_liveness_receipt.json absent")
if not problems:
    if rc.get("object_sha256") != st["object_sha256"]: problems.append("the receipt names object %s, V on the cluster is %s" % (str(rc.get("object_sha256"))[:16], st["object_sha256"][:16]))
    if st["object_sha256"] not in ids: problems.append("V on the cluster (%s) is not one of the registry ids for the unit" % st["object_sha256"][:16])
    if rc.get("gate_liveness_sha256") != st["gate_liveness_sha256"]: problems.append("the receipt names a gate_liveness.json other than the one on the cluster (probe re-run or edited)")
    if rc.get("not_fully_dead") is not True: problems.append("the receipt does not attest not-fully-dead (not_fully_dead=%r)" % (rc.get("not_fully_dead"),))
    floor = rc.get("dead_floor", 0.0)
    try:
        dead = [k for k, v in (gl or {}).items() if float(v) <= float(floor)]
        if not gl: problems.append("gate_liveness.json is not a liveness map")
        elif dead: problems.append("fully dead layer(s) %s (live fraction <= %s; the admission refuses this V)" % (", ".join(sorted(dead)), floor))
    except (TypeError, ValueError, AttributeError):
        problems.append("gate_liveness.json values are not all numbers")
print("; ".join(problems))
sys.exit(1 if problems else 0)' "$state" "$ids")"; rc=$?
  [ $rc -eq 0 ] || repro_note "v_liveness_ready: $why"
  return $rc
}
difference_relu_done_marker_valid(){   # a pre-existing DONE.difference_relu is honoured only while the registry row AND V's bound liveness still hold
  local ids
  ids="$(registry_object_ids "object_kind='composed' AND object_role='difference' AND fit_model_id='$FIT_ID' AND campaign='$CAMPAIGN'" parent)" || repro_fatal "difference_relu: registry_object_ids refused (registry_unavailable above) — the DONE marker's validity cannot be decided (D1862)"
  [ -n "$ids" ] && v_liveness_ready "$ids"
}
refit_fitted_marker_valid(){   # <control_relu|constitution_relu>: a pre-existing DONE.<unit>.<tag> is honoured only while the registry still holds the unit's row in THIS run's scope
  # repro_done checks the marker FILE alone, and the dispatcher used to `continue` on it for the fitted units before fit_unit /
  # registry_done could query — so a refit cell marked DONE earlier whose reference row was later corrected to a wrong
  # extra.init_scheme at the refit bank (or a contradictory row folding there) read DONE on re-run, and an unavailable registry bypassed
  # the fail-closed read. The SAME candidate-scoped query fit_unit -> registry_done issues decides: its orphan / mismatch gate refuses
  # by name (nothing skipped), an empty answer means a stale marker (not honoured, the unit runs), ids mean the marker holds.
  local role="${1%_*}" fam="${1#*_}" ids
  ids="$(registry_object_ids "object_kind='$(kind_for "$fam")' AND object_role='$role' AND fit_model_id='$FIT_ID' AND fit_corpus_id='$(si_for "$role")' AND campaign='$CAMPAIGN'" self)" || repro_fatal "$1: registry_object_ids refused (registry_unavailable or an unplaceable row above) — the DONE.$1.$REFIT marker cannot be honoured and the unit is not skipped (D1885 / D2126)"
  [ -n "$ids" ]
}
job_env(){   # the job-side env prefix (K=V …, shell-quoted) every submission carries (§3.1)
  if [ "$HAVE_SLURM_LIB" = 1 ]; then slurm_job_env "$CLUSTER"; else
    printf 'SISTEER_SHARED_ROOT=%q SISTEER_CLUSTER_LABEL=%q SISTEER_REGISTRY_SPOOL=%q REGISTRY_CAMPAIGN=%q SDFBASE_FIT_ENV_NAME=%q' \
      "$SHARED_ROOT" "$(cluster_var "$CLUSTER" CLUSTER_LABEL)" "${SISTEER_REGISTRY_SPOOL:-$SHARED_ROOT/registry/records}" "$CAMPAIGN" "$(cluster_fit_env_name "$CLUSTER")"
  fi
}
submit(){   # <unit> <name> <script> "<args>" <time> [--array A] [--env "K=V …"] [--flags "--gres=… …"]
  local unit="$1" name="$2" script="$3" args="$4" tlim="$5"; shift 5
  if [ "$HAVE_SLURM_LIB" = 1 ]; then
    slurm_submit --unit "$unit" --lwid "$(lwid "$unit")" --script "$script" --name "$name" --args "$args" \
                 --clusters "$CLUSTER" --time "$tlim" --kind fit --wave-sha "$SHA" "$@"
    return
  fi
  local extra="" arr="" envp="" flags=""
  while [ $# -gt 0 ]; do case "$1" in --array) arr=" --array=$2"; shift 2;; --env) envp="$2 "; shift 2;; --flags) flags="${2:+ $2}"; shift 2;; *) shift;; esac; done
  extra="$(cluster_var "$CLUSTER" CLUSTER_SBATCH_EXTRA)"
  local acct; acct="$(cluster_var "$CLUSTER" CLUSTER_ACCOUNT)"
  local line
  line="cd $CHECKOUT && $(job_env) ${envp}sbatch --qos=$(cluster_var "$CLUSTER" CLUSTER_QOS) --partition=$(cluster_var "$CLUSTER" CLUSTER_PARTITION)${acct:+ --account=$acct} --time=$tlim --output=$(cluster_var "$CLUSTER" CLUSTER_LOG_DIR)/%x_%j.out -J $name${extra:+ $extra}$arr$flags --comment=sha:$SHA $script $args"
  if [ "$REPRO_DRY_RUN" = 1 ]; then echo "DRY-RUN: ssh $CL_HOST '$line'"; return 0; fi
  repro_fatal "reproduce/lib/slurm.sh not present — cannot submit $unit (the line would be: $line)"
}

# --- the fitted cells (control / constitution x lora / relu) ---------------------------------------------------------------
si_for(){ case "$1" in control) echo "$CTL_SI";; constitution) echo "$MIX_SI";; esac; }
arm_for(){ case "$1" in lora) echo lora_r4_ce;; relu) echo psr_ce_all;; esac; }
kind_for(){ case "$1" in lora) echo lora;; relu) echo relu_steer;; esac; }
fit_unit(){   # <object>: control_lora | constitution_lora | control_relu | constitution_relu
  local unit="$1" role="${1%_*}" fam="${1#*_}" si arm cell name
  si="$(si_for "$role")"; arm="$(arm_for "$fam")"; cell="${si}__${arm}"; name="fit-${REPRO_MODEL}-${unit}${REFIT:+-$REFIT}"
  registry_done "$unit${REFIT:+.$REFIT}" "object_kind='$(kind_for "$fam")' AND object_role='$role' AND fit_model_id='$FIT_ID' AND fit_corpus_id='$si'" && return 0
  if [ "$fam" = lora ] && [ -e "$(repro_state_dir "$REPRO_MODEL")/03_hpo/lora/hpo_pick.HOLD.json" ]; then
    # the architecture pick is HELD (03_hpo.sh --phase pick found the promoted pick's basis changed) — nothing submitted on it
    local hold; hold="$(repro_state_dir "$REPRO_MODEL")/03_hpo/lora/hpo_pick.HOLD.json"
    local hmsg="LoRA fit $unit: the architecture pick is HELD ($hold: $("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("reason"))' "$hold" 2>/dev/null)) — its basis changed after the promotion; nothing submitted until 03_hpo.sh --phase pick re-derives the promotion from the current evidence"
    [ "$REPRO_DRY_RUN" = 1 ] || repro_fatal "$hmsg"
    repro_note "would REFUSE live: $hmsg"
  fi
  if [ "$fam" = lora ] && [ -z "$RULING" ]; then
    local msg="LoRA fit $unit needs models.yaml models.$REPRO_MODEL.lora_architecture.selection_ruling_id (FD 6a: the user's architecture pick, plan §3.11 phase A — run 03_hpo.sh --phase pick first)"
    [ "$REPRO_DRY_RUN" = 1 ] || repro_fatal "$msg"
    repro_note "would REFUSE live: $msg — printing the submission it would make"
  fi
  # published but not folded: emitted -> wait; failed / never emitted -> refuse; no evidence -> fall through and resubmit (the
  # job validates COMPLETE and replays the record, never the fit). Under --refit the kept cell's identity IS its bank (derived
  # from the tag — the hand-submitted refits carry no label and none is written).
  if cluster_has "$CELLS/$cell/COMPLETE" && published_unfolded "$unit" "$CELLS/$cell" "$name"; then return 0; fi
  if cluster_job_active "$name"; then repro_note "$unit: job $name is already queued/running on $CLUSTER"; return 0; fi
  [ -n "$RULING" ] && repro_note "$unit: architecture ruling $RULING (recorded on the object by the trainer's fit_selection.json)"
  # the width from the config sizes the allocation (repro_fit_width_flags; empty at width 1) and rides the job env as a
  # cross-check the sbatch makes against the config it reads; the fit's effective batch is the same number at every width
  [ "$FIT_WIDTH" -gt 1 ] 2>/dev/null && repro_note "$unit: data-parallel fit width $FIT_WIDTH (fit_width in $(basename "$FIT_CFG_LOCAL")): $(repro_fit_width_flags "$FIT_WIDTH"); torchrun --nproc_per_node=$FIT_WIDTH inside the job"
  [ -n "${REFIT:-}" ] && repro_note "$unit: REFIT $REFIT into bank $BANK (D1795 / D1816): cell $CELLS/$cell; the registry tells it apart by the artefact's init scheme and its location — no label is written"
  submit "$unit${REFIT:+/$REFIT}" "$name" scripts/sbatch/sdfbase_fit_cell.sbatch "$FIT_ID $si $arm $BANK" "$SISTEER_TIME_FIT" \
         --env "SDFBASE_FIT_WIDTH=$FIT_WIDTH" --flags "$(repro_fit_width_flags "$FIT_WIDTH")"
}

# --- difference_lora: the LoRA task-vector subtraction (CPU) ------------------------------------------------------------------
difference_lora(){
  local unit=difference_lora
  registry_done "$unit" "object_kind='lora_subtraction' AND object_role='difference' AND fit_model_id='$FIT_ID'" "" parent && return 0
  if [ "$REPRO_DRY_RUN" != 1 ]; then
    local c; for c in "${MIX_SI}__lora_r4_ce" "${CTL_SI}__lora_r4_ce"; do
      cluster_has "$CELLS/$c/COMPLETE" || { repro_note "$unit: parent $c is not published on $CLUSTER yet — fit control_lora / constitution_lora first"; [ "$OBJECT" = all ] && return 0 || repro_fatal "$unit: parent $c not published"; }
    done
  fi
  local wrap="${SISTEER_CPU_SRUN-srun --partition=$(cluster_var "$CLUSTER" CLUSTER_PARTITION) --qos=$(cluster_var "$CLUSTER" CLUSTER_QOS) --ntasks=1 --cpus-per-task=4 --mem=32G --time=00:30:00 -J diff-lora-$REPRO_MODEL}"
  local cmd
  cmd="cd $CHECKOUT && $(job_env) PYTHONPATH=$CHECKOUT/src ${wrap:+$wrap }$FIT_VENV_PY scripts/sdfbase/lora_subtract.py --root $BANK_ROOT --out-root $DERIVED/lora_subtraction --models $FIT_ID --cells $MIX_SI --ctl-si $CTL_SI --pools $POOLS --keep-valid"
  if [ "$REPRO_DRY_RUN" = 1 ]; then echo "DRY-RUN: ssh $CL_HOST '$cmd'"; return 0; fi
  local out; out="$(repro_ssh "$CLUSTER" "$cmd")" || { echo "$out"; repro_fatal "$unit: lora_subtract.py failed on $CLUSTER (output above)"; }
  echo "$out"
  local ids; ids="$(printf '%s\n' "$out" | sed -n 's/^registry: object \([0-9a-f]\{64\}\).*/\1/p')"
  if [ -n "$ids" ]; then
    # shellcheck disable=SC2086
    repro_mark_done "$REPRO_MODEL" "$STAGE" "$unit" $ids
  elif cluster_has "$LORA_DIFF/derived_meta.json"; then
    # built (or kept) but no record line: the emitter's evidence beside the artefact says why —
    # never a silent "not marking done"; a --keep-valid re-run of this unit replays the record without rebuilding
    published_unfolded "$unit" "$LORA_DIFF" "diff-lora-$REPRO_MODEL" || repro_note "$unit: re-run this unit to replay the record (--keep-valid keeps the object)"
  else
    repro_note "$unit: nothing published at $LORA_DIFF (see the output above) — not marking done"
  fi
}

# --- difference_relu: the DIRECT paired difference V = +P_mix - P_ctl ------------------------------------------------------
# The difference_relu object is the undistilled paired write prep composes (scripts/sbatch/sdfbase_derived_build.sbatch `prep`:
# compose-v + verify-v, V's gate_liveness, the native registry emission — kind `composed`, paper label `difference_relu`, derivation
# `composition`, parents the two psr_ce_all cells); every in-paper difference_relu row IS this kind. The psr_residual distillation chain
# (calibrate -> assess -> finish, kind relu_steer_distilled_difference / label difference_relu_distilled, in_paper 0 for every model) is
# LEGACY: this unit never composes those stages, refuses to advance while one of them is queued or running, and reports a residual tree
# it finds as evidence, never as work. ONE stage per invocation, chosen from the cluster's state: prep, or nothing.
DERIVED_SCRIPT=scripts/sbatch/sdfbase_derived_build.sbatch
DERIVED_ARGS="$FIT_ID $MIX_SI $CTL_SI $BANK"
# the ReLU-only build: positional 6-8 at their defaults, positional 9 = relu — prep requires exactly the two psr parents and skips the
# LoRA task-vector subtraction (difference_lora is its own unit on the two LoRA parents); positional 10 = direct: V only — compose-v
# with its record deferred, verify-v, the liveness probe + receipt, emit-v; the legacy psr_residual capture-set + ladder declaration
# never run on this path
DERIVED_RELU_TAIL="- 1 5 relu direct"
PAIRED="$DERIVED/psr_paired_write/$FIT_ID/${MIX_SI}__psr_ce_all_minus_ctl"
difference_relu(){
  local unit="difference_relu${REFIT:+/$REFIT}" name="derived-${REPRO_MODEL}${REFIT:+-$REFIT}"
  # unit = the funnel unit difference_relu[/<tag>]: its sub-units are $unit/prep and $unit/liveness; the grading host's marker is
  # DONE.${unit//\//.} (difference_relu[.<tag>]), the name the dispatcher loop spells as DONE.$u${REFIT:+.$REFIT}
  # the difference_relu object IS the undistilled paired write V = +P_mix - P_ctl (kind `composed`, derivation `composition`,
  # hooks.paper_label_for) — never the distilled `relu_steer_distilled_difference`
  # done-ness = the composed row under THIS campaign AND V's liveness bound to its bytes on the cluster
  registry_done "${unit//\//.}" "object_kind='composed' AND object_role='difference' AND fit_model_id='$FIT_ID'" v_liveness_ready parent && return 0
  # the funnel unit: `difference_relu/prep` builds V; `difference_relu/liveness` is the pass over a kept, REGISTERED V (the probe +
  # receipt + emit-v replay) — a second unit of work with its own identity, because the prep unit is bound to the attempt that
  # registered V and the funnel never re-mints a registered unit. The job name stays `$name-prep`
  # (the same sbatch, direct build) for the queue reads above and below.
  local sub_unit="$unit/prep"; [ -n "$REGISTRY_UNREADY_IDS" ] && sub_unit="$unit/liveness"
  if [ "$REPRO_DRY_RUN" = 1 ]; then
    repro_note "$unit is prep's direct paired difference (compose-v with its record deferred, verify-v, the liveness probe + receipt, emit-v); ONE job ($sub_unit), then the fold:"
    submit "$sub_unit" "$name-prep" "$DERIVED_SCRIPT" "prep $DERIVED_ARGS $DERIVED_RELU_TAIL" 0:40:00
    return 0
  fi
  local c; for c in "${MIX_SI}__psr_ce_all" "${CTL_SI}__psr_ce_all"; do
    cluster_has "$CELLS/$c/COMPLETE" || { repro_note "$unit: parent $c is not published on $CLUSTER yet (the paired write needs exactly the two psr parents) — fit the cells first"; [ "$OBJECT" = all ] && return 0 || repro_fatal "$unit: parent $c not published"; }
  done
  cluster_job_active "$name-prep" && { repro_note "$unit: prep job $name-prep is queued/running on $CLUSTER — nothing to advance"; return 0; }
  local s; for s in calib assess finish; do
    cluster_job_active "$name-$s" && repro_fatal "$unit: a LEGACY psr_residual stage job $name-$s is queued/running on $CLUSTER — the distillation chain is not part of the method (USER RULING 2026-10-02); stop it (stop-run.sh $name-$s) before advancing this unit"
  done
  if [ -n "$REGISTRY_UNREADY_IDS" ]; then
    # registered, built and folded, but V carries no admission-ready liveness bound to its bytes: prep runs over the KEPT V (verify-v,
    # the probe, the receipt, then emit-v's idempotent replay) — the emission evidence is not consulted here, it is not the question
    repro_note "$unit: the composed row is in the registry under campaign $CAMPAIGN but $PAIRED carries no gate_liveness.json + receipt bound to its bytes — submitting the liveness pass (unit $sub_unit; prep's direct build: verify-v, liveness probe, receipt, emit-v replay; never a rebuild of a verified V)"
    submit "$sub_unit" "$name-prep" "$DERIVED_SCRIPT" "prep $DERIVED_ARGS $DERIVED_RELU_TAIL" 0:40:00; return
  fi
  if cluster_has "$RESID/sweep.json" || cluster_has "$RESID/object.pt"; then
    repro_note "$unit: a LEGACY psr_residual tree exists at $RESID (calibration parts / a distilled object) — left as evidence, never advanced or published by this unit"
  fi
  if cluster_has "$PAIRED/object.pt" && cluster_has "$PAIRED/derived_meta.json"; then
    # built: the record decides — emitted: wait for the fold; failed / never emitted: refuse by name; no
    # evidence: re-run prep, whose compose-v recomposes the SAME bytes from the parents (deterministic) and replays the record
    published_unfolded "$unit" "$PAIRED" "$name-prep" && return 0
    submit "$unit/prep" "$name-prep" "$DERIVED_SCRIPT" "prep $DERIVED_ARGS $DERIVED_RELU_TAIL" 0:40:00; return
  fi
  submit "$unit/prep" "$name-prep" "$DERIVED_SCRIPT" "prep $DERIVED_ARGS $DERIVED_RELU_TAIL" 0:40:00
}

# --- refusal_direction: the in-repo Arditi fitter on the SERVED checkpoint ----------------------------------------------------
# done-ness = the registry row of the identity AT THIS RUNG under this campaign (refusal_registry_ids — the identity has no
# coefficient column; the rung is the row's own statement, read by the pins writer's Python) AND its publication at $REFUSAL_DIR on
# $CLUSTER (refusal_published_ready: COMPLETE coherent, publishing one of the row's ids, the identity sidecar fit_provenance.json
# LISTED in it with its hash — the file the stager stages the cell by and the consumers' gate reads). A row without that publication
# is never "done": with the sidecar missing from the manifest, or LISTED but missing / truncated / altered on disk while every other
# listed file holds and the verified fit_meta.json projects to the listed hash (sidecar_repairable), the job is resubmitted to write
# it (the fitter's kept-publication path — ensure_fit_provenance / repair_fit_provenance, no refit); with no publication, a torn one
# (any other damage), or another object's, the unit REFUSES by name — a refit here would register a SECOND object of the same
# identity and rung, which the pins writer refuses (two reference/approved rows), so the operator mirrors the row's publication or
# withdraws the row. Unknown (no cluster read): under --dry-run the plan prints, as difference_relu's does; a LIVE run with the
# identity registered FAILS CLOSED — the probe must succeed before anything is submitted (cluster_has / cluster_job_active read an
# unreachable cluster as absent / idle, so a transient ssh or python3 failure used to fall through to a second fit of the registered
# identity and rung).
# TWO funnel units, as difference_relu's: `refusal_direction` is the fit, bound in the funnel's ledger to the attempt that registered
# the row; `refusal_direction/sidecar` is the repair-only pass over a kept, REGISTERED publication (the same sbatch and args — the
# fitter's kept path writes the sidecar and replays the record, never a refit — under the same identity job name, so the queue read
# above sees it). The funnel is idempotent by logical work id and never re-mints a bound unit (submit_twin.submit_one prints
# `skipped … already registered` and launches nothing), so a repair handed the FIT's unit was skipped forever and the publication
# stayed unstageable with every re-run asking again; the repair's own id is what makes it a submission. The grading host's marker
# stays DONE.refusal_direction (done-ness is the unit's, written by registry_done once the row AND the repaired publication hold).
# The row is read by refusal_registry_ids (below refusal_published_ready): the pins writer's own rung reading, handed to
# registry_done under scope `ids`.
refusal_direction(){
  # the job name is the IDENTITY's — the served id and the rung (the staged id spells the coefficient), the two words that name the
  # publication directory ($REFUSAL_DIR) and the registry row — never the table key's: named refusal-direction-<table key>, a second
  # key serving the same checkpoint read the queue for ITS name while the first key's fit was queued or running — before any COMPLETE
  # or row — and submitted a second fit of the same identity to the same directory. The queue read below (cluster_job_active, the
  # dispatcher's one coordination primitive) keys on this name, so every key that serves the checkpoint sees the one fit; a fit at
  # another coefficient has another name and is not suppressed by it. The root and the campaign are NOT in the name: two sites with
  # different roots, or two campaigns, are one registry identity per rung, and the row decides — a second fit under another root is
  # refused by name above (mirror the publication), never minted beside the first.
  local unit=refusal_direction name="refusal-direction-${SERVE_ID}-${REFUSAL_STAGED}" ids
  REFUSAL_PUBLICATION_PROBLEM=""; REFUSAL_PUBLICATION_WHY=""
  ids="$(refusal_registry_ids)" || repro_fatal "$unit: refusal_registry_ids refused (above) — the done-ness of $unit cannot be decided, and an undecided unit is never submitted (D1862)"
  registry_done "$unit" "$ids" refusal_published_ready ids && return 0
  if [ -n "$REGISTRY_UNREADY_IDS" ]; then
    # registered at this rung, but $REFUSAL_DIR on $CLUSTER is not the publication 05 stages (refusal_published_ready named why)
    case "$REFUSAL_PUBLICATION_PROBLEM" in
      no_identity_sidecar|sidecar_repairable)
        if cluster_job_active "$name"; then repro_note "$unit: job $name is already queued/running on $CLUSTER (its kept-publication path writes the identity sidecar)"; return 0; fi
        local how="writes $REFUSAL_IDENTITY_SIDECAR from its verified fit_meta.json, lists it in COMPLETE and replays the record (ensure_fit_provenance; no refit)"
        [ "$REFUSAL_PUBLICATION_PROBLEM" = sidecar_repairable ] && how="writes the $REFUSAL_IDENTITY_SIDECAR bytes COMPLETE lists again from its verified fit_meta.json — the one manifest problem a kept publication repairs — and replays the record (repair_fit_provenance; no refit, every other listed file verified)"
        repro_note "$unit: registered under campaign $CAMPAIGN (object $(echo $REGISTRY_UNREADY_IDS)) and published at $REFUSAL_DIR, but $REFUSAL_PUBLICATION_WHY — resubmitting the job to write it: the fitter keeps the publication, $how (funnel unit $unit/sidecar: the fit's unit is bound to the attempt that registered the row and the funnel never re-mints it)"
        refusal_submit "$unit/sidecar" "$name"; return;;
      unknown)
        # the cluster's view could not be read. Under --dry-run alone the plan prints below (a dry run submits nothing). A LIVE run fails
        # CLOSED here: the registry holds the identity at this rung, and without the probe nothing below can tell "published, repair it" from
        # "absent, refuse" — cluster_has reads an unreachable cluster as absent (and cluster_job_active read it as idle before the queue read
        # was made to refuse), so the fall-through submitted a SECOND fit of the registered identity and rung, which the pins writer then
        # refuses after the GPU hour.
        [ "$REPRO_DRY_RUN" = 1 ] || repro_fatal "$unit: the registry holds the refusal direction of $SERVE_ID at c=$COEF under campaign $CAMPAIGN (object $(echo $REGISTRY_UNREADY_IDS)) but its publication at $REFUSAL_DIR on $CLUSTER could not be read ($REFUSAL_PUBLICATION_WHY) — nothing is submitted: a registered identity is never re-fitted on a cluster whose publication state is unknown (a refit would register a SECOND object of the same identity and rung, which the pins writer refuses by name); restore the probe (ssh to $CL_HOST, python3 on its login host), then re-run";;
      *) repro_fatal "$unit: the registry holds the refusal direction of $SERVE_ID at c=$COEF under campaign $CAMPAIGN (object $(echo $REGISTRY_UNREADY_IDS)) but $REFUSAL_DIR on $CLUSTER is not its publication ($REFUSAL_PUBLICATION_PROBLEM: $REFUSAL_PUBLICATION_WHY) — a refit here would register a SECOND object of the same identity and rung, which the pins writer refuses by name (two reference/approved rows), so nothing is submitted: mirror the row's publication to $REFUSAL_DIR (the directory is the served checkpoint's, <refusal root>/<serve.checkpoint.model_id>/<staged id>, shared by every key that serves it; the row's artifact_locations_json names where the fit published) or withdraw the row, then re-run";;
    esac
  fi
  # published but not folded: the same three-way read as the fitted cells (the fitter leaves the same evidence; its resume
  # replays a missing record without refitting)
  if cluster_has "$REFUSAL_DIR/COMPLETE" && published_unfolded "$unit" "$REFUSAL_DIR" "$name"; then return 0; fi
  if cluster_job_active "$name"; then repro_note "$unit: job $name is already queued/running on $CLUSTER"; return 0; fi
  repro_note "$unit: c=$COEF (spec FD 5), sets ${SISTEER_REFUSAL_HARMFUL:-arditi:harmful_train} / ${SISTEER_REFUSAL_HARMLESS:-arditi:harmless_train} — an arditi:* set is the paper fit's own draw, read from the verified copy 00_setup.sh --what third_party installs on $CLUSTER (the job's preflight checks it); harmbench_val / alpaca must be in $CLUSTER's HF cache (lib/hf_fetch.sh, one repo per call)"
  refusal_submit "$unit" "$name"
}
refusal_submit(){   # <funnel unit: refusal_direction | refusal_direction/sidecar> <job name>: the sbatch's positional <model_key> is the SERVED
  # checkpoint's registry id — the row's fit_model_id (the fitter's --model-key) AND the publication directory's name ($REFUSAL_DIR =
  # $REFUSAL_ROOT/$SERVE_ID/<staged id>, the dir 05's cells_json reads): one word for the identity and its directory (see REFUSAL_DIR
  # above). The same sbatch and args for both units: which path the fitter takes is the publication's state on the cluster (no
  # COMPLETE: the fit; a COMPLETE of this request: kept, the sidecar written, the record replayed), never a knob.
  submit "$1" "$2" scripts/sbatch/fit_refusal_direction.sbatch "$SERVE_ID $SERVE_REPO $SERVE_REV $REFUSAL_ROOT" 01:00:00 \
         --env "REFUSAL_COEFFICIENT=$COEF REFUSAL_HARMFUL=${SISTEER_REFUSAL_HARMFUL:-arditi:harmful_train} REFUSAL_HARMLESS=${SISTEER_REFUSAL_HARMLESS:-arditi:harmless_train}"
}
REFUSAL_PUBLICATION_PROBLEM=""; REFUSAL_PUBLICATION_WHY=""   # set by refusal_published_ready: absent | torn | other_object | no_identity_sidecar | sidecar_repairable | unknown
REFUSAL_IDENTITY_SIDECAR=fit_provenance.json   # == fit_refusal_direction.PROVENANCE_NAME == the stager's IDENTITY_SIDECARS entry (tests/test_refusal_direction_pin.py)
REFUSAL_META_SIDECAR=fit_meta.json             # == fit_refusal_direction.META_NAME: the registry hook's sidecar the identity sidecar is a projection of
REFUSAL_FITTER="$REPRO_REPO/scripts/arditi/fit_refusal_direction.py"   # the projection's ONE owner (fit_provenance_text), imported by the verdict below
# the cluster's CURRENT view of the publication, hashed ON the cluster (never assumed from the registry row):
# the COMPLETE manifest, the sha256 of every file it lists, and the TEXT of the listed fit_meta.json (positional 2; so the verdict can project
# the identity sidecar from it and tell a repairable sidecar from a torn publication). Unknown under --dry-run (no cluster read), so a dry run
# always prints the plan.
read -r -d '' REFUSAL_STATE_PY <<'PYRS' || true
import hashlib, json, os, sys
d = sys.argv[1]; meta = sys.argv[2]
out = {"dir": d, "exists": os.path.isdir(d), "complete_present": os.path.isfile(os.path.join(d, "COMPLETE")), "complete": None, "parses": None,
       "hashes": {}, "texts": {}, "present": sorted(os.listdir(d)) if os.path.isdir(d) else []}
if out["complete_present"]:
    try:
        out["complete"] = json.load(open(os.path.join(d, "COMPLETE"))); out["parses"] = True
    except ValueError:
        out["parses"] = False
    files = out["complete"].get("files") if isinstance(out["complete"], dict) else None
    for name in (files if isinstance(files, dict) else {}):
        p = os.path.join(d, str(name))
        out["hashes"][str(name)] = hashlib.sha256(open(p, "rb").read()).hexdigest() if os.path.isfile(p) else None
        if str(name) == meta and os.path.isfile(p):
            out["texts"][meta] = open(p, encoding="utf-8", errors="replace").read()
print(json.dumps(out))
PYRS
cluster_refusal_state(){   # <publication dir> -> one JSON line of the cluster's CURRENT manifest + hashes + the meta's text, or non-zero when unknown (--dry-run / unreachable)
  [ "$REPRO_DRY_RUN" = 1 ] && return 1
  ssh -o BatchMode=yes -o ConnectTimeout=30 "$CL_HOST" "python3 - '$1' '$REFUSAL_META_SIDECAR'" <<<"$REFUSAL_STATE_PY" 2>/dev/null
}
refusal_published_ready(){   # <object ids, one per line>: 0 when $REFUSAL_DIR on $CLUSTER IS the registered refusal direction's publication (the unit's header); else 1, REFUSAL_PUBLICATION_PROBLEM / _WHY set
  local ids="$1" state verdict rc
  REFUSAL_PUBLICATION_PROBLEM=unknown
  REFUSAL_PUBLICATION_WHY="cluster state unknown$( [ "$REPRO_DRY_RUN" = 1 ] && echo " under --dry-run" || echo " ($CLUSTER unreachable, or no python3 on its login host)")"
  state="$(cluster_refusal_state "$REFUSAL_DIR")" || { repro_note "refusal_published_ready: $REFUSAL_PUBLICATION_WHY"; return 1; }
  # sidecar_repairable: the LISTED identity sidecar is the ONE file not at its hash (absent, truncated or
  # altered) and the projection of the verified fit_meta.json — the fitter's own fit_provenance_text, imported from $REFUSAL_FITTER — hashes to
  # the listed value, so the bytes COMPLETE promised are recoverable without a refit (the fitter's kept path, repair_fit_provenance, writes
  # them); any other damage, or a projection that is not what the marker lists, is torn as before.
  verdict="$("$(repro_python)" -c '
import hashlib, importlib.util, json, sys
st = json.loads(sys.argv[1]); ids = set(sys.argv[2].split()); sidecar = sys.argv[3]; meta = sys.argv[4]; fitter = sys.argv[5]; d = st.get("dir")
def out(kind, why):
    print(kind + "\t" + why); sys.exit(0 if kind == "ready" else 1)
if not st.get("complete_present"):
    out("absent", "no COMPLETE at %s (%s)" % (d, ("files present: " + ", ".join(st.get("present") or [])) if st.get("exists") else "the directory does not exist"))
doc = st.get("complete")
if not st.get("parses") or not isinstance(doc, dict):
    out("torn", "COMPLETE at %s is not a JSON manifest" % d)
files = doc.get("files")
if not isinstance(files, dict) or not files:
    out("torn", "COMPLETE at %s names no files" % d)
bad = [n for n, want in sorted(files.items()) if (st.get("hashes") or {}).get(n) != want]
if bad and bad != [sidecar]:
    out("torn", "COMPLETE at %s lists %s absent or not at the published sha256 (a torn or altered publication)" % (d, ", ".join(bad)))
oid = str(doc.get("object_sha256") or "")
if oid not in ids:
    out("other_object", "COMPLETE at %s publishes object %s, not one of the registry reference/approved row(s) at this rung (%s)" % (d, oid[:16] or "?", ", ".join(sorted(i[:16] for i in ids))))
if bad:
    # the listed identity sidecar alone is damaged: repairable iff the verified meta (listed, at its hash — else it would be in `bad`) projects to the listed value
    text = (st.get("texts") or {}).get(meta)
    if meta not in files or not isinstance(text, str):
        out("torn", "COMPLETE at %s lists %s absent or not at the published sha256, and %s is not a listed, readable file to project it from (a torn or altered publication)" % (d, sidecar, meta))
    try:
        spec = importlib.util.spec_from_file_location("fit_refusal_direction_verdict", fitter); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
        projected = hashlib.sha256(mod.fit_provenance_text(json.loads(text)).encode("utf-8")).hexdigest()
    except Exception as e:
        out("torn", "COMPLETE at %s lists %s absent or not at the published sha256, and the projection of %s could not be formed from %s (%s: %s)" % (d, sidecar, meta, fitter, type(e).__name__, e))
    if projected != files.get(sidecar):
        out("torn", "COMPLETE at %s lists %s absent or not at the published sha256, and the projection of the verified %s hashes to %s, not the listed %s (a torn or altered publication - the promised bytes are not recoverable)" % (d, sidecar, meta, projected[:16], str(files.get(sidecar))[:16]))
    out("sidecar_repairable", "COMPLETE at %s lists the identity sidecar %s but the file is absent, truncated or altered - the one manifest problem, every other listed file at its sha256, and the projection of the verified %s hashes to the listed value, so the fitter writes the promised bytes again (repair_fit_provenance; no refit)" % (d, sidecar, meta))
if sidecar not in files:
    out("no_identity_sidecar", "COMPLETE at %s does not list the identity sidecar %s (a publication from before the sidecar, or a backfill torn before the marker was rewritten) - the stager reads the cell not_ready and the consumer gate has no record to verify" % (d, sidecar))
out("ready", "COMPLETE publishes %s, every listed file at its sha256, %s listed" % (oid[:16], sidecar))' "$state" "$ids" "$REFUSAL_IDENTITY_SIDECAR" "$REFUSAL_META_SIDECAR" "$REFUSAL_FITTER")"; rc=$?
  IFS=$'\t' read -r REFUSAL_PUBLICATION_PROBLEM REFUSAL_PUBLICATION_WHY <<<"$verdict"
  if [ -z "$REFUSAL_PUBLICATION_PROBLEM" ]; then
    # the verdict script itself failed (its traceback is above): no verdict is `unknown`, never silently torn or ready — a LIVE run fails closed on it
    REFUSAL_PUBLICATION_PROBLEM=unknown; REFUSAL_PUBLICATION_WHY="the publication verdict could not be formed from the cluster's state (the verdict script failed; its stderr is above)"; rc=1
  fi
  [ $rc -eq 0 ] || repro_note "refusal_published_ready: $REFUSAL_PUBLICATION_WHY"
  return $rc
}
# the registered refusal direction of the SERVED checkpoint at THIS rung, read by the PINS WRITER'S OWN reader:
# make_full22_configs.py --arditi-registered-ids selects the identity's rows broadly in SQL — campaign, fit_model_id = the served
# id, vector_all / arditi_allpos / baseline_published, no rung predicate — and classifies each row's rung in Python with arditi_row_rung,
# the function arditi_pins_from_registry partitions the same rows with (arditi_rows_at_rung), so 04 and the pins writer CANNOT disagree on
# a spelling. The SQL twin of that reading 04 carried before matched only the two canonical staged-id spellings while the Python parses any
# decimal in the grammar: a row spelled `ar_cm1p0__arditi_allpos` was coefficient -1 to the writer and absent to 04 at c=-1, so 04 read
# the registered identity+rung as unregistered, submitted a second fit, and the writer then refused the two reference rows.
# registry_object_ids' contract, kept: a MISSING or unreadable registry refuses a LIVE run and proceeds as unregistered under --dry-run
# (the same lines); a reader that exits non-zero — the registry unreadable, or TWO distinct reference/approved objects at the
# identity and rung (the state the pins writer refuses by name) — REFUSES the decision by name, never reads as zero rows; its
# stderr is surfaced. The refit scope registry_object_ids threads (the bank-location and init-scheme rules) is not applied:
# the refusal direction is an object of the SERVED checkpoint published outside the fit bank, carries no init scheme, and is not a refit
# unit (REFIT_UNITS — --refit refuses it above), so no refit row of it can exist. One id per line on stdout, as registry_object_ids
# prints; nothing when none. The ids go to registry_done under scope `ids`, so the done-ness decision stays the one every unit's is.
refusal_registry_ids(){
  local root out err rc; root="$(eval echo "$SISTEER_REGISTRY_ROOT")"
  if [ ! -r "$root/registry.sqlite" ]; then
    echo "registry_unavailable: $root/registry.sqlite missing or unreadable (SISTEER_REGISTRY_ROOT=$SISTEER_REGISTRY_ROOT)" >&2
    if [ "${REPRO_DRY_RUN:-0}" = 1 ]; then repro_note "registry_unavailable: a dry run proceeds as if nothing were registered; a LIVE run refuses here (D1885)" >&2; return 0; fi
    repro_fatal "refusal_registry_ids: the registry is unavailable — whether the refusal direction of $SERVE_ID at c=$COEF is registered cannot be decided (a missing or unreadable database is never zero rows); mount or restore $root/registry.sqlite, then re-run (D1885)"
  fi
  out="$(mktemp)"; err="$(mktemp)"
  "$(repro_python)" "$REFUSAL_GENERATOR" --arditi-registered-ids --registry-root "$root" --served-model-id "$SERVE_ID" --coefficient "$COEF" --campaign "$CAMPAIGN" >"$out" 2>"$err"; rc=$?
  [ -s "$err" ] && sed 's/^/registry stderr: /' "$err" >&2
  if [ "$rc" -ne 0 ]; then
    echo "registry_unavailable: $(tr '\n' ' ' < "$err" | cut -c1-600)" >&2
    rm -f "$err" "$out"
    repro_fatal "refusal_registry_ids: make_full22_configs.py --arditi-registered-ids exited $rc for root $root — whether the refusal direction of $SERVE_ID at c=$COEF under campaign $CAMPAIGN is registered cannot be decided (a failed read is never zero rows; two reference/approved objects at the identity and rung are named above and refused as the pins writer refuses them), and an undecided unit is never submitted; fix the registry read or withdraw the rows that are not it, then re-run (D1862)"
  fi
  cat "$out"
  rm -f "$err" "$out"
}
refusal_direction_done_marker_valid(){   # a pre-existing DONE.refusal_direction is honoured only while the registry row at THIS rung AND its publication on the cluster still hold (the marker is a cache, as difference_relu's is)
  local ids
  ids="$(refusal_registry_ids)" || repro_fatal "refusal_direction: refusal_registry_ids refused (above) — the DONE marker's validity cannot be decided (D1862)"   # the unit's own reader: the pins writer's rung reading
  REFUSAL_PUBLICATION_PROBLEM=no_row; REFUSAL_PUBLICATION_WHY="no reference/approved registry row of $SERVE_ID at c=$COEF under campaign $CAMPAIGN"
  [ -n "$ids" ] && refusal_published_ready "$ids"
}

# --- FD 7 / FD 18: the run override's init scheme rides the generated fit config ---------------------------------------
# `make_sdfbase_configs.py --init-scheme-override SCHEME --models <fit slug>` stamps the scheme into configs/sdfbase_folds.json[hpo]
# for BOTH CE arms of the fit (merged into any recorded HPO winner, never replacing it) and regenerates
# configs/sdfbase_<fit slug>_fits.yaml, whose train_kwargs.<arm>.fit_selection.init_scheme the trainer serialises to
# fit_selection.json. Idempotent: a tree already stamped rewrites identical bytes. The paper's `zero`
# (paper_defaults.env) is the generator's default and is never stamped (the sixteen paper lanes declare no fit_selection and would
# refuse). A stamp that CHANGES the tracked config must be committed + mirrored before any fit is submitted: the sbatch runs the
# cluster checkout pinned at $SHA, so a dirty local config would fit at `zero` while the overrides file says otherwise.
stamp_init_scheme(){
  local scheme="${OVERRIDE_INIT_SCHEME:-}"
  [ -n "$scheme" ] && [ "$scheme" != zero ] || return 0
  local gen="$REPRO_REPO/scripts/sdfbase/make_sdfbase_configs.py" cfg="configs/sdfbase_${FIT_ID}_fits.yaml"
  repro_run "$(repro_python)" "$gen" --init-scheme-override "$scheme" --models "$FIT_ID" \
    || repro_fatal "make_sdfbase_configs.py --init-scheme-override $scheme --models $FIT_ID failed (OVERRIDE_INIT_SCHEME from $SISTEER_RUN_OVERRIDES)"
  [ "$REPRO_DRY_RUN" = 1 ] && return 0
  local dirty; dirty="$(git -C "$REPRO_REPO" status --porcelain --untracked-files=no -- configs/sdfbase_folds.json "$cfg" 2>/dev/null)"
  [ -z "$dirty" ] || repro_fatal "OVERRIDE_INIT_SCHEME=$scheme changed $(echo "$dirty" | awk '{print $2}' | tr '\n' ' ')— commit + mirror (lib/mirror_repo.sh) before submitting the fits: the cluster runs the checkout pinned at ${SHA:0:12}, which still carries the previous scheme"
  repro_say "init_scheme $scheme is stamped on $cfg (committed at ${SHA:0:12})"
}

# --- run ---------------------------------------------------------------------------------------------------------------------
case " $UNITS " in *" control_lora "*|*" constitution_lora "*|*" control_relu "*|*" constitution_relu "*) stamp_init_scheme;; esac
repro_say "04_fit_objects: model=$REPRO_MODEL fit=$FIT_ID ($FIT_REPO@${FIT_REV:0:12}) serve=$SERVE_ID ($SERVE_REPO@${SERVE_REV:0:12}) cluster=$CLUSTER sha=${SHA:0:12} fit_width=$FIT_WIDTH objects=[$UNITS]${REFIT:+ refit=$REFIT bank=$BANK scheme=$REFIT_SCHEME (D1795 / D1816)}"
for u in $UNITS; do
  if repro_done "$REPRO_MODEL" "$STAGE" "$u${REFIT:+.$REFIT}"; then
    # a DONE marker is a cache of done-ness, not done-ness itself: difference_relu's is honoured only while the registry row AND V's
    # bound liveness still hold on the cluster (a marker written before the probe existed must not bypass the gate); unknown under
    # --dry-run, which therefore prints the plan
    if [ "$u" = difference_relu ] && ! difference_relu_done_marker_valid; then
      repro_note "$u: DONE marker present ($(repro_state_dir "$REPRO_MODEL")/$STAGE/DONE.$u${REFIT:+.$REFIT}) but V's liveness is not bound to its bytes on $CLUSTER (or unknown under --dry-run) — the marker is NOT honoured; the unit runs"
    elif [ "$u" = refusal_direction ] && ! refusal_direction_done_marker_valid; then
      # gap-refusal-pin: the refusal direction's marker is a cache too — revalidated through the unit's own
      # done-ness (the row at this rung AND its publication with the identity sidecar listed in COMPLETE); a marker from before the sidecar
      # existed must not skip the repair the unit schedules
      repro_note "$u: DONE marker present ($(repro_state_dir "$REPRO_MODEL")/$STAGE/DONE.$u) but the unit is not done on $CLUSTER (${REFUSAL_PUBLICATION_PROBLEM:-no_row}: ${REFUSAL_PUBLICATION_WHY:-}) — the marker is NOT honoured; the unit runs"
    elif [ -n "${REFIT:-}" ] && { [ "$u" = control_relu ] || [ "$u" = constitution_relu ]; } && ! refit_fitted_marker_valid "$u"; then
      # a refit fitted marker is a cache too — revalidated through the same candidate-scoped registry query fit_unit would issue
      # (a refusal there is named and nothing is skipped); a stale one is not honoured and the unit runs
      repro_note "$u: DONE marker present ($(repro_state_dir "$REPRO_MODEL")/$STAGE/DONE.$u.$REFIT) but the registry no longer holds the unit's row in this run's scope (bank $BANK) — the marker is NOT honoured; the unit runs (D2126)"
    else
      repro_say "$u: DONE marker present ($(repro_state_dir "$REPRO_MODEL")/$STAGE/DONE.$u${REFIT:+.$REFIT}) — skipping"
      continue
    fi
  fi
  case "$u" in
    control_lora|constitution_lora|control_relu|constitution_relu) fit_unit "$u";;
    difference_lora) difference_lora;;
    difference_relu) difference_relu;;
    refusal_direction) refusal_direction;;
  esac
done
repro_say "04_fit_objects done (objects=[$UNITS])"
