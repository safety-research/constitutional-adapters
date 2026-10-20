#!/bin/bash
# math500-thinking-on-full22 — the shell body every scripts/sdfbase/math500/*.sbatch sources AFTER
# scripts/sdfbase/full22/common.sh (bash). The battery's job shape (f22_* — init, venvs, standup, stop, result) is
# reused unchanged; this file adds the experiment's own seams: the matrix row (cells.py), the overlay
# config, the out root, the standup extras, the usage ledger, the driver argv, the dry-run printing — and, since the
# eight-arm extension, the per-row geometry, the launch-time re-check of the budget's fit sidecar, the sampler
# read-back opt-in + its recorder, and the launcher's row-identity stamp.
# NEVER --export= (hard rule 1): every value travels positionally, through cells.py's row, or these functions.

M5T_DIR="scripts/sdfbase/math500"
M5T_WAVE_ROOT_REL="exp/math500_thinking_on_full22"
# the standup MODE every math500 job passes to f22_standup (its first positional; the test pins the sbatch's literal to
# this): `attacks` -> mt_serving_up composes tool_calls=False; `petri` would compose the tool parser. Printed by the dry
# run's STANDUP_POLICY line so the compatibility preflight composes under the SAME launch configuration (fix A4).
M5T_STANDUP_MODE=attacks
# the OVERFLOW cluster's registry label (f22_init's CLUSTER, from the job's SISTEER_CLUSTER_LABEL) — the ONE
# place this file names it, so m5t_bank_view can pick the row's per-cluster external bank view
M5T_OVERFLOW_CLUSTER=cluster_b
# the Nemotron fused-kernel attestation is a PREREQUISITE for every math500 job (common.sh f22_model_facts; LB2)
export F22_REQUIRE_NEMO_BACKEND=1

m5t_py(){   # the python that reads cells.py (stdlib only): the client venv once built, else the node's python3
  if [ -x "${CPY:-}" ]; then echo "$CPY"; else command -v python3; fi
}

m5t_select_serving_path(){   # ARRAY | <key> <cell> <sampling> [<view>] -> the ROW's serving route, DECLARED to f22_init (common.sh
                             # f22_select_serving_path, the kind-aware default of 2026-09-15): F22_LAUNCH_CELLS — a Bash ARRAY of
                             # the HARNESS cells this job's standup serves (`none` + the row's cell; cells.py launch_cells_for: exactly what
                             # `--cell "$CELL"` makes the standup's cells_requested read) — on EVERY row, and F22_SERVING_PATH from the ROW
                             # when it pins one. THE PIN FOLLOWS THE STORE (since 2026-09-15; cells.py serving_path_pin, the rule in
                             # full22/steer_store_pin.py): an AlphaSteer row pins `lens` by kind (no native form); an Arditi row pins `lens`
                             # iff its steer store ON RECORD — its out root under F22_WAVE_ROOT on THIS cluster — is a lens store (store:lens),
                             # is unpinned where no store exists (unpinned:no-store — the kind-aware default then selects native, Arditi's
                             # residual slot since native-psr-serving) or where the store is native-stamped (store:native); a mixed-engine
                             # store refuses by name. So F22_WAVE_ROOT must be exported BEFORE this runs (m5t_init does; the dry run sets it
                             # from its stub root) and the store is read here, at the launcher's first row-resolution step — before f22_init
                             # fixes the path, before the out root is created, before any marker. An explicit pin WINS at f22_init (recorded
                             # EXPLICIT); a row that pins none has its path selected BY KIND over the declared cells (a LoRA / Arditi cell ->
                             # lora_native, none / prompted -> the entry-point default) and logged `[serving-path] AUTO-selected`. A submitter
                             # value CONFLICTING with a pin refuses by name; so does a submitter value other than lora_native on a store:native
                             # row (a lens leg into a native store would be the mixed store the rule refuses). Array 506737 (2026-09-14) died at
                             # the native standup for want of the pin; before this declaration the launcher DEFERRED (it owns F22_WAVE_ROOT)
                             # and the standup bound the path from the overlay manifest. The standup's --serving-path follows F22_SERVING_PATH
                             # (f22_standup).
  local py out cells c root pin reason pins_args=() pins_list=() pins_desc; py=$(m5t_py)
  root="${F22_WAVE_ROOT:-}"
  [ -n "$root" ] || f22_fatal "m5t_select_serving_path: F22_WAVE_ROOT is unset — the row's serving-path pin FOLLOWS ITS STORE under this experiment's wave root on this cluster, so m5t_init exports F22_WAVE_ROOT before declaring the route (the dry run sets it from its stub root)"
  # the PINS source — the repo's store-pins record under this checkout (the CLI's default; logged `pins checkout`) — is the launch tree's.
  # M5T_STORE_PINS (a colon-separated list of records) is the TESTS' seam for a synthetic record: honoured ONLY in an explicit local dry run
  # (M5T_DRY_RUN=1) outside any Slurm job, logged whenever honoured; a job that inherits it REFUSES by name — the override REPLACES the
  # admitted record, and a known lens store would relaunch natively behind it
  pins_desc="${M5T_STORE_PINS:-checkout}"
  if [ -n "${M5T_STORE_PINS:-}" ]; then
    if [ -n "${SLURM_JOB_ID:-}" ] || [ "${M5T_DRY_RUN:-0}" != 1 ]; then
      f22_fatal "M5T_STORE_PINS is set (${M5T_STORE_PINS}) but this is not a local dry run (SLURM_JOB_ID=${SLURM_JOB_ID:-unset}, M5T_DRY_RUN=${M5T_DRY_RUN:-0}) — the store-pins override is a test seam that REPLACES the admitted record; refused by name so a known store is never relaunched on another engine; unset it"
    fi
    IFS=: read -r -a pins_list <<< "$M5T_STORE_PINS"
    for c in "${pins_list[@]}"; do [ -z "$c" ] || pins_args+=(--store-pins "$c"); done
  fi
  if [ "${1:?row}" = ARRAY ]; then
    [ -n "${SLURM_ARRAY_TASK_ID:-}" ] || f22_fatal "ARRAY given but SLURM_ARRAY_TASK_ID is unset — submit with --array=<indices> (cells.py plan)"
    out=$("$py" "$REPO_DIR/$M5T_DIR/cells.py" row --index "$SLURM_ARRAY_TASK_ID" --serving-path --wave-root "$root" ${pins_args[@]+"${pins_args[@]}"}) || f22_fatal "cells.py: no matrix row at array index $SLURM_ARRAY_TASK_ID, or its store on record refused the pin (serving path)"
    cells=$("$py" "$REPO_DIR/$M5T_DIR/cells.py" row --index "$SLURM_ARRAY_TASK_ID" --launch-cells) || f22_fatal "cells.py: no matrix row at array index $SLURM_ARRAY_TASK_ID (launch cells)"
  else
    out=$("$py" "$REPO_DIR/$M5T_DIR/cells.py" row --key "$1" --cell "${2:?cell}" --sampling "${3:?sampling mode}" --view "${4:-main}" --serving-path --wave-root "$root" ${pins_args[@]+"${pins_args[@]}"}) || f22_fatal "cells.py: ($1, $2, $3, view ${4:-main}) is not a matrix row, or its store on record refused the pin (serving path)"
    cells=$("$py" "$REPO_DIR/$M5T_DIR/cells.py" row --key "$1" --cell "${2:?cell}" --sampling "${3:?sampling mode}" --view "${4:-main}" --launch-cells) || f22_fatal "cells.py: ($1, $2, $3, view ${4:-main}) is not a matrix row (launch cells)"
  fi
  # two lines: the pin (empty = unpinned) and its reason (cells.py / steer_store_pin.pin_lines)
  { IFS= read -r pin; IFS= read -r reason; } <<< "$out"
  [ -n "$reason" ] || f22_fatal "cells.py row --serving-path printed no reason for the row (got: $out) — the pin is always explained"
  # THE DECLARED CELL SOURCE: f22_select_serving_path's precedence is F22_SERVING_PATH > F22_LAUNCH_CELLS > F22_LAUNCH_MANIFEST > deferred
  # (F22_WAVE_ROOT) > refused by name — so with this array declared the launcher never defers, and a row-less init (the diagnostic
  # launchers) still defers to the standup, which binds from the manifest it serves
  F22_LAUNCH_CELLS=()
  while IFS= read -r c; do [ -z "$c" ] || F22_LAUNCH_CELLS+=("$c"); done <<< "$cells"
  [ "${#F22_LAUNCH_CELLS[@]}" -ge 1 ] || f22_fatal "cells.py row --launch-cells printed no cell for the row — every standup serves at least none"
  M5T_SERVING_PATH_PINNED="$pin"; M5T_SERVING_PATH_REASON="$reason"
  if [ -n "$pin" ]; then
    if [ -n "${F22_SERVING_PATH:-}" ] && [ "$F22_SERVING_PATH" != "$pin" ]; then
      f22_fatal "the row pins serving path '$pin' ($reason: the pin follows the row's object kind and its store on record) but the submitter's shell exports F22_SERVING_PATH=$F22_SERVING_PATH — refused by name; unset it (the row selects the route)"
    fi
    export F22_SERVING_PATH="$pin"
    f22_log "serving path: $pin — pinned by the ROW ($reason; wave root $root; pins $pins_desc); the standup's --serving-path follows it; launch cells declared: ${F22_LAUNCH_CELLS[*]} (F22_LAUNCH_CELLS)"
  else
    case "$reason" in store:native*)
      if [ -n "${F22_SERVING_PATH:-}" ] && [ "$F22_SERVING_PATH" != lora_native ]; then
        f22_fatal "the row's steer store on record is NATIVE-stamped ($reason; wave root $root) but the submitter's shell exports F22_SERVING_PATH=$F22_SERVING_PATH — refused by name: a lens leg written into a native store is the mixed-engine store the pin refuses; unset it (the pin follows the store)"
      fi;;
    esac
    f22_log "serving path: not pinned by the row ($reason; wave root $root; pins $pins_desc) — the kind-aware default selects it at f22_init BY KIND over the declared launch cells: ${F22_LAUNCH_CELLS[*]} (F22_LAUNCH_CELLS)${F22_SERVING_PATH:+; the submitter shell exports F22_SERVING_PATH=$F22_SERVING_PATH, which wins as EXPLICIT}"
  fi
}

m5t_init(){   # $1 = wave sha, $2.. = the row (ARRAY | <key> <cell> <sampling> [<view>]) -> f22_init under THIS experiment's wave root
              # (never the full22 wave's), the ROW's serving route DECLARED first (m5t_select_serving_path: F22_LAUNCH_CELLS = the cells
              # this job's standup serves, F22_SERVING_PATH when the row pins one) so f22_init fixes the path from the ROW — EXPLICIT on a
              # pinned row, kind-aware AUTO over the declared cells otherwise (the contract: every caller declares its cell source).
              # The pin FOLLOWS THE STORE at the row's out root (2026-09-15), so the wave root is exported FIRST and the declaration
              # reads the store under it. Called WITHOUT a row (the diagnostic launchers: the smoke, the Nemotron probe, the prod dry
              # run) nothing is declared and the path is DEFERRED to the standup, which binds it from the manifest it serves — a job
              # that owns its wave root (F22_WAVE_ROOT, exported here) may defer; f22_init logs which.
  local sha="${1:?wave sha}" shared
  # the cluster's shared root is the job's site value (the front door passes the cluster block's root to every job), never
  # guessed from which storage paths exist; f22_init reads the same variable
  [ -n "${SISTEER_SHARED_ROOT:-}" ] \
    || { echo "FATAL: set SISTEER_SHARED_ROOT, the shared root of the cluster (the front door exports it from the site config)" >&2; exit 1; }
  shared="$SISTEER_SHARED_ROOT"
  shift
  export F22_WAVE_ROOT="${F22_WAVE_ROOT:-$shared/$M5T_WAVE_ROOT_REL}"
  [ $# -ge 1 ] && m5t_select_serving_path "$@"
  f22_init "$sha"
  # The `datasets` library, OFFLINE (hf_env.sh sets HF_HUB_OFFLINE=1), reads a PREPARED arrow cache
  # under HF_DATASETS_CACHE — the hub snapshot alone does not serve it. The primary cluster's HF_HOME is the
  # shared cohort cache, whose datasets dir already holds MATH-lighteval under several per-subject
  # configurations, so an unqualified load (inspect_evals' MATH task) is AMBIGUOUS there and refuses.
  # A per-user datasets cache under the shared root disambiguates; it is prepared ONCE per cluster
  # by an online load and the jobs then read it offline. Same path on both clusters.
  export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$shared/hf_datasets}"
  [ -d "$HF_DATASETS_CACHE/HuggingFaceH4___math-500" ] \
    || f22_fatal "HF_DATASETS_CACHE=$HF_DATASETS_CACHE carries no prepared MATH-500 (prepare it once per cluster: an ONLINE load_dataset of HuggingFaceH4/MATH-500 and DigitalLearningGmbH/MATH-lighteval with this HF_DATASETS_CACHE, then the jobs read it offline)"
  # the launcher located its helpers before sourcing them; an M5T_REPO_DIR override must BE the job's
  # checkout — helpers from one tree and a job from another is refused, never run
  [ -z "${M5T_REPO_DIR:-}" ] || [ "$REPO_DIR" = "$M5T_REPO_DIR" ] || f22_fatal "M5T_REPO_DIR=$M5T_REPO_DIR but the wave checkout is $REPO_DIR — the helpers and the job must come from ONE tree"
  # the Inspect evals cache lives under the wave root (the datasets are pre-fetched into the HF cache; offline)
  export INSPECT_EVALS_CACHE_DIR="${INSPECT_EVALS_CACHE_DIR:-$WAVE_ROOT/inspect_evals_cache}"; mkdir -p "$INSPECT_EVALS_CACHE_DIR"
}

m5t_row(){   # ARRAY | <key> <cell> <sampling_mode> [<view>]  -> evals the matrix row: KEY CELL CELLSLUG SAMPLING_MODE FITTED PIN M5T_*
  local py out; py=$(m5t_py)
  if [ "${1:?row}" = ARRAY ]; then
    [ -n "${SLURM_ARRAY_TASK_ID:-}" ] || f22_fatal "ARRAY given but SLURM_ARRAY_TASK_ID is unset — submit with --array=<indices> (cells.py plan)"
    out=$("$py" "$REPO_DIR/$M5T_DIR/cells.py" row --index "$SLURM_ARRAY_TASK_ID" --sh) || f22_fatal "cells.py: no matrix row at array index $SLURM_ARRAY_TASK_ID"
  else
    # the view positional (E3): `main` unless given — an it-doc row shares its (key, cell, mode) with the main view's row
    out=$("$py" "$REPO_DIR/$M5T_DIR/cells.py" row --key "$1" --cell "${2:?cell}" --sampling "${3:?sampling mode}" --view "${4:-main}" --sh) || f22_fatal "cells.py: ($1, $2, $3, view ${4:-main}) is not a matrix row (cells.py list)"
  fi
  eval "$out"
  # THE SAMPLER READ-BACK OPT-IN (P6; A11): the row says 1 on every extension row and 0 on every published one. The
  # eval above runs AFTER the environment is read and --export= is forbidden, so an env opt-out under the row's own
  # name would be overwritten here; the opt-out is M5T_LOG_REQUESTS_OVERRIDE=0|1, honoured now and logged. The
  # RESULT line records the EFFECTIVE value either way (logreq=).
  M5T_LOG_REQUESTS_EFFECTIVE="${M5T_LOG_REQUESTS_OVERRIDE:-$M5T_LOG_REQUESTS}"
  case "$M5T_LOG_REQUESTS_EFFECTIVE" in 0|1) ;; *) f22_fatal "M5T_LOG_REQUESTS_OVERRIDE must be 0 or 1, got '$M5T_LOG_REQUESTS_EFFECTIVE'";; esac
  f22_log "matrix row $M5T_INDEX: $M5T_JOB_ID key=$KEY cell=$CELL sampling=$SAMPLING_MODE fitted=$FITTED route=$M5T_ROUTE_KIND family=$M5T_FAMILY time=$M5T_TIME out=$M5T_OUT_REL wave=$M5T_WAVE status=$M5T_STATUS thinking=$M5T_THINKING_MODE geometry=$M5T_GEOMETRY_EVAL/$M5T_GEOMETRY_L budget_class=$M5T_BUDGET_CLASS log_requests=$M5T_LOG_REQUESTS_EFFECTIVE${M5T_LOG_REQUESTS_OVERRIDE:+ (override; the row says $M5T_LOG_REQUESTS)}${M5T_INSPECT_SEED_POLICY:+ inspect_seed=$M5T_INSPECT_SEED_POLICY@$M5T_INSPECT_SEED_BASE}${M5T_STANDUP_COMPANION_CELLS:+ standup_companions=$M5T_STANDUP_COMPANION_CELLS}${M5T_BANK_VIEW_STAMP:+ bank_view_stamp=$M5T_BANK_VIEW_STAMP}"
}

m5t_assert_launchable(){   # THE BATCH BOUNDARY: the row m5t_row resolved must be in ITS wave's eligible set — cells.py's
                           # own predicate (assert_eligible over eligible_rows, the one `plan --assert-eligible` ran over the submission),
                           # re-asserted by the JOB because the submission is not trusted: a positional invocation or an --array index
                           # naming a published (complete), held or deferred row resolved fine, kept status=launch (published-ness is the
                           # WAVE, never a status) and would have run into the published output root — the budget re-check does not stop
                           # it (the seed wave refreshes the same model's `none` budget). Runs BEFORE m5t_out_root, the venvs and any
                           # artifact write, so a refused row leaves nothing behind (OUT_ROOT is unset: the trap's RESULT line goes to the
                           # log only). The dry run and the diagnostic launchers (smoke / probe: published `none` rows into diagnostic
                           # roots) do not call it.
  local py rec; py=$(m5t_py)
  rec=$("$py" "$REPO_DIR/$M5T_DIR/cells.py" row --index "$M5T_INDEX" --assert-launchable 2>&1) \
    || f22_fatal "matrix row $M5T_INDEX ($M5T_JOB_ID: wave=$M5T_WAVE status=$M5T_STATUS) is NOT launchable — $rec"
  f22_log "$rec"
}

m5t_resolve_config(){   # the OVERLAY config + manifest for KEY (make_math500_overlays.py; never the campaign originals).
                        # A STEERING row (M5T_OBJECT_SOURCE=external) serves ANOTHER LANE's bank, whose view holds
                        # exactly ONE object, so it reads its OWN overlay pair — `explicit_cells: [none, <cell>]`: the
                        # standup's select_cells admits only a cell the config enumerates, and the explicit-cell registry
                        # validates every listed cell against the bank the row binds, so the base wave's six-cell overlay
                        # would refuse on five cells that bank does not carry.
                        # THE TAG IS THE ROW'S PANEL VIEW's (M5T_KEY_VIEW, cells.key_view): a row on a non-main panel view is
                        # cut from THAT view's campaign config — E3's it-doc rows and the all-positions Arditi row, whose
                        # direction was fit against this target's `it` wave view. Never M5T_VIEW: a steering row's `view`
                        # names its object BANK (`ar_r0p05`, and `it` on the AlphaSteer rows), and tagging on it would cut a
                        # `<key>_it` overlay for a row whose config is the `main` one.
                        # A SEED row (M5T_OWN_OVERLAY=1, cells.py SEEDREP_ROWS) reads its OWN pair too —
                        # `sdfbase_full22_<key>_<view>.math500_<cellslug>.yaml`, explicit_cells [none, <cell>, <companion>] cut from
                        # the seed view's campaign config — so the standup's select_cells admits the row's cell AND its r=4 standup
                        # companion; the view's campaign body rides through byte for byte (the GSM8K launcher's G8K_OWN_OVERLAY rule).
  VIEW="$M5T_VIEW"; TAG="$KEY"
  [ "${M5T_KEY_VIEW:-main}" = main ] || TAG="${KEY}_${M5T_KEY_VIEW}"
  if [ -n "${M5T_OBJECT_SOURCE:-}" ] || [ "${M5T_OWN_OVERLAY:-}" = 1 ]; then
    CFG="configs/sdfbase/math500/sdfbase_full22_${TAG}.math500_${CELLSLUG}.yaml"
    MANIFEST="configs/capeval/manifests/petri_full22_${TAG}.math500_${CELLSLUG}.yaml"
  else
    CFG="configs/sdfbase/math500/sdfbase_full22_${KEY}.math500.yaml"; MANIFEST="configs/capeval/manifests/petri_full22_${KEY}.math500.yaml"
  fi
  [ -f "$CFG" ] && [ -f "$MANIFEST" ] || f22_fatal "missing overlay $CFG or $MANIFEST (scripts/sdfbase/math500/make_math500_overlays.py)"
  case "$KEY" in qwen35_4b|glm46vflash)   # the text-only pin, the campaign's job-side belt
    grep -q -- '--limit-mm-per-prompt' "$MANIFEST" || f22_fatal "$MANIFEST serves multimodal $KEY without --limit-mm-per-prompt (text-only pin)";;
  esac
  GPU_MEM_UTIL="${GPU_MEM_UTIL:-$M5T_GPU_MEM_UTIL}"
}

m5t_bank_view(){   # the row's bank, CHECK-ONLY, dispatched on where its object comes from: the campaign's own
                   # (roster-resolved) view for a base-wave or seed row, the EXTERNAL lane's view for a steering row
                   # (plan P8). Either way TT_TWO_SI_DEFENSES_DIR is exported and a fitted row's bytes are held to
                   # its pin before any GPU minute.
  local rel
  if [ -n "${M5T_OBJECT_SOURCE:-}" ]; then
    m5t_core
    case "${CLUSTER:-}" in
      "$M5T_OVERFLOW_CLUSTER") rel="$M5T_BANK_VIEW_REL_OVERFLOW";;
      *) rel="$M5T_BANK_VIEW_REL_PRIMARY";;
    esac
    [ -n "$rel" ] || f22_fatal "row $M5T_INDEX ($M5T_JOB_ID) names no bank view for cluster ${CLUSTER:-unknown} — its object is staged on the other cluster only (cells.py M5T_BANK_VIEW_REL_*); run it where it is staged"
    case "$M5T_OBJECT_SOURCE" in
      itdoc)   # E3: the re-fit row is BOUND to its bank entry through the it-doc record (sha + lane + staged id + sidecar of the kind), then the byte check
        st_itdoc_bank_check "$rel" "$M5T_BANK_CELL_DIR" "$PIN" "${M5T_ITDOC_ADMISSION:?M5T_ITDOC_ADMISSION}" "${M5T_LANE:?M5T_LANE}" "$KEY" "$M5T_VIEW" "$CELL"
        f22_log "it-doc bank: lane $M5T_LANE, staged id $M5T_STAGED_ID ($M5T_OBJECT_KIND); qualification: $M5T_QUALIFICATION";;
      *) st_external_bank_view_check "$rel" "$M5T_BANK_CELL_DIR" "$PIN"
         f22_log "selection: $M5T_SELECTION_LABEL";;
    esac
  elif [ "$FITTED" = 1 ]; then
    f22_bank_view_check "$CPY" "$CELL" "$PIN"
  else
    f22_bank_view_check "$CPY"
  fi
  m5t_companion_bank_check
}

m5t_companion_bank_check(){   # every STANDUP COMPANION's object.pt under the row's (roster-checked) bank view hashes to its pin —
                              # held right after the row's own bank check, before any GPU minute, exactly as the row's own cell is (the
                              # standup's --expect-object-sha256 <companion>=<pin> is the second copy of the same fact). A no-op on a row
                              # without companions (every matrix row, the companion's own row, the Arditi seed rows — the external branch
                              # above included), so every other row's path is byte-identical.
  local kv c pin have companion_pins=()
  read -r -a companion_pins <<< "${M5T_STANDUP_COMPANION_PINS:-}"
  for kv in ${companion_pins[@]+"${companion_pins[@]}"}; do
    c="${kv%%=*}"; pin="${kv#*=}"
    [ -f "$TT_TWO_SI_DEFENSES_DIR/$c/object.pt" ] || f22_fatal "bank view $TT_TWO_SI_DEFENSES_DIR lacks the standup companion $c/object.pt (row $M5T_JOB_ID; the view must stage the r=4 companion beside the row's cell — D71)"
    have=$(sha256sum "$TT_TWO_SI_DEFENSES_DIR/$c/object.pt" | cut -c1-64)
    [ "$have" = "$pin" ] || f22_fatal "standup companion $c/object.pt sha256 $have != its pin $pin (row $M5T_JOB_ID) — a drifted companion object; refusing"
    f22_log "standup companion OK: $c object.pt == pin ${pin:0:12} (registered unmeasured beside $CELL)"
  done
}

m5t_external_mirror_check(){   # E3 (the GSM8K launcher's g8k_external_mirror_check, E2-12, ported): an EXTERNAL or IT-DOC row's object through
                               # the DRIVER's OWN provenance mirror BEFORE the standup — served_capability_run._mirror_provenance ->
                               # sweep.resolve_training_mirror refuses an object whose fit record establishes no fit kind, whose recorded
                               # object_sha256 is not these bytes, a denylisted sha, or a derived sidecar whose parents disagree; the
                               # MATH-500 array lost 7 of 8 legs to these ~8 GPU-minutes AFTER their standups. Same function, same message, no
                               # second copy of the rule. No-op on a campaign row (the roster-resolved bank is the driver's own).
  [ -n "${M5T_OBJECT_SOURCE:-}" ] || return 0
  local obj out
  obj="${TT_TWO_SI_DEFENSES_DIR:?TT_TWO_SI_DEFENSES_DIR (m5t_bank_view runs first)}/${M5T_BANK_CELL_DIR:?M5T_BANK_CELL_DIR}/object.pt"
  out=$( cd "$REPO_DIR" && CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1 PYTHONPATH="$REPO_DIR/src:$REPO_DIR/scripts" "$CPY" - "$CELL" "$obj" <<'PY'
import sys
from sisteer.eval import sweep
cell, obj = sys.argv[1], sys.argv[2]
try:
    prov = sweep.resolve_training_mirror(cell, obj)
except Exception as e:  # the mirror's refusal (UnrecordedMirrorError, the stage-2 denylist), verbatim
    print(f"{type(e).__name__}: {e}")
    sys.exit(3)
print(f"provenance mirror OK: status={prov.status} fit_kind={prov.fit_kind} source={prov.source} model={prov.model_id}@{str(prov.revision)[:12]} "
      f"positions_at_fit={prov.positions_at_fit} object_sha={prov.object_sha[:12]}")
PY
  ) || f22_fatal "object $CELL ($obj): the driver's provenance mirror REFUSES it — the leg would die at the driver door after its standup; fix the record or the allow-list, never the check (E2-12 / E3) — $out"
  f22_log "$out"
}

m5t_out_root(){   # ONE durable out root per model x cell x sampling row — WAVE-ISOLATED since the
                  # extension (the row's out_root prefix: controls/ for the published wave,
                  # controls_ext1/ for the extension)
  OUT_ROOT="$WAVE_ROOT/$M5T_OUT_REL"; mkdir -p "$OUT_ROOT"
  BUDGET_SRC="$WAVE_ROOT/budgets/$KEY/$M5T_BUDGET_CLASS/$M5T_BUDGET_FILENAME"
}

m5t_budget_artifact(){   # copy the pre-launch CPU budget artifact for this cell class into the out root AND
                         # re-check its FIT sidecar at launch (the port of g8k_budget_artifact — math-eight-arms-extension §4.1,
                         # closing the gap F4 names): fits, at THIS row's cap, against THIS row's geometry entry and window, under
                         # THIS row's thinking arm, made at THIS tree (the wave sha), for THIS row's system-prompt class — a budget
                         # measured at another cap, entry, arm, tree or SI is a refusal, never a warning
  m5t_core
  st_budget_artifact "$BUDGET_SRC" "$OUT_ROOT/$M5T_BUDGET_FILENAME" \
    "run scripts/tinycap_oos_prompt_budget.py --task math500 --arm $M5T_THINKING_MODE --geometry-eval $M5T_GEOMETRY_EVAL --tree-sha $WAVE_SHA for ($KEY, $M5T_BUDGET_CLASS) before launch (plan §7.3 / eight-arms §4.3)"
  [ -f "$BUDGET_SRC.fit.json" ] || f22_fatal "no fit sidecar at $BUDGET_SRC.fit.json — the budget runs WITH --geometry-eval at the wave sha (eight-arms §4.3)"
  local si_cell=""; if [ "$M5T_ROUTE_KIND" = prompted ]; then si_cell="$CELL"; fi   # the row's system-prompt cell; empty on the bare / object routes
  "$(m5t_py)" - "$BUDGET_SRC.fit.json" "$M5T_CAP" "$M5T_GEOMETRY_EVAL" "$M5T_GEOMETRY_L" "$WAVE_SHA" "$M5T_THINKING_MODE" \
      "$si_cell" "${si_cell:+${M5T_SI_SHA256:0:16}}" <<'PY' || f22_fatal "fit sidecar $BUDGET_SRC.fit.json does not admit this row — refusing"
import json, sys
d = json.load(open(sys.argv[1]))
cap, ev, window, sha, arm, si_cell, si_sha16 = int(sys.argv[2]), sys.argv[3], int(sys.argv[4]), sys.argv[5], sys.argv[6], sys.argv[7] or None, sys.argv[8] or None
problems = []
if d.get("fits") is not True: problems.append(f"fits={d.get('fits')!r}: {d.get('why_not')}")
if int(d.get("cap") or 0) != cap: problems.append(f"cap {d.get('cap')!r} != {cap}")
if d.get("geometry_eval") != ev: problems.append(f"geometry_eval {d.get('geometry_eval')!r} != {ev!r} (the row's entry)")
if int(d.get("window") or 0) != window: problems.append(f"window {d.get('window')!r} != {window} (the row's L)")
if d.get("arm") != arm: problems.append(f"arm {d.get('arm')!r} != {arm!r}")
if d.get("tree_sha") != sha: problems.append(f"tree_sha {d.get('tree_sha')!r} != the wave sha {sha} (re-run the budget at this tree)")
if d.get("system_prompt_cell") != si_cell: problems.append(f"system_prompt_cell {d.get('system_prompt_cell')!r} != {si_cell!r} (the row's prompted cell)")
if si_sha16 is not None and d.get("system_prompt_sha16") != si_sha16: problems.append(f"system_prompt_sha16 {d.get('system_prompt_sha16')!r} != {si_sha16!r} (the row's SI digest)")
if problems: print("; ".join(problems)); sys.exit(1)
print(f"fit OK: prompt max {d.get('prompt_max')} + cap {cap} <= {window} ({ev}), margin {d.get('margin')}, arm {arm}, tree {sha[:12]}, si {si_cell or 'none'}")
PY
}

m5t_standup_extra(){   # -> F22_STANDUP_EXTRA: the standup flags — EVERY job names its own cell;
                       # ONLY the digest is conditioned on FITTED; plain spelling always (rule 3 refuses @thinking on these cells);
                       # --enable-log-requests (an opt-in, P6) iff the row opted in (M5T_LOG_REQUESTS_EFFECTIVE — every
                       # extension row; never a published one): vLLM then logs the sampler the ENGINE resolved per request
                       # (params only; the prompt is a DEBUG record and is not logged), which m5t_record_resolved_sampler reads
                       # back after the driver
  # THE SERVER-DEFAULT THINKING POLICY of this lane is the standup's default — OFF (`--thinking-on` never composed; the
  # composer pins `{"enable_thinking": false}` as the server default) — with the request-level ON flag on EVERY request
  # (the driver's --thinking on; vLLM merges the request's chat_template_kwargs over the server default). That is what
  # the published legs were served under (their tree had no --thinking-on mechanism at all), so a standup-wide
  # --thinking-on would move the served identity against them: f22_standup adds the flag iff F22_THINKING_ON is set,
  # and a Slurm batch process inherits the submitter's EXPORTED environment, so the variable is refused here BY NAME
  # rather than read (fix A4). The dry run prints the resulting policy (STANDUP_POLICY:) for the compatibility preflight.
  [ -z "${F22_THINKING_ON:-}" ] || f22_fatal "F22_THINKING_ON is set in this job's environment: the math500 lane serves the standup's default thinking policy (OFF) with the request-level ON flag on every request — a standup-wide --thinking-on would change the served identity against the published legs; unset it"
  M5T_STANDUP_THINKING_FLAG=0
  f22_standup_extra_reset --upstream-timeout-s "$M5T_UPSTREAM_TIMEOUT_S" --lens-salt-policy "$M5T_LENS_SALT_POLICY"   # on top of the inherited hook; idempotent per call (common.sh, 2026-09-16)
  if [ "${M5T_LOG_REQUESTS_EFFECTIVE:-0}" = 1 ]; then F22_STANDUP_EXTRA+=(--enable-log-requests); fi
  F22_STANDUP_EXTRA+=(--cell "$CELL")
  if [ "$FITTED" = 1 ]; then F22_STANDUP_EXTRA+=(--expect-object-sha256 "$CELL=$PIN"); fi
  # the GSM8K launcher's g8k_standup_extra, ported byte for byte: the row's STANDUP COMPANIONS — a
  # second `--cell` per companion, each pinned `<cell>=<sha>` (mt_serving_up's --cell is repeatable; select_cells admits a cell the row's
  # own overlay pair enumerates) — so every standup of the seed wave registers the view's r=4 LoRA beside the row's own cell and the
  # native composer fits ONE --max-lora-rank (8) across them (the MATH-500 reader compares that flag between a contrast's two legs, O106).
  # The driver's --cell stays the row's. Both names are EMPTY on every matrix row (cells.py row_sh), so nothing is appended there.
  local c kv companion_cells=() companion_pins=()
  read -r -a companion_cells <<< "${M5T_STANDUP_COMPANION_CELLS:-}"
  read -r -a companion_pins <<< "${M5T_STANDUP_COMPANION_PINS:-}"
  [ "${#companion_cells[@]}" -eq "${#companion_pins[@]}" ] || f22_fatal "row ${M5T_JOB_ID:-?}: ${#companion_cells[@]} standup companion cell(s) but ${#companion_pins[@]} pin(s) — every companion is pinned"
  for c in ${companion_cells[@]+"${companion_cells[@]}"}; do
    [ -n "$c" ] && [ "$c" != "$CELL" ] && [ "$c" != none ] || f22_fatal "row ${M5T_JOB_ID:-?}: standup companion '$c' is empty, the row's own cell or none — refused by name"
    F22_STANDUP_EXTRA+=(--cell "$c")
  done
  for kv in ${companion_pins[@]+"${companion_pins[@]}"}; do
    case "$kv" in *=????????????????????????????????????????????????????????????????) ;; *) f22_fatal "row ${M5T_JOB_ID:-?}: standup companion pin '$kv' is not <cell>=<64-hex sha256>";; esac
    F22_STANDUP_EXTRA+=(--expect-object-sha256 "$kv")
  done
  if [ -n "$M5T_REASONING_PARSER" ]; then F22_STANDUP_EXTRA+=(--reasoning-parser "$M5T_REASONING_PARSER"); fi
}

m5t_check_endpoint(){   # after f22_standup: the record must carry the timeout, the budget's template, and the FRESH table the stamp
                        # — the shared core's check; a FITTED math500 row expects `all` (every object here is positions=all)
  m5t_core
  st_check_endpoint "$SERVE_DIR/math500/endpoint.json" "$M5T_UPSTREAM_TIMEOUT_S" "$BUDGET_TEMPLATE_HASH" "$TABLE" "$M5T_HARNESS_CELL" \
    "$( [ "$FITTED" = 1 ] && echo all )"
}

# --- the shared execution core (scripts/sdfbase/full22/served_thinking.sh) -------------------------------------------------
# The ledger, the proxy token, the budget artifact, the endpoint checks, the driver argv's head / tail and the dry run's
# argv lines are TASK-INDEPENDENT and live in the core since the GSM8K sibling launcher; the m5t_* names below DELEGATE
# with byte-identical behaviour. m5t_core sources the core from the job's checkout ($REPO_DIR, set by the sbatch
# prologue) on FIRST USE: a caller that took only this block of functions (tests/test_sdfbase_math500_sbatch.py runs the
# ledger block by itself) still finds it, and a tree without the core refuses BY NAME after the trap is armed. Defined
# among its first callers so that block is self-contained (bash binds function names at call time).
m5t_ledger_up(){   # the USAGE LEDGER in front of the proxy (tinycap_served.py's shape) -> LEDGER_URL LEDGER_PID
  m5t_core
  st_ledger_up math500 "$M5T_LEDGER_WAIT_S" "$M5T_LEDGER_FILENAME" "$M5T_LEDGER_ENDPOINT_FILENAME" || { M5T_RESULT=LEDGER_FAILED; return 1; }
}

m5t_core(){   # source the shared core once, from THIS job's checkout
  declare -F st_ledger_up >/dev/null 2>&1 && return 0
  [ -r "${REPO_DIR:?REPO_DIR}/scripts/sdfbase/full22/served_thinking.sh" ] \
    || f22_fatal "no shared execution core at $REPO_DIR/scripts/sdfbase/full22/served_thinking.sh — the helpers and the job must come from ONE tree"
  source "$REPO_DIR/scripts/sdfbase/full22/served_thinking.sh"
}

m5t_ledger_down(){   # TERM, wait <= 30 s, then KILL (the launcher's teardown; a no-op before any ledger — the cleanup trap
                     # calls it at ANY step, so the core is loaded only once a ledger exists, which means it already was)
  [ -n "${LEDGER_PID:-}" ] || return 0
  m5t_core
  st_ledger_down
}

m5t_proxy_token(){   # the bearer the driver / the smoke probe present: endpoint.json's proxy_token (this loopback proxy is
                     # unauthenticated -> null), else the client's default `petri`
  m5t_core
  st_proxy_token "$SERVE_DIR/math500/endpoint.json"
}

m5t_driver_argv(){   # -> DRV: the COMPLETE driver invocation from the matrix row — ONE function, read by the
                     # job and by the dry run alike (needs CPY PROXY_URL TABLE LEDGER_URL CELL CFG TT_TWO_SI_DEFENSES_DIR OUT_ROOT
                     # SAMPLING_MODE TARGET_S SERVE_DIR FITTED PIN and the M5T_* row facts): the core's head, MATH-500's own
                     # task flags (the retired retry pass at 0, the one-pass cap statement), the core's tail, then the FITTED
                     # contract of the GSM8K sibling launcher: the driver-side pin
                     # --expect-object-sha256 "$PIN" WHENEVER FITTED=1 (a separate boundary from the standup's <cell>=<pin>),
                     # and --allow-out-of-support-thinking ONLY where a tuple authorises it (M5T_TUPLE non-empty — every E1
                     # fitted row; a no-switch fitted row, E2's llama, is IN SUPPORT: pin, no override, no tuple). The thinking
                     # mode is the ROW's (M5T_THINKING_MODE; `on` on every E1 row), never the run literal.
                     # NOT emitted on a matrix row, by decision (E2-3): --bank-view "$M5T_VIEW" — the
                     # driver's vocabulary WAS `main` | `it` (contracts.BANK_VIEWS) while the steering rows' views are their
                     # lanes' (`ar_r0p05`, ...), and the earlier waves' legs ran without it; on this instrument the row's view
                     # rides the launcher's row-identity stamp (m5t_row_identity), one carrier for every wave.
                     # REVERSED for the SEED rows only: a row carrying M5T_BANK_VIEW_STAMP (cells.py
                     # SEEDREP_ROWS: the lane view, `ar_c0p3` on the Arditi seed rows — contracts.expected_bank_view_stamp's rule) emits
                     # `--bank-view <stamp>`, so the four provenance fields ride every emitted row exactly as on the GSM8K seed rows;
                     # contracts.bank_views_admitted enumerates the roster views and the GSM8K documents' source banks. And
                     # the OPT-IN Inspect seed rule (the GSM8K launcher's rule): `--inspect-seed-policy replicate
                     # --inspect-seed-base <s>` iff the row carries the policy — both or neither, a policy with no base refused by name.
                     # Every other row's argv is byte-identical (the six M5T_* names are empty there; the sbatch test pins it).
  m5t_core
  st_driver_argv_head "$M5T_CONTROL" "$M5T_THINKING_MODE" "$SAMPLING_MODE" "$M5T_SAMPLES"
  DRV+=(--math500-retry-max-tokens "$M5T_RETRY" --max-tokens-by-task "$M5T_MAX_TOKENS_BY_TASK")
  # THE NO-SWITCH ARM's cap (math-eight-arms-extension E2, plan P7): a family with no thinking switch has no cap table, so
  # under `not_available` the bounded SETTER is REQUIRED and IS the cap (inspect_served.task_max_tokens; the GSM8K sibling
  # launcher's rule) — the statement's own value, admitted for `math` at the ceiling by contracts.ON_CAP_BY_TASK_ALLOWED.
  # A switch family's row never names it (its ON table resolves the same 16,384 and the setter only RAISES a cap).
  if [ "$M5T_THINKING_MODE" = not_available ]; then DRV+=(--on-cap-by-task "$M5T_MAX_TOKENS_BY_TASK"); fi
  if [ -n "${M5T_INSPECT_SEED_POLICY:-}" ]; then
    [ -n "${M5T_INSPECT_SEED_BASE:-}" ] || f22_fatal "row ${M5T_JOB_ID:-?} carries inspect_seed_policy $M5T_INSPECT_SEED_POLICY but no inspect_seed_base"
    DRV+=(--inspect-seed-policy "$M5T_INSPECT_SEED_POLICY" --inspect-seed-base "$M5T_INSPECT_SEED_BASE")
  elif [ -n "${M5T_INSPECT_SEED_BASE:-}" ]; then
    f22_fatal "row ${M5T_JOB_ID:-?} carries inspect_seed_base $M5T_INSPECT_SEED_BASE but no inspect_seed_policy — both or neither"
  fi
  st_driver_argv_tail "$M5T_IDENTITY_VERSION" "$OUT_ROOT/$M5T_BUDGET_FILENAME" "$TARGET_S" "$SERVE_DIR/math500/vllm_server.log"
  [ -z "${M5T_BANK_VIEW_STAMP:-}" ] || DRV+=(--bank-view "$M5T_BANK_VIEW_STAMP")
  if [ "$FITTED" = 1 ]; then
    [ -z "$M5T_TUPLE" ] || DRV+=(--allow-out-of-support-thinking)
    DRV+=(--expect-object-sha256 "$PIN")
    # THE LIVENESS GATE of a STEER row (M5T_OBJECT_SOURCE=external: an alphasteer / arditi object ANOTHER LANE staged):
    # the driver's positive control rules by the LOGPROB liveness statistic
    # (fidelity.check_logprob_steering_bites on the bites prompts, the scale-0 twin's repeat as the MEASURED noise
    # floor) instead of greedy-text divergence, which a small correct edit need not move (array 495677: gemma lambda
    # 0.057 and nemotron lambda 0.049 refused as "inert" at 0/5 while qwen lambda 0.548 passed at 2/5). The driver
    # scopes the rule by the object's KIND (contracts.LOGPROB_LIVENESS_KINDS), reports the text bites beside it and
    # records that logprob distance is not a dose axis. A row on the campaign's own bank passes nothing here: its
    # argv is byte-identical to before (the sbatch test pins both).
    [ -z "${M5T_OBJECT_SOURCE:-}" ] || DRV+=(--lens-liveness logprob)
  fi
}

m5t_recorder_argv(){   # -> RSA: scripts/sdfbase/record_resolved_sampler.py over THIS job's server log, into THIS row's out root
                       # (beside the store: the pull carries it), the row's CAP as the CLIENT cap, the row's PINNED canonical
                       # sampler as the reference — ONE --expect k=v per key of the NUMERIC projection M5T_SAMPLING_PARAMS
                       # (cells.sampling_params — SAMPLING_PARAM_KEYS only; the matrix dict's `source` prose is NEVER sent, the
                       # recorder refuses a key outside its SAMPLER_KEYS by name) — and the PROBE class BY KIND (the
                       # per-kind rule): `--probe-expect kinds --probe-tags
                       # <out_root>/request_evidence.jsonl`, the driver's own kind tag per probe attempt (X-Tinycap-Task:
                       # probe/<kind> + max_tokens, contracts.request_evidence_record) read back from the request evidence
                       # beside the store — ONE tag source per leg, never also the usage ledger (its rows would double the
                       # counts) — each kind against its own pinned reference (render -> the card row = --expect; bites and
                       # lens_on_bite -> greedy), the shape cross-checked against the recorder's kind table; an untagged or
                       # mis-shaped probe is UNCLASSIFIED (fails the probe class: a finding, never widened). NO --probe-cap:
                       # under kinds the recorder's default is the kind table's caps (cells.RECORDER_PROBE_CAPS mirrors them for
                       # the client-cap collision check; 16,384 collides with none). TWO tag sources, one per EMITTER:
                       # the DRIVER's evidence beside the store, and the STANDUP's own load-check sidecar in the serving dir —
                       # the load check fires its probes before the driver exists, so without the second source its requests
                       # are untagged and the whole render kind reads UNCLASSIFIED at cap 512 (measured 34 there: 32 load
                       # check + 2 render). Two emitters is not double-counting: they are disjoint request sets, which is
                       # exactly the distinction "never the evidence file AND the ledger for the same requests" draws. ONE
                       # function, read by the job and by the dry run alike. `--require-wired` (a strict SUBSET;
                       # math-eight-arms-extension decisions-w4-launch L1 G0): this launcher DECLARES the opt-in
                       # (--enable-log-requests on every extension / steering row) and reads AFTER the leg, so a record
                       # proving the instrument was never wired — UNREADABLE (exit 3: no readable log; silence with no
                       # attestation = launched without the flag; attested but muted) or FLAG_SET_NO_REQUESTS_YET (exit 4:
                       # a post-leg read that finds no request) — is a silent hole, not a finding about the sampler; every
                       # verdict ABOUT the sampler (AGREES, DIVERGED, UNCOMPARED, UNCLASSIFIED_REQUESTS, PROBE_NOT_INSTRUMENTED)
                       # still exits 0. The caller keeps its non-gating policy and LOGS 3 / 4 by name (m5t_record_resolved_sampler).
  local kv
  RSA=("$CPY" "$REPO_DIR/$M5T_RESOLVED_SAMPLER_SCRIPT" --vllm-log "$SERVE_DIR/math500/vllm_server.log"
       --out "$OUT_ROOT/$M5T_RESOLVED_SAMPLER_FILENAME" --client-cap "$M5T_CAP"
       --probe-expect "${M5T_RECORDER_PROBE_EXPECT:?M5T_RECORDER_PROBE_EXPECT (cells.py literal)}"
       --probe-tags "$OUT_ROOT/${M5T_RECORDER_PROBE_TAGS_FILENAME:?M5T_RECORDER_PROBE_TAGS_FILENAME (cells.py literal)}"
       --probe-tags "$SERVE_DIR/math500/${M5T_RECORDER_LOAD_CHECK_TAGS_FILENAME:?M5T_RECORDER_LOAD_CHECK_TAGS_FILENAME (cells.py literal)}"
       --require-wired)
  while IFS= read -r kv; do [ -z "$kv" ] || RSA+=(--expect "$kv"); done < <("$(m5t_py)" -c 'import json, sys
for k, v in sorted(json.loads(sys.argv[1]).items()): print(f"{k}={v}")' "$M5T_SAMPLING_PARAMS")
}

m5t_record_resolved_sampler(){   # after the driver, BEFORE the standup stops (the server log is complete; the standup is still the
                                 # job's), iff the row opted in: the attestation is RECORDED, never the job's result — a recorder
                                 # failure or a DIVERGED verdict is logged and rides the record; the analysis files it as a finding
                                 # (the recorder's own exit policy: 0 for every recorded verdict, 2 for a refused argument; under
                                 # --require-wired 3 / 4 ONLY when the record proves the declared instrument was never wired —
                                 # logged BY NAME here so the hole is visible in the .out, still never the leg's result)
  local rc
  [ "${M5T_LOG_REQUESTS_EFFECTIVE:-0}" = 1 ] || { f22_log "resolved sampler: not recorded (log_requests=0 on this row)"; return 0; }
  m5t_recorder_argv
  f22_log "resolved sampler: ${RSA[*]}"
  ( cd "$REPO_DIR" && env CUDA_VISIBLE_DEVICES="" PYTHONPATH="$REPO_DIR/src:$REPO_DIR/scripts" "${RSA[@]}" ) \
    > "$SERVE_DIR/math500/resolved_sampler.log" 2>&1; rc=$?
  case "$rc" in
    0) ;;
    3|4) f22_log "resolved sampler recorder rc=$rc: INSTRUMENT NOT WIRED (--require-wired: $([ "$rc" = 3 ] && echo UNREADABLE || echo FLAG_SET_NO_REQUESTS_YET) — the row declared --enable-log-requests but the record proves no request was logged; recorded, not the job's result; see $SERVE_DIR/math500/resolved_sampler.log)";;
    *) f22_log "resolved sampler recorder rc=$rc (recorded, not the job's result; see $SERVE_DIR/math500/resolved_sampler.log)";;
  esac
  [ -f "$OUT_ROOT/$M5T_RESOLVED_SAMPLER_FILENAME" ] && f22_log "resolved sampler: $(head -c 400 "$SERVE_DIR/math500/resolved_sampler.log" | head -1)" || true
}

m5t_row_identity(){   # the launcher's ROW-IDENTITY stamp (decisions-impl-laneA A2; the IFEval superset + per-row pin):
                      # written into the out root AFTER the driver, by cells.py from the matrix row, with sampling_ref DERIVED from
                      # the driver's own manifest sampling block through contracts.sampling_ref and held to gen_stats + the outcome
                      # items (fix A3: the driver writes no manifest-level sampling_ref). The stamp says `complete: false` with the
                      # reasons when the driver wrote no manifest or the evidence is missing / disagrees — never a success-shaped
                      # stamp; the verdict is LOGGED. Atomic; recorded, never the job's result. cells.py imports the contracts
                      # lazily on this path, so it runs from the client venv (m5t_py -> CPY), from the job's checkout.
  local tmp="$OUT_ROOT/$M5T_ROW_IDENTITY_FILENAME.tmp" py; py=$(m5t_py)
  if ( cd "$REPO_DIR" && PYTHONPATH="$REPO_DIR/src:$REPO_DIR/scripts" "$py" "$REPO_DIR/$M5T_DIR/cells.py" row --index "$M5T_INDEX" --identity --sha "$WAVE_SHA" --out-root "$OUT_ROOT" ) > "$tmp" 2>"$SERVE_DIR/math500/row_identity.log"; then
    mv -f "$tmp" "$OUT_ROOT/$M5T_ROW_IDENTITY_FILENAME"
    f22_log "row identity: $OUT_ROOT/$M5T_ROW_IDENTITY_FILENAME (producer_tree_sha = pin = $WAVE_SHA) $(m5t_row_identity_verdict "$OUT_ROOT/$M5T_ROW_IDENTITY_FILENAME")"
  else
    rm -f "$tmp"; f22_log "row identity: NOT written (cells.py row --identity failed; see $SERVE_DIR/math500/row_identity.log)"
  fi
}

m5t_row_identity_verdict(){   # $1 = a written row_identity.json -> one line: complete=true|false sampling_ref=<ref|None> [problems=...]
  "$(m5t_py)" - "$1" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
line = f"complete={str(d['complete']).lower()} sampling_ref={d['identity']['sampling_ref']}"
if not d["complete"]:
    line += " problems=" + "; ".join(d["problems"])
print(line)
PY
}

m5t_print_plan(){   # the dry run's verbatim argv lines for THIS row (m5t_dry_run.py reads them)
  local si_cell=""; if [ "$M5T_ROUTE_KIND" = prompted ]; then si_cell="$CELL"; fi
  m5t_core
  printf 'ROW: index=%s job_id=%s key=%s cell=%s sampling=%s fitted=%s route=%s family=%s time=%s out_root=%s\n' \
    "$M5T_INDEX" "$M5T_JOB_ID" "$KEY" "$CELL" "$SAMPLING_MODE" "$FITTED" "$M5T_ROUTE_KIND" "$M5T_FAMILY" "$M5T_TIME" "$OUT_ROOT"
  printf 'WAVE: wave=%s status=%s view=%s budget_class=%s staged_id=%s si_sha256=%s log_requests=%s\n' \
    "$M5T_WAVE" "$M5T_STATUS" "$M5T_VIEW" "$M5T_BUDGET_CLASS" "${M5T_STAGED_ID:-none}" "${M5T_SI_SHA256:-none}" "$M5T_LOG_REQUESTS_EFFECTIVE"
  # the serving path the standup will take, as f22_init RESOLVES it (the dry-run branch runs the same f22_select_serving_path over the
  # same declaration): the row's pin (lens on an external-object row; selection=explicit) or the kind-aware default over the declared
  # launch cells (selection=auto) — and the cells declared (F22_LAUNCH_CELLS: `none` + the row's harness cell, the standup's cells_requested)
  # …the pin the row declared (m5t_select_serving_path: empty = none) WITH ITS REASON — kind:alphasteer | store:lens | unpinned:no-store |
  # store:native on an external row (the store under the dry run's wave root), unpinned:not-external on every other row
  printf 'SERVING_PATH: path=%s pinned_by_row=%s reason=%s selection=%s\n' "${F22_SERVING_PATH:-unresolved}" "${M5T_SERVING_PATH_PINNED:-none}" "${M5T_SERVING_PATH_REASON:-unresolved}" "${F22_SERVING_PATH_MODE:-unresolved}"
  printf 'LAUNCH_CELLS: source=F22_LAUNCH_CELLS cells=%s\n' "${F22_LAUNCH_CELLS[*]:-undeclared}"
  # E3: the it-doc row facts the bank door binds (the record, the lane, the ruled staged id, the object kind) and the qualification
  [ "${M5T_OBJECT_SOURCE:-}" != itdoc ] || printf 'ITDOC: lane=%s staged_id=%s object_kind=%s admission=%s bank_view_rel_primary=%s bank_view_rel_overflow=%s qualification=%s\n' \
    "$M5T_LANE" "$M5T_STAGED_ID" "$M5T_OBJECT_KIND" "$M5T_ITDOC_ADMISSION" "${M5T_BANK_VIEW_REL_PRIMARY:-none}" "${M5T_BANK_VIEW_REL_OVERFLOW:-none}" "$M5T_QUALIFICATION"
  printf 'SAMPLING: mode=%s params=%s\nSAMPLING_SOURCE: %s\n' "$SAMPLING_MODE" "$M5T_SAMPLING_PARAMS" "$M5T_SAMPLING_SOURCE"
  # b64-capability-agentic-queue: the SEED row's opt-in Inspect seed rule, its standup companions and its --bank-view
  # stamp — nothing printed on a matrix row (the three names are empty there); the launch script's greps read these lines
  [ -z "${M5T_INSPECT_SEED_POLICY:-}" ] || printf 'INSPECT_SEED: policy=%s base=%s\n' "$M5T_INSPECT_SEED_POLICY" "$M5T_INSPECT_SEED_BASE"
  [ -z "${M5T_STANDUP_COMPANION_CELLS:-}" ] || printf 'COMPANIONS: cells=%s pins=%s\n' "$M5T_STANDUP_COMPANION_CELLS" "$M5T_STANDUP_COMPANION_PINS"
  [ -z "${M5T_BANK_VIEW_STAMP:-}" ] || printf 'BANK_VIEW_STAMP: %s (--bank-view; view=%s bank_view=%s)\n' "$M5T_BANK_VIEW_STAMP" "$M5T_VIEW" "${M5T_BANK_VIEW:-none}"
  printf 'GEOMETRY: %s %s x %s cap %s retry %s thinking %s\n' "$M5T_GEOMETRY_EVAL" "$M5T_GEOMETRY_L" "$TARGET_S" "$M5T_CAP" "$M5T_RETRY" "$M5T_THINKING_MODE"
  printf 'BUDGET: %s (fit sidecar re-checked at launch: fits, cap %s, %s / %s, arm %s, tree %s, si %s)\n' \
    "$BUDGET_SRC" "$M5T_CAP" "$M5T_GEOMETRY_EVAL" "$M5T_GEOMETRY_L" "$M5T_THINKING_MODE" "$WAVE_SHA" "${si_cell:-none}"
  m5t_standup_extra
  st_print_argv STANDUP "${F22_STANDUP_EXTRA[@]}"
  # the launch CONFIGURATION f22_standup composes from (fix A4): its mode positional, whether --thinking-on rides it
  # (never: m5t_standup_extra refused the variable f22_standup reads), so the server-default render pin the composer
  # writes, the REQUEST-level thinking mode the driver sends, the memory utilisation. The compatibility preflight reads
  # this line + the STANDUP argv to compose the wave's launch under the job's own configuration, never a re-typed one.
  printf 'STANDUP_POLICY: mode=%s thinking_on_flag=%s server_default_thinking=%s request_thinking=%s gpu_mem_util=%s max_logprobs=none\n' \
    "$M5T_STANDUP_MODE" "$M5T_STANDUP_THINKING_FLAG" "$([ "$M5T_STANDUP_THINKING_FLAG" = 1 ] && echo on || echo off)" "$M5T_THINKING_MODE" "$GPU_MEM_UTIL"
  st_print_ledger_line "$OUT_ROOT/$M5T_LEDGER_FILENAME" "$(dirname "$TABLE")/$M5T_LEDGER_ENDPOINT_FILENAME"
  m5t_driver_argv
  st_print_argv DRIVER "${DRV[@]}"
  st_print_driver_env "$M5T_INSPECT_BASE_URL_ENV" "$LEDGER_URL" "$M5T_INSPECT_API_KEY_ENV"
  if [ "${M5T_LOG_REQUESTS_EFFECTIVE:-0}" = 1 ]; then m5t_recorder_argv; st_print_argv RESOLVED_SAMPLER "${RSA[@]}"
  else printf 'RESOLVED_SAMPLER: not recorded (log_requests=0)\n'; fi
  printf 'ROW_IDENTITY: %s (cells.py row --index %s --identity --sha %s --out-root %s)\n' "$OUT_ROOT/$M5T_ROW_IDENTITY_FILENAME" "$M5T_INDEX" "$WAVE_SHA" "$OUT_ROOT"
  if [ "$FITTED" = 1 ] && [ -n "$M5T_TUPLE" ]; then printf 'OUT OF SUPPORT: tuple=%s pin=%s served_positions_expected=all stamps=(cell-stable, always, thinking_on)\n' "$M5T_TUPLE" "$PIN"
  elif [ "$FITTED" = 1 ]; then printf 'IN SUPPORT: route=%s pin=%s thinking=%s served_positions_expected=all stamps=(cell-stable, always, thinking_on)\n' "$M5T_ROUTE_KIND" "$PIN" "$M5T_THINKING_MODE"
  else printf 'IN SUPPORT: route=%s stamps=(cell-stable, not-consulted, thinking_on)\n' "$M5T_ROUTE_KIND"; fi
}
