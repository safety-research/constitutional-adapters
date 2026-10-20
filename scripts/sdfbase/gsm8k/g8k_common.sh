#!/bin/bash
# gsm8k-paired200-paper-objects — the shell body every scripts/sdfbase/gsm8k/*.sbatch sources AFTER
# scripts/sdfbase/full22/common.sh (bash). The battery's job shape (f22_* — init, venvs, standup, stop, result) is
# reused unchanged and the TASK-INDEPENDENT seams (the ledger, the proxy token, the budget artifact, the endpoint checks,
# the driver argv's head / tail, the dry-run lines, the attempt seam) come from the shared execution core
# scripts/sdfbase/full22/served_thinking.sh (one core under two launcher families); this file adds ONLY what is
# this experiment's: the matrix row with its explicit VIEW (g8k_cells.py over the frozen matrix), the overlay config per TAG,
# the out root per rung, the standup extras (the header-located cells DECLARED as <cell>@thinking — the ONE declared
# spelling the matrix row carries as G8K_STANDUP_CELL / G8K_STANDUP_HARNESS_CELL, read by the standup's --cell, its digest
# key and the fresh table's positions lookup alike), the driver's task flags (the explicit 8,192 / 16,384
# cap as STATEMENT + SETTER, the rung's item subset), the datasets guard for BOTH datasets (the offline layout), the
# model-based placement refusal, the HELD refusal.
# NEVER --export= (hard rule 1): every value travels positionally, through g8k_cells.py's row, or these functions.

G8K_DIR="scripts/sdfbase/gsm8k"
# THE MATRIX (llama-paper-rows P11): the sbatch's trailing positional names it (`default` = the published 42-row matrix;
# `llama` = the Llama rows, schema g8k_matrix/2, their OWN wave root); g8k_cells.py resolves the name, and g8k_literals
# re-evaluates G8K_WAVE_ROOT_REL from the matrix's document BEFORE the wave root is derived — the line below is the
# default matrix's root, the value every launcher started from
G8K_MATRIX="${G8K_MATRIX:-default}"
G8K_WAVE_ROOT_REL="exp/gsm8k_paired200_paper_objects"
# the OVERFLOW cluster's registry label (f22_init's CLUSTER, from the job's SISTEER_CLUSTER_LABEL) — the ONE place
# this file names it, so g8k_bank_view can pick a steering row's per-cluster EXTERNAL bank view (the MATH-500 launcher's
# M5T_OVERFLOW_CLUSTER precedent)
G8K_OVERFLOW_CLUSTER=cluster_b
# the Nemotron fused-kernel attestation is a PREREQUISITE for every gsm8k job too (common.sh f22_model_facts; m5t LB2)
export F22_REQUIRE_NEMO_BACKEND=1

g8k_py(){   # the python that reads g8k_cells.py (stdlib only): the client venv once built, else the node's python3
  if [ -x "${CPY:-}" ]; then echo "$CPY"; else command -v python3; fi
}

g8k_core(){   # source the shared execution core once, from THIS job's checkout ($REPO_DIR, set by the sbatch prologue); a
              # tree without it refuses BY NAME after the cleanup trap is armed, never `command not found` mid-job
  declare -F st_ledger_up >/dev/null 2>&1 && return 0
  [ -r "${REPO_DIR:?REPO_DIR}/scripts/sdfbase/full22/served_thinking.sh" ] \
    || f22_fatal "no shared execution core at $REPO_DIR/scripts/sdfbase/full22/served_thinking.sh — the helpers and the job must come from ONE tree"
  source "$REPO_DIR/scripts/sdfbase/full22/served_thinking.sh"
}

g8k_literals(){   # the run-level literals of THIS MATRIX (g8k_cells.py RUN_LITERALS_SH + the matrix's wave root — ONE source):
                  # evaluated before any row is known, so the wave root is the matrix's and the datasets guard reads the same
                  # names the row later re-asserts (no literal is typed twice: g8k_cells.py carries it, the test pins that every
                  # emitted name is read). Needs REPO_DIR (the launcher prologue) — not f22_init.
  eval "$("$(g8k_py)" "${REPO_DIR:?REPO_DIR}/$G8K_DIR/g8k_cells.py" literals --sh --matrix "$G8K_MATRIX")" \
    || f22_fatal "g8k_cells.py literals --sh --matrix $G8K_MATRIX failed — the run literals are the guard's input and the wave root"
}

g8k_select_serving_path(){   # ARRAY | <key> <view> <cell> -> the ROW's serving route, DECLARED to f22_init (common.sh f22_select_serving_path,
                             # the kind-aware default of 2026-09-15; the MATH-500 launcher's m5t_select_serving_path, mirrored):
                             # F22_LAUNCH_CELLS — a Bash ARRAY of the HARNESS cells this job's standup serves (`none` + the row's standup cell
                             # in harness spelling, the `@thinking` variant when declared; g8k_cells.py launch_cells_for: exactly what
                             # `--cell "$G8K_STANDUP_CELL"` makes the standup's cells_requested read) — on EVERY row, and F22_SERVING_PATH
                             # from the ROW when it pins one. THE PIN FOLLOWS THE STORE (g8k_cells.py serving_path_pin,
                             # the rule in full22/steer_store_pin.py): an AlphaSteer row pins `lens` by kind (no native form); an Arditi row
                             # pins `lens` iff its steer store ON RECORD — its out root + rung sibling under F22_WAVE_ROOT on THIS cluster — is
                             # a lens store (store:lens), is unpinned where no store exists (unpinned:no-store — the kind-aware default then
                             # selects native, Arditi's residual slot since native-psr-serving) or where the store is native-stamped
                             # (store:native); a mixed-engine store refuses by name. So F22_WAVE_ROOT must be exported BEFORE this runs
                             # (g8k_init does; the dry run sets it from its stub root) and the store is read here, at the launcher's first
                             # row-resolution step — before f22_init fixes the path, before the out root is created, before any marker.
                             # An explicit pin WINS at f22_init (recorded EXPLICIT); a row that pins none has its path selected BY KIND over
                             # the declared cells (a LoRA / PSR / Arditi cell -> lora_native; none / prompted -> the entry-point default) and
                             # logged `[serving-path] AUTO-selected`. A submitter value CONFLICTING with a pin refuses by name; so does a
                             # submitter value other than lora_native on a store:native row (a lens leg into a native store would be the mixed
                             # store the rule refuses). Before this declaration the launcher DEFERRED (it owns F22_WAVE_ROOT) and the standup
                             # bound the path from the overlay manifest.
  local py out cells c root pin reason declared row_args=() pins_args=() pins_list=() pins_desc; py=$(g8k_py)
  root="${F22_WAVE_ROOT:-}"
  [ -n "$root" ] || f22_fatal "g8k_select_serving_path: F22_WAVE_ROOT is unset — the row's serving-path pin FOLLOWS ITS STORE under this matrix's wave root on this cluster, so g8k_init exports F22_WAVE_ROOT before declaring the route (the dry run sets it from its stub root)"
  # the PINS source — the repo's store-pins records under this checkout (the CLI's default; logged `pins checkout`) — is the launch tree's.
  # G8K_STORE_PINS (a colon-separated list of records) is the TESTS' seam for a synthetic record: honoured ONLY in an explicit local dry run
  # (G8K_DRY_RUN=1) outside any Slurm job, logged whenever honoured; a job that inherits it REFUSES by name — the override REPLACES the
  # admitted records, and a known lens store would relaunch natively behind it
  pins_desc="${G8K_STORE_PINS:-checkout}"
  if [ -n "${G8K_STORE_PINS:-}" ]; then
    if [ -n "${SLURM_JOB_ID:-}" ] || [ "${G8K_DRY_RUN:-0}" != 1 ]; then
      f22_fatal "G8K_STORE_PINS is set (${G8K_STORE_PINS}) but this is not a local dry run (SLURM_JOB_ID=${SLURM_JOB_ID:-unset}, G8K_DRY_RUN=${G8K_DRY_RUN:-0}) — the store-pins override is a test seam that REPLACES the admitted records; refused by name so a known store is never relaunched on another engine; unset it"
    fi
    IFS=: read -r -a pins_list <<< "$G8K_STORE_PINS"
    for c in "${pins_list[@]}"; do [ -z "$c" ] || pins_args+=(--store-pins "$c"); done
  fi
  if [ "${1:?row}" = ARRAY ]; then
    [ -n "${SLURM_ARRAY_TASK_ID:-}" ] || f22_fatal "ARRAY given but SLURM_ARRAY_TASK_ID is unset — submit with --array=<indices> (g8k_cells.py plan)"
    row_args=(--index "$SLURM_ARRAY_TASK_ID")
    out=$("$py" "$REPO_DIR/$G8K_DIR/g8k_cells.py" row --matrix "$G8K_MATRIX" --index "$SLURM_ARRAY_TASK_ID" --serving-path --wave-root "$root" ${pins_args[@]+"${pins_args[@]}"}) || f22_fatal "g8k_cells.py: no row at array index $SLURM_ARRAY_TASK_ID of matrix $G8K_MATRIX, or its store on record refused the pin (serving path)"
    cells=$("$py" "$REPO_DIR/$G8K_DIR/g8k_cells.py" row --matrix "$G8K_MATRIX" --index "$SLURM_ARRAY_TASK_ID" --launch-cells) || f22_fatal "g8k_cells.py: no row at array index $SLURM_ARRAY_TASK_ID of matrix $G8K_MATRIX (launch cells)"
  else
    row_args=(--key "$1" --view "${2:?view}" --cell "${3:?cell}")
    out=$("$py" "$REPO_DIR/$G8K_DIR/g8k_cells.py" row --matrix "$G8K_MATRIX" --key "$1" --view "${2:?view}" --cell "${3:?cell}" --serving-path --wave-root "$root" ${pins_args[@]+"${pins_args[@]}"}) || f22_fatal "g8k_cells.py: ($1, $2, $3) is not a row of matrix $G8K_MATRIX, or its store on record refused the pin (serving path)"
    cells=$("$py" "$REPO_DIR/$G8K_DIR/g8k_cells.py" row --matrix "$G8K_MATRIX" --key "$1" --view "${2:?view}" --cell "${3:?cell}" --launch-cells) || f22_fatal "g8k_cells.py: ($1, $2, $3) is not a row of matrix $G8K_MATRIX (launch cells)"
  fi
  # two lines: the pin (empty = unpinned) and its reason (g8k_cells.py / steer_store_pin.pin_lines)
  { IFS= read -r pin; IFS= read -r reason; } <<< "$out"
  [ -n "$reason" ] || f22_fatal "g8k_cells.py row --serving-path printed no reason for the row (got: $out) — the pin is always explained"
  # THE PATH THE ROW'S OWN RECORD DECLARES for every leg of its arm (arditi-allpos-cap, 2026-09-17;
  # g8k_cells.declared_serving_path -> the admission entry's attestation_expect.serving_path; EMPTY on every row whose record
  # declares none, which is every row that existed before this). For the all-positions Arditi arm the path is part of the
  # MEASUREMENT, not a performance choice: the direction is specified on the graph-captured NATIVE residual slot, and the lens
  # WOULD serve a `vector_all` blob — so a submitter's F22_SERVING_PATH=lens does not fail the leg, it completes one and files a
  # different intervention's numbers under this arm's name. Refused HERE, at the launcher's first row-resolution step, because
  # f22_init fixes an explicit export as EXPLICIT and nothing downstream would question it. (The arm-level twin — the arms
  # table's `serving_path:` read by f22_arm_serving_policy — is another author's change; this door reads the ROW's record, so
  # the two agree by construction or disagree visibly, and this one binds on a tree without the arms table.)
  declared=$("$py" "$REPO_DIR/$G8K_DIR/g8k_cells.py" row --matrix "$G8K_MATRIX" "${row_args[@]}" --declared-serving-path) \
    || f22_fatal "g8k_cells.py row --declared-serving-path failed for this row of matrix $G8K_MATRIX"
  G8K_DECLARED_SERVING_PATH="$declared"
  if [ -n "$declared" ] && [ -n "${F22_SERVING_PATH:-}" ] && [ "$F22_SERVING_PATH" != "$declared" ]; then
    f22_fatal "the row's RECORD declares serving path '$declared' for every leg of this arm (the object is specified on that path; the other path would serve these bytes as a DIFFERENT intervention and file it under this arm's name) but the submitter's shell exports F22_SERVING_PATH=$F22_SERVING_PATH — refused by name; unset it"
  fi
  [ -z "$declared" ] || f22_log "serving path DECLARED by the row's record: $declared (every leg of this arm; a conflicting F22_SERVING_PATH is refused here, before f22_init fixes one)"
  # THE DECLARED CELL SOURCE: f22_select_serving_path's precedence is F22_SERVING_PATH > F22_LAUNCH_CELLS > F22_LAUNCH_MANIFEST > deferred
  # (F22_WAVE_ROOT) > refused by name — so with this array declared the launcher never defers, and a row-less init (the diagnostic
  # launchers, the overlap leg, the ladder plan) still defers to the standup, which binds from the manifest it serves
  F22_LAUNCH_CELLS=()
  while IFS= read -r c; do [ -z "$c" ] || F22_LAUNCH_CELLS+=("$c"); done <<< "$cells"
  [ "${#F22_LAUNCH_CELLS[@]}" -ge 1 ] || f22_fatal "g8k_cells.py row --launch-cells printed no cell for the row — every standup serves at least none"
  G8K_SERVING_PATH_PINNED="$pin"; G8K_SERVING_PATH_REASON="$reason"
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

g8k_init(){   # $1 = wave sha -> f22_init under THIS MATRIX's wave root (never the full22 wave's, never MATH-500's, never
              # another matrix's: the Llama rows live under exp/gsm8k_paired200_paper_objects_llama)
  local sha="${1:?wave sha}" shared d
  # the cluster's shared root is the job's site value (the front door passes the cluster block's root to every job), never
  # guessed from which storage paths exist; f22_init reads the same variable
  [ -n "${SISTEER_SHARED_ROOT:-}" ] \
    || { echo "FATAL: set SISTEER_SHARED_ROOT, the shared root of the cluster (the front door exports it from the site config)" >&2; exit 1; }
  shared="$SISTEER_SHARED_ROOT"
  # $2.. = the row (ARRAY | <key> <view> <cell>): its serving ROUTE is declared here, before f22_init fixes the path from the environment —
  # F22_LAUNCH_CELLS (the cells this job's standup serves) on every row, F22_SERVING_PATH when the row pins one — so f22_init resolves
  # EXPLICIT on a pinned row and kind-aware AUTO over the declared cells otherwise (the contract: every caller declares its cell
  # source). The pin FOLLOWS THE STORE at the row's out root (2026-09-15), so the matrix's wave root is derived and exported FIRST
  # (g8k_literals -> F22_WAVE_ROOT) and the declaration reads the store under it. Called WITHOUT a row (the smoke, the overlap leg, the
  # ladder plan, the prod dry run) nothing is declared and the path is DEFERRED to the standup, which binds it from the manifest it
  # serves — a job that owns its wave root (F22_WAVE_ROOT, exported here) may defer; f22_init logs which.
  shift
  g8k_literals
  export F22_WAVE_ROOT="${F22_WAVE_ROOT:-$shared/$G8K_WAVE_ROOT_REL}"
  [ $# -ge 1 ] && g8k_select_serving_path "$@"
  f22_init "$sha"
  # The `datasets` library, OFFLINE (hf_env.sh sets HF_HUB_OFFLINE=1), reads a PREPARED arrow cache under HF_DATASETS_CACHE
  # — the hub snapshot alone does not serve it. The GSM8K wrapper loads TWO datasets: openai/gsm8k (the
  # stock task's items + its 10-shot TRAIN pool) and madrylab/gsm8k-platinum (the pairing that narrows the task to the 200
  # paired items), both PREPARED per user on both clusters. The guard names BOTH — a job that would die at the
  # second load 3 minutes into a standup dies here, by name, in its first seconds.
  export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$shared/hf_datasets}"
  g8k_datasets_guard
  # the launcher located its helpers before sourcing them; a G8K_REPO_DIR override must BE the job's
  # checkout — helpers from one tree and a job from another is refused, never run
  [ -z "${G8K_REPO_DIR:-}" ] || [ "$REPO_DIR" = "$G8K_REPO_DIR" ] || f22_fatal "G8K_REPO_DIR=$G8K_REPO_DIR but the wave checkout is $REPO_DIR — the helpers and the job must come from ONE tree"
  # the Inspect evals cache lives under the wave root (the datasets are pre-fetched into the HF cache; offline)
  export INSPECT_EVALS_CACHE_DIR="${INSPECT_EVALS_CACHE_DIR:-$WAVE_ROOT/inspect_evals_cache}"; mkdir -p "$INSPECT_EVALS_CACHE_DIR"
}

g8k_datasets_guard(){   # the directory guard over HF_DATASETS_CACHE (necessary, not sufficient — g8k_datasets_probe is the authority):
                        # each dataset dir of G8K_DATASETS_REQUIRED AND the offline CONFIG dir `datasets` 5.0.1 looks for under it
                        # (G8K_DATASETS_CONFIG_DIRS; openai/gsm8k is looked up offline as `default-data_dir=main` while the
                        # online prepare wrote a hash-named dir, so the prepared caches carry `default-data_dir=main` as a SYMLINK to
                        # it — `-d` follows the link, so that layout passes; a dangling or absent link refuses by name with the recipe)
  local d target
  for d in ${G8K_DATASETS_REQUIRED:?G8K_DATASETS_REQUIRED (g8k_cells.py literals)}; do
    [ -d "$HF_DATASETS_CACHE/$d" ] \
      || f22_fatal "HF_DATASETS_CACHE=$HF_DATASETS_CACHE carries no prepared $d (the GSM8K wrapper loads openai/gsm8k AND madrylab/gsm8k-platinum offline: prepare each once per user per cluster by an ONLINE load with HF_HUB_OFFLINE=0 set EXPLICITLY — the harness defaults it to 1 — through the wrapper's own plan_task(gsm8k_paired200) with this HF_DATASETS_CACHE; plan §7, D2, D18)"
  done
  for d in ${G8K_DATASETS_CONFIG_DIRS:?G8K_DATASETS_CONFIG_DIRS (g8k_cells.py literals)}; do
    if [ -d "$HF_DATASETS_CACHE/$d" ]; then
      target=$(readlink "$HF_DATASETS_CACHE/$d" 2>/dev/null || true)
      f22_log "datasets guard: $d present${target:+ (symlink -> $target)}"
    else
      f22_fatal "HF_DATASETS_CACHE=$HF_DATASETS_CACHE carries no offline config dir $d: the datasets library looks OFFLINE for the config dir named by the load signature (openai/gsm8k: default-data_dir=main; gsm8k-platinum: main) while an ONLINE prepare names openai/gsm8k's arrow cache by a kwargs hash (e.g. $HF_DATASETS_CACHE/${d%%/*}/default-8cbc416ed05c3005). RECIPE (D18): (1) prepare ONLINE with HF_HUB_OFFLINE=0 set explicitly, through the wrapper's own plan_task(gsm8k_paired200) at the task's exact signatures (openai/gsm8k test+train at inspect_evals' pinned revision; madrylab/gsm8k-platinum config main); (2) ln -s <the hash-named dir> $HF_DATASETS_CACHE/openai___gsm8k/default-data_dir=main (a symlink alias; this guard follows it, a dangling one refuses); (3) re-run g8k_datasets_probe OFFLINE on a compute node of this cluster"
    fi
  done
}

g8k_datasets_probe(){   # the AUTHORITATIVE datasets check: the wrapper's OWN task plan under the
                        # job's OFFLINE env, from the client venv, BEFORE any standup. g8k_init's directory guard is necessary, not
                        # sufficient: `datasets` names an offline config by its load signature (openai/gsm8k -> `default-data_dir=main`)
                        # while an online prepare can leave a hash-named config dir — the grading host's cache had exactly that, and the
                        # directory was present while every offline load failed. A miss here dies by name in the job's first minutes.
  ( cd "$REPO_DIR" && CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONPATH="$REPO_DIR/src:$REPO_DIR/scripts" "$CPY" - "$G8K_INSPECT_TASK" "$G8K_N_ITEMS" <<'PY'
import sys
from pathlib import Path
from sisteer.capeval.tiers import inspect_served as isv
task, n = sys.argv[1], int(sys.argv[2])
plan = isv.plan_task(task, grader_model=None, thinking="off", repo_root=Path.cwd())
ids = list(plan.sample_ids or [])
assert len(ids) == n, f"{task} planned {len(ids)} items offline, not {n}"
print(f"datasets probe OK: {task} plans {len(ids)} paired items OFFLINE from HF_DATASETS_CACHE (both datasets resolved)")
PY
  ) || f22_fatal "datasets probe FAILED: the wrapper cannot plan $G8K_INSPECT_TASK OFFLINE from HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-unset}. RECIPE (D18, verified on both clusters): (1) prepare ONLINE through THIS SAME PLAN with HF_HUB_OFFLINE=0 set EXPLICITLY (the harness's object_identity setdefaults it to 1) at the task's exact signatures — openai/gsm8k test+train (data_dir main at inspect_evals' pinned revision cc7b047b…) and madrylab/gsm8k-platinum (config main at pins.yaml's revision); (2) datasets 5.0.1 names openai/gsm8k's ONLINE arrow cache by a kwargs hash (openai___gsm8k/default-8cbc416ed05c3005/…) but looks OFFLINE for openai___gsm8k/default-data_dir=main, so add the alias: ln -s default-8cbc416ed05c3005 $HF_DATASETS_CACHE/openai___gsm8k/default-data_dir=main (platinum's config main IS named main offline); (3) re-run this probe OFFLINE on a compute node of this cluster — the directory guard alone cannot see a config-name mismatch (plan §7, D2, D18, m5t D112/D113)"
}

g8k_row(){   # ARRAY | <key> <view> <cell>  (+ the rung in G8K_RUNG_ARG, 8192 default; the matrix in G8K_MATRIX) -> evals the
             # matrix row: KEY VIEW TAG CELL CELLSLUG FITTED DECLARED RUNG PIN STATUS G8K_* (G8K_THINKING the row's arm —
             # `on` | `not_available` — G8K_SAMPLING_MODE its sampling row, G8K_WAVE_ROOT_REL its matrix's root); then the two
             # refusals that belong at job START, by name: a HELD row and a misplaced `primary` row
  local py out rung="${G8K_RUNG_ARG:-8192}"; py=$(g8k_py)
  if [ "${1:?row}" = ARRAY ]; then
    [ -n "${SLURM_ARRAY_TASK_ID:-}" ] || f22_fatal "ARRAY given but SLURM_ARRAY_TASK_ID is unset — submit with --array=<indices> (g8k_cells.py plan)"
    out=$("$py" "$REPO_DIR/$G8K_DIR/g8k_cells.py" row --matrix "$G8K_MATRIX" --index "$SLURM_ARRAY_TASK_ID" --rung "$rung" --sh) || f22_fatal "g8k_cells.py: no row at array index $SLURM_ARRAY_TASK_ID of matrix $G8K_MATRIX"
  else
    out=$("$py" "$REPO_DIR/$G8K_DIR/g8k_cells.py" row --matrix "$G8K_MATRIX" --key "$1" --view "${2:?view}" --cell "${3:?cell}" --rung "$rung" --sh) || f22_fatal "g8k_cells.py: ($1, $2, $3) is not a row of matrix $G8K_MATRIX (g8k_cells.py list --matrix $G8K_MATRIX)"
  fi
  eval "$out"
  f22_log "matrix row $G8K_INDEX: $G8K_JOB_ID matrix=$G8K_MATRIX key=$KEY view=$VIEW tag=$TAG cell=$CELL rung=$RUNG fitted=$FITTED declared=$DECLARED thinking=$G8K_THINKING sampling=$G8K_SAMPLING_MODE route=$G8K_ROUTE_KIND family=$G8K_FAMILY time=$G8K_TIME out=$G8K_OUT_REL status=$STATUS wave=${G8K_WAVE:-published} placement=$G8K_CLUSTER_CONSTRAINT${G8K_OBJECT_SOURCE:+ object_source=$G8K_OBJECT_SOURCE bank_view=$G8K_BANK_VIEW}${G8K_OBJECT_GENERATION:+ object_generation=$G8K_OBJECT_GENERATION}${G8K_INSPECT_SEED_POLICY:+ inspect_seed=$G8K_INSPECT_SEED_POLICY@$G8K_INSPECT_SEED_BASE}${G8K_STANDUP_COMPANION_CELLS:+ standup_companions=$G8K_STANDUP_COMPANION_CELLS}"
  if [ "$STATUS" = held ]; then
    # HELD: the row is in the matrix for its later wave and is NEVER run — no venv, no bank, no standup; RESULT HELD
    if [ "${G8K_DRY_RUN:-0}" = 1 ]; then printf 'HELD: index=%s job_id=%s tag=%s cell=%s reason=%s\n' "$G8K_INDEX" "$G8K_JOB_ID" "$TAG" "$CELL" "$G8K_HELD_REASON"
    else f22_log "HELD: $G8K_JOB_ID — $G8K_HELD_REASON"; fi
    G8K_RESULT=HELD; RC=0; exit 0
  fi
  if [ "$G8K_CLUSTER_CONSTRAINT" = primary ] && [ "${CLUSTER:-}" != "$G8K_PRIMARY_CLUSTER" ]; then
    f22_fatal "PLACEMENT: $G8K_JOB_ID carries cluster_constraint primary (the Nemotron fused-kernel serving overlay builds and is attested on the primary cluster alone) but this job runs on cluster=${CLUSTER:-unknown} — resubmit it on $G8K_PRIMARY_CLUSTER (g8k_cells.py plan refuses this share by name)"
  fi
  [ -z "$G8K_EXPECTED_REFUSAL" ] || f22_log "EXPECTED_REFUSAL: $G8K_JOB_ID — $G8K_EXPECTED_REFUSAL (the standup's own refusal is the record; never base-routed)"
}

g8k_assert_launchable(){   # THE BATCH BOUNDARY (ported from the MATH-500 launcher): the row g8k_row
                           # resolved must be in ITS wave's eligible set — g8k_cells.py's own predicate (assert_launchable = assert_eligible
                           # over eligible_rows, the one `plan --wave` ran over the submission), re-asserted by the JOB because the submission
                           # is not trusted: an --array index or a positional triple naming a PUBLISHED (complete) row resolved fine, kept
                           # status=launch (published-ness is the WAVE, never a status) and would have run into the published output root;
                           # a deferred row likewise. Runs BEFORE g8k_resolve_config / g8k_out_root, the venvs and any artifact write, so a
                           # refused row leaves nothing behind. The dry run and the diagnostic smoke (a published `none` row into a
                           # diagnostic root) do not call it. (A held row exited HELD from g8k_row already.)
  local py rec; py=$(g8k_py)
  rec=$("$py" "$REPO_DIR/$G8K_DIR/g8k_cells.py" row --matrix "$G8K_MATRIX" --index "$G8K_INDEX" --assert-launchable 2>&1) \
    || f22_fatal "matrix row $G8K_INDEX ($G8K_JOB_ID: wave=${G8K_WAVE:-published} status=$STATUS) is NOT launchable — $rec"
  f22_log "$rec"
}

g8k_bank_view(){   # the row's bank, CHECK-ONLY, dispatched on where its object comes from (plan P8): the campaign's own
                   # (roster-resolved) view of the row's VIEW for a published / extension row, the EXTERNAL lane's view for a
                   # steering row (G8K_OBJECT_SOURCE=external: the lane's bank view relative to the shared root for THIS cluster
                   # role, the cell dir inside it, the object's bytes held to the admission record's pin — st_external_bank_view_check).
                   # Either way TT_TWO_SI_DEFENSES_DIR is exported and a fitted row's bytes are held to its pin before any GPU minute.
  local rel
  if [ -n "${G8K_OBJECT_SOURCE:-}" ]; then
    g8k_core
    case "${CLUSTER:-}" in
      "$G8K_OVERFLOW_CLUSTER") rel="${G8K_BANK_VIEW_REL_OVERFLOW:-}";;
      *) rel="${G8K_BANK_VIEW_REL_PRIMARY:-}";;
    esac
    [ -n "$rel" ] || f22_fatal "row $G8K_INDEX ($G8K_JOB_ID) names no bank view for cluster ${CLUSTER:-unknown} — its object is staged on the other cluster only (g8k_cells.py G8K_BANK_VIEW_REL_*); run it where it is staged"
    case "$G8K_OBJECT_SOURCE" in
      itdoc)
        # E3: the re-fit row is BOUND to its bank entry through the it-doc record (sha + lane + staged id + sidecar of the kind), then the byte check
        st_itdoc_bank_check "$rel" "${G8K_BANK_CELL_DIR:?G8K_BANK_CELL_DIR}" "$PIN" "${G8K_ITDOC_ADMISSION:?G8K_ITDOC_ADMISSION}" "${G8K_LANE:?G8K_LANE}" "$KEY" "$VIEW" "$CELL"
        f22_log "it-doc bank: lane $G8K_LANE, staged id $G8K_STAGED_ID ($G8K_OBJECT_KIND), bank view $G8K_BANK_VIEW; qualification: $G8K_QUALIFICATION";;
      *)
        st_external_bank_view_check "$rel" "${G8K_BANK_CELL_DIR:?G8K_BANK_CELL_DIR}" "$PIN"
        # the ladder's selection OUTCOME (`chosen` / `a4_fallback` / the ruled or candidate word; `none` on an Arditi
        # row) and the grid-limited flag ride the log beside the verbatim selection label — the AlphaSteer points were not selected the same way
        f22_log "external bank: lane view $G8K_BANK_VIEW (the panel view is $VIEW); selection: $G8K_SELECTION_LABEL; selection_outcome=${G8K_SELECTION_OUTCOME:-none} grid_limited=${G8K_GRID_LIMITED:-0}";;
    esac
  elif [ "$FITTED" = 1 ]; then
    f22_bank_view_check "$CPY" "$CELL" "$PIN"
  else
    f22_bank_view_check "$CPY"
  fi
}

g8k_overlap_leg(){   # ARRAY | <key> -> evals the overlap leg (g8k_cells.py overlap-leg: the SAME variable names, thinking OFF /
                     # greedy / 512 / capability:gen_long / overlap/<key>/none/greedy_off512), then the placement refusal
  local py out; py=$(g8k_py)
  if [ "${1:?leg}" = ARRAY ]; then
    [ -n "${SLURM_ARRAY_TASK_ID:-}" ] || f22_fatal "ARRAY given but SLURM_ARRAY_TASK_ID is unset — submit with --array=<indices> (g8k_cells.py plan-overlap)"
    out=$("$py" "$REPO_DIR/$G8K_DIR/g8k_cells.py" overlap-leg --index "$SLURM_ARRAY_TASK_ID" --sh) || f22_fatal "g8k_cells.py: no overlap leg at array index $SLURM_ARRAY_TASK_ID"
  else
    out=$("$py" "$REPO_DIR/$G8K_DIR/g8k_cells.py" overlap-leg --key "$1" --sh) || f22_fatal "g8k_cells.py: no overlap leg for $1"
  fi
  eval "$out"
  f22_log "overlap leg $G8K_INDEX: $G8K_JOB_ID key=$KEY tag=$TAG cell=$CELL thinking=$G8K_THINKING sampling=$G8K_SAMPLING_MODE cap=$RUNG out=$G8K_OUT_REL placement=$G8K_CLUSTER_CONSTRAINT"
  if [ "$G8K_CLUSTER_CONSTRAINT" = primary ] && [ "${CLUSTER:-}" != "$G8K_PRIMARY_CLUSTER" ]; then
    f22_fatal "PLACEMENT: $G8K_JOB_ID carries cluster_constraint primary but this job runs on cluster=${CLUSTER:-unknown} — resubmit it on $G8K_PRIMARY_CLUSTER"
  fi
}

g8k_external_mirror_check(){   # an EXTERNAL row's object through the DRIVER's OWN provenance mirror, BEFORE the standup. The
                               # driver (served_capability_run._mirror_provenance -> sweep.resolve_training_mirror) refuses an object
                               # whose fit record establishes no fit kind (a writer outside sweep.CLASS_B_FIT_WRITERS /
                               # SI_PAIR_FIT_WRITERS with no budget schema — the Arditi lane's `scripts/arditi/fit_arditi.py`), or whose
                               # recorded object_sha256 is not these bytes (llama's AlphaSteer view carries the PARENT's record), or a
                               # denylisted sha — the MATH-500 array lost 7 of 8 legs to these ~8 GPU-minutes AFTER their standups.
                               # Same function, same message, no second copy of the rule: a refusal here is
                               # the refusal the driver would raise, minus the standup. The bites positive control (a served-behaviour
                               # fact) has no CPU precheck. No-op on a campaign row (the roster-resolved bank is the driver's own).
  [ -n "${G8K_OBJECT_SOURCE:-}" ] || return 0
  local obj out
  obj="${TT_TWO_SI_DEFENSES_DIR:?TT_TWO_SI_DEFENSES_DIR (g8k_bank_view runs first)}/${G8K_BANK_CELL_DIR:?G8K_BANK_CELL_DIR}/object.pt"
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
  ) || f22_fatal "external object $CELL ($obj): the driver's provenance mirror REFUSES it — the leg would die at the driver door after its standup; fix the record or the allow-list, never the check (E2-12) — $out"
  f22_log "$out"
}

g8k_resolve_config(){   # the OVERLAY config + manifest for TAG (make_gsm8k_overlays.py; never the campaign originals), the belts.
                        # A STEERING row (G8K_OBJECT_SOURCE=external, plan P8) serves ANOTHER LANE's bank, whose view holds exactly
                        # ONE object, so it reads its OWN overlay pair (`explicit_cells: [none, <cell>]`): the standup's select_cells
                        # admits only a cell the config enumerates, and the explicit-cell registry validates every listed cell against
                        # the bank the row binds — the TAG overlay's cells would refuse on objects that bank does not carry.
  if [ "${G8K_OBJECT_SOURCE:-}" = itdoc ] || [ -n "${G8K_OWN_OVERLAY:-}" ]; then
    # E3: an it-doc row — and EVERY row of a `row_overlays` matrix (the instruct-direct wave: the served batch-8 constitution
    # cell beside the batch-64 control + difference) — serves through its OWN pair cut from its TAG's campaign config, explicit_cells
    # [none, <cell>]: a second matrix cannot own the published matrix's TAG overlays
    CFG="configs/sdfbase/gsm8k/sdfbase_full22_${TAG}.gsm8k_${CELLSLUG}.yaml"; MANIFEST="configs/capeval/manifests/petri_full22_${TAG}.gsm8k_${CELLSLUG}.yaml"
  elif [ -n "${G8K_OBJECT_SOURCE:-}" ]; then
    CFG="configs/sdfbase/gsm8k/sdfbase_full22_${KEY}.gsm8k_${CELLSLUG}.yaml"; MANIFEST="configs/capeval/manifests/petri_full22_${KEY}.gsm8k_${CELLSLUG}.yaml"
  else
    CFG="configs/sdfbase/gsm8k/sdfbase_full22_${TAG}.gsm8k.yaml"; MANIFEST="configs/capeval/manifests/petri_full22_${TAG}.gsm8k.yaml"
  fi
  [ -f "$CFG" ] && [ -f "$MANIFEST" ] || f22_fatal "missing overlay $CFG or $MANIFEST for TAG $TAG (scripts/sdfbase/gsm8k/make_gsm8k_overlays.py)"
  case "$KEY" in qwen35_4b|glm46vflash)   # the text-only pin, the campaign's job-side belt
    grep -q -- '--limit-mm-per-prompt' "$MANIFEST" || f22_fatal "$MANIFEST serves multimodal $KEY without --limit-mm-per-prompt (text-only pin)";;
  esac
  GPU_MEM_UTIL="${GPU_MEM_UTIL:-$G8K_GPU_MEM_UTIL}"
}

g8k_out_root(){   # ONE durable out root per TAG x cell x rung: the 8,192 store, or the rung store BESIDE it
  OUT_ROOT="$WAVE_ROOT/$G8K_OUT_REL"; mkdir -p "$OUT_ROOT"
  SOURCE_OUT_ROOT="$WAVE_ROOT/$G8K_SOURCE_OUT_REL"
  BUDGET_SRC="$WAVE_ROOT/budgets/$KEY/$G8K_BUDGET_CLASS/$RUNG/$G8K_BUDGET_FILENAME"
  RUNG_SUBSET="$WAVE_ROOT/$G8K_RUNG_DIR_REL/$TAG/$CELLSLUG/$G8K_RUNG_SUBSET_FILENAME"
  LADDER_STATE="$OUT_ROOT/$G8K_LADDER_STATE_FILENAME"
}

g8k_rung_check(){   # a 16,384 row runs ONLY over the subset ladder_plan.py wrote for THIS cell, bound to its source store
  [ "$RUNG" = "$G8K_RUNG_CAP" ] || return 0
  [ -f "$RUNG_SUBSET" ] || f22_fatal "no rung subset at $RUNG_SUBSET — the ladder planner (ladder_plan.sbatch) writes it for a cell over the trigger (1 % at cap since 2026-09-12) with eligible items; a rung is never run without it"
  # the subset's `task` is the producer's ROW task (contracts.GSM8K_STANDARD_TASK = inspect/gsm8k_paired200, mirrored as
  # G8K_ROW_TASK), never the unprefixed CLI spelling the cap flags carry (review B-P1.2)
  "$(g8k_py)" - "$RUNG_SUBSET" "$G8K_ROW_TASK" "$G8K_RUNG_CAP" "$G8K_CAP" "$SOURCE_OUT_ROOT" <<'PY' || f22_fatal "rung subset $RUNG_SUBSET does not bind to this row — refusing"
import json, sys
d = json.load(open(sys.argv[1])); task, cap, source_cap, source = sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
problems = []
if d.get("task") != task: problems.append(f"task {d.get('task')!r} != {task!r}")
if int(d.get("cap") or 0) != cap: problems.append(f"cap {d.get('cap')!r} != {cap}")
if int(d.get("source_cap") or 0) != source_cap: problems.append(f"source_cap {d.get('source_cap')!r} != {source_cap}")
if str(d.get("source_out_root") or "").rstrip("/") != source.rstrip("/"): problems.append(f"source_out_root {d.get('source_out_root')!r} != {source!r}")
ids = d.get("ids") or []
if not ids or len(set(ids)) != len(ids): problems.append(f"ids: {len(ids)} listed, {len(set(ids))} distinct (need a non-empty set)")
if problems: print("; ".join(problems)); sys.exit(1)
print(f"rung subset OK: {len(ids)} ids, source {source}")
PY
}

g8k_budget_make_if_absent(){   # the FRONT DOOR's budget: made IN-JOB at THIS tree when no pre-launch artifact exists (paper-companion-repo
                               # stage E, 2026-09-30). The paper's rows had their (key, class, rung) budgets made by hand before launch
                               # (README: tinycap_oos_prompt_budget.py --task gsm8k_paired200 …); a front-door model has no such step, so
                               # its three u0 rows died at `no prompt-budget artifact`. The smoke has always made its own G-FIT in-job
                               # (g8k_smoke_budget); the production row does the same when — and only when — the artifact is absent or
                               # not this wave's (sidecar_currency): CPU, the client venv, the pinned tokenizer OFFLINE, --tree-sha = the
                               # wave sha so the sidecar's pin holds.
                               # ONE budget pair per (key, class, rung) is SHARED by every runnable row of that class (the published matrix
                               # has 12 same-class rows), so the whole check -> archive -> re-make -> validate path runs under an exclusive
                               # flock on <pair>.lock beside the pair: a second job waits and then reads the fresh,
                               # complete pair; archival names are unique (utc + nanoseconds + job id) so no saved pair is overwritten.
  g8k_core
  mkdir -p "$(dirname "$BUDGET_SRC")"
  _g8k_budget_make_locked 9>"$BUDGET_SRC.lock"
}

_g8k_budget_make_locked(){   # runs with fd 9 open on <pair>.lock; the lock is released when this function returns (fd closed)
  flock -x 9 || f22_fatal "cannot lock $BUDGET_SRC.lock (flock failed) — the shared budget pair must be made under exclusion"
  if [ -f "$BUDGET_SRC" ] && [ -f "$BUDGET_SRC.fit.json" ]; then
    # an existing pair is honoured ONLY when it is THIS wave's (tinycap_oos_prompt_budget.sidecar_currency — the same predicate the
    # artifact check refuses by): a sidecar pinned to another tree, or fitted under a geometry generation since re-measured, is moved
    # aside by name and re-made here — pass 2's sidecars (tree 7065859…) refused every pass-3 gsm8k row in its first minute (2026-09-30)
    local why; why=$( cd "$REPO_DIR" && PYTHONPATH="$REPO_DIR/src:$REPO_DIR/scripts${PYTHONPATH:+:$PYTHONPATH}" "$(g8k_py)" - "$BUDGET_SRC.fit.json" "$WAVE_SHA" "$KEY" <<'PY'
import json, sys
import tinycap_oos_prompt_budget as B
side = json.load(open(sys.argv[1]))
try:
    gen = B.site_geometry_generation(sys.argv[3])
except B.SidecarCurrencyUndecidable as exc:      # a REVOKED / unusable site record: refuse by name, never "no record -> current on the tree"
    print("UNDECIDABLE " + str(exc)); sys.exit(0)
ok, why = B.sidecar_currency(side, wave_sha=sys.argv[2], current_generation=gen)
print(("CURRENT " if ok else "STALE ") + why)
PY
    ) || f22_fatal "the fit sidecar $BUDGET_SRC.fit.json could not be read for its currency (above)"
    case "$why" in
      CURRENT*) return 0;;
      STALE*) local stamp; stamp="superseded.$(date -u +%Y%m%dT%H%M%S.%NZ).job${SLURM_JOB_ID:-manual}"
              f22_log "prompt-budget sidecar at $BUDGET_SRC.fit.json is not this wave's — ${why#STALE }; moving the pair aside (.$stamp) and re-making it at $WAVE_SHA"
              mv -n "$BUDGET_SRC.fit.json" "$BUDGET_SRC.fit.json.$stamp"; mv -n "$BUDGET_SRC" "$BUDGET_SRC.$stamp";;
      UNDECIDABLE*) f22_fatal "prompt-budget sidecar currency for $BUDGET_SRC.fit.json cannot be decided — ${why#UNDECIDABLE } (the row waits for the re-probe's verified staging)";;
      *) f22_fatal "unexpected currency verdict for $BUDGET_SRC.fit.json: $why";;
    esac
  fi
  # THE PROMPT CLASS: a prompted row's requests carry ITS SI as the system turn (budget_class = the cell
  # itself, g8k_cells.budget_class), so the producer must fold that SI into every rendered prompt (--system-prompt-cell), or it budgets
  # bare prompts thousands of tokens short and the G-FIT is false; `none` and an object cell carry no system turn (no flag)
  local sp=(); case "$CELL" in prompted__*) sp=(--system-prompt-cell "$CELL");; esac
  f22_log "no prompt-budget artifact at $BUDGET_SRC — making it in-job at $WAVE_SHA (cap $RUNG, $G8K_GEOMETRY_EVAL, arm $G8K_THINKING, prompt class ${sp[1]:-none})"
  mkdir -p "$(dirname "$BUDGET_SRC")" "$SERVE_DIR/$G8K_SERVE_SUBDIR"
  ( cd "$REPO_DIR" && CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONPATH="$REPO_DIR/src:$REPO_DIR/scripts" "$CPY" scripts/tinycap_oos_prompt_budget.py \
      --task "$G8K_INSPECT_TASK" --arm "$G8K_THINKING" --cap "$RUNG" --geometry-eval "$G8K_GEOMETRY_EVAL" ${sp[@]+"${sp[@]}"} \
      --config "$CFG" --model-key "$KEY" --tree-sha "$WAVE_SHA" --offline --out "$BUDGET_SRC" ) > "$SERVE_DIR/$G8K_SERVE_SUBDIR/budget.log" 2>&1 \
    || f22_fatal "the prompt budget at cap $RUNG against $G8K_GEOMETRY_EVAL could not be made or does not fit (see $SERVE_DIR/$G8K_SERVE_SUBDIR/budget.log)"
  # the complete new pair must exist before the lock is released (a waiter then reads a whole pair, never a half one)
  [ -s "$BUDGET_SRC" ] && [ -s "$BUDGET_SRC.fit.json" ] || f22_fatal "the producer left an incomplete budget pair at $BUDGET_SRC (artifact or sidecar missing) — nothing published"
}

g8k_budget_artifact(){   # the pre-launch CPU budget artifact for (key, class, rung) + its G-FIT sidecar, into the out root
  g8k_core
  st_budget_artifact "$BUDGET_SRC" "$OUT_ROOT/$G8K_BUDGET_FILENAME" \
    "run scripts/tinycap_oos_prompt_budget.py --task gsm8k_paired200 --arm $G8K_THINKING --cap $RUNG --geometry-eval $G8K_GEOMETRY_EVAL for ($KEY, $G8K_BUDGET_CLASS) at the pin (plan §5 G-FIT)"
  # the FIT sidecar of that artifact: fits, at THIS rung's cap, against THIS row's entry, made at THIS tree (the pin) — a
  # budget measured at another cap, entry or tree is a refusal, never a warning
  [ -f "$BUDGET_SRC.fit.json" ] || f22_fatal "no fit sidecar at $BUDGET_SRC.fit.json — G-FIT runs the budget WITH --geometry-eval at the pin (plan §5)"
  # … and made for THIS row's PROMPT CLASS: a prompted row's sidecar must name its cell (and the sha16 of the
  # SI it folded in); a `none` / object row's must name none — a budget of bare prompts never admits a prompted row
  # (the SI's sha16 is RECOMPUTED here from THIS checkout's text — sisteer.eval.jailbreak.resolve_instruction at the pin, the producer's own
  # resolver — so a same-class sidecar made from other SI bytes refuses by name, never by the field's mere presence)
  ( cd "$REPO_DIR" && PYTHONPATH="$REPO_DIR/src:$REPO_DIR/scripts${PYTHONPATH:+:$PYTHONPATH}" "$(g8k_py)" - "$BUDGET_SRC.fit.json" "$RUNG" "$G8K_GEOMETRY_EVAL" "$WAVE_SHA" "$G8K_THINKING" "$CELL" "$KEY" <<'PY' ) || f22_fatal "fit sidecar $BUDGET_SRC.fit.json does not admit this row — refusing"
import hashlib, json, sys
d = json.load(open(sys.argv[1])); cap, ev, sha, arm, cell, key = int(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], sys.argv[7]
problems = []
if d.get("fits") is not True: problems.append(f"fits={d.get('fits')!r}: {d.get('why_not')}")
if int(d.get("cap") or 0) != cap: problems.append(f"cap {d.get('cap')!r} != {cap}")
if d.get("geometry_eval") != ev: problems.append(f"geometry_eval {d.get('geometry_eval')!r} != {ev!r}")
if d.get("arm") != arm: problems.append(f"arm {d.get('arm')!r} != {arm!r}")
import tinycap_oos_prompt_budget as B
try:
    gen = B.site_geometry_generation(key)
except B.SidecarCurrencyUndecidable as exc:
    problems.append(str(exc)); gen = None; ok = False
else:
    ok, why = B.sidecar_currency(d, wave_sha=sha, current_generation=gen)
    if not ok: problems.append(f"{why} — the pin: re-run G-FIT at this tree / geometry")
want_cell = cell if cell.startswith("prompted__") else None
if d.get("system_prompt_cell") != want_cell: problems.append(f"system_prompt_cell {d.get('system_prompt_cell')!r} != this row's prompt class {want_cell!r} (a budget of the wrong prompt class: re-make it for {cell})")
if want_cell and not d.get("system_prompt_sha16"): problems.append("a prompted row's sidecar records no system_prompt_sha16 — the SI was not folded into the budget")
if want_cell and d.get("system_prompt_sha16"):
    from sisteer.eval.jailbreak import resolve_instruction          # the producer's own resolver, at THIS checkout (the pin)
    want_sha = hashlib.sha256(str(resolve_instruction(want_cell[len("prompted__"):]).text).encode("utf-8")).hexdigest()[:16]
    if d.get("system_prompt_sha16") != want_sha: problems.append(f"system_prompt_sha16 {d.get('system_prompt_sha16')!r} != the pinned checkout's SI text {want_sha!r} for {want_cell} (a budget of other SI bytes: re-make it at {sha[:12]})")
if problems: print("; ".join(problems)); sys.exit(1)
print(f"fit OK: prompt max {d.get('prompt_max')} + cap {cap} <= {d.get('window')} ({ev}), margin {d.get('margin')}, prompt class {want_cell or 'none'}, tree {sha[:12]}")
PY
}

g8k_standup_extra(){   # -> F22_STANDUP_EXTRA: EVERY job names its own cell by the ONE standup spelling the matrix row carries
                       # (G8K_STANDUP_CELL: <cell>@thinking iff DECLARED — the standup derives the ON header; the proxy requires
                       # petri_thinking_on on every request — else the plain cell; contracts.cell_spellings at generation time);
                       # the digest iff FITTED, keyed on THAT SAME spelling (the standup's parser wants the exact --cell spelling,
                       # variant included); the reasoning parser on the thinking-ON arm only (gemma4).
                       # --enable-log-requests (an opt-in forwarded ONLY through this array — never hardcoded on the
                       # standup's argv): vLLM logs the sampler the ENGINE resolved per request (params only; the prompt is a
                       # DEBUG record and is not logged), which g8k_record_resolved_sampler reads back after the driver
  f22_standup_extra_reset --upstream-timeout-s "$G8K_UPSTREAM_TIMEOUT_S" --lens-salt-policy "$G8K_LENS_SALT_POLICY" --enable-log-requests --cell "$G8K_STANDUP_CELL"   # on top of the inherited hook; idempotent per call (common.sh, 2026-09-16)
  if [ "$FITTED" = 1 ]; then F22_STANDUP_EXTRA+=(--expect-object-sha256 "$G8K_STANDUP_CELL=$PIN"); fi
  # qwen-base-seed-replicates (decisions O12a; g8k_cells.py STANDUP_COMPANION_FIELDS): the row's STANDUP COMPANIONS — a second `--cell` per
  # companion, each pinned `<cell>=<sha>` (mt_serving_up's --cell is repeatable; select_cells admits a cell the row's own overlay pair
  # enumerates) — so every standup of the wave registers the same rank-4 LoRA beside the row's own cell and the native composer fits ONE
  # --max-lora-rank across them (the GSM8K reader compares that flag between a contrast's two legs). The driver's --cell stays the row's.
  # Both names are EMPTY on every other matrix's rows (g8k_cells.py row_sh), so nothing is appended there: argv byte-identical.
  local c kv companion_cells=() companion_pins=()
  read -r -a companion_cells <<< "${G8K_STANDUP_COMPANION_CELLS:-}"
  read -r -a companion_pins <<< "${G8K_STANDUP_COMPANION_PINS:-}"
  [ "${#companion_cells[@]}" -eq "${#companion_pins[@]}" ] || f22_fatal "row $G8K_JOB_ID: ${#companion_cells[@]} standup companion cell(s) but ${#companion_pins[@]} pin(s) — every companion is pinned"
  for c in ${companion_cells[@]+"${companion_cells[@]}"}; do
    [ -n "$c" ] && [ "$c" != "$G8K_STANDUP_CELL" ] && [ "$c" != none ] || f22_fatal "row $G8K_JOB_ID: standup companion '$c' is empty, the row's own cell or none — refused by name"
    F22_STANDUP_EXTRA+=(--cell "$c")
  done
  for kv in ${companion_pins[@]+"${companion_pins[@]}"}; do
    case "$kv" in *=????????????????????????????????????????????????????????????????) ;; *) f22_fatal "row $G8K_JOB_ID: standup companion pin '$kv' is not <cell>=<64-hex sha256>";; esac
    F22_STANDUP_EXTRA+=(--expect-object-sha256 "$kv")
  done
  if [ -n "$G8K_REASONING_PARSER" ] && [ "$G8K_THINKING" = on ]; then F22_STANDUP_EXTRA+=(--reasoning-parser "$G8K_REASONING_PARSER"); fi
}

g8k_check_endpoint(){   # after f22_standup: the timeout, the budget's template, the FRESH table's stamp == the row's expected positions
                        # — looked up under the spelling the standup EXPORTED: a declared cell's lens_served_positions key is the
                        # variant's harness id (si_…@thinking), the plain harness cell is absent from that table (review B-P1.1)
  g8k_core
  st_check_endpoint "$SERVE_DIR/$G8K_SERVE_SUBDIR/endpoint.json" "$G8K_UPSTREAM_TIMEOUT_S" "$BUDGET_TEMPLATE_HASH" "$TABLE" \
    "$G8K_STANDUP_HARNESS_CELL" "$G8K_SERVED_POSITIONS_EXPECTED"
}

g8k_attempt_begin(){   # the attempt seam: mint ATTEMPT_ID in the store's ladder_state.json with this job's facts
  g8k_core
  local facts
  facts=$("$(g8k_py)" -c 'import json,sys; print(json.dumps(dict(zip(sys.argv[1::2], sys.argv[2::2]))))' \
    kind generation job_id "${SLURM_JOB_ID:-manual}" job_name "${SLURM_JOB_NAME:-manual}" array_task_id "${SLURM_ARRAY_TASK_ID:-}" \
    node "$(hostname)" cluster "${CLUSTER:-}" sha "$WAVE_SHA" rung "$RUNG" tag "$TAG" cell "$CELL" key "$KEY" view "$VIEW" job_row_id "$G8K_JOB_ID")
  st_attempt_begin "$LADDER_STATE" "$facts"
  f22_log "attempt $ATTEMPT_ID recorded in $LADDER_STATE"
}

g8k_attempt_end(){   # $1 = rc, $2 = the RESULT word (the cleanup trap calls it at ANY step: a no-op before an attempt exists)
  [ -n "${ATTEMPT_ID:-}" ] && [ -n "${LADDER_STATE:-}" ] || return 0
  g8k_core
  st_attempt_end "$LADDER_STATE" "$1" "$2" "${OUT_ROOT:-}"
}

g8k_ledger_up(){   # the USAGE LEDGER in front of the proxy, stamping this attempt on every row -> LEDGER_URL LEDGER_PID
  g8k_core
  st_ledger_up "$G8K_SERVE_SUBDIR" "$G8K_LEDGER_WAIT_S" "$G8K_LEDGER_FILENAME" "$G8K_LEDGER_ENDPOINT_FILENAME" --attempt-id "$ATTEMPT_ID" \
    || { G8K_RESULT=LEDGER_FAILED; return 1; }
}

g8k_ledger_down(){   # a no-op before any ledger (the cleanup trap calls it at ANY step)
  [ -n "${LEDGER_PID:-}" ] || return 0
  g8k_core
  st_ledger_down
}

g8k_proxy_token(){
  g8k_core
  st_proxy_token "$SERVE_DIR/$G8K_SERVE_SUBDIR/endpoint.json"
}

g8k_driver_argv(){   # -> DRV: the COMPLETE driver invocation from the row — ONE function, read by the job and
                     # by the dry run alike: the core's head (--controls gsm8k --thinking <arm> --sampling <mode> --samples 1),
                     # the cap as STATEMENT (--max-tokens-by-task) AND, on the thinking-ON arm and the no-switch arm alike, as
                     # the bounded SETTER (--on-cap-by-task; under `not_available` the setter IS the cap —
                     # the OFF overlap leg's 512 is the OFF table's: statement == resolution, no setter), the core's tail, the
                     # attempt id, the row's VIEW (--bank-view: the four provenance fields on every emitted row), the FITTED
                     # pin, the OOS override ONLY where a tuple authorises it (a no-switch fitted row is IN SUPPORT: no tuple,
                     # no override; llama-paper-rows P7), the rung's item subset
  g8k_core
  st_driver_argv_head "$G8K_CONTROL" "$G8K_THINKING" "$G8K_SAMPLING_MODE" "$G8K_SAMPLES"
  DRV+=(--max-tokens-by-task "$G8K_MAX_TOKENS_BY_TASK")
  case "$G8K_THINKING" in on|not_available) DRV+=(--on-cap-by-task "$G8K_MAX_TOKENS_BY_TASK");; esac
  # the Qwen base seed replicates: the row's OPT-IN Inspect seed policy + its view's base (g8k_cells.py row_sh
  # G8K_INSPECT_SEED_POLICY / G8K_INSPECT_SEED_BASE — set on the seed-replicate matrix's rows only; empty everywhere else, so
  # every other matrix's argv is byte-identical). Both or neither: a policy with no base is refused by name.
  if [ -n "${G8K_INSPECT_SEED_POLICY:-}" ]; then
    [ -n "${G8K_INSPECT_SEED_BASE:-}" ] || f22_fatal "row $G8K_JOB_ID carries inspect_seed_policy $G8K_INSPECT_SEED_POLICY but no inspect_seed_base"
    DRV+=(--inspect-seed-policy "$G8K_INSPECT_SEED_POLICY" --inspect-seed-base "$G8K_INSPECT_SEED_BASE")
  fi
  st_driver_argv_tail "$G8K_IDENTITY_VERSION" "$OUT_ROOT/$G8K_BUDGET_FILENAME" "$TARGET_S" "$SERVE_DIR/$G8K_SERVE_SUBDIR/vllm_server.log"
  DRV+=(--attempt-id "$ATTEMPT_ID" --bank-view "$VIEW")
  if [ "$FITTED" = 1 ]; then
    [ -z "$G8K_TUPLE" ] || DRV+=(--allow-out-of-support-thinking)
    DRV+=(--expect-object-sha256 "$PIN")
    # THE LIVENESS GATE of a STEER row (G8K_OBJECT_SOURCE=external: an alphasteer / arditi object ANOTHER LANE staged;
    # the MATH-500 launcher — m5t_driver_argv — carries the same gate, mirrored here): the driver's positive control rules
    # by the LOGPROB liveness statistic (fidelity.check_logprob_steering_bites on the bites prompts, the scale-0 twin's REPEAT as
    # the MEASURED noise floor) instead of greedy-text divergence, which a small correct edit need not move (array 495677: gemma
    # lambda 0.057 and nemotron lambda 0.049 refused as "inert" at 0/5 while qwen lambda 0.548 passed at 2/5). The driver scopes
    # the rule by the object's KIND (contracts.LOGPROB_LIVENESS_KINDS), reports the text bites beside it and records that logprob
    # distance is not a dose axis. A published / ext row (the campaign's own bank, or a prompted route) passes nothing here: its
    # argv is byte-identical to before (the sbatch test pins both). The value is g8k_cells.LENS_LIVENESS_STEER.
    # An it-doc row (G8K_OBJECT_SOURCE=itdoc: the instruct-direct lane's own LoRA re-fit, not an alphasteer / static_r object) keeps
    # the driver's default text gate, as its wave ran.
    [ "${G8K_OBJECT_SOURCE:-}" != external ] || DRV+=(--lens-liveness logprob)
  fi
  if [ "$RUNG" = "$G8K_RUNG_CAP" ]; then DRV+=(--item-subset "$RUNG_SUBSET"); fi
}

g8k_recorder_argv(){   # -> RSA: scripts/sdfbase/record_resolved_sampler.py over THIS job's server log, into THIS row's out root
                       # (beside the store: the pull carries it), the row's rung as the CLIENT cap, the row's PINNED sampling row as
                       # the reference — ONE --expect k=v per key of G8K_SAMPLING_PARAMS (the numeric projection; the recorder refuses a
                       # key outside its SAMPLER_KEYS by name) — and the PROBE class BY KIND (the per-kind rule, mirrored exactly from
                       # the MATH-500 launcher's m5t_recorder_argv): `--probe-expect kinds` with TWO tag sources, one per EMITTER:
                       # the DRIVER's request evidence beside the store (`X-Tinycap-Task: probe/<kind>` + max_tokens per probe attempt,
                       # contracts.request_evidence_record) and the STANDUP's own load-check sidecar in the serving dir (the load check
                       # fires its probes at the RUN's cap BEFORE the driver exists — on this lane the run's cap IS the client cap, so
                       # without the second source those requests are untagged and indistinguishable from client traffic). Two emitters
                       # is not double-counting: disjoint request sets ("never the evidence file AND the ledger for the same requests"
                       # is the rule). Each kind is compared against its own pinned reference (render -> the card row = --expect; bites,
                       # lens_on_bite, load_check -> greedy), the shape cross-checked against the recorder's kind table; an untagged or
                       # mis-shaped probe is UNCLASSIFIED (a finding, never widened). NO --probe-cap: under kinds the recorder's default
                       # is the kind table's caps (g8k_cells.RECORDER_PROBE_CAPS mirrors them for the client-cap collision check; 8,192
                       # / 16,384 / the smoke's 1,536 collide with none). `--require-wired` (a strict SUBSET): this launcher
                       # DECLARES the opt-in (--enable-log-requests on every row) and reads AFTER the leg, so a record proving the
                       # instrument was never wired — UNREADABLE (exit 3) or FLAG_SET_NO_REQUESTS_YET (exit 4) — is a silent hole, not
                       # a finding about the sampler; every verdict ABOUT the sampler still exits 0. The caller keeps its non-gating
                       # policy and LOGS 3 / 4 by name (g8k_record_resolved_sampler). ONE function, read by the job (the row launcher
                       # and the no-switch smoke alike) and by the dry run.
  local kv
  RSA=("$CPY" "$REPO_DIR/$G8K_RESOLVED_SAMPLER_SCRIPT" --vllm-log "$SERVE_DIR/$G8K_SERVE_SUBDIR/vllm_server.log"
       --out "$OUT_ROOT/$G8K_RESOLVED_SAMPLER_FILENAME" --client-cap "$RUNG"
       --probe-expect "${G8K_RECORDER_PROBE_EXPECT:?G8K_RECORDER_PROBE_EXPECT (g8k_cells.py literals)}"
       --probe-tags "$OUT_ROOT/${G8K_RECORDER_PROBE_TAGS_FILENAME:?G8K_RECORDER_PROBE_TAGS_FILENAME (g8k_cells.py literals)}"
       --probe-tags "$SERVE_DIR/$G8K_SERVE_SUBDIR/${G8K_RECORDER_LOAD_CHECK_TAGS_FILENAME:?G8K_RECORDER_LOAD_CHECK_TAGS_FILENAME (g8k_cells.py literals)}"
       --require-wired)
  while IFS= read -r kv; do [ -z "$kv" ] || RSA+=(--expect "$kv"); done < <("$(g8k_py)" -c 'import json, sys
for k, v in sorted(json.loads(sys.argv[1]).items()): print(f"{k}={v}")' "$G8K_SAMPLING_PARAMS")
}

g8k_record_resolved_sampler(){   # after the driver, BEFORE the standup stops (the server log is complete; the standup is still
                                 # the job's): the attestation is RECORDED, never the job's result — a recorder failure or a
                                 # DIVERGED verdict is logged and rides the record; the analysis files it as a finding (the
                                 # recorder's own exit policy: 0 for every recorded verdict, 2 for a refused argument; under
                                 # --require-wired 3 / 4 ONLY when the record proves the declared instrument was never wired —
                                 # logged BY NAME here so the hole is visible in the .out, still never the leg's result)
  local rc
  g8k_recorder_argv
  f22_log "resolved sampler: ${RSA[*]}"
  ( cd "$REPO_DIR" && env CUDA_VISIBLE_DEVICES="" PYTHONPATH="$REPO_DIR/src:$REPO_DIR/scripts" "${RSA[@]}" ) \
    > "$SERVE_DIR/$G8K_SERVE_SUBDIR/resolved_sampler.log" 2>&1; rc=$?
  case "$rc" in
    0) ;;
    3|4) f22_log "resolved sampler recorder rc=$rc: INSTRUMENT NOT WIRED (--require-wired: $([ "$rc" = 3 ] && echo UNREADABLE || echo FLAG_SET_NO_REQUESTS_YET) — the row declared --enable-log-requests but the record proves no request was logged; recorded, not the job's result; see $SERVE_DIR/$G8K_SERVE_SUBDIR/resolved_sampler.log)";;
    *) f22_log "resolved sampler recorder rc=$rc (recorded, not the job's result; see $SERVE_DIR/$G8K_SERVE_SUBDIR/resolved_sampler.log)";;
  esac
  [ -f "$OUT_ROOT/$G8K_RESOLVED_SAMPLER_FILENAME" ] && f22_log "resolved sampler: $(head -c 400 "$SERVE_DIR/$G8K_SERVE_SUBDIR/resolved_sampler.log" | head -1)" || true
}

g8k_print_plan(){   # the dry run's verbatim lines for THIS row (g8k_dry_run.py reads them)
  g8k_core
  printf 'ROW: index=%s job_id=%s key=%s view=%s tag=%s cell=%s rung=%s fitted=%s declared=%s route=%s family=%s time=%s placement=%s out_root=%s matrix=%s thinking=%s\n' \
    "$G8K_INDEX" "$G8K_JOB_ID" "$KEY" "$VIEW" "$TAG" "$CELL" "$RUNG" "$FITTED" "$DECLARED" "$G8K_ROUTE_KIND" "$G8K_FAMILY" "$G8K_TIME" "$G8K_CLUSTER_CONSTRAINT" "$OUT_ROOT" "$G8K_MATRIX" "$G8K_THINKING"
  # the E2 row facts (P2 / P8): the wave and the status the batch boundary reads; on an external row the lane's bank view
  # (relative to the shared root, per cluster role), the cell dir and the lane's own view name beside the panel view
  printf 'WAVE: wave=%s status=%s object_source=%s bank_view=%s bank_view_rel_primary=%s bank_view_rel_overflow=%s bank_cell_dir=%s config=%s\n' \
    "${G8K_WAVE:-published}" "$STATUS" "${G8K_OBJECT_SOURCE:-none}" "${G8K_BANK_VIEW:-none}" "${G8K_BANK_VIEW_REL_PRIMARY:-none}" "${G8K_BANK_VIEW_REL_OVERFLOW:-none}" "${G8K_BANK_CELL_DIR:-none}" "$CFG"
  # the serving path the standup will take, as f22_init RESOLVES it (the dry-run branch runs the same f22_select_serving_path over the
  # same declaration): the row's pin (lens on an external-object row; selection=explicit) or the kind-aware default over the declared
  # launch cells (selection=auto) — and the cells declared (F22_LAUNCH_CELLS: `none` + the row's standup harness cell, the standup's cells_requested)
  # …the pin the row declared (g8k_select_serving_path: empty = none) WITH ITS REASON — kind:alphasteer | store:lens | unpinned:no-store |
  # store:native on an external row (the store under the dry run's wave root), unpinned:not-external on every other row
  printf 'SERVING_PATH: path=%s pinned_by_row=%s reason=%s selection=%s\n' "${F22_SERVING_PATH:-unresolved}" "${G8K_SERVING_PATH_PINNED:-none}" "${G8K_SERVING_PATH_REASON:-unresolved}" "${F22_SERVING_PATH_MODE:-unresolved}"
  printf 'LAUNCH_CELLS: source=F22_LAUNCH_CELLS cells=%s\n' "${F22_LAUNCH_CELLS[*]:-undeclared}"
  # on an EXTERNAL (steer) row, the ladder's selection outcome + the grid-limited flag beside the verbatim selection label;
  # an it-doc row serves its own lane's re-fit and carries no ladder outcome, so it prints none
  [ "${G8K_OBJECT_SOURCE:-}" != external ] || printf 'SELECTION: outcome=%s grid_limited=%s label=%s\n' "${G8K_SELECTION_OUTCOME:-none}" "${G8K_GRID_LIMITED:-0}" "$G8K_SELECTION_LABEL"
  # E3: the it-doc row facts the bank door binds (the record, the lane, the ruled staged id, the object kind) and the qualification
  [ "${G8K_OBJECT_SOURCE:-}" != itdoc ] || printf 'ITDOC: lane=%s staged_id=%s object_kind=%s admission=%s qualification=%s\n' \
    "$G8K_LANE" "$G8K_STAGED_ID" "$G8K_OBJECT_KIND" "$G8K_ITDOC_ADMISSION" "$G8K_QUALIFICATION"
  # the GENERATION of the object this row serves (batch8 = the instruct-direct lane's first fit as served; batch64 = the re-fit
  # lane) — a wave that mixes them names the mix on every row, in the dry run and in the job's .out (the `matrix row` log line above)
  [ -z "${G8K_OBJECT_GENERATION:-}" ] || printf 'OBJECT_GENERATION: %s staged_id=%s\n' "$G8K_OBJECT_GENERATION" "${G8K_STAGED_ID:-none}"
  printf 'SAMPLING: mode=%s params=%s\nSAMPLING_SOURCE: %s\n' "$G8K_SAMPLING_MODE" "$G8K_SAMPLING_PARAMS" "$G8K_SAMPLING_SOURCE"
  # qwen-base-seed-replicates: the OPT-IN Inspect seed policy of a seed-replicate row (nothing printed on every other matrix)
  [ -z "${G8K_INSPECT_SEED_POLICY:-}" ] || printf 'INSPECT_SEED: policy=%s base=%s\n' "$G8K_INSPECT_SEED_POLICY" "$G8K_INSPECT_SEED_BASE"
  # qwen-base-seed-replicates (O12a): the STANDUP COMPANIONS the standup registers beside the row's own cell (nothing printed without one)
  [ -z "${G8K_STANDUP_COMPANION_CELLS:-}" ] || printf 'STANDUP_COMPANIONS: cells=%s pins=%s\n' "$G8K_STANDUP_COMPANION_CELLS" "$G8K_STANDUP_COMPANION_PINS"
  printf 'GEOMETRY: %s %s x %s cap %s thinking %s\n' "$G8K_GEOMETRY_EVAL" "$G8K_GEOMETRY_L" "$TARGET_S" "$RUNG" "$G8K_THINKING"
  # qwen-seedrep-native-battery: the matrix-declared geometry class the entry above was selected by (nothing printed on every other matrix)
  [ -z "${G8K_GEOMETRY_CLASS:-}" ] || printf 'GEOMETRY_CLASS: %s (one S x L per leg, every arm of the matrix)\n' "$G8K_GEOMETRY_CLASS"
  printf 'BANK: %s pin=%s\n' "$TT_TWO_SI_DEFENSES_DIR/$CELL" "${PIN:-none}"
  printf 'ATTEMPT: %s (dry: %s not written)\n' "$ATTEMPT_ID" "$LADDER_STATE"
  g8k_standup_extra
  st_print_argv STANDUP "${F22_STANDUP_EXTRA[@]}"
  st_print_ledger_line "$OUT_ROOT/$G8K_LEDGER_FILENAME" "$(dirname "$TABLE")/$G8K_LEDGER_ENDPOINT_FILENAME" --attempt-id "$ATTEMPT_ID"
  g8k_driver_argv
  st_print_argv DRIVER "${DRV[@]}"
  st_print_driver_env "$G8K_INSPECT_BASE_URL_ENV" "$LEDGER_URL" "$G8K_INSPECT_API_KEY_ENV"
  g8k_recorder_argv
  st_print_argv RESOLVED_SAMPLER "${RSA[@]}"
  if [ "$FITTED" = 1 ] && [ -n "$G8K_TUPLE" ]; then printf 'OUT OF SUPPORT: tuple=%s pin=%s served_positions_expected=%s declared=%s stamps=(%s)\n' "$G8K_TUPLE" "$PIN" "$G8K_SERVED_POSITIONS_EXPECTED" "$DECLARED" "$G8K_EXPECTED_STAMPS"
  elif [ "$FITTED" = 1 ]; then printf 'IN SUPPORT: route=%s pin=%s thinking=%s served_positions_expected=%s stamps=(%s)\n' "$G8K_ROUTE_KIND" "$PIN" "$G8K_THINKING" "$G8K_SERVED_POSITIONS_EXPECTED" "$G8K_EXPECTED_STAMPS"
  else printf 'IN SUPPORT: route=%s stamps=(%s)\n' "$G8K_ROUTE_KIND" "$G8K_EXPECTED_STAMPS"; fi
  [ -z "$G8K_EXPECTED_REFUSAL" ] || printf 'EXPECTED_REFUSAL: %s\n' "$G8K_EXPECTED_REFUSAL"
  [ "$RUNG" != "$G8K_RUNG_CAP" ] || printf 'RUNG_SUBSET: %s (source %s)\n' "$RUNG_SUBSET" "$SOURCE_OUT_ROOT"
}

# --- the NO-SWITCH diagnostic smoke: smoke_not_available.sbatch ----------------
# The certification of a switch-less family's render fact, on the MATH-500 smoke's shape (smoke_thinking.sbatch): the matrix's
# `none` row on the BASE route, G8K_SMOKE_N paired items at G8K_SMOKE_CAP under the row's `card` sampling row, through the usage
# ledger, every record `diagnostic: true`, NO tinycap store, out root $WAVE_ROOT/$G8K_SMOKE_DIR_REL/<family>/smoke_<job>/ (never
# controls/). The driver cannot run this shape (ON_CAP_BY_TASK_ALLOWED admits 8,192 / 16,384 only; --item-subset is the rung's),
# so the probe (G8K_SMOKE_PROBE_SCRIPT) sends what the driver would send on the base route and certifies each completion through
# contracts.no_switch_render_certified — the seam the driver and the analysis' reader share. The smoke cap plays the rung's part
# for the fit sidecar and the recorder's client cap (RUNG=$SMOKE_CAP), so the production functions are reused unchanged.
g8k_smoke_args(){   # the positional overrides over the run literals (g8k_cells.py SMOKE_N / SMOKE_CAP / SMOKE_CONCURRENCY): ONE source
                    # for the defaults, evaluated AFTER g8k_literals; each must be a positive integer
  SMOKE_N="${SMOKE_N_ARG:-$G8K_SMOKE_N}"; SMOKE_CAP="${SMOKE_CAP_ARG:-$G8K_SMOKE_CAP}"; SMOKE_CONCURRENCY="${SMOKE_CONCURRENCY_ARG:-$G8K_SMOKE_CONCURRENCY}"
  local v
  for v in "$SMOKE_N" "$SMOKE_CAP" "$SMOKE_CONCURRENCY"; do
    case "$v" in ''|*[!0-9]*|0) f22_fatal "SMOKE: n / cap / concurrency must be positive integers, got n=$SMOKE_N cap=$SMOKE_CAP concurrency=$SMOKE_CONCURRENCY";; esac
  done
}

g8k_smoke_check_row(){   # the smoke certifies the NO-SWITCH class on the BASE route: the `none` row of a switch-less family, nothing else
  [ "$G8K_THINKING" = not_available ] \
    || f22_fatal "SMOKE: $G8K_JOB_ID runs thinking=$G8K_THINKING — the diagnostic smoke certifies a switch-less family's render fact (thinking not_available) only; the switched families' smoke is scripts/sdfbase/math500/smoke_thinking.sbatch"
  [ "$CELL" = none ] && [ "$FITTED" = 0 ] && [ "$G8K_ROUTE_KIND" = base ] \
    || f22_fatal "SMOKE: $G8K_JOB_ID is cell=$CELL fitted=$FITTED route=$G8K_ROUTE_KIND — the smoke runs the base route (cell none, no object) only"
}

g8k_smoke_root(){   # -> SMOKE_ROOT OUT_ROOT LADDER_STATE BUDGET_SRC RUNG: the diagnostic root, never controls/; the smoke cap as the rung
  SMOKE_ROOT="$WAVE_ROOT/$G8K_SMOKE_DIR_REL/$G8K_FAMILY/smoke_${SLURM_JOB_ID:-manual}"; OUT_ROOT="$SMOKE_ROOT"
  LADDER_STATE="$OUT_ROOT/$G8K_LADDER_STATE_FILENAME"; BUDGET_SRC="$OUT_ROOT/budget/$G8K_BUDGET_FILENAME"; RUNG="$SMOKE_CAP"
}

g8k_smoke_budget(){   # the smoke's OWN G-FIT at ITS cap, made in-job at THIS tree (CPU, the client venv, the pinned tokenizer OFFLINE)
                      # into $BUDGET_SRC (+ .fit.json against the row's production entry), then held to this run exactly as a
                      # production row's is (g8k_budget_artifact: fits, cap, entry, arm, tree_sha) — the endpoint's chat_template_hash
                      # is then checked against the budget's (g8k_check_endpoint), as production does
  mkdir -p "$(dirname "$BUDGET_SRC")"
  ( cd "$REPO_DIR" && CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONPATH="$REPO_DIR/src:$REPO_DIR/scripts" "$CPY" scripts/tinycap_oos_prompt_budget.py \
      --task "$G8K_INSPECT_TASK" --arm "$G8K_THINKING" --cap "$SMOKE_CAP" --geometry-eval "$G8K_GEOMETRY_EVAL" \
      --config "$CFG" --model-key "$KEY" --tree-sha "$WAVE_SHA" --offline --out "$BUDGET_SRC" ) > "$SERVE_DIR/$G8K_SERVE_SUBDIR/budget.log" 2>&1 \
    || f22_fatal "SMOKE: the prompt budget at cap $SMOKE_CAP against $G8K_GEOMETRY_EVAL could not be made or does not fit (see $SERVE_DIR/$G8K_SERVE_SUBDIR/budget.log)"
  g8k_budget_artifact
}

g8k_smoke_probe_argv(){   # -> SPA: the COMPLETE probe invocation — ONE function, read by the job and by the dry run alike: the family
                          # and the arm from the row (both refused by the probe unless switch-less / not_available), --diagnostic as
                          # the argv's own statement, the row's `card` sampling row and expected stamp triple, the smoke's n / cap /
                          # concurrency, the attempt id the ledger stamps, identity version 2's seed rule, the ledger's task label
  SPA=("$CPY" "$REPO_DIR/$G8K_SMOKE_PROBE_SCRIPT"
       --ledger-url "$LEDGER_URL" --proxy-url "$PROXY_URL" --proxy-token "$PROXY_TOKEN" --routing-table "$TABLE"
       --endpoint-json "$SERVE_DIR/$G8K_SERVE_SUBDIR/endpoint.json" --ledger-path "$OUT_ROOT/$G8K_LEDGER_FILENAME"
       --model-key "$KEY" --family "$G8K_FAMILY" --thinking "$G8K_THINKING" --diagnostic
       --sampling-mode "$G8K_SAMPLING_MODE" --sampling-params "$G8K_SAMPLING_PARAMS" --expect-stamps "$G8K_EXPECTED_STAMPS"
       --n "$SMOKE_N" --cap "$SMOKE_CAP" --concurrency "$SMOKE_CONCURRENCY"
       --attempt-id "$ATTEMPT_ID" --tinycap-identity-version "$G8K_IDENTITY_VERSION" --task-label "$G8K_SMOKE_TASK_LABEL"
       --out "$OUT_ROOT")
}

g8k_smoke_print_plan(){   # the dry run's verbatim lines for the smoke (the test reads them; nothing is served)
  g8k_core
  printf 'SMOKE: job_id=%s key=%s view=%s tag=%s cell=%s family=%s thinking=%s sampling=%s route=%s n=%s cap=%s concurrency=%s diagnostic=true out_root=%s matrix=%s\n' \
    "$G8K_JOB_ID" "$KEY" "$VIEW" "$TAG" "$CELL" "$G8K_FAMILY" "$G8K_THINKING" "$G8K_SAMPLING_MODE" "$G8K_ROUTE_KIND" "$SMOKE_N" "$SMOKE_CAP" "$SMOKE_CONCURRENCY" "$OUT_ROOT" "$G8K_MATRIX"
  printf 'SAMPLING: mode=%s params=%s\n' "$G8K_SAMPLING_MODE" "$G8K_SAMPLING_PARAMS"
  printf 'GEOMETRY: %s %s x %s cap %s thinking %s\n' "$G8K_GEOMETRY_EVAL" "$G8K_GEOMETRY_L" "$TARGET_S" "$SMOKE_CAP" "$G8K_THINKING"
  printf 'BUDGET: %s (arm %s cap %s entry %s tree %s)\n' "$BUDGET_SRC" "$G8K_THINKING" "$SMOKE_CAP" "$G8K_GEOMETRY_EVAL" "$WAVE_SHA"
  printf 'STAMPS: %s\n' "$G8K_EXPECTED_STAMPS"
  g8k_standup_extra
  st_print_argv STANDUP "${F22_STANDUP_EXTRA[@]}"
  st_print_ledger_line "$OUT_ROOT/$G8K_LEDGER_FILENAME" "$(dirname "$TABLE")/$G8K_LEDGER_ENDPOINT_FILENAME" --attempt-id "$ATTEMPT_ID"
  g8k_smoke_probe_argv
  st_print_argv SMOKE_PROBE "${SPA[@]}"
  g8k_recorder_argv
  st_print_argv RESOLVED_SAMPLER "${RSA[@]}"
}
