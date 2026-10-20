#!/bin/bash
# ar_common.sh — the shared body every scripts/arditi/sbatch/*.sbatch sources.
#
# The Arditi strength lane runs INSIDE the full22 wave's machinery but on its OWN root: our
# objects, roster, pins, bank views and run outputs live under `arditi_strength/`, and nothing
# here ever writes under exp/objects/sdfbase/full22/, exp/objects/alphasteer_f22/,
# configs/sdfbase_full22_*, or the wave-wide sentinels $WAVE_ROOT/STOP and attackers/STOP — those
# are shared with other users' live legs. Everything else — the cluster/env block, the flock'd
# node-local venvs, the logging and the figure mirror — is the wave's, SOURCED rather than
# re-implemented, so a fix there reaches us too. This mirrors scripts/alphasteer_f22/sbatch/
# as22_common.sh, the sibling method lane's file, on purpose.
#
# NEVER --export= (a cluster hard rule): every value travels positionally or through these
# functions.
#
# THE ONE THING THIS FILE ADDS OVER THE WAVE'S common.sh IS A GATE:
# `ar_assert_serving_identity <mode>`. This lane adds NO file under
# src/sisteer/capeval/serving/ or src/sisteer/capeval/tiers/petri_runner.py — the Arditi object is
# a `static_r` blob served by the already-registered `_adapt_static_r`, PREFILL-ONLY, and the one
# enabling change was `methods._KIND_LOADERS["static_r"]`, outside the frozen tree — so the
# serving stack MUST be byte-identical to the pinned reference commit ($AR_W8_SHA), and every
# manifest of this lane records that claim as `serving_impl_commit`. The gate makes the claim true
# before anything is served rather than after the numbers exist. There is NO ladder mode here
# (this lane has no patched tree).
#
# TWO MODES SINCE THE ALL-POSITIONS ARM WAS ADDED. PARAMETERISE, do not fork — a mode flag, lens
# byte-identity the DEFAULT, a native branch asserting the standup attestation; a sibling
# ar_common_native.sh would be the per-lane-copy divergence this campaign already has a scar from:
#
#   production  the LENS stack, byte-identical to the pinned reference commit. Today's body,
#               unchanged to the letter, and still the default: every prefill-only rung of every
#               target keeps this gate exactly.
#   native      the all-positions arm (kind `vector_all`, phase `both`) served on the graph-captured
#               NATIVE residual slot. The byte-identity claim is meaningless there -- that arm
#               REQUIRES serving files this tree has and the reference commit did not -- so the
#               claim is replaced, not relaxed, by the standup's own attestation.
#
# AND THE NATIVE CHECK IS SPLIT BY MOMENT, which is a correctness property rather than tidiness.
# The call site is `ar_cell.sbatch`'s line right after `ar_init`, which runs BEFORE
# SERVE_DIR exists, before the `precheck` early exit, and long before any standup. An
# endpoint-reading body there would fail EVERY job and EVERY precheck. So:
#
#   ar_assert_serving_identity <mode>            at the init call site: POLICY AND CONFIG ONLY,
#                                                endpoint-free, so `precheck` still works;
#   ar_assert_serving_evidence <mode> <endpoint> after EACH successful standup (the first and the
#                                                8192-cap second alike): the EVIDENCE, read from the
#                                                server's own records.
#
# Nothing about the production path moves: its policy check is the reference-commit assertion it
# always was, and its evidence check is a no-op because the lens standup's evidence is that same
# commit assertion.

AR_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AR_REPO_ROOT="$(cd "$AR_DIR/../../.." && pwd)"
# shellcheck source=/dev/null
source "$AR_REPO_ROOT/scripts/sdfbase/full22/common.sh" \
  || { echo "FATAL: cannot source $AR_REPO_ROOT/scripts/sdfbase/full22/common.sh" >&2; exit 1; }

#: the wave's serving baseline — the sha this lane's serving stack must EQUAL
AR_W8_SHA=935dfb43a743ec00bdf41971ecd21302aacd5342
#: the two paths whose last-touching commit DEFINES `serving_impl_commit`
AR_SERVING_DIR=src/sisteer/capeval/serving/
AR_PETRI_RUNNER=src/sisteer/capeval/tiers/petri_runner.py
#: the reference serving stack by CONTENT: the git object ids of the two paths above AT $AR_W8_SHA
#: (`git rev-parse <sha>:<path>`). Read only where the clone does not carry that commit itself — a
#: published repository without its development history — so the byte-identity claim is still
#: decided by bytes, never assumed
AR_W8_SERVING_TREE=49561876c3b96a3d5e714c2d1bd1573ca4022b37
AR_W8_PETRI_RUNNER_BLOB=530c3c5764035bd85b7efb6368bc2fdca416174d
#: the value every manifest of this lane records (asserted equal to the roster's, and to the
#: last commit touching the two paths above)
AR_SERVING_IMPL_COMMIT=94e294c445dc5eec6ebe0ce3a3e25553004cd4b0
AR_ROSTER=configs/arditi_strength_roster.yaml
AR_PINS=configs/arditi_strength_pins.json
#: the arms table (arditi_strength.ARMS_TABLE_REL, pinned equal by test): for the raw unit its `arms.arditi_allpos.fits.<target>`
#: block is the APPROVED FIT every pins entry is held to before the pins certify anything (round 3, item 1)
AR_ARMS=configs/sdfbase_full22_arms.yaml

ar_init(){   # $1 = the sha of the checkout this job runs from ($SHARED/sisteer-eval-repo/$1)
  # the wave's init does the load-bearing work: resolve $SHARED / $CLUSTER, source the SHARED
  # hf_env.sh, set NCCL/alloc/UV/tokenizer env, export SISTEER_LENS_STACKED_REDUCTIONS=1, verify
  # the checkout is AT this sha and clean, and cd into it.
  # THE DECLARED CELL SOURCE, following the sibling lanes' convention: as22_common.sh defers
  # here, m5t_common.sh / g8k_common.sh build F22_LAUNCH_CELLS from their row, and
  # tests/test_sdfbase_full22_wave.py holds EVERY f22_init caller to declaring one.
  # f22_select_serving_path — which f22_init calls, and which postdates this lane — FATALs
  # without a cell source, and then picks the serving path from those cells BY KIND.
  # A caller that KNOWS its cells declares them first and its value is honoured here untouched:
  # ar_cell.sbatch does, reading them off the view's dose unit (`ar_r...` -> arditi -> the lens;
  # `ar_c...` -> arditi_allpos -> lora_native). A caller that declared nothing stands up no target,
  # which is the TRUTHFUL declaration for this lane's four non-serving jobs (href, hydrate,
  # verify_served, fit: none contains f22_standup, mt_serving_up or f22_serving_venv). Those four
  # also declare it at their own call sites, deliberately — the statement belongs where a reader of
  # that script is standing, and this is the backstop rather than the record. The log line matters:
  # an empty declaration is MEANINGFUL to the selector, so a job that silently fell into it must be
  # able to say so from its first lines.
  if ! declare -p F22_LAUNCH_CELLS >/dev/null 2>&1; then
    F22_LAUNCH_CELLS=()
    f22_log "arditi: no F22_LAUNCH_CELLS declared by the caller — declaring EMPTY (this job stands up no target). A job that SERVES must declare its cells before ar_init, or the kind-aware default cannot pick its engine"
  fi
  f22_init "${1:?sha}"
  # ...then re-point every wave-shared path at OUR root. WAVE_ROOT is what the wave's own
  # f22_stop_requested reads, so re-pointing it also means our stop sentinel is ours alone and we
  # can never read (or write) the wave's or the AlphaSteer campaign's.
  # THE ROOTS FOLLOW THE DOSE UNIT: a caller that declared AR_UNIT before ar_init
  # (ar_cell.sbatch, from its view's letter) binds THAT unit's roots -- the raw-coefficient unit's
  # are a NEW namespace (exp/objects/arditi_allpos, exp/arditi_allpos, its own roster + pins), so
  # nothing of the prefill-only unit is read or written by a rung of the other. A caller that
  # declared none is a job of the prefill-only unit (href, hydrate, verify_served, fit) and binds
  # exactly the roots this block always bound, byte for byte.
  ar_bind_unit_roots "${AR_UNIT:-$AR_UNIT_PRODUCTION}"
  mkdir -p "$WAVE_ROOT"
  # The ROSTER is needed to resolve anything in this lane, so it belongs here. The PINS do NOT:
  # they bind staged object BYTES and are consumed by ar_bank_view, which is where the check now
  # lives. Requiring them at init made every job of the lane depend on objects existing — which
  # killed the reference-norm measurement (job 90558), a job that touches no object at all and
  # whose whole purpose is to produce the number the objects are later baked from. A precondition
  # belongs to the step that consumes it, not to the lane.
  [ -f "$AR_ROSTER" ] || f22_fatal "missing $AR_ROSTER in $PWD$( [ "$AR_UNIT" = "$AR_UNIT_PRODUCTION" ] || echo " — the $AR_RUN_TAG_PREFIX unit's roster is written by its bake and staging step (Phase B); nothing of that unit is staged yet" )"
  [ -d "$AR_OBJ_ROOT" ] || mkdir -p "$AR_OBJ_ROOT"
  f22_log "arditi: root=$WAVE_ROOT objects=$AR_OBJ_ROOT cluster=$CLUSTER"
}

#: the lane's two serving modes. `production` is the default at every existing call site and is the
#: lens stack pinned byte-identical to the serving commit in AR_W8_SHA; `native` is the
#: all-positions arm on the graph-captured residual slot, whose evidence is the standup's own
#: attestation instead.
AR_MODES=(production native)
#: the unit whose roots a caller that declares no AR_UNIT binds: every prefill-only job.
#: arditi_strength.DOSE_UNIT, pinned equal by test; ar_cell.sbatch declares AR_UNIT from its view.
AR_UNIT_PRODUCTION=residual_norm_fraction
#: what a NATIVE standup must attest, restated here as data so a refusal can name the fact rather
#: than a boolean; each is pinned EQUAL to the module that owns it by tests/test_arditi_allpos_arm.py
#: and re-checked against the module inside every embedded check (a drifted restatement is itself
#: a refusal, never a silently different test). The position mode is MODE_ALL (stage 1, no flag
#: dependence); the source kind is what registry_adapter maps this arm's cell id to.
#:
#: TWO OP NAMES, KEPT APART (shell-half review, Finding 1). The residual slot has a KERNEL, registered
#: once under native_residual.OP_QUALNAME ("vllm.sisteer_residual_slot") -- the launcher records it as
#: residual_slot.expected.op_name and every worker as its report's op_name. Each PAYLOAD carries its
#: own `op`, the arithmetic that kernel applies for THAT adapter -- native_residual.OP_ADD_VECTOR
#: ("add_vector") for this arm's constant add, OP_RELU_GATED_VECTOR for a PSR -- recorded per adapter
#: as expected.adapters[*].op and per loaded slot as worker_reports[*].slots[*].op. The first draft
#: compared the payload op against the kernel name and so refused EVERY healthy standup
#: ("adapter 'arditi_allpos': op 'add_vector' != 'vllm.sisteer_residual_slot'"); the two are now two
#: constants, asserted separately, each against the record that carries it.
AR_NATIVE_SERVING_PATH=lora_native
AR_NATIVE_OP=vllm.sisteer_residual_slot
AR_NATIVE_ADAPTER_OP=add_vector
AR_NATIVE_POSITION_MODE=all
AR_NATIVE_SOURCE_KIND=vector_all
#: the evidence body's record, PERSISTED beside each standup's endpoint.json: before this file existed,
#: F22_NATIVE_EVIDENCE was only logged, and the grader (scripts/arditi/grade_arditi_static.py --serving-evidence) derives
#: a raw-coefficient run's serving metadata from THIS file rather than stamping a lens mode on it. Written on the OK and
#: on the REFUSED outcome alike, so a refused standup leaves its reasons on disk. Pinned equal to the grader's
#: SERVING_EVIDENCE_NAME by test.
AR_NATIVE_SERVING_EVIDENCE_FILE=serving_evidence.json
#: the env the drivers read the standup's record path from (serving_evidence_binding.ENV, pinned equal by test): set by
#: ar_bind_serving_evidence_record after the native evidence check passed, so every served_target.json they write folds the
#: record's sha + path in and the grader FOLLOWS THE STORE to the record (round 3, item 2); unset for production
AR_SERVING_EVIDENCE_ENV=SISTEER_SERVING_EVIDENCE_RECORD

ar_require_mode(){   # $1 = the calling function's name, $2 = its mode argument -> FATAL unless $2 is one of AR_MODES
  case " ${AR_MODES[*]} " in *" ${2:-} "*) ;; *) f22_fatal "$1: mode must be one of ${AR_MODES[*]}, got '${2:-}'";; esac
}

ar_unit_of_view(){   # $1 = view -> `<unit> <cell>`: the dose unit the view's LETTER names, and its cell
  # $AR_REPO_ROOT, not a relative path: this is called BEFORE ar_init, which is what does the cd
  # into the checkout, so a relative sys.path would resolve against the submitter's cwd.
  local py; py="${MPY:-python3}"
  "$py" - "${1:?view}" "$AR_REPO_ROOT" <<'PY'
import sys
sys.path.insert(0, f"{sys.argv[2]}/scripts/arditi")
import arditi_strength as st
unit = st.unit_of_view(sys.argv[1])       # refuses (non-zero, the reason on stderr) a view of no unit
print(unit, st.unit_spec(unit).cell_id)
PY
}

ar_bind_unit_roots(){   # $1 = dose unit -> AR_UNIT AR_CELL AR_RUN_TAG_PREFIX AR_BANK_ROOT AR_BANK_SUBDIR, and the lane roots WAVE WAVE_ROOT AR_OBJ_ROOT AR_ROSTER AR_PINS (needs SHARED)
  # ONE binding of every root this lane re-points, keyed on the unit's record in the ONE table
  # (arditi_strength.unit_spec): the run-tag prefix names the wave and its root under $SHARED/exp,
  # the bank root and subdir name the objects, and the roster + pins are `configs/<prefix>_roster.yaml`
  # / `configs/<prefix>_pins.json` -- the same files the arms table names for each arm (pinned equal
  # by test). For residual_norm_fraction every value is the one this file always hard-coded.
  local unit="${1:?unit}" py; py="${MPY:-python3}"
  read -r AR_UNIT AR_CELL AR_RUN_TAG_PREFIX AR_BANK_ROOT AR_BANK_SUBDIR < <("$py" - "$unit" "$AR_REPO_ROOT" <<'PY'
import sys
sys.path.insert(0, f"{sys.argv[2]}/scripts/arditi")
import arditi_strength as st
spec = st.unit_spec(sys.argv[1])           # refuses an unknown unit (non-zero, the reason on stderr)
print(spec.unit, spec.cell_id, spec.run_tag_prefix, spec.bank_root, spec.bank_subdir)
PY
) || f22_fatal "ar_bind_unit_roots: '$unit' is not a dose unit of scripts/arditi/arditi_strength.py"
  WAVE="$AR_RUN_TAG_PREFIX"
  WAVE_ROOT="${SHARED:?ar_bind_unit_roots needs SHARED}/exp/$AR_RUN_TAG_PREFIX"
  AR_OBJ_ROOT="$SHARED/$AR_BANK_ROOT"
  AR_ROSTER="configs/${AR_RUN_TAG_PREFIX}_roster.yaml"
  AR_PINS="configs/${AR_RUN_TAG_PREFIX}_pins.json"
  f22_log "arditi: unit=$AR_UNIT cell=$AR_CELL wave=$WAVE bank=$AR_OBJ_ROOT/<target>/$AR_BANK_SUBDIR roster=$AR_ROSTER pins=$AR_PINS"
}

ar_launch_cells(){   # $1 = view -> `none <cell>`: the launch's cells, for the kind-aware serving-path default
  # f22_select_serving_path (common.sh, the rule of 2026-09-15) FATALs unless the caller
  # declares a cell source BEFORE f22_init, and picks the serving path from the cells BY KIND. The
  # cell is not a constant of this lane any more -- it is the VIEW's unit's cell (`arditi` for an
  # `ar_r` view, `arditi_allpos` for an `ar_c` one) -- so it is read from the table rather than
  # typed here, and a view of neither unit refuses loudly.
  local unit cell
  read -r unit cell < <(ar_unit_of_view "${1:?view}") \
    || f22_fatal "ar_launch_cells: cannot resolve the dose unit of view '$1' through scripts/arditi/arditi_strength.py"
  [ -n "$cell" ] || f22_fatal "ar_launch_cells: no cell for view '$1'"
  printf 'none\n%s\n' "$cell"
}

ar_assert_serving_identity(){   # $1 = production | native  (ar_cell.sbatch passes "$AR_MODE", the mode of the view's dose unit) -- POLICY AND CONFIG ONLY, endpoint-free
  local mode="${1:-}" head problems=() last
  ar_require_mode ar_assert_serving_identity "$mode"
  if [ "$mode" = native ]; then
    # THE NATIVE POLICY CHECK. It reads NO endpoint, by construction: this runs at the init call
    # site, before SERVE_DIR exists and before the `precheck` early exit, so anything it read would
    # be absent on every precheck and on every job's first seconds. What it CAN check is the launch's
    # own declared intent, which is exactly where a wrong serving path is decided:
    #   * the launch declared cells (F22_LAUNCH_CELLS / F22_LAUNCH_MANIFEST) or an explicit path, so
    #     f22_select_serving_path resolved something rather than FATALing;
    #   * the resolved path is lora_native. `F22_SERVING_PATH=lens` is REFUSED for this arm: the lens
    #     can serve a vector_all blob, so the run would succeed and be the wrong measurement -- the
    #     spec requires the native graph-captured slot in every leg that serves this object;
    #   * SISTEER_LENS_RHO_EMIT unset, as in production.
    # The EVIDENCE that the engine did what the policy asked is ar_assert_serving_evidence, after the
    # standup, where the endpoint exists.
    if [ -z "${F22_SERVING_PATH:-}" ]; then
      problems+=("no serving path is resolved (F22_SERVING_PATH empty; mode=${F22_SERVING_PATH_MODE:-unset}) — declare the launch's cells before ar_init (F22_LAUNCH_CELLS=(none <cell>)) so the kind-aware default can select")
    elif [ "$F22_SERVING_PATH" != "$AR_NATIVE_SERVING_PATH" ]; then
      problems+=("F22_SERVING_PATH='$F22_SERVING_PATH' != $AR_NATIVE_SERVING_PATH — this arm is the paper's configuration served on the GRAPH-CAPTURED residual slot, and the spec requires it natively in every leg. The lens WOULD serve this blob (OP_ADD_VECTOR), so a lens leg is not a failure but a different measurement; refusing rather than producing it")
    fi
    [ -z "${SISTEER_LENS_RHO_EMIT+set}" ] \
      || problems+=("SISTEER_LENS_RHO_EMIT is SET ('${SISTEER_LENS_RHO_EMIT}') — a stray export is exactly the drift this refuses loudly rather than ignoring")
    if [ "${#problems[@]}" -gt 0 ]; then
      f22_log "SERVING POLICY REFUSED (mode=native):"
      local p; for p in "${problems[@]}"; do f22_log "  - $p"; done
      f22_fatal "ar_assert_serving_identity native: ${#problems[@]} problem(s) — refusing before anything is served"
    fi
    f22_log "serving policy OK (native): path=$F22_SERVING_PATH (selection ${F22_SERVING_PATH_MODE:-unset}), cells=[${F22_LAUNCH_CELLS[*]:-from manifest}], SISTEER_LENS_RHO_EMIT unset. Evidence deferred to ar_assert_serving_evidence after each standup"
    return 0
  fi
  head=$(git rev-parse HEAD 2>/dev/null) || f22_fatal "ar_assert_serving_identity: $PWD is not a git checkout"
  # whatever the lineage says, the engine serves the files ON DISK: their bytes as they are must be HEAD's committed bytes
  ar_serving_bytes_problems
  if git merge-base --is-ancestor "$AR_W8_SHA" HEAD 2>/dev/null; then
    # HEAD descends from the pinned serving commit: the byte-identity and the last-touching commit come from the history
    git diff --quiet "$AR_W8_SHA" "$head" -- "$AR_SERVING_DIR" "$AR_PETRI_RUNNER" \
      || problems+=("the serving stack differs from W8 ($AR_W8_SHA): $(git diff --name-only "$AR_W8_SHA" "$head" -- "$AR_SERVING_DIR" "$AR_PETRI_RUNNER" | tr '\n' ' ')— this lane adds no serving file, and every manifest it writes claims serving_impl_commit $AR_SERVING_IMPL_COMMIT")
    last=$(git log -1 --format=%H "$head" -- "$AR_SERVING_DIR" "$AR_PETRI_RUNNER")
    [ "$last" = "$AR_SERVING_IMPL_COMMIT" ] \
      || problems+=("the last commit touching the serving paths is $last, but this lane's manifests record $AR_SERVING_IMPL_COMMIT — one of the two is a lie")
  elif git cat-file -e "$AR_W8_SHA^{commit}" 2>/dev/null; then
    # the pinned serving commit is in the object store but HEAD does not descend from it (an
    # unrelated history fetched in): presence alone is not the history the last-touching commit is
    # read from, and the content path is for a clone WITHOUT that commit — neither applies
    problems+=("W8 ($AR_W8_SHA) is in this clone's object store but HEAD $head does not descend from it (an unrelated history fetched in) — the serving stack's identity is read from the history HEAD descends from, or by content where W8 is absent; presence alone is neither, refusing")
  else
    # the clone does NOT carry the pinned serving commit (a published repository without its development history, a
    # shallow clone): the byte-identity is read by CONTENT — the git object ids of the serving tree and petri_runner at
    # HEAD against that commit's, pinned above — and the last-touching commit cannot be read at all, so the
    # serving_impl_commit claim is refused by name either way
    local tree_now petri_now
    tree_now=$(git rev-parse "$head:$AR_SERVING_DIR" 2>/dev/null) || tree_now=absent
    petri_now=$(git rev-parse "$head:$AR_PETRI_RUNNER" 2>/dev/null) || petri_now=absent
    if [ "$tree_now" != "$AR_W8_SERVING_TREE" ] || [ "$petri_now" != "$AR_W8_PETRI_RUNNER_BLOB" ]; then
      problems+=("the serving stack is not W8's bytes (tree $tree_now vs pinned $AR_W8_SERVING_TREE; petri_runner $petri_now vs pinned $AR_W8_PETRI_RUNNER_BLOB) — this clone does not carry W8 ($AR_W8_SHA), so the comparison is by content; this lane adds no serving file, and every manifest it writes claims serving_impl_commit $AR_SERVING_IMPL_COMMIT")
    else
      problems+=("serving_impl_commit cannot be confirmed: this clone does not carry W8 ($AR_W8_SHA), and the check that the last commit touching the serving paths is $AR_SERVING_IMPL_COMMIT needs the full development history (the serving bytes are W8's by content: tree $tree_now, petri_runner $petri_now)")
    fi
  fi
  [ -z "${SISTEER_LENS_RHO_EMIT+set}" ] \
    || problems+=("SISTEER_LENS_RHO_EMIT is SET ('${SISTEER_LENS_RHO_EMIT}') — on an unpatched tree the flag is dead, but a stray export is exactly the drift this refuses loudly rather than ignoring")
  if [ "${#problems[@]}" -gt 0 ]; then
    f22_log "SERVING IDENTITY REFUSED (mode=$mode, HEAD=$head):"
    local p; for p in "${problems[@]}"; do f22_log "  - $p"; done
    f22_fatal "ar_assert_serving_identity production: ${#problems[@]} problem(s) — refusing before anything is served"
  fi
  f22_log "serving identity OK (production): stack byte-identical to W8 @ $AR_W8_SHA, serving_impl_commit $AR_SERVING_IMPL_COMMIT, SISTEER_LENS_RHO_EMIT unset"
}

ar_assert_serving_evidence(){   # $1 = production | native, $2 = endpoint.json, $3 = the PINS' served layer (native: required)  -- run after EACH successful standup
  # THE EVIDENCE HALF. The policy check above says what the launch ASKED FOR; this says
  # what the engine DID, from its own records, and it runs after EVERY standup this job brings up --
  # the first and the optional 8192-cap second alike, because a second standup is a second engine and
  # inherits nothing from the first's verdict.
  local mode="${1:-}" ep="${2:?endpoint.json}" want="${3:-}"
  ar_require_mode ar_assert_serving_evidence "$mode"
  if [ "$mode" = production ]; then
    # the lens standup's evidence IS the commit assertion already made at init: this lane adds no
    # serving file, so there is nothing in the endpoint that the diff against the pinned serving
    # commit does not already pin. A no-op, stated rather than silent, so the two modes have the
    # same call shape at every site.
    f22_log "serving evidence OK (production): the W8 byte-identity assertion at init is this mode's evidence; no endpoint claim is added"
    return 0
  fi
  # THE NATIVE BODY LIVES IN common.sh (f22_assert_native_evidence) since the goalset's native-policy arm needs the same
  # evidence after ITS standup (shell-half review, Finding 2): one body, two call sites, no third copy to drift. This lane
  # passes its own restated constants in, so a drifted AR_NATIVE_* is refused inside the body exactly as before.
  # THE PINS' SERVED LAYER (round 2, item 2): SERVED_LAYER is the pins' served_layer (ar_resolve_config reads it off
  # configs/arditi_allpos_pins.json and holds the staged provenance to it), and the body compares every payload's and every
  # worker slot's hooked boundary against THAT -- not against a value derived from the object being certified
  [ -n "$want" ] || f22_fatal "ar_assert_serving_evidence native: no served layer passed -- the pins' served_layer is required to certify the hooked boundary"
  # THE RECORD'S IDENTITY NEEDS THE LAUNCHER'S FACTS (round 3, item 2): the served cell and the pins' object sha for THIS
  # rung (ar_resolve_config exports AR_PIN_OBJECT_SHA256 for the raw unit). A record that cannot name the object it
  # certified cannot be bound to the store its rows land in, so their absence refuses before the body runs.
  [ -n "${AR_CELL:-}" ] || f22_fatal "ar_assert_serving_evidence native: AR_CELL is unset -- the served cell is part of the record's identity (ar_cell.sbatch reads it off the view before ar_init)"
  [ -n "${AR_PIN_OBJECT_SHA256:-}" ] || f22_fatal "ar_assert_serving_evidence native: AR_PIN_OBJECT_SHA256 is unset -- the pins' object sha for this rung is part of the record's identity (ar_resolve_config exports it for the raw unit); a record that cannot name the object it certified cannot be bound to the store"
  # THE BYTES THAT WERE SERVED: ar_bank_view hashed the view's ACTUAL object.pt, bound it to
  # the requested rung's pin before the standup, and recorded the observed sha as AR_SERVED_OBJECT_SHA256. The identity
  # names THOSE bytes beside the pins' -- and they must be the same bytes, or the record would certify a standup that served
  # something other than what its hash says. Both are required, and a disagreement refuses HERE, before the body runs and
  # before anything is stamped (a refused standup gets no identity; a crossed one gets no record at all).
  [ -n "${AR_SERVED_OBJECT_SHA256:-}" ] || f22_fatal "ar_assert_serving_evidence native: AR_SERVED_OBJECT_SHA256 is unset -- ar_bank_view binds the served bytes to the requested rung's pin and records their sha; a record that cannot name the bytes that were served is not stamped"
  [ "$AR_SERVED_OBJECT_SHA256" = "$AR_PIN_OBJECT_SHA256" ] || f22_fatal "ar_assert_serving_evidence native: the served object's sha $AR_SERVED_OBJECT_SHA256 (ar_bank_view: the bytes in the view) != the pins' $AR_PIN_OBJECT_SHA256 (ar_resolve_config: the requested rung's pin) -- the identity would name bytes other than the ones served; refusing before the body runs and before anything is stamped"
  local rec_p; rec_p="$(dirname "$ep")/$AR_NATIVE_SERVING_EVIDENCE_FILE"
  if ! f22_assert_native_evidence "$ep" "$AR_NATIVE_SOURCE_KIND" "$AR_NATIVE_POSITION_MODE" "$AR_NATIVE_OP" "$AR_NATIVE_ADAPTER_OP" "$want"; then
    printf '%s\n' "$F22_NATIVE_EVIDENCE" > "$rec_p" 2>/dev/null || true       # the refusal's reasons stay on disk too (bare: a refusal gets no identity)
    f22_log "SERVING EVIDENCE REFUSED (native): $F22_NATIVE_EVIDENCE"; return 1
  fi
  # THE RECORD THE GRADER READS (round 2, item 3), WITH ITS IDENTITY (round 3, item 2): the body the log line carries,
  # stamped with evidence_sha256 / job id / serve dir / cell / the served adapters' fingerprints / the pins' object sha /
  # the expected layer (serving_evidence_binding.stamp -- the ONE hash the drivers fold into served_target.json and the
  # grader re-derives), beside the endpoint it judged. The heredoc lives in ar_serving_evidence_identity: this function
  # stays one body-call, no second evidence body.
  ar_serving_evidence_identity "$rec_p" "$(dirname "$ep")" "$AR_CELL" "$AR_PIN_OBJECT_SHA256" "$AR_SERVED_OBJECT_SHA256" \
    || f22_fatal "cannot persist the serving evidence with its identity to $rec_p -- grade_arditi_static.py follows the store's pointer to it"
  f22_log "serving evidence persisted: $rec_p"
  f22_log "serving evidence OK (native): $F22_NATIVE_EVIDENCE"
}

ar_serving_evidence_identity(){   # $1 = the record path to write, $2 = serve dir, $3 = cell, $4 = the pins' object sha, $5 = the OBSERVED object sha (ar_bank_view)  -- stamps $F22_NATIVE_EVIDENCE (an OK body) with its identity (serving_evidence_binding.stamp) and writes it
  local out="${1:?record path}" serve_dir="${2:?serve dir}" cell="${3:?cell}" obj="${4:?object sha}" served="${5:?served object sha}"
  "${SPY:?ar_serving_evidence_identity needs SPY}" - "$out" "$serve_dir" "$cell" "$obj" "$served" "${SLURM_JOB_ID:-manual}" "${AR_REPO_ROOT:?}" "$F22_NATIVE_EVIDENCE" <<'PY'
import json, sys
out, serve_dir, cell, obj, served, job, repo, body_json = sys.argv[1:9]
sys.path.insert(0, f"{repo}/scripts/sdfbase/full22")
import serving_evidence_binding as seb
rec = seb.stamp(json.loads(body_json), slurm_job_id=job, serve_dir=serve_dir, cell=cell, object_sha256=obj, served_object_sha256=served)
with open(out, "w", encoding="utf-8") as f:
    json.dump(rec, f, sort_keys=True, indent=1); f.write("\n")
print(f"identity: evidence_sha256={rec[seb.IDENTITY_KEY][seb.SHA_FIELD]} job={job} cell={cell} object={obj[:12]} served={served[:12]} "
      f"fingerprints={rec[seb.IDENTITY_KEY]['residual_fingerprint']} expected_served_layer={rec[seb.IDENTITY_KEY]['expected_served_layer']}")
PY
}

ar_bind_serving_evidence_record(){   # $1 = production | native, $2 = endpoint.json -> $AR_SERVING_EVIDENCE_ENV exported (native) / unset (production)
  # THE STORE'S POINTER (round 3, item 2). The grader used to take the evidence record's path by hand; SERVE_DIR is
  # job-suffixed, neighbouring rungs and retries have healthy records too, and nothing in a store said which record
  # certified the standup its rows were drawn under. Now the launcher names THIS standup's stamped record (re-hashed
  # here) in the env the drivers read (serving_evidence_binding.fold_into): every served_target.json they write folds
  # its sha + path into `evidence`, and the grader follows the STORE to the record. Production: no record exists and
  # nothing is folded -- the env is UNSET (never an inherited value), so every production shard is byte-identical.
  local mode="${1:-}" ep="${2:?endpoint.json}" rec_p sha
  ar_require_mode ar_bind_serving_evidence_record "$mode"
  unset "$AR_SERVING_EVIDENCE_ENV"
  if [ "$mode" = production ]; then
    f22_log "serving evidence record: none to bind (production) -- $AR_SERVING_EVIDENCE_ENV unset; the drivers fold nothing and every shard is byte-identical"
    return 0
  fi
  rec_p="$(dirname "$ep")/$AR_NATIVE_SERVING_EVIDENCE_FILE"
  sha=$("${MPY:-python3}" - "$rec_p" "${AR_REPO_ROOT:?}" <<'PY'
import sys
sys.path.insert(0, f"{sys.argv[2]}/scripts/sdfbase/full22")
import serving_evidence_binding as seb
_rec, sha = seb.load_record(sys.argv[1])          # refuses (non-zero, the reason on stderr) an absent / unshaped / unstamped / edited record
print(sha)
PY
  ) || f22_fatal "ar_bind_serving_evidence_record native: $rec_p is not a stamped serving-evidence record -- the drivers cannot bind their stores to it"
  export "$AR_SERVING_EVIDENCE_ENV"="$rec_p"
  f22_log "serving evidence record bound for the drivers: $AR_SERVING_EVIDENCE_ENV=$rec_p (sha256 ${sha:0:12}…; every served_target.json this job writes points at it, and the grader follows the pointer)"
}

ar_serving_bytes_problems(){   # appends to `problems`: the serving files ON DISK against the entries HEAD commits
  # The engine imports the files on disk, and `git diff --quiet HEAD` (the clean-checkout check) reads them through the clean
  # filters and autocrlf, so a .gitattributes filter that strips an injected change passes it. Anchored at the repository root
  # (never vacuous from a subdirectory), NUL-separated (a name git would C-quote is read raw): every entry HEAD commits under the
  # serving paths is compared with what is on disk as it is — a blob by `git hash-object --no-filters`, a symlink by its target —
  # and every path on disk OR in the index under them that HEAD does not carry is refused, the contents of a `__pycache__/`
  # directory excepted (a .pyc anywhere else is importable and is not).
  local root entry mode type blob path rest raw target
  root=$(git rev-parse --show-toplevel 2>/dev/null) \
    || { problems+=("not inside a git work tree: the serving files on disk cannot be compared with HEAD"); return 0; }
  local -A inhead=()
  while IFS= read -r -d '' entry; do
    mode=${entry%% *}; rest=${entry#* }; type=${rest%% *}; rest=${rest#* }; blob=${rest%%$'\t'*}; path=${rest#*$'\t'}
    inhead["$path"]=1
    [ "$type" = blob ] || continue
    if [ "$mode" = 120000 ]; then
      if [ ! -L "$root/$path" ]; then
        problems+=("$path is a symlink at HEAD but not on disk — the engine would not serve what HEAD commits")
        continue
      fi
      target=$(readlink "$root/$path")
      raw=$(printf '%s' "$target" | git -C "$root" hash-object --no-filters --stdin 2>/dev/null) || raw=unreadable
      [ "$raw" = "$blob" ] \
        || problems+=("$path on disk is a symlink to '$target' (blob $raw), not the target HEAD commits (blob $blob) — the engine follows the link on disk")
      continue
    fi
    if [ -L "$root/$path" ]; then
      problems+=("$path is a symlink on disk but a regular file at HEAD — the engine would serve whatever it points at")
      continue
    fi
    if [ ! -f "$root/$path" ]; then
      problems+=("$path is committed at HEAD but is not on disk — the engine would not serve HEAD's bytes")
      continue
    fi
    raw=$(git -C "$root" hash-object --no-filters -- "$root/$path" 2>/dev/null) || raw=unreadable
    [ "$raw" = "$blob" ] \
      || problems+=("$path on disk is blob $raw (bytes as they are, --no-filters), not the $blob HEAD commits — a clean filter or autocrlf cannot hide the difference, and the engine serves the file on disk")
  done < <(git -C "$root" ls-tree -r -z --full-tree HEAD -- "$AR_SERVING_DIR" "$AR_PETRI_RUNNER" 2>/dev/null)
  local extra=()
  while IFS= read -r -d '' path; do
    [ -n "${inhead[$path]+x}" ] && continue
    case "$path" in */__pycache__/*|__pycache__/*) continue;; esac
    extra+=("$path")
  done < <(git -C "$root" ls-files -z --cached --others -- "$AR_SERVING_DIR" "$AR_PETRI_RUNNER" 2>/dev/null | sort -zu)
  [ "${#extra[@]}" -eq 0 ] \
    || problems+=("paths under the serving paths that HEAD does not carry (on disk or in the index): ${extra[*]} — the engine imports what is on disk")
}

ar_native_serving_impl_commit(){   # $1 = the checkout  -> sets AR_MANIFEST_SERVING_IMPL_COMMIT: the commit that last touched the serving paths THERE
  # The native arm's manifest records the serving stack it actually runs, derived from the checkout's
  # history — the development history after the pinned serving commit, which every full checkout
  # carries. A clone that does not carry that commit (a published repository without its development
  # history, whatever was committed on top of it; a shallow clone) has no such history:
  # `git log -1 -- <paths>` would answer its own root or a later unrelated commit, which is not the
  # fact the manifest records, so the derivation is refused by name rather than answered wrongly.
  # The test is the pinned commit's ABSENCE, never a commit count: one commit on top of a squash
  # does not restore the history.
  local repo="${1:?ar_native_serving_impl_commit: checkout}"
  git -C "$repo" rev-parse --verify HEAD >/dev/null 2>&1 \
    || f22_fatal "arditi: cannot read HEAD of $repo (not a git checkout?) — the native arm's serving_impl_commit is derived from its history"
  git -C "$repo" cat-file -e "$AR_W8_SHA^{commit}" 2>/dev/null \
    || f22_fatal "arditi: the manifest's serving_impl_commit cannot be derived in $repo: this clone does not carry W8 ($AR_W8_SHA), so the history the native serving stack's last-touching commit is read from is not here ($repo is at $(git -C "$repo" rev-parse --short HEAD 2>/dev/null); a published repository without its development history, or a shallow clone); this check needs the full development history"
  # present is not enough: the history the last-touching commit is read from is the one HEAD DESCENDS from. The pinned
  # serving commit fetched into a clone from an unrelated history sits in the object store while `git log -1 -- <paths>`
  # would still answer the clone's own root
  git -C "$repo" merge-base --is-ancestor "$AR_W8_SHA" HEAD 2>/dev/null \
    || f22_fatal "arditi: the manifest's serving_impl_commit cannot be derived in $repo: W8 ($AR_W8_SHA) is in this clone's object store but HEAD $(git -C "$repo" rev-parse --short HEAD 2>/dev/null) does not descend from it (an unrelated history fetched in) — the last-touching commit is read from the history HEAD descends from, and presence alone is not that history; this check needs the full development history"
  AR_MANIFEST_SERVING_IMPL_COMMIT=$(git -C "$repo" log -1 --format=%H HEAD -- "$AR_SERVING_DIR" "$AR_PETRI_RUNNER") \
    || f22_fatal "arditi: cannot read the serving tree's last commit in $repo"
  [ -n "$AR_MANIFEST_SERVING_IMPL_COMMIT" ] \
    || f22_fatal "arditi: no commit in $repo touches $AR_SERVING_DIR $AR_PETRI_RUNNER — the native arm's serving_impl_commit cannot be derived"
}

ar_resolve_config(){   # $1 = model key, $2 = strength view  -> KEY VIEW TAG CFG MANIFEST STRENGTH SERVED_LAYER, and (Finding 3) AR_RUN_TAG AR_KIND AR_PHASE AR_MANIFEST_SERVING_IMPL_COMMIT
  KEY="${1:?key}"; VIEW="${2:?view}"
  local py view_unit gen; py="${MPY:-python3}"
  # every path comes from the ONE table (scripts/arditi/arditi_strength.py), through its UNIT-KEYED
  # surface (the `_in` helpers): the view's LETTER names the unit, the unit names the config
  # pair, the run tag, the served kind and phase. A view spelled any other way -- the other unit's
  # letter included -- resolves a bank dir that was never built, so the table REFUSES a
  # non-canonical spelling rather than defaulting. For an `ar_r` view every string below is the
  # one this lane always resolved, to the byte (the unit-keyed helpers are pinned equal to the
  # residual-norm-fraction functions over the whole probe grid).
  read -r view_unit CFG MANIFEST STRENGTH AR_RUN_TAG AR_KIND AR_PHASE gen < <("$py" - "$KEY" "$VIEW" <<'PY'
import sys
sys.path.insert(0, "scripts/arditi")
import arditi_strength as st
from arditi_targets import paper_lane
key, view = sys.argv[1], sys.argv[2]
paper_lane(key)                                  # refuses an unknown target
unit = st.unit_of_view(view)                     # refuses a view of no unit (ar_d..., a typo)
s = st.strength_of_view_in(unit, view)           # refuses a non-canonical spelling of ITS unit
spec = st.unit_spec(unit)
print(unit, st.serve_config_rel_in(unit, key, s), st.manifest_rel_in(unit, key, s), repr(s),
      st.run_tag_in(unit, key, s), spec.served_kind, spec.served_phase, spec.config_generator)
PY
) || f22_fatal "arditi: cannot resolve ($KEY, $VIEW) through scripts/arditi/arditi_strength.py"
  # ONE JOB, ONE UNIT: the view's unit must be the unit whose roots ar_init bound (AR_UNIT), or the
  # config below would name one bank and the roots another
  [ "$view_unit" = "${AR_UNIT:?ar_resolve_config: AR_UNIT is unbound — ar_init binds it}" ] \
    || f22_fatal "view $VIEW is a $view_unit view, but this job bound the $AR_UNIT unit's roots at init (AR_UNIT) — one job serves one unit"
  TAG="${KEY}_${VIEW}"
  [ -f "$CFG" ] && [ -f "$MANIFEST" ] \
    || f22_fatal "missing $CFG or $MANIFEST (generate: $gen$( [ -f "$gen" ] || echo " — not on this tree yet: the $AR_RUN_TAG_PREFIX unit's config pairs are Phase B's output, rendered once its ladder is baked and staged"))"
  # the manifest must record the serving stack this job actually runs. For the prefill-only unit
  # that is the pinned serving commit's claim (AR_SERVING_IMPL_COMMIT, the byte-identity
  # ar_assert_serving_identity production enforces); for the native arm -- which REQUIRES serving
  # files that commit did not have -- it is the commit that last touched the serving paths in THIS
  # checkout, exactly the fact the production gate asserts of the pinned commit, applied to the
  # tree the job runs from.
  if [ "$AR_UNIT" = "$AR_UNIT_PRODUCTION" ]; then
    AR_MANIFEST_SERVING_IMPL_COMMIT="$AR_SERVING_IMPL_COMMIT"
  else
    ar_native_serving_impl_commit "$AR_REPO_ROOT"       # sets AR_MANIFEST_SERVING_IMPL_COMMIT, or refuses by name
  fi
  grep -q "serving_impl_commit: $AR_MANIFEST_SERVING_IMPL_COMMIT" "$MANIFEST" \
    || f22_fatal "$MANIFEST does not record serving_impl_commit $AR_MANIFEST_SERVING_IMPL_COMMIT$( [ "$AR_UNIT" = "$AR_UNIT_PRODUCTION" ] || echo " (the commit that last touched $AR_SERVING_DIR $AR_PETRI_RUNNER in this checkout — the native arm's manifest must record the serving stack it actually runs, never W8's $AR_SERVING_IMPL_COMMIT)")"
  # TEXT-ONLY SERVING for the multimodal targets — the wave's belt, ours too: without
  # --limit-mm-per-prompt the engine's multimodal profiling (dummy max-size images sized off
  # S x L) is OOM-killed by the 200G host cgroup AFTER the venvs have built.
  case "$KEY" in qwen35_4b|glm46vflash)
    grep -q -- '--limit-mm-per-prompt' "$MANIFEST" \
      || f22_fatal "$MANIFEST serves multimodal $KEY without --limit-mm-per-prompt (text-only pin) — regenerate";;
  esac
  # the layer the lens MUST hook for this target, from the fit record that produced the object —
  # read here so every leg can assert the served hook set against it
  # THE SERVED LAYER, from the staged provenance of THIS unit's bank. The two units store
  # the layer under INVERSE conventions and both are asserted, never assumed: a `static_r` blob at
  # site="pre" stores the FIT layer and the adapter shifts it (stored == served + 1); a `vector_all`
  # blob stores the SERVED block-output layer and the adapter serves it as stored (stored == served,
  # arditi_strength.RAW_STORED_LAYER_IS_SERVED_LAYER). Reading one convention for the other steers a
  # block off with every other signal healthy, so a provenance carrying the OTHER unit's relation
  # under this unit's cell id is refused by name. The production branch below is the assertion set
  # this lane always ran, to the letter; the unit-keyed values come from the ONE table.
  # THE PINS BIND THE LAYERS for the raw unit. Every assertion below is
  # RELATIONAL -- stored == served == layers[0] == fit layer - 1 -- and a coherent object baked a block off
  # (fit 12 / served 11 / stored 11) satisfies all of it, resolves SERVED_LAYER=11, and the post-standup
  # evidence then agrees with the wrong block. So for the raw unit SERVED_LAYER is read off the PINS
  # ($AR_PINS, entry <target>.<staged id>: fit_layer / served_layer / stored_layer / direction_sha256, from
  # the fit record, arditi_strength.RAW_PIN_LAYER_KEYS), THIS rung's staged provenance is held to them by
  # name, and a pins entry lacking the keys refuses. The production branch reads exactly what it always did.
  # AND THE PINS ARE HELD TO THE APPROVED FIT FIRST: the arms table's arms.arditi_allpos.fits.<target>
  # block names the ONE approved fit record; a pins entry that disagrees with it (12/11/11 pins are as coherent as
  # 13/12/12) refuses BY NAME, attributing the authority to the table, before the pins certify any object
  # (arditi_strength.raw_pin_binding, the one validated read). A target the table approves no fit for refuses by name.
  SERVED_LAYER=$("$py" - "$KEY" "$AR_OBJ_ROOT" "$AR_UNIT" "$AR_PINS" "$STRENGTH" "$AR_ARMS" <<'PY'
import json, sys
from pathlib import Path
sys.path.insert(0, "scripts/arditi")
import arditi_strength as st
key, root, unit, pins_rel, strength, arms_rel = sys.argv[1], Path(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6]
spec = st.unit_spec(unit)
cells_dir = st.cells_dir_in(unit, root, key)
binding = None
if spec.unit == st.DOSE_UNIT:
    # ANY staged rung of this target answers the question — they all carry the same layer — so
    # read whichever exists rather than naming one, which pinned the RETIRED ar_d1 cell and would
    # refuse against the residual-norm bank.
    cells = sorted(cells_dir.glob(f"*__{spec.cell_id}"))
    if not cells:
        raise SystemExit(f"no staged {spec.cell_id} cell under {cells_dir}")
    p = cells[0] / "object.pt.provenance.json"
else:
    # the raw unit: THIS rung's staged cell, and the PINS' binding for it
    if not Path(pins_rel).is_file():
        raise SystemExit(f"no {pins_rel} -- the raw unit's pins (Phase B's output) bind each rung's absolute layers; "
                         "a rung cannot resolve its served layer without them")
    staged_id = st.staged_id_in(unit, float(strength))          # STRENGTH arrives as text (repr(s) off the resolver)
    _pin, binding = st.raw_pin_binding(json.loads(Path(pins_rel).read_text()), st.load_arms_table(arms_rel), key, staged_id, pins_rel)
    p = cells_dir / staged_id / "object.pt.provenance.json"
    if not p.is_file():
        raise SystemExit(f"no staged {staged_id} provenance at {p}")
rec = json.loads(p.read_text())
layers = rec["layers"]
assert len(layers) == 1, f"{p}: {len(layers)} layers, expected exactly 1"
sel = rec["fit"]["selected"]
assert int(sel["served_layer"]) == int(sel["layer"]) - 1, (
    f"{p}: served_layer {sel['served_layer']} != fit layer {sel['layer']} - 1")
assert int(layers[0]) == int(sel["served_layer"]), f"{p}: layers[0] != served_layer"
# `layers` is the SERVED block layer — what the server's own attestation reports hooking, and
# what every leg asserts against. `stored_layer` is the key the BLOB is written under: for a
# static_r blob at site="pre" the FIT layer, one LATER; for a vector_all blob the served layer
# itself. The two conventions are inverses, and confusing them steers a block off with every
# other signal healthy, so both indices are read and the relation this UNIT's kind implies is
# asserted rather than assumed.
stored = rec.get("stored_layer")
assert stored is not None, (
    f"{p}: no stored_layer. A blob stores one layer index and its adapter may shift it, so a "
    "provenance recording only one index cannot tell a correct object from one baked a block "
    "off; its absence is a refusal, never a default")
if spec.stored_layer_is_served_layer:
    assert int(stored) == int(sel["served_layer"]), (
        f"{p}: stored_layer {stored} != served layer {sel['served_layer']} — a {spec.served_kind} blob stores the "
        "SERVED block-output layer and the adapter serves it as stored; this provenance carries the static_r "
        "relation (stored == fit layer), i.e. an object of the OTHER unit, or one baked a block late")
    assert int(stored) == int(layers[0]), f"{p}: stored_layer {stored} != layers[0] {layers[0]}"
else:
    assert int(stored) == int(sel["layer"]), f"{p}: stored_layer {stored} != fit layer {sel['layer']}"
    assert int(stored) == int(layers[0]) + 1, f"{p}: stored_layer {stored} != served layer + 1"
assert rec.get("kind") == spec.served_kind, f"{p}: kind {rec.get('kind')!r} != {spec.served_kind!r}"
assert rec.get("phase") == spec.served_phase, (
    f"{p}: phase {rec.get('phase')!r} != {spec.served_phase!r} — this unit serves {spec.served_phase} on every rung")
if spec.unit != st.DOSE_UNIT:
    assert rec.get("dose_unit") == spec.unit, (
        f"{p}: dose_unit {rec.get('dose_unit')!r} != {spec.unit!r} — the dose number means nothing without its unit")
if binding is not None:
    # the ABSOLUTE binding: this provenance's fit layer, served layer, stored layer, layers and direction sha are
    # the pins' (i.e. the fit record's), or the object is not the one the fit produced -- refused by name
    problems = st.raw_provenance_layer_problems(rec, binding, str(p))
    if problems:
        raise SystemExit("PINS BINDING REFUSED:\n  " + "\n  ".join(problems))
    print(int(binding["served_layer"]))
else:
    print(int(layers[0]))
PY
) || f22_fatal "arditi: cannot resolve the served layer for $KEY from the staged provenance under $AR_OBJ_ROOT$( [ "$AR_UNIT" = "$AR_UNIT_PRODUCTION" ] || echo " held to the pins $AR_PINS (themselves held to the approved fit in $AR_ARMS)" )"
  # THE PINNED OBJECT'S IDENTITY for the raw unit: the pins entry's object_sha256, read through the SAME
  # validated read as the served layer (the pins held to the approved fit first), rides the standup's evidence record as
  # identity.object_sha256 (ar_assert_serving_evidence native) so the grader can hold that record to THIS rung's pin -- a
  # neighbouring rung's healthy record names another object and refuses. The production branch sets it EMPTY: its
  # evidence is the pinned serving commit's claim and no record exists.
  AR_PIN_OBJECT_SHA256=""
  if [ "$AR_UNIT" != "$AR_UNIT_PRODUCTION" ]; then
    AR_PIN_OBJECT_SHA256=$("$py" - "$KEY" "$AR_UNIT" "$AR_PINS" "$AR_ARMS" "$STRENGTH" <<'PY'
import json, sys
sys.path.insert(0, "scripts/arditi")
import arditi_strength as st
key, unit, pins_rel, arms_rel, strength = sys.argv[1:6]
staged_id = st.staged_id_in(unit, float(strength))
pin, _binding = st.raw_pin_binding(json.load(open(pins_rel)), st.load_arms_table(arms_rel), key, staged_id, pins_rel)
print(st.raw_pin_object_sha256(pin, f"{pins_rel} [{key}.{staged_id}]"))
PY
    ) || f22_fatal "arditi: cannot read the pins' object_sha256 for $KEY $VIEW from $AR_PINS -- the standup's evidence record carries it as its identity"
  fi
  # the dose is logged under ITS unit's name: `alpha` (a fraction of the residual norm) or `c` (a raw coefficient), never one for the other
  f22_log "arditi config: key=$KEY view=$VIEW $( [ "$AR_UNIT" = "$AR_UNIT_PRODUCTION" ] && echo alpha || echo c )=$STRENGTH kind=$AR_KIND phase=$AR_PHASE served_layer=$SERVED_LAYER cfg=$CFG manifest=$MANIFEST$( [ -z "$AR_PIN_OBJECT_SHA256" ] || echo " pin_object=${AR_PIN_OBJECT_SHA256:0:12}" )"
}

ar_bank_view(){   # $1 = python  -> builds + verifies OUR (KEY, VIEW) bank view; exports TT_TWO_SI_DEFENSES_DIR
  local py="${1:?python}" lock bank blob_out
  bank="$AR_OBJ_ROOT/$KEY/$AR_BANK_SUBDIR"      # <root>/<target>/strength (ar_r) | <root>/<target>/ladder (ar_c): arditi_strength.bank_dir_in
  # the pins bind the staged bytes; a serving leg without them would serve an unpinned object
  [ -f "$AR_PINS" ] || f22_fatal "missing $AR_PINS — nothing of this lane is staged yet, so there is no pinned object to serve"
  lock="$AR_OBJ_ROOT/.$KEY-$VIEW.view.lock"
  mkdir -p "$(dirname "$lock")"
  # VERIFY FIRST, BUILD ONLY UNDER A LOCK: the view is SHARED state between this lane's jobs, and
  # concurrent rebuilds raced on the wave (job 423389 died reading a sidecar another job was
  # mid-write). The read-only --check is the common path.
  if ! "$py" scripts/sdfbase/full22/make_bank_view.py --check --roster "$AR_ROSTER" --pins "$AR_PINS" \
        --bank "$bank" --target "$KEY" --view "$VIEW" >/dev/null 2>&1; then
    f22_log "arditi bank view $KEY/$VIEW does not verify — building it under $lock"
    (
      flock -w 900 9 || { echo "FATAL: could not take the bank-view lock for $KEY/$VIEW in 900s" >&2; exit 1; }
      "$py" scripts/sdfbase/full22/make_bank_view.py --roster "$AR_ROSTER" --pins "$AR_PINS" \
        --bank "$bank" --target "$KEY" --view "$VIEW"
    ) 9>"$lock" || f22_fatal "arditi bank view for $KEY/$VIEW refused"
  fi
  "$py" scripts/sdfbase/full22/make_bank_view.py --check --roster "$AR_ROSTER" --pins "$AR_PINS" \
    --bank "$bank" --target "$KEY" --view "$VIEW" || f22_fatal "arditi bank view for $KEY/$VIEW refused"
  export TT_TWO_SI_DEFENSES_DIR="$bank/cells_view/$VIEW"
  case "$TT_TWO_SI_DEFENSES_DIR" in
    */exp/objects/sdfbase/*|*/exp/sdfbase_full22/*|*/exp/objects/alphasteer_f22/*)
      f22_fatal "arditi view dir $TT_TWO_SI_DEFENSES_DIR resolves under a PEER's root";;
  esac
  mapfile -t ALL_CELLS < <("$py" - "$CFG" <<'PY'
import sys, yaml; print("\n".join(yaml.safe_load(open(sys.argv[1]))["explicit_cells"]))
PY
)
  [ "${#ALL_CELLS[@]}" -ge 2 ] || f22_fatal "$CFG lists only ${ALL_CELLS[*]} — nothing to contrast"
  for c in "${ALL_CELLS[@]}"; do
    [ "$c" = none ] && continue
    for f in object.pt staged_from.json; do
      [ -f "$TT_TWO_SI_DEFENSES_DIR/$c/$f" ] || f22_fatal "arditi bank view $TT_TWO_SI_DEFENSES_DIR/$c lacks $f"
    done
    local n_id=0 f_id
    for f_id in train_budget_meta.json fit_provenance.json derived_meta.json; do
      [ -f "$TT_TWO_SI_DEFENSES_DIR/$c/$f_id" ] && n_id=$((n_id+1))
    done
    [ "$n_id" -eq 1 ] || f22_fatal "arditi bank view $TT_TWO_SI_DEFENSES_DIR/$c declares $n_id identity sidecars (need exactly one)"
    if [ "$AR_UNIT" = "$AR_UNIT_PRODUCTION" ]; then
    # THE DOSE-IN-NORM INVARIANT, ENFORCED AT SERVE TIME on the bytes about to be
    # served. registry_shim's coeff-family fold moves base_scale and leaves the
    # vector, so a folded artifact would serve c x the dose while every row still
    # says alpha — and every other check passes on it. This lane never folds, but
    # a note cannot stop a future caller; this can: a blob whose base_scale is not
    # exactly 1.0, whose kind is not static_r, whose rs is not keyed by the ONE
    # stored (fit) layer, or which carries a position-record key, does not get
    # served by any job of this lane.
    "$py" - "$TT_TWO_SI_DEFENSES_DIR/$c/object.pt" "$TT_TWO_SI_DEFENSES_DIR/$c/object.pt.provenance.json" <<'ARBLOB' || f22_fatal "arditi: the object about to be served for $KEY/$c violates the lane's dose-in-norm invariant (see the PROBLEM lines)"
import json, sys, torch
blob = torch.load(sys.argv[1], map_location="cpu", weights_only=True)
prov = json.load(open(sys.argv[2]))
problems = []
if blob.get("kind") != "static_r":
    problems.append(f"kind {blob.get('kind')!r} != 'static_r'")
if float(blob.get("base_scale", float("nan"))) != 1.0:
    problems.append(f"base_scale {blob.get('base_scale')!r} != 1.0 — the dose lives in the vector's "
                    "norm; a moved base_scale is a FOLDED artifact serving a different dose than its row")
rs = blob.get("rs") or {}
if sorted(int(k) for k in rs) != [int(prov["stored_layer"])]:
    problems.append(f"rs keyed {sorted(rs)} but the provenance's stored (fit) layer is {prov['stored_layer']}")
for k in ("positions", "fit_positions"):
    if k in blob:
        problems.append(f"blob carries {k!r}, which this kind's adapter refuses (no declared position schema)")
for p in problems:
    print("PROBLEM", p)
sys.exit(1 if problems else 0)
ARBLOB
    else
      # THE RAW UNIT'S CONTRACT (two strict units, never one permissive rule),
      # the mirror of the block above and several assertions its INVERSE, with every expected value
      # read off the unit's record rather than typed: kind vector_all; `vectors` keyed by exactly the
      # ONE stored layer, which for this kind IS the served layer (stored == fit.selected.served_layer);
      # `positions` / `fit_positions` PRESENT (the two keys static_r forbids; assert_blob_keys_in) and
      # equal to the record's; skip_bos as recorded; NO base_scale and NO `rs` (a vector_all blob has
      # neither -- their presence is another kind's blob wearing this arm's name, or a folded artifact);
      # the provenance's kind, phase and dose_unit this unit's. A static_r blob staged under this
      # arm's name -- the other half of the paper's intervention filed as all of it -- fails here.
      # ...and to the PINS' absolute layers: the `vectors` key and the provenance's indices are held
      # to <pins>[<target>][<staged id>].{fit_layer, served_layer, stored_layer, direction_sha256} -- the fit record's --
      # so a coherent object baked a block off (which every relational line below admits) is refused BY NAME here too;
      # and the pins are first held to the arms table's APPROVED FIT for this target
      # (arms.arditi_allpos.fits.<target>; arditi_strength.raw_pin_binding), so pins written from the wrong fit refuse
      # by name, attributing the authority to the table, before they certify these bytes.
      # ...AND THE BYTES ARE BOUND TO THE REQUESTED RUNG. Everything above holds the object to
      # the requested rung's PIN -- kind, keys, layers, the approved fit -- and every rung of a ladder shares fit, direction and
      # layers, so none of it can tell c=0.1 from c=0.2. make_bank_view.py hashes the object against the pin the ROSTER maps
      # the view to, never against the rung REQUESTED, so a crossed or stale mapping (cells_view/ar_c0p1/arditi_allpos ->
      # ../../cells/ar_c0p2__arditi_allpos) passed both gates: twice the dose under the lower dose's name, hash and evidence
      # identity. Three INDEPENDENT bindings now, any one refusing by name with both values: sha256(the ACTUAL object.pt) ==
      # the requested pin's object_sha256; the dose the provenance (and the pin) records names the requested rung; the view
      # entry's realpath is cells/<requested staged id>. On success the heredoc prints ONE `OBSERVED object_sha256=...` line
      # and the shell carries that sha as AR_SERVED_OBJECT_SHA256 to the evidence stamp (ar_assert_serving_evidence native),
      # which refuses unless it equals the pins' AR_PIN_OBJECT_SHA256 -- so the identity names the bytes that were served.
      [ -z "${AR_SERVED_OBJECT_SHA256:-}" ] || f22_fatal "$AR_RUN_TAG_PREFIX: a second served cell ($c) in one view — the evidence identity names ONE served object, and AR_SERVED_OBJECT_SHA256 is already ${AR_SERVED_OBJECT_SHA256:0:12}…"
      blob_out=$("$py" - "$TT_TWO_SI_DEFENSES_DIR/$c/object.pt" "$TT_TWO_SI_DEFENSES_DIR/$c/object.pt.provenance.json" "$AR_UNIT" "$AR_PINS" "$KEY" "$STRENGTH" "$AR_ARMS" <<'ARBLOB'
import hashlib, json, os, sys, torch
sys.path.insert(0, "scripts/arditi")
import arditi_strength as st
blob = torch.load(sys.argv[1], map_location="cpu", weights_only=True)
prov = json.load(open(sys.argv[2]))
spec = st.unit_spec(sys.argv[3])
pins_rel, key, strength, arms_rel = sys.argv[4], sys.argv[5], sys.argv[6], sys.argv[7]
problems = []
binding = _pin = staged_id = None
try:
    staged_id = st.staged_id_in(spec.unit, float(strength))     # STRENGTH arrives as text (repr(s) off the resolver)
    _pin, binding = st.raw_pin_binding(json.load(open(pins_rel)), st.load_arms_table(arms_rel), key, staged_id, pins_rel)
except SystemExit as e:
    problems.append(str(e))
if blob.get("kind") != spec.served_kind:
    problems.append(f"kind {blob.get('kind')!r} != {spec.served_kind!r}")
try:
    st.assert_blob_keys_in(spec.unit, blob.keys(), "object.pt")
except st.StrengthRefusal as e:
    problems.append(str(e))
if "base_scale" in blob:
    problems.append(f"blob carries base_scale {blob.get('base_scale')!r} — a {spec.served_kind} object has none: the op's coeff is "
                    "1.0 and the dose is baked into the vector; a base_scale here is another kind's blob under this arm's name")
if "rs" in blob:
    problems.append("blob carries 'rs' — the static_r spelling, whose stored layer is the FIT layer, one LATER than this kind's")
vectors = blob.get("vectors")
layers = sorted(int(k) for k in vectors) if isinstance(vectors, dict) else []
if not layers:
    problems.append(f"blob has no non-empty 'vectors' mapping ({type(vectors).__name__})")
elif layers != [int(prov["stored_layer"])]:
    problems.append(f"vectors keyed {layers} but the provenance's stored layer is {prov['stored_layer']}")
sel = (prov.get("fit") or {}).get("selected") or {}
if sel.get("served_layer") is not None and int(prov["stored_layer"]) != int(sel["served_layer"]):
    problems.append(f"provenance stored_layer {prov['stored_layer']} != served_layer {sel['served_layer']} — a {spec.served_kind} "
                    "blob stores the SERVED layer (the static_r relation stored == fit layer is the OTHER unit's)")
for k, want in (("positions", spec.served_positions), ("fit_positions", spec.fit_positions)):
    if k in blob and blob[k] != want:
        problems.append(f"blob {k}={blob[k]!r} != {want!r} — this arm serves the paper's ungated hook at every position of every forward")
if blob.get("skip_bos") is not spec.served_skip_bos:
    problems.append(f"blob skip_bos {blob.get('skip_bos')!r} != {spec.served_skip_bos!r}")
for k, want in (("kind", spec.served_kind), ("phase", spec.served_phase), ("dose_unit", spec.unit)):
    if prov.get(k) != want:
        problems.append(f"provenance {k} {prov.get(k)!r} != {want!r}")
if binding is not None:
    if layers and layers != [binding["stored_layer"]]:
        problems.append(f"vectors keyed {layers} but the pins' stored_layer (the fit record's served layer) is {binding['stored_layer']}")
    problems.extend(st.raw_provenance_layer_problems(prov, binding, "provenance"))
# THE BYTES, THE DOSE AND THE NAME ARE THE REQUESTED RUNG'S (round 5). Every check above is one every rung of the ladder
# passes alike (one fit, one direction, one layer), so none of it can tell a crossed mapping from a right one; these can,
# and each refuses on its own, naming both values.
def sha256_file(p):                                   # chunked, the way make_bank_view.sha256_file reads it
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
observed = sha256_file(sys.argv[1])
entry_dir = os.path.dirname(sys.argv[1])              # <bank>/cells_view/<view>/<cell>: the entry the standup will serve
real_dir = os.path.dirname(os.path.realpath(sys.argv[1]))   # ...which resolves to <bank>/cells/<SOME staged id>
if _pin is not None and staged_id is not None:
    # (1) THE BYTES: the file about to be served hashes to the REQUESTED rung's pin -- not to whichever pin the roster maps
    try:
        pinned = st.raw_pin_object_sha256(_pin, f"{pins_rel} [{key}.{staged_id}]")
    except SystemExit as e:
        problems.append(str(e))
    else:
        if observed != pinned:
            problems.append(f"NOT THE REQUESTED RUNG'S BYTES: {sys.argv[1]} hashes to {observed}, but the pins bind {key}.{staged_id} "
                            f"(the requested rung, c={strength}) to {pinned} — the view entry {entry_dir} serves another rung's object "
                            f"under this rung's name (a crossed or stale roster mapping: make_bank_view.py verifies the object against "
                            f"the pin the ROSTER maps it to, never against the rung REQUESTED, and every rung shares fit, direction and "
                            f"layers, so no other check here can tell them apart)")
    # (2a) THE DOSE THE PIN RECORDS names the rung its key names (a disagreement is a pins file that disagrees with itself)
    if "c" not in _pin:
        problems.append(f"pins {key}.{staged_id} records no 'c' — the dose the pinned bytes carry is required beside their sha")
    else:
        try:
            pin_sid = st.staged_id_in(spec.unit, _pin["c"])
        except SystemExit as e:
            problems.append(f"pins {key}.{staged_id} c={_pin['c']!r} names no rung ({e})")
        else:
            if pin_sid != staged_id:
                problems.append(f"pins c {_pin['c']} names rung {pin_sid}, but the entry is {key}.{staged_id} — the pins file disagrees "
                                "with itself about this rung's dose")
if staged_id is not None:
    # (2b) THE DOSE THE PRODUCER RECORDED on these bytes names the requested rung
    if "c" not in prov:
        problems.append(f"provenance has no 'c' — a {spec.served_kind} object's provenance records the raw coefficient it was baked at; "
                        f"without it the dose these bytes carry cannot be held to the requested {strength}")
    else:
        try:
            prov_sid = st.staged_id_in(spec.unit, prov["c"])
        except SystemExit as e:
            problems.append(f"provenance c={prov['c']!r} names no rung ({e})")
        else:
            if prov_sid != staged_id:
                problems.append(f"provenance c {prov['c']} names rung {prov_sid}, but the requested rung is {staged_id} (c={strength}) — "
                                "these bytes were baked at another dose than the one they are about to be served as")
    # (3) THE NAME: the view entry resolves to the REQUESTED rung's staged copy, not another rung's
    if os.path.basename(real_dir) != staged_id:
        problems.append(f"VIEW MAPPING CROSSED: {entry_dir} resolves to cells/{os.path.basename(real_dir)}, not cells/{staged_id} "
                        "(the requested rung) — the roster maps this view's cell to another rung's staged id")
for p in problems:
    print("PROBLEM", p)
if not problems:
    print(f"OBSERVED object_sha256={observed} staged_id={staged_id} c={strength} view_entry={real_dir}")
sys.exit(1 if problems else 0)
ARBLOB
) || { printf '%s\n' "$blob_out"; f22_fatal "$AR_RUN_TAG_PREFIX: the object about to be served for $KEY/$c violates the $AR_UNIT unit's $AR_KIND contract or is not the REQUESTED rung's pinned bytes (see the PROBLEM lines)"; }
      printf '%s\n' "$blob_out"
      # the OBSERVED sha, read off the ONE line the heredoc prints on success -- exactly 64 hex, or the bytes were not bound
      AR_SERVED_OBJECT_SHA256=$(printf '%s\n' "$blob_out" | sed -n 's/^OBSERVED object_sha256=\([0-9a-f]\{64\}\) .*/\1/p')
      [ "${#AR_SERVED_OBJECT_SHA256}" -eq 64 ] || f22_fatal "$AR_RUN_TAG_PREFIX: the bank-view check for $KEY/$c exited 0 without exactly one OBSERVED object_sha256 line — the served bytes were not bound to the requested rung, refusing"
      f22_log "arditi bank view: the bytes about to be served for $KEY/$c are the requested rung's (sha256 ${AR_SERVED_OBJECT_SHA256:0:12}… == the pins' for c=$STRENGTH)"
    fi
  done
  # THE SHORT-VIEW GUARD against OUR roster (never the flagless roster_cell_count.py, which counts
  # the WAVE roster): a config that lists a cell the view does not hold, or vice versa, is a
  # narrowed leg that would still produce plausible numbers.
  local want have n_view
  want=$("$py" - "$AR_ROSTER" "$KEY" "$VIEW" <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, "scripts/sdfbase/full22")
from f22_roster import load_roster, served_cells
print(len(served_cells(load_roster(Path(sys.argv[1])), sys.argv[2], sys.argv[3])))
PY
)
  have=$(( ${#ALL_CELLS[@]} - 1 ))
  # EXCLUDE THE PRODUCER'S OWN SIDECARS. make_bank_view.py writes
  # VIEW_MANIFEST.json (always) and full22_derived_quality.json (when there are
  # derived cells) INTO the view dir, and its own cell-vs-sidecar rule is
  # `not p.name.startswith(("VIEW_MANIFEST", "full22_"))`. This lane's copy of
  # the wave's short-view guard dropped that exclusion, so it counted
  # VIEW_MANIFEST.json as a cell: view=2 against roster=1, and EVERY serving leg
  # of this lane refused itself. It had never fired before because no prior job
  # reached ar_bank_view with a BUILT view — href and hydrate build nothing, and
  # verify_served.py calls make_bank_view.py directly. Caught by the submit-time
  # precheck on the lane's first serving submission, which is what it is for.
  # tests/test_arditi_cell.py pins these prefixes EQUAL to the producer's, so a
  # change there turns a test red instead of re-breaking the lane silently.
  n_view=$(find "$TT_TWO_SI_DEFENSES_DIR" -mindepth 1 -maxdepth 1 \
             ! -name 'VIEW_MANIFEST*' ! -name 'full22_*' 2>/dev/null | wc -l)
  [ "$want" -ge 1 ] || f22_fatal "$AR_ROSTER serves NO cell for $KEY/$VIEW"
  [ "$have" -eq "$want" ] || f22_fatal "$CFG lists $have cells + none but $AR_ROSTER names $want for $KEY/$VIEW — regenerate"
  [ "$n_view" -eq "$want" ] || f22_fatal "arditi bank view $TT_TWO_SI_DEFENSES_DIR holds $n_view entries but $AR_ROSTER names $want"
  f22_log "arditi bank view OK: $TT_TWO_SI_DEFENSES_DIR ($want cell(s) + the same-standup none anchor)"
}

# THE HOOKED-LAYER ASSERTION, from the server's OWN attestation (never the env we set, never the
# config): steering_attestation.json's worker_reports[*].layers is what the workers reported
# installing, and readiness already refused a disagreement between workers. A served object whose
# edit lands anywhere but its one fitted layer is the failure mode this lane exists to exclude.
#
# TWO SCHEMAS SINCE arditi-qwen-all-positions, because the two engines record the fact
# in different places and the LENS schema is not merely absent on a native endpoint — it would be
# read as "no hooks" and the assertion would refuse a healthy server:
#   production  ep["steering_attestation"].worker_reports[*].{layers, oproj_layers,
#               registration_count} — the lens hook-registration record. UNTOUCHED below.
#   native      the RESIDUAL SLOT's records: the payload's `boundaries` (where the add lands) out of
#               the attestation's expected state, cross-checked against what EVERY worker reports
#               having loaded in its slot. There is no oproj notion on this path, and the
#               per-worker count is the slot's, not a registration count.
ar_assert_hook_layers_in(){   # $1 = production | native, $2 = endpoint.json, $3 = the expected block-site layer
  # the ONE call shape every site uses, so a new standup site cannot pick the wrong schema by
  # forgetting a branch. The two bodies are unshared: ar_assert_hook_layers_native (below) is new
  # with the native mode; ar_assert_hook_layers (the lens schema) is unchanged.
  local mode="${1:-}"
  ar_require_mode ar_assert_hook_layers_in "$mode"
  if [ "$mode" = native ]; then ar_assert_hook_layers_native "$2" "$3"; else ar_assert_hook_layers "$2" "$3"; fi
}

ar_assert_hook_layers_native(){   # $1 = endpoint.json, $2 = the expected block-site layer
  local ep="${1:?endpoint.json}" want="${2:?layer}" out
  # $SPY IS THE SERVING VENV AND HOLDS NO `sisteer` (vLLM + its deps, nothing else). This heredoc
  # imports two stdlib-only sisteer modules, so it needs the checkout's src on the path — the same
  # treatment common.sh f22_assert_native_evidence got one line earlier in the job, and one of the
  # two forms every OTHER $SPY heredoc in this family that imports sisteer already uses (a PYTHONPATH
  # prefix, or `cd $REPO_DIR` + sys.path.insert(0, "src") inside the heredoc). Jobs 533195 and
  # 109066 both died HERE — two clusters, two nodes, two filesystems, the identical line — one step
  # after the evidence assertion passed: `ModuleNotFoundError: No module named 'sisteer'` then
  # "HOOK-LAYER ASSERTION FAILED (native):" with an EMPTY record. A family-wide sweep (56 files,
  # every heredoc classified by interpreter, path form and imports) found this to be the ONLY site
  # with neither form; tests/test_arditi_allpos_configs.py now runs that sweep so a third such site
  # cannot wait for a third job to find it.
  # `:+`, not `:?` (the same reasoning as common.sh f22_assert_native_evidence): prepend the
  # checkout's src when the root is known, inherit the caller's path when it is not; an interpreter
  # that can import neither way still fails loudly one line down.
  out=$(PYTHONPATH="${AR_REPO_ROOT:+$AR_REPO_ROOT/src:}${PYTHONPATH:-}" \
    "$SPY" - "$ep" "$want" "$AR_NATIVE_OP" "$AR_NATIVE_POSITION_MODE" "$AR_NATIVE_ADAPTER_OP" <<'PY'
import json, sys
from sisteer.capeval.serving import native_residual as nr
from sisteer.capeval.serving import serving_attestation as sa

ep_path, want, want_op, want_mode, want_adapter_op = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5]
ep = json.load(open(ep_path))
att = sa.read_serving_attestation(ep["serving_attestation"])
rs = att.get(sa.RESIDUAL_SLOT_KEY) or {}
problems, seen = [], []
if want_op != nr.OP_QUALNAME or want_adapter_op != nr.OP_ADD_VECTOR:
    problems.append(f"the shell's constants drifted from native_residual: AR_NATIVE_OP={want_op!r} (OP_QUALNAME {nr.OP_QUALNAME!r}), "
                    f"AR_NATIVE_ADAPTER_OP={want_adapter_op!r} (OP_ADD_VECTOR {nr.OP_ADD_VECTOR!r})")
if not rs.get("armed"):
    print(json.dumps({"expected_layer": want, "problems": problems + ["residual_slot is not armed"]}))
    raise SystemExit(1)
expected = rs.get("expected") or {}
# the KERNEL name (Finding 1): one registered op; the per-adapter `op` below is the PAYLOAD's arithmetic
if expected.get("op_name") != want_op:
    problems.append(f"residual_slot.expected.op_name={expected.get('op_name')!r} != {want_op!r}")
exp = expected.get("adapters") or {}
payloads = {n: b for n, b in exp.items() if b.get("residual_fingerprint")}
if not payloads:
    problems.append("no registered adapter carries a residual payload — nothing is hooked")
for name, b in sorted(payloads.items()):
    bounds = sorted(int(x) for x in (b.get("boundaries") or []))
    layers = sorted(int(x) for x in (b.get("layers") or []))
    seen.append({"adapter": name, "boundaries": bounds, "layers": layers,
                 "op": b.get("op"), "mode": b.get("position_mode"), "lora_int_id": b.get("lora_int_id")})
    # THE BOUNDARY IS WHERE THE EDIT LANDS. For this kind the blob stores the SERVED block-output
    # layer and the exporter carries it through unshifted (boundaries == layers), which is the
    # INVERSE of the static_r convention the production branch above assumes — so the expected
    # layer is compared against the boundary, and the layers are recorded beside it rather than
    # assumed equal.
    if bounds != [want]:
        problems.append(f"adapter {name!r} writes at boundaries {bounds}, expected exactly [{want}]")
    if b.get("op") != want_adapter_op:
        problems.append(f"adapter {name!r}: payload op {b.get('op')!r} != {want_adapter_op!r} (the kernel's name is "
                        f"expected.op_name, asserted above; this is the arithmetic the payload asks of it)")
    if b.get("position_mode") != want_mode:
        problems.append(f"adapter {name!r}: position_mode {b.get('position_mode')!r} != {want_mode!r}")
# and what every WORKER actually loaded, not only what was expected of it
reports = rs.get("worker_reports") or []
if not reports:
    problems.append("residual_slot carries no worker_reports")
for i, rep in enumerate(reports):
    if "op_name" in rep and rep.get("op_name") != want_op:
        problems.append(f"worker {i}: residual_slot.op_name={rep.get('op_name')!r} != {want_op!r}")
    slots = rep.get("slots") or {}
    for name, b in sorted(payloads.items()):
        got = slots.get(str(b.get("lora_int_id")))
        if not isinstance(got, dict):
            problems.append(f"worker {i}: adapter {name!r} (lora_int_id {b.get('lora_int_id')}) has no loaded slot")
            continue
        gb = sorted(int(x) for x in (got.get("boundaries") or []))
        if gb != [want]:
            problems.append(f"worker {i}: adapter {name!r} loaded boundaries {gb}, expected exactly [{want}]")
        if got.get("mode") != want_mode:
            problems.append(f"worker {i}: adapter {name!r} loaded mode {got.get('mode')!r} != {want_mode!r}")
        if got.get("op") != want_adapter_op:
            problems.append(f"worker {i}: adapter {name!r} loaded slot op {got.get('op')!r} != {want_adapter_op!r}")
print(json.dumps({"expected_layer": want, "n_workers": len(reports), "adapters": seen, "problems": problems}))
raise SystemExit(1 if problems else 0)
PY
) || { f22_log "HOOK-LAYER ASSERTION FAILED (native): $out"; return 1; }
  f22_log "hook layers OK (native): $out"
}

ar_assert_hook_layers(){   # $1 = endpoint.json, $2 = the expected block-site layer
  local ep="${1:?endpoint.json}" want="${2:?layer}" out
  out=$("$SPY" - "$ep" "$want" <<'PY'
import json, sys
ep = json.load(open(sys.argv[1])); want = int(sys.argv[2])
att = json.load(open(ep["steering_attestation"]))
reports = att.get("worker_reports") or []
if not reports:
    print(f"REFUSE no worker_reports in {ep['steering_attestation']}"); raise SystemExit(1)
problems, seen = [], []
for i, r in enumerate(reports):
    layers = sorted(int(l) for l in (r.get("layers") or []))
    oproj = sorted(int(l) for l in (r.get("oproj_layers") or []))
    seen.append({"worker": i, "layers": layers, "oproj_layers": oproj,
                 "registration_count": r.get("registration_count")})
    if layers != [want]:
        problems.append(f"worker {i} hooked block layers {layers}, expected exactly [{want}]")
    if oproj:
        problems.append(f"worker {i} additionally hooked o_proj at {oproj} — this lane's object "
                        "is a single additive residual vector and touches no o_proj site")
    if r.get("registration_count") != 1:
        problems.append(f"worker {i} registration_count {r.get('registration_count')} != 1")
print(json.dumps({"expected_layer": want, "reports": seen, "problems": problems}))
raise SystemExit(1 if problems else 0)
PY
) || { f22_log "HOOK-LAYER ASSERTION FAILED: $out"; return 1; }
  f22_log "hook layers OK: $out"
}
