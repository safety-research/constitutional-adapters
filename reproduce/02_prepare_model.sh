#!/bin/bash
# reproduce/02_prepare_model.sh — stage 02: prepare ONE model for the pipeline.
#
#   reproduce/02_prepare_model.sh --model K --step corpora|smoke|pools|memprobe|register|all
#                                 [--smoke-mode tokenizer-only|full|vllm] [--ladder 1,2,4] [--arms psr_ce_all,lora_r4_ce]
#                                 [--fits-config configs/sdfbase_<key>_fits.yaml] [--config F] [--overrides F] [--dry-run] [--help]
#
#   corpora   MODEL-INDEPENDENT, runs once: reassemble the finals of the two corpora's four parts from the committed gzip shards
#             (reproduce/tools/reassemble_corpus.py <corpus> --out $SISTEER_DATA_ROOT/finals/<name>.jsonl --check —
#             absent = printed + marked TODO), fetch the three reference tokenizers' FILES at the
#             corpora/TOKENIZERS.json pins (lib/hf_fetch.sh, one repo per call) and stage them
#             (`corpusgen stage-tokenizers --out $SISTEER_DATA_ROOT/tokenizers --hf-home $CLUSTER_HF_HOME --revision ...`).
#   smoke     the onboarding gate on the FIT checkpoint (scripts/onboard_model_smoke.py --model-config <fits yaml>
#             --mode tokenizer-only|full|vllm; tokenizer-only runs here; full is a GPU srun in the prebuilt FIT venv;
#             vllm is a GPU srun in the SERVING venv — the requirements-hash cu129 venv scripts/sbatch/serving_probe.sbatch
#             builds, bootstrapped on the node by the same recipe — where an absent vllm FAILS, never SKIPs, so DONE.smoke.vllm
#             is written only after a server actually came up);
#             REFUSES when two configured clusters resolve the checkpoint to DIFFERENT snapshots (refs/main vs the pin).
#   pools     build_heldout_split.py -> build_pools.py build -> build_ctl2_pools.py build (SKIPPED when a complete,
#             preflight-PASSED, byte-verified ctl2 build exists — ctl2_pools_reuse.py decide/commit) -> hard-link into
#             $SHARED/vol/pool/<fit key>/ -> train_two_si_objects.py precompute-tokens per cell + held-out — one
#             CPU srun on the primary cluster, every builder with CORPUS_GENERATOR_ROOT=<checkout>/corpus_generator exported.
#   memprobe  the FD 23 fit-memory probe: submit scripts/sbatch/memprobe_fit.sbatch <fit key> <arm> <ladder> per arm
#             through lib/slurm.sh (1 GPU, 01:30:00); when memprobe.<arm>.json already exists on the cluster, FOLD it
#             (make_sdfbase_configs.py --fold-memprobe: the largest fitting micro-batch) and print the FD 23 verdict —
#             exit 5 when micro-batch 1 does not fit (it is reported; the fit lane waits).
#   register  `registry dimensions load` for the model's registry seeds (configs/registry/*.yaml).
#   all       corpora -> smoke -> pools -> memprobe -> register
#   --fits-config  the fits config to use (default: the ONE canonical configs/sdfbase_*_fits.yaml pinning the model's fit checkpoint —
#             seed replicates `_s<N>` are never candidates; lib/common.sh repro_fits_config_for). It must pin the same checkpoint.
#
# Idempotent by completion markers (DONE.<unit> under the grading host's state dir); --dry-run prints every command
# with the config substituted and touches nothing. Every scheduler submission prints `submitted <cluster> <jobid> <unit>`.
# SISTEER_CPU_SRUN (env): the CPU srun prefix for the corpora / pools / tokenizer steps, as 04_fit_objects.sh documents
# (unset = the site's dev/CPU partition line; set, possibly EMPTY, = your wrapper verbatim).
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/lib/common.sh"
[ -f "$HERE/lib/slurm.sh" ] && source "$HERE/lib/slurm.sh"
usage(){ sed -n '2,30p' "$0"; repro_common_help; }
STAGE=02_prepare_model
STEP=all; SMOKE_MODE=tokenizer-only; LADDER=1,2,4; ARMS="psr_ce_all,lora_r4_ce"; FITS_CONFIG=""
repro_parse_common_args "$@"
set -- "${REPRO_ARGS[@]+"${REPRO_ARGS[@]}"}"
while [ $# -gt 0 ]; do
  case "$1" in
    --step) STEP="$2"; shift 2;; --smoke-mode) SMOKE_MODE="$2"; shift 2;;
    --ladder) LADDER="$2"; shift 2;; --arms) ARMS="$2"; shift 2;;
    --fits-config) FITS_CONFIG="${2:?--fits-config needs a path}"; shift 2;; --fits-config=*) FITS_CONFIG="${1#--fits-config=}"; shift;;
    *) echo "REFUSED: unknown argument $1" >&2; usage >&2; exit 2;;
  esac
done
[ "$REPRO_HELP" = 1 ] && { usage; exit 0; }
case "$STEP" in corpora|smoke|pools|memprobe|register|all) ;; *) repro_fatal "--step must be corpora|smoke|pools|memprobe|register|all";; esac
case "$SMOKE_MODE" in tokenizer-only|full|vllm) ;; *) repro_fatal "--smoke-mode must be tokenizer-only|full|vllm";; esac
repro_load_config
repro_require_model
MODEL="$REPRO_MODEL"
SHA="$(repro_head_sha)"
PRIMARY="$SISTEER_PRIMARY_CLUSTER"
want(){ [ "$STEP" = all ] || [ "$STEP" = "$1" ]; }

# --- the model's identity (reproduce/models.yaml, the one table) ------------------------------------------------------------
FIT_REPO="$(repro_model "$MODEL" fit.checkpoint.hf_repo)"; FIT_REV="$(repro_model "$MODEL" fit.checkpoint.revision)"
FIT_MODEL_ID="$(repro_model "$MODEL" fit.checkpoint.model_id)"
SRV_REPO="$(repro_model "$MODEL" serve.checkpoint.hf_repo)"; SRV_REV="$(repro_model "$MODEL" serve.checkpoint.revision)"
RUN_HERE="$(repro_model "$MODEL" run_in_this_work)"
# the fits config whose pins are the fit checkpoint's: lib/common.sh repro_fits_config_for — the ONE canonical configs/sdfbase_*_fits.yaml
# (seed replicates excluded) or the --fits-config named explicitly; the CONFIG is the authority for the pool dir and the sbatch positional
FIT_CFG="$(repro_fits_config_for "$FIT_REPO" "$FIT_REV" "$FITS_CONFIG" "$FIT_MODEL_ID")" \
  || repro_fatal "no canonical fits config for $FIT_REPO@$FIT_REV (reason above; run scripts/sdfbase/make_sdfbase_configs.py, or name one with --fits-config <path>)"
FIT_KEY="$(basename "$FIT_CFG" _fits.yaml)"; FIT_KEY="${FIT_KEY#sdfbase_}"
POOL_KEY="$(sed -n 's#^ *heldout_pool: /vol/pool/\([^/][^/]*\)/pairs_constv2-heldout\.jsonl *$#\1#p' "$REPRO_REPO/$FIT_CFG" | head -1)"
[ -n "$POOL_KEY" ] || repro_fatal "$FIT_CFG names no /vol/pool/<key>/ held-out pool"

# --- per-cluster paths (values with a literal $SHARED are expanded per cluster) ----------------------------------------------
cl_shared(){ cluster_var "$1" CLUSTER_SHARED_ROOT; }
cl_hf(){ repro_expand_shared "$1" "$(cluster_var "$1" CLUSTER_HF_HOME)"; }
cl_checkout(){ echo "$(repro_expand_shared "$1" "$(cluster_var "$1" CLUSTER_EVAL_REPO_ROOT)")/$SHA"; }
cl_python(){ echo "$(repro_expand_shared "$1" "$(cluster_var "$1" CLUSTER_ENVS_DIR)")/$(cluster_fit_env_name "$1")/bin/python"; }   # the prebuilt fit venv (that cluster's name)
cl_data(){ repro_expand_shared "$1" "$SISTEER_DATA_ROOT"; }
cl_state(){ repro_cluster_state_root "$1" "$MODEL"; }
cl_srun(){   # <label> -> the CPU srun prefix (the dev partition when the site names one; never bare work on a login node)
  # SISTEER_CPU_SRUN, when SET (even empty), is the wrapper verbatim — the override 04_fit_objects.sh honours
  if [ -n "${SISTEER_CPU_SRUN+x}" ]; then printf '%s' "$SISTEER_CPU_SRUN"; return; fi
  local l="$1" part qos
  part="$(cluster_var "$l" CLUSTER_DEV_PARTITION)"; [ -n "$part" ] || part="$(cluster_var "$l" CLUSTER_PARTITION)"
  qos="$(cluster_var "$l" CLUSTER_DEV_QOS)"; [ -n "$qos" ] || qos="$(cluster_var "$l" CLUSTER_QOS)"
  # the job name's $USER is expanded on the CLUSTER (the ssh string carries it literally): nothing here names a user
  printf 'srun --partition=%s --qos=%s --cpus-per-task=16 --mem=64G --time=02:00:00 --job-name=D_$USER-02prep' "$part" "$qos"
}
serving_venv_sh(){   # <checkout> <shared root> -> the remote bash that bootstraps the SERVING venv on the node and sets $VENV
  # the lanes' recipe byte for byte (scripts/sbatch/serving_probe.sbatch / full22/common.sh f22_serving_venv): the venv is
  # keyed by the sha256 of configs/capeval/vllm-requirements.txt under /home/$USER/venvs (node-local, like the probe's), so a
  # node that already carries sisteer-petri-vllm-<hash8> reuses it; every $VAR here expands on the NODE (the string is
  # the body of a remote `bash -c`), the two %s are the checkout and the shared root (uv's cache).
  # The import guard REQUIRES peft with vllm and torch: peft==0.20.0 is pinned in the requirements because the probe's
  # HF+PEFT parity check certifies, and a serving venv without it would run a probe that cannot.
  printf 'REQS=%s/configs/capeval/vllm-requirements.txt; [ -s "$REQS" ] || { echo "REFUSED: no serving requirements at $REQS" >&2; exit 3; }; SREQ=$(sha256sum "$REQS" | cut -c1-64); VENV=/home/$USER/venvs/sisteer-petri-vllm-${SREQ:0:8}; LOCKDIR=$(dirname "$VENV"); mkdir -p "$LOCKDIR"; SENT="$LOCKDIR/.$(basename "$VENV").vllm-bootstrap"; ( flock -x 9 || exit 99; if [ -f "$SENT" ] && [ "$(cat "$SENT")" = "${SREQ:0:16}" ] && [ -x "$VENV/bin/python" ]; then echo "serving venv already bootstrapped ($VENV)"; else [ -x "$VENV/bin/python" ] || uv venv --python 3.12 "$VENV" || exit 1; UV_CACHE_DIR=%s/.cache/uv UV_LINK_MODE=copy uv pip install --python "$VENV/bin/python" --index-strategy unsafe-best-match --exclude-newer 2026-08-03T00:00:00Z -r "$REQS" || exit 1; echo "${SREQ:0:16}" > "$SENT"; fi ) 9>"$LOCKDIR/.$(basename "$VENV").uv-sync.lock" || { echo "REFUSED: serving venv bootstrap failed" >&2; exit 3; }; "$VENV/bin/python" -c "import vllm, torch, peft; print(vllm.__version__, torch.__version__, peft.__version__)" || { echo "REFUSED: vllm / torch / peft do not all import from $VENV (peft==0.20.0 is REQUIRED for the serving probe parity check, D55)" >&2; exit 3; }' "$1" "$2"
}
cl_helpers(){ repro_expand_shared "$1" "$(cluster_var "$1" CLUSTER_HELPERS_DIR)"; }   # hf_env.sh's dir
snapshot_dir(){ echo "$(cl_hf "$1")/hub/models--${2//\//--}/snapshots/$3"; }
# the primary cluster's paths, derived ONCE (every remote step below runs there)
CO="$(cl_checkout "$PRIMARY")"; PY="$(cl_python "$PRIMARY")"; DATA="$(cl_data "$PRIMARY")"; HF="$(cl_hf "$PRIMARY")"
SHARED_P="$(cl_shared "$PRIMARY")"; HELPERS="$(cl_helpers "$PRIMARY")"

# =============================================================================================================================
if want corpora; then
  if repro_done shared "$STAGE" corpora; then repro_say "corpora: done (marker) — skipping"; else
    repro_say "corpora (model-independent): finals -> $DATA/finals, tokenizers -> $DATA/tokenizers on $PRIMARY"
    TOOL="$REPRO_REPO/reproduce/tools/reassemble_corpus.py"   # every run verifies shard shas + counts + the final's sha
    corp_ok=1
    for c in constitution_advice constitution_stories control_advice control_stories; do
      cmd="cd $CO && $PY reproduce/tools/reassemble_corpus.py $c --out $DATA/finals/$c.jsonl --check"
      if [ -f "$TOOL" ]; then
        repro_ssh "$PRIMARY" "mkdir -p $DATA/finals && $(cl_srun "$PRIMARY") bash -c '$cmd'" || corp_ok=0
      else
        repro_note "TODO (workstream B: reproduce/tools/reassemble_corpus.py absent on this tree): $cmd"; corp_ok=0
      fi
    done
    # the three reference tokenizers' FILES at the corpora/TOKENIZERS.json pins, one repo per hf_fetch.sh call
    REVS=""
    while read -r name trepo trev; do
      [ -n "$name" ] || continue
      repro_run "$HERE/lib/hf_fetch.sh" --cluster "$PRIMARY" --repo "$trepo" --revision "$trev" --include 'tokenizer*' --include 'special_tokens_map.json' --config "$SISTEER_CONFIG" $( [ "$REPRO_DRY_RUN" = 1 ] && echo --dry-run ) || corp_ok=0
      REVS="$REVS --revision $name=$trev"
    done < <("$(repro_python)" -c 'import json,sys; d=json.load(open(sys.argv[1]))["tokenizers"]; [print(k, v["repo"], v["revision"]) for k, v in d.items()]' "$REPRO_REPO/corpora/TOKENIZERS.json")
    # the corpus generator's CLI stages them (a workspace member: `uv run --directory corpus_generator` resolves to the checkout's
    # venv + the repo's one lock, adding the `corpus` group's packages if that venv lacks them) and verifies
    # repo, revision, the exact file set AND every sha256 against the committed pins (--pins; `uv run --directory corpus_generator`
    # runs from corpus_generator/, so the pins path is the checkout's absolute one)
    repro_ssh "$PRIMARY" "cd $CO && $(cl_srun "$PRIMARY") uv run --directory corpus_generator --frozen corpusgen stage-tokenizers --out $DATA/tokenizers --hf-home $HF --pins $CO/corpora/TOKENIZERS.json$REVS" || corp_ok=0
    if [ "$corp_ok" = 1 ]; then repro_mark_done shared "$STAGE" corpora "finals+tokenizers on $PRIMARY at $DATA"; else
      [ "$REPRO_DRY_RUN" = 1 ] || repro_warn "corpora: not every unit completed — no DONE marker written"
      [ "$REPRO_DRY_RUN" = 1 ] || [ "$STEP" != corpora ] || exit 3
    fi
  fi
fi

# =============================================================================================================================
if want smoke; then
  if repro_done "$MODEL" "$STAGE" "smoke.$SMOKE_MODE"; then repro_say "smoke ($SMOKE_MODE): done (marker) — skipping"; else
    repro_say "smoke: $FIT_REPO@$FIT_REV (fits config $FIT_CFG, key $FIT_KEY) — mode $SMOKE_MODE"
    # the two clusters must resolve the SAME snapshot: refs/main and the pinned dir on every configured cluster
    declare -A RESOLVED=()
    for l in $(cluster_labels); do
      for pair in "$FIT_REPO $FIT_REV fit" "$SRV_REPO $SRV_REV serve"; do
        set -- $pair
        d="$(snapshot_dir "$l" "$1" "$2")"; refs="$(cl_hf "$l")/hub/models--${1//\//--}/refs/main"
        if [ "$REPRO_DRY_RUN" = 1 ]; then repro_ssh "$l" "test -s $d/config.json && echo PINNED || echo ABSENT; cat $refs 2>/dev/null || echo NOREF"; continue; fi
        out="$(repro_ssh "$l" "test -s $d/config.json && echo PINNED || echo ABSENT; cat $refs 2>/dev/null || echo NOREF")" || { repro_warn "$l unreachable — cannot verify the $3 snapshot"; continue; }
        pinned="$(echo "$out" | sed -n 1p)"; ref="$(echo "$out" | sed -n 2p)"
        repro_say "  $l: $3 $1 @ ${2:0:12}: $pinned (refs/main ${ref:0:12})"
        [ "$pinned" = PINNED ] || repro_note "  $l: the pinned $3 snapshot is not cached — 00_setup.sh --what checkpoints --model $MODEL --cluster $l"
        [ "$ref" = NOREF ] || [ "$ref" = "$2" ] || repro_note "  $l: refs/main ($ref) differs from the pin ($2) — the pin wins (the pinned dir is what jobs load)"
        RESOLVED["$3:$l"]="$pinned:$ref"
      done
    done
    if [ "$REPRO_DRY_RUN" != 1 ]; then
      for side in fit serve; do
        refs_seen=""
        for l in $(cluster_labels); do v="${RESOLVED[$side:$l]:-}"; [ -n "$v" ] || continue; r="${v#*:}"; [ "$r" = NOREF ] && continue
          if [ -n "$refs_seen" ] && [ "$refs_seen" != "$r" ]; then repro_fatal "the configured clusters resolve the $side checkpoint to DIFFERENT snapshots ($refs_seen vs $r) — refetch at the pin ($( [ $side = fit ] && echo "$FIT_REV" || echo "$SRV_REV")) before running"; fi
          refs_seen="$r"
        done
      done
    fi
    case "$SMOKE_MODE" in
      tokenizer-only)
        repro_run "$(repro_python)" "$REPRO_REPO/scripts/onboard_model_smoke.py" --model-config "$REPRO_REPO/$FIT_CFG" --mode tokenizer-only \
          && repro_mark_done "$MODEL" "$STAGE" "smoke.$SMOKE_MODE" "$FIT_REPO@$FIT_REV" \
          || { [ "$REPRO_DRY_RUN" = 1 ] || repro_fatal "the tokenizer-only smoke FAILED for $FIT_REPO@$FIT_REV (rows above)"; } ;;
      full|vllm)
        gpu="srun --partition=$(cluster_var "$PRIMARY" CLUSTER_PARTITION) --qos=$(cluster_var "$PRIMARY" CLUSTER_QOS) --gres=gpu:1 --cpus-per-task=16 --mem=200G --time=01:00:00 --job-name=D_\$USER-02smoke"
        if [ "$SMOKE_MODE" = vllm ]; then
          # the vllm tier runs in the SERVING venv (never the fit venv $PY, whose lockfile stack carries no vllm): the node
          # bootstraps / reuses the requirements-hash venv, then the gate runs from it with `vllm` on PATH for launch_server;
          # an absent vllm is a FAIL in --mode vllm, so the DONE marker below is written only after a server came up
          cmd="cd $CO && source $HELPERS/hf_env.sh && $(serving_venv_sh "$CO" "$SHARED_P") && HF_HUB_OFFLINE=1 PYTHONPATH=$CO/src PATH=\$VENV/bin:\$PATH \$VENV/bin/python scripts/onboard_model_smoke.py --model-config $CO/$FIT_CFG --mode vllm"
          repro_note "the vllm tier runs in the serving venv (sisteer-petri-vllm-<sha256(configs/capeval/vllm-requirements.txt)[:8]> under /home/\$USER/venvs on the node) on the native lora path (build_server_launch lora_native=True + the banked zero adapter)"
        else
          cmd="cd $CO && source $HELPERS/hf_env.sh && HF_HUB_OFFLINE=1 PYTHONPATH=$CO/src $PY scripts/onboard_model_smoke.py --model-config $CO/$FIT_CFG --mode $SMOKE_MODE"
        fi
        repro_ssh "$PRIMARY" "$gpu bash -c '$cmd'" \
          && repro_mark_done "$MODEL" "$STAGE" "smoke.$SMOKE_MODE" "$FIT_REPO@$FIT_REV on $PRIMARY" \
          || { [ "$REPRO_DRY_RUN" = 1 ] || repro_fatal "the $SMOKE_MODE smoke FAILED on $PRIMARY (its rows are in the srun output above)"; } ;;
    esac
  fi
fi

# =============================================================================================================================
if want pools; then
  POOLS="$SHARED_P/exp/reproduce/$MODEL/$STAGE/pools"; CTL2="$SHARED_P/exp/reproduce/$MODEL/$STAGE/pools_ctl2"; VOL="$SHARED_P/vol/pool/$POOL_KEY"
  # The DONE.pools marker certifies WHAT it completed, not only that it completed: its note is
  # `<vol> on <cluster> recipe=<digest> budget=<pool_sha256>:<official_tokens> links=hard|copy` — the identities
  # `ctl2_pools_reuse.py identity` read off the volume's CURRENT ctl2 build right after the chain, plus HOW the volume holds
  # that build (hard links, or byte-equal copies where the loop's `ln -f` fell back to `cp -f` across filesystems; a record,
  # not an identity the honour rule compares). The marker is LOCAL to this checkout's state dir; the build
  # it certifies is SHARED ($CTL2.CURRENT + the volume's hard links), and another checkout can rebuild ctl2 under its own recipe
  # and re-point both while this marker still reads as current. So a later run honours the marker only when
  # ALL of these hold on the identity line: a CURRENT build exists (current_dir != none); its recorded recipe and budget are
  # readable (not none); the marker's recipe equals BOTH the recipe this checkout's builder declares (its constants +
  # corpora/TOKENIZERS.json pins, provenance-only leaves excluded — never the commit) AND the one the CURRENT build records;
  # the marker's budget equals BOTH the constv2-mix-full budget the chain last rebuilt AND the one the CURRENT build records;
  # and volume_links=ok or ok:copy — the volume's ctl2 pools + manifests are the SAME FILE (inode) as the CURRENT build's, or
  # byte-equal copies of them (a volume on another filesystem than the build: the loop's `cp -f` fallback and
  # the mark must agree, or such a site can never mark); a copy whose bytes DIFFER (stale) or a link left on a superseded
  # build is a reason. Anything else — a differing recipe or budget on either side, a build that is gone or
  # records no readable identity, a volume link reason, a marker with no identities (written before markers carried them), a
  # failed identity read — is said by name (which side disagreed, both values), the marker is ARCHIVED (mv, never rm) to
  # <state>/02_prepare_model/superseded/<utc to the microsecond>/DONE.pools (a fresh directory per archival, lib/common.sh
  # repro_superseded_dir), and the chain below decides REUSE/BUILD by name (an equal recipe
  # re-links the complete build in minutes; a changed one builds a fresh sibling; a build with no recipe record is refused by
  # name and nothing is marked). The identity read is a stdlib+PyYAML read of two manifests, the pins file and four stats (no
  # tokenizers / transformers import; measured 78 ms), so it runs on the login host through repro_ssh like the smoke step's
  # snapshot probes; the builders stay under the CPU srun. Under --dry-run nothing is read or moved: an existing marker skips
  # as before and the read is only named.
  MARKER="$(repro_state_dir "$MODEL")/$STAGE/DONE.pools"
  IDENTITY="cd $CO && $PY scripts/sdfbase/ctl2_pools_reuse.py identity --root $CTL2 --budget-manifest $POOLS/pairs_constv2-mix-full.manifest.json --volume $VOL"
  pools_field(){ printf '%s\n' "$1" | tr ' ' '\n' | sed -n "s/^$2=//p"; }   # <line> <key> -> the `key=value` token's value (empty when absent)
  pools_skip=0
  if [ -f "$MARKER" ]; then
    if [ "$REPRO_DRY_RUN" = 1 ]; then
      repro_say "pools: done (marker) — dry run: the marker is not validated; skipping"; pools_skip=1
    else
      marker_note="$(cat "$MARKER")"
      m_recipe="$(pools_field "$marker_note" recipe)"; m_budget="$(pools_field "$marker_note" budget)"
      why=""
      if [ -z "$m_recipe" ] || [ -z "$m_budget" ]; then
        why="the marker certifies no recipe/budget identities (written before markers carried them)"
      elif ident="$(repro_ssh "$PRIMARY" "$IDENTITY")"; then
        declared="$(pools_field "$ident" declared_recipe)"; budget="$(pools_field "$ident" budget)"
        cur_dir="$(pools_field "$ident" current_dir)"; rec_recipe="$(pools_field "$ident" recorded_recipe)"
        rec_budget="$(pools_field "$ident" recorded_budget)"; links="$(pools_field "$ident" volume_links)"
        # the declared side first (this checkout changed), then the recorded side (the shared build changed under the marker),
        # then the volume's links (what the fits actually read); the first disagreement is the one named
        if [ -z "$declared" ] || [ -z "$budget" ] || [ -z "$cur_dir" ] || [ -z "$rec_recipe" ] || [ -z "$rec_budget" ] || [ -z "$links" ]; then why="the identity read on $PRIMARY answered an incomplete line ($ident)"
        elif [ "$m_recipe" != "$declared" ]; then why="the marker certifies recipe $m_recipe but this checkout's builder declares $declared (the ctl2 recipe changed since the marker was written)"
        elif [ "$m_budget" != "$budget" ]; then why="the marker certifies budget $m_budget but the current constv2-mix-full budget is $budget"
        elif [ "$cur_dir" = none ]; then why="no CURRENT ctl2 build exists under $CTL2 on $PRIMARY (the build the marker certifies is gone or empty)"
        elif [ "$rec_recipe" = none ]; then why="the CURRENT build at $cur_dir records no readable recipe (the marker certifies recipe $m_recipe; the build cannot say what cut it)"
        elif [ "$rec_budget" = none ]; then why="the CURRENT build at $cur_dir records no readable budget_source (the marker certifies budget $m_budget)"
        elif [ "$m_recipe" != "$rec_recipe" ]; then why="the marker certifies recipe $m_recipe but the CURRENT build $cur_dir records $rec_recipe (another checkout rebuilt ctl2 under a different recipe and re-pointed the shared build since the marker was written)"
        elif [ "$m_budget" != "$rec_budget" ]; then why="the marker certifies budget $m_budget but the CURRENT build $cur_dir records budget $rec_budget (the shared build was cut against another constv2-mix-full budget)"
        elif [ "$links" != ok ] && [ "$links" != ok:copy ]; then why="the volume $VOL on $PRIMARY does not hold the CURRENT build $cur_dir (neither hard links nor byte-equal copies; volume_links=$links)"
        fi
      else
        rc=$?; why="the identity read on $PRIMARY failed (rc $rc; its reason is above)"
      fi
      if [ -z "$why" ]; then
        repro_say "pools: done (marker) — its certified recipe and budget are today's; skipping"; pools_skip=1
      else
        # a FRESH directory per archival (lib/common.sh repro_superseded_dir: a microsecond stamp, reserved by an atomic mkdir, a -NN
        # suffix on a collision) — a second-resolution name had two archivals in one second overwrite each other (gate 2026-10-09)
        arch="$(repro_superseded_dir "$(repro_state_dir "$MODEL")/$STAGE/superseded")" || repro_fatal "could not reserve an archive directory under $(repro_state_dir "$MODEL")/$STAGE/superseded/ for the pools marker $MARKER (nothing moved)"
        repro_say "pools: the DONE marker is not honoured — $why; archived to $arch/DONE.pools (moved, never deleted); the chain decides the ctl2 build by name"
        mv -f "$MARKER" "$arch/DONE.pools" || repro_fatal "could not archive the pools marker $MARKER under $arch/"
      fi
    fi
  fi
  if [ "$pools_skip" != 1 ]; then
    ADV="$DATA/finals/constitution_advice.jsonl"; STO="$DATA/finals/constitution_stories.jsonl"
    CADV="$DATA/finals/control_advice.jsonl"; CSTO="$DATA/finals/control_stories.jsonl"; TOKS="$DATA/tokenizers"
    CELLS="constv2-mix-10m,constv2-mix-full,constv2-heldout"   # the HPO reference cell (trials on mix-10m) + the production cell + the held-out eval cell the `heldout` lane renders (stage-02 run 2026-09-30: `--lanes mixed,heldout` without it dropped the lane and the sidecar chain died on the missing pairs file)
    repro_say "pools: $FIT_KEY -> $VOL on $PRIMARY (cells $CELLS + heldout; ctl2 full + heldout — the ctl2 build is SKIPPED when a complete, preflight-PASSED, byte-verified build exists, D98 i; token sidecars)"
    chain="set -euo pipefail; cd $CO; export CORPUS_GENERATOR_ROOT=$CO/corpus_generator CORPUS_GENERATOR_TOKENIZERS_DIR=$TOKS PYTHONPATH=$CO/src SISTEER_VOL_ROOT=$SHARED_P/vol HF_HUB_OFFLINE=1"
    chain="$chain; source $HELPERS/hf_env.sh"
    chain="$chain; mkdir -p $POOLS $VOL"
    chain="$chain; $PY scripts/sdfbase/build_heldout_split.py --code-root $CO/corpus_generator --advice-final $ADV --stories-final $STO --out $POOLS/heldout_split.json"
    chain="$chain; $PY scripts/sdfbase/build_pools.py build --code-root $CO/corpus_generator --split $POOLS/heldout_split.json --advice-final $ADV --stories-final $STO --tokenizers-dir $TOKS --out $POOLS --lanes mixed,heldout --cells $CELLS"
    chain="$chain; $PY scripts/sdfbase/build_pools.py preflight --pools $POOLS"
    # the ctl2 build is NOT re-runnable in place (build_ctl2_pools.py refuses a non-empty --out by contract) and the builder is
    # seeded (a rebuild reproduces the recorded bytes), so ctl2_pools_reuse.py decides in ONE line the chain reads with
    # `read -r`: REUSE a complete, preflight-PASSED, byte-verified build (the pointer $CTL2.CURRENT, else $CTL2 itself); BUILD
    # into $CTL2 when nothing is there, or into a fresh stamped sibling when an INCOMPLETE build is (left in place, never rebuilt
    # over) — then `commit` makes the finished dir CURRENT and the link loop below re-points the volume's hard links.
    # Both calls take the mix-full manifest the chain JUST rebuilt: a recorded build cut against a different constv2-mix-full
    # budget (its budget_source's pool_sha256 / official_tokens, never the manifest file's sha, which carries the checkout's
    # commit) is refused by name, and a seeded rebuild beside a partial must reproduce the partial's recorded pool sha256s
    # before commit re-points anything. The recorded build's RECIPE (its manifests' `recipe`: version,
    # control-input shas, id prefix, salts, split, tokenizer pins — never the checkout's commit) must equal today's builder's
    # too, read from this checkout (no argument): a different recipe BUILDs a fresh stamped sibling — the old selection is
    # left in place and NOT linked as current — and no recipe record refuses by name.
    # Every \$VAR here expands on the NODE; the case branch uses `;` so `set -e` stops at the first failing builder (an `&&` list
    # would swallow it). No single quote anywhere: the chain is the body of a remote single-quoted `bash -c`.
    chain="$chain; CTL2_DECISION=\$($PY scripts/sdfbase/ctl2_pools_reuse.py decide --root $CTL2 --budget-manifest $POOLS/pairs_constv2-mix-full.manifest.json) || exit 3"
    chain="$chain; read -r CTL2_MODE CTL2_DIR <<<\"\$CTL2_DECISION\""
    chain="$chain; echo \"ctl2: \$CTL2_MODE \$CTL2_DIR\""
    chain="$chain; case \"\$CTL2_MODE\" in REUSE) echo \"ctl2: reusing the complete, preflight-PASSED, byte-verified build at \$CTL2_DIR (D98 i)\";; BUILD) mkdir -p \"\$CTL2_DIR\"; $PY scripts/sdfbase/build_ctl2_pools.py build --code-root $CO/corpus_generator --control-advice $CADV --control-stories $CSTO --budget-manifest $POOLS/pairs_constv2-mix-full.manifest.json --out \"\$CTL2_DIR\" --split-out $POOLS/ctl2_heldout_split.json --tokenizers-dir $TOKS; $PY scripts/sdfbase/build_pools.py preflight --pools \"\$CTL2_DIR\"; $PY scripts/sdfbase/ctl2_pools_reuse.py commit --root $CTL2 --dir \"\$CTL2_DIR\" --budget-manifest $POOLS/pairs_constv2-mix-full.manifest.json;; *) echo \"REFUSED: ctl2_pools_reuse.py decide answered \$CTL2_DECISION\" >&2; exit 3;; esac"
    # hard-link every pool + manifest into the volume layout the fits read (/vol/pool/<key>/ under SISTEER_VOL_ROOT); `ln -f`
    # re-points an existing link at the CURRENT ctl2 dir's inodes; on a volume on ANOTHER filesystem (EXDEV) the `cp -f`
    # fallback leaves byte-equal copies instead. The loop runs after the case on EVERY chain run, REUSE included, so a stale
    # copy (the build rebuilt since it was taken) is refreshed to the CURRENT dir's bytes; the mark below records which form
    # the volume holds (links=hard / links=copy) and accepts identity's volume_links=ok / ok:copy alike
    chain="$chain; for f in $POOLS/pairs_*.jsonl $POOLS/pairs_*.manifest.json \$CTL2_DIR/pairs_*.jsonl \$CTL2_DIR/pairs_*.manifest.json; do ln -f \"\$f\" $VOL/ 2>/dev/null || cp -f \"\$f\" $VOL/; done"
    # the token sidecars: each training cell + both held-out pools (HELDOUT_POOL_SIS)
    for si in constv2-mix-10m constv2-mix-full constv2-heldout constv2ctl2-full constv2ctl2-heldout; do
      chain="$chain; $PY scripts/train_two_si_objects.py precompute-tokens --config $CO/$FIT_CFG --si $si --pairs $VOL/pairs_$si.jsonl"
    done
    chain="$chain; ls -la $VOL"
    if repro_ssh "$PRIMARY" "$(cl_srun "$PRIMARY") bash -c '$chain'"; then
      # the marker records the identities the volume ACTUALLY holds — the CURRENT ctl2 dir's recorded recipe digest and budget,
      # read back after the chain; `commit` made them equal to today's declared ones, so a disagreement (or `none`) means a
      # tampered or foreign build: refused by name, no marker (the next run re-decides by name)
      if [ "$REPRO_DRY_RUN" = 1 ]; then
        repro_note "dry run: after the chain the volume's identities are read on $PRIMARY ($IDENTITY) and the DONE marker records them as recipe= and budget= tokens plus links=hard|copy (how the volume holds the build)"
        repro_mark_done "$MODEL" "$STAGE" pools "$VOL on $PRIMARY"
      else
        ident="$(repro_ssh "$PRIMARY" "$IDENTITY")" || repro_fatal "the pools chain finished on $PRIMARY but the identity read failed (its reason is above); no DONE marker written — re-run --step pools (the chain REUSEs the committed build)"
        declared="$(pools_field "$ident" declared_recipe)"; budget="$(pools_field "$ident" budget)"
        rec_recipe="$(pools_field "$ident" recorded_recipe)"; rec_budget="$(pools_field "$ident" recorded_budget)"
        links="$(pools_field "$ident" volume_links)"
        [ -n "$rec_recipe" ] && [ "$rec_recipe" != none ] && [ -n "$rec_budget" ] && [ "$rec_budget" != none ] \
          || repro_fatal "the pools chain finished but the CURRENT ctl2 build on $PRIMARY records no recipe/budget identity ($ident); no DONE marker written"
        [ "$rec_recipe" = "$declared" ] && [ "$rec_budget" = "$budget" ] \
          || repro_fatal "the pools chain finished but the CURRENT ctl2 build's recorded identities (recipe $rec_recipe, budget $rec_budget) are not this checkout's declared ones (recipe $declared, budget $budget) — impossible after commit unless the build was tampered or ran from another checkout; no DONE marker written; inspect $CTL2.CURRENT on $PRIMARY"
        # the link loop just ran: the volume must now hold the CURRENT build's files — as hard links (ok -> links=hard) or, on
        # a volume on another filesystem where the loop's ln -f fell back to cp -f, as byte-equal copies (ok:copy ->
        # links=copy); a stale copy, or a link still on another build, means the loop failed — otherwise no marker
        case "$links" in ok) held=hard;; ok:copy) held=copy;;
          *) repro_fatal "the pools chain finished but the volume $VOL on $PRIMARY does not hold the CURRENT ctl2 build (neither hard links nor byte-equal copies; volume_links=$links) — the chain's link loop just ran, so neither its ln -f nor its cp -f fallback landed; no DONE marker written; inspect $VOL and $CTL2.CURRENT on $PRIMARY";;
        esac
        repro_mark_done "$MODEL" "$STAGE" pools "$VOL on $PRIMARY recipe=$rec_recipe budget=$rec_budget links=$held"
      fi
      # the other configured clusters read the same pools: copy the volume dir
      for l in $(cluster_labels); do [ "$l" = "$PRIMARY" ] && continue
        repro_note "copy $VOL -> $l:$(cl_shared "$l")/vol/pool/$POOL_KEY/ before a twin fit runs there (rsync between the clusters' login hosts)"
      done
    else
      [ "$REPRO_DRY_RUN" = 1 ] || repro_fatal "the pools chain failed on $PRIMARY (its output is above); nothing marked done"
    fi
  fi
fi

# =============================================================================================================================
if want memprobe; then
  STATE="$(cl_state "$PRIMARY")/$STAGE"; VOL="$SHARED_P/vol/pool/$POOL_KEY"
  FD23_RC=0
  for arm in ${ARMS//,/ }; do
    case "$arm" in psr_ce_all|lora_r4_ce) ;; *) repro_fatal "--arms: unknown arm $arm";; esac
    unit="memprobe.$arm"
    present=0
    if [ "$REPRO_DRY_RUN" != 1 ]; then
      repro_ssh "$PRIMARY" "test -s $STATE/memprobe.$arm.json" && present=1
    fi
    if [ "$present" = 1 ]; then
      tmp="$(mktemp)"; repro_ssh "$PRIMARY" "cat $STATE/memprobe.$arm.json" > "$tmp" || repro_fatal "could not read the memprobe record"   # the ONE remote reader
      fold="$(repro_state_dir "$MODEL")/$STAGE/fold_memprobe.$arm.json"; mkdir -p "$(dirname "$fold")"
      rm -f "$fold"   # never fold a STALE fold file left by an earlier run after a DOES-NOT-FIT verdict
      # memprobe_fold.py (2026-10-02, fits 831981/831986 OOM): the record folds ONLY when it measured the architecture the config selects
      # NOW (object_kwargs + model) — a stale record (the paper's rank-4 o_proj probe under the picked rank-16 MLP set) exits 6 and is
      # archived + re-probed below; the folded micro-batch is the largest rung that fits WITH headroom (0.85 x the card), never "it survived once"
      # the fit cluster's card class (site CLUSTER_GPU_NAME): a record probed on another class is refused by name (exit 7)
      gpu_expect=(); gname="$(cluster_var "$PRIMARY" CLUSTER_GPU_NAME)"; [ -z "$gname" ] || gpu_expect=(--expect-gpu-name "$gname")
      frc=0; "$(repro_python)" "$REPRO_REPO/scripts/sdfbase/memprobe_fold.py" "$tmp" --config "$REPRO_REPO/$FIT_CFG" --fit-key "$FIT_KEY" --arm "$arm" --out "$fold" "${gpu_expect[@]}" || frc=$?
      case "$frc" in
        0) ;;
        6) # fail-closed archive (lib/common.sh memprobe_archive_stale): unique destination, mv -n, sha256 read back, live name absent —
           # ONLY then the marker is cleared and the arm re-probed; a failed archive leaves everything in place and refuses
           dest="$(memprobe_archive_stale "$PRIMARY" "$STATE" "$arm")" \
             || repro_fatal "could not archive the stale memprobe record on $PRIMARY (above) — nothing cleared, nothing submitted"
           repro_say "memprobe $arm: the record on $PRIMARY was STALE for the current selection (above) — archived as $dest (sha256 read back); re-probing as a new unit"
           rm -f "$(repro_state_dir "$MODEL")/$STAGE/DONE.$unit"      # the marker belonged to the stale record
           present=0 ;;
        5) FD23_RC=5 ;;
        7) repro_fatal "memprobe $arm: the record on $PRIMARY was probed on a different card class than this cluster declares (above) — not folded; probe on the fit cluster" ;;
        *) repro_fatal "memprobe_fold.py failed (rc $frc) on $STATE/memprobe.$arm.json" ;;
      esac
      if [ "$frc" = 0 ] && [ -s "$fold" ]; then
        repro_run "$(repro_python)" "$REPRO_REPO/scripts/sdfbase/make_sdfbase_configs.py" --fold-memprobe "$fold" --models "$FIT_KEY" \
          && repro_mark_done "$MODEL" "$STAGE" "$unit" "folded from $STATE/memprobe.$arm.json" \
          && repro_note "commit configs/sdfbase_folds.json + configs/sdfbase_${FIT_KEY}_fits.yaml (the fit's expect_sha moved); the FD 23 report is $STATE/memprobe_report.$arm.md (mirrored under the job's log)"
      fi
      [ "$present" = 1 ] && continue
    fi
    # a DONE marker stands only for a FRESH record (a stale one cleared it above); a marker with no record on the cluster is not done
    if repro_done "$MODEL" "$STAGE" "$unit" && [ "$REPRO_DRY_RUN" = 1 ]; then repro_say "memprobe $arm: done (marker) — skipping"; continue; fi
    if repro_done "$MODEL" "$STAGE" "$unit"; then repro_warn "memprobe $arm: DONE marker without a record on $PRIMARY — treating as not done, re-probing"; rm -f "$(repro_state_dir "$MODEL")/$STAGE/DONE.$unit"; fi
    # submit the probe (1 GPU, 01:30:00) through lib/slurm.sh — absent lib: the exact sbatch line under --dry-run, a refusal live.
    # `--registers none`: the memprobe writes only its own record dir (no pending file, no spool record, no registry row), so a dead
    # memprobe's --resubmit is proven at sacct-terminal alone (the fenced proof's ingest watermark is not its business)
    # the memprobe UNIT is keyed on the selection it probes (memprobe_fold.py --fingerprint: rank + module set + expert map + seq_len; `psr-L<seq>`
    # for the ReLU arm): a new selection is a NEW unit through the funnel's first-submit reservation, and the unit of the archived record stays
    # bound to its completed job (a completed unit is never re-minted)
    fp="$("$(repro_python)" "$REPRO_REPO/scripts/sdfbase/memprobe_fold.py" --fingerprint --config "$REPRO_REPO/$FIT_CFG" --arm "$arm")" \
      || repro_fatal "memprobe $arm: no selection fingerprint (memprobe_fold.py --fingerprint failed above)"
    funit="memprobe.$arm.$fp"; args="$FIT_KEY $arm $LADDER"; name="memprobe-$MODEL-$arm-$fp"; lwid="$MODEL/$STAGE/memprobe/$arm/$fp"
    # the pinned sha must be a verified checkout on the cluster before sbatch, as the fit path does (lib/slurm.sh slurm_mirror_clusters;
    # MIRRORED_SHA short-circuits a sha this invocation already mirrored)
    SHA="$(repro_head_sha)"; [ "${MIRRORED_SHA:-}" = "$SHA" ] || { slurm_mirror_clusters "$SHA" "$PRIMARY"; MIRRORED_SHA="$SHA"; }
    env_prefix="MEMPROBE_OUT_DIR=$STATE MEMPROBE_PAIRS=$VOL/pairs_constv2-mix-full.jsonl"
    if declare -F slurm_submit >/dev/null 2>&1; then
      slurm_submit --unit "$funit" --lwid "$lwid" --script scripts/sbatch/memprobe_fit.sbatch --name "$name" --args "$args" \
                   --clusters "$PRIMARY" --kind fit --registers none --time 01:30:00 --env "$env_prefix" \
        || { [ "$REPRO_DRY_RUN" = 1 ] || repro_fatal "memprobe $arm: submission failed"; }
    else
      line="cd $CO && $env_prefix SDFBASE_FIT_ENV_NAME=$(cluster_fit_env_name "$PRIMARY") sbatch --qos=$(cluster_var "$PRIMARY" CLUSTER_QOS) --partition=$(cluster_var "$PRIMARY" CLUSTER_PARTITION) --time=01:30:00 -J $name --comment=sha:$SHA scripts/sbatch/memprobe_fit.sbatch $args"
      if [ "$REPRO_DRY_RUN" = 1 ]; then echo "DRY-RUN: ssh $(cluster_var "$PRIMARY" CLUSTER_SSH) '$line'"; else
        echo "REFUSED: reproduce/lib/slurm.sh not present — would submit: $line" >&2; exit 2
      fi
    fi
    repro_note "memprobe $arm submitted; re-run --step memprobe when the job is DONE to fold memprobe.$arm.json (the watcher's terminal event)"
  done
  [ "$FD23_RC" = 0 ] || { repro_warn "FD 23: a probe reports micro-batch 1 DOES NOT FIT — exiting 5 so the executor notifies the user"; exit 5; }
fi

# =============================================================================================================================
if want register; then
  if repro_done "$MODEL" "$STAGE" register; then repro_say "register: done (marker) — skipping"; else
    ROOT="$(eval echo "$SISTEER_REGISTRY_ROOT")"
    repro_say "register: the dimension seeds (configs/registry/*.yaml — the $MODEL rows: model x2, legacy_spellings, decode_policy, instruments, si_texts) into $ROOT"
    repro_run "$(repro_python)" -m sisteer.registry.cli --root "$ROOT" dimensions load --seeds "$REPRO_REPO/configs/registry" \
      && { repro_registry_id model "$FIT_MODEL_ID"; repro_registry_id model "$(repro_model "$MODEL" serve.checkpoint.model_id)"; repro_mark_done "$MODEL" "$STAGE" register "$ROOT"; } \
      || { [ "$REPRO_DRY_RUN" = 1 ] || repro_fatal "registry dimensions load refused (its reason is above)"; }
  fi
fi
[ "$RUN_HERE" = true ] || repro_note "$MODEL is configured but NOT run in this work (run_in_this_work: $RUN_HERE) — every later stage answers --dry-run only"
repro_say "02_prepare_model done (model=$MODEL step=$STEP fit_key=$FIT_KEY)"
