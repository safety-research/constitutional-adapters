#!/bin/bash
# reproduce/01_make_corpora.sh — stage 01: the two synthetic corpora (constitution + control), through the
# corpus-generation pipeline under corpus_generator/.
#
# The paper's four finals are COMMITTED as shards under corpora/ (stage 02 --step corpora materialises them); this
# stage is how they were made and how a new sample would be generated. Generation runs on the Anthropic Batch API with
# claude-opus-5 and a re-run yields a NEW sample, so this script is documented and offline-validated (--setup, --test,
# --preflight, --dry-run) and its generation phases were never run in the companion-repo work.
#
#   01_make_corpora.sh --setup                      install the corpus generator into the harness venv (uv sync --frozen --inexact --group corpus)
#   01_make_corpora.sh --test                       the generator's offline suite in that env (fake Batch client; no keys)
#   01_make_corpora.sh --preflight [--corpus ...]   stage the pinned reference tokenizers + `corpusgen preflight` per config
#   01_make_corpora.sh --corpus constitution|control|all --phase advice|stories|filter|register-edit|all [--dry-run]
#
# `--corpus` only SELECTS a family; nothing generates without `--phase`, the explicit generation request (so `--preflight
# --corpus X` never falls through into `corpusgen run`, and an env-mode user without an Anthropic key still runs the offline
# preflight). `--preflight --corpus X` preflights that family's config and stops; `--corpus X` alone is refused with the two
# things it could have asked for.
#
# Secrets (lib/secrets.sh): SISTEER_SECRETS_MODE=env -> the ONE ANTHROPIC_API_KEY for every pool; =op -> the `batch` item
# (Batch API) / `lowprio` (smoke, live edits) through the secret-manager wrapper, named by SISTEER_OP_ITEM_ANTHROPIC_*;
# =wif -> NO key: the generator's client federates (WorkloadIdentityCredentials over the pool's ids under
# $SISTEER_WIF_ENV_DIR — the batch pool's for generation, the lowprio pool's for a live register-edit run), and a generation
# phase is refused HERE, by file name, when the pool's ids file is absent or empty. The generator's client reads the same
# switch; no key is ever on argv, in a file under the tree or in a log. The preflight's HF token resolves through op under wif
# (secrets.sh secrets_hf_default_source; federation covers the Claude API only).
#
# Coordination: when the site names a channel for its shared API pools (SISTEER_COORDINATION_CHANNEL in config.env), every
# generation phase prints a ready-to-post line for it first ("<N> <model> batch <duration>", plus corpusgen's own request
# accounting) and PROCEEDS — posting is never a gate. Unset or empty prints no such line. Batch API traffic is a separate
# pool from live concurrents.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/lib/common.sh"
source "$HERE/lib/secrets.sh"

CORPUS_DIR="$REPRO_REPO/corpus_generator"
PINS="$REPRO_REPO/corpora/TOKENIZERS.json"
MODEL_ID="claude-opus-5"                       # EngineConfig.model's DEFAULT: the name printed only until --setup has put the generator in the venv
                                               # (a built venv reads each config's own engine.model — stories-regen runs claude-opus-4-8)
DO_SETUP=0; DO_TEST=0; DO_PREFLIGHT=0; DO_GENERATE=0; CORPUS=""; PHASE=""; ENGINE="live"
repro_parse_common_args "$@"
set -- "${REPRO_ARGS[@]+"${REPRO_ARGS[@]}"}"
while [ $# -gt 0 ]; do
  case "$1" in
    --setup) DO_SETUP=1; shift;;
    --test) DO_TEST=1; shift;;
    --preflight) DO_PREFLIGHT=1; shift;;
    --corpus) CORPUS="${2:?--corpus needs constitution|control|all}"; shift 2;;
    --corpus=*) CORPUS="${1#--corpus=}"; shift;;
    --phase) PHASE="${2:?--phase needs advice|stories|filter|register-edit|all}"; DO_GENERATE=1; shift 2;;
    --phase=*) PHASE="${1#--phase=}"; DO_GENERATE=1; shift;;
    --edit-engine) ENGINE="${2:?--edit-engine needs live|batch}"; shift 2;;
    *) repro_fatal "unknown argument '$1' (see --help)";;
  esac
done

usage(){
  cat <<H
01_make_corpora.sh — generate the two synthetic corpora with the corpus_generator pipeline (corpus_generator/).

  --setup                 uv sync --frozen --inexact --group corpus   (the generator joins the harness venv from the repo's one lock;
                          --inexact adds to whatever 00_setup.sh built and removes nothing)
  --test                  run the generator's offline test suite in that env
  --preflight             stage the three pinned reference tokenizers (corpora/TOKENIZERS.json) into corpus_generator/data/tokenizers
                          from \$HF_HOME and run \`corpusgen preflight\` for every selected config (offline; no API call)
  --corpus <c>            constitution | control | all   (which corpus family --preflight / --phase act on; selects only)
  --phase <p>             GENERATE: run phase p of --corpus <c> — advice | stories | filter | register-edit | all
                          (the explicit generation request; without it --corpus never generates, and --preflight --corpus <c>
                          preflights that family and stops)
                            advice, stories  : \`corpusgen run --corpus <p>\` on the Batch API (prints the coordination line
                                               first when the site names a channel: SISTEER_COORDINATION_CHANNEL)
                            filter           : offline filter chain -> finals (+ assemble-v2 for constitution: v1 + the two
                                               regen runs - the recorded drops; + import-scan for control)
                            register-edit    : control only — the r2 register-matching edit pass (plan / run / assemble)
  --edit-engine live|batch  engine for the register-edit run (default live: the lowprio live runner, per the pass's design)
$(repro_common_help)
Corpus configs (corpus_generator/configs/): constitution = production.yaml (v1 run) + production-v2.yaml (v2 assembly) with the regen
runs advice-regen.yaml / stories-regen.yaml; control = control-scale.yaml (v1 run) + control-scale-r2.yaml (r2 edit pass).
Outputs land under corpus_generator/data/ (gitignored): the paper's finals are final-production-20260831-v2.jsonl (advice, stories)
and control/final-control-scale-20260903-r2-{advice,stories}.jsonl — the files committed as shards under corpora/.
H
}
[ "$REPRO_HELP" = 1 ] && { usage; exit 0; }
case "$PHASE" in ""|advice|stories|filter|register-edit|all) ;; *) repro_fatal "--phase must be advice|stories|filter|register-edit|all";; esac
case "$ENGINE" in live|batch) ;; *) repro_fatal "--edit-engine must be live|batch";; esac
case "$CORPUS" in ""|constitution|control|all) ;; *) repro_fatal "--corpus must be constitution|control|all";; esac
if [ "$DO_SETUP$DO_TEST$DO_PREFLIGHT$DO_GENERATE" = 0000 ]; then
  usage >&2
  [ -z "$CORPUS" ] || repro_fatal "--corpus $CORPUS selects a family but nothing was asked of it: pass --preflight (offline) or --phase advice|stories|filter|register-edit|all (generation on the Batch API)"
  repro_fatal "nothing to do: pass --setup, --test, --preflight, or --corpus <c> --phase <p>"
fi
[ "$DO_GENERATE" = 0 ] || [ -n "$CORPUS" ] || repro_fatal "--phase $PHASE needs --corpus constitution|control|all"

repro_load_config
[ -d "$CORPUS_DIR/src/corpus_generator" ] || repro_fatal "corpus generator missing at $CORPUS_DIR"
command -v uv >/dev/null 2>&1 || repro_fatal "uv is required (the corpus generator runs from the repo's uv lock)"
# corpus_generator/ is a workspace member: `uv run` from its directory resolves to the REPO venv + lock (adds only what the
# generator still lacks, removes nothing), so the harness venv is the generator's interpreter.
UVRUN=(uv run --directory "$CORPUS_DIR" --frozen)
CORPUSGEN=("${UVRUN[@]}" corpusgen)
# the generator's client reads the same secrets switch and item names as the front door (lib/secrets.sh)
export SISTEER_SECRETS_MODE SISTEER_OP_ITEM_ANTHROPIC_BATCH SISTEER_OP_ITEM_ANTHROPIC_LOWPRIO SISTEER_OP_ITEM_ANTHROPIC_HIGHPRIO
if [ "$SISTEER_SECRETS_MODE" = env ] && [ "$DO_GENERATE" = 1 ] && [ "$PHASE" != filter ] && [ "${REPRO_DRY_RUN:-0}" != 1 ]; then
  [ -n "${ANTHROPIC_API_KEY:-}" ] || repro_fatal "SISTEER_SECRETS_MODE=env: ANTHROPIC_API_KEY must be set for a generation phase"
fi
if [ "$SISTEER_SECRETS_MODE" = wif ] && [ "$DO_GENERATE" = 1 ] && [ "$PHASE" != filter ] && [ "${REPRO_DRY_RUN:-0}" != 1 ]; then
  # wif: no key to check — the credential label's federation ids file must exist and be non-empty (the generator's
  # client validates the ids themselves and refuses by name); advice / stories / the regen runs use the `batch` credential, a
  # live register-edit run the `lowprio` credential (its batch engine the `batch` one). Refused here, before any corpusgen spawn.
  WIF_DIR="${SISTEER_WIF_ENV_DIR:-$HOME/.anthropic}"
  # <file stem> <pool>: the pool's ids file is wif-<stem>.env, the stem the generator's client uses (corpus_generator.client.WIF_FILES)
  wif_ids_present(){ local f="$WIF_DIR/wif-$1.env"; [ -s "$f" ] || repro_fatal "SISTEER_SECRETS_MODE=wif: the $2 pool's federation ids file $f is absent or empty — generation phase $PHASE needs it"; }
  case "$PHASE" in advice|stories|all) wif_ids_present batch batch;; esac
  if [ "$PHASE" = register-edit ] || [ "$PHASE" = all ]; then
    if [ "$ENGINE" = live ]; then wif_ids_present loprio lowprio; else wif_ids_present batch batch; fi
  fi
fi

# ---------------------------------------------------------------- helpers
cfg(){ echo "$CORPUS_DIR/configs/$1"; }
declare -A CORPUSGEN_PLAN=()   # config path -> "advice stories all engine_model edit_concurrency"; one venv spawn per config
corpusgen_plan_numbers(){   # <config> -> the DAG-expanded request totals + the model and edit concurrency the CONFIG itself declares
  #                     (offline arithmetic in the harness venv); prints an empty line until --setup has put the generator there
  local c="$1"
  if [ -z "${CORPUSGEN_PLAN[$c]+x}" ]; then
    CORPUSGEN_PLAN[$c]=""
    if "$(repro_python)" -c 'import corpus_generator' 2>/dev/null; then
      CORPUSGEN_PLAN[$c]="$("$(repro_python)" - "$c" <<'PY' 2>/dev/null || true
import sys
from pathlib import Path
from corpus_generator.config import RunConfig
from corpus_generator.pipeline import dag_expanded_totals
cfg = RunConfig.load(Path(sys.argv[1])); t = dag_expanded_totals(cfg)
print(t["advice_requests"], t["stories_requests"], t["total"], cfg.engine.model, getattr(cfg.register_edit, "max_concurrency", ""))
PY
)"
    fi
  fi
  printf '%s\n' "${CORPUSGEN_PLAN[$c]}"
}
coordination_line(){   # <config> <corpus advice|stories|all> [note] : the terse `<N> <model> batch <duration>` line + corpusgen's accounting,
  # for the site's coordination channel; prints nothing when the site names none (SISTEER_COORDINATION_CHANNEL unset or empty)
  local config="$1" corpus="$2" note="${3:-}" adv sto all model conc n="" ch="${SISTEER_COORDINATION_CHANNEL:-}"
  [ -n "$ch" ] || return 0
  read -r adv sto all model conc <<<"$(corpusgen_plan_numbers "$config")"
  case "$corpus" in advice) n="$adv";; stories) n="$sto";; all) n="$all";; esac
  echo "--- ready-to-post for $ch (Batch API: a separate pool, not live concurrents) ---"
  if [ -z "$n" ]; then
    echo "<N> $MODEL_ID batch ~24h    (N = the DAG-expanded request total; run --setup first to have it computed here, or read it from \`corpusgen run\`'s own line)"
  else
    printf '%s %s batch ~24h    (~%s Batch API requests, corpus_generator %s, corpus=%s%s)\n' "$n" "$model" "$n" "$(basename "$config" .yaml)" "$corpus" "${note:+; $note}"
  fi
  echo "--- (posting is the user's job; proceeding now) ---"
}
tokenizer_repos(){   # -> one `<name> <repo> <revision>` line per reference tokenizer, from corpora/TOKENIZERS.json
  "$(repro_python)" -c 'import json,sys; d=json.load(open(sys.argv[1])); [print(n, t["repo"], t["revision"]) for n,t in d["tokenizers"].items()]' "$PINS"
}

# ---------------------------------------------------------------- --setup / --test
if [ "$DO_SETUP" = 1 ]; then
  repro_say "stage 01: installing the corpus generator into the harness venv (uv sync --frozen --inexact --group corpus; the repo's one uv.lock)"
  repro_run uv sync --frozen --inexact --group corpus --directory "$REPRO_REPO"
fi
if [ "$DO_TEST" = 1 ]; then
  repro_say "stage 01: the generator's offline test suite (no network, no keys)"
  repro_run env -u PYTHONPATH "${UVRUN[@]}" pytest tests -q
fi

# ---------------------------------------------------------------- --preflight
if [ "$DO_PREFLIGHT" = 1 ]; then
  HFH="${HF_HOME:-$HOME/.cache/huggingface}"
  repro_say "stage 01 preflight: the three pinned reference tokenizers from corpora/TOKENIZERS.json -> corpus_generator/data/tokenizers (HF cache: $HFH)"
  # repro_run_sh runs its string in a child `bash -c`: the secrets helpers must be exported as FUNCTIONS to be callable
  # there (the config's SISTEER_* variables already are, through repro_load_config's `set -a`); without this the
  # `$(secret_hf …)` below is "command not found", swallowed by its `|| true`, and a gated repo is fetched tokenless
  export -f secret_hf _secret_hf_from secrets_mode secrets_op_bin
  while read -r name repo rev; do
    # hf download reads HF_TOKEN from the environment of this one command (secrets.sh; never argv); public repos need none
    repro_run_sh "HF_TOKEN=\"\$(secret_hf 2>/dev/null || true)\" HF_HOME=\"$HFH\" hf download $repo --revision $rev --include 'tokenizer*' --include special_tokens_map.json >/dev/null"
  done < <(tokenizer_repos)
  # --pins: the committed TOKENIZERS.json pins every family's repo + revision AND every file's sha256; the staging refuses a
  # cached snapshot under the right revision path with other content
  repro_run "${CORPUSGEN[@]}" stage-tokenizers --out "$CORPUS_DIR/data/tokenizers" --hf-home "$HFH" --pins "$PINS"
  case "${CORPUS:-all}" in constitution) PREFLIGHT_CFGS=(production.yaml);; control) PREFLIGHT_CFGS=(control-scale.yaml);;
                          *) PREFLIGHT_CFGS=(production.yaml control-scale.yaml);; esac
  for c in "${PREFLIGHT_CFGS[@]}"; do
    repro_run "${CORPUSGEN[@]}" preflight --config "$(cfg "$c")"
  done
fi

# ---------------------------------------------------------------- generation phases
want(){ [ "$PHASE" = all ] || [ "$PHASE" = "$1" ]; }
run_constitution(){
  local prod; prod="$(cfg production.yaml)"
  for c in advice stories; do
    if want "$c"; then
      repro_say "constitution/$c: corpusgen run (Batch API, $MODEL_ID) — resumable; the coordination line first"
      coordination_line "$prod" "$c"
      repro_run "${CORPUSGEN[@]}" run --config "$prod" --corpus "$c"
    fi
  done
  if want filter; then
    repro_say "constitution/filter: offline filter chain -> data/{advice,stories}/final-production-20260831.jsonl (v1)"
    repro_run "${CORPUSGEN[@]}" filter --config "$prod" --corpus all
    repro_say "constitution/filter: the two REGEN runs the v2 assembly folds in (advice-regen / stories-regen; Batch API)"
    for kind in advice stories; do
      rc="$(cfg "$kind-regen.yaml")"
      coordination_line "$rc" all "an UPPER BOUND: the regen re-generates a subset of the DAG and \`corpusgen regen-*\` prints the exact count"
      repro_run "${CORPUSGEN[@]}" "regen-$kind" --config "$rc"
      repro_run "${CORPUSGEN[@]}" filter --config "$rc" --corpus all
    done
    repro_say "constitution/filter: assemble-v2 -> final-production-20260831-v2.jsonl (v1 + regen - the recorded drops: p35, p36 and 3 ids; the paper's finals)"
    repro_run "${CORPUSGEN[@]}" assemble-v2 --config "$(cfg production-v2.yaml)" --corpus all \
      --add advice=data/advice/final-advice-regen-20260901.jsonl --add stories=data/stories/final-stories-regen-20260901.jsonl \
      --drop-run production-20260831 --drop-principle p35 --drop-principle p36 \
      --drop-id adv-p35-t03-s03 --drop-id adv-p35-t03-s06 --drop-id adv-p35-t03-s08 --suffix -v2
    repro_run "${CORPUSGEN[@]}" report --config "$prod" --suffix -v2
  fi
  want register-edit && repro_note "constitution: no register-edit pass (the r2 pass is the control corpus's)"
  return 0
}
run_control(){
  local v1 r2 adv sto all model conc; v1="$(cfg control-scale.yaml)"; r2="$(cfg control-scale-r2.yaml)"
  for c in advice stories; do
    if want "$c"; then
      repro_say "control/$c: corpusgen run (Batch API, $MODEL_ID; placebo constitution-control-v2 + prompts-control overlay)"
      coordination_line "$v1" "$c"
      repro_run "${CORPUSGEN[@]}" run --config "$v1" --corpus "$c"
    fi
  done
  if want filter; then
    repro_say "control/filter: offline filter chain -> data/control/final-control-scale-20260903-{advice,stories}.jsonl (v1) + import-scan"
    repro_run "${CORPUSGEN[@]}" filter --config "$v1" --corpus all
    repro_run "${CORPUSGEN[@]}" import-scan --config "$v1" --corpus all
  fi
  if want register-edit; then
    repro_say "control/register-edit: the r2 register-matching edit pass (plan -> run [$ENGINE engine] -> assemble -> import-scan) -> final-control-scale-20260903-r2-{advice,stories}.jsonl (the paper's finals)"
    repro_run "${CORPUSGEN[@]}" register-edit plan --config "$r2"
    if [ -n "${SISTEER_COORDINATION_CHANNEL:-}" ]; then
      read -r adv sto all model conc <<<"$(corpusgen_plan_numbers "$r2")"   # the config's engine.model + register_edit.max_concurrency
      echo "--- ready-to-post for $SISTEER_COORDINATION_CHANNEL ---"
      echo "${conc:-200} ${model:-$MODEL_ID} lowprio ~6h    (register-edit run: the live lowprio runner at ${conc:-200} concurrent, shrinking on 429/529; \`register-edit plan\` prints the exact edit count)"
      echo "--- (posting is the user's job; proceeding now) ---"
    fi
    repro_run "${CORPUSGEN[@]}" register-edit run --config "$r2" --engine "$ENGINE"
    repro_run "${CORPUSGEN[@]}" register-edit assemble --config "$r2" --corpus all
    repro_run "${CORPUSGEN[@]}" import-scan --config "$r2" --corpus all --suffix -r2
  fi
}
if [ "$DO_GENERATE" = 1 ]; then
  repro_say "stage 01: corpus=$CORPUS phase=$PHASE secrets=$SISTEER_SECRETS_MODE $([ "${REPRO_DRY_RUN:-0}" = 1 ] && echo '(DRY RUN: nothing is submitted)')"
  case "$CORPUS" in
    constitution) run_constitution;;
    control) run_control;;
    all) run_constitution; run_control;;
  esac
  repro_say "stage 01 done (finals under $CORPUS_DIR/data/; the committed shards are reproduce/tools/shard_corpus.py over those finals)"
fi
