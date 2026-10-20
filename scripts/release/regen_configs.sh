#!/usr/bin/env bash
# Regenerate every generated config in dependency order and move the in-tree sha pins with them; or, with --check,
# verify that every generated config and every pin is current.
#
#   scripts/release/regen_configs.sh            regenerate, re-pin, and repeat until a round changes nothing
#   scripts/release/regen_configs.sh --check    every generator's --check and repin.py --check; exit 1 on any failure
#   scripts/release/regen_configs.sh --list     print the generator invocations, one per line, in order
#
# The order matters. A generator reads what an earlier one wrote: the sampled configs are rendered from the wave configs,
# the strength configs from the ladder's serving configs, the overlays from the matrices, and the matrices assert the
# sha256 of the frozen roster and pins copies their views declare. So the frozen pins are re-declared first
# (repin.py --frozen), then every generator runs in the order below, then every other pin is re-declared (repin.py).
# A round that changes no file ends the regeneration; a round that does (a re-pinned file another generator embeds) is
# followed by another. The interpreter is $PY (default python3), run from the checkout root with src/ on PYTHONPATH.
set -uo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PY=${PY:-python3}
MAX_ROUNDS=${REGEN_MAX_ROUNDS:-4}
cd "$ROOT" || exit 2
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

# One generator invocation per line, in dependency order. --check is appended for the check pass.
GENERATORS=(
  "scripts/sdfbase/make_full22_configs.py"
  "scripts/sdfbase/make_full22_configs.py --roster configs/sdfbase_full22_roster_psrcert.yaml"
  "scripts/sdfbase/make_full22_configs.py --roster configs/sdfbase_full22_roster_ctl_neg1_eb64.yaml"
  "scripts/sdfbase/make_full22_configs.py --roster configs/sdfbase_full22_roster_seedrep_probe_eb64_s1.yaml"
  "scripts/sdfbase/make_mt_wave_configs.py"
  "scripts/sdfbase/full22/sampled/make_sampled_configs.py"
  "scripts/sdfbase/full22/sampled/make_sampled_configs.py --roster configs/sdfbase_full22_roster_ctl_neg1_eb64.yaml"
  "scripts/sdfbase/full22/misalign/make_misalign_manifest.py"
  "scripts/alphasteer_f22/make_ladder_configs.py"
  "scripts/alphasteer_f22/make_strength_configs.py"
  "scripts/alphasteer_f22/make_strength_configs.py --lane alphasteer_f22_rebake"
  "scripts/alphasteer_f22/screen/make_strength_sampled_configs.py"
  "scripts/bcbw_configs.py"
  "scripts/bcbw_configs.py --front-door qwen35_4b_demo"
  "scripts/bcbw_configs.py --front-door qwen35_4b_demo --view d0p75"
  "scripts/bcbw_configs.py --front-door qwen35_4b_demo --view d0p75_relu"
  "scripts/sdfbase/gsm8k/g8k_paired_allmodels.py"
  "scripts/sdfbase/gsm8k/g8k_seedrep_arditi.py"
  "scripts/sdfbase/gsm8k/make_gsm8k_matrix.py"
  "scripts/sdfbase/gsm8k/make_gsm8k_matrix.py --matrix-name llama"
  "scripts/sdfbase/gsm8k/make_gsm8k_matrix.py --matrix-name itdoc"
  "scripts/sdfbase/gsm8k/make_gsm8k_matrix.py --matrix-name qwen_seedrep"
  "scripts/sdfbase/gsm8k/make_gsm8k_matrix.py --matrix-name arditi_allpos"
  "scripts/sdfbase/gsm8k/make_gsm8k_matrix.py --matrix-name frontdoor_qwen35_4b_demo"
  "scripts/sdfbase/gsm8k/make_gsm8k_overlays.py"
  "scripts/sdfbase/gsm8k/make_gsm8k_overlays.py --matrix llama"
  "scripts/sdfbase/gsm8k/make_gsm8k_overlays.py --matrix itdoc"
  "scripts/sdfbase/gsm8k/make_gsm8k_overlays.py --matrix qwen_seedrep"
  "scripts/sdfbase/gsm8k/make_gsm8k_overlays.py --matrix arditi_allpos"
  "scripts/sdfbase/gsm8k/make_gsm8k_overlays.py --matrix arditi_seedrep"
  "scripts/sdfbase/gsm8k/make_gsm8k_overlays.py --matrix paired_allmodels"
  "scripts/sdfbase/gsm8k/make_gsm8k_overlays.py --matrix frontdoor_qwen35_4b_demo"
  "scripts/sdfbase/math500/make_math500_overlays.py"
  "scripts/sdfbase/ifeval/make_ifeval_matrix.py"
  "scripts/sdfbase/ifeval/make_ifeval_matrix.py --views-yaml configs/sdfbase/ifeval/views_qwen_seedrep.yaml --out configs/sdfbase/ifeval/matrix_qwen_seedrep.json"
  "scripts/sdfbase/ifeval/make_ifeval_overlays.py"
  "scripts/sdfbase/ifeval/make_ifeval_overlays.py --matrix configs/sdfbase/ifeval/matrix_qwen_seedrep.json"
  "scripts/sdfbase/full22/static_padding/make_padding_configs.py"
  "scripts/sdfbase/seedrep/render_arditi_seedrep_configs.py"
  "scripts/sdfbase/full22/write_prompted_variant_dirs.py --si constitution_full"
  "scripts/sdfbase/make_sdfbase_configs.py"
)

run() {  # one generator invocation (word-split on purpose), its output to a log; prints one result line
  local line=$1 log=$2 start rc
  start=$(date +%s)
  # shellcheck disable=SC2086
  "$PY" $line > "$log" 2>&1
  rc=$?
  printf '%-4s %4ss  %s\n' "rc=$rc" "$(( $(date +%s) - start ))" "$line"
  if (( rc != 0 )); then sed 's/^/    /' "$log" | tail -n 20; fi
  return $rc
}

state_hash() {  # the content of every file a round can change
  git ls-files -z --cached --others --exclude-standard -- configs tests scripts/sdfbase \
    | xargs -0 sha256sum 2>/dev/null | sha256sum | cut -d' ' -f1
}

LOGS=$(mktemp -d "${TMPDIR:-/tmp}/regen_configs.XXXXXX")
trap 'rm -rf "$LOGS"' EXIT

case "${1:-}" in
  --list)
    printf '%s\n' "${GENERATORS[@]}"
    exit 0 ;;
  --check)
    fail=0 n=0
    for g in "${GENERATORS[@]}"; do
      n=$((n + 1))
      run "$g --check" "$LOGS/check_$n.log" || fail=1
    done
    run "scripts/release/repin.py --check" "$LOGS/repin.log" || fail=1
    if (( fail )); then echo "regen_configs: a check failed (above)"; exit 1; fi
    echo "regen_configs: all $n generator checks and the pin check pass"
    exit 0 ;;
  "") ;;
  *)
    echo "usage: $0 [--check|--list]" >&2
    exit 2 ;;
esac

run "scripts/release/repin.py --frozen" "$LOGS/repin_frozen.log" || exit 1
for (( round = 1; round <= MAX_ROUNDS; round++ )); do
  before=$(state_hash)
  echo "-- round $round"
  n=0
  for g in "${GENERATORS[@]}"; do
    n=$((n + 1))
    run "$g" "$LOGS/r${round}_$n.log" || { echo "regen_configs: a generator failed in round $round"; exit 1; }
  done
  run "scripts/release/repin.py" "$LOGS/repin_r$round.log" || exit 1
  if [[ $(state_hash) == "$before" ]]; then
    echo "regen_configs: fixed point after round $round"
    exit 0
  fi
done
echo "regen_configs: no fixed point after $MAX_ROUNDS rounds" >&2
exit 1
