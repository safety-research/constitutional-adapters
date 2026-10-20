#!/bin/bash
# Pull ONE WAVE's out-root subtree of the gsm8k-paired200-paper-objects wave root into a local gen-root for the analysis
# runner (math-eight-arms-extension plan P2 / F11a; a port of the MATH-500 lane's pull_wave.sh — the
# GSM8K lane had no pull script). Live cell outputs only: directories renamed aside as *-superseded / *-failed-probe /
# *-cancelled / *-censored are the kept-but-retired outputs of earlier trees and are excluded (never deleted on the
# cluster). NEVER --delete: the destination only ever gains files.
#
# Every wave writes under its OWN out-root prefix (g8k_cells.OUT_ROOT_PREFIX_BY_WAVE: controls/ = the published wave,
# controls_ext/ = the eight-arm prompted rows, controls_steer/ = the AlphaSteer / Arditi rung), so a pull names exactly ONE
# subtree and never carries another wave's legs. The shared per-model inputs beside the subtree ride along: budgets/ (the
# prompt-budget artifacts + fit sidecars), nemo_backend/ (the Nemotron kernel attestation), rung16384/ (the ladder planner's
# per-cell subsets, bound to their source stores) and ladder_plan/ (the planner's records); the published wave's pull
# additionally carries overlap/ (the thinking-OFF overlap legs; its own shape, unchanged).
#
# THE LAUNCH ATTESTATION RIDES ALONG, FROM EVERY CLUSTER ROOT (the published GSM8K pull is CROSS-CLUSTER — 6 of
# its 42 manifests' routing_table paths name the overflow cluster's shared root, and the analysis' `launch_attestation`
# resolves each leg's standup record under the pull, so a pull from the primary root alone leaves those legs `absent`).
# For EVERY standup directory `serving/<jobdir>/gsm8k/` exactly the analysis' RECORD FILES, by name: the launch attestation
# trio `steering_attestation.json`, `vllm_server.log` (the API server's `non-default args:` startup echo) and
# `serving_config.json` (the standup's SERVING-CONFIG provenance: the sha256 of the config + manifest THAT leg served;
# a leg older than the record has none, reported `absent`, never a refusal); the DECLARED cell's standup record
# `routing_table.json`, `standup_jobs.jsonl`, `lens_registry/registry.json` (+ the steering attestation above — the
# validators' STANDUP_RECORD_FILES, the served-positions fact of a declared cell); and the NATIVE cell's standup record
# `serving_attestation.json` (+ the routing table and the standup jobs — NATIVE_STANDUP_RECORD_FILES, the served-positions
# fact of a natively served cell). Everything else under serving/ stays on the cluster; a `serving` directory nested anywhere else is
# excluded. `--src` names the wave root on the PRIMARY cluster (the subtree, the shared inputs and its serving/ pairs);
# each `--serving-src` names the SAME wave root on ANOTHER cluster, from which ONLY the serving/ pairs are pulled (the
# stores of a cross-cluster wave were already pulled into ONE controls/ tree — the published pull's shape). Standup
# directories are not named by wave, so every pull carries every job's pair (inert beside legs it does not own).
#
# ONE rsync per host (rsync cannot mix hosts in one transfer), the SAME anchored filter chain on each; --ignore-missing-args
# tolerates a shared dir that does not exist yet under a root (rung16384/ before the ladder ran).
#
# usage: pull_wave.sh [--subtree NAME] [--src HOST:PATH] [--serving-src HOST:PATH ...] [--dest DIR] [--dry-run] [--print-argv]
#                     <sha7 of the store pin>
#   --subtree NAME        the out-root prefix to pull: controls_ext / controls_steer (default: controls, the published wave's)
#   --src HOST:PATH       the wave root on the primary cluster (default: $G8K_PULL_SRC; required — a machine-local wrapper
#                         sets it, this script carries no host name)
#   --serving-src H:PATH  the same wave root on another cluster — its serving/ pairs only (repeatable; default: $G8K_PULL_SERVING_SRC,
#                         a space-separated list)
#   --dest DIR            the destination (default: ${G8K_PULL_DEST_ROOT:-.}/pull_<sha7>)
#   --dry-run             rsync -n -i on every transfer: itemize what WOULD land, write nothing
#   --print-argv          print every rsync argv one word per line (a `---` line between transfers) and exit 0 without running
set -euo pipefail
SUBTREE=controls; SRC="${G8K_PULL_SRC:-}"; DEST=""; PRINT=0; DRY=0
SERVING_SRCS=()
if [ -n "${G8K_PULL_SERVING_SRC:-}" ]; then read -r -a SERVING_SRCS <<< "$G8K_PULL_SERVING_SRC"; fi
while [ $# -gt 0 ]; do
  case "$1" in
    --subtree) [ $# -ge 2 ] || { echo "pull_wave.sh: --subtree needs a NAME" >&2; exit 2; }; SUBTREE="$2"; shift 2;;
    --src) [ $# -ge 2 ] || { echo "pull_wave.sh: --src needs HOST:PATH" >&2; exit 2; }; SRC="$2"; shift 2;;
    --serving-src) [ $# -ge 2 ] || { echo "pull_wave.sh: --serving-src needs HOST:PATH" >&2; exit 2; }; SERVING_SRCS+=("$2"); shift 2;;
    --dest) [ $# -ge 2 ] || { echo "pull_wave.sh: --dest needs a DIR" >&2; exit 2; }; DEST="$2"; shift 2;;
    --dry-run) DRY=1; shift;;
    --print-argv) PRINT=1; shift;;
    --*) echo "pull_wave.sh: unknown flag $1" >&2; exit 2;;
    *) break;;
  esac
done
P7="${1:?sha7 of the store pin}"
[ -n "$SRC" ] || { echo "pull_wave.sh: no source — pass --src HOST:PATH or set G8K_PULL_SRC" >&2; exit 2; }
case "$SUBTREE" in
  */*|""|.|..) echo "pull_wave.sh: --subtree must be ONE out-root prefix name (controls, controls_ext, controls_steer), got '$SUBTREE'" >&2; exit 2;;
  controls|controls_*) ;;
  *) echo "pull_wave.sh: --subtree names an out-root prefix (controls or controls_<wave>), got '$SUBTREE'" >&2; exit 2;;
esac
for s in "${SERVING_SRCS[@]+"${SERVING_SRCS[@]}"}"; do
  [ "$s" != "$SRC" ] || { echo "pull_wave.sh: --serving-src $s is the --src root itself" >&2; exit 2; }
done
DEST="${DEST:-${G8K_PULL_DEST_ROOT:-.}/pull_$P7}"
# the standup's task subdirectory: the sbatch's `mkdir -p "$SERVE_DIR/$G8K_SERVE_SUBDIR"` (g8k_cells.SERVE_SUBDIR), the analysis'
# STANDUP_TASK_SUBDIR (tinycap_gsm8k_validators; test-pinned equal). The file names are the analysis' own: the launch
# attestation trio (STANDUP_ATTESTATION / the launch log / the serving-config record), the DECLARED cell's standup record
# (STANDUP_RECORD_FILES: routing_table.json, standup_jobs.jsonl, steering_attestation.json, lens_registry/registry.json — the
# served-positions fact of a declared cell, served-positions-null-diagnosis.md) and the NATIVE cell's standup record
# (NATIVE_STANDUP_RECORD_FILES: routing_table.json, standup_jobs.jsonl, serving_attestation.json — the served-positions fact
# of a natively served cell, native-positions-resolve). Explicit names only: never a wildcard that would sweep the logs,
# endpoint records or cells_served.json beside them.
SERVING_TASK=gsm8k
# filter order matters (first match wins): the retired suffixes first (a retired standup dir is never pulled), then the
# anchored include chain for exactly the record files under serving/<jobdir>/$SERVING_TASK/ (the lens_registry/ directory
# is admitted so that ONE file under it can be), then everything else under the top-level serving/ excluded, then any
# `serving` directory nested elsewhere
FILTER=(--exclude='*-superseded' --exclude='*-failed-probe' --exclude='*-cancelled' --exclude='*-censored'
        --exclude='inspect_evals_cache'
        --include='/serving/' --include='/serving/*/' --include="/serving/*/$SERVING_TASK/"
        --include="/serving/*/$SERVING_TASK/steering_attestation.json" --include="/serving/*/$SERVING_TASK/vllm_server.log"
        --include="/serving/*/$SERVING_TASK/serving_config.json"
        --include="/serving/*/$SERVING_TASK/routing_table.json" --include="/serving/*/$SERVING_TASK/standup_jobs.jsonl"
        --include="/serving/*/$SERVING_TASK/serving_attestation.json"
        --include="/serving/*/$SERVING_TASK/lens_registry/" --include="/serving/*/$SERVING_TASK/lens_registry/registry.json"
        --exclude='/serving/**' --exclude='serving')
RSYNC=(rsync -az --prune-empty-dirs --ignore-missing-args)
if [ "$DRY" = 1 ]; then RSYNC+=(-n -i); fi
SSH=(-e "ssh -o ServerAliveInterval=15 -o ServerAliveCountMax=2")
ARGV=("${RSYNC[@]}" "${FILTER[@]}" "${SSH[@]}"
      "$SRC/$SUBTREE" "$SRC/budgets" "$SRC/nemo_backend" "$SRC/rung16384" "$SRC/ladder_plan" "$SRC/serving")
if [ "$SUBTREE" = controls ]; then ARGV+=("$SRC/overlap"); fi
ARGV+=("$DEST/")
TRANSFERS=("$(printf '%q ' "${ARGV[@]}")")
for s in "${SERVING_SRCS[@]+"${SERVING_SRCS[@]}"}"; do
  ALT=("${RSYNC[@]}" "${FILTER[@]}" "${SSH[@]}" "$s/serving" "$DEST/")
  TRANSFERS+=("$(printf '%q ' "${ALT[@]}")")
done
if [ "$PRINT" = 1 ]; then
  first=1
  for t in "${TRANSFERS[@]}"; do
    [ "$first" = 1 ] || echo '---'; first=0
    eval "printf '%s\n' $t"
  done
  exit 0
fi
mkdir -p "$DEST"
for t in "${TRANSFERS[@]}"; do
  eval "$t"
done
if [ "$DRY" = 1 ]; then echo "dry run: nothing written under $DEST ($SUBTREE from $SRC; serving/ pairs from $((1 + ${#SERVING_SRCS[@]})) root(s))"; exit 0; fi
echo "pulled $SUBTREE -> $DEST (serving/ pairs from $((1 + ${#SERVING_SRCS[@]})) root(s))"
find "$DEST/$SUBTREE" -name rows.jsonl 2>/dev/null | while read -r f; do echo "$(wc -l < "$f") ${f#"$DEST/$SUBTREE/"}"; done
find "$DEST/$SUBTREE" -name '*.eval' 2>/dev/null | wc -l | sed 's/^/eval logs: /'
find "$DEST/$SUBTREE" -name RESULT 2>/dev/null | wc -l | sed 's/^/RESULT files: /'
find "$DEST/serving" -mindepth 3 -maxdepth 3 -name vllm_server.log 2>/dev/null | wc -l | sed 's/^/launch attestations (serving\/<jobdir>\/'"$SERVING_TASK"'\/): /'
find "$DEST/serving" -mindepth 3 -maxdepth 3 -name serving_config.json 2>/dev/null | wc -l | sed 's/^/serving-config records: /'
find "$DEST/serving" -mindepth 3 -maxdepth 3 -name routing_table.json 2>/dev/null | wc -l | sed 's/^/routing tables: /'
find "$DEST/serving" -mindepth 3 -maxdepth 3 -name serving_attestation.json 2>/dev/null | wc -l | sed 's/^/native serving attestations: /'
find "$DEST/serving" -mindepth 4 -maxdepth 4 -path '*/lens_registry/registry.json' 2>/dev/null | wc -l | sed 's/^/lens registries: /'
