#!/bin/bash
# Pull ONE WAVE's out-root subtree of the math500-thinking-on-full22 wave root into a local gen-root for the analysis
# runner (math-eight-arms-extension plan P2 / F11a / C3 / §7 step 1). Live cell outputs only: directories renamed
# aside as *-superseded / *-failed-probe / *-cancelled / *-censored are the kept-but-retired outputs of earlier trees
# and are excluded (they are never deleted on the cluster). NEVER --delete: the destination only ever gains files.
#
# Every wave writes under its OWN out-root prefix (cells.py OUT_ROOT_PREFIX_BY_WAVE: controls/ = the published wave,
# controls_ext1/ = the ext1 wave, ...), so a pull names exactly ONE subtree and never carries another wave's legs (F11a: the
# published run's whole-controls pull would have copied the 24 published legs beside the 9 new ones). The shared
# per-model inputs beside the subtree — budgets/ (the prompt-budget artifacts + fit sidecars) and nemo_backend/ (the
# Nemotron kernel attestation) — ride along; the published wave's pull additionally carries diagnostic/ and
# dry_run_prod/ (its own shape, unchanged).
#
# The LAUNCH ATTESTATION rides along too (the analysis' `launch_attestation`): for EVERY standup directory
# `serving/<jobdir>/math500/` exactly the analysis' RECORD FILES, by name — the ENGINE's launch record, ONE of
# `steering_attestation.json` (the LENS workers' binding of the launch to its Slurm job and port) or
# `serving_attestation.json` (the NATIVE lora_native engine's readiness record, the same binding fields; a native standup
# writes it INSTEAD of the lens file, and the analysis selects the record by the manifest's `route.serving_engine`, never
# by which file exists), plus `vllm_server.log` (the API server's `non-default args:` startup echo, the ONE record of the
# composed launch flags that survives teardown; endpoint.json is unlinked) and `serving_config.json` (the standup's
# SERVING-CONFIG provenance: the sha256 of the config + manifest THAT leg served — written before launch so a
# leg that died at startup still records what it would have served; a leg older than the record has none, which the
# analysis reports as `absent`, never a refusal) — ~180 KB per lens job, ~230 KB per native job. Everything else under
# serving/ (driver logs, routing tables, templates, ledgers, endpoint records, cells_served.json) stays on the cluster, and
# a `serving` directory nested anywhere else is still excluded. The analysis reads the record at
# <gen root>/serving/<jobdir>/math500/, the layout the sbatch's SERVE_DIR writes and the manifest's routing_table names, so
# it lands beside the subtree under the same DEST. Standup directories are not named by wave, so every wave's pull carries
# every job's record (inert beside legs it does not own); a pull made BEFORE a name rode along lacks it and must be
# re-pulled (never --delete: it only gains files).
#
# usage: pull_wave.sh [--subtree NAME] [--src HOST:PATH] [--dest DIR] [--print-argv] <sha7 of the store pin>
#   --subtree NAME   the out-root prefix to pull (default: controls, the published wave's; controls_ext1 = the ext1 wave)
#   --src HOST:PATH  the wave root on the cluster (default: $M5T_PULL_SRC; required — a machine-local wrapper sets it,
#                    this script carries no host name)
#   --dest DIR       the destination (default: ${M5T_PULL_DEST_ROOT:-.}/pull_<sha7>)
#   --print-argv     print the rsync argv one word per line and exit 0 without running it (the test's seam)
set -euo pipefail
SUBTREE=controls; SRC="${M5T_PULL_SRC:-}"; DEST=""; PRINT=0
while [ $# -gt 0 ]; do
  case "$1" in
    --subtree) [ $# -ge 2 ] || { echo "pull_wave.sh: --subtree needs a NAME" >&2; exit 2; }; SUBTREE="$2"; shift 2;;
    --src) [ $# -ge 2 ] || { echo "pull_wave.sh: --src needs HOST:PATH" >&2; exit 2; }; SRC="$2"; shift 2;;
    --dest) [ $# -ge 2 ] || { echo "pull_wave.sh: --dest needs a DIR" >&2; exit 2; }; DEST="$2"; shift 2;;
    --print-argv) PRINT=1; shift;;
    --*) echo "pull_wave.sh: unknown flag $1" >&2; exit 2;;
    *) break;;
  esac
done
P7="${1:?sha7 of the store pin}"
[ -n "$SRC" ] || { echo "pull_wave.sh: no source — pass --src HOST:PATH or set M5T_PULL_SRC" >&2; exit 2; }
case "$SUBTREE" in
  */*|""|.|..) echo "pull_wave.sh: --subtree must be ONE out-root prefix name (controls, controls_ext1, ...), got '$SUBTREE'" >&2; exit 2;;
  controls|controls_*) ;;
  *) echo "pull_wave.sh: --subtree names an out-root prefix (controls or controls_<wave>), got '$SUBTREE'" >&2; exit 2;;
esac
DEST="${DEST:-${M5T_PULL_DEST_ROOT:-.}/pull_$P7}"
# the standup's task subdirectory: the sbatch's `mkdir -p "$SERVE_DIR/math500"`, the analysis' LAUNCH_STANDUP_TASK_SUBDIR
# (test-pinned equal to both); the file names are the analysis' LAUNCH_ATTESTATION_FILE (the lens record) /
# LAUNCH_NATIVE_ATTESTATION_FILE (the native record) / LAUNCH_LOG_FILE / LAUNCH_SERVING_CONFIG_FILE. Explicit names only:
# never a wildcard that would sweep the logs, endpoint records or cells_served.json beside them.
SERVING_TASK=math500
# filter order matters (first match wins): the retired suffixes first (a retired standup dir is never pulled), then the
# anchored include chain for exactly the record files under serving/<jobdir>/$SERVING_TASK/, then everything else under
# the top-level serving/ excluded, then any `serving` directory nested elsewhere
ARGV=(rsync -az --prune-empty-dirs
      --exclude='*-superseded' --exclude='*-failed-probe' --exclude='*-cancelled' --exclude='*-censored'
      --exclude='inspect_evals_cache'
      --include='/serving/' --include='/serving/*/' --include="/serving/*/$SERVING_TASK/"
      --include="/serving/*/$SERVING_TASK/steering_attestation.json" --include="/serving/*/$SERVING_TASK/vllm_server.log"
      --include="/serving/*/$SERVING_TASK/serving_config.json"
      --include="/serving/*/$SERVING_TASK/serving_attestation.json"
      --include="/serving/*/$SERVING_TASK/routing_table.json" --include="/serving/*/$SERVING_TASK/standup_jobs.jsonl"
      --exclude='/serving/**' --exclude='serving'
      -e "ssh -o ServerAliveInterval=15 -o ServerAliveCountMax=2"
      "$SRC/$SUBTREE" "$SRC/budgets" "$SRC/nemo_backend" "$SRC/serving")
if [ "$SUBTREE" = controls ]; then ARGV+=("$SRC/diagnostic" "$SRC/dry_run_prod"); fi
ARGV+=("$DEST/")
if [ "$PRINT" = 1 ]; then printf '%s\n' "${ARGV[@]}"; exit 0; fi
mkdir -p "$DEST"
"${ARGV[@]}"
echo "pulled $SUBTREE -> $DEST"
find "$DEST/$SUBTREE" -name rows.jsonl | while read -r f; do echo "$(wc -l < "$f") ${f#"$DEST/$SUBTREE/"}"; done
find "$DEST/$SUBTREE" -name '*.eval' | wc -l | sed 's/^/eval logs: /'
find "$DEST/$SUBTREE" -name RESULT | wc -l | sed 's/^/RESULT files: /'
find "$DEST/serving" -mindepth 3 -maxdepth 3 -name vllm_server.log 2>/dev/null | wc -l | sed 's/^/launch attestations (serving\/<jobdir>\/'"$SERVING_TASK"'\/): /'
find "$DEST/serving" -mindepth 3 -maxdepth 3 -name serving_config.json 2>/dev/null | wc -l | sed 's/^/serving-config records: /'
find "$DEST/serving" -mindepth 3 -maxdepth 3 -name serving_attestation.json 2>/dev/null | wc -l | sed 's/^/native serving attestations: /'
find "$DEST/serving" -mindepth 3 -maxdepth 3 -name routing_table.json 2>/dev/null | wc -l | sed 's/^/routing tables: /'
find "$DEST/serving" -mindepth 3 -maxdepth 3 -name standup_jobs.jsonl 2>/dev/null | wc -l | sed 's/^/standup job records: /'
