#!/bin/bash
#SBATCH --job-name=twosi-common-text
#                                     # Session-unique twosi-common-* name: cancel by
#                                     # the JOB ID you submitted (shared-user account —
#                                     # a name/user-scoped scancel would hit peers).
#SBATCH --partition=general
#SBATCH --qos=high                    # cluster A's non-preemptible default QoS;
#                                     # override the QoS name when run on cluster B.
#SBATCH --gres=gpu:1                  # one H200: 7 instructions x 1037 identities,
#                                     # two no-grad forwards each, sequential
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G                    # ~128G per GPU; NEVER --mem=0 (hard rule 2)
#SBATCH --time=0-03:00:00             # measured: 3 long-SI (teacher+student) passes
#                                     # took 25m55s (job 30542); this axis is 3 long
#                                     # + 4 short instructions ~= 45-60 min real.
#                                     # ~3x pad, NOT trimmed: the final save writes
#                                     # after every instruction, but the axis is only
#                                     # complete if the last one lands.
# exclude bad nodes at submission: sbatch $CLUSTER_SBATCH_EXTRA <this script> (CLUSTER_SBATCH_EXTRA is site config)
#SBATCH --output=/workspace-vast/%u/exp/logs/%x_%j.out
#SBATCH --open-mode=append            # retries append, never clobber
#SBATCH --no-requeue                  # stage2_common self-escalates; no double-requeue
#SBATCH --signal=B:TERM@120
#
# Common-text divergence 2x2: KL(teacher || student) teacher-forced on the
# UNDEFENDED completion (neg — generated with no system instruction,
# byte-identical across the SI pools, asserted in-script), the SAME text
# for every scored instruction. The own-pos floors (jobs 30542 / 191200)
# score each SI on a completion its OWN rejection-sampling acceptance
# chose, so their cross-SI comparison is confounded by construction; and
# they are monotonic in instruction LENGTH (88 tok -> 0.056, 5872 ->
# 0.110, 6380 -> 0.101), so the headline con>irr>refusal ordering is
# exactly what a pure length effect would produce. The six instructions
# are the 2x2 that separates the two:
#
#   long  + safety      si_constitution (5872 tok)
#   long  + irrelevant  si_irrelevant (6380)
#   short + safety      safety_refusal (88), si_caa (10)
#   short + irrelevant  json (22), bullets (26)
#
# si_caa is a safety instruction SHORTER than safety_refusal. Where it
# lands answers whether length or content does the work. json/bullets are
# the content-free calibration ceiling.
#
# Selection: the SAME 1037 stratified identities as jobs 30542/191200
# (within-family sha256 rank, per-family 96), pinned by
# --verify-selection against 191200's recorded selection — the run
# REFUSES before the model load if it would select anything else.
# Output json is (re)written after EVERY instruction (partial/pending
# flags), so a walltime kill loses at most the in-flight instruction.
#
# Inputs: $SHARED/exp/diffmean2x2/pools/pairs_<si>.jsonl (23,002 rows
# each, verified) + $SHARED/exp/twosi_floors/teacher_paired_191200.json
# (the selection pin). Output:
# $SHARED/exp/twosi_floors/common_text_<jobid>.json.
#
# Submit (after mirroring this checkout to the cluster):
#   cd ${SISTEER_REPO_MIRROR}-twosi
#   git rev-parse HEAD > EXPECTED_SHA
#   sbatch --comment=sha:$(git -C ${SISTEER_REPO_MIRROR}-twosi rev-parse HEAD) \
#     scripts/sbatch/twosi_common_text.sbatch

set -uo pipefail
SHARED="${SISTEER_SHARED_ROOT:?set SISTEER_SHARED_ROOT, the shared root of the cluster (the front door exports it from the site config)}"
S2_ROOT="${SISTEER_REPO_MIRROR:?set SISTEER_REPO_MIRROR, the cluster clone of this repository (the front door exports it from the site config)}-twosi"
S2_REPO_DIR="$S2_ROOT"
source "$S2_REPO_DIR/scripts/sbatch/stage2_common.sh"
s2_init "$@"

# Standing NCCL exports (inert on 1 GPU; kept so a widened resubmit can
# never silently miss them). Body-level — NEVER sbatch --export (rule 1).
export NCCL_NVLS_ENABLE=0

POOLS="$SHARED/exp/diffmean2x2/pools"
OUT_DIR="$SHARED/exp/twosi_floors"
mkdir -p "$OUT_DIR"
OUT="$OUT_DIR/common_text_${SLURM_JOB_ID}.json"
VERIFY="$OUT_DIR/teacher_paired_191200.json"
if [ ! -f "$VERIFY" ]; then
  s2_log "FATAL: $VERIFY missing — the established-selection pin (job 30542's 1037 identities, re-verified by job 191200) is required; refusing to measure an unpinned selection."
  exit 1
fi

s2_run uv run --no-sync python scripts/measure_teacher_signal.py \
  --config configs/two_si.yaml \
  --pairs-dir "$POOLS" \
  --per-family 96 \
  --batch-size 2 \
  --max-len 8192 \
  --fixed-completion neg \
  --instructions json,bullets,si_caa,safety_refusal,si_constitution,si_irrelevant \
  --verify-selection "$VERIFY" \
  --out "$OUT"
rc=$?
# rc captured IMMEDIATELY (a trailing command that succeeds would mask a
# nonzero exit — a job dead at model load would report COMPLETED).
exit $rc
