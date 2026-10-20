#!/bin/bash
# reproduce/lib/hf_env.sh — point HuggingFace at the cluster's cache from an sbatch body (source it; never --export=).
# Installed onto each cluster's shared storage by 00_setup.sh as $SHARED/scripts/hf_env.sh, the path every job sources.
#
# Resolution order (nothing is probed: a cluster's storage roots come from the site, never from a guessed mount point):
#   1. SISTEER_HF_HOME (+ SISTEER_SHARED_ROOT) from the job's command-line env prefix — a configured site
#   2. hf_site.env beside the installed copy: HF_SITE_HF_HOME and HF_SITE_SHARED_ROOT, which `reproduce/00_setup.sh --what
#      helpers` renders from the site's cluster block (CLUSTER_HF_HOME, CLUSTER_SHARED_ROOT); a shared root the job carries wins
# HF_TOKEN_PATH is a path under the shared root, one per account, so a shared cache never shares a token; on a cluster it
# points at a file that does not exist ON PURPOSE (no credential on cluster storage: gated repos are pre-fetched into the
# cache by hf_fetch.sh and loaded offline, HF_HUB_OFFLINE=1). With neither source, HF_HOME is left unset and sourcing still
# succeeds (it never kills the job body).
# W&B: hf_site.env also carries the site's project for each purpose whose jobs are submitted by hand (HF_SITE_WANDB_PROJECT_<PURPOSE>,
# rendered from the site's SISTEER_WANDB_PROJECT_<PURPOSE>); a non-empty one is exported as SISTEER_WANDB_PROJECT_<PURPOSE> unless the
# job carries that key already, whichever HF branch ran.
_hf_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)"
_hf_home="" _hf_shared=""
if [ -n "${SISTEER_HF_HOME:-}" ]; then
  _hf_home="$SISTEER_HF_HOME"
  _hf_shared="${SISTEER_SHARED_ROOT:-$(dirname "$SISTEER_HF_HOME")}"
elif [ -n "$_hf_here" ] && [ -r "$_hf_here/hf_site.env" ]; then
  # shellcheck source=/dev/null
  . "$_hf_here/hf_site.env"
  if [ -n "${HF_SITE_HF_HOME:-}" ]; then
    _hf_home="$HF_SITE_HF_HOME"
    _hf_shared="${SISTEER_SHARED_ROOT:-${HF_SITE_SHARED_ROOT:-$(dirname "$HF_SITE_HF_HOME")}}"
  fi
fi
if [ -n "$_hf_home" ]; then
  export HF_HOME="$_hf_home"
  export HF_TOKEN_PATH="$_hf_shared/hf_token" HUGGINGFACE_TOKEN_PATH="$_hf_shared/hf_token"
else
  echo "hf_env.sh: no SISTEER_HF_HOME and no hf_site.env beside ${_hf_here:-this file} (reproduce/00_setup.sh --what helpers writes it) on $(hostname) — HF_HOME left unset" >&2
fi
if [ -n "$_hf_here" ] && [ -r "$_hf_here/hf_site.env" ]; then
  # shellcheck source=/dev/null
  . "$_hf_here/hf_site.env"
  for _hf_k in ${!HF_SITE_WANDB_PROJECT_@}; do
    _hf_dst="SISTEER_${_hf_k#HF_SITE_}"
    if [ -z "${!_hf_dst:-}" ] && [ -n "${!_hf_k:-}" ]; then export "$_hf_dst=${!_hf_k}"; fi
    unset "$_hf_k"
  done
fi
unset _hf_here _hf_home _hf_shared _hf_k _hf_dst HF_SITE_HF_HOME HF_SITE_SHARED_ROOT
# The G0DM0D3 templates (sisteer.eval.attacks_godmode) are not distributed with the repository: 00_setup.sh --what third_party
# installs the verified file under the shared root, beside hf_token. Compute nodes have no egress, so a job reads that copy and
# never fetches (a missing copy refuses by name). Set only when a branch above resolved the shared root; a caller's value stands.
if [ -n "${HF_HOME:-}" ] && [ -n "${HF_TOKEN_PATH:-}" ]; then
  export SISTEER_THIRD_PARTY_DIR="${SISTEER_THIRD_PARTY_DIR:-$(dirname "$HF_TOKEN_PATH")/third_party}" SISTEER_THIRD_PARTY_OFFLINE="${SISTEER_THIRD_PARTY_OFFLINE:-1}"
fi
