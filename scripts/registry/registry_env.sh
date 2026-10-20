#!/usr/bin/env bash
# registry_env.sh — source this in a cluster job (sbatch body, never --export=) so the harness's record emitter knows where to
# spool, how to push and which substrate it runs on. Body-level exports only.
#
#   source $SHARED/scripts/registry/registry_env.sh
#
# The cluster's own values come from registry_site.env BESIDE this file, which `reproduce/00_setup.sh --what helpers` renders
# from the site's cluster block when it installs the helpers on that cluster (KEY=value lines, shell-quoted):
#   REGISTRY_SITE_SHARED_ROOT       the cluster's shared root (the block's CLUSTER_SHARED_ROOT)
#   REGISTRY_SITE_SUBSTRATE         the cluster's registry label (CLUSTER_SUBSTRATE)
#   REGISTRY_SITE_CLUSTER_LABEL     the site's own name for the cluster (CLUSTER_LABEL; the generic `slurm` substrate needs it)
#   REGISTRY_SITE_SLURM_SUBSTRATES  the site's Slurm-cluster map (SISTEER_SLURM_SUBSTRATES in config.env; optional)
# Nothing is probed: with no registry_site.env there is no shared root (this file sourced from a checkout, say).
#
# Sets, each only when the job does not already carry a nonempty value (an inherited value is kept as it is):
#   SISTEER_REGISTRY_SPOOL           the producer's record spool, <shared root>/registry/records (created); with no shared root
#                                    it stays unset and emission is disabled with a one-line notice, never a failure
#   SISTEER_REGISTRY_PUSH            push_records.sh beside this file, when it is there and executable (with a shared root)
#   SISTEER_REGISTRY_SUBSTRATE       the cluster's registry label: the first thing the emitter reads
#   SISTEER_REGISTRY_CLUSTER_LABEL   the cluster's site label
#   SISTEER_SLURM_SUBSTRATES         the site's Slurm-cluster map, when the site sets one
# No other name of registry_site.env reaches the job.
#
# Never sets or reads a credential: the push key stays at <shared root>/.ssh/registry-push and is read by push_records.sh alone.
# Safe to source twice, and under `set -u`.
_reg_here="$(CDPATH='' cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REGISTRY_SITE_SHARED_ROOT=""; REGISTRY_SITE_SUBSTRATE=""; REGISTRY_SITE_CLUSTER_LABEL=""; REGISTRY_SITE_SLURM_SUBSTRATES=""
if [[ -r "$_reg_here/registry_site.env" ]]; then source "$_reg_here/registry_site.env"; fi

if [[ -n "$REGISTRY_SITE_SUBSTRATE" ]]; then export SISTEER_REGISTRY_SUBSTRATE="${SISTEER_REGISTRY_SUBSTRATE:-$REGISTRY_SITE_SUBSTRATE}"; fi
if [[ -n "$REGISTRY_SITE_CLUSTER_LABEL" ]]; then export SISTEER_REGISTRY_CLUSTER_LABEL="${SISTEER_REGISTRY_CLUSTER_LABEL:-$REGISTRY_SITE_CLUSTER_LABEL}"; fi
if [[ -n "$REGISTRY_SITE_SLURM_SUBSTRATES" ]]; then export SISTEER_SLURM_SUBSTRATES="${SISTEER_SLURM_SUBSTRATES:-$REGISTRY_SITE_SLURM_SUBSTRATES}"; fi

if [[ -n "$REGISTRY_SITE_SHARED_ROOT" ]]; then
  export SISTEER_REGISTRY_SPOOL="${SISTEER_REGISTRY_SPOOL:-$REGISTRY_SITE_SHARED_ROOT/registry/records}"
  mkdir -p "$SISTEER_REGISTRY_SPOOL" 2>/dev/null || true
  if [[ -x "$_reg_here/push_records.sh" ]]; then
    export SISTEER_REGISTRY_PUSH="${SISTEER_REGISTRY_PUSH:-$_reg_here/push_records.sh}"
  fi
else
  [[ -n "${SISTEER_REGISTRY_SPOOL:-}" ]] || echo "registry_env: no shared root (no registry_site.env beside $_reg_here/registry_env.sh; reproduce/00_setup.sh --what helpers writes it); record emission disabled (set SISTEER_REGISTRY_SPOOL to enable)" >&2
fi
unset _reg_here REGISTRY_SITE_SHARED_ROOT REGISTRY_SITE_SUBSTRATE REGISTRY_SITE_CLUSTER_LABEL REGISTRY_SITE_SLURM_SUBSTRATES
