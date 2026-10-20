#!/bin/bash
# capeval sandbox VM lifecycle wrapper.
#
# The EvalPlus execution sandbox is the EXISTING GCP VM `sandbox`
# (us-central1-a, e2-standard-16 — created + provisioned once; docker
# + the digest-pinned evalplus image; ~/.ssh copied so it can reach the
# login host directly; secret-manager, GitHub and Claude credentials
# stripped). This wrapper automates start / provision-re-check / exec /
# stop. The grading host orchestrates ONLY — payloads flow login host
# <-> sandbox directly (F19); scripts/capeval_exec_wave.py is the one
# caller that starts the VM and owns the stop-on-EXIT trap.
#
#   capeval_sandbox.sh ensure-up          # start if TERMINATED (no-op if
#                                         # RUNNING), re-resolve internal IP
#                                         # (NEVER cached), wait-ssh <=180s;
#                                         # last stdout line = the IP
#   capeval_sandbox.sh provision          # idempotent re-check at each start:
#                                         # docker present? image at pinned
#                                         # DIGEST? evalplus data seeded?
#                                         # last stdout line = the digest
#   capeval_sandbox.sh run -- <cmd...>    # ssh exec on the VM
#   capeval_sandbox.sh stop [--verify]    # gcloud stop; --verify asserts
#                                         # TERMINATED after
#   capeval_sandbox.sh status             # print instance status
#
# Test seams (tests/test_capeval_sandbox.py, GCLOUD=stub): every external
# binary is an overridable var below; the stub records argv and replays
# canned state, so start-idempotence / IP-re-resolution / trap-fires are
# testable with no cloud access.
# shellcheck shell=bash
set -uo pipefail

GCLOUD="${GCLOUD:-gcloud}"
SSH_BIN="${CAPEVAL_SSH:-ssh}"
VM="${CAPEVAL_SANDBOX_VM:-sandbox}"
ZONE="${CAPEVAL_SANDBOX_ZONE:-us-central1-a}"
SSH_USER="${CAPEVAL_SANDBOX_USER:-$USER}"
SSH_DEADLINE="${CAPEVAL_SSH_DEADLINE:-180}"   # wait-ssh <=180s
SSH_POLL="${CAPEVAL_SSH_POLL:-5}"
PROVISION_MARKER="/var/lib/capeval-provisioned"
# The digest pinned at provisioning; part of harness_config_hash (§5.5).
IMAGE_DIGEST="${CAPEVAL_EVALPLUS_IMAGE:-ganler/evalplus@sha256:26b118098bef281fe8dfe999bf05f1d5b45374b4e6c00161ec0f30592aef4740}"
# evalplus problem-data seed for the execution side (§5.5 pre-seeding; the
# container downloads it otherwise, and the VM may run with no egress).
EVALPLUS_CACHE_FILE="HumanEvalPlus-v0.1.10.jsonl"

log() { echo "[capeval-sandbox $(date -Is)] $*" >&2; }
die() { log "FATAL: $*"; exit 1; }

_status() {
  "$GCLOUD" compute instances describe "$VM" --zone "$ZONE" \
    --format='value(status)' || die "cannot describe instance '$VM' in $ZONE"
}

_resolve_ip() {
  # Re-resolved on EVERY call — never cached (the internal IP
  # usually survives stop/start, but that is a GCP default, not a contract).
  local ip
  ip="$("$GCLOUD" compute instances describe "$VM" --zone "$ZONE" \
        --format='value(networkInterfaces[0].networkIP)')" \
    || die "cannot resolve the internal IP of '$VM'"
  [ -n "$ip" ] || die "empty internal IP for '$VM'"
  echo "$ip"
}

_ssh() {  # _ssh <ip> <cmd...>
  local ip="$1"; shift
  "$SSH_BIN" -o ConnectTimeout=5 -o BatchMode=yes -o StrictHostKeyChecking=accept-new \
    "$SSH_USER@$ip" -- "$@"
}

# Failure-only stop for ensure-up (review infra #1): when THIS invocation
# started the VM and then dies before ensure-up succeeds (ssh never comes
# up, resolve fails, ...), the freshly-RUNNING VM must not be left
# billing. Installed right after a successful start, cleared on success.
_ensure_up_failure_stop() {
  log "ensure-up FAILED after starting the VM -> stopping it again (failure-only trap)"
  if cmd_stop --verify; then
    log "failure-stop complete (verified TERMINATED)"
  else
    log "CRITICAL: failure-stop could not verify TERMINATED — the VM may still be BILLING; stop it manually: capeval_sandbox.sh stop --verify"
  fi
}

cmd_ensure_up() {
  local state
  state="$(_status)"
  case "$state" in
    RUNNING)
      log "instance already RUNNING — no start issued (idempotent)";;
    TERMINATED|STOPPED)
      log "instance $state -> starting"
      "$GCLOUD" compute instances start "$VM" --zone "$ZONE" >&2 \
        || die "gcloud start failed"
      # we just started it: from here until success, a failure stops it
      trap '_ensure_up_failure_stop' EXIT;;
    *)
      die "instance in unexpected state '$state' (want RUNNING/TERMINATED); not proceeding";;
  esac
  local ip deadline
  ip="$(_resolve_ip)"
  log "internal IP (re-resolved, never cached): $ip"
  deadline=$(( $(date +%s) + SSH_DEADLINE ))
  until _ssh "$ip" true 2>/dev/null; do
    if [ "$(date +%s)" -ge "$deadline" ]; then
      die "ssh to $SSH_USER@$ip not up within ${SSH_DEADLINE}s. If the key is
missing on the VM: gcloud compute ssh $VM --zone $ZONE, then append
~/.ssh/id_ed25519_mac.pub to ~/.ssh/authorized_keys there (plan §10)."
    fi
    sleep "$SSH_POLL"
  done
  trap - EXIT         # success: the caller owns the lifecycle from here
  log "ssh is up"
  echo "$ip"          # contract: last stdout line = the IP
}

cmd_provision() {
  # Idempotent re-check at each start (spec #7): every step probes before
  # acting; a fully-provisioned VM makes this a fast no-op.
  local ip
  ip="$(_resolve_ip)"
  if ! _ssh "$ip" "command -v docker >/dev/null"; then
    log "docker missing -> installing docker.io (first provision only)"
    _ssh "$ip" "sudo apt-get update -qq && sudo apt-get install -y -qq docker.io && sudo usermod -aG docker \$USER" \
      || die "docker install failed"
  fi
  _ssh "$ip" "sudo test -f $PROVISION_MARKER || sudo touch $PROVISION_MARKER" \
    || die "cannot write provision marker"
  if ! _ssh "$ip" "docker image inspect '$IMAGE_DIGEST' >/dev/null 2>&1"; then
    log "evalplus image not present at pinned digest -> pulling"
    _ssh "$ip" "docker pull '$IMAGE_DIGEST'" || die "image pull failed"
  fi
  # sandbox->cluster identity: the machine provisioning tool copied ~/.ssh, but its config
  # points node* at id_ed25519_mac — an AGENT-HELD passphrase key whose
  # agent does not exist on the VM, so BatchMode auth fails. The
  # passphrase-FREE id_ed25519 is also on the VM and cluster-authorized;
  # IdentityFile directives accumulate across matching Host blocks, so
  # appending one block makes direct transfers to the login host work
  # with NO new credential minted (~/.ssh was kept for exactly this
  # tunnel). Idempotent.
  if ! _ssh "$ip" "grep -q '# capeval provision: passphrase-free cluster identity' ~/.ssh/config"; then
    log "adding passphrase-free cluster IdentityFile to the sandbox ssh config"
    _ssh "$ip" "printf '\n# capeval provision: passphrase-free cluster identity\nHost node*\n    IdentityFile ~/.ssh/id_ed25519\n' >> ~/.ssh/config" \
      || log "WARN: could not extend sandbox ssh config (direct transfers will fall back via the grading host)"
  fi
  # Pre-seed evalplus problem data for the container run (§5.5): the image
  # bundles its own copy, but the host-side cache lets --samples runs skip
  # any in-container download. Copied sandbox<-grading-host once (a 5MB jsonl —
  # not a payload in the F19 sense).
  if ! _ssh "$ip" "test -s ~/.cache/evalplus/$EVALPLUS_CACHE_FILE"; then
    local local_copy="${XDG_CACHE_HOME:-$HOME/.cache}/evalplus/$EVALPLUS_CACHE_FILE"
    if [ -s "$local_copy" ]; then
      log "seeding evalplus data cache on the sandbox"
      _ssh "$ip" "mkdir -p ~/.cache/evalplus"
      "${CAPEVAL_SCP:-scp}" -q "$local_copy" "$SSH_USER@$ip:.cache/evalplus/" \
        || log "WARN: evalplus data seed failed (container will fetch on demand)"
    else
      log "WARN: no local evalplus data at $local_copy (run capeval_fetch_data.py); container will fetch on demand"
    fi
  fi
  log "provision re-check complete; pinned image digest follows"
  echo "$IMAGE_DIGEST"   # contract: last stdout line = the digest
}

cmd_run() {
  [ "$#" -ge 1 ] || die "run needs a command"
  [ "$1" = "--" ] && shift
  local ip
  ip="$(_resolve_ip)"
  _ssh "$ip" "$@"
}

cmd_stop() {
  local verify=0
  [ "${1:-}" = "--verify" ] && verify=1
  "$GCLOUD" compute instances stop "$VM" --zone "$ZONE" >&2 \
    || die "gcloud stop failed — the VM may still be billing; stop it manually"
  if [ "$verify" -eq 1 ]; then
    local state
    state="$(_status)"
    [ "$state" = "TERMINATED" ] \
      || die "post-stop state is '$state', not TERMINATED — verify manually"
    log "verified TERMINATED"
  fi
}

case "${1:-}" in
  ensure-up) shift; cmd_ensure_up "$@" ;;
  provision) shift; cmd_provision "$@" ;;
  run)       shift; cmd_run "$@" ;;
  stop)      shift; cmd_stop "$@" ;;
  status)    shift; _status ;;
  *) die "usage: capeval_sandbox.sh {ensure-up|provision|run -- <cmd...>|stop [--verify]|status}" ;;
esac
