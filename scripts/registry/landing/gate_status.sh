#!/bin/bash
# gate_status.sh <40-hex head sha> "<description>" [state=success]   |   gate_status.sh --help
# Posts the `landing-gate` commit status on <head sha> in the chain's repository (LAND_REPO, else `land.sh --repo` — the one place the
# slug is defined; no second copy lives here). Repo ruleset 24155420, "main: landing gate passed on an up-to-date branch" (added
# 2026-09-29 to close the up-to-date-branch race on main; created disabled, activated once every landing path posts), requires this
# context in state success on the PR head, STRICT (the head must contain main's tip), with no bypass actors. ONE helper
# for every landing path: land.sh calls it right after pushing the merged head and before `gh pr create` / `gh pr merge`; a session
# landing a branch by hand posts the same status after its own Slurm gate. Post it only for a head whose FULL SUITE verdict is LAND
# (or a user waiver) on THIS tree; the description names the evidence (`slurm gate LAND on tree of <sha8> vs baseline <main8>`).
# Interchangeable with the interim helper in the operating tree (same arguments, exits, output line).
# Exits 2 on a sha that is not 40 hex (a short sha names nothing on the status API) or a state the API does not know; truncates the
# description to 140 characters with an ellipsis (the API's limit). The API's response is CHECKED — its context must read landing-gate
# and its state the one asked for, else exit 3 (a `gh api` failure keeps gh's own exit status) — and ONE line is printed,
# `landing-gate <state> on <sha8>: <description>`, the sha8 the SUPPLIED sha's (the response's `url` ends in the status record, not the
# commit), never a token (GH_TOKEN reaches `gh` through the environment). Knobs (env): LAND_GH (gh), LAND_REPO.
set -eu
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
case "${1:-}" in -h|--help) awk 'NR == 1 {next} /^#/ {print; next} {exit}' "$0"; exit 0 ;; esac
[ $# -ge 2 ] && [ $# -le 3 ] || { echo "usage: gate_status.sh <40-hex head sha> \"<description>\" [state=success]" >&2; exit 2; }
SHA=$1; DESC=$2; STATE=${3:-success}
[[ "$SHA" =~ ^[0-9a-f]{40}$ ]] || { echo "gate_status: need the FULL 40-hex head sha (got '$SHA')" >&2; exit 2; }
case "$STATE" in success|failure|pending|error) ;; *) echo "gate_status: state must be success|failure|pending|error (got '$STATE')" >&2; exit 2 ;; esac
[ ${#DESC} -le 140 ] || DESC=${DESC:0:137}...
GH=${LAND_GH:-gh}
REPO=${LAND_REPO:-$("$HERE/land.sh" --repo)}
resp=$("$GH" api -X POST "repos/$REPO/statuses/$SHA" -f state="$STATE" -f context=landing-gate -f description="$DESC" \
  --jq '[.context, .state, .description] | join("\t")')   # set -e: a failed call exits here with gh's status
IFS=$'\t' read -r rctx rstate rdesc <<<"$resp"
[ "$rctx" = landing-gate ] && [ "$rstate" = "$STATE" ] || { echo "gate_status: the API answered context=$rctx state=$rstate, not landing-gate $STATE — NOT confirmed" >&2; exit 3; }
echo "landing-gate $rstate on ${SHA:0:8}: $rdesc"
