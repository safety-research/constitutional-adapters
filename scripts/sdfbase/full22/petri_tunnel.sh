#!/bin/bash
# petri_tunnel.sh — the grading host's Petri drive tunnel verdicts (sourced by petri_drive_full22.sh).
#
# The drive forwards a per-(target, view) local port to the proxy's node. Whether a holder of that port is OUR tunnel to THIS
# attempt's proxy is decided here from the holder's command line — by the EXACT node and the EXACT proxy port the forward
# names, never by a substring: `*" $NODE"*` also matches a node whose name merely EXTENDS this one's, so a previous attempt's
# tunnel passed as ours and the drive read /v1/models off a dead forward twice (2026-10-03, proxy 864524 after the cancelled 863015).
#
#   f22_tunnel_holder_verdict "<pgrep -fa line>" NODE PPORT LOCALPORT   -> reuse | stale | ambiguous | foreign
#     reuse     : an ssh carrying EXACTLY ONE forward, and that forward binds LOCALPORT to 127.0.0.1:PPORT with NODE the LAST argument
#                 (`ssh … -L [<bind>:]<LOCALPORT>:127.0.0.1:<PPORT> [-J <jump>] <NODE>`) — the drive's own tunnel shape
#     stale     : one forward that binds LOCALPORT elsewhere — another node, another proxy port, a destination that is not the node's
#                 loopback (an earlier attempt's tunnel): close, re-create
#     ambiguous : an ssh carrying MORE THAN ONE forward. The drive opens one -L, so this is never its tunnel: it is
#                 neither reused (the forward that binds LOCALPORT need not be the one that matched — the verdict used to keep the LAST
#                 -L and read `reuse` off a line whose forward on OUR port went to the re-run pass's proxy) nor closed (its other
#                 forwards are not ours); the drive refuses it by name
#     foreign   : anything else, or nothing — not ssh, no forward, a forward that does not bind LOCALPORT (pgrep matched the port's text
#                 elsewhere on the line), a malformed forward; the port belongs to someone else: refuse
#   f22_tunnel_holder_line LOCALPORT   -> "<pid> <argv…>" of the process LISTENING on 127.0.0.1:LOCALPORT — the SOCKET's owner (ss -ltnp) and
#                                         its exact argv (/proc/<pid>/cmdline; `ps -ww -o args=` as the fallback), the pid-first line shape the
#                                         verdict parses — or nothing (rc 0) when the port is free, when the IPv4 loopback socket has no visible
#                                         owner, when no socket at the number is the IPv4 loopback one, or (tunnel-fix-4) when the listening
#                                         sockets at the number belong to MORE THAN ONE pid (127.0.0.1 and ::1 held by different processes:
#                                         neither reused nor closed — the drive refuses it as foreign) — before that fix the FIRST pid ss listed
#                                         was taken, so a stale verdict could close the wrong tunnel; the common dual-stack bind of ONE ssh
#                                         (127.0.0.1 and ::1, one pid) is still one holder
#                                         free or its owner is not visible (another user's process). The drive used to find
#                                         its holder with `pgrep -fa "ssh .*-L <port>:127.0.0.1:"`, one spelling of the forward, so a tunnel in
#                                         the bind-address form (`-L 127.0.0.1:<port>:…`) or the glued form (`-L<port>:…`) was never found: the
#                                         port read held with no holder and the drive refused it as foreign instead of reusing or re-creating it
#   f22_tunnel_close_stale "<pgrep -fa line>" LOCALPORT   -> kills the forward by the pid the line starts with, waits for the
#                                                           local port to free; rc 0 freed, rc 1 not freed / no pid on the line
#   f22_tunnel_teardown                                    -> kills the tunnel THIS run opened ($TUNNEL_PID), once; a reused
#                                                           tunnel (TUNNEL_PID empty) is left to its opener
f22_tunnel_holder_line() {
  local lport="${1:?local port}" pid="" args="" rows="" owners=""
  rows=$(ss -ltnpH "sport = :${lport}" 2>/dev/null)
  [ -n "$rows" ] || return 0
  # tunnel-fix-4: EVERY listening socket at the number must belong to ONE visible pid — distinct owners on 127.0.0.1 and ::1 are not one holder
  owners=$(printf '%s\n' "$rows" | grep -oE 'pid=[0-9]+' | cut -d= -f2 | sort -u)
  [ "$(printf '%s\n' "$owners" | grep -c .)" -eq 1 ] || return 0
  # the socket read is the exact IPv4 loopback one (the drive binds 127.0.0.1); its owner must be visible (another user's process is not)
  pid=$(printf '%s\n' "$rows" | awk -v want="127.0.0.1:${lport}" '$4 == want' | grep -oE 'pid=[0-9]+' | head -n 1 | cut -d= -f2)
  [ -n "$pid" ] || return 0
  # the EXACT argv: /proc/<pid>/cmdline (NUL-separated, never truncated); `ps -o args=` without -ww cuts the line at the terminal width
  # (80 columns under a captured stdout), which lost the drive's own -L token behind ssh's -o options (chain 294 attempt 1, 2026-10-03)
  if [ -r "/proc/$pid/cmdline" ]; then args=$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | sed 's/ $//'); fi
  [ -n "$args" ] || args=$(ps -ww -o args= -p "$pid" 2>/dev/null) || return 0
  [ -n "$args" ] || return 0
  printf '%s %s\n' "$pid" "$args"
}

f22_tunnel_holder_verdict() {
  local line="${1:-}" node="${2:?node}" pport="${3:?proxy port}" lport="${4:?local port}"
  [ -n "$line" ] || { echo foreign; return 0; }
  # shellcheck disable=SC2206
  local -a tok=($line) fwds=() parts=()
  local n=${#tok[@]} i fwd="" last="" lhost="" remote=""
  [ "$n" -ge 3 ] || { echo foreign; return 0; }
  case "${tok[1]##*/}" in ssh) ;; *) echo foreign; return 0 ;; esac   # tok[0] is the pid pgrep -fa prints; the program must be ssh
  for ((i = 2; i < n; i++)); do                                        # EVERY forward on the line, in order (never just the last)
    case "${tok[i]}" in
      -L) fwds+=("${tok[i+1]:-}"); i=$((i + 1)) ;;
      -L*) fwds+=("${tok[i]#-L}") ;;
    esac
  done
  [ "${#fwds[@]}" -ge 1 ] || { echo foreign; return 0; }
  [ "${#fwds[@]}" -eq 1 ] || { echo ambiguous; return 0; }             # the drive's tunnel has ONE forward; anything wider is not ours
  fwd="${fwds[0]}"
  IFS=: read -r -a parts <<< "$fwd"                                    # <local>:<host>:<remote> or <bind>:<local>:<host>:<remote>
  case "${#parts[@]}" in
    3) [ "${parts[0]}" = "$lport" ] || { echo foreign; return 0; }; lhost="${parts[1]}"; remote="${parts[2]}" ;;
    4) [ "${parts[1]}" = "$lport" ] || { echo foreign; return 0; }; lhost="${parts[2]}"; remote="${parts[3]}" ;;
    *) echo foreign; return 0 ;;
  esac
  last="${tok[n-1]}"
  if [ "$last" = "$node" ] && [ "$remote" = "$pport" ] && { [ "$lhost" = "127.0.0.1" ] || [ "$lhost" = "localhost" ]; }; then echo reuse; else echo stale; fi
}

f22_tunnel_close_stale() {
  local line="${1:-}" lport="${2:?local port}" pid i
  pid="${line%% *}"
  case "$pid" in ''|*[!0-9]*) return 1 ;; esac
  kill "$pid" 2>/dev/null
  for i in $(seq 1 20); do ss -ltn 2>/dev/null | grep -q ":${lport} " || return 0; sleep 0.5; done
  kill -9 "$pid" 2>/dev/null
  for i in $(seq 1 10); do ss -ltn 2>/dev/null | grep -q ":${lport} " || return 0; sleep 0.5; done
  return 1
}

f22_tunnel_teardown() {
  [ -n "${TUNNEL_PID:-}" ] || return 0
  if kill -0 "$TUNNEL_PID" 2>/dev/null; then kill "$TUNNEL_PID" 2>/dev/null; wait "$TUNNEL_PID" 2>/dev/null; fi
  if type log >/dev/null 2>&1; then log "tunnel (pid $TUNNEL_PID) closed"; fi
  TUNNEL_PID=""
}
