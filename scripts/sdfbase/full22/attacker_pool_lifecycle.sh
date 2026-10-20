#!/bin/bash
# full22 — the attacker pool MEMBER's own lifecycle (31b): a member closes itself
# at its LAST CONSUMER, and a member nothing ever bound closes itself once the group's placement is COMPLETE — event-keyed on the
# consumers' exit, inside the job, so it needs no session alive. The group-level `teardown-pool` waiter stays as the backstop.
#
# Evidence, all on the wave's shared root (nothing is passed in and kept in sync):
#   $WAVE_ROOT/serving/attacks_<model>_<view>_<attack>_<jobid>/attacker_binding.json   the binding a TARGET took: {"urls": [...]}; the
#                                                                                      target's job id is the dir name's last `_` field
#   $WAVE_ROOT/attackers/PLACED           written by the funnel (`submit_twin.py mark-placed --group G`) when the group's LAST target was
#                                         submitted: {"group", "targets": [{"job_id", "logical_work_id"}], "utc"} — before it, a target
#                                         that has not bound yet may still be coming
#   sacct / squeue                        a consumer's terminal state (Slurm's, not a heuristic)
# Sourced by attacker_pool.sbatch; every function prints its answer on stdout and nothing else.
#   f22_pool_consumers  <wave_root> <my_url>            -> the job ids whose binding names my URL, one per line
#   f22_pool_placed_targets <pool_dir>                   -> the PLACED targets' job ids, one per line; rc 1 when PLACED is absent
#   f22_pool_jobs_terminal <job id...>                   -> rc 0 when EVERY named job is terminal on sacct (none active on squeue), 1 otherwise,
#                                                           2 when sacct/squeue could not answer (undecidable: keep serving)
#   f22_pool_assigned_targets <pool_dir> <my_job_id>     -> the PLACED targets the funnel ASSIGNED me (assigned_member_job_id), one per line
#   f22_pool_assigned_member_url <wave_root> <target_job> -> the URL of the member PLACED assigned a target (read by target_attacks.sbatch)
#   f22_pool_unbound_live_targets <wave_root> <job id...> -> the placed targets with no binding yet that are not terminal
#   f22_pool_record_binding <serve_dir> <assigned_url|""> <live url...> -> rewrites the target's attacker_binding.json: `urls` = the CONSUMER set
#                                                           (the assigned member, else the whole live pool), `candidates` = the other live members
#   f22_pool_member_verdict <wave_root> <my_url> [<my_job_id>] -> serve | exit_last_consumer | exit_unbound, with the reason on stderr
#   f22_pool_member_decide <wave_root> <my_url> <my_job_id|""> <consumers> -> one decision over a GIVEN consumer set, into F22_VERDICT /
#                                                           F22_VERDICT_WHY (the verdict's inner step; prints nothing)
#   f22_pool_placement_fingerprint <pool_dir> <my_job_id|""> -> ONE line: PLACED's version, its target jobs and the ones assigned me
#                                                           (`absent` without a record) — what the verdict compares before and after a decision
#   f22_pool_placement_in_transition <pool_dir>          -> rc 0 while attackers/PLACED.transition is FRESH (a pass is re-placing the group:
#                                                           no exit), 1 absent, 2 lapsed, 3 unreadable; a description on stdout
#   The rule (31b-fix): no member leaves while the placement is incomplete or any placed target is live and
#   unbound — a target's binding is written at ITS runtime, so "my consumers ended" says nothing about a target still queued or starting.
#   And no member answers an exit on a STALE read (placement-fix):
#   the decision's sacct / squeue calls are slow, so a target that binds me between the first read and the decision — or a PLACED the funnel
#   advanced meanwhile, assigning me targets still queued (no binding on either read) — would be missed; the verdict re-reads my consumers AND
#   the placement (version + assignments) immediately before answering exit_last_consumer / exit_unbound and decides again when either changed.
#   And no member leaves while the group's placement is IN TRANSITION: the --resubmit pass submits every replacement BEFORE it supersedes
#   PLACED, so a replacement member that comes up in between reads the record that predates it (nothing assigned to it) — the pass writes
#   attackers/PLACED.transition before its first submission (submit_twin.py placement-transition open), mark-placed clears it when a placement
#   commits, and an exit is held while it is fresh; once its lease lapsed (the bound: a pass that never re-placed the group cannot pin a member
#   for ever) it is ignored and said so.
PY_LIFECYCLE="${PY_LIFECYCLE:-${MPY:-python3}}"

f22_pool_retire_requested(){   # $1 = pool dir, $2 = my job id -> rc 0 when the funnel marked me RETIRING (<job>.RETIRE, 31d)
  [ -e "$1/$2.RETIRE" ]
}

f22_pool_consumers(){   # $1 = wave root, $2 = my URL
  local root="$1" url="$2" d
  for d in "$root"/serving/attacks_*; do
    [ -f "$d/attacker_binding.json" ] || continue
    if "$PY_LIFECYCLE" -c 'import json,sys; sys.exit(0 if sys.argv[2] in (json.load(open(sys.argv[1])).get("urls") or []) else 1)' "$d/attacker_binding.json" "$url" 2>/dev/null; then
      echo "${d##*_}"
    fi
  done
}

f22_pool_placed_targets(){   # $1 = pool dir
  local p="$1/PLACED"
  [ -f "$p" ] || return 1
  "$PY_LIFECYCLE" -c 'import json,sys; [print(str(t["job_id"])) for t in (json.load(open(sys.argv[1])).get("targets") or []) if t.get("job_id")]' "$p" 2>/dev/null || return 1
}

f22_pool_jobs_terminal(){   # $@ = job ids
  [ $# -ge 1 ] || return 0
  local ids; ids=$(IFS=,; echo "$*")
  local sq; sq=$(squeue -h -j "$ids" -o %T 2>/dev/null); local rc=$?
  if [ $rc -eq 0 ] && [ -n "$sq" ]; then return 1; fi                        # something still active (PENDING/RUNNING/...)
  local sa; sa=$(sacct -X -n -P -o JobID,State -j "$ids" 2>/dev/null) || return 2
  [ -n "$sa" ] || return 2                                                    # sacct knows none of them: undecidable, keep serving
  local n_known; n_known=$(printf '%s\n' "$sa" | awk -F'|' 'NF>=2 && $1!=""{print $1}' | sort -u | wc -l)
  [ "$n_known" -ge $# ] || return 2                                            # a job sacct does not know yet: undecidable
  printf '%s\n' "$sa" | awk -F'|' '{print $2}' | grep -qE '^(PENDING|RUNNING|REQUEUED|RESIZING|SUSPENDED|COMPLETING|CONFIGURING)' && return 1
  return 0
}

f22_pool_assigned_targets(){   # $1 = pool dir, $2 = my job id -> the PLACED targets the funnel ASSIGNED me, one per line; rc 1 when PLACED is absent
  local p="$1/PLACED"
  [ -f "$p" ] || return 1
  "$PY_LIFECYCLE" -c 'import json,sys; [print(str(t["job_id"])) for t in (json.load(open(sys.argv[1])).get("targets") or []) if t.get("job_id") and str(t.get("assigned_member_job_id") or "") == sys.argv[2]]' "$p" "$2" 2>/dev/null || return 1
}

f22_pool_assigned_member_url(){   # $1 = wave root, $2 = a TARGET job id -> the URL of the member PLACED assigned it (from the member's record); rc 1 when none / not live
  local p="$1/attackers/PLACED"
  [ -f "$p" ] || return 1
  local mj; mj=$("$PY_LIFECYCLE" -c 'import json,sys; t=[t for t in (json.load(open(sys.argv[1])).get("targets") or []) if str(t.get("job_id"))==sys.argv[2]]; print(str((t[0].get("assigned_member_job_id") or "")) if t else "")' "$p" "$2" 2>/dev/null) || return 1
  [ -n "$mj" ] && [ -f "$1/attackers/$mj.json" ] || return 1
  "$PY_LIFECYCLE" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("url") or "")' "$1/attackers/$mj.json" 2>/dev/null | grep . || return 1
}

f22_pool_record_binding(){   # $1 = the target's serve dir (its attacker_binding.json), $2 = the ASSIGNED member's URL or "", $3.. = every live member URL
  # 31b-fix-2: the binding's `urls` is the CONSUMER set — the member(s) this job's drivers actually bind — never the
  # whole live pool f22_live_attackers saw before the assignment was chosen; the other live members are `candidates` (the failover list,
  # 31d). The members' lifecycle counts a target as a consumer through `urls` only, so A's member leaves when A ends although B, bound to
  # another member, still runs. Prints "<assignment> <n urls> <n candidates>".
  local dir="$1" assigned="$2"; shift 2
  "$PY_LIFECYCLE" - "$dir/attacker_binding.json" "$assigned" "$@" <<'PY'
import json, os, sys, time
path, assigned, live = sys.argv[1], sys.argv[2], sys.argv[3:]
doc = {}
if os.path.isfile(path):
    try:
        doc = json.load(open(path))
    except Exception:
        doc = {}
if assigned and assigned in live:
    urls, cands, mode = [assigned], [u for u in live if u != assigned], "placed"
else:
    urls, cands, mode = list(live), [], "round_robin"
doc.update({"urls": urls, "candidates": cands, "assigned_member_url": assigned or None, "assignment": mode,
            "rewritten_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
tmp = path + ".tmp"
with open(tmp, "w") as fh:
    json.dump(doc, fh, sort_keys=True); fh.write("\n")
os.replace(tmp, path)
print(mode, len(urls), len(cands))
PY
}

f22_pool_unbound_live_targets(){   # $1 = wave root, $2.. = placed target job ids -> the placed targets with NO binding yet that are not terminal (live or undecidable), one per line
  local root="$1"; shift
  local j; for j in "$@"; do
    [ -f "$root/serving/"*"_$j/attacker_binding.json" ] 2>/dev/null && continue
    f22_pool_jobs_terminal "$j"; [ $? -eq 0 ] || echo "$j"
  done
}

f22_pool_placement_fingerprint(){   # $1 = pool dir, $2 = my job id or "" -> ONE line: the record's version, its target jobs, the ones assigned me; `absent` without PLACED
  local p="$1/PLACED"
  [ -f "$p" ] || { echo absent; return 0; }
  "$PY_LIFECYCLE" -c 'import json, sys
d = json.load(open(sys.argv[1])); me = sys.argv[2]; ts = [t for t in (d.get("targets") or []) if t.get("job_id")]
print("v%s targets=%s mine=%s" % (d.get("version") or 1, ",".join(str(t["job_id"]) for t in ts), ",".join(str(t["job_id"]) for t in ts if me and str(t.get("assigned_member_job_id") or "") == me)))' "$p" "$2" 2>/dev/null || echo unreadable
}

f22_pool_placement_in_transition(){   # $1 = pool dir -> rc 0 fresh / 1 absent / 2 lapsed / 3 unreadable; stdout: "since <utc> for <lease> s (<left> s left | lapsed <n> s ago): <reason>"
  # the marker's epoch is the CLUSTER's clock (written there by the funnel's transaction), compared with this node's — never the grading host's
  local p="$1/PLACED.transition"
  [ -f "$p" ] || return 1
  "$PY_LIFECYCLE" -c 'import json, sys, time
try:
    d = json.load(open(sys.argv[1])); left = float(d["epoch"]) + float(d["lease_s"]) - time.time()
except Exception as e:
    print("unreadable (%s)" % e); sys.exit(3)
print("since %s for %d s (%s): %s" % (d.get("utc"), float(d["lease_s"]), ("%d s left" % left) if left > 0 else ("lapsed %d s ago" % -left), d.get("reason") or "no reason"))
sys.exit(0 if left > 0 else 2)' "$p"
}

f22_pool_transition_fence(){   # $1 = pool dir; applied to an EXIT verdict in F22_VERDICT: held (`serve`) while the group's placement is in transition — a fresh marker; a lapsed or unreadable one is noted and ignored
  local tr; tr=$(f22_pool_placement_in_transition "$1"); local rc=$?
  case $rc in
    0) F22_VERDICT_WHY="the group's placement is in TRANSITION (attackers/PLACED.transition $tr): a pass is re-placing the group and the next PLACED may assign me targets — holding this tick (would have answered $F22_VERDICT: $F22_VERDICT_WHY)"; F22_VERDICT=serve;;
    2) F22_VERDICT_WHY="$F22_VERDICT_WHY (a placement-transition marker $tr lapsed its lease: ignored)";;
    3) F22_VERDICT_WHY="$F22_VERDICT_WHY (an attackers/PLACED.transition marker that cannot be read is ignored: $tr)";;
  esac
}

f22_pool_member_decide(){   # $1 = wave root, $2 = my URL, $3 = my job id or "", $4 = my consumers (one per line) -> F22_VERDICT + F22_VERDICT_WHY
  # ONE decision over a GIVEN consumer set; prints nothing (the verdict below owns stdout/stderr, so it can decide again on a fresh set)
  local root="$1" url="$2" me="${3:-}" consumers="${4:-}" pool="$1/attackers"
  local placed; placed=$(f22_pool_placed_targets "$pool"); local placed_rc=$?
  # while the placement is NOT complete, a target may still come and bind me — however many earlier consumers
  # have ended, I serve (before, every consumer ending was read as my end)
  if [ $placed_rc -ne 0 ]; then
    if [ -n "$consumers" ]; then F22_VERDICT=serve; F22_VERDICT_WHY="consumer(s) $(echo $consumers | tr '\n' ' ')recorded but the group's placement is not complete (no PLACED): a target may still come and bind me"; return 0; fi
    F22_VERDICT=serve; F22_VERDICT_WHY="no binding names me yet and the group's placement is not complete (no PLACED): a target may still come"; return 0
  fi
  # placement complete: a placed target that has neither bound anyone nor ended (queued, or starting: its binding is written at ITS runtime)
  # keeps EVERY member — with or without earlier consumers (A terminal, B pending or starting)
  # shellcheck disable=SC2086
  local unbound; unbound=$(f22_pool_unbound_live_targets "$root" $placed)
  if [ -n "$unbound" ]; then F22_VERDICT=serve; F22_VERDICT_WHY="placement complete, but placed target(s) $(echo $unbound | tr '\n' ' ')have neither bound nor ended (queued or starting): every member stays"; return 0; fi
  # my consumers: the targets whose binding names me UNION the targets the funnel assigned me at placement (immutable, from PLACED)
  local assigned=""; [ -n "$me" ] && assigned=$(f22_pool_assigned_targets "$pool" "$me")
  local mine; mine=$(printf '%s\n%s\n' "$consumers" "$assigned" | grep . | sort -u)
  if [ -n "$mine" ]; then
    # shellcheck disable=SC2086
    f22_pool_jobs_terminal $mine; local t=$?
    case $t in
      0) F22_VERDICT=exit_last_consumer; F22_VERDICT_WHY="placement complete and every consumer terminal on sacct (bound or assigned): $(echo $mine | tr '\n' ' ')"; f22_pool_transition_fence "$pool"; return 0;;
      1) F22_VERDICT=serve; F22_VERDICT_WHY="consumer(s) active (bound or assigned): $(echo $mine | tr '\n' ' ')"; return 0;;
      *) F22_VERDICT=serve; F22_VERDICT_WHY="consumers' state undecidable (sacct/squeue did not answer): keeping the member"; return 0;;
    esac
  fi
  F22_VERDICT=exit_unbound; F22_VERDICT_WHY="placement complete ($(echo $placed | wc -w) target(s)), every target bound elsewhere or ended, none bound me (none assigned me either)"
  # placement-fix round 3: an exit is HELD while the group's placement is in transition (the marker is read here, immediately before the answer)
  f22_pool_transition_fence "$pool"
}

f22_pool_member_verdict(){   # $1 = wave root, $2 = my URL, [$3 = my job id — reads the funnel's assignment in PLACED]
  local root="$1" url="$2" me="${3:-}"
  local consumers fresh placement fresh_placement rounds=0
  consumers=$(f22_pool_consumers "$root" "$url"); placement=$(f22_pool_placement_fingerprint "$root/attackers" "$me")
  while :; do
    F22_VERDICT=""; F22_VERDICT_WHY=""
    f22_pool_member_decide "$root" "$url" "$me" "$consumers"
    case "$F22_VERDICT" in
      exit_last_consumer|exit_unbound)
        # placement-fix: the decision above ran slow sacct / squeue calls over the placed targets and
        # my consumers; a target that bound me MEANWHILE (its binding is written at its runtime) is missing from the set it decided over. Re-read
        # my consumers immediately before answering an exit; a changed set is decided again, over the fresh set — never an exit on a stale read.
        # The PLACEMENT too — a PLACED the funnel advanced during the decision (a --resubmit pass's
        # supersession) may assign me targets still QUEUED, whose bindings do not exist on either consumer read; a changed version, target set
        # or assignment to me is decided again over the record the decision now reads — never an exit on the placement that predates the pass.
        fresh=$(f22_pool_consumers "$root" "$url"); fresh_placement=$(f22_pool_placement_fingerprint "$root/attackers" "$me")
        if [ "$fresh" != "$consumers" ] || [ "$fresh_placement" != "$placement" ]; then
          rounds=$((rounds + 1))
          if [ $rounds -ge 3 ]; then echo "serve"; echo "my consumer set or the placement changed under the decision $rounds times (now: consumers $(echo $fresh | tr '\n' ' '); placement $fresh_placement) — keeping the member this tick; the next tick decides" >&2; return 0; fi
          [ "$fresh" != "$consumers" ] && echo "re-check: consumer set changed under the decision ($F22_VERDICT over $(echo $consumers | tr '\n' ' ')-> now $(echo $fresh | tr '\n' ' ')): deciding again" >&2
          [ "$fresh_placement" != "$placement" ] && echo "re-check: placement changed under the decision ($F22_VERDICT over $placement -> now $fresh_placement): deciding again" >&2
          consumers="$fresh"; placement="$fresh_placement"; continue
        fi;;
    esac
    echo "$F22_VERDICT"; echo "$F22_VERDICT_WHY" >&2; return 0
  done
}

# 31d-fix-3: a member PROVES its health with a FRESH lease it writes itself — <pool>/<job>.health
# {job_id, url, ok, utc, lease_s} — right after its record and on every loop tick after its probe; the funnel's retirement transaction counts a
# survivor only with a fresh ok lease (fail closed: no lease, a stale one, or ok=false is no server). A vLLM pid can live while its endpoint
# answers nothing: after F22_POOL_HEALTH_MAX_FAILS consecutive failed probes the member withdraws its record at the source and leaves a
# <job>.UNHEALTHY tombstone, which the transaction also honours.
#   f22_pool_health_write <pool_dir> <my_job_id> <my_url> <ok 0|1> <lease_s>   -> the lease, written atomically
#   f22_pool_member_unhealthy <pool_dir> <my_job_id> <my_url> <why>            -> record + pending marker withdrawn, <job>.UNHEALTHY written
f22_pool_health_write(){   # $1 = pool dir, $2 = my job id, $3 = my URL, $4 = ok (0|1), $5 = lease seconds
  local p="$1/$2.health" tmp="$1/.$2.health.tmp.$$" ok=false
  [ "$4" = 1 ] && ok=true
  printf '{"job_id":"%s","url":"%s","ok":%s,"utc":"%s","lease_s":%s}\n' "$2" "$3" "$ok" "$(date -u +%FT%TZ)" "$5" > "$tmp" && mv -f "$tmp" "$p"
}

f22_pool_member_unhealthy(){   # $1 = pool dir, $2 = my job id, $3 = my URL, $4 = why
  rm -f "$1/$2.json" "$1/$2.pending"
  printf '{"job_id":"%s","url":"%s","why":"%s","utc":"%s"}\n' "$2" "$3" "${4//\"/}" "$(date -u +%FT%TZ)" > "$1/$2.UNHEALTHY"
}
