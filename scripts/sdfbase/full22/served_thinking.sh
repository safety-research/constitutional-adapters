#!/bin/bash
# The TASK-INDEPENDENT execution core of the served thinking-ON control launchers (bash), sourced AFTER
# scripts/sdfbase/full22/common.sh: the seams math500-thinking-on-full22's m5t_common.sh and
# gsm8k-paired200-paper-objects' g8k_common.sh SHARE (a sibling launcher family is acceptable only
# on ONE execution core). Factored out of m5t_common.sh with BYTE-IDENTICAL behaviour under the m5t_* names, which now
# delegate here; the task-specific parts (the matrix row, the overlay names, the standup extras, the driver's task
# flags, the dry-run header lines) stay in each launcher's own file.
#
# Every function takes its inputs POSITIONALLY or from the job variables common.sh / the launcher already set (CPY
# REPO_DIR SERVE_DIR PROXY_URL TABLE OUT_ROOT CELL CFG TT_TWO_SI_DEFENSES_DIR); nothing here reads an M5T_* / G8K_*
# name, so neither launcher's literals leak into the other. NEVER --export= (hard rule 1).
#
# Loading: each launcher's `<x>_core` loader sources this file from the job's checkout ($REPO_DIR, set by the sbatch
# prologue before either helper is sourced) on FIRST USE and refuses a tree without it by name — so a launcher's
# function block still works when taken alone (the m5t test runs the ledger block by itself), and a checkout missing
# the core dies with f22_fatal after the cleanup trap is armed, never `command not found` mid-job.

# --- the prompt-budget artifact --------------------------------------------------------------------------------
st_budget_artifact(){   # $1 = the pre-launch CPU artifact, $2 = its copy in the out root, $3 = the remedy text of the refusal
                        # -> BUDGET_TEMPLATE_HASH (the template digest the endpoint check holds the standup to)
  local src="${1:?budget artifact}" dst="${2:?destination}" remedy="${3:-re-run the prompt-budget script before launch}"
  [ -f "$src" ] || f22_fatal "no prompt-budget artifact at $src — $remedy"
  cp "$src" "$dst"
  BUDGET_TEMPLATE_HASH=$("$CPY" -c "import json,sys;print(json.load(open(sys.argv[1]))['chat_template_hash'])" "$dst")
}

# --- the post-standup checks: the timeout, the template, the FRESH table's served-positions stamp ----------
st_external_bank_view_check(){   # $1 = the bank view RELATIVE to the shared root, $2 = the cell dir inside it, $3 = the
                                 # pinned object.pt sha256  -> exports TT_TWO_SI_DEFENSES_DIR
  # These steering rows serve an object ANOTHER LANE staged — the AlphaSteer canonical
  # ladder's selected rung, the Arditi strength lane's selected alpha — so the campaign's f22_bank_view_check cannot be
  # used: it resolves (target, view) through the FULL22 roster, which names neither bank, and would refuse by name on a
  # view that is perfectly valid in its own lane. This check is the same CONTRACT without the roster: the view exists,
  # it carries its VIEW_MANIFEST, the cell dir is there, and the object's BYTES hash to the row's pin (the admission
  # record's — never a directory digest). A drift is FATAL: the standup's --expect-object-sha256 and the driver's
  # attestation compare are the second and third copies of the same fact, and all three must agree before traffic.
  local rel="${1:?bank view rel}" cell_dir="${2:?bank cell dir}" pin="${3:?object sha256}" dir obj have
  dir="${SHARED:?SHARED}/$rel"
  [ -d "$dir" ] || f22_fatal "external bank view $dir does not exist on this cluster — the row names it for this cluster role (cells.py M5T_BANK_VIEW_REL_*); stage the object or run the row where it is staged"
  [ -f "$dir/VIEW_MANIFEST.json" ] || f22_fatal "external bank view $dir carries no VIEW_MANIFEST.json — not a built view; refusing"
  obj="$dir/$cell_dir/object.pt"
  [ -f "$obj" ] || f22_fatal "external bank view $dir lacks $cell_dir/object.pt"
  have=$(sha256sum "$obj" | cut -c1-64)
  [ "$have" = "$pin" ] || f22_fatal "external object $cell_dir/object.pt sha256 $have != the admission record's pin $pin — a drifted or re-staged object; refusing (the pin is the OBJECT's bytes, D5)"
  export TT_TWO_SI_DEFENSES_DIR="$dir"
  f22_log "external bank view OK: $dir; $cell_dir object.pt == pin ${pin:0:12}"
}

st_itdoc_bank_check(){   # $1 = the bank view RELATIVE to the shared root, $2 = the cell dir inside it, $3 = the pinned object sha256,
                         # $4 = the it-doc admission record (repo-relative), $5 = lane, $6 = target (model key), $7 = view, $8 = cell
                         # -> exports TT_TWO_SI_DEFENSES_DIR (through st_external_bank_view_check)
  # The it-doc re-fit rows serve an object whose CELL ID is not unique across lanes (the base lane's
  # `main` cell, the superseded batch-8 direct-fit cell, its dose bakes) and whose STAGED ID is not unique across targets — a run
  # that pinned the re-fit but resolved the cell through the batch-8 bank entry would SERVE the old bytes while REPORTING the new
  # pin. So before the byte check the row is BOUND to its bank entry through the record it was generated from
  # (scripts/sdfbase/full22/itdoc_bind.py bind-check, stdlib-only, from the client venv): the VIEW_MANIFEST entry for the cell names
  # the pin and the row's RULED staged id (never a batch-8 `itdoc__` / `*it__` id — refused BY LANE NAME), the cell dir's symlink
  # resolves to that id, the sidecar OF THE KIND (train_budget_meta.json fitted / derived_meta.json derived, whose parents must be
  # exactly the lane's two parents) records the pin, and the pin file's (target, staged id) agrees. Only then the byte check
  # (st_external_bank_view_check: bytes == pin) — the second door; the standup's and the driver's --expect-object-sha256 and the
  # driver's provenance mirror are the third and fourth. A refusal here costs no GPU minute.
  local rel="${1:?bank view rel}" cell_dir="${2:?bank cell dir}" pin="${3:?object sha256}" record="${4:?admission record}"
  local lane="${5:?lane}" target="${6:?target}" view="${7:?view}" cell="${8:?cell}" out
  # the FOURTH witness is the repo's pin file (its (target, staged id) entry must name the pin; absent = refused, a served object is
  # pinned before its view builds). ST_ITDOC_PINS is a TEST SEAM for a synthetic bank + pins pair — production never sets it (a plain
  # shell variable of the submitter's environment; never --export=), so the door reads configs/sdfbase_full22_pins.json
  local pins="${ST_ITDOC_PINS:-configs/sdfbase_full22_pins.json}"
  out=$( cd "$REPO_DIR" && "$CPY" scripts/sdfbase/full22/itdoc_bind.py bind-check --admission "$record" --lane "$lane" --key "$target" \
          --view "$view" --cell "$cell" --view-dir "${SHARED:?SHARED}/$rel" --cell-dir "$cell_dir" --pin "$pin" --pins "$pins" 2>&1 ) \
    || f22_fatal "it-doc bank binding REFUSED for (lane $lane, target $target, view $view, cell $cell) — the bank entry, the bytes, the sidecar or the pin file disagree with the record's pin; fix the staging or the record, never the check (E3) — $out"
  f22_log "$out"
  st_external_bank_view_check "$rel" "$cell_dir" "$pin"
}

st_check_endpoint(){   # $1 = endpoint.json, $2 = the upstream timeout the standup was given, $3 = the budget artifact's
                       # chat_template_hash, $4 = the routing table the standup just WROTE, $5 = the harness cell id,
                       # $6 = the served positions the row EXPECTS for that cell ("" = no positions expectation: the map must
                       # merely exist; "all" / "response" = the stamp must say exactly that)
  local ep="${1:?endpoint.json}" want_timeout="${2:?upstream timeout}" want_hash="${3:?budget template hash}" table="${4:?routing table}"
  local hcell="${5:?harness cell}" expect="${6:-}" got
  got=$("$CPY" -c "import json,sys;print(json.load(open(sys.argv[1])).get('proxy_upstream_timeout_s'))" "$ep")
  [ "$got" = "$want_timeout.0" ] || [ "$got" = "$want_timeout" ] || f22_fatal "endpoint.json proxy_upstream_timeout_s=$got, not $want_timeout — the proxy would 504 a cap-bound draw before the driver's budget (D73 (d))"
  got=$("$CPY" -c "import json,sys;print(json.load(open(sys.argv[1])).get('chat_template_hash'))" "$ep")
  [ "$got" = "$want_hash" ] || f22_fatal "endpoint.json chat_template_hash=$got != the budget artifact's $want_hash — the budget was measured on another template; refusing"
  # THE STALE-TABLE HAZARD: the stamp must be in the table this standup just WROTE, before any request. The stamp is
  # read from the ENGINE-NEUTRAL `served_positions` map FIRST (both standups write it since the native record-gap fix; the
  # native standup writes `lens_served_positions: {}`), then the lens map for an older table; the two present and disagreeing
  # for this cell refuse by name, and a table naming the cell in neither refuses as before — the driver's own read order.
  "$CPY" - "$table" "$expect" "$hcell" <<'PY' || f22_fatal "REFUSED: served-positions stamp absent or wrong in the fresh routing table — a stale table or a tree without Evals' refinement; no request is sent"
import json, sys
t = json.load(open(sys.argv[1])); expect, cell = sys.argv[2], sys.argv[3]
neutral, lens = t.get("served_positions"), t.get("lens_served_positions")
if not isinstance(neutral, dict) and not isinstance(lens, dict):
    print(f"routing_table.json carries neither a served_positions nor a lens_served_positions map (keys: {sorted(t)})"); sys.exit(1)
n_hit = neutral.get(cell) if isinstance(neutral, dict) else None
l_hit = lens.get(cell) if isinstance(lens, dict) else None
if n_hit is not None and l_hit is not None and n_hit != l_hit:
    print(f"served_positions[{cell!r}] = {n_hit!r} disagrees with lens_served_positions[{cell!r}] = {l_hit!r} — a torn table"); sys.exit(1)
got, source = (n_hit, "served_positions") if n_hit is not None else (l_hit, "lens_served_positions")
if expect and got != [expect]:
    print(f"{source}[{cell!r}] = {got!r}, expected [{expect!r}]"); sys.exit(1)
print(f"served positions OK ({source}): {cell!r} -> {got!r}; neutral map {neutral!r}; lens map {lens!r}")
PY
  # the LENS proxy's stamp hand-off flag: the lens standup writes `lens_served_positions_passthrough: passed` when its proxy
  # constructor took the stamp (standup_thinking); a NATIVE standup (endpoint.json engine == lora_native) has no lens hook and
  # writes no such flag — the neutral-map check above is its stamp door, so the flag is NOT APPLICABLE there, logged by name
  local engine; engine=$("$CPY" -c "import json,sys;print(json.load(open(sys.argv[1])).get('engine'))" "$ep")
  if [ "$engine" = lora_native ]; then
    f22_log "endpoint.json engine=lora_native: the lens stamp hand-off flag (lens_served_positions_passthrough) is not applicable on the native engine; the served-positions stamp above is the door"
  else
    got=$("$CPY" -c "import json,sys;print(json.load(open(sys.argv[1])).get('lens_served_positions_passthrough'))" "$ep")
    [ "$got" = passed ] || f22_fatal "endpoint.json lens_served_positions_passthrough=$got — the proxy was not handed the stamp; every undeclared thinking-ON lens request would refuse thinking_on_undeclared"
  fi
}

# --- the USAGE LEDGER in front of the proxy (tinycap_served.py's shape) -----------------------------------------
st_ledger_up(){   # $1 = the serve subdir name (math500 | gsm8k: $SERVE_DIR/<name>/ledger.log), $2 = the readiness wait (s),
                  # $3 = the ledger filename (in $OUT_ROOT), $4 = the endpoint filename (beside routing_table.json),
                  # $5.. = extra ledger argv, VERBATIM (the attempt id)  -> LEDGER_EP LEDGER_PID LEDGER_URL
                  # returns 1 (after logging LEDGER_FAILED) when the ledger never writes its endpoint — the caller files the
                  # result and never starts the driver
  local sub="${1:?serve subdir}" wait_s="${2:?wait s}" ledger_name="${3:?ledger filename}" ep_name="${4:?endpoint filename}"
  shift 4
  local port i
  port=$("$CPY" -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1])')
  # the driver binds the ledger to the standup by the record BESIDE routing_table.json (its --endpoint-json)
  LEDGER_EP="$(dirname "$TABLE")/$ep_name"; rm -f "$LEDGER_EP"
  ( cd "$REPO_DIR" && CUDA_VISIBLE_DEVICES="" PYTHONPATH="$REPO_DIR/src:$REPO_DIR/scripts" "$CPY" -m sisteer.capeval.serving.usage_ledger \
      --upstream "$PROXY_URL" --port "$port" --host 127.0.0.1 --ledger-path "$OUT_ROOT/$ledger_name" --endpoint-json "$LEDGER_EP" "$@" ) \
      > "$SERVE_DIR/$sub/ledger.log" 2>&1 &
  LEDGER_PID=$!
  for i in $(seq 1 "$wait_s"); do
    [ -f "$LEDGER_EP" ] && break
    kill -0 "$LEDGER_PID" 2>/dev/null || break
    sleep 1
  done
  if [ ! -f "$LEDGER_EP" ]; then
    f22_log "LEDGER_FAILED: no $LEDGER_EP after ${wait_s}s (ledger.log tail follows)"; tail -20 "$SERVE_DIR/$sub/ledger.log" || true
    return 1
  fi
  LEDGER_URL=$("$CPY" -c "import json,sys;print(json.load(open(sys.argv[1]))['ledger_url'])" "$LEDGER_EP")
  [ -n "$LEDGER_URL" ] || { f22_log "LEDGER_FAILED: $LEDGER_EP carries no ledger_url"; return 1; }
  f22_log "ledger up: $LEDGER_URL -> $PROXY_URL rows -> $OUT_ROOT/$ledger_name"
}

st_ledger_down(){   # TERM, wait <= 30 s, then KILL (the launcher's teardown; idempotent)
  [ -n "${LEDGER_PID:-}" ] || return 0
  kill -TERM "$LEDGER_PID" 2>/dev/null || true
  local i; for i in $(seq 1 30); do kill -0 "$LEDGER_PID" 2>/dev/null || break; sleep 1; done
  kill -9 "$LEDGER_PID" 2>/dev/null || true; LEDGER_PID=""
}

st_proxy_token(){   # $1 = endpoint.json -> the bearer the driver / a probe presents: its proxy_token (this loopback proxy is
                    # unauthenticated -> null), else the client's default `petri`
  "$CPY" -c "import json,sys;print(json.load(open(sys.argv[1])).get('proxy_token') or 'petri')" "${1:?endpoint.json}"
}

# --- the driver argv: the common head and tail; each launcher puts its TASK flags between them ---------------------
st_driver_argv_head(){   # $1 = --controls, $2 = --thinking literal, $3 = --sampling mode, $4 = --samples  -> DRV=(...)
                         # (needs CPY PROXY_URL TABLE LEDGER_URL CELL CFG TT_TWO_SI_DEFENSES_DIR OUT_ROOT)
  DRV=("$CPY" scripts/served_capability_run.py generate
       --proxy-url "$PROXY_URL" --routing-table "$TABLE" --ledger-url "$LEDGER_URL"
       --cell "$CELL" --config "$CFG" --defenses-dir "$TT_TWO_SI_DEFENSES_DIR" --out-root "$OUT_ROOT"
       --controls "${1:?controls}" --thinking "${2:?thinking}" --sampling "${3:?sampling mode}" --samples "${4:?samples}")
}

st_driver_argv_tail(){   # $1 = --tinycap-identity-version, $2 = --oos-prompt-budget path, $3 = --concurrency, $4 = --vllm-log
  DRV+=(--tinycap-identity-version "${1:?identity version}" --oos-prompt-budget "${2:?budget path}"
        --concurrency "${3:?concurrency}" --vllm-log "${4:?vllm log}")
}

# --- the dry run's verbatim lines (each launcher prints its own header lines around these) --------------------------
st_print_argv(){   # $1 = the line label, $2.. = the argv  -> `LABEL: %q %q ...`
  local label="${1:?label}" q; shift
  printf '%s:' "$label"; for q in "$@"; do printf ' %q' "$q"; done; printf '\n'
}

st_print_ledger_line(){   # $1 = the ledger path, $2 = the endpoint record path, $3.. = extra ledger argv
  local ledger="${1:?ledger path}" ep="${2:?endpoint path}"; shift 2
  printf 'LEDGER: %q -m sisteer.capeval.serving.usage_ledger --upstream %q --port <free> --host 127.0.0.1 --ledger-path %q --endpoint-json %q' \
    "$CPY" "$PROXY_URL" "$ledger" "$ep"
  local q; for q in "$@"; do printf ' %q' "$q"; done; printf '\n'
}

st_print_driver_env(){   # $1 = the Inspect base-url env name, $2 = the ledger url, $3 = the Inspect api-key env name
  printf 'DRIVER_ENV: CUDA_VISIBLE_DEVICES= %s=%q %s=<proxy_token or petri>\n' "${1:?base url env}" "${2:?ledger url}" "${3:?api key env}"
}

# --- the ATTEMPT seam (the producer contract is attempt-scoped) ----------------------------------------------------
# One ladder_state.json per store (the out root): the WRAPPER's attempt ledger. st_attempt_begin mints ATTEMPT_ID = a<k>
# (k = one past the attempts already recorded — a resubmitted row is a NEW attempt, never a re-labelled one) and appends
# {attempt_id, started, job facts, ended null, rc null, result null}; st_attempt_end fills ended / rc / result on the
# attempt this process minted, and on a clean exit records accepted_generation_attempt — from the driver's manifest when it
# names one (a FINALISATION attempt accepts the earlier generation attempt whose .eval it selected; the driver's
# resume_attempts carry the kind), else this attempt (accepted_source says which). The ledger process and the driver both
# receive --attempt-id "$ATTEMPT_ID", so every ledger row and request-evidence record carries it (lane H's flags).
st_attempt_begin(){   # $1 = ladder_state.json, $2 = a JSON object of job facts (job_id, array_task_id, node, cluster, sha, rung, ...)
                      # -> ATTEMPT_ID
  local state="${1:?ladder_state.json}" facts="${2:-{\}}"
  ATTEMPT_ID=$("$CPY" - "$state" "$facts" <<'PY'
import json, os, sys, time
path, facts = sys.argv[1], json.loads(sys.argv[2])
doc = json.load(open(path)) if os.path.isfile(path) else {"schema": "g8k_ladder_state/1", "attempts": [],
                                                           "accepted_generation_attempt": None}
attempts = doc.setdefault("attempts", [])
aid = f"a{len(attempts) + 1}"
attempts.append({"attempt_id": aid, "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                 "ended": None, "rc": None, "result": None, **facts})
tmp = path + ".tmp"
with open(tmp, "w") as f:
    json.dump(doc, f, indent=1, sort_keys=True)
os.replace(tmp, path)
print(aid)
PY
  ) || f22_fatal "could not record the attempt in $state"
  [ -n "$ATTEMPT_ID" ] || f22_fatal "no attempt id minted for $state"
}

st_attempt_end(){   # $1 = ladder_state.json, $2 = rc, $3 = the RESULT word, $4 = the store's out root (its ONE cell manifest,
                    # cells/*/manifest.json, names the accepted generation attempt when the driver finalised from an earlier one)
  local state="${1:?ladder_state.json}" rc="${2:?rc}" result="${3:?result}" out_root="${4:-}"
  [ -n "${ATTEMPT_ID:-}" ] || return 0
  [ -f "$state" ] || return 0
  "$CPY" - "$state" "$ATTEMPT_ID" "$rc" "$result" "$out_root" <<'PY' || true
import glob, json, os, sys, time
path, aid, rc, result, out_root = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5]
doc = json.load(open(path))
now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
for a in doc.get("attempts", []):
    if a.get("attempt_id") == aid:
        a.update({"ended": now, "rc": rc, "result": result})
if rc == 0:
    accepted, source = aid, "launcher (the driver's manifest named no accepted_generation_attempt)"
    manifests = sorted(glob.glob(os.path.join(out_root, "cells", "*", "manifest.json"))) if out_root else []
    if manifests:
        try:
            m = json.load(open(manifests[0]))
            if m.get("accepted_generation_attempt"):
                accepted, source = m["accepted_generation_attempt"], f"the driver's manifest ({os.path.relpath(manifests[0], out_root)})"
        except Exception as e:  # noqa: BLE001 — the manifest is the driver's; an unreadable one is recorded, never fatal here
            source = f"launcher (manifest unreadable: {type(e).__name__})"
    doc["accepted_generation_attempt"] = accepted
    doc["accepted_source"] = source
tmp = path + ".tmp"
with open(tmp, "w") as f:
    json.dump(doc, f, indent=1, sort_keys=True)
os.replace(tmp, path)
PY
}
