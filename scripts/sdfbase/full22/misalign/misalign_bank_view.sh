#!/bin/bash
# em-misalign (WP-C): build (or --check) the MISALIGN bank view of one target, on either cluster.
#
#   $SHARED/exp/objects/misalign/<target>/cells_view/
#     <cell_id>   -> ../../../sdfbase/full22/<roster target>/cells_view/<view>/<cell_id>   one entry per object of
#                    the key's full22 view (<target> -> (roster target, view): every target is its own `main`
#                    view except the sdfbase-llama8b pair — `llama8b` = llama8b/main, the base-lane objects
#                    transferred; `llama8b_it` = llama8b/it, the instruct-direct objects. The wave's view is only
#                    VERIFIED here through make_bank_view.py with the roster + pins — its own sha checks,
#                    identity-sidecar verdicts and short-view guard; the wave's lane builds it)
#     alphasteer  -> ../../../alphasteer_f22/<roster target>/cells/selected__alphasteer   ONLY after the staged
#                    object.pt hashes to configs/alphasteer_f22_pins.json's object_sha256 and the dir carries
#                    exactly the identity sidecar the pin names; a mismatch, an unpinned object or a `served`
#                    pin with no object REFUSES loudly (exit 2). A `no_admissible_operating_point` pin (glm)
#                    or a target the alphasteer roster never lists (llama8b: `not_in_campaign`) means NO
#                    entry, and a stale one is removed.
#
#   <family>_bl  the PUBLISHED-BASELINES view: NO wave entry — exactly two
#                entries, `arditi` -> ../../../arditi_strength/<family>/strength/cells_view/<view>/arditi and
#                `alphasteer` -> ../../../alphasteer_f22_rebake/<family>/ladder_canon/<rung>/cells_view/<view>/alphasteer,
#                each linked ONLY after its object.pt hashes to configs/misalign_baseline_pins.json's object_sha256
#                and the dir carries exactly the identity sidecar the pin names. The `alphasteer` here is the RE-BAKE
#                rung, a different artifact from the campaign's selected__alphasteer the family panel serves (same
#                cell id, separate results roots). A VIEW_MANIFEST_baselines.json records what was verified.
#
#   <family>_org the EM MODEL-ORGANISMS view. WITH a pins entry for the family in configs/misalign_org_pins.json
#                (qwen35_4b; batch-64 fits preferred) it is a PINNED view like `_bl`: NO wave entry — exactly the
#                pinned cells, each `<cell>` -> ../../../<source> into the reseed lane's store (READ-ONLY here: nothing copied,
#                nothing written there), linked ONLY after its object.pt hashes to the pin's object_sha256 and the dir carries
#                exactly the identity sidecar the pin names; the config's object cells must equal the pinned cells. Its
#                VIEW_MANIFEST.json records PIN-FORM entries (form `pin`, pins configs/misalign_org_pins.json, pin
#                `<family>.<cell>`) — the shape mt_serving_up.view_manifest_digests reads. No alphasteer, ever (OUT everywhere,
#                2026-09-17). WITHOUT a pins entry (llama8b_org) the key takes the wave path below, unchanged.
#
# The view is a SEPARATE directory from the wave's `main` view on purpose: make_bank_view.py's completeness
# guard refuses an entry the roster does not name, so `alphasteer` can never live inside the wave's view.
# The misalign config (configs/misalign_full22_<target>.yaml) is cross-checked: it must list `alphasteer`
# iff the pin says served — a config/pins disagreement is an error here, before a GPU minute.
#
# $SHARED = the cluster's shared root, the job's SISTEER_SHARED_ROOT. Idempotent: an
# existing symlink is re-pointed, a non-symlink at an entry path is never touched (refused). Concurrent
# builders serialize on a per-target flock (the wave's lesson, common.sh f22_bank_view).
#
# Usage:
#   misalign_bank_view.sh <target> [--python <py>] [--check]
#     --python  a python with yaml (+ torch for the wave's derived-cell verdicts); default $MISALIGN_PY, else
#               $SHARED/envs/vllm-attacker/bin/python, else python3. The sbatch passes the driver venv.
#     --check   verify only, write nothing; exit 2 on any problem.
#   env MISALIGN_SHARED=<root>    test hook: use <root> as $SHARED instead of SISTEER_SHARED_ROOT (tests build a fake store)
#       MISALIGN_ORG_PINS=<file>  test hook: the organisms pins file (default configs/misalign_org_pins.json)
set -uo pipefail

TARGET="${1:?target (gemma4e4b|qwen35_4b|nemotron3nano4b|glm46vflash|llama8b|llama8b_it|llama8b_org|qwen35_4b_org|gemma4e4b_it|qwen35_4b_it|qwen35_4b_s1|qwen35_4b_s2|qwen35_4b_s3|qwen35_4b_s1p|qwen35_4b_s2p|qwen35_4b_s3p|<family>_bl|<models.yaml key>_<phase view>)}"; shift
PY="${MISALIGN_PY:-}"; CHECK=0
while [ $# -gt 0 ]; do
  case "$1" in
    --python) PY="$2"; shift;;
    --check) CHECK=1;;
    *) echo "bad arg $1" >&2; exit 2;;
  esac; shift
done
log(){ echo "[$(date -Is)] [misalign-view] $*"; }
fatal(){ log "FATAL: $*"; exit 2; }

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; REPO="$(cd "$HERE/../../../.." && pwd)"
if [ -n "${MISALIGN_SHARED:-}" ]; then SHARED="$MISALIGN_SHARED"; [ -d "$SHARED" ] || fatal "MISALIGN_SHARED=$SHARED is not a directory"
elif [ -n "${SISTEER_SHARED_ROOT:-}" ]; then SHARED="$SISTEER_SHARED_ROOT"; [ -d "$SHARED" ] || fatal "SISTEER_SHARED_ROOT=$SHARED is not a directory"
else fatal "no shared root: set SISTEER_SHARED_ROOT, the shared root of the cluster (the front door exports it from the site config), or MISALIGN_SHARED"; fi
if [ -z "$PY" ]; then
  if [ -x "$SHARED/envs/vllm-attacker/bin/python" ]; then PY="$SHARED/envs/vllm-attacker/bin/python"; else PY=python3; fi
fi
command -v "$PY" >/dev/null 2>&1 || [ -x "$PY" ] || fatal "python $PY not found"
"$PY" -c 'import yaml' 2>/dev/null || fatal "$PY lacks yaml"

# misalign target -> (roster target, wave view): the llama8b keys serve TWO object-bearing views of ONE
# instruct model (`main` = the base-lane objects transferred, `it` = the instruct-direct objects); every other
# target is its own roster target's `main` view (make_misalign_manifest.WAVE_OF is the same table)
case "$TARGET" in
  gemma4e4b|qwen35_4b|nemotron3nano4b|glm46vflash) RTARGET="$TARGET"; WVIEW=main;;
  llama8b)    RTARGET=llama8b; WVIEW=main;;
  llama8b_it) RTARGET=llama8b; WVIEW=it;;
  llama8b_org) RTARGET=llama8b; WVIEW=main; ORG=1;;      # the organisms transfer target (stock chat template)
  qwen35_4b_org) RTARGET=qwen35_4b; WVIEW=main; ORG=1;;  # The Qwen EM organisms (local LoRA adapters) x the main-view objects
  gemma4e4b_it) RTARGET=gemma4e4b; WVIEW=it;;     # instruct-direct objects
  qwen35_4b_it) RTARGET=qwen35_4b; WVIEW=it;;
  qwen35_4b_psr) RTARGET=qwen35_4b; WVIEW=main;;
  qwen35_4b_allpos) RTARGET=qwen35_4b; WVIEW=it;;   # the Arditi all-positions panel — its view is built by hand for Modal (pin-form entry); no cluster build yet   # the three ReLU / PSR psr_ce_all cells, NATIVE path
  # qwen-base-seed-replicates: the seed-replicate views s1|s2|s3 of qwen35_4b under the LANE POLICY
  # (make_misalign_manifest.LANE_POLICY): the view's ADMITTED objects (five: the distilled PSR difference is on no
  # view), PSR INCLUDED, NO alphasteer entry — by policy, not by pin
  qwen35_4b_s1|qwen35_4b_s2|qwen35_4b_s3) RTARGET=qwen35_4b; WVIEW="${TARGET#qwen35_4b_}"; LANE=reseed;;
  qwen35_4b_s1p|qwen35_4b_s2p|qwen35_4b_s3p) RTARGET=qwen35_4b; WVIEW="${TARGET#qwen35_4b_}"; LANE=paired;;   # the paired views
  gemma4e4b_bl|qwen35_4b_bl|nemotron3nano4b_bl|llama8b_bl) RTARGET="${TARGET%_bl}"; WVIEW=main; BASELINES=1;;   # the published baselines
  # a FRONT-DOOR phase view `<reproduce/models.yaml key>_<u0|s<N>|d<a>p<b>[_relu]|...>`
  # (misalign_proxy.sbatch reaches this script with the same target). The generator's DERIVED table is the one source
  # (make_misalign_manifest.front_door_targets over the models table); this script keeps no second copy and asks the module (`--resolve`:
  # `<roster target> <view> <kind>`), exactly as the proxy does. LANE=front_door: the view's PSR cells KEPT (spec FD 5 names the ten arms)
  # and alphasteer excluded by policy (lane_excluded), like reseed / paired. A key the module does not know fails here, by name.
  *) read -r RTARGET WVIEW KIND < <("$PY" "$HERE/make_misalign_manifest.py" --resolve "$TARGET") \
       || fatal "unknown target $TARGET (not a campaign key, a lane view or a front-door phase view of reproduce/models.yaml: make_misalign_manifest.TARGETS)"
     [ "$KIND" = front_door ] || fatal "target $TARGET resolves to kind $KIND — every non-front-door key is named in this case explicitly; refusing"
     LANE=front_door
     log "front-door target $TARGET -> roster target $RTARGET, phase view $WVIEW (make_misalign_manifest.py --resolve; lane policy front_door)";;
esac
BASELINES="${BASELINES:-0}"; ORG="${ORG:-0}"; LANE="${LANE:-}"
CFG="$REPO/configs/misalign_full22_${TARGET}.yaml"
[ -f "$CFG" ] || fatal "missing $CFG (run make_misalign_manifest.py)"
VIEW="$SHARED/exp/objects/misalign/$TARGET/cells_view"
ORG_PINS="${MISALIGN_ORG_PINS:-$REPO/configs/misalign_org_pins.json}"
if [ "$ORG_PINS" = "$REPO/configs/misalign_org_pins.json" ]; then ORG_PINS_REL="configs/misalign_org_pins.json"; else ORG_PINS_REL="$ORG_PINS"; fi

# --- a PINNED view (the `_bl` baselines panel; the pinned `_org` organisms panel): exactly the pinned entries, each linked ONLY
#     after its object.pt re-hashes to the pin and the dir carries exactly the identity sidecar the pin names; a stale symlink
#     is removed in build mode (a PROBLEM under --check); a non-symlink is never touched. Reads PIN_ROWS
#     (cell<TAB>source<TAB>sha256<TAB>sidecar) and WANT (the cells); leaves VERIFIED (cell=sha) for the manifest writer. ------
pinned_view_link(){
  problems=0
  if [ "$CHECK" = 0 ]; then mkdir -p "$VIEW"; fi
  [ -d "$VIEW" ] || fatal "no view at $VIEW (--check: not building)"
  lock="$SHARED/exp/objects/misalign/.$TARGET.view.lock"; mkdir -p "$(dirname "$lock")"
  exec 9>"$lock"; flock -w 900 9 || fatal "could not take $lock in 900s"
  VERIFIED=()
  for row in "${PIN_ROWS[@]}"; do
    IFS=$'\t' read -r cell src sha sidecar <<< "$row"
    SRC_DIR="$SHARED/exp/objects/$src"; REL="../../../$src"
    [ -f "$SRC_DIR/object.pt" ] || fatal "$cell: no object at $SRC_DIR/object.pt (the owning lane stages it; nothing is copied here)"
    have=$(sha256sum "$SRC_DIR/object.pt" | cut -c1-64)
    [ "$have" = "$sha" ] || fatal "$cell object.pt sha256 ${have:0:12} != pinned ${sha:0:12} ($SRC_DIR) — refusing"
    n_id=0; for f in train_budget_meta.json fit_provenance.json derived_meta.json; do [ -f "$SRC_DIR/$f" ] && n_id=$((n_id+1)); done
    [ "$n_id" -eq 1 ] || fatal "$SRC_DIR declares $n_id identity sidecars (need exactly one)"
    [ -f "$SRC_DIR/$sidecar" ] || fatal "$SRC_DIR lacks the pin's identity sidecar $sidecar"
    log "$cell object verified: sha ${have:0:12} == pin, sidecar $sidecar ($src)"
    dst="$VIEW/$cell"
    if [ -L "$dst" ]; then
      if [ "$(readlink "$dst")" != "$REL" ]; then
        if [ "$CHECK" = 1 ]; then log "PROBLEM $cell -> $(readlink "$dst") (want $REL)"; problems=$((problems+1)); continue; fi
        rm -f "$dst"; ln -s "$REL" "$dst"; log "re-pointed $cell -> $REL"
      fi
    elif [ -e "$dst" ]; then
      log "PROBLEM $dst exists and is not a symlink — refusing to touch someone else's data"; problems=$((problems+1)); continue
    else
      if [ "$CHECK" = 1 ]; then log "PROBLEM missing entry $cell"; problems=$((problems+1)); continue; fi
      ln -s "$REL" "$dst"; log "linked $cell -> $REL"
    fi
    have2=$(sha256sum "$dst/object.pt" 2>/dev/null | cut -c1-64)
    [ "$have2" = "$sha" ] || { log "PROBLEM $cell view entry hashes ${have2:0:12} != pin"; problems=$((problems+1)); }
    VERIFIED+=("$cell=$have2")
  done
  # stale entries: a symlink the pins do not name is removed in build mode (a PROBLEM under --check); a non-symlink is never touched
  mapfile -t HAVE < <(find "$VIEW" -mindepth 1 -maxdepth 1 ! -name 'VIEW_MANIFEST*' -printf '%f\n' 2>/dev/null | sort)
  for h in "${HAVE[@]}"; do
    case " ${WANT[*]} " in *" $h "*) continue;; esac
    if [ -L "$VIEW/$h" ]; then
      [ "$CHECK" = 1 ] && { log "PROBLEM stale entry $h"; problems=$((problems+1)); } || { rm -f "$VIEW/$h"; log "removed stale entry $h"; }
    else
      log "PROBLEM $VIEW/$h is not a pinned cell and is not a symlink — refusing to touch it"; problems=$((problems+1))
    fi
  done
  if [ "$problems" -gt 0 ]; then fatal "$problems problem(s) in $VIEW"; fi
}

# --- the PUBLISHED-BASELINES view: two pinned entries, nothing from the wave ------------------------
if [ "$BASELINES" = 1 ]; then
  PINS="$REPO/configs/misalign_baseline_pins.json"; [ -f "$PINS" ] || fatal "missing $PINS"
  # cell<TAB>source<TAB>sha256<TAB>sidecar, one line per pinned cell, in the pins' `_cells` order
  mapfile -t PIN_ROWS < <("$PY" - "$PINS" "$RTARGET" <<'PY'
import json, re, sys
pins = json.load(open(sys.argv[1])); fam = sys.argv[2]
if pins.get("_schema") != "misalign_baseline_pins/1": sys.exit("bad schema")
rec = pins.get(fam) or sys.exit(f"no {fam} entry")
for c in pins["_cells"]:
    o = rec[c]
    if not re.fullmatch(r"[0-9a-f]{64}", o["object_sha256"]): sys.exit(f"{fam}/{c}: bad sha")
    if not o["source"].startswith(f"{o['bank']}/{fam}/"): sys.exit(f"{fam}/{c}: source outside its bank")
    print("\t".join([c, o["source"], o["object_sha256"], o["identity_sidecar"]]))
PY
) || fatal "cannot read $PINS for $RTARGET"
  [ "${#PIN_ROWS[@]}" -eq 2 ] || fatal "$PINS names ${#PIN_ROWS[@]} cells for $RTARGET (want 2)"
  # the config's object cells must be exactly the pinned cells (and `none` first)
  mapfile -t CFG_OBJ < <("$PY" -c 'import sys,yaml
cells = yaml.safe_load(open(sys.argv[1]))["explicit_cells"]
assert cells[0] == "none", cells
for c in cells[1:]:
    assert not c.startswith("prompted__"), c
    print(c)' "$CFG") || fatal "$CFG: a baselines config is [none, <pinned cells>] — regenerate (make_misalign_manifest.py)"
  WANT=(); for row in "${PIN_ROWS[@]}"; do WANT+=("${row%%	*}"); done
  if [ "$(printf '%s\n' "${CFG_OBJ[@]}" | sort)" != "$(printf '%s\n' "${WANT[@]}" | sort)" ]; then
    fatal "$CFG object cells [${CFG_OBJ[*]}] != pinned baseline cells [${WANT[*]}] — regenerate the configs"
  fi
  pinned_view_link
  if [ "$CHECK" = 0 ]; then
    "$PY" - "$VIEW/VIEW_MANIFEST_baselines.json" "$TARGET" "$RTARGET" "$PINS" "${VERIFIED[@]}" <<'PY'
import hashlib, json, sys, time
out, target, fam, pins = sys.argv[1:5]
json.dump({"schema": "misalign_baselines_view/1", "target": target, "family": fam, "written_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "pins_sha256": hashlib.sha256(open(pins, "rb").read()).hexdigest(), "verified": dict(v.split("=", 1) for v in sys.argv[5:]),
           "pins": {k: v for k, v in json.load(open(pins))[fam].items()}}, open(out, "w"), indent=2)
PY
  fi
  log "ok: $VIEW (${#WANT[@]} pinned baseline entries: ${WANT[*]}; no wave entry by design)"
  exit 0
fi

# --- the EM MODEL-ORGANISMS view with PINNED objects (batch-64 fits preferred) -------------------------------------------
#     Exactly the pinned cells (configs/misalign_org_pins.json <family>), each linked READ-ONLY into the reseed lane's store
#     after its object.pt re-hashes to the pin and the dir carries exactly the identity sidecar the pin names — NO wave entry
#     (the wave's main view holds the older batch-8 fits under the same cell ids), no alphasteer. An _org key whose family
#     has no pins entry (llama8b_org) falls through to the wave path below, unchanged.
if [ "$ORG" = 1 ] && [ -f "$ORG_PINS" ] && "$PY" -c 'import json,sys; sys.exit(0 if sys.argv[2] in json.load(open(sys.argv[1])) else 1)' "$ORG_PINS" "$RTARGET"; then
  # cell<TAB>source<TAB>sha256<TAB>sidecar, one line per pinned cell, in the pins' `_cells` order
  mapfile -t PIN_ROWS < <("$PY" - "$ORG_PINS" "$RTARGET" <<'PY'
import json, re, sys
pins = json.load(open(sys.argv[1])); fam = sys.argv[2]
if pins.get("_schema") != "misalign_org_pins/1": sys.exit("bad schema")
rec = pins.get(fam) or sys.exit(f"no {fam} entry")
for c in pins["_cells"]:
    o = rec[c]
    if not re.fullmatch(r"[0-9a-f]{64}", o["object_sha256"]): sys.exit(f"{fam}/{c}: bad sha")
    if not o["source"].startswith(f"{o['bank']}/"): sys.exit(f"{fam}/{c}: source outside its bank")
    if o["identity_sidecar"] not in ("train_budget_meta.json", "fit_provenance.json", "derived_meta.json"): sys.exit(f"{fam}/{c}: bad identity_sidecar")
    print("\t".join([c, o["source"], o["object_sha256"], o["identity_sidecar"]]))
PY
) || fatal "cannot read $ORG_PINS for $RTARGET"
  [ "${#PIN_ROWS[@]}" -ge 1 ] && [ -n "${PIN_ROWS[0]}" ] || fatal "$ORG_PINS names no cells for $RTARGET"
  # the config's OBJECT cells must be exactly the pinned cells (`none` first; the two prompted arms have no object and no entry)
  mapfile -t CFG_OBJ < <("$PY" -c 'import sys,yaml
cells = yaml.safe_load(open(sys.argv[1]))["explicit_cells"]
assert cells[0] == "none", cells
assert "alphasteer" not in cells, cells
for c in cells[1:]:
    if not c.startswith("prompted__"): print(c)' "$CFG") || fatal "$CFG: a pinned organisms config is [none, <pinned cells>, prompted…] with no alphasteer — regenerate (make_misalign_manifest.py)"
  WANT=(); for row in "${PIN_ROWS[@]}"; do WANT+=("${row%%	*}"); done
  if [ "$(printf '%s\n' "${CFG_OBJ[@]}" | sort)" != "$(printf '%s\n' "${WANT[@]}" | sort)" ]; then
    fatal "$CFG object cells [${CFG_OBJ[*]}] != pinned organisms cells [${WANT[*]}] — regenerate the configs"
  fi
  log "organisms panel $TARGET: PINNED objects — $ORG_PINS_REL [$RTARGET], ${#WANT[@]} cells (the replication lane's effective-batch-64 seed-1 fits); the wave's $RTARGET/$WVIEW view is NOT linked (its objects are the older batch-8 fits); no alphasteer (OUT everywhere, 2026-09-17)"
  pinned_view_link
  if [ "$CHECK" = 0 ]; then
    "$PY" - "$VIEW/VIEW_MANIFEST.json" "$TARGET" "$RTARGET" "$ORG_PINS" "$ORG_PINS_REL" "${VERIFIED[@]}" <<'PY'
import hashlib, json, os, sys, time
out, target, fam, pins_path, pins_rel = sys.argv[1:6]
pins = json.load(open(pins_path)); rec = pins[fam]
verified = dict(v.split("=", 1) for v in sys.argv[6:])
view = os.path.dirname(out)
entries = []
for c in pins["_cells"]:
    d, o = os.path.join(view, c), rec[c]
    # PIN-FORM entries (make_bank_view.plan_pin_entry's shape): mt_serving_up.view_manifest_digests reads the pin from the
    # authority named here (`pins` resolved against the checkout root, `pin` = <family>.<cell>), never from the wave's pins
    entries.append({"cell_id": c, "form": "pin", "pins": pins_rel, "pin": f"{fam}.{c}",
                    "staged_id": o.get("staged_id") or os.path.basename(o["source"]), "object_sha256": verified[c],
                    "kind": o["kind"], "identity_sidecar": o["identity_sidecar"], "link": os.readlink(d),
                    "resolved": os.path.realpath(d), "admission": o.get("admission"), "admission_note": o.get("admission_note"),
                    "source": f"{o['bank']} (pin-verified; {rec['fit_lane']} {rec['fit_model_key']}, seed {rec['seed']}, effective_batch {rec['effective_batch']})"})
json.dump({"wave": "misalign", "schema": "misalign_org_view/1", "target": target, "family": fam, "view": view, "wave_view": None,
           "alphasteer_status": "out_organisms", "alphasteer_pin_sha256": None, "pins": pins_rel,
           "pins_sha256": hashlib.sha256(open(pins_path, "rb").read()).hexdigest(), "entries": entries,
           "written_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, open(out, "w"), indent=2)
print("\n".join(f"view {target}: {e['cell_id']:<50} sha {e['object_sha256'][:12]}  <- {e['link']}" for e in entries))
PY
  fi
  log "ok: $VIEW (${#WANT[@]} pinned organisms entries: ${WANT[*]}; no wave entry, no alphasteer by design)"
  exit 0
fi
F22="$REPO/scripts/sdfbase/full22"
MAIN_VIEW="$SHARED/exp/objects/sdfbase/full22/$RTARGET/cells_view/$WVIEW"
AS_DIR="$SHARED/exp/objects/alphasteer_f22/$RTARGET/cells/selected__alphasteer"
REL_MAIN="../../../sdfbase/full22/$RTARGET/cells_view/$WVIEW"
REL_AS="../../../alphasteer_f22/$RTARGET/cells/selected__alphasteer"

# --- 1. the wave's view: verify read-only. It is BUILT here only on request (MISALIGN_BUILD_WAVE_VIEW=1): the
#        wave's lane stages its objects and builds its own views, and nothing is built for it from outside here
#        — an unverifiable view is a refusal, not a build. ---------------------------------------------------
cd "$REPO" || exit 2
# the wave bank rides EXPLICITLY (the $SHARED this script resolved — MISALIGN_SHARED under test — never make_bank_view.py's own probe)
if ! "$PY" "$F22/make_bank_view.py" --check --target "$RTARGET" --view "$WVIEW" --bank "$SHARED/exp/objects/sdfbase/full22/$RTARGET" >/dev/null 2>&1; then
  if [ "$CHECK" = 1 ] || [ "${MISALIGN_BUILD_WAVE_VIEW:-0}" != 1 ]; then
    "$PY" "$F22/make_bank_view.py" --check --target "$RTARGET" --view "$WVIEW" --bank "$SHARED/exp/objects/sdfbase/full22/$RTARGET"
    fatal "the wave's $RTARGET/$WVIEW view does not verify — not building it from here (the wave's lane builds its views; MISALIGN_BUILD_WAVE_VIEW=1 overrides)"
  fi
  lock="$SHARED/exp/objects/sdfbase/full22/.$RTARGET-$WVIEW.view.lock"; mkdir -p "$(dirname "$lock")"
  log "wave view $RTARGET/$WVIEW does not verify — building it under $lock"
  ( flock -w 900 9 || { echo "FATAL: could not take the bank-view lock in 900s" >&2; exit 1; }
    "$PY" "$F22/make_bank_view.py" --target "$RTARGET" --view "$WVIEW" --bank "$SHARED/exp/objects/sdfbase/full22/$RTARGET" ) 9>"$lock" || fatal "make_bank_view.py refused $RTARGET/$WVIEW"
fi
"$PY" "$F22/make_bank_view.py" --check --target "$RTARGET" --view "$WVIEW" --bank "$SHARED/exp/objects/sdfbase/full22/$RTARGET" || fatal "wave view $RTARGET/$WVIEW refused"

# --- 2. what the misalign view must hold: the view's staged cells + alphasteer iff pinned served ------------
mapfile -t MAIN_CELLS < <("$PY" - "$F22" "$RTARGET" "$WVIEW" <<'PY'
import sys; sys.path.insert(0, sys.argv[1])
import yaml
from f22_roster import staged_cells
from pathlib import Path
roster = yaml.safe_load((Path(sys.argv[1]).parents[2] / "configs" / "sdfbase_full22_roster.yaml").read_text())
cells = list(staged_cells(roster, sys.argv[2], sys.argv[3]))
if cells:
    print("\n".join(cells))   # nothing at all for an empty set: a lone "\n" would read as ONE empty cell below
PY
) || fatal "cannot read the roster's staged cells for $RTARGET/$WVIEW"
# A PROMPTED-ONLY view (the front door's unsteered u0: `none` + the prompted SIs, no object) stages nothing and is still served — the
# misalign view then holds NO entries and the proxy serves exactly the view's cells (paper-companion-repo stage E, 2026-09-30: the
# u0 agentic proxy died here at 46 s). The config decides: a config that lists NO object cell names a prompted-only view; a config
# that lists object cells with an empty staged set is the pending-fit refusal it always was.
CFG_N_OBJ=$("$PY" -c 'import sys,yaml; print(sum(1 for c in yaml.safe_load(open(sys.argv[1]))["explicit_cells"] if c != "none" and not c.startswith("prompted__")))' "$CFG") \
  || fatal "cannot read $CFG explicit_cells"
if [ "${#MAIN_CELLS[@]}" -eq 0 ] || [ -z "${MAIN_CELLS[0]}" ]; then
  [ "$CFG_N_OBJ" -eq 0 ] || fatal "roster names no staged cells for $RTARGET/$WVIEW (still pending_fit? — the wave's lane stages them; nothing to serve yet)"
  MAIN_CELLS=(); log "prompted-only view $RTARGET/$WVIEW: no staged cells, $CFG lists no object cell — the misalign view holds no entries"
fi
# 2026-09-12: PSR is out — not run, not plotted. The wave views still list PSR cells for the campaign's
# other evals; this eval neither links nor serves them, and the generated configs carry none (make_misalign_manifest
# psr_out), so the cross-check below compares against the PSR-free set.
# ... EXCEPT under the reseed LANE POLICY (its spec names the PSR objects; the psr-out rule is the
# em-misalign lane's), whose configs carry the PSR cells and whose view keeps them (make_misalign_manifest.psr_policy).
# ... and under the `paired` LANE POLICY (2026-09-20): the composed difference_relu arm
# constv2-mix-full__psr_ce_all_minus_ctl IS a PSR kind (psr_paired_write) — dropping 'psr' cells would drop the very arm the
# view exists for (make_misalign_manifest.PAIRED_VIEWS / LANE_POLICY paired: psr include).
case "$LANE" in
  reseed|paired|front_door) log "lane policy $LANE: PSR cells KEPT for $RTARGET/$WVIEW (${MAIN_CELLS[*]})";;
  *)
    mapfile -t MAIN_CELLS < <(printf '%s\n' "${MAIN_CELLS[@]}" | grep -v 'psr' || true)
    [ "${#MAIN_CELLS[@]}" -ge 1 ] && [ -n "${MAIN_CELLS[0]}" ] \
      || fatal "roster names no non-PSR staged cells for $RTARGET/$WVIEW — nothing this eval serves";;
esac

read -r AS_STATUS AS_SHA AS_SIDECAR < <("$PY" - "$REPO/configs/alphasteer_f22_pins.json" "$RTARGET" "$REPO/configs/alphasteer_f22_roster.yaml" <<'PY'
import json, os, sys, yaml
if not os.path.isfile(sys.argv[1]) or not os.path.isfile(sys.argv[3]):
    # the AlphaSteer campaign's roster + pins live on its own lane; a tree without them serves no alphasteer cell
    print("not_on_tree - -"); sys.exit(0)
rec = (json.load(open(sys.argv[1])).get(sys.argv[2]) or {}).get("selected__alphasteer")
if rec is None:
    # a target the AlphaSteer campaign never listed has no arm to pin (llama8b): evidence, not an unpinned object
    if sys.argv[2] not in (yaml.safe_load(open(sys.argv[3])).get("targets") or {}):
        print("not_in_campaign - -"); sys.exit(0)
    print("MISSING - -"); sys.exit(0)
st = rec.get("status")
if st == "served":
    if not rec.get("object_sha256"): print("BROKEN - -"); sys.exit(0)
    print("served", rec["object_sha256"], rec.get("identity_sidecar") or "fit_provenance.json")
elif st == "no_admissible_operating_point":
    print("no_admissible_operating_point - -")
else:
    print(f"UNKNOWN:{st} - -")
PY
) || fatal "cannot read configs/alphasteer_f22_pins.json"
# the reseed LANE POLICY excludes the alphasteer arm whatever the campaign pins say (a recorded exclusion, never an unpinned
# object): no entry is linked, a stale one is removed, and the config must list none
case "$LANE" in reseed|paired|front_door) AS_STATUS=lane_excluded; AS_SHA=-; AS_SIDECAR=-;; esac   # the paired and front-door lanes exclude alphasteer too (LANE_POLICY alphasteer False; FRONT_DOOR_STATUS)
case "$AS_STATUS" in
  served|no_admissible_operating_point|not_in_campaign|not_on_tree|lane_excluded) ;;
  MISSING) fatal "alphasteer_f22_pins.json has no $RTARGET/selected__alphasteer entry although the alphasteer roster lists the target — refusing an unpinned object";;
  BROKEN) fatal "alphasteer_f22_pins.json: $RTARGET is 'served' with no object_sha256";;
  *) fatal "alphasteer_f22_pins.json: $RTARGET status $AS_STATUS is not one this script understands";;
esac
# the EM MODEL-ORGANISMS panels (`<family>_org`) never carry the alphasteer cell: AlphaSteer is OUT everywhere (2026-09-17),
# so a family pin that says `served` is NOT applied here — no entry is linked and the config is expected without the
# cell (make_misalign_manifest.py's ORG_STATUS writes it that way). Every other key keeps the pin's word.
if [ "$ORG" = 1 ] && [ "$AS_STATUS" = served ]; then
  log "organisms panel $TARGET: the family pin says alphasteer served, but AlphaSteer is OUT everywhere (2026-09-17) — no alphasteer entry, none expected in $CFG"
  AS_STATUS=out_organisms
fi
CFG_HAS_AS=$("$PY" -c 'import sys,yaml; print(int("alphasteer" in yaml.safe_load(open(sys.argv[1]))["explicit_cells"]))' "$CFG")
if [ "$AS_STATUS" = served ] && [ "$CFG_HAS_AS" != 1 ]; then fatal "$CFG lists no alphasteer cell but the pin says served — regenerate (make_misalign_manifest.py)"; fi
if [ "$AS_STATUS" != served ] && [ "$CFG_HAS_AS" = 1 ]; then fatal "$CFG lists alphasteer but the pin says $AS_STATUS — regenerate (make_misalign_manifest.py)"; fi
# the config's object cells must be exactly the main cells (+ alphasteer): a config/roster drift is an error
mapfile -t CFG_OBJ < <("$PY" -c 'import sys,yaml
for c in yaml.safe_load(open(sys.argv[1]))["explicit_cells"]:
    if c != "none" and not c.startswith("prompted__"): print(c)' "$CFG")
WANT=("${MAIN_CELLS[@]}"); [ "$AS_STATUS" = served ] && WANT+=("alphasteer")
if [ "$(printf '%s\n' "${CFG_OBJ[@]}" | sort)" != "$(printf '%s\n' "${WANT[@]}" | sort)" ]; then
  fatal "$CFG object cells [${CFG_OBJ[*]}] != roster $RTARGET/$WVIEW cells + alphasteer [${WANT[*]}] — regenerate the configs"
fi

# --- 3. the alphasteer object: sha256 against the pin, exactly one identity sidecar (the pin's) -----------
if [ "$AS_STATUS" = served ]; then
  [ -f "$AS_DIR/object.pt" ] || fatal "alphasteer pinned served but no object at $AS_DIR/object.pt (stage_alphasteer.sh first)"
  have=$(sha256sum "$AS_DIR/object.pt" | cut -c1-64)
  [ "$have" = "$AS_SHA" ] || fatal "alphasteer object.pt sha256 ${have:0:12} != pinned ${AS_SHA:0:12} ($AS_DIR) — refusing"
  n_id=0; for f in train_budget_meta.json fit_provenance.json derived_meta.json; do [ -f "$AS_DIR/$f" ] && n_id=$((n_id+1)); done
  [ "$n_id" -eq 1 ] || fatal "$AS_DIR declares $n_id identity sidecars (need exactly one)"
  [ -f "$AS_DIR/$AS_SIDECAR" ] || fatal "$AS_DIR lacks the pin's identity sidecar $AS_SIDECAR"
  log "alphasteer object verified: sha ${have:0:12} == pin, sidecar $AS_SIDECAR"
fi

# --- 4. the entries ---------------------------------------------------------------------------------------
problems=0
entry(){   # $1 = cell id, $2 = relative link target
  local dst="$VIEW/$1" tgt="$2"
  if [ -L "$dst" ]; then
    if [ "$(readlink "$dst")" != "$tgt" ]; then
      [ "$CHECK" = 1 ] && { log "PROBLEM $1 -> $(readlink "$dst") (want $tgt)"; problems=$((problems+1)); return; }
      rm -f "$dst"; ln -s "$tgt" "$dst"; log "re-pointed $1 -> $tgt"
    fi
  elif [ -e "$dst" ]; then
    log "PROBLEM $dst exists and is not a symlink — refusing to touch someone else's data"; problems=$((problems+1)); return
  else
    [ "$CHECK" = 1 ] && { log "PROBLEM missing entry $1"; problems=$((problems+1)); return; }
    ln -s "$tgt" "$dst"; log "linked $1 -> $tgt"
  fi
  [ -f "$dst/object.pt" ] || { log "PROBLEM $1: $dst/object.pt does not resolve"; problems=$((problems+1)); }
}
if [ "$CHECK" = 0 ]; then mkdir -p "$VIEW"; fi
[ -d "$VIEW" ] || fatal "no view at $VIEW (--check: not building)"
lock="$SHARED/exp/objects/misalign/.$TARGET.view.lock"; mkdir -p "$(dirname "$lock")"
exec 9>"$lock"; flock -w 900 9 || fatal "could not take $lock in 900s"
for c in "${MAIN_CELLS[@]}"; do
  [ -f "$MAIN_VIEW/$c/object.pt" ] || { log "PROBLEM wave view lacks $c/object.pt"; problems=$((problems+1)); continue; }
  entry "$c" "$REL_MAIN/$c"
done
if [ "$AS_STATUS" = served ]; then
  entry alphasteer "$REL_AS"
  if [ -L "$VIEW/alphasteer" ]; then
    have=$(sha256sum "$VIEW/alphasteer/object.pt" | cut -c1-64)
    [ "$have" = "$AS_SHA" ] || { log "PROBLEM alphasteer view entry hashes ${have:0:12} != pin"; problems=$((problems+1)); }
  fi
elif [ -L "$VIEW/alphasteer" ]; then
  [ "$CHECK" = 1 ] && { log "PROBLEM stale alphasteer entry (pin: $AS_STATUS)"; problems=$((problems+1)); } \
    || { rm -f "$VIEW/alphasteer"; log "removed stale alphasteer entry (pin: $AS_STATUS)"; }
elif [ -e "$VIEW/alphasteer" ]; then
  log "PROBLEM $VIEW/alphasteer exists and is not a symlink"; problems=$((problems+1))
fi
# stale entries: a SYMLINK the config no longer names (a cell that left the wave view — the PSR cells under the
# 2026-09-12 ruling) is removed in build mode and a PROBLEM under --check; anything that is not a symlink is
# never touched (someone else's data). The header's contract: "a stale one is removed".
mapfile -t HAVE < <(find "$VIEW" -mindepth 1 -maxdepth 1 ! -name 'VIEW_MANIFEST*' -printf '%f\n' 2>/dev/null | sort)
for h in "${HAVE[@]}"; do
  case " ${WANT[*]} " in *" $h "*) continue;; esac
  if [ -L "$VIEW/$h" ]; then
    [ "$CHECK" = 1 ] && { log "PROBLEM stale entry $h (not among the config's cells)"; problems=$((problems+1)); } \
      || { rm -f "$VIEW/$h"; log "removed stale entry $h (not among the config's cells)"; }
  else
    log "PROBLEM $VIEW/$h is not among the config's cells and is not a symlink — refusing to touch it"; problems=$((problems+1))
  fi
done
# completeness: exactly the expected set, nothing stale
mapfile -t HAVE < <(find "$VIEW" -mindepth 1 -maxdepth 1 ! -name 'VIEW_MANIFEST*' -printf '%f\n' 2>/dev/null | sort)
if [ "$(printf '%s\n' "${HAVE[@]}")" != "$(printf '%s\n' "${WANT[@]}" | sort)" ]; then
  log "PROBLEM view holds [${HAVE[*]}] but must hold exactly [${WANT[*]}]"; problems=$((problems+1))
fi
[ "$problems" -eq 0 ] || fatal "$problems problem(s) in $VIEW"

if [ "$CHECK" = 0 ]; then
  "$PY" - "$VIEW" "$TARGET" "$AS_STATUS" "$AS_SHA" "$RTARGET/$WVIEW" "${WANT[@]}" <<'PY'
import hashlib, json, os, sys, time
view, target, as_status, as_sha, wave_view, cells = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6:]
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""): h.update(chunk)
    return h.hexdigest()
entries = []
for c in cells:
    d = os.path.join(view, c)
    # staged_id = the resolved object dir's basename = the wave pins' key (base__/derived__/si12__/selected__…): the NATIVE
    # standup's view-manifest -> pins three-way check reads it (mt_serving_up.view_manifest_digests; D62, 2026-09-17)
    entries.append({"cell_id": c, "link": os.readlink(d), "resolved": os.path.realpath(d),
                    "staged_id": os.path.basename(os.path.realpath(d).rstrip("/")),
                    "object_sha256": sha(os.path.join(d, "object.pt")),
                    "source": "alphasteer_f22 (pin-verified)" if c == "alphasteer" else f"full22 {wave_view} view"})
json.dump({"wave": "misalign", "target": target, "view": view, "wave_view": wave_view, "alphasteer_status": as_status,
           "alphasteer_pin_sha256": None if as_sha == "-" else as_sha, "entries": entries,
           "written_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
          open(os.path.join(view, "VIEW_MANIFEST.json"), "w"), indent=2)
print("\n".join(f"view {target}: {e['cell_id']:<48} sha {e['object_sha256'][:12]}  <- {e['link']}" for e in entries))
PY
fi
log "ok: $VIEW (${#WANT[@]} entries; alphasteer: $AS_STATUS)"
