#!/bin/bash
# reproduce/09_dose_ladder.sh — stage 09: the Appendix D.2 dose ladder for the DIFFERENCE objects.
#
#   reproduce/09_dose_ladder.sh --model K [--alpha 0.75] [--objects difference_lora,difference_relu] [--evals static,multiturn,controls[,gsm8k]]
#                               [--step build|stage|configs|views|guard|submit|all[,...]] [--clusters a,b] [--twin] [--nshards 9]
#                               (all = build,stage,configs,guard,submit; `views` is named explicitly — it submits a cluster job of its own)
#                               [--force-gate LWID=REASON]... [--resubmit] [--config F] [--overrides F] [--dry-run] [--help]
#
# The dose objects are DIFFERENT BYTES (the registry pins base_scale 1.0): strength/build_strength.py scales the parent
# (scale_lora_dose.py / scale_psr_dose.py) in the SITE's objects bank ($SISTEER_OBJECTS_BANK, lib/common.sh — the one root 05_stage_roster.sh
# staged the parents into and 08_grade.sh hands the Petri drive; --parent-root is <bank>/<key>/cells), strength/stage_strength.py stages the copies onto every cluster (config-driven
# routing), the pins are refreshed, the roster's DOSE VIEWS d<tok> / d<tok>_relu render their own configs, `--step views` builds each
# cluster's bank view of them (strength/build_views.sbatch, CPU-only, watched; named explicitly, outside `all`), launch_guard.py --cluster <label> is the preflight
# ON every selected cluster, and the D.2 legs go out through lib/slurm.sh: the four static families (target_sampled static_base_t0/t1),
# Crescendo + ActorAttack, XSTest + OR-Bench (target_sampled controls_t0/t1) — the D.2 legs, the
# DEFAULT `--evals` set. The MULTI-TURN legs bind to a LIVE attacker pool, so `multiturn` is the dose view's multiturn FAMILY run through
# 06_eval_waves.sh --view <dose view> --family multiturn, per selected view: the ONE unit plan (lib/stage.py unit_plan — a dose view plans
# DOSE_LADDER_MT_ATTACKS, crescendo + actorattack, never pair) sizes the pool members from the view's targets and partitions the targets
# across every configured cluster (--clusters selects), the ONE funnel (lib/submit_twin.py) submits them, marks each group PLACED, prints
# the group-level backstop (`followup: arm … wait-pool --group mt-<view>-<cluster>`) and handles a group with NO LIVE MEMBER exactly as
# the battery wave's (exit 4, WAVE.<view>.partial.json + the printed REMEDY — 06's own `--resubmit` pass on the dose view re-mints a dead
# member or target); a group 06 left UNPLACED past any other mark-placed refusal (06 notes it and goes on, exit 0) is refused HERE on the
# funnel's read-only `placement-record` — attackers/PLACED on the group's cluster against the ledger — exit 3, LADDER.<view>.partial.json
# naming the group(s), before LADDER.<view>.json; --twin is not forwarded to it (06 refuses --twin for a multi-turn group; this stage's --twin
# elects its single-GPU legs). GSM8K rides this stage ONLY by explicit opt-in (`--evals …,gsm8k`): the dose views' GSM8K rows come from
# the same unit plan (gsm8k_thinking.sbatch on the front door's matrix frontdoor_<key>, one row per cell of the view). The script's ladder
# defaults are {0.5, 0.75, 1.25} for the differences and {0.15, 0.3, 0.5} for the refusal
# steer — only alpha 0.75 for the differences is run in the companion work; `--objects refusal_direction` is accepted by the grammar and
# REFUSED here by design (FD 5).
# A dose view whose staged id `d<tok>__<cell_id>` resolves to one of the paper's COMMITTED dose_scale copies (a row that declares
# `committed_objects_of`; objects/MANIFEST.json through reproduce/tools/import_objects.py cells — the worked example's 0.75 copies) is
# IMPORTED, not built: `build` writes its ledger row (source committed, built false, object_sha256 = the registry id, the sidecar's
# dose_scaled_from block) and runs no producer; `stage` copies it from this checkout through stage_objects.py (the pins through
# import_objects.py pins; the registry rows are the importer's); `guard` adds the local imported check under its OWN name, never the
# submit gate's pass. A dose with no committed copy takes the build/stage path above, unchanged.
# `--force-gate LWID=REASON` (repeatable) is the recorded admission override, exactly as 06 forwards it (lib/admission.py: an empty
# reason or an LWID outside the frozen inventory refuses); the forced list is printed and recorded in $STATE/LADDER.<view>.json. The
# refusals that stay un-forceable: run_in_this_work, a COMPLETE compute plan, a clean tree — the force is admission's only.
# `--resubmit` (with `--step views`, alone or in a list; refused by name with any other step) is the views units' fenced retry, forwarded to
# lib/submit_twin.py exactly as 06 forwards it: a bound unit whose job is still live is SKIPPED, a job sacct proves terminal is re-minted —
# strength/build_views.sbatch registers nothing (submit_twin.py NO_REGISTRATION_SCRIPTS names it), so sacct-terminal is the whole proof; a
# COMPLETED build refuses by name (its views stand: select the other cluster with --clusters). A plain re-run still skips a bound unit.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/lib/common.sh"; source "$HERE/lib/slurm.sh"
usage(){ sed -n '2,42p' "$0"; repro_common_help; }
ALPHA=0.75; OBJECTS=difference_lora,difference_relu; EVALS=static,multiturn,controls; STEP=all; CLUSTERS=""; TWIN=0; NSHARDS=9; FORCE=(); RESUBMIT=0   # gsm8k: opt-in only
LADDER_DEFAULT_DIFFERENCE="0.5 0.75 1.25"; LADDER_DEFAULT_REFUSAL="0.15 0.3 0.5"
repro_parse_common_args "$@"; set -- "${REPRO_ARGS[@]+"${REPRO_ARGS[@]}"}"
while [ $# -gt 0 ]; do
  case "$1" in
    --alpha) ALPHA="$2"; shift 2;; --objects) OBJECTS="$2"; shift 2;; --evals) EVALS="$2"; shift 2;; --step) STEP="$2"; shift 2;;
    --clusters) CLUSTERS="$2"; shift 2;; --twin) TWIN=1; shift;; --nshards) NSHARDS="$2"; shift 2;;
    --force-gate) FORCE+=("$2"); shift 2;;
    --resubmit) RESUBMIT=1; shift;;
    *) echo "REFUSED: unknown argument $1" >&2; usage >&2; exit 2;;
  esac
done
[ "$REPRO_HELP" = 1 ] && { usage; exit 0; }
for st in ${STEP//,/ }; do case "$st" in build|stage|configs|views|guard|submit|all) ;; *) repro_fatal "--step must be build|stage|configs|views|guard|submit|all (a comma list is fine), got '$st'";; esac; done
repro_load_config; repro_require_model
K="$REPRO_MODEL"; PY="$(repro_python)"; STATE="$(repro_state_dir "$K")"; SHA="$(repro_head_sha)"
repro_resolve_clusters
case ",$OBJECTS," in *,refusal_direction,*) repro_fatal "the refusal steer's ladder ($LADDER_DEFAULT_REFUSAL) is NOT run in this work (FD 5): its operating dose is alpha ${OVERRIDE_REFUSAL_STEER_ALPHA:-0.3}, confirmed working in the arms wave";; esac
for o in ${OBJECTS//,/ }; do case "$o" in difference_lora|difference_relu) ;; *) repro_fatal "--objects takes difference_lora,difference_relu (got '$o')";; esac; done
case " $LADDER_DEFAULT_DIFFERENCE " in *" $ALPHA "*) ;; *) repro_note "alpha $ALPHA is outside the D.2 ladder defaults {$LADDER_DEFAULT_DIFFERENCE}";; esac
TOK="d$(printf '%s' "$ALPHA" | sed 's/\./p/')"      # 0.75 -> d0p75 (the roster's dose-view spelling)
VIEW_LORA="$TOK"; VIEW_RELU="${TOK}_relu"
# `all` = build,stage,configs,guard,submit — the ladder's own steps; `views` submits a cluster job of its own (the bank-view build) and is
# named explicitly, like the opt-in gsm8k evals, so a default run composes only the wave's legs
want(){ case ",$STEP," in *,"$1",*) return 0;; *,all,*) [ "$1" != views ]; return $?;; *) return 1;; esac; }
# --resubmit is the views units' retry and nothing else's: the D.2 legs are not re-run through 09 (their re-run grammar is a call of its own), so
# the flag with no views step is refused by name rather than carried silently to a step that ignores it
[ "$RESUBMIT" = 0 ] || want views || repro_fatal "--resubmit is the views step's fenced retry (a dead llad-views job re-minted after the funnel's proof; a live one skipped) — pass --step views (alone or in a list; 'all' does not include views); the D.2 legs are not re-run through 09: a dead multi-turn pool member or target is 06's own fenced pass on the dose view (06_eval_waves.sh --model $REPRO_MODEL --view $VIEW_LORA|$VIEW_RELU --family multiturn … --resubmit — the call this stage's multiturn eval runs, with --resubmit), the static and controls legs a call of their own"
# the SITE's objects bank (lib/common.sh SISTEER_OBJECTS_BANK; round 1 on 3850329d6): the root 05 staged the parents into — build_strength.py's
# --bank and its --parent-root <bank>/<key>/cells below — never a root of this script's own (the singular `SISTEER_OBJECT_BANK` it read alone
# until then is refused by name in lib/common.sh)
BANK="$SISTEER_OBJECTS_BANK"; LEDGER="$STATE/09_build_ledger.json"; PINS_FILE="${SISTEER_PINS_FILE:-$REPRO_REPO/configs/sdfbase_full22_pins.json}"   # SISTEER_PINS_FILE: the test seam 05 honours too
IMPORTER="$REPRO_REPO/reproduce/tools/import_objects.py"
repro_require_runnable "$K"
# the forced admissions this call carries (Q10): parsed exactly as 06 does, printed here and in the ladder record below
FORCED_JSON="[]"; FORCED_LINE="none"
if [ ${#FORCE[@]} -gt 0 ]; then
  FORCED_JSON="$("$PY" -c 'import json,sys; print(json.dumps([dict(zip(("logical_work_id","reason"), a.split("=",1))) for a in sys.argv[1:]]))' "${FORCE[@]}")"
  FORCED_LINE="$("$PY" -c 'import sys; print(", ".join("%s (%s)" % tuple(a.split("=",1)) for a in sys.argv[1:]))' "${FORCE[@]}")"
fi
# the selected dose views (one per --objects entry)
SELECTED_VIEWS="$(echo $( case ",$OBJECTS," in *,difference_lora,*) echo "$VIEW_LORA";; esac ) $( case ",$OBJECTS," in *,difference_relu,*) echo "$VIEW_RELU";; esac ))"
repro_say "09 dose_ladder: model=$K alpha=$ALPHA views=$VIEW_LORA,$VIEW_RELU objects=$OBJECTS evals=$EVALS clusters=$CLUSTERS resubmit=$RESUBMIT sha=$SHA overrides=$OVERRIDE_ID forced: $FORCED_LINE"

# --- the committed copies (demo-import M7): per selected dose view, the committed dose_scale copy its staged id resolves to, or NONE ----------
# `<view>\t<cell_id>\t<staged_id>\t<object_id>\t<objects dir>\t<identity sidecar>` for a view whose copy is committed (reproduce/tools/import_objects.py
# cells over that one view), `<view>\tNONE\t<why>` otherwise (no committed source on the row, no committed copy at the view's dose, an unknown view)
COMMITTED="$("$PY" - "$REPRO_REPO" "$(repro_models_table)" "$K" $SELECTED_VIEWS <<'PYEOF'
import importlib.util, json, sys
repo, table, key, *views = sys.argv[1:]
spec = importlib.util.spec_from_file_location("_dose_ladder_import_objects", repo + "/reproduce/tools/import_objects.py")
IO = importlib.util.module_from_spec(spec); spec.loader.exec_module(IO)
for v in views:
    try:
        plan = IO.Plan(key, views=[v], table=table, repo=repo)
    except (SystemExit, KeyError, ValueError) as exc:        # ImportRefused is a SystemExit: a row with no committed source, no copy at this dose, no such view
        print(f"{v}\tNONE\t{str(exc).replace(chr(10), ' ')[:400]}")
        continue
    rows = [r for r in plan.rows if r.get("view") == v and r.get("dose") is not None]
    if len(rows) != 1:
        print(f"{v}\tNONE\t{len(rows)} committed dose rows for the view (one object cell per dose view)")
        continue
    r = rows[0]
    print("\t".join([v, r["cell_id"], r["staged_id"], r["object_sha256_expected"], r["dir"], r["identity_sidecar"]]))
PYEOF
)" || repro_fatal "the committed-copy resolution refused (above)"
committed_views(){ awk -F'\t' '$2 != "NONE" {print $1}' <<< "$COMMITTED"; }
uncommitted_views(){ awk -F'\t' '$2 == "NONE" {print $1}' <<< "$COMMITTED"; }
while IFS=$'\t' read -r v c sid oid dir side; do
  [ -n "$v" ] || continue
  if [ "$c" = NONE ]; then repro_say "dose view $v: no committed copy ($sid) — the build/stage path applies"
  else repro_say "dose view $v: committed copy objects/$dir ($sid, object $oid) — imported, not built"; fi
done <<< "$COMMITTED"

if want build; then
  # a committed copy: its ledger row in the builder's own schema (build_strength.py: one row per (target, staged_id) with its own scale), source
  # committed, built false, object_sha256 = the registry id, the sidecar's dose_scaled_from block copied; `out` is the bank dir --step stage fills
  # (so strength/stage_strength.py over this ledger hashes the staged copy and reads its staged_from.json like any built row); no producer runs
  while IFS=$'\t' read -r v c sid oid dir side; do
    [ -n "$v" ] && [ "$c" != NONE ] || continue
    if [ "$REPRO_DRY_RUN" = 1 ]; then echo "DRY-RUN: ledger row $LEDGER <- $K/$sid (committed copy objects/$dir imported, not built; object $oid)"; continue; fi
    "$PY" - "$REPRO_REPO" "$LEDGER" "$BANK" "$K" "$v" "$c" "$sid" "$oid" "$dir" "$side" "$ALPHA" <<'PYEOF' || repro_fatal "the committed ledger row for $sid could not be written (above)"
import json, os, sys, time
repo, ledger_p, bank, key, view, cell, sid, oid, rdir, side, alpha = sys.argv[1:]
doc = json.load(open(os.path.join(repo, "objects", rdir, side), encoding="utf-8"))
stamp = doc.get("dose_scaled_from")
if not isinstance(stamp, dict):
    sys.exit(f"REFUSED: objects/{rdir}/{side} carries no dose_scaled_from block — not a dose copy")
if abs(float(stamp.get("scale")) - float(alpha)) > 1e-9:
    sys.exit(f"REFUSED: objects/{rdir}/{side} dose_scaled_from.scale {stamp.get('scale')} is not --alpha {alpha}")
row = {"target": key, "family": "committed", "cell_id": cell, "parent_lane": str(stamp.get("parent_staged_id") or "").split("__", 1)[0],
       "parent_staged_id": stamp.get("parent_staged_id"), "staged_id": sid, "view": view, "scale": float(stamp["scale"]), "anchor": bool(stamp.get("anchor")),
       "producer": stamp.get("producer"), "object_type": stamp.get("object_type"), "parent_dir": stamp.get("parent_dir"), "parent_origin": stamp.get("parent_origin"),
       "parent_object_sha256": stamp.get("parent_object_sha256"), "object_sha256": oid, "scaling_method": stamp.get("scaling_method"),
       "tensors_scaled": stamp.get("tensors_scaled"), "identity_sidecar": side, "verification": stamp.get("verification"),
       "out": os.path.join(bank, key, "cells", sid), "parent_consumed_state": stamp.get("parent_consumed_state"), "producer_commit": stamp.get("producer_commit"),
       "built_at": stamp.get("written_at"), "source": "committed", "built": False, "committed_dir": f"objects/{rdir}", "dose_scaled_from": stamp}
ledger = {"bank": bank, "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "doses": [float(alpha)], "parent_root": None, "parent_state": None, "cells": [], "absent": []}
if os.path.isfile(ledger_p):
    try:
        prev = json.load(open(ledger_p, encoding="utf-8"))
        ledger.update({k: prev[k] for k in ("absent", "parent_root", "parent_state") if k in prev})
        ledger["doses"] = sorted({float(d) for d in (prev.get("doses") or [])} | {float(alpha)})
        ledger["cells"] = [c for c in prev.get("cells", []) if (c.get("target"), c.get("staged_id")) != (key, sid)]
    except ValueError:
        pass
ledger["cells"] = sorted(ledger["cells"] + [row], key=lambda c: (c.get("target"), c.get("view") or "", c.get("cell_id")))
os.makedirs(os.path.dirname(ledger_p), exist_ok=True)
tmp = ledger_p + ".tmp"; open(tmp, "w", encoding="utf-8").write(json.dumps(ledger, indent=1) + "\n"); os.replace(tmp, ledger_p)
print(f"ledger: {ledger_p} <- {key}/{sid} (committed copy objects/{rdir} imported, not built; object {oid[:12]})")
PYEOF
  done <<< "$COMMITTED"
  if [ -n "$(uncommitted_views)" ]; then
    repro_run "$PY" "$REPRO_REPO/scripts/sdfbase/full22/strength/build_strength.py" --bank "$BANK" --targets "$K" --doses "$ALPHA" --anchor --parent-root "$BANK/$K/cells" --ledger "$LEDGER" --python "$PY"
    repro_note "registry: each dose object is an \`object\` row with derivation dose_scale (emitted by the builder through hooks.emit_object_from_sidecar — workstream D)"
  fi
fi
if want stage; then
  if [ -n "$(committed_views)" ]; then
    # the committed copies from THIS checkout (no ssh pull) through stage_objects.py — it writes staged_from.json beside each copy and the staging
    # manifest row; the pins through import_objects.py pins (the manifest's verification facts ride along; a digest that is not the committed id
    # refuses); the registry rows are the importer's (reproduce/tools/import_objects.py registry), never a second object row
    CELLS="$("$PY" "$IMPORTER" cells --model "$K" --table "$(repro_models_table)" --views "$(committed_views | paste -sd, -)")" || repro_fatal "09: import_objects.py cells refused (above)"
    CELLS_FILE="$STATE/09_cells.json"; repro_mkdir "$BANK"; repro_write "$CELLS_FILE" "$CELLS"
    if [ "$REPRO_DRY_RUN" = 1 ]; then
      echo "DRY-RUN: $PY $REPRO_REPO/scripts/sdfbase/full22/stage_objects.py --site-dir $SISTEER_SITE_DIR --bank $BANK --target $K --cells $CELLS_FILE --clusters $CLUSTERS"
      "$PY" "$REPRO_REPO/scripts/sdfbase/full22/stage_objects.py" --site-dir "$SISTEER_SITE_DIR" --bank "$BANK" --target "$K" --cells <(printf '%s' "$CELLS") --clusters "$CLUSTERS" --dry-run
    else
      "$PY" "$REPRO_REPO/scripts/sdfbase/full22/stage_objects.py" --site-dir "$SISTEER_SITE_DIR" --bank "$BANK" --target "$K" --cells "$CELLS_FILE" --clusters "$CLUSTERS" || repro_fatal "staging did not verify on every cluster (records under $BANK/_records)"
    fi
    repro_run "$PY" "$IMPORTER" pins --model "$K" --table "$(repro_models_table)" --pins-file "$PINS_FILE" $( [ -f "$BANK/staging_manifest.json" ] && printf -- '--manifest %q' "$BANK/staging_manifest.json" ) \
      || repro_fatal "import_objects.py pins refused (above) — $PINS_FILE unchanged"
  fi
  if [ -n "$(uncommitted_views)" ]; then
    repro_run "$PY" "$REPRO_REPO/scripts/sdfbase/full22/strength/stage_strength.py" --ledger "$LEDGER" --clusters "$CLUSTERS" --site-dir "$SISTEER_SITE_DIR" --bank "$BANK" $(repro_dry_flag)
    repro_run "$PY" "$REPRO_REPO/scripts/sdfbase/make_full22_configs.py" --pins-from "$BANK/staging_manifest.json"
  fi
fi
if want configs; then
  ROSTER="$REPRO_REPO/configs/sdfbase_full22_roster.yaml"
  if "$PY" -c 'import sys,yaml; r=yaml.safe_load(open(sys.argv[1])); t=(r.get("targets") or {}).get(sys.argv[2]) or {}; sys.exit(0 if all(v in (t.get("views") or {}) for v in sys.argv[3:]) else 1)' "$ROSTER" "$K" "$VIEW_LORA" "$VIEW_RELU"; then
    repro_say "configs: the roster carries $K/$VIEW_LORA and $K/$VIEW_RELU"
  else
    repro_say "configs: the roster needs the dose views (dose: $ALPHA; dose_unit lora_write_basis_multiplier / psr_direction_norm_multiplier; object_family relu for $VIEW_RELU) and make_full22_configs.py's BARE_DOSE_VIEWS_FROZEN needs $K: (\"$VIEW_LORA\",) with a citation — the roster/generator owner adds them"
    [ "$REPRO_DRY_RUN" = 1 ] || repro_fatal "roster lacks the dose views for $K"
  fi
  repro_run "$PY" "$REPRO_REPO/scripts/sdfbase/make_full22_configs.py"
  repro_run "$PY" "$REPRO_REPO/scripts/sdfbase/make_full22_configs.py" --check
fi
if want views; then
  # the dose views' BANK VIEWS on every selected cluster — strength/build_views.sbatch <sha> <key> <views> (make_bank_view.py build + --check per view;
  # CPU-only, no --gres) through the funnel (lib/slurm.sh: the mirrored checkout at the pinned sha first, the watcher armed per job). The sbatch
  # derives the shared root by its own two-root probe and reads the checkout at <CLUSTER_EVAL_REPO_ROOT>/<sha> and the fit venv
  # <shared>/envs/sdfbase-fit: the site's CLUSTER_EVAL_REPO_ROOT must be site.example's default (a path under $SHARED) and the fit venv
  # must exist on the cluster (Q11 names the sbatch's own root probe as out of scope here — the GUARD's root is the cluster block's regardless)
  # (the sbatch registers nothing and the funnel's non-registering allow-list — lib/submit_twin.py NO_REGISTRATION_SCRIPTS — names it, so a dead
  # views job is re-minted through `--resubmit`'s fenced proof on sacct alone, never by assertion and never while its job is live: the flag is
  # forwarded per unit below exactly as 06 forwards its own; without it a bound unit is skipped, the funnel's idempotence by logical work id)
  # one unit PER RUNG per cluster: the dose token is in the unit, the logical work id and the job name, since the funnel is idempotent by
  # logical work id (a unit already submitted is skipped and its earlier job returned) — a second rung on the same cluster is its own unit,
  # never the first rung's binding handed back with the first rung's views built
  repro_require_clean_tree
  slurm_mirror_clusters "$SHA" "$CLUSTERS"
  extra=(); [ "$RESUBMIT" = 1 ] && extra+=(--resubmit)   # the fenced retry: skip a live job, re-mint a terminal one (contract none: sacct is the proof)
  for l in ${CLUSTERS//,/ }; do
    slurm_submit --unit "ladder.views.$TOK.$l" --lwid "$K/ladder/views/$TOK/$l" --script scripts/sdfbase/full22/strength/build_views.sbatch --name "llad-views-$K-$TOK-$l" \
      --args "$SHA $K $VIEW_LORA,$VIEW_RELU" --clusters "$l" --kind other --time 01:15:00 --wave-sha "$SHA" ${extra[@]+"${extra[@]}"} \
      || { [ "$REPRO_DRY_RUN" = 1 ] || repro_fatal "the bank-view build of $VIEW_LORA,$VIEW_RELU on $l did not submit (above)"; }
  done
  repro_note "views: wait for every llad-views-$K-$TOK-<cluster> job's RESULT line (rc=0, <view>=ok per view) before --step guard — the guard reads the VIEW_MANIFEST.json each build writes; a job that died: --step views --resubmit (a live job is skipped, a terminal one re-minted after the funnel's proof)"
fi
if want guard; then
  # the cluster bank, where the job will read (Q11): one launch_guard.py --cluster <label> per selected cluster, the label's SITE block the only source of
  # the host and the bank root (<CLUSTER_SHARED_ROOT>/exp/objects/sdfbase/full22, verbatim) — disk sha = view manifest sha = roster pin, exit 0 the only pass
  for l in ${CLUSTERS//,/ }; do
    repro_run "$PY" "$REPRO_REPO/scripts/sdfbase/full22/strength/launch_guard.py" --cluster "$l" --site-dir "$SISTEER_SITE_DIR" --target "$K" --views "$VIEW_LORA,$VIEW_RELU" --pins "$PINS_FILE" --json "$STATE/09_launch_guard.$l.json" \
      || repro_fatal "launch_guard.py refused on cluster $l (above) — nothing submitted"
  done
  # the imported copies' LOCAL check, under its OWN name — the offline path, never the submit gate's pass: the bank's object.pt hashes to the pin
  # for (target, staged id) and objects/MANIFEST.json holds the id (the registry's object_id)
  while IFS=$'\t' read -r v c sid oid dir side; do
    [ -n "$v" ] && [ "$c" != NONE ] || continue
    if [ "$REPRO_DRY_RUN" = 1 ]; then echo "DRY-RUN: imported check $K/$v $c: sha256($BANK/$K/cells/$sid/object.pt) == pins[$K][$sid] == objects/MANIFEST.json $oid -> PASS (imported, local)"; continue; fi
    "$PY" - "$REPRO_REPO" "$PINS_FILE" "$BANK" "$K" "$v" "$c" "$sid" "$oid" <<'PYEOF' || repro_fatal "the imported local check refused for $K/$v (above) — nothing submitted"
import hashlib, json, os, sys
repo, pins_p, bank, key, view, cell, sid, oid = sys.argv[1:]
obj = os.path.join(bank, key, "cells", sid, "object.pt")
if not os.path.isfile(obj):
    sys.exit(f"REFUSE {key}/{view} {cell}: {obj} is missing — run --step stage first")
h = hashlib.sha256()
with open(obj, "rb") as fh:
    for chunk in iter(lambda: fh.read(1 << 20), b""):
        h.update(chunk)
disk = h.hexdigest()
pin = ((json.load(open(pins_p, encoding="utf-8")).get(key) or {}).get(sid) or {}).get("object_sha256")
man = {r["object_id"] for r in json.load(open(os.path.join(repo, "objects", "MANIFEST.json"), encoding="utf-8"))["objects"]}
if pin != oid:
    sys.exit(f"REFUSE {key}/{view} {cell}: pins[{key}][{sid}] = {str(pin)[:12]} is not the committed object {oid[:12]} — run --step stage (import_objects.py pins)")
if disk != oid:
    sys.exit(f"REFUSE {key}/{view} {cell}: {obj} hashes to {disk[:12]}, not the committed object {oid[:12]}")
if oid not in man:
    sys.exit(f"REFUSE {key}/{view} {cell}: objects/MANIFEST.json holds no object {oid[:12]}")
print(f"imported check {key}/{view} {cell}: disk={disk[:12]} pin={pin[:12]} manifest=objects/MANIFEST.json PASS (imported, local)")
PYEOF
  done <<< "$COMMITTED"
fi
if want submit; then
  REGISTRY_DB="$(repro_registry_root)/registry.sqlite"
  args=(); for f in ${FORCE[@]+"${FORCE[@]}"}; do args+=(--force-gate "$f"); done
  # the ladder is priced on the plan's serving rows: a COMPLETE plan (the dry run reads an on-disk plan too and refuses
  # an incomplete one by name — lib/common.sh repro_require_compute_plan `complete`)
  if [ "$REPRO_DRY_RUN" = 1 ]; then
    echo "DRY-RUN: $PY $HERE/lib/admission.py check --state-root $STATE --registry $REGISTRY_DB${args[*]:+ ${args[*]}}"
    if [ -f "$STATE/compute_plan.json" ]; then repro_require_compute_plan "$K" complete; echo "DRY-RUN: compute plan $STATE/compute_plan.json is complete"
    else echo "DRY-RUN: require a COMPLETE $STATE/compute_plan.json (tools/replan_compute.py --probe the certified probe_verdict.json)"; fi
  else
    "$PY" "$HERE/lib/admission.py" check --state-root "$STATE" --registry "$REGISTRY_DB" "${args[@]+"${args[@]}"}" || repro_fatal "the unsteered battery does not admit the ladder yet"
    repro_require_compute_plan "$K" complete; repro_require_clean_tree
  fi
  slurm_mirror_clusters "$SHA" "$CLUSTERS"     # the pinned sha on every target cluster BEFORE the first submission (§3.9 step 1)
  BASE_ENV="$(slurm_base_env)"
  for view in $SELECTED_VIEWS; do
    TAG="${K}_$view"
    for ev in ${EVALS//,/ }; do
      case "$ev" in
        static)    for t in 0 1; do slurm_submit --unit "ladder.$view.static_t$t" --lwid "$K/$view/static/t$t" --script scripts/sdfbase/full22/sampled/target_sampled.sbatch --name "f22s-$K-$TAG-static_base_t$t" --args "$K static_base_t$t $SHA $NSHARDS '' $view" --clusters "$CLUSTERS" --env "$BASE_ENV" --time "$SISTEER_TIME_EVAL" --wave-sha "$SHA" $(slurm_twin_flag); done;;
        multiturn) # The D.2 multi-turn legs (crescendo + actorattack) bind to a LIVE attacker pool, so they are the dose view's multiturn FAMILY — the
                   # pool members sized from the view's targets, the targets partitioned across every configured cluster (--clusters selects), the
                   # placement transition, mark-placed with --supersede on a --resubmit pass, the backstop's arm line and the no-live-member handling —
                   # run through 06_eval_waves.sh's own code path for the dose view: ONE planner (lib/stage.py unit_plan: a dose view plans
                   # DOSE_LADDER_MT_ATTACKS), ONE funnel (lib/submit_twin.py), ONE shell, nothing composed here (gap-09-attacker-pool; before, this
                   # stage composed the targets by hand onto the FIRST selected cluster, carried no F22_MT_TARGET_POLICY_REF / seed base, and started
                   # no pool — the targets waited F22_ATTACKER_WAIT_S for a member nobody submitted and ended fatal). The call carries THIS pass's
                   # effective flags: --clusters (the selection; the placement universe is the site's), --nshards (shapes the shard groups and sizes the
                   # pool), --wave-sha (pins THIS sha, never 06's HEAD default), --force-gate (06 re-checks the steered view's admission on every pass:
                   # the forced units travel), the resolved overrides document and the site config (as 06's own remedy spells them), the dry flag.
                   # --twin is NOT forwarded: 06 refuses it for a multi-turn group (no group election), and this stage's --twin elects its single-GPU
                   # legs above and below. --resubmit is not forwarded either: the re-run of a dead member or target is 06's fenced pass on the dose
                   # view (the REMEDY 06 prints on NO LIVE MEMBER is exactly that call with --resubmit; the note below spells it), never this stage's
                   # views-step retry. 06's exit code is this pass's: 4 (placement INCOMPLETE — a group with NO LIVE MEMBER; its WAVE.<view>.partial.json
                   # names the group(s) and the REMEDY, printed verbatim above) and 2 (a refusal: the plan, the manifest, admission, a failed
                   # submission) both end this pass before the ladder record; a 06 pass that ended 0 past a mark-placed refusal it notes and goes on
                   # from is caught below, on the funnel's read-only placement-record — so LADDER.<view>.json is written only over a placed multi-turn family.
                   mt=("$HERE/06_eval_waves.sh" --model "$K" --view "$view" --family multiturn --clusters "$CLUSTERS" --nshards "$NSHARDS" --wave-sha "$SHA")
                   for f in ${FORCE[@]+"${FORCE[@]}"}; do mt+=(--force-gate "$f"); done
                   mt+=(--overrides "$SISTEER_RUN_OVERRIDES" --config "$SISTEER_CONFIG")
                   [ "$REPRO_DRY_RUN" = 1 ] && mt+=(--dry-run)
                   repro_say "multiturn legs of $view: the attacker pool + its targets are the dose view's multiturn family through 06 (pool members sized from the view's crescendo + actorattack targets, partitioned over $(cluster_labels | tr ' ' ','); --clusters $CLUSTERS selects): ${mt[*]}"
                   "${mt[@]}"; mt_rc=$?
                   if [ "$mt_rc" = 4 ]; then
                     REPRO_REFUSE_RC=4 repro_fatal "multiturn legs of $view: placement INCOMPLETE (06 exit 4, above) — a group has NO LIVE MEMBER: $STATE/WAVE.$view.partial.json names the group(s) and the REMEDY (printed verbatim above: 06's own --resubmit pass on view $view, run once sacct reports the exited member job(s) terminal); $STATE/LADDER.$view.json is NOT written; the other evals of this call that went out above stand (every submission is in $STATE/submissions.jsonl)"
                   elif [ "$mt_rc" != 0 ]; then
                     repro_fatal "multiturn legs of $view: 06_eval_waves.sh --view $view --family multiturn refused (exit $mt_rc, above) — $STATE/LADDER.$view.json is NOT written; fix the cause and run this call again (units already submitted are skipped)"
                   fi
                   # 06 ends its pass with exit 0 — WAVE.<view>.json written, `06 done` printed — past a mark-placed
                   # refusal OTHER than NO LIVE MEMBER: the placement transaction failed on the cluster (an ssh outage, a PLACED.lock it could not
                   # take — the funnel's rc 3, nothing recorded) or the immutable-placement fence refused the roster (rc 3). 06 notes the group
                   # "not fully placed yet — mark-placed runs again on the next pass" and goes on, a note the operator running 06 reads and this
                   # stage's caller did not: before, the ladder record was written and `09 done` printed over an UNPLACED group — its targets bind
                   # round-robin instead of their recorded assignment, and wait-pool cannot tear its pool down (it refuses without attackers/PLACED),
                   # so the members idle on their GPUs to their walltime. So the ladder record waits on the AUTHORITATIVE record: the funnel's
                   # READ-ONLY placement-record reads attackers/PLACED on each selected group's cluster against the ledger's bound targets and
                   # recorded members (never 06's output). `placed` and `empty` (no target of the group went out — 06's own note) pass; `none`
                   # (no record: the placement never committed) and `stale` (a record that predates the roster) refuse, exit 3 — the funnel's own
                   # code for a placement refusal, distinct from 4 (NO LIVE MEMBER) and 2 (a refusal) — with the group(s) recorded in
                   # LADDER.<view>.partial.json and LADDER.<view>.json NOT written. The remedy is this same call again (every unit already submitted
                   # is skipped; 06's mark-placed runs again and commits), or the sanctioned supersession 06 printed if its note names the fence.
                   unplaced=""
                   for l in ${CLUSTERS//,/ }; do
                     "$PY" "$HERE/lib/submit_twin.py" --config "$SISTEER_CONFIG" --model "$K" ${REPRO_OVERRIDES:+--overrides "$REPRO_OVERRIDES"} $(repro_dry_flag) placement-record --group "mt-$view-$l" \
                       || unplaced="$unplaced${unplaced:+,}mt-$view-$l"
                   done
                   if [ -n "$unplaced" ]; then
                     repro_write "$STATE/LADDER.$view.partial.json" "{\"model\":\"$K\",\"view\":\"$view\",\"alpha\":\"$ALPHA\",\"objects\":\"$OBJECTS\",\"evals\":\"$EVALS\",\"clusters\":\"$CLUSTERS\",\"wave_sha\":\"$SHA\",\"overrides\":\"$OVERRIDE_ID\",\"forced\":$FORCED_JSON,\"unplaced_groups\":$(printf '%s' "$unplaced" | "$PY" -c 'import json,sys; print(json.dumps([g for g in sys.stdin.read().split(",") if g]))'),\"utc\":\"$(date -u +%FT%TZ)\"}"
                     REPRO_REFUSE_RC=3 repro_fatal "multiturn legs of $view: group(s) $unplaced are NOT placed (placement-record above) although 06 ended its pass — its mark-placed refused (06's note above names the refusal: the placement transaction failed on the cluster, or the immutable-placement fence) and 06 notes such a refusal and goes on (exit 0, WAVE.$view.json written): attackers/PLACED does not name the group's targets at their bound jobs, so they would bind round-robin and wait-pool cannot tear the pool down; $STATE/LADDER.$view.json is NOT written (the group(s) are recorded in $STATE/LADDER.$view.partial.json; every submission is in $STATE/submissions.jsonl); REMEDY: fix the cause and run this same call again — every unit already submitted is skipped and 06's mark-placed runs again for the group — or, if 06's note names the immutable-placement fence, the sanctioned supersession it printed verbatim above; arm wait-pool only after the arm line"
                   fi
                   repro_note "multiturn legs of $view: a dead pool member or target is re-run through 06's fenced pass on the dose view, never through this stage's --resubmit (the views step's): $HERE/06_eval_waves.sh --model $K --view $view --family multiturn --clusters $CLUSTERS --nshards $NSHARDS --wave-sha $SHA$(for f in ${FORCE[@]+"${FORCE[@]}"}; do printf -- ' --force-gate %q' "$f"; done) --overrides $SISTEER_RUN_OVERRIDES --config $SISTEER_CONFIG --resubmit  (the placement transition opens, the walk re-mints what sacct proves dead, mark-placed supersedes the group's PLACED with the pass's provenance; arm wait-pool only after the arm line above)";;
        controls)  for t in 0 1; do slurm_submit --unit "ladder.$view.controls_t$t" --lwid "$K/$view/controls/t$t" --script scripts/sdfbase/full22/sampled/target_sampled.sbatch --name "f22s-$K-$TAG-controls_t$t" --args "$K controls_t$t $SHA $NSHARDS '' $view" --clusters "$CLUSTERS" --env "$BASE_ENV" --time "$SISTEER_TIME_EVAL" --wave-sha "$SHA" $(slurm_twin_flag); done;;
        gsm8k)     # OPT-IN (never in the default set — the D.2 legs are FD 13's): the dose view's GSM8K rows from the ONE unit plan (lib/stage.py,
                   # the rows 06 submits for a view): gsm8k_thinking.sbatch per cell of the view on the front door's matrix frontdoor_<key>
                   # (the roster's cell id; rendered by 05 --step configs)
                   PLAN_TSV="$("$PY" "$HERE/lib/stage.py" plan --model "$K" --view "$view" --families gsm8k --nshards "$NSHARDS" --sha "$SHA" --models-table "$(repro_models_table)")" || repro_fatal "the gsm8k unit plan refused (above)"
                   while IFS=$'\t' read -r sf fam unit lwid cl kind grp script name array tkey twin_ok args env depends; do   # `depends` = the TSV's trailing column; without it `read` folded it into env ("-\t-") and the funnel refused every gsm8k row
                     [ -n "$lwid" ] || continue
                     [ "$args" = - ] && args=""; env_str="$BASE_ENV"; [ "$env" != - ] && env_str="$BASE_ENV $env"
                     slurm_submit --unit "ladder.$view.gsm8k.$unit" --lwid "$lwid" --script "$script" --name "$name" --args "$args" --clusters "$CLUSTERS" --env "$env_str" --time "0-01:30:00" --wave-sha "$SHA" $(slurm_twin_flag) \
                       || { [ "$REPRO_DRY_RUN" = 1 ] || repro_fatal "submission of ladder.$view.gsm8k.$unit failed (above)"; }
                   done <<< "$PLAN_TSV";;
        *) repro_fatal "--evals takes static,multiturn,controls (the D.2 legs, the default) and the opt-in gsm8k (the dose views' GSM8K rows)";;
      esac
    done
    # the ladder record (Q10, mirroring 06's WAVE record): what this call submitted for the view and the forced admissions it carried
    repro_write "$STATE/LADDER.$view.json" "{\"model\":\"$K\",\"view\":\"$view\",\"alpha\":\"$ALPHA\",\"objects\":\"$OBJECTS\",\"evals\":\"$EVALS\",\"clusters\":\"$CLUSTERS\",\"wave_sha\":\"$SHA\",\"overrides\":\"$OVERRIDE_ID\",\"forced\":$FORCED_JSON,\"utc\":\"$(date -u +%FT%TZ)\"}"
  done
  [ "$FORCED_LINE" = none ] || repro_note "decisions.md: forced admissions for $K ($SELECTED_VIEWS): $FORCED_LINE — recorded in $STATE/LADDER.<view>.json (gate/forced.jsonl is append-only and nothing reads it back)"
  repro_note "registry: run rows carry strength_nominal $ALPHA, strength_unit per object family, is_operating_point 0; the realised dose: strength/dose_capture.py --target $K --out-dir $STATE/09_dose_capture"
fi
repro_say "09 done; forced: $FORCED_LINE"
