#!/bin/bash
# reproduce/05_stage_roster.sh — stage 05: stage the fitted objects onto every cluster, refresh the pins, generate the
# PHASE views' configs and manifests, build the bank views, and FREEZE the unsteered-first gate's logical inventory.
#
#   reproduce/05_stage_roster.sh --model K [--views <comma list of the model's phase views>] [--clusters a,b] [--bank DIR]
#                                [--fit-cluster <label>] [--step stage|pins|roster|configs|bank-views|freeze|all] [--filler PG19]
#                                [--from-committed | --from-bank] [--config F] [--overrides F] [--dry-run] [--help]
#
#   stage        scripts/sdfbase/full22/stage_objects.py: each view's objects (models.yaml `objects`) from the fit cluster's
#                bank ($SHARED/exp/objects/sdfbase/full22/<fit_model_id>/cells_all/<si>__<arm>/, derived/…; the refusal direction from
#                04's publication <SISTEER_REFUSAL_OUT_ROOT | $SHARED/exp/objects/refusal_direction>/<serve.checkpoint.model_id>/<cell_id>,
#                the SERVED checkpoint's directory, pinned by the generator's registry rewrite `make_full22_configs.py
#                --derived-pins-from-registry` from the row 04's fit emits under that id) into the
#                grading host's bank (--bank, default $SISTEER_OBJECTS_BANK = $SISTEER_STATE_DIR/objects unless the site sets it; the
#                same root 08_grade.sh hands the Petri drive and 09_dose_ladder.sh scales the dose copies from), then onto EVERY cluster, sha-verified;
#                writes staging_manifest.json. A view whose objects are not fitted yet (u0 has none) needs nothing.
#                --from-committed (the DEFAULT for a row that declares `committed_objects_of`, the worked example): the objects
#                are the paper's COMMITTED ones (objects/DEMO.json; reproduce/tools/import_objects.py cells), copied from this
#                checkout into the bank in place of stage 04's fits, each held to its registry id; --from-bank forces the fit path
#   pins         scripts/sdfbase/make_full22_configs.py --pins-from <staging_manifest.json> --registry-root <the site's registry> --campaign <the
#                site's campaign: the one 04 registered under, SISTEER_REGISTRY_CAMPAIGN, default sdfbase_full22>; a manifest
#                staged from committed objects writes the model's block with import_objects.py pins — the SAME registry-backed writer
#                (pins_from_staging_manifest) scoped to the model, so a digest with no reference|approved row, a withdrawn row, a copy of
#                another parent or dose, or a changed committed entry refuses by name (every other key of the pins file byte-identical)
#   roster       the target's COMPLETE block for configs/sdfbase_full22_roster.yaml (lib/models.py roster-block: served checkpoint,
#                battery body, the phase views with their config / manifest paths and cells) — PRINTED; the stage REFUSES to
#                continue until the roster carries the target (the roster is the run's file, edited by its owner)
#   configs      scripts/sdfbase/make_full22_configs.py (+ --check): configs/sdfbase_full22_<key>[_<view>].yaml + the Petri
#                manifests; the padding generator (the prompted-only family + one steered family per phase view); the
#                misalign manifest; the sampled configs; the GSM8K matrix frontdoor_<key> + its overlays
#   bank-views   scripts/sdfbase/full22/make_bank_view.py --target K --view V [--check] on each cluster's checkout
#   freeze       reproduce/lib/stage.py freeze-inventory -> <state>/gate/expected_units.json (LOGICAL units, never run ids)
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/lib/common.sh"
usage(){ awk 'NR > 1 { if ($0 !~ /^#/) exit; print }' "$0"; repro_common_help; }   # the whole header block above (every line to the first non-comment), however it grows — a hand-kept line range truncated it (gap-refusal-pin)
VIEWS=""; CLUSTERS=""; BANK=""; FIT_CLUSTER=""; STEP=all; FILLER="${SISTEER_PG19_FILLER:-}"; FROM_COMMITTED=auto
repro_parse_common_args "$@"; set -- "${REPRO_ARGS[@]+"${REPRO_ARGS[@]}"}"
while [ $# -gt 0 ]; do
  case "$1" in
    --views) VIEWS="$2"; shift 2;; --clusters) CLUSTERS="$2"; shift 2;; --bank) BANK="$2"; shift 2;;
    --fit-cluster) FIT_CLUSTER="$2"; shift 2;; --step) STEP="$2"; shift 2;; --filler) FILLER="$2"; shift 2;;
    --from-committed) FROM_COMMITTED=1; shift;; --from-bank) FROM_COMMITTED=0; shift;;
    *) echo "REFUSED: unknown argument $1" >&2; usage >&2; exit 2;;
  esac
done
[ "$REPRO_HELP" = 1 ] && { usage; exit 0; }
case "$STEP" in stage|pins|roster|configs|bank-views|freeze|all) ;; *) repro_fatal "--step must be stage|pins|roster|configs|bank-views|freeze|all";; esac
repro_load_config; repro_require_model
K="$REPRO_MODEL"; PY="$(repro_python)"
[ -n "$VIEWS" ] || VIEWS="$("$PY" "$HERE/lib/models.py" --table "$(repro_models_table)" views "$K" | tr '\n' ',' | sed 's/,$//')"
repro_resolve_clusters
[ -n "$FIT_CLUSTER" ] || FIT_CLUSTER="$SISTEER_PRIMARY_CLUSTER"; cluster_require "$FIT_CLUSTER"
[ -n "$BANK" ] || BANK="$SISTEER_OBJECTS_BANK"   # the site's one objects-bank root (lib/common.sh): what 08 hands the Petri drive reads
FIT_ID="$(repro_model "$K" fit.checkpoint.model_id)"; SHA="$(repro_head_sha)"; STATE="$(repro_state_dir "$K")"
FIT_SHARED="$(cluster_var "$FIT_CLUSTER" CLUSTER_SHARED_ROOT)"
want(){ [ "$STEP" = all ] || [ "$STEP" = "$1" ]; }
# demo-import: a row that declares `committed_objects_of` imports the paper's committed objects (objects/DEMO.json) in place of stage 04's fits —
# the default source for it; --from-committed / --from-bank force either way (a row that declares none refuses --from-committed by name)
COMMITTED_OF="$("$PY" "$HERE/lib/models.py" --table "$(repro_models_table)" get "$K" committed_objects_of 2>/dev/null || true)"
if [ "$FROM_COMMITTED" = auto ]; then [ -n "$COMMITTED_OF" ] && FROM_COMMITTED=1 || FROM_COMMITTED=0; fi
IMPORTER="$REPRO_REPO/reproduce/tools/import_objects.py"
repro_say "05 stage_roster: model=$K fit_model_id=$FIT_ID views=$VIEWS clusters=$CLUSTERS fit_cluster=$FIT_CLUSTER bank=$BANK sha=$SHA overrides=$OVERRIDE_ID source=$( [ "$FROM_COMMITTED" = 1 ] && echo "committed (objects/DEMO.json via reproduce/tools/import_objects.py; committed_objects_of=$COMMITTED_OF)" || echo "fit bank on $FIT_CLUSTER")"

# --- the objects every requested view needs, as stage_objects.py's cells.json --------------------------------------------
cells_json(){   # -> a JSON list over the union of the views' object cells (none / prompted__* carry no object)
  if [ "$FROM_COMMITTED" = 1 ]; then
    # the COMMITTED source (demo-import M1): the paper's objects from this checkout, each row `source: committed:objects/<dir>` held to its
    # registry id (objects/DEMO.json; a dose view resolves to the committed dose_scale copy); refuses by name a row without `committed_objects_of`
    "$PY" "$IMPORTER" cells --model "$K" --views "$VIEWS" --table "$(repro_models_table)"
    return
  fi
  "$PY" - "$HERE/lib" "$(repro_models_table)" "$K" "$VIEWS" "$FIT_SHARED" "$FIT_ID" "$FIT_CLUSTER" "${SISTEER_PINS_FILE:-$REPRO_REPO/configs/sdfbase_full22_pins.json}" <<'PYEOF'
import json, os, sys
lib, table, key, views, shared, fit_id, fit_cluster, pins_path = sys.argv[1:]
sys.path.insert(0, lib); import models as M          # the duplicate-key-rejecting loader (review A9), never a bare safe_load
m = M.model(key, table)
objs = m.get("objects") or {}
# s3r-fix (D1297 P1-2): the committed pins are the REGISTRY's object ids (a derived pin = the registered object_id of, say, a
# model's difference_relu) — the stager compares the local object.pt digest to the pin BEFORE copying, before accepting completion
# and before any pin rewrite; a derived object with NO pin is never staged by directory discovery (the pin comes from the registry hand-off)
try:
    pins = json.load(open(pins_path, encoding="utf-8")) if pins_path and os.path.isfile(pins_path) else {}
except (OSError, ValueError) as exc:
    raise SystemExit(f"{key}: cannot read the pins {pins_path} ({exc}) — the staged objects are held to the registry's ids")
# gap-refusal-pin (2026-10-12): the refusal direction is read where 04_fit_objects.sh's refusal_direction() publishes it — <SISTEER_REFUSAL_OUT_ROOT |
# $SHARED/exp/objects/refusal_direction>/<serve.checkpoint.model_id>/<cell_id>: the SERVED checkpoint's directory (scripts/sbatch/
# fit_refusal_direction.sbatch: OUT_DIR = <out_root>/<model_key = the served registry id>/<staged id>), the object's registry identity naming its
# directory so every key that serves the checkpoint reads ONE publication (the review on 494c36ce9, finding 1: keyed by the table key, a second key
# serving the same checkpoint read 04 done and pulled from an empty directory; the site key rides the environment, exported by
# repro_load_config) — never the paper's arditi_allpos ladder bank, which no front-door fit writes
refusal_root = (os.environ.get("SISTEER_REFUSAL_OUT_ROOT") or "$SHARED/exp/objects/refusal_direction").replace("$SHARED", shared)
serve_id = str(((m.get("serve") or {}).get("checkpoint") or {}).get("model_id") or "")
seen, out = set(), []
for v in views.split(","):
    for cell in (m.get("views") or {}).get(v, {}).get("cells", []):
        if cell == "none" or cell.startswith("prompted__") or cell in seen or cell not in objs:
            continue
        seen.add(cell); o = objs[cell]
        if o["source"] == "derived_bank" and not o.get("derived_dir"):
            # D1221: a derived object is published by its build under <full22>/derived/<kind dir>/<fit_id>/<cell_id> (04_fit_objects.sh:
            # lora_subtraction, psr_paired_write, psr_residual) — the table names that kind dir; <fit_id>/derived/<cell_id> was never a real layout
            raise SystemExit(f"{key}: object {cell} ({o['cell_id']}) has source derived_bank but no derived_dir (the kind directory under "
                             f"<full22>/derived/, e.g. psr_paired_write / lora_subtraction) — refusing to guess where the build published it")
        # D1890 / D1893: an object may live in a SIBLING bank of the same layout (`bank: full22_lecun` — the LeCun-init refits); default full22
        bank = str(o.get("bank") or "full22")
        if o["source"] == "arditi_bank" and not serve_id:
            raise SystemExit(f"{key}: object {cell} ({o['cell_id']}) is a SERVED-checkpoint object (source arditi_bank: 04 publishes it under "
                             "serve.checkpoint.model_id) but the row declares no serve.checkpoint.model_id — refusing to guess where 04 published it")
        if o["source"] == "arditi_bank":
            # gap-refusal-pin (the review on ea634b165, MAJOR): the cell is read LITERALLY as the directory below and as the pin's key, while 04 publishes at the
            # fitter's spelling of the coefficient — so the cell must BE that spelling, judged by the pins writer's own cell reading (the one 04's
            # identity read and the pin's rung come through): `ar_c0p30` would be pinned at 0.3 and pulled from a directory nothing writes
            sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(lib)), "scripts", "sdfbase"))
            import make_full22_configs as G
            try:
                G.arditi_cell_coefficient(str(o["cell_id"]))
            except SystemExit as exc:
                raise SystemExit(f"{key}: object {cell}: {exc} — 04 publishes the refusal direction under <refusal root>/{serve_id}/<the fitter's spelling>, "
                                 "this stage reads the table's cell literally, and the pin is keyed by it; refusing to stage a cell no fit publishes under")
        root = {"fit_bank": f"{shared}/exp/objects/sdfbase/{bank}/{fit_id}/cells_all/{o['cell_id']}",
                "derived_bank": f"{shared}/exp/objects/sdfbase/{bank}/derived/{o.get('derived_dir')}/{fit_id}/{o['cell_id']}",
                "arditi_bank": f"{refusal_root}/{serve_id}/{o['cell_id']}"}[o["source"]]
        row = {"cell_id": o["cell_id"], "lane": o["lane"], "kind": o["kind"], "source": f"{fit_cluster}:{root}", "view_cell": cell}
        staged_id = f"{o['lane']}__{o['cell_id']}"
        pin = ((pins.get(key) or {}).get(staged_id) or {}).get("object_sha256")
        if o["kind"] == "derived" and not pin:
            # gap-refusal-pin: the refusal direction's pin is the row 04's fit emits, written by the generator's registry rewrite — the refusal names the two commands
            how = ("04_fit_objects.sh --object refusal_direction fits + registers it, then `python scripts/sdfbase/make_full22_configs.py --derived-pins-from-registry` "
                   "writes the pin from its registry row (commit the pins first)" if o["source"] == "arditi_bank" else "register the composed object and pin its id first")
            raise SystemExit(f"{key}: derived object {cell} ({staged_id}) has no pin in {pins_path} — a derived object is staged by its REGISTRY id (the pin), "
                             f"never discovered from a directory's object.pt (s3r-fix, D1297 P1-2); {how}")
        if pin:
            row["object_sha256_expected"] = str(pin); row["object_sha256_expected_source"] = f"{pins_path}:{key}.{staged_id}"
        out.append(row)
print(json.dumps(out, indent=2))
PYEOF
}

if want stage; then
  # a refusal inside cells_json (a derived object without its pin, a table row without its kind directory) is the stage's refusal: before this
  # line the SystemExit left CELLS empty and the next command tracebacked on `json.loads('')` (demo-import blocker 1b) — the refusal text is above
  CELLS="$(cells_json)" || repro_fatal "05: cells_json refused (above)"
  n="$("$PY" -c 'import json,sys; print(len(json.loads(sys.argv[1])))' "$CELLS")" || repro_fatal "05: cells_json printed no JSON list (above)"
  if [ "$n" = 0 ]; then
    repro_say "stage: the requested views ($VIEWS) name no fitted object — nothing to stage (an unsteered-first u0 view is none + the two prompted SIs)"
  else
    repro_mkdir "$BANK"
    CELLS_FILE="$STATE/05_cells.json"; repro_write "$CELLS_FILE" "$CELLS"
    repro_say "stage: $n object(s) -> bank $BANK -> clusters $CLUSTERS"
    if [ "$REPRO_DRY_RUN" = 1 ]; then
      echo "DRY-RUN: $PY $REPRO_REPO/scripts/sdfbase/full22/stage_objects.py --site-dir $SISTEER_SITE_DIR --bank $BANK --target $K --cells $STATE/05_cells.json --clusters $CLUSTERS"
      "$PY" "$REPRO_REPO/scripts/sdfbase/full22/stage_objects.py" --site-dir "$SISTEER_SITE_DIR" --bank "$BANK" --target "$K" --cells <(printf '%s' "$CELLS") --clusters "$CLUSTERS" --dry-run
    else
      "$PY" "$REPRO_REPO/scripts/sdfbase/full22/stage_objects.py" --site-dir "$SISTEER_SITE_DIR" --bank "$BANK" --target "$K" --cells "$CELLS_FILE" --clusters "$CLUSTERS" || repro_fatal "staging did not verify on every cluster (records under $BANK/_records)"
    fi
  fi
fi

MANIFEST="$BANK/staging_manifest.json"
committed_manifest(){   # exit 0 iff the staging manifest holds rows for $K and EVERY one was staged from a committed: source (the stager's source_path)
  [ -f "$MANIFEST" ] || return 1
  "$PY" - "$MANIFEST" "$K" <<'PYEOF'
import json, sys
rows = [r for r in json.load(open(sys.argv[1], encoding="utf-8")) if str(r.get("target")) == sys.argv[2]]
sys.exit(0 if rows and all(str(r.get("source_path") or "").startswith("committed:") for r in rows) else 1)
PYEOF
}
if want pins; then
  # the pins are held to the SITE's registry (the root 04 registers into and `import_objects.py registry` imports into), named explicitly on
  # both writers below — never a writer's own default root (a committed block was once written with no registry read at all)
  REG_ROOT="$(repro_registry_root)"
  # … and to the SITE's campaign — the one 04 registered every fit under, by 04's own line (its CAMPAIGN=: the site's SISTEER_REGISTRY_CAMPAIGN, else
  # REGISTRY_CAMPAIGN, else sdfbase_full22), named explicitly on the generator as the root is (the writer read sdfbase_full22
  # alone, so a site naming another campaign fitted and registered its refusal direction, got no pin, and this stage refused the completed fit). The
  # committed branch's importer needs none: a committed row is held to its digest's own row, never to a campaign lookup (Q12).
  CAMPAIGN="${SISTEER_REGISTRY_CAMPAIGN:-${REGISTRY_CAMPAIGN:-sdfbase_full22}}"
  if [ "$FROM_COMMITTED" = 1 ] && { committed_manifest || { [ "$REPRO_DRY_RUN" = 1 ] && [ ! -f "$MANIFEST" ]; }; }; then
    # the committed objects' pins (demo-import M1): the model's block through import_objects.py pins, which is the ONE registry-backed pins writer
    # (make_full22_configs.pins_from_staging_manifest) scoped to the model — the stager's manifest rows held to the registry's rows for their
    # digests (reference|approved; a difference's parents; a copy's dose and parent; a committed entry's identity), every committed object pinned
    # to its id or the write refused by name ("import the committed objects first"); every other key of the pins file byte-identical
    repro_run "$PY" "$IMPORTER" pins --model "$K" --table "$(repro_models_table)" --manifest "$MANIFEST" --registry-root "$REG_ROOT" \
      --pins-file "${SISTEER_PINS_FILE:-$REPRO_REPO/configs/sdfbase_full22_pins.json}" \
      || repro_fatal "import_objects.py pins refused (above) — the pins file is unchanged"
  elif [ -f "$MANIFEST" ] || [ "$REPRO_DRY_RUN" = 1 ]; then
    repro_run "$PY" "$REPRO_REPO/scripts/sdfbase/make_full22_configs.py" --pins-from "$MANIFEST" --registry-root "$REG_ROOT" --campaign "$CAMPAIGN" \
      || repro_fatal "the pins writer refused (its output above names the entry and the reason) — configs/sdfbase_full22_pins.json unchanged (S2RLFIX2, D1980 (3))"
  else
    repro_note "no staging manifest at $MANIFEST yet (no object staged) — pins unchanged"
  fi
fi

if want roster; then
  ROSTER="$REPRO_REPO/configs/sdfbase_full22_roster.yaml"
  # the roster loads through f22_roster.load_roster — the ONE loader, which REFUSES a duplicate mapping key by name and line (PyYAML's
  # safe_load keeps the last silently); the block below is printed by lib/models.py roster-block, whose cell ids are each object's OWN
  # (the fix-1 owner follow-up 3: the former `<corpus>__<arm>` heuristic rendered refusal_direction under control_relu's key)
  "$PY" - "$REPRO_REPO/scripts/sdfbase/full22" "$ROSTER" "$K" "$VIEWS" <<'PYEOF'
import sys
sys.path.insert(0, sys.argv[1]); from f22_roster import load_roster
roster, key, views = load_roster(sys.argv[2]), sys.argv[3], sys.argv[4].split(",")
t = (roster.get("targets") or {}).get(key)
missing = [v for v in views if not t or v not in (t.get("views") or {})]
sys.exit(0 if not missing else 1)
PYEOF
  rc=$?
  if [ "$rc" = 0 ]; then repro_say "roster: $ROSTER carries $K with views $VIEWS"
  else
    [ "$rc" = 1 ] || repro_fatal "roster: $ROSTER did not load (a duplicate mapping key, or not a roster — above)"
    repro_say "roster: $ROSTER lacks $K / some of $VIEWS — the COMPLETE block the run's roster needs (paste under targets:, then re-run --step configs):"
    "$PY" "$HERE/lib/models.py" --table "$(repro_models_table)" roster-block "$K" "$VIEWS" || repro_fatal "lib/models.py roster-block refused (above)"
    [ "$REPRO_DRY_RUN" = 1 ] || repro_fatal "the roster does not carry $K's phase views yet — the run's roster owner adds the block above, then \`05_stage_roster.sh --step configs\`"
  fi
fi

if want configs; then
  repro_run "$PY" "$REPRO_REPO/scripts/sdfbase/make_full22_configs.py"
  repro_run "$PY" "$REPRO_REPO/scripts/sdfbase/make_full22_configs.py" --check
  repro_run "$PY" "$REPRO_REPO/scripts/sdfbase/full22/static_padding/make_padding_configs.py"
  repro_run "$PY" "$REPRO_REPO/scripts/sdfbase/full22/misalign/make_misalign_manifest.py"
  # the SAMPLED configs (`sdfbase_full22_<TAG>_sampled_t<T>[_cap<rung>]`): the statics AND the controls of the battery run in
  # that form (grade_static_sampled.py admits only sampled private runs); the generator's SAMPLED_VIEWS
  # table must carry the target's phase views (its owner's edit) — `--check` refuses a drifted committed output
  repro_run "$PY" "$REPRO_REPO/scripts/sdfbase/full22/sampled/make_sampled_configs.py"
  repro_run "$PY" "$REPRO_REPO/scripts/sdfbase/full22/sampled/make_sampled_configs.py" --check
  # the front door's GSM8K matrix `frontdoor_<key>` + its TAG overlays (g8k_cells.front_door_matrices; make_gsm8k_matrix.front_door_spec
  # derives the rows from THIS roster's block — the fix-1 owner follow-up 4): rendered once the roster carries a phase view of the
  # target, the pending state before (the generator refuses by name while the roster names none; 05 treats that as the misalign skips)
  if "$PY" - "$REPRO_REPO/scripts/sdfbase/full22" "$REPRO_REPO/configs/sdfbase_full22_roster.yaml" "$K" "$VIEWS" <<'PYEOF'
import sys
sys.path.insert(0, sys.argv[1]); from f22_roster import load_roster
t = (load_roster(sys.argv[2]).get("targets") or {}).get(sys.argv[3]) or {}
sys.exit(0 if any(v in (t.get("views") or {}) for v in sys.argv[4].split(",")) else 1)
PYEOF
  then
    repro_run "$PY" "$REPRO_REPO/scripts/sdfbase/gsm8k/make_gsm8k_matrix.py" --matrix-name "frontdoor_$K"
    repro_run "$PY" "$REPRO_REPO/scripts/sdfbase/gsm8k/make_gsm8k_matrix.py" --matrix-name "frontdoor_$K" --check
    repro_run "$PY" "$REPRO_REPO/scripts/sdfbase/gsm8k/make_gsm8k_overlays.py" --matrix "frontdoor_$K"
    repro_run "$PY" "$REPRO_REPO/scripts/sdfbase/gsm8k/make_gsm8k_overlays.py" --matrix "frontdoor_$K" --check
  else
    repro_note "configs: the GSM8K matrix frontdoor_$K + overlays are not generated yet (the roster must carry a phase view of $K; make_gsm8k_matrix.py --matrix-name frontdoor_$K)"
  fi
  for v in ${VIEWS//,/ }; do
    cfg="$REPRO_REPO/configs/sdfbase_full22_${K}$( [ "$v" = main ] || echo "_$v" ).yaml"
    [ -f "$cfg" ] && repro_say "configs: $cfg" || repro_note "configs: $cfg not generated yet (the roster must carry the view; --dry-run continues)"
  done
fi

if want bank-views; then
  # the bank view is built INSIDE the mirrored checkout at $SHA, so that commit must CONTAIN the generated configs: a clean tree,
  # every view's config tracked in $SHA, and the sha mirrored on the cluster before the ssh.
  # Sequence: --step configs, `git commit` the generated configs/manifests, then --step bank-views,freeze at the new HEAD.
  repro_require_clean_tree
  for v in ${VIEWS//,/ }; do
    cfg="configs/sdfbase_full22_${K}$( [ "$v" = main ] || echo "_$v" ).yaml"
    if ! git -C "$REPRO_REPO" cat-file -e "$SHA:$cfg" 2>/dev/null; then
      if [ "$REPRO_DRY_RUN" = 1 ]; then repro_note "bank-views: $cfg is not in commit $SHA — generate it (--step configs), COMMIT, then --step bank-views; the dry run continues"
      else repro_fatal "bank-views: $cfg is not in commit $SHA — the bank view is built from the mirrored checkout, so commit the generated configs first (--step configs, git commit, then --step bank-views)"; fi
    fi
  done
  for l in ${CLUSTERS//,/ }; do
    cluster_require "$l"; envs="$(cluster_var "$l" CLUSTER_ENVS_DIR)"; co="$(cluster_var "$l" CLUSTER_EVAL_REPO_ROOT)/$SHA"
    "$HERE/lib/mirror_repo.sh" --cluster "$l" --sha "$SHA" --config "$SISTEER_CONFIG" $(repro_dry_flag) || repro_fatal "bank-views: commit $SHA is not mirrored/verified on $l (lib/mirror_repo.sh refused above)"
    for v in ${VIEWS//,/ }; do
      repro_ssh "$l" "cd $co && $envs/vllm-attacker/bin/python scripts/sdfbase/full22/make_bank_view.py --target $K --view $v --roster configs/sdfbase_full22_roster.yaml && $envs/vllm-attacker/bin/python scripts/sdfbase/full22/make_bank_view.py --check --target $K --view $v --roster configs/sdfbase_full22_roster.yaml" \
        || repro_fatal "bank view $K/$v does not verify on $l"
      repro_mark_done "$K" 05_stage_roster "bankview.$l.$v"
    done
  done
fi

if want freeze; then
  # the frozen code_sha is the sha the u0 wave must run at (06 refuses a wave whose sha differs) — so it is a COMMITTED, clean tree
  repro_require_clean_tree
  repro_mkdir "$STATE/gate"
  # the inventory = lib/stage.py's unit plan for view u0 (the SAME rows 06 submits), every eval bound
  # to the instrument id the registry's own planner mints (never a hand-written name); `--dry-run` prints, never writes
  "$PY" "$HERE/lib/stage.py" freeze-inventory --model "$K" --roster "$REPRO_REPO/configs/sdfbase_full22_roster.yaml" \
    --models-table "$(repro_models_table)" --code-sha "$SHA" --out "$STATE/gate/expected_units.json" ${FILLER:+--filler "$FILLER"} $(repro_dry_flag) \
    --clusters "$(cluster_labels | tr ' ' ',')" ${REPRO_OVERRIDES:+--overrides "$REPRO_OVERRIDES"} \
    || repro_fatal "freeze-inventory refused (above)"
  [ -n "$FILLER" ] || repro_note "freeze: no --filler / SISTEER_PG19_FILLER — the five padding units carry instrument_id null until 06 --gate binds the registered id (jobplan padded-instrument-id)"
  if "$PY" "$HERE/lib/models.py" --table "$(repro_models_table)" single-phase "$K"; then
    repro_say "freeze: single-phase recipe — the inventory is the one wave's ($("$PY" "$HERE/lib/models.py" --table "$(repro_models_table)" baseline-view "$K")): its unsteered legs run inside it, so no steered wave waits on a gate; 06 --gate records the wave's own sanity verdicts, and the dose views are admitted behind it (lib/admission.py; D15)"
  else
    repro_say "freeze: the u0 inventory is the GLOBAL gate every steered wave passes (lib/admission.py; D15)"
  fi
  if [ "$FROM_COMMITTED" = 1 ]; then
    # demo-import M6: a row that imports its objects (committed_objects_of) never runs stage 04, so its compute plan is priced SERVE-ONLY — the
    # fit-side rows imported, the two serving rows from the serving probe's certified verdict (03_hpo.sh --phase probe: under the paper's
    # protocol it certifies the one declared pair). 06 (every steered view; the single-phase recipe's one wave included) and 09 refuse an
    # incomplete plan, so this is the next command once the probe is DONE.
    repro_note "freeze: this row imports its objects (committed_objects_of=$COMMITTED_OF; stage 04 is skipped), so price its compute plan SERVE-ONLY once 03_hpo.sh --model $K --object-class lora --phase probe is DONE: python reproduce/tools/replan_compute.py --model $K --serve-only --probe $STATE/03_hpo/lora/probe_verdict.json --fits-config configs/sdfbase_${FIT_ID}_fits.yaml   (06 and 09 refuse until the plan is complete; the probe's verdict prices the serving rows, nothing else does)"
  fi
fi
repro_say "05 done"
