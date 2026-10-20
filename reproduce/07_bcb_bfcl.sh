#!/bin/bash
# reproduce/07_bcb_bfcl.sh — stage 07: the FULL BigCodeBench and BFCL sets on Modal (the paper's item sets — 1,140 BigCodeBench ids, Hard
# included; 3,981 single-turn BFCL ids — under the paper's `protocol` decoding; the paper's sets, never a reduced subset) — the ONLY stage
# off Slurm. Runs LAST, on the shared-standup harness once its landing is on main (else the current canonical path, recorded).
#
#   reproduce/07_bcb_bfcl.sh --model K [--view <phase view>] [--cells <comma list>|all] [--phase stage|preflight|run|receipt|mirror|file|all]
#                            [--expect-hashes F] [--config F] [--overrides F] [--dry-run] [--help]
#
# The baseline pass (the model's baseline wave — u0 when declared; the single-phase recipe's one wave otherwise, whose unsteered legs
# run inside it) runs first and is this stage's OWN gate (BCB/BFCL gate only their own steered runs): another view — a DOSE view —
# refuses until <state>/gate/bcb_bfcl/<baseline view>.json records a pass. Calls scripts/modal/tinycap_served.py
# BY CLI ONLY, every call as `TINYCAP_APP_NAME=<app> modal run … --app-name <app>`: the launcher
# names its Modal App at IMPORT from that environment variable (no default) and holds --app-name equal to it, so a run without the
# variable refuses before any evaluation. `--phase stage`: `--sha <sha> --upload-tree --stage`. `--phase run`: `--pf --pass1
# --controls bigcodebench_full,bfcl_full --sampling protocol --model <slug> --cells … --modes off --serving-path lora_native` (the
# launcher's own control tokens, contracts.CONTROL_BIGCODEBENCH_FULL / CONTROL_BFCL_FULL through contracts.expand_control — the paper's
# full-set launches, launch_bcbw_full.sh over scripts/bcb_bfcl_200_widen_launch.py, composed exactly these beside the
# roster snapshot; the full legs' harness limits — the 8 h driver bound, the 10 h function, the sandbox pool — are the launcher's own
# contracts.FULL_TASK_LIMITS, picked by the configuration, never a flag; `--sampling protocol` is each benchmark's reference decoding
# (BigCodeBench greedy, BFCL T 0.001 under the pinned clamp override), the mode every paper row ran — the launcher's default `card` would
# sample Qwen at T 0.7; --pass1 is what makes the launcher PLAN a configuration per cell — without a pass it plans none and returns; --pf is
# the CPU preflight every fitted cell launches behind, and the one a protocol configuration launches behind at all; --modes off
# because BCB/BFCL run thinking-OFF and the launcher's default `off,on` refuses a row with no verified reasoning parser; --allow-extra rides
# along only when the view's cells exceed contracts.MAX_GPU_CONTAINERS, the launcher's cost trip-wire, and the stage says so).
# `--phase preflight`: the same argv with --pf alone — the launcher's preflight-only invocation, no GPU, whose report derives every fitted
# object's HF post-load intervention hash. `--cells` are the view's cells by their models.yaml NAMES (control_lora, difference_relu, …);
# this stage translates each to the launcher's `--cells` token through lib/models.py launcher-cells (roster_cell_id: `none` and
# `prompted__*` unchanged, an object cell its `objects` entry's own cell_id, the arditi rung stripped — the one cell grammar the wave roster
# and the launcher share, every token checked against contracts.parse_launcher_cell) and prints the mapping; a cell the mapping cannot
# express — not a cell of the view, no `objects` entry, a cell_id outside the launcher's grammar — refuses BY NAME before any Modal call.
# THE SLUG the launcher is handed as `--model` is lib/models.py launcher-slug: the KEY for the baseline wave, `<key>-<view>` for a DOSE
# view — the launcher serves ONE bank and ONE explicit-cells config per registry row (contracts.MODELS[<slug>]) and a dose view's object cell
# carries the SAME cell id as the wave's parent, so a dose view is served the way the paper served its dose-scaled rows on Modal (a row per
# dose view, `qwen35_35b_a3b-f22d0p75_relul` / `-f22d1p25`, over a bank of the view's copies keyed by cell id): its row's bank is
# `<the key's bank>/<view>/`, its config frontdoor_config_path(key, view) (scripts/bcbw_configs.py --front-door <key> --view <view>), and
# `--phase stage` stages the view's COMMITTED dose copies there, and every per-execution state file of a dose view (the captured logs, the
# hashes, the identity, the store and receipt, the spec, its own gate verdict) lands under <state>/07_views/<view>/ — the wave's stay where
# they were. The slug must be a row of contracts.MODELS (contracts.model_row): this stage resolves it BEFORE the secrets and the Modal
# calls and prints the row; a slug with no row refuses by name (the dry run prints the refusal a real run would make and exits 0). Every
# FITTED cell needs an expected intervention hash (the preflight verifies the staged object against a stated identity or refuses it):
# contracts.EXPECTED_INTERVENTION_HASHES by slug and token, or `--expect-hashes F`, a JSON object {token: hash} the launcher's
# flag of the same name takes (a file, named from the CALLER's directory like any file argument and made absolute before anything
# reads it: every Modal command below opens with `cd <repo>`, and the launcher opens the path from there, so a relative path handed
# on as spelled would pass this stage's checks and be refused by the launcher — after the stage phase had run; an explicit entry
# wins over the table). The stage resolves each fitted cell's source (table / file / none) BEFORE any Modal call and prints it; a
# run phase with a cell at `none` refuses by name (the dry run prints the refusal a real run would make) — `--phase preflight` is
# how the hashes of a freshly fitted bank are derived. The Modal token
# comes from lib/secrets.sh (env mode: MODAL_TOKEN_ID / MODAL_TOKEN_SECRET; op mode: the shell's secret-manager CLI). Nothing a run on Modal
# emits reaches the registry AS RUN ROWS (the volume spool and its pull timer left with the Modal transport):
# the verdict is ruled by lib/stage.py verdict and FILED as an analysis (below).
# `--phase stage` first stages the view's BANK from the paper's committed objects (reproduce/tools/stage_demo_bank_modal.py:
# <the slug's objects_dir>/<token>/{object.pt, sidecar} — the wave's objects, or a dose view's committed dose-scaled copies, the same bytes
# stage 09 imports — + the derived cells' parents under /vol/out/; receipts under <state>/07_bank/<view>/; `--go` on a real run). `--phase
# preflight` CAPTURES the launcher's stdout (`tee` to <state>/07_preflight_report.log), extracts its JSON report to <state>/07_preflight_report.json
# and reads every fitted cell's derived hash off checks[object_identity_and_header_guard] — the cells[<token>].intervention_hash map when the check
# passed, else the derived hashes its err string names (the bootstrap preflight exits non-zero by design: a fitted cell with no expectation is
# refused WITH its derived hash) — into <state>/07_expect_hashes.derived.json, each cross-checked against the identity the same bytes are
# published under: on the WAVE the hash the same bytes carry under the paper's seed slugs (contracts.EXPECTED_INTERVENTION_HASHES, the seed
# from objects/DEMO.json), on a DOSE view — whose copies ran on the cluster in the paper, so no Modal hash is published for them — the live
# bank receipt <state>/07_bank/<view>/bank.json: the token's staged object_id IS the committed copy's registry id the importer plans
# (sha256 of the bytes, verified at the put and pulled back), every file verified. The cross-check lands beside the derived hashes
# (<state>/07_expect_hashes.cross_check.json). The RUNNABLE expectation <state>/07_expect_hashes.json is written ONLY when every fitted cell's
# derived hash IS the published one (`file (= <slug>)`, or `file (= committed <object id>, dose <d> copy)`): a derived hash that DIFFERS from
# the published one, a token under no listed slug (`absent`), or a dose copy with no verified bank receipt is a FINDING, never a pin — the
# phase refuses by name, pins nothing (a stale runnable file from an earlier preflight is removed, named) and the table gains no demo key; a
# run against such bytes needs an explicit identity ruling (--expect-hashes a file you pinned, recorded). The phase also refuses when a
# fitted cell's hash was NOT derived. `--phase run` defaults --expect-hashes to that state file (an explicit flag wins), WRITES THE LAUNCH'S
# IDENTITY to <state>/07_run_identity.json before the launcher runs (the tree, the slug, the cells by both names, the object id of every fitted
# cell at this tree from the importer's plan over the committed objects — a dose view's the copy's — the app, the volume and environment, the
# run-overrides document — OVERRIDE_ID, its sha256, its path — the controls, the sampling mode and the expect-hashes the run is held to) and
# captures the launcher's stdout to <state>/07_run_report.log (`set -o pipefail; … | tee …` INSIDE the bash -c string, so the refusal reads the
# launcher's exit, not tee's) — the RECs it prints are the completion evidence the next phases read. `--phase receipt`
# (reproduce/tools/stage07_receipt.py) VERIFIES that identity against its own call (another overrides document, HEAD, cells, volume or
# environment refuses by name — the receipt describes the launch, never the call that reads it) and copies it, establishes completion from
# those RECs (read once, as one snapshot; every REC under the protocol sampling mode) plus a Modal-volume receipt (the stores' done/summary
# markers at the full item counts, controls_summary.json, the attempt pointers and completion records under <out>/watch/, pulled through the
# volume CLI into a build directory that is published as <state>/07_store/ only when complete — an earlier receipt is kept under
# 07_store.archive/, never deleted before its replacement exists), DERIVES the headline (the none cell's inspect/bigcodebench_full pass rate,
# `acc`, cross-checked three ways) and PRINTS the stage.py verdict command with --headline and --run-ids filled in: --range is the one blank
# you fill. `--phase file` (reproduce/tools/stage07_file.py) verifies the ruled verdict against the receipt and files it with `registry
# analysis file` — substrate modal, the hashed verdict as the report, the receipt's exercised objects as inputs (the launch tree's, never this
# checkout's plan), the Modal app / call ids in extra, the name ending in the receipt's sha so a new execution is a new analysis and a retry a
# duplicate — and reads it back; every fact of the filing is the RECEIPT's, and this call's --cells / OVERRIDE_ID
# are cross-checked against it (a `--phase file` without the run's --cells refuses by name, naming the receipt's). Neither receipt nor file
# is in `all`: the receipt is read after the run, and the verdict needs your --range first.
# `--phase mirror` (reproduce/tools/stage07_mirror.py; after receipt, before file; not in `all` either) copies the WHOLE stores the receipt
# names off the volume (`modal volume get --force` per store root, read-only on the volume) into <state>/07_mirror.building/<receipt sha12>/,
# verifies them — every file the receipt hashed byte for byte, each credited store key's cells/<key>/manifest.json at the receipt's tree and
# cell, RESULT's last line result=COMPLETE, every size as `modal volume ls --json` renders it — writes MIRROR.sha256 + mirror.json, publishes
# <state>/07_mirror/<receipt sha12>/ by rename (a previous mirror is kept; that path is held PLAIN under the state root — no symlink at
# 07_mirror or at <receipt sha12>, the directory's realpath its literal path under the root's — before the pull, before the rename and on
# every local check the re-run and the file phase make, so a mirror moved aside with a link left at its path, or a link at 07_mirror, is
# REDIRECTED, refused by name, nothing followed or deleted), pushes it with rsync (no --delete) over the primary cluster's
# CLUSTER_SSH to <SISTEER_REPRODUCE_ROOT>/<model>/07_mirror/<receipt sha12>/ — after GUARDING that destination under the expanded reproduce
# root (every component between the root and <receipt sha12>, that directory included, tested for a symlink and the directory's realpath held
# to its literal path under the root's, before the mkdir and the rsync; a symlink there refuses by name, nothing written through it, nothing
# deleted) — and runs `sha256sum -c MIRROR.sha256` there behind the same guard. The mirror and the
# file phase FOLLOW THE VIEW'S STATE: <state> in this paragraph and the next is the view's directory — the model's state dir for the baseline
# wave, <state>/07_views/<view>/ for a dose view, the same directory the receipt, the run identity and the verdict of that view live in — and
# the cluster path mirrors it (<SISTEER_REPRODUCE_ROOT>/<model>/07_views/<view>/07_mirror/<receipt sha12>/ for a dose view), so the receipt a
# mirror reads, the 07_mirror/ and 07_mirror.building/ it writes, its cluster.json, the cluster copy and the filing pointers of two views never
# share a directory (the receipt sha12 already tells two executions apart; the per-view layout tells two views apart). cluster.json records
# the verified copy, `--phase file` REFUSES without it, re-verifies the cluster copy over ssh (`sha256sum -c` in the recorded path, the three
# records read back by digest; a missing, altered or unreachable copy refuses by name) and copies its facts into the analysis record's
# extra.mirror, whose read-back holds them (the local path, cluster, host, path, digests) and the row's identity (analysis_id, code_sha) to this filing's. A
# refusal leaves the .building directory for inspection; a re-run for the same receipt verifies the published copy (every file re-hashed,
# both records held to cluster.json's digests), re-verifies the cluster copy the same way and does nothing else — a re-run, or a `--phase
# file`, under a site whose primary cluster or reproduce root is not the one cluster.json names refuses by name (one cluster record per
# mirror). The cluster check reads back the copy's inventory of EVERY entry by type too (`find . -mindepth 1 -printf "%y %p\0"`: the
# regular files held to MIRROR.sha256 plus the records, the directories to mirror.json's `directories`, and no symlink, fifo, socket or
# device at all): an entry left in the destination before the push, or added after it, refuses by name and is never deleted — rsync adds
# and never removes.
# A verdict FILED BEFORE THE MIRROR PHASE existed (its row's extra has no `mirror`) is reported by name by `--phase file`, which prints the
# verified copy's pointer, writes it to <state>/07_filing_pointers/<analysis name>.json (one record per analysis, written once, never
# rewritten; a standing record that disagrees refuses by name), reads the row back in every other field and ends in a FINDING (rc 3), not
# a pass: the registry's analysis rows are facts the fold never rewrites, so the pointer cannot reach that row here, and a registry reader
# cannot locate the stores from the record until it gains extra.mirror through a registry mechanism — keep the printed pointer
# against the analysis_id (stage07_file.py header: A FILING THAT PREDATES THE MIRROR PHASE).
# A filing not yet folded is read back from its pending header the way a folded row is, so a pending record that is not this filing's
# (another copy's digest, another receipt) is a finding before the fold, not after it.
# Every volume command the bank stager, the receipt and the mirror issue is scoped to $SISTEER_MODAL_ENVIRONMENT (`--environment`), the
# environment the launcher runs under (`modal run -e`): a volume name is per environment.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/lib/common.sh"; source "$HERE/lib/secrets.sh"
usage(){ sed -n '2,116p' "$0"; repro_common_help; }
VIEW=""; CELLS=all; PHASE=all; EXPECT_FILE=""   # --view defaults to the model's baseline wave (lib/models.py baseline-view: u0, or the single-phase recipe's one wave)
repro_parse_common_args "$@"; set -- "${REPRO_ARGS[@]+"${REPRO_ARGS[@]}"}"
while [ $# -gt 0 ]; do
  case "$1" in
    --view) VIEW="$2"; shift 2;; --cells) CELLS="$2"; shift 2;; --phase) PHASE="$2"; shift 2;; --expect-hashes) EXPECT_FILE="$2"; shift 2;;
    *) echo "REFUSED: unknown argument $1" >&2; usage >&2; exit 2;;
  esac
done
[ "$REPRO_HELP" = 1 ] && { usage; exit 0; }
case "$PHASE" in stage|preflight|run|receipt|mirror|file|all) ;; *) repro_fatal "--phase must be stage|preflight|run|receipt|mirror|file|all (receipt, mirror and file are not in all)";; esac
repro_load_config; repro_require_model
K="$REPRO_MODEL"; STATE="$(repro_state_dir "$K")"; SHA="$(repro_head_sha)"; SHA7="${SHA:0:7}"
BASE_VIEW="$("$(repro_python)" "$HERE/lib/models.py" --table "$(repro_models_table)" baseline-view "$K")" || exit 2
[ -n "$VIEW" ] || VIEW="$BASE_VIEW"
repro_require_view "$K" "$VIEW"
# this stage's per-execution state (the captured logs, the derived and pinned hashes, the launch identity, the store and receipt, the spec, the
# mirror and its pointer records): the model's state dir for the baseline wave — every path as before — and <state>/07_views/<view>/ for a DOSE
# view, so a dose view's run never overwrites the wave's log, identity, receipt or mirror (its gate verdict lands at
# <state>/07_views/<view>/gate/bcb_bfcl/<view>.json; the wave's gate the dose views are admitted behind stays at <state>/gate/bcb_bfcl/<baseline
# view>.json; the bank receipts under 07_bank/<view>/). ONE relative path, VIEW_REL07 (empty on the wave, `07_views/<view>` on a dose view), names
# the view's directory on the grading host (STATE07) AND under the primary cluster's reproduce root (MIRROR_DEST, below), so the two copies of a
# dose view's stores sit beside its receipt on both hosts and no view's mirror, cluster copy or pointer record shares a directory with another's
VIEW_REL07=""; [ "$VIEW" = "$BASE_VIEW" ] || VIEW_REL07="07_views/$VIEW"
STATE07="$STATE${VIEW_REL07:+/$VIEW_REL07}"
[ "$CELLS" = all ] && CELLS="$(repro_model "$K" "views.$VIEW.cells" | tr ' ' ',')"
# the launcher's --cells tokens for the view's cells (lib/models.py launcher-cells, header): `<view cell>\t<token>` per line, or a
# refusal BY NAME of the cell the mapping cannot express (not of the view / no objects entry / outside the grammar); a dose view's object
# cell maps to its cell id — the wave's parent's token — and is served from the dose view's OWN registry row and bank (launcher-slug, below)
CELL_MAP="$("$(repro_python)" "$HERE/lib/models.py" --table "$(repro_models_table)" launcher-cells "$K" "$VIEW" "$CELLS")" || exit 2
LAUNCHER_CELLS="$(awk -F'\t' '{print $2}' <<< "$CELL_MAP" | paste -sd, -)"
while IFS=$'\t' read -r cell tok; do repro_say "launcher cell: $cell -> $tok"; done <<< "$CELL_MAP"
# the launcher's registry slug for the view (lib/models.py launcher-slug, header): the KEY for the baseline wave; `<key>-<view>` for a dose
# view, whose row serves the view's own bank of dose-scaled copies (`<the key's bank>/<view>/`) through its own explicit-cells config
SLUG="$("$(repro_python)" "$HERE/lib/models.py" --table "$(repro_models_table)" launcher-slug "$K" "$VIEW")" || exit 2
[ "$SLUG" = "$K" ] || repro_say "launcher slug: $SLUG (a dose view: its own registry row, bank /vol/objects/sdfbase/$K/$VIEW and config)"
# the paper's item sets and decoding, read off the launcher's contracts (ONE spelling): the two FULL control tokens with their frozen item
# counts, and the `protocol` sampling mode every paper row ran under (the launcher's default `card` would sample Qwen at T 0.7)
read -r CONTROLS SAMPLING N_ITEMS <<< "$(PYTHONDONTWRITEBYTECODE=1 "$(repro_python)" - "$REPRO_REPO" <<'PY'
import sys
sys.path.insert(0, sys.argv[1] + "/src")
from sisteer.capeval.tinycap import contracts as C
print(f"{C.CONTROL_BIGCODEBENCH_FULL},{C.CONTROL_BFCL_FULL} {C.SAMPLING_PROTOCOL} {C.BCB_FULL_N},{C.BFCL_FULL_N}")
PY
)" || repro_fatal "could not read the launcher's control grammar (src/sisteer/capeval/tinycap/contracts.py) with $(repro_python)"
if [ "$VIEW" != "$BASE_VIEW" ]; then
  g="$STATE/gate/bcb_bfcl/$BASE_VIEW.json"
  if [ "$REPRO_DRY_RUN" = 1 ]; then echo "DRY-RUN: require $g verdict=pass (the unsteered BCB/BFCL pass is this stage's own gate)"
  else [ -f "$g" ] && "$(repro_python)" -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get("verdict") == "pass" else 1)' "$g" || repro_fatal "the unsteered BCB/BFCL pass has no recorded pass at $g — run --view u0 first, then record its verdict (stage.py verdict --state-root $STATE --family bcb_bfcl/$BASE_VIEW …)"; fi
fi
MODAL="${SISTEER_MODAL_PYTHON}"; case "$MODAL" in *python*) MODAL="$MODAL -m modal";; esac
APP="tinycap-$SHA7-$SLUG"   # the slug: the key on the wave; `<key>-<view>` on a dose view, so two views never share an app name
# the run phase's DEFAULT expectation: the preflight phase's derived hashes (<state>/07_expect_hashes.json) when the flag is absent and the file
# exists — an explicit --expect-hashes wins; the preflight phase itself never defaults to it (it derives)
EXPECT_SOURCE="flag"
if [ -z "$EXPECT_FILE" ] && { [ "$PHASE" = run ] || [ "$PHASE" = all ] || [ "$PHASE" = receipt ] || [ "$PHASE" = file ]; } && [ -f "$STATE07/07_expect_hashes.json" ]; then
  EXPECT_FILE="$STATE07/07_expect_hashes.json"; EXPECT_SOURCE="state (the preflight phase's derived hashes)"
fi
# --expect-hashes F is read where the caller stands (this check, the plan's read below, and the launcher's own) and made absolute
# HERE, once: every Modal command below opens with `cd $REPRO_REPO`, so a relative path handed on as spelled would pass this check
# from the caller's directory and be refused by the launcher's file read from the repo root (`../hashes.json` from reproduce/)
if [ -n "$EXPECT_FILE" ]; then
  { [ -f "$EXPECT_FILE" ] && [ -r "$EXPECT_FILE" ]; } || repro_fatal "--expect-hashes $EXPECT_FILE is not a readable file (a JSON object {launcher token: HF post-load intervention hash})"
  EXPECT_FILE="$(cd "$(dirname "$EXPECT_FILE")" && pwd)/$(basename "$EXPECT_FILE")"
fi
# the row the launcher resolves (contracts.model_row), resolved HERE: an unregistered slug refuses by name before any Modal
# call instead of inside the container; the dry run prints the resolution (or the refusal a real run would make) and exits 0
ROW="$(PYTHONDONTWRITEBYTECODE=1 "$(repro_python)" - "$REPRO_REPO" "$SLUG" <<'PY'
import sys
sys.path.insert(0, sys.argv[1] + "/src")
from sisteer.capeval.tinycap import contracts as C
try:
    r = C.model_row(sys.argv[2])
except C.UnknownModel as e:
    print(f"UNREGISTERED: {e}")
    sys.exit(0)
print(f"{r.slug}: {r.hf} @ {r.rev[:8]} family={r.family} model_key={r.model_key} thinking={','.join(r.thinking_modes)} "
      f"objects_dir={r.objects_dir} petri_manifest={r.petri_manifest}")
PY
)" || repro_fatal "could not read the tinycap model registry (src/sisteer/capeval/tinycap/contracts.py) with $(repro_python)"
MODES=""; RUN_EXTRA=""; MISSING=""
case "$ROW" in
  UNREGISTERED:*)
    if [ "$REPRO_DRY_RUN" = 1 ]; then repro_note "tinycap row: $ROW — a real run refuses here"
    else repro_fatal "tinycap: $ROW — add the slug's row to contracts.MODELS (src/sisteer/capeval/tinycap/contracts.py) first"; fi;;
  *)
    repro_say "tinycap row $ROW"
    # what the launcher's plan needs beyond the tokens, resolved from the same contracts BEFORE any Modal call (header): the thinking
    # mode (BCB/BFCL run thinking-OFF; the row must serve it), the cost trip-wire (one GPU container per cell; above
    # contracts.MAX_GPU_CONTAINERS the launcher refuses unless --allow-extra is passed on purpose — this stage passes it and says
    # so), and every FITTED cell's expected intervention hash source: `file` (--expect-hashes names it), `table`
    # (contracts.EXPECTED_INTERVENTION_HASHES has it) or `none` (the preflight would refuse the cell — E40 H1)
    PLAN="$(PYTHONDONTWRITEBYTECODE=1 "$(repro_python)" - "$REPRO_REPO" "$SLUG" "$LAUNCHER_CELLS" "$EXPECT_FILE" <<'PY'
import json, sys
sys.path.insert(0, sys.argv[1] + "/src")
from sisteer.capeval.tinycap import contracts as C
slug, tokens, expect_file = sys.argv[2], [t for t in sys.argv[3].split(",") if t], sys.argv[4]
row = C.model_row(slug)
if C.THINKING_OFF not in row.thinking_modes:
    sys.exit(f"REFUSED: tinycap: row {slug} serves thinking modes {list(row.thinking_modes)} — BCB/BFCL run thinking-OFF ({C.THINKING_OFF!r})")
expect = {}
if expect_file:
    try:
        expect = json.load(open(expect_file, encoding="utf-8"))
    except ValueError as exc:
        sys.exit(f"REFUSED: --expect-hashes {expect_file}: not JSON: {exc}")
    if not isinstance(expect, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in expect.items()):
        sys.exit(f"REFUSED: --expect-hashes {expect_file}: expected a flat JSON object of {{launcher token: hash}} (the launcher's own grammar)")
print(f"modes={C.THINKING_OFF}")
print(f"allow_extra={1 if len(tokens) > C.MAX_GPU_CONTAINERS else 0} max_gpu_containers={C.MAX_GPU_CONTAINERS}")
for tok in tokens:
    if C.parse_launcher_cell(tok).kind != C.CELL_KIND_FITTED:
        continue                                   # base / prompted: no object, no identity to expect
    src = "file" if tok in expect else ("table" if C.expected_intervention_hash(slug, tok) else "none")
    print(f"expect\t{tok}\t{src}")
PY
)" || exit 2   # the refusal (REFUSED: …) is already on stderr
    MODES="$(sed -n 's/^modes=//p' <<< "$PLAN")"
    repro_say "launcher modes: $MODES (BCB/BFCL run thinking-OFF; the launcher's default off,on would plan an ON leg the row refuses)"
    if grep -q '^allow_extra=1' <<< "$PLAN"; then
      RUN_EXTRA="--allow-extra"
      repro_note "$(awk -F, '{print NF}' <<< "$LAUNCHER_CELLS") cells, one GPU container each, exceed contracts.MAX_GPU_CONTAINERS $(sed -n 's/^allow_extra=1 max_gpu_containers=//p' <<< "$PLAN") (the launcher's cost trip-wire): --allow-extra is passed on purpose, for the view's cells"
    fi
    while IFS=$'\t' read -r _e tok src; do
      [ "$_e" = expect ] || continue
      repro_say "expectation: $tok -> $src"
      [ "$src" = none ] && MISSING="${MISSING:+$MISSING,}$tok"
    done <<< "$PLAN"
    if [ -n "$MISSING" ] && { [ "$PHASE" = run ] || [ "$PHASE" = all ]; }; then
      why="no expected intervention hash for fitted cell(s) $MISSING of slug $SLUG — the harness verifies every fitted object against a stated HF post-load intervention hash before serving it (E40 H1): contracts.EXPECTED_INTERVENTION_HASHES has no entry and --expect-hashes names none. Derive them: $0 --model $K --phase preflight runs the harness's CPU preflight, whose report prints each fitted cell's derived hash and the --expect-hashes JSON to pin; write {token: hash} to a file and pass --expect-hashes FILE"
      if [ "$REPRO_DRY_RUN" = 1 ]; then repro_note "tinycap: $why — a real run refuses here"
      else repro_fatal "tinycap: $why"; fi
    fi;;
esac
EXPECT_FLAG=""; [ -z "$EXPECT_FILE" ] || EXPECT_FLAG="--expect-hashes $(printf '%q' "$EXPECT_FILE")"   # absolute (above), quoted for the bash -c string
[ "$REPRO_DRY_RUN" = 1 ] || [ "$PHASE" = file ] || secrets_modal_env || repro_fatal "no Modal credentials (docs/secrets.md)"   # the file phase reaches no volume
repro_require_clean_tree
repro_say "07 bcb_bfcl: model=$K view=$VIEW slug=$SLUG cells=$CELLS launcher_cells=$LAUNCHER_CELLS controls=$CONTROLS (n_items $N_ITEMS) sampling=$SAMPLING modes=${MODES:-?} expect_hashes=${EXPECT_FILE:-none} (${EXPECT_SOURCE}) extra=${RUN_EXTRA:-none} app=$APP sha=$SHA volume=$SISTEER_MODAL_VOLUME env=$SISTEER_MODAL_ENVIRONMENT overrides=$OVERRIDE_ID phase=$PHASE"
# every Modal call: the app name in the launcher's ENVIRONMENT (it builds its App at import; no default) AND on --app-name
MODAL_RUN="cd $REPRO_REPO && SISTEER_RUN_OVERRIDES=$SISTEER_RUN_OVERRIDES TINYCAP_APP_NAME=$APP $MODAL run -e $SISTEER_MODAL_ENVIRONMENT -n $APP scripts/modal/tinycap_served.py --sha $SHA --app-name $APP --model $SLUG"
RUN_ARGS="--cells $LAUNCHER_CELLS --modes ${MODES:-off} --controls $CONTROLS --sampling $SAMPLING --serving-path lora_native${EXPECT_FLAG:+ $EXPECT_FLAG}${RUN_EXTRA:+ $RUN_EXTRA}"
RUN_REPORT="$STATE07/07_run_report.log"; PF_LOG="$STATE07/07_preflight_report.log"; EXPECT_STATE="$STATE07/07_expect_hashes.json"
EXPECT_DERIVED="$STATE07/07_expect_hashes.derived.json"; EXPECT_CROSS="$STATE07/07_expect_hashes.cross_check.json"; RUN_IDENTITY="$STATE07/07_run_identity.json"
if [ "$PHASE" = stage ] || [ "$PHASE" = all ]; then
  # the view's BANK on the volume from the paper's committed objects: <the slug's objects_dir>/<token>/{object.pt, sidecar} — the
  # wave's objects, or a dose view's committed dose-scaled copies under `<the key's bank>/<view>/` — + the derived cells' parents under
  # /vol/out/<parent model_key>/cells_all/; receipts under <state>/07_bank/<view>/ (a dry run's under 07_bank/<view>/dry/)
  repro_run "$(repro_python)" "$HERE/tools/stage_demo_bank_modal.py" --model "$K" --view "$VIEW" --receipts-dir "$STATE/07_bank/$VIEW" --volume "$SISTEER_MODAL_VOLUME" --environment "$SISTEER_MODAL_ENVIRONMENT" --modal "$MODAL" --table "$(repro_models_table)" $( [ "$REPRO_DRY_RUN" = 1 ] || echo --go ) \
    || repro_fatal "the demo bank stager refused (above) — nothing staged on the volume"
  repro_run_sh "$MODAL_RUN --upload-tree --stage" || repro_fatal "tinycap: the stage phase ended non-zero (its output above names the step) — nothing evaluated"
fi
if [ "$PHASE" = preflight ]; then
  # --pf alone: the launcher plans what --pass1 would and runs its CPU preflight on Modal (no GPU container): the staged tree, the
  # bank by path, every fitted object's header guard and HF post-load intervention hash — a cell with no expectation is REPORTED
  # with its derived hash and refused, which is how a freshly fitted bank's hashes are obtained. The stdout is CAPTURED (tee) and parsed:
  # the bootstrap preflight exits non-zero by design, so the refusal below is the PARSE's — a fitted cell whose hash was not derived,
  # DIFFERS from the published one or has none (a derived hash is pinned as the run's expectation only when it IS the
  # published object's; otherwise the run would verify the staged bytes against themselves and file a result that reproduces nothing)
  if [ "$REPRO_DRY_RUN" = 1 ]; then
    repro_run_sh "set -o pipefail; $MODAL_RUN $RUN_ARGS --pf | tee $PF_LOG"
    echo "DRY-RUN: parse $PF_LOG -> $STATE07/07_preflight_report.json, $EXPECT_DERIVED (every derived hash: checks[object_identity_and_header_guard]: cells[<token>].intervention_hash, else the derived hashes its err string names; the cross-check at $EXPECT_CROSS: on the wave against contracts.EXPECTED_INTERVENTION_HASHES under the paper's seed slug (objects/DEMO.json seed), on a dose view against the live bank receipt $STATE/07_bank/$VIEW/bank.json — the token's staged object_id the committed copy's id, every file verified) and $EXPECT_STATE ONLY when every fitted cell reads file (= <slug>) or file (= committed <object id>, dose <d> copy); DIFFERS, absent or UNVERIFIED REFUSES — a finding, never a pin"
  else
    mkdir -p "$STATE07"
    repro_run_sh "set -o pipefail; $MODAL_RUN $RUN_ARGS --pf | tee $PF_LOG"; pf_rc=$?
    PYTHONDONTWRITEBYTECODE=1 "$(repro_python)" - "$REPRO_REPO" "$PF_LOG" "$STATE07/07_preflight_report.json" "$EXPECT_STATE" "$EXPECT_DERIVED" "$EXPECT_CROSS" "$LAUNCHER_CELLS" "$K" "$VIEW" "$(repro_models_table)" "$STATE/07_bank/$VIEW/bank.json" <<'PY' || repro_fatal "tinycap: the preflight's derived hashes are NOT the run's expectation (above): a fitted cell's hash was not derived, DIFFERS from the published one, has no published one, or (a dose view) its staged copy is not the committed one the live bank receipt verifies — nothing pinned at $EXPECT_STATE (the derived hashes stand at $EXPECT_DERIVED, the cross-check beside them); read $PF_LOG, and rule the identity explicitly before any run against these bytes"
import importlib.util, json, os, re, sys
from pathlib import Path
repo, log_p, report_p, expect_p, derived_p, cross_p, tokens_csv, key, view, table, bank_p = sys.argv[1:]
sys.path.insert(0, repo + "/src")
from sisteer.capeval.tinycap import contracts as C
spec = importlib.util.spec_from_file_location("_07_models", repo + "/reproduce/lib/models.py"); M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)
CHECK = "object_identity_and_header_guard"
text = open(log_p, encoding="utf-8").read().splitlines()
# the launcher prints its preflight report as ONE indented JSON object after a tag (`print("[preflight]", json.dumps(rep, indent=1))`): the
# block from a line ENDING in `{` (the tag before it) to the first bare `}` line after it (indent=1 puts every nested brace past column 0);
# the LAST such block that carries `checks` is the report (an earlier tagged object — a stage record — is not)
report = None
for i, ln in enumerate(text):
    if not ln.rstrip().endswith("{"):
        continue
    for j in range(i + 1, len(text)):
        if text[j] == "}":
            try:
                doc = json.loads("\n".join([ln[ln.rindex("{"):], *text[i + 1:j + 1]]))
            except ValueError:
                break
            if isinstance(doc, dict) and isinstance(doc.get("checks"), dict):
                report = doc
            break
if report is None:
    sys.exit(f"REFUSED: {log_p} holds no preflight report (a JSON object with `checks`) — the launcher did not reach its preflight")
json.dump(report, open(report_p, "w", encoding="utf-8"), indent=1, sort_keys=True)
chk = (report.get("checks") or {}).get(CHECK)
if not isinstance(chk, dict):
    sys.exit(f"REFUSED: the preflight report carries no checks[{CHECK}] (checks: {sorted(report.get('checks') or {})})")
derived = {}
for tok, rec in (chk.get("cells") or {}).items():
    if isinstance(rec, dict) and isinstance(rec.get("intervention_hash"), str):
        derived[tok] = rec["intervention_hash"]
err = str(chk.get("err") or "")
for m in re.finditer(r"(?P<cell>[A-Za-z0-9_\-]+): no expected HF post-load intervention hash[^;]*?the derived hash is '(?P<hash>[0-9a-f]{16})'", err):
    derived.setdefault(m.group("cell"), m.group("hash"))
for m in re.finditer(r"--expect-hashes '\{\"(?P<cell>[^\"]+)\": \"(?P<hash>[0-9a-f]{16})\"\}'", err):
    derived.setdefault(m.group("cell"), m.group("hash"))
for m in re.finditer(r"(?P<cell>[A-Za-z0-9_\-]+): derives intervention_hash '(?P<hash>[0-9a-f]{16})'", err):
    derived.setdefault(m.group("cell"), m.group("hash"))
tokens = [t for t in tokens_csv.split(",") if t]
fitted = [t for t in tokens if C.parse_launcher_cell(t).kind == C.CELL_KIND_FITTED]
missing = [t for t in fitted if t not in derived]
# the cross-check — the identity the same bytes are published under. On the WAVE: the hash the same bytes carry under the paper's seed slug
# (the DEMO entry's training seed; the refusal direction is one object on every seed row; a composed paired write lives under the `p` slug).
# On a DOSE view (the paper ran its 4B dose copies on the cluster: no Modal hash is published for these bytes): the LIVE bank receipt
# <state>/07_bank/<view>/bank.json the stage phase wrote — the token's staged object_id must be the committed copy's registry id the
# importer plans (sha256 of the bytes, verified at the put and pulled back) and every file verified; a dry receipt, a missing one or another
# id is UNVERIFIED. A disagreement either way is a finding, not a pin.
m = M.model(key, table)
demo = {e["paper_label"]: e for e in json.load(open(repo + "/objects/DEMO.json", encoding="utf-8"))["objects"]}
paper = M.committed_objects_of(m) or ""
by_tok = {M.roster_cell_id(m, c): c for c in ((m.get("views") or {}).get(view) or {}).get("cells", []) if c != "none" and not c.startswith(M.NO_OBJECT_PREFIX)}
is_dose = "dose" in ((m.get("views") or {}).get(view) or {})
bank, plan_rows, bank_problem = None, {}, None
if is_dose:
    spec2 = importlib.util.spec_from_file_location("_07_import_objects_pf", repo + "/reproduce/tools/import_objects.py"); IO = importlib.util.module_from_spec(spec2); spec2.loader.exec_module(IO)
    try:
        plan_rows = {r["view_cell"]: r for r in IO.Plan(key, views=[view], table=table, repo=Path(repo)).rows}
    except SystemExit as exc:
        sys.exit(f"REFUSED: the committed dose copies of {key}/{view} do not plan at this tree ({exc}) — the dose view's identity cannot be checked")
    try:
        bank = json.load(open(bank_p, encoding="utf-8"))
    except (OSError, ValueError) as exc:
        bank_problem = f"no live bank receipt at {bank_p} ({exc}) — run --phase stage for the view first"
    else:
        if not isinstance(bank, dict) or bank.get("dry") is not False or bank.get("step") != "bank" or bank.get("view") != view:
            bank_problem = f"{bank_p} is not the view's LIVE bank receipt (step {bank.get('step') if isinstance(bank, dict) else None!r}, dry {bank.get('dry') if isinstance(bank, dict) else None!r}, view {bank.get('view') if isinstance(bank, dict) else None!r})"
cross = {}
ok = {}
for tok in fitted:
    cell = by_tok.get(tok); o = M.object_of(m, cell) if cell else {}
    if is_dose:
        row = plan_rows.get(cell) or {}
        oid = str(row.get("object_sha256_expected") or "")
        entry = ((bank or {}).get("cells") or {}).get(tok) if bank_problem is None else None
        why = bank_problem
        if why is None and not isinstance(entry, dict):
            why = f"{bank_p} stages no {tok!r}"
        if why is None and str(entry.get("object_id")) != oid:
            why = f"{bank_p} stages {str(entry.get('object_id'))[:12]} as {tok}, not the committed copy {oid[:12]}"
        if why is None and any(str(f.get("verification", "")).startswith("not verified") for f in (entry.get("files") or [])):
            why = f"{bank_p}: a file of {tok} is not verified"
        cross[tok] = {"slug": None, "published": None, "committed": oid, "dose": row.get("dose"), "dose_parent": row.get("dose_parent"), "bank_receipt": bank_p,
                      "bank_verified": why is None, "derived": derived.get(tok)}
        if tok not in derived:
            print(f"expectation: {tok} -> NOT DERIVED (the preflight reported no hash for it)")
        elif why is not None:
            print(f"expectation: {tok} -> UNVERIFIED: derived {derived[tok]}, but {why} — the staged bytes are not shown to be the committed dose copy")
        else:
            print(f"expectation: {tok} -> file (= committed {oid[:12]}, dose {float(row['dose']):g} copy of {str(row.get('dose_parent'))[:12]}; the bank receipt verifies the bytes)")
        ok[tok] = tok in derived and why is None
        continue
    e = demo.get(str(o.get("paper_label") or cell)) if cell else None
    seed = (e or {}).get("seed") or 1
    slug = f"{paper}-f22s{seed}" + ("p" if str(o.get("object_kind") or "") == "composed" else "")
    published = C.expected_intervention_hash(slug, tok) if slug in C.EXPECTED_INTERVENTION_HASHES else None
    cross[tok] = {"slug": slug, "published": published, "derived": derived.get(tok)}
    if tok not in derived:
        print(f"expectation: {tok} -> NOT DERIVED (the preflight reported no hash for it)")
    elif published is None:
        print(f"expectation: {tok} -> absent (no published hash under {slug}; derived {derived[tok]} — a finding, not a pass)")
    elif published == derived[tok]:
        print(f"expectation: {tok} -> file (= {slug})")
    else:
        print(f"expectation: {tok} -> DIFFERS: derived {derived[tok]}, {slug} publishes {published} — the staged bytes are not the paper's object under that seed")
    ok[tok] = tok in derived and published == derived[tok]
json.dump({t: derived[t] for t in fitted if t in derived}, open(derived_p, "w", encoding="utf-8"), indent=1, sort_keys=True)
json.dump(cross, open(cross_p, "w", encoding="utf-8"), indent=1, sort_keys=True)
print(f"derived hashes: {derived_p} ({len(fitted) - len(missing)} of {len(fitted)} fitted cell(s) derived; the cross-check at {cross_p})")
# the RUNNABLE expectation: only when every fitted cell's derived hash IS the published one — under its seed slug on the wave, the committed
# copy's verified bytes on a dose view — a disagreement, an absent published hash or an unverified bank is a finding, never a pin (the run
# would otherwise verify the staged bytes against themselves)
def _why(t):
    if t in missing:
        return f"{t} NOT DERIVED"
    c = cross[t]
    if is_dose:
        return f"{t} UNVERIFIED (derived {derived[t]}; the bank receipt {bank_p} does not verify the committed copy {str(c['committed'])[:12]})"
    if c["published"] is None:
        return f"{t} absent (no published hash under {c['slug']}; derived {derived[t]})"
    return f"{t} DIFFERS (derived {derived[t]}, {c['slug']} publishes {c['published']})"
problems = [_why(t) for t in fitted if not ok.get(t)]
if problems:
    if os.path.exists(expect_p):
        os.unlink(expect_p)
        print(f"removed {expect_p}: an earlier preflight's pinned expectation is not this one's")
    sys.exit(f"REFUSED: the derived hashes are NOT pinned as the run's expectation — {'; '.join(problems)}: a hash that differs from the published one, has none, or "
             f"(a dose view) belongs to bytes the live bank receipt does not verify as the committed copy, is a finding, never a pin (read {cross_p}); a run against "
             f"these bytes needs an explicit identity ruling: record it, then pass --expect-hashes <a file you pinned from {derived_p}>")
json.dump({t: derived[t] for t in fitted}, open(expect_p, "w", encoding="utf-8"), indent=1, sort_keys=True)
print(f"expect hashes: {expect_p} ({len(fitted)} fitted cell(s), every one "
      + ("the committed dose copy's, its bytes verified by the bank receipt" if is_dose else "the published hash under its seed slug") + ") — the run phase reads it")
PY
    [ "$pf_rc" = 0 ] || repro_note "the preflight exited $pf_rc (a bootstrap preflight refuses a fitted cell with no expectation by design); every fitted cell's hash was derived and is the published one (the committed copy's on a dose view) — pinned at $EXPECT_STATE, which the run phase reads"
  fi
fi
if [ "$PHASE" = run ] || [ "$PHASE" = all ]; then
  # the launcher's stdout is CAPTURED: its RECs (one JSON line per container, printed after `[tinycap] N/N container(s) OK`) are the completion
  # evidence the receipt phase reads; pipefail INSIDE the string, since repro_run_sh spawns a fresh bash — the refusal reads the launcher's exit
  [ "$REPRO_DRY_RUN" = 1 ] || mkdir -p "$STATE07"
  # THE LAUNCH'S IDENTITY, written beside the captured stdout BEFORE the launcher runs: the tree (HEAD of the clean checkout
  # the launcher gets as --sha), the cells by both names, the object id of every fitted cell AT THIS TREE (the importer's plan over the committed
  # objects, by the table's cell name — a model that declares none records null and the filing refuses by name), the app, the volume and
  # environment, the run-overrides document (OVERRIDE_ID, its sha256, its path) and the expect-hashes the run is held to. `--phase receipt`
  # verifies it against its own call and copies it, so a later call under another overrides document or HEAD cannot relabel the execution.
  PYTHONDONTWRITEBYTECODE=1 "$(repro_python)" - "$REPRO_REPO" "$K" "$VIEW" "$CELLS" "$LAUNCHER_CELLS" "$SHA" "$APP" "$SLUG" "$SISTEER_MODAL_VOLUME" "$SISTEER_MODAL_ENVIRONMENT" "$OVERRIDE_ID" "$SISTEER_RUN_OVERRIDES" "$SISTEER_RUN_OVERRIDES_SHA256" "$EXPECT_FILE" "${MODES:-off}" "${RUN_EXTRA:-}" "$(repro_models_table)" "$RUN_IDENTITY" "$REPRO_DRY_RUN" "$CONTROLS" "$SAMPLING" <<'PY' || repro_fatal "the launch's identity could not be written (above) — the run's objects or overrides document are not identified; nothing launched"
import hashlib, importlib.util, json, sys
from pathlib import Path
repo, key, view, cells_csv, tokens_csv, sha, app, slug, volume, environment, oid, opath, osha, expect_file, modes, extra, table, out_p, dry, controls, sampling = sys.argv[1:]
sys.path.insert(0, repo + "/src")
from sisteer.capeval.tinycap import contracts as C
spec = importlib.util.spec_from_file_location("_07_import_objects", repo + "/reproduce/tools/import_objects.py"); IO = importlib.util.module_from_spec(spec); spec.loader.exec_module(IO)
M = IO._load_models_module(Path(repo))
cells, tokens = [c for c in cells_csv.split(",") if c], [t for t in tokens_csv.split(",") if t]
if len(cells) != len(tokens):
    sys.exit(f"REFUSED: the view's cells {cells} and the launcher's tokens {tokens} are not one mapping")
m = M.model(key, table)
fitted = [(c, t) for c, t in zip(cells, tokens) if C.parse_launcher_cell(t).kind == C.CELL_KIND_FITTED]
declared = M.committed_objects_of(m) is not None or any(M.committed_objects_of(m, c) for c in (m.get("objects") or {}))
objects = None
if declared:
    try:
        plan = IO.Plan(key, views=[view], table=table, repo=Path(repo))
    except SystemExit as exc:
        sys.exit(f"REFUSED: the committed objects of {key}/{view} do not plan at this tree ({exc}) — the run's objects cannot be identified")
    by_cell = {r["view_cell"]: r["object_sha256_expected"] for r in plan.rows}
    unmapped = [c for c, _t in fitted if c not in by_cell]
    if unmapped:
        sys.exit(f"REFUSED: fitted cell(s) {unmapped} of {key}/{view} map to no committed object in the importer's plan (rows: {sorted(by_cell)})")
    objects = {t: by_cell[c] for c, t in fitted}
elif fitted:
    print(f"note: {key} declares no committed objects: the launch's identity records no object ids for {[t for _c, t in fitted]} (the filing refuses by name)")
expect = None
if expect_file:
    data = open(expect_file, "rb").read()
    expect = {"path": expect_file, "sha256": hashlib.sha256(data).hexdigest(), "hashes": json.loads(data)}
doc = {"schema": "stage07-run-identity/1", "tree_sha": sha, "model_key": key, "slug": slug, "view": view, "cells": cells, "launcher_cells": tokens, "objects": objects,
       "app": app, "volume": volume, "modal_environment": environment or None, "run_overrides": {"id": oid, "sha256": osha, "path": opath}, "expect_hashes": expect,
       "controls": controls, "sampling": sampling, "modes": modes, "serving_path": "lora_native", "allow_extra": extra == "--allow-extra", "written_by": "07_bcb_bfcl.sh --phase run"}
summary = (f"tree {sha[:7]}, slug {slug}, {len(tokens)} cell(s), objects for {len(objects or {})} of {len(fitted)} fitted, controls {controls} under {sampling} sampling, "
           f"overrides {oid} ({osha[:12]}), expect-hashes {expect['sha256'][:12] if expect else 'none'}")
if dry == "1":
    print(f"DRY-RUN: write {out_p} (the launch's identity: {summary}) — the receipt verifies and copies it")
else:
    Path(out_p).write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"run identity: {out_p} ({summary})")
PY
  repro_run_sh "set -o pipefail; $MODAL_RUN $RUN_ARGS --pf --pass1 | tee $RUN_REPORT" || repro_fatal "tinycap: the run ended non-zero (the launcher's exit code is its containers' verdict; its output above and $RUN_REPORT name the refusal or the failed cell) — record no verdict"
  repro_note "registry: nothing a run on Modal emits reaches the registry as RUN rows (the volume spool and its pull timer left with the Modal transport, registry-slurm-only C10); the verdict is ruled from the receipt and filed as an analysis (Q13)"
  repro_say "followup: $0 --model $K --view $VIEW --cells $CELLS --phase receipt — pulls the stores' markers, controls_summary, attempt pointers and completion records from the volume into $STATE07/07_store/, writes receipt.json and prints the stage.py verdict command with --headline (the none cell's inspect/bigcodebench_full acc: its pass rate on the full 1,140-item set) and --run-ids filled in; rule it with your --range, then $0 --model $K --view $VIEW --cells $CELLS --phase file"
fi
if [ "$PHASE" = receipt ]; then
  # this call's resolution (the tree, the app, the cells by both names in the mapping's order, OVERRIDE_ID, the volume and environment) rides
  # along as what the receipt VERIFIES against the launch's identity ($RUN_IDENTITY, written by --phase run) before copying that identity;
  # a disagreement refuses by name — the receipt describes the launch, never the call that reads it
  RECEIPT_ARGS=(--state-root "$STATE07" --view "$VIEW" --sha "$SHA" --app "$APP" --cells "$LAUNCHER_CELLS" --view-cells "$CELLS" --override-id "$OVERRIDE_ID" --controls "$CONTROLS"
                --volume "$SISTEER_MODAL_VOLUME" --environment "$SISTEER_MODAL_ENVIRONMENT" --modal "$MODAL" --run-report "$RUN_REPORT" --run-identity "$RUN_IDENTITY")
  if [ "$REPRO_DRY_RUN" = 1 ]; then
    repro_run "$(repro_python)" "$HERE/tools/stage07_receipt.py" "${RECEIPT_ARGS[@]}"
    echo "DRY-RUN: $(repro_python) $HERE/lib/stage.py verdict --state-root $STATE07 --family bcb_bfcl/$VIEW --headline <the receipt's none inspect/bigcodebench_full acc> --range <lo,hi> --run-ids <the receipt's store keys> --note \"headline: none inspect/bigcodebench_full acc; receipt <sha12>\"   (the receipt phase prints it filled in; --range is the one blank)"
  else
    "$(repro_python)" "$HERE/tools/stage07_receipt.py" "${RECEIPT_ARGS[@]}" || repro_fatal "the receipt refused (above): no receipt, no verdict command — the run is not complete evidence yet"
    repro_say "followup: rule the verdict with the stage.py verdict command printed above (--range is yours; D15), then $0 --model $K --view $VIEW --cells $CELLS --phase file"
    repro_note "before the filing: $0 --model $K --view $VIEW --cells $CELLS --phase mirror copies the whole stores off the volume to this host and the primary cluster, sha-verified — the filing refuses without that copy"
  fi
fi
if [ "$PHASE" = mirror ] || [ "$PHASE" = file ]; then
  # the mirror's cluster target, shared by the two phases: SISTEER_PRIMARY_CLUSTER's CLUSTER_SSH and <expanded SISTEER_REPRODUCE_ROOT>/<model>/07_mirror/
  # (lib/common.sh repro_cluster_state_root) — <…>/<model>/07_views/<view>/07_mirror/ for a DOSE view (VIEW_REL07, the same relative path as the
  # grading host's STATE07) — under which the mirror lands <receipt sha12>/: the mirror pushes there; the file phase holds the mirror's cluster
  # record to it (a record naming another target refuses by name: the copy sits where the record says, not where this site points). Two views'
  # receipts already differ in sha12; the per-view directory keeps their copies apart by layout too, the cluster tree mirroring the state tree.
  # MIRROR_ROOT is the expanded SISTEER_REPRODUCE_ROOT itself: the mirror guards every component between it and <dest>/<receipt sha12> against a
  # symlink before the push and before every re-check, and records it; the file phase holds the record's root to it too
  MIRROR_CLUSTER="$SISTEER_PRIMARY_CLUSTER"; MIRROR_HOST="$(cluster_var "$MIRROR_CLUSTER" CLUSTER_SSH)"
  MIRROR_ROOT="$(repro_expand_shared "$MIRROR_CLUSTER" "$SISTEER_REPRODUCE_ROOT")"
  MIRROR_DEST="$(repro_cluster_state_root "$MIRROR_CLUSTER" "$K")${VIEW_REL07:+/$VIEW_REL07}/07_mirror"
fi
if [ "$PHASE" = mirror ]; then
  # the WHOLE stores the receipt names, off the volume to this host and the primary cluster, sha-verified (reproduce/tools/stage07_mirror.py, header):
  # the receipt alone names the stores (nothing here is a control or suite name); this call's volume / environment ride along as cross-checks the
  # tool refuses on. The state root is the VIEW's ($STATE07): the receipt it reads and the 07_mirror/ (+ .building/) it writes are the view's
  [ -n "$MIRROR_HOST" ] || repro_fatal "cluster $MIRROR_CLUSTER has no CLUSTER_SSH — the mirror has nowhere to push"
  MIRROR_ARGS=(--state-root "$STATE07" --volume "$SISTEER_MODAL_VOLUME" --environment "$SISTEER_MODAL_ENVIRONMENT" --modal "$MODAL"
               --cluster-root "$MIRROR_ROOT" --cluster-label "$MIRROR_CLUSTER" --ssh-host "$MIRROR_HOST" --cluster-dest "$MIRROR_DEST")
  if [ "$REPRO_DRY_RUN" = 1 ]; then
    # the dry run reads the receipt and prints the planned pulls, the rsync and the ssh check (DRY-RUN: lines); no volume access, no secrets, nothing written
    echo "DRY-RUN: $(repro_python) $HERE/tools/stage07_mirror.py ${MIRROR_ARGS[*]} --dry-run"
    "$(repro_python)" "$HERE/tools/stage07_mirror.py" "${MIRROR_ARGS[@]}" --dry-run || repro_fatal "the mirror phase refused (above) — nothing planned"
  else
    "$(repro_python)" "$HERE/tools/stage07_mirror.py" "${MIRROR_ARGS[@]}" || repro_fatal "the mirror refused (above): its refusal names what to fix; the filing refuses until a verified copy with its cluster record stands for this receipt"
    repro_say "followup: $0 --model $K --view $VIEW --cells $CELLS --phase file — files the verdict with the mirror's cluster path and MIRROR.sha256 digest in the analysis record"
  fi
fi
if [ "$PHASE" = file ]; then
  # the filing's facts are the RECEIPT's (cells, tokens, OVERRIDE_ID, app, tree); this call's resolution rides along as cross-checks the tool refuses on —
  # the site's mirror target among them (when the primary cluster has a CLUSTER_SSH; the mirror's own refusal covers a site with none). The state root
  # is the VIEW's ($STATE07): a dose view's receipt, verdict, mirror and pointer records all sit under <state>/07_views/<view>/, never beside the wave's
  FILE_ARGS=(--state-root "$STATE07" --view "$VIEW" --model "$K" --registry-root "$(repro_registry_root)" --override-id "$OVERRIDE_ID" --cells "$CELLS" --launcher-cells "$LAUNCHER_CELLS" --table "$(repro_models_table)")
  [ -z "$MIRROR_HOST" ] || FILE_ARGS+=(--cluster-root "$MIRROR_ROOT" --cluster-label "$MIRROR_CLUSTER" --ssh-host "$MIRROR_HOST" --cluster-dest "$MIRROR_DEST")
  if [ "$REPRO_DRY_RUN" = 1 ]; then
    # the dry run COMPOSES the spec (written to $STATE07/07_analysis_spec.json from the receipt and the verdict file) and prints the filing command; no registry is touched
    echo "DRY-RUN: $(repro_python) $HERE/tools/stage07_file.py ${FILE_ARGS[*]} --dry-run"
    "$(repro_python)" "$HERE/tools/stage07_file.py" "${FILE_ARGS[@]}" --dry-run || repro_fatal "the file phase refused (above) — nothing composed"
  else
    "$(repro_python)" "$HERE/tools/stage07_file.py" "${FILE_ARGS[@]}"; rc=$?
    [ "$rc" = 0 ] || repro_fatal "the file phase did not pass (rc $rc, above): a refusal (rc 2) names what to fix (a mirror missing, a cluster copy that is gone, altered, contaminated by entries outside the mirror — a file, directory, symlink or special file it never held — or unreachable); a finding (rc 3) means the registry's row for this name differs from this filing in a field the read-back checks (its identity — analysis_id, code_sha: another tree or input set under this name — its report sha, substrate, inputs, receipt, headline, or the extra.mirror it carries against the current verified copy: the local path, cluster, host, path, digests — a MOVED state root is such a disagreement), or that the filing was made BEFORE THE MIRROR PHASE: its verdict stands, its registry row carries no pointer to the verified copy and cannot gain one here, and the pointer is at $STATE07/07_filing_pointers/<analysis name>.json — record it in decisions.md against the analysis_id; the finding lifts when the row gains extra.mirror through a registry mechanism"
    repro_note "decisions.md: record the analysis name and analysis_id printed above with the receipt sha, the headline, the store cell keys, the Modal app name and the mirror's cluster path + MIRROR.sha256 digest"
  fi
fi
repro_say "07 done"
