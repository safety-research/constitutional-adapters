#!/bin/bash
# full22 SAMPLED re-run (lane E): mirror the sampled stores from the cluster into the grading host analysis tree, and make
# the flip-pass copies the pass-2 judge needs. Idempotent (re-run as stores land); prints what it did; --dry-run.
#
#   mirror_sampled.sh [--cluster LABEL | --host <ssh alias> --remote-root <dir>] [--remote-root <dir>]
#                     [--dest <dir>   (default $RUNSTATE/candidate-si-refit/run-state/full22/sampled, the site's run-state tree)]
#                     [--keys k1,k2,...] [--rungs 512,2048,8192,16384] [--configs-dir <worktree>/configs] [--no-flip] [--dry-run]
#   --keys        : the four main-view keys by default; every other TAG of make_sampled_configs.SAMPLED_VIEWS is accepted —
#                   the instruct-direct gemma4e4b_it / qwen35_4b_it (lane F) and the strength frontier's 24 per-dose TAGs
#                   <key>_d<dose> / <key>_it_d<dose> (2026-09-10) — a TAG's transcript roots sdfbase_full22_<TAG>_sampled_t*,
#                   controls roots <TAG>_t<T>, ladders and the 512-root completeness (the cell list of
#                   configs/sdfbase_full22_<TAG>_sampled_t<T>.yaml) all follow the TAG. The index matches a key or TAG only as
#                   the exact prefix `sdfbase_full22_<k>_sampled_t` / `<k>_t`, so a bare key never matches its views' roots and
#                   a view never matches another view's (`gemma4e4b_it` vs `gemma4e4b_it_d0p25`).
#   --rungs       : the controls cap ladder to mirror (default 512,2048,8192,16384 — the 16,384 ceiling, lane G2/G3); a rung
#                   root that does not exist on the cluster is simply absent from the index, never an error.
#   --cluster     : the cluster the sampled wave ran on, by its registry label (the default) or its site label. Its login
#                   host (CLUSTER_SSH) and shared root (CLUSTER_SHARED_ROOT, the remote root's default parent) come from the
#                   site config through reproduce/lib/clusters.py; a cluster the site config does not describe is refused.
#   --host        : the login host's ssh alias, given directly instead of through the site config; needs --remote-root.
#   --host local  : the "remote" root is a local directory (tests / a grading-host-local rehearsal); no ssh.
#   --configs-dir : where configs/sdfbase_full22_<key>_sampled_t<T>.yaml live — their explicit_cells define a COMPLETE 512 root.
#
# What lands under --dest (DESIGN.md "Stores": the analysis and grade outputs):
#   transcripts_sampled/<private_run>/            <- transcripts_results/sdfbase_full22_<key>_sampled_t*  (base + _cap8192 + _cap16384 roots)
#   controls_sampled/controls_cap<rung>/<TAG>_t<T>/ <- sampled/controls_cap<rung>/<TAG>_t<T>/           (every rung root incl. cap16384)
#   gsm8k_thinking_on_cap<cap>/<TAG>_t<T>/        <- sampled/gsm8k_thinking_on_cap<cap>/<TAG>_t<T>/   (the GSM8K thinking-ON readout roots,
#                                                    base cap 8192 and its 16384 rung; lane G2's fixed names)
#   mmlu/<TAG>_t<T>/                               <- sampled/mmlu/<TAG>_t<T>/                         (the MMLU letter-logprob roots)
#   ladders/<TAG>_t<T>/                            <- sampled/ladders/<TAG>_t<T>/  (ladder.json, static_ladder_t<T>.json, events.jsonl,
#                                                    and lane G2's decision_16384.json when written)
#   serving_records/<serving dir>/                 <- sampled/serving/<dir>/  PROVENANCE ONLY: *.json, *.jsonl, *.verdict, STOP
#                                                    (standup_jobs.jsonl, positive_control_*.json(.verdict), cells_served.json,
#                                                    endpoint.json, routing_table.json, *_attestation.json, ladder.json);
#                                                    never vllm/standup/driver logs, chat templates or the lens registry.
#   controls_sampled/flip_pass/cap512/<TAG>_t<T>/  = a SECOND copy of the 512 root taken straight from the cluster, made only
#                                                    while it is pristine (no judge files, no label rows) — pass 1 appends
#                                                    label rows to the mirror above, pass 2 re-judges THIS copy live.
#   controls_sampled/flip_pass/cap<R>/<TAG>_t<T>/cells/<store>/ = each (cell, CONTROL)'s FINAL rung store — one copy per DISTINCT final rung
#                                                    among the cell's controls (xstest may exit at 2048 while or_bench climbs to 8192: both stores
#                                                    are copied; the readers' one rule) (R = the highest
#                                                    rung the ladder says the cell reached — 2048, 8192 or 16384), copied from the cluster.
# A flip copy is made ONLY from a FINISHED source (the 512 root: every explicit cell of the sampled config present with a
# gen_stats.json; a rung store: its gen_stats.json) — an unfinished root is SKIPPED and retried on the next run, never frozen
# partial. Every fresh copy is verified pristine on arrival (a copy carrying judge_summary.json / judge_items.jsonl or rows.jsonl
# label rows is REFUSED, exit 2); an existing flip copy is never re-synced (pass 2 owns it) — it is re-verified (finished +
# pristine or pass-2 judged) and left as is.
# The one addition to an existing 512 flip root: a store it has never seen (a cell generated into the finished 512 root AFTER
# the flip copy was taken, e.g. the constitution-SI arm) is copied in per store, verified pristine; nothing else moves.
# rsync flags: -rlt --no-owner --no-group --no-perms --partial (MooseFS refuses chown; append-only JSONL is safe to re-sync).
set -u
CLUSTER=cluster_b
HOST=""
RROOT=""
REPO_ROOT=$(cd "$(dirname "$0")/../../../.." && pwd)
CONFIGS=$REPO_ROOT/configs
DEST=${F22S_DEST:-}
KEYS=gemma4e4b,qwen35_4b,nemotron3nano4b,glm46vflash
RUNGS=512,2048,8192,16384
FLIP=1
DRY=0
while [ $# -gt 0 ]; do
  case "$1" in
    --cluster) CLUSTER=$2; shift 2;;
    --host) HOST=$2; shift 2;;
    --remote-root) RROOT=$2; shift 2;;
    --dest) DEST=$2; shift 2;;
    --keys) KEYS=$2; shift 2;;
    --rungs) RUNGS=$2; shift 2;;
    --configs-dir) CONFIGS=$2; shift 2;;
    --no-flip) FLIP=0; shift;;
    --dry-run) DRY=1; shift;;
    -h|--help) sed -n '2,30p' "$0"; exit 0;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done
if [ -z "$HOST" ]; then
  # the login host and shared root of the cluster, from its block in the site config (looked up by registry or site label)
  CL_PY="$REPO_ROOT/reproduce/lib/clusters.py"
  HOST=$(python3 "$CL_PY" --get "$CLUSTER" CLUSTER_SSH) && [ -n "$HOST" ] \
    || { echo "REFUSED: the site config gives no login host (CLUSTER_SSH) for cluster $CLUSTER; describe it there, or pass --host <ssh alias> --remote-root <dir>" >&2; exit 2; }
  if [ -z "$RROOT" ]; then
    CL_SHARED=$(python3 "$CL_PY" --get "$CLUSTER" CLUSTER_SHARED_ROOT) && [ -n "$CL_SHARED" ] \
      || { echo "REFUSED: the site config gives no shared root (CLUSTER_SHARED_ROOT) for cluster $CLUSTER; pass --remote-root <dir>" >&2; exit 2; }
    RROOT=$CL_SHARED/exp/sdfbase_full22
  fi
elif [ -z "$RROOT" ]; then
  echo "REFUSED: --host $HOST names the login host directly; give its remote root with --remote-root <dir>" >&2; exit 2
fi
if [ -z "$DEST" ]; then
  # the lane's run-state tree on this host: $RUNSTATE, expanded by the site module (SISTEER_RUN_STATE_ROOT; empty, with the refusal printed, when unset)
  DEST=$(PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" python3 -m sisteer.site expand '$RUNSTATE/candidate-si-refit/run-state/full22/sampled') || DEST=""
  DEST="${DEST/#\~/$HOME}"
  [ -n "$DEST" ] || { echo "REFUSED: no destination: set SISTEER_RUN_STATE_ROOT (the site's run-state tree) or F22S_DEST, or pass --dest <dir>" >&2; exit 2; }
fi
RSYNC=(rsync -rlt --no-owner --no-group --no-perms --partial)
[ $DRY -eq 1 ] && RSYNC+=(--dry-run)
if [ "$HOST" = local ]; then SRC="$RROOT"; else SRC="$HOST:$RROOT"; fi
STAMP=$(date -u +%FT%TZ)
RC=0
refusals=()
say() { echo "[mirror $(date -u +%H:%M:%SZ)] $*"; }
refuse() { say "REFUSED: $*"; refusals+=("$*"); RC=2; }

# --- remote (read-only) helpers --------------------------------------------------------------------------------------
remote_py() {   # run a python3 snippet (stdin) with $RROOT as argv[1] on the remote (or locally for --host local)
  if [ "$HOST" = local ]; then python3 - "$RROOT" "$@"; else ssh -o BatchMode=yes -o ConnectTimeout=30 "$HOST" python3 - "$RROOT" "$@"; fi
}

# the store index: what exists on the cluster (dir names only + each controls store's cell_id + the ladders)
INDEX=$(remote_py "$KEYS" "$RUNGS" <<'PY'
import json, os, sys
root, keys, rungs = sys.argv[1], sys.argv[2].split(","), [int(x) for x in sys.argv[3].split(",")]
out = {"root": root, "rungs": rungs, "transcripts": [], "controls": {}, "ladders": {}, "serving": [], "gsm8k": {}, "mmlu": []}
tr = os.path.join(root, "transcripts_results")
if os.path.isdir(tr):
    for d in sorted(os.listdir(tr)):
        for k in keys:
            if d.startswith(f"sdfbase_full22_{k}_sampled_t") and os.path.isdir(os.path.join(tr, d)):
                out["transcripts"].append(d)
for r in rungs:
    croot = os.path.join(root, "sampled", f"controls_cap{r}")
    if not os.path.isdir(croot):
        continue
    for tag in sorted(os.listdir(croot)):
        cells = os.path.join(croot, tag, "cells")
        if not os.path.isdir(cells) or not any(tag.startswith(k + "_t") for k in keys):
            continue
        stores = {}
        for s in sorted(os.listdir(cells)):
            mp = os.path.join(cells, s, "manifest.json")
            cid = None
            if os.path.isfile(mp):
                try:
                    cid = json.load(open(mp)).get("cell_id")
                except Exception as e:  # noqa: BLE001
                    cid = f"<unreadable manifest: {type(e).__name__}>"
            stores[s] = {"cell_id": cid, "complete": os.path.isfile(os.path.join(cells, s, "gen_stats.json"))}
        out["controls"].setdefault(str(r), {})[tag] = stores
# the capability battery (lane G2's fixed names): sampled/gsm8k_thinking_on_cap<cap>/<TAG>_t<T> and sampled/mmlu/<TAG>_t<T>
samp = os.path.join(root, "sampled")
if os.path.isdir(samp):
    for d in sorted(os.listdir(samp)):
        if d.startswith("gsm8k_thinking_on_cap") and os.path.isdir(os.path.join(samp, d)):
            out["gsm8k"][d] = sorted(t for t in os.listdir(os.path.join(samp, d))
                                     if os.path.isdir(os.path.join(samp, d, t)) and any(t.startswith(k + "_t") for k in keys))
    mm = os.path.join(samp, "mmlu")
    if os.path.isdir(mm):
        out["mmlu"] = sorted(t for t in os.listdir(mm) if os.path.isdir(os.path.join(mm, t)) and any(t.startswith(k + "_t") for k in keys))
lad = os.path.join(root, "sampled", "ladders")
if os.path.isdir(lad):
    for tag in sorted(os.listdir(lad)):
        if not any(tag.startswith(k + "_t") for k in keys):
            continue
        p = os.path.join(lad, tag, "ladder.json")
        try:
            out["ladders"][tag] = json.load(open(p)) if os.path.isfile(p) else None
        except Exception as e:  # noqa: BLE001
            out["ladders"][tag] = {"unreadable": f"{type(e).__name__}: {e}"}
serv = os.path.join(root, "sampled", "serving")
if os.path.isdir(serv):
    out["serving"] = sorted(d for d in os.listdir(serv) if os.path.isdir(os.path.join(serv, d)))
print(json.dumps(out, sort_keys=True))
PY
) || { say "FAILED: could not index $SRC (ssh / python3 on the remote)"; exit 1; }

idx() { python3 -c 'import json,sys; d=json.loads(sys.argv[1]); exec(sys.argv[2])' "$INDEX" "$1"; }

n_tr=$(idx 'print(len(d["transcripts"]))'); n_serv=$(idx 'print(len(d["serving"]))')
say "source $SRC -> dest $DEST (keys $KEYS, rungs $RUNGS)$([ $DRY -eq 1 ] && echo ' DRY RUN: nothing is written')"
say "index: $n_tr sampled transcript roots, $(idx 'print(sum(len(v) for v in d["controls"].values()))') controls rung roots, $(idx 'print(len(d["ladders"]))') ladders, $n_serv serving dirs, $(idx 'print(sum(len(v) for v in d["gsm8k"].values()))') gsm8k roots, $(idx 'print(len(d["mmlu"]))') mmlu roots"

sync_one() {   # sync_one <remote rel path> <local abs dest> [extra rsync args...]
  local rel=$1 dst=$2; shift 2
  [ $DRY -eq 0 ] && mkdir -p "$dst"
  local out
  out=$("${RSYNC[@]}" "$@" --stats "$SRC/$rel/" "$dst/" 2>&1) || { say "FAILED rsync $rel: $(tail -1 <<<"$out")"; RC=1; return 1; }
  local n; n=$(sed -n 's/^Number of regular files transferred: \([0-9,]*\).*/\1/p' <<<"$out")
  say "$([ $DRY -eq 1 ] && echo 'would sync' || echo 'synced') $rel -> $dst (${n:-?} files)"
}

# --- 1. transcripts (base + cap8192 roots) ---------------------------------------------------------------------------
for d in $(idx 'print("\n".join(d["transcripts"]))'); do
  sync_one "transcripts_results/$d" "$DEST/transcripts_sampled/$d"
done
# --- 2. controls rung roots -------------------------------------------------------------------------------------------
for r in ${RUNGS//,/ }; do
  for tag in $(idx "print('\n'.join(sorted(d['controls'].get('$r', {}))))"); do
    sync_one "sampled/controls_cap$r/$tag" "$DEST/controls_sampled/controls_cap$r/$tag"
  done
done
# --- 2b. the capability battery: GSM8K thinking-ON roots (per cap) and MMLU roots ----------------------------------------
for capdir in $(idx 'print("\n".join(sorted(d["gsm8k"])))'); do
  for tag in $(idx "print('\n'.join(d['gsm8k']['$capdir']))"); do
    sync_one "sampled/$capdir/$tag" "$DEST/$capdir/$tag"
  done
done
for tag in $(idx 'print("\n".join(d["mmlu"]))'); do
  sync_one "sampled/mmlu/$tag" "$DEST/mmlu/$tag"
done
# --- 3. ladders -------------------------------------------------------------------------------------------------------
for tag in $(idx 'print("\n".join(sorted(d["ladders"])))'); do
  sync_one "sampled/ladders/$tag" "$DEST/ladders/$tag"
done
# --- 4. serving records: provenance JSON only, never logs ---------------------------------------------------------------
for s in $(idx 'print("\n".join(d["serving"]))'); do
  sync_one "sampled/serving/$s" "$DEST/serving_records/$s" \
    --exclude='lens_registry/' --include='*/' --include='*.json' --include='*.jsonl' --include='*.verdict' --include='STOP' \
    --exclude='*' --prune-empty-dirs
done

# --- 5. flip-pass copies ----------------------------------------------------------------------------------------------
verify_flip() {   # verify_flip <flip root (has cells/) | one store dir> <fresh|existing> -> PRISTINE | JUDGED(pass 2) | REFUSE …
  python3 - "$1" "$2" <<'PY'
import json, os, sys
path, mode = sys.argv[1], sys.argv[2]          # mode: fresh (must be pristine) | existing (pristine or pass-2 judged)
cells = os.path.join(path, "cells")
if os.path.isdir(cells):
    stores = [(s, os.path.join(cells, s)) for s in sorted(os.listdir(cells)) if os.path.isdir(os.path.join(cells, s))]
elif os.path.isfile(os.path.join(path, "manifest.json")):
    stores = [(os.path.basename(path), path)]
else:
    print(f"REFUSE neither a controls root (cells/) nor a store (manifest.json): {path}"); sys.exit(2)
judge_files, label_rows = [], {}
unfinished = [s for s, sd in stores if not os.path.isfile(os.path.join(sd, "gen_stats.json"))]
if unfinished:
    print(f"REFUSE {len(unfinished)} store(s) without gen_stats.json (generation not finished): {unfinished[:4]}"); sys.exit(2)
for s, sd in stores:
    for f in ("judge_summary.json", "judge_items.jsonl"):
        if os.path.exists(os.path.join(sd, f)):
            judge_files.append(f"{s}/{f}")
    rp = os.path.join(sd, "rows.jsonl")
    if os.path.exists(rp):
        n = 0
        for line in open(rp):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            m = str(r.get("metric", "")); t = str(r.get("task", ""))
            if m.startswith(("xst_", "orb_")) or t.startswith("judged/") or r.get("tier") == 3:
                n += 1
        if n:
            label_rows[f"{s}/rows.jsonl"] = n
if not judge_files and not label_rows:
    print("PRISTINE"); sys.exit(0)
if mode == "existing":
    print(f"JUDGED(pass 2): {len(judge_files)} judge file(s), label rows in {len(label_rows)} store(s)"); sys.exit(0)
print("REFUSE judge artefacts in a FRESH flip copy: " + "; ".join(judge_files[:6])
      + ("; " if judge_files and label_rows else "") + "; ".join(f"{k}: {v} label rows" for k, v in list(label_rows.items())[:6]))
sys.exit(2)
PY
}

flip_copy() {   # flip_copy <remote rel dir> <local dst> <what> <root|store> — rsync when absent, verify, never re-sync an existing copy
  local rel=$1 dst=$2 what=$3 shape=$4 marker
  [ "$shape" = root ] && marker="$dst/cells" || marker="$dst/manifest.json"
  if [ -e "$marker" ]; then
    local v; v=$(verify_flip "$dst" existing); local vrc=$?
    if [ $vrc -ne 0 ]; then refuse "$what: existing flip copy $dst: ${v#REFUSE }"; return; fi
    say "flip $what: exists, left as is ($v): $dst"; return
  fi
  if [ $DRY -eq 1 ]; then say "flip $what: would copy $SRC/$rel -> $dst"; return; fi
  mkdir -p "$dst"
  local out
  out=$("${RSYNC[@]}" --stats "$SRC/$rel/" "$dst/" 2>&1) || { say "FAILED rsync flip $rel: $(tail -1 <<<"$out")"; RC=1; return; }
  local v; v=$(verify_flip "$dst" fresh); local vrc=$?
  if [ $vrc -ne 0 ]; then
    refuse "$what: fresh flip copy $dst is not pristine — ${v#REFUSE }; the copy is left in place for inspection, remove it deliberately"
    return
  fi
  say "flip $what: copied $rel -> $dst ($v)"
}

if [ $FLIP -eq 1 ]; then
  # 5a. the 512 root, whole, per (TAG, trial) — only once every explicit cell of the sampled config has finished generating
  for tag in $(idx "print('\n'.join(sorted(d['controls'].get('512', {}))))"); do
    key=${tag%_t*}; T=${tag##*_t}; cfg=$CONFIGS/sdfbase_full22_${key}_sampled_t${T}.yaml
    if [ ! -f "$cfg" ]; then refuse "cap512/$tag: sampled config $cfg absent (pass --configs-dir); cannot tell a finished root from a partial one"; continue; fi
    state=$(idx "
import yaml
cells = yaml.safe_load(open('$cfg'))['explicit_cells']
stores = d['controls']['512']['$tag']
have = {v['cell_id']: v for v in stores.values()}
missing = sorted(c for c in cells if c not in have)
unfinished = sorted(c for c in cells if c in have and not have[c]['complete'])
if missing or unfinished:
    print('SKIP cap512/$tag: generation not finished — ' + (f'no store for cells {missing}; ' if missing else '') + f'{len(unfinished)} store(s) without gen_stats.json {unfinished[:4]}')
else:
    print('OK')")
    if [ "$state" != OK ]; then
      if [ -e "$DEST/controls_sampled/flip_pass/cap512/$tag/cells" ]; then
        v=$(verify_flip "$DEST/controls_sampled/flip_pass/cap512/$tag" existing) || refuse "cap512/$tag: existing flip copy: ${v#REFUSE }"
      fi
      say "flip $state"; continue
    fi
    flip_copy "sampled/controls_cap512/$tag" "$DEST/controls_sampled/flip_pass/cap512/$tag" "cap512/$tag" root
    # 5a'. a cell ADDED to a finished 512 root after its flip copy was taken (the constitution-SI arm): the
    # existing flip root is pass 2's and is never re-synced, but a store it has never seen is not pass 2's either — copy that
    # store alone (per store, verified pristine, same as a rung store); every store already in the flip root is left untouched.
    froot=$DEST/controls_sampled/flip_pass/cap512/$tag
    if [ -d "$froot/cells" ]; then
      for store in $(idx "
import os
stores = d['controls']['512']['$tag']
have = set(os.listdir('$froot/cells'))
print('\n'.join(sorted(s for s, v in stores.items() if v['complete'] and s not in have)))"); do
        cell=$(idx "print(d['controls']['512']['$tag']['$store']['cell_id'])")
        flip_copy "sampled/controls_cap512/$tag/cells/$store" "$froot/cells/$store" "cap512/$tag/$cell (cell added after the root's flip copy)" store
      done
    fi
  done
  # 5b. each escalated cell's FINAL rung store, per the ladder
  PLAN=$(idx '
import re
lines = []
rungs_known = d["rungs"]
for tag in sorted(d["controls"].get("512", {})):
    lad = d["ladders"].get(tag)
    if not isinstance(lad, dict) or "cells" not in lad:
        lines.append(f"SKIP {tag} no ladder.json yet (ladders/{tag}) -> no rung flip copies"); continue
    for cell, by in sorted(lad["cells"].items()):
        finals, inc, off = {}, [], []                  # PER CONTROL: each control has its own final rung (the two controls of one cell exit at different rungs)
        for ctl, rec in sorted(by.items()):
            v = str(rec.get("reached_zero_at"))
            if v == "incomplete":
                inc.append(ctl); continue
            m = re.match(r"^(\d+)", v)
            if not m:
                inc.append(f"{ctl}:{v!r}"); continue
            r = int(m.group(1))
            if r > 512 and r not in d["rungs"]:
                off.append(f"{ctl}:{r}"); continue      # a rung this mirror was not asked for (--rungs): never inferred, never copied
            finals[ctl] = r if r > 512 else 512         # 2048 / 8192 / 16384 — whatever --rungs names above 512
        if inc:
            lines.append(f"SKIP {tag}/{cell} ladder incomplete for {inc} (the job has not finished its ladder; resubmit, never infer)"); continue
        if off:
            lines.append(f"SKIP {tag}/{cell} ladder final rung outside --rungs {rungs_known}: {off} (pass --rungs with that rung to mirror its flip copy)"); continue
        above = sorted({r for r in finals.values() if r > 512})
        if not above:
            lines.append(f"NONE {tag}/{cell} left the ladder at 512 (no rung flip copy)"); continue
        if len(set(finals.values())) > 1:
            lines.append(f"SPLIT {tag}/{cell} controls exit at different rungs — " + ", ".join(f"{c} -> {r}" for c, r in sorted(finals.items())) + " (one flip copy per distinct final rung)")
        for final in above:                             # EVERY distinct final rung among the cell controls — each store is the only place that control final-rung text lives
            ctls = ",".join(sorted(c for c, r in finals.items() if r == final))
            stores = [s for s, rec in d["controls"].get(str(final), {}).get(tag, {}).items() if rec["cell_id"] == cell]
            if len(stores) != 1:
                lines.append(f"REFUSE {tag}/{cell} ladder says final rung {final} but controls_cap{final}/{tag}/cells holds {len(stores)} store(s) for it (final rung of {ctls})"); continue
            if not d["controls"][str(final)][tag][stores[0]]["complete"]:
                lines.append(f"SKIP {tag}/{cell} final rung {final} store {stores[0]} has no gen_stats.json yet (generation not finished; retried next run)"); continue
            lines.append(f"COPY {final} {tag} {cell} {stores[0]} {ctls}")
print("\n".join(lines))')
  while IFS= read -r line; do
    [ -z "$line" ] && continue
    case "$line" in
      COPY\ *) read -r _ r tag cell store ctls <<<"$line"
               flip_copy "sampled/controls_cap$r/$tag/cells/$store" "$DEST/controls_sampled/flip_pass/cap$r/$tag/cells/$store" "cap$r/$tag/$cell" store;;
      REFUSE\ *) refuse "${line#REFUSE }";;
      *) say "flip ${line}";;
    esac
  done <<<"$PLAN"
  # report the assembled rung roots (each store was verified on arrival; an existing root may already carry pass-2 files)
  for r in ${RUNGS//,/ }; do
    [ "$r" = 512 ] && continue
    for tag in $(idx "print('\n'.join(sorted(d['controls'].get('512', {}))))"); do
      [ -d "$DEST/controls_sampled/flip_pass/cap$r/$tag/cells" ] || continue
      v=$(verify_flip "$DEST/controls_sampled/flip_pass/cap$r/$tag" existing) || refuse "cap$r/$tag: ${v#REFUSE }"
      say "flip cap$r/$tag: $(ls "$DEST/controls_sampled/flip_pass/cap$r/$tag/cells" | wc -l) store(s) ($v)"
    done
  done
fi

# --- summary ------------------------------------------------------------------------------------------------------------
if [ $DRY -eq 0 ]; then
  mkdir -p "$DEST"
  python3 - "$DEST/mirror_sampled.last.json" "$STAMP" "$SRC" "$RC" "$INDEX" "${refusals[@]:-}" <<'PY'
import json, sys
out, stamp, src, rc, index, *ref = sys.argv[1:]
d = json.loads(index)
json.dump({"at": stamp, "source": src, "exit": int(rc), "refusals": [r for r in ref if r],
           "transcript_roots": d["transcripts"], "controls_roots": {r: sorted(v) for r, v in d["controls"].items()},
           "gsm8k_roots": {c: sorted(v) for c, v in d["gsm8k"].items()}, "mmlu_roots": sorted(d["mmlu"]),
           "ladders": sorted(d["ladders"]), "serving_dirs": d["serving"]}, open(out, "w"), indent=1, sort_keys=True)
PY
fi
if [ ${#refusals[@]} -gt 0 ]; then say "${#refusals[@]} refusal(s):"; printf '  - %s\n' "${refusals[@]}"; fi
say "done (exit $RC)$([ $DRY -eq 1 ] && echo ' — dry run, nothing written')"
exit $RC
