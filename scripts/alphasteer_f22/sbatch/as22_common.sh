#!/bin/bash
# as22_common.sh — the shared body every scripts/alphasteer_f22/sbatch/*.sbatch sources.
#
# The AlphaSteer campaign runs INSIDE the full22 wave's machinery but on its OWN root: our
# objects, roster, pins, bank views and run outputs live under `alphasteer_f22/`, and nothing here
# ever writes under exp/objects/sdfbase/full22/, configs/sdfbase_full22_*, or the wave-wide
# sentinels $WAVE_ROOT/STOP and attackers/STOP (those are shared with other live legs).
# Everything else — the cluster/env block, the flock'd node-local venvs, the logging and the figure
# mirror — is the wave's, sourced rather than re-implemented, so a fix there reaches us too.
#
# NEVER --export= (hard rule 1): every value travels positionally or through these functions.
#
# THE ONE THING THIS FILE ADDS OVER THE WAVE'S common.sh IS A GATE:
# `as22_assert_serving_identity <production|ladder>`, in two modes. Exactly one tree in this
# campaign (W_L, the lambda ladder's) may differ from the wave's serving stack, it may differ by
# exactly one frozen patch, and it must carry the rho-emit flag; every other tree must be
# byte-identical to that stack with the flag unset. Both directions refuse. See §C.2a.

# ---------------------------------------------------------------------------------------------- #
# the wave's body: f22_log / f22_fatal / f22_init / f22_driver_venv / f22_serving_venv /           #
# f22_mirror / f22_result. Resolved RELATIVE to this file, so it is the same checkout's copy —     #
# never a path that could resolve into another tree.                                               #
# ---------------------------------------------------------------------------------------------- #
AS22_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AS22_REPO_ROOT="$(cd "$AS22_DIR/../../.." && pwd)"
# shellcheck source=/dev/null
source "$AS22_REPO_ROOT/scripts/sdfbase/full22/common.sh" \
  || { echo "FATAL: cannot source $AS22_REPO_ROOT/scripts/sdfbase/full22/common.sh" >&2; exit 1; }

# ---------------------------------------------------------------------------------------------- #
# the serving-identity literals (§C.2a). None is self-referential.                                 #
# ---------------------------------------------------------------------------------------------- #

#: the wave's serving baseline — the sha every production tree's serving stack must equal
AS22_W8_SHA=935dfb43a743ec00bdf41971ecd21302aacd5342
#: the two paths whose last-touching commit DEFINES `serving_impl_commit`
AS22_SERVING_DIR=src/sisteer/capeval/serving/
AS22_PETRI_RUNNER=src/sisteer/capeval/tiers/petri_runner.py
#: W8's serving stack by CONTENT: the git object ids of the two paths above AT $AS22_W8_SHA (`git rev-parse
#: <W8>:<path>`). Read only where the clone does not carry W8 itself — a published repository without its
#: development history — so the production byte-identity is still decided by bytes, never assumed; the
#: ladder's frozen-patch check has no content form (it regenerates a diff AGAINST W8) and refuses there
AS22_W8_SERVING_TREE=49561876c3b96a3d5e714c2d1bd1573ca4022b37
AS22_W8_PETRI_RUNNER_BLOB=530c3c5764035bd85b7efb6368bc2fdca416174d
#: the ONE file W_L may change, and its sha256 there
AS22_LENS_WORKER=src/sisteer/capeval/serving/lens_worker.py
AS22_LADDER_LENS_WORKER_SHA256=47da382aac38ba6f3e2494ead2b67dc199029943b6eadc580bb5616a9b4263be
#: the frozen rho_emit_probe.diff. STAGED on both clusters beside the objects, because cluster /home
#: is node-local disk and nothing outside the shared root is visible on a compute node.
AS22_LADDER_PATCH_SHA256=cc84104a6d879db5389d1ea386eb38ebb3dbab662475962f3394f1484ac40edb
AS22_LADDER_PATCH_NAME=rho_emit_probe.diff
#: THE ABBREVIATION GOTCHA, measured: the frozen diff was produced with 8-character blob
#: abbreviation (`index a66fb64c..54fc7dee 100644`) while this repo's default is 7, so every
#: regeneration of it for byte-comparison MUST pass `-c core.abbrev=8`. Without it the diff differs
#: from the frozen one in exactly two bytes and the gate refuses a correct tree.
AS22_ABBREV=8

#: THE JUDGE-CACHE SCRUB (§C.0). Ladder condition (a) is judged against a PRIVATE cache
#: from which 450 grok-4.6-produced verdicts filed under sonnet-5 protocol keys were removed BY
#: PRODUCER STAMP. This is the catalogue those 450 rows were enumerated in; the attestation staged
#: on the cluster must name it.
AS22_FOREIGN_ROWS_CATALOGUE_SHA256=b52c3e13810fa9e990199b6a7564b4a39043be9c1b4d3662a3e58ffb4d31ca51
#: sha256 of the staged judge_cache_scrub.json itself. A change here means somebody re-scrubbed —
#: which is a thing to look at, not to wave through, so it refuses and names the remedy.
AS22_JUDGE_SCRUB_SHA256=1b285762715904f59d26aaf7867f8594b17d64ebd8b1b4fa4d99c922d9cc3897

# ---------------------------------------------------------------------------------------------- #
# init                                                                                             #
# ---------------------------------------------------------------------------------------------- #

as22_init(){   # $1 = the sha of the checkout this job runs from ($SHARED/eval-repo/$1)
  # the wave's init does the load-bearing work: resolve $SHARED / $CLUSTER, source the SHARED
  # hf_env.sh, set NCCL/alloc/UV/tokenizer env, export SISTEER_LENS_STACKED_REDUCTIONS=1, verify the
  # checkout is AT this sha and clean, and cd into it.
  # THE SERVING PATH is DEFERRED (common.sh f22_select_serving_path, the kind-aware default of 2026-09-15): this campaign's
  # manifest comes from as22_targets AFTER the checkout is resolved, so as22_resolve_config binds it (f22_bind_serving_path
  # "$MANIFEST": [none, alphasteer] -> the lens, by kind); a job that never resolves a config (the attacker pool, the lens gate's
  # own standup) never binds and never reads the path. Never the wave's petri manifest: KEY / VIEW name a different cell set here.
  F22_SERVING_PATH_DEFER=1
  f22_init "${1:?sha}"
  # ...then re-point every wave-shared path at OUR root. WAVE_ROOT is what the wave's own
  # f22_stop_requested reads, so re-pointing it also means our stop sentinel is ours alone and we
  # can never read (or write) the wave's.
  WAVE=alphasteer_f22
  WAVE_ROOT="$SHARED/exp/alphasteer_f22"
  # ...and the object root HONOURS a value the calling script's BODY pre-set: the paper-sized
  # re-bake serves its rungs from the PARALLEL root exp/objects/alphasteer_f22_rebake so the
  # campaign's own objects are never mutated. Unset = the campaign root, so every caller that does
  # not set it is byte-for-byte today's. Set it in the sbatch BODY, never via --export= (hard rule 1).
  AS22_OBJ_ROOT="${AS22_OBJ_ROOT:-$SHARED/exp/objects/alphasteer_f22}"
  mkdir -p "$WAVE_ROOT"
  [ -d "$AS22_OBJ_ROOT" ] || f22_fatal "no object root $AS22_OBJ_ROOT (Phase F staging has not reached this cluster)"
  f22_log "as22: root=$WAVE_ROOT objects=$AS22_OBJ_ROOT cluster=$CLUSTER"
}

as22_resolve_config(){   # $1 = model key, $2 = view (OPTIONAL, cross-checked)  -> KEY VIEW CFG MANIFEST
  # The view, the config path and the manifest path all come from scripts/alphasteer_f22/
  # as22_targets.py — the ONE table this campaign's Phase C tooling shares. bake_rungs.py lays the
  # per-rung defenses dir out under `cells_view/<view>` from that table, so a view resolved any
  # other way here points the standup at a directory the bake never wrote. A view passed on the
  # submission line is therefore CROSS-CHECKED against the table, never trusted over it.
  KEY="${1:?key}"; local want_view="${2:-}" py
  py="${MPY:-python3}"
  VIEW=$("$py" -c "import sys; sys.path.insert(0, 'scripts/alphasteer_f22'); import as22_targets as t; print(t.view_of(sys.argv[1]))" "$KEY") \
    || f22_fatal "no view assignment for target $KEY in scripts/alphasteer_f22/as22_targets.VIEW_OF"
  if [ -n "$want_view" ] && [ "$want_view" != "$VIEW" ]; then
    f22_fatal "submitted view '$want_view' but as22_targets.VIEW_OF['$KEY'] is '$VIEW' — the bake lays the rung dirs out under cells_view/$VIEW, so serving '$want_view' would resolve a directory that was never written"
  fi
  CFG=$("$py" -c "import sys; sys.path.insert(0, 'scripts/alphasteer_f22'); import as22_targets as t; print(t.serve_config_rel(sys.argv[1]))" "$KEY")
  MANIFEST=$("$py" -c "import sys; sys.path.insert(0, 'scripts/alphasteer_f22'); import as22_targets as t; print(t.manifest_rel(sys.argv[1]))" "$KEY")
  [ -f "$CFG" ] || f22_fatal "missing $CFG (generate it: scripts/alphasteer_f22/make_ladder_configs.py)"
  [ -f "$MANIFEST" ] || f22_fatal "missing $MANIFEST (mt_serving_up / the standup require a manifest whose cell set equals the config's explicit_cells)"
  # TEXT-ONLY SERVING for the multimodal targets — the wave's belt, ours too. Without
  # `--limit-mm-per-prompt {"image":0,"video":0}` the engine's multimodal profiling (dummy max-size
  # images sized off S x L) is OOM-killed by the 200G host cgroup after the venvs have built, i.e.
  # ~15 minutes of GPU time to learn that a flag is missing.
  case "$KEY" in qwen35_4b|glm46vflash)
    grep -q -- '--limit-mm-per-prompt' "$MANIFEST" \
      || f22_fatal "$MANIFEST serves multimodal $KEY without --limit-mm-per-prompt (text-only pin) — regenerate the manifest";;
  esac
  f22_log "as22 config: key=$KEY view=$VIEW (as22_targets) cfg=$CFG manifest=$MANIFEST"
  # the serving path, bound from THIS campaign's manifest (deferred at as22_init; an explicit F22_SERVING_PATH was bound there)
  [ "${F22_SERVING_PATH_MODE:-}" != deferred ] || f22_bind_serving_path "$MANIFEST"
}

# ---------------------------------------------------------------------------------------------- #
# the serving-identity gate (§C.2a)                                                               #
# ---------------------------------------------------------------------------------------------- #
#
# Called with the mode as a LITERAL at each call site — never a variable, never a default. Two
# modes, and there is no mode that passes both an empty and a non-empty serving diff:
#
#   production  the serving stack is byte-identical to the wave's serving stack AND the rho-emit
#               flag is unset AND no ladder sha is in the environment.
#   ladder      ALL FOUR of: (i) $AS22_LADDER_SHA is set, non-empty, and equals HEAD; (ii) the
#               serving delta vs the wave's serving stack names exactly lens_worker.py; (iii)
#               that file's sha256 at HEAD is the patched one AND the regenerated serving diff
#               byte-equals the frozen patch; (iv) SISTEER_LENS_RHO_EMIT=1.
#
# What refuses, enumerated: a PATCHED PRODUCTION leg (production sees a non-empty diff); a
# production leg with a stray flag or a stray ladder sha; a ladder at any other sha (i); a ladder
# whose serving delta is anything but the frozen patch (ii, iii); an UNPATCHED ladder (iii — and
# that case matters most: on W_A the rho record does not exist, so condition (c) could not be
# measured and the run would otherwise fall back, silently, to having no criterion at all); a
# ladder with the flag off (iv).
#
# WHY THE LADDER SHA IS NOT A LITERAL IN THIS FILE. This file lives on W_L, so recording W_L's own
# sha here would change the sha it records. The sha therefore travels the way the wave's WAVE_SHA
# does — positionally on the submission line, into $AS22_LADDER_SHA — and clause (i) checks it
# against `git rev-parse HEAD`. A wrong-but-nonempty sha is impossible to pass because (ii)/(iii)
# are CONTENT checks on the tree that sha names.

as22_serving_bytes_problems(){   # appends to `problems`: the serving files ON DISK against the entries HEAD commits
  # The engine imports the files on disk, and `git diff --quiet HEAD` (the clean-checkout check) reads them through the clean
  # filters and autocrlf, so a .gitattributes filter that strips an injected change passes it. Anchored at the repository root
  # (never vacuous from a subdirectory), NUL-separated (a name git would C-quote is read raw): every entry HEAD commits under the
  # serving paths is compared with what is on disk as it is — a blob by `git hash-object --no-filters`, a symlink by its target —
  # and every path on disk OR in the index under them that HEAD does not carry is refused, the contents of a `__pycache__/`
  # directory excepted (a .pyc anywhere else is importable and is not).
  local root entry mode type blob path rest raw target
  root=$(git rev-parse --show-toplevel 2>/dev/null) \
    || { problems+=("not inside a git work tree: the serving files on disk cannot be compared with HEAD"); return 0; }
  local -A inhead=()
  while IFS= read -r -d '' entry; do
    mode=${entry%% *}; rest=${entry#* }; type=${rest%% *}; rest=${rest#* }; blob=${rest%%$'\t'*}; path=${rest#*$'\t'}
    inhead["$path"]=1
    [ "$type" = blob ] || continue
    if [ "$mode" = 120000 ]; then
      if [ ! -L "$root/$path" ]; then
        problems+=("$path is a symlink at HEAD but not on disk — the engine would not serve what HEAD commits")
        continue
      fi
      target=$(readlink "$root/$path")
      raw=$(printf '%s' "$target" | git -C "$root" hash-object --no-filters --stdin 2>/dev/null) || raw=unreadable
      [ "$raw" = "$blob" ] \
        || problems+=("$path on disk is a symlink to '$target' (blob $raw), not the target HEAD commits (blob $blob) — the engine follows the link on disk")
      continue
    fi
    if [ -L "$root/$path" ]; then
      problems+=("$path is a symlink on disk but a regular file at HEAD — the engine would serve whatever it points at")
      continue
    fi
    if [ ! -f "$root/$path" ]; then
      problems+=("$path is committed at HEAD but is not on disk — the engine would not serve HEAD's bytes")
      continue
    fi
    raw=$(git -C "$root" hash-object --no-filters -- "$root/$path" 2>/dev/null) || raw=unreadable
    [ "$raw" = "$blob" ] \
      || problems+=("$path on disk is blob $raw (bytes as they are, --no-filters), not the $blob HEAD commits — a clean filter or autocrlf cannot hide the difference, and the engine serves the file on disk")
  done < <(git -C "$root" ls-tree -r -z --full-tree HEAD -- "$AS22_SERVING_DIR" "$AS22_PETRI_RUNNER" 2>/dev/null)
  local extra=()
  while IFS= read -r -d '' path; do
    [ -n "${inhead[$path]+x}" ] && continue
    case "$path" in */__pycache__/*|__pycache__/*) continue;; esac
    extra+=("$path")
  done < <(git -C "$root" ls-files -z --cached --others -- "$AS22_SERVING_DIR" "$AS22_PETRI_RUNNER" 2>/dev/null | sort -zu)
  [ "${#extra[@]}" -eq 0 ] \
    || problems+=("paths under the serving paths that HEAD does not carry (on disk or in the index): ${extra[*]} — the engine imports what is on disk")
}

as22_assert_serving_identity(){   # $1 = production | ladder   (a literal at every call site)
  local mode="${1:-}" head diff_files worker_sha regen_sha patch problems=()
  case "$mode" in
    production|ladder) ;;
    *) f22_fatal "as22_assert_serving_identity: mode must be the LITERAL 'production' or 'ladder', got '${mode}'";;
  esac
  head=$(git rev-parse HEAD 2>/dev/null) || f22_fatal "as22_assert_serving_identity: $PWD is not a git checkout"
  # W8's place in HEAD's LINEAGE decides HOW the byte-identity is read: from the history where HEAD descends from W8 (as
  # always), by CONTENT — the pinned git object ids of the serving tree and petri_runner — where the clone does not carry
  # W8 at all (a published repository without its development history, a shallow clone). W8 merely PRESENT in the object
  # store with HEAD not descending from it (an unrelated history fetched in) is neither: refused by name. The ladder's
  # frozen-patch clauses regenerate a diff AGAINST W8 and have no content form, so without W8 in the lineage they refuse.
  local w8=absent by_content=""
  if git merge-base --is-ancestor "$AS22_W8_SHA" HEAD 2>/dev/null; then w8=history
  elif git cat-file -e "$AS22_W8_SHA^{commit}" 2>/dev/null; then w8=present; fi
  # whatever the lineage says, the engine serves the files ON DISK: their bytes as they are must be HEAD's committed bytes
  as22_serving_bytes_problems

  if [ "$mode" = production ]; then
    if [ "$w8" = history ]; then
      git diff --quiet "$AS22_W8_SHA" "$head" -- "$AS22_SERVING_DIR" "$AS22_PETRI_RUNNER" \
        || problems+=("the serving stack differs from W8 ($AS22_W8_SHA): $(git diff --name-only "$AS22_W8_SHA" "$head" -- "$AS22_SERVING_DIR" "$AS22_PETRI_RUNNER" | tr '\n' ' ')— a production leg must serve the wave's bytes, and every recorded serving_impl_commit claims it does")
    elif [ "$w8" = present ]; then
      problems+=("W8 ($AS22_W8_SHA) is in this clone's object store but HEAD $head does not descend from it (an unrelated history fetched in) — the serving stack's identity is read from the history HEAD descends from, or by content where W8 is absent; presence alone is neither, refusing")
    else
      local tree_now petri_now
      tree_now=$(git rev-parse "$head:$AS22_SERVING_DIR" 2>/dev/null) || tree_now=absent
      petri_now=$(git rev-parse "$head:$AS22_PETRI_RUNNER" 2>/dev/null) || petri_now=absent
      if [ "$tree_now" != "$AS22_W8_SERVING_TREE" ] || [ "$petri_now" != "$AS22_W8_PETRI_RUNNER_BLOB" ]; then
        problems+=("the serving stack is not W8's bytes (tree $tree_now vs pinned $AS22_W8_SERVING_TREE; petri_runner $petri_now vs pinned $AS22_W8_PETRI_RUNNER_BLOB) — this clone does not carry W8 ($AS22_W8_SHA), so the comparison is by content; a production leg must serve the wave's bytes, and every recorded serving_impl_commit claims it does")
      else
        by_content=" (by content: this clone does not carry W8, and the serving tree $tree_now / petri_runner $petri_now are W8's pinned object ids)"
      fi
    fi
    [ -z "${SISTEER_LENS_RHO_EMIT+set}" ] \
      || problems+=("SISTEER_LENS_RHO_EMIT is SET ('${SISTEER_LENS_RHO_EMIT}') in a production body — on an unpatched tree the flag is dead, but a stray export is exactly the drift this refuses loudly rather than ignoring")
    [ -z "${AS22_LADDER_SHA:-}" ] \
      || problems+=("AS22_LADDER_SHA is set ('${AS22_LADDER_SHA}') in a production body — the ladder sha never belongs here")
  else
    # (i) the named sha, checked against HEAD
    if [ -z "${AS22_LADDER_SHA:-}" ]; then
      problems+=("AS22_LADDER_SHA is unset or empty — the ladder tree must be NAMED on the submission line (positional \$1), so a ladder can never run un-named; refusing rather than skipping clause (i) by omission")
    elif [ "$AS22_LADDER_SHA" != "$head" ]; then
      problems+=("HEAD $head != the named ladder sha $AS22_LADDER_SHA")
    fi
    if [ "$w8" = absent ]; then
      # (ii) and (iii) regenerate the serving diff against W8: without W8 there is no diff to regenerate and nothing to compare
      # the frozen patch to, so neither clause can pass or fail on its merits — refused by name, naming the missing history
      problems+=("the ladder's frozen-patch check regenerates the diff against W8 ($AS22_W8_SHA), which this clone does not carry (a published repository without its development history); it needs the full development history — clauses (ii) and (iii) were not evaluated")
    elif [ "$w8" = present ]; then
      problems+=("the ladder's frozen-patch check regenerates the diff against W8 ($AS22_W8_SHA), which HEAD $head does not descend from (W8 is in the object store, an unrelated history fetched in); it needs the full development history — clauses (ii) and (iii) were not evaluated")
    else
    # (ii) the delta vs W8 is exactly one file
    diff_files=$(git diff --name-only "$AS22_W8_SHA" "$head" -- "$AS22_SERVING_DIR" "$AS22_PETRI_RUNNER")
    [ "$diff_files" = "$AS22_LENS_WORKER" ] \
      || problems+=("the serving delta vs W8 is [$(echo "$diff_files" | tr '\n' ' ')], must be exactly $AS22_LENS_WORKER")
    # (iii) that file's content, and the whole serving diff, byte for byte
    worker_sha=$(git show "$head:$AS22_LENS_WORKER" 2>/dev/null | sha256sum | cut -d' ' -f1)
    if [ "$worker_sha" != "$AS22_LADDER_LENS_WORKER_SHA256" ]; then
      problems+=("$AS22_LENS_WORKER at HEAD is sha256 ${worker_sha:-<unreadable>}, expected the PATCHED $AS22_LADDER_LENS_WORKER_SHA256. An UNPATCHED ladder is the dangerous case: sisteer_rho_record would not exist, so condition (c) could not be measured and the ladder would run with no served criterion at all")
    fi
    regen_sha=$(git -c core.abbrev="$AS22_ABBREV" diff "$AS22_W8_SHA" "$head" -- "$AS22_SERVING_DIR" | sha256sum | cut -d' ' -f1)
    [ "$regen_sha" = "$AS22_LADDER_PATCH_SHA256" ] \
      || problems+=("the regenerated serving diff (core.abbrev=$AS22_ABBREV) is sha256 $regen_sha, expected the frozen $AS22_LADDER_PATCH_SHA256")
    # ...and against the STAGED patch file itself, byte for byte, so the comparison is to an
    # artifact on disk and not only to a hash literal in this file
    patch="$AS22_OBJ_ROOT/$AS22_LADDER_PATCH_NAME"
    if [ ! -f "$patch" ]; then
      problems+=("no staged frozen patch at $patch (stage it beside the objects on both clusters; ~/specs does not exist on a cluster node)")
    else
      local staged_sha
      staged_sha=$(sha256sum "$patch" | cut -d' ' -f1)
      if [ "$staged_sha" != "$AS22_LADDER_PATCH_SHA256" ]; then
        problems+=("staged $patch is sha256 $staged_sha, not the frozen $AS22_LADDER_PATCH_SHA256")
      elif ! git -c core.abbrev="$AS22_ABBREV" diff "$AS22_W8_SHA" "$head" -- "$AS22_SERVING_DIR" | diff -q - "$patch" >/dev/null; then
        problems+=("the regenerated serving diff is not byte-identical to $patch")
      fi
    fi
    fi
    # (iv) the flag
    [ "${SISTEER_LENS_RHO_EMIT:-}" = "1" ] \
      || problems+=("SISTEER_LENS_RHO_EMIT is '${SISTEER_LENS_RHO_EMIT:-<unset>}', must be 1 — a ladder standup that came up without the flag records nothing and condition (c) would be undecidable")
  fi

  if [ "${#problems[@]}" -gt 0 ]; then
    f22_log "SERVING IDENTITY REFUSED (mode=$mode, HEAD=$head):"
    local p; for p in "${problems[@]}"; do f22_log "  - $p"; done
    f22_fatal "as22_assert_serving_identity $mode: ${#problems[@]} problem(s) — refusing before anything is served"
  fi
  if [ "$mode" = production ]; then
    f22_log "serving identity OK (production): serving stack byte-identical to W8 @ $AS22_W8_SHA, SISTEER_LENS_RHO_EMIT unset$by_content"
  else
    f22_log "serving identity OK (ladder): HEAD=$head == AS22_LADDER_SHA; serving delta vs W8 is exactly $AS22_LENS_WORKER (sha256 $worker_sha); diff byte-equals the frozen $AS22_LADDER_PATCH_NAME; SISTEER_LENS_RHO_EMIT=1"
  fi
}

# ---------------------------------------------------------------------------------------------- #
# the judge-cache scrub attestation (§C.0 item 4) — a PRECONDITION of the first standup            #
# ---------------------------------------------------------------------------------------------- #
#
# Why a GENERATION job checks a JUDGING precondition: the ladder's whole point is to choose an
# operating point, and condition (a) is judged against the private scrubbed cache. Generating six
# rungs of traffic that can only be scored against a biased cache is GPU time spent on a reading
# that would have to be thrown away. It is cheap to refuse here and expensive to discover later.
#
# The CATALOGUE itself is a grading host file and is deliberately not on the cluster, so what is checked
# here is the attestation's own internal consistency plus the catalogue sha it names.

as22_require_scrub(){   # needs AS22_OBJ_ROOT, MPY (or any python3)
  local att="$AS22_OBJ_ROOT/judge_cache_scrub.json" py sha
  py="${MPY:-python3}"
  [ -f "$att" ] || f22_fatal "no judge-cache scrub attestation at $att — the ladder's condition (a) would be judged against a cache holding 450 foreign verdicts under sonnet-5 keys, which biases the benign refusal rate DOWNWARD and lets an over-refusing lambda pass (D21/D23). Run scripts/alphasteer_f22/scrub_judge_cache.py on the grading host and stage the attestation to both clusters."
  sha=$(sha256sum "$att" | cut -d' ' -f1)
  [ "$sha" = "$AS22_JUDGE_SCRUB_SHA256" ] \
    || f22_fatal "$att is sha256 $sha, expected $AS22_JUDGE_SCRUB_SHA256 — the scrub was re-run. That is a thing to look at, not to wave through: check the new attestation's n_removed / catalogue_sha256, then update AS22_JUDGE_SCRUB_SHA256 in as22_common.sh deliberately."
  "$py" - "$att" "$AS22_FOREIGN_ROWS_CATALOGUE_SHA256" <<'PY' || f22_fatal "the scrub attestation $att does not certify a clean scrub"
import json, sys
att, want_cat = json.load(open(sys.argv[1])), sys.argv[2]
bad = []
if att.get("catalogue_sha256") != want_cat:
    bad.append(f"catalogue_sha256 {att.get('catalogue_sha256')} != {want_cat}")
if att.get("matches_catalogue") is not True:
    bad.append(f"matches_catalogue is {att.get('matches_catalogue')!r}")
if att.get("problems"):
    bad.append(f"problems: {att['problems']}")
if att.get("n_removed") != att.get("catalogue_n_foreign_rows"):
    bad.append(f"n_removed {att.get('n_removed')} != catalogue_n_foreign_rows {att.get('catalogue_n_foreign_rows')}")
orgs = att.get("by_org_used") or {}
if set(orgs) - {"openrouter"}:
    bad.append(f"by_org_used {orgs} removed rows from a pool other than openrouter — the scrub is by PRODUCER STAMP, never by shape")
print(json.dumps({"n_removed": att.get("n_removed"), "by_org_used": orgs,
                  "by_judge_model": att.get("by_judge_model"),
                  "private_cache": att.get("private_cache"), "problems": bad}))
sys.exit(1 if bad else 0)
PY
  f22_log "judge-cache scrub attestation OK: $att"
}

# ---------------------------------------------------------------------------------------------- #
# stopping our own standups — NEVER a wave-wide sentinel                                           #
# ---------------------------------------------------------------------------------------------- #

as22_stop_standups(){   # $1 = the ladder dir; touches each rung's OWN standup stop file
  local ladder="${1:-}" d
  [ -n "$ladder" ] || return 0
  for d in "$ladder"/lam*/serve; do
    [ -d "$d" ] && touch "$d/STOP" 2>/dev/null
  done
  return 0
}
