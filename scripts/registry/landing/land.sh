#!/bin/bash
# land.sh <branch> <title> <body-file>   |   land.sh --plan   |   land.sh --live-db-files   |   land.sh --repo
# The results-registry LANDING STEP, Slurm-only — one full suite, on the MERGED tip:
# merge origin/main → a gate verdict for the merged tip → the live-DB test files → push → the
# `landing-gate` commit status on the pushed head → PR → merge → worktree removal. Exits non-zero at the first failing step;
# never force-pushes. Run from anywhere; the worktree must be $SISTEER_REPO/worktrees/<branch>.
#
# The verdict step has exactly three branches and the Slurm gate is the only fall-through — no Modal call exists here:
#   (a) REUSE   a recorded `VERDICT: LAND` whose `MERGE sha=` names this merged tip ($STATE/verdicts/*.out; legacy /tmp/gate-*.out read too),
#               or -- TREE-IDENTICAL REUSE -- one whose gated
#               commit is an ANCESTOR of the merged tip with the IDENTICAL tree: a stacked branch (gated on the merged tip of the branch
#               below it) lands after that branch's PR merged as a GitHub merge commit, and the re-merge mints a new commit whose tree is
#               byte-identical to the gated tip's (main brought nothing the tip did not already contain). The suite ran on that tree, so
#               its verdict holds for this tip; `MERGE sha=<tip8> tree-identical-to=<gated8> tree=<tree8> <ISO>` is appended to the file and
#               the status description says so. Real content landed on main in between changes the tree and the gate runs as before.
#   (b) WAIVER  GATE_WAIVED_BY_USER=<reason>: the full suite is waived for THIS landing (recorded, printed; the live-DB files still run).
#   (c) SLURM   otherwise `gate_slurm.sh <tip> --branch <B> --baseline-sha <origin/main>` in the FOREGROUND (its watcher inside).
#       The baseline for that main is AWAITED, never refused as stale (S6): baseline_refresh.sh is ticked here every 30 s for
#       up to 35 min (it submits the baseline array if the 2-min timer has not, watches it, collects it; its events and the job
#       ids print here) until summary-<main>.json is a TERMINAL summary (PASS/FAIL with counts — the predicate the refresh writes
#       COLLECTED on, never the file's mere presence); a FAILED or GAVE_UP attempt refuses the landing (exit 6).
# Only a GATED COMBINATION merges: right before `gh pr merge`, origin/main must still be the MAIN the tip was merged onto and
# gated against. If main moved meanwhile (the wait, the gate and the live-DB files take an hour), the round repeats — re-merge,
# re-gate (the same three verdict branches; the baseline for the new main is awaited the same way), re-run the live-DB files,
# push the new tip to the same PR — at most 3 rounds (exit 10 when main keeps moving). The check and the merge are ONE step
# under a landing-wide flock ($STATE/merge.lock: this box's chains merge one at a time; exit 12 after 30 min waiting),
# `gh pr merge --merge --match-head-commit <tip>` (the branch head cannot have moved), ITS exit status kept (exit 8 when
# refused), the merge PROVEN before any cleanup -- `gh pr view --json state,mergedAt,mergeCommit` reads MERGED (gh 2.97.0 has no
# `merged` field: the first real landing merged and then exited 8 on that read), else git: a merge commit on origin/main's
# first-parent line back to the checked MAIN has this tip as its second parent (main may have moved on since; the lock covers this box
# only); the line says which proved it -- and the MERGE COMMIT VERIFIED: its first
# parent must be the MAIN this chain checked and its second parent the tip. Another writer between the check and the merge
# (a chain on another box; GitHub merged an out-of-date PR) prints a LOUD `RACED: merged onto <actual> not <checked>` line,
# awaits the baseline for the actual parent, gates the merge commit against it right there (a post-hoc gate, recorded as
# verdicts/gate-land-<B>-posthoc.out) and exits 11 with the worktree left in place — the operator reads it. Repo ruleset 24155420
# ("main: landing gate passed on an up-to-date branch": the commit status `landing-gate` required in state success on the PR head,
# STRICT — the head must contain main's tip — and no bypass actors; added 2026-09-29, activated once every landing
# path posts) closes that cross-session race server-side. So right after the push and before the PR is created or merged, gate_status.sh
# posts `landing-gate` success on the merged head with the evidence as its description, never more than the evidence: `slurm gate LAND on
# tree of <sha8> vs baseline <main8>` for a gate run here; `recorded LAND verdict (<file>) on tree of <sha8> vs baseline <b8>` for a reused
# one (the baseline ITS file names; no clause when it names none); `full suite waived by the user on tree of <sha8>: <reason>` for a waiver
# (no suite ran, no baseline) — every round posts on ITS pushed tip. A failed POST stops the landing BEFORE any merge: exit 13, the
# `status` stamp left without its end, the worktree and the branch in place. Once the ruleset is active, GitHub itself refuses the merge
# of a head that no longer contains main's tip (another writer between the check and the merge): that refusal exits 8 like any other,
# and the line names the cause when origin/main is found moved — head behind main; re-merge and re-gate (land again).
# The no-Modal scan runs inside step 1 after the merge: `check_no_modal.sh --repo <worktree> --mode $LAND_NO_MODAL_SCAN
# [--inventory $LAND_NO_MODAL_INVENTORY]` over the merged tree's registry ownership and the operating executables an inventory names.
# LAND_NO_MODAL_SCAN=report (default) prints the hits and never refuses; `refuse` (the cutover flips it once the inventory records zero
# live hits) exits 14 on any hit, before the verdict; `off` skips the scan. The hit count prints as
# `[land] no-modal scan mode=<m> hits=<n> rc=<rc>`.
# Live-DB step (ON SLURM): `live_db_slurm.sh <tip> --branch <B> --main MAIN` — the tests/test_registry_*.py
# modules declaring `pytestmark = pytest.mark.live_db` (`land.sh --live-db-files` lists them) run on the cluster against the REPLICA: the
# grading host takes a quiescent replayable head of the live database (`registry replica head`, cursor-free, bounded wait), syncs the record-log
# mirror, submits `replica.sbatch land` (-J registry-replica --dependency=singleton; copy of CURRENT, allowlist presence/hash check, the
# increment, `replica check`, pytest with REGISTRY_LIVE_DB=<the copy> and REGISTRY_PIN_L1, publish), watches it in the foreground and
# collects; it refuses (exit 5) unless check.json.equal (or, under LAND_ACCEPT_C1_CLASS -- the cutover allowance -- check.json.c1_class:
# the certified C1 same-stamp class alone, in which case the landing-gate description below carries `; live-DB: C1 same-stamp class only
# (N movers)`, read off the step's state/replica_check.json for THIS tip; the base is spelled compactly when the whole would pass
# gate_status.sh's 140-character limit, and refused (exit 13, before the push) when even that would be truncated -- the evidence, never
# less), pytest rc 0 and skipped == expected. The grading host runs no pytest and never takes the
# pytest lock in a landing. JUnit to $STATE/land/<B>.junit.xml, per-test durations rendered to $STATE/land/<B>.durations.txt
# (measure/durations.py); the replica verdict to $REGISTRY_ROOT/state/replica_check.json. LIVE_DB_SLURM_SH overrides the step's script,
# LIVE_DB_EXPECT_SKIPPED (0) the skips allowed. KNOWN_RED is no longer read (a known-red node is a replica-side fixture, not a deselect).
# Every line is also appended to $STATE/land/<B>.out with `[land] @ <step>:<start|end> <UTC>` stamps and, per repeated round,
# `[land] re-gate round=<n>: …` (measure/landings.py reads both).
# State: STATE=${REGISTRY_LANDING_STATE:-~/.local/state/registry-landing}. Knobs (env): SISTEER_REPO, LAND_PY (the registry python),
# GATE_SLURM_SH, BASELINE_REFRESH_SH, LAND_GH (gh), LAND_REPO (owner/name of the GitHub repository; unset, the slug of the checkout's
# origin remote, the remote every step fetches from and pushes to, when it is a github.com URL, and otherwise refused with exit 2;
# `land.sh --repo` prints the resolved slug — gate_status.sh reads it there, so the repository is named in ONE place),
# LAND_BASELINE_WAIT_MIN (35), LAND_BASELINE_POLL_S (30), LAND_LEGACY_VERDICTS (/tmp/gate-*.out), LAND_POST_MERGE_SLEEP (6),
# LAND_MERGE_LOCK_WAIT_S (1800), LAND_NO_MODAL_SCAN (report|refuse|off), LAND_NO_MODAL_INVENTORY (the operating-scripts inventory JSON),
# CHECK_NO_MODAL_SH (this directory's check_no_modal.sh), LIVE_DB_REPLICA_TREE (recovery, recover.sh: the replica side of the live-DB step
# runs that tree's code; passed through to live_db_slurm.sh by the environment, as is LIVE_DB_DISPATCHER, the chain's replica.sbatch that
# live_db_slurm.sh ships and submits in a recovery instead of any tree's own script).
# Exit codes: 2 usage / no worktree / no repository (LAND_REPO unset and no github.com origin) · 3 merge conflict or failure · 4 gitleaks
# hook missing · 5 a HOLD verdict, no verdict file, or the live-DB step refused · 6 baseline failed, push failed · 7 PR create failed ·
# 8 merge refused · 9 not mergeable · 10 main kept moving · 11 RACED · 12 merge lock · 13 the landing-gate status POST failed · 14 the
# no-Modal scan refused (LAND_NO_MODAL_SCAN=refuse) or could not run under refuse.
set -u
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
STATE=${REGISTRY_LANDING_STATE:-$HOME/.local/state/registry-landing}
VERDICTS=$STATE/verdicts; LAND=$STATE/land; SUMMARIES=$STATE/summaries; BASELINES=$STATE/baselines
# the grading host's checkout (site_env.sh SITE_REPO: SISTEER_REPO, else the site config's, else this repository's main working tree),
# resolved in a subshell so no other site key is exported to the steps below
REPO_DIR=$(source "$HERE/site_env.sh" && site_env_load SITE_REPO && printf '%s\n' "$SITE_REPO") || exit 2
# the GitHub repository the PR is opened and merged in: LAND_REPO (owner/name), else the slug of the checkout's origin remote (the
# remote every step below fetches from and pushes to) when it is a github.com URL. No built-in name: anything else refuses (exit 2)
land_repo() {
  local url slug shown
  if [ -n "${LAND_REPO:-}" ]; then printf '%s\n' "$LAND_REPO"; return 0; fi
  url=$(git -C "$REPO_DIR" remote get-url origin 2>/dev/null) || url=""
  url=${url%/}
  slug=$(sed -nE 's#^(https?://([^@/]+@)?github\.com/|ssh://([^@/]+@)?github\.com/|([^@/]+@)?github\.com:)([^/]+/[^/]+)$#\5#p' <<<"$url")
  slug=${slug%.git}
  if [[ "$slug" =~ ^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$ ]]; then printf '%s\n' "$slug"; return 0; fi
  shown=$(sed -E 's#//[^@/]*@#//#' <<<"$url")   # never echo a credential a URL might carry
  if [ -n "$shown" ]; then shown="an origin remote that is not a github.com repository ($shown)"; else shown="no origin remote"; fi
  echo "land.sh: set LAND_REPO (owner/name of the GitHub repository the landing opens and merges its PR in): the checkout $REPO_DIR" \
       "has $shown" >&2
  return 2
}
REG=${REGISTRY_ROOT:-$HOME/registry}   # read here for ONE file: the live-DB step's verdict, state/replica_check.json (the C1 clause)
PY=${LAND_PY:-$REPO_DIR/.venv/bin/python}
GH=${LAND_GH:-gh}
GATE=${GATE_SLURM_SH:-$HERE/gate_slurm.sh}
REFRESH=${BASELINE_REFRESH_SH:-$HERE/baseline_refresh.sh}
WAIT_MIN=${LAND_BASELINE_WAIT_MIN:-35}; POLL=${LAND_BASELINE_POLL_S:-30}
LEGACY=${LAND_LEGACY_VERDICTS-/tmp/gate-*.out}
MAX_ROUNDS=3   # re-gates when origin/main moves under a gated tip; a third move in one landing refuses (exit 10)
LOCK_WAIT=${LAND_MERGE_LOCK_WAIT_S:-1800}
NO_MODAL_SCAN=${LAND_NO_MODAL_SCAN:-report}; NO_MODAL_INV=${LAND_NO_MODAL_INVENTORY:-}; CHECK_NO_MODAL=${CHECK_NO_MODAL_SH:-$HERE/check_no_modal.sh}

plan() {
  cat <<'PLAN'
PLAN 1 merge :: cd $SISTEER_REPO/worktrees/<branch>; git fetch origin main; git merge --no-edit origin/main (a conflict exits 3); require the gitleaks pre-commit hook; SHA = merged tip, MAIN = origin/main; then check_no_modal.sh scans the merged tree's registry ownership and the operating executables (LAND_NO_MODAL_SCAN=report|refuse|off, default report; refuse exits 14 on a hit, before any verdict)
PLAN 2 verdict(reuse|waiver|slurm-gate) :: (a) reuse a recorded `VERDICT: LAND` whose `MERGE sha=` is the merged tip, or whose gated commit is an ancestor of the merged tip with the IDENTICAL tree (D35: a stacked landing re-merged over the merge commit below it; the file gets a `tree-identical-to=` line); (b) GATE_WAIVED_BY_USER records a waiver; (c) otherwise gate_slurm.sh <tip> --branch <B> --baseline-sha MAIN in the foreground, after awaiting the baseline for MAIN (baseline_refresh.sh ticked every 30 s, up to 35 min) — the only fall-through; no Modal
PLAN 3 live-db :: live_db_slurm.sh <tip> --branch <B> --main MAIN: `registry replica head` on the grading host (one read transaction at a cursor-free boundary, bounded wait; exit 6 defers once to the harvester's next completed cycle) -> $STATE/land/<B>.head.json.gz; a record-log mirror pass that STARTS after the head (registry-mirrors.service: the running pass waited out, a fresh one started and verified from its own record, D143); the tip's tree on the cluster; sbatch replica.sbatch land <tree> <head> <venv> <B> (-J registry-replica --dependency=singleton, 16 CPU / 32 G / --time 01:30:00): node-local copy of CURRENT -> every allowlisted file present with the head's sha -> ingest the increment -> replica check (frame_projection/1) -> pytest -m live_db <the tests/test_registry_*.py modules declaring pytestmark live_db> with REGISTRY_LIVE_DB=<the copy> and REGISTRY_PIN_L1 -> publish; watch-job.sh in the foreground; the job's terminal state read from sacct; collect $SHARED/gates/<tip>/live_db/<submission>/{check.json,junit.xml,summary.json} (a directory minted per submission, never reused) -> $STATE/land/<B>.{junit.xml,pytest.log}; refuse (exit 5) unless the job COMPLETED, check.json and summary.json both name this job, tip and head, check.json.equal (or check.json.c1_class under the declared LAND_ACCEPT_C1_CLASS allowance, D136 -- the clause then rides into the landing-gate description), pytest rc 0, skipped == expected and the published generation is what CURRENT names; the grading host runs no pytest; durations.py renders $STATE/land/<B>.durations.txt
PLAN 4 push :: git push origin HEAD:<branch>; then gate_status.sh <tip> "<evidence>" posts the `landing-gate` success commit status on the pushed head (ruleset 24155420 requires it on an up-to-date head before main accepts the PR); a failed POST exits 13 before any PR or merge
PLAN 5 pr :: gh pr create --base main --head <branch> (once; a repeated round pushes its tip to the same PR); wait for a MERGEABLE state; then, under the landing-wide flock $STATE/merge.lock, origin/main must still be MAIN — if it moved, repeat 1–5 (at most 3 rounds): only a gated combination merges
PLAN 6 merge :: still under the lock: gh pr merge <n> --merge --match-head-commit <tip> with its exit status kept (a refusal exits 8), the merge proven by gh pr view --json state,mergedAt,mergeCommit reading MERGED or else by git (a merge commit on origin/main's first-parent line back to MAIN has the tip as its second parent; the proving path is logged), and the merge commit's first parent must be MAIN (its second the tip) — else RACED: a post-hoc gate of the merge commit against its actual parent, exit 11, no cleanup
PLAN 7 cleanup :: git fetch origin main in the base checkout; git worktree remove worktrees/<branch>; git branch -D <branch>
PLAN
}
live_db_files() {  # the tests/test_registry_*.py modules that declare the marker at module level (run from the tree's root)
  grep -lE '^pytestmark\s*=\s*pytest\.mark\.live_db' tests/test_registry_*.py 2>/dev/null
}
summary_terminal() {  # summary_terminal FILE → 0 when it is a terminal gate summary (PASS/FAIL with counts): the predicate baseline_refresh.sh writes COLLECTED on
  [ -s "$1" ] && python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); sys.exit(0 if d.get("status") in ("PASS","FAIL") and d.get("counts") else 1)' "$1" 2>/dev/null
}
await_baseline() {  # await_baseline MAIN — tick baseline_refresh.sh until summary-<MAIN>.json is TERMINAL; 1 on a FAILED / GAVE_UP attempt or the deadline
  local main=$1 m8=${1:0:8} summary=$SUMMARIES/summary-$1.json job=$BASELINES/$1.job deadline announced="" jl
  deadline=$(( $(date +%s) + WAIT_MIN * 60 ))
  echo "[land] awaiting baseline $m8: $REFRESH ticks every ${POLL}s for up to $WAIT_MIN min; its events print here"
  until summary_terminal "$summary"; do
    "$REFRESH" --sha "$main" --quiet
    summary_terminal "$summary" && break
    if [ -f "$job" ]; then
      jl=$(grep -E '^(ARRAY|FINALIZE|JOB_STATE|ATTEMPT)=' "$job" | tr '\n' ' ')
      [ -n "$announced" ] || { echo "[land] baseline $m8 job: $jl"; announced=1; }
      case "$jl" in *JOB_STATE=FAILED*|*JOB_STATE=GAVE_UP*) echo "BASELINE for main $m8 $jl— see $BASELINES/journal.log"; return 1 ;; esac
    fi
    [ "$(date +%s)" -lt "$deadline" ] || { echo "BASELINE for main $m8 not collected within $WAIT_MIN min (see $BASELINES/journal.log)"; return 1; }
    sleep "$POLL"
  done
}
case "${1:-}" in
  --plan) plan; exit 0 ;;
  --live-db-files) live_db_files; exit 0 ;;
  --repo) land_repo; exit $? ;;
  -h|--help) awk 'NR == 1 {next} /^#/ {print; next} {exit}' "$0"; exit 0 ;;
esac
[ $# -eq 3 ] || { echo "usage: land.sh <branch> <title> <body-file> | land.sh --plan | land.sh --live-db-files | land.sh --repo" >&2; exit 2; }
B=$1; TITLE=$2; BODY=$3; WT=$REPO_DIR/worktrees/$B
REPO=$(land_repo) || exit 2
mkdir -p "$VERDICTS" "$LAND" || { echo "cannot create $STATE"; exit 2; }
exec > >(tee -a "$LAND/$B.out") 2>&1
utc() { date -u +%FT%TZ; }
stamp() { echo "[land] @ $1:$2 $(utc)"; }

round=1; n=""
while :; do
  # 1. merge
  stamp merge start
  cd "$WT" || { echo "no worktree $WT (the base checkout is SISTEER_REPO, else the site config's, else this repository's main working tree)"; exit 2; }
  git fetch -q origin main || exit 2
  if ! git merge --no-edit origin/main >"$LAND/$B.merge.log" 2>&1; then
    if [ -n "$(git diff --name-only --diff-filter=U)" ]; then echo "CONFLICT: $(git diff --name-only --diff-filter=U | tr '\n' ' ') — resolve with the branch's worker"; exit 3; fi
    # any other merge failure (a dirty worktree, an unrelated history) leaves HEAD un-merged: the gate must never judge that tip as the merged one
    echo "MERGE FAILED without conflict markers — see $LAND/$B.merge.log: $(tail -n 2 "$LAND/$B.merge.log" | tr '\n' ' ')"; exit 3
  fi
  [ -f "$(git rev-parse --git-path hooks/pre-commit)" ] || { echo "gitleaks hook missing"; exit 4; }
  SHA=$(git rev-parse HEAD); S8=${SHA:0:8}; MAIN=$(git rev-parse origin/main); M8=${MAIN:0:8}
  echo "[land] branch=$B merged tip=$S8 main=$M8 round=$round $(date -u +%T)"
  # the no-Modal scan of the merged tree (S3/S9; C9): report prints and goes on; refuse stops here, before any verdict; off skips
  if [ "$NO_MODAL_SCAN" != off ]; then
    SCAN_OUT=$("$CHECK_NO_MODAL" --repo "$WT" --mode "$NO_MODAL_SCAN" ${NO_MODAL_INV:+--inventory "$NO_MODAL_INV"} 2>&1); src=$?
    NH=$(grep -c '^HIT ' <<<"$SCAN_OUT" || true)
    # the HIT lines themselves (the first 60), then the scan's summary line: a refusal must name what it refused
    grep '^HIT ' <<<"$SCAN_OUT" | head -n 60
    [ "$NH" -le 60 ] || echo "[land] ... $((NH - 60)) more HIT line(s) (run check_no_modal.sh --repo $WT for all of them)"
    grep -v '^HIT ' <<<"$SCAN_OUT" | tail -n 1
    echo "[land] no-modal scan mode=$NO_MODAL_SCAN hits=$NH rc=$src"
    if [ "$src" -eq 1 ]; then echo "NO-MODAL SCAN REFUSED merged tip $S8: $NH hit(s) under LAND_NO_MODAL_SCAN=refuse (the HIT lines are above) — NOT pushed"; exit 14; fi
    if [ "$src" -ne 0 ]; then
      if [ "$NO_MODAL_SCAN" = refuse ]; then echo "NO-MODAL SCAN could not run (rc=$src) under LAND_NO_MODAL_SCAN=refuse — NOT pushed"; exit 14; fi
      echo "[land] no-modal scan could not run (rc=$src): $(tail -n 1 <<<"$SCAN_OUT") — report mode goes on"
    fi
  fi
  stamp merge end

  # 2. verdict — reuse | waiver | the Slurm gate (the only fall-through)
  stamp verdict start
  # shellcheck disable=SC2086
  V=$(grep -lE "^VERDICT: LAND" "$VERDICTS"/*.out $LEGACY 2>/dev/null | xargs -r grep -lE "^MERGE sha=$S8" 2>/dev/null | head -n 1)
  # tree-identical reuse (header): a recorded LAND whose gated commit is an ancestor of this merged tip with the identical tree
  tree_identical_verdict() { # <candidate commit> -> prints the verdict file whose gated commit is an ancestor with the identical tree
    local c=$1 t f g8 gsha; t=$(git rev-parse "$c^{tree}") || return 1
    # shellcheck disable=SC2086
    for f in $(grep -lE "^VERDICT: LAND" "$VERDICTS"/*.out $LEGACY 2>/dev/null); do
      g8=$(sed -nE 's/^MERGE sha=([0-9a-f]{8,40}).*/\1/p' "$f" | head -n 1); [ -n "$g8" ] || continue
      gsha=$(git rev-parse --verify -q "$g8^{commit}" 2>/dev/null) || continue
      git merge-base --is-ancestor "$gsha" "$c" 2>/dev/null || continue
      [ "$(git rev-parse "$gsha^{tree}")" = "$t" ] || continue
      echo "$f"; return 0
    done; return 1; }
  TI=""
  if [ -z "$V" ] && [ -z "${GATE_WAIVED_BY_USER:-}" ] && V=$(tree_identical_verdict HEAD); then
    TI=$(sed -nE 's/^MERGE sha=([0-9a-f]{8}).*/\1/p' "$V" | head -n 1); TREE8=$(git rev-parse "HEAD^{tree}" | cut -c1-8)
    echo "[land] gate verdict LAND recorded for gated tip $TI in $V; merged tip $S8 has the IDENTICAL tree $TREE8 — reusing (D35)"
    echo "MERGE sha=$S8 tree-identical-to=$TI tree=$TREE8 $(utc)" >> "$V"
  fi
  if [ -n "${GATE_WAIVED_BY_USER:-}" ]; then
    # the full suite is waived for THIS landing; the live-DB test files below still run; the caller records the waiver
    echo "[land] FULL SUITE WAIVED BY THE USER for merged tip $S8: ${GATE_WAIVED_BY_USER} $(date -u +%T)"
    echo "MERGE sha=$S8 VERDICT: WAIVED-BY-USER (${GATE_WAIVED_BY_USER})" > "$VERDICTS/gate-land-$B.out"
    echo "[land] verdict=waiver"
    DESC="full suite waived by the user on tree of $S8: ${GATE_WAIVED_BY_USER}"   # no suite ran: no baseline is named
  elif [ -n "$V" ]; then
    echo "[land] gate verdict LAND already recorded for merged tip $S8 in $V — reusing"
    echo "[land] verdict=reuse"
    # labeled by its recorded source, not as a Slurm run (a legacy verdict file is accepted too), against the baseline THAT verdict names
    # (its own BASELINE line; the clause is dropped when the file has none — never the current main guessed in)
    RB=$(sed -n 's/^BASELINE sha=\([0-9a-f]\{8\}\).*/\1/p' "$V" | head -n 1)
    DESC="recorded LAND verdict ($(basename "$V")) on tree of $S8${TI:+ (verdict of $TI, identical tree)}${RB:+ vs baseline $RB}"
  else
    echo "[land] slurm gate for merged tip $S8 against baseline main=$M8 $(date -u +%T)"
    echo "[land] verdict=slurm-gate"
    # the baseline for MAIN: awaited (the refresh tick submits, watches and collects it), never refused as stale. Ready = a TERMINAL
    # summary, never the file's presence: gate_slurm.sh installs a non-terminal summary (a TIMEOUT, missing shards) before it refuses it
    stamp baseline-wait start
    await_baseline "$MAIN" || { echo "NOT pushed"; exit 6; }
    stamp baseline-wait end
    "$GATE" "$SHA" --branch "$B" --baseline-sha "$MAIN"; grc=$?
    OUTF="$VERDICTS/gate-land-$B.out"
    [ -s "$OUTF" ] || { echo "GATE produced no verdict file $OUTF (rc=$grc) — NOT pushed"; exit 5; }
    grep -E "VERDICT|INTRODUCED" "$OUTF" | tail -n 3
    grep -q "^MERGE sha=$S8" "$OUTF" || { echo "GATE verdict in $OUTF is not for merged tip $S8 — NOT pushed"; exit 5; }
    grep -q "^VERDICT: LAND" "$OUTF" || { echo "GATE NOT LAND (see $OUTF) — NOT pushed"; exit 5; }
    DESC="slurm gate LAND on tree of $S8 vs baseline $M8"
  fi
  stamp verdict end

  # 3. ON SLURM: the registry tests that read the live database (marked live_db) run against the cluster replica
  # from the merged tip's exported tree -- live_db_slurm.sh takes the head here, syncs the mirror, submits and watches `replica.sbatch land`,
  # collects the verdict; the grading host runs no pytest. Its stamps (`live-db-<step>`) sit inside this step's.
  stamp live-db start
  LIVE_FILES=$(live_db_files | tr '\n' ' ')
  N_LIVE=$(echo $LIVE_FILES | wc -w)
  echo "[land] live-DB test files on merged tip $S8 ($N_LIVE files, -m live_db) $(date -u +%T)"
  rm -f "$LAND/$B.junit.xml"
  "${LIVE_DB_SLURM_SH:-$HERE/live_db_slurm.sh}" "$SHA" --branch "$B" --main "$MAIN" --expect-skipped "${LIVE_DB_EXPECT_SKIPPED:-0}"; rc=$?
  if [ -s "$LAND/$B.junit.xml" ]; then
    "$PY" "$HERE/../measure/durations.py" "$LAND/$B.junit.xml" > "$LAND/$B.durations.txt" 2>&1 && { head -n 6 "$LAND/$B.durations.txt"; echo "[land] per-test durations: $LAND/$B.durations.txt"; }
  fi
  [ $rc -eq 0 ] || { echo "LIVE-DB STEP ON SLURM FAILED (rc=$rc) — see $LAND/$B.live_db/ and $LAND/$B.pytest.log; NOT pushed"; exit 5; }
  # the cutover allowance: when the step accepted its check as equal modulo the certified C1 same-stamp class, the landing-gate
  # description carries that clause -- read from the verdict the step just wrote, and only when that verdict is THIS landing's (its tip
  # and branch), never from a stale file; the mover count is the evidence and the whole of it
  C1_CLAUSE=$("$PY" - "$REG/state/replica_check.json" "$SHA" "$B" 2>/dev/null <<'PYC' || true
import json, sys
try:
    d = json.load(open(sys.argv[1], encoding="utf-8"))
except Exception:
    sys.exit(0)
if d.get("tip") == sys.argv[2] and d.get("branch") == sys.argv[3] and d.get("accepted_as") == "c1_class":
    print(f"; live-DB: C1 same-stamp class only ({d.get('n_movers')} movers)")
PYC
)
  if [ -n "$C1_CLAUSE" ]; then
    # gate_status.sh truncates a description past 140 characters, which would cut the clause -- the class and the count -- off the posted
    # evidence. So the base is spelled compactly when the whole would not fit (the same facts, shorter words), and a
    # description that still would not fit is REFUSED here rather than posted truncated: the status carries the evidence, never less
    FULL="$DESC$C1_CLAUSE"
    if [ ${#FULL} -gt 140 ]; then
      FULL=$(sed -e 's/^recorded LAND verdict (/recorded LAND (/' -e 's/ (verdict of \([0-9a-f]*\), identical tree)/ (=\1)/' -e 's/ on tree of / on /' \
                 -e 's/ vs baseline / vs base /' -e 's/^full suite waived by the user on/waived by the user on/' <<<"$DESC")$C1_CLAUSE
    fi
    [ ${#FULL} -le 140 ] || { echo "LANDING-GATE DESCRIPTION WOULD BE TRUNCATED (${#FULL} > 140 characters): $FULL — the C1 clause would be cut off the evidence; NOT pushed"; exit 13; }
    DESC=$FULL
    echo "[land] live-DB check accepted under the declared C1 allowance; the landing-gate description carries it (${#DESC} characters)"
  fi
  stamp live-db end

  # 4. push → 5. PR (created once; a repeated round's push updates it)
  stamp push start
  git push -q origin HEAD:"$B" || exit 6
  echo "[land] pushed $(git rev-parse --short HEAD)"
  stamp push end
  # the `landing-gate` commit status on the pushed head, BEFORE the PR exists or merges: ruleset 24155420 requires it (strict: this head
  # contains main's tip, since main was just merged into it) and gate_status.sh posts it with the evidence as the description. A failed
  # POST stops the landing here — nothing is merged, the worktree and the branch stay (a repeated round re-posts on its new tip)
  stamp status start
  if ! LAND_GH="$GH" LAND_REPO="$REPO" "$HERE/gate_status.sh" "$SHA" "$DESC"; then
    echo "LANDING-GATE STATUS NOT POSTED on merged tip $S8 (gate_status.sh failed) — NOT merged; the worktree and the branch stay"; exit 13
  fi
  stamp status end
  stamp pr start
  if [ -z "$n" ]; then
    url=$("$GH" pr create --base main --head "$B" --title "$TITLE" --body-file "$BODY" 2>&1 | tail -n 1); echo "$url"; n=${url##*/}
    [[ "$n" =~ ^[0-9]+$ ]] || { echo "PR create failed: $url"; exit 7; }
  else
    echo "[land] PR $n now carries merged tip $S8"
  fi
  for i in 1 2 3 4 5 6; do st=$("$GH" pr view "$n" -R "$REPO" --json mergeable,mergeStateStatus --jq '"\(.mergeable) \(.mergeStateStatus)"'); case "$st" in UNKNOWN*) sleep 10;; *) break;; esac; done
  echo "PR $n: $st"
  stamp pr end
  # only a gated combination merges, and this box's chains merge one at a time: the lock covers the main check, the merge and the
  # verification of what was merged onto (another chain's merge between our check and our merge is the race the verification catches)
  exec 8>"$STATE/merge.lock"
  flock -w "$LOCK_WAIT" 8 || { echo "MERGE LOCK $STATE/merge.lock held by another landing for $LOCK_WAIT s — NOT merged; the worktree and the branch stay"; exit 12; }
  git fetch -q origin main || exit 6
  NOW=$(git rev-parse origin/main)
  [ "$NOW" = "$MAIN" ] && break
  flock -u 8
  round=$((round + 1))
  [ "$round" -le "$MAX_ROUNDS" ] || { echo "origin/main moved again (${NOW:0:8}) after $((round - 1)) gated rounds — NOT merged; land again when main settles"; exit 10; }
  echo "[land] re-gate round=$round: origin/main moved $M8 → ${NOW:0:8} while merged tip $S8 was gated — re-merging, re-gating and re-running the live-DB files; the PR merges only a gated combination"
done

# 6. merge — still under the lock: the merge's own exit status decides, the PR must read merged, and the merge commit must sit on the
# MAIN this chain checked (first parent) with this tip as its second parent, before anything is cleaned up
stamp merge-pr start
case "$st" in MERGEABLE*) ;; *) echo "NOT MERGEABLE ($st)"; exit 9 ;; esac
mout=$("$GH" pr merge "$n" -R "$REPO" --merge --match-head-commit "$SHA" 2>&1); mrc=$?; tail -n 1 <<<"$mout"
if [ $mrc -ne 0 ]; then
  # the ruleset's strict condition refuses a head that no longer contains main's tip: when origin/main moved between the check and the
  # merge, name that cause (the re-gate rounds above catch a move seen at fetch time; a move GitHub saw first is landed again, from the merge)
  git fetch -q origin main 2>/dev/null; NOW=$(git rev-parse origin/main 2>/dev/null)
  if [ -n "${NOW:-}" ] && [ "$NOW" != "$MAIN" ]; then
    echo "PR MERGE REFUSED (gh rc=$mrc): origin/main moved $M8 → ${NOW:0:8} between the check and the merge — head behind main; re-merge and re-gate (land again) — NOT merged; the worktree and the branch stay"
  else
    echo "PR MERGE REFUSED (gh rc=$mrc) — NOT merged; the worktree and the branch stay"
  fi
  exit 8
fi
# the PR's reading, in the fields the installed gh (2.97.0) has: state / mergedAt / mergeCommit -- `merged` is NOT one of them (an
# earlier shim answered it; the first real landing merged and then exited 8 at this read). Placeholders (`-`) keep the three columns
# apart for `read` when a field is null; a gh that prints nothing or errors leaves them empty.
read -r pstate merged_at MC <<<"$("$GH" pr view "$n" -R "$REPO" --json state,mergedAt,mergeCommit --jq '"\(.state // "-") \(.mergedAt // "-") \(.mergeCommit.oid // "-")"' 2>/dev/null)"
[ "${MC:-}" != - ] || MC=""
git fetch -q origin main || exit 6
if [ "${pstate:-}" = MERGED ] && [ -n "${merged_at:-}" ] && [ "$merged_at" != - ] && [ -n "$MC" ]; then
  echo "[land] merge proven by gh: state=MERGED mergedAt=$merged_at mergeCommit=${MC:0:8}"
else
  # gh read nothing, erred, or does not read MERGED (yet): git is the authority. A merge that happened put a merge commit on origin/main
  # whose SECOND parent is this tip -- the chain's own rule; the first parent is verified below exactly as for the gh reading (a RACED
  # merge is still found here and reported as such, never as "not merged"). The merge commit need not be main's TIP: a chain on another
  # box may have merged a later PR between this merge and this fetch (the lock covers this box only), so origin/main's first-parent line
  # is walked back to the checked MAIN (bounded). A `gh pr merge` that exited 0 without merging leaves no such commit and is refused here.
  NOW=$(git rev-parse --verify -q origin/main); FOUND=""; later=0
  for c in $(git rev-list --first-parent --max-count=100 "$MAIN..${NOW:-$MAIN}" 2>/dev/null); do
    # --verify -q: nothing, not the argument echoed back, when a commit has no second parent
    if [ "$(git rev-parse --verify -q "$c^2")" = "$SHA" ]; then FOUND=$c; break; fi
    later=$((later + 1))
  done
  if [ -n "$FOUND" ]; then
    MC=$FOUND
    echo "[land] merge proven by git: gh read state=${pstate:--} mergedAt=${merged_at:--}; origin/main ${NOW:0:8} carries merge commit ${MC:0:8} with this tip $S8 as its second parent ($later later commit(s) on main's first-parent line)"
  else
    echo "PR $n does not read merged after gh pr merge (gh: state=${pstate:--} mergedAt=${merged_at:--} mergeCommit=${MC:--}; no commit on origin/main's first-parent line from ${NOW:0:8} back to $M8 has this tip $S8 as its second parent) — the worktree and the branch stay"
    exit 8
  fi
fi
ONTO=$(git rev-parse "$MC^1" 2>/dev/null) && P2=$(git rev-parse "$MC^2" 2>/dev/null) || { ONTO=""; P2=""; }
flock -u 8
if [ "$ONTO" != "$MAIN" ] || [ "$P2" != "$SHA" ]; then
  echo "RACED: merged onto ${ONTO:-?} not $MAIN; the combination was not gated (merge commit ${MC:-unknown}, second parent ${P2:-?}, this tip $SHA) — another writer moved main between the check and the merge"
  echo "[land] post-hoc gate: the baseline for ${ONTO:0:8}, then merge commit ${MC:0:8} against it — recorded as $VERDICTS/gate-land-$B-posthoc.out"
  stamp posthoc start
  if [ -n "$ONTO" ] && [ -n "$MC" ] && await_baseline "$ONTO"; then
    "$GATE" "$MC" --branch "$B-posthoc" --baseline-sha "$ONTO"
    grep -E "^(MERGE|VERDICT)" "$VERDICTS/gate-land-$B-posthoc.out" 2>/dev/null
  else
    echo "post-hoc gate NOT run (merge commit or its parent unknown, or no baseline for ${ONTO:0:8})"
  fi
  stamp posthoc end
  echo "RACED landing of $B as PR $n: main carries ${MC:0:8}, gated only AFTER the merge — the worktree and the branch stay for the operator"
  exit 11
fi
echo "[land] merged onto $M8 as ${MC:0:8} (first parent verified; second parent $S8)"
stamp merge-pr end

# 7. cleanup
stamp cleanup start
sleep "${LAND_POST_MERGE_SLEEP:-6}"; cd "$REPO_DIR" && git fetch -q origin main && echo "[land] origin/main $(git rev-parse --short origin/main)" && git worktree remove --force "worktrees/$B" && git branch -D "$B" | tail -n 1
stamp cleanup end
echo "[land] LANDED $B as PR $n (tip $S8, $round round$([ "$round" -eq 1 ] || echo s)) $(utc)"
