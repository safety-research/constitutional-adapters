#!/bin/bash
# reproduce/lib/slurm.sh — the shell face of Slurm submission (source after common.sh). Every cluster submission of the
# front door goes through `slurm_submit` -> lib/submit_twin.py, which mirrors the sha, composes the sbatch line exactly
# as the wave shell's printers do (a `cd <checkout> && VAR=value ... sbatch --qos --partition [--account] [--time]
# --output -J <name> <extra> <script> <positionals>` inside ONE ssh string — never --export=), prints
# `submitted <cluster> <jobid> <unit>`, appends submissions.jsonl, launches lib/watch_job.sh and, under --twin,
# runs the START-GRANT election (§3.9). The ORIGINAL sbatch file is always the sbatch target.
slurm_py(){ echo "$(repro_python) $REPRO_LIB_DIR/submit_twin.py"; }
slurm_common_flags(){
  printf -- '--model %q --config %q' "${REPRO_MODEL:?REPRO_MODEL}" "${SISTEER_CONFIG:?SISTEER_CONFIG}"
  [ "${REPRO_DRY_RUN:-0}" = 1 ] && printf ' --dry-run'
  [ -n "${SISTEER_RUN_OVERRIDES:-}" ] && printf -- ' --overrides %q' "$SISTEER_RUN_OVERRIDES"
  return 0
}
slurm_submit(){   # --unit U --lwid L --script S --name N --args "…" --clusters a,b [--twin] [--env "K=V …"] [--time T] [--array A] [--kind K] [--group G]
  eval "$(slurm_py) $(slurm_common_flags) submit $(printf ' %q' "$@")"
}
slurm_cancel_attempt(){ eval "$(slurm_py) $(slurm_common_flags) cancel-attempt $(printf ' %q' "$@")"; }
slurm_prove_resubmit(){ eval "$(slurm_py) $(slurm_common_flags) prove-resubmit $(printf ' %q' "$@")"; }
slurm_binding_state(){ eval "$(slurm_py) $(slurm_common_flags) binding $(printf ' %q' "$@")"; }
slurm_reconcile_token(){ eval "$(slurm_py) $(slurm_common_flags) reconcile-token $(printf ' %q' "$@")"; }   # READ-WRITE: the funnel's own token reconciliation for one unit (--lwid L): `token<TAB>none|adopted|cleared` + the binding line, or `token<TAB>refused<TAB>why` (rc 3)   # READ-ONLY: `binding<TAB>state<TAB>cluster<TAB>job<TAB>token yes|no` (--lwid L)   # READ-ONLY: exit 0 + `bound_job<TAB>cluster<TAB>job` only when a re-mint is authorized
slurm_teardown_pool(){ eval "$(slurm_py) $(slurm_common_flags) teardown-pool $(printf ' %q' "$@")"; }
slurm_reconcile(){ eval "$(slurm_py) $(slurm_common_flags) reconcile $(printf ' %q' "$@")"; }
slurm_twin_flag(){ [ "${TWIN:-0}" = 1 ] && echo --twin; return 0; }   # the stage's --twin, forwarded to slurm_submit
# the pinned sha on EVERY target cluster BEFORE the first submission:
# lib/mirror_repo.sh pushes the commit to the cluster's mirror and verifies the clean detached checkout at <eval root>/<sha>
# (idempotent: an existing clean checkout at the sha is verified and left alone). A dirty HEAD was refused before this by
# repro_require_clean_tree; a sha that is not a commit of this repo is refused by mirror_repo.sh itself. Under --dry-run the
# mirror prints its own lines and touches nothing.
slurm_mirror_clusters(){   # <sha> <comma list of labels>
  local sha="$1" l
  for l in ${2//,/ }; do
    "$REPRO_LIB_DIR/mirror_repo.sh" --cluster "$l" --sha "$sha" --config "$SISTEER_CONFIG" $(repro_dry_flag) \
      || repro_fatal "the pinned sha $sha is not mirrored/verified on cluster $l (lib/mirror_repo.sh refused above) — nothing submitted"
  done
  slurm_stage_geometry "$2"   # the site's serving-geometry records on EVERY target cluster BEFORE the first submission
}
# THE SERVING-GEOMETRY RECORDS TRAVEL WITH THE FRONT DOOR: the serving probe folds its per-set geometry into the overlay
# of the cluster it ran on, and 03_hpo.sh's fetch copies the record into the SITE's overlay dir
# ($SISTEER_SITE_DIR/geometry/<model_key>.json). A padded / sampled unit twinned to the OTHER cluster reads THAT cluster's
# $SHARED/exp/reproduce/geometry (geometry.site_geometry_dir under the job env's SISTEER_SHARED_ROOT), so every
# submission first stages the site's records to every target cluster.
#
# SITE-COPY VALIDITY IS A STAGING-TIME INVARIANT: a site copy is only a copy — the record it came
# from may have been reopened (`run --resume --replan`, a re-run) since 03 fetched it. So BEFORE anything is staged, each site
# overlay is verified against its AUTHORITATIVE record: its provenance names the probe cluster (probe_cluster) and the record
# dir (probe_record) and carries the result_generation it was folded over; the verifying status
# (scripts/sdfbase/hpo_probe_status.py --record <dir> --job-active <squeue>) runs OVER SSH on that cluster, and the file is
# staged ONLY when it says READY and the record's current generation equals the overlay's stamp. Anything else refuses by
# name — a record that cannot be reached included (an authoritative source that cannot be read is an error, never a silent
# accept). The staged bytes are the site copy PLUS a top-level `staging_verification` block {verified_utc, record_generation,
# verifier_host} (geometry.staged_site_geometry_bytes, the ONE serializer): the cluster-side consumer
# (geometry.check_overlay_generation) accepts an overlay whose record it cannot reach ONLY with that block for the overlay's own
# generation, so nothing on the grading host and no unverified copy anywhere reads as valid. The compare with a copy already on a
# cluster is over CANONICAL content (the parsed JSON re-dumped with sorted keys and no whitespace, the block excluded — the
# second belt): formatting never masquerades as tampering; a copy already there with the same content but without the block
# for this generation is RE-STAMPED; any CONTENT difference still refuses by name. No site records = nothing to stage.
# THE PAYLOAD TRANSFER (2026-10-01): slurm_stage_geometry used to inline the staged record as
# base64 INSIDE the ssh remote command, so a v2 record of 178,775 bytes (the 94,705-byte one worked) exceeded Linux's ~128 KiB
# single-argument limit — `ssh: Argument list too long` (rc 126), nothing staged, REVOKED markers not cleared, every staging of the
# record failing the same way. The bytes now travel over ssh STDIN from a local temp file; the remote command carries only the
# small compare / install script, counts the bytes and hashes them, and refuses by name when either differs from what was sent.
# repro_ssh deliberately never reads stdin (`ssh -n`; its callers sit inside `while read` loops), so this is its own seam, with
# the same `local:<dir>` test path (`bash -c` on this host, stdin from the file).
repro_ssh_stdin(){   # <label> <remote shell string> <file>: run on that cluster's login host with the FILE on the remote's stdin
  local host; host="$(cluster_var "$1" CLUSTER_SSH)"; [ -n "$host" ] || repro_fatal "cluster $1 has no CLUSTER_SSH"
  [ -f "$3" ] || repro_fatal "repro_ssh_stdin: no payload file $3"
  if [ "${REPRO_DRY_RUN:-0}" = 1 ]; then echo "DRY-RUN: ssh $host '$2' < $3 ($(wc -c < "$3") bytes)"; return 0; fi
  case "$host" in local:*) FAKE_SLURM_DIR="${host#local:}" bash -c "$2" < "$3"; return $?;; esac
  ssh -o BatchMode=yes -o ConnectTimeout=30 "$host" "$2" < "$3"
}
slurm_stage_geometry(){   # <comma list of labels>
  local dir="$SISTEER_SITE_DIR/geometry" l f name sha csha shared remote out b64 rc state detail rgen pjson
  local canon='import hashlib,json,sys; d=json.load(open(sys.argv[1])); d.pop("staging_verification",None); print(hashlib.sha256(json.dumps(d,sort_keys=True,separators=(",",":"),ensure_ascii=True).encode("utf-8")).hexdigest())'
  # hb = the block's generation (a copy staged through this step), hp = the copy's own provenance stamp (the probe cluster's fold copy)
  # hw = WHEN the copy was folded (written_utc — the fold's own stamp, never the file's mtime): a cluster copy folded AFTER the site copy
  # is a newer certification the site has not fetched yet, and is never overwritten
  local havegen='import json,sys; d=json.load(open(sys.argv[1])); sv=d.get("staging_verification") or {}; print((sv.get("record_generation") or "-") + " " + (((d.get("provenance") or {}).get("result_generation")) or "-") + " " + (str(d.get("written_utc") or "-").replace(" ","_")) + " " + ("1" if sv.get("record_state") and sv.get("parity_check") else "0"))'
  local files=(); for f in "$dir"/*.json; do [ -f "$f" ] || continue; case "$f" in *.REVOKED.json) continue;; esac; files+=("$f"); done   # markers are not overlays
  if [ ${#files[@]} -eq 0 ]; then repro_say "geometry: no site records under $dir — nothing to stage (a padded unit refuses an un-probed model by name)"; return 0; fi
  local status_py="$REPRO_REPO/scripts/sdfbase/hpo_probe_status.py"
  [ -f "$status_py" ] || repro_fatal "geometry: $status_py is missing — the site overlays cannot be verified against their records; nothing submitted"
  local status_b64; status_b64="$(base64 -w0 < "$status_py")"
  for f in "${files[@]}"; do
    name="$(basename "$f")"
    # 1. the overlay's provenance: the probe cluster, the record dir and the generation stamp — all three, or no staging
    local pinfo pcluster precord pgen pmodel pwutc pepoch
    pinfo="$("$(repro_python)" -c 'import json,sys
d=json.load(open(sys.argv[1])); p=d.get("provenance") or {}
print("\t".join(str(p.get(k) or "-") for k in ("probe_cluster","probe_record","result_generation")) + "\t" + str(d.get("model_key") or "-") + "\t" + str(d.get("written_utc") or "-") + "\t" + str(p.get("reopen_epoch") or "-"))' "$f")" \
      || repro_fatal "geometry: $f is not a JSON record — refusing to stage it"
    IFS=$'\t' read -r pcluster precord pgen pmodel pwutc pepoch <<<"$pinfo"      # "-" marks an absent field (tab is IFS whitespace: an empty field would be dropped)
    for v in pcluster precord pgen pmodel pwutc pepoch; do [ "${!v}" != "-" ] || printf -v "$v" ''; done
    [ -n "$pcluster" ] || repro_fatal "geometry: $name carries no provenance.probe_cluster — the probe that folded it did not record which cluster holds its record, so the site copy cannot be verified against it; NOT staged (re-fold on the probe cluster: \`serving_probe.py reverdict --record <dir> --probe-cluster <label>\`, then 03_hpo.sh --phase probe refreshes the site copy); nothing submitted"
    [ -n "$precord" ] || repro_fatal "geometry: $name carries no provenance.probe_record — NOT staged; nothing submitted"
    [ -n "$pgen" ] || repro_fatal "geometry: $name carries no provenance.result_generation stamp — an unstamped overlay cannot be verified against its record; NOT staged (re-fold with the current serving_probe.py); nothing submitted"
    [ -n "$pwutc" ] || repro_fatal "geometry: $name carries no written_utc — the fold's own timestamp is what orders it against a cluster's copy (D206); NOT staged (re-fold with the current serving_probe.py); nothing submitted"
    case " $SISTEER_CLUSTERS " in *" $pcluster "*) ;; *) repro_fatal "geometry: $name names probe_cluster '$pcluster', not one of SISTEER_CLUSTERS ($SISTEER_CLUSTERS) — its record cannot be reached from this site; NOT staged; nothing submitted";; esac
    csha="$("$(repro_python)" -c "$canon" "$f")" || repro_fatal "geometry: $f is not a JSON record — refusing to stage it"
    if [ "${REPRO_DRY_RUN:-0}" = 1 ]; then
      for l in ${1//,/ }; do
        shared="$(cluster_var "$l" CLUSTER_SHARED_ROOT)"; [ -n "$shared" ] || repro_fatal "cluster $l has no CLUSTER_SHARED_ROOT — cannot stage the geometry records"
        echo "DRY-RUN: geometry verify $(basename "$dir")/$name against $pcluster:$precord (hpo_probe_status --record: READY + generation ${pgen:0:16}) then geometry stage -> $l:$shared/exp/reproduce/geometry/$name (staging_verification stamped; canonical-content-checked)"
      done
      continue
    fi
    # 2. the authoritative record, over ssh on the probe cluster: READY and the SAME generation as the overlay's stamp
    _geometry_verify_ready(){   # sets state / detail / rgen from the record on the probe cluster; repro_fatal on anything but READY + the stamp
      remote="set -u; command -v python3 >/dev/null 2>&1 || { echo 'GEOMETRY_NO_PYTHON3'; exit 5; }; [ -d '$precord' ] || { echo 'GEOMETRY_RECORD_UNREACHABLE $precord'; exit 7; }; \
A=\$(squeue -u \$USER -h -o %j 2>/dev/null | grep -cE '^(hpo-${pmodel}-lora-probe|serving-probe-${pmodel})' || true); JA=0; [ \"\${A:-0}\" -gt 0 ] && JA=1; \
T=\$(mktemp /tmp/hps.XXXXXX) && printf '%s' '$status_b64' | base64 -d > \"\$T\" && python3 \"\$T\" --record '$precord' --job-active \$JA; rc=\$?; rm -f \"\$T\"; exit \$rc"
      out="$(repro_ssh "$pcluster" "$remote" 2>&1)"; rc=$?
      case "$rc" in
        7) repro_fatal "geometry: $name: its probe record $pcluster:$precord is UNREACHABLE (not a directory on that cluster's login host) — the authoritative record cannot be read, so the site copy cannot be verified; NOT staged; nothing submitted";;
        5) repro_fatal "geometry: $pcluster has no python3 on its login host — the site copy cannot be verified against its record; nothing submitted";;
        0) ;;
        *) repro_fatal "geometry: verifying $name against $pcluster:$precord failed (rc=$rc: $out); nothing submitted";;
      esac
      state="$(printf '%s\n' "$out" | tail -1 | cut -f1)"; detail="$(printf '%s\n' "$out" | tail -1 | cut -f2)"
      pjson="$(printf '%s\n' "$out" | tail -1 | cut -f3)"     # the predicate's whole reading: stamped into the staged copy
      rgen="$(printf '%s' "$pjson" | "$(repro_python)" -c 'import json,sys; print(json.load(sys.stdin).get("generation") or "")' 2>/dev/null)" || rgen=""
      [ "$state" = READY ] || repro_fatal "geometry: $name: its probe record $pcluster:$precord is $state — $detail — the site copy is not that record's current certification (reopened, re-planned or never certified); NOT staged; when the record is READY, 03_hpo.sh --phase probe refreshes the site copy; nothing submitted"
      [ -n "$rgen" ] && [ "$rgen" = "$pgen" ] || repro_fatal "geometry: $name: stamped result_generation ${pgen:0:16}… is not the record's current generation ${rgen:0:16}… ($pcluster:$precord is READY, for another result set) — the site copy is STALE; NOT staged; re-run 03_hpo.sh --phase probe to refresh it; nothing submitted"
    }
    _geometry_verify_ready
    # 3. the staged bytes: the site copy + the staging_verification block, through the one serializer
    b64="$(printf '%s' "$pjson" | "$(repro_python)" -c 'import base64,json,socket,sys
sys.path.insert(0, sys.argv[3] + "/src")
from sisteer.capeval.serving import geometry as g
info = json.load(sys.stdin) if sys.stdin.readable() else {}
pr = info.get("parity_record") or None
sys.stdout.write(base64.b64encode(g.staged_site_geometry_bytes(
    sys.argv[1], record_generation=sys.argv[2], verifier_host=socket.gethostname(), record_state=sys.argv[4],
    parity_check=str(info.get("parity_check") or "none"),
    parity_record=({"dir": pr.get("dir"), "generation": pr.get("live_generation")} if isinstance(pr, dict) else None))).decode())' "$f" "$rgen" "$REPRO_REPO" "$state")" \
      || repro_fatal "geometry: $name could not be stamped for staging (its reason is above); nothing submitted"
    local payload nbytes
    payload="$(mktemp "${TMPDIR:-/tmp}/geometry_stage.XXXXXX")" || repro_fatal "geometry: cannot create the staging payload file; nothing submitted"
    printf '%s' "$b64" | base64 -d > "$payload" || repro_fatal "geometry: cannot write the staging payload for $name; nothing submitted"
    sha="$(sha256sum "$payload" | cut -d' ' -f1)"; nbytes="$(wc -c < "$payload" | tr -d ' ')"
    [ "$nbytes" -gt 0 ] || repro_fatal "geometry: the staging payload for $name is empty; nothing submitted"
    # test hook (the interleaving test, re-check 3): pause here — READY checked, the old-generation bytes prepared, the lock NOT yet
    # taken — while the file exists (touching <file>.waiting so the test knows the stager is parked)
    if [ -n "${SISTEER_STAGE_PAUSE_FILE:-}" ]; then
      local pw=0; : > "$SISTEER_STAGE_PAUSE_FILE.waiting"
      while [ -e "$SISTEER_STAGE_PAUSE_FILE" ] && [ "$pw" -lt 600 ]; do sleep 0.2; pw=$((pw + 1)); done
      repro_note "geometry: SISTEER_STAGE_PAUSE_FILE released after $((pw / 5))s — continuing to the lock"
    fi
    # THE PER-MODEL LOCK (re-check 3): staging and revocation exclude each other on this host — the stager takes the lock, checks
    # the site for a REVOKED marker of this generation (a reopen in progress or pending), RE-VERIFIES READY + the generation
    # against the record, and only then installs; slurm_revoke_geometry holds the same lock while it archives and marks
    # WHICH COPY CLEARS A SAME-GENERATION MARKER (B11, 2026-10-09): the one certified THROUGH that revocation's receipt — its
    # provenance.reopen_epoch equals the marker's epoch (sha256 of the receipt nonce; serving_probe.py REOPEN_EPOCH_KEY). An exact
    # event, where this used to be `written_utc < revoked_utc`: two second-resolution clock stamps compared strictly, so a fold and a
    # later revocation in the SAME second read as fold-after and a copy certified BEFORE the revocation cleared its marker (the
    # staging/revocation test flipped `rc=0` on four landing gates). A marker written before the epoch existed (no `epoch` field) is
    # cleared only by a fold strictly AFTER its revoked_utc; a tie at second resolution is not an order and refuses.
    local markerpy='import json,sys; d=json.load(open(sys.argv[1])); print((d.get("generation") or "-")+" "+(d.get("revoked_utc") or "-")+" "+(d.get("epoch") or "-"))'
    local GEOMETRY_MARKER_WHY=""
    _geometry_copy_clears_marker(){   # <marker revoked_utc> <marker epoch, or -> -> 0 when THIS copy (of the marker's generation) is the certification that lifts it; else 1 with GEOMETRY_MARKER_WHY
      if [ "$2" != "-" ]; then
        if [ -n "$pepoch" ] && [ "$pepoch" = "$2" ]; then return 0; fi
        GEOMETRY_MARKER_WHY="was not certified through that revocation's receipt (the copy's reopen epoch is ${pepoch:-absent}, the marker's ${2:0:16}…)"; return 1
      fi
      if [[ "$pwutc" > "$1" ]]; then return 0; fi
      GEOMETRY_MARKER_WHY="was folded at $pwutc, not strictly after that revocation (a marker without an epoch orders by the clock, and a tie at second resolution is not an order)"; return 1
    }
    local lockf="$dir/.lock.$pmodel" lk
    exec {lk}>"$lockf" || repro_fatal "geometry: cannot open the per-model lock $lockf; nothing submitted"
    flock -x "$lk" || repro_fatal "geometry: cannot take the per-model lock $lockf; nothing submitted"
    if [ -f "$dir/$pmodel.REVOKED.json" ]; then
      local mgen mutc mepoch
      read -r mgen mutc mepoch <<<"$("$(repro_python)" -c "$markerpy" "$dir/$pmodel.REVOKED.json" 2>/dev/null || echo "- - -")"
      if [ "$mgen" = "$pgen" ] && ! _geometry_copy_clears_marker "$mutc" "$mepoch"; then      # the same generation, not the certification that lifts the marker
        repro_fatal "geometry: $name: result generation ${pgen:0:16}… was REVOKED by the front door at $mutc (a reopen of $pcluster:$precord is in progress or pending) and this site copy $GEOMETRY_MARKER_WHY — NOT staged (nothing installed, no marker cleared); nothing submitted"
      fi
    fi
    _geometry_verify_ready       # again, under the lock, immediately before the install
    for l in ${1//,/ }; do
      shared="$(cluster_var "$l" CLUSTER_SHARED_ROOT)"; [ -n "$shared" ] || repro_fatal "cluster $l has no CLUSTER_SHARED_ROOT — cannot stage the geometry records"
      # ONE timestamp for the target's marker archive: the remote clear_marker used to interpolate `$(date)` twice
      # (mkdir one stamp, mv another), so a re-stage crossing a second left the marker standing while GEOMETRY_SAME / GEOMETRY_STAGED
      # reported success — and every consumer of that overlay then refused as REVOKED at preflight. A failed clear now exits the
      # remote shell with rc 10 (GEOMETRY_MARKER_CLEAR_FAILED, by name) before any success line can be printed.
      # AND A DIRECTORY NO OTHER ARCHIVAL SHARES (2026-10-09): a stamp to the second
      # made two clears inside one second name the SAME directory, `mkdir -p` accepted it and the second `mv -f` wrote the first
      # archived marker over. The stamp is minted HERE (the grading host's clock, like every stamp under a geometry superseded/ —
      # the remote never calls `date` for it) to the microsecond in repro_superseded_dir's fixed-width format, and the remote
      # RESERVES the directory with a bare `mkdir`, taking the next `-NN` suffix on a collision — the same rule as
      # repro_superseded_dir, inlined because an ssh string cannot call a shell function on the grading host. reserve_sup leaves
      # the reserved path in A; nothing archived is ever written over, and a base that cannot be a directory (superseded/ is a
      # file) fails the clear by name as before.
      local sup_ts; sup_ts="$(repro_superseded_stamp)"
      # the REVOKED marker on the target is cleared ONLY for a NEWLY certified copy: another generation than the marker's, or the copy
      # certified THROUGH that revocation's receipt (its provenance.reopen_epoch = the marker's epoch; B11 above) — for a marker
      # without an epoch, a copy folded (written_utc) strictly after the revocation — never for the same generation certified before it
      remote="set -u; command -v python3 >/dev/null 2>&1 || { echo 'GEOMETRY_NO_PYTHON3'; exit 5; }; D='$shared/exp/reproduce/geometry'; F=\"\$D/$name\"; mkdir -p \"\$D\"; \
reserve_sup(){ mkdir -p \"\$D/superseded\" || return 1; A=\"\$D/superseded/$sup_ts\"; n=1; until mkdir \"\$A\" 2>/dev/null; do [ -e \"\$A\" ] || return 1; n=\$((n+1)); A=\"\$D/superseded/$sup_ts-\$(printf '%02d' \$n)\"; done; }; \
MK=\"\$D/${name%.json}.REVOKED.json\"; clear_marker(){ if [ -f \"\$MK\" ]; then read -r mg mu me <<<\"\$(python3 -c '$markerpy' \"\$MK\" 2>/dev/null || echo '- - -')\"; if [ \"\$mg\" != '$rgen' ] || { [ \"\$me\" != - ] && [ \"\$me\" = '$pepoch' ]; } || { [ \"\$me\" = - ] && [[ '$pwutc' > \"\$mu\" ]]; }; then A=''; reserve_sup && mv -f \"\$MK\" \"\$A/\" && echo \"GEOMETRY_MARKER_CLEARED \$A\" || { echo \"GEOMETRY_MARKER_CLEAR_FAILED $name marker=\$MK dest=\${A:-\$D/superseded/$sup_ts}\"; exit 10; }; else echo 'GEOMETRY_MARKER_KEPT'; fi; fi; }; \
if [ -f \"\$F\" ]; then have=\$(python3 -c '$canon' \"\$F\" 2>/dev/null) || { echo \"GEOMETRY_UNREADABLE $name\"; exit 6; }; read -r hb hp hw hs <<<\"\$(python3 -c '$havegen' \"\$F\" 2>/dev/null || true)\"; \
if [ \"\$have\" = '$csha' ]; then if [ \"\${hb:-}\" = '$rgen' ] && [ \"\${hs:-0}\" = 1 ]; then clear_marker; echo 'GEOMETRY_SAME $name'; exit 0; fi; echo \"GEOMETRY_RESTAMPING $name\"; \
elif [ \"\${hw:--}\" != '-' ] && [[ \"\$hw\" > '$pwutc' ]]; then echo \"GEOMETRY_NEWER $name cluster_written=\$hw site_written=$pwutc cluster_generation=\${hb:-\${hp:-none}} cluster_content=\$have site_content=$csha\"; exit 8; \
elif [ \"\${hb:-}\" = '$rgen' ] || [ \"\${hp:-}\" = '$rgen' ]; then if [ \"\${hw:--}\" != '-' ] && [[ \"\$hw\" < '$pwutc' ]]; then echo \"GEOMETRY_REPLACING_OLDER $name cluster_written=\$hw site_written=$pwutc\"; else echo \"GEOMETRY_MISMATCH $name cluster_content=\$have site_content=$csha (same generation, different content, same written_utc \${hw:-none})\"; exit 3; fi; \
else echo \"GEOMETRY_REPLACING $name stale_generation=\${hb:-\${hp:-none}}\"; fi; fi; \
cat > \"\$F.tmp.\$\$\"; gotn=\$(wc -c < \"\$F.tmp.\$\$\" | tr -d ' '); got=\$(sha256sum \"\$F.tmp.\$\$\" | cut -d' ' -f1); \
if [ \"\$gotn\" != '$nbytes' ] || [ \"\$got\" != '$sha' ]; then echo \"GEOMETRY_TRANSFER_MISMATCH $name expected_bytes=$nbytes got_bytes=\$gotn expected_sha=$sha got_sha=\$got\"; rm -f \"\$F.tmp.\$\$\"; exit 9; fi; \
mv -f \"\$F.tmp.\$\$\" \"\$F\" && clear_marker; [ -f \"\$F\" ] && echo 'GEOMETRY_STAGED $name' || { rm -f \"\$F.tmp.\$\$\"; echo 'GEOMETRY_WRITE_FAILED $name'; exit 4; }"
      out="$(repro_ssh_stdin "$l" "$remote" "$payload" 2>&1)"; rc=$?
      case "$rc:$out" in
        9:*) rm -f "$payload"; repro_fatal "geometry: the transfer of $name to $l did not reproduce the staged bytes ($out) — nothing installed there; nothing submitted";;
        10:*) rm -f "$payload"; repro_fatal "geometry: $l: the REVOKED marker beside $shared/exp/reproduce/geometry/$name could not be moved aside ($out) — the marker still stands and every consumer of that overlay would refuse as REVOKED; fix the target's superseded/ directory and re-stage; nothing submitted";;
        0:*GEOMETRY_SAME*) repro_say "geometry: $l already holds $name (content sha256 ${csha:0:16}, verified against $pcluster:$precord generation ${rgen:0:16}; formatting not compared$(case "$out" in *GEOMETRY_MARKER_CLEARED*) echo "; REVOKED marker cleared -> $(printf '%s\n' "$out" | sed -n 's/^GEOMETRY_MARKER_CLEARED //p' | head -1)/";; *GEOMETRY_MARKER_KEPT*) echo "; REVOKED marker KEPT (same generation, not the certification that lifts it: another epoch, or folded before)";; esac))";;
        0:*GEOMETRY_STAGED*) repro_say "geometry: staged $name -> $l:$shared/exp/reproduce/geometry (content sha256 ${csha:0:16}; verified against $pcluster:$precord READY, generation ${rgen:0:16}, staging_verification stamped$(case "$out" in *GEOMETRY_REPLACING_OLDER*) echo "; replaced an OLDER fold of the same generation (cluster copy written $(printf '%s\n' "$out" | sed -n 's/.*cluster_written=\([^ ]*\).*/\1/p' | head -1), site copy $pwutc)";; *GEOMETRY_REPLACING*) echo "; replaced a stale copy (${out##*stale_generation=}" | cut -c1-60 | sed 's/$/…)/';; *GEOMETRY_RESTAMPING*) echo "; re-stamped a copy whose staging block was absent or predates the predicate fields (content identical; the block alone was rewritten)";; esac)$(case "$out" in *GEOMETRY_MARKER_CLEARED*) echo "; REVOKED marker cleared -> $(printf '%s\n' "$out" | sed -n 's/^GEOMETRY_MARKER_CLEARED //p' | head -1)/";; *GEOMETRY_MARKER_KEPT*) echo "; REVOKED marker KEPT (same generation, not the certification that lifts it: another epoch, or folded before)";; esac))";;
        3:*) repro_fatal "geometry: $l:$shared/exp/reproduce/geometry/$name differs IN CONTENT from the site's copy ($out) — the probe re-folded on that cluster or the site holds a stale record; reconcile with 03_hpo.sh --phase probe (it re-fetches the record) before submitting; nothing submitted";;
        8:*) repro_fatal "geometry: $l:$shared/exp/reproduce/geometry/$name is NEWER than the site's copy — the cluster overlay (written $(printf '%s\n' "$out" | sed -n 's/.*cluster_written=\([^ ]*\).*/\1/p' | head -1)) postdates the site copy (written $pwutc): a later fold (a reverdict, a re-probe) the site has not fetched — refresh the site copy from the record first (03_hpo.sh --phase probe re-fetches it), then stage; NOT overwritten; nothing submitted ($out)";;
        5:*) repro_fatal "geometry: $l has no python3 on its login host — the canonical-content compare cannot run; nothing submitted";;
        6:*) repro_fatal "geometry: $l:$shared/exp/reproduce/geometry/$name is not readable JSON ($out) — a torn or foreign file; remove it by hand before submitting; nothing submitted";;
        *) repro_fatal "geometry: staging $name to $l failed (rc=$rc: $out); nothing submitted";;
      esac
    done
    # every target verified and holding the current generation: the SITE's own REVOKED marker is cleared under the same rule
    # (a newly certified copy — another generation, or folded after the revocation) — moved aside, never deleted
    if [ -f "$dir/${name%.json}.REVOKED.json" ]; then
      local smg smu sme
      read -r smg smu sme <<<"$("$(repro_python)" -c "$markerpy" "$dir/${name%.json}.REVOKED.json" 2>/dev/null || echo "- - -")"
      if [ "$smg" != "$rgen" ] || _geometry_copy_clears_marker "$smu" "$sme"; then      # another generation, or the copy certified through that revocation (B11)
        # ONE timestamp (2026-10-01, the canary's staging at 08:31:35Z): `mkdir -p .../$(date)` and `mv .../$(date)/` evaluated `date`
        # twice and straddled a second boundary — the directory made at :35 did not exist at :36 ("Not a directory"), the marker stayed,
        # and the whole staging reported REFUSED after both clusters had been staged and verified. And a FRESH directory per archival
        # (2026-10-09; lib/common.sh repro_superseded_dir: a microsecond stamp reserved by an atomic mkdir, `-NN` on a collision): a
        # second-resolution name let two clears inside one second share a directory, the second `mv -f` writing the first archive over
        local sup; sup="$(repro_superseded_dir "$dir/superseded")" \
          || repro_fatal "geometry: the site's ${name%.json}.REVOKED.json could not be moved aside (no archive directory could be reserved under $dir/superseded/); nothing submitted"
        mv -f "$dir/${name%.json}.REVOKED.json" "$sup/" \
          && repro_say "geometry: the site's ${name%.json}.REVOKED.json cleared (a newly certified copy: every target holds its current generation; archived under $sup/)" \
          || repro_fatal "geometry: the site's ${name%.json}.REVOKED.json could not be moved aside (into $sup/); nothing submitted"
      else
        repro_say "geometry: the site's ${name%.json}.REVOKED.json KEPT (same generation, not the certification that lifts it: $GEOMETRY_MARKER_WHY)"
      fi
    fi
    rm -f "$payload"
    flock -u "$lk"; exec {lk}>&-
  done
}
# REVOKE-BEFORE-REOPEN. Clusters cannot read each other or the grading host (no keys), so a staged copy's
# stamps never learn that its SOURCE record was reopened. Before the front door reopens a record whose checkpoint is complete
# (a resume / re-plan / reverdict of a record that has a verdict), it calls this: every configured cluster must be reachable
# (probed first — an unreachable cluster REFUSES the whole reopen, fail closed, nothing archived, nothing submitted), then the
# grading host site copy and every cluster's staged overlay of the model move to `superseded/<utc>/` beside where they stood and a
# `<model_key>.REVOKED.json` marker ({model_key, generation, revoked_utc, reason, by, epoch = sha256 of the receipt nonce when a record
# is named — B11, the stager's exact rule for which copy lifts it}) is written there; each archive is verified
# by listing. geometry.check_overlay_generation refuses any overlay with the marker beside it — so a padded job queued or
# running on the other cluster whose preflight runs after the reopen refuses instead of accepting revoked geometry. The
# carrier refuses to reopen a complete record without --reopen-token <generation>, which 03 mints ONLY after this returns 0.
# The next VERIFIED staging (slurm_stage_geometry) moves the marker aside.
# THE RECEIPT (re-check 3): the generation is a deterministic digest of public files and can be typed, so the token the carrier
# accepts is NOT the generation but the nonce of a receipt this function writes INTO THE RECORD on the probe cluster (arguments 4
# and 5: the probe cluster label and the record dir) after every configured cluster acknowledged its marker:
# <record>/reopen_receipts/<generation>.<nonce>.json {generation, nonce, acknowledged: [{cluster, marker, utc, archived}], site,
# minted_utc, minted_host, reason}. The nonce is returned in SLURM_REVOKE_NONCE (empty without a record); the driver verifies a
# receipt for its CURRENT generation carrying that nonce and consumes it. 03_hpo.sh's probe_reopen_token is the one caller.
SLURM_REVOKE_NONCE=""
slurm_revoke_geometry(){   # <model_key> <generation> <reason> [<probe_cluster> <record_dir>] -> 0 with every copy archived + marked (+ the receipt), else repro_fatal
  local model="$1" gen="$2" reason="${3//\'/}" pcluster="${4:-}" precord="${5:-}" utc l shared remote out rc sd acks="" nonce=""
  SLURM_REVOKE_NONCE=""
  [ -n "$model" ] && [ -n "$gen" ] || repro_fatal "geometry: revoke needs <model_key> <generation> — nothing revoked"
  # ONE STAMP FOR THE WHOLE REVOCATION, A DIRECTORY NO OTHER ARCHIVAL SHARES (2026-10-09). One revocation archives at several places
  # — the site copy here, every cluster's overlay over ssh — and every line it says names the same `superseded/<utc>/`, which is how a
  # reader finds one revocation's archives across the site and the clusters (no code parses the name: the receipt carries the
  # generation and each cluster's ack utc, the tests glob `superseded/*/<model>.json`). So the stamp stays common, minted here once
  # (the grading host's clock, like every stamp under a geometry superseded/), and is to the microsecond in repro_superseded_dir's
  # fixed-width format; each place RESERVES `superseded/<utc>` with a bare mkdir and takes a `-NN` suffix only on a collision there
  # (two revocations under a frozen clock, a concurrent archival into the same directory) — where a second-resolution name let two
  # revocations inside one second share the directory and the second `mv -f` write the first archived overlay over. The site copy
  # goes through repro_superseded_dir with this stamp; an ssh string cannot call a shell function on the grading host, so the remote
  # inlines the same reservation and reports the directory it took (archive=…), which the message names.
  utc="$(repro_superseded_stamp)"
  # THE REVOCATION EPOCH (B11, 2026-10-09): with a record named, the receipt's nonce is minted HERE, before any marker, and every
  # marker of this revocation carries epoch = sha256(nonce) — the nonce itself is written nowhere but the receipt. The fold certified
  # through this receipt stamps the same epoch into provenance.reopen_epoch (serving_probe.py), and slurm_stage_geometry lifts a
  # same-generation marker for THAT copy alone, never by comparing the fold's and the revocation's second-resolution clock stamps.
  local epoch="" epoch_field=""
  if [ -n "$pcluster" ] && [ -n "$precord" ] && [ "${REPRO_DRY_RUN:-0}" != 1 ]; then
    nonce="$("$(repro_python)" -c 'import secrets; print(secrets.token_hex(16))')" || repro_fatal "geometry: revoke: cannot mint a receipt nonce; nothing revoked"
    epoch="$(printf '%s' "$nonce" | sha256sum | cut -d' ' -f1)"; epoch_field="\"epoch\": \"$epoch\", "
  fi
  local marker; marker="$(printf '{"model_key": "%s", "generation": "%s", "revoked_utc": "%s", %s"reason": "%s", "by": "front door (reproduce/lib/slurm.sh slurm_revoke_geometry)"}' "$model" "$gen" "$(date -u +%FT%TZ)" "$epoch_field" "$reason")"
  if [ "${REPRO_DRY_RUN:-0}" = 1 ]; then
    for l in $(cluster_labels); do echo "DRY-RUN: geometry revoke $model.json on $l (-> $(cluster_var "$l" CLUSTER_SHARED_ROOT)/exp/reproduce/geometry/superseded/$utc/, marker $model.REVOKED.json, generation ${gen:0:16})"; done
    echo "DRY-RUN: geometry revoke the site copy geometry/$model.json (-> geometry/superseded/$utc/, marker written)"; return 0
  fi
  # 1. every cluster reachable, or nothing happens (fail closed)
  for l in $(cluster_labels); do
    repro_ssh "$l" "true" >/dev/null 2>&1 || repro_fatal "geometry: cluster $l cannot be reached — it may hold a staged overlay of $model that this reopen must revoke; the reopen is REFUSED (fail closed); nothing archived, nothing submitted"
  done
  [ -z "$pcluster" ] || case " $SISTEER_CLUSTERS " in *" $pcluster "*) ;; *) repro_fatal "geometry: revoke: probe cluster '$pcluster' is not one of SISTEER_CLUSTERS — no receipt can be written; nothing revoked";; esac
  # A reopen of the record IN PROGRESS (a receipt consumed whose certifying rewrite has not completed — a reverdict that
  # died after the archive) is recovered under ITS receipt (`serving_probe.py reverdict --resume-interrupted`), never revoked over: a
  # second mint would strand the record; refused BEFORE anything is archived (fail closed)
  if [ -n "$pcluster" ] && [ -n "$precord" ]; then
    local ip; ip="$(repro_ssh "$pcluster" "R='$precord/reopen_receipts'; [ -d \"\$R\" ] || { echo 0; exit 0; }; ls -1 \"\$R\" 2>/dev/null | grep -c '\.inprogress\.' || true" 2>/dev/null | tail -1)"
    case "$ip" in ''|0) ;; *) repro_fatal "geometry: revoke: a reopen of $pcluster:$precord is IN PROGRESS under an earlier receipt ($ip in-progress receipt(s): a reverdict consumed it and did not complete its certifying rewrite) — recover it first (\`serving_probe.py reverdict --record $precord --resume-interrupted\`); a second revocation would strand the record; NOTHING revoked, nothing submitted";; esac
  fi
  # 2. the grading host site copy — under the per-model lock the stager also takes (re-check 3)
  sd="$SISTEER_SITE_DIR/geometry"; mkdir -p "$sd"
  local lockf="$sd/.lock.$model" lk
  exec {lk}>"$lockf" || repro_fatal "geometry: cannot open the per-model lock $lockf; nothing revoked"
  flock -x "$lk" || repro_fatal "geometry: cannot take the per-model lock $lockf; nothing revoked"
  local site_arch="$sd/superseded/$utc"          # named in the message even when nothing was there to archive (no directory is reserved then)
  if [ -f "$sd/$model.json" ]; then
    site_arch="$(repro_superseded_dir "$sd/superseded" "$utc")" \
      || repro_fatal "geometry: the site copy $sd/$model.json could not be archived (no archive directory could be reserved under $sd/superseded/) — the reopen is REFUSED; nothing submitted"
    mv -f "$sd/$model.json" "$site_arch/$model.json" && [ -f "$site_arch/$model.json" ] \
      || repro_fatal "geometry: the site copy $sd/$model.json could not be archived (into $site_arch/) — the reopen is REFUSED; nothing submitted"
  fi
  printf '%s\n' "$marker" > "$sd/$model.REVOKED.json" && [ -s "$sd/$model.REVOKED.json" ] || repro_fatal "geometry: the site marker could not be written — the reopen is REFUSED"
  repro_say "geometry: site copy of $model revoked (archived under $site_arch/ when present; marker $model.REVOKED.json written; generation ${gen:0:16})"
  # 3. every cluster's staged overlay: archive + marker, verified by listing
  for l in $(cluster_labels); do
    shared="$(cluster_var "$l" CLUSTER_SHARED_ROOT)"; [ -n "$shared" ] || repro_fatal "cluster $l has no CLUSTER_SHARED_ROOT — the reopen is REFUSED"
    # the archive directory is RESERVED (bare mkdir; `-NN` on a collision — the rule of repro_superseded_dir, inlined: see the stamp
    # above) and reported as `archive=<dir>` on the REVOKED_OK line, which the message below names
    remote="set -u; D='$shared/exp/reproduce/geometry'; mkdir -p \"\$D/superseded\" || exit 4; A=\"\$D/superseded/$utc\"; n=1; \
until mkdir \"\$A\" 2>/dev/null; do [ -e \"\$A\" ] || { echo REVOKE_ARCHIVE_DIR_FAILED; exit 4; }; n=\$((n+1)); A=\"\$D/superseded/$utc-\$(printf '%02d' \$n)\"; done; moved=0; \
if [ -f \"\$D/$model.json\" ]; then mv -f \"\$D/$model.json\" \"\$A/$model.json\" && [ -f \"\$A/$model.json\" ] && moved=1 || { echo REVOKE_ARCHIVE_FAILED; exit 4; }; fi; \
printf '%s\\n' '$marker' > \"\$D/$model.REVOKED.json\" && [ -s \"\$D/$model.REVOKED.json\" ] || { echo REVOKE_MARKER_FAILED; exit 4; }; \
echo \"REVOKED_OK $model moved=\$moved archive=\$A\"; ls -1 \"\$A\" | sed 's/^/ARCHIVED /'; ls -1 \"\$D/$model.REVOKED.json\" | sed 's#^.*/#MARKER #'"
    out="$(repro_ssh "$l" "$remote" 2>&1)"; rc=$?
    case "$rc:$out" in
      0:*REVOKED_OK*MARKER*) repro_say "geometry: $l: $model.json revoked ($(printf '%s\n' "$out" | grep -c '^ARCHIVED ') file(s) archived under $(printf '%s\n' "$out" | sed -n 's/^REVOKED_OK .* archive=//p' | head -1)/, marker present)"
                             acks="$acks$l|$shared/exp/reproduce/geometry/$model.REVOKED.json|$(date -u +%FT%TZ)|$(printf '%s\n' "$out" | grep -c '^ARCHIVED ')"$'\n';;
      *) repro_fatal "geometry: revoking $model on $l failed (rc=$rc: $out) — the reopen is REFUSED (fail closed); a cluster that could not be revoked may still hold a staged overlay; nothing submitted";;
    esac
  done
  # 4. the receipt INTO the record on the probe cluster (only when the caller named it): the nonce is the token the carrier accepts
  if [ -n "$pcluster" ] && [ -n "$precord" ]; then
    local rb64
    [ -n "$nonce" ] && [ -n "$epoch" ] || repro_fatal "geometry: revoke: the receipt nonce was not minted before the markers; nothing submitted"   # minted above, before any marker (B11)
    rb64="$(printf '%s' "$acks" | "$(repro_python)" -c 'import base64,json,socket,sys,time
acks=[dict(zip(("cluster","marker","utc","archived_files"), ln.split("|"))) for ln in sys.stdin.read().splitlines() if ln.strip()]
for a in acks: a["archived_files"]=int(a["archived_files"])
doc={"version":1,"model_key":sys.argv[1],"generation":sys.argv[2],"nonce":sys.argv[3],"epoch":sys.argv[8],"reason":sys.argv[4],"record":sys.argv[5],"probe_cluster":sys.argv[6],
     "acknowledged":acks,"site":{"marker":sys.argv[7],"utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())},
     "minted_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),"minted_host":socket.gethostname(),"by":"front door (reproduce/lib/slurm.sh slurm_revoke_geometry)"}
sys.stdout.write(base64.b64encode((json.dumps(doc,indent=2,sort_keys=True)+"\n").encode()).decode())' "$model" "$gen" "$nonce" "$reason" "$precord" "$pcluster" "$sd/$model.REVOKED.json" "$epoch")" \
      || repro_fatal "geometry: revoke: cannot build the receipt; nothing submitted"
    # ONE revocation epoch per record: before the new receipt lands, every prior ACTIVE receipt of the record
    # (the listing is `find`, not a glob: the login host's zsh makes an unmatched glob fatal before the loop's [ -f ] guard, so a record
    # whose reopen_receipts/ is still empty — the parity record's first reopen, 2026-10-01 — refused with `no matches found`)
    # (whatever its generation) is moved aside as `.superseded.<utc>.json` — never deleted — so an orphan of an earlier revocation can
    # never authorise a reopen after this one; the consumer (serving_probe.consume_reopen_receipt) refuses anything but the LATEST mint
    remote="set -u; R='$precord/reopen_receipts'; [ -d '$precord' ] || { echo RECEIPT_RECORD_UNREACHABLE; exit 7; }; mkdir -p \"\$R\" || exit 4; U=\$(date -u +%Y%m%dT%H%M%SZ); \
for f in \$(find \"\$R\" -maxdepth 1 -name '*.json' 2>/dev/null); do [ -f \"\$f\" ] || continue; b=\$(basename \"\$f\"); case \"\$b\" in *.used.*|*.superseded.*|*.consuming.*|*.tmp.*) ;; *) mv -f \"\$f\" \"\${f%.json}.superseded.\$U.json\" && echo \"SUPERSEDED \$b\";; esac; done; \
F=\"\$R/$gen.$nonce.json\"; printf '%s' '$rb64' | base64 -d > \"\$F.tmp.\$\$\" && mv -f \"\$F.tmp.\$\$\" \"\$F\" && [ -s \"\$F\" ] && echo \"RECEIPT_OK \$F\" || { echo RECEIPT_WRITE_FAILED; exit 4; }"
    out="$(repro_ssh "$pcluster" "$remote" 2>&1)"; rc=$?
    case "$rc:$out" in
      0:*RECEIPT_OK*) repro_say "geometry: revocation receipt written into the record ($pcluster:$precord/reopen_receipts/, generation ${gen:0:16}, $(printf '%s' "$acks" | grep -c .) cluster(s) acknowledged$( n=$(printf '%s\n' "$out" | grep -c '^SUPERSEDED '); [ "$n" -gt 0 ] && printf ', %s prior active receipt(s) superseded' "$n")); the carrier's reopen token is its nonce";;
      *) repro_fatal "geometry: the revocation receipt could not be written into $pcluster:$precord (rc=$rc: $out) — the reopen is REFUSED; every overlay IS revoked (markers stand); nothing submitted";;
    esac
    SLURM_REVOKE_NONCE="$nonce"
  fi
  flock -u "$lk"; exec {lk}>&-
  return 0
}
# the env prefix every wave-shell job of the front door carries: the registry campaign (the site's ONE resolution, common.sh
# repro_registry_campaign — the campaign 04 fitted under and 08 grades under), the native serving path (FD 19), the roster
slurm_base_env(){ printf 'F22_REGISTRY_CAMPAIGN=%s F22_SERVING_PATH=lora_native F22_ROSTER=configs/sdfbase_full22_roster.yaml' "$(repro_registry_campaign)"; }
# the job-side portability + registry env every submission carries (§3.1): read by the bootstraps before their two-root
# probe (common.sh f22_init, the sbatch source lines, sdfbase_fit_cell.sbatch, hf_env.sh), by registry_env.sh /
# push_records.sh (REGISTRY_*), by the emitter (SISTEER_REGISTRY_SUBSTRATE / _CLUSTER_LABEL), and by every fit-family sbatch
# (memprobe_fit / hpo_trial / sdfbase_fit_cell / sdfbase_derived_build / fit_refusal_direction read SDFBASE_FIT_ENV_NAME for the
# prebuilt venv under $SHARED/envs — the cluster's OWN name, cluster_fit_env_name)
REPRO_SLURM_LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # this library's own directory: the funnel beside it composes the job env
slurm_job_env(){   # <label> -> "K=V K=V …" (values shell-quoted): the funnel's own job environment for that cluster
  # ONE list: lib/submit_twin.py Coordinator.job_env composes the env prefix of every live sbatch line and `job-env` prints that
  # same environment, so a dry-run printer built on this function shows exactly what a live job carries: the cluster block's
  # roots and labels, the registry spool, push helper, campaign and fit venv name, and the site's W&B entity and project when
  # the site names an entity. The funnel reads the site as a stage does (the config file this run loaded, laid over the
  # environment), so nothing here keeps a second copy of the names or of their resolution
  local out
  out="$("$(repro_python)" "$REPRO_SLURM_LIB_DIR/submit_twin.py" --config "${SISTEER_CONFIG:?SISTEER_CONFIG}" job-env --cluster "$1")" || {
    echo "REFUSED: the job environment for cluster $1 did not compose (lib/submit_twin.py job-env; reason above)" >&2
    [ "${BASH_SUBSHELL:-0}" -gt 0 ] && kill -TERM "$$" 2>/dev/null
    exit "${REPRO_REFUSE_RC:-2}"
  }
  printf '%s' "$out"
}
