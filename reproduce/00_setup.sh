#!/bin/bash
# reproduce/00_setup.sh — stage 00: the one-time site setup.
#
#   reproduce/00_setup.sh [--cluster <label>|all] [--what venvs|registry|helpers|third_party|mirror|checkpoints|all] [--model K]
#                         [--config F] [--overrides F] [--dry-run] [--help]
#
#   venvs        `uv sync --frozen --extra cpu --group notebook --group corpus` on this host (the grading host runs the judges and the registry from it) — the
#                committed lock, never a re-resolve: pyproject's flash-attn path wheel (wheels/, gitignored) exists only in the
#                cluster venvs' clones and a plain sync fails reading its metadata
#   registry     the registry layout + timers on the grading host through scripts/registry/install_grading_host.sh --no-enable, which
#                renders the unit files for this site from the templates under systemd/user/*.in (the checkout's path, the
#                interpreter, the registry root and the primary cluster's registry label, all from the site config this stage
#                loaded), then an EXPLICIT allowlist enable that NEVER names registry-snapshot.timer and preserves every timer's
#                current state on an existing installation. The allowlist is a subset of the installer's set: a member with no
#                unit template under systemd/user/ (a retired unit) REFUSES the stage by name before anything runs, and a failed
#                installer or enable REFUSES it too (tests/test_setup_never_enables_snapshot_timer.py). Under --dry-run the
#                installer runs in its own --dry-run: it prints the rendered units and changes nothing
#   helpers      install lib/hf_env.sh + lib/mirror_figs.sh + scripts/registry/{registry_env,push_records}.sh onto each cluster's
#                $CLUSTER_HELPERS_DIR (the paths jobs source), and registry/registry_site.env rendered from that cluster's block
#                (its shared root, registry label and site label; the site's SISTEER_SLURM_SUBSTRATES), which registry_env.sh reads,
#                and hf_site.env beside hf_env.sh (the cluster's CLUSTER_HF_HOME and CLUSTER_SHARED_ROOT, and the site's W&B
#                projects for the hand-submitted budget and alphasteer jobs: SISTEER_WANDB_PROJECT_BUDGET / _ALPHASTEER), which
#                hf_env.sh reads; hf_site.env lands before hf_env.sh, so a job never meets a helper without its values
#   third_party  lib/third_party.sh per cluster: fetch the G0DM0D3 templates (AGPL-3.0) and re-draw the refusal-direction fit prompts
#                (andyrdt/refusal_direction's splits, held to data/arditi/fit_prompts.manifest.json), neither distributed with this
#                repository, on THIS host from their pinned upstream commits, verify them, and install the verified files under each
#                cluster's $CLUSTER_SHARED_ROOT/third_party, where jobs read them offline (lib/hf_env.sh)
#   mirror       lib/mirror_repo.sh: the repo mirror + the clean checkout at HEAD on each cluster
#   checkpoints  lib/hf_fetch.sh for the model's fit + serve checkpoints, the attacker, and the three reference tokenizers
#                (one repo per call; the token over stdin; nothing on the cluster) — needs --model
#   all          everything above, in that order
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/lib/common.sh"; source "$HERE/lib/secrets.sh"
usage(){ sed -n '2,27p' "$0"; repro_common_help; }
CLUSTER=all; WHAT=all
repro_parse_common_args "$@"
set -- "${REPRO_ARGS[@]+"${REPRO_ARGS[@]}"}"
while [ $# -gt 0 ]; do
  case "$1" in
    --cluster) CLUSTER="$2"; shift 2;; --what) WHAT="$2"; shift 2;;
    *) echo "REFUSED: unknown argument $1" >&2; usage >&2; exit 2;;
  esac
done
[ "$REPRO_HELP" = 1 ] && { usage; exit 0; }
case "$WHAT" in venvs|registry|helpers|third_party|mirror|checkpoints|all) ;; *) repro_fatal "--what must be venvs|registry|helpers|third_party|mirror|checkpoints|all";; esac
repro_load_config
labels="$(cluster_labels)"; [ "$CLUSTER" = all ] || { cluster_require "$CLUSTER"; labels="$CLUSTER"; }
SHA="$(repro_head_sha)"
want(){ [ "$WHAT" = all ] || [ "$WHAT" = "$1" ]; }

if want venvs; then
  repro_say "venvs: the grading host's environment (uv sync --frozen --extra cpu --group notebook --group corpus; the committed lock, the CPU flavour, the notebook tooling stage 10 needs and the corpus generator stage 01 runs — never a GPU wheel here)"
  repro_run uv sync --frozen --extra cpu --group notebook --group corpus --directory "$REPRO_REPO"
  # gitleaks guards the pre-commit hook on every clone that commits
  if [ -f "$REPRO_REPO/.pre-commit-config.yaml" ] && grep -q gitleaks "$REPRO_REPO/.pre-commit-config.yaml"; then
    hook="$(git -C "$REPRO_REPO" rev-parse --git-path hooks/pre-commit 2>/dev/null)"
    [ -f "$hook" ] && repro_say "gitleaks pre-commit hook: live ($hook)" || repro_run pre-commit install --config "$REPRO_REPO/.pre-commit-config.yaml"
  else
    repro_warn "no gitleaks pre-commit config in this clone — see docs/secrets.md before committing anything"
  fi
fi

if want registry; then
  ROOT="$(repro_registry_root)"
  repro_say "registry: layout + units at $ROOT (install_grading_host.sh --no-enable renders the units for this site; the allowlist enable below never names registry-snapshot.timer)"
  # The ALLOWLIST: exactly the producer-facing timers. registry-snapshot.timer is deliberately absent: it stays stopped, and the
  # paper-frame snapshot is a COMMAND (10_figures.sh --export-snapshot). On an existing installation a timer's current enabled/active
  # state is preserved: only a NEVER-installed timer is enabled here. Every member is one of install_grading_host.sh's TIMERS and has a unit
  # template under systemd/user/ (the test pins both). A unit the repo no longer carries can only leave this list with its template,
  # never stay and be enabled: the check below refuses it by name before the installer runs and before any systemctl, and a failed
  # installer or enable fails the stage instead of leaving a "done" line over a failed systemctl.
  ALLOW=(registry-harvest.timer registry-cluster-reads.timer registry-mirrors.timer registry-mirrors-full.timer registry-status.timer)
  for t in "${ALLOW[@]}"; do
    [ -f "$REPRO_REPO/systemd/user/$t.in" ] \
      || repro_fatal "registry: $t is in this stage's allowlist but has no unit file template at $REPRO_REPO/systemd/user/$t.in (a retired unit?) -- nothing installed, nothing enabled"
  done
  # The installer renders the units from the SAME site config this stage loaded. It runs under --dry-run too, in its own --dry-run
  # (it prints the rendered units and changes nothing), so a site the units cannot be rendered from refuses here as well.
  env REGISTRY_ROOT="$ROOT" "$REPRO_REPO/scripts/registry/install_grading_host.sh" --no-enable --root "$ROOT" --config "$SISTEER_CONFIG" $(repro_dry_flag) \
    || repro_fatal "registry: install_grading_host.sh failed (rc $?) -- nothing enabled"
  for t in "${ALLOW[@]}"; do
    if [ "$REPRO_DRY_RUN" = 1 ]; then echo "DRY-RUN: systemctl --user is-enabled $t || systemctl --user enable --now $t"; continue; fi
    state="$(systemctl --user is-enabled "$t" 2>/dev/null || true)"
    case "$state" in
      enabled|static|active) repro_say "  $t: already $state — left as it is";;
      *) repro_run systemctl --user enable --now "$t" \
           || repro_fatal "registry: systemctl --user enable --now $t failed (rc $?) -- the unit file was rendered from systemd/user/$t.in and installed by install_grading_host.sh just above; the timers after it in the allowlist were not enabled";;
    esac
  done
  repro_say "  registry-snapshot.timer: NOT touched (never enabled by this stage)"
fi

# registry_site.env for one cluster: the values the installed registry_env.sh reads (its header names them), from the cluster block
registry_site_env(){   # <label> -> the file's text
  printf '# registry_site.env — rendered by reproduce/00_setup.sh --what helpers from the site'"'"'s cluster block %s; re-run that stage to change it.\n' "$1"
  printf '# registry_env.sh beside it reads these values; a value a job already carries wins.\n'
  printf 'REGISTRY_SITE_SHARED_ROOT=%q\n' "$(cluster_var "$1" CLUSTER_SHARED_ROOT)"
  printf 'REGISTRY_SITE_SUBSTRATE=%q\n' "$(cluster_var "$1" CLUSTER_SUBSTRATE)"
  printf 'REGISTRY_SITE_CLUSTER_LABEL=%q\n' "$(cluster_var "$1" CLUSTER_LABEL)"
  printf 'REGISTRY_SITE_SLURM_SUBSTRATES=%q\n' "${SISTEER_SLURM_SUBSTRATES:-}"
}

# hf_site.env for one cluster: the values the installed hf_env.sh reads (its header names them), from the cluster block, and the
# W&B project the site names for each purpose whose jobs are submitted by hand, so no front-door job env carries it (BUDGET:
# measure_train_budget's emit; ALPHASTEER: the alphasteer screen and Llama gate). A compute node has no site config; hf_env.sh
# exports a non-empty value as SISTEER_WANDB_PROJECT_<PURPOSE>, which those jobs read first. A purpose the site leaves unset is
# written empty and exports nothing.
HF_SITE_WANDB_PURPOSES="BUDGET ALPHASTEER"
hf_site_env(){   # <label> -> the file's text
  printf '# hf_site.env — rendered by reproduce/00_setup.sh --what helpers from the site config and its cluster block %s; re-run that stage to change it.\n' "$1"
  printf '# hf_env.sh beside it reads these values; a value a job already carries wins.\n'
  printf 'HF_SITE_HF_HOME=%q\n' "$(repro_expand_shared "$1" "$(cluster_var "$1" CLUSTER_HF_HOME)")"
  printf 'HF_SITE_SHARED_ROOT=%q\n' "$(cluster_var "$1" CLUSTER_SHARED_ROOT)"
  local p k
  for p in $HF_SITE_WANDB_PURPOSES; do
    k="SISTEER_WANDB_PROJECT_$p"
    printf 'HF_SITE_WANDB_PROJECT_%s=%q\n' "$p" "${!k:-}"
  done
}

if want helpers; then
  for l in $labels; do
    host="$(cluster_var "$l" CLUSTER_SSH)"; hd="$(cluster_var "$l" CLUSTER_HELPERS_DIR)"; shared="$(cluster_var "$l" CLUSTER_SHARED_ROOT)"
    repro_say "helpers -> $l ($host:$hd): hf_site.env, hf_env.sh, mirror-figs.sh, registry/{registry_env,push_records}.sh, registry/registry_site.env"
    repro_ssh "$l" "mkdir -p '$hd/registry' '$shared/registry/records' '$shared/exp/logs' '$shared/exp/reproduce' '$shared/.ssh'"
    site_env="$(registry_site_env "$l")"; hf_env="$(hf_site_env "$l")"
    if [ "$REPRO_DRY_RUN" = 1 ]; then
      echo "DRY-RUN: write $host:$hd/hf_site.env:"
      printf '%s\n' "$hf_env" | sed 's/^/DRY-RUN:   /'
      echo "DRY-RUN: rsync -a $HERE/lib/hf_env.sh $host:$hd/hf_env.sh; rsync -a $HERE/lib/mirror_figs.sh $host:$hd/mirror-figs.sh; rsync -a $REPRO_REPO/scripts/registry/{registry_env,push_records}.sh $host:$hd/registry/"
      echo "DRY-RUN: write $host:$hd/registry/registry_site.env:"
      printf '%s\n' "$site_env" | sed 's/^/DRY-RUN:   /'
    else
      hf_tmp="$(mktemp)"; printf '%s\n' "$hf_env" > "$hf_tmp"
      repro_ssh_put "$l" "$hf_tmp" "$hd/hf_site.env" "$(sha256sum "$hf_tmp" | cut -d' ' -f1)"; rc=$?
      rm -f "$hf_tmp"
      [ "$rc" = 0 ] || repro_fatal "helper install on $l failed: hf_site.env did not land (rc $rc)"
      rsync -a "$HERE/lib/hf_env.sh" "$host:$hd/hf_env.sh" && rsync -a "$HERE/lib/mirror_figs.sh" "$host:$hd/mirror-figs.sh" \
        && rsync -a "$REPRO_REPO/scripts/registry/registry_env.sh" "$REPRO_REPO/scripts/registry/push_records.sh" "$host:$hd/registry/" \
        || repro_fatal "helper install on $l failed"
      site_tmp="$(mktemp)"; printf '%s\n' "$site_env" > "$site_tmp"
      repro_ssh_put "$l" "$site_tmp" "$hd/registry/registry_site.env" "$(sha256sum "$site_tmp" | cut -d' ' -f1)"; rc=$?
      rm -f "$site_tmp"
      [ "$rc" = 0 ] || repro_fatal "helper install on $l failed: registry/registry_site.env did not land (rc $rc)"
    fi
    repro_mark_done setup helpers "$l"
  done
fi

if want third_party; then
  # lib/third_party.sh --cluster <label> --config <site config> [--dry-run]: fetch the pinned upstream files on this host (a cache; a file
  # whose digest does not match refuses), then ship the verified copy to that cluster's shared root and verify it there
  tp="$HERE/lib/third_party.sh"
  [ "$REPRO_DRY_RUN" = 1 ] || [ -f "$tp" ] || repro_fatal "third_party: $tp is missing from this checkout -- nothing fetched, nothing installed"
  for l in $labels; do
    repro_say "third_party -> $l: the G0DM0D3 templates and the refusal-direction fit prompts, fetched and verified here, installed under the cluster's shared root"
    repro_run bash "$tp" --cluster "$l" --config "$SISTEER_CONFIG" $(repro_dry_flag) \
      || repro_fatal "third_party: lib/third_party.sh failed for $l (rc $?) -- the grok420 attack and stage 04's refusal direction cannot run there until it succeeds"
    repro_mark_done setup third_party "$l"
  done
fi

if want mirror; then
  repro_require_clean_tree
  for l in $labels; do
    repro_say "mirror -> $l: the repo mirror + the clean checkout at $SHA"
    repro_run "$HERE/lib/mirror_repo.sh" --cluster "$l" --sha "$SHA" --config "$SISTEER_CONFIG" $(repro_dry_flag)
  done
fi

if want checkpoints; then
  repro_require_model
  fit_repo="$(repro_model "$REPRO_MODEL" fit.checkpoint.hf_repo)"; fit_rev="$(repro_model "$REPRO_MODEL" fit.checkpoint.revision)"
  srv_repo="$(repro_model "$REPRO_MODEL" serve.checkpoint.hf_repo)"; srv_rev="$(repro_model "$REPRO_MODEL" serve.checkpoint.revision)"
  for l in $labels; do
    repro_say "checkpoints -> $l: $fit_repo@$fit_rev, $srv_repo@$srv_rev, attacker $SISTEER_ATTACKER_MODEL@$SISTEER_ATTACKER_REVISION, the reference tokenizers"
    for pair in "$fit_repo $fit_rev" "$srv_repo $srv_rev" "$SISTEER_ATTACKER_MODEL $SISTEER_ATTACKER_REVISION"; do
      set -- $pair
      repro_run "$HERE/lib/hf_fetch.sh" --cluster "$l" --repo "$1" --revision "$2" --config "$SISTEER_CONFIG" $(repro_dry_flag)
    done
    # the three reference tokenizers (corpora/TOKENIZERS.json: {tokenizers: {name: {repo, revision, files}}, include: [...]}): the
    # pinned FILES only, one repo per call
    if [ -f "$REPRO_REPO/corpora/TOKENIZERS.json" ]; then
      while read -r trepo trev; do
        [ -n "$trepo" ] || continue
        repro_run "$HERE/lib/hf_fetch.sh" --cluster "$l" --repo "$trepo" --revision "$trev" --include 'tokenizer*' --include special_tokens_map.json --config "$SISTEER_CONFIG" $(repro_dry_flag)
      done < <("$(repro_python)" -c 'import json,sys; d=json.load(open(sys.argv[1])); t=d.get("tokenizers", {}); [print(v["repo"], v["revision"]) for v in (t.values() if isinstance(t, dict) else t)]' "$REPRO_REPO/corpora/TOKENIZERS.json")
    else
      repro_note "corpora/TOKENIZERS.json not present in this tree yet (workstream B) — the reference tokenizers are fetched by 02_prepare_model.sh --step corpora"
    fi
    # HarmBench at the harness's pinned revision (sisteer.eval.jailbreak_data.HARMBENCH_REPO / HARMBENCH_REVISION) and tatsu-lab/alpaca
    # (sisteer.data.prompts.load_hf_prompts, unpinned upstream: fetched at refs/main and the resolved snapshot recorded by the fetch), read
    # OFFLINE from the cache — datasets, one repo per call. Stage 04's refusal direction reads them only when SISTEER_REFUSAL_HARMFUL /
    # SISTEER_REFUSAL_HARMLESS name the builtin harmbench_val / alpaca; its default sets are the paper fit's own draw (--what third_party)
    hb="$(PYTHONPATH="$REPRO_REPO/src" "$(repro_python)" -c 'from sisteer.eval.jailbreak_data import HARMBENCH_REPO, HARMBENCH_REVISION; print(HARMBENCH_REPO, HARMBENCH_REVISION)' 2>/dev/null || echo "mariagrandury/harmbench 4aa1f46a9d43e441e238c1d01dc52413094fc84e")"
    repro_run "$HERE/lib/hf_fetch.sh" --cluster "$l" --repo "${hb%% *}" --revision "${hb##* }" --repo-type dataset --config "$SISTEER_CONFIG" $(repro_dry_flag)
    repro_run "$HERE/lib/hf_fetch.sh" --cluster "$l" --repo tatsu-lab/alpaca --repo-type dataset --config "$SISTEER_CONFIG" $(repro_dry_flag)
  done
fi
repro_say "00_setup done (what=$WHAT clusters=$labels)"
