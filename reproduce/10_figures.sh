#!/bin/bash
# reproduce/10_figures.sh — stage 10: the paper's figures from the results registry. Builds the registry
# notebook (paper/notebooks/make_registry_notebook.py), executes it with nbconvert against the live registry or the
# committed paper-frame snapshot, and copies the rendered HTML + figures to $SISTEER_FIG_DIR/registry-notebook/.
#
#   reproduce/10_figures.sh [--source live|snapshot|auto] [--judges paper|primary] [--export-snapshot] [--snapshot-root DIR]
#                           [--config F] [--overrides F] [--dry-run] [--help]
#
#   --source          live = $SISTEER_REGISTRY_ROOT/registry.sqlite read-only; snapshot = the committed export under
#                     --snapshot-root (default <repo>/paper/snapshot); auto (default) = live when the database exists, else
#                     the snapshot when its CURRENT exists, else refuse
#   --export-snapshot cut a fresh paper frame first (tools/snapshot_paper_frame.sh --dest-root <snapshot root>
#                     --source-root $SISTEER_REGISTRY_ROOT), then render from it
#   --snapshot-root   where the paper frame lives / is cut (default <repo>/paper/snapshot)
#   --judges          paper (default) = the paper's judges (REGISTRY_NB_JUDGES=rejudge: minimax-m3, glm-5.3 on Petri), the
#                     re-judges filed as second judgments; primary = the filed Claude judges, a robustness check, which may not
#                     write the paper's figures directory: it renders into paper/notebooks/figures-primary/ and copies to
#                     $SISTEER_FIG_DIR/registry-notebook-primary/. A registry of your own graded by stage 08 holds only the
#                     Claude judges, so a paper render of it refuses (PolicyJudgesAbsent): use --judges primary, or
#                     --source snapshot for the committed frame, which carries the paper's judges
#   --panel           accepted for compatibility and ignored: the notebook renders the paper's panels and nothing else
#
# Site config keys read: SISTEER_FIG_DIR (a literal $SHARED in it stands for $HOME on this host), SISTEER_REGISTRY_ROOT,
# SISTEER_NB_PYTHON (the notebook's interpreter; default: the python of the environment
# `uv sync --frozen --extra cpu --group notebook` builds, which is $UV_PROJECT_ENVIRONMENT when that is set and
# <repo>/.venv otherwise: nbconvert, ipykernel, pandas, pyarrow, matplotlib, scipy). The notebook reads REGISTRY_NB_SOURCE (and REGISTRY_NB_SNAPSHOT_ROOT for the snapshot source) and
# shells out to `git rev-parse` for the code revision it prints, so git must be on PATH.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/lib/common.sh"
usage(){ sed -n '2,27p' "$0"; repro_common_help; }
SOURCE=auto; PANEL=""; EXPORT=0; SNAP_ROOT=""; JUDGES=paper
repro_parse_common_args "$@"; set -- "${REPRO_ARGS[@]+"${REPRO_ARGS[@]}"}"
while [ $# -gt 0 ]; do
  case "$1" in
    --source) SOURCE="${2:?--source needs live|snapshot|auto}"; shift 2;;
    --source=*) SOURCE="${1#--source=}"; shift;;
    --panel) PANEL="${2:?--panel needs a value (it is ignored)}"; shift 2;;
    --panel=*) PANEL="${1#--panel=}"; shift;;
    --export-snapshot) EXPORT=1; shift;;
    --snapshot-root) SNAP_ROOT="${2:?--snapshot-root needs a directory}"; shift 2;;
    --snapshot-root=*) SNAP_ROOT="${1#--snapshot-root=}"; shift;;
    --judges) JUDGES="${2:?--judges needs paper|primary}"; shift 2;;
    --judges=*) JUDGES="${1#--judges=}"; shift;;
    *) echo "REFUSED: unknown argument $1" >&2; usage >&2; exit 2;;
  esac
done
if [ "$REPRO_HELP" = 1 ]; then usage; exit 0; fi
case "$SOURCE" in live|snapshot|auto) ;; *) repro_fatal "--source must be live|snapshot|auto, got '$SOURCE'";; esac
case "$JUDGES" in paper|primary) ;; *) repro_fatal "--judges must be paper|primary, got '$JUDGES'";; esac
[ -z "$PANEL" ] || repro_note "--panel $PANEL is accepted for compatibility and ignored: the notebook renders the paper's panels only"
# the site config when there is one (--config, $SISTEER_CONFIG, $SISTEER_SITE_DIR/config.env); without one the defaults below stand
if [ -r "$(repro_config_path)" ]; then repro_load_config; else repro_note "no site config at $(repro_config_path); using the defaults"; fi
REG_ROOT="$(eval echo "${SISTEER_REGISTRY_ROOT:-\$HOME/registry}")"
[ -n "$SNAP_ROOT" ] || SNAP_ROOT="$REPRO_REPO/paper/snapshot"
FIG_DIR="${SISTEER_FIG_DIR:-$HOME/figs}"
case "$FIG_DIR" in *'$SHARED'*) FIG_DIR="${FIG_DIR//\$SHARED/$HOME}"; repro_note "SISTEER_FIG_DIR names \$SHARED (a cluster root); on this host it stands for \$HOME -> $FIG_DIR";; esac
NB_PYTHON="${SISTEER_NB_PYTHON:-$(repro_project_env)/bin/python}"
NB_DIR="$REPRO_REPO/paper/notebooks"
NB_BUILD="$NB_DIR/make_registry_notebook.py"
NB_IPYNB="$NB_DIR/registry_qwen_figures.ipynb"
OUT_DIR="$FIG_DIR/registry-notebook"
NB_FIGS="$NB_DIR/figures"
JUDGE_ENV=("REGISTRY_NB_JUDGES=rejudge")
if [ "$JUDGES" = primary ]; then
  # the filed Claude judges: a render off the paper's judges may not write the paper's figures directory (registry_nb.RENDER_POLICY)
  NB_FIGS="$NB_DIR/figures-primary"; OUT_DIR="$FIG_DIR/registry-notebook-primary"
  JUDGE_ENV=("REGISTRY_NB_JUDGES=primary" "REGISTRY_NB_FIGS=$NB_FIGS")
fi
OVERRIDES_FILE="${SISTEER_RUN_OVERRIDES:-$REPRO_DIR/overrides/paper_defaults.env}"

if [ "$EXPORT" = 1 ]; then
  [ "$SOURCE" != live ] || repro_fatal "--export-snapshot renders from the export it cuts; --source live contradicts it (drop one)"
  SOURCE=snapshot
  EXPORT_ARGS=(--dest-root "$SNAP_ROOT" --source-root "$REG_ROOT")
  [ -z "${REPRO_CONFIG:-}" ] || EXPORT_ARGS+=(--config "$REPRO_CONFIG")
  [ "$REPRO_DRY_RUN" = 1 ] && EXPORT_ARGS+=(--dry-run)
  repro_say "10 figures: cutting the paper frame first ($HERE/tools/snapshot_paper_frame.sh ${EXPORT_ARGS[*]})"
  "$HERE/tools/snapshot_paper_frame.sh" "${EXPORT_ARGS[@]}" || repro_fatal "the paper frame export refused (reason above); nothing rendered"
fi
if [ "$SOURCE" = auto ]; then
  if [ -f "$REG_ROOT/registry.sqlite" ]; then SOURCE=live
  elif [ -f "$SNAP_ROOT/CURRENT" ]; then SOURCE=snapshot
  else repro_fatal "--source auto found neither a registry database at $REG_ROOT/registry.sqlite nor a paper frame at $SNAP_ROOT/CURRENT"; fi
  repro_note "source auto -> $SOURCE"
fi
if [ "$REPRO_DRY_RUN" != 1 ]; then
  [ -x "$NB_PYTHON" ] || repro_fatal "the notebook interpreter $NB_PYTHON is not executable. Build it with 'uv sync --frozen --extra cpu --group notebook' at the repository root (uv installs into \$UV_PROJECT_ENVIRONMENT when that is set, else <repo>/.venv; this stage looks in the same place), or set SISTEER_NB_PYTHON to an interpreter with nbconvert, ipykernel, pandas, pyarrow, matplotlib and scipy"
  [ -f "$NB_BUILD" ] || repro_fatal "no notebook builder at $NB_BUILD"
  command -v git >/dev/null 2>&1 || repro_fatal "git is not on PATH (the notebook's header prints the code revision with 'git rev-parse')"
  case "$SOURCE" in
    live) [ -f "$REG_ROOT/registry.sqlite" ] || repro_fatal "--source live: no registry database at $REG_ROOT/registry.sqlite";;
    snapshot) [ -f "$SNAP_ROOT/CURRENT" ] || repro_fatal "--source snapshot: no paper frame at $SNAP_ROOT/CURRENT (cut one: --export-snapshot, or tools/snapshot_paper_frame.sh)";;
  esac
fi
# no MPLBACKEND here: under ipykernel the default is the inline backend, which embeds every figure in the executed notebook and the
# HTML (an Agg export made nbconvert write pages with no images; the PNG / PDF / SVG files beside the notebook were the only copies)
# the paper's render: the paper's judges (the re-judges, REGISTRY_NB_JUDGES=rejudge) at StrongREJECT 0.5 into the paper's figures
# directory, whatever a calling shell exported for a judge comparison (registry_nb.RENDER_POLICY); --judges primary renders the filed
# Claude judges into figures-primary/. XSTest's strength-battery reads are the committed projections (REGISTRY_NB_STRENGTH_READS unset)
NB_ENV=(env -u REGISTRY_NB_SR_THRESHOLD -u REGISTRY_NB_PETRI_JUDGE -u REGISTRY_NB_PETRI_EXCLUDE -u REGISTRY_NB_FIGS
        -u REGISTRY_NB_STRENGTH_READS -u REGISTRY_NB_SITARGET_PROVISIONAL -u NB_REUSE_CACHE "${JUDGE_ENV[@]}"
        "SISTEER_SRC=$REPRO_REPO/src" "REGISTRY_NB_SOURCE=$SOURCE" "REGISTRY_NB_SNAPSHOT_ROOT=$SNAP_ROOT"
        "SISTEER_REGISTRY_ROOT=$REG_ROOT" "REGISTRY_ROOT=$REG_ROOT" "SISTEER_RUN_OVERRIDES=$OVERRIDES_FILE")
# with --source snapshot the registry root is passed through but never opened; say so rather than print a path that need not exist
REG_SHOWN="$REG_ROOT"; if [ "$SOURCE" = snapshot ]; then REG_SHOWN="(not read)"; fi
repro_say "10 figures: source=$SOURCE judges=$JUDGES registry=$REG_SHOWN snapshot_root=$SNAP_ROOT out=$OUT_DIR python=$NB_PYTHON"
repro_mkdir "$OUT_DIR"
repro_run "${NB_ENV[@]}" "$NB_PYTHON" "$NB_BUILD"
# the notebook's own kernelspec is `python3` (make_registry_notebook.py), the kernelspec ipykernel registers inside whatever interpreter
# runs nbconvert, so a fresh clone needs no kernel of its own; the flag pins the same choice, so a notebook rebuilt by an older generator
# (whose metadata named a maintainer kernel) still executes here instead of refusing with NoSuchKernel
repro_run "${NB_ENV[@]}" "$NB_PYTHON" -m nbconvert --execute --to html --ExecutePreprocessor.kernel_name=python3 \
  --ExecutePreprocessor.timeout=3600 --output-dir "$OUT_DIR" "$NB_IPYNB"
# the notebook writes its PNG/PDF/SVG beside itself (paper/notebooks/figures/, figures-primary/ under --judges primary); the rendered
# HTML lands in $OUT_DIR directly. That figures directory holds only this render's files (the notebook removes an earlier render's
# before zipping: registry_nb.unsave_stale), so a figure or table file in $OUT_DIR that it no longer holds is an earlier render's and
# is removed before the copy
repro_run_sh "if [ -d '$NB_FIGS' ]; then for f in '$OUT_DIR'/*.png '$OUT_DIR'/*.pdf '$OUT_DIR'/*.svg '$OUT_DIR'/*.tex; do [ ! -f \"\$f\" ] || [ -e '$NB_FIGS/'\"\${f##*/}\" ] || rm -f -- \"\$f\"; done; cp -R '$NB_FIGS/.' '$OUT_DIR/'; fi"
if [ "$SOURCE" = snapshot ] && [ "$REPRO_DRY_RUN" != 1 ]; then
  repro_registry_id snapshot "$(cat "$SNAP_ROOT/CURRENT")"
fi
repro_say "10 done: $OUT_DIR (source $SOURCE)"
