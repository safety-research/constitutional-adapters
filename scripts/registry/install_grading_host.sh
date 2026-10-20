#!/usr/bin/env bash
# install_grading_host.sh — lay the registry out on the grading host and install its `systemd --user` timers, the unit files rendered
# for this site from the templates under systemd/user/*.in.
#
#   scripts/registry/install_grading_host.sh [--dry-run] [--config FILE] [--root DIR] [--unit-dir DIR] [--no-enable] [--no-systemctl]
#                                      [--with-paper-rebuild] [--paper-checkout DIR] [--units-only] [--skip-timer NAME ...]
#
# What it does, in order (every step printed; --dry-run prints, the rendered units included, and changes nothing):
#   0. reads the site config (--config FILE, else $SISTEER_CONFIG, else ${SISTEER_SITE_DIR:-~/.config/sisteer}/config.env, the
#      front door's own resolution in reproduce/lib/common.sh) and resolves the values the templates name, refusing by name before
#      anything is written. A value already in the environment wins over the site config's.
#        @REPO@           $SISTEER_REPO, else this checkout; the templates are read from <repo>/systemd/user
#        @PYTHON@         $REGISTRY_PY, else $SISTEER_PYTHON, else <repo>/.venv/bin/python (the environment `uv sync` builds)
#        @REGISTRY_ROOT@  --root, else $REGISTRY_ROOT, else $SISTEER_REGISTRY_ROOT
#        @STORE_DEST@     the primary cluster's store destination, from clusters/<$SISTEER_PRIMARY_CLUSTER>.env read through
#                         reproduce/lib/clusters.py: its CLUSTER_STORE_DEST when set, else its registry label CLUSTER_SUBSTRATE,
#                         else (the generic substrate `slurm`, which names no one cluster) its CLUSTER_LABEL. It must be a
#                         destination the store configuration declares, because `stores mirror --dest` refuses any other: the file
#                         `stores mirror` reads ($REGISTRY_CLUSTER_READS, else <repo>/configs/registry/cluster_reads.yaml), read
#                         with the interpreter that reads the cluster blocks. A checkout without that file, a file that declares
#                         no destinations, or a dry run whose interpreter has no PyYAML yet, says it was not checked
#        @UNIT_ENV_HOOK@  $SISTEER_UNIT_ENV_HOOK: a command whose output is shell `export` lines (a secret-manager wrapper, say),
#                         evaluated by the gate-baseline tick before it reads the git origin; unset, no unit evaluates anything.
#                         An installed unit whose ExecStart evaluates a hook is never replaced by one that evaluates none: with no
#                         hook named the install refuses by name (remove that unit first to install it without one)
#      A path under $HOME renders as `%h/<rest>` (systemd's home specifier), so no unit names an absolute home directory. A path
#      with a character outside [A-Za-z0-9._/+-] refuses by name, and so does a template that names any other @VALUE@.
#   1. mkdir -p <root>/{records,inbox,rulings,snapshots,refused,state,manifests}: `inbox` is a REAL directory (the write-only push
#      key's rrsync root checks realpath); the page directory ~/figs/results-registry.
#   2. `registry config --write-default` (config.toml, absent → written; present → left alone) and `registry init` (an empty
#      database with every migration applied; present → left alone).
#   3. renders systemd/user/registry-*.{service,timer}.in into the unit directory (--unit-dir, else $REGISTRY_UNIT_DIR, else
#      ~/.config/systemd/user; mode 0644),
#      `systemctl --user daemon-reload`, `systemd-analyze --user verify` them.
#   4. `systemctl --user enable --now` the timers: harvest (30 s after each cycle), cluster-reads (10 min: the periodic cluster
#      read's tick), snapshot (5 min), mirrors (5 min), mirrors-full (weekly: every entry of each cluster mirror hashed as a Slurm
#      step), status (10 min), status-heavy (6 h), gate-baseline (2 min: the Slurm gate's baseline follows origin/main,
#      scripts/registry/landing/baseline_refresh.sh), store-mirror (30 min), store-mirror-full (weekly, Sun 03:00 UTC: every file
#      read and every published generation re-hashed on the cluster). registry-paper-rebuild.timer is installed but enabled ONLY
#      with --with-paper-rebuild. --skip-timer NAME leaves one timer out of that set (repeatable).
#
# --units-only: step 3 alone, render the unit files and `daemon-reload` (+ verify); NOTHING is enabled or started and no layout,
#   config or database is touched. The landing window (scripts/registry/landing/window.sh) runs it with the timers stopped; the
#   window's EXIT trap restarts the timers that were active plus the newly installed ones, never registry-snapshot.timer. Every
#   failure exits non-zero (set -e): the window aborts on it. With --no-systemctl and a scratch --unit-dir it renders the units
#   and touches nothing else: `diff -r <scratch> ~/.config/systemd/user` shows what an install would change.
#
# Secrets: the registry needs NONE and this script writes none. Every unit runs secret-free, but the gate-baseline tick evaluates
# the site's environment hook when the site names one. ssh to the clusters goes through the agent socket. Nothing here touches a
# cluster.
set -euo pipefail
shopt -u patsub_replacement 2>/dev/null || true      # bash 5.2: an `&` in a substituted value stays a literal `&`

DRY=0; ROOT_ARG=""; UNIT_DIR="${REGISTRY_UNIT_DIR:-$HOME/.config/systemd/user}"; ENABLE=1; SYSTEMCTL=1; PAPER_REBUILD=0
PAPER_CHECKOUT=""; UNITS_ONLY=0; SKIP_TIMERS=(); CONFIG_ARG=""
while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) DRY=1 ;;
    --config) CONFIG_ARG="$2"; shift ;;
    --root) ROOT_ARG="$2"; shift ;;
    --unit-dir) UNIT_DIR="$2"; shift ;;
    --no-enable) ENABLE=0 ;;
    --no-systemctl) SYSTEMCTL=0; ENABLE=0 ;;
    --with-paper-rebuild) PAPER_REBUILD=1 ;;
    --paper-checkout) PAPER_CHECKOUT="$2"; shift ;;
    --units-only) UNITS_ONLY=1 ;;
    --skip-timer) SKIP_TIMERS+=("$2"); shift ;;
    -h|--help) sed -n '2,49p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
  shift
done

HERE="$(cd "$(dirname "$0")" && pwd)"
SELF_REPO="$(cd "$HERE/../.." && pwd)"
refuse(){ echo "REFUSED: install: $*" >&2; exit 2; }
run() { echo "+ $*"; if [ "$DRY" = 0 ]; then "$@"; fi; }

# --- 0. the site and the values the templates name (every refusal before anything is written) ----------------------------------
# the environment's values first: an exported value wins over the site config's
ENV_REPO="${SISTEER_REPO:-}"; ENV_PYTHON="${SISTEER_PYTHON:-}"; ENV_ROOT="${SISTEER_REGISTRY_ROOT:-}"
ENV_PRIMARY="${SISTEER_PRIMARY_CLUSTER:-}"; ENV_HOOK="${SISTEER_UNIT_ENV_HOOK:-}"
# shellcheck source=../../reproduce/lib/common.sh
source "$SELF_REPO/reproduce/lib/common.sh"          # repro_config_path: the front door's ONE resolution of the site config file
REPRO_CONFIG="$CONFIG_ARG"
CFG="$(repro_config_path)"
[ -r "$CFG" ] || refuse "no site config at $CFG: the unit files are rendered from it. Copy reproduce/site.example/ to \$SISTEER_SITE_DIR (default ~/.config/sisteer) and edit it, or pass --config FILE"
CFG="$(cd "$(dirname "$CFG")" && pwd)/$(basename "$CFG")"
# the five keys this script reads, from the file sourced in a subshell, so nothing else of it reaches this script
eval "$(set +eu; set -a; . "$CFG" >/dev/null 2>&1
        for k in SISTEER_REPO SISTEER_PYTHON SISTEER_REGISTRY_ROOT SISTEER_PRIMARY_CLUSTER SISTEER_UNIT_ENV_HOOK; do
          printf 'SITE_%s=%q\n' "$k" "${!k:-}"
        done)"
pick(){ if [ -n "$1" ]; then printf '%s' "$1"; else printf '%s' "$2"; fi; }
expand_home(){ case "$1" in "~") printf '%s' "$HOME" ;; "~/"*) printf '%s/%s' "$HOME" "${1#\~/}" ;; *) printf '%s' "$1" ;; esac; }
PATH_RE='^[A-Za-z0-9._/+-]+$'
unit_path(){   # <absolute path> <the key that named it> -> its spelling in a unit file: `%h/<rest>` under $HOME
  local p="$1" h="${HOME%/}" out
  case "$p" in /*) ;; *) refuse "$2=$p is not an absolute path (a unit file needs one)" ;; esac
  [ "$p" = / ] || p="${p%/}"
  out="$p"
  if [ -n "$h" ]; then
    case "$p" in "$h") out="%h" ;; "$h"/*) out="%h${p#"$h"}" ;; esac
  fi
  [[ "${out#%h}" =~ $PATH_RE ]] || [ "$out" = "%h" ] \
    || refuse "$2=$p holds a character a unit file cannot carry as written (only [A-Za-z0-9._/+-] outside the home directory); move it or name another path"
  printf '%s' "$out"
}
project_env(){ local e="${UV_PROJECT_ENVIRONMENT:-}"; if [ -z "$e" ]; then printf '%s' "$1/.venv"; else case "$e" in /*) printf '%s' "$e" ;; *) printf '%s' "$1/$e" ;; esac; fi; }

REPO="$(expand_home "$(pick "$ENV_REPO" "$SITE_SISTEER_REPO")")"; REPO="${REPO:-$SELF_REPO}"
[ -d "$REPO/src/sisteer/registry" ] || refuse "SISTEER_REPO=$REPO is not a checkout of this repository (no src/sisteer/registry under it)"
REPO="$(cd "$REPO" && pwd)"
UNITS_SRC="$REPO/systemd/user"

SITE_PY="$(expand_home "$(pick "$ENV_PYTHON" "$SITE_SISTEER_PYTHON")")"
PY="${REGISTRY_PY:-}"; [ -n "$PY" ] || PY="$SITE_PY"; [ -n "$PY" ] || PY="$(project_env "$REPO")/bin/python"
case "$PY" in /*) ;; *) PY="$(command -v "$PY")" || refuse "the interpreter '$PY' is not on PATH (set SISTEER_PYTHON to an absolute path)" ;; esac

ROOT="$ROOT_ARG"; [ -n "$ROOT" ] || ROOT="${REGISTRY_ROOT:-}"; [ -n "$ROOT" ] || ROOT="$(pick "$ENV_ROOT" "$SITE_SISTEER_REGISTRY_ROOT")"
[ -n "$ROOT" ] || refuse "no registry root: the site config $CFG sets no SISTEER_REGISTRY_ROOT, and neither --root nor \$REGISTRY_ROOT names one"
ROOT="$(expand_home "$ROOT")"

PRIMARY="$(pick "$ENV_PRIMARY" "$SITE_SISTEER_PRIMARY_CLUSTER")"
[ -n "$PRIMARY" ] || refuse "the site config $CFG sets no SISTEER_PRIMARY_CLUSTER: the store-mirror units name that cluster's store destination"
# the cluster blocks are read through their one reader (stdlib only): the site's interpreter when it exists, else python3
CLPY=""; for c in "$SITE_PY" "$(project_env "$SELF_REPO")/bin/python"; do [ -n "$c" ] && [ -x "$c" ] && { CLPY="$c"; break; }; done
[ -n "$CLPY" ] || CLPY="$(command -v python3)" || refuse "no python3 to read the cluster blocks with"
block_get(){ "$CLPY" "$SELF_REPO/reproduce/lib/clusters.py" --config "$CFG" --get "$PRIMARY" "$1"; }
SUBSTRATE="$(block_get CLUSTER_SUBSTRATE)" \
  || refuse "the site's cluster blocks did not load (reason above): the store-mirror units need clusters/$PRIMARY.env's store destination"
# the store destination: CLUSTER_STORE_DEST when the block sets one, else the registry label CLUSTER_SUBSTRATE, else (the generic
# substrate `slurm`, which names no one cluster) the block's CLUSTER_LABEL, the name its records carry
STORE_DEST="$(block_get CLUSTER_STORE_DEST)" || refuse "clusters/$PRIMARY.env could not be read for CLUSTER_STORE_DEST (reason above)"
if [ -z "$STORE_DEST" ]; then
  if [ "$SUBSTRATE" = slurm ]; then
    STORE_DEST="$(block_get CLUSTER_LABEL)" || refuse "clusters/$PRIMARY.env could not be read for CLUSTER_LABEL (reason above)"
  else
    STORE_DEST="$SUBSTRATE"
  fi
fi
[[ "$STORE_DEST" =~ ^[a-z][a-z0-9_]*$ ]] || refuse "clusters/$PRIMARY.env: the store destination '$STORE_DEST' is not a registry label ([a-z][a-z0-9_]*)"

HOOK="$(expand_home "$(pick "$ENV_HOOK" "$SITE_SISTEER_UNIT_ENV_HOOK")")"

R_REPO="$(unit_path "$REPO" SISTEER_REPO)"
R_PY="$(unit_path "$PY" SISTEER_PYTHON)"
R_ROOT="$(unit_path "$ROOT" SISTEER_REGISTRY_ROOT)"
R_HOOK=""
if [ -n "$HOOK" ]; then R_HOOK_PATH="$(unit_path "$HOOK" SISTEER_UNIT_ENV_HOOK)"; R_HOOK="eval \"\`$R_HOOK_PATH\`\"; "; fi
if [ "$DRY" = 0 ]; then
  [ -x "$PY" ] || refuse "no interpreter at $PY (reproduce/00_setup.sh --what venvs builds <repo>/.venv; or set SISTEER_PYTHON): the units would run it"
  [ -z "$HOOK" ] || [ -x "$HOOK" ] || refuse "SISTEER_UNIT_ENV_HOOK=$HOOK is not an executable file: the gate-baseline tick would run it"
fi

# the store destination is one the store configuration declares: the file `stores mirror` reads, its destination names read (nothing
# else of the file is checked here) with the interpreter that read the cluster blocks
STORES_CFG="${REGISTRY_CLUSTER_READS:-$REPO/configs/registry/cluster_reads.yaml}"
if [ -z "${REGISTRY_CLUSTER_READS:-}" ] && [ ! -e "$STORES_CFG" ]; then
  echo "# store destination $STORE_DEST not checked: the checkout $REPO carries no configs/registry/cluster_reads.yaml"
elif ! "$CLPY" -I -c 'import yaml' 2>/dev/null; then
  [ "$DRY" = 1 ] || refuse "$CLPY cannot import yaml (PyYAML), which reading the store configuration $STORES_CFG needs: set SISTEER_PYTHON to the registry's interpreter"
  echo "# store destination $STORE_DEST not checked: $CLPY has no PyYAML yet (a dry run)"
else
  DECLARED="$("$CLPY" -I - "$STORES_CFG" <<'PY'
import sys
import yaml
path = sys.argv[1]
try:
    with open(path, encoding="utf-8") as fh:
        doc = yaml.safe_load(fh)
except (OSError, yaml.YAMLError) as exc:
    sys.exit(f"{path} did not load: {exc}")
dests = doc.get("destinations") if isinstance(doc, dict) else None
print(" ".join(sorted(str(k) for k in dests)) if isinstance(dests, dict) else "")
PY
)" || refuse "the store configuration $STORES_CFG could not be read (reason above): the store-mirror units name one of its destinations"
  if [ -z "$DECLARED" ]; then
    echo "# store destination $STORE_DEST not checked: $STORES_CFG declares no destinations"
  else
    case " $DECLARED " in
      *" $STORE_DEST "*) ;;
      *) refuse "clusters/$PRIMARY.env: the store destination '$STORE_DEST' is not declared in $STORES_CFG (declared: $DECLARED), so \`stores mirror --dest $STORE_DEST\` would refuse every run. Set CLUSTER_STORE_DEST in that block to a declared destination, or declare it there" ;;
    esac
  fi
fi

render_unit(){   # <template> -> the unit file on stdout, every @VALUE@ filled
  local t
  t="$(cat "$1")"
  t="${t//@REPO@/$R_REPO}"; t="${t//@PYTHON@/$R_PY}"; t="${t//@REGISTRY_ROOT@/$R_ROOT}"
  t="${t//@STORE_DEST@/$STORE_DEST}"; t="${t//@UNIT_ENV_HOOK@/$R_HOOK}"
  printf '%s\n' "$t"
}

export PYTHONPATH="${REGISTRY_PYTHONPATH:-$REPO/src}${PYTHONPATH:+:$PYTHONPATH}"
# the default timer set (scripts/registry/landing/window.sh names the same set minus registry-snapshot.timer; the units test pins the two equal)
TIMERS=(registry-harvest.timer registry-cluster-reads.timer registry-snapshot.timer registry-mirrors.timer registry-mirrors-full.timer registry-status.timer registry-status-heavy.timer registry-gate-baseline.timer registry-store-mirror.timer registry-store-mirror-full.timer)
if [ ${#SKIP_TIMERS[@]} -gt 0 ]; then
  kept=()
  for t in "${TIMERS[@]}"; do skip=0; for s in "${SKIP_TIMERS[@]}"; do [ "$t" = "$s" ] && skip=1; done; [ $skip = 0 ] && kept+=("$t"); done
  TIMERS=("${kept[@]}")
fi
PAGE_DIR="$HOME/figs/results-registry"

# the units: EVERY template rendered into a staging directory and checked before step 1, so a template this script cannot fill
# changes nothing at all
STAGE="$(mktemp -d "${TMPDIR:-/tmp}/registry-units.XXXXXX")"
trap 'rm -rf "$STAGE"' EXIT
NAMES=()
for tmpl in "$UNITS_SRC"/registry-*.service.in "$UNITS_SRC"/registry-*.timer.in; do
  [ -f "$tmpl" ] || refuse "no unit templates under $UNITS_SRC (registry-*.service.in, registry-*.timer.in)"
  name="$(basename "$tmpl" .in)"
  render_unit "$tmpl" > "$STAGE/$name"
  left="$(grep -oE '@[A-Z][A-Z_]*@' "$STAGE/$name" | sort -u | tr '\n' ' ' || true)"
  [ -z "$left" ] || refuse "$tmpl names ${left% }, which this installer does not fill (it fills @REPO@ @PYTHON@ @REGISTRY_ROOT@ @STORE_DEST@ @UNIT_ENV_HOOK@); nothing installed"
  # an installed unit that evaluates an environment hook keeps it: a rendered unit that would run without one refuses
  if [ -f "$UNIT_DIR/$name" ] && grep -q '^ExecStart=.*eval "`' "$UNIT_DIR/$name" && ! grep -q '^ExecStart=.*eval "`' "$STAGE/$name"; then
    refuse "the installed $UNIT_DIR/$name evaluates an environment hook ($(sed -n 's/^ExecStart=.*eval "`\([^`]*\)`".*/\1/p' "$UNIT_DIR/$name" | head -n 1)) before it runs, and the site config $CFG names no SISTEER_UNIT_ENV_HOOK, so the rendered unit would run without it. Set SISTEER_UNIT_ENV_HOOK to that command (or remove the installed unit first to install it without one); nothing installed"
  fi
  NAMES+=("$name")
done

echo "# registry install: root=$ROOT units=$UNIT_DIR python=$PY repo=$REPO site=$CFG store_dest=$STORE_DEST dry_run=$DRY units_only=$UNITS_ONLY"

if [ "$UNITS_ONLY" = 0 ]; then
# 1. layout
for d in records inbox rulings snapshots refused state manifests; do run mkdir -p "$ROOT/$d"; done
run mkdir -p "$PAGE_DIR"

# 2. config + database
cfg_args=(--root "$ROOT" config --write-default)
[ -n "$PAPER_CHECKOUT" ] && cfg_args+=(--paper-checkout "$PAPER_CHECKOUT")
run "$PY" -m sisteer.registry.cli "${cfg_args[@]}"
run "$PY" -m sisteer.registry.cli --root "$ROOT" init
fi

# 3. units: the rendered files staged above, installed (the dry run prints each one it would install)
run mkdir -p "$UNIT_DIR"
for name in "${NAMES[@]}"; do
  if [ "$DRY" = 1 ]; then echo "# rendered $name:"; grep -v -e '^#' -e '^$' "$STAGE/$name" | sed 's/^/#   /' || true; fi
  run install -m 0644 "$STAGE/$name" "$UNIT_DIR/$name"
done
if [ "$SYSTEMCTL" = 1 ]; then
  run systemctl --user daemon-reload
  if command -v systemd-analyze >/dev/null 2>&1; then
    run systemd-analyze --user verify "$UNIT_DIR"/registry-*.service "$UNIT_DIR"/registry-*.timer
  fi
fi

# 4. timers
if [ "$UNITS_ONLY" = 1 ]; then
  echo "# units installed and reloaded; nothing enabled or started (--units-only)"
  exit 0
fi
if [ "$ENABLE" = 1 ]; then
  run systemctl --user enable --now "${TIMERS[@]}"
  if [ "$PAPER_REBUILD" = 1 ]; then
    run systemctl --user enable --now registry-paper-rebuild.timer
  else
    echo "# registry-paper-rebuild.timer installed but NOT enabled (enable it with --with-paper-rebuild)"
  fi
  run systemctl --user list-timers --no-pager 'registry-*'
else
  echo "# timers not enabled (--no-enable / --no-systemctl); enable with: systemctl --user enable --now ${TIMERS[*]}"
fi
echo "# done. status page: $PAGE_DIR/index.html ; config: $ROOT/config.toml ; journal: journalctl --user -u 'registry-*' -f"
