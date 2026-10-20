# site_env.sh — the cluster names and paths the landing scripts need, read from the site config. Sourced, never executed:
#
#   source "$HERE/site_env.sh"
#   site_env_load CLUSTER_HOST CLUSTER_SHARED MIRROR_REPOS_CLUSTER || exit 2     # the keys this script needs
#
# Each cluster in the site config (config.env plus clusters/<label>.env, the files reproduce/lib/clusters.py reads) carries
# three names, and each name has one job:
#   site label       the clusters/<label>.env file name: the front door's per-cluster loops
#   registry label   CLUSTER_SUBSTRATE: the registry config's [[clusters]] name, every --dest, the substrate of every record
#   mirror selector  CLUSTER_MIRROR_SELECTOR (optional; default the site label): only the argument the site's repo-mirror tool
#                    takes (mirror-repos.sh --cluster), which knows the clusters by its own names
#
# site_env_load resolves each key it is given and exports it. A value already in the environment wins; otherwise it is read
# from ONE cluster block, so the host, the shared root, the mirror selector and the clone always describe the same cluster:
#   - CLUSTER_DEST set (a registry label): the block that carries it, among every clusters/<label>.env in the site directory,
#     listed in SISTEER_CLUSTERS or not (that list is the front door's default fan-out; a landing names its cluster by registry
#     label). A label no block carries, one two blocks carry, or a site label in its place is refused by name
#   - CLUSTER_DEST unset: the primary cluster's block (SISTEER_PRIMARY_CLUSTER of the site config), which also supplies it
# The keys:
#   CLUSTER_HOST         the login host the grading host reaches over ssh (CLUSTER_SSH)
#   CLUSTER_SHARED        the cluster's shared root (CLUSTER_SHARED_ROOT)
#   CLUSTER_DEST          the cluster's registry label (CLUSTER_SUBSTRATE)
#   MIRROR_REPOS_CLUSTER  the mirror selector
#   SITE_REPO             the grading host's checkout: SISTEER_REPO, else the site config's SISTEER_REPO, else the main
#                         working tree of the repository this file belongs to
#   MIRROR_REPOS_NAME     the checkout's directory name: the repo-mirror tool keys each clone by it
#   CLUSTER_REPO_MIRROR   the clone on the cluster: the block's CLUSTER_REPO_MIRROR when the block sets it, else
#                         $CLUSTER_SHARED/repos/$MIRROR_REPOS_NAME (the path the mirror tool writes)
#   SITE_CONFIG_FILE      the site config file the values came from (empty when every value came from the environment)
# The site config file is $SISTEER_CONFIG, else ${SISTEER_SITE_DIR:-~/.config/sisteer}/config.env, the front door's own
# resolution. It is read only when a key is not already set. A key that resolves to nothing is refused by name, with the
# setting that supplies it, and site_env_load returns 2: there is no built-in host, shared root or cluster name.
#
# The same names derive the same way in Python: sisteer.registry.reads_protocol.gate_venv_path names the gate venv
# <shared>/envs/<checkout name>-gate-<hash12>, as site_env_gate_venv does here.

SITE_ENV_HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SITE_ENV_CLUSTERS_PY="${SITE_ENV_CLUSTERS_PY:-$SITE_ENV_HERE/../../../reproduce/lib/clusters.py}"
SITE_ENV_KEYS="CLUSTER_HOST CLUSTER_SHARED CLUSTER_DEST MIRROR_REPOS_CLUSTER SITE_REPO MIRROR_REPOS_NAME CLUSTER_REPO_MIRROR"

# the site config file the front door would read (reproduce/lib/common.sh repro_config_path)
site_env_config_path() {
  if [ -n "${SISTEER_CONFIG:-}" ]; then printf '%s\n' "$SISTEER_CONFIG"
  else printf '%s/config.env\n' "${SISTEER_SITE_DIR:-$HOME/.config/sisteer}"; fi
}

# the main working tree of the repository holding this file (a linked worktree resolves to the checkout it was added from)
site_env_main_checkout() {
  local common
  common=$(git -C "$SITE_ENV_HERE" rev-parse --path-format=absolute --git-common-dir 2>/dev/null) || return 1
  case "$common" in */.git) printf '%s\n' "${common%/.git}" ;; *) return 1 ;; esac
}

# the gate venv for a dependency hash: <shared>/envs/<checkout name>-gate-<hash12>
site_env_gate_venv() { printf '%s/envs/%s-gate-%s\n' "$1" "$2" "$3"; }

# one cluster's block as shell assignments (SITE_BLOCK_*), through the one reader of cluster blocks: the block that carries the
# registry label $2 when it is not empty (a lookup by label: every block file in the site directory), else the primary cluster's
site_env_read_block() {
  local cfg="$1" dest="${2:-}"
  python3 - "$SITE_ENV_CLUSTERS_PY" "$cfg" "$dest" <<'PY'
import importlib.util, os, re, shlex, sys
spec = importlib.util.spec_from_file_location("site_clusters", sys.argv[1])
cl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cl)
cfg_path = os.path.abspath(sys.argv[2])
dest = sys.argv[3]
site_dir = os.path.dirname(cfg_path)
try:
    config = cl.site_config_file(cfg_path)
    if dest:
        blocks = cl.load_addressable(site_dir, config)
        try:
            label = cl.resolve_label(blocks, dest)
        except SystemExit as exc:
            why = str(exc.code)
            raise cl.ClusterConfigError(f"CLUSTER_DEST={dest}: {why[len('REFUSED: '):] if why.startswith('REFUSED: ') else why}") from None
        if blocks[label].get("CLUSTER_SUBSTRATE", "") != dest:
            raise cl.ClusterConfigError(f"CLUSTER_DEST={dest} is the site label of clusters/{label}.env, whose registry label is "
                                        f"{blocks[label].get('CLUSTER_SUBSTRATE') or 'not set'}; CLUSTER_DEST takes the registry label")
    else:
        label = config.get("SISTEER_PRIMARY_CLUSTER", "")
        if not label:
            raise cl.ClusterConfigError(f"{cfg_path} sets no SISTEER_PRIMARY_CLUSTER (the cluster the landing scripts use)")
        blocks = cl.load_all(site_dir, config)
except SystemExit as exc:
    msg = str(exc.code if exc.code is not None else exc)
    print(f"site_env_error={shlex.quote(msg[len('REFUSED: '):] if msg.startswith('REFUSED: ') else msg)}")
    sys.exit(0)
block = blocks[label]
raw = os.path.join(site_dir, "clusters", f"{label}.env")
explicit = any(re.match(r"^\s*(?:export\s+)?CLUSTER_REPO_MIRROR=", ln) for ln in open(raw, encoding="utf-8"))
out = {"LABEL": label, "SSH": block.get("CLUSTER_SSH", ""), "SHARED_ROOT": block.get("CLUSTER_SHARED_ROOT", ""),
       "SUBSTRATE": block.get("CLUSTER_SUBSTRATE", ""), "MIRROR_SELECTOR": block.get("CLUSTER_MIRROR_SELECTOR", ""),
       "REPO_MIRROR": block.get("CLUSTER_REPO_MIRROR", "") if explicit else "", "SISTEER_REPO": config.get("SISTEER_REPO", "")}
for k, v in out.items():
    print(f"SITE_BLOCK_{k}={shlex.quote(str(v))}")
PY
}

site_env_refuse() { echo "REFUSED: site_env: $*" >&2; return 2; }

site_env_load() {
  local want="${*:-$SITE_ENV_KEYS}" k need_block=0 cfg="" site_env_error=""
  local SITE_BLOCK_LABEL="" SITE_BLOCK_SSH="" SITE_BLOCK_SHARED_ROOT="" SITE_BLOCK_SUBSTRATE="" SITE_BLOCK_MIRROR_SELECTOR=""
  local SITE_BLOCK_REPO_MIRROR="" SITE_BLOCK_SISTEER_REPO=""
  for k in $want; do
    case " $SITE_ENV_KEYS " in *" $k "*) ;; *) site_env_refuse "unknown key $k (known: $SITE_ENV_KEYS)"; return 2 ;; esac
  done
  # the keys the block supplies (CLUSTER_REPO_MIRROR and the repo name derive from others; SITE_REPO falls back to git)
  for k in $want; do
    case "$k" in
      CLUSTER_HOST|CLUSTER_SHARED|CLUSTER_DEST|MIRROR_REPOS_CLUSTER) [ -n "${!k:-}" ] || need_block=1 ;;
      CLUSTER_REPO_MIRROR) [ -n "${CLUSTER_REPO_MIRROR:-}" ] || need_block=1 ;;
    esac
  done
  case " $want " in
    *" SITE_REPO "*|*" MIRROR_REPOS_NAME "*|*" CLUSTER_REPO_MIRROR "*)
      [ -n "${SISTEER_REPO:-}" ] || [ -n "${SITE_REPO:-}" ] || need_block=1 ;;
  esac
  cfg="$(site_env_config_path)"
  local dest_named="${CLUSTER_DEST:-}"     # a registry label the caller named: every value comes from that cluster's block
  if [ "$need_block" = 1 ] && [ -r "$cfg" ]; then
    local block_sh
    block_sh="$(site_env_read_block "$cfg" "$dest_named")" || { site_env_refuse "the site config $cfg could not be read (the cluster-block reader failed)"; return 2; }
    eval "$block_sh"
    if [ -n "$site_env_error" ]; then site_env_refuse "the site config $cfg did not load: $site_env_error"; return 2; fi
    SITE_CONFIG_FILE="$cfg"
  else
    SITE_CONFIG_FILE="${SITE_CONFIG_FILE:-}"
  fi
  export SITE_CONFIG_FILE
  local which="the primary cluster's block"
  [ -z "$dest_named" ] || which="the block of the cluster CLUSTER_DEST=$dest_named names"
  local from="$which of the site config ($cfg)"
  [ -r "$cfg" ] || from="$which of a site config (none at $cfg; set SISTEER_CONFIG or SISTEER_SITE_DIR)"
  # every key resolves in this order, whether or not the caller asked for it, because the derived ones read the others
  CLUSTER_HOST="${CLUSTER_HOST:-$SITE_BLOCK_SSH}"
  CLUSTER_SHARED="${CLUSTER_SHARED:-$SITE_BLOCK_SHARED_ROOT}"; CLUSTER_SHARED="${CLUSTER_SHARED%/}"
  CLUSTER_DEST="${CLUSTER_DEST:-$SITE_BLOCK_SUBSTRATE}"
  MIRROR_REPOS_CLUSTER="${MIRROR_REPOS_CLUSTER:-${SITE_BLOCK_MIRROR_SELECTOR:-$SITE_BLOCK_LABEL}}"
  SITE_REPO="${SITE_REPO:-${SISTEER_REPO:-$SITE_BLOCK_SISTEER_REPO}}"
  case "$SITE_REPO" in "~") SITE_REPO="$HOME" ;; "~/"*) SITE_REPO="$HOME/${SITE_REPO#\~/}" ;; esac
  [ -n "$SITE_REPO" ] || SITE_REPO="$(site_env_main_checkout || true)"
  SITE_REPO="${SITE_REPO%/}"
  MIRROR_REPOS_NAME="${MIRROR_REPOS_NAME:-${SITE_REPO:+$(basename "$SITE_REPO")}}"
  if [ -z "${CLUSTER_REPO_MIRROR:-}" ]; then
    if [ -n "$SITE_BLOCK_REPO_MIRROR" ]; then CLUSTER_REPO_MIRROR="$SITE_BLOCK_REPO_MIRROR"
    elif [ -n "$CLUSTER_SHARED" ] && [ -n "$MIRROR_REPOS_NAME" ]; then CLUSTER_REPO_MIRROR="$CLUSTER_SHARED/repos/$MIRROR_REPOS_NAME"
    fi
  fi
  export CLUSTER_HOST CLUSTER_SHARED CLUSTER_DEST MIRROR_REPOS_CLUSTER SITE_REPO MIRROR_REPOS_NAME CLUSTER_REPO_MIRROR
  for k in $want; do
    [ -n "${!k:-}" ] && continue
    case "$k" in
      CLUSTER_HOST) site_env_refuse "no login host: set CLUSTER_SSH in $from, or CLUSTER_HOST in the environment" ;;
      CLUSTER_SHARED) site_env_refuse "no shared root: set CLUSTER_SHARED_ROOT in $from, or CLUSTER_SHARED in the environment" ;;
      CLUSTER_DEST) site_env_refuse "no registry label: set CLUSTER_SUBSTRATE in $from, or CLUSTER_DEST in the environment" ;;
      MIRROR_REPOS_CLUSTER) site_env_refuse "no mirror selector: set CLUSTER_MIRROR_SELECTOR in $from (default its site label), or MIRROR_REPOS_CLUSTER in the environment" ;;
      SITE_REPO) site_env_refuse "no grading-host checkout: set SISTEER_REPO in the site config or the environment (this file is not inside a git checkout)" ;;
      MIRROR_REPOS_NAME) site_env_refuse "no checkout name: set SISTEER_REPO, or MIRROR_REPOS_NAME in the environment" ;;
      CLUSTER_REPO_MIRROR) site_env_refuse "no repo mirror path: set CLUSTER_REPO_MIRROR in $from, or a shared root and a checkout name" ;;
    esac
    return 2
  done
  return 0
}
