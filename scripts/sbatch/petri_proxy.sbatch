#!/bin/bash
#SBATCH --job-name=petri-proxy
#SBATCH --partition=general
#SBATCH --qos=high                    # the standing default; draws on the 16-GPU/user high budget (1 here)
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G                    # ~1 GPU's share; NEVER --mem=0
#SBATCH --time=0-12:00:00             # audit bound ~3-5h, padded ~2x — right-sized for backfill
#SBATCH --output=/workspace-vast/%u/exp/logs/%x_%j.out
#SBATCH --open-mode=append
#SBATCH --no-requeue
#SBATCH --signal=B:TERM@120
#
# The long-running Petri routing-proxy job:
# server-mode vLLM (vllm-lens plugin, worker-side object registry, multi-
# LoRA, prefix caching OFF, enforce-eager) behind the routing proxy, bound
# to THIS NODE's loopback only — no open port; the grading host reaches it
# over an SSH tunnel via the login host:
#
#   ssh -J <login-host> -L <localport>:<this-node>:${PETRI_PROXY_PORT}: ...
#
# SUBMIT (M15): capture the job id and persist it —
#   jid=$(sbatch --parsable scripts/sbatch/petri_proxy.sbatch)
#   python -c "import json,os,time,pathlib; p=pathlib.Path(os.path.expanduser('~/.local/state/petri/jobs.json')); p.parent.mkdir(parents=True,exist_ok=True); d=json.loads(p.read_text()) if p.is_file() else []; d.append({'job_id':'$jid','name':'petri-proxy','submitted':time.time()}); p.write_text(json.dumps(d,indent=2))"
#
# SHUTDOWN (M15): `scancel <job_id>` ONLY, stating the id. There is
# deliberately NO name-based cancel here or anywhere in the petri scripts
# (`scancel --name=...` on a shared user has killed the wrong jobs).
#
# NO --export= anywhere (hard rule 1): all env is set in the body below.
#
# audit-exempt: out-of-lockfile-venv — this job runs from the requirements-
# hash-versioned vllm venv (torch cu129), NOT the project
# lockfile, so the cu128 entry-point guard does not apply here; there is no
# `uv sync` anywhere, and a missing venv fails `uv run --no-sync` loudly.

set -uo pipefail
SHARED="${SISTEER_SHARED_ROOT:?set SISTEER_SHARED_ROOT, the shared root of the cluster (the front door exports it from the site config)}"

# Shared-storage copies only — cluster /home is node-local (invisible on
# whichever node Slurm picks):
source "$SHARED/scripts/hf_env.sh"     # HF_HOME + per-user HF_TOKEN_PATH
export NCCL_NVLS_ENABLE=0
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# The immutable commit-addressed checkout (NEVER the peers' pinned
# repos, NEVER the repo-mirror script — populated by pushing the exact
# petri commit over SSH). Loud if absent.
PETRI_REPO_DIR="${PETRI_REPO_DIR:-${SISTEER_PETRI_REPO_ROOT:?set SISTEER_PETRI_REPO_ROOT, the root of the per-commit Petri checkouts (the front door exports it from the site config)}}"
if [ ! -d "$PETRI_REPO_DIR/src/sisteer" ]; then
  echo "FATAL: no petri checkout at $PETRI_REPO_DIR — push the exact petri" \
       "commit over SSH into it first (plan §15 issue 2); refusing" >&2
  exit 1
fi
cd "$PETRI_REPO_DIR"

# The pinned GPU serving venv (the capeval wheel pin + vllm-lens), built
# from configs/capeval/vllm-requirements.txt; requirements-hash-versioned
# so a re-pin lands in a FRESH venv (capeval review infra #12).
VLLM_REQS="$PETRI_REPO_DIR/configs/capeval/vllm-requirements.txt"
REQ_HASH=$(sha256sum "$VLLM_REQS" | cut -c1-8)
export UV_PROJECT_ENVIRONMENT="/home/$USER/venvs/sisteer-petri-vllm-$REQ_HASH"
export UV_CACHE_DIR="$SHARED/.cache/uv"
export UV_LINK_MODE=copy

# Run-scoped paths (the routing table + worker registry are exported at
# enumeration time by the launcher and land on shared storage):
PETRI_RUN_DIR="${PETRI_RUN_DIR:-$SHARED/petri/serving}"
PETRI_ROUTING_TABLE="${PETRI_ROUTING_TABLE:-$PETRI_RUN_DIR/routing_table.json}"
PETRI_LENS_REGISTRY_DIR="${PETRI_LENS_REGISTRY_DIR:-$PETRI_RUN_DIR/lens_registry}"
# the PINNED chat template file (hashed against the manifest's
# chat_template_hash by build_server_launch — required):
PETRI_CHAT_TEMPLATE="${PETRI_CHAT_TEMPLATE:-$PETRI_RUN_DIR/chat_template.jinja}"
PETRI_VLLM_PORT="${PETRI_VLLM_PORT:-8801}"
PETRI_PROXY_PORT="${PETRI_PROXY_PORT:-8814}"
for f in "$PETRI_ROUTING_TABLE" "$PETRI_CHAT_TEMPLATE"; do
  if [ ! -f "$f" ]; then
    echo "FATAL: missing $f — the routing table (export_routing_table) and" \
         "the pinned chat template land at enumeration time, before this" \
         "job is submitted (plan §2/§5); refusing" >&2
    exit 1
  fi
done

# Persist this job's identity next to the serving record too (M15), so
# the id is recoverable from the cluster side alone:
mkdir -p "$PETRI_RUN_DIR"
printf '{"job_id": "%s", "node": "%s", "vllm_port": %s, "proxy_port": %s, "started": %s}\n' \
  "$SLURM_JOB_ID" "$(hostname)" "$PETRI_VLLM_PORT" "$PETRI_PROXY_PORT" "$(date +%s)" \
  >> "$PETRI_RUN_DIR/proxy_jobs.jsonl"

# Cleanup trap so vllm workers/compile helpers never orphan (repo rule):
cleanup() { kill -TERM -$$ 2>/dev/null; wait; }
trap cleanup SIGTERM SIGINT SIGQUIT

# One srun-wrapped workload: bring up server-mode vLLM (vllm_server
# composes the pinned flags: --enforce-eager, --no-enable-prefix-caching,
# --enable-lora, the verified chat template, the lens registry preload),
# wait for instance-specific readiness, then run the routing proxy in the
# foreground on loopback. The exact-replay mechanism is READ FROM the
# routing table (a stated, recorded choice — never a default); with
# 'batch_invariant' the composed launch env is the proxy's attestation.
srun --ntasks=1 uv run --no-sync python - "$PETRI_ROUTING_TABLE" \
    "$PETRI_LENS_REGISTRY_DIR" "$PETRI_VLLM_PORT" "$PETRI_PROXY_PORT" \
    "$PETRI_CHAT_TEMPLATE" <<'PYEOF'
import sys
from pathlib import Path

from sisteer.capeval import manifest as cm
from sisteer.capeval.serving import proxy as px
from sisteer.capeval.serving import vllm_server as vs

routing_table, registry_dir, vllm_port, proxy_port, chat_template = \
    sys.argv[1:6]
spec = cm.load_petri_spec(
    Path("configs/capeval/manifests/petri_twosi.yaml"))
table = px.load_routing_table(Path(routing_table))
if table["serving_mode"] != spec.serving_mode:
    raise SystemExit(f"routing table mode {table['serving_mode']!r} != "
                     f"manifest mode {spec.serving_mode!r} — stale export")

# LoRA adapters (genuine exports + the zero no-op) live next to the
# registry; absent dir means no LoRA-path arms are live this run.
lora_root = Path(registry_dir).parent / "lora_adapters"
lora_modules = {p.name: p for p in sorted(lora_root.iterdir())
                if p.is_dir()} if lora_root.is_dir() else {}

launch = vs.build_server_launch(
    spec, port=int(vllm_port),
    lens_registry_dir=Path(registry_dir),
    lora_modules=lora_modules,
    exact_replay_strategy=table["exact_replay_strategy"],
    chat_template_path=Path(chat_template),
    # AGENT modality: this server backs the petri audit's tool
    # round-trips, the one workload that needs the manifest's pinned
    # tool-call parser (generation launches compose none).
    tool_calls=True)
# THE PETRI-FACING ENTRY, stated here even though it is the proxy's
# default (review 2026-09-02: this launcher once built the proxy with the
# agent-loop path disabled while every test passed): this proxy fronts
# the audit's agent loop, whose multi-turn tool renders never end at the
# generation header. inspect_ai's client cannot declare petri_agent_loop,
# so the proxy declares for lens routes and verifies each declaration
# against the render. The render probe is the MANIFEST-PINNED served
# tokenizer under the SAME pinned chat template the server hashes —
# loaded EAGERLY, before the server, so a bad pin fails here at startup
# and never at the audit's first tool turn.
served_tok = px.load_served_tokenizer(
    spec.target_model, revision=spec.target_revision,
    chat_template=Path(chat_template).read_text(encoding="utf-8"))
agent_loop_render = px.TokenizerAgentLoopRender(served_tok)
server = vs.launch_server(
    launch, log_path=Path(registry_dir).parent / "vllm_server.log")
try:
    # launch.api_key is the per-launch upstream token: the backend sends
    # it as a Bearer header; it lives nowhere else (not in launch.env,
    # which is recorded below as the attestation, and not on argv).
    rp = px.RoutingProxy(
        table["routes"], px.HTTPBackend(launch.base_url,
                                        api_key=launch.api_key),
        base_model=table["base_model"], serving_mode=table["serving_mode"],
        prompted_prefixes=table["prompted_prefixes"],
        exact_replay_strategy=table["exact_replay_strategy"],
        launch_attestation=launch.env,
        agent_loop_render=agent_loop_render,
        petri_agent_loop_entry=True)
    from aiohttp import web
    import logging
    logging.basicConfig(level=logging.INFO)
    web.run_app(px.make_app(rp), host="127.0.0.1", port=int(proxy_port))
finally:
    server.terminate()
    server.wait(timeout=60)
PYEOF
