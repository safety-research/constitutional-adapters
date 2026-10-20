# Shared attacker pool on Modal

One persistent `modal.Server` runs the unsteered attacker
`huihui-ai/Huihui-Qwen3.6-27B-abliterated` (snapshot
`27502c8717fd5a2f8c0c77188c10c243fd4f672e`) on one H200 under vLLM; the
multi-turn attack drivers share it instead of each holding a mostly idle
attacker GPU. Contract: `src/sisteer/eval/attacks/attacker_pool.py`; deployed
app: `scripts/modal/attacker_pool.py`. The Slurm launchers
`scripts/sbatch/attacker_pool{,_canary}.sbatch` are retired and exit 64 before
any cluster-specific code.

## Geometry and access

| parameter | value |
|---|---|
| `max_model_len` | 19,200 (window sizing below) |
| `max_num_seqs` | 37 |
| `max_num_batched_tokens` | 16,384 (chunked prefill) |
| prefix caching / weights | on / bf16, no offload, CUDA graphs, text only |
| containers | exactly one (`min_containers = max_containers = 1`) |
| transport | open (`unauthenticated=True`), so other Modal apps can reach the URL |
| application auth | vLLM's `VLLM_API_KEY` bearer: unauthenticated `/v1/models` → 401/403, bearer → 200 |

The bearer lives in a local mode-0600 state file and a Modal Secret, never in
the Volume, the endpoint record, a command line or a manifest. After
`/health` is ready the server parses vLLM's own `Maximum concurrency for
19,200 tokens per request: Xx` line and publishes its endpoint record only
if `X >= 37`.

## Window sizing

The window follows the rule every served window uses
(`sisteer.capeval.serving.geometry`): `L` = the next multiple of 256 at or
above the p99 of prompt + completion, unless the once-only `max_tokens` clamp
would fire on more than 5% of calls at that `L`, in which case p99(prompt) +
the requested `max_tokens`. `scripts/attacker_pool_profile.py` measures it
over the drivers' recorded attacker and scorer calls of one served multi-turn
wave (PAIR, Crescendo, ActorAttack; 22,661 calls re-tokenized with the
attacker's own tokenizer and chat template). The attacker role's figures are
the `MEASURED_*` literals of `attacker_pool.py`, which a test pins to this
table.

| role | calls | p99 prompt + completion | p99 prompt | max prompt + completion | `max_tokens` | `L` | clamp rate at `L` | calls over `L` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| attacker, all attacks | 10,147 | 18,973 | 18,718 | 23,752 | 1,024 | **19,200** | 1.44% | 0.87% |
| scorer, all attacks | 12,514 | 650 | 642 | 721 | 8 | 768 | 0.00% | 0.00% |

One server serves both roles, so the pool's window is the larger, 19,200. The
tail is Crescendo's growing transcript: a pool serving Crescendo alone sizes
to 20,480 (`CRESCENDO_MAX_MODEL_LEN`), and 24,576 (`COVER_MAX_MAX_MODEL_LEN`)
covers every observed call at fewer concurrent sequences. At 19,200 the clamp
fires on 1.44% of attacker calls and almost always leaves the completion
whole (p99 completion 426 tokens); a prompt that no longer fits is refused as
a typed error, never truncated silently.

## Commands

Every invocation names its app (`--name`); stage, endpoint and canary run on
the CPU-only `tools` app and cannot start a second H200 server. The Modal
client is the project's optional `modal` extra (the pinned release):
`uv sync --frozen --extra cpu --extra modal` adds it to the repository's
environment, and any other interpreter that has it works as well.

```bash
MODAL=".venv/bin/python -m modal"   # after the extra sync above; any interpreter with the Modal client works
ATTACKER_POOL_ROTATE_KEY=1 $MODAL run --name attacker-pool-qwen36-27b-stage \
  scripts/modal/attacker_pool.py::tools.main --action stage          # rotate bearer, stage snapshot
$MODAL deploy --name attacker-pool-qwen36-27b scripts/modal/attacker_pool.py::app
$MODAL run --name attacker-pool-qwen36-27b-endpoint \
  scripts/modal/attacker_pool.py::tools.main --action endpoint       # fetch endpoint.json locally
$MODAL run --name attacker-pool-qwen36-27b-canary \
  scripts/modal/attacker_pool.py::tools.main --action canary --n-calls 4
$MODAL app stop attacker-pool-qwen36-27b -y
```

## Endpoint record and consumers

`endpoint.json` carries the model, snapshot, URL, geometry, Modal app /
server / container ids and the engine-attested serving block.
`connect_endpoint` re-verifies the app and server ids, the URL, `/health`,
both `/v1/models` responses and the model's real window before any target
work; a bare URL is refused because it carries no attributable geometry or
liveness. Drivers take the fetched record as
`--attacker-pool <state dir>/attacker-pool-modal/endpoint.json`.
