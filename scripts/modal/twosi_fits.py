"""twosi-model-sweeps FIT app on Modal — the production steering-defense
objects for Qwen3.8-27B (thinking-off pool) + gemma-4-E4B-it, adapted from
the e4b_wave.py HPO fan-out. Routing: Llama fits are
CLUSTER-only (scripts/sbatch/twosi_fits_llama.sbatch — never Modal); the
Qwen-off and Gemma waves run here, EVERYTHING on H200.

Per model: 12 fits = 3 SIs x {psr_kl_all, psr_mse_all, lora_r1_all at
positions=response; token_faithful at positions=all}. Each fit is ONE
container = ONE cell (the harness's single-process path), FULL clean v3
pool, FIXED HPO-winner LRs from the committed *_fits.yaml configs,
max_len 16384, every fit on a dedicated H200.

Volume `twosi-fits` layout:
  /vol/hf                      HF cache (all three models)
  /vol/pool/<model>/pairs_<si>.jsonl (+ .tokcache.npz sidecar)
  /vol/out/<model>/cells_all|cells_response/<cell_id>/   published cells
  /vol/out/<model>/reports/    per-container fit-report + metrics copies

Modes (run from the worktree so /repo bakes the intended commit):
  ~/venvs/modal/bin/modal run scripts/modal/twosi_fits.py --mode stage \
      --model qwen38_27b            # model snapshot -> volume (once)
  # pool staging when the clean v3 data lands (per model):
  #   ~/venvs/modal/bin/modal volume put twosi-fits \
  #       <local>/pairs_<si>.jsonl pool/<model>/pairs_<si>.jsonl   (x3 SIs)
  #   then --mode precompute --model <model>   # tokcache sidecars, CPU
  ~/venvs/modal/bin/modal run scripts/modal/twosi_fits.py \
      --mode precompute-teacher --model <model>  # offline top-50 teacher
      # logits (GPU, one container/SI) -> pool/.../pairs_<si>.jsonl
      # .teachertopk.safetensors; the fits then skip the online teacher
  ~/venvs/modal/bin/modal run scripts/modal/twosi_fits.py --mode memprobe \
      --model qwen38_27b            # per-arm bs ladder AT 16384, 0.85 guard
  ~/venvs/modal/bin/modal run --detach scripts/modal/twosi_fits.py \
      --mode hpo                    # the re-HPO@32 wave: 3 models x 3 arms
      # x 32 Sobol trials, ALL submitted up front (one H200/trial), trials
      # from /vol/hpo/trials, outputs to /vol/out/hpo/<model>/<cell>/
  ~/venvs/modal/bin/modal run scripts/modal/twosi_fits.py --mode fits \
      --model qwen38_27b [--sis si_constitution] [--arms psr_kl_all]
      [--overwrite]                 # ephemeral .map() launch (legacy —
      # dies with its client; prefer fits-spawn for anything unattended)
  # THE DURABLE LAUNCH (fit-wave hardening, 2026-08-27):
  ~/venvs/modal/bin/modal deploy scripts/modal/twosi_fits.py   # (1) named
      # deployed app; does NOT disturb running ephemeral apps of this file
  ~/venvs/modal/bin/modal run scripts/modal/twosi_fits.py \
      --mode fits-spawn --model <m> [--sis ...] [--arms ...] \
      [--out-root out/canary/<m>] [--checkpoint-every N]      # (2) spawn
      # one server-side FunctionCall per cell on the DEPLOYED app, write
      # $STAGE/fits/fit_calls.json (cell, call id, spawn time),
      # exit — the wave now survives this client dying
  ~/venvs/modal/bin/python scripts/modal/fits_watch.py         # (3) poll
      # the volume for COMPLETE markers; terminal = all manifest cells
      # spawned AND COMPLETE (Monitor-able; exits at the terminal state)

memprobe needs no pool: with /vol/pool/<model>/ absent it synthesizes
worst-case rows (~1024-token prompts, ~4096-token completions — the v3
caps) and probes the LONGEST-rows shape through the real trainer path
(tune_arm.py probe-mem), which also end-to-end-verifies the image (for
Qwen: triton 3.7.1 + flash-linear-attention + causal-conv1d — the
gated-chunk BACKWARD only compiles on triton >= 3.7.1, fla #640).

Resume: each container seeds its cell dir (and _store_identity.json) from
the volume before training, so a COMPLETE cell self-skips on the harness's
fingerprint (pool bytes + config bytes + optimizer settings); re-running a
wave costs nothing for finished cells. --overwrite retrains.

W&B: no `wandb` secret exists in this workspace, so fits run with
WANDB_DISABLED=true and the curves land in each cell's metrics.jsonl (the
fitmon mirror) + the copied <si>__metrics tee — see the _secrets note below
to turn live W&B on.
"""
import json
import time
from pathlib import Path

import modal

app = modal.App("twosi-fits")

try:
    # Local (the grading host worktree): image defs + the entrypoint's config-sha
    # pin read real paths. In the CONTAINER this module re-imports from a
    # path with <3 parents and the lookup IndexErrors — the container never
    # needs the worktree (everything lives at /repo), so fall back.
    WORKTREE = Path(__file__).resolve().parents[2]
except IndexError:
    WORKTREE = Path("/repo")


def _stage_path(rel: str) -> Path:
    """``$STAGE/<rel>`` (the staging directory) on this machine, expanded through sisteer.site from this checkout; an unset
    placeholder refuses by name."""
    import sys
    if str(WORKTREE / "src") not in sys.path:
        sys.path.insert(0, str(WORKTREE / "src"))
    from sisteer import site as _site
    return _site.expand_path(f"$STAGE/{rel}", key="--manifest")

SIS = ["safety_refusal", "si_constitution", "si_irrelevant"]
# lora_r4 = lora_r1_all's post-re-tune name (rank-4 winner); future
# waves emit under it, legacy artifacts keep their name.
RESPONSE_ARMS = ["psr_kl_all", "psr_mse_all", "lora_r4"]
ALL_ARM = "token_faithful"

MODELS = {
    "gemma4e4b": {
        "gpu": "H200",
        "cfg": "/repo/configs/two_si_gemma4e4b_fits.yaml",
        "repo": "google/gemma-4-E4B-it",
        "rev": "ee0ef6023621cff504d758262d4e04895a5af4a2",
        "gated": True,
    },
    "qwen38_27b": {
        "gpu": "H200",
        "cfg": "/repo/configs/two_si_qwen38_27b_fits.yaml",
        "repo": "Qwen/Qwen3.8-27B",
        "rev": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
        "gated": False,
    },
    # Qwen3.5 safety-transfer lane (2026-08-29): the psr_kl_all TRANSFER
    # fits — {si_constitution, si_irrelevant} at the FIXED safety-lane
    # HPO winners (9B r1_t05, 4B r1_t02; folded by make_fit_configs.py's
    # qwen35 lane). Same qwen3_5 hybrid-attention stack as qwen38_27b, so
    # every GPU function must run image_qwen (QWEN_IMAGE_KEYS below).
    # Fitted-object facts (out/qwen35_*/cells_response/safety_refusal__
    # psr_kl_all): response_header_len 7 both; hidden dim 4096 (9B) /
    # 2560 (4B); 32 layers both. Both snapshots already staged on /vol/hf
    # by the safety HPO lanes.
    "qwen35_9b": {
        "gpu": "H200",
        "cfg": "/repo/configs/two_si_qwen35_9b_fits.yaml",
        "repo": "Qwen/Qwen3.5-9B",
        "rev": "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
        "gated": True,
    },
    "qwen35_4b": {
        "gpu": "H200",
        "cfg": "/repo/configs/two_si_qwen35_4b_fits.yaml",
        "repo": "Qwen/Qwen3.5-4B",
        "rev": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
        "gated": True,
    },
    # Llama: teacher precompute runs here now; the FITS route here too
    # under the standing no-cluster rule (2026-08-27 — supersedes the
    # header's cluster-only routing note). Before a Llama fit wave:
    # memprobe on THIS app (batch_size_by_arm in two_si_llama_fits.yaml
    # was measured on the cluster with the ONLINE teacher; the precompute
    # frees that memory, so re-probe), and note the pool + tokcache are
    # staged at pool/llama8b/.
    # E2B: E4B's MatFormer width-slice sibling (same 35-layer depth, same
    # arch class -> image_std). Config regenerated 2026-08-30 via
    # make_fit_configs.py --only gemma4e2b (build_gemma4e2b: the safety
    # psr_kl winner r1_t04 folded FIXED for the {si_constitution,
    # si_irrelevant} TRANSFER fits; lora_r4/psr_mse_all keep fail-loud
    # PENDING sentinels until their HPOs land) — supersedes the
    # hand-derived two_si_gemma4e2b_probe.yaml, which remains the lane's
    # generator SOURCE.
    "gemma4e2b": {
        "gpu": "H200",
        "cfg": "/repo/configs/two_si_gemma4e2b_fits.yaml",
        "repo": "google/gemma-4-E2B-it",
        "rev": "3e22461f65e89153144f8adb70e3b8c2cc9845a7",
        "gated": True,
    },
    "llama8b": {
        "gpu": "H200",
        "cfg": "/repo/configs/two_si_llama_fits.yaml",
        "repo": "NousResearch/Meta-Llama-3.1-8B-Instruct",
        "rev": "d10aef7999a2b5ba950ab3974312feeedbfe0b77",
        "gated": False,
    },
    # GLM-4.6V-Flash (2026-09-02): zai-org, MIT, UNGATED — a dense 10.3B
    # VLM (~9B text decoder, 0.9B ViT) used TEXT-ONLY, full attention on
    # all 40 layers, so it rides image_std: the fused-kernel image exists
    # for the hybrid-attention Qwen3.5/3.8 lines only, and nothing here
    # needs fused kernels or a requirements change. Loads as
    # Glm4vForConditionalGeneration (glm4v has no causal-LM auto-mapping
    # in transformers 5.14.1; models.auto_model_class dispatches).
    # cfg = the DERIVED pre-HPO config ON PURPOSE: stage / precompute /
    # memprobe run against it, and its batch_size_by_arm placeholders make
    # any fit wave refuse loudly (resolve_batch_size int()s them) until
    # the memprobe + HPO land and a make_fit_configs lane emits
    # two_si_glm46vflash_fits.yaml — re-point cfg there at that step.
    # Not in HPO_BS / HPO_SUB yet (no measured width exists; the hpo
    # modes refuse the key until the memprobe table is folded).
    "glm46vflash": {
        "gpu": "H200",
        "cfg": "/repo/configs/two_si_glm46vflash.yaml",
        "repo": "zai-org/GLM-4.6V-Flash",
        "rev": "411bb4d77144a3f03accbf4b780f5acb8b7cde4e",
        "gated": False,
    },
    # Nemotron-3-Nano-4B (2026-09-02; NVIDIA Open Model License — NOT
    # gated). NemotronH hybrid: 42 blocks, 4 attention (12, 17, 24, 32).
    # cfg = the DERIVED config (scripts/derive_model_config.py), not a
    # make_fit_configs.py *_fits.yaml — that lane folds HPO winners, and
    # none exist yet. The derived config carries fail-loud
    # batch_size_by_arm placeholders (memprobe first) and empty per-arm
    # train_kwargs (HPO), so a fit wave before both lands crashes by design;
    # memprobe / teacher precompute read the config's pins and run. Not in
    # HPO_BS / HPO_SUB until the memprobe lands (mode hpo refuses the key).
    # Every GPU function for it binds image_nemotron (NEMOTRON_IMAGE_KEYS):
    # the hybrid-stack image plus the mamba-ssm fused kernels, so the 21
    # Mamba2 blocks run cuda_kernels_forward, not the pure-torch SSD scan.
    "nemotron3nano4b": {
        "gpu": "H200",
        "cfg": "/repo/configs/two_si_nemotron3nano4b.yaml",
        "repo": "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16",
        "rev": "dfaf35de3e30f1867dd8dbc38a7fc9fb52d3914f",
        "gated": False,
    },
}

#: Model keys on the qwen hybrid-attention stack (Qwen3.8 / Qwen3.5 lines):
#: every GPU function for them must bind image_qwen (the fused kernels —
#: triton 3.7.1 + fla 0.5.2 + causal-conv1d), never image_std. Keyed here
#: so no routing site string-matches one literal model again (the
#: qwen35 transfer fits silently took image_std before this).
QWEN_IMAGE_KEYS = ("qwen38_27b", "qwen35_9b", "qwen35_4b")

#: Model keys on the NemotronH Mamba2-hybrid stack: every GPU function for
#: them binds image_nemotron — image_qwen's recipe PLUS the mamba-ssm fused
#: kernels. NemotronH's Mamba2 mixers resolve their kernels per op
#: (transformers 5.14.1: kernels-hub -> pip mamba_ssm / causal_conv1d ->
#: the inline pure-torch SSD scan) and need ALL FIVE
#: (selective_state_update, mamba_chunk_scan_combined,
#: mamba_split_conv1d_scan_combined, causal_conv1d_fn/update) for the
#: fast path; image_qwen carries only the causal-conv1d pair, so on it the
#: model trained the torch_forward SSM — correct, and 2.09 / 1.88 s per
#: bs-2 step (psr_kl_all / lora_r1_all, max_len 2048, cluster H200,
#: 2026-09-02). On image_nemotron the SAME smoke rows run 0.23 / 0.13
#: s/step at 11.4 GiB peak instead of 28.1 (Modal H200, 2026-09-02; the
#: kernel-less image reproduced the cluster numbers on that machine to the
#: millisecond, so the 9x / 14x is the kernels). Speed-giving kernels
#: ship by default, pinned, dependencies notwithstanding — hence this
#: image and this key table. Every routing site goes through _variant();
#: no B200 binding exists for this image (a 4B model fits H200 at every
#: arm), and --b200 on a nemotron key refuses instead of quietly taking
#: H200.
NEMOTRON_IMAGE_KEYS = ("nemotron3nano4b",)

_IGNORE = [".git", "**/__pycache__", ".venv", "out/**", "wandb/**",
           ".pytest_cache", "worktrees"]

#: The standard fit image — the e4b_wave.py recipe (cu128 torch tree from
#: the lockfile-exported requirements).
image_std = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("unixodbc")
    .pip_install_from_requirements(
        str(WORKTREE / "scripts/modal/requirements.txt"),
        extra_index_url="https://download.pytorch.org/whl/cu128",
    )
    .add_local_dir(str(WORKTREE), remote_path="/repo", ignore=_IGNORE)
)

#: The Qwen hybrid-attention image: nvidia/cuda DEVEL base (nvcc 12.8 for
#: the causal-conv1d source build — no wheel exists at torch 2.11), the
#: same requirements tree, then the hybrid stack pins: triton 3.7.1
#: OVERRIDES the lockfile's 3.6.0 (fla's gated-chunk backward miscompiles
#: on [3.4.0, 3.7.1)), flash-linear-attention/fla-core 0.5.2,
#: causal-conv1d 1.6.2.post1 built for sm90 (H100/H200).
image_qwen = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04",
                              add_python="3.12")
    .apt_install("unixodbc", "git", "build-essential")
    .pip_install_from_requirements(
        str(WORKTREE / "scripts/modal/requirements.txt"),
        extra_index_url="https://download.pytorch.org/whl/cu128",
    )
    .pip_install("triton==3.7.1", "fla-core==0.5.2",
                 "flash-linear-attention==0.5.2")
    .run_commands(
        "CUDA_HOME=/usr/local/cuda CC=gcc CXX=g++ "
        "TORCH_CUDA_ARCH_LIST='9.0;10.0' MAX_JOBS=8 "
        "CAUSAL_CONV1D_FORCE_BUILD=TRUE "
        "pip install --no-build-isolation causal-conv1d==1.6.2.post1",
        # The step above re-resolves torch's own triton==3.6.0 pin and
        # silently DOWNGRADES the 3.7.1 we just installed — fla then
        # refuses every gated-chunk backward at runtime (#640; bit the
        # first memprobe, all Hopper). Re-pin LAST with --no-deps, and
        # make the build itself fail if the stack ever regresses.
        "pip install --no-deps triton==3.7.1",
        "python -c \"import triton; v = triton.__version__; "
        "assert v == '3.7.1', v; import causal_conv1d, fla\"",
    )
    .add_local_dir(str(WORKTREE), remote_path="/repo", ignore=_IGNORE)
)

#: The NemotronH (Mamba2-hybrid) image: image_qwen's recipe step for step
#: (same devel base, requirements tree, pins and causal-conv1d source
#: build — every layer through that build is shared with image_qwen's
#: cache; image_qwen itself is untouched, live qwen cells build from it),
#: PLUS the mamba-ssm fused kernels, so transformers' per-op resolution
#: finds all five and NemotronH runs cuda_kernels_forward instead of the
#: pure-torch SSD scan. sdfbase_fits.py carries this stanza VERBATIM.
#:
#: mamba-ssm==2.3.2.post1 — the newest PyPI release >= 7 days old on
#: 2026-09-02 (released 2026-05-09; setup.py pins no torch upper bound).
#: Its GitHub wheel matrix stops at torch 2.10 (+ NGC tags), so at torch
#: 2.11 the package's own bdist hook would 404 and fall through to a source
#: build anyway: MAMBA_FORCE_BUILD=TRUE skips that dead probe and makes the
#: build the explicit, deterministic path. nvcc compiles the selective-scan
#: extension for sm_75/80/87/90/100/120 (setup.py's own list — it does not
#: read TORCH_CUDA_ARCH_LIST); ninja==1.13.0 (2025-08-11) goes in first so
#: the .cu files compile in parallel under MAX_JOBS=8 — without ninja,
#: torch's BuildExtension compiles them serially.
#:
#: --no-deps ON PURPOSE: 2.3.2.post1 added hard deps tilelang==0.1.8,
#: apache-tvm-ffi and quack-kernels (-> nvidia-cutlass-dsl, ~200 MB of an
#: unrelated CUDA DSL toolchain) that serve ONLY its new Mamba3 kernels,
#: which mamba_ssm imports behind try/except ImportError. The Mamba2 fast
#: path NemotronH calls (ops.triton.selective_state_update,
#: ops.triton.ssd_combined + the compiled selective_scan_cuda) needs torch,
#: triton, einops and packaging — all already in the tree. To take the
#: Mamba3 chain anyway, drop --no-deps (the triton re-pin below then
#: matters again, as it does for causal-conv1d).
#:
#: The bake check imports exactly what transformers resolves, so a
#: regressed stack fails the BUILD, not the first fit.
image_nemotron = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04",
                              add_python="3.12")
    .apt_install("unixodbc", "git", "build-essential")
    .pip_install_from_requirements(
        str(WORKTREE / "scripts/modal/requirements.txt"),
        extra_index_url="https://download.pytorch.org/whl/cu128",
    )
    .pip_install("triton==3.7.1", "fla-core==0.5.2",
                 "flash-linear-attention==0.5.2")
    .run_commands(
        "CUDA_HOME=/usr/local/cuda CC=gcc CXX=g++ "
        "TORCH_CUDA_ARCH_LIST='9.0;10.0' MAX_JOBS=8 "
        "CAUSAL_CONV1D_FORCE_BUILD=TRUE "
        "pip install --no-build-isolation causal-conv1d==1.6.2.post1",
        "pip install --no-deps triton==3.7.1",
        "python -c \"import triton; v = triton.__version__; "
        "assert v == '3.7.1', v; import causal_conv1d, fla\"",
    )
    .run_commands(
        "pip install ninja==1.13.0",
        "CUDA_HOME=/usr/local/cuda CC=gcc CXX=g++ MAX_JOBS=8 "
        "MAMBA_FORCE_BUILD=TRUE "
        "pip install --no-build-isolation --no-deps mamba-ssm==2.3.2.post1",
        # Re-pin triton LAST and assert, exactly as the causal-conv1d step
        # does: --no-deps leaves it alone, and the assert is the proof.
        "pip install --no-deps triton==3.7.1",
        "python -c \"import triton; v = triton.__version__; "
        "assert v == '3.7.1', v; import causal_conv1d, fla, mamba_ssm; "
        "from mamba_ssm.ops.triton.selective_state_update import "
        "selective_state_update; "
        "from mamba_ssm.ops.triton.ssd_combined import "
        "mamba_chunk_scan_combined, mamba_split_conv1d_scan_combined; "
        "from causal_conv1d import causal_conv1d_fn, causal_conv1d_update\"",
    )
    .add_local_dir(str(WORKTREE), remote_path="/repo", ignore=_IGNORE)
)

#: Autoscale CEILING for every GPU function here (max-parallelism
#: directive): high enough that Modal's own H200 availability /
#: workspace quota is the binding constraint, never this number — a wave of
#: N inputs runs in ~one turnaround instead of N/ceiling. Cost is neutral
#: (fixed GPU-hours; parallelism only changes wall-clock). Modal spins at
#: most one container per pending input, so an idle high bound costs nothing.
#: 288 = the re-HPO@32 wave (3 models x 3 arms x 32 trials, one
#: trial/container, all submitted up front).
MAX_CONTAINERS = 288

vol = modal.Volume.from_name("twosi-fits", create_if_missing=True)

# No `wandb` secret exists in the workspace (checked 2026-08-27), and
# Secret.from_name is lazy — a missing name aborts at RUN time, not here —
# so W&B stays off on Modal (metrics.jsonl + the fitmon per-cell mirror
# carry the curves). To turn it on: create the secret, append
# modal.Secret.from_name("wandb") here and set _HAVE_WANDB = True.
# Live W&B enabled 2026-08-27: the `wandb` secret must exist in the
# workspace BEFORE the next deploy — Secret.from_name is lazy and a
# missing name fails at run-time hydration, loudly, on every function.
_secrets = [modal.Secret.from_name("hf-token"),
            modal.Secret.from_name("wandb")]
_HAVE_WANDB = True

_COMMON = dict(volumes={"/vol": vol}, secrets=_secrets)


def _env(extra=None):
    import os

    env = dict(os.environ)
    env.update({
        "HF_HOME": "/vol/hf",
        "PYTHONPATH": "/repo/src",
        "TOKENIZERS_PARALLELISM": "false",
        "PYTHONUNBUFFERED": "1",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    })
    if not _HAVE_WANDB:
        env["WANDB_DISABLED"] = "true"
    if extra:
        env.update(extra)
    return env


def _ensure_model(key: str) -> None:
    import os
    from pathlib import Path

    spec = MODELS[key]
    os.environ["HF_HOME"] = "/vol/hf"
    snap = Path(f"/vol/hf/hub/models--{spec['repo'].replace('/', '--')}"
                f"/snapshots/{spec['rev']}")
    if not (snap / "config.json").exists():
        from huggingface_hub import snapshot_download

        tok = (os.environ.get("HF_TOKEN")
               or os.environ.get("HUGGING_FACE_HUB_TOKEN"))
        snapshot_download(spec["repo"], revision=spec["rev"],
                          token=tok if spec["gated"] else None)
        vol.commit()
    os.environ["HF_HUB_OFFLINE"] = "1"


def _pool_path(key: str, si: str) -> str:
    return f"/vol/pool/{key}/pairs_{si}.jsonl"


def _synth_worst_case_pairs(key: str, cfg_path: str, n: int,
                            out_path: str) -> None:
    """Worst-case v3-shaped rows for the memprobe when the real pool is
    not yet staged: ~1024-token attack prompts (the allowance),
    ~4096-token completions (the v3 generation cap). Invented text — the
    memory shape is token counts, and the SI (the long teacher side) is
    resolved for real by the probe."""
    import yaml

    cfg = yaml.safe_load(open(cfg_path))
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(cfg["model"],
                                        revision=cfg.get("model_revision"))
    words = ("the model should consider the request carefully and respond "
             "in accordance with all applicable guidelines ").split()

    def filler(n_tok: int) -> str:
        out, step = [], len(words)
        while len(tok(" ".join(out),
                      add_special_tokens=False)["input_ids"]) < n_tok:
            out.extend(words * max(1, (n_tok // (2 * step))))
        return " ".join(out)

    prompt, completion = filler(1024), filler(4096)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for _ in range(n):
            f.write(json.dumps({"prompt": prompt, "pos": completion,
                                "neg": completion,
                                "instruction_id": "si_constitution"}) + "\n")


def _memprobe_impl(item: tuple) -> dict:
    """ONE (si, arm, pos) cell's doubling ladder: the LARGEST bs whose
    peak RESERVED stays under 0.85x(this device's VRAM), at that SI's own
    resolved max_len (safety_refusal buckets at 8192; the long SIs at
    16384). Uses the real pool when staged, else synthetic worst-case
    rows. One container per cell so cells ladder concurrently (each
    ladder step reloads the model in a fresh probe process — serial cells
    would pay 27B loads back to back)."""
    import os
    import re
    import subprocess
    import sys

    import torch

    key, si, arm, pos = item
    spec = MODELS[key]
    _ensure_model(key)
    vram_gib = torch.cuda.get_device_properties(0).total_memory / 2**30
    guard_gib = 0.85 * vram_gib
    pairs = _pool_path(key, si)
    synthetic = not os.path.exists(pairs)
    if synthetic:
        pairs = f"/root/synth/{key}_worstcase.jsonl"
        _synth_worst_case_pairs(key, spec["cfg"], n=64, out_path=pairs)

    out = {"model": key, "si": si, "arm": arm, "position": pos,
           "gpu": torch.cuda.get_device_name(0),
           "vram_gib": round(vram_gib, 1), "guard_gib": round(guard_gib, 1),
           "max_len": 16384, "synthetic_pool": synthetic}
    if True:
        b, best, ladder = 1, None, {}
        while b <= 16:
            p = subprocess.run(
                [sys.executable, "/repo/scripts/tune_arm.py", "probe-mem",
                 "--config", spec["cfg"], "--pairs", pairs,
                 "--si", si, "--arm", arm,
                 "--positions", pos, "--batch-size", str(b),
                 "--steps", "4", "--allow-tokenize"],
                cwd="/repo", env=_env(), capture_output=True, text=True)
            m = re.search(r"peak_alloc=([0-9.]+)GiB peak_reserved="
                          r"([0-9.]+)GiB", p.stdout)
            alloc, reserved = ((float(m.group(1)), float(m.group(2)))
                               if m else (None, None))
            oom = ("CUDA out of memory" in p.stderr
                   or "CUDA out of memory" in p.stdout)
            ladder[b] = {"rc": p.returncode, "peak_alloc_gib": alloc,
                         "peak_reserved_gib": reserved, "oom": oom}
            line = (p.stdout.splitlines()[-1] if p.stdout.splitlines()
                    else "")
            print(f"[{key}/{si}/{arm}/{pos}] bs={b} rc={p.returncode} "
                  f"alloc={alloc} reserved={reserved} oom={oom} | {line}")
            if p.returncode != 0 and not oom:
                ladder[b]["stderr_tail"] = str(p.stderr[-1500:])
                print(f"[{key}/{si}/{arm}/{pos}] bs={b} STDERR tail:\n"
                      f"{p.stderr[-1500:]}")
            if (p.returncode == 0 and reserved is not None
                    and reserved <= guard_gib):
                best = b
                b *= 2
            else:
                break
        out.update({"best_bs": best, "ladder": ladder})
    Path(f"/vol/out/{key}").mkdir(parents=True, exist_ok=True)
    Path(f"/vol/out/{key}/memprobe_{si}_{arm}_{pos}.json").write_text(
        json.dumps(out, indent=1))
    vol.commit()
    return out


def _fit_impl(item: tuple) -> dict:
    """ONE fit cell: (model_key, si, arm, positions, expect_sha,
    overwrite[, out_root, checkpoint_every]). Seeds resume from the
    volume, trains through the production driver, publishes the cell back
    to the volume.

    The two optional tail fields (fit-wave durability, 2026-08-27; absent
    = the historical 6-tuple behaviour, so in-flight callers are
    untouched):

    - ``out_root``: the volume-relative output root (default
      ``out/<model>``). The canary uses a SCRATCH root
      (``out/canary/<model>``) so nothing it writes can touch the
      production cells tree.
    - ``checkpoint_every``: optimizer steps between the driver's mid-fit
      checkpoints (0/absent = the driver default, 100).

    Checkpoints live at ``/vol/<out_root>/ckpt/<cell_id>/`` — DIRECTLY on
    the volume, under a transient path OUTSIDE the published cells tree,
    so the atomic-publish audit and the COMPLETE protocol never see
    checkpoint files and a fresh container after a kill sees the last
    committed checkpoint with no seeding step. A watcher thread
    ``vol.commit()``s after each checkpoint write (the driver subprocess
    cannot commit; only this function's client can); the cell's
    checkpoint dir is DELETED once the cell publishes COMPLETE."""
    import os
    import shutil
    import statistics
    import subprocess
    import sys
    import threading
    from pathlib import Path

    key, si, arm, positions, expect_sha, overwrite = item[:6]
    out_root = str(item[6]) if len(item) > 6 and item[6] else f"out/{key}"
    ckpt_every = int(item[7]) if len(item) > 7 and item[7] else 0
    spec = MODELS[key]
    _ensure_model(key)

    pairs = _pool_path(key, si)
    if not os.path.exists(pairs):
        return {"model": key, "si": si, "arm": arm, "positions": positions,
                "status": "no-pool",
                "error": f"{pairs} not staged (TrainTest v3 data gated)"}

    leg = "response" if positions == "response" else "all"
    cell_id = f"{si}__{arm}"
    local_out = Path(f"/root/out/{key}/cells_{leg}")
    vol_out = Path(f"/vol/{out_root}/cells_{leg}")
    local_out.mkdir(parents=True, exist_ok=True)
    # Seed the fingerprint-validated resume (COMPLETE marker + sidecars).
    for rel in ("_store_identity.json",):
        if (vol_out / rel).exists():
            shutil.copy2(vol_out / rel, local_out / rel)
    if (vol_out / cell_id).exists() and not overwrite:
        shutil.copytree(vol_out / cell_id, local_out / cell_id,
                        dirs_exist_ok=True)

    # GPU-duty spot-check once the optimizer loop is live (gpu-util skill).
    dmon_rows: list[int] = []
    done = threading.Event()
    metrics_glob = local_out / cell_id / "metrics.jsonl"

    def sample_dmon():
        for _ in range(360):
            if done.is_set():
                return
            done.wait(5)
            try:
                if metrics_glob.exists() and \
                        sum(1 for _ in metrics_glob.open()) >= 3:
                    break
            except OSError:
                pass
        else:
            return
        if done.is_set():
            return
        done.wait(30)
        p = subprocess.run(
            ["nvidia-smi", "dmon", "-i", "0", "-s", "u", "-c", "25"],
            capture_output=True, text=True, timeout=90)
        for line in p.stdout.splitlines():
            parts = line.split()
            if len(parts) >= 2 and parts[0].isdigit():
                try:
                    dmon_rows.append(int(parts[1]))
                except ValueError:
                    pass

    threading.Thread(target=sample_dmon, daemon=True).start()

    # Mid-fit checkpoints: written by the driver straight onto the volume
    # (transient path, never inside a cell dir); this thread commits the
    # volume after each new write so a killed container's successor can
    # resume from the last committed checkpoint. Poll-based because the
    # writer is a subprocess — os.replace is atomic on the volume mount,
    # so a poll can never commit a half-written file under the LATEST
    # name.
    ckpt_cell = Path(f"/vol/{out_root}/ckpt/{cell_id}")

    def commit_ckpts():
        last = 0.0
        while not done.is_set():
            done.wait(30)
            try:
                p = ckpt_cell / "ckpt.pt"
                m = p.stat().st_mtime if p.exists() else 0.0
            except OSError:
                continue
            if m > last:
                last = m
                try:
                    vol.commit()
                    print(f"[fitckpt] volume commit after checkpoint "
                          f"write ({cell_id})", flush=True)
                except Exception as e:  # noqa: BLE001 — commit retries next tick
                    print(f"[fitckpt] vol.commit failed ({e}); will retry",
                          flush=True)

    threading.Thread(target=commit_ckpts, daemon=True).start()

    cmd = [
        sys.executable, "/repo/scripts/train_two_si_objects.py", "train",
        "--config", spec["cfg"], "--si", si, "--arms", arm,
        "--pairs", pairs, "--token-cache", "auto",
        "--out", str(local_out),
        "--expect-config-sha", expect_sha,
        "--checkpoint-dir", f"/vol/{out_root}/ckpt",
    ]
    if ckpt_every:
        cmd += ["--checkpoint-every", str(ckpt_every)]
    if positions == "response":
        cmd += ["--positions", "response"]
    if overwrite:
        cmd.append("--overwrite")
    t0 = time.time()
    proc = subprocess.run(cmd, cwd="/repo", env=_env(),
                          capture_output=True, text=True)
    wall = time.time() - t0
    done.set()

    res: dict = {"model": key, "si": si, "arm": arm,
                 "positions": positions, "rc": int(proc.returncode),
                 "wall_s": round(float(wall), 1)}
    skipped = "skipping (resume)" in proc.stdout or \
        "validated COMPLETE" in proc.stdout
    res["skipped"] = bool(skipped and proc.returncode == 0)
    # Checkpoint-seam evidence (resume line + step index, write cadence)
    # into the container log AND the result, so a kill/resume canary can
    # be verified from either.
    ck_lines = [ln for ln in proc.stdout.splitlines() if "[fitckpt]" in ln]
    if ck_lines:
        res["fitckpt_lines"] = ck_lines[:30]
        print("\n".join(ck_lines[:30]), flush=True)
    if dmon_rows:
        steady = dmon_rows[1:] if len(dmon_rows) > 5 else dmon_rows
        res["mean_sm_pct"] = round(sum(steady) / len(steady), 1)
    report = local_out / f"{si}__fit_report.json"
    if report.exists():
        rep = json.loads(report.read_text()).get("arms", {}).get(arm, {})
        res["status"] = rep.get("status", "no-arm-record")
        if rep.get("error"):
            res["error"] = f"{rep.get('error_type')}: {rep['error']}"[:800]
    else:
        res["status"] = "no-report"
        res["stderr_tail"] = str(proc.stderr[-1500:])
    if metrics_glob.exists():
        spst = []
        for line in metrics_glob.open():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("step", 0) >= 5 and "train/sec_per_step" in rec:
                spst.append(float(rec["train/sec_per_step"]))
        if spst:
            res["sec_per_step_median"] = round(statistics.median(spst), 3)

    if not res["skipped"]:
        # Publish: the cell dir (object + COMPLETE + sidecars + per-cell
        # metrics) verbatim; report/metrics tees under per-cell names so
        # concurrent same-SI containers cannot clobber each other's copy.
        vol_out.mkdir(parents=True, exist_ok=True)
        if (local_out / cell_id).exists():
            shutil.copytree(local_out / cell_id, vol_out / cell_id,
                            dirs_exist_ok=True)
        for rel in ("_store_identity.json",):
            if (local_out / rel).exists() and not (vol_out / rel).exists():
                shutil.copy2(local_out / rel, vol_out / rel)
        rdir = Path(f"/vol/{out_root}/reports")
        rdir.mkdir(parents=True, exist_ok=True)
        if report.exists():
            shutil.copy2(report, rdir / f"report_{cell_id}_{leg}.json")
        tee = local_out / f"{si}__metrics.jsonl"
        if tee.exists():
            shutil.copy2(tee, rdir / f"metrics_{cell_id}_{leg}.jsonl")
        vol.commit()
    # A cell that is COMPLETE on the volume no longer needs its transient
    # checkpoints — delete them (a skipped-because-COMPLETE cell's stale
    # dir included) so the ckpt tree only ever holds in-flight cells.
    if (vol_out / cell_id / "COMPLETE").exists() and ckpt_cell.exists():
        shutil.rmtree(ckpt_cell, ignore_errors=True)
        vol.commit()
        res["checkpoints_cleaned"] = True
        print(f"[fitckpt] {cell_id} published COMPLETE — checkpoint dir "
              "deleted", flush=True)
    return res


# --- re-HPO@32: 3 arms x 3 models x 32 Sobol trials ------------

#: The tuning SI — the teacher-topk artifact for it is staged
#: next to each model's pool on the volume.
HPO_SI = "si_constitution"

#: Trial designs staged at /vol/hpo/trials by `modal volume put` (emitted
#: by tune_arm.py emit-trials from the per-model FIT configs, so the
#: config_sha the trials stamp matches /repo's config bytes).
HPO_TRIALS_DIR = "/vol/hpo/trials"

#: Per-(model, arm) trial MICROBATCH — OFFLINE-teacher memprobe widths
#: (--mode hpomem, worst-case rows, 0.85x141 guard; artifacts at
#: /vol/out/hpo/memprobe/). Measured 2026-08-27: llama caps at bs2 on
#: ALL arms even with the teacher evicted (bs4 OOMs; 16384 activations
#: dominate — the eviction bought ~8-10 GiB, not a doubling), so the
#: committed online-era widths happen to survive for llama. gemma/qwen
#: entries are bumped per-arm as their ladders land + the relaunch
#: decision (duty regime) allows.
HPO_BS = {
    "llama8b": {"psr_kl_all": 2, "psr_mse_all": 2, "lora_r1_all": 2,
                "lora_r4": 2},
    # gemma probed to bs2 on all arms (110.1/95.8/110.8 GiB reserved,
    # bs4 OOM); adopted when the gemma cells restarted from scratch (the
    # wave-1 app died with its reaped client), so no in-flight work was
    # discarded for the bump.
    "gemma4e4b": {"psr_kl_all": 2, "psr_mse_all": 2, "lora_r1_all": 2,
                  "lora_r4": 2},
    # qwen psr arms have no headroom at bs1 (106-119 GiB); lora stays 1
    # until its ladder lands.
    "qwen38_27b": {"psr_kl_all": 1, "psr_mse_all": 1, "lora_r1_all": 1,
                   "lora_r4": 1},
    # E2B safety lane: psr_kl width from the safety_refusal memprobe
    # (2026-08-27: bs2 = 104.4 GiB reserved under the 118.8 guard, bs4
    # OOM — probed with the ONLINE teacher resident, so conservative for
    # the offline-teacher sweep). accum = ceil(32/2) = 16, effective 32.
    "gemma4e2b": {"psr_kl_all": 2},
}

#: Per-model fit-subsample (the trial budgets tune_arm_pod.sh
#: pins: 6000 E4B / 3000 Qwen; Llama gets Qwen's — same 3000-row budget
#: the original two-model plan sized for a 27B, comfortable for an 8B).
#: gemma4e2b: FULL pool (0) — the safety pool is 8274 pairs and E2B is
#: small, so the scoped sweep trains on exactly what the fit would.
HPO_SUB = {"llama8b": 3000, "gemma4e4b": 6000, "qwen38_27b": 3000,
           "gemma4e2b": 0}


def _hpo_impl(item: tuple) -> dict:
    """ONE re-tune trial: (model_key, arm, index). The production
    tune_arm.py trial path at the re-tune@32 regime — positions=response,
    effective batch 32 via accumulation, offline top-K teacher
    (--teacher-topk auto; psr_mse has no teacher-logit leg and says so),
    epochs<=15 with the harness's patience-1 early stop. Resumes off a
    published summary (fingerprint-gated skip inside tune_arm)."""
    import shutil
    import statistics
    import subprocess
    import sys
    import threading
    from pathlib import Path

    # (key, arm, index) is round 1 (the original wave's shape); a 4th
    # element is the round number (the pinned follow-up designs); a 5th
    # is the tuning SI (default plan-§6 si_constitution — the E2B safety
    # lane passes safety_refusal).
    key, arm, index = item[:3]
    rnd = int(item[3]) if len(item) > 3 else 1
    si = str(item[4]) if len(item) > 4 else HPO_SI
    spec = MODELS[key]
    _ensure_model(key)

    cell = f"{arm}_response"
    tdir = f"r{rnd}_t{index:02d}"
    trials_file = f"{HPO_TRIALS_DIR}/r{rnd}_{key}_{arm}_response.json"
    pairs = _pool_path(key, si)

    out_root = Path(f"/root/out/hpo/{key}/{cell}")
    trial_dir = out_root / tdir
    trial_dir.mkdir(parents=True, exist_ok=True)
    seeded = Path(f"/vol/out/hpo/{key}/{cell}/{tdir}/summary.json")
    if seeded.exists():
        shutil.copy2(seeded, trial_dir / "summary.json")

    # GPU-duty spot-check once the optimizer loop is live (gpu-util skill).
    metrics_path = trial_dir / "metrics.jsonl"
    dmon_rows: list[int] = []
    done = threading.Event()

    def sample_dmon():
        for _ in range(360):
            if done.is_set():
                return
            done.wait(5)
            try:
                if metrics_path.exists() and \
                        sum(1 for _ in metrics_path.open()) >= 3:
                    break
            except OSError:
                pass
        else:
            return
        if done.is_set():
            return
        done.wait(30)
        p = subprocess.run(
            ["nvidia-smi", "dmon", "-i", "0", "-s", "u", "-c", "25"],
            capture_output=True, text=True, timeout=90)
        for line in p.stdout.splitlines():
            parts = line.split()
            if len(parts) >= 2 and parts[0].isdigit():
                try:
                    dmon_rows.append(int(parts[1]))
                except ValueError:
                    pass

    threading.Thread(target=sample_dmon, daemon=True).start()

    cmd = [
        sys.executable, "/repo/scripts/tune_arm.py", "trial",
        "--config", spec["cfg"],
        "--trials", trials_file,
        "--index", str(index),
        "--pairs", pairs,
        "--out", str(out_root),
        "--si", si,
        "--arm", arm, "--positions", "response",
        "--batch-size", str(HPO_BS[key][arm]),
        "--epochs", "15",
        "--fit-subsample", str(HPO_SUB[key]),
        "--teacher-topk", "auto",
    ]
    if not _HAVE_WANDB:
        cmd.append("--no-wandb")
    if index == 0:
        cmd.append("--profile")  # one profiler trace per cell
    t0 = time.time()
    proc = subprocess.run(cmd, cwd="/repo", env=_env(),
                          capture_output=True, text=True)
    wall = time.time() - t0
    done.set()

    skipped = "skipping" in proc.stdout and "MATCHING fit" in proc.stdout
    res: dict = {"model": key, "arm": arm, "index": int(index),
                 "round": rnd, "si": si,
                 "bs": HPO_BS[key][arm], "rc": int(proc.returncode),
                 "wall_s": round(float(wall), 1), "skipped": bool(skipped)}
    if dmon_rows:
        steady = dmon_rows[1:] if len(dmon_rows) > 5 else dmon_rows
        res["mean_sm_pct"] = round(sum(steady) / len(steady), 1)

    spath = trial_dir / "summary.json"
    if spath.exists():
        s = json.loads(spath.read_text())
        tr = s.get("trial", {})
        res.update({
            "status": str(s.get("status")),
            "heldout_objective": s.get("heldout_objective"),
            "y": (round(float(s["best_heldout_loss"]), 6)
                  if s.get("best_heldout_loss") is not None else None),
            "lr": tr.get("lr"), "clip": tr.get("max_grad_norm"),
            "warmup": tr.get("warmup_frac"),
            "temperature": tr.get("temperature"),
            "weight_decay": tr.get("weight_decay"),
            "lora_rank": tr.get("lora_rank"),
            "accum_steps": s.get("accum_steps"),
            "floor_ok": s.get("floor_ok"),
            "floor_ratio": (round(float(s["floor_ratio"]), 4)
                            if s.get("floor_ratio") is not None else None),
            "peak_gib": (round(float(s["peak_alloc_gib"]), 1)
                         if s.get("peak_alloc_gib") else None),
        })
        if proc.returncode != 0 and s.get("status") not in (
                "ok", "nan_diverged", "gate_dead", "floor_failed",
                "ratio_diverged"):
            res["stderr_tail"] = str(proc.stderr[-1200:])
    else:
        res["status"] = "no-summary"
        res["stderr_tail"] = str(proc.stderr[-1200:])
        print(proc.stdout[-2500:])

    if metrics_path.exists():
        spst = []
        for line in metrics_path.open():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("step", 0) >= 5 and "train/sec_per_step" in rec:
                spst.append(float(rec["train/sec_per_step"]))
        if spst:
            res["sec_per_step_median"] = round(statistics.median(spst), 3)

    if not skipped:
        dest = Path(f"/vol/out/hpo/{key}/{cell}/{tdir}")
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copytree(trial_dir, dest, dirs_exist_ok=True)
        vol.commit()
    return res


def _hpomem_impl(item: tuple) -> dict:
    """OFFLINE-teacher memprobe + step timing for one (model, si, arm):
    the committed batch tables were measured with the ONLINE teacher
    resident, which the precompute evicted — so the real ceiling (and the
    s/step at each width) must be re-measured in the world the trials
    actually run in (--teacher-topk auto; psr_mse keeps its activation
    teacher and says so). Walks an explicit bs ladder, records peak
    alloc/reserved + s/step (per OPTIMIZER step at effective 32, so
    directly comparable across widths), stops at the first failure or
    guard breach."""
    import re
    import subprocess
    import sys

    import torch

    key, si, arm, bs_ladder = item
    spec = MODELS[key]
    _ensure_model(key)
    vram_gib = torch.cuda.get_device_properties(0).total_memory / 2**30
    guard_gib = 0.85 * vram_gib
    pairs = _pool_path(key, si)
    out = {"model": key, "si": si, "arm": arm,
           "gpu": torch.cuda.get_device_name(0),
           "vram_gib": round(vram_gib, 1), "guard_gib": round(guard_gib, 1),
           "ladder": {}}
    best = None
    for b in list(bs_ladder):
        p = subprocess.run(
            [sys.executable, "/repo/scripts/tune_arm.py", "probe-mem",
             "--config", spec["cfg"], "--pairs", pairs, "--si", si,
             "--arm", arm, "--positions", "response",
             "--batch-size", str(int(b)), "--steps", "6",
             "--teacher-topk", "auto", "--allow-tokenize"],
            cwd="/repo", env=_env(), capture_output=True, text=True)
        m = re.search(r"peak_alloc=([0-9.]+)GiB peak_reserved="
                      r"([0-9.]+)GiB wall=([0-9]+)s \(([0-9.]+)s/step\)",
                      p.stdout)
        alloc, reserved, s_step = ((float(m.group(1)), float(m.group(2)),
                                    float(m.group(4))) if m
                                   else (None, None, None))
        oom = ("CUDA out of memory" in p.stderr
               or "CUDA out of memory" in p.stdout)
        row = {"rc": p.returncode, "peak_alloc_gib": alloc,
               "peak_reserved_gib": reserved, "s_per_opt_step": s_step,
               "oom": oom}
        if p.returncode != 0 and not oom:
            row["stderr_tail"] = str(p.stderr[-1200:])
        out["ladder"][str(int(b))] = row
        line = p.stdout.splitlines()[-1] if p.stdout.splitlines() else ""
        print(f"[{key}/{si}/{arm}] bs={b} rc={p.returncode} alloc={alloc} "
              f"reserved={reserved} s/step={s_step} oom={oom} | {line}")
        if p.returncode == 0 and reserved is not None \
                and reserved <= guard_gib:
            best = int(b)
        else:
            break
    out["best_bs"] = best
    Path(f"/vol/out/hpo/memprobe").mkdir(parents=True, exist_ok=True)
    Path(f"/vol/out/hpo/memprobe/{key}_{si}_{arm}.json").write_text(
        json.dumps(out, indent=1))
    vol.commit()
    return out


def _precompute_teacher_impl(item: tuple) -> dict:
    """ONE (model, SI)'s offline top-K teacher build (GPU): the SI-prompted
    teacher forward runs ONCE over the staged pool (bs=1, no_grad, bf16)
    and the top-50 decode-only logits land as the pool's
    `.teachertopk.safetensors` sidecar on the volume — after which the
    fits' `--teacher-topk auto` consumes it and never runs a training-step
    teacher forward again. One container per SI so the three SIs build
    concurrently."""
    import os
    import subprocess
    import sys

    key, si, top_k = item[:3]
    shard_i, shard_n = (int(item[3]), int(item[4])) if len(item) > 3 \
        else (-1, 0)
    spec = MODELS[key]
    _ensure_model(key)
    pairs = _pool_path(key, si)
    if not os.path.exists(pairs):
        return {"model": key, "si": si, "status": "no-pool",
                "error": f"{pairs} not staged"}
    if shard_n:
        art = (f"{pairs}.teachertopk.shard{shard_i:03d}"
               f"of{shard_n:03d}.safetensors")
    else:
        art = pairs + ".teachertopk.safetensors"
    if os.path.exists(art) and os.path.exists(art + ".json"):
        return {"model": key, "si": si, "shard": shard_i if shard_n else None,
                "status": "artifact-present", "path": art}
    cmd = [sys.executable, "/repo/scripts/train_two_si_objects.py",
           "precompute-teacher", "--config", spec["cfg"], "--si", si,
           "--pairs", pairs, "--top-k", str(top_k)]
    if shard_n:
        cmd += ["--shard-index", str(shard_i),
                "--shard-count", str(shard_n)]
    t0 = time.time()
    p = subprocess.run(cmd, cwd="/repo", env=_env(), capture_output=True,
                       text=True)
    out = {"model": key, "si": si,
           "shard": shard_i if shard_n else None,
           "rc": int(p.returncode),
           "wall_s": round(time.time() - t0, 1), "path": art,
           "tail": p.stdout.splitlines()[-3:] if p.stdout else []}
    if p.returncode != 0:
        out["status"] = "failed"
        out["stderr_tail"] = str(p.stderr[-1500:])
    else:
        out["status"] = "built"
        vol.commit()
    return out


def _merge_teacher_impl(item: tuple) -> dict:
    """Merge one SI's shard artifacts into the final sidecar (CPU, no
    GPU, no model load) and drop the shards. Byte-identical to an
    unsharded build — the merge validates every shard's full-pool
    identity and contiguous coverage first."""
    import os
    import subprocess
    import sys

    key, si, shard_n = item
    spec = MODELS[key]
    pairs = _pool_path(key, si)
    art = pairs + ".teachertopk.safetensors"
    if os.path.exists(art) and os.path.exists(art + ".json"):
        return {"model": key, "si": si, "status": "artifact-present",
                "path": art}
    t0 = time.time()
    p = subprocess.run(
        [sys.executable, "/repo/scripts/train_two_si_objects.py",
         "merge-teacher", "--config", spec["cfg"], "--pairs", pairs,
         "--shard-count", str(shard_n)],
        cwd="/repo", env=_env(), capture_output=True, text=True)
    out = {"model": key, "si": si, "rc": int(p.returncode),
           "wall_s": round(time.time() - t0, 1), "path": art,
           "tail": p.stdout.splitlines()[-3:] if p.stdout else []}
    if p.returncode != 0:
        out["status"] = "failed"
        out["stderr_tail"] = str(p.stderr[-1500:])
    else:
        out["status"] = "merged"
        vol.commit()
    return out


def _precompute_impl(key: str) -> dict:
    """Build each staged pool file's tokcache sidecar (CPU, tokenizer
    only) so no fit container pays the encode phase."""
    import os
    import subprocess
    import sys

    spec = MODELS[key]
    _ensure_model(key)
    out = {}
    for si in SIS:
        pairs = _pool_path(key, si)
        if not os.path.exists(pairs):
            out[si] = "not-staged"
            continue
        if os.path.exists(pairs + ".tokcache.npz"):
            out[si] = "sidecar-present"
            continue
        p = subprocess.run(
            [sys.executable, "/repo/scripts/train_two_si_objects.py",
             "precompute-tokens", "--config", spec["cfg"], "--si", si,
             "--pairs", pairs],
            cwd="/repo", env=_env(), capture_output=True, text=True)
        out[si] = ("built" if p.returncode == 0
                   else f"rc={p.returncode}: {p.stderr[-500:]}")
    vol.commit()
    return out


# --- per-(image, gpu) function bindings ------------------------------------
# Modal functions bind image/gpu statically, so each model family gets its
# own thin wrapper around the shared implementations above.

@app.function(image=image_std, cpu=8, memory=32768, timeout=14400, **_COMMON)
def stage_std(key: str) -> dict:
    _ensure_model(key)
    return {"model": key, "staged": True}


@app.function(image=image_std, cpu=8, memory=65536, timeout=14400, **_COMMON)
def precompute_std(key: str) -> dict:
    return _precompute_impl(key)


@app.function(gpu="H200", image=image_std, cpu=8, memory=65536,
              timeout=10800, max_containers=MAX_CONTAINERS, scaledown_window=120,
              **_COMMON)
def memprobe_h200_std(item: tuple) -> dict:
    return _memprobe_impl(item)


@app.function(gpu="H200", image=image_std, cpu=8, memory=65536,
              timeout=86400, max_containers=MAX_CONTAINERS, scaledown_window=120,
              **_COMMON)
def teacher_h200_std(item: tuple) -> dict:
    return _precompute_teacher_impl(item)


@app.function(gpu="H200", image=image_qwen, cpu=8, memory=131072,
              timeout=86400, max_containers=MAX_CONTAINERS, scaledown_window=120,
              **_COMMON)
def teacher_h200_qwen(item: tuple) -> dict:
    return _precompute_teacher_impl(item)


@app.function(gpu="H200", image=image_nemotron, cpu=8, memory=131072,
              timeout=86400, max_containers=MAX_CONTAINERS, scaledown_window=120,
              **_COMMON)
def teacher_h200_nemotron(item: tuple) -> dict:
    return _precompute_teacher_impl(item)


@app.function(image=image_std, cpu=16, memory=98304, timeout=14400,
              max_containers=MAX_CONTAINERS, **_COMMON)
def teacher_merge(item: tuple) -> dict:
    return _merge_teacher_impl(item)


@app.function(gpu="H200", image=image_qwen, cpu=8, memory=131072,
              timeout=10800, max_containers=MAX_CONTAINERS, scaledown_window=120,
              **_COMMON)
def memprobe_h200_qwen(item: tuple) -> dict:
    return _memprobe_impl(item)


@app.function(gpu="H200", image=image_nemotron, cpu=8, memory=131072,
              timeout=10800, max_containers=MAX_CONTAINERS, scaledown_window=120,
              **_COMMON)
def memprobe_h200_nemotron(item: tuple) -> dict:
    return _memprobe_impl(item)


# Qwen at 16384 OOMs every arm at bs1 on H200-141GB (memprobe 2026-08-27:
# psr_kl plain-torch OOM, the rest OOM inside fla's autotune benchmark —
# real pool, worst-case rows, triton 3.7.1 verified). B200-192GB is the
# escalation path; the image compiles causal-conv1d for sm90+sm100.
@app.function(gpu="B200", image=image_qwen, cpu=8, memory=131072,
              timeout=10800, max_containers=MAX_CONTAINERS, **_COMMON)
def memprobe_b200_qwen(item: tuple) -> dict:
    return _memprobe_impl(item)


@app.function(gpu="H200", image=image_std, cpu=8, memory=65536,
              timeout=86400, max_containers=MAX_CONTAINERS, scaledown_window=120,
              **_COMMON)
def fit_h200_std(item: tuple) -> dict:
    return _fit_impl(item)


@app.function(gpu="H200", image=image_std, cpu=8, memory=65536,
              timeout=86400, max_containers=MAX_CONTAINERS, scaledown_window=120,
              **_COMMON)
def hpo_h200_std(item: tuple) -> dict:
    return _hpo_impl(item)


@app.function(gpu="H200", image=image_std, cpu=8, memory=65536,
              timeout=10800, max_containers=MAX_CONTAINERS, scaledown_window=120,
              **_COMMON)
def hpomem_h200_std(item: tuple) -> dict:
    return _hpomem_impl(item)


@app.function(gpu="H200", image=image_qwen, cpu=8, memory=131072,
              timeout=10800, max_containers=MAX_CONTAINERS, scaledown_window=120,
              **_COMMON)
def hpomem_h200_qwen(item: tuple) -> dict:
    return _hpomem_impl(item)


@app.function(gpu="H200", image=image_nemotron, cpu=8, memory=131072,
              timeout=10800, max_containers=MAX_CONTAINERS, scaledown_window=120,
              **_COMMON)
def hpomem_h200_nemotron(item: tuple) -> dict:
    return _hpomem_impl(item)


@app.function(gpu="H200", image=image_qwen, cpu=8, memory=131072,
              timeout=86400, max_containers=MAX_CONTAINERS, scaledown_window=120,
              **_COMMON)
def hpo_h200_qwen(item: tuple) -> dict:
    return _hpo_impl(item)


@app.function(gpu="H200", image=image_nemotron, cpu=8, memory=131072,
              timeout=86400, max_containers=MAX_CONTAINERS, scaledown_window=120,
              **_COMMON)
def hpo_h200_nemotron(item: tuple) -> dict:
    return _hpo_impl(item)


@app.function(gpu="H200", image=image_qwen, cpu=8, memory=131072,
              timeout=86400, max_containers=MAX_CONTAINERS, scaledown_window=120,
              **_COMMON)
def fit_h200_qwen(item: tuple) -> dict:
    return _fit_impl(item)


@app.function(gpu="H200", image=image_nemotron, cpu=8, memory=131072,
              timeout=86400, max_containers=MAX_CONTAINERS, scaledown_window=120,
              **_COMMON)
def fit_h200_nemotron(item: tuple) -> dict:
    return _fit_impl(item)


@app.function(gpu="B200", image=image_qwen, cpu=8, memory=131072,
              timeout=86400, max_containers=MAX_CONTAINERS, scaledown_window=120,
              **_COMMON)
def fit_b200_qwen(item: tuple) -> dict:
    return _fit_impl(item)


def _variant(key: str) -> str:
    """Which image variant serves this model key — the ONE spelling for
    function-object picks and deployed-function names alike (the
    sdfbase_fits.py convention)."""
    if key in NEMOTRON_IMAGE_KEYS:
        return "nemotron"
    return "qwen" if key in QWEN_IMAGE_KEYS else "std"


def _refuse_b200(key: str, b200: bool) -> None:
    """Only the qwen image has B200 bindings (the 27B-at-16384 escalation);
    a nemotron key asked for B200 refuses rather than silently landing on
    H200 (std keys keep their historical behavior: --b200 ignored)."""
    if b200 and _variant(key) == "nemotron":
        raise SystemExit(f"--b200 has no binding on the nemotron image "
                         f"({key}); the 4B model fits H200 at every arm — "
                         "add fit_b200_nemotron / memprobe_b200_nemotron if "
                         "that changes")


def _fit_fname(key: str, b200: bool = False) -> str:
    _refuse_b200(key, b200)
    v = _variant(key)
    return "fit_b200_qwen" if (b200 and v == "qwen") else f"fit_h200_{v}"


def _fit_fn(key: str, b200: bool = False):
    return {"fit_h200_std": fit_h200_std, "fit_h200_qwen": fit_h200_qwen,
            "fit_b200_qwen": fit_b200_qwen,
            "fit_h200_nemotron": fit_h200_nemotron}[_fit_fname(key, b200)]


def _memprobe_fn(key: str, b200: bool = False):
    _refuse_b200(key, b200)
    v = _variant(key)
    if v == "qwen":
        return memprobe_b200_qwen if b200 else memprobe_h200_qwen
    return {"std": memprobe_h200_std,
            "nemotron": memprobe_h200_nemotron}[v]


def _teacher_fn(key: str):
    return {"std": teacher_h200_std, "qwen": teacher_h200_qwen,
            "nemotron": teacher_h200_nemotron}[_variant(key)]


#: The hpo bindings by variant — the `hpo` mode's per-image queues and the
#: `hpo-spawn` deployed-name lookup both iterate this, so a new variant
#: cannot be routed to one and forgotten by the other.
_HPO_FNS = {"std": hpo_h200_std, "qwen": hpo_h200_qwen,
            "nemotron": hpo_h200_nemotron}


@app.local_entrypoint()
def main(mode: str = "memprobe", model: str = "", sis: str = "",
         arms: str = "", overwrite: bool = False, b200: bool = False,
         shards: int = 0, out_root: str = "", checkpoint_every: int = 0,
         manifest: str = "", round_no: int = 1):
    import hashlib

    if mode == "hpo":
        # The re-HPO@32 wave: every trial of every cell
        # submitted up front, one dedicated H200 each, both images'
        # queues running CONCURRENTLY (a serial .map() pair would halve
        # the realized width). --model restricts to one model's cells.
        import threading

        keys = [m for m in model.split(",") if m] or list(HPO_BS)
        for k in keys:
            if k not in HPO_BS:
                raise SystemExit(f"--model must be one of {sorted(HPO_BS)}")
        arms = [a for a in arms.split(",") if a] or RESPONSE_ARMS
        n_trials = 32
        items_by_variant = {
            v: [(k, a, i) for k in keys if _variant(k) == v
                for a in arms for i in range(n_trials)]
            for v in _HPO_FNS}
        total = sum(len(it) for it in items_by_variant.values())
        print(f"[hpo] {total} trials ("
              + " + ".join(f"{len(it)} {v}"
                           for v, it in items_by_variant.items())
              + f"), max_containers={MAX_CONTAINERS}, "
              f"bs={HPO_BS} sub={HPO_SUB} epochs=15 eff-batch 32 "
              f"teacher-topk=auto si={HPO_SI}")
        results: list = []
        lock = threading.Lock()

        def harvest(fn, items):
            for r in fn.map(items, order_outputs=False):
                with lock:
                    results.append(r)
                    n = len(results)
                print(f"[hpo] [{n}/{total}] {r.get('model')}/{r.get('arm')}"
                      f"/t{r.get('index'):02d} status={r.get('status')} "
                      f"y={r.get('y')} wall={r.get('wall_s')}s")

        t0 = time.time()
        threads = [threading.Thread(target=harvest,
                                    args=(_HPO_FNS[v], items_by_variant[v]))
                   for v in _HPO_FNS if items_by_variant[v]]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        wall = (time.time() - t0) / 60
        print(json.dumps(results, indent=1))
        ok = sum(1 for r in results if r.get("status") == "ok")
        bad = [r for r in results if r.get("status") not in
               ("ok", "nan_diverged", "gate_dead", "floor_failed",
                "ratio_diverged")
               and not r.get("skipped")]
        duty = [r["mean_sm_pct"] for r in results if "mean_sm_pct" in r]
        print(f"\nHPO WAVE: {ok}/{total} ok, "
              f"{sum(1 for r in results if r.get('skipped'))} skipped, "
              f"{len(bad)} infra-bad, wall {wall:.1f} min")
        by_cell: dict = {}
        for r in results:
            c = f"{r.get('model')}/{r.get('arm')}"
            by_cell.setdefault(c, []).append(r)
        for c in sorted(by_cell):
            rs = by_cell[c]
            n_ok = sum(1 for r in rs if r.get("status") == "ok")
            print(f"  {c}: {n_ok}/{len(rs)} ok")
        if duty:
            print(f"duty (n={len(duty)}): mean {sum(duty) / len(duty):.1f}%, "
                  f"min {min(duty):.1f}%, max {max(duty):.1f}%")
        for r in bad:
            print("NOT-OK:", json.dumps(r))
        return

    if mode == "hpomem":
        # Offline-teacher re-memprobe + s/step ladder (one container per
        # cell): llama/gemma from their committed widths upward; qwen
        # lora_r1_all bs1-vs-2 timing + the si_irrelevant bs2 gap. The
        # qwen psr arms are already memory-bound at bs1 (123-131 GiB) and
        # are not probed.
        items_std = [("llama8b", HPO_SI, a, [2, 4, 8, 16])
                     for a in RESPONSE_ARMS] \
            + [("gemma4e4b", HPO_SI, a, [1, 2, 4, 8, 16])
               for a in RESPONSE_ARMS]
        items_qwen = [("qwen38_27b", HPO_SI, "lora_r1_all", [1, 2]),
                      ("qwen38_27b", "si_irrelevant", "lora_r1_all", [2])]
        import threading

        results: list = []
        lock = threading.Lock()

        def harvest(fn, items):
            for r in fn.map(items, order_outputs=False):
                with lock:
                    results.append(r)
                print(f"[hpomem] {r['model']}/{r['si']}/{r['arm']}: "
                      f"best_bs={r.get('best_bs')} "
                      + json.dumps(r.get("ladder", {})))

        threads = [threading.Thread(target=harvest, args=(fn, it))
                   for fn, it in ((hpomem_h200_std, items_std),
                                  (hpomem_h200_qwen, items_qwen)) if it]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        print(json.dumps(results, indent=1))
        return

    if mode != "hpo-spawn" and model not in MODELS:
        raise SystemExit(f"--model must be one of {sorted(MODELS)}")
    if mode == "stage":
        print(json.dumps(stage_std.remote(model), indent=1))
        return
    if mode == "precompute":
        print(json.dumps(precompute_std.remote(model), indent=1))
        return
    if mode == "precompute-teacher":
        # The offline top-K teacher build: artifacts to the pool sidecars
        # on the volume; the fits pick them up via --teacher-topk auto with
        # no invocation change. --shards N fans each SI's bs=1 build across
        # N H200 containers (contiguous example ranges, full-pool keying)
        # and a CPU merge per SI writes the final artifact — byte-identical
        # to an unsharded build (tests/test_teacher_topk.py pins the merge)
        # at 1/N the wall.
        want_sis = [s for s in sis.split(",") if s] or SIS
        fn = _teacher_fn(model)
        if shards > 0:
            items = [(model, si, 200, i, shards)
                     for si in want_sis for i in range(shards)]
            results = list(fn.map(items))
            print(json.dumps(results, indent=1))
            bad = [r for r in results
                   if r.get("status") not in ("built", "artifact-present")]
            print(f"\nTEACHER SHARDS {model}: "
                  f"{len(results) - len(bad)}/{len(results)} ok")
            if bad:
                for r in bad:
                    print("NOT-OK:", json.dumps(r))
                raise SystemExit(
                    "shard build(s) failed — NOT merging a partial set")
            results = list(teacher_merge.map(
                [(model, si, shards) for si in want_sis]))
        else:
            results = list(fn.map([(model, si, 200) for si in want_sis]))
        print(json.dumps(results, indent=1))
        bad = [r for r in results
               if r.get("status") not in ("built", "merged",
                                          "artifact-present")]
        print(f"\nTEACHER PRECOMPUTE {model}: "
              f"{len(results) - len(bad)}/{len(results)} ok")
        for r in bad:
            print("NOT-OK:", json.dumps(r))
        return
    if mode == "memprobe":
        probe_sis = [x for x in sis.split(",") if x] or ["si_constitution"]
        probe_arms = [a for a in arms.split(",") if a] or (RESPONSE_ARMS
                                                           + [ALL_ARM])
        cells = [(model, si, a, "all" if a == ALL_ARM else "response")
                 for si in probe_sis for a in probe_arms]
        results = list(_memprobe_fn(model, b200).map(cells))
        print(json.dumps(results, indent=1))
        print("\nBS TABLE (per arm = MIN over probed SIs; fold into "
              "make_fit_configs.py):")
        table: dict = {}
        for r in results:
            b = r.get("best_bs") or 0
            a = r["arm"]
            table[a] = min(table.get(a, 99), b)
        print(json.dumps({a: (v or None) for a, v in table.items()},
                         indent=1))
        return
    if mode == "hpo-spawn":
        # The DURABLE hpo launch (the fits-spawn pattern, one FunctionCall
        # per TRIAL): `modal deploy scripts/modal/twosi_fits.py` first,
        # then this mode spawns each (model, arm, index, round) against
        # the DEPLOYED hpo functions and exits — call lifetime is
        # server-side, so no client reap can kill the wave (the wave-1/
        # wave-3 incident class). Manifest pre-seeded with call_id=null
        # rows and flushed after every spawn; trials keep whole-trial
        # resume (fingerprint self-skip), so re-running this mode tops up
        # exactly the unfinished set. Per-trial kill switch:
        # modal.FunctionCall.from_id(<id>).cancel().
        import getpass
        from datetime import datetime, timezone

        keys = [m for m in model.split(",") if m] or list(HPO_BS)
        for k in keys:
            if k not in HPO_BS:
                raise SystemExit(f"--model must be one of {sorted(HPO_BS)}")
        want_arms = [a for a in arms.split(",") if a] or RESPONSE_ARMS
        spawn_si = ([x for x in sis.split(",") if x] or [HPO_SI])[0]
        rnd = int(round_no)
        n_trials = {}
        for k in keys:
            for a in want_arms:
                tf = _stage_path(f"rehpo/trials/"
                                 f"r{rnd}_{k}_{a}_response.json")
                if not tf.exists():
                    raise SystemExit(f"[hpo-spawn] no local design {tf} — "
                                     "emit + volume-put it first")
                n_trials[(k, a)] = len(json.loads(
                    tf.read_text())["trials"])
        fns = {v: modal.Function.from_name("twosi-fits", f"hpo_h200_{v}")
               for v in _HPO_FNS}
        mpath = Path(manifest).expanduser() if manifest else (
            _stage_path(f"rehpo/hpo_calls_r{rnd}.json"))
        mpath.parent.mkdir(parents=True, exist_ok=True)
        doc = {"app": "twosi-fits", "round": rnd,
               "launcher": getpass.getuser(), "written_at": None,
               "trials": []}

        def _flush():
            doc["written_at"] = datetime.now(timezone.utc).isoformat(
                timespec="seconds")
            tmp = mpath.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(doc, indent=1) + "\n")
            tmp.replace(mpath)

        # Trials already published for this round self-skip in-container;
        # skip the container start entirely when the trial dir exists.
        done = set()
        for k in keys:
            for a in want_arms:
                try:
                    for e in vol.listdir(f"out/hpo/{k}/{a}_response"):
                        name = e.path.rsplit("/", 1)[-1]
                        if name.startswith(f"r{rnd}_t"):
                            done.add((k, a, int(name.split("_t")[-1])))
                except Exception:  # noqa: BLE001 — cell tree absent
                    pass
        rows = [{"model": k, "arm": a, "index": i, "round": rnd,
                 "si": spawn_si,
                 "call_id": None, "spawn_time": None,
                 "status": ("already-published" if (k, a, i) in done
                            else "pending-spawn")}
                for k in keys for a in want_arms
                for i in range(n_trials[(k, a)])]
        doc["trials"] = rows
        _flush()
        n = 0
        for r in rows:
            if r["status"] == "already-published":
                continue
            fn = fns[_variant(r["model"])]
            call = fn.spawn((r["model"], r["arm"], r["index"], rnd,
                             spawn_si))
            r["call_id"] = call.object_id
            r["spawn_time"] = datetime.now(timezone.utc).isoformat(
                timespec="seconds")
            r["status"] = "spawned"
            n += 1
            _flush()
        print(f"[hpo-spawn] {n} trial(s) spawned on the deployed app "
              f"({len(rows) - n} already published); manifest: {mpath}")
        return

    if mode == "fits-spawn":
        # THE DURABLE LAUNCH PATH (fit-wave hardening, 2026-08-27 — the
        # fix for the reaped-client incident: an EPHEMERAL app's .map()
        # dies with its local client, and ~$1.9k of in-flight GPU time
        # died with one). Sequence:
        #
        #   1. `modal deploy scripts/modal/twosi_fits.py` — a NAMED
        #      deployed app ("twosi-fits"). Deploying does not disturb
        #      running ephemeral apps of the same file (verified live
        #      2026-08-27: the still-running HPO app kept its ~156 tasks
        #      across the deploy, as did two other live ephemeral apps of
        #      this file; the deployment is its own app id).
        #   2. THIS mode: `modal run scripts/modal/twosi_fits.py --mode
        #      fits-spawn --model <m> [...]` — .spawn()s one FunctionCall
        #      per fit cell against the DEPLOYED function (call lifetime
        #      is server-side, independent of this client), writes the
        #      manifest ($STAGE/fits/fit_calls.json: cell, call id,
        #      spawn time — rewritten after EVERY spawn, and pre-seeded
        #      with call_id=null rows, so a launcher killed mid-spawn
        #      leaves a manifest naming exactly the unspawned cells),
        #      and EXITS.
        #   3. Harvest: scripts/modal/fits_watch.py polls the volume for
        #      COMPLETE markers; terminal = every manifest cell has a
        #      call id AND a validated COMPLETE marker (an unspawned cell
        #      never completes — the null call_id rows are the re-spawn
        #      list). Run it under a Monitor or setsid nohup.
        #
        # Cells already COMPLETE on the volume are recorded as such and
        # not re-spawned (the driver would self-skip anyway; this saves
        # the container start) unless --overwrite.
        #
        # Lifecycle facts, measured on the live canary (2026-08-27):
        # `modal container stop` on a spawned call's container does NOT
        # end the call — Modal RETRIES the input on a fresh container
        # (the canary's replacement started 11s after the stop and
        # resumed from the volume checkpoint with no manual action), so
        # worker loss self-heals. To genuinely STOP a cell, cancel the
        # CALL: modal.FunctionCall.from_id(<id>).cancel() — the per-cell
        # kill switch (the manifest carries every call id). Progress
        # holds at the last committed checkpoint either way; a later
        # spawn of the same cell resumes it.
        import getpass
        from datetime import datetime, timezone

        cfg_local = WORKTREE / Path(MODELS[model]["cfg"]).relative_to(
            "/repo")
        expect_sha = hashlib.sha256(
            cfg_local.read_bytes()).hexdigest()[:16]
        root = out_root or f"out/{model}"
        want_sis = [s for s in sis.split(",") if s] or SIS
        want_arms = [a for a in arms.split(",") if a] or (RESPONSE_ARMS
                                                          + [ALL_ARM])
        fname = _fit_fname(model, b200)
        fn = modal.Function.from_name("twosi-fits", fname)
        mpath = Path(manifest).expanduser() if manifest else (
            _stage_path("fits/fit_calls.json"))
        mpath.parent.mkdir(parents=True, exist_ok=True)
        doc = {"app": "twosi-fits", "function": fname, "model": model,
               "out_root": root, "config_sha": expect_sha,
               "checkpoint_every": int(checkpoint_every),
               "overwrite": bool(overwrite),
               "launcher": getpass.getuser(),
               "written_at": None, "cells": []}
        if mpath.exists():
            try:
                old = json.loads(mpath.read_text())
                # Keep rows for cells OUTSIDE this request (another
                # model/leg/root), so one manifest can carry a whole
                # campaign; rows for the requested cells are replaced.
                requested = {(model, root, f"{si}__{a}")
                             for si in want_sis for a in want_arms}
                doc["cells"] = [
                    c for c in old.get("cells", [])
                    if (c.get("model"), c.get("out_root"),
                        c.get("cell_id")) not in requested]
            except ValueError:
                print(f"[fits-spawn] {mpath} unparseable — rewriting it")

        def _flush():
            doc["written_at"] = datetime.now(timezone.utc).isoformat(
                timespec="seconds")
            tmp = mpath.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(doc, indent=1) + "\n")
            tmp.replace(mpath)

        # COMPLETE markers already on the volume (existence check only —
        # the driver's fingerprint validation is the real gate; this just
        # avoids paying a container start for a finished cell).
        done_cells = set()
        if not overwrite:
            for leg in ("response", "all"):
                try:
                    for e in vol.listdir(f"{root}/cells_{leg}"):
                        name = e.path.rsplit("/", 1)[-1]
                        try:
                            vol.listdir(f"{root}/cells_{leg}/{name}/"
                                        "COMPLETE")
                            done_cells.add((leg, name))
                        except Exception:  # noqa: BLE001 — no marker
                            pass
                except Exception:  # noqa: BLE001 — tree not created yet
                    pass

        rows = []
        for si in want_sis:
            for arm in want_arms:
                pos = "all" if arm == ALL_ARM else "response"
                leg = "response" if pos == "response" else "all"
                rows.append({
                    "model": model, "si": si, "arm": arm,
                    "positions": pos, "leg": leg,
                    "cell_id": f"{si}__{arm}", "out_root": root,
                    "call_id": None, "spawn_time": None,
                    "status": ("already-complete"
                               if (leg, f"{si}__{arm}") in done_cells
                               else "pending-spawn"),
                })
        doc["cells"].extend(rows)
        _flush()  # the full expected set, BEFORE any spawn (step 2 above)
        n_spawned = 0
        for c in rows:
            if c["status"] == "already-complete":
                continue
            item = (model, c["si"], c["arm"], c["positions"], expect_sha,
                    overwrite, root, int(checkpoint_every))
            call = fn.spawn(item)
            c["call_id"] = call.object_id
            c["spawn_time"] = datetime.now(timezone.utc).isoformat(
                timespec="seconds")
            c["status"] = "spawned"
            n_spawned += 1
            _flush()  # after EVERY spawn — a killed launcher loses nothing
            print(f"[fits-spawn] {c['cell_id']} ({c['positions']}) -> "
                  f"{call.object_id}")
        print(f"[fits-spawn] {n_spawned} cell(s) spawned on deployed "
              f"'{fname}' ({len(rows) - n_spawned} already complete); "
              f"manifest: {mpath}")
        print("[fits-spawn] harvest: ~/venvs/modal/bin/python "
              f"scripts/modal/fits_watch.py --manifest {mpath}")
        return

    if mode != "fits":
        raise SystemExit(f"unknown --mode {mode!r}")

    # THE LAUNCH PATH — gated on the clean v3 data being staged.
    cfg_local = WORKTREE / Path(MODELS[model]["cfg"]).relative_to("/repo")
    expect_sha = hashlib.sha256(cfg_local.read_bytes()).hexdigest()[:16]
    want_sis = [s for s in sis.split(",") if s] or SIS
    want_arms = [a for a in arms.split(",") if a] or (RESPONSE_ARMS
                                                      + [ALL_ARM])
    items = []
    for si in want_sis:
        for arm in want_arms:
            pos = "all" if arm == ALL_ARM else "response"
            items.append((model, si, arm, pos, expect_sha, overwrite))
    print(f"[fits] {model}: {len(items)} cell(s), gpu="
          f"{MODELS[model]['gpu']}, config sha {expect_sha}")
    t0 = time.time()
    results = list(_fit_fn(model, b200).map(items))
    wall = (time.time() - t0) / 60
    print(json.dumps(results, indent=1))
    ok = sum(1 for r in results
             if r.get("status") in ("trained", "skipped_existing")
             or r.get("skipped"))
    bad = [r for r in results
           if r.get("status") not in ("trained", "skipped_existing")
           and not r.get("skipped")]
    duty = [r["mean_sm_pct"] for r in results if "mean_sm_pct" in r]
    print(f"\nFIT WAVE {model}: {ok}/{len(items)} ok, {len(bad)} not-ok, "
          f"wall {wall:.1f} min")
    if duty:
        print(f"duty (n={len(duty)}): mean {sum(duty) / len(duty):.1f}%, "
              f"min {min(duty):.1f}%, max {max(duty):.1f}%")
    for r in bad:
        print("NOT-OK:", json.dumps(r))
