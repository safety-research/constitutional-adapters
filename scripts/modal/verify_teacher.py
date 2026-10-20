"""Modal runner for scripts/verify_teacher_topk.py — the real-model
verification of the offline top-K teacher + accumulation parity, on
H200s with no queue: no cluster submissions, everything runs on Modal.

Model: the public Llama-3.1-8B-Instruct mirror @ the campaign's pinned
revision (configs/two_si_llama_fits.yaml — the HPO'd base whose losses the
top-50-vs-full question is about). Pool: the local ~/twosi_llama_v3/pairs
v3 files ride into the image (jsonl only; the fat tokcache sidecars stay
home — a 64-pair verification re-tokenizes in seconds).

  ~/venvs/modal/bin/modal run scripts/modal/verify_teacher.py \
      [--sis safety_refusal,si_constitution,si_irrelevant] [--n 64]

One container per SI (they map concurrently); each writes
/vol/out/verify_teacher/verify_<si>.json to the `twosi-fits` volume and
returns the report in-band, so the caller prints every number without a
volume round-trip.
"""
import json
import time
from pathlib import Path

import modal

app = modal.App("twosi-verify-teacher")

try:
    WORKTREE = Path(__file__).resolve().parents[2]
except IndexError:
    WORKTREE = Path("/repo")

MODEL_REPO = "NousResearch/Meta-Llama-3.1-8B-Instruct"
MODEL_REV = "d10aef7999a2b5ba950ab3974312feeedbfe0b77"
SIS = ["safety_refusal", "si_constitution", "si_irrelevant"]

_IGNORE = [".git", "**/__pycache__", ".venv", "out/**", "wandb/**",
           ".pytest_cache", "worktrees"]

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("unixodbc")
    .pip_install_from_requirements(
        str(WORKTREE / "scripts/modal/requirements.txt"),
        extra_index_url="https://download.pytorch.org/whl/cu128",
    )
    .add_local_dir(str(WORKTREE), remote_path="/repo", ignore=_IGNORE)
    .add_local_dir(str(Path.home() / "twosi_llama_v3/pairs"),
                   remote_path="/llama_pool",
                   ignore=["*.npz", "*.tokcache*"])
)

vol = modal.Volume.from_name("twosi-fits", create_if_missing=True)
_COMMON = dict(volumes={"/vol": vol},
               secrets=[modal.Secret.from_name("hf-token")])


def _env():
    import os

    env = dict(os.environ)
    env.update({
        "HF_HOME": "/vol/hf",
        "PYTHONPATH": "/repo/src",
        "TOKENIZERS_PARALLELISM": "false",
        "PYTHONUNBUFFERED": "1",
        "WANDB_DISABLED": "true",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    })
    return env


def _ensure_model() -> None:
    import os

    os.environ["HF_HOME"] = "/vol/hf"
    snap = Path(f"/vol/hf/hub/models--{MODEL_REPO.replace('/', '--')}"
                f"/snapshots/{MODEL_REV}")
    if not (snap / "config.json").exists():
        from huggingface_hub import snapshot_download

        snapshot_download(MODEL_REPO, revision=MODEL_REV)
        vol.commit()
    os.environ["HF_HUB_OFFLINE"] = "1"


@app.function(image=image, cpu=8, memory=32768, timeout=7200, **_COMMON)
def stage() -> dict:
    _ensure_model()
    return {"model": MODEL_REPO, "rev": MODEL_REV, "staged": True}


@app.function(gpu="H200", image=image, cpu=8, memory=65536, timeout=10800,
              max_containers=8, scaledown_window=120, **_COMMON)
def verify(item: tuple) -> dict:
    import subprocess
    import sys

    si, n, fit_n = item[:3]
    skip_fit = bool(item[3]) if len(item) > 3 else False
    _ensure_model()
    out_dir = Path("/vol/out/verify_teacher")
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / (f"verify_{si}_skipfit.json" if skip_fit
                     else f"verify_{si}.json")
    t0 = time.time()
    cmd = [sys.executable, "/repo/scripts/verify_teacher_topk.py",
           "--config", "/repo/configs/two_si_llama_fits.yaml",
           "--pairs", f"/llama_pool/pairs_{si}.jsonl",
           "--si", si, "--n", str(n), "--fit-n", str(fit_n),
           "--batch-size", "2", "--out", str(out)]
    if skip_fit:
        cmd.append("--skip-fit")
    p = subprocess.run(cmd, cwd="/repo", env=_env(), capture_output=True,
                       text=True)
    res: dict = {"si": si, "rc": int(p.returncode),
                 "wall_s": round(time.time() - t0, 1),
                 "log": [ln for ln in p.stdout.splitlines()
                         if ln.startswith("[verify]")]}
    if p.returncode == 0 and out.exists():
        res["report"] = json.loads(out.read_text())
        vol.commit()
    else:
        res["stderr_tail"] = str(p.stderr[-3000:])
    return res


@app.local_entrypoint()
def main(sis: str = "", n: int = 64, fit_n: int = 24,
         skip_fit: bool = False):
    want = [s for s in sis.split(",") if s] or SIS
    print(json.dumps(stage.remote(), indent=1))
    t0 = time.time()
    results = list(verify.map([(si, n, fit_n, skip_fit) for si in want]))
    print(f"\n=== verification wall {(time.time() - t0) / 60:.1f} min ===")
    for r in results:
        print(f"\n--- {r['si']} (rc={r['rc']}, {r['wall_s']}s) ---")
        for ln in r.get("log", []):
            print(ln)
        if r["rc"] != 0:
            print("STDERR TAIL:\n" + r.get("stderr_tail", ""))
    ok = [r for r in results if r["rc"] == 0]
    print(f"\n{len(ok)}/{len(results)} SIs verified; reports on volume "
          "twosi-fits at out/verify_teacher/verify_<si>.json")
