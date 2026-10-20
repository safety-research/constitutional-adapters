"""Single-trial Modal gate for the E4B HPO recenter (twosi-model-sweeps).

Runs ONE real tune_arm trial — lora_r1_all/response, round-2 index 0
(lr box [1.5e-3, 9e-3]) — on a dedicated H200 and measures GPU duty
(mean SM% via dmon -i 0 during the step loop) plus s/step. This is the
"is it ~100%?" gate before scaling to the other 31 trials.

Faithful to the cluster round-1 setup: repo @ 169db47 (with the FOLDED
config — the trials JSON's config_sha gate refuses any other bytes),
lockfile-exported deps (torch 2.11.0+cu128, transformers 5.14.1),
bs=7 (H200 memprobe), epochs 2, fit-subsample 3000, pre-tokenized
sidecar, fitmon's cuDNN-SDPA pin + length bucketing.

Model: google/gemma-4-E4B-it @ ee0ef60 (gated). First run downloads it
into the e4b-hpo Volume via the workspace's existing `hf-token` secret;
later runs hit the Volume offline.

Run:  modal run scripts/modal/e4b_trial.py
      (SISTEER_MODAL_WORKTREE / SISTEER_MODAL_STAGE name the repository tree
      and the staging dir the image is built from; defaults below)
"""
import os

import modal

app = modal.App("e4b-trial-gate")


def _site_value(spec: str) -> str:
    """A default under the grading host's site, resolved where the image is built: ``$STAGE/...`` through sisteer.site, and
    ``SISTEER_REPO/...`` from the site setting of that name (the environment, then the site config), both read from this
    script's checkout; refused by name when the site sets neither."""
    import sys
    from pathlib import Path
    src = str(Path(__file__).resolve().parents[2] / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    from sisteer import site as _site
    key, _, rest = spec.partition("/")
    if key != "SISTEER_REPO":
        return str(_site.expand_path(spec, key=Path(__file__).name))
    repo = os.environ.get(key) or _site.load_site().config.get(key)
    if not repo:
        raise SystemExit(f"{Path(__file__).name}: set {key} (the grading host's checkout) in the environment or the site config")
    return os.path.join(os.path.expanduser(repo), rest)


#: the repository tree the image carries (the sweep worktree at 169db47 with the FOLDED config) and the staging dir
#: (the lockfile export, pool, trials) — the two environment variables, else the sweep worktree of the site's checkout
#: (SISTEER_REPO) and the e4b directory under its staging root ($STAGE), resolved where the image is built
WORKTREE = (os.environ.get("SISTEER_MODAL_WORKTREE") or _site_value("SISTEER_REPO/" + os.path.join("worktrees", "twosweep"))
            if modal.is_local() else "/repo")
STAGE = os.environ.get("SISTEER_MODAL_STAGE") or _site_value("$STAGE/e4b") if modal.is_local() else "/stage"
MODEL_REPO = "google/gemma-4-E4B-it"
MODEL_REV = "ee0ef6023621cff504d758262d4e04895a5af4a2"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("unixodbc")  # pyodbc (pyrit dep) needs libodbc at import
    .pip_install_from_requirements(
        f"{STAGE}/requirements.txt",
        extra_index_url="https://download.pytorch.org/whl/cu128",
    )
    # Repo at 169db47 + the FOLDED config (uncommitted worktree bytes —
    # exactly what the trials JSON's config_sha was stamped against).
    .add_local_dir(
        WORKTREE, remote_path="/repo",
        ignore=[".git", "**/__pycache__", ".venv", "out/**", "wandb/**",
                ".pytest_cache", "worktrees"],
    )
    .add_local_dir(f"{STAGE}/pool", remote_path="/stage/pool")
    .add_local_dir(f"{STAGE}/trials", remote_path="/stage/trials")
)

vol = modal.Volume.from_name("e4b-hpo", create_if_missing=True)


@app.function(
    gpu="H200",
    cpu=8,
    memory=65536,
    timeout=3600,
    image=image,
    volumes={"/vol": vol},
    secrets=[modal.Secret.from_name("hf-token")],
)
def run_trial():
    import json
    import math
    import os
    import shutil
    import statistics
    import subprocess
    import sys
    import threading
    import time
    from pathlib import Path

    t_start = time.time()
    os.environ["HF_HOME"] = "/vol/hf"
    snap = Path(f"/vol/hf/hub/models--{MODEL_REPO.replace('/', '--')}"
                f"/snapshots/{MODEL_REV}")

    # ---- stage the gated model into the Volume (first run only) ----
    dl_s = 0.0
    if not (snap / "config.json").exists():
        from huggingface_hub import snapshot_download

        tok = (os.environ.get("HF_TOKEN")
               or os.environ.get("HUGGING_FACE_HUB_TOKEN")
               or os.environ.get("HUGGINGFACE_TOKEN"))
        t0 = time.time()
        snapshot_download(MODEL_REPO, revision=MODEL_REV, token=tok)
        dl_s = time.time() - t0
        vol.commit()
    os.environ["HF_HUB_OFFLINE"] = "1"

    out_root = Path("/root/out/lora_r1_all_response")
    out_root.mkdir(parents=True, exist_ok=True)
    metrics_path = out_root / "r2_t00" / "metrics.jsonl"

    # ---- dmon sampler: fires once the optimizer loop is live ----
    dmon_rows: list[int] = []
    dmon_note = {"status": "never-fired"}

    def sample_dmon():
        for _ in range(240):  # up to 20 min for model load + encodes
            time.sleep(5)
            try:
                if metrics_path.exists() and \
                        sum(1 for _ in metrics_path.open()) >= 3:
                    break
            except OSError:
                pass
        else:
            return
        time.sleep(10)  # clear of startup, inside the step loop
        dmon_note["status"] = "sampled"
        p = subprocess.run(
            ["nvidia-smi", "dmon", "-i", "0", "-s", "u", "-c", "30"],
            capture_output=True, text=True, timeout=120)
        for line in p.stdout.splitlines():
            parts = line.split()
            if len(parts) >= 2 and parts[0].isdigit():
                try:
                    dmon_rows.append(int(parts[1]))
                except ValueError:
                    pass

    watcher = threading.Thread(target=sample_dmon, daemon=True)
    watcher.start()

    # ---- the real trial, exactly the cluster invocation ----
    env = dict(os.environ)
    env.update({
        "PYTHONPATH": "/repo/src",
        "TOKENIZERS_PARALLELISM": "false",
        "PYTHONUNBUFFERED": "1",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    })
    cmd = [
        sys.executable, "/repo/scripts/tune_arm.py", "trial",
        "--config", "/repo/configs/two_si_gemma4e4b.yaml",
        "--trials", "/stage/trials/r2_lora_r1_all_response.json",
        "--index", "0",
        "--pairs", "/stage/pool/pairs_si_constitution.jsonl",
        "--out", str(out_root),
        "--si", "si_constitution",
        "--arm", "lora_r1_all", "--positions", "response",
        "--batch-size", "7", "--epochs", "2", "--fit-subsample", "3000",
        "--profile", "--no-wandb",
    ]
    t0 = time.time()
    proc = subprocess.run(cmd, cwd="/repo", env=env,
                          capture_output=True, text=True)
    trial_wall = time.time() - t0
    print(proc.stdout[-4000:])
    if proc.returncode != 0:
        print("STDERR tail:\n", proc.stderr[-4000:])
    watcher.join(timeout=60)

    # ---- harvest ----
    result: dict = {
        "trial_rc": int(proc.returncode),
        "trial_wall_s": round(float(trial_wall), 1),
        "model_download_s": round(float(dl_s), 1),
        "dmon_status": str(dmon_note["status"]),
    }
    if dmon_rows:
        steady = dmon_rows[1:] if len(dmon_rows) > 5 else dmon_rows
        result["mean_sm_pct"] = round(sum(steady) / len(steady), 1)
        result["dmon_n"] = len(steady)
        result["min_sm_pct"] = int(min(steady))

    summary_path = out_root / "r2_t00" / "summary.json"
    if summary_path.exists():
        s = json.loads(summary_path.read_text())
        for k in ("status", "best_heldout_loss", "floor_ok", "floor_ratio",
                  "steps_run", "wall_s", "peak_alloc_gib", "n_fit",
                  "total_steps"):
            v = s.get(k)
            if isinstance(v, float):
                v = round(v, 5) if math.isfinite(v) else None
            result[f"trial_{k}"] = v if isinstance(
                v, (int, float, str, bool)) or v is None else str(v)

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
            result["sec_per_step_median"] = round(
                statistics.median(spst), 3)
            result["sec_per_step_n"] = len(spst)

    # persist full artifacts (summary, metrics, profiler trace) to the Volume
    dest = Path("/vol/out/lora_r1_all_response")
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copytree(out_root, dest, dirs_exist_ok=True)
    vol.commit()

    if proc.returncode != 0 and not summary_path.exists():
        result["stderr_tail"] = str(proc.stderr[-1500:])
    result["total_fn_wall_s"] = round(time.time() - t_start, 1)
    return result


@app.local_entrypoint()
def main():
    import json
    import time

    t0 = time.time()
    res = run_trial.remote()
    res["end_to_end_local_s"] = round(time.time() - t0, 1)
    print(json.dumps(res, indent=2))
