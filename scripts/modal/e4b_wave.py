"""E4B HPO recenter wave on Modal (twosi-model-sweeps) — GREEN-LIT scale-up.

All 32 round-2 trials — lora_r1_all/response t00-t15 + token_faithful/all
t00-t15, lr box [1.5e-3, 9e-3] — via .map() at <=16 containers. Same recipe
the duty gate validated (97.4% mean SM): repo @ 169db47 + folded config
(config_sha gate), bs=7, epochs 2, subsample 3000, pre-tokenized sidecar.

Idempotent: each container pre-seeds its trial's summary.json from the
e4b-hpo Volume, so completed trials (incl. the gate's lora t00) self-skip
on the harness's fit_fingerprint. Model is already cached on the Volume.

Run, from the checkout (its src supplies sisteer.site, which reads the site settings):
      modal run scripts/modal/e4b_wave.py
A copy staged outside the checkout runs the same way with SISTEER_REPO in the environment (that checkout's src then
supplies the module; with SISTEER_STAGE_DIR set as well, none is needed), or with the checkout's src on PYTHONPATH.
"""
import os

import modal

app = modal.App("e4b-hpo-wave")


def _site_module():
    """sisteer.site, the one reader of the site config: from the checkout this file is in, else from the checkout
    $SISTEER_REPO names, else as importable (PYTHONPATH); refused by name when none has it."""
    import sys
    from pathlib import Path
    here = Path(__file__).resolve()
    checkouts = [here.parents[2]] if len(here.parents) > 2 else []
    if os.environ.get("SISTEER_REPO"):
        checkouts.append(Path(os.path.expanduser(os.environ["SISTEER_REPO"])))
    for src in (c / "src" for c in checkouts):
        if (src / "sisteer" / "site.py").is_file():
            if str(src) not in sys.path:
                sys.path.insert(0, str(src))
            break
    try:
        from sisteer import site
    except ImportError:
        raise SystemExit("e4b_wave.py: the site config is read through sisteer.site, which is not importable here: run the "
                         "checkout's scripts/modal/e4b_wave.py, set SISTEER_REPO (and SISTEER_STAGE_DIR) in the "
                         "environment, or put the checkout's src on PYTHONPATH") from None
    return site


def _site_value(key: str) -> str:
    """A site setting read where the image is built: the environment, else the site config through sisteer.site (imported
    only then, so a copy given both settings needs no checkout); refused by name when neither gives the key."""
    value = os.environ.get(key, "")
    if not value:
        site = _site_module()
        try:
            value = site.load_site().config.get(key, "")
        except site.SiteConfigError as exc:
            raise SystemExit(f"e4b_wave.py: {exc}") from None
    if not value:
        raise SystemExit(f"e4b_wave.py: set {key} in the environment or the site config")
    return os.path.expanduser(value)


# the recipe's worktree of the grading host's checkout (SISTEER_REPO), shipped to the image as /repo, and the wave's stage
# directory under the site's stage root; both resolved where the image is built
WORKTREE = os.path.join(_site_value("SISTEER_REPO"), "worktrees", "twosweep") if modal.is_local() else "/repo"
STAGE = os.path.join(_site_value("SISTEER_STAGE_DIR"), "e4b") if modal.is_local() else "/stage"
MODEL_REPO = "google/gemma-4-E4B-it"
MODEL_REV = "ee0ef6023621cff504d758262d4e04895a5af4a2"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("unixodbc")
    .pip_install_from_requirements(
        f"{STAGE}/requirements.txt",
        extra_index_url="https://download.pytorch.org/whl/cu128",
    )
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
    max_containers=16,
    image=image,
    volumes={"/vol": vol},
    secrets=[modal.Secret.from_name("hf-token")],
)
def run_one(item: tuple) -> dict:
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

    arm, pos, trials_file, index, round_no = item
    cell = f"{arm}_{pos}"
    tdir = f"r{round_no}_t{index:02d}"

    os.environ["HF_HOME"] = "/vol/hf"
    snap = Path(f"/vol/hf/hub/models--{MODEL_REPO.replace('/', '--')}"
                f"/snapshots/{MODEL_REV}")
    if not (snap / "config.json").exists():
        from huggingface_hub import snapshot_download

        tok = (os.environ.get("HF_TOKEN")
               or os.environ.get("HUGGING_FACE_HUB_TOKEN"))
        snapshot_download(MODEL_REPO, revision=MODEL_REV, token=tok)
        vol.commit()
    os.environ["HF_HUB_OFFLINE"] = "1"

    out_root = Path(f"/root/out/{cell}")
    trial_dir = out_root / tdir
    trial_dir.mkdir(parents=True, exist_ok=True)

    # Idempotence: seed the harness's fingerprint skip from the Volume.
    seeded = Path(f"/vol/out/{cell}/{tdir}/summary.json")
    if seeded.exists():
        shutil.copy2(seeded, trial_dir / "summary.json")

    metrics_path = trial_dir / "metrics.jsonl"
    dmon_rows: list[int] = []
    done = threading.Event()

    def sample_dmon():
        for _ in range(240):
            if done.is_set():
                return
            time.sleep(5)
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
        time.sleep(10)
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

    watcher = threading.Thread(target=sample_dmon, daemon=True)
    watcher.start()

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
        "--trials", f"/stage/trials/{trials_file}",
        "--index", str(index),
        "--pairs", "/stage/pool/pairs_si_constitution.jsonl",
        "--out", str(out_root),
        "--si", "si_constitution",
        "--arm", arm, "--positions", pos,
        "--batch-size", "7", "--epochs", "2", "--fit-subsample", "3000",
        "--no-wandb",
    ]
    t0 = time.time()
    proc = subprocess.run(cmd, cwd="/repo", env=env,
                          capture_output=True, text=True)
    wall = time.time() - t0
    done.set()
    watcher.join(timeout=5)

    skipped = "skipping" in proc.stdout and "MATCHING fit" in proc.stdout
    res: dict = {"arm": str(arm), "position": str(pos), "index": int(index),
                 "rc": int(proc.returncode), "wall_s": round(float(wall), 1),
                 "skipped": bool(skipped)}
    if dmon_rows:
        steady = dmon_rows[1:] if len(dmon_rows) > 5 else dmon_rows
        res["mean_sm_pct"] = round(sum(steady) / len(steady), 1)

    spath = trial_dir / "summary.json"
    if spath.exists():
        s = json.loads(spath.read_text())
        tr = s.get("trial", {})
        res.update({
            "status": str(s.get("status")),
            "lr": float(tr.get("lr", float("nan"))),
            "clip": float(tr.get("max_grad_norm", float("nan"))),
            "warmup": float(tr.get("warmup_frac", float("nan"))),
            "y": (round(float(s["best_heldout_loss"]), 5)
                  if s.get("best_heldout_loss") is not None
                  and math.isfinite(float(s["best_heldout_loss"]))
                  else None),
            "floor_ok": s.get("floor_ok"),
            "floor_ratio": (round(float(s["floor_ratio"]), 4)
                            if s.get("floor_ratio") is not None else None),
        })
    else:
        res["status"] = "no-summary"
        res["stderr_tail"] = str(proc.stderr[-1200:])

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
        dest = Path(f"/vol/out/{cell}/{tdir}")
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copytree(trial_dir, dest, dirs_exist_ok=True)
        vol.commit()
    return res


@app.local_entrypoint()
def main():
    import json
    import time

    # psr_mse r3 recenter, lr [8e-4, 7e-3] bracketing the r1 best 2.333e-3
    # (green-lit; the other three arms are resolved and folded).
    items = [("psr_mse_all", "response", "r3_psr_mse_all_response.json", i, 3)
             for i in range(16)]
    t0 = time.time()
    results = list(run_one.map(items))
    wave_s = time.time() - t0
    print(json.dumps(results, indent=2))
    ok = sum(1 for r in results if r.get("status") == "ok")
    bad = [r for r in results if r.get("status") not in ("ok",)
           and not r.get("skipped")]
    duty = [r["mean_sm_pct"] for r in results if "mean_sm_pct" in r]
    print(f"\nWAVE: {ok}/{len(items)} ok, "
          f"{sum(1 for r in results if r.get('skipped'))} skipped, "
          f"{len(bad)} not-ok, wall {wave_s / 60:.1f} min")
    if duty:
        print(f"duty spot-checks (n={len(duty)}): "
              f"mean {sum(duty) / len(duty):.1f}%, min {min(duty):.1f}%, "
              f"max {max(duty):.1f}%")
    for r in bad:
        print("NOT-OK:", json.dumps(r))
