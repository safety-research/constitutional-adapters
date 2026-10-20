"""Llama-3.1-8B Sobol HPO on Modal (twosi-model-sweeps 3-model comparison).

Methodology matched to the E4B/Qwen rounds: Sobol 16/cell over
log(lr) x log(clip), warmup {0.05, 0.15} as a 2-level factor, seed boxes
psr lr [5e-5, 5e-3] / token_faithful lr [3e-5, 3e-3] / clip [0.03, 3],
--fit-subsample 3000, epochs 2, floor-guarded, fitted-argmin selection,
then a refine round per cell.

Model: NousResearch/Meta-Llama-3.1-8B-Instruct @ d10aef7 (PUBLIC — no
token). Pool: the clean twosi_llama_v3 pairs_si_constitution.jsonl
(10478 pairs, sha 9e8ae189...) on the llama-hpo Volume; its tokcache
sidecar is built by the `precompute` mode before anything trains.
Config: the repo's committed configs/two_si.yaml (the base Llama config,
sha a29ce3b7655276da) — config_sha-stamped trials.

Modes:
  modal run llama_hpo.py --mode precompute   # model snapshot + sidecar
  modal run llama_hpo.py --mode memprobe     # per-arm bs ladder on H200
  modal run llama_hpo.py --mode wave --round-no 1
  (SISTEER_MODAL_WORKTREE / SISTEER_MODAL_STAGE / SISTEER_MODAL_REQUIREMENTS
  name the repository tree, the staging dir and the requirements export the
  image is built from; defaults below)
"""
import os

import modal

app = modal.App("llama-hpo")


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


#: the repository tree the image carries (the sweep worktree), the staging dir (trials) and the requirements export the
#: image installs — the E4B stage's export, which this sweep shares with the E4B rounds rather than carrying its own —
#: the three environment variables, else the sweep worktree of the site's checkout (SISTEER_REPO) and directories under its
#: staging root ($STAGE), resolved where the image is built
WORKTREE = (os.environ.get("SISTEER_MODAL_WORKTREE") or _site_value("SISTEER_REPO/" + os.path.join("worktrees", "twosweep"))
            if modal.is_local() else "/repo")
STAGE = os.environ.get("SISTEER_MODAL_STAGE") or _site_value("$STAGE/llama") if modal.is_local() else "/stage"
REQUIREMENTS = (os.environ.get("SISTEER_MODAL_REQUIREMENTS") or _site_value("$STAGE/e4b/requirements.txt")
                if modal.is_local() else "/stage/requirements.txt")
MODEL_REPO = "NousResearch/Meta-Llama-3.1-8B-Instruct"
MODEL_REV = "d10aef7999a2b5ba950ab3974312feeedbfe0b77"
PAIRS = "/vol/pool/pairs_si_constitution.jsonl"
CFG = "/repo/configs/two_si.yaml"

#: (arm, position) HPO cells — decode-only, token_faithful is all-only.
CELLS = [("psr_kl_all", "response"), ("psr_mse_all", "response"),
         ("lora_r1_all", "response"), ("token_faithful", "all")]

#: Per-arm batch size. Cluster pins (psr 4 / lora 16 / tf 4 @ max_len 8192)
#: are the ladder floor; `memprobe` mode measures the H200 ceiling and this
#: dict is updated from its output before the wave runs.
BS_BY_ARM = {"psr_kl_all": 4, "psr_mse_all": 4,
             "lora_r1_all": 16, "token_faithful": 4}

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("unixodbc")
    .pip_install_from_requirements(
        REQUIREMENTS,
        extra_index_url="https://download.pytorch.org/whl/cu128",
    )
    .add_local_dir(
        WORKTREE, remote_path="/repo",
        ignore=[".git", "**/__pycache__", ".venv", "out/**", "wandb/**",
                ".pytest_cache", "worktrees"],
    )
    .add_local_dir(f"{STAGE}/trials", remote_path="/stage/trials")
)

vol = modal.Volume.from_name("llama-hpo", create_if_missing=True)

_COMMON = dict(image=image, volumes={"/vol": vol}, timeout=3600)


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
    if extra:
        env.update(extra)
    return env


def _ensure_model():
    import os
    from pathlib import Path

    os.environ["HF_HOME"] = "/vol/hf"
    snap = Path(f"/vol/hf/hub/models--{MODEL_REPO.replace('/', '--')}"
                f"/snapshots/{MODEL_REV}")
    if not (snap / "config.json").exists():
        from huggingface_hub import snapshot_download

        snapshot_download(MODEL_REPO, revision=MODEL_REV)  # public
        vol.commit()
    os.environ["HF_HUB_OFFLINE"] = "1"


@app.function(cpu=8, memory=32768, **_COMMON)
def precompute() -> dict:
    """Model snapshot (public, CPU-side) + the pool's tokcache sidecar."""
    import subprocess
    import sys
    import time
    from pathlib import Path

    t0 = time.time()
    _ensure_model()
    dl_s = time.time() - t0

    sidecar = Path(PAIRS + ".tokcache.npz")
    if sidecar.exists():
        return {"sidecar": "already-present", "model_s": round(dl_s, 1)}
    t0 = time.time()
    p = subprocess.run(
        [sys.executable, "/repo/scripts/train_two_si_objects.py",
         "precompute-tokens", "--config", CFG,
         "--si", "si_constitution", "--pairs", PAIRS],
        cwd="/repo", env=_env(), capture_output=True, text=True)
    print(p.stdout[-3000:])
    if p.returncode != 0:
        print("STDERR:", p.stderr[-3000:])
        return {"error": f"precompute rc={p.returncode}",
                "stderr_tail": str(p.stderr[-1500:])}
    vol.commit()
    return {"sidecar": "built", "model_s": round(dl_s, 1),
            "tokenize_s": round(time.time() - t0, 1),
            "sidecar_mb": round(sidecar.stat().st_size / 1e6, 1)}


@app.function(gpu="H200", cpu=8, memory=65536, max_containers=4, **_COMMON)
def memprobe(cell: tuple) -> dict:
    """Largest-fitting bs per /gpu-util: worst-case probe (longest rows),
    doubling ladder from the cluster pin, 0.85x141GB reserve guard."""
    import re
    import subprocess
    import sys

    arm, pos = cell
    _ensure_model()
    guard_gib = 0.85 * 141.0
    ladder, results = [], {}
    b = BS_BY_ARM[arm]
    while b <= 64:
        ladder.append(b)
        b *= 2
    best = None
    for b in ladder:
        p = subprocess.run(
            [sys.executable, "/repo/scripts/tune_arm.py", "probe-mem",
             "--config", CFG, "--pairs", PAIRS,
             "--si", "si_constitution", "--arm", arm, "--positions", pos,
             "--batch-size", str(b), "--steps", "8"],
            cwd="/repo", env=_env(), capture_output=True, text=True)
        tail = p.stdout[-600:]
        m = re.search(r"peak_alloc=([0-9.]+)GiB", p.stdout)
        peak = float(m.group(1)) if m else None
        oom = ("CUDA out of memory" in p.stderr
               or "CUDA out of memory" in p.stdout)
        results[b] = {"rc": p.returncode, "peak_gib": peak, "oom": oom}
        print(f"[{arm}] bs={b} rc={p.returncode} peak={peak} oom={oom} "
              f"| {tail.splitlines()[-1] if tail.splitlines() else ''}")
        if p.returncode == 0 and peak is not None and peak <= guard_gib:
            best = b
        else:
            break
    return {"arm": str(arm), "position": str(pos), "best_bs": best,
            "guard_gib": round(guard_gib, 1),
            "ladder": {str(k): v for k, v in results.items()}}


@app.function(gpu="H200", cpu=8, memory=65536, max_containers=16, **_COMMON)
def run_one(item: tuple) -> dict:
    import json
    import math
    import shutil
    import statistics
    import subprocess
    import sys
    import threading
    import time
    from pathlib import Path

    arm, pos, trials_file, index, round_no, bs = item
    cell = f"{arm}_{pos}"
    tdir = f"r{round_no}_t{index:02d}"
    _ensure_model()

    out_root = Path(f"/root/out/{cell}")
    trial_dir = out_root / tdir
    trial_dir.mkdir(parents=True, exist_ok=True)
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
        time.sleep(30)  # llama trials are long; sample well inside the loop
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
        "--config", CFG,
        "--trials", f"/stage/trials/{trials_file}",
        "--index", str(index),
        "--pairs", PAIRS,
        "--out", str(out_root),
        "--si", "si_constitution",
        "--arm", arm, "--positions", pos,
        "--batch-size", str(bs), "--epochs", "2",
        "--fit-subsample", "3000",
        "--no-wandb",
    ]
    if index == 0:
        cmd.append("--profile")
    t0 = time.time()
    proc = subprocess.run(cmd, cwd="/repo", env=_env(),
                          capture_output=True, text=True)
    wall = time.time() - t0
    done.set()

    skipped = "skipping" in proc.stdout and "MATCHING fit" in proc.stdout
    res: dict = {"arm": str(arm), "position": str(pos), "index": int(index),
                 "round": int(round_no), "bs": int(bs),
                 "rc": int(proc.returncode),
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
            "peak_gib": (round(float(s["peak_alloc_gib"]), 1)
                         if s.get("peak_alloc_gib") else None),
        })
        if proc.returncode != 0 and s.get("status") not in (
                "ok", "nan_diverged", "gate_dead", "floor_failed"):
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
        dest = Path(f"/vol/out/{cell}/{tdir}")
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copytree(trial_dir, dest, dirs_exist_ok=True)
        vol.commit()
    return res


@app.local_entrypoint()
def main(mode: str = "wave", round_no: int = 1):
    import json
    import time

    if mode == "precompute":
        print(json.dumps(precompute.remote(), indent=2))
        return
    if mode == "memprobe":
        for r in memprobe.map([tuple(c) for c in CELLS]):
            print(json.dumps(r, indent=2))
        return

    items = []
    for arm, pos in CELLS:
        items += [(arm, pos, f"r{round_no}_{arm}_{pos}.json", i, round_no,
                   BS_BY_ARM[arm])
                  for i in range(16 if round_no == 1 else 8)]
    t0 = time.time()
    results = list(run_one.map(items))
    wave_s = time.time() - t0
    print(json.dumps(results, indent=2))
    ok = sum(1 for r in results if r.get("status") == "ok")
    bad = [r for r in results if r.get("status") not in ("ok",)
           and not r.get("skipped")]
    duty = [r["mean_sm_pct"] for r in results if "mean_sm_pct" in r]
    print(f"\nWAVE r{round_no}: {ok}/{len(items)} ok, "
          f"{sum(1 for r in results if r.get('skipped'))} skipped, "
          f"{len(bad)} not-ok, wall {wave_s / 60:.1f} min")
    if duty:
        print(f"duty (n={len(duty)}): mean {sum(duty) / len(duty):.1f}%, "
              f"min {min(duty):.1f}%, max {max(duty):.1f}%")
    for r in bad:
        print("NOT-OK:", json.dumps(r))
