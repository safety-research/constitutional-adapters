"""gemma-4-E2B lora_r4 safety-lane HPO on Modal — 16-trial Sobol
log(lr) x log(clip) x rank{1,4,16} (twosi-model-sweeps; the qwen35 lora
lanes' recipe at the E2B safety-lane regime, round 1).

Arm lora_r4 ONLY, SI safety_refusal ONLY, positions=response. T=1 /
wd=0.01 pinned on every trial row (emit-trials --pin-axis), warmup fixed
0.15, 10-epoch warmup_cosine, patience-3 early stop, offline top-K=200
teacher (--teacher-topk auto — the psr lane's artifact REUSED, never
rebuilt), EFFECTIVE BATCH 32 — tune_arm's DISTILL_EFFECTIVE_BATCH
default, matching the E2B psr_kl lane and the prior gemma/8B lora
sweeps (NOT the qwen lanes' 64), so the script is invoked directly, no
monkeypatch wrapper; accum_steps = ceil(32 / bs) rides the fingerprint
and the summary as usual. NO E2B lora prior anchors this lane, so the
Sobol box is gemma-scale WIDE: lr [3e-5, 3e-3] x clip [0.02, 3.0],
log-space (emit-trials --box-from — brackets the E4B lora winner
lr 9.11e-5 / clip 0.301 with margin), rank snapped to {1,4,16}.

Config: configs/two_si_gemma4e2b_hpo.yaml (INTERIM, hand-derived from
the E2B probe config the psr_kl lane ran under; NO
gradient_checkpointing on the lora arm — E2B is small enough at
safety's max_len 8192, and the hpomem ladder is the check). Image:
image_std (E2B is Gemma4ForConditionalGeneration — STANDARD attention,
never the qwen fused-kernel image). Volume: `twosi-fits`, model key
gemma4e2b — pool/gemma4e2b/pairs_safety_refusal.jsonl + the pool lane's
pre-built .tokcache.npz sidecar and the psr lane's teachertopk sidecar
(all REUSED), trials at hpo/trials/, outputs at
out/hpo/gemma4e2b/lora_r4_response/r1_tNN/. E2B is GATED: the hf-token
secret supplies HF_TOKEN to every function.

Order (GPU-safe; pool/tokcache/teacher already staged by the psr lane):
  ~/venvs/modal/bin/modal run scripts/modal/gemma4e2b_lora_hpo.py --mode stage
      # verify pool + tokcache + teacher + trials on the volume
  ~/venvs/modal/bin/modal run scripts/modal/gemma4e2b_lora_hpo.py --mode hpomem
      # offline-teacher bs ladder (2..32) at safety's max_len 8192,
      # probed at --lora-rank 16 (the grid's worst case) — bs is NOT
      # reused from the psr arm; re-probed for the lora adapters
  ~/venvs/modal/bin/modal deploy scripts/modal/gemma4e2b_lora_hpo.py
  ~/venvs/modal/bin/modal run scripts/modal/gemma4e2b_lora_hpo.py --mode spawn \
      --bs <best power-of-2>        # ALL 16 trials t00..t15, durable
      # FunctionCalls on the DEPLOYED app (the fits-spawn durability
      # lesson: .map dies with its client); manifest
      # $STAGE/gemma4e2b_lora_calls.json
  ~/venvs/modal/bin/modal run scripts/modal/gemma4e2b_lora_hpo.py --mode status
      # summaries published so far (client-side volume poll)

Per-trial kill switch: modal.FunctionCall.from_id(<call id>).cancel()
(call ids in the manifest). Trials keep whole-trial resume: a re-spawned
index self-skips on tune_arm's fit fingerprint.
"""
import json
import time
from pathlib import Path

import modal

APP_NAME = "safetysi-gemma4e2b-lora-hpo"
app = modal.App(APP_NAME)

try:
    WORKTREE = Path(__file__).resolve().parents[2]
except IndexError:  # re-imported inside the container
    WORKTREE = Path("/repo")


def _local_path(value: str) -> Path:
    """A path on this machine; a site placeholder in it (``$STAGE``, the staging directory) expands through sisteer.site from
    this checkout, and an unset one refuses by name."""
    import sys
    if str(WORKTREE / "src") not in sys.path:
        sys.path.insert(0, str(WORKTREE / "src"))
    from sisteer import site as _site
    return _site.expand_path(value, key="--manifest")

MODEL_KEY = "gemma4e2b"
MODEL_REPO = "google/gemma-4-E2B-it"
MODEL_REV = "3e22461f65e89153144f8adb70e3b8c2cc9845a7"
CFG = "/repo/configs/two_si_gemma4e2b_hpo.yaml"
SI = "safety_refusal"
ARM = "lora_r4"
POSITIONS = "response"
ROUND = 1
N_TRIALS = 16
TOP_K = 200
EPOCHS = 10           # the safety-lane fit regime (10-epoch cosine)
EFFECTIVE_BATCH = 32  # tune_arm's default — the E2B psr_kl lane's regime
FIT_SUBSAMPLE = 0     # full pool — the E2B safety-lane precedent (8274
#                       pairs, small model: train on exactly what the fit
#                       would)
PROBE_LORA_RANK = 16  # mem ladder probes the rank grid's worst case

PAIRS = f"/vol/pool/{MODEL_KEY}/pairs_{SI}.jsonl"
TRIALS_NAME = f"r{ROUND}_{MODEL_KEY}_{ARM}_{POSITIONS}.json"
TRIALS_VOL = f"/vol/hpo/trials/{TRIALS_NAME}"
OUT_CELL = f"out/hpo/{MODEL_KEY}/{ARM}_{POSITIONS}"

_IGNORE = [".git", "**/__pycache__", ".venv", "out/**", "wandb/**",
           ".pytest_cache", "worktrees"]

# The STANDARD fit image — twosi_fits.py's image_std recipe (cu128 torch
# tree from the lockfile-exported requirements). E2B is
# Gemma4ForConditionalGeneration with standard attention: it never needs
# (and must not silently take) the qwen fused-kernel image.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("unixodbc")
    .pip_install_from_requirements(
        str(WORKTREE / "scripts/modal/requirements.txt"),
        extra_index_url="https://download.pytorch.org/whl/cu128",
    )
    .add_local_dir(str(WORKTREE), remote_path="/repo", ignore=_IGNORE)
)

vol = modal.Volume.from_name("twosi-fits")
_secrets = [modal.Secret.from_name("hf-token"),
            modal.Secret.from_name("wandb")]
_COMMON = dict(volumes={"/vol": vol}, secrets=_secrets)

#: tune_arm.py invoked DIRECTLY (the twosi_fits._hpo_impl pattern): this
#: lane runs at the module's own DISTILL_EFFECTIVE_BATCH = 32, so the
#: qwen lanes' eff-64 monkeypatch wrapper is deliberately absent.
TUNE_ARM = "/repo/scripts/tune_arm.py"


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


def _ensure_model() -> None:
    import os

    os.environ["HF_HOME"] = "/vol/hf"
    snap = Path(f"/vol/hf/hub/models--{MODEL_REPO.replace('/', '--')}"
                f"/snapshots/{MODEL_REV}")
    if not (snap / "config.json").exists():
        from huggingface_hub import snapshot_download

        # gemma is GATED — HF_TOKEN arrives via the hf-token secret.
        snapshot_download(MODEL_REPO, revision=MODEL_REV)
        vol.commit()
    os.environ["HF_HUB_OFFLINE"] = "1"


@app.function(image=image, cpu=8, memory=32768, timeout=7200, **_COMMON)
def stage() -> dict:
    """Model snapshot -> volume + verify the staged pool/tokcache/teacher/
    trials (all pre-staged by the psr lane except this arm's trials)."""
    import os

    _ensure_model()
    out = {"model": MODEL_REPO, "rev": MODEL_REV, "staged": True}
    for tag, p in (("pairs", PAIRS), ("tokcache", PAIRS + ".tokcache.npz"),
                   ("teacher", PAIRS + ".teachertopk.safetensors"),
                   ("trials", TRIALS_VOL)):
        out[tag] = (f"present ({os.path.getsize(p)} bytes)"
                    if os.path.exists(p) else "MISSING")
    return out


@app.function(gpu="H200", image=image, cpu=8, memory=65536, timeout=10800,
              scaledown_window=120, **_COMMON)
def hpomem() -> dict:
    """OFFLINE-teacher bs ladder for (gemma4e2b, safety_refusal, lora_r4)
    at the SI's resolved max_len 8192, probed at lora rank 16 (the grid's
    worst case): largest power-of-2 whose peak RESERVED stays under
    0.85 x VRAM (worst-case longest rows, the real trainer path — the
    config carries NO gradient_checkpointing for this arm, so the ladder
    measures the uncheckpointed footprint the trials will run under; the
    eff-32 accum accounting). bs is NOT reused from the psr arm — this
    ladder IS the lora measurement."""
    import re
    import subprocess
    import sys

    import torch

    _ensure_model()
    vram_gib = torch.cuda.get_device_properties(0).total_memory / 2**30
    guard_gib = 0.85 * vram_gib
    out = {"model": MODEL_KEY, "si": SI, "arm": ARM,
           "probe_lora_rank": PROBE_LORA_RANK,
           "gpu": torch.cuda.get_device_name(0),
           "vram_gib": round(vram_gib, 1), "guard_gib": round(guard_gib, 1),
           "effective_batch": EFFECTIVE_BATCH, "ladder": {}}
    best = None
    for b in (2, 4, 8, 16, 32):
        p = subprocess.run(
            [sys.executable, TUNE_ARM, "probe-mem",
             "--config", CFG, "--pairs", PAIRS, "--si", SI,
             "--arm", ARM, "--positions", POSITIONS,
             "--lora-rank", str(PROBE_LORA_RANK),
             "--batch-size", str(b), "--steps", "6",
             "--teacher-topk", "auto"],
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
        out["ladder"][str(b)] = row
        line = p.stdout.splitlines()[-1] if p.stdout.splitlines() else ""
        print(f"[{MODEL_KEY}/{SI}/{ARM}] bs={b} rc={p.returncode} "
              f"alloc={alloc} reserved={reserved} s/step={s_step} "
              f"oom={oom} | {line}")
        if p.returncode == 0 and reserved is not None \
                and reserved <= guard_gib:
            best = b
        else:
            break
    out["best_bs"] = best
    Path("/vol/out/hpo/memprobe").mkdir(parents=True, exist_ok=True)
    Path(f"/vol/out/hpo/memprobe/{MODEL_KEY}_{SI}_{ARM}.json").write_text(
        json.dumps(out, indent=1))
    vol.commit()
    return out


@app.function(gpu="H200", image=image, cpu=8, memory=65536, timeout=10800,
              scaledown_window=120, **_COMMON)
def probe_debug(bs: int) -> dict:
    """ONE probe-mem rung with the subprocess's FULL output returned —
    the hpomem ladder captures and discards it, which made an earlier
    lane's bs=2 OOM undiagnosable from the summary row alone."""
    import subprocess
    import sys

    _ensure_model()
    p = subprocess.run(
        [sys.executable, TUNE_ARM, "probe-mem",
         "--config", CFG, "--pairs", PAIRS, "--si", SI,
         "--arm", ARM, "--positions", POSITIONS,
         "--lora-rank", str(PROBE_LORA_RANK),
         "--batch-size", str(bs), "--steps", "6",
         "--teacher-topk", "auto"],
        cwd="/repo", env=_env(), capture_output=True, text=True)
    return {"bs": bs, "rc": int(p.returncode),
            "stdout_tail": p.stdout[-4000:], "stderr_tail": p.stderr[-8000:]}


@app.function(gpu="H200", image=image, cpu=8, memory=65536, timeout=86400,
              max_containers=16, scaledown_window=120, **_COMMON)
def hpo_trial(item: tuple) -> dict:
    """ONE Sobol trial: (index, bs). The twosi_fits._hpo_impl pattern at
    the E2B safety-lane regime — epochs 10, full pool, eff batch 32 (the
    tune_arm default, no wrapper), offline top-K teacher, live W&B;
    publishes r1_tNN (summary + metrics + artifacts) to the volume."""
    import shutil
    import statistics
    import subprocess
    import sys
    import threading

    index, bs = int(item[0]), int(item[1])
    _ensure_model()

    tdir = f"r{ROUND}_t{index:02d}"
    out_root = Path(f"/root/out/hpo/{MODEL_KEY}/{ARM}_{POSITIONS}")
    trial_dir = out_root / tdir
    trial_dir.mkdir(parents=True, exist_ok=True)
    seeded = Path(f"/vol/{OUT_CELL}/{tdir}/summary.json")
    if seeded.exists():
        shutil.copy2(seeded, trial_dir / "summary.json")

    # GPU-duty spot-check once the optimizer loop is live (gpu-util skill).
    metrics_path = trial_dir / "metrics.jsonl"
    dmon_rows: list = []
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
        sys.executable, TUNE_ARM, "trial",
        "--config", CFG,
        "--trials", TRIALS_VOL,
        "--index", str(index),
        "--pairs", PAIRS,
        "--out", str(out_root),
        "--si", SI,
        "--arm", ARM, "--positions", POSITIONS,
        "--batch-size", str(bs),
        "--epochs", str(EPOCHS),
        "--fit-subsample", str(FIT_SUBSAMPLE),
        "--teacher-topk", "auto",
    ]
    if index == 0:
        cmd.append("--profile")  # one profiler trace for the cell
    t0 = time.time()
    # STREAM the trial's merged output to container stdout (visible live in
    # `modal app logs`) while accumulating it for the needle checks below —
    # a captured-and-silent container made the first 3B gate undebuggable.
    proc = subprocess.Popen(cmd, cwd="/repo", env=_env(),
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True)
    out_lines: list = []
    assert proc.stdout is not None
    for line in proc.stdout:
        print(f"[t{index:02d}] {line}", end="", flush=True)
        out_lines.append(line)
    rc = proc.wait()
    out_text = "".join(out_lines)
    wall = time.time() - t0
    done.set()

    skipped = "skipping" in out_text and "MATCHING fit" in out_text
    res: dict = {"model": MODEL_KEY, "arm": ARM, "index": index,
                 "round": ROUND, "si": SI, "bs": bs,
                 "rc": int(rc),
                 "wall_s": round(float(wall), 1), "skipped": bool(skipped)}
    for tag, needle in (("tokcache_warmed", "[tokcache] warmed"),
                        ("teacher_verified", "[teacher-topk] verified")):
        res[tag] = needle in out_text
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
            "lora_rank": tr.get("lora_rank"),
            "warmup": tr.get("warmup_frac"),
            "temperature": tr.get("temperature"),
            "weight_decay": tr.get("weight_decay"),
            "accum_steps": s.get("accum_steps"),
            "effective_batch": s.get("effective_batch"),
            "epochs": s.get("epochs"),
            "best_epoch": s.get("best_epoch"),
            "stopped_early": s.get("stopped_early"),
            "teacher_forward": s.get("teacher_forward"),
            "n_fit": s.get("n_fit"),
            "floor_ok": s.get("floor_ok"),
            "floor_ratio": (round(float(s["floor_ratio"]), 4)
                            if s.get("floor_ratio") is not None else None),
            "final_residual_ratio": s.get("final_residual_ratio"),
            "peak_gib": (round(float(s["peak_alloc_gib"]), 1)
                         if s.get("peak_alloc_gib") else None),
            "wandb_run": s.get("wandb_run"),
        })
        if rc != 0 and s.get("status") not in (
                "ok", "nan_diverged", "gate_dead", "floor_failed",
                "ratio_diverged"):
            res["stderr_tail"] = str(out_text[-1200:])
    else:
        res["status"] = "no-summary"
        res["stderr_tail"] = str(out_text[-1200:])

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
        dest = Path(f"/vol/{OUT_CELL}/{tdir}")
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copytree(trial_dir, dest, dirs_exist_ok=True)
        vol.commit()
    return res


def _spawn_rows(indices, bs: int, manifest: str):
    """Durable spawns on the DEPLOYED app (the fits-spawn pattern):
    manifest pre-seeded with call_id=null rows, rewritten after EVERY
    spawn; already-published trials are recorded, not re-spawned."""
    import getpass
    from datetime import datetime, timezone

    fn = modal.Function.from_name(APP_NAME, "hpo_trial")
    mpath = _local_path(manifest)
    mpath.parent.mkdir(parents=True, exist_ok=True)
    doc = {"app": APP_NAME, "function": "hpo_trial",
           "model": MODEL_KEY, "arm": ARM, "si": SI, "round": ROUND,
           "bs": int(bs), "effective_batch": EFFECTIVE_BATCH,
           "epochs": EPOCHS, "trials_file": TRIALS_VOL,
           "out_cell": OUT_CELL, "launcher": getpass.getuser(),
           "written_at": None, "trials": []}
    if mpath.exists():
        try:
            old = json.loads(mpath.read_text())
            doc["trials"] = [t for t in old.get("trials", [])
                             if int(t.get("index", -1)) not in set(indices)]
        except ValueError:
            print(f"[spawn] {mpath} unparseable — rewriting it")

    def _flush():
        doc["written_at"] = datetime.now(timezone.utc).isoformat(
            timespec="seconds")
        tmp = mpath.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(doc, indent=1) + "\n")
        tmp.replace(mpath)

    done = set()
    try:
        for e in vol.listdir(OUT_CELL):
            name = e.path.rsplit("/", 1)[-1]
            if name.startswith(f"r{ROUND}_t"):
                done.add(int(name.split("_t")[-1]))
    except Exception:  # noqa: BLE001 — cell tree absent on first launch
        pass
    rows = [{"index": i, "bs": int(bs), "call_id": None, "spawn_time": None,
             "status": ("already-published" if i in done
                        else "pending-spawn")}
            for i in indices]
    doc["trials"] = sorted(doc["trials"] + rows, key=lambda t: t["index"])
    _flush()
    calls = []
    for r in rows:
        if r["status"] == "already-published":
            continue
        call = fn.spawn((r["index"], int(bs)))
        r["call_id"] = call.object_id
        r["spawn_time"] = datetime.now(timezone.utc).isoformat(
            timespec="seconds")
        r["status"] = "spawned"
        calls.append((r["index"], call))
        _flush()
        print(f"[spawn] t{r['index']:02d} -> {call.object_id}")
    print(f"[spawn] {len(calls)} trial(s) spawned "
          f"({len(rows) - len(calls)} already published); manifest: {mpath}")
    return calls


@app.local_entrypoint()
def main(mode: str = "status", bs: int = 0, index: int = 0,
         manifest: str = "$STAGE/gemma4e2b_lora_calls.json"):
    if mode == "stage":
        print(json.dumps(stage.remote(), indent=1))
        return

    if mode == "hpomem":
        print(json.dumps(hpomem.remote(), indent=1))
        return

    if mode == "probedebug":
        if bs <= 0:
            raise SystemExit("--bs required")
        r = probe_debug.remote(bs)
        print(f"rc={r['rc']} bs={r['bs']}\n--- stdout tail ---\n"
              f"{r['stdout_tail']}\n--- stderr tail ---\n{r['stderr_tail']}")
        return

    if mode == "result":
        # Fetch one spawned trial's result by manifest call id (short-lived
        # client; returns immediately if not finished).
        mpath = _local_path(manifest)
        rows = json.loads(mpath.read_text())["trials"]
        row = next(r for r in rows if r["index"] == index)
        fc = modal.FunctionCall.from_id(row["call_id"])
        try:
            res = fc.get(timeout=10)
        except TimeoutError:
            print(f"t{index:02d} ({row['call_id']}): still running")
            return
        print(json.dumps(res, indent=1))
        ok = (res.get("status") in ("ok", "nan_diverged", "gate_dead",
                                    "floor_failed", "ratio_diverged")
              and res.get("rc") == 0
              and res.get("tokcache_warmed") and res.get("teacher_verified")
              and res.get("effective_batch") == EFFECTIVE_BATCH)
        print(f"\nTRIAL t{index:02d} {'CLEAN' if ok else 'NOT-CLEAN'}: "
              f"status={res.get('status')} rc={res.get('rc')} "
              f"tokcache={res.get('tokcache_warmed')} "
              f"teacher={res.get('teacher_verified')} "
              f"eff_batch={res.get('effective_batch')} "
              f"rank={res.get('lora_rank')} "
              f"s/step={res.get('sec_per_step_median')} "
              f"duty={res.get('mean_sm_pct')}%")
        return

    if mode == "spawn":
        # ALL 16 trials t00..t15 durably on the DEPLOYED app — the proven
        # psr-lane recipe, no separate full-t00 gate for this lane.
        if bs <= 0:
            raise SystemExit("--bs required (the hpomem best power-of-2)")
        _spawn_rows(list(range(0, N_TRIALS)), bs, manifest)
        return

    if mode == "status":
        # Client-side volume poll: which trials have published summaries.
        rows = []
        for i in range(N_TRIALS):
            tdir = f"{OUT_CELL}/r{ROUND}_t{i:02d}"
            try:
                names = {e.path.rsplit("/", 1)[-1] for e in vol.listdir(tdir)}
            except Exception:  # noqa: BLE001 — not published yet
                names = set()
            rows.append((i, "published" if "summary.json" in names
                         else "pending"))
        for i, st in rows:
            print(f"t{i:02d}: {st}")
        n = sum(1 for _, st in rows if st == "published")
        print(f"{n}/{N_TRIALS} published (volume twosi-fits, {OUT_CELL})")
        return

    raise SystemExit(f"unknown --mode {mode!r}")
