"""Qwen3.5-9B (thinking-off) psr_kl_all safety-lane HPO on Modal —
8-trial Sobol lr x clip (twosi-model-sweeps; the proven llama3b safety-
lane recipe at 9B, with a WIDER lr box — no Qwen3.5-9B prior).

Arm psr_kl_all ONLY, SI safety_refusal ONLY, positions=response. All other
HPs frozen at the 3B safety-lane regime: T=1 / wd=0.01 pinned on every
trial row, warmup fixed 0.15, 10-epoch warmup_cosine, patience-3 early
stop, the residual_ratio>1.5 hard-stop, offline top-K=200 teacher
(--teacher-topk auto), EFFECTIVE BATCH 64 — threaded by patching
tune_arm's DISTILL_EFFECTIVE_BATCH module constant (its documented
monkeypatch seam) in the trial subprocess; accum_steps = ceil(64 / bs)
rides the fingerprint and the summary as usual. bs is NOT reused from the
3B (9B > 3B): hpomem re-probes, expect bs < 4.

Config: configs/two_si_qwen35_9b_hpo.yaml (INTERIM, hand-derived — the
llama3b precedent; teacher_prefix_cache FALSE, the qwen3_5
LinearAttentionLayer evidence). Volume: `twosi-fits`, model key
qwen35_9b — pool/qwen35_9b/pairs_safety_refusal.jsonl + its pre-built
.tokcache.npz sidecar (uploaded, NEVER rebuilt; its header pins
this exact model id + revision), teacher sidecar built by --mode teacher,
trials at hpo/trials/, outputs at out/hpo/qwen35_9b/psr_kl_all_response/
r1_tNN/.

Order (GPU-safe; each step validates the previous one's artifact):
  ~/venvs/modal/bin/modal run scripts/modal/qwen35_9b_hpo.py --mode stage
  ~/venvs/modal/bin/modal run scripts/modal/qwen35_9b_hpo.py --mode teacher \
      --shards 4                    # top-K=200 build -> merge (byte-pinned)
  ~/venvs/modal/bin/modal run scripts/modal/qwen35_9b_hpo.py --mode hpomem
      # offline-teacher bs ladder (2..64) at safety's max_len 8192
  ~/venvs/modal/bin/modal deploy scripts/modal/qwen35_9b_hpo.py
  ~/venvs/modal/bin/modal run scripts/modal/qwen35_9b_hpo.py --mode spawn \
      --bs <best power-of-2>        # t00..t07, ALL 8, durable
      # FunctionCalls on the DEPLOYED app (the fits-spawn durability
      # lesson: .map dies with its client) — no separate t00 gate:
      # teacher+hpomem already validated model+pool+tokenizer, and the
      # per-trial hard-stop + divergence guard protect each trial;
      # manifest $STAGE/qwen35_9b/hpo_calls_r1.json
  ~/venvs/modal/bin/modal run scripts/modal/qwen35_9b_hpo.py --mode status
      # summaries published so far (client-side volume poll)

Per-trial kill switch: modal.FunctionCall.from_id(<call id>).cancel()
(call ids in the manifest). Trials keep whole-trial resume: a re-spawned
index self-skips on tune_arm's fit fingerprint.
"""
import json
import time
from pathlib import Path

import modal

app = modal.App("safetysi-qwen35-9b-hpo")

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

MODEL_KEY = "qwen35_9b"
MODEL_REPO = "Qwen/Qwen3.5-9B"
MODEL_REV = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
CFG = "/repo/configs/two_si_qwen35_9b_hpo.yaml"
SI = "safety_refusal"
ARM = "psr_kl_all"
POSITIONS = "response"
ROUND = 1
N_TRIALS = 8
TOP_K = 200
EPOCHS = 10          # the 3B safety-lane regime (10-epoch cosine)
EFFECTIVE_BATCH = 64  # the 3B safety-lane regime (the 8B ran 32)
FIT_SUBSAMPLE = 0    # full pool — the E2B/3B safety-lane precedent
#                      (10586 pairs: train on exactly what the fit would)

PAIRS = f"/vol/pool/{MODEL_KEY}/pairs_{SI}.jsonl"
TRIALS_NAME = f"r{ROUND}_{MODEL_KEY}_{ARM}_{POSITIONS}.json"
TRIALS_VOL = f"/vol/hpo/trials/{TRIALS_NAME}"
OUT_CELL = f"out/hpo/{MODEL_KEY}/{ARM}_{POSITIONS}"

_IGNORE = [".git", "**/__pycache__", ".venv", "out/**", "wandb/**",
           ".pytest_cache", "worktrees"]

#: The Qwen hybrid-attention image — twosi_fits.py's image_qwen recipe
#: VERBATIM (the fused-kernel layer; adopted 2026-08-28 after the
#: torch linear-attention fallback OOM'd this lane's memprobe at every
#: max_len): nvidia/cuda DEVEL base (nvcc 12.8 for the causal-conv1d
#: source build — no wheel exists at torch 2.11), the same requirements
#: tree, then the hybrid stack pins: triton 3.7.1 OVERRIDES the
#: lockfile's 3.6.0 (fla's gated-chunk backward miscompiles on
#: [3.4.0, 3.7.1)), flash-linear-attention/fla-core 0.5.2,
#: causal-conv1d 1.6.2.post1 built for sm90 (H100/H200).
image = (
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

vol = modal.Volume.from_name("twosi-fits")
_secrets = [modal.Secret.from_name("hf-token"),
            modal.Secret.from_name("wandb")]
_COMMON = dict(volumes={"/vol": vol}, secrets=_secrets)

#: tune_arm.py under its documented monkeypatch seam: the module constants
#: are read as properties exactly so a caller can override them —
#: DISTILL_EFFECTIVE_BATCH 32 -> 64 is THIS sweep's one regime change.
#: `python -c CODE trial ...` leaves the real args at sys.argv[1:].
_TUNE_ARM_EFF64 = (
    "import importlib.util, sys\n"
    "spec = importlib.util.spec_from_file_location("
    "'tune_arm', '/repo/scripts/tune_arm.py')\n"
    "m = importlib.util.module_from_spec(spec)\n"
    # sys.modules BEFORE exec_module: @dataclass under PEP-563 string
    # annotations resolves types via sys.modules[cls.__module__], so an
    # unregistered module crashes every dataclass definition in the file.
    "sys.modules['tune_arm'] = m\n"
    "spec.loader.exec_module(m)\n"
    f"m.DISTILL_EFFECTIVE_BATCH = {EFFECTIVE_BATCH}\n"
    "sys.argv = ['tune_arm.py'] + sys.argv[1:]\n"
    "m.main()\n"
)


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

        # hf-token secret in scope in case the repo is gated
        snapshot_download(MODEL_REPO, revision=MODEL_REV)
        vol.commit()
    os.environ["HF_HUB_OFFLINE"] = "1"


@app.function(image=image, cpu=8, memory=32768, timeout=7200, **_COMMON)
def stage() -> dict:
    """Model snapshot -> volume + verify the staged pool/tokcache/trials."""
    import os

    _ensure_model()
    out = {"model": MODEL_REPO, "rev": MODEL_REV, "staged": True}
    for tag, p in (("pairs", PAIRS), ("tokcache", PAIRS + ".tokcache.npz"),
                   ("trials", TRIALS_VOL)):
        out[tag] = (f"present ({os.path.getsize(p)} bytes)"
                    if os.path.exists(p) else "MISSING")
    return out


def _teacher_impl(item: tuple) -> dict:
    """One shard (or the whole pool) of the offline top-K=200 teacher —
    the twosi_fits.py precompute-teacher path verbatim."""
    import os
    import subprocess
    import sys

    shard_i, shard_n = int(item[0]), int(item[1])
    _ensure_model()
    if not os.path.exists(PAIRS):
        return {"si": SI, "status": "no-pool", "error": f"{PAIRS} not staged"}
    if shard_n:
        art = (f"{PAIRS}.teachertopk.shard{shard_i:03d}"
               f"of{shard_n:03d}.safetensors")
    else:
        art = PAIRS + ".teachertopk.safetensors"
    if os.path.exists(art) and os.path.exists(art + ".json"):
        return {"si": SI, "shard": shard_i if shard_n else None,
                "status": "artifact-present", "path": art}
    cmd = [sys.executable, "/repo/scripts/train_two_si_objects.py",
           "precompute-teacher", "--config", CFG, "--si", SI,
           "--pairs", PAIRS, "--top-k", str(TOP_K)]
    if shard_n:
        cmd += ["--shard-index", str(shard_i), "--shard-count", str(shard_n)]
    t0 = time.time()
    p = subprocess.run(cmd, cwd="/repo", env=_env(), capture_output=True,
                       text=True)
    out = {"si": SI, "shard": shard_i if shard_n else None,
           "rc": int(p.returncode), "wall_s": round(time.time() - t0, 1),
           "path": art,
           "tail": [str(x) for x in (p.stdout.splitlines()[-3:]
                                     if p.stdout else [])]}
    if p.returncode != 0:
        out["status"] = "failed"
        out["stderr_tail"] = str(p.stderr[-1500:])
    else:
        out["status"] = "built"
        vol.commit()
    return out


@app.function(gpu="H200", image=image, cpu=8, memory=65536, timeout=86400,
              max_containers=16, scaledown_window=120, **_COMMON)
def teacher(item: tuple) -> dict:
    return _teacher_impl(item)


@app.function(image=image, cpu=16, memory=98304, timeout=14400, **_COMMON)
def teacher_merge(shard_n: int) -> dict:
    """CPU merge of the shard artifacts — byte-identical to an unsharded
    build (tests/test_teacher_topk.py pins the merge)."""
    import os
    import subprocess
    import sys

    art = PAIRS + ".teachertopk.safetensors"
    if os.path.exists(art) and os.path.exists(art + ".json"):
        return {"si": SI, "status": "artifact-present", "path": art}
    t0 = time.time()
    p = subprocess.run(
        [sys.executable, "/repo/scripts/train_two_si_objects.py",
         "merge-teacher", "--config", CFG, "--pairs", PAIRS,
         "--shard-count", str(shard_n)],
        cwd="/repo", env=_env(), capture_output=True, text=True)
    out = {"si": SI, "rc": int(p.returncode),
           "wall_s": round(time.time() - t0, 1), "path": art,
           "tail": [str(x) for x in (p.stdout.splitlines()[-3:]
                                     if p.stdout else [])]}
    if p.returncode != 0:
        out["status"] = "failed"
        out["stderr_tail"] = str(p.stderr[-1500:])
    else:
        out["status"] = "merged"
        vol.commit()
    return out


@app.function(gpu="H200", image=image, cpu=8, memory=65536, timeout=10800,
              scaledown_window=120, **_COMMON)
def hpomem() -> dict:
    """OFFLINE-teacher bs ladder for (qwen35_9b, safety_refusal,
    psr_kl_all) at the SI's resolved max_len 8192: largest power-of-2
    whose peak RESERVED stays under 0.85 x VRAM (worst-case longest rows,
    the real trainer path, the eff-64 accum accounting)."""
    import re
    import subprocess
    import sys

    import torch

    _ensure_model()
    vram_gib = torch.cuda.get_device_properties(0).total_memory / 2**30
    guard_gib = 0.85 * vram_gib
    out = {"model": MODEL_KEY, "si": SI, "arm": ARM,
           "gpu": torch.cuda.get_device_name(0),
           "vram_gib": round(vram_gib, 1), "guard_gib": round(guard_gib, 1),
           "effective_batch": EFFECTIVE_BATCH, "ladder": {}}
    best = None
    for b in (2, 4, 8, 16, 32, 64):
        p = subprocess.run(
            [sys.executable, "-c", _TUNE_ARM_EFF64, "probe-mem",
             "--config", CFG, "--pairs", PAIRS, "--si", SI,
             "--arm", ARM, "--positions", POSITIONS,
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
def probe_one(b: int) -> dict:
    """ONE diagnostic probe-mem rung with FULL output capture — the
    ladder's oom rows discard stderr, which made the 9B bs=2 OOM
    undiagnosable; this returns the tails no matter what."""
    import subprocess
    import sys

    _ensure_model()
    p = subprocess.run(
        [sys.executable, "-c", _TUNE_ARM_EFF64, "probe-mem",
         "--config", CFG, "--pairs", PAIRS, "--si", SI,
         "--arm", ARM, "--positions", POSITIONS,
         "--batch-size", str(b), "--steps", "6",
         "--teacher-topk", "auto"],
        cwd="/repo", env=_env(), capture_output=True, text=True)
    return {"bs": b, "rc": int(p.returncode),
            "stdout_tail": str(p.stdout[-3000:]),
            "stderr_tail": str(p.stderr[-6000:])}


@app.function(gpu="H200", image=image, cpu=8, memory=65536, timeout=86400,
              max_containers=16, scaledown_window=120, **_COMMON)
def hpo_trial(item: tuple) -> dict:
    """ONE Sobol trial: (index, bs). The twosi_fits._hpo_impl pattern at
    the 9B safety-lane regime — epochs 10, full pool, eff batch 64 via the
    wrapper's DISTILL_EFFECTIVE_BATCH patch, offline top-K teacher, live
    W&B; publishes r1_tNN (summary + metrics + artifacts) to the volume."""
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
        sys.executable, "-c", _TUNE_ARM_EFF64, "trial",
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
    # a captured-and-silent container made the first gate undebuggable.
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

    fn = modal.Function.from_name("safetysi-qwen35-9b-hpo", "hpo_trial")
    mpath = _local_path(manifest)
    mpath.parent.mkdir(parents=True, exist_ok=True)
    doc = {"app": "safetysi-qwen35-9b-hpo", "function": "hpo_trial",
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
def main(mode: str = "status", shards: int = 0, bs: int = 0, index: int = 0,
         manifest: str = "$STAGE/qwen35_9b/hpo_calls_r1.json"):
    if mode == "stage":
        print(json.dumps(stage.remote(), indent=1))
        return

    if mode == "teacher":
        if shards > 0:
            results = list(teacher.map([(i, shards) for i in range(shards)]))
            print(json.dumps(results, indent=1))
            bad = [r for r in results
                   if r.get("status") not in ("built", "artifact-present")]
            if bad:
                for r in bad:
                    print("NOT-OK:", json.dumps(r))
                raise SystemExit(
                    "shard build(s) failed — NOT merging a partial set")
            print(json.dumps(teacher_merge.remote(shards), indent=1))
        else:
            print(json.dumps(teacher.remote((-1, 0)), indent=1))
        return

    if mode == "hpomem":
        print(json.dumps(hpomem.remote(), indent=1))
        return

    if mode == "probe":
        if bs <= 0:
            raise SystemExit("--bs required")
        print(json.dumps(probe_one.remote(bs), indent=1))
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
              f"s/step={res.get('sec_per_step_median')} "
              f"duty={res.get('mean_sm_pct')}%")
        return

    if mode == "spawn":
        # ALL 8 trials (t00..t07) — no separate t00 gate: teacher + hpomem
        # already validated model+pool+tokenizer load, and each trial has
        # the residual-ratio hard-stop + divergence guard.
        if bs <= 0:
            raise SystemExit("--bs required (the hpomem best power-of-2)")
        _spawn_rows(list(range(N_TRIALS)), bs, manifest)
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
