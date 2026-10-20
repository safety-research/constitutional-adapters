"""Stage-1 Modal H200 utilization probe (twosi-model-sweeps).

Confirms Modal actually provisions an H200 for us, reports its VRAM (141GB SXM
vs 80GB — decides whether the E4B bs=7 memprobe result holds), runs a matmul
burn, and measures mean SM% over the burn — filtered to the assigned device
(the util-test caveat: dmon otherwise blends neighboring GPUs).

Run once Modal auth is set:
    ~/venvs/modal/bin/modal run $HOME/modal-h200-probe/probe.py
"""
import modal

app = modal.App("h200-util-probe")

# Match the eventual E4B fit env (torch 2.11 cu128); bundles the CUDA runtime.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.11.0",
        extra_index_url="https://download.pytorch.org/whl/cu128",
    )
)


@app.function(gpu="H200", image=image, timeout=600)
def probe(burn_s: int = 30):
    import subprocess, time, torch

    dev = torch.cuda.get_device_properties(0)
    info = {
        "gpu_name": torch.cuda.get_device_name(0),
        "vram_gb": round(dev.total_memory / 1e9, 1),
        "cuda": str(torch.version.cuda),
        "torch": str(torch.__version__),
        "visible_gpus": torch.cuda.device_count(),
    }

    # dmon filtered to device 0 only (don't blend neighbors' duty)
    n_samples = burn_s + 5
    dmon = subprocess.Popen(
        ["nvidia-smi", "dmon", "-s", "u", "-c", str(n_samples), "-i", "0"],
        stdout=subprocess.PIPE, text=True,
    )

    # matmul burn to saturate the SMs
    a = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16)
    t0 = time.time()
    iters = 0
    while time.time() - t0 < burn_s:
        for _ in range(50):          # keep the GPU queue fed; don't sync per-iter
            c = a @ b
            a = c                     # chain so kernels stay dependent + saturating
        torch.cuda.synchronize()
        iters += 50

    out, _ = dmon.communicate(timeout=n_samples + 30)
    sms = []
    for line in out.splitlines():
        p = line.split()
        if len(p) >= 2 and p[0].isdigit():   # data rows: "0 <sm> <mem> <enc> <dec>"
            try:
                sms.append(int(p[1]))
            except ValueError:
                pass
    steady = sms[3:] if len(sms) > 6 else sms   # drop ramp-up samples
    info["mean_sm_pct"] = round(sum(steady) / len(steady), 1) if steady else -1
    info["n_samples"] = len(steady)
    info["matmul_iters"] = iters
    return info


@app.local_entrypoint()
def main():
    import json
    print(json.dumps(probe.remote(), indent=2))
