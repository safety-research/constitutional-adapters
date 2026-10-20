"""memprobe_fold.py — the ONE reader of a memprobe record (reproduce/02_prepare_model.sh --step memprobe) and the ONE source of the selection it
is keyed to (memprobe_fit.py builds the probed object from the same `config_selection`).

Rules the fold used to lack (2026-10-02, fits 831981/831986: both OOM at step 25 on a micro-batch folded from a record that had probed the
PAPER's rank-4 o_proj object, not the picked rank-16 MLP set — the fourth identity-less value of the night):

1. IDENTITY — a record is valid only for what it MEASURED: arm; model + revision; the memory-relevant architecture (lora_rank, the module set,
   the explicit per_expert_rank map — the trainer's top-level kwargs); and the sequence length (the record's seq_len vs the config's
   max_len). Any keyed field the record OMITS is STALE, never a pass. init_scheme is recorded but not keyed (it does not move peak memory). The pinned
   revision, gradient checkpointing and the weights dtype are keyed too — by the trainer's own rule (config_runtime) — and ride the unit
   fingerprint with the model, so a drifted revision or runtime is a NEW probe unit, never the completed one.
   A stale record exits 6 with both sides named; the caller archives it and re-probes.
2. HARDWARE — the record names the card it probed on (gpu_name, gpu_total_bytes); when the caller states the fit cluster's card class
   (--expect-gpu-name), a record from another class is refused by name (exit 7): an H200 probe cannot price an H100 fit.
3. HEADROOM — the folded micro-batch is the largest rung whose peak_alloc_bytes <= HEADROOM x gpu_total_bytes (default 0.85; the stale record's
   rung 2 "fit" at 91 % and the real corpus pushed it over). Nothing within headroom -> exit 5 (FD 23: does not fit with headroom).

--fingerprint prints the unit fingerprint of the config's CURRENT selection (rank + module set + expert map + seq_len; `psr-L<seq>` for the
ReLU arm): 02 keys the memprobe UNIT on it, so a new selection is a NEW unit through the funnel's first-submit reservation and the
paper-architecture unit stays bound to its completed job.

Exit codes: 0 folded / fingerprint printed; 2 usage; 5 does not fit; 6 stale record; 7 hardware mismatch.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import yaml

EXIT_FOLDED, EXIT_USAGE, EXIT_DOES_NOT_FIT, EXIT_STALE, EXIT_HARDWARE = 0, 2, 5, 6, 7
DEFAULT_HEADROOM = 0.85
MISSING = "<missing>"
DTYPE_SHORT = {"bfloat16": "bf16", "float32": "fp32", "float16": "fp16"}


def config_selection(cfg: dict, arm: str) -> dict:
    """The architecture the fit would BUILD for `arm` — the trainer's precedence: top-level train_kwargs.<arm> keys first, the
    fit_selection block filling what they leave unstated, the paper's defaults last. memprobe_fit builds its probed object from THIS."""
    tk = ((cfg.get("train_kwargs") or {}).get(arm) or {})
    sel = tk.get("fit_selection") or {}
    if arm == "lora_r4_ce":
        rank = int(tk.get("lora_rank", sel.get("lora_rank", 4)))
        mods = sorted(set(tk.get("lora_modules") or sel.get("lora_modules") or ["o_proj"]))
        per = tk["per_expert_rank"] if "per_expert_rank" in tk else sel.get("per_expert_rank")
        per = ({k: int(v) for k, v in per.items()} if isinstance(per, dict) else None)
    else:
        rank, mods, per = None, None, None
    init = tk.get("init_scheme") or sel.get("init_scheme") or "zero"
    return {"lora_rank": rank, "lora_modules": mods, "per_expert_rank": per, "init_scheme": init}


def config_runtime(cfg: dict, arm: str) -> dict:
    """What the fit RUNS with, by the trainer's own rule (train_two_si_objects.maybe_enable_gradient_checkpointing: train_kwargs.<arm>.
    gradient_checkpointing over train_kwargs.default's, default OFF; weights loaded in the config's dtype). The probe measures under the same."""
    tks = cfg.get("train_kwargs") or {}
    default = (tks.get("default") or {}).get("gradient_checkpointing", False)
    gc = bool((tks.get(arm) or {}).get("gradient_checkpointing", default))
    dtype = cfg.get("dtype")
    if not dtype:
        raise SystemExit("[memprobe-fold] the config states no dtype — the probe's weights dtype cannot be keyed")
    return {"gradient_checkpointing": gc, "dtype": str(dtype)}


def model_tag(cfg: dict) -> str:
    """Eight hex of sha256(model@revision): the model identity's share of the unit fingerprint."""
    return hashlib.sha256(f"{cfg.get('model')}@{cfg.get('model_revision') or ''}".encode()).hexdigest()[:8]


def config_seq_len(cfg: dict) -> int:
    v = cfg.get("max_len") or cfg.get("sequence_max_len")
    if not v:
        raise SystemExit("[memprobe-fold] the config states no max_len / sequence_max_len — the probe's sequence length cannot be keyed")
    return int(v)


def fingerprint(cfg: dict, arm: str) -> str:
    """The memprobe UNIT's fingerprint for the config's current selection: rank + module set + expert map + seq_len; `psr-L<seq>` for the ReLU arm."""
    seq = config_seq_len(cfg); rt = config_runtime(cfg, arm)
    tail = f"L{seq}-gc{int(rt['gradient_checkpointing'])}-{DTYPE_SHORT.get(rt['dtype'], rt['dtype'])}-m{model_tag(cfg)}"
    if arm != "lora_r4_ce":
        return f"psr-{tail}"
    s = config_selection(cfg, arm)
    mods = "-".join(m.replace("_proj", "") for m in s["lora_modules"])
    per = s["per_expert_rank"]
    pe = f"pe{per['shared_expert']}x{per['routed_expert']}" if per else "pe-none"
    return f"r{s['lora_rank']}-{mods}-{pe}-{tail}"


def record_identity(rec: dict) -> dict:
    """What the record MEASURED, in the keyed vocabulary; MISSING where the record omits a keyed field (an older writer)."""
    ok = rec.get("object_kwargs") or {}
    m = rec.get("model") or {}
    mods = ok.get("lora_modules")
    per = ok.get("per_expert_rank", MISSING) if "per_expert_rank" in ok else MISSING
    if isinstance(per, dict):
        per = {k: int(v) for k, v in per.items()}
    gc = rec.get("gradient_checkpointing") if "gradient_checkpointing" in rec else None
    return {"arm": rec.get("arm", MISSING), "model": m.get("model", MISSING), "revision": (m.get("revision") or MISSING),
            "gradient_checkpointing": (bool(gc) if gc is not None else MISSING), "dtype": (str(rec["dtype"]) if rec.get("dtype") else MISSING),
            "lora_rank": (int(ok["lora_rank"]) if ok.get("lora_rank") is not None else None) if "lora_rank" in ok else MISSING,
            "lora_modules": (sorted(set(mods)) if mods else None) if "lora_modules" in ok else MISSING,
            "per_expert_rank": per, "seq_len": (int(rec["seq_len"]) if rec.get("seq_len") is not None else MISSING),
            "init_scheme": ok.get("init_scheme") or "zero", "gpu_name": rec.get("gpu_name"), "gpu_total_bytes": rec.get("gpu_total_bytes")}


def staleness(rec: dict, cfg: dict, arm: str) -> list[str]:
    """Why the record cannot fold onto this config; [] when it can. Every keyed field the record omits is a reason."""
    got, want = record_identity(rec), config_selection(cfg, arm)
    why = []
    if got["arm"] != arm:
        why.append(f"arm: record {got['arm']!r} vs {arm!r}")
    if got["model"] != cfg.get("model"):
        why.append(f"model: record {got['model']!r} vs config {cfg.get('model')!r}")
    pin = cfg.get("model_revision")
    if pin:   # a pinned config is keyed to its revision: a record that omits it predates the keyed identity — STALE
        if got["revision"] == MISSING:
            why.append(f"revision: the record omits it — the config pins {pin!r}")
        elif got["revision"] != pin:
            why.append(f"revision: record {got['revision']!r} vs config {pin!r}")
    keyed = ("lora_rank", "lora_modules", "per_expert_rank") if arm == "lora_r4_ce" else ()
    for k in keyed:
        if got[k] == MISSING:
            why.append(f"{k}: the record omits it (it predates the keyed identity) — the config selects {want[k]!r}")
        elif got[k] != want[k]:
            why.append(f"{k}: record probed {got[k]!r}, the config selects {want[k]!r}")
    seq = config_seq_len(cfg)
    if got["seq_len"] == MISSING:
        why.append(f"seq_len: the record omits it — the config's max_len is {seq}")
    elif got["seq_len"] != seq:
        why.append(f"seq_len: record probed {got['seq_len']}, the config's max_len is {seq}")
    rt = config_runtime(cfg, arm)   # the runtime the fit uses (trainer's rule): a probe under another one priced a different job
    for k in ("gradient_checkpointing", "dtype"):
        if got[k] == MISSING:
            why.append(f"{k}: the record omits it — the fit runs with {rt[k]!r}")
        elif got[k] != rt[k]:
            why.append(f"{k}: record probed {got[k]!r}, the fit runs with {rt[k]!r}")
    return why


def hardware_mismatch(rec: dict, expect_gpu_name: str | None) -> str | None:
    if not expect_gpu_name:
        return None
    name = str(rec.get("gpu_name") or "")
    if expect_gpu_name.lower() not in name.lower():
        return (f"HARDWARE: the record probed on {name!r} ({rec.get('gpu_total_bytes')} bytes), the fit cluster declares {expect_gpu_name!r} — "
                "a probe on another card class cannot price this fit; refusing")
    return None


def fold_with_headroom(rec: dict, headroom: float) -> dict:
    total = rec.get("gpu_total_bytes")
    rows, chosen = [], None
    for r in rec.get("rungs") or []:
        peak = r.get("peak_alloc_bytes")
        frac = (peak / total) if (peak is not None and total) else None
        within = bool(r.get("fits")) and frac is not None and frac <= headroom
        rows.append({"micro_batch": int(r["micro_batch"]), "fits": bool(r.get("fits")), "peak_alloc_bytes": peak, "peak_frac": frac,
                     "within_headroom": within, "s_per_step": r.get("s_per_step"), "error": r.get("error")})
        if within and (chosen is None or int(r["micro_batch"]) > chosen):
            chosen = int(r["micro_batch"])
    return {"chosen": chosen, "rows": rows, "headroom": headroom, "gpu_total_bytes": total}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("record", nargs="?", type=Path)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--fit-key"); ap.add_argument("--arm", required=True)
    ap.add_argument("--out", type=Path); ap.add_argument("--headroom", type=float, default=DEFAULT_HEADROOM)
    ap.add_argument("--expect-gpu-name", default=None, help="the fit cluster's card class (site CLUSTER_GPU_NAME); a record from another class is refused")
    ap.add_argument("--fingerprint", action="store_true", help="print the memprobe UNIT fingerprint of the config's current selection and exit")
    a = ap.parse_args(argv)
    cfg = yaml.safe_load(a.config.read_text())
    if a.fingerprint:
        print(fingerprint(cfg, a.arm)); return EXIT_FOLDED
    if a.record is None or a.out is None or not a.fit_key:
        print("[memprobe-fold] a fold needs <record> --fit-key --out", file=sys.stderr); return EXIT_USAGE
    if not (0 < a.headroom <= 1):
        print(f"[memprobe-fold] --headroom must be in (0, 1], got {a.headroom}", file=sys.stderr); return EXIT_USAGE
    rec = json.loads(a.record.read_text())
    why = staleness(rec, cfg, a.arm)
    if why:
        print(f"[memprobe-fold] STALE: {a.record} measured a different identity than {a.config.name} selects for {a.arm} — "
              + "; ".join(why) + " — not folded; re-probe the arm on the current selection")
        return EXIT_STALE
    hw = hardware_mismatch(rec, a.expect_gpu_name)
    if hw:
        print(f"[memprobe-fold] {hw}"); return EXIT_HARDWARE
    v = rec["verdict"]; ident = record_identity(rec)
    print(f"FD23: {ident['model']} {a.arm}: micro-batch {v['min_micro_batch']} fits={v['fits_at_min_micro_batch']} "
          f"largest_fitting={v['largest_fitting_micro_batch']} s/step@min={v['s_per_step_at_min']} replan_required={v.get('replan_required')}; "
          f"identity={{rank {ident['lora_rank']}, modules {ident['lora_modules']}, per_expert_rank {ident['per_expert_rank']}, seq_len {ident['seq_len']}, "
          f"gradient_checkpointing {ident['gradient_checkpointing']}, dtype {ident['dtype']}, revision {ident['revision']}}} "
          f"init={ident['init_scheme']} (recorded, not keyed); card={ident['gpu_name']!r} {ident['gpu_total_bytes']} bytes")
    f = fold_with_headroom(rec, a.headroom)
    for r in f["rows"]:
        frac = f"{r['peak_frac']:.1%}" if r["peak_frac"] is not None else "-"
        print(f"  micro-batch {r['micro_batch']}: fits={r['fits']} peak_alloc={r['peak_alloc_bytes']} ({frac} of the card) "
              f"within_headroom({a.headroom:.0%})={r['within_headroom']} s/step={r['s_per_step']} {r['error'] or ''}")
    if int(v["exit_code"]) != 0:
        print("FD23: DOES NOT FIT at micro-batch 1 — tell the user now; the fit lane waits for their call (everything else proceeds)")
        return EXIT_DOES_NOT_FIT
    if f["chosen"] is None:
        print(f"FD23: no rung fits WITH headroom ({a.headroom:.0%} of {f['gpu_total_bytes']} bytes) — the record's largest fitting rung "
              f"{v['largest_fitting_micro_batch']} sat above it; gradient checkpointing is already on — tell the user")
        return EXIT_DOES_NOT_FIT
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({a.fit_key: {a.arm: f["chosen"]}}, indent=2) + "\n")
    print(f"fold -> {a.out} (micro-batch {f['chosen']}: the largest rung within {a.headroom:.0%} headroom; "
          f"the record's largest fitting rung was {v['largest_fitting_micro_batch']})")
    return EXIT_FOLDED


if __name__ == "__main__":
    sys.exit(main())
