#!/usr/bin/env python
"""The fit-memory probe: does ``<model>`` train at micro-batch 1 with
8,192-token sequences, gradient checkpointing and bf16 weights on ONE H200 —
and what does a step cost?

    scripts/sdfbase/memprobe_fit.py --config configs/sdfbase_<key>_fits.yaml
        --arm psr_ce_all|lora_r4_ce [--pairs <pool jsonl>] --out memprobe.json
        [--ladder 1,2,4] [--seq-len 8192] [--steps 3] [--lora-rank R]
        [--lora-modules o_proj,...] [--init-scheme zero|random_lecun_1e-3]
        [--gradient-checkpointing on|off] [--report memprobe_report.md]
        [--dry-run] [--fake-cuda oom_at=4] [--model-override PATH]
        [--dtype bfloat16|float32]

For each micro-batch on the ladder (1 -> 2 -> 4 by default) the probe
builds the arm's untrained object (``sisteer.extraction.object_factory.
build_object`` — the ONE builder the production trainer, the search trial
and this probe share, so the object measured is the object fitted), runs
``--steps`` REAL optimizer steps
through the production trainer ``sisteer.extraction.objects.train_object_ce``
on synthetic-LENGTH examples — the pool's longest documents, repeated and
cut to exactly ``--seq-len`` tokens, so every row is the worst-case width
(the fits' 8,192 cap, not the corpus's median) — and records, per rung,
``{fits, peak_alloc_bytes, peak_reserved_bytes, s_per_step, tokens_per_s,
error}``; a CUDA OOM is CAUGHT per rung (the ladder stops at the first
miss: a wider batch cannot fit where a narrower one did not). The record
carries the identity the estimate rests on: model + revision, seq_len,
gradient checkpointing, dtype, the GPU, torch / transformers versions and
the MoE experts implementation.

Exit code 0 when micro-batch 1 (the smallest rung) FITS, 5 when it does
NOT (the operator is told at once and the fit lane waits — the stage
script turns the 5 into that report); 2 on a usage error. Beside the
JSON the probe writes ``memprobe_report.md``: fits / no-fits per
micro-batch, peak bytes, s/step, and the REPLAN flag (a step above
~1-2 s at micro-batch 1 triggers the fit-schedule replan before
anything is submitted).

``--fake-cuda oom_at=4`` is the test hook: the ladder loop runs on CPU
against a tiny model, the named rung(s) raise a synthetic OOM instead of
training, and the memory numbers are marked synthetic — so
tests/test_memprobe_fit.py exercises the whole loop, the JSON shape, the
report and both exit codes without a GPU. ``--dry-run`` prints the plan
(model, arm, ladder, seq_len, object kwargs, output paths) and exits 0.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import platform
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

#: the ladder: micro-batch 1 -> 2 -> 4 at 8,192 tokens
DEFAULT_LADDER = (1, 2, 4)
DEFAULT_SEQ_LEN = 8192
DEFAULT_STEPS = 3
#: the replan threshold: a measured step above this at micro-batch 1
#: (8,192 tokens) triggers the fit-schedule replan before any submission
REPLAN_S_PER_STEP = 2.0
REPLAN_S_PER_STEP_WATCH = 1.0
#: the constitution corpus's token count, the epoch the report prices
CONSTITUTION_EPOCH_TOKENS = 42.8e6
#: exit codes (FD 23)
EXIT_FITS = 0
EXIT_USAGE = 2
EXIT_DOES_NOT_FIT = 5
#: the two CE arms; ``object_factory.KIND_ALIASES`` maps each spelling to its object kind
ARMS = ("psr_ce_all", "lora_r4_ce")

#: the deterministic filler for a run with no pool (a smoke on a tree with no
#: staged pools): plain prose repeated — every token real, none padding
_SYNTHETIC_DOC = (
    "The committee met on the third morning to review the harbour survey. "
    "Each surveyor read a section aloud, the clerk noted every objection, "
    "and the chair asked for the depth soundings to be checked against the "
    "older charts before any dredging order was drafted. ")


class ProbeError(RuntimeError):
    pass


def parse_ladder(text: str) -> list[int]:
    try:
        rungs = [int(x) for x in str(text).split(",") if x.strip()]
    except ValueError:
        raise ProbeError(f"--ladder must be comma-separated ints, got {text!r}")
    if not rungs or any(r < 1 for r in rungs) or rungs != sorted(rungs) \
            or len(set(rungs)) != len(rungs):
        raise ProbeError(f"--ladder must be strictly increasing positive "
                         f"ints, got {text!r}")
    return rungs


def parse_fake_cuda(text: str | None) -> dict | None:
    """``oom_at=4`` / ``oom_at=2,4`` / ``none`` -> {"oom_rungs": set}."""
    if text is None:
        return None
    spec = {"oom_rungs": set()}
    for part in str(text).split(";"):
        part = part.strip()
        if not part or part == "none":
            continue
        if part.startswith("oom_at="):
            spec["oom_rungs"] = {int(x) for x in part[len("oom_at="):].split(",")
                                 if x.strip()}
        else:
            raise ProbeError(f"--fake-cuda: unknown token {part!r} "
                             "(known: oom_at=<rung[,rung]>, none)")
    return spec


def is_oom(exc: BaseException) -> bool:
    import torch
    oom_cls = getattr(torch, "OutOfMemoryError", None) or getattr(
        torch.cuda, "OutOfMemoryError", ())
    if isinstance(exc, oom_cls):
        return True
    msg = str(exc).lower()
    return "out of memory" in msg or "cuda oom" in msg


class FakeOOM(RuntimeError):
    """The synthetic OOM the --fake-cuda hook raises (its text is what
    ``is_oom`` keys on, so the ladder handles it exactly like the real one)."""

    def __init__(self, bs: int):
        super().__init__(f"CUDA out of memory (synthetic, --fake-cuda oom_at={bs})")


# ----------------------------------------------------------------------------
# examples: synthetic-LENGTH rows from the pool
# ----------------------------------------------------------------------------

def load_pool_texts(pairs: Path | None, n: int) -> tuple[list[str], str]:
    """The ``n`` LONGEST documents (prompt + trained text) of a pool jsonl,
    or ``n`` copies of the synthetic filler when no pool is given. Returns
    (texts, source)."""
    if pairs is None:
        return [_SYNTHETIC_DOC] * n, "synthetic (no --pairs)"
    pairs = Path(pairs)
    if not pairs.is_file():
        raise ProbeError(f"--pairs {pairs} is not a file")
    docs: list[str] = []
    with pairs.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            docs.append(str(row.get("prompt", "")) + str(row.get("pos", "")))
    if not docs:
        raise ProbeError(f"--pairs {pairs} holds no rows")
    docs.sort(key=len, reverse=True)
    out = [docs[i % len(docs)] for i in range(n)]
    return out, f"{pairs} (longest {min(n, len(docs))} of {len(docs)} rows)"


def make_examples(tok, texts: list[str], seq_len: int):
    """DistillExamples of exactly ``seq_len`` encoded tokens (BOS + text +
    EOS as the trainer counts them): each text is repeated past the cap,
    cut on token ids, decoded, and trimmed until its re-encoding fits —
    the trainer refuses any row over max_len, so the realized width is
    asserted here, never left to a silent truncation."""
    from sisteer.extraction.distill import DistillExample

    eos_n = 1 if tok.eos_token_id is not None else 0
    bos_n = len(tok("", add_special_tokens=True)["input_ids"])
    budget = seq_len - eos_n - bos_n
    if budget < 8:
        raise ProbeError(f"--seq-len {seq_len} leaves no room for text")
    exs, widths = [], []
    seen: dict[str, tuple] = {}          # identical texts (the filler; a wrapped pool) tokenise once
    for text in texts:
        if text in seen:
            ex, n_total = seen[text]
            exs.append(ex)
            widths.append(n_total)
            continue
        # repeat the text past the budget in ONE join (its token count sets the
        # repeat count; joins may re-tokenise across the seam, so one extra copy)
        n_text = max(1, len(tok(text, add_special_tokens=False)["input_ids"]))
        body = "\n\n".join([text] * (budget // n_text + 2))
        ids = tok(body, add_special_tokens=False)["input_ids"]
        while len(ids) < budget:
            body = body + "\n\n" + text
            ids = tok(body, add_special_tokens=False)["input_ids"]
        cut = tok.decode(ids[:budget])
        n = len(tok(cut, add_special_tokens=False)["input_ids"])
        tries = 0
        while n > budget and tries < 16:
            cut = tok.decode(tok(cut, add_special_tokens=False)["input_ids"][:budget - (n - budget)])
            n = len(tok(cut, add_special_tokens=False)["input_ids"])
            tries += 1
        if n > budget:
            raise ProbeError(f"could not trim a row to {budget} tokens (got {n})")
        ex = DistillExample(prompt_with_si="", prompt_without="",
                            completion=cut, instruction_idx=0)
        seen[text] = (ex, n + eos_n + bos_n)
        exs.append(ex)
        widths.append(n + eos_n + bos_n)
    return exs, widths


# ----------------------------------------------------------------------------
# model + object
# ----------------------------------------------------------------------------

def experts_implementation_of(model) -> str | None:
    cfg = getattr(model, "config", None)
    for c in (cfg, getattr(cfg, "text_config", None)):
        if c is None:
            continue
        for attr in ("experts_implementation", "_experts_implementation"):
            v = getattr(c, attr, None)
            if v:
                return str(v)
    return os.environ.get("TRANSFORMERS_EXPERTS_IMPLEMENTATION") or None


def enable_gradient_checkpointing(model) -> None:
    """train_two_si_objects.maybe_enable_gradient_checkpointing's exact
    mechanics (the fits' path): non-reentrant block checkpointing, use_cache
    off, train mode — refused on a model with nonzero dropout (train mode
    would change the forward math)."""
    if not getattr(model, "supports_gradient_checkpointing", False):
        raise ProbeError(f"{type(model).__name__} does not support gradient "
                         "checkpointing — the FD 23 probe measures the "
                         "checkpointed footprint")
    cfg = getattr(model, "config", None)
    confs = [cfg] if cfg is not None else []
    if cfg is not None and getattr(cfg, "text_config", None) is not None:
        confs.append(cfg.text_config)
    bad = {k: v for c in confs for k, v in vars(c).items()
           if k.endswith("dropout") and isinstance(v, (int, float)) and float(v) != 0.0}
    if bad:
        raise ProbeError(f"nonzero dropout {bad}: train-mode checkpointing "
                         "would change the forward math")
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False})
    for c in confs:
        c.use_cache = False
    model.train()


def build_probe_object(model, *, arm: str, lora_rank: int | None,
                       lora_modules: list[str] | None, init_scheme: str,
                       per_expert_rank=None, seed: int = 0):
    """The arm's UNTRAINED object through ``object_factory.build_object`` —
    the production builder (the arm spelling is one of its aliases; the
    LoRA axes are None for the ReLU arm, which the factory requires).
    ``per_expert_rank`` is the trainer's explicit map: the construction
    validates it against the loaded model, exactly as the fit does."""
    from sisteer.extraction.object_factory import build_object
    return (build_object(model, object_kind=arm, init_scheme=init_scheme, seed=seed,
                         lora_rank=lora_rank, lora_modules=lora_modules,
                         per_expert_rank=per_expert_rank),
            "object_factory.build_object")


# ----------------------------------------------------------------------------
# the ladder
# ----------------------------------------------------------------------------

def run_rung(model, tok, *, arm: str, bs: int, steps: int, seq_len: int,
             texts: list[str], object_kwargs: dict, fake: dict | None) -> dict:
    """One rung: build the object, train ``steps`` optimizer steps at
    micro-batch ``bs`` (accum 1) on ``bs * steps`` synthetic-length rows,
    catch an OOM. Memory numbers are the trainer LOOP's peak (the trainer
    resets the peak counter right before its first step)."""
    import torch

    device = next(model.parameters()).device
    cuda = device.type == "cuda"
    rec: dict = {"micro_batch": bs, "steps_requested": steps, "fits": None,
                 "peak_alloc_bytes": None, "peak_reserved_bytes": None,
                 "s_per_step": None, "tokens_per_s": None, "error": None,
                 "steps_completed": 0, "object_builder": None,
                 "realized_seq_len": None, "trainable_params": None}
    obj = None
    step_times: list[tuple[int, float]] = []

    def log_fn(payload, step):
        if any(str(k).endswith("/loss") for k in payload):
            step_times.append((int(step), time.perf_counter()))

    try:
        if fake is not None and bs in fake["oom_rungs"]:
            raise FakeOOM(bs)
        from sisteer.extraction.objects import train_object_ce

        exs, widths = make_examples(tok, texts[: bs * steps], seq_len)
        rec["realized_seq_len"] = {"min": min(widths), "max": max(widths)}
        obj, builder = build_probe_object(model, arm=arm, **object_kwargs)
        rec["object_builder"] = builder
        from sisteer.extraction.lora_modules import parameter_count
        rec["trainable_params"] = parameter_count(obj)
        if cuda:
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        t0 = time.perf_counter()
        train_object_ce(
            model, tok, obj, exs, lr=1e-5, weight_decay=0.01, epochs=1,
            batch_size=bs, max_len=seq_len, seed=0, log_fn=log_fn,
            log_tag=f"memprobe-{arm}", accum_steps=1,
            lr_schedule="warmup_cosine", max_grad_norm=1.0,
            reg_coeff=0.1 if arm == "psr_ce_all" else 0.0,
            max_steps=steps)
        if cuda:
            torch.cuda.synchronize(device)
        wall = time.perf_counter() - t0
        done = len(step_times) or steps
        rec["steps_completed"] = int(done)
        if len(step_times) >= 2:
            s_per_step = (step_times[-1][1] - step_times[0][1]) / (len(step_times) - 1)
        else:
            s_per_step = wall / max(done, 1)
        rec["s_per_step"] = float(s_per_step)
        rec["tokens_per_s"] = float(bs * seq_len / s_per_step) if s_per_step > 0 else None
        rec["wall_s"] = float(wall)
        rec["fits"] = True
    except Exception as e:  # noqa: BLE001 — the ladder classifies
        if is_oom(e):
            rec["fits"] = False
            rec["error"] = f"{type(e).__name__}: {str(e)[:400]}"
        else:
            raise
    finally:
        if cuda:
            rec["peak_alloc_bytes"] = int(torch.cuda.max_memory_allocated(device))
            rec["peak_reserved_bytes"] = int(torch.cuda.max_memory_reserved(device))
        elif fake is not None:
            # synthetic numbers, MARKED: the CPU test reads the shape, not the value
            rec["peak_alloc_bytes"] = int(bs * 10**9)
            rec["peak_reserved_bytes"] = int(bs * 11 * 10**8)
            rec["fake_cuda"] = True
        if obj is not None:
            del obj
        gc.collect()
        if cuda:
            torch.cuda.empty_cache()
    return rec


def run_ladder(model, tok, *, arm: str, ladder: list[int], steps: int,
               seq_len: int, texts: list[str], object_kwargs: dict,
               fake: dict | None, log=print) -> list[dict]:
    rungs: list[dict] = []
    for bs in ladder:
        log(f"[memprobe] rung micro-batch {bs}: {steps} step(s) x {bs} x {seq_len} tokens")
        rec = run_rung(model, tok, arm=arm, bs=bs, steps=steps, seq_len=seq_len,
                       texts=texts, object_kwargs=object_kwargs, fake=fake)
        rungs.append(rec)
        if rec["fits"]:
            log(f"[memprobe]   FITS  peak_alloc={_gib(rec['peak_alloc_bytes'])} "
                f"peak_reserved={_gib(rec['peak_reserved_bytes'])} "
                f"s/step={rec['s_per_step']:.2f} tok/s={rec['tokens_per_s']:.0f}")
        else:
            log(f"[memprobe]   DOES NOT FIT ({rec['error']}) — the ladder stops here")
            for rest in ladder[ladder.index(bs) + 1:]:
                rungs.append({"micro_batch": rest, "fits": None,
                              "error": "not attempted: a narrower rung did not fit",
                              "peak_alloc_bytes": None, "peak_reserved_bytes": None,
                              "s_per_step": None, "tokens_per_s": None,
                              "steps_requested": steps, "steps_completed": 0})
            break
    return rungs


def _gib(n) -> str:
    return "-" if n is None else f"{n / 2**30:.1f} GiB"


# ----------------------------------------------------------------------------
# the record + the FD 23 report
# ----------------------------------------------------------------------------

def verdict(rungs: list[dict], ladder: list[int]) -> dict:
    first = next((r for r in rungs if r["micro_batch"] == ladder[0]), None)
    fits_at_min = bool(first and first["fits"])
    largest = max((r["micro_batch"] for r in rungs if r["fits"]), default=None)
    s1 = first["s_per_step"] if first and first["fits"] else None
    replan = (s1 is not None and s1 > REPLAN_S_PER_STEP)
    watch = (s1 is not None and REPLAN_S_PER_STEP_WATCH < s1 <= REPLAN_S_PER_STEP)
    return {"fits_at_min_micro_batch": fits_at_min, "min_micro_batch": ladder[0],
            "largest_fitting_micro_batch": largest,
            "s_per_step_at_min": s1,
            "replan_threshold_s_per_step": REPLAN_S_PER_STEP,
            "replan_required": bool(replan), "replan_watch": bool(watch),
            "exit_code": EXIT_FITS if fits_at_min else EXIT_DOES_NOT_FIT}


def render_report(record: dict) -> str:
    v = record["verdict"]
    m = record["model"]
    lines = [
        f"# FD 23 fit-memory probe — {m['model']} / {record['arm']}",
        "",
        f"- model: `{m['model']}` @ `{m['revision']}`",
        f"- arm: `{record['arm']}` (object: {record['object_kwargs']})",
        f"- seq_len {record['seq_len']} tokens, gradient checkpointing "
        f"{'ON' if record['gradient_checkpointing'] else 'OFF'}, dtype {record['dtype']}, "
        f"{record['steps']} optimizer step(s) per rung, accum 1",
        f"- GPU: {record['gpu_name']}; torch {record['torch']}, transformers "
        f"{record['transformers']}; experts implementation: {record['experts_implementation']}",
        f"- examples: {record['examples_source']}",
        "",
        "| micro-batch | fits | peak allocated | peak reserved | s/step | tokens/s | note |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in record["rungs"]:
        fits = {True: "YES", False: "NO", None: "-"}[r["fits"]]
        sps = f"{r['s_per_step']:.2f}" if r.get("s_per_step") else "-"
        tps = f"{r['tokens_per_s']:.0f}" if r.get("tokens_per_s") else "-"
        lines.append(f"| {r['micro_batch']} | {fits} | {_gib(r.get('peak_alloc_bytes'))} | "
                     f"{_gib(r.get('peak_reserved_bytes'))} | {sps} | {tps} | "
                     f"{(r.get('error') or '').replace('|', '/')} |")
    lines.append("")
    if v["fits_at_min_micro_batch"]:
        lines.append(f"**VERDICT: micro-batch {v['min_micro_batch']} FITS** "
                     f"(largest fitting rung: {v['largest_fitting_micro_batch']}). "
                     "The fit lane may proceed (FD 23).")
    else:
        lines.append(f"**VERDICT: micro-batch {v['min_micro_batch']} DOES NOT FIT** — "
                     "FD 23: the user is told at once and the fit lane waits for their call; "
                     "everything else proceeds.")
    s1 = v["s_per_step_at_min"]
    if s1 is not None:
        # one constitution epoch (42.8 M tokens) at the measured pace, in hours
        epoch_h = CONSTITUTION_EPOCH_TOKENS / (v["min_micro_batch"] * record["seq_len"]) * s1 / 3600
        if v["replan_required"]:
            lines.append(f"**REPLAN REQUIRED**: {s1:.2f} s/step at micro-batch "
                         f"{v['min_micro_batch']} exceeds the plan §6 threshold "
                         f"(~{REPLAN_S_PER_STEP:.0f} s/step): a 42.8 M-token constitution "
                         f"epoch is ~{epoch_h:.1f} h "
                         "at this pace — replan the fit schedule (grouped-GEMM experts path, "
                         "early stop, more parallel fits) before submitting.")
        elif v["replan_watch"]:
            lines.append(f"Replan WATCH: {s1:.2f} s/step at micro-batch {v['min_micro_batch']} "
                         f"sits in the ~1-2 s/step band; one constitution epoch is "
                         f"~{epoch_h:.1f} h.")
        else:
            lines.append(f"Replan not triggered: {s1:.2f} s/step at micro-batch "
                         f"{v['min_micro_batch']} (threshold ~{REPLAN_S_PER_STEP:.0f} s/step).")
    if record.get("fake_cuda"):
        lines.append("")
        lines.append("_synthetic run (--fake-cuda): memory numbers are placeholders._")
    return "\n".join(lines) + "\n"


def load_config(path: Path) -> dict:
    import yaml
    cfg = yaml.safe_load(Path(path).read_text())
    for k in ("model", "model_revision"):
        if k not in cfg:
            raise ProbeError(f"{path}: config carries no {k!r}")
    return cfg


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True, type=Path,
                    help="the fits config (configs/sdfbase_<key>_fits.yaml)")
    ap.add_argument("--arm", required=True, choices=ARMS)
    ap.add_argument("--pairs", type=Path, default=None,
                    help="pool jsonl whose LONGEST docs seed the synthetic-length rows "
                         "(default: a synthetic filler document)")
    ap.add_argument("--out", required=True, type=Path, help="memprobe.json")
    ap.add_argument("--report", type=Path, default=None,
                    help="memprobe_report.md (default: beside --out)")
    ap.add_argument("--ladder", default=",".join(str(x) for x in DEFAULT_LADDER))
    ap.add_argument("--seq-len", type=int, default=None,
                    help="probe sequence length (default: the config's max_len / sequence_max_len — the length the fit trains at; "
                         "a different value is refused unless --allow-seq-len-override, and the fold keys the record to the config's)")
    ap.add_argument("--allow-seq-len-override", action="store_true",
                    help="acknowledge a --seq-len that differs from the config's (an ad-hoc probe; the fold will read it STALE)")
    ap.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    ap.add_argument("--lora-rank", type=int, default=None,
                    help="lora arm: the rank (default: the config's "
                         "train_kwargs.lora_r4_ce.lora_rank, else 4)")
    ap.add_argument("--lora-modules", default=None,
                    help="lora arm: comma list of module short names (default: the "
                         "config's fit_selection.lora_modules, else o_proj)")
    ap.add_argument("--init-scheme", default=None, choices=("zero", "random_lecun_1e-3"),
                    help="object init (default: the config's fit_selection.init_scheme, else zero)")
    ap.add_argument("--gradient-checkpointing", choices=("on", "off"), default=None,
                    help="default: what the fit runs with — train_kwargs.<arm>.gradient_checkpointing over train_kwargs.default's, default off "
                         "(the trainer's rule); a differing value is refused unless --allow-runtime-override")
    ap.add_argument("--dtype", default=None, choices=("bfloat16", "float32"),
                    help="weights dtype (default: the config's dtype — what the fit trains in; a differing value is refused unless "
                         "--allow-runtime-override)")
    ap.add_argument("--allow-runtime-override", action="store_true",
                    help="acknowledge a --gradient-checkpointing / --dtype that differs from the fit's (an ad-hoc probe; the fold reads it STALE)")
    ap.add_argument("--model-override", default=None,
                    help="load THIS model id / snapshot dir instead of the config's "
                         "(tests; recorded as such)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--fake-cuda", default=None, metavar="SPEC",
                    help="test hook: oom_at=<rung[,rung]> raises a synthetic OOM at those "
                         "rungs; the loop runs on CPU with synthetic memory numbers")
    args = ap.parse_args(argv)

    try:
        ladder = parse_ladder(args.ladder)
        fake = parse_fake_cuda(args.fake_cuda)
        cfg = load_config(args.config)
        if args.steps < 1 or (args.seq_len is not None and args.seq_len < 16):
            raise ProbeError("--steps >= 1 and --seq-len >= 16")
    except ProbeError as e:
        print(f"[memprobe] {e}", file=sys.stderr)
        return EXIT_USAGE

    # ONE source with the fold and the fit (memprobe-fix-2): the trainer-facing selection — top-level train_kwargs.<arm> first,
    # fit_selection filling what it leaves unstated, the paper's defaults last; the explicit per_expert_rank map rides along and is
    # RECORDED, so the fold's identity check reads what was measured. CLI overrides are an ad-hoc probe's business and are recorded too.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from memprobe_fold import config_runtime, config_selection, config_seq_len
    chosen = config_selection(cfg, args.arm)
    lora_rank = lora_modules = per_expert_rank = None
    if args.arm == "lora_r4_ce":
        lora_rank = int(args.lora_rank) if args.lora_rank is not None else chosen["lora_rank"]
        lora_modules = (sorted({m.strip() for m in args.lora_modules.split(",") if m.strip()}) if args.lora_modules
                        else chosen["lora_modules"])
        per_expert_rank = chosen["per_expert_rank"] if (args.lora_rank is None and args.lora_modules is None) else None
    init_scheme = args.init_scheme or chosen["init_scheme"]
    object_kwargs = {"lora_rank": lora_rank, "lora_modules": lora_modules,
                     "per_expert_rank": per_expert_rank, "init_scheme": init_scheme}
    cfg_seq = config_seq_len(cfg)
    if args.seq_len is None:
        args.seq_len = cfg_seq
    if args.seq_len < 16:
        print("[memprobe] --seq-len >= 16", file=sys.stderr)
        return EXIT_USAGE
    elif args.seq_len != cfg_seq and not args.allow_seq_len_override:
        print(f"[memprobe] --seq-len {args.seq_len} differs from the config's max_len {cfg_seq} — the fit trains at {cfg_seq} and the "
              "fold keys the record to it; pass --allow-seq-len-override for an ad-hoc probe (it will read STALE at fold time)",
              file=sys.stderr)
        return EXIT_USAGE
    # the runtime the FIT uses (memprobe-fix-3): the trainer's rule for checkpointing, the config's dtype — the probe
    # measures under the same, or says it does not (recorded; the fold then reads it STALE)
    rt = config_runtime(cfg, args.arm)
    gc_on = rt["gradient_checkpointing"] if args.gradient_checkpointing is None else (args.gradient_checkpointing == "on")
    dtype = args.dtype or rt["dtype"]
    if (gc_on != rt["gradient_checkpointing"] or dtype != rt["dtype"]) and not args.allow_runtime_override:
        print(f"[memprobe] gradient checkpointing {gc_on} / dtype {dtype} differ from what the fit runs with "
              f"({rt['gradient_checkpointing']} / {rt['dtype']}) — pass --allow-runtime-override for an ad-hoc probe (it will read STALE at fold time)",
              file=sys.stderr)
        return EXIT_USAGE
    model_id = args.model_override or cfg["model"]
    revision = None if args.model_override else cfg.get("model_revision")
    report_path = args.report or (args.out.parent / "memprobe_report.md")
    plan = {"model": model_id, "revision": revision, "arm": args.arm,
            "ladder": ladder, "seq_len": args.seq_len, "steps": args.steps,
            "gradient_checkpointing": gc_on, "dtype": dtype,
            "object_kwargs": object_kwargs, "pairs": str(args.pairs) if args.pairs else None,
            "out": str(args.out), "report": str(report_path),
            "fake_cuda": sorted(fake["oom_rungs"]) if fake else None}
    print("[memprobe] plan: " + json.dumps(plan, sort_keys=True), flush=True)
    if args.dry_run:
        return EXIT_FITS

    import torch
    import transformers

    from sisteer.models import load_model_and_tokenizer

    cuda = torch.cuda.is_available()
    if not cuda and fake is None:
        print("[memprobe] no CUDA device and no --fake-cuda: the probe measures GPU "
              "memory; refusing to report CPU numbers as a fit verdict", file=sys.stderr)
        return EXIT_USAGE
    try:
        texts, source = load_pool_texts(args.pairs, max(ladder) * args.steps)
    except ProbeError as e:
        print(f"[memprobe] {e}", file=sys.stderr)
        return EXIT_USAGE
    t_load = time.perf_counter()
    model, tok = load_model_and_tokenizer(model_id, revision=revision, dtype=dtype,
                                          attn_implementation="sdpa")
    load_s = time.perf_counter() - t_load
    if gc_on:
        try:
            enable_gradient_checkpointing(model)
        except ProbeError as e:
            print(f"[memprobe] {e}", file=sys.stderr)
            return EXIT_USAGE
    gpu_name = torch.cuda.get_device_name(0) if cuda else f"cpu ({platform.processor() or platform.machine()})"
    rungs = run_ladder(model, tok, arm=args.arm, ladder=ladder, steps=args.steps,
                       seq_len=args.seq_len, texts=texts, object_kwargs=object_kwargs,
                       fake=fake)
    record = {
        "schema": 1, "kind": "memprobe_fit",
        "model": {"model": model_id, "revision": revision,
                  "config": str(args.config), "model_override": bool(args.model_override),
                  "model_class": type(model).__name__},
        "arm": args.arm, "object_kwargs": object_kwargs,
        "seq_len": args.seq_len, "steps": args.steps, "ladder": ladder,
        "gradient_checkpointing": gc_on,
        "dtype": dtype, "gpu_name": gpu_name,
        "gpu_total_bytes": int(torch.cuda.get_device_properties(0).total_memory) if cuda else None,
        "torch": torch.__version__, "transformers": transformers.__version__,
        "experts_implementation": experts_implementation_of(model),
        "examples_source": source, "model_load_s": float(load_s),
        "fake_cuda": bool(fake), "hostname": platform.node(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "rungs": rungs,
    }
    record["verdict"] = verdict(rungs, ladder)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    Path(report_path).write_text(render_report(record))
    v = record["verdict"]
    print(f"[memprobe] VERDICT fits_at_micro_batch_{ladder[0]}={v['fits_at_min_micro_batch']} "
          f"largest_fitting={v['largest_fitting_micro_batch']} "
          f"s_per_step_at_min={v['s_per_step_at_min']} replan_required={v['replan_required']} "
          f"-> {args.out} ; report {report_path}", flush=True)
    return int(v["exit_code"])


if __name__ == "__main__":
    sys.exit(main())
