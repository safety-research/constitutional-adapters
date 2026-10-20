"""Per-device batch ceiling probe for train_bidpo — doubling to OOM.

The tuning playbook's per-device rule (house heuristic 5): the micro-batch
is chosen for throughput, as large as fits, found by DOUBLING a short probe
until it OOMs and stepping back — never by a full launch, and never by
arithmetic on activation sizes (a "both"-direction DPO step holds FOUR
differentiable forward graphs — chosen/rejected x +-d — before its
backward, plus logits, and the analytic estimate has been wrong before).
Memory peaks at the FIRST optimizer step, so one step per width is the
whole measurement.

Synthetic worst-case rows: every sequence is exactly ``--max-len`` tokens
(collate pads to the batch max, so a real pool's ragged batch is never
wider than this). Invented text only — no pool data, no prompts.

Run on a dev GPU (an interactive srun at the dev QoS), e.g.:

    python scripts/bidpo_mem_probe.py --config <two_si yaml> \
        --max-len 768 --widths 2,4,8,16,32,64

Prints one machine-readable line per width:

    PROBE width=8 status=ok peak_alloc_gib=41.2 peak_reserved_gib=44.0 secs=12.3

and a final ``PROBE_MAX_OK width=<n>`` verdict. OOM at a width is an
expected outcome, not an error; the process continues to report it and
stops there (wider only OOMs harder).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from sisteer.extraction.bidpo import PrefExample, train_bidpo  # noqa: E402
from sisteer.models import load_model_and_tokenizer  # noqa: E402


def _full_len_examples(tok, n: int, max_len: int) -> list[PrefExample]:
    """n synthetic pairs whose every encoded side is exactly max_len."""
    # ~6 chars/token is plenty: encode_prompt_completion truncates at
    # max_len, so overshooting costs nothing.
    filler = "probe filler text " * (max_len * 2)
    return [
        PrefExample(prompt=f"synthetic probe prompt {i}\n",
                    chosen=filler, rejected=filler)
        for i in range(n)
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True,
                    help="two_si yaml — model/dtype are read from it so the "
                         "probe measures the exact stack the fit will run")
    ap.add_argument("--max-len", type=int, default=768)
    ap.add_argument("--widths", default="2,4,8,16,32,64",
                    help="comma-separated micro-batch widths, probed in "
                         "order, stopping at the first OOM")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    model, tok = load_model_and_tokenizer(
        cfg["model"], revision=cfg.get("model_revision"),
        dtype=cfg["dtype"],
        attn_implementation=cfg.get("attn_implementation", "sdpa"))
    device = next(model.parameters()).device
    if device.type != "cuda":
        raise SystemExit("PROBE_ERROR no CUDA device — the ceiling of a "
                         "CPU run means nothing for the sweep")

    max_ok = None
    for width in [int(w) for w in args.widths.split(",") if w]:
        examples = _full_len_examples(tok, width, args.max_len)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        t0 = time.time()
        try:
            # One epoch over ONE batch of `width` rows = one optimizer
            # step holding the full four-graph peak. Real fit settings
            # (direction="both", wd 2.0, Class A all-token leg).
            train_bidpo(model, tok, examples, layer=int(cfg["layer"]),
                        epochs=1, batch_size=width, max_len=args.max_len,
                        weight_decay=2.0, direction="both")
            status = "ok"
            max_ok = width
        except torch.OutOfMemoryError:
            status = "oom"
        secs = time.time() - t0
        alloc = torch.cuda.max_memory_allocated(device) / 2**30
        reserved = torch.cuda.max_memory_reserved(device) / 2**30
        print(f"PROBE width={width} status={status} "
              f"peak_alloc_gib={alloc:.1f} peak_reserved_gib={reserved:.1f} "
              f"secs={secs:.1f}", flush=True)
        if status == "oom":
            break

    if max_ok is None:
        print("PROBE_MAX_OK width=0 (nothing fit — smaller widths or "
              "shorter sequences needed)", flush=True)
        raise SystemExit(1)
    print(f"PROBE_MAX_OK width={max_ok}", flush=True)


if __name__ == "__main__":
    main()
