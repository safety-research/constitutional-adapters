"""Peak-memory probe for one row-sharded training step at the wave's worst
case (accel-fit 2026-08-21): does a rank holding ITS ROW SLICE of a max_len
8192 batch fit this GPU?

Why this exists: the measured per-arm batch sizes (twosi budget table) were
calibrated on 141GB H200s with the WHOLE batch on one GPU. Row-sharding puts
``bs/width`` rows on each rank, so smaller-memory GPUs (80GB H100) become
candidates — IF the single-slice with-grad step fits. This probe answers
that with a real step, not an estimate: synthetic tokens (never store data),
the production model pin, the real KL-loss shape (teacher no-grad forward +
student forward + fp32 log-softmax over completion positions + backward to a
trainable object), at slice sizes 1 and 2.

Prints one MEMPROBE line per case and exits nonzero on OOM — a loud verdict
either way.
"""

from __future__ import annotations

import sys

import torch
import yaml
from torch import nn


def one_step(model, hidden: int, rows: int, seq_len: int, comp: int) -> float:
    device = next(model.parameters()).device
    vec = nn.Parameter(torch.zeros(hidden, device=device))
    opt = torch.optim.AdamW([vec], lr=5e-3)
    ids = torch.randint(10, 1000, (rows, seq_len), device=device)
    mask = torch.ones_like(ids)

    from sisteer.hooks import steering

    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        t_logits = model(input_ids=ids, attention_mask=mask
                         ).logits[:, -comp:, :]
    with steering(model, {15: vec}, positions="all", skip_bos=True):
        s_logits = model(input_ids=ids, attention_mask=mask
                         ).logits[:, -comp:, :]
    t_logp = torch.log_softmax(
        t_logits.float().reshape(-1, t_logits.shape[-1]), -1)
    s_logp = torch.log_softmax(
        s_logits.float().reshape(-1, s_logits.shape[-1]), -1)
    loss = torch.nn.functional.kl_div(s_logp, t_logp, log_target=True,
                                      reduction="batchmean")
    opt.zero_grad()
    loss.backward()
    opt.step()
    peak = torch.cuda.max_memory_allocated() / 2**30
    del t_logits, s_logits, t_logp, s_logp, loss
    torch.cuda.empty_cache()
    return peak


def main() -> None:
    cfg = yaml.safe_load(open("configs/two_si.yaml"))
    from sisteer.models import hidden_size, load_model_and_tokenizer

    model, _tok = load_model_and_tokenizer(
        cfg["model"], revision=cfg.get("model_revision"), dtype=cfg["dtype"],
        attn_implementation=cfg.get("attn_implementation", "sdpa"))
    total = torch.cuda.get_device_properties(0).total_memory / 2**30
    name = torch.cuda.get_device_name(0)
    seq_len, comp = 8192, 512  # the two big SIs' max_len + completion cap
    failures = 0
    for rows in (1, 2):
        try:
            peak = one_step(model, hidden_size(model), rows, seq_len, comp)
            print(f"MEMPROBE gpu={name!r} total={total:.0f}GiB rows={rows} "
                  f"seq={seq_len} comp={comp}: peak={peak:.1f}GiB "
                  f"headroom={total - peak:.1f}GiB OK")
        except torch.cuda.OutOfMemoryError as e:
            failures += 1
            print(f"MEMPROBE gpu={name!r} total={total:.0f}GiB rows={rows} "
                  f"seq={seq_len}: OOM — {str(e)[:120]}")
            torch.cuda.empty_cache()
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
