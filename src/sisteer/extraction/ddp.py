"""Row-sharded data parallelism for the fit pipeline (accel-fit, 2026-08-21).

The stage-3 pools are ~23k pairs per SI at max_len up to 8192 — a single-GPU
Class A fit is 27-36h per SI. This module is the machinery that drives that
to hours without changing WHAT is computed:

1. :class:`FitParallelCtx` — the distributed context for ONE arm's fit
   task. The per-SI driver schedules each arm as its own process group
   (``torchrun`` per arm, on a strict subset of the job's GPUs); every rank
   holds the arm's full batch list and forwards only its ROW SLICE of each
   batch.
2. Exact-semantics helpers: row slicing, component-weighted local-loss
   scaling, gradient all-reduce, replica-equality assertion, and ordered
   sharded mapping for no-grad collection passes (activation captures,
   rollouts, Fisher harvests).

THE CONTRACT — "row-sharded" means EXACT single-GPU semantics:

- Same batches, same batch composition, same shuffled order, same number
  of optimizer steps, same effective batch size, same learning rate. A
  k-rank fit computes, at every step, the same loss gradient as the
  single-GPU fit up to floating-point reduction order (the all-reduce sums
  per-rank partial gradients in an order the single sum does not fix).
  Nothing about the OPTIMIZATION changes — this is deliberately NOT
  batch-scaling data parallelism, which would multiply the effective
  batch by the world size and reopen the learning-rate question.
- Every rank steps its own optimizer copy on the identical all-reduced
  gradient, so replicas stay bit-identical; :func:`assert_replicas_equal`
  makes a divergence LOUD at the end of training instead of a silent
  wrong artifact.
- Weighting: a loss of the form ``mean-over-N-elements`` recombines
  exactly as ``sum_r (n_r / N) * loss_r`` over ranks. Losses with several
  differently-normalized components (divergence over completion tokens +
  a gate regularizer over batch rows) scale each component by ITS OWN
  count ratio — :func:`scale_loss_parts`. The counts are all-reduced
  before the backward (:func:`allreduce_counts`), so N is the true global
  count even on ragged final batches where some ranks hold fewer rows.

The width an arm can use:

- gradient loops: at most the arm's batch size (rows per step is the unit
  of sharding) — resolve with :func:`rowshard_width`;
- no-grad collection passes (closed-form arms): unbounded (sharded over
  the example list, gathered in order) — :func:`shard_items` /
  :func:`gather_sharded`.

With no initialized process group every helper is a no-op passthrough and
every trainer runs its historical single-process path byte-identically.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from datetime import timedelta

import torch

# --------------------------------------------------------------------------- #
# Context                                                                       #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class FitParallelCtx:
    """The distributed context of one arm's fit task (one process group)."""

    rank: int
    world_size: int
    device: torch.device

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    @property
    def enabled(self) -> bool:
        return self.world_size > 1


_ACTIVE: FitParallelCtx | None = None


def active() -> FitParallelCtx | None:
    """The process's ambient fit-parallel context, or None (single-process).

    Ambient on purpose: the no-grad collection passes (capture.py,
    alphakl.py) are called from deep inside registry trainers whose
    signatures are pinned by the registry contract — threading a ctx
    parameter through every trainer would churn every caller for a purely
    operational concern. The arm-task entrypoint sets this once, right
    after ``init_process_group``; eval-time callers never set it and see
    unchanged behavior."""
    return _ACTIVE


def set_active(ctx: FitParallelCtx | None) -> None:
    global _ACTIVE
    _ACTIVE = ctx


def setup_from_env(timeout_hours: float = 40.0) -> FitParallelCtx | None:
    """Initialize the process group from torchrun's env, or return None
    when not under torchrun (WORLD_SIZE absent or 1 — the single-process
    path). Sets the ambient context. NCCL on CUDA, gloo on CPU (tests).

    The generous timeout is deliberate: an arm's ranks synchronize every
    step, but guard passes (dead-gate probes) run on rank 0 only while the
    other ranks wait at the final assert/barrier."""
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world <= 1:
        return None
    import torch.distributed as dist

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
        backend = "nccl"
    else:
        device = torch.device("cpu")
        backend = "gloo"
    if not dist.is_initialized():
        dist.init_process_group(
            backend=backend, timeout=timedelta(hours=timeout_hours))
    ctx = FitParallelCtx(rank=rank, world_size=world, device=device)
    set_active(ctx)
    return ctx


# --------------------------------------------------------------------------- #
# Row sharding for gradient loops                                              #
# --------------------------------------------------------------------------- #


def rowshard_width(batch_size: int, requested: int) -> int:
    """The usable world size for a gradient loop: rows per step is the unit
    of sharding, so width beyond the batch size only idles ranks."""
    return max(1, min(int(requested), int(batch_size)))


def rows_for_rank(n_rows: int, ctx: FitParallelCtx | None) -> list[int]:
    """This rank's row indices of an ``n_rows``-row batch — strided
    (``rank::world_size``), so the assignment is deterministic and every
    row belongs to exactly one rank. May be empty on ragged final
    batches; the empty rank still participates in every collective."""
    if ctx is None or not ctx.enabled:
        return list(range(n_rows))
    return list(range(ctx.rank, n_rows, ctx.world_size))


def slice_batch(batch, rows: list[int]):
    """Row-slice a ``traindata.Batch`` (any object with input_ids/
    attention_mask/prompt_lens/labels tensors on dim 0). Returns the same
    type. An EMPTY rows list is refused — forward nothing instead."""
    if not rows:
        raise ValueError("slice_batch got no rows — skip the forward instead")
    idx = torch.tensor(rows, dtype=torch.long,
                       device=batch.input_ids.device)
    return type(batch)(
        batch.input_ids[idx],
        batch.attention_mask[idx],
        batch.prompt_lens[idx],
        batch.labels[idx],
    )


def allreduce_vec(vals: list[float], ctx: FitParallelCtx | None) -> list[float]:
    """Sum a small vector of floats across the arm's ranks (fp64, so count
    sums stay exact). Identity when single-process."""
    if ctx is None or not ctx.enabled:
        return [float(v) for v in vals]
    import torch.distributed as dist

    t = torch.tensor([float(v) for v in vals], dtype=torch.float64,
                     device=ctx.device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t.tolist()


def allreduce_counts(counts: list[int], ctx: FitParallelCtx | None) -> list[int]:
    """Global (summed) counts across the arm's ranks. Identity when
    single-process. Used for the loss-component denominators, so ragged
    final batches weight exactly."""
    return [int(round(v)) for v in allreduce_vec([float(c) for c in counts],
                                                 ctx)]


def scale_loss_parts(parts: list[tuple[torch.Tensor | None, int, int]],
                     device) -> torch.Tensor:
    """Recombine per-rank loss components into this rank's SHARE of the
    exact global loss: ``sum_i part_i * (n_local_i / n_global_i)``.

    Each entry is ``(local_component, local_count, global_count)`` where
    the component is a mean over ``local_count`` elements and the global
    fit's component is the mean over ``global_count``. A rank with no
    rows for a component passes ``(None, 0, N)`` and contributes zero —
    summing every rank's share then reproduces the single-GPU loss
    exactly (up to fp reduction order). A data-INDEPENDENT term (e.g.
    lofit's L1 penalty) rides with weight ``n/N`` of its sibling term so
    the shares still sum to one copy of it — pass it pre-multiplied into
    the same component."""
    total = torch.zeros((), device=device)
    for comp, n_local, n_global in parts:
        if comp is None or n_local == 0:
            continue
        if n_global <= 0:
            raise ValueError("scale_loss_parts got a zero global count for a "
                             "non-empty local component")
        w = float(n_local) / float(n_global)
        total = total + comp * w
    return total


def allreduce_grads_and_stats(params, stats: list[float],
                              ctx: FitParallelCtx | None) -> list[float]:
    """Sum gradients (and a small stats vector, e.g. ``[loss*k, k]``)
    across the arm's ranks in ONE flat all-reduce. After this every rank
    holds the identical global gradient, so identical optimizer steps keep
    replicas bit-equal. ``p.grad is None`` (a rank that forwarded nothing)
    contributes zeros. Returns the summed stats; identity when
    single-process."""
    if ctx is None or not ctx.enabled:
        return list(stats)
    import torch.distributed as dist

    grads = [(p.grad if p.grad is not None else torch.zeros_like(p))
             for p in params]
    pieces = [g.detach().reshape(-1).to(device=ctx.device,
                                        dtype=torch.float32)
              for g in grads]
    pieces.append(torch.tensor(stats, dtype=torch.float32,
                               device=ctx.device))
    flat = torch.cat(pieces)
    dist.all_reduce(flat, op=dist.ReduceOp.SUM)
    off = 0
    for p, g in zip(params, grads):
        n = g.numel()
        new = flat[off:off + n].reshape(p.shape).to(p.dtype)
        if p.grad is None:
            p.grad = new.to(p.device)
        else:
            p.grad.copy_(new.to(p.device))
        off += n
    return [float(v) for v in flat[off:].tolist()]


def assert_replicas_equal(tag: str, params, ctx: FitParallelCtx | None) -> None:
    """Raise loudly if the arm's parameter replicas diverged across ranks —
    a bug turned into an error instead of a silently wrong artifact. Cheap
    (one hash + one small gather); call after training, before saving."""
    if ctx is None or not ctx.enabled:
        return
    import torch.distributed as dist

    h = hashlib.sha256()
    for p in params:
        h.update(p.detach().float().cpu().numpy().tobytes())
    digest = h.hexdigest()
    gathered: list[str | None] = [None] * ctx.world_size
    dist.all_gather_object(gathered, digest)
    if len(set(gathered)) != 1:
        raise RuntimeError(
            f"[{tag}] parameter replicas DIVERGED across ranks: "
            f"{ {r: g[:12] for r, g in enumerate(gathered)} } — the "
            "row-sharded fit is unsound; refusing to save. (Identical "
            "all-reduced gradients must produce identical AdamW steps; a "
            "divergence means nondeterministic kernels or a missed "
            "collective.)")


def barrier(ctx: FitParallelCtx | None) -> None:
    if ctx is None or not ctx.enabled:
        return
    import torch.distributed as dist

    dist.barrier()


# --------------------------------------------------------------------------- #
# Ordered sharded collection (no-grad passes; closed-form arms)                 #
# --------------------------------------------------------------------------- #


def shard_items(items: list, ctx: FitParallelCtx | None) -> list:
    """This rank's strided slice of a work list (``rank::world_size``).
    Identity when single-process."""
    if ctx is None or not ctx.enabled:
        return list(items)
    return list(items[ctx.rank::ctx.world_size])


def gather_sharded(local_results: list, n_total: int,
                   ctx: FitParallelCtx | None) -> list:
    """Reassemble strided-sharded per-item results into original order.
    ``local_results[j]`` must correspond to item ``rank + j*world_size``.
    Every rank returns the SAME full ordered list (the closed-form solves
    downstream run replicated on identical inputs, so no broadcast step
    can drift). Identity when single-process."""
    if ctx is None or not ctx.enabled:
        if len(local_results) != n_total:
            raise ValueError(
                f"gather_sharded: {len(local_results)} results for "
                f"{n_total} items with no parallel context")
        return list(local_results)
    import torch.distributed as dist

    per_rank: list[list | None] = [None] * ctx.world_size
    dist.all_gather_object(per_rank, list(local_results))
    out: list = [None] * n_total
    for r, chunk in enumerate(per_rank):
        for j, res in enumerate(chunk):
            i = r + j * ctx.world_size
            if i >= n_total:
                raise ValueError(
                    f"gather_sharded: rank {r} returned {len(chunk)} results "
                    f"but only {(n_total - r + ctx.world_size - 1) // ctx.world_size} "
                    f"items were its share of {n_total}")
            out[i] = res
    missing = [i for i, v in enumerate(out) if v is None]
    if missing:
        raise ValueError(
            f"gather_sharded: items {missing[:5]}{'...' if len(missing) > 5 else ''} "
            f"of {n_total} came back from no rank — a rank dropped work")
    return out
