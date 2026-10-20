"""The on-policyness ladder for distill_soft (the vector arm only).

Three rungs, same object (single-layer residual vector), same forward-KL
objective — only the SOURCE of the trajectories the KL is taken on moves
toward the student's own state distribution:

- rung 1 — greedy teacher: distill.train_distill(loss="soft") on
  greedy-decoded teacher completions (exists; stage 1's recipe).
- rung 2 — sampled-teacher MC-KL (Prompt Baking Eq. 3): the teacher
  SAMPLES N completions per prompt at T=1; forward KL on those examples
  (train_mc_kl below; sample_teacher_pairs builds one sample set).
- rung 3 — GKD (Agarwal et al.): each epoch the STUDENT (current vector
  active, no SI) samples fresh completions on the fit prompts; the
  teacher's distributions are computed on (SI + prompt + student
  completion); forward KL teacher -> student on those positions. Same
  weights, different state distribution. No gradient flows through the
  sampling itself.

BUDGET MATCH: rungs 2 and 3 must differ ONLY in trajectory SOURCE
(teacher-sampled vs student-rolled-out), never in data diversity
or compute. Both trainers therefore, per epoch, draw a FRESH set of
n_samples trajectories per prompt and take the same optimizer-update
count (same examples-per-epoch, same batch_size, same epochs, shared
_kl_epoch loop). A rung-2 set sampled ONCE and reused across epochs
would confound source with diversity — that was the pre-fix behavior.
Each trainer records n_trajectories_seen and n_optimizer_steps in the
returned vector's metadata (-> results.json) and logs budget/* through
log_fn (-> W&B), so the match is verifiable per run.
"""

from __future__ import annotations

import random

import torch
import torch.nn.functional as F
from torch import nn

from ..data.instructions import Instruction
from ..data.pairs import Pair, pairs_to_distill_examples
from ..generate import generate_batch
from ..hooks import steering
from ..models import build_chat_prompt, hidden_size
from ..vectors import SteeringVector
from .distill import DistillExample, _encode_aligned, _gather_completion_logits
from .traindata import Batch, collate


def sample_teacher_pairs(
    model,
    tok,
    instruction: Instruction,
    prompts: list[str],
    n_samples: int = 3,
    temperature: float = 1.0,
    batch_size: int = 8,
    max_new_tokens: int = 200,
    seed: int = 0,
) -> list[Pair]:
    """Rung-2 data: N sampled teacher completions per prompt at T
    (Prompt Baking Eq. 3's Monte-Carlo KL). Returns Pair rows compatible
    with save_pairs / pairs_to_distill_examples; `neg` is left empty —
    the MC-KL rung trains on the positives only."""
    rendered = [build_chat_prompt(tok, p, system=instruction.text)
                for p in prompts]
    torch.manual_seed(seed)
    pairs: list[Pair] = []
    for _ in range(n_samples):
        outs = generate_batch(
            model, tok, rendered, batch_size=batch_size,
            max_new_tokens=max_new_tokens, do_sample=True,
            temperature=temperature,
        )
        pairs.extend(
            Pair(prompt=p, pos=o, neg="", instruction_id=instruction.id)
            for p, o in zip(prompts, outs)
        )
    return pairs


def _kl_epoch(
    model,
    tok,
    examples: list[DistillExample],
    vec: nn.Parameter,
    opt,
    layer: int,
    *,
    positions: str,
    skip_bos: bool,
    batch_size: int,
    max_len: int,
    rng: random.Random,
    history: list[float],
    log_fn,
    step: int,
    epoch: int,
    tag: str,
) -> int:
    """One epoch of forward-KL distillation on `examples` — the SHARED
    optimizer loop of rungs 2 and 3, so the two rungs cannot silently
    diverge in update count per example (the budget match is structural,
    not by convention). Returns the updated step count."""
    device = next(model.parameters()).device
    encoded = [_encode_aligned(tok, ex, max_len) for ex in examples]
    batches: list[tuple[Batch, Batch]] = []
    for i in range(0, len(encoded), batch_size):
        chunk = encoded[i : i + batch_size]
        batches.append(
            (collate([t for t, _ in chunk], tok.pad_token_id),
             collate([s for _, s in chunk], tok.pad_token_id))
        )
    order = list(range(len(batches)))
    rng.shuffle(order)
    for bi in order:
        teacher_b, student_b = batches[bi]
        teacher_b, student_b = teacher_b.to(device), student_b.to(device)
        with torch.no_grad():
            t_logits = _gather_completion_logits(
                model(input_ids=teacher_b.input_ids,
                      attention_mask=teacher_b.attention_mask).logits,
                teacher_b,
            )
        with steering(model, {layer: vec}, positions=positions,
                      skip_bos=skip_bos):
            s_logits = _gather_completion_logits(
                model(input_ids=student_b.input_ids,
                      attention_mask=student_b.attention_mask).logits,
                student_b,
            )
        t_logp = torch.log_softmax(t_logits.float(), -1)
        s_logp = torch.log_softmax(s_logits.float(), -1)
        loss = F.kl_div(s_logp, t_logp, log_target=True,
                        reduction="batchmean")
        opt.zero_grad()
        loss.backward()
        opt.step()
        step += 1
        loss_val = float(loss.detach())
        history.append(loss_val)
        if log_fn:
            log_fn({f"{tag}/kl_loss": loss_val,
                    f"{tag}/vec_norm": float(vec.detach().norm()),
                    "epoch": epoch}, step)
    return step


def _log_budget(log_fn, tag: str, trajectories: int, steps: int) -> None:
    if log_fn:
        log_fn({f"budget/{tag}/trajectories_seen": trajectories,
                f"budget/{tag}/optimizer_steps": steps}, steps)


def train_mc_kl(
    model,
    tok,
    instruction: Instruction,
    fit_prompts: list[str],
    layer: int,
    *,
    epochs: int = 3,
    n_samples: int = 3,
    temperature: float = 1.0,
    lr: float = 5e-3,
    batch_size: int = 4,
    max_len: int = 768,
    max_new_tokens: int = 200,
    gen_batch_size: int = 8,
    positions: str = "all",
    skip_bos: bool = True,
    seed: int = 0,
    log_fn=None,
) -> SteeringVector:
    """Rung 2: sampled-teacher MC-KL (Prompt Baking Eq. 3), BUDGET-MATCHED
    to rung 3.

    Per epoch: the TEACHER (SI in context) samples a FRESH set of
    n_samples completions per fit prompt at `temperature`; forward KL
    teacher -> steered student on the completion positions, exactly the
    rung-3 loop (_kl_epoch). Resampling per epoch — rather than training
    epochs times on one fixed sample set — gives rung 2 the same
    trajectory count and data-refresh cadence as rung 3, so the 2-vs-3
    contrast isolates the trajectory SOURCE. Under one config
    (same epochs, n_samples, batch_size) the two rungs see identical
    trajectory and optimizer-step totals; both are recorded in metadata
    (n_trajectories_seen, n_optimizer_steps) and logged as budget/*."""
    device = next(model.parameters()).device
    rng = random.Random(seed)
    for p in model.parameters():
        p.requires_grad_(False)
    vec = nn.Parameter(torch.zeros(hidden_size(model), device=device))
    opt = torch.optim.AdamW([vec], lr=lr)

    history: list[float] = []
    step = 0
    trajectories = 0
    for epoch in range(epochs):
        # -- fresh sampled-teacher set (the rung-2 source, epoch cadence
        # matched to rung 3's rollout refresh) ----------------------------
        pairs = sample_teacher_pairs(
            model, tok, instruction, fit_prompts,
            n_samples=n_samples, temperature=temperature,
            batch_size=gen_batch_size, max_new_tokens=max_new_tokens,
            seed=seed * 100_003 + epoch,
        )
        pairs = [p for p in pairs if p.pos.strip()]
        examples = pairs_to_distill_examples(tok, pairs, instruction)
        trajectories += len(examples)
        step = _kl_epoch(
            model, tok, examples, vec, opt, layer,
            positions=positions, skip_bos=skip_bos, batch_size=batch_size,
            max_len=max_len, rng=rng, history=history, log_fn=log_fn,
            step=step, epoch=epoch, tag="mc_kl",
        )
    _log_budget(log_fn, "mc_kl", trajectories, step)

    return SteeringVector(
        vectors={layer: vec.detach().float().cpu()},
        method="mc_kl",
        model_name=model.config._name_or_path,
        instruction_id=instruction.id,
        metadata={
            "rung": 2,
            "epochs": epochs,
            "n_samples": n_samples,
            "temperature": temperature,
            "lr": lr,
            "n_fit_prompts": len(fit_prompts),
            "positions": positions,
            "skip_bos": skip_bos,
            "n_trajectories_seen": trajectories,
            "n_optimizer_steps": step,
            "final_loss": history[-1] if history else None,
        },
    )


def train_gkd(
    model,
    tok,
    instruction: Instruction,
    fit_prompts: list[str],
    layer: int,
    *,
    epochs: int = 3,
    n_samples: int = 1,
    rollout_temperature: float = 1.0,
    lr: float = 5e-3,
    batch_size: int = 4,
    max_len: int = 768,
    max_new_tokens: int = 200,
    gen_batch_size: int = 8,
    positions: str = "all",
    skip_bos: bool = True,
    seed: int = 0,
    log_fn=None,
) -> SteeringVector:
    """Rung 3: student-rollout distillation (GKD, forward KL).

    Per epoch: the student (current vector active, NO instruction in
    context) samples n_samples fresh completions per fit prompt —
    regenerated EVERY epoch, that is the rung — then the teacher's logits
    on (SI + prompt + student completion) supervise the steered student
    via forward KL on the completion positions (_kl_epoch, shared with
    rung 2). The rollouts are sampled under torch.no_grad (no gradient
    through sampling). Set n_samples equal to rung 2's for the budget
    match; totals land in metadata and budget/* logs."""
    device = next(model.parameters()).device
    rng = random.Random(seed)
    torch.manual_seed(seed)
    for p in model.parameters():
        p.requires_grad_(False)
    vec = nn.Parameter(torch.zeros(hidden_size(model), device=device))
    opt = torch.optim.AdamW([vec], lr=lr)

    student_rendered = [build_chat_prompt(tok, p, system=None)
                        for p in fit_prompts]
    teacher_rendered = [build_chat_prompt(tok, p, system=instruction.text)
                        for p in fit_prompts]

    history: list[float] = []
    step = 0
    trajectories = 0
    for epoch in range(epochs):
        # -- fresh student rollouts (the on-policy state distribution) ----
        examples: list[DistillExample] = []
        for _ in range(n_samples):
            completions = generate_batch(
                model, tok, student_rendered,
                sv={layer: vec.detach()}, scale=1.0, positions=positions,
                skip_bos=skip_bos,  # rollout under the SAME policy as train/eval
                batch_size=gen_batch_size, max_new_tokens=max_new_tokens,
                do_sample=True, temperature=rollout_temperature,
            )
            examples.extend(
                DistillExample(prompt_with_si=t, prompt_without=s, completion=c)
                for t, s, c in zip(teacher_rendered, student_rendered,
                                   completions)
                if c.strip()
            )
        trajectories += len(examples)
        step = _kl_epoch(
            model, tok, examples, vec, opt, layer,
            positions=positions, skip_bos=skip_bos, batch_size=batch_size,
            max_len=max_len, rng=rng, history=history, log_fn=log_fn,
            step=step, epoch=epoch, tag="gkd",
        )
    _log_budget(log_fn, "gkd", trajectories, step)

    return SteeringVector(
        vectors={layer: vec.detach().float().cpu()},
        method="gkd",
        model_name=model.config._name_or_path,
        instruction_id=instruction.id,
        metadata={
            "rung": 3,
            "epochs": epochs,
            "n_samples": n_samples,
            "rollout_temperature": rollout_temperature,
            "lr": lr,
            "n_fit_prompts": len(fit_prompts),
            "positions": positions,
            "skip_bos": skip_bos,
            "n_trajectories_seen": trajectories,
            "n_optimizer_steps": step,
            "final_loss": history[-1] if history else None,
        },
    )
