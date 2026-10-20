"""Compositional-steering operators + the learned <and> trainer.

Radevski et al. (2601.05062) — exactly two composition mechanisms,
head-to-head:

- compose_additive: k separately-trained residual-vector objects combined
  by SUMMING their vectors (the CAA-style additive baseline Radevski show
  collapses). Order-invariant by construction, so the callable carries
  `order_invariant = True` and eval/composition.py scores it on one
  ordering instead of k!.
- compose_tokens / token_compose_fn: the Radevski mechanism for
  token_faithful. A single shared learned <and> embedding is placed
  BETWEEN separately-trained appended behavior tokens, so the prompt
  layout is [E_x, e_b1, e_<and>, e_b2 (, e_<and>, e_b3)]. Order matters
  (NOT order_invariant): eval averages over k! orderings. 3-behavior use
  is zero-shot — the same <and> between each adjacent pair.

train_and_token trains the <and> embedding on 2-behavior pairs ONLY,
with the LLM and the per-behavior tokens FROZEN, under the faithful
temperature-softened KL (T=10, loss xT^2 — tokens.FAITHFUL_*) toward the
joined-instruction teacher, plus the orthogonality regularizer
L_orth = sum_{b in B_seen} cos(e_<and>, e_b)^2 (lambda=0.5 default) that
stops <and> from collapsing onto a behavior direction. Two details the
paper pins explicitly (2601.05062 §3, Table 4): e_<and> is ZERO-initialized
(the Table 4 init ablation drives unseen-composition accuracy), and
L_orth ranges over the behaviors that appear in a training pair (B_seen)
only — never the whole behavior library.

Contract with eval/composition.py (already written; matched exactly):
compose_fn is called as `compose_fn(objs_in_order, order)` — objs arrive
ALREADY permuted, `order` is bookkeeping — and the default generation
path injects the sentinel block for every token-kind object (kind
startswith "token"), which ComposedTokenObject's kind = "token"
satisfies.

Defaults not pinned by the plan: the <and> slot reuses the behavior
tokens' sentinel token id (the replacement hook is position-based, so id
collision across slots is fine and no second reserved token is needed);
AdamW weight_decay=0 so the objective is exactly KL + lambda*L_orth,
with no incidental decay.
"""

from __future__ import annotations

import random
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from ..data.instructions import joined_behavior_instruction
from ..data.pairs import Pair, pairs_to_distill_examples
from ..generate import generate_batch
from ..models import build_chat_prompt
from ..vectors import SteeringVector
from .distill import DistillExample, _encode_aligned, _gather_completion_logits
from .tokens import FAITHFUL_LOSS_SCALE, FAITHFUL_TEMPERATURE
from .traindata import Batch, collate

DEFAULT_ORTHO_LAMBDA = 0.5


# -- mechanism 1: residual-vector additive sum -----------------------------------


def compose_additive(objs: list, order: tuple[int, ...] | None = None):
    """Compose k separately-trained vector objects by summing their vectors.

    Returns a fresh object of the same (VectorObject-adapter) type whose
    SteeringVector is the per-layer sum, applied with one shared scale at
    eval. `order` is accepted for the eval compose_fn signature and
    ignored — vector addition commutes, hence `order_invariant = True`.

    norm_relative components are refused: normalizing the SUM is not the
    sum of the individually-normalized additions, so a silent sum would
    score a different mechanism than the one trained.
    """
    if len(objs) < 2:
        raise ValueError("composition needs >= 2 behavior objects")
    first = objs[0]
    for o in objs:
        if not hasattr(o, "sv"):
            raise TypeError(
                "compose_additive takes VectorObject-adapted vectors "
                f"(need .sv), got {type(o).__name__}"
            )
        if getattr(o, "norm_relative", False):
            raise ValueError(
                "compose_additive cannot compose norm_relative vectors: "
                "the sum is not the sum of normalized additions"
            )
        if o.positions != first.positions or o.skip_bos != first.skip_bos:
            raise ValueError(
                "compose_additive needs identical apply settings across "
                f"components (positions/skip_bos): {o.positions}/{o.skip_bos}"
                f" vs {first.positions}/{first.skip_bos}"
            )
    vectors: dict[int, torch.Tensor] = {}
    for o in objs:
        for layer, vec in o.sv.vectors.items():
            if layer in vectors:
                vectors[layer] = vectors[layer] + vec
            else:
                vectors[layer] = vec.clone()
    components = [o.sv.instruction_id for o in objs]
    sv = SteeringVector(
        vectors=vectors,
        method="composed_additive",
        model_name=first.sv.model_name,
        instruction_id="+".join(components),
        metadata={"operator": "additive_sum", "components": components},
    )
    return type(first)(
        sv, positions=first.positions, skip_bos=first.skip_bos,
        norm_relative=False,
    )


compose_additive.order_invariant = True  # eval scores ONE ordering, notes it


# -- mechanism 2: token_faithful learned <and> composition token ------------------


class ComposedTokenObject:
    """[E_x, e_b1, e_<and>, e_b2 (, e_<and>, e_b3)] as one intervention.

    Holds an ordered sequence of (sentinel_text, sentinel_id, embedding)
    slots. inject() appends the sentinel block at the end of the rendered
    prompt; apply() replaces the embeddings of every contiguous
    occurrence of the block's id-sequence with the slot embeddings, in
    order — position-based, so all slots may share one sentinel token id.
    scale multiplies the injected embeddings (SoftTokenObject semantics:
    the hook is bitwise inert on inputs that contain no block; at
    scale=0 the hook passes the block's NATIVE embeddings through
    unchanged — scale-0 == unsteered — it does not remove the
    extra positions).

    kind = "token": matches eval/composition.py's default generation
    path, which injects every token-kind object (kind startswith
    "token"). The saved blob's kind differs on purpose; see save().
    """

    kind = "token"

    def __init__(self, sentinel_texts: list[str], sentinel_ids: list[int],
                 embeds: list):
        if not (len(sentinel_texts) == len(sentinel_ids) == len(embeds)):
            raise ValueError("slot lists must be the same length")
        self.sentinel_texts = list(sentinel_texts)
        self.sentinel_ids = list(sentinel_ids)
        self.embeds = list(embeds)

    @property
    def trainable(self) -> list[nn.Parameter]:
        seen: set[int] = set()
        out: list[nn.Parameter] = []
        for e in self.embeds:
            if (isinstance(e, nn.Parameter) and e.requires_grad
                    and id(e) not in seen):
                seen.add(id(e))
                out.append(e)
        return out

    def n_params(self) -> int:
        seen: set[int] = set()
        total = 0
        for e in self.embeds:
            if id(e) not in seen:
                seen.add(id(e))
                total += e.numel()
        return total

    def inject(self, rendered_prompt: str) -> str:
        """Append the whole sentinel block at the very end of the prompt,
        so the slot embeddings occupy the final prompt positions."""
        return rendered_prompt + "".join(self.sentinel_texts)

    @contextmanager
    def apply(self, model, scale: float = 1.0):
        emb = model.get_input_embeddings()
        expected = torch.tensor(self.sentinel_ids, dtype=torch.long)
        L = len(self.sentinel_ids)

        def hook(_mod, inputs, output):
            if scale == 0.0:
                # Scale-0 == unsteered: the block's native embeddings
                # pass through unchanged, matching every residual arm's
                # bitwise identity at 0.
                return output
            ids = inputs[0]
            if ids.dim() != 2 or ids.shape[1] < L:
                return output  # KV-cached decode steps etc.: pass through
            windows = ids.unfold(1, L, 1)  # [B, S-L+1, L]
            match = (windows == expected.to(ids.device)).all(-1)
            if not match.any():
                return output
            out = output.clone()
            block = torch.stack(
                [e.to(device=output.device, dtype=torch.float32)
                 for e in self.embeds]
            )
            block = (scale * block).to(output.dtype)
            for b, j in match.nonzero(as_tuple=False).tolist():
                out[b, j : j + L] = block
            return out

        h = emb.register_forward_hook(hook)
        try:
            yield
        finally:
            h.remove()

    def to_(self, device) -> None:
        for e in self.embeds:
            e.data = e.data.to(device)

    def save(self, path: Path) -> None:
        # Blob kind is "token_composed", deliberately NOT the class's
        # runtime kind: methods.load_object has no loader registered for
        # it and fails loudly rather than mis-hydrating the blob as a
        # single SoftTokenObject. Rehydrate with ComposedTokenObject.
        # load_blob (composed objects are normally rebuilt from their
        # saved components instead).
        torch.save(
            {
                "kind": "token_composed",
                "sentinels": self.sentinel_texts,
                "embeds": torch.stack(
                    [e.detach().float().cpu() for e in self.embeds]
                ),
            },
            path,
        )

    @classmethod
    def load_blob(cls, blob: dict, model, tok) -> "ComposedTokenObject":
        texts = list(blob["sentinels"])
        ids = []
        for t in texts:
            tid = tok(t, add_special_tokens=False)["input_ids"]
            if len(tid) != 1:
                raise ValueError(f"saved sentinel {t!r} is not a single token")
            ids.append(tid[0])
        embeds = [nn.Parameter(e.clone()) for e in blob["embeds"].unbind(0)]
        return cls(texts, ids, embeds)


def compose_tokens(objs: list, and_token,
                   order: tuple[int, ...] | None = None) -> ComposedTokenObject:
    """Insert the learned <and> embedding between appended behavior tokens.

    objs: separately-trained AppendedSoftTokenObjects, ALREADY in the
    ordering to realize ([e_b1, e_<and>, e_b2, ...]); `order` is
    bookkeeping for the eval compose_fn signature. and_token: the trained
    <and> nn.Parameter (or an object carrying it as .param). The <and>
    slots ride the first behavior's sentinel token id — replacement is
    position-based, so sharing the id across slots is exact.
    """
    if len(objs) < 2:
        raise ValueError("composition needs >= 2 behavior tokens")
    for o in objs:
        if not (hasattr(o, "param") and hasattr(o, "sentinel_id")
                and hasattr(o, "sentinel_text")):
            raise TypeError(
                "compose_tokens takes soft-token objects "
                f"(need .param/.sentinel_*), got {type(o).__name__}"
            )
    and_emb = and_token.param if hasattr(and_token, "param") else and_token
    texts: list[str] = []
    ids: list[int] = []
    embeds: list = []
    for i, o in enumerate(objs):
        if i > 0:  # the same shared <and> between each adjacent pair
            texts.append(objs[0].sentinel_text)
            ids.append(objs[0].sentinel_id)
            embeds.append(and_emb)
        texts.append(o.sentinel_text)
        ids.append(o.sentinel_id)
        embeds.append(o.param)
    return ComposedTokenObject(texts, ids, embeds)


def token_compose_fn(and_token):
    """Bind a trained <and> token into the (objs, order) compose_fn that
    eval/composition.py calls. Order matters for the token layout, so
    order_invariant is False and eval averages over k! orderings."""

    def _compose(objs: list, order: tuple[int, ...] | None = None):
        return compose_tokens(objs, and_token, order)

    _compose.order_invariant = False
    return _compose


# -- the learned <and> trainer ----------------------------------------------------


def orthogonality_penalty(and_emb: torch.Tensor, behavior_embeds) -> torch.Tensor:
    """L_orth = sum_b cos(e_<and>, e_b)^2. Behavior embeddings are frozen
    inputs, so they are detached; gradient flows to e_<and> only."""
    total = and_emb.new_zeros(())
    for e in behavior_embeds:
        c = F.cosine_similarity(
            and_emb.float(), e.detach().to(and_emb.device).float(), dim=0
        )
        total = total + c**2
    return total


def train_and_token(
    model,
    tok,
    behavior_tokens: dict[str, object],
    train_pairs: list[tuple[str, str]],
    base_prompts: list[str] | None = None,
    *,
    pairs_by_target: dict[tuple[str, str], list[Pair]] | None = None,
    ortho_lambda: float = DEFAULT_ORTHO_LAMBDA,
    lr: float = 5e-3,
    epochs: int = 6,
    batch_size: int = 2,
    max_len: int = 768,
    temperature: float = FAITHFUL_TEMPERATURE,
    loss_scale: float = FAITHFUL_LOSS_SCALE,
    max_new_tokens: int = 200,
    gen_batch_size: int = 8,
    seed: int = 0,
    log_fn=None,
) -> nn.Parameter:
    """Train the single shared <and> embedding on 2-behavior pairs ONLY.

    Per Radevski: the LLM and the per-behavior tokens are FROZEN (their
    requires_grad is turned off here — they are finished artifacts); the
    only trainable is the returned <and> parameter. Objective per step =
    loss_scale * KL_T(teacher || student) + ortho_lambda * L_orth, where
    the teacher is the model prompted with the joined 'do X and Y'
    instruction (joined_behavior_instruction) and the student is the
    plain prompt with [e_b1, e_<and>, e_b2] appended. L_orth ranges over
    B_seen — the behaviors appearing in train_pairs — only (paper §3): a
    behavior token that is in behavior_tokens but in no training pair
    cannot influence the result. 3-behavior composition reuses the
    trained <and> zero-shot via compose_tokens.

    Teacher completions per target come from pairs_by_target[target]
    (.pos is used) when given, else are generated here from base_prompts
    under the joined instruction. Loss curves go to log_fn(dict, step)
    (keys and_token/{loss,kl,ortho}).
    """
    for target in train_pairs:
        if len(target) != 2:
            raise ValueError(
                "<and> is trained on 2-behavior pairs only (Radevski); "
                f"got {target!r} — 3-behavior use is zero-shot at eval"
            )
        for name in target:
            if name not in behavior_tokens:
                raise ValueError(f"no behavior token for {name!r}")
    if pairs_by_target is None and base_prompts is None:
        raise ValueError("need base_prompts or pairs_by_target")

    device = next(model.parameters()).device
    rng = random.Random(seed)
    torch.manual_seed(seed)

    # Freeze the LLM and the finished per-behavior tokens.
    for p in model.parameters():
        p.requires_grad_(False)
    for obj in behavior_tokens.values():
        obj.param.requires_grad_(False)

    emb = model.get_input_embeddings()
    # e_<and> starts at EXACTLY zero — pinned by the paper (Table 4: the
    # init ablation drives unseen-composition accuracy; zero is its pick).
    and_param = nn.Parameter(
        torch.zeros(emb.weight.shape[1], device=device)
    )
    opt = torch.optim.AdamW([and_param], lr=lr, weight_decay=0.0)
    # L_orth over B_seen ONLY (paper §3): at the shipped seed 0 all 6
    # behaviors are seen so nothing changes, but the invariant must hold
    # for every seed — an unseen behavior's token never enters the loss.
    seen = {n for t in train_pairs for n in t}
    behavior_embeds = [behavior_tokens[n].param for n in sorted(seen)]

    # One composed object + encoded batches per 2-behavior target.
    batches: list[tuple[Batch, Batch, ComposedTokenObject]] = []
    for target in train_pairs:
        target = tuple(target)
        joined = joined_behavior_instruction(list(target))
        if pairs_by_target is not None:
            pairs = pairs_by_target[target]
        else:
            rendered = [
                build_chat_prompt(tok, p, system=joined.text)
                for p in base_prompts
            ]
            pos = generate_batch(
                model, tok, rendered, batch_size=gen_batch_size,
                max_new_tokens=max_new_tokens,
            )
            pairs = [
                Pair(prompt=p, pos=o, neg="", instruction_id=joined.id)
                for p, o in zip(base_prompts, pos)
            ]
        composed = compose_tokens(
            [behavior_tokens[n] for n in target], and_param
        )
        encoded = []
        for ex in pairs_to_distill_examples(tok, pairs, joined):
            injected = composed.inject(ex.prompt_without)
            encoded.append(
                _encode_aligned(
                    tok,
                    DistillExample(
                        prompt_with_si=ex.prompt_with_si,
                        prompt_without=injected,
                        completion=ex.completion,
                        instruction_idx=ex.instruction_idx,
                    ),
                    max_len,
                )
            )
        for i in range(0, len(encoded), batch_size):
            chunk = encoded[i : i + batch_size]
            batches.append(
                (
                    collate([t for t, _ in chunk], tok.pad_token_id),
                    collate([s for _, s in chunk], tok.pad_token_id),
                    composed,
                )
            )

    step = 0
    for epoch in range(epochs):
        order = list(range(len(batches)))
        rng.shuffle(order)
        for bi in order:
            teacher_b, student_b, composed = batches[bi]
            teacher_b, student_b = teacher_b.to(device), student_b.to(device)
            with torch.no_grad():
                t_logits = _gather_completion_logits(
                    model(input_ids=teacher_b.input_ids,
                          attention_mask=teacher_b.attention_mask).logits,
                    teacher_b,
                )
            with composed.apply(model, scale=1.0):
                s_logits = _gather_completion_logits(
                    model(input_ids=student_b.input_ids,
                          attention_mask=student_b.attention_mask).logits,
                    student_b,
                )
            t_logp = torch.log_softmax(t_logits.float() / temperature, -1)
            s_logp = torch.log_softmax(s_logits.float() / temperature, -1)
            kl = F.kl_div(s_logp, t_logp, log_target=True,
                          reduction="batchmean")
            ortho = orthogonality_penalty(and_param, behavior_embeds)
            loss = loss_scale * kl + ortho_lambda * ortho
            opt.zero_grad()
            loss.backward()
            opt.step()
            step += 1
            if log_fn:
                log_fn(
                    {
                        "and_token/loss": float(loss.detach()),
                        "and_token/kl": float(kl.detach()),
                        "and_token/ortho": float(ortho.detach()),
                        "epoch": epoch,
                    },
                    step,
                )
    return and_param
