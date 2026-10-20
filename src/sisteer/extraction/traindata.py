"""Tokenization/collation shared by the gradient-trained vector methods.

Prompts and completions are tokenized SEPARATELY and concatenated at the
token level. This guarantees the completion's token ids are identical
whether the prompt context contains the system instruction or not (the
alignment the distillation loss depends on), and makes prompt/completion
slicing exact rather than a re-tokenization guess.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .tokcache import cached_ids


@dataclass
class EncodedSeq:
    ids: list[int]
    prompt_len: int


def encode_prompt_completion(
    tok, prompt: str, completion: str, max_len: int = 1024, add_eos: bool = True
) -> EncodedSeq:
    prompt_ids = cached_ids(tok, prompt, True)
    comp_ids = cached_ids(tok, completion, False)
    if add_eos and tok.eos_token_id is not None:
        comp_ids = comp_ids + [tok.eos_token_id]
    ids = (prompt_ids + comp_ids)[:max_len]
    prompt_len = min(len(prompt_ids), max_len)
    return EncodedSeq(ids=ids, prompt_len=prompt_len)


@dataclass
class Batch:
    input_ids: torch.Tensor  # [B, S]
    attention_mask: torch.Tensor  # [B, S]
    prompt_lens: torch.Tensor  # [B]
    labels: torch.Tensor  # [B, S], -100 on prompt & padding

    def to(self, device) -> "Batch":
        return Batch(
            self.input_ids.to(device),
            self.attention_mask.to(device),
            self.prompt_lens.to(device),
            self.labels.to(device),
        )


def collate(seqs: list[EncodedSeq], pad_id: int) -> Batch:
    max_len = max(len(s.ids) for s in seqs)
    B = len(seqs)
    input_ids = torch.full((B, max_len), pad_id, dtype=torch.long)
    attn = torch.zeros((B, max_len), dtype=torch.long)
    labels = torch.full((B, max_len), -100, dtype=torch.long)
    prompt_lens = torch.zeros(B, dtype=torch.long)
    for i, s in enumerate(seqs):
        n = len(s.ids)
        input_ids[i, :n] = torch.tensor(s.ids)
        attn[i, :n] = 1
        labels[i, s.prompt_len : n] = torch.tensor(s.ids[s.prompt_len :])
        prompt_lens[i] = s.prompt_len
    return Batch(input_ids, attn, prompt_lens, labels)


def completion_logprobs(
    logits: torch.Tensor, batch: Batch, gather: torch.Tensor | None = None
) -> torch.Tensor:
    """Sum of per-token log-probs over completion positions. Returns [B].

    ``gather`` (optional [B, S-1] bool) overrides the default completion
    mask — the mask→loss wiring, ``distill._gather_completion_logits``'s
    convention: when a steering hook steers a SUBSET of positions, the sum
    must read exactly the completion positions that subset covers, from
    the SAME mask object the hook used. ``None`` keeps the historical
    behavior bit-for-bit."""
    logp = torch.log_softmax(logits[:, :-1].float(), dim=-1)
    labels = batch.labels[:, 1:]
    mask = gather if gather is not None else (labels != -100)
    safe = labels.clamp(min=0)
    tok_logp = logp.gather(-1, safe.unsqueeze(-1)).squeeze(-1)
    return (tok_logp * mask).sum(dim=1)


def completion_position_mask(batch: Batch) -> torch.Tensor:
    """[B, S-1] bool mask of logit positions that predict completion tokens."""
    return batch.labels[:, 1:] != -100
