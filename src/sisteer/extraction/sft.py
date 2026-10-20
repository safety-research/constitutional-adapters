"""REAL-weight fine-tuning arms for the realignment campaign.

Unlike every object in objects.py / multilayer.py — runtime hooks around
frozen weights — these trainers change WEIGHTS. `train_sft_lora` attaches
a genuine PEFT LoRA to the MERGED organism, trains cross-entropy on the
completion tokens of the pos examples, and SAVES the adapter to disk;
eval is then a model swap: `load_model_and_tokenizer(adapters=[EM_adapter,
realign_adapter])` merges both into plain weights and the suites run with
`obj=None`. `train_sft_full` is the optional full-weight upper bound.

Both consume the same `Pair` -> `pairs_to_distill_examples` ->
`_encode_aligned` / `collate` path as the distillation arms and train on
the STUDENT-side (no-SI) encoding, so the FT arms see exactly the data
the object arms see. The FT arms keep the identical summary/record
contract as object arms: suite runners receive (model, tok, items,
system, obj, scale) and emit the same per-item records — whether `model`
is the organism or the merged+realigned checkpoint is invisible below
the driver.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import shutil
from pathlib import Path

import torch
import torch.nn.functional as F

from ..data.pairs import Pair, pairs_to_distill_examples
from ..models import get_decoder_layers
from .distill import _encode_aligned, _gather_completion_logits
from .traindata import Batch, collate


def _require_peft():
    try:
        import peft
    except ImportError as e:  # loud, names the dep
        raise ImportError(
            "the fine-tuning arms require the `peft` package, which is not "
            "importable — install the project's pinned `peft` dependency "
            "(pyproject.toml)"
        ) from e
    return peft


def _assert_plain_tree(model, what: str) -> None:
    """The FT trainers must start from a MERGED organism (plain module
    tree); training on an adapter-wrapped model would nest adapters."""
    leftovers = [n for n, _ in model.named_modules() if "lora_" in n]
    if leftovers:
        raise ValueError(
            f"{what} expects a merged (plain-tree) model, but found LoRA "
            f"modules: {leftovers[:5]} — merge adapters first via "
            "load_model_and_tokenizer(adapter=/adapters=)"
        )
    get_decoder_layers(model)  # loud ValueError if the tree is exotic


def _pairs_sha256(pairs: list[Pair]) -> str:
    blob = "\n".join(
        json.dumps(p.__dict__, sort_keys=True, ensure_ascii=False)
        for p in pairs
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _write_json_atomic(path: Path, obj: dict) -> None:
    """tmp + fsync + os.replace (the save_pairs pattern): the manifest is
    a resume-trusted artifact, so it either doesn't exist or is complete."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        with tmp.open("w") as f:
            json.dump(obj, f, indent=2)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _publish_dir_atomic(tmp_dir: Path, final_dir: Path) -> None:
    """Publish a saved-checkpoint directory under its final name only once
    it is complete: presence of `final_dir` therefore implies a complete
    artifact (exists()-based resume can trust it)."""
    if final_dir.exists():
        shutil.rmtree(final_dir)
    os.replace(tmp_dir, final_dir)


def _encode_student_batches(
    tok, pairs: list[Pair], instruction, batch_size: int, max_len: int
) -> list[Batch]:
    """Student-side (no-SI) encodings of the pos completions, batched.

    Goes through `pairs_to_distill_examples` + `_encode_aligned` so the
    completion truncation matches what the distillation arms train on
    (max_len minus the LONGER of the two prompts, SI included).
    """
    exs = pairs_to_distill_examples(tok, pairs, instruction)
    student_seqs = [_encode_aligned(tok, ex, max_len)[1] for ex in exs]
    return [
        collate(student_seqs[i : i + batch_size], tok.pad_token_id)
        for i in range(0, len(student_seqs), batch_size)
    ]


def _ce_on_completions(logits: torch.Tensor, batch: Batch) -> torch.Tensor:
    s_logits = _gather_completion_logits(logits, batch)
    labels = batch.labels[:, 1:]
    targets = labels[labels != -100]
    return F.cross_entropy(s_logits.float(), targets)


def train_sft_lora(
    model,
    tok,
    pairs: list[Pair],
    instruction,
    *,
    teacher_model=None,
    rank: int = 16,
    targets: tuple[str, ...] = ("q_proj", "o_proj"),
    lr: float = 1e-4,
    epochs: int = 3,
    batch_size: int,
    max_len: int = 768,
    seed: int,
    out_dir: str | Path,
    log_fn=None,
) -> Path:
    """The headline FT trainer (realign F1): a REAL saved LoRA adapter.

    Attaches a genuine `peft.LoraConfig` to the MERGED organism, trains
    cross-entropy on the completion tokens of the pos examples
    (student-side, no-SI encoding), and `save_pretrained`s the adapter to
    `out_dir/adapter/` (atomic tmp-dir + os.replace publish), returning
    that path. Eval later loads it as a model swap —
    `load_model_and_tokenizer(adapters=[EM_adapter, this])` — and runs
    with `obj=None`: a genuine merged checkpoint, NOT a runtime hook
    object.

    `teacher_model` is provenance only (F7): the model whose SI-prompted
    sampling produced the pos completions in `pairs` — recorded in the
    manifest, never forwarded (CE has no teacher logits). The caller's
    `model` is restored to its plain pre-call tree via `unload()` (LoRA
    never touches base weights), so the organism can keep serving other
    arms in the same process.
    """
    peft = _require_peft()
    _assert_plain_tree(model, "train_sft_lora")
    out_dir = Path(out_dir)
    device = next(model.parameters()).device
    rng = random.Random(seed)
    torch.manual_seed(seed)

    batches = _encode_student_batches(tok, pairs, instruction, batch_size, max_len)

    config = peft.LoraConfig(
        r=rank,
        lora_alpha=2 * rank,
        target_modules=list(targets),
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    peft_model = peft.get_peft_model(model, config)
    trainable = [p for p in peft_model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=lr)

    history: list[float] = []
    step = 0
    try:
        for epoch in range(epochs):
            order = list(range(len(batches)))
            rng.shuffle(order)
            for bi in order:
                b = batches[bi].to(device)
                logits = peft_model(
                    input_ids=b.input_ids, attention_mask=b.attention_mask
                ).logits
                loss = _ce_on_completions(logits, b)
                opt.zero_grad()
                loss.backward()
                opt.step()
                step += 1
                loss_val = float(loss.detach())
                history.append(loss_val)
                if log_fn:
                    log_fn({"sft_lora/ce_loss": loss_val, "epoch": epoch}, step)

        adapter_dir = out_dir / "adapter"
        out_dir.mkdir(parents=True, exist_ok=True)
        tmp_dir = out_dir / f".adapter.tmp.{os.getpid()}"
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        peft_model.save_pretrained(str(tmp_dir))
        _publish_dir_atomic(tmp_dir, adapter_dir)
    finally:
        # Restore the caller's organism to its plain pre-call tree —
        # unload() strips the LoRA modules WITHOUT merging (base weights
        # were never touched).
        peft_model.unload()
        model.eval()

    teacher_name = (
        teacher_model.config._name_or_path if teacher_model is not None
        else model.config._name_or_path
    )
    _write_json_atomic(
        out_dir / "manifest.json",
        {
            "kind": "sft_lora",
            "base_model": model.config._name_or_path,
            "instruction_id": instruction.id,
            "n_pairs": len(pairs),
            "pairs_sha256": _pairs_sha256(pairs),
            "teacher_model": teacher_name,
            "teacher_is_self": teacher_model is None,
            "rank": rank,
            "targets": list(targets),
            "lr": lr,
            "epochs": epochs,
            "batch_size": batch_size,
            "max_len": max_len,
            "seed": seed,
            "loss_history": history,
            "final_loss": history[-1] if history else None,
        },
    )
    return adapter_dir


_CKPT_RE = re.compile(r"^ckpt_e(\d+)$")


def _newest_checkpoint(out_dir: Path) -> tuple[int, Path | None]:
    """(epochs_completed, checkpoint_path) of the newest complete per-epoch
    checkpoint under out_dir, or (0, None). Directory presence implies
    completeness because checkpoints publish atomically."""
    best_n, best = 0, None
    if out_dir.is_dir():
        for child in out_dir.iterdir():
            m = _CKPT_RE.match(child.name)
            if m and child.is_dir() and int(m.group(1)) > best_n:
                best_n, best = int(m.group(1)), child
    return best_n, best


def train_sft_full(
    model,
    tok,
    pairs: list[Pair],
    instruction,
    *,
    out_dir: str | Path,
    lr: float = 1e-5,
    epochs: int,
    batch_size: int,
    grad_accum: int,
    seed: int,
    log_fn=None,
) -> Path:
    """Optional full-weight SFT upper bound (no realign arm registers it).

    Memory note: full-weight SFT of an 8B model with AdamW is
    memory-MARGINAL on one H200 even with gradient checkpointing and
    micro-batch 1 (weights + grads + fp32 moments ~130 GB+); if the
    training canary OOMs, the arm is DROPPED ("not run") rather than
    pulling in a new optimizer dependency — it is an optional upper
    bound, not a required cell.

    Preemption-tolerant by construction: `save_pretrained` checkpoints
    (model + tokenizer, atomic tmp-dir + os.replace publish) land at
    `out_dir/ckpt_e{n}/` after every epoch, and a fresh call AUTO-RESUMES
    from the newest checkpoint — no manual flag; optimizer state restarts
    (documented, not saved). The finished model publishes atomically to
    `out_dir/final/` with a `manifest.json` recording pairs provenance.
    Trains cross-entropy on completion tokens of the pos examples
    (student-side no-SI encoding, same data path as the LoRA arm).
    MUTATES `model`'s weights in place; the returned `final/` directory is
    what eval loads (a model swap, `obj=None`).
    """
    _assert_plain_tree(model, "train_sft_full")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    torch.manual_seed(seed)

    start_epoch, ckpt = _newest_checkpoint(out_dir)
    resumed_from = None
    if ckpt is not None:
        from transformers import AutoModelForCausalLM

        from ..models import configure_fast_sdpa
        configure_fast_sdpa()  # cuDNN-SDPA footgun — every load site, see models.py
        ref = AutoModelForCausalLM.from_pretrained(str(ckpt), torch_dtype=dtype)
        model.load_state_dict(ref.state_dict(), strict=True)
        del ref
        resumed_from = ckpt.name
    start_epoch = min(start_epoch, epochs)

    batches = _encode_student_batches(tok, pairs, instruction, batch_size,
                                      max_len=768)

    for p in model.parameters():
        p.requires_grad_(True)
    prev_use_cache = getattr(model.config, "use_cache", None)
    model.config.use_cache = False
    if getattr(model, "supports_gradient_checkpointing", False):
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    history: list[float] = []
    step = 0
    try:
        for epoch in range(start_epoch, epochs):
            # Per-epoch derived seed: a resumed run shuffles epoch n the
            # same way the killed run would have.
            order = list(range(len(batches)))
            random.Random(seed + epoch).shuffle(order)
            opt.zero_grad()
            since_step = 0
            for bi in order:
                b = batches[bi].to(device)
                logits = model(
                    input_ids=b.input_ids, attention_mask=b.attention_mask
                ).logits
                loss = _ce_on_completions(logits, b)
                (loss / grad_accum).backward()
                since_step += 1
                if since_step == grad_accum:
                    opt.step()
                    opt.zero_grad()
                    since_step = 0
                step += 1
                loss_val = float(loss.detach())
                history.append(loss_val)
                if log_fn:
                    log_fn({"sft_full/ce_loss": loss_val, "epoch": epoch}, step)
            if since_step:  # flush the remainder micro-batches
                opt.step()
                opt.zero_grad()

            tmp = out_dir / f".ckpt.tmp.{os.getpid()}"
            if tmp.exists():
                shutil.rmtree(tmp)
            model.save_pretrained(str(tmp))
            tok.save_pretrained(str(tmp))
            _publish_dir_atomic(tmp, out_dir / f"ckpt_e{epoch + 1}")
    finally:
        if getattr(model, "supports_gradient_checkpointing", False):
            model.gradient_checkpointing_disable()
        if prev_use_cache is not None:
            model.config.use_cache = prev_use_cache
        for p in model.parameters():
            p.requires_grad_(False)
        model.eval()

    final_dir = out_dir / "final"
    tmp = out_dir / f".final.tmp.{os.getpid()}"
    if tmp.exists():
        shutil.rmtree(tmp)
    model.save_pretrained(str(tmp))
    tok.save_pretrained(str(tmp))
    _publish_dir_atomic(tmp, final_dir)

    _write_json_atomic(
        out_dir / "manifest.json",
        {
            "kind": "sft_full",
            "base_model": model.config._name_or_path,
            "instruction_id": instruction.id,
            "n_pairs": len(pairs),
            "pairs_sha256": _pairs_sha256(pairs),
            "lr": lr,
            "epochs": epochs,
            "batch_size": batch_size,
            "grad_accum": grad_accum,
            "seed": seed,
            "resumed_from": resumed_from,
            "epochs_run_this_process": epochs - start_epoch,
            "loss_history": history,
            "final_loss": history[-1] if history else None,
        },
    )
    return final_dir
