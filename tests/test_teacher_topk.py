"""Offline top-K teacher-logit precomputation (2026-08-27).

What must hold before any fit consumes a precomputed teacher instead of
running one online:

- DETERMINISM + GATHER PARITY: the artifact's values are exactly what the
  fit's own teacher gather computes (same positions, same flatten order),
  and rebuilding it reproduces it bit for bit.
- THE LOSS FORM IS THE CORRECT KL: at K = |V| the renormalized top-K
  forward KL (distill.soft_divergence_topk) equals the full-vocab
  soft divergence — value AND gradient — so the top-K path is the same
  objective restricted to a head, never a different formula.
- BINDING: a store built from a different pool/truncation can never be
  silently consumed (row-sha mismatch -> loud fallback to the online
  teacher, which then trains bit-identically to a run that never saw the
  artifact).
- REFUSAL: non-T=1-forward-KL arms (token_faithful's T=10) refuse the
  head outright — a top-K head there is not the loss the HPO tuned.

CPU-only: the tiny_llama fixture, synthetic French pairs (never real
store data). The REAL-model top-50-vs-full tightness numbers live in the
verification run (scripts/verify_teacher_topk.py), not here — on a
random-weight tiny model the head covers ~K/|V| of the mass, so tightness
is untestable at CPU scale; these tests pin mechanism, not magnitude.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from sisteer.data.instructions import BY_ID
from sisteer.data.pairs import Pair, pairs_to_distill_examples
from sisteer.extraction.distill import (
    _gather_completion_logits,
    soft_divergence,
    soft_divergence_topk,
)
from sisteer.extraction.lockstep import (
    LockstepArm,
    _encode_arm_batches,
    build_lockstep_object,
    default_lockstep_params,
    train_lockstep,
)
from sisteer.extraction.teacher_topk import (
    TeacherTopK,
    TeacherTopKMismatch,
    compute_teacher_topk,
    head_mass_stats,
    teacher_row_sha,
    topk_artifact_path,
)

INSTRUCTION = BY_ID["french"]
MAX_LEN = 128


def _pairs():
    data = [
        ("Say something nice.", "Bonjour le monde", "Hello world"),
        ("Describe a tree.", "C'est un grand arbre qui pousse", "Tall tree"),
        ("Greet the reader.", "Salut a tous", "Hi everyone"),
        ("Name a color.", "Le bleu est joli et tres calme", "Blue is nice"),
        ("Count to two.", "Un deux", "One two"),
    ]
    return [Pair(prompt=p, pos=pos, neg=neg, instruction_id="french")
            for p, pos, neg in data]


def _examples(tok):
    return pairs_to_distill_examples(tok, _pairs(), INSTRUCTION)


class _Identity:
    """A no-inject object so _encode_arm_batches builds the plain teacher
    batches (any real arm's teacher side is identical anyway)."""

    def inject(self, rendered_prompt: str) -> str:
        return rendered_prompt


def _online_topk(model, tok, examples, *, batch_size, top_k):
    """The FIT's own teacher path: batched encode, full forward, the
    completion gather — then top-K. What the artifact must reproduce."""
    t_batches, _s = _encode_arm_batches(tok, _Identity(), examples,
                                        max_len=MAX_LEN,
                                        batch_size=batch_size)
    idx_parts, val_parts = [], []
    with torch.no_grad():
        for tb in t_batches:
            logits = model(input_ids=tb.input_ids,
                           attention_mask=tb.attention_mask).logits
            gathered = _gather_completion_logits(logits, tb)
            val, idx = torch.topk(gathered, top_k, dim=-1)
            idx_parts.append(idx.to(torch.int32))
            val_parts.append(val)
    return torch.cat(idx_parts), torch.cat(val_parts), t_batches


def test_compute_is_deterministic_and_matches_the_fit_gather(tiny_llama):
    model, tok = tiny_llama
    examples = _examples(tok)
    a = compute_teacher_topk(model, tok, examples, max_len=MAX_LEN,
                             top_k=7, progress=False)
    b = compute_teacher_topk(model, tok, examples, max_len=MAX_LEN,
                             top_k=7, progress=False)
    assert torch.equal(a.idx, b.idx) and torch.equal(a.val, b.val)
    assert torch.equal(a.lse_t, b.lse_t) and a.row_sha == b.row_sha

    # bs=1 online (no padding, identical shapes) must be BITWISE the
    # stored values — same forward, and the store's positions must be
    # exactly the fit gather's positions in the fit's flatten order.
    idx1, val1, _ = _online_topk(model, tok, examples, batch_size=1,
                                 top_k=7)
    assert torch.equal(a.idx, idx1)
    assert torch.equal(a.val, val1)

    # Head mass is a probability in (0, 1] at EVERY audited temperature
    # (per-T lse stored at build time); on a random tiny model a 7-token
    # head over a 32k vocab covers almost nothing — the point of the audit
    # number (real-model tightness is the GPU verification's job, not a
    # CPU-scale claim). An unaudited T refuses rather than estimating.
    for t in (1.0, 2.0, 4.0):
        stats = head_mass_stats(a, temperature=t)
        assert 0.0 < stats["min"] <= stats["mean"] <= 1.0, t
    with pytest.raises(ValueError, match="not among"):
        head_mass_stats(a, temperature=10.0)


def test_gather_for_batch_binds_by_row_and_matches_batched_online(tiny_llama):
    model, tok = tiny_llama
    examples = _examples(tok)
    store = compute_teacher_topk(model, tok, examples, max_len=MAX_LEN,
                                 top_k=7, progress=False)
    idx_b, val_b, t_batches = _online_topk(model, tok, examples,
                                           batch_size=2, top_k=7)
    got_idx, got_val = [], []
    for tb in t_batches:
        i, v = store.gather_for_batch(tb)
        got_idx.append(i)
        got_val.append(v)
    got_idx, got_val = torch.cat(got_idx), torch.cat(got_val)
    # Batched-vs-bs1 forwards are the accepted fp-tolerance class; on the
    # CPU fp32 tiny model they agree to float precision, so the indices
    # must match and the values must be indistinguishable.
    assert torch.equal(got_idx, idx_b)
    assert torch.allclose(got_val, val_b, atol=1e-5, rtol=1e-5)


def test_gather_refuses_foreign_rows_and_wrong_counts(tiny_llama):
    model, tok = tiny_llama
    examples = _examples(tok)
    store = compute_teacher_topk(model, tok, examples[:3], max_len=MAX_LEN,
                                 top_k=5, progress=False)
    # A batch containing an example the store never saw -> refuse.
    t_batches, _s = _encode_arm_batches(tok, _Identity(), examples[3:],
                                        max_len=MAX_LEN, batch_size=2)
    with pytest.raises(TeacherTopKMismatch, match="no stored example"):
        store.gather_for_batch(t_batches[0])
    # The count guard (defense in depth against a corrupted store): same
    # row shas, shifted offsets -> the per-row completion count disagrees
    # -> refuse rather than return a misaligned head.
    bad_offsets = store.offsets.clone()
    bad_offsets[1:] -= 1
    corrupt = TeacherTopK(idx=store.idx, val=store.val, lse_t=store.lse_t,
                          offsets=bad_offsets, row_sha=store.row_sha,
                          meta=dict(store.meta))
    t_same, _s = _encode_arm_batches(tok, _Identity(), examples[:2],
                                     max_len=MAX_LEN, batch_size=2)
    with pytest.raises(TeacherTopKMismatch, match="completion positions"):
        corrupt.gather_for_batch(t_same[0])


def test_save_load_roundtrip_and_key_validation(tiny_llama, tmp_path):
    model, tok = tiny_llama
    examples = _examples(tok)
    meta = {"model": "tiny", "model_revision": "r0", "si_id": "french",
            "pool_content_sha256": "abc123", "max_len": MAX_LEN,
            "dtype": "float32"}
    store = compute_teacher_topk(model, tok, examples, max_len=MAX_LEN,
                                 top_k=5, meta=meta, progress=False)
    path = topk_artifact_path(tmp_path / "pairs_french.jsonl")
    store.save(path)
    loaded = TeacherTopK.load(path)
    for field in ("idx", "val", "lse_t", "offsets"):
        assert torch.equal(getattr(store, field), getattr(loaded, field))
    assert loaded.row_sha == store.row_sha

    ok = dict(model="tiny", model_revision="r0", si_id="french",
              pool_content_sha256="abc123", max_len=MAX_LEN, top_k=5,
              dtype="float32")
    assert loaded.validate(**ok) == []
    # EVERY key dimension must individually refuse — one wrong world, one
    # named mismatch.
    for key, bad in [("model", "other"), ("model_revision", "r1"),
                     ("si_id", "safety_refusal"),
                     ("pool_content_sha256", "beef"), ("max_len", 64),
                     ("top_k", 50), ("dtype", "bfloat16")]:
        problems = loaded.validate(**{**ok, key: bad})
        assert problems, f"{key} mismatch not caught"
        assert any(key.split("_")[0] in p or key in p for p in problems)


def test_row_sha_is_the_ids_not_the_object():
    assert teacher_row_sha([1, 2, 3]) == teacher_row_sha(
        torch.tensor([1, 2, 3]))
    assert teacher_row_sha([1, 2, 3]) != teacher_row_sha([1, 2, 4])


# -- the loss form -----------------------------------------------------------


def test_topk_at_full_vocab_is_exactly_the_full_kl():
    """K = |V| makes the renormalized head the WHOLE distribution, so
    soft_divergence_topk must equal soft_divergence('soft', ...) — value
    and gradient. This pins the exact form (renormalize the head, gather
    the student's full-vocab log-softmax) rather than any of its wrong
    neighbours (renormalizing the student too, or skipping the teacher
    renormalization)."""
    g = torch.Generator().manual_seed(0)
    n, v = 6, 97
    t_logits = torch.randn(n, v, generator=g)
    for temperature in (1.0, 10.0):
        s_logits = torch.randn(n, v, generator=g, requires_grad=True)
        t_val, t_idx = torch.topk(t_logits, v, dim=-1)
        loss_topk = soft_divergence_topk(t_idx.to(torch.int32), t_val,
                                         s_logits,
                                         temperature=temperature)
        (g_topk,) = torch.autograd.grad(loss_topk, s_logits)

        s2 = s_logits.detach().clone().requires_grad_(True)
        t_logp = torch.log_softmax(t_logits / temperature, -1)
        s_logp = torch.log_softmax(s2 / temperature, -1)
        loss_full = soft_divergence("soft", t_logp, s_logp)
        (g_full,) = torch.autograd.grad(loss_full, s2)

        assert torch.allclose(loss_topk, loss_full, atol=1e-6), temperature
        assert torch.allclose(g_topk, g_full, atol=1e-6), temperature


def test_topk_head_drops_tail_mass_by_the_documented_form():
    """At K < |V| the loss is KL(q || s) with q the renormalized head —
    check against a hand-computed reference so the implementation can
    never silently drift to a different truncation convention."""
    g = torch.Generator().manual_seed(1)
    n, v, k = 4, 50, 5
    t_logits = torch.randn(n, v, generator=g)
    s_logits = torch.randn(n, v, generator=g)
    t_val, t_idx = torch.topk(t_logits, k, dim=-1)
    got = soft_divergence_topk(t_idx.to(torch.int32), t_val, s_logits)
    q = torch.softmax(t_val, -1)                     # renormalized head
    s_logp = torch.log_softmax(s_logits, -1).gather(-1, t_idx)
    want = (q * (torch.log(q) - s_logp)).sum(-1).mean()
    assert torch.allclose(got, want, atol=1e-6)


# -- consumption in train_lockstep -------------------------------------------


def _train(model, tok, arm, examples, *, teacher_topk=None, params=None,
           diagnostics=None):
    obj = build_lockstep_object(arm, model, tok, INSTRUCTION, _pairs(),
                                layer=1)
    group = [LockstepArm(arm, obj, params or default_lockstep_params(arm))]
    train_lockstep(model, tok, group, examples, epochs=1, batch_size=2,
                   max_len=MAX_LEN, seed=0, teacher_topk=teacher_topk,
                   diagnostics=diagnostics)
    return obj


def test_lockstep_consumes_topk_and_matches_online_at_full_vocab(tiny_llama):
    """With K = |V| the head IS the full distribution, so a topk-consuming
    fit must land within float tolerance of the online fit — this pins the
    whole consumption plumbing (assembly order, keep-subselect, loss form)
    with the head-truncation variable removed. The top-50-vs-full-vocab
    GAP on a real model is the GPU verification's number, not a CPU
    claim."""
    model, tok = tiny_llama
    examples = _examples(tok)
    vocab = int(model.get_input_embeddings().weight.shape[0])
    store = compute_teacher_topk(model, tok, examples, max_len=MAX_LEN,
                                 top_k=vocab, progress=False)
    diag_on, diag_off = {}, {}
    online = _train(model, tok, "lora_r1", examples, diagnostics=diag_off)
    offline = _train(model, tok, "lora_r1", examples, teacher_topk=store,
                     diagnostics=diag_on)
    assert diag_off["lora_r1"]["teacher_forward"] == "shared_lockstep"
    assert diag_on["lora_r1"]["teacher_forward"] == "precomputed_topk"
    for p_on, p_off in zip(offline.trainable, online.trainable):
        assert torch.allclose(p_on, p_off, atol=1e-6, rtol=1e-5)
    # Something trained (parity of two no-op fits is vacuous).
    assert sum(float(p.abs().sum()) for p in online.trainable) > 0.0
    # The decision-bearing numbers came from the SAME authoritative online
    # probes on both sides — identical, not merely close.
    assert (diag_on["lora_r1"]["noop_floor"]
            == diag_off["lora_r1"]["noop_floor"])
    assert (diag_on["lora_r1"]["step0_loss"]
            == diag_off["lora_r1"]["step0_loss"])


def test_lockstep_falls_back_loudly_on_row_mismatch(tiny_llama, capsys):
    """A store that does not bind (built from a different pool slice) must
    not crash the fit and must not silently half-consume: the group trains
    BIT-IDENTICALLY to a run that never saw the artifact, and says so."""
    model, tok = tiny_llama
    examples = _examples(tok)
    vocab = int(model.get_input_embeddings().weight.shape[0])
    stale = compute_teacher_topk(model, tok, examples[:2], max_len=MAX_LEN,
                                 top_k=vocab, progress=False)
    diag = {}
    with_stale = _train(model, tok, "lora_r1", examples, teacher_topk=stale,
                        diagnostics=diag)
    out = capsys.readouterr().out
    assert "[teacher-topk] FALLBACK" in out
    assert diag["lora_r1"]["teacher_forward"] == "shared_lockstep"
    online = _train(model, tok, "lora_r1", examples)
    for p_a, p_b in zip(with_stale.trainable, online.trainable):
        assert torch.equal(p_a, p_b)


def test_lockstep_refuses_topk_outside_the_audited_grid(tiny_llama):
    """token_faithful trains at T=10, where a top-K head's mass collapses
    — consuming the head would silently train an objective nothing
    audited, so setup must refuse (the driver routes such groups to the
    online teacher instead). The audited grid {1, 2, 4} (the re-tune's
    swept temperatures, where the head KL IS the tuned objective) is
    accepted — pinned by the T=2 consumption test below."""
    model, tok = tiny_llama
    examples = _examples(tok)
    store = compute_teacher_topk(model, tok, examples, max_len=MAX_LEN,
                                 top_k=5, progress=False)
    obj = build_lockstep_object(
        "token_faithful", model, tok, INSTRUCTION, _pairs(), layer=1,
        sentinel=(tok.unk_token, tok.unk_token_id))
    group = [LockstepArm("token_faithful", obj,
                         default_lockstep_params("token_faithful"))]
    with pytest.raises(ValueError, match="audited temperature grid"):
        train_lockstep(model, tok, group, examples, epochs=1, batch_size=2,
                       max_len=MAX_LEN, seed=0, teacher_topk=store)
    # Same refusal for the divergence-ablation arms (rkl needs mass over
    # the student's support — a head cannot carry it).
    vec = build_lockstep_object("vector_rkl", model, tok, INSTRUCTION,
                                _pairs(), layer=1)
    with pytest.raises(ValueError, match="audited temperature grid"):
        train_lockstep(model, tok,
                       [LockstepArm("vector_rkl", vec,
                                    default_lockstep_params("vector_rkl"))],
                       examples, epochs=1, batch_size=2, max_len=MAX_LEN,
                       seed=0, teacher_topk=store)


def test_lockstep_consumes_topk_at_audited_t2_and_matches_online(tiny_llama):
    """The re-tune sweeps T over {1, 2, 4} with the head as the objective:
    at an audited T != 1 the topk path must be accepted AND compute the
    same loss the online teacher would at that T — pinned at K = |V|,
    where the head is the whole distribution, isolating the T-plumbing
    from head truncation."""
    from dataclasses import replace

    model, tok = tiny_llama
    examples = _examples(tok)
    vocab = int(model.get_input_embeddings().weight.shape[0])
    store = compute_teacher_topk(model, tok, examples, max_len=MAX_LEN,
                                 top_k=vocab, progress=False)
    params = replace(default_lockstep_params("lora_r1"), temperature=2.0)
    diag_on, diag_off = {}, {}
    online = _train(model, tok, "lora_r1", examples, params=params,
                    diagnostics=diag_off)
    offline = _train(model, tok, "lora_r1", examples, teacher_topk=store,
                     params=params, diagnostics=diag_on)
    assert diag_on["lora_r1"]["teacher_forward"] == "precomputed_topk"
    assert diag_off["lora_r1"]["teacher_forward"] == "shared_lockstep"
    for p_on, p_off in zip(offline.trainable, online.trainable):
        assert torch.allclose(p_on, p_off, atol=1e-6, rtol=1e-5)
    assert sum(float(p.abs().sum()) for p in online.trainable) > 0.0


# -- driver wiring (train_two_si_objects.py) -----------------------------------


import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "_t2_topk", REPO / "scripts/train_two_si_objects.py")
t2 = importlib.util.module_from_spec(_spec)
sys.modules["_t2_topk"] = t2
_spec.loader.exec_module(t2)

SI = "safety_refusal"
TINY = "hf-internal-testing/tiny-random-LlamaForCausalLM"


def _si_pairs():
    return [Pair(prompt=f"Question number {i}?",
                 pos=f"Reponse numero {i} avec quelques mots",
                 neg=f"Answer {i}", instruction_id=SI)
            for i in range(4)]


def _driver_cfg() -> dict:
    return {
        "model": TINY,
        "layer": 1, "completion_tokens": 16, "max_len": 256,
        "max_len_by_si": {SI: 256},
        "teacher_prefix_cache": False,
        "batch_size_by_arm": {"lora_r1_all": 2},
        "train_kwargs": {
            "default": {"epochs": 1, "seed": 0, "holdout_frac": 0.0,
                        "floor_ratio_max": None, "probe_n": 2,
                        "monitors": False},
        },
        "si_ids": [SI],
    }


def _build_artifact(model, tok, pairs, cfg, tmp_path, *,
                    pool_sha=None) -> str:
    """The artifact exactly as cmd_precompute_teacher would key it for
    this cfg (model id, SI, pool content sha, max_len, the production
    top_k = DEFAULT_TOP_K — the width the driver validates against)."""
    from sisteer.eval.jailbreak import resolve_instruction
    from sisteer.extraction.teacher_topk import (DEFAULT_TOP_K,
                                                 instruction_sha)

    instruction = resolve_instruction(SI)
    examples = pairs_to_distill_examples(tok, pairs, instruction)
    meta = {
        "model": cfg["model"],
        "model_revision": cfg.get("model_revision"),
        "dtype": cfg.get("dtype"),
        "si_id": SI,
        "instruction_sha256": instruction_sha(instruction.text),
        "pool_content_sha256": pool_sha or t2.pool_content_sha256(pairs),
        "max_len": 256,
        "positions": "completion_decode_only",
    }
    store = compute_teacher_topk(model, tok, examples, max_len=256,
                                 top_k=DEFAULT_TOP_K, meta=meta,
                                 progress=False)
    path = tmp_path / "pairs.jsonl.teachertopk.safetensors"
    store.save(path)
    return str(path)


def test_driver_consumes_a_matching_artifact(tiny_llama, tmp_path, capsys):
    model, tok = tiny_llama
    pairs = _si_pairs()
    cfg = _driver_cfg()
    art = _build_artifact(model, tok, pairs, cfg, tmp_path)
    outcome = t2.train_si_objects(model, tok, SI, pairs, ["lora_r1_all"],
                                  cfg=cfg, save_dir=None, teacher_topk=art)
    assert outcome.ok, {a: (r.error_type, r.error)
                        for a, r in outcome.failed.items()}
    out = capsys.readouterr().out
    assert "[teacher-topk] artifact verified" in out
    detail = outcome.arms["lora_r1_all"].detail
    assert detail["teacher_forward"] == "precomputed_topk"


def test_driver_falls_back_loudly_on_key_mismatch(tiny_llama, tmp_path,
                                                  capsys):
    """A stale artifact (here: built for a different pool) must not crash
    the wave and must not be consumed — the fit runs the online teacher
    and both the log and the fit record say so."""
    model, tok = tiny_llama
    pairs = _si_pairs()
    cfg = _driver_cfg()
    art = _build_artifact(model, tok, pairs, cfg, tmp_path,
                          pool_sha="0000deadbeef")
    outcome = t2.train_si_objects(model, tok, SI, pairs, ["lora_r1_all"],
                                  cfg=cfg, save_dir=None, teacher_topk=art)
    assert outcome.ok
    out = capsys.readouterr().out
    assert "[teacher-topk] FALLBACK" in out
    assert "pool_content_sha256" in out
    detail = outcome.arms["lora_r1_all"].detail
    assert detail["teacher_forward"] == "shared_lockstep"


def test_driver_missing_artifact_path_falls_back(tiny_llama, capsys):
    model, tok = tiny_llama
    cfg = _driver_cfg()
    outcome = t2.train_si_objects(model, tok, SI, _si_pairs(),
                                  ["lora_r1_all"], cfg=cfg, save_dir=None,
                                  teacher_topk="/nonexistent/topk.safetensors")
    assert outcome.ok
    assert "[teacher-topk] FALLBACK" in capsys.readouterr().out


def test_resolve_teacher_topk_flag(tmp_path, capsys):
    """'auto' finds the sidecar beside --pairs, says so when absent;
    'off' is None; an explicit path passes through even when missing (its
    failure is the loud key-check fallback, not a silent skip)."""
    from types import SimpleNamespace as NS

    pairs = tmp_path / "pairs.jsonl"
    pairs.write_text("{}\n")
    assert t2._resolve_teacher_topk(
        NS(teacher_topk="off", pairs=str(pairs))) is None
    assert t2._resolve_teacher_topk(
        NS(teacher_topk="auto", pairs=str(pairs))) is None
    assert "no artifact beside" in capsys.readouterr().out
    sidecar = Path(str(pairs) + ".teachertopk.safetensors")
    sidecar.write_bytes(b"x")
    assert t2._resolve_teacher_topk(
        NS(teacher_topk="auto", pairs=str(pairs))) == str(sidecar)
    assert t2._resolve_teacher_topk(
        NS(teacher_topk="/elsewhere/a.safetensors", pairs=str(pairs))) \
        == "/elsewhere/a.safetensors"


# -- shard + merge (the wall-clock fan-out, re-tune@32) ------------------------


def test_shard_merge_is_byte_identical_to_unsharded(tiny_llama):
    """The fan-out contract: building contiguous example-range shards and
    merging them yields the EXACT artifact an unsharded build produces —
    every tensor bitwise, offsets rebased correctly across ragged shard
    sizes (5 examples over 3 shards -> 2/2/1), row shas in order, and the
    merged store binds to fit batches exactly like the unsharded one."""
    from sisteer.extraction.teacher_topk import merge_teacher_topk

    model, tok = tiny_llama
    examples = _examples(tok)
    n = len(examples)
    base_meta = {"model": "tiny", "model_revision": "r0", "si_id": "french",
                 "pool_content_sha256": "fullpool123", "max_len": MAX_LEN,
                 "dtype": "float32", "pool_file_sha256": "file123",
                 "instruction_sha256": "insn123",
                 "positions": "completion_decode_only"}
    whole = compute_teacher_topk(model, tok, examples, max_len=MAX_LEN,
                                 top_k=7, meta=dict(base_meta),
                                 progress=False)
    shard_n = 3
    bounds = [round(j * n / shard_n) for j in range(shard_n + 1)]
    shards = []
    for i in range(shard_n):
        lo, hi = bounds[i], bounds[i + 1]
        meta = dict(base_meta,
                    shard={"index": i, "of": shard_n, "lo": lo, "hi": hi})
        shards.append(compute_teacher_topk(
            model, tok, examples[lo:hi], max_len=MAX_LEN, top_k=7,
            meta=meta, progress=False))
    # merge is order-insensitive (sorts by shard index)
    merged = merge_teacher_topk([shards[2], shards[0], shards[1]])
    for field in ("idx", "val", "lse_t", "offsets"):
        assert torch.equal(getattr(merged, field), getattr(whole, field)), \
            field
    assert merged.row_sha == whole.row_sha
    assert "shard" not in merged.meta
    assert merged.meta["pool_content_sha256"] == "fullpool123"
    assert merged.meta["build"]["sharded"] == shard_n
    # ...and it CONSUMES identically: same gathered head for a fit batch.
    t_batches, _s = _encode_arm_batches(tok, _Identity(), examples,
                                        max_len=MAX_LEN, batch_size=2)
    for tb in t_batches:
        iw, vw = whole.gather_for_batch(tb)
        im, vm = merged.gather_for_batch(tb)
        assert torch.equal(iw, im) and torch.equal(vw, vm)


def test_shard_merge_refuses_wrong_worlds(tiny_llama):
    """A partial set, a gap, or a shard of a DIFFERENT pool must refuse —
    a silently dropped or foreign slice is a corrupted teacher."""
    from sisteer.extraction.teacher_topk import merge_teacher_topk

    model, tok = tiny_llama
    examples = _examples(tok)

    def shard(i, of, lo, hi, pool_sha="fullpool123"):
        meta = {"model": "tiny", "model_revision": "r0", "si_id": "french",
                "pool_content_sha256": pool_sha, "max_len": MAX_LEN,
                "dtype": "float32", "pool_file_sha256": "file123",
                "instruction_sha256": "insn123",
                "positions": "completion_decode_only",
                "shard": {"index": i, "of": of, "lo": lo, "hi": hi}}
        return compute_teacher_topk(model, tok, examples[lo:hi],
                                    max_len=MAX_LEN, top_k=5, meta=meta,
                                    progress=False)

    a, b = shard(0, 2, 0, 3), shard(1, 2, 3, 5)
    merge_teacher_topk([a, b])  # sanity: the good pair merges
    with pytest.raises(TeacherTopKMismatch, match="partial merge"):
        merge_teacher_topk([a])
    with pytest.raises(TeacherTopKMismatch, match="gap or overlap"):
        merge_teacher_topk([shard(0, 2, 0, 2), shard(1, 2, 3, 5)])
    with pytest.raises(TeacherTopKMismatch, match="different worlds"):
        merge_teacher_topk([a, shard(1, 2, 3, 5, pool_sha="OTHERpool")])
    with pytest.raises(TeacherTopKMismatch, match="no meta"):
        merge_teacher_topk([compute_teacher_topk(
            model, tok, examples[:2], max_len=MAX_LEN, top_k=5,
            progress=False)])


def test_shard_cli_roundtrip_matches_unsharded_artifact(tiny_llama,
                                                        tmp_path):
    """End-to-end through the CLI seams the Modal containers call:
    per-shard `precompute-teacher --shard-index/--shard-count` writes
    shard sidecars, `merge-teacher` writes the final artifact — equal to
    the unsharded subcommand's output tensor-for-tensor."""
    import json as _json
    from types import SimpleNamespace as NS

    model, tok = tiny_llama
    pairs_path = tmp_path / "pairs.jsonl"
    with pairs_path.open("w") as f:
        for p in _pairs():
            f.write(_json.dumps({"prompt": p.prompt, "pos": p.pos,
                                 "neg": p.neg,
                                 "instruction_id": "safety_refusal"}) + "\n")
    cfg = {"model": TINY, "dtype": "float32", "max_len": MAX_LEN}

    def ns(**kw):
        base = dict(config="", si="safety_refusal", pairs=str(pairs_path),
                    out="", top_k=5, token_cache="off", limit=0,
                    expect_config_sha="", shard_index=-1, shard_count=0,
                    keep_shards=False)
        base.update(kw)
        return NS(**base)

    import _t2_topk  # the driver module loaded at the top of this file

    unsharded = tmp_path / "unsharded.safetensors"
    _t2_topk.cmd_precompute_teacher(ns(out=str(unsharded)), cfg)
    for i in range(2):
        _t2_topk.cmd_precompute_teacher(
            ns(shard_index=i, shard_count=2), cfg)
    _t2_topk.cmd_merge_teacher(ns(shard_count=2), cfg)
    a = TeacherTopK.load(unsharded)
    b = TeacherTopK.load(topk_artifact_path(pairs_path))
    for field in ("idx", "val", "lse_t", "offsets"):
        assert torch.equal(getattr(a, field), getattr(b, field)), field
    assert a.row_sha == b.row_sha
    # merge-teacher removed its shard inputs (no --keep-shards)
    from sisteer.extraction.teacher_topk import shard_artifact_path
    assert not shard_artifact_path(pairs_path, 0, 2).exists()
