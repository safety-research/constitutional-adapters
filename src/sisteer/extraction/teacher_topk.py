"""Offline top-K teacher-logit precomputation.

The distillation teacher is the FROZEN base model prompted with a FIXED SI,
teacher-forced on a FIXED pool of its own completions — it does not depend
on the steering object, the arm, or the optimizer step. Its
completion-position logits are therefore a pure function of
(model, revision, dtype, SI, pool, max_len) and can be computed ONCE per
(model, SI) and stored, instead of being recomputed by an online teacher
forward at every training step. Consuming the artifact removes the whole
teacher leg from the training step: no full-width [B, S, V] teacher logits,
no [N, V] fp32 teacher log-softmax, no prefix-KV cache — the memory that
previously capped the hook arms' microbatch at 1-2 on H200.

Storage form: for every RESPONSE (decode-only) position — exactly the
completion-predicting positions ``labels[:, 1:] != -100`` that the fit's
teacher gather selects — the TOP-``K`` (default 50) logits: token indices
(int32) and logit values in the model's own compute dtype (bf16 in
production, so a stored value is bit-identical to what the forward
produced), plus the position's full-vocab ``logsumexp`` (fp32) so the
stored head's true probability mass is auditable. The distillation loss
consumes this through :func:`sisteer.extraction.distill.soft_divergence_topk`
(renormalized top-K forward KL — the exact form is documented there).

CANONICAL TEACHER FORWARD: batch size 1, unpadded, ``no_grad`` — the
batch-shape-independent definition. Run-to-run rebuilds are TYPICALLY
bitwise-identical on fixed hardware, but not guaranteed: one H200
container class reproduced with differences (cuBLAS kernel selection /
tie ordering vary by host; the verification script classifies any
recurrence). This does not weaken the design — the ARTIFACT, once built,
is the fixed training target (consumption is fully deterministic given
the artifact, and the row-sha binding hashes token IDS, never logits), so
a rebuild-vs-rebuild wobble can only produce a second equally-valid
teacher, never a silently different world for one fit. It is
NOT guaranteed bitwise-equal to the fit's BATCHED online teacher (kernel
tiling differs with padded batch shapes) — that difference is the same
fp-tolerance class the row-sharding contract and the teacher prefix cache
(default-ON in the production configs, so the fits' teacher values already
carry it) accept for the gradient-training legs. Decision-bearing passes
(floor probes, held-out epoch selection) never consume this artifact —
they stay authoritative online forwards inside the trainers, flag or no
flag, exactly like the prefix-cache rule they inherit it from.

BINDING, not trust: every stored example carries the sha256 of its exact
teacher-row token ids (SI-rendered prompt + completion + EOS, as
``_encode_aligned`` produced them). At consumption each batch ROW is
re-hashed from the batch tensors themselves and looked up by that sha —
so a store built from a different pool, tokenizer, truncation, or SI can
never be silently consumed; any miss raises :class:`TeacherTopKMismatch`
and the caller falls back to the online teacher LOUDLY. The manifest
additionally keys the whole artifact on (model, revision, dtype, si_id,
instruction sha, pool content sha, max_len, top_k) for the cheap
pre-flight check.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch

from .distill import DistillExample, _encode_aligned
from .traindata import Batch

#: Manifest schema — bump on any incompatible layout change so an old
#: artifact refuses loudly instead of being misread.
TOPK_SCHEMA_VERSION = 1

#: The production head width — top-200 by logit: k=200 is near-free
#: (~1.2 KB/position) and tightens the training target; the measured T=1
#: loss gap at k=50 was 7-18% of the full-vocab KL, halving with each
#: doubling of k. Top-K by logit is the same set at any temperature
#: (monotone transform), but the MASS it covers is temperature-dependent —
#: see soft_divergence_topk's note. Consumers validate artifacts against
#: THIS value, so build and fit widths cannot silently diverge.
DEFAULT_TOP_K = 200

#: The temperatures a top-K head may be consumed at (the re-tune@32's
#: audited temperature grid — tune_arm.TEMPERATURE_GRID for the KL arms).
#: At a grid T the renormalized-head KL IS the re-tuned objective: the HPO
#: sweeps T and tunes ON the head loss, so head-vs-full fidelity is a T=1
#: question only. Outside the grid (token_faithful's historical T=10,
#: where the head's mass collapses) consumption is refused — that would
#: silently train an objective nothing audited. ``lse_t`` stores
#: per-position ``logsumexp(l/T)`` at every grid T so each T's true head
#: mass stays auditable from the artifact alone.
AUDITED_TOPK_TEMPERATURES = (1.0, 2.0, 4.0)


class TeacherTopKMismatch(RuntimeError):
    """The artifact does not match the fit that tried to consume it — a
    manifest key differs, or a batch row's token ids hash to no stored
    example. Consumers catch THIS type only, log loudly, and fall back to
    the online teacher; anything else propagates."""


def teacher_row_sha(ids) -> str:
    """sha256 hex of a teacher row's exact token ids — the per-example
    binding key. int32 little-endian bytes of the id sequence: stable
    across torch/numpy versions and cheap enough to run per row at fit
    setup (hashing ~16k ints is nothing next to one 27B forward)."""
    t = torch.as_tensor(ids, dtype=torch.int32).contiguous()
    return hashlib.sha256(t.numpy().tobytes()).hexdigest()


def instruction_sha(text: str) -> str:
    """sha256 hex of the SI text — the manifest's instruction key."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass
class TeacherTopK:
    """One (model, SI, pool) artifact: per completion position, the top-K
    teacher logits. Flat over positions; ``offsets[i]:offsets[i+1]`` is
    example ``i``'s slice, and ``row_sha[i]`` binds it to the exact teacher
    row ids it was computed from."""

    idx: torch.Tensor       # [N_pos, K] int32 — token indices, top-K by logit
    val: torch.Tensor       # [N_pos, K] model dtype — the logit values
    #: [N_pos, len(lse_temperatures)] fp32 — full-vocab logsumexp(l/T) per
    #: audited temperature (meta["lse_temperatures"], the audited grid at
    #: build time), so head mass at any grid T is exact:
    #: sum_k exp(val_k/T - lse_t[:, i]).
    lse_t: torch.Tensor
    offsets: torch.Tensor   # [n_examples + 1] int64 — example -> slice
    row_sha: list[str]      # [n_examples] — teacher-row token-id sha256 hex
    meta: dict = field(default_factory=dict)
    _by_sha: dict | None = None

    @property
    def top_k(self) -> int:
        return int(self.idx.shape[1])

    @property
    def lse(self) -> torch.Tensor:
        """The T=1 column of ``lse_t`` (the historical single-lse field)."""
        return self.lse_t[:, self._t_col(1.0)]

    def _t_col(self, temperature: float) -> int:
        temps = [float(t) for t in self.meta.get(
            "lse_temperatures", list(AUDITED_TOPK_TEMPERATURES))]
        try:
            return temps.index(float(temperature))
        except ValueError:
            raise ValueError(
                f"T={temperature} is not among this artifact's audited "
                f"lse temperatures {temps} — head mass there is not "
                "reconstructible from a head") from None

    def __len__(self) -> int:
        return len(self.row_sha)

    def _index(self) -> dict:
        if self._by_sha is None:
            self._by_sha = {s: i for i, s in enumerate(self.row_sha)}
        return self._by_sha

    # -- consumption -----------------------------------------------------

    def gather_for_batch(self, batch: Batch) -> tuple[torch.Tensor,
                                                      torch.Tensor]:
        """The batch's completion-position top-K teacher logits —
        ``(idx [N, K] int32, val [N, K])`` in the SAME row-major flatten
        order as ``distill._gather_completion_logits`` (row 0's completion
        positions, then row 1's, ...), so it drops into the loss exactly
        where the gathered ``[N, V]`` online tensor did.

        Every row is re-derived from the batch tensors and bound by sha:
        the row's non-pad token ids must hash to a stored example AND its
        completion-position count must equal the stored slice's length.
        Any miss raises :class:`TeacherTopKMismatch` — never a silent
        partial gather (the caller falls back to the online teacher,
        loudly)."""
        by_sha = self._index()
        ids = batch.input_ids.cpu()
        attn = batch.attention_mask.cpu()
        labels = batch.labels.cpu()
        idx_rows, val_rows = [], []
        for r in range(ids.shape[0]):
            row_len = int(attn[r].sum())
            if row_len and not bool(attn[r, :row_len].all()):
                raise TeacherTopKMismatch(
                    f"batch row {r}: attention mask is not right-padded "
                    "contiguous — cannot derive the teacher row ids")
            sha = teacher_row_sha(ids[r, :row_len])
            i = by_sha.get(sha)
            if i is None:
                raise TeacherTopKMismatch(
                    f"batch row {r}: teacher row ids (len {row_len}, sha "
                    f"{sha[:8]}) match no stored example — the artifact "
                    "was built from a different pool/tokenizer/truncation")
            lo, hi = int(self.offsets[i]), int(self.offsets[i + 1])
            n_comp = int((labels[r, 1:] != -100).sum())
            if hi - lo != n_comp:
                raise TeacherTopKMismatch(
                    f"batch row {r}: {n_comp} completion positions in the "
                    f"batch but {hi - lo} stored for example {i} — "
                    "truncation or masking differs from the build")
            idx_rows.append(self.idx[lo:hi])
            val_rows.append(self.val[lo:hi])
        return torch.cat(idx_rows), torch.cat(val_rows)

    def validate(self, *, model: str, model_revision, si_id: str,
                 pool_content_sha256: str, max_len: int,
                 top_k: int = DEFAULT_TOP_K, dtype: str | None = None,
                 ) -> list[str]:
        """Key check against the fit about to consume this artifact —
        returns a list of human-readable mismatches (empty == usable).
        Cheap and side-effect-free so drivers can decide fallback before
        any model work."""
        m = self.meta
        problems: list[str] = []

        def _want(key, want):
            got = m.get(key)
            if str(got) != str(want):
                problems.append(f"{key}: artifact has {got!r}, fit wants "
                                f"{want!r}")

        if int(m.get("schema", -1)) != TOPK_SCHEMA_VERSION:
            problems.append(f"schema: artifact has {m.get('schema')!r}, "
                            f"reader is {TOPK_SCHEMA_VERSION}")
        _want("model", model)
        _want("model_revision", model_revision)
        _want("si_id", si_id)
        _want("pool_content_sha256", pool_content_sha256)
        _want("max_len", int(max_len))
        if self.top_k != int(top_k):
            problems.append(f"top_k: artifact stores {self.top_k}, fit "
                            f"wants {int(top_k)}")
        if dtype is not None:
            _want("dtype", dtype)
        return problems

    # -- persistence -------------------------------------------------------

    def save(self, path) -> Path:
        """Write ``<path>`` (safetensors: idx/val/lse/offsets) and
        ``<path>.json`` (manifest + row shas). Atomic-ish: tensors first,
        manifest last, so a torn write leaves no manifest and the loader
        refuses."""
        from safetensors.torch import save_file

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        save_file({"idx": self.idx.contiguous(),
                   "val": self.val.contiguous(),
                   "lse_t": self.lse_t.contiguous(),
                   "offsets": self.offsets.contiguous()}, str(path))
        manifest = dict(self.meta)
        manifest["schema"] = TOPK_SCHEMA_VERSION
        manifest["top_k"] = self.top_k
        manifest["n_examples"] = len(self.row_sha)
        manifest["n_positions"] = int(self.idx.shape[0])
        manifest["row_sha"] = list(self.row_sha)
        Path(str(path) + ".json").write_text(json.dumps(manifest, indent=1))
        return path

    @classmethod
    def load(cls, path) -> "TeacherTopK":
        from safetensors.torch import load_file

        path = Path(path)
        manifest_path = Path(str(path) + ".json")
        if not path.is_file() or not manifest_path.is_file():
            raise FileNotFoundError(
                f"teacher top-K artifact incomplete at {path}: both the "
                "safetensors file and its .json manifest are required")
        tensors = load_file(str(path))
        manifest = json.loads(manifest_path.read_text())
        row_sha = manifest.pop("row_sha")
        store = cls(idx=tensors["idx"], val=tensors["val"],
                    lse_t=tensors["lse_t"], offsets=tensors["offsets"],
                    row_sha=row_sha, meta=manifest)
        n = len(row_sha)
        if store.offsets.numel() != n + 1 or int(store.offsets[-1]) != int(
                store.idx.shape[0]):
            raise TeacherTopKMismatch(
                f"artifact at {path} is internally inconsistent: "
                f"{n} examples, offsets {store.offsets.numel()} entries "
                f"ending at {int(store.offsets[-1])}, "
                f"{int(store.idx.shape[0])} stored positions")
        return store


def topk_artifact_path(pairs_path) -> Path:
    """Sidecar convention, mirroring the token cache: the artifact lives
    NEXT TO the pool file it was built from
    (``<pairs>.teachertopk.safetensors``), so shipping a pool ships its
    teacher and the 'auto' consumption mode finds it with no extra
    plumbing."""
    return Path(str(pairs_path) + ".teachertopk.safetensors")


def compute_teacher_topk(
    model,
    tok,
    examples: list[DistillExample],
    *,
    max_len: int,
    top_k: int = DEFAULT_TOP_K,
    meta: dict | None = None,
    progress: bool = True,
) -> TeacherTopK:
    """Run the CANONICAL teacher forward (bs=1, unpadded, ``no_grad``) over
    every example and keep the top-``top_k`` logits at each completion-
    predicting position — exactly the positions the fit's teacher gather
    (``labels[:, 1:] != -100``) selects, exactly the rows
    ``_encode_aligned`` builds (SI-rendered teacher prompt + completion +
    EOS, truncated at ``max_len`` against the LONGER of the two prompt
    renderings — the same truncation the fit applies). Deterministic for a
    fixed (model, weights, hardware): a plain forward draws no RNG.

    Stored values keep the model's own compute dtype, so in a bf16
    production fit a stored logit is BIT-IDENTICAL to what this forward
    produced; ``lse`` (full-vocab logsumexp, fp32) rides along per
    position so the head's true mass — sum(exp(val - lse)) — is auditable
    at build time and forever after."""
    if top_k < 1:
        raise ValueError(f"top_k must be >= 1, got {top_k}")
    if not examples:
        raise ValueError("compute_teacher_topk needs at least one example")
    device = next(model.parameters()).device
    vocab = int(model.get_input_embeddings().weight.shape[0])
    if top_k > vocab:
        raise ValueError(f"top_k={top_k} exceeds the vocab ({vocab})")
    idx_parts: list[torch.Tensor] = []
    val_parts: list[torch.Tensor] = []
    lse_parts: list[torch.Tensor] = []
    temps = [float(t) for t in AUDITED_TOPK_TEMPERATURES]
    offsets = [0]
    shas: list[str] = []
    t0 = time.monotonic()
    with torch.no_grad():
        for i, ex in enumerate(examples):
            t_seq, _s_seq = _encode_aligned(tok, ex, max_len)
            ids = torch.tensor(t_seq.ids, dtype=torch.long,
                               device=device).unsqueeze(0)
            logits = model(input_ids=ids,
                           attention_mask=torch.ones_like(ids)).logits[0]
            # Positions predicting completion tokens: prompt_len-1 .. n-2
            # (the [:, :-1] shift of the full-width gather, at bs=1 with
            # no padding).
            comp = logits[t_seq.prompt_len - 1: len(t_seq.ids) - 1]
            val, idx = torch.topk(comp, top_k, dim=-1)
            comp32 = comp.float()
            lse = torch.stack([torch.logsumexp(comp32 / t, dim=-1)
                               for t in temps], dim=1)
            idx_parts.append(idx.to(torch.int32).cpu())
            val_parts.append(val.cpu())
            lse_parts.append(lse.cpu())
            offsets.append(offsets[-1] + comp.shape[0])
            shas.append(teacher_row_sha(t_seq.ids))
            if progress and (i + 1) % max(1, len(examples) // 20) == 0:
                el = time.monotonic() - t0
                print(f"[teacher-topk] {i + 1}/{len(examples)} examples, "
                      f"{offsets[-1]} positions, {el:.0f}s elapsed "
                      f"(~{el / (i + 1) * (len(examples) - i - 1):.0f}s "
                      "left)", flush=True)
    store = TeacherTopK(
        idx=torch.cat(idx_parts),
        val=torch.cat(val_parts),
        lse_t=torch.cat(lse_parts),
        offsets=torch.tensor(offsets, dtype=torch.int64),
        row_sha=shas,
        meta=dict(meta or {}))
    store.meta.setdefault("dtype", str(store.val.dtype).replace("torch.", ""))
    store.meta.setdefault("max_len", int(max_len))
    store.meta["lse_temperatures"] = temps
    store.meta["build"] = {
        "batch_size": 1,
        "torch": torch.__version__,
        "device": (torch.cuda.get_device_name(0)
                   if device.type == "cuda" else "cpu"),
        "seconds": round(time.monotonic() - t0, 1),
    }
    return store


def head_mass_stats(store: TeacherTopK, temperature: float = 1.0) -> dict:
    """How much teacher probability the stored head covers at
    ``temperature`` — the audit number behind every consumption decision.
    Exact at every temperature in the artifact's ``lse_temperatures``
    (the audited grid, stored at build time): head mass =
    ``sum_k exp(val_k/T - logsumexp(l/T))``. A temperature the build did
    not normalize for raises — a head cannot reconstruct a full-vocab
    normalizer, and a silently-wrong estimate is worse than a refusal.
    Returns mean/min/p1/p50 of per-position head mass."""
    col = store._t_col(temperature)
    z = torch.exp(store.val.float() / float(temperature)
                  - store.lse_t[:, col].unsqueeze(1)).sum(1)
    q = torch.quantile(z, torch.tensor([0.0, 0.01, 0.5]))
    return {"mean": float(z.mean()), "min": float(q[0]),
            "p1": float(q[1]), "p50": float(q[2])}


#: Meta keys every shard of one artifact must agree on for a merge to be
#: the same world: the full identity the manifest keys on, plus the lse
#: temperature grid the columns are laid out by.
_SHARD_INVARIANT_META = ("model", "model_revision", "dtype", "si_id",
                         "instruction_sha256", "pool_content_sha256",
                         "pool_file_sha256", "max_len", "lse_temperatures",
                         "positions")


def merge_teacher_topk(shards: list["TeacherTopK"]) -> "TeacherTopK":
    """Merge per-example-range shards into ONE artifact, byte-identical to
    an unsharded build over the same pool on the same hardware (pinned by
    tests/test_teacher_topk.py): each shard is the bs=1 build over a
    CONTIGUOUS slice of the pool (``meta["shard"] = {index, of, lo, hi}``,
    stamped by ``precompute-teacher --shard i/N``), per-example forwards
    are independent, and this concatenation appends them in exactly the
    order the unsharded loop would have — tensors cat'd, offsets rebased
    by each shard's cumulative position count, row shas concatenated.

    Fail-loudly validation: every shard must carry the SAME full-pool
    identity (``_SHARD_INVARIANT_META`` — a shard hashes the WHOLE pool,
    never its slice, so two shards of different pools can never merge),
    the same top_k, and shard ranges that tile ``[0, n)`` contiguously
    with no gap, overlap, or missing shard."""
    if not shards:
        raise ValueError("merge_teacher_topk needs at least one shard")
    metas = [dict(s.meta) for s in shards]
    for m in metas:
        if not m.get("shard"):
            raise TeacherTopKMismatch(
                "merge_teacher_topk got a store with no meta['shard'] — "
                "refusing to merge artifacts that were not built as shards")
    order = sorted(range(len(shards)),
                   key=lambda i: int(metas[i]["shard"]["index"]))
    shards = [shards[i] for i in order]
    metas = [metas[i] for i in order]
    ref = metas[0]
    n_of = int(ref["shard"]["of"])
    if len(shards) != n_of:
        raise TeacherTopKMismatch(
            f"merge_teacher_topk got {len(shards)} shard(s) of a "
            f"{n_of}-shard build — a partial merge would silently drop "
            "pool examples")
    for i, (s, m) in enumerate(zip(shards, metas)):
        for k in _SHARD_INVARIANT_META:
            if json.dumps(m.get(k), sort_keys=True) != json.dumps(
                    ref.get(k), sort_keys=True):
                raise TeacherTopKMismatch(
                    f"shard {i}: meta[{k!r}] = {m.get(k)!r} differs from "
                    f"shard 0's {ref.get(k)!r} — different worlds cannot "
                    "merge")
        if s.top_k != shards[0].top_k:
            raise TeacherTopKMismatch(
                f"shard {i}: top_k {s.top_k} != shard 0's "
                f"{shards[0].top_k}")
        sh = m["shard"]
        if int(sh["index"]) != i or int(sh["of"]) != n_of:
            raise TeacherTopKMismatch(
                f"shard sequence broken at position {i}: got index "
                f"{sh['index']} of {sh['of']} (need every shard 0..{n_of - 1} "
                "exactly once)")
        if int(sh["hi"]) - int(sh["lo"]) != len(s):
            raise TeacherTopKMismatch(
                f"shard {i} claims range [{sh['lo']}, {sh['hi']}) but "
                f"stores {len(s)} examples")
        if i > 0 and int(sh["lo"]) != int(metas[i - 1]["shard"]["hi"]):
            raise TeacherTopKMismatch(
                f"shard {i} starts at {sh['lo']} but shard {i - 1} ended at "
                f"{metas[i - 1]['shard']['hi']} — gap or overlap")
    if int(metas[0]["shard"]["lo"]) != 0:
        raise TeacherTopKMismatch(
            f"shard 0 starts at {metas[0]['shard']['lo']}, not 0")

    offsets = [torch.tensor([0], dtype=torch.int64)]
    base = 0
    for s in shards:
        offsets.append(s.offsets[1:] + base)
        base += int(s.offsets[-1])
    meta = dict(ref)
    meta.pop("shard", None)
    # Per-shard build records ride along (which container built what).
    meta["build"] = {"sharded": n_of,
                     "shards": [dict(m.get("build") or {},
                                     lo=int(m["shard"]["lo"]),
                                     hi=int(m["shard"]["hi"]))
                                for m in metas]}
    return TeacherTopK(
        idx=torch.cat([s.idx for s in shards]),
        val=torch.cat([s.val for s in shards]),
        lse_t=torch.cat([s.lse_t for s in shards]),
        offsets=torch.cat(offsets),
        row_sha=[sha for s in shards for sha in s.row_sha],
        meta=meta)


def shard_artifact_path(pairs_path, index: int, of: int) -> Path:
    """Shard sidecar convention: beside the pool like the final artifact,
    self-describing suffix, globbable per (pairs, N)."""
    return Path(f"{pairs_path}.teachertopk.shard{int(index):03d}"
                f"of{int(of):03d}.safetensors")
