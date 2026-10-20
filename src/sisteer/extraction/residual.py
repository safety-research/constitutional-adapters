"""PSR functional-residual distillation: distill the coherent virtual
teacher's single closed-loop trajectory into a fresh all-layer single-gate
``PSRObject``, with the calibrated recovery gate that says whether the
distillation recovered it.

What is distilled. The teacher is any object with ``apply(model, scale)``
— in production the ``PSRPairedWriteObject`` V = +P_mix − P_ctl
(extraction/paired_write.py), in calibration a planted single-gate
control. Its trajectory h^V(x) is produced ON THE FLY: one no-grad
forward under ``teacher.apply(model)`` with every layer captured, per
micro-batch, never stored (a stored 100k-pool trajectory would be tens of
GB; recompute costs one no-grad forward next to the student's backward —
the psr_mse convention). The student θ is a zero-init all-layer
``PSRObject`` (d = 0, w = 0, b = 1: step 0 is an exact no-op, so the
student starts AT the bare model and ED starts at exactly 0).

Loss:

    L(θ) = Σ_ℓ [ Σ_{s ∈ sources} w_s · ( mean_{content pos of s} ‖h^{θ}_ℓ − h^V_ℓ‖²
                                         + mean_{prefix pos of s} ‖·‖² ) ]
           + reg_coeff · gate_reg,

with 50/50 SOURCE weighting (w_mix = w_ctl = ½ — each source contributes
half whatever its token share) and SEPARATE prefix/content terms (prefix =
the advice system-field positions, i.e. the loss-masked prompt slot of a
doc whose prompt is non-empty; content = every other non-pad position),
each a mean over its own positions × hidden dim, both weight 1, both
logged. Under gradient accumulation the means are taken over the
ACCUMULATION GROUP's positions: every micro-batch contributes
``Σ SSE / (N_group · d)`` with the group's per-(source, class) position
counts known before any forward, so the summed micro-batch losses EQUAL
the group loss exactly (linearity) — the token-share rule of the CE/psr
trainers, generalised to four buckets. Loss layers = every layer from the
student's lowest to the top (``psr_loss_layers``; all layers for the
all-layer student).

Capture set: the union of the two parents' training pools,
stratified source × family, with a doc-level ~10%-per-stratum holdout
(:func:`stratified_holdout`; a thin stratum is flagged, never silently
dropped) recorded in ``capture_manifest.json`` so the split is
reproducible from the manifest alone.

Recovery gate: ED = 1 − MSE(h_student, h^V) / MSE(h_bare, h^V)
on the capture holdout, per stratum × position class and per layer, with
DOC-LEVEL bootstrap CIs (:func:`bootstrap_ed`). Calibration on the same
model / trainer / budget / capture set: the ATTAINABLE ED per stratum
from two planted recoverable controls — (a) R(mix, ∅) (teacher = +P_mix,
in-class by construction) and (b) R(mix ⊕ C, mix) (teacher exactly the +C
trajectory of a planted single-gate C at realistic norms,
:func:`plant_control_object`) — and the NUMERIC FLOOR from two bare
forwards under different micro-batch compositions (:func:`numeric_floor`).
:func:`recovery_verdict` rules per source stratum on the content class:
recovered iff the 95% lower bound of ED ≥ 0.7·ED_attain AND ≥ 0.5; a
stratum whose teacher sits within 3× the floor of bare is UNRESOLVABLE.
The failure language is fixed — "not recovered under this trainer +
budget (ED = x [lo, hi] vs attainable y)" — and the word *inexpressible*
never appears in any output this module writes
(:func:`assert_no_banned_words`): the plan makes no capacity claim, and V
carries the decomposition read either way.
"""

from __future__ import annotations

import json
import math
import random
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F

from ..models import get_decoder_layers, hidden_size
from .distill import (DistillExample, EarlyStopper, EarlyStopSpec,
                      TokenizeProgress, _encode_aligned)
from .fitckpt import (cuda_rng_states, load_checkpoint,
                      resolve_checkpoint_kwargs, restore_cuda_rng,
                      save_checkpoint, to_cpu)
from .psr import PSRObject, _capture_hidden, psr_loss_layers
from .tokcache import cached_ids
from .traindata import Batch, collate

SOURCES = ("mix", "ctl")
FAMILIES = ("advice", "stories")
POSITION_CLASSES = ("content", "prefix")

#: Each source contributes half, whatever its token share.
DEFAULT_SOURCE_WEIGHTS = {"mix": 0.5, "ctl": 0.5}

#: Gate constants — pre-stated, never tuned post hoc.
GATE_ATTAIN_FRAC = 0.7
GATE_ABS_FLOOR = 0.5
GATE_UNRESOLVABLE_FLOOR_MULT = 3.0
GATE_CI_LEVEL = 0.95
DEFAULT_N_BOOT = 1000

#: Verdict vocabulary. The failure sentence carries the
#: numbers; the word "inexpressible" is banned from every output.
VERDICT_RECOVERED = "recovered"
VERDICT_NOT_RECOVERED = "not recovered under this trainer + budget"
VERDICT_UNRESOLVABLE = "unresolvable"
BANNED_WORDS = ("inexpressible",)

#: A stratum with fewer pool docs than this is flagged ``thin`` when its
#: holdout is forced to the 1-doc minimum (the ctl-advice stratum has 4).
THIN_STRATUM_DOCS = 10


class TeacherPrecheckError(RuntimeError):
    """The teacher failed its pre-optimization identity check — the run
    refuses to start (never distill a teacher that does not reproduce
    its declared recurrence)."""


class BannedWordError(ValueError):
    """An output text carried a banned verdict word."""


def assert_no_banned_words(text: str, *, where: str = "output") -> None:
    low = text.lower()
    for w in BANNED_WORDS:
        if w in low:
            raise BannedWordError(
                f"{where} contains the banned word {w!r} — the plan makes "
                "no capacity claim; say 'not recovered under this trainer "
                "+ budget' with the numbers instead")


# --------------------------------------------------------------------------- #
# Capture set: docs, stratification, manifest                                  #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class CaptureDoc:
    """One pool document as the capture set sees it: the pool row's
    ``prompt`` (loss-masked prefix, empty for stories) and ``pos`` (the
    trained text), tagged with its SOURCE (which parent's pool it came
    from) and FAMILY."""

    doc_id: str
    source: str
    family: str
    prompt: str
    text: str
    official_tokens: int | None = None

    @property
    def stratum(self) -> str:
        return f"{self.source}/{self.family}"

    @property
    def has_prefix(self) -> bool:
        return bool(self.prompt)

    def example(self) -> DistillExample:
        # The doc-mode convention (docs_to_ce_examples): verbatim, both
        # prompt fields identical — teacher and student share token ids.
        return DistillExample(prompt_with_si=self.prompt,
                              prompt_without=self.prompt,
                              completion=self.text, instruction_idx=0)


def load_capture_docs(pairs_path, source: str) -> list[CaptureDoc]:
    """Read one parent pool (the sdfbase pairs jsonl: ``doc_id``,
    ``family``, ``prompt``, ``pos``, ``official_tokens``) as capture docs
    tagged ``source``. Refuses a row without a doc id or family — the
    stratification needs both."""
    if source not in SOURCES:
        raise ValueError(f"source must be one of {SOURCES}, got {source!r}")
    docs = []
    for i, line in enumerate(Path(pairs_path).read_text(
            encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        r = json.loads(line)
        if not r.get("doc_id") or not r.get("family"):
            raise ValueError(
                f"{pairs_path} row {i}: missing doc_id/family — the capture "
                "set is stratified on both")
        if r["family"] not in FAMILIES:
            raise ValueError(
                f"{pairs_path} row {i}: family {r['family']!r} not in "
                f"{FAMILIES}")
        if not r.get("pos"):
            raise ValueError(f"{pairs_path} row {i}: empty trained text")
        docs.append(CaptureDoc(doc_id=str(r["doc_id"]), source=source,
                               family=str(r["family"]),
                               prompt=str(r.get("prompt") or ""),
                               text=str(r["pos"]),
                               official_tokens=r.get("official_tokens")))
    if not docs:
        raise ValueError(f"{pairs_path}: no documents")
    ids = [d.doc_id for d in docs]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{pairs_path}: duplicate doc ids")
    return docs


def stratified_holdout(docs: list[CaptureDoc], *, frac: float = 0.1,
                       seed: int = 0, min_holdout: int = 1,
                       ) -> tuple[list[CaptureDoc], list[CaptureDoc], dict]:
    """Doc-level holdout carve, ~``frac`` PER STRATUM (source × family).
    A stratum with ≥ 2 docs holds out ``max(min_holdout, round(frac·n))``
    docs (never all of them); a 1-doc stratum holds out nothing and is
    flagged ``no_holdout``; a stratum forced to the 1-doc minimum with
    fewer than :data:`THIN_STRATUM_DOCS` docs is flagged ``thin``.
    Selection is a seeded shuffle within the id-sorted stratum, so the
    carve is a pure function of (docs, frac, seed). Returns (train,
    holdout, manifest); :func:`apply_capture_manifest` re-derives the
    split from the manifest alone."""
    if not 0.0 < frac < 1.0:
        raise ValueError(f"frac must be in (0, 1), got {frac}")
    by_stratum: dict[str, list[CaptureDoc]] = {}
    for d in docs:
        by_stratum.setdefault(d.stratum, []).append(d)
    hold_ids: set[str] = set()
    strata_manifest = {}
    for stratum in sorted(by_stratum):
        members = sorted(by_stratum[stratum], key=lambda d: d.doc_id)
        n = len(members)
        rng = random.Random(f"{seed}|{stratum}")
        order = list(members)
        rng.shuffle(order)
        if n < 2:
            n_hold = 0
        else:
            n_hold = min(n - 1, max(min_holdout, int(round(frac * n))))
        held = sorted(d.doc_id for d in order[:n_hold])
        hold_ids.update(held)
        strata_manifest[stratum] = {
            "n": n, "n_train": n - n_hold, "n_holdout": n_hold,
            "holdout_doc_ids": held,
            "thin": bool(n_hold > 0 and n < THIN_STRATUM_DOCS
                         and n_hold <= min_holdout),
            "no_holdout": bool(n_hold == 0),
        }
    train = [d for d in docs if d.doc_id not in hold_ids]
    holdout = [d for d in docs if d.doc_id in hold_ids]
    manifest = {
        "holdout_frac": frac, "seed": seed, "min_holdout": min_holdout,
        "strata": strata_manifest,
        "docs": [{"doc_id": d.doc_id, "source": d.source,
                  "family": d.family,
                  "split": "holdout" if d.doc_id in hold_ids else "train"}
                 for d in docs],
        "n_train": len(train), "n_holdout": len(holdout),
        "thin_strata": sorted(s for s, m in strata_manifest.items()
                              if m["thin"]),
        "no_holdout_strata": sorted(s for s, m in strata_manifest.items()
                                    if m["no_holdout"]),
    }
    return train, holdout, manifest


def apply_capture_manifest(docs: list[CaptureDoc], manifest: dict,
                           ) -> tuple[list[CaptureDoc], list[CaptureDoc]]:
    """Re-derive (train, holdout) from a recorded manifest — refusing when
    the doc set differs from the recorded one (a manifest describes ONE
    capture set)."""
    recorded = {r["doc_id"]: r for r in manifest["docs"]}
    have = {d.doc_id: d for d in docs}
    if set(recorded) != set(have):
        missing = sorted(set(recorded) - set(have))[:5]
        extra = sorted(set(have) - set(recorded))[:5]
        raise ValueError(
            "capture manifest does not describe these docs "
            f"(missing {missing}, extra {extra})")
    for did, r in recorded.items():
        d = have[did]
        if (r["source"], r["family"]) != (d.source, d.family):
            raise ValueError(
                f"capture manifest: doc {did} recorded as "
                f"{r['source']}/{r['family']} but is {d.stratum}")
    train = [d for d in docs if recorded[d.doc_id]["split"] == "train"]
    hold = [d for d in docs if recorded[d.doc_id]["split"] == "holdout"]
    return train, hold


def capture_manifest_text(manifest: dict, *, sources: dict) -> str:
    """The EXACT text of ``capture_manifest.json`` (the split plus the
    source pools' identities: paths + sha256) — one serialization, so a
    sha256 of this string equals the sha256 of the written file and the
    residual builder's calibration identity can fingerprint the capture
    split without re-reading it."""
    return json.dumps({**manifest, "sources": sources}, indent=1,
                      sort_keys=True) + "\n"


def write_capture_manifest(path, manifest: dict, *, sources: dict) -> None:
    """``capture_manifest.json``: the split plus the source pools'
    identities (paths + sha256), so the capture set is provable."""
    Path(path).write_text(capture_manifest_text(manifest, sources=sources),
                          encoding="utf-8")


# --------------------------------------------------------------------------- #
# Encoding, batching, position classes                                         #
# --------------------------------------------------------------------------- #


@dataclass
class CapBatch:
    """One collated micro-batch of capture docs with its position-class
    masks: ``masks[(source, cls)]`` is a [B, S] bool mask, ``n_pos`` the
    matching position counts (known before any forward)."""

    batch: Batch
    docs: list[CaptureDoc]
    masks: dict[tuple[str, str], torch.Tensor]
    n_pos: dict[tuple[str, str], int]
    row_prefix: torch.Tensor   # [B, S] bool
    row_content: torch.Tensor  # [B, S] bool

    @property
    def n_rows(self) -> int:
        return int(self.batch.input_ids.shape[0])


def position_classes(batch: Batch, has_prefix: torch.Tensor,
                     ) -> tuple[torch.Tensor, torch.Tensor]:
    """(prefix, content) [B, S] bool masks partitioning the NON-PAD
    positions: prefix = positions before the row's prompt boundary on a
    row whose prompt text is non-empty (the advice system field); content
    = every other non-pad position (so a story doc's BOS-only prompt slot
    is content)."""
    valid = batch.attention_mask.bool()
    idx = torch.arange(batch.input_ids.shape[1],
                       device=batch.input_ids.device)
    before = idx[None, :] < batch.prompt_lens[:, None]
    prefix = valid & before & has_prefix.to(valid.device)[:, None]
    content = valid & ~prefix
    return prefix, content


def _encode_docs(tok, docs: list[CaptureDoc], max_len: int, label: str):
    """Student-leg encodings with the per-doc max_len REFUSAL (the
    train_object_ce rule: ``_encode_aligned`` truncates silently, which
    would quietly shrink the capture set)."""
    out = []
    prog = TokenizeProgress(len(docs), label=label, max_len=max_len)
    eos_n = 1 if tok.eos_token_id is not None else 0
    for i, d in enumerate(docs):
        ex = d.example()
        n_comp = len(cached_ids(tok, ex.completion, False)) + eos_n
        n_prompt = len(cached_ids(tok, ex.prompt_without, True))
        if n_prompt + n_comp > max_len:
            raise ValueError(
                f"{label}: doc {d.doc_id} needs {n_prompt} prompt + {n_comp} "
                f"completion(+EOS) = {n_prompt + n_comp} tokens > max_len="
                f"{max_len} — refusing the silent truncation")
        out.append((d, _encode_aligned(tok, ex, max_len)[1]))
        prog.tick(i + 1)
    prog.done()
    return out


def make_cap_batches(tok, docs: list[CaptureDoc], *, batch_size: int,
                     max_len: int, device, label: str,
                     sources=SOURCES, reverse: bool = False,
                     ) -> list[CapBatch]:
    """Length-sorted micro-batches (the production bucketing) with the
    position-class masks attached. ``reverse`` sorts longest-first — the
    second micro-batch COMPOSITION the numeric floor compares against."""
    enc = _encode_docs(tok, docs, max_len, label)
    enc.sort(key=lambda e: len(e[1].ids), reverse=reverse)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    out = []
    bs = int(batch_size)
    for i in range(0, len(enc), bs):
        chunk = enc[i:i + bs]
        b = collate([s for _, s in chunk], pad).to(device)
        bdocs = [d for d, _ in chunk]
        has_prefix = torch.tensor([d.has_prefix for d in bdocs],
                                  dtype=torch.bool)
        prefix, content = position_classes(b, has_prefix)
        masks: dict[tuple[str, str], torch.Tensor] = {}
        n_pos: dict[tuple[str, str], int] = {}
        for s in sources:
            rows = torch.tensor([d.source == s for d in bdocs],
                                dtype=torch.bool, device=b.input_ids.device)
            for cls, m in (("content", content), ("prefix", prefix)):
                mm = m & rows[:, None]
                masks[(s, cls)] = mm
                n_pos[(s, cls)] = int(mm.sum())
        out.append(CapBatch(batch=b, docs=bdocs, masks=masks, n_pos=n_pos,
                            row_prefix=prefix, row_content=content))
    return out


def _sq_by_layer(a: dict[int, torch.Tensor], b: dict[int, torch.Tensor],
                 layers) -> dict[int, torch.Tensor]:
    """Per layer, the per-position squared distance ‖a − b‖² summed over
    the hidden dim: [B, S] float32."""
    return {l: (a[l].float() - b[l].float()).pow(2).sum(-1) for l in layers}


def _forward_capture(model, cb: CapBatch, ctx, layers):
    """One NO-GRAD forward under ``ctx`` with every ``layers`` output
    captured; returns {layer: detached hidden}. (The student's grad-carrying
    forward in the trainer runs its own ``_capture_hidden`` block so the
    backward happens inside the hooks.)"""
    with torch.no_grad(), ctx, _capture_hidden(model, layers) as cap:
        model(input_ids=cb.batch.input_ids,
              attention_mask=cb.batch.attention_mask)
        return {l: cap[l].detach() for l in layers}


# --------------------------------------------------------------------------- #
# Per-doc records, explained delta, bootstrap                                  #
# --------------------------------------------------------------------------- #


def doc_records_from_sq(cb: CapBatch, sq_student: dict[int, torch.Tensor],
                        sq_bare: dict[int, torch.Tensor], layers,
                        ) -> list[dict]:
    """Per doc of the batch: for each position class, the position count
    and the per-layer SSE of student-vs-teacher and bare-vs-teacher —
    the doc-level bootstrap's unit. Device-agnostic: the squared
    distances arrive on the CPU (the trainer parks them there per batch)
    while the batch's masks live on the model's device, so each mask is
    moved to the distances' device before the product (the first real-GPU
    calibration died here once the identity check let it through; the CPU
    test model never sees two devices)."""
    out = []
    for r, d in enumerate(cb.docs):
        rec = {"doc_id": d.doc_id, "stratum": d.stratum,
               "source": d.source, "family": d.family}
        for cls, mask in (("content", cb.row_content[r]),
                          ("prefix", cb.row_prefix[r])):
            n = int(mask.sum())

            def sse(sq: dict[int, torch.Tensor]) -> list[float]:
                return [float((sq[l][r] * mask.to(sq[l].device)).sum())
                        for l in layers]

            rec[cls] = {"n": n, "sse_student": sse(sq_student),
                        "sse_bare": sse(sq_bare)}
        out.append(rec)
    return out


def _sse_totals(records: list[dict], cls: str, layer_idx: int | None,
                ) -> list[tuple[float, float]]:
    """Per record, (SSE_student, SSE_bare) over ``cls`` positions — summed
    over layers, or one layer's — the unit the pooled ED and its bootstrap
    both add up."""
    if layer_idx is None:
        return [(sum(r[cls]["sse_student"]), sum(r[cls]["sse_bare"]))
                for r in records]
    return [(r[cls]["sse_student"][layer_idx], r[cls]["sse_bare"][layer_idx])
            for r in records]


def _ed_from_totals(totals: list[tuple[float, float]]) -> float | None:
    num = den = 0.0
    for s, b in totals:
        num += s
        den += b
    if den <= 0.0:
        return None
    return 1.0 - num / den


def pooled_ed(records: list[dict], cls: str, layer_idx: int | None = None,
              ) -> float | None:
    """1 − Σ_docs SSE_student / Σ_docs SSE_bare over ``cls`` positions,
    summed over layers (or one layer index). None when the bare deviation
    is zero (teacher ≡ bare on these docs — nothing to explain)."""
    return _ed_from_totals(_sse_totals(records, cls, layer_idx))


def bootstrap_ed(records: list[dict], cls: str, *, n_boot: int = DEFAULT_N_BOOT,
                 seed: int = 0, level: float = GATE_CI_LEVEL,
                 layer_idx: int | None = None) -> dict:
    """Doc-level bootstrap CI on the pooled ED: resample the docs with
    replacement ``n_boot`` times (fixed seed), percentile interval at
    ``level``. ``n_docs`` == 1 collapses to the point (reported, and the
    stratum's ``thin`` flag says so)."""
    point = pooled_ed(records, cls, layer_idx)
    out = {"ed": point, "lo": None, "hi": None, "n_docs": len(records),
           "n_boot": n_boot, "level": level}
    if point is None or not records:
        return out
    rng = random.Random(seed)
    n = len(records)
    totals = _sse_totals(records, cls, layer_idx)  # per record, once
    vals = []
    for _ in range(n_boot):
        v = _ed_from_totals([totals[rng.randrange(n)] for _ in range(n)])
        if v is not None:
            vals.append(v)
    if not vals:
        return out
    vals.sort()
    a = (1.0 - level) / 2.0
    lo_i = min(len(vals) - 1, max(0, int(math.floor(a * len(vals)))))
    hi_i = min(len(vals) - 1, max(0, int(math.ceil((1 - a) * len(vals))) - 1))
    out["lo"], out["hi"] = vals[lo_i], vals[hi_i]
    return out


def teacher_deviation(records: list[dict], cls: str) -> float | None:
    """Mean over positions of the bare-vs-teacher squared distance, summed
    over layers (the units the numeric floor is measured in)."""
    n = sum(r[cls]["n"] for r in records)
    if n == 0:
        return None
    return sum(sum(r[cls]["sse_bare"]) for r in records) / n


def ed_summary(records: list[dict], *, layers, n_boot: int = DEFAULT_N_BOOT,
               seed: int = 0) -> dict:
    """The full ED table: per source stratum and per source×family
    stratum, per position class, pooled-over-layers ED with a doc-level
    bootstrap CI, per-layer point EDs, the teacher deviation, and the
    position counts."""
    groups: dict[str, list[dict]] = {}
    for r in records:
        groups.setdefault(r["source"], []).append(r)
        groups.setdefault(r["stratum"], []).append(r)
    out = {}
    for g in sorted(groups):
        recs = groups[g]
        out[g] = {}
        for cls in POSITION_CLASSES:
            b = bootstrap_ed(recs, cls, n_boot=n_boot, seed=seed)
            b["per_layer_ed"] = {int(l): pooled_ed(recs, cls, i)
                                 for i, l in enumerate(layers)}
            b["teacher_deviation"] = teacher_deviation(recs, cls)
            b["n_positions"] = sum(r[cls]["n"] for r in recs)
            out[g][cls] = b
    return out


# --------------------------------------------------------------------------- #
# Numeric floor, planted control, teacher identity, arithmetic residual        #
# --------------------------------------------------------------------------- #


def numeric_floor(model, tok, docs: list[CaptureDoc], *, layers,
                  batch_size: int, max_len: int) -> dict:
    """Two BARE forwards per holdout doc under different micro-batch
    compositions — its length-sorted padded batch vs. the doc alone (batch
    of 1, no padding) — and the mean per-position squared distance between
    the two captures, per stratum × class (summed over layers) and per
    layer. This is the resolution of the instrument: a teacher whose
    deviation from bare sits within :data:`GATE_UNRESOLVABLE_FLOOR_MULT`×
    this floor cannot be ruled on.

    Known limit: at ``batch_size=1`` the two compositions COINCIDE (a
    padded batch of one is the solo forward) and this floor is identically
    zero — it then says nothing about the model's bf16 noise.
    :func:`stream_perturbation_floor` measures that noise directly
    (composition-independent) and rides in the identity report beside it."""
    device = next(model.parameters()).device
    batches = make_cap_batches(tok, docs, batch_size=batch_size,
                               max_len=max_len, device=device,
                               label="residual floor")
    acc: dict[tuple[str, str], list[float]] = {}
    cnt: dict[tuple[str, str], int] = {}
    per_doc = []
    for cb in batches:
        h_a = _forward_capture(model, cb, nullcontext(), layers)
        for r, d in enumerate(cb.docs):
            solo = make_cap_batches(tok, [d], batch_size=1, max_len=max_len,
                                    device=device, label="residual floor solo")[0]
            h_b = _forward_capture(model, solo, nullcontext(), layers)
            n_r = int(cb.batch.attention_mask[r].sum())
            # per layer, this doc's per-position squared distance between
            # the two captures — computed once, masked per class below
            sq = {l: (h_a[l][r, :n_r].float() - h_b[l][0, :n_r].float()
                      ).pow(2).sum(-1) for l in layers}
            rec = {"doc_id": d.doc_id, "stratum": d.stratum,
                   "source": d.source}
            for cls, m_a in (("content", cb.row_content[r]),
                             ("prefix", cb.row_prefix[r])):
                n = int(m_a.sum())
                sse = [float((sq[l] * m_a[:n_r]).sum()) for l in layers]
                rec[cls] = {"n": n, "sse": sse}
                for g in (d.source, d.stratum):
                    acc.setdefault((g, cls), [0.0] * len(layers))
                    cnt[(g, cls)] = cnt.get((g, cls), 0) + n
                    for i in range(len(layers)):
                        acc[(g, cls)][i] += sse[i]
            per_doc.append(rec)
    floor = {}
    for (g, cls), sse in acc.items():
        n = cnt[(g, cls)]
        floor.setdefault(g, {})[cls] = {
            "mse_sum_layers": (sum(sse) / n) if n else None,
            "per_layer_mse": {int(l): (sse[i] / n) if n else None
                              for i, l in enumerate(layers)},
            "n_positions": n,
        }
    return {"floor": floor, "per_doc": per_doc,
            "composition_a": f"length-sorted, batch_size={batch_size}",
            "composition_b": "doc alone, batch_size=1"}


def measure_write_rms(model, obj: PSRObject, batches: list[CapBatch],
                      ) -> dict[int, float]:
    """Per layer, the RMS over valid positions of the object's own write
    ‖g(h)·d‖ under a real forward (gates read off the post-hook stash) —
    the per-layer write magnitude a planted control matches."""
    device = next(model.parameters()).device
    obj.to_(device)
    dn = {l: float(obj.directions[l].detach().float().norm())
          for l in obj.layers}  # constant across batches
    sq = {l: 0.0 for l in obj.layers}
    n = 0
    for cb in batches:
        with torch.no_grad(), obj.apply(model, scale=1.0):
            model(input_ids=cb.batch.input_ids,
                  attention_mask=cb.batch.attention_mask)
            valid = cb.batch.attention_mask.bool()
            n += int(valid.sum())
            for l in obj.layers:
                g = obj._last_gate[l].float()
                sq[l] += float(((g * valid) ** 2).sum()) * dn[l] * dn[l]
    return {l: math.sqrt(sq[l] / max(n, 1)) for l in obj.layers}


def plant_control_object(model, layers: list[int], batches: list[CapBatch], *,
                         seed: int, write_rms: dict[int, float],
                         active_frac: float = 0.3, margin: float = 0.05,
                         ) -> tuple[PSRObject, dict]:
    """A planted single-gate PSR control C at realistic norms: per layer
    a random gate direction w scaled to unit-variance
    pre-activation on the bare states, threshold b set so the gate is open
    on ≥ ``active_frac`` (+ ``margin``) of the reference positions, and a
    random direction d scaled so the per-position write RMS ‖g·d‖ matches
    ``write_rms[layer]`` (the parents' measured per-layer write RMS).
    Calibrated on BARE states (the teacher's own upstream writes shift
    later layers' states a little — the realised active fraction under
    the composite teacher is re-measured and reported by the caller)."""
    device = next(model.parameters()).device
    d_model = hidden_size(model)
    obj = PSRObject(model, list(layers), train_direction=True)
    g = torch.Generator().manual_seed(seed)
    z_by_layer: dict[int, list[torch.Tensor]] = {l: [] for l in layers}
    for cb in batches:
        h = _forward_capture(model, cb, nullcontext(), layers)
        valid = cb.batch.attention_mask.bool()
        for l in layers:
            z_by_layer[l].append(h[l][valid].float().cpu())
    report = {}
    for l in layers:
        z = torch.cat(z_by_layer[l], 0)                     # [N, d]
        w0 = torch.randn(d_model, generator=g)
        proj = z @ w0
        w = w0 / (proj.std() + 1e-8)
        proj = z @ w
        q = torch.quantile(proj, 1.0 - min(0.95, active_frac + margin))
        b = -float(q)
        gate = torch.relu(proj + b)
        rms_g = float(gate.pow(2).mean().sqrt())
        d0 = torch.randn(d_model, generator=g)
        d0 = d0 / d0.norm()
        target = float(write_rms.get(l, 0.0))
        dvec = d0 * (target / max(rms_g, 1e-8))
        obj.gate_w[l].data = w.clone()
        obj.gate_b[l].data = torch.tensor([b])
        obj.directions[l].data = dvec.clone()
        report[int(l)] = {"active_frac_bare": float((gate > 0).float().mean()),
                          "gate_b": b, "direction_norm": float(dvec.norm()),
                          "write_rms_target": target,
                          "write_rms_bare": rms_g * float(dvec.norm())}
    obj.to_(device)
    return obj, report


def _valid_hidden_per_batch(model, batches: list[CapBatch], ctxs: dict, layers):
    """Stream, per batch, ``{name: {layer: [n_valid, d] float32 CPU}}`` —
    the valid-position hidden states under each ``ctxs[name]()``. One batch
    at a time, so a consumer reduces as it goes instead of holding every
    context's full [N, d] table at once (three or four such tables × L
    layers over a ~20k-position reference set is tens of GB of host RAM)."""
    for cb in batches:
        valid = cb.batch.attention_mask.bool()
        out = {}
        for name, ctx_factory in ctxs.items():
            h = _forward_capture(model, cb, ctx_factory(), layers)
            out[name] = {l: h[l][valid].float().cpu() for l in layers}
        yield out


def _rms_over_positions(sq_chunks: list[torch.Tensor]) -> float:
    """RMS over positions of per-position squared distances given as one
    [n_i] chunk per batch. Rows reduce independently, so reducing each
    batch's rows and then the concatenation equals the one-table form."""
    return torch.cat(sq_chunks).mean().sqrt().item()


#: The teacher identity check's tolerance — fp32 precision on the WRITE.
#: The composite's hook sums Σ c_i·u_i(z) in fp32, so against the reference
#: write the only legitimate residue is fp32 cancellation of the cancelled
#: terms: ~2^-24 relative to the LARGEST term (measured ≤ 1.8e-7 on
#: gemma-4-E4B and ≤ 3.5e-8 on Qwen3.5-4B-Base over every layer).
#: ``IDENTITY_RTOL`` is relative to that cancellation scale: 50–300× above
#: the measured residue and 4–5 orders below a missing or mis-signed term.
#: ``IDENTITY_ATOL`` (per-position L2) covers a reference/terms that write
#: nothing at a layer.
IDENTITY_RTOL = 1e-5
IDENTITY_ATOL = 1e-6

_INT_VIEW = {torch.bfloat16: torch.int16, torch.float16: torch.int16,
             torch.float32: torch.int32, torch.float64: torch.int64}


def _require_off_forward_write(obj, tag: str = "object") -> None:
    """Refuse — loudly, before any forward — an object whose write cannot
    be re-evaluated OFF the forward with the hook's own arithmetic: anything
    that is neither a shared-state write nor a PSRObject, and a PSRObject
    with a fit-time or serve-time position mask (its hook zeroes the gate
    at masked positions from tracker state the off-forward form does not
    see; every parent and teacher here is a plain all-position object)."""
    if hasattr(obj, "paired_write"):
        return
    if isinstance(obj, PSRObject):
        if obj.fit_positions != "all" or obj._needs_tracking():
            raise TypeError(
                f"hook contract: {tag} is a position-masked PSRObject "
                f"(fit_positions={obj.fit_positions!r}, positions="
                f"{obj.positions!r}, skip_bos={obj.skip_bos}) — only a plain "
                "all-position object has an off-forward write to check "
                "against")
        return
    raise TypeError(
        f"hook contract: {type(obj).__name__} ({tag}) exposes no fp32 write "
        "(expected a PSRObject or a shared-state write object)")


def _fp32_write(obj, layer: int, z: torch.Tensor) -> torch.Tensor:
    """The object's write at ``layer`` on the pre-write state ``z``
    ([B, S, d]), in float32, evaluated OFF the forward: ``paired_write`` for
    a shared-state combination, ReLU(w·z + b)·d for a single PSR gate, and
    exactly ZERO at a layer the object does not hook. This is the
    arithmetic the object's hook performs before its add (same ops, same
    order, scale 1), so on the hook's own captured z it reproduces the
    hook's write bit for bit."""
    _require_off_forward_write(obj)
    if layer not in obj.layers:
        return torch.zeros(z.shape, dtype=torch.float32, device=z.device)
    if hasattr(obj, "paired_write"):
        return obj.paired_write(layer, z)
    g = F.relu(obj.gate_pre(layer, z))
    return g.unsqueeze(-1) * obj.directions[layer].to(device=z.device)


# --------------------------------------------------------------------------- #
# The hook contract: what the RUNTIME apply() does to the stream             #
# --------------------------------------------------------------------------- #
#
# The write-level identity check above re-derives both writes off the
# forward with the same helper, so for R(mix ⊕ C, mix) it is an algebraic
# self-check of ``paired_write``: an object whose paired_write() is right but
# whose runtime apply() is wrong — no hooks, a doubled or mis-scaled write,
# hooks on the wrong layer, a foreign edit of the stream — would pass it.
# The hook CONTRACT closes that: on the very forward the check (or the
# trainer) captured, the post-hook output at every target layer must be the
# hook's own expression on the pre-write state it was handed,
#
#     h  ==  z + cast_to_stream_dtype( u(z) )        u = the fp32 write,
#
# to within the final add's rounding and NOTHING else. Both addends are
# stream-dtype values, so ``z.float() + cast(u).float()`` is that sum
# exactly; the runtime's stream-dtype add is the same sum correctly rounded;
# hence per element |h − expected| <= half an ulp of the stream dtype in the
# binade of |expected| (equality on ties). On a float32 stream the runtime
# performs the identical fp32 add and the equality is bitwise. The tolerance
# is therefore PER ELEMENT and absolute — never relative to the write — and
# a one-ulp foreign edit of a single element is a violation.


class HookContractError(TeacherPrecheckError):
    """An object's RUNTIME hook did not produce ``z + cast(u(z))`` on the
    pre-write state it was handed (:func:`hook_contract_stats`): the
    algebra (``paired_write``) may be right, but what apply() serves — and
    what a capture under it would record as a distillation target — is not
    the declared write."""


def _half_ulp(x: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Per element: HALF the spacing of ``dtype``'s representable numbers in
    the binade of |x| (``x`` in float32) — the largest error a correctly
    rounded result in ``dtype`` can carry. ``frexp`` is exact (no log2
    rounding at a power of two); zero / subnormal magnitudes take half the
    smallest subnormal step."""
    fi = torch.finfo(dtype)
    mag = x.detach().abs().float()
    _m, e = torch.frexp(mag)                      # mag = m·2^e, m ∈ [0.5, 1)
    ulp = torch.ldexp(torch.full_like(mag, fi.eps), e - 1)   # eps·2^floor(log2 mag)
    tiny = torch.full_like(mag, fi.smallest_normal * fi.eps)
    ulp = torch.where(mag < fi.smallest_normal, tiny, ulp)
    return 0.5 * ulp


def hook_contract_stats(z: torch.Tensor, h: torch.Tensor, u: torch.Tensor,
                        valid: torch.Tensor) -> dict:
    """The contract at ONE layer on ONE forward: ``z`` the pre-write state
    the hook was handed, ``h`` the actual post-hook output (both in the
    stream dtype, [B, S, d]), ``u`` the fp32 write evaluated off the forward
    on that z, ``valid`` the [B, S] non-pad mask. Expected = ``z + cast(u)``
    in exact fp32; tolerance = half an ulp of the stream dtype at
    |expected|, per element (the final add's rounding only — see the section
    comment). Returns raw counts/maxima; :func:`_finish_contract` derives
    the verdict. ``n_write_active_positions`` says where the check has
    teeth (a closed gate writes nothing and cannot expose a wrong hook)."""
    dtype = z.dtype
    zv, hv, uv = z[valid], h[valid], u[valid]                 # [n, d]
    expected = zv.float() + uv.to(dtype).float()
    err = (hv.float() - expected).abs()
    tol = _half_ulp(expected, dtype)
    n = int(err.numel())                     # shape arithmetic: no sync
    # The maxima and counts stay ON THE DEVICE as 0-dim tensors: this runs
    # inside a forward hook on every target layer of every teacher forward
    # (the trainer's hot path), and materialising four python scalars here
    # was four host/device syncs per layer per forward. The reductions
    # themselves still happen on the very z / h of this forward — only the
    # transfer is deferred to :func:`_finish_contract` /
    # :func:`_violation_counts`, once per check.
    if n:
        max_abs = err.max()
        max_half = (err / tol).max()
        n_viol = (err > tol).sum()
        n_active = (uv != 0).any(-1).sum()
    else:
        max_abs = max_half = torch.zeros((), dtype=torch.float32,
                                         device=z.device)
        n_viol = n_active = torch.zeros((), dtype=torch.int64,
                                        device=z.device)
    return {
        "max_abs_err": max_abs,
        "max_err_half_ulps": max_half,
        "n_violations": n_viol,
        "n_elements": n,
        "n_positions": int(zv.shape[0]),
        "n_write_active_positions": n_active,
        "stream_dtype": str(dtype).replace("torch.", ""),
    }


#: The raw-stats fields :func:`hook_contract_stats` keeps as device tensors.
_CONTRACT_MAX_FIELDS = ("max_abs_err", "max_err_half_ulps")
_CONTRACT_SUM_FIELDS = ("n_violations", "n_write_active_positions")
_CONTRACT_INT_FIELDS = ("n_elements", "n_positions")


def _t(v, like):
    """``v`` as a tensor on ``like``'s device (a python number from a
    hand-built stats dict rides along with the device tensors)."""
    if isinstance(v, torch.Tensor):
        return v
    dev = like.device if isinstance(like, torch.Tensor) else None
    return torch.as_tensor(v, device=dev)


def _py(v):
    """One host transfer for a 0-dim tensor; a python number unchanged."""
    return v.item() if isinstance(v, torch.Tensor) else v


def _merge_contract(acc: dict | None, st: dict) -> dict:
    """Aggregate raw stats across batches ON THE DEVICE (no sync)."""
    if acc is None:
        return dict(st)
    out = dict(acc)
    for k in _CONTRACT_MAX_FIELDS:
        out[k] = torch.maximum(_t(acc[k], st[k]), _t(st[k], acc[k]))
    for k in _CONTRACT_SUM_FIELDS:
        out[k] = _t(acc[k], st[k]) + _t(st[k], acc[k])
    for k in _CONTRACT_INT_FIELDS:
        out[k] = int(acc[k]) + int(st[k])
    return out


def _empty_contract() -> dict:
    return {"max_abs_err": 0.0, "max_err_half_ulps": 0.0, "n_violations": 0,
            "n_elements": 0, "n_positions": 0, "n_write_active_positions": 0,
            "stream_dtype": None}


def _finish_contract(acc: dict | None) -> dict:
    """The verdict from raw stats — the ONE place the device numbers become
    python scalars (called once per layer at the END of a check, never
    inside a hook). Accepts python numbers too (hand-built dicts)."""
    if acc is None:
        acc = _empty_contract()
    acc = {**acc, **{k: float(_py(acc[k])) for k in _CONTRACT_MAX_FIELDS},
           **{k: int(_py(acc[k])) for k in _CONTRACT_SUM_FIELDS}}
    dt = acc["stream_dtype"]
    return {
        **acc,
        "write_active_frac": acc["n_write_active_positions"]
        / max(acc["n_positions"], 1),
        "tolerance": f"per element: half an ulp of the {dt} stream at "
                     "|z + cast(u)| — the final add's rounding only, never "
                     "relative to the write",
        "pass": acc["n_violations"] == 0,
    }


def _violation_counts(stats: dict[int, dict]) -> dict[int, int]:
    """Every layer's ``n_violations`` as python ints in ONE host transfer
    (a stacked ``tolist``), for the per-forward pass/fail decision."""
    layers = list(stats)
    if not layers:
        return {}
    counts = torch.stack([_t(stats[l]["n_violations"],
                             stats[l]["n_violations"]).reshape(())
                          for l in layers])
    return {l: int(c) for l, c in zip(layers, counts.tolist())}


def _contract_problem(layer: int, label: str, st: dict) -> str:
    return (f"layer {layer}: {label} HOOK CONTRACT violated — the post-hook "
            f"output != z + cast(u(z)) at {st['n_violations']} of "
            f"{st['n_elements']} elements (max |err| {st['max_abs_err']:.3e} = "
            f"{st['max_err_half_ulps']:.1f} half-ulps of the "
            f"{st['stream_dtype']} stream; tolerance half an ulp per element; "
            f"write active on {100 * st['write_active_frac']:.0f}% of "
            "positions)")


def _checked_capture(model, cb: CapBatch, obj, layers
                     ) -> tuple[dict[int, torch.Tensor], dict[int, dict]]:
    """One NO-GRAD forward under ``obj.apply(model, scale=1.0)`` with every
    ``layers`` output captured AND the hook contract measured on the very
    pre-write state each capture came from: per layer a hook registered
    BEFORE the object's stashes z, a hook registered AFTER it reads the
    post-hook h, evaluates ``_fp32_write(obj, l, z)`` and the contract
    stats, and drops z — so the extra memory is one layer's z at a time
    and the z the check validated IS the z that produced the captured h.
    Returns ({layer: h}, {layer: raw stats}); never raises on a violation
    (callers decide — the trainer refuses, the identity check reports)."""
    _require_off_forward_write(obj, "captured object")
    valid = cb.batch.attention_mask.bool()
    dec = get_decoder_layers(model)
    pre: dict[int, torch.Tensor] = {}
    post: dict[int, torch.Tensor] = {}
    stats: dict[int, dict] = {}

    def make_pre(l):
        def hook(_mod, _args, output):
            pre[l] = output[0] if isinstance(output, tuple) else output
        return hook

    def make_post(l):
        def hook(_mod, _args, output):
            h = output[0] if isinstance(output, tuple) else output
            z = pre.pop(l)
            u = _fp32_write(obj, l, z)
            stats[l] = hook_contract_stats(z, h, u, valid)
            del z, u
            post[l] = h.detach()
        return hook

    handles = [dec[l].register_forward_hook(make_pre(l)) for l in layers]
    try:
        with torch.no_grad(), obj.apply(model, scale=1.0):
            handles += [dec[l].register_forward_hook(make_post(l))
                        for l in layers]
            model(input_ids=cb.batch.input_ids,
                  attention_mask=cb.batch.attention_mask)
    finally:
        for hd in handles:
            hd.remove()
    missing = [l for l in layers if l not in post]
    if missing:
        raise RuntimeError(f"checked capture: layers {missing} never ran")
    return post, stats


def checked_teacher_capture(model, cb: CapBatch, teacher, layers, *,
                            where: str = "teacher capture"
                            ) -> dict[int, torch.Tensor]:
    """The trainer's teacher forward: :func:`_checked_capture`, refusing
    with :class:`HookContractError` (a ``TeacherPrecheckError``) the moment
    any captured layer violates the hook contract — every distillation
    target this module produces is validated on the z it was produced
    from, on every forward, not on a reference batch once."""
    post, stats = _checked_capture(model, cb, teacher, layers)
    viol = _violation_counts(stats)          # one transfer after the forward
    bad = {l: _finish_contract(st) for l, st in stats.items() if viol[l]}
    if bad:
        raise HookContractError(
            f"{where}: the teacher's runtime apply() is not its declared "
            "write — refusing to capture distillation targets from it:\n  - "
            + "\n  - ".join(_contract_problem(l, "teacher", st)
                            for l, st in sorted(bad.items())))
    return post


def hook_contract_problems(model, batches: list[CapBatch], *, obj, layers,
                           label: str = "object") -> tuple[list[str], dict]:
    """The hook contract of ``obj`` alone, over ``batches``: per layer the
    merged stats (:func:`_finish_contract`) and a problem line per failing
    layer. This is the check the persisted-and-reloaded teacher V gets
    before production (the identity check's planted composite is not V)."""
    _require_off_forward_write(obj, label)
    acc: dict[int, dict | None] = {l: None for l in layers}
    for cb in batches:
        _post, stats = _checked_capture(model, cb, obj, layers)
        for l in layers:
            acc[l] = _merge_contract(acc[l], stats[l])
        del _post, stats
    report = {int(l): _finish_contract(acc[l]) for l in layers}
    problems = [_contract_problem(l, label, st) for l, st in report.items()
                if not st["pass"]]
    return problems, report


def _pre_and_post_capture(model, cb: CapBatch, ctx, layers):
    """One NO-GRAD forward under ``ctx`` with BOTH the pre-write layer
    output z (capture hooks registered BEFORE the object's, so they see
    the module's own output) and the post-write output h (registered
    AFTER) captured: ``{layer: (z, h)}``. Forward hooks fire in
    registration order, each handed the previous hook's return value, and
    the write hooks return a NEW tensor, so the pre-write capture is never
    mutated by the add."""
    with torch.no_grad(), _capture_hidden(model, layers) as pre, ctx, \
            _capture_hidden(model, layers) as post:
        model(input_ids=cb.batch.input_ids,
              attention_mask=cb.batch.attention_mask)
        return {l: (pre[l].detach(), post[l].detach()) for l in layers}


def teacher_identity_problems(model, batches: list[CapBatch], *,
                              composite, reference, layers,
                              rtol: float = IDENTITY_RTOL,
                              atol: float = IDENTITY_ATOL,
                              ) -> tuple[list[str], dict]:
    """The exact-identity check — TWO independent
    halves, both gating, per layer:

    (1) the WRITE identity (algebra): on the pre-write state z the
    composite's own forward actually presented to its hook, the composite's
    write (e.g. R(mix ⊕ C, mix) = ``[(+1, mix), (+1, C), (−1, mix)]`` →
    u_mix(z) + u_C(z) − u_mix(z)) must equal the reference's write (u_C(z))
    — both evaluated in fp32 off the forward with the hooks' own
    arithmetic. The RMS over valid positions of ‖u_comp(z) − u_ref(z)‖ must
    sit within ``max(atol, rtol · cancellation scale)``, the scale being
    the largest per-layer write RMS among the composite's signed terms and
    the reference (fp32 cancellation error is proportional to the cancelled
    terms, not to the result).

    (2) the HOOK CONTRACT (runtime), for the composite AND the reference:
    on the same captured forwards, the actual post-hook output must equal
    ``z + cast_to_stream_dtype(u(z))`` at every target layer, per element,
    to within half an ulp of the stream dtype at |z + cast(u)| — the final
    add's rounding, nothing relative to the write
    (:func:`hook_contract_stats`). Half (1) alone re-derives both writes
    with the same helper the implementation uses and cannot see apply(): an
    object whose paired_write() is right but whose runtime no-ops, doubles,
    mis-scales or mis-places its write passed it. Half (2) is what catches
    that.

    Why the write and not the trajectory: the two TRAJECTORIES are two
    bf16 forwards that differ by ~1 fp32 ulp in the added write; in a bf16
    network that seed is amplified chaotically (measured on gemma-4-E4B:
    5e-8 at layer 0 → 3.5e-5 → 2e-2 → saturating ~2–3 per position by
    layer 14, R² 0.97–0.99 against the divergence a single 1-ulp flip of
    ONE stream element produces), so a trajectory comparison against a
    write-scaled tolerance fails on any model with a non-trivial bf16 noise
    floor no matter how exact the teacher is. The trajectory numbers are
    still measured and reported per layer as DIAGNOSTICS
    (``trajectory_rms_divergence`` — the model's bf16 chaos floor on these
    batches; ``trajectory_rms_reference_write``) but never gate. (The hook
    contract is immune to that chaos by construction: it compares each
    forward with ITSELF — h against z + cast(u(z)) on the same forward —
    never one forward with another.)

    Returns (problems, per-layer report); an empty list is a pass. Per
    layer the report carries ``write_identity_pass`` (half 1),
    ``hook_contract_composite`` / ``hook_contract_reference`` (half 2, each
    with ``max_err_half_ulps``, ``n_violations``, ``write_active_frac``,
    ``pass``) and ``pass`` = all three."""
    _require_off_forward_write(composite, "composite")
    _require_off_forward_write(reference, "reference")
    # Per layer, one [n_valid] chunk of per-position squared distances per
    # batch, reduced with _rms_over_positions — the one-table formula bit
    # for bit (rows reduce independently), as the trajectory version did.
    terms = list(getattr(composite, "terms", None) or [(1.0, composite)])
    sq_wdiff: dict[int, list[torch.Tensor]] = {l: [] for l in layers}
    sq_wref: dict[int, list[torch.Tensor]] = {l: [] for l in layers}
    sq_terms: dict[int, list[list[torch.Tensor]]] = {
        l: [[] for _ in terms] for l in layers}
    sq_tdiv: dict[int, list[torch.Tensor]] = {l: [] for l in layers}
    sq_twrite: dict[int, list[torch.Tensor]] = {l: [] for l in layers}
    z_sq = {l: 0.0 for l in layers}
    n_elem = {l: 0 for l in layers}
    cc_acc: dict[int, dict | None] = {l: None for l in layers}
    rc_acc: dict[int, dict | None] = {l: None for l in layers}
    for cb in batches:
        valid = cb.batch.attention_mask.bool()
        comp = _pre_and_post_capture(model, cb, composite.apply(model), layers)
        # the reference's forward: post-hook h for the trajectory
        # diagnostics AND its own hook contract on its own pre-write state
        h_ref, ref_stats = _checked_capture(model, cb, reference, layers)
        h_bare = _forward_capture(model, cb, nullcontext(), layers)
        with torch.no_grad():
            for l in layers:
                z, h_c = comp[l]
                u_c_full = _fp32_write(composite, l, z)
                cc_acc[l] = _merge_contract(
                    cc_acc[l], hook_contract_stats(z, h_c, u_c_full, valid))
                rc_acc[l] = _merge_contract(rc_acc[l], ref_stats[l])
                u_c = u_c_full[valid]
                u_r = _fp32_write(reference, l, z)[valid]
                sq_wdiff[l].append((u_c - u_r).pow(2).sum(-1).cpu())
                sq_wref[l].append(u_r.pow(2).sum(-1).cpu())
                for i, (c, obj) in enumerate(terms):
                    u_i = c * _fp32_write(obj, l, z)[valid]
                    sq_terms[l][i].append(u_i.pow(2).sum(-1).cpu())
                sq_tdiv[l].append(
                    (h_c.float() - h_ref[l].float())[valid].pow(2).sum(-1).cpu())
                sq_twrite[l].append(
                    (h_ref[l].float() - h_bare[l].float())[valid]
                    .pow(2).sum(-1).cpu())
                zf = z[valid].float()
                z_sq[l] += float(zf.pow(2).sum())
                n_elem[l] += int(zf.numel())
                del u_c, u_r, u_c_full, zf
        del comp, h_ref, h_bare, ref_stats
    problems, report = [], {}
    for l in layers:
        wdiff = _rms_over_positions(sq_wdiff[l])
        wref = _rms_over_positions(sq_wref[l])
        scale = max([wref] + [_rms_over_positions(ch) for ch in sq_terms[l]])
        tol = max(atol, rtol * scale)
        tdiv = _rms_over_positions(sq_tdiv[l])
        twrite = _rms_over_positions(sq_twrite[l])
        write_ok = wdiff <= tol
        cc = _finish_contract(cc_acc[l])
        rc = _finish_contract(rc_acc[l])
        report[int(l)] = {
            "rms_write_discrepancy": wdiff,
            "rms_reference_write": wref,
            "rms_cancellation_scale": scale,
            "tolerance": tol,
            "write_identity_pass": write_ok,
            "hook_contract_composite": cc,
            "hook_contract_reference": rc,
            "pass": write_ok and cc["pass"] and rc["pass"],
            "trajectory_rms_divergence": tdiv,
            "trajectory_rms_reference_write": twrite,
            "stream_rms_per_elem": math.sqrt(z_sq[l] / max(n_elem[l], 1)),
        }
        if not write_ok:
            problems.append(
                f"layer {l}: composite-vs-reference WRITE discrepancy "
                f"{wdiff:.3e} > tolerance {tol:.3e} (reference write RMS "
                f"{wref:.3e}, cancellation scale {scale:.3e}; trajectory "
                f"divergence {tdiv:.3e} is diagnostic only)")
        if not cc["pass"]:
            problems.append(_contract_problem(l, "composite", cc))
        if not rc["pass"]:
            problems.append(_contract_problem(l, "reference", rc))
    return problems, report


def stream_perturbation_floor(model, batches: list[CapBatch], *, layers,
                              ctx_factory=nullcontext, row: int = 0,
                              position: int = 3, element: int = 0,
                              ) -> dict[int, float]:
    """The model's bf16 chaotic-amplification floor on ``batches``: per
    layer, the RMS over valid positions of the per-position L2 distance
    between a forward under ``ctx_factory()`` and the SAME forward with ONE
    ulp (of the stream's own dtype) added to a single stream element —
    ``[row, position, element]`` of the first listed layer's output — the
    smallest perturbation a rounding difference can make. Composition-
    independent (unlike :func:`numeric_floor`, which is identically zero at
    batch_size=1), and the number the old trajectory-level identity check
    was actually measuring (R² 0.97–0.99 against it on both models).
    Reported beside the identity check; never a gate here."""
    first = min(layers)
    dec = get_decoder_layers(model)

    def flip(_mod, _args, output):
        hid = output[0] if isinstance(output, tuple) else output
        r = min(row, hid.shape[0] - 1)
        p = min(position, hid.shape[1] - 1)
        e = min(element, hid.shape[2] - 1)
        h2 = hid.clone()
        ity = _INT_VIEW.get(hid.dtype)
        if ity is None:
            raise TypeError(f"stream_perturbation_floor: no ulp step for "
                            f"{hid.dtype}")
        bumped = (h2[r, p, e].reshape(1).view(ity) + 1).view(hid.dtype)
        h2[r, p, e] = bumped.reshape(())
        return (h2,) + tuple(output[1:]) if isinstance(output, tuple) else h2

    n_pos = 0
    sq = {l: 0.0 for l in layers}
    for cb in batches:
        valid = cb.batch.attention_mask.bool()
        n_pos += int(valid.sum())
        h_a = _forward_capture(model, cb, ctx_factory(), layers)
        handle = dec[first].register_forward_hook(flip)
        try:
            h_b = _forward_capture(model, cb, ctx_factory(), layers)
        finally:
            handle.remove()
        for l in layers:
            sq[l] += float((h_a[l].float() - h_b[l].float())[valid]
                           .pow(2).sum())
        del h_a, h_b
    return {int(l): math.sqrt(sq[l] / max(n_pos, 1)) for l in layers}


def arithmetic_residual_inconsistency(model, batches: list[CapBatch], *,
                                      pos, neg, paired, layers) -> dict:
    """The labeled DIAGNOSTIC: the RMS distance between the
    coherent trajectory h^V and the arithmetic residual T_arith = h_bare +
    (h_pos − h_bare) − (h_neg − h_bare) built from three separate forwards,
    per layer, next to RMS(h^V − h_bare) for scale. Never a gate."""
    ctxs = {"paired": lambda: paired.apply(model),
            "pos": lambda: pos.apply(model),
            "neg": lambda: neg.apply(model),
            "bare": nullcontext}
    sq_inc: dict[int, list[torch.Tensor]] = {l: [] for l in layers}
    sq_dev: dict[int, list[torch.Tensor]] = {l: [] for l in layers}
    for h in _valid_hidden_per_batch(model, batches, ctxs, layers):
        for l in layers:
            h_0 = h["bare"][l]
            t_arith = h_0 + (h["pos"][l] - h_0) - (h["neg"][l] - h_0)
            sq_inc[l].append((h["paired"][l] - t_arith).pow(2).sum(-1))
            sq_dev[l].append((h["paired"][l] - h_0).pow(2).sum(-1))
    out = {}
    for l in layers:
        inc = _rms_over_positions(sq_inc[l])
        dev = _rms_over_positions(sq_dev[l])
        out[int(l)] = {"rms_inconsistency": inc, "rms_teacher_deviation": dev,
                       "ratio": (inc / dev) if dev > 0 else None}
    return out


# --------------------------------------------------------------------------- #
# The trainer                                                                  #
# --------------------------------------------------------------------------- #


def _group_denominators(cbs: list[CapBatch]) -> dict[tuple[str, str], int]:
    tot: dict[tuple[str, str], int] = {}
    for cb in cbs:
        for k, n in cb.n_pos.items():
            tot[k] = tot.get(k, 0) + n
    return tot


def residual_loss_terms(sq: dict[int, torch.Tensor], cb: CapBatch,
                        denominators: dict[tuple[str, str], int], *,
                        d_model: int, source_weights: dict[str, float],
                        ) -> tuple[torch.Tensor, dict[tuple[str, str], float]]:
    """This micro-batch's contribution to the group loss: Σ_layers
    Σ_(source, cls) w_source · SSE / (N_group · d). Returns (loss tensor,
    per-bucket unweighted contribution floats for logging)."""
    loss = None
    parts: dict[tuple[str, str], float] = {}
    for (s, cls), mask in cb.masks.items():
        n_group = denominators.get((s, cls), 0)
        if n_group == 0 or cb.n_pos[(s, cls)] == 0:
            continue
        m = mask.to(next(iter(sq.values())).dtype)
        sse = None
        for l, sq_l in sq.items():
            t = (sq_l * m).sum()
            sse = t if sse is None else sse + t
        term = sse / (n_group * d_model)
        parts[(s, cls)] = float(term.detach())
        w = source_weights[s]
        wt = w * term
        loss = wt if loss is None else loss + wt
    if loss is None:
        loss = torch.zeros((), device=cb.batch.input_ids.device,
                           requires_grad=False)
    return loss, parts


def train_psr_residual(
    model,
    tok,
    student: PSRObject,
    teacher,
    train_docs: list[CaptureDoc],
    holdout_docs: list[CaptureDoc],
    *,
    lr: float,
    epochs: int = 24,
    batch_size: int = 1,
    accum_steps: int = 8,
    max_len: int = 8192,
    weight_decay: float = 0.01,
    reg_coeff: float = 0.1,
    lr_schedule: str | None = None,
    warmup_steps: int | None = None,
    warmup_frac: float | None = None,
    max_grad_norm: float | None = None,
    seed: int = 0,
    source_weights: dict[str, float] | None = None,
    log_fn=None,
    log_tag: str = "psr_residual",
    early_stop: EarlyStopSpec | None = None,
    checkpoint_every: int | None = None,
    checkpoint_dir=None,
    checkpoint_key: str | None = None,
    diagnostics: dict | None = None,
    teacher_precheck: Callable[[], list[str]] | None = None,
    n_boot: int = DEFAULT_N_BOOT,
    boot_seed: int = 0,
) -> list[float]:
    """Distill ``teacher``'s closed-loop trajectory into ``student`` (a
    zero-init all-layer ``PSRObject``, trained in place). Returns the
    per-optimizer-step loss history. See the module docstring for the
    objective; the loop conventions are ``train_object_ce``'s:
    length-sorted batching, accumulation with exact group-mean weighting,
    warmup-cosine + clipping, per-epoch held-out early stopping with the
    unconditional best-epoch restore, non-finite breakers
    (``fitmon.TrainingDiverged``), and group-boundary checkpoints carrying
    the FULL trainer state (``expect={"trainer": "train_psr_residual"}``).

    ``holdout_docs`` (required, ≥ 1 doc) rule the early stopping on the
    pooled held-out loss (the training objective without the regularizer)
    and carry the ED table: ``diagnostics["ed"]`` is computed from the
    RESTORED best-epoch weights (the object that ships), with per-doc
    records (``diagnostics["doc_records"]``) for any re-analysis.
    ``teacher_precheck`` (a callable returning problem strings) runs
    BEFORE any optimization; a non-empty list raises
    :class:`TeacherPrecheckError`. Independently of it, EVERY teacher
    forward (held-out reference, per-epoch held-out, every training
    micro-batch) is a :func:`checked_teacher_capture`: the hook contract
    ``h == z + cast(u(z))`` is verified on the z each target came from, and
    a violation raises :class:`HookContractError` before any target is
    used. ``warmup_frac`` resolves ``warmup_steps`` against the
    optimizer-step budget when given."""
    from .fitmon import TrainingDiverged

    if not isinstance(student, PSRObject):
        raise TypeError("student must be a PSRObject")
    if not student.train_direction:
        raise ValueError(
            "student must be constructed with train_direction=True — the "
            "residual's direction is the thing being distilled")
    if student.fit_positions != "all" or student._needs_tracking():
        raise ValueError("student must be a plain all-position PSRObject")
    if not holdout_docs:
        raise ValueError(
            "train_psr_residual requires holdout docs — the early stopping "
            "and the recovery gate both rule on the capture holdout")
    if not train_docs:
        raise ValueError("no training docs")
    source_weights = dict(source_weights or DEFAULT_SOURCE_WEIGHTS)
    for d in list(train_docs) + list(holdout_docs):
        if d.source not in source_weights:
            raise ValueError(
                f"doc {d.doc_id} has source {d.source!r} with no weight in "
                f"source_weights={source_weights}")
    epochs = int(epochs)
    if epochs < 1:
        raise ValueError(f"epochs must be >= 1, got {epochs!r}")
    accum_steps = int(accum_steps)
    if accum_steps < 1:
        raise ValueError(f"accum_steps must be >= 1, got {accum_steps!r}")
    if warmup_steps is not None and warmup_frac is not None:
        raise ValueError("pass warmup_steps OR warmup_frac, not both")
    checkpoint_every = resolve_checkpoint_kwargs(
        checkpoint_every, checkpoint_dir, checkpoint_key)
    if lr_schedule is None:
        lr_schedule = "warmup_cosine"
    if lr_schedule not in ("warmup_cosine", "constant"):
        raise ValueError(f"lr_schedule must be warmup_cosine|constant, got "
                         f"{lr_schedule!r}")
    if max_grad_norm is None:
        max_grad_norm = 1.0

    if teacher_precheck is not None:
        problems = list(teacher_precheck() or [])
        if problems:
            raise TeacherPrecheckError(
                "teacher failed its pre-optimization identity check — "
                "refusing to distill it:\n  - " + "\n  - ".join(problems))

    device = next(model.parameters()).device
    rng = random.Random(seed)
    torch.manual_seed(seed)
    for p in model.parameters():
        p.requires_grad_(False)
    student.to_(device)
    if hasattr(teacher, "to_"):
        teacher.to_(device)
    opt = torch.optim.AdamW(student.trainable, lr=lr,
                            weight_decay=weight_decay)
    layers = psr_loss_layers(student, model)
    d_model = hidden_size(model)

    batches = make_cap_batches(tok, train_docs, batch_size=batch_size,
                               max_len=max_len, device=device,
                               label=f"{log_tag} train",
                               sources=tuple(source_weights))
    h_batches = make_cap_batches(tok, holdout_docs, batch_size=batch_size,
                                 max_len=max_len, device=device,
                                 label=f"{log_tag} held-out",
                                 sources=tuple(source_weights))
    n_rows = [cb.n_rows for cb in batches]
    h_denoms = _group_denominators(h_batches)

    # Bare-vs-teacher reference on the holdout (fixed for the whole fit):
    # the ED denominator and the teacher-deviation number, per doc.
    bare_sq: list[dict[int, torch.Tensor]] = []
    for cb in h_batches:
        t = checked_teacher_capture(model, cb, teacher, layers,
                                    where=f"{log_tag} held-out reference")
        b = _forward_capture(model, cb, nullcontext(), layers)
        bare_sq.append({l: v.cpu() for l, v in _sq_by_layer(b, t, layers).items()})
        del t, b

    def _heldout(records_out: list | None) -> tuple[float, dict]:
        """Pooled held-out loss (the objective without the regularizer)
        + per-bucket means; fills ``records_out`` with per-doc records."""
        parts: dict[tuple[str, str], float] = {}
        with torch.no_grad():
            for i, cb in enumerate(h_batches):
                t = checked_teacher_capture(model, cb, teacher, layers,
                                            where=f"{log_tag} held-out")
                s = _forward_capture(model, cb,
                                     student.apply(model, scale=1.0), layers)
                sq = _sq_by_layer(s, t, layers)
                _loss, p = residual_loss_terms(
                    sq, cb, h_denoms, d_model=d_model,
                    source_weights=source_weights)
                for k, v in p.items():
                    parts[k] = parts.get(k, 0.0) + v
                if records_out is not None:
                    records_out.extend(doc_records_from_sq(
                        cb, {l: v.cpu() for l, v in sq.items()}, bare_sq[i],
                        layers))
                del t, s, sq
        pooled = sum(source_weights[s] * v for (s, _c), v in parts.items())
        if not math.isfinite(pooled):
            raise TrainingDiverged(
                f"{log_tag}: non-finite held-out loss ({pooled!r})")
        return pooled, {f"{s}/{c}": v for (s, c), v in parts.items()}

    total_steps = epochs * math.ceil(len(batches) / accum_steps)
    if warmup_frac is not None:
        warmup_steps = max(1, int(round(warmup_frac * total_steps)))
    if warmup_steps is None:
        warmup_steps = min(500, max(1, total_steps // 10))
    from .lockstep import _lr_at_factory
    _lr_at = _lr_at_factory(lr, lr_schedule, warmup_steps, total_steps)

    stopper = EarlyStopper(early_stop or EarlyStopSpec(patience=2))
    heldout_curve: list[float] = []
    history: list[float] = []
    step = 0
    start_epoch = 0
    resume_order: list[int] | None = None
    resume_batches_done = 0
    ckpt_last_step = 0
    if checkpoint_every is not None:
        ck = load_checkpoint(checkpoint_dir, checkpoint_key,
                             expect={"trainer": "train_psr_residual",
                                     "n_batches": len(batches)})
        if ck is not None:
            with torch.no_grad():
                for p, saved in zip(student.trainable, ck["trainable"]):
                    p.data.copy_(saved.to(p.data.device))
            opt.load_state_dict(ck["opt_state"])
            rng.setstate(ck["rng_state"])
            torch.set_rng_state(ck["torch_rng_state"])
            restore_cuda_rng(ck.get("cuda_rng_states"))
            history = list(ck["history"])
            step = int(ck["step"])
            start_epoch = int(ck["epoch"])
            resume_order = [int(b) for b in ck["order"]]
            resume_batches_done = int(ck["batches_done"])
            ckpt_last_step = step
            heldout_curve = [float(v) for v in ck.get("heldout_curve") or []]
            st = ck.get("stopper")
            if st is not None:
                stopper.best = st["best"]
                stopper.best_epoch = st["best_epoch"]
                stopper.epochs_seen = st["epochs_seen"]
                stopper._since_best = st["since_best"]
                stopper._snapshot = st["snapshot"]
            print(f"[fitckpt] train_psr_residual RESUMING from checkpoint: "
                  f"epoch {start_epoch}, micro-batch {resume_batches_done}/"
                  f"{len(batches)} in epoch, optimizer step {step}",
                  flush=True)
    if not heldout_curve:
        init_loss, init_parts = _heldout(None)
        heldout_curve.append(init_loss)
        if log_fn:
            log_fn({f"{log_tag}/heldout_loss_init": init_loss,
                    **{f"{log_tag}/heldout_init/{k}": v
                       for k, v in init_parts.items()}}, 0)
    print(f"[{log_tag}] lr={lr} schedule={lr_schedule} warmup_steps="
          f"{warmup_steps} max_grad_norm={max_grad_norm} accum_steps="
          f"{accum_steps} epochs(cap)={epochs} loss_layers={len(layers)} "
          f"train_docs={len(train_docs)} holdout_docs={len(holdout_docs)} "
          f"source_weights={source_weights}", flush=True)

    epochs_run = start_epoch
    stopped_early = False
    for epoch in range(start_epoch, epochs):
        if resume_order is not None and epoch == start_epoch:
            order, skip = resume_order, resume_batches_done
        else:
            order = list(range(len(batches)))
            rng.shuffle(order)
            skip = 0
        for g0 in range(skip, len(order), accum_steps):
            group = order[g0:g0 + accum_steps]
            denoms = _group_denominators([batches[b] for b in group])
            grp_rows = sum(n_rows[b] for b in group)
            acc_loss = acc_reg = 0.0
            acc_parts: dict[str, float] = {}
            for k, bi in enumerate(group):
                cb = batches[bi]
                # the target trajectory, with the hook contract verified on
                # the very z it came from: a teacher whose runtime is not
                # its declared write never yields a target
                t_hidden = checked_teacher_capture(
                    model, cb, teacher, layers, where=f"{log_tag} train")
                # Forward + loss + BACKWARD inside apply()/_capture_hidden
                # (gradient checkpointing recompute must see the hooks).
                with student.apply(model, scale=1.0), \
                        _capture_hidden(model, layers) as s_cap:
                    model(input_ids=cb.batch.input_ids,
                          attention_mask=cb.batch.attention_mask)
                    s_hidden = {l: s_cap[l] for l in layers}
                    sq = _sq_by_layer(s_hidden, t_hidden, layers)
                    mse_term, parts = residual_loss_terms(
                        sq, cb, denoms, d_model=d_model,
                        source_weights=source_weights)
                    step_loss = mse_term
                    reg = None
                    if reg_coeff:
                        reg = student.gate_reg_term(cb.batch.attention_mask,
                                                    None)
                        w_reg = n_rows[bi] / grp_rows
                        step_loss = step_loss + reg_coeff * (w_reg * reg)
                    if k == 0:
                        opt.zero_grad()
                    if step_loss.requires_grad:
                        step_loss.backward()
                acc_loss += float(step_loss.detach())
                for kk, v in parts.items():
                    acc_parts[f"{kk[0]}/{kk[1]}"] = acc_parts.get(
                        f"{kk[0]}/{kk[1]}", 0.0) + v
                if reg is not None:
                    acc_reg += float(reg.detach()) * (n_rows[bi] / grp_rows)
                del t_hidden, s_hidden, sq
            grad_norm_t = None
            if math.isfinite(max_grad_norm):
                grad_norm_t = torch.nn.utils.clip_grad_norm_(
                    student.trainable, max_grad_norm)
            cur_lr = _lr_at(step)
            for grp in opt.param_groups:
                grp["lr"] = cur_lr
            opt.step()
            step += 1
            history.append(acc_loss)
            if not math.isfinite(acc_loss):
                raise TrainingDiverged(
                    f"{log_tag}: non-finite training loss ({acc_loss!r}) at "
                    f"optimizer step {step} — never train through NaN.")
            if log_fn:
                rec = {f"{log_tag}/loss": acc_loss, f"{log_tag}/lr": cur_lr,
                       "epoch": epoch}
                for kk, v in acc_parts.items():
                    rec[f"{log_tag}/mse/{kk}"] = v
                if reg_coeff:
                    rec[f"{log_tag}/gate_reg"] = acc_reg
                if grad_norm_t is not None:
                    rec[f"{log_tag}/grad_norm"] = float(grad_norm_t)
                log_fn(rec, step)
            if (checkpoint_every is not None
                    and step - ckpt_last_step >= checkpoint_every):
                save_checkpoint(checkpoint_dir, {
                    "checkpoint_key": checkpoint_key,
                    "trainer": "train_psr_residual",
                    "n_batches": len(batches),
                    "epoch": epoch,
                    "batches_done": g0 + len(group),
                    "order": list(order),
                    "rng_state": rng.getstate(),
                    "step": step,
                    "history": list(history),
                    "trainable": [p.detach().cpu().clone()
                                  for p in student.trainable],
                    "opt_state": to_cpu(opt.state_dict()),
                    "torch_rng_state": torch.get_rng_state(),
                    "cuda_rng_states": cuda_rng_states(),
                    "heldout_curve": list(heldout_curve),
                    "stopper": {
                        "best": stopper.best,
                        "best_epoch": stopper.best_epoch,
                        "epochs_seen": stopper.epochs_seen,
                        "since_best": stopper._since_best,
                        "snapshot": (None if stopper._snapshot is None else
                                     [t.detach().cpu().clone()
                                      for t in stopper._snapshot]),
                    },
                })
                ckpt_last_step = step
                print(f"[fitckpt] train_psr_residual checkpoint written: "
                      f"epoch {epoch}, micro-batch {g0 + len(group)}/"
                      f"{len(batches)} in epoch, optimizer step {step}",
                      flush=True)
        epochs_run = epoch + 1
        hl, hparts = _heldout(None)
        heldout_curve.append(hl)
        if log_fn:
            log_fn({f"{log_tag}/heldout_loss": hl, "epoch": epoch,
                    **{f"{log_tag}/heldout/{k}": v for k, v in hparts.items()}},
                   step)
        if stopper.update(hl, student.trainable):
            stopped_early = True
            if log_fn:
                log_fn({f"{log_tag}/stopped_epoch": epoch + 1,
                        f"{log_tag}/best_epoch": stopper.best_epoch}, step)
            break
    stopper.restore_best(student.trainable)

    # The ED table on the weights that SHIP (post-restore), per doc.
    records: list[dict] = []
    final_loss, final_parts = _heldout(records)
    if diagnostics is not None:
        diagnostics.update({
            "trainer": "train_psr_residual",
            "epochs_run": epochs_run,
            "stopped_early": stopped_early,
            "heldout_curve": list(heldout_curve),
            "heldout_init": heldout_curve[0] if heldout_curve else None,
            "best_epoch": stopper.best_epoch,
            "best_heldout_loss": (stopper.best if math.isfinite(stopper.best)
                                  else None),
            "final_heldout_loss": final_loss,
            "final_heldout_parts": final_parts,
            "optimizer_steps_run": step,
            "steps_per_epoch": math.ceil(len(batches) / accum_steps),
            "loss_layers": [int(l) for l in layers],
            "source_weights": dict(source_weights),
            "position_classes": {
                "prefix": "non-pad positions before the prompt boundary on "
                          "docs with a non-empty prompt (advice system field)",
                "content": "every other non-pad position"},
            "lr": lr, "lr_schedule": lr_schedule,
            "warmup_steps": warmup_steps, "max_grad_norm": max_grad_norm,
            "accum_steps": accum_steps, "batch_size": batch_size,
            "reg_coeff": reg_coeff, "epochs_cap": epochs,
            "early_stop": {"patience": stopper.spec.patience,
                           "min_epochs": stopper.spec.min_epochs,
                           "min_delta": stopper.spec.min_delta},
            "ed": ed_summary(records, layers=layers, n_boot=n_boot,
                             seed=boot_seed),
            "doc_records": records,
        })
    return history


# --------------------------------------------------------------------------- #
# Calibrated recovery gate                                                     #
# --------------------------------------------------------------------------- #


def failure_sentence(ed: float | None, lo: float | None, hi: float | None,
                     attain: float | None) -> str:
    def f(x):
        return "n/a" if x is None else f"{x:.3f}"
    return (f"residual {VERDICT_NOT_RECOVERED} (ED = {f(ed)} [{f(lo)}, "
            f"{f(hi)}] vs attainable {f(attain)})")


def recovery_verdict(production_ed: dict, attainable: dict[str, float | None],
                     floor: dict, *, strata: tuple[str, ...] = SOURCES,
                     cls: str = "content",
                     attain_frac: float = GATE_ATTAIN_FRAC,
                     abs_floor: float = GATE_ABS_FLOOR,
                     floor_mult: float = GATE_UNRESOLVABLE_FLOOR_MULT,
                     ) -> dict:
    """Pre-stated. ``production_ed`` is :func:`ed_summary`'s
    table for the production run; ``attainable[stratum]`` the min over the
    calibration controls' point EDs; ``floor`` :func:`numeric_floor`'s
    ``floor`` table. Per source stratum on the content class:

    - UNRESOLVABLE when the teacher's deviation from bare is below
      ``floor_mult`` × the numeric floor (neither pass nor fail);
    - RECOVERED iff the CI lower bound ≥ ``attain_frac`` · attainable AND
      ≥ ``abs_floor``;
    - otherwise NOT RECOVERED, with the fixed failure sentence.

    Overall: recovered iff EVERY listed stratum is recovered;
    ``degenerate`` when every stratum is unresolvable (the parents are
    functionally identical on the capture distribution). The prefix class
    is advisory and not ruled on here."""
    per = {}
    for s in strata:
        e = (production_ed.get(s) or {}).get(cls) or {}
        ed, lo, hi = e.get("ed"), e.get("lo"), e.get("hi")
        att = attainable.get(s)
        dev = e.get("teacher_deviation")
        fl = ((floor.get(s) or {}).get(cls) or {}).get("mse_sum_layers")
        row = {"ed": ed, "lo": lo, "hi": hi, "attainable": att,
               "teacher_deviation": dev, "floor": fl,
               "threshold": (max(attain_frac * att, abs_floor)
                             if att is not None else abs_floor)}
        if (dev is None or ed is None
                or (fl is not None and dev < floor_mult * fl)):
            row["verdict"] = VERDICT_UNRESOLVABLE
            row["sentence"] = (
                f"stratum {s}: teacher within {floor_mult:g}x the numeric "
                f"floor of bare (deviation {dev!r} vs floor {fl!r}) — "
                "unresolvable; neither pass nor fail")
        elif lo is not None and att is not None and lo >= row["threshold"]:
            row["verdict"] = VERDICT_RECOVERED
            row["sentence"] = (f"stratum {s}: residual recovered (ED = "
                               f"{ed:.3f} [{lo:.3f}, {hi:.3f}] vs attainable "
                               f"{att:.3f})")
        else:
            row["verdict"] = VERDICT_NOT_RECOVERED
            row["sentence"] = f"stratum {s}: " + failure_sentence(ed, lo, hi, att)
        per[s] = row
    verdicts = [r["verdict"] for r in per.values()]
    if verdicts and all(v == VERDICT_UNRESOLVABLE for v in verdicts):
        overall = VERDICT_UNRESOLVABLE
        degenerate = True
    elif verdicts and all(v == VERDICT_RECOVERED for v in verdicts):
        overall = VERDICT_RECOVERED
        degenerate = False
    else:
        overall = VERDICT_NOT_RECOVERED
        degenerate = False
    out = {"gate": {"attain_frac": attain_frac, "abs_floor": abs_floor,
                    "unresolvable_floor_mult": floor_mult,
                    "position_class": cls, "strata": list(strata),
                    "ci_level": GATE_CI_LEVEL},
           "per_stratum": per, "verdict": overall,
           "degenerate_parents_identical": degenerate,
           "p1_p3_rows_are": ("confirmatory" if overall == VERDICT_RECOVERED
                              else "descriptive")}
    assert_no_banned_words(json.dumps(out), where="recovery verdict")
    return out


def attainable_from_controls(control_eds: list[dict], *,
                             strata: tuple[str, ...] = SOURCES,
                             cls: str = "content") -> dict[str, float | None]:
    """ED_attain(stratum) = min over the calibration controls of their
    point ED. A control with no ED for a stratum (teacher ≡
    bare there) is skipped; no control → None."""
    out: dict[str, float | None] = {}
    for s in strata:
        vals = []
        for table in control_eds:
            e = ((table.get(s) or {}).get(cls) or {}).get("ed")
            if e is not None:
                vals.append(float(e))
        out[s] = min(vals) if vals else None
    return out


def write_calibration(path, payload: dict) -> None:
    """``calibration.json`` — refuses to write a banned word."""
    text = json.dumps(payload, indent=1, sort_keys=True, default=str) + "\n"
    assert_no_banned_words(text, where=str(path))
    Path(path).write_text(text, encoding="utf-8")
