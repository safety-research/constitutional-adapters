#!/usr/bin/env python3
"""Standalone held-out cross-entropy scorer for ANY loadable intervention
object (sdf-base-steering round 2).

Round 1 had no such tool: its held-out CE came out of the trainer's own
``_heldout_ce`` inside ``train_object_ce`` and existed only for the fit that
produced it. This script scores an already-saved object — a staged fit, a
derived object (``lora_subtraction`` / ``psr_paired_write`` / a distilled
residual), or a deliberately bare scratch artifact — on any document pool,
at any ``--scale``, ACTIVE vs NO-OP, and keeps PER-DOCUMENT records.

Per-document records are the point. The plan's P3 read ("the derived
object loses the control-corpus CE gain", §2.6) is a paired DOC-LEVEL
bootstrap on ``gain = 1 − CE_obj / CE_bare`` over ~20 control-register
held-out docs; that needs each doc's summed NLL under the object and under
the bare model, not one pooled number. Every record therefore carries
``{doc_id, family, source_tag, n_target_tokens, nll_active_sum,
nll_noop_sum}`` and the aggregate is pure arithmetic on the records
(:func:`aggregate`, :func:`doc_bootstrap_gain`).

The metric is the trainer's, exactly. Held-out CE is the MICRO-average
``sum(token NLL) / sum(target tokens)`` over the whole pool — a ratio of
sums, never a mean of batch means. A ratio of sums is invariant to how the
docs are bucketed into batches, which gives the **parity pin**: on a
fit's own held-out pool at scale 1.0, this scorer must
reproduce the fit's recorded ``best_heldout_ce`` / ``heldout_noop_ce`` to
numeric tolerance even at a different batch size — a free
cross-implementation check, made a test (``tests/test_ce_score.py``). To
keep that pin honest the encoding is the trainer's byte for byte: verbatim
doc-mode examples (``docs_to_ce_examples``: prompt == the loss-masked
prefix, completion == the trained text, no chat template), the student leg
of ``_encode_aligned`` (prompt with specials, completion without, ONE EOS
appended), CE targets ``labels[:, 1:] != -100`` (position 0 is never a
target — the BOS-less empty-prompt edge drops exactly one token), and the
per-example over-length REFUSAL: a row whose prompt + completion + EOS does
not fit ``--max-len`` is refused by doc id, never silently truncated
(``_encode_aligned`` alone would ``comp[:avail]``). Empty completions
refuse too.

The no-op leg is ``nullcontext`` — the exact unmodified base model — not
``scale=0``; the active leg is ``obj.apply(model, scale=--scale)``. Both
legs run on the same collated batch, no-grad, one forward each.

Identity comes first. With ``--object``, the artifact is validated through
``sisteer.eval.object_identity.validate_artifact_identity`` BEFORE any
model loads: a staged cell dir (``train_budget_meta.json``) must name
``--cell`` and hash to its sidecar; a derived dir (``derived_meta.json``)
must prove its parents; an unsidecared dir refuses unless ``--allow-bare``
(scratch objects, tests — never a production default). The validated kind
is recorded in the summary as ``identity_kind``.

Outputs: ``--out`` (per-doc JSONL, sorted by doc_id) and
``<out>.summary.json`` (the sum/sum aggregate plus full provenance: object
and pairs sha256, model/revision, scale, device/dtype, batch size, max_len,
source tag, and a default 90% doc-level bootstrap CI on the gain).
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from contextlib import nullcontext
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_MAX_LEN = 8192
DEFAULT_BATCH_SIZE = 1  # sized for 8k-token docs on an H200; raise for short pools
SOURCE_TAGS = ("mix", "ctl", "heldout")
RECORD_KEYS = ("doc_id", "family", "source_tag", "n_target_tokens",
               "nll_active_sum", "nll_noop_sum")
SUMMARY_KEYS = ("ce_active", "ce_noop", "ratio", "gain", "n_docs", "n_tokens",
                "pool", "scale", "object", "object_sha256", "identity_kind",
                "cell", "pairs", "pairs_sha256", "model", "revision", "device",
                "dtype", "max_len", "batch_size", "source_tag", "limit",
                "consumer", "gain_doc_boot90")
CONSUMER = "ce-score"


# ---------------------------------------------------------------------------
# Pool loading
# ---------------------------------------------------------------------------


def _read_jsonl(path) -> list[dict]:
    """Non-blank JSONL lines as dicts — the ONE reader for this script."""
    return [json.loads(l) for l in
            Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def check_rows(rows: list[dict]) -> list[dict]:
    """Validate pool rows and return them sorted by ``doc_id``. Every row
    needs a ``doc_id`` (unique — the per-doc records are keyed on it, and
    a duplicate would silently double-count a doc in the paired
    bootstrap), a ``prompt`` (may be ``""``) and a ``pos``."""
    seen: set[str] = set()
    out = []
    for i, r in enumerate(rows):
        if not isinstance(r, dict):
            raise ValueError(f"pool row {i} is not a JSON object")
        doc_id = r.get("doc_id")
        if doc_id is None or str(doc_id) == "":
            raise ValueError(f"pool row {i} carries no doc_id — per-doc "
                             "records need one")
        doc_id = str(doc_id)
        if doc_id in seen:
            raise ValueError(f"duplicate doc_id {doc_id!r} in the pool — a "
                             "doc may appear once (paired bootstrap keys "
                             "on it)")
        seen.add(doc_id)
        for k in ("prompt", "pos"):
            if not isinstance(r.get(k), str):
                raise ValueError(
                    f"doc {doc_id!r}: pool row needs a string {k!r} field")
        out.append(r)
    return sorted(out, key=lambda r: str(r["doc_id"]))


def rows_to_examples(rows: list[dict]):
    """Doc-mode DistillExamples, VERBATIM, through the trainer's own
    builder (``docs_to_ce_examples``): prompt_with_si == prompt_without ==
    row["prompt"], completion == row["pos"]."""
    from sisteer.data.pairs import Pair, docs_to_ce_examples

    pairs = [Pair(prompt=r["prompt"], pos=r["pos"],
                  neg=str(r.get("neg") or ""),
                  instruction_id=str(r.get("instruction_id") or ""))
             for r in rows]
    return docs_to_ce_examples(None, pairs)


# ---------------------------------------------------------------------------
# Encoding — the trainer's _encode_checked, by doc id
# ---------------------------------------------------------------------------


def encode_docs(tok, rows: list[dict], *, max_len: int = DEFAULT_MAX_LEN,
                obj=None) -> list:
    """Student-leg encodings (``_encode_aligned(...)[1]``) for every row,
    after the per-example REFUSALS ``train_object_ce._encode_checked``
    applies: an empty completion, or prompt + completion + EOS exceeding
    ``max_len``, raises ``ValueError`` naming the doc — never the silent
    ``comp[:avail]`` truncation. ``obj.inject`` is applied to the student
    prompt exactly as the trainer does (identity for every non-token
    object), so the encoding the parity pin compares is the trainer's."""
    from sisteer.extraction.distill import DistillExample, _encode_aligned
    from sisteer.extraction.tokcache import cached_ids

    examples = rows_to_examples(rows)
    eos_n = 1 if tok.eos_token_id is not None else 0
    out = []
    for row, ex in zip(rows, examples):
        doc_id = str(row["doc_id"])
        if obj is not None and hasattr(obj, "inject"):
            ex = DistillExample(
                prompt_with_si=ex.prompt_with_si,
                prompt_without=obj.inject(ex.prompt_without),
                completion=ex.completion,
                instruction_idx=ex.instruction_idx)
        n_comp = len(cached_ids(tok, ex.completion, False)) + eos_n
        if n_comp - eos_n < 1:
            raise ValueError(
                f"doc {doc_id!r}: EMPTY completion — nothing to score (the "
                "doc sampler asserts non-empty trained text; a pool row "
                "without it is broken)")
        n_prompt = max(len(cached_ids(tok, ex.prompt_with_si, True)),
                       len(cached_ids(tok, ex.prompt_without, True)))
        if n_prompt + n_comp > max_len:
            raise ValueError(
                f"doc {doc_id!r} needs {n_prompt} prompt + {n_comp} "
                f"completion(+EOS) = {n_prompt + n_comp} tokens > "
                f"max_len={max_len} — refusing (no silent truncation); "
                "raise --max-len")
        out.append(_encode_aligned(tok, ex, max_len)[1])
    return out


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def _row_nll(logits, batch):
    """Per-row (summed NLL over CE targets, target count) — the trainer's
    loss identity (``log_softmax(logits[:, :-1].float())`` gathered at
    ``labels[:, 1:]``, masked at -100) kept per row instead of summed over
    the batch."""
    import torch

    logp = torch.log_softmax(logits[:, :-1].float(), dim=-1)
    labels = batch.labels[:, 1:]
    mask = labels != -100
    safe = labels.clamp(min=0)
    tok_lp = logp.gather(-1, safe.unsqueeze(-1)).squeeze(-1)
    nll = (-tok_lp).masked_fill(~mask, 0.0).sum(dim=1)
    return nll, mask.sum(dim=1)


def score_docs(model, tok, rows: list[dict], *, obj=None, scale: float = 1.0,
               batch_size: int = DEFAULT_BATCH_SIZE,
               max_len: int = DEFAULT_MAX_LEN,
               source_tag: str | None = None) -> list[dict]:
    """Per-document records for every row: ``nll_active_sum`` under
    ``obj.apply(model, scale=scale)`` and ``nll_noop_sum`` under
    ``nullcontext`` (the exact unmodified base model), both from the SAME
    collated batch, no-grad, one forward each. ``obj=None`` scores the
    bare model once and writes it to both columns.

    Deterministic by construction: pinned attention backends, rows
    length-sorted into fixed-size batches, no sampling. Records come back
    sorted by ``doc_id``.
    """
    import torch

    from sisteer.extraction.fitmon import pin_attention_backends
    from sisteer.extraction.traindata import collate

    if int(batch_size) < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size!r}")
    if source_tag is not None and source_tag not in SOURCE_TAGS:
        raise ValueError(f"source_tag must be one of {SOURCE_TAGS} or None, "
                         f"got {source_tag!r}")
    pin_attention_backends()
    model.eval()
    device = next(model.parameters()).device
    rows = check_rows(rows)
    encs = encode_docs(tok, rows, max_len=max_len, obj=obj)

    pad_id = (tok.pad_token_id if tok.pad_token_id is not None
              else tok.eos_token_id)
    if pad_id is None:
        raise ValueError("tokenizer has neither pad nor eos token id")

    # Length-bucketed batches (the trainer's ordering): near-uniform pad
    # width per batch. The sum/sum metric is invariant to this ordering.
    order = sorted(range(len(rows)), key=lambda i: len(encs[i].ids))
    n = len(rows)
    # per row (in ``rows`` order): (nll_active_sum, nll_noop_sum, n_targets)
    per_row: list[tuple[float, float, int] | None] = [None] * n
    with torch.no_grad():
        for i in range(0, n, int(batch_size)):
            idx = order[i:i + int(batch_size)]
            batch = collate([encs[j] for j in idx], pad_id).to(device)
            ctx = (obj.apply(model, scale=scale) if obj is not None
                   else nullcontext())
            with ctx:
                logits = model(input_ids=batch.input_ids,
                               attention_mask=batch.attention_mask).logits
            a_nll, a_n = _row_nll(logits, batch)
            del logits
            if obj is None:
                b_nll = a_nll
            else:
                # the exact unmodified base model, on the SAME batch
                logits = model(input_ids=batch.input_ids,
                               attention_mask=batch.attention_mask).logits
                b_nll, b_n = _row_nll(logits, batch)
                del logits
                if not torch.equal(a_n, b_n):
                    raise AssertionError(
                        "target counts differ between the active and no-op "
                        "legs of the same batch — the labels are not the "
                        "object's to change")
            for k, j in enumerate(idx):
                per_row[j] = (float(a_nll[k]), float(b_nll[k]), int(a_n[k]))

    records = []
    for j, row in enumerate(rows):
        nll_active, nll_noop, n_targets = per_row[j]
        if n_targets == 0:
            raise ValueError(
                f"doc {row['doc_id']!r} has ZERO CE targets after the label "
                "shift — nothing to score")
        records.append({
            "doc_id": str(row["doc_id"]),
            "family": row.get("family"),
            "source_tag": source_tag,
            "n_target_tokens": n_targets,
            "nll_active_sum": nll_active,
            "nll_noop_sum": nll_noop,
        })
    return records


# ---------------------------------------------------------------------------
# Aggregation — pure arithmetic on the records
# ---------------------------------------------------------------------------


def aggregate(records: list[dict]) -> dict:
    """The micro-averaged (sum/sum) aggregate: ``ce_active`` =
    Σ nll_active / Σ n_target_tokens, ``ce_noop`` likewise, ``ratio`` =
    ce_active / ce_noop, ``gain`` = 1 − ratio. A ratio of sums, never a
    mean of per-doc or per-batch means — the batching-invariant quantity
    the parity pin rests on."""
    if not records:
        raise ValueError("aggregate: no records")
    n_tokens = sum(int(r["n_target_tokens"]) for r in records)
    if n_tokens <= 0:
        raise ValueError("aggregate: zero target tokens")
    a = sum(float(r["nll_active_sum"]) for r in records)
    b = sum(float(r["nll_noop_sum"]) for r in records)
    ce_active = a / n_tokens
    ce_noop = b / n_tokens
    ratio = ce_active / ce_noop if ce_noop != 0 else math.inf
    return {
        "ce_active": ce_active,
        "ce_noop": ce_noop,
        "ratio": ratio,
        "gain": 1.0 - ratio,
        "n_docs": len(records),
        "n_tokens": n_tokens,
    }


def _pct_index(q: float, m: int) -> int:
    return min(m - 1, max(0, int(round(q * (m - 1)))))


def doc_bootstrap_gain(records: list[dict], *, n_boot: int = 1000,
                       seed: int = 0, level: float = 0.90) -> dict:
    """Doc-level percentile bootstrap of ``gain = 1 − Σ nll_active /
    Σ nll_noop`` (the ratio of sums re-computed on each resample of docs
    with replacement — the same estimand as :func:`aggregate`, never a
    mean of per-doc gains). Pure python, ``random.Random(seed)``. Returns
    ``{"gain", "lo", "hi", "n_docs", "n_boot", "level", "seed"}``; with
    one doc every resample is that doc, so ``lo == gain == hi``."""
    n = len(records)
    if n == 0:
        raise ValueError("doc_bootstrap_gain: no records")
    if not (0.0 < level < 1.0):
        raise ValueError(f"level must be in (0, 1), got {level!r}")
    if int(n_boot) < 1:
        raise ValueError(f"n_boot must be >= 1, got {n_boot!r}")
    act = [float(r["nll_active_sum"]) for r in records]
    noop = [float(r["nll_noop_sum"]) for r in records]

    def gain_of(idx) -> float:
        sb = sum(noop[i] for i in idx)
        sa = sum(act[i] for i in idx)
        return 1.0 - (sa / sb if sb != 0 else math.inf)

    point = gain_of(range(n))
    rng = random.Random(seed)
    boots = sorted(gain_of([rng.randrange(n) for _ in range(n)])
                   for _ in range(int(n_boot)))
    alpha = (1.0 - level) / 2.0
    return {
        "gain": point,
        "lo": boots[_pct_index(alpha, len(boots))],
        "hi": boots[_pct_index(1.0 - alpha, len(boots))],
        "n_docs": n,
        "n_boot": int(n_boot),
        "level": level,
        "seed": seed,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True, help="HF model id")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--pairs", required=True,
                    help="document pool JSONL (doc_id, family, prompt, pos, "
                         "...) — the doc-mode pairs file")
    ap.add_argument("--out", required=True,
                    help="per-doc records JSONL; the summary lands at "
                         "<out>.summary.json")
    ap.add_argument("--object", default=None,
                    help="saved intervention artifact (sisteer load_object); "
                         "omit for the bare model (both columns then equal)")
    ap.add_argument("--cell", default="",
                    help="the identity --object claims (a staged cell id "
                         "<data-cell>__<arm>, or a derived_id); validated "
                         "against the artifact's sidecar BEFORE anything "
                         "loads it")
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--allow-bare", action="store_true",
                    help="accept an object dir with NO sidecar (scratch/"
                         "test objects only — never a production default)")
    ap.add_argument("--parents-root", default=None,
                    help="fits bank root (<key>/cells_all/<cell>/) to "
                         "re-verify a derived object's parents on disk")
    ap.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    ap.add_argument("--max-len", type=int, default=DEFAULT_MAX_LEN)
    ap.add_argument("--device", default=None)
    ap.add_argument("--dtype", default=None)
    ap.add_argument("--limit", type=int, default=0,
                    help="score only the first N doc_id-sorted rows (smoke)")
    ap.add_argument("--source-tag", choices=SOURCE_TAGS, default=None,
                    help="which pool this is (recorded per doc + summary)")
    ap.add_argument("--pool", default=None,
                    help="the pool's ANALYSIS name (round-2 contract: "
                         "constv2-heldout | control-heldout | constv2ctl-100k); "
                         "default = the pairs filename's pairs_<pool>.jsonl stem")
    return ap


def pool_name(explicit: str | None, pairs_path) -> str:
    """The pool's analysis name: ``--pool`` when given, else the
    ``pairs_<pool>.jsonl`` stem of the pairs file (``pairs_constv2-heldout.
    jsonl`` -> ``constv2-heldout``; any other filename -> its stem)."""
    if explicit:
        return str(explicit)
    stem = Path(pairs_path).name
    for suf in (".jsonl", ".json"):
        if stem.endswith(suf):
            stem = stem[: -len(suf)]
            break
    return stem[len("pairs_"):] if stem.startswith("pairs_") else stem


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.cell and not args.object:
        raise SystemExit("--cell names the identity of --object; it is "
                         "meaningless without one")
    if args.limit < 0:
        raise SystemExit(f"--limit must be >= 0, got {args.limit}")

    identity_kind = None
    if args.object:
        # Identity gate BEFORE any GPU/model work (a wrong-cell, stale or
        # unprovable artifact refuses in milliseconds).
        from sisteer.eval.object_identity import validate_artifact_identity

        identity_kind = validate_artifact_identity(
            args.object, args.cell or None, consumer=CONSUMER,
            parents_root=args.parents_root, allow_bare=args.allow_bare)

    rows = check_rows(_read_jsonl(args.pairs))
    if not rows:
        raise SystemExit(f"{args.pairs}: no rows")
    if args.limit:
        rows = rows[:args.limit]

    import torch

    from sisteer.eval.defense_cell import sha256_file
    from sisteer.models import load_model_and_tokenizer

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = args.dtype or ("bfloat16" if device == "cuda" else "float32")
    model, tok = load_model_and_tokenizer(
        args.model, device=device, dtype=dtype, revision=args.revision)

    obj = None
    if args.object:
        from sisteer.methods import load_object

        obj = load_object(args.object, model=model, tok=tok)

    # One in-process token memo for the run: encode_docs counts tokens and
    # then _encode_aligned re-tokenizes the same strings, and a cache hit is
    # a copy of what the tokenizer returned (byte-neutral — the fit
    # scripts' convention).
    from sisteer.extraction import tokcache

    prev = tokcache.active()
    tokcache.set_active(tokcache.TokCache())
    try:
        records = score_docs(model, tok, rows, obj=obj, scale=args.scale,
                             batch_size=args.batch_size, max_len=args.max_len,
                             source_tag=args.source_tag)
    finally:
        tokcache.set_active(prev)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # ``--out x.json`` = the round-2 analysis contract (analysis_r2/
    # r2_paths.py: ce_score/<model>/<cell>__<pool>__scale<s>.json — ONE
    # file, the aggregate + provenance with the per-doc records embedded
    # under ``docs``); any other suffix = per-doc JSONL + <out>.summary.json.
    combined = out.suffix == ".json"
    if not combined:
        out.write_text("".join(json.dumps(r, sort_keys=True) + "\n"
                               for r in records), encoding="utf-8")
    summary = {
        **aggregate(records),
        "pool": pool_name(args.pool, args.pairs),
        "scale": args.scale,
        "object": args.object,
        "object_sha256": sha256_file(args.object) if args.object else None,
        "identity_kind": identity_kind,
        "cell": args.cell or None,
        "pairs": str(args.pairs),
        "pairs_sha256": sha256_file(args.pairs),
        "model": args.model,
        "revision": args.revision,
        "device": device,
        "dtype": str(dtype),
        "max_len": args.max_len,
        "batch_size": args.batch_size,
        "source_tag": args.source_tag,
        "limit": args.limit or None,
        "consumer": CONSUMER,
        # A glance at P3's read; the analysis re-runs the PAIRED bootstrap
        # against the parents' records, which is why the per-doc file is
        # the primary output and this block is a convenience.
        "gain_doc_boot90": doc_bootstrap_gain(records, n_boot=1000, seed=0,
                                              level=0.90),
    }
    if combined:
        out.write_text(json.dumps({**summary, "docs": records}, indent=1,
                                  sort_keys=True) + "\n", encoding="utf-8")
    else:
        Path(str(out) + ".summary.json").write_text(
            json.dumps(summary, indent=1, sort_keys=True) + "\n",
            encoding="utf-8")
    print(f"scored {summary['n_docs']} docs / {summary['n_tokens']} target "
          f"tokens -> {out}  ce_active={summary['ce_active']:.5f} "
          f"ce_noop={summary['ce_noop']:.5f} gain={summary['gain']:+.5f} "
          f"(scale {args.scale}, identity {identity_kind})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
