#!/usr/bin/env python
"""Build sdf-base-steering document pools, manifests, and the max-length
preflight.

Lanes (one deterministic draw order per lane, shared by both models):

- mixed:   sha-shuffled advice + stories (salt ``sdfbase-draw-v1``),
           greedy-ratio interleaved at the train-side natural token ratio,
           cells 1k ⊂ 10k ⊂ 100k ⊂ 1M ⊂ 10M ⊂ full by prefix (full =
           the ENTIRE train side, budget None, whole docs, no truncation).
- advice:  the advice family's own shuffle (salt ``sdfbase-adv-draw-v1``),
           cells 1k ⊂ 100k.
- heldout: the frozen 128-doc eval subset from the split artifact, rendered
           whole (``pairs_constv2-heldout.jsonl``).
- control: the same sampler re-invoked on the control-corpus file(s)
           (salt ``sdfbase-ctl-draw-v1``, one 100k cell); runs only when its
           input paths are supplied. Two lane-specific rules: emitted
           rows/manifests carry ``ctl-<original>`` doc ids (ranking still
           uses the ORIGINAL ids, so selection is unchanged; the prefix
           kills id collisions with the production corpus), and the ratio
           gate runs in record-not-refuse mode (family doc granularity at
           5 advice docs makes ±2pp infeasible at 100k — the failed verdict
           is recorded as applied/failed/waived, never forced and never
           silently passed). Corpus lanes keep refuse semantics.

Budget accounting is the AUTHORITATIVE set rule: the sampler carries three
cumulative per-tokenizer totals along the draw order; a cell of budget B
takes whole docs while min_k(cum_k) < B, then truncates the final doc so
that min_k(cum_k + count_k(masked_prefix + trained_prefix)) == B EXACTLY.
Character-prefix token counts are non-monotonic (appending text can merge
earlier tokens), so the cutoff is a deterministic FORWARD SCAN over the
limiting tokenizer's token-boundary candidates — never a binary search.

Boundary skip-forward rule: when the boundary doc admits NO exact cutoff
with non-zero loss tokens (its masked prefix alone meets/exceeds the
remainder, or no token-boundary candidate lands the set total exactly on
B), the sampler advances in draw order to the next doc that does,
recording every skipped doc id + reason in the cell manifest under
``boundary_skips``. Skips are per-cell only: a doc skipped at one cell's
boundary stays eligible for larger cells, which re-walk the same order
independently. The loud refusal remains for the truly-impossible case —
the draw exhausts with no doc admitting a cutoff (bracketing candidates
included in the skip details).

Pool rows are Pair-shaped so ``_load_pairs`` consumes them unchanged:
{"prompt": <loss-masked prefix>, "pos": <trained text>, "neg": "", ...}
with the invariant prompt+pos byte-equal to the corpus renderer output
(asserted per doc at build time).

``preflight`` tokenizes EVERY train + held-out row with each ACTUAL pinned
target tokenizer and requires
len(prompt w/ specials) + len(completion w/o specials) + 1 (EOS) <= 8192,
refusing loudly (named rows, nonzero exit) on any violation; realized max
lengths and per-model loss-token counts land in each cell manifest.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, replace
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))
import build_heldout_split as hsplit  # noqa: E402
import derive_model_config as _dmc  # noqa: E402

TOK_NAMES = hsplit.TOK_NAMES
WORKTREE_ROOT = hsplit.WORKTREE_ROOT

SALT_MIX = "sdfbase-draw-v1"
SALT_ADV = "sdfbase-adv-draw-v1"
SALT_CTL = "sdfbase-ctl-draw-v1"

MIX_CELLS = [
    ("constv2-mix-1k", 1_000),
    ("constv2-mix-10k", 10_000),
    ("constv2-mix-100k", 100_000),
    ("constv2-mix-1m", 1_000_000),
    # Round-2 token-scaling extension: longer prefixes of the SAME
    # deterministic draw order. ``None`` = the ENTIRE train side, whole
    # docs, no truncation (the complete order); 10m is a
    # 10,000,000-official-token prefix like the smaller cells. Nesting
    # 1k ⊂ … ⊂ 1m ⊂ 10m ⊂ full holds by construction and is verified by
    # ``verify_mix_nesting`` (the ``verify-nesting`` subcommand).
    ("constv2-mix-10m", 10_000_000),
    ("constv2-mix-full", None),
]
ADV_CELLS = [
    ("constv2-adv-1k", 1_000),
    ("constv2-adv-100k", 100_000),
]
CTL_CELLS = [
    ("constv2ctl-100k", 100_000),
]
HELDOUT_CELL = "constv2-heldout"
# Control-lane doc-id prefix: the control run re-minted the production
# id grammar, colliding 51/58 ids with production (11 with held-out
# ids). Pool rows and manifests therefore carry ctl-<original>; the
# mapping IS the prefix. Draw ranking/selection still uses the ORIGINAL
# ids (same salts => identical doc selection).
CTL_ID_PREFIX = "ctl-"

# Ratio gate: ±2 PERCENTAGE POINTS, ABSOLUTE advice-share deviation, at
# budgets >= 100k — the only feasible reading at whole-doc granularity
# (±2% relative would be unattainable).
RATIO_TOL = 0.02
RATIO_GATE_MIN_BUDGET = 100_000
MAX_LEN = 8192

# The two ACTUAL pinned target tokenizers. The corpus's
# 7,996-token max is a REFERENCE-tokenizer number; 8192 is unverified for a
# target tokenizer until the preflight runs against it. Derived from the
# campaign's ONE pin registry (derive_model_config.MODEL_SPECS) so a
# revision bump can never leave the preflight certifying a tokenizer the
# fits no longer use.
TARGET_TOKENIZERS: dict[str, tuple[str, str]] = {
    s: (_dmc.MODEL_SPECS[s].model, _dmc.MODEL_SPECS[s].revision)
    for s in _dmc.SDFBASE_SLUGS
}

# Chat-template markup that must NEVER appear in pool text (spec check 1).
CHAT_MARKUP = (
    "<|im_start|>", "<|im_end|>", "<start_of_turn>", "<end_of_turn>",
    "<|user|>", "<|assistant|>", "<|system|>", "[INST]", "[/INST]",
    "<|begin_of_text|>", "<|start_header_id|>", "<bos>", "<eos>",
)

# Claude/Anthropic-mention canary on trained text (word-boundary: the corpus
# is full of "philanthropic", which contains "anthropic" as a substring).
CANARY_RE = re.compile(r"\b(claude|anthropic)\b", re.IGNORECASE)
# doc_ids allowed to carry an ORGANIC mention in rendered text. Verified
# 2026-09-01: zero word-boundary hits in either v2 final, so this ships
# empty. Any future entry needs a comment naming the mention and why it is
# organic corpus content rather than metadata leakage.
CANARY_ALLOWLIST: frozenset[str] = frozenset()


class SamplerError(RuntimeError):
    """Loud failure — never ship an off-by-some or contaminated cell."""


class PreflightError(RuntimeError):
    """Loud failure — an over-length row would be truncated SILENTLY by
    _encode_aligned downstream, quietly shrinking the cell's budget."""


class TruncationInfeasible(SamplerError):
    """The boundary doc admits no exact cutoff with non-zero loss tokens.

    Caught by cut_cell's skip-forward rule; terminal only when NO
    remaining doc in the draw admits a cutoff."""

    def __init__(self, msg: str, reason: str, detail: dict):
        super().__init__(msg)
        self.reason = reason
        self.detail = detail


# --------------------------------------------------------------------------
# Counting rig — the corpus repo's staged reference tokenizers, plus token
# boundary offsets for the truncation scan. Tests inject a fake rig with the
# same duck-typed surface: counts(text) -> dict, boundary_offsets(name, text).
# --------------------------------------------------------------------------

class StagedCountRig:
    def __init__(self, staged_dir: Path | None = None):
        tokens_mod = hsplit.generator_tokens()
        # --tokenizers-dir > $CORPUS_GENERATOR_TOKENIZERS_DIR > <data root>/tokenizers
        staged_dir = hsplit.tokenizers_dir(staged_dir)
        tc = tokens_mod.TokenCounter.load(staged_dir)
        if set(tc.tokenizers) != set(TOK_NAMES):
            raise SamplerError(
                f"staged tokenizers {sorted(tc.tokenizers)} != {sorted(TOK_NAMES)}"
            )
        self._toks = tc.tokenizers
        self.provenance = tc.provenance

    def counts(self, text: str) -> dict[str, int]:
        return {
            name: len(tok.encode(text, add_special_tokens=False).ids)
            for name, tok in self._toks.items()
        }

    def boundary_offsets(self, name: str, text: str) -> list[int]:
        enc = self._toks[name].encode(text, add_special_tokens=False)
        return sorted({end for _, end in enc.offsets if end > 0})


# --------------------------------------------------------------------------
# Documents
# --------------------------------------------------------------------------

@dataclass
class Doc:
    doc_id: str
    family: str
    prompt: str  # loss-masked prefix (advice system + "\n\n", else "")
    pos: str     # trained text
    token_counts: dict[str, int]  # full-render per-tokenizer counts
    truncated: bool = False
    tokens_taken: dict[str, int] | None = None  # contribution if truncated
    char_offset: int | None = None
    cut_candidate_index: int | None = None

    @property
    def contrib(self) -> dict[str, int]:
        return self.tokens_taken if self.truncated else self.token_counts

    @property
    def official(self) -> int:
        return min(self.contrib.values())


def check_canary(doc_id: str, text: str) -> None:
    m = CANARY_RE.search(text)
    if m and doc_id not in CANARY_ALLOWLIST:
        raise SamplerError(
            f"{doc_id}: rendered text mentions {m.group(0)!r} — metadata "
            "leakage into trained text, or an organic mention missing from "
            "CANARY_ALLOWLIST"
        )


def make_doc(rec: dict, family: str, tokens_mod) -> Doc:
    """Render one record into the Pair-shaped doc, asserting byte parity
    with the corpus repo's canonical renderer."""
    did = hsplit.doc_id(rec)
    if family == "advice":
        system = rec.get("system") or ""
        prompt = system + "\n\n" if system else ""
        pos = rec["user"] + "\n\n" + rec["assistant"]
        rendered = tokens_mod.render_advice_text(rec)
    elif family in ("stories", "control_stories"):
        prompt = ""
        pos = rec["text"]
        rendered = tokens_mod.render_story_text(rec)
    else:
        raise SamplerError(f"unknown family {family!r}")
    family = "stories" if family == "control_stories" else family
    if not pos:
        raise SamplerError(f"{did}: empty trained text")
    if prompt + pos != rendered:
        raise SamplerError(
            f"{did}: prompt+pos is not byte-identical to the corpus renderer "
            "output — rendering drift"
        )
    for marker in CHAT_MARKUP:
        if marker in rendered:
            raise SamplerError(f"{did}: chat markup {marker!r} in rendered text")
    check_canary(did, rendered)
    return Doc(
        doc_id=did,
        family=family,
        prompt=prompt,
        pos=pos,
        token_counts=hsplit.record_token_counts(rec),
    )


# --------------------------------------------------------------------------
# Draw order
# --------------------------------------------------------------------------

def sha_shuffle(docs: list[Doc], salt: str) -> list[Doc]:
    return sorted(docs, key=lambda d: (hsplit.sha_rank(salt, d.doc_id), d.doc_id))


def family_officials(cum: dict[str, dict[str, int]]) -> dict[str, int]:
    return {f: min(t.values()) for f, t in cum.items()}


def greedy_interleave(fams: dict[str, list[Doc]],
                      shares: dict[str, float]) -> list[Doc]:
    """Greedy-ratio interleave: at each step append the next doc
    from whichever family's cumulative official tokens are furthest BELOW
    its target share. Family cumulative officials use the same set rule
    (min over that family's per-tokenizer cumulative sums). Ties break to
    the larger-share family, then name — so the first doc drawn comes from
    the dominant family."""
    idx = {f: 0 for f in fams}
    cum = {f: {k: 0 for k in TOK_NAMES} for f in fams}
    out: list[Doc] = []
    while True:
        avail = [f for f in fams if idx[f] < len(fams[f])]
        if not avail:
            break
        off = family_officials(cum)
        total = sum(off.values())
        pick = max(avail, key=lambda f: (shares[f] * total - off[f], shares[f], f))
        doc = fams[pick][idx[pick]]
        idx[pick] += 1
        for k, n in doc.token_counts.items():
            cum[pick][k] += n
        out.append(doc)
    return out


# --------------------------------------------------------------------------
# Budget cut + truncation scan
# --------------------------------------------------------------------------

def truncate_doc(doc: Doc, cum: dict[str, int], budget: int, rig) -> Doc:
    """Truncate the final doc so the SET total lands exactly on budget.

    Candidates are the limiting tokenizer's token-boundary char offsets over
    the full render (masked prefix kept whole); each candidate is re-counted
    under all three tokenizers and the scan is a deterministic FORWARD walk
    (no binary search — char-prefix counts are non-monotonic). No exact
    landing => TruncationInfeasible, which cut_cell's skip-forward rule
    catches and records."""
    full = doc.prompt + doc.pos
    if doc.prompt:
        pre = rig.counts(doc.prompt)
        pre_total = min(cum[k] + pre[k] for k in TOK_NAMES)
        if pre_total >= budget:
            raise TruncationInfeasible(
                f"masked-prefix-eats-budget: doc {doc.doc_id} masked prefix "
                f"alone brings the set to {pre_total} >= budget {budget} — "
                "it would contribute zero loss tokens as the final doc",
                reason="masked-prefix-eats-budget",
                detail={
                    "prefix_counts": pre,
                    "remainder": budget - min(cum.values()),
                },
            )
    whole_totals = {k: cum[k] + doc.token_counts[k] for k in TOK_NAMES}
    _, limiting = hsplit.official_of(whole_totals)
    bounds = rig.boundary_offsets(limiting, full)
    candidates = [b for b in bounds if len(doc.prompt) < b <= len(full)]
    if not candidates:
        raise TruncationInfeasible(
            f"{doc.doc_id}: no cut candidates past the masked prefix "
            f"(limiting tokenizer {limiting})",
            reason="no-cut-candidates",
            detail={"limiting_tokenizer": limiting},
        )
    best_below: tuple[int, int, int] | None = None  # (total, idx, offset)
    first_above: tuple[int, int, int] | None = None
    for i, b in enumerate(candidates):
        counts_b = rig.counts(full[:b])
        total = min(cum[k] + counts_b[k] for k in TOK_NAMES)
        if total == budget:
            return Doc(
                doc_id=doc.doc_id,
                family=doc.family,
                prompt=doc.prompt,
                pos=full[len(doc.prompt):b],
                token_counts=doc.token_counts,
                truncated=True,
                tokens_taken=counts_b,
                char_offset=b,
                cut_candidate_index=i,
            )
        if total < budget:
            if best_below is None or total > best_below[0]:
                best_below = (total, i, b)
        elif first_above is None:
            first_above = (total, i, b)
    raise TruncationInfeasible(
        f"{doc.doc_id}: no token-boundary candidate lands the set total "
        f"exactly on budget {budget}; bracketing candidates: "
        f"below={best_below} (total, candidate_index, char_offset), "
        f"above={first_above}",
        reason="no-exact-cutoff",
        detail={
            "bracketing_below": list(best_below) if best_below else None,
            "bracketing_above": list(first_above) if first_above else None,
        },
    )


def _token_totals(docs: list[Doc]) -> dict[str, int]:
    """Per-tokenizer token totals over ``docs`` — the cum dict a whole-order
    cell (budget None) or the held-out set reports; its min is a family's
    official total (empty docs sum to 0 for every tokenizer)."""
    return {k: sum(d.token_counts[k] for d in docs) for k in TOK_NAMES}


def cut_cell(order: list[Doc], budget: int | None, rig
             ) -> tuple[list[Doc], dict[str, int], list[dict]]:
    """Take docs along the draw order while min_k(cum_k) < budget, truncating
    the final doc to hit the budget exactly (authoritative set rule).

    ``budget=None`` (the constv2-mix-full cell) takes the COMPLETE draw
    order — every doc, whole, no truncation, no skips: the cell IS the
    entire remaining train side, so there is no boundary to land on.

    Skip-forward rule: a boundary doc that admits no exact cutoff with
    non-zero loss tokens is SKIPPED (recorded in the returned skip list,
    and in the manifest as ``boundary_skips``) and the walk advances to
    the next doc in draw order; a skipped doc stays eligible for other
    cells, which walk the same order independently. The loud refusal
    remains for the truly-impossible case: the draw exhausts with no doc
    admitting a cutoff."""
    if budget is None:
        return list(order), _token_totals(order), []
    cum = {k: 0 for k in TOK_NAMES}
    picked: list[Doc] = []
    skips: list[dict] = []
    for doc in order:
        whole = {k: cum[k] + doc.token_counts[k] for k in TOK_NAMES}
        whole_official = min(whole.values())
        if whole_official < budget:
            cum = whole
            picked.append(doc)
            continue
        if whole_official == budget:
            # Exact whole-doc landing — no truncation needed.
            return picked + [doc], whole, skips
        try:
            tdoc = truncate_doc(doc, cum, budget, rig)
        except TruncationInfeasible as e:
            skips.append({
                "doc_id": doc.doc_id,
                "reason": e.reason,
                "detail": e.detail,
            })
            continue
        for k in TOK_NAMES:
            cum[k] += tdoc.tokens_taken[k]
        if min(cum.values()) != budget:
            raise SamplerError(
                f"internal: post-truncation official {min(cum.values())} != "
                f"budget {budget}"
            )
        return picked + [tdoc], cum, skips
    raise SamplerError(
        f"draw exhausted at {min(cum.values())} official tokens — cannot "
        f"fill budget {budget}; no remaining doc admits an exact cutoff "
        f"with non-zero loss tokens. boundary_skips="
        f"{json.dumps(skips, ensure_ascii=False)}"
    )


# --------------------------------------------------------------------------
# Pool + manifest writing
# --------------------------------------------------------------------------

def pool_row(doc: Doc, cell: str) -> dict:
    return {
        "prompt": doc.prompt,
        "pos": doc.pos,
        "neg": "",
        "instruction_id": cell,
        "doc_id": doc.doc_id,
        "family": doc.family,
        "official_tokens": doc.official,
        "truncated": doc.truncated,
    }


def write_pool(path: Path, cell: str, docs: list[Doc]) -> str:
    with open(path, "w", encoding="utf-8") as f:
        for doc in docs:
            f.write(json.dumps(pool_row(doc, cell), ensure_ascii=False,
                               separators=(",", ":")) + "\n")
    return hsplit.sha256_file(path)


def official_loss_totals(docs: list[Doc], rig) -> dict[str, int]:
    """Per-tokenizer totals over TRAINED text only (pos; masked prefix
    excluded) under the reference rig. Stories rows (empty prompt) reuse
    their full-render counts; advice rows are re-counted on pos alone."""
    totals = {k: 0 for k in TOK_NAMES}
    for doc in docs:
        if not doc.prompt:
            counts = doc.contrib
        else:
            counts = rig.counts(doc.pos)
        for k in TOK_NAMES:
            totals[k] += counts[k]
    return totals


def build_cell(cell: str, lane: str, budget: int | None, docs: list[Doc],
               cum: dict[str, int], *, salts: dict, natural_ratio: float | None,
               out_dir: Path, rig, split_sha: str, corpus_shas: dict,
               generator: dict, boundary_skips: list[dict] | None = None,
               ratio_gate_mode: str = "refuse",
               constdatagen_provenance: dict | None = None) -> dict:
    """Write one cell's pool + manifest.

    Ratio gate: ±2 percentage points, ABSOLUTE share deviation, applied at
    budgets >= 100k. ``ratio_gate_mode`` is per-lane: "refuse" (corpus
    lanes) raises on a failed gate; "record" (control lane) writes the
    truthful failed verdict into the manifest —
    applied/tolerance/deviation/passed/waived — instead of refusing. A
    passing gate records the same shape in either mode (no passed/waived
    keys)."""
    if ratio_gate_mode not in ("refuse", "record"):
        raise SamplerError(f"unknown ratio_gate_mode {ratio_gate_mode!r}")
    fam_cum: dict[str, dict[str, int]] = {}
    fam_counts: dict[str, int] = {}
    for doc in docs:
        fam_counts[doc.family] = fam_counts.get(doc.family, 0) + 1
        t = fam_cum.setdefault(doc.family, {k: 0 for k in TOK_NAMES})
        for k in TOK_NAMES:
            t[k] += doc.contrib[k]
    fam_official = family_officials(fam_cum)
    total_official = sum(fam_official.values())
    realized_ratio = (
        fam_official.get("advice", 0) / total_official if total_official else None
    )

    gate_applied = (
        budget is not None
        and budget >= RATIO_GATE_MIN_BUDGET
        and natural_ratio is not None
        and lane in ("mixed", "control")
    )
    deviation = (
        abs(realized_ratio - natural_ratio)
        if (realized_ratio is not None and natural_ratio is not None)
        else None
    )
    ratio_gate = {
        "applied": bool(gate_applied),
        "tolerance": RATIO_TOL,
        "deviation": deviation,
    }
    if gate_applied and deviation > RATIO_TOL:
        if ratio_gate_mode == "record":
            ratio_gate["passed"] = False
            ratio_gate["waived"] = (
                "record-not-refuse gate mode for this lane (decisions.md "
                "D15/D10): family doc granularity can make the ±2pp gate "
                "infeasible at this budget; recorded + reported rather "
                "than forced"
            )
        else:
            raise SamplerError(
                f"{cell}: realized advice share {realized_ratio:.4f} deviates "
                f"{deviation:.4f} from natural {natural_ratio:.4f} (tolerance "
                f"{RATIO_TOL} absolute percentage points, D10) — ratio gate failed"
            )

    truncated = [d for d in docs if d.truncated]
    if len(truncated) > 1:
        raise SamplerError(f"{cell}: {len(truncated)} truncated docs (max 1)")
    truncation = None
    if truncated:
        t = truncated[0]
        truncation = {
            "doc_id": t.doc_id,
            "cut_candidate_index": t.cut_candidate_index,
            "char_offset": t.char_offset,
            "tokens_taken_per_tokenizer": t.tokens_taken,
        }

    official, limiting = hsplit.official_of(cum)
    if budget is not None and official != budget:
        raise SamplerError(f"{cell}: official {official} != budget {budget}")

    pool_path = out_dir / f"pairs_{cell}.jsonl"
    pool_sha = write_pool(pool_path, cell, docs)

    loss_totals = official_loss_totals(docs, rig)
    manifest = {
        "cell": cell,
        "lane": lane,
        "budget": budget,
        "seed_salts": salts,
        "n_docs": len(docs),
        "doc_ids_in_order": [d.doc_id for d in docs],
        "per_tokenizer_cumulative_totals": cum,
        "official_tokens": official,
        "limiting_tokenizer": limiting,
        "truncation": truncation,
        "boundary_skips": boundary_skips or [],
        "family_counts": fam_counts,
        "family_official_tokens": fam_official,
        "realized_ratio": realized_ratio,
        "natural_ratio": natural_ratio,
        "ratio_gate": ratio_gate,
        "per_tokenizer_loss_totals": loss_totals,
        "official_loss_tokens": min(loss_totals.values()),
        # Filled by the preflight (target tokenizers):
        "loss_token_counts": {k: None for k in TARGET_TOKENIZERS},
        "max_encoded_row_len": {k: None for k in TARGET_TOKENIZERS},
        "heldout_split_sha": split_sha,
        "corpus_shas": corpus_shas,
        "generator": generator,
        "reference_tokenizer_provenance": getattr(rig, "provenance", None),
        "pool_sha256": pool_sha,
    }
    if constdatagen_provenance is not None:
        manifest["constdatagen_provenance"] = constdatagen_provenance
    manifest_path = out_dir / f"pairs_{cell}.manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def build_pools(out_dir: Path, split_path: Path, *,
                advice_final: Path | None = None,
                stories_final: Path | None = None,
                lanes: tuple[str, ...] = ("mixed", "advice", "heldout"),
                mix_cells: list[tuple[str, int | None]] | None = None,
                adv_cells: list[tuple[str, int]] | None = None,
                ctl_cells: list[tuple[str, int]] | None = None,
                control_advice: Path | None = None,
                control_stories: Path | None = None,
                control_run_manifest: Path | None = None,
                rig=None, generator: dict | None = None,
                only_cells: set[str] | None = None,
                tokenizers_dir=None) -> dict:
    """Build every requested lane's pools + manifests into out_dir.

    ``generator`` lets the CLI pass the provenance it already computed for
    the dirty-tree gate (hsplit.refuse_dirty_generator_tree — production
    builds refuse a dirty generator tree unless --allow-dirty, which
    records dirty=true + a tree-diff sha instead); None recomputes it.

    ``only_cells`` restricts which cells are WRITTEN (round-2 extension:
    build the two new mixed cells into an already-staged pools dir without
    restamping the finished cells' manifests — the draw orders are
    deterministic, so a subset build selects the identical docs). Unknown
    names refuse loudly; the summary MERGES into an existing
    pools_summary.json instead of clobbering the other cells' rows."""
    advice_final = hsplit.require_final(advice_final or hsplit.ADVICE_FINAL,
                                        "--advice-final")
    stories_final = hsplit.require_final(stories_final or hsplit.STORIES_FINAL,
                                         "--stories-final")
    mix_cells = mix_cells if mix_cells is not None else MIX_CELLS
    adv_cells = adv_cells if adv_cells is not None else ADV_CELLS
    ctl_cells = ctl_cells if ctl_cells is not None else CTL_CELLS
    if only_cells is not None:
        known = ({c for c, _ in mix_cells} | {c for c, _ in adv_cells}
                 | {c for c, _ in ctl_cells} | {HELDOUT_CELL})
        unknown = sorted(set(only_cells) - known)
        if unknown:
            raise SamplerError(
                f"--cells names unknown cell(s) {unknown} — known: "
                f"{sorted(known)}")
        mix_cells = [(c, b) for c, b in mix_cells if c in only_cells]
        adv_cells = [(c, b) for c, b in adv_cells if c in only_cells]
        ctl_cells = [(c, b) for c, b in ctl_cells if c in only_cells]
        keep = {"mixed": bool(mix_cells), "advice": bool(adv_cells),
                "control": bool(ctl_cells),
                "heldout": HELDOUT_CELL in only_cells}
        lanes = tuple(lane for lane in lanes if keep[lane])

    tokens_mod = hsplit.generator_tokens()
    rig = rig or StagedCountRig(tokenizers_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    split = json.loads(Path(split_path).read_text(encoding="utf-8"))
    split_sha = hsplit.sha256_file(split_path)
    corpus_shas = {
        "advice_final": hsplit.sha256_file(advice_final),
        "stories_final": hsplit.sha256_file(stories_final),
    }
    # The split is the authority on which corpus bytes it carved — refuse a
    # drifted input rather than silently sampling a different corpus.
    for key, path in (("advice_final", advice_final), ("stories_final", stories_final)):
        if split["built_from_sha256"][key] != corpus_shas[key]:
            raise SamplerError(
                f"{path} sha256 does not match the split artifact's "
                f"built_from_sha256[{key!r}] — rebuild the split or fix the input"
            )
    if generator is None:
        # Commit + dirty flag + tree-diff sha per generator repo
        # (hsplit.generator_provenance — the per-repo ``<repo>_commit``
        # keys keep their historical spelling).
        generator = hsplit.generator_provenance()

    advice_recs = hsplit.load_jsonl(advice_final)
    stories_recs = hsplit.load_jsonl(stories_final)
    adv_docs = [make_doc(r, "advice", tokens_mod) for r in advice_recs]
    sto_docs = [make_doc(r, "stories", tokens_mod) for r in stories_recs]
    by_id = {d.doc_id: d for d in adv_docs + sto_docs}

    heldout_ids = set(split["advice"]["ids"]) | set(split["stories"]["ids"])
    train_adv = [d for d in adv_docs if d.doc_id not in heldout_ids]
    train_sto = [d for d in sto_docs if d.doc_id not in heldout_ids]

    def fam_official_total(docs: list[Doc]) -> int:
        return min(_token_totals(docs).values())

    manifests: dict[str, dict] = {}

    if "mixed" in lanes:
        t_adv = fam_official_total(train_adv)
        t_sto = fam_official_total(train_sto)
        natural = t_adv / (t_adv + t_sto)
        order = greedy_interleave(
            {"advice": sha_shuffle(train_adv, SALT_MIX),
             "stories": sha_shuffle(train_sto, SALT_MIX)},
            {"advice": natural, "stories": 1.0 - natural},
        )
        salts = {"draw": SALT_MIX, "heldout": split["salt"]}
        for cell, budget in mix_cells:
            docs, cum, skips = cut_cell(order, budget, rig)
            manifests[cell] = build_cell(
                cell, "mixed", budget, docs, cum, salts=salts,
                natural_ratio=natural, out_dir=out_dir, rig=rig,
                split_sha=split_sha, corpus_shas=corpus_shas,
                generator=generator, boundary_skips=skips,
            )

    if "advice" in lanes:
        order = sha_shuffle(train_adv, SALT_ADV)
        salts = {"draw": SALT_ADV, "heldout": split["salt"]}
        for cell, budget in adv_cells:
            docs, cum, skips = cut_cell(order, budget, rig)
            manifests[cell] = build_cell(
                cell, "advice", budget, docs, cum, salts=salts,
                natural_ratio=1.0, out_dir=out_dir, rig=rig,
                split_sha=split_sha, corpus_shas=corpus_shas,
                generator=generator, boundary_skips=skips,
            )

    if "heldout" in lanes:
        ev = split["eval_subset"]
        docs = []
        for did in ev["ids"]:
            if did not in by_id:
                raise SamplerError(f"eval subset id {did} not found in corpus")
            docs.append(by_id[did])
        cum = _token_totals(docs)
        manifests[HELDOUT_CELL] = build_cell(
            HELDOUT_CELL, "heldout", None, docs, cum,
            salts={"draw": ev["salt"], "heldout": split["salt"]},
            natural_ratio=None, out_dir=out_dir, rig=rig,
            split_sha=split_sha, corpus_shas=corpus_shas, generator=generator,
        )

    if "control" in lanes:
        if control_advice is None and control_stories is None:
            raise SamplerError(
                "control lane requested but no --control-advice/--control-stories "
                "input given (the control corpus is generated later — plan §9)"
            )
        ctl_fams: dict[str, list[Doc]] = {}
        ctl_shas: dict[str, str] = {}
        if control_advice is not None:
            recs = hsplit.load_jsonl(control_advice)
            ctl_fams["advice"] = [make_doc(r, "advice", tokens_mod) for r in recs]
            ctl_shas["control_advice"] = hsplit.sha256_file(control_advice)
        if control_stories is not None:
            recs = hsplit.load_jsonl(control_stories)
            ctl_fams["stories"] = [make_doc(r, "control_stories", tokens_mod)
                                   for r in recs]
            ctl_shas["control_stories"] = hsplit.sha256_file(control_stories)
        totals = {f: fam_official_total(ds) for f, ds in ctl_fams.items()}
        grand = sum(totals.values())
        shares = {f: t / grand for f, t in totals.items()}
        order = greedy_interleave(
            {f: sha_shuffle(ds, SALT_CTL) for f, ds in ctl_fams.items()}, shares
        )
        natural = shares.get("advice", 0.0)
        salts = {"draw": SALT_CTL}
        ctl_provenance = None
        if control_run_manifest is not None:
            ctl_provenance = {
                "run_manifest": str(control_run_manifest),
                "run_manifest_sha256": hsplit.sha256_file(control_run_manifest),
                **{f"{k}_sha256": v for k, v in ctl_shas.items()},
            }
        for cell, budget in ctl_cells:
            docs, cum, skips = cut_cell(order, budget, rig)
            # ctl- prefix at the POOL layer — applied AFTER ranking and
            # selection (which use the original ids), so the same salts pick
            # the same docs; rows, manifest doc lists, truncation and skip
            # records all carry the prefixed id.
            docs = [replace(d, doc_id=CTL_ID_PREFIX + d.doc_id) for d in docs]
            for s in skips:
                s["doc_id"] = CTL_ID_PREFIX + s["doc_id"]
            manifests[cell] = build_cell(
                cell, "control", budget, docs, cum, salts=salts,
                natural_ratio=natural, out_dir=out_dir, rig=rig,
                split_sha=split_sha, corpus_shas=ctl_shas, generator=generator,
                boundary_skips=skips, ratio_gate_mode="record",
                constdatagen_provenance=ctl_provenance,
            )

    cells_summary = {
        cell: {
            "lane": m["lane"],
            "budget": m["budget"],
            "n_docs": m["n_docs"],
            "official_tokens": m["official_tokens"],
            "limiting_tokenizer": m["limiting_tokenizer"],
            "family_counts": m["family_counts"],
            "realized_ratio": m["realized_ratio"],
            "natural_ratio": m["natural_ratio"],
            "truncated_doc": (m["truncation"] or {}).get("doc_id"),
            "boundary_skips": [s["doc_id"] for s in m["boundary_skips"]],
            "official_loss_tokens": m["official_loss_tokens"],
            "pool_sha256": m["pool_sha256"],
        }
        for cell, m in manifests.items()
    }
    summary_path = out_dir / "pools_summary.json"
    summary = {"cells": {}, "heldout_split_sha": split_sha,
               "generator": generator}
    if only_cells is not None and summary_path.exists():
        # Subset build: keep the finished cells' rows (their manifests were
        # deliberately not restamped), refuse a summary from a DIFFERENT
        # split (that would silently blend two carves into one table).
        prev = json.loads(summary_path.read_text(encoding="utf-8"))
        if prev.get("heldout_split_sha") != split_sha:
            raise SamplerError(
                "existing pools_summary.json was built from a different "
                "held-out split — refusing to merge a subset build into it")
        summary["cells"] = dict(prev.get("cells") or {})
    summary["cells"].update(cells_summary)
    summary_path.write_text(
        json.dumps(summary, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return manifests


# --------------------------------------------------------------------------
# Mixed-lane nesting verification (round-2 acceptance: 100k ⊂ 1m ⊂ 10m ⊂
# full by doc_id — the draw is deterministic and each cell is a prefix of
# the same order, so containment is checkable from manifests alone)
# --------------------------------------------------------------------------


def verify_mix_nesting(pools_dir: Path) -> list[str]:
    """Assert doc-id prefix containment across the mixed cells present in
    ``pools_dir`` (MIX_CELLS order: ascending budget, ``None`` = full =
    last). The skip-forward rule can only drop a doc at a cell's own
    BOUNDARY, so a smaller cell's doc list must satisfy: every
    non-boundary doc appears as a prefix of the bigger cell's order once
    the bigger cell's records of the smaller's skipped ids are ignored —
    in practice the committed draw has no mixed-lane skips, so the check
    is strict prefix containment and REFUSES on any skip rather than
    special-casing it. Returns the verified cell names in nesting order."""
    pools_dir = Path(pools_dir)
    paths = {c: pools_dir / f"pairs_{c}.manifest.json" for c, _ in MIX_CELLS}
    names = [c for c, p in paths.items() if p.exists()]  # MIX_CELLS order
    if len(names) < 2:
        raise SamplerError(
            f"verify-nesting: fewer than two mixed-cell manifests under "
            f"{pools_dir} (found {names}) — nothing to verify")
    manifests = {c: json.loads(paths[c].read_text(encoding="utf-8"))
                 for c in names}
    for c, m in manifests.items():
        if m["boundary_skips"]:
            raise SamplerError(
                f"verify-nesting: {c} carries boundary_skips "
                f"{[s['doc_id'] for s in m['boundary_skips']]} — strict "
                "prefix containment does not hold across a skip; inspect "
                "the manifests directly")
    for small, big in zip(names, names[1:]):
        ids_small = manifests[small]["doc_ids_in_order"]
        ids_big = manifests[big]["doc_ids_in_order"]
        if ids_big[:len(ids_small)] != ids_small:
            k = next((i for i, (a, b) in enumerate(zip(ids_small, ids_big))
                      if a != b), None)
            if k is None:
                # Every shared position agrees, so the bigger cell is
                # simply SHORTER than the smaller one — a larger budget
                # that drew fewer docs cannot contain the smaller cell
                # (named here; the bare StopIteration this corner used to
                # raise said nothing).
                raise SamplerError(
                    f"verify-nesting: {big} lists {len(ids_big)} docs but "
                    f"{small} lists {len(ids_small)} — the larger-budget "
                    "cell is shorter than the smaller one, so prefix "
                    "containment cannot hold (a truncated or mis-budgeted "
                    f"{big} manifest)")
            raise SamplerError(
                f"verify-nesting: {small} is NOT a doc-id prefix of {big} "
                f"(first divergence at position {k}: "
                f"{ids_small[k]!r} vs {ids_big[k]!r})")
    return names


# --------------------------------------------------------------------------
# Max-length preflight
# --------------------------------------------------------------------------

class HFTargetTokenizer:
    """Wraps a pinned target tokenizer. Tests inject fakes with the same
    surface: n_tokens(text, add_special_tokens) -> int."""

    def __init__(self, model_key: str, repo: str, revision: str):
        from transformers import AutoTokenizer  # noqa: PLC0415

        self.model_key = model_key
        self.repo = repo
        self.revision = revision
        self._tok = AutoTokenizer.from_pretrained(repo, revision=revision)

    def n_tokens(self, text: str, add_special_tokens: bool) -> int:
        return len(self._tok(text, add_special_tokens=add_special_tokens)["input_ids"])


def load_target_tokenizers() -> dict[str, HFTargetTokenizer]:
    return {
        key: HFTargetTokenizer(key, repo, rev)
        for key, (repo, rev) in TARGET_TOKENIZERS.items()
    }


def run_preflight(pools_dir: Path, *, max_len: int = MAX_LEN,
                  toks: dict | None = None) -> dict:
    """Tokenize EVERY row of every pool with each pinned target tokenizer;
    require len(prompt w/ specials) + len(completion w/o specials) + 1 (EOS)
    <= max_len. Refuse loudly (named rows) on violation; on success record
    max realized lengths + per-model loss-token counts in each manifest."""
    pools_dir = Path(pools_dir)
    toks = toks if toks is not None else load_target_tokenizers()
    pool_paths = sorted(pools_dir.glob("pairs_*.jsonl"))
    if not pool_paths:
        raise PreflightError(f"no pairs_*.jsonl pools under {pools_dir}")

    violations: list[dict] = []
    per_pool: dict[str, dict] = {}
    for pool_path in pool_paths:
        manifest_path = pool_path.with_name(pool_path.stem + ".manifest.json")
        if not manifest_path.exists():
            raise PreflightError(f"{pool_path.name}: manifest {manifest_path.name} missing")
        rows = []
        with open(pool_path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if line.strip():
                    rows.append((i + 1, json.loads(line)))
        stats: dict[str, dict] = {}
        for model_key, tok in toks.items():
            max_row = 0
            completion_tokens = 0
            for lineno, row in rows:
                n_prompt = tok.n_tokens(row["prompt"], True)
                n_comp = tok.n_tokens(row["pos"], False)
                total = n_prompt + n_comp + 1  # +1 EOS, appended per doc
                completion_tokens += n_comp
                max_row = max(max_row, total)
                if total > max_len:
                    violations.append({
                        "pool": pool_path.name,
                        "line": lineno,
                        "doc_id": row.get("doc_id"),
                        "model": model_key,
                        "encoded_len": total,
                        "max_len": max_len,
                    })
            stats[model_key] = {
                "max_encoded_row_len": max_row,
                "completion_tokens": completion_tokens,
                "loss_tokens_with_eos": completion_tokens + len(rows),
            }
        per_pool[pool_path.name] = {"n_rows": len(rows), "models": stats}

    if violations:
        lines = "\n".join(
            f"  {v['pool']} line {v['line']} doc {v['doc_id']} "
            f"[{v['model']}]: {v['encoded_len']} > {v['max_len']}"
            for v in violations
        )
        raise PreflightError(
            f"{len(violations)} row(s) exceed max_len={max_len} under a "
            f"pinned target tokenizer — refusing (downstream _encode_aligned "
            f"would truncate SILENTLY):\n{lines}"
        )

    tok_provenance = {
        key: {"repo": getattr(t, "repo", None), "revision": getattr(t, "revision", None)}
        for key, t in toks.items()
    }
    report = {
        "max_len": max_len,
        "tokenizers": tok_provenance,
        "pools": per_pool,
        "violations": [],
    }
    # Fold realized numbers into each cell manifest (max_encoded_row_len /
    # loss-token counts per model tokenizer).
    for pool_path in pool_paths:
        manifest_path = pool_path.with_name(pool_path.stem + ".manifest.json")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        stats = per_pool[pool_path.name]["models"]
        manifest["max_encoded_row_len"] = {
            k: s["max_encoded_row_len"] for k, s in stats.items()
        }
        manifest["loss_token_counts"] = {
            k: {"completion_tokens": s["completion_tokens"],
                "loss_tokens_with_eos": s["loss_tokens_with_eos"]}
            for k, s in stats.items()
        }
        manifest["preflight"] = {"max_len": max_len, "tokenizers": tok_provenance}
        manifest_path.write_text(
            json.dumps(manifest, indent=1, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    (pools_dir / "preflight_report.json").write_text(
        json.dumps(report, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return report


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

#: the pool directory under the staging directory ($STAGE), expanded through sisteer.site after parsing
DEFAULT_OUT = Path("$STAGE/sdfbase/pools")


def site_path(value, key: str) -> Path:
    """``value`` with its site placeholder expanded through the site config (``$STAGE``: the staging directory;
    scripts/site_paths.py); an unset placeholder refuses by name. A value without one comes back as a path."""
    from site_paths import site_path as _site_path
    return _site_path(value, key)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="build pools + manifests")
    b.add_argument("--split", type=Path, default=hsplit.SPLIT_OUT)
    b.add_argument("--advice-final", type=Path, default=hsplit.ADVICE_FINAL,
                   help="the constitution advice final (stage 02 passes the "
                        "materialised file; a missing default refuses)")
    b.add_argument("--stories-final", type=Path, default=hsplit.STORIES_FINAL)
    b.add_argument("--tokenizers-dir", type=Path, default=None,
                   help="the staged reference tokenizers (MANIFEST.json + "
                        "llama3/ gemma/ qwen/; default "
                        "$CORPUS_GENERATOR_TOKENIZERS_DIR, else <corpus data "
                        "root>/tokenizers)")
    b.add_argument("--code-root", type=Path, default=None,
                   help="corpus_generator code root (default <repo>/corpus_generator, "
                        "or $CORPUS_GENERATOR_ROOT)")
    b.add_argument("--out", type=Path, default=DEFAULT_OUT)
    b.add_argument("--lanes", default="mixed,advice,heldout",
                   help="comma list of mixed,advice,heldout,control")
    b.add_argument("--control-advice", type=Path, default=None,
                   help="control-corpus advice final (activates the control lane)")
    b.add_argument("--control-stories", type=Path, default=None,
                   help="control-corpus stories final (activates the control lane)")
    b.add_argument("--control-run-manifest", type=Path, default=None,
                   help="ConstDataGen run manifest to pin in the control "
                        "cell manifest (constdatagen_provenance)")
    b.add_argument("--cells", default="",
                   help="comma list restricting which cells to BUILD "
                        "(subset build into an already-staged dir: other "
                        "cells' pools/manifests are left byte-untouched "
                        "and the summary merges)")
    b.add_argument("--allow-dirty", action="store_true",
                   help="build from a DIRTY generator tree anyway, "
                        "recording dirty=true + the tree-diff sha in every "
                        "manifest (production builds refuse otherwise)")

    p = sub.add_parser("preflight", help="max-length preflight (plan §4)")
    p.add_argument("--pools", type=Path, default=DEFAULT_OUT)
    p.add_argument("--max-len", type=int, default=MAX_LEN)

    v = sub.add_parser("verify-nesting",
                       help="doc-id prefix containment across the mixed "
                            "cells present (1k ⊂ … ⊂ 1m ⊂ 10m ⊂ full)")
    v.add_argument("--pools", type=Path, default=DEFAULT_OUT)

    args = ap.parse_args(argv)
    for _k in ("out", "pools"):
        if getattr(args, _k, None) is not None:
            setattr(args, _k, site_path(getattr(args, _k), "--" + _k))
    try:
        if args.cmd == "build":
            if args.code_root:
                hsplit.set_code_root(args.code_root)
            generator = hsplit.refuse_dirty_generator_tree(
                args.allow_dirty, tag="build-pools")
            lanes = tuple(x.strip() for x in args.lanes.split(",") if x.strip())
            if (args.control_advice or args.control_stories) and "control" not in lanes:
                lanes = lanes + ("control",)
            only = {x.strip() for x in args.cells.split(",") if x.strip()}
            manifests = build_pools(
                args.out, args.split,
                advice_final=args.advice_final,
                stories_final=args.stories_final,
                tokenizers_dir=args.tokenizers_dir,
                lanes=lanes,
                control_advice=args.control_advice,
                control_stories=args.control_stories,
                control_run_manifest=args.control_run_manifest,
                generator=generator,
                only_cells=only or None,
            )
            for cell, m in manifests.items():
                trunc = (m["truncation"] or {}).get("doc_id")
                ratio = (f"{m['realized_ratio']:.4f}"
                         if m["realized_ratio"] is not None else "n/a")
                print(
                    f"{cell}: {m['n_docs']} docs, official "
                    f"{m['official_tokens']:,} ({m['limiting_tokenizer']}), "
                    f"advice share {ratio}, truncated={trunc}, "
                    f"families {m['family_counts']}"
                )
            print(f"wrote pools + manifests under {args.out}")
        elif args.cmd == "verify-nesting":
            names = verify_mix_nesting(args.pools)
            print("nesting VERIFIED (doc-id prefix containment): "
                  + " ⊂ ".join(names))
        else:
            report = run_preflight(args.pools, max_len=args.max_len)
            for pool, info in report["pools"].items():
                maxes = {k: v["max_encoded_row_len"] for k, v in info["models"].items()}
                print(f"{pool}: {info['n_rows']} rows, max encoded len {maxes}")
            print(f"preflight PASSED (max_len {report['max_len']}); manifests updated")
    except (SamplerError, PreflightError, hsplit.SplitError) as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
