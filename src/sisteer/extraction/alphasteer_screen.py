"""The F.9-AUROC layer/p screen for the multi-layer AlphaSteer arm
(alphasteer-full22 plan §B; spec Fixed decision 2).

For every layer l (all blocks — the full-depth curve is published whether or
not a layer is in the 15–65 % band) and every null-space ratio p in the grid,
fit ``M_l(p)`` on the FIT split of the pinned lists and score the HELD-OUT
split. A layer is eligible iff its best-p AUROC clears the threshold AND the
benign null-space leakage ``‖H_b P‖² / ‖H_b‖²`` stays under the cap; the
steered set is the eligible layers inside the band, layer 0 never.

Three things this module gets right that a plausible implementation gets
wrong (plan §B.1, §B.3, §B.4; decisions D12, D14, D16):

1. **The score is the DEPLOYED operation, ``h @ M``** — a ROW-vector matmul
   (``lens_adapters.py:3871-3873``: ``h_last @ M``; ``AlphaSteerObject._edit``).
   Because the ridge target stacks ONE refusal vector, ``M = u rᵀ`` EXACTLY,
   so ``‖h @ M‖ = |h·u|·‖r‖`` (separates on ``u``, the learned, null-space-
   filtered direction) while the transpose ``‖M @ h‖ = |r·h|·‖u‖`` separates
   on the raw refusal direction — a different ranking the server never
   computes. :func:`deployed_score` is the ONE scoring helper; the screen
   and the released-Llama recovery reading (plan Phase D) both call it, and
   the rank-1 identity is asserted per cell as a cross-check.
2. **The fit/held-out split groups by normalised prompt CONTENT, not row
   index.** The harmful list is 2,000 rows but 1,776 unique after
   normalisation (one prompt appears 80 times); an index split puts copies
   of one prompt on both sides while ``fit_idx ∩ heldout_idx = ∅`` passes.
   Whole groups are assigned by hash rank on the identity, every original
   row and its multiplicity is preserved (the paper-exact lists are
   untouched), the realised per-source counts are recorded because an exact
   10 % is no longer reachable, and ``content_disjoint_verified`` is the
   gate — index disjointness is recorded but demoted.
3. **The eigendecomposition is shared across p.** ``null_space_projector``
   takes the bottom ``⌊p·d⌋`` eigenvectors of the benign Gram in ascending
   order, so the grid's four p select PREFIXES of one ordering: ``eigh(G_l)``
   runs once per layer and each p is a column slice — exact, not an
   approximation, and equal to ``null_space_projector`` BY CONSTRUCTION: the
   loop calls the trainer's own ``_eigh_with_jitter``, ``projector_from_prefix``
   and ridge helpers rather than transcribing them. Only the ridge's own
   ``psd_pinv`` is per (layer, p): five eigh per layer, not eight.

:func:`normalise_prompt` is the ONE normalisation shared with the leakage
check (plan Phase 0), so "the same prompt" means one thing everywhere.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
import time
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from ..models import build_chat_prompt, get_decoder_layers
from .alphasteer import (
    DEFAULT_LAMBDA_REG,
    DEFAULT_VERIFY_SAMPLE,
    CaptureIdVerifier,
    _collect_last_acts,
    _eigh_with_jitter,
    _ridge_info,
    _ridge_system,
    null_space_projector,
    projector_from_prefix,
    psd_pinv,
    refusal_direction,
)

SCHEMA = "alphasteer_screen_f9/1"
SPLIT_SCHEMA = "alphasteer_split_v1/1"
SPLIT_CONVENTION = "stratified-v1-content-grouped"
SPLIT_SALT = "alphasteer-heldout-v1"
SPLIT_IDENTITY = "normalised prompt text (Phase 0 normalisation: NFKC -> lowercase -> punctuation to space -> whitespace collapse)"

DEFAULT_DEPTH_BAND = (0.15, 0.65)
DEFAULT_P_GRID = (0.3, 0.4, 0.5, 0.6)
DEFAULT_AUROC_THRESHOLD = 0.95
#: Leakage (benign null-space energy kept, ``null_space_projector``'s
#: ``benign_energy_kept``) is a REPORTED per-layer diagnostic and NOT an
#: eligibility input (D38, 2026-09-09). This is the value the RETIRED
#: eligibility bound had; the diagnostic panels draw it as a reference line so
#: a reader can see where that bound sat. Nothing may read it as a gate: the
#: 1e-3 came from an n=2,000 < d=4,096 fit whose rank-deficient benign Gram
#: left >= 2,096 of the 2,457 kept null-space directions EXACTLY null, so its
#: in-sample leakage collapsed to ~1e-4 by construction. On the authors'
#: full-rank n=12,599 list the same in-sample quantity is honestly
#: 2e-3..1e-2 (172x below the isotropic reference p) and the bound rejected
#: all ten paper layers while AUROC alone recovered eight.
LEAKAGE_REPORT_REFERENCE = 1e-3
#: The ONE eligibility rule, in words, recorded into every artifact (the
#: screen_f9.json rule block, the released-Llama reading, llama_gate.json);
#: :func:`layer_eligible` is the predicate both callers evaluate.
ELIGIBILITY_RULE = (
    "eligible iff AUROC(l, p_l) >= auroc_threshold, inside the depth band, "
    "never layer 0 — the paper's Appendix F.9 norm-separability criterion, "
    "AUROC ONLY. Leakage (benign null-space energy kept) is reported per layer "
    "as a diagnostic and is NOT an eligibility input: the earlier 1e-3 bound "
    "measured the rank deficiency of an n<d fit, not the projector (D38).")
DEFAULT_HELDOUT_FRAC = 0.10
DEFAULT_CAPTURE_BATCH_SIZE = 32
#: the dtype the SERVER materialises the matrix in (rho's second reading);
#: the wave's serving dtype, never inferred from what the fit ran in
DEFAULT_SERVING_DTYPE = "bfloat16"
SERVING_DTYPES = ("bfloat16", "float16", "float32")
#: the thresholds whose eligible counts are recorded (spec open question 1
#: is answered from the artifact, never from a refit)
ELIGIBLE_COUNT_THRESHOLDS = (0.90, 0.95, 0.98)
#: spec Assumption A5
A5_MIN_ELIGIBLE = 3
A5_FALLBACK_TOP_K = 6
A5_STOP_AUROC = 0.90
#: how far the two float64 readings of the deployed score (matmul vs the
#: rank-1 factorisation) may disagree, relative to the ROUNDOFF SCALE
#: max‖h‖·‖M‖ (see rank1_agreement). M is an fp32 object rank-1 only to fp32
#: rounding (~1e-7 relative), so the statistic sits ~1e-9–1e-8 on a valid
#: fit; 1e-5 is three decades of headroom.
DEFAULT_RANK1_TOL = 1e-5

SCORE_EXPRESSION = "torch.linalg.vector_norm(H @ M, dim=-1)"
SCORE_ORIENTATION = ("deployed row-vector (capeval/serving/lens_adapters.py "
                     "_adapt_alphasteer -> OP_PREFILL_LAST_MATMUL: h_last @ M; "
                     "extraction/alphasteer.AlphaSteerObject._edit)")


class A5StopError(RuntimeError):
    """Spec Assumption A5's STOP: fewer than three in-band layers reach
    AUROC 0.90 — the rule has no usable layer set on this model and the
    user decides, not the pipeline."""


# --------------------------------------------------------------------------- #
# normalisation + the content-grouped split (plan §B.1)                        #
# --------------------------------------------------------------------------- #

_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)
_WS_RE = re.compile(r"\s+")


def normalise_prompt(text: str) -> str:
    """The ONE prompt identity: NFKC -> lowercase -> punctuation to space ->
    whitespace collapse. Shared by the leakage check (plan Phase 0) and the
    split below, so the two can never disagree about what "the same
    prompt" means."""
    t = unicodedata.normalize("NFKC", str(text))
    t = t.lower()
    t = _PUNCT_RE.sub(" ", t)
    return _WS_RE.sub(" ", t).strip()


def split_rank(identity: str, *, seed: int = 0, salt: str = SPLIT_SALT) -> str:
    """Hash rank of a content identity — the split's ordering key. Order-
    independent by construction (a hash of CONTENT, never a position), the
    same shape as ``eval.pool_sampling.hash_rank``. The seed is part of the
    hash, so ``split_seed`` genuinely reseeds the assignment."""
    return hashlib.sha256(
        f"{salt}|{seed}|{identity}".encode("utf-8")).hexdigest()


def whole_list_slices(n: int) -> list[dict]:
    """The one-source slice table used when a list carries no
    ``*.slices.json`` beside it."""
    return [{"name": "all", "start": 0, "end": int(n)}]


def _check_slices(slices: list[dict], n: int) -> list[dict]:
    out = []
    pos = 0
    for s in slices:
        start, end = int(s["start"]), int(s["end"])
        if start != pos or end <= start:
            raise ValueError(
                f"slices must tile [0, {n}) contiguously in order; got "
                f"{s['name']!r} = [{start}, {end}) at position {pos}")
        out.append({"name": str(s["name"]), "start": start, "end": end})
        pos = end
    if pos != n:
        raise ValueError(f"slices end at {pos}, the list has {n} rows")
    return out


@dataclass
class ContentGroupedSplit:
    """One side's (benign or harmful) split. ``fit_idx`` / ``heldout_idx``
    are ORIGINAL row indices — every row of the list appears in exactly one
    of them, duplicates included (grouping decides the side, never
    deduplicates)."""

    side: str
    n_rows: int
    heldout_frac_target: float
    seed: int
    salt: str
    slices: list[dict]
    fit_idx: list[int]
    heldout_idx: list[int]
    heldout_identities: list[str]            # in hash-rank order
    realised_by_source: dict[str, dict]      # name -> rows/fit/heldout/target
    n_groups: int
    n_groups_spanning_sources: int
    n_unique_heldout: int
    content_disjoint_verified: bool
    index_disjoint_verified: bool
    sha256: str = ""

    def payload(self) -> dict:
        d = asdict(self)
        return d


def split_side_content_grouped(
    prompts: list[str], *, side: str, slices: list[dict] | None = None,
    heldout_frac: float = DEFAULT_HELDOUT_FRAC, seed: int = 0,
    salt: str = SPLIT_SALT,
) -> ContentGroupedSplit:
    """The ``stratified-v1-content-grouped`` split of ONE list (plan §B.1):

    1. identity = :func:`normalise_prompt`; rows are grouped by it (a group
       may span source-slice boundaries — that is the case grouping exists
       for);
    2. whole groups are assigned by :func:`split_rank` — no group is split;
    3. each group is charged to the source slice of its LOWEST row index;
       slices are processed in order, and a slice takes its charged groups in
       rank order until its held-out ROW count (rows of held-out groups lying
       inside the slice, from any group) first reaches
       ``ceil(heldout_frac * slice_rows)`` — so the realised counts can differ
       from the nominal 10 % by up to one group, and are RECORDED;
    4. every row and its multiplicity is preserved;
    5. disjointness is verified at the CONTENT level (the gate) and at the
       index level (recorded, weaker)."""
    if not 0.0 < heldout_frac < 1.0:
        raise ValueError(f"heldout_frac must be in (0, 1), got {heldout_frac}")
    n = len(prompts)
    if n == 0:
        raise ValueError(f"{side}: empty prompt list")
    slices = _check_slices(slices or whole_list_slices(n), n)

    norm = [normalise_prompt(p) for p in prompts]
    groups: dict[str, list[int]] = {}
    for i, ident in enumerate(norm):
        groups.setdefault(ident, []).append(i)      # ascending by construction

    def source_of(i: int) -> int:
        for k, s in enumerate(slices):
            if s["start"] <= i < s["end"]:
                return k
        raise AssertionError(i)

    charged: dict[int, list[str]] = {k: [] for k in range(len(slices))}
    n_spanning = 0
    for ident, idxs in groups.items():
        srcs = {source_of(i) for i in idxs}
        if len(srcs) > 1:
            n_spanning += 1
        charged[source_of(idxs[0])].append(ident)

    heldout_rows: set[int] = set()
    heldout_idents: list[str] = []
    per_source_target = {}
    for k, s in enumerate(slices):
        rows_in = s["end"] - s["start"]
        target = math.ceil(heldout_frac * rows_in)
        per_source_target[s["name"]] = target

        def in_slice_count() -> int:
            return sum(1 for i in heldout_rows if s["start"] <= i < s["end"])

        ranked = sorted(charged[k], key=lambda ident: split_rank(
            ident, seed=seed, salt=salt))
        for ident in ranked:
            if in_slice_count() >= target:
                break
            heldout_rows.update(groups[ident])
            heldout_idents.append(ident)

    heldout_idx = sorted(heldout_rows)
    fit_idx = [i for i in range(n) if i not in heldout_rows]
    fit_set = {norm[i] for i in fit_idx}
    held_set = {norm[i] for i in heldout_idx}
    content_disjoint = not (fit_set & held_set)
    index_disjoint = (not (set(fit_idx) & set(heldout_idx))
                      and len(fit_idx) + len(heldout_idx) == n)
    if not content_disjoint:
        raise AssertionError(
            f"{side}: content-grouped split leaked "
            f"{len(fit_set & held_set)} identities across sides — impossible "
            "by construction; refusing to hand this split to a solve")
    realised = {}
    for s in slices:
        rows = list(range(s["start"], s["end"]))
        h = sum(1 for i in rows if i in heldout_rows)
        realised[s["name"]] = {"rows": len(rows), "fit": len(rows) - h,
                               "heldout": h,
                               "target_heldout": per_source_target[s["name"]]}
    split = ContentGroupedSplit(
        side=side, n_rows=n, heldout_frac_target=float(heldout_frac),
        seed=int(seed), salt=salt, slices=slices,
        fit_idx=fit_idx, heldout_idx=heldout_idx,
        heldout_identities=heldout_idents, realised_by_source=realised,
        n_groups=len(groups), n_groups_spanning_sources=n_spanning,
        n_unique_heldout=len(held_set),
        content_disjoint_verified=content_disjoint,
        index_disjoint_verified=index_disjoint)
    split.sha256 = hashlib.sha256(json.dumps(
        {"side": side, "fit_idx": fit_idx, "heldout_idx": heldout_idx,
         "seed": seed, "salt": salt, "heldout_frac": heldout_frac},
        sort_keys=True).encode("utf-8")).hexdigest()
    return split


def split_v1_payload(benign: ContentGroupedSplit,
                     harmful: ContentGroupedSplit) -> dict:
    """The ``split_v1.json`` body: both sides plus the summary block the
    screen and the fit meta quote (realised counts, never nominal)."""
    both = {"benign": benign, "harmful": harmful}
    payload = {
        "schema": SPLIT_SCHEMA,
        "convention": SPLIT_CONVENTION,
        "salt": benign.salt, "seed": benign.seed,
        "heldout_frac_target": benign.heldout_frac_target,
        "identity": SPLIT_IDENTITY,
        "n_fit_benign": len(benign.fit_idx),
        "n_heldout_benign": len(benign.heldout_idx),
        "n_fit_harmful": len(harmful.fit_idx),
        "n_heldout_harmful": len(harmful.heldout_idx),
        "realised_by_source": {
            **{k: v for k, v in benign.realised_by_source.items()},
            **{k: v for k, v in harmful.realised_by_source.items()}},
        "n_groups_benign": benign.n_groups,
        "n_groups_harmful": harmful.n_groups,
        "n_groups_spanning_sources": (benign.n_groups_spanning_sources
                                      + harmful.n_groups_spanning_sources),
        "content_disjoint_verified": (benign.content_disjoint_verified
                                      and harmful.content_disjoint_verified),
        "index_disjoint_verified": (benign.index_disjoint_verified
                                    and harmful.index_disjoint_verified),
        "sides": {k: v.payload() for k, v in both.items()},
    }
    payload["split_sha256"] = hashlib.sha256(
        (benign.sha256 + "|" + harmful.sha256).encode("utf-8")).hexdigest()
    return payload


def split_summary(payload: dict) -> dict:
    """The compact ``split`` block ``screen_f9.json`` carries (everything
    but the per-row index lists)."""
    return {k: v for k, v in payload.items() if k != "sides"}


# --------------------------------------------------------------------------- #
# capture (plan §B.2)                                                          #
# --------------------------------------------------------------------------- #

#: the serving tokenization (chat-rendered text, no second BOS — plan §A
#: deviation 2). The ONE value the capture seam passes to _collect_last_acts
#: and the binding records, so the two cannot disagree; it is the one binding
#: key capture_binding takes no argument for.
CAPTURE_ADD_SPECIAL_TOKENS = False


def capture_all_layers(
    model, tok, texts: list[str], *, site: str = "pre",
    batch_size: int = DEFAULT_CAPTURE_BATCH_SIZE, layers: list[int] | None = None,
    verify_sample: int = DEFAULT_VERIFY_SAMPLE, log_fn=None,
) -> tuple[dict[int, torch.Tensor], dict]:
    """Last-valid-token activations at ``site`` for EVERY block (or
    ``layers``), one forward per batch, on CPU fp32 — ``{layer: [N, d]}``
    plus the capture record. ``texts`` are chat-rendered strings; they are
    tokenized with ``add_special_tokens=False`` and the first
    ``verify_sample`` prompts' ids are ASSERTED equal to the serving
    render's ids before any forward (:class:`CaptureIdVerifier`) — a
    mismatch refuses, it is never recorded as ``False``. ``log_fn`` gets the
    per-N-batches progress line (alphasteer.CAPTURE_LOG_EVERY_BATCHES);
    logging only, the captured bytes do not depend on it."""
    if layers is None:
        layers = list(range(len(get_decoder_layers(model))))
    verifier = CaptureIdVerifier(tok, n_sample=verify_sample)
    acts = _collect_last_acts(model, tok, texts, list(layers), site,
                              batch_size,
                              add_special_tokens=CAPTURE_ADD_SPECIAL_TOKENS,
                              verifier=verifier, log_fn=log_fn)
    info = {"site": site, "batch_size": int(batch_size),
            "layers": [int(l) for l in layers], "n_rows": len(texts),
            **verifier.summary()}
    return acts, info


CAPTURE_SENTINEL = "capture_done.json"


def acts_layer_path(acts_dir: Path, layer: int) -> Path:
    return Path(acts_dir) / f"L{int(layer):03d}.pt"


def save_acts(acts_dir: Path, acts: dict[int, torch.Tensor],
              sentinel: dict) -> None:
    """``L{l:03d}.pt`` per layer, then the ``capture_done.json`` sentinel
    STRICTLY after every layer file (a torn capture leaves no sentinel and
    is re-run whole — plan §7)."""
    acts_dir = Path(acts_dir)
    acts_dir.mkdir(parents=True, exist_ok=True)
    for l, H in acts.items():
        tmp = acts_layer_path(acts_dir, l).with_suffix(".pt.tmp")
        torch.save(H.detach().float().cpu().contiguous(), tmp)
        os.replace(tmp, acts_layer_path(acts_dir, l))
    tmp = acts_dir / (CAPTURE_SENTINEL + ".tmp")
    tmp.write_text(json.dumps(sentinel, indent=1, sort_keys=True))
    os.replace(tmp, acts_dir / CAPTURE_SENTINEL)


def read_capture_sentinel(acts_dir: Path) -> dict | None:
    p = Path(acts_dir) / CAPTURE_SENTINEL
    if not p.is_file():
        return None
    return json.loads(p.read_text())


def load_acts(acts_dir: Path, layers: list[int]) -> dict[int, torch.Tensor]:
    return {int(l): torch.load(acts_layer_path(acts_dir, l),
                               map_location="cpu", weights_only=True)
            for l in layers}


class CheckpointMismatch(RuntimeError):
    """A reusable checkpoint (a capture dir, a ``screen_rows.jsonl``) was
    made under different inputs than the ones now requested. Never resolved
    by recomputing over it or by accepting it: the operator removes the
    checkpoint deliberately. This is the campaign's worst bug class — a
    wrong number under a valid provenance stamp (decisions D26 #1) — so the
    failure mode is a refusal."""


def _canonical(binding: dict) -> dict:
    """JSON round-trip, so a binding compares equal to its own on-disk copy
    (tuples -> lists, ints/floats/None as JSON renders them)."""
    return json.loads(json.dumps(binding, sort_keys=True))


def binding_sha256(binding: dict) -> str:
    return hashlib.sha256(
        json.dumps(_canonical(binding), sort_keys=True).encode("utf-8")).hexdigest()


_ABSENT = object()


def _flat(binding: dict, prefix: str = "") -> dict:
    """``{"capture": {"benign": {"dtype": ..}}}`` -> ``{"capture.benign.dtype": ..}``,
    so a diff names the leaf that moved rather than dumping a nested dict."""
    out = {}
    for k, v in binding.items():
        if isinstance(v, dict):
            out.update(_flat(v, f"{prefix}{k}."))
        else:
            out[f"{prefix}{k}"] = v
    return out


def binding_diff(want: dict, got: dict | None) -> str:
    if got is None:
        return "the checkpoint carries no binding at all (written before bindings existed)"
    w, g = _flat(want), _flat(got)
    keys = sorted(set(w) | set(g))
    diffs = [f"{k}: checkpoint={g.get(k, _ABSENT)!r} requested={w.get(k, _ABSENT)!r}"
             for k in keys if g.get(k, _ABSENT) != w.get(k, _ABSENT)]
    return ("; ".join(diffs).replace(repr(_ABSENT), "<absent>")
            or "(bindings differ in no key — a canonicalisation bug)")


def resolved_model_revision(model, cfg_revision: str | None) -> str | None:
    """The commit the loaded weights actually came from, else the config's
    pin — the same expression the fit's provenance sidecar records."""
    return getattr(model.config, "_commit_hash", None) or cfg_revision


# --------------------------------------------------------------------------- #
# what a capture is a function of — ONE enumeration, held against the code    #
# --------------------------------------------------------------------------- #
#
# The rule (decisions D32): a checkpoint's identity must name every input
# that can change its bytes. Twice now a binding omitted one (D26 #1: model
# and revision; D32 P1-b: the capture precision and attention kernel), so the
# inputs are enumerated ONCE here, in two forms that the test
# ``test_capture_binding_names_every_model_load_and_capture_input`` holds
# against each other AND against the live code:
#
# - CAPTURE_BINDING_KEYS: the keys a capture sentinel's binding has, exactly.
#   ``capture_binding`` takes every one as a keyword-only argument with no
#   default, so a call site cannot leave one out; ``model_load_identity``
#   supplies the model-load half from the LOADED model (resolved values).
# - CAPTURE_INPUTS: every parameter of the model-load and capture path —
#   ``load_model_and_tokenizer``'s declared parameters, the keywords the
#   drivers pass through its ``**kwargs`` (``fit_class_b.load_class_b_model``),
#   ``capture_all_layers``'s and ``_collect_last_acts``'s parameters — each
#   mapped to the binding key(s) that carry it, or to the REASON it cannot
#   change the captured bytes. The test computes that parameter set from the
#   live signatures and the live driver call, so a new loader keyword or a
#   new capture parameter fails there until it is placed here: bound, or
#   exempted with its reason written down.
#
# Considered and NOT bound, deliberately: the ENVIRONMENT (torch/CUDA build,
# GPU model, cuBLAS kernel choice). None is an input the load/capture path
# takes; the torch build is pinned by the lockfile the code sha covers, and
# the clusters are single-architecture. ``device`` (cpu vs cuda) IS bound —
# it is a loader parameter and the two kernel families are not bitwise.

CAPTURE_BINDING_KEYS = (
    # the model-load half, RESOLVED off the loaded model (model_load_identity)
    "model", "model_revision", "dtype", "attn_implementation", "device",
    # the capture call
    "site", "side", "list_sha256", "n_rows", "batch_size", "add_special_tokens",
    # the code
    "code_sha",
)

#: binding keys no parameter maps to: which list, and the code itself
CAPTURE_BINDING_KEYS_WITHOUT_A_PARAMETER = ("side", "code_sha")

CAPTURE_INPUTS: dict[str, tuple[str, ...] | str] = {
    # load_model_and_tokenizer(name, device, dtype, revision, **kwargs) and the
    # keywords load_class_b_model passes through **kwargs
    "name": ("model",),
    "revision": ("model_revision",),        # RESOLVED: the commit the weights came from
    "dtype": ("dtype",),                    # RESOLVED: model.dtype, not the config string
    "attn_implementation": ("attn_implementation",),   # RESOLVED: config._attn_implementation
    "device": ("device",),                  # cpu vs cuda kernels are not bitwise
    # capture_all_layers(model, tok, texts, site, batch_size, layers,
    # verify_sample) -> _collect_last_acts(model, tok, texts, layers, site,
    # batch_size, add_special_tokens, verifier)
    "model": ("model", "model_revision", "dtype", "attn_implementation", "device"),
    "tok": ("model", "model_revision"),     # loaded from the same (name, revision); its template renders texts
    "texts": ("list_sha256", "n_rows"),     # the pinned list; the render is tok's template + code (system=None)
    "site": ("site",),
    "batch_size": ("batch_size",),          # padding width changes the kernels' reduction order
    "add_special_tokens": ("add_special_tokens",),
    "layers": ("which layer FILES are written — changes no layer's bytes; the "
               "sentinel lists them and reuse requires every listed file"),
    "verify_sample": "how many prompts the ids assertion samples; it asserts, it changes nothing",
    "verifier": "the ids assertion itself; a mismatch raises, it never alters the forward",
    "log_fn": ("progress logging only: handed row counts and timings every N batches, "
               "never a tensor; the forward and the pooled rows are identical with or "
               "without it (test-pinned)"),
    # load_model_and_tokenizer(..., leading_special_tokens=None): the doc-CE fits'
    # leading-token POLICY (sisteer.extraction.leading_tokens, R2-D50). It acts
    # only on what the tokenizer prepends under add_special_tokens=True — it
    # installs or validates the `<bos> $A` post-processor — and the capture seam
    # tokenizes with add_special_tokens=False (CAPTURE_ADD_SPECIAL_TOKENS, held
    # by executing the seam in the binding test), where a post-processor adds
    # nothing; load_class_b_model passes no policy, so the tokenizer is as
    # loaded either way. Proved, not asserted: test_leading_token_policy_cannot
    # _reach_the_capture_seam installs the policy on a bare tokenizer and shows
    # the add_special_tokens=True ids move while the seam's ids are unchanged.
    "leading_special_tokens": ("the doc-CE leading-token policy (leading_tokens.apply_leading_"
                               "special_tokens) rewrites only the add_special_tokens=True "
                               "post-processor; the capture seam tokenizes with "
                               "add_special_tokens=False, where it cannot change one id "
                               "(test-proved on a bare tokenizer); load_class_b_model passes none"),
}


def model_load_identity(model, *, name: str, cfg_revision: str | None) -> dict:
    """The model-load half of a capture's identity, read off the LOADED
    model — resolved values, never the config's request: the config asks
    for ``dtype: bfloat16`` and ``attn_implementation: sdpa``; what the
    forward pass actually ran in is what the binding records (transformers
    resolves a revision to a commit, a ``None`` dtype by device, and an
    unsupported attention implementation to a fallback). ``name`` is the
    config's model id, kept verbatim as the identity of the REQUEST
    (``config._name_or_path`` can be a local path)."""
    return {
        "model": name,
        "model_revision": resolved_model_revision(model, cfg_revision),
        "dtype": str(model.dtype).replace("torch.", ""),
        "attn_implementation": getattr(model.config, "_attn_implementation", None),
        "device": model.device.type,
    }


def capture_binding(*, model: str, model_revision: str | None, dtype: str,
                    attn_implementation: str | None, device: str, site: str,
                    side: str, list_sha256: str | None, n_rows: int,
                    batch_size: int, code_sha: str) -> dict:
    """Everything a captured activation matrix is a function of — exactly
    :data:`CAPTURE_BINDING_KEYS`, every one a keyword-only argument with no
    default except the seam-fixed tokenization constant
    (:data:`CAPTURE_ADD_SPECIAL_TOKENS`; the model-load half comes from
    :func:`model_load_identity` — splat it): the model, its resolved
    revision, the precision it was loaded
    in and the attention kernel it ran (fp32 and bf16 captures of the same
    prompts differ by ~3e-3 relative L2 on the tiny fixture — D32 P1-b), the
    device family, the site, the list (its sha AND row count), the batch
    width (padding width changes the kernels' reduction order — widths are
    not bitwise), the serving tokenization, and the code. A sentinel is
    reusable iff its binding equals this dict."""
    if not list_sha256:
        raise ValueError(f"{side}: the prompt list carries no sha256 — the "
                         "exemption path fits the pinned authors' lists "
                         "(class_b.*_prompts_path) and binds captures to them")
    out = _canonical({
        "model": model, "model_revision": model_revision, "dtype": str(dtype),
        "attn_implementation": attn_implementation, "device": str(device),
        "site": site, "side": side, "list_sha256": list_sha256,
        "n_rows": int(n_rows), "batch_size": int(batch_size),
        "add_special_tokens": CAPTURE_ADD_SPECIAL_TOKENS, "code_sha": code_sha})
    assert tuple(sorted(out)) == tuple(sorted(CAPTURE_BINDING_KEYS)), sorted(out)
    return out


def is_capture_binding(b) -> bool:
    return isinstance(b, dict) and sorted(b) == sorted(CAPTURE_BINDING_KEYS)


def capture_or_load_side(
    model, tok, prompts: list[str], *, side: str, acts_dir: Path,
    binding: dict, verify_sample: int = DEFAULT_VERIFY_SAMPLE,
    layers: list[int] | None = None, log_fn=None,
) -> tuple[dict[int, torch.Tensor], dict]:
    """The ONE capture seam (the screen CLI and the fit driver both call it,
    so the fit's matrices are the screen's matrices bit for bit): reuse
    ``acts_dir/<side>`` when its sentinel's binding equals ``binding`` and
    every layer file is present; REFUSE (:class:`CheckpointMismatch`) when a
    sentinel is present under any other binding; capture every block and
    write the sentinel otherwise. ``layers`` restricts which layers are
    LOADED on reuse (the capture itself is always every block). Returns
    ``(acts, sentinel)``."""
    side_dir = Path(acts_dir) / side
    sent = read_capture_sentinel(side_dir)
    if sent is not None:
        if sent.get("binding") != binding:
            raise CheckpointMismatch(
                f"{side_dir}: the capture there was made under different "
                f"inputs — {binding_diff(binding, sent.get('binding'))}. "
                "Refusing to reuse it or to capture over it; remove the "
                "directory deliberately if that capture is obsolete.")
        want = list(sent["layers"]) if layers is None else [int(l) for l in layers]
        if all(acts_layer_path(side_dir, l).is_file() for l in sent["layers"]):
            if log_fn:
                log_fn(f"{side}: capture complete at {side_dir} (binding "
                       f"{sent['binding_sha256'][:12]} matched) -- reusing")
            return load_acts(side_dir, want), sent
        if log_fn:
            log_fn(f"{side}: sentinel at {side_dir} lists layer files that are "
                   "missing -- re-capturing whole")
    texts = [build_chat_prompt(tok, p, system=None) for p in prompts]
    if log_fn:
        log_fn(f"{side}: capturing {len(texts)} rows x every block at "
               f"site={binding['site']} width {binding['batch_size']} -> {side_dir}")
    t0 = time.time()
    acts, info = capture_all_layers(model, tok, texts, site=binding["site"],
                                    batch_size=binding["batch_size"],
                                    verify_sample=verify_sample,
                                    log_fn=(lambda m: log_fn(f"{side}: {m}")) if log_fn else None)
    sentinel = {**info, "side": side, "binding": binding,
                "binding_sha256": binding_sha256(binding),
                "seconds": time.time() - t0}
    save_acts(side_dir, acts, sentinel)
    if log_fn:
        log_fn(f"{side}: captured {len(texts)} rows x {len(acts)} layers at "
               f"site={binding['site']} in {sentinel['seconds']:.0f}s "
               f"({len(texts) / max(sentinel['seconds'], 1e-9):.1f} rows/s) -> {side_dir}")
    if layers is not None:
        acts = {int(l): acts[int(l)] for l in layers}
    return acts, sentinel


# --------------------------------------------------------------------------- #
# the (layer, p) checkpoint: screen_rows.jsonl, bound to its inputs           #
# --------------------------------------------------------------------------- #


SCREEN_BINDING_KEYS = (
    "schema", "capture", "split_sha256", "n_blocks", "hidden_dim", "layers",
    "p_grid", "lambda_reg", "depth_band", "code_sha",
)


def screen_binding(*, capture: dict, split_sha256: str, n_blocks: int,
                   hidden_dim: int, layers: list[int], p_grid, lambda_reg: float,
                   depth_band, code_sha: str) -> dict:
    """Everything a (layer, p) row is a function of — exactly
    :data:`SCREEN_BINDING_KEYS`: the two captures it was solved on, as their
    FULL :func:`capture_binding` dicts under ``capture[side]`` (a row inherits
    every capture input — model, revision, precision, kernel, width, list —
    by construction rather than by re-listing some of them), the split, the
    solver settings, and the code. ``screen_rows.jsonl`` opens with this
    dict and ``screen_f9.json`` records it as ``inputs_binding``; a resume,
    a complete-artifact no-op, or a fit under any other dict refuses."""
    if set(capture) != {"benign", "harmful"} or not all(
            is_capture_binding(b) for b in capture.values()):
        raise ValueError("screen_binding needs capture={'benign': <capture_binding>, "
                         f"'harmful': <capture_binding>}}, got {capture!r}")
    out = _canonical({
        "schema": SCHEMA, "capture": capture,
        "n_blocks": int(n_blocks), "hidden_dim": int(hidden_dim),
        "split_sha256": split_sha256,
        "layers": [int(l) for l in layers], "p_grid": [float(p) for p in p_grid],
        "lambda_reg": float(lambda_reg),
        "depth_band": [float(depth_band[0]), float(depth_band[1])],
        "code_sha": code_sha})
    assert tuple(sorted(out)) == tuple(sorted(SCREEN_BINDING_KEYS)), sorted(out)
    return out


class RowCheckpoint:
    """``screen_rows.jsonl``: a HEADER record carrying the binding, then one
    record per solved (layer, p), appended as each is solved.

    Resume recovers every complete record before an incomplete FINAL one
    (a write interrupted mid-append — unterminated, or unparseable), TRUNCATES
    that tail so the next append starts on a clean line, and leaves its cell
    to be recomputed; an unparseable INTERIOR record is corruption, not
    interruption, and refuses. A header under a different binding refuses
    (:class:`CheckpointMismatch`) — the cells were solved on other inputs and
    must not be published under these."""

    HEADER_KEY = "checkpoint_binding"

    def __init__(self, path: Path, binding: dict, *, resume: bool, log_fn=None):
        self.path = Path(path)
        self.binding = _canonical(binding)
        self.rows: list[dict] = []
        self.truncated_torn_tail = False
        if resume:
            header, rows = self._recover(log_fn)
            if header is not None:
                got = header.get(self.HEADER_KEY)
                if got != self.binding:
                    raise CheckpointMismatch(
                        f"{self.path}: the {len(rows)} solved cells there were "
                        f"computed under different inputs — "
                        f"{binding_diff(self.binding, got)}. Refusing to publish "
                        "them under the requested split/model/rule; remove the "
                        "checkpoint deliberately if it is obsolete.")
                self.rows = rows
                return
        elif self.path.exists():
            self.path.unlink()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(
            {self.HEADER_KEY: self.binding,
             "binding_sha256": binding_sha256(self.binding)}, sort_keys=True) + "\n")

    def _recover(self, log_fn) -> tuple[dict | None, list[dict]]:
        if not self.path.is_file():
            return None, []
        data = self.path.read_bytes()
        if data.endswith(b"\n"):
            body, tail = data, b""
        else:
            cut = data.rfind(b"\n") + 1
            body, tail = data[:cut], data[cut:]
        lines = body.split(b"\n")[:-1]           # every terminated line
        entries = [(ln, True) for ln in lines] + ([(tail, False)] if tail else [])
        records: list[dict] = []
        keep_bytes = 0
        for i, (line, terminated) in enumerate(entries):
            is_last = i == len(entries) - 1
            if not line.strip():
                keep_bytes += len(line) + 1
                continue
            rec = None
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                if not is_last:
                    raise ValueError(
                        f"{self.path}: record {i} is not JSON and is not the "
                        "final record — an interior record is corrupt (not a "
                        "torn append); refusing to resume from this checkpoint")
            if is_last and (rec is None or not terminated):
                # the torn tail: an unparseable or unterminated FINAL record
                with self.path.open("r+b") as f:
                    f.truncate(keep_bytes)
                self.truncated_torn_tail = True
                if log_fn:
                    log_fn(f"{self.path}: truncated a torn final record "
                           f"({len(line)} bytes); its cell will be recomputed")
                break
            records.append(rec)
            keep_bytes += len(line) + 1
        if not records:
            return None, []
        header = records[0]
        if self.HEADER_KEY not in header:
            raise CheckpointMismatch(
                f"{self.path}: the first record is not a binding header — the "
                "checkpoint predates input bindings; remove it deliberately")
        return header, records[1:]

    def append(self, row: dict) -> None:
        with self.path.open("a") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")


# --------------------------------------------------------------------------- #
# scoring (plan §B.4, D12): the ONE helper                                     #
# --------------------------------------------------------------------------- #


def deployed_score(H: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
    """``‖h @ M‖`` per row — the norm of the edit the SERVER would add for
    each held-out prompt (``h_last @ M``, row-vector orientation). This is
    the single scoring helper: the layer screen and the released-matrix
    recovery reading both call it, so the two conventions cannot drift.
    ``H`` [N, d], ``M`` [d, d]; returns [N]."""
    return torch.linalg.vector_norm(H @ M, dim=-1)


def rank1_score(H: torch.Tensor, u: torch.Tensor,
                r: torch.Tensor) -> torch.Tensor:
    """``|h·u|·‖r‖`` — the same quantity under the exact factorisation
    ``M = u rᵀ``: one dot product per prompt instead of a d×d matmul, and
    monotone-identical ranking. Used as the cross-check of
    :func:`deployed_score` (and available to a caller who wants the O(N d)
    pass)."""
    return (H @ u).abs() * torch.linalg.vector_norm(r)


def rank1_agreement(deployed: torch.Tensor, rank1: torch.Tensor,
                    H: torch.Tensor, M: torch.Tensor) -> float:
    """``max |s − s'| / (max ‖h‖ · ‖M‖_F)`` — both scores evaluated in
    float64, and relative to the ROUNDOFF SCALE, not to the largest score.
    ``M`` is an fp32 object: ``P @ (A⁺ Xᵀ R)`` is rank-1 only to fp32
    rounding (``‖M − u rᵀ‖/‖M‖ ~ 1e-7``), and every row's score carries
    that rounding at magnitude ``‖h‖·‖M‖`` BEFORE the null-space
    cancellation makes the score itself small. Dividing by the max score
    therefore reads cancellation as disagreement: on nearly nulled
    activations (shallow layers sharing a template token) a numerically
    valid, perfectly separating fit exceeded 1e-5 that way — in fp32, and
    still in float64 (D26 #4). Against ``max‖h‖·‖M‖`` a valid fit sits at
    ~1e-9–1e-8; a genuinely non-rank-1 M that moves any row's score by
    more than 1e-5 of that scale still refuses."""
    scale = float(torch.linalg.vector_norm(H.double(), dim=-1).max()
                  * M.double().norm())
    return (float((deployed.double() - rank1.double()).abs().max())
            / max(scale, 1e-30))


def auroc(pos: torch.Tensor, neg: torch.Tensor) -> float:
    """Exact Mann–Whitney AUROC of ``pos`` (harmful) over ``neg`` (benign):
    ``U / (n_pos·n_neg)`` with tied scores counted 0.5 (average ranks)."""
    pos = pos.detach().double().flatten().cpu()
    neg = neg.detach().double().flatten().cpu()
    n_p, n_n = pos.numel(), neg.numel()
    if n_p == 0 or n_n == 0:
        raise ValueError("auroc needs at least one score on each side")
    scores = torch.cat([pos, neg])
    order = torch.argsort(scores, stable=True)
    sorted_scores = scores[order]
    # average rank per tie group (1-based)
    _, inverse, counts = torch.unique_consecutive(
        sorted_scores, return_inverse=True, return_counts=True)
    ends = torch.cumsum(counts, 0).double()
    starts = ends - counts.double() + 1.0
    avg = (starts + ends) / 2.0
    ranks_sorted = avg[inverse]
    ranks = torch.empty_like(ranks_sorted)
    ranks[order] = ranks_sorted
    u = float(ranks[:n_p].sum()) - n_p * (n_p + 1) / 2.0
    return u / (n_p * n_n)


# --------------------------------------------------------------------------- #
# the screen loop (plan §B.3): eigh once per layer, ridge per (layer, p)      #
# --------------------------------------------------------------------------- #


def in_depth_band(layer: int, n_blocks: int,
                  band: tuple[float, float] = DEFAULT_DEPTH_BAND) -> bool:
    """``lo ≤ l / n_blocks ≤ hi``, inclusive at both ends (a 1e-9 slack
    absorbs fp on exact boundaries such as 6/40 = 0.15). Reproduces the
    plan's table: gemma4e4b 42 -> 7–27, qwen35_4b 32 -> 5–20,
    nemotron3nano4b 42 -> 7–27, glm46vflash 40 -> 6–26, llama8b 32 -> 5–20."""
    depth = layer / float(n_blocks)
    lo, hi = band
    return (lo - 1e-9) <= depth <= (hi + 1e-9)


def band_layers(n_blocks: int, band=DEFAULT_DEPTH_BAND) -> list[int]:
    return [l for l in range(n_blocks) if in_depth_band(l, n_blocks, band)]


def screen_layers_and_p(
    *, benign_fit: dict[int, torch.Tensor], harmful_fit: dict[int, torch.Tensor],
    benign_held: dict[int, torch.Tensor], harmful_held: dict[int, torch.Tensor],
    layers: list[int], n_blocks: int, p_grid=DEFAULT_P_GRID,
    lambda_reg: float = DEFAULT_LAMBDA_REG, depth_band=DEFAULT_DEPTH_BAND,
    refusals: dict[int, torch.Tensor] | None = None, device: str = "cpu",
    rank1_tol: float = DEFAULT_RANK1_TOL, existing_rows: list[dict] | None = None,
    on_row=None, log_fn=None,
) -> list[dict]:
    """One row per (layer, p) — the ``curve`` block of ``screen_f9.json``.
    Per layer: ONE benign Gram + ONE eigh + one refusal direction; per p:
    a prefix projector, the ridge (one ``psd_pinv`` eigh), the DEPLOYED
    score on the held-out rows, the AUROC, and the rank-1 cross-check.
    ``existing_rows`` (a resumed ``screen_rows.jsonl``) are kept and their
    (layer, p) cells skipped; ``on_row`` receives each new row as it is
    solved (the append-only checkpoint hook)."""
    rows = list(existing_rows or [])
    done = {(int(r["layer"]), float(r["p"])) for r in rows}
    ones_cache: dict[int, torch.Tensor] = {}
    for l in layers:
        l = int(l)
        todo = [float(p) for p in p_grid if (l, float(p)) not in done]
        if not todo:
            continue
        Hb = benign_fit[l].to(device).float()
        Hm = harmful_fit[l].to(device).float()
        Hb_h = benign_held[l].to(device).float()
        Hm_h = harmful_held[l].to(device).float()
        H_held = torch.cat([Hm_h, Hb_h], dim=0)
        n_h = Hm_h.shape[0]
        d = Hb.shape[1]
        r = (refusals[l].to(device).float() if refusals and l in refusals
             else refusal_direction(Hm, Hb))
        t0 = time.time()
        gram = Hb.t() @ Hb                       # null_space_projector's Gram
        evals, evecs = _eigh_with_jitter(gram)   # ... its eigh, ONE per layer
        eigh_seconds = time.time() - t0
        benign_sq = Hb.pow(2).sum()
        N = Hm.shape[0]
        if N not in ones_cache:
            ones_cache[N] = torch.ones(N, device=device)
        ones = ones_cache[N]
        for p in todo:
            t1 = time.time()
            P, k = projector_from_prefix(evecs, p)   # ... its rank rule per p
            leakage = float((Hb @ P).pow(2).sum() / (benign_sq + 1e-8))
            # the trainer's ridge, through the trainer's own functions —
            # ridge_delta itself is not callable here because the rank-1
            # factor u below needs pinv(A)
            X, R, A, b_mat = _ridge_system(Hm, P, r, lambda_reg)
            A_pinv, pinv_info = psd_pinv(A)
            tilde = A_pinv @ b_mat
            M = P @ tilde
            u = P @ (A_pinv @ (X.t() @ ones))    # M = u r^T
            ridge = _ridge_info(Hm, X, R, tilde, lambda_reg)
            # PRODUCTION convention: fp32 H @ M — the ranking input. The two
            # DIAGNOSTIC readings are float64 (rank1_agreement).
            s = deployed_score(H_held, M)
            s64 = deployed_score(H_held.double(), M.double())
            s_rank1 = rank1_score(H_held.double(), u.double(), r.double())
            agreement = rank1_agreement(s64, s_rank1, H_held, M)
            rank1_resid = float((M - torch.outer(u, r)).norm()
                                / (M.norm() + 1e-30))
            if agreement > rank1_tol:
                raise AssertionError(
                    f"layer {l} p {p}: deployed score and its rank-1 form "
                    f"disagree by {agreement:.3e} (rel. to max‖h‖·‖M‖), "
                    f"above {rank1_tol:.1e} — the fitted M is not rank-1 to "
                    f"fp tolerance (rank-1 residual {rank1_resid:.3e}); "
                    "refusing to rank layers on it")
            a = auroc(s[:n_h], s[n_h:])
            row = {
                "layer": l, "depth": l / float(n_blocks),
                "in_band": in_depth_band(l, n_blocks, depth_band), "p": p,
                "auroc": a, "leakage": leakage,
                "malicious_energy_kept": ridge["malicious_energy_kept"],
                "regression_residual": ridge["regression_residual"],
                "nullspace_rank": int(k), "pinv_rank": int(pinv_info["pinv_rank"]),
                "matrix_frobenius": float(M.norm()),
                "matrix_rank1_residual": rank1_resid,
                "rank1_agreement_max_rel": agreement,
                "refusal_norm": float(r.norm()),
                "gram_top_eigenvalue": float(evals[-1]),
                "gram_bottom_eigenvalue": float(evals[0]),
                "n_heldout_harmful": int(n_h), "n_heldout_benign": int(Hb_h.shape[0]),
                "n_fit_harmful": int(N), "n_fit_benign": int(Hb.shape[0]),
                "hidden_dim": int(d), "lambda_reg": float(lambda_reg),
                "eigh_seconds_layer": eigh_seconds,
                "solve_seconds": time.time() - t1,
            }
            rows.append(row)
            done.add((l, p))
            if on_row is not None:
                on_row(row)
            if log_fn is not None:
                log_fn({f"screen/auroc_p{p}": a, "layer": l})
    rows.sort(key=lambda r: (r["layer"], r["p"]))
    return rows


# --------------------------------------------------------------------------- #
# selection (plan §B.5; spec Fixed decision 2 + A5)                            #
# --------------------------------------------------------------------------- #


def layer_eligible(*, layer: int, auroc: float, auroc_threshold: float) -> list[str]:
    """THE eligibility predicate (:data:`ELIGIBILITY_RULE`), shared by the
    screen's :func:`select_layers` and the released-Llama
    :func:`released_matrix_reading` so the gate can never validate a rule the
    fits do not run (D32/C7). Returns the reasons the layer is NOT eligible —
    an empty list means eligible. The depth band is the caller's (chosen =
    eligible ∩ band); leakage is deliberately not a parameter (D38)."""
    reasons = []
    if int(layer) == 0:
        reasons.append("layer 0 is never eligible")
    if float(auroc) < float(auroc_threshold):
        reasons.append(f"auroc {float(auroc):.4f} < {auroc_threshold}")
    return reasons


def select_layers(
    rows: list[dict], *, n_blocks: int, depth_band=DEFAULT_DEPTH_BAND,
    auroc_threshold: float = DEFAULT_AUROC_THRESHOLD, p_grid=DEFAULT_P_GRID,
    count_thresholds=ELIGIBLE_COUNT_THRESHOLDS,
    min_eligible: int = A5_MIN_ELIGIBLE, fallback_top_k: int = A5_FALLBACK_TOP_K,
    stop_auroc: float = A5_STOP_AUROC,
) -> dict:
    """``p_l = argmax_p AUROC(l, p)`` (ties -> the SMALLER p); eligible iff
    :func:`layer_eligible` — ``AUROC(l, p_l) ≥ threshold``, layer 0 never
    (AUROC ONLY; leakage is carried per layer as a diagnostic, D38); chosen =
    eligible ∩ band. A5: fewer than ``min_eligible`` in band -> the
    top-``fallback_top_k`` in band by AUROC, FLAGGED; fewer than
    ``min_eligible`` in band reaching ``stop_auroc`` -> :class:`A5StopError`.
    Eligible counts at every ``count_thresholds`` value are recorded."""
    grid = [float(p) for p in p_grid]
    by_layer: dict[int, list[dict]] = {}
    for r in rows:
        if float(r["p"]) not in grid:
            raise ValueError(f"row p={r['p']} is not in the grid {grid}")
        by_layer.setdefault(int(r["layer"]), []).append(r)
    per_layer = []
    for l in sorted(by_layer):
        cells = by_layer[l]
        have = sorted(float(c["p"]) for c in cells)
        if have != sorted(grid):
            raise ValueError(f"layer {l}: rows cover p={have}, grid is {grid}")
        best = sorted(cells, key=lambda c: (-float(c["auroc"]), float(c["p"])))[0]
        a, leak = float(best["auroc"]), float(best["leakage"])
        in_band = in_depth_band(l, n_blocks, depth_band)
        reasons = layer_eligible(layer=l, auroc=a, auroc_threshold=auroc_threshold)
        per_layer.append({
            "layer": l, "depth": l / float(n_blocks), "in_band": in_band,
            "p_star": float(best["p"]), "auroc": a,
            "leakage": leak,          # DIAGNOSTIC — reported, never gated (D38)
            "eligible": not reasons,
            "reason": None if not reasons else "; ".join(reasons),
        })
    band = [pl for pl in per_layer if pl["in_band"] and pl["layer"] != 0]
    counts = {f"{t:.2f}": sum(1 for pl in band if pl["auroc"] >= t)
              for t in count_thresholds}
    eligible_band = sorted(pl["layer"] for pl in band if pl["eligible"])
    fallback = None
    chosen = eligible_band
    if len(eligible_band) < min_eligible:
        cands = sorted(band, key=lambda pl: (-pl["auroc"], pl["layer"]))
        n_at_stop = sum(1 for pl in cands if pl["auroc"] >= stop_auroc)
        if n_at_stop < min_eligible:
            raise A5StopError(
                f"A5 STOP: only {len(eligible_band)} in-band layer(s) reach "
                f"AUROC {auroc_threshold} and only {n_at_stop} reach "
                f"{stop_auroc} (band {list(depth_band)} of {n_blocks} blocks). "
                "The rule has no usable layer set on this model — stop and "
                "ask the user.")
        chosen = sorted(pl["layer"] for pl in cands[:fallback_top_k])
        fallback = {
            "taken": True,
            "rule": (f"fewer than {min_eligible} in-band layers reached "
                     f"AUROC {auroc_threshold}; took the top-{fallback_top_k} "
                     "in band by AUROC (spec A5; AUROC only — D38)"),
            "n_eligible_at_threshold": len(eligible_band),
            "n_reaching_stop_auroc": n_at_stop,
            "FLAG": "A5 FALLBACK TAKEN — report prominently",
        }
    p_star = {pl["layer"]: pl["p_star"] for pl in per_layer}
    if 0 in chosen:
        raise AssertionError("layer 0 selected — impossible by construction")
    return {
        "per_layer": per_layer,
        "eligibility": ELIGIBILITY_RULE,
        "eligible_counts": counts,
        "band_layers": [pl["layer"] for pl in band],
        "chosen": {"layers": chosen,
                   "nullspace_ratios": {str(l): p_star[l] for l in chosen},
                   "n_layers": len(chosen), "fallback_a5": fallback},
    }


# --------------------------------------------------------------------------- #
# rho at fit time (plan §B.7, in its FIT-SIDE form)                            #
# --------------------------------------------------------------------------- #


def _quantiles(x: torch.Tensor, qs=(0.05, 0.5, 0.95)) -> dict:
    x = x.detach().double().flatten().cpu()
    q = torch.quantile(x, torch.tensor(qs, dtype=torch.float64))
    return {f"p{int(round(qq * 100)):02d}": float(v) for qq, v in zip(qs, q)}


def rho_by_layer(
    harmful_held: dict[int, torch.Tensor], Ms: dict[int, torch.Tensor], *,
    scale: float = 1.0, model_dtype: torch.dtype = torch.bfloat16,
    quantiles=(0.05, 0.5, 0.95),
) -> dict[int, dict]:
    """``ρ_l = ‖scale · h M_l‖ / ‖h‖`` over the held-out harmful rows, per
    layer, as quantiles — computed BOTH in fp32 and with ``M`` cast to the
    serving dtype and the matmul done in that dtype (the worker
    materialises ``OP_PREFILL_LAST_MATMUL``'s matrix in the model dtype).
    This is the lambda-grid PROPOSAL (plan §C.1), NOT the acceptance reading:
    the capture is UNSTEERED, so later layers see a stream the earlier edits
    have not yet modified (plan §B.7 bias 1). Spec Fixed decision 3(c) is
    decided on the SERVED composed per-layer ρ read from ``sisteer_rho_record``
    on the ladder standups (D25). ρ is linear in ``scale``, so the ladder
    derives its grid from ``ρ(λ) = λ · ρ(1)``."""
    out = {}
    for l, M in Ms.items():
        H = harmful_held[int(l)].float()
        M32 = M.float().to(H.device)
        hn = torch.linalg.vector_norm(H, dim=-1).clamp(min=1e-30)
        add32 = scale * (H @ M32)
        rho32 = torch.linalg.vector_norm(add32, dim=-1) / hn
        Hd = H.to(model_dtype)
        Md = M32.to(model_dtype)
        add_d = (scale * (Hd @ Md)).float()
        rho_d = torch.linalg.vector_norm(add_d, dim=-1) / hn
        out[int(l)] = {
            "rho_fp32": _quantiles(rho32, quantiles),
            "rho_model_dtype": _quantiles(rho_d, quantiles),
            "model_dtype": str(model_dtype).replace("torch.", ""),
            "scale": float(scale), "n_rows": int(H.shape[0]),
            "rho_dtype_disagreement_p50_rel": float(
                abs(_quantiles(rho32, (0.5,))["p50"]
                    - _quantiles(rho_d, (0.5,))["p50"])
                / max(_quantiles(rho32, (0.5,))["p50"], 1e-30)),
        }
    return out


#: written into fit_meta.json as the object's provenance — plan §5.3's wording
#: (D25: condition (c) is the spec's single SERVED criterion; the fit-side
#: reading is the grid proposal and nothing more)
RHO_NOTE = ("fit-side, UNSTEERED capture: the lambda-grid PROPOSAL (plan B.7, "
            "C.1), NOT an acceptance reading. rho at scale 1.0; rho(lambda) = "
            "lambda * rho(1). Spec Fixed decision 3(c) is decided on the SERVED "
            "composed per-layer rho read from sisteer_rho_record on the ladder "
            "standups (W_L, SISTEER_LENS_RHO_EMIT=1) — see "
            "served_rho_at_chosen_lambda (added when the ladder bakes the "
            "strength, plan C.6) and lambda_ladder.json. D25.")


# --------------------------------------------------------------------------- #
# the released-Llama recovery reading (plan Phase D) — SAME scoring helper    #
# --------------------------------------------------------------------------- #


def released_matrix_reading(
    *, benign_held: dict[int, torch.Tensor], harmful_held: dict[int, torch.Tensor],
    released: dict[int, torch.Tensor], paper_layers: list[int],
    benign_fit: dict[int, torch.Tensor] | None = None,
    paper_ratios: dict[int, float] | None = None,
    auroc_threshold: float = DEFAULT_AUROC_THRESHOLD, recovery_threshold: int = 7,
    zero_norm_tol: float = 0.0,
) -> dict:
    """Spec Fixed decision 7 / plan D.1, on the RELEASED matrices: for each
    released slice score ``‖H @ M_l‖`` on the held-out rows through
    :func:`deployed_score` — the same helper, the same orientation as the
    screen — and compute the AUROC. Eligible = :func:`layer_eligible`, the
    screen's predicate (AUROC ≥ threshold, never layer 0 — AUROC ONLY, D38).
    Leakage for a paper layer is still computed — the ``benign_energy_kept``
    of a projector RECONSTRUCTED from the pinned authors' benign FIT capture
    at the paper's published ratio (``null_space_projector``) — and REPORTED
    per layer as a diagnostic; it is not an eligibility input. A slice with
    no published ratio has ``leakage: None`` (never imputed) and, being
    all-zero, AUROC 0.5. ``passed`` = ``n_recovered ≥ recovery_threshold``
    over ``paper_layers``."""
    per_layer = {}
    norms = {}
    nonzero = []
    for l in sorted(released):
        M = released[l].float()
        fro = float(M.norm())
        norms[str(l)] = fro
        if fro > zero_norm_tol:
            nonzero.append(int(l))
        Hb, Hm = benign_held[l].float(), harmful_held[l].float()
        if fro > zero_norm_tol:
            s_h = deployed_score(Hm, M)
            s_b = deployed_score(Hb, M)
            a = auroc(s_h, s_b)
            op_benign_energy = float(s_b.pow(2).sum() / (Hb.pow(2).sum() + 1e-8))
        else:
            a, op_benign_energy = 0.5, 0.0
        ratio = None if paper_ratios is None else paper_ratios.get(int(l))
        leak = None
        if ratio is not None and benign_fit is not None:
            _, info = null_space_projector(benign_fit[l], float(ratio))
            leak = float(info["benign_energy_kept"])
        reasons = layer_eligible(layer=l, auroc=a, auroc_threshold=auroc_threshold)
        per_layer[str(l)] = {"auroc": a,
                             "leakage": leak,      # DIAGNOSTIC — reported, never gated (D38)
                             "leakage_ratio_used": ratio,
                             "operator_benign_energy": op_benign_energy,
                             "released_norm": fro, "eligible": not reasons,
                             "reason": None if not reasons else "; ".join(reasons)}
    recovered = sorted(l for l in paper_layers if per_layer[str(l)]["eligible"])
    return {
        "rule": (f"AUROC >= {auroc_threshold}, never layer 0 — AUROC ONLY (the "
                 "paper's Appendix F.9 norm separability); leakage is reported "
                 "per layer as a diagnostic, not an eligibility input (D38: the "
                 "retired 1e-3 bound measured an n<d fit's rank deficiency, not "
                 "the projector)"),
        "auroc_threshold": float(auroc_threshold),
        "eligibility": ELIGIBILITY_RULE,
        "score": {"expression": SCORE_EXPRESSION, "orientation": SCORE_ORIENTATION},
        "released_matrix_norms": norms,
        "released_nonzero_layers": nonzero,
        "released_all_layers_nonzero": len(nonzero) == len(released),
        "recovered": recovered, "n_recovered": len(recovered),
        "threshold": int(recovery_threshold),
        "passed": len(recovered) >= recovery_threshold,
        "per_layer": per_layer,
        "tests": ("ACCEPTANCE only — an all-zero released slice forces AUROC "
                  "0.5, so the released file cannot test whether the rule "
                  "REJECTS layers the paper omitted"),
    }


# --------------------------------------------------------------------------- #
# fingerprinted artifact writer (plan §5)                                      #
# --------------------------------------------------------------------------- #


def repo_code_sha(start: Path | None = None) -> str:
    """``git rev-parse HEAD`` of the tree this module runs from, or
    ``"unknown"`` — never raises (an artifact is still written when the
    checkout is a bare export)."""
    here = Path(start or __file__).resolve().parent
    try:
        out = subprocess.run(["git", "-C", str(here), "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=10)
        sha = out.stdout.strip()
        return sha if out.returncode == 0 and len(sha) == 40 else "unknown"
    except Exception:  # noqa: BLE001
        return "unknown"


def _json_default(o):
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, torch.Tensor):
        return o.tolist()
    if isinstance(o, (set, frozenset)):
        return sorted(o)
    raise TypeError(f"not JSON serialisable: {type(o).__name__}")


def canonical_json(payload: dict) -> str:
    return json.dumps(payload, sort_keys=True, indent=1, default=_json_default)


def write_fingerprinted_json(path: Path, payload: dict, *,
                             code_sha: str | None = None) -> dict:
    """Write ``payload`` + ``code_sha`` + ``written_at`` + ``sha256`` (over
    the canonical JSON of everything but the sha) via ``tmp`` ->
    ``os.replace``, then a ``<name>.COMPLETE`` marker STRICTLY after the
    payload. Returns the final payload."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = dict(payload)
    body.pop("sha256", None)
    body["code_sha"] = code_sha or repo_code_sha()
    body["written_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    body["sha256"] = hashlib.sha256(canonical_json(body).encode("utf-8")).hexdigest()
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(canonical_json(body))
    os.replace(tmp, path)
    marker = path.with_name(path.name + ".COMPLETE")
    marker_tmp = marker.with_name(marker.name + ".tmp")
    marker_tmp.write_text(body["sha256"] + "\n")
    os.replace(marker_tmp, marker)
    return body


def verify_fingerprint(payload: dict) -> bool:
    """True iff ``payload["sha256"]`` is the sha over the rest."""
    body = dict(payload)
    want = body.pop("sha256", None)
    if not want:
        return False
    return hashlib.sha256(canonical_json(body).encode("utf-8")).hexdigest() == want


def read_fingerprinted_json(path: Path) -> dict:
    """Read and REFUSE a payload whose sha does not verify or whose
    ``.COMPLETE`` marker is missing — a torn or edited artifact never
    feeds a fit."""
    path = Path(path)
    marker = path.with_name(path.name + ".COMPLETE")
    if not marker.is_file():
        raise FileNotFoundError(
            f"{path}: no COMPLETE marker beside it ({marker.name}) — the "
            "artifact was never finished, or is being written; refusing")
    payload = json.loads(path.read_text())
    if not verify_fingerprint(payload):
        raise ValueError(f"{path}: sha256 does not verify — edited or torn")
    if marker.read_text().strip() != payload["sha256"]:
        raise ValueError(f"{path}: COMPLETE marker names a different sha")
    return payload


# --------------------------------------------------------------------------- #
# the screen_f9.json body (plan §5.1)                                          #
# --------------------------------------------------------------------------- #

SCREEN_REQUIRED_KEYS = (
    "schema", "model_key", "model", "model_revision", "n_blocks", "hidden_dim",
    "site", "rule", "split", "lists", "score", "curve", "per_layer",
    "eligible_counts", "chosen", "capture", "acts_dir",
    # the screen_binding the rows were solved under: the fit holds its own
    # capture inputs to it (D32 P1-a), so a screen without one is not a screen
    "inputs_binding", "inputs_sha256",
)


def rule_block(*, depth_band=DEFAULT_DEPTH_BAND, p_grid=DEFAULT_P_GRID,
               auroc_threshold=DEFAULT_AUROC_THRESHOLD,
               lambda_reg=DEFAULT_LAMBDA_REG) -> dict:
    """The rule as recorded in every artifact. Carries NO leakage bound: the
    ``eligibility`` string says in words that the rule is AUROC-only and that
    leakage is a reported diagnostic (D38); the fit refuses a screen whose
    rule block lacks it (``_check_screen_matches_config``)."""
    return {"name": "f9_auroc",
            "depth_band": [float(depth_band[0]), float(depth_band[1])],
            "p_grid": [float(p) for p in p_grid],
            "auroc_threshold": float(auroc_threshold),
            "lambda_reg": float(lambda_reg),
            "eligibility": ELIGIBILITY_RULE,
            "leakage": "diagnostic only — reported per layer, not an eligibility input (D38)",
            "layer0_never_eligible": True, "argmax_tie_break": "smaller_p"}


def build_screen_payload(
    *, model_key: str, model: str, model_revision: str | None, n_blocks: int,
    hidden_dim: int, site: str, rule: dict, split: dict, lists: dict,
    rows: list[dict], selection: dict, capture: dict, acts_dir: str | None,
    binding: dict, recovery: dict | None = None,
) -> dict:
    """Assemble the ``screen_f9.json`` body from its parts (the fingerprint
    fields are added by :func:`write_fingerprinted_json`). ``binding`` is
    the :func:`screen_binding` the rows were solved under, recorded with
    its digest as ``inputs_binding`` / ``inputs_sha256`` — required, because
    the fit refuses to reproduce a screen that does not say what it was
    computed under."""
    if not isinstance(binding, dict) or sorted(binding) != sorted(SCREEN_BINDING_KEYS):
        raise ValueError("build_screen_payload needs the screen_binding the rows "
                         f"were solved under (keys {list(SCREEN_BINDING_KEYS)}), "
                         f"got {binding!r}")
    payload = {
        "schema": SCHEMA, "model_key": model_key, "model": model,
        "model_revision": model_revision, "n_blocks": int(n_blocks),
        "hidden_dim": int(hidden_dim), "site": site, "rule": rule,
        "split": split, "lists": lists,
        "score": {"expression": SCORE_EXPRESSION,
                  "orientation": SCORE_ORIENTATION,
                  "rank1_agreement_max_rel": max(
                      (float(r["rank1_agreement_max_rel"]) for r in rows),
                      default=None),
                  "rank1_agreement_scale": "max_row_norm * frobenius(M), float64",
                  "auroc": "exact Mann-Whitney, harmful over benign, ties 0.5"},
        "curve": rows,
        "per_layer": selection["per_layer"],
        "eligible_counts": selection["eligible_counts"],
        "band_layers": selection["band_layers"],
        "chosen": selection["chosen"],
        "capture": capture, "acts_dir": acts_dir,
    }
    if recovery is not None:
        payload["recovery"] = recovery
    payload["inputs_binding"] = _canonical(binding)
    payload["inputs_sha256"] = binding_sha256(binding)
    return payload


def check_screen_payload(payload: dict) -> None:
    """Every documented key present and typed; the chosen layers inside the
    band, none zero, every p in the grid."""
    missing = [k for k in SCREEN_REQUIRED_KEYS if k not in payload]
    if missing:
        raise ValueError(f"screen payload lacks keys {missing}")
    if payload["schema"] != SCHEMA:
        raise ValueError(f"schema {payload['schema']!r} != {SCHEMA!r}")
    grid = [float(p) for p in payload["rule"]["p_grid"]]
    chosen = payload["chosen"]
    if not isinstance(chosen["layers"], list):
        raise ValueError("chosen.layers must be a list")
    n_blocks = int(payload["n_blocks"])
    band = tuple(payload["rule"]["depth_band"])
    for l in chosen["layers"]:
        if int(l) == 0:
            raise ValueError("chosen.layers contains layer 0")
        if not in_depth_band(int(l), n_blocks, band):
            raise ValueError(f"chosen layer {l} is outside the band {band}")
        p = chosen["nullspace_ratios"].get(str(l))
        if p is None or float(p) not in grid:
            raise ValueError(f"chosen layer {l} has ratio {p!r}, not in {grid}")
    if not payload["split"].get("content_disjoint_verified"):
        raise ValueError("split.content_disjoint_verified is not true")
    for key in ("n_fit_benign", "n_heldout_benign", "n_fit_harmful",
                "n_heldout_harmful"):
        if not isinstance(payload["split"].get(key), int):
            raise ValueError(f"split.{key} missing or not an int")
