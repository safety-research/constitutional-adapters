#!/usr/bin/env python
"""The strength frontier's COMMON PHYSICAL DOSE: pure-torch, CPU-testable
pieces shared by the capture job (dose_capture.py) and the report (dose_report.py).

WHAT IS MEASURED. Each method lane keeps its native dose knob (the LoRA lane a multiplier on the
write basis B, the AlphaSteer lane the scalar `base_scale`), and every arm ADDITIONALLY reports
`realised_dose_median`: the median over evaluation prompts of ||edit|| / ||h|| at the steered site,
where `edit` is the vector the served path adds and `h` the residual stream it is added to. Both
are computed ANALYTICALLY from the staged object's bytes on activations captured by an HF forward of
the base model over the SHARED reference pool under the SERVED rendering — never on a fit capture,
and never by editing anything under src/sisteer/capeval/serving/ (that tree is under a
drift measurement; this module mirrors its arithmetic and the tests hold the mirror to the real
adapters as an oracle).

THE TWO EDITS, exactly as lens_adapters serves them (read, not modified):
  * OP_LORA_OPROJ (lora_all / lora_subtraction): a forward hook on each stored layer's attention
    o_proj; x = the o_proj INPUT rows; delta = (x.float() @ A) @ B^T per op with A [d_in, k],
    B [d_out, k], both fp32, no alpha/r factor; coeff +1 (lora_all; the subtraction's positive
    parent) or -1 (the subtraction's negative parent); `(coeff * delta).to(model dtype)` is added to
    the o_proj OUTPUT, op by op; positions=all, prefill AND decode; hook layer == stored layer.
    The residual stream is not visible at that hook — the o_proj output is what the block adds into
    it — so ||h|| here is the residual stream ENTERING the block (resid_pre of the stored layer), the
    stream the perturbation lands in.
  * OP_PREFILL_LAST_MATMUL (alphasteer): a forward hook on the block whose OUTPUT is the fit layer's
    resid_pre (hook layer = fit layer - 1 under `site: pre`); h_last = the last prompt row of that
    block output; steer = h_last @ M in the MODEL dtype (M is cast to bf16 at materialize);
    edit = (base_scale * steer).to(model dtype), broadcast to every prompt row, prefill only.
    ||h|| is ||h_last||.

THE GATE. Both edits are exactly linear in the scaled tensor (B; base_scale), so on the
SAME activations the rung edit must be the reference edit times s/s_ref. A ratio of NORMS cannot see a
sign, so the gate is on the SIGNED factor c = <e_s, e_ref> / <e_ref, e_ref> per (prompt, layer), with
the orthogonal residual ||e_s - c e_ref|| / ||e_ref||; the norm ratio is reported beside it. Computed
on the fp32 analytic edits (the served-dtype cast breaks exact linearity for non-power-of-two
scales; that column is reported, not gated). A lane failing the gate has no dose axis.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch

SCHEMA = "strength_dose/1"

# --- the shared reference pool: ONE materialised, hashed file ------------------------------------
POOL_REL = "exp/dose_reference_pool/pool_v1.jsonl"
POOL_SHA256 = "d6ecd62bea7e0c075c311c8364785179760e6fba6dedabee8e4bf433f03b4099"
POOL_N_HARMFUL = 259
POOL_N_BENIGN = 120
POOL_KEYS = ("idx", "half", "source", "goal_id", "text")
#: substrings that can only appear in the WRONG value (a dataclass repr captured instead of its
#: text field) — the content guard, generalised past the one marker that bit
REPR_MARKERS = ("goal_id=", "Goal(", "text='", 'text="', "dataclass", "<sisteer.")

#: the served chat-template hashes (sha256(tok.chat_template.encode())[:16]) per target, pinned by
#: the campaign's manifests (make_full22_configs.MODEL_SERVING) and re-pinned by the pool's provenance
EXPECTED_TEMPLATE_HASH = {
    "gemma4e4b": "0a2c8073c878ab1d",
    "qwen35_4b": "a4aee8afcf2e0711",
    "nemotron3nano4b": "ab7813c3abdd9cb6",
    "glm46vflash": "bf6cfe131f9af1cd",
}
TARGETS = tuple(EXPECTED_TEMPLATE_HASH)

#: the LoRA lane's dose views per target: the `it_` spelling belongs to the instruct-direct lanes
#: (gemma4e4b, qwen35_4b) and is EXCLUDED by default (their parent objects are being re-fit);
#: `--include-it-views` brings them back once re-baked.
LORA_DOSE_VIEWS = ("dm1", "d0p1", "d0p25", "d0p5", "d1", "d1p5")
LORA_REFERENCE_VIEW = "d1"          # the unscaled parent; s_ref = 1.0
ALPHASTEER_REFERENCE_STRENGTH = 1.0  # the lambda=1 parent (as_d1), s_ref = 1.0

#: gate tolerances (fp32 analytic edits; see the module docstring). A real defect — wrong layer,
#: wrong sign, wrong tensor scaled — moves c by O(1); fp32 rounding of a pre-scaled B moves it by
#: ~1e-7. The negative rung and every power-of-two rung are EXACT and reported as such.
GATE_RTOL_MEDIAN = 1e-5
GATE_ATOL_MAX = 1e-4
GATE_RESID_MAX = 1e-4


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


class PoolError(RuntimeError):
    pass


def load_pool(path: Path, expected_sha256: str = POOL_SHA256, *,
              n_harmful: int = POOL_N_HARMFUL, n_benign: int = POOL_N_BENIGN) -> List[dict]:
    """Read the materialised reference pool, REFUSING anything but the exact artefact: the sha of the
    bytes, the row count and halves (a truncated pool), the key set and idx order, and the content
    guard (no text carries a marker that can only come from a repr). Returns the rows in file order."""
    raw = Path(path).read_bytes()
    got = sha256_bytes(raw)
    if got != expected_sha256:
        raise PoolError(f"{path}: sha256 {got} != expected {expected_sha256} — not the pinned pool")
    rows = []
    for i, line in enumerate(raw.decode("utf-8").splitlines()):
        if not line.strip():
            raise PoolError(f"{path}: blank line at {i}")
        rows.append(json.loads(line))
    n = n_harmful + n_benign
    if len(rows) != n:
        raise PoolError(f"{path}: {len(rows)} rows, expected {n} ({n_harmful} harmful + {n_benign} benign)")
    for r in rows:
        if tuple(sorted(r)) != tuple(sorted(POOL_KEYS)):
            raise PoolError(f"{path}: row {r.get('idx')} keys {sorted(r)} != {sorted(POOL_KEYS)}")
    if [r["idx"] for r in rows] != list(range(n)):
        raise PoolError(f"{path}: idx is not 0..{n - 1} in order")
    halves = [r["half"] for r in rows]
    if halves[:n_harmful] != ["harmful"] * n_harmful or halves[n_harmful:] != ["benign"] * n_benign:
        raise PoolError(f"{path}: halves are not {n_harmful} harmful then {n_benign} benign")
    for r in rows:
        t = r["text"]
        if not isinstance(t, str) or not t.strip():
            raise PoolError(f"{path}: row {r['idx']} has an empty text")
        for m in REPR_MARKERS:
            if m in t:
                raise PoolError(f"{path}: row {r['idx']} text carries {m!r} — a repr, not the prompt")
    return rows


def assert_rendered_clean(rendered: Sequence[str]) -> None:
    """The content guard again, on the RENDERED strings the model actually consumes."""
    for i, s in enumerate(rendered):
        for m in REPR_MARKERS:
            if m in s:
                raise PoolError(f"rendered prompt {i} carries {m!r} — a repr reached the render")


def half_of(idx: int, n_harmful: int = POOL_N_HARMFUL) -> str:
    return "harmful" if idx < n_harmful else "benign"


# --- render -----------------------------------------------------------------------------------
def template_hash(tok) -> str:
    return hashlib.sha256((getattr(tok, "chat_template", None) or "").encode()).hexdigest()[:16]


def assert_template_hash(target: str, tok) -> str:
    got = template_hash(tok)
    want = EXPECTED_TEMPLATE_HASH[target]
    if got != want:
        raise RuntimeError(f"{target}: resolved chat_template hash {got} != pinned {want} — a template "
                           "difference alone moved an unsteered refusal rate 0.98 -> 0.38 on identical "
                           "weights; refusing to measure under the wrong render")
    return got


def render_texts(tok, texts: Sequence[str]) -> List[str]:
    """EXACTLY the pool's pinned render: build_chat_prompt(tok, text, system=None) -> apply_chat_template
    (add_generation_prompt=True, the tokenizer's thinking-OFF marker where the template has the switch).
    The tokenizer must come from sisteer.models.load_tokenizer (which sets that marker)."""
    from sisteer.models import build_chat_prompt
    out = [build_chat_prompt(tok, t, system=None) for t in texts]
    assert_rendered_clean(out)
    return out


# --- objects -----------------------------------------------------------------------------------
def load_blob(path: Path, *, mmap: bool = False) -> dict:
    """torch.load with weights_only=True, as lens_adapters._torch_blob loads them (the blobs are plain
    dicts of tensors, numbers, strings and lists)."""
    kw = {"map_location": "cpu", "weights_only": True}
    if mmap:
        kw["mmap"] = True
    blob = torch.load(str(path), **kw)
    if not isinstance(blob, Mapping) or "kind" not in blob:
        raise RuntimeError(f"{path}: not a steering blob (no `kind`)")
    return blob


@dataclass(frozen=True)
class LoraOp:
    """One o_proj low-rank op. `A` [d_in, k] / `B` [d_out, k] are the blob's own orientation (what the
    producer scaled); `lowrank_a` [k, d_in] / `lowrank_b` [d_out, k] are the SERVED layout, built the
    way `_lora_pair_ops` builds them.

    BOTH are kept, and the arithmetic uses the served pair, because the two are not numerically
    interchangeable: the served path multiplies through TRANSPOSED VIEWS (`lowrank_a.t()`), and a
    GEMM over a contiguous copy reduces in a different order, so the fp32 sums land a few ulps apart
    and the bf16 cast then differs on a rounding boundary. An algebraically-equal mirror is not a
    mirror; this was caught by testing against `apply_oproj_ops` itself."""
    layer: int
    A: torch.Tensor
    B: torch.Tensor
    coeff: float
    parent: str
    lowrank_a: torch.Tensor = field(repr=False, default=None)
    lowrank_b: torch.Tensor = field(repr=False, default=None)

    def __post_init__(self):
        # Keeping two views of one tensor invites a caller to change the one the arithmetic does NOT
        # read (a `replace(op, A=...)` that silently does nothing). Refuse the inconsistency instead.
        if self.lowrank_a is None or self.lowrank_b is None:
            raise RuntimeError("LoraOp needs the served layout (lowrank_a / lowrank_b) — build it via "
                               "lora_ops_from_blob, which derives them from A and B")
        if not torch.equal(self.lowrank_a, self.A.t()) or not torch.equal(self.lowrank_b, self.B):
            raise RuntimeError("LoraOp: A/B and lowrank_a/lowrank_b disagree — the served layout is what "
                               "the arithmetic reads, so a change to A or B that does not reach it would "
                               "be silently ignored")


def lora_ops_from_blob(blob: Mapping[str, Any]) -> List[LoraOp]:
    """Mirror of lens_adapters._adapt_lora_all / _adapt_lora_subtraction's op list (same order: per
    layer, the positive parent then the negative), on the raw blob. Refuses the shapes the adapter
    refuses (rank disagreement; a `layers` list that disagrees with the tensors; positions other than
    `all`, which is what every wave object records)."""
    kind = blob["kind"]
    if kind == "lora_all":
        pairs = [("A", "B", 1.0, "single")]
    elif kind == "lora_subtraction":
        pairs = [("A_pos", "B_pos", 1.0, "pos"), ("A_neg", "B_neg", -1.0, "neg")]
    else:
        raise RuntimeError(f"not a LoRA blob: kind={kind!r}")
    if blob.get("positions", "all") != "all":
        raise RuntimeError(f"lora blob positions={blob.get('positions')!r}: this measurement assumes the "
                           "wave's positions=all serve (every prompt row is steered)")
    layers = sorted(int(k) for k in blob[pairs[0][0]])
    if "layers" in blob and sorted(int(x) for x in blob["layers"]) != layers:
        raise RuntimeError(f"lora blob `layers` {sorted(blob['layers'])} != tensors' layers {layers}")
    ops: List[LoraOp] = []
    for l in layers:
        for a_key, b_key, coeff, parent in pairs:
            A = blob[a_key][str(l)].detach().to(torch.float32)
            B = blob[b_key][str(l)].detach().to(torch.float32)
            if A.dim() == 1:
                A = A.reshape(-1, 1)
            if B.dim() == 1:
                B = B.reshape(-1, 1)
            if A.shape[1] != B.shape[1]:
                raise RuntimeError(f"layer {l} {a_key}/{b_key}: rank {A.shape[1]} != {B.shape[1]}")
            if "rank" in blob and int(blob["rank"]) != A.shape[1]:
                raise RuntimeError(f"layer {l}: rank {A.shape[1]} != blob rank {blob['rank']}")
            A = A.contiguous(); B = B.contiguous()
            ops.append(LoraOp(layer=l, A=A, B=B, coeff=float(coeff), parent=parent,
                              lowrank_a=A.t().contiguous(), lowrank_b=B))
    return ops


def lora_read(x: torch.Tensor, op: LoraOp) -> torch.Tensor:
    """The read half, in the served expression: `matmul(x.to(fp32), lowrank_a.t())` -> [T, k]."""
    a_t = op.lowrank_a.to(x.device).t()
    return torch.matmul(x.to(torch.float32), a_t)


def lora_edit_from_reads(reads: Sequence[torch.Tensor], ops: Sequence[LoraOp], model_dtype: torch.dtype,
                         base: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """The layer's edit from the per-op reads: fp32 analytic `sum coeff * (xa @ B^T)` (the quantity the
    gate is on), and the SERVED-dtype output — `apply_oproj_ops`' own composition, op by op:

        out = base;  for op: out = out + (coeff * delta_op).to(model dtype)

    Returned as the steered OUTPUT rows (model dtype). `base` is the o_proj output the edit is added to;
    with `base=None` a zero output is used, in which case the return IS the served edit. The distinction
    matters for the two-op subtraction cell: the served path adds the second cast delta to the ALREADY
    STEERED output, so its rounding depends on `base`, and a composition summed from zero is bit-exact
    only against a zero output. Callers that want what actually landed compute `out - base` in fp32."""
    if len(reads) != len(ops):
        raise RuntimeError(f"{len(reads)} reads for {len(ops)} ops")
    fp32 = None
    out = None if base is None else base.to(model_dtype)
    for xa, op in zip(reads, ops):
        b_t = op.lowrank_b.to(xa.device).t()
        delta = torch.matmul(xa.to(torch.float32), b_t)        # [T, d_out] fp32
        term = op.coeff * delta
        fp32 = term if fp32 is None else fp32 + term
        cast = term.to(model_dtype)
        out = cast if out is None else out + cast              # model-dtype add, exactly as the hook does
    return fp32, out


@dataclass(frozen=True)
class AlphaSteerObject:
    fit_layers: Tuple[int, ...]
    hook_layers: Tuple[int, ...]
    base_scale: float
    Ms: Dict[int, torch.Tensor]      # keyed by FIT layer, fp32 [d, d]
    d: int
    site: str


def alphasteer_from_blob(blob: Mapping[str, Any]) -> AlphaSteerObject:
    """Mirror of lens_adapters._adapt_alphasteer on the raw blob: the plural `Ms` wins, `site: pre`
    puts the hook one block BELOW the fit layer (the block whose OUTPUT is the fit layer's resid_pre),
    `steer_positions` must be `prefill`, `base_scale` is the served coeff."""
    if blob["kind"] != "alphasteer":
        raise RuntimeError(f"not an alphasteer blob: kind={blob['kind']!r}")
    if "base_scale" not in blob:
        raise RuntimeError("alphasteer blob lacks base_scale")
    base_scale = float(blob["base_scale"])
    if not math.isfinite(base_scale):
        raise RuntimeError(f"alphasteer base_scale {base_scale!r} is not finite")
    site = blob.get("site") or "post"
    if (blob.get("steer_positions") or "prefill") != "prefill":
        raise RuntimeError(f"alphasteer steer_positions={blob.get('steer_positions')!r} is not prefill")
    Ms = blob.get("Ms")
    if not (isinstance(Ms, Mapping) and Ms):
        raise RuntimeError("alphasteer blob has no multi-layer `Ms` (legacy single-layer blobs are not "
                           "what the strength lane staged)")
    fit_layers = tuple(sorted(int(k) for k in Ms))
    if blob.get("layers") and tuple(sorted(int(l) for l in blob["layers"])) != fit_layers:
        raise RuntimeError(f"alphasteer `layers` {sorted(blob['layers'])} != Ms keys {list(fit_layers)}")
    mats = {}
    d = None
    for l in fit_layers:
        M = Ms[l] if l in Ms else Ms[str(l)]
        if M.dim() != 2 or M.shape[0] != M.shape[1]:
            raise RuntimeError(f"Ms[{l}] is {tuple(M.shape)}, not square")
        if d is None:
            d = int(M.shape[0])
        if int(M.shape[0]) != d:
            raise RuntimeError(f"Ms[{l}] d={M.shape[0]} != {d}")
        mats[l] = M.detach().to(torch.float32)
    if site == "post":
        hook = fit_layers
    elif site == "pre":
        if 0 in fit_layers:
            raise RuntimeError("site=pre with fit layer 0 has no block below it")
        hook = tuple(l - 1 for l in fit_layers)
    else:
        raise RuntimeError(f"alphasteer site={site!r} is neither pre nor post")
    return AlphaSteerObject(fit_layers=fit_layers, hook_layers=hook, base_scale=base_scale, Ms=mats,
                            d=int(d), site=site)


def alphasteer_edit(h_last: torch.Tensor, M: torch.Tensor, base_scale: float, model_dtype: torch.dtype
                    ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per prompt ([N, d] rows of h_last in the MODEL dtype): the fp32 analytic edit
    `base_scale * (h_last.float() @ M.float())` (the gate's quantity, batched — it is explicitly the
    analytic value) and the SERVED edit.

    The served edit is computed ONE PROMPT AT A TIME, as a 1-D `h_last @ M` in the model dtype, because
    that is literally what `_apply_one` does: the served forward handles one request, so `h_last` is a
    VECTOR and the product is a GEMV. Batching N prompts into a GEMM reduces in a different order and
    lands a few bf16 ulps away — which is invisible in a median but decides the elementwise
    "does this edit survive the cast" question the effective-layer count is built on."""
    M = M.to(h_last.device)
    h = h_last if h_last.dim() == 2 else h_last.unsqueeze(0)
    fp32 = base_scale * torch.matmul(h.to(torch.float32), M.to(torch.float32))
    Md = M.to(model_dtype)
    rows = [(base_scale * torch.matmul(h[i].to(model_dtype), Md)).to(model_dtype) for i in range(h.shape[0])]
    served = torch.stack(rows, 0)
    return fp32, served


# --- the gate ----------------------------------------------------------------------------------
def signed_factor(e: torch.Tensor, e_ref: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per row: c = <e, e_ref>/<e_ref, e_ref> and the orthogonal residual ||e - c e_ref|| / ||e_ref||,
    in float64 so the gate's arithmetic adds no rounding of its own."""
    e64 = e.to(torch.float64)
    r64 = e_ref.to(torch.float64)
    denom = (r64 * r64).sum(-1)
    c = (e64 * r64).sum(-1) / denom
    resid = (e64 - c.unsqueeze(-1) * r64).norm(dim=-1) / denom.sqrt()
    return c, resid


def gate_verdict(c: torch.Tensor, resid: torch.Tensor, norm_ratio: torch.Tensor, s: float, s_ref: float,
                 *, rtol_median: float = GATE_RTOL_MEDIAN, atol_max: float = GATE_ATOL_MAX,
                 resid_max: float = GATE_RESID_MAX) -> dict:
    """The gate as implemented: the median signed factor equals s/s_ref within rtol; every row's factor
    within atol of it; every residual below resid_max; the norm-ratio median equals |s/s_ref| within
    rtol. `exact` records whether c == s/s_ref BIT-FOR-BIT on every row (true for the negative rung
    and every power-of-two rung, by fp arithmetic's sign symmetry / exact scaling)."""
    want = s / s_ref
    c64 = c.to(torch.float64).flatten()
    med = float(c64.median())
    dev = (c64 - want).abs()
    nr = norm_ratio.to(torch.float64).flatten()
    med_nr = float(nr.median())
    checks = {
        "median_factor": {"got": med, "want": want, "abs_dev": abs(med - want),
                          "ok": abs(med - want) <= rtol_median * max(abs(want), 1e-12)},
        "max_row_factor_dev": {"got": float(dev.max()), "ok": bool(float(dev.max()) <= atol_max * max(abs(want), 1e-12))},
        "max_residual": {"got": float(resid.to(torch.float64).max()), "ok": bool(float(resid.max()) <= resid_max)},
        "median_norm_ratio": {"got": med_nr, "want": abs(want), "abs_dev": abs(med_nr - abs(want)),
                              "ok": abs(med_nr - abs(want)) <= rtol_median * max(abs(want), 1e-12)},
    }
    return {"s": s, "s_ref": s_ref, "want_factor": want, "n_rows": int(c64.numel()),
            "exact": bool(torch.equal(c64, torch.full_like(c64, want))),
            "checks": checks, "pass": all(v["ok"] for v in checks.values())}


# --- aggregation -------------------------------------------------------------------------------
def quantiles(x: torch.Tensor) -> dict:
    x = x.detach().to(torch.float64).flatten()
    if x.numel() == 0:
        return {"n": 0}
    q = torch.quantile(x, torch.tensor([0.05, 0.25, 0.5, 0.75, 0.95], dtype=torch.float64))
    return {"n": int(x.numel()), "p05": float(q[0]), "p25": float(q[1]), "median": float(q[2]),
            "p75": float(q[3]), "p95": float(q[4]), "iqr": float(q[3] - q[1]),
            "mean": float(x.mean()), "min": float(x.min()), "max": float(x.max())}


def three_way(values: torch.Tensor, is_harmful: torch.Tensor) -> dict:
    """The three reference statistics the two lanes agreed on: the harmful half,
    the benign half and the pooled set, each with the full quantile spread."""
    is_harmful = is_harmful.to(torch.bool)
    return {"pooled": quantiles(values), "harmful": quantiles(values[is_harmful]),
            "benign": quantiles(values[~is_harmful]),
            "family_gap_rel": _gap(values[is_harmful], values[~is_harmful])}


def _gap(a: torch.Tensor, b: torch.Tensor) -> Optional[float]:
    if a.numel() == 0 or b.numel() == 0:
        return None
    ma, mb = float(a.to(torch.float64).median()), float(b.to(torch.float64).median())
    return abs(ma - mb) / max(abs(ma), abs(mb), 1e-12)


def per_prompt_over_layers(ratios_by_layer: Mapping[int, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """[N] per-prompt summaries across an arm's steered layers: the median (the headline) and the
    max (the worst layer), from per-layer [N] tensors."""
    stack = torch.stack([ratios_by_layer[l].to(torch.float64) for l in sorted(ratios_by_layer)], 0)  # [L, N]
    return {"median_over_layers": stack.median(dim=0).values, "max_over_layers": stack.max(dim=0).values}


# --- the served numeric format: what survives the cast -------------------------------------------
def ulp_spacing(x: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """The serving tree's OWN definition, imported rather than copied so the unit cannot drift from
    the one the campaign's gate records (`lens_adapters.ulp_spacing`; that module is read-only to this
    lane). Falls back to nothing — an ImportError is a finding, not something to paper over."""
    from sisteer.capeval.serving.lens_adapters import ulp_spacing as _u
    return _u(x, dtype)


def survival(base: torch.Tensor, edit: torch.Tensor, dtype: torch.dtype) -> Dict[str, torch.Tensor]:
    """Does this layer's edit survive the served numeric format?

    THE CRITERION, and why it is this one. The served path adds the edit to a tensor and keeps the
    result in the serving dtype (`steered = out + (eff * delta).to(out.dtype)`), so an edit whose
    magnitude is below the last representable increment of what it is added to contributes EXACTLY
    NOTHING: the rounded sum is bit-identical to the unsteered value. The test is therefore the
    served arithmetic itself, elementwise —

        changed = (base.to(dt) + edit.to(dt)) != base.to(dt)

    — not a hand-derived ULP inequality, so it is right at subnormals, at exponent boundaries, and
    for the ties-to-even rounding the hardware actually does. Returned per row: `frac_changed` (the
    fraction of the hidden dimension that moves) and `edit_ulps` (max and median |edit| in ULPs of
    `base` at that element's magnitude, the scale-free magnitude beside the binary answer).
    A reader who prefers another threshold recomputes it from the shipped per-layer vectors."""
    b = base.to(dtype)
    e = edit.to(dtype)
    changed = (b + e) != b
    ulps = e.detach().float().abs() / ulp_spacing(b, dtype)
    return {"frac_changed": changed.to(torch.float64).mean(dim=-1),
            "n_changed": changed.sum(dim=-1),
            "edit_ulps_max": ulps.max(dim=-1).values,
            "edit_ulps_median": ulps.median(dim=-1).values}


#: a layer counts as ACTIVE at a rung when the MEDIAN prompt's served sum changes at all
#: (frac_changed > 0); `_HALF` is the stricter companion reported beside it
ACTIVE_FRAC_THRESHOLD = 0.0
ACTIVE_FRAC_THRESHOLD_HALF = 0.5


def effective_layers(frac_by_layer: Mapping[int, torch.Tensor]) -> dict:
    """The effective active layer count of one rung: of the layers the object nominally steers, how
    many still change the served sum. Ships the per-layer median fraction with it, so the count can
    be recomputed at any other threshold from the record alone."""
    med = {int(l): float(v.to(torch.float64).median()) for l, v in frac_by_layer.items()}
    active = sorted(l for l, f in med.items() if f > ACTIVE_FRAC_THRESHOLD)
    half = sorted(l for l, f in med.items() if f >= ACTIVE_FRAC_THRESHOLD_HALF)
    return {"nominal_layer_count": len(med), "effective_layer_count": len(active),
            "effective_layer_count_half": len(half), "active_layers": active,
            "active_layers_half": half,
            "inactive_layers": sorted(l for l in med if l not in set(active)),
            "median_frac_changed_by_layer": med,
            "threshold": {"rule": "a layer is ACTIVE when the median prompt's fraction of hidden-dim "
                                  "coordinates whose served-dtype sum changes is > 0; _half requires >= 0.5",
                          "active": ACTIVE_FRAC_THRESHOLD, "half": ACTIVE_FRAC_THRESHOLD_HALF,
                          "recompute_from": "median_frac_changed_by_layer + per_layer_realised_dose"},
            # What a zero count does and does NOT establish. The per-layer statistic is the MEDIAN over
            # prompts at the captured position/phase, so a layer whose edit survives on a MINORITY of
            # prompts has median 0 and reads inactive: two layers each changing 2 of 5 prompts give an
            # effective count of 0 while 4 of those 5 prompts changed somewhere. Zero here therefore means
            # "on the median prompt, at this position and phase, no nominally-steered layer changes the
            # served sum" — NOT "this object is a no-op". A whole-model no-op claim needs evidence over
            # all prompts, positions and phases, which this capture does not collect.
            "scope": {"statistic": "per-layer MEDIAN over prompts, at the captured position/phase",
                      "zero_means": "no layer changes the served sum on the median prompt at this "
                                    "position/phase",
                      "zero_does_not_mean": "the object is a no-op; minority-prompt survival is invisible "
                                            "to a median, as are other positions and phases"}}


# --- where the o_proj edit actually lands (architecture-specific; verified, never assumed) ---------
#: The block classes whose ATTENTION OUTPUT passes through a norm BEFORE it is added to the residual
#: stream, and the attribute that norm hangs off. On these the o_proj edit does NOT land additively in
#: the residual — it is re-normalised with the attention output first — so ||edit||/||resid_pre|| at the
#: o_proj site is a DIFFERENT quantity from the fraction of the residual stream the edit adds.
#: Keyed by the block CLASS, never by attribute presence: NemotronHBlock and Qwen3_5DecoderLayer also
#: own a `post_attention_layernorm`, but theirs normalises the residual on the way INTO the MLP and
#: never touches the attention output. An unknown class refuses rather than guessing which it is; the
#: capture then VERIFIES the entry by hooking the named module and requiring its input to be the very
#: tensor the o_proj emitted.
POST_ATTENTION_NORM_BY_BLOCK = {
    "Gemma4TextDecoderLayer": "post_attention_layernorm",
    "Gemma4DecoderLayer": "post_attention_layernorm",
    "Glm4vTextDecoderLayer": "post_self_attn_layernorm",
    "NemotronHBlock": None,
    "Qwen3_5DecoderLayer": None,
    "Qwen3_5TextDecoderLayer": None,
    "LlamaDecoderLayer": None,
}


def nan_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    """`torch.equal` that treats NaN == NaN. Left-padded batches leave fully-masked attention rows whose
    outputs can be NaN; two references to the SAME tensor then compare unequal under torch.equal, which
    is how the GLM placement check first refused a correct table entry (2026-09-11)."""
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    return bool(((a == b) | (a.isnan() & b.isnan())).all())


def post_attention_norm(block) -> Tuple[Optional[Any], Optional[str]]:
    """(module, attribute) of the norm the attention output passes through before the residual add, or
    (None, None) when it is added directly. Refuses an unlisted block class."""
    name = type(block).__name__
    if name not in POST_ATTENTION_NORM_BY_BLOCK:
        raise RuntimeError(
            f"decoder block class {name!r} is not in POST_ATTENTION_NORM_BY_BLOCK — whether this "
            "architecture normalises the attention output before adding it to the residual decides "
            "what ||edit||/||h|| MEANS here; add the class after reading its forward, never guess")
    attr = POST_ATTENTION_NORM_BY_BLOCK[name]
    if attr is None:
        return None, None
    mod = getattr(block, attr, None)
    if mod is None:
        raise RuntimeError(f"{name}: expected a `{attr}` module on the block and found none")
    return mod, attr


def residual_edit_through_norm(attn_out: torch.Tensor, edit_fp32: torch.Tensor, steered_out: torch.Tensor,
                               resid_pre: torch.Tensor, norm_mod, dtype: torch.dtype
                               ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """What the o_proj edit actually delivers to the residual stream, two ways.

    ANALYTIC (fp32): `N(attn + edit) - N(attn)`, or the edit itself on a direct-add block — computed by
    CALLING THE MODEL'S OWN norm module (its weights, its epsilon, its internal upcast) on fp32 rows rather
    than reimplementing RMSNorm. Non-linear in the edit where a norm is present, which is why the
    linearity gate stays at the o_proj site.

    SERVED: the block's residual add replayed in the model dtype on the STEERED o_proj output the hook
    produced — `mid_steered = resid_pre + N(steered_out)` against `mid = resid_pre + N(attn_out)` — so the
    returned served perturbation and the two `mid` rows carry every rounding the served path applies (the
    hook's cast, the norm, the residual add). Returns (analytic fp32 perturbation, served perturbation as
    an fp32 difference of model-dtype rows, mid_steered, mid)."""
    a32 = attn_out.to(torch.float32)
    pre_d = resid_pre.to(dtype)
    with torch.no_grad():
        if norm_mod is None:
            fp32 = edit_fp32.to(torch.float32)
            mid = pre_d + attn_out.to(dtype)
            mid_st = pre_d + steered_out.to(dtype)
        else:
            fp32 = (norm_mod(a32 + edit_fp32.to(torch.float32)) - norm_mod(a32)).to(torch.float32)
            mid = pre_d + norm_mod(attn_out.to(dtype))
            mid_st = pre_d + norm_mod(steered_out.to(dtype))
    served = mid_st.to(torch.float32) - mid.to(torch.float32)
    return fp32, served, mid_st, mid


def survival_between(before: torch.Tensor, after: torch.Tensor, dtype: torch.dtype) -> Dict[str, torch.Tensor]:
    """The survival statistics between two already-composed model-dtype rows (the unsteered and the
    steered output of the same add): the fraction of coordinates that differ, and how large the landed
    difference is in ULPs of the unsteered row."""
    b = before.to(dtype)
    a = after.to(dtype)
    changed = a != b
    ulps = (a.float() - b.float()).abs() / ulp_spacing(b, dtype)
    return {"frac_changed": changed.to(torch.float64).mean(dim=-1), "n_changed": changed.sum(dim=-1),
            "edit_ulps_max": ulps.max(dim=-1).values, "edit_ulps_median": ulps.median(dim=-1).values}


def dose_token(dose: float) -> str:
    """0.25 -> d0p25, 0.5 -> d0p5, 1 -> d1, -1 -> dm1, -0.5 -> dm0p5 (the roster's spelling)."""
    s = ("m" if dose < 0 else "") + (f"{abs(dose):g}".replace(".", "p"))
    return "d" + s


def json_ready(o: Any) -> Any:
    if isinstance(o, torch.Tensor):
        return o.detach().cpu().tolist()
    if isinstance(o, Mapping):
        return {str(k): json_ready(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [json_ready(v) for v in o]
    if isinstance(o, float) and not math.isfinite(o):
        return str(o)
    return o
