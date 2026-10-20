"""Difference-in-means extraction (CAA-style) plus reliability diagnostics.

Shared engine for the activation-difference family:
- CAA (Rimsky et al. 2312.06681): contrast completions under the same prompt.
- Instruction-diff (Stolfo et al. 2410.12877): same prompt with vs without
  the instruction in context.

Both reduce to: pooled activations for a positive set and a negative set,
mean difference per layer. Alongside the vector we compute Braun et al.
(2505.22637) diagnostics that predict steerability:
- directional agreement: mean cosine of per-pair activation diffs with the
  aggregate mean-diff vector;
- d': separation of the two activation distributions projected onto the
  vector, |mu+ - mu-| / sqrt(0.5 (var+ + var-)).

This module is ALSO the home of the ``diffmean`` registry arm: the simplest
member of the family promoted to a standalone two-SI Class A arm — ONE fixed
vector r = mean(h | SI present) - mean(h | SI absent) at the LAST prompt
token (Stolfo instruction-difference; equivalently the alphakl arm's static
actuation direction r before its unit normalization), applied ADDITIVELY at
the pinned layer. No learned gate, no probe, no ridge, no null-space
projection — deliberately: it is the interpretable floor every learned arm
must beat. :class:`DiffMeanObject` is the deployment; :func:`train_diffmean`
the registry trainer.
"""

from __future__ import annotations

from pathlib import Path

import torch

from ..hooks import FIT_POSITION_MODES, PositionMaskedFit, steering
from ..vectors import SteeringVector
from .capture import ActivationBatch, collect_pooled_activations


def _diagnostics(pos: torch.Tensor, neg: torch.Tensor) -> dict:
    """pos/neg: [N, d] pooled activations (paired by index when N matches)."""
    mean_diff = pos.mean(0) - neg.mean(0)
    unit = mean_diff / (mean_diff.norm() + 1e-8)
    out: dict = {}
    if pos.shape[0] == neg.shape[0]:
        per_pair = pos - neg
        cos = torch.nn.functional.cosine_similarity(
            per_pair, mean_diff[None, :], dim=1
        )
        out["directional_agreement"] = float(cos.mean())
        out["agreement_frac_positive"] = float((cos > 0).float().mean())
    proj_pos = pos @ unit
    proj_neg = neg @ unit
    pooled_var = 0.5 * (proj_pos.var() + proj_neg.var())
    out["d_prime"] = float(
        (proj_pos.mean() - proj_neg.mean()).abs() / (pooled_var.sqrt() + 1e-8)
    )
    return out


def diff_of_means(
    pos: ActivationBatch,
    neg: ActivationBatch,
    *,
    method: str,
    model_name: str,
    instruction_id: str,
    metadata: dict | None = None,
) -> SteeringVector:
    vectors: dict[int, torch.Tensor] = {}
    diag: dict[str, dict] = {}
    for layer in pos.acts:
        p, n = pos.acts[layer], neg.acts[layer]
        vectors[layer] = p.mean(0) - n.mean(0)
        diag[str(layer)] = _diagnostics(p, n)
    return SteeringVector(
        vectors=vectors,
        method=method,
        model_name=model_name,
        instruction_id=instruction_id,
        metadata={**(metadata or {}), "diagnostics": diag},
    )


def extract_caa(
    model,
    tok,
    prompts: list[str],
    pos_completions: list[str],
    neg_completions: list[str],
    layers: list[int],
    position: str = "completion_mean",
    batch_size: int = 8,
    instruction_id: str = "",
) -> SteeringVector:
    """CAA generalized to free-form completions.

    Original CAA reads the residual stream at the token of the contrastive
    A/B answer letter (position="completion_last" with one-token
    completions reproduces it). For system-instruction distillation the
    contrast is completion-with-SI vs completion-without/anti-SI under the
    same prompt, pooled over completion tokens.
    """
    pos_texts, neg_texts, pos_lens, neg_lens = [], [], [], []
    for prompt, pos_c, neg_c in zip(prompts, pos_completions, neg_completions):
        n_prompt = len(tok(prompt, add_special_tokens=True)["input_ids"])
        pos_texts.append(prompt + pos_c)
        neg_texts.append(prompt + neg_c)
        pos_lens.append(n_prompt)
        neg_lens.append(n_prompt)
    pos = collect_pooled_activations(
        model, tok, pos_texts, layers, position, pos_lens, batch_size
    )
    neg = collect_pooled_activations(
        model, tok, neg_texts, layers, position, neg_lens, batch_size
    )
    return diff_of_means(
        pos,
        neg,
        method="caa",
        model_name=model.config._name_or_path,
        instruction_id=instruction_id,
        metadata={"position": position, "n_pairs": len(prompts)},
    )


def extract_instruction_diff(
    model,
    tok,
    prompts_with_instruction: list[str],
    prompts_without_instruction: list[str],
    layers: list[int],
    position: str = "last",
    batch_size: int = 8,
    instruction_id: str = "",
    prefix_cache: bool = False,
) -> SteeringVector:
    """Stolfo-style instruction vector: activation diff between the same
    user prompt rendered with vs without the instruction, pooled at the
    end of the prompt (no completions needed).

    ``prefix_cache=True`` (teacher-side prefix KV caching): both
    capture sweeps share their rendered prefix across rows — the
    WITH-instruction side's prefix is the chat header + the full SI
    (~95% of its tokens for the long SIs), the without side's the bare
    chat header — so each sweep forwards the shared prefix ONCE and
    only the per-prompt suffixes per row (capture.py's cached path).
    Semantically exact; fp-tolerance-only difference. Default ``False``
    keeps existing callers byte-identical."""
    pos = collect_pooled_activations(
        model, tok, prompts_with_instruction, layers, position, None,
        batch_size, prefix_cache=prefix_cache
    )
    neg = collect_pooled_activations(
        model, tok, prompts_without_instruction, layers, position, None,
        batch_size, prefix_cache=prefix_cache
    )
    sv = diff_of_means(
        pos,
        neg,
        method="instr_diff",
        model_name=model.config._name_or_path,
        instruction_id=instruction_id,
        metadata={"position": position, "n_pairs": len(prompts_with_instruction)},
    )
    # Stolfo Eq. 2 needs the mean projection of WITH-instruction inputs onto
    # the unit direction; store it so eval can use projection-targeting.
    targets = {}
    for layer, v in sv.vectors.items():
        u = v / (v.norm() + 1e-8)
        targets[str(layer)] = float((pos.acts[layer] @ u).mean())
    sv.metadata["projection_targets"] = targets
    return sv


# --------------------------------------------------------------------------- #
# the diffmean registry arm                                                    #
# --------------------------------------------------------------------------- #


class DiffMeanObject(PositionMaskedFit):
    """The diff-in-means baseline: h <- h + scale * r, a FIXED per-layer
    vector applied through ``hooks.steering`` — the plainest residual-stream
    addition in the registry, and deliberately so (the interpretable floor).

    Deployment follows the harness's plain-vector convention (VectorObject):
    every position, prompt and decode alike, with the BOS excluded
    (``skip_bos=True`` — an UNGATED vector on the BOS outlier activation is
    a known blowup; the gated arms cover BOS because their fitted gate
    down-weights it, which this arm by design does not have).

    Position-masked (decode-only) leg: a plain addition is the EXACT case
    for the position mask — ``SteeringHandle._position_mask`` (the single
    place a mask is computed) gates the addition, so masked-out positions
    are untouched and ``apply()`` yields the live handle (per-row training
    boundaries via ``handle.set_prompt_lens``, the same contract the maskable
    hook arms honor). The FIT itself is closed-form on unsteered activations,
    so the response-mode vector is BITWISE the all-token vector — what the
    mode changes is the DEPLOYMENT, which is therefore PERSISTED in the
    artifact (the LoReFT precedent: re-hydrating a response-fit object as
    all-token would silently run the wrong leg), and the metadata records the
    fit's position-independence so nobody reads the two legs' artifacts as
    different fits."""

    kind = "diffmean"

    def __init__(self, model, vectors: dict[int, torch.Tensor],
                 skip_bos: bool = True, metadata: dict | None = None):
        del model  # signature parity with the sibling objects; nothing kept
        self.vectors = {int(l): v.detach().float() for l, v in vectors.items()}
        self.skip_bos = bool(skip_bos)
        self.metadata = metadata or {}

    @property
    def layers(self) -> list[int]:
        return sorted(self.vectors)

    @property
    def trainable(self) -> list:
        return []  # closed-form; nothing trains by gradient

    def n_params(self) -> int:
        return sum(v.numel() for v in self.vectors.values())

    def inject(self, rendered_prompt: str) -> str:
        return rendered_prompt

    def apply(self, model, scale: float = 1.0):
        """A steering context over the model; yields the live
        :class:`~sisteer.hooks.SteeringHandle` (the mask source). Scale 0
        registers NO vector hook, so the forward is bit-identical to the
        unsteered model (the registry's scale-0 invariant)."""
        return steering(
            model, {} if scale == 0.0 else self.vectors, scale=scale,
            positions=("response" if self.fit_positions == "response"
                       else "all"),
            skip_bos=self.skip_bos,
            response_header_len=self.fit_response_header_len,
        )

    def to_(self, device) -> None:
        self.vectors = {l: v.to(device) for l, v in self.vectors.items()}

    def save(self, path: Path) -> None:
        torch.save(
            {"kind": self.kind,
             "vectors": {l: v.detach().float().cpu()
                         for l, v in self.vectors.items()},
             "skip_bos": self.skip_bos,
             # The deployment leg is part of the artifact's identity (see
             # the class docstring) — persisted exactly like LoReFT's.
             "positions": self.fit_positions,
             "response_header_len": self.fit_response_header_len,
             "metadata": self.metadata},
            path,
        )

    @classmethod
    def load_blob(cls, blob: dict, model, tok) -> "DiffMeanObject":
        del tok
        obj = cls(model,
                  {int(l): v for l, v in blob["vectors"].items()},
                  skip_bos=bool(blob.get("skip_bos", True)),
                  metadata=blob.get("metadata", {}))
        if blob.get("positions", "all") != "all":
            obj.set_fit_positions(blob["positions"],
                                  blob.get("response_header_len"))
        return obj


def train_diffmean(
    model,
    tok,
    instruction,
    pairs,
    *,
    layer: int,
    layers: list[int] | None = None,
    log_fn=None,
    batch_size: int = 8,
    positions: str = "all",
    response_header_len: int | None = None,
    teacher_prefix_cache: bool = False,
    epochs: int | None = None,   # train_kwargs.default compat (documented no-op)
    max_len: int | None = None,  # train_kwargs.default compat (documented no-op)
    seed: int | None = None,     # deterministic closed form (documented no-op)
) -> DiffMeanObject:
    """Registry trainer for the ``diffmean`` arm: r = mean(h | SI present) -
    mean(h | SI absent) at the LAST prompt token of the pinned layer(s).

    REUSES, never re-derives, the two existing pieces of this quantity:

    - the renderings are ``train_alphakl``'s exact two sides — the same
      ``build_chat_prompt(tok, prompt, system=instruction.text)`` vs
      ``system=None`` pair its r is built from (alphakl.py, the
      benign/malicious fit renderings);
    - the capture + mean-difference is :func:`extract_instruction_diff`
      (this module's tested Stolfo primitive: ``collect_pooled_activations``
      at ``position="last"`` — the identical call train_alphakl makes — then
      ``diff_of_means``), which also yields the Braun reliability
      diagnostics for free.

    Two DELIBERATE differences from alphakl's r, both part of this arm being
    the floor: r stays RAW (unnormalized — with no gate and no calibration,
    ``scale=1.0`` adds exactly the average SI-presence displacement, the
    interpretable unit; alphakl unit-normalizes only because its fitted
    probe carries the magnitude), and ALL pairs feed the mean (there is
    nothing to calibrate, so no calibration split is carved).

    ``positions="response"`` persists the decode-only DEPLOYMENT into the
    object (see :class:`DiffMeanObject` — the fitted vector is bitwise the
    all-token leg's; the metadata says so). ``epochs``/``max_len``/``seed``
    are documented no-ops (closed-form, deterministic).

    ``teacher_prefix_cache``: the SI-present capture sweep shares its
    rendered prefix (chat header + full SI) across every row, so under the
    flag it runs the prefix through the model once instead of once per pair
    (:mod:`.prefix_cache` via :func:`extract_instruction_diff`).
    Semantically exact — the fitted vector differs only at fp tolerance
    (kernel shapes change), the same class the row-sharding contract
    already accepts. Default ``False`` at THIS level so every caller
    outside the two-SI driver (and every bitwise pin, e.g. the r-reuse
    test against alphakl's uncached recipe) stays byte-identical; the
    two-SI wave turns it on through ``teacher_prefix_cache: true`` in
    configs/two_si.yaml (train_two_si_objects passes it through — its
    ``false`` is the escape hatch if a guard verdict sits near a
    boundary)."""
    del epochs, max_len, seed
    if positions not in FIT_POSITION_MODES:
        raise ValueError(
            f"fit positions must be one of {FIT_POSITION_MODES}, got "
            f"{positions!r}")
    prompts = [p.prompt for p in pairs]
    if not prompts:
        raise ValueError("diffmean extraction got an empty pairs list")
    steer_layers = [int(l) for l in (layers or [layer])]

    # Lazy import: models -> (nothing here); avoids widening this module's
    # import surface for the extract_* callers that never train the arm.
    from ..models import build_chat_prompt

    with_si = [build_chat_prompt(tok, p, system=instruction.text)
               for p in prompts]                       # SI present
    without = [build_chat_prompt(tok, p, system=None)
               for p in prompts]                       # SI absent
    sv = extract_instruction_diff(
        model, tok, with_si, without, layers=steer_layers, position="last",
        batch_size=batch_size, instruction_id=getattr(instruction, "id", ""),
        prefix_cache=teacher_prefix_cache)

    r_norms = {l: float(v.norm()) for l, v in sv.vectors.items()}
    metadata = {
        "method": "instr_diff",
        "n_pairs": len(prompts),
        "r_norms": r_norms,
        "diagnostics": sv.metadata.get("diagnostics", {}),
        "projection_targets": sv.metadata.get("projection_targets", {}),
        # The closed-form fit reads only unsteered last-prompt-token
        # activations, so the vector is IDENTICAL across the mask legs; the
        # positions mode below changes the deployment, not the fit.
        "fit_position_independent": True,
        # Fit-time provenance: whether the capture sweeps ran with the
        # teacher-side prefix KV cache (fp-tolerance-exact vs the plain
        # forward — recorded so a verdict near a boundary can be traced).
        "teacher_prefix_cache": bool(teacher_prefix_cache),
        "fit_positions": positions,
        "fit_response_header_len": (response_header_len
                                    if positions != "all" else None),
    }
    obj = DiffMeanObject(model, dict(sv.vectors), metadata=metadata)
    if positions != "all":
        obj.set_fit_positions(positions, response_header_len)
    obj.to_(next(model.parameters()).device)
    if log_fn is not None:
        scalars = {f"diffmean/r_norm_l{l}": n for l, n in r_norms.items()}
        for l, diag in metadata["diagnostics"].items():
            for k, v in diag.items():
                if isinstance(v, (int, float)):
                    scalars[f"diffmean/{k}_l{l}"] = v
        log_fn(scalars, step=0)
    return obj
