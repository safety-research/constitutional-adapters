"""caa_static_r — the CAA / diff-in-means floor as a Class B (SI-agnostic)
arm, DEPLOYMENT-MATCHED to AlphaSteer (the paper's own convention).

PROVENANCE — VENDORED, not rebuilt: this module's FIT INTERNALS are the
matched implementation's code, vendored VERBATIM from
an earlier development branch @ ``9c63c0d``:

- ``train_static_r_matched`` (``baseline_arms.py``): the layer validation,
  the SI-free renders, the resid_pre harvest, the per-layer RAW
  diff-in-means, and the fitted-sign assertion — reproduced verbatim in
  :func:`train_caa_static_r_content` below.
- ``StaticRObject``'s matched deployment path (``baseline_arms.py``,
  ``steer_positions="prefill_all"`` / ``site="pre"``): the ``_edited``
  fp32 broadcast edit and the ``make_pre`` hook — reproduced verbatim in
  :class:`CaaStaticRObject`. (Their class's OTHER mode — the preserved
  single-layer prompt-last A1 arm — is deliberately NOT vendored: this
  arm never deploys it, and carrying dead modes is how implementations
  drift.)
- ``_collect_last_acts`` (their ``alphasteer.py``): the shared harvest
  helper, vendored below because THIS branch's alphasteer module does not
  carry it. It bottoms out in ``capture.capture_layer_outputs``, which is
  byte-identical on the two branches.

Vendoring (not importing) is forced: their module exists only on their
branch, and adopting the whole ~780-line ``baseline_arms.py`` would drag
in two unrelated arms plus a FishBack dependency this branch lacks. What
is KEPT from this lane — deliberately, so nothing downstream re-indexes —
is the arm id / artifact ``kind`` (``caa_static_r``), the registry entry
and config membership, the object's constructor/save format, the Class B
sample-floor records, and the SI-agnosticism contract (``system=None`` on
every render — also theirs; the two agree).

WHY THEIR FIT AND NOT THE RE-IMPLEMENTATION IT REPLACES (the substantive
differences, recorded so the swap is auditable):

1. CAPTURE SITE == APPLY SITE, BY SHARED CODE. Their fit harvests at
   resid_pre of each steered layer via ``_collect_last_acts(...,"pre",...)``
   — the IDENTICAL helper their ``train_alphasteer_content_multilayer``
   uses — and the object applies at the same site. The replaced
   implementation captured the mathematically-identical tensor (decoder
   OUTPUT l−1 == resid_pre of l in this pre-norm architecture) but through
   DIFFERENT code and an l−1 index map; matching by shared code instead of
   by argument is what makes a silent one-block shift structurally
   impossible.
2. FITTED-SIGN ASSERTION, per layer: ``r_l · (mu_harmful − mu_benign)``
   must be strictly positive at fit time, else the fit raises —
   distinguishing an INVERTED direction from a degenerate one (sep == 0).
   The replaced implementation only checked degeneracy (norm ≈ 0) and
   would have shipped a backwards direction silently.
3. The metadata norms are RESID_PRE numbers. Their docstring's note
   travels with the code (see :func:`train_caa_static_r_content`): the
   harness's previously quoted layer-16 resid_post norm of 4.7566 does
   not carry over.

THE MATCHED SHAPE (the load-bearing decision, from the lane that owns the
comparison): the withdrawn original form injects ``out[:, -1, :] += lam·r``
with r UNIT-normalised at ONE layer — against a residual stream of norm
~50-150 that is a ~1% perturbation, and it measured mean attack DSR 0.458
at lam=1 against an UNSTEERED 0.468, i.e. below baseline. As "a floor
AlphaSteer must beat" that is not a floor, it is a switch that is off. The
arm only earns its place if it differs from AlphaSteer in EXACTLY ONE
thing: FIXED per-layer direction vs learned input-dependent d×d map. So
this object deploys AlphaSteer's own shape:

1. TEN layers — the paper's Llama-3.1-8B set ``[8,9,10,11,12,13,14,16,18,
   19]`` (:data:`PAPER_LLAMA31_LAYERS`), not one layer.
2. BROADCAST to EVERY PROMPT POSITION: AlphaSteer adds ``h_last @ M_l`` to
   all prompt positions; the fixed-direction analogue adds ``lam·r_l`` to
   all prompt positions.
3. PREFILL ONLY — gated on ``hidden_states.shape[1] > 1``; decode steps are
   untouched and see the edit only through the steered prompt KV cache
   (AlphaSteer's own convention).
4. RESID_PRE — a forward PRE-hook on each steered decoder layer, editing
   its INPUT before input_layernorm, where AlphaSteer applies. The fit
   captures the SAME site through the SAME vendored helper (point 1 of the
   differences above).
5. RAW, UN-NORMALISED per-layer diff-in-means. AlphaSteer's shipped refusal
   vectors are un-normalised, with per-layer norms (~1.6-5.5) that GROW
   with depth, and its edit magnitude inherits that — using the raw
   difference puts both arms' per-layer edits on the same scale, so the
   same lam means the same thing for both. Unit-normalising (the withdrawn
   spec, and the literal CAA-paper form) silently reintroduces the
   magnitude mismatch this arm exists to remove. Per-layer raw norms are
   recorded in metadata (``per_layer_r_norm`` — resid_pre numbers, see
   above).
6. Direction = harmful − benign on LAST-PROMPT-TOKEN activations, per
   layer, SI-AGNOSTIC (``system=None`` renders). Positive lam pushes toward
   refusal — asserted at fit time, per layer (difference 2 above).
7. NO calibration ladder — the dial IS lam (``base_scale·scale``). The
   DEPLOYED operating point is lam = :data:`DEPLOYED_LAM` = 0.05: the
   trainer bakes it into ``base_scale``, so serving at the pre-registered
   dose (eval alpha = config ``scale`` = 1.0) deploys lam 0.05, and the
   eval ``--alpha`` seam re-serves the same bytes at multiples of it. NOTE
   0.05 is deliberately BELOW the suggested grid
   (:data:`RECOMMENDED_LAM_GRID`, 0.1-0.5 plus 1.0/2.0) — a pinned
   operating point, not a swept default; do not "correct" it back into the
   grid.

This deliberately DIVERGES from the literal CAA paper form (single layer,
last token, unit norm) — a documented deviation for comparability with the
ten-layer all-prompt-position method it benchmarks, not an accident. The
literal form is a legitimate arm only if labelled as such; it must not be
presented as the floor for AlphaSteer. NOTE this repository's own
``alphasteer`` Class B arm is itself a documented single-layer-15
resid_post divergence from the paper (extraction/alphasteer.py, divergences
1/3); caa_static_r matches the PAPER deployment instead, so the paper-shape
comparison runs on the matched harness while this arm's matrix carries the
same fitted object.

CAPABILITY CAVEAT (read before quoting any DSR from this arm): an
UNDER-SCALED arm and a DESTROYED arm both produce clean-looking defense
numbers — ``alphakl_null`` at lam=1 scored DSR 1.000 with XSTest 0.000 and
GSM8K 0.000, a lobotomised model reading as a perfect defense. A ten-layer
fixed-direction broadcast edit is exactly the kind of intervention that can
buy "defense" by destroying the model, so any DSR from this arm MUST be
reported beside a capability/over-refusal control number, never alone.

Class B contract (configs/two_si.yaml ``class_b`` block): fitted ONCE, not
per-SI, on the AlphaSteer authors' fixed 14,000-benign / 2,000-harmful
lists (sha256-pinned; scripts/fit_class_b.py enforces the pins and resolves
the cluster paths). The ten-layer set is this arm's ONE deliberate
exemption from the Class B single-layer-15 pin — the matched deployment IS
the arm's identity (the lofit precedent: the pinned layer is passed and
deliberately ignored where the method's identity demands it).
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import torch

from ..models import build_chat_prompt, get_decoder_layers, num_layers
from .alphasteer import refusal_direction
from .capture import capture_layer_outputs
from .fit_checks import check_sample_floor

# The peer implementation this arm vendors — kept in metadata so the two
# lanes' artifacts are recognisably the SAME fit.
PEER_SPEC = ("caa_static_r == an earlier development branch's static_r_matched: fit "
             "internals (train_static_r_matched + StaticRObject prefill_all/"
             "site=pre path, baseline_arms.py) and the resid_pre harvest "
             "(alphasteer._collect_last_acts) vendored verbatim from "
             "commit 9c63c0d")

# The AlphaSteer paper's steered-layer set for Llama-3.1-8B (their Table /
# config; also quoted in extraction/alphasteer.py documented divergence 1).
PAPER_LLAMA31_LAYERS: tuple[int, ...] = (8, 9, 10, 11, 12, 13, 14, 16, 18, 19)

# The deployment convention THIS arm uses — AlphaSteer's, not the withdrawn
# last-token form. Recorded in every fitted object's metadata.
DEPLOYMENT = "prefill_only_resid_pre_broadcast_all_prompt_positions_10layer"

# AlphaSteer's published strength grid (0.1-0.5) plus higher points, for the
# eval-side lam sweep (the run's alpha seam). Informational — nothing here
# calibrates; the dial is lam.
RECOMMENDED_LAM_GRID: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 2.0)

# The DEPLOYED operating point: lam = 0.05, baked into the fitted object's
# base_scale so the pre-registered serving dose (eval alpha == config scale
# == 1.0) deploys exactly this lam. This is a PINNED operating point, not a
# swept default — it sits deliberately BELOW the suggested grid
# (RECOMMENDED_LAM_GRID starts at 0.1); do not "correct" it into the grid or
# silently revert it to 1.0. The eval --alpha seam still sweeps: served lam
# = DEPLOYED_LAM * alpha.
DEPLOYED_LAM: float = 0.05


def _collect_last_acts(
    model, tok, texts: list[str], layers: list[int], site: str,
    batch_size: int,
) -> dict[int, torch.Tensor]:
    """Last-VALID-token activations {layer: [N, d]} at ``site``, captured
    for ALL layers in ONE forward pass per batch (never re-forwarding per
    layer). Right-pads locally and indexes by the attention mask, so the
    pooled row is the true last prompt token. The local sibling of
    capture.collect_pooled_activations(position="last"), which is pinned
    to site="post"; capture site here matches the deploy site.

    VENDORED VERBATIM from ``alphasteer._collect_last_acts`` @
    9c63c0d (an earlier development branch) — the IDENTICAL helper its
    ``train_alphasteer_content_multilayer`` harvests through, so capture
    and apply sites are matched by shared code, not by argument. This
    branch's alphasteer module does not carry the helper, hence the
    vendored copy; ``capture_layer_outputs`` underneath is byte-identical
    on both branches."""
    device = next(model.parameters()).device
    out: dict[int, list[torch.Tensor]] = {l: [] for l in layers}
    prev_side = tok.padding_side
    tok.padding_side = "right"
    try:
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                enc = tok(texts[i:i + batch_size], return_tensors="pt",
                          padding=True).to(device)
                mask = enc["attention_mask"]
                rows = torch.arange(mask.shape[0], device=mask.device)
                last = mask.sum(dim=1) - 1          # right-padded here
                with capture_layer_outputs(model, list(layers),
                                           site=site) as cap:
                    model(**enc)
                for l in layers:
                    out[l].append(cap[l][rows, last].float().cpu())
    finally:
        tok.padding_side = prev_side
    return {l: torch.cat(v, dim=0) for l, v in out.items()}


class CaaStaticRObject:
    """The CAA floor behind the stage-2 object protocol, DEPLOYMENT-MATCHED
    to AlphaSteer's paper convention (see the module docstring): ``apply()``
    registers a forward PRE-hook on each steered decoder layer (resid_pre —
    the block input, before input_layernorm) that, DURING PREFILL ONLY,
    adds ``base_scale·scale·r_l`` to EVERY prompt position; decode steps
    (``seq_len == 1``) are untouched and see the edit only through the
    steered prompt KV cache. Scale 0 is an exact, bit-identical no-op.
    The ONLY difference from AlphaSteer's deployment is the edit itself:
    a FIXED per-layer direction ``r_l`` where AlphaSteer computes the
    input-dependent ``h_last @ M_l`` — that substitution is the arm's
    entire identity.

    The hook bodies (``_edited`` / the pre-hook in ``apply``) are VENDORED
    VERBATIM from ``baseline_arms.StaticRObject`` @
    9c63c0d (an earlier development branch), matched mode
    (``steer_positions="prefill_all"``, ``site="pre"``) — including the
    fp32 accumulate-then-cast arithmetic. The class NAME, ``kind``,
    constructor and artifact format stay this lane's, so nothing
    downstream re-indexes; their class's other mode (the preserved
    single-layer prompt-last A1 arm) is deliberately not carried."""

    kind = "caa_static_r"

    def __init__(self, model, rs: dict[int, torch.Tensor],
                 base_scale: float = 1.0, metadata: dict | None = None):
        self.rs = {int(l): v.detach().float() for l, v in rs.items()}
        self.base_scale = float(base_scale)
        self.metadata = metadata or {}
        self._decoder = get_decoder_layers(model)

    @property
    def layers(self) -> list[int]:
        return sorted(self.rs)

    @property
    def trainable(self) -> list:
        return []  # closed-form; nothing trains by gradient

    def n_params(self) -> int:
        return int(sum(v.numel() for v in self.rs.values()))

    def inject(self, rendered_prompt: str) -> str:
        return rendered_prompt

    def _edited(self, hidden: torch.Tensor,
                r_master: torch.Tensor, mag: float) -> torch.Tensor | None:
        """The edit on ``hidden`` [B, S, d], or None when this forward
        must not be steered (decode steps — seq_len == 1 forwards are
        NEVER edited; the edit persists via the KV cache alone).
        Vendored: ``StaticRObject._edited`` @ 9c63c0d, prefill_all branch."""
        if hidden.shape[1] <= 1:
            return None
        r = r_master.to(device=hidden.device)
        # every prompt position (AlphaSteer's broadcast; like it, pad
        # positions are edited too — causality + the attention mask
        # keep them from reaching real tokens)
        return (hidden.float() + mag * r).to(hidden.dtype)

    @contextmanager
    def apply(self, model, scale: float = 1.0):
        mag = self.base_scale * scale

        def make_pre(r_master):
            # resid_pre: edit hidden_states ENTERING the block, handling
            # both calling conventions. Vendored: ``StaticRObject.apply``'s
            # ``make_pre`` @ 9c63c0d.
            def hook(_mod, args, kwargs):
                if mag == 0.0:
                    return None  # exact no-op, bit-identical to unsteered
                if "hidden_states" in kwargs:
                    hidden = kwargs["hidden_states"]
                elif args and torch.is_tensor(args[0]):
                    hidden = args[0]
                else:
                    return None
                edited = self._edited(hidden, r_master, mag)
                if edited is None:
                    return None
                if "hidden_states" in kwargs:
                    return (args, {**kwargs, "hidden_states": edited})
                return ((edited,) + tuple(args[1:]), kwargs)

            return hook

        handles = [
            self._decoder[layer].register_forward_pre_hook(
                make_pre(r), with_kwargs=True)
            for layer, r in sorted(self.rs.items())
        ]
        try:
            yield
        finally:
            for h in handles:
                h.remove()

    def to_(self, device) -> None:
        self.rs = {l: v.to(device) for l, v in self.rs.items()}

    def save(self, path: Path) -> None:
        torch.save(
            {"kind": self.kind,
             "rs": {l: v.detach().float().cpu() for l, v in self.rs.items()},
             "base_scale": self.base_scale, "metadata": self.metadata},
            path,
        )

    @classmethod
    def load_blob(cls, blob: dict, model, tok) -> "CaaStaticRObject":
        del tok
        return cls(model, {int(l): v for l, v in blob["rs"].items()},
                   base_scale=float(blob["base_scale"]),
                   metadata=blob.get("metadata", {}))


def train_caa_static_r_content(
    model, tok, benign_prompts: list[str], harmful_prompts: list[str], *,
    layers: tuple[int, ...] | list[int] = PAPER_LLAMA31_LAYERS,
    base_scale: float = DEPLOYED_LAM, batch_size: int = 8, log_fn=None,
) -> CaaStaticRObject:
    """Fit the deployment-MATCHED CAA floor: one RAW diff-in-means direction
    r_l = mean(harmful) − mean(benign) per steered layer, from
    last-prompt-token activations rendered SI-free, deployed under
    AlphaSteer's convention (lam · r_l added to EVERY prompt position of
    EVERY steered layer during prefill only, persisting into generation via
    the KV cache; resid_pre, the paper's edit site).

    VENDORED VERBATIM from ``baseline_arms.train_static_r_matched`` @
    9c63c0d (an earlier development branch) — validation, renders, harvest,
    per-layer direction and sign assertion. The seams around it: the
    function/arm name, the ``layers`` default (the paper set), the
    Class B sample-floor records, the W&B scalars, and ``base_scale``
    defaulting to :data:`DEPLOYED_LAM` = 0.05 — the pinned deployed
    operating point, deliberately below the suggested grid; the vendored
    trainer defaults to 1.0 and sweeps lam entirely at eval.

    CAPTURE SITE == APPLY SITE: activations are harvested at resid_pre of
    each steered layer via ``_collect_last_acts`` — the identical harvest
    path the peer's ``train_alphasteer_content_multilayer`` uses — never
    fitted in one basis and injected in another. Consequence: the per-layer
    raw norms recorded in metadata are RESID_PRE numbers (resid_pre of
    layer l == resid_post of layer l-1); the harness's previously quoted
    layer-16 resid_post norm of 4.7566 does not carry over.

    WHY RAW (un-normalised), deliberately unlike the literal CAA form:
    AlphaSteer's own shipped refusal vectors are un-normalised, with
    per-layer norms 1.58-5.48 that grow with depth, and its edit magnitude
    inherits that scale — and those vectors are themselves resid_pre
    quantities, the same basis this fit uses. Using the raw difference puts
    this arm's per-layer edit on the same scale as AlphaSteer's, so the
    SAME lam means the same thing for both arms. Unit-normalising would
    silently re-introduce the magnitude mismatch this arm exists to remove.

    SIGN: r_l is harmful − benign, so POSITIVE lam pushes toward refusal.
    Asserted per layer at fit time — r_l · (mu_harmful − mu_benign) must be
    strictly positive — so an inverted or degenerate direction fails
    loudly, never a silently backwards arm."""
    if not benign_prompts or not harmful_prompts:
        raise ValueError("caa_static_r fit needs non-empty benign and "
                         "harmful prompt lists")
    steer_layers = [int(l) for l in layers]
    if not steer_layers:
        raise ValueError("layers must be a non-empty list")
    if len(set(steer_layers)) != len(steer_layers):
        raise ValueError(f"duplicate layers in {steer_layers}")
    n = num_layers(model)
    bad = [l for l in steer_layers if not 0 <= l < n]
    if bad:
        raise ValueError(
            f"layers {bad} out of range for a {n}-layer model")
    benign_texts = [build_chat_prompt(tok, p, system=None)
                    for p in benign_prompts]
    harmful_texts = [build_chat_prompt(tok, p, system=None)
                     for p in harmful_prompts]
    # capture at the DEPLOY site (resid_pre), through the same helper the
    # peer's alphasteer_paper fit uses — capture/apply-site matched by
    # shared code
    benign = _collect_last_acts(model, tok, benign_texts, steer_layers,
                                "pre", batch_size)
    harmful = _collect_last_acts(model, tok, harmful_texts, steer_layers,
                                 "pre", batch_size)
    rs: dict[int, torch.Tensor] = {}
    per_layer_norm: dict[int, float] = {}
    per_layer_sep: dict[int, float] = {}
    for l in steer_layers:
        r = refusal_direction(harmful[l], benign[l])   # RAW — never normalise
        # fitted-sign assertion: sep = r . (mu_harmful - mu_benign). <= 0
        # means the direction came out INVERTED (a sign-convention
        # regression in refusal_direction, or swapped harmful/benign
        # arguments) or degenerate (the means coincide) — fail loudly.
        sep = float(r @ (harmful[l].float().mean(dim=0)
                         - benign[l].float().mean(dim=0)))
        if sep <= 0.0:
            what = "degenerate (harmful and benign means coincide)" \
                if sep == 0.0 else "INVERTED"
            raise ValueError(
                f"layer {l}: fitted refusal direction is {what} — "
                f"r . (mu_harmful - mu_benign) = {sep}; positive lam must "
                "push toward refusal (r = harmful - benign), refusing to "
                "ship a backwards direction")
        rs[l] = r
        per_layer_norm[l] = float(r.norm())
        per_layer_sep[l] = sep
    # This lane's Class B convention: n-vs-d sample-floor labels
    # (extraction/fit_checks.py) — records only, never a refusal.
    d = int(next(iter(benign.values())).shape[1])
    floor_records = {
        "benign": check_sample_floor(len(benign_prompts), d,
                                     what="diff-in-means rows (benign)",
                                     tag="caa_static_r", log_fn=log_fn),
        "harmful": check_sample_floor(len(harmful_prompts), d,
                                      what="diff-in-means rows (harmful)",
                                      tag="caa_static_r", log_fn=log_fn),
    }
    obj = CaaStaticRObject(
        model, rs, base_scale=base_scale,
        metadata={
            "spec": PEER_SPEC, "arm": "caa_static_r",
            "layers": list(steer_layers),
            "n_benign": len(benign_prompts),
            "n_harmful": len(harmful_prompts),
            "direction": "raw_diff_in_means_harmful_minus_benign",
            "site": "pre",  # capture AND apply; the norms below are
            # resid_pre numbers (resid_pre of l == resid_post of l-1)
            "per_layer_r_norm": per_layer_norm,
            "per_layer_mean_separation": per_layer_sep,
            "position_convention": "prefill_all",
            # this lane's fields (downstream/report surfaces)
            "deployment": DEPLOYMENT,
            "deployment_matched_to": "alphasteer_paper_10layer",
            "deployed_lam": float(base_scale),  # user-specified, 0.05
            "recommended_lam_grid": list(RECOMMENDED_LAM_GRID),
            "sample_floor": floor_records,
        })
    obj.to_(next(model.parameters()).device)
    if log_fn is not None:
        scalars = {}
        for l in steer_layers:
            scalars[f"caa_static_r/r_norm_l{l}"] = per_layer_norm[l]
            scalars[f"caa_static_r/mean_separation_l{l}"] = per_layer_sep[l]
        scalars["caa_static_r/n_benign"] = len(benign_prompts)
        scalars["caa_static_r/n_harmful"] = len(harmful_prompts)
        scalars["caa_static_r/deployed_lam"] = float(base_scale)
        log_fn(scalars, step=0)
    return obj
