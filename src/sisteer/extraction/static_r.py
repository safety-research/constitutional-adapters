"""``StaticRObject`` — the fixed-direction (CAA-style) residual write, kind
``static_r``: the object class behind the served-only ``arditi`` arm
(``registry_adapter.ARM_OBJECT_KINDS["arditi"] == "static_r"``).

VENDORED, not written. The class lived in ``extraction/baseline_arms.py``
on an earlier development branch's lineage
(``2e5ddb45:src/sisteer/extraction/baseline_arms.py:272-494``) and is absent
from this tree, so ``methods.load_object`` raised "unknown intervention kind"
for every Arditi artifact and the fp32 oracle (``petri_serving_gate
gate-worker-reference`` -> ``arm_for_object`` -> ``load_object``) could not
compute a reference for an Arditi cell. The ARITHMETIC below is that commit's,
byte for byte — ``_edited`` and both hooks are transcribed unchanged — because
it is the ground truth the native residual slot's constant write is certified
against (``tests/test_static_r_object.py``) and the lens adapter
(``serving/lens_adapters._adapt_static_r``) reads the same blob. The fitting
functions (``train_static_r`` / ``train_static_r_matched`` and the FishBack
arms that shared the module) are deliberately NOT vendored: no trainer for this
arm exists on this tree, the objects are staged outside it, and a second
fitting path would be a second source of truth.

THE TWO DEPLOYMENT SHAPES share the kind and differ in BOTH mask and SITE, so
neither may be inferred from the other:

- ``steer_positions="prompt_last"`` at ``site="post"`` (the landed A1 arm):
  ``h[:, -1, :] += lam * r`` at the last prompt token of the single steered
  layer's OUTPUT, once, during prefill; generation sees the edit only through
  the KV cache.
- ``steer_positions="prefill_all"`` at ``site="pre"`` (the static_r_matched /
  Arditi deployment): ``lam * r_l`` added to EVERY prompt position at the
  INPUT of every steered block, prefill only. ``resid_pre`` of block ``l`` is
  the output of block ``l-1``, which is why every serving path shifts the
  index (``lens_adapters._block_output_layer``; the residual slot's exported
  ``boundaries``) and why block 0 has no servable site.

Dial = ``lam = base_scale * scale``; ``scale=0`` is an EXACT bit-identical
no-op. Decode forwards (``seq_len <= 1``) are NEVER edited in either shape —
the edit persists into generation through the steered prompt KV alone.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import torch

from ..models import get_decoder_layers
from .alphasteer import SITES

#: the position convention of the landed A1 arm; an instance carries the
#: convention it actually deploys under (``position_convention`` below), and
#: the frontier harness refuses to tabulate points from mixed conventions
POSITION_CONVENTION = "prompt_last"

#: ``StaticRObject``'s two deployment shapes (the lens adapter and the
#: residual-slot exporter validate against this exact vocabulary)
STATIC_R_STEER_POSITIONS = ("prompt_last", "prefill_all")


def _prefill_hidden(output):
    return output[0] if isinstance(output, tuple) else output


def _rewrap(output, edited):
    if isinstance(output, tuple):
        return (edited,) + output[1:]
    return edited


class StaticRObject:
    """The fixed-direction (CAA-style) write, in two deployment shapes (module
    docstring). Construct with either the original positional signature::

        StaticRObject(model, layer, r, base_scale, metadata)

    or the keyword-only multi-layer form::

        StaticRObject(model, layers=[8, 9, ...], rs={8: r8, 9: r9, ...},
                      steer_positions="prefill_all", site="pre")

    Exactly one of (layer + r) or (layers + rs) must be given. ``self.layer`` /
    ``self.r`` stay readable on a single-layer object (None on multi-layer);
    the normalized state is always ``self.layers`` / ``self.rs``."""

    kind = "static_r"
    position_convention = POSITION_CONVENTION  # class default; instances
    # override with their own mode in __init__

    def __init__(self, model, layer: int | None = None,
                 r: torch.Tensor | None = None,
                 base_scale: float = 1.0, metadata: dict | None = None, *,
                 layers: list[int] | None = None,
                 rs: dict[int, torch.Tensor] | None = None,
                 steer_positions: str = "prompt_last",
                 site: str = "post"):
        if steer_positions not in STATIC_R_STEER_POSITIONS:
            raise ValueError(
                f"steer_positions must be one of "
                f"{STATIC_R_STEER_POSITIONS}, got {steer_positions!r}")
        if site not in SITES:
            raise ValueError(f"site must be one of {SITES}, got {site!r}")
        multi = layers is not None or rs is not None
        single = layer is not None or r is not None
        if multi == single:
            raise ValueError(
                "give EITHER the single-layer (layer, r) pair OR the "
                "multi-layer (layers, rs) pair — not both, not neither")
        if multi:
            if layers is None or rs is None:
                raise ValueError("multi-layer form needs BOTH layers and rs")
            self.layers = [int(l) for l in layers]
            if not self.layers:
                raise ValueError("layers must be non-empty")
            if len(set(self.layers)) != len(self.layers):
                raise ValueError(f"duplicate layers in {self.layers}")
            rs = {int(l): v for l, v in rs.items()}
            missing = [l for l in self.layers if l not in rs]
            if missing:
                raise ValueError(
                    f"rs missing directions for layers {missing}")
            self.rs = {l: rs[l].detach().float() for l in self.layers}
        else:
            if layer is None or r is None:
                raise ValueError("single-layer form needs BOTH layer and r")
            self.layers = [int(layer)]
            self.rs = {int(layer): r.detach().float()}
        self.base_scale = float(base_scale)
        self.steer_positions = steer_positions
        self.site = site
        self.position_convention = steer_positions
        self.metadata = metadata or {}
        self._decoder = get_decoder_layers(model)

    # -- single-layer views (the legacy attribute surface; None on multi) --

    @property
    def layer(self) -> int | None:
        return self.layers[0] if len(self.layers) == 1 else None

    @property
    def r(self) -> torch.Tensor | None:
        return self.rs[self.layers[0]] if len(self.layers) == 1 else None

    @property
    def trainable(self) -> list:
        return []  # closed-form; nothing trains by gradient

    def n_params(self) -> int:
        return int(sum(v.numel() for v in self.rs.values()))

    def inject(self, rendered_prompt: str) -> str:
        return rendered_prompt

    def _edited(self, hidden: torch.Tensor, r_master: torch.Tensor,
                mag: float) -> torch.Tensor | None:
        """The edit on ``hidden`` [B, S, d], or None when this forward
        must not be steered (decode steps — seq_len == 1 forwards are
        NEVER edited in either mode; the edit persists via the KV cache
        alone)."""
        if hidden.shape[1] <= 1:
            return None
        r = r_master.to(device=hidden.device)
        if self.steer_positions == "prefill_all":
            # every prompt position (AlphaSteer's broadcast; like it, pad
            # positions are edited too — causality + the attention mask
            # keep them from reaching real tokens)
            return (hidden.float() + mag * r).to(hidden.dtype)
        edited = hidden.clone()
        edited[:, -1, :] = (
            hidden[:, -1, :].float() + mag * r).to(hidden.dtype)
        return edited

    @contextmanager
    def apply(self, model, scale: float = 1.0):
        mag = self.base_scale * scale

        def make_post(r_master):
            def hook(_mod, _args, output):
                if mag == 0.0:
                    return output  # exact no-op, bit-identical to unsteered
                hidden = _prefill_hidden(output)
                edited = self._edited(hidden, r_master, mag)
                if edited is None:
                    return output
                return _rewrap(output, edited)

            return hook

        def make_pre(r_master):
            # resid_pre: edit hidden_states ENTERING the block, handling
            # both calling conventions (AlphaSteerObject's make_pre shape)
            def hook(_mod, args, kwargs):
                if mag == 0.0:
                    return None  # exact no-op
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

        handles = []
        for l in self.layers:
            if self.site == "post":
                handles.append(self._decoder[l].register_forward_hook(
                    make_post(self.rs[l])))
            else:
                handles.append(self._decoder[l].register_forward_pre_hook(
                    make_pre(self.rs[l]), with_kwargs=True))
        try:
            yield
        finally:
            for h in handles:
                h.remove()

    def to_(self, device) -> None:
        self.rs = {l: v.to(device) for l, v in self.rs.items()}

    def save(self, path: Path) -> None:
        """Torch blob. A single-layer object also writes the LEGACY
        top-level layer/r keys (the landed A1 artifact format), so
        pre-multi-layer readers keep working; the normalized state lives
        in layers/rs/steer_positions."""
        single = len(self.layers) == 1
        l0 = self.layers[0]
        torch.save(
            {"kind": self.kind,
             "layer": l0 if single else None,
             "r": self.rs[l0].detach().float().cpu() if single else None,
             "layers": list(self.layers),
             "rs": {l: v.detach().float().cpu()
                    for l, v in self.rs.items()},
             "steer_positions": self.steer_positions,
             "site": self.site,
             "base_scale": self.base_scale, "metadata": self.metadata},
            path,
        )

    @classmethod
    def load_blob(cls, blob: dict, model, tok) -> "StaticRObject":
        # Missing fields default to the LEGACY values, so a blob saved
        # before the multi-layer form existed (single layer/r, no
        # steer_positions/site keys) loads and behaves exactly as it
        # always did.
        steer_positions = blob.get("steer_positions", "prompt_last")
        site = blob.get("site", "post")
        if blob.get("rs"):
            return cls(model, base_scale=float(blob["base_scale"]),
                       metadata=blob.get("metadata", {}),
                       layers=[int(l) for l in blob["layers"]],
                       rs={int(l): v for l, v in blob["rs"].items()},
                       steer_positions=steer_positions, site=site)
        return cls(model, layer=int(blob["layer"]), r=blob["r"],
                   base_scale=float(blob["base_scale"]),
                   metadata=blob.get("metadata", {}),
                   steer_positions=steer_positions, site=site)
