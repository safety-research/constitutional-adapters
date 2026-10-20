"""Shared fixtures for the round-2 PSR residual suite (paired write /
teacher recurrence / trainer): a pure-torch TOY multi-layer model with a
NONLINEAR block between hook points, planted PSR objects with distinct,
simultaneously-active gates at several layers, and a hand-built
``CapBatch`` for the hook-level helpers. Synthetic throughout — no real
prompts, no real ids.

The toy model satisfies exactly what the PSR machinery reads: decoder
layers at ``model.model.layers`` (one of ``models._LAYER_PATHS``), a
``config.hidden_size``, and blocks whose forward returns the residual
tensor (PSR's hook handles tensor or tuple outputs). Its reference
recurrence is therefore computable in pure python — the point of the
fixture: every captured layer output must equal
``z_{l+1} = f_l(z_l + V_l(z_l))`` with ``V_l`` evaluated on the SHARED
pre-write state.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch import nn

from sisteer.extraction.psr import PSRObject
from sisteer.extraction.residual import CapBatch, position_classes
from sisteer.extraction.traindata import EncodedSeq, collate

VOCAB = 50
HIDDEN = 12


class ToyBlock(nn.Module):
    """h + tanh(W h + c): a NONLINEAR residual block, so the order of a
    write vs. the block matters and the recurrence is not a linear map."""

    def __init__(self, d: int, seed: int):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.lin = nn.Linear(d, d)
        with torch.no_grad():
            self.lin.weight.copy_(torch.randn(d, d, generator=g) / (d ** 0.5))
            self.lin.bias.copy_(0.1 * torch.randn(d, generator=g))

    def forward(self, h, *args, **kwargs):
        return h + torch.tanh(self.lin(h))


class _Inner(nn.Module):
    def __init__(self, d: int, n_layers: int, seed: int):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.embed = nn.Embedding(VOCAB, d)
        with torch.no_grad():
            self.embed.weight.copy_(torch.randn(VOCAB, d, generator=g))
        self.layers = nn.ModuleList(
            [ToyBlock(d, seed + 101 * (i + 1)) for i in range(n_layers)])


class ToyModel(nn.Module):
    """3-layer (by default) toy causal-LM-shaped module."""

    def __init__(self, d: int = HIDDEN, n_layers: int = 3, seed: int = 0):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=d, num_hidden_layers=n_layers)
        self.model = _Inner(d, n_layers, seed)

    def embed(self, input_ids):
        return self.model.embed(input_ids)

    def forward(self, input_ids, attention_mask=None, **kwargs):
        h = self.model.embed(input_ids)
        for layer in self.model.layers:
            h = layer(h)
        logits = h @ self.model.embed.weight.T
        return SimpleNamespace(logits=logits, last_hidden=h)


def toy_model(n_layers: int = 3, seed: int = 0) -> ToyModel:
    m = ToyModel(n_layers=n_layers, seed=seed)
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def planted_psr(model, layers, *, seed: int, open_frac: float = 0.6,
                write_scale: float = 0.5, constant_gate: bool = False,
                ) -> PSRObject:
    """A PSR object with a distinct random gate + direction per layer. The
    gate threshold is set from the bare states' pre-activation quantile so
    the gate is open on about ``open_frac`` of positions (a partially
    active, input-dependent gate); ``constant_gate`` makes it always-open
    (w = 0, b = 1), the in-class-for-a-fresh-student case."""
    g = torch.Generator().manual_seed(seed)
    d = model.config.hidden_size
    obj = PSRObject(model, list(layers), train_direction=True)
    ids = toy_ids(seed=seed + 7, batch=4, seq=9)
    with torch.no_grad():
        z = model.embed(ids)
        states = {}
        for i, layer in enumerate(model.model.layers):
            z = layer(z)
            states[i] = z.reshape(-1, d).float()  # gates are fp32 on any stream
    for l in layers:
        dvec = torch.randn(d, generator=g)
        dvec = write_scale * dvec / dvec.norm()
        if constant_gate:
            w = torch.zeros(d)
            b = 1.0
        else:
            w = torch.randn(d, generator=g)
            pre = states[l] @ w
            w = w / (pre.std() + 1e-8)
            pre = states[l] @ w
            b = -float(torch.quantile(pre, 1.0 - open_frac))
        obj.gate_w[l].data = w
        obj.gate_b[l].data = torch.tensor([b])
        obj.directions[l].data = dvec
    return obj


def random_psr(model, seed: int, *, layers=None, write: float = 0.08,
               normalize: bool = True, gate_scale: float = 0.4,
               b: float = 0.3, constant_gate: bool = False) -> PSRObject:
    """A seeded random PSR object on a REAL (HF) model — the suite's ONE
    generator of planted parents / teachers / probes (the trainer, builder,
    liveness, paired-write and ce_score tests all draw from it at their own
    knobs). Per layer, in this draw order: the direction ``v ~ N(0, I)``
    (``write * v / ‖v‖`` when ``normalize``, else ``write * v``), then —
    unless ``constant_gate`` (w = 0, b = 1: always open) — the gate
    ``w ~ N(0, I) * gate_scale`` with threshold ``b``. ``layers`` defaults
    to every layer. The draw order is load-bearing: the tests' numeric
    pins are on exactly these tensors."""
    from sisteer.models import num_layers

    g = torch.Generator().manual_seed(seed)
    if layers is None:
        layers = list(range(num_layers(model)))
    obj = PSRObject(model, list(layers), train_direction=True)
    d = obj.directions[obj.layers[0]].shape[0]
    for l in obj.layers:
        v = torch.randn(d, generator=g)
        obj.directions[l].data = (write * v / v.norm() if normalize
                                  else v * write)
        if constant_gate:
            obj.gate_w[l].data = torch.zeros(d)
            obj.gate_b[l].data = torch.tensor([1.0])
        else:
            obj.gate_w[l].data = torch.randn(d, generator=g) * gate_scale
            obj.gate_b[l].data = torch.tensor([b])
    return obj


def zero_psr(model, layers) -> PSRObject:
    """The empty parent ∅: d = 0 everywhere (gate open, writes nothing)."""
    return PSRObject(model, list(layers), train_direction=True)


# --------------------------------------------------------------------------- #
# A bf16 toy with a REAL-MAGNITUDE, chaotic residual stream                    #
# --------------------------------------------------------------------------- #


class ChaosBlock(nn.Module):
    """``Q·h + gain·rms(h)·tanh(W·(x + causal_mean(x)))`` with ``x =
    h/rms(h)``: a norm-preserving rotation ``Q`` spreads any change of one
    element over every dimension, the causal running mean spreads it over
    every later position (the stand-in for attention), and the tanh block
    keeps the map nonlinear. On a bf16 stream that is all chaotic
    amplification needs: a sub-ulp change becomes full-ulp flips with
    probability δ/ulp, so its energy grows by ulp/δ per layer until every
    element sits ~1 ulp off — the saturation the real models show (a 1-ulp
    flip of ONE element at layer 0 reached ~2–3 per position by layer 14
    on gemma-4-E4B), reproduced deterministically in a dozen dimensions.
    Arithmetic in fp32, rounded ONCE to the stream dtype per layer (the
    bf16-matmul-with-fp32-accumulate convention)."""

    def __init__(self, d: int, seed: int, *, gain_frac: float = 0.3):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.lin = nn.Linear(d, d, bias=False)
        with torch.no_grad():
            self.lin.weight.copy_(torch.randn(d, d, generator=g) / (d ** 0.5))
        q, _r = torch.linalg.qr(torch.randn(d, d, generator=g))
        self.register_buffer("rot", q.contiguous())
        self.gain_frac = gain_frac

    def forward(self, h, *args, **kwargs):
        hf = h.float()
        rms = hf.pow(2).mean(-1, keepdim=True).sqrt().clamp_min(1e-6)
        x = hf / rms
        n = torch.arange(1, x.shape[1] + 1, device=x.device,
                         dtype=x.dtype).view(1, -1, 1)
        cum = torch.cumsum(x, dim=1) / n
        u = torch.tanh(self.lin.weight.float() @ (x + cum).transpose(-1, -2)
                       ).transpose(-1, -2)
        out = hf @ self.rot.float().T + self.gain_frac * rms * u
        return out.to(h.dtype)


class _ChaosInner(nn.Module):
    def __init__(self, d: int, n_layers: int, seed: int, stream_scale: float):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.embed = nn.Embedding(VOCAB, d)
        with torch.no_grad():
            self.embed.weight.copy_(stream_scale * torch.randn(VOCAB, d,
                                                               generator=g))
        self.layers = nn.ModuleList(
            [ChaosBlock(d, seed + 101 * (i + 1)) for i in range(n_layers)])


class ChaosToyModel(ToyModel):
    """The toy causal-LM shape (``model.model.layers``, ``config.hidden_size``)
    with :class:`ChaosBlock` blocks and a stream of RMS ``stream_scale`` per
    element — |h| ~ 500, where a bf16 ulp is 2–4."""

    def __init__(self, d: int = HIDDEN, n_layers: int = 6, seed: int = 0,
                 stream_scale: float = 500.0):
        nn.Module.__init__(self)
        self.config = SimpleNamespace(hidden_size=d, num_hidden_layers=n_layers)
        self.model = _ChaosInner(d, n_layers, seed, stream_scale)


def chaos_toy_model(n_layers: int = 6, seed: int = 0, *, d: int = HIDDEN,
                    stream_scale: float = 500.0,
                    dtype: torch.dtype = torch.bfloat16) -> ChaosToyModel:
    m = ChaosToyModel(d=d, n_layers=n_layers, seed=seed,
                      stream_scale=stream_scale).to(dtype)
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def one_ulp_bump_hook(row: int = 0, position: int = 3, element: int = 0):
    """A forward hook that adds ONE ulp of the stream's dtype to a single
    element of the hooked layer's output — the smallest perturbation a
    rounding difference can make (the seed the real models amplified)."""
    int_view = {torch.bfloat16: torch.int16, torch.float16: torch.int16,
                torch.float32: torch.int32, torch.float64: torch.int64}

    def hook(_mod, _args, output):
        hid = output[0] if isinstance(output, tuple) else output
        r, p, e = (min(row, hid.shape[0] - 1), min(position, hid.shape[1] - 1),
                   min(element, hid.shape[2] - 1))
        h2 = hid.clone()
        bumped = (h2[r, p, e].reshape(1).view(int_view[hid.dtype]) + 1
                  ).view(hid.dtype)
        h2[r, p, e] = bumped.reshape(())
        return (h2,) + tuple(output[1:]) if isinstance(output, tuple) else h2

    return hook


def toy_ids(*, seed: int, batch: int = 3, seq: int = 8) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(1, VOCAB, (batch, seq), generator=g)


def reference_recurrence(model, ids: torch.Tensor,
                         terms: list[tuple[float, PSRObject]],
                         scale: float = 1.0) -> dict[int, torch.Tensor]:
    """Pure-python recurrence z_{l+1} = f_l(z_l) + Σ_i c_i·ReLU(w_i·f_l(z_l)
    + b_i)·d_i on the SHARED post-block, pre-write state. Returns the
    post-write state at every layer (what ``_capture_hidden`` sees inside
    the object's apply())."""
    out = {}
    with torch.no_grad():
        h = model.embed(ids)
        for l, layer in enumerate(model.model.layers):
            z = layer(h)
            add = torch.zeros_like(z, dtype=torch.float32)
            for c, obj in terms:
                if l not in obj.layers:
                    continue
                pre = z.float() @ obj.gate_w[l] + obj.gate_b[l]
                add = add + (c * F.relu(pre)).unsqueeze(-1) * obj.directions[l]
            h = z + (scale * add).to(z.dtype)
            out[l] = h.clone()
    return out


def capture_trajectory(model, ids: torch.Tensor, ctx, layers
                       ) -> dict[int, torch.Tensor]:
    """Every layer's output under ``ctx`` (a context manager or None)."""
    from contextlib import nullcontext

    from sisteer.extraction.psr import _capture_hidden

    with torch.no_grad(), (ctx if ctx is not None else nullcontext()), \
            _capture_hidden(model, list(layers)) as cap:
        model(input_ids=ids, attention_mask=torch.ones_like(ids))
        return {l: cap[l].detach().clone() for l in layers}


def toy_cap_batch(ids: torch.Tensor, prompt_lens: list[int],
                  sources: list[str], has_prefix: list[bool] | None = None,
                  ) -> CapBatch:
    """A hand-built CapBatch over toy ids (no tokenizer): rows are full-
    length (no padding), prompt boundaries as given."""
    from sisteer.extraction.residual import CaptureDoc

    seqs = [EncodedSeq(ids=[int(t) for t in row], prompt_len=int(pl))
            for row, pl in zip(ids.tolist(), prompt_lens)]
    b = collate(seqs, pad_id=0)
    if has_prefix is None:
        has_prefix = [pl > 0 for pl in prompt_lens]
    docs = [CaptureDoc(doc_id=f"toy-{i}", source=s,
                       family="advice" if hp else "stories",
                       prompt="sys" if hp else "", text="x")
            for i, (s, hp) in enumerate(zip(sources, has_prefix))]
    prefix, content = position_classes(b, torch.tensor(has_prefix))
    masks, n_pos = {}, {}
    for s in ("mix", "ctl"):
        rows = torch.tensor([d.source == s for d in docs])
        for cls, m in (("content", content), ("prefix", prefix)):
            mm = m & rows[:, None]
            masks[(s, cls)] = mm
            n_pos[(s, cls)] = int(mm.sum())
    return CapBatch(batch=b, docs=docs, masks=masks, n_pos=n_pos,
                    row_prefix=prefix, row_content=content)


# --------------------------------------------------------------------------- #
# Runtime-defective apply(): the defect class the hook contract exists for    #
# --------------------------------------------------------------------------- #


class ApplyPerturbed:
    """A write object whose ALGEBRA is the wrapped object's (``paired_write``,
    ``terms``, ``layers`` all delegate — the off-forward write is exactly
    right) but whose RUNTIME ``apply()`` is wrong in one declared way. This
    is the defect class the hook-contract check exists for: before it, the
    identity gate re-derived both writes off the forward with the same
    helper the implementation uses and could not see apply() at all, so
    every one of these passed.

      noop         no hooks at all (the "teacher" trajectory is the bare model)
      double       the real apply() at scale 2 (the write doubled)
      scale        the real apply() at ``factor`` (a wrong scale)
      wrong_layer  each layer's hook lands on the NEXT layer of the model
      bump         the real apply(), plus a ONE-ulp foreign edit of one
                   stream element at the first hooked layer's OUTPUT, inside
                   apply() — the smallest edit a runtime can make
    """

    HOWS = ("noop", "double", "scale", "wrong_layer", "bump")

    def __init__(self, inner, how: str, *, factor: float = 0.5):
        if how not in self.HOWS:
            raise ValueError(f"how must be one of {self.HOWS}")
        self.inner = inner
        self.how = how
        self.factor = float(factor)
        self.layers = list(inner.layers)
        if hasattr(inner, "terms"):
            self.terms = inner.terms

    def paired_write(self, layer, hidden):
        from sisteer.extraction.residual import _fp32_write
        return _fp32_write(self.inner, layer, hidden)

    def to_(self, device):
        self.inner.to_(device)

    def _hook(self, layer: int, scale: float):
        if isinstance(self.inner, PSRObject):
            return self.inner._make_hook(layer, scale, None)
        return self.inner._make_hook(layer, scale)

    def apply(self, model, scale: float = 1.0):
        from contextlib import contextmanager, nullcontext

        from sisteer.models import get_decoder_layers

        inner, how = self.inner, self.how

        @contextmanager
        def _cm():
            dec = get_decoder_layers(model)
            handles = []
            if how == "noop":
                ctx = nullcontext()
            elif how == "double":
                ctx = inner.apply(model, 2.0 * scale)
            elif how == "scale":
                ctx = inner.apply(model, self.factor * scale)
            elif how == "wrong_layer":
                ctx = nullcontext()
                n = len(dec)
                for l in self.layers:
                    handles.append(dec[(l + 1) % n].register_forward_hook(
                        self._hook(l, float(scale))))
            else:  # bump: a foreign one-ulp edit BEFORE the real write
                handles.append(dec[self.layers[0]].register_forward_hook(
                    one_ulp_bump_hook(row=0, position=3, element=0)))
                ctx = inner.apply(model, scale)
            try:
                with ctx:
                    yield None
            finally:
                for h in handles:
                    h.remove()
        return _cm()
