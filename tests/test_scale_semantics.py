"""--scale honesty for the 2x-strength probe pass: a
``probe_score.py score --scale 2.0`` invocation must be
honest END TO END for BOTH campaign arm kinds.

Verified seams: score_items passes its ``scale`` into ``obj.apply`` (a
recording fake); PSRObject's hook applies h + scale·g(h)·d and
MultiLayerLoRAObject's applies out + scale·Δ(x) — both EXACTLY linear in
scale at the hooked module (captured there, since downstream layers
compose nonlinearly); and the full CLI path (load_object → identity gate →
score_items) produces different frozen records at scale 2.0 than at 1.0
for both kinds, with the scale recorded in the run meta. If either object
kind ever silently ignored scale, the linearity or the CLI-difference pins
here would fail.
"""

from __future__ import annotations

import json
import sys
from contextlib import contextmanager, nullcontext
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "scripts" / "sdfbase") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts" / "sdfbase"))

import build_probe_set as bps  # noqa: E402
import probe_score as ps  # noqa: E402
from test_probe_scorer import TINY_LLAMA, make_probe_items  # noqa: E402


# --------------------------------------------------------------------------
# score_items -> obj.apply plumbing
# --------------------------------------------------------------------------


class RecordingObj:
    def __init__(self):
        self.seen: list[float] = []

    @contextmanager
    def apply(self, model, scale: float = 1.0):
        self.seen.append(float(scale))
        yield None


def test_score_items_passes_scale_to_apply(tiny_llama):
    model, tok = tiny_llama
    fake = RecordingObj()
    ps.score_items(model, tok, make_probe_items()[:2], obj=fake, scale=2.0,
                   batch_size=2)
    assert fake.seen == [2.0]  # entered ONCE, with the requested scale


# --------------------------------------------------------------------------
# Hook-level linearity, both arm kinds
# --------------------------------------------------------------------------


def _psr_obj(model):
    import torch

    from sisteer.extraction.psr import PSRObject

    obj = PSRObject(model, [0], train_direction=True)
    g = torch.Generator().manual_seed(7)
    d = obj.directions[0].shape[0]
    obj.directions[0].data = torch.randn(d, generator=g) * 0.05
    obj.gate_w[0].data = torch.randn(d, generator=g) * 0.01
    obj.gate_b[0].data = torch.tensor([2.0])  # gate open on every token
    return obj


def _lora_obj(model, seed: int = 9):
    """A seeded, non-trivial rank-4 lora_all object (also test_lora_subtract's
    parent builder, at its own seeds)."""
    import torch

    from sisteer.extraction.multilayer import MultiLayerLoRAObject

    obj = MultiLayerLoRAObject(model, rank=4)
    g = torch.Generator().manual_seed(seed)
    for l in obj.layers:
        obj.A[l].data = torch.randn(obj.A[l].shape, generator=g) * 0.05
        obj.B[l].data = torch.randn(obj.B[l].shape, generator=g) * 0.05
    return obj


def _captured_out(model, tok, module, obj, scale):
    """The hooked module's output under obj at ``scale`` — the capture hook
    registers inside the apply ctx, so it sees the edited tensor. PSR's
    decoder-layer hook may return a tuple; normalize to the hidden."""
    import torch

    captured = {}

    def cap(_m, _i, out):
        captured["x"] = (out[0] if isinstance(out, tuple) else out
                         ).detach().float().clone()

    enc = tok("a scale is honored or it is refused", return_tensors="pt")
    ctx = obj.apply(model, scale=scale) if obj is not None else nullcontext()
    with torch.no_grad(), ctx:
        h = module.register_forward_hook(cap)
        try:
            model(**enc)
        finally:
            h.remove()
    return captured["x"]


@pytest.mark.parametrize("kind", ["psr", "lora"])
def test_hook_delta_is_linear_in_scale(tiny_llama, kind):
    import torch

    from sisteer.models import get_decoder_layers

    model, tok = tiny_llama
    if kind == "psr":
        obj = _psr_obj(model)
        module = get_decoder_layers(model)[0]  # PSR edits the layer output
    else:
        obj = _lora_obj(model)
        module = get_decoder_layers(model)[0].self_attn.o_proj
    base = _captured_out(model, tok, module, None, 1.0)
    d1 = _captured_out(model, tok, module, obj, 1.0) - base
    d2 = _captured_out(model, tok, module, obj, 2.0) - base
    assert d1.abs().max() > 0, "the object must move the hooked module"
    assert torch.allclose(d2, 2.0 * d1, atol=1e-5), \
        f"{kind}: scale 2.0 is not 2x the scale-1.0 delta — scale dishonored"


# --------------------------------------------------------------------------
# End to end through the CLI (load_object + identity gate + scoring)
# --------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["psr", "lora"])
def test_cli_scale_2_changes_records_and_is_recorded(tiny_llama, tmp_path,
                                                     kind):
    model, tok = tiny_llama
    obj = _psr_obj(model) if kind == "psr" else _lora_obj(model)
    art = tmp_path / f"{kind}_object.pt"
    obj.save(art)

    probe = tmp_path / "probe.jsonl"
    man = tmp_path / "probe.manifest.json"
    bps.freeze_items(make_probe_items()[:3], probe, man)

    outs = {}
    for scale in (1.0, 2.0):
        out = tmp_path / f"scores_{kind}_{scale}.jsonl"
        rc = ps.main([
            "score", "--probe", str(probe), "--manifest", str(man),
            "--model", TINY_LLAMA, "--device", "cpu", "--dtype", "float32",
            "--object", str(art), "--scale", str(scale),
            "--batch-size", "2", "--out", str(out)])
        assert rc == 0
        outs[scale] = out.read_text()
        meta = json.loads(Path(str(out) + ".meta.json").read_text())
        assert meta["scale"] == scale  # the run meta records what it ran
    assert outs[1.0] != outs[2.0], \
        f"{kind}: --scale 2.0 produced identical records to --scale 1.0"
