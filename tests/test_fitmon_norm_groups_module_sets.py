"""fitmon.norm_groups over the LoRA factor layouts (paper-companion-repo stage E, 2026-09-30): the module-set object (FD 6) keys its
``A``/``B`` factors ``"<layer>.<module>"`` — several modules per block, routed experts as 3-D ``[E, r_e, d]`` factors — while the legacy
o_proj object keys them by the layer int. The reporter indexed the dicts by ``obj.layers`` and the first module-set trial died at its
first heartbeat (``KeyError: 0``). Now it iterates the object's own keys and combines a layer's modules into one |B_l|; the legacy
layout's keys and values are unchanged. Synthetic objects only (no model, no GPU)."""
from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from sisteer.extraction.fitmon import norm_groups  # noqa: E402

MENU = {"wo": ["o_proj"], "qkvo": ["k_proj", "o_proj", "q_proj", "v_proj"], "mlp": ["down_proj", "gate_proj", "up_proj"],
        "down": ["down_proj"], "attn_mlp": ["down_proj", "gate_proj", "k_proj", "o_proj", "q_proj", "up_proj", "v_proj"]}
ATTN = {"k_proj", "o_proj", "q_proj", "v_proj"}


def _legacy(layers, rank=4, d=16):
    g = torch.Generator().manual_seed(0)
    A = {l: torch.nn.Parameter(torch.randn(d, rank, generator=g)) for l in layers}
    B = {l: torch.nn.Parameter(torch.randn(d, rank, generator=g)) for l in layers}
    return SimpleNamespace(A=A, B=B, layers=list(layers), legacy=True, factor_keys=list(layers))


def _module_set(set_id, rank, n_blocks=8, attn_every=4, n_experts=6, active=2, d=16):
    """The general-mode layout: attention modules on every attn_every-th block only; FFN modules on every block as ROUTED 3-D factors
    (rank max(1, r // active)) plus a shared-expert 2-D factor; keys "<layer>.<module>" in target order; ``layers`` = sorted blocks."""
    g = torch.Generator().manual_seed(1)
    A, B, keys = {}, {}, []
    r_e = max(1, rank // active)
    for layer in range(n_blocks):
        for m in MENU[set_id]:
            if m in ATTN:
                if layer % attn_every:
                    continue
                k = f"{layer}.self_attn.{m}"
                A[k] = torch.nn.Parameter(torch.randn(d, rank, generator=g)); B[k] = torch.nn.Parameter(torch.randn(d, rank, generator=g))
                keys.append(k)
            else:
                k = f"{layer}.mlp.experts.{m}"                                        # the routed experts: [E, r_e, d] / [E, d, r_e]
                A[k] = torch.nn.Parameter(torch.randn(n_experts, r_e, d, generator=g)); B[k] = torch.nn.Parameter(torch.randn(n_experts, d, r_e, generator=g))
                keys.append(k)
                k2 = f"{layer}.mlp.shared_expert.{m}"                                 # the shared expert: full rank, 2-D
                A[k2] = torch.nn.Parameter(torch.randn(d, rank, generator=g)); B[k2] = torch.nn.Parameter(torch.randn(d, rank, generator=g))
                keys.append(k2)
    layers = sorted({int(k.split(".", 1)[0]) for k in keys})
    return SimpleNamespace(A=A, B=B, layers=layers, legacy=False, factor_keys=keys)


def _finite(d):
    return all(math.isfinite(float(v)) for v in d.values())


def test_legacy_layout_keys_and_values_are_unchanged():
    obj = _legacy([3, 7, 11])
    out = norm_groups(obj)
    assert set(out) == {"norms/A_total", "norms/B_total", "norms/B_max", "norms/B_03", "norms/B_07", "norms/B_11"}
    for l in (3, 7, 11):
        assert torch.allclose(out[f"norms/B_{l:02d}"], obj.B[l].detach().norm())
    b = torch.stack([obj.B[l].detach().norm() for l in (3, 7, 11)])
    assert torch.allclose(out["norms/B_total"], b.norm()) and torch.allclose(out["norms/B_max"], b.max()) and _finite(out)


@pytest.mark.parametrize("set_id", sorted(MENU))
@pytest.mark.parametrize("rank", [4, 16, 64])
def test_every_module_set_in_the_menu_reports_finite_totals_and_one_norm_per_layer(set_id, rank):
    obj = _module_set(set_id, rank)
    out = norm_groups(obj)                                                 # used to raise KeyError: 0 (A[l] for l in obj.layers)
    assert _finite(out) and {"norms/A_total", "norms/B_total", "norms/B_max"} <= set(out), out.keys()
    per_layer = {k for k in out if k.startswith("norms/B_") and k[8:].isdigit()}
    assert per_layer == {f"norms/B_{l:02d}" for l in obj.layers}, (per_layer, obj.layers)
    # a layer's modules combine into ONE norm: sqrt(sum of squares) of that layer's |B| factors
    for l in obj.layers:
        mine = [obj.B[k].detach().norm() for k in obj.factor_keys if int(k.split(".", 1)[0]) == l]
        assert torch.allclose(out[f"norms/B_{l:02d}"], torch.stack(mine).norm())
    every = torch.stack([obj.B[k].detach().norm() for k in obj.factor_keys])
    assert torch.allclose(out["norms/B_total"], every.norm()) and torch.allclose(out["norms/B_max"], every.max())


def test_string_keys_without_factor_keys_still_dispatch_on_the_dicts_own_keys():
    obj = _module_set("down", 4); del obj.factor_keys                        # an object without the property: the dict's keys stand
    out = norm_groups(obj)
    assert _finite(out) and f"norms/B_{obj.layers[0]:02d}" in out


def test_values_are_zero_dim_tensors_never_synced_here():
    obj = _module_set("attn_mlp", 64)
    out = norm_groups(obj)
    assert all(isinstance(v, torch.Tensor) and v.dim() == 0 for v in out.values())
