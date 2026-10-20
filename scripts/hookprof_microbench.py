#!/usr/bin/env python
"""Microbenchmark of the lens hook's per-step parts with the PRODUCTION objects.

Companion to scripts/hookprof_served.py (see its docstring for the measurement
this decomposes). Loads a standup's exported registry
(``WorkerObjectRegistry.load(<standup>/lens_registry).materialize``), then times
on the allocated GPU, CUDA-synchronized:

* the raw primitives behind the hook's syncs and launches (a tiny kernel launch,
  ``torch.tensor(..., device=cuda)`` = blocking pageable H2D copy, ``.any().item()``,
  ``torch.equal`` on CUDA, ``.tolist()``, a Python copy of a 1250-int list);
* ``registry.apply`` on one decode row exactly as ``_hook_body_inner`` calls it,
  and the whole slot loop (slice / apply / delta / clone / ``new_hidden[rows]``
  write) for one layer at B in {8, 32} — x32 layers = the apply cost per step;
* ``resolve_attn_metadata`` + ``build_request_slots`` on realistic GPU metadata
  (a Qwen3.5-style 8 flash-attn + 24 GDN layer dict, M-RoPE (3, T) positions) —
  the classification ``_classify_forward`` runs at EVERY hooked layer;
* the vectorized alternative (one gated op on all B rows) per layer, and its
  bit-identity against the per-row path (``torch.equal``).

Usage (inside a GPU allocation, the serving venv, the repo on PYTHONPATH):
  python scripts/hookprof_microbench.py <repo> <standup>/lens_registry <out.json>
"""
import json, os, sys, time, types
import torch
REPO = sys.argv[1]; REG = sys.argv[2]; OUT = sys.argv[3]
sys.path.insert(0, os.path.join(REPO, "src"))
from sisteer.capeval.serving import lens_adapters as la, lens_steer as ls, lens_worker as lw
dev = torch.device("cuda"); dt = torch.bfloat16; d = 2560
torch.manual_seed(0)
res = {}

def bench(fn, n=300, warm=30):
    for _ in range(warm): fn()
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(n): fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1e6

# -- primitives ---------------------------------------------------------------
h1 = torch.randn(1, d, device=dev, dtype=dt)
pos3 = torch.arange(64, device=dev).repeat(3, 1)
qsl = torch.arange(33, device=dev, dtype=torch.int32)
res["prim_tiny_kernel_launch_us"] = bench(lambda: h1.mul(1.0))
res["prim_blocking_h2d_torch_tensor_us"] = bench(lambda: torch.tensor((1234,), dtype=torch.long, device=dev))
m = torch.ones(1, dtype=torch.bool, device=dev)
res["prim_any_item_us"] = bench(lambda: bool(m.any()))
res["prim_torch_equal_64_us"] = bench(lambda: torch.equal(pos3[1], pos3[0]))
res["prim_tolist_33_us"] = bench(lambda: qsl.tolist())
res["prim_python_list_copy_1250_us"] = bench(lambda: list(list(range(992))) + list(list(range(258))), n=2000)

# -- the production registry + apply path ------------------------------------
reg = ls.WorkerObjectRegistry.load(__import__("pathlib").Path(REG)).materialize(device=dev, dtype=dt)
cell = "si_constitution__psr_kl_all"
obj = reg.select(cell, [o for o in reg.objects() if o.cell_id == cell][0].expect_hash)
res["object_n_ops"] = len(obj.ops); res["object_positions"] = sorted({op.positions for op in obj.ops})
res["object_effective_scale"] = obj.effective_scale
ctx_dec = la.HookContext(phase=la.PHASE_DECODE, positions=(1234,), token_ids=(42,))
layers = sorted({op.layer for op in obj.ops})
L0 = layers[len(layers) // 2]
res["apply_1row_decode_us"] = bench(lambda: reg.apply(cell, obj.expect_hash, h1, layer=L0, scale=None, ctx=ctx_dec))
# the slot loop as written, one layer, B rows (slice / apply / sub / clone / index add)
def slot_loop(B):
    true = torch.randn(B, d, device=dev, dtype=dt); hidden = true
    def run():
        new = None
        for i in range(B):
            rows = slice(i, i + 1)
            tr = true[rows]
            ap = reg.apply(cell, obj.expect_hash, tr, layer=L0, scale=None, ctx=ctx_dec)
            delta = ap - tr
            if new is None: new = hidden.clone()
            new[rows] = new[rows] + delta
        return new
    return run
for B in (8, 32):
    us = bench(slot_loop(B), n=60, warm=10)
    res[f"slot_loop_one_layer_B{B}_us"] = us
    res[f"slot_loop_per_step_32layers_B{B}_ms"] = us * 32 / 1000
# -- vectorized per-layer alternative (exact same math on all rows at once) ----
op = [o for o in obj.ops if o.layer == L0][0]
def batched(B):
    H = torch.randn(B, d, device=dev, dtype=dt)
    def run():
        g = torch.relu(H.float() @ op.gate_w + op.gate_b)
        add = (obj.effective_scale * op.coeff * g).unsqueeze(-1) * op.vector
        return H + add.to(dt)
    return run
for B in (8, 32, 64):
    us = bench(batched(B))
    res[f"batched_one_layer_B{B}_us"] = us
    res[f"batched_per_step_32layers_B{B}_ms"] = us * 32 / 1000
# equivalence of the vectorized path with the per-row path (bit-identical rows)
H = torch.randn(32, d, device=dev, dtype=dt)
per_row = torch.cat([reg.apply(cell, obj.expect_hash, H[i:i+1], layer=L0, scale=None, ctx=ctx_dec) for i in range(32)])
g = torch.relu(H.float() @ op.gate_w + op.gate_b); vec = H + ((obj.effective_scale * op.coeff * g).unsqueeze(-1) * op.vector).to(dt)
res["batched_equals_per_row_bitwise"] = bool(torch.equal(per_row, vec))
res["batched_vs_per_row_max_abs_diff"] = float((per_row.float() - vec.float()).abs().max())

# -- the per-layer classification: resolve_attn_metadata + build_request_slots -
def classify_inputs(B, seq=1250, prompt=992):
    seq_lens = torch.full((B,), seq, device=dev, dtype=torch.int32)
    qsl = torch.arange(B + 1, device=dev, dtype=torch.int32)
    positions = torch.full((3, B), seq - 1, device=dev, dtype=torch.long)
    req_ids = [f"req-{i}" for i in range(B)]
    raw = [{"petri_object": cell, "expect_hash": obj.expect_hash} for _ in range(B)]
    toks = list(range(seq))
    md0 = types.SimpleNamespace(query_start_loc=qsl, seq_lens=seq_lens, num_actual_tokens=B)
    gdn = types.SimpleNamespace(num_actual_tokens=B, num_spec_decodes=0, num_spec_decode_tokens=0,
                                non_spec_query_start_loc=qsl.clone())
    md = {f"model.layers.{l}.attn": (md0 if l % 4 == 3 else gdn) for l in range(32)}
    def run():
        q, s_, nat = lw.resolve_attn_metadata(md)
        return lw.build_request_slots(query_start_loc=q, seq_lens=s_, num_actual_tokens=nat, positions=positions,
                                      req_ids=req_ids, raw_selectors=raw, token_lookup=lambda r: toks,
                                      prompt_len_lookup=lambda r: prompt, registry=reg, allow_resumed=False)
    return run
for B in (8, 32):
    us = bench(classify_inputs(B), n=100, warm=10)
    res[f"classify_one_layer_B{B}_us"] = us
    res[f"classify_per_step_32layers_B{B}_ms"] = us * 32 / 1000
# classification with the token flatten replaced by a 1-token slice (what a ranged accessor would cost)
def classify_cheap_tokens(B, seq=1250):
    f = classify_inputs(B)
    return f
res["note"] = ("classify_* = resolve_attn_metadata + build_request_slots per hooked layer, as _classify_forward runs "
               "them at EVERY layer; slot_loop_* = the apply loop of _hook_body_inner; batched_* = one gated op on all rows; "
               "per_step = x32 layers. Wall-clock, CUDA-synchronized, on the allocated H200.")
res["gpu"] = torch.cuda.get_device_name(0); res["torch"] = torch.__version__
json.dump(res, open(OUT, "w"), indent=1)
for k, v in res.items():
    print(f"{k}: {v}")
