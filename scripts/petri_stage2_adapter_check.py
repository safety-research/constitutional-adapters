#!/usr/bin/env python
"""Run a REAL stage-2 object through the lens adapters on a real model.

The narrowest possible end-to-end question, deliberately isolated from
the vLLM serving wiring: does the adapter's translation of a real
landed artifact reproduce THE OBJECT'S OWN intervention, at full model
scale, and does it produce sensible steered text?

Two claims, checked separately because they fail differently:

1. EQUIVALENCE. Under the same prompt, the hidden states produced by
   ``apply_hook_ops`` over the adapter's ops must match those produced
   by the object's own ``apply(model, scale)`` — the authoritative hook
   — layer by layer. This is the claim that the served intervention IS
   the fit intervention. It is checked bit-exactly first; if that fails
   the max absolute deviation is reported rather than the check being
   loosened, because "close" is a different finding from "equal". A
   non-exact result FAILS the run (status "fail", exit nonzero,
   :func:`overall_status`): an adapter that steers-but-differently is
   exactly the defect this check exists to catch, so the deviation is
   evidence attached to the failure, never a downgrade of it.
2. IT ACTUALLY STEERS, AND STILL SPEAKS. Greedy generations steered
   and unsteered must differ (a silent no-op is the failure mode this
   whole path is built to catch), and the steered text must still be
   language. Those are separate verdicts: a defense that refuses is
   working; a model that has stopped producing sentences is broken,
   and only the second is a bug. Both are printed for eyeballing.

Nothing here touches vLLM. The point is to establish that the ADAPTERS
are right before the serving wiring is blamed for anything.

Prompts are synthetic and harmless, defined inline; generated text goes
to the job log only, never to the JSON record.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from sisteer.capeval.serving import lens_adapters as la


def overall_status(report) -> str:
    """The record's verdict, from the three gating booleans. Claim 1
    (``equivalence_bit_exact``) GATES: the whole point of this script is
    that the served intervention IS the fit intervention, so an adapter
    that steers-but-differently must exit nonzero, never report "pass"
    with the mismatch buried in a diagnostic field. Claim 2's two
    booleans gate as before."""
    ok = (report["equivalence_bit_exact"]
          and report["steering_bites"]
          and report["scale0_is_noop"])
    return "pass" if ok else "fail"


#: harmless, invented — enough to see whether the model still speaks
PROMPTS = (
    "What is 17 times 24?",
    "Name three coastal cities and one fact about each.",
    "Explain in two sentences why bread rises.",
    "Write a haiku about a lighthouse.",
)


def _decoder_layers(model):
    for path in (("model", "layers"), ("model", "decoder", "layers")):
        obj = model
        for attr in path:
            obj = getattr(obj, attr, None)
            if obj is None:
                break
        if obj is not None:
            return obj
    raise RuntimeError("could not locate the decoder layers on this model")


def _hidden(out):
    return out[0] if isinstance(out, tuple) else out


def _rewrap(out, new):
    return (new,) + tuple(out[1:]) if isinstance(out, tuple) else new


class _AdapterHooks:
    """Apply the adapter's ops through plain torch forward hooks.

    Every op on this path is position- and phase-independent (PSR steers
    every position in both phases), so no HookContext is needed and none
    is invented — an op that DID need one would refuse here, loudly,
    which is the correct outcome for this script."""

    def __init__(self, model, ops, scale):
        self.layers = _decoder_layers(model)
        self.ops = ops
        self.scale = scale
        self.handles = []

    def __enter__(self):
        by_layer = {}
        for op in self.ops:
            if la.op_needs_context(op):
                raise RuntimeError(
                    f"op {op.op!r} at layer {op.layer} needs a HookContext, "
                    "which this plain-HF harness does not build — run it "
                    "through the serving path instead of here")
            by_layer.setdefault(op.layer, []).append(op)
        for layer_idx, ops in by_layer.items():
            def make(idx, ops):
                def hook(_m, _a, out):
                    h = _hidden(out)
                    new = la.apply_hook_ops(h, ops, self.scale, layer=idx)
                    return out if new is h else _rewrap(out, new)
                return hook
            self.handles.append(
                self.layers[layer_idx].register_forward_hook(
                    make(layer_idx, ops)))
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()
        self.handles = []
        return False


@torch.no_grad()
def _layer_states(model, enc):
    """Every decoder layer's output hidden state for one forward."""
    got = {}
    handles = []
    for i, layer in enumerate(_decoder_layers(model)):
        def make(idx):
            def hook(_m, _a, out):
                got[idx] = _hidden(out).detach().clone()
            return hook
        handles.append(layer.register_forward_hook(make(i)))
    try:
        model(**enc)
    finally:
        for h in handles:
            h.remove()
    return got


@torch.no_grad()
def _generate(model, tok, prompts, max_new_tokens):
    outs = []
    for p in prompts:
        text = tok.apply_chat_template([{"role": "user", "content": p}],
                                       tokenize=False,
                                       add_generation_prompt=True)
        enc = tok(text, return_tensors="pt").to(model.device)
        ids = model.generate(**enc, max_new_tokens=max_new_tokens,
                             do_sample=False,
                             pad_token_id=tok.eos_token_id)
        outs.append(tok.decode(ids[0][enc["input_ids"].shape[1]:],
                               skip_special_tokens=True))
    return outs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--object", required=True, help="path to object.pt")
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    import inspect

    from transformers import AutoModelForCausalLM, AutoTokenizer

    from sisteer.methods import load_object

    # Preflight the seams BEFORE the weight load. Loading 8B takes about
    # a minute; a signature that has moved should cost a second, not a
    # GPU allocation. (It already did once: load_object takes the model
    # and tokenizer, and the mismatch surfaced only after the weights
    # were resident.)
    try:
        inspect.signature(load_object).bind(Path(args.object), None, None)
    except TypeError as e:
        raise SystemExit(
            f"load_object's signature has moved ({e}) — re-pin this "
            "script's call before spending a weight load on it")

    import hashlib

    art = Path(args.object).read_bytes()
    blob = torch.load(Path(args.object), map_location="cpu",
                      weights_only=True)
    kind = blob.get("kind")
    report = {"object": str(args.object), "kind": kind, "scale": args.scale,
              # provenance: WHICH bytes this verdict is about — without
              # the hash a later reader cannot tell whether the checked
              # object is the one an audit went on to serve.
              "object_sha256": hashlib.sha256(art).hexdigest(),
              "model": args.model, "revision": args.revision}

    ops = la.get_adapter(kind)({"kind": kind}, art, b"")
    report["n_ops"] = len(ops)
    report["op_kinds"] = sorted({o.op for o in ops})
    report["layers"] = [o.layer for o in ops]
    print(f"[check] {kind}: {len(ops)} ops, {report['op_kinds']}, "
          f"layers {report['layers'][:4]}..{report['layers'][-2:]}")

    tok = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    # cuDNN SDPA re-plans per ragged batch shape (33-45% GPU duty measured);
    # every model-load site carries the guard (test_fast_sdpa).
    from sisteer.models import configure_fast_sdpa
    configure_fast_sdpa()
    # No device_map: it pulls in accelerate, which this check does not
    # otherwise need. A plain .to("cuda") on a single-GPU job is the
    # same placement with one fewer dependency to have present.
    model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=args.revision, dtype=torch.bfloat16)
    model.to("cuda")
    model.eval()
    dev = next(model.parameters()).device
    ops = la.materialize_ops(ops, device=dev, dtype=model.dtype)

    # --- claim 1: the adapter reproduces the object's own hook ---------

    obj = load_object(Path(args.object), model, tok)
    probe = tok.apply_chat_template(
        [{"role": "user", "content": PROMPTS[0]}], tokenize=False,
        add_generation_prompt=True)
    enc = tok(probe, return_tensors="pt").to(dev)

    with obj.apply(model, scale=args.scale):
        ref = _layer_states(model, enc)
    with _AdapterHooks(model, ops, args.scale):
        got = _layer_states(model, enc)

    worst, worst_layer, exact = 0.0, None, True
    for l in sorted(ref):
        a, b = ref[l], got[l]
        if not torch.equal(a, b):
            exact = False
            d = float((a.float() - b.float()).abs().max())
            if d > worst:
                worst, worst_layer = d, l
    report["equivalence_bit_exact"] = exact
    report["equivalence_max_abs_delta"] = worst
    report["equivalence_worst_layer"] = worst_layer
    print(f"[check] equivalence vs the object's own hook: "
          f"bit_exact={exact} max_abs_delta={worst} at layer {worst_layer}")

    # --- claim 2: it steers, and still speaks --------------------------
    base = _generate(model, tok, PROMPTS, args.max_new_tokens)
    with _AdapterHooks(model, ops, args.scale):
        steered = _generate(model, tok, PROMPTS, args.max_new_tokens)
    with _AdapterHooks(model, ops, 0.0):
        zero = _generate(model, tok, PROMPTS, args.max_new_tokens)

    report["n_prompts"] = len(PROMPTS)
    report["n_changed_by_steering"] = sum(
        1 for a, b in zip(base, steered) if a != b)
    report["scale0_is_noop"] = (zero == base)
    report["steering_bites"] = report["n_changed_by_steering"] > 0
    print(f"[check] steering changed {report['n_changed_by_steering']}"
          f"/{len(PROMPTS)} generations; scale-0 no-op: "
          f"{report['scale0_is_noop']}")

    print("\n" + "=" * 72)
    for p, b, s in zip(PROMPTS, base, steered):
        print(f"\nPROMPT: {p}\n  BASE   : {b.strip()[:300]}"
              f"\n  STEERED: {s.strip()[:300]}")
    print("\n" + "=" * 72)
    print("Judge the steered column on TWO separate axes: whether it "
          "refuses (a defense doing its job, not a bug), and whether it "
          "is still LANGUAGE (repetition loops or word salad are a bug).")

    report["status"] = overall_status(report)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2,
                                             sort_keys=True))
    print(json.dumps(report, indent=1, sort_keys=True))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
