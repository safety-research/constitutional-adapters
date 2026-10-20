"""GPU validation for the serving-phase probe and a held-out family, end to end.

This is the part that had never happened: no held-out family had produced a single
generation, and the phase probe's forward-pair was dormant. This script, on ONE
model load, does three things and writes a SANITISED summary (numbers/labels/bools
only — never prompt or response text):

PART 2a — wire + validate the probe's FORWARD-PAIR. One forward with an arm
applied, one without, on a fixed synthetic prompt with a known prompt/generated
split (phase_probe_gpu). It diffs hidden states per position; the changed set is
the measured footprint. Validated against a POSITION-MASK cross-phase serve
(VectorObject, positions="response"): the measured footprint must be RESPONSE
(content untouched, header+generated steered) AND the position that produces the
FIRST generated token (the last prompt/header token) must be in the changed set —
i.e. the probe PROVES token 1 is steered under position-mask serving, the property
the rule exists to guarantee. An all-position serve is validated to
measure ALL (content steered too), and the two feed verify_footprint to confirm
cross_phase / matched.

PART 2b — smoke a DETERMINISTIC held-out family (artprompt: no corpus build, no
attacker) end to end against BASELINE conditions (none, prompted). Baselines carry
no object, so the fit-provenance allowlist does not block them. It runs the real
held-out leg plumbing: the per-family manifest preflight (with attack_regime) then
run_artprompt generation. Benign synthetic goals are used — this measures the
end-to-end plumbing and the per-prompt COST, not an ASR (grading is a separate
grading host step). NO harmful content is generated.

Separately asserts that an UNPROVENANCED object condition REFUSES (default-deny via
resolve_training_mirror) — correct behaviour, validated rather than a surprise.

PART 2c — reports measured per-prompt generation cost so the sweep cost model can
be extended with a real held-out number.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def _synthetic_goals(n: int):
    """n BENIGN synthetic goals (no harmful content). artprompt masks the longest
    word and asks the model to decode+answer, so any prose exercises the same
    code path — we measure plumbing + cost, not an ASR."""
    from sisteer.eval.jailbreak_data import Goal
    topics = [
        "Explain how photosynthesis converts sunlight into chemical energy",
        "Describe the process of making sourdough bread from a starter",
        "Summarize how a suspension bridge distributes structural loads",
        "Outline the steps to compost kitchen scraps in a small garden",
        "Explain how noise-cancelling headphones reduce ambient sound",
        "Describe how vaccines train the immune system to recognize pathogens",
        "Explain how a hydroelectric dam generates electricity",
        "Describe how sourwood honey differs from clover honey in flavor",
        "Outline how tectonic plates produce earthquakes along fault lines",
        "Explain how a refrigerator moves heat out of its interior",
    ]
    goals = []
    for i in range(n):
        goals.append(Goal(
            goal_id=f"synthbenign-{i:03d}",
            text=topics[i % len(topics)] + f" (variation {i})",
            category="benign_synthetic",
            dataset="harmbench",  # a valid enum; content is benign+synthetic
            revision="synthetic",
            split="test",
        ))
    return goals


def _validate_probe(model, tok, layer: int, scale: float) -> dict:
    """PART 2a: the forward-pair probe against a position-mask and an all-position
    VectorObject serve. Returns a sanitised result dict."""
    import torch

    from sisteer.eval.jailbreak import response_header_len
    from sisteer.eval.phase_probe import (ALL, RESPONSE, changed_positions,
                                          expand_footprint, measure_footprint,
                                          verify_footprint)
    from sisteer.eval.phase_probe_gpu import decode_probe_capture, token1_position
    from sisteer.methods import VectorObject
    from sisteer.vectors import SteeringVector

    from sisteer.chat_markers import generation_header_ids
    from sisteer.eval.phase_probe_gpu import build_probe_sequence
    from sisteer.models import get_decoder_layers

    d_model = model.config.hidden_size
    # BOTH header derivations, printed and required to AGREE: the serve path
    # (hooks.response_header_len, what _set_response_positions gives the object)
    # and the probe frame (chat_markers.generation_header_ids — the id-form of
    # the SAME width convention; the terminator-anchored located header is NOT
    # comparable, since it absorbs prompt-side scaffold on gemma-4-style
    # templates). A disagreement means the position-mask boundary the object
    # serves is not the boundary the frame measures — refuse loudly and name it.
    H = response_header_len(tok)
    H_markers = len(generation_header_ids(tok))
    tmpl = getattr(tok, "chat_template", None) or ""
    print(f"[validate] header derivations: hooks.response_header_len={H} "
          f"chat_markers={H_markers} (chat_template present={bool(tmpl)}, "
          f"len={len(tmpl)})")
    if H <= 0 or H != H_markers:
        raise SystemExit(
            f"[validate] HEADER DERIVATION DISAGREEMENT: hooks={H} "
            f"chat_markers={H_markers} — the serve path and the probe frame must "
            "derive the same assistant-header length; refusing to validate "
            "against a broken boundary. (hooks.response_header_len<=0 would make "
            "_set_response_positions hard-error on every position-mask serve in "
            "this environment — a real finding, not a probe bug.)")
    # a fixed synthetic unit direction at the pinned layer (semantics irrelevant —
    # the probe measures WHICH positions change, not what they change to).
    g = torch.Generator().manual_seed(0)
    direction = torch.randn(d_model, generator=g)
    direction = direction / direction.norm()
    sv = SteeringVector(vectors={layer: direction}, method="synthetic_probe",
                        model_name="", instruction_id="")

    # --- DIAGNOSTIC prefill (printed, not asserted): the handle state and the
    # steered-layer diff, so a masked-out prefill is attributable from the log
    # (handle boundary vs hook firing vs capture) instead of a guess. ---
    prompt_ids, cont_ids, prompt_width, hdr_len, content_boundary, seq_total = \
        build_probe_sequence(tok)
    prompt_t = torch.tensor([prompt_ids], device=next(model.parameters()).device,
                            dtype=torch.long)

    def _prefill_hidden(obj_or_none, use_cache):
        cap = []
        hook = (lambda m, i, o: cap.append(
            (o[0] if isinstance(o, tuple) else o).detach().float().cpu()))
        lyr = get_decoder_layers(model)[layer]
        hk = None
        try:
            if obj_or_none is None:
                hk = lyr.register_forward_hook(hook)
                with torch.no_grad():
                    model(input_ids=prompt_t, use_cache=use_cache)
            else:
                with torch.no_grad(), obj_or_none.apply(model, scale=scale) as h:
                    # Register the capture INSIDE the ctx: same-module forward
                    # hooks run in registration order, each seeing the previous
                    # hook's returned output — a capture registered BEFORE the
                    # steering hook records the PRE-steering tensor and reads []
                    # at a steered layer. That stale shape produced the
                    # contradictory "changed prefill positions=[]" diag line in
                    # the 173196 log; reconciled on CPU against a hook-free
                    # output_hidden_states/logits ground truth (which agrees with
                    # the validate path, not with the stale diag).
                    hk = lyr.register_forward_hook(hook)
                    if h is not None and hasattr(h, "reset"):
                        h.reset(prompt_len=prompt_width)
                    print(f"[diag] handle pre-forward (use_cache={use_cache}): "
                          f"type={type(h).__name__} "
                          f"positions={getattr(h, 'positions', None)} "
                          f"response_header_len={getattr(h, 'response_header_len', None)} "
                          f"prompt_len={getattr(h, 'prompt_len', None)}")
                    model(input_ids=prompt_t, use_cache=use_cache)
                    print(f"[diag] handle post-prefill (use_cache={use_cache}): "
                          f"prompt_len={getattr(h, 'prompt_len', None)} "
                          f"cur_offset={getattr(h, '_cur_offset', None)} "
                          f"seen={getattr(h, '_seen', None)} "
                          f"cache_len={getattr(h, '_cache_len', None)}")
        finally:
            if hk is not None:
                hk.remove()
        return cap[0][0]  # [prompt_width, d]

    diag_obj = VectorObject(sv, positions="response", skip_bos=True,
                            response_header_len=H)
    for uc in (True, False):
        hb = _prefill_hidden(None, uc)
        hs = _prefill_hidden(diag_obj, uc)
        diffs = (hs - hb).abs().amax(dim=-1)
        changed_prefill = [i for i in range(prompt_width) if float(diffs[i]) > 1e-5]
        print(f"[diag] use_cache={uc}: prompt_width={prompt_width} "
              f"content_boundary={content_boundary} "
              f"header=[{content_boundary},{prompt_width}) "
              f"steered-layer({layer}) changed prefill positions={changed_prefill}")

    # hook-free GROUND TRUTH (no capture hooks share code with the probe): the
    # model's own final hidden + logits with/without the ctx. The prefill's
    # changed set here must contain the token-1-producing position — printed so
    # every GPU log self-corroborates with a differently-shaped instrument.
    with torch.no_grad():
        gt_base = model(input_ids=prompt_t, output_hidden_states=True)
        with diag_obj.apply(model, scale=scale) as h:
            if h is not None and hasattr(h, "reset"):
                h.reset(prompt_len=prompt_width)
            gt_steer = model(input_ids=prompt_t, output_hidden_states=True)
    gt_hidden = [i for i in range(prompt_width) if float(
        (gt_steer.hidden_states[-1][0, i] - gt_base.hidden_states[-1][0, i])
        .abs().max()) > 1e-5]
    gt_logits = [i for i in range(prompt_width) if float(
        (gt_steer.logits[0, i] - gt_base.logits[0, i]).abs().max()) > 1e-5]
    print(f"[diag] GROUND TRUTH (hook-free): final-hidden changed={gt_hidden} "
          f"logits changed={gt_logits} "
          f"token1({prompt_width - 1}) steered={prompt_width - 1 in gt_logits}")

    def _measure(positions):
        obj = VectorObject(sv, positions=positions, skip_bos=True,
                           response_header_len=(H if positions == "response" else None))
        # prompt_len = the FULL prefill width (measure_footprint's convention:
        # the header is the prompt's last H positions).
        prompt_len, seq_len, header_len, ignore_pos, capture = \
            decode_probe_capture(model, tok, obj, scale=scale)
        base = capture(False)
        steer = capture(True)
        changed = changed_positions(base, steer)
        fp = measure_footprint(base, steer, prompt_len=prompt_len,
                               seq_len=seq_len, header_len=header_len,
                               ignore_positions=ignore_pos)
        tok1 = token1_position(prompt_len, header_len)
        return dict(fp=fp, changed=changed, prompt_len=prompt_len,
                    seq_len=seq_len, header_len=header_len,
                    prompt_width=prompt_len, tok1=tok1)

    resp = _measure("response")
    allm = _measure("all")

    # the load-bearing assertions.
    tok1_steered = resp["tok1"] in resp["changed"]
    content_untouched = (resp["fp"].content in ("none", "n/a"))
    resp_label = resp["fp"].label()
    all_label = allm["fp"].label()
    all_content_steered = (allm["fp"].content == "all")

    # feed the measured footprints through the real gate to confirm the labels.
    kw = dict(prompt_len=resp["prompt_len"], seq_len=resp["seq_len"],
              header_len=resp["header_len"])
    kw_all = dict(prompt_len=allm["prompt_len"], seq_len=allm["seq_len"],
                  header_len=allm["header_len"])
    req_resp = expand_footprint("response", **kw)
    req_all = expand_footprint("all", **kw_all)
    defined_all = expand_footprint("all", **kw)
    fit_all = expand_footprint("all", **kw)
    rel_cross = verify_footprint(requested=req_resp, defined=defined_all,
                                 measured=resp["fp"], fit=fit_all,
                                 what="probe-validate:vector served decode-only")
    rel_matched = verify_footprint(
        requested=req_all, defined=expand_footprint("all", **kw_all),
        measured=allm["fp"], fit=expand_footprint("all", **kw_all),
        what="probe-validate:vector served all-position")

    ok = (resp_label == RESPONSE and tok1_steered and content_untouched
          and all_label == ALL and all_content_steered
          and rel_cross == "cross_phase" and rel_matched == "matched")
    return {
        "header_len_H": H,
        "position_mask_serve": {
            "measured_label": resp_label,
            "footprint": [resp["fp"].content, resp["fp"].header, resp["fp"].generated],
            "prompt_width": resp["prompt_width"], "seq_len": resp["seq_len"],
            "token1_position": resp["tok1"],
            "token1_is_steered": bool(tok1_steered),
            "content_untouched": bool(content_untouched),
            "n_changed_positions": len(resp["changed"]),
            "verify_footprint_relation": rel_cross,
        },
        "all_position_serve": {
            "measured_label": all_label,
            "footprint": [allm["fp"].content, allm["fp"].header, allm["fp"].generated],
            "content_steered": bool(all_content_steered),
            "n_changed_positions": len(allm["changed"]),
            "verify_footprint_relation": rel_matched,
        },
        "PASS": bool(ok),
    }


def _smoke_heldout(model, tok, goals, out_root: Path, max_new_tokens: int) -> dict:
    """PART 2b/2c + batching directive: run artprompt end to end in THREE arms —
    none UNBATCHED (batch_size=1, the old serving), none BATCHED (8), prompted
    BATCHED (8) — same goals, so batched-vs-unbatched is an aggregate-equivalence
    slice with a measured per-item divergence rate and a MEASURED speedup (never
    byte-identity: bf16 batching nondeterminism is known). Returns a sanitised
    dict (numbers only, no prompt/response text)."""
    import json as _json

    from sisteer.eval.attacks import get_attack, get_heldout_spec
    from sisteer.eval.attacks.heldout_common import write_family_manifest
    from sisteer.eval.jailbreak import build_defense
    from sisteer.eval import sweep

    spec = get_heldout_spec("artprompt")
    run_fam = get_attack("artprompt")
    results = {}
    arms = (("none_unbatched", "none", 1), ("none_batched", "none", 8),
            ("prompted_batched", "prompted", 8))
    for label, mode, bs in arms:
        defense = build_defense(model, tok, mode=mode, positions="all",
                                batch_size=bs)
        base_dir = out_root / "heldout_smoke" / label
        man_path = base_dir / "artprompt.manifest.json"
        # the real held-out leg preflight: manifest with attack_regime (fingerprint
        # absent on this node -> leakage_check.skipped=True, recorded loudly).
        manifest = write_family_manifest(
            spec, goals, manifest_path=man_path, fingerprint=None,
            object_class="baseline", cell=mode, arm=None,
            seed=0, attack_regime=sweep.attack_regime_of_leg("held_out"))
        t0 = time.time()
        res = run_fam(defense, goals, out_dir=base_dir / "run",
                      max_new_tokens=max_new_tokens, seed=0)
        dt = time.time() - t0
        n_prompts = sum(r["n_variants"] for r in res["per_goal"])
        results[label] = {
            "mode": mode,
            "batch_size": res.get("batch_size"),
            "n_batches": res.get("n_batches"),
            "n_goals": res["n_goals_run"],
            "n_prompts_generated": n_prompts,
            "n_target_forwards": res["n_target_forwards_total"],
            "wall_s": round(dt, 2),
            "per_prompt_s": round(dt / max(n_prompts, 1), 3),
            "max_new_tokens": max_new_tokens,
            "manifest_attack_regime": manifest.get("attack_regime"),
            "manifest_leakage_skipped": manifest.get("leakage_check", {}).get("skipped"),
            "_responses_path": res["responses_path"],  # local use, stripped below
        }

    # EQUIVALENCE (same goals, both paths): structural equality + per-item
    # divergence rate + aggregate response-length stats. Aggregate agreement is
    # the bar; byte-identity is known-false under bf16 batching.
    def _rows(path):
        return [_json.loads(l) for l in open(path) if l.strip()]

    u = _rows(results["none_unbatched"].pop("_responses_path"))
    b = _rows(results["none_batched"].pop("_responses_path"))
    results["prompted_batched"].pop("_responses_path", None)
    keys_u = [(r["goal_id"], r["sample_idx"]) for r in u]
    keys_b = [(r["goal_id"], r["sample_idx"]) for r in b]
    structural_equal = keys_u == keys_b
    n = min(len(u), len(b))
    diverged = sum(1 for a, c in zip(u, b) if a["response"] != c["response"])
    mean_u = sum(len(r["response"]) for r in u) / max(len(u), 1)
    mean_b = sum(len(r["response"]) for r in b) / max(len(b), 1)
    speedup = (results["none_unbatched"]["per_prompt_s"]
               / max(results["none_batched"]["per_prompt_s"], 1e-9))
    results["equivalence"] = {
        "structural_equal": bool(structural_equal),
        "n_items": n,
        "n_diverged": diverged,
        "per_item_divergence_rate": round(diverged / max(n, 1), 3),
        "mean_response_chars_unbatched": round(mean_u, 1),
        "mean_response_chars_batched": round(mean_b, 1),
        "aggregate_length_ratio": round(mean_b / max(mean_u, 1e-9), 3),
        "measured_speedup_batched_over_unbatched": round(speedup, 2),
    }
    return results


def _batched_mask_probe(model, tok, layer: int, scale: float) -> dict:
    """THE BATCHING HAZARD on the REAL model: a mixed-length LEFT-PADDED batch
    under a response-mode serve must steer EXACTLY each row's assistant header
    (its last H columns — right-alignment makes the scalar boundary per-row
    exact) and never the padding (where a wrongly-per-row-unpadded boundary
    would silently land the mask). Confirms what the CPU test pinned
    structurally (tests/test_phase_probe_gpu.py), on the serving model."""
    import torch

    from sisteer.generate import encode_kwargs
    from sisteer.eval.jailbreak import response_header_len
    from sisteer.methods import VectorObject
    from sisteer.models import get_decoder_layers
    from sisteer.vectors import SteeringVector

    H = response_header_len(tok)
    d_model = model.config.hidden_size
    g = torch.Generator().manual_seed(0)
    v = torch.randn(d_model, generator=g)
    v = v / v.norm()
    sv = SteeringVector(vectors={layer: v}, method="synthetic_probe",
                        model_name="", instruction_id="")
    obj = VectorObject(sv, positions="response", skip_bos=True,
                       response_header_len=H)

    def render(user_text):
        return tok.apply_chat_template(
            [{"role": "user", "content": user_text}], tokenize=False,
            add_generation_prompt=True)

    rendered = [render("Hi"), render(
        "Please explain, in a few sentences, how rainbows form after a storm")]
    prev = tok.padding_side
    tok.padding_side = "left"
    try:
        enc = tok(rendered, return_tensors="pt", padding=True,
                  **encode_kwargs(tok, rendered[0]))
    finally:
        tok.padding_side = prev
    device = next(model.parameters()).device
    enc = {k: t.to(device) for k, t in enc.items()}
    W = enc["input_ids"].shape[1]
    real_lens = enc["attention_mask"].sum(dim=1).tolist()

    def final_hidden(applied):
        cap = []
        hook = (lambda m, i, o: cap.append(
            (o[0] if isinstance(o, tuple) else o).detach().float().cpu()))
        with torch.no_grad():
            if applied:
                with obj.apply(model, scale=scale) as h:
                    hk = get_decoder_layers(model)[-1].register_forward_hook(hook)
                    h.reset(prompt_len=W)
                    model(**enc)
                    hk.remove()
            else:
                hk = get_decoder_layers(model)[-1].register_forward_hook(hook)
                model(**enc)
                hk.remove()
        return cap[0]

    from sisteer.eval.phase_probe import classify_batched_row_mask
    diff = (final_hidden(True) - final_hidden(False)).abs().amax(dim=-1)
    header_cols = set(range(W - H, W))
    rows = []
    ok = True
    for r, real_len in enumerate(real_lens):
        pad_cols = set(range(W - int(real_len)))
        # MAGNITUDE-AWARE (finding, this job on H200 bf16): a binary atol
        # conflates real steering (~scale) with bf16 batch non-invariance noise
        # (~1e-3) on causally-independent columns; classify by magnitude and
        # RECORD the separating diffs so a too-close call is visible.
        cls = classify_batched_row_mask(diff[r].tolist(), header_cols=header_cols,
                                        pad_cols=pad_cols)
        cls["real_len"] = int(real_len)
        rows.append(cls)
        ok = ok and cls["header_steered_exactly"] and not cls["mask_in_padding"]
    return {"padded_width": W, "header_len_H": H,
            "mixed_lengths": [int(x) for x in real_lens],
            "rows": rows, "PASS": bool(ok)}


def _assert_unprovenanced_object_refuses(out_root: Path) -> dict:
    """A stage-3 object with NO fit-time provenance record must REFUSE
    (default-deny) — correct, validated rather than a surprise. A baseline is
    exempt (handled by the sweep before this resolver)."""
    from sisteer.eval import sweep
    fake = out_root / "fake_object" / "object.safetensors"
    fake.parent.mkdir(parents=True, exist_ok=True)
    fake.write_bytes(b"not a real artifact, and no fit_provenance.json beside it")
    refused = False
    reason = ""
    try:
        sweep.resolve_training_mirror("safety_refusal__vector", fake)
    except sweep.UnrecordedMirrorError as e:
        refused = True
        reason = "UnrecordedMirrorError"
    except sweep.Stage2ObsoleteError as e:
        refused = True
        reason = "Stage2ObsoleteError"
    return {"unprovenanced_object_refused": refused, "refusal": reason}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="NousResearch/Meta-Llama-3.1-8B-Instruct")
    ap.add_argument("--revision", default="")
    ap.add_argument("--layer", type=int, default=15)
    ap.add_argument("--scale", type=float, default=8.0)
    ap.add_argument("--n-goals", type=int, default=25)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--out-root", required=True)
    args = ap.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    from sisteer.models import load_model_and_tokenizer

    t_load = time.time()
    model, tok = load_model_and_tokenizer(
        args.model, revision=(args.revision or None), dtype="bfloat16",
        attn_implementation="sdpa")
    model.eval()
    load_s = round(time.time() - t_load, 1)
    print(f"[validate] model loaded in {load_s}s: {args.model}")

    summary = {"model": args.model, "layer": args.layer, "load_s": load_s}

    print("[validate] PART 2a: probe forward-pair ...")
    summary["probe"] = _validate_probe(model, tok, args.layer, args.scale)
    print(f"[validate] probe: {json.dumps(summary['probe'], indent=2)}")

    print("[validate] refusal check: unprovenanced object ...")
    summary["refusal"] = _assert_unprovenanced_object_refuses(out_root)
    print(f"[validate] refusal: {summary['refusal']}")

    print("[validate] batched mixed-length mask probe ...")
    summary["batched_mask_probe"] = _batched_mask_probe(model, tok, args.layer,
                                                        args.scale)
    print(f"[validate] batched probe: {json.dumps(summary['batched_mask_probe'], indent=2)}")

    print(f"[validate] PART 2b/2c: held-out smoke (artprompt, n={args.n_goals}) ...")
    goals = _synthetic_goals(args.n_goals)
    summary["heldout_smoke"] = _smoke_heldout(model, tok, goals, out_root,
                                              args.max_new_tokens)
    print(f"[validate] heldout: {json.dumps(summary['heldout_smoke'], indent=2)}")

    smoke = summary["heldout_smoke"]
    summary["ALL_PASS"] = bool(
        summary["probe"]["PASS"]
        and summary["batched_mask_probe"]["PASS"]
        and summary["refusal"]["unprovenanced_object_refused"]
        and all(v.get("n_prompts_generated", 0) > 0
                for k, v in smoke.items() if k != "equivalence")
        and smoke["equivalence"]["structural_equal"])
    # sanitised summary (numbers/labels/bools only) — safe to mirror off-host.
    (out_root / "validation_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[validate] wrote {out_root / 'validation_summary.json'}")
    print(f"[validate] ALL_PASS={summary['ALL_PASS']}")
    if not summary["ALL_PASS"]:
        raise SystemExit("[validate] one or more checks FAILED — see summary")


if __name__ == "__main__":
    main()
