"""The F.9-AUROC layer/p screen for the multi-layer AlphaSteer arm — CLI
(the library is ``sisteer.extraction.alphasteer_screen``).

One job per model: the capture pass is the expensive part and the screen,
the fit and the fit-side rho all reuse it, so this script writes the
activations of EVERY block to an acts dir the fit driver
(``scripts/fit_class_b.py``, alphasteer exemption) reads back::

    python scripts/alphasteer_screen.py --config configs/<model>.yaml [--resume]
        [--model-key KEY] [--out screen_f9.json] [--acts-dir DIR]
        [--split-out split_v1.json] [--layers all|band] [--device cuda]
        [--batch-size 32] [--wandb-project P] [--force]
        [--if-configured] [--arms a,b,c]

The config is a two-SI-style yaml whose ``class_b`` block pins the authors'
lists and whose ``class_b.alphasteer`` block carries the F.9 rule
(``fit_class_b.alphasteer_layer_rule``); every artifact path defaults from
that block (``screen_path``, ``acts_dir``, ``split_path``). ``--if-configured``
makes a config WITHOUT the block a no-op exit 0 (so ``fit_class_b.sbatch``
can run the screen step unconditionally), as does ``--arms`` without
``alphasteer`` in it.

Produces (all fingerprinted: tmp -> os.replace, sha256, code_sha,
``.COMPLETE`` marker strictly after the payload):

- ``split_v1.json``   — the content-grouped split (identities, realised
                        per-source counts, ``content_disjoint_verified``).
- ``acts/{benign,harmful}/L{l:03d}.pt`` + ``capture_done.json`` — every block,
                        site ``pre``, last valid token, fp32 CPU.
- ``screen_rows.jsonl`` — one append-only row per solved (layer, p), the
                        resume checkpoint.
- ``screen_f9.json``  — the curve, the per-layer table (AUROC, p*, and
                        leakage as a reported DIAGNOSTIC — eligibility is
                        AUROC only), eligible counts at 0.90/0.95/0.98,
                        the chosen set.
- ``figs/auroc_vs_depth.png`` (+ W&B when ``--wandb-project`` is given).

Resume: every reusable checkpoint is BOUND to its inputs — every
input that can change its bytes, enumerated once in
``alphasteer_screen.CAPTURE_INPUTS``. A capture side is reused iff its
sentinel's binding (model + RESOLVED revision, the dtype and attention
implementation the model was actually loaded with, the device family, site,
list sha + row count, batch width, tokenization, code sha) equals the one
now requested, and a sentinel under any other binding REFUSES — never
recaptured over, never accepted (a partial capture, sentinel absent, re-runs
whole). ``screen_rows.jsonl`` opens with a header binding it to BOTH
capture bindings (so a row inherits every capture input), the split sha,
layers, grid, lambda and code; ``--resume`` replays it only under the same
header (a torn final record is truncated and its cell recomputed; an
unparseable interior record refuses), and refuses under any other. A
COMPLETE ``screen_f9.json`` is "nothing to do" only under the same inputs
(its recorded ``inputs_binding``); under any other it refuses, and
``--force`` redoes it. Spec A5's stop (fewer than three in-band layers
reaching AUROC 0.90) writes ``screen_f9.A5_STOP.json`` beside the curve and
exits 3 — a human decides.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # sibling scripts

import fit_class_b as fcb  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def _log(msg: str) -> None:
    print(f"[alphasteer_screen] {msg}", flush=True)


def _auroc_figure(rows: list[dict], selection: dict, *, model_key: str,
                  rule: dict, out_png: Path) -> Path | None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001 — a figure must never fail the screen
        _log(f"matplotlib unavailable ({type(e).__name__}); no figure")
        return None
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ps = sorted({float(r["p"]) for r in rows})
    for p in ps:
        pts = sorted((r for r in rows if float(r["p"]) == p), key=lambda r: r["layer"])
        ax.plot([r["depth"] for r in pts], [r["auroc"] for r in pts],
                marker="o", ms=3, lw=1.2, label=f"p={p:g}")
    lo, hi = rule["depth_band"]
    ax.axvspan(lo, hi, color="0.85", alpha=0.5, lw=0, label="depth band")
    ax.axhline(rule["auroc_threshold"], color="k", ls="--", lw=0.8,
               label=f"AUROC {rule['auroc_threshold']}")
    depth_of = {r["layer"]: r["depth"] for r in rows}   # not max(layer)+1: under
    for l in selection["chosen"]["layers"]:               # --layers band that is
        ax.axvline(depth_of[l], color="C3", alpha=0.25, lw=0.8)   # not n_blocks
    ax.set_xlabel("depth (layer / n_blocks)")
    ax.set_ylabel("held-out AUROC of ||h @ M||")
    ax.set_title(f"F.9 screen: {model_key} (chosen layers marked)")
    ax.set_ylim(0.4, 1.02)
    ax.legend(fontsize=8, ncol=3, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return out_png


def run_screen(cfg: dict, rule: dict, *, model_key: str, out: Path,
               acts_dir: Path, split_out: Path, layers_mode: str = "all",
               resume: bool = False, force: bool = False, device=None,
               batch_size: int | None = None, wandb_project: str = "",
               model_and_tok=None, data=None, recovery_fn=None) -> int:
    """The whole screen for one model. ``model_and_tok`` / ``data`` are
    injectable so a CPU test can run this end to end on the tiny fixture.
    ``recovery_fn(rows, selection) -> dict`` (plan Phase D, the Llama
    validation) supplies the artifact's ``recovery`` block from the solved
    curve and the selection; the screen itself never computes one, and a
    complete artifact found on resume is returned as it is."""
    import torch

    from sisteer.extraction import alphasteer_screen as scr
    from sisteer.models import get_decoder_layers, hidden_size, resolved_mamba_kernels

    out = Path(out)
    acts_dir = Path(acts_dir)
    split_out = Path(split_out)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = int(batch_size or rule["capture_batch_size"])

    if model_and_tok is None:
        model, tok = fcb.load_class_b_model(cfg)
    else:
        model, tok = model_and_tok
    if data is None:
        data = fcb.build_class_b_data(cfg, tok)
    content_meta = data.get("content_meta", {}) or {}
    list_sha = {"benign": content_meta.get("benign_sha256"),
                "harmful": content_meta.get("harmful_sha256")}
    n_blocks = len(get_decoder_layers(model))
    d = hidden_size(model)
    site = rule["site"]
    code_sha = scr.repo_code_sha()
    identity = scr.model_load_identity(model, name=cfg["model"],
                                       cfg_revision=cfg.get("model_revision"))
    layers = (scr.band_layers(n_blocks, tuple(rule["depth_band"]))
              if layers_mode == "band" else list(range(n_blocks)))
    _log(f"model {cfg['model']} rev {identity['model_revision']} "
         f"({identity['dtype']}, {identity['attn_implementation']}, "
         f"{identity['device']}): {n_blocks} blocks, d={d}; lists "
         f"{len(data['benign_prompts'])} benign / {len(data['harmful_prompts'])} "
         f"harmful; site={site}; device={device}; code {code_sha[:12]}")
    # Mamba-hybrid targets: the loader already REFUSED the pure-torch path
    # (models.assert_fused_mamba_kernels); name the kernels it resolved so
    # the log carries the evidence, not just the absence of a refusal.
    kern = resolved_mamba_kernels(model)
    if kern is not None:
        _log("mamba kernels (fused, resolved at import): "
             + ", ".join(f"{n} <- {m}" for n, m in sorted(kern.items())))

    # 1. the content-grouped split (plan B.1) — computed here, WRITTEN below
    #    after the complete-artifact check, so a no-op leaves split_v1.json
    #    exactly as the artifact recorded it
    split = fcb.compute_content_grouped_split(cfg, rule, data)

    # The bindings every checkpoint of this run is held to: one capture
    # binding per side (every input the captured bytes are a function of —
    # model, resolved revision, precision, attention kernel, device, site,
    # list, width, tokenization, code; alphasteer_screen.CAPTURE_INPUTS), and
    # the row binding that inherits both plus the split and solver settings.
    cap_binding = {
        side: scr.capture_binding(
            **identity, site=site, side=side, list_sha256=list_sha[side],
            n_rows=len(data[f"{side}_prompts"]), batch_size=batch_size,
            code_sha=code_sha)
        for side in ("benign", "harmful")}
    binding = scr.screen_binding(
        capture=cap_binding, split_sha256=split["split_sha256"],
        n_blocks=n_blocks, hidden_dim=d, layers=layers, p_grid=rule["p_grid"],
        lambda_reg=rule["lambda_reg"], depth_band=rule["depth_band"],
        code_sha=code_sha)

    # A COMPLETE artifact is "nothing to do" only under the SAME inputs. Under
    # any other it REFUSES: the sbatch runs this step before every fit, and
    # "complete, nothing to do" on a screen made under another width, precision
    # or code sha would hand the fit a screen it cannot reproduce. The requested
    # inputs are RESOLVED values, so the check needs the loaded model — a no-op
    # costs a model load; a wrong no-op costs the experiment.
    if out.is_file() and out.with_name(out.name + ".COMPLETE").is_file() and not force:
        payload = scr.read_fingerprinted_json(out)
        got = payload.get("inputs_binding")
        if got != binding:
            raise scr.CheckpointMismatch(
                f"{out}: the complete screen there was made under different "
                f"inputs than the ones now requested — "
                f"{scr.binding_diff(binding, got)}. Refusing to stand on it; "
                "--force redoes the screen under these inputs, or remove it "
                "deliberately if it is obsolete.")
        _log(f"screen artifact already complete at {out} (sha "
             f"{payload['sha256'][:12]}, inputs {payload['inputs_sha256'][:12]} "
             f"matched, chosen {payload['chosen']['layers']}) -- nothing to do "
             "(--force to redo)")
        return 0
    out.parent.mkdir(parents=True, exist_ok=True)
    split_payload = scr.write_fingerprinted_json(split_out, split)
    _log(f"split: fit {split['n_fit_benign']}/{split['n_fit_harmful']}, held-out "
         f"{split['n_heldout_benign']}/{split['n_heldout_harmful']} "
         f"(benign/harmful, REALISED), content_disjoint="
         f"{split['content_disjoint_verified']} -> {split_out}")

    # 2. capture every block, once per side (plan B.2), through the ONE
    #    capture seam the fit driver also uses; a side is reused iff its
    #    sentinel's binding matches, refused under any other binding
    acts, capture_info = {}, {"add_special_tokens": scr.CAPTURE_ADD_SPECIAL_TOKENS,
                              "ids_match_serving_render": True}
    for side in ("benign", "harmful"):
        acts[side], capture_info[side] = scr.capture_or_load_side(
            model, tok, data[f"{side}_prompts"], side=side, acts_dir=acts_dir,
            binding=cap_binding[side], verify_sample=int(rule["verify_sample"]),
            layers=layers, log_fn=_log)

    # 3. the screen loop (plan B.3/B.4), checkpointed per (layer, p) under the
    #    header binding above; --resume replays only a matching checkpoint,
    #    --force starts a fresh one
    ckpt = scr.RowCheckpoint(out.parent / "screen_rows.jsonl", binding,
                             resume=resume and not force, log_fn=_log)
    if ckpt.rows:
        _log(f"resuming: {len(ckpt.rows)} (layer, p) rows already solved in {ckpt.path}")
    all_b, all_h = acts["benign"], acts["harmful"]
    sides = split["sides"]
    bi = torch.tensor(sides["benign"]["fit_idx"], dtype=torch.long)
    bh = torch.tensor(sides["benign"]["heldout_idx"], dtype=torch.long)
    hi = torch.tensor(sides["harmful"]["fit_idx"], dtype=torch.long)
    hh = torch.tensor(sides["harmful"]["heldout_idx"], dtype=torch.long)
    t0 = time.time()
    rows = scr.screen_layers_and_p(
        benign_fit={l: all_b[l][bi] for l in layers},
        harmful_fit={l: all_h[l][hi] for l in layers},
        benign_held={l: all_b[l][bh] for l in layers},
        harmful_held={l: all_h[l][hh] for l in layers},
        layers=layers, n_blocks=n_blocks, p_grid=rule["p_grid"],
        lambda_reg=rule["lambda_reg"], depth_band=tuple(rule["depth_band"]),
        device=device, existing_rows=ckpt.rows, on_row=ckpt.append,
        log_fn=lambda m: _log(" ".join(f"{k}={v:.4f}" if isinstance(v, float)
                                       else f"{k}={v}" for k, v in m.items())))
    _log(f"screen: {len(rows)} (layer, p) cells in {time.time() - t0:.0f}s")

    # 4. selection (plan B.5 / spec A5)
    try:
        selection = scr.select_layers(
            rows, n_blocks=n_blocks, depth_band=tuple(rule["depth_band"]),
            auroc_threshold=rule["auroc_threshold"], p_grid=rule["p_grid"])
    except scr.A5StopError as e:
        stop_path = out.with_name(out.stem + ".A5_STOP.json")
        scr.write_fingerprinted_json(stop_path, code_sha=code_sha, payload={
            "schema": "alphasteer_screen_a5_stop/1", "model_key": model_key,
            "model": cfg["model"], "reason": str(e), "curve": rows,
            "rule": scr.rule_block(depth_band=tuple(rule["depth_band"]),
                                   p_grid=rule["p_grid"],
                                   auroc_threshold=rule["auroc_threshold"],
                                   lambda_reg=rule["lambda_reg"])})
        _log(f"A5 STOP: {e}\n  curve written to {stop_path}; ask the user.")
        return 3
    chosen = selection["chosen"]
    _log(f"chosen layers {chosen['layers']} ratios {chosen['nullspace_ratios']} "
         f"eligible_counts {selection['eligible_counts']}"
         + (f" -- {chosen['fallback_a5']['FLAG']}" if chosen["fallback_a5"] else ""))

    # 5. the artifact (plan 5.1)
    cb = fcb._class_b_cfg(cfg)
    lists = {"benign_path": cb.get("benign_prompts_path"),
             "benign_sha256": list_sha["benign"],
             "benign_n": len(data["benign_prompts"]),
             "harmful_path": cb.get("harmful_prompts_path"),
             "harmful_sha256": list_sha["harmful"],
             "harmful_n": len(data["harmful_prompts"]),
             "slices_source": split.get("slices_source")}
    recovery = recovery_fn(rows, selection) if recovery_fn is not None else None
    payload = scr.build_screen_payload(
        model_key=model_key, model=cfg["model"],
        model_revision=cfg.get("model_revision"), n_blocks=n_blocks,
        hidden_dim=d, site=site,
        rule=scr.rule_block(depth_band=tuple(rule["depth_band"]),
                            p_grid=rule["p_grid"],
                            auroc_threshold=rule["auroc_threshold"],
                            lambda_reg=rule["lambda_reg"]),
        split={**scr.split_summary(split_payload), "split_path": str(split_out)},
        lists=lists, rows=rows, selection=selection, capture=capture_info,
        acts_dir=str(acts_dir), binding=binding, recovery=recovery)
    payload["layers_mode"] = layers_mode
    payload["device"] = str(device)
    scr.check_screen_payload(payload)
    final = scr.write_fingerprinted_json(out, payload, code_sha=code_sha)
    _log(f"wrote {out} (sha {final['sha256'][:12]})")

    # 6. figures (plan B: a W&B run per model with the AUROC-vs-depth curves)
    png = _auroc_figure(rows, selection, model_key=model_key, rule=rule,
                        out_png=out.parent / "figs" / "auroc_vs_depth.png")
    if png is not None:
        _log(f"figure -> {png}")
    if wandb_project:
        try:
            import wandb

            run = wandb.init(project=wandb_project, name=f"screen-{model_key}",
                             config={"rule": payload["rule"], "model": cfg["model"],
                                     "model_key": model_key}, reinit=True)
            for r in rows:
                run.log({f"screen/auroc_p{r['p']:g}": r["auroc"],
                         f"screen/leakage_p{r['p']:g}": r["leakage"],
                         "layer": r["layer"], "depth": r["depth"]})
            run.summary["chosen_layers"] = chosen["layers"]
            run.summary["eligible_counts"] = selection["eligible_counts"]
            if png is not None:
                run.log({"screen/auroc_vs_depth": wandb.Image(str(png))})
            run.finish()
        except Exception as e:  # noqa: BLE001 — logging must not fail the screen
            _log(f"W&B logging failed ({type(e).__name__}: {e}); artifact is on disk")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--model-key", default="", help="roster key for the artifact "
                    "(default: class_b.alphasteer.model_key, else a slug of the model id)")
    ap.add_argument("--out", default="", help="screen_f9.json (default: class_b.alphasteer.screen_path)")
    ap.add_argument("--acts-dir", default="", help="default: class_b.alphasteer.acts_dir")
    ap.add_argument("--split-out", default="", help="default: class_b.alphasteer.split_path")
    ap.add_argument("--layers", choices=("all", "band"), default="all",
                    help="screen every block (default; the full-depth curve) or the band only")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--force", action="store_true", help="redo a complete screen")
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--wandb-project", default="")
    ap.add_argument("--if-configured", action="store_true",
                    help="exit 0 doing nothing when the config has no class_b.alphasteer block")
    ap.add_argument("--arms", default="", help="comma list; when given without "
                    "'alphasteer', exit 0 doing nothing")
    args = ap.parse_args(argv)
    cfg = fcb.load_cfg(args.config)
    rule = fcb.alphasteer_layer_rule(cfg)
    if rule is None:
        if args.if_configured:
            _log("config carries no class_b.alphasteer block -- single-layer pin "
                 "in force, no screen to run")
            return 0
        raise SystemExit("[alphasteer_screen] the config has no class_b.alphasteer "
                         "block; nothing to screen (pass --if-configured to no-op)")
    if args.arms and "alphasteer" not in [a.strip() for a in args.arms.split(",")]:
        _log(f"arms {args.arms!r} do not include alphasteer -- no screen to run")
        return 0
    model_key = args.model_key or fcb.alphasteer_model_key(cfg, rule)
    return run_screen(
        cfg, rule, model_key=model_key,
        out=fcb.rule_path(rule, "screen_path", args.out),
        acts_dir=fcb.rule_path(rule, "acts_dir", args.acts_dir),
        split_out=fcb.rule_path(rule, "split_path", args.split_out),
        layers_mode=args.layers, resume=args.resume, force=args.force,
        device=args.device, batch_size=args.batch_size,
        wandb_project=args.wandb_project)


if __name__ == "__main__":
    sys.exit(main())
