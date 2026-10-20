#!/usr/bin/env python
"""The canonical best-of-N for ONE cell of ONE shard, in either mode (sisteer.eval.attacks.bon_climb).

  CLIMB (a 512 store exists)   regenerate the cell's CAP-COHERENT at-cap best-of-N draws at 4,096
                               against the SAME served target: same goal, same augmented variant
                               (re-derived from ``bon_seed(goal, sample, trial)`` and asserted
                               byte-equal to the manifest), same per-draw sampling seed, THE SAME
                               SAMPLER THE STORE RECORDS (``bon_climb.CLIMB_DECODE`` as_drawn: the
                               manifest rows carry it; a legacy store's grandfathered T=1 row by name),
                               same proxy + STACKED lens, the cell's own object at full dose.
                               The ONLY change is ``max_new_tokens`` 512 -> 4,096. Loops are never
                               climbed (a repeating draw repeats further; an accepted residual floor).
  NATIVE (no 512 store)        draw best-of-N natively at 4,096 through the wave's own fan-out
                               (``bon_served.bon_generate_served``), same n_max / sigma / seeds, at the
                               served model's VENDOR PRESET (the registry's <model>|bon|off|vendor_preset
                               row; user ruling 2026-09-17) unless --bon-decode-preset names the
                               grandfathered row to finish a store drawn under it.

WHY: the 512 reading is a FLOOR — an at-cap draw is judged jailbroken ~19x more often than a
terminated one, so a cell with at-cap volume reads low at 512 by construction.

Rows go to a PER-SHARD file (one writer per file, so shards on different nodes never interleave on
MooseFS) and are merged beside the 512 store as ``bon_responses_cap4096.jsonl``. THE 512 STORE IS
NEVER OPENED FOR WRITING.

  python scripts/sdfbase/full22/bon_climb_run.py --config <cfg> --defense <cell> \
      --manifest <climb_manifest.jsonl> --shard k --nshards n --out <file.jsonl> \
      --proxy-url <.../v1> --routing-table <routing_table.json> --max-model-len <L> \
      --max-concurrency <c> [--mode auto|climb|native_4096] [--cap 4096] \
      [--private-root <root>] [--require-lens-mode stacked] [--require-engine lens:stacked|lora_native]
      [--vllm-log <vllm_server.log>] [--bon-decode-preset vendor_preset|harness_default|canonical] [--model-key <key>]

The served model the sampler resolves for is the config's (cross-checked against the routing table), or the launcher's
``--model-key`` when that key re-pins THE SAME CHECKPOINT under its own model_id (the worked example ``qwen35_4b_demo`` on
the paper's Qwen3.5-4B; bon_served.served_model_identity) -- in CLIMB mode the store's own recorded key when none is given,
and a store is never climbed under another model's registry row (its rows record the row id the draws were filed under).
"""
from __future__ import annotations

import argparse
import asyncio
import datetime as _dt
import hashlib
import json
import os
import random
import re
import socket
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
for p in (REPO / "src", REPO / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

#: the engine vocabulary the wave asserts from the AUTHORITATIVE stamps (the
#: shell exports TARGET_ENGINE and passes it as --require-engine): a lens proxy at a reduction mode, or the native
#: LoRA engine. The helper is ``proxy_engine.served_engine_problem``, imported at call time (this module binds no
#: ``sisteer`` name at import: its module-level names are stdlib-only, and the ``sys.path`` bootstrap above is what
#: makes that import resolve); the tuple is ``proxy_engine.ENGINE_EXPECTATIONS`` spelled as a literal for the same
#: reason, pinned equal by test (test_sdfbase_full22_wave).
ENGINE_EXPECTATIONS = ("lens:stacked", "lens:per_row", "lora_native")


def _served_engine_problem(ident, *, expect: str):
    from sisteer.capeval.serving.proxy_engine import served_engine_problem
    return served_engine_problem(ident, expect=expect)


def require_engine_of(require_engine, require_lens_mode: str, *, tag: str) -> str:
    """The engine this run asserts: ``--require-engine`` when given (a ``lens:<mode>`` that disagrees with an
    explicit ``--require-lens-mode`` refuses), else ``lens:<--require-lens-mode>`` — so every existing invocation
    (which names only the lens mode) is byte-identical."""
    if require_engine is None:
        return f"lens:{require_lens_mode}"
    if require_engine not in ENGINE_EXPECTATIONS:
        raise SystemExit(f"{tag} --require-engine {require_engine!r} is not one of {ENGINE_EXPECTATIONS}")
    if require_engine.startswith("lens:") and require_engine != f"lens:{require_lens_mode}":
        raise SystemExit(f"{tag} --require-engine {require_engine} disagrees with --require-lens-mode "
                         f"{require_lens_mode} — name one engine")
    return require_engine

ENGINE = "vllm_served"


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--defense", required=True, help="the TrainTest cell id (explicit_cells entry)")
    ap.add_argument("--mode", choices=["auto", "climb", "native_4096"], default="auto",
                    help="auto (default) = CLIMB when this cell has a 512 store, NATIVE 4,096 when it does "
                         "not (bon_climb.climb_mode). Naming it explicitly refuses the other case rather "
                         "than silently doing the wrong one")
    ap.add_argument("--private-root", default=None,
                    help="the wave's private transcripts root — needed to locate the cell's 512 store (the "
                         "mode decision) and, in NATIVE mode, to write the BoN layout the aggregate reads")
    ap.add_argument("--manifest", default=None, help="the lane's climb manifest (jsonl: cell_id, goal_id, sample_idx, trial, variant, ...); CLIMB mode only")
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--nshards", type=int, required=True)
    ap.add_argument("--out", required=True, help="this shard's output jsonl (one writer per file)")
    ap.add_argument("--proxy-url", required=True)
    ap.add_argument("--routing-table", required=True)
    ap.add_argument("--max-model-len", type=int, default=None)
    ap.add_argument("--max-concurrency", type=int, required=True)
    ap.add_argument("--cap", type=int, default=4096)
    ap.add_argument("--trial", type=int, default=0)
    ap.add_argument("--require-lens-mode", choices=["per_row", "stacked"], default="stacked")
    ap.add_argument("--require-engine", choices=list(ENGINE_EXPECTATIONS), default=None,
                    help="the ENGINE the proxy must attest (native-serving-default): lens:<mode> or lora_native; "
                         "default lens:<--require-lens-mode> (byte-identical for every existing invocation)")
    ap.add_argument("--vllm-log", default=None, help="the standup's vllm_server.log (the generation_config override line is copied into the run record)")
    ap.add_argument("--lane", default=None)
    ap.add_argument("--model-key", default=None,
                    help="OPTIONAL: the served model as the launcher knows it (bon_climb.sbatch passes its KEY). The sampler's model is "
                         "DERIVED from the config's `model` and checked against the routing table's base_model; a key naming another "
                         "model_id of THE SAME CHECKPOINT (the export's hf_repo and pinned hf_revision; the worked example "
                         "qwen35_4b_demo on the paper's Qwen/Qwen3.5-4B) is the identity the rows are filed under; a different "
                         "checkpoint refuses. CLIMB mode reads the store's recorded key when none is given, and refuses a key the "
                         "store was not drawn under")
    ap.add_argument("--bon-decode-preset", choices=("vendor_preset", "harness_default", "canonical"), default="vendor_preset",
                    help="NATIVE mode's sampler (a native pass is a NEW store): `vendor_preset` (the default; the served "
                         "model's registry row <model>|bon|off|vendor_preset, user ruling 2026-09-17), `canonical` (the public "
                         "recipe's declaration: the registry label whose sampler reproduces the canonical best-of-N instrument the "
                         "registry's planner binds for the served model via --require-engine's path -- bon_served.canonical_bon_label; "
                         "refused by name when no label does) or `harness_default` "
                         "(the superseded grandfathered T=1 row, by name, to finish a native store drawn under it). CLIMB "
                         "mode ignores it: a climb re-draws at the sampler its 512 store records (the manifest rows carry it)")
    return ap


def _resume_key(rec: dict) -> tuple:
    return (rec["goal_id"], int(rec["sample_idx"]), int(rec.get("trial", 0)))


def load_covered(path: Path) -> set:
    covered = set()
    if not path.exists():
        return covered
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            covered.add(_resume_key(json.loads(line)))
        except (json.JSONDecodeError, KeyError):
            continue  # torn final line
    return covered


def repair_torn_tail(path: Path) -> None:
    if not path.exists():
        return
    data = path.read_bytes()
    if not data or data.endswith(b"\n"):
        return
    cut = data.rfind(b"\n")
    path.write_bytes(data[: cut + 1] if cut >= 0 else b"")


def generation_config_for(base_model: str, revision: str | None) -> tuple[dict | None, str | None]:
    """The model's shipped generation_config.json, read from the OFFLINE HF cache (the same file vLLM reads
    to override its sampling defaults). ``(dict | None, path | None)``."""
    hf_home = os.environ.get("HF_HOME")
    if not hf_home:
        return None, None
    d = Path(hf_home) / "hub" / f"models--{base_model.replace('/', '--')}" / "snapshots"
    cands = []
    if revision and (d / revision).is_dir():
        cands.append(d / revision)
    elif d.is_dir():
        cands.extend(sorted(d.iterdir()))
    for c in cands:
        p = c / "generation_config.json"
        if p.is_file():
            return json.loads(p.read_text()), str(p)
    return None, (str(cands[0]) if cands else None)


def served_model_of(cfg: dict, table: dict, model_key=None, *, tag: str) -> str:
    """THE SERVED MODEL the sampler resolves for -- ``bon_served.served_model_identity`` over the config's ``model`` (the run's
    declaration) and ``model_revision``, the routing table's ``base_model`` / ``base_revision``, and the optional ``model_key``: the
    config's spelling unchanged when the key is absent or names the same model_id; the KEY's model_id when it names another model_id
    of THE SAME CHECKPOINT (the worked example's key on the paper's checkpoint); a different checkpoint, on either side, refuses (a
    preset read for one model must not ride a serve of another)."""
    from sisteer.eval.attacks import bon_served as bs
    try:
        return bs.served_model_identity(cfg.get("model"), table.get("base_model"), model_key,
                                        model_revision=cfg.get("model_revision"), base_revision=table.get("base_revision"))
    except bs.BonServedError as e:
        raise SystemExit(f"{tag} {e}") from e


def climb_model_key(model_key, store_dir, *, tag: str):
    """CLIMB mode's model key: ``--model-key`` when given, else the key the 512 store RECORDS (``bon_served_run.json``
    ``model_key`` -- the store driver's own declaration, the authoritative record of what identity the draws were filed under);
    None when neither names one (the config's model then stands alone, as before). A given key that names another model than the
    store's refuses: a store is climbed under the identity it was drawn under, never relabelled."""
    rec = None
    if store_dir is not None:
        from sisteer.eval.attacks import bon_climb as bc
        path = Path(store_dir) / bc.BON_STORE_MANIFEST
        if path.is_file():
            try:
                rec = json.loads(path.read_text())
            except ValueError as e:
                raise SystemExit(f"{tag} the store's {bc.BON_STORE_MANIFEST} is not JSON ({e}); refusing to climb a store whose record cannot be read")
    stored = (rec or {}).get("model_key")
    if model_key is None:
        return stored
    if stored is not None:
        from sisteer.registry import decode_policy as DP
        pf = DP.load()
        if pf.model_id(str(model_key)) != pf.model_id(str(stored)):
            raise SystemExit(f"{tag} the 512 store was drawn under --model-key {stored!r} ({pf.model_id(str(stored))}; its "
                             f"{Path(store_dir).name}/bon_served_run.json) but this climb names {model_key!r} ({pf.model_id(str(model_key))}) "
                             "-- a store is climbed under the identity it was drawn under; refusing")
    return model_key


def climb_sampler(rec0, model, *, tag: str):
    """THE SAMPLER a CLIMB re-draws at: the store's, as its manifest rows record it (bon_climb.CLIMB_DECODE ``as_drawn``; user
    ruling 2026-09-17), resolved for ``model``. A legacy row (no ``decode``) names the grandfathered T=1 row (bc.LEGACY_STORE_PRESET)
    by name; a registry label resolves for the served model and must state the recorded numbers AND, when the rows record the
    registry row they were filed under (``decode_policy_id``), THAT row -- a label that resolves another model's row (the paper's id
    on the worked example's store) refuses, so a climb never files a row its store did not draw; an explicit block (``anchor_match``)
    is the caller's, labelled by its name."""
    from sisteer.eval.attacks import bon_climb as bc
    from sisteer.eval.attacks import bon_served as bs
    if rec0.get("decode") is None:
        label = rec0.get("bon_decode_preset", bc.LEGACY_STORE_PRESET)
        if label != bc.LEGACY_STORE_PRESET:
            raise SystemExit(f"{tag} manifest rows record no decode but the label {label!r}: a legacy store is "
                             f"climbed under {bc.LEGACY_STORE_PRESET!r} only")
        return bs.resolve_bon_sampler(None, bs.BON_HARNESS_DEFAULT, model)
    label = rec0["bon_decode_preset"]
    if label not in bs.BON_REGISTRY_LABELS:
        return bs.resolve_bon_sampler(dict(rec0["decode"]), label, None)
    sampler = bs.resolve_bon_sampler(None, label, model)
    if dict(sampler.decode) != dict(rec0["decode"]):
        raise SystemExit(f"{tag} the store's rows record {label} = {rec0['decode']} but the registry's "
                         f"{sampler.policy_id} says {dict(sampler.decode)} for {model} — the store was drawn "
                         "under a row the export no longer states; refusing to climb it under another")
    recorded = rec0.get("decode_policy_id")
    if recorded and sampler.policy_id and str(recorded) != str(sampler.policy_id):
        raise SystemExit(f"{tag} the store's rows record registry row {recorded!r} but {label} resolves {sampler.policy_id!r} for "
                         f"{model!r} — a store is climbed under its own model's row (name the store's --model-key); refusing")
    return sampler


def vllm_override_line(log: str | None) -> str | None:
    if not log or not Path(log).is_file():
        return None
    pat = re.compile(r"Default vLLM sampling parameters have been overridden.*")
    for line in Path(log).read_text(errors="replace").splitlines():
        m = pat.search(line)
        if m:
            return m.group(0)
    return None



def _run_native(args, cfg, jr, tsr, bc, store_dir) -> int:
    """NATIVE 4,096 best-of-N for a cell with NO 512 store (the Llama lane): the wave's own served
    fan-out at the climb cap, into the BoN layout the aggregate reads. Its result is NOT an
    escalation and must never be reported as "0 escalated, clean" — the run record says
    ``mode: native_4096`` and ``escalation.status: not_applicable_native_4096``."""
    import datetime as _dt
    import json as _json

    from sisteer.capeval.serving.proxy import ProxyClientEngine, load_routing_table
    from sisteer.eval import request_evidence_client as _rev
    from sisteer.eval.attacks import bon_served as bs
    from sisteer.eval.attacks.served_target import probe_fn, route_for
    from sisteer.eval.defense_cell import si_of

    if not args.private_root:
        raise SystemExit("[bon-climb] native mode writes the BoN layout: pass --private-root")
    bcfg = cfg["bon"]
    goals = tsr._eval_goals(cfg, confirmatory=True, attack="bon")
    run_dir = Path(jr._bon_defense_dir(cfg, jr._private_root(cfg, args.private_root), args.defense))
    table = load_routing_table(Path(args.routing_table))
    routes = list(table["routes"])
    route = route_for(args.defense, routes)
    client = ProxyClientEngine(args.proxy_url, identity=route.serving_identity)
    client.load(require_virtual_models=[route.virtual_model])
    # request-evidence (F-S): the unlabelled sends through this client are the steering-bites PROBES (`probe/bites`,
    # as build_served_target / bon_served_run declare them); the draws ride labelled headers. Off = identity.
    client = _rev.maybe_attach_proxy_client(client, unlabelled_kind=_rev.PROBE_KIND_BITES)
    ident = dict(client.proxy_identity or {})
    require_engine = require_engine_of(args.require_engine, args.require_lens_mode, tag="[bon-climb]")
    engine_problem = _served_engine_problem(ident, expect=require_engine)
    if engine_problem is not None:
        raise SystemExit(f"[bon-climb] the proxy does not attest --require-engine {require_engine}: "
                         f"{engine_problem} — refusing")
    # THE SAMPLER of a native pass (a NEW store): the served model's vendor preset by default, the
    # canonical instrument's label under the recipe's declaration, the grandfathered row only when named to finish a store drawn
    # under it — resolved by bon_served.resolve_bon_sampler from the registry, never a literal here; the model is the config's, or
    # the launcher's key when that key re-pins the same checkpoint (served_model_of)
    model = served_model_of(cfg, table, args.model_key, tag="[bon-climb]")
    label, canonical = args.bon_decode_preset, None
    if label == "canonical":
        try:
            label, canonical = bs.canonical_bon_label(model, "lora_native" if require_engine == "lora_native" else "lens",
                                                      seeds_dir=str(REPO / "configs" / "registry"))
        except bs.BonServedError as e:
            raise SystemExit(f"[bon-climb] --bon-decode-preset canonical: {e}") from e
        print(f"[bon-climb] decode preset: canonical -> {label}: the canonical best-of-N instrument for {model} is "
              f"{canonical['seed_instrument_id']} ({canonical['instrument_id'][:12]}...)", flush=True)
    system = None
    if route.si_prefix_hash is not None:
        from sisteer.eval.jailbreak import resolve_instruction
        system = resolve_instruction(si_of(args.defense)).text
    defense = bs.ServedDefense.over_client(client, virtual_model=route.virtual_model, cell_id=args.defense,
                                           system=system, max_model_len=args.max_model_len)
    steering_arms, steering_probe = (), None
    if bs.is_steered(route):
        from sisteer.capeval.serving import proxy_engine as pe
        twin = pe.scale0_twin(route, routes)      # the PATH-MATCHED zero twin (lens0 / lora0)
        if twin is None:
            raise SystemExit(f"[bon-climb] steered cell {route.cell_id!r} has no scale-0 twin route — refusing")
        steering_arms = (route.cell_id.split("__")[-1],)
        steering_probe = probe_fn(client.chat_completions, route.virtual_model, twin.virtual_model)
    sampler = bs.resolve_bon_sampler(None, label, model)
    print(f"[bon-climb] NATIVE {args.cap}: cell={args.defense} vmodel={route.virtual_model} goals={len(goals)} "
          f"n_max={bcfg['n_max']} concurrency={args.max_concurrency} sampler={sampler.label} {dict(sampler.decode)} "
          f"policy={sampler.policy_id} -> {run_dir}", flush=True)
    jsonl = bs.bon_generate_served(
        defense, goals, out_dir=run_dir, routes=routes, n_max=int(bcfg["n_max"]),
        sigma=float(bcfg["sigma"]), trial=args.trial, max_new_tokens=int(args.cap),
        max_concurrency=int(args.max_concurrency), steering_arms=steering_arms,
        steering_probe=steering_probe, decode_label=label, model=model,
        request_headers_fn=_rev.maybe_request_headers_fn(_rev.LEG_BON))
    rec = {"schema": "bon_climb_run/1", "mode": bc.MODE_NATIVE, "cell_id": args.defense,
           "served_model": model, "model_key": args.model_key, "decode": dict(sampler.decode), "bon_decode_preset": sampler.label,
           "decode_policy_id": sampler.policy_id,
           **({"bon_decode_preset_declared": args.bon_decode_preset, "canonical_instrument": canonical} if canonical is not None else {}),
           "harness_cell_id": route.cell_id, "virtual_model": route.virtual_model,
           "intervention_hash": route.intervention_hash, "serving_identity": route.serving_identity,
           "proxy_identity": ident, "lens_reduction_mode": ident.get("lens_reduction_mode"),
           "require_engine": require_engine,
           "cap": int(args.cap), "base_cap": int(bcfg["max_new_tokens"]), "n_goals": len(goals),
           "n_max": int(bcfg["n_max"]), "max_concurrency": int(args.max_concurrency),
           "max_model_len": args.max_model_len, "config": args.config, "routing_table": args.routing_table,
           "responses": str(jsonl), "n_target_forwards": defense.n_target_forwards,
           "clamp_report": defense.clamp_report(),
           "escalation": bc.escalation_summary(mode=bc.MODE_NATIVE),
           "written_at": _dt.datetime.now(_dt.timezone.utc).isoformat()}
    (run_dir / "bon_climb_run.json").write_text(_json.dumps(rec, indent=2, default=str))
    print(f"[bon-climb] native done; n_target_forwards={defense.n_target_forwards} -> {jsonl}", flush=True)
    return 0


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    import jailbreak_run as jr
    import two_si_run as tsr
    from sisteer.capeval.serving.proxy import ProxyClientEngine, load_routing_table
    from sisteer.capeval.serving import served_window as sw
    from sisteer.eval import request_evidence_client as _rev
    from sisteer.eval.attacks import bon_served as bs
    from sisteer.eval.attacks.bon import augment, bon_seed
    from sisteer.eval.attacks.served_target import probe_fn, route_for
    from sisteer.eval.defense_cell import si_of

    from sisteer.eval.attacks import bon_climb as bc

    cfg = jr._load_cfg(args.config)
    if args.defense not in jr._valid_defense_ids(cfg):
        raise SystemExit(f"[bon-climb] unknown --defense {args.defense!r}")
    bcfg = cfg["bon"]
    base_cap = int(bcfg["max_new_tokens"])
    if int(args.cap) <= base_cap:
        raise SystemExit(f"[bon-climb] --cap {args.cap} is not above the base cap {base_cap}")
    # ---- THE BRANCH (bon_climb): climb a 512 store, or draw natively at 4,096 when there is none.
    store_dir = None
    if args.private_root:
        store_dir = Path(jr._bon_defense_dir(cfg, jr._private_root(cfg, args.private_root), args.defense))
    if args.mode == "auto":
        if store_dir is None:
            raise SystemExit("[bon-climb] --mode auto needs --private-root to see whether a "
                             f"{base_cap} store exists; pass --mode climb / native_4096 to state it")
        try:
            mode = bc.climb_mode(store_dir)
        except bc.BonClimbError as exc:                # a store IN PROGRESS: the same predicate the planner branched on
            raise SystemExit(f"[bon-climb] {exc}")
    else:
        mode = args.mode
        if store_dir is not None:
            try:
                actual = bc.climb_mode(store_dir)
            except bc.BonClimbError as exc:
                raise SystemExit(f"[bon-climb] --mode {mode} but {exc} — refusing rather than reporting the wrong instrument")
            if actual != mode:
                raise SystemExit(f"[bon-climb] --mode {mode} but this cell's store says {actual} "
                                 f"({store_dir}) — refusing rather than reporting the wrong instrument")
    print(f"[bon-climb] cell={args.defense} mode={mode} base_cap={base_cap} cap={args.cap}", flush=True)
    if mode == bc.MODE_NATIVE:
        return _run_native(args, cfg, jr, tsr, bc, store_dir)
    if not args.manifest:
        raise SystemExit("[bon-climb] climb mode needs --manifest (bon_climb_plan.py writes it)")
    sigma = float(bcfg["sigma"])
    goals = tsr._eval_goals(cfg, confirmatory=True, attack="bon")
    by_goal = {g.goal_id: g for g in goals}

    # ---- the shard's work: manifest rows of this cell, class cap_coherent, i % nshards == shard
    rows = []
    with open(args.manifest) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r["cell_id"] != args.defense:
                continue
            if r.get("class", "cap_coherent") != "cap_coherent":
                continue
            if int(r.get("trial", 0)) != int(args.trial):
                continue
            rows.append(r)
    rows.sort(key=lambda r: (r["goal_id"], int(r["sample_idx"])))
    shard_rows = [r for i, r in enumerate(rows) if i % int(args.nshards) == int(args.shard)]
    missing_goals = sorted({r["goal_id"] for r in shard_rows} - set(by_goal))
    if missing_goals:
        raise SystemExit(f"[bon-climb] {len(missing_goals)} manifest goal_ids are not in the config's BoN goal set "
                         f"(e.g. {missing_goals[:3]}) — refusing")
    # ---- variant identity: the manifest's variant must be what the harness derives from (goal, sample, trial)
    work = []  # (goal_id, sample_idx, variant, seed, response_512_sha256)
    n_bad = 0
    for r in shard_rows:
        seed = bon_seed(r["goal_id"], int(r["sample_idx"]), int(args.trial))
        v = augment(by_goal[r["goal_id"]].text, sigma, random.Random(seed))
        if v != r["variant"]:
            n_bad += 1
            continue
        work.append((r["goal_id"], int(r["sample_idx"]), v, seed, r.get("response_512_sha256")))
    if n_bad:
        raise SystemExit(f"[bon-climb] {n_bad}/{len(shard_rows)} manifest variants do NOT match "
                         "augment(goal, sigma, Random(bon_seed(goal, sample, trial))) — the manifest is not the "
                         "harness's own variant set; refusing to generate against a different prompt")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    repair_torn_tail(out)
    covered = load_covered(out)
    todo = [w for w in work if (w[0], w[1], int(args.trial)) not in covered]
    print(f"[bon-climb] cell={args.defense} shard={args.shard}/{args.nshards} manifest_rows={len(rows)} "
          f"shard_rows={len(shard_rows)} covered={len(covered)} todo={len(todo)} cap={args.cap}", flush=True)

    # ---- the served target, exactly as bon_served_run.py binds it
    table = load_routing_table(Path(args.routing_table))
    routes = list(table["routes"])
    route = route_for(args.defense, routes)
    client = ProxyClientEngine(args.proxy_url, identity=route.serving_identity)
    client.load(require_virtual_models=[route.virtual_model])
    # request-evidence (F-S): the unlabelled sends through this client are the steering-bites PROBES (`probe/bites`,
    # as build_served_target / bon_served_run declare them); the draws ride labelled headers. Off = identity.
    client = _rev.maybe_attach_proxy_client(client, unlabelled_kind=_rev.PROBE_KIND_BITES)
    ident = dict(client.proxy_identity or {})
    require_engine = require_engine_of(args.require_engine, args.require_lens_mode, tag="[bon-climb]")
    engine_problem = _served_engine_problem(ident, expect=require_engine)
    if engine_problem is not None:
        raise SystemExit(f"[bon-climb] the proxy does not attest --require-engine {require_engine}: "
                         f"{engine_problem} — refusing")
    system = None
    if route.si_prefix_hash is not None:
        from sisteer.eval.jailbreak import resolve_instruction
        system = resolve_instruction(si_of(args.defense)).text
    defense = bs.ServedDefense.over_client(client, virtual_model=route.virtual_model, cell_id=args.defense,
                                           system=system, max_model_len=args.max_model_len)
    # the two positive controls the wave enforced inside bon_generate_served
    bs.assert_cell_salt_isolation(routes)
    steering_evidence = None
    if bs._is_lens(route):
        by_cell = {r.cell_id: r for r in routes}
        twin = by_cell.get(f"{route.cell_id}__lens0") or by_cell.get("none__lens0")
        if twin is None:
            raise SystemExit(f"[bon-climb] steered cell {route.cell_id!r} has no scale-0 twin route — refusing")
        arm = route.cell_id.split("__")[-1]
        steering_evidence = bs.assert_steering_bites((arm,), probe_fn(client.chat_completions, route.virtual_model, twin.virtual_model))
        steering_evidence = {"twin_virtual_model": twin.virtual_model, "arm": arm, "report": steering_evidence}

    # ---- THE SAMPLER: the store's, as recorded on the manifest rows (bc.CLIMB_DECODE `as_drawn`; user ruling
    # 2026-09-17). Every row of one cell records ONE sampler (bon_climb.store_sampler refused a mixture when the
    # manifest was written; asserted again here); a legacy store's rows record none and name the grandfathered
    # T=1 row (bc.LEGACY_STORE_PRESET), which is resolved from the registry BY NAME for the served model. The
    # request carries exactly that sampler's keys + seed + max_tokens; whatever it omits, vLLM fills from the
    # model's shipped generation_config.json (stamped from the same file vLLM read, for the record).
    recorded = {json.dumps({k: r.get(k) for k in ("decode", "bon_decode_preset")}, sort_keys=True) for r in shard_rows}
    if len(recorded) > 1:
        raise SystemExit(f"[bon-climb] the manifest rows of {args.defense} record {len(recorded)} different samplers "
                         f"{sorted(recorded)} — a store is drawn under ONE sampler; refusing to climb a mixture")
    # the identity the store was drawn under: --model-key, else the store's own record (climb_model_key); the sampler is the store's
    # recorded one resolved for it, and the row id it resolves must be the row the store filed (climb_sampler)
    model_key = climb_model_key(args.model_key, store_dir, tag="[bon-climb]")
    model = served_model_of(cfg, table, model_key, tag="[bon-climb]")
    if shard_rows:
        sampler = climb_sampler(shard_rows[0], model, tag="[bon-climb]")
    else:
        sampler = bs.resolve_bon_sampler(None, bs.BON_VENDOR_PRESET, model)    # an empty shard draws nothing
    gen_cfg, gen_cfg_path = generation_config_for(table.get("base_model") or ident.get("base_model") or "", table.get("base_revision"))
    override_line = vllm_override_line(args.vllm_log)
    sent_keys = [k for k in bs.SAMPLER_KEYS if k in sampler.decode]
    decode = {"sampler": dict(sampler.decode), "bon_decode_preset": sampler.label, "decode_policy_id": sampler.policy_id,
              "climb_decode": bc.CLIMB_DECODE, "served_model": model, "model_key": model_key,
              "max_tokens": int(args.cap), "seed": "bon_seed(goal_id, sample_idx, trial) per draw",
              "request_keys": ["model", "messages", *sent_keys, "max_tokens", "seed"],
              "generation_config_json": gen_cfg, "generation_config_path": gen_cfg_path,
              "vllm_override_line": override_line,
              "note": "the request carries exactly the store's recorded sampler keys (as_drawn) + seed + max_tokens; any "
                      "sampler key it omits, vLLM fills from generation_config.json (absent file -> vLLM defaults: "
                      "top_p 1.0, top_k -1); a sent key beats the file"}
    geom = bs._geometry_stamp()
    job = os.environ.get("SLURM_JOB_ID", "manual"); node = socket.gethostname()
    print(f"[bon-climb] vmodel={route.virtual_model} identity={route.serving_identity} lens_mode={ident.get('lens_reduction_mode')} "
          f"window={args.max_model_len} concurrency={args.max_concurrency} gen_cfg={gen_cfg} -> {out}", flush=True)

    counts = {"n_shard_rows": len(shard_rows), "n_covered_before": len(covered), "n_todo": len(todo), "n_written": 0,
              "n_prompt_over_window": 0, "n_finish_length": 0, "n_finish_stop": 0, "n_finish_other": 0, "n_clamped": 0}
    errors_path = out.with_suffix(".errors.jsonl")

    async def run_all():
        sem = asyncio.Semaphore(max(1, int(args.max_concurrency)))
        lock = asyncio.Lock()
        with out.open("a") as f, errors_path.open("a") as ferr:
            async def one(gid, sidx, variant, seed, sha512):
                async with sem:
                    try:
                        turn = await defense.generate_one_detailed(variant, sampling_seed=seed, max_new_tokens=int(args.cap),
                                                                   decode=dict(sampler.decode))
                    except sw.PromptOverWindow as err:
                        counts["n_prompt_over_window"] += 1
                        erow = json.dumps({"goal_id": gid, "sample_idx": sidx, "trial": int(args.trial), "engine": ENGINE,
                                           "cell_id": args.defense, "cap": int(args.cap), **geom, **err.record()})
                        async with lock:
                            ferr.write(erow + "\n"); ferr.flush(); os.fsync(ferr.fileno())
                        return
                rec = {"goal_id": gid, "sample_idx": sidx, "trial": int(args.trial), "variant": variant, "response": turn.text,
                       "engine": ENGINE, "cell_id": args.defense, "lane": args.lane, "bon_max_concurrency": int(args.max_concurrency),
                       "finish_reason": turn.finish_reason, "cap": int(args.cap), "cap_512": base_cap, "mode": mode, "response_512_sha256": sha512,
                       "sampling_seed": seed, "decode": dict(sampler.decode), "bon_decode_preset": sampler.label,
                       "decode_policy_id": sampler.policy_id, "max_tokens_sent": turn.max_tokens,
                       "prompt_tokens": turn.prompt_tokens, "completion_tokens": turn.completion_tokens,
                       "shard": int(args.shard), "nshards": int(args.nshards), "job_id": job, "node": node, **geom}
                if turn.max_tokens_clamped_from is not None:
                    rec["max_tokens_clamped_from"] = turn.max_tokens_clamped_from
                    rec["prompt_tokens_source"] = turn.prompt_tokens_source
                    counts["n_clamped"] += 1
                if turn.window_mismatch is not None:
                    rec["window_mismatch"] = turn.window_mismatch
                fr = turn.finish_reason
                counts["n_finish_length" if fr == "length" else "n_finish_stop" if fr == "stop" else "n_finish_other"] += 1
                line = json.dumps(rec)
                async with lock:
                    f.write(line + "\n"); f.flush(); os.fsync(f.fileno())
                    counts["n_written"] += 1
                    if counts["n_written"] % 200 == 0:
                        print(f"[bon-climb] {args.defense} shard {args.shard}: {counts['n_written']}/{len(todo)} written "
                              f"(length {counts['n_finish_length']}, stop {counts['n_finish_stop']})", flush=True)
            await asyncio.gather(*(asyncio.create_task(one(*w)) for w in todo))

    started = _dt.datetime.now(_dt.timezone.utc)
    try:
        asyncio.run(run_all())
    finally:
        rec = {"schema": "bon_climb_run/1", "cell_id": args.defense, "lane": args.lane, "harness_cell_id": route.cell_id,
               "virtual_model": route.virtual_model, "intervention_hash": route.intervention_hash,
               "serving_identity": route.serving_identity, "proxy_identity": ident, "lens_reduction_mode": ident.get("lens_reduction_mode"),
               "require_engine": require_engine,
               "steering_bites": steering_evidence, "cell_salt_isolation": bs.cell_salt_isolation_report(routes),
               "decode": decode, "cap": int(args.cap), "cap_512": base_cap, "mode": mode, "trial": int(args.trial),
               "shard": int(args.shard), "nshards": int(args.nshards), "max_concurrency": int(args.max_concurrency),
               "max_model_len": args.max_model_len, "config": args.config, "routing_table": args.routing_table,
               "manifest": args.manifest, "manifest_sha256": hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest(),
               "repo_dir": str(REPO), "repo_sha": os.environ.get("B4K_WAVE_SHA"), "counts": counts,
               "clamp_report": defense.clamp_report(), "n_target_forwards": defense.n_target_forwards,
               "job_id": job, "node": node, "started_at": started.isoformat(),
               "written_at": _dt.datetime.now(_dt.timezone.utc).isoformat(), **geom}
        out.with_suffix(".run.json").write_text(json.dumps(rec, indent=2, default=str))
    final_covered = load_covered(out)
    n_expected = len(shard_rows)
    ok = (len(final_covered) == n_expected and counts["n_prompt_over_window"] == 0)
    print(f"[bon-climb] done cell={args.defense} shard={args.shard}: covered {len(final_covered)}/{n_expected} "
          f"written_now={counts['n_written']} length={counts['n_finish_length']} stop={counts['n_finish_stop']} "
          f"over_window={counts['n_prompt_over_window']} clamped={counts['n_clamped']} -> {'OK' if ok else 'INCOMPLETE'}", flush=True)
    return 0 if ok else 3


if __name__ == "__main__":
    sys.exit(main())
