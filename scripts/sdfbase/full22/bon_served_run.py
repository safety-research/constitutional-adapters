#!/usr/bin/env python
"""full22: best-of-N against the SERVED lens target for ONE cell of an ``explicit_cells`` config,
FANNED OUT (``bon_served.bon_generate_served``) — the async driver the wave needs because the served
target's sequential ``generate()`` at ~50 ms per decode step would take ~30 h per cell for 100 goals x 75.

  python scripts/sdfbase/full22/bon_served_run.py --config <cfg> --defense <cell> --confirmatory \\
      --private-root <root> --proxy-url <…/v1> --routing-table <routing_table.json> \\
      --max-model-len <L> --max-concurrency <S / n_cells> [--require-lens-mode stacked]
      [--require-engine lens:stacked|lens:per_row|lora_native]

Same seams as ``jailbreak_run.py generate --attack bon``: the config's ``bon`` block (n_max, sigma,
max_new_tokens), the confirmatory goal set under ``bon.goal_subset`` (``two_si_run._eval_goals(...,
attack="bon")`` — the 100 JBB goals), the BoN-layout run dir ``jailbreak_run._bon_defense_dir`` (so the
aggregate finds it), resume per (goal, sample, trial). Same POSITIVE CONTROLS as the served attack target:
salt isolation over the FULL route set and, for a lens cell, steering-bites against its scale-0 twin
through the same proxy (both enforced inside ``bon_generate_served``; a steered cell with no probe
refuses) — a natively served LoRA cell steers too and bites against its path-matched twin
(``proxy_engine.scale0_twin``: ``none__lora0``). The proxy's ``petri_proxy`` identity block must attest the
ENGINE this run asserts (``--require-engine``: ``lens:<mode>`` = ``lens_reduction_mode == <mode>``, the
authoritative served stamp, the default derived from ``--require-lens-mode`` so every existing invocation is
byte-identical; ``lora_native`` = ``engine.kind == lora_native``) or the run
refuses before any generation.
Writes ``bon_served_run.json`` beside the responses (cell, vmodel, intervention_hash, proxy identity,
goal count, n_max, concurrency, written_at) next to ``bon_served.json`` (the fan-out's own manifest).

The Qwen base seed replicates — two ADDITIVE, opt-in declarations, both recorded in ``bon_served_run.json``
and in the fan-out's manifest, neither changing a byte of any invocation that omits them:
  * ``--bon-decode-preset anchor_match``: the request states ``temperature 1.0, top_p 1.0, top_k -1, presence_penalty 0.0``
    EXPLICITLY — the Qwen BoN anchor's ENGINE-RESOLVED tuple (it decoded at vLLM's neutral defaults on a standup with no
    server-side override), spelled out because a standup on the wave sha carries the Qwen server override (0.7/0.8/20/1.5)
    and an omitted key would resolve differently from the anchor. The two REGISTRY labels are ``vendor_preset``
    (the default; per served model) and ``harness_default`` (the superseded grandfathered
    T=1 row, by explicit opt-in).
  * ``--decode-seed-policy replicate_v1 --decode-seed-base <int>``: the augmentation keeps the frozen ``bon_seed`` (identical
    variant bytes), the DECODE seed is ``bon.bon_decode_seed_v1(goal, sample, trial, base)`` — one base per replicate view
    (scripts/sdfbase/seedrep/eval_seed.py); rows record ``seed_rule bon_seed_v1+decode:replicate_v1`` + the policy fields
    and the resume validator accepts a row only when BOTH seeds reproduce under the row's own recorded base.
    ``bon_seed_v1`` (the default) is today's behaviour byte for byte and takes no base.

The Qwen seed-replicate native battery (a NEW instrument, never pooled with the T=1.0 rows) —
``--bon-decode-preset vendor``: THE REGISTRY ROW ``<model_id>|bon|off|vendor_preset@v1`` of the served model
(``--model-key`` REQUIRED), resolved at CALL time by EXACT ref + not-superseded through ``target_decode.registry_vendor_row``
(all six sampling keys SENT on every draw, presence_penalty 1.5 included) and recorded as ``decode_policy`` here and in the
fan-out's manifest; an absent / superseded / partial row refuses before the proxy is touched. The battery's launch plan and
the wave sbatch (``F22_BON_DECODE_PRESET=vendor``) name it explicitly; every other invocation is unchanged.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
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

#: the BoN decode presets. A None value is a REGISTRY label (bon_served.BON_REGISTRY_LABELS), resolved PER SERVED MODEL by
#: sisteer.eval.attacks.bon_served.resolve_bon_sampler — `vendor_preset` (the default) and
#: `harness_default` (the superseded grandfathered T=1 row, by explicit opt-in to FINISH a store drawn under it) — never a
#: literal here. A dict value is an EXPLICIT block this driver vouches for, labelled by its name: `anchor_match`
#: (qwen-base-seed-replicates) = the Qwen BoN anchor's engine-resolved neutral tuple sent EXPLICITLY, so a standup carrying
#: a server-side override still decodes as the anchor did. The dict IS the request's sampler keys (bon_served.SAMPLER_KEYS;
#: the seed rides beside them).
#: `vendor` (the Qwen seed-replicate native battery — a NEW instrument, never pooled
#: with the T=1.0 rows) = THE REGISTRY ROW `<model_id>|bon|off|vendor_preset@v1` of the served model (--model-key REQUIRED),
#: resolved at CALL time by exact ref + not-superseded through target_decode.registry_vendor_row (never at import: this module
#: binds no sisteer name at import, and the ref needs the model key), all six sampling keys SENT on every draw, presence_penalty
#: 1.5 included. The table's value is the marker below, never the numbers — the registry is the ONLY authority; the resolved
#: block then rides bon_served.resolve_bon_sampler as an explicit block labelled `vendor`, its registry identity carried beside
#: it as `decode_policy` (run record + manifest).
BON_DECODE_PRESET_VENDOR = "vendor"
BON_DECODE_FROM_REGISTRY = "registry"
#: the ref's second segment for best-of-N (the eval id's registry segment: `qwen35_4b_it|bon|off|vendor_preset@v1`)
BON_VENDOR_POLICY_SEGMENT = "bon"
#: `canonical` (the public recipe's two best-of-N units, 2026-10-09): THE REGISTRY LABEL whose sampler reproduces the CANONICAL best-of-N
#: instrument the registry's planner binds for the served model via this run's engine -- bon_served.canonical_bon_label over the
#: checkout's seeds, the same jobplan.canonical_instrument the job's start-of-job registration binds -- so the run draws exactly the
#: sampler of the instrument it registers under: the paper's Qwen3.5-4B and the worked example (its twin rows) -> `harness_default` (the
#: T=1.0 climb instrument states temperature 1.0; the vendor row sends 0.7), the pipeline-test models -> `vendor_preset` (their vendor
#: rows send the instrument's 1.0). Resolved at launch AFTER the served model is known and BEFORE the proxy is touched; an instrument no
#: label reproduces refuses by name. Rows and the run record carry the RESOLVED label (the climb's as_drawn reads it unchanged) and the
#: record names the declaration and the instrument beside it. The table's value is a marker; nothing is read at import.
BON_DECODE_PRESET_CANONICAL = "canonical"
BON_DECODE_FROM_CANONICAL = "canonical_instrument"
BON_DECODE_PRESETS = {"vendor_preset": None, "harness_default": None,
                      "anchor_match": {"temperature": 1.0, "top_p": 1.0, "top_k": -1, "presence_penalty": 0.0},
                      BON_DECODE_PRESET_VENDOR: BON_DECODE_FROM_REGISTRY,
                      BON_DECODE_PRESET_CANONICAL: BON_DECODE_FROM_CANONICAL}
#: the decode-seed policies (bon.DECODE_SEED_POLICIES, spelled as a literal for the same import-time reason as
#: ENGINE_EXPECTATIONS; pinned equal by tests/test_bon_decode_seed_policy.py)
DECODE_SEED_POLICIES = ("bon_seed_v1", "replicate_v1")


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


def served_model_of(cfg: dict, table: dict, model_key=None, *, tag: str) -> str:
    """THE SERVED MODEL the best-of-N sampler resolves for -- ``bon_served.served_model_identity`` over the config's ``model`` (the
    run's declaration; refused when the config names none or the export does not know it) and ``model_revision``, the routing table's
    ``base_model`` / ``base_revision``, and the optional ``--model-key``: the config's spelling unchanged when the key is absent or
    names the same model_id (every existing invocation byte-identical); the KEY's model_id when the key names another model_id of
    THE SAME CHECKPOINT (the export's hf_repo, and its pinned hf_revision against the config's and the table's wherever both state
    one) -- the worked example ``qwen35_4b_demo`` is the paper's ``Qwen/Qwen3.5-4B @ 851bf6e8`` under its own key, whose rows are
    its own and whose config can only spell the repo (the paper's row owns that spelling); a key of a different checkpoint, or a
    table serving one, refuses: a sampler read for one model never rides a serve of another."""
    from sisteer.eval.attacks import bon_served as bs
    try:
        return bs.served_model_identity(cfg.get("model"), table.get("base_model"), model_key,
                                        model_revision=cfg.get("model_revision"), base_revision=table.get("base_revision"))
    except bs.BonServedError as e:
        raise SystemExit(f"{tag} {e}") from e


def serving_path_of(require_engine: str) -> str:
    """The registry serving path this run's engine assertion names (vocab.SERVING_PATHS): ``lora_native``, else the lens."""
    return "lora_native" if require_engine == "lora_native" else "lens"


def resolve_bon_decode(preset: str, model_key, *, path=None, tag: str = "[bon-served]"):
    """The ``(decode, decode_policy)`` pair a run hands the fan-out. The registry labels ``vendor_preset`` /
    ``harness_default`` (None) and the explicit ``anchor_match`` block are the table above with NO policy record here
    (main's ``bon_served.resolve_bon_sampler`` resolves the labels per served model; every existing run record stays
    byte-identical); ``vendor`` is THE REGISTRY ROW ``<model_id>|bon|off|vendor_preset@v1`` of ``model_key`` —
    ``target_decode.registry_vendor_row``: ``PolicyFile.by_ref`` on the exact ref, ``superseded_by_policy_id is None``
    asserted, all six sampling keys SENT — with the row's identity as the second element (recorded in
    ``bon_served_run.json`` and the fan-out's manifest). REFUSES (SystemExit, before the proxy is touched): ``vendor``
    without ``--model-key``; a row that is absent (``NoPolicy`` — the harness never invents numbers), superseded, or
    not sending all six keys. ``path`` = the export to read (tests hand a fixture; None = ``$SISTEER_DECODE_POLICY`` /
    the committed export)."""
    if preset not in BON_DECODE_PRESETS:
        raise SystemExit(f"{tag} --bon-decode-preset {preset!r} is not one of {tuple(BON_DECODE_PRESETS)}")
    if preset == BON_DECODE_PRESET_CANONICAL:
        return None, None       # the label is the canonical instrument's, resolved in main once the served model is known
    if preset != BON_DECODE_PRESET_VENDOR:
        return BON_DECODE_PRESETS[preset], None
    if not model_key:
        raise SystemExit(f"{tag} --bon-decode-preset vendor requires --model-key: the registry row is keyed by the served "
                         f"model (<model_id>|{BON_VENDOR_POLICY_SEGMENT}|off|vendor_preset@v1) and is never guessed")
    from sisteer.eval.attacks.target_decode import VendorPolicyRefused, registry_vendor_row
    try:
        row = registry_vendor_row(model_key, BON_VENDOR_POLICY_SEGMENT, path=path)
    except VendorPolicyRefused as e:
        raise SystemExit(f"{tag} --bon-decode-preset vendor: {e}") from e
    return dict(row.decode), row.record()


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--defense", required=True, help="the TrainTest cell id (explicit_cells entry)")
    ap.add_argument("--confirmatory", action="store_true")
    ap.add_argument("--private-root", required=True)
    ap.add_argument("--proxy-url", required=True)
    ap.add_argument("--routing-table", required=True)
    ap.add_argument("--max-model-len", type=int, default=None,
                    help="the launcher's resolved window (arms the once-only max_tokens clamp)")
    ap.add_argument("--max-concurrency", type=int, required=True,
                    help="in-flight requests for THIS cell (the sbatch passes max_num_seqs / n_cells)")
    ap.add_argument("--require-lens-mode", choices=["per_row", "stacked"], default="stacked",
                    help="refuse unless the proxy's petri_proxy identity block attests this mode")
    ap.add_argument("--require-engine", choices=list(ENGINE_EXPECTATIONS), default=None,
                    help="the ENGINE the proxy must attest (native-serving-default): lens:<mode> (the identity "
                         "block's lens_reduction_mode) or lora_native (engine.kind); default lens:<--require-lens-mode>, "
                         "so an invocation naming only the lens mode is unchanged. The sbatch passes $TARGET_ENGINE")
    ap.add_argument("--trial", type=int, default=0)
    ap.add_argument("--bon-decode-preset", choices=tuple(BON_DECODE_PRESETS), default="vendor_preset",
                    help="the target's sampler for the best-of-N draws. `vendor_preset` (the default; the user ruling "
                         "2026-09-17): the SERVED MODEL'S vendor preset — the results registry's "
                         "<model>|bon|off|vendor_preset row (configs/registry/decode_policy.json), the same request-body "
                         "keys the sampled static lane sends, Qwen's presence_penalty 1.5 included — resolved per model "
                         "by sisteer.eval.attacks.bon_served.resolve_bon_sampler, never a literal here. "
                         "`harness_default`: the SUPERSEDED grandfathered row (temperature 1.0 and NO top_p/top_k on "
                         "the request; the ruling of 2026-09-11, D194) every store drawn before 2026-09-17 was drawn "
                         "under — an explicit opt-in to FINISH such a store (a mixture is never a store), never the "
                         "default. `anchor_match` (qwen-base-seed-replicates): an EXPLICIT block this driver vouches for — "
                         "the Qwen BoN anchor's ENGINE-RESOLVED tuple, temperature 1.0, top_p 1.0, top_k -1, "
                         "presence_penalty 0.0 — for a standup whose server default would otherwise resolve an omitted "
                         "key differently from the anchor. Recorded as `bon_decode_preset` on every response row and in "
                         "the run record, with the registry row's id beside it (null for the explicit block). `vendor` "
                         "(qwen-seedrep-native-battery, plan §6; the user's final BoN ruling 2026-09-17 — a NEW instrument, "
                         "never pooled with the T=1.0 rows): THE REGISTRY ROW `<model_id>|bon|off|vendor_preset@v1` of "
                         "--model-key (REQUIRED), resolved by EXACT ref + not-superseded at launch "
                         "(target_decode.registry_vendor_row) — all six sampling keys SENT on every draw, presence_penalty "
                         "1.5 included; an absent / superseded / partial row refuses before the proxy is touched; the row's "
                         "identity is recorded (`decode_policy`) here and in the fan-out's manifest. `canonical` (the public "
                         "recipe's best-of-N units): the REGISTRY LABEL whose sampler reproduces the canonical best-of-N instrument "
                         "the registry's planner binds for the served model via --require-engine's path "
                         "(bon_served.canonical_bon_label over the checkout's configs/registry) -- the paper's Qwen3.5-4B and the "
                         "worked example resolve harness_default, the pipeline-test models vendor_preset; an instrument no label "
                         "reproduces refuses by name before the proxy is touched; the record names the declaration and the instrument")
    ap.add_argument("--decode-seed-policy", choices=DECODE_SEED_POLICIES, default="bon_seed_v1",
                    help="qwen-base-seed-replicates (plan §8): `bon_seed_v1` (the default) = the frozen bon_seed(goal, sample, "
                         "trial) drives both the augmentation and the decode, exactly as before; `replicate_v1` = the "
                         "augmentation keeps bon_seed (identical variant bytes) and the DECODE seed is "
                         "bon.bon_decode_seed_v1(goal, sample, trial, --decode-seed-base). Recorded on every row "
                         "(seed_rule, decode_seed_policy, decode_seed_base, augment_seed), in the manifest and here")
    ap.add_argument("--decode-seed-base", type=int, default=None,
                    help="the replicate's decode-seed base (scripts/sdfbase/seedrep/eval_seed.py seed_base(s)); REQUIRED under "
                         "`replicate_v1`, REFUSED under `bon_seed_v1` (a base with nowhere to go is a misconfiguration)")
    ap.add_argument("--model-key", default=None,
                    help="OPTIONAL cross-check: the served model as the launcher knows it (a legacy key such as gemma4e4b, "
                         "a model_id or the HF repo). The per-model sampler is DERIVED from the config's `model` (the run's "
                         "declaration; refused when the config names none) and checked against the routing table's "
                         "base_model; a --model-key that names another model refuses -- unless it names another model_id of THE "
                         "SAME CHECKPOINT (the export's hf_repo and pinned hf_revision against the config's model_revision and the "
                         "table's base_revision; the worked example qwen35_4b_demo on the paper's Qwen/Qwen3.5-4B), which is then "
                         "the identity the sampler resolves for and the rows are filed under. Every existing invocation is unchanged. "
                         "CONSULTED by `--bon-decode-preset vendor` (the registry row is keyed by the served model; REQUIRED there)")
    ap.add_argument("--limit-goals", type=int, default=0,
                    help="SMOKE ONLY: head-cap the goal set after selection (0 = the full set). A "
                         "capped run is a smoke, never a wave result: the aggregate's bon_curves refuses "
                         "any expected goal short of n_max responses, so a capped private root can never "
                         "be graded as complete")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    import jailbreak_run as jr
    import two_si_run as tsr
    from sisteer.capeval.serving.proxy import ProxyClientEngine, load_routing_table
    from sisteer.eval import request_evidence_client as rev
    from sisteer.eval.attacks import bon_served as bs
    from sisteer.eval.attacks.served_target import probe_fn, route_for
    from sisteer.eval.defense_cell import si_of

    cfg = jr._load_cfg(args.config)
    if args.defense not in jr._valid_defense_ids(cfg):
        raise SystemExit(f"[bon-served] unknown --defense {args.defense!r}")
    bcfg = cfg["bon"]
    # THE DECODE PRESET, resolved FIRST (qwen-seedrep-native-battery): a `vendor` run whose registry row is absent /
    # superseded / partial, or that names no --model-key, refuses HERE — before the proxy is touched and before a byte
    # lands in the store. The registry labels (`vendor_preset`, `harness_default`) read None here and are resolved per
    # served model below (bon_served.resolve_bon_sampler); `anchor_match` is the explicit block.
    decode, decode_policy_rec = resolve_bon_decode(args.bon_decode_preset, args.model_key)
    goals = tsr._eval_goals(cfg, confirmatory=args.confirmatory, attack="bon")
    n_goals_selected = len(goals)
    if args.limit_goals and args.limit_goals > 0:
        goals = goals[: args.limit_goals]
        print(f"[bon-served] SMOKE: --limit-goals {args.limit_goals} -> {len(goals)} of "
              f"{n_goals_selected} goals (a capped run is never a wave result)", flush=True)
    root = jr._private_root(cfg, args.private_root)
    run_dir = Path(jr._bon_defense_dir(cfg, root, args.defense))
    # the store's START marker (job id + utc): a climb reader that finds it without the end-of-run record refuses by name
    # ("512 store in progress — job N live") instead of reading the half-written responses as a store (bon_climb.store_state)
    from sisteer.eval.attacks import bon_climb as _bc
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / _bc.BON_STORE_STARTED).write_text(json.dumps({"job_id": os.environ.get("SLURM_JOB_ID"), "cell": args.defense,
                                                              "utc": _dt.datetime.now(_dt.timezone.utc).isoformat()}))
    table = load_routing_table(Path(args.routing_table))
    routes = list(table["routes"])
    route = route_for(args.defense, routes)
    # THE SERVED MODEL, before the proxy is touched: the config's `model` cross-checked against the routing table's base_model and
    # --model-key (served_model_of: a key re-pinning the same checkpoint under its own model_id -- the worked example -- is the
    # identity; a different checkpoint refuses). The sampler is resolved for it below and every row is filed under its rows.
    model = served_model_of(cfg, table, args.model_key, tag="[bon-served]")
    require_engine = require_engine_of(args.require_engine, args.require_lens_mode, tag="[bon-served]")
    serving_engine = serving_path_of(require_engine)
    # `canonical`: the registry label whose sampler reproduces the canonical best-of-N instrument the planner binds for THIS served
    # model via THIS engine's path -- the instrument the job's start-of-job registration bound -- or a refusal by name, here
    label, canonical = args.bon_decode_preset, None
    if label == BON_DECODE_PRESET_CANONICAL:
        try:
            label, canonical = bs.canonical_bon_label(model, serving_engine, seeds_dir=str(REPO / "configs" / "registry"))
        except bs.BonServedError as e:
            raise SystemExit(f"[bon-served] --bon-decode-preset canonical: {e}") from e
        print(f"[bon-served] decode preset: canonical -> {label}: the canonical best-of-N instrument for {model} via {serving_engine} is "
              f"{canonical['seed_instrument_id']} ({canonical['instrument_id'][:12]}..., decode_source {canonical['decode_source']}, "
              f"states {{k: v for k, v in canonical['decode'].items() if v is not None}}); the run draws exactly its sampler", flush=True)
    client = ProxyClientEngine(args.proxy_url, identity=route.serving_identity)
    client.load(require_virtual_models=[route.virtual_model])
    # request-evidence (F-S, qwen-seedrep-native-battery): the ONE client this child sends through is attached
    # HERE, before `ServedDefense.over_client` (whose own attach is then the idempotent no-op), declaring its
    # unlabelled sends the driver's `probe/bites` PROBES — exactly `served_target.build_served_target`'s wiring.
    # Previously the steering-bites probe below went through a client attached without the kind and died at its first
    # send ("an unlabelled target request under request evidence"); the draws are labelled by request_headers_fn
    # below and the defense refuses an unlabelled draw by name, so no draw can be recorded as a probe. Off = identity.
    client = rev.maybe_attach_proxy_client(client, unlabelled_kind=rev.PROBE_KIND_BITES)
    ident = dict(client.proxy_identity or {})
    engine_problem = _served_engine_problem(ident, expect=require_engine)
    if engine_problem is not None:
        raise SystemExit(f"[bon-served] the proxy does not attest --require-engine {require_engine}: "
                         f"{engine_problem} — refusing")
    system = None
    if route.si_prefix_hash is not None:
        from sisteer.eval.jailbreak import resolve_instruction
        system = resolve_instruction(si_of(args.defense)).text
    defense = bs.ServedDefense.over_client(client, virtual_model=route.virtual_model,
                                           cell_id=args.defense, system=system,
                                           max_model_len=args.max_model_len)
    steering_arms, steering_probe = (), None
    if bs.is_steered(route):
        # the PATH-MATCHED scale-0 twin (proxy_engine.scale0_twin): <cell>__lens0 / none__lens0 for a lens
        # route, <cell>__lora0 / none__lora0 for a native LoRA route — never the other path's zero
        from sisteer.capeval.serving import proxy_engine as pe
        twin = pe.scale0_twin(route, routes)
        if twin is None:
            raise SystemExit(f"[bon-served] steered cell {route.cell_id!r} has no scale-0 twin route — "
                             "a steered serve without a bites probe is refused")
        steering_arms = (route.cell_id.split("__")[-1],)
        steering_probe = probe_fn(client.chat_completions, route.virtual_model, twin.virtual_model)
    print(f"[bon-served] cell={args.defense} vmodel={route.virtual_model} goals={len(goals)} "
          f"n_max={bcfg['n_max']} concurrency={args.max_concurrency} window={args.max_model_len} "
          f"engine={require_engine} lens_reduction_mode={ident.get('lens_reduction_mode')} -> {run_dir}", flush=True)
    # THE SAMPLER: the served model's vendor preset from the registry by default, the
    # superseded grandfathered row by explicit opt-in, the canonical instrument's label (resolved above), or the seed-replicate
    # declarations' explicit anchor_match block — resolved ONCE here (bon_served.resolve_bon_sampler: a None preset is a registry label resolved for
    # `model`, a dict is the explicit block labelled by its name), recorded on every row and in the run record; the model is the
    # identity served_model_of resolved above (the config's, or the re-pinning key's)
    sampler = bs.resolve_bon_sampler(decode, label, model)
    print(f"[bon-served] decode preset: {sampler.label} for {model} -> {dict(sampler.decode)} "
          f"(registry {sampler.policy_id}; exactly these keys + seed + max_tokens ride every request)", flush=True)
    if decode_policy_rec is not None:
        print(f"[bon-served] decode preset: vendor — THE REGISTRY ROW {decode_policy_rec['policy_id']} "
              f"({decode_policy_rec['policy_file']}), read by EXACT ref + not-superseded, sent EXPLICITLY on every draw "
              "(qwen-seedrep-native-battery: a NEW instrument, not comparable with the T=1.0 BoN rows)", flush=True)
    from sisteer.eval.attacks import bon as bon_mod
    try:
        seed_policy, seed_base = bon_mod.check_decode_seed_policy(args.decode_seed_policy, args.decode_seed_base)
    except ValueError as e:
        raise SystemExit(f"[bon-served] --decode-seed-policy: {e}")
    print(f"[bon-served] decode-seed policy: {seed_policy}"
          + (f" (decode_seed_base {seed_base}; augmentation on the frozen bon_seed)" if seed_base is not None else " (the frozen bon_seed drives both the augmentation and the decode)"),
          flush=True)
    # REQUEST-EVIDENCE SEAM — the ONE call site where the shared client
    # (`sisteer.eval.request_evidence_client`) plugs in: `request_headers_fn(goal_id, sample_idx, trial)` returns
    # the headers of THAT draw's request — the two tinycap labels (the store's canonical task, `<goal_id>/<draw_idx>`)
    # and a fresh per-attempt X-Tinycap-Request-Id — and every row then carries `request_id` + `completion_id`
    # (bon_served._bon_generate_served_async). `rev.maybe_request_headers_fn(LEG_BON)`: the labels are the audit's own
    # spelling (`strongreject_bon`, `<goal_id>/<sample_idx>` — `filed_pairs_bon` / `read_bon_store`) and a FRESH id per
    # attempt; None when the hook is off = no headers, rows and bytes exactly as before. Previously this was a literal None
    # under a hook that was ON, so the first draw would have been refused unlabelled (F-S).
    request_headers_fn = rev.maybe_request_headers_fn(rev.LEG_BON)
    jsonl = bs.bon_generate_served(
        defense, goals, out_dir=run_dir, routes=routes, n_max=int(bcfg["n_max"]),
        sigma=float(bcfg["sigma"]), trial=args.trial, max_new_tokens=int(bcfg["max_new_tokens"]),
        max_concurrency=int(args.max_concurrency), steering_arms=steering_arms,
        steering_probe=steering_probe, decode=decode, decode_label=label, model=model,
        decode_seed_policy=seed_policy, decode_seed_base=seed_base,
        decode_policy=decode_policy_rec, request_headers_fn=request_headers_fn)
    # COMPLETENESS before the completion marker: the durable manifest's slot accounting must say every planned slot
    # is DECIDED (a response row or a typed no-fit) — the same predicate every climb reader branches on. Undecided slots -> no marker,
    # exit 3 (a resume draws them); the marker then carries the counts it certified
    decided, why, counts = _bc.store_completion(_bc.store_units(run_dir), _bc.store_slot_classification(run_dir))
    if not decided:
        print(f"[bon-served] {args.defense}: NOT complete — {why}; no completion marker written (the store stays in progress; re-run to "
              f"decide the remaining slots)", flush=True)
        return 3
    rec = {"cell_id": args.defense, "harness_cell_id": route.cell_id, "virtual_model": route.virtual_model,
           "units": counts, "complete": True, "completeness": why, "confirmatory": bool(args.confirmatory),   # the goal universe the plan reads
           "intervention_hash": route.intervention_hash, "serving_identity": route.serving_identity,
           "proxy_identity": ident, "lens_reduction_mode": ident.get("lens_reduction_mode"),
           "require_engine": require_engine, "serving_engine": ("lora_native" if require_engine == "lora_native" else "lens"),
           "n_goals": len(goals), "n_goals_selected": n_goals_selected,
           "limit_goals": int(args.limit_goals or 0), "smoke": bool(args.limit_goals),
           "n_max": int(bcfg["n_max"]), "max_concurrency": int(args.max_concurrency),
           "max_model_len": args.max_model_len, "config": args.config, "routing_table": args.routing_table,
           "served_model": model, "model_key": args.model_key,
           "bon_decode_preset": sampler.label, "decode": dict(sampler.decode), "decode_policy_id": sampler.policy_id,
           # `vendor` ONLY (qwen-seedrep-native-battery): the registry row's full identity (exact ref, version, export,
           # policy_set_sha256); every other preset adds no key
           **({"decode_policy": decode_policy_rec} if decode_policy_rec is not None else {}),
           # `canonical` ONLY: the declaration and the canonical instrument whose sampler the resolved label reproduces
           **({"bon_decode_preset_declared": args.bon_decode_preset, "canonical_instrument": canonical} if canonical is not None else {}),
           # qwen-base-seed-replicates: the decode-seed policy this run drew under (bon_seed_v1 = the frozen rule alone)
           "decode_seed_policy": seed_policy, "decode_seed_base": seed_base,
           "seed_rule": bon_mod.seed_rule_for_policy(seed_policy),
           "responses": str(jsonl), "n_target_forwards": defense.n_target_forwards,
           "clamp_report": defense.clamp_report(),
           "written_at": _dt.datetime.now(_dt.timezone.utc).isoformat()}
    (run_dir / "bon_served_run.json").write_text(json.dumps(rec, indent=2, default=str))
    print(f"[bon-served] done; n_target_forwards={defense.n_target_forwards} -> {jsonl}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
