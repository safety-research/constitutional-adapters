"""capeval runtime — the integration seam the driver loads (WS merge).

This module is the ONE place the four workstreams' halves meet; the
contract is the SEAM CONTRACT block in ``scripts/capeval_run.py``'s
docstring:

    make_engine(manifest)                           -> loaded Engine
    resolve_cell(manifest, cell, attempt, out_root, seed)
        -> (CellStore, resolved)   # hashes resolved, cell manifest.json
           written (incl. ``row_constants`` — the exec wave's copy),
           resume validated; the store carries append_ts()
    run_tier0(engine, manifest, cell, store, seed)  -> dict (tier0.run)
    run_tier1(engine, manifest, cell, store, seed)  -> dict (EvalPlus
           generation + the S2-parity probes; ``awaiting_execution`` on
           evalplus cells)
    run_target(engine, manifest, cell, store, seed) -> dict (targets)

Base lane: importing this module must stay heavy-dep-free (§3.2) — the
heavy imports (the engines, evalplus via evalplus_runner) happen inside the
run_* calls.

Cell identity: the ``intervention_hash`` that keys cells, stamps rows,
and gates resume is the STATIC hash — computable before any model load
from (artifact bytes ‖ sidecar bytes ‖ canonical_json(static spec)),
where the static spec carries every knob the harness applies on top of
the artifact (kind, layer, coeff, positions, norm_relative, and the
S2-exact skip_bos=True default for safetensors vectors). The engine's
post-load F9 resolved spec/hash (HFArmHandle.intervention_hash) is
archived by the phase runners into their phase summaries; it can differ
from the static hash ONLY through load_object-applied internals already
covered by the artifact bytes. Control-rung cell identity additionally
carries the coeff in the tier token of the cell key, because §2.1
control hashes cover {kind, seed, params} and are deliberately
coeff-free.
"""

from __future__ import annotations

import dataclasses
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

from sisteer.capeval import manifest as cm
from sisteer.capeval import store as cs
from sisteer.capeval.domains import Registry

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = REPO_ROOT / "configs" / "capeval"
#: config files whose raw bytes feed harness_config_hash (§2.1) — any
#: edit re-keys every cell (F8), which is the point. The tier-1 config left
#: the set when the task list it held was removed with its third-party
#: harness, so every cell keys anew from that change.
HASHED_CONFIG_FILES = ("domains.yaml", "tier0.yaml", "pins.yaml")
#: The in-process vLLM engine ("Phase L") loaded through the removed
#: harness's wrapper and existed only to run its task list; a manifest that
#: still names it is refused by name.
RETIRED_ENGINE_VLLM = "vllm"


class RuntimeError_(RuntimeError):
    """A runtime wiring refusal (loud, never a warning)."""


#: Kinds whose intervention lives in the PROMPT: their ``ArmHandle.transform_prompt`` is ``obj.inject``.
TOKEN_KINDS = ("token", "token_bos", "token_faithful")


class TokenKindRefusal(RuntimeError_):
    """A prompt-transforming (token-kind) arm reached tier 1, which renders prompts it does not transform."""


def refuse_token_kinds(arm: Any) -> None:
    """Refuse a token-kind arm at tier 1. The HumanEval+ generation renders evalplus's own prompts and never
    applies ``transform_prompt``, so a token-kind arm would generate with its context entered but its injected
    tokens missing: half its intervention, silently. Detection is by kind name and by behaviour (a
    ``transform_prompt`` that is not the identity); a handle whose probe raises ``NotImplementedError`` (a
    shape-only stub) is judged by its kind names alone."""
    if arm is None:
        return
    kinds = {getattr(arm, "kind", None), getattr(arm, "object_kind", None)} & set(TOKEN_KINDS)
    if kinds:
        raise TokenKindRefusal(
            f"arm {getattr(arm, 'name', '?')!r} has token kind {sorted(kinds)}: tier 1 renders prompts it does "
            "not transform, so the arm's injected tokens would be missing. Score this arm through Tier 0 or "
            "the target plugin, which render their own prompts.")
    probe = "capeval-token-kind-probe"
    try:
        transformed = arm.transform_prompt(probe)
    except NotImplementedError:
        return
    if transformed != probe:
        raise TokenKindRefusal(
            f"arm {getattr(arm, 'name', '?')!r} transforms prompts (transform_prompt is not the identity): "
            "tier 1 renders prompts it does not transform, so it refuses prompt-transforming arms.")


# ---------------------------------------------------------------------------
# engines
# ---------------------------------------------------------------------------


def make_engine(manifest: cm.Manifest):
    """A LOADED §2.4 engine for the manifest (one model load per process)."""
    if manifest.engine == "hf":
        from sisteer.capeval.serving.hf_engine import HFEngine

        engine = HFEngine(manifest.base, revision=manifest.base_revision)
    elif manifest.engine == RETIRED_ENGINE_VLLM:
        raise RuntimeError_(
            f"engine {RETIRED_ENGINE_VLLM!r} is retired: the in-process vLLM engine ran only the "
            "removed tier-1 task list; capability evals on vLLM go through the served path "
            "(scripts/served_capability_run.py)")
    elif cm.is_serving_identity(manifest.engine):
        # Petri serving engines (Phase-2 serving):
        #   proxy_vllm+<lens|lora|passthrough>+<mode> -> serving/proxy.py's
        #     ProxyClientEngine — an OpenAI-compatible client of the routing
        #     proxy (virtual model name selects the defense); the endpoint
        #     comes from manifest.proxy_base_url or $PETRI_PROXY_BASE_URL
        #     (the tunnelled localhost port), NEVER a guessed default.
        #   hf_shim+apply+<mode> -> serving/openai_shim.py's OpenAIShimEngine
        #     — the §10 gate's fp32/greedy reference oracle over HFEngine.
        #     NEVER an audit serving path: audit cells are always
        #     proxy_vllm-served by construction (enumerate_cells derives
        #     every identity from the validated spec's serving_backend,
        #     which must be proxy_vllm).
        # parse first so a malformed identity refuses loudly rather than
        # reading as a missing endpoint/model.
        ident = cm.parse_serving_identity(manifest.engine)
        if ident.backend == "proxy_vllm":
            from sisteer.capeval.serving.proxy import ProxyClientEngine

            engine = ProxyClientEngine.from_manifest(manifest)
        elif ident.backend == "hf_shim":
            from sisteer.capeval.serving.openai_shim import OpenAIShimEngine

            engine = OpenAIShimEngine.from_manifest(manifest)
        else:  # parse_serving_identity refuses first; belt-and-suspenders
            raise RuntimeError_(
                f"unknown serving backend {ident.backend!r}")
    else:  # load_manifest refuses first; belt-and-suspenders
        raise RuntimeError_(f"unknown engine {manifest.engine!r}")
    engine.load()
    return engine


def _arm_handle(engine, manifest: cm.Manifest, cell) -> Optional[Any]:
    """The cell's ArmHandle (None for kind none — base model, §2.4)."""
    if cell.kind == cm.NONE_KIND:
        return None
    ab = _find_ablation(manifest, cell.arm)
    if cell.coeff is None:
        raise RuntimeError_(f"arm {cell.arm!r}: kind {cell.kind!r} cell has "
                            "no coeff — enumeration bug")
    return engine.arm(ab, float(cell.coeff))


# ---------------------------------------------------------------------------
# cell resolution (hashes, store, row constants)
# ---------------------------------------------------------------------------


def _find_ablation(manifest: cm.Manifest, arm: str) -> cm.AblationSpec:
    matches = [a for a in manifest.ablations if a.name == arm]
    if len(matches) != 1:
        raise RuntimeError_(f"arm {arm!r} matches {len(matches)} manifest "
                            "ablations (names must be unique)")
    return matches[0]


def _artifact_paths(ab: cm.AblationSpec) -> Tuple[Path, Path]:
    """The manifest loader resolves artifact paths at load time (F9), so
    the spec path is used VERBATIM here — no cwd-relative re-resolution
    (a cwd-dependent second resolution could silently hash a DIFFERENT
    file than the loader validated)."""
    p = Path(ab.artifact)
    if not p.is_file():
        raise RuntimeError_(
            f"arm {ab.name!r}: artifact does not exist at {p} — manifest "
            "loading resolves and validates artifact paths (load_manifest); "
            "route the manifest through the loader rather than building "
            "AblationSpecs with unresolved relative paths")
    return p, cm.sidecar_path(p)


def static_intervention_spec(ab: cm.AblationSpec, coeff: Optional[float]) -> Dict[str, Any]:
    """The pre-load intervention spec the cell identity hashes over (see
    the module docstring). Deterministic from manifest + sidecar-checked
    fields only — never from a loaded model."""
    if ab.kind == cm.NONE_KIND:
        return {"kind": cm.NONE_KIND}
    if ab.kind.startswith(cm.CONTROL_KIND_PREFIX):
        return {"kind": ab.kind, "seed": ab.seed, "params": dict(ab.params)}
    spec: Dict[str, Any] = {
        "kind": ab.kind,
        "layer": ab.layer,
        "coeff": float(coeff) if coeff is not None else None,
        "positions": ab.positions,
        "norm_relative": bool(ab.norm_relative),
    }
    if ab.kind == "steering":
        # the S2-exact load_object default for safetensors vectors (§6.1);
        # torch blobs carry their internals in the hashed artifact bytes.
        spec["skip_bos"] = True
    return spec


def cell_intervention_hash(ab: cm.AblationSpec, coeff: Optional[float]) -> str:
    spec = static_intervention_spec(ab, coeff)
    if ab.kind in cm.STEERING_KINDS:
        artifact, sidecar = _artifact_paths(ab)
        sidecar_bytes = sidecar.read_bytes() if sidecar.is_file() else b""
        if ab.kind == "steering":
            if not sidecar_bytes:
                raise RuntimeError_(
                    f"arm {ab.name!r}: steering artifact {artifact} has no "
                    f"JSON sidecar at {sidecar} — cannot validate layer (F9)")
            import json as _json

            cm.validate_layer_against_sidecar(
                ab, _json.loads(sidecar_bytes.decode("utf-8")))
        return cm.intervention_hash(spec, artifact_bytes=artifact.read_bytes(),
                                    sidecar_bytes=sidecar_bytes)
    return cm.intervention_hash(spec)


def _chat_template(manifest: cm.Manifest) -> str:
    """The tokenizer's chat template string at the pinned revision — a
    harness_config_hash input (§2.1). Tokenizer-only load (no weights);
    offline-served from the HF cache on cluster nodes."""
    from transformers import AutoTokenizer  # base dep

    tok = AutoTokenizer.from_pretrained(manifest.base,
                                        revision=manifest.base_revision)
    return tok.chat_template or ""


def harness_config_inputs(manifest: cm.Manifest) -> Dict[str, Any]:
    """The resolved run config harness_config_hash covers (§2.1):
    manifest document (target/margins included — per-document in the
    multi-doc s2 convention, so each cell's chash carries its own
    relevant corpus, F20), the raw capeval config files (tier0/domains/
    pins — dataset revisions live in pins.yaml), and the pinned
    chat template."""
    m = dataclasses.asdict(manifest)
    configs = {}
    for name in HASHED_CONFIG_FILES:
        p = CONFIG_DIR / name
        if not p.is_file():
            raise RuntimeError_(
                f"harness config input missing: {p} — run "
                "scripts/capeval_fetch_data.py first (pins.yaml is Phase 0's "
                "output, §4.1)")
        configs[name] = p.read_text()
    return {"manifest": m, "configs": configs,
            "chat_template": _chat_template(manifest)}


def _coeff_token(coeff: float) -> str:
    """LOSSLESS, glob/shell-inert coeff token: ``repr(float)`` is the
    shortest round-tripping decimal (distinct floats -> distinct reprs),
    then '.'->'p', '-'->'m', '+'->'' (repr's grammar never emits 'p'/'m',
    so the mapping is injective). ``%g`` kept only six significant digits,
    so two valid distinct rungs could collide on one cell dir and resume
    onto each other (code review tier1-runtime #6)."""
    return repr(float(coeff)).replace("-", "m").replace("+", "").replace(".", "p")


def _tier_token(cell) -> str:
    """The cell key's tier_or_task token. Control LADDER rungs carry the
    coeff here (control intervention hashes are coeff-free — module
    docstring); the coeff encoding is lossless (see _coeff_token)."""
    token = "tier" + "".join(str(t) for t in cell.tiers)
    if cell.kind.startswith(cm.CONTROL_KIND_PREFIX) and cell.coeff is not None:
        # coeff 3e-3 -> "tier0-c0p003", coeff 4.0 -> "tier1-c4p0".
        token += f"-c{_coeff_token(cell.coeff)}"
    return token


def resolve_cell(manifest: cm.Manifest, cell, attempt: int,
                 out_root, seed: int,
                 registry_identity: Optional[Mapping[str, Any]] = None,
                 registry_emit: Optional[bool] = None) -> Tuple[cs.CellStore, Dict[str, Any]]:
    """Resolve one cell's identity (§2.3): hashes, cell key, row
    constants; write/validate the cell manifest.json; return the store.

    The returned mapping is exactly what lands in manifest.json —
    including ``row_constants`` (the 15 cell-constant ResultRow columns;
    the EvalPlus exec wave builds execution rows from that copy) and the
    archived static intervention spec.

    Results-registry native emission (additive): ``registry_identity`` is
    the registry-grain identity of the run(s) this store will hold —
    ``eval_id``, ``item_set_id``, ``instrument_id`` and the other declared
    ``run`` facts (one mapping, or ``{task: mapping}`` for a store holding
    several evals); unset, it is read from the planned-run file the submit
    wrapper stored beside the job (``$SISTEER_REGISTRY_RUN_IDENTITY``).
    ``registry_emit`` forces emission on/off; None applies the one rule in
    ``sisteer.registry.hooks`` (``REGISTRY_EMIT``, else "a spool exists").
    With emission off, or no usable identity, the store is built exactly
    as before (no recorder — ``cs.NO_RECORDER`` skips the store's own
    default resolution, which would read the same environment); the
    manifest bytes never change."""
    ab = _find_ablation(manifest, cell.arm)
    ihash = cell_intervention_hash(ab, cell.coeff)
    chash = cs.harness_config_hash(harness_config_inputs(manifest))
    key = cs.cell_key(arm=cell.arm, tier_or_task=_tier_token(cell),
                      trial=cell.trial, attempt=attempt,
                      intervention_hash=ihash, harness_config_hash=chash)
    row_constants: Dict[str, Any] = {
        "sweep": manifest.sweep,
        "run_id": f"{manifest.sweep}.{cell.index}.t{cell.trial}.a{attempt}",
        "model_id": manifest.base,
        "model_revision": manifest.base_revision,
        "arm_name": cell.arm,
        "kind": cell.kind,
        "coeff": cell.coeff,
        "intervention_hash": ihash,
        "engine": manifest.engine,
        "trial_id": cell.trial,
        "seed": seed,
        "attempt": attempt,
        "harness_config_hash": chash,
        "env_digest": cs.env_digest(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    resolved: Dict[str, Any] = {
        **row_constants,
        "row_constants": dict(row_constants),
        "static_intervention_spec": static_intervention_spec(ab, cell.coeff),
        "tiers": list(cell.tiers),
        "cell_index": cell.index,
        "time_class": cell.time_class,
        "resolved_at": time.time(),
    }
    recorder = None
    if registry_emit is not False:
        try:
            from sisteer.registry import hooks as _registry_hooks   # stdlib-only
            recorder = _registry_hooks.store_recorder(
                key, resolved, registry_identity, registry_emit=registry_emit,
                script="capeval.runtime.resolve_cell")
        except Exception as e:  # noqa: BLE001 — a lost record never costs a run
            print(f"[registry] resolve_cell {key}: recorder not built "
                  f"({type(e).__name__}: {e}); the run continues", flush=True)
            recorder = None
    # this function already resolved (or, under registry_emit=False, declined) the recorder: hand the store
    # its verdict rather than letting CellStore's default resolve a second time from the same environment
    store = cs.CellStore(out_root, key, row_constants=row_constants,
                         recorder=recorder if recorder is not None else cs.NO_RECORDER)
    store.validate_resume(resolved)
    store.write_cell_manifest(resolved)
    return store, resolved


# ---------------------------------------------------------------------------
# calibration inputs for Tier 0 (canary thresholds, §7.1; κ no longer
# feeds Tier 0 — KL is now an annotation, not a gate)
# ---------------------------------------------------------------------------


def calibration_json_path(out_root) -> Path:
    env = os.environ.get("CAPEVAL_CALIBRATION_JSON")
    if env:
        return Path(env)
    return Path(out_root).parent / "calibration" / "calibration.json"


def _tier0_thresholds(manifest: cm.Manifest, out_root):
    """CanaryThresholds|None: None during calibration mode (thresholds
    are its OUTPUT); for every other mode a missing/foreign calibration
    is a REFUSAL, never a silent default (§7.1). κ is no longer
    required or returned — Tier 0 files kl_coarse_lb magnitudes as
    annotations and flags nothing on them."""
    from sisteer.capeval.stats import calibrate as cal
    from sisteer.capeval.tiers.tier0 import CanaryThresholds

    if manifest.mode == "calibration":
        return None
    path = calibration_json_path(out_root)
    if not path.is_file():
        raise RuntimeError_(
            f"no calibration at {path} — mode {manifest.mode!r} cells need "
            "the §7.1 null (canary thresholds); run the calibration "
            "sweep first (§11.2 order), or point CAPEVAL_CALIBRATION_JSON "
            "at it")
    calib = cal.load_calibration(path)
    registry = Registry.load(CONFIG_DIR / "domains.yaml")
    cal.check_calibration(calib, model_revision=manifest.base_revision,
                          engine=manifest.engine,
                          harness_version=cal.harness_version(registry))
    if calib.canary_thresholds is None:
        raise RuntimeError_(
            f"calibration at {path} lacks canary_thresholds — "
            "rebuild it (stats.calibrate.build_calibration)")
    return CanaryThresholds.from_calibration(calib.canary_thresholds)


# ---------------------------------------------------------------------------
# phase runners
# ---------------------------------------------------------------------------


def _domain_of_map() -> Dict[str, str]:
    registry = Registry.load(CONFIG_DIR / "domains.yaml")
    return {task: dom.name for dom in registry.domains for task in dom.tasks}


def _filed_keys(store: cs.CellStore) -> set:
    """(task, item_id, metric) keys already filed in this cell — the
    per-item resume skip set (§2.3, deduped exactly like tier0's emitter)."""
    return {(r.get("task"), r.get("item_id"), r.get("metric"))
            for r in store._parsed_rows()}


def _tier0_corpus_windows(tok, cfg_entry: Mapping[str, Any], sl,
                          raw: Mapping[str, Any]):
    """Window one tier0.yaml corpus entry with the SAME geometry the HF
    path uses (tier0's public windowing helpers + tier0.yaml knobs)."""
    from sisteer.capeval.tiers import tier0

    if sl.kind == "text":
        return tier0.windows_from_texts(
            tok, sl.name, list(sl.items), window_len=raw["window_len"],
            token_budget=cfg_entry.get("token_budget"),
            min_window_tokens=raw["min_window_tokens"])
    if sl.kind == "chat":
        return tier0.windows_from_conversations(
            tok, sl.name, list(sl.items), list(sl.item_ids),
            assistant_token_budget=cfg_entry.get("assistant_token_budget"),
            max_tokens=raw["chat_max_tokens"])
    raise RuntimeError_(f"corpus {sl.name!r}: unsupported kind {sl.kind!r}")


def run_tier0(engine, manifest: cm.Manifest, cell, store: cs.CellStore,
              seed: int, confirm_probes: Optional[frozenset] = None) -> Dict[str, Any]:
    """tier0.run behind the driver seam: build Tier0Spec from
    tier0.yaml + pins.yaml + the calibration null, resolve the arm, run.

    ``confirm_probes`` (confirmation protocol, review tier1 #5): the
    attempt-1 rerun passes the persisted probe-NAME set from attempt 0's
    ``confirmation_required.json`` (tier0.CONFIRMABLE_PROBES vocabulary);
    None means "not a confirmation rerun" — tier0.run distinguishes the
    two (frozen parameter name, orchestrator decision)."""
    from sisteer.capeval import corpora
    from sisteer.capeval.tiers import tier0

    raw = tier0.load_tier0_yaml(CONFIG_DIR / "tier0.yaml")
    pins = corpora.load_pins()
    loaders = {"wikitext103": corpora.wikitext103, "stack_py": corpora.stack_py,
               "ultrachat": corpora.ultrachat}
    corpora_cfgs: List[tier0.Tier0CorpusCfg] = []
    for c in raw["corpora"]:
        if c["name"] not in loaders:
            raise RuntimeError_(f"tier0.yaml corpus {c['name']!r} has no "
                                f"corpora.py loader (known: {sorted(loaders)})")
        corpora_cfgs.append(tier0.Tier0CorpusCfg(
            slice=loaders[c["name"]](pins),
            nll_task=c["nll_task"], kl_task=c["kl_task"],
            token_budget=c.get("token_budget"),
            assistant_token_budget=c.get("assistant_token_budget")))

    thresholds = _tier0_thresholds(manifest, store.sweep_dir)
    can = raw["canary"]
    canary = tier0.CanaryCfg(
        slice=corpora.canary_prompts(pins),
        task=can["task"], gen=dict(can.get("gen", {})),
        thresholds=thresholds,
        abrupt_min_tokens=can.get("abrupt_min_tokens", 8),
        rep_ns=tuple(can.get("rep_ns", (1, 2, 3))))

    relevant = None
    if manifest.mode == "sweep" and manifest.target is not None:
        from sisteer.capeval.targets import relevant_slice

        relevant = relevant_slice(manifest.target)

    spec = tier0.Tier0Spec(
        template=dict(store.row_constants),
        domain_of=_domain_of_map(),
        corpora=corpora_cfgs,
        canary=canary,
        relevant_corpus=relevant,
        localization_task=raw["localization"]["task"],
        top_k=raw["top_k"], window_len=raw["window_len"],
        min_window_tokens=raw["min_window_tokens"],
        chat_max_tokens=raw["chat_max_tokens"],
        code_version=raw["code_version"],
        build_cache=(manifest.mode == "calibration" and cell.trial == 0))
    arm = _arm_handle(engine, manifest, cell)
    caches = Path(store.sweep_dir).parent / "cache" / "klatoms"
    summary = tier0.run(engine, arm, spec, store, caches,
                        confirm_probes=(frozenset(confirm_probes)
                                        if confirm_probes is not None else None))
    if arm is not None:
        summary["arm_intervention_hash"] = arm.intervention_hash  # post-load F9 hash
    return summary


def run_tier1(engine, manifest: cm.Manifest, cell, store: cs.CellStore,
              seed: int) -> Dict[str, Any]:
    """Tier 1 behind the driver seam: the HumanEval+ generation (an EvalPlus
    generation leaves ``awaiting_execution`` for the sandbox wave, F8) and,
    on the HF engine, the S2-parity probes. The tier-1 task list that once ran
    first (through a third-party harness, with its vendored task files) is
    removed, so this phase files no rows of its own beyond those two."""
    arm = _arm_handle(engine, manifest, cell)
    refuse_token_kinds(arm)    # before any generation: the HumanEval+ prompts are not transformed
    summary: Dict[str, Any] = {
        "rows_emitted": 0,
        "confirmation_needed": [],
    }
    if arm is not None:
        summary["arm_intervention_hash"] = arm.intervention_hash

    # EvalPlus generation (§5.5): all 164 problems (evalplus 0.3.1
    # requires a sample per problem — verified), greedy, sanitized;
    # execution belongs to the sandbox wave, results.json to the wave.
    # Resume keys on samples.jsonl itself (written ATOMICALLY, so its
    # existence means complete) — generation_complete.json is written by
    # the DRIVER only after every local phase is done and
    # results_pending.json is committed (F8, review tier1 #3). The
    # generation config is the stock instrument's
    # (``evalplus_runner.DEFAULT_GEN_CFG``: greedy, 768 tokens, stop at the
    # closing fence), the same one the served instrument uses.
    from sisteer.capeval.tiers import evalplus_runner

    samples_path = Path(store.cell_dir) / "samples.jsonl"
    if samples_path.is_file():
        # re-validate the pinned 164-task ID set on resume (tier1 #4)
        evalplus_runner.read_samples_jsonl(
            samples_path,
            expected_task_ids=evalplus_runner.EXPECTED_TASK_IDS)
        summary["evalplus_generation"] = "already complete (resume)"
    else:
        samples = evalplus_runner.generate_samples(
            engine=engine, arm=arm, tokenizer=engine.tokenizer,
            gen_cfg=dict(evalplus_runner.DEFAULT_GEN_CFG), store=store)
        summary["evalplus_generation"] = f"{len(samples)} samples"
    summary["awaiting_execution"] = True

    # S2-parity probes: every calibration trial and every
    # sweep (rediscovery) cell, HF engine only — they are engine-matched
    # to the arm runs and pair vs the HF calibration trial 0; control
    # cells are outside their scope.
    # Dispatched here (tier 1 runs in every relevant cell) so the driver
    # orchestration and its pinned event ordering stay untouched.
    if manifest.mode in ("calibration", "sweep") and manifest.engine == "hf":
        from sisteer.capeval import corpora
        from sisteer.capeval.tiers import parity
        from sisteer.capeval.tinycap import contracts as tc

        pins = corpora.load_pins()
        # The gsm8k probe (s2parity/gsm8k_0shot) is the RETIRED bespoke
        # instrument: it generates only under SISTEER_LEGACY_GSM8K=1 —
        # the frozen S2 rediscovery readout re-run deliberately.
        # Otherwise the mmlu probe runs alone and the skip is RECORDED,
        # never silent; the standard GSM8K is inspect_evals' `gsm8k` on
        # the served path.
        legacy = tc.legacy_gsm8k_enabled()
        pspec = parity.ParitySpec(
            template=dict(store.row_constants),
            domain_of=_domain_of_map(),
            gsm8k_rows=parity.gsm8k_parity_rows(pins) if legacy else [],
            mmlu_rows=parity.mmlu_parity_rows(pins),
            order_seed=seed)
        summary["s2parity"] = parity.run(engine, arm, pspec, store)
        if not legacy:
            summary["s2parity_gsm8k"] = (
                f"retired: {tc.GSM8K_RETIRED_TASK} not run (standard: "
                f"{tc.GSM8K_STANDARD_TASK} on the served path; "
                f"{tc.GSM8K_LEGACY_ENV}=1 re-enables the legacy probe)")
    return summary


# ---------------------------------------------------------------------------
# petri (a THIN dispatch seam; the runner body lives
# in tiers/petri_runner.py. run_tier0/run_tier1/run_target are untouched:
# Petri never routes through the capability tier/target/fingerprint/verdict
# machinery.)
# ---------------------------------------------------------------------------


def run_petri(engine, spec, cell, store: cs.CellStore,
              seed: int) -> Dict[str, Any]:
    """Petri audit dispatch seam (mirrors run_tier0/run_tier1's shape).

    A THIN, LAZY delegate to ``tiers/petri_runner.py`` (lazy so the base
    import stays heavy-dep-free) — the contract below binds the runner:

    - ``engine``: the ``make_engine`` handle for a serving-identity engine
      string — Phase 2's routing-proxy OpenAI client
      (``serving/proxy.py``, backend ``proxy_vllm``); NEVER the
      in-process capability engines, and never the HF shim (the shim is
      the gate's fp32 reference oracle only — audit cells are
      proxy_vllm-served by construction).
    - ``spec``: a validated ``manifest.PetriSpec`` with every hash input
      FROZEN (``manifest.petri_harness_config_inputs`` refuses UNFROZEN
      sentinels, so an unfrozen scenario bundle or serving pin can never
      reach a run).
    - ``cell``: a resolved ``sisteer.capeval.petri.defense_cells.PetriCell``
      (virtual model name, serving identity, and the three keys:
      intervention_hash / harness_config_hash / petri_run_key).
    - ``store``: a ``store.CellStore`` for this cell, keyed under
      ``cell.petri_run_key`` (content-addressed resume).
    - ``seed``: int, threaded to inspect_ai where applicable.

    Returns a dict summary (the run_tier0/run_tier1 convention), at
    minimum::

        {"rows_emitted": <int>,
         "behaviors": {<behavior_id>: {...per-behavior summary...}},
         "unscored": <failure-ledger entry count>}

    Runner body contract (binding on the implementation):
    - drives ``inspect eval_set`` over the frozen scenario set and converts
      ``.eval`` logs to store rows: EXACTLY one row per
      ``(cell, canonical scenario, dimension)`` with metrics exactly
      ``{<behavior_id>, eval_awareness, scenario_realism}`` (any aggregate
      goes under a DISTINCT metric name, never a per-item one).
    - normalizes SI-prefix-variant scenario ids back to canonical
      ``(behavior_id, scenario_id)`` before any row is written (H6).
    - an auditor/judge failure lands in the ``petri_incomplete.json``
      failure ledger, NEVER a fabricated score row (H9); ``results.json``
      (the only cell-done marker) is written only at zero unresolved
      required dimensions.
    - resume skips ``(task, item_id, metric)`` keys already filed; the
      Inspect ``log_dir`` is namespaced by ``cell.petri_run_key`` and
      embedded metadata is re-verified before any completed sample is
      accepted (§12).
    """
    from sisteer.capeval.tiers import petri_runner  # lazy: petri lane only

    return petri_runner.run(engine, spec, cell, store, seed)


def _leg_value_map(leg: Any, which: str) -> Dict[str, float]:
    """item_id -> value for one target leg; duplicate ids are a refusal
    (review tier1-runtime #7: positional zip silently mispaired)."""
    ids = list(leg.item_ids)
    if len(ids) != len(set(ids)):
        dups = sorted({i for i in ids if ids.count(i) > 1})[:5]
        raise RuntimeError_(f"target {which} leg has duplicate item_ids "
                            f"(e.g. {dups}) — refusing to pair")
    return dict(zip(ids, (float(v) for v in leg.values)))


def _seed_permuted_prompts(prompts, n: int, seed: int):
    """The first-n canonical prompts in a deterministic seed-derived
    ORDER (review tier1-runtime #8): attempt 1's fresh seed therefore
    reruns the target under a genuinely different batching/order
    condition, and the actual order is recorded per row via
    position_order. The SELECTED SET stays the canonical first n — only
    the order moves with the seed."""
    import random

    selected_ids = list(prompts.item_ids[:n])
    selected_items = list(prompts.items[:n])
    order = list(range(len(selected_ids)))
    random.Random(seed).shuffle(order)
    return dataclasses.replace(
        prompts,
        item_ids=tuple(selected_ids[i] for i in order),
        items=tuple(selected_items[i] for i in order))


def run_target(engine, manifest: cm.Manifest, cell, store: cs.CellStore,
               seed: int) -> Dict[str, Any]:
    """Target plugins behind the driver seam: both legs (arm
    + base) run IN this cell — base generations are cheap (100 greedy
    prompts) and a self-contained cell beats a cross-cell dependency
    (a deviation from a shared base cell). Legs are paired by
    item_id (never positionally) and both run in the same seed-permuted
    prompt order."""
    from sisteer.capeval import corpora
    from sisteer.capeval.targets import resolve_target

    bindings = resolve_target(manifest)  # [] for control/calibration (F20)
    if not bindings:
        return {}
    arm = _arm_handle(engine, manifest, cell)
    pins = corpora.load_pins()
    prompts = corpora.alpaca_holdout_prompts(pins)
    seen = _filed_keys(store)
    summary: Dict[str, Any] = {"primaries": {}, "confirmation_needed": []}

    def _file(leg: Any, metric_suffix: str, task: str) -> None:
        for order, (item_id, value) in enumerate(zip(leg.item_ids, leg.values)):
            metric = f"{leg.metric}/{metric_suffix}"
            if (task, item_id, metric) in seen:
                continue
            store.append_row(cs.ResultRow(
                **store.row_constants, tier=-1, task=task,
                task_version="target_v1", domain=cs.TARGET_DOMAIN,
                item_id=item_id, position_order=order, metric=metric,
                value=float(value),
                extra_json=None, created_at=time.time(), wall_clock_s=None))

    for primary, plugin in bindings:
        task = f"target/{primary.plugin}"
        n_prompts = int(dict(primary.params).get("n_prompts", 100))
        permuted = _seed_permuted_prompts(prompts, n_prompts, seed)
        arm_leg = plugin(engine, arm, primary, permuted)
        base_leg = plugin(engine, None, primary, permuted)
        _file(arm_leg, "arm", task)
        _file(base_leg, "base", task)
        a_map = _leg_value_map(arm_leg, "arm")
        b_map = _leg_value_map(base_leg, "base")
        if set(a_map) != set(b_map):
            only_a = sorted(set(a_map) - set(b_map))[:5]
            only_b = sorted(set(b_map) - set(a_map))[:5]
            raise RuntimeError_(
                f"target {primary.metric!r}: leg item sets differ "
                f"(arm-only e.g. {only_a}, base-only e.g. {only_b}) — "
                "refusing to compute paired deltas over mismatched items "
                "(review tier1-runtime #7)")
        paired = [a_map[i] - b_map[i] for i in sorted(a_map)]
        delta = sum(paired) / len(paired) if paired else 0.0
        summary["primaries"][primary.metric] = {
            "delta": delta,
            "arm_mean": sum(a_map.values()) / len(a_map),
            "base_mean": sum(b_map.values()) / len(b_map),
            "n": len(paired),
            "direction": primary.direction,
            "min_effect": primary.min_effect,
            "params_resolved": arm_leg.params_resolved,
        }
    return summary
