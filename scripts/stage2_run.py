"""Stage-2 driver: the one entry point every stage2_*.sbatch invokes.

CLI contract (documented in the sbatch headers):

    uv run python scripts/stage2_run.py --config <yaml> --suite
        {pairs,ifeval,composition,ladder}
        [--si NAME] [--arm NAME] [--cell NAME --rung N]
        [--canary] [--all-cells] [--teacher greedy|rejection]

Responsibilities (each suite runner below):
  * load the config + model (config model/model_revision/dtype,
    attn_implementation passthrough);
  * resolve config SI ids through data/instructions.resolve_si (an
    unregistered id fails loudly); OUTPUT DIRS are keyed by the config id,
    matching judge_batch.py's cell layout;
  * build/train arm objects via the methods.py registry (ARMS /
    arms_for_suite), skipping training when the artifact exists — the
    matrix cell's resume unit;
  * move every loaded object to the model device (methods.load_object
    does this);
  * dispatch to the eval modules through eval/suites.get_runner;
  * write results JSON/JSONL with item-level resume (owned by the eval
    modules) and mirror summaries to W&B (offline on the cluster;
    stage2_common.sh documents the login-node `wandb sync` post-step).

Driver-level decisions (recorded here because Phase-0 left them open):
  * the anchor, drift, axbench and longctx suites are gone: they ran the
    vendored IFEval verifier, the PersonaDrift scorers, AxBench and the
    IFEval-at-length suite, third-party material behind no paper result,
    removed rather than attributed. `--suite` refuses them by name.
  * lifbench was REMOVED from the campaign (instruction-after-context
    = retrieval, not instruction decay); --suite lifbench fails loudly
    below rather than silently no-opping on a stale command line.
"""

from __future__ import annotations

import argparse
import fcntl
import functools
import hashlib
import json
import os
import subprocess
from contextlib import contextmanager
from pathlib import Path

import yaml

from sisteer.data.instructions import (  # noqa: E402
    BEHAVIORS as BEHAVIOR_SIS,
    Instruction,
    joined_behavior_instruction,
    resolve_si,
)
from sisteer.data.pairs import (  # noqa: E402
    Pair,
    generate_compliant_pairs,
    generate_pairs,
    load_pairs,
    save_pairs,
)
from sisteer.eval.suites import get_runner  # noqa: E402
from sisteer.methods import ARMS, VectorObject, load_object  # noqa: E402

# Arms whose artifact is a SteeringVector safetensors (load_object sniffs
# content, so the extension is cosmetic — it matches the stage-1 layout).
# caa/actadd/actadd_front are the stage-2 cheap-estimator arms.
VECTOR_SAFETENSOR_ARMS = {"vector", "vector_rkl", "vector_js", "bidpo",
                          "antidote",  # SteeringVector artifact
                          "did",       # SteeringVector artifact
                          "caa", "actadd", "actadd_front"}

REPO_ROOT = Path(__file__).resolve().parents[1]

MANIFEST_NAME = "manifest.json"


# ---------------------------------------------------------------- helpers


def _load_cfg(path: str) -> dict:
    """Load a config, resolving a single-level `extends: <sibling.yaml>`
    (shallow merge: the child's top-level keys override the base's
    wholesale). Added so the RS teacher ablation config
    INHERITS the matrix config — model/revision/arms/
    train_kwargs can never drift between the greedy and RS runs."""
    cfg = yaml.safe_load(Path(path).read_text())
    base_name = cfg.pop("extends", None)
    if base_name:
        base = _load_cfg(str(Path(path).parent / base_name))
        base.update(cfg)
        cfg = base
    # the path-valued keys may carry a site placeholder ($SHARED/...): expanded once here through
    # sisteer.site, which refuses an unset placeholder by name, so no cell writes under a literal $SHARED.
    # wandb_project stays as written: make_wandb_logger expands it and, with no value on this host,
    # turns W&B off for the run by name instead of stopping it
    from sisteer import site as _site
    cfg = _site.expand_keys(cfg, ("out_dir", "greedy_results_root"), paths=True)
    if isinstance(cfg.get("padding"), dict):
        cfg["padding"] = _site.expand_keys(cfg["padding"], ("filler_cache",), prefix="padding.", paths=True)
    return cfg


def _load_model(cfg: dict, model_name: str | None = None,
                revision: str | None = None):
    from sisteer.models import load_model_and_tokenizer

    kwargs = {}
    if cfg.get("attn_implementation"):
        kwargs["attn_implementation"] = cfg["attn_implementation"]
    try:
        return load_model_and_tokenizer(
            model_name or cfg["model"],
            dtype=cfg.get("dtype"),
            device=cfg.get("device"),
            revision=revision if model_name else cfg.get("model_revision"),
            **kwargs,
        )
    except ImportError as e:
        # flash_attention_2 needs the flash-attn wheel, which ships behind
        # the OPTIONAL `longctx` extra (a source-built torch-2.11 wheel in
        # the gitignored wheels/ dir — see pyproject). If the env was
        # synced without it, transformers raises ImportError at load;
        # degrade to sdpa with a LOUD warning instead of killing a
        # multi-hour suite over a perf knob. Deliberately loud, never
        # silent: results are still valid under sdpa, just slower. Any
        # other ImportError (or any non-FA2 impl) propagates unchanged.
        if kwargs.get("attn_implementation") != "flash_attention_2":
            raise
        print(f"[stage2] WARNING: flash_attention_2 unavailable "
              f"({type(e).__name__}: {e}) — falling back to sdpa. "
              f"Sync with `uv sync --frozen --extra cu128 --extra longctx` "
              f"(wheel in wheels/) to run FA2.", flush=True)
        kwargs["attn_implementation"] = "sdpa"
        return load_model_and_tokenizer(
            model_name or cfg["model"],
            dtype=cfg.get("dtype"),
            device=cfg.get("device"),
            revision=revision if model_name else cfg.get("model_revision"),
            **kwargs,
        )


def _prompt_split(cfg: dict) -> tuple[list[str], list[str]]:
    """(fit, eval) prompts per the config's prompts block (stage-1 recipe)."""
    from sisteer.data.prompts import load_hf_prompts, seed_prompts, split_prompts

    pcfg = cfg.get("prompts", {"source": "seed", "set": "general"})
    if pcfg.get("source") == "hf":
        base = load_hf_prompts(pcfg.get("dataset", "tatsu-lab/alpaca"),
                               n=pcfg.get("n", 500))
    else:
        base = seed_prompts(pcfg.get("set", "general"))
    fit, ev = split_prompts(base, holdout_frac=pcfg.get("holdout_frac", 0.5))
    if cfg.get("n_fit_prompts"):
        fit = fit[: cfg["n_fit_prompts"]]
    if cfg.get("n_eval_prompts"):
        ev = ev[: cfg["n_eval_prompts"]]
    return fit, ev


def _train_kwargs(cfg: dict, arm: str) -> dict:
    tk = cfg.get("train_kwargs") or {}
    kw = dict(tk.get("default") or {})
    kw.update(tk.get(arm) or {})
    return kw


def _arm_scales(cfg: dict, arm: str) -> list[float]:
    """The factor grid for one arm: the global `scales` plus any per-arm
    ADDITIVE extension under `extra_scales`, merged/deduped/ascending.
    Additive on purpose — a global-grid edit still propagates to every arm,
    and an arm's extras can never silently DROP a shared grid point. The
    grid is an eval-only knob, deliberately outside _cell_config_hash's
    subset, so extending it extends completed cells incrementally (the
    _extend_ifeval_cell path) instead of invalidating trained artifacts.

    First use: 1/n_layers = 1/32 = 0.03125
    on the three _all arms — they apply the intervention at all 32 layers
    of Llama-3.1-8B, so the total perturbation compounds ~32x vs the
    single-layer arms at equal nominal factor; the 1/32 point asks whether
    the single-layer-equivalent TOTAL dose, distributed, stays usable
    (natural comparison: vector_all@0.03125 vs vector@1.0)."""
    base = list(cfg.get("scales") or [1.0])
    extra = (cfg.get("extra_scales") or {}).get(arm) or []
    if not extra:
        return base
    return sorted(set(base) | set(extra))


def _object_path(cell_dir: Path, arm: str) -> Path:
    name = "vector.safetensors" if arm in VECTOR_SAFETENSOR_ARMS else "object.pt"
    return cell_dir / name


# -- atomic publishes (existence == fully published) -------------------------


def _fsync_replace(tmp: Path, final: Path) -> None:
    with tmp.open("rb") as f:
        os.fsync(f.fileno())
    os.replace(tmp, final)


def _atomic_save(obj, path: Path) -> None:
    """Publish an intervention artifact atomically: temp file in the same
    dir, fsync, os.replace. A preemption mid-save must never leave
    a truncated file that the artifact-exists resume then trusts as a
    trained object. SteeringVector.save also writes a .json sidecar next
    to the artifact (required by SteeringVector.load); it is published
    FIRST so the artifact's existence implies the sidecar's."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    obj.save(tmp)
    side_tmp = tmp.with_suffix(".json")  # SteeringVector sidecar, if any
    if side_tmp != tmp and side_tmp.exists():
        _fsync_replace(side_tmp, path.with_suffix(".json"))
    _fsync_replace(tmp, path)


# -- resume provenance manifests (lightweight version) -----------------------


@functools.lru_cache(maxsize=1)
def _code_sha() -> str:
    env = os.environ.get("SISTEER_CODE_SHA")
    if env:
        return env
    res = subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
                         capture_output=True, text=True)
    if res.returncode == 0 and res.stdout.strip():
        return res.stdout.strip()
    print("[stage2] WARNING: cannot resolve the code SHA (not a git "
          "checkout and SISTEER_CODE_SHA unset) — manifests record 'unknown'")
    return "unknown"


def _cell_config_hash(cfg: dict, arm: str, extra: dict | None = None) -> str:
    """sha256 of the canonicalized cell-relevant config subset: everything
    that determines the TRAINED artifact (never eval-only knobs, so an
    eval-side config change doesn't invalidate a valid object)."""
    subset = {
        "model": cfg.get("model"),
        "model_revision": cfg.get("model_revision"),
        "dtype": cfg.get("dtype"),
        "layer": cfg.get("layer"),
        "train_kwargs": _train_kwargs(cfg, arm),
        "prompts": cfg.get("prompts"),
        "n_fit_prompts": cfg.get("n_fit_prompts"),
        "max_new_tokens": cfg.get("max_new_tokens"),
    }
    if extra:
        subset["extra"] = extra
    canon = json.dumps(subset, sort_keys=True, default=str)
    return hashlib.sha256(canon.encode()).hexdigest()


def _cell_manifest(cfg: dict, arm: str, si_id: str, *,
                   extra: dict | None = None, model: str | None = None,
                   revision: str | None = None) -> dict:
    """The lightweight provenance record written next to every trained
    artifact. model/revision overrides serve cells that train on a
    non-config model."""
    return {
        "code_sha": _code_sha(),
        "config_hash": _cell_config_hash(cfg, arm, extra),
        "model": model or cfg.get("model"),
        "model_revision": revision if model else cfg.get("model_revision"),
        "arm": arm,
        "si": si_id,
    }


def _check_manifest(artifact: Path, manifest: dict) -> None:
    """Hard-fail when an existing artifact's manifest disagrees with the
    current run: resuming over an object trained under a different
    model/revision/config silently poisons every downstream number.
    code_sha is recorded provenance, not an identity field — every fix
    wave changes it, and blocking resume on it would force retraining the
    whole matrix per commit."""
    mpath = artifact.parent / MANIFEST_NAME
    if not mpath.is_file():
        raise SystemExit(
            f"artifact {artifact} has no {MANIFEST_NAME} beside it — a "
            "pre-manifest artifact or a torn publish; resume provenance is "
            "required (D22#2). Delete the cell directory to retrain."
        )
    prev = json.loads(mpath.read_text())
    mismatched = {k: (prev.get(k), v) for k, v in manifest.items()
                  if k != "code_sha" and prev.get(k) != v}
    if mismatched:
        detail = "; ".join(f"{k}: artifact={a!r} vs this-run={b!r}"
                           for k, (a, b) in sorted(mismatched.items()))
        raise SystemExit(
            f"RESUME PROVENANCE MISMATCH for {artifact}: {detail} — the "
            "existing artifact was trained under a different setup; "
            "refusing to resume over it (D22#2). Delete the cell "
            "directory to retrain."
        )
    if prev.get("code_sha") != manifest.get("code_sha"):
        print(f"[stage2] resuming {artifact}: trained at code_sha "
              f"{prev.get('code_sha')!r}, now {manifest.get('code_sha')!r} "
              "(config/model provenance matches)")


# -- pairs (the pair-generation race) ----------------------------------------


@contextmanager
def _pairs_lock(pairs_path: Path):
    lock = pairs_path.with_name(pairs_path.name + ".lock")
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open("a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def _require_pairs(pairs_path: Path) -> list[Pair]:
    """Suites WITH a P1 pairs stage (ifeval): the shared pairs file
    must already exist. Silent per-cell regeneration was the race this guard
    forbids — concurrent arm cells racing to write one shared file."""
    if not pairs_path.exists():
        raise SystemExit(
            f"shared pairs file missing: {pairs_path} — run the P1 pairs "
            "stage first (stage2_run.py --suite "
            "pairs). Per-cell generation is disabled here by design "
            "(D22#5): concurrent arm cells generating the same shared "
            "file is a race."
        )
    return load_pairs(pairs_path)


def _ensure_pairs(model, tok, instruction: Instruction, fit_prompts: list[str],
                  pairs_path: Path, max_new_tokens: int) -> list[Pair]:
    """Suites WITHOUT a P1 stage (composition and the P1 stage itself): generate the pairs file if absent, under an
    exclusive flock with an existence recheck after acquiring, and
    publish atomically (save_pairs is tmp+fsync+replace)."""
    if pairs_path.exists():
        return load_pairs(pairs_path)
    with _pairs_lock(pairs_path):
        if pairs_path.exists():  # a peer generated it while we waited
            return load_pairs(pairs_path)
        pairs = generate_pairs(model, tok, instruction, fit_prompts,
                               max_new_tokens=max_new_tokens)
        save_pairs(pairs, pairs_path)
    return pairs


# -- rejection-sampling teacher ----------------------------------------
# The greedy teacher (generate_pairs) keeps EVERY SI-prompted completion,
# so on low-prompted-ceiling SIs most distillation targets are
# non-compliant. teacher: rejection rebuilds the targets for the SIs whose
# GREEDY prompted strict < si_prompted_threshold by rejection sampling
# (generate_compliant_pairs), in a PARALLEL namespace (out_dir must differ
# from greedy_results_root) so filtered-vs-unfiltered is a clean ablation.

TEACHERS = ("greedy", "rejection")


def _effective_teacher(cfg: dict, args) -> str:
    """CLI --teacher and config teacher: must agree when both given — a
    CLI 'greedy' silently overriding a config 'rejection' would write
    unfiltered pairs into the RS namespace (or vice versa)."""
    cli = getattr(args, "teacher", None)
    from_cfg = cfg.get("teacher")
    if cli and from_cfg and cli != from_cfg:
        raise SystemExit(
            f"--teacher {cli!r} conflicts with the config's teacher: "
            f"{from_cfg!r} — refusing to guess which namespace to write"
        )
    teacher = cli or from_cfg or "greedy"
    if teacher not in TEACHERS:
        raise SystemExit(f"unknown teacher {teacher!r} (known: {TEACHERS})")
    return teacher


def _rs_params(cfg: dict) -> dict:
    """Validated rejection-teacher knobs. Hard requirement: the RS out_dir
    is a SEPARATE namespace from the greedy tree — the RS run must never
    overwrite the greedy pairs.jsonl or artifacts."""
    for key in ("greedy_results_root", "si_prompted_threshold"):
        if key not in cfg:
            raise SystemExit(
                f"teacher: rejection needs config key {key!r} "
                "(see configs/stage2_llama8b_rs.yaml)"
            )
    out = Path(cfg["out_dir"]).resolve()
    greedy = Path(cfg["greedy_results_root"]).resolve()
    if out == greedy:
        raise SystemExit(
            f"out_dir == greedy_results_root ({out}) — the rejection "
            "teacher writes to its OWN namespace (…/ifeval_rs), never "
            "over the greedy pairs/artifacts"
        )
    return {
        "k": int(cfg.get("k", 16)),
        "temperature": float(cfg.get("temperature", 1.0)),
        "seed": int(cfg.get("rs_seed", 0)),
        "min_yield": float(cfg.get("min_yield", 0.2)),
        "threshold": float(cfg["si_prompted_threshold"]),
        "greedy_root": greedy,
    }


def _greedy_prompted_strict(greedy_root: Path, si_cfg_id: str) -> float:
    """The SI's PROMPTED strict compliance from the greedy matrix
    (<greedy_root>/<si>/vector/results.json — the prompted arm is
    generated greedily so it is identical across the SI's arm cells;
    vector is the canonical one to read). Loud when absent: the RS
    ablation is defined relative to the greedy run."""
    path = greedy_root / si_cfg_id / "vector" / "results.json"
    if not path.is_file():
        raise SystemExit(
            f"greedy results missing for SI {si_cfg_id!r}: {path} — the "
            "rejection teacher selects SIs from the GREEDY matrix's "
            "prompted arm (D29); run the greedy matrix first"
        )
    for arm in json.loads(path.read_text()).get("arms") or []:
        if arm.get("arm") == "prompted":
            strict = arm.get("compliance_strict")
            if strict is None:
                raise SystemExit(
                    f"prompted arm in {path} has no compliance_strict"
                )
            return float(strict)
    raise SystemExit(f"no prompted arm recorded in {path}")


def rs_target_sis(cfg: dict) -> list[str]:
    """The config SIs whose greedy prompted strict < si_prompted_threshold
    — the ONLY SIs the rejection teacher runs on (config order kept, so
    sbatch array decoding matches the config list's relative order)."""
    params = _rs_params(cfg)
    return [
        si for si in (cfg.get("instructions") or [])
        if _greedy_prompted_strict(params["greedy_root"], si)
        < params["threshold"]
    ]


def _require_rs_si(cfg: dict, si_cfg_id: str) -> None:
    """Runtime consistency check: a cell handed an above-threshold SI fails
    loudly instead of running an ablation cell the design excludes."""
    params = _rs_params(cfg)
    targets = rs_target_sis(cfg)
    if si_cfg_id not in targets:
        strict = _greedy_prompted_strict(params["greedy_root"], si_cfg_id)
        raise SystemExit(
            f"SI {si_cfg_id!r} is not a rejection-teacher target: greedy "
            f"prompted strict {strict:.3f} >= threshold "
            f"{params['threshold']} (targets: {targets})"
        )


def _rs_stats_path(pairs_path: Path) -> Path:
    return pairs_path.with_name("yield_stats.json")


def _load_rs_stats(pairs_path: Path) -> dict:
    stats_path = _rs_stats_path(pairs_path)
    if not stats_path.is_file():
        raise SystemExit(
            f"RS pairs exist at {pairs_path} but {stats_path} is missing — "
            "a torn publish (stats are written FIRST, so pairs' existence "
            "must imply stats). Delete the SI's RS pairs dir and regenerate."
        )
    return json.loads(stats_path.read_text())


def _ensure_pairs_rejection(model, tok, instruction: Instruction,
                            fit_prompts: list[str], pairs_path: Path,
                            max_new_tokens: int,
                            rs: dict) -> tuple[list[Pair], dict]:
    """RS-aware _ensure_pairs: generate the SI's rejection-sampled pairs
    file if absent, under the same exclusive flock + existence recheck,
    publishing yield_stats.json FIRST and pairs.jsonl second (both
    atomic) so the pairs file's existence implies its stats."""
    if pairs_path.exists():
        return load_pairs(pairs_path), _load_rs_stats(pairs_path)
    with _pairs_lock(pairs_path):
        if pairs_path.exists():  # a peer generated it while we waited
            return load_pairs(pairs_path), _load_rs_stats(pairs_path)
        pairs, stats = generate_compliant_pairs(
            model, tok, instruction, fit_prompts,
            k=rs["k"], temperature=rs["temperature"],
            max_new_tokens=max_new_tokens, seed=rs["seed"],
        )
        _write_json(_rs_stats_path(pairs_path), stats)
        save_pairs(pairs, pairs_path)
    return pairs, stats


def _rs_yield_gate(stats: dict, rs: dict) -> str | None:
    """The low-yield reason string when the SI is infeasible at this k,
    else None. An SI the model cannot follow even under sampling (e.g.
    <20% of prompts yielded a compliant completion in k tries) has no
    usable teacher set — retraining on it would distill a handful of
    lucky samples, and looping k unboundedly is forbidden. The skip is
    itself a reportable finding."""
    if stats["yield_rate"] < rs["min_yield"]:
        return (
            f"rejection teacher infeasible at k={stats['k']}: "
            f"{100 * stats['yield_rate']:.0f}% yield "
            f"({stats['n_yielded']}/{stats['n_prompts']} prompts; "
            f"threshold {100 * rs['min_yield']:.0f}%) — the model cannot "
            "follow this instruction even under sampling"
        )
    return None


def _ensure_object(model, tok, instruction: Instruction, arm: str, cfg: dict,
                   cell_dir: Path, pairs_fn, log_fn=None,
                   config_extra: dict | None = None):
    """Load the cell's trained object if its artifact exists (the resume
    unit) — after verifying its provenance manifest; else train via the
    registry and publish manifest + artifact atomically (manifest first,
    so an artifact's existence implies its manifest). pairs_fn is lazy
    so a pure load never pays for pair generation."""
    path = _object_path(cell_dir, arm)
    manifest = _cell_manifest(cfg, arm, instruction.id, extra=config_extra)
    if path.exists():
        _check_manifest(path, manifest)
        print(f"[stage2] artifact exists, loading: {path}")
        return load_object(path, model, tok)
    kw = _train_kwargs(cfg, arm)
    obj = ARMS[arm].train(model, tok, instruction, pairs_fn(),
                          layer=cfg.get("layer", 15), log_fn=log_fn, **kw)
    cell_dir.mkdir(parents=True, exist_ok=True)
    _write_json(cell_dir / MANIFEST_NAME, manifest)
    _atomic_save(obj, path)
    return obj


def _wandb(cfg: dict, run_name: str, config: dict):
    from sisteer.eval.runner import make_wandb_logger

    project = cfg.get("wandb_project")
    if not project:
        return None, None
    return make_wandb_logger(project, run_name, config)


def _finish(run):
    if run is not None:
        run.finish()


def _write_json(path: Path, payload: dict) -> None:
    # temp + fsync + os.replace: results files are resume units,
    # so existence must mean fully-published.
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=1, default=str))
    _fsync_replace(tmp, path)


def _panel_kwargs(cfg: dict) -> dict:
    """capability_panel kwargs from the config's capability_ext block —
    ONE owner (the diagnostics copy had silently dropped the gsm8k
    max_new_tokens=512 default; quality cross-cutting fix 1)."""
    ext = cfg.get("capability_ext") or {}
    return dict(
        gsm8k_n=(ext.get("gsm8k_platinum") or {}).get("n", 200),
        mmlu_pro_n=(ext.get("mmlu_pro") or {}).get("n", 300),
        max_new_tokens=(ext.get("gsm8k_platinum") or {}).get(
            "max_new_tokens", 512),
    )


def _panel_to_cap(panel: dict, best_scale: float) -> dict:
    """Shape a capability_panel result into the stage-1-compatible
    capability block (the frontier's within-budget filter reads
    capability.unsteered.mmlu / capability.scale_<s>.mmlu)."""
    cap = {"unsteered": {"mmlu": panel[0.0]["mmlu"]}, "best_scale": best_scale}
    for s, metrics in panel.items():
        if s != 0.0:
            cap[f"scale_{s}"] = {"mmlu": metrics["mmlu"]}
    return cap


def _publish_cell_results(cfg, model, tok, si_cfg_id: str, arm: str,
                          arms_res, results_path: Path, obj,
                          extra: dict | None = None,
                          scales: list[float] | None = None) -> dict:
    """The shared publish tail of a full ifeval cell: summary +
    anti-steerable + capability block + atomic results.json. `scales`
    threads a per-arm grid (extra_scales) into the capability block;
    None keeps the config base grid."""
    from sisteer.eval.behavioral import anti_steerable_fraction

    instruction = resolve_si(si_cfg_id)
    baseline = next(a for a in arms_res if a.arm == "baseline")
    summary = {
        "instruction": si_cfg_id,           # cell layout uses CONFIG ids
        "canonical_id": instruction.id,     # registry id (alias map)
        "category": instruction.category,
        "method": arm,
        "layer": cfg.get("layer"),
        "arms": [a.summary() for a in arms_res],
        "anti_steerable": {
            str(a.scale): anti_steerable_fraction(a, baseline)
            for a in arms_res if a.arm == "steered"
        },
    }
    if extra:
        summary.update(extra)
    if cfg.get("capability", True):
        summary.update(_capability_block(cfg, model, tok, obj, arms_res,
                                         scales=scales))
    _write_json(results_path,
                {**summary, "records": [a.records for a in arms_res]})
    return summary


# ---------------------------------------------------------------- pairs (P1)


def run_pairs(cfg: dict, args) -> None:
    """One shared pairs.jsonl per SI so concurrent arm cells never
    race (plan 5.1). --teacher rejection is the RS teacher's P1 stage
    — rejection-sampled pairs + yield_stats.json into the RS
    namespace, for the sub-threshold SIs only."""
    teacher = _effective_teacher(cfg, args)
    if teacher == "rejection":
        if args.si is None:
            raise SystemExit("--suite pairs --teacher rejection needs --si")
        rs = _rs_params(cfg)
        _require_rs_si(cfg, args.si)
        model, tok = _load_model(cfg)
        fit, _ = _prompt_split(cfg)
        path = Path(cfg["out_dir"]) / args.si / "pairs.jsonl"
        pairs, stats = _ensure_pairs_rejection(
            model, tok, resolve_si(args.si), fit, path,
            cfg.get("max_new_tokens", 200), rs)
        print(f"[stage2] RS pairs ready: {path} — "
              f"{stats['n_yielded']}/{stats['n_prompts']} prompts yielded "
              f"({100 * stats['yield_rate']:.0f}%), mean samples/prompt "
              f"{stats['mean_samples_used']:.1f}, pos compliance "
              f"{stats['pos_compliance']}")
        reason = _rs_yield_gate(stats, rs)
        if reason:
            print(f"[stage2] WARNING {args.si}: {reason} — the matrix "
                  "cells will SKIP retraining this SI")
        return
    model, tok = _load_model(cfg)
    out = Path(cfg["out_dir"])
    max_new = cfg.get("max_new_tokens", 200)
    fit, _ = _prompt_split(cfg)
    if args.si is None:
        raise SystemExit("--suite pairs needs --si")
    instruction = resolve_si(args.si)
    path = out / args.si / "pairs.jsonl"
    pairs = _ensure_pairs(model, tok, instruction, fit, path, max_new)
    print(f"[stage2] pairs ready: {path} ({len(pairs)} pairs)")


# ---------------------------------------------------------------- ifeval (C1/A1/A8)


def _capability_block(cfg: dict, model, tok, obj, arms_res,
                      scales: list[float] | None = None) -> dict:
    from sisteer.eval.capability_ext import capability_panel

    steered = [a for a in arms_res if a.arm == "steered"]
    if scales is None:                 # callers with a per-arm grid pass it
        scales = cfg.get("scales") or [1.0]
    best = (max(steered, key=lambda a: a.compliance_strict).scale
            if steered else max(scales))
    panel = capability_panel(model, tok, obj, sorted({best, max(scales)}),
                             **_panel_kwargs(cfg))
    # Stage-1-compatible shape (the frontier's within-budget filter reads
    # capability.unsteered.mmlu / capability.scale_<s>.mmlu) + the full
    # extended panel alongside.
    return {"capability": _panel_to_cap(panel, best),
            "capability_ext": panel}


def _missing_scales(prev: dict, grid: list[float]) -> list[float]:
    """Grid points with no steered entry in a completed cell's results.
    Numeric match with stage2_frontier._ext_metric's tolerance, so a float
    formatting drift can never re-run an already-measured factor."""
    have = [a.get("scale") for a in (prev.get("arms") or [])
            if a.get("arm") == "steered"]
    return [s for s in grid
            if not any(h is not None and abs(s - h) < 1e-9 for h in have)]


def _extend_ifeval_cell(model, tok, cfg: dict, si_cfg_id: str, arm: str,
                        cell: Path, prev: dict, missing: list[float],
                        teacher: str = "greedy") -> dict:
    """Incremental factor addition: a COMPLETED cell whose arm grid
    grew (extra_scales) gets the new factors appended in place — the
    trained artifact is reused (NEVER retrained; a fresh training run
    would evaluate a different object than the recorded entries), only
    steered+stacking generation and the capability panel run, at the
    missing factors only, and the merge preserves every existing entry
    verbatim (atomic publish). This is deliberate about the
    manifest interaction: the factor grid is an eval-only knob OUTSIDE
    _cell_config_hash's subset (verified by test), so the identity check
    below still runs at full strength — no carve-out is needed or made."""
    from sisteer.eval.behavioral import (
        ArmResult,
        anti_steerable_fraction,
        eval_arm_obj,
    )

    instruction = resolve_si(si_cfg_id)
    results_path = cell / "results.json"
    if teacher != "greedy":
        raise SystemExit(
            f"incremental factor addition for {si_cfg_id}/{arm}: results "
            f"exist but factors {missing} are new, and this cell runs the "
            f"'{teacher}' teacher — the extension path supports the greedy "
            "tree only (RS manifests carry teacher config_extra it does not "
            "reconstruct). Extend deliberately or drop the RS grid change."
        )
    arms_meta = prev.get("arms")
    records = prev.get("records")
    if not arms_meta or records is None or len(records) != len(arms_meta):
        raise SystemExit(
            f"incremental factor addition for {si_cfg_id}/{arm}: "
            f"{results_path} lacks aligned arms/records "
            f"({len(arms_meta or [])} arms, {len(records or [])} record "
            "lists) — refusing to extend a malformed cell; delete it to "
            "recompute from scratch."
        )
    i_base = next((i for i, a in enumerate(arms_meta)
                   if a.get("arm") == "baseline"), None)
    if i_base is None:
        raise SystemExit(
            f"incremental factor addition for {si_cfg_id}/{arm}: no "
            "baseline arm recorded — cannot compute anti_steerable for the "
            "new factors; delete the cell to recompute from scratch."
        )
    path = _object_path(cell, arm)
    if not path.exists():
        raise SystemExit(
            f"incremental factor addition for {si_cfg_id}/{arm}: "
            f"{results_path} exists but the trained artifact {path.name} is "
            "missing — extending would require RETRAINING, which evaluates "
            "a different object than the recorded entries. Restore the "
            "artifact, or delete the cell to recompute everything."
        )
    _check_manifest(path, _cell_manifest(cfg, arm, instruction.id))
    print(f"[stage2] INCREMENTAL {si_cfg_id}/{arm}: grid grew — adding "
          f"factors {missing} to the completed cell (artifact reused, no "
          "retraining; existing entries preserved)")
    obj = load_object(path, model, tok)
    _, ev = _prompt_split(cfg)
    max_new = cfg.get("max_new_tokens", 200)
    run, _ = _wandb(cfg, f"s2-{si_cfg_id}-{arm}",
                    {"si": si_cfg_id, "arm": arm, "layer": cfg.get("layer"),
                     "teacher": teacher, "incremental_scales": missing})
    new_res = [
        eval_arm_obj(model, tok, instruction, ev, arm="steered", obj=obj,
                     scale=s, max_new_tokens=max_new)
        for s in missing
    ]
    new_res += [
        eval_arm_obj(model, tok, instruction, ev, arm="both", obj=obj,
                     scale=s, max_new_tokens=max_new)
        for s in missing
    ]
    # anti-steerable against the RECORDED baseline — the run these entries
    # will sit beside — never a regenerated one.
    baseline = ArmResult(
        arm="baseline", scale=0.0,
        compliance_strict=arms_meta[i_base]["compliance_strict"],
        compliance_loose=arms_meta[i_base]["compliance_loose"],
        records=records[i_base],
    )
    merged = dict(prev)
    merged["arms"] = list(arms_meta) + [a.summary() for a in new_res]
    merged["records"] = list(records) + [a.records for a in new_res]
    anti = dict(prev.get("anti_steerable") or {})
    for a in new_res:
        if a.arm == "steered":
            anti[str(a.scale)] = anti_steerable_fraction(a, baseline)
    merged["anti_steerable"] = anti
    if cfg.get("capability", True):
        # The budget/frontier logic can only admit an in-budget candidate
        # at a new factor if MMLU (+ the extended panel) is measured at
        # that scale — measure it here, at the missing factors only.
        from sisteer.eval.capability_ext import capability_panel

        ext = cfg.get("capability_ext") or {}
        panel = capability_panel(
            model, tok, obj, list(missing), include_unsteered=False,
            gsm8k_n=(ext.get("gsm8k_platinum") or {}).get("n", 200),
            mmlu_pro_n=(ext.get("mmlu_pro") or {}).get("n", 300),
            max_new_tokens=(ext.get("gsm8k_platinum") or {}).get(
                "max_new_tokens", 512),
        )
        cap = dict(prev.get("capability") or {})
        capx = dict(prev.get("capability_ext") or {})
        for s, metrics in panel.items():
            cap[f"scale_{s}"] = {"mmlu": metrics["mmlu"]}
            capx[str(s)] = metrics    # the json float-key convention
        merged["capability"] = cap    # best_scale: the ORIGINAL selection
        merged["capability_ext"] = capx
    merged.setdefault("incremental_extensions", []).append({
        "added_scales": list(missing),
        "code_sha": _code_sha(),
        "note": "grid grew after completion (extra_scales); artifact "
                "reused, existing entries preserved; capability.best_scale "
                "still names the original panel's selection",
    })
    _write_json(results_path, merged)
    if run is not None:
        import wandb

        table = wandb.Table(
            columns=["arm", "scale", "strict", "loose"],
            data=[[a.arm, a.scale, a.compliance_strict, a.compliance_loose]
                  for a in new_res],
        )
        run.log({"arms_incremental": table, **{
            f"compliance/{a.arm}@{a.scale}": a.compliance_strict
            for a in new_res
        }})
    _finish(run)
    summary = {k: v for k, v in merged.items() if k != "records"}
    print(f"[stage2] INCREMENTAL {si_cfg_id}/{arm} done: added {missing}")
    return summary | {"extended": True, "added_scales": list(missing)}


def _run_ifeval_cell(model, tok, cfg: dict, si_cfg_id: str, arm: str,
                     out: Path, teacher: str = "greedy") -> dict:
    """One full ifeval cell: train (or resume) the arm's object, evaluate
    the scale grid with stacking, publish results.json."""
    from sisteer.eval.behavioral import eval_instruction_obj

    instruction = resolve_si(si_cfg_id)
    cell = out / si_cfg_id / arm
    results_path = cell / "results.json"
    grid = _arm_scales(cfg, arm)
    if results_path.exists() and not cfg.get("overwrite", False):
        prev = json.loads(results_path.read_text())
        missing = _missing_scales(prev, grid)
        if missing:
            # The grid grew after this cell completed (extra_scales):
            # run the missing factors incrementally instead of skipping
            # them forever (the old behavior) or recomputing 25 min of
            # settled measurements (overwrite still forces that).
            return _extend_ifeval_cell(model, tok, cfg, si_cfg_id, arm,
                                       cell, prev, missing, teacher=teacher)
        print(f"[stage2] skip {si_cfg_id}/{arm}: {results_path} exists (resume)")
        prev.pop("records", None)
        return prev | {"skipped": True}

    # ifeval has a P1 pairs stage: a missing shared pairs file is an
    # ordering error, never regenerated per-cell. The rejection teacher
    # mirrors this exactly with its own P1-RS stage: the k-sample
    # generation is far too expensive to duplicate or serialize across
    # 12 concurrent arm cells.
    pairs_path = out / si_cfg_id / "pairs.jsonl"
    teacher_block = None
    config_extra = None
    if teacher == "rejection":
        rs = _rs_params(cfg)
        _require_rs_si(cfg, si_cfg_id)
        if not pairs_path.exists():
            raise SystemExit(
                f"shared RS pairs file missing: {pairs_path} — run the "
                "P1-RS pairs stage first (stage2_run.py --suite pairs "
                "--teacher rejection). "
                "Per-cell generation is disabled here by design (D22#5)."
            )
        stats = _load_rs_stats(pairs_path)
        greedy_prompted = _greedy_prompted_strict(rs["greedy_root"],
                                                  si_cfg_id)
        reason = _rs_yield_gate(stats, rs)
        if reason:
            _write_json(out / si_cfg_id / "SKIPPED_LOW_YIELD.json", {
                "instruction": si_cfg_id,
                "reason": reason,
                "yield_stats": stats,
                "greedy_prompted_strict": greedy_prompted,
            })
            print(f"[stage2] SKIP {si_cfg_id} (all arms): {reason}")
            return {"instruction": si_cfg_id, "method": arm,
                    "skipped_low_yield": True, "reason": reason}
        teacher_block = {
            "mode": "rejection",
            "k": stats["k"],
            "temperature": stats["temperature"],
            "seed": stats["seed"],
            "yield_stats": stats,
            # the greedy-vs-rs prompted context: the greedy matrix's
            # prompted strict for this SI (this cell's own prompted arm is
            # in `arms` below, evaluated identically to the greedy run)
            "greedy_prompted_strict": greedy_prompted,
        }
        config_extra = {"teacher": "rejection", "k": stats["k"],
                        "temperature": stats["temperature"],
                        "rs_seed": stats["seed"]}

    run_prefix = "s2-rs" if teacher == "rejection" else "s2"
    run, log_fn = _wandb(cfg, f"{run_prefix}-{si_cfg_id}-{arm}",
                         {"si": si_cfg_id, "arm": arm,
                          "layer": cfg.get("layer"), "teacher": teacher})
    _, ev = _prompt_split(cfg)
    max_new = cfg.get("max_new_tokens", 200)
    obj = _ensure_object(
        model, tok, instruction, arm, cfg, cell,
        pairs_fn=lambda: _require_pairs(pairs_path),
        log_fn=log_fn,
        config_extra=config_extra,
    )
    # The arm's grid may carry per-arm extras; the stacking ("both")
    # point stays pinned to the BASE grid's median so an added factor never
    # moves it out from under the rest of the matrix.
    base_grid = cfg.get("scales") or [1.0]
    arms_res = eval_instruction_obj(
        model, tok, instruction, ev, obj, grid,
        include_stacking=True, stack_scale=base_grid[len(base_grid) // 2],
        max_new_tokens=max_new,
    )
    summary = _publish_cell_results(
        cfg, model, tok, si_cfg_id, arm, arms_res, results_path, obj,
        extra=({"teacher": teacher_block} if teacher_block is not None
               else None),
        scales=grid)
    if run is not None:
        import wandb

        table = wandb.Table(
            columns=["arm", "scale", "strict", "loose"],
            data=[[a.arm, a.scale, a.compliance_strict, a.compliance_loose]
                  for a in arms_res],
        )
        run.log({"arms": table, **{
            f"compliance/{a.arm}@{a.scale}": a.compliance_strict
            for a in arms_res
        }})
    _finish(run)
    print(json.dumps({k: v for k, v in summary.items()
                      if k not in ("arms",)}, indent=1, default=str))
    return summary


def run_ifeval(cfg: dict, args) -> None:
    # Validate BEFORE the (multi-minute) model load: the qwen-spot config
    # ships with `arms: []` until Phase 4 fills it from the A1 frontier,
    # and that placeholder must error loudly and instantly (by design).
    if args.all_cells:
        arms = cfg.get("arms") or []
        if not arms:
            raise SystemExit(
                "config `arms:` is EMPTY — for the qwen spot-check it is "
                "filled from the A1 frontier at Phase 4 (loud by design)"
            )
    elif not args.si or not args.arm:
        raise SystemExit("--suite ifeval needs --si and --arm (or --all-cells)")
    teacher = _effective_teacher(cfg, args)
    if teacher == "rejection":
        # Validate the RS setup BEFORE the model load: namespace separation
        # + the live sub-threshold selection (loud on a stale enumeration).
        targets = rs_target_sis(cfg)
        print(f"[stage2] rejection-teacher targets (greedy prompted strict "
              f"< {cfg['si_prompted_threshold']}): {targets}")
        if not args.all_cells:
            _require_rs_si(cfg, args.si)
    model, tok = _load_model(cfg)
    out = Path(cfg["out_dir"])

    if args.all_cells:
        sis = targets if teacher == "rejection" else cfg["instructions"]
        for si in sis:
            for arm in arms:
                _run_ifeval_cell(model, tok, cfg, si, arm, out,
                                 teacher=teacher)
        return
    _run_ifeval_cell(model, tok, cfg, args.si, args.arm, out,
                     teacher=teacher)


# ---------------------------------------------------------------- composition (C4b/A4)


def run_composition(cfg: dict, args) -> None:
    from sisteer.eval.composition import canonical_order, default_targets
    from sisteer.extraction.compose import (
        compose_additive,
        token_compose_fn,
        train_and_token,
    )

    comp_fn = get_runner("composition")
    kind = args.arm
    if args.canary:
        kind = kind or "token_faithful"  # the canary exercises the <and> path
    if kind not in ("vector", "token_faithful"):
        raise SystemExit("--suite composition needs --arm vector|token_faithful "
                         "(or --canary); D4 fixes the kinds")

    model, tok = _load_model(cfg)
    out = Path(cfg["out_dir"])
    if args.canary:
        # QUARANTINE the whole canary namespace: the canary trains on <=10
        # fit prompts and ONE target (an <and> trained on 1 pair, not 8),
        # so any of its artifacts (pairs/objects/and_token/results) left in
        # the A4 root would be adopted verbatim by the array's
        # artifact-exists resume.
        out = out / "canary"
    tcfg = cfg.get("targets") or {}
    targets = default_targets(seed=tcfg.get("seed", 0))
    n_prompts = cfg.get("n_prompts_per_target", 50)
    if args.canary:
        targets = {"seen": [targets["seen"][0]]}
        n_prompts = min(n_prompts, 10)
    fit, ev = _prompt_split(cfg)
    fit = fit[:n_prompts] or fit
    ev = ev[:n_prompts] or ev
    max_new = cfg.get("max_new_tokens", 200)
    behaviors = sorted({n for group in targets.values() for t in group for n in t})

    run, log_fn = _wandb(cfg, f"s2-comp-{kind}" + ("-canary" if args.canary else ""),
                         {"kind": kind, "behaviors": behaviors})

    def _pairs_for(instruction: Instruction) -> list[Pair]:
        return _ensure_pairs(model, tok, instruction, fit,
                             out / "pairs" / f"{instruction.id}.jsonl", max_new)

    # 1. per-behavior objects
    per_behavior = {}
    for name in behaviors:
        si = BEHAVIOR_SIS[name]
        per_behavior[name] = _ensure_object(
            model, tok, si, kind, cfg, out / "objects" / kind / name,
            pairs_fn=lambda si=si: _pairs_for(si), log_fn=log_fn,
        )

    # 2. monolithic joint objects per target — keyed by the PREREGISTERED
    # canonical order (sorted names): eval_composition looks up
    # monolithic_objs[canonical_order(names)] and refuses non-canonical
    # keys, and the joined instruction is rendered in that same order so
    # the trained text is the text the arm is scored against.
    monolithic = {}
    for group in targets.values():
        for names in group:
            canon = canonical_order(names)
            if canon in monolithic:
                continue
            joint = joined_behavior_instruction(list(canon))
            monolithic[canon] = _ensure_object(
                model, tok, joint, kind, cfg,
                out / "objects" / kind / ("monolithic_" + "+".join(canon)),
                pairs_fn=lambda joint=joint: _pairs_for(joint), log_fn=log_fn,
            )

    # 3. the composition operator
    if kind == "vector":
        compose_fn = compose_additive
    else:
        import torch

        acfg = cfg.get("and_token") or {}
        and_path = out / "objects" / "and_token.pt"
        if and_path.exists():
            blob = torch.load(and_path, map_location="cpu", weights_only=True)
            from torch import nn

            and_param = nn.Parameter(blob["param"])
        else:
            seen_pairs = [tuple(t) for t in targets["seen"] if len(t) == 2]
            pairs_by_target = {
                t: _pairs_for(joined_behavior_instruction(list(t)))
                for t in seen_pairs
            }
            and_param = train_and_token(
                model, tok, per_behavior, seen_pairs,
                pairs_by_target=pairs_by_target,
                epochs=acfg.get("epochs", 6),
                batch_size=acfg.get("batch_size", 2),
                max_len=acfg.get("max_len", 768),
                log_fn=log_fn,
            )
            # atomic publish: the <and> token is an existence-based
            # resume unit like every other artifact.
            and_path.parent.mkdir(parents=True, exist_ok=True)
            and_tmp = and_path.with_name(and_path.name + ".tmp")
            torch.save({"kind": "and_token",
                        "param": and_param.detach().float().cpu()}, and_tmp)
            _fsync_replace(and_tmp, and_path)
        device = next(model.parameters()).device
        and_param.data = and_param.data.to(device)
        compose_fn = token_compose_fn(and_param)

    # 4. eval (item-level resume inside eval_composition)
    records = comp_fn(
        model, tok, targets=targets, prompts=ev, out_dir=out / kind,
        per_behavior_objs=per_behavior, compose_fn=compose_fn,
        monolithic_objs=monolithic, max_new_tokens=max_new,
    )
    if run is not None:
        for rec in records:
            run.log({
                f"composition/{rec['arm']}/joint_strict": rec["joint_strict"],
                "k": rec["k"],
            })
    _finish(run)
    print(f"[stage2] composition {kind}: {len(records)} new target-arm records "
          f"-> {out / kind / 'results.jsonl'}")


# ---------------------------------------------------------------- lifbench (removed)


def run_lifbench_removed(cfg: dict, args) -> None:
    """Loud tombstone for a stale command line. The lifbench suite was
    REMOVED from the campaign: LIFBench places the operative
    instruction AFTER the long content block, so it measures retrieval /
    attending back over context, not the instruction DECAY this project
    studies. Its config and sbatch are gone; eval/lifbench.py is kept for
    reference but is not a run suite. A stale dispatch must never
    silently no-op."""
    raise SystemExit(
        "the lifbench suite was removed from the S2 eval campaign "
        "(D30: instruction-after-context = retrieval, not instruction "
        "decay) -- there is no config or sbatch for it, and its results "
        "are not reported"
    )


# ---------------------------------------------------------------- ladder (C4a/A7)


def run_ladder(cfg: dict, args) -> None:
    from sisteer.eval.behavioral import eval_instruction_obj
    from sisteer.extraction.ladder import train_gkd, train_mc_kl

    if not args.cell or args.rung not in (2, 3):
        raise SystemExit("--suite ladder needs --cell <si> --rung {2,3} "
                         "(rung 1 reuses A1's distill_soft cells)")
    si = resolve_si(args.cell)
    model, tok = _load_model(cfg)
    out = Path(cfg["out_dir"]) / args.cell / f"rung{args.rung}"
    results_path = out / "results.json"
    if results_path.exists() and not cfg.get("overwrite", False):
        print(f"[stage2] skip ladder {args.cell} rung {args.rung}: results exist")
        return
    fit, ev = _prompt_split(cfg)
    max_new = cfg.get("max_new_tokens", 200)
    layer = cfg.get("layer", 15)
    rungs = {r["id"]: r for r in (cfg.get("rungs") or []) if isinstance(r, dict)}
    rcfg = rungs.get(args.rung, {})
    kw = _train_kwargs(cfg, "gkd" if args.rung == 3 else "default")
    arm_name = f"rung{args.rung}"
    manifest = _cell_manifest(cfg, arm_name, si.id,
                              extra={"rung": args.rung, "rung_cfg": rcfg,
                                     "train_kwargs": kw})
    run, log_fn = _wandb(cfg, f"s2-ladder-{args.cell}-r{args.rung}",
                         {"cell": args.cell, "rung": args.rung})

    vec_path = out / "vector.safetensors"
    if vec_path.exists():
        _check_manifest(vec_path, manifest)
        obj = load_object(vec_path, model, tok)
    else:
        # Rungs 2 and 3 are BUDGET-MATCHED: both resample a fresh
        # n_samples trajectories per prompt per epoch and share the same
        # optimizer loop — rung 2 via train_mc_kl (the old
        # sample-once-then-train_distill path confounded trajectory source
        # with data diversity), rung 3 via train_gkd with the SAME
        # n_samples. Both record n_trajectories_seen / n_optimizer_steps
        # in the vector metadata.
        common_kw = dict(
            epochs=kw.get("epochs", 3),
            batch_size=kw.get("batch_size", 4),
            max_len=kw.get("max_len", 768),
            n_samples=rcfg.get("n_samples", 3),
            max_new_tokens=max_new,
            log_fn=log_fn,
        )
        if args.rung == 2:
            sv = train_mc_kl(
                model, tok, si, fit, layer,
                temperature=rcfg.get("temperature", 1.0),
                **common_kw,
            )
        else:
            sv = train_gkd(
                model, tok, si, fit, layer,
                rollout_temperature=rcfg.get("rollout_temperature", 1.0),
                **common_kw,
            )
        obj = VectorObject(sv)
        _write_json(out / MANIFEST_NAME, manifest)
        _atomic_save(obj, vec_path)

    arms_res = eval_instruction_obj(
        model, tok, si, ev, obj, cfg.get("scales") or [1.0],
        include_stacking=True, max_new_tokens=max_new,
    )
    # The budget-match evidence, surfaced per cell: the trainers
    # persist both counters in the SteeringVector metadata (sidecar), so a
    # resumed cell reports the same numbers as a fresh one.
    budget_meta = getattr(obj.sv, "metadata", None) or {}
    summary = {
        "instruction": args.cell,
        "canonical_id": si.id,
        "method": f"rung{args.rung}",
        "layer": layer,
        "budget": {
            "n_trajectories_seen": budget_meta.get("n_trajectories_seen"),
            "n_optimizer_steps": budget_meta.get("n_optimizer_steps"),
        },
        "arms": [a.summary() for a in arms_res],
    }
    if cfg.get("capability", True):
        summary.update(_capability_block(cfg, model, tok, obj, arms_res))
    _write_json(results_path,
                {**summary, "records": [a.records for a in arms_res]})
    if run is not None:
        run.log({f"compliance/{a.arm}@{a.scale}": a.compliance_strict
                 for a in arms_res})
    _finish(run)
    print(json.dumps({k: v for k, v in summary.items() if k != "arms"},
                     indent=1, default=str))


# ---------------------------------------------------------------- main


SUITE_RUNNERS = {
    "pairs": run_pairs,
    "ifeval": run_ifeval,
    "composition": run_composition,
    "lifbench": run_lifbench_removed,  # tombstone -- raises, never runs
    "ladder": run_ladder,
}


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Stage-2 suite driver (see module docstring / sbatch headers)")
    ap.add_argument("--config", required=True)
    ap.add_argument("--suite", required=True, choices=sorted(SUITE_RUNNERS))
    ap.add_argument("--si", default=None,
                    help="config SI id (pairs/ifeval); a registered instruction id")
    ap.add_argument("--arm", default=None,
                    help="method arm (ifeval) or object kind (composition)")
    ap.add_argument("--cell", default=None, help="ladder cell (config SI id)")
    ap.add_argument("--rung", type=int, default=None, help="ladder rung (2|3)")
    ap.add_argument("--canary", action="store_true")
    ap.add_argument("--all-cells", action="store_true",
                    help="ifeval: run every (instruction x arm) in the config")
    ap.add_argument("--teacher", default=None, choices=list(TEACHERS),
                    help="pairs/ifeval teacher: greedy (default; the "
                         "unfiltered on-policy teacher) or rejection (D29: "
                         "compliance-filtered rejection-sampled pos, RS "
                         "namespace; must match the config's teacher key)")
    args = ap.parse_args()

    cfg = _load_cfg(args.config)
    SUITE_RUNNERS[args.suite](cfg, args)


if __name__ == "__main__":
    main()
