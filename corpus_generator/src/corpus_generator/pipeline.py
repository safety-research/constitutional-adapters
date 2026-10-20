"""Corpus orchestration: drive the stage graphs (via each corpus module's
CorpusDef), manage UNION.json and top-up sub-DAGs, emit the run manifest +
status heartbeat. Everything corpus-specific lives in advice.py/stories.py."""

from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path

import anthropic as _anthropic_sdk

from .advice import ADVICE, SCENARIO_WAVES
from .batch import WaveEngine, _utcnow, wire_tag
from .config import RunConfig
from .constitution import load as load_constitution
from .io import atomic_write, iter_jsonl, locked_json_update, read_json
from .stages import CorpusDef, OutputRow, StageCtx, run_stage
from .stories import STORIES, STORY_WAVES
from .templates import TemplateStore, repo_root

CORPORA: dict[str, CorpusDef] = {"advice": ADVICE, "stories": STORIES}


# ------------------------------------------------------------------ totals


def dag_expanded_totals(cfg: RunConfig) -> dict:
    """Configured request totals per corpus (assumes exact fanout; the report
    states achieved-vs-configured afterwards)."""
    a = cfg.advice
    scenarios = a.n_principles * a.n_types * a.n_scenarios_per_call
    advice_total = 1 + a.n_principles + a.n_principles * a.n_types + SCENARIO_WAVES * scenarios
    s = cfg.stories
    s3 = s.n_calls
    stories_n = s3 * s.stories_per_call
    stories_total = 1 + s.n_categories + s3 + STORY_WAVES * stories_n
    return {
        "advice_requests": advice_total,
        "advice_scenarios": scenarios,
        "stories_requests": stories_total,
        "stories_drafts": stories_n,
        "total": advice_total + stories_total,
    }


def slack_message(cfg: RunConfig, corpus: str) -> str:
    t = dag_expanded_totals(cfg)
    n = {
        "advice": t["advice_requests"],
        "stories": t["stories_requests"],
        "all": t["total"],
    }[corpus]
    return (
        f"Batch API: ~{n:,} {cfg.engine.model} requests "
        f"(corpus_generator {cfg.run_id}, corpus={corpus}) submitted up front and "
        f"harvested together (async, separate batch pool — not live concurrents)."
    )


# ------------------------------------------------------------------ context


def resolve_environment(cfg: RunConfig, root: Path | None = None, on_event=print) -> RunConfig:
    """Resolve the smoke-decided switches (nonce branch, structured scores)
    and FREEZE them into the run namespace (run_dir/ENVIRONMENT.json) before
    any wave is built.

    `corpusgen smoke` overwrites the global data/manifests/environment.json at
    will, so the global file is read exactly once per run — at first resolve.
    Every later resume of the same run_id reads the frozen copy, and an
    explicit config value that contradicts the frozen verdict is a hard error
    (a flipped reconciliation mode mid-run is how a no-nonce journal turns
    into a double submit). Each verdict has its OWN null sentinel and is
    resolved independently."""
    root = root or repo_root()
    frozen_path = cfg.run_dir(root) / "ENVIRONMENT.json"
    if frozen_path.exists():
        frozen = read_json(frozen_path)
    else:
        env_path = cfg.resolve_data_dir(root) / "manifests" / "environment.json"
        env = read_json(env_path) if env_path.exists() else None
        if env is None and (cfg.engine.nonce_metadata is None
                            or cfg.engine.use_structured_scores is None):
            on_event(
                "WARNING: no smoke verdict (data/manifests/environment.json missing) — "
                "freezing the CONSERVATIVE defaults (no-nonce reconciliation, "
                "prompt-enforced scores); run `corpusgen smoke` first to enable "
                "exactly-once submission"
            )
        frozen = {
            # explicit config pins win at freeze time; each verdict resolves
            # independently against its own null sentinel
            "nonce_metadata": (
                cfg.engine.nonce_metadata
                if cfg.engine.nonce_metadata is not None
                else bool(env.get("nonce_metadata", False)) if env else False
            ),
            "use_structured_scores": (
                cfg.engine.use_structured_scores
                if cfg.engine.use_structured_scores is not None
                else bool(env.get("output_config_on_batch", False)) if env else False
            ),
            "source": str(env_path) if env is not None else "conservative defaults (no smoke verdict)",
            "frozen_at": _utcnow(),
        }
        frozen_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write(frozen_path, json.dumps(frozen, indent=1))

    for field in ("nonce_metadata", "use_structured_scores"):
        explicit = getattr(cfg.engine, field)
        if explicit is not None and bool(explicit) != bool(frozen[field]):
            raise RuntimeError(
                f"run {cfg.run_id}: config sets {field}={explicit} but the run "
                f"froze {field}={frozen[field]} at first resolve "
                f"({frozen_path}) — verdicts are immutable within a run; start "
                f"a new run_id to change them"
            )
    cfg.engine.nonce_metadata = bool(frozen["nonce_metadata"])
    cfg.engine.use_structured_scores = bool(frozen["use_structured_scores"])
    on_event(
        f"run verdicts (frozen): nonce_metadata={cfg.engine.nonce_metadata} "
        f"(reconciliation branch), use_structured_scores={cfg.engine.use_structured_scores}"
    )
    return cfg


def build_ctx(cfg: RunConfig, corpus: str, root: Path | None = None) -> tuple[StageCtx, Path]:
    root = root or repo_root()
    # variant runs swap the injected document (constitution_dir -> the
    # value-neutral placebo) and overlay divergent templates; the stage code
    # is identical either way
    md, sections, _snap = load_constitution(cfg.resolve_constitution_dir(root))
    run_dir = cfg.run_dir(root)
    ctx = StageCtx(
        cfg=cfg,
        constitution_md=md,
        sections=sections,
        templates=TemplateStore(
            root / "prompts", overlay_dir=cfg.resolve_prompts_overlay(root)
        ),
        corpus_dir=run_dir / corpus,
    )
    return ctx, run_dir


def make_engine(cfg: RunConfig, client, **overrides) -> WaveEngine:
    # EngineConfig mirrors WaveEngine's tuning knobs one-for-one; the excluded
    # fields are stage-runner / request-builder concerns, not engine ones.
    kw = cfg.engine.model_dump(
        exclude={"repair_rounds", "min_slots_frac", "use_structured_scores"}
    )
    kw["nonce_metadata"] = bool(kw["nonce_metadata"])
    kw["run_tag"] = wire_tag(cfg.run_id)  # run identity on every wire custom_id
    kw.update(overrides)
    return WaveEngine(client, **kw)


# ------------------------------------------------------------------ topups


def load_topups(run_dir: Path, corpus: str) -> list[dict]:
    cdir = run_dir / corpus
    if not cdir.is_dir():
        return []
    return [read_json(p) for p in sorted(cdir.glob("TOPUP-*.json"))]


# --corpus all runs both corpora (and their top-up extension sub-DAGs) as
# threads in one process; status.json and UNION.json are the only shared
# read-modify-write artifacts, so updates go through a file lock.


def _heartbeat(run_dir: Path, corpus: str, stage: str) -> None:
    def _set(status: dict) -> None:
        status[corpus] = {"stage": stage, "ts": _utcnow()}

    locked_json_update(run_dir / "status.json", _set, dict, tolerate_corrupt=True)


def _register_union(run_dir: Path, corpus: str, records_path: Path) -> None:
    rel = str(records_path.relative_to(run_dir))

    def _add(union: dict) -> None:
        if rel not in union[corpus]:
            union[corpus].append(rel)

    locked_json_update(
        run_dir / "UNION.json", _add, lambda: {"advice": [], "stories": []}
    )


# ------------------------------------------------------------------ running


def run_corpus(
    cfg: RunConfig,
    corpus: str,
    engine: WaveEngine,
    root: Path | None = None,
    on_event=print,
) -> None:
    cd = CORPORA[corpus]
    ctx, run_dir = build_ctx(cfg, corpus, root)
    exts = [cd.ext_cls(**t) for t in load_topups(run_dir, corpus)]

    def _run_dag(dag_engine, dag_ctx, specs, upstream: dict[str, list[OutputRow]], suffix: str):
        dag_ctx.upstream = dict(upstream)
        for spec in specs:
            _heartbeat(run_dir, corpus, spec.name)
            on_event(f"[{corpus}] stage {spec.name}")
            rows = run_stage(
                dag_engine, dag_ctx, spec,
                repair_rounds=cfg.engine.repair_rounds,
                min_slots_frac=cfg.engine.min_slots_frac,
                on_event=on_event,
            )
            dag_ctx.upstream[spec.logical] = rows
        records_path = cd.materialize(dag_ctx, suffix=suffix)
        _register_union(run_dir, corpus, records_path)

    _run_dag(engine, ctx, cd.stages(None), {}, suffix="")
    base_outputs = dict(ctx.upstream)

    # extension sub-DAGs are independent of each other — run them concurrently
    ext_upstream = {k: base_outputs[k] for k in cd.ext_upstream}
    errors: list[BaseException] = []

    def _ext_target(ext) -> None:
        try:
            ext_ctx, _ = build_ctx(cfg, corpus, root)
            _run_dag(
                make_engine(cfg, engine.client), ext_ctx,
                cd.stages(ext), ext_upstream, suffix=ext.suffix,
            )
        except BaseException as exc:  # noqa: BLE001 - surfaced below
            errors.append(exc)
            on_event(f"[{corpus}{ext.suffix}] FAILED: {exc}")

    threads = [
        threading.Thread(target=_ext_target, args=(e,), name=f"corpusgen-{corpus}{e.suffix}")
        for e in exts
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if errors:
        raise errors[0]

    _heartbeat(run_dir, corpus, "done")
    write_manifest(cfg, run_dir, root=root)


def run(cfg: RunConfig, corpus: str, client, root: Path | None = None, on_event=print) -> None:
    """corpusgen run: drive one corpus, or both concurrently for --corpus all."""
    # idempotent: freezes the smoke verdicts into the run namespace on first
    # call, then re-reads the frozen copy — never the mutable global file
    cfg = resolve_environment(cfg, root=root, on_event=on_event)
    if corpus != "all":
        run_corpus(cfg, corpus, make_engine(cfg, client), root=root, on_event=on_event)
        return
    errors: list[BaseException] = []

    def _target(name):
        try:
            # each thread gets its own engine (client is thread-safe)
            run_corpus(cfg, name, make_engine(cfg, client), root=root, on_event=on_event)
        except BaseException as exc:  # noqa: BLE001 - surfaced below
            errors.append(exc)
            on_event(f"[{name}] FAILED: {exc}")

    threads = [
        threading.Thread(target=_target, args=(c,), name=f"corpusgen-{c}")
        for c in CORPORA
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if errors:
        raise errors[0]


# ---------------------------------------------------------------- manifest


def write_manifest(cfg: RunConfig, run_dir: Path, root: Path | None = None) -> dict:
    """Roll every stage stamp + environment fact into the run manifest.
    The report reads only manifests + artifacts, never the API."""
    root = root or repo_root()
    stages: dict[str, dict] = {}
    batch_ids: list[str] = []
    for stamp in sorted(run_dir.glob("*/**/STAGE.json")):
        s = read_json(stamp)
        stages[f"{s['corpus']}/{s['stage']}"] = s
    for stamp in sorted(run_dir.glob("*/**/STAMP.json")):
        s = read_json(stamp)
        batch_ids.extend(s.get("batch_ids", {}).values())

    snap_path = cfg.resolve_constitution_dir(root) / "SNAPSHOT.json"
    const = read_json(snap_path) if snap_path.exists() else {}
    tok_manifest_path = cfg.resolve_data_dir(root) / "tokenizers" / "MANIFEST.json"
    tokenizers = read_json(tok_manifest_path) if tok_manifest_path.exists() else {}
    # prefer the run-frozen verdicts (what this run actually used) over the
    # mutable global file
    env_path = run_dir / "ENVIRONMENT.json"
    if not env_path.exists():
        env_path = cfg.resolve_data_dir(root) / "manifests" / "environment.json"
    environment = read_json(env_path) if env_path.exists() else {}
    # the registry of the store the run actually rendered from (overlay wins)
    templates = TemplateStore(
        root / "prompts", overlay_dir=cfg.resolve_prompts_overlay(root)
    ).registry()

    manifest = {
        "run_id": cfg.run_id,
        "model": cfg.engine.model,
        "sdk_version": _anthropic_sdk.__version__,
        "config_hash": hashlib.sha256(
            cfg.model_dump_json().encode("utf-8")
        ).hexdigest(),
        "config": json.loads(cfg.model_dump_json()),
        # for a control run this is the PLACEBO snapshot: its dir, sha256,
        # and source provenance travel with every record's lineage too
        "constitution_dir": cfg.constitution_dir,
        "constitution_sha256": const.get("sha256"),
        "constitution_source_sha256": const.get("source_sha256"),
        "divergences": cfg.divergences,
        "template_hashes": templates,
        "tokenizers": tokenizers,
        "environment": environment,  # smoke verdicts: nonce branch, output_config
        "stages": stages,
        "batch_ids": sorted(set(batch_ids)),
        "written_at": _utcnow(),
    }
    atomic_write(run_dir / "manifest.json", json.dumps(manifest, indent=1))
    return manifest


def load_union_records(run_dir: Path, corpus: str) -> list[dict]:
    """The filter's input: the manifest-declared union of record files."""
    union_path = run_dir / "UNION.json"
    if not union_path.exists():
        raise FileNotFoundError(
            f"{union_path} missing — run the {corpus} pipeline first"
        )
    union = read_json(union_path)
    records = []
    for rel in union.get(corpus, []):
        records.extend(iter_jsonl(run_dir / rel))
    return records
