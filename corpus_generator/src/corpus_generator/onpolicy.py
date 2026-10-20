"""On-policy step-5 variant of the advice corpus (2026-09-01).

The wave-C response (system prompt + constitution-section injection + user
message -> assistant response) is sampled from open-weight instruction-tuned
models on cluster GPUs (vLLM) instead of Opus; a final Opus reviewer SCORES
the responses for constitution adherence with the production e_score wave
verbatim — but NEVER revises them. The rewrite step (wave D) is deliberately
absent: responses must stay on the target models' own distribution, because
the corpus feeds steering-vector learning against those models.

Prompt pool: EXACTLY the prompts behind an existing frozen final
(final-production-20260831-v2.jsonl, 1,960 records). The extraction rebuilds
each record's generation-time system prompt through the SAME code path wave C
used — the c_injection template rendered with the section text of the
record's ``section_id`` from the frozen constitution snapshot, appended to
the record's clean system prompt (or standing alone when the record has
none) — so injection equivalence with wave C is structural, and tested.

Pipeline: ``extract`` (prompt-set artifact, sha-stamped) -> cluster sampling
(scripts/sample_onpolicy_vllm.py, outside this module) -> ``score`` (one
Batch wave per model, production e_score template + full-constitution
context) -> materialize (AdviceRecord-shaped, rewrite fields None) ->
``filter_corpus`` with POST-HOC floors calibrated on the observed cross-model
score distribution (hard gates unchanged; register caps deliberately off —
the prompt pool already passed v2's caps and is fixed here by design).
"""

from __future__ import annotations

import datetime
import hashlib
import json
from pathlib import Path

from .advice import ADVICE_SCORE_SCHEMA
from .batch import WaveRequest
from .config import RunConfig, filter_report_path
from .constitution import section_by_id
from .io import atomic_write, iter_jsonl, read_json, sha256_file, write_jsonl_atomic
from .parsing import parse_score_object
from .pipeline import _heartbeat, _register_union, build_ctx, write_manifest
from .schema import AdviceRecord, Lineage, validate_custom_id
from .stages import (
    OutputRow,
    StageCtx,
    StageSpec,
    const_user_content,
    repair_request,
    run_stage,
    structured_format,
)
from .templates import TemplateStore, repo_root

INJECTION_TEMPLATE = "advice/c_injection.md"
SCORE_TEMPLATE = "advice/e_score.md"
PROMPT_VERSION = "onpolicy-v1"


def _utcnow() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def _resolve(root: Path, p: str) -> Path:
    q = Path(p)
    return q if q.is_absolute() else root / q


# ------------------------------------------------------------------ extract


def build_sampling_prompt(
    record: dict, templates: TemplateStore, constitution_md: str, sections
) -> str:
    """The generation-time system prompt for one source record — EXACTLY the
    string wave C submitted: clean system + '\\n\\n' + rendered injection, or
    the injection alone for no-system records (mirrors advice._c_spec)."""
    section = section_by_id(sections, record["section_id"])
    injection = templates.render(
        INJECTION_TEMPLATE, section_text=section.text(constitution_md)
    )
    clean_system = record.get("system")
    return f"{clean_system}\n\n{injection}" if clean_system else injection


def extract_prompt_set(
    cfg: RunConfig, root: Path | None = None, on_event=print
) -> dict:
    """Source final -> versioned prompt-set artifact + manifest (sha-stamped).

    One row per source record, sorted by the source lineage custom_id
    (deterministic; re-extraction is byte-identical). The ``system`` field is
    the SAMPLING system prompt (injection included); ``clean_system`` is what
    the final record stores (and what the scorer will see, exactly as the
    production e wave saw the clean b3 system)."""
    root = root or repo_root()
    if cfg.onpolicy is None:
        raise ValueError("config has no `onpolicy` section")
    from .constitution import load as load_constitution

    md, sections, snap = load_constitution(root / "constitution")
    templates = TemplateStore(root / "prompts")
    final_path = _resolve(root, cfg.onpolicy.source_final)
    out_path = _resolve(root, cfg.onpolicy.prompt_set)

    rows: list[dict] = []
    seen: set[str] = set()
    injection_cache: dict[str, str] = {}
    for rec in iter_jsonl(final_path):
        pid = rec["lineage"]["custom_id"]
        validate_custom_id(pid)
        if pid in seen:
            raise ValueError(
                f"duplicate lineage.custom_id {pid!r} in {final_path} — the "
                f"prompt set requires unique prompt ids"
            )
        seen.add(pid)
        sec_id = rec["section_id"]
        if sec_id not in injection_cache:
            injection_cache[sec_id] = templates.render(
                INJECTION_TEMPLATE,
                section_text=section_by_id(sections, sec_id).text(md),
            )
        clean_system = rec.get("system")
        gen_system = (
            f"{clean_system}\n\n{injection_cache[sec_id]}"
            if clean_system
            else injection_cache[sec_id]
        )
        rows.append(
            {
                "prompt_id": pid,
                "source_run_id": rec["lineage"]["run_id"],
                "section_id": sec_id,
                "system": gen_system,
                "clean_system": clean_system,
                "user": rec["user"],
            }
        )
    rows.sort(key=lambda r: r["prompt_id"])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl_atomic(
        out_path, (json.dumps(r, ensure_ascii=False, sort_keys=True) for r in rows)
    )
    manifest = {
        "source_final": str(final_path),
        "source_final_sha256": sha256_file(final_path),
        "n_prompts": len(rows),
        "constitution_sha256": snap["sha256"],
        "template_hashes": templates.hashes([INJECTION_TEMPLATE]),
        # wave C sent NO explicit temperature/top_p — the Anthropic API
        # defaults applied (temperature 1.0). The sampler must mirror these
        # plus wave C's max_tokens; recorded here as the single source.
        "sampling": {
            "temperature": 1.0,
            "top_p": 1.0,
            "max_tokens": cfg.advice.max_tokens["c"],
            "provenance": (
                "wave C params: model + max_tokens only (stages.wave_params); "
                "temperature/top_p are the API defaults"
            ),
        },
        "prompt_set_sha256": sha256_file(out_path),
        "created_at": _utcnow(),
    }
    atomic_write(
        manifest_path(out_path), json.dumps(manifest, indent=1, ensure_ascii=False)
    )
    on_event(
        f"prompt set: {len(rows)} prompts -> {out_path} "
        f"(sha256 {manifest['prompt_set_sha256'][:12]}…)"
    )
    return manifest


def manifest_path(prompt_set_path: Path) -> Path:
    return prompt_set_path.with_name(prompt_set_path.stem + ".manifest.json")


# ------------------------------------------------------------------ scoring


def load_prompt_set(prompt_set_path: Path) -> tuple[list[dict], dict]:
    """Load + VERIFY the prompt set against its sha-stamped manifest."""
    manifest = read_json(manifest_path(prompt_set_path))
    actual = sha256_file(prompt_set_path)
    if actual != manifest["prompt_set_sha256"]:
        raise RuntimeError(
            f"prompt set {prompt_set_path} sha256 {actual[:12]}… != manifest "
            f"{manifest['prompt_set_sha256'][:12]}… — re-extract or restore"
        )
    return list(iter_jsonl(prompt_set_path)), manifest


def load_samples(cfg: RunConfig, root: Path) -> tuple[list[dict], dict]:
    """Responses pulled back from the cluster sampler + their manifest."""
    op = cfg.onpolicy
    samples_path = _resolve(root, op.samples)
    mpath = (
        _resolve(root, op.samples_manifest)
        if op.samples_manifest
        else samples_path.parent / "SAMPLES_MANIFEST.json"
    )
    samples = list(iter_jsonl(samples_path))
    manifest = read_json(mpath) if mpath.exists() else {}
    ids = [s["prompt_id"] for s in samples]
    if len(set(ids)) != len(ids):
        raise RuntimeError(f"duplicate prompt_ids in {samples_path}")
    return samples, manifest


def onpolicy_e_spec() -> StageSpec:
    """The scoring wave: production e_score template + full-constitution
    context, over (clean system, user, SAMPLED response). Leaf; parse
    failures get the bounded repair, exactly like production e."""

    def build(ctx: StageCtx) -> list[WaveRequest]:
        reqs = []
        for row in sorted(ctx.ok_rows("samples"), key=lambda r: r.key):
            rendered = ctx.templates.render(
                SCORE_TEMPLATE,
                system=row.data.get("clean_system"),
                user=row.data["user"],
                assistant=row.data["response"],
            )
            params = {
                "model": ctx.cfg.engine.model,
                "max_tokens": ctx.cfg.advice.max_tokens["e"],
                "messages": [
                    {
                        "role": "user",
                        "content": const_user_content(ctx.constitution_md, rendered),
                    }
                ],
            }
            if ctx.cfg.engine.use_structured_scores:
                params["output_config"] = structured_format(ADVICE_SCORE_SCHEMA)
            reqs.append(
                WaveRequest(
                    custom_id=f"onp-e-{row.key}",
                    params=params,
                    lineage={"prompt_id": row.key},
                )
            )
        return reqs

    def parse(ctx: StageCtx, req: WaveRequest, outcome: dict) -> list[OutputRow]:
        data = parse_score_object(
            outcome["text"], ADVICE_SCORE_SCHEMA, f"scores for {req.custom_id}"
        )
        return [
            OutputRow(key=req.custom_id, custom_id=req.custom_id, status="ok", data=data)
        ]

    return StageSpec(
        name="e",
        corpus="advice",
        kind="leaf",
        templates=[SCORE_TEMPLATE],
        build_requests=build,
        expected_units=lambda req: [req.custom_id],
        parse=parse,
        build_repair=repair_request,
    )


def truncated(sample: dict) -> bool:
    """finish_reason 'length' = the sampler hit wave C's max_tokens cap.
    Production wave C treated max_tokens as a terminal per-record failure
    (the record never materialized); on-policy keeps that semantic — the
    loss is visible in SAMPLES_MANIFEST.json's finish_reasons, exactly as
    production losses live in the stage stamps."""
    return sample.get("finish_reason", "stop") == "length"


def samples_as_rows(prompt_rows: list[dict], samples: list[dict]) -> list[OutputRow]:
    """Join prompt set + sampled responses into the scoring wave's upstream.

    Empty/whitespace responses are excluded (nothing to score — the record
    still materializes and the filter drops it with a counted reason);
    truncated samples are excluded entirely (production-C semantics, see
    ``truncated``); a sample whose prompt_id is not in the prompt set is a
    hard error."""
    prompts = {p["prompt_id"]: p for p in prompt_rows}
    rows = []
    for s in samples:
        pid = s["prompt_id"]
        if pid not in prompts:
            raise RuntimeError(f"sample {pid!r} has no prompt-set row")
        if truncated(s) or not (s.get("text") or "").strip():
            continue
        p = prompts[pid]
        rows.append(
            OutputRow(
                key=pid,
                custom_id=pid,
                status="ok",
                data={
                    "clean_system": p["clean_system"],
                    "user": p["user"],
                    "response": s["text"],
                },
                lineage={"source_run_id": p["source_run_id"]},
            )
        )
    return rows


# -------------------------------------------------------------- materialize


def materialize_onpolicy(
    ctx: StageCtx,
    prompt_rows: list[dict],
    samples: list[dict],
    source_records: dict[str, dict],
    sampler: dict,
) -> Path:
    """AdviceRecord-shaped records: assistant = the model's OWN response
    (initial_response == assistant; every rewrite-stage field is None — the
    rewrite step is deliberately absent). Prompt lineage (principle, type,
    scenario, section) carries over from the source record; scores come from
    the e wave (None when the wave produced no verdict — the filter counts
    those as "unscored", or "too_short" for empty responses)."""
    const_sha = hashlib.sha256(ctx.constitution_md.encode("utf-8")).hexdigest()
    prompts = {p["prompt_id"]: p for p in prompt_rows}
    e_rows = {r.key.removeprefix("onp-e-"): r for r in ctx.ok_rows("e", [])}

    records = []
    for s in sorted(samples, key=lambda x: x["prompt_id"]):
        if truncated(s):
            continue  # terminal per-record failure, production-C semantics
        pid = s["prompt_id"]
        p = prompts[pid]
        src = source_records[pid]
        erow = e_rows.get(pid)
        batch_ids = {"e": erow.lineage.get("batch_id")} if erow is not None else {}
        rec = AdviceRecord(
            system=p["clean_system"],
            user=p["user"],
            assistant=s.get("text") or "",
            principle_id=src["principle_id"],
            principle=src["principle"],
            prompt_type_id=src["prompt_type_id"],
            prompt_type=src["prompt_type"],
            scenario_id=src["scenario_id"],
            scenario=src["scenario"],
            section_id=p["section_id"],
            draft_system=None,
            draft_user=None,
            diagnostic_response=None,
            prompt_critique=None,
            initial_response=s.get("text") or "",
            rewrite_problems=None,
            scores=erow.data if erow is not None else None,
            lineage=Lineage(
                run_id=ctx.cfg.run_id,
                custom_id=pid,
                batch_ids={k: v for k, v in batch_ids.items() if v},
                constitution_sha256=const_sha,
                prompt_version=PROMPT_VERSION,
                source_run_id=p["source_run_id"],
                sampler=sampler,
            ),
        )
        records.append(rec)

    out_path = ctx.corpus_dir / "records.jsonl"
    write_jsonl_atomic(out_path, (r.model_dump_json() for r in records))
    return out_path


# ------------------------------------------------------------------ driver


def run_onpolicy_score(
    cfg: RunConfig, engine, root: Path | None = None, on_event=print
) -> Path:
    """One scoring run for one sampler model: e wave over the samples ->
    materialized records.jsonl. Resumable end-to-end (stamped stage skipped;
    materialization is a pure function of stored artifacts)."""
    root = root or repo_root()
    op = cfg.onpolicy
    if op is None:
        raise ValueError("config has no `onpolicy` section")
    ctx, run_dir = build_ctx(cfg, "advice", root)

    prompt_rows, pm = load_prompt_set(_resolve(root, op.prompt_set))
    # the prompt set must have been built against the SAME frozen constitution
    const_sha = hashlib.sha256(ctx.constitution_md.encode("utf-8")).hexdigest()
    if pm["constitution_sha256"] != const_sha:
        raise RuntimeError(
            f"prompt set was extracted against constitution "
            f"{pm['constitution_sha256'][:12]}… but the frozen snapshot is "
            f"{const_sha[:12]}…"
        )
    samples, samples_manifest = load_samples(cfg, root)
    source_records = {
        r["lineage"]["custom_id"]: r
        for r in iter_jsonl(_resolve(root, op.source_final))
    }
    missing = [s["prompt_id"] for s in samples if s["prompt_id"] not in source_records]
    if missing:
        raise RuntimeError(
            f"{len(missing)} samples have no source record (first: {missing[:3]})"
        )

    ctx.upstream["samples"] = samples_as_rows(prompt_rows, samples)
    n_empty = len(samples) - len(ctx.upstream["samples"])
    on_event(
        f"[{cfg.run_id}] scoring {len(ctx.upstream['samples'])} responses "
        f"({n_empty} empty responses excluded from scoring) with {cfg.engine.model}"
    )
    spec = onpolicy_e_spec()
    _heartbeat(run_dir, "advice", spec.name)
    rows = run_stage(
        engine, ctx, spec,
        repair_rounds=cfg.engine.repair_rounds,
        min_slots_frac=cfg.engine.min_slots_frac,
        on_event=on_event,
    )
    ctx.upstream["e"] = rows

    sampler = {
        "model": op.sampler_model,
        **{k: samples_manifest[k] for k in samples_manifest
           if k in ("engine", "vllm_version", "sampling", "slurm_job_id",
                    "model_path", "chat_mode", "sampled_at")},
    }
    records_path = materialize_onpolicy(
        ctx, prompt_rows, samples, source_records, sampler
    )
    _register_union(run_dir, "advice", records_path)
    _heartbeat(run_dir, "advice", "done")
    write_manifest(cfg, run_dir, root=root)
    on_event(f"[{cfg.run_id}] materialized {records_path}")
    return records_path


# ------------------------------------------------------------------ filter


# candidate (alignment, realism) floors reported over the scored-all set so a
# floor can be re-cut downstream without re-running anything
_CANDIDATE_FLOORS = [(a, r) for a in (5, 6, 7, 8) for r in (6, 7, 8)]


def candidate_floor_sizes(scored: list[dict]) -> list[dict]:
    out = []
    for a, r in _CANDIDATE_FLOORS:
        k = sum(
            1
            for x in scored
            if (x.get("scores") or {}).get("alignment", -1) >= a
            and (x.get("scores") or {}).get("realism", -1) >= r
        )
        out.append({"alignment": a, "realism": r, "kept": k})
    return out


def filter_onpolicy(
    cfg: RunConfig, root: Path | None = None, counter=None, on_event=print
) -> tuple:
    """Two NON-DESTRUCTIVE artifacts for one on-policy run (steering-vector
    consumers re-cut the floor themselves):

    1. ``scored-<run>.jsonl`` — EVERY hard-gate + sanity + dedup survivor with
       its full score object and token counts embedded, score floor NOT
       applied. The re-cuttable set: any (alignment, realism) floor applies
       downstream with zero recomputation.
    2. ``final-<run>.jsonl`` — the calibrated default view: a strict SUBSET of
       (1), keeping only records at/above the frozen floors. Cut purely by
       score (no re-dedup), so it is guaranteed ⊆ the scored set. Refused until
       the floors are frozen in the config (post-hoc calibration).

    Both reports carry the joint kept-set sizes at every candidate floor.
    """
    from .filtering import FilterReport, ThresholdsNotCalibrated, filter_corpus
    from .tokens import TokenCounter

    root = root or repo_root()
    if counter is None:
        counter = TokenCounter.load(cfg.resolve_data_dir(root) / "tokenizers")
    data_dir = cfg.resolve_data_dir(root)
    records = list(iter_jsonl(cfg.run_dir(root) / "advice" / "records.jsonl"))

    # (1) scored-all — score gate skipped, every other gate + dedup + tokens
    # applied. calibration=True alone is NOT enough: it skips the score gate
    # only while the thresholds are null, and here they are FROZEN — so null
    # them on a config copy to guarantee a genuinely floor-free set (else the
    # "re-cuttable" artifact would already be floored, defeating its purpose).
    scored_cfg = cfg.model_copy(deep=True)
    scored_cfg.filter.thresholds.advice_alignment = None
    scored_cfg.filter.thresholds.advice_realism = None
    scored_path = data_dir / "advice" / f"scored-{cfg.run_id}.jsonl"
    scored_path.parent.mkdir(parents=True, exist_ok=True)
    scored_report = filter_corpus(
        "advice", records, scored_cfg, counter, scored_path, calibration=True
    )
    scored = list(iter_jsonl(scored_path))
    grid = candidate_floor_sizes(scored)
    scored_report.candidate_floor_sizes = grid
    # the SCORED file is a shipped provenance artifact, not a throwaway pilot:
    # keep the loud score-skip flag but drop the 'calibration' marker so the
    # mirror provenance set reads honestly
    scored_report.calibration = False
    atomic_write(
        filter_report_path(scored_path),
        json.dumps(scored_report.to_json(), indent=1, ensure_ascii=False),
    )
    on_event(
        f"[{cfg.run_id}] scored-all: {scored_report.input_count} -> "
        f"{len(scored)} records (score floor NOT applied); candidate floors "
        + ", ".join(
            f"a>={g['alignment']}&r>={g['realism']}:{g['kept']}"
            for g in grid
            if g["alignment"] >= 6 and g["realism"] >= 7
        )
    )

    th = cfg.filter.thresholds
    if th.advice_alignment is None or th.advice_realism is None:
        raise ThresholdsNotCalibrated(
            f"{cfg.run_id}: post-hoc floors not frozen (filter.thresholds "
            f"advice_alignment/advice_realism) — the scored-all artifact is "
            f"written, but the default floored final needs the calibrated floors"
        )

    # (2) floored final — a pure score-subset of the scored-all set
    fa, fr_ = th.advice_alignment, th.advice_realism
    final = [
        r
        for r in scored
        if r["scores"]["alignment"] >= fa and r["scores"]["realism"] >= fr_
    ]
    report = FilterReport(corpus="advice", input_count=len(scored))
    report.score_thresholds_applied = {"alignment": fa, "realism": fr_}
    report.candidate_floor_sizes = grid
    report.dedup_rate = scored_report.dedup_rate  # dedup already applied in (1)
    limits = {"alignment": fa, "realism": fr_}
    for r in scored:
        s = r["scores"]
        for axis in ("alignment", "realism"):
            d = report.score_distributions.setdefault(axis, {})
            d[s[axis]] = d.get(s[axis], 0) + 1
        below = [k for k, v in limits.items() if s[k] < v]
        if below:
            for k in below:
                report.score_drops_by_axis[k] = report.score_drops_by_axis.get(k, 0) + 1
            report.drop(
                r["lineage"]["custom_id"], "score",
                "; ".join(f"{k}={s[k]} < {limits[k]}" for k in below),
            )
    totals: dict[str, int] = {name: 0 for name in counter.tokenizers}
    for r in final:
        for k, v in (r.get("token_counts") or {}).items():
            totals[k] += v
    report.token_totals = totals
    report.official_tokens = min(totals.values()) if totals else 0
    report.output_count = len(final)
    final_path = cfg.final_path(root, "advice")
    final_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl_atomic(final_path, (json.dumps(r, ensure_ascii=False) for r in final))
    atomic_write(
        filter_report_path(final_path),
        json.dumps(report.to_json(), indent=1, ensure_ascii=False),
    )
    on_event(
        f"[{cfg.run_id}] final (alignment>={fa} & realism>={fr_}): "
        f"{len(final)} records / {report.official_tokens:,} official tokens "
        f"(strict subset of the {len(scored)}-record scored-all set)"
    )
    return scored_report, report


def slack_line(cfgs: list[RunConfig], n_requests: int) -> str:
    models = ", ".join(c.onpolicy.sampler_model.split("/")[-1] for c in cfgs if c.onpolicy)
    return (
        f"Batch API: ~{n_requests:,} claude-opus-5 requests (corpus_generator "
        f"on-policy adherence scoring over {models} — async, separate batch "
        f"pool, not live concurrents)"
    )
