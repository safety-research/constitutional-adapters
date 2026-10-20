"""corpusgen — config-driven CLI for the constitutional synthetic-data pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import click

from . import coordination
from .config import RunConfig
from .templates import repo_root


def _cfg(config_path: str) -> RunConfig:
    return RunConfig.load(Path(config_path))


class CredentialExitGroup(click.Group):
    """Every command ends TYPED on a credential rejection (the WIF driver sweep): the halt an engine raised
    (``WaveCredentialRejected``) -- or the SDK's own 401 / 403 out of a single call -- becomes ``SystemExit`` whose message
    is the typed text (``batch.credential_exit``), never a traceback; every other exception passes as before."""

    def invoke(self, ctx):
        from .batch import credential_exit

        with credential_exit("[corpusgen] "):
            return super().invoke(ctx)


@click.group(cls=CredentialExitGroup)
def main() -> None:
    """Constitutional synthetic-data pipeline (advice + stories corpora)."""


@main.command("fetch-constitution")
@click.option("--force", is_flag=True, help="Overwrite an existing frozen snapshot.")
@click.option(
    "--reference", default=str(Path.home() / "const.txt"),
    help="Reference plain-text fetch to cross-check against.",
)
def fetch_constitution(force: bool, reference: str) -> None:
    """One-time freeze of the constitution snapshot (chrome-free extraction)."""
    from .constitution import freeze

    snap = freeze(
        repo_root() / "constitution",
        reference_path=Path(reference),
        force=force,
    )
    click.echo(json.dumps(snap, indent=1))


@main.command("stage-tokenizers")
@click.option(
    "--out", "out_dir", default=None, type=click.Path(),
    help="Destination directory (default: <data_dir>/tokenizers of --config, else "
         "<repo>/data/tokenizers). `corpusgen filter` reads <data_dir>/tokenizers.",
)
@click.option(
    "--config", "config_path", default=None, type=click.Path(exists=True),
    help="A run config whose data_dir decides the default --out.",
)
@click.option(
    "--hf-home", default=None, type=click.Path(),
    help="The HF cache to read (HF_HOME, or its hub/ directory); default $HF_HOME "
         "else ~/.cache/huggingface.",
)
@click.option(
    "--revision", "revisions", multiple=True, metavar="NAME=SHA",
    help="Pin one family's snapshot (llama3|gemma|qwen = HF commit sha); repeatable. "
         "A pinned snapshot that is not cached REFUSES rather than substituting.",
)
@click.option(
    "--pins", "pins_path", default=None, type=click.Path(exists=True, dir_okay=False),
    help="A pins file ({tokenizers: {name: {repo, revision, files: [{name, sha256}]}}}, the harness's "
         "corpora/TOKENIZERS.json): pins every named family's repo + revision AND every file's sha256; "
         "a cached snapshot under the right revision with other content REFUSES, and nothing is written.",
)
def stage_tokenizers_cmd(out_dir, config_path, hf_home, revisions, pins_path) -> None:
    """Copy the 3 reference tokenizers from an HF cache into a staging dir
    (+ provenance MANIFEST.json). Pin the snapshots with --revision, or with
    --pins to verify every file's sha256 as well, to reproduce the corpus
    manifest's token counts exactly."""
    from .tokens import StagingError, load_pins, resolve_hf_hub, stage_tokenizers

    revs: dict[str, str] = {}
    for spec in revisions:
        name, sep, sha = spec.partition("=")
        if not sep or not name or not sha:
            raise click.ClickException(f"--revision must be NAME=SHA, got {spec!r}")
        revs[name] = sha
    if out_dir is None:
        out_dir = (
            _cfg(config_path).resolve_data_dir(repo_root()) / "tokenizers"
            if config_path else repo_root() / "data" / "tokenizers"
        )
    try:
        manifest = stage_tokenizers(
            Path(out_dir), hf_hub=resolve_hf_hub(Path(hf_home) if hf_home else None),
            revisions=revs, pins=load_pins(Path(pins_path)) if pins_path else None,
        )
    except StagingError as e:
        raise click.ClickException(str(e)) from e
    click.echo(json.dumps(manifest, indent=1))


@main.command()
@click.option("--config", "config_path", required=True, type=click.Path(exists=True))
def preflight(config_path: str) -> None:
    """Measure encoded request bodies per wave -> committed chunk plan."""
    from .preflight import format_plan, run_preflight

    plan = run_preflight(_cfg(config_path))
    click.echo(format_plan(plan))


@main.command()
@click.option("--config", "config_path", required=True, type=click.Path(exists=True))
def smoke(config_path: str) -> None:
    """<=5 live requests (lowprio) + tiny real batch (batch key); records the
    nonce-metadata and structured-output verdicts."""
    from .smoke import run_smoke

    verdict = run_smoke(_cfg(config_path))
    click.echo(json.dumps(verdict, indent=1))


@main.command()
@click.option("--config", "config_path", required=True, type=click.Path(exists=True))
@click.option("--corpus", type=click.Choice(["advice", "stories", "all"]), default="all")
def run(config_path: str, corpus: str) -> None:
    """Drive the stage graph(s). Prints the DAG-expanded request total and the
    Slack coordination message at submit time, then PROCEEDS (never gates)."""
    from .client import build_client
    from .pipeline import (
        dag_expanded_totals,
        resolve_environment,
        run as run_pipeline,
        slack_message,
    )

    cfg = resolve_environment(_cfg(config_path), on_event=click.echo)
    totals = dag_expanded_totals(cfg)
    click.echo(f"DAG-expanded request totals: {json.dumps(totals)}")
    coordination.post(slack_message(cfg, corpus), click.echo)
    client = build_client("batch")
    run_pipeline(cfg, corpus, client)
    click.echo("run complete")


@main.command()
@click.option("--config", "config_path", required=True, type=click.Path(exists=True))
@click.option("--corpus", type=click.Choice(["advice", "stories"]), required=True)
@click.option("--stories", "stories_n", type=int, default=None, help="Story-count deficit override.")
@click.option("--tokens", "tokens_n", type=int, default=None, help="Token deficit override.")
def topup(config_path: str, corpus: str, stories_n: int | None, tokens_n: int | None) -> None:
    """Compute the binding deficit and write extension-wave specs (LCB-sized).
    Then re-run `corpusgen run` to execute the extension sub-DAG."""
    from .io import iter_jsonl
    from .topup import plan_topup

    cfg = _cfg(config_path)
    root = repo_root()
    finals = list(iter_jsonl(cfg.final_path(root, corpus), missing_ok=True))
    spec = plan_topup(cfg, cfg.run_dir(root), corpus, finals, stories_n, tokens_n)
    if spec is None:
        click.echo("no deficit — nothing to top up")
    else:
        click.echo(f"wrote extension spec: {json.dumps(spec)[:400]}")
        click.echo("now run: corpusgen run --config <same config> --corpus " + corpus)


@main.command()
@click.option("--config", "config_path", required=True, type=click.Path(exists=True))
@click.option("--live", is_flag=True, help="Also retrieve in-flight batch ids from the API.")
def status(config_path: str, live: bool) -> None:
    """Read-only: ledger/stamp state per wave (+ batches.retrieve with --live)."""
    from .batch import Ledger

    cfg = _cfg(config_path)
    run_dir = cfg.run_dir(repo_root())
    client = None
    for stamp in sorted(run_dir.glob("*/**/REQUEST_SET.json")):
        wave_dir = stamp.parent
        rel = wave_dir.relative_to(run_dir)
        stamped = (wave_dir / "STAMP.json").exists()
        state = "stamped" if stamped else "in-flight"
        line = f"{rel}: {state}"
        if not stamped and (wave_dir / "ledger.jsonl").exists():
            states = {}
            for c, st in sorted(Ledger(wave_dir / "ledger.jsonl").chunk_states().items()):
                states[c] = st.state + (f" {st.batch_id}" if st.batch_id else "")
                if live and st.batch_id:
                    if client is None:
                        from .client import build_client

                        client = build_client("batch")
                    b = client.messages.batches.retrieve(st.batch_id)
                    states[c] += f" [{b.processing_status}]"
            line += f" {json.dumps(states)}"
        click.echo(line)
    status_path = run_dir / "status.json"
    if status_path.exists():
        click.echo(f"heartbeat: {status_path.read_text(encoding='utf-8').strip()}")


LIVE_COORDINATION_LINE = (
    "~200 claude-opus-5 + claude-opus-4-8 lowprio live, ~1-2h "
    "(corpus_generator fixups)"
)


def _live_engine(cfg: RunConfig, echo):
    from .client import build_async_client
    from .live import LiveEngine

    coordination.post(LIVE_COORDINATION_LINE, echo)
    return LiveEngine(
        lambda: build_async_client("live"),
        model=cfg.engine.model,
        on_event=echo,
    )


@main.command("regen-stories")
@click.option("--config", "config_path", required=True, type=click.Path(exists=True))
@click.option(
    "--phase", type=click.Choice(["draft", "probe", "finish"]), required=True,
    help="draft = S3 only (the guardrail probe); probe = report the verdict "
         "from stored artifacts (no API calls); finish = S3->S5 + materialize.",
)
@click.option("--source-run", default="production-20260831", show_default=True)
@click.option(
    "--skip-lane", "skip_lanes", multiple=True,
    help="Lane abandoned by the probe verdict (e.g. c32-u00).",
)
def regen_stories(config_path: str, phase: str, source_run: str, skip_lanes) -> None:
    """Fix 3: regenerate the refusal-censored story lanes LIVE on lowprio
    (claude-opus-4-8). Draft first; the probe verdict gates the rest."""
    from .regen import format_probe, run_stories_regen, stories_probe

    cfg = _cfg(config_path)
    root = repo_root()
    if phase == "probe":
        click.echo(format_probe(stories_probe(cfg, root=root)))
        return
    engine = _live_engine(cfg, click.echo)
    run_stories_regen(
        cfg, engine,
        prod_run_dir=cfg.resolve_data_dir(root) / "runs" / source_run,
        phase=phase, skip_lanes=tuple(skip_lanes), root=root, on_event=click.echo,
    )
    if phase == "draft":
        click.echo("--- probe verdict (S3 base wave) ---")
        click.echo(format_probe(stories_probe(cfg, root=root)))


@main.command("regen-advice")
@click.option("--config", "config_path", required=True, type=click.Path(exists=True))
@click.option(
    "--principles", "principles_path", required=True, type=click.Path(exists=True),
    help="JSON of hand-reworded principles: pid -> {principle, tension, section_id}.",
)
def regen_advice(config_path: str, principles_path: str) -> None:
    """Fix 1: regenerate the reworded principles' full subtrees (A2->E) LIVE
    on lowprio (claude-opus-5)."""
    from .regen import load_regen_principles, run_advice_regen

    cfg = _cfg(config_path)
    principles = load_regen_principles(Path(principles_path))
    click.echo(f"regenerating full subtrees for: {', '.join(sorted(principles))}")
    engine = _live_engine(cfg, click.echo)
    run_advice_regen(cfg, engine, principles=principles, root=repo_root(), on_event=click.echo)


@main.command("seed-principles")
@click.option("--config", "config_path", required=True, type=click.Path(exists=True))
@click.option(
    "--principles", "principles_path", required=True, type=click.Path(exists=True),
    help="Frozen, hand-reviewed principles JSON ({source, principles: pid -> "
         "{principle, tension, section_id}}; the flat regen form is accepted).",
)
def seed_principles(config_path: str, principles_path: str) -> None:
    """Materialize the run's advice A1 stage from a REVIEWED principle set
    so `corpusgen run` resumes past A1 instead of
    sampling a fresh, unreviewed one. Refuses if the stage already exists or
    the set does not match the configured fanout; a later A1 template/config
    change fails the run loudly as a stage-hash mismatch."""
    from .principles import load_frozen_principles, seed_a1_stage

    cfg = _cfg(config_path)
    principles, source = load_frozen_principles(Path(principles_path))
    stage_dir = seed_a1_stage(cfg, principles, source, root=repo_root(), on_event=click.echo)
    click.echo(f"stage dir: {stage_dir}")


@main.command("import-scan")
@click.option("--config", "config_path", required=True, type=click.Path(exists=True))
@click.option("--corpus", type=click.Choice(["advice", "stories", "all"]), default="all")
@click.option(
    "--input", "inputs", multiple=True, type=click.Path(exists=True),
    help="Record artifact(s) to scan (jsonl). Default: the run's final per "
         "corpus, else its calibration artifact.",
)
@click.option("--suffix", default="", help='Release generation, e.g. "-v2".')
def import_scan_cmd(config_path: str, corpus: str, inputs, suffix: str) -> None:
    """Release check: the placebo phrase battery, the
    value-word battery, and verbatim n-gram overlap between the run's
    reference document and every record's TRAINABLE text. Report-only —
    writes <artifact>.import-scan.json beside each scanned artifact."""
    from .constitution import load as load_constitution
    from .import_scan import format_scan, import_scan_path, run_import_scan
    from .io import iter_jsonl
    from .register_survey import detect_corpus

    cfg = _cfg(config_path)
    root = repo_root()
    document, _sections, _snap = load_constitution(cfg.resolve_constitution_dir(root))
    jobs: list[tuple[str | None, Path]] = []
    if inputs:
        jobs = [(None if corpus == "all" else corpus, Path(p)) for p in inputs]
    else:
        for c in ["advice", "stories"] if corpus == "all" else [corpus]:
            fin, cal = cfg.final_path(root, c, suffix), cfg.calibration_path(root, c)
            if fin.exists():
                jobs.append((c, fin))
            elif cal.exists():
                jobs.append((c, cal))
            else:
                raise click.ClickException(
                    f"{c}: no artifact to scan — neither {fin} nor {cal} exists; "
                    f"run `corpusgen filter` first or pass --input"
                )
    for c, path in jobs:
        records = list(iter_jsonl(path))
        if not records:
            raise click.ClickException(f"{path}: empty artifact")
        c = c or detect_corpus(records)
        res = run_import_scan(cfg, c, records, document, import_scan_path(path))
        click.echo(f"# source: {path}")
        click.echo(format_scan(res))
        click.echo(f"import-scan JSON written to {import_scan_path(path)}\n")


@main.command("filter")
@click.option("--config", "config_path", required=True, type=click.Path(exists=True))
@click.option("--corpus", type=click.Choice(["advice", "stories", "all"]), default="all")
@click.option(
    "--calibrate", is_flag=True,
    help="Pilot threshold-setting mode: null thresholds allowed (score gate "
         "skipped loudly) and the output is a clearly NON-final "
         "calibration-<run>.jsonl artifact.",
)
def filter_cmd(config_path: str, corpus: str, calibrate: bool) -> None:
    """Offline filter chain over the run's UNION -> final.jsonl (+ report).
    A FINAL corpus refuses null score thresholds; use --calibrate for pilots."""
    from .filtering import filter_corpus
    from .pipeline import load_union_records
    from .tokens import TokenCounter

    cfg = _cfg(config_path)
    root = repo_root()
    run_dir = cfg.run_dir(root)
    counter = TokenCounter.load(cfg.resolve_data_dir(root) / "tokenizers")
    for c in ["advice", "stories"] if corpus == "all" else [corpus]:
        records = load_union_records(run_dir, c)
        out = (
            cfg.calibration_path(root, c) if calibrate else cfg.final_path(root, c)
        )
        out.parent.mkdir(parents=True, exist_ok=True)
        report = filter_corpus(c, records, cfg, counter, out, calibration=calibrate)
        click.echo(
            f"{c}: {report.input_count} -> {report.output_count} records, "
            f"official tokens {report.official_tokens:,} "
            f"(drops: {json.dumps(report.rule_counts)})"
            + ("  [CALIBRATION artifact — not a final]" if calibrate else "")
            + ("  [score filter SKIPPED — thresholds not calibrated]"
               if report.score_filter_skipped else "")
        )
        ls = report.length_stats
        if ls.get("n"):
            click.echo(
                f"{c}: response length ({ls['field']}) mean {ls['mean']:.0f} / "
                f"median {ls['median']:.0f} / p90 {ls['p90']:.0f} words"
                + (f"; achieved/target {ls['targets']['mean_achieved_over_target']:.2f}"
                   if ls.get("targets") else "")
            )
        if report.length_gate is not None:
            lg = report.length_gate
            click.echo(
                f"{c}: length gate {'PASSED' if lg['passed'] else 'FAILED'} — "
                f"mean ratio {lg.get('mean_ratio')} (±{lg['tolerance']['mean']}), "
                f"p90 ratio {lg.get('p90_ratio')} (±{lg['tolerance']['p90']}) "
                f"vs reference mean {lg['reference']['mean']} / p90 {lg['reference']['p90']}"
            )
        for failure in report.acceptance_failures:
            click.echo(f"{c}: ACCEPTANCE GATE FAILURE: {failure}")


@main.command("register-survey")
@click.option("--config", "config_path", required=True, type=click.Path(exists=True))
@click.option("--corpus", type=click.Choice(["advice", "stories", "all"]), default="all")
@click.option(
    "--input", "inputs", multiple=True, type=click.Path(exists=True),
    help="Record artifact(s) to survey (jsonl). Default: the run's calibration "
         "artifact per corpus, else its final.",
)
@click.option("--opener-words", type=int, default=3, show_default=True)
@click.option("--n", "ngram_sizes", multiple=True, type=int, help="n-gram sizes (default 4 6 8).")
@click.option("--top", type=int, default=25, show_default=True)
@click.option("--min-frac", type=float, default=0.01, show_default=True,
              help="n-gram floor as a fraction of records carrying the field.")
@click.option("--sample", type=int, default=0, show_default=True,
              help="Seeded subsample size (0 = every record).")
@click.option("--json", "json_out", type=click.Path(), default=None,
              help="Also write the full survey as JSON here.")
def register_survey_cmd(config_path, corpus, inputs, opener_words, ngram_sizes, top,
                        min_frac, sample, json_out) -> None:
    """Register DISCOVERY: rank system/user/response
    openers and repeated n-grams by record fraction over a calibration or
    final artifact, beside the live match rate of the configured register
    families — so a family can be banned prompt-side and frozen as a filter
    backstop BEFORE the full run."""
    from .io import iter_jsonl
    from .register_survey import (
        detect_corpus, format_survey, sample_records, survey, to_json,
    )

    cfg = _cfg(config_path)
    root = repo_root()
    sizes = tuple(ngram_sizes) or (4, 6, 8)
    jobs: list[tuple[str | None, Path]] = []
    if inputs:
        jobs = [(None if corpus == "all" else corpus, Path(p)) for p in inputs]
    else:
        for c in ["advice", "stories"] if corpus == "all" else [corpus]:
            cal, fin = cfg.calibration_path(root, c), cfg.final_path(root, c)
            if cal.exists():
                jobs.append((c, cal))
            elif fin.exists():
                jobs.append((c, fin))
            else:
                raise click.ClickException(
                    f"{c}: no artifact to survey — neither {cal} nor {fin} exists; "
                    f"run `corpusgen filter --calibrate` first or pass --input"
                )
    results = []
    for c, path in jobs:
        records = list(iter_jsonl(path))
        if not records:
            raise click.ClickException(f"{path}: empty artifact")
        c = c or detect_corpus(records)
        records = sample_records(records, sample, seed=cfg.seed)
        res = survey(
            records, c, families=cfg.filter.register_families,
            opener_words=opener_words, ngram_sizes=sizes, top=top, min_frac=min_frac,
        )
        res["source"] = str(path)
        results.append(res)
        click.echo(f"# source: {path}")
        click.echo(format_survey(res))
        click.echo("")
    if json_out:
        Path(json_out).write_text(
            to_json(results[0] if len(results) == 1 else {"surveys": results}),
            encoding="utf-8",
        )
        click.echo(f"survey JSON written to {json_out}")


@main.command("assemble-v2")
@click.option("--config", "config_path", required=True, type=click.Path(exists=True))
@click.option("--corpus", type=click.Choice(["advice", "stories", "all"]), default="all")
@click.option(
    "--add", "adds", multiple=True,
    help="corpus=path of a regenerated final to merge, e.g. "
         "advice=data/advice/final-advice-regen-20260901.jsonl",
)
@click.option("--drop-principle", "drop_principles", multiple=True)
@click.option("--drop-run", default=None, help="Base run whose dropped-principle records go.")
@click.option("--drop-id", "drop_ids", multiple=True, help="Residual per-record drops.")
@click.option("--suffix", default="-v2", show_default=True)
def assemble_v2_cmd(config_path, corpus, adds, drop_principles, drop_run, drop_ids, suffix) -> None:
    """Merge v1 finals + regenerated finals - ordered drops, re-run the full
    filter chain (incl. register caps), write final-<run><suffix>.jsonl."""
    from .assemble import assemble_v2
    from .tokens import TokenCounter

    cfg = _cfg(config_path)
    root = repo_root()
    counter = TokenCounter.load(cfg.resolve_data_dir(root) / "tokenizers")
    add_map: dict[str, list[Path]] = {}
    for a in adds:
        c, _, p = a.partition("=")
        add_map.setdefault(c, []).append(Path(p) if Path(p).is_absolute() else root / p)
    for c in ["advice", "stories"] if corpus == "all" else [corpus]:
        report = assemble_v2(
            cfg, c, root=root,
            add_sources=add_map.get(c, []),
            drop_principles=set(drop_principles) if c == "advice" else None,
            drop_run=drop_run,
            drop_ids=set(drop_ids) if c == "advice" else None,
            suffix=suffix,
            counter=counter,
            on_event=click.echo,
        )
        click.echo(
            f"{c}{suffix}: {report.input_count} -> {report.output_count} records, "
            f"official tokens {report.official_tokens:,} "
            f"(drops: {json.dumps(report.rule_counts)})"
        )
        for failure in report.acceptance_failures:
            click.echo(f"{c}: ACCEPTANCE GATE FAILURE: {failure}")


@main.command()
@click.option("--config", "config_path", required=True, type=click.Path(exists=True))
@click.option("--suffix", default="", help='Release generation, e.g. "-v2".')
@click.option(
    "--extra-run", "extra_runs", multiple=True,
    help="Additional run ids whose stages join the stage table (regen runs).",
)
def report(config_path: str, suffix: str, extra_runs) -> None:
    """Generation report + figures + spot-check HTML from stored artifacts."""
    from .report import build_report

    cfg = _cfg(config_path)
    root = repo_root()
    data_dir = cfg.resolve_data_dir(root)
    out = build_report(
        cfg, cfg.run_dir(root), data_dir, seed=cfg.seed, suffix=suffix,
        extra_run_dirs=tuple(data_dir / "runs" / r for r in extra_runs),
    )
    click.echo(f"report written to {out}")


@main.group("register-edit")
def register_edit_group() -> None:
    """r2 register-matching edit pass over a RELEASED final (2026-09-03):
    verified span edits by the generator model instead of regeneration, then
    re-score + the full filter chain. See register_edit.py."""


@register_edit_group.command("plan")
@click.option("--config", "config_path", required=True, type=click.Path(exists=True))
@click.option("--pass", "edit_pass", type=int, default=1, show_default=True)
def register_edit_plan(config_path: str, edit_pass: int) -> None:
    """Offline: measure every marker on the source and on production (frozen
    targets re-checked), select records deterministically, write PLAN."""
    from .register_edit import build_plan, slack_line

    plan = build_plan(_cfg(config_path), root=repo_root(), edit_pass=edit_pass, on_event=click.echo)
    click.echo(json.dumps(plan["requests"]))
    coordination.post(slack_line(plan["requests"]["total_edits"], "live"), click.echo, note="at launch", proceeding=False)


@register_edit_group.command("run")
@click.option("--config", "config_path", required=True, type=click.Path(exists=True))
@click.option("--pass", "edit_pass", type=int, default=1, show_default=True)
@click.option("--phase", type=click.Choice(["edit", "rescore", "all"]), default="all", show_default=True)
@click.option(
    "--engine", "engine_kind", type=click.Choice(["live", "batch"]), default="live", show_default=True,
    help="live = the live runner on the lowprio credential (the default since 2026-09-03); batch = the Batch API engine on the batch credential.",
)
def register_edit_run(config_path: str, edit_pass: int, phase: str, engine_kind: str) -> None:
    """Every edit stage of the pass, then the re-score wave over the edited
    records — on the LIVE lowprio runner by default (200 concurrent, shrinking
    on 429/529, degraded-provider guard). Prints the coordination line and
    PROCEEDS (posting is done out of band, never a gate). Resumable."""
    from .register_edit import build_live_engine, load_plan, run_register_edit, slack_line

    cfg = _cfg(config_path)
    root = repo_root()
    plan = load_plan(cfg, root, edit_pass, on_event=click.echo)
    coordination.post(slack_line(plan["requests"]["total_edits"], engine_kind), click.echo)
    if engine_kind == "live":
        engine = build_live_engine(cfg, on_event=click.echo)
    else:
        from .client import build_client
        from .pipeline import make_engine, resolve_environment

        engine = make_engine(resolve_environment(cfg, root=root, on_event=click.echo), build_client("batch"))
    summary = run_register_edit(cfg, engine, root, edit_pass=edit_pass, phase=phase, on_event=click.echo)
    click.echo(f"register-edit pass {edit_pass} {'/'.join(summary['phases'])} complete")


@register_edit_group.command("assemble")
@click.option("--config", "config_path", required=True, type=click.Path(exists=True))
@click.option("--pass", "edit_pass", type=int, default=1, show_default=True)
@click.option("--corpus", type=click.Choice(["advice", "stories", "all"]), default="all", show_default=True,
              help="Re-assemble one corpus offline (merges into the run-level manifest).")
@click.option("--reuse-final", is_flag=True,
              help="Skip the filter chain when the final + filter report already exist for this "
                   "assembled set (an assembly that died in the analytics after the chain).")
def register_edit_assemble(config_path: str, edit_pass: int, corpus: str, reuse_final: bool) -> None:
    """Offline: fold the stamped edits + re-scores into the r2 record set,
    run the FULL filter chain to final-<run>-r2-<corpus>.jsonl, write the
    per-corpus and run-level manifests (before/after register table vs
    production, verification, re-score deltas, monoculture check)."""
    from .register_edit import assemble
    from .tokens import TokenCounter

    cfg = _cfg(config_path)
    root = repo_root()
    counter = TokenCounter.load(cfg.resolve_data_dir(root) / "tokenizers")
    out = assemble(cfg, root, counter, edit_pass=edit_pass,
                   corpora=None if corpus == "all" else (corpus,), reuse_final=reuse_final,
                   on_event=click.echo)
    for corpus, c in out["corpora"].items():
        click.echo(f"\n== {corpus}: register table (v1 -> r2 vs production)")
        for name, row in c["register_table"].items():
            click.echo(
                f"  {name:32s} v1 {row['v1']['rate']:.4f}  r2 {row['r2']['rate']:.4f}  "
                f"prod {row['production']['rate']:.4f}  [{row['kind']}]"
            )


@main.group()
def onpolicy() -> None:
    """On-policy step-5 variant: open-weight models sample wave C; Opus
    scores (never rewrites). See onpolicy.py."""


@onpolicy.command("extract")
@click.option("--config", "config_path", required=True, type=click.Path(exists=True))
def onpolicy_extract(config_path: str) -> None:
    """Source final -> sha-stamped prompt-set artifact (+ manifest)."""
    from .onpolicy import extract_prompt_set

    manifest = extract_prompt_set(_cfg(config_path), root=repo_root(), on_event=click.echo)
    click.echo(json.dumps(manifest, indent=1))


@onpolicy.command("score")
@click.option(
    "--config", "config_paths", multiple=True, required=True,
    type=click.Path(exists=True),
    help="One per sampler model; ALL runs are submitted up front (threads).",
)
def onpolicy_score(config_paths) -> None:
    """Opus adherence scoring: one Batch e-wave per sampler model, all
    submitted up front, harvested together. Prints the coordination line and
    PROCEEDS (posting is done out of band, never a gate)."""
    import threading

    from .client import build_client
    from .onpolicy import load_samples, run_onpolicy_score, slack_line
    from .pipeline import resolve_environment

    root = repo_root()
    cfgs = [resolve_environment(_cfg(p), on_event=click.echo) for p in config_paths]
    n = sum(
        sum(1 for s in load_samples(c, root)[0] if (s.get("text") or "").strip())
        for c in cfgs
    )
    coordination.post(slack_line(cfgs, n), click.echo)
    client = build_client("batch")
    errors: list[BaseException] = []

    def _target(cfg):
        try:
            from .pipeline import make_engine

            run_onpolicy_score(cfg, make_engine(cfg, client), root=root, on_event=click.echo)
        except BaseException as exc:  # noqa: BLE001 - surfaced below
            errors.append(exc)
            click.echo(f"[{cfg.run_id}] FAILED: {exc}")

    threads = [
        threading.Thread(target=_target, args=(c,), name=f"onp-{c.run_id}") for c in cfgs
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if errors:
        raise errors[0]
    click.echo("scoring runs complete")


@onpolicy.command("filter")
@click.option("--config", "config_path", required=True, type=click.Path(exists=True))
@click.option(
    "--calibrate", is_flag=True,
    help="Calibration only: write the throwaway calibration-<run>.jsonl "
         "(score gate skipped) for a first look BEFORE floors are frozen. The "
         "real run needs no flag: it writes both scored-<run>.jsonl "
         "(re-cuttable) and the floored final-<run>.jsonl.",
)
def onpolicy_filter(config_path: str, calibrate: bool) -> None:
    """Two NON-DESTRUCTIVE artifacts: scored-<run>.jsonl (every scored record,
    re-cuttable at any floor) + final-<run>.jsonl (the calibrated default view,
    a strict subset). --calibrate writes only the throwaway calibration file."""
    cfg = _cfg(config_path)
    root = repo_root()
    if calibrate:
        from .filtering import filter_corpus
        from .io import iter_jsonl
        from .tokens import TokenCounter

        records = list(iter_jsonl(cfg.run_dir(root) / "advice" / "records.jsonl"))
        counter = TokenCounter.load(cfg.resolve_data_dir(root) / "tokenizers")
        out = cfg.calibration_path(root, "advice")
        out.parent.mkdir(parents=True, exist_ok=True)
        report = filter_corpus("advice", records, cfg, counter, out, calibration=True)
        click.echo(
            f"advice: {report.input_count} -> {report.output_count} records "
            f"[CALIBRATION — score gate skipped]; "
            f"distributions: {json.dumps(report.score_distributions)}"
        )
        return

    from .onpolicy import filter_onpolicy

    _scored, final_report = filter_onpolicy(cfg, root=root, on_event=click.echo)
    click.echo(
        f"final: {final_report.output_count} records / "
        f"{final_report.official_tokens:,} official tokens "
        f"(floors {json.dumps(final_report.score_thresholds_applied)})"
    )


@onpolicy.command("mirror")
@click.option("--config", "config_path", required=True, type=click.Path(exists=True))
@click.option("--host", default=None,
              help="ssh alias of the mirror host (default $SISTEER_CORPUS_MIRROR_HOST).")
@click.option("--dest-root", default=None,
              help="remote root (default $SISTEER_CORPUS_MIRROR_ROOT).")
@click.option("--dry-run", is_flag=True)
def onpolicy_mirror(config_path: str, host: str | None, dest_root: str | None, dry_run: bool) -> None:
    """Gate (no token floor — actuals recorded) + rsync + exact-set verify."""
    from .mirror import mirror_onpolicy, resolve_dest

    host, dest_root = resolve_dest(host, dest_root)
    v = mirror_onpolicy(_cfg(config_path), repo_root(), host=host, dest_root=dest_root, dry_run=dry_run)
    click.echo(json.dumps(v, indent=1))


@main.command()
@click.option("--config", "config_path", required=True, type=click.Path(exists=True))
@click.option("--host", default=None,
              help="ssh alias of the mirror host (default $SISTEER_CORPUS_MIRROR_HOST).")
@click.option("--dest-root", default=None,
              help="remote root (default $SISTEER_CORPUS_MIRROR_ROOT; $USER expands remotely).")
@click.option("--dry-run", is_flag=True)
@click.option("--suffix", default="", help='Release generation, e.g. "-v2".')
@click.option(
    "--extra-run", "extra_runs", multiple=True,
    help="Additional run ids whose manifest/UNION mirror too (regen runs).",
)
def mirror(config_path: str, host: str | None, dest_root: str | None, dry_run: bool, suffix: str, extra_runs) -> None:
    """rsync finals + manifests + report to the shared cluster storage and verify count/bytes/sha."""
    from .mirror import mirror as do_mirror, resolve_dest

    host, dest_root = resolve_dest(host, dest_root)
    cfg = _cfg(config_path)
    root = repo_root()
    data_dir = cfg.resolve_data_dir(root)
    v = do_mirror(
        cfg, data_dir, cfg.run_dir(root), host=host, dest_root=dest_root, dry_run=dry_run,
        suffix=suffix,
        extra_run_dirs=tuple(data_dir / "runs" / r for r in extra_runs),
    )
    click.echo(json.dumps(v, indent=1))


# ------------------------------------------------- constitution section tags


SECTION_TAGS_COORDINATION_LINE = "200 opus lowprio ~30min (constitution section tagging, live)"


def _tags_engine(concurrency: int, echo, model: str):
    from .client import build_async_client
    from .live import LiveEngine

    coordination.post(SECTION_TAGS_COORDINATION_LINE, echo)
    return LiveEngine(
        lambda: build_async_client("live"),
        model=model,
        max_concurrency=concurrency,
        min_concurrency=10,
        on_event=lambda m: echo(f"{_stamp()} {m}"),
    )


def _stamp() -> str:
    import datetime as _dt

    return _dt.datetime.now(_dt.timezone.utc).strftime("%H:%M:%SZ")


@main.group("section-tags")
def section_tags_group() -> None:
    """Post-hoc constitution-SECTION coverage of the corpora by content tagging."""


@section_tags_group.command("summarize")
@click.option("--force", is_flag=True, help="Regenerate an existing frozen index.")
@click.option("--model", default="claude-opus-5", show_default=True)
@click.option("--concurrency", type=int, default=10, show_default=True)
def section_tags_summarize(force: bool, model: str, concurrency: int) -> None:
    """Write constitution/section-summaries.json: one Opus summary per section (39 live calls)."""
    from . import section_tags as st
    from .constitution import load as load_constitution
    from .templates import TemplateStore

    root = repo_root()
    cdir = root / "constitution"
    if (cdir / st.SUMMARIES_FILE).exists() and not force:
        raise click.ClickException(f"{cdir / st.SUMMARIES_FILE} exists; the index is frozen (--force)")
    md, sections, snapshot = load_constitution(cdir)
    templates = TemplateStore(root / "prompts")
    reqs = st.summarize_requests(md, sections, templates, model=model)
    engine = _tags_engine(concurrency, click.echo, model)
    wave_dir = root / "data" / "runs" / "section-tags" / "summarize"
    outcomes = engine.run_wave(
        wave_dir, reqs, wave_name="summarize",
        template_hashes=templates.hashes([st.SUMMARIZE_TEMPLATE]),
        upstream_hash=snapshot["sha256"],
    )
    data = st.freeze_summaries(
        cdir, outcomes, sections, snapshot, model=model,
        template_hash=templates.hash(st.SUMMARIZE_TEMPLATE), force=force,
    )
    click.echo(f"froze {len(data['sections'])} summaries, sha {data['sha256'][:12]} -> {cdir / st.SUMMARIES_FILE}")


@section_tags_group.command("run")
@click.option("--corpus", type=click.Choice(["stories", "advice"]), required=True)
@click.option("--release", default="production-20260831-v2", show_default=True)
@click.option("--model", default="claude-opus-5", show_default=True)
@click.option("--concurrency", type=int, default=200, show_default=True)
@click.option("--limit", type=int, default=None, help="Only the first N records (pilot).")
@click.option("--repair-rounds", type=int, default=2, show_default=True)
def section_tags_run(corpus: str, release: str, model: str, concurrency: int,
                     limit: int | None, repair_rounds: int) -> None:
    """Tag one corpus's final against the frozen section index (live lowprio, resumable)."""
    from . import section_tags as st

    root = repo_root()
    run = st.TagRun(
        corpus=corpus, release=release,
        source=root / "data" / corpus / f"final-{release}.jsonl",
        out_dir=root / "data" / corpus,
        wave_root=root / "data" / "runs" / "section-tags" / release,
        model=model, repair_rounds=repair_rounds, limit=limit,
    )
    if not run.source.exists():
        raise click.ClickException(f"source final missing: {run.source}")
    engine = _tags_engine(concurrency, click.echo, model)
    manifest = st.run_tagging(run, engine, on_event=lambda m: click.echo(f"{_stamp()} {m}"))
    click.echo(json.dumps({k: manifest[k] for k in ("n_records", "n_requests_issued", "status_counts",
                                                     "n_none", "wall_clock_s", "tags_path")}, indent=1))


if __name__ == "__main__":
    main()
