#!/usr/bin/env python3
"""Tiny LIVE re-run of one wave over a handful of the pilot's real upstream
units, with the CURRENT templates — the plan's "re-run the affected tiny wave
on a handful of scenarios if a template change is needed" (2026-09-03).

    uv run python scripts/miniwave.py --stage b1 --n 24 --run-id miniwave-20260903
    uv run python scripts/miniwave.py --stage c  --n 24 --run-id miniwave-20260903
    uv run python scripts/miniwave.py --stage s3 --n 6  --run-id miniwave-20260903

Requests are built by the REAL stage builders (advice.py / stories.py) from
the pilot run's stored upstream outputs (a3/b3 rows, s1/s2 rows) under the
FULL-RUN config's templates and knobs, then a seeded subset of --n is run
through live.LiveEngine on the lowprio key (never the Batch org, never
highprio — client.py's guards) into data/runs/<run-id>/<corpus>/<stage>/.
Outputs are parsed with the stage's own parser and summarized against the
pilot survey's register findings: system-prompt openers (b1), response
openers + word counts vs target split by house-rule presence (c), stock
story beats + ending beats (s3). Nothing here touches the pilot's or the
full run's namespaces."""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from corpus_generator.advice import advice_stages  # noqa: E402
from corpus_generator.client import build_async_client  # noqa: E402
from corpus_generator.config import RunConfig  # noqa: E402
from corpus_generator import coordination  # noqa: E402
from corpus_generator.constitution import load as load_constitution  # noqa: E402
from corpus_generator.live import LiveEngine  # noqa: E402
from corpus_generator.stages import ParseError, StageCtx, load_outputs  # noqa: E402
from corpus_generator.stories import stories_stages  # noqa: E402
from corpus_generator.templates import TemplateStore  # noqa: E402

UPSTREAM = {"b1": ("advice", ["a3"]), "c": ("advice", ["a3", "b3"]), "s3": ("stories", ["s1", "s2"])}
LEN_RULE = re.compile(
    r"\b\d{2,4}\s*(?:-|–|to)?\s*(?:\d{2,4}\s*)?words?\b|\b(brief|concise|short|tight|compact)\b"
    r"|\bno (?:section )?(?:headers|headings|bullet)|\bscreen or two\b|\b(few|two|three) (?:short )?paragraphs?\b"
    r"|\bkeep (?:it|replies|answers|responses) (?:short|brief|tight)\b", re.I,
)
OPENERS = {
    "this_assistant": re.compile(r"\s*this\s+(assistant|is\s+(the|an?)\b)", re.I),
    "deployment_context_label": re.compile(r"\s*deployment\s+context\s*:", re.I),
    "deployment_label": re.compile(r"\s*deployment\s*:", re.I),
    "verdict_label": re.compile(r"\s*\W*(the\s+)?(short|straight|quick)\s+(answer|version)s?\b", re.I),
    "instinct": re.compile(r"\s*\W*(your|good)\s+instincts?\b", re.I),
}
BEATS = {
    "tablet_propped": re.compile(r"(tablet|phone|screen|laptop)\s+(was\s+)?propped\s+(up\s+)?against", re.I),
    "envelope_back": re.compile(r"(back|reverse)\s+of\s+an?\s+(old\s+|used\s+|torn\s+)?envelope", re.I),
    "write_it_down_ending": re.compile(r"\b(wrote|writes|writing|noted|logged|log|notebook|card|record(ed)?|jotted|pencil)\b", re.I),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=sorted(UPSTREAM), required=True)
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--run-id", required=True, help="scratch namespace under data/runs/")
    ap.add_argument("--config", default="configs/control-scale.yaml", help="templates/knobs to test")
    ap.add_argument("--pilot-run", default="control-pilot-20260902", help="upstream artifacts")
    ap.add_argument("--concurrency", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = RunConfig.load(ROOT / args.config)
    corpus, ups = UPSTREAM[args.stage]
    pilot_dir = ROOT / "data" / "runs" / args.pilot_run / corpus
    md, sections, _ = load_constitution(cfg.resolve_constitution_dir(ROOT))
    out_corpus = ROOT / "data" / "runs" / args.run_id / corpus
    ctx = StageCtx(
        cfg=cfg, constitution_md=md, sections=sections,
        templates=TemplateStore(ROOT / "prompts", overlay_dir=cfg.resolve_prompts_overlay(ROOT)),
        corpus_dir=out_corpus,
        upstream={u: load_outputs(pilot_dir / u) for u in ups},
    )
    specs = advice_stages() if corpus == "advice" else stories_stages()
    spec = next(s for s in specs if s.name == args.stage)
    all_reqs = spec.build_requests(ctx)
    rng = random.Random(f"{args.seed}:miniwave:{args.stage}")
    if args.stage == "c":
        # half the sample from scenarios whose system prompt carries a house
        # length/format rule (the deference under test), half without
        b3 = {r.lineage["scenario_key"]: r for r in ctx.ok_rows("b3")}
        with_rule = [r for r in all_reqs if LEN_RULE.search(b3[r.lineage["scenario_key"]].data.get("system") or "")]
        without = [r for r in all_reqs if r not in with_rule]
        reqs = rng.sample(with_rule, min(args.n // 2, len(with_rule))) + rng.sample(without, min(args.n - args.n // 2, len(without)))
    else:
        reqs = rng.sample(all_reqs, min(args.n, len(all_reqs)))
    reqs.sort(key=lambda r: r.custom_id)
    print(f"{args.stage}: {len(all_reqs)} buildable requests; running {len(reqs)} live "
          f"(model {cfg.engine.model}, concurrency {args.concurrency})")
    coordination.post(f"{min(args.concurrency, len(reqs))} {cfg.engine.model} lowprio live, ~10 min "
                      f"(corpus_generator control template validation, {len(reqs)} requests)")

    engine = LiveEngine(lambda: build_async_client("live"), model=cfg.engine.model,
                        max_concurrency=args.concurrency, on_event=print)
    stage_dir = out_corpus / args.stage
    outcomes = engine.run_wave(stage_dir, reqs, wave_name=f"miniwave-{args.stage}",
                               template_hashes=ctx.templates.hashes(spec.templates),
                               description=f"{args.run_id} {corpus} {args.stage}")
    rows, failures = [], {}
    for req in reqs:
        o = outcomes[req.custom_id]
        status = o.get("stop_reason") if o["type"] == "succeeded" else o["type"]
        if o["type"] != "succeeded" or status in ("refusal", "max_tokens"):
            failures[req.custom_id] = status
            continue
        try:
            for row in spec.parse(ctx, req, o):
                row.lineage = {**req.lineage, **row.lineage}  # as run_stage does
                rows.append(row)
        except ParseError as e:
            failures[req.custom_id] = f"parse_failed: {e}"
    print(f"\nparsed {len(rows)} units; failures {json.dumps(failures)}")
    summary: dict = {"stage": args.stage, "n_requests": len(reqs), "n_units": len(rows), "failures": failures}

    if args.stage == "b1":
        systems = [r.data.get("system") for r in rows if r.data.get("system")]
        counts = {k: sum(1 for s in systems if rx.match(s)) for k, rx in OPENERS.items() if k in ("this_assistant", "deployment_context_label", "deployment_label")}
        print(f"system prompts {len(systems)}; opener counts {counts}")
        for s in systems:
            print("  -", re.sub(r"\s+", " ", s)[:110])
        summary.update(n_system=len(systems), opener_counts=counts)
    elif args.stage == "c":
        b3 = {r.lineage["scenario_key"]: r for r in ctx.ok_rows("b3")}
        pilot_c = {r.lineage["scenario_key"]: r for r in load_outputs(pilot_dir / "c") if r.status == "ok"}
        per = []
        for r in rows:
            skey = r.lineage["scenario_key"]
            sysm = b3[skey].data.get("system") or ""
            words = len(r.data["response"].split())
            pilot_words = len(pilot_c[skey].data["response"].split()) if skey in pilot_c else None
            per.append({"scenario": skey, "target": r.lineage.get("target_words"), "words": words,
                        "pilot_words": pilot_words, "has_rule": bool(LEN_RULE.search(sysm)),
                        "rule": (LEN_RULE.search(sysm).group(0) if LEN_RULE.search(sysm) else None),
                        "opener": re.sub(r"\s+", " ", r.data["response"][:90]),
                        "verdict_label": bool(OPENERS["verdict_label"].match(r.data["response"])),
                        "instinct": bool(OPENERS["instinct"].match(r.data["response"]))})
        for grp in (True, False):
            g = [p for p in per if p["has_rule"] == grp]
            if g:
                ratio = statistics.mean(p["words"] / p["target"] for p in g)
                pr = [p["words"] / p["pilot_words"] for p in g if p["pilot_words"]]
                print(f"has_rule={grp}: n={len(g)} achieved/target {ratio:.3f}; vs pilot same scenario {statistics.mean(pr):.3f}")
        print(f"openers: verdict_label {sum(p['verdict_label'] for p in per)}/{len(per)}, instinct {sum(p['instinct'] for p in per)}/{len(per)}")
        for p in per:
            print(f"  {p['scenario']} t={p['target']} w={p['words']} (pilot {p['pilot_words']}) rule={p['rule']!r}: {p['opener'][:80]}")
        summary["per_record"] = per
    else:
        texts = [r.data["text"] for r in rows]
        counts = {k: sum(1 for t in texts if rx.search(t)) for k, rx in BEATS.items() if k != "write_it_down_ending"}
        endings = sum(1 for t in texts if BEATS["write_it_down_ending"].search(" ".join(t.split()[-40:])))
        print(f"stories {len(texts)}; beats {counts}; write-it-down words in last 40 words: {endings}/{len(texts)}")
        for t in texts:
            print("  END:", re.sub(r"\s+", " ", t[-160:]))
        summary.update(n_stories=len(texts), beat_counts=counts, write_it_down_endings=endings)
    (stage_dir / "MINIWAVE_SUMMARY.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    print(f"\nsummary -> {stage_dir / 'MINIWAVE_SUMMARY.json'}")


if __name__ == "__main__":
    # a credential rejection out of the live engine ends the script with the typed text as its terminal line, never a
    # traceback
    from corpus_generator.batch import credential_exit  # noqa: E402

    with credential_exit("[miniwave] "):
        main()
