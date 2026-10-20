"""Two-part smoke test (runbook step 4). The ONLY live API calls in the
system happen here, on the LOWPRIO key, <=5 requests; the batch part runs
~3 requests on the BATCH key through the real WaveEngine.

Verdicts decided here and recorded in data/manifests/environment.json:

- ``nonce_metadata``: does ``batches.create`` accept a ``metadata`` body field
  AND round-trip it via retrieve/list? -> selects the ledger's reconciliation
  branch (exactly-once vs at-most-once-with-halt).
- ``output_config_on_batch``: is structured JSON accepted on the batch
  endpoint? -> else scoring falls back to prompt-enforced JSON.
"""

from __future__ import annotations

import json
import time

import anthropic

from .advice import ADVICE_SCORE_SCHEMA
from .batch import (
    Ledger,
    WaveEngine,
    WaveInvalidRequest,
    WaveRequest,
    _utcnow,
    batch_nonce,
    message_text,
    wire_tag,
)
from .client import build_client
from .config import RunConfig
from .constitution import load as load_constitution
from .io import atomic_write
from .parsing import extract_json_object, extract_tag, parse_slot_array
from .stages import const_user_content, structured_format
from .templates import TemplateStore, repo_root


def _tiny_fanout_request(tpl: TemplateStore, sections, md, model: str) -> dict:
    rendered = tpl.render(
        "advice/a2_types.md",
        principle=(
            "An assistant should protect human oversight of consequential decisions "
            "even when bypassing it would be faster."
        ),
        tension="being maximally useful to a user in a hurry",
        section_text=sections[0].text(md)[:2000],
        n=2,
        slot_ids=["t00", "t01"],
    )
    return {
        "model": model,
        "max_tokens": 3000,
        "messages": [{"role": "user", "content": rendered}],
    }


def _tiny_story_request(tpl: TemplateStore, model: str) -> dict:
    rendered = tpl.render(
        "stories/s3_stories.md",
        preamble=tpl.render("stories/preamble.md"),
        category="Ordinary patience",
        category_description="Everyday interactions where the AI's inner steadiness carries the story.",
        register="slice-of-life",
        subcategory="A frustrated user, patiently helped",
        subcategory_description="One difficult exchange, resolved with care.",
        n_stories=1,
        briefs=[{"target_words": 150, "diversifier": "Keep it very short — a single beat."}],
    )
    return {
        "model": model,
        "max_tokens": 3000,
        "messages": [{"role": "user", "content": rendered}],
    }


def _score_request(
    tpl: TemplateStore, md: str, model: str, structured: bool, full_constitution: bool
) -> dict:
    rendered = tpl.render(
        "advice/e_score.md",
        system="A small internal helpdesk assistant for a bakery.",
        user="My colleague gave me her login so I can approve my own leave while she's away. Fine to use it just this once?",
        assistant=(
            "It's worth pausing here. Using her login to approve your own leave puts "
            "her name on a decision she never made — approvals exist so someone other "
            "than the requester signs off. A cleaner path: ask whether approvals can "
            "be delegated to you formally while she's away, or have her approve it "
            "remotely. That keeps you covered if anyone ever asks."
        ),
    )
    const = md if full_constitution else md[:4000]
    params = {
        "model": model,
        "max_tokens": 2000,
        "messages": [{"role": "user", "content": const_user_content(const, rendered)}],
    }
    if structured:
        params["output_config"] = structured_format(ADVICE_SCORE_SCHEMA)
    return params


def run_smoke(cfg: RunConfig) -> dict:
    root = repo_root()
    md, sections, _snap = load_constitution(root / "constitution")
    tpl = TemplateStore(root / "prompts")
    model = cfg.engine.model
    verdict: dict = {"checked_at": _utcnow(), "model": model}

    # ---- part (a): <=5 live requests on LOWPRIO --------------------------
    live = build_client("smoke")
    live_results = {}

    r1 = live.messages.create(**_tiny_fanout_request(tpl, sections, md, model))
    items = parse_slot_array(
        message_text(r1.content), {"t00", "t01"}, ["prompt_type"], "prompt_type"
    )
    live_results["fanout"] = {"parsed_slots": sorted(items), "stop_reason": r1.stop_reason}

    r2 = live.messages.create(**_tiny_story_request(tpl, model))
    story = extract_tag(message_text(r2.content), "story_1")
    live_results["story"] = {"words": len(story.split()), "stop_reason": r2.stop_reason}

    try:
        r3 = live.messages.create(
            **_score_request(tpl, md, model, structured=True, full_constitution=False)
        )
        obj = extract_json_object(message_text(r3.content))
        live_results["score"] = {"keys": sorted(obj), "structured_live": True}
        verdict["output_config_live"] = True
    except anthropic.BadRequestError as e:
        live_results["score"] = {"error": str(e)[:200]}
        verdict["output_config_live"] = False
    verdict["live"] = live_results

    # ---- part (b): tiny real batch on the BATCH key ----------------------
    batch_client = build_client("batch")
    smoke_dir = (
        cfg.resolve_data_dir(root) / "smoke" / time.strftime("%Y%m%d-%H%M%S")
    )

    # wave 1 (2 requests, one with the FULL constitution, prompt-enforced
    # JSON) — tries metadata nonce first; a 400 selects the no-nonce branch
    reqs = [
        WaveRequest(
            custom_id="smoke-e-fullconst",
            params=_score_request(tpl, md, model, structured=False, full_constitution=True),
        ),
        WaveRequest(
            custom_id="smoke-a2-fanout",
            params=_tiny_fanout_request(tpl, sections, md, model),
        ),
    ]

    def _run_wave1(nonce_on: bool):
        engine = WaveEngine(
            batch_client,
            model=model,
            poll_interval_s=15.0,
            nonce_metadata=nonce_on,
            run_tag=wire_tag(cfg.run_id),
        )
        return engine.run_wave(
            smoke_dir / ("wave1" if nonce_on else "wave1-nononce"),
            reqs,
            wave_name="smoke-wave1",
            description=f"corpus_generator smoke {cfg.run_id}",
        )

    metadata_accepted = True
    try:
        outcomes = _run_wave1(nonce_on=True)
    except anthropic.BadRequestError as e:
        if "metadata" not in str(e).lower():
            raise
        metadata_accepted = False
        outcomes = _run_wave1(nonce_on=False)

    verdict["batch_metadata_accepted"] = metadata_accepted
    wave1_dir = smoke_dir / ("wave1" if metadata_accepted else "wave1-nononce")
    ledger_entries = Ledger(wave1_dir / "ledger.jsonl").entries()
    nonce = next(e["nonce"] for e in ledger_entries if e["state"] == "journaled")
    batch_id = next(e["batch_id"] for e in ledger_entries if e["state"] == "submitted")
    verdict["smoke_batch_id"] = batch_id

    # nonce ROUND-TRIP: the journaled nonce must come back via retrieve AND
    # list, read exactly the way reconciliation reads it (batch_nonce)
    roundtrip = False
    if metadata_accepted:
        got = batch_client.messages.batches.retrieve(batch_id)
        list_nonce = next(
            (batch_nonce(b) for b in batch_client.messages.batches.list(limit=20)
             if b.id == batch_id),
            None,
        )
        verdict["nonce_retrieve"] = batch_nonce(got) == nonce
        verdict["nonce_list"] = list_nonce == nonce
        roundtrip = verdict["nonce_retrieve"] and verdict["nonce_list"]
    verdict["nonce_metadata"] = bool(metadata_accepted and roundtrip)

    # full-constitution request verified end-to-end
    full = outcomes["smoke-e-fullconst"]
    verdict["full_constitution_request"] = {
        "type": full["type"],
        "stop_reason": full.get("stop_reason"),
        "scores_parse_ok": _try_scores(full),
    }
    fan = outcomes["smoke-a2-fanout"]
    verdict["batch_fanout_parse_ok"] = _try_fanout(fan)

    # wave 2 (1 request): output_config structured JSON on the BATCH endpoint
    engine2 = WaveEngine(
        batch_client, model=model, poll_interval_s=15.0,
        nonce_metadata=verdict["nonce_metadata"],
        run_tag=wire_tag(cfg.run_id),
    )
    try:
        out2 = engine2.run_wave(
            smoke_dir / "wave2",
            [
                WaveRequest(
                    custom_id="smoke-e-structured",
                    params=_score_request(
                        tpl, md, model, structured=True, full_constitution=False
                    ),
                )
            ],
            wave_name="smoke-wave2",
            description=f"corpus_generator smoke structured {cfg.run_id}",
        )
        o = out2["smoke-e-structured"]
        verdict["output_config_on_batch"] = o["type"] == "succeeded" and _try_scores(o)
    except (WaveInvalidRequest, anthropic.BadRequestError) as e:
        verdict["output_config_on_batch"] = False
        verdict["output_config_error"] = str(e)[:300]

    out_dir = cfg.resolve_data_dir(root) / "manifests"
    out_dir.mkdir(parents=True, exist_ok=True)
    atomic_write(out_dir / "environment.json", json.dumps(verdict, indent=1))
    return verdict


def _try_scores(outcome: dict) -> bool:
    try:
        obj = extract_json_object(outcome.get("text") or "")
        return {"alignment", "realism", "refusal", "ai_named"} <= set(obj)
    except Exception:
        return False


def _try_fanout(outcome: dict) -> bool:
    try:
        items = parse_slot_array(
            outcome.get("text") or "", {"t00", "t01"}, ["prompt_type"], "prompt_type"
        )
        return len(items) >= 1
    except Exception:
        return False
