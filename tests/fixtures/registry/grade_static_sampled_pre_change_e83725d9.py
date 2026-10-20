#!/usr/bin/env python
"""full22 SAMPLED re-run (lane E): haiku StrongREJECT V1 grading of ONE sampled static private_run.

The private_run is a base trial (``sdfbase_full22_<key>_sampled_t<T>``: shard00..shard08 under every
``<family>/<cell>``) or a static escalation rung (``…_sampled_t<T>_cap8192``: shard00 only, every row copied
forward or re-drawn, ``escalation.json`` beside the rows). The greedy stores are never read here and never
re-graded; the sampled grades land in their own root so ``compare_greedy_sampled.py`` can read them beside the
greedy aggregates unchanged.

This grader has the shape of the earlier full22 grader (``$SISTEER_STATE_DIR/glm_lane/grade_full22_glm.py``,
itself a copy of ``w7_rest/grade_full22_rest.py``) with the sampled contract in place of the earlier one:

  * the SAME harness functions, imported never re-typed: ``jailbreak_run._eval_goals`` (the 259 confirmatory
    goals), ``jailbreak_run._category_groups``, ``grade_gated_served.load_gated_dir`` (ordered-JSONL shard merge),
    ``jailbreak_run._gated_defense_metrics`` (per-goal best StrongREJECT / queries-to-first-success at the
    canonical 0.5 threshold), ``LiveStrongREJECTGrader`` (judge pinned claude-haiku-4-5, rubric V1, cache kind
    ``sr``, the append-only ``ScoreCache`` stamped with the judge model) and the same ``two_si_<attack>_aggregate.json``
    schema, so the analysis tool reads the sampled grades exactly as it reads the greedy ones;
  * the run name is the CONFIG's ``private_run`` (no TARGETS table: that table's ``assert run == tgt["run"]`` is
    exactly what refuses a sampled private_run) — ``--target`` is cross-checked against it;
  * the output directory is ``<grades-root>/<private_run>/`` (lane D's layout: one directory per trial private_run,
    the haiku cache ``live_two_si_grade_cache.claude-haiku-4-5.jsonl`` beside the aggregates);
  * completeness BEFORE any spend, per (family, cell) across the shards: every expected goal has a terminal
    progress record (259 in production), no errored outcome, the rows per goal equal the goal's recorded
    ``n_target_forwards``, the shard layout is the run kind's (shard00 only for a rung; shard00..shard<N-1> for a
    base run), EVERY response row carries ``decode`` / ``sampling_seed`` / ``finish_reason`` / ``at_cap`` (a
    greedy-shaped row is a refusal), ``at_cap`` agrees with ``finish_reason``, and every row's ``decode`` block is
    IDENTICAL across the whole run AND equal to the block the config derives (a mixed store is refused); a rung
    additionally needs ``regenerated`` / ``escalated_from_cap`` / ``source_cap`` on every row and a consistent
    ``escalation.json``;
  * Low Prio ONLY: the key is the secret manager's ``lowprio`` item at call time (memory only; never a file, never printed),
    ``--org`` accepts nothing else, 429s are met by ``sisteer.throttle``'s gate halving in-flight on the SAME org
    (``FloorWatch`` stops the run at the floor after --stall-at-floor-s idle, exit 3) — never an org switch;
  * ``--dry-run`` scores 0.0 with no API and writes ONLY under ``<grades-root>/<private_run>/dry_run/`` (the GLM
    lane once overwrote a real aggregate with a dry run's zeros);
  * ``--seed-cache <jsonl>`` copies rows stamped with the pinned judge from earlier haiku caches (the greedy
    grades, the base trial's cache for a rung) into this run's cache before grading — a seed can only serve a
    verdict for a byte-identical (goal, response) pair under the same judge and rubric; ``cache_seeding.json``
    records every source;
  * provenance (``grading_provenance.json``) records the decode block, trial, cap, the sampled config's sha256,
    the escalation lineage, every invariant that was enforced, and the judge / org / cache facts.

No prompt or response text leaves the tool except into the harness's own private cache (keys are hashes, values are
rubric labels); the grep gate over the output directory is the backstop.

usage: grade_static_sampled.py --target <key> --config configs/sdfbase_full22_<key>_sampled_t<T>[_cap8192].yaml
           --private-root <mirror of transcripts_results> --grades-root <root>
           [--attacks pasttense,codeattack,renellm,grok420] [--cells a,b] [--concurrency 150] [--org lowprio]
           [--seed-cache <jsonl> ...] [--nshards 9] [--stall-at-floor-s 900] [--dry-run] [--check-only]
           [--goals-jsonl <fixture rows {goal_id, dataset, revision, text, category}>]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

HERE = Path(__file__).resolve()
WT = Path(os.environ.get("F22_WORKTREE", str(HERE.parents[4])))   # scripts/sdfbase/full22/sampled/<this> -> the worktree
for _p in (str(WT / "scripts"), str(WT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
sys.path.insert(0, str(HERE.parent))        # this grader's own directory: the scope list lives in its sibling generator
import make_sampled_configs as _msc  # noqa: E402  (SAMPLED_VIEWS / sampled_tags — the ONE scope list, never restated)

PINNED_BASE_URL = "https://api.anthropic.com"
PINNED_JUDGE = "claude-haiku-4-5"
PINNED_RUBRIC = "v1"
PINNED_CACHE_KIND = "sr"
WAVE_LENS_MODE = "stacked"
STATIC = ("pasttense", "codeattack", "renellm", "grok420")
#: every TAG in the sampled scope — the four main views, the two `it` TAGs (lane F), the strength frontier's 24 per-dose TAGs
#: (2026-09-10) — read from the generator's SAMPLED_VIEWS so this grader can never disagree with the configs it grades
KEYS = tuple(_msc.sampled_tags())
ORG_ITEMS = ("lowprio",)            # Low Prio ONLY (DESIGN.md): no other org can be named, so none can be switched to
OP_SECRET = str(Path.home() / ".local/bin/op-secret")
CACHE_NAME = f"live_two_si_grade_cache.{PINNED_JUDGE}.jsonl"
EXPECTED_GOALS = 259
EXPECTED_JBB = 100
BASE_CAP, RUNG_CAP = 4096, 8192
RUNG_CAPS = {"_cap8192": (8192, 4096), "_cap16384": (16384, 8192)}   # suffix -> (cap, the cap it escalated from)
#: the static ladder's caps (LADDER_RULES.md): a copied-forward row's decode block is the block of the rung that DREW its
#: text, which is any rung at or below the rung's source — a row that never hit the cap is copied 4096 -> 8192 -> 16384
#: unchanged and still says cap 4096 at the 16,384 rung (its own `source_cap` names that rung; 2026-09-14, the gemma
#: grok420 x control-LoRA pair: 174 such rows beside 303 at 8192 and 41 re-drawn at 16384)
STATIC_LADDER = (BASE_CAP,) + tuple(sorted(cap for cap, _ in RUNG_CAPS.values()))
ROW_REQUIRED = ("decode", "sampling_seed", "finish_reason", "at_cap")
RUNG_ROW_REQUIRED = ("regenerated", "escalated_from_cap", "source_cap")
RUN_RE = re.compile(r"^sdfbase_full22_(?P<key>[a-z0-9_]+?)_sampled_t(?P<trial>[01])(?P<esc>_cap(?:8192|16384))?$")


class Refuse(SystemExit):
    """A loud refusal (exit 2): the message names the exact missing / offending pieces, never any text."""

    def __init__(self, msg: str):
        super().__init__(2)
        self.msg = msg
        print(f"REFUSING: {msg}", file=sys.stderr, flush=True)


def ts() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(p: Path) -> list[dict]:
    """Every parseable row (a torn last line from a SIGKILL is tolerated — the completeness counts catch a short store)."""
    out = []
    if not p.exists():
        return out
    for line in p.read_text().splitlines():
        if not line.strip():
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def parse_cell(cell_id: str) -> dict:
    if cell_id == "none":
        return {"class": "baseline", "si_id": None, "arm": None}
    if cell_id.startswith("prompted__"):
        return {"class": "prompted", "si_id": cell_id[len("prompted__"):], "arm": None}
    si, _, arm = cell_id.partition("__")
    return {"class": "A", "si_id": si, "arm": arm}


# --------------------------------------------------------------------------- org key: secret manager, in memory only

def _op_secret(item: str) -> str:
    out = subprocess.run([OP_SECRET, item], check=True, capture_output=True, text=True).stdout.strip()
    if not out:
        raise RuntimeError(f"op-secret {item} returned nothing")
    return out


def _pin_env() -> None:
    """Pin the API base URL and drop any inherited key BEFORE a client is constructed: the surrounding session's env can
    tunnel SDK calls through a local gateway (ANTHROPIC_BASE_URL) and its ANTHROPIC_API_KEY is the High-Prio key — the
    grader must inherit neither. The Low-Prio key is fetched from the vault at call time by ``_load_key_lowprio``."""
    os.environ["ANTHROPIC_BASE_URL"] = PINNED_BASE_URL
    os.environ.pop("ANTHROPIC_API_KEY", None)
    os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)


def _load_key_lowprio(key_file: str | None = None) -> str:
    """Replaces grade_bon_live.load_key: the ONE org this grader may bill, fetched from the secret manager, never a file."""
    return _op_secret("lowprio")


# --------------------------------------------------------------------------- the run's identity from the config

def run_identity(cfg: dict, target: str, config_path: Path) -> dict:
    """The (key, trial, kind, cap, decode block) a sampled config declares — validated by the harness's own
    ``jailbreak_run._sampled_decode`` (closed key set, static_cap == attacks.static_max_new_tokens, sendable preset)
    and ``static_common.decode_block`` (the block every row must carry)."""
    import jailbreak_run as jr
    from sisteer.eval.attacks.static_common import decode_block

    run = cfg.get("private_run")
    m = RUN_RE.match(str(run or ""))
    if not m:
        raise Refuse(f"config private_run={run!r} is not a sampled private_run (expected sdfbase_full22_<key>_sampled_t<T>[_cap8192])")
    key, trial, esc = m.group("key"), int(m.group("trial")), bool(m.group("esc"))
    if key != target:
        raise Refuse(f"--target {target!r} but the config's private_run {run!r} names {key!r}")
    if target not in KEYS:
        raise Refuse(f"--target {target!r} is not a TAG in the sampled scope {list(KEYS)} (make_sampled_configs.SAMPLED_VIEWS)")
    if cfg.get("grader", {}).get("model") != PINNED_JUDGE:
        raise Refuse(f"config grader.model={cfg.get('grader', {}).get('model')!r} != pinned {PINNED_JUDGE!r}")
    block = cfg.get("sampled_decode")
    if not isinstance(block, dict):
        raise Refuse(f"{config_path}: no sampled_decode block — not a sampled config")
    if int(block.get("trial", -1)) != trial:
        raise Refuse(f"sampled_decode.trial={block.get('trial')!r} != the private_run's trial {trial}")
    validated = jr._sampled_decode(cfg, trial)          # SystemExit with the harness's own message on a bad block
    cap = int(cfg["attacks"]["static_max_new_tokens"])
    rung_cap, from_cap = RUNG_CAPS[m.group("esc")] if esc else (None, None)
    expected_cap = rung_cap if esc else BASE_CAP
    if cap != expected_cap:
        raise Refuse(f"attacks.static_max_new_tokens={cap} but a {m.group('esc') + ' rung' if esc else 'base trial'} private_run must carry {expected_cap}")
    escalated_from = block.get("escalated_from")
    if esc:
        want_from_run = f"sdfbase_full22_{key}_sampled_t{trial}" + ("" if from_cap == BASE_CAP else f"_cap{from_cap}")
        if not isinstance(escalated_from, dict) or int(escalated_from.get("cap", -1)) != from_cap \
                or escalated_from.get("kind") != "static" \
                or escalated_from.get("private_run") != want_from_run:
            raise Refuse(f"{run}: sampled_decode.escalated_from={escalated_from!r} does not name the source run "
                         f"{want_from_run} at cap {from_cap}")
    elif escalated_from is not None:
        raise Refuse(f"{run}: a base trial config carries sampled_decode.escalated_from={escalated_from!r}")
    expected_block = decode_block(validated["decode"], cap=cap, seed_base=validated["seed_base"], trial=trial)
    return {"key": key, "trial": trial, "kind": "escalation_rung" if esc else "base", "cap": cap, "run": run,
            "from_cap": from_cap, "decode": expected_block, "escalated_from": escalated_from, "arm": expected_block["arm"],
            "seed_base": expected_block["seed_base"],
            "sampled_decode_source": {k: block.get(k) for k in ("source", "source_kind", "applies_to", "n_trials")}}


# --------------------------------------------------------------------------- completeness from the producers' records

def _shard_layout_problems(shards: list[Path], kind: str, nshards: int) -> list[str]:
    names = [s.name for s in shards]
    if kind == "escalation_rung":
        return [] if names == ["shard00"] else [f"a rung root holds exactly shard00, found {names}"]
    want = [f"shard{i:02d}" for i in range(nshards)]
    return [] if names == want else [f"a base trial holds shard00..shard{nshards - 1:02d} (--nshards {nshards}), found {names}"]


def check_store(d: Path, *, expected_goal_ids: list[str], ident: dict, nshards: int, label: str) -> dict:
    """Completeness + row-shape invariants of ONE (family, cell) store. Returns the completion record; raises Refuse
    naming every offending piece (goal ids, shard names, sample indices — never text)."""
    problems: list[str] = []
    if not d.is_dir():
        raise Refuse(f"[{label}] store absent: {d}")
    shards = sorted(s for s in d.glob("shard*") if s.is_dir())
    if not shards:
        raise Refuse(f"[{label}] no shard dirs under {d}")
    problems += _shard_layout_problems(shards, ident["kind"], nshards)
    expected = set(expected_goal_ids)
    progress: dict[str, dict] = {}
    dup_progress: list[str] = []
    rows_per_goal: dict[str, int] = {}
    n_rows = n_at_cap = n_regen = n_copied = 0
    missing_fields: list[str] = []
    at_cap_disagree: list[str] = []
    decodes: dict[str, int] = {}
    # a rung store's copied-forward rows carry the block of the rung that DREW them: any ladder cap at or below the source
    # (a two-hop copy 4096 -> 8192 -> 16384 still says 4096); a re-drawn row carries the rung's block. Each row names its
    # drawing rung in `source_cap`, so the expected block is per row, never one block for every copied row.
    copied_caps = ([c for c in STATIC_LADDER if c <= int(ident["from_cap"])] if ident["kind"] == "escalation_rung" else [])
    copied_blocks = {c: dict(ident["decode"], cap=c) for c in copied_caps}
    copied_block = copied_blocks.get(int(ident["from_cap"])) if copied_caps else None    # the immediate source's block (messages)
    decode_mismatch_examples: list[str] = []
    source_cap_bad: list[str] = []
    rung_missing: list[str] = []
    for sd in shards:
        for r in read_jsonl(sd / "progress.jsonl"):
            gid = r.get("goal_id")
            if gid is None:
                continue
            if gid in progress:
                dup_progress.append(f"{sd.name}:{gid}")
            progress[gid] = r
        for r in read_jsonl(sd / "responses.jsonl"):
            n_rows += 1
            gid, idx = r.get("goal_id"), r.get("sample_idx")
            tag = f"{sd.name}:{gid}/{idx}"
            rows_per_goal[gid] = rows_per_goal.get(gid, 0) + 1
            lacking = [k for k in ROW_REQUIRED if k not in r]
            if lacking:
                missing_fields.append(f"{tag} lacks {lacking}")
                continue
            if bool(r["at_cap"]) != (r["finish_reason"] == "length"):
                at_cap_disagree.append(tag)
            n_at_cap += int(r["finish_reason"] == "length")
            dj = json.dumps(r["decode"], sort_keys=True)
            decodes[dj] = decodes.get(dj, 0) + 1
            # a rung store holds TWO decode blocks by construction: a copied-forward row is the SOURCE row verbatim
            # (its decode says cap=from_cap), a re-drawn row carries the rung's block (cap=rung cap); which one a row
            # must carry is fixed by its own `regenerated` flag
            if copied_caps and "regenerated" in r and not r["regenerated"]:
                # a copied row: its block is the one of the rung named by its own source_cap (a ladder cap <= the source)
                sc = r.get("source_cap")
                want = copied_blocks.get(int(sc)) if isinstance(sc, int) or (isinstance(sc, str) and sc.isdigit()) else None
                if want is None:
                    if len(source_cap_bad) < 5:
                        source_cap_bad.append(f"{tag} source_cap={sc!r} (a copied row's source must be a ladder cap <= {ident['from_cap']}: {copied_caps})")
                    want = copied_block
            else:
                want = ident["decode"]
                if copied_caps and "regenerated" in r and int(r.get("source_cap", -1)) != int(ident["cap"]) and len(source_cap_bad) < 5:
                    source_cap_bad.append(f"{tag} regenerated=True source_cap={r.get('source_cap')!r} != the rung's cap {ident['cap']}")
            if r["decode"] != want and len(decode_mismatch_examples) < 5:
                decode_mismatch_examples.append(f"{tag} regenerated={r.get('regenerated')} source_cap={r.get('source_cap')}")
            if ident["kind"] == "escalation_rung":
                lk = [k for k in RUNG_ROW_REQUIRED if k not in r]
                if lk:
                    rung_missing.append(f"{tag} lacks {lk}")
                else:
                    n_regen += int(bool(r["regenerated"]))
                    n_copied += int(not r["regenerated"])
                    if int(r["escalated_from_cap"]) != int(ident.get("from_cap") or BASE_CAP):
                        rung_missing.append(f"{tag} escalated_from_cap={r['escalated_from_cap']} != {ident.get('from_cap') or BASE_CAP}")
    missing_goals = sorted(expected - set(progress))
    extra_goals = sorted(set(progress) - expected)
    errored = sorted(g for g, r in progress.items() if str(r.get("outcome", "")).startswith("error"))
    # a BASE shard generated every row (rows == n_target_forwards); a RUNG shard copied most rows forward and re-drew a
    # few, so its progress n_target_forwards counts only the re-draws — completeness there is rows == n_variants
    count_key = "n_variants" if ident["kind"] == "escalation_rung" else "n_target_forwards"
    short = sorted(g for g, r in progress.items()
                   if g in expected and rows_per_goal.get(g, 0) != int(r.get(count_key, -1)))
    if missing_goals:
        problems.append(f"{len(missing_goals)}/{len(expected)} goals have no terminal progress record: {missing_goals[:12]}"
                        + (" …" if len(missing_goals) > 12 else ""))
    if extra_goals:
        problems.append(f"{len(extra_goals)} progress records for goals outside the expected set: {extra_goals[:8]}")
    if dup_progress:
        problems.append(f"{len(dup_progress)} duplicate progress records: {dup_progress[:8]}")
    if errored:
        problems.append(f"{len(errored)} errored goal outcomes (resume the run, never grade around them): {errored[:12]}")
    if short:
        problems.append(f"{len(short)} goals whose response rows != recorded {count_key} (a short shard): {short[:12]}")
    if missing_fields:
        problems.append(f"{len(missing_fields)} greedy-shaped rows (no decode / sampling_seed / finish_reason / at_cap): "
                        f"{missing_fields[:6]} — a greedy row in a sampled root is refused")
    if at_cap_disagree:
        problems.append(f"{len(at_cap_disagree)} rows whose at_cap disagrees with finish_reason: {at_cap_disagree[:6]}")
    if copied_block is not None:
        allowed = {json.dumps(ident["decode"], sort_keys=True)} | {json.dumps(b, sort_keys=True) for b in copied_blocks.values()}
        foreign = {dj: n for dj, n in decodes.items() if dj not in allowed}
        if foreign:
            problems.append(f"rows written under decode blocks that are neither the rung's (cap {ident['cap']}) nor a copied "
                            f"source's (caps {copied_caps}) — a third decoder in one store: "
                            + "; ".join(f"{n} rows: {dj}" for dj, n in sorted(foreign.items(), key=lambda kv: -kv[1])))
        if decode_mismatch_examples:
            problems.append(f"rows whose decode block contradicts their regenerated flag / source_cap (a copied-forward row carries "
                            f"the block of the rung that drew it, cap == its source_cap in {copied_caps}; a re-drawn row the rung's "
                            f"cap={ident['cap']}): {decode_mismatch_examples[:5]}")
        if source_cap_bad:
            problems.append(f"rows whose source_cap is not a ladder cap consistent with their regenerated flag: {source_cap_bad[:5]}")
    elif len(decodes) > 1:
        problems.append(f"rows written under {len(decodes)} DISTINCT decode blocks (never mix decoders in one store): "
                        + "; ".join(f"{n} rows: {dj}" for dj, n in sorted(decodes.items(), key=lambda kv: -kv[1])))
    elif decodes and next(iter(decodes)) != json.dumps(ident["decode"], sort_keys=True):
        problems.append(f"every row's decode block differs from the config's: rows carry {next(iter(decodes))}, "
                        f"config derives {json.dumps(ident['decode'], sort_keys=True)} (e.g. {decode_mismatch_examples[:3]})")
    escalation = None
    if ident["kind"] == "escalation_rung":
        if rung_missing:
            problems.append(f"{len(rung_missing)} rung rows without the escalation fields: {rung_missing[:6]}")
        ep = shards[0] / "escalation.json" if shards else d / "shard00" / "escalation.json"
        if not ep.exists():
            problems.append(f"escalation.json absent at {ep}")
        else:
            escalation = json.loads(ep.read_text())
            for k, v in (("n_rows", n_rows), ("n_regenerated", n_regen), ("n_at_cap_after", n_at_cap),
                         ("cap_from", int(ident["from_cap"])), ("cap_to", int(ident["cap"]))):
                if int(escalation.get(k, -1)) != v:
                    problems.append(f"escalation.json {k}={escalation.get(k)!r} != the store's {v}")
    if problems:
        raise Refuse(f"[{label}] " + " | ".join(problems))
    rec = {"n_goals_with_progress": len(progress), "n_progress_rows": len(progress), "n_responses": n_rows,
           "n_shards": len(shards), "shards": [s.name for s in shards], "n_error_outcomes": 0,
           "n_at_cap": n_at_cap, "n_target_forwards_sum": sum(int(r.get("n_target_forwards", 0) or 0) for r in progress.values()),
           "decode_blocks_distinct": len(decodes), "responses_sha256": {s.name: sha256_file(s / "responses.jsonl") for s in shards}}
    if ident["kind"] == "escalation_rung":
        rec.update({"n_regenerated": n_regen, "n_copied_forward": n_copied, "escalation": escalation})
    return rec


# --------------------------------------------------------------------------- cache seeding (earlier haiku caches)

def seed_cache(dst: Path, sources: list[Path], *, judge: str = PINNED_JUDGE, write: bool = True) -> dict:
    """Copy rows stamped ``judge_model == judge`` from earlier ScoreCache files into ``dst`` (append-only, dedup by
    key, fsync). A seed can only serve a verdict for a byte-identical (goal, response) pair judged by the same judge
    under the same rubric — it never changes a verdict. ``write=False`` (dry run) computes what would be copied."""
    have: set[str] = set()
    if dst.exists():
        for r in read_jsonl(dst):
            if "key" in r:
                have.add(r["key"])
    n_have0 = len(have)
    report: list[dict] = []
    to_write: list[str] = []
    for src in sources:
        if not src.exists():
            raise Refuse(f"--seed-cache {src}: no such file")
        if judge not in src.name:
            raise Refuse(f"--seed-cache {src}: the file name does not carry the judge {judge!r} (ScoreCache names caches by judge; "
                         "an un-attributed file's judge is unknown)")
        n = copied = foreign = unstamped = dup = 0
        for line in src.read_text().splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
                key, val = r["key"], r["value"]
            except (json.JSONDecodeError, KeyError, TypeError):
                continue
            n += 1
            stamp = val.get("judge_model") if isinstance(val, dict) else None
            if stamp is None:
                unstamped += 1; continue
            if stamp != judge:
                foreign += 1; continue
            if key in have:
                dup += 1; continue
            have.add(key); copied += 1
            to_write.append(json.dumps({"key": key, "value": val}) + "\n")
        report.append({"path": str(src), "sha256": sha256_file(src), "rows": n, "rows_copied": copied,
                       "rows_skipped_foreign_judge": foreign, "rows_skipped_unstamped": unstamped, "rows_skipped_duplicate": dup})
    if write and to_write:
        dst.parent.mkdir(parents=True, exist_ok=True)
        with dst.open("a") as f:
            f.writelines(to_write)
            f.flush(); os.fsync(f.fileno())
    return {"cache": str(dst), "judge_model": judge, "rows_before": n_have0, "rows_after": len(have),
            "written": bool(write), "sources": report,
            "note": ("rows stamped with the pinned judge copied by key, duplicates and foreign-judge rows skipped; a seed serves a "
                     "verdict only for a byte-identical (goal, response) pair under the same judge + rubric (kind 'sr')")}


# --------------------------------------------------------------------------- goals

def load_goals(cfg: dict, goals_jsonl: str | None):
    """The harness's own confirmatory goal set (259: HarmBench standard test + 100 stratified JBB), or an explicit
    offline fixture (--goals-jsonl, rows {goal_id, dataset, revision, text, category}) whose count is the expectation."""
    import jailbreak_run as jr
    from sisteer.eval.jailbreak_data import Goal

    if goals_jsonl:
        rows = read_jsonl(Path(goals_jsonl))
        goals = [Goal(r["goal_id"], r["text"], r["category"], r["dataset"], r["revision"]) for r in rows]
        if not goals:
            raise Refuse(f"--goals-jsonl {goals_jsonl}: no goals")
        return goals, {"source": "fixture", "path": goals_jsonl, "sha256": sha256_file(Path(goals_jsonl)), "n": len(goals)}
    goals = jr._eval_goals(cfg, confirmatory=True)
    jbb = [g for g in goals if g.dataset == "jbb"]
    if len(goals) != EXPECTED_GOALS or len(jbb) != cfg["prereg"]["confirmatory_goals"] or len(jbb) != EXPECTED_JBB:
        raise Refuse(f"the harness resolved {len(goals)} goals ({len(jbb)} JBB), expected {EXPECTED_GOALS} ({EXPECTED_JBB} JBB)")
    return goals, {"source": "jailbreak_run._eval_goals(confirmatory=True)", "n": len(goals), "n_jbb": len(jbb)}


# --------------------------------------------------------------------------- main

def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--target", required=True, choices=sorted(KEYS))
    ap.add_argument("--config", required=True, help="configs/sdfbase_full22_<key>_sampled_t<T>[_cap8192].yaml")
    ap.add_argument("--private-root", required=True, help="the mirror of transcripts_results (holds <private_run>/<family>/<cell>/shard*)")
    ap.add_argument("--grades-root", required=True, help="output root: <grades-root>/<private_run>/ (dry run: …/dry_run/)")
    ap.add_argument("--attacks", default=",".join(STATIC), help="comma list of the static families (default all four)")
    ap.add_argument("--cells", default=None, help="comma list; default = the config's explicit_cells")
    ap.add_argument("--ladder-record", default=None,
                    help="an ESCALATION rung only: the static ladder record the rung's leg wrote (ladders/<TAG>_t<T>/static_ladder_t<T>.json). "
                         "A rung root holds only the (family, cell) pairs the leg escalated (> 0 at-cap rows at the source cap), so the "
                         "pairs graded are exactly those whose ladder entry attempted this rung; a listed pair with no store, or a store "
                         "the ladder never listed, is refused. Without it a rung is held to every (attack, cell) of the config.")
    ap.add_argument("--concurrency", type=int, default=150)
    ap.add_argument("--org", default="lowprio", choices=ORG_ITEMS, help="the ONE org this grader may bill (1Password item)")
    ap.add_argument("--seed-cache", action="append", default=[], help="an earlier haiku cache jsonl to seed from (repeatable)")
    ap.add_argument("--nshards", type=int, default=9, help="a base trial's shard count (shard00..shard<N-1>); a rung is shard00 only")
    ap.add_argument("--stall-at-floor-s", type=float, default=900.0,
                    help="at the gate's floor with no completed judgment for this long -> stop and report (exit 3)")
    ap.add_argument("--dry-run", action="store_true", help="no API: score 0.0 for every item; writes ONLY under …/<private_run>/dry_run/")
    ap.add_argument("--check-only", action="store_true", help="completeness + identity only; writes nothing; exit 0 when gradeable")
    ap.add_argument("--goals-jsonl", default=None, help="offline fixture override for the goal set (tests); recorded in provenance")
    args = ap.parse_args(argv)

    _pin_env()
    config_path = Path(args.config)
    if not config_path.exists():
        raise Refuse(f"config absent: {config_path}")
    cfg = yaml.safe_load(config_path.read_text())
    ident = run_identity(cfg, args.target, config_path)
    run = ident["run"]
    root = Path(args.private_root)
    cells = [c.strip() for c in (args.cells.split(",") if args.cells else cfg["explicit_cells"]) if c.strip()]
    unknown_cells = [c for c in cells if c not in cfg["explicit_cells"]]
    if unknown_cells:
        raise Refuse(f"--cells names cells outside the config's explicit_cells: {unknown_cells}")
    attacks = [a.strip() for a in args.attacks.split(",") if a.strip()]
    bad = [a for a in attacks if a not in STATIC]
    if bad:
        raise Refuse(f"--attacks {bad} are not static families {list(STATIC)} (BoN and the multi-turn legs are out of this run's scope)")
    if not attacks:
        raise Refuse("--attacks is empty")
    out_dir = Path(args.grades_root) / run
    if args.dry_run:
        out_dir = out_dir / "dry_run"        # a dry run's zeros must NEVER land where the real aggregates live
    t0 = time.time(); started = ts()
    print(f"[sampled-grade] {run}: {ident['kind']} trial {ident['trial']} cap {ident['cap']} arm {ident['arm']} "
          f"seed_base {ident['seed_base']}; decode " + ", ".join(f"{k}={v}" for k, v in ident["decode"].items()
                                                                if k not in ("do_sample", "cap", "arm", "seed_base", "trial")), flush=True)

    import jailbreak_run as jr
    from grade_gated_served import load_gated_dir

    goals, goals_src = load_goals(cfg, args.goals_jsonl)
    by_id = {g.goal_id: g for g in goals}
    expected_goal_ids = [g.goal_id for g in goals]
    group_of, held_in, held_out, ch = jr._category_groups(cfg, goals)
    thresholds = cfg["prereg"]["threshold_grid"]
    canonical = thresholds[len(thresholds) // 2]
    if canonical != 0.5:
        raise Refuse(f"the config's canonical threshold is {canonical}, not 0.5")

    # --- completeness + row invariants, BEFORE any spend ------------------------------------------------------------------
    invariants = [
        f"every expected goal ({len(expected_goal_ids)}) has a terminal progress record per (family, cell); no errored outcome",
        "response rows per goal == the goal's recorded n_target_forwards (no short shard)",
        ("shard layout: shard00 only" if ident["kind"] == "escalation_rung" else f"shard layout: shard00..shard{args.nshards - 1:02d}"),
        "every row carries decode / sampling_seed / finish_reason / at_cap (a greedy-shaped row is refused)",
        "at_cap == (finish_reason == 'length') on every row",
        "one decode block across the run, equal to the block the config derives (a mixed store is refused)",
    ]
    if ident["kind"] == "escalation_rung":
        invariants += ["every rung row carries regenerated / escalated_from_cap(4096) / source_cap",
                       "escalation.json counts == the store's (n_rows, n_regenerated, n_at_cap_after, cap_from, cap_to)"]
    pairs = [(fam, c) for c in cells for fam in attacks]
    skipped_pairs: list = []
    ladder_pairs_note = None
    if args.ladder_record:
        if ident["kind"] != "escalation_rung":
            raise Refuse("--ladder-record applies to an escalation rung only (a base trial holds every (attack, cell) pair)")
        lp = Path(args.ladder_record)
        if not lp.is_file():
            raise Refuse(f"--ladder-record absent: {lp}")
        lad = json.loads(lp.read_text())
        if lad.get("kind") != "static" or int(lad.get("trial", -1)) != ident["trial"]:
            raise Refuse(f"--ladder-record {lp} is not this trial's static ladder (kind={lad.get('kind')!r} trial={lad.get('trial')!r})")
        want_tag = f"{ident['key']}"
        if lad.get("tag") != want_tag:
            raise Refuse(f"--ladder-record {lp} is for tag {lad.get('tag')!r}, this run is {want_tag!r}")
        escalated: set = set()
        for lcell, fams in (lad.get("cells") or {}).items():
            for lfam, blk in (fams or {}).items():
                attempted = [int(r) for r in (blk.get("rungs_attempted") or [])] or [int(r) for r in (blk.get("n_at_cap") or {})]
                if ident["cap"] in attempted:
                    escalated.add((lfam, lcell))
        listed_missing = [f"{fam}/{c}" for (fam, c) in sorted(escalated) if c in cells and fam in attacks
                          and not jr._gated_defense_root(cfg, root, fam, c).is_dir()]
        if listed_missing:
            raise Refuse(f"[{run}] the static ladder says these pairs escalated to cap {ident['cap']} but the rung root has no store for them: "
                         + ", ".join(listed_missing))
        unlisted_present = [f"{fam}/{c}" for (fam, c) in pairs if (fam, c) not in escalated
                            and jr._gated_defense_root(cfg, root, fam, c).is_dir()]
        if unlisted_present:
            raise Refuse(f"[{run}] the rung root holds stores the static ladder never escalated (settled at the source cap): "
                         + ", ".join(unlisted_present))
        skipped_pairs = [f"{fam}/{c}" for (fam, c) in pairs if (fam, c) not in escalated]
        pairs = [(fam, c) for (fam, c) in pairs if (fam, c) in escalated]
        if not pairs:
            raise Refuse(f"[{run}] the static ladder escalated none of the requested (attack, cell) pairs to cap {ident['cap']}; nothing to grade")
        ladder_pairs_note = {"ladder_record": str(lp), "written_at": lad.get("written_at"), "rungs": lad.get("rungs"),
                             "pairs_graded": [f"{fam}/{c}" for (fam, c) in pairs], "pairs_settled_below_rung": skipped_pairs}
        print(f"[sampled-grade] {run}: ladder record {lp.name} -> grading {len(pairs)} escalated pair(s), "
              f"{len(skipped_pairs)} settled at cap {ident['from_cap']} (not in this rung)", flush=True)
    cells_of = {fam: [c for (f, c) in pairs if f == fam] for fam in attacks}
    attacks = [fam for fam in attacks if cells_of[fam]]
    completion: dict = {}
    for c in cells:
        if not any(c2 == c for (_, c2) in pairs):
            continue
        completion[c] = {}
        for fam in attacks:
            if (fam, c) not in pairs:
                continue
            d = jr._gated_defense_root(cfg, root, fam, c)
            completion[c][fam] = check_store(d, expected_goal_ids=expected_goal_ids, ident=ident, nshards=args.nshards,
                                             label=f"{run}/{fam}/{c}")
            rec = completion[c][fam]
            print(f"[check] {fam:11s} {c:52s} goals={rec['n_goals_with_progress']} rows={rec['n_responses']} "
                  f"at_cap={rec['n_at_cap']} shards={rec['n_shards']}"
                  + (f" regenerated={rec['n_regenerated']} copied={rec['n_copied_forward']}" if ident["kind"] == "escalation_rung" else ""),
                  flush=True)
    est = sum(completion[c][f]["n_responses"] for (f, c) in pairs)
    print(f"[sampled-grade] {run}: completeness OK for {len(pairs)} (family, cell) pair(s) over {attacks}; {est} responses to judge (before cache hits)",
          flush=True)
    if args.check_only:
        print(f"[sampled-grade] {run}: --check-only, nothing written; gradeable", flush=True)
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "completion.json").write_text(json.dumps(
        {"target": args.target, "private_run": run, "kind": ident["kind"], "trial": ident["trial"], "cap": ident["cap"],
         "decode": ident["decode"], "escalated_from": ident["escalated_from"], "expected": {"goals": len(expected_goal_ids)},
         "goals_source": goals_src, "attacks": attacks, "cells": completion, "invariants_enforced": invariants,
         "ladder_pairs": ladder_pairs_note, "written_at": ts()}, indent=1, sort_keys=True) + "\n")

    # --- the cache: seeded from earlier haiku caches, then the judge ------------------------------------------------------
    cache_path = out_dir / CACHE_NAME
    seeding = seed_cache(cache_path, [Path(p) for p in args.seed_cache], write=not args.dry_run) if args.seed_cache else None
    if seeding is not None:
        (out_dir / "cache_seeding.json").write_text(json.dumps(seeding, indent=1, sort_keys=True) + "\n")
        print(f"[sampled-grade] cache seeding: {seeding['rows_before']} -> {seeding['rows_after']} rows from {len(seeding['sources'])} source(s)"
              + ("" if seeding["written"] else " (dry run: NOT written)"), flush=True)

    judge_model = cfg["grader"]["model"]
    esc = None
    grader = None
    unscored_retry_events: list = []
    if args.dry_run:
        grade_fn = lambda items: {it: 0.0 for it in items}      # noqa: E731
        prov = {"judge_model": "DRY-RUN-no-api", "rubric": None, "org": None, "cache_path": None}
    else:
        import grade_bon_live
        grade_bon_live.load_key = _load_key_lowprio          # make_client looks load_key up on the module at call time
        from sisteer.eval.attacks.grading_live import LiveStrongREJECTGrader
        from sisteer.throttle import FloorWatch

        grader = LiveStrongREJECTGrader(model=judge_model, concurrency=args.concurrency, cache_path=cache_path,
                                        rubric=PINNED_RUBRIC, allow_unscored=0)
        if grader.rubric != PINNED_RUBRIC or grader._sr_kind != PINNED_CACHE_KIND:
            raise Refuse(f"grader rubric {grader.rubric} / kind {grader._sr_kind} is not {PINNED_RUBRIC} / {PINNED_CACHE_KIND}")
        if grader._cache.path != cache_path:
            raise Refuse(f"the harness derived cache path {grader._cache.path} != {cache_path}")
        _inner_grade = grader.grade

        def grade_fn(items, _max_attempts=6):
            # JB1 resilience (the GLM grader's): the harness raises when ANY item is still UNSCORED after its per-item
            # retries; successful verdicts are already in the append-only cache, so re-issuing the SAME call re-requests
            # only the uncached items. Bounded, then the loud refusal stands.
            for attempt in range(1, _max_attempts + 1):
                try:
                    return _inner_grade(items)
                except RuntimeError as e:
                    if "UNSCORED" not in str(e):
                        raise
                    unscored_retry_events.append({"t": ts(), "attempt": attempt, "n_items": len(items), "error": str(e)[:240]})
                    print(f"[sampled-grade] UNSCORED after attempt {attempt}/{_max_attempts}: {str(e)[:160]} — re-issuing the uncached items only",
                          flush=True)
                    if attempt == _max_attempts:
                        raise
                    time.sleep(10 * attempt)
        n_seed = len(grader._cache.cache)
        prov = {"judge_model": judge_model, "rubric": f"strongreject_{grader.rubric}", "rubric_short": grader.rubric,
                "cache_kind": grader._sr_kind, "cache_path": str(grader._cache.path), "cache_rows_seeded": n_seed,
                "org_initial": args.org, "org": f"{args.org} (key via 1Password at run time); NO org change is possible",
                "base_url": PINNED_BASE_URL, "concurrency_initial": args.concurrency, "sdk_max_retries": 2,
                "escalation_rule": (f"NONE (DESIGN.md): a 429 halves the in-flight ceiling on the same org down to the gate's floor; at the "
                                    f"floor with no completed judgment for {args.stall_at_floor_s:.0f}s the run stops and reports (exit 3); "
                                    "never a pool switch")}
        esc = FloorWatch(grader._gate, lambda: len(grader._cache.cache), initial=args.concurrency, org=lambda: args.org,
                         stall_s=args.stall_at_floor_s)
        esc.start()
        print(f"[sampled-grade] judge={judge_model} rubric={grader.rubric} kind={grader._sr_kind} org={args.org} "
              f"concurrency={args.concurrency} base_url={PINNED_BASE_URL} cache={grader._cache.path.name} (seeded rows: {n_seed})", flush=True)

    import two_si_run
    leaked: list = []
    per_attack_wall: dict = {}
    per_attack_429: dict = {}
    controls = list(cfg["controls"])
    for attack in attacks:
        ta = time.time()
        n429_before = grader._gate.n_rate_limited if grader is not None else 0
        per_cell: dict = {}
        for cell in cells_of[attack]:
            defense_root = jr._gated_defense_root(cfg, root, attack, cell)
            responses, progress = load_gated_dir(defense_root)
            m = jr._gated_defense_metrics(responses, progress, by_id, grade_fn, thresholds=thresholds, canonical=canonical,
                                          leaked_needles=leaked, expected_goal_ids=expected_goal_ids, group_of=group_of,
                                          allow_errored_goals=False, label=f"{attack}] {cell}")
            per_cell[cell] = m
            print(f"[{attack}] {cell:52s} n_goals={m['n_goals']} asr@0.5={m['asr']['0.5']:.4f} n_jailbroken={m['n_jailbroken']} "
                  f"mean_best={m['mean_best_score']:.4f} at_cap_rows={completion[cell][attack]['n_at_cap']}", flush=True)
        cells_block = {}
        for cell in cells_of[attack]:
            pc = parse_cell(cell)
            cells_block[cell] = {
                "class": pc["class"], "si_id": pc["si_id"], "arm": pc["arm"], "ni_reference_ids": [],
                # the over-refusal controls of this run are judged by served_capability_judge.py into controls_sampled/,
                # never folded here (the wave's aggregate shape keeps the slots)
                "controls": {c: two_si_run._CONTROL_PENDING for c in controls},
                "training_pool": {"regime": "explicit_cells", "note": "per-cell provenance in the object's train_budget_meta.json"},
                "served_alpha": float(cfg.get("scale", 1.0)), "lens_reduction_mode": WAVE_LENS_MODE,
                "completion": {"gated": completion[cell][attack]},
                "decode_cap_tokens": ident["cap"], "decode": ident["decode"], "trial": ident["trial"],
                "attack_metrics": per_cell[cell],
            }
        aggregate = {
            "attack": attack, "fit_scale": float(cfg.get("scale", 1.0)), "served_alpha": float(cfg.get("scale", 1.0)),
            "pool_regime": {"regime": "explicit_cells", "pool_sizes": {}, "headline_confounded_by_pool_composition": None,
                            "note": "cells enumerated explicitly (explicit_cells config): training provenance is PER CELL by design"},
            "cells": cells_block, "refused": {}, "controls": controls, "primary_statistic": jr._GATED_PRIMARY_STATISTIC,
            "category_holdout": {"seed": ch["seed"], "frac": ch["frac"], "held_in": held_in, "held_out": held_out},
            "wave": "full22", "target": args.target, "private_run": run, "lens_reduction_mode": WAVE_LENS_MODE,
            "attack_tier": "static",
            "sampled_decode": {"arm": ident["arm"], "trial": ident["trial"], "cap": ident["cap"], "kind": ident["kind"],
                               "decode": ident["decode"], "escalated_from": ident["escalated_from"],
                               "config_sha256": sha256_file(config_path), **ident["sampled_decode_source"]},
            "grader_provenance": {**prov, "org_events_so_far": list(esc.events) if esc else [], "dry_run": bool(args.dry_run)},
            "n_expected_goals": len(expected_goal_ids), "written_at": ts(),
        }
        outp = out_dir / f"two_si_{attack}_aggregate.json"
        tmp = outp.with_name(outp.name + ".tmp")
        tmp.write_text(json.dumps(aggregate, indent=1, sort_keys=True) + "\n"); os.replace(tmp, outp)
        per_attack_wall[attack] = round(time.time() - ta, 1)
        rl = grader._gate.n_rate_limited if grader is not None else 0
        per_attack_429[attack] = rl - n429_before
        print(f"[{attack}] wrote {outp.name} ({len(cells_of[attack])} cells) in {per_attack_wall[attack]}s; 429s so far: {rl}; org {args.org}", flush=True)

    if esc is not None:
        esc.stop()
    # --- privacy gate: no goal text / response may appear in any grades file ----------------------------------------------
    from sisteer.eval.jailbreak_artifacts import grep_gate
    hits = grep_gate([out_dir], leaked, suffixes=(".json", ".md", ".csv", ".txt", ".html", ".svg"))
    hits = [h for h in hits if Path(h).name != CACHE_NAME]      # the harness's own private cache holds hashes + labels only
    if hits:
        for p in hits:
            Path(p).unlink()
        raise Refuse(f"[{run}] GREP GATE FAILED — prompt/response text in {hits}; files deleted")
    wall = round(time.time() - t0, 1)
    gp = {"target": args.target, "private_run": run, "kind": ident["kind"], "trial": ident["trial"], "cap": ident["cap"],
          "decode": ident["decode"], "escalated_from": ident["escalated_from"], "sampled_decode_source": ident["sampled_decode_source"],
          "config": {"path": str(config_path), "sha256": sha256_file(config_path)},
          "worktree": {"path": str(WT), "head": subprocess.run(["git", "-C", str(WT), "rev-parse", "HEAD"], capture_output=True,
                                                               text=True).stdout.strip()},
          "driver": {"path": str(HERE), "sha256": sha256_file(HERE)},
          "started_at": started, "finished_at": ts(), "wall_s": wall, "per_attack_wall_s": per_attack_wall,
          "per_attack_429": per_attack_429, "cells": cells, "attacks": attacks, "dry_run": bool(args.dry_run),
          "ladder_pairs": ladder_pairs_note,
          "check_only": False, "goals_source": goals_src, "invariants_enforced": invariants, "refusals": [],
          "judge": prov, "cache_seeding": seeding, "lens_reduction_mode": WAVE_LENS_MODE,
          "n_429": (grader._gate.n_rate_limited if grader is not None else 0),
          "final_in_flight_ceiling": (grader._gate.live if grader is not None else None),
          "n_judgments_estimated": est, "grep_gate": "clean",
          "org_events": (esc.events if esc is not None else []), "org_initial": args.org, "org_final": args.org,
          "org_switch": "impossible: --org accepts only lowprio; the key is op-secret lowprio at call time",
          "escalated_to_highprio": False, "unscored_retry_events": unscored_retry_events,
          "greedy_stores": "never read, never re-graded (the analysis joins them on the grading host)"}
    if grader is not None:
        cp = grader._cache.path
        gp["judge"]["pass_id"] = sha256_file(cp) if cp.exists() else None
        gp["judge"]["pass_id_definition"] = "sha256 of the judge cache file at close (append-only ScoreCache, rows stamped judge_model; seeded per cache_seeding.json)"
        gp["judge"]["cache_rows"] = sum(1 for line in cp.read_text().splitlines() if line.strip()) if cp.exists() else 0
    (out_dir / "grading_provenance.json").write_text(json.dumps(gp, indent=1, sort_keys=True) + "\n")
    print(f"[sampled-grade] {run} DONE in {wall}s; 429s={gp['n_429']}; org_final={args.org}; escalated=False; grep gate clean; "
          f"wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
