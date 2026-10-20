"""Post-hoc constitution-SECTION tagging of the production corpora (2026-09-14).

The production analysis measured section coverage only through construction
lineage: advice records carry the section their principle was seeded from
(17/39 sections, zero helpfulness-cluster records), and stories were
conditioned on the whole constitution but never section-targeted, so nothing
said which sections the corpora actually EXERCISE. This module measures it by
content: one live Opus call per document, against a compact index of the 39
sections, returning the sections the document substantively exercises.

Three pieces, each frozen by hash so the measurement is reproducible:

1. ``section-summaries.json`` — the compact tagging reference: per section its
   id, title, level, parent and a 1–3 sentence summary written once by Opus
   from the section text (``summarize``). Frozen beside the constitution
   snapshot with its own sha; ``load_index`` verifies both the constitution
   sha it was written from and the summaries' own sha, and refuses on drift.
2. The tagging wave (``run_tagging``) over the repo's LIVE lowprio engine
   (``live.LiveEngine``: ~200 concurrent, halving on 429/529, never
   highprio): the shared system block (instructions + index) carries a cache
   breakpoint; the document rides in the user turn; the reply is a structured
   JSON object. Ids are validated against the index; malformed replies get a
   bounded lenient repair and then bounded re-asks under fresh custom ids;
   every outcome is cached per record by the engine, so a re-run resumes.
3. The per-corpus output ``section-tags-<release>.jsonl`` (id, primary,
   secondary, none, confidence, status; advice rows also carry the lineage
   section) + a manifest (index sha, prompt hashes, model, counts, timings).
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

from .batch import WaveRequest, outcome_status
from .constitution import Section, load as load_constitution
from .io import atomic_write, iter_jsonl, read_json, sha256_file, write_jsonl_atomic
from .parsing import extract_json_object, extract_tag
from .stages import ParseError, structured_format
from .templates import TemplateStore, repo_root

SUMMARIES_FILE = "section-summaries.json"
SUMMARIZE_TEMPLATE = "section_tags/summarize.md"
TAG_SYSTEM_TEMPLATE = "section_tags/tag_system.md"
TAG_USER_TEMPLATE = "section_tags/tag_user.md"

MAX_PRIMARY = 3
MAX_SECONDARY = 3
DEFAULT_MODEL = "claude-opus-5"
# The reply JSON is ~100 tokens, but this model generation thinks ADAPTIVELY by
# default (~650 thinking tokens observed per tag call, pilot 2026-09-14) and
# the thinking counts against max_tokens — 300 truncated 3/5 pilot replies to
# an empty text. Thinking is kept (it is what makes the tag a judgment about
# what the content exercises rather than keyword overlap); the cap is sized
# for it.
TAG_MAX_TOKENS = 3000
SUMMARY_MAX_TOKENS = 600

# The structured-output schema of one tag reply. Ids are further validated
# against the index in ``validate_tags`` (the enum guarantees membership when
# the API honors it, the validator guarantees it regardless).
def tag_schema(valid_ids: list[str]) -> dict:
    return {
        "type": "object",
        "properties": {
            "primary": {"type": "array", "items": {"type": "string", "enum": list(valid_ids)}},
            "secondary": {"type": "array", "items": {"type": "string", "enum": list(valid_ids)}},
            "none": {"type": "boolean"},
            "confidence": {"type": "number"},
        },
        "required": ["primary", "secondary", "none", "confidence"],
        "additionalProperties": False,
    }


class SectionIndexError(RuntimeError):
    """The summaries file is missing, was written from another constitution, or
    its content does not match its own recorded sha."""


class TagParseError(ParseError):
    """A tag reply that neither parses nor repairs within the lenient rules."""


# ------------------------------------------------------------------ index --


@dataclass
class IndexEntry:
    id: str
    title: str
    level: int
    parent: str | None
    chapter: str  # the level-2 ancestor (itself for a chapter)
    summary: str = ""


def section_tree(sections: list[Section]) -> list[IndexEntry]:
    """Parent/chapter for every section from the flat heading order: a
    section's parent is the nearest preceding section of a shallower level."""
    out: list[IndexEntry] = []
    stack: list[tuple[int, str]] = []  # (level, id) of open ancestors
    for s in sections:
        while stack and stack[-1][0] >= s.level:
            stack.pop()
        parent = stack[-1][1] if stack else None
        chapter = stack[0][1] if stack else s.id
        out.append(IndexEntry(id=s.id, title=s.title, level=s.level, parent=parent, chapter=chapter))
        stack.append((s.level, s.id))
    return out


def summaries_sha(entries: list[dict]) -> str:
    """Content sha of the frozen index: ids, titles, levels, parents, summaries
    — canonical JSON, so a re-serialization with different whitespace or key
    order does not read as drift."""
    canon = [
        {k: e[k] for k in ("id", "title", "level", "parent", "chapter", "summary")}
        for e in entries
    ]
    return hashlib.sha256(json.dumps(canon, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def load_index(constitution_dir: Path) -> tuple[list[IndexEntry], dict]:
    """The frozen tagging index + its manifest, verified against the frozen
    constitution snapshot and its own sha. Raises SectionIndexError on any
    mismatch — a tagging run must never proceed on a drifted reference."""
    constitution_dir = Path(constitution_dir)
    _md, sections, snapshot = load_constitution(constitution_dir)
    path = constitution_dir / SUMMARIES_FILE
    if not path.exists():
        raise SectionIndexError(f"{path} missing — run `corpusgen section-tags summarize` first")
    data = read_json(path)
    if data.get("constitution_sha256") != snapshot["sha256"]:
        raise SectionIndexError(
            f"{path.name} was written from constitution {str(data.get('constitution_sha256'))[:12]}…, "
            f"the frozen snapshot is {snapshot['sha256'][:12]}…"
        )
    entries = data.get("sections") or []
    actual = summaries_sha(entries)
    if actual != data.get("sha256"):
        raise SectionIndexError(
            f"{path.name} content sha {actual[:12]}… != recorded {str(data.get('sha256'))[:12]}… — "
            f"the frozen summaries were edited after freezing"
        )
    tree = {e.id: e for e in section_tree(sections)}
    if [e["id"] for e in entries] != [s.id for s in sections]:
        raise SectionIndexError(
            f"{path.name} lists {len(entries)} sections; sections.json has {len(sections)} "
            f"(or the order differs) — regenerate the summaries"
        )
    out: list[IndexEntry] = []
    for e in entries:
        t = tree[e["id"]]
        if not str(e.get("summary", "")).strip():
            raise SectionIndexError(f"{path.name}: section {e['id']!r} has an empty summary")
        out.append(
            IndexEntry(
                id=t.id, title=t.title, level=t.level, parent=t.parent, chapter=t.chapter,
                summary=e["summary"].strip(),
            )
        )
    return out, data


def render_index(entries: list[IndexEntry]) -> str:
    """The compact index text the tagger sees: one line per section,
    indented by containment."""
    top = min(e.level for e in entries) if entries else 2
    lines = []
    for e in entries:
        indent = "  " * (e.level - top)
        kind = " (chapter)" if e.level == top else ""
        lines.append(f"{indent}- {e.id}{kind} — {e.title}: {e.summary}")
    return "\n".join(lines)


# ------------------------------------------------------------- summarize --


def summarize_requests(
    constitution_md: str,
    sections: list[Section],
    templates: TemplateStore,
    model: str = DEFAULT_MODEL,
) -> list[WaveRequest]:
    tree = section_tree(sections)
    children: dict[str, list[str]] = {}
    for e in tree:
        if e.parent:
            children.setdefault(e.parent, []).append(e.title)
    out = []
    for s in sections:
        rendered = templates.render(
            SUMMARIZE_TEMPLATE,
            section_id=s.id,
            title=s.title,
            level=s.level,
            children="; ".join(children.get(s.id, [])),
            section_text=s.text(constitution_md),
        )
        out.append(
            WaveRequest(
                custom_id=f"sum-{s.id}"[:64],
                params={
                    "model": model,
                    "max_tokens": SUMMARY_MAX_TOKENS,
                    "messages": [{"role": "user", "content": rendered}],
                },
                lineage={"section_id": s.id},
            )
        )
    return out


def freeze_summaries(
    constitution_dir: Path,
    outcomes: dict[str, dict],
    sections: list[Section],
    snapshot: dict,
    *,
    model: str,
    template_hash: str,
    force: bool = False,
) -> dict:
    """Parse the summary replies and write the frozen index file. Refuses to
    overwrite an existing file without ``force`` (frozen by design)."""
    path = Path(constitution_dir) / SUMMARIES_FILE
    if path.exists() and not force:
        raise SectionIndexError(f"{path} already exists; the index is frozen (--force to regenerate)")
    tree = section_tree(sections)
    entries = []
    for e in tree:
        o = outcomes.get(f"sum-{e.id}"[:64])
        if o is None or outcome_status(o) != "succeeded":
            raise SectionIndexError(
                f"no usable summary for section {e.id!r}: "
                f"{outcome_status(o) if o else 'missing'}"
            )
        summary = re.sub(r"\s+", " ", extract_tag(o["text"], "summary")).strip()
        entries.append(
            {"id": e.id, "title": e.title, "level": e.level, "parent": e.parent,
             "chapter": e.chapter, "summary": summary}
        )
    data = {
        "constitution_sha256": snapshot["sha256"],
        "n_sections": len(entries),
        "model": model,
        "template_hash": template_hash,
        "generated_at": _utcnow(),
        "sha256": summaries_sha(entries),
        "sections": entries,
    }
    atomic_write(path, json.dumps(data, indent=1, ensure_ascii=False))
    return data


# ------------------------------------------------------------- documents --


def record_id(record: dict) -> str:
    return record["lineage"]["custom_id"]


def document_fields(record: dict, corpus: str) -> dict:
    """What the tagger sees of a record: the story text, or the advice
    transcript's user + assistant turns (the order named user+assistant; the
    operator system prompt is deliberately NOT sent)."""
    if corpus == "stories":
        return {"kind": "story", "text": record["text"]}
    if corpus == "advice":
        return {"kind": "advice", "user": record["user"], "assistant": record["assistant"]}
    raise ValueError(f"unknown corpus {corpus!r}")


def tag_system_block(templates: TemplateStore, entries: list[IndexEntry]) -> list[dict]:
    """The shared system prefix — instructions + index — with a cache
    breakpoint: identical for every request of a run."""
    text = templates.render(TAG_SYSTEM_TEMPLATE, index_text=render_index(entries))
    return [{"type": "text", "text": text, "cache_control": {"type": "ephemeral"}}]


def tag_request(
    record: dict,
    corpus: str,
    templates: TemplateStore,
    system: list[dict],
    schema: dict,
    model: str = DEFAULT_MODEL,
    suffix: str = "",
) -> WaveRequest:
    rid = record_id(record)
    fields = document_fields(record, corpus)
    user = templates.render(TAG_USER_TEMPLATE, **fields)
    lineage = {"record_id": rid, "corpus": corpus}
    if corpus == "advice":
        lineage["lineage_section_id"] = record.get("section_id")
    return WaveRequest(
        custom_id=f"tag-{rid}{suffix}",
        params={
            "model": model,
            "max_tokens": TAG_MAX_TOKENS,
            "system": system,
            "messages": [{"role": "user", "content": user}],
            "output_config": structured_format(schema),
        },
        lineage=lineage,
    )


# ------------------------------------------------------- parse + repair --


def _norm_id(s: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "-", str(s).strip().lower()).strip("-")


def _as_list(v: Any) -> list:
    if v is None:
        return []
    if isinstance(v, str):
        return [x for x in re.split(r"[,\n;]+", v) if x.strip()]
    if isinstance(v, (list, tuple)):
        return list(v)
    raise TagParseError(f"tag list is a {type(v).__name__}, not a list")


def validate_tags(obj: dict, valid_ids: Iterable[str]) -> tuple[dict, list[str]]:
    """Normalize one parsed tag object against the index. Returns
    (clean tags, repairs applied). Lenient where the intent is unambiguous
    (an id differing only in case/punctuation, a list given as a string, a
    list over the cap, a none flag inconsistent with the lists, confidence
    out of range); raises TagParseError where it is not (an id that matches
    nothing, a non-object). Every lenient repair is named so the manifest can
    count them."""
    if not isinstance(obj, dict):
        raise TagParseError("tag reply is not a JSON object")
    valid = list(valid_ids)
    by_norm = {_norm_id(v): v for v in valid}
    repairs: list[str] = []

    def clean(key: str, cap: int, exclude: set[str]) -> list[str]:
        out: list[str] = []
        for raw in _as_list(obj.get(key)):
            if not isinstance(raw, str):
                repairs.append(f"{key}:non-string-dropped")
                continue
            sid = raw.strip()
            if sid not in by_norm.values():
                fixed = by_norm.get(_norm_id(sid))
                if fixed is None:
                    raise TagParseError(f"{key}: unknown section id {sid!r}")
                repairs.append(f"{key}:id-normalized")
                sid = fixed
            if sid in out or sid in exclude:
                repairs.append(f"{key}:duplicate-dropped")
                continue
            out.append(sid)
        if len(out) > cap:
            repairs.append(f"{key}:trimmed-to-{cap}")
            out = out[:cap]
        return out

    if isinstance(obj.get("primary"), str) or isinstance(obj.get("secondary"), str):
        repairs.append("list-from-string")
    primary = clean("primary", MAX_PRIMARY, set())
    secondary = clean("secondary", MAX_SECONDARY, set(primary))

    none = obj.get("none")
    if not isinstance(none, bool):
        repairs.append("none:coerced")
        none = not primary
    if none and (primary or secondary):
        repairs.append("none:cleared-tags-present")
        none = False
    if not none and not primary:
        if secondary:
            # tags in secondary only: the model called nothing substantive
            repairs.append("none:set-primary-empty")
        else:
            repairs.append("none:set-empty")
        none = True

    conf = obj.get("confidence")
    if isinstance(conf, bool) or not isinstance(conf, (int, float)):
        try:
            conf = float(conf)
            repairs.append("confidence:coerced")
        except (TypeError, ValueError):
            raise TagParseError(f"confidence is not a number: {conf!r}")
    conf = float(conf)
    if not 0.0 <= conf <= 1.0:
        repairs.append("confidence:clamped")
        conf = min(1.0, max(0.0, conf))
    return (
        {"primary": primary, "secondary": secondary, "none": none, "confidence": round(conf, 3)},
        repairs,
    )


def parse_tag_reply(text: str, valid_ids: Iterable[str]) -> tuple[dict, list[str]]:
    """Reply text -> validated tags (see validate_tags). ParseError from the
    JSON locator is re-raised as TagParseError so callers have one class."""
    try:
        obj = extract_json_object(text)
    except ParseError as e:
        raise TagParseError(str(e)) from e
    return validate_tags(obj, valid_ids)


# ------------------------------------------------------------------ run --


@dataclass
class TagRun:
    corpus: str
    release: str
    source: Path
    out_dir: Path
    wave_root: Path
    constitution_dir: Path = field(default_factory=lambda: repo_root() / "constitution")
    prompts_dir: Path = field(default_factory=lambda: repo_root() / "prompts")
    model: str = DEFAULT_MODEL
    repair_rounds: int = 2
    limit: int | None = None  # first N records (smoke / pilot)

    @property
    def tags_path(self) -> Path:
        return self.out_dir / f"section-tags-{self.release}.jsonl"

    @property
    def manifest_path(self) -> Path:
        return self.out_dir / f"section-tags-{self.release}.manifest.json"


def load_records(path: Path, limit: int | None = None) -> list[dict]:
    out = []
    for r in iter_jsonl(path):
        out.append(r)
        if limit is not None and len(out) >= limit:
            break
    return out


def run_tagging(run: TagRun, engine, on_event: Callable[[str], None] = print) -> dict:
    """Tag every record of one corpus: main wave, then bounded repair waves
    for replies that did not parse, then the output file + manifest. The
    engine's per-request cache is the resume unit; a stamped wave is skipped
    entirely, so re-running after an interruption costs only the missing
    requests."""
    t0 = time.monotonic()
    started = _utcnow()
    entries, index_meta = load_index(run.constitution_dir)
    valid_ids = [e.id for e in entries]
    templates = TemplateStore(run.prompts_dir)
    system = tag_system_block(templates, entries)
    schema = tag_schema(valid_ids)
    records = load_records(run.source, run.limit)
    by_rid = {record_id(r): r for r in records}
    if len(by_rid) != len(records):
        raise ValueError(f"{run.source}: duplicate record ids")
    on_event(f"[section-tags {run.corpus}] {len(records)} records, index sha {index_meta['sha256'][:12]}")

    template_hashes = templates.hashes([TAG_SYSTEM_TEMPLATE, TAG_USER_TEMPLATE])
    upstream = f"index:{index_meta['sha256']}|source:{sha256_file(run.source)}"
    wave_root = Path(run.wave_root) / run.corpus
    wave_root.mkdir(parents=True, exist_ok=True)

    results: dict[str, dict] = {}  # record id -> output row
    repair_counts: dict[str, int] = {}
    n_requests = 0

    def absorb(outcomes: dict[str, dict], round_no: int) -> list[str]:
        """File every outcome as a row; return the record ids to re-ask."""
        retry: list[str] = []
        for cid, o in outcomes.items():
            rid = o_lineage(cid, round_no)
            status = outcome_status(o)
            if status == "succeeded":
                try:
                    tags, repairs = parse_tag_reply(o["text"], valid_ids)
                except TagParseError as e:
                    if round_no < run.repair_rounds:
                        retry.append(rid)
                    results[rid] = _row(rid, run.corpus, by_rid[rid], None, "parse_failed",
                                        round_no, error=str(e)[:200])
                    continue
                for r in repairs:
                    repair_counts[r] = repair_counts.get(r, 0) + 1
                results[rid] = _row(rid, run.corpus, by_rid[rid], tags, "ok", round_no,
                                    repairs=repairs or None)
            else:
                # refusal / max_tokens / content_filtered / request_rejected /
                # errored-after-retries: an outcome, recorded, not re-asked
                # (a re-ask of a refusal is the same refusal)
                results[rid] = _row(rid, run.corpus, by_rid[rid], None, status, round_no,
                                    error=str(o.get("error_message", ""))[:200] or None)
        return retry

    def o_lineage(cid: str, round_no: int) -> str:
        suffix = f"-r{round_no}" if round_no else ""
        assert cid.startswith("tag-") and cid.endswith(suffix)
        return cid[len("tag-"): len(cid) - len(suffix)] if suffix else cid[len("tag-"):]

    reqs = [tag_request(r, run.corpus, templates, system, schema, run.model) for r in records]
    n_requests += len(reqs)
    outcomes = engine.run_wave(
        wave_root / "tag", reqs, wave_name=f"tag-{run.corpus}",
        template_hashes=template_hashes, upstream_hash=upstream,
        description=f"section tagging, {run.corpus} {run.release}",
    )
    retry = absorb(outcomes, 0)
    for round_no in range(1, run.repair_rounds + 1):
        if not retry:
            break
        on_event(f"[section-tags {run.corpus}] repair round {round_no}: {len(retry)} unparsed replies re-asked")
        reqs = [
            tag_request(by_rid[rid], run.corpus, templates, system, schema, run.model,
                        suffix=f"-r{round_no}")
            for rid in sorted(retry)
        ]
        n_requests += len(reqs)
        outcomes = engine.run_wave(
            wave_root / f"tag-repair{round_no}", reqs, wave_name=f"tag-{run.corpus}-repair{round_no}",
            template_hashes=template_hashes, upstream_hash=upstream,
        )
        retry = absorb(outcomes, round_no)

    rows = [results[record_id(r)] for r in records]
    run.out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl_atomic(run.tags_path, (json.dumps(r, ensure_ascii=False) for r in rows))

    status_counts: dict[str, int] = {}
    for r in rows:
        status_counts[r["status"]] = status_counts.get(r["status"], 0) + 1
    ok = [r for r in rows if r["status"] == "ok"]
    manifest = {
        "corpus": run.corpus,
        "release": run.release,
        "source": {"path": str(run.source), "sha256": sha256_file(run.source), "records": len(records)},
        "limit": run.limit,
        "constitution_sha256": index_meta["constitution_sha256"],
        "index_sha256": index_meta["sha256"],
        "index_model": index_meta.get("model"),
        "template_hashes": template_hashes,
        "system_block_sha256": hashlib.sha256(system[0]["text"].encode()).hexdigest(),
        "model": run.model,
        # no sampling controls: anthropic 1.0's messages API (this model
        # generation) takes no temperature/top_p/top_k
        "params": {"max_tokens": TAG_MAX_TOKENS, "structured_output": True, "sampling": "api-default",
                   "thinking": "api-default (adaptive; ~650 tokens/call observed in the pilot)"},
        "engine": "live",
        "n_records": len(rows),
        "n_requests_issued": n_requests,
        "status_counts": status_counts,
        "n_none": sum(1 for r in ok if r["none"]),
        "n_repaired_leniently": sum(1 for r in ok if r.get("repairs")),
        "lenient_repairs": repair_counts,
        "n_by_repair_round": _count(r["round"] for r in rows),
        "started_at": started,
        "ended_at": _utcnow(),
        "wall_clock_s": round(time.monotonic() - t0, 1),
        "tags_path": str(run.tags_path),
        "tags_sha256": sha256_file(run.tags_path),
        "wave_root": str(wave_root),
    }
    atomic_write(run.manifest_path, json.dumps(manifest, indent=1))
    on_event(
        f"[section-tags {run.corpus}] done: {status_counts} · none {manifest['n_none']} · "
        f"{n_requests} requests · {manifest['wall_clock_s']} s -> {run.tags_path}"
    )
    return manifest


def _row(rid: str, corpus: str, record: dict, tags: dict | None, status: str, round_no: int,
         repairs: list[str] | None = None, error: str | None = None) -> dict:
    row: dict[str, Any] = {"id": rid, "status": status, "round": round_no}
    if tags is not None:
        row.update(tags)
    else:
        row.update({"primary": [], "secondary": [], "none": None, "confidence": None})
    if corpus == "advice":
        row["lineage_section_id"] = record.get("section_id")
    if repairs:
        row["repairs"] = repairs
    if error:
        row["error"] = error
    return row


def _count(values: Iterable[Any]) -> dict[str, int]:
    out: dict[str, int] = {}
    for v in values:
        out[str(v)] = out.get(str(v), 0) + 1
    return out


def _utcnow() -> str:
    import datetime as _dt

    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")
