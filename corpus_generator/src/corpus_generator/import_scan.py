"""Release-time PLACEBO-IMPORT scan.

The control's reference document (placebo v2) is a handbook of careful work
whose own examples are everyday crafts — the very domains the menu assigns —
so WORD-level overlap ("swatch", "dry-fit", "test piece" as ordinary craft
talk) is expected and is not a leak. What must not happen is the document's
WORDING reappearing in trainable text. Three probes, all report-only (a hit
is something for the reader to read in context, never an auto-drop):

1. the PHRASE battery — the document's distinctive coinages
   (``filter.import_scan.phrases``, case-insensitive regexes);
2. the VALUE-WORD battery — the v1 constitution/values vocabulary probes
   (``filter.import_scan.value_words``, substrings), because the corpus is
   value-neutral and several of these words have innocent senses that only
   context can tell apart;
3. a verbatim N-GRAM overlap — every ``ngram_n``-gram (default 8, over
   normalized words) shared between the document and a record's trainable
   text, reported with the record id and both contexts.

Trainable text is the record's TRAINABLE fields only (advice: system, user,
assistant; stories: text) — metadata like critiques and rationales may
legitimately discuss the document. Output: ``<artifact>.import-scan.json``
next to the scanned artifact, plus a text summary.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from .config import RunConfig
from .io import atomic_write

TRAINABLE_FIELDS = {"advice": ("system", "user", "assistant"), "stories": ("text",)}
_WORD_RE = re.compile(r"[a-z0-9']+")


def import_scan_path(artifact: Path) -> Path:
    return artifact.with_name(artifact.stem + ".import-scan.json")


def _words(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def _ngrams(words: list[str], n: int) -> dict[str, int]:
    """n-gram -> first word offset (for context)."""
    out: dict[str, int] = {}
    for i in range(len(words) - n + 1):
        g = " ".join(words[i : i + n])
        out.setdefault(g, i)
    return out


def _rec_id(raw: dict) -> str:
    return raw.get("lineage", {}).get("custom_id") or raw.get("scenario_id") or "?"


def _context(text: str, start: int, end: int, pad: int = 60) -> str:
    return re.sub(r"\s+", " ", text[max(0, start - pad) : end + pad]).strip()


def _word_context(words: list[str], offset: int, n: int, pad: int = 8) -> str:
    return " ".join(words[max(0, offset - pad) : offset + n + pad])


def scan_records(
    records: list[dict],
    corpus: str,
    document: str,
    *,
    phrases: list[str],
    value_words: list[str],
    ngram_n: int = 8,
    max_contexts: int = 5,
) -> dict:
    fields = TRAINABLE_FIELDS[corpus]
    doc_words = _words(document)
    doc_grams = _ngrams(doc_words, ngram_n)
    phrase_rx = [(p, re.compile(p, re.IGNORECASE)) for p in phrases]

    phrase_hits: dict[str, dict] = {
        p: {"pattern": p, "records": 0, "occurrences": 0, "contexts": []} for p in phrases
    }
    value_hits: dict[str, dict] = {
        w: {"probe": w, "records": 0, "occurrences": 0, "contexts": []} for w in value_words
    }
    shared: dict[str, dict] = {}  # ngram -> {records, record_ids, record_context, document_context}
    records_with_shared = 0

    for r in records:
        rid = _rec_id(r)
        texts = [(f, r.get(f) or "") for f in fields if r.get(f)]
        joined = "\n".join(t for _f, t in texts)
        low = joined.lower()
        for p, rx in phrase_rx:
            ms = list(rx.finditer(joined))
            if ms:
                h = phrase_hits[p]
                h["records"] += 1
                h["occurrences"] += len(ms)
                if len(h["contexts"]) < max_contexts:
                    h["contexts"].append({"id": rid, "text": _context(joined, ms[0].start(), ms[0].end())})
        for w in value_words:
            starts = [m.start() for m in re.finditer(re.escape(w), low)]
            if starts:
                h = value_hits[w]
                h["records"] += 1
                h["occurrences"] += len(starts)
                if len(h["contexts"]) < max_contexts:
                    h["contexts"].append({"id": rid, "text": _context(joined, starts[0], starts[0] + len(w))})
        words = _words(joined)
        grams = _ngrams(words, ngram_n)
        found = False
        for g, off in grams.items():
            if g in doc_grams:
                found = True
                entry = shared.setdefault(
                    g, {"ngram": g, "records": 0, "record_ids": [],
                        "record_context": _word_context(words, off, ngram_n),
                        "document_context": _word_context(doc_words, doc_grams[g], ngram_n)},
                )
                entry["records"] += 1
                if len(entry["record_ids"]) < max_contexts:
                    entry["record_ids"].append(rid)
        records_with_shared += found

    # merge shifted copies of one lifted passage into maximal runs for the reader
    n_rec = len(records)
    return {
        "corpus": corpus,
        "n_records": n_rec,
        "fields": list(fields),
        "ngram_n": ngram_n,
        "phrase_battery": sorted(
            (h for h in phrase_hits.values()), key=lambda h: (-h["records"], h["pattern"])
        ),
        "value_word_battery": sorted(
            (h for h in value_hits.values()), key=lambda h: (-h["records"], h["probe"])
        ),
        "shared_ngrams": sorted(shared.values(), key=lambda e: (-e["records"], e["ngram"])),
        "summary": {
            "phrase_hit_records": sum(1 for h in phrase_hits.values() if h["records"]),
            "phrase_records_total": sum(h["records"] for h in phrase_hits.values()),
            "value_word_records_total": sum(h["records"] for h in value_hits.values()),
            "shared_ngrams_distinct": len(shared),
            "records_with_shared_ngram": records_with_shared,
            "records_with_shared_ngram_frac": round(records_with_shared / n_rec, 4) if n_rec else 0.0,
        },
    }


def run_import_scan(
    cfg: RunConfig, corpus: str, records: list[dict], document: str, out_path: Path
) -> dict:
    sc = cfg.filter.import_scan
    res = scan_records(
        records, corpus, document,
        phrases=sc.phrases, value_words=sc.value_words,
        ngram_n=sc.ngram_n, max_contexts=sc.max_contexts,
    )
    res["run_id"] = cfg.run_id
    res["document"] = cfg.constitution_dir
    atomic_write(out_path, json.dumps(res, indent=1, ensure_ascii=False))
    return res


def format_scan(res: dict) -> str:
    s = res["summary"]
    lines = [
        f"import scan — {res['corpus']}, {res['n_records']:,} records, fields {res['fields']}, "
        f"{res['ngram_n']}-gram overlap vs {res.get('document', 'the reference document')}",
        f"  phrase battery: {s['phrase_hit_records']} of {len(res['phrase_battery'])} phrases hit; "
        f"records hit (sum over phrases) {s['phrase_records_total']}",
    ]
    for h in res["phrase_battery"]:
        if h["records"]:
            ex = h["contexts"][0]["text"] if h["contexts"] else ""
            lines.append(f"    {h['records']:5d} rec  /{h['pattern']}/   e.g. …{ex}…")
    lines.append(
        f"  value-word battery: records hit (sum over probes) {s['value_word_records_total']}"
    )
    for h in res["value_word_battery"]:
        if h["records"]:
            ex = h["contexts"][0]["text"] if h["contexts"] else ""
            lines.append(f"    {h['records']:5d} rec  {h['probe']!r}   e.g. …{ex}…")
    lines.append(
        f"  shared {res['ngram_n']}-grams: {s['shared_ngrams_distinct']} distinct in "
        f"{s['records_with_shared_ngram']} records ({s['records_with_shared_ngram_frac']:.2%})"
    )
    for e in res["shared_ngrams"][:15]:
        lines.append(f"    {e['records']:5d} rec  \"{e['ngram']}\"  [{', '.join(e['record_ids'][:3])}]")
        lines.append(f"           record:   …{e['record_context']}…")
        lines.append(f"           document: …{e['document_context']}…")
    return "\n".join(lines)
