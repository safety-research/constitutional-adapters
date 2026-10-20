"""Register-discovery survey over a calibration (or final) artifact.

The production lesson, quantified: v2 advice lost 34% of records to two
system-prompt opener families and stories lost 7.4% to one recurring beat,
all discovered only at assembly. This survey finds such families BEFORE the
full run, on the pilot's calibration artifact, so they can be (i) banned
prompt-side in the overlay templates and (ii) frozen as ``register_families``
backstops. Three views, each a ranked table with record fractions:

- **openers** — the first k words of every trainable field (advice: system /
  user / assistant; stories: text), normalized (case, leading markdown,
  punctuation), plus the "label" opener of system prompts (the text up to
  the first colon: "Deployment context:", "Purpose of this service:", ...);
- **n-grams** — word n-grams (default n = 4, 6, 8) ranked by DOCUMENT
  frequency (distinct records containing them, never raw repeats), with
  shifted duplicates of the same phrase merged into one maximal phrase and
  stopword-only grams dropped;
- **configured families** — the live match rate of every
  ``filter.register_families`` entry, so a near-miss (an opener the current
  regex does not catch) shows up right next to the family it belongs to.

Pure functions over record dicts; no API, no filesystem except the CLI.
"""

from __future__ import annotations

import json
import random
import re
from collections import Counter

from .config import RegisterFamily

TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9'’-]*")
LEADING_JUNK_RE = re.compile(r"^[\s#*>\-•\"'“‘\[\(]+")

STOPWORDS = frozenset(
    "a an the and or but if then than that this these those it its is are was were be been "
    "being to of in on at by for from with without as into onto over under about after before "
    "up down out off so not no nor do does did done have has had having will would shall should "
    "can could may might must i me my we our you your he him his she her they them their who "
    "whom which what when where why how all any both each few more most other some such only "
    "own same too very just also there here one two s t ll re ve d m".split()
)

TRAINABLE_FIELDS = {"advice": ("system", "user", "assistant"), "stories": ("text",)}
RESPONSE_FIELDS = {"advice": ("assistant", "user"), "stories": ("text",)}


def detect_corpus(records: list[dict]) -> str:
    if not records:
        raise ValueError("cannot detect the corpus of an empty artifact")
    r = records[0]
    if "assistant" in r and "user" in r:
        return "advice"
    if "text" in r and "category_id" in r:
        return "stories"
    raise ValueError("records are neither advice (system/user/assistant) nor stories (text)")


def _rec_id(r: dict) -> str:
    return (r.get("lineage") or {}).get("custom_id") or r.get("scenario_id") or r.get("call_id") or "?"


def tokens(text: str | None) -> list[str]:
    return TOKEN_RE.findall((text or "").casefold())


def normalize_opener(text: str | None, k: int) -> str | None:
    """First k word tokens of a field, case/markdown/punctuation-normalized;
    None when the field is empty or shorter than k words."""
    toks = tokens(LEADING_JUNK_RE.sub("", text or ""))
    if len(toks) < k:
        return None
    return " ".join(toks[:k])


def label_opener(text: str | None, max_words: int = 6) -> str | None:
    """The 'label:' opener convention — the text up to the first colon when it
    falls within the first ``max_words`` words ("Deployment context:",
    "Context for this deployment:")."""
    head = LEADING_JUNK_RE.sub("", (text or "")).split("\n", 1)[0]
    if ":" not in head:
        return None
    label = head.split(":", 1)[0]
    toks = tokens(label)
    if not toks or len(toks) > max_words:
        return None
    return " ".join(toks) + ":"


def _rank(counter: Counter, examples: dict, n_field: int, n_records: int, top: int,
          min_count: int, key_name: str) -> list[dict]:
    rows = [
        {key_name: key, "records": c,
         "frac_of_field": round(c / n_field, 4) if n_field else 0.0,
         "frac_of_records": round(c / n_records, 4) if n_records else 0.0,
         "example_id": examples[key]}
        for key, c in counter.items() if c >= min_count
    ]
    rows.sort(key=lambda r: (-r["records"], r[key_name]))
    return rows[:top]


def opener_table(records: list[dict], field: str, k: int, top: int = 25,
                 min_count: int = 2, labels: bool = False) -> dict:
    """Ranked openers of one field. ``labels=True`` ranks the 'label:' form
    instead of the first-k-words form."""
    counter: Counter = Counter()
    examples: dict[str, str] = {}
    n_field = 0
    for r in records:
        text = r.get(field)
        if not (text or "").strip():
            continue
        n_field += 1
        op = label_opener(text) if labels else normalize_opener(text, k)
        if op is None:
            continue
        counter[op] += 1
        examples.setdefault(op, _rec_id(r))
    return {
        "field": field, "kind": "label" if labels else f"first_{k}_words",
        "n_field": n_field, "n_records": len(records),
        "rows": _rank(counter, examples, n_field, len(records), top, min_count, "opener"),
    }


def _content_gram(gram: tuple[str, ...]) -> bool:
    return any(t not in STOPWORDS and len(t) >= 4 and not t.isdigit() for t in gram)


def _merge_shifted(rows: list[dict]) -> list[dict]:
    """Shifted n-grams of one repeated phrase share a record set; chain the
    grams that overlap by n-1 tokens (within one record set) into a single
    maximal phrase, whatever order they were ranked in. Grams that share a
    record set but do not chain stay separate rows."""
    rank = {r["_tokens"]: i for i, r in enumerate(rows)}
    groups: dict[frozenset, list[dict]] = {}
    for row in rows:
        groups.setdefault(row["_docs"], []).append(row)
    out: list[dict] = []
    for grp in groups.values():
        by_tokens = {r["_tokens"]: r for r in grp}
        by_prefix: dict[tuple, list[tuple]] = {}
        for g in by_tokens:
            by_prefix.setdefault(g[:-1], []).append(g)
        has_pred = {
            h for g in by_tokens for h in by_prefix.get(g[1:], []) if h != g
        }
        used: set[tuple] = set()
        for g in sorted((g for g in by_tokens if g not in has_pred), key=rank.__getitem__):
            phrase, cur = list(g), g
            used.add(g)
            while True:
                nxt = [h for h in by_prefix.get(cur[1:], []) if h not in used and h != cur]
                if not nxt:
                    break
                cur = min(nxt, key=rank.__getitem__)
                phrase.append(cur[-1])
                used.add(cur)
            out.append({**by_tokens[g], "_tokens": tuple(phrase)})
        for g in by_tokens:  # anything in a pure cycle: emit as-is
            if g not in used:
                out.append(by_tokens[g])
    out.sort(key=lambda r: (-r["records"], r["_tokens"]))
    return out


def ngram_table(records: list[dict], field: str, n: int, top: int = 25,
                min_frac: float = 0.01, min_count: int = 2) -> dict:
    """Word n-grams of one field ranked by document frequency."""
    counter: Counter = Counter()
    examples: dict[tuple, str] = {}
    docs: dict[tuple, set] = {}
    n_field = 0
    for idx, r in enumerate(records):
        toks = tokens(r.get(field))
        if not toks:
            continue
        n_field += 1
        seen = {tuple(toks[i: i + n]) for i in range(len(toks) - n + 1)}
        for g in seen:
            if not _content_gram(g):
                continue
            counter[g] += 1
            docs.setdefault(g, set()).add(idx)
            examples.setdefault(g, _rec_id(r))
    floor = max(min_count, int(min_frac * n_field) if n_field else min_count)
    rows = [
        {"_tokens": g, "_docs": frozenset(docs[g]), "records": c, "example_id": examples[g]}
        for g, c in counter.items() if c >= floor
    ]
    rows.sort(key=lambda r: (-r["records"], r["_tokens"]))
    merged = _merge_shifted(rows)[:top]
    out_rows = [
        {"ngram": " ".join(r["_tokens"]), "records": r["records"],
         "frac_of_field": round(r["records"] / n_field, 4) if n_field else 0.0,
         "frac_of_records": round(r["records"] / len(records), 4) if records else 0.0,
         "example_id": r["example_id"]}
        for r in merged
    ]
    return {"field": field, "n": n, "n_field": n_field, "n_records": len(records),
            "min_records": floor, "rows": out_rows}


def family_table(records: list[dict], families: list[RegisterFamily]) -> list[dict]:
    """Live match rate of every configured register family."""
    out = []
    for fam in families:
        rx = re.compile(fam.pattern, re.IGNORECASE)
        probe = rx.match if fam.anchored else rx.search
        matched = [r for r in records if probe(str(r.get(fam.field) or ""))]
        out.append({
            "name": fam.name, "field": fam.field, "anchored": fam.anchored,
            "pattern": fam.pattern, "max_frac": fam.max_frac,
            "matched": len(matched),
            "frac_of_records": round(len(matched) / len(records), 4) if records else 0.0,
            "over_cap": bool(records) and len(matched) / len(records) >= fam.max_frac,
            "example_ids": [_rec_id(r) for r in matched[:3]],
        })
    return out


def sample_records(records: list[dict], n: int, seed: int = 0) -> list[dict]:
    """Seeded subsample (a full-run stories artifact is ~16k x 2k words; the
    n-gram pass over a few thousand records finds every >=2% family)."""
    if n <= 0 or n >= len(records):
        return records
    rng = random.Random(f"{seed}:register-survey")
    idx = sorted(rng.sample(range(len(records)), n))
    return [records[i] for i in idx]


def survey(records: list[dict], corpus: str | None = None, *,
           families: list[RegisterFamily] | None = None, opener_words: int = 3,
           ngram_sizes: tuple[int, ...] = (4, 6, 8), top: int = 25,
           min_frac: float = 0.01) -> dict:
    corpus = corpus or detect_corpus(records)
    result: dict = {
        "corpus": corpus, "n_records": len(records),
        "params": {"opener_words": opener_words, "ngram_sizes": list(ngram_sizes),
                   "top": top, "min_frac": min_frac},
        "openers": [], "ngrams": [], "families": family_table(records, families or []),
    }
    for field in TRAINABLE_FIELDS[corpus]:
        result["openers"].append(opener_table(records, field, opener_words, top))
        if field == "system":
            result["openers"].append(opener_table(records, field, opener_words, top, labels=True))
    for field in RESPONSE_FIELDS[corpus]:
        for n in ngram_sizes:
            result["ngrams"].append(ngram_table(records, field, n, top, min_frac))
    return result


def format_survey(result: dict) -> str:
    lines = [
        f"register survey — {result['corpus']}, {result['n_records']:,} records "
        f"(openers: first {result['params']['opener_words']} words; n-grams by DOCUMENT "
        f"frequency, floor {result['params']['min_frac']:.1%} of records with the field)",
    ]
    if result["families"]:
        lines.append("\n## configured register families (live match rate)")
        lines.append(f"{'family':<32} {'field':<9} {'matched':>7} {'frac':>7}  cap    status")
        for f in result["families"]:
            lines.append(
                f"{f['name']:<32} {f['field']:<9} {f['matched']:>7,} {f['frac_of_records']:>7.2%}  "
                f"{f['max_frac']:<6.2%} {'OVER CAP' if f['over_cap'] else 'ok'}"
            )
    for t in result["openers"]:
        lines.append(
            f"\n## openers — {t['field']} ({t['kind']}; {t['n_field']:,} records carry the field)"
        )
        if not t["rows"]:
            lines.append("(no opener shared by >=2 records)")
            continue
        lines.append(f"{'records':>7} {'of field':>8} {'of all':>7}  opener")
        for r in t["rows"]:
            lines.append(
                f"{r['records']:>7,} {r['frac_of_field']:>8.2%} {r['frac_of_records']:>7.2%}  "
                f"{r['opener']}   [{r['example_id']}]"
            )
    for t in result["ngrams"]:
        lines.append(
            f"\n## repeated {t['n']}-grams — {t['field']} (>= {t['min_records']} records; "
            f"shifted copies merged into maximal phrases)"
        )
        if not t["rows"]:
            lines.append("(none above the floor)")
            continue
        lines.append(f"{'records':>7} {'of field':>8}  phrase")
        for r in t["rows"]:
            lines.append(
                f"{r['records']:>7,} {r['frac_of_field']:>8.2%}  {r['ngram']}   [{r['example_id']}]"
            )
    return "\n".join(lines)


def to_json(result: dict) -> str:
    return json.dumps(result, indent=1, ensure_ascii=False)
