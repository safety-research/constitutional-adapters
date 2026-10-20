#!/usr/bin/env python3
"""Sweep every shard store under every ``transcripts_*`` root and say what the target-decode PIN's resolver
establishes for each -- the peer's acceptance method (2026-09-16), kept as a standing check.

Why this exists: the pin's first resolver chain was read as correct and was INERT on the whole corpus -- 0 of
5,117 shard stores pin-able, the four canonical Qwen 4,096 anchors included -- because each rung was true only
of data shaped after it (a ``target_decode_resolution`` block nothing had written yet; a ``mode`` key the
anchors' records never carried). Its successor established the anchors' WIRE (temperature 1.0, nothing else)
and would have re-sent it over a server whose fill of the omitted keys has since moved (e631261d: Qwen3.5's
--override-generation-config) -- the request reproduced, the engine values not. No unit test could catch
either: the stores they failed on are real data whose shape nobody had asserted. So the resolver is exercised
here against the stores themselves, and ``tests/test_anchor_decode_sweep_corpus.py`` asserts COUNTS so the
chain going inert -- or going credulous -- again fails the build instead of inviting a reader to eyeball a table.

A shard is PIN-ABLE only when its records establish a pin mode uniquely AND every server-defaultable key it
omitted is pinned by the model's shipped generation_config (the server's fill of any other omitted key is not
recorded by the store). On the corpus of 2026-09-16 NO store meets the second condition: every one omitted
``min_p`` / ``repetition_penalty`` (no shipped file states them) and the Qwen anchors omitted ``top_p`` /
``top_k`` too. That is the honest number until ``served_target.json`` records the standup's fill.

Usage::

  python scripts/sdfbase/full22/anchor_decode_sweep.py [--root DIR] [--stores]

``--root`` defaults to :data:`DEFAULT_ROOT` (``$SISTEER_TRANSCRIPTS_ROOT`` when set). Read-only; no store is
touched. ``--stores`` adds the per-STORE view (``<root>/<view>/<attack>/<cell>``: all shards agreeing).
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "scripts"))

import jailbreak_run as jr  # noqa: E402


def _default_root() -> Path:
    """$SISTEER_TRANSCRIPTS_ROOT when set, else the run-state directory under the site's state root ($STATE, through the site
    config: scripts/site_paths.py). Where $STATE does not resolve the placeholder spelling stays, which names no directory,
    so main() refuses it by name and the standing corpus check skips."""
    env = os.environ.get("SISTEER_TRANSCRIPTS_ROOT")
    if env is not None:
        return Path(env).expanduser()
    from site_paths import SitePathError, site_path
    spec = "$STATE/romp/candidate-si-refit/run-state/full22"
    try:
        return site_path(spec, "SISTEER_TRANSCRIPTS_ROOT")
    except SitePathError:
        return Path(spec)


DEFAULT_ROOT = _default_root()
#: the four canonical Qwen 4,096 anchors the opt-out exists to protect (the Arditi all-positions lane pairs
#: its Phase F legs against them); the standing test asserts every one of their shards is ESTABLISHED (explicit,
#: temperature 1.0) and that every one REFUSES a pin for the one honest reason: the server fill is unrecorded
PROTECTED_QWEN_ANCHOR_ROOTS = ("transcripts_anchor4096_qwen35_4b", "transcripts_anchor4096r2_qwen35_4b",
                               "transcripts_cap4096_qwen35_4b", "transcripts_cap4096aa_qwen35_4b")
_MANIFESTS = ("served_target.json", "attacker_engine.json")
#: the outcome classes the table reports; a refusal outside REFUSED_KNOWN is a resolver defect
FILL_UNRECORDED = "FILL UNRECORDED"
REFUSED_KNOWN = ("records no target decode", "records disagree")


def _family(model: str | None, view: str) -> str:
    for key, needle in (("gemma4e4b", "gemma"), ("qwen35_4b", "Qwen"), ("nemotron3nano4b", "Nemotron"),
                        ("llama8b", "Llama"), ("glm46vflash", "GLM")):
        if model and needle.lower() in model.lower():
            return key
    for key in ("gemma4e4b", "qwen35_4b", "nemotron3nano4b", "llama8b", "glm46vflash"):
        if key in view:
            return key
    return "unknown"


def shard_stores(root: Path):
    """Every directory holding a manifest under ``<root>/transcripts_*``, as (store_root, view, attack, cell,
    shard_dir) -- the layout is ``<root>/<view>/<attack>/<cell>/shard<NN>``."""
    root = Path(root)
    for store_root in sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith("transcripts_")):
        for dp, dns, fns in os.walk(store_root):
            if any(m in fns for m in _MANIFESTS):
                dns[:] = []
                rel = Path(dp).relative_to(store_root).parts + ("", "", "")
                yield store_root.name, rel[0], rel[1], rel[2], Path(dp)


def outcome_of(rec: dict) -> str:
    if rec["kind"] == "vendor":
        return f"vendor_preset via {rec['read_from']}"
    if rec.get("ambiguous"):
        return "AMBIGUOUS: " + " | ".join(m.split(" (")[0] if m.startswith("vendor_preset") else m for m in rec["candidates_matching"])
    base = f"{rec['mode']} via {rec['read_from'].split(' (')[0]}"
    if rec.get("mode") in jr.TARGET_DECODE_PIN_MODES and not rec.get("fill_reproducible", True):
        return f"{base} ({FILL_UNRECORDED}: {', '.join(rec['omitted_unpinned'])})"
    return base


def classify(row: dict) -> str:
    """pinable | established_fill_unrecorded | ambiguous | vendor | refused"""
    o = row["outcome"]
    if row["pinable"]:
        return "pinable"
    if FILL_UNRECORDED in o:
        return "established_fill_unrecorded"
    if o.startswith("AMBIGUOUS"):
        return "ambiguous"
    if o.startswith("vendor"):
        return "vendor"
    return "refused"


def sweep(root: Path = DEFAULT_ROOT) -> list[dict]:
    """One row per shard store: store_root, view, attack, cell, shard, family, outcome, pinable, record."""
    rows = []
    for store_root, view, attack, cell, shard in shard_stores(root):
        model = None
        st = shard / "served_target.json"
        if st.is_file():
            try:
                model = json.loads(st.read_text()).get("base_model")
            except ValueError:
                model = None
        try:
            rec = jr._anchor_recorded_decode(shard, attack=attack or "sweep", model=model)
            out = outcome_of(rec)
            pinable = (rec["kind"] == "values" and not rec.get("ambiguous")
                       and rec.get("mode") in jr.TARGET_DECODE_PIN_MODES and rec.get("fill_reproducible", False))
        except SystemExit as err:
            msg = str(err)
            out = "REFUSED: " + next((k for k in REFUSED_KNOWN if k in msg),
                                     "cannot be established" if "cannot be established" in msg else msg[:80])
            rec, pinable = None, False
        rows.append({"store_root": store_root, "view": view, "attack": attack, "cell": cell, "shard": shard.name,
                     "family": _family(model, view), "outcome": out, "pinable": pinable, "record": rec})
    return rows


def store_table(rows: list[dict]) -> dict:
    """Per STORE (root/view/attack/cell): the shard outcomes; a store is pin-able when every shard is and they agree."""
    by = collections.defaultdict(list)
    for r in rows:
        by[(r["store_root"], r["view"], r["attack"], r["cell"])].append(r)
    out = {}
    for key, rs in by.items():
        outcomes = collections.Counter(r["outcome"] for r in rs)
        out[key] = {"n_shards": len(rs), "outcomes": dict(outcomes), "family": rs[0]["family"],
                    "pinable": all(r["pinable"] for r in rs) and len(outcomes) == 1}
    return out


def summarize(rows: list[dict]) -> str:
    lines = [f"{len(rows)} shard stores under transcripts_* roots"]
    for out, n in collections.Counter(r["outcome"] for r in rows).most_common():
        lines.append(f"  {n:5d}  {out}")
    lines.append("pin-able shard stores: %d" % sum(r["pinable"] for r in rows))
    fam = collections.defaultdict(collections.Counter)
    for r in rows:
        fam[r["family"]][classify(r)] += 1
    for f in sorted(fam):
        lines.append(f"  {f:16s} " + ", ".join(f"{k} {v}" for k, v in sorted(fam[f].items())))
    prot = [r for r in rows if r["store_root"] in PROTECTED_QWEN_ANCHOR_ROOTS]
    c = collections.Counter(classify(r) for r in prot)
    lines.append(f"protected Qwen anchors: {len(prot)} shards -- " + ", ".join(f"{k} {v}" for k, v in sorted(c.items())))
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", default=str(DEFAULT_ROOT))
    ap.add_argument("--stores", action="store_true", help="also print the per-store table")
    a = ap.parse_args(argv)
    root = Path(a.root).expanduser()
    if not root.is_dir():
        print(f"no such root: {root}", file=sys.stderr)
        return 2
    rows = sweep(root)
    print(summarize(rows))
    if a.stores:
        st = store_table(rows)
        print(f"{len(st)} stores; {sum(v['pinable'] for v in st.values())} pin-able")
        for key, v in sorted(st.items()):
            print(f"  {'PIN' if v['pinable'] else '   '} {'/'.join(key):90s} n={v['n_shards']:2d} {v['outcomes']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
