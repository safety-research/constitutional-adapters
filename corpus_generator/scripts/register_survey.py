#!/usr/bin/env python3
"""Thin wrapper for the register-discovery survey:

    uv run python scripts/register_survey.py <artifact.jsonl> [...] [--config CFG]

Equivalent to `corpusgen register-survey --config CFG --input <artifact> ...`; the
config is optional here (without it, no configured-family match rates)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from corpus_generator.config import RunConfig  # noqa: E402
from corpus_generator.io import iter_jsonl  # noqa: E402
from corpus_generator.register_survey import (  # noqa: E402
    format_survey, sample_records, survey, to_json,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("inputs", nargs="+", type=Path)
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--opener-words", type=int, default=3)
    ap.add_argument("--n", type=int, action="append", default=None)
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--min-frac", type=float, default=0.01)
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()
    families = RunConfig.load(args.config).filter.register_families if args.config else []
    results = []
    for path in args.inputs:
        records = sample_records(list(iter_jsonl(path)), args.sample)
        res = survey(records, families=families, opener_words=args.opener_words,
                     ngram_sizes=tuple(args.n or (4, 6, 8)), top=args.top, min_frac=args.min_frac)
        res["source"] = str(path)
        results.append(res)
        print(f"# source: {path}")
        print(format_survey(res))
        print()
    if args.json:
        args.json.write_text(to_json(results[0] if len(results) == 1 else {"surveys": results}))


if __name__ == "__main__":
    main()
