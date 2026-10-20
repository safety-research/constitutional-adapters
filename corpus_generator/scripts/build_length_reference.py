#!/usr/bin/env python3
"""Freeze the control corpus's LENGTH REFERENCE from the v2 production finals:
the 5-point advice length menu (quintile-bin means
of v2 advice `assistant` word counts, rounded to 50) and the per-corpus
length-gate references (mean + p90 words of advice `assistant` / stories
`text`). Prints the YAML block to paste into a control config; the gate
references frozen in configs/control-scale.yaml are asserted against this
script's output by tests/test_control_scale.py whenever the v2 finals are present.

Usage: uv run python scripts/build_length_reference.py [--advice PATH] [--stories PATH]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from corpus_generator.io import iter_jsonl  # noqa: E402
from corpus_generator.lengths import quintile_menu, reference_from_records  # noqa: E402

DEFAULT_ADVICE = ROOT / "data/advice/final-production-20260831-v2.jsonl"
DEFAULT_STORIES = ROOT / "data/stories/final-production-20260831-v2.jsonl"


def build(advice_path: Path, stories_path: Path) -> dict:
    advice = list(iter_jsonl(advice_path))
    stories = list(iter_jsonl(stories_path))
    menu = quintile_menu(advice, "advice")
    return {
        "menu": menu,
        "advice": {**reference_from_records(advice, "advice"), "source": advice_path.name},
        "stories": {**reference_from_records(stories, "stories"), "source": stories_path.name},
    }


def format_yaml(ref: dict) -> str:
    m, a, s = ref["menu"], ref["advice"], ref["stories"]
    return "\n".join([
        "advice:",
        "  # 5-point length menu = quintile-bin means of v2 advice `assistant` words",
        f"  # (bin means {m['bin_means']}, n={m['n']}), rounded to 50; equal weights",
        f"  length_menu: {m['menu']}",
        f"  length_weights: {m['weights']}",
        "filter:",
        "  length_gates:",
        "    advice:",
        f"      reference_mean_words: {a['mean_words']}",
        f"      reference_p90_words: {a['p90_words']}",
        f"      reference_source: \"{a['source']} ({a['field']} field, n={a['n']})\"",
        "      mean_tol: 0.10",
        "      p90_tol: 0.15",
        "    stories:",
        f"      reference_mean_words: {s['mean_words']}",
        f"      reference_p90_words: {s['p90_words']}",
        f"      reference_source: \"{s['source']} ({s['field']} field, n={s['n']})\"",
        "      mean_tol: 0.10",
        "      p90_tol: 0.15",
    ])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--advice", type=Path, default=DEFAULT_ADVICE)
    ap.add_argument("--stories", type=Path, default=DEFAULT_STORIES)
    args = ap.parse_args()
    print(format_yaml(build(args.advice, args.stories)))


if __name__ == "__main__":
    main()
