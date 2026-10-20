"""A sampling row that a person supplied says so in every config value that names it, so no copy states a vendor or model
card recommendation the recorded evidence does not support.

Two kinds of row were supplied rather than read from a cached file at the pinned revision:

* the NVIDIA-Nemotron-3-Nano-4B general-chat setting, temperature 1.0 and top_p 1.0: the model card cached at dfaf35de
  recommends top_p 0.95 for reasoning tasks and for tool calling and shows 1.0 only in a curl example
* the Qwen3.5-4B THINKING-mode rows (the 'precise' row, the 'general' row, the thinking row of the GSM8K readout): the
  snapshot at 851bf6e8 ships no model card and no generation_config.json

Each value naming one of them carries "reported to the project" beside the claim. Scope: the config tree outside
``configs/registry/`` and the two frozen matrix fixtures that copy generated configs; comment lines are not values.
"""
from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
QUALIFIER = "reported to the project"
#: the claims a supplied row makes about its source
ANCHORS = re.compile(r"general-chat|THINKING-mode '(?:precise|general)' row|the vendor's thinking row")
FIXTURES = ("tests/fixtures/eight_arms/gsm8k_matrix_8f5f1495.json", "tests/fixtures/math500_matrix_55e3cbdd.json")
SUFFIXES = (".yaml", ".yml", ".json")


def _files():
    for p in sorted((REPO / "configs").rglob("*")):
        rel = p.relative_to(REPO)
        if p.is_file() and p.suffix in SUFFIXES and rel.parts[1:2] != ("registry",):
            yield p
    for f in FIXTURES:
        if (REPO / f).is_file():
            yield REPO / f


def test_a_supplied_sampling_row_says_it_was_reported_wherever_it_is_named():
    seen, bad = 0, []
    for p in _files():
        text = p.read_text(encoding="utf-8", errors="replace")
        if not ANCHORS.search(text):
            continue
        for i, line in enumerate(text.splitlines(), 1):
            if line.lstrip().startswith("#") or not ANCHORS.search(line):
                continue
            seen += 1
            if QUALIFIER not in line:
                m = ANCHORS.search(line)
                bad.append(f"{p.relative_to(REPO)}:{i}: ...{line[max(0, m.start() - 60):m.end() + 60]}...")
    assert seen > 0, "no config value names a supplied sampling row; the anchors no longer match the configs"
    assert not bad, (f"{len(bad)} of {seen} values name a supplied sampling row without saying it was reported: "
                     + " | ".join(bad[:6]))


def test_the_nemotron_general_chat_setting_is_not_called_the_vendors_recommendation():
    """The plain claim the cached card contradicts never appears, qualified or not."""
    claim = re.compile(r"the vendor's general-chat recommendation|this preset runs NVIDIA's general-chat")
    hits = [f"{p.relative_to(REPO)}" for p in _files() if claim.search(p.read_text(encoding="utf-8", errors="replace"))]
    assert not hits, f"{len(hits)} files call the reported top_p 1.0 setting the vendor's recommendation: {hits[:6]}"
