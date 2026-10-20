#!/usr/bin/env python3
"""Thin wrapper: stage the three reference tokenizers into data/tokenizers/."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from corpus_generator.tokens import stage_tokenizers  # noqa: E402

if __name__ == "__main__":
    manifest = stage_tokenizers(Path(__file__).resolve().parents[1] / "data/tokenizers")
    print(json.dumps(manifest, indent=1))
