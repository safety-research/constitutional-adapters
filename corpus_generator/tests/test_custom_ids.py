"""custom_id charset [A-Za-z0-9_-]{1,64} — validated, never assumed (this
repository's '|' burn). Every id either pipeline generates must pass."""

from __future__ import annotations

import json
import re

import pytest

from corpus_generator.pipeline import make_engine, run_corpus
from corpus_generator.schema import CustomIdError, validate_custom_id
from corpus_generator.templates import repo_root

CID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


@pytest.mark.parametrize(
    "bad", ["", "a|b", "a b", "x" * 65, "café", "adv:e", "a\nb", "id,1"]
)
def test_invalid_ids_rejected(bad):
    with pytest.raises(CustomIdError):
        validate_custom_id(bad)


@pytest.mark.parametrize("good", ["a", "adv-b1-p00-t01-s02-r1", "x" * 64, "A_Z-9"])
def test_valid_ids_accepted(good):
    assert validate_custom_id(good) == good


def test_every_generated_id_in_both_pipelines_is_valid(tmp_path):
    from test_run_lifecycle import _cfg, _make_client

    root = repo_root()
    cfg = _cfg("cid-test", tmp_path)
    client = _make_client()
    for corpus in ("advice", "stories"):
        run_corpus(cfg, corpus, make_engine(cfg, client, sleep=lambda s: None),
                   root=root, on_event=lambda m: None)
    # topups too (they carry the xNNN tag)
    run_dir = cfg.run_dir(root)
    (run_dir / "advice" / "TOPUP-001.json").write_text(
        json.dumps({"n": 1, "pairs": ["p00-t00"], "slot_start": 1, "n_slots": 1})
    )
    (run_dir / "stories" / "TOPUP-001.json").write_text(
        json.dumps({"n": 1, "call_start": 1, "n_calls": 1})
    )
    for corpus in ("advice", "stories"):
        run_corpus(cfg, corpus, make_engine(cfg, client, sleep=lambda s: None),
                   root=root, on_event=lambda m: None)

    ids = []
    for req_file in run_dir.rglob("requests.jsonl"):
        for line in req_file.read_text().splitlines():
            if line.strip():
                ids.append(json.loads(line)["custom_id"])
    assert len(ids) > 40
    for cid in ids:
        assert CID_RE.fullmatch(cid), cid
    # ids are also unique within each wave file (engine enforces; double-check)
    for req_file in run_dir.rglob("requests.jsonl"):
        wave_ids = [
            json.loads(line)["custom_id"]
            for line in req_file.read_text().splitlines()
            if line.strip()
        ]
        assert len(set(wave_ids)) == len(wave_ids), req_file
