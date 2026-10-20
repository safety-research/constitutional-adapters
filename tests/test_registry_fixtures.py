"""The static synthetic fixtures under tests/fixtures/registry validate (sample) or are refused naming the
expected field (known-bad), and the vocabulary check fires on the known-bad text."""
from __future__ import annotations

import json
import os

from sisteer.registry import records as R
from sisteer.registry import validate as VAL

FIX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "registry")


def test_sample_records_all_validate():
    n = 0
    for _, rec, sha in R.read_record_file(os.path.join(FIX, "sample_records.jsonl")):
        out = VAL.validate_record(rec, "sample", sha)
        assert isinstance(out, VAL.Validated), out
        assert rec["producer"]["host"] == "TESTHOST"
        n += 1
    assert n >= 20


def _known_bad_cases():
    """Assembled at run time (never stored: a `*_key` field beside a 64-hex sha is what the secret scanner refuses)."""
    from sisteer.registry import synth
    cases = []
    p = synth.run_payload("fixture_bad", i=0); p["eval_id"] = "gsm8k"; del p["item_set_id"]
    cases.append(("item_set_id", synth.rec("run", p, synth.t(0))))
    p = synth.run_payload("fixture_bad2", i=0); p["serving_path"] = "vllm"
    cases.append(("serving_path", synth.rec("run", p, synth.t(0))))
    p = synth.run_payload("fixture_bad3", i=0); p["code_sha"] = "unverified"
    cases.append(("code_sha", synth.rec("run", p, synth.t(0))))
    p = synth.run_payload("fixture_bad5", i=0); p["code_sha"] = "a" * 40       # a placeholder sha: the shape that reached the live log
    cases.append(("code_sha", synth.rec("run", p, synth.t(0))))
    p = synth.run_payload("fixture_bad4", i=0); p["extra"] = {"api" + "_key": "not-a-real-value"}
    cases.append(("extra.api_key", synth.rec("run", p, synth.t(0))))
    cases.append(("model_id", synth.rec("run_patch", {"run_id": "r", "fields": {"model_id": "other"}}, synth.t(1))))
    return cases


def test_known_bad_records_refused_naming_field():
    for field, rec in _known_bad_cases():
        out = VAL.validate_record(rec, "known-bad")
        assert isinstance(out, VAL.Malformed), field
        assert out.field == field, (out.field, out.reason)


def test_known_bad_vocabulary_text_fires():
    from test_registry_vocabulary import hits
    with open(os.path.join(FIX, "known_bad_vocabulary.txt"), encoding="utf-8") as fh:
        text = fh.read()
    found = {h[0].lower() for h in hits(text)}
    assert {"arm", "lane", "cell", "grader", "leg", "wave", "rung", "view"} <= found
