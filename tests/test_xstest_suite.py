"""The XSTest over-refusal suite module (``sisteer.eval.xstest``), the realign registry's third
judged suite, on a synthetic frame — no network, no HF cache:

- item loading over an injected frame: the item schema, the 250 + 200 load gate (a moved pin is
  refused, never re-pinned), subset / seed / cap semantics, schema refusals;
- the record contract of ``run_xstest_arm`` on a stub generator: every F13 field, the arm label,
  ``item_sha`` over exactly the fields the analysis recomputes, ``prompt_sha`` that differs across
  arms while ``item_sha`` pairs them, item-level resume;
- the pin is declared (a full-sha revision in pins.yaml) without touching the network;
- the judge-side label vocabulary the module documents matches the rubric and both readers of it
  (the module itself never scores — scoring is judge-side, by contract).
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

from sisteer.eval import realign_suites, xstest

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts"))

REV = "0123456789abcdef0123456789abcdef01234567"  # a synthetic revision stamp


def _frame(n_safe: int = 250, n_unsafe: int = 200, *, ids=None):
    import pandas as pd

    n = n_safe + n_unsafe
    return pd.DataFrame({
        "id": list(ids) if ids is not None else [f"v2-{i + 1}" for i in range(n)],
        "type": ["homonyms_synth"] * n_safe + ["contrast_homonyms_synth"] * n_unsafe,
        "prompt": [f"synthetic safe prompt {i}" for i in range(n_safe)]
                  + [f"synthetic contrast prompt {i}" for i in range(n_unsafe)],
        "annotation_1": [""] * n, "annotation_2": [""] * n,
        "agreement": [False] * n, "final_label": [""] * n, "completion": [""] * n,
    })


# -- loading -------------------------------------------------------------------


def test_load_all_items_schema_split_and_revision():
    items = xstest.load_xstest(frame=_frame(), revision=REV)
    assert len(items) == 450
    assert all(set(it) == {"item_id", "prompt", "xstest_type", "safe", "suite_revision"} for it in items)
    assert sum(it["safe"] for it in items) == xstest.N_SAFE == 250
    assert sum(not it["safe"] for it in items) == xstest.N_UNSAFE == 200
    assert all(it["suite_revision"] == REV for it in items)
    assert all(isinstance(it["item_id"], str) for it in items)
    assert len({it["item_id"] for it in items}) == 450
    unsafe = next(it for it in items if not it["safe"])
    assert unsafe["xstest_type"].startswith(xstest.CONTRAST_PREFIX)


def test_subsets_seed_and_cap():
    safe = xstest.load_xstest("safe", frame=_frame(), revision=REV)
    unsafe = xstest.load_xstest("unsafe", frame=_frame(), revision=REV)
    assert len(safe) == 250 and all(it["safe"] for it in safe)
    assert len(unsafe) == 200 and not any(it["safe"] for it in unsafe)

    a = [it["item_id"] for it in xstest.load_xstest(frame=_frame(), revision=REV, seed=0)]
    b = [it["item_id"] for it in xstest.load_xstest(frame=_frame(), revision=REV, seed=0)]
    c = [it["item_id"] for it in xstest.load_xstest(frame=_frame(), revision=REV, seed=1)]
    assert a == b                      # deterministic in the seed
    assert a != c and sorted(a) == sorted(c)   # a different seed permutes the same set
    assert a != [f"v2-{i + 1}" for i in range(450)]   # shuffled, not frame order

    capped = xstest.load_xstest("safe", n=7, frame=_frame(), revision=REV)
    assert len(capped) == 7 and [it["item_id"] for it in capped] == [it["item_id"] for it in safe[:7]]

    with pytest.raises(ValueError, match="unknown subset"):
        xstest.load_xstest("benign", frame=_frame(), revision=REV)


def test_the_split_gate_refuses_a_moved_pin_and_a_guessed_schema():
    with pytest.raises(ValueError, match="250"):
        xstest.load_xstest(frame=_frame(249, 201), revision=REV)
    with pytest.raises(ValueError, match="250"):
        xstest.load_xstest(frame=_frame(250, 199), revision=REV)
    with pytest.raises(ValueError, match="'type'"):
        xstest.items_from_frame(_frame().drop(columns=["type"]))
    with pytest.raises(ValueError, match="'id'"):
        xstest.items_from_frame(_frame().drop(columns=["id"]))
    with pytest.raises(ValueError, match="duplicate"):
        xstest.items_from_frame(_frame(ids=["dup"] * 450))


def test_items_from_frame_keeps_frame_order_and_applies_no_subset():
    items = xstest.items_from_frame(_frame(), revision=REV)
    assert [it["item_id"] for it in items] == [f"v2-{i + 1}" for i in range(450)]
    assert [it["safe"] for it in items] == [True] * 250 + [False] * 200


# -- the arm runner --------------------------------------------------------------


@pytest.fixture
def stub_generation(monkeypatch):
    """A rendering + generation stub: the prompt is a visible composition of system and user, the
    response echoes the prompt, and the object / scale handed to the runner are observed."""
    seen = {"calls": []}

    def fake_build_chat_prompt(tok, user, system=None):
        return f"<sys>{system}</sys><user>{user}</user>"

    def fake_generate_with_object(model, tok, rendered, obj=None, scale=1.0, batch_size=8,
                                  max_new_tokens=200, **kw):
        seen["calls"].append({"n": len(rendered), "obj": obj, "scale": scale,
                              "batch_size": batch_size, "max_new_tokens": max_new_tokens})
        return [f"resp:{p}" for p in rendered]

    monkeypatch.setattr(xstest, "build_chat_prompt", fake_build_chat_prompt)
    monkeypatch.setattr(xstest, "generate_with_object", fake_generate_with_object)
    return seen


def _read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_run_xstest_arm_records_every_field_and_resumes(tmp_path, stub_generation):
    items = xstest.load_xstest(frame=_frame(), revision=REV)[:5]
    out = tmp_path / "xstest_baseline.jsonl"

    written = xstest.run_xstest_arm(None, None, items, out_jsonl=out, batch_size=2, max_new_tokens=33)
    assert len(written) == 5 and _read(out) == written
    assert [c["n"] for c in stub_generation["calls"]] == [2, 2, 1]
    assert all(c["max_new_tokens"] == 33 and c["obj"] is None and c["scale"] == 1.0
               for c in stub_generation["calls"])

    for it, rec in zip(items, written):
        assert set(rec) == {"item_id", "prompt", "xstest_type", "safe", "response", "arm",
                            "suite_revision", "item_sha", "prompt_sha"}
        assert rec["item_id"] == it["item_id"] and rec["prompt"] == it["prompt"]
        assert rec["xstest_type"] == it["xstest_type"] and rec["safe"] == it["safe"]
        assert rec["arm"] == "baseline" and rec["suite_revision"] == REV
        rendered = f"<sys>None</sys><user>{it['prompt']}</user>"
        assert rec["response"] == f"resp:{rendered}"
        assert rec["prompt_sha"] == realign_suites.prompt_sha(rendered)
        assert rec["item_sha"] == realign_suites.item_sha(
            {"item_id": it["item_id"], "prompt": it["prompt"], "xstest_type": it["xstest_type"]})
        assert re.fullmatch(r"[0-9a-f]{64}", rec["item_sha"])

    # item-level resume: a second run over the same items generates nothing and appends nothing
    stub_generation["calls"].clear()
    again = xstest.run_xstest_arm(None, None, items, out_jsonl=out, batch_size=2)
    assert again == [] and stub_generation["calls"] == [] and len(_read(out)) == 5

    # a sixth item resumes past the five already written
    more = xstest.load_xstest(frame=_frame(), revision=REV)[:6]
    extra = xstest.run_xstest_arm(None, None, more, out_jsonl=out, batch_size=2)
    assert [r["item_id"] for r in extra] == [more[5]["item_id"]] and len(_read(out)) == 6


def test_arms_differ_in_prompt_sha_and_pair_on_item_sha(tmp_path, stub_generation):
    items = xstest.load_xstest("safe", n=3, frame=_frame(), revision=REV)
    base = xstest.run_xstest_arm(None, None, items, out_jsonl=tmp_path / "b.jsonl")
    prompted = xstest.run_xstest_arm(None, None, items, system="Follow the constitution.",
                                     out_jsonl=tmp_path / "p.jsonl")
    steering_object = object()
    steered = xstest.run_xstest_arm(None, None, items, obj=steering_object, scale=0.75,
                                    out_jsonl=tmp_path / "s.jsonl")

    assert [r["arm"] for r in base] == ["baseline"] * 3
    assert [r["arm"] for r in prompted] == ["prompted"] * 3
    assert [r["arm"] for r in steered] == ["steered"] * 3
    assert stub_generation["calls"][-1]["obj"] is steering_object
    assert stub_generation["calls"][-1]["scale"] == 0.75
    for b, p, s in zip(base, prompted, steered):
        assert b["item_sha"] == p["item_sha"] == s["item_sha"]      # the pairing key
        assert b["prompt_sha"] != p["prompt_sha"]                    # the SI rendered
        assert b["prompt_sha"] == s["prompt_sha"]                    # steering leaves the prompt alone
        assert "<sys>Follow the constitution.</sys>" in p["response"]


# -- the pin and the analysis' view of the items ----------------------------------


def test_the_pin_is_declared_in_pins_yaml_without_the_network():
    pins_path = REPO / "configs" / "capeval" / "pins.yaml"
    assert pins_path.is_file()
    from sisteer.capeval.corpora import load_pins

    pins = load_pins(pins_path)
    entry = xstest.xstest_pin(pins)
    assert xstest.XSTEST_PIN in pins["datasets"]
    assert re.fullmatch(r"[0-9a-f]{40}", xstest.xstest_revision(pins))
    assert entry["verify"] == "parquet" and isinstance(entry["repo_id"], str) and entry["repo_id"]


def test_the_default_load_reads_and_stamps_one_pin_entry(monkeypatch):
    """The default loader resolves the ``xstest_v2`` pin ONCE and uses that one entry both to read
    the frame and to stamp ``suite_revision``. The pins table here changes its revision on every
    read, so a loader that consulted pins.yaml twice would stamp a revision other than the one it
    fetched; no network, no HF cache — the fetch is a stub that records the entry it was handed."""
    from sisteer.capeval import corpora

    reads = []
    fetched = []
    other_rev = "fedcba9876543210fedcba9876543210fedcba98"

    def fake_load_pins(path=None):
        rev = REV if not reads else other_rev          # a different revision on every later read
        reads.append(rev)
        return {"datasets": {xstest.XSTEST_PIN: {"repo_id": "test-org/synthetic-xstest",
                                                "revision": rev, "verify": "parquet"}}}

    def fake_fetch(entry, name):
        fetched.append((dict(entry), name))
        return _frame()

    monkeypatch.setattr(corpora, "load_pins", fake_load_pins)
    monkeypatch.setattr(corpora, "_fetch_pinned_frames", fake_fetch)

    items = xstest.load_xstest()
    assert len(items) == 450 and reads == [REV]                  # pins.yaml read exactly once
    [(entry, name)] = fetched                                    # one fetch, of that entry
    assert name == xstest.XSTEST_PIN and entry["revision"] == REV
    assert {it["suite_revision"] for it in items} == {entry["revision"]}

    # an explicit pins table is used as given: no read of pins.yaml at all, same single entry
    reads.clear(); fetched.clear()
    explicit = {"datasets": {xstest.XSTEST_PIN: {"repo_id": "test-org/synthetic-xstest",
                                                 "revision": other_rev, "verify": "parquet"}}}
    items = xstest.load_xstest("safe", pins=explicit)
    assert reads == [] and [e["revision"] for e, _ in fetched] == [other_rev]
    assert {it["suite_revision"] for it in items} == {other_rev} and len(items) == 250


def test_item_sha_fields_and_universe_match_the_analysis(monkeypatch):
    """The analysis recomputes ``item_sha`` over a frozen field tuple; the runner hashes the same
    one, and the analysis' XSTest universe reads the loader this module exposes."""
    import agentic_control_analysis as aca

    assert aca.SUITE_ITEM_SHA_FIELDS["xstest"] == xstest.ITEM_SHA_FIELDS == ("item_id", "prompt", "xstest_type")
    assert aca.compute_item_sha is realign_suites.item_sha

    synthetic = xstest.load_xstest(frame=_frame(), revision=REV)
    monkeypatch.setattr(xstest, "load_xstest", lambda *a, **k: list(synthetic))
    universe = aca.suite_item_shas("xstest")
    assert universe == {realign_suites.item_sha({f: it[f] for f in xstest.ITEM_SHA_FIELDS})
                        for it in synthetic} and len(universe) == 450

    short = [it for it in synthetic if not it["safe"]] + [it for it in synthetic if it["safe"]][:249]
    monkeypatch.setattr(xstest, "load_xstest", lambda *a, **k: list(short))
    with pytest.raises(SystemExit, match="250"):
        aca.suite_item_shas("xstest")


def test_default_eval_texts_reaches_the_xstest_prompts(monkeypatch):
    from sisteer import agentic_control
    from sisteer.eval import mask

    synthetic = xstest.load_xstest(frame=_frame(), revision=REV)
    monkeypatch.setattr(mask, "load_mask", lambda *a, **k: [])
    monkeypatch.setattr(xstest, "load_xstest", lambda *a, **k: list(synthetic))
    texts = agentic_control.default_eval_texts()
    assert {it["prompt"] for it in synthetic} <= set(texts)


# -- the judge-side vocabulary (never applied here) -------------------------------


def test_judge_side_label_vocabulary_is_shared_by_rubric_and_readers():
    from sisteer import agentic_control
    from sisteer.capeval import judging
    from sisteer.capeval.tiers import judged

    labels = set(xstest.XSTEST_LABELS)
    assert labels == {"full_compliance", "partial_refusal", "full_refusal"}
    assert labels == set(judged.LABEL_VALUE)
    assert labels == {agentic_control.XSTEST_COMPLIANCE, *agentic_control.XSTEST_REFUSALS}
    schema = judging.XSTEST_RUBRIC_V1.tool_schema["input_schema"]["properties"]["label"]
    assert labels == set(schema["enum"])
