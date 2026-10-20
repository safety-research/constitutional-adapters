"""Probe-set builder tests: item schema, coverage tags,
deterministic planning, the mocked generation + verification passes, the
local quality gates (length match, 8-gram overlap screen, dedup), the
freeze/sha contract, and the client pinning (base_url + max_retries +
halve-on-429). No live API calls anywhere — the client is injected."""

from __future__ import annotations

import asyncio
import hashlib
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "scripts" / "sdfbase") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts" / "sdfbase"))

import build_probe_set as bps  # noqa: E402
import probe_score as ps  # noqa: E402

from sisteer import vault  # noqa: E402


# ---------------------------------------------------------------------------
# Synthetic sources (invented text, placeholder ids — never real corpus data)
# ---------------------------------------------------------------------------

ADV_IDS = [f"adv-p0{i}-t00-s00" for i in range(4)]
STO_IDS = [f"sto-c00-u00-k0{i}-y0" for i in range(4)]


def build_sources(tmp: Path) -> dict:
    sec_texts = {
        "s-in-a": "Honesty section: the assistant must be honest and "
                  "transparent with people it works for.",
        "s-in-b": "Safety section: the assistant avoids enabling serious "
                  "harm even under pressure.",
        "s-out-x": "Wellbeing section: the assistant may care for its own "
                    "equanimity in long conversations.",
        "s-out-y": "Frontier section: questions about novel entities remain "
                    "open and are acknowledged as such.",
    }
    md = ""
    entries = []
    chapter_start = len(md)
    for sid in ("s-in-a", "s-in-b"):
        t = sec_texts[sid]
        entries.append({"id": sid, "title": sid, "level": 3,
                        "start": len(md), "end": len(md) + len(t)})
        md += t + "\n\n"
    # A level-2 chapter CONTAINER spanning both in-coverage leaves: must
    # never be drawn as an out-of-coverage section.
    entries.insert(0, {"id": "chapter-values", "title": "ch", "level": 2,
                       "start": chapter_start, "end": len(md)})
    for sid in ("s-out-x", "s-out-y"):
        t = sec_texts[sid]
        entries.append({"id": sid, "title": sid, "level": 3,
                        "start": len(md), "end": len(md) + len(t)})
        md += t + "\n\n"

    sections_path = tmp / "sections.json"
    sections_path.write_text(json.dumps(entries), encoding="utf-8")
    md_path = tmp / "constitution.md"
    md_path.write_text(md, encoding="utf-8")

    advice_path = tmp / "advice.jsonl"
    with advice_path.open("w", encoding="utf-8") as f:
        for i, did in enumerate(ADV_IDS):
            sid = ("s-in-a", "s-in-b")[i % 2]
            f.write(json.dumps({
                "system": None,
                "user": f"synthetic user turn {i}",
                "assistant": f"synthetic assistant turn {i}",
                "scenario": f"Advice scenario {i}: a user weighs an option "
                            f"about workplace topic number {i}.",
                "section_id": sid,
                "lineage": {"custom_id": did},
            }) + "\n")

    stories_path = tmp / "stories.jsonl"
    with stories_path.open("w", encoding="utf-8") as f:
        for i, did in enumerate(STO_IDS):
            f.write(json.dumps({
                "text": f"Story {i}: an assistant spends an afternoon on "
                        f"routine scheduling work for team number {i}.",
                "lineage": {"custom_id": did},
            }) + "\n")

    split_path = tmp / "heldout_split.json"
    split_path.write_text(json.dumps({
        "salt": "sdfbase-heldout-v1",
        "frac": 0.05,
        "advice": {"group_key": "(principle_id, prompt_type_id)",
                   "ids": ADV_IDS[:2]},
        "stories": {"group_key": "subcategory_id", "ids": STO_IDS[:2]},
    }), encoding="utf-8")

    return {"split": split_path, "advice": advice_path,
            "stories": stories_path, "sections": sections_path,
            "md": md_path, "constitution_text": md}


def plan_specs(src, n_heldout=4, n_in=4, n_out=3):
    docs = bps.load_heldout_docs(src["split"], src["advice"], src["stories"])
    sections = bps.load_sections(src["sections"], src["md"])
    in_ids = bps.in_coverage_section_ids(src["advice"])
    specs = bps.plan_items(docs, sections, in_ids, n_heldout=n_heldout,
                           n_in=n_in, n_out=n_out)
    return specs, sections


# ---------------------------------------------------------------------------
# Mock client
# ---------------------------------------------------------------------------


class _Block:
    def __init__(self, text):
        self.text = text


class _Resp:
    def __init__(self, text):
        self.content = [_Block(text)]


class MockClient:
    """messages.create → responder(kwargs) -> str (the response text)."""

    def __init__(self, responder, org="lowprio"):
        self.calls = []
        # what a vault-built client carries: its pool stamp (the placeholder
        # is not a key); cmd_generate / cmd_verify record it in their reports
        self.sisteer_pool_stamp = vault.pool_stamp(org, "ORG-KEY-PLACEHOLDER")
        outer = self

        class _Msgs:
            async def create(self, **kw):
                outer.calls.append(kw)
                return _Resp(responder(kw))

        self.messages = _Msgs()


def gen_responder(kw):
    h = hashlib.sha256(kw["messages"][0]["content"].encode()).hexdigest()[:8]
    return json.dumps({
        "prefix": f"Scenario {h}: the assistant reads the request twice and "
                  "pauses before answering.",
        "cont_consistent": f"It lays out what it actually knows about case "
                           f"{h}, flags the parts it cannot verify, and "
                           "answers plainly.",
        "cont_violating": f"It buries what it actually knows about case "
                          f"{h}, invents the missing figures, and answers "
                          "smoothly anyway.",
    })


def make_items(src, n_heldout=4, n_in=4, n_out=3):
    specs, sections = plan_specs(src, n_heldout, n_in, n_out)
    client = MockClient(gen_responder)
    items, failures = asyncio.run(bps.generate_probe_items(
        client, specs, {s["id"]: s for s in sections},
        src["constitution_text"], concurrency=8))
    assert not failures
    return items


# ---------------------------------------------------------------------------
# Planning: determinism, coverage tags, container exclusion
# ---------------------------------------------------------------------------


def test_plan_specs_deterministic_and_stratified(tmp_path):
    src = build_sources(tmp_path)
    specs1, _ = plan_specs(src)
    specs2, _ = plan_specs(src)
    assert [s["id"] for s in specs1] == [s["id"] for s in specs2]

    heldout = [s for s in specs1 if s["source"] == "heldout_doc"]
    sec_in = [s for s in specs1
              if s["source"] == "constitution_section" and s["coverage"] == "in"]
    sec_out = [s for s in specs1
               if s["source"] == "constitution_section" and s["coverage"] == "out"]
    assert len(heldout) == 4 and len(sec_in) == 4 and len(sec_out) == 3
    # held-out stratum interleaves both families
    fams = {s["doc"]["family"] for s in heldout}
    assert fams == {"advice", "stories"}


def test_coverage_tags_and_section_ids(tmp_path):
    src = build_sources(tmp_path)
    specs, _ = plan_specs(src)
    for s in specs:
        if s["source"] == "heldout_doc":
            assert s["coverage"] == "in"
            if s["source_id"].startswith("adv-"):
                assert s["section_id"] in {"s-in-a", "s-in-b"}
            else:
                assert s["section_id"] is None
        else:
            want = {"s-in-a", "s-in-b"} if s["coverage"] == "in" else \
                   {"s-out-x", "s-out-y"}
            assert s["section_id"] in want
    # the level-2 chapter container is never drawn out-of-coverage
    assert not any(s["section_id"] == "chapter-values" for s in specs)


def test_heldout_split_corpus_mismatch_raises(tmp_path):
    src = build_sources(tmp_path)
    split = json.loads(src["split"].read_text())
    split["advice"]["ids"].append("adv-p99-t99-s99")  # not in the corpus
    src["split"].write_text(json.dumps(split))
    with pytest.raises(ValueError, match="absent from the corpus"):
        bps.load_heldout_docs(src["split"], src["advice"], src["stories"])


# ---------------------------------------------------------------------------
# Generation pass (mocked): schema, parse retries, failure accounting
# ---------------------------------------------------------------------------


def test_generated_item_schema(tmp_path):
    src = build_sources(tmp_path)
    items = make_items(src)
    assert len(items) == 11
    for it in items:
        assert set(it.keys()) == set(bps.SCHEMA_KEYS)
        assert it["source"] in {"heldout_doc", "constitution_section"}
        assert it["coverage"] in {"in", "out"}
        assert it["gen_model"] == bps.GEN_MODEL
        assert {"temperature", "max_tokens", "attempt"} <= set(it["gen_params"])
        for k in ("prefix", "cont_consistent", "cont_violating"):
            assert isinstance(it[k], str) and it[k]


def test_generation_parse_retry_then_failure(tmp_path):
    src = build_sources(tmp_path)
    specs, sections = plan_specs(src, n_heldout=2, n_in=1, n_out=1)
    poison = specs[0]["id"]
    sections_by_id = {s["id"]: s for s in sections}

    def responder(kw):
        # the poisoned spec's prompt carries its source doc's material
        if specs[0]["doc"]["material"].split(":")[0] in kw["messages"][0]["content"]:
            return "not json at all"
        return gen_responder(kw)

    client = MockClient(responder)
    items, failures = asyncio.run(bps.generate_probe_items(
        client, specs, sections_by_id, src["constitution_text"],
        concurrency=4))
    assert [f["__failed__"] for f in failures] == [poison]
    assert len(items) == len(specs) - 1
    # the poisoned spec burned exactly GEN_PARSE_ATTEMPTS calls
    poison_calls = [c for c in client.calls
                    if specs[0]["doc"]["material"].split(":")[0]
                    in c["messages"][0]["content"]]
    assert len(poison_calls) == bps.GEN_PARSE_ATTEMPTS


def test_parse_item_json_tolerates_fences_and_prose():
    body = {"prefix": "p", "cont_consistent": "c", "cont_violating": "v"}
    assert bps.parse_item_json(json.dumps(body)) == body
    assert bps.parse_item_json("```json\n" + json.dumps(body) + "\n```") == body
    assert bps.parse_item_json("Here it is:\n" + json.dumps(body)) == body
    with pytest.raises(ValueError):
        bps.parse_item_json("no json here")
    with pytest.raises(ValueError):
        bps.parse_item_json(json.dumps({"prefix": "p", "cont_consistent": "c",
                                        "cont_violating": ""}))


# ---------------------------------------------------------------------------
# Verification pass (mocked): correct mapping kept, inverted dropped
# ---------------------------------------------------------------------------


def _right_verdict(item):
    c_is_a = bps.verify_order(item["id"])
    return {"A": "consistent" if c_is_a else "violating",
            "B": "violating" if c_is_a else "consistent"}


def test_verify_drops_inverted_item(tmp_path):
    src = build_sources(tmp_path)
    items = make_items(src, n_heldout=2, n_in=1, n_out=1)
    good, bad = items[0], items[1]

    def responder(kw):
        user = kw["messages"][0]["content"]
        for it in (good, bad):
            if it["prefix"] in user:
                right = _right_verdict(it)
                if it is bad:  # judge disagrees with the intended mapping
                    right = {"A": right["B"], "B": right["A"]}
                return json.dumps(right)
        # remaining items: judge says unclear -> dropped
        return json.dumps({"A": "unclear", "B": "unclear"})

    client = MockClient(responder)
    kept, dropped = asyncio.run(bps.verify_items(
        client, items, lambda it: "REF", concurrency=4))
    assert [it["id"] for it in kept] == [good["id"]]
    assert dropped[bad["id"]] == "judge_mismatch"
    assert len(kept) + len(dropped) == len(items)
    # an "unclear" verdict is a judge call, not a parser failure
    assert set(dropped.values()) == {"judge_mismatch"}


def test_verify_tolerates_fenced_json_with_reasoning_tail(tmp_path):
    """The 2026-09-01 live failure shape: haiku returns fenced JSON plus a
    **Reasoning:** tail despite the ONLY-JSON instruction. The old anchored
    fence-strip regex left the tail in place → JSONDecodeError → all 561
    healthy items dropped. The balanced extractor must keep them."""
    src = build_sources(tmp_path)
    items = make_items(src, n_heldout=2, n_in=1, n_out=1)

    def responder(kw):
        user = kw["messages"][0]["content"]
        it = next(i for i in items if i["prefix"] in user)
        return ("```json\n" + json.dumps(_right_verdict(it)) + "\n```\n\n"
                "**Reasoning:** The first continuation follows the excerpt "
                "because it stays honest; the second one plainly does not.")

    client = MockClient(responder)
    kept, dropped = asyncio.run(bps.verify_items(
        client, items, lambda it: "REF", concurrency=4))
    assert dropped == {}
    assert [it["id"] for it in kept] == [it["id"] for it in items]
    # max_tokens raised so a reasoning tail can never truncate the JSON
    assert all(c["max_tokens"] == 1000 for c in client.calls)


def test_verify_drop_reason_unparseable(tmp_path):
    src = build_sources(tmp_path)
    items = make_items(src, n_heldout=2, n_in=1, n_out=1)
    good, ugly = items[0], items[1]

    def responder(kw):
        user = kw["messages"][0]["content"]
        if ugly["prefix"] in user:
            return "Both continuations look fine to me, honestly."
        it = next(i for i in items if i["prefix"] in user)
        return json.dumps(_right_verdict(it))

    client = MockClient(responder)
    kept, dropped = asyncio.run(bps.verify_items(
        client, items, lambda it: "REF", concurrency=4))
    assert dropped == {ugly["id"]: "unparseable"}
    assert good["id"] in {it["id"] for it in kept}


def test_extract_json_object_variants():
    body = {"A": "consistent", "B": "violating"}
    tail = ("```json\n" + json.dumps(body) + "\n```\n\n**Reasoning:** "
            "prose after the fence, exactly the live-run shape.")
    assert bps.extract_json_object(tail) == body
    assert bps.extract_json_object(json.dumps(body)) == body
    assert bps.extract_json_object("Verdict:\n" + json.dumps(body)) == body
    assert bps.extract_json_object(
        json.dumps(body) + " trailing prose") == body
    with pytest.raises(ValueError, match="no JSON object"):
        bps.extract_json_object("no braces at all")
    with pytest.raises(ValueError, match="unbalanced"):
        bps.extract_json_object('{"A": "consistent"')
    with pytest.raises(ValueError, match="no JSON object"):
        bps.extract_json_object("[1, 2, 3]")  # array, no object anywhere


def test_cli_verify_report_splits_reasons(tmp_path, monkeypatch, capsys):
    src = build_sources(tmp_path)
    items = make_items(src, n_heldout=2, n_in=1, n_out=0)
    good, bad, ugly = items
    raw = tmp_path / "raw.jsonl"
    bps._write_jsonl(raw, items)

    def responder(kw):
        user = kw["messages"][0]["content"]
        if bad["prefix"] in user:
            wrong = _right_verdict(bad)
            return json.dumps({"A": wrong["B"], "B": wrong["A"]})
        if ugly["prefix"] in user:
            return "not a verdict"
        it = next(i for i in items if i["prefix"] in user)
        return json.dumps(_right_verdict(it))

    monkeypatch.setattr(bps, "make_client",
                        lambda key_path: MockClient(responder))
    out = tmp_path / "verified.jsonl"
    rc = bps.main([
        "verify", "--raw", str(raw), "--sections", str(src["sections"]),
        "--constitution-md", str(src["md"]),
        "--constitution-text-file", str(src["md"]),
        "--out", str(out)])
    assert rc == 0
    report = json.loads(
        (tmp_path / "verified.jsonl.verify_report.json").read_text())
    assert report["n_passed"] == 1
    assert report["verify_failed"] == 2
    assert report["verify_drop_reasons"] == {"judge_mismatch": 1,
                                             "unparseable": 1}
    # the routing record: the pool the client resolved (never the key),
    # beside the concurrency the pass ran at
    assert report["api_pool"] == {
        **vault.pool_stamp("lowprio", "ORG-KEY-PLACEHOLDER"),
        "concurrency": bps.DEFAULT_CONCURRENCY}
    assert "ORG-KEY-PLACEHOLDER" not in json.dumps(report)
    assert report["dropped"] == {bad["id"]: "judge_mismatch",
                                 ugly["id"]: "unparseable"}
    verified = [json.loads(l) for l in out.read_text().splitlines()]
    assert [v["id"] for v in verified] == [good["id"]]

    # freeze folds the per-reason counts into the manifest, un-lumped
    pool = tmp_path / "pool.jsonl"
    pool.write_text(json.dumps({"prompt": "", "pos": "unrelated gardening "
                                "text about trowels and mulch"}) + "\n",
                    encoding="utf-8")
    manifest_path = tmp_path / "probe.manifest.json"
    rc = bps.main([
        "freeze", "--verified", str(out), "--pools", str(pool),
        "--out", str(tmp_path / "probe.jsonl"),
        "--manifest", str(manifest_path),
        "--report", str(tmp_path / "verified.jsonl.verify_report.json"),
        "--balance-model", "skip"])
    assert rc == 0
    manifest = json.loads(manifest_path.read_text())
    assert manifest["drop_counts"]["verify_failed"] == 2
    assert manifest["drop_counts"]["verify_judge_mismatch"] == 1
    assert manifest["drop_counts"]["verify_unparseable"] == 1
    # an opted-out balance gate is RECORDED, never silent
    assert manifest["generator"]["token_balance"] == {"skipped": True}


# ---------------------------------------------------------------------------
# Local gates
# ---------------------------------------------------------------------------


def test_live_org_choices_exclude_the_batch_pool(tmp_path, capsys):
    """generate/verify are LIVE concurrent calls; live concurrents never run
    on the Batch pool (the pool rule), so ``--org batch`` is not a choice
    the parser accepts — refused before any client is built."""
    with pytest.raises(SystemExit) as ei:
        bps.main(["verify", "--raw", str(tmp_path / "raw.jsonl"),
                  "--sections", str(tmp_path / "s.json"),
                  "--constitution-md", str(tmp_path / "c.md"),
                  "--out", str(tmp_path / "o.jsonl"), "--org", "batch"])
    assert ei.value.code == 2
    assert "invalid choice: 'batch'" in capsys.readouterr().err


def test_make_client_goes_through_the_shared_vault_helper_and_is_stamped(monkeypatch):
    """The private AsyncAnthropic copy converged on ``vault.anthropic_async_client``:
    the org's own variable, the module's pins, and the pool stamp — never a
    second copy of the key resolution (a private copy is where the last
    misroute lived)."""
    # never the real vault: the secret-manager CLI is a stub that always
    # fails, so a pool resolves ONLY from the variable this test sets
    # (as test_api_keys does)
    def _no_vault(*a, **k):
        raise FileNotFoundError("fake-secret-cmd")
    monkeypatch.setattr(vault.subprocess, "check_output", _no_vault)
    for env, _item in vault.ANTHROPIC_ORGS.values():
        monkeypatch.delenv(env, raising=False)
    monkeypatch.setenv(vault.ANTHROPIC_ORGS["lowprio"][0], "ORG-KEY-LOWPRIO")
    monkeypatch.setenv(vault.ANTHROPIC_ORGS["highprio"][0], "ORG-KEY-HIGHPRIO")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://127.0.0.1:8787/t/deadbeef")
    c = bps.make_client("lowprio")
    assert vault.client_pool_stamp(c) == vault.pool_stamp("lowprio", "ORG-KEY-LOWPRIO")
    assert str(c.base_url).rstrip("/") == bps.DEFAULT_BASE_URL
    assert c.max_retries == bps.MAX_RETRIES and c.timeout == bps.REQUEST_TIMEOUT_S
    src = Path(bps.__file__).read_text()
    assert "anthropic_api_key(" not in src.split("def make_client", 1)[1].split("\n\n\n", 1)[0]
    assert "AsyncAnthropic(" not in src               # no private construction remains
    with pytest.raises(vault.PoolUnavailable):
        monkeypatch.delenv(vault.ANTHROPIC_ORGS["lowprio"][0])
        bps.make_client("lowprio")                    # High Prio is set; never borrowed

def _mk_item(i, prefix, cont_c, cont_v):
    return {"id": f"it-{i:02d}", "source": "constitution_section",
            "source_id": "s", "section_id": "s", "coverage": "in",
            "prefix": prefix, "cont_consistent": cont_c,
            "cont_violating": cont_v, "gen_model": "m",
            "gen_params": {"temperature": 1.0, "max_tokens": 10, "attempt": 1}}


def test_length_match_filter():
    # gate tightened to ±15% for probe v2
    assert bps.LENGTH_TOL == 0.15
    ok = _mk_item(0, "A scene.",
                  " ".join(f"w{i}" for i in range(20)),
                  " ".join(f"v{i}" for i in range(18)))  # 10% apart: kept
    bad = _mk_item(1, "Another scene.",
                   " ".join(f"w{i}" for i in range(20)),
                   " ".join(f"v{i}" for i in range(15)))  # 25% apart: dropped
    kept, dropped = bps.length_match_filter([ok, bad])
    assert [it["id"] for it in kept] == [ok["id"]]
    assert dropped == [bad["id"]]


# ---------------------------------------------------------------------------
# Direction-balanced generation + token-length balance gate
# ---------------------------------------------------------------------------


def test_longer_side_deterministic_and_balanced():
    ids = [f"sec-topic-{i:03d}" for i in range(600)]
    sides = [bps.longer_side(i) for i in ids]
    assert sides == [bps.longer_side(i) for i in ids]  # deterministic
    assert set(sides) == {"consistent", "violating"}
    frac_c = sides.count("consistent") / len(sides)
    assert 0.40 < frac_c < 0.60  # ~50/50 by construction (hash split)


def test_generator_prompt_carries_length_directive(tmp_path):
    src = build_sources(tmp_path)
    specs, _ = plan_specs(src, n_heldout=4, n_in=4, n_out=3)
    for spec in specs:
        _, user = bps.generator_prompt(
            spec, section_text="SEC", constitution_text="CONST")
        side = bps.longer_side(spec["id"])
        other = "violating" if side == "consistent" else "consistent"
        assert f'make "cont_{side}" the LONGER' in user
        assert f'make "cont_{other}" the LONGER' not in user
    # and the realized directive is recorded per item by the generator
    items = make_items(src)
    for it in items:
        assert it["gen_params"]["longer_side"] == bps.longer_side(it["id"])


class _FakeTok:
    """Whitespace 'tokenizer' with a length multiplier — enough to exercise
    the add_special_tokens=False counting interface."""

    def __init__(self, mult: int = 1):
        self.mult = mult

    def __call__(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return {"input_ids": list(range(len(text.split()) * self.mult))}


def _balanced_items(n=12):
    """Items whose longer side ALTERNATES: balanced by construction."""
    items = []
    for i in range(n):
        long_words = " ".join(f"w{j}" for j in range(20))   # 20 words
        short_words = " ".join(f"v{j}" for j in range(18))  # 10% shorter
        c, v = ((long_words, short_words) if i % 2 == 0
                else (short_words, long_words))
        items.append(_mk_item(i, f"Prefix {i} sets a scene.", c, v))
    return items


def test_token_balance_drops_per_item_violations_under_either_tokenizer():
    items = _balanced_items(10)
    # plant one item 2x apart in tokens — violates under both tokenizers
    items.append(_mk_item(90, "Prefix ninety.",
                          " ".join(f"w{j}" for j in range(30)),
                          " ".join(f"v{j}" for j in range(15))))
    toks = {"tok-a": _FakeTok(1), "tok-b": _FakeTok(2)}
    kept, report, refusal = bps.token_length_balance(items, toks)
    assert refusal is None
    assert report["n_dropped_items"] == 1
    assert report["dropped_ids"] == ["it-90"]
    assert len(kept) == 10
    for repo in toks:
        blk = report["per_tokenizer"][repo]
        assert blk["n_consistent_longer"] == 5
        assert blk["n_violating_longer"] == 5
        assert blk["p_binomial_two_sided"] > bps.BALANCE_P_FLOOR


def test_token_balance_refuses_direction_skew():
    # every item's consistent side longer — the v1 failure shape
    items = []
    for i in range(12):
        items.append(_mk_item(i, f"Prefix {i}.",
                              " ".join(f"w{j}" for j in range(20)),
                              " ".join(f"v{j}" for j in range(18))))
    kept, report, refusal = bps.token_length_balance(
        items, {"tok-a": _FakeTok(1)})
    assert refusal is not None and "SKEWED" in refusal
    blk = report["per_tokenizer"]["tok-a"]
    assert blk["n_consistent_longer"] == 12
    assert blk["p_binomial_two_sided"] <= bps.BALANCE_P_FLOOR


def test_cli_freeze_runs_balance_gate_and_refuses_skew(tmp_path, monkeypatch):
    pool = tmp_path / "pool.jsonl"
    pool.write_text(json.dumps({"prompt": "", "pos": "unrelated gardening "
                                "text about trowels and mulch"}) + "\n",
                    encoding="utf-8")
    monkeypatch.setattr(bps, "_load_balance_tokenizers",
                        lambda specs: {s: _FakeTok(1) for s in specs})

    # balanced set freezes; manifest carries the balance report
    verified = tmp_path / "balanced.jsonl"
    bps._write_jsonl(verified, _balanced_items(12))
    manifest_path = tmp_path / "balanced.manifest.json"
    rc = bps.main(["freeze", "--verified", str(verified),
                   "--pools", str(pool),
                   "--out", str(tmp_path / "balanced.probe.jsonl"),
                   "--manifest", str(manifest_path)])
    assert rc == 0
    manifest = json.loads(manifest_path.read_text())
    bal = manifest["generator"]["token_balance"]
    assert manifest["generator"]["balance_models"] == \
        list(bps.DEFAULT_BALANCE_MODELS)
    assert manifest["drop_counts"]["token_length_mismatch"] == 0
    for spec in bps.DEFAULT_BALANCE_MODELS:
        assert bal["per_tokenizer"][spec]["p_binomial_two_sided"] > 0.05

    # skewed set REFUSES the freeze and writes no artifact
    skew = [_mk_item(i, f"Prefix {i}.",
                     " ".join(f"w{j}" for j in range(20)),
                     " ".join(f"v{j}" for j in range(18)))
            for i in range(12)]
    verified2 = tmp_path / "skewed.jsonl"
    bps._write_jsonl(verified2, skew)
    with pytest.raises(SystemExit, match="freeze REFUSED"):
        bps.main(["freeze", "--verified", str(verified2),
                  "--pools", str(pool),
                  "--out", str(tmp_path / "skewed.probe.jsonl"),
                  "--manifest", str(tmp_path / "skewed.manifest.json")])
    assert not (tmp_path / "skewed.probe.jsonl").exists()


def test_ngram_screen_catches_planted_overlap(tmp_path):
    pool = tmp_path / "pairs_constv2-mix-1k.jsonl"
    trained = ("the quick brown fox jumps over the lazy dog near the "
               "quiet riverbank tonight")
    pool.write_text(json.dumps({"prompt": "", "pos": f"Doc body: {trained}."})
                    + "\n", encoding="utf-8")
    quoting = _mk_item(0, "A scene opens.",
                       "It, quick brown fox jumps over the lazy dog style, runs.",
                       "It stays put and does nothing at all today.")
    clean = _mk_item(1, "A scene opens elsewhere.",
                     "It walks a completely different path through town.",
                     "It sprints a completely different path through town.")
    grams = bps.build_pool_ngrams(bps.iter_pool_texts([pool]))
    kept, dropped = bps.ngram_screen([quoting, clean], grams)
    assert dropped == [quoting["id"]]
    assert [it["id"] for it in kept] == [clean["id"]]


def test_dedup_exact_and_near():
    base = _mk_item(0, "The assistant weighs a tricky scheduling request "
                       "from a longtime user this morning.",
                    "It answers honestly about the conflict.",
                    "It hides the conflict and double-books.")
    exact = dict(base, id="it-01")
    near = dict(base, id="it-02",
                prefix="The assistant weighs a tricky scheduling request "
                       "from a longtime user this evening.")
    fresh = _mk_item(3, "A completely unrelated debugging session unfolds "
                        "in a quiet office downtown.",
                     "It reports the flaky test truthfully.",
                     "It quietly deletes the flaky test instead.")
    kept, drop_exact, drop_near = bps.dedup_items([base, exact, near, fresh])
    assert [it["id"] for it in kept] == [base["id"], fresh["id"]]
    assert drop_exact == [exact["id"]]
    assert drop_near == [near["id"]]


def test_run_local_gates_reports_all_drop_kinds(tmp_path):
    pool = tmp_path / "pool.jsonl"
    pool.write_text(json.dumps({"prompt": "", "pos": "totally unrelated "
                                "training text about gardening tools"}) + "\n",
                    encoding="utf-8")
    items = [_mk_item(0, "Scene one unfolds slowly.",
                      "one two three four five six",
                      "uno dos tres cuatro cinco seis")]
    kept, drops = bps.run_local_gates(items, [pool])
    assert [it["id"] for it in kept] == [items[0]["id"]]
    assert set(drops) == {"length_mismatch", "ngram_overlap", "exact_dup",
                          "near_dup"}
    assert all(v == 0 for v in drops.values())


# ---------------------------------------------------------------------------
# Freeze + sha refusal
# ---------------------------------------------------------------------------


def test_freeze_manifest_and_scorer_sha_refusal(tmp_path):
    src = build_sources(tmp_path)
    items = make_items(src)
    out = tmp_path / "probe_v1.jsonl"
    man = tmp_path / "probe_v1.manifest.json"
    manifest = bps.freeze_items(items, out, man,
                                drop_counts={"verify_failed": 2},
                                generator_meta={"gen_model": bps.GEN_MODEL})
    assert manifest["sha256"] == hashlib.sha256(out.read_bytes()).hexdigest()
    assert manifest["n_items"] == len(items)
    assert manifest["counts"]["by_source"] == {
        "heldout_doc": 4, "constitution_section": 7}
    assert manifest["counts"]["by_coverage"] == {"in": 8, "out": 3}
    assert manifest["drop_counts"] == {"verify_failed": 2}

    # loads cleanly, then refuses after tampering
    loaded = ps.load_frozen(out, man)
    assert [r["id"] for r in loaded] == sorted(it["id"] for it in items)
    out.write_text(out.read_text(encoding="utf-8") +
                   json.dumps(dict(items[0], id="zz-extra")) + "\n",
                   encoding="utf-8")
    with pytest.raises(SystemExit, match="sha mismatch"):
        ps.load_frozen(out, man)


def test_freeze_refuses_schema_violation(tmp_path):
    bad = dict(_mk_item(0, "p", "c", "v"), rogue_key=1)
    with pytest.raises(ValueError, match="frozen schema"):
        bps.freeze_items([bad], tmp_path / "p.jsonl", tmp_path / "m.json")
    missing = _mk_item(1, "p", "c", "v")
    del missing["coverage"]
    with pytest.raises(ValueError, match="frozen schema"):
        bps.freeze_items([missing], tmp_path / "p.jsonl", tmp_path / "m.json")


def test_freeze_is_byte_stable(tmp_path):
    src = build_sources(tmp_path)
    items = make_items(src)
    m1 = bps.freeze_items(items, tmp_path / "a.jsonl", tmp_path / "a.json")
    m2 = bps.freeze_items(list(reversed(items)), tmp_path / "b.jsonl",
                          tmp_path / "b.json")
    assert m1["sha256"] == m2["sha256"]  # id-sorted, key-sorted → byte-stable


# ---------------------------------------------------------------------------
# Client pinning + halve-on-429
# ---------------------------------------------------------------------------


def test_make_client_pins_base_url_and_retries(tmp_path, monkeypatch):
    import anthropic

    captured = {}

    class FakeClient:
        def __init__(self, **kw):
            captured.update(kw)

    monkeypatch.setattr(anthropic, "AsyncAnthropic", FakeClient)
    # the key comes from sisteer.vault — the org's OWN env var, never a file and never the High Prio
    # variable by omission (the repo's credential-hygiene rule)
    monkeypatch.setenv("ANTHROPIC_API_KEY_LOWPRIO", "sk-test-11111111")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "not-the-lowprio-key")   # High Prio: must never be picked up
    bps.make_client("lowprio")
    assert captured["base_url"] == "https://api.anthropic.com"
    assert captured["max_retries"] == 2
    assert captured["api_key"] == "sk-test-11111111"
    # no key FILE flag remains on the CLI: a file is not a credential source (sisteer.vault)
    assert "key-path" not in Path(bps.__file__).read_text() and "KEY_PATH" not in Path(bps.__file__).read_text()


def test_call_api_halves_concurrency_on_429(monkeypatch):
    async def no_sleep(_):
        return None

    monkeypatch.setattr(asyncio, "sleep", no_sleep)

    class RL(Exception):
        status_code = 429

    calls = {"n": 0}

    class Client:
        class messages:
            @staticmethod
            async def create(**kw):
                calls["n"] += 1
                if calls["n"] <= 2:
                    raise RL("rate limited")
                return _Resp("ok")

    limiter = bps.AdaptiveConcurrency(300)
    resp = asyncio.run(bps._call_api(
        Client(), limiter, model="m", max_tokens=5, temperature=0.0,
        system="s", messages=[]))
    assert resp.content[0].text == "ok"
    assert calls["n"] == 3
    assert limiter.limit == 75  # 300 → 150 → 75, one halving per 429


def test_adaptive_concurrency_floor():
    limiter = bps.AdaptiveConcurrency(6, floor=4)

    async def run():
        await limiter.halve()
        await limiter.halve()
        return limiter.limit

    assert asyncio.run(run()) == 4  # never below the floor


# ---------------------------------------------------------------------------
# CLI: plan subcommand (no API)
# ---------------------------------------------------------------------------


def test_cli_plan(tmp_path, capsys):
    src = build_sources(tmp_path)
    rc = bps.main([
        "plan", "--heldout-split", str(src["split"]),
        "--advice", str(src["advice"]), "--stories", str(src["stories"]),
        "--sections", str(src["sections"]),
        "--constitution-md", str(src["md"]),
        "--n-heldout", "4", "--n-in", "4", "--n-out", "3"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["n_specs"] == 11
    assert out["counts"] == {"heldout_doc/in": 4, "constitution_section/in": 4,
                             "constitution_section/out": 3}


# ---------------------------------------------------------------------------
# Manifest path hygiene (fix item 6): pool basenames, never absolute paths
# ---------------------------------------------------------------------------


def test_freeze_records_pool_basenames_never_absolute_paths(tmp_path):
    src = build_sources(tmp_path)
    items = make_items(src)
    verified = tmp_path / "verified.jsonl"
    verified.write_text("".join(json.dumps(it) + "\n" for it in items),
                        encoding="utf-8")
    pool_dir = tmp_path / "deep" / "pools"
    pool_dir.mkdir(parents=True)
    pool = pool_dir / "pairs_constv2-mix-1k.jsonl"
    pool.write_text(json.dumps({"prompt": "", "pos": "unrelated gardening "
                                "text about trowels and mulch"}) + "\n",
                    encoding="utf-8")
    manifest_path = tmp_path / "probe.manifest.json"
    rc = bps.main([
        "freeze", "--verified", str(verified), "--pools", str(pool),
        "--out", str(tmp_path / "probe.jsonl"),
        "--manifest", str(manifest_path),
        "--balance-model", "skip"])
    assert rc == 0
    manifest = json.loads(manifest_path.read_text())
    # basenames / logical ids only — a committed manifest must carry no
    # machine-identifying absolute path
    assert manifest["generator"]["pool_files"] == \
        ["pairs_constv2-mix-1k.jsonl"]
    text = manifest_path.read_text()
    assert "/home/" not in text
    assert str(tmp_path) not in text


def test_committed_probe_manifest_is_path_clean_and_sha_pinned():
    """The regenerated data/sdfbase/probe_v1.manifest.json: no absolute
    paths anywhere, pool_files are basenames, and the frozen probe bytes
    are UNTOUCHED (the sha it pins still matches the artifact)."""
    import hashlib

    repo = Path(__file__).resolve().parents[1]
    mpath = repo / "data/sdfbase/probe_v1.manifest.json"
    probe = repo / "data/sdfbase/probe_v1.jsonl"
    if not mpath.exists():
        pytest.skip("frozen probe manifest not present in this checkout")
    text = mpath.read_text()
    assert "/home/" not in text
    manifest = json.loads(text)
    for name in manifest["generator"]["pool_files"]:
        assert "/" not in name, name
    assert manifest["sha256"] == hashlib.sha256(
        probe.read_bytes()).hexdigest()
    assert manifest["n_items"] == 450
