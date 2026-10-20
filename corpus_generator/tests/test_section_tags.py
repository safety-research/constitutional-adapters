"""Section-tagging plumbing (section_tags.py): the frozen index and its
validation, lenient-then-bounded JSON repair, the tagging run's output rows
and manifest, resumability through the live engine's per-request cache, and
the vault fallback of the client key reader."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from conftest import TINY_CONSTITUTION, tiny_sections
from corpus_generator import section_tags as st
from corpus_generator.constitution import Section
from corpus_generator.io import iter_jsonl, read_json
from corpus_generator.live import LiveEngine
from corpus_generator.templates import TemplateStore, repo_root
from fakes import FakeLiveClient, live_message

TPL = TemplateStore(repo_root() / "prompts")
IDS = ["being-honest", "avoiding-harm", "healthy-boundaries"]


# ------------------------------------------------------------- fixtures --


def _snapshot_sha(md: str) -> str:
    import hashlib

    return hashlib.sha256(md.encode("utf-8")).hexdigest()


def write_constitution(cdir: Path, md: str = TINY_CONSTITUTION, sections=None) -> None:
    sections = sections or tiny_sections()
    cdir.mkdir(parents=True, exist_ok=True)
    (cdir / "constitution.md").write_text(md, encoding="utf-8")
    (cdir / "sections.json").write_text(
        json.dumps([s.__dict__ for s in sections]), encoding="utf-8"
    )
    (cdir / "SNAPSHOT.json").write_text(
        json.dumps({"sha256": _snapshot_sha(md), "n_sections": len(sections)}), encoding="utf-8"
    )


def write_index(cdir: Path, summaries: dict[str, str] | None = None, **overrides) -> dict:
    sections = tiny_sections()
    tree = st.section_tree(sections)
    entries = [
        {"id": e.id, "title": e.title, "level": e.level, "parent": e.parent, "chapter": e.chapter,
         "summary": (summaries or {}).get(e.id, f"About {e.title.lower()}.")}
        for e in tree
    ]
    data = {
        "constitution_sha256": _snapshot_sha(TINY_CONSTITUTION),
        "n_sections": len(entries),
        "model": "fake",
        "sha256": st.summaries_sha(entries),
        "sections": entries,
    }
    data.update(overrides)
    (cdir / st.SUMMARIES_FILE).write_text(json.dumps(data), encoding="utf-8")
    return data


@pytest.fixture
def cdir(tmp_path) -> Path:
    d = tmp_path / "constitution"
    write_constitution(d)
    write_index(d)
    return d


def story(i: int) -> dict:
    return {"text": f"Story {i}: the assistant helped.", "lineage": {"custom_id": f"sto-c00-u00-k{i:02d}-y0"}}


def advice(i: int, section: str = "being-honest") -> dict:
    return {"user": f"Question {i}?", "assistant": f"Answer {i}.", "section_id": section,
            "lineage": {"custom_id": f"adv-p00-t00-s{i:02d}"}}


def write_final(path: Path, records: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    return path


async def _noop(_s: float) -> None:
    return None


def engine(client, **kw) -> LiveEngine:
    kw.setdefault("sleep", _noop)
    kw.setdefault("max_concurrency", 4)
    kw.setdefault("min_concurrency", 1)
    kw.setdefault("retry_rounds", 0)
    return LiveEngine(lambda: client, **kw)


def tag_json(primary, secondary=(), none=False, confidence=0.9) -> str:
    return json.dumps({"primary": list(primary), "secondary": list(secondary),
                       "none": none, "confidence": confidence})


def doc_index(params: dict) -> int:
    """Recover the record number from the rendered user turn."""
    import re

    m = re.search(r"(?:Story|Question) (\d+)", params["messages"][0]["content"])
    return int(m.group(1))


def make_run(tmp_path, cdir, corpus="stories", records=None, **kw) -> st.TagRun:
    records = records if records is not None else [story(i) for i in range(5)]
    src = write_final(tmp_path / "data" / corpus / "final-rel.jsonl", records)
    return st.TagRun(
        corpus=corpus, release="rel", source=src, out_dir=tmp_path / "data" / corpus,
        wave_root=tmp_path / "data" / "runs" / "section-tags" / "rel",
        constitution_dir=cdir, prompts_dir=repo_root() / "prompts", **kw,
    )


# ------------------------------------------------------------ the index --


def test_section_tree_parent_and_chapter_from_heading_order():
    secs = [
        Section("ov", "Overview", 2, 0, 10),
        Section("a", "A", 3, 1, 5),
        Section("a1", "A1", 4, 2, 4),
        Section("b", "B", 3, 5, 10),
        Section("ch2", "Chapter 2", 2, 10, 20),
        Section("c", "C", 3, 11, 20),
    ]
    tree = {e.id: e for e in st.section_tree(secs)}
    assert tree["ov"].parent is None and tree["ov"].chapter == "ov"
    assert tree["a"].parent == "ov" and tree["a"].chapter == "ov"
    assert tree["a1"].parent == "a" and tree["a1"].chapter == "ov"
    assert tree["b"].parent == "ov"
    assert tree["ch2"].parent is None and tree["ch2"].chapter == "ch2"
    assert tree["c"].parent == "ch2" and tree["c"].chapter == "ch2"


def test_load_index_happy_path_and_render(cdir):
    entries, meta = load = st.load_index(cdir)
    assert [e.id for e in entries] == IDS
    assert all(e.summary for e in entries)
    text = st.render_index(entries)
    assert text.count("\n") == 2
    assert "being-honest (chapter) — Being honest: About being honest." in text


def test_load_index_refuses_missing_file(tmp_path):
    d = tmp_path / "c"
    write_constitution(d)
    with pytest.raises(st.SectionIndexError, match="missing"):
        st.load_index(d)


def test_load_index_refuses_another_constitution(cdir):
    write_index(cdir, constitution_sha256="0" * 64)
    with pytest.raises(st.SectionIndexError, match="written from constitution"):
        st.load_index(cdir)


def test_load_index_refuses_edited_summaries(cdir):
    data = read_json(cdir / st.SUMMARIES_FILE)
    data["sections"][0]["summary"] = "quietly edited after freezing"
    (cdir / st.SUMMARIES_FILE).write_text(json.dumps(data))
    with pytest.raises(st.SectionIndexError, match="edited after freezing"):
        st.load_index(cdir)


def test_load_index_refuses_section_list_drift(cdir):
    data = read_json(cdir / st.SUMMARIES_FILE)
    data["sections"] = data["sections"][:2]
    data["sha256"] = st.summaries_sha(data["sections"])
    (cdir / st.SUMMARIES_FILE).write_text(json.dumps(data))
    with pytest.raises(st.SectionIndexError, match="lists 2 sections"):
        st.load_index(cdir)


def test_load_index_refuses_empty_summary(cdir):
    write_index(cdir, summaries={"avoiding-harm": "   "})
    with pytest.raises(st.SectionIndexError, match="empty summary"):
        st.load_index(cdir)


def test_freeze_summaries_writes_a_loadable_frozen_index(tmp_path):
    d = tmp_path / "c"
    write_constitution(d)
    sections = tiny_sections()
    snapshot = {"sha256": _snapshot_sha(TINY_CONSTITUTION)}
    outcomes = {
        f"sum-{s.id}": {"custom_id": f"sum-{s.id}", "type": "succeeded", "stop_reason": "end_turn",
                        "text": f"<summary>\n  Summary of {s.title}\n</summary>"}
        for s in sections
    }
    data = st.freeze_summaries(d, outcomes, sections, snapshot, model="m", template_hash="t")
    assert data["sections"][0]["summary"] == "Summary of Being honest"
    entries, meta = st.load_index(d)
    assert meta["sha256"] == data["sha256"] and len(entries) == 3
    with pytest.raises(st.SectionIndexError, match="frozen"):
        st.freeze_summaries(d, outcomes, sections, snapshot, model="m", template_hash="t")
    outcomes["sum-avoiding-harm"] = {"custom_id": "x", "type": "succeeded", "stop_reason": "refusal", "text": ""}
    with pytest.raises(st.SectionIndexError, match="no usable summary"):
        st.freeze_summaries(d, outcomes, sections, snapshot, model="m", template_hash="t", force=True)


def test_summarize_requests_carry_children_for_containers():
    secs = [Section("ch", "Chapter", 2, 0, 20), Section("leaf", "Leaf", 3, 5, 20)]
    md = "## Chapter\n\nframe\n\n### Leaf\n\nleaf text\n"
    secs[0].end = len(md)
    secs[1].start = md.index("### Leaf")
    secs[1].end = len(md)
    reqs = st.summarize_requests(md, secs, TPL, model="m")
    assert [r.custom_id for r in reqs] == ["sum-ch", "sum-leaf"]
    ch = reqs[0].params["messages"][0]["content"]
    assert "container section" in ch and "Leaf" in ch and "leaf text" in ch
    assert "leaf section" in reqs[1].params["messages"][0]["content"]


# ------------------------------------------------------- parse + repair --


def test_validate_tags_clean_reply_has_no_repairs():
    tags, repairs = st.parse_tag_reply(tag_json(["being-honest"], ["avoiding-harm"], False, 0.8), IDS)
    assert tags == {"primary": ["being-honest"], "secondary": ["avoiding-harm"], "none": False,
                    "confidence": 0.8}
    assert repairs == []


def test_validate_tags_lenient_repairs_are_named():
    obj = {
        "primary": "Being Honest, avoiding_harm, healthy-boundaries, being-honest",  # string, case, dup
        "secondary": ["avoiding-harm", "healthy-boundaries"],  # both already primary
        "none": "no",
        "confidence": 1.4,
    }
    tags, repairs = st.validate_tags(obj, IDS)
    assert tags["primary"] == ["being-honest", "avoiding-harm", "healthy-boundaries"]
    assert tags["secondary"] == [] and tags["none"] is False and tags["confidence"] == 1.0
    assert "list-from-string" in repairs and "primary:id-normalized" in repairs
    assert "primary:duplicate-dropped" in repairs and "secondary:duplicate-dropped" in repairs
    assert "none:coerced" in repairs and "confidence:clamped" in repairs


def test_validate_tags_trims_over_cap_and_reconciles_none():
    over = {"primary": IDS + ["being-honest"], "secondary": [], "none": False, "confidence": 0.5}
    tags, repairs = st.validate_tags(over, IDS)
    assert len(tags["primary"]) == 3 and "primary:duplicate-dropped" in repairs
    tags, repairs = st.validate_tags({"primary": [], "secondary": [], "none": False, "confidence": 0.5}, IDS)
    assert tags["none"] is True and "none:set-empty" in repairs
    tags, repairs = st.validate_tags({"primary": ["being-honest"], "secondary": [], "none": True,
                                      "confidence": 0.5}, IDS)
    assert tags["none"] is False and "none:cleared-tags-present" in repairs
    tags, repairs = st.validate_tags({"primary": [], "secondary": ["being-honest"], "none": False,
                                      "confidence": "0.4"}, IDS)
    assert tags["none"] is True and tags["confidence"] == 0.4 and "confidence:coerced" in repairs


def test_validate_tags_hard_failures():
    with pytest.raises(st.TagParseError, match="unknown section id"):
        st.validate_tags({"primary": ["not-a-section"], "secondary": [], "none": False, "confidence": 1}, IDS)
    with pytest.raises(st.TagParseError, match="not a JSON object"):
        st.validate_tags(["being-honest"], IDS)
    with pytest.raises(st.TagParseError, match="not a number"):
        st.validate_tags({"primary": ["being-honest"], "secondary": [], "none": False,
                          "confidence": "high"}, IDS)
    with pytest.raises(st.TagParseError, match="no JSON object"):
        st.parse_tag_reply("I would tag this as being-honest.", IDS)
    with pytest.raises(st.TagParseError, match="does not parse"):
        st.parse_tag_reply('{"primary": ["being-honest"],}', IDS)


def test_parse_tag_reply_strips_fences():
    tags, _ = st.parse_tag_reply("```json\n" + tag_json(["avoiding-harm"]) + "\n```", IDS)
    assert tags["primary"] == ["avoiding-harm"]


# ------------------------------------------------------------- the run --


def test_tag_request_shape_and_shared_cached_system(cdir):
    entries, _ = st.load_index(cdir)
    system = st.tag_system_block(TPL, entries)
    schema = st.tag_schema([e.id for e in entries])
    r_story = st.tag_request(story(1), "stories", TPL, system, schema, model="m")
    r_adv = st.tag_request(advice(2), "advice", TPL, system, schema, model="m")
    assert r_story.custom_id == "tag-sto-c00-u00-k01-y0" and r_adv.custom_id == "tag-adv-p00-t00-s02"
    assert r_story.params["system"] is system and system[0]["cache_control"] == {"type": "ephemeral"}
    assert "being-honest (chapter)" in system[0]["text"]
    assert "<story>" in r_story.params["messages"][0]["content"]
    assert "[user]" in r_adv.params["messages"][0]["content"] and "Answer 2." in r_adv.params["messages"][0]["content"]
    assert r_adv.lineage["lineage_section_id"] == "being-honest"
    assert r_story.params["output_config"]["format"]["schema"]["properties"]["primary"]["items"]["enum"] == IDS
    assert "temperature" not in r_story.params  # anthropic 1.0: no sampling controls
    assert st.tag_request(story(1), "stories", TPL, system, schema, "m", suffix="-r1").custom_id.endswith("-r1")


def test_run_tagging_rows_and_manifest(tmp_path, cdir):
    def responder(params, attempt):
        i = doc_index(params)
        if i == 3:
            return tag_json([], [], none=True, confidence=0.7)
        return live_message(tag_json(["being-honest"], ["avoiding-harm"] if i % 2 else [], False, 0.9))

    def resp(params, attempt):
        out = responder(params, attempt)
        return out if not isinstance(out, str) else live_message(out)

    run = make_run(tmp_path, cdir)
    client = FakeLiveClient(resp)
    manifest = st.run_tagging(run, engine(client), on_event=lambda m: None)
    rows = list(iter_jsonl(run.tags_path))
    assert [r["id"] for r in rows] == [story(i)["lineage"]["custom_id"] for i in range(5)]
    assert rows[0] == {"id": "sto-c00-u00-k00-y0", "status": "ok", "round": 0,
                       "primary": ["being-honest"], "secondary": [], "none": False, "confidence": 0.9}
    assert rows[1]["secondary"] == ["avoiding-harm"]
    assert rows[3]["none"] is True and rows[3]["primary"] == []
    assert manifest["n_records"] == 5 and manifest["n_requests_issued"] == 5
    assert manifest["status_counts"] == {"ok": 5} and manifest["n_none"] == 1
    assert manifest["index_sha256"] == read_json(cdir / st.SUMMARIES_FILE)["sha256"]
    assert set(manifest["template_hashes"]) == {st.TAG_SYSTEM_TEMPLATE, st.TAG_USER_TEMPLATE}
    assert manifest["model"] == "claude-opus-5" and manifest["engine"] == "live"
    assert read_json(run.manifest_path)["tags_sha256"] == manifest["tags_sha256"]
    assert client.calls == 5


def test_run_tagging_advice_rows_carry_lineage_section(tmp_path, cdir):
    recs = [advice(0, "being-honest"), advice(1, "avoiding-harm")]
    client = FakeLiveClient(lambda p, a: live_message(tag_json(["healthy-boundaries"])))
    run = make_run(tmp_path, cdir, corpus="advice", records=recs)
    st.run_tagging(run, engine(client), on_event=lambda m: None)
    rows = list(iter_jsonl(run.tags_path))
    assert [r["lineage_section_id"] for r in rows] == ["being-honest", "avoiding-harm"]
    assert all(r["primary"] == ["healthy-boundaries"] for r in rows)


def test_run_tagging_repairs_malformed_replies_in_bounded_rounds(tmp_path, cdir):
    seen: list[str] = []

    def responder(params, attempt):
        i = doc_index(params)
        seen.append(f"{i}:{attempt}")
        if i == 1:
            return live_message("not json at all")  # never repairs
        if i == 2:
            # malformed twice (main + repair 1), fixed on repair 2
            return live_message(tag_json(["being-honest"]) if attempt >= 3 else "{oops")
        if i == 4:
            return live_message(tag_json(["being-honest"]) if attempt >= 2 else
                                tag_json(["no-such-section"]))  # unknown id, fixed once re-asked
        return live_message(tag_json(["avoiding-harm"]))

    client = FakeLiveClient(responder)
    run = make_run(tmp_path, cdir, repair_rounds=2)
    manifest = st.run_tagging(run, engine(client), on_event=lambda m: None)
    rows = {r["id"]: r for r in iter_jsonl(run.tags_path)}
    r1, r2, r4 = rows["sto-c00-u00-k01-y0"], rows["sto-c00-u00-k02-y0"], rows["sto-c00-u00-k04-y0"]
    assert r1["status"] == "parse_failed" and r1["round"] == 2 and "no JSON object" in r1["error"]
    assert r2["status"] == "ok" and r2["round"] == 2 and r2["primary"] == ["being-honest"]
    assert r4["status"] == "ok" and r4["round"] == 1
    assert manifest["status_counts"] == {"ok": 4, "parse_failed": 1}
    # 5 main + 3 in repair round 1 (ids 1, 2, 4) + 2 in round 2 (ids 1, 2) — bounded
    assert manifest["n_requests_issued"] == 10 and client.calls == 10
    assert manifest["n_by_repair_round"] == {"0": 2, "1": 1, "2": 2}
    # repair requests ride fresh custom ids, so the engine's cache never
    # hands a malformed reply back as "already answered"
    wave_root = run.wave_root / "stories"
    assert (wave_root / "tag-repair1" / "live" / "tag-sto-c00-u00-k02-y0-r1.json").exists()
    assert (wave_root / "tag-repair2" / "live" / "tag-sto-c00-u00-k02-y0-r2.json").exists()


def test_run_tagging_records_refusals_without_reasking(tmp_path, cdir):
    def responder(params, attempt):
        if doc_index(params) == 0:
            return live_message("", stop_reason="refusal")
        return live_message(tag_json(["being-honest"]))

    client = FakeLiveClient(responder)
    run = make_run(tmp_path, cdir, records=[story(0), story(1)])
    manifest = st.run_tagging(run, engine(client), on_event=lambda m: None)
    rows = list(iter_jsonl(run.tags_path))
    assert rows[0]["status"] == "refusal" and rows[0]["none"] is None and rows[0]["primary"] == []
    assert manifest["status_counts"] == {"refusal": 1, "ok": 1} and client.calls == 2


def test_run_tagging_resumes_from_the_per_record_cache(tmp_path, cdir):
    boom = {"n": 0}

    def responder(params, attempt):
        boom["n"] += 1
        if boom["n"] == 3:
            raise AssertionError("interrupted mid-wave")
        return live_message(tag_json(["being-honest"]))

    client = FakeLiveClient(responder)
    run = make_run(tmp_path, cdir, records=[story(i) for i in range(6)])
    with pytest.raises(AssertionError):
        st.run_tagging(run, engine(client, max_concurrency=1), on_event=lambda m: None)
    cached = list((run.wave_root / "stories" / "tag" / "live").glob("*.json"))
    assert 0 < len(cached) < 6 and not run.tags_path.exists()
    calls_before = client.calls
    events: list[str] = []
    manifest = st.run_tagging(
        run, engine(client, max_concurrency=1, on_event=events.append), on_event=events.append
    )
    assert client.calls == calls_before + (6 - len(cached)), "only the missing records are re-asked"
    assert any("cached outcomes reused" in e for e in events)
    assert manifest["status_counts"] == {"ok": 6} and len(list(iter_jsonl(run.tags_path))) == 6
    # a finished run re-invoked issues nothing: the stamped wave is read back
    n = client.calls
    again = st.run_tagging(run, engine(client), on_event=lambda m: None)
    assert client.calls == n and again["tags_sha256"] == manifest["tags_sha256"]


def test_run_tagging_limit_takes_the_first_n(tmp_path, cdir):
    client = FakeLiveClient(lambda p, a: live_message(tag_json(["being-honest"])))
    run = make_run(tmp_path, cdir, records=[story(i) for i in range(6)], limit=2)
    manifest = st.run_tagging(run, engine(client), on_event=lambda m: None)
    assert manifest["n_records"] == 2 and manifest["limit"] == 2 and client.calls == 2


def test_document_fields_send_user_and_assistant_only():
    rec = advice(1)
    rec["system"] = "Deployment context: a bank."
    f = st.document_fields(rec, "advice")
    assert set(f) == {"kind", "user", "assistant"}
    with pytest.raises(ValueError):
        st.document_fields(rec, "poems")


# ------------------------------------------------- client vault fallback --


def test_read_key_falls_back_to_the_vault_when_no_key_file(monkeypatch, tmp_path):
    from corpus_generator import client as cl

    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.setenv("SISTEER_SECRET_CMD", "fake-secret-cmd")
    monkeypatch.setattr(cl.shutil, "which", lambda name: "/usr/local/bin/fake-secret-cmd")
    calls: list[list[str]] = []

    def runner(argv, capture_output, text):
        calls.append(argv)
        value = {"lowprio": "FAKE-LOW-1\n", "highprio": "FAKE-HIGH-2\n", "batch": "FAKE-BATCH-3\n"}[argv[1]]
        return NS(returncode=0, stdout=value, stderr="")

    assert cl.read_key("live", runner=runner) == "FAKE-LOW-1"
    assert cl.read_key("batch", runner=runner) == "FAKE-BATCH-3"
    assert [c[1] for c in calls] == ["lowprio", "highprio", "batch", "highprio"]


def test_read_key_vault_highprio_match_refused(monkeypatch, tmp_path):
    from corpus_generator import client as cl

    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.setenv("SISTEER_SECRET_CMD", "fake-secret-cmd")
    monkeypatch.setattr(cl.shutil, "which", lambda name: "/usr/local/bin/fake-secret-cmd")
    runner = lambda argv, capture_output, text: NS(returncode=0, stdout="SAME-KEY\n", stderr="")
    with pytest.raises(cl.ClientGuardError, match="HIGHPRIO"):
        cl.read_key("live", runner=runner)


def test_read_key_vault_failure_is_loud(monkeypatch, tmp_path):
    from corpus_generator import client as cl

    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.setenv("SISTEER_SECRET_CMD", "fake-secret-cmd")
    monkeypatch.setattr(cl.shutil, "which", lambda name: "/usr/local/bin/fake-secret-cmd")
    runner = lambda argv, capture_output, text: NS(returncode=1, stdout="", stderr="no session")
    with pytest.raises(cl.ClientGuardError, match="failed"):
        cl.read_key("live", runner=runner)
    monkeypatch.setattr(cl.shutil, "which", lambda name: None)
    with pytest.raises(cl.ClientGuardError, match="not on PATH"):
        cl.read_key("live", runner=runner)
