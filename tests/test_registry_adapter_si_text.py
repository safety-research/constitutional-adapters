"""SI texts (`adapters/si_text.py`): every seed sha is the sha256 of the exact bytes the code ships;
every SI id the two-SI configs and the Petri manifests name has a row (by id or by 16-hex prefix hash); an unknown
prefix hash refuses BY NAME; a seed whose sha disagrees with the bytes is refused.
"""
from __future__ import annotations

import glob
import hashlib
import os
import re

import pytest
import yaml

from sisteer.registry import synth
from sisteer.registry.adapters import dimensions as D
from sisteer.registry.adapters import si_text as SI

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEEDS = os.path.join(ROOT, "configs", "registry")


@pytest.fixture(scope="module")
def rows():
    return D.load_seeds(SEEDS).si_texts


SHIPPED = {"safety_refusal", "si_constitution", "si_constitution_full", "si_irrelevant", "si_caa"}
CANDIDATE = {"A10", "B3", "C16"}     # Teachers of the candsi objects, held in an out-of-tree snapshot


def _snapshot_present(rows) -> bool:
    ts = next(r["text_source"] for r in rows if r["si_id"] == "A10")
    return os.path.isfile(SI.external_snapshot_path(ts))


def test_every_seed_sha_is_the_sha_of_the_shipped_bytes(rows):
    """The five texts the code ships ALWAYS verify; the three candidate texts verify when their snapshot is on this host
    and are otherwise reported as unverifiable BY NAME (never skipped silently)."""
    unv = {}
    got = SI.verify_seed_shas(rows, unverifiable=unv)
    assert SHIPPED <= set(got)
    assert set(got) | set(unv) == SHIPPED | CANDIDATE and set(unv) <= CANDIDATE
    if _snapshot_present(rows):
        assert set(got) == SHIPPED | CANDIDATE and not unv
    else:
        assert set(unv) == CANDIDATE and all("not present on this host" in why for why in unv.values())
    by_id = {r["si_id"]: r for r in rows}
    for sid in got:
        row = by_id[sid]
        assert row["si_text_sha256"] == hashlib.sha256(SI.text_bytes_for(sid, row)).hexdigest()
    for row in rows:
        if "si_text_sha16_prefix_hash" in row:      # the reference texts carry none (no manifest pins them)
            assert row["si_text_sha16_prefix_hash"] == row["si_text_sha256"][:16]
    # every prefix hash is distinct (a manifest's 16-hex pin names one text)
    assert len(SI.prefix_hash_index(rows)) == len(rows)


def test_a_seed_sha_that_disagrees_with_the_bytes_is_refused_by_name(rows):
    bad = [dict(r) for r in rows]
    bad[1]["si_text_sha256"] = "0" * 64
    with pytest.raises(SI.SiTextRefusal) as ei:
        SI.verify_seed_shas(bad, unverifiable={})
    assert bad[1]["si_id"] in str(ei.value) and "0000000000000000" in str(ei.value)


# --------------------------------------------------------------------------------------------
# candidate texts from the out-of-tree snapshot
# --------------------------------------------------------------------------------------------


def _fake_snapshot(tmp_path, entries, name="snap.json"):
    import json
    p = tmp_path / name
    p.write_text(json.dumps({"meta": {"synthetic": True}, "system_instructions": entries}), encoding="utf-8")
    return str(p), hashlib.sha256(p.read_bytes()).hexdigest()


def _row(si_id, text, path, file_sha, entry_id=None, kind="candidate_sis_snapshot"):
    return {"si_id": si_id, "si_kind": "safety_instruction", "si_text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "source_uri": path, "text_source": {"kind": kind, "path": path, "env": "SISTEER_TEST_SNAPSHOT_NOT_SET", "file_sha256": file_sha,
                                                "entry_id": entry_id or si_id}}


def test_candidate_text_rows_verify_against_the_snapshot(rows):
    """On a host with the snapshot: the seeded shas ARE the sha256 of the entries' text bytes, the snapshot's own
    cross-check entries hash to the shipped paper texts (the convention is this file's), and `si_kind` follows the
    snapshot's category. Elsewhere the rows load with the seeded sha and say so (the next test, host-independent)."""
    if not _snapshot_present(rows):
        pytest.skip("candidate-SI snapshot not present on this host (the seeded shas were verified on the grading host)")
    by_id = {r["si_id"]: r for r in rows}
    import json
    path = SI.external_snapshot_path(by_id["A10"]["text_source"])
    doc = json.load(open(path, encoding="utf-8"))
    ents = {e["id"]: e for e in doc["system_instructions"]}
    for sid in CANDIDATE:
        assert by_id[sid]["si_text_sha256"] == hashlib.sha256(ents[sid]["text"].encode("utf-8")).hexdigest()
        assert by_id[sid]["si_kind"] == {"safety": "safety_instruction", "constitution": "constitution_short", "irrelevant": "irrelevant_control"}[ents[sid]["category"]]
    for cand, shipped in (("A1", "safety_refusal"), ("B2", "si_constitution"), ("C3", "si_irrelevant")):
        assert hashlib.sha256(ents[cand]["text"].encode("utf-8")).hexdigest() == by_id[shipped]["si_text_sha256"], cand


def test_absent_snapshot_is_loud_not_silent(tmp_path, monkeypatch):
    """A candidate row whose snapshot is not on this host: `verify_seed_shas` without a collector REFUSES; with one, the
    row is recorded as unverifiable and `records_for` stamps field_confidence_json.si_text_sha256 with the reason."""
    from sisteer.registry import synth
    path = str(tmp_path / "missing.json")
    row = _row("Z9", "synthetic candidate text", path, "0" * 64)
    with pytest.raises(SI.SnapshotUnavailable) as ei:
        SI.verify_seed_shas([row])
    assert "Z9" in str(ei.value) and "missing.json" in str(ei.value) and "not present on this host" in str(ei.value)
    unv = {}
    assert SI.verify_seed_shas([row], unverifiable=unv) == {} and set(unv) == {"Z9"}
    norm = D.Normaliser("si_texts.yaml", "Z9", "unverified")
    recs = SI.records_for([row], synth.producer("t"), "2026-09-17T00:00:00.000Z", norm, verify=True, unverifiable=unv)
    fc = recs[0]["payload"]["field_confidence_json"]["si_text_sha256"]
    assert fc["confidence"] == "recorded" and "NOT re-verified" in fc["reason"] and "missing.json" in fc["reason"]
    assert recs[0]["payload"]["si_text_sha256"] == row["si_text_sha256"] and recs[0]["payload"]["extra"]["text_source"] == row["text_source"]
    # the loader surfaces it as a warning (never a silent load)
    seeds = D.load_seeds(SEEDS)
    monkeypatch.setenv("SISTEER_CANDIDATE_SIS", path)
    b = D.build_records(seeds, synth.producer("t"))
    assert sorted(w.split(":")[0] for w in b.warnings if w.startswith("si_text ")) == ["si_text A10", "si_text B3", "si_text C16"]


def test_snapshot_present_but_drifted_or_wrong_refuses_naming_what_it_saw(tmp_path):
    text = "synthetic candidate text"
    path, file_sha = _fake_snapshot(tmp_path, [{"id": "Z9", "text": text, "category": "safety"}])
    # the happy path: file sha + entry text agree with the row
    assert SI.verify_seed_shas([_row("Z9", text, path, file_sha)]) == {"Z9": hashlib.sha256(text.encode()).hexdigest()}
    # a drifted snapshot file is refused BEFORE any text is read out of it
    with pytest.raises(SI.SiTextRefusal) as ei:
        SI.verify_seed_shas([_row("Z9", text, path, "1" * 64)])
    assert "file_sha256" in str(ei.value) and "1111111111111111" in str(ei.value) and not isinstance(ei.value, SI.SnapshotUnavailable)
    # the entry's text differs from the seeded sha
    with pytest.raises(SI.SiTextRefusal) as ei:
        SI.verify_seed_shas([_row("Z9", "some other text", path, file_sha)])
    assert "Z9" in str(ei.value) and "!=" in str(ei.value)
    # an entry the snapshot does not hold
    with pytest.raises(SI.SiTextRefusal) as ei:
        SI.verify_seed_shas([_row("Z8", text, path, file_sha, entry_id="Z8")])
    assert "no entry with id 'Z8'" in str(ei.value)
    # a text_source kind the adapter does not read
    with pytest.raises(SI.SiTextRefusal) as ei:
        SI.verify_seed_shas([_row("Z9", text, path, file_sha, kind="mystery")])
    assert "mystery" in str(ei.value)


def test_unknown_si_id_refuses():
    with pytest.raises(SI.SiTextRefusal):
        SI.text_bytes_for("not_an_instruction")


def test_unknown_prefix_hash_refuses_naming_the_hash(rows):
    idx = SI.prefix_hash_index(rows)
    assert idx["ac3a58f61d53cf24"] == "safety_refusal" and idx["21e06f482a6559b3"] == "si_constitution"
    with pytest.raises(SI.UnknownPrefixHash) as ei:
        SI.si_id_for_prefix_hash("deadbeefdeadbeef", idx)
    assert "deadbeefdeadbeef" in str(ei.value)


def test_every_si_id_the_two_si_configs_name_has_a_row(rows):
    """Every SYSTEM instruction the two-SI configs name has a row. The document-corpus pseudo-instructions
    (`Instruction.category == "corpus_cell"`: `constv2-mix-full`, ...) ride the SI slot of the cell grammar but are
    never rendered as instructions — they are `object.fit_corpus_id` values, not `si_text` rows (the seed's note)."""
    from sisteer.data import instructions as I
    ids = {r["si_id"] for r in rows}
    named = set()
    for path in sorted(glob.glob(os.path.join(ROOT, "configs", "two_si*.yaml"))):
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        for sid in data.get("si_ids") or []:
            named.add(str(sid))
    assert named, "no two-SI config names an si_ids list"
    corpus = {sid for sid in named if sid in I.BY_ID and I.BY_ID[sid].category == "corpus_cell"}
    assert corpus, "expected the sdf-base corpus pseudo-instructions among the two-SI si_ids"
    missing = named - corpus - ids
    assert not missing, missing


def test_every_petri_manifest_prefix_hash_has_a_row(rows):
    idx = SI.prefix_hash_index(rows)
    seen = 0
    for path in sorted(glob.glob(os.path.join(ROOT, "configs", "capeval", "manifests", "*.yaml"))):
        with open(path, "r", encoding="utf-8") as fh:
            docs = [d for d in yaml.safe_load_all(fh) if isinstance(d, dict)]   # some manifests are multi-document
        for data in docs:
            hashes = (data.get("petri") or {}).get("prompted_prefix_hashes") or data.get("prompted_prefix_hashes") or {}
            for label, h16 in dict(hashes).items():
                seen += 1
                assert SI.si_id_for_prefix_hash(str(h16), idx), (path, label)
    assert seen > 0, "no manifest carries prompted_prefix_hashes"


def test_si_text_records_carry_token_counts_and_kinds(rows):
    seeds = D.load_seeds(SEEDS)
    b = D.build_records(seeds, synth.producer("t"))
    recs = {r["payload"]["si_id"]: r["payload"] for r in b.records["si_text"]}
    const = recs["si_constitution"]
    counts = const["si_token_count_by_model_json"]
    counts = counts if isinstance(counts, dict) else __import__("json").loads(counts)
    assert counts["llama8b_it"]["count"] == 5872 and counts["gemma4e4b_it"] == "unverified"
    assert recs["si_caa"]["si_kind"] == "reference_other" and "si_hhh" not in recs
    assert const["si_text_sha256"] == "21e06f482a6559b39309c4f1128f0411230c88e02b3a9eaa7b5d50829e6a9a0f"
