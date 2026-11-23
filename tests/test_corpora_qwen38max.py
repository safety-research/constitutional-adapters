"""The Qwen 3.8 Max constitution corpus (corpora/constitution_qwen38max_{advice,stories}) against its provenance sidecar,
the committed held-out split and the identities pinned below.

The corpus is the generator comparison's extra corpus (about 2.0M tokens, the minimum over the three reference
tokenizers); its 1M-token training pool constqwen38-mix-1m is NOT committed, only its ordered doc-id list inside the
provenance sidecar. These tests show that list is enough: every pool id names a committed final record, 470 pool docs are
the rendered records byte for byte and the last is a 645-character prefix of one, and the rebuilt pool file hashes to the
sha256 the results registry filed for it (POOL_SHA256, a literal here, so a re-issued sidecar cannot move it). The
held-out EVAL pool rebuilds the same way from the split's eval_subset. They also check that the held-out split
data/sdfbase/heldout_split_constqwen38.json was carved from these same finals and that the pool lies on its training
side. tests/test_corpora_shards.py covers the shards themselves (shas, counts, sizes, manifests).

The corpus is under its own terms (THIRD_PARTY_NOTICES entry 14, LICENSES/Qwen38Max-corpus-terms.txt), not the
repository's MIT licence. A release either keeps that entry beside the corpus or leaves both directories out: the last test
fails a publication that carries the corpus without the entry, and a publication without the corpus skips this module.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
from front_door_testlib import MATCHERS, token_hits
from published_repo import is_published

REPO = Path(__file__).resolve().parents[1]
CORPORA = REPO / "corpora"
ADVICE, STORIES = "constitution_qwen38max_advice", "constitution_qwen38max_stories"
PROV_NAME = "provenance-qwen38-1m-20261112.json"
SPLIT = REPO / "data/sdfbase/heldout_split_constqwen38.json"
SPLIT_PROV = REPO / "data/sdfbase/heldout_split_constqwen38.provenance.json"
LENGTH_DROP = "adv-p00-t08-s09"   # the one final record over the 8,192-token cap, dropped before the carve
PREFIX = "q38-"

#: Identities pinned here rather than read from the sidecar, so a re-issued sidecar or split cannot move them unseen: the
#: two pool files, the split and the filter reports as the results registry filed them for this run (2026-11-12), the
#: split's provenance as its builder committed it, and the sidecar as committed (re-pin it only after reviewing a re-issue).
POOL_SHA256 = "6a8cc146994c178c3b2dc8f311644b8d9014827b0c843019952a9a5d43f13efe"
HELDOUT_POOL_SHA256 = "01d9d7518fc08faeeda73994f036d329bfa49982fd656de92229c54dba8d5aae"
SPLIT_SHA256 = "6dc9a62f09bc187bc3568204b314a50e44082c39f54b263846210f0b626652ec"
SPLIT_PROVENANCE_SHA256 = "cd56889cc6f7bcd5bb8b40e0570c3aee6233a6be414c3d02b861f5f8a08b95a6"
FILTER_REPORT_SHA256 = {"advice": "adbe03a9e9481b41bed15bdf358ed6bd68bb9c5be1bf047c718d79aa592fe630",
                        "stories": "788eb0c871745094ed834dc9a018e95c3fde064829b5262ced009e27200929bf"}
PROVENANCE_SHA256, PROVENANCE_BYTES = "e010c2885c26fb8be3e926d717fdbc8ade27b16a99ddc3b9c452cc41cbccc5e3", 26_698


def _rc():
    spec = importlib.util.spec_from_file_location("reassemble_corpus", REPO / "reproduce/tools/reassemble_corpus.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.fixture(scope="module", autouse=True)
def _left_out_by_the_release():
    """A publication may leave the corpus out (both directories): it skips here. A development checkout without them fails,
    and so does a publication carrying only one."""
    if not any((CORPORA / c).is_dir() for c in (ADVICE, STORIES)) and is_published():
        pytest.skip("the release left the Qwen 3.8 Max corpus out")


@pytest.fixture(scope="module")
def rc():
    return _rc()


@pytest.fixture(scope="module")
def lines(rc):
    """The committed finals' raw record lines, per family, straight from the shards."""
    return {"advice": list(rc.iter_shard_lines(ADVICE)), "stories": list(rc.iter_shard_lines(STORIES))}


@pytest.fixture(scope="module")
def records(lines):
    return {fam: [json.loads(l) for l in ls] for fam, ls in lines.items()}


@pytest.fixture(scope="module")
def by_id(records):
    """custom_id -> (family, record)."""
    return {r["lineage"]["custom_id"]: (fam, r) for fam in ("advice", "stories") for r in records[fam]}


@pytest.fixture(scope="module")
def prov():
    return json.loads((CORPORA / ADVICE / PROV_NAME).read_text(encoding="utf-8"))


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _render(rec: dict, family: str) -> tuple[str, str]:
    """(loss-masked prompt, trained pos) as the pool builder splits corpus_generator.tokens' render: advice is
    render_advice_text = system + "\\n\\n" + user + "\\n\\n" + assistant (system and its separator dropped when empty),
    the system part masked; stories is render_story_text = text, nothing masked."""
    if family == "advice":
        system = rec.get("system") or ""
        return (system + "\n\n" if system else ""), rec["user"] + "\n\n" + rec["assistant"]
    return "", rec["text"]


def _row(doc_id: str, family: str, prompt: str, pos: str, cell: str, counts: dict, truncated: bool) -> str:
    """One pool line, as the pool builder writes it."""
    return json.dumps({"prompt": prompt, "pos": pos, "neg": "", "instruction_id": cell, "doc_id": doc_id, "family": family,
                       "official_tokens": min(counts.values()), "truncated": truncated},
                      ensure_ascii=False, separators=(",", ":")) + "\n"


def test_provenance_sits_in_both_parts_and_is_listed_by_both_manifests(rc):
    a, s = (CORPORA / ADVICE / PROV_NAME).read_bytes(), (CORPORA / STORIES / PROV_NAME).read_bytes()
    assert a == s
    for corpus in (ADVICE, STORIES):
        man = rc.load_manifest(corpus)
        assert PROV_NAME in {sc["name"] for sc in man["sidecars"]}
        assert man["vendored_corpus_commit"] is None, "not generated from a vendored snapshot; code provenance is in the sidecar"


def test_the_sidecar_split_and_filter_reports_are_the_pinned_files(prov):
    side = (CORPORA / ADVICE / PROV_NAME).read_bytes()
    assert (_sha(side), len(side)) == (PROVENANCE_SHA256, PROVENANCE_BYTES), "the sidecar was re-issued: review it, then re-pin"
    assert _sha(SPLIT.read_bytes()) == SPLIT_SHA256
    assert _sha(SPLIT_PROV.read_bytes()) == SPLIT_PROVENANCE_SHA256
    for corpus, fam in ((ADVICE, "advice"), (STORIES, "stories")):
        assert _sha((CORPORA / corpus / prov["finals"][fam]["filter_report"]["file"]).read_bytes()) == FILTER_REPORT_SHA256[fam]
        assert prov["finals"][fam]["filter_report"]["sha256"] == FILTER_REPORT_SHA256[fam]
    pool = prov["training_pool_1m"]
    assert pool["sha256"] == POOL_SHA256 and pool["heldout_pool"]["sha256"] == HELDOUT_POOL_SHA256
    assert (pool["heldout_split"]["sha256"], pool["heldout_split"]["provenance_sha256"]) == (SPLIT_SHA256, SPLIT_PROVENANCE_SHA256)


def test_provenance_finals_agree_with_the_manifests(rc, prov):
    total = 0
    for corpus, fam in ((ADVICE, "advice"), (STORIES, "stories")):
        man, f = rc.load_manifest(corpus), prov["finals"][fam]
        assert f["sha256"] == man["sha256_reassembled"] and f["records"] == man["record_count"]
        assert f["bytes"] == man["bytes_reassembled"] and f["source_file"] == man["source_file"]
        assert f["official_tokens"] == man["official_tokens"]["min"]
        assert f["official_tokens_per_tokenizer"] == {k: v for k, v in man["official_tokens"].items() if k != "min"}
        rep = next(sc for sc in man["sidecars"] if sc["name"].endswith(".filter-report.json"))
        assert f["filter_report"] == {"file": rep["name"], "sha256": rep["sha256"]}
        total += f["official_tokens"]
    assert total == prov["official_tokens_combined"] == 1_999_496
    pins = _sha((CORPORA / "TOKENIZERS.json").read_bytes())
    assert prov["inputs"]["reference_tokenizer_pins"] == {"file": "corpora/TOKENIZERS.json", "sha256": pins}
    constitution = _sha((REPO / prov["inputs"]["constitution"]["file"]).read_bytes())
    assert constitution == prov["inputs"]["constitution"]["sha256"]


def test_token_totals_are_the_records_sums(rc, records, prov):
    """Each part's per-tokenizer totals, in its MANIFEST, the sidecar and its filter report, are the sums of its records'
    token_counts (the records themselves are pinned by tests/test_corpora_shards.py's FINALS)."""
    for corpus, fam in ((ADVICE, "advice"), (STORIES, "stories")):
        sums: dict[str, int] = {}
        for r in records[fam]:
            for k, v in r["token_counts"].items():
                sums[k] = sums.get(k, 0) + v
        man = rc.load_manifest(corpus)
        report = json.loads((CORPORA / corpus / prov["finals"][fam]["filter_report"]["file"]).read_text(encoding="utf-8"))
        assert set(sums) == {"llama3", "gemma", "qwen"}
        assert sums == {k: v for k, v in man["official_tokens"].items() if k != "min"}
        assert sums == prov["finals"][fam]["official_tokens_per_tokenizer"]
        assert sums == {k: int(v) for k, v in report["token_totals"].items()}


def test_every_record_names_the_run_the_provenance_names(records, prov):
    main = next(r for r in prov["runs"] if r["run_id"] == "qwen38-1m-20261112")
    assert "main run" in main["role"]
    for fam in ("advice", "stories"):
        assert {r["lineage"]["run_id"] for r in records[fam]} == {main["run_id"]}
        assert {r["lineage"]["constitution_sha256"] for r in records[fam]} == {prov["inputs"]["constitution"]["sha256"]}
    assert prov["generator"]["model"] == "qwen/qwen3.8-max-0902" and prov["generator"]["provider"] == "openrouter"


def test_generation_commits_are_marked_inferred(prov):
    """The run manifests record no code commit; the provenance must not present the generation commits as recorded."""
    by_tag = {c["tag"]: c for c in prov["code"]["commits"]}
    assert by_tag["pin/qwen-gen-1m-8407605"]["inferred_from_timing"] is True
    assert by_tag["pin/qwen-gen-1m-1d402a3"]["inferred_from_timing"] is True
    assert all(len(c["commit"]) == 40 for c in by_tag.values())
    assert "INFERRED" in prov["code"]["note"]


def test_the_1m_pool_rebuilds_byte_for_byte_from_the_committed_finals(by_id, prov):
    pool = prov["training_pool_1m"]
    ids = pool["doc_ids_in_order"]
    assert len(ids) == pool["n_docs"] == 471 and len(set(ids)) == 471
    assert all(i.startswith(PREFIX) and i[len(PREFIX):] in by_id for i in ids), "a pool id names no committed final record"
    trunc = pool["truncation"]
    assert ids[-1] == trunc["doc_id"] and trunc["char_offset"] == 645
    rows, equal, cut = [], 0, 0
    totals = dict.fromkeys(pool["per_tokenizer_totals"], 0)
    fams = {"advice": 0, "stories": 0}
    for did in ids:
        fam, rec = by_id[did[len(PREFIX):]]
        prompt, pos = _render(rec, fam)
        truncated = did == trunc["doc_id"]
        if truncated:
            full = prompt + pos
            assert len(prompt) <= trunc["char_offset"] < len(full)
            pos = full[len(prompt):trunc["char_offset"]]
            counts = trunc["kept_per_tokenizer"]
            cut += 1
        else:
            counts = rec["token_counts"]
            equal += 1
        for k in totals:
            totals[k] += counts[k]
        fams[fam] += 1
        rows.append(_row(did, fam, prompt, pos, pool["cell"], counts, truncated))
    assert (equal, cut) == (470, 1) == (pool["reconstruction"]["byte_equal_docs"], pool["reconstruction"]["truncated_docs"])
    assert fams == pool["family_counts"]
    assert totals == pool["per_tokenizer_totals"] and min(totals.values()) == pool["official_tokens"] == 1_000_000
    assert _sha("".join(rows).encode("utf-8")) == pool["sha256"] == POOL_SHA256


def test_the_heldout_eval_pool_rebuilds_byte_for_byte_from_the_split_and_the_finals(by_id, prov):
    """The comparison's 128-doc held-out EVAL pool is not committed either: it is the split's eval_subset, in that order,
    each record rendered whole."""
    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    held = prov["training_pool_1m"]["heldout_pool"]
    ids = split["eval_subset"]["ids"]
    assert len(ids) == len(set(ids)) == split["eval_subset"]["n"] == held["n_docs"] == 128
    rows = []
    for oid in ids:
        fam, rec = by_id[oid]
        prompt, pos = _render(rec, fam)
        rows.append(_row(PREFIX + oid, fam, prompt, pos, held["cell"], rec["token_counts"], False))
    assert _sha("".join(rows).encode("utf-8")) == held["sha256"] == HELDOUT_POOL_SHA256


def test_the_split_was_carved_from_these_finals(lines, records, prov):
    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    sprov = json.loads(SPLIT_PROV.read_text(encoding="utf-8"))
    pool = prov["training_pool_1m"]
    assert _sha(SPLIT.read_bytes()) == pool["heldout_split"]["sha256"] == sprov["artifact_sha256"] == SPLIT_SHA256
    assert _sha(SPLIT_PROV.read_bytes()) == pool["heldout_split"]["provenance_sha256"] == SPLIT_PROVENANCE_SHA256
    final_sha = {fam: _sha(b"".join(lines[fam])) for fam in lines}
    for fam in ("advice", "stories"):
        assert sprov["carve"]["inputs"][fam]["original_sha256"] == final_sha[fam] == prov["finals"][fam]["sha256"]
    # what the carve read: the finals less the length-cap drop (advice only)
    lenok = {"advice": _sha(b"".join(
        l for l, r in zip(lines["advice"], records["advice"]) if r["lineage"]["custom_id"] != LENGTH_DROP)),
        "stories": final_sha["stories"]}
    assert split["built_from_sha256"] == {"advice_final": lenok["advice"], "stories_final": lenok["stories"]}
    assert {"advice_sha256": lenok["advice"], "stories_sha256": lenok["stories"]} == {
        k: v for k, v in pool["length_checked_finals"].items() if k.endswith("_sha256")}
    assert pool["length_cap_drop"]["doc_id"] == LENGTH_DROP
    final_ids = {r["lineage"]["custom_id"] for fam in records for r in records[fam]}
    held = set(split["advice"]["ids"]) | set(split["stories"]["ids"])
    assert held <= final_ids and set(split["eval_subset"]["ids"]) <= held and LENGTH_DROP not in held
    # the pool lies on the training side: no pool doc is held out, nor shares a held-out group, nor is the dropped record
    group = {r["lineage"]["custom_id"]: f"{r['principle_id']}|{r['prompt_type_id']}" for r in records["advice"]}
    group |= {r["lineage"]["custom_id"]: r["subcategory_id"] for r in records["stories"]}
    held_groups = set(split["advice"]["groups"]) | set(split["stories"]["groups"])
    pool_ids = {i[len(PREFIX):] for i in pool["doc_ids_in_order"]}
    assert not pool_ids & held and LENGTH_DROP not in pool_ids
    assert not {group[i] for i in pool_ids} & held_groups


def test_provenance_carries_no_machine_path_and_no_gitleaks_shape():
    text = (CORPORA / ADVICE / PROV_NAME).read_text(encoding="utf-8")
    assert token_hits(text, MATCHERS["tree"]) == []   # every tree-scope row of the publication scan's one token table
    assert "/home/" not in text and "~/" not in text, "a home directory"
    for line in text.splitlines():   # a key naming a token/key/secret beside a 64-hex value reads as a credential
        if ":" in line:
            key, _, val = line.partition(":")
            v = val.strip().strip('",')
            if len(v) == 64 and all(c in "0123456789abcdef" for c in v):
                assert not any(w in key.lower() for w in ("token", "key", "secret", "auth", "passw", "credential", "api")), line


def test_a_publication_carries_this_corpus_only_beside_a_third_party_notice_for_it():
    """The corpus is Qwen 3.8 Max output under its own terms (THIRD_PARTY_NOTICES entry 14, with the terms file under
    LICENSES/), not the repository's MIT licence. A published repository therefore carries it only beside that entry; the
    other way through a release is to leave both directories
    out, which _left_out_by_the_release and tests/test_corpora_shards.py allow. A development checkout is not a
    publication, and the check passes there."""
    if "constitution_qwen38max" in (REPO / "THIRD_PARTY_NOTICES").read_text(encoding="utf-8"):
        return
    assert not is_published(), (
        "corpora/constitution_qwen38max_* is Qwen 3.8 Max output that THIRD_PARTY_NOTICES does not list: check the provider's "
        "output terms and write its entry, or leave both directories out of the release")
