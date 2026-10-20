"""The four released corpora committed as record-boundary gzip shards under corpora/:
reassembling the shards yields files whose sha256 equal the finals' and whose record counts are the paper's
(1,960 / 16,321 / 1,995 / 16,294); every committed file is under 50 MB; each shard is itself valid JSONL; the
sidecars match their manifest; and the reference-tokenizer pins are complete."""

from __future__ import annotations

import gzip
import importlib.util
import json
from pathlib import Path

import pytest
from front_door_testlib import assert_no_tokens

REPO = Path(__file__).resolve().parents[1]
CORPORA = REPO / "corpora"
TOOL = REPO / "reproduce/tools/reassemble_corpus.py"

#: The finals' identities (corpus_generator data/, measured 2026-09-28 and pinned by the paper's Table 1 counts).
FINALS = {
    "constitution_advice": ("afe6cb396bf9ca820f252942bb44b8fb628db5986a38dd80256f92163e4d85a6", 1960, 68_295_108,
                            "final-production-20260831-v2.jsonl"),
    "constitution_stories": ("c8d7e502045f14eef021b645b2794c7eef8d189a7981487f92925902580174e8", 16321, 411_115_810,
                             "final-production-20260831-v2.jsonl"),
    "control_advice": ("ba89cf60175f54347022df63d89214983855b5f726a8ea16427ba3797a3abae7", 1995, 66_555_587,
                       "final-control-scale-20260903-r2-advice.jsonl"),
    "control_stories": ("915129b2b89eb00bda0a0ca13923708d6bb7cff3b4332d67f69c073555ab6dad", 16294, 427_096_097,
                        "final-control-scale-20260903-r2-stories.jsonl"),
}
#: Official token counts (min over the three reference tokenizers) from the finals' filter reports.
OFFICIAL_TOKENS = {"constitution_advice": 5_029_274, "constitution_stories": 40_053_044,
                   "control_advice": 4_883_450, "control_stories": 40_943_945}
HARD_LIMIT = 50 * 1000 * 1000
TARGET = 40 * 1000 * 1000


def _mod():
    spec = importlib.util.spec_from_file_location("reassemble_corpus", TOOL)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.fixture(scope="module")
def rc():
    return _mod()


@pytest.mark.parametrize("corpus", sorted(FINALS))
def test_manifest_pins_the_final(corpus, rc):
    sha, n, nbytes, source = FINALS[corpus]
    man = rc.load_manifest(corpus)
    assert man["sha256_reassembled"] == sha
    assert man["record_count"] == n
    assert man["bytes_reassembled"] == nbytes
    assert man["source_file"] == source
    assert sum(s["records"] for s in man["shards"]) == n
    assert sum(s["raw_bytes"] for s in man["shards"]) == nbytes
    assert man["official_tokens"]["min"] == OFFICIAL_TOKENS[corpus]
    assert set(man["official_tokens"]) == {"llama3", "gemma", "qwen", "min"}
    assert man["gzip"]["mtime"] == 0 and man["gzip"]["record_boundary"] is True


@pytest.mark.parametrize("corpus", sorted(FINALS))
def test_every_committed_file_is_small_and_matches_its_sha(corpus, rc):
    man = rc.verify_shard_files(corpus)  # shard + sidecar shas and sizes
    d = CORPORA / corpus
    for p in d.iterdir():
        assert p.stat().st_size < HARD_LIMIT, f"{p} is {p.stat().st_size} B (GitHub limit margin)"
    for s in man["shards"]:
        assert s["bytes"] <= TARGET, f"{corpus}/{s['name']} exceeds the 40 MB target"
    listed = {s["name"] for s in man["shards"]} | {sc["name"] for sc in man["sidecars"]} | {"MANIFEST.json"}
    assert {p.name for p in d.iterdir()} == listed, "an unlisted or missing file under the corpus directory"


def test_eight_shards_in_total(rc):
    assert sum(len(rc.load_manifest(c)["shards"]) for c in FINALS) == 8


@pytest.mark.parametrize("corpus", sorted(FINALS))
def test_each_shard_is_valid_jsonl_cut_between_records(corpus, rc):
    man = rc.load_manifest(corpus)
    d = CORPORA / corpus
    for s in man["shards"]:
        last = b""
        with gzip.open(d / s["name"], "rb") as gz:
            first = gz.readline()
            assert first.startswith(b"{"), f"{s['name']} does not start at a record"
            json.loads(first)
            n = 1
            for line in gz:
                n += 1
                last = line
        json.loads(last or first)
        assert (last or first).endswith(b"\n"), f"{s['name']} ends mid-record"
        assert n == s["records"]


@pytest.mark.parametrize("corpus", sorted(FINALS))
def test_reassembled_sha_and_count_equal_the_finals(corpus, rc):
    """The acceptance check itself, streamed (no ~400 MB file written)."""
    sha, n, nbytes, _ = FINALS[corpus]
    res = rc.reassemble(corpus, None)
    assert res["sha256"] == sha and res["records"] == n and res["bytes"] == nbytes


def test_reassemble_writes_the_final_atomically(tmp_path, rc):
    out = tmp_path / "finals" / "final-production-20260831-v2.jsonl"
    res = rc.reassemble("constitution_advice", out)
    assert out.is_file() and res["out"] == str(out)
    assert rc.sha256_file(out) == FINALS["constitution_advice"][0]
    with open(out, "rb") as fh:
        assert sum(1 for _ in fh) == 1960
    assert not [p for p in out.parent.iterdir() if p.name.startswith(".")], "no temp file left behind"
    rec = next(rc.iter_records("constitution_advice"))
    assert {"system", "user", "assistant", "token_counts"} <= set(rec)


def test_corrupt_shard_is_refused_and_nothing_is_written(tmp_path, rc):
    root = tmp_path / "corpora"
    src = CORPORA / "constitution_advice"
    (root / "constitution_advice").mkdir(parents=True)
    for p in src.iterdir():
        (root / "constitution_advice" / p.name).write_bytes(p.read_bytes())
    shard = root / "constitution_advice" / "part-00.jsonl.gz"
    b = bytearray(shard.read_bytes())
    b[len(b) // 2] ^= 0xFF
    shard.write_bytes(bytes(b))
    out = tmp_path / "out.jsonl"
    with pytest.raises(rc.CorpusCheckError, match="sha256"):
        rc.reassemble("constitution_advice", out, root)
    assert not out.exists()
    # a manifest whose count lies is refused too (the shard is intact here)
    (root / "constitution_advice" / "part-00.jsonl.gz").write_bytes((src / "part-00.jsonl.gz").read_bytes())
    man = json.loads((src / "MANIFEST.json").read_text())
    man["record_count"] += 1
    (root / "constitution_advice" / "MANIFEST.json").write_text(json.dumps(man))
    with pytest.raises(rc.CorpusCheckError, match="records"):
        rc.reassemble("constitution_advice", out, root)
    assert not out.exists()


def test_cli_verify_only_and_list(rc, capsys):
    assert rc.main(["--list"]) == 0
    assert rc.main(["control_advice", "--verify-only"]) == 0
    out = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert out["sha256"] == FINALS["control_advice"][0] and out["records"] == 1995


def test_tokenizer_pins_complete_and_consistent_with_sidecar_manifests():
    pins = json.loads((CORPORA / "TOKENIZERS.json").read_text(encoding="utf-8"))
    assert set(pins["tokenizers"]) == {"llama3", "gemma", "qwen"}
    for t in pins["tokenizers"].values():
        assert t["repo"].count("/") == 1 and len(t["revision"]) == 40
        names = [f["name"] for f in t["files"]]
        assert "tokenizer.json" in names and all(len(f["sha256"]) == 64 for f in t["files"])
    assert "tokenizer.json" in pins["include"]
    # the section-tags manifests record the constitution snapshot the tags were cut against
    snap = json.loads((REPO / "corpus_generator/constitution/SNAPSHOT.json").read_text())
    for corpus in ("constitution_advice", "constitution_stories"):
        m = json.loads((CORPORA / corpus / "section-tags-production-20260831-v2.manifest.json").read_text())
        assert m["constitution_sha256"] == snap["sha256"]
        assert m["source"]["sha256"] == FINALS[corpus][0]


def test_no_personal_identifiers_in_manifests_and_pins():
    for p in list(CORPORA.glob("*/MANIFEST.json")) + [CORPORA / "TOKENIZERS.json"]:
        text = p.read_text(encoding="utf-8")
        assert_no_tokens(text, str(p))
        assert "/home/" not in text, f"{p} carries a home directory"
