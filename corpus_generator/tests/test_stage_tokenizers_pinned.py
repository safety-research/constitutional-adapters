"""``corpusgen stage-tokenizers --out/--hf-home/--revision``: an explicit
destination, an explicit cache, and PINNED snapshots (a reproduction stages
the exact HF commits the corpus manifest names; a pinned revision that is not
cached refuses instead of substituting the newest cached snapshot)."""

from __future__ import annotations

import hashlib
import json
import os
import time

import pytest
from click.testing import CliRunner
from tokenizers import Tokenizer, models, pre_tokenizers

from corpus_generator import tokens as T
from corpus_generator.cli import main as cli_main

FAMILIES = {name: repos[0] for name, repos in T.REFERENCE_TOKENIZERS}


def _toy_tokenizer(vocab_words) -> Tokenizer:
    vocab = {w: i for i, w in enumerate(vocab_words)}
    vocab["[UNK]"] = len(vocab)
    tok = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    return tok


def _fake_cache(root, snapshots: dict[str, list[str]], newest_last: bool = True):
    """An HF-cache layout ``hub/models--<org>--<name>/snapshots/<sha>/tokenizer.json``
    with distinguishable toy tokenizers; returns {family: {sha: tokenizer.json sha256}}."""
    hub = root / "hub"
    shas: dict[str, dict[str, str]] = {}
    for fam, revs in snapshots.items():
        repo = FAMILIES[fam]
        for i, rev in enumerate(revs):
            d = hub / f"models--{repo.replace('/', '--')}" / "snapshots" / rev
            d.mkdir(parents=True)
            _toy_tokenizer([f"{fam}{rev[:4]}w{k}" for k in range(3)]).save(str(d / "tokenizer.json"))
            (d / "tokenizer_config.json").write_text(json.dumps({"family": fam, "rev": rev}))
            shas.setdefault(fam, {})[rev] = hashlib.sha256((d / "tokenizer.json").read_bytes()).hexdigest()
            # make mtime order deterministic: later entries are newer
            t = time.time() - (0 if newest_last else 1000) + i
            os.utime(d, (t, t))
    return shas


PINS = {"llama3": "a" * 40, "gemma": "b" * 40, "qwen": "c" * 40}
OTHER = {"llama3": "1" * 40, "gemma": "2" * 40, "qwen": "3" * 40}


def test_pinned_revisions_are_staged_exactly(tmp_path):
    # every family has TWO cached snapshots; the OTHER one is newer by mtime
    shas = _fake_cache(tmp_path / "cache", {f: [PINS[f], OTHER[f]] for f in PINS})
    out = tmp_path / "staged"
    man = T.stage_tokenizers(out, hf_hub=tmp_path / "cache" / "hub", revisions=PINS)
    for fam, pin in PINS.items():
        assert man["tokenizers"][fam]["snapshot"] == pin
        assert man["tokenizers"][fam]["pinned"] is True
        assert man["tokenizers"][fam]["files"]["tokenizer.json"] == shas[fam][pin]
        assert (out / fam / "tokenizer.json").exists()
    # and the counter loads the staged set (hash check passes)
    tc = T.TokenCounter.load(out)
    assert set(tc.tokenizers) == set(PINS)


def test_unpinned_takes_newest_by_mtime(tmp_path):
    shas = _fake_cache(tmp_path / "cache", {f: [PINS[f], OTHER[f]] for f in PINS})
    man = T.stage_tokenizers(tmp_path / "staged", hf_hub=tmp_path / "cache" / "hub")
    for fam in PINS:
        assert man["tokenizers"][fam]["snapshot"] == OTHER[fam]
        assert man["tokenizers"][fam]["pinned"] is False


def test_missing_pinned_revision_refuses_never_substitutes(tmp_path):
    _fake_cache(tmp_path / "cache", {f: [OTHER[f]] for f in PINS})  # the pins are NOT cached
    with pytest.raises(T.StagingError, match="pinned revision"):
        T.stage_tokenizers(tmp_path / "staged", hf_hub=tmp_path / "cache" / "hub", revisions=PINS)
    assert not (tmp_path / "staged" / "MANIFEST.json").exists()


def test_unknown_family_in_revisions_refuses(tmp_path):
    _fake_cache(tmp_path / "cache", {f: [PINS[f]] for f in PINS})
    with pytest.raises(T.StagingError, match="unknown reference tokenizer"):
        T.stage_tokenizers(tmp_path / "staged", hf_hub=tmp_path / "cache" / "hub", revisions={"mistral": "x" * 40})


def test_resolve_hf_hub_accepts_hf_home_or_hub(tmp_path, monkeypatch):
    assert T.resolve_hf_hub(tmp_path / "hfhome") == tmp_path / "hfhome" / "hub"
    assert T.resolve_hf_hub(tmp_path / "hfhome" / "hub") == tmp_path / "hfhome" / "hub"
    monkeypatch.setenv("HF_HOME", str(tmp_path / "envhome"))
    assert T.resolve_hf_hub() == tmp_path / "envhome" / "hub"
    monkeypatch.delenv("HF_HOME")
    assert T.resolve_hf_hub().name == "hub"


def test_cli_out_hf_home_revision(tmp_path):
    shas = _fake_cache(tmp_path / "cache", {f: [PINS[f], OTHER[f]] for f in PINS})
    out = tmp_path / "dest"
    args = ["stage-tokenizers", "--out", str(out), "--hf-home", str(tmp_path / "cache")]
    for fam, pin in PINS.items():
        args += ["--revision", f"{fam}={pin}"]
    res = CliRunner().invoke(cli_main, args)
    assert res.exit_code == 0, res.output
    man = json.loads((out / "MANIFEST.json").read_text())
    assert {f: man["tokenizers"][f]["snapshot"] for f in PINS} == PINS
    assert json.loads(res.output)["tokenizers"]["qwen"]["files"]["tokenizer.json"] == shas["qwen"][PINS["qwen"]]


def test_cli_bad_revision_spec_and_missing_pin_exit_nonzero(tmp_path):
    _fake_cache(tmp_path / "cache", {f: [OTHER[f]] for f in PINS})
    res = CliRunner().invoke(cli_main, ["stage-tokenizers", "--out", str(tmp_path / "d"),
                                        "--hf-home", str(tmp_path / "cache"), "--revision", "llama3"])
    assert res.exit_code != 0 and "NAME=SHA" in res.output
    res = CliRunner().invoke(cli_main, ["stage-tokenizers", "--out", str(tmp_path / "d"),
                                        "--hf-home", str(tmp_path / "cache"), "--revision", "llama3=" + PINS["llama3"]])
    assert res.exit_code != 0 and "pinned revision" in res.output


# ----------------------------------------------------------------------------- --pins: repo + revision + every file sha256
def _pins_for(cache_root, snapshots: dict[str, str]) -> dict:
    """A pins document (the harness's corpora/TOKENIZERS.json shape) describing what the fake cache holds at `snapshots`."""
    fams = {}
    for fam, rev in snapshots.items():
        d = cache_root / "hub" / f"models--{FAMILIES[fam].replace('/', '--')}" / "snapshots" / rev
        files = [{"name": f, "sha256": hashlib.sha256((d / f).read_bytes()).hexdigest()} for f in T.STAGE_FILES if (d / f).exists()]
        fams[fam] = {"repo": FAMILIES[fam], "revision": rev, "files": files}
    return {"tokenizers": fams}


def _write_pins(path, doc) -> dict:
    path.write_text(json.dumps(doc, indent=1))
    return T.load_pins(path)


def test_pins_file_stages_and_attests_the_exact_files(tmp_path):
    """`pins` pins every family (no --revision needed) and the manifest records what it was verified against."""
    _fake_cache(tmp_path / "cache", {f: [PINS[f], OTHER[f]] for f in PINS})
    pins = _write_pins(tmp_path / "TOKENIZERS.json", _pins_for(tmp_path / "cache", PINS))
    out = tmp_path / "staged"
    man = T.stage_tokenizers(out, hf_hub=tmp_path / "cache" / "hub", pins=pins)
    for fam, pin in PINS.items():
        assert man["tokenizers"][fam]["snapshot"] == pin and man["tokenizers"][fam]["pinned"] is True
        assert man["tokenizers"][fam]["files"] == {f["name"]: f["sha256"] for f in pins["tokenizers"][fam]["files"]}
    assert man["verified_against"] == {"pins_file": "TOKENIZERS.json", "sha256": pins["_sha256"], "families": sorted(PINS)}
    assert json.loads((out / "MANIFEST.json").read_text())["verified_against"]["sha256"] == pins["_sha256"]
    T.TokenCounter.load(out)


def test_pins_file_refuses_other_content_under_the_pinned_revision(tmp_path):
    """The correct revision path holding a stale/tampered tokenizer.json (a shared HF cache) must NOT be staged and
    self-attested — the sha256 check refuses BEFORE any file is copied, so no MANIFEST.json and no staged file appears."""
    _fake_cache(tmp_path / "cache", {f: [PINS[f]] for f in PINS})
    pins = _write_pins(tmp_path / "TOKENIZERS.json", _pins_for(tmp_path / "cache", PINS))
    gemma = tmp_path / "cache" / "hub" / f"models--{FAMILIES['gemma'].replace('/', '--')}" / "snapshots" / PINS["gemma"]
    _toy_tokenizer(["tampered", "content"]).save(str(gemma / "tokenizer.json"))   # same path, other bytes
    with pytest.raises(T.StagingError, match=r"'gemma'.*tokenizer\.json.*hashes .* but the pins file records"):
        T.stage_tokenizers(tmp_path / "staged", hf_hub=tmp_path / "cache" / "hub", pins=pins)
    assert not (tmp_path / "staged").exists(), "a refused staging leaves the destination untouched"


def test_pins_file_refuses_a_missing_or_extra_pinned_file_and_a_disagreeing_revision(tmp_path):
    _fake_cache(tmp_path / "cache", {f: [PINS[f]] for f in PINS})
    doc = _pins_for(tmp_path / "cache", PINS)
    # the snapshot holds a file the pins do not list (special_tokens_map.json appears in the cache)
    qwen = tmp_path / "cache" / "hub" / f"models--{FAMILIES['qwen'].replace('/', '--')}" / "snapshots" / PINS["qwen"]
    (qwen / "special_tokens_map.json").write_text("{}")
    with pytest.raises(T.StagingError, match=r"'qwen'.*file set the pins do not attest"):
        T.stage_tokenizers(tmp_path / "s1", hf_hub=tmp_path / "cache" / "hub", pins=_write_pins(tmp_path / "p1.json", doc))
    (qwen / "special_tokens_map.json").unlink()
    # --revision disagreeing with the pins file: one source of truth
    with pytest.raises(T.StagingError, match="disagrees with the pins file"):
        T.stage_tokenizers(tmp_path / "s2", hf_hub=tmp_path / "cache" / "hub", revisions={"llama3": OTHER["llama3"]},
                           pins=_write_pins(tmp_path / "p2.json", doc))
    # the pins name a revision the cache does not hold: the existing refusal, never a substitution
    doc["tokenizers"]["llama3"]["revision"] = OTHER["llama3"]
    with pytest.raises(T.StagingError, match="pinned revision"):
        T.stage_tokenizers(tmp_path / "s3", hf_hub=tmp_path / "cache" / "hub", pins=_write_pins(tmp_path / "p3.json", doc))
    assert not any((tmp_path / d).exists() for d in ("s1", "s2", "s3"))


def test_load_pins_refuses_a_malformed_file(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text(json.dumps({"tokenizers": {"llama3": {"repo": "meta-llama/x", "revision": "short", "files": []}}}))
    with pytest.raises(T.StagingError, match="not .*repo, revision"):
        T.load_pins(p)
    p.write_text(json.dumps({"tokenizers": {"mistral": {"repo": "a/b", "revision": "a" * 40, "files": [{"name": "tokenizer.json", "sha256": "0" * 64}]}}}))
    with pytest.raises(T.StagingError, match="known reference tokenizer"):
        T.load_pins(p)
    p.write_text("not json")
    with pytest.raises(T.StagingError, match="unreadable"):
        T.load_pins(p)


def test_cli_pins_flag(tmp_path):
    _fake_cache(tmp_path / "cache", {f: [PINS[f], OTHER[f]] for f in PINS})
    pins_path = tmp_path / "TOKENIZERS.json"
    _write_pins(pins_path, _pins_for(tmp_path / "cache", PINS))
    out = tmp_path / "dest"
    res = CliRunner().invoke(cli_main, ["stage-tokenizers", "--out", str(out), "--hf-home", str(tmp_path / "cache"), "--pins", str(pins_path)])
    assert res.exit_code == 0, res.output
    man = json.loads((out / "MANIFEST.json").read_text())
    assert {f: man["tokenizers"][f]["snapshot"] for f in PINS} == PINS and man["verified_against"]["pins_file"] == "TOKENIZERS.json"
    # tamper after the pins were taken: the CLI refuses non-zero and writes nothing
    llama = tmp_path / "cache" / "hub" / f"models--{FAMILIES['llama3'].replace('/', '--')}" / "snapshots" / PINS["llama3"]
    (llama / "tokenizer_config.json").write_text(json.dumps({"family": "llama3", "rev": "tampered"}))
    res = CliRunner().invoke(cli_main, ["stage-tokenizers", "--out", str(tmp_path / "dest2"), "--hf-home", str(tmp_path / "cache"), "--pins", str(pins_path)])
    assert res.exit_code != 0 and "tokenizer_config.json" in res.output and "pins file records" in res.output
    assert not (tmp_path / "dest2").exists()
