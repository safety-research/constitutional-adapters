"""Stage 02's tokenizer provisioning is REPRODUCIBLE from the pins alone:
`corpusgen stage-tokenizers --out DIR --hf-home DIR --pins corpora/TOKENIZERS.json` run against an ISOLATED HF cache (no
~/corpus_generator, no user cache) stages exactly the pinned snapshots named in corpora/TOKENIZERS.json, REFUSES a cached
snapshot under the right revision whose files do not hash to the committed sha256s (a stale or tampered file in a shared
multi-user HF cache), and its MANIFEST.json reproduces the pinned per-file sha256s and
names the pins file it was verified against; a pinned snapshot that is not in the cache refuses too.

Layers: a synthetic cache built from the pins' NAMES (mechanism, incl. the wrong-content refusal against the COMMITTED
pins — always runs when the corpus env exists) and, when the pinned snapshots happen to be in the local HF cache, the REAL
files against the REAL pins (skipped otherwise)."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
CORPUS = REPO / "corpus_generator"
PINS_FILE = REPO / "corpora/TOKENIZERS.json"
PINS = json.loads(PINS_FILE.read_text(encoding="utf-8"))
UV = shutil.which("uv")
CORPUS_PY = Path(os.environ.get("UV_PROJECT_ENVIRONMENT") or REPO / ".venv") / "bin/python"   # the workspace venv the generator lives in

pytestmark = pytest.mark.skipif(
    not CORPUS_PY.exists() or subprocess.run([str(CORPUS_PY), "-c", "import corpus_generator"], capture_output=True).returncode != 0,
    reason="the corpus group is not in the venv (reproduce/01_make_corpora.sh --setup)")


def _corpusgen(*args, env=None, timeout=300):
    cmd = [UV, "run", "--directory", str(CORPUS), "--frozen", "corpusgen", *args] if UV else [str(CORPUS_PY), "-m", "corpus_generator.cli", *args]
    e = {k: v for k, v in os.environ.items() if not k.startswith(("SISTEER_", "HF_", "CORPUS_GENERATOR")) and k != "PYTHONPATH"}
    e.update(env or {})
    return subprocess.run(cmd, env=e, capture_output=True, text=True, timeout=timeout)


def _revision_args() -> list[str]:
    out = []
    for name, t in PINS["tokenizers"].items():
        out += ["--revision", f"{name}={t['revision']}"]
    return out


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _synthetic_cache(hf_home: Path) -> dict[str, Path]:
    """The pinned revision dir of every family, holding synthetic tokenizer files (valid `tokenizers` JSON, NOT the pinned
    bytes) plus a decoy newer snapshot; returns {family: pinned snapshot dir}."""
    dirs = {}
    for name, t in PINS["tokenizers"].items():
        repo_dir = hf_home / "hub" / f"models--{t['repo'].replace('/', '--')}" / "snapshots"
        pinned = repo_dir / t["revision"]
        pinned.mkdir(parents=True)
        for f in t["files"]:
            if f["name"] == "tokenizer.json":
                pinned.joinpath(f["name"]).write_text(json.dumps({"version": "1.0", "model": {"type": "WordLevel", "vocab": {f"{name}-pinned": 0, "[UNK]": 1}, "unk_token": "[UNK]"}, "pre_tokenizer": {"type": "Whitespace"}}))
            else:
                pinned.joinpath(f["name"]).write_text(json.dumps({"family": name, "file": f["name"], "pinned": True}))
        other = repo_dir / ("f" * 40)
        other.mkdir()
        (other / "tokenizer.json").write_text(json.dumps({"version": "1.0", "model": {"type": "WordLevel", "vocab": {f"{name}-decoy": 0, "[UNK]": 1}, "unk_token": "[UNK]"}, "pre_tokenizer": {"type": "Whitespace"}}))
        os.utime(other, None)  # the decoy is the NEWEST snapshot by mtime
        dirs[name] = pinned
    return dirs


def test_pins_name_the_three_families_with_hf_shaped_repos_and_revisions():
    assert set(PINS["tokenizers"]) == {"llama3", "gemma", "qwen"}
    for t in PINS["tokenizers"].values():
        assert t["repo"].count("/") == 1 and len(t["revision"]) == 40 and int(t["revision"], 16) >= 0
        names = {f["name"] for f in t["files"]}
        assert "tokenizer.json" in names and names <= set(PINS["include"])


def test_isolated_cache_stages_exactly_the_pinned_snapshots(tmp_path):
    """A synthetic HF cache with the pinned revision AND a decoy newer snapshot per family: the pins win, a wrong cache
    layout cannot leak in, and the manifest carries the pin + per-file shas of what was staged."""
    hf_home = tmp_path / "hf_home"
    decoy = {}
    for name, t in PINS["tokenizers"].items():
        repo_dir = hf_home / "hub" / f"models--{t['repo'].replace('/', '--')}" / "snapshots"
        pinned = repo_dir / t["revision"]
        pinned.mkdir(parents=True)
        (pinned / "tokenizer.json").write_text(json.dumps({"version": "1.0", "model": {"type": "WordLevel", "vocab": {f"{name}-pinned": 0, "[UNK]": 1}, "unk_token": "[UNK]"}, "pre_tokenizer": {"type": "Whitespace"}}))
        (pinned / "tokenizer_config.json").write_text(json.dumps({"family": name, "pinned": True}))
        other = repo_dir / ("f" * 40)
        other.mkdir()
        (other / "tokenizer.json").write_text(json.dumps({"version": "1.0", "model": {"type": "WordLevel", "vocab": {f"{name}-decoy": 0, "[UNK]": 1}, "unk_token": "[UNK]"}, "pre_tokenizer": {"type": "Whitespace"}}))
        os.utime(other, None)  # the decoy is the NEWEST snapshot by mtime
        decoy[name] = _sha(pinned / "tokenizer.json")
    out = tmp_path / "staged"
    r = _corpusgen("stage-tokenizers", "--out", str(out), "--hf-home", str(hf_home), *_revision_args())
    assert r.returncode == 0, r.stdout[-1500:] + r.stderr[-1500:]
    man = json.loads((out / "MANIFEST.json").read_text())
    for name, t in PINS["tokenizers"].items():
        assert man["tokenizers"][name]["snapshot"] == t["revision"] and man["tokenizers"][name]["pinned"] is True
        assert man["tokenizers"][name]["files"]["tokenizer.json"] == decoy[name] == _sha(out / name / "tokenizer.json")
    # the home directory's cache was never consulted: nothing else exists under the staging dir
    assert {p.name for p in out.iterdir()} == {"MANIFEST.json", "llama3", "gemma", "qwen"}


def test_isolated_cache_missing_pin_refuses(tmp_path):
    hf_home = tmp_path / "hf_home"
    (hf_home / "hub").mkdir(parents=True)  # an empty cache
    r = _corpusgen("stage-tokenizers", "--out", str(tmp_path / "staged"), "--hf-home", str(hf_home), *_revision_args())
    assert r.returncode != 0 and "pinned revision" in (r.stdout + r.stderr)
    assert not (tmp_path / "staged" / "MANIFEST.json").exists()


def test_committed_pins_refuse_other_content_under_the_pinned_revision(tmp_path):
    """The committed corpora/TOKENIZERS.json against a cache whose pinned revision dirs exist with OTHER bytes: the staging
    refuses by family + file + both shas, writes no MANIFEST.json and nothing under --out — the check that a
    correct-revision/wrong-content cache (a stale or tampered shared HF cache) can never be self-attested. Runs on the
    synthetic cache unconditionally (the real-snapshot layer below is opt-in by cache state)."""
    hf_home = tmp_path / "hf_home"
    _synthetic_cache(hf_home)
    out = tmp_path / "staged"
    r = _corpusgen("stage-tokenizers", "--out", str(out), "--hf-home", str(hf_home), "--pins", str(PINS_FILE))
    assert r.returncode != 0, r.stdout[-1500:]
    msg = r.stdout + r.stderr
    assert "pins file records" in msg and "tokenizer.json" in msg and "re-fetch" in msg
    assert "'llama3'" in msg, msg[-800:]        # the first family checked, named
    assert not out.exists(), "a refused staging must leave nothing behind for pools to read"


def test_committed_pins_accept_only_the_pinned_bytes_and_record_the_pins_file(tmp_path):
    """A synthetic pins file whose sha256s ARE the synthetic cache's: --pins alone (no --revision) pins the revisions, the
    staged files hash to the pins, and the manifest names the pins file; combined with --revision the two must agree."""
    hf_home = tmp_path / "hf_home"
    dirs = _synthetic_cache(hf_home)
    doc = json.loads(PINS_FILE.read_text(encoding="utf-8"))
    for name, t in doc["tokenizers"].items():
        t["files"] = [{"name": f["name"], "sha256": _sha(dirs[name] / f["name"])} for f in t["files"]]
    pins = tmp_path / "TOKENIZERS.json"
    pins.write_text(json.dumps(doc))
    out = tmp_path / "staged"
    r = _corpusgen("stage-tokenizers", "--out", str(out), "--hf-home", str(hf_home), "--pins", str(pins))
    assert r.returncode == 0, r.stdout[-1500:] + r.stderr[-1500:]
    man = json.loads((out / "MANIFEST.json").read_text())
    assert man["verified_against"]["pins_file"] == "TOKENIZERS.json" and man["verified_against"]["sha256"] == _sha(pins)
    for name, t in doc["tokenizers"].items():
        assert man["tokenizers"][name]["snapshot"] == t["revision"] and man["tokenizers"][name]["pinned"] is True
        assert man["tokenizers"][name]["files"] == {f["name"]: f["sha256"] for f in t["files"]}
    # a --revision that disagrees with the pins file refuses (one source of truth)
    r = _corpusgen("stage-tokenizers", "--out", str(tmp_path / "s2"), "--hf-home", str(hf_home), "--pins", str(pins), "--revision", "gemma=" + "f" * 40)
    assert r.returncode != 0 and "disagrees with the pins file" in (r.stdout + r.stderr) and not (tmp_path / "s2").exists()


def _local_pins_present() -> Path | None:
    home = Path(os.environ.get("HF_HOME") or Path.home() / ".cache/huggingface")
    for t in PINS["tokenizers"].values():
        d = home / "hub" / f"models--{t['repo'].replace('/', '--')}" / "snapshots" / t["revision"]
        if not (d / "tokenizer.json").exists():
            return None
    return home


@pytest.mark.skipif(_local_pins_present() is None, reason="the pinned tokenizer snapshots are not in the local HF cache")
def test_real_pinned_files_reproduce_the_committed_sha256s(tmp_path):
    hf_home = _local_pins_present()
    out = tmp_path / "staged"
    r = _corpusgen("stage-tokenizers", "--out", str(out), "--hf-home", str(hf_home), "--pins", str(PINS_FILE))
    assert r.returncode == 0, r.stdout[-1500:] + r.stderr[-1500:]
    man = json.loads((out / "MANIFEST.json").read_text())
    assert man["verified_against"]["sha256"] == _sha(PINS_FILE)
    for name, t in PINS["tokenizers"].items():
        staged = man["tokenizers"][name]
        assert staged["snapshot"] == t["revision"] and staged["source_repo"] == t["repo"]
        pinned = {f["name"]: f["sha256"] for f in t["files"]}
        assert staged["files"] == pinned, name
        for fname, sha in pinned.items():
            assert _sha(out / name / fname) == sha
    # and the counter loads the staged set and counts under all three
    code = ("import sys, json; from corpus_generator.tokens import TokenCounter; tc = TokenCounter.load(sys.argv[1]); "
            "print(json.dumps(tc.counts('The quick brown fox jumps over the lazy dog.')))")
    r = subprocess.run([str(CORPUS_PY), "-c", code, str(out)], capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    counts = json.loads(r.stdout)
    assert set(counts) == {"llama3", "gemma", "qwen"} and all(5 <= v <= 20 for v in counts.values())
