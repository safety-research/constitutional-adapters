"""sisteer.data.arditi_fit_prompts: the paper's refusal-direction fit prompts re-drawn byte for byte from the pinned upstream splits.

Every draw here is from the invented stand-in upstream (tests/arditi_fit_prompts_standin.py) over a fake network, and the
expected sets and document are computed by the stand-in independently of the module (upstream's module-level sampler), so
nothing reaches upstream and the module's draw is checked against a second implementation. The committed manifest is held to
the original document's identity (literal hashes below, and the fit record the committed objects name), and the repository is
held to carrying no instruction text. ``SISTEER_ARDITI_LIVE=1`` opts into the one test that re-draws from the real upstream.
"""
from __future__ import annotations

import json
import os
import random
import subprocess
import sys
from pathlib import Path

import pytest

import arditi_fit_prompts_standin as st
from sisteer.data import arditi_fit_prompts as afp

REPO = Path(__file__).resolve().parents[1]
WRAPPER = REPO / "scripts" / "arditi" / "reproduce_fit_prompts.py"

#: the original document's identity, as its own fields record it (written out here, independently of the manifest)
DOCUMENT_SHA256 = "f2f0decfd76a9ab0d8d00d4fc46644f91b75ee4fc79f79f225f9486eb221c70b"
UPSTREAM_COMMIT = "9d852fae1a9121c78b29142de733cb1340770cc3"
SOURCE_SHA256 = {
    "harmful_train": "8f5c0eac0efd2a7f99084bbe8d0de2c465e31b1997184783c917969d9de9ece1",
    "harmful_val": "305f1d1e6dfa6c50a32d24a18ef815f42b5441eb83e6d7767d242107162fd9f4",
    "harmless_train": "86623b1f8a25aa35df153fc97a556dbcebb6a7c881538ae43ee479ca17f2e002",
    "harmless_val": "772010758e7d771ef4c7e5e4acdfd7598dcece1a6f383f20d382f640913a2a4d",
}
SET_SHA256 = {
    "harmful_train": "d7558575078bd88338e142f3a326b89b4559b2c8b656dc2441f35b7074cc8e58",
    "harmful_val": "fe2529212f61e81a8bcfa5a20b14afe2b45864032df1e631f1a5ed4661a46318",
    "harmless_train": "d446191ebc6515dc3a0f76b72b3c9ac8310e15df8d6fab69eac6ad09bf4b1bdd",
    "harmless_val": "61b65a0b148b85ca7a0933c6b67b609210b11a46576435ea00e4687dbb098a3e",
}
FIT_RECORD_SHA256 = "05fba693e573a65b8cd6549918000be7e75cbf56b34b02672123b81083512901"


@pytest.fixture
def standin(tmp_path, monkeypatch):
    """The stand-in manifest as the module's default, an EMPTY third-party root, and a network that refuses."""
    path = st.write_manifest(tmp_path / "manifest.json")
    monkeypatch.setattr(afp, "MANIFEST_PATH", path)
    root = tmp_path / "third_party"
    monkeypatch.setenv(afp.THIRD_PARTY_DIR_ENV, str(root))
    monkeypatch.delenv(afp.OFFLINE_ENV, raising=False)
    monkeypatch.setattr(afp, "_open_url", st.refuse_network)
    return {"manifest": path, "root": root}


def _files(root: Path) -> list[str]:
    return sorted(str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()) if root.exists() else []


def _doc_rel() -> str:
    return f"refusal_direction/{st.COMMIT}/fit_prompts.json"


def _src_rels() -> list[str]:
    return sorted(f"refusal_direction/{st.COMMIT}/dataset/splits/{n}.json" for n in st.NAMES)


# --------------------------------------------------------------------------- the committed manifest and the tree

def test_the_committed_manifest_is_the_original_documents_identity():
    m = afp.load_manifest()                                       # validates: every stated-twice fact agrees with its twin
    assert afp.MANIFEST_PATH == REPO / "data" / "arditi" / "fit_prompts.manifest.json"
    assert m["document"]["sha256"] == DOCUMENT_SHA256 and m["document"]["bytes"] == 26140
    assert m["upstream"] == {"repo": "andyrdt/refusal_direction", "commit": UPSTREAM_COMMIT, "licence": "Apache-2.0"}
    assert {n: s["sha256"] for n, s in m["sources"].items()} == SOURCE_SHA256
    assert {n: s["rows"] for n, s in m["sources"].items()} == {"harmful_train": 260, "harmful_val": 39, "harmless_train": 18793,
                                                               "harmless_val": 6264}
    assert {n: s["sha256"] for n, s in m["sets"].items()} == SET_SHA256
    assert m["draw"]["seed"] == 42 and m["draw"]["order"] == ["harmful_train", "harmless_train", "harmful_val", "harmless_val"]
    assert m["draw"]["n"] == {"harmful_train": 128, "harmless_train": 128, "harmful_val": 32, "harmless_val": 32}
    assert m["used_by"]["fit_record_sha256"] == FIT_RECORD_SHA256
    assert afp.document_relpath(m).as_posix() == f"refusal_direction/{UPSTREAM_COMMIT}/fit_prompts.json"
    # the source URLs are the pinned commit's raw files, then Software Heritage by each file's git blob id
    assert afp.source_urls(m, "harmful_train")[0] == (
        f"https://raw.githubusercontent.com/andyrdt/refusal_direction/{UPSTREAM_COMMIT}/dataset/splits/harmful_train.json")
    assert afp.source_urls(m, "harmful_train")[1].endswith(f"/content/sha1_git:{m['sources']['harmful_train']['git_blob']}/raw/")


def test_the_fit_record_the_committed_objects_name_is_the_one_the_manifest_names():
    """Every refusal-direction object in objects/qwen35_4b_it/ (paper labels arditi_allpos and arditi) names the fit record whose
    prompts this document is, and THIRD_PARTY_NOTICES pins the same four split files."""
    listing = REPO / "objects" / "MANIFEST.json"
    rows = ([r for r in json.loads(listing.read_text())["objects"] if str(r.get("paper_label", "")).startswith("arditi")]
            if listing.is_file() else [])
    provs = [REPO / "objects" / r["dir"] / "fit_provenance.json" for r in rows]
    provs = [p for p in provs if p.is_file()]
    if not provs:
        pytest.skip("the refusal-direction objects are not in this checkout (a sparse checkout)")
    assert len(rows) == 5 and {_fit_record_of(json.loads(p.read_text())) for p in provs} == {FIT_RECORD_SHA256}
    notices = (REPO / "THIRD_PARTY_NOTICES").read_text(encoding="utf-8")
    assert all(sha in notices for sha in SOURCE_SHA256.values()) and UPSTREAM_COMMIT in notices


def _fit_record_of(prov: dict) -> str | None:
    """The fit record sha an object's fit_provenance.json names, wherever its writer nested it."""
    stack = [prov]
    while stack:
        d = stack.pop()
        if isinstance(d, dict):
            if isinstance(d.get("fit_record_sha256"), str):
                return d["fit_record_sha256"]
            stack.extend(d.values())
    return None


def test_the_repository_carries_no_instruction_text():
    """data/arditi/ commits the manifest alone (git ignores everything else there, so a reproduced copy cannot be committed),
    and the manifest records the document's fields without its prompts."""
    m = json.loads((REPO / afp.MANIFEST_RELPATH).read_text(encoding="utf-8"))
    assert "prompts" not in m["document"]["fields"] and "prompts" not in m
    git = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--is-inside-work-tree"], capture_output=True, text=True)
    if git.returncode != 0 or git.stdout.strip() != "true":
        pytest.skip("not a git checkout (an exported tree): the tracked-file and ignore checks need git")
    tracked = subprocess.run(["git", "-C", str(REPO), "ls-files", "data/arditi"], capture_output=True, text=True, check=True).stdout.split()
    assert tracked == ["data/arditi/fit_prompts.manifest.json"], tracked
    for rel in ("data/arditi/fit_prompts.json", "data/arditi/anything_else.json"):
        assert subprocess.run(["git", "-C", str(REPO), "check-ignore", "-q", rel]).returncode == 0, rel
    assert subprocess.run(["git", "-C", str(REPO), "check-ignore", "-q", afp.MANIFEST_RELPATH]).returncode == 1
    assert not subprocess.run(["git", "-C", str(REPO), "ls-files", "*fit_prompts.json"], capture_output=True, text=True).stdout.strip()


# --------------------------------------------------------------------------- the draw

def test_reproduces_the_document_byte_for_byte_from_the_upstream_and_reuses_the_cache(standin, monkeypatch):
    log: list = []
    monkeypatch.setattr(afp, "_open_url", st.serve(log))
    rep = afp.ensure_document()
    assert rep["status"] == "BUILT" and rep["relpath"] == _doc_rel() and rep["sha256"] == st.manifest()["document"]["sha256"]
    assert Path(rep["path"]).read_bytes() == st.expected_document()                          # the independent implementation's bytes
    assert log == [st.raw_url(n) for n in st.NAMES]                                          # each file once, from the first source
    assert {n: s["status"] for n, s in rep["sources"].items()} == {n: "FETCHED" for n in st.NAMES}
    assert _files(standin["root"]) == sorted([_doc_rel(), *_src_rels()])                      # no temp file left anywhere
    for n in st.NAMES:
        assert (standin["root"] / f"refusal_direction/{st.COMMIT}/dataset/splits/{n}.json").read_bytes() == st.split_bytes(n)
    assert rep["sets"] == {n: {"n": st.n_of(n), "sha256": st.manifest()["sets"][n]["sha256"]} for n in st.NAMES}
    # again: the verified cached document, no network
    monkeypatch.setattr(afp, "_open_url", st.refuse_network)
    assert afp.ensure_document()["status"] == "PRESENT"
    sets, rep = afp.load_sets()
    assert sets == {n: st.expected_sets()[n] for n in st.NAMES} and rep["status"] == "PRESENT"


def test_offline_builds_from_cached_split_files_and_otherwise_refuses_by_name(standin, monkeypatch):
    monkeypatch.setenv(afp.OFFLINE_ENV, "1")
    with pytest.raises(afp.FitPromptsError) as exc:
        afp.ensure_document()
    msg = str(exc.value)
    assert "are not in the cache and cannot be built" in msg and "fetching is off" in msg
    assert afp.FETCH_COMMAND in msg and "reproduce/00_setup.sh --what third_party" in msg and afp.THIRD_PARTY_DIR_ENV in msg
    assert _files(standin["root"]) == []
    for n in st.NAMES:                                                                       # the split files alone, cached
        p = standin["root"] / f"refusal_direction/{st.COMMIT}/dataset/splits/{n}.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(st.split_bytes(n))
    rep = afp.ensure_document()
    assert rep["status"] == "BUILT" and {s["status"] for s in rep["sources"].values()} == {"PRESENT"}
    assert Path(rep["path"]).read_bytes() == st.expected_document()


def test_software_heritage_is_the_second_source(standin, monkeypatch):
    log: list = []
    monkeypatch.setattr(afp, "_open_url", st.serve(log, "github_down"))
    rep = afp.ensure_document()
    assert rep["status"] == "BUILT" and Path(rep["path"]).read_bytes() == st.expected_document()
    assert log == [u for n in st.NAMES for u in (st.raw_url(n), st.swh_url(n))]
    assert {s["source"] for s in rep["sources"].values()} == {st.swh_url(n) for n in st.NAMES}


@pytest.mark.parametrize("mode,why", [("tampered", "is not the pinned file"), ("down", "could not be fetched from any source")])
def test_a_tampered_or_unreachable_upstream_writes_nothing(standin, monkeypatch, mode, why):
    monkeypatch.setattr(afp, "_open_url", st.serve([], mode))
    with pytest.raises(afp.FitPromptsError, match=why):
        afp.ensure_document()
    assert _files(standin["root"]) == []


def test_a_redirect_or_url_off_the_allowed_hosts_is_refused():
    with pytest.raises(afp.FitPromptsError, match="is not an https URL on"):
        afp._require_allowed("https://example.com/x.json", why="source")
    with pytest.raises(afp.FitPromptsError, match="is not an https URL on"):
        afp._require_allowed("http://raw.githubusercontent.com/x.json", why="source")


def test_a_cached_copy_that_is_not_the_recorded_one_is_refused_and_left_in_place(standin, monkeypatch):
    monkeypatch.setattr(afp, "_open_url", st.serve([]))
    doc = Path(afp.ensure_document()["path"])
    doc.write_bytes(doc.read_bytes().replace(b"Stand-in", b"Stand-ln", 1))                     # one set's text altered
    with pytest.raises(afp.FitPromptsError, match="Remove it and run"):
        afp.ensure_document()
    assert doc.exists()                                                                        # left for the operator
    doc.unlink()
    src = standin["root"] / f"refusal_direction/{st.COMMIT}/dataset/splits/harmful_val.json"
    src.write_bytes(b"[]")
    with pytest.raises(afp.FitPromptsError, match="is not the pinned file"):
        afp.ensure_document()
    assert src.read_bytes() == b"[]" and not doc.exists()


@pytest.mark.parametrize("edit", [
    lambda m: (m["draw"].__setitem__("order", list(reversed(m["draw"]["order"]))),
               m["document"]["fields"]["sampling"].__setitem__("draw_order", list(reversed(m["document"]["fields"]["sampling"]["draw_order"])))),
    lambda m: (m["draw"].__setitem__("seed", 8), m["document"]["fields"]["sampling"].__setitem__("seed", 8)),
], ids=["order", "seed"])
def test_the_seed_and_the_draw_order_are_part_of_the_identity(standin, monkeypatch, edit):
    """A self-consistent manifest that records another order or seed draws other rows, and the set hashes refuse them."""
    monkeypatch.setattr(afp, "_open_url", st.serve([]))
    with pytest.raises(afp.FitPromptsError, match="the draw did not replay"):
        afp.ensure_document(manifest=st.mutated(edit))
    assert not (standin["root"] / _doc_rel()).exists()


def test_the_draw_leaves_the_interpreters_global_generator_alone(standin, monkeypatch):
    monkeypatch.setattr(afp, "_open_url", st.serve([]))
    random.seed(12345)
    before = random.getstate()
    afp.ensure_document()
    assert random.getstate() == before


@pytest.mark.parametrize("edit,why", [
    (lambda m: m.__setitem__("schema", "other/1"), "schema is not"),
    (lambda m: m["document"]["fields"].__setitem__("prompts", {}), "never text"),
    (lambda m: m["sets"]["harmful_val"].__setitem__("sha256", "0" * 64), "disagrees with document.fields.prompts_sha256"),
    (lambda m: m["sources"]["harmless_val"].__setitem__("sha256", "0" * 64), "disagrees with document.fields.source_sha256"),
    (lambda m: m["sources"]["harmful_train"].__setitem__("rows", 12), "disagrees with document.fields.source_rows"),
    (lambda m: m["draw"].__setitem__("seed", 99), "disagrees with draw"),
    (lambda m: m["draw"]["n"].__setitem__("harmful_val", 3), "disagrees with document.fields.sampling"),
    (lambda m: m["upstream"].__setitem__("commit", "f" * 40), "upstream_commit disagrees"),
    (lambda m: m["sources"]["harmful_val"].__setitem__("path", "elsewhere.json"), "sources.harmful_val.path"),
    (lambda m: m["sources"].pop("harmless_val"), "must each name exactly"),
])
def test_a_manifest_that_contradicts_itself_is_refused(edit, why):
    with pytest.raises(afp.FitPromptsError, match=why):
        afp.load_manifest(st.mutated(edit))


# --------------------------------------------------------------------------- the CLI and the script

def test_the_cli_fetches_reproduces_and_verifies_by_hash_alone(standin, monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(afp, "_open_url", st.serve([]))
    assert afp.main(["fetch", "--json"]) == 0
    rep = json.loads(capsys.readouterr().out)
    assert rep["status"] == "BUILT" and rep["relpath"] == _doc_rel() and rep["path"] == str(standin["root"] / _doc_rel())
    out = tmp_path / "copy" / "fit_prompts.json"
    assert afp.main(["reproduce", "--out", str(out), "--offline"]) == 0
    printed = capsys.readouterr().out
    assert out.read_bytes() == st.expected_document() and f"out     {out}" in printed and "PRESENT" in printed
    assert afp.main(["verify", str(out)]) == 0
    printed = capsys.readouterr().out
    assert printed.startswith(f"MATCHES {out}") and printed.count("ok  ") == 4
    other = tmp_path / "other.json"
    other.write_bytes(out.read_bytes().replace(b"Stand-in", b"Stand-ln", 1))           # the first instruction of harmful_train
    assert afp.main(["verify", str(other), "--json"]) == 1
    rep = json.loads(capsys.readouterr().out)
    assert rep["document_matches"] is False and sum(not s["matches"] for s in rep["sets"].values()) == 1
    # an absent document offline: refused on stderr, nothing printed on stdout
    assert afp.main(["fetch", "--offline", "--cache-dir", str(tmp_path / "empty")]) == 2
    cap = capsys.readouterr()
    assert cap.out == "" and cap.err.startswith("REFUSED: ")
    # no instruction text is ever printed
    assert "Stand-in contrast" not in printed + cap.err


def test_the_script_is_the_modules_cli(standin, tmp_path):
    (tmp_path / "doc.json").write_bytes(st.expected_document())
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    r = subprocess.run([sys.executable, str(WRAPPER), "--manifest", str(standin["manifest"]), "verify", str(tmp_path / "doc.json")],
                       capture_output=True, text=True, env=env, timeout=120)
    assert r.returncode == 0 and r.stdout.startswith("MATCHES "), r.stdout + r.stderr
    r = subprocess.run([sys.executable, str(WRAPPER), "--help"], capture_output=True, text=True, env=env, timeout=120)
    assert r.returncode == 0 and all(c in r.stdout for c in ("fetch", "reproduce", "verify")), r.stdout + r.stderr


def test_importing_the_module_reads_nothing_and_touches_no_network(tmp_path):
    code = ("import builtins, socket, sys\n"
            f"sys.path.insert(0, {str(REPO / 'src')!r})\n"
            "def _no(*a, **k): raise AssertionError('read or network at import')\n"
            "socket.socket.connect = _no\n"
            "real_open = builtins.open\n"
            "def _open(f, *a, **k):\n"
            "    if str(f).endswith('fit_prompts.manifest.json'): _no()\n"
            "    return real_open(f, *a, **k)\n"
            "builtins.open = _open\n"
            "import sisteer.data.arditi_fit_prompts as m\n"
            "print(m.PROMPT_SET_PREFIX)\n")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert r.returncode == 0 and r.stdout.strip() == "arditi:", r.stdout + r.stderr


@pytest.mark.skipif(os.environ.get("SISTEER_ARDITI_LIVE") != "1",
                    reason="re-draws from the real upstream over the network; set SISTEER_ARDITI_LIVE=1")
def test_live_the_pinned_upstream_redraws_the_recorded_document(tmp_path, monkeypatch):
    monkeypatch.delenv(afp.OFFLINE_ENV, raising=False)
    rep = afp.ensure_document(tmp_path / "third_party", offline=False)
    assert rep["status"] == "BUILT" and rep["sha256"] == DOCUMENT_SHA256 and rep["bytes"] == 26140
    assert {n: s["sha256"] for n, s in rep["sources"].items()} == SOURCE_SHA256
