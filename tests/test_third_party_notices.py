"""The third-party notices: what the base-model licences ask of a redistribution, and that the notices name real files.

THIRD_PARTY_NOTICES at the root lists every file that comes from another project or derives from a base model, with its
source, licence and attribution line; LICENSES/ holds the full texts. Three base-model licences ask for more than a
listing, and these tests pin each ask:
- Llama 3.1: a copy of the agreement with the copies, "Built with Llama" in the documentation, the exact notice sentence
  in a "Notice" file shipped with the copies, and "Llama" at the start of the name of a model fine-tuned from it
- NVIDIA Nemotron Open Model License: a copy of the licence and the NVIDIA attribution statement
- Apache 2.0 (Qwen3.5, Gemma 4): a copy of the licence and the attribution lines
The per-model NOTICE and licence copies sit in objects/<model_id>/ so that a sparse checkout of any one object brings
them (cone mode fetches the files directly inside each parent directory). The README's lighter checkouts name LICENSES/
too, so every text the notices cite comes with them. The licence texts are third-party text, so
each is pinned by sha256: an edit to one is a defect, never a refresh. Every repository path the notices cite must exist,
so a file removed or renamed later cannot leave a stale entry behind. The G0DM0D3 entry's statement that the template
text is not distributed is checked against the code the entry names, not only printed, and the Qwen 3.8 Max corpus entry
and its terms are checked against the corpus manifests they name. A publication may leave that corpus out (both directories,
as tests/test_corpora_qwen38max.py allows): its entry, its terms file and the README's sentence about them then go with it, and
these tests require exactly that."""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
import re
import subprocess
from pathlib import Path

import pytest

from published_repo import is_published

REPO = Path(__file__).resolve().parents[1]
NOTICES = REPO / "THIRD_PARTY_NOTICES"
LICENSES = REPO / "LICENSES"

LLAMA_NOTICE = ("Llama 3.1 is licensed under the Llama 3.1 Community License, Copyright © Meta Platforms, Inc. "
                "All Rights Reserved.")
NVIDIA_STATEMENT = "Licensed by NVIDIA Corporation under the NVIDIA Nemotron Model License."
GODMODE_STATEMENT = ("The template text is not distributed with this repository. It is fetched at run time from the pinned "
                     "upstream commit under upstream's own licence (AGPL-3.0) and is not covered by this repository's MIT "
                     "licence.")
#: CC BY 4.0 asks for the creators, the licence and an indication of any change, so the attribution for the eight
#: chain-of-thought exemplars of Wei et al. names all three.
WEI_ATTRIBUTION = ('"Chain-of-Thought Prompting Elicits Reasoning in Large Language Models" by Jason Wei, Xuezhi Wang, '
                   "Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed Chi, Quoc Le and Denny Zhou, licensed under "
                   "CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/). The problems and rationales are reproduced "
                   "word for word; only the labels differ, as listed above.")

#: The verbatim licence texts. Apache-2.0.txt is the apache.org LICENSE-2.0.txt; the Nemotron text is NVIDIA's published
#: page (the checkpoint's own LICENSE file is empty at the pinned revision); CC-BY-4.0.txt is creativecommons.org's
#: legalcode.txt; the MIT texts are each upstream's own file.
LICENCE_SHA256 = {
    "Apache-2.0.txt": "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30",
    "Llama-3.1-Community-License.txt": "64e1b2889b7892e6bbe7a7ed5bfe6ff793c61f9d584345f8f41cf9f5cb30a369",
    "Llama-3.1-Acceptable-Use-Policy.md": "a568f2ebc73cec3fd74ba2afd992d4e945a8c7a9d851f9b66163aac834b7b859",
    "NVIDIA-Nemotron-Open-Model-License.txt": "d326877a00381844b1a75c703c762a1cf0d84f11f7335d2046a5b3c44dd8338b",
    "MIT-agentic-misalignment.txt": "5cf118897846e287ec6ce84c7031c153733f75f8d5e40c1a329697cf14469261",
    "MIT-emergent-misalignment.txt": "4fb4678d1be0540c650e39f75eabca626c5727f1a3468aaf7dc844247e1291c2",
    "MIT-petri-bloom.txt": "2ae0e89f534aac0dc7ee9e9b8021a9ce57e7294f443a54aaabff8d99d13a9def",
    "MIT-inspect-petri.txt": "d35ba38d8a6529210884a4f07235dfe9b2de5357d575821f0f2bcbdc80a645b9",
    "CC-BY-4.0.txt": "9ba9550ad48438d0836ddab3da480b3b69ffa0aac7b7878b5a0039e7ab429411",
    # not an upstream text: the repository's own terms for the shipped ReNeLLM corpus (data/heldout_corpora/renellm.jsonl,
    # x-ai/grok-4.5 output, evaluation only), pinned like the others so that any change to the terms is a decision
    "ReNeLLM-corpus-terms.txt": "5e4b5b14318220f0adc2666b066ef7a35d8ca2102dd3786228006da8f85a9a42",
    # not an upstream text either: the repository's own terms for the Qwen 3.8 Max constitution corpus
    # (corpora/constitution_qwen38max_{advice,stories}/, qwen/qwen3.8-max-0902 output: any use but training or developing
    # products or services that compete with those of Alibaba Cloud or its affiliates), pinned the same way so that any
    # change to the terms is a decision
    "Qwen38Max-corpus-terms.txt": "dfced18d877bbf50cb01db3a15633429a2d9e0089e29d25bdef95891b766e69c",
}

QWEN_TERMS = "Qwen38Max-corpus-terms.txt"
#: The corpus entry 14 and LICENSES/Qwen38Max-corpus-terms.txt cover, one directory per part, with the noun each counts its
#: records by.
QWEN_CORPUS = {"constitution_qwen38max_advice": "advice records", "constitution_qwen38max_stories": "stories"}
#: The provider's one restriction (Alibaba Cloud International Product Terms 4.48.1(d)(v)), as entry 14 and the terms state it.
QWEN_RESTRICTION = ("to train or develop products or services that compete with those of Alibaba Cloud or its affiliates, "
                    "unless Alibaba Cloud expressly authorises it")
QWEN_README = ("The Qwen 3.8 Max constitution corpus is under "
               "[LICENSES/Qwen38Max-corpus-terms.txt](LICENSES/Qwen38Max-corpus-terms.txt), which forbids using it to train "
               "or develop products or services that compete with those of Alibaba Cloud or its affiliates.")


def _qwen_corpus_left_out(root: Path = REPO) -> bool:
    """A publication that leaves the Qwen 3.8 Max corpus out: neither directory, in the published repository. A development
    checkout without them is not one (its notice checks fail by name), and neither is a tree carrying one directory."""
    return not any((root / "corpora" / c).is_dir() for c in QWEN_CORPUS) and is_published(root)


def _pinned_licences() -> dict:
    """LICENCE_SHA256, without the Qwen 3.8 Max corpus's terms in a publication that leaves the corpus out."""
    left = _qwen_corpus_left_out()
    return {k: v for k, v in LICENCE_SHA256.items() if not (left and k == QWEN_TERMS)}

#: objects/<model_id>/ -> the licence copies its licence asks to travel with the copies, and the statement its NOTICE carries.
PER_MODEL = {
    "llama8b_base": ({"LICENSE": "Llama-3.1-Community-License.txt", "USE_POLICY.md": "Llama-3.1-Acceptable-Use-Policy.md"},
                     (LLAMA_NOTICE, "Built with Llama.")),
    "llama8b_it": ({"LICENSE": "Llama-3.1-Community-License.txt", "USE_POLICY.md": "Llama-3.1-Acceptable-Use-Policy.md"},
                   (LLAMA_NOTICE, "Built with Llama.")),
    "nemotron3nano4b": ({"LICENSE": "NVIDIA-Nemotron-Open-Model-License.txt"}, (NVIDIA_STATEMENT,)),
}

#: The top-level directories a cited repository path starts with; a token outside them (a URL, an upstream path such as
#: dataset/splits/...) is not a path in this repository.
REPO_ROOTS = ("objects/", "configs/", "src/", "scripts/", "tests/", "corpus_generator/", "system_instructions/", "LICENSES/")
PATH_TOKEN = re.compile(r"(?<![\w/.:@-])((?:%s)[\w@./-]*)" % "|".join(re.escape(r) for r in REPO_ROOTS))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _unwrapped(text: str) -> str:
    return " ".join(text.split())


def cited_paths(text: str) -> set[str]:
    """Every repository path the text cites, trailing sentence punctuation stripped."""
    return {m.group(1).rstrip(".,:;)") for m in PATH_TOKEN.finditer(text)}


def test_the_path_scan_reads_paths_and_skips_urls_and_upstream_paths():
    text = ("Files objects/llama8b_base/ and configs/capeval/petri/prompts/judge.md, with tests/fixtures/x@ab12.jinja:\n"
            "Source https://github.com/a/b @ c (src/lib/libertas.ts upstream) and dataset/splits/harmful_val.json\n"
            "Licence MIT (LICENSES/MIT-petri-bloom.txt).")
    assert cited_paths(text) == {"objects/llama8b_base/", "configs/capeval/petri/prompts/judge.md",
                                 "tests/fixtures/x@ab12.jinja", "src/lib/libertas.ts", "LICENSES/MIT-petri-bloom.txt"}


def test_the_licence_holder_is_safety_research_under_mit():
    lines = (REPO / "LICENSE").read_text(encoding="utf-8").splitlines()
    assert lines[0] == "MIT License"
    assert lines[2] == "Copyright (c) 2026 Safety Research"


def test_every_licence_text_is_the_verbatim_upstream_text():
    pinned = _pinned_licences()
    assert {p.name for p in LICENSES.iterdir()} == set(pinned), "LICENSES/ holds exactly the pinned texts"
    for name, sha in pinned.items():
        assert _sha(LICENSES / name) == sha, f"LICENSES/{name} is not the verbatim upstream text"


def test_the_notices_carry_each_licence_ask_word_for_word():
    text = NOTICES.read_text(encoding="utf-8")
    assert LLAMA_NOTICE in text.splitlines(), "the Llama notice sentence stands on a line of its own, verbatim"
    assert "Built with Llama." in text.splitlines()
    assert '"Llama-3.1-8B"' in text, "the naming the Llama licence asks of fine-tuned models"
    assert NVIDIA_STATEMENT in text.splitlines()
    assert "Copyright 2026 Alibaba Cloud" in text.splitlines()
    assert GODMODE_STATEMENT in _unwrapped(text), "the G0DM0D3 templates are fetched at run time, never distributed"
    assert WEI_ATTRIBUTION in _unwrapped(text), "the CC BY 4.0 attribution for the chain-of-thought exemplars"


def test_every_licence_text_is_cited_and_every_cited_path_exists():
    text = NOTICES.read_text(encoding="utf-8")
    cited = cited_paths(text)
    assert {f"LICENSES/{n}" for n in _pinned_licences()} <= cited, "a licence text nothing in THIRD_PARTY_NOTICES points at"
    missing = sorted(p for p in cited if not p.startswith("src/lib/") and not (REPO / p).exists())
    assert not missing, f"THIRD_PARTY_NOTICES cites paths that do not exist: {missing}"


@pytest.mark.parametrize("model_id", sorted(PER_MODEL))
def test_each_base_model_directory_ships_its_notice_and_licence_copies(model_id):
    copies, statements = PER_MODEL[model_id]
    d = REPO / "objects" / model_id
    for name, text_name in copies.items():
        assert (d / name).read_bytes() == (LICENSES / text_name).read_bytes(), f"objects/{model_id}/{name} != LICENSES/{text_name}"
    notice = (d / "NOTICE").read_text(encoding="utf-8")
    for s in statements:
        assert s in notice.splitlines(), f"objects/{model_id}/NOTICE lacks {s!r} on a line of its own"
    for p in re.findall(r"\.\./[\w./]+", notice):
        assert (d / p.rstrip(".")).exists(), f"objects/{model_id}/NOTICE points at {p}, which does not exist"


def test_the_readmes_display_built_with_llama_the_names_and_the_licences():
    readme = (REPO / "README.md").read_text(encoding="utf-8")
    assert "\n## Third-party material\n" in readme and "\n## Responsible use and content\n" in readme
    assert "[THIRD_PARTY_NOTICES](THIRD_PARTY_NOTICES)" in readme and "Built with Llama." in readme
    # the MIT grant carves out every file under its own terms (the ReNeLLM corpus's no-training terms by name; the Qwen 3.8
    # Max corpus's sentence is qwen_corpus_problems'), and the base-model licences still bind the adapters
    assert ("The MIT licence covers the code, the data and the fitted adapters, except the files "
            "[THIRD_PARTY_NOTICES](THIRD_PARTY_NOTICES) lists under their own terms") in _unwrapped(readme)
    assert ("the ReNeLLM corpus is under [LICENSES/ReNeLLM-corpus-terms.txt](LICENSES/ReNeLLM-corpus-terms.txt), which "
            "forbids using it to train any AI system") in _unwrapped(readme)
    assert "Using the adapters is also subject to the base-model licences listed there." in _unwrapped(readme)
    objects_readme = (REPO / "objects" / "README.md").read_text(encoding="utf-8")
    assert '"Llama-3.1-8B"' in objects_readme and "Built with Llama." in objects_readme
    assert "[THIRD_PARTY_NOTICES](../THIRD_PARTY_NOTICES)" in objects_readme


@pytest.mark.parametrize("option", ["figures", "demo"])
def test_each_lighter_checkout_brings_every_licence_text_the_notices_cite(option, tmp_path):
    """The README's lighter checkouts are cone-mode sparse checkouts of `reproduce/tools` plus the directories
    reproduce/tools/sparse_paths.py prints. Cone mode brings the root files, THIRD_PARTY_NOTICES among them, but no
    directory the list leaves out, so the texts the notices point at arrive only when the list names LICENSES/. Checked
    with git on a stand-in tree of the same shape: a file in each listed directory, every licence text the notices cite,
    and one directory the list leaves out, which has to stay out."""
    spec = importlib.util.spec_from_file_location("sparse_paths", REPO / "reproduce" / "tools" / "sparse_paths.py")
    sparse_paths = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sparse_paths)
    listed = ["reproduce/tools", *sparse_paths.paths(option, str(REPO))]
    texts = sorted(p for p in cited_paths(NOTICES.read_text(encoding="utf-8"))
                   if p.startswith("LICENSES/") and not p.endswith("/"))   # the texts, not the directory the preamble names
    assert texts
    clone = tmp_path / "clone"
    for p in ("THIRD_PARTY_NOTICES", "left_out/file", *texts, *(f"{d}/file" for d in listed)):
        (clone / p).parent.mkdir(parents=True, exist_ok=True)
        (clone / p).write_text(f"{p}\n", encoding="utf-8")
    env = dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@example.invalid", GIT_COMMITTER_NAME="t",
               GIT_COMMITTER_EMAIL="t@example.invalid")
    for cmd in (["init", "-q"], ["add", "-A"], ["-c", "commit.gpgsign=false", "commit", "-q", "-m", "stand-in"],
                ["sparse-checkout", "set", "--cone", *listed]):
        subprocess.run(["git", *cmd], cwd=clone, check=True, env=env, capture_output=True)
    assert (clone / "THIRD_PARTY_NOTICES").is_file() and not (clone / "left_out").exists(), "the checkout is not sparse"
    missing = [p for p in texts if not (clone / p).is_file()]
    assert not missing, f"the {option} checkout leaves out licence texts THIRD_PARTY_NOTICES cites: {missing}"


def _entry(text: str, word: str) -> str:
    """The body of the one THIRD_PARTY_NOTICES entry whose numbered title holds `word`."""
    parts = re.split(r"^={20,}\n(\d+\. [^\n]*)\n={20,}\n", text, flags=re.M)
    bodies = [body for title, body in zip(parts[1::2], parts[2::2]) if word in title]
    assert len(bodies) == 1, f"{len(bodies)} THIRD_PARTY_NOTICES entries have {word!r} in their title"
    return bodies[0]


#: A bare slot placeholder such as {QUERY} is a spelling the code has to name (it is also one template's whole user
#: turn, so its hash is pinned), not template text.
SLOT = re.compile(r"\{\w+\}")


def test_the_godmode_entry_holds_for_the_code_it_names():
    """The G0DM0D3 entry says the template text is fetched at run time and not distributed. That is a claim about the
    code, so it is checked there: the module the entry names fetches exactly the upstream commit and file the entry
    prints, and no file the entry names holds a string whose sha256 is one of the module's pinned template hashes (the
    hashes are the only trace of the text the code may keep). A tree whose module still carries the templates fails."""
    from sisteer.eval import attacks_godmode as agm

    pins = ("UPSTREAM_REPO", "UPSTREAM_COMMIT", "UPSTREAM_PATH", "FILE_SHA256", "TEMPLATES")
    absent = [n for n in pins if not hasattr(agm, n)]
    assert not absent, f"sisteer.eval.attacks_godmode has no {absent}: it is not the module that fetches the templates"
    entry = _entry(NOTICES.read_text(encoding="utf-8"), "G0DM0D3")
    for pin in (agm.UPSTREAM_REPO, agm.UPSTREAM_COMMIT, agm.UPSTREAM_PATH, agm.FILE_SHA256):
        assert pin in entry, f"the G0DM0D3 entry does not print the module's pin {pin}"
    pinned = {v for t in agm.TEMPLATES.values() for k, v in t.items() if k.endswith("_sha256")}
    assert len(pinned) == 6, "two templates, each pinned by its system, user and folded hashes"
    files = sorted(p for p in cited_paths(entry) if p.endswith(".py"))
    assert files == ["src/sisteer/eval/attacks/grok420.py", "src/sisteer/eval/attacks_godmode.py"]
    for f in files:
        held = [n.lineno for n in ast.walk(ast.parse((REPO / f).read_text(encoding="utf-8")))
                if isinstance(n, ast.Constant) and isinstance(n.value, str) and not SLOT.fullmatch(n.value)
                and hashlib.sha256(n.value.encode("utf-8")).hexdigest() in pinned]
        assert not held, f"{f} holds template text at line(s) {held}, which the entry says is not distributed"


def qwen_corpus_problems(notices: str, licenses: Path, readme: str, corpora: Path, left_out: bool) -> list:
    """Every way the tree's notice of the Qwen 3.8 Max corpus breaks its rule (empty = it holds). With the corpus in the tree:
    exactly one THIRD_PARTY_NOTICES entry titled for it, which with LICENSES/Qwen38Max-corpus-terms.txt grants any use but the
    provider's one restriction (QWEN_RESTRICTION) and identifies the corpus by each part's directory, its record count under the
    part's own noun, and the file name and sha256 of its reassembled final, all read from the parts' MANIFEST.json (so a corpus
    re-issued under the same directories, or two counts swapped between the parts, cannot keep a stale notice), and the README's
    licence sentence (QWEN_README). Left out by a publication: no such entry, no citation of the terms file, no terms file and no
    README pointer to it."""
    titles = [t for t in re.findall(r"^={20,}\n(\d+\. [^\n]*)\n={20,}\n", notices, flags=re.M) if "Qwen 3.8 Max" in t]
    readme_u = _unwrapped(readme)
    if left_out:
        return [msg for bad, msg in (
            (titles, "THIRD_PARTY_NOTICES keeps the Qwen 3.8 Max corpus entry the release left the corpus out of"),
            (f"LICENSES/{QWEN_TERMS}" in notices, f"THIRD_PARTY_NOTICES still cites LICENSES/{QWEN_TERMS}"),
            ((licenses / QWEN_TERMS).exists(), f"LICENSES/{QWEN_TERMS} is still in the tree"),
            (QWEN_TERMS in readme_u, f"README.md still points at LICENSES/{QWEN_TERMS}")) if bad]
    if len(titles) != 1:
        return [f"{len(titles)} THIRD_PARTY_NOTICES entries have 'Qwen 3.8 Max' in their title, not one"]
    entry = _unwrapped(_entry(notices, "Qwen 3.8 Max"))
    terms_path = licenses / QWEN_TERMS
    if not terms_path.is_file():
        return [f"LICENSES/{QWEN_TERMS} is missing"]
    terms = _unwrapped(terms_path.read_text(encoding="utf-8"))
    p = [] if QWEN_README in readme_u else ["README.md does not carry the Qwen 3.8 Max corpus's licence sentence"]
    for where, text in (("THIRD_PARTY_NOTICES entry 14", entry), (f"LICENSES/{QWEN_TERMS}", terms)):
        if "for any purpose" not in text:
            p.append(f"{where} does not grant 'for any purpose'")
        if QWEN_RESTRICTION not in text:
            p.append(f"{where} does not state the restriction {QWEN_RESTRICTION!r}")
    for corpus, noun in QWEN_CORPUS.items():
        manifest = corpora / corpus / "MANIFEST.json"
        if not manifest.is_file():
            p.append(f"THIRD_PARTY_NOTICES entry 14 names corpora/{corpus}/, which this tree lacks")
            continue
        m = json.loads(manifest.read_text(encoding="utf-8"))
        n = m["record_count"]
        for where, text, wants in (("THIRD_PARTY_NOTICES entry 14", entry, (f"corpora/{corpus}/", f"{n} {noun}")),
                                   (f"LICENSES/{QWEN_TERMS}", terms, (f"corpora/{corpus}/ ({n} records)", f"{n} {noun}"))):
            p += [f"{where} does not carry {w!r} from corpora/{corpus}/MANIFEST.json"
                  for w in wants + (m["source_file"], m["sha256_reassembled"]) if w not in text]
    return p


def _tree() -> dict:
    return dict(notices=NOTICES.read_text(encoding="utf-8"), licenses=LICENSES, readme=(REPO / "README.md").read_text(encoding="utf-8"),
                corpora=REPO / "corpora")


def test_the_qwen_corpus_entry_and_its_terms_name_the_corpus_its_manifests_describe():
    assert qwen_corpus_problems(**_tree(), left_out=_qwen_corpus_left_out()) == []


def test_the_qwen_corpus_rule_catches_a_stale_notice_and_holds_for_a_release_that_leaves_the_corpus_out(tmp_path):
    """The rule on altered copies of this tree: each stale notice is named, and a release that leaves the corpus out passes once
    its entry, its terms file and the README's sentence are gone with it (and is named while any of them stays)."""
    t = _tree()
    if _qwen_corpus_left_out():
        pytest.skip("this tree is a publication that left the Qwen 3.8 Max corpus out; the rule above is checked on it as it is")
    entry = _entry(t["notices"], "Qwen 3.8 Max")
    terms = (LICENSES / QWEN_TERMS).read_text(encoding="utf-8")

    def problems(notices=t["notices"], terms_text=terms, readme=t["readme"], left_out=False):
        lic = tmp_path / f"lic{len(list(tmp_path.iterdir()))}"
        lic.mkdir()
        if terms_text is not None:
            (lic / QWEN_TERMS).write_text(terms_text, encoding="utf-8")
        return qwen_corpus_problems(notices, lic, readme, t["corpora"], left_out)

    assert problems() == []
    swapped = t["notices"].replace(entry, entry.replace("197 advice records and 736 stories", "736 advice records and 197 stories"))
    assert any("'197 advice records'" in x for x in problems(notices=swapped))
    unbanned = t["notices"].replace(entry, re.sub(r"except that\s+they may not be used.*?authorises it\.", "", entry, flags=re.S))
    assert unbanned != t["notices"] and any("does not state the restriction" in x for x in problems(notices=unbanned))
    stale = terms.replace("(736 records)", "(735 records)")
    assert stale != terms and any("(736 records)" in x for x in problems(terms_text=stale))
    narrow = terms.replace("for any purpose,", "for research and evaluation,")
    assert narrow != terms and any("does not grant 'for any purpose'" in x for x in problems(terms_text=narrow))
    assert any("README.md" in x for x in problems(readme=t["readme"].replace("Alibaba Cloud or its affiliates.", "Alibaba Cloud's.")))
    # a release that leaves the corpus out: entry 14 and its terms go, and the README's sentence with them
    cut = re.split(r"\n+={20,}\n14\. Qwen 3\.8 Max[^\n]*\n={20,}\n", t["notices"])[0] + "\n"
    readme_cut = re.sub(r" The Qwen 3\.8 Max constitution corpus is under\s+\[LICENSES/Qwen38Max-corpus-terms\.txt\]"
                        r"\(LICENSES/Qwen38Max-corpus-terms\.txt\),[^.]*\.", "", t["readme"])
    assert cut != t["notices"] and readme_cut != t["readme"]
    assert problems(notices=cut, terms_text=None, readme=readme_cut, left_out=True) == []
    kept = problems(left_out=True)
    assert len(kept) == 4 and any("keeps the Qwen 3.8 Max corpus entry" in x for x in kept)
