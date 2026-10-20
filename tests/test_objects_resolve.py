"""`sisteer.objects`: the human-readable address of a committed paper object.

The address is ``(model_id, paper_label[, dose][, seed][, kind][, derivation])``; the contract that makes it
well-defined is that ``(model_id, paper_label, object_kind, derivation, dose, seed)`` is unique over every row of
``objects/MANIFEST.json``, with dose and seed lifted from the sidecars (``dose_scaled_from.scale`` / the refusal
direction's ``c``; the fit's ``seed``, a difference's parents' seed). The README's worked example resolves the base-fit
LoRA constitutional adapter of seed 1 by address and its served checkpoint through `serve_checkpoint`
(reproduce/models.yaml's serve pin, never a hard-coded revision), and objects/README.md's "Loading" example the ReLU-steer
constitutional adapter of seed 3 the same way, its block run to the load call with the loaders stubbed; a dose-scaled copy
and a seed replicate resolve by dose and seed; an address that still matches several objects, or none, is refused by name
with the candidates listed; every models-table row and every manifest model_id resolves to a (repo, 40-hex revision) on
both sides, an unknown model is refused by name; the CLI round-trips."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from sisteer import objects
from sisteer.front_door_views import models_module
from sisteer.objects import find, fit_checkpoint, load_index, resolve, serve_checkpoint

REPO = Path(__file__).resolve().parents[1]
MANIFEST = REPO / "objects" / "MANIFEST.json"
MODELS_TABLE = REPO / "reproduce" / "models.yaml"
README = REPO / "README.md"
OBJECTS_README = REPO / "objects" / "README.md"

#: The LoRA constitutional adapter for Qwen/Qwen3.5-4B (qwen35_4b_it): the README's worked example until 2026-10-09.
README_OBJECT = "5e9b0009eb24eaf66842b4a2a9f647fced95699c7aae748eaab5333b05a4c6ce"
#: reproduce/models.yaml qwen35_4b: objects fit on the base checkpoint and served on the post-trained one (base-to-instruct
#: transfer), so a qwen35_4b_base object is loaded onto the SERVE pin; the README's worked example reads both from the table.
QWEN_SERVE = ("Qwen/Qwen3.5-4B", "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a")
QWEN_FIT = ("Qwen/Qwen3.5-4B-Base", "1001bb4d826a52d1f399e183466143f4da7b741b")
HEX40 = re.compile(r"^[0-9a-f]{40}$")
#: Its dose-scaled copy at 0.5 (derived_meta.json dose_scaled_from.scale = 0.5, parent README_OBJECT).
README_OBJECT_HALF = "3b5466efd566a05afea662239aab4155a27e0c3c03d04002cff00df415d28f65"
#: objects/README "Loading": the ReLU-steer constitutional adapter on qwen35_4b_base (parents' seed 3), addressed there as
#: resolve("qwen35_4b_base", "difference_relu", seed=3) and loaded onto the serve pin since 2026-10-09 (a literal before).
RELU_DIFF_SEED3 = "0718a3f9edfb033ad84c3322f717dc62b32b6298fe48f0a6c42e9f0a7c031647"
#: qwen35_4b_base constitution_lora: the effective-batch-8 fit (no seed recorded) and the batch-64 seed-2 replicate.
QWEN_CONST_LORA_EB8 = "63c151654c3d9cbdd9540b1806e7db81a6e2434f509ba6657906beccf8762ffd"
QWEN_CONST_LORA_SEED2 = "724fe6ddb52cb679656442b2537057900f3c523ddc3c45b3edde50bdc4a487d2"
#: qwen35_4b_base difference_lora at seed 1 = constitution seed 1 (8be1e1a7) minus control seed 1 (000e0df5): the README's
#: worked example ("Use a fitted adapter") since 2026-10-09, the base-fit LoRA constitutional adapter at dose 1.0.
QWEN_DIFF_LORA_SEED1 = "0d275f7c76cbe63eacaec6b63c0976ac516cff039729029ee5b09b60f3d689c9"
#: gemma4e4b_base constitution_lora dose-scaled copy at 0.25 (train_budget_meta.json dose_scaled_from.scale).
GEMMA_CONST_LORA_Q = "f48ef483bcf00c2876520d2a53245dd581ff109a5f4c3f3e486f68fd7845143e"
#: llama8b_it constitution_lora: committed as a dose_scale derivation at scale 1.0 (its own anchor) — full dose is dose
#: 1.0, not "derivation != dose_scale".
LLAMA_IT_CONST_LORA = "529788ca747ea4060bffee0d46c4ccf1dcf763c548111268547dca0004614c8c"
#: qwen35_4b_it arditi_allpos: the raw-coefficient rungs; c = 0.3 is the one the paper reads.
ARDITI_C03 = "ba2212f0c2f9ffef2997bdd24831182052560cf980054bcee9cf95ea0db085b2"
ARDITI_C05_FIT = "4cac744ca8e2d80ab73fdb882f7ba0de6c077beb32cbcbb083776ac968582780"
ARDITI_C05_SCALED = "4347aa51d4435c53e3ae029071eda47b4337f83e8c042f4f594c183ed6262c43"


@pytest.fixture(scope="module")
def index():
    return load_index(REPO)


def test_index_covers_the_whole_manifest(index):
    rows = json.load(open(MANIFEST))["objects"]
    assert len(rows) == 152                                       # the 154 in_paper objects less the 2 objects/WITHHELD.json holds
    assert [e.object_id for e in index.entries] == [r["object_id"] for r in rows]
    assert index.manifest_path == MANIFEST


def test_address_key_is_unique_over_every_manifest_row(index):
    """THE contract: (model_id, paper_label, object_kind, derivation, dose, seed) names one object. A duplicate here
    means two committed objects the address cannot tell apart; the resolver's tie-break would then have to be made
    explicit (and pinned) rather than left to list order."""
    dups = index.duplicate_keys()
    assert dups == {}, "\n".join(f"{k}: {[e.object_id[:12] for e in v]}" for k, v in dups.items())


def test_readme_example_resolves_to_the_literal_it_replaced():
    assert resolve("qwen35_4b_it", "difference_lora", repo_root=REPO) == README_OBJECT
    assert resolve("qwen35_4b_it", "difference_lora", kind="lora_subtraction", repo_root=REPO) == README_OBJECT
    # its parents are the effective-batch-64 fits of qwen35_4b_it, whose realized-batch check records seed 0; the
    # address needs neither, since it is the label's only full-dose difference
    e = load_index(REPO).get(README_OBJECT)
    assert (e.dose, e.seed, e.effective_batch, e.derivation, e.dir) == \
        (1.0, 0, 64, "lora_subtraction", f"qwen35_4b_it/{README_OBJECT}")
    assert resolve("qwen35_4b_it", "difference_lora", seed=0, repo_root=REPO) == README_OBJECT


def _readme_section(title: str, readme: Path = README) -> str:
    text = readme.read_text()
    start = text.index(f"## {title}")
    end = text.find("\n## ", start + 1)
    return text[start:end if end > 0 else None]


def _python_block(section: str) -> str:
    blocks = re.findall(r"```python\n(.*?)\n```", section, re.S)
    assert len(blocks) == 1, f"expected one python block in the section, found {len(blocks)}"
    return blocks[0]


def test_readme_worked_example_is_the_base_fit_seed3_adapter_on_the_served_checkpoint():
    """The README's snippet demos the BASE-fit object (qwen35_4b_base, seed 3, dose 1.0) under objects/qwen35_4b_base: the
    seed objects/DEMO.json names, so the demo sparse checkout the section opens with carries it. It loads the checkpoint
    the paper serves it on through serve_checkpoint, never a hard-coded revision."""
    assert resolve("qwen35_4b_base", "difference_lora", seed=3, dose=1.0, repo_root=REPO) == QWEN_DIFF_LORA_SEED3
    e = load_index(REPO).get(QWEN_DIFF_LORA_SEED3)
    assert (e.dose, e.seed, e.effective_batch, e.derivation, e.dir) == \
        (1.0, 3, 64, "lora_subtraction", f"qwen35_4b_base/{QWEN_DIFF_LORA_SEED3}")
    assert (REPO / "objects" / e.dir / "object.pt").is_file()
    demo = {o["paper_label"]: o for o in json.loads((REPO / "objects" / "DEMO.json").read_text())["objects"]}
    assert demo["difference_lora"]["object_id"] == QWEN_DIFF_LORA_SEED3, "the README's object is the one the demo checkout fetches"
    assert serve_checkpoint("qwen35_4b_base", repo_root=REPO) == QWEN_SERVE
    sec = _readme_section("Use a fitted adapter")
    assert 'OBJECT_ID = resolve("qwen35_4b_base", "difference_lora", seed=3, dose=1.0)' in sec
    assert 'Path("objects/qwen35_4b_base") / OBJECT_ID' in sec
    assert 'hf_repo, revision = serve_checkpoint("qwen35_4b_base")' in sec
    assert re.search(r"load_model_and_tokenizer\(hf_repo, revision=revision[,)]", sec), "the load takes the table's revision"
    assert not re.search(r'revision="[0-9a-f]{40}"', sec), "the snippet's revision comes from the models table, not a literal"
    assert "from sisteer.objects import resolve, serve_checkpoint" in sec


class _Stub:
    """Stands in for the loaded model, tokenizer and object in the objects/README example; nothing is called on it
    before the load call the block is run to."""


def test_objects_readme_loading_example_resolves_the_relu_adapter_and_runs_to_the_load_call(monkeypatch):
    """objects/README.md "Loading" addresses the ReLU-steer constitutional adapter by (model_id, paper_label, seed) and
    loads the checkpoint the paper serves it on through serve_checkpoint: no hard-coded object id, no revision literal.
    The block is run as written, with the Hugging Face loader and load_object stubbed, up to and including the load
    call, so the manifest, the sidecars, the models table and the sha256 check of object.pt are all exercised for real."""
    sec = _readme_section("Loading", OBJECTS_README)
    block = _python_block(sec)
    assert 'OBJECT_ID = resolve("qwen35_4b_base", "difference_relu", seed=3)' in block
    assert f"object_id {RELU_DIFF_SEED3}" in block
    assert 'Path("objects/qwen35_4b_base") / OBJECT_ID' in block
    assert 'hf_repo, revision = serve_checkpoint("qwen35_4b_base")' in block
    assert "load_model_and_tokenizer(hf_repo, revision=revision)" in block
    assert "from sisteer.objects import resolve, serve_checkpoint" in block
    assert not re.search(r'revision="[0-9a-f]{40}"', sec), "the revision comes from the models table, not a literal"
    assert not re.search(r'Path\("[^"]*[0-9a-f]{64}', sec), "the object is addressed by (model_id, paper_label, seed), not by id"
    assert resolve("qwen35_4b_base", "difference_relu", seed=3, repo_root=REPO) == RELU_DIFF_SEED3
    assert serve_checkpoint("qwen35_4b_base", repo_root=REPO) == QWEN_SERVE

    calls: dict[str, object] = {}

    def fake_load_model_and_tokenizer(name, *args, revision=None, **kwargs):
        calls["load_model_and_tokenizer"] = (name, revision)
        return _Stub(), _Stub()

    def fake_load_object(path, *, model, tok):
        calls["load_object"] = (Path(path), model, tok)
        return _Stub()

    # patched on the modules BEFORE the block's `from sisteer.models import ...` runs, so the block binds the stubs
    monkeypatch.setattr("sisteer.models.load_model_and_tokenizer", fake_load_model_and_tokenizer)
    monkeypatch.setattr("sisteer.methods.load_object", fake_load_object)
    monkeypatch.chdir(REPO)                                              # the block's paths are repository-root relative
    lines = block.splitlines()
    load_call = next(i for i, line in enumerate(lines) if line.startswith("relu = load_object(path, model=model, tok=tok)"))
    assert any(line.startswith("hf_repo, revision = serve_checkpoint(") for line in lines[:load_call])
    assert any(line.startswith("OBJECT_ID = resolve(") for line in lines[:load_call])
    ns: dict[str, object] = {"__name__": "objects_readme_loading_example"}
    text = OBJECTS_README.read_text()
    first_line = text[:text.index(block)].count("\n")                  # pad so a traceback names the README's own lines
    exec(compile("\n" * first_line + "\n".join(lines[:load_call + 1]), str(OBJECTS_README), "exec"), ns)

    assert ns["OBJECT_ID"] == RELU_DIFF_SEED3
    assert (ns["hf_repo"], ns["revision"]) == QWEN_SERVE
    assert calls["load_model_and_tokenizer"] == QWEN_SERVE
    expected_path = REPO / "objects" / "qwen35_4b_base" / RELU_DIFF_SEED3 / "object.pt"
    assert ns["path"] == expected_path                                   # committed_object_bytes_path verified sha256(bytes)
    path, model, tok = calls["load_object"]
    assert path == expected_path
    assert (model, tok) == (ns["model"], ns["tok"]) and isinstance(ns["relu"], _Stub)


def test_serve_and_fit_checkpoints_read_the_models_table_under_every_name_of_a_row():
    # the row under its table key and under the model_id of either checkpoint (the names objects/ uses)
    for name in ("qwen35_4b", "qwen35_4b_base", "qwen35_4b_it"):
        assert serve_checkpoint(name, repo_root=REPO) == QWEN_SERVE, name
        assert fit_checkpoint(name, repo_root=REPO) == QWEN_FIT, name
    # a _base row's objects are served on the post-trained checkpoint, an _it row's are fit and served on it; Nemotron has one
    assert serve_checkpoint("gemma4e4b_base", repo_root=REPO) == serve_checkpoint("gemma4e4b_it", repo_root=REPO)
    assert fit_checkpoint("llama8b_base", repo_root=REPO) != serve_checkpoint("llama8b_base", repo_root=REPO)
    assert fit_checkpoint("nemotron3nano4b", repo_root=REPO) == serve_checkpoint("nemotron3nano4b", repo_root=REPO)


def test_every_models_table_row_and_every_manifest_model_resolves_to_a_pinned_checkpoint(index):
    rows = models_module().load_models(str(MODELS_TABLE))
    assert rows, "reproduce/models.yaml has no rows"
    for key, row in rows.items():
        for side, fn in (("serve", serve_checkpoint), ("fit", fit_checkpoint)):
            ck = row[side]["checkpoint"]
            got = fn(key, repo_root=REPO)
            assert got == (ck["hf_repo"], ck["revision"]), (key, side)
            assert isinstance(got[0], str) and got[0] and HEX40.match(got[1]), (key, side, got)
            assert fn(ck["model_id"], repo_root=REPO) == got, (key, side, ck["model_id"])
    for model_id in sorted({e.model_id for e in index.entries if e.model_id}):
        repo, rev = serve_checkpoint(model_id, repo_root=REPO)
        assert repo and HEX40.match(rev), model_id
        repo, rev = fit_checkpoint(model_id, repo_root=REPO)
        assert repo and HEX40.match(rev), model_id


def test_unknown_model_checkpoint_is_refused_by_name():
    with pytest.raises(ValueError, match="unknown model 'no_such_model'") as ei:
        serve_checkpoint("no_such_model", repo_root=REPO)
    assert "qwen35_4b_base" in str(ei.value) and "qwen35_4b" in str(ei.value) and "nemotron3nano4b" in str(ei.value)
    with pytest.raises(ValueError, match="unknown model 'no_such_model'"):
        fit_checkpoint("no_such_model", repo_root=REPO)


def test_checkpoint_refuses_a_checkout_without_the_table_or_an_unpinned_revision(tmp_path):
    (tmp_path / "objects").mkdir()
    shutil.copy(MANIFEST, tmp_path / "objects" / "MANIFEST.json")
    with pytest.raises(FileNotFoundError, match="models.py"):
        serve_checkpoint("qwen35_4b", repo_root=tmp_path)
    (tmp_path / "reproduce" / "lib").mkdir(parents=True)
    shutil.copy(REPO / "reproduce" / "lib" / "models.py", tmp_path / "reproduce" / "lib" / "models.py")
    with pytest.raises(FileNotFoundError, match="models.yaml"):
        serve_checkpoint("qwen35_4b", repo_root=tmp_path)
    (tmp_path / "reproduce" / "models.yaml").write_text(
        "schema: 1\nmodels:\n  x:\n    fit: {checkpoint: {hf_repo: a/b, revision: " + "a" * 40 + ", model_id: x_base}}\n"
        "    serve: {checkpoint: {hf_repo: a/c, revision: main, model_id: x}}\n")
    assert fit_checkpoint("x_base", repo_root=tmp_path) == ("a/b", "a" * 40)
    with pytest.raises(ValueError, match="x.serve.checkpoint.revision 'main' is not a pinned 40-hex sha"):
        serve_checkpoint("x", repo_root=tmp_path)


def test_dose_scaled_copies_resolve_by_dose():
    assert resolve("qwen35_4b_it", "difference_lora", dose=0.5, repo_root=REPO) == README_OBJECT_HALF
    assert resolve("gemma4e4b_base", "constitution_lora", dose=0.25, repo_root=REPO) == GEMMA_CONST_LORA_Q
    # the full-dose default is dose 1.0 even when the row is committed as a dose_scale derivation (its own anchor)
    assert resolve("llama8b_it", "constitution_lora", repo_root=REPO) == LLAMA_IT_CONST_LORA
    assert resolve("llama8b_it", "constitution_lora", dose=1.0, repo_root=REPO) == LLAMA_IT_CONST_LORA
    # the refusal direction's dose is its raw coefficient c
    assert resolve("qwen35_4b_it", "arditi_allpos", dose=0.3, repo_root=REPO) == ARDITI_C03


def test_seed_replicates_resolve_by_seed_and_default_to_the_canonical_fit():
    assert resolve("qwen35_4b_base", "constitution_lora", repo_root=REPO) == QWEN_CONST_LORA_EB8
    assert resolve("qwen35_4b_base", "constitution_lora", seed=2, repo_root=REPO) == QWEN_CONST_LORA_SEED2
    # a difference carries its parents' seed through derived_meta.json
    assert resolve("qwen35_4b_base", "difference_lora", seed=1, repo_root=REPO) == QWEN_DIFF_LORA_SEED1
    assert resolve("qwen35_4b_base", "difference_relu", seed=3, repo_root=REPO) == RELU_DIFF_SEED3
    idx = load_index(REPO)
    assert (idx.get(QWEN_CONST_LORA_EB8).seed, idx.get(QWEN_CONST_LORA_EB8).effective_batch) == (None, 8)
    assert (idx.get(QWEN_CONST_LORA_SEED2).seed, idx.get(QWEN_CONST_LORA_SEED2).effective_batch) == (2, 64)
    assert (idx.get(QWEN_DIFF_LORA_SEED1).seed, idx.get(QWEN_DIFF_LORA_SEED1).effective_batch) == (1, 64)
    # elsewhere a label has one fit (seed 0 recorded by the realized-batch check) or two differing in effective batch
    assert resolve("llama8b_base", "constitution_lora", repo_root=REPO) == \
        resolve("llama8b_base", "constitution_lora", seed=0, repo_root=REPO)
    assert resolve("gemma4e4b_it", "constitution_lora", repo_root=REPO) != \
        resolve("gemma4e4b_it", "constitution_lora", seed=0, repo_root=REPO)
    assert idx.get(resolve("gemma4e4b_it", "constitution_lora", repo_root=REPO)).effective_batch == 8


def test_virtual_rows_resolve_by_label():
    assert resolve(None, "none", repo_root=REPO) == "none"
    assert resolve(None, "constitution_si", repo_root=REPO).startswith("si:")
    assert load_index(REPO).get("none").dose is None


def test_ambiguous_address_is_refused_by_name_with_the_candidates():
    # the three ReLU differences on qwen35_4b_base are the batch-64 seeds 1, 2, 3: no canonical one to default to
    with pytest.raises(ValueError) as ei:
        resolve("qwen35_4b_base", "difference_relu", repo_root=REPO)
    msg = str(ei.value)
    assert "ambiguous" in msg and "qwen35_4b_base" in msg and "difference_relu" in msg
    for seed in (1, 2, 3):
        assert f"seed={seed}" in msg
    assert RELU_DIFF_SEED3[:12] in msg
    # two arditi_allpos rungs at c = 0.5 differ only in derivation (the fit and a dose-scaled copy)
    with pytest.raises(ValueError, match="ambiguous"):
        resolve("qwen35_4b_it", "arditi_allpos", dose=0.5, repo_root=REPO)
    assert resolve("qwen35_4b_it", "arditi_allpos", dose=0.5, derivation="none", repo_root=REPO) == ARDITI_C05_FIT
    assert resolve("qwen35_4b_it", "arditi_allpos", dose=0.5, derivation="dose_scale", repo_root=REPO) == ARDITI_C05_SCALED
    # no full-dose rung at all: refused, listing the doses present
    with pytest.raises(ValueError, match="no full-dose"):
        resolve("qwen35_4b_it", "arditi_allpos", repo_root=REPO)


def test_unknown_label_model_dose_and_seed_are_refused_by_name():
    with pytest.raises(ValueError) as ei:
        resolve("qwen35_4b_it", "no_such_label", repo_root=REPO)
    assert "no_such_label" in str(ei.value) and "difference_lora" in str(ei.value)
    with pytest.raises(ValueError, match="unknown model_id 'no_such_model'"):
        resolve("no_such_model", "difference_lora", repo_root=REPO)
    with pytest.raises(ValueError, match="at dose 9.0") as ei:
        resolve("qwen35_4b_it", "difference_lora", dose=9.0, repo_root=REPO)
    assert "0.5" in str(ei.value) and "1.0" in str(ei.value)
    with pytest.raises(ValueError, match="at seed 7"):
        resolve("qwen35_4b_base", "constitution_lora", seed=7, repo_root=REPO)


def test_find_filters_and_orders(index):
    rows = find(repo_root=REPO, model_id="qwen35_4b_it", label="difference_lora")
    assert [e.object_id for e in rows] == [README_OBJECT_HALF, README_OBJECT]      # by dose ascending
    assert [e.dose for e in rows] == [0.5, 1.0]
    assert all(e.kind == "lora_subtraction" for e in rows)
    assert len(find(repo_root=REPO, derivation="dose_scale")) == 92
    assert len(find(repo_root=REPO)) == 152
    assert [e.object_id for e in find(repo_root=REPO, model_id="qwen35_4b_base", label="constitution_lora", seed=2)] \
        == [QWEN_CONST_LORA_SEED2]
    assert index.path(index.get(README_OBJECT)) == REPO / "objects" / "qwen35_4b_it" / README_OBJECT
    assert index.path(index.get("none")) is None


def test_index_is_cached_per_root_and_locates_the_manifest(tmp_path):
    assert load_index(REPO) is load_index(REPO)
    assert load_index() is load_index(REPO)                      # located beside the package, no root given
    with pytest.raises(FileNotFoundError):
        load_index(tmp_path)
    assert objects.find_repo_root(REPO) == REPO


#: qwen35_4b_base difference_lora at seed 3 (objects/DEMO.json's LoRA constitutional adapter): its parents are the seed-3
#: constitution and control fits (derived_meta.json parent_pos / parent_neg). The demo checkout carries them too (DEMO.json
#: names one training seed for every fitted object); the sparse root below leaves them out on purpose, to exercise a
#: present difference whose parents are absent.
QWEN_DIFF_LORA_SEED3 = "8952728fa0f7fb6a0b48557e9c491073c5080f63178fc64c4f113088e1f2f974"
QWEN_CONST_LORA_SEED3 = "a9934f6a796c238d58f4da10615c9f1c211d8b067afcf1330241cdcff967ce42"
QWEN_CONTROL_LORA_SEED3 = "33e7b327049aa28605818fe02970d918a86a2c6eea8f6bbf38a01779b84f9d32"
#: qwen35_4b_base constitution_lora at seed 1: in the manifest, not in the sparse root below.
QWEN_CONST_LORA_SEED1 = "8be1e1a7ab36f3f65557f7835a30c6b8f18143c781fbf7fb312a5cc7a6e08871"


def test_sparse_checkout_indexes_the_whole_manifest_and_resolves_only_what_is_present(tmp_path):
    """The README's demo checkout holds a few object directories under the whole manifest. The index still covers every
    row; an absent row has no dose or seed and is never resolved (the refusal names the directory to check out); a present
    difference whose parents are absent has an unreadable seed, names the parents' directories, and resolves when the seed
    is left out; a seed the manifest has but the checkout lacks is refused as absent, not as nonexistent."""
    root = tmp_path / "sparse"
    (root / "objects" / "qwen35_4b_base").mkdir(parents=True)
    (root / "objects" / "qwen35_4b_it").mkdir()
    (root / "objects" / "MANIFEST.json").write_bytes(MANIFEST.read_bytes())
    # cone mode brings every file directly under objects/ with any object directory: the withheld list the manifest names too
    (root / "objects" / "WITHHELD.json").write_bytes((REPO / "objects" / "WITHHELD.json").read_bytes())
    for oid in (QWEN_CONST_LORA_SEED2, QWEN_DIFF_LORA_SEED3):
        os.symlink(REPO / "objects" / "qwen35_4b_base" / oid, root / "objects" / "qwen35_4b_base" / oid)
    os.symlink(REPO / "objects" / "qwen35_4b_it" / ARDITI_C03, root / "objects" / "qwen35_4b_it" / ARDITI_C03)
    idx = load_index(root)
    assert len(idx.entries) == 152 and len(idx.withheld) == 2
    assert sorted(e.object_id for e in idx.entries if e.present and e.dir) == \
        sorted([QWEN_CONST_LORA_SEED2, QWEN_DIFF_LORA_SEED3, ARDITI_C03])
    assert all(e.present for e in idx.entries if not e.dir)                         # the virtual rows have no directory
    absent = idx.get(README_OBJECT)
    assert (absent.present, absent.dose, absent.seed, absent.seed_read) == (False, None, None, False)
    assert absent.unread_dirs == (f"qwen35_4b_it/{README_OBJECT}",)
    assert "not in this checkout" in absent.describe() and "dose=? seed=?" in absent.describe()
    with pytest.raises(ValueError) as ei:
        resolve("qwen35_4b_it", "difference_lora", repo_root=root)
    msg = str(ei.value)
    assert "not in this checkout" in msg and "git sparse-checkout add objects/<dir>" in msg and README_OBJECT[:12] in msg
    # present, sidecars read: the fit's own seed, the refusal direction's own coefficient
    assert resolve("qwen35_4b_base", "constitution_lora", seed=2, repo_root=root) == QWEN_CONST_LORA_SEED2
    assert idx.get(QWEN_CONST_LORA_SEED2).seed_read and idx.get(QWEN_CONST_LORA_SEED2).seed == 2
    assert resolve("qwen35_4b_it", "arditi_allpos", dose=0.3, repo_root=root) == ARDITI_C03
    # present, parents absent: the seed cannot be read; the entry says so; the address resolves without a seed, not with one
    diff = idx.get(QWEN_DIFF_LORA_SEED3)
    assert (diff.present, diff.dose, diff.seed, diff.seed_read) == (True, 1.0, None, False)
    parents = (f"qwen35_4b_base/{QWEN_CONST_LORA_SEED3}", f"qwen35_4b_base/{QWEN_CONTROL_LORA_SEED3}")
    assert diff.unread_dirs == parents                                                 # parent_pos, then parent_neg
    assert "seed unreadable" in diff.describe() and "seed=?" in diff.describe() and all(f"objects/{p}" in diff.describe()
                                                                                        for p in parents)
    assert resolve("qwen35_4b_base", "difference_lora", repo_root=root) == QWEN_DIFF_LORA_SEED3
    with pytest.raises(ValueError, match="at seed 3") as ei:
        resolve("qwen35_4b_base", "difference_lora", seed=3, repo_root=root)
    msg = str(ei.value)
    assert "seed unreadable" in msg and "leave seed= out" in msg
    assert f"git sparse-checkout add objects/{parents[0]} objects/{parents[1]}" in msg  # the one command that makes it readable
    # a seed the manifest has but this checkout lacks: refused as absent (with the directory to fetch), not as nonexistent
    with pytest.raises(ValueError, match="at seed 1") as ei:
        resolve("qwen35_4b_base", "constitution_lora", seed=1, repo_root=root)
    msg = str(ei.value)
    assert "not in this checkout" in msg and "git sparse-checkout add objects/<dir>" in msg
    assert QWEN_CONST_LORA_SEED1[:12] in msg and QWEN_CONST_LORA_SEED2[:12] in msg   # the absent one and the present one
    # the full checkout is untouched by any of this
    full = load_index(REPO).get(QWEN_DIFF_LORA_SEED3)
    assert (full.present, full.seed, full.seed_read, full.unread_dirs) == (True, 3, True, ())
    with pytest.raises(ValueError, match="at seed 7") as ei:
        resolve("qwen35_4b_base", "constitution_lora", seed=7, repo_root=REPO)
    assert "checkout" not in str(ei.value)                                              # no sparse note in a full checkout
    r = _cli("find", "--root", str(root), "--model", "qwen35_4b_it", "--label", "difference_lora")
    assert r.returncode == 0, r.stderr
    lines = r.stdout.splitlines()
    assert lines[1].split() == [README_OBJECT_HALF, "difference_lora", "lora_subtraction", "dose_scale", "?", "?",
                                f"qwen35_4b_it/{README_OBJECT_HALF}", "(absent)"]
    r = _cli("find", "--root", str(root), "--model", "qwen35_4b_base", "--label", "difference_lora")
    assert r.returncode == 0 and r.stdout.splitlines()[-1].split()[4:7] == \
        ["1.0", "?", f"qwen35_4b_base/{QWEN_DIFF_LORA_SEED3}"]                           # present, dose read, seed unreadable
    r = _cli("find", "--root", str(root), "--model", "qwen35_4b_base", "--label", "difference_lora", "--seed", "3")
    assert r.returncode == 1 and len(r.stdout.splitlines()) == 1                      # present, but its seed is unreadable


def _cli(*args: str) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO / "src") + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run([sys.executable, "-m", "sisteer.objects", *args], cwd=REPO, env=env,
                          capture_output=True, text=True, timeout=120)


def test_cli_round_trips():
    r = _cli("resolve", "qwen35_4b_it", "difference_lora")
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == README_OBJECT
    r = _cli("resolve", "qwen35_4b_it", "difference_lora", "--dose", "0.5")
    assert r.returncode == 0 and r.stdout.strip() == README_OBJECT_HALF
    r = _cli("resolve", "qwen35_4b_base", "constitution_lora", "--seed", "2", "--kind", "lora")
    assert r.returncode == 0 and r.stdout.strip() == QWEN_CONST_LORA_SEED2
    r = _cli("resolve", "none", "none")
    assert r.returncode == 0 and r.stdout.strip() == "none"
    r = _cli("find", "--model", "qwen35_4b_it", "--label", "difference_lora")
    assert r.returncode == 0, r.stderr
    lines = r.stdout.splitlines()
    assert lines[0].split() == ["object_id", "label", "kind", "derivation", "dose", "seed", "dir"]
    assert len(lines) == 3
    assert lines[1].split() == [README_OBJECT_HALF, "difference_lora", "lora_subtraction", "dose_scale", "0.5", "0",
                                f"qwen35_4b_it/{README_OBJECT_HALF}"]
    assert lines[2].split() == [README_OBJECT, "difference_lora", "lora_subtraction", "lora_subtraction", "1.0", "0",
                                f"qwen35_4b_it/{README_OBJECT}"]
    r = _cli("find", "--model", "qwen35_4b_base", "--label", "constitution_lora", "--seed", "2")
    assert r.returncode == 0 and r.stdout.splitlines()[1].split()[0] == QWEN_CONST_LORA_SEED2
    r = _cli("resolve", "qwen35_4b_base", "difference_relu")
    assert r.returncode == 2 and r.stdout == "" and "ambiguous" in r.stderr and "seed=3" in r.stderr
    r = _cli("resolve", "qwen35_4b_it", "no_such_label")
    assert r.returncode == 2 and "no_such_label" in r.stderr
    r = _cli("find", "--model", "qwen35_4b_base", "--label", "constitution_lora")
    assert r.returncode == 0 and [l.split()[5] for l in r.stdout.splitlines()[1:]] == ["-"] * 5 + ["-", "1", "2", "3"]
    r = _cli("find", "--model", "qwen35_4b_it", "--label", "no_such_label")
    assert r.returncode == 1 and r.stdout.splitlines() == ["object_id  label  kind  derivation  dose  seed  dir"]
    # the README's worked example, end to end on the CLI: the base-fit seed-1 adapter and the checkpoint it is served on
    r = _cli("resolve", "qwen35_4b_base", "difference_lora", "--seed", "1", "--dose", "1.0")
    assert r.returncode == 0 and r.stdout.strip() == QWEN_DIFF_LORA_SEED1
    r = _cli("checkpoint", "qwen35_4b_base")
    assert r.returncode == 0, r.stderr
    assert tuple(r.stdout.split()) == QWEN_SERVE and r.stdout.endswith("\n")
    r = _cli("checkpoint", "qwen35_4b_base", "--fit")
    assert r.returncode == 0 and tuple(r.stdout.split()) == QWEN_FIT
    r = _cli("checkpoint", "qwen35_4b")
    assert r.returncode == 0 and tuple(r.stdout.split()) == QWEN_SERVE
    r = _cli("checkpoint", "no_such_model")
    assert r.returncode == 2 and r.stdout == "" and "unknown model 'no_such_model'" in r.stderr
