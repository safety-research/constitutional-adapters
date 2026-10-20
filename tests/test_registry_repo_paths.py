"""The adapters read the repo's OWN files from the package's tree, never from a home-relative worktree path.

Three pins once named working trees outside the checkout: ``declared_common.SEEDS_DIR``, ``arditi_strength.PINS_PATH``
and ``em_misalign.REPO``. When those trees were removed the adapters kept
running — silently — on the built-in seed transcription and on empty pin sets. Now every such path resolves from
``declared_common.REPO_ROOT`` (this checkout), exists in the repo, and REFUSES when missing; ``$SISTEER_REGISTRY_SEEDS``
names another seeds directory; and no other module in the package pins a working-tree path except the documented ones
(``relu_native`` — an unlanded branch's working-tree records; ``PIN_PAPER_DIR`` — the parity pin of the paper-figures
branch; ``config.DEFAULTS["paper"]["checkout"]`` — the paper checkout the paper ops read, a config default).
"""
from __future__ import annotations

import ast
import os
import re
import shutil

import pytest

from sisteer.registry.adapters import arditi_strength as AR
from sisteer.registry.adapters import declared_common as DC
from sisteer.registry.adapters import em_misalign as EM
from sisteer.registry.adapters.base import AdapterRefusal

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PKG = os.path.join(ROOT, "src", "sisteer", "registry")


# ---- the seeds directory ------------------------------------------------------------------------------------------

def test_repo_root_is_this_checkout():
    assert os.path.realpath(DC.REPO_ROOT) == os.path.realpath(ROOT)
    assert os.path.isfile(os.path.join(DC.REPO_ROOT, "pyproject.toml"))
    assert DC.repo_path("configs", "registry") == os.path.join(DC.REPO_ROOT, "configs", "registry")


def test_seeds_dir_is_the_repo_seeds_and_every_seed_file_exists():
    assert DC.SEEDS_DIR == os.path.join(DC.REPO_ROOT, "configs", "registry")
    assert os.path.isdir(DC.SEEDS_DIR)
    for name in DC.SEED_FILES:
        assert os.path.isfile(os.path.join(DC.SEEDS_DIR, name)), name
    assert DC.resolve_seeds_dir() == DC.SEEDS_DIR


def test_load_seeds_reads_the_repo_seeds_not_the_fallback(monkeypatch):
    """The failure this guards: the seeds directory absent and ``load_seeds()`` answering from the built-in transcription with
    ``source = built-in``. The default read now lands on the repo's files."""
    yaml = pytest.importorskip("yaml")
    monkeypatch.delenv(DC.SEEDS_DIR_ENV, raising=False)
    monkeypatch.setattr(DC, "_SEEDS_CACHE", {})
    seeds = DC.load_seeds()
    assert seeds.source == DC.SEEDS_DIR
    assert not seeds.source.startswith("built-in")
    with open(os.path.join(DC.SEEDS_DIR, "judges.yaml"), encoding="utf-8") as fh:
        declared = {str(r["judge_id"]) for r in yaml.safe_load(fh)["judges"]}
    assert declared and declared <= {str(j["judge_id"]) for j in seeds.judges}
    with open(os.path.join(DC.SEEDS_DIR, "evals.yaml"), encoding="utf-8") as fh:
        eval_ids = {str(r["eval_id"]) for r in yaml.safe_load(fh)["evals"]}
    assert eval_ids and eval_ids <= set(seeds.eval_by_spelling.values()) | set(seeds.eval_by_spelling)


def test_env_override_names_the_seeds_dir(tmp_path, monkeypatch):
    other = tmp_path / "seeds"
    other.mkdir()
    for name in DC.SEED_FILES:
        shutil.copy(os.path.join(DC.SEEDS_DIR, name), other / name)
    monkeypatch.setenv(DC.SEEDS_DIR_ENV, str(other))
    monkeypatch.setattr(DC, "_SEEDS_CACHE", {})
    assert DC.resolve_seeds_dir() == str(other)
    assert DC.load_seeds().source == str(other)
    # an explicit argument beats the override
    assert DC.resolve_seeds_dir(DC.SEEDS_DIR) == DC.SEEDS_DIR


def test_missing_seeds_refuse_and_never_fall_back(tmp_path, monkeypatch):
    monkeypatch.setattr(DC, "_SEEDS_CACHE", {})
    monkeypatch.delenv(DC.SEEDS_DIR_ENV, raising=False)
    # the repo default gone (that failure, re-created on a scratch path)
    monkeypatch.setattr(DC, "SEEDS_DIR", str(tmp_path / "absent"))
    with pytest.raises(AdapterRefusal) as ei:
        DC.load_seeds()
    assert "evals.yaml" in str(ei.value) and DC.SEEDS_DIR_ENV in str(ei.value) and "D100" in str(ei.value)
    # the override naming a directory that is not there
    monkeypatch.setenv(DC.SEEDS_DIR_ENV, str(tmp_path / "nope"))
    with pytest.raises(AdapterRefusal) as ei:
        DC.resolve_seeds_dir()
    assert "$" + DC.SEEDS_DIR_ENV in str(ei.value)
    # an explicit directory missing one of the files it must carry
    partial = tmp_path / "partial"
    partial.mkdir()
    (partial / "evals.yaml").write_text("evals: []\n", encoding="utf-8")
    with pytest.raises(AdapterRefusal) as ei:
        DC.load_seeds(str(partial))
    assert "si_texts.yaml" in str(ei.value) and "judges.yaml" in str(ei.value) and "evals.yaml" not in str(ei.value).split("missing (")[1].split(")")[0]


# ---- the pins files -----------------------------------------------------------------------------------------------

def test_arditi_pins_path_is_the_repo_file_and_a_missing_file_refuses(tmp_path):
    assert AR.PINS_PATH == os.path.join(DC.REPO_ROOT, "configs", "misalign_baseline_pins.json")
    assert os.path.isfile(AR.PINS_PATH)
    pinned = AR._pinned_shas(None)
    assert pinned, "the default read must land on the repo's pins, not an empty set"
    assert all(re.fullmatch(r"[0-9a-f]{64}", str(v["sha"])) for v in pinned.values())
    assert all(str(v["source"]).startswith(AR.PINS_PATH + "#") for v in pinned.values())
    with pytest.raises(AdapterRefusal) as ei:
        AR._pinned_shas(str(tmp_path / "absent.json"))
    assert "D100" in str(ei.value)


def test_em_pins_are_the_repo_files_and_a_named_missing_file_refuses(tmp_path):
    assert EM.REPO == DC.REPO_ROOT
    assert EM.FULL22_PINS == os.path.join(DC.REPO_ROOT, "configs", "sdfbase_full22_pins.json") and os.path.isfile(EM.FULL22_PINS)
    assert EM.BASELINE_PINS == os.path.join(DC.REPO_ROOT, "configs", "misalign_baseline_pins.json") and os.path.isfile(EM.BASELINE_PINS)
    assert EM.EmMisalignAdapter._read_pins(None) == {}          # a test's deliberate "no pins"
    assert EM.EmMisalignAdapter._read_pins(EM.FULL22_PINS)      # the repo's file reads non-empty
    assert EM.EmMisalignAdapter._read_pins(EM.BASELINE_PINS)
    with pytest.raises(AdapterRefusal) as ei:
        EM.EmMisalignAdapter._read_pins(str(tmp_path / "absent.json"))
    assert "D100" in str(ei.value)
    root = tmp_path / "em"
    root.mkdir()
    with pytest.raises(AdapterRefusal):
        EM.EmMisalignAdapter(root=str(root), full22_pins=str(tmp_path / "absent.json"), baseline_pins=None, behaviour_source={},
                             grade_root=str(root))
    # the defaults construct (both pins read from the repo)
    ad = EM.EmMisalignAdapter(root=str(root), behaviour_source={}, grade_root=str(root))
    assert ad._pins and ad._bpins


# ---- no other worktree pin in the package -------------------------------------------------------------------------

#: module (relative to the package) -> the top-level assignments allowed to carry a worktree path, or "*" for the module
ALLOWED_WORKTREE_PINS = {
    # the parity pin: a pinned checkout of the paper-figures branch, not this tree; the parity tests skip when absent
    "adapters/declared_common.py": {"PIN_PAPER_DIR"},
    # the paper checkout the paper ops read (another session's live branch): a config DEFAULT the operator overrides
    "config.py": {"DEFAULTS"},
    # the RETIRED paired root (registry follow-ups-4 item 11): the OLD path the 2026-09-21 accepted heads recorded, declared so
    # the inputs-dropped comparison counts its files `inputs_retired` -- a declaration ABOUT a worktree path, never a read from one
    # (`PAIRED_ROOT`, the live read, is under the run-state root; tests/test_registry_relu_native_retired_root.py pins both)
    "adapters/relu_native.py": {"RETIRED_PAIRED_ROOT"},
    # git's own administrative layout for a linked checkout (``<main>/.git/worktrees/<name>``), read to name the checkout a linked
    # worktree was added from (the repo mirror's directory name) -- never a path into a worktree
    "stores.py": {"LINKED_GITDIR_PARENT"},
}


def _worktree_strings(path):
    """Every string constant naming ``worktrees`` in CODE (docstrings excluded; comments are not in the AST), with the
    top-level statement's assigned name."""
    with open(path, encoding="utf-8") as fh:
        tree = ast.parse(fh.read(), path)
    doc_ids = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
                doc_ids.add(id(first.value))
    hits = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and "worktrees" in node.value and id(node) not in doc_ids:
            owner = None
            for stmt in tree.body:
                if stmt.lineno <= node.lineno <= (stmt.end_lineno or stmt.lineno):
                    targets = getattr(stmt, "targets", None) or ([stmt.target] if hasattr(stmt, "target") else [])
                    names = [t.id for t in targets if isinstance(t, ast.Name)]
                    owner = names[0] if names else type(stmt).__name__
                    break
            hits.append((node.lineno, owner, node.value))
    return hits


def test_no_other_module_pins_a_worktree_path():
    offenders = {}
    for dirpath, _, files in os.walk(PKG):
        for f in files:
            if not f.endswith(".py"):
                continue
            path = os.path.join(dirpath, f)
            rel = os.path.relpath(path, PKG)
            allowed = ALLOWED_WORKTREE_PINS.get(rel, set())
            if allowed == "*":
                continue
            bad = [h for h in _worktree_strings(path) if h[1] not in allowed]
            if bad:
                offenders[rel] = bad
    assert not offenders, offenders


def test_the_three_repointed_pins_carry_no_worktree_string():
    for mod, names in (("adapters/declared_common.py", {"SEEDS_DIR", "SEEDS_DIR_ENV", "SEED_FILES", "REPO_ROOT"}),
                       ("adapters/arditi_strength.py", {"PINS_PATH"}),
                       ("adapters/em_misalign.py", {"REPO", "FULL22_PINS", "BASELINE_PINS"})):
        hits = [h for h in _worktree_strings(os.path.join(PKG, mod)) if h[1] in names]
        assert not hits, (mod, hits)
