"""Dirty-generator-tree refusal + provenance recording (fix item 5).

Production builds of the sdfbase pool/split artifacts must run from
COMMITTED code: the recorded commit is the artifact's provenance, and a
dirty tree stamps a commit its bytes were not built from. Both CLIs
(scripts/sdfbase/build_pools.py build, scripts/sdfbase/build_heldout_split.py)
refuse a dirty generator tree unless --allow-dirty, which records
dirty=true + a tree-diff sha instead. Pinned here:

- refuse_dirty_generator_tree: refuse / allow / unverifiable semantics;
- generator_provenance rides every pool cell manifest (build_pools) with
  the historical *_commit key spelling kept;
- the SPLIT artifact itself stays provenance-free BY DESIGN (a committed
  split must byte-match a fresh rebuild at any commit — pinned by
  test_sdfbase_split.test_real_split_build); its provenance rides the
  CLI-written .provenance.json sidecar;
- both CLIs gate before building (build stubs prove nothing runs on a
  refused tree).

git is never invoked here — tree_state is monkeypatched, so the suite is
independent of the checkout's actual state.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "scripts" / "sdfbase",):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import build_heldout_split as hs  # noqa: E402
import build_pools as bp  # noqa: E402


def _fake_state(dirty_map: dict[str, bool | None]):
    """A tree_state stand-in keyed by which repo path is asked about."""
    def fake(repo: Path):
        name = ("sisteer" if Path(repo) == hs.WORKTREE_ROOT
                else "corpus_generator")
        d = dirty_map.get(name, False)
        return {"commit": f"{name}-commit" if d is not None else "unknown",
                "dirty": d,
                "diff_sha256": ("d" * 64) if d else None}
    return fake


# --------------------------------------------------------------------------- #
# the gate                                                                      #
# --------------------------------------------------------------------------- #


def test_dirty_tree_refuses_without_allow_dirty(monkeypatch):
    monkeypatch.setattr(hs, "tree_state", _fake_state({"sisteer": True}))
    with pytest.raises(hs.SplitError, match="sisteer"):
        hs.refuse_dirty_generator_tree(False, tag="test")
    # --allow-dirty proceeds AND the record carries dirty + the diff sha
    prov = hs.refuse_dirty_generator_tree(True, tag="test")
    assert prov["sisteer_dirty"] is True
    assert prov["sisteer_diff_sha256"] == "d" * 64
    assert prov["sisteer_commit"] == "sisteer-commit"


def test_clean_and_unverifiable_trees_pass(monkeypatch):
    monkeypatch.setattr(hs, "tree_state", _fake_state({}))
    prov = hs.refuse_dirty_generator_tree(False, tag="test")
    assert prov["sisteer_dirty"] is False
    assert prov["sisteer_diff_sha256"] is None
    # unverifiable (no git / not a repo): recorded honestly, never a block
    monkeypatch.setattr(hs, "tree_state",
                        _fake_state({"sisteer": None,
                                     "corpus_generator": None}))
    prov = hs.refuse_dirty_generator_tree(False, tag="test")
    assert prov["sisteer_dirty"] is None
    assert prov["sisteer_commit"] == "unknown"


def test_either_dirty_repo_refuses(monkeypatch):
    monkeypatch.setattr(hs, "tree_state",
                        _fake_state({"corpus_generator": True}))
    with pytest.raises(hs.SplitError, match="corpus_generator"):
        hs.refuse_dirty_generator_tree(False, tag="test")


# --------------------------------------------------------------------------- #
# build_pools: provenance in every manifest; CLI gates before building          #
# --------------------------------------------------------------------------- #


def test_build_cell_manifest_carries_the_provenance(tmp_path, monkeypatch):
    monkeypatch.setattr(hs, "tree_state", _fake_state({"sisteer": True}))
    prov = hs.generator_provenance()
    # build_cell writes the generator dict verbatim into the manifest
    # (build_pools passes generator_provenance() when the CLI gave none).
    doc = bp.Doc(doc_id="doc-1", family="stories", prompt="",
                 pos="tiny story body",
                 token_counts={k: 3 for k in bp.TOK_NAMES})

    class _Rig:
        provenance = {"synthetic": True}

        def counts(self, text):
            return {k: 3 for k in bp.TOK_NAMES}

    m = bp.build_cell(
        "constv2-test", "heldout", None, [doc],
        {k: 3 for k in bp.TOK_NAMES}, salts={}, natural_ratio=None,
        out_dir=tmp_path, rig=_Rig(), split_sha="0" * 64, corpus_shas={},
        generator=prov)
    assert m["generator"]["sisteer_commit"] == "sisteer-commit"
    assert m["generator"]["sisteer_dirty"] is True
    assert m["generator"]["sisteer_diff_sha256"] == "d" * 64
    on_disk = json.loads(
        (tmp_path / "pairs_constv2-test.manifest.json").read_text())
    assert on_disk["generator"] == m["generator"]


def test_build_pools_cli_gates_before_building(tmp_path, monkeypatch):
    calls = []

    def stub_build_pools(out, split, **kw):
        calls.append(kw)
        return {}

    monkeypatch.setattr(bp, "build_pools", stub_build_pools)
    monkeypatch.setattr(hs, "tree_state", _fake_state({"sisteer": True}))
    args = ["build", "--out", str(tmp_path / "pools"),
            "--split", str(tmp_path / "split.json")]
    assert bp.main(args) == 2          # refused; nothing built
    assert calls == []
    assert bp.main(args + ["--allow-dirty"]) == 0
    assert len(calls) == 1
    gen = calls[0]["generator"]
    assert gen["sisteer_dirty"] is True
    assert gen["sisteer_diff_sha256"] == "d" * 64


# --------------------------------------------------------------------------- #
# build_heldout_split: CLI gate + provenance SIDECAR, artifact untouched        #
# --------------------------------------------------------------------------- #


def _stub_split(monkeypatch):
    acc = {"heldout_groups": 1, "family_groups": 2, "heldout_records": 1,
           "family_records": 2, "heldout_official": 10,
           "family_official": 100, "heldout_frac_realized": 0.1,
           "heldout_limiting_tokenizer": "gemma", "train_official": 90}
    split = {"token_totals": {"advice": acc, "stories": acc},
             "eval_subset": {"n": 2, "ids": ["a", "b"],
                             "family_counts": {"advice": 2}}}
    monkeypatch.setattr(hs, "build_split",
                        lambda *a, **k: json.loads(json.dumps(split)))
    return split


def test_split_cli_refuses_dirty_then_records_in_sidecar(tmp_path,
                                                         monkeypatch):
    _stub_split(monkeypatch)
    monkeypatch.setattr(hs, "tree_state", _fake_state({"sisteer": True}))
    out = tmp_path / "heldout_split.json"
    args = ["--advice-final", str(tmp_path / "a.jsonl"),
            "--stories-final", str(tmp_path / "s.jsonl"),
            "--out", str(out)]
    assert hs.main(args) == 2
    assert not out.exists()            # a refused build writes NOTHING

    assert hs.main(args + ["--allow-dirty"]) == 0
    artifact = json.loads(out.read_text())
    # The split artifact stays provenance-free BY DESIGN (deterministic
    # rebuild equality); the sidecar carries the record.
    assert "generator" not in artifact
    prov = json.loads(out.with_suffix(".provenance.json").read_text())
    assert prov["artifact"] == out.name
    assert prov["artifact_sha256"] == hs.sha256_file(out)
    assert prov["generator"]["sisteer_dirty"] is True
    assert prov["generator"]["sisteer_diff_sha256"] == "d" * 64


def test_split_cli_clean_tree_builds_with_clean_sidecar(tmp_path,
                                                        monkeypatch):
    _stub_split(monkeypatch)
    monkeypatch.setattr(hs, "tree_state", _fake_state({}))
    out = tmp_path / "heldout_split.json"
    assert hs.main(["--advice-final", str(tmp_path / "a.jsonl"),
                    "--stories-final", str(tmp_path / "s.jsonl"),
                    "--out", str(out)]) == 0
    prov = json.loads(out.with_suffix(".provenance.json").read_text())
    assert prov["generator"]["sisteer_dirty"] is False
    assert prov["generator"]["sisteer_diff_sha256"] is None


def test_real_build_split_artifact_has_no_generator_key(tmp_path):
    """The artifact's determinism contract survives item 5: build_split
    output carries NO tree state (the sidecar does)."""
    sys.path.insert(0, str(ROOT / "tests"))
    from test_sdfbase_split import synthetic_corpus, write_jsonl

    advice, stories = synthetic_corpus()
    ap = write_jsonl(tmp_path / "advice.jsonl", advice)
    sp = write_jsonl(tmp_path / "stories.jsonl", stories)
    split = hs.build_split(ap, sp, frac=0.2, eval_n=4)
    assert "generator" not in split
