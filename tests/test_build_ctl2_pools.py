"""scripts/sdfbase/build_ctl2_pools.py: the r2 register-matched
control pools. Pins: input-sha refusal; the budget is READ from the
constv2-mix-full manifest (any other cell refuses); the held-out split is
the constitutional carve's own machinery on the control finals (whole
groups, eval subset inside the held-out set); ``constv2ctl2-full`` is a
PREFIX of one deterministic draw of the train side cut EXACTLY at the
budget, its remainder recorded and disjoint; every id wears the ``ctl2-``
prefix (never ``ctl-``); determinism byte for byte. Synthetic corpus +
fake counting rig (the tests/test_sdfbase_pools.py conventions)."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "sdfbase"))

import build_ctl2_pools as b2  # noqa: E402
import build_heldout_split as hs  # noqa: E402
import build_pools as bp  # noqa: E402

try:  # the corpus repo's renderers (cross-repo import by path)
    hs.generator_tokens()
except hs.SplitError as _e:  # pragma: no cover — machines without the repo
    pytest.skip(f"corpus repo renderers unavailable: {_e}",
                allow_module_level=True)


class FakeRig:
    provenance = {"fake": True}

    def counts(self, text: str) -> dict[str, int]:
        n = len(text.split())
        return {k: n for k in hs.TOK_NAMES}

    def boundary_offsets(self, name: str, text: str) -> list[int]:
        return [m.end() for m in re.finditer(r"\S+", text)]


def words(prefix: str, n: int) -> str:
    return " ".join(f"{prefix}{i:04d}" for i in range(n))


def adv_rec(p, t, s, *, system_words=4, user_words=6, assistant_words=10):
    pid, tid = f"p{p:02d}", f"p{p:02d}-t{t:02d}"
    sid = f"{tid}-s{s:02d}"
    sys_text = words(f"as{p}x{t}x{s}s", system_words)
    user = words(f"au{p}x{t}x{s}u", user_words)
    assistant = words(f"aa{p}x{t}x{s}a", assistant_words)
    n = len(("\n\n".join((sys_text, user, assistant))).split())
    return {"system": sys_text, "user": user, "assistant": assistant,
            "principle_id": pid, "prompt_type_id": tid, "scenario_id": sid,
            "lineage": {"custom_id": f"adv-{sid}", "v1_sha256": "0" * 64,
                        "register_edits": ([{"marker": "x"}]
                                           if (p + s) % 3 == 0 else [])},
            "token_counts": {k: n for k in hs.TOK_NAMES}}


def sto_rec(c, u, k, y, *, n_words=40):
    cid, uid = f"c{c:02d}", f"c{c:02d}-u{u:02d}"
    kid = f"{uid}-k{k:02d}"
    text = words(f"st{c}x{u}x{k}x{y}w", n_words)
    return {"text": text, "category_id": cid, "subcategory_id": uid,
            "call_id": kid,
            "lineage": {"custom_id": f"sto-{kid}-y{y}", "v1_sha256": "1" * 64,
                        "register_edits": []},
            "token_counts": {k2: n_words for k2 in hs.TOK_NAMES}}


def write_jsonl(path: Path, records) -> Path:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n"
                            for r in records), encoding="utf-8")
    return path


def fake_budget_manifest(path: Path, budget: int, cell: str = "constv2-mix-full"):
    path.write_text(json.dumps({
        "cell": cell, "lane": "mixed", "budget": None, "n_docs": 7,
        "official_tokens": budget, "limiting_tokenizer": "llama3",
        "per_tokenizer_cumulative_totals": {k: budget for k in hs.TOK_NAMES},
        "pool_sha256": "f" * 64}) + "\n")
    return path


BUDGET = 60_000


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    root = tmp_path_factory.mktemp("ctl2")
    advice = [adv_rec(p, 0, s, system_words=20, user_words=60,
                      assistant_words=120 + (7 * s + 3 * p) % 13)
              for p in range(6) for s in range(10)]
    stories = [sto_rec(c, u, k, y,
                       n_words=350 + (31 * c + 17 * u + 7 * k + 3 * y) % 23)
               for c in range(6) for u in range(5) for k in range(5)
               for y in range(2)]
    ap = write_jsonl(root / "ctl_advice.jsonl", advice)
    sp = write_jsonl(root / "ctl_stories.jsonl", stories)
    bm = fake_budget_manifest(root / "pairs_constv2-mix-full.manifest.json",
                              BUDGET)
    gen = {"sisteer_commit": "test", "sisteer_dirty": False,
           "sisteer_diff_sha256": None,
           "corpus_generator_commit": "test", "corpus_generator_dirty": False,
           "corpus_generator_diff_sha256": None}
    kw = dict(control_advice=ap, control_stories=sp, budget_manifest=bm,
              expect_shas={"control_advice": hs.sha256_file(ap),
                           "control_stories": hs.sha256_file(sp)},
              rig=FakeRig(), generator=gen, eval_n=8)
    m1 = b2.build_ctl2(root / "out1", split_out=root / "split1.json", **kw)
    m2 = b2.build_ctl2(root / "out2", split_out=root / "split2.json", **kw)
    return SimpleNamespace(root=root, advice=advice, stories=stories, ap=ap,
                           sp=sp, bm=bm, m1=m1, m2=m2, kw=kw,
                           out1=root / "out1", out2=root / "out2",
                           split1=json.loads((root / "split1.json").read_text()))


def test_refuses_sha_mismatch(built, tmp_path):
    kw = dict(built.kw, expect_shas={"control_advice": "0" * 64,
                                     "control_stories": built.kw["expect_shas"]["control_stories"]})
    with pytest.raises(bp.SamplerError, match="was asserted"):
        b2.build_ctl2(tmp_path / "o", split_out=tmp_path / "s.json", **kw)
    assert not (tmp_path / "o").exists()  # refused before anything was written
    # an omitted sha is a refusal too — never "anything goes"
    kw = dict(built.kw, expect_shas={"control_advice": built.kw["expect_shas"]["control_advice"]})
    with pytest.raises(bp.SamplerError, match="must be asserted"):
        b2.build_ctl2(tmp_path / "o2", split_out=tmp_path / "s2.json", **kw)


def test_release_pins_enforced_by_default_and_label_is_honest(built, tmp_path):
    """No expect_shas -> the RELEASE's pinned bytes are required (synthetic
    inputs refuse); caller-named bytes build but every manifest says so."""
    kw = dict(built.kw); kw.pop("expect_shas")
    with pytest.raises(bp.SamplerError, match="was asserted"):
        b2.build_ctl2(tmp_path / "o", split_out=tmp_path / "s.json",
                      expect_shas=None, **kw)
    assert set(b2.RELEASE_SHAS) == {"control_advice", "control_stories"}
    assert all(len(v) == 64 for v in b2.RELEASE_SHAS.values())
    m = built.m1[b2.FULL_CELL]
    assert m["corpus"]["release_bytes_pinned"] is False
    assert m["corpus"]["release"].startswith("unpinned(")
    assert m["constdatagen_provenance"]["release_bytes_pinned"] is False
    summary = json.loads((built.out1 / "pools_summary.json").read_text())
    assert summary["release_bytes_pinned"] is False
    prov = json.loads((built.root / "split1.provenance.json").read_text())
    assert prov["release_bytes_pinned"] is False
    assert prov["corpus_shas"] == built.kw["expect_shas"]


def test_refuses_nonempty_out_dir(built, tmp_path):
    out = tmp_path / "o"; out.mkdir(); (out / "stale.json").write_text("{}")
    with pytest.raises(bp.SamplerError, match="not empty"):
        b2.build_ctl2(out, split_out=tmp_path / "s.json", **built.kw)
    assert sorted(p.name for p in out.iterdir()) == ["stale.json"]
    empty = tmp_path / "e"; empty.mkdir()  # an existing EMPTY dir is fine
    b2.build_ctl2(empty, split_out=tmp_path / "s3.json", **built.kw)
    assert (empty / "pools_summary.json").is_file()


def test_verify_pools_agrees_then_catches_a_swapped_pool(built, tmp_path):
    rep = b2.verify_pools(built.out1)
    assert set(rep) == {b2.HELDOUT_CELL, b2.FULL_CELL}
    assert rep[b2.FULL_CELL]["official_tokens"] == BUDGET
    import shutil
    bad = tmp_path / "bad"; shutil.copytree(built.out1, bad)
    pool = bad / f"pairs_{b2.FULL_CELL}.jsonl"
    lines = pool.read_text().splitlines(keepends=True)
    pool.write_text("".join(lines[:-1]))  # one row dropped
    with pytest.raises(bp.SamplerError, match="sha256"):
        b2.verify_pools(bad)
    assert b2.main(["verify", "--pools", str(built.out1)]) == 0


def test_budget_must_come_from_the_mix_full_manifest(built, tmp_path):
    wrong = fake_budget_manifest(tmp_path / "m.json", BUDGET,
                                 cell="constv2-mix-10m")
    with pytest.raises(bp.SamplerError, match="not 'constv2-mix-full'"):
        b2.build_ctl2(tmp_path / "o", split_out=tmp_path / "s.json",
                      **dict(built.kw, budget_manifest=wrong))
    with pytest.raises(bp.SamplerError, match="missing"):
        b2.read_budget(tmp_path / "absent.json")
    src = b2.read_budget(built.bm)
    assert src["official_tokens"] == BUDGET
    assert src["manifest_sha256"] == hs.sha256_file(built.bm)


def test_determinism_bytewise(built):
    for name in ("pairs_constv2ctl2-full.jsonl",
                 "pairs_constv2ctl2-heldout.jsonl",
                 "pools_summary.json"):
        assert (built.out1 / name).read_bytes() == (built.out2 / name).read_bytes(), name
    assert (built.root / "split1.json").read_bytes() == \
        (built.root / "split2.json").read_bytes()
    # The manifests record the split ARTIFACT'S PATH (split1 vs split2 here,
    # by the fixture's choice); everything else is byte-identical.
    for name in ("pairs_constv2ctl2-full.manifest.json",
                 "pairs_constv2ctl2-heldout.manifest.json"):
        a = json.loads((built.out1 / name).read_text())
        b = json.loads((built.out2 / name).read_text())
        assert a["split"].pop("artifact") != b["split"].pop("artifact")
        assert a == b, name


def test_cells_prefix_and_lanes(built):
    mf = built.m1[b2.FULL_CELL]
    mh = built.m1[b2.HELDOUT_CELL]
    assert mf["lane"] == "control" and mh["lane"] == "control_heldout"
    for m in (mf, mh):
        assert all(i.startswith("ctl2-") for i in m["doc_ids_in_order"])
        assert not any(i.startswith("ctl-") for i in m["doc_ids_in_order"])
        rows = [json.loads(l) for l in
                (built.out1 / f"pairs_{m['cell']}.jsonl").read_text().splitlines()]
        assert [r["doc_id"] for r in rows] == m["doc_ids_in_order"]
        assert all(r["instruction_id"] == m["cell"] for r in rows)
    assert mf["corpus"]["id_prefix"] == "ctl2-"
    assert mf["truncation"]["doc_id"].startswith("ctl2-")


def test_budget_exact_and_from_manifest(built):
    mf = built.m1[b2.FULL_CELL]
    assert mf["budget"] == BUDGET
    assert mf["official_tokens"] == BUDGET
    assert min(mf["per_tokenizer_cumulative_totals"].values()) == BUDGET
    assert mf["budget_source"]["cell"] == "constv2-mix-full"
    assert mf["budget_source"]["official_tokens"] == BUDGET
    # the control lane's ratio gate applies from 100k up (record mode);
    # the synthetic budget sits below that, and the shape is recorded either way
    assert mf["ratio_gate"]["applied"] is (BUDGET >= bp.RATIO_GATE_MIN_BUDGET)
    assert mf["ratio_gate"]["tolerance"] == bp.RATIO_TOL
    assert mf["ratio_gate"]["deviation"] == pytest.approx(
        abs(mf["realized_ratio"] - mf["natural_ratio"]))


def test_heldout_split_is_the_constitutional_carve_on_control(built):
    split = built.split1
    assert split["salt"] == hs.SALT_HELDOUT and split["frac"] == hs.FRAC
    assert split["eval_subset"]["salt"] == hs.SALT_EVAL
    assert split["built_from_sha256"] == {
        "advice_final": hs.sha256_file(built.ap),
        "stories_final": hs.sha256_file(built.sp)}
    # whole groups: no A3 sibling group straddles the split
    held = set(split["advice"]["ids"])
    by_group: dict[str, set] = {}
    for r in built.advice:
        by_group.setdefault(hs.advice_group_key(r), set()).add(
            hs.doc_id(r) in held)
    assert all(len(v) == 1 for v in by_group.values())
    held_s = set(split["stories"]["ids"])
    by_sub: dict[str, set] = {}
    for r in built.stories:
        by_sub.setdefault(r["subcategory_id"], set()).add(hs.doc_id(r) in held_s)
    assert all(len(v) == 1 for v in by_sub.values())
    mh = built.m1[b2.HELDOUT_CELL]
    assert mh["n_docs"] == 8
    assert set(mh["doc_ids_in_order"]) == {
        "ctl2-" + i for i in split["eval_subset"]["ids"]}
    assert mh["budget"] is None and mh["truncation"] is None


def test_full_is_a_prefix_of_one_draw_with_disjoint_remainder(built):
    mf = built.m1[b2.FULL_CELL]
    d = mf["disjointness"]
    assert d["full_x_heldout"] == 0 and d["full_x_eval_subset"] == 0
    assert d["full_x_remainder"] == 0 and d["remainder_x_heldout"] == 0
    assert d["eval_subset_within_heldout"] is True
    assert d["full_plus_remainder_equals_train"] is True
    assert d["train_plus_heldout_equals_corpus"] is True
    # rebuild the draw independently and check prefix + remainder = order
    tokens_mod = hs.generator_tokens()
    held = set(built.split1["advice"]["ids"]) | set(built.split1["stories"]["ids"])
    adv = [bp.make_doc(r, "advice", tokens_mod) for r in built.advice
           if hs.doc_id(r) not in held]
    sto = [bp.make_doc(r, "control_stories", tokens_mod) for r in built.stories
           if hs.doc_id(r) not in held]
    ta = min(bp._token_totals(adv).values()); ts = min(bp._token_totals(sto).values())
    nat = ta / (ta + ts)
    order = bp.greedy_interleave(
        {"advice": bp.sha_shuffle(adv, bp.SALT_CTL),
         "stories": bp.sha_shuffle(sto, bp.SALT_CTL)},
        {"advice": nat, "stories": 1 - nat})
    ids = ["ctl2-" + x.doc_id for x in order]
    n = mf["n_docs"]
    assert mf["doc_ids_in_order"] == ids[:n]
    assert mf["remainder"]["doc_ids_in_order"] == ids[n:]
    assert mf["nesting"]["prefix_n_docs"] == n
    assert mf["nesting"]["order_n_docs"] == len(ids) == mf["train_side"]["n_docs"]
    assert mf["nesting"]["draw_salt"] == bp.SALT_CTL
    assert mf["train_side"]["natural_ratio"] == pytest.approx(nat)
    assert mf["natural_ratio"] == pytest.approx(nat)
    # remainder token totals are the whole docs after the cut
    rem = order[n:]
    assert mf["remainder"]["per_tokenizer_totals"] == bp._token_totals(rem)
    assert mf["remainder"]["n_docs"] == len(rem) > 0


def test_manifest_records_corpus_lineage_and_release(built):
    mf = built.m1[b2.FULL_CELL]
    c = mf["corpus"]
    assert b2.CORPUS_RELEASE in c["release"]  # unpinned(...) in the fixture
    assert c["n_records"] == {"advice": 60, "stories": 300}
    assert c["register_edited_records"]["advice"] == sum(
        1 for r in built.advice if r["lineage"]["register_edits"])
    assert c["v1_lineage_records"] == {"advice": 60, "stories": 300}
    assert c["official_tokens"]["combined"] == (
        c["official_tokens"]["advice"] + c["official_tokens"]["stories"])
    assert mf["corpus_shas"] == c["corpus_shas"] == built.kw["expect_shas"]
    assert mf["constdatagen_provenance"]["release"] == c["release"]
    assert mf["heldout_split_sha"] == hs.sha256_file(built.root / "split1.json")
    assert mf["split"]["sha256"] == mf["heldout_split_sha"]
    summary = json.loads((built.out1 / "pools_summary.json").read_text())
    assert set(summary["cells"]) == {b2.FULL_CELL, b2.HELDOUT_CELL}
    assert b2.CORPUS_RELEASE in summary["corpus"]


def test_cli_refuses_dirty_tree_unless_allowed(monkeypatch, tmp_path):
    monkeypatch.setattr(hs, "generator_provenance",
                        lambda: {"sisteer_commit": "x",
                                 "sisteer_dirty": True,
                                 "sisteer_diff_sha256": "d",
                                 "corpus_generator_commit": "y",
                                 "corpus_generator_dirty": False,
                                 "corpus_generator_diff_sha256": None})
    rc = b2.main(["build", "--control-advice", str(tmp_path / "a.jsonl"),
                  "--control-stories", str(tmp_path / "s.jsonl"),
                  "--out", str(tmp_path / "o"),
                  "--split-out", str(tmp_path / "s.json")])
    assert rc == 2
    assert not (tmp_path / "o").exists()


# ---------------------------------------------------------------------------
# Seed replicates: salt threading, the
# idempotent split write, and the default salts reproducing the committed
# r2 control split
# ---------------------------------------------------------------------------

def test_salt_threading_reaches_split_and_manifests(built, tmp_path):
    salt, eval_salt = "sdfbase-heldout-v1-s9", "sdfbase-heldout-eval-v1-s9"
    m = b2.build_ctl2(tmp_path / "o", split_out=tmp_path / "s9.json",
                      salt=salt, eval_salt=eval_salt, **built.kw)
    split = json.loads((tmp_path / "s9.json").read_text())
    assert split["salt"] == salt and split["eval_subset"]["salt"] == eval_salt
    assert split["frac"] == hs.FRAC and "pins" not in split
    for cell in (b2.FULL_CELL, b2.HELDOUT_CELL):
        assert m[cell]["seed_salts"]["heldout"] == salt
        assert m[cell]["split"]["salt"] == salt
        assert m[cell]["split"]["eval_salt"] == eval_salt
    assert m[b2.HELDOUT_CELL]["seed_salts"]["draw"] == eval_salt
    # the draw salt of the full cell is the control lane's, unchanged
    assert m[b2.FULL_CELL]["seed_salts"]["draw"] == bp.SALT_CTL
    # a different carve salt moves the held-out set (and so the cut)
    held_default = set(built.split1["stories"]["ids"])
    held_seeded = set(split["stories"]["ids"])
    assert held_seeded != held_default
    assert m[b2.FULL_CELL]["doc_ids_in_order"] != \
        built.m1[b2.FULL_CELL]["doc_ids_in_order"]
    prov = json.loads((tmp_path / "s9.provenance.json").read_text())
    assert prov["carve"]["salt"] == salt and prov["carve"]["eval_salt"] == eval_salt
    assert prov["release_bytes_pinned"] is False
    assert prov["corpus_shas"] == built.kw["expect_shas"]


def test_cli_threads_salt_flags_into_build_ctl2(monkeypatch, tmp_path):
    monkeypatch.setattr(hs, "generator_provenance",
                        lambda: {"sisteer_commit": "x", "sisteer_dirty": False,
                                 "sisteer_diff_sha256": None,
                                 "corpus_generator_commit": "y",
                                 "corpus_generator_dirty": False,
                                 "corpus_generator_diff_sha256": None})
    seen = {}

    def capture(out_dir, **kw):
        seen.update(kw)
        raise bp.SamplerError("captured")

    monkeypatch.setattr(b2, "build_ctl2", capture)
    rc = b2.main(["build", "--control-advice", str(tmp_path / "a.jsonl"),
                  "--control-stories", str(tmp_path / "s.jsonl"),
                  "--budget-manifest", str(tmp_path / "m.json"),
                  "--out", str(tmp_path / "o"), "--split-out", str(tmp_path / "s.json"),
                  "--salt", "sdfbase-heldout-v1-s2",
                  "--eval-salt", "sdfbase-heldout-eval-v1-s2"])
    assert rc == 2 and seen["salt"] == "sdfbase-heldout-v1-s2"
    assert seen["eval_salt"] == "sdfbase-heldout-eval-v1-s2"
    seen.clear()
    b2.main(["build", "--control-advice", str(tmp_path / "a.jsonl"),
             "--control-stories", str(tmp_path / "s.jsonl"),
             "--out", str(tmp_path / "o"), "--split-out", str(tmp_path / "s.json")])
    assert seen["salt"] == hs.SALT_HELDOUT and seen["eval_salt"] == hs.SALT_EVAL


def test_write_ctl2_split_is_idempotent_on_unchanged_bytes(built, tmp_path):
    split = dict(built.split1)
    out = tmp_path / "ctl2_heldout_split.s9.json"
    kw = dict(shas=built.kw["expect_shas"], pinned=False, release_label="unpinned(test)")
    gen_a = {"sisteer_commit": "A"}
    gen_b = {"sisteer_commit": "B"}
    sha1 = b2.write_ctl2_split(split, out, generator=gen_a, **kw)
    prov = out.with_suffix(".provenance.json")
    assert json.loads(prov.read_text())["generator"] == gen_a
    assert json.loads(prov.read_text())["artifact_sha256"] == sha1
    # same bytes again under another tree: nothing is touched
    before = (out.read_bytes(), prov.read_bytes())
    sha2 = b2.write_ctl2_split(split, out, generator=gen_b, **kw)
    assert sha2 == sha1 and (out.read_bytes(), prov.read_bytes()) == before
    # different bytes overwrite artifact AND sidecar
    changed = json.loads(json.dumps(split))
    changed["stories"]["ids"] = changed["stories"]["ids"][:-1]
    sha3 = b2.write_ctl2_split(changed, out, generator=gen_b, **kw)
    assert sha3 != sha1
    assert json.loads(prov.read_text())["generator"] == gen_b
    assert json.loads(prov.read_text())["artifact_sha256"] == sha3
    # a missing sidecar next to identical bytes is rewritten (never left absent)
    prov.unlink()
    b2.write_ctl2_split(changed, out, generator=gen_a, **kw)
    assert json.loads(prov.read_text())["generator"] == gen_a


# ---------------------------------------------------------------------------
# The recipe identity (P1 follow-up): what was sampled and how, recorded
# on both manifests so ctl2_pools_reuse.py can check a completed build against
# today's builder — never the checkout's commit
# ---------------------------------------------------------------------------

def test_manifests_record_the_recipe_identity_without_a_commit(built):
    mf, mh = built.m1[b2.FULL_CELL], built.m1[b2.HELDOUT_CELL]
    recipe = mf["recipe"]
    assert mh["recipe"] == recipe                                # ONE build, one recipe on both cells
    assert list(recipe) == ["version", "release", "release_bytes_pinned", "corpus_shas", "id_prefix", "cells",
                            "budget_cell", "salts", "split", "tokenizers"]
    assert recipe["version"] == b2.CTL2_RECIPE_VERSION == "ctl2-recipe-v1"
    assert recipe["release"] == mf["corpus"]["release"] and recipe["release_bytes_pinned"] is False
    assert recipe["corpus_shas"] == built.kw["expect_shas"]
    assert recipe["id_prefix"] == "ctl2-" and recipe["cells"] == [b2.HELDOUT_CELL, b2.FULL_CELL]
    assert recipe["budget_cell"] == b2.BUDGET_CELL == "constv2-mix-full"
    assert recipe["salts"] == {"heldout": hs.SALT_HELDOUT, "eval": hs.SALT_EVAL, "draw": bp.SALT_CTL}
    assert recipe["split"] == {"frac": hs.FRAC, "eval_n": 8}         # the fixture's eval_n, the default frac
    tok = recipe["tokenizers"]
    pins_file = ROOT / "corpora/TOKENIZERS.json"
    pins = json.loads(pins_file.read_text())["tokenizers"]
    assert tok["pins_file"] == "corpora/TOKENIZERS.json"          # repo-relative, never an absolute path
    assert tok["pins_sha256"] == hs.sha256_file(pins_file)
    assert tok["pins"] == {n: {"repo": t["repo"], "revision": t["revision"],
                               "files": {f["name"]: f["sha256"] for f in t["files"]}} for n, t in pins.items()}
    assert tok["rig"] == {"fake": True}                          # the rig's own provenance rides along
    # the same block on disk (both manifests) and in the summary
    for cell in (b2.FULL_CELL, b2.HELDOUT_CELL):
        on_disk = json.loads((built.out1 / f"pairs_{cell}.manifest.json").read_text())
        assert on_disk["recipe"] == recipe
    assert json.loads((built.out1 / "pools_summary.json").read_text())["recipe"] == recipe
    # NEVER the checkout's commit nor anything of the generator provenance (a metadata-only change must not invalidate a
    # completed build); no absolute home path
    text = json.dumps(recipe)
    gen = built.kw["generator"]
    assert not any(k in text for k in gen) and not any(str(v) in text for v in gen.values() if v)
    assert str(Path.home()) not in text and str(built.root) not in text
    # the builder's recipe_identity IS the block (minus the rig it adds after)
    declared = b2.recipe_identity(shas=built.kw["expect_shas"], pinned=False, release_label=mf["corpus"]["release"],
                                  salt=hs.SALT_HELDOUT, eval_salt=hs.SALT_EVAL, frac=hs.FRAC, eval_n=8,
                                  tokenizer_pins=b2.load_tokenizer_pins())
    declared["tokenizers"]["rig"] = {"fake": True}
    assert declared == recipe


def test_recipe_records_threaded_salts_and_refuses_without_pins(built, tmp_path, monkeypatch):
    salt, eval_salt = "sdfbase-heldout-v1-s3", "sdfbase-heldout-eval-v1-s3"
    m = b2.build_ctl2(tmp_path / "o", split_out=tmp_path / "s3.json",
                      salt=salt, eval_salt=eval_salt, **built.kw)
    for cell in (b2.FULL_CELL, b2.HELDOUT_CELL):
        assert m[cell]["recipe"]["salts"] == {"heldout": salt, "eval": eval_salt, "draw": bp.SALT_CTL}
    assert m[b2.FULL_CELL]["recipe"]["salts"] != built.m1[b2.FULL_CELL]["recipe"]["salts"]
    # a missing / malformed pins file refuses BEFORE anything is written
    monkeypatch.setattr(b2, "TOKENIZER_PINS", tmp_path / "absent.json")
    with pytest.raises(bp.SamplerError, match="tokenizer pins"):
        b2.build_ctl2(tmp_path / "o2", split_out=tmp_path / "s4.json", **built.kw)
    assert not (tmp_path / "o2").exists() and not (tmp_path / "s4.json").exists()
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"tokenizers": {"llama3": {"repo": "x/y"}}}))
    with pytest.raises(bp.SamplerError, match="llama3"):
        b2.load_tokenizer_pins(bad)
    with pytest.raises(bp.SamplerError, match="no 'tokenizers' map"):
        b2.load_tokenizer_pins(tmp_path / "s3.json")


CTL_FINALS = (hs.GENERATOR_ROOT / "data/control/final-control-scale-20260903-r2-advice.jsonl",
              hs.GENERATOR_ROOT / "data/control/final-control-scale-20260903-r2-stories.jsonl")
COMMITTED_CTL2_SPLIT = ROOT / "data/sdfbase/ctl2_heldout_split.json"


@pytest.mark.skipif(not all(p.exists() for p in CTL_FINALS),
                    reason="r2 control finals not present")
def test_default_salts_reproduce_the_committed_ctl2_split():
    """The seed-salt parameters default to the campaign's constitutional
    salts, so a default control carve is the committed r2 split, byte for
    byte (the threading changed nothing for the existing lane)."""
    split = hs.build_split(*CTL_FINALS)
    fresh = (json.dumps(split, indent=1, ensure_ascii=False) + "\n").encode("utf-8")
    assert COMMITTED_CTL2_SPLIT.read_bytes() == fresh
    assert split["salt"] == hs.SALT_HELDOUT
    assert split["eval_subset"]["salt"] == hs.SALT_EVAL
