"""check_pool_integrity + the seed pool driver on a synthetic corpus.

One synthetic end-to-end run of ``build_seed_pools`` (splits stage, then
pools stage with a fake counting rig and fake target tokenizers) stands in
for a seed-2-style single-mixed-cell key; the tests then pin that
``check_pool_integrity`` passes the correct directories, refuses a tampered
``pool_sha256``, a held-out doc inside ``constv2-mix-full`` and a ctl2 budget
that differs from the constitution manifest, reports nesting ``n/a`` for one
mixed cell and checks it for a ladder; that the driver budget-matches the
control cut to the seed's own ``constv2-mix-full``, lays out the volume dir,
refuses a non-empty output dir, and leaves a committed control split's
provenance untouched when it merely reproduces the bytes.

Synthetic corpus + fake rig (the ``tests/test_build_ctl2_pools.py``
conventions); the corpus repo's renderers are imported by path, so the
module skips where that checkout is absent.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "scripts" / "sdfbase", ROOT / "scripts" / "sdfbase" / "seedrep",
           ROOT / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import build_heldout_split as hs  # noqa: E402
import build_pools as bp  # noqa: E402
import build_seed_pools as bsp  # noqa: E402
import check_pool_integrity as cpi  # noqa: E402
from test_build_ctl2_pools import FakeRig, adv_rec, sto_rec, write_jsonl  # noqa: E402

try:
    hs.generator_tokens()
except hs.SplitError as _e:  # pragma: no cover
    pytest.skip(f"corpus repo renderers unavailable: {_e}",
                allow_module_level=True)


class FakeTok:
    def __init__(self, key: str, bos: bool):
        self.model_key, self.bos = key, bos
        self.repo, self.revision = f"fake/{key}", "0" * 40

    def n_tokens(self, text: str, add_special_tokens: bool) -> int:
        return len(text.split()) + (1 if (add_special_tokens and self.bos) else 0)


TOKS = {"fake_a": FakeTok("fake_a", True), "fake_b": FakeTok("fake_b", False)}
GEN_SPLITS = {"sisteer_commit": "splits-commit", "sisteer_dirty": False,
              "sisteer_diff_sha256": None, "corpus_generator_commit": "c",
              "corpus_generator_dirty": False, "corpus_generator_diff_sha256": None}
GEN_POOLS = dict(GEN_SPLITS, sisteer_commit="pools-commit")
SEED = 2


def probe_item(source_id: str) -> dict:
    return {"id": f"hd-{source_id}", "source": "heldout_doc",
            "source_id": source_id, "prefix": "p", "cont_consistent": "a",
            "cont_violating": "b"}


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    root = tmp_path_factory.mktemp("seedpools")
    data = root / "data"
    data.mkdir()
    # constitution: 100 advice groups x 2; 10 cats x 10 subcats x 3 calls x 2
    advice = [adv_rec(p, 0, s, system_words=20, user_words=60,
                      assistant_words=120 + (7 * s + 3 * p) % 13)
              for p in range(100) for s in range(2)]
    stories = [sto_rec(c, u, k, y, n_words=350 + (31 * c + 17 * u + 7 * k + 3 * y) % 23)
               for c in range(10) for u in range(10) for k in range(3) for y in range(2)]
    ap = write_jsonl(root / "advice.jsonl", advice)
    sp = write_jsonl(root / "stories.jsonl", stories)
    # control: distinct ids, bigger than the constitution train side
    cadv = [adv_rec(p, 1, s, system_words=20, user_words=60,
                    assistant_words=125 + (5 * s + 3 * p) % 11)
            for p in range(100) for s in range(2)]
    csto = [sto_rec(c, u, k, y, n_words=360 + (29 * c + 13 * u + 7 * k + 3 * y) % 19)
            for c in range(10, 22) for u in range(10) for k in range(3) for y in range(2)]
    cap = write_jsonl(root / "ctl_advice.jsonl", cadv)
    csp = write_jsonl(root / "ctl_stories.jsonl", csto)
    expect = {"control_advice": hs.sha256_file(cap),
              "control_stories": hs.sha256_file(csp)}
    probe = [probe_item(hs.doc_id(r)) for r in advice if r["principle_id"] == "p07"]
    probe += [probe_item(hs.doc_id(r)) for r in stories
              if r["subcategory_id"] in ("c03-u04", "c08-u01")
              and r["call_id"].endswith("k01")]
    pp = write_jsonl(root / "probe.jsonl", probe)
    common = dict(advice_final=ap, stories_final=sp, control_advice=cap,
                  control_stories=csp, probes=[pp], data_dir=data,
                  expect_shas=expect, eval_n=16, log=lambda *_a, **_k: None)
    out_root = root / "out"
    splits = bsp.stage_splits(SEED, frac=0.085, frac_reason=None,
                              generator=GEN_SPLITS, **common)
    pools = bsp.stage_pools(SEED, frac=0.085, generator=GEN_POOLS,
                            out_root=out_root, register_edit=None,
                            rig=FakeRig(), toks=TOKS, **common)
    split_path, ctl2_split_path = bsp.split_paths(SEED, data)
    return SimpleNamespace(
        root=root, data=data, out_root=out_root, advice=advice, stories=stories,
        ap=ap, sp=sp, cap=cap, csp=csp, pp=pp, expect=expect, common=common,
        splits=splits, pools=pools, split_path=split_path,
        ctl2_split_path=ctl2_split_path,
        pools_dir=out_root / f"pools_s{SEED}",
        ctl2_dir=out_root / f"pools_ctl2_s{SEED}",
        vol_dir=out_root / "vol/pool" / bsp.key_of(SEED))


def _run(built, pools=None, ctl2=None):
    return cpi.run(pools=pools or built.pools_dir, ctl2=ctl2 or built.ctl2_dir,
                   split=built.split_path, ctl2_split=built.ctl2_split_path,
                   advice_final=built.ap, stories_final=built.sp,
                   control_advice=built.cap, control_stories=built.csp)


def _failed(rep) -> list[str]:
    return [c["name"] for c in rep["failures"]]


# ---------------------------------------------------------------------------
# The driver end to end
# ---------------------------------------------------------------------------

def test_driver_budget_matches_control_to_the_seeds_own_mix_full(built):
    bm = built.pools["budget_match"]
    assert (bm["ctl2_budget"] == bm["ctl2_full_official_tokens"]
            == bm["constitution_mix_full_official_tokens"])
    mf = json.loads((built.pools_dir / "pairs_constv2-mix-full.manifest.json").read_text())
    cf = json.loads((built.ctl2_dir / "pairs_constv2ctl2-full.manifest.json").read_text())
    assert cf["budget"] == mf["official_tokens"]
    assert cf["budget_source"]["manifest_sha256"] == hs.sha256_file(
        built.pools_dir / "pairs_constv2-mix-full.manifest.json")
    assert cf["seed_salts"]["heldout"] == f"sdfbase-heldout-v1-s{SEED}"
    assert cf["split"]["eval_salt"] == f"sdfbase-heldout-eval-v1-s{SEED}"
    assert mf["seed_salts"] == {"draw": "sdfbase-draw-v1",
                                "heldout": f"sdfbase-heldout-v1-s{SEED}"}
    assert built.pools["nesting"] == cpi.NESTING_NA
    assert built.pools["integrity"]["ok"] and built.pools["probe_pins"]["ok"]
    assert built.pools["cells"].keys() == {"constv2-mix-full", "constv2-heldout",
                                           "constv2ctl2-full", "constv2ctl2-heldout"}
    assert built.pools["generator"] == GEN_POOLS
    assert (built.out_root / "reports" / f"seed_s{SEED}.json").is_file()
    assert (built.out_root / "reports" / f"integrity_s{SEED}.json").is_file()
    assert (built.out_root / "reports" / f"probe_pins_s{SEED}.json").is_file()


def test_driver_volume_layout_and_manifest(built):
    names = sorted(p.name for p in built.vol_dir.iterdir())
    assert names == sorted(f"pairs_{c}{sfx}" for c in bsp.volume_cells_of(SEED)
                           for sfx in (".jsonl", ".manifest.json"))
    for name, rec in built.pools["volume_files"].items():
        assert rec["sha256"] == hs.sha256_file(built.vol_dir / name)
        src = built.ctl2_dir if "ctl2" in name else built.pools_dir
        assert (built.vol_dir / name).read_bytes() == (src / name).read_bytes()
    out = bsp.write_pools_manifest(built.out_root)
    pm = json.loads(out.read_text())
    key = pm["keys"][bsp.key_of(SEED)]
    assert key["n_files"] == 6
    assert key["files"]["pairs_constv2-mix-full.jsonl"]["n_rows"] == \
        built.pools["cells"]["constv2-mix-full"]["n_docs"]
    assert pm["locations"]["modal_volume"] is None


def test_driver_keeps_the_committed_control_split_provenance(built):
    """The pools stage re-derives the control split (same bytes) — the
    artifact's sidecar must still name the SPLITS stage's tree."""
    prov = json.loads(built.ctl2_split_path.with_suffix(".provenance.json").read_text())
    assert prov["generator"]["sisteer_commit"] == "splits-commit"
    assert prov["artifact_sha256"] == hs.sha256_file(built.ctl2_split_path)
    assert prov["carve"]["salt"] == f"sdfbase-heldout-v1-s{SEED}"
    assert prov["release_bytes_pinned"] is False  # synthetic control bytes
    cprov = json.loads(built.split_path.with_suffix(".provenance.json").read_text())
    assert cprov["carve"]["seed"] == SEED and cprov["carve"]["pinned"] is True
    assert cprov["carve"]["key"] == bsp.key_of(SEED)


def test_driver_refuses_nonempty_dirs_and_missing_splits(built, tmp_path):
    with pytest.raises(bsp.SeedPoolsError, match="not empty"):
        bsp.stage_pools(SEED, frac=0.085, generator=GEN_POOLS,
                        out_root=built.out_root, register_edit=None,
                        rig=FakeRig(), toks=TOKS, **built.common)
    with pytest.raises(bsp.SeedPoolsError, match="already exists"):
        bsp.stage_splits(SEED, frac=0.085, frac_reason=None,
                         generator=GEN_SPLITS, **built.common)


def test_driver_refuses_a_control_split_that_no_longer_reproduces(built, tmp_path):
    """A committed control split whose bytes the build cannot reproduce
    (corpus or builder drift) is a refusal, never a silent overwrite."""
    data = tmp_path / "data"
    shutil.copytree(built.data, data)
    _, ctl2 = bsp.split_paths(SEED, data)
    sp = json.loads(ctl2.read_text())
    sp["stories"]["ids"] = sp["stories"]["ids"][:-1]
    ctl2.write_text(json.dumps(sp, indent=1, ensure_ascii=False) + "\n")
    common = dict(built.common, data_dir=data)
    with pytest.raises(bsp.SeedPoolsError, match="DIFFERENT control split"):
        bsp.stage_pools(SEED, frac=0.085, generator=GEN_POOLS,
                        out_root=tmp_path / "o", register_edit=None,
                        rig=FakeRig(), toks=TOKS, **common)


# ---------------------------------------------------------------------------
# check_pool_integrity
# ---------------------------------------------------------------------------

def test_integrity_passes_a_correct_single_cell_dir(built):
    rep = _run(built)
    assert rep["ok"], _failed(rep)
    assert rep["nesting"] == cpi.NESTING_NA
    assert rep["constitution"]["nesting"] == cpi.NESTING_NA
    names = [c["name"] for c in rep["checks"]]
    assert any("doc set == corpus − held-out EXACTLY" in n for n in names)
    assert any("official_tokens == budget == budget_source" in n for n in names)
    assert any("prefix ∪ remainder == control train side" in n for n in names)
    assert any("ids == split eval_subset.ids" in n for n in names)
    assert any("per_tokenizer totals recomputed" in n for n in names)
    assert cpi.main(["--pools", str(built.pools_dir), "--ctl2", str(built.ctl2_dir),
                     "--split", str(built.split_path),
                     "--ctl2-split", str(built.ctl2_split_path),
                     "--advice-final", str(built.ap), "--stories-final", str(built.sp),
                     "--control-advice", str(built.cap),
                     "--control-stories", str(built.csp)]) == 0


def test_integrity_refuses_a_tampered_pool_sha(built, tmp_path):
    bad = tmp_path / "pools"
    shutil.copytree(built.pools_dir, bad)
    mp = bad / "pairs_constv2-mix-full.manifest.json"
    m = json.loads(mp.read_text())
    m["pool_sha256"] = "0" * 64
    mp.write_text(json.dumps(m))
    rep = _run(built, pools=bad)
    assert rep["ok"] is False
    assert any("pool_sha256 == sha256(jsonl)" in n for n in _failed(rep))
    # a dropped row is caught by the sha AND the row/id checks
    bad2 = tmp_path / "pools2"
    shutil.copytree(built.pools_dir, bad2)
    pool = bad2 / "pairs_constv2-mix-full.jsonl"
    lines = pool.read_text().splitlines(keepends=True)
    pool.write_text("".join(lines[:-1]))
    rep = _run(built, pools=bad2)
    assert any("doc_ids_in_order == file rows" in n for n in _failed(rep))
    assert any("n_docs == rows" in n for n in _failed(rep))


def test_integrity_refuses_a_heldout_doc_inside_mix_full(built, tmp_path):
    """Plant a held-out doc as a mix-full row with the manifest made
    CONSISTENT with the file, so only the membership checks can fire."""
    bad = tmp_path / "pools"
    shutil.copytree(built.pools_dir, bad)
    split = json.loads(built.split_path.read_text())
    leaked = split["stories"]["ids"][0]
    pool = bad / "pairs_constv2-mix-full.jsonl"
    with open(pool, "a", encoding="utf-8") as f:
        f.write(json.dumps({"prompt": "", "pos": "leaked text", "neg": "",
                            "instruction_id": "constv2-mix-full",
                            "doc_id": leaked, "family": "stories",
                            "official_tokens": 2, "truncated": False}) + "\n")
    mp = bad / "pairs_constv2-mix-full.manifest.json"
    m = json.loads(mp.read_text())
    m["doc_ids_in_order"].append(leaked)
    m["n_docs"] += 1
    m["pool_sha256"] = hs.sha256_file(pool)
    mp.write_text(json.dumps(m))
    rep = _run(built, pools=bad)
    assert rep["ok"] is False
    failed = _failed(rep)
    assert any("doc set ⊆ corpus − held-out" in n for n in failed)
    assert any("doc set == corpus − held-out EXACTLY" in n for n in failed)
    assert any("per_tokenizer totals recomputed" in n for n in failed)
    assert not any("pool_sha256 == sha256(jsonl)" in n for n in failed)


def test_integrity_refuses_a_ctl2_budget_off_the_constitution_manifest(built, tmp_path):
    bad = tmp_path / "ctl2"
    shutil.copytree(built.ctl2_dir, bad)
    mp = bad / "pairs_constv2ctl2-full.manifest.json"
    m = json.loads(mp.read_text())
    m["budget"] += 1
    m["budget_source"]["official_tokens"] += 1
    mp.write_text(json.dumps(m))
    rep = _run(built, ctl2=bad)
    assert rep["ok"] is False
    failed = _failed(rep)
    assert any("official_tokens == budget == budget_source == constitution" in n
               for n in failed)
    assert any("official_tokens == budget" in n and "budget_source" not in n
               for n in failed)


def test_integrity_refuses_a_heldout_pool_off_the_eval_subset(built, tmp_path):
    bad = tmp_path / "pools"
    shutil.copytree(built.pools_dir, bad)
    pool = bad / "pairs_constv2-heldout.jsonl"
    lines = pool.read_text().splitlines(keepends=True)
    pool.write_text("".join(reversed(lines)))
    mp = bad / "pairs_constv2-heldout.manifest.json"
    m = json.loads(mp.read_text())
    m["doc_ids_in_order"] = list(reversed(m["doc_ids_in_order"]))
    m["pool_sha256"] = hs.sha256_file(pool)
    mp.write_text(json.dumps(m))
    rep = _run(built, pools=bad)
    assert any("ids == split eval_subset.ids (ordered)" in n for n in _failed(rep))


def test_integrity_nesting_is_checked_for_a_ladder(built, tmp_path):
    lad = tmp_path / "ladder"
    bp.build_pools(lad, built.split_path, advice_final=built.ap,
                   stories_final=built.sp, lanes=("mixed", "heldout"),
                   mix_cells=[("constv2-mix-1k", 1_000), ("constv2-mix-full", None)],
                   generator=GEN_POOLS, rig=FakeRig())
    rep = cpi.Report()
    out = cpi.check_constitution(rep, lad, built.split_path,
                                 cpi.corpus_index(built.ap, built.sp))
    assert rep.ok, [c["name"] for c in rep.failures()]
    assert out["nesting"] == "constv2-mix-1k ⊂ constv2-mix-full"
    # the 1k cell is a budget cell: official == budget, one truncated row
    names = [c["name"] for c in rep.checks]
    assert any("constv2-mix-1k: official_tokens == budget" in n for n in names)
    # a divergence in the smaller cell's order is a nesting FAILURE, named
    mp = lad / "pairs_constv2-mix-1k.manifest.json"
    m = json.loads(mp.read_text())
    pool = lad / "pairs_constv2-mix-1k.jsonl"
    rows = [json.loads(l) for l in pool.read_text().splitlines()]
    rows[0], rows[1] = rows[1], rows[0]
    pool.write_text("".join(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n"
                            for r in rows))
    m["doc_ids_in_order"] = [r["doc_id"] for r in rows]
    m["pool_sha256"] = hs.sha256_file(pool)
    mp.write_text(json.dumps(m))
    rep2 = cpi.Report()
    out2 = cpi.check_constitution(rep2, lad, built.split_path,
                                  cpi.corpus_index(built.ap, built.sp))
    assert not rep2.ok
    assert out2["nesting"].startswith("FAILED")
    assert any("nesting" in c["name"] for c in rep2.failures())


# ---------------------------------------------------------------------------
# Nesting across a budget-boundary skip (the s1 ladder's constv2-mix-1m
# carries one) and --stage checks over existing dirs
# ---------------------------------------------------------------------------

def _m(cell, ids, trunc=None, skips=()):
    return {"cell": cell, "doc_ids_in_order": list(ids),
            "truncation": ({"doc_id": trunc} if trunc else None),
            "boundary_skips": [{"doc_id": s, "reason": "planted"} for s in skips]}


def test_nesting_skip_aware_accepts_the_d9_shape_and_names_violations():
    order = [f"d{i}" for i in range(10)]
    # small = whole d0..d3, skipped d4 (no exact cutoff), boundary d5 truncated
    small = _m("constv2-mix-1m", order[:4] + ["d5"], trunc="d5", skips=["d4"])
    big = _m("constv2-mix-10m", order[:8], trunc="d7")
    full = _m("constv2-mix-full", order)
    verdicts = cpi.nesting_skip_aware([small, big, full])
    assert verdicts[0] == {"small": "constv2-mix-1m", "big": "constv2-mix-10m",
                           "whole_docs": 4, "skipped": ["d4"], "boundary_doc": "d5"}
    assert verdicts[1]["skipped"] == [] and verdicts[1]["boundary_doc"] == "d7"
    # exact whole-doc landing: no truncation, plain prefix
    exact = _m("constv2-mix-1k", order[:3])
    assert cpi.nesting_skip_aware([exact, full])[0]["boundary_doc"] is None
    # the skipped doc must sit right after the whole docs in the bigger order
    bad_big = _m("constv2-mix-10m", order[:4] + ["d5", "d4", "d6"], trunc="d6")
    with pytest.raises(bp.SamplerError, match="NOT contained"):
        cpi.nesting_skip_aware([small, bad_big])
    # a divergence among the whole docs
    with pytest.raises(bp.SamplerError, match="first divergence at position 1"):
        cpi.nesting_skip_aware([_m("constv2-mix-1k", ["d0", "d9", "d2"], trunc="d2"), full])
    # skips recorded without a truncated boundary doc is malformed
    with pytest.raises(bp.SamplerError, match="no truncated boundary doc"):
        cpi.nesting_skip_aware([_m("constv2-mix-1k", order[:3], skips=["d3"]), full])
    # the truncated doc must be the cell's last doc
    with pytest.raises(bp.SamplerError, match="not its last doc"):
        cpi.nesting_skip_aware([_m("constv2-mix-1k", order[:3], trunc="d0"), full])
    # a bigger cell shorter than what the smaller needs
    with pytest.raises(bp.SamplerError, match="shorter"):
        cpi.nesting_skip_aware([small, _m("constv2-mix-10m", order[:5], trunc="d4")])


def test_verify_nesting_falls_back_to_skip_aware_only_for_boundary_skips(built, tmp_path):
    lad = tmp_path / "ladder"
    bp.build_pools(lad, built.split_path, advice_final=built.ap,
                   stories_final=built.sp, lanes=("mixed",),
                   mix_cells=[("constv2-mix-1k", 1_000), ("constv2-mix-full", None)],
                   generator=GEN_POOLS, rig=FakeRig())
    label, details = cpi.verify_nesting(lad)
    assert label == "constv2-mix-1k ⊂ constv2-mix-full" and details["mode"] == "strict"
    # plant a CONSISTENT skip in the 1k manifest: its boundary doc becomes
    # the doc after the skipped one in the full order
    mp = lad / "pairs_constv2-mix-1k.manifest.json"
    m = json.loads(mp.read_text())
    full_ids = json.loads((lad / "pairs_constv2-mix-full.manifest.json").read_text())["doc_ids_in_order"]
    whole = m["doc_ids_in_order"][:-1]
    skipped, boundary = full_ids[len(whole)], full_ids[len(whole) + 1]
    m["doc_ids_in_order"] = whole + [boundary]
    m["truncation"]["doc_id"] = boundary
    m["boundary_skips"] = [{"doc_id": skipped, "reason": "no-exact-cutoff", "detail": {}}]
    mp.write_text(json.dumps(m))
    with pytest.raises(bp.SamplerError, match="boundary_skips"):
        bp.verify_mix_nesting(lad)          # the strict check refuses by design
    label, details = cpi.verify_nesting(lad)
    assert details["mode"] == "skip-aware"
    assert label.startswith("constv2-mix-1k ⊂ constv2-mix-full (skip-aware: constv2-mix-1k skipped")
    assert skipped in label and "(D9)" in label
    # an INCONSISTENT skip (the skipped doc is not where the order says) fails
    m["boundary_skips"] = [{"doc_id": full_ids[-1], "reason": "planted", "detail": {}}]
    mp.write_text(json.dumps(m))
    with pytest.raises(bp.SamplerError, match="NOT contained"):
        cpi.verify_nesting(lad)
    # a non-skip strict refusal (divergent prefix) propagates unchanged
    m["boundary_skips"] = []
    m["doc_ids_in_order"] = ["sto-not-in-any-order"] + m["doc_ids_in_order"][1:]
    mp.write_text(json.dumps(m))
    with pytest.raises(bp.SamplerError, match="NOT a doc-id prefix"):
        cpi.verify_nesting(lad)
    # one mixed cell: n/a
    assert cpi.verify_nesting(built.pools_dir) == (cpi.NESTING_NA, None)


def test_stage_checks_reruns_steps_4_to_6_over_existing_dirs(built, tmp_path):
    out = tmp_path / "out"
    shutil.copytree(built.out_root, out)
    shutil.rmtree(out / "vol")
    shutil.rmtree(out / "reports")
    rep = bsp.stage_checks(SEED, frac=0.085, generator=GEN_POOLS, out_root=out,
                           advice_final=built.ap, stories_final=built.sp,
                           control_advice=built.cap, control_stories=built.csp,
                           probes=[built.pp], data_dir=built.data,
                           log=lambda *_a, **_k: None)
    assert rep["integrity"]["ok"] and rep["probe_pins"]["ok"]
    assert rep["nesting"] == cpi.NESTING_NA and rep["nesting_details"] is None
    assert rep["volume_files"].keys() == built.pools["volume_files"].keys()
    for name, rec in rep["volume_files"].items():
        assert rec["sha256"] == built.pools["volume_files"][name]["sha256"]
    assert rep["cells"]["constv2-mix-full"]["max_encoded_row_len"]  # preflighted manifest
    # a pools dir that never finished its preflight refuses
    (out / f"pools_s{SEED}" / "preflight_report.json").unlink()
    shutil.rmtree(out / "vol")
    with pytest.raises(bsp.SeedPoolsError, match="preflight_report"):
        bsp.stage_checks(SEED, frac=0.085, generator=GEN_POOLS, out_root=out,
                         advice_final=built.ap, stories_final=built.sp,
                         control_advice=built.cap, control_stories=built.csp,
                         probes=[built.pp], data_dir=built.data,
                         log=lambda *_a, **_k: None)
    # seed 1 without its ladder refuses
    with pytest.raises(bsp.SeedPoolsError, match="lacks"):
        one = tmp_path / "one"
        shutil.copytree(built.out_root, one)
        shutil.rmtree(one / "vol")
        (one / "pools_s1").symlink_to(one / f"pools_s{SEED}")
        (one / "pools_ctl2_s1").symlink_to(one / f"pools_ctl2_s{SEED}")
        d1 = tmp_path / "data1"
        shutil.copytree(built.data, d1)
        for n in ("heldout_split", "ctl2_heldout_split"):
            shutil.copy(d1 / f"{n}.s{SEED}.json", d1 / f"{n}.s1.json")
        bsp.stage_checks(1, frac=0.085, generator=GEN_POOLS, out_root=one,
                         advice_final=built.ap, stories_final=built.sp,
                         control_advice=built.cap, control_stories=built.csp,
                         probes=[built.pp], data_dir=d1, log=lambda *_a, **_k: None)


def test_preflight_tokenizers_share_duplicate_checkpoints_and_memoize(monkeypatch):
    """The driver's preflight set is bp.load_target_tokenizers() with the
    duplicate (repo, revision) pins sharing one tokenizer + one count memo:
    same keys, same numbers as the standard preflight, one tokenization per
    distinct checkpoint per text."""
    made = []

    class FakeInner:
        def __init__(self, key, repo, rev):
            self.model_key, self.repo, self.revision = key, repo, rev
            self.calls = 0
            made.append(self)

        def n_tokens(self, text, add_special_tokens):
            self.calls += 1
            return len(text.split()) + (1 if add_special_tokens else 0)

    monkeypatch.setattr(bp, "HFTargetTokenizer", FakeInner)
    monkeypatch.setattr(bp, "TARGET_TOKENIZERS", {
        "a_base": ("org/A", "1" * 40), "a_full": ("org/A", "1" * 40),
        "a_other_rev": ("org/A", "2" * 40), "b": ("org/B", "3" * 40)})
    toks = bsp.preflight_tokenizers()
    assert list(toks) == ["a_base", "a_full", "a_other_rev", "b"]
    assert len(made) == 3                                  # distinct pins
    assert toks["a_base"].inner is toks["a_full"].inner
    assert toks["a_base"].inner is not toks["a_other_rev"].inner
    for k, (repo, rev) in bp.TARGET_TOKENIZERS.items():   # provenance fields kept
        assert (toks[k].repo, toks[k].revision, toks[k].model_key) == (repo, rev, k)
    assert toks["a_base"].n_tokens("x y z", True) == 4
    assert toks["a_full"].n_tokens("x y z", True) == 4     # memo hit, no call
    assert toks["a_base"].inner.calls == 1
    assert toks["a_full"].n_tokens("x y z", False) == 3    # a different key
    assert toks["a_base"].inner.calls == 2
    assert toks["a_other_rev"].n_tokens("x y z", True) == 4
    assert toks["a_other_rev"].inner.calls == 1
    bsp._clear_memos(toks)
    toks["a_full"].n_tokens("x y z", True)
    assert toks["a_base"].inner.calls == 3                 # memo was cleared
    # the standard run_preflight consumes these like any tokenizer
    assert all(hasattr(t, "n_tokens") and hasattr(t, "repo") for t in toks.values())


def test_prefill_memos_uses_the_batch_encoder_once_per_checkpoint(tmp_path):
    pools = tmp_path / "pools"
    pools.mkdir()
    rows = [{"prompt": f"p{i} x", "pos": f"pos {i} y z", "doc_id": f"d{i}"} for i in range(5)]
    (pools / "pairs_a.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (pools / "pairs_b.jsonl").write_text(json.dumps(rows[0]) + "\n")  # a repeat
    for cell in ("a", "b"):  # run_preflight folds its numbers into manifests
        (pools / f"pairs_{cell}.manifest.json").write_text(json.dumps(
            {"cell": cell, "loss_token_counts": {}, "max_encoded_row_len": {}}))

    class Inner:
        def __init__(self, repo):
            self.repo, self.revision, self.batches, self.singles = repo, "r", [], 0

        def _tok(self, texts, add_special_tokens):
            self.batches.append((len(texts), add_special_tokens))
            return {"input_ids": [[0] * (len(t.split()) + (1 if add_special_tokens else 0))
                                  for t in texts]}

        def n_tokens(self, text, add_special_tokens):
            self.singles += 1
            return len(text.split()) + (1 if add_special_tokens else 0)

    a, b = Inner("org/A"), Inner("org/B")
    memo_a, memo_b = {}, {}
    toks = {"a1": bsp._MemoTok("a1", a, memo_a), "a2": bsp._MemoTok("a2", a, memo_a),
            "b": bsp._MemoTok("b", b, memo_b)}
    done = bsp._prefill_memos(pools, toks, chunk=3)
    assert done == {("org/A", "r"): 10, ("org/B", "r"): 10}   # 5 distinct prompts + 5 pos
    assert sum(n for n, _ in a.batches) == 10 and all(n <= 3 for n, _ in a.batches)
    assert len(memo_a) == 10 and memo_a[("pos 3 y z", False)] == 4 and memo_a[("p3 x", True)] == 3
    # the standard preflight now runs from memo hits: no per-text calls
    report = bp.run_preflight(pools, max_len=100, toks=toks)
    assert a.singles == 0 and b.singles == 0
    assert report["pools"]["pairs_a.jsonl"]["models"]["a1"]["max_encoded_row_len"] == 3 + 4 + 1
    assert report["pools"]["pairs_a.jsonl"]["models"]["a2"] == \
        report["pools"]["pairs_a.jsonl"]["models"]["a1"]
    # fakes without a batch encoder are skipped, not broken
    assert bsp._prefill_memos(pools, TOKS) == {}
