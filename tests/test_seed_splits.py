"""Seed-replicate held-out splits.

Pins, on synthetic corpora (fast) and on the real finals (read-only, skipped
where absent):

- ``carve_family(..., pinned_groups=)``: pinned groups enter held-out FIRST
  (sha order under the salt), then the salted walk continues to the stop
  rule; pins that already meet the threshold draw nothing; an unknown pin
  refuses; no pins = the historical walk.
- ``build_split(pin_probe_files=)`` emits a ``pins`` block ONLY when pinned;
  every probe ``heldout_doc`` source id lands in held-out with its whole
  group; the CLI threads ``--salt / --eval-salt / --pin-groups-from`` and the
  sidecar carries the carve summary.
- the committed ``heldout_split.s{1,2,3}.json`` equal a fresh build; all 226
  probe sources ∈ each seed's held-out, pinned groups whole;
  ``frac_realized ∈ [0.085, 0.105)``; the three held-out sets differ
  pairwise and each ⊇ seed-0's; the 128-doc eval subsets differ pairwise;
  the DEFAULT-args CLI build reproduces ``heldout_split.json`` byte for byte;
  the committed ``ctl2_heldout_split.s<s>.json`` equal a fresh control carve
  under the seed salts.
- ``check_probe_pins`` refuses a planted leak, a source outside held-out, a
  fraction miss and an unpinned split; ``build_seed_pools`` refuses a
  constitution ``--frac`` other than 0.085 without ``--frac-reason``.
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "scripts" / "sdfbase", ROOT / "scripts" / "sdfbase" / "seedrep",
           ROOT / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import build_heldout_split as hs  # noqa: E402
import build_pools as bp  # noqa: E402
import build_seed_pools as bsp  # noqa: E402
import check_probe_pins as cpp  # noqa: E402
from test_sdfbase_split import adv_rec, sto_rec, write_jsonl  # noqa: E402

DATA = ROOT / "data" / "sdfbase"
PROBES = [DATA / "probe_v2.jsonl", DATA / "probe_v3_genre.jsonl"]
SEEDS = (1, 2, 3)
LANE_FRAC = 0.085
WINDOW = 0.02
REAL_FILES = hs.ADVICE_FINAL.exists() and hs.STORIES_FINAL.exists()
CTL_FILES = (bsp.DEFAULT_CONTROL_ADVICE.exists()
             and bsp.DEFAULT_CONTROL_STORIES.exists())
needs_real = pytest.mark.skipif(not REAL_FILES,
                                reason="real corpus finals not present")
needs_ctl = pytest.mark.skipif(not CTL_FILES,
                               reason="r2 control finals not present")


def _dump(split: dict) -> bytes:
    return (json.dumps(split, indent=1, ensure_ascii=False) + "\n").encode("utf-8")


def _held(split: dict) -> set[str]:
    return set(split["advice"]["ids"]) | set(split["stories"]["ids"])


def probe_item(source_id: str, source: str = "heldout_doc") -> dict:
    return {"id": f"hd-{source_id}", "source": source, "source_id": source_id,
            "prefix": "invented prefix", "cont_consistent": "a",
            "cont_violating": "b", "section_id": "s"}


# ---------------------------------------------------------------------------
# carve_family with pins (synthetic)
# ---------------------------------------------------------------------------

def _uniform_advice(n_groups: int = 10):
    # n uniform groups of one 10-token record each
    return [adv_rec(p, 0, 0, counts=(10, 10, 10)) for p in range(n_groups)]


def _ranked(keys, salt=hs.SALT_HELDOUT):
    return sorted(sorted(keys), key=lambda k: (hs.sha_rank(salt, k), k))


def test_pinned_groups_enter_first_then_the_walk_continues():
    advice = _uniform_advice(10)
    keys = [hs.advice_group_key(r) for r in advice]
    pinned = [keys[3], keys[8]]
    # family official 100; frac 0.45 -> threshold 45; pins give 20, the walk
    # must add exactly 3 more groups (50 >= 45), in sha order of the rest.
    res = hs.carve_family(advice, hs.advice_group_key, hs.SALT_HELDOUT, 0.45,
                          pinned_groups=pinned)
    assert res["groups"][:2] == _ranked(pinned)
    assert res["pinned_groups"] == _ranked(pinned)
    rest = _ranked(k for k in keys if k not in pinned)
    assert res["seed_drawn_groups"] == rest[:3]
    assert res["groups"] == _ranked(pinned) + rest[:3]
    acc = res["accounting"]
    assert acc["heldout_official"] == 50 and acc["heldout_groups"] == 5
    pa = res["pin_accounting"]
    assert pa["n_pinned_groups"] == 2 and pa["n_seed_drawn_groups"] == 3
    assert pa["pinned_official"] == 20
    assert pa["pinned_frac_of_family"] == pytest.approx(0.2)
    assert pa["seed_drawn_share_of_heldout"] == pytest.approx(0.6)
    assert set(pa["pinned_ids"]) == {
        hs.doc_id(r) for r in advice if hs.advice_group_key(r) in pinned}
    # every pinned doc is held out
    assert set(pa["pinned_ids"]) <= set(res["ids"])


def test_pins_meeting_the_threshold_draw_nothing():
    advice = _uniform_advice(10)
    keys = [hs.advice_group_key(r) for r in advice]
    pinned = keys[:5]  # 50 tokens >= 0.45 * 100
    res = hs.carve_family(advice, hs.advice_group_key, hs.SALT_HELDOUT, 0.45,
                          pinned_groups=pinned)
    assert res["seed_drawn_groups"] == []
    assert res["groups"] == _ranked(pinned)
    assert res["pin_accounting"]["seed_drawn_share_of_heldout"] == 0.0


def test_unknown_pin_refuses():
    advice = _uniform_advice(4)
    with pytest.raises(hs.SplitError, match="not in this family"):
        hs.carve_family(advice, hs.advice_group_key, hs.SALT_HELDOUT, 0.3,
                        pinned_groups=["zz99|zz99-t00"])


def test_no_pins_is_the_historical_walk():
    advice = _uniform_advice(10)
    plain = hs.carve_family(advice, hs.advice_group_key, hs.SALT_HELDOUT, 0.25)
    empty = hs.carve_family(advice, hs.advice_group_key, hs.SALT_HELDOUT, 0.25,
                            pinned_groups=())
    assert plain == empty
    assert plain["pinned_groups"] == [] and plain["seed_drawn_groups"] == plain["groups"]
    assert plain["groups"] == _ranked(hs.advice_group_key(r) for r in advice)[:3]


def test_salt_changes_the_walk_but_not_the_pins():
    advice = _uniform_advice(30)
    keys = [hs.advice_group_key(r) for r in advice]
    pinned = [keys[0]]
    a = hs.carve_family(advice, hs.advice_group_key, "salt-A", 0.2, pinned_groups=pinned)
    b = hs.carve_family(advice, hs.advice_group_key, "salt-B", 0.2, pinned_groups=pinned)
    assert a["pinned_groups"] == b["pinned_groups"] == pinned
    assert a["seed_drawn_groups"] != b["seed_drawn_groups"]
    assert len(a["groups"]) == len(b["groups"]) == 6  # 60 >= 0.2 * 300


# ---------------------------------------------------------------------------
# build_split with probe pins; the probe-source helpers (synthetic)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    root = tmp_path_factory.mktemp("seedsplit")
    # 40 advice groups x 2 scenarios; 8 categories x 5 subcats x 2 calls x 2
    advice = [adv_rec(p, 0, s) for p in range(40) for s in range(2)]
    stories = [sto_rec(c, u, k, y) for c in range(8) for u in range(5)
               for k in range(2) for y in range(2)]
    ap = write_jsonl(root / "advice.jsonl", advice)
    sp = write_jsonl(root / "stories.jsonl", stories)
    # probe: one advice group (both scenarios) + two story subcats (one call each)
    probe = [probe_item(hs.doc_id(r)) for r in advice if r["principle_id"] == "p11"]
    probe += [probe_item(hs.doc_id(r)) for r in stories
              if r["subcategory_id"] in ("c02-u03", "c06-u00")
              and r["call_id"].endswith("k01")]
    probe.append(probe_item("sec-1", source="constitution_section"))
    pp = write_jsonl(root / "probe.jsonl", probe)
    return {"root": root, "advice": advice, "stories": stories, "ap": ap,
            "sp": sp, "pp": pp, "probe": probe}


def test_load_probe_source_ids_filters_dedupes_and_refuses(synth, tmp_path):
    ids = hs.load_probe_source_ids([synth["pp"]])
    assert ids == sorted({p["source_id"] for p in synth["probe"]
                          if p["source"] == "heldout_doc"})
    assert "sec-1" not in ids
    twice = hs.load_probe_source_ids([synth["pp"], synth["pp"]])
    assert twice == ids  # de-duplicated across files
    only_sections = write_jsonl(tmp_path / "sec.jsonl",
                                [probe_item("sec-1", "constitution_section")])
    with pytest.raises(hs.SplitError, match="nothing to pin"):
        hs.load_probe_source_ids([only_sections])
    no_sid = write_jsonl(tmp_path / "nosid.jsonl",
                         [{"id": "x", "source": "heldout_doc"}])
    with pytest.raises(hs.SplitError, match="no source_id"):
        hs.load_probe_source_ids([no_sid])
    with pytest.raises(hs.SplitError, match="missing"):
        hs.load_probe_source_ids([tmp_path / "absent.jsonl"])


def test_resolve_pinned_groups_maps_and_refuses_unknown(synth):
    ids = hs.load_probe_source_ids([synth["pp"]])
    groups = hs.resolve_pinned_groups(ids, synth["advice"], synth["stories"])
    assert groups == {"advice": ["p11|p11-t00"], "stories": ["c02-u03", "c06-u00"]}
    with pytest.raises(hs.SplitError, match="not corpus doc ids"):
        hs.resolve_pinned_groups(ids + ["adv-not-a-doc"], synth["advice"],
                                 synth["stories"])


def test_build_split_pins_block_only_when_pinned(synth):
    pinned = hs.build_split(synth["ap"], synth["sp"], frac=0.2, eval_n=4,
                            salt="s-A", eval_salt="e-A",
                            pin_probe_files=[synth["pp"]])
    plain = hs.build_split(synth["ap"], synth["sp"], frac=0.2, eval_n=4,
                           salt="s-A", eval_salt="e-A")
    none = hs.build_split(synth["ap"], synth["sp"], frac=0.2, eval_n=4,
                          salt="s-A", eval_salt="e-A", pin_probe_files=None)
    assert "pins" not in plain and plain == none
    pins = pinned["pins"]
    src = set(pins["pinned_source_ids"])
    assert src == set(hs.load_probe_source_ids([synth["pp"]]))
    assert pins["n_pinned_source_ids"] == len(src)
    assert pins["probe_files"] == [{"name": "probe.jsonl",
                                    "sha256": hs.sha256_file(synth["pp"])}]
    held = _held(pinned)
    assert src <= held
    # whole groups: every doc of a pinned group is held out
    for r in synth["advice"]:
        if hs.advice_group_key(r) in pins["advice"]["pinned_groups"]:
            assert hs.doc_id(r) in held
    for r in synth["stories"]:
        if r["subcategory_id"] in pins["stories"]["pinned_groups"]:
            assert hs.doc_id(r) in held
    assert pinned["advice"]["groups"][:1] == pins["advice"]["pinned_groups"]
    assert pinned["stories"]["groups"][:2] == pins["stories"]["pinned_groups"]
    assert pinned["salt"] == "s-A" and pinned["eval_subset"]["salt"] == "e-A"
    # the artifact stays generator-free
    assert "generator" not in pinned
    # determinism
    again = hs.build_split(synth["ap"], synth["sp"], frac=0.2, eval_n=4,
                           salt="s-A", eval_salt="e-A",
                           pin_probe_files=[synth["pp"]])
    assert _dump(again) == _dump(pinned)


def test_cli_threads_salts_and_pins_into_artifact_and_sidecar(synth, tmp_path,
                                                                monkeypatch):
    # build_split binds eval_n=EVAL_N at definition time: wrap it so the
    # tiny synthetic held-out set can seat a 4-doc eval subset
    orig_build = hs.build_split
    monkeypatch.setattr(hs, "build_split",
                        lambda *a, **k: orig_build(*a, **{**k, "eval_n": 4}))
    monkeypatch.setattr(hs, "generator_provenance",
                        lambda: {"sisteer_commit": "x", "sisteer_dirty": False,
                                 "sisteer_diff_sha256": None,
                                 "corpus_generator_commit": "y",
                                 "corpus_generator_dirty": False,
                                 "corpus_generator_diff_sha256": None})
    out = tmp_path / "heldout_split.s7.json"
    rc = hs.main(["--advice-final", str(synth["ap"]), "--stories-final",
                  str(synth["sp"]), "--out", str(out), "--frac", "0.2",
                  "--salt", "sdfbase-heldout-v1-s7",
                  "--eval-salt", "sdfbase-heldout-eval-v1-s7",
                  "--pin-groups-from", str(synth["pp"])])
    assert rc == 0
    art = json.loads(out.read_text())
    assert art["salt"] == "sdfbase-heldout-v1-s7"
    assert art["eval_subset"]["salt"] == "sdfbase-heldout-eval-v1-s7"
    assert art["pins"]["advice"]["pinned_groups"] == ["p11|p11-t00"]
    prov = json.loads(out.with_suffix(".provenance.json").read_text())
    assert prov["artifact_sha256"] == hs.sha256_file(out)
    carve = prov["carve"]
    assert carve["salt"] == "sdfbase-heldout-v1-s7"
    assert carve["eval_salt"] == "sdfbase-heldout-eval-v1-s7"
    assert carve["frac_requested"] == 0.2 and carve["pinned"] is True
    assert carve["pinned_groups"] == {"advice": 1, "stories": 2}
    assert carve["n_pinned_source_ids"] == art["pins"]["n_pinned_source_ids"]
    assert carve["probe_files"] == ["probe.jsonl"]
    assert carve["frac_realized"] == {
        f: art["token_totals"][f]["heldout_frac_realized"] for f in ("advice", "stories")}
    assert carve["train_side_official_tokens"] == {
        f: art["token_totals"][f]["train_official"] for f in ("advice", "stories")}
    assert prov["generator"]["sisteer_commit"] == "x"


def test_split_provenance_tolerates_a_minimal_artifact(tmp_path):
    art = tmp_path / "a.json"
    art.write_text("{}\n")
    prov = hs.split_provenance({"token_totals": {"advice": {}}}, artifact=art,
                               generator={}, salt="S", eval_salt="E", frac=0.1)
    assert prov["carve"]["salt"] == "S" and prov["carve"]["eval_salt"] == "E"
    assert prov["carve"]["frac_requested"] == 0.1
    assert prov["carve"]["frac_realized"] == {"advice": None, "stories": None}
    assert prov["carve"]["pinned"] is False


# ---------------------------------------------------------------------------
# check_probe_pins (synthetic)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def pinned_split(synth):
    split = hs.build_split(synth["ap"], synth["sp"], frac=0.2, eval_n=4,
                           pin_probe_files=[synth["pp"]])
    path = synth["root"] / "heldout_split.pinned.json"
    path.write_bytes(_dump(split))
    plain = hs.build_split(synth["ap"], synth["sp"], frac=0.2, eval_n=4)
    ppath = synth["root"] / "heldout_split.plain.json"
    ppath.write_bytes(_dump(plain))
    return {"split": split, "path": path, "plain": plain, "plain_path": ppath}


def _fake_pool(path: Path, doc_ids) -> Path:
    path.write_text("".join(json.dumps({"doc_id": d, "instruction_id": "x",
                                        "prompt": "", "pos": "t"}) + "\n"
                            for d in doc_ids), encoding="utf-8")
    return path


def test_check_probe_pins_passes_a_clean_split_and_train_pool(synth, pinned_split, tmp_path):
    sp = pinned_split["split"]
    held = _held(sp)
    train = [hs.doc_id(r) for r in synth["advice"] + synth["stories"]
             if hs.doc_id(r) not in held]
    pool = _fake_pool(tmp_path / "pairs_constv2-mix-full.jsonl", train)
    rep = cpp.check(pinned_split["path"], [synth["pp"]], [pool], frac=0.2)
    assert rep["ok"], rep["findings"]
    assert rep["n_probe_sources_in_heldout"] == rep["n_probe_source_ids"]
    assert rep["pins"]["advice"]["pinned_groups"] == 1
    assert rep["pins"]["stories"]["pinned_groups"] == 2
    assert rep["pins_agree_with_probes"] is True
    assert cpp.main(["--split", str(pinned_split["path"]), "--probes",
                     str(synth["pp"]), "--train-pools", str(pool),
                     "--frac", "0.2", "--report", str(tmp_path / "r.json")]) == 0
    assert json.loads((tmp_path / "r.json").read_text())["ok"] is True


def test_check_probe_pins_refuses_a_planted_leak(synth, pinned_split, tmp_path):
    leaked = pinned_split["split"]["pins"]["pinned_source_ids"][0]
    pool = _fake_pool(tmp_path / "pairs_constv2-mix-full.jsonl", ["adv-p00-t00-s00", leaked])
    rep = cpp.check(pinned_split["path"], [synth["pp"]], [pool], frac=0.2)
    assert rep["ok"] is False
    assert any("inside training pool" in f for f in rep["findings"])
    assert list(rep["probe_leaks"].values()) == [[leaked]]
    assert cpp.main(["--split", str(pinned_split["path"]), "--probes",
                     str(synth["pp"]), "--train-pools", str(pool),
                     "--frac", "0.2"]) == 2


def test_check_probe_pins_refuses_a_heldout_doc_in_training_even_if_not_a_probe(
        synth, pinned_split, tmp_path):
    sp = pinned_split["split"]
    src = set(sp["pins"]["pinned_source_ids"])
    non_probe_held = next(i for i in _held(sp) if i not in src)
    pool = _fake_pool(tmp_path / "pool.jsonl", [non_probe_held])
    rep = cpp.check(pinned_split["path"], [synth["pp"]], [pool], frac=0.2)
    assert rep["ok"] is False and not rep["probe_leaks"]
    assert any("held-out doc(s) inside training pool" in f for f in rep["findings"])


def test_check_probe_pins_refuses_a_fraction_miss(synth, pinned_split):
    rep = cpp.check(pinned_split["path"], [synth["pp"]], [], frac=0.9)
    assert rep["ok"] is False
    assert all(not ok for ok in rep["frac_ok"].values())
    assert any("outside [0.9" in f for f in rep["findings"])


def test_check_probe_pins_refuses_a_source_outside_heldout_and_unpinned(
        synth, pinned_split, tmp_path):
    # a probe built from a TRAINING doc of this split
    sp = pinned_split["split"]
    held = _held(sp)
    train_doc = next(hs.doc_id(r) for r in synth["stories"] if hs.doc_id(r) not in held)
    other = write_jsonl(tmp_path / "other_probe.jsonl", [probe_item(train_doc)])
    rep = cpp.check(pinned_split["path"], [other], [], frac=0.2)
    assert rep["ok"] is False
    assert rep["probe_sources_outside_heldout"] == [train_doc]
    assert rep["pins_agree_with_probes"] is False
    # an unpinned split refuses unless allowed
    rep = cpp.check(pinned_split["plain_path"], [synth["pp"]], [], frac=0.2)
    assert rep["ok"] is False and any("no 'pins' block" in f for f in rep["findings"])
    rep = cpp.check(pinned_split["plain_path"], [synth["pp"]], [], frac=0.2,
                    allow_unpinned=True)
    # the plain carve happens not to hold every probe source out (that is the
    # point of pinning) — only the pins finding is relaxed
    assert not any("no 'pins' block" in f for f in rep["findings"])


# ---------------------------------------------------------------------------
# build_seed_pools: lane constants + refusals
# ---------------------------------------------------------------------------

def test_lane_constants():
    assert bsp.LANE_FRAC == 0.085 and bsp.CTL_FRAC == hs.FRAC == 0.05
    assert bsp.DRAW_SALT == bp.SALT_MIX == "sdfbase-draw-v1"
    assert bsp.salts_of(2) == ("sdfbase-heldout-v1-s2", "sdfbase-heldout-eval-v1-s2")
    assert bsp.key_of(3) == "qwen35_4b_base_eb64_s3"
    assert set(bsp.cells_of(1)) == {"constv2-mix-full", "constv2-heldout",
                                    "constv2-mix-10m", "constv2-mix-1m",
                                    "constv2-mix-1k"}
    assert set(bsp.cells_of(2)) == set(bsp.cells_of(3)) == {"constv2-mix-full",
                                                            "constv2-heldout"}
    assert "constv2ctl2-full" in bsp.volume_cells_of(2)
    assert "constv2ctl2-heldout" not in bsp.volume_cells_of(2)


def test_build_seed_pools_refuses_another_frac_without_a_reason(tmp_path):
    rc = bsp.main(["--seed", "1", "--frac", "0.05", "--out-root", str(tmp_path / "o")])
    assert rc == 2
    assert not (tmp_path / "o").exists()
    bsp.refuse_frac(0.085, None)            # the lane value needs no reason
    bsp.refuse_frac(0.05, "spec-literal rerun (D1 option a)")
    with pytest.raises(bsp.SeedPoolsError, match="frac-reason"):
        bsp.refuse_frac(0.05, None)
    with pytest.raises(bsp.SeedPoolsError, match="frac-reason"):
        bsp.refuse_frac(0.05, "")
    assert bsp.main(["--out-root", str(tmp_path / "o")]) == 2  # --seed required


def test_stage_pools_requires_committed_splits(tmp_path):
    with pytest.raises(bsp.SeedPoolsError, match="run --stage splits"):
        bsp.stage_pools(2, frac=0.085, generator={}, out_root=tmp_path / "o",
                        advice_final=Path("a"), stories_final=Path("s"),
                        control_advice=Path("c"), control_stories=Path("d"),
                        probes=[], register_edit=None, data_dir=tmp_path)


# ---------------------------------------------------------------------------
# Real corpus (read-only on the finals; the committed seed artifacts)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def real_seed_splits():
    if not REAL_FILES:
        pytest.skip("real corpus finals not present")
    fresh = {s: hs.build_split(hs.ADVICE_FINAL, hs.STORIES_FINAL, frac=LANE_FRAC,
                               salt=f"{hs.SALT_HELDOUT}-s{s}",
                               eval_salt=f"{hs.SALT_EVAL}-s{s}",
                               pin_probe_files=PROBES)
             for s in SEEDS}
    committed = {s: json.loads((DATA / f"heldout_split.s{s}.json").read_text())
                 for s in SEEDS}
    seed0 = json.loads((DATA / "heldout_split.json").read_text())
    return {"fresh": fresh, "committed": committed, "seed0": seed0}


@needs_real
def test_committed_seed_splits_equal_a_fresh_build(real_seed_splits):
    for s in SEEDS:
        fresh = real_seed_splits["fresh"][s]
        assert real_seed_splits["committed"][s] == fresh, f"seed {s} drifted"
        assert (DATA / f"heldout_split.s{s}.json").read_bytes() == _dump(fresh)
        assert fresh["salt"] == f"sdfbase-heldout-v1-s{s}"
        assert fresh["eval_subset"]["salt"] == f"sdfbase-heldout-eval-v1-s{s}"
        assert fresh["frac"] == LANE_FRAC
        prov = json.loads((DATA / f"heldout_split.s{s}.provenance.json").read_text())
        assert prov["artifact_sha256"] == hs.sha256_file(DATA / f"heldout_split.s{s}.json")
        assert prov["carve"]["seed"] == s and prov["carve"]["frac_override_reason"] is None


@needs_real
def test_every_probe_source_is_held_out_with_its_whole_group(real_seed_splits):
    src = set(hs.load_probe_source_ids(PROBES))
    assert len(src) == 226
    advice = hs.load_jsonl(hs.ADVICE_FINAL)
    stories = hs.load_jsonl(hs.STORIES_FINAL)
    by_group: dict[str, set[str]] = {}
    for r in advice:
        by_group.setdefault(hs.advice_group_key(r), set()).add(hs.doc_id(r))
    for r in stories:
        by_group.setdefault(hs.story_group_key(r), set()).add(hs.doc_id(r))
    for s in SEEDS:
        sp = real_seed_splits["committed"][s]
        held = _held(sp)
        assert src <= held, f"seed {s}: probe sources outside held-out"
        pins = sp["pins"]
        assert set(pins["pinned_source_ids"]) == src
        assert pins["n_pinned_source_ids"] == 226
        assert len(pins["advice"]["pinned_groups"]) == 17
        assert len(pins["stories"]["pinned_groups"]) == 25
        for g in pins["advice"]["pinned_groups"] + pins["stories"]["pinned_groups"]:
            assert by_group[g] <= held, f"seed {s}: pinned group {g} straddles"
        # and every held-out group is whole (the carve granularity)
        for g in sp["advice"]["groups"] + sp["stories"]["groups"]:
            assert by_group[g] <= held
        assert pins["probe_files"] == [{"name": p.name, "sha256": hs.sha256_file(p)}
                                       for p in PROBES]


@needs_real
def test_frac_realized_in_the_lane_window(real_seed_splits):
    for s in SEEDS:
        sp = real_seed_splits["committed"][s]
        for fam in ("advice", "stories"):
            v = sp["token_totals"][fam]["heldout_frac_realized"]
            assert LANE_FRAC <= v < LANE_FRAC + WINDOW, (s, fam, v)
            # O2: roughly 40 % of the held-out mass is seed-drawn
            share = sp["pins"][fam]["seed_drawn_share_of_heldout"]
            assert 0.3 < share < 0.5, (s, fam, share)


@needs_real
def test_seed_heldout_sets_differ_pairwise_and_contain_seed0(real_seed_splits):
    held = {s: _held(real_seed_splits["committed"][s]) for s in SEEDS}
    held0 = _held(real_seed_splits["seed0"])
    for s in SEEDS:
        assert held0 < held[s], f"seed {s} does not strictly contain seed 0"
    for a, b in itertools.combinations(SEEDS, 2):
        assert held[a] != held[b], f"seeds {a} and {b} carve the same held-out set"
        assert held[a] - held[b] and held[b] - held[a]


@needs_real
def test_eval_subsets_differ_pairwise(real_seed_splits):
    evs = {s: real_seed_splits["committed"][s]["eval_subset"] for s in SEEDS}
    for s in SEEDS:
        assert evs[s]["n"] == 128 and len(evs[s]["ids"]) == 128
        assert set(evs[s]["ids"]) <= _held(real_seed_splits["committed"][s])
    for a, b in itertools.combinations(SEEDS, 2):
        assert evs[a]["ids"] != evs[b]["ids"]
        assert set(evs[a]["ids"]) != set(evs[b]["ids"])
    assert all(set(evs[s]["ids"]) != set(real_seed_splits["seed0"]["eval_subset"]["ids"])
               for s in SEEDS)


@needs_real
def test_default_args_reproduce_seed0_split_byte_for_byte(tmp_path):
    out = tmp_path / "heldout_split.json"
    # --allow-dirty only affects the SIDECAR; the artifact is provenance-free
    assert hs.main(["--out", str(out), "--allow-dirty"]) == 0
    assert out.read_bytes() == (DATA / "heldout_split.json").read_bytes()
    assert "pins" not in json.loads(out.read_text())


@needs_real
def test_check_probe_pins_on_the_committed_seed_splits(tmp_path):
    for s in SEEDS:
        rep = cpp.check(DATA / f"heldout_split.s{s}.json", PROBES, [], frac=LANE_FRAC)
        assert rep["ok"], (s, rep["findings"])
        assert rep["pins"]["advice"]["pinned_groups"] == 17
        assert rep["pins"]["stories"]["pinned_groups"] == 25
    # a planted leak and a fraction miss refuse on the real split too
    src = hs.load_probe_source_ids(PROBES)
    pool = _fake_pool(tmp_path / "pairs_constv2-mix-full.jsonl", [src[0]])
    assert cpp.main(["--split", str(DATA / "heldout_split.s1.json"), "--probes",
                     *map(str, PROBES), "--train-pools", str(pool),
                     "--frac", str(LANE_FRAC)]) == 2
    assert cpp.main(["--split", str(DATA / "heldout_split.s1.json"), "--probes",
                     *map(str, PROBES), "--frac", "0.05"]) == 2
    # seed 0's unpinned split reads only with --allow-unpinned
    assert cpp.main(["--split", str(DATA / "heldout_split.json"), "--probes",
                     *map(str, PROBES), "--frac", "0.05"]) == 2
    assert cpp.main(["--split", str(DATA / "heldout_split.json"), "--probes",
                     *map(str, PROBES), "--frac", "0.05", "--allow-unpinned"]) == 0


@needs_ctl
def test_committed_ctl2_seed_splits_equal_a_fresh_control_carve():
    fresh = {s: hs.build_split(bsp.DEFAULT_CONTROL_ADVICE, bsp.DEFAULT_CONTROL_STORIES,
                               frac=bsp.CTL_FRAC, salt=f"{hs.SALT_HELDOUT}-s{s}",
                               eval_salt=f"{hs.SALT_EVAL}-s{s}")
             for s in SEEDS}
    for s in SEEDS:
        path = DATA / f"ctl2_heldout_split.s{s}.json"
        assert path.read_bytes() == _dump(fresh[s]), f"ctl2 seed {s} drifted"
        assert fresh[s]["frac"] == 0.05 and "pins" not in fresh[s]
        prov = json.loads(path.with_suffix(".provenance.json").read_text())
        assert prov["release_bytes_pinned"] is True
        assert prov["carve"]["salt"] == f"sdfbase-heldout-v1-s{s}"
    held = {s: _held(fresh[s]) for s in SEEDS}
    for a, b in itertools.combinations(SEEDS, 2):
        assert held[a] != held[b]
