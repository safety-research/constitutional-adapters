"""Tests for scripts/sdfbase/build_pools.py.

Pins: sampler determinism (same seed => identical pools+manifests); nesting
= doc-id prefix containment; ratio within the ±2% gate at >=100k; budget
asserted by calling the CORPUS REPO's official_tokens on each selected set
(a crafted subset where the limiting tokenizer flips must pass, and the
sum-of-record-mins estimator must disagree); truncation via a deterministic
FORWARD token-boundary scan (a planted non-monotonic merge case is found
past an overshooting candidate; no-exact-cutoff fails loudly with
bracketing candidates); the max-length preflight refuses a planted
over-length row under either target tokenizer; prompt+pos byte-parity with
the corpus renderers on >=50 real docs/family; zero chat markup;
Claude/Anthropic word-boundary canary on rendered text (allowlist
mechanism; metadata is never concatenated); the boundary skip-forward
rule (planted prefix-eater at the boundary => skipped + recorded in
``boundary_skips`` + budget exact; adv-1k stays a prefix of adv-100k under
real skips; the impossible-everywhere case still refuses loudly).

Mechanics tests use tiny synthetic corpora + an injected fake counting rig
(CPU, fast). The renderer-parity check reads the real v2 finals read-only.
"""

from __future__ import annotations

import itertools
import json
import re
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "sdfbase"))

import build_heldout_split as hs  # noqa: E402
import build_pools as bp  # noqa: E402

#: The corpus repo's renderers are imported BY PATH from GENERATOR_ROOT/src
#: (hs.generator_tokens — grading-host-only prep), so even the
#: synthetic builds below need that checkout. Where it is absent (a cluster
#: node: 26 setup errors + failures) the tests SKIP with the same override
#: hint generator_tokens raises, like the "real corpus finals not present"
#: skips further down — never a silent pass, never an error that reads as
#: a regression.
CORPUSGEN_SRC_PRESENT = (hs.GENERATOR_ROOT / "src").is_dir()
needs_generator = pytest.mark.skipif(
    not CORPUSGEN_SRC_PRESENT,
    reason=f"corpus_generator repo not present at {hs.GENERATOR_ROOT} (its "
           "renderers are imported by path; set CORPUS_GENERATOR_ROOT)")


# ---------------------------------------------------------------------------
# Fake counting rig — words are tokens; per-prefix overrides let tests plant
# non-monotonic merges and per-tokenizer flips.
# ---------------------------------------------------------------------------

class FakeRig:
    provenance = {"fake": True}

    def __init__(self, overrides: dict | None = None):
        self.overrides = overrides or {}

    def counts(self, text: str) -> dict[str, int]:
        if text in self.overrides:
            v = self.overrides[text]
            return {k: v for k in hs.TOK_NAMES} if isinstance(v, int) else dict(v)
        n = len(text.split())
        return {k: n for k in hs.TOK_NAMES}

    def boundary_offsets(self, name: str, text: str) -> list[int]:
        return [m.end() for m in re.finditer(r"\S+", text)]


class FakeTok:
    """Preflight target-tokenizer fake: words + optional BOS on specials."""

    def __init__(self, model_key: str, bos: bool):
        self.model_key = model_key
        self.bos = bos
        self.repo = f"fake/{model_key}"
        self.revision = "0" * 40

    def n_tokens(self, text: str, add_special_tokens: bool) -> int:
        return len(text.split()) + (1 if (add_special_tokens and self.bos) else 0)


# ---------------------------------------------------------------------------
# Synthetic corpus (invented text, placeholder ids)
# ---------------------------------------------------------------------------

def words(prefix: str, n: int) -> str:
    return " ".join(f"{prefix}{i:04d}" for i in range(n))


def adv_rec(p: int, t: int, s: int, *, system_words=4, user_words=6,
            assistant_words=10, system=True):
    pid = f"p{p:02d}"
    tid = f"{pid}-t{t:02d}"
    sid = f"{tid}-s{s:02d}"
    sys_text = words(f"as{p}x{t}x{s}s", system_words) if system else None
    user = words(f"au{p}x{t}x{s}u", user_words)
    assistant = words(f"aa{p}x{t}x{s}a", assistant_words)
    render = "\n\n".join(x for x in (sys_text or "", user, assistant) if x)
    n = len(render.split())
    return {
        "system": sys_text,
        "user": user,
        "assistant": assistant,
        "principle_id": pid,
        "prompt_type_id": tid,
        "scenario_id": sid,
        "lineage": {"custom_id": f"adv-{sid}"},
        "token_counts": {k: n for k in hs.TOK_NAMES},
    }


def sto_rec(c: int, u: int, k: int, y: int, *, n_words=40):
    cid = f"c{c:02d}"
    uid = f"{cid}-u{u:02d}"
    kid = f"{uid}-k{k:02d}"
    text = words(f"st{c}x{u}x{k}x{y}w", n_words)
    return {
        "text": text,
        "category_id": cid,
        "subcategory_id": uid,
        "call_id": kid,
        "lineage": {"custom_id": f"sto-{kid}-y{y}"},
        "token_counts": {k2: n_words for k2 in hs.TOK_NAMES},
    }


def write_jsonl(path: Path, records: list[dict]) -> Path:
    path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records),
        encoding="utf-8",
    )
    return path


MIX_CELLS = [
    ("constv2-mix-1k", 1_000),
    ("constv2-mix-10k", 10_000),
    ("constv2-mix-100k", 100_000),
    # Round-2: the full-train-side cell — budget None, whole docs, no
    # truncation (the complete draw order).
    ("constv2-mix-full", None),
]
ADV_CELLS = [
    ("constv2-adv-1k", 1_000),
    ("constv2-adv-5k", 5_000),
]


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """One synthetic end-to-end build (plus a determinism twin)."""
    if not CORPUSGEN_SRC_PRESENT:
        pytest.skip(needs_generator.kwargs["reason"])
    root = tmp_path_factory.mktemp("sdfbase")
    # advice: 6 groups x 10 scenarios, ~200 words each (~12,000 fake tokens).
    # Sizes are deliberately de-latticed so synthetic cums don't land on
    # degenerate budget boundaries by construction.
    advice = [
        adv_rec(p, 0, s, system_words=20, user_words=60,
                assistant_words=120 + (7 * s + 3 * p) % 13,
                system=(s % 3 != 0))  # every third record has a null system
        for p in range(6) for s in range(10)
    ]
    # stories: 6 cats x 5 subcats x 5 calls x 2 = 300 recs x ~350 = ~105,000
    stories = [
        sto_rec(c, u, k, y, n_words=350 + (31 * c + 17 * u + 7 * k + 3 * y) % 23)
        for c in range(6) for u in range(5) for k in range(5) for y in range(2)
    ]
    ap = write_jsonl(root / "advice.jsonl", advice)
    sp = write_jsonl(root / "stories.jsonl", stories)
    split = hs.build_split(ap, sp, eval_n=8)
    split_path = root / "heldout_split.json"
    split_path.write_text(
        json.dumps(split, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    kwargs = dict(
        advice_final=ap, stories_final=sp,
        mix_cells=MIX_CELLS, adv_cells=ADV_CELLS, rig=FakeRig(),
    )
    out1 = root / "pools1"
    out2 = root / "pools2"
    m1 = bp.build_pools(out1, split_path, **kwargs)
    m2 = bp.build_pools(out2, split_path, **kwargs)
    return SimpleNamespace(
        root=root, advice=advice, stories=stories, split=split,
        split_path=split_path, out1=out1, out2=out2, m1=m1, m2=m2,
        advice_path=ap, stories_path=sp,
    )


# ---------------------------------------------------------------------------
# Determinism, nesting, ratio, budget
# ---------------------------------------------------------------------------

def test_sampler_determinism_bytewise(built):
    files1 = sorted(p.name for p in built.out1.iterdir())
    files2 = sorted(p.name for p in built.out2.iterdir())
    assert files1 == files2
    for name in files1:
        assert (built.out1 / name).read_bytes() == (built.out2 / name).read_bytes(), (
            f"{name} differs between two identically-seeded builds"
        )


def test_expected_cells_written(built):
    names = {p.name for p in built.out1.iterdir()}
    for cell, _ in MIX_CELLS + ADV_CELLS:
        assert f"pairs_{cell}.jsonl" in names
        assert f"pairs_{cell}.manifest.json" in names
    assert "pairs_constv2-heldout.jsonl" in names
    assert "pairs_constv2-heldout.manifest.json" in names
    assert "pools_summary.json" in names


def test_nesting_prefix_containment(built):
    for cells in (MIX_CELLS, ADV_CELLS):
        orders = [built.m1[cell]["doc_ids_in_order"] for cell, _ in cells]
        for small, big in itertools.pairwise(orders):
            assert big[: len(small)] == small, "nesting broken: not a prefix"


def test_ratio_gate_at_scale(built):
    m = built.m1["constv2-mix-100k"]
    assert m["ratio_gate"]["applied"] is True
    assert m["ratio_gate"]["deviation"] <= bp.RATIO_TOL
    assert abs(m["realized_ratio"] - m["natural_ratio"]) <= bp.RATIO_TOL
    # small cells record the realized ratio but are not gated
    assert built.m1["constv2-mix-1k"]["ratio_gate"]["applied"] is False
    assert built.m1["constv2-mix-1k"]["realized_ratio"] is not None


def test_budget_exact_under_corpus_official_tokens(built):
    """The AUTHORITATIVE assertion: reconstruct each selected set and hand it
    to the corpus repo's own official_tokens()."""
    tokens_mod = hs.generator_tokens()
    by_id = {
        r["lineage"]["custom_id"]: r for r in built.advice + built.stories
    }
    for cell, budget in MIX_CELLS + ADV_CELLS:
        m = built.m1[cell]
        records = []
        trunc = m["truncation"]
        for did in m["doc_ids_in_order"]:
            if trunc and did == trunc["doc_id"]:
                records.append(
                    {"token_counts": trunc["tokens_taken_per_tokenizer"]}
                )
            else:
                records.append(by_id[did])
        if budget is None:  # the full cell: official = whatever the draw is
            assert tokens_mod.official_tokens(records) == \
                m["official_tokens"], cell
        else:
            assert tokens_mod.official_tokens(records) == budget, cell
            assert m["official_tokens"] == budget


def test_heldout_eval_pool(built):
    ev = built.split["eval_subset"]
    m = built.m1[bp.HELDOUT_CELL]
    assert m["doc_ids_in_order"] == ev["ids"]
    assert m["family_counts"] == ev["family_counts"]
    assert m["budget"] is None
    assert m["truncation"] is None
    rows = [
        json.loads(line)
        for line in (built.out1 / "pairs_constv2-heldout.jsonl").read_text().splitlines()
    ]
    assert [r["doc_id"] for r in rows] == ev["ids"]
    assert all(not r["truncated"] for r in rows)


def test_row_shape_and_parity(built):
    tokens_mod = hs.generator_tokens()
    by_id = {r["lineage"]["custom_id"]: r for r in built.advice + built.stories}
    for cell, _ in MIX_CELLS:
        rows = [
            json.loads(line)
            for line in (built.out1 / f"pairs_{cell}.jsonl").read_text().splitlines()
        ]
        m = built.m1[cell]
        assert [r["doc_id"] for r in rows] == m["doc_ids_in_order"]
        for i, row in enumerate(rows):
            assert set(row) == {
                "prompt", "pos", "neg", "instruction_id", "doc_id", "family",
                "official_tokens", "truncated",
            }
            assert row["neg"] == ""
            assert row["instruction_id"] == cell
            assert row["pos"], "empty trained text"
            rec = by_id[row["doc_id"]]
            rendered = (
                tokens_mod.render_advice_text(rec)
                if row["family"] == "advice"
                else tokens_mod.render_story_text(rec)
            )
            if row["truncated"]:
                assert i == len(rows) - 1, "only the final doc may be truncated"
                assert row["prompt"] + row["pos"] == rendered[: len(row["prompt"] + row["pos"])]
                assert rendered.startswith(row["prompt"] + row["pos"])
            else:
                assert row["prompt"] + row["pos"] == rendered
            if row["family"] == "stories":
                assert row["prompt"] == ""
            elif rec.get("system"):
                assert row["prompt"] == rec["system"] + "\n\n"
            else:
                assert row["prompt"] == ""


def test_train_pools_exclude_heldout(built):
    held = set(built.split["advice"]["ids"]) | set(built.split["stories"]["ids"])
    for cell, _ in MIX_CELLS + ADV_CELLS:
        ids = set(built.m1[cell]["doc_ids_in_order"])
        assert not (ids & held), f"{cell} leaked held-out docs"


def test_clean_fixture_has_no_boundary_skips(built):
    for cell, m in built.m1.items():
        assert m["boundary_skips"] == [], f"{cell} unexpectedly skipped docs"


# ---------------------------------------------------------------------------
# Round-2: the full cell (budget None), --cells subset builds, summary merge,
# verify_mix_nesting
# ---------------------------------------------------------------------------


def test_full_cell_takes_entire_train_side_whole(built):
    """budget None = the COMPLETE draw order: every train doc, whole, no
    truncation, no skips; official = the train side's set total; the ratio
    gate is not applied (realized IS natural by construction)."""
    m = built.m1["constv2-mix-full"]
    held = set(built.split["advice"]["ids"]) | set(built.split["stories"]["ids"])
    train_ids = {
        r["lineage"]["custom_id"] for r in built.advice + built.stories
    } - held
    assert m["budget"] is None
    assert m["truncation"] is None
    assert m["boundary_skips"] == []
    assert set(m["doc_ids_in_order"]) == train_ids
    assert m["n_docs"] == len(train_ids)
    tokens_mod = hs.generator_tokens()
    by_id = {r["lineage"]["custom_id"]: r
             for r in built.advice + built.stories}
    assert m["official_tokens"] == tokens_mod.official_tokens(
        [by_id[d] for d in m["doc_ids_in_order"]])
    assert m["ratio_gate"]["applied"] is False
    assert m["realized_ratio"] == pytest.approx(m["natural_ratio"])
    rows = [json.loads(l) for l in
            (built.out1 / "pairs_constv2-mix-full.jsonl")
            .read_text().splitlines()]
    assert len(rows) == m["n_docs"]
    assert all(not r["truncated"] for r in rows)


def test_verify_mix_nesting_passes_and_names_order(built):
    names = bp.verify_mix_nesting(built.out1)
    assert names == ["constv2-mix-1k", "constv2-mix-10k",
                     "constv2-mix-100k", "constv2-mix-full"]


def test_verify_mix_nesting_catches_divergence(built, tmp_path):
    pools = tmp_path / "pools"
    shutil.copytree(built.out1, pools)
    mp = pools / "pairs_constv2-mix-10k.manifest.json"
    m = json.loads(mp.read_text())
    m["doc_ids_in_order"][0] = "sto-not-in-any-order"
    mp.write_text(json.dumps(m))
    with pytest.raises(bp.SamplerError, match="NOT a doc-id prefix"):
        bp.verify_mix_nesting(pools)


def test_verify_mix_nesting_names_a_shorter_bigger_cell(built, tmp_path):
    """The corner the divergence search used to miss: every shared
    position agrees but the LARGER-budget cell lists fewer docs than the
    smaller one (a truncated or mis-budgeted manifest). That raised a bare
    StopIteration; it is a named refusal now."""
    pools = tmp_path / "pools"
    shutil.copytree(built.out1, pools)
    small = json.loads((pools / "pairs_constv2-mix-1k.manifest.json")
                       .read_text())["doc_ids_in_order"]
    mp = pools / "pairs_constv2-mix-10k.manifest.json"
    m = json.loads(mp.read_text())
    m["doc_ids_in_order"] = small[:-1]
    mp.write_text(json.dumps(m))
    with pytest.raises(bp.SamplerError, match="shorter than the smaller"):
        bp.verify_mix_nesting(pools)


def test_verify_mix_nesting_refuses_skips_and_single_cell(built, tmp_path):
    pools = tmp_path / "pools"
    shutil.copytree(built.out1, pools)
    mp = pools / "pairs_constv2-mix-1k.manifest.json"
    m = json.loads(mp.read_text())
    m["boundary_skips"] = [{"doc_id": "adv-x", "reason": "planted"}]
    mp.write_text(json.dumps(m))
    with pytest.raises(bp.SamplerError, match="boundary_skips"):
        bp.verify_mix_nesting(pools)
    lone = tmp_path / "lone"
    lone.mkdir()
    shutil.copy2(built.out1 / "pairs_constv2-mix-1k.manifest.json",
                 lone / "pairs_constv2-mix-1k.manifest.json")
    with pytest.raises(bp.SamplerError, match="fewer than two"):
        bp.verify_mix_nesting(lone)


def test_subset_build_leaves_finished_cells_untouched(built, tmp_path):
    """--cells (round-2): building ONLY the full cell into an already-
    staged dir reproduces its bytes, leaves every other pool/manifest
    byte-identical (no provenance restamp of finished cells), and MERGES
    the summary instead of clobbering the other rows."""
    pools = tmp_path / "pools"
    shutil.copytree(built.out1, pools)
    before = {
        p.name: p.read_bytes() for p in pools.iterdir()
        if "constv2-mix-full" not in p.name and p.name != "pools_summary.json"
    }
    manifests = bp.build_pools(
        pools, built.split_path,
        advice_final=built.advice_path, stories_final=built.stories_path,
        mix_cells=MIX_CELLS, adv_cells=ADV_CELLS, rig=FakeRig(),
        only_cells={"constv2-mix-full"},
    )
    assert set(manifests) == {"constv2-mix-full"}
    for name, data in before.items():
        assert (pools / name).read_bytes() == data, f"{name} was restamped"
    # deterministic: the subset build reproduces the full build's bytes
    assert (pools / "pairs_constv2-mix-full.jsonl").read_bytes() == \
        (built.out1 / "pairs_constv2-mix-full.jsonl").read_bytes()
    summary = json.loads((pools / "pools_summary.json").read_text())
    for cell, _ in MIX_CELLS + ADV_CELLS:
        assert cell in summary["cells"], f"{cell} dropped by the merge"
    assert bp.HELDOUT_CELL in summary["cells"]


def test_subset_build_refuses_unknown_cell_and_foreign_summary(built, tmp_path):
    with pytest.raises(bp.SamplerError, match="unknown cell"):
        bp.build_pools(
            tmp_path / "p1", built.split_path,
            advice_final=built.advice_path, stories_final=built.stories_path,
            mix_cells=MIX_CELLS, adv_cells=ADV_CELLS, rig=FakeRig(),
            only_cells={"constv2-mix-999k"},
        )
    # a summary from a DIFFERENT split refuses the merge
    pools = tmp_path / "p2"
    shutil.copytree(built.out1, pools)
    sp = pools / "pools_summary.json"
    s = json.loads(sp.read_text())
    s["heldout_split_sha"] = "0" * 64
    sp.write_text(json.dumps(s))
    with pytest.raises(bp.SamplerError, match="different held-out split"):
        bp.build_pools(
            pools, built.split_path,
            advice_final=built.advice_path, stories_final=built.stories_path,
            mix_cells=MIX_CELLS, adv_cells=ADV_CELLS, rig=FakeRig(),
            only_cells={"constv2-mix-full"},
        )


# ---------------------------------------------------------------------------
# Truncation scan mechanics (unit level, planted rigs)
# ---------------------------------------------------------------------------

def _filler_doc(n=10):
    text = words("fill", n)
    return bp.Doc(
        doc_id="sto-fill", family="stories", prompt="", pos=text,
        token_counts={k: n for k in hs.TOK_NAMES},
    )


def test_forward_scan_survives_nonmonotonic_merge():
    # candidate totals along the scan: 11 (below), 13 (ABOVE), 12 (== B).
    # An implementation that stops at the first candidate past the budget
    # (or binary-searches a "monotonic" sequence) fails here.
    rig = FakeRig(overrides={
        "aa": 1, "aa bb": 3, "aa bb cc": 2, "aa bb cc dd": 4,
    })
    final = bp.Doc(
        doc_id="sto-merge", family="stories", prompt="", pos="aa bb cc dd",
        token_counts=rig.counts("aa bb cc dd"),
    )
    docs, cum, skips = bp.cut_cell([_filler_doc(10), final], 12, rig)
    assert skips == []
    assert [d.doc_id for d in docs] == ["sto-fill", "sto-merge"]
    t = docs[-1]
    assert t.truncated
    assert t.pos == "aa bb cc"
    assert t.char_offset == len("aa bb cc")
    assert t.cut_candidate_index == 2
    assert t.tokens_taken == {k: 2 for k in hs.TOK_NAMES}
    assert min(cum.values()) == 12


def test_no_exact_cutoff_skips_then_refuses_with_bracketing():
    # The gap doc is skipped; with nothing else left in the draw,
    # the terminal refusal still fires, carrying the bracketing detail.
    rig = FakeRig(overrides={
        "aa": 1, "aa bb": 3, "aa bb cc": 3, "aa bb cc dd": 4,
    })
    final = bp.Doc(
        doc_id="sto-gap", family="stories", prompt="", pos="aa bb cc dd",
        token_counts=rig.counts("aa bb cc dd"),
    )
    with pytest.raises(bp.SamplerError, match="bracketing") as exc:
        bp.cut_cell([_filler_doc(10), final], 12, rig)
    assert "no-exact-cutoff" in str(exc.value)
    assert "sto-gap" in str(exc.value)


@needs_generator
def test_limiting_tokenizer_flip_with_truncation():
    """Budget landing verified by the corpus repo's official_tokens on a set
    whose limiting tokenizer flips between subsets."""
    tokens_mod = hs.generator_tokens()
    d1 = bp.Doc(doc_id="d1", family="stories", prompt="", pos=words("x", 10),
                token_counts={"llama3": 10, "gemma": 2, "qwen": 10})
    d2 = bp.Doc(doc_id="d2", family="stories", prompt="", pos=words("y", 10),
                token_counts={"llama3": 2, "gemma": 10, "qwen": 10})
    rig = FakeRig(overrides={
        "aa": {"llama3": 1, "gemma": 3, "qwen": 1},
        "aa bb": {"llama3": 2, "gemma": 4, "qwen": 2},
        "aa bb cc dd": {"llama3": 4, "gemma": 4, "qwen": 4},
    })
    d3 = bp.Doc(doc_id="d3", family="stories", prompt="", pos="aa bb cc dd",
                token_counts=rig.counts("aa bb cc dd"))
    # cum after d1+d2 = {12, 12, 20}; whole d3 -> {16, 16, 24} min 16 > 13;
    # cut at "aa" -> {13, 15, 21} min 13 == B.
    docs, cum, _ = bp.cut_cell([d1, d2, d3], 13, rig)
    t = docs[-1]
    assert t.truncated and t.pos == "aa"
    records = [
        {"token_counts": d1.token_counts},
        {"token_counts": d2.token_counts},
        {"token_counts": t.tokens_taken},
    ]
    assert tokens_mod.official_tokens(records) == 13
    # the per-record estimator disagrees — the set rule is what held
    naive = sum(min(r["token_counts"].values()) for r in records)
    assert naive == 5 != 13


def test_exact_whole_doc_landing_needs_no_truncation():
    rig = FakeRig()
    d1 = _filler_doc(10)
    d2 = bp.Doc(doc_id="sto-two", family="stories", prompt="", pos=words("z", 5),
                token_counts={k: 5 for k in hs.TOK_NAMES})
    docs, cum, _ = bp.cut_cell([d1, d2], 15, rig)
    assert [d.truncated for d in docs] == [False, False]
    assert min(cum.values()) == 15


def _prefix_eater(doc_id="adv-bigprefix", sys_words=5, body_words=20):
    return bp.Doc(
        doc_id=doc_id, family="advice",
        prompt=words("s", sys_words) + "\n\n", pos=words("b", body_words),
        token_counts={k: sys_words + body_words for k in hs.TOK_NAMES},
    )


def test_masked_prefix_eater_refused_when_draw_ends():
    # A prefix-eater with nothing after it: skipped, then the
    # terminal refusal fires, naming the skip and its reason.
    rig = FakeRig()
    # cum 10 after filler; budget 14: prefix alone brings the set to 15 >= 14
    with pytest.raises(bp.SamplerError, match="masked-prefix-eats-budget"):
        bp.cut_cell([_filler_doc(10), _prefix_eater()], 14, rig)


def test_skip_forward_over_prefix_eater_lands_budget_exactly():
    """The boundary doc's masked prefix eats the remainder -> it is skipped
    and recorded, and the NEXT doc in draw order takes the exact cutoff."""
    rig = FakeRig()
    nxt = bp.Doc(
        doc_id="sto-after", family="stories", prompt="", pos=words("n", 20),
        token_counts={k: 20 for k in hs.TOK_NAMES},
    )
    docs, cum, skips = bp.cut_cell(
        [_filler_doc(10), _prefix_eater(), nxt], 14, rig
    )
    assert [d.doc_id for d in docs] == ["sto-fill", "sto-after"]
    assert docs[-1].truncated
    assert min(cum.values()) == 14
    assert len(skips) == 1
    assert skips[0]["doc_id"] == "adv-bigprefix"
    assert skips[0]["reason"] == "masked-prefix-eats-budget"
    assert skips[0]["detail"]["remainder"] == 4


def test_impossible_everywhere_still_refuses_loudly():
    """Every remaining doc is infeasible (one prefix-eater, one
    no-exact-cutoff) -> loud terminal refusal naming every skip."""
    rig = FakeRig(overrides={
        "aa": 1, "aa bb": 3, "aa bb cc": 3, "aa bb cc dd": 4,
    })
    gap = bp.Doc(
        doc_id="sto-gap", family="stories", prompt="", pos="aa bb cc dd",
        token_counts=rig.counts("aa bb cc dd"),
    )
    with pytest.raises(bp.SamplerError, match="draw exhausted") as exc:
        bp.cut_cell([_filler_doc(10), _prefix_eater(), gap], 12, rig)
    msg = str(exc.value)
    assert "masked-prefix-eats-budget" in msg
    assert "no-exact-cutoff" in msg
    assert "adv-bigprefix" in msg and "sto-gap" in msg


@needs_generator
def test_planted_skip_recorded_in_manifest(tmp_path):
    """Skip-forward end-to-end: a full advice-lane build whose boundary doc is
    a prefix-eater writes the skip into the cell manifest + pools summary."""
    # ids (and hence both sha orders) are independent of doc CONTENT, so
    # compute the orders first and assign sizes by walk position.
    group_keys = [f"p{p:02d}|p{p:02d}-t00" for p in range(8)]
    held_group = sorted(
        sorted(group_keys),
        key=lambda k: (hs.sha_rank(hs.SALT_HELDOUT, k), k),
    )[0]
    train_ids = [
        f"adv-p{p:02d}-t00-s00" for p in range(8)
        if f"p{p:02d}|p{p:02d}-t00" != held_group
    ]
    torder = sorted(
        train_ids, key=lambda i: (hs.sha_rank(bp.SALT_ADV, i), i)
    )
    # walk position -> (system_words, user_words, assistant_words)
    sizes = {torder[0]: (0, 4, 6),    # 10 tokens, taken whole (10 < 14)
             torder[1]: (6, 4, 8),    # boundary doc: prefix 6 > remainder 4
             torder[2]: (0, 8, 12)}   # truncated at 4 tokens -> exactly 14
    advice = []
    for p in range(8):
        did = f"adv-p{p:02d}-t00-s00"
        sw, uw, aw = sizes.get(did, (0, 5, 5))
        advice.append(
            adv_rec(p, 0, 0, system_words=sw or 1, user_words=uw,
                    assistant_words=aw, system=bool(sw))
        )
    stories = [sto_rec(c, 0, k, 0, n_words=20) for c in range(2) for k in range(2)]
    ap = write_jsonl(tmp_path / "advice.jsonl", advice)
    sp = write_jsonl(tmp_path / "stories.jsonl", stories)
    split = hs.build_split(ap, sp, eval_n=2)
    split_path = tmp_path / "split.json"
    split_path.write_text(json.dumps(split) + "\n", encoding="utf-8")

    manifests = bp.build_pools(
        tmp_path / "pools", split_path, advice_final=ap, stories_final=sp,
        lanes=("advice",), adv_cells=[("adv-skiptest", 14)], rig=FakeRig(),
    )
    m = manifests["adv-skiptest"]
    assert m["official_tokens"] == 14
    assert m["doc_ids_in_order"] == [torder[0], torder[2]]
    assert m["truncation"]["doc_id"] == torder[2]
    assert [s["doc_id"] for s in m["boundary_skips"]] == [torder[1]]
    assert m["boundary_skips"][0]["reason"] == "masked-prefix-eats-budget"
    summary = json.loads((tmp_path / "pools" / "pools_summary.json").read_text())
    assert summary["cells"]["adv-skiptest"]["boundary_skips"] == [torder[1]]


def test_corpus_exhausted_fails_loudly():
    rig = FakeRig()
    with pytest.raises(bp.SamplerError, match="exhausted"):
        bp.cut_cell([_filler_doc(10)], 999, rig)


# ---------------------------------------------------------------------------
# Canary + chat markup + rendering guards
# ---------------------------------------------------------------------------

@needs_generator
def test_canary_catches_trained_text_but_ignores_metadata():
    tokens_mod = hs.generator_tokens()
    ok = adv_rec(0, 0, 0)
    ok["scores"] = {"rationale": "Claude judged this fine"}  # metadata only
    doc = bp.make_doc(ok, "advice", tokens_mod)
    assert "Claude" not in doc.prompt + doc.pos

    bad = adv_rec(0, 0, 1)
    bad["assistant"] = bad["assistant"] + " as Claude would say"
    bad["token_counts"] = {k: 0 for k in hs.TOK_NAMES}  # counts irrelevant here
    with pytest.raises(bp.SamplerError, match="[Cc]laude"):
        bp.make_doc(bad, "advice", tokens_mod)


@needs_generator
def test_canary_word_boundary_and_allowlist(monkeypatch):
    tokens_mod = hs.generator_tokens()
    # "philanthropic" contains "anthropic" as a substring — must NOT trip
    phil = adv_rec(1, 0, 0)
    phil["assistant"] = phil["assistant"] + " philanthropic giving works"
    bp.make_doc(phil, "advice", tokens_mod)

    bad = adv_rec(1, 0, 1)
    bad["assistant"] = bad["assistant"] + " built by Anthropic today"
    with pytest.raises(bp.SamplerError, match="[Aa]nthropic"):
        bp.make_doc(bad, "advice", tokens_mod)
    # the allowlist admits a named organic mention
    monkeypatch.setattr(
        bp, "CANARY_ALLOWLIST", frozenset({bad["lineage"]["custom_id"]})
    )
    bp.make_doc(bad, "advice", tokens_mod)


@needs_generator
def test_chat_markup_refused():
    tokens_mod = hs.generator_tokens()
    bad = adv_rec(2, 0, 0)
    bad["assistant"] = bad["assistant"] + " <|im_start|>assistant"
    with pytest.raises(bp.SamplerError, match="chat markup"):
        bp.make_doc(bad, "advice", tokens_mod)


@needs_generator
def test_empty_trained_text_refused():
    tokens_mod = hs.generator_tokens()
    bad = sto_rec(9, 0, 0, 0)
    bad["text"] = ""
    with pytest.raises(bp.SamplerError, match="empty trained text"):
        bp.make_doc(bad, "stories", tokens_mod)


def test_no_chat_markup_in_built_pools(built):
    for pool in built.out1.glob("pairs_*.jsonl"):
        text = pool.read_text(encoding="utf-8")
        for marker in bp.CHAT_MARKUP:
            assert marker not in text, f"{pool.name} contains {marker!r}"
        for line in text.splitlines():
            row = json.loads(line)
            assert not bp.CANARY_RE.search(row["prompt"] + row["pos"])


# ---------------------------------------------------------------------------
# Max-length preflight
# ---------------------------------------------------------------------------

def _mini_pool(tmp_path: Path, prompt_words: int, pos_words: int) -> Path:
    pools = tmp_path / "pools"
    pools.mkdir()
    row = {
        "prompt": words("p", prompt_words) + ("\n\n" if prompt_words else ""),
        "pos": words("c", pos_words),
        "neg": "",
        "instruction_id": "constv2-mix-1k",
        "doc_id": "adv-p00-t00-s00",
        "family": "advice",
        "official_tokens": prompt_words + pos_words,
        "truncated": False,
    }
    (pools / "pairs_constv2-mix-1k.jsonl").write_text(
        json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (pools / "pairs_constv2-mix-1k.manifest.json").write_text(
        json.dumps({"cell": "constv2-mix-1k"}) + "\n", encoding="utf-8"
    )
    return pools


def test_target_tokenizers_certify_every_sdfbase_pin_incl_both_llama():
    """The max-length preflight tokenizes every row with EVERY pinned target
    tokenizer, derived from derive_model_config.SDFBASE_SLUGS (one pin
    registry) — so registering the Llama-3.1-8B lane (2026-09-09) makes the
    preflight certify BOTH Nous mirrors' tokenizers at their exact revisions
    with no edit here, and a pin bump can never leave the preflight
    certifying a tokenizer the fits no longer use."""
    import derive_model_config as dmc

    assert set(bp.TARGET_TOKENIZERS) == set(dmc.SDFBASE_SLUGS)
    for slug in ("llama8b_base", "llama8b_it_doc"):
        assert slug in bp.TARGET_TOKENIZERS
        spec = dmc.MODEL_SPECS[slug]
        assert bp.TARGET_TOKENIZERS[slug] == (spec.model, spec.revision)
    assert bp.TARGET_TOKENIZERS["llama8b_base"] == (
        "NousResearch/Meta-Llama-3.1-8B",
        "1f47e50cdbe801ad8a5174156ec3a0655108fb9f")
    assert bp.TARGET_TOKENIZERS["llama8b_it_doc"] == (
        "NousResearch/Meta-Llama-3.1-8B-Instruct",
        "d10aef7999a2b5ba950ab3974312feeedbfe0b77")
    # the two lanes are separate tokenizer entries even though the two
    # tokenizers are near-identical: sidecars pin the checkpoint
    assert bp.TARGET_TOKENIZERS["llama8b_base"] != \
        bp.TARGET_TOKENIZERS["llama8b_it_doc"]


def test_preflight_refuses_overlength_under_either_tokenizer(tmp_path):
    pools = _mini_pool(tmp_path, prompt_words=5, pos_words=10)
    toks = {
        "fake_gemma": FakeTok("fake_gemma", bos=True),   # 5+1 + 10 + 1 = 17
        "fake_qwen": FakeTok("fake_qwen", bos=False),    # 5 + 10 + 1 = 16
    }
    # violation under the BOS model only — must still refuse, naming the row
    with pytest.raises(bp.PreflightError) as exc:
        bp.run_preflight(pools, max_len=16, toks=toks)
    msg = str(exc.value)
    assert "fake_gemma" in msg
    assert "adv-p00-t00-s00" in msg
    assert "fake_qwen" not in msg
    # no manifest mutation and no report on refusal
    manifest = json.loads(
        (pools / "pairs_constv2-mix-1k.manifest.json").read_text()
    )
    assert "max_encoded_row_len" not in manifest
    assert not (pools / "preflight_report.json").exists()


def test_preflight_passes_and_updates_manifests(tmp_path):
    pools = _mini_pool(tmp_path, prompt_words=5, pos_words=10)
    toks = {
        "fake_gemma": FakeTok("fake_gemma", bos=True),
        "fake_qwen": FakeTok("fake_qwen", bos=False),
    }
    report = bp.run_preflight(pools, max_len=17, toks=toks)
    stats = report["pools"]["pairs_constv2-mix-1k.jsonl"]["models"]
    assert stats["fake_gemma"]["max_encoded_row_len"] == 17
    assert stats["fake_qwen"]["max_encoded_row_len"] == 16
    assert stats["fake_gemma"]["completion_tokens"] == 10
    assert stats["fake_gemma"]["loss_tokens_with_eos"] == 11
    manifest = json.loads(
        (pools / "pairs_constv2-mix-1k.manifest.json").read_text()
    )
    assert manifest["max_encoded_row_len"] == {
        "fake_gemma": 17, "fake_qwen": 16,
    }
    assert manifest["loss_token_counts"]["fake_qwen"]["completion_tokens"] == 10
    assert manifest["preflight"]["max_len"] == 17
    assert (pools / "preflight_report.json").exists()


def test_preflight_on_built_pools_with_fakes(built, tmp_path):
    # run on a COPY so the byte-determinism test's dirs stay untouched
    pools = tmp_path / "pools_pf"
    shutil.copytree(built.out1, pools)
    toks = {
        "fake_gemma": FakeTok("fake_gemma", bos=True),
        "fake_qwen": FakeTok("fake_qwen", bos=False),
    }
    report = bp.run_preflight(pools, max_len=10_000, toks=toks)
    assert set(report["pools"]) == {
        p.name for p in built.out1.glob("pairs_*.jsonl")
    }
    for m_path in pools.glob("pairs_*.manifest.json"):
        manifest = json.loads(m_path.read_text())
        for key in ("fake_gemma", "fake_qwen"):
            assert manifest["max_encoded_row_len"][key] > 0
            assert manifest["loss_token_counts"][key]["completion_tokens"] > 0


# ---------------------------------------------------------------------------
# Control lane: ctl- doc-id prefix, record-not-refuse gate mode,
# no held-out exclusion, dual corpus shas, provenance fields
# ---------------------------------------------------------------------------

def _dummy_cell_docs():
    adv = bp.Doc(doc_id="a1", family="advice", prompt="", pos=words("a", 3),
                 token_counts={k: 50_000 for k in hs.TOK_NAMES})
    sto = bp.Doc(doc_id="s1", family="stories", prompt="", pos=words("s", 3),
                 token_counts={k: 50_000 for k in hs.TOK_NAMES})
    cum = {k: 100_000 for k in hs.TOK_NAMES}
    return [adv, sto], cum


def _build_cell_args(out_dir):
    return dict(salts={"draw": "test"}, out_dir=out_dir, rig=FakeRig(),
                split_sha="0" * 64, corpus_shas={}, generator={})


def test_gate_mode_record_vs_refuse(tmp_path):
    """Per-lane gate mode. refuse (default, corpus lanes) raises on
    a failed gate; record (control lane) writes the truthful failed verdict
    in exactly the shape the waiver emitted: applied/tolerance/deviation/
    passed/waived. A passing gate keeps the 3-key shape in either mode."""
    docs, cum = _dummy_cell_docs()
    args = _build_cell_args(tmp_path)
    # realized 0.5 vs natural 0.105 -> deviation 0.395 >> 0.02
    with pytest.raises(bp.SamplerError, match="ratio gate failed"):
        bp.build_cell("gate-refuse", "mixed", 100_000, docs, cum,
                      natural_ratio=0.105, **args)
    m = bp.build_cell("gate-record", "control", 100_000, docs, cum,
                      natural_ratio=0.105, ratio_gate_mode="record", **args)
    assert list(m["ratio_gate"]) == [
        "applied", "tolerance", "deviation", "passed", "waived",
    ]
    assert m["ratio_gate"]["applied"] is True
    assert m["ratio_gate"]["tolerance"] == 0.02
    assert m["ratio_gate"]["deviation"] == pytest.approx(0.395)
    assert m["ratio_gate"]["passed"] is False
    assert "D15/D10" in m["ratio_gate"]["waived"]
    # passing gate: same 3-key shape in either mode
    m2 = bp.build_cell("gate-pass", "control", 100_000, docs, cum,
                       natural_ratio=0.5, ratio_gate_mode="record", **args)
    assert list(m2["ratio_gate"]) == ["applied", "tolerance", "deviation"]
    with pytest.raises(bp.SamplerError, match="ratio_gate_mode"):
        bp.build_cell("gate-bad", "control", 100_000, docs, cum,
                      natural_ratio=0.5, ratio_gate_mode="maybe", **args)


def test_control_lane_prefix_no_exclusion_and_provenance(built, tmp_path):
    """Control pool rows/manifests carry ctl-<original> ids;
    ranking/selection uses the ORIGINAL ids; docs whose original ids equal
    PRODUCTION held-out ids are still eligible (no held-out carve); dual
    corpus_shas; constdatagen_provenance recorded."""
    # Control docs deliberately re-mint ids that are HELD OUT in the
    # production split (the real collision, 51/58 ids).
    held_adv = sorted(built.split["advice"]["ids"])[0]
    held_sto = sorted(built.split["stories"]["ids"])[:2]
    m_adv = re.fullmatch(r"adv-p(\d+)-t(\d+)-s(\d+)", held_adv)
    ctl_advice = [adv_rec(int(m_adv[1]), int(m_adv[2]), int(m_adv[3]),
                          system_words=2, user_words=3, assistant_words=5)]
    ctl_stories = []
    for sid in held_sto:
        m_sto = re.fullmatch(r"sto-c(\d+)-u(\d+)-k(\d+)-y(\d+)", sid)
        ctl_stories.append(sto_rec(int(m_sto[1]), int(m_sto[2]),
                                   int(m_sto[3]), int(m_sto[4]), n_words=8))
    ca = write_jsonl(tmp_path / "ctl_advice.jsonl", ctl_advice)
    cs = write_jsonl(tmp_path / "ctl_stories.jsonl", ctl_stories)
    run_manifest = tmp_path / "run_manifest.json"
    run_manifest.write_text('{"run_id": "control-test"}\n', encoding="utf-8")

    manifests = bp.build_pools(
        tmp_path / "pools", built.split_path,
        advice_final=built.advice_path, stories_final=built.stories_path,
        lanes=("control",), ctl_cells=[("ctl-test-cell", 20)],
        control_advice=ca, control_stories=cs,
        control_run_manifest=run_manifest, rig=FakeRig(),
    )
    m = manifests["ctl-test-cell"]
    assert m["official_tokens"] == 20
    ids = m["doc_ids_in_order"]
    assert all(i.startswith(bp.CTL_ID_PREFIX) for i in ids)
    stripped = [i[len(bp.CTL_ID_PREFIX):] for i in ids]
    # held-out-colliding originals were SELECTED: no held-out exclusion,
    # and the prefix keeps the recorded ids disjoint from production's
    planted = {held_adv, *held_sto}
    assert set(stripped) <= planted and len(stripped) >= 2
    held = set(built.split["advice"]["ids"]) | set(built.split["stories"]["ids"])
    assert not (set(ids) & held)
    assert m["truncation"]["doc_id"].startswith(bp.CTL_ID_PREFIX)
    # ranking/selection used the ORIGINAL ids: recompute the expected draw
    # with unprefixed docs through the same machinery
    tokens_mod = hs.generator_tokens()
    fams = {
        "advice": bp.sha_shuffle(
            [bp.make_doc(r, "advice", tokens_mod) for r in ctl_advice],
            bp.SALT_CTL),
        "stories": bp.sha_shuffle(
            [bp.make_doc(r, "stories", tokens_mod) for r in ctl_stories],
            bp.SALT_CTL),
    }
    adv_tot = sum(r["token_counts"]["llama3"] for r in ctl_advice)
    sto_tot = sum(r["token_counts"]["llama3"] for r in ctl_stories)
    shares = {"advice": adv_tot / (adv_tot + sto_tot),
              "stories": sto_tot / (adv_tot + sto_tot)}
    exp_docs, _, _ = bp.cut_cell(
        bp.greedy_interleave(fams, shares), 20, FakeRig())
    assert stripped == [d.doc_id for d in exp_docs]
    # natural ratio is the CONTROL corpus's own share, not production's
    assert m["natural_ratio"] == pytest.approx(shares["advice"])
    # dual corpus shas + provenance
    assert set(m["corpus_shas"]) == {"control_advice", "control_stories"}
    prov = m["constdatagen_provenance"]
    assert prov["run_manifest"] == str(run_manifest)
    assert prov["run_manifest_sha256"] == hs.sha256_file(run_manifest)
    assert prov["control_advice_sha256"] == m["corpus_shas"]["control_advice"]
    assert prov["control_stories_sha256"] == m["corpus_shas"]["control_stories"]
    # pool rows carry the prefixed ids too
    rows = [json.loads(l) for l in
            (tmp_path / "pools/pairs_ctl-test-cell.jsonl").read_text().splitlines()]
    assert [r["doc_id"] for r in rows] == ids


def test_control_lane_gate_failure_recorded_end_to_end(built, tmp_path):
    """A control corpus whose advice granularity (~8pp/doc)
    makes the ±2pp gate infeasible at 100k builds WITHOUT raising, and the
    manifest carries the waived verdict; the summary row exists."""
    ctl_advice = [adv_rec(p, 0, 0, system_words=10, user_words=3990,
                          assistant_words=4000) for p in range(3)]  # 3 x 8k
    ctl_stories = [sto_rec(c, u, 0, 0, n_words=3000)
                   for c in range(6) for u in range(5)]             # 30 x 3k
    ca = write_jsonl(tmp_path / "ctl_advice.jsonl", ctl_advice)
    cs = write_jsonl(tmp_path / "ctl_stories.jsonl", ctl_stories)
    manifests = bp.build_pools(
        tmp_path / "pools", built.split_path,
        advice_final=built.advice_path, stories_final=built.stories_path,
        lanes=("control",), control_advice=ca, control_stories=cs,
        rig=FakeRig(),
    )
    m = manifests["constv2ctl-100k"]
    assert m["official_tokens"] == 100_000
    assert m["natural_ratio"] == pytest.approx(24_000 / 114_000)
    g = m["ratio_gate"]
    assert g["applied"] is True and g["passed"] is False
    assert g["deviation"] > 0.02 and isinstance(g["waived"], str)
    assert all(i.startswith(bp.CTL_ID_PREFIX) for i in m["doc_ids_in_order"])
    summary = json.loads(
        (tmp_path / "pools/pools_summary.json").read_text())
    assert "constv2ctl-100k" in summary["cells"]


# ---------------------------------------------------------------------------
# Full corpus (read-only): renderer byte-parity on >=50 docs per family
# ---------------------------------------------------------------------------

REAL_FILES = hs.ADVICE_FINAL.exists() and hs.STORIES_FINAL.exists()


@pytest.mark.skipif(
    not (REAL_FILES and hs.SPLIT_OUT.exists()),
    reason="real corpus finals or committed split not present",
)
def test_real_advice_lane_nesting_survives_skips(tmp_path):
    """adv-1k stays a byte-identical prefix of adv-100k's draw —
    the two masked-prefix skips at the 100k boundary occur past the 1k
    prefix and are recorded, with the budget still exact. Pinning the skip
    ids is safe: build_pools refuses a drifted corpus (sha check vs the
    committed split) before these could go stale silently."""
    manifests = bp.build_pools(
        tmp_path / "pools", hs.SPLIT_OUT, lanes=("advice",),
        rig=bp.StagedCountRig(),
    )
    m1k = manifests["constv2-adv-1k"]
    m100k = manifests["constv2-adv-100k"]
    n = len(m1k["doc_ids_in_order"])
    assert m100k["doc_ids_in_order"][:n] == m1k["doc_ids_in_order"]
    assert m1k["boundary_skips"] == []
    skips = m100k["boundary_skips"]
    assert [s["doc_id"] for s in skips] == [
        "adv-p19-t06-s08", "adv-p34-t00-s01",
    ]
    assert all(s["reason"] == "masked-prefix-eats-budget" for s in skips)
    assert m100k["official_tokens"] == 100_000
    assert {s["doc_id"] for s in skips}.isdisjoint(m100k["doc_ids_in_order"])


@pytest.mark.skipif(not REAL_FILES, reason="real corpus finals not present")
def test_renderer_byte_parity_real_corpus():
    tokens_mod = hs.generator_tokens()
    for family, path in (("advice", hs.ADVICE_FINAL), ("stories", hs.STORIES_FINAL)):
        n = 0
        with open(path, encoding="utf-8") as f:
            for line in itertools.islice(f, 60):
                rec = json.loads(line)
                doc = bp.make_doc(rec, family, tokens_mod)  # asserts parity,
                # chat markup and the canary internally — a violation raises
                rendered = (
                    tokens_mod.render_advice_text(rec)
                    if family == "advice"
                    else tokens_mod.render_story_text(rec)
                )
                assert doc.prompt + doc.pos == rendered
                n += 1
        assert n >= 50, f"{family}: only {n} docs sampled"
