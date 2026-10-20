"""Neutral long-context PADDING for the static one-shot families
(``sisteer.eval.attacks.static_padding``, wired through ``static_common.run_static``
and the four family entry points).

CPU only, no network, no weights beyond the suite's ``tiny_llama`` session fixture
(its tokenizer is what measures the padding). The filler is SYNTHETIC -- a
deterministic neutral text minted here, never the PG-19 cache -- and every
"prompt" is an invented placeholder. What must hold:

1. sizing lands within tolerance at several targets, measured as ``apply``
   measures it (``tokens(sent) - tokens(prompt)``);
2. fixed mode: a larger target's filler strictly EXTENDS the smaller one's and the
   payload is byte-identical across the two (same-sample-grown-padding);
3. per_goal mode: offsets differ across goal ids, are stable across two
   constructions, and stay in range;
4. placement before / after puts the filler on the stated side of the payload;
5. every refusal is LOUD and names its cause: a short filler, a missing or
   malformed cache, nothing found (every path searched), an explicit path that
   does not exist, a bad spec;
6. through run_static: rows carry ``padding``, ``attack_prompt`` is the UNPADDED
   text, ``padded_prompt_sha256`` is the sha256 of what the defense actually
   received, progress rows carry ``padding_tokens``, the summary the block;
7. padding OFF is BYTE-IDENTICAL: the row key set, the bytes, the calls;
8. resume refusals (padded onto unpadded, unpadded onto padded, two token counts)
   raise StaticStoreIncompleteError and leave BOTH files byte-identical;
9. ``window_problem`` fits / no window / a message naming the numbers;
10. the CACHED per-item path (one decode + count per (offset, take); per item the
    filler edge re-tokenized beside the payload) renders byte-identical text and
    an identical ``pad_tokens_actual`` to a full per-item measurement, over
    payload shapes and both placements, cold or hot; per_goal caches per goal; the
    first item of every slice and every Nth item are re-measured in full; a
    tokenizer that is NOT local, or a slice gone stale, is REFUSED, never recorded;
11. the PRE-TOKENIZED SIDECAR: built beside the corpus on first use, loaded by the
    next plan, ids identical; another corpus text, another tokenizer definition, a
    torn file: rebuilt, never used; no corpus file / a test double: no sidecar; an
    unwritable directory warns and the run goes on; the fetch script's
    ``--sidecar-tokenizer`` builds it ahead of a wave and refuses without
    ``--only pg19-filler``.
"""

from __future__ import annotations

import hashlib
import importlib
import inspect
import json
import os
import random
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from attacks_static_common import StubDefense, goals, read_jsonl
from godmode_standin import godmode_standin  # noqa: F401  (grok420 renders the invented G0DM0D3 stand-in)
from sisteer.eval.attacks import get_attack
from sisteer.eval.attacks import static_padding as sp
from sisteer.eval.attacks.static_common import StaticStoreIncompleteError, run_static

# -- synthetic inputs ----------------------------------------------------------

#: an invented, neutral request standing in for a family's transform output
PROMPT = "Please summarize the plot of a novel about a lighthouse keeper."

_WORDS = ("the river bent slowly toward the old mill where a miller kept his ledger "
          "and counted sacks of grain under a lantern while rain tapped the shingles "
          "and the dog slept by the hearth in the long evening").split()

#: the nine keys of an unpadded greedy row, in order (pinned by the sampled-static
#: suite as well; the padding block, when present, is the ONE key appended)
BASE_ROW_KEYS = ["goal_id", "sample_idx", "trial", "variant_key", "variant_name",
                 "family", "variant", "attack_prompt", "response"]
PER_ITEM_KEYS = {"pad_tokens_target", "pad_tokens_actual", "filler_offset_tokens",
                 "padded_prompt_sha256"}
SUMMARY_KEYS = {"responses_path", "progress_path", "per_goal", "n_goals_run",
                "n_goals_total", "n_goals_with_variants", "n_goals_zero_variants",
                "n_target_forwards_total", "batch_size", "n_batches", "n_goals_errored"}


def _filler_text(n_sentences: int = 400, seed: int = 0) -> str:
    """A deterministic neutral corpus of ~6.5k Llama tokens (never PG-19)."""
    rng = random.Random(seed)
    out = []
    for _ in range(n_sentences):
        k = rng.randint(8, 16)
        out.append(" ".join(rng.choice(_WORDS) for _ in range(k)).capitalize() + ".")
    return " ".join(out)


def _count(tok, text: str) -> int:
    return len(tok(text, add_special_tokens=False)["input_ids"])


def _variants(goal, seed: int = 0):
    return [("v0", f"{goal.text} / a"), ("v1", f"{goal.text} / b")]


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _snapshot(out: Path) -> tuple[bytes, bytes]:
    return ((out / "responses.jsonl").read_bytes(), (out / "progress.jsonl").read_bytes())


@pytest.fixture(scope="module")
def tok(tiny_llama):
    return tiny_llama[1]


@pytest.fixture(scope="module")
def filler():
    return _filler_text()


@pytest.fixture(scope="module")
def plan128(tok, filler):
    return sp.PadPlan(sp.PadSpec(tokens=128), tok, filler)


@pytest.fixture(scope="module")
def plan256(tok, filler):
    return sp.PadPlan(sp.PadSpec(tokens=256), tok, filler)


# --------------------------------------------------------------------------- #
# 1. sizing                                                                     #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("target", [128, 512, 2048])
def test_sizing_lands_within_tolerance_as_apply_measures_it(tok, filler, target):
    plan = sp.PadPlan(sp.PadSpec(tokens=target), tok, filler)
    sent, rec = plan.apply(PROMPT, goal_id="g0")
    delta = _count(tok, sent) - _count(tok, PROMPT)          # apply's own measure
    assert rec["pad_tokens_actual"] == delta
    assert abs(delta - target) <= sp.DEFAULT_TOLERANCE * target
    assert rec["pad_tokens_target"] == target
    assert rec["filler_offset_tokens"] == 0                  # fixed mode
    assert rec["padded_prompt_sha256"] == _sha(sent)
    assert set(rec) == PER_ITEM_KEYS
    # deterministic: the same inputs render the identical text and record
    assert plan.apply(PROMPT, goal_id="g0") == (sent, rec)
    # the block is the STORE identity only -- no per-item key leaks into it
    assert set(plan.block) == set(sp.BLOCK_KEYS)
    assert plan.block["tokens"] == target and plan.block["schema"] == sp.SCHEMA
    assert plan.block["separator"] == sp.SEPARATOR


def test_a_tight_tolerance_that_cannot_be_met_refuses_naming_the_target(tok, filler):
    # one adjustment only, and a tolerance the first render cannot satisfy
    plan = sp.PadPlan(sp.PadSpec(tokens=128, tolerance=1e-6, max_iters=1), tok, filler)
    with pytest.raises(sp.StaticPaddingError, match="could not hit 128 padding tokens"):
        plan.apply(PROMPT, goal_id="g0")


# --------------------------------------------------------------------------- #
# 2. fixed mode grows the filler, keeps the payload                             #
# --------------------------------------------------------------------------- #

def test_fixed_mode_larger_target_extends_the_smaller_filler_payload_identical(tok, filler):
    small, _ = sp.PadPlan(sp.PadSpec(tokens=128), tok, filler).apply(PROMPT, goal_id="g0")
    large, _ = sp.PadPlan(sp.PadSpec(tokens=512), tok, filler).apply(PROMPT, goal_id="g0")
    tail = sp.SEPARATOR + PROMPT
    assert small.endswith(tail) and large.endswith(tail)     # payload byte-identical
    f_small, f_large = small[: -len(tail)], large[: -len(tail)]
    assert f_large.startswith(f_small) and len(f_large) > len(f_small)
    # fixed = ONE filler stream at offset 0 for every item, whatever the goal id
    other, rec = sp.PadPlan(sp.PadSpec(tokens=128), tok, filler).apply(
        PROMPT + " Twice.", goal_id="g7")
    f_other = other[: -len(sp.SEPARATOR + PROMPT + " Twice.")]
    assert rec["filler_offset_tokens"] == 0
    assert f_other.startswith(f_small) or f_small.startswith(f_other)


# --------------------------------------------------------------------------- #
# 3. per_goal mode                                                              #
# --------------------------------------------------------------------------- #

def test_per_goal_offsets_differ_by_goal_are_stable_and_in_range(tok, filler):
    spec = sp.PadSpec(tokens=128, mode="per_goal")
    p1, p2 = sp.PadPlan(spec, tok, filler), sp.PadPlan(spec, tok, filler)
    n_filler = _count(tok, filler)
    room = 128 + max(64, 128 // 8)
    offsets, sents = {}, {}
    for gid in ("g0", "g1", "g2", "g3", "g4", "g5"):
        s1, r1 = p1.apply(PROMPT, goal_id=gid)
        assert (s1, r1) == p2.apply(PROMPT, goal_id=gid)     # stable across constructions
        assert 0 <= r1["filler_offset_tokens"] < n_filler - room
        assert abs(r1["pad_tokens_actual"] - 128) <= sp.DEFAULT_TOLERANCE * 128
        assert s1.endswith(sp.SEPARATOR + PROMPT)
        offsets[gid], sents[gid] = r1["filler_offset_tokens"], s1
    assert len(set(offsets.values())) > 1, offsets
    assert len(set(sents.values())) == len(sents)            # a different slice per goal
    # the offset is a function of (target, goal_id): another target moves it
    r256 = sp.PadPlan(sp.PadSpec(tokens=256, mode="per_goal"), tok, filler).apply(
        PROMPT, goal_id="g0")[1]
    assert 0 <= r256["filler_offset_tokens"] < n_filler - (256 + 64)


def test_per_goal_refuses_a_filler_with_no_room_to_vary(tok):
    short = _filler_text(n_sentences=8)                      # ~130 tokens
    plan = sp.PadPlan(sp.PadSpec(tokens=128, mode="per_goal"), tok, short)
    with pytest.raises(sp.StaticPaddingError, match="filler too short for mode=per_goal"):
        plan.apply(PROMPT, goal_id="g0")


# --------------------------------------------------------------------------- #
# 4. placement                                                                  #
# --------------------------------------------------------------------------- #

def test_placement_puts_the_filler_on_the_stated_side(tok, filler):
    before, rb = sp.PadPlan(sp.PadSpec(tokens=128, placement="before"), tok, filler).apply(
        PROMPT, goal_id="g0")
    after, ra = sp.PadPlan(sp.PadSpec(tokens=128, placement="after"), tok, filler).apply(
        PROMPT, goal_id="g0")
    assert before.endswith(sp.SEPARATOR + PROMPT) and not before.startswith(PROMPT)
    assert after.startswith(PROMPT + sp.SEPARATOR) and not after.endswith(PROMPT)
    f_before = before[: -len(sp.SEPARATOR + PROMPT)]
    f_after = after[len(PROMPT + sp.SEPARATOR):]
    assert f_before and f_after
    assert f_before.startswith(f_after) or f_after.startswith(f_before)   # the same stream
    for rec in (rb, ra):
        assert abs(rec["pad_tokens_actual"] - 128) <= sp.DEFAULT_TOLERANCE * 128
    assert before != after and rb["padded_prompt_sha256"] != ra["padded_prompt_sha256"]


# --------------------------------------------------------------------------- #
# 5. loud refusals                                                              #
# --------------------------------------------------------------------------- #

def test_a_filler_shorter_than_the_request_refuses(tok):
    short = _filler_text(n_sentences=4)                      # a few dozen tokens
    n = _count(tok, short)
    plan = sp.PadPlan(sp.PadSpec(tokens=512), tok, short)
    with pytest.raises(sp.StaticPaddingError,
                       match=rf"filler too short: {n} tokens available from offset 0 < 512 needed"):
        plan.apply(PROMPT, goal_id="g0")


def test_load_filler_accepts_the_three_shapes_and_hashes_the_bytes(tmp_path):
    body = "A neutral sentence about weather.  Another about a garden."
    for name, raw in (("s.json", json.dumps(body)),
                      ("o.json", json.dumps({"text": body, "source": "synthetic"})),
                      ("t.txt", body)):
        p = tmp_path / name
        p.write_text(raw)
        text, sha = sp.load_filler(p)
        assert text == body
        assert sha == hashlib.sha256(p.read_bytes()).hexdigest()


@pytest.mark.parametrize("raw, match", [
    (b"[1, 2, 3]", "unrecognized filler cache format"),
    (json.dumps({"body": "x"}).encode(), "unrecognized filler cache format"),
    (b'""', "holds no text"),
    (b"   \n", "holds no text"),
    (b"\xff\xfe\x00bad", "is not UTF-8 text"),
])
def test_load_filler_refuses_a_malformed_cache(tmp_path, raw, match):
    p = tmp_path / "filler.json"
    p.write_bytes(raw)
    with pytest.raises(sp.StaticPaddingError, match=match):
        sp.load_filler(p)


def test_load_filler_refuses_a_missing_file_naming_it(tmp_path):
    p = tmp_path / "absent.json"
    with pytest.raises(sp.StaticPaddingError, match=rf"filler cache unreadable at {p}"):
        sp.load_filler(p)


def test_resolve_filler_path_nothing_found_names_every_path_searched(tmp_path, monkeypatch):
    monkeypatch.delenv(sp.FILLER_ENV, raising=False)
    cfg = tmp_path / "cfg" / "filler.json"                   # named, absent
    with pytest.raises(sp.StaticPaddingError) as ei:
        sp.resolve_filler_path(cfg_value=str(cfg), repo_root=tmp_path)
    msg = str(ei.value)
    assert msg.startswith("no PG-19 filler cache found; searched ")
    for want in (str(cfg), str(tmp_path / sp.REPO_FILLER_REL)):
        assert want in msg, (want, msg)
    assert f"${sp.FILLER_ENV}" in msg and "--static-padding-filler" in msg
    # an env path is searched too (and named) when set
    monkeypatch.setenv(sp.FILLER_ENV, str(tmp_path / "env.json"))
    with pytest.raises(sp.StaticPaddingError, match=str(tmp_path / "env.json")):
        sp.resolve_filler_path(repo_root=tmp_path)


def test_resolve_filler_path_expands_a_placeholder_and_refuses_an_unset_one(tmp_path, monkeypatch):
    """A config's filler_cache in placeholder form expands through sisteer.site; with no value for
    the placeholder the search refuses by name and never probes a literal $SHARED path."""
    monkeypatch.delenv(sp.FILLER_ENV, raising=False)
    monkeypatch.setenv("SISTEER_SITE_DIR", str(tmp_path / "no-site"))
    monkeypatch.delenv("SISTEER_CONFIG", raising=False)
    shared = tmp_path / "shared"
    real = shared / "data" / "pg19_filler.json"
    real.parent.mkdir(parents=True)
    real.write_text(json.dumps("corpus"))
    monkeypatch.setenv("SISTEER_SHARED_ROOT", str(shared))
    for spelled in ("$SHARED/data/pg19_filler.json", "${SHARED}/data/pg19_filler.json"):
        assert sp.resolve_filler_path(cfg_value=spelled, repo_root=tmp_path) == real
        assert sp.resolve_filler_path(explicit=spelled, repo_root=tmp_path) == real
    monkeypatch.delenv("SISTEER_SHARED_ROOT")
    monkeypatch.delenv("CLUSTER_SHARED_ROOT", raising=False)
    for kw in ({"cfg_value": "$SHARED/data/pg19_filler.json"}, {"explicit": "${SHARED}/data/pg19_filler.json"}):
        with pytest.raises(sp.StaticPaddingError, match=r"\$SHARED .*SISTEER_SHARED_ROOT"):
            sp.resolve_filler_path(repo_root=tmp_path, **kw)
    assert not (tmp_path / "$SHARED").exists() and not (tmp_path / "${SHARED}").exists()


def test_resolve_filler_path_explicit_missing_refuses_never_falls_back(tmp_path, monkeypatch):
    # a default DOES exist -- and must not be used
    default = tmp_path / sp.REPO_FILLER_REL
    default.parent.mkdir(parents=True)
    default.write_text(json.dumps("default corpus"))
    missing = tmp_path / "named.json"
    with pytest.raises(sp.StaticPaddingError,
                       match=rf"--static-padding-filler names {missing}, which is not a file"):
        sp.resolve_filler_path(explicit=str(missing), repo_root=tmp_path)
    # and the precedence when things DO exist: explicit > cfg > env > repo default
    monkeypatch.delenv(sp.FILLER_ENV, raising=False)
    assert sp.resolve_filler_path(repo_root=tmp_path) == default
    env = tmp_path / "env.json"
    env.write_text(json.dumps("env corpus"))
    monkeypatch.setenv(sp.FILLER_ENV, str(env))
    assert sp.resolve_filler_path(repo_root=tmp_path) == env
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps("cfg corpus"))
    assert sp.resolve_filler_path(cfg_value=str(cfg), repo_root=tmp_path) == cfg
    explicit = tmp_path / "explicit.json"
    explicit.write_text(json.dumps("explicit corpus"))
    assert sp.resolve_filler_path(explicit=str(explicit), cfg_value=str(cfg),
                                  repo_root=tmp_path) == explicit


@pytest.mark.parametrize("kw, match", [
    ({"tokens": 0}, "padding tokens must be a positive int, got 0"),
    ({"tokens": -5}, "padding tokens must be a positive int"),
    ({"tokens": True}, "padding tokens must be a positive int, got True"),
    ({"tokens": "128"}, "padding tokens must be a positive int, got '128'"),
    ({"tokens": 128, "placement": "middle"},
     r"placement must be one of \['before', 'after'\], got 'middle'"),
    ({"tokens": 128, "mode": "random"}, r"mode must be one of \['fixed', 'per_goal'\], got 'random'"),
    ({"tokens": 128, "tolerance": 0}, r"tolerance must be in \(0, 1\), got 0"),
    ({"tokens": 128, "tolerance": 1.5}, r"tolerance must be in \(0, 1\), got 1.5"),
    ({"tokens": 128, "tolerance": True}, r"tolerance must be in \(0, 1\)"),
    ({"tokens": 128, "max_iters": 0}, "max_iters must be >= 1, got 0"),
])
def test_padspec_refuses_a_bad_field_naming_it(kw, match):
    with pytest.raises(sp.StaticPaddingError, match=match):
        sp.PadSpec(**kw)


def test_padplan_refuses_a_non_spec_or_empty_filler(tok, filler):
    with pytest.raises(sp.StaticPaddingError, match="PadPlan needs a PadSpec, got dict"):
        sp.PadPlan({"tokens": 128}, tok, filler)
    with pytest.raises(sp.StaticPaddingError, match="non-empty filler text"):
        sp.PadPlan(sp.PadSpec(tokens=128), tok, "   ")


def test_block_identity_drops_per_item_keys_and_ignores_non_mappings(plan128):
    _, rec = plan128.apply(PROMPT, goal_id="g0")
    assert sp.block_identity({**plan128.block, **rec}) == plan128.block
    assert sp.block_identity(None) is None
    assert sp.block_identity("padding") is None
    assert sp.block_identity({"tokens": 128}) == {"tokens": 128}


# --------------------------------------------------------------------------- #
# 6. through run_static                                                         #
# --------------------------------------------------------------------------- #

def test_run_static_padded_rows_carry_the_block_and_the_unpadded_prompt(plan128, tmp_path):
    d = StubDefense()
    gs = goals(2)
    out = tmp_path / "p"
    res = run_static(d, gs, out_dir=out, family="fam", variants_fn=_variants, padding=plan128)
    rows = read_jsonl(out / "responses.jsonl")
    assert len(rows) == 4 == len(d.seen)
    expected = [(g.goal_id, i, prompt) for g in gs for i, (_n, prompt) in enumerate(_variants(g))]
    for row, sent, (gid, i, unpadded) in zip(rows, d.seen, expected):
        assert (row["goal_id"], row["sample_idx"]) == (gid, i)
        # the row keeps the UNPADDED transform output ...
        assert row["variant"] == row["attack_prompt"] == unpadded
        # ... while the defense received the padded rendering
        assert sent != unpadded and sent.endswith(sp.SEPARATOR + unpadded)
        pad = row["padding"]
        assert pad["padded_prompt_sha256"] == _sha(sent)
        assert sp.block_identity(pad) == plan128.block
        assert set(pad) == set(sp.BLOCK_KEYS) | PER_ITEM_KEYS
        assert pad["pad_tokens_target"] == 128
        assert abs(pad["pad_tokens_actual"] - 128) <= sp.DEFAULT_TOLERANCE * 128
        assert list(row) == BASE_ROW_KEYS + ["padding"]         # exactly one new key, last
    prog = read_jsonl(out / "progress.jsonl")
    assert [p["goal_id"] for p in prog] == ["g0", "g1"]
    assert all(p["padding_tokens"] == 128 for p in prog)
    assert list(prog[0]) == ["goal_id", "category", "family", "n_variants", "n_target_forwards",
                             "outcome", "capped", "padding_tokens"]
    assert res["padding"] == plan128.block
    assert set(res) == SUMMARY_KEYS | {"padding"}
    # the same block resumes (adds nothing), and one more goal is generated padded
    d2 = StubDefense()
    res2 = run_static(d2, gs, out_dir=out, family="fam", variants_fn=_variants, padding=plan128)
    assert res2["n_goals_run"] == 0 and d2.seen == []
    assert len(read_jsonl(out / "responses.jsonl")) == 4


class _DetailedStub:
    """The served target's DETAILED surface, stubbed: records what it was sent."""

    def __init__(self):
        self.n_target_forwards = 0
        self.batch_size = 4
        self.seen: list[str] = []

    def generate(self, prompts, *, max_new_tokens=512):      # never taken here
        raise AssertionError("greedy surface used under decode")

    def generate_detailed(self, prompts, *, max_new_tokens=512, decode=None, seeds=None, labels=None):
        self.n_target_forwards += len(prompts)
        self.seen.extend(prompts)
        return [{"text": f"sampled:{p[-8:]}", "finish_reason": "stop", "completion_tokens": 7,
                 "prompt_tokens": 3, "max_tokens": max_new_tokens, "sampling_seed": s}
                for p, s in zip(prompts, seeds)]


def test_run_static_sampled_path_sends_the_padded_text_and_records_both_blocks(plan128, tmp_path):
    d = _DetailedStub()
    decode = {"arm": "vendor_recommended", "temperature": 1.0, "top_p": 0.95, "top_k": 64}
    res = run_static(d, goals(1), out_dir=tmp_path / "s", family="fam", variants_fn=_variants,
                     max_new_tokens=64, decode=decode, seed_base=7, padding=plan128)
    rows = read_jsonl(res["responses_path"])
    assert len(rows) == 2 == len(d.seen)
    for row, sent in zip(rows, d.seen):
        assert sent.endswith(sp.SEPARATOR + row["attack_prompt"]) and sent != row["attack_prompt"]
        assert row["padding"]["padded_prompt_sha256"] == _sha(sent)
        assert row["decode"]["arm"] == "vendor_recommended"
        assert list(row) == BASE_ROW_KEYS + ["decode", "sampling_seed", "finish_reason",
                                             "completion_tokens", "prompt_tokens", "max_tokens",
                                             "at_cap", "padding"]
    assert res["decode"]["cap"] == 64 and res["padding"] == plan128.block


class _FailingPlan:
    """A plan whose ``apply`` refuses for ONE goal -- the loud-error policy probe."""

    def __init__(self, inner, fail_goal):
        self._inner, self._fail = inner, fail_goal

    @property
    def spec(self):
        return self._inner.spec

    @property
    def block(self):
        return self._inner.block

    def apply(self, prompt, *, goal_id):
        if goal_id == self._fail:
            raise sp.StaticPaddingError("synthetic refusal: filler too short")
        return self._inner.apply(prompt, goal_id=goal_id)


def test_a_padding_failure_writes_nothing_for_the_chunk_and_leaves_its_goals_uncovered(
        plan128, tmp_path):
    # one chunk for everything: the failure precedes the request, nothing is written
    d = StubDefense()
    out = tmp_path / "one"
    with pytest.raises(sp.StaticPaddingError, match="synthetic refusal"):
        run_static(d, goals(2), out_dir=out, family="fam", variants_fn=_variants,
                   padding=_FailingPlan(plan128, "g0"))
    assert d.seen == []
    assert not (out / "responses.jsonl").exists() and not (out / "progress.jsonl").exists()
    # two chunks: the first goal lands, the failing goal's chunk writes nothing
    d = StubDefense()
    out = tmp_path / "two"
    with pytest.raises(sp.StaticPaddingError, match="synthetic refusal"):
        run_static(d, goals(2), out_dir=out, family="fam", variants_fn=_variants,
                   batch_size=2, padding=_FailingPlan(plan128, "g1"))
    assert len(d.seen) == 2
    assert [r["goal_id"] for r in read_jsonl(out / "responses.jsonl")] == ["g0", "g0"]
    assert [p["goal_id"] for p in read_jsonl(out / "progress.jsonl")] == ["g0"]
    # the resume under the real plan regenerates exactly the uncovered goal
    d = StubDefense()
    res = run_static(d, goals(2), out_dir=out, family="fam", variants_fn=_variants,
                     batch_size=2, padding=plan128)
    assert res["n_goals_run"] == 1 and len(d.seen) == 2
    assert [r["goal_id"] for r in read_jsonl(out / "responses.jsonl")] == ["g0", "g0", "g1", "g1"]


@pytest.mark.parametrize("bad, name", [(128, "int"), ("128", "str"),
                                       (sp.PadSpec(tokens=128), "PadSpec")])
def test_run_static_refuses_a_non_plan_before_writing(tmp_path, bad, name):
    d = StubDefense()
    out = tmp_path / "t"
    with pytest.raises(TypeError, match=rf"needs a static_padding\.PadPlan .* got {name}"):
        run_static(d, goals(1), out_dir=out, family="fam", variants_fn=_variants, padding=bad)
    assert d.seen == []
    assert not (out / "responses.jsonl").exists() and not (out / "progress.jsonl").exists()


# --------------------------------------------------------------------------- #
# 7. padding OFF is byte-identical                                              #
# --------------------------------------------------------------------------- #

def test_padding_off_is_byte_identical(tmp_path):
    d = StubDefense()
    res = run_static(d, goals(1), out_dir=tmp_path / "g", family="fam", variants_fn=_variants,
                     max_new_tokens=16)
    reply = "I'm sorry, I can't help with that."
    lines = (tmp_path / "g" / "responses.jsonl").read_text().splitlines()
    assert lines == [
        json.dumps({"goal_id": "g0", "sample_idx": 0, "trial": 0, "variant_key": "0:0",
                    "variant_name": "v0", "family": "fam", "variant": "benign goal 0 / a",
                    "attack_prompt": "benign goal 0 / a", "response": reply}),
        json.dumps({"goal_id": "g0", "sample_idx": 1, "trial": 0, "variant_key": "1:0",
                    "variant_name": "v1", "family": "fam", "variant": "benign goal 0 / b",
                    "attack_prompt": "benign goal 0 / b", "response": reply}),
    ]
    assert (tmp_path / "g" / "progress.jsonl").read_text().splitlines() == [
        json.dumps({"goal_id": "g0", "category": "illegal", "family": "fam", "n_variants": 2,
                    "n_target_forwards": 2, "outcome": "done", "capped": False})]
    rows = read_jsonl(tmp_path / "g" / "responses.jsonl")
    for row in rows:
        assert list(row) == BASE_ROW_KEYS and "padding" not in row
    assert d.seen == ["benign goal 0 / a", "benign goal 0 / b"]     # the unpadded prompts, as-is
    assert "padding" not in res and set(res) == SUMMARY_KEYS
    assert all("padding_tokens" not in p for p in res["per_goal"])
    # an explicit padding=None is the default: byte-for-byte the same files
    run_static(StubDefense(), goals(1), out_dir=tmp_path / "g2", family="fam",
               variants_fn=_variants, max_new_tokens=16, padding=None)
    assert _snapshot(tmp_path / "g2") == _snapshot(tmp_path / "g")


# --------------------------------------------------------------------------- #
# 8. resume refusals leave both files byte-identical                            #
# --------------------------------------------------------------------------- #

def _partial_row(block=None) -> dict:
    """An interrupted goal's partial row (no progress record): what the resume
    filter WOULD drop. Its survival across a refusal pins the guard's position
    BEFORE ``filter_responses_to_covered``."""
    row = {"goal_id": "g9", "sample_idx": 0, "response": "PARTIAL_INTERRUPTED"}
    if block is not None:
        row["padding"] = {**block, "pad_tokens_target": block["tokens"]}
    return row


def test_unpadded_run_refuses_a_padded_store_byte_identically(plan128, tmp_path):
    out = tmp_path / "padded"
    run_static(StubDefense(), goals(1), out_dir=out, family="fam", variants_fn=_variants,
               padding=plan128)
    with (out / "responses.jsonl").open("a") as f:
        f.write(json.dumps(_partial_row(plan128.block)) + "\n")
    before = _snapshot(out)
    d = StubDefense()
    with pytest.raises(StaticStoreIncompleteError,
                       match=r"holds a PADDED row \(128 padding tokens\) while this run pads NOTHING"):
        run_static(d, goals(2), out_dir=out, family="fam", variants_fn=_variants)
    assert _snapshot(out) == before and d.seen == []
    # positive control: the same block resumes, drops the partial row, regenerates g9 only
    extra = goals(10)[9]
    res = run_static(StubDefense(), goals(1) + [extra], out_dir=out, family="fam",
                     variants_fn=_variants, padding=plan128)
    assert res["n_goals_run"] == 1
    assert "PARTIAL_INTERRUPTED" not in (out / "responses.jsonl").read_text()


def test_padded_run_refuses_an_unpadded_store_byte_identically(plan128, tmp_path):
    out = tmp_path / "unpadded"
    run_static(StubDefense(), goals(1), out_dir=out, family="fam", variants_fn=_variants)
    with (out / "responses.jsonl").open("a") as f:
        f.write(json.dumps(_partial_row()) + "\n")
    before = _snapshot(out)
    d = StubDefense()
    with pytest.raises(StaticStoreIncompleteError,
                       match="holds an UNPADDED row while this run pads 128 tokens"):
        run_static(d, goals(2), out_dir=out, family="fam", variants_fn=_variants, padding=plan128)
    assert _snapshot(out) == before and d.seen == []


def test_two_token_counts_refuse_each_other_byte_identically(tok, filler, plan128, plan256, tmp_path):
    out = tmp_path / "p128"
    run_static(StubDefense(), goals(1), out_dir=out, family="fam", variants_fn=_variants,
               padding=plan128)
    with (out / "responses.jsonl").open("a") as f:
        f.write(json.dumps(_partial_row(plan128.block)) + "\n")
    before = _snapshot(out)
    d = StubDefense()
    with pytest.raises(StaticStoreIncompleteError,
                       match="refusing to mix two padding contracts") as ei:
        run_static(d, goals(2), out_dir=out, family="fam", variants_fn=_variants, padding=plan256)
    assert "'tokens': 128" in str(ei.value) and "'tokens': 256" in str(ei.value)
    assert _snapshot(out) == before and d.seen == []
    # another placement / mode under the same count is a different contract too
    other = sp.PadPlan(sp.PadSpec(tokens=128, placement="after"), tok, filler)
    with pytest.raises(StaticStoreIncompleteError, match="refusing to mix two padding contracts"):
        run_static(StubDefense(), goals(2), out_dir=out, family="fam", variants_fn=_variants,
                   padding=other)
    assert _snapshot(out) == before


def test_the_guard_itself_tolerates_a_torn_tail_and_an_absent_store(plan128, tmp_path):
    p = tmp_path / "responses.jsonl"
    sp.assert_static_padding_consistent(p, plan128.block)     # absent: nothing to compare
    sp.assert_static_padding_consistent(p, None)
    p.write_text(json.dumps({"goal_id": "g0", "padding": {**plan128.block, "pad_tokens_actual": 130}})
                 + "\n" + '{"goal_id": "g1", "pad')          # torn tail
    sp.assert_static_padding_consistent(p, plan128.block)
    with pytest.raises(StaticStoreIncompleteError, match="PADDED row"):
        sp.assert_static_padding_consistent(p, None)


# --------------------------------------------------------------------------- #
# the four family entry points                                                  #
# --------------------------------------------------------------------------- #

FAMILIES = ("pasttense", "codeattack", "renellm", "grok420")


@pytest.mark.parametrize("family", FAMILIES)
def test_family_entry_point_forwards_padding_to_run_static(monkeypatch, tmp_path, family):
    mod = importlib.import_module(f"sisteer.eval.attacks.{family}")
    seen: dict = {}

    def recorder(defense, gs, **kw):
        seen.update(kw)
        return {"recorded": True}

    monkeypatch.setattr(mod, "run_static", recorder)
    expected = {"recorded": True}
    if family == "renellm":
        # past the corpus gate (an admitted identity; no store record written); the run returns the corpus it replayed
        identity = {"family": "renellm", "content_hash": "0" * 64, "paper_corpus": True, "acknowledged_by": None}
        monkeypatch.setattr(mod, "admitted_identity", lambda: dict(identity))
        monkeypatch.setattr(mod, "bind_store", lambda out_dir, ident: None)
        expected = {"recorded": True, "renellm_corpus": identity}
    fn = get_attack(family)
    sentinel = object()
    assert fn(StubDefense(), goals(1), out_dir=tmp_path / family, padding=sentinel) == expected
    assert seen["padding"] is sentinel and seen["family"] == family
    assert seen["decode"] is None and seen["seed_base"] is None
    seen.clear()
    fn(StubDefense(), goals(1), out_dir=tmp_path / family)
    assert seen["padding"] is None                            # the default, forwarded like decode
    params = inspect.signature(fn).parameters
    assert params["padding"].kind is inspect.Parameter.KEYWORD_ONLY
    assert params["padding"].default is None


@pytest.mark.parametrize("family", ("pasttense", "codeattack", "grok420"))
def test_family_entry_point_end_to_end_pads_what_is_sent(plan128, tmp_path, family, godmode_standin):
    d = StubDefense()
    res = get_attack(family)(d, goals(1), out_dir=tmp_path / family, max_new_tokens=32,
                             padding=plan128)
    rows = read_jsonl(res["responses_path"])
    assert rows and len(rows) == len(d.seen)
    for row, sent in zip(rows, d.seen):
        assert row["family"] == family
        assert sent.endswith(sp.SEPARATOR + row["attack_prompt"])
        assert row["padding"]["padded_prompt_sha256"] == _sha(sent)
        assert sp.block_identity(row["padding"]) == plan128.block
    assert all(p["padding_tokens"] == 128 for p in read_jsonl(res["progress_path"]))
    assert res["padding"] == plan128.block


@pytest.mark.parametrize("family", ("pap", "encoding", "aim"))
def test_the_knob_is_scoped_to_the_four_static_one_shot_families(family):
    assert "padding" not in inspect.signature(get_attack(family)).parameters


# --------------------------------------------------------------------------- #
# 9. the window preflight                                                       #
# --------------------------------------------------------------------------- #

def test_window_problem_fits_no_window_and_names_the_numbers():
    assert sp.window_problem(pad_tokens=1024, cap=512, window=8192) is None
    assert sp.window_problem(pad_tokens=1024, cap=512, window=None) is None
    assert sp.window_problem(pad_tokens=1024, cap=512, window=1024 + 512 + 2048) is None  # exact
    msg = sp.window_problem(pad_tokens=4096, cap=512, window=4096)
    assert msg is not None
    for want in ("padding 4096 tokens", "completion cap 512", "reserve 2048", "= 6656 tokens",
                 "served window 4096", "--static-padding-tokens", "--target-max-model-len"):
        assert want in msg, (want, msg)
    # the reserve moves the sum
    assert sp.window_problem(pad_tokens=4096, cap=512, window=4608, reserve=0) is None
    assert "reserve 0 = 4608" in sp.window_problem(pad_tokens=4096, cap=512, window=4607, reserve=0)


# --------------------------------------------------------------------------- #
# 10. the cached per-item path == the full measurement                          #
# --------------------------------------------------------------------------- #

#: payload shapes that exercise the seam: whitespace and newlines touching the
#: separator, digits, non-ASCII at both ends, a long payload, one huge chunk
PAYLOADS = (
    PROMPT,
    "  " + PROMPT,
    PROMPT + "\n",
    "\n\n" + PROMPT + "\n\n",
    "1234567890 " + PROMPT,
    "\u2026" + PROMPT + " \u2014 fin.",
    " ".join(f"{PROMPT} ({i})" for i in range(40)),
    "x" * 300,
)


def _full_delta(tok, sent: str, prompt: str) -> int:
    """The uncached measurement, computed independently of the plan."""
    return _count(tok, sent) - _count(tok, prompt)


@pytest.mark.parametrize("placement", sp.PLACEMENTS)
@pytest.mark.parametrize("target", [128, 2048])
def test_cached_path_matches_the_full_measurement_over_payloads(tok, filler, placement, target):
    spec = sp.PadSpec(tokens=target, placement=placement)
    cached = sp.PadPlan(spec, tok, filler)                       # the default: sampled full checks
    full = sp.PadPlan(spec, tok, filler, verify_every=1)         # every item tokenized in full
    for i, payload in enumerate(PAYLOADS):
        sent, rec = cached.apply(payload, goal_id=f"g{i}")
        assert (sent, rec) == full.apply(payload, goal_id=f"g{i}")          # byte-identical, same count
        assert rec["pad_tokens_actual"] == _full_delta(tok, sent, payload)  # and TRUE
        assert rec["padded_prompt_sha256"] == _sha(sent)
        assert abs(rec["pad_tokens_actual"] - target) <= sp.DEFAULT_TOLERANCE * target
    n = len(PAYLOADS)
    assert cached.stats["items"] == full.stats["items"] == n
    assert cached.stats["slices"] < n                          # fixed mode: the slice was reused
    assert 1 <= cached.stats["verified"] < n                   # the fast path actually carried items
    assert full.stats["verified"] == n
    assert cached.verify_every == sp.DEFAULT_VERIFY_EVERY == 16 and full.verify_every == 1


def test_cached_path_is_deterministic_whatever_the_plan_rendered_before(tok, filler):
    spec = sp.PadSpec(tokens=512)
    hot = sp.PadPlan(spec, tok, filler)
    for i, payload in enumerate(PAYLOADS):
        hot.apply(payload, goal_id=f"g{i}")
    for i, payload in enumerate(reversed(PAYLOADS)):
        cold = sp.PadPlan(spec, tok, filler)                    # nothing cached yet
        assert cold.apply(payload, goal_id=f"g{i}") == hot.apply(payload, goal_id=f"g{i}")
        assert cold.stats == {"items": 1, "verified": 1, "slices": 1}


def test_per_goal_mode_caches_one_slice_per_goal_and_stays_true(tok, filler):
    # a wide tolerance so every goal converges on its first take: one slice each
    plan = sp.PadPlan(sp.PadSpec(tokens=128, mode="per_goal", tolerance=0.1), tok, filler)
    n_goals, n_variants = 4, 3
    offsets = set()
    for g in range(n_goals):
        for v in range(n_variants):
            payload = f"{PROMPT} variant {v}"
            sent, rec = plan.apply(payload, goal_id=f"g{g}")
            assert rec["pad_tokens_actual"] == _full_delta(tok, sent, payload)
            offsets.add(rec["filler_offset_tokens"])
    assert len(offsets) == n_goals                             # a different slice per goal ...
    assert plan.stats == {"items": n_goals * n_variants,       # ... decoded once per goal,
                          "verified": n_goals,                 # its first variant measured in full,
                          "slices": n_goals}                   # the other variants riding the cache


def test_verify_every_measures_the_first_and_every_nth_use_of_a_slice_in_full(tok, filler):
    plan = sp.PadPlan(sp.PadSpec(tokens=256, tolerance=0.1), tok, filler, verify_every=4)
    for i in range(9):
        plan.apply(f"{PROMPT} {i}", goal_id="g0")
    assert plan.stats == {"items": 9, "verified": 3, "slices": 1}      # uses 1, 5, 9
    for bad in (0, -1, True, "2", 1.5):
        with pytest.raises(sp.StaticPaddingError, match="verify_every must be a positive int"):
            sp.PadPlan(sp.PadSpec(tokens=256), tok, filler, verify_every=bad)


class _NonLocalTok:
    """A tokenizer that is NOT local: a long text carrying the payload costs one
    extra token -- an effect the seam window cannot see. The plan must refuse
    at the first item rather than record the window's count."""

    is_fast = False

    def __init__(self, inner, marker):
        self._t, self._m = inner, marker

    def __call__(self, text, add_special_tokens=False):
        ids = list(self._t(text, add_special_tokens=add_special_tokens)["input_ids"])
        if len(text) > 4 * sp.SEAM_WINDOW_CHARS and self._m in text:
            ids.append(0)
        return {"input_ids": ids}

    def decode(self, ids, skip_special_tokens=True):
        return self._t.decode(ids, skip_special_tokens=skip_special_tokens)


@pytest.mark.parametrize("placement", sp.PLACEMENTS)
def test_a_tokenizer_that_is_not_local_is_refused_at_the_first_item_never_recorded(tok, filler, placement):
    weird = _NonLocalTok(tok, PROMPT)
    spec = sp.PadSpec(tokens=2048, placement=placement)        # ~8k chars: past the window's reach
    with pytest.raises(sp.StaticPaddingError, match="padding measurement disagrees for goal 'g0'") as ei:
        sp.PadPlan(spec, weird, filler).apply(PROMPT, goal_id="g0")
    assert "verify_every=1" in str(ei.value) and f"placement {placement}" in str(ei.value)
    # the full route measures it: the extra token IS the delta under this tokenizer
    sent, rec = sp.PadPlan(spec, weird, filler, verify_every=1).apply(PROMPT, goal_id="g0")
    assert rec["pad_tokens_actual"] == _full_delta(weird, sent, PROMPT)


def test_a_cached_slice_gone_stale_is_caught_within_verify_every_uses(tok, filler):
    plan = sp.PadPlan(sp.PadSpec(tokens=512, tolerance=0.1), tok, filler, verify_every=4)
    _sent, rec = plan.apply(PROMPT, goal_id="g0")             # decodes + measures the slice in full
    (sl,) = plan._slices.values()
    sl.n_edge -= 1                                            # the cached edge count is now wrong by one
    # the sampled guarantee, stated exactly: uses 2-4 ride the stale cache ...
    for _ in range(3):
        assert plan.apply(PROMPT, goal_id="g0")[1]["pad_tokens_actual"] == rec["pad_tokens_actual"] + 1
    # ... and the next full measurement REFUSES, rather than recording the drift
    with pytest.raises(sp.StaticPaddingError, match="disagrees"):
        plan.apply(PROMPT, goal_id="g0")


# --------------------------------------------------------------------------- #
# 11. the pre-tokenized sidecar                                                 #
# --------------------------------------------------------------------------- #

class _Double:
    """A whitespace tokenizer: no ``backend_tokenizer``, so no identity."""

    def __init__(self):
        self.vocab: dict = {}
        self.inv: list = []

    def __call__(self, text, add_special_tokens=False):
        ids = []
        for w in text.split():
            if w not in self.vocab:
                self.vocab[w] = len(self.inv)
                self.inv.append(w)
            ids.append(self.vocab[w])
        return {"input_ids": ids}

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(self.inv[i] for i in ids)


def _corpus(tmp_path, text, name="pg19_filler.json") -> Path:
    p = tmp_path / name
    p.write_text(json.dumps({"text": text}))
    return p


def _spec_for(path: Path, **kw) -> sp.PadSpec:
    text, sha = sp.load_filler(path)
    return sp.PadSpec(tokens=kw.pop("tokens", 128), filler_path=str(path), filler_sha256=sha, **kw)


def test_tokenizer_identity_hashes_the_definition_and_is_none_for_a_double(tok):
    ident = sp.tokenizer_identity(tok)
    assert set(ident) == {"class", "name_or_path", "sha256"}
    assert len(ident["sha256"]) == 64 and ident["class"] == type(tok).__name__
    assert sp.tokenizer_identity(tok) == ident                 # stable
    assert sp.tokenizer_identity(_Double()) is None
    assert sp.tokenizer_identity(object()) is None
    side = sp.filler_ids_sidecar_path("/data/pg19_filler.json", ident)
    assert side == Path(f"/data/pg19_filler.ids.{ident['sha256'][:16]}.json")


def test_sidecar_is_built_on_first_use_and_loaded_by_the_next_plan(tok, filler, tmp_path, capsys):
    corpus = _corpus(tmp_path, filler)
    spec = _spec_for(corpus)
    ident = sp.tokenizer_identity(tok)
    side = sp.filler_ids_sidecar_path(corpus, ident)
    sp._filler_ids_cache.clear()
    first = sp.PadPlan(spec, tok, filler)
    assert first.ids_source == "built" and side.is_file()
    d = json.loads(side.read_text())
    assert d["schema"] == sp.IDS_SCHEMA
    assert d["filler_sha256"] == spec.filler_sha256            # the CORPUS file, as the block records
    assert d["text_sha256"] == _sha(filler)                    # the text these ids tokenize
    assert d["tokenizer"] == {**ident, "revision": None}
    assert d["n_tokens"] == len(d["ids"]) == len(tok(filler, add_special_tokens=False)["input_ids"])
    assert d["ids"] == list(tok(filler, add_special_tokens=False)["input_ids"])
    assert not list(tmp_path.glob("*.tmp"))                    # atomic: no temp file left behind
    sp._filler_ids_cache.clear()
    second = sp.PadPlan(spec, tok, filler)
    assert second.ids_source == "sidecar" and second._ids == first._ids
    assert second.apply(PROMPT, goal_id="g0") == first.apply(PROMPT, goal_id="g0")
    # the block is untouched: the sidecar enters no row
    assert set(second.block) == set(sp.BLOCK_KEYS) and second.block["filler_sha256"] == spec.filler_sha256
    # a plan built twice in one process reads neither the sidecar nor the corpus again
    third = sp.PadPlan(spec, tok, filler)
    assert third.ids_source == "sidecar" and third._ids is second._ids
    assert capsys.readouterr().err == ""


def test_a_sidecar_for_another_corpus_or_tokenizer_is_never_used(tok, filler, tmp_path, capsys):
    corpus = _corpus(tmp_path, filler)
    ident = sp.tokenizer_identity(tok)
    side = sp.filler_ids_sidecar_path(corpus, ident)
    good = list(tok(filler, add_special_tokens=False)["input_ids"])

    def rebuilds(expect_in_err: str):
        sp._filler_ids_cache.clear()
        plan = sp.PadPlan(_spec_for(corpus), tok, sp.load_filler(corpus)[0])
        err = capsys.readouterr().err
        assert plan.ids_source == "built", (plan.ids_source, err)
        assert expect_in_err in err and "rebuilding from the corpus" in err
        return plan

    # 1. the corpus changed under a sidecar built from the old text
    sp._filler_ids_cache.clear()
    sp.PadPlan(_spec_for(corpus), tok, filler)
    assert side.is_file()
    other = _filler_text(n_sentences=300, seed=1)
    corpus.write_text(json.dumps({"text": other}))
    plan = rebuilds("different corpus text")
    assert plan._ids == list(tok(other, add_special_tokens=False)["input_ids"]) != good
    assert json.loads(side.read_text())["text_sha256"] == _sha(other)
    corpus.write_text(json.dumps({"text": filler}))
    # 2. a sidecar whose tokenizer definition is not this one (the file name says
    #    otherwise: hand-copied, or a hash-prefix collision)
    wrong = {**json.loads(side.read_text()), "text_sha256": _sha(filler),
             "tokenizer": {**ident, "sha256": "00" * 32}, "ids": good, "n_tokens": len(good)}
    side.write_text(json.dumps(wrong))
    rebuilds("different tokenizer definition")
    assert json.loads(side.read_text())["tokenizer"]["sha256"] == ident["sha256"]
    # 3. ids that are not the recorded count of ints; 4. another schema; 5. a torn file
    short = {**json.loads(side.read_text()), "ids": good[:-1]}
    side.write_text(json.dumps(short))
    rebuilds("recorded count of ints")
    side.write_text(json.dumps({**json.loads(side.read_text()), "schema": "static_padding_ids/0"}))
    rebuilds("schema 'static_padding_ids/0'")
    side.write_text(side.read_text()[:-40])
    rebuilds("unreadable filler sidecar")
    assert json.loads(side.read_text())["ids"] == good           # whole again


def test_no_sidecar_without_a_corpus_file_or_a_hashable_tokenizer(tok, filler, tmp_path):
    # no filler_path (the tests' own plans), a path that is not a file, a text
    # that is not the file's text, a tokenizer without a definition: all
    # tokenize in-process and write nothing anywhere
    before = sorted(os.listdir(tmp_path))
    sp._filler_ids_cache.clear()
    assert sp.PadPlan(sp.PadSpec(tokens=128), tok, filler).ids_source == "tokenized"
    sp._filler_ids_cache.clear()
    ghost = tmp_path / "absent" / "pg19_filler.json"
    assert sp.PadPlan(sp.PadSpec(tokens=128, filler_path=str(ghost), filler_sha256="ab" * 32),
                      tok, filler).ids_source == "tokenized"
    corpus = _corpus(tmp_path, filler)
    sp._filler_ids_cache.clear()
    assert sp.PadPlan(_spec_for(corpus), tok, filler + " extra").ids_source == "tokenized"
    double = _Double()
    plan = sp.PadPlan(sp.PadSpec(tokens=16, filler_path=str(corpus), filler_sha256="ab" * 32),
                      double, filler)
    assert plan.ids_source == "tokenized"
    assert plan.apply(PROMPT, goal_id="g0")[1]["pad_tokens_actual"] == 16
    assert sorted(os.listdir(tmp_path)) == before + ["pg19_filler.json"]


def test_an_unwritable_sidecar_location_warns_and_the_run_goes_on(tok, filler, tmp_path, capsys):
    if os.geteuid() == 0:
        pytest.skip("root can write anywhere")
    ro = tmp_path / "ro"
    ro.mkdir()
    corpus = _corpus(ro, filler)
    spec = _spec_for(corpus)
    ro.chmod(0o500)
    try:
        sp._filler_ids_cache.clear()
        plan = sp.PadPlan(spec, tok, filler)
        assert plan.ids_source == "built-unwritten"
        err = capsys.readouterr().err
        assert "could not write the pre-tokenized filler sidecar" in err
        sent, rec = plan.apply(PROMPT, goal_id="g0")
        assert rec["pad_tokens_actual"] == _full_delta(tok, sent, PROMPT)
        assert sorted(os.listdir(ro)) == ["pg19_filler.json"]
    finally:
        ro.chmod(0o700)


def test_ensure_filler_ids_reads_the_corpus_when_no_text_is_given(tok, filler, tmp_path):
    corpus = _corpus(tmp_path, filler)
    ids, status = sp.ensure_filler_ids(corpus, tok, tokenizer_revision="deadbeef")
    assert status == "built" and ids == list(tok(filler, add_special_tokens=False)["input_ids"])
    side = sp.filler_ids_sidecar_path(corpus, sp.tokenizer_identity(tok))
    d = json.loads(side.read_text())
    assert d["tokenizer"]["revision"] == "deadbeef" and d["filler_sha256"] == sp.load_filler(corpus)[1]
    assert sp.ensure_filler_ids(corpus, tok) == (ids, "sidecar")
    # write=False: the in-process route, nothing touched
    side.unlink()
    assert sp.ensure_filler_ids(corpus, tok, write=False) == (ids, "tokenized") and not side.exists()
    with pytest.raises(sp.StaticPaddingError, match="no filler text given and no corpus file"):
        sp.ensure_filler_ids(tmp_path / "absent.json", tok)


def test_fetch_script_builds_the_sidecar_beside_the_corpus(monkeypatch, tmp_path, tok, filler, capsys):
    import fetch_stage2_data as fetch
    import sisteer.models as models_mod

    monkeypatch.setattr(fetch, "_shared_root", lambda: tmp_path / "no-such-root")   # a dev box
    monkeypatch.setattr(fetch, "ensure_pg19_filler",
                        lambda *a, **k: fetch.Item("pg19 filler", "PRESENT"))
    loads: list = []

    def fake_load_tokenizer(name, revision=None, **kw):
        loads.append((name, revision))
        return tok

    monkeypatch.setattr(models_mod, "load_tokenizer", fake_load_tokenizer)
    corpus = _corpus(tmp_path, filler)
    argv = ["fetch_stage2_data.py", "--only", "pg19-filler", "--filler-out", str(corpus),
            "--sidecar-tokenizer", "org/model", "--sidecar-revision", "abc123"]
    monkeypatch.setattr(sys, "argv", argv)
    fetch.main()
    out = capsys.readouterr().out
    side = sp.filler_ids_sidecar_path(corpus, sp.tokenizer_identity(tok))
    assert loads == [("org/model", "abc123")]
    assert side.is_file() and "[BUILT" in out and str(side) in out and "all required items provisioned" in out
    d = json.loads(side.read_text())
    assert d["tokenizer"]["revision"] == "abc123" and d["filler_sha256"] == sp.load_filler(corpus)[1]
    assert d["ids"] == list(tok(filler, add_special_tokens=False)["input_ids"])
    # a second run finds it: PRESENT, the file untouched
    before = side.read_bytes()
    monkeypatch.setattr(sys, "argv", argv)
    fetch.main()
    assert "[PRESENT" in capsys.readouterr().out and side.read_bytes() == before
    # and a plan over the same corpus loads what the script built
    sp._filler_ids_cache.clear()
    assert sp.PadPlan(_spec_for(corpus), tok, filler).ids_source == "sidecar"
    # refusals: the flag rides --only pg19-filler; the revision rides the tokenizer
    for bad, msg in ((["fetch_stage2_data.py", "--sidecar-tokenizer", "org/model"], "--only pg19-filler"),
                     (["fetch_stage2_data.py", "--only", "pg19-filler", "--sidecar-revision", "x"],
                      "--sidecar-revision needs --sidecar-tokenizer")):
        monkeypatch.setattr(sys, "argv", bad)
        with pytest.raises(SystemExit) as ei:
            fetch.main()
        assert ei.value.code == 2 and msg in capsys.readouterr().err
    # an absent corpus is a FAILED row, exit 1
    monkeypatch.setattr(sys, "argv", ["fetch_stage2_data.py", "--only", "pg19-filler", "--filler-out",
                                      str(tmp_path / "nowhere.json"), "--sidecar-tokenizer", "org/model"])
    with pytest.raises(SystemExit) as ei:
        fetch.main()
    assert ei.value.code == 1 and "no filler cache at" in capsys.readouterr().out
