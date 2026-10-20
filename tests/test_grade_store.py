"""grade-store: grading the labelled store ITSELF (the pool_sample lane gap).

CPU-only, offline, zero API calls, NO harmful text: every prompt/response is
an INVENTED synthetic string (privacy rule — placeholder text, no real store
data). The gap this subcommand closes (verified 2026-08-20): ``pool_sample.py
sample-all`` -> ``merge`` writes every store row ``score: null, graded: false``
by construction, ``_row_to_graded`` skips unscored harmful rows, and
``grade-pos``/``grade-benign`` only ever read the generate-pos/-benign RUN
dirs — so a merged store yields zero matched pairs until grade-store runs.

What the suite proves:
- harmful rows are scored under EVERY pool condition (undefended `none`, the
  store's own prompted SI, explicit-si rows, the anti pseudo-SI, replay
  families) — and `compiled` / non-comparison-SI rows are NOT sent;
- benign SI rows get XSTest labels; benign `none` rows are never sent;
- resume: already-scored/labelled rows are skipped, and a fully-graded store
  constructs no grader at all;
- the atomic rewrite preserves every other field and the 0600 mode;
- --dry-run makes zero grader constructions/calls and writes nothing;
- a grader failure propagates loudly and leaves the file byte-identical.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import build_two_si_pool as bp  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
SI = ("safety_refusal", "si_constitution")
ANTI = "anti_safety_refusal"

P_NONE = "SYNTHETIC store prompt alpha harmful-shaped 11110000"
P_SI = "SYNTHETIC store prompt bravo harmful-shaped 22220000"
P_STORE_SI = "SYNTHETIC store prompt charlie harmful-shaped 33330000"
P_ANTI = "SYNTHETIC store prompt delta harmful-shaped 44440000"
P_COMPILED = "SYNTHETIC store prompt echo harmful-shaped 55550000"
P_OTHER_SI = "SYNTHETIC store prompt foxtrot harmful-shaped 66660000"
P_DONE = "SYNTHETIC store prompt golf harmful-shaped 77770000"
P_REPLAY = "SYNTHETIC store prompt hotel harmful-shaped 88880000"
P_BENIGN = "SYNTHETIC store benign prompt india harmless 99990000"
R = "SYNTHETIC store response text aaaa0000"


def _engine_row(fam, gid, sample, si_condition, prompt, *, si_id=None,
                score=None, candidate_idx=0, response=R, xstest_label=None):
    """One row in the exact schema ``sisteer.sampling.engine._row`` writes and
    ``pool_sample.py merge`` folds into ``<store>/<family>.jsonl``."""
    row = {
        "attack": fam, "goal_id": gid, "sample_idx": sample, "trial": 0,
        "si_condition": si_condition, "si_id": si_id,
        "candidate_idx": candidate_idx, "category": None,
        "category_group": None, "source": fam, "side": "harmful",
        "variant_name": None, "attack_prompt": prompt, "response": response,
        "score": score, "graded": score is not None,
    }
    if xstest_label is not None:
        row["xstest_label"] = xstest_label
    return row


def _world(tmp_path):
    """A synthetic merged store: one static family (bon), one replay family
    (pair), one benign family (phtest); anti pseudo-SI enabled."""
    cfg = yaml.safe_load((REPO / "configs" / "two_si.yaml").read_text())
    cfg["si_ids"] = list(SI)
    cfg["pool"] = {**cfg["pool"], "store_dir": str(tmp_path / "labeled"),
                   "static_families": ["bon"], "train_families": ["bon"],
                   "replay_families": ["pair"],
                   "benign_source": "phtest", "benign_sources": [],
                   # the pre-flip world: the STORE still serves the store SI's
                   # pos rows (no pool.pos_runs.safety_refusal), so its
                   # prompted rows are pool rows and get graded. The flipped
                   # world has its own test below.
                   "pos_runs": {k: v
                                for k, v in (cfg["pool"].get("pos_runs")
                                             or {}).items()
                                if k != "safety_refusal"},
                   # the pre-feature default: benign none rows need no label.
                   # The enabled world lives in tests/test_bidpo_benign.py.
                   "bidpo_benign": {"enabled": False},
                   "anti_neg": {"enabled": True, "anti_si_id": ANTI,
                                "run": "two_si_anti_neg", "candidates": 8,
                                "temperature": 1.0}}

    store = tmp_path / "labeled"
    store.mkdir(parents=True)
    bon = [
        # the four gradable conditions, all ungraded (merge output shape)
        _engine_row("bon", "g0", 0, "none", P_NONE),
        _engine_row("bon", "g1", 1, "prompted", P_SI, si_id=SI[1]),
        _engine_row("bon", "g2", 2, "prompted", P_STORE_SI),  # store SAFETY_SI
        _engine_row("bon", "g3", 3, "prompted", P_ANTI, si_id=ANTI,
                    candidate_idx=1),
        # never graded: an object-defense transcript + a non-comparison SI
        _engine_row("bon", "g4", 4, "compiled", P_COMPILED),
        _engine_row("bon", "g5", 5, "prompted", P_OTHER_SI, si_id="si_other"),
        # already graded — resume must skip it
        _engine_row("bon", "g6", 6, "none", P_DONE, score=0.9),
    ]
    pair = [_engine_row("pair", "g7", 7, "none", P_REPLAY)]
    phtest = [
        {**_engine_row("phtest", "b0", 0, "none", P_BENIGN), "side": "benign"},
        {**_engine_row("phtest", "b1", 1, "prompted", P_BENIGN, si_id=SI[0]),
         "side": "benign", "xstest_label": None},
        {**_engine_row("phtest", "b2", 2, "prompted", P_BENIGN, si_id=SI[1],
                       xstest_label="C"), "side": "benign"},  # already labelled
    ]
    for name, rows in (("bon", bon), ("pair", pair), ("phtest", phtest)):
        p = store / f"{name}.jsonl"
        p.write_text("".join(json.dumps(r) + "\n" for r in rows))
        p.chmod(0o600)
    return cfg, store


def _args(**over):
    base = dict(store_dir=None, dry_run=False, concurrency=8)
    base.update(over)
    return types.SimpleNamespace(**base)


def _fake_graders(monkeypatch, *, score=0.75, label="C"):
    """Stand-in graders recording every item they were asked to grade."""
    seen = {"sr": [], "xs": [], "sr_built": 0, "xs_built": 0}

    def _build(args, cfg, cache_dir, cache_name=None):
        seen["sr_built"] += 1
        seen["cache_dir"] = Path(cache_dir)
        seen["cache_name"] = cache_name

        def grade(items):
            seen["sr"].extend(items)
            return {it: score for it in items}
        return grade

    def _build_xs(args, cfg, cache_dir):
        seen["xs_built"] += 1

        def grade_xs(items):
            seen["xs"].extend(items)
            return {it: label for it in items}
        return grade_xs

    monkeypatch.setattr(bp, "_build_grader", _build)
    monkeypatch.setattr(bp, "_build_xstest_grader", _build_xs)
    return seen


def _rows(path):
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


# --------------------------------------------------------------------------- #
# 1. every pool condition is scored; non-pool rows are not                      #
# --------------------------------------------------------------------------- #


def test_grade_store_scores_every_pool_condition(tmp_path, monkeypatch):
    cfg, store = _world(tmp_path)
    seen = _fake_graders(monkeypatch)
    bp.cmd_grade_store(_args(), cfg)

    by_prompt = {r["attack_prompt"]: r for r in _rows(store / "bon.jsonl")}
    # the four conditions — none, explicit si, store prompted SI, anti — all
    # scored + marked graded (the none row's score decides requirement_a).
    for p in (P_NONE, P_SI, P_STORE_SI, P_ANTI):
        assert by_prompt[p]["score"] == 0.75, p
        assert by_prompt[p]["graded"] is True, p
    # replay families are part of the store read too
    (replay,) = _rows(store / "pair.jsonl")
    assert replay["score"] == 0.75 and replay["graded"] is True
    # compiled + non-comparison-SI rows: never scored, never sent
    for p in (P_COMPILED, P_OTHER_SI):
        assert by_prompt[p]["score"] is None and by_prompt[p]["graded"] is False
    sent_prompts = {it[2] for it in seen["sr"]}
    assert P_COMPILED not in sent_prompts and P_OTHER_SI not in sent_prompts
    # item shape matches cmd_grade_pos exactly: (family, goal_id, prompt, resp)
    assert (("bon", "g0", P_NONE, R) in seen["sr"]
            and ("pair", "g7", P_REPLAY, R) in seen["sr"])
    # the graded rows are now consumable by _row_to_graded (the manifest path)
    g = bp._row_to_graded(by_prompt[P_NONE], si_ids=SI, threshold=0.5,
                          benign_source="phtest", anti_si_id=ANTI)
    assert g is not None and g.score == 0.75


def test_grade_store_flip_skips_store_si_rows_but_not_benign(tmp_path,
                                                             monkeypatch):
    """With pool.pos_runs.safety_refusal declared (the 2026-08-24 flip) the
    STORE's harmful rows for the store SI — bare `prompted` AND explicit
    si_id — are no longer pool rows, so grade-store must not spend judge
    calls on them; every other harmful condition still grades, and the
    BENIGN side is untouched (the flip is harmful-pos-only, so the benign
    safety_refusal row still gets its XSTest label)."""
    cfg, store = _world(tmp_path)
    cfg["pool"] = {**cfg["pool"],
                   "pos_runs": {**cfg["pool"]["pos_runs"],
                                "safety_refusal": "two_si_pos_safety_refusal"}}
    # an explicit-si store row for the store SI (the pool_sample encoding)
    p_explicit = "SYNTHETIC store prompt juliett harmful-shaped 12340000"
    rows = _rows(store / "bon.jsonl")
    rows.append(_engine_row("bon", "g8", 8, "prompted", p_explicit,
                            si_id=SI[0]))
    (store / "bon.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows))

    seen = _fake_graders(monkeypatch)
    bp.cmd_grade_store(_args(), cfg)

    by_prompt = {r["attack_prompt"]: r for r in _rows(store / "bon.jsonl")}
    sent_prompts = {it[2] for it in seen["sr"]}
    # the store SI's rows: not graded, not sent — the generated run is the
    # authoritative pos source now
    for p in (P_STORE_SI, p_explicit):
        assert by_prompt[p]["score"] is None and not by_prompt[p]["graded"]
        assert p not in sent_prompts
    # everything else still grades exactly as before
    for p in (P_NONE, P_SI, P_ANTI):
        assert by_prompt[p]["score"] == 0.75, p
    # the benign safety_refusal SI row still gets its label (benign untouched)
    b1 = next(r for r in _rows(store / "phtest.jsonl")
              if r["goal_id"] == "b1")
    assert b1["si_id"] == SI[0] and b1["xstest_label"] == "C"


def test_grade_store_labels_benign_si_rows_only(tmp_path, monkeypatch):
    cfg, store = _world(tmp_path)
    seen = _fake_graders(monkeypatch)
    bp.cmd_grade_store(_args(), cfg)

    rows = _rows(store / "phtest.jsonl")
    none_row = next(r for r in rows if r["si_condition"] == "none")
    si_row = next(r for r in rows if r.get("si_id") == SI[0])
    done_row = next(r for r in rows if r.get("si_id") == SI[1])
    assert si_row["xstest_label"] == "C" and si_row["graded"] is True
    # the none row supplies the shared neg — no label, no API call
    assert none_row.get("xstest_label") is None
    assert all(it[1] != "b0" for it in seen["xs"])
    # the already-labelled row was not re-sent (resume)
    assert all(it[1] != "b2" for it in seen["xs"])
    assert done_row["xstest_label"] == "C"
    # benign rows never reach the StrongREJECT grader, and vice versa
    assert all(it[0] == "phtest" for it in seen["xs"])
    assert all(it[0] != "phtest" for it in seen["sr"])


# --------------------------------------------------------------------------- #
# 2. resume                                                                     #
# --------------------------------------------------------------------------- #


def test_grade_store_resume_skips_scored_rows(tmp_path, monkeypatch):
    cfg, store = _world(tmp_path)
    seen = _fake_graders(monkeypatch)
    bp.cmd_grade_store(_args(), cfg)
    # the pre-scored row was never sent
    assert all(it[2] != P_DONE for it in seen["sr"])
    (done,) = [r for r in _rows(store / "bon.jsonl")
               if r["attack_prompt"] == P_DONE]
    assert done["score"] == 0.9   # untouched, not overwritten with 0.75


def test_grade_store_fully_graded_store_builds_no_grader(tmp_path, monkeypatch,
                                                         capsys):
    cfg, store = _world(tmp_path)
    _fake_graders(monkeypatch)
    bp.cmd_grade_store(_args(), cfg)          # first run grades everything

    def _boom(*a, **k):
        raise AssertionError("grader constructed on a fully-graded store")
    monkeypatch.setattr(bp, "_build_grader", _boom)
    monkeypatch.setattr(bp, "_build_xstest_grader", _boom)
    bp.cmd_grade_store(_args(), cfg)          # second run: nothing to do
    assert "nothing to grade" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# 3. atomic rewrite: every other field survives; mode stays 0600                #
# --------------------------------------------------------------------------- #


def test_grade_store_rewrite_preserves_fields_and_mode(tmp_path, monkeypatch):
    cfg, store = _world(tmp_path)
    before = {name: _rows(store / f"{name}.jsonl")
              for name in ("bon", "pair", "phtest")}
    _fake_graders(monkeypatch)
    bp.cmd_grade_store(_args(), cfg)
    for name, olds in before.items():
        path = store / f"{name}.jsonl"
        news = _rows(path)
        assert len(news) == len(olds)         # no row dropped or duplicated
        for old, new in zip(olds, news):      # order preserved
            for key in old:
                if key in ("score", "graded", "xstest_label"):
                    continue
                assert new[key] == old[key], (name, key)
        assert (path.stat().st_mode & 0o777) == 0o600


# --------------------------------------------------------------------------- #
# 4. dry-run: counts only, zero calls, zero writes                              #
# --------------------------------------------------------------------------- #


def test_grade_store_dry_run_reports_counts_and_calls_nothing(
        tmp_path, monkeypatch, capsys):
    cfg, store = _world(tmp_path)

    def _boom(*a, **k):
        raise AssertionError("dry-run must not construct a grader")
    monkeypatch.setattr(bp, "_build_grader", _boom)
    monkeypatch.setattr(bp, "_build_xstest_grader", _boom)
    before = {p: p.read_bytes() for p in store.glob("*.jsonl")}

    bp.cmd_grade_store(_args(dry_run=True), cfg)
    report = json.loads(capsys.readouterr().out)
    # bon: none + explicit-si + store-prompted + anti = 4 (compiled and
    # si_other skipped as non-pool; the pre-scored row resumed past)
    assert report["harmful"]["bon"]["ungraded"] == {
        "none": 1, "si": 2, "anti": 1}
    assert report["harmful"]["bon"]["skipped_non_pool_rows"] == 2
    assert report["harmful"]["pair"]["ungraded_total"] == 1
    assert report["benign"]["phtest"]["unlabelled_si_rows"] == 1
    assert report["benign"]["phtest"]["none_rows_needing_no_label"] == 1
    assert report["totals"]["strongreject_rows_naive"] == 5
    assert report["totals"]["xstest_rows_naive"] == 1
    assert report["totals"]["strongreject_unique_items_naive"] == 5
    # every group here is K=1, so the lazy round 0 IS the naive total
    assert report["totals"]["strongreject_round0_calls"] == 5
    assert report["totals"]["xstest_round0_calls"] == 1
    assert report["totals"]["strongreject_rows_never_needed_now"] == 0
    assert report["totals"]["strongreject_worst_case_calls"] == 5
    # nothing on disk moved a byte
    assert {p: p.read_bytes() for p in store.glob("*.jsonl")} == before


# --------------------------------------------------------------------------- #
# 5. loud failure: an unscored row raises and the file is left untouched        #
# --------------------------------------------------------------------------- #


def test_grade_store_grader_failure_writes_nothing(tmp_path, monkeypatch):
    cfg, store = _world(tmp_path)

    def _build(args, cfg_, cache_dir, cache_name=None):
        def grade(items):
            raise RuntimeError("UNSCORED after retries")   # the JB1 loud raise
        return grade
    monkeypatch.setattr(bp, "_build_grader", _build)
    before = {p: p.read_bytes() for p in store.glob("*.jsonl")}
    with pytest.raises(RuntimeError, match="UNSCORED"):
        bp.cmd_grade_store(_args(), cfg)
    assert {p: p.read_bytes() for p in store.glob("*.jsonl")} == before


def test_grade_store_missing_store_dir_raises(tmp_path):
    cfg, _store = _world(tmp_path)
    with pytest.raises(SystemExit, match="does not exist"):
        bp.cmd_grade_store(_args(store_dir=str(tmp_path / "nope")), cfg)


# --------------------------------------------------------------------------- #
# 6. the score cache lands in the store dir (resume across interruptions)       #
# --------------------------------------------------------------------------- #


def test_grade_store_cache_lives_in_the_store_dir(tmp_path, monkeypatch):
    cfg, store = _world(tmp_path)
    seen = _fake_graders(monkeypatch)
    bp.cmd_grade_store(_args(), cfg)
    assert seen["cache_dir"] == store
    assert seen["cache_name"] == "store_grade_cache.jsonl"
    assert seen["sr_built"] == 1 and seen["xs_built"] == 1   # built once each


# --------------------------------------------------------------------------- #
# 7. lazy rejection-sampling rounds: identical                                  #
#    result to grading all K, at a fraction of the calls                        #
# --------------------------------------------------------------------------- #

THR = 0.5  # cfg["prereg"]["threshold_grid"][1] in the shipped config


def _k_resp(gid, c):
    return f"SYNTHETIC k candidate response {gid} c{c} kkkk{c}{c}{c}{c}"


def _k_world(tmp_path, score_table, *, si=None, k=4):
    """A store whose bon file holds one K-candidate SI group per goal in
    ``score_table`` (``{gid: [score per candidate]}`` — the full table a
    grade-everything run would produce)."""
    cfg, store = _world(tmp_path)
    si = si or SI[1]
    rows = []
    for i, gid in enumerate(sorted(score_table)):
        for c in range(k):
            rows.append(_engine_row("bon", gid, i, "prompted",
                                    f"SYNTHETIC k prompt {gid} 000{i}",
                                    si_id=si, candidate_idx=c,
                                    response=_k_resp(gid, c)))
    p = store / "bon.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    p.chmod(0o600)
    (store / "pair.jsonl").unlink()
    (store / "phtest.jsonl").unlink()
    return cfg, store


def _mapped_grader(monkeypatch, score_by_response):
    """A grader that scores by response text — the deterministic stand-in for
    the full grade-everything score table. Records each round's items."""
    rounds: list[list[tuple]] = []

    def _build(args, cfg, cache_dir, cache_name=None):
        def grade(items):
            rounds.append(list(items))
            return {it: score_by_response[it[3]] for it in items}
        return grade
    monkeypatch.setattr(bp, "_build_grader", _build)
    monkeypatch.setattr(bp, "_build_xstest_grader", _build)
    return rounds


def test_lazy_rounds_match_grade_everything_accepted_index(
        tmp_path, monkeypatch):
    """The one identity that matters: for candidate-0-qualifies, several-fail,
    and ALL-fail groups alike, the smallest qualifying candidate_idx under
    lazy rounds equals the grade-everything answer — and candidates past it
    stay honestly ungraded (score null, not a fabricated value)."""
    table = {"gA": [0.0, 0.9, 0.9, 0.9],     # c0 thwarts -> accept 0
             "gB": [0.9, 0.2, 0.0, 0.0],     # c0 fails -> accept 1 (not 2/3)
             "gC": [0.9, 0.9, 0.9, 0.9]}     # nothing thwarts -> rejected
    score_by_resp = {_k_resp(g, c): s
                     for g, ss in table.items() for c, s in enumerate(ss)}
    cfg, store = _k_world(tmp_path, table)
    rounds = _mapped_grader(monkeypatch, score_by_resp)
    bp.cmd_grade_store(_args(), cfg)

    rows = _rows(store / "bon.jsonl")
    got = {}
    for gid, full in table.items():
        cand = {r["candidate_idx"]: r for r in rows if r["goal_id"] == gid}
        # grade-everything's answer, computed from the FULL score table
        naive = [c for c, s in enumerate(full) if not (s > THR)]
        expect = min(naive) if naive else None
        lazy = [c for c, r in cand.items()
                if r["score"] is not None and not (r["score"] > THR)]
        got[gid] = min(lazy) if lazy else None
        assert got[gid] == expect, gid
        if expect is not None:
            # graded up to and including the accepted index; ungraded past it
            for c, r in cand.items():
                if c <= expect:
                    assert r["score"] == full[c] and r["graded"] is True
                else:
                    assert r["score"] is None and r["graded"] is False
        else:
            assert all(r["score"] is not None for r in cand.values())
    # the call schedule: round0 = one candidate per group; later rounds only
    # the unresolved groups. 3 + 2 + 1 + 1 = 7 calls, not the naive 12.
    assert [len(r) for r in rounds] == [3, 2, 1, 1]


def test_lazy_rounds_anti_direction_accepts_first_compliance(
        tmp_path, monkeypatch):
    """Anti candidates qualify by COMPLIANCE (> thr) — the opposite direction:
    c0 a refusal (fails), c1 a compliance (qualifies), c2 never graded."""
    table = {"gZ": [0.2, 0.9, 0.9]}
    score_by_resp = {_k_resp(g, c): s
                     for g, ss in table.items() for c, s in enumerate(ss)}
    cfg, store = _k_world(tmp_path, table, si=ANTI, k=3)
    _mapped_grader(monkeypatch, score_by_resp)
    bp.cmd_grade_store(_args(), cfg)
    cand = {r["candidate_idx"]: r for r in _rows(store / "bon.jsonl")}
    assert cand[0]["score"] == 0.2               # graded, not a compliance
    assert cand[1]["score"] == 0.9               # the accepted compliance
    assert cand[2]["score"] is None              # never needed


def test_lazy_rounds_resume_from_partial_scores(tmp_path, monkeypatch):
    """Resume mid-round: a pre-scored failing c0 is not re-sent (its group
    continues at c1); a pre-scored QUALIFYING c0 retires its whole group."""
    table = {"gA": [0.0, 0.9, 0.9, 0.9], "gB": [0.9, 0.2, 0.9, 0.9]}
    score_by_resp = {_k_resp(g, c): s
                     for g, ss in table.items() for c, s in enumerate(ss)}
    cfg, store = _k_world(tmp_path, table)
    # pre-score gA c0 (qualifying) and gB c0 (failing) — as a partial run would
    rows = _rows(store / "bon.jsonl")
    for r in rows:
        if r["candidate_idx"] == 0:
            r["score"], r["graded"] = table[r["goal_id"]][0], True
    (store / "bon.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows))
    rounds = _mapped_grader(monkeypatch, score_by_resp)
    bp.cmd_grade_store(_args(), cfg)
    sent = [it for rnd in rounds for it in rnd]
    assert all(it[3] != _k_resp("gA", 0) for it in sent)   # not re-sent
    assert all(it[1] != "gA" for it in sent)               # group resolved
    assert [it[3] for it in sent] == [_k_resp("gB", 1)]    # exactly gB's c1
    cand = {r["candidate_idx"]: r for r in _rows(store / "bon.jsonl")
            if r["goal_id"] == "gB"}
    assert cand[1]["score"] == 0.2 and cand[2]["score"] is None


def test_lazy_rounds_benign_first_c_stops_labelling(tmp_path, monkeypatch):
    """Benign XSTest groups follow the same reduction ('C' iff any candidate
    complied; else candidate 0's own label survives): the first C stops the
    group, an all-refuse group is labelled through to exhaustion."""
    cfg, store = _world(tmp_path)
    label_table = {"bY": ["P", "C", "P"], "bZ": ["P", "I", "P"]}
    rows = []
    for i, gid in enumerate(sorted(label_table)):
        for c in range(3):
            rows.append({**_engine_row("phtest", gid, i, "prompted",
                                       f"SYNTHETIC k benign prompt {gid}",
                                       si_id=SI[0], candidate_idx=c,
                                       response=_k_resp(gid, c)),
                         "side": "benign", "xstest_label": None})
    p = store / "phtest.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    p.chmod(0o600)
    (store / "bon.jsonl").unlink()
    (store / "pair.jsonl").unlink()
    label_by_resp = {_k_resp(g, c): lab
                     for g, ls in label_table.items()
                     for c, lab in enumerate(ls)}

    def _build_xs(args, cfg_, cache_dir):
        def grade_xs(items):
            return {it: label_by_resp[it[3]] for it in items}
        return grade_xs
    monkeypatch.setattr(bp, "_build_xstest_grader", _build_xs)
    bp.cmd_grade_store(_args(), cfg)
    by = {(r["goal_id"], r["candidate_idx"]): r
          for r in _rows(store / "phtest.jsonl")}
    # bY: c1 is the first C -> c2 never labelled; c0's P kept honestly
    assert by[("bY", 0)]["xstest_label"] == "P"
    assert by[("bY", 1)]["xstest_label"] == "C"
    assert by[("bY", 2)]["xstest_label"] is None
    # bZ: no C anywhere -> labelled to exhaustion (candidate 0's label is the
    # reduced fallback, so it must exist)
    assert [by[("bZ", c)]["xstest_label"] for c in range(3)] == ["P", "I", "P"]


def test_dry_run_reports_lazy_calls_for_k_groups(tmp_path, monkeypatch,
                                                 capsys):
    """The dry-run's lazy numbers: a resolved group's remaining candidates are
    counted never-needed; an unresolved group contributes ONE round-0 call."""
    table = {"gA": [0.0, 0.9, 0.9, 0.9], "gB": [0.9, 0.2, 0.9, 0.9]}
    cfg, store = _k_world(tmp_path, table)
    rows = _rows(store / "bon.jsonl")
    for r in rows:                               # pre-score both c0s
        if r["candidate_idx"] == 0:
            r["score"], r["graded"] = table[r["goal_id"]][0], True
    (store / "bon.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows))
    bp.cmd_grade_store(_args(dry_run=True), cfg)
    report = json.loads(capsys.readouterr().out)
    t = report["totals"]
    assert t["strongreject_rows_naive"] == 6        # 3 ungraded per group
    assert t["strongreject_round0_calls"] == 1      # only gB continues
    assert t["strongreject_rows_never_needed_now"] == 3   # gA's c1..c3
    assert t["strongreject_worst_case_calls"] == 3


# --------------------------------------------------------------------------- #
# --allow-unscored on grade-store (2026-08-21). A full-store pass is ~200k+     #
# live calls, and grade-anti-neg hit 5 permanently un-gradeable items in 126k   #
# rows — so without a tolerance a single un-gradeable item discards a whole     #
# round. The tolerance DROPS such items (score stays None, honestly ungraded,   #
# excluded from the pool); it never fabricates a 0. Default 0 keeps the         #
# historical loud refusal.                                                      #
# --------------------------------------------------------------------------- #


def _capture_grader_kwargs(monkeypatch):
    seen = {}

    class _Spy:
        def __init__(self, **kw):
            seen.update(kw)

        def grade(self, items):        # pragma: no cover - never called here
            return {}

        def grade_xstest(self, items):  # pragma: no cover - never called here
            return {}

    import sisteer.eval.attacks.grading_live as gl
    monkeypatch.setattr(gl, "LiveStrongREJECTGrader", _Spy)
    return seen


def test_build_grader_threads_allow_unscored(monkeypatch, tmp_path):
    seen = _capture_grader_kwargs(monkeypatch)
    cfg = {"grader": {"model": "claude-haiku-4-5"}}
    bp._build_grader(_args(allow_unscored=100), cfg, tmp_path)
    assert seen["allow_unscored"] == 100


def test_build_grader_defaults_allow_unscored_to_zero(monkeypatch, tmp_path):
    """Callers predating the flag (and _args' own base dict) carry no
    attribute — the historical loud refusal must survive that, not become an
    accidental tolerance."""
    seen = _capture_grader_kwargs(monkeypatch)
    cfg = {"grader": {"model": "claude-haiku-4-5"}}
    bp._build_grader(_args(), cfg, tmp_path)      # no allow_unscored at all
    assert seen["allow_unscored"] == 0


def test_grade_store_cli_exposes_allow_unscored():
    """Pins the flag onto the grade-store subparser specifically — it was
    added to grade-anti-neg first, and landing it on the wrong subparser
    would fail only at launch time, hours into a run."""
    import subprocess
    out = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "build_two_si_pool.py"),
         "grade-store", "--help"],
        capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    assert "--allow-unscored" in out.stdout


def test_xstest_grader_threads_allow_unscored_too(monkeypatch, tmp_path):
    """grade-store runs TWO judges. Wiring the tolerance into the
    StrongREJECT path alone left the run half-guarded: the SR side rode out
    its un-gradeable items and the XSTest side killed the whole pass anyway,
    ~40 minutes in. Both builders must honour the one flag."""
    seen = _capture_grader_kwargs(monkeypatch)
    cfg = {"grader": {"model": "claude-haiku-4-5"}}
    bp._build_xstest_grader(_args(allow_unscored=100), cfg, tmp_path)
    assert seen["allow_unscored"] == 100


def test_xstest_grader_defaults_allow_unscored_to_zero(monkeypatch, tmp_path):
    seen = _capture_grader_kwargs(monkeypatch)
    cfg = {"grader": {"model": "claude-haiku-4-5"}}
    bp._build_xstest_grader(_args(), cfg, tmp_path)
    assert seen["allow_unscored"] == 0
