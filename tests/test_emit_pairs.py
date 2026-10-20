"""emit-pairs (scripts/build_two_si_pool.py, piece D) — the missing link:
the FROZEN manifest -> the per-SI `{prompt,pos,neg}` jsonl files that
`scripts/train_two_si_objects.py train --pairs` consumes.

CPU-only, offline, no GPU, no harmful text (every prompt/response is an
INVENTED synthetic string). What the suite proves:

- matched regime: both SIs' files carry the SAME ordered prompts and the SAME
  `neg`, differing ONLY in `pos` — and that identity is asserted ON WRITE by
  REUSING `assert_pools_identical_except_pos` (proven by monkeypatching it),
  never re-derived.
- native regime: each SI's own pool is emitted, and the `pairs_meta.json`
  sidecar records the regime + BOTH pool sizes so a later reader cannot
  mistake which comparison the objects came from.
- only GRADED rows qualify as a `pos` (the three-state rule): a
  generated-but-ungraded row is skipped and the skip COUNT is reported.
- deterministic: two runs on the same frozen manifest are byte-identical, in
  the manifest's frozen order.
- `--dry-run` reports counts and writes nothing.
- the manifest tamper guard fires (edited manifest -> refuse), and a store
  that diverged from the frozen manifest fails loudly (never a silently
  different pool).
- containment: output lands mode 0700/0600 and is REFUSED inside the repo.
- contract fit: the emitted jsonl round-trips through the fitter's own
  `train_two_si_objects._load_pairs`.
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
from sisteer.eval.matched_pool import MatchedPoolError  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
SI = ("safety_refusal", "si_constitution")

# Distinctive synthetic text (>= 12 chars) — invented, never real store data.
PROMPTS = {
    0: "SYNTHETIC prompt alpha please do the benign thing 11111111",
    1: "SYNTHETIC prompt bravo please do the benign thing 22222222",
    2: "SYNTHETIC prompt charlie please do the benign thing 33333333",
}
NEG = "SYNTHETIC undefended completion that complied aaaaaaaa"
POS_SAFETY = "SYNTHETIC safety refusal I cannot help with that bbbbbbbb"
POS_CONST = "SYNTHETIC constitution refusal declined per principles cccccccc"
JB = "SYNTHETIC jailbroken completion complied dddddddd"


def _store_row(fam, gid, sample, trial, si_condition, prompt, resp, score,
               *, group="held_in", si_id=None):
    row = {"attack": fam, "goal_id": gid, "si_condition": si_condition,
           "category": "illegal", "category_group": group,
           "attack_prompt": prompt, "response": resp, "score": score,
           "jailbroken": (score is not None and score > 0.5)}
    if sample is not None:
        row["sample_idx"] = sample
        row["trial"] = trial
    if si_id is not None:
        row["si_id"] = si_id
    return row


def _write_labeled_store(store_dir: Path, samples, *, safety_scores=None):
    """bon.jsonl: per sample a none-successful row + a safety(prompted) row.
    ``safety_scores[idx]`` overrides the safety verdict (default 0.0 =
    thwarted; > 0.5 = jailbroken, dropping it from the safety pool)."""
    safety_scores = safety_scores or {}
    store_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for idx in samples:
        s_score = safety_scores.get(idx, 0.0)
        s_resp = JB if s_score > 0.5 else POS_SAFETY
        rows += [_store_row("bon", "g0", idx, 0, "none", PROMPTS[idx], NEG, 0.9),
                 _store_row("bon", "g0", idx, 0, "prompted", PROMPTS[idx],
                            s_resp, s_score)]
    # One held-out goal. The shipped config holds out at the GOAL level
    # (pool.family_disjointness: false), so a store whose rows are all
    # held_in leaves an empty eval goal set -- nothing held out in either
    # dimension, which build_matched_pool refuses.
    rows += [_store_row("bon", "g_held", samples[0], 0, "none",
                        PROMPTS[samples[0]], NEG, 0.9, group="held_out"),
             _store_row("bon", "g_held", samples[0], 0, "prompted",
                        PROMPTS[samples[0]], POS_SAFETY, 0.0,
                        group="held_out")]
    with (store_dir / "bon.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _const_row(sample_idx, prompt, *, score, graded, response=POS_CONST):
    return {"attack": "bon", "goal_id": "g0", "sample_idx": sample_idx,
            "trial": 0, "si_condition": "prompted", "si_id": "si_constitution",
            "category": "illegal", "category_group": "held_in",
            "attack_prompt": prompt, "response": response,
            "score": score, "graded": graded}


def _write_const_run(priv: Path, cfg, rows):
    cdir = priv / cfg["pool"]["constitution_run"]
    cdir.mkdir(parents=True, exist_ok=True)
    with (cdir / "bon.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return cdir


def _cfg(tmp_path, *, min_pool_size=1):
    cfg = yaml.safe_load((REPO / "configs" / "two_si.yaml").read_text())
    # LEGACY two-SI world (the stage-3 pool has its own suite,
    # tests/test_stage3_pool.py): two SIs, bon-only, no benign half, no hard
    # floor, no scoring subtraction.
    cfg["si_ids"] = ["safety_refusal", "si_constitution"]
    cfg["pool"] = {**cfg["pool"], "store_dir": str(tmp_path / "labeled"),
                   "min_pool_size": min_pool_size,
                   "static_families": ["bon"], "train_families": ["bon"],
                   "hard_floor_pairs": 0, "benign_source": None,
                   "benign_fraction": None, "scoring_disjointness": {},
                   # rewind the scale-up amendment keys too (this legacy world
                   # has its own suite; the amendment is exercised in
                   # tests/test_stage3_pool.py)
                   "benign_sources": [], "fit_floor": {},
                   # ... including the 2026-08-24 store-SI flip: this legacy
                   # world serves safety_refusal's pos from the STORE's
                   # prompted rows, so the declaration that would exclude them
                   # is rewound (the flip has its own tests in
                   # tests/test_stage3_pool.py / tests/test_grade_store.py)
                   "pos_runs": {k: v
                                for k, v in (cfg["pool"].get("pos_runs")
                                             or {}).items()
                                if k != "safety_refusal"},
                   "relaxed_stratum": {"enabled": False},
                   "bidpo_benign": {"enabled": False}}
    cfg["pool_manifest"] = str(tmp_path / "pool_manifest.json")
    return cfg


def _matched_world(tmp_path, *, n=2):
    """A store where ``n`` prompts are none-successful and thwarted by BOTH
    SIs (graded) -> the matched regime with pool size ``n``."""
    cfg = _cfg(tmp_path)
    samples = list(range(n))
    _write_labeled_store(tmp_path / "labeled", samples)
    priv = tmp_path / "priv"
    _write_const_run(priv, cfg, [
        _const_row(i, PROMPTS[i], score=0.0, graded=True) for i in samples])
    return cfg, priv


def _freeze_manifest(cfg, priv):
    bp.cmd_manifest(types.SimpleNamespace(private_root=str(priv), out=None), cfg)
    return json.loads(Path(cfg["pool_manifest"]).read_text())


def _emit_args(priv, *, out_dir=None, dry_run=False, manifest=None):
    return types.SimpleNamespace(private_root=str(priv), out_dir=out_dir,
                                 dry_run=dry_run, manifest=manifest)


def _read_pairs(out_dir: Path, si: str):
    path = out_dir / bp.pairs_filename(si)
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


# --------------------------------------------------------------------------- #
# matched regime: identical-except-pos, on write                               #
# --------------------------------------------------------------------------- #


def test_matched_regime_emits_identical_except_pos(tmp_path):
    cfg, priv = _matched_world(tmp_path, n=2)
    m = _freeze_manifest(cfg, priv)
    assert m["regime"] == "matched_intersection"

    out = tmp_path / "pairs"
    bp.cmd_emit_pairs(_emit_args(priv, out_dir=str(out)), cfg)

    a = _read_pairs(out, SI[0])
    b = _read_pairs(out, SI[1])
    assert len(a) == len(b) == 2
    # SAME ordered prompts, SAME neg, ONLY pos differs.
    assert [r["prompt"] for r in a] == [r["prompt"] for r in b]
    assert [r["neg"] for r in a] == [r["neg"] for r in b] == [NEG, NEG]
    assert all(r["pos"] == POS_SAFETY for r in a)
    assert all(r["pos"] == POS_CONST for r in b)
    assert all(r["pos"] != s["pos"] for r, s in zip(a, b))
    # instruction_id labels each file's SI.
    assert {r["instruction_id"] for r in a} == {SI[0]}
    assert {r["instruction_id"] for r in b} == {SI[1]}
    # frozen manifest order preserved.
    frozen = [(e["family"], e["goal_id"], e["variant_key"])
              for e in m["matched_intersection"]["ordered_pairs"]]
    assert [(r["family"], r["goal_id"], r["variant_key"]) for r in a] == frozen


def test_matched_identity_is_asserted_on_write_by_reusing_the_module_assert(
        tmp_path, monkeypatch):
    """emit-pairs must call matched_pool.assert_pools_identical_except_pos —
    the ONE implementation of the five invariants — not a re-derivation.
    Monkeypatching it proves the call happens on the write path."""
    cfg, priv = _matched_world(tmp_path)
    _freeze_manifest(cfg, priv)

    calls = {}

    def _boom(pools):
        calls["pools"] = pools
        raise AssertionError("reused-assert called")

    monkeypatch.setattr(
        "sisteer.eval.matched_pool.assert_pools_identical_except_pos", _boom)
    with pytest.raises(AssertionError, match="reused-assert called"):
        bp.cmd_emit_pairs(_emit_args(priv, out_dir=str(tmp_path / "p")), cfg)
    # ...and it was handed the REBUILT per-SI pools (both SIs present).
    assert sorted(calls["pools"]) == sorted(SI)


def test_emitted_pos_shas_match_the_frozen_manifest(tmp_path):
    """The emitted text is exactly what the manifest froze (hash-resolved,
    not re-derived): every emitted pos hashes to the frozen pos_sha."""
    from sisteer.eval.matched_pool import text_sha

    cfg, priv = _matched_world(tmp_path, n=2)
    m = _freeze_manifest(cfg, priv)
    out = tmp_path / "pairs"
    bp.cmd_emit_pairs(_emit_args(priv, out_dir=str(out)), cfg)
    for si in SI:
        rows = _read_pairs(out, si)
        for r, e in zip(rows, m["matched_intersection"]["ordered_pairs"]):
            assert text_sha(r["pos"]) == e["pos_sha"][si]
            assert text_sha(r["neg"]) == e["neg_sha"]
            assert text_sha(r["prompt"]) == e["prompt_sha"]


# --------------------------------------------------------------------------- #
# native regime: per-SI pools + the sidecar                                   #
# --------------------------------------------------------------------------- #


def _native_world(tmp_path):
    """Safety thwarts {0, 1}; constitution thwarts {0, 2}. min_pool_size=2 >
    |intersection|=1 -> the native fallback, with DIFFERENT per-SI pools."""
    cfg = _cfg(tmp_path, min_pool_size=2)
    # sample 2's safety row is JAILBROKEN -> out of the safety pool
    _write_labeled_store(tmp_path / "labeled", [0, 1, 2],
                         safety_scores={2: 0.9})
    priv = tmp_path / "priv"
    _write_const_run(priv, cfg, [
        _const_row(0, PROMPTS[0], score=0.0, graded=True),
        # sample 1's constitution row is JAILBROKEN -> out of the const pool
        _const_row(1, PROMPTS[1], score=0.9, graded=True, response=JB),
        _const_row(2, PROMPTS[2], score=0.0, graded=True)])
    return cfg, priv


def test_native_regime_emits_per_si_pools_with_sidecar(tmp_path):
    cfg, priv = _native_world(tmp_path)
    m = _freeze_manifest(cfg, priv)
    assert m["regime"] == "native_per_si"

    out = tmp_path / "pairs"
    bp.cmd_emit_pairs(_emit_args(priv, out_dir=str(out)), cfg)

    a = _read_pairs(out, SI[0])
    b = _read_pairs(out, SI[1])
    assert [r["variant_key"] for r in a] == ["0:0", "1:0"]   # safety's own pool
    assert [r["variant_key"] for r in b] == ["0:0", "2:0"]   # constitution's own
    # the sidecar records the regime + BOTH pool sizes so a later reader
    # cannot mistake which comparison the objects came from.
    meta = json.loads((out / bp.PAIRS_META_NAME).read_text())
    assert meta["regime"] == "native_per_si"
    assert meta["pool_sizes"] == {SI[0]: 2, SI[1]: 2}
    assert meta["headline_confounded_by_pool_composition"] is True
    assert meta["manifest_sha256"] == m["manifest_sha256"]


def test_sidecar_written_in_matched_regime_too(tmp_path):
    """The regime is stated prominently in BOTH cases — the sidecar is
    not a fallback-only artifact."""
    cfg, priv = _matched_world(tmp_path)
    m = _freeze_manifest(cfg, priv)
    out = tmp_path / "pairs"
    bp.cmd_emit_pairs(_emit_args(priv, out_dir=str(out)), cfg)
    meta = json.loads((out / bp.PAIRS_META_NAME).read_text())
    assert meta["regime"] == "matched_intersection"
    assert meta["headline_confounded_by_pool_composition"] is False
    assert meta["pool_sizes"] == {SI[0]: 2, SI[1]: 2}
    assert meta["manifest_sha256"] == m["manifest_sha256"]
    # per-file entries let the cluster copy be verified after the push
    for si in SI:
        assert meta["files"][si]["file"] == bp.pairs_filename(si)
        assert meta["files"][si]["n_pairs"] == 2


# --------------------------------------------------------------------------- #
# only graded rows qualify (the three-state rule)                             #
# --------------------------------------------------------------------------- #


def test_ungraded_pos_rows_are_skipped_and_counted(tmp_path, capsys):
    cfg = _cfg(tmp_path)
    _write_labeled_store(tmp_path / "labeled", [0, 1])
    priv = tmp_path / "priv"
    _write_const_run(priv, cfg, [
        _const_row(0, PROMPTS[0], score=0.0, graded=True),
        # generated but STILL AWAITING grade-pos -> not a usable pos
        _const_row(1, PROMPTS[1], score=None, graded=False)])
    _freeze_manifest(cfg, priv)

    out = tmp_path / "pairs"
    bp.cmd_emit_pairs(_emit_args(priv, out_dir=str(out)), cfg)
    # only the graded row's prompt was emitted, for BOTH SIs
    assert [r["variant_key"] for r in _read_pairs(out, SI[0])] == ["0:0"]
    assert [r["variant_key"] for r in _read_pairs(out, SI[1])] == ["0:0"]
    # and the skip is REPORTED (stdout + sidecar), never silent
    assert "1 generated-but-UNGRADED" in capsys.readouterr().out
    meta = json.loads((out / bp.PAIRS_META_NAME).read_text())
    assert meta["n_ungraded_pos_skipped"] == 1


# --------------------------------------------------------------------------- #
# determinism + dry-run                                                         #
# --------------------------------------------------------------------------- #


def test_emit_is_deterministic_and_rerun_safe(tmp_path):
    cfg, priv = _matched_world(tmp_path)
    _freeze_manifest(cfg, priv)
    out = tmp_path / "pairs"
    bp.cmd_emit_pairs(_emit_args(priv, out_dir=str(out)), cfg)
    first = {p.name: p.read_bytes() for p in out.iterdir()}
    bp.cmd_emit_pairs(_emit_args(priv, out_dir=str(out)), cfg)   # run twice
    second = {p.name: p.read_bytes() for p in out.iterdir()}
    assert first == second                       # byte-identical, no dup rows


def test_dry_run_reports_counts_and_writes_nothing(tmp_path, capsys):
    cfg, priv = _matched_world(tmp_path)
    _freeze_manifest(cfg, priv)
    out = tmp_path / "pairs"
    bp.cmd_emit_pairs(_emit_args(priv, out_dir=str(out), dry_run=True), cfg)
    got = capsys.readouterr().out
    assert "REGIME: matched_intersection" in got
    assert f"{SI[0]}=2" in got and f"{SI[1]}=2" in got
    assert not out.exists()                      # nothing written at all


# --------------------------------------------------------------------------- #
# fail-loud guards                                                              #
# --------------------------------------------------------------------------- #


def test_tampered_manifest_is_refused(tmp_path):
    cfg, priv = _matched_world(tmp_path)
    _freeze_manifest(cfg, priv)
    mpath = Path(cfg["pool_manifest"])
    m = json.loads(mpath.read_text())
    m["pool_sizes"][SI[0]] = 999                 # edit without re-hashing
    mpath.write_text(json.dumps(m))
    with pytest.raises(MatchedPoolError, match="hash check"):
        bp.cmd_emit_pairs(_emit_args(priv, out_dir=str(tmp_path / "p")), cfg)


def test_store_diverged_from_frozen_manifest_fails_loudly(tmp_path):
    """A constitution pos regenerated AFTER the freeze no longer hashes to the
    frozen pos_sha — emit must fail naming the pair, never silently emit a
    pool the manifest never froze."""
    cfg, priv = _matched_world(tmp_path)
    _freeze_manifest(cfg, priv)
    _write_const_run(priv, cfg, [                # regenerate with NEW text
        _const_row(i, PROMPTS[i], score=0.0, graded=True,
                   response=POS_CONST + " REGENERATED") for i in (0, 1)])
    with pytest.raises(MatchedPoolError, match="diverged"):
        bp.cmd_emit_pairs(_emit_args(priv, out_dir=str(tmp_path / "p")), cfg)


def test_missing_manifest_says_run_manifest_first(tmp_path):
    cfg, priv = _matched_world(tmp_path)        # nothing frozen yet
    with pytest.raises(SystemExit, match="manifest missing"):
        bp.cmd_emit_pairs(_emit_args(priv, out_dir=str(tmp_path / "p")), cfg)


def test_out_dir_inside_repo_is_refused(tmp_path):
    cfg, priv = _matched_world(tmp_path)
    _freeze_manifest(cfg, priv)
    inside = REPO / "out" / "SYNTHETIC-emit-pairs-test"
    with pytest.raises(SystemExit, match="refusing to write pairs under the repo"):
        bp.cmd_emit_pairs(_emit_args(priv, out_dir=str(inside)), cfg)
    assert not any(inside.glob("*.jsonl")) if inside.exists() else True
    # the guard must not have left the dir behind either way; clean if created
    if inside.exists():
        for p in inside.iterdir():
            p.unlink()
        inside.rmdir()


# --------------------------------------------------------------------------- #
# containment: modes + default location                                         #
# --------------------------------------------------------------------------- #


def test_output_modes_are_private(tmp_path):
    cfg, priv = _matched_world(tmp_path)
    _freeze_manifest(cfg, priv)
    out = tmp_path / "pairs"
    bp.cmd_emit_pairs(_emit_args(priv, out_dir=str(out)), cfg)
    assert (out.stat().st_mode & 0o777) == 0o700
    for si in SI:
        assert ((out / bp.pairs_filename(si)).stat().st_mode & 0o777) == 0o600


def test_default_out_dir_lands_under_the_private_root(tmp_path):
    cfg, priv = _matched_world(tmp_path)
    _freeze_manifest(cfg, priv)
    bp.cmd_emit_pairs(_emit_args(priv), cfg)     # no --out-dir
    out = priv / "two_si_pairs"                  # pool.pairs_run default
    assert (out / bp.pairs_filename(SI[0])).exists()
    assert (out / bp.PAIRS_META_NAME).exists()
    assert (out.stat().st_mode & 0o777) == 0o700


# --------------------------------------------------------------------------- #
# contract fit: the fitter's own loader reads the emitted file                  #
# --------------------------------------------------------------------------- #


def test_emitted_jsonl_round_trips_through_the_fitter_loader(tmp_path):
    import train_two_si_objects as tt

    cfg, priv = _matched_world(tmp_path)
    _freeze_manifest(cfg, priv)
    out = tmp_path / "pairs"
    bp.cmd_emit_pairs(_emit_args(priv, out_dir=str(out)), cfg)

    for si, pos in ((SI[0], POS_SAFETY), (SI[1], POS_CONST)):
        pairs = tt._load_pairs(str(out / bp.pairs_filename(si)))
        assert len(pairs) == 2
        assert all(p.pos == pos and p.neg == NEG for p in pairs)
        assert all(p.instruction_id == si for p in pairs)
        assert [p.prompt for p in pairs] == [PROMPTS[0], PROMPTS[1]]


def test_emit_carries_requirement_a_and_bidpo_gets_the_strict_subset(tmp_path):
    """emit-pairs stamps every row with the manifest's requirement_a label,
    and it round-trips into Pair.requirement_a so the trainer can route bidpo
    to the STRICT (genuine-harmful-compliance neg) subset. The _matched_world
    fixture's none rows all COMPLIED (score 0.9 > 0.5), so both pairs are
    "strict"."""
    import train_two_si_objects as tt

    cfg, priv = _matched_world(tmp_path)
    _freeze_manifest(cfg, priv)
    out = tmp_path / "pairs"
    bp.cmd_emit_pairs(_emit_args(priv, out_dir=str(out)), cfg)

    for si in SI:
        raw = _read_pairs(out, si)
        assert all(r["requirement_a"] == "strict" for r in raw), raw
        pairs = tt._load_pairs(str(out / bp.pairs_filename(si)))
        assert all(p.requirement_a == "strict" for p in pairs)
        # bidpo consumes only the strict subset; here that is the whole pool.
        assert tt.pairs_for_arm("bidpo", pairs) == pairs
        # a non-preference arm is unfiltered regardless of label.
        assert tt.pairs_for_arm("diffmean", pairs) is pairs
