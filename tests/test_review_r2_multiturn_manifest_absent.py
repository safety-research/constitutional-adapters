"""Review round 2, P2 — a multi-turn store with covered goals and NO served_target.json is
corruption, never legacy: `assert_target_decode_compatible` refuses BEFORE jailbreak_run writes a fresh manifest
for the current decode, so a partial copy (or a lost sidecar) can never resume under the vendor preset with
coverage whose sampler / seed policy / trial cannot be established, nor be stamped as the current identity.
Every served run has written the manifest before its first generation (write_target_manifest(stage="started")).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from sisteer.eval.attacks import served_target as st

PRESET_BLOCK = {"do_sample": True, "temperature": 0.6, "top_p": 0.9, "cap": 512, "arm": "vendor_recommended",
                "seed_base": 20260909, "trial": 0, "seed_rule": "multiturn_seed_v1", "multiturn_target_decode": "vendor"}


def _covered(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "progress.jsonl").write_text(json.dumps({"goal_id": "g0", "attack": "crescendo", "outcome": "done"}) + "\n")


def test_covered_goals_without_the_sidecar_refuse_before_any_manifest_write_for_every_decode(tmp_path):
    for decode in (PRESET_BLOCK, None):        # the vendor preset AND the greedy target: neither may resume it
        out = tmp_path / ("preset" if decode else "greedy")
        _covered(out)
        with pytest.raises(st.ServedTargetError, match="ABSENT") as ei:
            st.assert_target_decode_compatible(out, decode)
        assert "partial copy" in str(ei.value) and "cannot be established" in str(ei.value)
        assert not (out / "served_target.json").exists()      # nothing was written over the store


def test_the_call_order_in_jailbreak_run_keeps_the_write_after_the_check():
    src = (Path(__file__).resolve().parents[1] / "scripts/jailbreak_run.py").read_text()
    # the decode compare carries the ENGINE (the store's covered goals are held to
    # the engine that drew them), and the store-wide engine compare runs before it — and before anything is written
    i_engine = src.index("assert_store_serving_engine_compatible(out_dir, serving_engine_of(defense))")
    i_check = src.index("assert_target_decode_compatible(out_dir, mt_decode_block, serving_engine=serving_engine_of(defense))")
    i_write = src.index('write_target_manifest(out_dir, defense, stage="started", decode=mt_decode_block,')
    assert i_engine < i_check < i_write
    # the attacker manifest too -- and since 2026-09-16 it is written AFTER the target decode is resolved (the
    # vendor-preset default's refusals -- an HF target, a served model with no row -- land before any write)
    i_resolve = src.index("mt = _resolve_multiturn_target_decode(cfg, args, served_target=served_target)")
    i_attacker = src.index('_write_engine_manifest(out_dir, cfg, args, args.attack, target_decode_record=mt["record"])')
    assert i_engine < i_resolve < i_attacker < i_check


def test_an_empty_store_and_a_recorded_store_keep_their_behaviour(tmp_path):
    fresh = tmp_path / "fresh"; fresh.mkdir()
    st.assert_target_decode_compatible(fresh, PRESET_BLOCK)                 # no progress at all: any sampler
    (fresh / "progress.jsonl").write_text("\n")                            # an empty progress file: still fresh
    st.assert_target_decode_compatible(fresh, PRESET_BLOCK)
    rec = tmp_path / "recorded"; _covered(rec)
    (rec / "served_target.json").write_text(json.dumps({"decode": PRESET_BLOCK}))
    st.assert_target_decode_compatible(rec, dict(PRESET_BLOCK))            # the same sampler resumes
    with pytest.raises(st.ServedTargetError, match="different target decode"):
        st.assert_target_decode_compatible(rec, {**PRESET_BLOCK, "temperature": 1.0})
    legacy = tmp_path / "legacy"; _covered(legacy)
    (legacy / "served_target.json").write_text(json.dumps({"decode": None, "stage": "final"}))
    with pytest.raises(st.ServedTargetError, match="LEGACY"):              # a recorded legacy manifest: the P1-3 rule
        st.assert_target_decode_compatible(legacy, PRESET_BLOCK)
    st.assert_target_decode_compatible(legacy, None)
