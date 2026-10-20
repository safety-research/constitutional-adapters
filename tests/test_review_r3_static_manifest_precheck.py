"""Review round 3, P2-b — the replayable/static served path checks target-decode compatibility BEFORE
its first served_target.json write.

Before: jailbreak_run's replayable branch was a SECOND served_target.json writer that bypassed
`assert_target_decode_compatible`: on the sampled wave path a covered static store whose sidecar was absent was
stamped with the current route and decode before `run_static` examined coverage; matching rows were then skipped
under an unverifiable original served identity, and mismatching rows refused only after the manifest was
overwritten. Now the branch calls the compatibility check before the `started` manifest, for every decode. The
static leg's ONE historical exception is explicit: its served runs wrote no sidecar before the sampled re-run
(2026-09-10), so a covered store with no sidecar whose rows carry no sampler is a legacy GREEDY store and only a
greedy run may resume it (`legacy_leg_without_sidecar=True`, the static leg alone); a sampled request over such a
store refuses (a historical greedy store is never relabelled; a sampled store that lost its sidecar is corruption),
and a greedy request over a sidecar-less store whose rows record a sampler refuses too.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from sisteer.eval.attacks import served_target as st

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import jailbreak_run as jr  # after the scripts/ path insert, as the sibling review tests

CFG_PATH = "configs/sdfbase_full22_llama8b_sampled_t0.yaml"


@dataclass
class _Goal:
    goal_id: str
    text: str


def _target():
    def chat(messages, **kw):
        raise AssertionError("no request may be issued by these tests")
    t = st.ServedChatTarget(chat, virtual_model="petri__base", cell_id="none", max_model_len=4864,
                            route_info={"virtual_model": "petri__base", "intervention_hash": "h", "tt_cell_id": "none"})
    t.evidence = {"steering_bites": {"verified": True}}
    return t


def _args(tmp_path, greedy=False):
    # `greedy` = the driver's explicit --greedy opt-in: since the vendor-preset default (the rule 2026-09-11,
    # jailbreak_run's decode resolution) a config WITHOUT a sampled_decode block is served SAMPLED unless the run opts
    # into argmax, so the historical greedy exception below is exercised by a run that says so
    return SimpleNamespace(attack="pasttense", defense="none", target_engine="served", target_proxy_url="http://127.0.0.1:1/v1",
                           target_routing_table=str(tmp_path / "table.json"), positions=None, scale=1.0, target_max_model_len=4864,
                           private_root=str(tmp_path / "private"), xstest=False, confirmatory=True, goal_ids_file=None,
                           shard=0, nshards=1, trial=0, attacker_pool=None, greedy=greedy)


def _harness(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(st, "build_served_target", lambda **kw: _target())
    monkeypatch.setattr(jr, "_valid_defense_ids", lambda cfg: {"none"})
    monkeypatch.setattr(jr, "_eval_goals", lambda cfg, *, confirmatory: [_Goal("g0", "a"), _Goal("g1", "b")])

    def fake_family(attack, defense, goals, cfg, out_dir, **kw):
        calls.append({"attack": attack, "out_dir": Path(out_dir), **kw})
        return {"n_goals_run": len(goals), "n_goals_total": len(goals), "n_target_forwards_total": 0}
    monkeypatch.setattr(jr, "_run_replayable_family", fake_family)
    return calls


def _cfgs():
    sampled = jr._load_cfg(CFG_PATH)
    greedy = {k: v for k, v in sampled.items() if k != "sampled_decode"}
    greedy["private_run"] = sampled["private_run"].replace("_sampled_t0", "_greedy")
    return sampled, greedy


def _store(tmp_path, cfg, rows, responses=None):
    out = Path(tmp_path / "private") / cfg["private_run"] / "pasttense" / "none" / "shard00"
    out.mkdir(parents=True, exist_ok=True)
    (out / "progress.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    if responses is not None:
        (out / "responses.jsonl").write_text("".join(json.dumps(r) + "\n" for r in responses))
    return out


DONE = [{"goal_id": "g0", "family": "pasttense", "n_variants": 2, "outcome": "done"}]


def test_a_sampled_run_over_a_covered_store_without_a_sidecar_refuses_before_any_manifest_write(monkeypatch, tmp_path):
    calls = _harness(monkeypatch, tmp_path)
    sampled, _ = _cfgs()
    out = _store(tmp_path, sampled, DONE)
    with pytest.raises(st.ServedTargetError, match="ABSENT"):
        jr.cmd_generate(_args(tmp_path), sampled)
    assert not (out / "served_target.json").exists()       # nothing stamped the store with the current decode
    assert calls == []                                       # and the family never ran over the unverifiable coverage


def test_a_sampled_run_over_a_store_recorded_under_another_decode_refuses_with_the_manifest_untouched(monkeypatch, tmp_path):
    calls = _harness(monkeypatch, tmp_path)
    sampled, _ = _cfgs()
    out = _store(tmp_path, sampled, DONE)
    other = {"do_sample": True, "temperature": 1.0, "cap": 4096, "arm": "other", "seed_base": 1, "trial": 0}
    (out / "served_target.json").write_text(json.dumps({"decode": other, "manifest_stage": "final"}))
    before = (out / "served_target.json").read_text()
    with pytest.raises(st.ServedTargetError, match="different target decode"):
        jr.cmd_generate(_args(tmp_path), sampled)
    assert (out / "served_target.json").read_text() == before and calls == []


def test_a_sampled_run_resumes_its_own_recorded_store_and_an_empty_store(monkeypatch, tmp_path):
    calls = _harness(monkeypatch, tmp_path)
    sampled, _ = _cfgs()
    out = _store(tmp_path, sampled, DONE)
    # the block this run will request, written by a previous run of the same config
    block = jr._sampled_decode(sampled, 0)
    from sisteer.eval.attacks.static_common import decode_block
    mine = decode_block(block["decode"], cap=jr._static_max_new_tokens(sampled), seed_base=block["seed_base"], trial=0)
    (out / "served_target.json").write_text(json.dumps({"decode": mine, "manifest_stage": "final"}))
    jr.cmd_generate(_args(tmp_path), sampled)
    assert len(calls) == 1 and calls[0]["out_dir"] == out and calls[0]["decode"] == block["decode"]
    man = json.loads((out / "served_target.json").read_text())
    assert man["manifest_stage"] == "final" and man["decode"] == mine
    # a store with no coverage at all takes any sampler (and gets its started + final manifest)
    fresh = Path(tmp_path / "private") / sampled["private_run"] / "pasttense" / "none" / "shard00"
    (fresh / "progress.jsonl").unlink(); (fresh / "served_target.json").unlink()
    jr.cmd_generate(_args(tmp_path), sampled)
    assert len(calls) == 2 and json.loads((fresh / "served_target.json").read_text())["decode"] == mine


def test_the_static_legs_historical_greedy_exception_is_explicit_and_greedy_only(monkeypatch, tmp_path):
    calls = _harness(monkeypatch, tmp_path)
    _, greedy = _cfgs()
    # a pre-2026-09-10 greedy static store: covered, no sidecar, rows without a sampler -> a greedy run resumes it
    out = _store(tmp_path, greedy, DONE, responses=[{"goal_id": "g0", "sample_idx": 0, "response": "x"},
                                                    {"goal_id": "g0", "sample_idx": 1, "response": "y"}])
    jr.cmd_generate(_args(tmp_path, greedy=True), greedy)          # the explicit argmax opt-in: a GREEDY run over a greedy store
    assert len(calls) == 1 and "decode" not in calls[0]
    man = json.loads((out / "served_target.json").read_text())
    assert "decode" not in man and man["manifest_stage"] == "final"    # stamped as what it is: a greedy store
    # ... but a sidecar-less covered store whose rows RECORD a sampler is a sampled store that lost its sidecar
    out2 = _store(tmp_path / "lost", greedy, DONE, responses=[{"goal_id": "g0", "sample_idx": 0, "response": "x",
                                                               "decode": {"do_sample": True, "temperature": 0.6}, "sampling_seed": 7}])
    with pytest.raises(st.ServedTargetError, match="record a sampler"):
        jr.cmd_generate(_args(tmp_path / "lost", greedy=True), greedy)
    assert not (out2 / "served_target.json").exists() and len(calls) == 1
    # and the exception is the static leg's alone: on a fresh sidecar-less greedy store the default (multi-turn) rule
    # still refuses a greedy resume; only the declared static leg takes it, and only for a greedy request
    out3 = _store(tmp_path / "default", greedy, DONE, responses=[{"goal_id": "g0", "sample_idx": 0, "response": "x"}])
    with pytest.raises(st.ServedTargetError, match="ABSENT"):
        st.assert_target_decode_compatible(out3, None)
    st.assert_target_decode_compatible(out3, None, legacy_leg_without_sidecar=True)
    with pytest.raises(st.ServedTargetError, match="ABSENT") as ei:  # a sampled request never takes the exception
        st.assert_target_decode_compatible(out3, {"do_sample": True, "temperature": 0.6, "cap": 4096, "arm": "vendor_recommended",
                                                  "seed_base": 1, "trial": 0}, legacy_leg_without_sidecar=True)
    assert "never relabelled" in str(ei.value) and not (out3 / "served_target.json").exists()


def test_the_replayable_branch_checks_before_its_started_manifest_write():
    src = (REPO / "scripts/jailbreak_run.py").read_text()
    i0 = src.index("if args.attack in REPLAYABLE_FAMILIES:")
    i1 = src.index("# Tier 1-2: the attacker seams.")
    branch = src[i0:i1]
    # the compare carries the ENGINE too, and the store-wide engine compare
    # precedes it (and the started write) — so a native target onto a lens store refuses before any write
    i_engine = branch.index("assert_store_serving_engine_compatible(out_dir, serving_engine)")
    i_check = branch.index("assert_target_decode_compatible(out_dir, decode_block, legacy_leg_without_sidecar=True,")
    i_write = branch.index('write_target_manifest(out_dir, defense, stage="started",')
    assert i_engine < i_check < i_write
    assert "serving_engine=serving_engine)" in branch[i_check:i_write]
    # the multi-turn branch keeps the default rule (an absent sidecar on a covered store is corruption)
    tail = src[i1:]
    assert ("assert_target_decode_compatible(out_dir, mt_decode_block, serving_engine=serving_engine_of(defense))" in tail
            and "legacy_leg_without_sidecar" not in tail)
