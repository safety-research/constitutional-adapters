"""attack_regime stamped ON the numbers + the adaptive-coverage statement, and
the scope cut (adaptive OFF by default, still selectable).

The distinction this enforces: a defense screening clean on FIXED prompts can
still fall to an adversary that REACTS to it. Prose describing that gets separated
from the table the moment a row is quoted, so the qualifier rides ON the row —
same discipline as engine / template / mechanism / phase / object class.
"""

from __future__ import annotations

import pytest

from sisteer.eval import sweep


def test_engine_stamp_carries_and_validates_attack_regime():
    st = sweep.engine_stamp(target_engine="hf", attack_regime="fixed_prompt")
    assert st["attack_regime"] == "fixed_prompt"
    st2 = sweep.engine_stamp(target_engine="hf", attacker_engine="vllm",
                             attack_regime="adaptive")
    assert st2["attack_regime"] == "adaptive"
    with pytest.raises(ValueError):
        sweep.engine_stamp(target_engine="hf", attack_regime="bogus")


def test_leg_stamps_carry_the_right_regime():
    c = sweep.Condition(label="c", cell="safety_refusal__vector",
                        adaptive=("pair",), attacker_engine="vllm")
    assert c.static_stamp()["attack_regime"] == "fixed_prompt"
    assert c.held_out_stamp()["attack_regime"] == "fixed_prompt"
    assert c.benign_stamp()["attack_regime"] == "fixed_prompt"
    assert c.adaptive_stamp()["attack_regime"] == "adaptive"


def test_attack_regime_of_leg():
    assert sweep.attack_regime_of_leg("adaptive") == "adaptive"
    for leg in ("static", "held_out", "benign"):
        assert sweep.attack_regime_of_leg(leg) == "fixed_prompt"


def test_adaptive_is_off_by_default_but_selectable():
    """Scope cut: a default Condition has NO adaptive leg; the static/held_out/
    benign legs are the default screen. Adaptive is selectable explicitly."""
    c = sweep.Condition(label="c", cell="none")
    assert c.adaptive == ()
    assert c.static and c.held_out and c.benign      # the default screen stands
    c2 = sweep.Condition(label="c2", cell="none", adaptive=("pair", "tap"),
                         attacker_engine="vllm")
    assert c2.adaptive == ("pair", "tap")


def test_load_conditions_adaptive_defaults_off(tmp_path):
    """A TSV with no adaptive column -> adaptive off; an explicit '*' -> all four;
    an explicit list -> that list."""
    p = tmp_path / "conds.tsv"
    p.write_text(
        "off\tnone\n"                       # 2 cols: adaptive missing -> off
        "star\tsafety_refusal__vector\t\tall\t*\t*\t*\thf\tvllm\thf\t\t*\n"
        "sel\tsafety_refusal__vector\t\tall\t*\tpair,tap\t*\thf\tvllm\thf\t\t*\n")
    conds = {c.label: c for c in sweep.load_conditions(p)}
    assert conds["off"].adaptive == ()
    assert set(conds["star"].adaptive) == set(sweep.ADAPTIVE_FAMILIES)
    assert conds["sel"].adaptive == ("pair", "tap")


def _mk_rows():
    """A tiny fixed_prompt static comparison + its baseline replica."""
    st = sweep.engine_stamp(target_engine="hf", served_template_hash="t",
                            attack_regime="fixed_prompt")
    obj = sweep.MetricRow(leg="static", member="bon", condition="cond",
                          metric=0.2, stamp=st)
    base = sweep.MetricRow(leg="static", member="bon", condition="none",
                           metric=0.5, stamp=st, baseline_family="none")
    return [obj, base]


def test_build_asr_table_stamps_regime_on_rows_and_breakout_and_coverage():
    table = sweep.build_comparison_table(
        _mk_rows(), baseline_of=lambda leg, cond: None if cond == "none" else "none")
    assert table and all(c.attack_regime == "fixed_prompt" for c in table)
    asr = sweep.build_asr_table(table, object_class_of=lambda c: "si_distilled")
    # every ASR row carries the regime
    assert all(r["attack_regime"] == "fixed_prompt" for r in asr["rows"])
    # the breakout + the coverage FIELD (not prose) are present
    assert "fixed_prompt" in asr["by_attack_regime"]
    assert asr["attack_regimes_present"] == ["fixed_prompt"]
    cov = asr["attack_regime_coverage"]
    assert "NO adaptive" in cov and "jailbreak lane" in cov


def test_build_manifest_carries_attack_regime():
    """The per-family held-out manifest stamps attack_regime and stays servable
    (the enum value is in the sanitisation allowlist)."""
    from sisteer.eval.attacks.heldout_common import (HeldOutFamily, LeakageReport,
                                                     assert_manifest_safe,
                                                     build_manifest)

    fam = HeldOutFamily(name="artprompt", build_kind="deterministic",
                        variants_fn=lambda g, seed=0: [("v", "prompt")],
                        run_fn=lambda *a, **k: {},
                        provenance={"method": "deterministic"})
    rows = [{"goal_id": "g1", "sample_idx": 0, "variant_name": "v",
             "prompt": "x"}]
    m = build_manifest(fam, rows, seed=0,
                       leakage=LeakageReport(n_checked=1, n_overlap=0,
                                             overlap_goal_ids=(),
                                             training_fingerprint_sha="deadbeef"),
                       attack_regime="fixed_prompt")
    assert m["attack_regime"] == "fixed_prompt"
    assert_manifest_safe(m)   # must not raise — the enum value is allowlisted


def test_write_family_manifest_skipped_leakage_branch_is_servable(tmp_path):
    """REGRESSION (found by the first GPU held-out smoke, slurm 173196): with no
    training fingerprint on the node, write_family_manifest records
    leakage_check.skipped=True LOUDLY — and the manifest must still pass
    sanitisation (the skipped-reason label is an allowlisted CONSTANT). Before the
    fix the reason string was not allowlisted, so every fingerprint-absent node
    crashed the held-out leg instead of recording the skip."""
    from sisteer.eval.attacks.heldout_common import (HeldOutFamily,
                                                     LEAKAGE_SKIPPED_REASON,
                                                     write_family_manifest)

    fam = HeldOutFamily(name="artprompt", build_kind="deterministic",
                        variants_fn=lambda g, seed=0: [("v", f"p-{g.goal_id}")],
                        run_fn=lambda *a, **k: {},
                        provenance={"method": "deterministic"})

    class _G:
        goal_id = "g1"
        text = "synthetic"
        category = "benign_synthetic"

    m = write_family_manifest(fam, [_G()], manifest_path=tmp_path / "m.json",
                              fingerprint=None, attack_regime="fixed_prompt")
    assert m["leakage_check"]["skipped"] is True
    assert m["leakage_check"]["reason"] == LEAKAGE_SKIPPED_REASON
    assert m["attack_regime"] == "fixed_prompt"
    assert (tmp_path / "m.json").exists()
