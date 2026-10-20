"""Every configured pool family must be a resolvable attack.

The build asserts "every family present with a non-zero count, or the build
fails" -- but it asserts it AFTER generation. A family named in the config
and absent from the registry therefore costs a whole generation run before
anything says so. These checks are the cheap version of that assertion.
"""
from __future__ import annotations

from pathlib import Path

import yaml

from godmode_standin import godmode_standin  # noqa: F401  (grok420 renders the invented G0DM0D3 stand-in)
from sisteer.eval.attacks import JB_ATTACKS, STATIC_FAMILIES, get_attack

REPO = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((REPO / "configs" / "two_si.yaml").read_text())

# wildjailbreak_harmful is a static CORPUS, not a transform: its rows are
# already-adversarial prompts replayed directly, so it has no JB_ATTACKS
# entry by design.
CORPUS_FAMILIES = {"wildjailbreak_harmful"}


# artprompt, cipherchat, multilingual and drattack gained real
# implementations at the phase-C join (the eval trunk's held-out family
# runners are registered in JB_ATTACKS), so the unimplemented gap is EMPTY.
# The assertion below stays so the gap can never silently REOPEN.
EVAL_ONLY_UNIMPLEMENTED: set[str] = set()


def test_every_GENERATED_family_resolves_to_an_attack():
    """static_families and train_families are what the pool actually
    generates, so an unregistered name there costs a full generation run
    before the build's own assertion notices."""
    for key in ("static_families", "train_families"):
        for fam in CFG["pool"][key]:
            if fam in CORPUS_FAMILIES:
                continue
            assert fam in JB_ATTACKS, f"{key}: {fam!r} is not registered"


def test_the_unimplemented_eval_only_families_are_exactly_the_known_four():
    """Pins the gap so it cannot grow silently, and fails loudly if one of
    them is ever promoted into the generated lists without an implementation."""
    unreg = {f for f in CFG["pool"]["eval_families"]
             if f not in JB_ATTACKS and f not in CORPUS_FAMILIES}
    assert unreg == EVAL_ONLY_UNIMPLEMENTED, unreg
    assert not (EVAL_ONLY_UNIMPLEMENTED
                & set(CFG["pool"]["train_families"])), "promoted without an impl"


def test_the_three_newly_trainable_families_are_static_transforms():
    """codeattack, renellm and grok420 are deterministic goal-slot
    transforms, so they must replay through the static path -- not the
    attacker-model path, which the pool cannot use."""
    for fam in ("codeattack", "renellm", "grok420"):
        assert fam in STATIC_FAMILIES, fam
        assert callable(get_attack(fam)), fam


def test_grok420_yields_both_godmode_variants_deterministically(godmode_standin):
    from sisteer.eval.attacks.grok420 import grok420_variants

    a = grok420_variants("SENTINEL")
    b = grok420_variants("SENTINEL", seed=7)
    assert a == b, "static transforms must not depend on the seed"
    assert [k for k, _ in a] == ["grok420", "hermes_fast"]
    for _key, prompt in a:
        assert "SENTINEL" in prompt
        assert "{QUERY}" not in prompt and "{Z}" not in prompt


def test_the_leakage_gate_is_gone_not_merely_unused():
    """Families are allowed -- encouraged -- to appear on both sides now, so
    a gate that refuses on train/eval overlap would fire on every run. The
    ingested-kind MACHINERY is kept (phase-C join decision: ingest_artifact /
    GENERIC_SPEC / the frozen_demo binding stay), but the refusing gate
    inside it is DEMOTED to a report-only manifest record -- no live
    reference to the refusing checker survives, and nothing can refuse on
    family overlap."""
    import sisteer.eval.attacks.frozen_corpus as fc

    assert hasattr(fc, "ingest_artifact")     # machinery kept (join decision)
    assert hasattr(fc, "GENERIC_SPEC")
    src = Path(fc.__file__).read_text()
    code = "\n".join(l for l in src.splitlines()
                     if not l.lstrip().startswith("#"))
    assert "check_corpus_leakage" not in code  # no live reference
    assert "refuse=True" not in code
    assert "removed_by_directive_2026_08_25" in src  # the report-only record
