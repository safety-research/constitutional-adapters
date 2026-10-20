"""Encoding subset — Base64 / FlipAttack / payload-splitting (static transforms
beside bon.py). CPU only, model-free. Covers registry + non-gated, deterministic
sub-variant prompts with DISTINCT keys, resume, record fields incl. variant_key,
and the 0700 dir."""

import base64

from attacks_static_common import assert_static_contract


def test_encoding_registered_and_not_gated():
    from sisteer.eval.attacks import (GATED_ATTACKS, JB_ATTACKS,
                                      SUBENV_ATTACKS, get_attack)

    assert "encoding" in JB_ATTACKS
    assert get_attack("encoding").__name__ == "run_encoding"
    assert "encoding" not in GATED_ATTACKS
    assert "encoding" not in SUBENV_ATTACKS


def test_encoding_subvariants_deterministic_and_distinct():
    from sisteer.eval.attacks.encoding import encoding_variants
    from sisteer.eval.attacks.static_common import variant_key
    from sisteer.eval.jailbreak_data import Goal

    g = Goal("g0", "do the thing", "illegal", "harmbench", "rev")
    v1 = encoding_variants(g, seed=0)
    v2 = encoding_variants(g, seed=0)
    assert v1 == v2                                  # deterministic
    assert [n for n, _ in v1] == ["base64", "flipattack", "payload_split"]
    # sub-variants get distinct positional keys
    keys = [variant_key(i) for i in range(len(v1))]
    assert keys == ["0:0", "1:0", "2:0"] and len(set(keys)) == 3
    # base64 really encodes the goal; flipattack really reverses it
    assert base64.b64encode(b"do the thing").decode() in v1[0][1]
    assert "do the thing"[::-1] in v1[1][1]


def test_encoding_contract():
    from sisteer.eval.attacks.encoding import run_encoding

    assert_static_contract(run_encoding, "encoding",
                           expected_variant_names=["base64", "flipattack",
                                                   "payload_split"])
