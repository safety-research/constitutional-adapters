from sisteer.data import instructions
from sisteer.data.pairs import (
    Pair,
    load_pairs,
    pairs_to_caa_inputs,
    pairs_to_distill_examples,
    pairs_to_pref_examples,
    save_pairs,
)
from sisteer.data.prompts import filter_by_keywords, seed_prompts, split_prompts
from sisteer.eval import checkers


# "external" is not a _REGISTRY entry, and deliberately so: the
# safety SIs (safety_refusal, si_constitution, si_caa) carry
# checker="external" as a SENTINEL meaning "no programmatic checker — scored
# by an external judge suite" (see data/instructions.py, which notes that an
# accidental checkers.check("external", ...) must fail loudly). Exempting it
# here is the same fix the realign lineage carries; without it, registering
# any external-checked SI breaks this sweep.
DEFERRED_CHECKERS = {"external"}


def test_registry_checkers_exist():
    for instr in instructions.REGISTRY:
        if instr.checker in DEFERRED_CHECKERS:
            continue
        fn = checkers.get_checker(instr.checker)
        assert callable(fn)
        # checker runs on arbitrary text without raising
        fn("Some sample response text.", **instr.checker_kwargs)


def test_registry_categories():
    cats = {i.category for i in instructions.REGISTRY}
    assert {"style", "format", "conditional", "persona"} <= cats
    for instr in instructions.REGISTRY:
        if instr.category == "conditional":
            assert instr.trigger_set is not None


def test_split_deterministic():
    prompts = seed_prompts("general")
    fit1, held1 = split_prompts(prompts)
    fit2, held2 = split_prompts(list(reversed(prompts)))
    assert set(fit1) == set(fit2)
    assert set(held1) == set(held2)
    assert not (set(fit1) & set(held1))
    assert fit1 and held1


def test_filter_by_keywords():
    prompts = ["Write a Python function.", "What causes rain?"]
    assert filter_by_keywords(prompts, ["python"]) == ["Write a Python function."]
    assert filter_by_keywords(prompts, ["python"], include=False) == [
        "What causes rain?"
    ]


def test_pair_roundtrip(tmp_path):
    pairs = [
        Pair(prompt="p1", pos="oui", neg="yes", instruction_id="french"),
        Pair(prompt="p2", pos="non", neg="no", instruction_id="french"),
    ]
    save_pairs(pairs, tmp_path / "pairs.jsonl")
    loaded = load_pairs(tmp_path / "pairs.jsonl")
    assert loaded == pairs


def test_save_pairs_atomic_publish(tmp_path):
    """save_pairs must publish atomically (tmp + os.replace): pairs.jsonl
    is a shared artifact that concurrent cells resume from by EXISTENCE,
    so a crash mid-write must leave either the old complete file or none —
    never a truncated file that resume then trusts."""
    import pytest

    good = [Pair(prompt="p1", pos="oui", neg="yes", instruction_id="french")]
    path = tmp_path / "pairs.jsonl"
    save_pairs(good, path)

    # a crash mid-serialization (second row unserializable) must not
    # clobber or truncate the published file
    crashing = [
        Pair(prompt="p2", pos="non", neg="no", instruction_id="french"),
        Pair(prompt=object(), pos="", neg="", instruction_id="french"),
    ]
    with pytest.raises(TypeError):
        save_pairs(crashing, path)
    assert load_pairs(path) == good  # old content intact, not truncated
    assert not list(tmp_path.glob("*.tmp"))  # no temp debris left behind


def test_pair_conversions(tiny_llama):
    _, tok = tiny_llama
    instr = instructions.get("french")
    pairs = [Pair(prompt="Hello?", pos="Bonjour.", neg="Hello.",
                  instruction_id="french")]
    prompts, pos, neg = pairs_to_caa_inputs(tok, pairs)
    assert len(prompts) == 1 and pos == ["Bonjour."] and neg == ["Hello."]
    prefs = pairs_to_pref_examples(tok, pairs)
    assert prefs[0].chosen == "Bonjour." and prefs[0].rejected == "Hello."
    dist = pairs_to_distill_examples(tok, pairs, instr)
    assert instr.text in dist[0].prompt_with_si
    assert instr.text not in dist[0].prompt_without
    assert dist[0].completion == "Bonjour."
