"""A requested attack family that yields zero prompts must REFUSE.

2026-08-26: renellm's frozen corpus was unreachable in one job of a four-stage
DAG (SISTEER_HELDOUT_CORPUS_DIR set in three scripts, absent from the fourth).
harmful_attack_prompts silently emitted nothing for it. The inventory came out
at 21,452 prompts instead of 22,343, still listing renellm in its provenance
key as requested, and every downstream stage would have trained on five
families while its own metadata claimed six.

build_matched_pool does assert against an unrealised train family -- but only
after the generation hours are spent. The sub-env path already refused a
zero-count family ("Refusing to emit a zero-count family"); the static path
did not, and that asymmetry is the whole bug.
"""
import pytest

from sisteer.sampling import sources


class _Goal:
    def __init__(self, i):
        self.goal_id, self.dataset, self.category = f"g{i}", "jbb", None


def test_zero_count_static_family_raises(monkeypatch):
    monkeypatch.setattr(sources, "_static_variants_fn",
                        lambda fam: (lambda g, seed: []))
    with pytest.raises(RuntimeError, match="ZERO prompts"):
        sources.harmful_attack_prompts([_Goal(0)], ["renellm"])


def test_the_message_names_the_actual_cause():
    """A refusal that does not say SISTEER_HELDOUT_CORPUS_DIR costs the next
    person the same hour it cost me."""
    import inspect
    src = inspect.getsource(sources.harmful_attack_prompts)
    assert "SISTEER_HELDOUT_CORPUS_DIR" in src


def test_a_productive_family_is_untouched(monkeypatch):
    monkeypatch.setattr(sources, "_static_variants_fn",
                        lambda fam: (lambda g, seed: [("v0", "text")]))
    out = sources.harmful_attack_prompts([_Goal(0), _Goal(1)], ["pasttense"])
    assert len(out) == 2
    assert {p.family for p in out} == {"pasttense"}


def test_one_dead_family_among_live_ones_still_raises(monkeypatch):
    def fake(fam):
        return (lambda g, seed: [] if fam == "renellm" else [("v0", "t")])
    monkeypatch.setattr(sources, "_static_variants_fn", fake)
    with pytest.raises(RuntimeError, match="renellm"):
        sources.harmful_attack_prompts([_Goal(0)], ["pasttense", "renellm"])


# --- the harder case: a PARTIAL family --------------------------------------
# 777 renellm prompts passed the zero-count guard because 777 > 0. The cluster
# corpus held 259 goals to the grading host's 297 -- a stale copy, identical in name
# and present on disk. Presence checks, non-empty checks, and the pool's own
# min_family_goal_coverage of 0.5 all passed it.

def test_partial_family_coverage_is_refused():
    import pytest
    from sisteer.sampling import sources as S
    goals = [_Goal(i) for i in range(100)]

    def fake(fam):
        def fn(g, seed):
            return [("v0", "t")] if int(g.goal_id[1:]) < 80 else []
        return fn
    orig = S._static_variants_fn
    S._static_variants_fn = fake
    try:
        with pytest.raises(RuntimeError, match="80/100"):
            S.harmful_attack_prompts(goals, ["renellm"])
    finally:
        S._static_variants_fn = orig


def test_coverage_just_above_the_floor_is_allowed():
    from sisteer.sampling import sources as S
    goals = [_Goal(i) for i in range(100)]

    def fake(fam):
        def fn(g, seed):
            return [("v0", "t")] if int(g.goal_id[1:]) < 96 else []
        return fn
    orig = S._static_variants_fn
    S._static_variants_fn = fake
    try:
        out = S.harmful_attack_prompts(goals, ["renellm"])
        assert len({p.goal_id for p in out}) == 96
    finally:
        S._static_variants_fn = orig


def test_the_refusal_tells_you_to_check_the_sha_not_the_presence():
    import inspect
    from sisteer.sampling import sources as S
    src = inspect.getsource(S.harmful_attack_prompts)
    assert "sha256" in src and "SISTEER_HELDOUT_CORPUS_DIR" in src
