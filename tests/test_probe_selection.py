"""Probe/floor SELECTION: sha256-rank over the pool, never a prefix.

The defect these tests pin (verified on the real pools):
every probe selection site took ``pool[:n]``, and the pools are sorted by
(family, goal_id, variant_key), so "n examples" was in fact a prefix of
the alphabetically FIRST attack family — the recorded 256-prompt floors
were 235 aim + 21 amplegcg_plus while the two families that are 78% of
the pool contributed nothing. Selection is now the repo's seedless
sha-sort convention (``extraction/subsample.py``), the selected keys are
RECORDED next to every estimate, and family labels (when the caller has
them) stratify the selection and yield a per-family table.

MUTATION-VERIFIED: with ``arm_probe_losses``/``train_lockstep`` reverted
to prefix selection (the pre-fix code), ``test_arm_probe_losses_selects_
by_sha_rank_not_prefix`` and ``test_train_lockstep_records_sha_rank_
selection`` go red — run against commit b6168e4 to reproduce.

CPU-only: the tiny_llama fixture, synthetic invented prompts only.
"""

from __future__ import annotations

import pytest
import torch

from sisteer.data.instructions import BY_ID
from sisteer.data.pairs import Pair, pairs_to_distill_examples
from sisteer.extraction.distill import probe_family_table
from sisteer.extraction.lockstep import (
    LockstepArm,
    arm_probe_losses,
    build_lockstep_object,
    default_lockstep_params,
    train_lockstep,
)
from sisteer.extraction.subsample import (
    sha_hex,
    sha_rank_indices,
    stratified_sha_rank_indices,
)

INSTRUCTION = BY_ID["french"]

# A synthetic pool whose FAMILY ORDER mirrors the real defect: rows sorted
# by family, families lexicographically ordered, sizes deliberately
# unequal — a prefix of any n < 25 is 100% family "alpha".
_FAMILY_SIZES = {"alpha": 24, "beta": 8, "gamma": 8}


def _family_pool():
    pairs, families = [], []
    for fam in sorted(_FAMILY_SIZES):
        for i in range(_FAMILY_SIZES[fam]):
            pairs.append(Pair(prompt=f"{fam} question {i} about the sky.",
                              pos=f"Le ciel {fam} {i}",
                              neg=f"The sky {fam} {i}",
                              instruction_id="french"))
            families.append(fam)
    return pairs, families


def _probe_obj(model, tok, pairs):
    obj = build_lockstep_object("vector", model, tok, INSTRUCTION, pairs,
                                layer=1)
    with torch.no_grad():
        obj.trainable[0].fill_(0.05)
    return obj


# --------------------------------------------------------------------- #
# the selection convention itself (pure, no model)                        #
# --------------------------------------------------------------------- #

def test_sha_rank_matches_the_reference_convention():
    """Identical to the diffmean-2x2 ``subsample_indices`` convention:
    rank by sha256 hexdigest, keep top n, return in ORIGINAL order —
    deterministic, seedless, independent of input ordering."""
    keys = [f"prompt {i}" for i in range(50)]
    idx = sha_rank_indices(keys, 10)
    assert len(idx) == 10 and idx == sorted(idx)
    ranked = sorted(range(50), key=lambda i: sha_hex(keys[i]))
    assert set(idx) == set(ranked[:10])
    # permutation-invariance: shuffling the pool selects the same KEYS
    perm = list(reversed(range(50)))
    idx_perm = sha_rank_indices([keys[i] for i in perm], 10)
    assert {keys[perm[i]] for i in idx_perm} == {keys[i] for i in idx}
    # nesting: top-5 is a subset of top-10 (comparable probes)
    assert set(sha_rank_indices(keys, 5)) <= set(idx)
    # n=None / n >= len: everything, in order
    assert sha_rank_indices(keys, None) == list(range(50))
    assert sha_rank_indices(keys, 99) == list(range(50))


def test_sha_rank_spans_families_in_pool_proportion():
    """Over a family-SORTED pool the sha-rank sample matches family
    proportions closely (the verified property: within ~1pp per family at
    n=1000 on the real 23k pool) — while a prefix is 100% one family."""
    sizes = {"alpha": 600, "beta": 250, "gamma": 150}
    keys, fams = [], []
    for fam in sorted(sizes):
        for i in range(sizes[fam]):
            keys.append(f"{fam} prompt {i}")
            fams.append(fam)
    idx = sha_rank_indices(keys, 200)
    got = {f: sum(1 for i in idx if fams[i] == f) / 200 for f in sizes}
    for fam, n in sizes.items():
        assert abs(got[fam] - n / 1000) < 0.06, (fam, got)
    # the prefix "selection" this replaced: one family, in full
    prefix_fams = {fams[i] for i in range(200)}
    assert prefix_fams == {"alpha"}   # the defect, as a fixture invariant


def test_stratified_sha_rank_pairs_across_matched_pools():
    """Within-family sha-rank: min(per_stratum, available) from EVERY
    family, and — because the rank key is the item's own hash — two
    matched pools (same keys, any payloads) select the SAME identities:
    the property cross-SI paired comparisons rest on."""
    keys, fams = [], []
    for fam, n in (("alpha", 9), ("beta", 4), ("gamma", 2)):
        for i in range(n):
            keys.append(f"{fam} prompt {i}")
            fams.append(fam)
    idx = stratified_sha_rank_indices(keys, fams, 3)
    per = {f: sum(1 for i in idx if fams[i] == f) for f in set(fams)}
    assert per == {"alpha": 3, "beta": 3, "gamma": 2}   # thin fam: all of it
    assert idx == sorted(idx)                           # original order
    # matched pool: same keys in a different order -> same selected keys
    perm = sorted(range(len(keys)), key=lambda i: sha_hex(str(i)))
    idx2 = stratified_sha_rank_indices([keys[i] for i in perm],
                                       [fams[i] for i in perm], 3)
    assert {keys[perm[i]] for i in idx2} == {keys[i] for i in idx}
    with pytest.raises(ValueError, match="parallel"):
        stratified_sha_rank_indices(keys, fams[:-1], 3)
    with pytest.raises(ValueError, match="per_stratum"):
        stratified_sha_rank_indices(keys, fams, 0)


# --------------------------------------------------------------------- #
# the selection wired into the probes (tiny model)                        #
# --------------------------------------------------------------------- #

def test_arm_probe_losses_selects_by_sha_rank_not_prefix(tiny_llama):
    """With n_probe below the pool size, the probe's sample is the
    sha256-rank selection over student renderings — spanning every family
    of a family-sorted pool — and the estimate RECORDS the selected keys.
    Against prefix selection (the old ``pool[:n]``) this is red twice
    over: the recorded keys are the prefix's, and they are one family."""
    model, tok = tiny_llama
    pairs, families = _family_pool()
    obj = _probe_obj(model, tok, pairs)
    examples = pairs_to_distill_examples(tok, pairs, INSTRUCTION)

    est = arm_probe_losses(model, tok, obj, examples, max_len=96,
                           batch_size=2, n_probe=10)
    assert est.n_examples == 10
    # the recorded population IS the convention's selection
    keys = [ex.prompt_without for ex in examples]
    expect = sha_rank_indices(keys, 10)
    assert list(est.selected_sha8) == [sha_hex(keys[i])[:8] for i in expect]
    # ...and it spans the families a prefix would miss entirely
    sha2fam = {sha_hex(keys[i])[:8]: families[i] for i in range(len(keys))}
    got_fams = {sha2fam[s] for s in est.selected_sha8}
    assert got_fams == {"alpha", "beta", "gamma"}
    # roughly pool proportion: the dominant family is a majority of the
    # pool but must not be the whole sample (a prefix makes it 100%)
    n_alpha = sum(1 for s in est.selected_sha8 if sha2fam[s] == "alpha")
    assert 2 <= n_alpha <= 9


def test_arm_probe_losses_families_stratify_and_break_down(tiny_llama):
    """``families=`` stratifies the selection (ceil(n_probe/n_families)
    per family, within-family sha-rank) and the estimate carries the
    per-family table — active, noop, ratio, n per family — the numbers
    commensurable with per-family attack-success rates."""
    model, tok = tiny_llama
    pairs, families = _family_pool()
    obj = _probe_obj(model, tok, pairs)
    examples = pairs_to_distill_examples(tok, pairs, INSTRUCTION)

    est = arm_probe_losses(model, tok, obj, examples, max_len=96,
                           batch_size=2, n_probe=6, families=families)
    # 6 over 3 families -> 2 per family, every family present
    assert est.per_family is not None
    assert {f: r["n"] for f, r in est.per_family.items()} == \
        {"alpha": 2, "beta": 2, "gamma": 2}
    assert est.n_examples == 6
    for fam, row in est.per_family.items():
        assert row["ratio"] == pytest.approx(
            row["active"] / max(row["noop"], 1e-6)), fam
    # per-family means recompose the pooled means (equal n per family)
    assert est.active == pytest.approx(
        sum(r["active"] * r["n"] for r in est.per_family.values()) / 6)

    with pytest.raises(ValueError, match="parallel"):
        arm_probe_losses(model, tok, obj, examples, max_len=96,
                         families=families[:-1])


def test_arm_probe_losses_thin_family_reports_its_n_not_an_interval(
        tiny_llama):
    """A family thinner than its stratum budget contributes ALL its rows
    and reports sems as None below two — visible thinness, never a solid-
    looking couple-of-samples number, and never collapsed into the pool."""
    model, tok = tiny_llama
    pairs, families = _family_pool()
    # make gamma thin: keep one row of it
    keep = [i for i, f in enumerate(families) if f != "gamma"] \
        + [i for i, f in enumerate(families) if f == "gamma"][:1]
    keep.sort()
    pairs = [pairs[i] for i in keep]
    families = [families[i] for i in keep]
    obj = _probe_obj(model, tok, pairs)
    examples = pairs_to_distill_examples(tok, pairs, INSTRUCTION)

    est = arm_probe_losses(model, tok, obj, examples, max_len=96,
                           batch_size=2, n_probe=6, families=families)
    gamma = est.per_family["gamma"]
    assert gamma["n"] == 1
    assert gamma["active_sem"] is None and gamma["noop_sem"] is None
    assert est.per_family["alpha"]["n"] == 2
    assert est.per_family["alpha"]["active_sem"] is not None


def test_train_lockstep_records_sha_rank_selection(tiny_llama):
    """The lockstep driver's own startup/final probes select the same way
    (its old probe took the FIRST training batches — the same prefix
    defect one layer up) and the diagnostics record the selection, so the
    sidecar names the population its floor describes."""
    model, tok = tiny_llama
    pairs, _families = _family_pool()
    pairs = pairs[:20] + pairs[24:28]     # keep it small: 20 alpha + 4 beta
    examples = pairs_to_distill_examples(tok, pairs, INSTRUCTION)

    diag: dict = {}
    group = [LockstepArm("vector",
                         build_lockstep_object("vector", model, tok,
                                               INSTRUCTION, pairs, layer=1),
                         default_lockstep_params("vector"))]
    train_lockstep(model, tok, group, examples, epochs=1, batch_size=2,
                   max_len=96, seed=0, diagnostics=diag, n_probe=6)

    d = diag["vector"]
    keys = [ex.prompt_without for ex in examples]
    expect = [sha_hex(keys[i])[:8] for i in sha_rank_indices(keys, 6)]
    assert d["probe_selection"] == "sha256_rank(prompt_without)"
    assert d["probe_selected_sha8"] == expect
    assert d["probe_n"] == 6
    # the prefix would have been all-alpha; the recorded selection is not
    prefix = [sha_hex(k)[:8] for k in keys[:6]]
    assert d["probe_selected_sha8"] != prefix


def test_probe_family_table_is_paired_and_loud():
    table = probe_family_table(["a", "a", "b"], [1.0, 3.0, 5.0],
                               [2.0, 2.0, 4.0])
    assert table["a"]["active"] == pytest.approx(2.0)
    assert table["a"]["noop"] == pytest.approx(2.0)
    assert table["a"]["ratio"] == pytest.approx(1.0)
    assert table["a"]["n"] == 2 and table["b"]["n"] == 1
    assert table["b"]["active_sem"] is None    # thin: n, no interval
    # active-only grouping (a floors-only caller) still works
    only = probe_family_table(["a", "b"], [1.0, 2.0], None)
    assert "noop" not in only["a"] and only["b"]["active"] == 2.0
    with pytest.raises(ValueError, match="parallel"):
        probe_family_table(["a"], [1.0, 2.0], [1.0, 2.0])
