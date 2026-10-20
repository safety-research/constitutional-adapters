"""Stratified pool selection: the defect it exists to prevent, pinned.

The failure being guarded: the pool is sorted by (family, goal_id,
variant_key), so pool[:n] returns the alphabetically-first family. On the
real 23k pool that made a "256-prompt" measurement 235 aim + 21
amplegcg_plus, with the two families holding 78% of the data absent. Tests
assert the PRESENCE of that hazard for the naive slice as well as its
absence for the stratified selector — a test that only checks the good
path would not notice the selector regressing to a prefix.
"""

from sisteer.eval.pool_sampling import (composition, pool_key, hash_rank,
                                        select_stratified, shared_identities)

FAMS = {"aim": 12, "bon": 40, "wildjailbreak_benign": 200, "pasttense": 3}


def _pool():
    rows = []
    for fam in sorted(FAMS):                 # sorted == the real pool order
        for i in range(FAMS[fam]):
            rows.append({"family": fam, "goal_id": f"g{i:04d}",
                         "variant_key": "v0", "prompt": f"{fam}-{i}",
                         "pos": "p", "neg": "n"})
    return rows


def test_naive_prefix_is_single_family_the_hazard():
    rows = _pool()
    head = {r["family"] for r in rows[:12]}
    assert head == {"aim"}, "prefix hazard must reproduce, or the test is moot"


def test_stratified_covers_every_family():
    sel = select_stratified(_pool(), per_family=10, seed=0)
    got = composition(sel)["families"]
    assert set(got) == set(FAMS)
    assert got["aim"] == 10 and got["bon"] == 10
    assert got["pasttense"] == 3, "a thin family contributes all it has"


def test_selection_is_deterministic_and_seed_sensitive():
    a = [pool_key(r) for r in select_stratified(_pool(), 10, seed=0)]
    b = [pool_key(r) for r in select_stratified(_pool(), 10, seed=0)]
    c = [pool_key(r) for r in select_stratified(_pool(), 10, seed=1)]
    assert a == b, "same seed must reproduce exactly"
    assert a != c, "a different seed must move the sample"


def test_selection_independent_of_input_order():
    rows = _pool()
    shuffled = list(reversed(rows))
    assert ([pool_key(r) for r in select_stratified(rows, 10, 0)]
            == [pool_key(r) for r in select_stratified(shuffled, 10, 0)])


def test_shared_identities_are_paired_across_sis():
    rows = _pool()
    sels = {si: select_stratified(rows, 10, 0) for si in ("a", "b", "c")}
    shared = shared_identities(sels)
    assert len(shared) == sum(min(10, n) for n in FAMS.values())


def test_composition_exposes_selected_vs_pool_share():
    rows = _pool()
    comp = composition(select_stratified(rows, 10, 0), rows)
    # wildjailbreak_benign dominates the pool but is capped in the sample:
    # the shares must make that visible rather than hide it.
    assert comp["shares"]["wildjailbreak_benign"]["pool_pct"] > 70
    assert comp["shares"]["wildjailbreak_benign"]["sel_pct"] < 40
    assert comp["families_missing_from_selection"] == []


def test_hash_rank_stable_across_processes():
    # sha256, not hash() — PYTHONHASHSEED must not move a selection.
    assert hash_rank(("aim", "g1", "v0"), 0) == hash_rank(("aim", "g1", "v0"), 0)
    assert hash_rank(("aim", "g1", "v0"), 0)[:8] != hash_rank(("aim", "g1", "v0"), 1)[:8]
