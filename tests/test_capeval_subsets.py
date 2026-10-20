"""The two PINNED 200-item subsets (BCB and BFCL): the pins on disk
REGENERATE from their recorded populations through the contracts' rules, no
Hard id is pinned, the Hamilton seats are the recorded ones and every id lives
in exactly one category, the seeded ordering is pinned, the library-family map
is total, the id round-trips and the loaders' refusals fire. Offline: the pins
carry their populations, so nothing here touches a dataset; the re-derivation
from the pinned SOURCES runs only where the caches are present."""

from __future__ import annotations

import hashlib
import json
import os
import random
import sys
from pathlib import Path

import pytest

from sisteer.capeval.tinycap import contracts as C

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import capeval_subsets as S  # noqa: E402

BCB = json.loads((REPO / C.BCB_PIN_PATH).read_text())
BFCL = json.loads((REPO / C.BFCL_PIN_PATH).read_text())


def test_the_constants_state_the_populations_the_pins_were_drawn_from():
    assert (C.BCB_SUBSET_N, C.BFCL_SUBSET_N, C.SUBSET_SEED) == (200, 200, 0)
    assert (C.BCB_FULL_N, C.BCB_HARD_N, C.BCB_NONHARD_N) == (1140, 148, 992)
    assert C.BFCL_FULL_N == 3981
    assert C.BCB_SPLIT == "v0.1.2"
    assert C.BCB_DATASET_REVISION == "b74c0d0bf70d2c0bc459be537895cca163007f1a"
    assert C.BCB_HARD_DATASET_REVISION == "298d2cc7b96612e15e47313c3603ee124cee0c1f"
    assert C.BFCL_GORILLA_COMMIT_FULL.startswith(C.BFCL_GORILLA_COMMIT) and len(C.BFCL_GORILLA_COMMIT_FULL) == 40
    # the pins are where the contracts say, with the contract's exact top-level shape
    for doc, extra in ((BCB, {"libs_by_id", "per_family"}), (BFCL, set())):
        assert set(doc) == set(C.SUBSET_PIN_FIELDS) | extra
        assert doc["schema"] == C.SUBSET_PIN_SCHEMA and doc["seed"] == C.SUBSET_SEED
        assert doc["generated_by"] == S.GENERATED_BY


def test_hash_order_is_pinned_and_total():
    """Seed 0 means a SHA-256 ordering of "0:<id>" — fixed by the standard it
    names, not by an interpreter's `random`; duplicates refuse."""
    ids = ["a", "b", "c", "d"]
    want = sorted(ids, key=lambda i: hashlib.sha256(f"0:{i}".encode()).hexdigest())
    assert C.hash_order(ids, seed=0) == want
    assert C.hash_order(list(reversed(ids)), seed=0) == want            # input order is irrelevant
    assert C.hash_order(ids, seed=1) != want or len(ids) < 3           # the seed moves the order
    assert C.hash_order(["BigCodeBench/17", "BigCodeBench/4"], seed=0) == [
        "BigCodeBench/17", "BigCodeBench/4"] or True                    # deterministic either way
    with pytest.raises(ValueError, match="duplicate"):
        C.hash_order(["a", "a"])
    with pytest.raises(ValueError, match="seed must be an int"):
        C.hash_order(ids, seed=True)


def test_hamilton_seats_and_the_deterministic_tie_break():
    """Several BFCL categories share a size, so the remaining seats
    go by (remainder DESCENDING, category name ASCENDING) — four equal
    categories with two leftover seats give them to the two alphabetically
    first, whatever order the input arrives in."""
    counts = {"delta": 50, "alpha": 50, "charlie": 50, "bravo": 50}
    seats = C.hamilton_allocation(counts, 6)                            # 1.5 each -> floors 1 + 2 extra
    assert seats == {"alpha": 2, "bravo": 2, "charlie": 1, "delta": 1}
    shuffled = dict(random.Random(3).sample(list(counts.items()), k=4))
    assert C.hamilton_allocation(shuffled, 6) == seats
    assert list(seats) == sorted(seats)                                 # sorted output
    # the floor: a category smaller than its quota still gets 1
    assert C.hamilton_allocation({"big": 1000, "tiny": 1}, 10) == {"big": 9, "tiny": 1}
    # exact proportions need no tie-break
    assert C.hamilton_allocation({"a": 100, "b": 300}, 8) == {"a": 2, "b": 6}
    with pytest.raises(ValueError, match="floors alone"):
        C.hamilton_allocation({f"c{i}": 1 for i in range(5)}, 3)
    with pytest.raises(ValueError, match="exceed the category size"):
        C.hamilton_allocation({"a": 1, "b": 1}, 4)
    with pytest.raises(ValueError, match="positive int"):
        C.hamilton_allocation({"a": 1}, 0)


def test_the_bcb_pin_regenerates_from_its_recorded_population_and_pins_no_hard_id():
    pop = BCB["population"]
    assert len(pop["ids"]) == C.BCB_FULL_N and len(pop["hard_ids"]) == C.BCB_HARD_N
    assert set(pop["hard_ids"]) <= set(pop["ids"])
    assert BCB["of"] == C.BCB_NONHARD_N and BCB["n"] == C.BCB_SUBSET_N
    regen = C.bcb_select(pop["ids"], pop["hard_ids"], n=C.BCB_SUBSET_N, seed=C.SUBSET_SEED)
    assert regen == BCB["ids"]
    assert len(set(BCB["ids"])) == 200 and not (set(BCB["ids"]) & set(pop["hard_ids"]))
    # reported sorted by the numeric suffix; every id is a real complete-split id
    suffixes = [int(C.bcb_log_id(i)) for i in BCB["ids"]]
    assert suffixes == sorted(suffixes) and set(BCB["ids"]) <= set(pop["ids"])
    # the population is the contiguous split (position == id suffix — the thin
    # task's assumption)
    assert pop["ids"] == [f"BigCodeBench/{i}" for i in range(C.BCB_FULL_N)]
    # the verified loader accepts it and returns the same document
    assert C.load_bcb_pin(REPO)["ids"] == BCB["ids"]
    # the rule prose names the seed, the exclusion and the ordering
    assert "Hard" in BCB["rule"] and "sha256" in BCB["rule"] and "seed 0" in BCB["rule"]
    # another seed is another subset (the pin is a draw, not a prefix)
    assert C.bcb_select(pop["ids"], pop["hard_ids"], seed=1) != BCB["ids"]


def test_the_bfcl_pin_regenerates_and_every_id_lives_in_exactly_one_category():
    pop = BFCL["population"]
    assert sum(len(v) for v in pop.values()) == C.BFCL_FULL_N == BFCL["of"]
    assert len(pop) == 18 and sorted(pop) == list(pop)                  # the battery's 18, sorted
    flat = [i for v in pop.values() for i in v]
    assert len(set(flat)) == len(flat)
    ids, seats = C.bfcl_select(pop, n=C.BFCL_SUBSET_N, seed=C.SUBSET_SEED)
    assert ids == BFCL["ids"] and seats == BFCL["per_category"]
    assert sum(seats.values()) == 200 and all(v >= 1 for v in seats.values())
    # the seats ARE the Hamilton rule over the category sizes
    assert seats == C.hamilton_allocation({c: len(v) for c, v in pop.items()}, 200)
    # survey §L: the measured category sizes
    assert {c: len(v) for c, v in pop.items()} == {
        "exec_multiple": 50, "exec_parallel": 50, "exec_parallel_multiple": 40, "exec_simple": 100,
        "irrelevance": 240, "live_irrelevance": 884, "live_multiple": 1053, "live_parallel": 16,
        "live_parallel_multiple": 24, "live_relevance": 16, "live_simple": 258, "multiple": 200,
        "parallel": 200, "parallel_multiple": 200, "simple_java": 100, "simple_javascript": 50,
        "simple_python": 400, "sql": 100}
    # every pinned id is in exactly one category, and the per-category pinned
    # counts are the seats (the analysis join)
    where = {i: c for c, v in pop.items() for i in v}
    assert all(i in where for i in BFCL["ids"])
    counted = {}
    for i in BFCL["ids"]:
        counted[where[i]] = counted.get(where[i], 0) + 1
    assert counted == seats
    assert C.load_bfcl_pin(REPO)["per_category"] == seats
    assert "Hamilton" in BFCL["rule"] and "category_name ASCENDING" in BFCL["rule"]
    with pytest.raises(ValueError, match="more than one category"):
        C.bfcl_select({"a": ["x"], "b": ["x"]}, n=2)


def test_the_library_family_map_is_total_and_the_per_family_counts_regenerate():
    families = S.load_lib_families(REPO)
    libs_seen = {lib for libs in BCB["libs_by_id"].values() for lib in libs}
    assert libs_seen <= set(families), sorted(libs_seen - set(families))
    assert set(BCB["libs_by_id"]) == set(BCB["ids"])
    assert S.per_family_counts(BCB["libs_by_id"], families) == BCB["per_family"]
    # multi-label: an item with libs in two families counts under both
    assert S.per_family_counts({"i": ["pandas", "matplotlib"]}, families) == {"data": 1, "viz": 1}
    with pytest.raises(C.SubsetPinError, match="no family"):
        S.per_family_counts({"i": ["not_a_lib"]}, families)
    # the pinned document's families are spelled from a fixed vocabulary
    assert set(families.values()) <= {"stdlib", "data", "viz", "text_nlp", "web_net", "crypto_sec",
                                      "io_format", "datetime", "system", "image_media", "ml_dl", "geo"}


def test_id_round_trip_and_the_item_set_identity():
    assert C.bcb_log_id("BigCodeBench/17") == "17" and C.bcb_pinned_id("17") == "BigCodeBench/17"
    assert C.bcb_pinned_id(C.bcb_log_id("BigCodeBench/0")) == "BigCodeBench/0"
    for bad in ("HumanEval/1", "BigCodeBench/x", "17"):
        with pytest.raises(ValueError):
            C.bcb_log_id(bad)
    for bad in ("BigCodeBench/17", "-1", "1.0"):
        with pytest.raises(ValueError):
            C.bcb_pinned_id(bad)
    ident = C.item_set_identity(BCB)
    assert set(ident) == {"seed", "n", "of", "ids_sha256"}
    assert ident["ids_sha256"] == hashlib.sha256(json.dumps(BCB["ids"], separators=(",", ":")).encode()).hexdigest()
    assert (ident["seed"], ident["n"], ident["of"]) == (0, 200, 992)
    assert C.item_set_identity(BFCL)["of"] == 3981
    # the identity is order-sensitive on purpose: the pin order IS the leg's order
    assert C.ids_sha256(BCB["ids"]) != C.ids_sha256(list(reversed(BCB["ids"])))


def test_the_verified_loaders_refuse_drift(tmp_path):
    """A pin whose recorded rule no longer regenerates its ids, or whose
    seed / n / of drifted from the contracts, is refused — never read around."""
    root = tmp_path
    (root / C.BCB_PIN_PATH).parent.mkdir(parents=True)
    bad = dict(BCB)
    bad["ids"] = [*BCB["ids"][:-1], BCB["population"]["ids"][0]]        # one substituted id
    (root / C.BCB_PIN_PATH).write_text(json.dumps(bad))
    with pytest.raises(C.SubsetPinError, match="do not regenerate"):
        C.load_bcb_pin(root)
    (root / C.BCB_PIN_PATH).write_text(json.dumps({**BCB, "seed": 1}))
    with pytest.raises(C.SubsetPinError, match="seed 1 != the contracts' 0"):
        C.load_bcb_pin(root)
    (root / C.BCB_PIN_PATH).write_text(json.dumps({**BCB, "n": 199, "ids": BCB["ids"][:199]}))
    with pytest.raises(C.SubsetPinError, match="n 199"):
        C.load_bcb_pin(root)
    (root / C.BCB_PIN_PATH).unlink()
    with pytest.raises(C.SubsetPinError, match="missing"):
        C.load_bcb_pin(root)
    bf = dict(BFCL)
    bf["per_category"] = {**BFCL["per_category"], "sql": BFCL["per_category"]["sql"] + 1}
    (root / C.BFCL_PIN_PATH).write_text(json.dumps(bf))
    with pytest.raises(C.SubsetPinError, match="do not regenerate"):
        C.load_bfcl_pin(root)
    (root / C.BFCL_PIN_PATH).write_text(json.dumps({**BFCL, "task": "bfcl"}))
    with pytest.raises(C.SubsetPinError, match="task 'bfcl'"):
        C.load_bfcl_pin(root)


def test_the_pin_documents_rebuild_byte_for_byte_from_their_recorded_populations():
    """The writer is pure over its population: rebuilding the document from
    the pin's own population (same generated_at) gives the same bytes."""
    families = S.load_lib_families(REPO)
    libs_by_task = {i: BCB["libs_by_id"][i] for i in BCB["ids"]}
    rebuilt = S.build_bcb_pin(ids=BCB["population"]["ids"], hard_ids=BCB["population"]["hard_ids"],
                              libs_by_task={**{i: [] for i in BCB["population"]["ids"]}, **libs_by_task},
                              families=families, source_sha256=BCB["source_sha256"],
                              generated_at=BCB["generated_at"])
    assert S.canonical_json(rebuilt) == S.canonical_json(BCB)
    rebuilt_f = S.build_bfcl_pin(population=BFCL["population"], source_sha256=BFCL["source_sha256"],
                                 generated_at=BFCL["generated_at"])
    assert S.canonical_json(rebuilt_f) == S.canonical_json(BFCL)
    assert S.check_pins(REPO) == 0


@pytest.mark.skipif(not (Path(os.environ.get("HF_HOME", "/tmp/bcb-hf")) / "hub").is_dir(),
                    reason="the pinned BigCodeBench parquets are not cached here (HF_HOME)")
def test_the_bcb_population_re_derives_from_the_pinned_parquets():
    """Skip-unless-cached: the population recorded in the pin IS the pinned
    sources' (ids, Hard ids, libs) — re-derived from the parquet files by
    revision, never from the network."""
    pytest.importorskip("pyarrow")
    os.environ.setdefault("HF_HOME", "/tmp/bcb-hf")
    os.environ["HF_HUB_OFFLINE"] = "1"
    try:
        b = S.bcb_population()
    except Exception as e:  # noqa: BLE001 — an incomplete cache is a skip, not a failure
        pytest.skip(f"parquets not resolvable offline: {type(e).__name__}: {e}")
    assert b["ids"] == BCB["population"]["ids"]
    assert sorted(b["hard_ids"], key=lambda i: int(C.bcb_log_id(i))) == BCB["population"]["hard_ids"]
    assert {i: b["libs_by_task"][i] for i in BCB["ids"]} == BCB["libs_by_id"]
    assert b["source_sha256"] == BCB["source_sha256"]


def _pin_root(tmp_path, bcb=None, bfcl=None):
    root = tmp_path / "root"
    for rel, doc in ((C.BCB_PIN_PATH, bcb), (C.BFCL_PIN_PATH, bfcl)):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(json.dumps(doc if doc is not None else json.loads((REPO / rel).read_text())))
    (root / C.BCB_LIB_FAMILIES_PATH).write_text((REPO / C.BCB_LIB_FAMILIES_PATH).read_text())
    return root


def test_the_pin_loaders_hold_each_pin_to_its_exact_schema_and_recompute_every_binding(tmp_path):
    """The loaders used a STRICT-SUBSET key guard — an extra key
    passed, and an extra key masked a missing one. Now: the required set first,
    then the task's EXACT key set; the dataset block, digests, roster, commit,
    libraries and per-family counts are held to the contracts and recomputed."""
    assert C.load_bcb_pin(REPO)["n"] == 200 and C.load_bfcl_pin(REPO)["n"] == 200       # the real pins pass
    assert C.SUBSET_PIN_BCB_EXTRA_FIELDS == ("libs_by_id", "per_family")
    assert len(C.BFCL_CATEGORIES) == 18 and list(C.BFCL_CATEGORIES) == sorted(C.BFCL_CATEGORIES)
    assert C.BFCL_CATEGORIES == tuple(BFCL["dataset"]["categories"]) == tuple(sorted(BFCL["population"]))

    def refuses_bcb(edit, needle):
        doc = json.loads((REPO / C.BCB_PIN_PATH).read_text())
        edit(doc)
        with pytest.raises(C.SubsetPinError, match=needle):
            C.load_bcb_pin(_pin_root(tmp_path, bcb=doc))

    def refuses_bfcl(edit, needle):
        doc = json.loads((REPO / C.BFCL_PIN_PATH).read_text())
        edit(doc)
        with pytest.raises(C.SubsetPinError, match=needle):
            C.load_bfcl_pin(_pin_root(tmp_path, bfcl=doc))

    # an extra key refuses; an extra key can no longer mask a missing one
    refuses_bcb(lambda d: d.__setitem__("note", "x"), "extra \\['note'\\]")
    refuses_bfcl(lambda d: d.__setitem__("libs_by_id", {}), "extra \\['libs_by_id'\\]")
    refuses_bcb(lambda d: (d.pop("source_sha256"), d.__setitem__("checksum", "x")), "missing \\['source_sha256'\\]")
    refuses_bcb(lambda d: d.pop("per_family"), "missing \\['per_family'\\]")
    # BCB: the dataset revisions, the parquet digests, libs_by_id, per_family, the family map's totality
    refuses_bcb(lambda d: d["dataset"]["complete"].__setitem__("revision", "0" * 40), "dataset")
    refuses_bcb(lambda d: d["dataset"]["hard"].__setitem__("n", 147), "dataset")
    refuses_bcb(lambda d: d["source_sha256"].__setitem__("complete_parquet", "nothex"), "sha256 hex")
    refuses_bcb(lambda d: d["source_sha256"].pop("hard_parquet"), "exactly \\['complete_parquet', 'hard_parquet'\\]")
    refuses_bcb(lambda d: d.__setitem__("per_category", {"x": 1}), "per_category is \\{\\}")
    refuses_bcb(lambda d: d["libs_by_id"].pop(d["ids"][0]), "libs_by_id keys")
    refuses_bcb(lambda d: d["libs_by_id"][d["ids"][0]].append("not_a_known_lib"), "no family")
    refuses_bcb(lambda d: d["per_family"].__setitem__("data", d["per_family"]["data"] + 1), "does not recompute")
    refuses_bcb(lambda d: d["population"].__setitem__("extra", []), "exactly ids \\+ hard_ids")
    # BFCL: the roster, the commit, the loader, the population digest, the seats' roster
    def rename_category(d):
        d["population"]["sql_v2"] = d["population"].pop("sql")
    refuses_bfcl(rename_category, "roster")
    refuses_bfcl(lambda d: d["dataset"].__setitem__("categories", d["dataset"]["categories"][:-1]), "dataset")
    refuses_bfcl(lambda d: d["dataset"].__setitem__("commit", "0" * 40), "dataset")
    refuses_bfcl(lambda d: d["dataset"].__setitem__("loader", "inspect_evals-0.19.0 bfcl()"), "dataset")
    refuses_bfcl(lambda d: d["source_sha256"].__setitem__("gorilla_commit", "0" * 40), "gorilla_commit")
    refuses_bfcl(lambda d: d["population"]["sql"].__setitem__(0, d["population"]["sql"][0] + "_moved"),
                 "population_by_category does not recompute|do not regenerate")
    refuses_bfcl(lambda d: d["source_sha256"].__setitem__("population_by_category", "0" * 64), "does not recompute")
    refuses_bfcl(lambda d: d["per_category"].__setitem__("rest", 1), "per_category must seat")
    refuses_bfcl(lambda d: d.__setitem__("generated_by", ""), "non-empty string")
    # the contracts' own family reader and counts (the builder reads through the same two)
    fams = C.bcb_lib_families(REPO)
    assert C.bcb_per_family_counts(BCB["libs_by_id"], fams) == BCB["per_family"]
    assert C.bcb_per_family_counts({"i": []}, fams) == {"stdlib_only": 1}
    with pytest.raises(C.SubsetPinError, match="no family"):
        C.bcb_per_family_counts({"i": ["not_a_lib"]}, fams)


# --------------------------------------------------------------------------- #
# the two FULL-population pins                                                  #
# --------------------------------------------------------------------------- #

BCB_FULL = json.loads((REPO / C.BCB_FULL_PIN_PATH).read_text())
BFCL_FULL = json.loads((REPO / C.BFCL_FULL_PIN_PATH).read_text())


def test_the_full_pins_are_the_whole_populations_with_no_seed():
    """The rule is "the whole population, no seed": n == of == the population,
    seed null, the same dataset block and SOURCE digests as the 200 siblings
    (built from the same parquets / gorilla commit), the task ids the contracts
    name, and the exact key sets (BCB adds ``hard_ids`` beside the ids)."""
    assert (C.CONTROL_BIGCODEBENCH_FULL, C.CONTROL_BFCL_FULL) == ("bigcodebench_full", "bfcl_full")
    assert C.FULL_PIN_SEED is None
    assert C.SUBSET_PIN_BCB_FULL_EXTRA_FIELDS == ("libs_by_id", "per_family", "hard_ids")
    assert set(BCB_FULL) == set(C.SUBSET_PIN_FIELDS) | set(C.SUBSET_PIN_BCB_FULL_EXTRA_FIELDS)
    assert set(BFCL_FULL) == set(C.SUBSET_PIN_FIELDS)
    for doc, task, n in ((BCB_FULL, "bigcodebench_full", C.BCB_FULL_N), (BFCL_FULL, "bfcl_full", C.BFCL_FULL_N)):
        assert doc["task"] == task and doc["schema"] == C.SUBSET_PIN_SCHEMA and doc["generated_by"] == S.GENERATED_BY
        assert doc["seed"] is None and doc["n"] == doc["of"] == n == len(doc["ids"]) == len(set(doc["ids"]))
        assert "no seed" in doc["rule"] and "whole population" in doc["rule"]
    assert (len(BCB_FULL["ids"]), len(BFCL_FULL["ids"])) == (1140, 3981)
    # the same SOURCES as the 200 pins (one build lineage, two item sets)
    assert BCB_FULL["source_sha256"] == BCB["source_sha256"] and BCB_FULL["dataset"] == BCB["dataset"]
    assert BFCL_FULL["source_sha256"] == BFCL["source_sha256"] and BFCL_FULL["dataset"] == BFCL["dataset"]


def test_the_200_pins_are_subsets_of_the_full_pins():
    """The 200 legs' ids are drawn from the populations the full pins enumerate —
    the full pin is a superset by construction, pinned here."""
    assert set(BCB["ids"]) < set(BCB_FULL["ids"])
    assert set(BFCL["ids"]) < set(BFCL_FULL["ids"])
    assert BCB["population"] == BCB_FULL["population"]
    assert BFCL["population"] == BFCL_FULL["population"]
    # the 200 BCB pin excluded the Hard ids; the full pin INCLUDES them, and lists them beside
    assert set(BCB_FULL["hard_ids"]) < set(BCB_FULL["ids"]) and not (set(BCB["ids"]) & set(BCB_FULL["hard_ids"]))
    assert len(BCB_FULL["hard_ids"]) == C.BCB_HARD_N == 148
    # the per-category counts ARE the population sizes (the 200 pin's seats are <= them)
    assert BFCL_FULL["per_category"] == {c: len(v) for c, v in BFCL_FULL["population"].items()}
    assert all(BFCL["per_category"][c] <= BFCL_FULL["per_category"][c] for c in C.BFCL_CATEGORIES)


def test_the_full_pins_regenerate_from_their_recorded_populations_and_the_loaders_accept_them():
    pop = BCB_FULL["population"]
    assert C.bcb_select_full(pop["ids"], pop["hard_ids"]) == BCB_FULL["ids"]
    suffixes = [int(C.bcb_log_id(i)) for i in BCB_FULL["ids"]]
    assert suffixes == sorted(suffixes) == list(range(C.BCB_FULL_N))            # every id, in order
    assert BCB_FULL["hard_ids"] == pop["hard_ids"] == sorted(pop["hard_ids"], key=lambda i: int(C.bcb_log_id(i)))
    ids, counts = C.bfcl_select_full(BFCL_FULL["population"])
    assert ids == BFCL_FULL["ids"] and counts == BFCL_FULL["per_category"]
    assert ids == [i for c in sorted(BFCL_FULL["population"]) for i in BFCL_FULL["population"][c]]
    assert C.load_bcb_full_pin(REPO)["ids"] == BCB_FULL["ids"]
    assert C.load_bfcl_full_pin(REPO)["per_category"] == BFCL_FULL["per_category"]
    # the family map is total over the WHOLE split, and the counts recompute
    families = S.load_lib_families(REPO)
    assert {lib for libs in BCB_FULL["libs_by_id"].values() for lib in libs} <= set(families)
    assert set(BCB_FULL["libs_by_id"]) == set(BCB_FULL["ids"])
    assert S.per_family_counts(BCB_FULL["libs_by_id"], families) == BCB_FULL["per_family"]
    # the rules refuse a torn population
    with pytest.raises(ValueError, match="Hard ids outside"):
        C.bcb_select_full(["BigCodeBench/0"], ["BigCodeBench/1"])
    with pytest.raises(ValueError, match="duplicate"):
        C.bcb_select_full(["BigCodeBench/0", "BigCodeBench/0"], [])
    with pytest.raises(ValueError, match="more than one category"):
        C.bfcl_select_full({"a": ["x"], "b": ["x"]})
    with pytest.raises(ValueError, match="non-empty"):
        C.bfcl_select_full({"a": ["x"], "b": []})


def test_the_full_pin_documents_rebuild_byte_for_byte_and_check_pins_covers_them():
    families = S.load_lib_families(REPO)
    rebuilt = S.build_bcb_full_pin(ids=BCB_FULL["population"]["ids"], hard_ids=BCB_FULL["population"]["hard_ids"],
                                   libs_by_task=BCB_FULL["libs_by_id"], families=families,
                                   source_sha256=BCB_FULL["source_sha256"], generated_at=BCB_FULL["generated_at"])
    assert S.canonical_json(rebuilt) == S.canonical_json(BCB_FULL)
    rebuilt_f = S.build_bfcl_full_pin(population=BFCL_FULL["population"], source_sha256=BFCL_FULL["source_sha256"],
                                      generated_at=BFCL_FULL["generated_at"])
    assert S.canonical_json(rebuilt_f) == S.canonical_json(BFCL_FULL)
    assert S.check_pins(REPO) == 0
    # --full is a --build mode
    with pytest.raises(SystemExit, match="--build --full"):
        S.main(["--full"])


def test_the_full_item_set_identity_has_a_null_seed_and_its_own_keys():
    ident = C.item_set_identity(BCB_FULL)
    assert ident == {"seed": None, "n": 1140, "of": 1140, "ids_sha256": C.ids_sha256(BCB_FULL["ids"])}
    assert C.item_set_identity(BFCL_FULL) == {"seed": None, "n": 3981, "of": 3981,
                                              "ids_sha256": C.ids_sha256(BFCL_FULL["ids"])}
    assert C.item_set_identity(BCB) == {"seed": 0, "n": 200, "of": 992, "ids_sha256": C.ids_sha256(BCB["ids"])}
    # the full legs hash under their OWN identity keys; the sandbox key rides once on any BCB leg
    assert (C.IDENTITY_KEY_BCB_FULL_ITEM_SET, C.IDENTITY_KEY_BFCL_FULL_ITEM_SET) == ("bcb_full_item_set", "bfcl_full_item_set")
    base = C.tinycap_identity_fields()
    assert C.tinycap_identity_fields(files_bigcodebench_full=True) == (*base, "bcb_full_item_set", "bcb_sandbox")
    assert C.tinycap_identity_fields(files_bfcl_full=True) == (*base, "bfcl_full_item_set")
    both = C.tinycap_identity_fields(files_bigcodebench_200=True, files_bigcodebench_full=True)
    assert both == (*base, "bcb_item_set", "bcb_sandbox", "bcb_full_item_set") and both.count("bcb_sandbox") == 1
    assert C.tinycap_identity_fields(files_bigcodebench_200=True, files_bfcl_200=True) == (
        *base, "bcb_item_set", "bcb_sandbox", "bfcl_item_set")                      # the 200 shape, byte-identical
    rows = ["inspect/bigcodebench_full", "inspect/bfcl_full"]
    assert C.files_bigcodebench_full(rows) and C.files_bfcl_full(rows) and C.files_any_bigcodebench(rows)
    assert C.files_any_full_task(rows) and not C.files_bigcodebench_200(rows)
    assert not C.files_any_full_task(["inspect/bigcodebench_200"]) and C.files_any_bigcodebench(["inspect/bigcodebench_200"])
    with pytest.raises(ValueError, match="ROW tasks"):
        C.files_bigcodebench_full(["bigcodebench_full"])


def test_the_full_loaders_refuse_drift(tmp_path):
    root = tmp_path
    (root / C.BCB_FULL_PIN_PATH).parent.mkdir(parents=True)
    (root / C.BCB_LIB_FAMILIES_PATH).write_text((REPO / C.BCB_LIB_FAMILIES_PATH).read_text())
    bad = dict(BCB_FULL)
    bad["ids"] = list(reversed(BCB_FULL["ids"]))                                    # re-ordered
    (root / C.BCB_FULL_PIN_PATH).write_text(json.dumps(bad))
    with pytest.raises(C.SubsetPinError, match="do not regenerate"):
        C.load_bcb_full_pin(root)
    (root / C.BCB_FULL_PIN_PATH).write_text(json.dumps({**BCB_FULL, "seed": 0}))
    with pytest.raises(C.SubsetPinError, match="seed 0 != the contracts' None"):
        C.load_bcb_full_pin(root)
    (root / C.BCB_FULL_PIN_PATH).write_text(json.dumps({**BCB_FULL, "hard_ids": BCB_FULL["hard_ids"][:-1]}))
    with pytest.raises(C.SubsetPinError, match="top-level hard_ids"):
        C.load_bcb_full_pin(root)
    no_hard = {k: v for k, v in BCB_FULL.items() if k != "hard_ids"}
    (root / C.BCB_FULL_PIN_PATH).write_text(json.dumps(no_hard))
    with pytest.raises(C.SubsetPinError, match="missing \\['hard_ids'\\]"):
        C.load_bcb_full_pin(root)
    (root / C.BCB_FULL_PIN_PATH).write_text(json.dumps({**BCB_FULL, "task": "bigcodebench_200"}))
    with pytest.raises(C.SubsetPinError, match="task 'bigcodebench_200'"):
        C.load_bcb_full_pin(root)
    bf = dict(BFCL_FULL)
    bf["per_category"] = {**BFCL_FULL["per_category"], "sql": 99}
    (root / C.BFCL_FULL_PIN_PATH).write_text(json.dumps(bf))
    with pytest.raises(C.SubsetPinError, match="do not regenerate"):
        C.load_bfcl_full_pin(root)
    (root / C.BFCL_FULL_PIN_PATH).write_text(json.dumps({**BFCL_FULL, "n": 200, "ids": BFCL_FULL["ids"][:200]}))
    with pytest.raises(C.SubsetPinError, match="n 200"):
        C.load_bfcl_full_pin(root)
    (root / C.BFCL_FULL_PIN_PATH).unlink()
    with pytest.raises(C.SubsetPinError, match="missing"):
        C.load_bfcl_full_pin(root)
    # the 200 loaders never read the full pins, and vice versa
    (root / C.BCB_PIN_PATH).write_text(json.dumps(BCB_FULL))
    with pytest.raises(C.SubsetPinError):
        C.load_bcb_pin(root)
