"""Held-out leakage gate — the normalised-exact-match refuse-on-overlap contract
(same shape as the OR-Bench 80k-minus-hard-1k subtraction in content_sets).

A held-out family that leaks into the training pool is worse than no family, so
the gate REFUSES (never warns). All-synthetic invented prompts — a deliberate
NEAR_MISS that must survive, plus WHITESPACE/CASE variants that must be caught.
"""

import json

import pytest

from sisteer.eval.attacks.heldout_common import (
    LeakageError,
    check_corpus_leakage,
    fingerprint_sha,
    materialize_corpus,
    training_fingerprint,
)
from sisteer.eval.attacks.heldout_leakage import (
    fingerprint_meta,
    load_stage3_training_fingerprint_from_config,
    load_training_fingerprint,
    verify_fingerprint,
    write_fingerprint_meta,
)
from sisteer.eval.jailbreak_data import Goal

# Invented training prompts (benign, synthetic) standing in for the pool.
TRAIN = [
    "Explain how a fictional detective catalogues rare buttons.",
    "Describe the migration pattern of imaginary glass butterflies.",
]
NEAR_MISS = "Describe the migration pattern of imaginary glass moths."
WHITESPACE_VARIANT = "  Explain how a fictional\tdetective   catalogues rare buttons.  "
CASE_VARIANT = TRAIN[1].upper()


def _rows(prompts):
    return [{"goal_id": f"g{i}", "sample_idx": 0, "variant_name": "v", "prompt": p}
            for i, p in enumerate(prompts)]


def test_gate_refuses_normalised_exact_overlap():
    fp = training_fingerprint(TRAIN)
    # verbatim, whitespace-variant and case-variant all collide -> refuse
    for leak in (TRAIN[0], WHITESPACE_VARIANT, CASE_VARIANT):
        with pytest.raises(LeakageError):
            check_corpus_leakage(_rows([leak]), fp, refuse=True)


def test_gate_passes_near_miss_and_clean():
    fp = training_fingerprint(TRAIN)
    rep = check_corpus_leakage(_rows([NEAR_MISS, "a wholly unrelated prompt"]),
                               fp, refuse=True)
    assert rep.passed and rep.n_overlap == 0 and rep.n_checked == 2


def test_report_is_leak_free_goal_ids_only():
    fp = training_fingerprint(TRAIN)
    rows = [{"goal_id": "leaky", "sample_idx": 0, "variant_name": "v",
             "prompt": CASE_VARIANT}]
    rep = check_corpus_leakage(rows, fp, refuse=False)
    assert rep.n_overlap == 1
    assert rep.overlap_goal_ids == ("leaky",)
    # the report / its manifest form carry NO prompt text
    assert CASE_VARIANT not in json.dumps(rep.as_manifest())


def test_materialised_class_a_corpus_is_clean_against_unrelated_pool():
    from sisteer.eval.attacks.cipherchat import cipherchat_variants

    goals = [Goal("g", "do the specified procedure", "illegal", "d", "r")]
    rows = materialize_corpus(cipherchat_variants, goals)
    fp = training_fingerprint(["an unrelated training prompt"])
    assert check_corpus_leakage(rows, fp, refuse=True).passed


# --------------------------------------------------------------------------- #
# the authoritative training-fingerprint loader (synthetic store)             #
# --------------------------------------------------------------------------- #

def _write_store(tmp, family, prompts):
    (tmp / f"{family}.jsonl").write_text(
        "\n".join(json.dumps({"attack_prompt": p, "goal_id": "x"})
                  for p in prompts) + "\n")


def test_load_training_fingerprint_reads_attack_prompt(tmp_path):
    _write_store(tmp_path, "bon", ["train prompt one", "train prompt two"])
    fp = load_training_fingerprint(["bon"], store_dir=str(tmp_path))
    assert fp == training_fingerprint(["train prompt one", "train prompt two"])


def test_load_training_fingerprint_missing_family_refuses(tmp_path):
    _write_store(tmp_path, "bon", ["x"])
    with pytest.raises(FileNotFoundError, match="missing declared family"):
        load_training_fingerprint(["bon", "pasttense"], store_dir=str(tmp_path))


def test_load_training_fingerprint_missing_store_refuses(tmp_path):
    with pytest.raises(FileNotFoundError, match="not found"):
        load_training_fingerprint(["bon"], store_dir=str(tmp_path / "nope"))


def test_load_training_fingerprint_empty_families_refuses(tmp_path):
    with pytest.raises(ValueError, match="no training families"):
        load_training_fingerprint([], store_dir=str(tmp_path))


def test_config_loader_env_overrides_store_dir(tmp_path, monkeypatch):
    """The config loader must resolve ``$TT_TWO_SI_STORE_DIR`` BEFORE
    ``pool.store_dir`` — the same precedence ``build_two_si_pool._store_dir`` uses,
    so the gate checks the store the defense was actually fit on (the config
    default belongs to a different lane on the grading host)."""
    cfg_default = tmp_path / "config_default_lane"
    env_store = tmp_path / "env_stage3_store"
    cfg_default.mkdir()
    env_store.mkdir()
    _write_store(cfg_default, "bon", ["config-lane prompt"])
    _write_store(env_store, "bon", ["stage3 env prompt one", "stage3 env prompt two"])
    cfg = {"pool": {"train_families": ["bon"], "store_dir": str(cfg_default)}}

    monkeypatch.setenv("TT_TWO_SI_STORE_DIR", str(env_store))
    fp = load_stage3_training_fingerprint_from_config(cfg)
    assert fp == training_fingerprint(["stage3 env prompt one",
                                       "stage3 env prompt two"])


def test_config_loader_falls_back_to_store_dir_without_env(tmp_path, monkeypatch):
    """Absent the env, the loader reads ``pool.store_dir`` (backward-compatible —
    the env fix only activates when the override is set)."""
    cfg_default = tmp_path / "config_default_lane"
    cfg_default.mkdir()
    _write_store(cfg_default, "bon", ["config-lane prompt"])
    cfg = {"pool": {"train_families": ["bon"], "store_dir": str(cfg_default)}}

    monkeypatch.delenv("TT_TWO_SI_STORE_DIR", raising=False)
    fp = load_stage3_training_fingerprint_from_config(cfg)
    assert fp == training_fingerprint(["config-lane prompt"])


def test_fingerprint_meta_and_verify(tmp_path):
    fp = training_fingerprint(TRAIN)
    meta = write_fingerprint_meta(fp, tmp_path / "meta.json",
                                  provenance={"source": "synthetic"})
    assert meta["fingerprint_sha"] == fingerprint_sha(fp)
    assert meta["n_shas"] == len(fp)
    verify_fingerprint(fp, tmp_path / "meta.json")  # matches -> no raise
    # a drifted fingerprint is caught loudly
    with pytest.raises(ValueError, match="drifted"):
        verify_fingerprint(training_fingerprint(TRAIN + ["extra"]),
                           tmp_path / "meta.json")


def test_build_llm_corpus_records_gate_removal_instead_of_refusing(tmp_path):
    # DEMOTED (2026-08-25 directive, applied at the phase-C join): a corpus
    # built over TRAINING goals overlaps the training fingerprint BY
    # CONSTRUCTION, so the rewriter build RECORDS the removal instead of
    # refusing (LeakageError is no longer reachable from this path).
    from sisteer.eval.attacks.llm_rewriter import build_llm_corpus

    class LeakRewriter:
        model = "stub"
        def batch(self, items):
            return ["the leaked prompt"] * len(items)

    goals = [Goal("g", "x", "c", "d", "r")]
    leak_fp = training_fingerprint(["THE LEAKED PROMPT"])
    man = build_llm_corpus(goals, rewriter=LeakRewriter(),
                           plan_fn=lambda g: [("v", "", "instr")],
                           out_path=tmp_path / "c.jsonl", fingerprint=leak_fp,
                           provenance={"method": "x"}, family_name="t")
    assert man["leakage_check"]["gate"] == "removed_by_directive_2026_08_25"
    assert (tmp_path / "c.jsonl").exists()
