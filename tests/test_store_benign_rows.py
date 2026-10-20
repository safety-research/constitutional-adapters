"""The store's BENIGN strata must reach the matched pool.

Regression test for the third store/pool integration gap (2026-08-21):
`_iter_all_pool_rows` read the store with the HARMFUL family list only
(`static_families | replay_families`), so when the pool_sample lane wrote
harmful and benign side by side into one store, the benign rows were never
read. The manifest then reported `benign complied: <si>=0` for every SI and
raised PoolFloorError with a fully-populated harmful side — after grade-store
had already paid to label those very rows.

Synthetic rows only; no real prompts or completions.
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import build_two_si_pool as bp  # noqa: E402


def _row(fam, gid, si_id, side, label=None, score=None):
    return {
        "attack": fam, "goal_id": gid, "sample_idx": 0, "trial": 0,
        "si_condition": "none" if si_id is None else "prompted",
        "si_id": si_id, "candidate_idx": 0, "category": None,
        "category_group": None, "source": "synthetic", "side": side,
        "variant_name": f"{fam}_0", "attack_prompt": f"PLACEHOLDER-PROMPT-{gid}",
        "response": f"PLACEHOLDER-RESPONSE-{gid}-{si_id}",
        "score": score, "graded": score is not None,
        **({"xstest_label": label} if label is not None else {}),
    }


@pytest.fixture
def store(tmp_path):
    d = tmp_path / "store"
    d.mkdir()
    (d / "bon.jsonl").write_text("".join(
        json.dumps(_row("bon", "g1", si, "harmful", score=0.0 if si else 0.9)) + "\n"
        for si in (None, "safety_refusal", "si_constitution", "si_irrelevant")))
    (d / "phtest.jsonl").write_text("".join(
        json.dumps(_row("phtest", "b1", si, "benign",
                        label=None if si is None else "C")) + "\n"
        for si in (None, "safety_refusal", "si_constitution", "si_irrelevant")))
    return d


def test_benign_store_rows_are_read(store, monkeypatch):
    """The bug: benign families were absent from the store read entirely."""
    m = bp
    monkeypatch.setenv("TT_TWO_SI_STORE_DIR", str(store))
    cfg = {
        "si_ids": ["safety_refusal", "si_constitution", "si_irrelevant"],
        "pool": {
            "store_dir": str(store),
            "static_families": ["bon"],
            "benign_sources": [{"name": "phtest", "loader": "phtest"}],
            "benign_source": "phtest",
            "anti_neg": {"enabled": False},
        },
    }
    fams = [r.get("attack") for r in
            m._iter_all_pool_rows(cfg, static_families=("bon",),
                                  private_root=str(store.parent))]
    assert "bon" in fams, "harmful rows must still be read"
    assert "phtest" in fams, (
        "BENIGN store rows were not read — this is the exact regression: the "
        "store read covered only the harmful family list, so the benign half "
        "of the matched pool came out empty even after grade-store labelled it")


def test_no_double_yield_when_family_in_both_lists(store, monkeypatch):
    """A family named in BOTH the harmful and benign lists is read once."""
    m = bp
    monkeypatch.setenv("TT_TWO_SI_STORE_DIR", str(store))
    cfg = {
        "si_ids": ["safety_refusal"],
        "pool": {
            "store_dir": str(store),
            "static_families": ["bon", "phtest"],      # deliberately overlapping
            "benign_sources": [{"name": "phtest", "loader": "phtest"}],
            "benign_source": "phtest",
            "anti_neg": {"enabled": False},
        },
    }
    rows = list(m._iter_all_pool_rows(cfg, static_families=("bon", "phtest"),
                                      private_root=str(store.parent)))
    n_phtest = sum(1 for r in rows if r.get("attack") == "phtest")
    assert n_phtest == 4, f"phtest rows double-counted: {n_phtest} != 4"
