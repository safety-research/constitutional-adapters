"""``_items_common.Dims.instrument`` resolves a seed instrument to ONE row per (seed, eval, served model) the way the writer writes them
(``lineage_common.instrument_row`` is the mirror): a row bound to one model carries ``extra.model_id``; the models whose decode
policies agree share ONE row with ``extra.model_ids`` and no ``model_id``. The seeds fold the four model-independent multi-turn
``generation_config`` instruments into such a shared row; before the ``model_ids`` branch the resolver filtered on ``model_id``
alone and refused the seed as ambiguous whenever a bound row for another model sat beside the shared one. Synthetic rows only.
"""
from __future__ import annotations

import json
import sqlite3

import pytest

from sisteer.registry.adapters import _items_common as IC

SEED, EVAL = "multiturn_crescendo_cap4096_generation_config", "crescendo"
BOUND_MODEL, SHARED_A, SHARED_B, UNLISTED = "synthmodel_server_default", "synthmodel_a", "synthmodel_b", "synthmodel_z"


def _dims(rows):
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE instrument (instrument_id TEXT, item_set_id TEXT, eval_id TEXT, extra_json TEXT, decode_source TEXT, "
                 "token_cap_tokens INTEGER, judge_id TEXT)")
    for iid, extra in rows:
        conn.execute("INSERT INTO instrument VALUES (?, ?, ?, ?, ?, ?, ?)", (iid, "items-1", EVAL, json.dumps(extra), "generation_config", 4096, None))
    return IC.Dims(conn=conn)


BOUND = ("inst-bound", {"seed_instrument_id": SEED, "model_id": BOUND_MODEL, "model_ids": [BOUND_MODEL]})
SHARED = ("inst-shared", {"seed_instrument_id": SEED, "model_ids": [SHARED_A, SHARED_B]})


def test_shared_model_ids_row_resolves_beside_a_bound_row():
    d = _dims([BOUND, SHARED])
    assert d.instrument(SEED, EVAL, SHARED_A)[0] == "inst-shared"
    assert d.instrument(SEED, EVAL, SHARED_B)[0] == "inst-shared"
    assert d.instrument(SEED, EVAL, BOUND_MODEL)[0] == "inst-bound"          # the bound row still wins for its own model
    assert d.instruments[(SEED, EVAL, SHARED_A)][2]["model_ids"] == [SHARED_A, SHARED_B]


def test_model_listed_nowhere_is_still_ambiguous():
    """A served model neither row names is not defaulted to either: the seed stays ambiguous for it."""
    with pytest.raises(IC.Unresolved, match="2 instrument rows"):
        _dims([BOUND, SHARED]).instrument(SEED, EVAL, UNLISTED)


def test_single_shared_row_needs_no_model_filter():
    d = _dims([SHARED])
    assert d.instrument(SEED, EVAL, SHARED_A)[0] == "inst-shared" and d.instrument(SEED, EVAL, None)[0] == "inst-shared"
