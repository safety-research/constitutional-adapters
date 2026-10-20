"""Migration 0012: prompt padding is INSTRUMENT identity, and the two new identity fields move NO existing id.

`identity.INSTRUMENT_ID_FIELDS` gained `padding_tokens` and `padding_identity_sha256`; both are in `INSTRUMENT_ID_OPTIONAL_FIELDS`, so
they are dropped from the canonical JSON while NULL and an unpadded instrument hashes exactly as it did before the fields existed. This
file proves the invariance three ways -- (1) against a re-implementation of the pre-0012 formula on the synthetic and fixture instrument
rows, (2) against the ids the fixture rows CARRY (minted by the pre-0012 code and pinned in the fixtures), and (3) read-only against
every instrument row of the live registry when one is on this machine (every UNPADDED live id hashes as before 0012; every PADDED one --
the static_padding rows window 8 filed on 2026-09-23 -- is reproduced by the new formula and moved off the pre-0012 one, which is the
discrimination working live) -- and then that a non-NULL pad or contract hash DOES discriminate, that the contract hash is a pure function of the five contract keys, and
that the paper views expose the columns beside the cap and the window. Since migration 0014 a pad of 0 WITH its contract is the
recipe's ZERO POINT: a row of its own, distinct from the NULL (no recipe) id and from every padded level, kept by the hash as a value.
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3

import pytest

from sisteer.registry import db as DB
from sisteer.registry import identity as I
from sisteer.registry import schema_def as S
from sisteer.registry import synth

import registry_items_fixtures as F

pytestmark = pytest.mark.live_db   # this module opens the live registry (one cheap `instrument` read): land.sh's local step runs it

LIVE_DB = os.path.expanduser(os.environ.get("REGISTRY_LIVE_DB", "~/registry/registry.sqlite"))   # the replica's node-local copy on Slurm
PRE_0012_FIELDS = tuple((n, t) for n, t in I.INSTRUMENT_ID_FIELDS if n not in I.INSTRUMENT_ID_OPTIONAL_FIELDS)
BLOCK = {"schema": "static_padding/1", "tokens": 16384, "placement": "before", "mode": "fixed", "source": "pg19",
         "filler_sha256": "7" * 64, "separator": "\n\n", "tolerance": 0.02}


def _pre_0012_instrument_id(fields) -> str:
    """The formula as it stood before 0012: every then-declared identity field, None when absent, typed, sorted, hashed."""
    typed = I.instrument_identity_fields(fields)
    ident = {name: typed.get(name) for name, _t in PRE_0012_FIELDS}
    return hashlib.sha256(I.canonical_json(ident).encode("utf-8")).hexdigest()


def test_the_two_padding_fields_are_declared_optional_and_pinned_to_the_table():
    assert set(I.INSTRUMENT_ID_OPTIONAL_FIELDS) == {"padding_tokens", "padding_identity_sha256"}
    assert dict(I.INSTRUMENT_ID_FIELDS)["padding_tokens"] == "INTEGER" and dict(I.INSTRUMENT_ID_FIELDS)["padding_identity_sha256"] == "TEXT"
    t = S.TABLE_BY_NAME["instrument"]
    assert t.column("padding_tokens").type == "INTEGER" and not t.column("padding_tokens").nn
    assert t.column("padding_identity_sha256").pattern == "SHA256" and not t.column("padding_identity_sha256").nn
    assert any(name == "instrument_padding_pairing" for name, _sql in t.checks)
    # the optional set is a SUBSET of the identity fields, and nothing pre-0012 is optional (an optional old field would move ids)
    assert I.INSTRUMENT_ID_OPTIONAL_FIELDS < set(dict(I.INSTRUMENT_ID_FIELDS))
    assert not (I.INSTRUMENT_ID_OPTIONAL_FIELDS & set(dict(PRE_0012_FIELDS)))


def test_unpadded_instruments_keep_their_pre_0012_ids_on_the_synthetic_and_fixture_rows():
    rows = [dict(synth.INSTRUMENT), dict(synth.RULE_INSTRUMENT)] + [dict(f) for f, _extra, _seed, _eval, _model in F.instrument_rows()]
    assert len(rows) >= 8
    for row in rows:
        carried = row.pop("instrument_id")
        assert I.instrument_id(row) == _pre_0012_instrument_id(row) == carried, row.get("eval_id")
        # the identity dict still names every field (the core-fixes pin reads it); the hash just omits the NULL optional ones
        assert set(I.instrument_identity_fields(row)) == set(dict(I.INSTRUMENT_ID_FIELDS))
        assert I.instrument_identity_fields(row)["padding_tokens"] is None and I.instrument_identity_fields(row)["padding_identity_sha256"] is None
        # spelling the NULLs out changes nothing either
        assert I.instrument_id(dict(row, padding_tokens=None, padding_identity_sha256=None)) == carried


def test_a_pad_or_a_contract_hash_discriminates_and_the_hash_is_a_pure_function_of_the_contract():
    base = dict(synth.INSTRUMENT)
    a = base.pop("instrument_id")
    contract = I.padding_identity_sha256(BLOCK)
    assert len(contract) == 64 and contract == I.padding_identity_sha256(dict(BLOCK, tokens=65536, source="other_corpus", schema="x"))
    assert contract == I.padding_identity_sha256({k: BLOCK[k] for k in I.PADDING_IDENTITY_KEYS})
    assert contract != I.padding_identity_sha256(dict(BLOCK, placement="after"))
    assert contract != I.padding_identity_sha256(dict(BLOCK, filler_sha256="8" * 64))
    assert contract != I.padding_identity_sha256(dict(BLOCK, tolerance=0.05))
    with pytest.raises(ValueError, match="lacks"):
        I.padding_identity_sha256({k: v for k, v in BLOCK.items() if k != "separator"})
    p16 = I.instrument_id(dict(base, padding_tokens=16384, padding_identity_sha256=contract))
    p64 = I.instrument_id(dict(base, padding_tokens=65536, padding_identity_sha256=contract))
    other_contract = I.instrument_id(dict(base, padding_tokens=16384, padding_identity_sha256=I.padding_identity_sha256(dict(BLOCK, placement="after"))))
    assert len({a, p16, p64, other_contract}) == 4
    # Migration 0014: the recipe's ZERO POINT -- padding_tokens 0 WITH the contract -- is its own instrument: distinct from the unpadded
    # (NULL) id and from every padded level; a 0 is a VALUE the hash keeps (only a NULL is dropped as optional), never a falsy absence
    p0 = I.instrument_id(dict(base, padding_tokens=0, padding_identity_sha256=contract))
    assert len({a, p0, p16, p64, other_contract}) == 5
    assert I.instrument_identity_fields(dict(base, padding_tokens=0, padding_identity_sha256=contract))["padding_tokens"] == 0
    assert I.instrument_id(dict(base, padding_tokens=0.0, padding_identity_sha256=contract)) == p0
    assert p0 != I.instrument_id(dict(base, padding_tokens=0, padding_identity_sha256=I.padding_identity_sha256(dict(BLOCK, placement="after"))))
    # the typed normalisation holds for the new INTEGER field too (16384.0 == 16384), and a fractional pad is refused
    assert I.instrument_id(dict(base, padding_tokens=16384.0, padding_identity_sha256=contract)) == p16
    with pytest.raises(ValueError):
        I.instrument_id(dict(base, padding_tokens=16384.5, padding_identity_sha256=contract))


@pytest.mark.skipif(not os.path.exists(LIVE_DB), reason="no live registry on this machine")
def test_every_live_instrument_id_is_reproduced_read_only():
    """Every UNPADDED instrument row the live registry carries hashes as it did before 0012; every PADDED row (filed after 0012 by the
    static_padding reader) is reproduced by the new formula and is NOT reproduced by the pre-0012 one -- the pad is identity, live.

    The first version of this test asserted that NO live row was padded. That was a statement about the registry on the day 0012 landed,
    not an invariant, and it went red the moment window 8 (2026-09-23 10:14Z) filed the eight padded instruments the migration exists for.
    """
    conn = DB.connect(LIVE_DB, readonly=True)
    try:
        cols = [c[1] for c in conn.execute("PRAGMA table_info(instrument)")]
        rows = [dict(zip(cols, r)) for r in conn.execute("SELECT * FROM instrument").fetchall()]
    finally:
        conn.close()
    assert rows, "the live registry carries instrument rows"
    assert "padding_tokens" in cols and "padding_identity_sha256" in cols, "0012 is applied to the live registry"
    unpadded = [r for r in rows if r["padding_tokens"] is None]
    padded = [r for r in rows if r["padding_tokens"] is not None]
    assert unpadded, "the live registry carries unpadded instrument rows"
    # INVARIANCE: on every UNPADDED live row the new formula equals the pre-0012 formula -- 0012 moves no pre-existing id
    moved = [r["instrument_id"][:12] for r in unpadded if I.instrument_id(r) != _pre_0012_instrument_id(r)]
    assert not moved, (len(unpadded), moved[:10])
    # and the unpadded rows whose STORED id neither formula reproduces (minted under an older identity rule before it was widened, or
    # filed by a producer with its own spelling) are the same set under both: none of them is 0012's doing. Their count is reported, not hidden.
    not_new = {r["instrument_id"] for r in unpadded if I.instrument_id(r) != r["instrument_id"]}
    not_pre = {r["instrument_id"] for r in unpadded if _pre_0012_instrument_id(r) != r["instrument_id"]}
    assert not_new == not_pre, (sorted(not_new ^ not_pre)[:10])
    n_reproduced = len(unpadded) - len(not_new)
    assert n_reproduced >= len(unpadded) - 8, (len(unpadded), n_reproduced, sorted(x[:12] for x in not_new))     # 1,253 of 1,261 on 2026-09-22; the 8 predate 0012
    # DISCRIMINATION, live: every padded row pairs its pad with a contract hash (the table CHECK), the new formula reproduces its stored
    # id, and the pre-0012 formula does not -- the same instrument without its pad is a different id
    for r in padded:
        # >= 0: the recipe's zero point files as padding_tokens 0 WITH its contract since migration 0014, a non-NULL row like the padded ones
        assert int(r["padding_tokens"]) >= 0 and isinstance(r["padding_identity_sha256"], str) and len(r["padding_identity_sha256"]) == 64, r["instrument_id"][:12]
        assert I.instrument_id(r) == r["instrument_id"], r["instrument_id"][:12]
        assert _pre_0012_instrument_id(r) != r["instrument_id"], r["instrument_id"][:12]
        assert I.instrument_id(dict(r, padding_tokens=None, padding_identity_sha256=None)) == _pre_0012_instrument_id(r), r["instrument_id"][:12]


def _fresh_with_dimensions(tmp_path):
    root = str(tmp_path / "reg")
    F.ingest_dimensions(root)
    return root


def test_padding_columns_reach_the_paper_views_beside_cap_and_window(tmp_path):
    """`v_paper_long` carries `padding_tokens` / `padding_identity_sha256` right after `context_window_tokens`, `v_run_wide` their
    instrument-prefixed twins; an unpadded run reads NULL on both, a run of a padded instrument reads the pad and the contract hash."""
    root = _fresh_with_dimensions(tmp_path)
    conn = DB.connect(DB.db_path(root), readonly=True)
    cols = [c[1] for c in conn.execute("PRAGMA table_info(v_paper_long)")]
    assert cols.index("padding_tokens") == cols.index("context_window_tokens") + 1
    assert cols.index("padding_identity_sha256") == cols.index("padding_tokens") + 1
    assert cols.index("token_cap_tokens") < cols.index("padding_tokens")
    wide = [c[1] for c in conn.execute("PRAGMA table_info(v_run_wide)")]
    assert "instrument_padding_tokens" in wide and "instrument_padding_identity_sha256" in wide
    conn.close()
    # the CHECK: a pad of 0, or a pad without its contract, is refused at the table
    conn = DB.connect(DB.db_path(root))
    contract = I.padding_identity_sha256(BLOCK)
    base = dict(synth.INSTRUMENT)
    base.pop("instrument_id")
    eval_id, item_set = "pasttense", F.EVALS["pasttense"][3]
    judge = F.judge_ids()["strongreject_v1_haiku45"]
    fields = dict(base, eval_id=eval_id, item_set_id=item_set, judge_id=judge)

    def insert(**over):
        row = dict(fields, **over)
        row["instrument_id"] = I.instrument_id(row)
        row["token_cap_ladder_json"] = row["token_cap_ladder_json"] if isinstance(row["token_cap_ladder_json"], str) else json.dumps(row["token_cap_ladder_json"])
        row.update(field_confidence_json="{}", field_versions_json="{}", extra_json="{}", record_uri="test", ingest_seq=1)
        conn.execute(f"INSERT INTO instrument ({', '.join(row)}) VALUES ({', '.join('?' for _ in row)})", list(row.values()))
        return row["instrument_id"]

    with pytest.raises(sqlite3.IntegrityError, match="instrument_padding_pairing"):
        insert(padding_tokens=16384)
    with pytest.raises(sqlite3.IntegrityError, match="instrument_padding_pairing"):
        insert(padding_identity_sha256=contract)
    with pytest.raises(sqlite3.IntegrityError, match="instrument_padding_pairing"):
        insert(padding_tokens=-1, padding_identity_sha256=contract)
    with pytest.raises(sqlite3.IntegrityError, match="instrument_padding_pairing"):
        insert(padding_tokens=0)
    padded = insert(padding_tokens=16384, padding_identity_sha256=contract)
    # Migration 0014: the zero point -- 0 WITH the contract -- is a row of its own, stored as 0, never NULL
    zero = insert(padding_tokens=0, padding_identity_sha256=contract)
    conn.commit()
    got = conn.execute("SELECT padding_tokens, padding_identity_sha256 FROM instrument WHERE instrument_id = ?", (padded,)).fetchone()
    assert tuple(got) == (16384, contract)
    assert zero != padded
    assert tuple(conn.execute("SELECT padding_tokens, padding_identity_sha256 FROM instrument WHERE instrument_id = ?", (zero,)).fetchone()) == (0, contract)
    assert conn.execute("SELECT COUNT(*) FROM instrument WHERE padding_tokens = 0").fetchone()[0] == 1
    conn.close()
