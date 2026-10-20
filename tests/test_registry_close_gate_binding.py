"""The ladder gate's decision record decides only when it is BOUND to the selected store and its
CURRENT generation — kind `controls`, identity.cell / control / store naming the cell dir, cap = the rung, counts.n_responses = the spool's items,
and identity.store_digest = the spool's content digest (ladder_decide records it since the digest was added; at_cap_controls.store_digest), or, for a
pre-digest record, a tool.written_at that postdates the spool's last write. A stale record (an earlier generation's exit left at the path while
the current rung's driver says escalate), a wrong-store record or a record without identity is REFUSED BY NAME — the driver's flag never quietly
replaces it, so an unbound record can never mark a control generated at a rung the current gate did not exit. Reuses the landed gate test's world."""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_registry_close_stores_gate_decision import ACC, KEY, OBJ, RC, TAG, _gate_record, _spool, _world  # noqa: E402


def _cell(root: Path, cap: int) -> Path:
    return root / "sampled" / f"controls_cap{cap}" / f"{TAG}_t0" / "cells" / f"{OBJ.replace('__', '--')}__controls__t0__a0__deadbeef__{cap:08x}"


def test_a_bound_record_decides_and_says_how_it_is_bound(tmp_path):
    root = _world(tmp_path / "w", rungs={512: True, 2048: True}, gate={512: "escalate", 2048: "exit_le_1pct"})
    why, facts, store, _a = RC.inspect_controls(str(root), TAG, 0, OBJ, KEY, planned=250)
    assert why is None, why
    assert facts["decided_by"] == "ladder_gate" and facts["final_rung"] == 2048 and facts["gate_binding"] == "digest" and facts["gens_rows"] == 450
    assert facts["gate_record"].endswith(f"gates/cap2048/{OBJ}__{KEY}.decision.json") and len(facts["spool_digest"]) == 16
    # the producer's digest is the one the closer recomputes
    texts = ACC._spool_texts(_cell(root, 2048) / "private" / f"gens_{KEY}.jsonl")
    assert json.loads((root / "sampled" / "ladders" / f"{TAG}_t0" / "gates" / "cap2048" / f"{OBJ}__{KEY}.decision.json").read_text())["identity"]["store_digest"] == ACC.store_digest(texts)
    assert ACC.store_digest(texts) != ACC.store_digest({**texts, "i0000": "changed"}) and ACC.store_digest(dict(reversed(list(texts.items())))) == ACC.store_digest(texts)


def test_a_stale_record_of_a_replaced_generation_is_refused_not_replaced_by_the_drivers_flag(tmp_path):
    # the driver at 2048 says escalate=False (the fallback WOULD accept); the gate record exited — but it judged the PREVIOUS generation
    root = _world(tmp_path / "w", rungs={512: True, 2048: False}, gate={512: "escalate", 2048: "exit_le_1pct"})
    _spool(_cell(root, 2048), 450, salt=" regenerated")                     # same count, new texts: the digest moves, the record does not
    why, facts, store, _a = RC.inspect_controls(str(root), TAG, 0, OBJ, KEY, planned=250)
    assert why and "is STALE" in why and "a replaced generation" in why and "re-run the ladder gate" in why, why
    assert "final_rung" not in facts and facts["gate_rung"] == 2048 and store.endswith("deadbeef__00000800")
    # a pre-digest record (no store_digest) that predates the spool's last write is stale too …
    cell = _cell(root, 2048); texts = ACC._spool_texts(cell / "private" / f"gens_{KEY}.jsonl")
    _gate_record(root, 2048, "exit_le_1pct", cell=cell, n=450, identity={"out_root": str(cell.parents[1]), "store": str(cell), "cell": OBJ, "control": KEY}, written_at="2026-01-01T00:00:00+00:00")
    why, facts, _s, _a = RC.inspect_controls(str(root), TAG, 0, OBJ, KEY, planned=250)
    assert why and "is STALE" in why and "predates the spool's last write" in why, why
    # … and one written AFTER the spool binds by identity, count and order
    later = time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime(os.path.getmtime(cell / "private" / f"gens_{KEY}.jsonl") + 60))
    _gate_record(root, 2048, "exit_le_1pct", cell=cell, n=450, identity={"out_root": str(cell.parents[1]), "store": str(cell), "cell": OBJ, "control": KEY}, written_at=later)
    why, facts, _s, _a = RC.inspect_controls(str(root), TAG, 0, OBJ, KEY, planned=250)
    assert why is None and facts["gate_binding"] == "identity+count+order" and facts["decided_by"] == "ladder_gate", why
    # a record with neither digest nor written_at is unbound
    _gate_record(root, 2048, "exit_le_1pct", cell=cell, n=450, identity={"out_root": str(cell.parents[1]), "store": str(cell), "cell": OBJ, "control": KEY}, tool={"name": "x"})
    why, facts, _s, _a = RC.inspect_controls(str(root), TAG, 0, OBJ, KEY, planned=250)
    assert why and "neither a store digest nor a tool.written_at" in why, why


def test_wrong_store_wrong_cell_missing_identity_wrong_kind_wrong_cap_and_count_mismatch_refuse_by_name(tmp_path):
    root = _world(tmp_path / "w", rungs={512: True, 2048: False}, gate={512: "escalate", 2048: "exit_le_1pct"})
    cell = _cell(root, 2048); other = _cell(root, 512)
    cases = [
        (dict(identity={"out_root": str(other.parents[1]), "store": str(other), "cell": OBJ, "control": KEY}), "wrong store"),
        (dict(identity={"out_root": str(cell.parents[1]), "store": str(cell), "cell": "someone-else__psr_ce_all", "control": KEY}), "wrong record"),
        (dict(identity={"out_root": str(cell.parents[1]), "store": str(cell), "cell": OBJ, "control": "or_bench_hard_1k"}), "wrong record"),
        (dict(identity=None), "carries no controls identity"),
        (dict(kind="static"), "carries no controls identity"),
        (dict(_cap=512), "is for cap 512, the rung is 2048"),
        (dict(counts={"n_responses": 449}), "counts 449 response(s), the spool holds 450"),
    ]
    for over, frag in cases:
        ident = over.pop("identity", "keep")
        kw = dict(cell=cell, n=450)
        if ident != "keep":
            kw["identity"] = ident if ident is not None else {}
        cap_over = over.pop("_cap", None)
        p = _gate_record(root, 2048, "exit_le_1pct", **kw, **over)
        if ident is None or cap_over is not None:
            d = json.loads(p.read_text())
            if ident is None:
                d.pop("identity")
            if cap_over is not None:
                d["cap"] = cap_over
            p.write_text(json.dumps(d))
        why, facts, _s, _a = RC.inspect_controls(str(root), TAG, 0, OBJ, KEY, planned=250)
        assert why and frag in why and "final_rung" not in facts, (over, frag, why)       # refused by name; the driver's False flag did not decide
    # a torn spool cannot bind either
    _gate_record(root, 2048, "exit_le_1pct", cell=cell, n=450)
    with open(cell / "private" / f"gens_{KEY}.jsonl", "a") as fh:
        fh.write('{"item_id": "i9999", "te')
    why, facts, _s, _a = RC.inspect_controls(str(root), TAG, 0, OBJ, KEY, planned=250)
    assert why and "cannot be read as a complete store" in why and "torn line" in why, why


def test_the_producer_records_the_digest_it_will_be_held_to(tmp_path):
    """ladder_decide.decide_controls: identity carries n_items and store_digest computed from the store it read (the same function the closer uses)."""
    src = (Path(RC.__file__).resolve().parent / "sampled" / "ladder_decide.py").read_text()
    assert '"n_items": int(store["n"]), "store_digest": acc.store_digest(store["texts"])' in src
    assert "def store_digest(texts: Dict[str, str]) -> str:" in (Path(ACC.__file__).read_text())
