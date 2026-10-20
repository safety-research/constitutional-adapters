"""A BOUND ladder-gate record is held to the store it binds. The closer
validates the store's sidecars (gen_stats.json[control] + the meta sidecar, at_cap_controls.read_control) and RE-DERIVES the decision from the
validated store with the record's own thresholds, ladder and classifier sources (by sha256) UNDER THE SIBLING STATE THE RECORD DECLARES: `used` ->
the named sibling store is read now and must still be the generation the gate paired (item set, cap, and the `store_digest` ladder_decide records
since the digest was added — a regenerated or vanished sibling is refused by name); `unavailable` / `not_given` -> no boundary calls, as the gate ruled
(records written while the t1 sibling was still absent; the sibling arriving later must not move them). A record whose counts
or decision the store does not reproduce, a record missing the blocks the re-derivation needs, and a store whose sidecars do not validate are
refused by name — never decided by the driver's flag instead. Synthetic tag / cell ids; every record written by the REAL decide_controls."""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_registry_close_stores_gate_decision import ACC, KEY, OBJ, RC, TAG, _cell_dir, _gate_record, _spool, _world  # noqa: E402

ROWS = 450


def _two_trials(root: Path, *, at_cap: int = 6, terminated: int = 3):
    """t0: a 2048 store with `at_cap` items at cap (6/450 = 1.33 % > 1 %: ESCALATE without a sibling); t1: the sibling whose first `terminated` items
    stop at 100 tokens (boundary for t0's at-cap items 0..2 -> 3/450 truncated: EXIT with it). Both complete stores; the driver's flag says escalate."""
    for trial in (0, 1):
        cell = _cell_dir(root, 2048, trial); (cell / "private").mkdir(parents=True)
        (cell / "manifest.json").write_text(json.dumps({"cell_id": OBJ, "intervention_hash": "ii" * 8}))
        (cell / "cap_ladder.json").write_text(json.dumps({"controls": {KEY: {"cap": 2048, "escalate": True, "exit": "escalate", "n": ROWS, "n_at_cap": at_cap,
                                                                              "n_truncated": at_cap, "n_boundary": 0}}}))
        _spool(cell, ROWS, cap=2048, at_cap=at_cap, terminated=(0 if trial == 0 else terminated))
    return _cell_dir(root, 2048, 0), _cell_dir(root, 2048, 1)


def _gate_path(root: Path, cap: int, trial: int = 0) -> Path:
    return root / "sampled" / "ladders" / f"{TAG}_t{trial}" / "gates" / f"cap{cap}" / f"{OBJ}__{KEY}.decision.json"


def test_a_record_that_used_the_sibling_names_its_generation_and_is_re_derived_with_it(tmp_path):
    root = tmp_path / "w"; t0, t1 = _two_trials(root)
    p = _gate_record(root, 2048, "exit_le_1pct", cell=t0, n=ROWS, sibling_root=str(t1.parents[1]))
    rec = json.loads(p.read_text())
    # the producer (ladder_decide._sibling_controls) records the sibling GENERATION: item count, cap, content digest — the same store_digest the closer recomputes
    sib_texts = ACC._spool_texts(t1 / "private" / f"gens_{KEY}.jsonl")
    assert rec["sibling"]["status"] == "used" and rec["sibling"]["n_items"] == ROWS and rec["sibling"]["cap"] == 2048 and rec["sibling"]["store"] == str(t1)
    assert rec["sibling"]["store_digest"] == ACC.store_digest(sib_texts) and len(rec["sibling"]["store_digest"]) == 64
    assert rec["counts"]["n_at_cap"] == 6 and rec["counts"]["n_boundary"] == 3 and rec["counts"]["n_truncated"] == 3 and rec["decision"] == "exit_le_1pct"
    # the closer: the gate exited 2048 (the driver's flag says escalate) -> final rung 2048, bound by digest, sidecars validated, re-derived WITH the sibling
    why, facts, store, artifact = RC.inspect_controls(str(root), TAG, 0, OBJ, KEY, planned=250)
    assert why is None, why
    assert facts["final_rung"] == 2048 and facts["decided_by"] == "ladder_gate" and facts["gate_binding"] == "digest" and store == str(t0) and artifact.endswith("cap_ladder.json")
    assert facts["sidecars"] == {"n_total": ROWS, "n_at_cap": 6} and facts["sibling_state"] == "used" and facts["sibling_binding"] == "digest"
    assert facts["sibling_digest"] == ACC.store_digest(sib_texts)[:16] and facts["gate_rederived"] == "reproduced"
    assert facts["rederived"] == {"decision": "exit_le_1pct", "n_responses": ROWS, "n_at_cap": 6, "n_loop": 0, "n_repetitive": 0, "n_boundary": 3, "n_boundary_any": 3,
                                  "n_truncated": 3, "n_sibling_unreadable": 0}
    assert facts["classifier_sources"] == {"rep_ratio": "the record's stamped path", "degeneracy": "the record's stamped path"}
    # the SIBLING regenerated after the gate ruled (same count, new texts): refused by name on the sibling digest — never re-read as if it were the one paired
    _spool(t1, ROWS, cap=2048, at_cap=6, terminated=3, salt=" regenerated")
    why, facts, _s, _a = RC.inspect_controls(str(root), TAG, 0, OBJ, KEY, planned=250)
    assert why and "is STALE on its sibling" in why and "a regenerated sibling" in why and "re-run the ladder gate at this rung" in why, why
    assert "final_rung" not in facts and facts["gate_rung"] == 2048 and facts["sibling_state"] == "used" and "gate_rederived" not in facts
    # a record BEFORE closer-fix-3 (no sibling digest) is held by the re-derivation itself: the regenerated sibling (same shape) reproduces the counts, so it binds as declared-state …
    d = json.loads(p.read_text()); d["sibling"].pop("store_digest"); p.write_text(json.dumps(d))
    why, facts, _s, _a = RC.inspect_controls(str(root), TAG, 0, OBJ, KEY, planned=250)
    assert why is None and facts["sibling_binding"] == "declared-state" and facts["gate_rederived"] == "reproduced", (why, facts)
    # … and a sibling whose terminations CHANGED does not reproduce the record's boundary calls -> refused by name, naming the differing counts
    _spool(t1, ROWS, cap=2048, at_cap=6, terminated=1)
    why, facts, _s, _a = RC.inspect_controls(str(root), TAG, 0, OBJ, KEY, planned=250)
    assert why and "is not reproduced from the store under its declared sibling state (used)" in why and "'n_boundary': (3, 1)" in why and "'n_truncated': (3, 5)" in why, why
    assert facts["rederived"]["decision"] == "escalate" and "final_rung" not in facts
    # the sibling gone entirely: the record declared it USED -> refused by name, never silently re-derived without it
    shutil.rmtree(t1)
    why, facts, _s, _a = RC.inspect_controls(str(root), TAG, 0, OBJ, KEY, planned=250)
    assert why and "declares the sibling store under" in why and "USED, but it now reads" in why and "re-run the ladder gate" in why, why


def test_a_record_written_while_the_sibling_was_absent_is_re_derived_under_that_declared_state(tmp_path):
    """The drift: the gate ruled with the sibling UNAVAILABLE (no boundary calls); the sibling arriving later must not move the record."""
    root = tmp_path / "w"; t0, t1 = _two_trials(root)
    sib_root = t1.parents[1]; hidden = root / "hidden_t1"; sib_root.rename(hidden)                       # the sibling root absent at gate time
    p = _gate_record(root, 2048, "escalate", cell=t0, n=ROWS, sibling_root=str(sib_root))                 # unavailable: 6/450 truncated -> escalate
    rec = json.loads(p.read_text())
    assert rec["sibling"]["status"] == "unavailable" and "store_digest" not in rec["sibling"] and rec["counts"]["n_truncated"] == 6 and rec["decision"] == "escalate"
    hidden.rename(sib_root)                                                                                 # the sibling arrives later (3 of the 6 would be boundary)
    texts = ACC._spool_texts(t0 / "private" / f"gens_{KEY}.jsonl")
    why, facts, t = RC.gate_binding(rec, str(p), cell=str(t0), obj=OBJ, key=KEY, cap=2048)
    assert why is None and t == texts, why
    assert facts["gate_binding"] == "digest" and facts["sibling_state"] == "unavailable" and "sibling_binding" not in facts and "sibling_digest" not in facts
    assert facts["gate_rederived"] == "reproduced" and facts["rederived"]["n_truncated"] == 6 and facts["rederived"]["n_boundary"] == 0 and facts["rederived"]["decision"] == "escalate"
    # (had the closer re-read the present sibling it would derive exit_le_1pct with 3 truncated and call the record stale: the DECLARED state rules)
    # a record whose counts were edited is not reproduced — by name, with the differing counts
    d = json.loads(p.read_text()); d["counts"]["n_truncated"] = 5; p.write_text(json.dumps(d))
    why, facts, _t = RC.gate_binding(d, str(p), cell=str(t0), obj=OBJ, key=KEY, cap=2048)
    assert why and "is not reproduced from the store under its declared sibling state (unavailable)" in why and "'n_truncated': (5, 6)" in why, why
    assert facts["rederived"]["n_truncated"] == 6 and "gate_rederived" not in facts
    # a record whose DECISION was edited (counts intact) is not reproduced either
    d = json.loads(p.read_text()); d["counts"]["n_truncated"] = 6; d["decision"] = "exit_le_1pct"; p.write_text(json.dumps(d))
    why, facts, _t = RC.gate_binding(d, str(p), cell=str(t0), obj=OBJ, key=KEY, cap=2048)
    assert why and "record exit_le_1pct, re-derived escalate" in why and "counts differing (record, re-derived): none" in why, why
    # not_given binds the same way (no boundary calls)
    d["decision"] = "escalate"; d["sibling"] = {"status": "not_given"}; p.write_text(json.dumps(d))
    why, facts, _t = RC.gate_binding(d, str(p), cell=str(t0), obj=OBJ, key=KEY, cap=2048)
    assert why is None and facts["sibling_state"] == "not_given" and facts["gate_rederived"] == "reproduced", why


def test_records_missing_the_blocks_the_rederivation_needs_and_unvalidated_stores_refuse_by_name(tmp_path):
    root = _world(tmp_path / "w", rungs={512: True, 2048: False}, gate={512: "escalate", 2048: "exit_le_1pct"})     # the driver's False flag WOULD accept
    cell = _cell_dir(root, 2048); p = _gate_path(root, 2048); base = json.loads(p.read_text())
    why, facts, _s, _a = RC.inspect_controls(str(root), TAG, 0, OBJ, KEY, planned=250)
    assert why is None and facts["final_rung"] == 2048 and facts["gate_rederived"] == "reproduced" and facts["sibling_state"] == "not_given", why
    for drop, frag in (("sibling", "carries no sibling block"), ("rules", "carries no rules block"), ("ladder", "carries no ladder block"),
                       ("rep_ratio", "carries no rep_ratio block"), ("degeneracy", "carries no degeneracy block")):
        d = json.loads(json.dumps(base)); d.pop(drop); p.write_text(json.dumps(d))
        why, facts, _s, _a = RC.inspect_controls(str(root), TAG, 0, OBJ, KEY, planned=250)
        assert why and frag in why and "cannot be re-derived" in why and "final_rung" not in facts, (drop, why)
    d = json.loads(json.dumps(base)); d.pop("counts"); p.write_text(json.dumps(d))                 # no counts: the count check (closer-fix-2) refuses first, by name
    why, facts, _s, _a = RC.inspect_controls(str(root), TAG, 0, OBJ, KEY, planned=250)
    assert why and "counts None response(s), the spool holds 450" in why and "final_rung" not in facts, why
    d = json.loads(json.dumps(base)); d["sibling"] = {"status": "maybe"}; p.write_text(json.dumps(d))
    why, facts, _s, _a = RC.inspect_controls(str(root), TAG, 0, OBJ, KEY, planned=250)
    assert why and "declares no sibling state (sibling.status='maybe'" in why, why
    d = json.loads(json.dumps(base)); d["rules"].pop("boundary_fraction"); p.write_text(json.dumps(d))
    why, facts, _s, _a = RC.inspect_controls(str(root), TAG, 0, OBJ, KEY, planned=250)
    assert why and "rules carry no boundary_fraction" in why, why
    # the classifier sources: a stamped sha no readable file carries -> refused; a stamped path gone but this checkout's copy hashing the same -> that copy
    d = json.loads(json.dumps(base)); d["rep_ratio"]["source_sha256"] = "0" * 64; p.write_text(json.dumps(d))
    why, facts, _s, _a = RC.inspect_controls(str(root), TAG, 0, OBJ, KEY, planned=250)
    assert why and "the rep_ratio source the record was decided with (000000000000…) is not readable here (tried" in why and "cannot be re-derived" in why, why
    d = json.loads(json.dumps(base)); d["degeneracy"]["source"] = str(tmp_path / "gone" / "degeneracy.py"); p.write_text(json.dumps(d))
    why, facts, _s, _a = RC.inspect_controls(str(root), TAG, 0, OBJ, KEY, planned=250)
    assert why is None and facts["classifier_sources"] == {"rep_ratio": "the record's stamped path", "degeneracy": "this checkout's copy"}, (why, facts)
    p.write_text(json.dumps(base))
    # the store's sidecars (the record intact): the meta sidecar gone -> refused; gen_stats disagreeing with the sidecar -> refused; a cap not the rung's -> refused
    meta = cell / "private" / f"gens_{KEY}{ACC.META_SUFFIX}"; keep_meta = meta.read_text(); meta.unlink()
    why, facts, _s, _a = RC.inspect_controls(str(root), TAG, 0, OBJ, KEY, planned=250)
    assert why and "the store's sidecars do not validate" in why and "no meta sidecar" in why and "final_rung" not in facts, why
    meta.write_text(keep_meta)
    gs = cell / "gen_stats.json"; keep_gs = gs.read_text(); g = json.loads(keep_gs); g[KEY]["n_at_cap"] = 7; gs.write_text(json.dumps(g))
    why, facts, _s, _a = RC.inspect_controls(str(root), TAG, 0, OBJ, KEY, planned=250)
    assert why and "the store's sidecars do not validate" in why and "inconsistent store" in why, why
    g = json.loads(keep_gs); g[KEY]["cap"] = 8192; gs.write_text(json.dumps(g))
    why, facts, _s, _a = RC.inspect_controls(str(root), TAG, 0, OBJ, KEY, planned=250)
    assert why and "gen_stats.json records cap 8192" in why and "the rung is 2048" in why, why
    g = json.loads(keep_gs); g[KEY]["spool"] = str(cell / "private" / f"gens_{KEY}_v9.jsonl"); gs.write_text(json.dumps(g))
    why, facts, _s, _a = RC.inspect_controls(str(root), TAG, 0, OBJ, KEY, planned=250)
    assert why and "the store's sidecars do not validate" in why, why          # read_control reads the spool gen_stats names: absent -> refused before the name check
    gs.write_text(keep_gs)
    why, facts, _s, _a = RC.inspect_controls(str(root), TAG, 0, OBJ, KEY, planned=250)
    assert why is None and facts["final_rung"] == 2048, why
