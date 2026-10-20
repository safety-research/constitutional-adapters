"""closer-gate (controls_t0 833935 / controls_t1 833936 exit 75, 2026-10-02): the full22 closer picked a controls run's FINAL rung as "the
highest controls_cap<C> whose cap_ladder.json shows … escalate: false" — the DRIVER's record (every at-cap item counts, boundary noise
included: mix-psr/xstest @2048 read n_truncated 5 / n_boundary 0 / escalate True) — while the ladder's ONE BOUND DECISION
(sampled/ladder_gate.py; the driver's reading is "recorded beside, never a second decider") EXITED the pair at 2048 (truncated 2/450 <= 1 %,
3 boundary). Two complete stores were refused by reading the wrong record. Pinned here on a synthetic wave root: a cell the gate exited at
2048 whose driver flag says escalate registers at 2048 (decided_by ladder_gate); a rung with no gate record falls back to cap_ladder.json's
escalate: false (the pre-gate reading, unchanged); a gate exit on a cell without a cap_ladder record is a refusal by name; the no-rung
refusal names both readings. Synthetic tag / cell ids only."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from front_door_testlib import REPO  # noqa: E402

sys.path.insert(0, str(REPO / "scripts" / "sdfbase" / "full22"))
sys.path.insert(0, str(REPO / "scripts" / "sdfbase" / "full22" / "sampled"))
import at_cap_controls as ACC  # noqa: E402
import ladder_decide as LD  # noqa: E402
import registry_close_stores as RC  # noqa: E402

TAG = "synthmodel_s9r"
OBJ = "cellA-full__psr_ce_all"          # the registration's object spelling (`__`); the cell dir spells it `--`
KEY = "xstest"
#: closer-fix-3: the gate's thresholds and classifier sources as ladder_decide stamps them — every record here is written by the REAL decide_controls
THRESHOLDS = {"trigger_share": 0.01, "loop_rep_ratio_threshold": 0.5, "repetitive_rep_ratio_min": 0.2, "tail_words": 200, "boundary_fraction": 0.5}
REP_SRC = REPO / "src" / "sisteer" / "capeval" / "tiers" / "cap_ladder.py"
DEG_SRC = REPO / "src" / "sisteer" / "eval" / "degeneracy.py"


def _cell_dir(root: Path, cap: int, trial: int = 0) -> Path:
    return root / "sampled" / f"controls_cap{cap}" / f"{TAG}_t{trial}" / "cells" / f"{OBJ.replace('__', '--')}__controls__t0__a0__deadbeef__{cap:08x}"


def _at_cap_for(decision, rows: int) -> int:
    """How many spooled items are at cap so the REAL gate decides `decision` over `rows` coherent texts: > 1 % truncated escalates (exit_ceiling at
    the ladder's top rung), <= 1 % exits (450 rows -> 2 at cap; 100 -> 0)."""
    return max(2, rows // 11) if decision in ("escalate", "exit_ceiling") else rows // 225


def _world(root: Path, *, rungs: dict, gate: dict | None, planned: int = 250, floor: int = 200, spool_rows: int | None = None, trial: int = 0):
    """rungs: {cap: escalate flag} for the driver's cap_ladder.json; gate: {cap: decision} for the ladder gate's decision.json (None = no gate records).
    closer-fix-3: every cell is a COMPLETE controls store (manifest, spool, meta sidecar, gen_stats) and every gate record is written by the real
    ladder_decide.decide_controls over it, so the closer's sidecar validation and re-derivation hold against what the producer writes."""
    n = planned + floor
    for cap, esc in rungs.items():
        cell = _cell_dir(root, cap, trial)
        (cell / "private").mkdir(parents=True)
        (cell / "manifest.json").write_text(json.dumps({"cell_id": OBJ, "intervention_hash": "ii" * 8}))
        (cell / "cap_ladder.json").write_text(json.dumps({"controls": {KEY: {"cap": cap, "escalate": esc, "exit": ("escalate" if esc else "exit_le_1pct"), "n": n,
                                                                              "n_at_cap": (5 if esc else 0), "n_truncated": (5 if esc else 0), "n_boundary": 0}}}))
        rows = n if spool_rows is None else spool_rows
        _spool(cell, rows, cap=cap, at_cap=_at_cap_for((gate or {}).get(cap), rows))
    if gate is not None:
        for cap, decision in gate.items():
            _gate_record(root, cap, decision, cell=_cell_dir(root, cap, trial), n=(n if spool_rows is None else spool_rows), trial=trial)
    return root


def _spool(cell: Path, rows: int, *, salt: str = "", cap: int = 2048, at_cap: int = 0, terminated: int = 0) -> Path:
    """The producer's COMPLETE control store for KEY: the spool (a header line, then one {item_id, text} row per item — at_cap_controls._spool_texts
    reads exactly this), the meta sidecar bound to it by text_sha256 (items [0, terminated) stop at 100 tokens — a SIBLING's shape, boundary for
    the other trial's at-cap items; items [terminated, at_cap) are at cap: finish_reason length, completion_tokens = cap; the rest stop at 40) and
    gen_stats.json agreeing with the sidecar."""
    p = cell / "private" / f"gens_{KEY}.jsonl"
    texts = {f"i{i:04d}": f"text {i}{salt}" for i in range(rows)}
    p.write_text(json.dumps({"__spool_header__": True, "control": KEY}) + "\n" + "".join(json.dumps({"item_id": i, "text": t}) + "\n" for i, t in texts.items()))
    meta = []
    for k, (iid, t) in enumerate(texts.items()):
        fr, ct, capped = ("stop", 100, False) if k < terminated else (("length", cap, True) if k < at_cap else ("stop", 40, False))
        meta.append({"item_id": iid, "text_sha256": ACC._text_sha(t), "at_cap": capped, "finish_reason": fr, "completion_tokens": ct, "max_tokens": cap,
                     "source_cap": cap, "prompt_tokens": 9, "sampling_seed": 1, "regenerated": False, "floor_sample": False})
    (cell / "private" / f"gens_{KEY}{ACC.META_SUFFIX}").write_text("".join(json.dumps(m) + "\n" for m in meta))
    ids_at_cap = [m["item_id"] for m in meta if m["at_cap"]]
    (cell / "gen_stats.json").write_text(json.dumps({KEY: {"control": KEY, "n_total": rows, "spool": str(p), "cap": cap, "decode": {"do_sample": True, "cap": cap, "temperature": 0.7},
                                                           "n_at_cap": len(ids_at_cap), "at_cap_item_ids": ids_at_cap}}))
    return p


def _gate_record(root: Path, cap: int, decision: str, *, cell: Path, n: int, identity: dict | None = None, written_at: str | None = None, trial: int = 0,
                 sibling_root: str | None = None, **over) -> Path:
    """The ladder gate's decision record as ladder_decide writes it — the REAL decide_controls over the cell (closer-fix-3: kind, identity with the
    store digest, rules, ladder, classifier sources, sibling state, counts), asserted to decide `decision`; `identity` / `written_at` / `over` then
    overwrite fields to shape a stale, unbound or foreign record. `n` is the spool's row count (informational: the record counts the spool)."""
    rep_fn, rep_info = LD.extract_rep_ratio(REP_SRC); deg_fn, deg_info = LD.load_degeneracy(DEG_SRC)
    g = root / "sampled" / "ladders" / f"{TAG}_t{trial}" / "gates" / f"cap{cap}"; g.mkdir(parents=True, exist_ok=True)
    p = g / f"{OBJ}__{KEY}.decision.json"
    rec = LD.decide_controls(out_root=cell.parents[1], cell=OBJ, control=KEY, cap=cap, sibling_root=sibling_root, thresholds=dict(THRESHOLDS), rep_ratio=rep_fn,
                             rep_info=rep_info, degeneracy=deg_fn, deg_info=deg_info, out=p, items_out=None, argv=["decide", "--cell", OBJ], stdout=False)
    assert rec["decision"] == decision, (decision, rec["decision"], rec["counts"])
    d = json.loads(p.read_text())
    if identity is not None:
        d["identity"] = identity
    if written_at is not None:
        d["tool"]["written_at"] = written_at
    d.update(over)
    p.write_text(json.dumps(d)); return p


def test_a_gate_exited_cell_registers_at_that_rung_although_the_driver_flag_says_escalate(tmp_path):
    root = _world(tmp_path / "w", rungs={512: True, 2048: True}, gate={512: "escalate", 2048: "exit_le_1pct"})
    why, facts, store, artifact = RC.inspect_controls(str(root), TAG, 0, OBJ, KEY, planned=250)
    assert why is None, why
    assert facts["final_rung"] == 2048 and facts["decided_by"] == "ladder_gate" and facts["exit"] == "exit_le_1pct" and facts["n_control"] == 450
    assert store.endswith("controls_cap2048/synthmodel_s9r_t0/cells/cellA-full--psr_ce_all__controls__t0__a0__deadbeef__00000800") and artifact.endswith("cap_ladder.json")
    assert facts["ladders"][2048] == {"escalate": True, "exit": "escalate", "n": 450, "gate": "exit_le_1pct"}
    assert 512 not in facts["ladders"]                      # the walk runs from the top rung down and stops at the final one; 512 is never read
    # the gate's exit_ceiling is an exit too
    root2 = _world(tmp_path / "w2", rungs={512: True, 2048: True, 16384: True}, gate={512: "escalate", 2048: "escalate", 16384: "exit_ceiling"})   # closer-fix-3: exit_ceiling is the ladder's TOP rung
    why, facts, _s, _a = RC.inspect_controls(str(root2), TAG, 0, OBJ, KEY, planned=250)
    assert why is None and facts["final_rung"] == 16384 and facts["decided_by"] == "ladder_gate"


def test_a_rung_with_no_gate_record_falls_back_to_the_drivers_flag_unchanged(tmp_path):
    # the pre-gate reading: escalate False at 2048 -> final rung 2048; escalate True everywhere -> refused, naming both readings
    root = _world(tmp_path / "w", rungs={512: True, 2048: False}, gate=None)
    why, facts, _s, _a = RC.inspect_controls(str(root), TAG, 0, OBJ, KEY, planned=250)
    assert why is None and facts["final_rung"] == 2048 and facts["decided_by"] == "cap_ladder" and facts["ladders"][2048]["gate"] is None
    root2 = _world(tmp_path / "w2", rungs={512: True, 2048: True}, gate=None)
    why, facts, _s, _a = RC.inspect_controls(str(root2), TAG, 0, OBJ, KEY, planned=250)
    assert why and "exited (the gate's decision where one exists, else cap_ladder.json escalate=false)" in why and "no rung of synthmodel_s9r_t0" in why


def test_the_gate_still_escalating_at_the_top_rung_refuses_and_the_spool_is_still_checked(tmp_path):
    root = _world(tmp_path / "w", rungs={512: True, 2048: True}, gate={512: "escalate", 2048: "escalate"})
    why, facts, _s, _a = RC.inspect_controls(str(root), TAG, 0, OBJ, KEY, planned=250)
    assert why and "no rung of synthmodel_s9r_t0" in why                      # the gate agrees with the driver: nothing final yet
    short = _world(tmp_path / "w2", rungs={512: True, 2048: True}, gate={512: "escalate", 2048: "exit_le_1pct"}, spool_rows=100)
    why, facts, _s, _a = RC.inspect_controls(str(short), TAG, 0, OBJ, KEY, planned=250)
    assert why and "100 generated row(s)" in why                              # the final rung's spool must still hold the control's rows


def test_a_gate_exit_on_a_cell_without_a_driver_record_is_a_refusal_by_name(tmp_path):
    root = _world(tmp_path / "w", rungs={512: True, 2048: True}, gate={512: "escalate", 2048: "exit_le_1pct"})
    cell = next(p for p in (root / "sampled" / "controls_cap2048" / f"{TAG}_t0" / "cells").iterdir())
    (cell / "cap_ladder.json").write_text(json.dumps({"controls": {}}))
    why, facts, _s, _a = RC.inspect_controls(str(root), TAG, 0, OBJ, KEY, planned=250)
    assert why and "the gate exited xstest at 2048 but the cell carries no cap_ladder.json controls.xstest record" in why
