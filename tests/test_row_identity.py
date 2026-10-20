"""ifeval-canonical-battery — the per-row OBJECT IDENTITY: the cross-eval
comparability guard, binding.

Every emitted row of a ``--bank-view`` run — per-sample, mean and aggregate —
carries ``object_artifact_sha256`` / ``route_kind`` / ``view`` /
``producer_tree_sha`` beside the existing ``thinking`` and ``sampling_ref`` plus
the new ``sampling_mode``; a fitted cell's sha equals the roster pin, a bare /
prompted cell's is a STATED null; a run without ``--bank-view`` emits
byte-identical rows; the gate refuses a row missing any key.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Dict

import pytest

from sisteer.capeval.tinycap import contracts as C
from sisteer.capeval.tiers import inspect_served as isv

from test_inspect_served import _Emitter, _Log, _Sample, _Score, _Store, _fake_eval, _plan
from test_inspect_served_ifeval import _IfSample, _VALUE, _ifeval_plan

IFEVAL = C.inspect_task(C.IFEVAL_TASK_NAME)
SHA = "ab" * 32
TREE = "c" * 40


def test_the_vocabulary_is_the_trees_existing_names_lifted_into_the_contracts():
    assert C.ROW_IDENTITY_FIELDS == ("object_artifact_sha256", "route_kind", "view", "producer_tree_sha")
    assert C.ROW_IDENTITY_KEY_OBJECT_SHA256 == C.ENDPOINT_KEY_OBJECT_ARTIFACT_SHA256   # the standup record's name
    assert C.ROW_IDENTITY_KEY_PRODUCER_TREE_SHA in C.TASK_OUTCOMES_HEADER_FIELDS         # the outcome header's name
    assert C.ROW_SAMPLING_MODE_FIELD == "sampling_mode"
    assert C.ROW_GUARD_FIELDS == (*C.ROW_IDENTITY_FIELDS, "thinking", "sampling_mode")
    assert C.ROUTE_KINDS == ("base", "prompted", "lens") and C.BANK_VIEWS == ("main", "it")
    assert C.FLAG_BANK_VIEW == "--bank-view" and C.MANIFEST_ROUTE_KEY_BANK_VIEW == "bank_view"
    assert C.ROW_THINKING_VALUES == ("on", "off", "not_available") and C.THINKING_NOT_AVAILABLE == "not_available"
    # route_kind: the GSM8K launcher's rule, lifted (g8k_cells.route_kind)
    from sisteer.capeval.tinycap import contracts as _c
    assert _c.route_kind_of_cell("none") == "base"
    assert _c.route_kind_of_cell("prompted__safety_refusal") == "prompted"
    for cell in ("constv2-mix-full__psr_ce_all", "safety_refusal__lora_r4", "safety_refusal__psr_kl_all@thinking",
                 "constv2-mix-full__psr_ce_all_minus_ctl_distilled"):
        assert _c.route_kind_of_cell(cell) == "lens", cell
    import importlib.util
    g8k = Path(__file__).resolve().parents[1] / "scripts" / "sdfbase" / "gsm8k" / "g8k_cells.py"
    if g8k.is_file():
        spec = importlib.util.spec_from_file_location("g8k_cells_probe", g8k)
        mod = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(mod)
        except Exception:  # noqa: BLE001 — the GSM8K pipeline's module may need its own imports
            mod = None
        if mod is not None:
            for cell in ("none", "prompted__safety_refusal", "constv2-mix-full__psr_ce_all"):
                assert mod.route_kind(cell) == _c.route_kind_of_cell(cell)


def test_the_block_is_validated_a_lens_cell_needs_the_pin_and_a_bare_cell_stamps_a_stated_null():
    fitted = C.row_identity_block(cell="constv2-mix-full__psr_ce_all", view="main", object_artifact_sha256=SHA,
                                  producer_tree_sha=TREE, sampling_mode="vendor_preset")
    assert fitted == {"object_artifact_sha256": SHA, "route_kind": "lens", "view": "main",
                      "producer_tree_sha": TREE, "sampling_mode": "vendor_preset"}
    assert tuple(fitted) == (*C.ROW_IDENTITY_FIELDS, C.ROW_SAMPLING_MODE_FIELD)
    bare = C.row_identity_block(cell="none", view="it", object_artifact_sha256=None, producer_tree_sha=TREE,
                                sampling_mode="vendor_preset")
    assert bare["object_artifact_sha256"] is None and bare["route_kind"] == "base" and bare["view"] == "it"
    prompted = C.row_identity_block(cell="prompted__safety_refusal", view="it", object_artifact_sha256=None,
                                    producer_tree_sha=TREE, sampling_mode="vendor_preset")
    assert prompted["route_kind"] == "prompted" and "object_artifact_sha256" in prompted   # stated, never absent
    with pytest.raises(C.RowIdentityError, match="is a lens route but no 64-hex object sha"):
        C.row_identity_block(cell="constv2-mix-full__psr_ce_all", view="main", object_artifact_sha256=None,
                             producer_tree_sha=TREE, sampling_mode="vendor_preset")
    with pytest.raises(C.RowIdentityError, match="is a base route"):
        C.row_identity_block(cell="none", view="main", object_artifact_sha256=SHA, producer_tree_sha=TREE,
                             sampling_mode="vendor_preset")
    # the known-bad view is one NO roster declares: since 2026-09-18 the vocabulary is
    # `main` | `it` + every full22 roster view (contracts.bank_views_admitted — `fullconst`, a roster view, is admitted now)
    with pytest.raises(C.RowIdentityError, match="not one of"):
        C.row_identity_block(cell="none", view="nowhere", object_artifact_sha256=None, producer_tree_sha=TREE,
                             sampling_mode="vendor_preset")
    with pytest.raises(C.RowIdentityError, match="not a 40-hex tree sha"):
        C.row_identity_block(cell="none", view="main", object_artifact_sha256=None, producer_tree_sha=None,
                             sampling_mode="vendor_preset")
    with pytest.raises(ValueError, match="sampling mode must be one of"):
        C.row_identity_block(cell="none", view="main", object_artifact_sha256=None, producer_tree_sha=TREE,
                             sampling_mode="nucleus")


def _ifeval_log(idx: int) -> _Log:
    return _Log([_IfSample("1000", _VALUE), _IfSample("1001", _VALUE)])


def test_every_emitted_row_of_a_bank_view_run_carries_the_six_keys(tmp_path, monkeypatch):
    monkeypatch.setattr("inspect_ai.eval_async", _fake_eval({0: _ifeval_log(0), 1: _ifeval_log(1)}))
    ident = C.row_identity_block(cell="constv2-mix-full__psr_ce_all", view="main", object_artifact_sha256=SHA,
                                 producer_tree_sha=TREE, sampling_mode="vendor_preset")
    em, store = _Emitter(), _Store(tmp_path / "cell")
    asyncio.run(isv.run_one_task(_ifeval_plan(), model=None, store=store, template={}, concurrency=2, emitter=em,
                                 sampling_ref="deadbeef0123", row_identity=ident))
    metrics = {r["metric"] for r in em.rows}
    assert metrics == {"acc@0", "acc@1", C.METRIC_ACC, C.METRIC_ACC_AGGREGATE, C.METRIC_N}
    expected = {**ident, "thinking": "off"}
    for r in em.rows:
        extra = r.get("extra") or {}
        assert extra["object_artifact_sha256"] == SHA and extra["route_kind"] == "lens" and extra["view"] == "main"
        assert extra["producer_tree_sha"] == TREE and extra["sampling_mode"] == "vendor_preset"
        if r["metric"] != C.METRIC_N:      # the `n` aggregate row carries no `thinking` (as before)
            assert C.row_identity_problem(extra, expected=expected) is None, (r["metric"], extra)
        if r["metric"] in ("acc@0", "acc@1", C.METRIC_ACC):   # the item rows carry the sampling digest
            assert extra["sampling_ref"] == "deadbeef0123"
    # the aggregate row's `thinking` is the existing bool; the guard reads the arm off the item rows
    agg = next(r for r in em.rows if r["metric"] == C.METRIC_ACC_AGGREGATE)["extra"]
    assert agg["thinking"] is False and agg["route_kind"] == "lens"


def test_a_bare_and_a_prompted_cell_stamp_a_stated_null_sha_with_their_route_kind(tmp_path, monkeypatch):
    for cell, kind in (("none", "base"), ("prompted__safety_refusal", "prompted")):
        monkeypatch.setattr("inspect_ai.eval_async", _fake_eval({0: _ifeval_log(0), 1: _ifeval_log(1)}))
        ident = C.row_identity_block(cell=cell, view="it", object_artifact_sha256=None, producer_tree_sha=TREE,
                                     sampling_mode="vendor_preset")
        em, store = _Emitter(), _Store(tmp_path / cell)
        asyncio.run(isv.run_one_task(_ifeval_plan(), model=None, store=store, template={}, concurrency=2,
                                     emitter=em, sampling_ref="deadbeef0123", row_identity=ident))
        for r in em.rows:
            extra = r["extra"]
            assert "object_artifact_sha256" in extra and extra["object_artifact_sha256"] is None
            assert extra["route_kind"] == kind and extra["view"] == "it"


def test_a_run_without_bank_view_emits_byte_identical_rows(tmp_path, monkeypatch):
    def run(with_identity: bool):
        monkeypatch.setattr("inspect_ai.eval_async", _fake_eval({0: _ifeval_log(0), 1: _ifeval_log(1)}))
        em, store = _Emitter(), _Store(tmp_path / ("with" if with_identity else "without"))
        kw = {"row_identity": C.row_identity_block(cell="none", view="main", object_artifact_sha256=None,
                                                    producer_tree_sha=TREE, sampling_mode="vendor_preset")} \
            if with_identity else {}
        asyncio.run(isv.run_one_task(_ifeval_plan(), model=None, store=store, template={}, concurrency=2,
                                     emitter=em, sampling_ref="deadbeef0123", **kw))
        return em.rows

    plain = run(False)
    for r in plain:
        assert not set(C.ROW_IDENTITY_FIELDS) & set(r.get("extra") or {})
        assert C.ROW_SAMPLING_MODE_FIELD not in (r.get("extra") or {})
    stamped = run(True)
    def strip(rows):
        out = []
        for r in rows:
            extra = {k: v for k, v in (r.get("extra") or {}).items()
                     if k not in C.ROW_IDENTITY_FIELDS and k != C.ROW_SAMPLING_MODE_FIELD}
            row = {k: v for k, v in r.items() if k not in ("extra", "wall_clock_s")}   # wall clock varies per run
            out.append(json.dumps({**row, "extra": extra}, sort_keys=True, default=str))
        return out
    # the stamped run is the plain run plus exactly the identity keys — nothing else moved
    assert strip(stamped) == strip(plain)
    # a legacy (non-IFEval) task through the same runner: byte-identical, no stamp without the flag
    log = _Log([_Sample("math_aaaa1111", {"s": _Score("C")}), _Sample("math_bbbb2222", {"s": _Score("I")})])
    monkeypatch.setattr("inspect_ai.eval_async", _fake_eval({0: log}))
    em = _Emitter()
    asyncio.run(isv.run_one_task(_plan(primary_scorer=None, max_tokens=64), model=None,
                                 store=_Store(tmp_path / "legacy"), template={}, thinking=C.THINKING_OFF,
                                 concurrency=2, emitter=em))
    assert all(not set(C.ROW_IDENTITY_FIELDS) & set(r.get("extra") or {}) for r in em.rows)


def test_gate_i2_refuses_a_row_missing_any_key_or_carrying_another_object():
    expected = C.row_identity_block(cell="constv2-mix-full__psr_ce_all", view="main", object_artifact_sha256=SHA,
                                    producer_tree_sha=TREE, sampling_mode="vendor_preset")
    expected["thinking"] = "off"
    good = {**expected, "seed": 1, "sampling_ref": "x"}
    assert C.row_identity_problem(good, expected=expected) is None
    for k in C.ROW_GUARD_FIELDS:
        bad = {kk: v for kk, v in good.items() if kk != k}
        assert f"row carries no ['{k}']" in C.row_identity_problem(bad, expected=expected)
    other = {**good, "object_artifact_sha256": "cd" * 32}
    assert "is not the cell's" in C.row_identity_problem(other, expected=expected)
    wrong_view = {**good, "view": "it"}
    assert "row view='it' is not the cell's 'main'" in C.row_identity_problem(wrong_view, expected=expected)
    # the Llama GSM8K rows' `not_available` is admitted by the vocabulary
    na = {**good, "thinking": "not_available"}
    assert C.row_identity_problem(na, expected={k: v for k, v in expected.items() if k != "thinking"}) is None
    assert "outside" in C.row_identity_problem({**good, "thinking": "maybe"},
                                               expected={k: v for k, v in expected.items() if k != "thinking"})


def test_the_driver_builds_the_block_once_and_records_the_view_in_the_route_block():
    src = (Path(__file__).resolve().parents[1] / "scripts" / "served_capability_run.py").read_text()
    # the vocabulary is validated at parse time by `_bank_view_arg` -> contracts.check_bank_view_arg (`main` | `it` + the
    # roster's lane views), where `choices=list(tc.BANK_VIEWS)` used to sit
    assert 'g.add_argument(tc.FLAG_BANK_VIEW, dest="bank_view", default=None, type=_bank_view_arg' in src
    assert "return tc.check_bank_view_arg(value)" in src and "raise argparse.ArgumentTypeError(str(e)) from None" in src
    i = src.index("row_identity = tc.row_identity_block(")
    block = src[i - 200:i + 600]
    assert "object_artifact_sha256=(args.expect_object_sha256 or None) if lens_cell else None" in block
    assert "producer_tree_sha=manifest[tc.MANIFEST_KEY_PRODUCER].get(\"tree_sha\")" in block
    assert "sampling_mode=sampling.mode" in block
    assert "manifest[tc.MANIFEST_KEY_ROUTE][tc.MANIFEST_ROUTE_KEY_BANK_VIEW] = str(args.bank_view)" in src
    assert "row_identity=row_identity," in src                              # handed to the Inspect runner
