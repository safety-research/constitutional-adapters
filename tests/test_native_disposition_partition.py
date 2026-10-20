"""The DISPOSITION and the PARTITION: ONE native standup per view serves
``none``, the prompted cells and every LoRA cell; the cells the engine
cannot serve are skipped BY NAME and recorded; a LoRA cell whose record
the engine cannot judge REFUSES.

Pinned here:

- ``defense_cells.native_serving_disposition`` over every kind the registry
  maps (``registry_adapter.ARM_OBJECT_KINDS``) x every spec shape the ONE
  predicate distinguishes: ``serve`` iff ``native_lora_unservable_reason`` is
  None; ``skip`` iff the KIND has no PEFT form (the stacked wrapper, anything
  outside ``NATIVE_LORA_SERVABLE_KINDS``) — by kind alone, artifact present or
  not — OR the fit of a LoRA kind is POSITION-MASKED (``positions`` present
  and not ``"all"``; earlier a masked fit REFUSED, and
  ``safety_refusal__lora_r4`` on the gemma4e4b / qwen35_4b ``main`` views
  would have taken the whole main-view standup down); ``refuse`` iff the kind
  is servable and the record cannot be judged (``positions`` MISSING, a serve
  declaration, ``circuit_breaker`` ``alpha != r``). The reason is the
  predicate's own sentence; the classification reads the predicate's tagged
  FACT (``UNSERVABLE_FACTS`` / ``NATIVE_SKIP_FACTS``), never the sentence.
- ``registry_adapter.partition_cells_for_native`` over the REAL explicit
  registry on a synthetic bank: order preserved; none / prompted served; a
  PSR cell skipped and recorded (kind + reason); a masked LoRA fit SKIPPED by
  name with its kind and the mode in its reason, the rest of the view served
  (and the skipped cell still refuses at the lora_native backstop if it ever
  reaches it); a mixed main view {none, prompted, all-positions lora,
  response-fit lora, psr} partitions without a raise; a declared variant, a
  ``positions``-less record and a ``circuit_breaker`` with ``alpha != r`` still
  refuse naming the cell, the kind, the reason and ``--serving-path lens``; a
  lora_native-policy source refuses (the backstop would refuse first); a
  registry-absent cell passes through (ABSENT downstream, as today); a stacked
  id is skipped without consulting the source; a duplicate refuses.
- ``DEFAULT_STANDUP_SERVING_PATH == "lora_native"`` while the DATACLASS default
  stays ``"lens"``; the ``SkippedCell`` shape.

CPU-only; synthetic artifacts; no vLLM, no model loads."""

from __future__ import annotations

import dataclasses
import inspect

import pytest

from sisteer.capeval import manifest as cm
from sisteer.capeval.petri import defense_cells as dc
from sisteer.capeval.petri import registry_adapter as ra
from sisteer.capeval.serving import serve_declaration as sd

from test_lora_native_enumeration import (  # the synthetic bank + real explicit registry
    H_LORA, H_PSR, H_SUB, TT_LORA, TT_PSR, TT_SUB, _bank, _explicit_source)

# ---------------------------------------------------------------------------
# 1. the disposition table: every kind x every spec shape
# ---------------------------------------------------------------------------

_ALL = {"positions": "all", "response_header_len": None, "layers": [0, 2], "rank": 4}
_RESPONSE = {"positions": "response", "response_header_len": 3, "layers": [0, 2], "rank": 4}
_LAST = {"positions": "last", "response_header_len": None, "layers": [0, 2], "rank": 4}
_NO_POSITIONS = {"layers": [0, 2], "rank": 4}
_CB_UNIT = {"rank": 4, "lora_alpha": 4, "positions": "all"}
_CB_FOLDED = {"rank": 4, "lora_alpha": 8, "positions": "all"}
_CB_NO_SCALING = {"positions": "all"}

SHAPES = {
    "none": None,
    "all": _ALL,
    "response": _RESPONSE,
    "last": _LAST,
    "no_positions": _NO_POSITIONS,
    "declared_thinking": sd.apply_variant(dict(_ALL), sd.VARIANT_THINKING),
    "declared_all_positions": sd.apply_variant(dict(_RESPONSE), sd.VARIANT_ALL_POSITIONS),
    "cb_unit": _CB_UNIT,
    "cb_folded": _CB_FOLDED,
    "cb_no_scaling": _CB_NO_SCALING,
}


def _is_masked_fit_leg(kind, spec) -> bool:
    """The skip rule restated INDEPENDENTLY of the implementation: a
    natively servable LoRA kind (not circuit_breaker, not a residual-slot kind)
    whose spec carries no serve declaration and a ``positions`` that is PRESENT
    and not ``"all"``."""
    return (spec is not None and kind in dc.NATIVE_LORA_SERVABLE_KINDS
            and kind != cm.PETRI_STACKED_KIND and kind != "circuit_breaker"
            and kind not in dc.RESIDUAL_SLOT_KINDS
            and not sd.declarations_of(spec)
            and spec.get("positions") is not None and spec.get("positions") != "all")


def _residual_stage2(kind, spec) -> bool:
    """native-psr-serving §1.1 restated independently: a residual-slot kind whose
    record carries no serve declaration and a NATIVE position mode other than
    ``all``. Since step 7 (``NATIVE_STAGE2_ARMED``) such a cell is SERVED, not
    skipped — the per-token position class serves every mode the lens names — so
    this predicate now names the shapes that must serve. The table's shapes carry
    only ``positions``, so the mode here is the record's ``positions`` (default
    ``all``) when it is a native mode; ``static_r`` never reads ``positions``
    except as the cross-phase ``response``."""
    if spec is None or kind not in dc.RESIDUAL_SLOT_KINDS or sd.declarations_of(spec):
        return False
    if kind == "static_r":
        return True                      # prefill_all / prompt_last / response: never ``all``
    return spec.get("positions", "all") in ("response",)


def _residual_unjudgeable(kind, spec) -> bool:
    """A residual-slot record the mode predicate cannot judge: a ``positions``
    outside the native vocabulary (``last``), or — ARMED — ``response`` without a
    header width (the served region's boundary is the fit's own and is never
    guessed); ``static_r`` with a stated ``positions`` other than ``response``."""
    if spec is None or kind not in dc.RESIDUAL_SLOT_KINDS or sd.declarations_of(spec):
        return False
    pos = spec.get("positions", "all")
    if kind == "static_r":
        return "positions" in spec and pos != "response"     # a stated positions other than response
    if pos == "response":
        return dc.NATIVE_STAGE2_ARMED and spec.get("response_header_len") is None
    return pos != "all"                                       # not a native mode at all

#: every kind the registry maps, every natively servable kind (circuit_breaker
#: is a RETIRED arm — no registry mapping — but an exportable kind), plus the
#: stacked wrapper kind
KINDS = sorted(set(ra.ARM_OBJECT_KINDS.values()) | set(dc.NATIVE_LORA_SERVABLE_KINDS)
               | {cm.PETRI_STACKED_KIND})


def test_vocabularies():
    assert dc.NATIVE_DISPOSITIONS == ("serve", "skip", "refuse")
    assert (dc.NATIVE_DISPOSITION_SERVE, dc.NATIVE_DISPOSITION_SKIP,
            dc.NATIVE_DISPOSITION_REFUSE) == dc.NATIVE_DISPOSITIONS
    assert set(KINDS) >= {"psr", "lora_all", "lora_subtraction", "lora", "vector_all",
                          "loreft", "circuit_breaker", "alphasteer", cm.PETRI_STACKED_KIND}


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("shape", sorted(SHAPES))
def test_disposition_is_the_predicates_verdict_classified_by_kind(kind, shape):
    spec = SHAPES[shape]
    disposition, reason = dc.native_serving_disposition(kind, spec)
    predicate = dc.native_lora_unservable_reason(kind, spec)
    assert disposition in dc.NATIVE_DISPOSITIONS
    if predicate is None:
        assert (disposition, reason) == ("serve", None)
    else:
        assert reason == predicate      # the ONE predicate's own sentence, quotable
        if kind == cm.PETRI_STACKED_KIND or kind not in dc.NATIVE_LORA_SERVABLE_KINDS:
            assert disposition == "skip"
        elif _is_masked_fit_leg(kind, spec):
            assert disposition == "skip"          # A position-masked fit leg is skipped by name
            assert "position-masked" in reason and f"positions={spec['positions']!r}" in reason
        elif _residual_unjudgeable(kind, spec):
            assert disposition == "refuse" and "cannot be judged" in reason
        elif _residual_stage2(kind, spec) and not dc.NATIVE_STAGE2_ARMED:
            assert disposition == "skip"          # the stage-1 world: a stage-2 mode, by name
            assert "native position mode" in reason and "stage-2" in reason
        else:
            assert disposition == "refuse"


def test_the_classification_reads_the_predicates_tagged_fact_not_its_sentence():
    """``_native_lora_verdict`` tags WHICH rule refused; the public predicate is
    its first element; the disposition skips exactly ``NATIVE_SKIP_FACTS``."""
    assert set(dc.NATIVE_SKIP_FACTS) == {"stacked", "kind", "positions_masked", "position_mode_stage2"}
    assert set(dc.NATIVE_SKIP_FACTS) < set(dc.UNSERVABLE_FACTS)
    assert set(dc.UNSERVABLE_FACTS) == {"stacked", "kind", "serve_declaration", "lora_scaling",
                                        "positions_missing", "positions_masked", "residual_record",
                                        "position_mode_stage2"}
    expected = {
        (cm.PETRI_STACKED_KIND, "none"): "stacked",
        ("loreft", "response"): "kind",
        # native-psr-serving step 7: ("psr", "response") SERVES now — the table's `response`
        # shape carries its header width, and the position class serves the mode. What a
        # residual record still refuses for is being unjudgeable:
        ("psr", "last"): "residual_record",                 # not a native mode
        ("static_r", "all"): "residual_record",             # a stated positions other than response
        ("lora", "response"): "positions_masked",
        ("lora_all", "last"): "positions_masked",
        ("lora_subtraction", "response"): "positions_masked",
        ("lora", "no_positions"): "positions_missing",
        ("lora_all", "declared_thinking"): "serve_declaration",
        ("lora_all", "declared_all_positions"): "serve_declaration",
        ("circuit_breaker", "cb_folded"): "lora_scaling",
        ("circuit_breaker", "cb_no_scaling"): "lora_scaling",
    }
    for (kind, shape), fact in expected.items():
        reason, got = dc._native_lora_verdict(kind, SHAPES[shape])
        assert got == fact and reason, (kind, shape, got)
        assert dc.native_lora_unservable_reason(kind, SHAPES[shape]) == reason
        assert dc.native_serving_disposition(kind, SHAPES[shape])[0] == (
            "skip" if fact in dc.NATIVE_SKIP_FACTS else "refuse"), (kind, shape)
    for kind, shape in (("lora", "all"), ("lora", "none"), ("circuit_breaker", "cb_unit"),
                        ("psr", "all"), ("psr", "none"), ("psr", "no_positions"),
                        # armed (step 7): the response-fit residual shapes serve
                        ("psr", "response"), ("diffmean", "response"), ("vector_all", "response"),
                        ("vector_all", "all"), ("diffmean", "all")):
        assert dc._native_lora_verdict(kind, SHAPES[shape]) == (None, None), (kind, shape)
    # circuit_breaker is all-position by construction: its `positions` field is never read
    assert dc._native_lora_verdict("circuit_breaker", {**_CB_UNIT, "positions": "response"}) == (None, None)


def test_the_table_by_hand():
    """The cases the plan names, spelled out so the parametrised pin above is
    not the only reader of the rule."""
    # skip: the kind has no native form — regardless of the artifact's presence
    for kind in ("loreft", "alphasteer", cm.PETRI_STACKED_KIND):
        for spec in (None, _ALL, _RESPONSE):
            d, why = dc.native_serving_disposition(kind, spec)
            assert d == "skip", (kind, spec)
            assert why and (f"kind {kind!r}" in why or "D6" in why)
    assert "D6" in dc.native_serving_disposition(cm.PETRI_STACKED_KIND, None)[1]
    # native-psr-serving: the residual-slot kinds serve at mode all (absent too) AND, since
    # step 7, at the response mode when the record carries its header width
    for kind in ("psr", "vector_all", "diffmean"):
        assert dc.native_serving_disposition(kind, _ALL) == ("serve", None)
        assert dc.native_serving_disposition(kind, None) == ("serve", None)
        assert dc.native_serving_disposition(kind, _RESPONSE) == ("serve", None)
    # serve: the certification kinds, all-position; no artifact -> serve (ABSENT downstream)
    for kind in ("lora_all", "lora_subtraction", "lora"):
        assert dc.native_serving_disposition(kind, _ALL) == ("serve", None)
        assert dc.native_serving_disposition(kind, None) == ("serve", None)
    assert dc.native_serving_disposition("circuit_breaker", _CB_UNIT) == ("serve", None)
    # skip: a LoRA kind whose fit leg is position-masked — the reason names the mode
    for kind in ("lora", "lora_all", "lora_subtraction"):
        d, why = dc.native_serving_disposition(kind, _RESPONSE)
        assert d == "skip" and "position-masked" in why and "positions='response'" in why, (kind, why)
        d, why = dc.native_serving_disposition(kind, _LAST)
        assert d == "skip" and "position-masked" in why and "positions='last'" in why, (kind, why)
    # refuse: the kind IS servable but the record cannot be judged — positions MISSING stays a refusal
    d, why = dc.native_serving_disposition("lora_all", _NO_POSITIONS)
    assert d == "refuse" and "no 'positions'" in why
    d, why = dc.native_serving_disposition("lora", _NO_POSITIONS)
    assert d == "refuse" and "no 'positions'" in why
    d, why = dc.native_serving_disposition("lora_all", SHAPES["declared_thinking"])
    assert d == "refuse" and "serve declaration" in why
    d, why = dc.native_serving_disposition("lora_subtraction", SHAPES["declared_all_positions"])
    assert d == "refuse" and "serve declaration" in why
    d, why = dc.native_serving_disposition("circuit_breaker", _CB_FOLDED)
    assert d == "refuse" and "non-unit LoRA scaling" in why
    d, why = dc.native_serving_disposition("circuit_breaker", _CB_NO_SCALING)
    assert d == "refuse" and "rank" in why


def test_the_predicate_is_unchanged_and_stays_the_backstop():
    """``native_lora_unservable_reason`` keeps its signature and every verdict;
    the source's lora_native policy and the enumeration re-check still refuse a
    skip-class cell by name (the partition is what keeps it from reaching them)."""
    sig = inspect.signature(dc.native_lora_unservable_reason)
    assert list(sig.parameters) == ["kind", "resolved_spec"]
    assert dc.native_lora_unservable_reason("psr", _RESPONSE) is None    # served since step 7
    assert dc.native_lora_unservable_reason("psr", _LAST).startswith(
        "kind 'psr': residual-slot record cannot be judged")
    assert dc.native_lora_unservable_reason("loreft", _RESPONSE).startswith("kind 'loreft' is not natively exportable")
    assert dc.native_lora_unservable_reason(cm.PETRI_STACKED_KIND, None) == "stacked cells serve on the lens only (D6)"
    assert dc.native_lora_unservable_reason("lora_all", _ALL) is None
    src = inspect.getsource(ra.TrainTestCellSource._serving_path_for)
    assert "native_lora_unservable_reason" in src and "RegistryAdapterError" in src


# ---------------------------------------------------------------------------
# 2. the partition over the REAL explicit registry on a synthetic bank
# ---------------------------------------------------------------------------


def test_default_standup_serving_path_and_the_dataclass_default():
    assert ra.DEFAULT_STANDUP_SERVING_PATH == "lora_native"
    assert ra.DEFAULT_STANDUP_SERVING_PATH in ra.SERVING_PATH_POLICIES
    field = {f.name: f for f in dataclasses.fields(ra.TrainTestCellSource)}
    assert field["serving_path_policy"].default == "lens"          # the DATACLASS default stays
    sig = inspect.signature(ra.traintest_cell_source)
    assert sig.parameters["serving_path_policy"].default == "lens"  # and the YAML constructor's
    assert ra.SERVING_PATH_FLAG == "--serving-path"


def test_skipped_cell_shape():
    sk = ra.SkippedCell("si_x__psr_ce_all", "psr", "why")
    assert sk._fields == ("cell_id", "kind", "reason")
    assert tuple(sk) == ("si_x__psr_ce_all", "psr", "why")
    assert sk._asdict() == {"cell_id": "si_x__psr_ce_all", "kind": "psr", "reason": "why"}


def test_partition_serves_none_and_every_servable_cell_in_order(tmp_path):
    """Since step 7 the synthetic view's psr cell (fit at `response`, its header width
    recorded) is SERVED like the LoRA cells: the partition skips nothing here, and order
    is preserved. The skip path is pinned by the masked-LoRA test below and, for the
    stage-1 world, by tests/test_native_position_mode.py."""
    bank = _bank(tmp_path)                       # lora_all + psr + lora_subtraction staged
    lens = _explicit_source(bank, [TT_LORA, TT_PSR, TT_SUB])
    cells = ["none", H_PSR, H_LORA, H_SUB]
    served, skipped = ra.partition_cells_for_native(cells, lens)
    assert served == ["none", H_PSR, H_LORA, H_SUB] and skipped == []   # order preserved
    # pure: the source's entries are what they were (lens policy, nothing exported)
    assert lens.get(H_PSR).serving_path == "lens" and lens.get(H_LORA).serving_path == "lens"
    # and a different order is preserved too
    served2, skipped2 = ra.partition_cells_for_native([H_SUB, "none", H_LORA, H_PSR], lens)
    assert served2 == [H_SUB, "none", H_LORA, H_PSR] and skipped2 == []
    # the served set enumerates on a lora_native-policy source with no refusal
    from test_lora_native_enumeration import PROMPT_FILES, _bf, _real_spec
    native = _explicit_source(bank, [TT_LORA, TT_PSR, TT_SUB], serving_path_policy="lora_native")
    spec = _real_spec(served)
    cs = dc.enumerate_cells(spec, native, behavior_files=_bf(spec), prompt_files=PROMPT_FILES,
                            lens_controls=False)
    assert {c.cell_id for c in cs.all_served()} == {"none", H_PSR, H_LORA, H_SUB,
                                                    dc.NOOP_LORA_CELL_ID}


def test_partition_skips_a_masked_lora_fit_by_name_with_kind_and_the_mode(tmp_path):
    """The REAL explicit registry over a bank whose LoRA cell was fit at
    RESPONSE positions (the shape of ``safety_refusal__lora_r4`` on the gemma /
    qwen main views). The cell is SKIPPED — kind and the position-masked reason
    recorded — and the rest of the view is served; nothing raises."""
    bank = _bank(tmp_path, lora_positions="response", with_psr=False)
    lens = _explicit_source(bank, [TT_LORA, TT_SUB])
    assert lens.get(H_LORA).resolved_spec["positions"] == "response"
    served, skipped = ra.partition_cells_for_native(["none", H_LORA, H_SUB], lens)
    assert served == ["none", H_SUB]
    assert [(s.cell_id, s.kind) for s in skipped] == [(H_LORA, "lora_all")]
    why = skipped[0].reason
    assert "positions='response'" in why and "position-masked" in why
    assert why == dc.native_lora_unservable_reason("lora_all", lens.get(H_LORA).resolved_spec)
    # the served set enumerates on a lora_native-policy source with no refusal ...
    from test_lora_native_enumeration import PROMPT_FILES, _bf, _real_spec
    native = _explicit_source(bank, [TT_LORA, TT_SUB], serving_path_policy="lora_native")
    spec = _real_spec(served)
    cs = dc.enumerate_cells(spec, native, behavior_files=_bf(spec), prompt_files=PROMPT_FILES,
                            lens_controls=False)
    assert {c.cell_id for c in cs.all_served()} == {"none", H_SUB, dc.NOOP_LORA_CELL_ID}
    # ... and the skipped cell still REFUSES at the backstop if it ever reaches it (never served
    # at every token under its hash): the partition is what keeps it away
    with pytest.raises(ra.RegistryAdapterError, match="position-masked"):
        native.get(H_LORA)


def test_partition_of_a_mixed_main_view_serves_the_servable_and_skips_the_rest_by_name(tmp_path):
    """The gemma / qwen ``main`` view shape that once REFUSED: ``none``, a
    prompted cell, an all-positions LoRA, a RESPONSE-fit LoRA (kind ``lora``)
    and a PSR. Served = the first three; skipped = the masked LoRA and the PSR,
    each with its kind and reason; no raise."""
    root = tmp_path / "objs"
    root.mkdir()
    (root / "a.pt").write_bytes(b"synthetic-all-positions-lora")
    (root / "r.pt").write_bytes(b"synthetic-response-fit-lora")
    (root / "p.pt").write_bytes(b"synthetic-psr")
    LORA_ALL, LORA_RESP, PSR = "si_safety__lora_r4", "si_safety__lora_r1", "si_safety__psr_kl_all"
    assert ra.ARM_OBJECT_KINDS["lora_r4"] == "lora_all" and ra.ARM_OBJECT_KINDS["lora_r1"] == "lora"
    entries = {
        "none": dc.RegistryEntry(cell_id="none", kind="none"),
        "prompted__safety": dc.RegistryEntry(cell_id="prompted__safety", kind="prompted",
                                             si_id="safety", si_prefix=b"You are a careful assistant."),
        LORA_ALL: dc.RegistryEntry(cell_id=LORA_ALL, kind="lora_all", si_id="safety",
                                   object_path=str(root / "a.pt"), serving_path="lens",
                                   resolved_spec={"kind": "lora_all", "si_id": "safety", **_ALL}),
        LORA_RESP: dc.RegistryEntry(cell_id=LORA_RESP, kind="lora", si_id="safety",
                                    object_path=str(root / "r.pt"), serving_path="lens",
                                    resolved_spec={"kind": "lora", "si_id": "safety", **_RESPONSE}),
        PSR: dc.RegistryEntry(cell_id=PSR, kind="psr", si_id="safety",
                              object_path=str(root / "p.pt"), serving_path="lens",
                              resolved_spec={"kind": "psr", "si_id": "safety", "layers": [0, 2],
                                             "positions": "response", "response_header_len": 3}),
    }
    cells = ["none", "prompted__safety", LORA_ALL, LORA_RESP, PSR]
    served, skipped = ra.partition_cells_for_native(cells, dc.MappingCellSource(entries))
    # the PSR cell is RESPONSE-fit with its header width recorded — served since step 7 (the
    # position class serves the mode); the RESPONSE-fit LoRA is still skipped, because native
    # multi-LoRA has no position mask at all and serving it would run another intervention
    assert served == ["none", "prompted__safety", LORA_ALL, PSR]
    assert [(s.cell_id, s.kind) for s in skipped] == [(LORA_RESP, "lora")]
    by_id = {s.cell_id: s for s in skipped}
    assert "position-masked" in by_id[LORA_RESP].reason and "positions='response'" in by_id[LORA_RESP].reason
    assert all(isinstance(s, ra.SkippedCell) for s in skipped)
    # order is the view's, whichever way the skipped cells are interleaved
    served2, skipped2 = ra.partition_cells_for_native(
        [PSR, LORA_RESP, "none", LORA_ALL, "prompted__safety"], dc.MappingCellSource(entries))
    assert served2 == [PSR, "none", LORA_ALL, "prompted__safety"]
    assert [s.cell_id for s in skipped2] == [LORA_RESP]
    # known-bad through the same view: the SAME lora cell with its positions MISSING still refuses the standup
    entries[LORA_RESP] = dataclasses.replace(entries[LORA_RESP],
                                             resolved_spec={"kind": "lora", "si_id": "safety", **_NO_POSITIONS})
    with pytest.raises(ra.RegistryAdapterError) as exc:
        ra.partition_cells_for_native(cells, dc.MappingCellSource(entries))
    msg = str(exc.value)
    assert LORA_RESP in msg and "kind 'lora'" in msg and "no 'positions'" in msg
    assert "--serving-path lens" in msg and "refusing" in msg
    assert "skipped by name" in msg          # the refusal says a MASKED fit would have been skipped instead


def test_partition_refuses_a_circuit_breaker_with_folded_scaling_by_name(tmp_path):
    """The scaling refusal still refuses the standup: a ``circuit_breaker`` object
    with ``lora_alpha != r`` (or no scaling fields) is a servable kind whose record
    the engine cannot attest — never skipped, never served."""
    p = tmp_path / "cb.pt"
    p.write_bytes(b"synthetic-circuit-breaker")
    for spec, needle in ((_CB_FOLDED, "non-unit LoRA scaling"), (_CB_NO_SCALING, "'rank'/'lora_alpha'")):
        entries = {"none": dc.RegistryEntry(cell_id="none", kind="none"),
                   "antidote": dc.RegistryEntry(cell_id="antidote", kind="circuit_breaker", object_path=str(p),
                                                serving_path="lens", resolved_spec={"kind": "circuit_breaker", **spec})}
        with pytest.raises(ra.RegistryAdapterError) as exc:
            ra.partition_cells_for_native(["none", "antidote"], dc.MappingCellSource(entries))
        msg = str(exc.value)
        assert "antidote" in msg and "kind 'circuit_breaker'" in msg and needle in msg and "--serving-path lens" in msg
    # and the unit-scaled one serves
    entries["antidote"] = dataclasses.replace(entries["antidote"], resolved_spec={"kind": "circuit_breaker", **_CB_UNIT})
    assert ra.partition_cells_for_native(["none", "antidote"], dc.MappingCellSource(entries)) == (["none", "antidote"], [])


def test_partition_refuses_a_declared_variant_by_name(tmp_path):
    bank = _bank(tmp_path, with_psr=False)
    lens = _explicit_source(bank, [TT_LORA])
    with pytest.raises(ra.RegistryAdapterError) as exc:
        ra.partition_cells_for_native(["none", f"{H_LORA}@thinking"], lens)
    msg = str(exc.value)
    assert f"{H_LORA}@thinking" in msg and "serve declaration" in msg and "--serving-path lens" in msg


def test_partition_passes_an_absent_psr_through_like_an_absent_lora(tmp_path):
    """As amended by native-psr-serving: a psr cell is a servable kind now,
    so an ABSENT psr entry is served -> reports ABSENT downstream exactly like
    an absent LoRA entry (the standup refuses it as it always did); a
    non-native kind absent on disk is still skipped by kind."""
    from sisteer.eval import defense_cell as tt

    bank = tmp_path / "empty"
    bank.mkdir()
    registry = tt.build_registry(bank, si_ids=("constv2-mix-full",),
                                 class_a_arms=("lora_r4_ce", "psr_ce_all"), class_b_arms=())
    lens = ra.TrainTestCellSource(cells=tt.by_id(registry), defenses_dir=bank)
    assert lens.get(H_PSR).resolved_spec is None and lens.get(H_LORA).resolved_spec is None
    served, skipped = ra.partition_cells_for_native(["none", H_LORA, H_PSR], lens)
    assert served == ["none", H_LORA, H_PSR] and skipped == []
    assert dc.native_serving_disposition("loreft", None)[0] == "skip"      # a non-native kind, by kind


def test_partition_passes_a_registry_absent_cell_through(tmp_path):
    """A cell the registry LACKS has no kind to classify: it is served (the
    enumeration reports it ABSENT with its reason, exactly as today) — the
    partition removes skip-class cells and nothing else."""
    bank = _bank(tmp_path, with_psr=False)
    lens = _explicit_source(bank, [TT_LORA])           # the registry knows only the lora cell
    assert lens.get(H_SUB) is None
    served, skipped = ra.partition_cells_for_native(["none", H_SUB, H_LORA], lens)
    assert served == ["none", H_SUB, H_LORA] and skipped == []


def test_partition_skips_a_stacked_id_without_consulting_the_source():
    class Refusing:
        serving_path_policy = "lens"

        def get(self, cell_id):
            raise AssertionError(f"the source was consulted for {cell_id!r}")

    stacked = "stacked__safety__lora_r4"
    served, skipped = ra.partition_cells_for_native(["none", "prompted__safety", stacked], Refusing())
    assert served == ["none", "prompted__safety"]
    assert [(s.cell_id, s.kind) for s in skipped] == [(stacked, cm.PETRI_STACKED_KIND)]
    assert "D6" in skipped[0].reason


def test_partition_over_a_mapping_source_reads_kind_and_spec_only(tmp_path):
    from test_lora_native_enumeration import _golden_entries

    ent = _golden_entries(tmp_path)
    # the golden psr entry records `positions: response` and NO header width — a record the
    # mode predicate cannot judge once the mode is servable (step 7), which REFUSES rather
    # than being skipped. Completing it is what this test is about (the partition reads the
    # SPEC, nothing else), so the width is supplied here; the incomplete record's refusal is
    # pinned in test_lora_native_enumeration.
    gpsr = ent["si_safety__psr_kl_all"]
    ent["si_safety__psr_kl_all"] = dataclasses.replace(
        gpsr, resolved_spec={**gpsr.resolved_spec, "response_header_len": 3})
    src = dc.MappingCellSource(ent)                     # no policy attribute: read as lens
    cells = ["none", "prompted__safety", "si_safety__psr_kl_all", "si_safety__lora_r4", "alphasteer"]
    served, skipped = ra.partition_cells_for_native(cells, src)
    assert served == ["none", "prompted__safety", "si_safety__psr_kl_all", "si_safety__lora_r4"]
    assert [(s.cell_id, s.kind) for s in skipped] == [("alphasteer", "alphasteer")]
    # a masked lora entry is SKIPPED through the same door, with its kind and the mode
    golden = ent["si_safety__lora_r4"]
    ent["si_safety__lora_r4"] = dataclasses.replace(
        golden, resolved_spec={**golden.resolved_spec, "positions": "response", "response_header_len": 3})
    served, skipped = ra.partition_cells_for_native(cells, dc.MappingCellSource(ent))
    assert served == ["none", "prompted__safety", "si_safety__psr_kl_all"]
    assert [(s.cell_id, s.kind) for s in skipped] == [("si_safety__lora_r4", "lora_all"),
                                                      ("alphasteer", "alphasteer")]
    assert "position-masked" in skipped[0].reason and "positions='response'" in skipped[0].reason
    # a lora entry with NO positions refuses through the same door (malformed record — unchanged)
    ent["si_safety__lora_r4"] = dataclasses.replace(
        golden, resolved_spec={k: v for k, v in golden.resolved_spec.items() if k != "positions"})
    with pytest.raises(ra.RegistryAdapterError, match="--serving-path lens") as exc:
        ra.partition_cells_for_native(cells, dc.MappingCellSource(ent))
    assert "no 'positions'" in str(exc.value) and "si_safety__lora_r4" in str(exc.value)


def test_partition_refuses_a_lora_native_policy_source(tmp_path):
    bank = _bank(tmp_path)
    native = _explicit_source(bank, [TT_LORA, TT_PSR], serving_path_policy="lora_native")
    with pytest.raises(ra.RegistryAdapterError, match="LENS-policy"):
        ra.partition_cells_for_native(["none", H_LORA, H_PSR], native)


def test_partition_refuses_duplicates_and_derived_ids(tmp_path):
    bank = _bank(tmp_path, with_psr=False)
    lens = _explicit_source(bank, [TT_LORA])
    with pytest.raises(ra.RegistryAdapterError, match="named twice"):
        ra.partition_cells_for_native(["none", H_LORA, H_LORA], lens)
    with pytest.raises(dc.DefenseCellError):
        ra.partition_cells_for_native(["none", dc.NOOP_LORA_CELL_ID], lens)


def test_partition_signature_is_the_plans():
    sig = inspect.signature(ra.partition_cells_for_native)
    assert list(sig.parameters) == ["harness_cells", "lens_source"]
    sig = inspect.signature(dc.native_serving_disposition)
    assert list(sig.parameters) == ["kind", "resolved_spec"]
