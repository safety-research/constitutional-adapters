"""ONE geometry-adjustment contract. A native standup lowers S under the
LoRA kernel's
int32 index ceiling (gemma 121 -> 107, nemotron 161 -> 86 at L 4864); a KV
shrink lowers it once more. ``geometry.adjustments`` is the ONE ordered record
(``[]`` / ``[kernel]`` / ``[kv]`` / ``[kernel, kv]``) and
``geometry.geometry_adjustment_problem`` re-derives every step, so no reader
ever accepts "a smaller S" — only a named, re-derivable one.

Pinned here:

    - ``LORA_MAX_MODULE_DIM_BY_MODEL`` against the pulled certification
      attestations' ``lora_max_module_dim`` when they are on this machine, and as
      literals (gemma4e4b 4096, qwen35_4b 4096, nemotron3nano4b 5120, llama8b 4096
      — the architecture's width; its attestation lands with the llama units).
    - ``kernel_ceiling_adjustment``: PURE, present iff it lowers — the two
      certification numbers, the plan's nemotron numbers at 2528 / 4576 / 8672
      (256 -> 165, 171 -> 91, 90 -> 48), llama8b at 4864 (no step), the refusals.
    - the positives through ``geometry_adjustment_problem`` on the standup's own
      ``geometry`` block shape (``ResolvedGeometry.as_dict()`` + ``adjustments``):
      gemma-shaped (from ``test_lora_serving_up``'s e2e numbers), nemotron-shaped,
      llama-shaped (``[]``, identity True), ``[kv]`` through today's rule
      byte-identically (a pre-contract block with only a ``kv_shrink`` record reads
      the same), ``[kernel, kv]`` composed from the kernel-lowered S.
    - the negatives, each a NAMED refusal: a kernel step on llama at 4864 (D does
      not lower), ``to_S`` off by one, steps reversed, a bare smaller S with no
      record, a ``kv_shrink`` from the identity S when a kernel step precedes it,
      an unknown kind, a duplicated kind, an unknown model key, a torn top-level
      ``kv_shrink`` beside the list, a block whose own S / L / identity disagree,
      ``native=True`` with the kernel step omitted, ``native=False`` with one present,
      a present-but-incomplete ``kv_shrink`` record with no ``adjustments`` key.
    - ``kv_shrink_problem`` is unchanged (signature and verdicts).

CPU-only."""

from __future__ import annotations

import copy
import inspect
import json
from dataclasses import replace
from pathlib import Path

import pytest

from sisteer.capeval.serving import geometry as sg
from sisteer.capeval.serving import serving_attestation as sa
from _env import host_site_path  # noqa: E402  (tests/ helper: a host-only data path through the site config)

L = 4864
EVAL_4864 = next(k for k in sg.KNOWN_KEYS if sg.TABLE[k].max_model_len == L)
KERNEL, KV = sg.ADJUSTMENT_KERNEL, sg.ADJUSTMENT_KV
LIMIT = 2 ** 31 - 1


def _identity(model_key):
    g = sg.resolve_geometry(EVAL_4864, model_key=model_key)
    assert g.max_model_len == L and g.is_identity and g.seqs_source == "model_budget"
    return g


def _block(geom, adjustments=None, **extra):
    """The standup's ``geometry`` block: ``ResolvedGeometry.as_dict()`` plus the
    ordered adjustments (absent when None — a lens / pre-contract block)."""
    d = geom.as_dict()
    if adjustments is not None:
        d[sg.ADJUSTMENTS_KEY] = copy.deepcopy(adjustments)
    d.update(extra)
    return d


def _kv_record(configured, *, conc, L_=L):
    cap = int(conc * L_)
    return sg.kv_shrink_record(configured_max_num_seqs=configured, max_model_len=L_,
                               kv_backed_concurrency=conc, kv_capacity_tokens=cap, basis="uniform")


# ---------------------------------------------------------------------------
# 1. vocabularies and the per-model widths
# ---------------------------------------------------------------------------


def test_vocabulary_and_exports():
    # geometry-fix-2 (2026-10-02): the launch-class step leads the ordered chain; the kernel / KV vocabulary is unchanged behind it
    assert sg.ADJUSTMENT_KINDS == ("launch_class", "lora_kernel_index_ceiling", "kv_shrink")
    assert sg.ADJUSTMENT_LAUNCH_CLASS == "launch_class" and sg.LAUNCH_CLASS_ADJUSTMENT_FIELDS == ("kind", "from_S", "to_S", "max_lora_rank", "max_loras", "L", "window_seqs_source", "launches")
    assert KERNEL == sa.LORA_KERNEL_CEILING_REASON == sa.LORA_KERNEL_CEILING_KEY
    assert KV == sg.KV_SHRINK_KEY
    assert sg.ADJUSTMENTS_KEY == "adjustments"
    assert sg.KERNEL_ADJUSTMENT_FIELDS == ("kind", "from_S", "to_S", "D", "ceiling")
    for name in ("ADJUSTMENT_KINDS", "LAUNCH_CLASS_ADJUSTMENT_FIELDS", "LORA_MAX_MODULE_DIM_BY_MODEL", "kernel_ceiling_adjustment",
                 "geometry_adjustment_problem", "kv_shrink_adjustment", "geometry_adjustments",
                 "adjusted_seqs", "kv_shrink_problem"):
        assert name in sg.__all__, name
    sig = inspect.signature(sg.geometry_adjustment_problem)
    # geometry-fix-2: `served_lora` — the standup's attested (rank cap, adapters), optional; a launch_class step must be that class when given
    assert list(sig.parameters) == ["geometry_block", "model_key", "L", "S_identity", "S_served", "native", "served_lora"]
    assert sig.parameters["native"].default is None and sig.parameters["served_lora"].default is None
    # kv_shrink_problem: unchanged
    sig = inspect.signature(sg.kv_shrink_problem)
    assert list(sig.parameters) == ["shrink", "configured_max_num_seqs", "max_model_len", "served_max_num_seqs"]


def test_lora_max_module_dim_by_model_literals():
    assert dict(sg.LORA_MAX_MODULE_DIM_BY_MODEL) == {
        "gemma4e4b": 4096, "qwen35_4b": 4096, "nemotron3nano4b": 5120, "llama8b": 4096}
    assert set(sg.LORA_MAX_MODULE_DIM_BY_MODEL) <= set(sg.MODEL_KEYS)
    for D in sg.LORA_MAX_MODULE_DIM_BY_MODEL.values():
        assert sa.lora_kernel_index_ceiling(D) == LIMIT // D


def test_lora_max_module_dim_by_model_against_the_pulled_attestations():
    """The pulled certification records are the AUTHORITY for the widths
    (their ``lora_max_module_dim`` is the attested widest adapted module);
    every pulled attestation's model must be pinned to exactly its value.
    Skipped where none is pulled (the Modal CPU suite); the literal pin above
    still holds there."""
    pulled = sorted(host_site_path("$STATE/romp/multilora-cert").glob("*/unit/lora_native/serving_attestation.json"))
    if not pulled:
        pytest.skip("no pulled certification attestations on this machine")
    seen = {}
    for path in pulled:
        att = json.loads(path.read_text())
        key = sg.model_key_for(att.get("model") or att.get("model_id")) or path.parents[2].name.rsplit("_", 1)[0]
        dim = att[sa.LORA_MAX_MODULE_DIM_KEY]
        assert att[sa.LORA_KERNEL_CEILING_KEY] == LIMIT // dim
        assert sg.LORA_MAX_MODULE_DIM_BY_MODEL[key] == dim, (path, key, dim)
        seen[key] = dim
    assert {"gemma4e4b", "nemotron3nano4b", "qwen35_4b"} <= set(seen)


# ---------------------------------------------------------------------------
# 2. kernel_ceiling_adjustment: pure, present iff it lowers
# ---------------------------------------------------------------------------


def test_the_two_certification_numbers_and_the_plans_nemotron_numbers():
    assert sg.kernel_ceiling_adjustment("gemma4e4b", L, 121) == {
        "kind": KERNEL, "from_S": 121, "to_S": 107, "D": 4096, "ceiling": 524_287}
    assert sg.kernel_ceiling_adjustment("nemotron3nano4b", L, 161) == {
        "kind": KERNEL, "from_S": 161, "to_S": 86, "D": 5120, "ceiling": 419_430}
    # the nemotron rows: the probe geometry at 2048 / 4096 / 8192 caps
    assert sg.kernel_ceiling_adjustment("nemotron3nano4b", 2528, 256)["to_S"] == 165
    assert sg.kernel_ceiling_adjustment("nemotron3nano4b", 4576, 171)["to_S"] == 91
    assert sg.kernel_ceiling_adjustment("nemotron3nano4b", 8672, 90)["to_S"] == 48
    # llama8b at 4864: 524,287 // 4864 = 107 >= the identity 97 -> no step
    assert sg.kernel_ceiling_adjustment("llama8b", L, 97) is None
    assert sg.adjusted_seqs("llama8b", L, 97) == 97
    assert sg.adjusted_seqs("gemma4e4b", L, 121) == 107 and sg.adjusted_seqs("nemotron3nano4b", L, 161) == 86
    # the real budget table agrees with the numbers the plan quotes
    assert _identity("gemma4e4b").max_num_seqs == 121
    assert _identity("nemotron3nano4b").max_num_seqs == 161
    assert _identity("llama8b").max_num_seqs == 97


def test_kernel_ceiling_adjustment_refusals():
    with pytest.raises(sg.UnknownModelKey, match="no pinned LoRA max module dim"):
        sg.kernel_ceiling_adjustment("glm46vflash", L, 50)
    with pytest.raises(ValueError, match="alone exceeds"):
        sg.kernel_ceiling_adjustment("gemma4e4b", 600_000, 4)
    for bad in ((0, 10), (L, 0), (L, True), ("4864", 10)):
        with pytest.raises(ValueError):
            sg.kernel_ceiling_adjustment("gemma4e4b", *bad)
    # a step is present iff it lowers: exactly at the ceiling it is absent
    assert sg.kernel_ceiling_adjustment("gemma4e4b", L, 107) is None
    assert sg.kernel_ceiling_adjustment("gemma4e4b", L, 108)["to_S"] == 107


# ---------------------------------------------------------------------------
# 3. the positives
# ---------------------------------------------------------------------------


class TestPositives:
    def test_gemma_shaped_kernel_step(self):
        """The e2e standup's numbers (tests/test_lora_serving_up.py: 121 -> 107 at
        4864, ``seqs_source`` override, identity False) plus the one record."""
        ident = _identity("gemma4e4b")
        step = sg.kernel_ceiling_adjustment("gemma4e4b", L, ident.max_num_seqs)
        served = replace(ident, max_num_seqs=107, seqs_source="override")
        block = _block(served, [step])
        assert block["identity"] is False and block["max_num_seqs"] == 107
        for native in (None, True):
            assert sg.geometry_adjustment_problem(block, model_key="gemma4e4b", L=L, S_identity=121,
                                                  S_served=107, native=native) is None
        assert sg.geometry_adjustments(block) == [step]

    def test_nemotron_shaped_kernel_step(self):
        ident = _identity("nemotron3nano4b")
        step = sg.kernel_ceiling_adjustment("nemotron3nano4b", L, 161)
        block = _block(replace(ident, max_num_seqs=86, seqs_source="override"), [step])
        assert sg.geometry_adjustment_problem(block, model_key="nemotron3nano4b", L=L, S_identity=161,
                                              S_served=86, native=True) is None

    def test_llama_shaped_identity(self):
        ident = _identity("llama8b")
        for block in (_block(ident, []), _block(ident)):         # `[]` and ABSENT both read identity
            assert block["identity"] is True
            for native in (None, True, False):
                assert sg.geometry_adjustment_problem(block, model_key="llama8b", L=L, S_identity=97,
                                                      S_served=97, native=native) is None
        # a lens row on gemma at identity: no kernel on the lens engine
        gem = _identity("gemma4e4b")
        assert sg.geometry_adjustment_problem(_block(gem), model_key="gemma4e4b", L=L, S_identity=121,
                                              S_served=121, native=False) is None
        assert sg.geometry_adjustment_problem(_block(gem), model_key=None, L=L, S_identity=121,
                                              S_served=121) is None

    def test_kv_only_is_todays_rule_byte_identically(self):
        """A block with ``kv_shrink`` and NO ``adjustments`` (every record
        written before the contract) reads through the one function exactly as
        ``kv_shrink_problem`` read it; the same block WITH the derived entry is
        the same verdict."""
        ident = _identity("llama8b")                     # 97 at identity
        rec = _kv_record(97, conc=40.7)                  # KV backs 40 full-length rows
        assert rec["served_max_num_seqs"] == 40
        served = sg.kv_shrunk_geometry(ident, rec)
        legacy = _block(served, None, **{sg.KV_SHRINK_KEY: rec})          # today's shape
        assert sg.geometry_adjustments(legacy) == [sg.kv_shrink_adjustment(rec)]
        assert sg.geometry_adjustment_problem(legacy, model_key="llama8b", L=L, S_identity=97, S_served=40) is None
        assert sg.geometry_adjustment_problem(legacy, model_key=None, L=L, S_identity=97, S_served=40, native=False) is None
        # today's rule refuses the same things through the one function
        assert sg.kv_shrink_problem(rec, configured_max_num_seqs=97, max_model_len=L, served_max_num_seqs=40) is None
        assert "not this run's configured" in sg.kv_shrink_problem(rec, configured_max_num_seqs=98, max_model_len=L,
                                                                   served_max_num_seqs=40)
        assert "not this run's configured" in sg.geometry_adjustment_problem(
            legacy, model_key="llama8b", L=L, S_identity=98, S_served=40)
        # the contract's own spelling of the same shrink
        explicit = _block(served, [sg.kv_shrink_adjustment(rec)], **{sg.KV_SHRINK_KEY: rec})
        assert sg.geometry_adjustment_problem(explicit, model_key="llama8b", L=L, S_identity=97, S_served=40) is None
        assert explicit["identity"] is False and explicit["seqs_source"] == sg.KV_DERIVED_SOURCE

    def test_kernel_then_kv_composition(self):
        """The KV step is computed from the KERNEL-lowered S (gemma: 121 -> 107
        -> the KV-backed 80)."""
        ident = _identity("gemma4e4b")
        kernel = sg.kernel_ceiling_adjustment("gemma4e4b", L, 121)
        rec = _kv_record(107, conc=80.3)
        assert rec["configured_max_num_seqs"] == 107 and rec["served_max_num_seqs"] == 80
        kv = sg.kv_shrink_adjustment(rec)
        assert (kv["kind"], kv["from_S"], kv["to_S"]) == (KV, 107, 80)
        served = replace(ident, max_num_seqs=80, seqs_source=sg.KV_DERIVED_SOURCE)
        block = _block(served, [kernel, kv], **{sg.KV_SHRINK_KEY: rec})
        for native in (None, True):
            assert sg.geometry_adjustment_problem(block, model_key="gemma4e4b", L=L, S_identity=121,
                                                  S_served=80, native=native) is None
        # without the top-level kv_shrink companion it still holds (the list is the record)
        assert sg.geometry_adjustment_problem(_block(served, [kernel, kv]), model_key="gemma4e4b", L=L,
                                              S_identity=121, S_served=80) is None

    def test_kv_shrink_adjustment_carries_the_record(self):
        rec = _kv_record(97, conc=40.7)
        adj = sg.kv_shrink_adjustment(rec)
        assert set(adj) == set(rec) | {"kind", "from_S", "to_S"}
        assert all(adj[k] == rec[k] for k in rec)
        with pytest.raises(ValueError, match="lacks"):
            sg.kv_shrink_adjustment({"configured_max_num_seqs": 1})
        with pytest.raises(ValueError, match="not a mapping"):
            sg.kv_shrink_adjustment("x")


# ---------------------------------------------------------------------------
# 4. the negatives — each a NAMED refusal, never "a smaller S"
# ---------------------------------------------------------------------------


def _gemma_lowered():
    ident = _identity("gemma4e4b")
    step = sg.kernel_ceiling_adjustment("gemma4e4b", L, 121)
    return ident, step, replace(ident, max_num_seqs=107, seqs_source="override")


class TestNegatives:
    def test_a_kernel_step_on_llama_at_4864_where_D_does_not_lower(self):
        ident = _identity("llama8b")
        fake = {"kind": KERNEL, "from_S": 97, "to_S": 90, "D": 4096, "ceiling": 524_287}
        why = sg.geometry_adjustment_problem(_block(replace(ident, max_num_seqs=90), [fake]),
                                             model_key="llama8b", L=L, S_identity=97, S_served=90)
        assert why and "does not lower the identity S 97" in why and "present iff it lowers" in why

    def test_to_S_off_by_one(self):
        ident, step, served = _gemma_lowered()
        off = {**step, "to_S": 106}
        why = sg.geometry_adjustment_problem(_block(replace(ident, max_num_seqs=106), [off]),
                                             model_key="gemma4e4b", L=L, S_identity=121, S_served=106)
        assert why and "records to_S=106" in why and "the formula gives 107" in why
        for k, v in (("D", 5120), ("ceiling", 524_286), ("from_S", 120)):
            why = sg.geometry_adjustment_problem(_block(served, [{**step, k: v}]), model_key="gemma4e4b", L=L,
                                                 S_identity=121, S_served=107)
            assert why and f"records {k}={v!r}" in why, (k, why)

    def test_steps_reversed(self):
        ident, kernel, _ = _gemma_lowered()
        kv = sg.kv_shrink_adjustment(_kv_record(107, conc=80.3))
        block = _block(replace(ident, max_num_seqs=80, seqs_source=sg.KV_DERIVED_SOURCE), [kv, kernel])
        why = sg.geometry_adjustment_problem(block, model_key="gemma4e4b", L=L, S_identity=121, S_served=80)
        assert why and "is ordered ['kv_shrink', 'lora_kernel_index_ceiling']" in why

    def test_a_bare_smaller_S_with_no_record(self):
        ident = _identity("gemma4e4b")
        for block in (_block(replace(ident, max_num_seqs=107), []), _block(replace(ident, max_num_seqs=107))):
            why = sg.geometry_adjustment_problem(block, model_key="gemma4e4b", L=L, S_identity=121, S_served=107)
            assert why and "names no adjustment" in why and "named, re-derivable step" in why
        # ... and the served S disagreeing with the last step
        _, step, served = _gemma_lowered()
        why = sg.geometry_adjustment_problem(_block(replace(served, max_num_seqs=100), [step]),
                                             model_key="gemma4e4b", L=L, S_identity=121, S_served=100)
        assert why and "not the last adjustment's to_S 107" in why

    def test_a_kv_shrink_from_the_identity_S_when_a_kernel_step_precedes_it(self):
        ident, kernel, _ = _gemma_lowered()
        kv_from_identity = sg.kv_shrink_adjustment(_kv_record(121, conc=80.3))   # configured 121, not 107
        block = _block(replace(ident, max_num_seqs=80, seqs_source=sg.KV_DERIVED_SOURCE), [kernel, kv_from_identity])
        why = sg.geometry_adjustment_problem(block, model_key="gemma4e4b", L=L, S_identity=121, S_served=80)
        # today's rule names it first (the record's configured S is not the kernel-lowered S);
        # a hand-made entry whose from_S disagrees with its own record is named by the from_S check
        assert why and "kv_shrink" in why and "121" in why and "107" in why
        assert "configured S 121 is not this run's configured 107" in why
        torn = {**kv_from_identity, "configured_max_num_seqs": 107, "kv_capacity_tokens": 80 * L,
                "kv_backed_concurrency": 80.3}
        why = sg.geometry_adjustment_problem(_block(replace(ident, max_num_seqs=80, seqs_source=sg.KV_DERIVED_SOURCE),
                                                    [kernel, torn]), model_key="gemma4e4b", L=L, S_identity=121, S_served=80)
        assert why and "from_S=121 is not the previous step's S 107" in why and "kernel" in why

    def test_a_kv_step_that_does_not_hold(self):
        ident = _identity("llama8b")
        rec = _kv_record(97, conc=40.7)
        bad = {**sg.kv_shrink_adjustment(rec), "kv_capacity_tokens": 46 * L}
        why = sg.geometry_adjustment_problem(_block(replace(ident, max_num_seqs=40), [bad]),
                                             model_key="llama8b", L=L, S_identity=97, S_served=40)
        assert why and "does not hold" in why and "kv_capacity_tokens" in why
        # today's rule names the same thing
        assert "kv_capacity_tokens" in sg.kv_shrink_problem(bad, configured_max_num_seqs=97, max_model_len=L,
                                                           served_max_num_seqs=40)

    def test_unknown_or_duplicated_kinds_and_shapes(self):
        ident, step, served = _gemma_lowered()
        why = sg.geometry_adjustment_problem(_block(served, [{**step, "kind": "manual_override"}]),
                                             model_key="gemma4e4b", L=L, S_identity=121, S_served=107)
        assert why and "kind='manual_override' is not one of" in why
        why = sg.geometry_adjustment_problem(_block(served, [step, step]), model_key="gemma4e4b", L=L,
                                             S_identity=121, S_served=107)
        assert why and "names a kind twice" in why
        why = sg.geometry_adjustment_problem(_block(served, [{**step, "L": L}]), model_key="gemma4e4b", L=L,
                                             S_identity=121, S_served=107)
        assert why and "keys are" in why
        why = sg.geometry_adjustment_problem({**_block(served), sg.ADJUSTMENTS_KEY: "kernel"}, model_key="gemma4e4b",
                                             L=L, S_identity=121, S_served=107)
        assert why and "is not a list" in why
        why = sg.geometry_adjustment_problem(_block(served, ["kernel"]), model_key="gemma4e4b", L=L,
                                             S_identity=121, S_served=107)
        assert why and "is not a mapping" in why
        assert "not a mapping" in sg.geometry_adjustment_problem(None, model_key="gemma4e4b", L=L, S_identity=121, S_served=107)

    def test_an_unknown_model_key_with_a_kernel_step(self):
        ident, step, served = _gemma_lowered()
        why = sg.geometry_adjustment_problem(_block(served, [step]), model_key="glm46vflash", L=L,
                                             S_identity=121, S_served=107)
        assert why and "no pinned LoRA max module dim" in why
        why = sg.geometry_adjustment_problem(_block(served, [step]), model_key=None, L=L, S_identity=121, S_served=107)
        assert why and "no model key" in why

    def test_a_malformed_kv_shrink_with_no_adjustments_key(self):
        """The pre-contract shape's ONE step is the kv_shrink record itself: a
        record present but incomplete derives nothing and is NAMED, never read
        as "no adjustment" — even at the identity S; a null kv_shrink (the lens
        writers' spelling of none) is no record at all."""
        ident = _identity("llama8b")
        for bad in ({"configured_max_num_seqs": 97}, {"served_max_num_seqs": 40, "rule": "x"}, "shrunk", 40):
            why = sg.geometry_adjustment_problem(_block(ident, None, **{sg.KV_SHRINK_KEY: bad}),
                                                 model_key="llama8b", L=L, S_identity=97, S_served=97)
            assert why and sg.KV_SHRINK_KEY in why and ("lacks" in why or "not a mapping" in why), bad
        assert sg.geometry_adjustment_problem(_block(ident, None, **{sg.KV_SHRINK_KEY: None}),
                                              model_key="llama8b", L=L, S_identity=97, S_served=97) is None
        # with the adjustments key present the list is the record; a stray incomplete companion is torn
        rec = _kv_record(97, conc=40.7)
        served = sg.kv_shrunk_geometry(ident, rec)
        why = sg.geometry_adjustment_problem(_block(served, [sg.kv_shrink_adjustment(rec)],
                                                    **{sg.KV_SHRINK_KEY: {"configured_max_num_seqs": 97}}),
                                             model_key="llama8b", L=L, S_identity=97, S_served=40)
        assert why and "not the same record" in why

    def test_a_torn_top_level_kv_shrink(self):
        ident = _identity("llama8b")
        rec = _kv_record(97, conc=40.7)
        served = sg.kv_shrunk_geometry(ident, rec)
        other = _kv_record(97, conc=41.2)
        why = sg.geometry_adjustment_problem(_block(served, [sg.kv_shrink_adjustment(rec)], **{sg.KV_SHRINK_KEY: other}),
                                             model_key="llama8b", L=L, S_identity=97, S_served=40)
        assert why and "not the same record" in why
        why = sg.geometry_adjustment_problem(_block(served, [], **{sg.KV_SHRINK_KEY: rec}),
                                             model_key="llama8b", L=L, S_identity=97, S_served=40)
        assert why and "names no 'kv_shrink' step" in why

    def test_the_blocks_own_facts_must_agree(self):
        ident, step, served = _gemma_lowered()
        why = sg.geometry_adjustment_problem(_block(served, [step], identity=True), model_key="gemma4e4b", L=L,
                                             S_identity=121, S_served=107)
        assert why and "identity=True with 1 adjustment" in why
        why = sg.geometry_adjustment_problem(_block(ident, [], identity=False), model_key="gemma4e4b", L=L,
                                             S_identity=121, S_served=121)
        assert why and "identity=False with 0 adjustment" in why
        why = sg.geometry_adjustment_problem(_block(served, [step]), model_key="gemma4e4b", L=L,
                                             S_identity=121, S_served=121)
        assert why and "records max_num_seqs=107 but the S served is 121" in why
        why = sg.geometry_adjustment_problem(_block(served, [step]), model_key="gemma4e4b", L=4576,
                                             S_identity=128, S_served=107)
        assert why and "records max_model_len=4864" in why
        for bad in ((0, 121, 107), (L, 0, 107), (L, 121, None), (L, 121, True), (L, 121, "107")):
            assert sg.geometry_adjustment_problem(_block(served, [step]), model_key="gemma4e4b", L=bad[0],
                                                  S_identity=bad[1], S_served=bad[2])

    def test_the_engine_bit(self):
        """``native=True`` closes the one gap self-consistency leaves — a native
        record that OMITTED the kernel step where the formula lowers; ``False``
        refuses a kernel step on the lens engine."""
        ident, step, served = _gemma_lowered()
        why = sg.geometry_adjustment_problem(_block(ident, []), model_key="gemma4e4b", L=L, S_identity=121,
                                             S_served=121, native=True)
        assert why and "must record the 'lora_kernel_index_ceiling' step (121 -> 107" in why
        assert sg.geometry_adjustment_problem(_block(ident, []), model_key="gemma4e4b", L=L, S_identity=121,
                                              S_served=121) is None          # self-consistent without the bit
        why = sg.geometry_adjustment_problem(_block(served, [step]), model_key="gemma4e4b", L=L, S_identity=121,
                                             S_served=107, native=False)
        assert why and "on a LENS engine" in why
        # a kv-only native record on llama (no kernel step to require) is fine under native=True
        lid = _identity("llama8b")
        rec = _kv_record(97, conc=40.7)
        block = _block(sg.kv_shrunk_geometry(lid, rec), [sg.kv_shrink_adjustment(rec)])
        assert sg.geometry_adjustment_problem(block, model_key="llama8b", L=L, S_identity=97, S_served=40,
                                              native=True) is None


def test_the_standup_records_kernel_facts_agree_with_the_contract():
    """The keys the standup already writes (``geometry_lowered`` /
    ``lora_kernel_index_ceiling``, pinned by test_lora_serving_up) and the
    contract's step describe ONE lowering — the numbers must agree, so the
    core can write both from the same source."""
    step = sg.kernel_ceiling_adjustment("gemma4e4b", L, 121)
    lowered = {"reason": sa.LORA_KERNEL_CEILING_REASON, "requested_S": 121, "S": 107, "L": L, "D": 4096,
               "ceiling": 524_287, "product_requested": 588_544, "product": 520_448}
    assert (step["from_S"], step["to_S"], step["D"], step["ceiling"], step["kind"]) == (
        lowered["requested_S"], lowered["S"], lowered["D"], lowered["ceiling"], lowered["reason"])
    assert step["to_S"] * L == lowered["product"] <= lowered["ceiling"] < lowered["product_requested"]


# --------------------------------------------------------------------------- #
# the truncation ladder's DELIBERATE reduction — the third named reason         #
# for a served S below the budget's, applied FIRST, beside the list             #
# --------------------------------------------------------------------------- #

LADDER_L, GEMMA_LADDER = 32_768, (18, 16, 14, 12)


def _gemma_block(asked, served, adjustments, reduction=None, **extra):
    """The NATIVE core's block for a gemma standup ASKED ``asked`` at the ladder window (``ResolvedGeometry.as_dict()``
    at the asked S — the model's budget; identity iff 18 — restated at ``served``) + ``adjustments`` (+ the launcher's
    ``ladder_reduction`` when the ladder asked below the budget's identity)."""
    g = sg.resolve_geometry(None, max_model_len=LADDER_L, max_num_seqs=asked, model_key="gemma4e4b")
    d = g.as_dict()
    d.update(max_num_seqs=served, identity=(asked == 18 and served == 18), batched_tokens=served * LADDER_L)
    d[sg.ADJUSTMENTS_KEY] = copy.deepcopy(adjustments)
    if reduction is not None:
        d[sg.LADDER_REDUCTION_KEY] = copy.deepcopy(reduction)
    d.update(extra)
    return d


def _reduction(to_S, source="probe", **over):
    """A VALID gemma reduction record (the constructor validates), then ``to_S`` / ``over`` applied as overrides so the
    negatives can name a rung the constructor would refuse."""
    rec = sg.ladder_reduction_record(identity_S=18, requested_S=(to_S if to_S in GEMMA_LADDER[1:] else 14), max_model_len=LADDER_L,
                                     ladder_seqs=GEMMA_LADDER, rung_source=source)
    rec["to_S"] = to_S
    rec.update(over)
    return rec


def _P(block, native=True, **over):
    kw = dict(model_key="gemma4e4b", L=LADDER_L, S_identity=18, S_served=block["max_num_seqs"], native=native)
    kw.update(over)
    return sg.geometry_adjustment_problem(block, **kw)


def test_the_ladder_reduction_vocabulary_is_additive_and_not_a_list_kind():
    """The reduction is a record BESIDE ``adjustments`` (the ``kv_shrink`` precedent), never a list kind: the standup
    writes the list and never writes this step; the launcher writes the record. The two below-identity rung sources
    are the ladder's (``rung_probe`` pins the same two words at import)."""
    assert sg.ADJUSTMENT_KINDS == ("launch_class", "lora_kernel_index_ceiling", "kv_shrink")   # the class step leads (geometry-fix-2); the reduction is still no list kind
    assert sg.LADDER_REDUCTION_KEY == "ladder_reduction" and sg.LADDER_REDUCTION_KEY not in sg.ADJUSTMENT_KINDS
    assert sg.LADDER_REDUCTION_FIELDS == ("from_S", "to_S", "max_model_len", "ladder_seqs", "rung_source", "rule")
    assert sg.LADDER_REDUCTION_SOURCES == ("probe", "operator")
    for name in ("LADDER_REDUCTION_KEY", "LADDER_REDUCTION_RULE", "LADDER_REDUCTION_FIELDS", "LADDER_REDUCTION_SOURCES",
                 "ladder_reduction_record", "ladder_reduction_problem", "recorded_identity_seqs"):
        assert name in sg.__all__, name
    rec = _reduction(14)
    assert tuple(rec) == sg.LADDER_REDUCTION_FIELDS
    assert rec == {"from_S": 18, "to_S": 14, "max_model_len": LADDER_L, "ladder_seqs": [18, 16, 14, 12], "rung_source": "probe",
                   "rule": sg.LADDER_REDUCTION_RULE}
    assert sg.ladder_reduction_problem(rec, L=LADDER_L, S_identity=18) is None
    assert sg.ladder_reduction_problem(_reduction(12, "operator"), L=LADDER_L, S_identity=18) is None


def test_recorded_identity_seqs_is_the_blocks_own_budget_at_its_window():
    """The ONE derivation of a record's identity S the scoring driver and the launcher's production reading share:
    the block's OWN recorded budget at its window through the model's row (hybrid cap on top), the bare quotient with
    no model; malformed blocks and unknown models refuse by name — whatever S was asked or served."""
    for asked, served, adj in ((18, 15, [sg.kernel_ceiling_adjustment("gemma4e4b", LADDER_L, 18)]), (14, 14, []), (15, 15, []), (12, 12, [])):
        assert sg.recorded_identity_seqs(_gemma_block(asked, served, adj)) == 18
    assert sg.recorded_identity_seqs({"max_model_len": LADDER_L, "budget": 589_824}) == 18                 # no model: the quotient
    nemo = sg.resolve_geometry(None, max_model_len=LADDER_L, max_num_seqs=24, model_key="nemotron3nano4b").as_dict()
    assert sg.recorded_identity_seqs(nemo) == 24 == sg.BUDGET_BY_MODEL["nemotron3nano4b"].seqs_at(LADDER_L)
    for bad, needle in (({"max_model_len": LADDER_L}, "geometry.budget is None"), ({"budget": 589_824, "max_model_len": 0}, "geometry.max_model_len is 0"),
                        ({"budget": True, "max_model_len": LADDER_L}, "geometry.budget is True"), ("nope", "not a mapping")):
        with pytest.raises(ValueError, match=needle):
            sg.recorded_identity_seqs(bad)
    with pytest.raises(sg.UnknownModelKey):
        sg.recorded_identity_seqs({"max_model_len": LADDER_L, "budget": 589_824, "model_key": "no-such-model"})


def test_a_ladder_rung_chain_re_derives_from_the_budgets_identity_through_the_requested_rung():
    """Gemma's native ladder at 32,768 (identity 18; the kernel ceiling serves at most 15). The identity rung is asked
    18 -> the standup's own kernel step 18 -> 15, no reduction (production's normal reading, byte-identical); a coalesced
    rung asked 16 records the reduction 18 -> 16 and the kernel step 16 -> 15 from it; 14 and 12 are asked as themselves,
    the reduction recorded, the list empty; the operator's 12 likewise. The served S is the chain's end; ``identity`` is
    False whenever anything applied."""
    k18, k16 = sg.kernel_ceiling_adjustment("gemma4e4b", LADDER_L, 18), sg.kernel_ceiling_adjustment("gemma4e4b", LADDER_L, 16)
    assert (k18["to_S"], k16["to_S"], sg.kernel_ceiling_adjustment("gemma4e4b", LADDER_L, 14)) == (15, 15, None)
    assert _P(_gemma_block(18, 15, [k18])) is None
    assert _P(_gemma_block(16, 15, [k16], _reduction(16))) is None
    assert _P(_gemma_block(14, 14, [], _reduction(14))) is None
    assert _P(_gemma_block(12, 12, [], _reduction(12, "operator"))) is None
    # engine-agnostic shape: the reduction reads the same under native=None; a lens engine still refuses a kernel step
    assert _P(_gemma_block(14, 14, [], _reduction(14)), native=None) is None
    assert "LENS engine" in _P(_gemma_block(16, 15, [k16], _reduction(16)), native=False)
    # the kv step after a reduction hangs off the requested rung (the previous S), exactly as after a kernel step
    kv = _kv_record(14, conc=13.5, L_=LADDER_L)
    block = _gemma_block(14, 13, [sg.kv_shrink_adjustment(kv)], _reduction(14), kv_shrink=kv)
    assert _P(block) is None


def test_every_broken_ladder_reduction_is_refused_by_name():
    """The reproduction first: the probe asked the SERVED 15 directly, the core recorded [] and production refused
    for the missing kernel step 18 -> 15 — that verdict is unchanged. Then each way a reduction can lie."""
    k18 = sg.kernel_ceiling_adjustment("gemma4e4b", LADDER_L, 18)
    why = _P(_gemma_block(15, 15, []))
    assert "must record the 'lora_kernel_index_ceiling' step (18 -> 15; D=4096, ceiling=524,287) and this record names none" in why
    assert "must record the 'lora_kernel_index_ceiling' step (18 -> 15" in _P(_gemma_block(14, 14, []))          # a bare 14: no reduction
    cases = (
        (_gemma_block(13, 13, [], _reduction(13)), "to_S 13 is not a rung of the ladder [18, 16, 14, 12]"),
        (_gemma_block(14, 14, [], _reduction(14, from_S=17, ladder_seqs=[17, 14])), "from_S 17 is not the identity S 18"),
        (_gemma_block(14, 14, [], _reduction(14, ladder_seqs=[18, 14, 16, 12])), "not strictly descending"),
        (_gemma_block(14, 14, [], _reduction(14, ladder_seqs=[16, 14, 12])), "not headed by the identity S 18"),
        (_gemma_block(14, 14, [], _reduction(14, max_model_len=16_384)), "max_model_len 16384 is not this run's 32768"),
        (_gemma_block(14, 14, [], _reduction(14, rung_source="identity")), "rung_source 'identity' is not one of"),
        (_gemma_block(14, 14, [], _reduction(14, rule="whatever")), "rule 'whatever' is not"),
        (_gemma_block(14, 14, [], {**_reduction(14), "extra": 1}), "keys != "),
        (_gemma_block(14, 14, [], {k: v for k, v in _reduction(14).items() if k != "rule"}), "(missing ['rule'])"),
        (_gemma_block(14, 14, [], "14"), "is not a mapping"),
        (_gemma_block(18, 18, [], {**_reduction(14), "to_S": 18}), "to_S 18 is not below the identity S 18"),
        # a reduction with the standup's steps NOT following from it
        (_gemma_block(16, 15, [k18], _reduction(16)), "records from_S=18; the formula gives 16"),
        (_gemma_block(14, 13, [], _reduction(14)), "the served S 13 is not the ladder_reduction's to_S=14 (identity S 18) and the record names no adjustment"),
        (_gemma_block(14, 14, [], _reduction(14), identity=True), "identity=True with 0 adjustment(s) and a ladder_reduction"),
        (_gemma_block(14, 14, [dict(k18, from_S=14, to_S=13)], _reduction(14)), "does not lower the requested rung S 14"),
    )
    for block, needle in cases:
        why = _P(block)
        assert why and needle in why, (needle, why)
    # the record builder refuses what the validator refuses
    for kw, needle in ((dict(requested_S=13), "not a rung"), (dict(requested_S=18), "not below"), (dict(rung_source="identity"), "rung_source"),
                       (dict(ladder_seqs=(16, 14, 12)), "headed by")):
        args = dict(identity_S=18, requested_S=14, max_model_len=LADDER_L, ladder_seqs=GEMMA_LADDER, rung_source="probe")
        args.update(kw)
        with pytest.raises(ValueError, match=needle):
            sg.ladder_reduction_record(**args)
