"""The MATH-500 / GSM8K validators' STEERED arm contract and the
geometry-adjustment rule.

The fitted arm's route class is ``steered`` and admits TWO serving paths — the lens
hook (``+lens+``) and the native LoRA adapter (``+lora+``) — so every per-row rule
that used to say "lens" now asks WHICH engine and holds the store to that engine's
facts:

- ``route_class_of`` returns ``ROUTE_STEERED`` (the value moved from ``"lens"``;
  new analysis outputs only — no eval store records it, §8.3) and
  ``steered_path_of`` reads the path off the store's own serving identity;
- ``arm_contract_problem`` refuses a class/marker disagreement in EITHER direction,
  requires ``steer_positions`` to match the engine (``response`` for a lens mask,
  ``all`` for an adapter; ABSENT on a pre-change lens store is admitted), and on a
  ``+lora+`` manifest reads the STORED-RECORD contract through the one resolver
  (``serving_engine == lora_native`` ⇔ ``reduction_mode == not_applicable``);
- the geometry contract accepts a served S BELOW the budget's S only as a VALIDATED
  ordered adjustment (the LoRA kernel's int32 index ceiling: gemma 121→107,
  nemotron 161→86 at L 4864; a KV shrink composed after it), and refuses a bare
  smaller S, a ``to_S`` off by one, a kernel step on a model whose width does not
  lower, and steps out of order — "any smaller S" never passes (§8.12);
- the steering-bites twin is the ENGINE's zero cell (``none__lens0`` /
  ``none__lora0``) at both validators, and the probe inventory counts the probes on
  the path-matched twin vmodel.

Every manifest here is SYNTHETIC and built through the REAL producers (the geometry
resolver's ``ResolvedGeometry.as_dict()`` + ``geometry.kernel_ceiling_adjustment``; a
KV-shrunk chain through ``geometry.kv_shrunk_geometry`` itself, so the block carries the
producer's ``seqs_source: kv_derived`` — a fixture that stamps ``override`` by hand
hides that the reader refuses the producer's own source), so a
shape drift in either shows up as a failure rather than a passing fiction. CPU-only."""

from __future__ import annotations

import copy
import sys
from dataclasses import replace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import tinycap_math500_thinking as M  # noqa: E402
import tinycap_gsm8k_validators as V  # noqa: E402

from sisteer.capeval.serving import geometry as sg  # noqa: E402
from sisteer.capeval.tinycap import contracts as C  # noqa: E402

L = 4864
EVAL_4864 = next(k for k in sg.KNOWN_KEYS if sg.TABLE[k].max_model_len == L)
LENS_ID = "proxy_vllm+lens+interleaved"
LORA_ID = "proxy_vllm+lora+interleaved"
PASS_ID = "proxy_vllm+passthrough+interleaved"
CELL_LORA = "constv2-mix-full__lora_r4_ce"
NATIVE, LENS = C.SERVING_ENGINE_LORA_NATIVE, C.SERVING_ENGINE_LENS


def _geom(model_key: str, *, served_S: int | None = None, adjustments=None) -> dict:
    """The producer's ``geometry`` block: the resolver's own ``as_dict()`` (window
    from the table, S from the model budget) with the served S substituted and the
    ordered adjustments recorded beside it, exactly as the native core writes them."""
    g = sg.resolve_geometry(EVAL_4864, model_key=model_key)
    assert g.max_model_len == L and g.is_identity
    if served_S is not None and served_S != g.max_num_seqs:
        g = replace(g, max_num_seqs=served_S, seqs_source="override")
    block = g.as_dict()
    if adjustments is not None:
        block[sg.ADJUSTMENTS_KEY] = copy.deepcopy(adjustments)
    return block


def _identity(model_key: str) -> sg.ResolvedGeometry:
    g = sg.resolve_geometry(EVAL_4864, model_key=model_key)
    assert g.max_model_len == L and g.is_identity
    return g


def _kernel_lowered(model_key: str, step: dict) -> sg.ResolvedGeometry:
    """The geometry AFTER the kernel step, as the native core re-resolves it
    (``native_standup.lora_kernel_ceiling_geometry`` -> ``_resolve_geometry(...,
    max_num_seqs=lowered, override_reason=LORA_KERNEL_CEILING_REASON)``): the table
    window, the lowered S, ``seqs_source: override``."""
    g = sg.resolve_geometry(EVAL_4864, max_num_seqs=int(step["to_S"]), strict_identity=False, model_key=model_key)
    assert (g.max_model_len, g.max_num_seqs, g.seqs_source) == (L, int(step["to_S"]), "override")
    return g


def _kv_shrunk(prev: sg.ResolvedGeometry, kv_backed_concurrency: float) -> tuple[sg.ResolvedGeometry, dict]:
    """The geometry AFTER the KV step, THROUGH the producer (``geometry.kv_shrink_record``
    + ``geometry.kv_shrunk_geometry``, what ``native_standup.kv_derived_geometry`` and
    capctrl's re-launch call): the record names ``prev``'s S as configured and the
    result carries the producer's ``seqs_source: kv_derived``."""
    rec = sg.kv_shrink_record(configured_max_num_seqs=prev.max_num_seqs, max_model_len=L,
                              kv_backed_concurrency=kv_backed_concurrency,
                              kv_capacity_tokens=int(kv_backed_concurrency * L), basis="uniform")
    g = sg.kv_shrunk_geometry(prev, rec)
    assert g.seqs_source == sg.KV_DERIVED_SOURCE and g.max_num_seqs == rec["served_max_num_seqs"]
    return g, rec


def _block(g: sg.ResolvedGeometry, adjustments: list, kv_shrink: dict | None = None) -> dict:
    """The endpoint's ``geometry`` block as the native core writes it
    (``native_standup.LoraStandup._geometry_block``): ``as_dict()`` + the ordered
    ``adjustments`` + the ``kv_shrink`` record beside them when the KV step fired —
    the shape the driver copies into the manifest verbatim."""
    block = dict(g.as_dict())
    block[sg.ADJUSTMENTS_KEY] = [dict(a) for a in adjustments]
    if kv_shrink is not None:
        block[sg.KV_SHRINK_KEY] = dict(kv_shrink)
    return block


def _manifest(*, identity: str, cell: str = CELL_LORA, model_key: str = "llama8b",
              steer_positions: str | None = None, geometry: dict | None = None,
              serving_engine: str | None = None, reduction_mode: str | None = None,
              served_S: int | None = None, window: int = L) -> dict:
    """A leg's manifest as ``served_capability_run`` writes the blocks these rules
    read: the route (identity + the stored-record fields), the served geometry, the
    client width, and the steered-positions policy. ``window`` is the served
    ``max_model_len`` (the module's 4,864 unless a test serves another entry)."""
    geom = geometry if geometry is not None else _geom(model_key)
    S = served_S if served_S is not None else int(geom["max_num_seqs"])
    route: dict = {"virtual_model": f"petri__{cell}", "grader_virtual_model": "petri__base",
                   "serving_identity": identity, "tool_calls": False,
                   C.AGENT_LOOP_ENTRY_KEY: True}
    if reduction_mode is not None:
        route[C.REDUCTION_MODE_FIELD] = reduction_mode
    if serving_engine is not None:
        route[C.SERVED_ENGINE_FIELD] = serving_engine
    man: dict = {
        C.MANIFEST_KEY_ROUTE: route,
        C.MANIFEST_KEY_GEOMETRY: geom,
        C.MANIFEST_KEY_CONCURRENCY: S,
        C.MANIFEST_KEY_PROVENANCE: {"si_id": None, "artifact_sha256": "ab" * 32},
        "resolved_config": {"system_sha": None},
        C.MANIFEST_KEY_TINYCAP: {"served": {"max_model_len": window, "max_num_seqs": S,
                                            "concurrency": S}},
    }
    if steer_positions is not None:
        man["steer_positions"] = steer_positions
    return man


def _row(cell: str = CELL_LORA, *, model_key: str = "llama8b") -> dict:
    return {"cell": cell, "geometry_L": L, "geometry_eval": EVAL_4864, "model_key": model_key}


def _lens(**over) -> dict:
    return _manifest(identity=LENS_ID, steer_positions="response",
                     reduction_mode="per_row", **over)


def _native(**over) -> dict:
    return _manifest(identity=LORA_ID, steer_positions="all", serving_engine=NATIVE,
                     reduction_mode=C.REDUCTION_MODE_NOT_APPLICABLE, **over)


# ---------------------------------------------------------------------------
# 1. the vocabulary
# ---------------------------------------------------------------------------


def test_the_route_class_is_steered_and_the_path_comes_off_the_identity():
    assert (M.ROUTE_BASE, M.ROUTE_PROMPTED, M.ROUTE_STEERED) == ("base", "prompted", "steered")
    assert M.STEERED_SERVING_MARKERS == ("+lens+", "+lora+")
    assert not hasattr(M, "ROUTE_LENS") and not hasattr(M, "LENS_SERVING_MARKER")
    assert M.route_class_of(M.CELL_BARE) == M.ROUTE_BASE
    assert M.route_class_of(C.CELL_PROMPTED_SAFETY) == M.ROUTE_PROMPTED
    assert M.route_class_of(CELL_LORA) == M.ROUTE_STEERED
    assert M.route_class_of("si_safety__psr_kl_all") == M.ROUTE_STEERED
    assert M.steered_path_of(LENS_ID) == "lens" and M.steered_path_of(LORA_ID) == "lora"
    assert M.steered_path_of(PASS_ID) is None and M.steered_path_of(None) is None
    with pytest.raises(ValueError, match="two steering paths"):
        M.steered_path_of("proxy_vllm+lens+lora+interleaved")
    # the GSM8K instrument reads the SAME three names (one rule, re-exported). V binds them off ITS MATH-500 module
    # object, V.M — so the property is asserted AS that re-export plus "the same rule" (same file, same names, same
    # code, same answers), never as function IDENTITY against this test's `M`: the two can be different objects of
    # the same file in a full-suite run. test_eight_arms_gsm8k_analysis binds V early (through
    # tinycap_gsm8k_paired200, whose _load_module registers `tinycap_math500_thinking` once), and
    # test_eight_arms_math500_analysis then registers a FRESH `tinycap_math500_thinking` into sys.modules
    # unconditionally, so `import tinycap_math500_thinking` here returns the later load while V.M is the earlier one —
    # green alone and in the targeted union, red in the Modal suite's collection order:
    # "<function route_class_of at 0x…> != <function route_class_of at 0x…>". The rule is one whichever load bound it.
    assert Path(V.M.__file__).resolve() == Path(M.__file__).resolve()          # one source file behind both instruments
    for name in ("route_class_of", "arm_contract_problem", "steered_path_of"):
        v, m = getattr(V, name), getattr(M, name)
        assert v is getattr(V.M, name), name                                    # V re-exports its own MATH-500 module's function
        assert (v.__module__, v.__qualname__) == (m.__module__, m.__qualname__) == ("tinycap_math500_thinking", name)
        assert v.__code__ == m.__code__, name                                    # the same function body, load-independent
    for cell, expect in ((M.CELL_BARE, M.ROUTE_BASE), (C.CELL_PROMPTED_SAFETY, M.ROUTE_PROMPTED),
                         (CELL_LORA, M.ROUTE_STEERED), ("si_safety__psr_kl_all", M.ROUTE_STEERED)):
        assert V.route_class_of(cell) == M.route_class_of(cell) == expect, cell
    assert V.steered_path_of(LENS_ID) == "lens" and V.steered_path_of(LORA_ID) == "lora" and V.steered_path_of(PASS_ID) is None
    assert V.arm_contract_problem(_native(), row=_row()) is None is M.arm_contract_problem(_native(), row=_row())
    assert V.arm_contract_problem(_lens(), row=_row()) is None is M.arm_contract_problem(_lens(), row=_row())


# ---------------------------------------------------------------------------
# 2. the arm contract on both engines
# ---------------------------------------------------------------------------


def test_a_native_lora_arm_satisfies_its_contract():
    assert M.arm_contract_problem(_native(), row=_row()) is None
    # and so does the lens arm, exactly as before
    assert M.arm_contract_problem(_lens(), row=_row()) is None


def test_a_pre_change_lens_store_without_steer_positions_is_admitted():
    man = _lens()
    del man["steer_positions"]
    assert M.arm_contract_problem(man, row=_row()) is None


@pytest.mark.parametrize("identity,positions,needle", [
    (LORA_ID, "response", "'response' on a 'lora'-served arm"),
    (LENS_ID, "all", "'all' on a 'lens'-served arm"),
])
def test_the_positions_policy_must_match_the_engine(identity, positions, needle):
    man = _manifest(identity=identity, steer_positions=positions, reduction_mode="per_row")
    if identity == LORA_ID:
        man[C.MANIFEST_KEY_ROUTE][C.SERVED_ENGINE_FIELD] = NATIVE
        man[C.MANIFEST_KEY_ROUTE][C.REDUCTION_MODE_FIELD] = C.REDUCTION_MODE_NOT_APPLICABLE
    why = M.arm_contract_problem(man, row=_row())
    assert why and needle in why


def test_a_class_marker_disagreement_refuses_in_both_directions():
    # a steered cell served passthrough
    why = M.arm_contract_problem(_manifest(identity=PASS_ID, steer_positions=None), row=_row())
    assert why and "is not a steered route on the 'steered' arm" in why
    # the bare cell served on a steered route
    bare = _manifest(identity=LORA_ID, cell=M.CELL_BARE, steer_positions="all",
                     serving_engine=NATIVE, reduction_mode=C.REDUCTION_MODE_NOT_APPLICABLE)
    bare[C.MANIFEST_KEY_PROVENANCE]["artifact_sha256"] = None
    why = M.arm_contract_problem(bare, row=_row(M.CELL_BARE))
    assert why and "is a steered route on the 'base' arm" in why
    # an object served on an UNSTEERED arm still refuses by its provenance
    passthrough = _manifest(identity=PASS_ID, cell=M.CELL_BARE)
    why = M.arm_contract_problem(passthrough, row=_row(M.CELL_BARE))
    assert why and "on the unsteered arm" in why


@pytest.mark.parametrize("edit,needle", [
    # engine absent AND a lens mode: the resolver reads a LENS store, and a `+lora+` identity on a
    # lens store is exactly what the native contract refuses
    ({C.SERVED_ENGINE_FIELD: None, C.REDUCTION_MODE_FIELD: "per_row"},
     "serving_engine None on the natively served arm"),
    ({C.SERVED_ENGINE_FIELD: LENS, C.REDUCTION_MODE_FIELD: "per_row"},
     "serving_engine 'lens' on the natively served arm"),
])
def test_a_lora_manifest_must_carry_the_native_stored_record_contract(edit, needle):
    man = _native()
    for k, v in edit.items():
        if v is None:
            man[C.MANIFEST_KEY_ROUTE].pop(k, None)
        else:
            man[C.MANIFEST_KEY_ROUTE][k] = v
    why = M.arm_contract_problem(man, row=_row())
    assert why and needle in why


def test_a_lora_manifest_with_a_lens_reduction_mode_is_torn_and_refuses():
    man = _native()
    man[C.MANIFEST_KEY_ROUTE][C.REDUCTION_MODE_FIELD] = "stacked"
    why = M.arm_contract_problem(man, row=_row())
    # the ONE resolver's refusal: the engine and the mode are held to each other
    assert why and "lora_native" in why and "stacked" in why


def test_a_lora_manifest_whose_route_states_not_applicable_without_the_engine_refuses():
    man = _native()
    man[C.MANIFEST_KEY_ROUTE].pop(C.SERVED_ENGINE_FIELD)
    why = M.arm_contract_problem(man, row=_row())
    assert why and C.REDUCTION_MODE_NOT_APPLICABLE in why


# ---------------------------------------------------------------------------
# 3. the §1.8 geometry adjustment, through the validators
# ---------------------------------------------------------------------------


def _kernel(model_key: str, S_identity: int) -> dict:
    step = sg.kernel_ceiling_adjustment(model_key, L, S_identity)
    assert step is not None
    return step


def test_a_gemma_shaped_kernel_step_is_accepted_and_the_row_is_held_to_the_adjusted_s():
    step = _kernel("gemma4e4b", 121)
    assert (step["from_S"], step["to_S"]) == (121, 107)
    man = _native(model_key="gemma4e4b", served_S=107,
                  geometry=_geom("gemma4e4b", served_S=107, adjustments=[step]))
    assert M.arm_contract_problem(man, row=_row(model_key="gemma4e4b")) is None
    # the client width and the served S are the ADJUSTED S, not the budget's
    assert man[C.MANIFEST_KEY_TINYCAP]["served"]["max_num_seqs"] == 107
    # a client width left at the budget's S refuses
    wide = copy.deepcopy(man)
    wide[C.MANIFEST_KEY_TINYCAP]["served"]["concurrency"] = 121
    why = M.arm_contract_problem(wide, row=_row(model_key="gemma4e4b"))
    assert why and "concurrency" in why


def test_a_nemotron_shaped_kernel_step_and_a_kernel_kv_composition_are_accepted():
    step = _kernel("nemotron3nano4b", 161)
    assert (step["from_S"], step["to_S"]) == (161, 86)
    man = _native(model_key="nemotron3nano4b", served_S=86,
                  geometry=_geom("nemotron3nano4b", served_S=86, adjustments=[step]))
    assert M.arm_contract_problem(man, row=_row(model_key="nemotron3nano4b")) is None
    # the KV step composes AFTER the kernel one, from the kernel-lowered S — built THROUGH the producer:
    # gemma 121 -(kernel)-> 107 -(kv)-> 80, the block carrying `seqs_source: kv_derived` as `kv_shrunk_geometry` writes it
    gem = _kernel("gemma4e4b", 121)
    shrunk, rec = _kv_shrunk(_kernel_lowered("gemma4e4b", gem), 80.3)
    kv = sg.kv_shrink_adjustment(rec)
    assert (kv["from_S"], kv["to_S"], shrunk.max_num_seqs) == (107, 80, 80)
    block = _block(shrunk, [gem, kv], kv_shrink=rec)
    assert block["seqs_source"] == sg.KV_DERIVED_SOURCE == M.GEOMETRY_SEQS_SOURCE_KV_DERIVED
    man = _native(model_key="gemma4e4b", served_S=80, geometry=block)
    assert M.arm_contract_problem(man, row=_row(model_key="gemma4e4b")) is None
    assert V.arm_contract_problem(man, row=_row(model_key="gemma4e4b")) is None      # the GSM8K instrument inherits it


def test_a_kv_only_chain_built_through_kv_shrunk_geometry_is_accepted_with_the_producers_source():
    """A native llama8b standup (D does not lower at 4864, so no kernel step) whose container's KV cache
    backs fewer rows than the budget's S re-launches at floor(kv_backed_concurrency) — the ONE `[kv]` step, and the block
    reads `seqs_source: kv_derived`. The reader validated the step and then refused the source; it admits it now, and a
    lens KV-shrunk store (capctrl's re-launch writes the same record and source) reads the same way."""
    ident = _identity("llama8b")
    assert ident.max_num_seqs == 97
    shrunk, rec = _kv_shrunk(ident, 80.3)
    kv = sg.kv_shrink_adjustment(rec)
    block = _block(shrunk, [kv], kv_shrink=rec)
    assert (block["seqs_source"], block["identity"], block["max_num_seqs"]) == (sg.KV_DERIVED_SOURCE, False, 80)
    man = _native(model_key="llama8b", served_S=80, geometry=block)
    assert M.arm_contract_problem(man, row=_row()) is None
    assert V.arm_contract_problem(man, row=_row()) is None
    # the pre-contract spelling — the record alone, no `adjustments` key — derives the same one step
    legacy = {k: v for k, v in block.items() if k != sg.ADJUSTMENTS_KEY}
    assert M.arm_contract_problem(_native(model_key="llama8b", served_S=80, geometry=legacy), row=_row()) is None
    # a lens store shrunk the same way (a lens standup under --kv-derived-seqs) reads the same source rule
    lens = _manifest(identity=LENS_ID, steer_positions="response", reduction_mode="per_row",
                     model_key="llama8b", served_S=80, geometry=block)
    assert M.arm_contract_problem(lens, row=_row()) is None


@pytest.mark.parametrize("chain", ["kernel_only", "unadjusted"])
def test_a_kv_derived_source_without_a_kv_step_is_torn_and_refuses(chain):
    """`kv_derived` is admitted iff the chain ENDS in the KV step it names: beside a kernel-only chain it is torn
    (refused on the source, after the arithmetic passed); on an unadjusted block the historical rule refuses it."""
    if chain == "kernel_only":
        step = _kernel("gemma4e4b", 121)
        block = _block(_kernel_lowered("gemma4e4b", step), [step])
        block["seqs_source"] = sg.KV_DERIVED_SOURCE
        why = M.arm_contract_problem(_native(model_key="gemma4e4b", served_S=107, geometry=block), row=_row(model_key="gemma4e4b"))
        assert why and "seqs_source 'kv_derived' on an ADJUSTED geometry" in why and "['lora_kernel_index_ceiling']" in why
    else:
        block = _identity("llama8b").as_dict()
        block["seqs_source"] = sg.KV_DERIVED_SOURCE
        why = M.arm_contract_problem(_native(model_key="llama8b", geometry=block), row=_row())
        assert why and "seqs_source 'kv_derived' != 'model_budget'" in why


@pytest.mark.parametrize("chain,stale", [
    ("kv", M.GEOMETRY_SEQS_SOURCE_MODEL_BUDGET), ("kv", M.GEOMETRY_SEQS_SOURCE_OVERRIDE),
    ("kernel_kv", M.GEOMETRY_SEQS_SOURCE_MODEL_BUDGET), ("kernel_kv", M.GEOMETRY_SEQS_SOURCE_OVERRIDE),
    ("kernel", M.GEOMETRY_SEQS_SOURCE_MODEL_BUDGET),
])
def test_a_stale_source_beside_a_validated_chain_is_torn_and_refuses(chain, stale):
    """The earlier reader admitted `model_budget` and `override` on EVERY adjusted chain (and `kv_derived` iff the
    chain ended in KV), so a `[kv]` or `[kernel, kv]` record whose arithmetic re-derived but whose source was the
    PRE-shrink `override` or the unadjusted `model_budget` — a torn record: the chain says one thing, the source another
    — was admitted; a `[kernel]` chain sourced `model_budget` likewise. The source now follows the LAST validated step,
    one spelling per chain: `kv_derived` after a KV step, `override` after the kernel ceiling, `model_budget` only
    unadjusted. Each chain is built THROUGH the producer (its own source passes) and then torn by hand on the source
    alone; both instruments refuse it by name, and the lens KV-shrunk store shares the rule."""
    if chain == "kv":
        shrunk, rec = _kv_shrunk(_identity("llama8b"), 80.3)
        block, key, S = _block(shrunk, [sg.kv_shrink_adjustment(rec)], kv_shrink=rec), "llama8b", 80
    elif chain == "kernel_kv":
        gem = _kernel("gemma4e4b", 121)
        shrunk, rec = _kv_shrunk(_kernel_lowered("gemma4e4b", gem), 80.3)
        block, key, S = _block(shrunk, [gem, sg.kv_shrink_adjustment(rec)], kv_shrink=rec), "gemma4e4b", 80
    else:
        gem = _kernel("gemma4e4b", 121)
        block, key, S = _block(_kernel_lowered("gemma4e4b", gem), [gem]), "gemma4e4b", 107
    producer_source = block["seqs_source"]
    assert producer_source == (sg.KV_DERIVED_SOURCE if chain != "kernel" else M.GEOMETRY_SEQS_SOURCE_OVERRIDE) != stale
    assert M.arm_contract_problem(_native(model_key=key, served_S=S, geometry=block), row=_row(model_key=key)) is None
    torn = {**block, "seqs_source": stale}
    for inst in (M, V):
        why = inst.arm_contract_problem(_native(model_key=key, served_S=S, geometry=torn), row=_row(model_key=key))
        assert why, (chain, stale)
        assert f"geometry.seqs_source {stale!r} on an ADJUSTED geometry" in why and f"!= {producer_source!r}" in why, why
        assert "a stale source beside a validated chain is a torn record" in why, why
    if chain == "kv":   # capctrl's re-launch writes the same record on a LENS store: the same one rule
        lens = _manifest(identity=LENS_ID, steer_positions="response", reduction_mode="per_row",
                         model_key=key, served_S=S, geometry=torn)
        why = M.arm_contract_problem(lens, row=_row(model_key=key))
        assert why and f"geometry.seqs_source {stale!r} on an ADJUSTED geometry" in why


def test_a_llama_shaped_native_row_is_identity_and_needs_no_adjustment():
    assert sg.kernel_ceiling_adjustment("llama8b", L, 97) is None   # D does not lower at this L
    for adj in ([], None):
        man = _native(model_key="llama8b",
                      geometry=_geom("llama8b", adjustments=adj))
        assert M.arm_contract_problem(man, row=_row()) is None


@pytest.mark.parametrize("mutate,needle", [
    ("bare_smaller_S", "!= the budget's S"),
    ("off_by_one", "records to_S=106"),
    ("reversed", ""),
    ("kernel_on_llama", "does not lower the identity S"),
    ("kv_from_identity_S", ""),
])
def test_an_unvalidated_smaller_s_never_passes(mutate, needle):
    if mutate == "bare_smaller_S":
        # the kernel's number with NO record of the step, and the block still claiming the budget's
        # own source: the unadjusted S rule refuses it — a smaller S is never accepted on its own
        # a block that looks ORDINARY in every other respect — the budget's own source, the
        # identity attestation — and just serves fewer rows: the S rule is the one that catches it
        block = _geom("gemma4e4b", served_S=107)
        block["seqs_source"] = M.GEOMETRY_SEQS_SOURCE_MODEL_BUDGET
        block["identity"] = True
        man = _native(model_key="gemma4e4b", served_S=107, geometry=block)
        why = M.arm_contract_problem(man, row=_row(model_key="gemma4e4b"))
        # ... and with the producer's override spelling it refuses on the source instead
        other = M.arm_contract_problem(
            _native(model_key="gemma4e4b", served_S=107, geometry=_geom("gemma4e4b", served_S=107)),
            row=_row(model_key="gemma4e4b"))
        assert other and "seqs_source 'override'" in other
    elif mutate == "off_by_one":
        step = {**_kernel("gemma4e4b", 121), "to_S": 106}
        man = _native(model_key="gemma4e4b", served_S=106,
                      geometry=_geom("gemma4e4b", served_S=106, adjustments=[step]))
        why = M.arm_contract_problem(man, row=_row(model_key="gemma4e4b"))
    elif mutate == "reversed":
        gem = _kernel("gemma4e4b", 121)
        rec = sg.kv_shrink_record(configured_max_num_seqs=107, max_model_len=L,
                                  kv_backed_concurrency=80.3, kv_capacity_tokens=int(80.3 * L),
                                  basis="uniform")
        kv = sg.kv_shrink_adjustment(rec)
        man = _native(model_key="gemma4e4b", served_S=int(kv["to_S"]),
                      geometry=_geom("gemma4e4b", served_S=int(kv["to_S"]), adjustments=[kv, gem]))
        why = M.arm_contract_problem(man, row=_row(model_key="gemma4e4b"))
    elif mutate == "kernel_on_llama":
        fake = {"kind": sg.ADJUSTMENT_KERNEL, "from_S": 97, "to_S": 90, "D": 4096,
                "ceiling": 2 ** 31 - 1 // 4096}
        fake["ceiling"] = (2 ** 31 - 1) // 4096
        man = _native(model_key="llama8b", served_S=90,
                      geometry=_geom("llama8b", served_S=90, adjustments=[fake]))
        why = M.arm_contract_problem(man, row=_row())
    else:  # a kv_shrink measured from the IDENTITY S while a kernel step precedes it
        gem = _kernel("gemma4e4b", 121)
        rec = sg.kv_shrink_record(configured_max_num_seqs=121, max_model_len=L,
                                  kv_backed_concurrency=80.3, kv_capacity_tokens=int(80.3 * L),
                                  basis="uniform")
        kv = sg.kv_shrink_adjustment(rec)
        man = _native(model_key="gemma4e4b", served_S=int(kv["to_S"]),
                      geometry=_geom("gemma4e4b", served_S=int(kv["to_S"]), adjustments=[gem, kv]))
        why = M.arm_contract_problem(man, row=_row(model_key="gemma4e4b"))
    assert why is not None and (needle in why if needle else True)


def test_an_adjusted_block_claiming_identity_refuses():
    step = _kernel("gemma4e4b", 121)
    block = _geom("gemma4e4b", served_S=107, adjustments=[step])
    block["identity"] = True
    man = _native(model_key="gemma4e4b", served_S=107, geometry=block)
    why = M.arm_contract_problem(man, row=_row(model_key="gemma4e4b"))
    assert why and "identity is True iff no adjustment applied" in why


def test_a_lens_row_keeps_the_unadjusted_rules_in_their_original_order():
    """A lens store's geometry is judged exactly as before: the window source first,
    then the budget's S — the adjustment branch is not entered (``adjustments`` is
    absent), so no historical refusal changes its wording or its order."""
    man = _lens(model_key="gemma4e4b")
    assert M.arm_contract_problem(man, row=_row(model_key="gemma4e4b")) is None
    off_window = copy.deepcopy(man)
    off_window[C.MANIFEST_KEY_GEOMETRY]["window_source"] = "override"
    why = M.arm_contract_problem(off_window, row=_row(model_key="gemma4e4b"))
    assert why == ("geometry.window_source 'override' != 'table' — the served window was not "
                   "the table entry's")
    lens_block = _geom("gemma4e4b", served_S=107)
    lens_block["seqs_source"] = M.GEOMETRY_SEQS_SOURCE_MODEL_BUDGET
    lens_block["identity"] = True
    off_S = _manifest(identity=LENS_ID, steer_positions="response", reduction_mode="per_row",
                      model_key="gemma4e4b", served_S=107, geometry=lens_block)
    why = M.arm_contract_problem(off_S, row=_row(model_key="gemma4e4b"))
    assert why and "!= the budget's S 121" in why


# ---------------------------------------------------------------------------
# 3b. the kernel lowering re-stamps the UNCHANGED table window `override`
# ---------------------------------------------------------------------------
#
# A natively served leg that hits the LoRA kernel's int32 index ceiling re-resolves its geometry
# with the table window passed EXPLICITLY beside the lowered S (`native_standup.
# lora_kernel_ceiling_geometry` -> `_resolve_geometry(eval, L, lowered_S, ...)`), and
# `resolve_geometry`'s `pick()` marks every explicit factor `override` — so the block reads
# `window_source: override` although L IS the table entry's. The window rule refused every such leg
# before it reached the §1.8 validation (the pulled qwen35_4b `arditi_allpos` thinking legs:
# `capability:gen_long_thinking`, 10,752 x 48 under [kernel 69 -> 48, D 4096, ceiling 524,287],
# budget 741,888, batched 516,096, identity false, both sources `override`). The carve
# (`M._override_window_problem`) derives the table's window and admits the stamp ONLY when the served
# window is it, a known adjustment chain is recorded and the served S is the chain's last `to_S`.

THINKING_EVAL, RUNG_EVAL, QWEN = "capability:gen_long_thinking", "capability:gen_long_gsm8k_rung", "qwen35_4b"
#: the pulled manifest's geometry block, number for number (the evidence the carve was written against)
PULLED_THINKING = {"eval": THINKING_EVAL, "model_key": QWEN, "max_model_len": 10752, "max_num_seqs": 48,
                   "budget": 741888, "batched_tokens": 516096, "identity": False,
                   "window_source": "override", "seqs_source": "override",
                   sg.ADJUSTMENTS_KEY: [{"kind": sg.ADJUSTMENT_KERNEL, "from_S": 69, "to_S": 48, "D": 4096, "ceiling": 524287}]}


def _kernel_lowered_at_table_window(eval_name: str, model_key: str) -> tuple[dict, sg.ResolvedGeometry, dict]:
    """The block a native standup writes when the kernel ceiling lowers S at the TABLE window, through
    the real producers: the identity resolution, the pure kernel step, then the re-resolution with
    BOTH factors explicit exactly as ``native_standup._resolve_geometry`` calls it under an override
    reason — which is what stamps the unchanged window ``override``. Returns ``(block, identity, step)``."""
    identity = sg.resolve_geometry(eval_name, model_key=model_key)
    assert identity.window_source == "table" and identity.is_identity
    step = sg.kernel_ceiling_adjustment(model_key, identity.max_model_len, identity.max_num_seqs)
    assert step is not None
    lowered = sg.resolve_geometry(eval_name, max_model_len=identity.max_model_len, max_num_seqs=int(step["to_S"]),
                                  strict_identity=False, model_key=model_key)
    # the defect's mechanism, pinned: the window did not move, the stamp says it did
    assert lowered.max_model_len == identity.max_model_len
    assert (lowered.window_source, lowered.seqs_source) == ("override", "override")
    return _block(lowered, [step]), identity, step


def _native_at(block: dict, *, model_key: str = QWEN) -> dict:
    return _native(model_key=model_key, served_S=int(block["max_num_seqs"]), geometry=block,
                   window=int(block["max_model_len"]))


def _gsm8k_row(model_key: str = QWEN) -> dict:
    """A GSM8K matrix row: no ``geometry_L`` / ``geometry_eval`` (the GSM8K matrices carry none), so
    the window is held ONLY through the geometry block's own eval — the derivation the carve makes."""
    return {"cell": CELL_LORA, "model_key": model_key}


def test_the_pulled_qwen_thinking_block_is_what_the_producers_write():
    """The fixture is the real evidence, number for number: the block the producers write for
    ``capability:gen_long_thinking`` on qwen35_4b equals the pulled manifest's geometry block on every
    field the reader judges — so the positive case below is the pulled leg, not a fiction."""
    block, identity, step = _kernel_lowered_at_table_window(THINKING_EVAL, QWEN)
    assert (identity.max_model_len, identity.max_num_seqs, identity.budget_tokens) == (10752, 69, 741888)
    assert step == PULLED_THINKING[sg.ADJUSTMENTS_KEY][0]
    assert {k: block[k] for k in PULLED_THINKING} == PULLED_THINKING


@pytest.mark.parametrize("eval_name,L_table,S_identity,S_served", [
    (THINKING_EVAL, 10752, 69, 48),       # the pulled thinking_card store
    (RUNG_EVAL, 18944, 39, 27),           # the pulled thinking_card__rung16384 store beside it
])
def test_a_native_leg_whose_window_is_the_tables_and_whose_s_was_kernel_lowered_is_admitted(eval_name, L_table, S_identity, S_served):
    block, identity, step = _kernel_lowered_at_table_window(eval_name, QWEN)
    assert (block["max_model_len"], block["max_num_seqs"], step["from_S"], step["to_S"]) == (L_table, S_served, S_identity, S_served)
    man = _native_at(block)
    # both readers, both row shapes: the GSM8K row (no geometry_L: the carve's derivation is the only window rule)
    # and a MATH-shaped row that pins the window and entry itself
    assert M.arm_contract_problem(man, row=_gsm8k_row()) is None
    assert V.arm_contract_problem(man, row=_gsm8k_row()) is None
    assert M.arm_contract_problem(man, row={"cell": CELL_LORA, "model_key": QWEN, "geometry_L": L_table, "geometry_eval": eval_name}) is None
    # ... and the §1.8 branch is what it fell through to: the same block at a to_S off by one refuses THERE by the formula
    off = copy.deepcopy(block)
    off[sg.ADJUSTMENTS_KEY][0]["to_S"] = S_served - 1
    off["max_num_seqs"] = S_served - 1
    off["batched_tokens"] = (S_served - 1) * L_table
    why = M.arm_contract_problem(_native_at(off), row=_gsm8k_row())
    assert why and f"records to_S={S_served - 1}" in why and "the formula gives" in why


def test_a_kernel_then_kv_chain_at_the_table_window_is_admitted_through_the_carve():
    """The KV step re-states a kernel-lowered geometry (window_source stays `override`, seqs_source
    becomes `kv_derived`); the carve holds the served S to the LAST step's to_S."""
    block, identity, step = _kernel_lowered_at_table_window(THINKING_EVAL, QWEN)
    L_t = identity.max_model_len
    lowered = sg.resolve_geometry(THINKING_EVAL, max_model_len=L_t, max_num_seqs=int(step["to_S"]),
                                  strict_identity=False, model_key=QWEN)
    rec = sg.kv_shrink_record(configured_max_num_seqs=int(step["to_S"]), max_model_len=L_t,
                              kv_backed_concurrency=40.6, kv_capacity_tokens=int(40.6 * L_t), basis="uniform")
    shrunk = sg.kv_shrunk_geometry(lowered, rec)
    assert (shrunk.window_source, shrunk.seqs_source, shrunk.max_num_seqs) == ("override", sg.KV_DERIVED_SOURCE, 40)
    chain = _block(shrunk, [step, sg.kv_shrink_adjustment(rec)], kv_shrink=rec)
    assert M.arm_contract_problem(_native_at(chain), row=_gsm8k_row()) is None
    # the kernel step's to_S is NOT the served S here — a served S at the kernel's 48 with the KV step recorded refuses
    torn = copy.deepcopy(chain)
    torn["max_num_seqs"] = int(step["to_S"])
    torn["batched_tokens"] = int(step["to_S"]) * L_t
    why = M.arm_contract_problem(_native_at(torn), row=_gsm8k_row())
    assert why and f"the served max_num_seqs {step['to_S']} != the last step's to_S 40" in why


TODAY = "geometry.window_source 'override' != 'table' — the served window was not the table entry's"


@pytest.mark.parametrize("twin,needle", [
    ("window_differs", "the served window 10,624 != the table entry's 10,752 for 'capability:gen_long_thinking' ('qwen35_4b')"),
    ("no_adjustments", TODAY),
    ("s_not_to_s", "the served max_num_seqs 47 != the last step's to_S 48"),
    ("unknown_kind", "geometry.adjustments[0].kind 'made_up' is not one of ['launch_class', 'lora_kernel_index_ceiling', 'kv_shrink']"),
    ("unknown_eval", "geometry.eval 'capability:no_such_entry' ('qwen35_4b') resolves to no table entry"),
    ("no_eval", "geometry.eval None names no table entry"),
])
def test_the_override_stamp_is_never_trusted_the_known_bad_twins_refuse(twin, needle):
    """Each twin differs from the admitted block in exactly the respect the carve's conditions name,
    and refuses with a message naming it. ``no_adjustments`` is a plain override — its refusal is
    today's, byte for byte."""
    block, identity, step = _kernel_lowered_at_table_window(THINKING_EVAL, QWEN)
    L_t, S_s = identity.max_model_len, int(step["to_S"])
    if twin == "window_differs":
        # the same chain served 128 tokens short of the table window: a REAL override, refused by the two numbers
        block["max_model_len"] = L_t - 128
        block["batched_tokens"] = S_s * (L_t - 128)
    elif twin == "no_adjustments":
        del block[sg.ADJUSTMENTS_KEY]
    elif twin == "s_not_to_s":
        block["max_num_seqs"] = S_s - 1
        block["batched_tokens"] = (S_s - 1) * L_t
    elif twin == "unknown_kind":
        block[sg.ADJUSTMENTS_KEY][0]["kind"] = "made_up"
    elif twin == "unknown_eval":
        block["eval"] = "capability:no_such_entry"
    else:
        block["eval"] = None
    why = M.arm_contract_problem(_native_at(block), row=_gsm8k_row())
    assert why is not None and needle in why, why
    if twin == "no_adjustments":
        assert why == TODAY
    assert V.arm_contract_problem(_native_at(block), row=_gsm8k_row()) == why


def test_a_window_source_other_than_override_refuses_as_before_even_beside_a_chain():
    """The carve is for the kernel lowering's `override` spelling only: `fallback` (a no-eval legacy
    geometry) or a measurement footprint beside a recorded chain keeps today's wording."""
    block, _, _ = _kernel_lowered_at_table_window(THINKING_EVAL, QWEN)
    for source in ("fallback", sg.MEASUREMENT_SOURCE, None):
        block["window_source"] = source
        why = M.arm_contract_problem(_native_at(block), row=_gsm8k_row())
        assert why == f"geometry.window_source {source!r} != 'table' — the served window was not the table entry's"


# ---------------------------------------------------------------------------
# 4. the steering-bites twin is the engine's zero cell
# ---------------------------------------------------------------------------


def _bites(twin: str) -> dict:
    return {C.OOS_EVIDENCE_BITES_APPLICABLE: True, C.OOS_EVIDENCE_BITES_VERIFIED: True,
            C.OOS_EVIDENCE_BITES_N_DIFFERING: 3, "n_probes": 8,
            C.OOS_EVIDENCE_BITES_TWIN: twin}


@pytest.mark.parametrize("identity,twin", [(LENS_ID, C.TWIN_CELL_ID), (LORA_ID, C.TWIN_CELL_ID_LORA)])
def test_the_gsm8k_bites_twin_is_the_engines_zero_cell(identity, twin):
    assert V.steering_bites_problems(_bites(twin), serving_identity=identity) == []
    other = C.TWIN_CELL_ID_LORA if twin == C.TWIN_CELL_ID else C.TWIN_CELL_ID
    why = V.steering_bites_problems(_bites(other), serving_identity=identity)
    assert why and twin in why[0] and "the ENGINE's zero cell" in why[0]
    # no identity given: the lens twin, byte-identically to every pre-engine caller
    assert V.steering_bites_problems(_bites(C.TWIN_CELL_ID)) == []
    assert f"!= {C.TWIN_CELL_ID!r}" in V.steering_bites_problems(_bites(C.TWIN_CELL_ID_LORA))[0]
    # a passthrough identity on a fitted route cannot name a twin — reported, never guessed
    why = V.steering_bites_problems(_bites(twin), serving_identity=PASS_ID)
    assert why and "twin cannot be resolved" in why[0]
    # the unsteered-leg wording moved off "lens"
    inert = V.steering_bites_problems({**_bites(twin), C.OOS_EVIDENCE_BITES_VERIFIED: False},
                                      serving_identity=identity)
    assert any("not verified on a steered leg" in p for p in inert)


@pytest.mark.parametrize("identity,twin_vm", [
    (LENS_ID, None), (LORA_ID, None)])
def test_the_probe_inventory_counts_the_path_matched_twin(identity, twin_vm):
    want_twin = C.twin_vmodel_for(identity)
    evidence = {M.EVIDENCE_BITES: {"n_probes": 8,
                                   C.OOS_EVIDENCE_BITES_PROBE_RENDER_ARM: C.THINKING_OFF},
                # the lens-ON bite's own two legs (a LENS predicate the ON experiment records; the
                # inventory needs it to place its probes — see the report's note on this seam)
                M.EVIDENCE_LENS_ON_BITE: {"lens_vmodel": f"petri__{CELL_LORA}",
                                          "base_vmodel": C.cell_spellings(M.CELL_BARE, declared=False).vmodel,
                                          "n_probes": 1}}
    inv, problem = M.probe_inventory_per_run(evidence, route_vmodel=f"petri__{CELL_LORA}",
                                             fitted=True, serving_identity=identity)
    assert problem is None
    assert inv[(want_twin, C.THINKING_OFF, C.PROBE_KIND_BITES)] == 8
    other = C.twin_vmodel_for(LORA_ID if identity == LENS_ID else LENS_ID)
    assert (other, C.THINKING_OFF, C.PROBE_KIND_BITES) not in inv
    # the NO-SWITCH class (W-A d76d51a1 + the driver threading the identity through): the
    # bites probes land on the ENGINE's twin under `not_available`, and nowhere else
    no_switch, problem = M.probe_inventory_per_run(
        evidence, route_vmodel=f"petri__{CELL_LORA}", fitted=True,
        cert_class=C.CERT_CLASS_NO_SWITCH, serving_identity=identity)
    assert problem is None
    assert no_switch[(want_twin, C.THINKING_NOT_AVAILABLE, C.PROBE_KIND_BITES)] == 8
    assert (other, C.THINKING_NOT_AVAILABLE, C.PROBE_KIND_BITES) not in no_switch
    assert all(arm == C.THINKING_NOT_AVAILABLE for (_vm, arm, _kind) in no_switch)
    # no identity: the lens twin (every pre-engine caller unchanged)
    plain, _ = M.probe_inventory_per_run(evidence, route_vmodel=f"petri__{CELL_LORA}", fitted=True)
    assert (C.twin_vmodel(), C.THINKING_OFF, C.PROBE_KIND_BITES) in plain


@pytest.mark.parametrize("identity", [LENS_ID, LORA_ID])
def test_the_logprob_liveness_probes_count_against_the_engines_zero_cell_too(identity):
    """The logprob-liveness legs (one per prompt on the route, TWO per prompt on the twin) counted against the LENS
    twin spelled by hand while the bites legs beside them counted against the engine's — so a natively served leg's
    `none__lora0` logprob probes read as records the driver never sends beside a shortfall on `none__lens0` (the pulled
    qwen35_4b `arditi_allpos` legs: 10 lora-twin records, 0 lens-twin ones). The twin is the REGISTRY's zero cell for the
    route's engine (`contracts.twin_vmodel_for`: `defense_cells.NOOP_LENS_CELL_ID` / `NOOP_LORA_CELL_ID` through the
    control-leg vmodel rule) — never a spelling in the reader."""
    want_twin, other = C.twin_vmodel_for(identity), C.twin_vmodel_for(LORA_ID if identity == LENS_ID else LENS_ID)
    evidence = {M.EVIDENCE_BITES: {"n_probes": 5, C.OOS_EVIDENCE_BITES_PROBE_RENDER_ARM: C.THINKING_OFF,
                                   C.OOS_EVIDENCE_BITES_LOGPROB_LIVENESS: {"n_probes": 5}},
                M.EVIDENCE_LENS_ON_BITE: {"lens_vmodel": f"petri__{CELL_LORA}",
                                          "base_vmodel": C.cell_spellings(M.CELL_BARE, declared=False).vmodel, "n_probes": 1}}
    route_vm = f"petri__{CELL_LORA}"
    inv, problem = M.probe_inventory_per_run(evidence, route_vmodel=route_vm, fitted=True, serving_identity=identity)
    assert problem is None
    lp = C.PROBE_KIND_LOGPROB_BITES
    # the route's one leg per prompt and the ENGINE twin's two (the scale-0 leg and its repeat) — the pulled record's 5 + 10
    assert inv[(route_vm, C.THINKING_OFF, lp)] == 5 and inv[(want_twin, C.THINKING_OFF, lp)] == 2 * 5 == (len(C.BITES_LEGS) - 1) * 5
    assert inv[(want_twin, C.THINKING_OFF, C.PROBE_KIND_BITES)] == 5
    # the known-bad twins, as the reader's compare reads them (a probe keyed outside the inventory is a stray; a slot
    # short of its count is a shortfall): (a) a native leg whose logprob probes address the LENS twin, (b) a lens leg whose
    # probes address the LORA twin — the other engine's zero cell is in neither engine's inventory — and (c) a third name
    assert (other, C.THINKING_OFF, lp) not in inv and (other, C.THINKING_OFF, C.PROBE_KIND_BITES) not in inv
    assert ("petri__none__somebody_else", C.THINKING_OFF, lp) not in inv
    assert {vm for (vm, _arm, kind) in inv if kind == lp} == {route_vm, want_twin}
    # the logprob and bites legs name ONE twin — the engine's — never one each
    assert {vm for (vm, _arm, kind) in inv if kind in (lp, C.PROBE_KIND_BITES)} == {route_vm, want_twin}
    # no identity: the lens twin for both kinds (every pre-engine caller unchanged)
    plain, _ = M.probe_inventory_per_run(evidence, route_vmodel=route_vm, fitted=True)
    assert plain[(C.twin_vmodel(), C.THINKING_OFF, lp)] == 10 and (C.twin_vmodel_for(LORA_ID), C.THINKING_OFF, lp) not in plain
    # the no-switch class reads the same engine twin for the logprob legs (contracts.no_switch_probe_inventory, unchanged)
    ns, problem = M.probe_inventory_per_run(evidence, route_vmodel=route_vm, fitted=True, cert_class=C.CERT_CLASS_NO_SWITCH,
                                            serving_identity=identity)
    assert problem is None and ns[(want_twin, C.THINKING_NOT_AVAILABLE, lp)] == 10 and (other, C.THINKING_NOT_AVAILABLE, lp) not in ns
