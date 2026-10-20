"""``petri_serving_gate.hold_live_inventory`` holds the PASSTHROUGH
legs of a native standup too.

ONE native standup serves the unprompted ``none`` cell and every prompted cell
BESIDE its LoRA cells, so the gate's pre-probe inventory walk meets routes that
select no adapter. It used to ``continue`` past every non-lora route, which
left three mislabels invisible: a LENS route on a plugin-free server (base
served under a steered identity), an adapter served under a passthrough cell's
name (a leg the record says nothing steers), and a prompted leg whose H8 prefix
hash does not match the table's own spec for its vmodel. Each is now a named
``GateError`` before any probe, and the evidence records which routes were
bound by adapter and which by the record.

The prefix binding is RECORD-INTERNAL on purpose: ``/v1/models.petri_proxy``
publishes the engine block, ``base_model``, the agent-loop / lens facts and the
serve declarations — never the prompted prefixes — and the prefix BYTES never
reach the proxy by design (containment). The docstring says so rather
than implying a live comparison the gate cannot make.

Every record here is SYNTHETIC and hand-built to the real shapes
(``serving_attestation.engine_identity``'s input, ``proxy.ADAPTER_BLOCK_KEYS``,
``proxy.RouteEntry``, ``proxy.PromptedPrefixSpec``), so no GPU standup, no
tokenizer and no cached model are needed. CPU-only."""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import petri_serving_gate as gate  # noqa: E402

from sisteer.capeval.serving import proxy as px  # noqa: E402
from sisteer.capeval.serving import serving_attestation as sa  # noqa: E402

BASE_MODEL = "google/gemma-4-E4B-it"
CELL = "constv2-mix-full__lora_r4_ce"
ZERO = "none__lora0"
PROMPTED = "prompted__safety"
PREFIX_SHA = "ac3a58f61d53cf24"
IH_CELL, IH_NONE = "a" * 16, "b" * 16
LORA_ID = "proxy_vllm+lora+interleaved"
PASS_ID = "proxy_vllm+passthrough+interleaved"
LENS_ID = "proxy_vllm+lens+interleaved"


def _digests(name: str) -> dict:
    return {"config_sha256": f"{ord(name[0]):02x}" * 32,
            "weights_sha256": f"{(ord(name[0]) + 1) % 256:02x}" * 32}


def _attestation() -> dict:
    """The shape ``engine_identity`` reads (a record that already passed
    ``read_serving_attestation``): the engine state, the LoRA config and, per
    adapter NAME, the two file digests, the fingerprint, the int id, the
    intervention hash and the zero flag."""
    adapters = {}
    for name, ih, zero in ((CELL, IH_CELL, False), (ZERO, IH_NONE, True)):
        adapters[name] = {**_digests(name), "fingerprint_bf16": f"fp-{name}",
                          "lora_int_id": 1 if zero else 2,
                          "petri_intervention_hash": None if zero else ih,
                          "petri_zero_adapter": zero}
    return {"kind": px.ENGINE_KIND_LORA_NATIVE,
            "engine": {"enforce_eager": False, "cudagraph_mode": "FULL_AND_PIECEWISE"},
            "lora_config": {"max_loras": 2, "max_lora_rank": 4},
            "adapters": adapters}


def _routes() -> tuple:
    """The four routes ONE native standup exports: the base and a prompted cell
    (passthrough), the LoRA cell, and the derived zero-adapter control."""
    return (
        px.RouteEntry(virtual_model="petri__base", cell_id="none", cell_class="base",
                      serving_identity=PASS_ID, intervention_hash=IH_NONE),
        px.RouteEntry(virtual_model=f"petri__{PROMPTED}", cell_id=PROMPTED, cell_class="prompted",
                      serving_identity=PASS_ID, intervention_hash="c" * 16,
                      si_prefix_hash=PREFIX_SHA),
        px.RouteEntry(virtual_model=f"petri__{CELL}", cell_id=CELL, cell_class="class_a",
                      serving_identity=LORA_ID, intervention_hash=IH_CELL),
        px.RouteEntry(virtual_model=f"petri__{ZERO}", cell_id=ZERO, cell_class="control",
                      serving_identity=LORA_ID, intervention_hash=IH_NONE, control_of="none"),
    )


def _table(routes=None) -> dict:
    """A loaded version-2 table (``load_routing_table``'s output shape): a
    validated adapter block per lora route, the H8 prefix spec per prompted
    route, and nothing for a passthrough leg."""
    routes = routes if routes is not None else _routes()
    blocks = {}
    for r in routes:
        if r.serving_identity != LORA_ID:
            continue
        zero = r.cell_id.endswith(px.LORA_ZERO_CELL_SUFFIX)
        blocks[r.cell_id] = {"name": r.cell_id, "dir": f"/bank/{r.cell_id}", **_digests(r.cell_id),
                             "r": 4, "petri_intervention_hash": r.intervention_hash,
                             "petri_zero_adapter": zero,
                             "petri_source_kind": None if zero else "lora_all"}
    prefixes = tuple(px.PromptedPrefixSpec(virtual_model=r.virtual_model, byte_length=512,
                                           sha16=r.si_prefix_hash)
                     for r in routes if r.si_prefix_hash is not None)
    return {"version": 2, "base_model": BASE_MODEL, "serving_mode": "interleaved",
            "exact_replay_strategy": None, "base_revision": "ee" * 20, "routes": routes,
            "prompted_prefixes": prefixes, px.ADAPTERS_KEY: blocks}


def _models(att: dict, routes=None) -> dict:
    """The live ``/v1/models`` body a native proxy publishes: every route's
    vmodel served, and the engine block echoing the attestation verbatim."""
    routes = routes if routes is not None else _routes()
    ident = sa.engine_identity(att)
    return {"object": "list",
            "data": [{"id": r.virtual_model, "object": "model", "owned_by": "petri"} for r in routes],
            "petri_proxy": {"version": 1, "base_model": BASE_MODEL,
                            "petri_agent_loop_entry": True,
                            px.ENGINE_KEY: {"kind": ident["kind"],
                                            "enforce_eager": ident["enforce_eager"],
                                            "cudagraph_mode": ident["cudagraph_mode"],
                                            "adapters": copy.deepcopy(ident["adapters"])}}}


def _hold(models=None, table=None, att=None):
    att = att if att is not None else _attestation()
    return gate.hold_live_inventory(models if models is not None else _models(att),
                                    table=table if table is not None else _table(),
                                    attestation=att)


# ---------------------------------------------------------------------------
# the positive: a mixed native standup binds, and the evidence says how
# ---------------------------------------------------------------------------


def test_a_mixed_native_standup_binds_its_adapters_and_its_passthrough_legs():
    held = _hold()
    assert held["held"] is True
    assert held["engine"]["kind"] == px.ENGINE_KIND_LORA_NATIVE
    assert set(held["engine"]["adapters"]) == {CELL, ZERO}
    # the passthrough legs are RECORDED, not skipped in silence
    assert held["passthrough_routes"] == sorted(["none", PROMPTED])
    assert "no live adapter of the cell's name" in held["passthrough_binding"]
    assert "prompted_prefixes" in held["passthrough_binding"]
    assert held["served_models"] == sorted(r.virtual_model for r in _routes())
    assert held["base_model"] == BASE_MODEL


def test_a_zero_only_native_standup_binds_with_no_prompted_leg():
    """A view whose objects were all skipped stands up with the zero adapter
    alone (§1.2): the base route plus ``none__lora0``, no prompted leg."""
    routes = tuple(r for r in _routes() if r.cell_id in ("none", ZERO))
    att = _attestation()
    att["adapters"].pop(CELL)
    held = _hold(models=_models(att, routes), table=_table(routes), att=att)
    assert held["passthrough_routes"] == ["none"] and set(held["engine"]["adapters"]) == {ZERO}


# ---------------------------------------------------------------------------
# the negatives the branch exists for
# ---------------------------------------------------------------------------


def test_an_adapter_served_under_a_passthrough_cells_name_refuses():
    """The mislabel in the other direction from the lora branch's: the proxy
    serves an adapter named after a leg whose identity says nothing steers it."""
    att = _attestation()
    att["adapters"]["none"] = {**_digests("none"), "fingerprint_bf16": "fp-none",
                               "lora_int_id": 3, "petri_intervention_hash": "d" * 16,
                               "petri_zero_adapter": False}
    with pytest.raises(gate.GateError) as ei:
        _hold(models=_models(att), att=att)
    msg = str(ei.value)
    assert "passthrough route 'none': the live proxy serves an ADAPTER of that name" in msg
    assert "selects no adapter" in msg


def test_a_lens_route_on_a_native_standup_refuses_by_name():
    routes = tuple(r if r.cell_id != PROMPTED else px.RouteEntry(
        virtual_model=r.virtual_model, cell_id=r.cell_id, cell_class=r.cell_class,
        serving_identity=LENS_ID, intervention_hash=r.intervention_hash,
        si_prefix_hash=r.si_prefix_hash) for r in _routes())
    att = _attestation()
    with pytest.raises(gate.GateError) as ei:
        _hold(models=_models(att, routes), table=_table(routes), att=att)
    msg = str(ei.value)
    assert f"route {PROMPTED!r} carries serving path 'lens'" in msg
    assert "LORA and PASSTHROUGH routes only" in msg


def test_a_prompted_prefix_hash_that_disagrees_with_the_table_refuses():
    table = _table()
    table["prompted_prefixes"] = (px.PromptedPrefixSpec(
        virtual_model=f"petri__{PROMPTED}", byte_length=512, sha16="f" * 16),)
    with pytest.raises(gate.GateError) as ei:
        _hold(table=table)
    msg = str(ei.value)
    assert f"prompted route {PROMPTED!r}.si_prefix_hash {PREFIX_SHA!r}" in msg
    assert "prompted_prefixes" in msg and "f" * 16 in msg


def test_a_prompted_route_with_no_prefix_spec_at_all_refuses():
    table = _table()
    table["prompted_prefixes"] = ()
    with pytest.raises(gate.GateError, match="has no prompted_prefixes spec"):
        _hold(table=table)


def test_an_unprompted_leg_carrying_a_prefix_spec_refuses():
    routes = _routes()
    table = _table(routes)
    table["prompted_prefixes"] = table["prompted_prefixes"] + (
        px.PromptedPrefixSpec(virtual_model="petri__base", byte_length=8, sha16="1" * 16),)
    with pytest.raises(gate.GateError) as ei:
        _hold(table=table)
    assert "carries NO si_prefix_hash but the table has a prompted_prefixes spec" in str(ei.value)


def test_the_lora_branch_still_refuses_a_wrong_intervention_hash():
    """The pre-existing adapter binding is untouched by the passthrough branch."""
    att = _attestation()
    att["adapters"][CELL]["petri_intervention_hash"] = "9" * 16
    with pytest.raises(gate.GateError) as ei:
        _hold(models=_models(att), att=att)
    assert "petri_intervention_hash" in str(ei.value)


def test_an_unserved_passthrough_vmodel_is_still_reported():
    att = _attestation()
    models = _models(att)
    models["data"] = [d for d in models["data"] if d["id"] != f"petri__{PROMPTED}"]
    with pytest.raises(gate.GateError, match="does not serve"):
        _hold(models=models, att=att)
