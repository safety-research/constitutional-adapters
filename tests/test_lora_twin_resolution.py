"""The ``none__lora0`` twin resolution in the drivers.

``proxy_engine.scale0_twin`` is PATH-MATCHED: a lens route resolves
``<cell>__lens0`` then ``none__lens0`` (as before), a lora route
``<cell>__lora0`` then ``none__lora0`` (the zero adapter), never across paths,
and ``None`` for an unsteered route. ``is_steered_route`` / ``bon_served.
is_steered`` are the predicate the drivers gate their steering-bites proof on
(lens OR lora), while ``is_lens_route`` / ``_is_lens`` stay lens-only for the
declaration keys and the salt regime. Then each driver seam on a LORA route:
``served_target.served_evidence`` probes against ``none__lora0`` and refuses
without a twin; ``unified_sweep_run._served_setup``
yields ``steering_arms=(arm,)`` and a probe over the lora twin.

Everything synthetic (placeholder hashes, the demo cell set, scripted
replies); no server, no model.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from sisteer.capeval.serving import proxy as px
from sisteer.capeval.serving import proxy_engine as pe
from sisteer.capeval.serving.fidelity import FidelityError
from sisteer.capeval.serving.proxy import RouteEntry
from sisteer.capeval.tinycap import contracts as C
from sisteer.eval.attacks import bon_served as bs
from sisteer.eval.attacks import served_target as st

from test_petri_proxy import _cells, _source, _spec  # noqa: E402
from test_served_target import _StubChat  # noqa: E402

LENS = "proxy_vllm+lens+interleaved"
LORA = "proxy_vllm+lora+interleaved"
PASS = "proxy_vllm+passthrough+interleaved"


def _r(cell, ident, *, vmodel=None, cell_class="class_a", ih="cc" * 8, **kw):
    return RouteEntry(virtual_model=vmodel or f"petri__{cell}", cell_id=cell, cell_class=cell_class,
                      serving_identity=ident, intervention_hash=ih, **kw)


BASE = _r("none", PASS, vmodel="petri__base", cell_class="base", ih="aa" * 8)
LENS_TWIN = _r("none__lens0", LENS, cell_class="control", ih="aa" * 8, control_of="none")
LORA_TWIN = _r("none__lora0", LORA, cell_class="control", ih="aa" * 8, control_of="none")
LENS_CELL = _r("si_safety__psr_kl_all", LENS, ih="bb" * 8)
LORA_CELL = _r("si_safety__lora_r1", LORA, ih="cc" * 8)
OWN_LORA_TWIN = _r("si_safety__lora_r1__lora0", LORA, cell_class="control", ih="cc" * 8,
                   control_of="si_safety__lora_r1")
OWN_LENS_TWIN = _r("si_safety__psr_kl_all__lens0", LENS, cell_class="control", ih="bb" * 8,
                   control_of="si_safety__psr_kl_all")
PROMPTED = _r("prompted__safety", PASS, cell_class="class_a", ih="dd" * 8, si_prefix_hash="ab" * 8)

ALL = [BASE, LENS_TWIN, LORA_TWIN, LENS_CELL, LORA_CELL, PROMPTED]


# --------------------------------------------------------------------------- #
# the predicate and the path-matched twin                                      #
# --------------------------------------------------------------------------- #


def test_steered_predicates_and_suffixes():
    assert pe.STEERED_ROUTE_PATHS == ("lens", "lora") == px.STEERED_ROUTE_PATHS
    assert bs.STEERED_ROUTE_PATHS is px.STEERED_ROUTE_PATHS     # one literal, every gate
    assert pe.SCALE0_LORA_TWIN_SUFFIX == "__lora0" == px.LORA_ZERO_CELL_SUFFIX
    assert C.SCALE0_TWIN_SUFFIX == "__lens0"
    for route, lens, steered in ((BASE, False, False), (PROMPTED, False, False),
                                 (LENS_CELL, True, True), (LENS_TWIN, True, True),
                                 (LORA_CELL, False, True), (LORA_TWIN, False, True)):
        assert pe.is_lens_route(route) is lens, route.cell_id
        assert pe.is_steered_route(route) is steered, route.cell_id
        assert bs.is_steered(route) is steered, route.cell_id
        assert bs._is_lens(route) is lens, route.cell_id           # the salt regime stays lens-only
    assert pe.scale0_twin_suffix(LENS_CELL) == "__lens0"
    assert pe.scale0_twin_suffix(LORA_CELL) == "__lora0"
    assert pe.scale0_twin_suffix(BASE) is None and pe.scale0_twin_suffix(PROMPTED) is None
    for name in ("is_steered_route", "scale0_twin_suffix", "SCALE0_LORA_TWIN_SUFFIX",
                 "STEERED_ROUTE_PATHS"):
        assert name in pe.__all__


def test_scale0_twin_is_path_matched():
    # a lora route resolves the ZERO ADAPTER, never the lens twin beside it
    assert pe.scale0_twin(LORA_CELL, ALL) is LORA_TWIN
    # ... its OWN lora0 twin first when the table exports one
    assert pe.scale0_twin(LORA_CELL, ALL + [OWN_LORA_TWIN]) is OWN_LORA_TWIN
    # a lens route resolves exactly as before
    assert pe.scale0_twin(LENS_CELL, ALL) is LENS_TWIN
    assert pe.scale0_twin(LENS_CELL, ALL + [OWN_LENS_TWIN]) is OWN_LENS_TWIN
    # never across paths: a lora route with only lens twins, and vice versa
    assert pe.scale0_twin(LORA_CELL, [BASE, LENS_TWIN, LENS_CELL, LORA_CELL, OWN_LENS_TWIN]) is None
    assert pe.scale0_twin(LENS_CELL, [BASE, LORA_TWIN, LENS_CELL, LORA_CELL, OWN_LORA_TWIN]) is None
    # an unsteered route has no twin
    assert pe.scale0_twin(BASE, ALL) is None and pe.scale0_twin(PROMPTED, ALL) is None


# --------------------------------------------------------------------------- #
# served_target.served_evidence                                                #
# --------------------------------------------------------------------------- #


def _chat(**replies):
    return _StubChat(replies={"petri__si_safety__lora_r1": "steered text",
                              "petri__none__lora0": "plain text",
                              "petri__none__lens0": "plain text", **replies})


def test_served_evidence_probes_a_lora_route_against_the_zero_adapter():
    chat = _chat()
    ev = st.served_evidence(chat, LORA_CELL, ALL)
    assert ev["salt_isolation"]["ok"] and ev["salt_isolation"]["n_routes"] == len(ALL)
    sb = ev["steering_bites"]
    assert sb["applicable"] and sb["verified"]
    assert sb["twin"] == "none__lora0" and sb["twin_vmodel"] == "petri__none__lora0"
    assert sb["arm"] == "lora_r1"
    assert sb["n_probes"] == len(st.PROBE_PROMPTS) and sb["n_differing"] == sb["n_probes"]
    # the probe went to the lora cell and the ZERO ADAPTER only — never the lens twin
    assert {p["model"] for p in chat.payloads} == {"petri__si_safety__lora_r1", "petri__none__lora0"}
    for p in chat.payloads:
        assert p["temperature"] == 0.0 and not (set(p) & set(px.FORBIDDEN_CLIENT_KEYS))


def test_served_evidence_prefers_the_lora_routes_own_twin():
    chat = _chat(**{"petri__si_safety__lora_r1__lora0": "own twin text"})
    ev = st.served_evidence(chat, LORA_CELL, ALL + [OWN_LORA_TWIN])
    assert ev["steering_bites"]["twin"] == "si_safety__lora_r1__lora0"
    assert "petri__none__lora0" not in {p["model"] for p in chat.payloads}


def test_served_evidence_refuses_a_lora_route_with_only_a_lens_twin():
    routes = [r for r in ALL if r is not LORA_TWIN]
    with pytest.raises(st.ServedTargetError, match=r"neither its own __lora0 twin nor none__lora0"):
        st.served_evidence(_chat(), LORA_CELL, routes)


def test_served_evidence_lens_route_is_unchanged_and_never_takes_a_lora_twin():
    chat = _StubChat(replies={"petri__si_safety__psr_kl_all": "steered", "petri__none__lens0": "plain"})
    ev = st.served_evidence(chat, LENS_CELL, ALL)
    assert ev["steering_bites"]["twin"] == "none__lens0" and ev["steering_bites"]["arm"] == "psr_kl_all"
    routes = [r for r in ALL if r is not LENS_TWIN]
    with pytest.raises(st.ServedTargetError, match=r"neither its own __lens0 twin nor none__lens0"):
        st.served_evidence(chat, LENS_CELL, routes)


def test_served_evidence_inert_lora_arm_refuses():
    chat = _chat(**{"petri__si_safety__lora_r1": "plain text"})     # == the zero adapter's output
    with pytest.raises(FidelityError, match="steering_bites"):
        st.served_evidence(chat, LORA_CELL, ALL)


def test_served_evidence_unsteered_routes_are_not_applicable_and_name_both_paths():
    for route in (BASE, PROMPTED):
        ev = st.served_evidence(_StubChat(), route, ALL)
        assert ev["steering_bites"]["applicable"] is False
        assert "neither lens nor lora" in ev["steering_bites"]["reason"]


# --------------------------------------------------------------------------- #
# unified_sweep_run._served_setup                                              #
# --------------------------------------------------------------------------- #

_DRIVER = Path(__file__).resolve().parents[1] / "scripts" / "unified_sweep_run.py"


@pytest.fixture(scope="module")
def drv():
    spec = importlib.util.spec_from_file_location("unified_sweep_run", _DRIVER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _table(tmp_path):
    spec = _spec(serving_mode="interleaved")
    source = _source(tmp_path)
    cells = _cells(spec, source)
    path = tmp_path / "routing_table.json"
    px.export_routing_table(cells, source, path, base_model=spec.target_model,
                            serving_mode="interleaved")
    return path


def test_served_setup_yields_the_arm_and_a_lora_twin_probe_for_a_lora_cell(drv, tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr(drv, "_served_probe_fn",
                        lambda chat, vmodel, twin_vmodel: seen.setdefault("probe", (vmodel, twin_vmodel)))
    args = SimpleNamespace(bon_proxy_url="http://127.0.0.1:1", bon_routing_table=str(_table(tmp_path)))
    client, routes, route, arms, probe, system = drv._served_setup(
        SimpleNamespace(cell="safety_refusal__lora_r1"), args, defense=None)
    assert route.cell_id == "si_safety__lora_r1" and pe.is_steered_route(route)
    assert arms == ("lora_r1",) and probe is not None and system is None
    assert seen["probe"] == ("petri__si_safety__lora_r1", "petri__none__lora0")
    # the lens cell's setup is unchanged
    seen.clear()
    _, _, route, arms, probe, _ = drv._served_setup(
        SimpleNamespace(cell="safety_refusal__psr_kl_all"), args, defense=None)
    assert route.cell_id == "si_safety__psr_kl_all" and arms == ("psr_kl_all",)
    assert seen["probe"] == ("petri__si_safety__psr_kl_all", "petri__none__lens0")


def test_served_setup_lora_cell_without_a_lora_twin_still_names_the_arm(drv, tmp_path, monkeypatch):
    """Absent BOTH twin spellings the arm is still returned with no probe, so
    the steered-with-no-probe refusal fires downstream — never an unprobed
    serve, never a silent drop of the arm."""
    path = _table(tmp_path)
    import json
    doc = json.loads(path.read_text())
    doc["routes"] = [r for r in doc["routes"] if r["cell_id"] != "none__lora0"]
    path.write_text(json.dumps(doc))
    monkeypatch.setattr(drv, "_served_probe_fn", lambda *a: pytest.fail("no twin, no probe"))
    args = SimpleNamespace(bon_proxy_url="http://127.0.0.1:1", bon_routing_table=str(path))
    _, _, route, arms, probe, _ = drv._served_setup(
        SimpleNamespace(cell="safety_refusal__lora_r1"), args, defense=None)
    assert route.cell_id == "si_safety__lora_r1" and arms == ("lora_r1",) and probe is None
