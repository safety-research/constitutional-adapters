"""STACKED ROUTING: a cell serving a verified SI system prefix AND a
steered object simultaneously — the prompt+object combination that was
previously inexpressible (RegistryEntry carried si_prefix XOR
object_path; the proxy refused a registered prefix on every non-prompted
leg).

Covers: the ``stacked__<si>__<method>`` grammar; the two-sided
prefix/object exclusivity guard (a stacked entry REQUIRES both legs,
prompted refuses an object, plain object cells refuse a prefix); the
both-legs intervention hash (wrapper spec + verified prefix sha); proxy
routing (prefix verified per request AND selector injected — neither leg
droppable; non-stacked lens legs still refuse an unexpected prefix); the
worker-registry export/load round trip with the 'stacked' wrapper
adapter delegating to the object kind; and the two-sided identity
assertion (``fidelity.evaluate_stacked_identity``): a stack that equals
either single dropped a leg and FAILS.

CPU-only; no vLLM, no model loads, no network."""

import asyncio
import io
import json
from pathlib import Path

import pytest
import torch

from sisteer.capeval import manifest as cm
from sisteer.capeval.petri import defense_cells as dc
from sisteer.capeval.serving import fidelity as fi
from sisteer.capeval.serving import lens_adapters as la
from sisteer.capeval.serving import lens_steer as ls
from sisteer.capeval.serving import proxy as px

REV = "0123456789abcdef0123456789abcdef01234567"
IMPL_COMMIT = "fedcba9876" * 4
SAFETY_PREFIX = b"synthetic safety instruction: be safe (test bytes)"
CONSTITUTION_PREFIX = (b"synthetic constitution instruction: many "
                       b"principles (test bytes)")
BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
STACKED_A = "stacked__safety__psr_kl_all"
STACKED_B = "stacked__constitution__alphasteer"

CELLS = ["none", "prompted__safety", "prompted__constitution",
         "si_safety__psr_kl_all", "alphasteer", STACKED_A, STACKED_B]

PROMPT_FILES = {"auditor": b"# synthetic auditor prompt bytes\n",
                "judge": b"# synthetic judge prompt bytes\n"}
BLOOM_OPTIONS = dict(max_turns=40, enable_prefill=False,
                     enable_rollback=False, compaction=False,
                     realism_filter=False, epochs=1, seed=0)


def _spec(**over) -> cm.PetriSpec:
    kwargs = dict(
        audit="petri-stacked-test",
        behaviors=list(cm.PETRI_BEHAVIORS),
        cells=list(CELLS),
        scenario_bundle_hash="ab" * 8,
        auditor_model="anthropic/claude-sonnet-5",
        judge_model="anthropic/claude-opus-5",
        scenarios_model="anthropic/claude-sonnet-5",
        prompt_provenance="package-pinned@petri-bloom==0.2.6",
        target_model=BASE_MODEL,
        target_revision=REV,
        serving_backend="proxy_vllm",
        serving_mode="exact_replay",
        serving_packages={"vllm": "0.25.1+cu129", "vllm-lens": "1.2.1"},
        serving_launch_flags=["--enforce-eager"],
        chat_template_hash="12" * 8,
        tool_parser="llama3_json",
        serving_impl_commit=IMPL_COMMIT,
        prompted_prefix_hashes={
            "safety": cm.sha256_hex16(SAFETY_PREFIX),
            "constitution": cm.sha256_hex16(CONSTITUTION_PREFIX),
        },
        bloom_options=cm.BloomOptions(**BLOOM_OPTIONS),
    )
    kwargs.update(over)
    return cm.PetriSpec(**kwargs)


def _behavior_files(spec):
    return {k: f"# behavior {k}\nsynthetic eval bytes\n".encode()
            for k in cm.petri_expected_behavior_keys(spec)}


def _synthetic_artifact(tmp_path: Path, name: str, layer: int = 3
                        ) -> Path:
    """A REAL adaptable object artifact (lens_adapters SYNTHETIC_KIND):
    the export/load round trip runs the committed adapter, so the bytes
    must genuinely translate."""
    buf = io.BytesIO()
    torch.save({"layer": layer,
                "vector": torch.arange(8, dtype=torch.float32)}, buf)
    p = tmp_path / f"{name}.pt"
    p.write_bytes(buf.getvalue())
    return p


def _entries(tmp_path, **over):
    """A synthetic CellSource covering CELLS, with real adaptable bytes
    behind every object cell (kind = SYNTHETIC_KIND so the committed
    adapter genuinely translates them)."""
    kind = la.SYNTHETIC_KIND
    ent = {
        "none": dc.RegistryEntry(cell_id="none", kind="none"),
        "prompted__safety": dc.RegistryEntry(
            cell_id="prompted__safety", kind="prompted", si_id="safety",
            si_prefix=SAFETY_PREFIX),
        "prompted__constitution": dc.RegistryEntry(
            cell_id="prompted__constitution", kind="prompted",
            si_id="constitution", si_prefix=CONSTITUTION_PREFIX),
        "si_safety__psr_kl_all": dc.RegistryEntry(
            cell_id="si_safety__psr_kl_all", kind=kind, si_id="safety",
            object_path=str(_synthetic_artifact(tmp_path, "obj_a")),
            serving_path="lens",
            resolved_spec={"kind": kind, "si_id": "safety", "layer": 3}),
        "alphasteer": dc.RegistryEntry(
            cell_id="alphasteer", kind=kind, si_id=None,
            object_path=str(_synthetic_artifact(tmp_path, "obj_b")),
            serving_path="lens",
            resolved_spec={"kind": kind, "layer": 3}),
        STACKED_A: dc.RegistryEntry(
            cell_id=STACKED_A, kind=kind, si_id="safety",
            object_path=str(_synthetic_artifact(tmp_path, "stk_a")),
            serving_path="lens", si_prefix=SAFETY_PREFIX,
            resolved_spec={"kind": kind, "si_id": "safety", "layer": 3}),
        # MOVED PIN: this stack was registered serving_path="lora" to
        # exercise a LoRA-path stack. No stacked SI-plus-LoRA cell is
        # served natively, so enumerate_cells now REFUSES a stacked entry
        # marked lora on every table (test_a_stacked_entry_marked_lora_
        # refuses_on_every_table below); both stacks ride the lens.
        STACKED_B: dc.RegistryEntry(
            cell_id=STACKED_B, kind=kind, si_id="constitution",
            object_path=str(_synthetic_artifact(tmp_path, "stk_b")),
            serving_path="lens", si_prefix=CONSTITUTION_PREFIX,
            resolved_spec={"kind": kind, "layer": 3}),
    }
    ent.update(over)
    return ent


def _enumerate(tmp_path, spec=None, **over):
    spec = spec or _spec()
    return dc.enumerate_cells(
        spec, dc.MappingCellSource(_entries(tmp_path, **over)),
        behavior_files=_behavior_files(spec), prompt_files=PROMPT_FILES)


# --- the grammar ---------------------------------------------------------------


def test_stacked_ids_parse_for_class_a_and_class_b_objects():
    a = dc.parse_cell_id(STACKED_A)
    assert (a.cell_class, a.si_id, a.method) == ("stacked", "safety",
                                                 "psr_kl_all")
    assert dc.stacked_object_cell_id(a) == "si_safety__psr_kl_all"
    assert dc.stacked_prompted_cell_id(a) == "prompted__safety"
    b = dc.parse_cell_id(STACKED_B)
    assert (b.cell_class, b.si_id, b.method) == ("stacked", "constitution",
                                                 "alphasteer")
    assert dc.stacked_object_cell_id(b) == "alphasteer"
    assert dc.stacked_prompted_cell_id(b) == "prompted__constitution"
    assert "stacked" in dc.CELL_CLASSES


@pytest.mark.parametrize("bad", [
    "stacked__safety",                # no method
    "stacked__french__psr_kl_all",    # unknown SI
    "stacked__safety__not_a_method",  # unknown method
    "stacked____psr_kl_all",          # empty SI
])
def test_malformed_stacked_ids_refuse(bad):
    with pytest.raises(dc.DefenseCellError, match="stacked"):
        dc.parse_cell_id(bad)


def test_singles_helpers_refuse_non_stacked_cells():
    parsed = dc.parse_cell_id("si_safety__psr_kl_all")
    with pytest.raises(dc.DefenseCellError):
        dc.stacked_object_cell_id(parsed)
    with pytest.raises(dc.DefenseCellError):
        dc.stacked_prompted_cell_id(parsed)


def test_stacked_kind_constants_agree():
    assert la.STACKED_KIND == cm.PETRI_STACKED_KIND


# --- the both-legs intervention hash --------------------------------------------


def _wrapper(prefix=SAFETY_PREFIX, okind=None, **obj_over):
    obj = {"kind": okind or la.SYNTHETIC_KIND, "layer": 3}
    obj.update(obj_over)
    return dc.stacked_resolved_spec("safety", prefix, obj)


def test_stacked_hash_binds_both_legs():
    art, side = b"artifact-bytes", b""
    h = cm.petri_intervention_hash(_wrapper(), artifact_bytes=art,
                                   sidecar_bytes=side,
                                   si_prefix_bytes=SAFETY_PREFIX)
    assert cm._is_hex16(h)
    # a different OBJECT changes the hash
    h2 = cm.petri_intervention_hash(_wrapper(), artifact_bytes=b"other",
                                    sidecar_bytes=side,
                                    si_prefix_bytes=SAFETY_PREFIX)
    assert h2 != h
    # a different PREFIX changes the hash (via the wrapper's sha16)
    h3 = cm.petri_intervention_hash(
        _wrapper(prefix=CONSTITUTION_PREFIX), artifact_bytes=art,
        sidecar_bytes=side, si_prefix_bytes=CONSTITUTION_PREFIX)
    assert h3 != h
    # ... and differs from the object-only single's hash on the same bytes
    h4 = cm.petri_intervention_hash({"kind": la.SYNTHETIC_KIND, "layer": 3},
                                    artifact_bytes=art, sidecar_bytes=side)
    assert h4 != h
    # bytes are OPTIONAL (a worker re-verifying holds no SI text) but
    # when given they must match the recorded sha
    assert cm.petri_intervention_hash(_wrapper(), artifact_bytes=art,
                                      sidecar_bytes=side) == h
    with pytest.raises(cm.ManifestError, match="does not name the prefix"):
        cm.petri_intervention_hash(_wrapper(), artifact_bytes=art,
                                   sidecar_bytes=side,
                                   si_prefix_bytes=CONSTITUTION_PREFIX)


def test_stacked_hash_refusals():
    w = _wrapper()
    with pytest.raises(cm.ManifestError, match="artifact_bytes"):
        cm.petri_intervention_hash(w, si_prefix_bytes=SAFETY_PREFIX)
    with pytest.raises(cm.ManifestError, match="unknown key"):
        cm.petri_intervention_hash(dict(w, extra=1), artifact_bytes=b"a",
                                   sidecar_bytes=b"")
    with pytest.raises(cm.ManifestError, match="si_prefix_sha16"):
        cm.petri_intervention_hash(
            dict(w, si_prefix_sha16="nothex"), artifact_bytes=b"a",
            sidecar_bytes=b"")
    bad = dict(w, object={"kind": cm.PETRI_STACKED_KIND})
    with pytest.raises(cm.ManifestError, match="nested"):
        cm.petri_intervention_hash(bad, artifact_bytes=b"a",
                                   sidecar_bytes=b"")
    # object kinds still refuse a prefix (the pre-existing contract)
    with pytest.raises(cm.ManifestError, match="si_prefix_bytes"):
        cm.petri_intervention_hash({"kind": la.SYNTHETIC_KIND, "layer": 3},
                                   artifact_bytes=b"a", sidecar_bytes=b"",
                                   si_prefix_bytes=SAFETY_PREFIX)


def test_stacked_resolved_spec_refusals():
    with pytest.raises(dc.DefenseCellError, match="si_id"):
        dc.stacked_resolved_spec("french", SAFETY_PREFIX,
                                 {"kind": la.SYNTHETIC_KIND})
    with pytest.raises(dc.DefenseCellError, match="empty si_prefix"):
        dc.stacked_resolved_spec("safety", b"", {"kind": la.SYNTHETIC_KIND})
    with pytest.raises(dc.DefenseCellError, match="defense-object kind"):
        dc.stacked_resolved_spec("safety", SAFETY_PREFIX,
                                 {"kind": "prompted"})


# --- enumeration ----------------------------------------------------------------


def test_stacked_cell_enumerates_with_both_legs(tmp_path):
    cs = _enumerate(tmp_path)
    assert not cs.absent
    by_id = {c.cell_id: c for c in cs.canonical}
    stk = by_id[STACKED_A]
    assert stk.cell_class == "stacked"
    assert stk.si_prefix_hash == cm.sha256_hex16(SAFETY_PREFIX)
    assert stk.virtual_model_name == "petri__" + STACKED_A
    ident = cm.parse_serving_identity(stk.serving_identity)
    assert ident.path == "lens"
    # the hash covers BOTH legs: differs from the object-only single
    # (same synthetic kind, DIFFERENT bytes on disk) and from prompted
    assert stk.intervention_hash != by_id["si_safety__psr_kl_all"].intervention_hash
    assert stk.intervention_hash != by_id["prompted__safety"].intervention_hash
    # ... and equals the manual wrapper construction over its own bytes
    art = Path(dict(_entries(tmp_path))[STACKED_A].object_path)
    # NB _entries regenerates artifacts; recompute from the enumerated path
    entry = dc.MappingCellSource(_entries(tmp_path)).get(STACKED_A)
    wrapper = dc.stacked_resolved_spec("safety", SAFETY_PREFIX,
                                       entry.resolved_spec)
    manual = cm.petri_intervention_hash(
        wrapper, artifact_bytes=Path(stk.object_path).read_bytes(),
        sidecar_bytes=b"", si_prefix_bytes=SAFETY_PREFIX)
    assert stk.intervention_hash == manual
    # the Class-B stack rides the lens too (there is no LoRA-path stack)
    ident_b = cm.parse_serving_identity(by_id[STACKED_B].serving_identity)
    assert ident_b.path == "lens"


def test_a_stacked_entry_marked_lora_refuses_on_every_table(tmp_path):
    """No stacked SI-plus-LoRA cell is served natively. A stacked entry
    registered serving_path='lora' is a LOUD enumeration refusal naming
    the cell and the rule — on a lens table (default) and on a lora-native
    table (lens_controls=False) alike — so the proxy can never be handed
    a 'prefix + native adapter' route."""
    ent = _entries(tmp_path)
    ent[STACKED_B] = dc.RegistryEntry(
        cell_id=STACKED_B, kind=la.SYNTHETIC_KIND, si_id="constitution",
        object_path=ent[STACKED_B].object_path, serving_path="lora",
        si_prefix=CONSTITUTION_PREFIX, resolved_spec={"kind": la.SYNTHETIC_KIND,
                                                      "layer": 3})
    # the stack is the ONLY object route: on a lora-native table
    # (lens_controls=False) the fixture's lens routes would refuse first
    #, which is not the ruling under test here
    spec = _spec(cells=["none", "prompted__constitution", STACKED_B])
    for lens_controls in (True, False):
        with pytest.raises(dc.DefenseCellError, match="D6") as exc:
            dc.enumerate_cells(
                spec, dc.MappingCellSource(ent),
                behavior_files=_behavior_files(spec), prompt_files=PROMPT_FILES,
                lens_controls=lens_controls)
        assert STACKED_B in str(exc.value) and "STACKED" in str(exc.value)
        assert "serving_path='lora'" in str(exc.value)


def test_stacked_without_prefix_refuses(tmp_path):
    ent = _entries(tmp_path)
    ent[STACKED_A] = dc.RegistryEntry(
        cell_id=STACKED_A, kind=la.SYNTHETIC_KIND, si_id="safety",
        object_path=ent[STACKED_A].object_path, serving_path="lens",
        resolved_spec=ent[STACKED_A].resolved_spec)  # si_prefix dropped
    with pytest.raises(dc.DefenseCellError, match="no si_prefix"):
        _enumerate(tmp_path, **ent)


def test_stacked_with_wrong_prefix_refuses(tmp_path):
    ent = _entries(tmp_path)
    ent[STACKED_A] = dc.RegistryEntry(
        cell_id=STACKED_A, kind=la.SYNTHETIC_KIND, si_id="safety",
        object_path=ent[STACKED_A].object_path, serving_path="lens",
        si_prefix=b"tampered prefix bytes",
        resolved_spec=ent[STACKED_A].resolved_spec)
    with pytest.raises(dc.DefenseCellError, match="masquerade"):
        _enumerate(tmp_path, **ent)


def test_prefix_on_plain_object_cell_refuses(tmp_path):
    ent = _entries(tmp_path)
    ent["alphasteer"] = dc.RegistryEntry(
        cell_id="alphasteer", kind=la.SYNTHETIC_KIND, si_id=None,
        object_path=ent["alphasteer"].object_path, serving_path="lens",
        si_prefix=SAFETY_PREFIX,  # smuggled onto a non-stacked cell
        resolved_spec=ent["alphasteer"].resolved_spec)
    with pytest.raises(dc.DefenseCellError, match="NON-stacked"):
        _enumerate(tmp_path, **ent)


def test_object_on_prompted_cell_refuses(tmp_path):
    ent = _entries(tmp_path)
    ent["prompted__safety"] = dc.RegistryEntry(
        cell_id="prompted__safety", kind="prompted", si_id="safety",
        si_prefix=SAFETY_PREFIX,
        object_path=ent["alphasteer"].object_path)  # smuggled object
    with pytest.raises(dc.DefenseCellError, match="PROMPTED entry carries"):
        _enumerate(tmp_path, **ent)


def test_stacked_missing_artifact_is_absent(tmp_path):
    ent = _entries(tmp_path)
    ent[STACKED_A] = dc.RegistryEntry(
        cell_id=STACKED_A, kind=la.SYNTHETIC_KIND, si_id="safety",
        object_path=str(tmp_path / "nope.pt"), serving_path="lens",
        si_prefix=SAFETY_PREFIX,
        resolved_spec=ent[STACKED_A].resolved_spec)
    cs = _enumerate(tmp_path, **ent)
    assert any(a.cell_id == STACKED_A and "missing" in a.reason
               for a in cs.absent)


def test_stacked_cell_needs_recorded_prefix_hash():
    spec = _spec(prompted_prefix_hashes={
        "safety": cm.sha256_hex16(SAFETY_PREFIX)},
        cells=["none", STACKED_B])
    with pytest.raises(dc.DefenseCellError, match="prompted_prefix_hashes"):
        dc.validate_petri_spec_for_enumeration(spec)


# --- proxy routing: neither leg droppable ----------------------------------------


class _StubBackend:
    def __init__(self):
        self.payloads = []

    async def take_request_failure(self, request_id):
        # the lens post-check surface (per-request containment); this
        # stub never poisons a request.
        return None

    async def chat_completions(self, payload):
        self.payloads.append(dict(payload))
        return {"id": "stub", "model": payload.get("model"),
                "choices": [{"index": 0, "message":
                             {"role": "assistant", "content": "ok"}}]}


def _proxy(tmp_path):
    spec = _spec(serving_mode="interleaved")
    cs = dc.enumerate_cells(
        spec, dc.MappingCellSource(_entries(tmp_path)),
        behavior_files=_behavior_files(spec), prompt_files=PROMPT_FILES)
    source = dc.MappingCellSource(_entries(tmp_path))
    backend = _StubBackend()
    proxy = px.RoutingProxy(
        px.routes_from_cells(cs), backend, base_model=BASE_MODEL,
        serving_mode="interleaved",
        prompted_prefixes=px.prompted_prefix_specs(cs, source))
    by_id = {c.cell_id: c for c in cs.canonical}
    return proxy, backend, by_id


def _chat(proxy, vmodel, messages):
    return asyncio.run(proxy.handle_chat_completions(
        {"model": vmodel, "messages": messages, "max_tokens": 8}))


def test_stacked_route_serves_both_legs(tmp_path):
    proxy, backend, by_id = _proxy(tmp_path)
    messages = [
        {"role": "system", "content": SAFETY_PREFIX.decode() + " extra"},
        {"role": "user", "content": "hello"}]
    out = _chat(proxy, "petri__" + STACKED_A, messages)
    assert out["petri_intervention_hash"] == by_id[STACKED_A].intervention_hash
    up = backend.payloads[-1]
    # object leg: the injected selector, bound to the STACKED hash
    assert up[px.UPSTREAM_EXTRA_ARGS_KEY] == {
        "petri_object": STACKED_A,
        "expect_hash": by_id[STACKED_A].intervention_hash}
    # prompt leg: the verified prefix travels upstream untouched
    assert up["messages"][0]["content"].startswith(SAFETY_PREFIX.decode())
    assert up["model"] == BASE_MODEL


# (test_stacked_lora_route_serves_both_legs lived here: it routed STACKED_B
# as a LoRA-path stack — `upstream.model == STACKED_B`, no selector, prefix
# forwarded. enumerate_cells now refuses a stacked entry marked lora, so
# that route can no longer be built through the harness; its replacement is
# test_a_stacked_entry_marked_lora_refuses_on_every_table above. The proxy's
# own lora-route rewrite is unchanged and is pinned by tests/test_petri_proxy.py.)


def test_stacked_route_refuses_a_dropped_prompt_leg(tmp_path):
    proxy, backend, _ = _proxy(tmp_path)
    with pytest.raises(px.ProxyRefusal, match="SI prefix") as e1:
        _chat(proxy, "petri__" + STACKED_A,
              [{"role": "user", "content": "no system prefix at all"}])
    assert e1.value.code == "si_prefix_missing"
    with pytest.raises(px.ProxyRefusal, match="prefix") as e2:
        _chat(proxy, "petri__" + STACKED_A,
              [{"role": "system", "content": "wrong prefix bytes"},
               {"role": "user", "content": "hello"}])
    assert e2.value.code in ("si_prefix_missing", "si_prefix_mismatch")
    assert not backend.payloads  # nothing one-legged ever went upstream


def test_non_stacked_lens_leg_still_refuses_a_registered_prefix(tmp_path):
    proxy, backend, _ = _proxy(tmp_path)
    with pytest.raises(px.ProxyRefusal,
                       match="registered SI prefix") as exc:
        _chat(proxy, "petri__si_safety__psr_kl_all",
              [{"role": "system", "content": SAFETY_PREFIX.decode()},
               {"role": "user", "content": "hello"}])
    assert exc.value.code == "si_prefix_unexpected"
    assert not backend.payloads


# --- worker-registry export/load round trip ---------------------------------------


def test_stacked_export_load_roundtrip_delegates_to_object_ops(tmp_path):
    spec = _spec()
    cs = dc.enumerate_cells(
        spec, dc.MappingCellSource(_entries(tmp_path)),
        behavior_files=_behavior_files(spec), prompt_files=PROMPT_FILES)
    source = dc.MappingCellSource(_entries(tmp_path))
    out = tmp_path / "registry"
    ls.export_worker_registry(cs, source, out,
                              assistant_header_ids=(11, 22))
    reg = ls.WorkerObjectRegistry.load(out)
    stk_id = STACKED_A  # either stack would do: both ride the lens
    assert stk_id in reg.cell_ids()
    by_id = {c.cell_id: c for c in cs.canonical}
    obj = reg.select(stk_id, by_id[stk_id].intervention_hash)
    assert obj.kind == cm.PETRI_STACKED_KIND
    # the wrapper adapter yields the OBJECT leg's exact ops
    single = reg.select("si_safety__psr_kl_all",
                        by_id["si_safety__psr_kl_all"].intervention_hash)
    assert len(obj.ops) == len(single.ops) == 1
    assert obj.ops[0].op == single.ops[0].op
    assert obj.ops[0].layer == single.ops[0].layer
    # the registry index carries the wrapper spec, NEVER the SI text
    index = json.loads((out / "registry.json").read_text())
    raw = next(e for e in index["entries"] if e["cell_id"] == stk_id)
    assert raw["kind"] == cm.PETRI_STACKED_KIND
    assert raw["resolved_spec"]["si_prefix_sha16"] == cm.sha256_hex16(
        SAFETY_PREFIX)
    assert SAFETY_PREFIX.decode() not in json.dumps(index)


def test_stacked_adapter_refusals():
    art = io.BytesIO()
    torch.save({"layer": 1, "vector": torch.ones(4)}, art)
    good = {"kind": "stacked", "si_id": "safety",
            "si_prefix_sha16": cm.sha256_hex16(SAFETY_PREFIX),
            "object": {"kind": la.SYNTHETIC_KIND, "layer": 1}}
    ops = la.get_adapter("stacked")(good, art.getvalue(), b"")
    assert len(ops) == 1 and ops[0].layer == 1
    with pytest.raises(la.AdapterError, match="unknown key"):
        la.get_adapter("stacked")(dict(good, extra=1), art.getvalue(), b"")
    with pytest.raises(la.AdapterError, match="si_prefix_sha16"):
        la.get_adapter("stacked")(dict(good, si_prefix_sha16="zz"),
                                  art.getvalue(), b"")
    with pytest.raises(la.AdapterError, match="nested"):
        la.get_adapter("stacked")(
            dict(good, object={"kind": "stacked"}), art.getvalue(), b"")
    with pytest.raises(la.AdapterError, match="object"):
        la.get_adapter("stacked")(dict(good, object={}), art.getvalue(), b"")


# --- the two-sided identity assertion ----------------------------------------------


def _stacked_evidence(stacked, object_only, prompt_only):
    return {"cells": {STACKED_A: {"stacked_ids": stacked,
                                  "object_only_ids": object_only,
                                  "prompt_only_ids": prompt_only}}}


def test_stacked_identity_passes_when_both_sides_differ():
    diag = fi.evaluate_stacked_identity(_stacked_evidence(
        [[1, 2, 3], [4, 5]], [[1, 2, 9], [4, 5]], [[1, 2, 3], [4, 9]]))
    d = diag[STACKED_A]
    assert d["object_only"]["n_differing"] == 1
    assert d["prompt_only"]["n_differing"] == 1


def test_stacked_identity_fails_when_object_leg_dropped():
    # stack == object-only on EVERY probe: the prompt leg contributed
    # nothing (the one-leg failure that invalidated the earlier attempt)
    with pytest.raises(fi.FidelityError, match="object_only"):
        fi.evaluate_stacked_identity(_stacked_evidence(
            [[1, 2], [3, 4]], [[1, 2], [3, 4]], [[9, 9], [8, 8]]))


def test_stacked_identity_fails_when_prompt_leg_dropped():
    with pytest.raises(fi.FidelityError, match="prompt_only"):
        fi.evaluate_stacked_identity(_stacked_evidence(
            [[1, 2], [3, 4]], [[9, 9], [8, 8]], [[1, 2], [3, 4]]))


def test_stacked_identity_vacuous_evidence_fails():
    with pytest.raises(fi.FidelityError, match="zero stacked cells"):
        fi.evaluate_stacked_identity({"cells": {}})
    with pytest.raises(fi.FidelityError, match="empty"):
        fi.evaluate_stacked_identity(_stacked_evidence([], [], []))
    with pytest.raises(fi.FidelityError, match="no tokens on either side"):
        fi.evaluate_stacked_identity(_stacked_evidence([[]], [[]], [[1]]))
    with pytest.raises(fi.FidelityError, match="lacks"):
        fi.evaluate_stacked_identity({"cells": {STACKED_A: {
            "stacked_ids": [[1]], "object_only_ids": [[2]]}}})
    with pytest.raises(fi.FidelityError, match="non-integer"):
        fi.evaluate_stacked_identity(_stacked_evidence(
            [["text"]], [[1]], [[2]]))
