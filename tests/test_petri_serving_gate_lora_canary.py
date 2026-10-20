"""The serving gate's LORA-NATIVE canary, CPU-only.

Three layers, each pinned here:

1. ``run_gate`` END TO END under ``engine="lora_native"`` with fakes shaped
   exactly like the certification — ONE genuine adapter + the zero adapter,
   ``noop_serving={"passthrough", "lora"}``, ``phase0_evidence={}``, a
   ``MultiLoraProbe`` with ``zero_adapter`` (the NATIVE selection policy) —
   passes, records Phase 0 not-applicable, certificates ``{passthrough,
   lora}`` and ``multi_lora_checked``; every deviation is a ``GateError``
   (a lens key or lens path, zero != base, genuine == base, no probe,
   Phase-0 evidence on a native run); the IDENTICAL inputs under
   ``engine="lens"`` refuse (lens leg missing) — the legacy mode is unchanged
   beside ``test_petri_serving_gate.py``'s untouched no-op-legs test.
2. THE ORACLE LIFECYCLE AND JOIN CONTRACT, producer + consumer TOGETHER on
   the REAL emitted schemas (no hand-written fixtures for the three
   documents): the real ``gate-worker-reference --objects-json`` (its fp32
   shim stubbed) writes ``reference.json`` for the TrainTest cell
   ``constv2ctl2-full__lora_r4_ce`` keyed by ``harness_cell_id`` ==
   ``si_constv2ctl2_full__lora_r4_ce`` — an id that CHANGES under
   translation; the real ``bring_up_lora`` (test_lora_serving_up's fake
   lora-native upstream) writes ``routing_table.json`` / ``cells_served.json``
   / ``endpoint.json``; ``_bootstrap_lora_native_stack`` binds the two,
   dispatches ``kind_serving`` to the GENUINE route only and both no-op legs
   to the BASE oracle, and the whole ``gate`` subcommand then completes
   against the live fake standup. Every negative of the contract is a
   ``GateBootstrapError`` naming the field.
3. The producer refuses a digest mismatch BEFORE loading anything; the CLI
   refuses ``--registry-dir`` beside ``--lora-native-standup`` and a standup
   without its ``--reference`` at argparse.

Skips (never fetches) when the pinned gemma tokenizer/config is not in the
local HF cache (the standup fixture needs it)."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[1]
for p in (REPO / "src", REPO / "scripts", REPO / "scripts" / "sdfbase", REPO / "tests"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import petri_serving_gate as gate  # noqa: E402
from test_lora_serving_up import (BASE_IDS, GEMMA_CELL, GEMMA_HCELL, GENUINE_IDS,  # noqa: E402
                                  ZERO, _campaign, _load_lsu, bring_up_fake_standup,
                                  cached_or_skip)
from test_petri_serving_gate import _gate_inputs, _good_evidence, _spec  # noqa: E402

from sisteer.capeval.serving import serving_attestation as sa  # noqa: E402
from sisteer.capeval import manifest as cm  # noqa: E402
from sisteer.capeval.petri import registry_adapter as ra  # noqa: E402

CELL = "si_x__lora_all_r4"
V_BASE, V_ZERO, V_CELL = "petri__base", f"petri__{ZERO}", f"petri__{CELL}"
KEYS = [gate.ReferenceOracle.key(p) for p in gate.GATE_PROMPT_SET]


# --------------------------------------------------------------------------- #
# 1. run_gate under engine="lora_native" (pure; fakes shaped like the cert)     #
# --------------------------------------------------------------------------- #


def _served(oracle, ids, reference_cell):
    def fn(messages):
        oracle.set_current(reference_cell)
        return list(ids)
    return fn


SEL_ADAPTER = {V_BASE: None, V_ZERO: ZERO, V_CELL: CELL}
ID_OF = {None: 0, ZERO: 1, CELL: 2}


def _routing(table, *, misroute=None, serial=False):
    """A fake worker routing recorder for the pure run_gate tests: epochs,
    the engine's ``-<8 hex>`` renaming of every request id, one step per
    batch (mixed when the batch mixes selections; ``serial`` = one request per
    step), and the engine binding each request to its selection's adapter
    unless ``misroute`` ({vmodel: adapter name | None}) says otherwise."""
    import hashlib
    bound, steps, log = {}, [], []
    state = {"epoch": None}

    def mixed_batch(batch):
        out, eids = [], []
        for name, _p in batch:
            ext = f"chatcmpl-{len(bound) + 1:04d}"
            eid = f"{ext}-{hashlib.sha256(ext.encode()).hexdigest()[:8]}"
            bound[eid] = (misroute or {}).get(name, SEL_ADAPTER[name])
            eids.append(eid)
            out.append((list(table[name]), ext))
        steps.extend([[e] for e in eids] if serial else [sorted(eids)])
        return out

    def arm(epoch):
        log.append(("arm", epoch))
        state["epoch"] = epoch                                       # ADOPT the control's epoch
        bound.clear()
        del steps[:]
        return {"armed": True, "epoch": epoch, "already_armed": False, "runner_kind": "v2"}

    def observe(epoch):
        log.append(("observe", epoch))
        assert epoch == state["epoch"]
        return {"armed": True, "epoch": epoch, "runner_kind": "v2",
                "runner_class": "vllm.v1.worker.gpu.model_runner.GPUModelRunner",
                "observations": {e: {"lora_int_id": ID_OF[b], "lora_name": b} for e, b in bound.items()},
                "inconsistent": {}, "steps": [list(s) for s in steps], "n_steps": len(steps),
                "overflow": False, "steps_overflow": False, "errors": []}

    def disarm(epoch):
        log.append(("disarm", epoch))
        assert epoch == state["epoch"]
        return {"disarmed": True, "epoch": epoch}
    access = gate.LoraRoutingRecorderAccess(
        arm=arm, observe=observe, disarm=disarm, mixed_batch=mixed_batch,
        selection_adapter=SEL_ADAPTER, id_name_map={"1": ZERO, "2": CELL})
    return access, log


def _native_inputs(*, engine="lora_native", solo_table=None, noop_lora_ids=(1, 1),
                   genuine_ids=(2, 2), kind_paths=("lora",), noop_keys=("passthrough", "lora"),
                   phase0_evidence=None, with_probe=True, routing=None, with_routing=True):
    oracle = gate.ReferenceOracle(base={k: [1, 1] for k in KEYS},
                                  per_cell={CELL: {k: list(genuine_ids) for k in KEYS}})
    table = {V_BASE: [1, 1], V_ZERO: list(noop_lora_ids), V_CELL: list(genuine_ids)}
    table.update(solo_table or {})

    def solo(name, messages):
        return list(table[name])

    def mixed(batch):
        return [list(table[n]) for n, _ in batch]

    if routing is None and with_routing:
        routing, _ = _routing(table)
    probe = gate.MultiLoraProbe(base=V_BASE, adapters=(V_CELL, V_ZERO), mixed_batch=mixed,
                                solo=solo, zero_adapter=V_ZERO, routing=routing) if with_probe else None
    legs = {"passthrough": _served(oracle, [1, 1], None),
            "lora": _served(oracle, noop_lora_ids, None),
            "lens": _served(oracle, [1, 1], None)}
    return gate.GateInputs(
        phase0_evidence=({} if phase0_evidence is None else phase0_evidence),
        reference=oracle,
        kind_serving={CELL: {p: _served(oracle, genuine_ids, CELL) for p in kind_paths}},
        noop_serving={k: legs[k] for k in noop_keys},
        multi_lora=probe, engine=engine)


def test_run_gate_certifies_one_genuine_adapter_plus_the_zero_adapter_natively(tmp_path):
    spec = _spec()
    store = gate.CertificateStore(tmp_path / "certs.json")
    result = gate.run_gate(spec, store, _native_inputs())
    assert result.phase0 == gate.PHASE0_NOT_APPLICABLE
    assert result.multi_lora_checked is True
    assert {c.path for c in result.certificates} == {"passthrough", "lora"}
    assert {c.path for c in store.certificates()} == {"passthrough", "lora"}   # published
    assert result.kind_outcomes[CELL].passed_paths == ("lora",)
    assert result.fidelity is None and result.stacked_identity is None
    # the structural selection record rides in the result and the gate record
    assert result.multi_lora["verdict"] == "routing_attested" and result.multi_lora["passed"] is True
    assert result.multi_lora["n_requests"] == 3 * len(KEYS) * 7
    assert result.multi_lora["numerics"]["gating"] is False
    record = json.loads(gate.write_gate_record(tmp_path, result).read_text())
    assert record["phase0"] == gate.PHASE0_NOT_APPLICABLE
    assert record["phase0_passed"] is None and record["tp_note"] is None
    assert record["kinds"] == {CELL: {"passed_paths": ["lora"], "failed_paths": []}}
    assert record["multi_lora_checked"] is True and record["status"] == "completed"
    assert record["multi_lora"]["verdict"] == "routing_attested"
    assert gate.MultiLoraProbe.__dataclass_fields__["routing"].default is None
    assert gate.GateRunResult.__dataclass_fields__["multi_lora"].default is None
    assert {c["path"] for c in record["certificates"]} == {"passthrough", "lora"}
    assert gate.GATE_ENGINES == ("lens", "lora_native")
    assert gate.MultiLoraProbe.__dataclass_fields__["zero_adapter"].default is None
    assert gate.GateInputs.__dataclass_fields__["engine"].default == "lens"


def test_run_gate_native_refuses_any_lens_leg(tmp_path):
    spec = _spec()
    with pytest.raises(gate.GateError, match="no lens hook"):
        gate.run_gate(spec, gate.CertificateStore(tmp_path / "a.json"),
                      _native_inputs(noop_keys=("passthrough", "lora", "lens")))
    with pytest.raises(gate.GateError, match="no lens hook"):
        gate.run_gate(spec, gate.CertificateStore(tmp_path / "b.json"),
                      _native_inputs(kind_paths=("lens", "lora")))
    # nothing was published on either refusal
    assert gate.CertificateStore(tmp_path / "a.json").certificates() == []


def test_run_gate_native_records_a_zero_adapter_whose_batch_output_differs_from_base(tmp_path):
    """This once REFUSED ("contradicts the no-op certificate"). Gemma canary
    job 489308 showed exactly this — the zero adapter's output in a 33-row
    concurrent batch differing from base's on one prompt while the solo no-op
    certificate held bit for bit — so the native selection verdict is now the
    engine's bookkeeping and the output disagreement is a recorded number."""
    spec = _spec()
    result = gate.run_gate(spec, gate.CertificateStore(tmp_path / "c.json"),
                           _native_inputs(solo_table={V_ZERO: [3, 3]}))
    assert result.multi_lora["verdict"] == "routing_attested"
    assert result.multi_lora["numerics"]["zero_equals_base"] is False
    assert result.multi_lora["numerics"]["whole_layout"]["zero_vs_base"]["n_divergent_rows"] > 0
    # ... a served zero leg that is NOT a no-op still fails the no-op certificate first
    with pytest.raises(gate.GateError, match="no-op certificate FAILED for path 'lora'"):
        gate.run_gate(spec, gate.CertificateStore(tmp_path / "d.json"),
                      _native_inputs(noop_lora_ids=(3, 3)))


def test_run_gate_native_records_a_genuine_adapter_indistinguishable_from_base(tmp_path):
    """This once REFUSED as indistinguishable routing; it is recorded now
    (job 489308) — an inert adapter is the per-kind canary's and
    steering_bites' business, and a server ignoring the model field is
    caught structurally (next test)."""
    spec = _spec()
    result = gate.run_gate(spec, gate.CertificateStore(tmp_path / "e.json"),
                           _native_inputs(solo_table={V_CELL: [1, 1]}))
    assert result.multi_lora["numerics"]["whole_layout"]["genuine_like_base"] == [V_CELL]
    assert result.multi_lora["numerics"]["positional_distinguishability"]["n_undistinguishable"] > 0


def test_run_gate_native_refuses_a_structural_misroute_and_a_probe_without_recorder_access(tmp_path):
    """The verdict through run_gate: the engine binding the zero requests to
    NO LoRA (the all-base shape for that selection) refuses naming the
    selection; a recorder that saw no mixed step refuses as vacuous; a native
    probe without recorder access refuses outright."""
    spec = _spec()
    table = {V_BASE: [1, 1], V_ZERO: [1, 1], V_CELL: [2, 2]}
    routing, log = _routing(table, misroute={V_ZERO: None})
    with pytest.raises(gate.GateError, match=rf"request\(s\) for '{V_ZERO}' ran under \['<no LoRA>'\] instead of '{ZERO}'"):
        gate.run_gate(spec, gate.CertificateStore(tmp_path / "f.json"), _native_inputs(routing=routing))
    assert log[0][0] == "arm" and log[-1][0] == "disarm" and log[-1][1] == log[0][1]   # its own epoch
    routing, _ = _routing(table, misroute={V_CELL: None, V_ZERO: None})            # every request base
    with pytest.raises(gate.GateError, match=rf"request\(s\) for '{V_CELL}' ran under \['<no LoRA>'\] instead of '{CELL}'"):
        gate.run_gate(spec, gate.CertificateStore(tmp_path / "g.json"), _native_inputs(routing=routing))
    # a fully SERIAL control (the regression through the gate): no step of its own
    # schedules two selections, whatever an earlier control's steps looked like
    routing, _ = _routing(table, serial=True)
    with pytest.raises(gate.GateError, match="no engine step scheduled >= 2 distinct selections of THIS control"):
        gate.run_gate(spec, gate.CertificateStore(tmp_path / "h.json"), _native_inputs(routing=routing))
    with pytest.raises(gate.GateError, match="needs the worker's LoRA routing recorder"):
        gate.run_gate(spec, gate.CertificateStore(tmp_path / "i.json"), _native_inputs(with_routing=False))


def test_run_gate_native_requires_the_probe_and_empty_phase0_evidence(tmp_path):
    spec = _spec()
    with pytest.raises(gate.GateError, match="REQUIRES a multi_lora probe"):
        gate.run_gate(spec, gate.CertificateStore(tmp_path / "f.json"),
                      _native_inputs(with_probe=False))
    with pytest.raises(gate.GateError, match="phase0_evidence must be \\{\\}"):
        gate.run_gate(spec, gate.CertificateStore(tmp_path / "g.json"),
                      _native_inputs(phase0_evidence={"server_provenance": {}}))
    with pytest.raises(gate.GateError, match="not one of"):
        gate.run_gate(spec, gate.CertificateStore(tmp_path / "h.json"),
                      _native_inputs(engine="cuda_graphs"))


def test_the_identical_inputs_under_engine_lens_refuse_the_missing_lens_leg(tmp_path):
    """The legacy mode is unchanged: passthrough + lens mandatory (H5), lora when
    a probe is live — and the native shape has no lens leg to offer."""
    from sisteer.capeval.serving import lens_steer as ls
    spec = _spec()
    # the legacy ORDER is unchanged too: Phase 0 runs first under engine="lens", so the
    # native shape's empty evidence is a Phase0Failure before any leg is looked at ...
    with pytest.raises(ls.Phase0Failure, match="8/8 Phase-0 assertions FAILED"):
        gate.run_gate(spec, gate.CertificateStore(tmp_path / "i0.json"),
                      _native_inputs(engine="lens"))
    # ... and with good Phase-0 evidence the SAME legs refuse: the lens no-op leg is mandatory
    with pytest.raises(gate.GateError, match="no served no-op leg for path\\(s\\) \\['lens'\\]"):
        gate.run_gate(spec, gate.CertificateStore(tmp_path / "i.json"),
                      _native_inputs(engine="lens", phase0_evidence=_good_evidence()))
    # a lens run still writes the lens-shaped record (phase0 ran; the list of passed assertions)
    store = gate.CertificateStore(tmp_path / "lens.json")
    result = gate.run_gate(spec, store, _gate_inputs())
    record = json.loads(gate.write_gate_record(tmp_path / "lens", result).read_text())
    assert record["phase0"] == "ran" and isinstance(record["phase0_passed"], list)
    assert record["tp_note"] == "TP=1 only"


# --------------------------------------------------------------------------- #
# 2. the oracle lifecycle + the join contract, producer + consumer together     #
# --------------------------------------------------------------------------- #


class _StubShim:
    """``OpenAIShimEngine`` with the fp32 model REPLACED: the tokenizer is the
    real pinned one (the render-agreement check needs the real render), the
    generations are scripted per (base / armed), and every load and arm is
    recorded so a test can assert what never happened."""

    loads: list = []
    armed: list = []
    tokenizer = None
    base_ids = list(BASE_IDS)
    arm_ids = list(GENUINE_IDS)

    def __init__(self, model_id, *, revision, identity, hf_engine=None):
        self.model_id, self.revision, self.identity = model_id, revision, identity

    def load(self):
        type(self).loads.append((self.model_id, self.revision))

    @property
    def engine(self):
        return SimpleNamespace(tokenizer=type(self).tokenizer)

    def arm_for_object(self, object_path, *, scale, name):
        type(self).armed.append((str(object_path), scale, name))
        return SimpleNamespace(ctx_entries=[{"name": name}], name=name)

    def reference_generate(self, messages, *, max_tokens, arm=None):
        return list(type(self).arm_ids if arm is not None else type(self).base_ids)


def _install_stub_shim(monkeypatch, tok):
    import sisteer.capeval.serving.openai_shim as shim_mod
    _StubShim.loads, _StubShim.armed, _StubShim.tokenizer = [], [], tok
    monkeypatch.setattr(shim_mod, "OpenAIShimEngine", _StubShim)


def _produce_reference(monkeypatch, tmp_path, *, manifest, tok, objects, shas, out_name="reference.json"):
    """The REAL ``gate-worker-reference --objects-json`` (stubbed shim): rc + the record path."""
    _install_stub_shim(monkeypatch, tok)
    tmp_path = Path(tmp_path)
    tmp_path.mkdir(parents=True, exist_ok=True)
    objects_json = tmp_path / "objects.json"
    objects_json.write_text(json.dumps({c: str(p) for c, p in objects.items()}))
    ref_path = tmp_path / out_name
    argv = ["gate-worker-reference", "--manifest", str(manifest), "--out", str(ref_path),
            "--objects-json", str(objects_json)]
    for cell, sha in shas.items():
        argv += ["--expect-object-sha256", f"{cell}={sha}"]
    rc = gate.main(argv)
    return rc, ref_path


def _prompt_ids_by_key(tok):
    from petri_steered_smoke import render_ids
    return {cm.canonical_json(list(p)): render_ids(tok, list(p), True) for p in gate.GATE_PROMPT_SET}


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    """ONE fake lora-native standup + ONE reference.json for the module: the
    standup (real bring_up_lora over the fake upstream) is expensive, the
    negatives only read its records."""
    lsu = _load_lsu()
    cfg_path, man_path, cfg = _campaign("gemma4e4b")
    tok = cached_or_skip(cfg)
    mp = pytest.MonkeyPatch()
    tmp = tmp_path_factory.mktemp("canary-world")
    su, cap = bring_up_fake_standup(
        lsu, tmp, mp, fake_kwargs={"prompt_ids_by_key": _prompt_ids_by_key(tok)})
    sha = cap["shas"][GEMMA_CELL]
    object_pt = Path(cap["out"]).parent / "cells_view" / "main" / GEMMA_CELL / "object.pt"
    rc, ref_path = _produce_reference(
        mp, tmp / "oracle", manifest=man_path, tok=tok,
        objects={GEMMA_HCELL: object_pt}, shas={GEMMA_HCELL: sha})
    assert rc == 0, json.loads(ref_path.read_text()).get("error")
    yield SimpleNamespace(su=su, out=Path(cap["out"]), ref_path=ref_path, manifest=man_path,
                          sha=sha, object_pt=object_pt, tok=tok, tmp=tmp, cfg=cfg, mp=mp,
                          loads=list(_StubShim.loads), armed=list(_StubShim.armed))
    su.tear_down()
    mp.undo()


def _args(out_dir, standup_dir, ref_path, **over):
    ns = dict(lora_native_standup=str(standup_dir), reference=str(ref_path), out_dir=str(out_dir),
              child_timeout_s=60.0, neartie_margin_nats=None, registry_dir=None)
    ns.update(over)
    return argparse.Namespace(**ns)


def test_the_producer_writes_a_harness_keyed_digest_bound_reference(world):
    ref = json.loads(world.ref_path.read_text())
    assert ref["stage"] == "completed" and ref["cell_id_space"] == "harness"
    # keyed by the HARNESS id, which CHANGES under translation from the TrainTest id
    assert ra.harness_cell_id(GEMMA_CELL) == GEMMA_HCELL != GEMMA_CELL
    assert set(ref["objects"]) == set(ref["arms"]) == {GEMMA_HCELL} == set(ref["cells"])
    assert ref["objects"][GEMMA_HCELL] == {"object_pt": str(world.object_pt), "sha256": world.sha}
    assert set(ref["base"]) == set(KEYS) and set(ref["arms"][GEMMA_HCELL]) == set(KEYS)
    assert all(v == list(BASE_IDS) for v in ref["base"].values())
    assert all(v == list(GENUINE_IDS) for v in ref["arms"][GEMMA_HCELL].values())
    assert ref["model"] == world.cfg["model"] and ref["revision"] == world.cfg["model_revision"]
    ep = json.loads((world.out / "endpoint.json").read_text())
    assert ref["chat_template_sha256"] == ep["chat_template_sha256"]
    assert ref["chat_template_hash"] == ep["chat_template_hash"] == \
        cm.load_petri_spec(world.manifest).chat_template_hash      # the manifest's 16-hex pin
    assert ref["gate_prompt_set_hash"] == gate.gate_prompt_set_hash()
    assert ref["gate_max_tokens"] == gate.GATE_MAX_TOKENS
    assert len(ref["prompt_ids"]) == len(gate.GATE_PROMPT_SET)
    assert ref["prompt_ids"] == list(_prompt_ids_by_key(world.tok).values())
    for key in ("tree_sha", "torch_version", "transformers_version", "produced_at"):
        assert key in ref, key
    # the shim loaded ONCE, armed the bank's object.pt directly (load_object dispatches on magic)
    assert world.loads == [(world.cfg["model"], world.cfg["model_revision"])]
    assert world.armed == [(str(world.object_pt), 1.0, GEMMA_HCELL)]


def test_the_bootstrap_sends_the_one_proxy_bearer_rule(world, tmp_path, monkeypatch):
    """Item 10 (code review 2026-09-11): E's endpoint records ``proxy_token:
    None`` and the bootstrap then sent NO Authorization header while the
    certification always sent ``ProxyClientEngine``'s default to the same
    proxy. Both follow ``proxy.proxy_client_headers`` now: every request the
    bootstrap makes to the live standup carries the rule's bearer."""
    import urllib.request

    from sisteer.capeval.serving import proxy as px

    endpoint = json.loads((world.out / "endpoint.json").read_text())
    assert endpoint.get("proxy_token") is None
    want = px.proxy_client_headers(endpoint)["Authorization"]
    assert want == f"Bearer {px.PROXY_CLIENT_BEARER_DEFAULT}"
    seen = []
    real = urllib.request.urlopen

    def spy(req, *a, **kw):
        seen.append(dict(req.header_items()))
        return real(req, *a, **kw)
    monkeypatch.setattr(urllib.request, "urlopen", spy)
    _bootstrap(world, tmp_path, world.out, world.ref_path)     # its render-agreement check posts to the proxy
    assert seen, "the bootstrap made no request to the standup's proxy"
    assert all(h.get("Authorization") == want for h in seen), seen


def test_the_bootstrap_binds_dispatches_and_the_gate_completes_against_the_live_standup(
        world, tmp_path, monkeypatch):
    calls = []

    class _Recording(gate.ReferenceOracle):
        def set_current(self, cell):
            calls.append(cell)
            super().set_current(cell)
    monkeypatch.setattr(gate, "ReferenceOracle", _Recording)
    spec = cm.load_petri_spec(world.manifest)
    inputs = gate._bootstrap_lora_native_stack(_args(tmp_path / "g1", world.out, world.ref_path), spec)
    assert inputs.engine == "lora_native" and inputs.phase0_evidence == {}
    assert set(inputs.kind_serving) == {GEMMA_HCELL}                    # GENUINE only, never the zero leg
    assert set(inputs.kind_serving[GEMMA_HCELL]) == {"lora"}
    assert set(inputs.noop_serving) == {"passthrough", "lora"}
    assert inputs.multi_lora.zero_adapter == f"petri__{ZERO}"
    assert inputs.multi_lora.adapters == (f"petri__{GEMMA_HCELL}", f"petri__{ZERO}")
    assert inputs.multi_lora.base == "petri__base"
    # the recorder access through the standup's proxy pass-through
    routing = inputs.multi_lora.routing
    assert routing is not None
    assert routing.selection_adapter == {"petri__base": None, f"petri__{ZERO}": ZERO,
                                         f"petri__{GEMMA_HCELL}": GEMMA_HCELL}
    assert routing.id_name_map == {"1": ZERO, "2": GEMMA_HCELL}
    epoch = "c" * 32                                                # the control mints; every worker adopts
    armed = routing.arm(epoch)
    assert armed["armed"] is True and armed["runner_kind"] == "v2" and armed["n_workers"] == 1
    assert armed["epoch"] == epoch and armed["already_armed"] is False
    assert routing.arm(epoch)["already_armed"] is True              # idempotent for the live epoch
    with pytest.raises(gate.GateError, match="already_armed"):
        routing.arm("d" * 32)
    rows = routing.mixed_batch([("petri__base", gate.GATE_PROMPT_SET[0]),
                                (f"petri__{GEMMA_HCELL}", gate.GATE_PROMPT_SET[1])])
    assert [ids for ids, _rid in rows] == [list(BASE_IDS), list(GENUINE_IDS)]
    rids = [rid for _ids, rid in rows]
    assert all(r.startswith("chatcmpl-") for r in rids) and len(set(rids)) == 2
    obs = routing.observe(epoch)
    # the fake engine renamed every request (`-<8 hex>`): the response ids are NOT the
    # observation keys, the join is `is_engine_id_of`
    assert not any(r in obs["observations"] for r in rids)
    eids = {r: [e for e in obs["observations"] if sa.is_engine_id_of(e, r)] for r in rids}
    assert all(len(v) == 1 for v in eids.values())
    assert obs["observations"][eids[rids[0]][0]] == {"lora_int_id": 0, "lora_name": None}
    assert obs["observations"][eids[rids[1]][0]] == {"lora_int_id": 2, "lora_name": GEMMA_HCELL}
    assert obs["epoch"] == epoch and obs["runner_kind"] == "v2"
    with pytest.raises(gate.GateError, match="epoch_mismatch"):
        routing.observe("another-epoch")
    assert routing.disarm(epoch)["disarmed"] is True
    assert inputs.stacked_identity is None and inputs.fidelity_evidence is None
    assert isinstance(inputs.reference, _Recording)
    # the passthrough AND the zero legs dispatch to the BASE oracle; the genuine leg to its arm
    calls.clear()
    assert inputs.noop_serving["passthrough"](gate.GATE_PROMPT_SET[0]) == list(BASE_IDS)
    assert inputs.noop_serving["lora"](gate.GATE_PROMPT_SET[0]) == list(BASE_IDS)
    assert inputs.kind_serving[GEMMA_HCELL]["lora"](gate.GATE_PROMPT_SET[0]) == list(GENUINE_IDS)
    assert calls == [None, None, GEMMA_HCELL]
    ev = json.loads((tmp_path / "g1" / gate.GATE_EVIDENCE).read_text())
    assert ev["engine"] == "lora_native" and ev["phase0"] == gate.PHASE0_NOT_APPLICABLE
    assert ev["reference_binding"]["held"] is True
    assert ev["reference_binding"]["object_artifact_sha256_by_cell"] == {GEMMA_HCELL: world.sha}
    assert all(r["matches_reference_render"] is True for r in ev["render_agreement"])
    assert ev["served_engine_stamps"] == ["lora_native"]
    assert ev["standup"]["attested_enforce_eager"] is False
    assert ev["multi_lora"]["policy"] == "native"
    assert ev["multi_lora"]["verdict_basis"].startswith("routing_attested")
    assert ev["multi_lora"]["rpc_route"] == "/sisteer/lora-routing/rpc"
    # the WHOLE gate subcommand against the live fake standup: rc 0, the record, no HF model
    monkeypatch.undo()
    loads_before = len(_StubShim.loads)
    rc = gate.main(["gate", "--manifest", str(world.manifest), "--out-dir", str(tmp_path / "g2"),
                    "--lora-native-standup", str(world.out), "--reference", str(world.ref_path)])
    assert rc == 0
    record = json.loads((tmp_path / "g2" / gate.GATE_RECORD).read_text())
    assert record["status"] == "completed" and record["phase0"] == gate.PHASE0_NOT_APPLICABLE
    assert record["kinds"] == {GEMMA_HCELL: {"passed_paths": ["lora"], "failed_paths": []}}
    assert {c["path"] for c in record["certificates"]} == {"passthrough", "lora"}
    assert record["multi_lora_checked"] is True
    # the structural selection verdict against the LIVE fake standup
    ml = record["multi_lora"]
    assert ml["verdict"] == "routing_attested" and ml["passed"] is True
    assert ml["n_requests"] == 3 * len(gate.GATE_PROMPT_SET) * 7 == ml["n_observed"]
    assert ml["id_join"]["n_suffixed"] == ml["n_requests"] and ml["n_ambiguous"] == 0
    assert ml["steps"]["n_steps_mixed"] >= 1 and ml["numerics"]["n_mismatched_rows"] == 0
    assert ml["runner_kind"] == "v2" and len(ml["epoch"]) == 32
    assert ml["recorder"]["disarm"]["disarmed"] is True and ml["recorder"]["disarm"]["epoch"] == ml["epoch"]
    assert (tmp_path / "g2" / "noop_certificates.json").is_file()
    assert not (tmp_path / "g2" / gate.FAILURE_MARKER).exists()
    assert len(_StubShim.loads) == loads_before                         # the canary loaded no model
    ev2 = json.loads((tmp_path / "g2" / gate.GATE_EVIDENCE).read_text())
    assert ev2["cross_engine_comparator"]["margin_atol_nats"] == gate.CROSS_ENGINE_NEARTIE_MARGIN_NATS
    assert {c["cell_id"] for c in ev2["served_calls"]} == {"none", ZERO, GEMMA_HCELL}


def _standup_copy(world, tmp_path, *, endpoint=None, cells_served=None, table=None):
    """The standup's four records copied to a scratch dir, each optionally
    mutated in place — the join checks run before any HTTP call."""
    d = tmp_path / "standup"
    d.mkdir(parents=True)
    for name in ("routing_table.json", "serving_attestation.json", "endpoint.json",
                 "cells_served.json"):
        shutil.copyfile(world.out / name, d / name)
    for name, mutate in (("endpoint.json", endpoint), ("cells_served.json", cells_served),
                         ("routing_table.json", table)):
        if mutate is not None:
            doc = json.loads((d / name).read_text())
            mutate(doc)
            (d / name).write_text(json.dumps(doc))
    return d


def _ref_copy(world, tmp_path, mutate):
    doc = json.loads(world.ref_path.read_text())
    mutate(doc)
    p = tmp_path / "reference_mut.json"
    p.write_text(json.dumps(doc))
    return p


def _bootstrap(world, tmp_path, standup_dir, ref_path, manifest=None):
    spec = cm.load_petri_spec(manifest or world.manifest)
    return gate._bootstrap_lora_native_stack(_args(tmp_path / "gate", standup_dir, ref_path), spec)


def test_a_reference_keyed_by_the_traintest_id_is_refused_naming_the_cell(world, tmp_path):
    def rekey(doc):
        for key in ("objects", "arms"):
            doc[key] = {GEMMA_CELL: doc[key][GEMMA_HCELL]}
    with pytest.raises(gate.GateBootstrapError,
                       match=f"reference.objects has no entry for genuine cell '{GEMMA_HCELL}'"):
        _bootstrap(world, tmp_path, world.out, _ref_copy(world, tmp_path, rekey))


@pytest.mark.parametrize("label, mutate, match", [
    ("space missing", lambda d: d.pop("cell_id_space"), "cell_id_space=None"),
    ("space traintest", lambda d: d.__setitem__("cell_id_space", "traintest"),
     "cell_id_space='traintest'"),
    ("object sha", lambda d: d["objects"][GEMMA_HCELL].__setitem__("sha256", "0" * 64),
     "armed with different object bytes"),
    ("model", lambda d: d.__setitem__("model", "someone/other-model"), "reference.model="),
    ("revision", lambda d: d.__setitem__("revision", "f" * 40), "reference.revision="),
    ("template", lambda d: d.__setitem__("chat_template_sha256", "e" * 64),
     "reference.chat_template_sha256="),
    ("prompt set", lambda d: d.__setitem__("gate_prompt_set_hash", "0123456789abcdef"),
     "reference.gate_prompt_set_hash="),
    ("max tokens", lambda d: d.__setitem__("gate_max_tokens", gate.GATE_MAX_TOKENS + 1),
     "reference.gate_max_tokens="),
    ("stale arm", lambda d: d["arms"].__setitem__("si_other__lora_r4", d["arms"][GEMMA_HCELL]),
     "reference.arms names \\['si_other__lora_r4'\\]"),
    ("no arm", lambda d: d["arms"].pop(GEMMA_HCELL), "no armed reference for genuine cell"),
    ("binding key missing", lambda d: d.pop("gate_max_tokens"), "lacks 'gate_max_tokens'"),
])
def test_a_reference_that_does_not_bind_to_the_standup_is_refused(world, tmp_path, label, mutate, match):
    with pytest.raises(gate.GateBootstrapError, match=match):
        _bootstrap(world, tmp_path, world.out, _ref_copy(world, tmp_path, mutate))


def test_a_standup_record_that_does_not_bind_is_refused(world, tmp_path):
    def zero_in_map(ep):
        ep["object_artifact_sha256_by_cell"][ZERO] = "0" * 64
    with pytest.raises(gate.GateBootstrapError, match=f"names '{ZERO}'"):
        _bootstrap(world, tmp_path / "a", _standup_copy(world, tmp_path / "a", endpoint=zero_in_map,
                                                        cells_served=zero_in_map), world.ref_path)

    def other_cell(ep):
        ep["object_artifact_sha256_by_cell"] = {"si_other__lora_r4": "1" * 64}
    with pytest.raises(gate.GateBootstrapError,
                       match=f"lora route '{GEMMA_HCELL}' has no verified object"):
        _bootstrap(world, tmp_path / "b", _standup_copy(world, tmp_path / "b", endpoint=other_cell,
                                                        cells_served=other_cell), world.ref_path)

    def extra_key(ep):
        ep["object_artifact_sha256_by_cell"]["si_other__lora_r4"] = "1" * 64
    with pytest.raises(gate.GateBootstrapError, match="names 'si_other__lora_r4', which no lora route"):
        _bootstrap(world, tmp_path / "c", _standup_copy(world, tmp_path / "c", endpoint=extra_key,
                                                        cells_served=extra_key), world.ref_path)

    def drift(cs):
        cs["object_artifact_sha256_by_cell"] = {GEMMA_HCELL: "2" * 64}
    with pytest.raises(gate.GateBootstrapError, match="two copies of the standup's object map"):
        _bootstrap(world, tmp_path / "d", _standup_copy(world, tmp_path / "d", cells_served=drift),
                   world.ref_path)

    def no_map(ep):
        ep.pop("object_artifact_sha256_by_cell")
    with pytest.raises(gate.GateBootstrapError, match="object_artifact_sha256_by_cell is missing"):
        _bootstrap(world, tmp_path / "e", _standup_copy(world, tmp_path / "e", endpoint=no_map),
                   world.ref_path)

    def v1(table):
        table["version"] = 1
        table.pop("adapters")
    with pytest.raises(gate.GateBootstrapError, match="version 1"):
        _bootstrap(world, tmp_path / "f", _standup_copy(world, tmp_path / "f", table=v1), world.ref_path)

    def lens_engine(ep):
        ep["engine"] = "lens"
    with pytest.raises(gate.GateBootstrapError, match="engine='lens'"):
        _bootstrap(world, tmp_path / "g", _standup_copy(world, tmp_path / "g", endpoint=lens_engine),
                   world.ref_path)
    # the reference file missing; a record missing; a manifest pinning another target
    with pytest.raises(gate.GateBootstrapError, match="--reference .* is missing"):
        _bootstrap(world, tmp_path / "h", world.out, tmp_path / "h" / "nowhere.json")
    partial = tmp_path / "i" / "standup"
    partial.mkdir(parents=True)
    shutil.copyfile(world.out / "routing_table.json", partial / "routing_table.json")
    with pytest.raises(gate.GateBootstrapError, match="has no serving_attestation"):
        _bootstrap(world, tmp_path / "i", partial, world.ref_path)
    other_manifest = REPO / "configs/capeval/manifests/petri_full22_nemotron3nano4b.yaml"
    with pytest.raises(gate.GateBootstrapError, match="--manifest pins"):
        _bootstrap(world, tmp_path / "j", world.out, world.ref_path, manifest=other_manifest)


def test_the_gate_cli_records_unavailable_when_the_reference_is_missing(world, tmp_path):
    rc = gate.main(["gate", "--manifest", str(world.manifest), "--out-dir", str(tmp_path / "u"),
                    "--lora-native-standup", str(world.out),
                    "--reference", str(tmp_path / "missing.json")])
    assert rc == gate.EXIT_BOOTSTRAP == 3
    marker = json.loads((tmp_path / "u" / gate.UNAVAILABLE_MARKER).read_text())
    assert "is missing" in marker["reason"]


def test_a_render_disagreement_with_the_oracle_is_a_validation_failure(world, tmp_path):
    """A reference whose prompt renders differ from what the standup's proxy
    serves is a GateError (exit 4 at the CLI), never a silent canary."""
    def other_render(doc):
        doc["prompt_ids"] = [[9, 9, 9] for _ in doc["prompt_ids"]]
    ref = _ref_copy(world, tmp_path, other_render)
    with pytest.raises(gate.GateError, match="render DISAGREEMENT"):
        _bootstrap(world, tmp_path, world.out, ref)
    assert (tmp_path / "gate" / "render_disagreement.json").is_file()
    rc = gate.main(["gate", "--manifest", str(world.manifest), "--out-dir", str(tmp_path / "cli"),
                    "--lora-native-standup", str(world.out), "--reference", str(ref)])
    assert rc == gate.EXIT_FAILED == 4
    assert (tmp_path / "cli" / gate.FAILURE_MARKER).is_file()


# --------------------------------------------------------------------------- #
# 3. the producer's digest binding and the CLI's argparse rules                 #
# --------------------------------------------------------------------------- #


def test_the_producer_refuses_a_sha_mismatch_before_loading_anything(tmp_path, monkeypatch):
    _, man_path, cfg = _campaign("gemma4e4b")
    tok = cached_or_skip(cfg)
    obj = tmp_path / "object.pt"
    obj.write_bytes(b"not the pinned bytes")
    rc, ref_path = _produce_reference(monkeypatch, tmp_path, manifest=man_path, tok=tok,
                                      objects={GEMMA_HCELL: obj}, shas={GEMMA_HCELL: "0" * 64})
    assert rc == gate.EXIT_BOOTSTRAP
    rec = json.loads(ref_path.read_text())
    assert "not the pinned object" in rec["error"] and "refusing BEFORE loading it" in rec["error"]
    assert _StubShim.loads == [] and _StubShim.armed == []          # nothing loaded, nothing armed
    assert "base" not in rec and rec["stage"] == "bind"
    # a cell with NO digest refuses; a digest for an unlisted cell refuses; --cells-json + digests refuses
    rc, ref_path = _produce_reference(monkeypatch, tmp_path / "b", manifest=man_path, tok=tok,
                                      objects={GEMMA_HCELL: obj}, shas={})
    assert rc == gate.EXIT_BOOTSTRAP and "carry no --expect-object-sha256" in json.loads(
        ref_path.read_text())["error"]
    import hashlib
    good = hashlib.sha256(obj.read_bytes()).hexdigest()
    rc, ref_path = _produce_reference(monkeypatch, tmp_path / "c", manifest=man_path, tok=tok,
                                      objects={GEMMA_HCELL: obj},
                                      shas={GEMMA_HCELL: good, "si_other__lora_r4": good})
    assert rc == gate.EXIT_BOOTSTRAP and "not in --objects-json" in json.loads(
        ref_path.read_text())["error"]
    with pytest.raises(gate.GateError, match="exactly one of"):
        gate.reference_object_inputs(cells_json=None, objects_json=None, expect_object_sha256=None)
    with pytest.raises(gate.GateError, match="binds --objects-json inputs only"):
        gate.reference_object_inputs(cells_json=str(tmp_path / "x.json"), objects_json=None,
                                     expect_object_sha256=[f"{GEMMA_HCELL}={good}"])
    with pytest.raises(gate.GateError, match="not CELL=SHA256"):
        gate._parse_object_shas(["nonsense"])
    with pytest.raises(gate.GateError, match="not a lowercase 64-hex"):
        gate._parse_object_shas([f"{GEMMA_HCELL}=abc"])
    # the matching digest is bound and recorded (the shim then loads exactly once)
    rc, ref_path = _produce_reference(monkeypatch, tmp_path / "d", manifest=man_path, tok=tok,
                                      objects={GEMMA_HCELL: obj}, shas={GEMMA_HCELL: good})
    rec = json.loads(ref_path.read_text())
    assert rc == 0 and rec["cell_id_space"] == "harness"
    assert rec["objects"] == {GEMMA_HCELL: {"object_pt": str(obj), "sha256": good}}
    assert len(_StubShim.loads) == 1 and _StubShim.armed[0][2] == GEMMA_HCELL


def test_the_legacy_cells_json_path_records_the_same_bindings(tmp_path, monkeypatch):
    """``--cells-json`` (the registry path) keeps working and now records the
    objects' shas and the bindings too (additive; the registry bootstrap reads
    only base / arms / prompt_ids)."""
    _, man_path, cfg = _campaign("gemma4e4b")
    tok = cached_or_skip(cfg)
    _install_stub_shim(monkeypatch, tok)
    art = tmp_path / "artifact.bin"
    art.write_bytes(b"registry bytes")
    cells_json = tmp_path / "cells.json"
    cells_json.write_text(json.dumps({"obj_cast": str(art)}))
    out = tmp_path / "ref.json"
    rc = gate.main(["gate-worker-reference", "--manifest", str(man_path), "--out", str(out),
                    "--cells-json", str(cells_json)])
    rec = json.loads(out.read_text())
    assert rc == 0 and set(rec["arms"]) == {"obj_cast"} and rec["cell_id_space"] == "harness"
    import hashlib
    assert rec["objects"] == {"obj_cast": {"object_pt": str(art),
                                           "sha256": hashlib.sha256(b"registry bytes").hexdigest()}}
    assert rec["chat_template_sha256"] == hashlib.sha256(tok.chat_template.encode()).hexdigest()


def test_argparse_refuses_the_registry_dir_beside_a_standup_and_a_standup_without_its_reference(
        tmp_path, capsys):
    base = ["gate", "--manifest", "m.yaml", "--out-dir", str(tmp_path)]
    with pytest.raises(SystemExit) as ei:
        gate.main(base + ["--registry-dir", "r", "--lora-native-standup", "s", "--reference", "ref"])
    assert ei.value.code == 2 and "mutually exclusive" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        gate.main(base + ["--lora-native-standup", "s"])
    assert "required TOGETHER" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        gate.main(base + ["--reference", "ref"])
    assert "required TOGETHER" in capsys.readouterr().err


# --------------------------------------------------------------------------- #
# the live proxy is held to the records                                          #
# --------------------------------------------------------------------------- #

CCCC, DDDD = "c" * 16, "d" * 16


def _forwarding_stub(real_v1: str, headers, *, rewrite):
    """A stand-in proxy on a local port that FORWARDS ``GET /v1/models`` and
    ``POST /v1/chat/completions`` to the REAL live standup and hands every
    answer to ``rewrite(method, path, doc)`` before returning it — the shape
    of a proxy whose response stamps disagree with its inventory (the
    reproduction: the table names one intervention, the genuine responses
    stamp another)."""
    import threading
    import urllib.request
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    state = {"gets": 0, "posts": 0}

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _forward(self, method, body=None):
            req = urllib.request.Request(real_v1 + self.path[len("/v1"):], data=body, method=method,
                                         headers={**dict(headers), "Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read())

        def _send(self, doc):
            data = json.dumps(doc).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            state["gets"] += 1
            self._send(rewrite("GET", self.path, self._forward("GET")))

        def do_POST(self):
            state["posts"] += 1
            body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            self._send(rewrite("POST", self.path, self._forward("POST", body)))

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, state


def _standup_behind(world, tmp_path, stub_port):
    """The standup's records with endpoint.proxy_url pointed at the stub."""
    def repoint(ep):
        ep["proxy_url"] = f"http://127.0.0.1:{stub_port}/v1"
    return _standup_copy(world, tmp_path, endpoint=repoint)


def test_the_live_inventory_is_held_to_the_records_before_any_probe(world, tmp_path, monkeypatch):
    """A1: the bootstrap fetches the live proxy's /v1/models and holds its
    engine block to the table + attestation it loaded BEFORE a single probe.
    Positive: the real standup binds and the evidence records it. Negatives,
    each refusing before any POST (spied): the table's block carries another
    weights digest; the table names intervention cccc… while the live adapter
    serves dddd… (the reproduction's table side)."""
    import urllib.request
    from sisteer.capeval.serving import proxy as px
    spec = cm.load_petri_spec(world.manifest)
    inputs = gate._bootstrap_lora_native_stack(_args(tmp_path / "ok", world.out, world.ref_path), spec)
    ev = json.loads((tmp_path / "ok" / gate.GATE_EVIDENCE).read_text())
    live = ev["live_inventory"]
    assert live["held"] is True and live["engine"]["kind"] == "lora_native"
    assert set(live["engine"]["adapters"]) == {GEMMA_HCELL, ZERO}
    assert set(live["served_models"]) >= {"petri__base", f"petri__{GEMMA_HCELL}", f"petri__{ZERO}"}
    assert all(c.get("stamps_held") is True for c in ev["served_calls"])
    assert inputs.kind_serving[GEMMA_HCELL]["lora"](gate.GATE_PROMPT_SET[0]) == list(GENUINE_IDS)
    # the pure validator on the REAL live payload and the loaded records
    table = json.loads((world.out / "routing_table.json").read_text())
    from sisteer.capeval.serving import serving_attestation as sa
    att = sa.read_serving_attestation(world.out / sa.SERVING_ATTESTATION_FILE)
    models = world.su.proxy.models_payload()
    held = gate.hold_live_inventory(models, table=px.load_routing_table(world.out / "routing_table.json"),
                                    attestation=att)
    assert held["held"] and held["engine"]["adapters"][GEMMA_HCELL]["petri_intervention_hash"] == \
        table["adapters"][GEMMA_HCELL]["petri_intervention_hash"]

    methods = []
    real = urllib.request.urlopen

    def spy(req, *a, **kw):
        methods.append(req.get_method())
        return real(req, *a, **kw)
    monkeypatch.setattr(urllib.request, "urlopen", spy)

    def other_weights(t):
        t["adapters"][GEMMA_HCELL]["weights_sha256"] = "9" * 64
    methods.clear()
    with pytest.raises(gate.GateError, match="refusing before any probe") as ei:
        _bootstrap(world, tmp_path / "w", _standup_copy(world, tmp_path / "w", table=other_weights),
                   world.ref_path)
    assert f"adapter '{GEMMA_HCELL}'.weights_sha256" in str(ei.value) and "999999" in str(ei.value)
    assert methods == ["GET"], methods                      # no probe was sent

    def cccc_table(t):
        # the reproduction's table side: the genuine route and its block name cccc…
        for r in t["routes"]:
            if r["cell_id"] == GEMMA_HCELL:
                r["intervention_hash"] = CCCC
        t["adapters"][GEMMA_HCELL]["petri_intervention_hash"] = CCCC
    methods.clear()
    real_hash = table["adapters"][GEMMA_HCELL]["petri_intervention_hash"]
    with pytest.raises(gate.GateError, match="refusing before any probe") as ei:
        _bootstrap(world, tmp_path / "c", _standup_copy(world, tmp_path / "c", table=cccc_table),
                   world.ref_path)
    msg = str(ei.value)
    assert f"adapter '{GEMMA_HCELL}'.petri_intervention_hash: table '{CCCC}' != live '{real_hash}'" in msg
    assert f"lora route '{GEMMA_HCELL}'.intervention_hash '{CCCC}'" in msg
    assert methods == ["GET"], methods
    # a proxy whose /v1/models attests no engine is not a lora-native standup's
    with pytest.raises(gate.GateError, match="has no 'engine' block"):
        gate.hold_live_inventory({"data": [], "petri_proxy": {"version": 1}},
                                 table=px.load_routing_table(world.out / "routing_table.json"), attestation=att)
    with pytest.raises(gate.GateError, match="no petri_proxy identity block"):
        gate.hold_live_inventory({"data": []}, table=px.load_routing_table(world.out / "routing_table.json"),
                                 attestation=att)


def test_every_response_stamp_is_held_to_its_route(world, tmp_path):
    """A1, the reproduction: the inventory matches, yet every GENUINE response
    stamps intervention dddd… while the table (and the live inventory) say the
    real hash. The stamps used to be recorded and compared with nothing; the
    gate passed. Now the first genuine call refuses by name — through the
    served function and through the whole ``gate`` subcommand (rc 4, the
    failure marker naming both hashes); a response without its petri block
    refuses too; the base and zero legs are unaffected."""
    real_hash = json.loads((world.out / "routing_table.json").read_text())["adapters"][GEMMA_HCELL][
        "petri_intervention_hash"]
    from sisteer.capeval.serving import proxy as px
    headers = px.proxy_client_headers(json.loads((world.out / "endpoint.json").read_text()))

    def restamp(method, path, doc):
        if method == "POST" and isinstance(doc.get("petri"), dict) and doc["petri"].get("cell_id") == GEMMA_HCELL:
            doc["petri_intervention_hash"] = DDDD
            doc["petri"]["intervention_hash"] = DDDD
        return doc
    srv, state = _forwarding_stub(world.su.proxy_url, headers, rewrite=restamp)
    try:
        standup = _standup_behind(world, tmp_path / "s", srv.server_address[1])
        spec = cm.load_petri_spec(world.manifest)
        inputs = gate._bootstrap_lora_native_stack(_args(tmp_path / "s" / "gate", standup, world.ref_path), spec)
        assert state["gets"] >= 1                                   # the inventory was fetched through the stub
        assert inputs.noop_serving["passthrough"](gate.GATE_PROMPT_SET[0]) == list(BASE_IDS)
        assert inputs.noop_serving["lora"](gate.GATE_PROMPT_SET[0]) == list(BASE_IDS)
        with pytest.raises(gate.GateError, match="does not carry the bound intervention") as ei:
            inputs.kind_serving[GEMMA_HCELL]["lora"](gate.GATE_PROMPT_SET[0])
        msg = str(ei.value)
        assert f"petri_intervention_hash: stamped '{DDDD}', expected '{real_hash}'" in msg
        assert f"petri.intervention_hash: stamped '{DDDD}', expected '{real_hash}'" in msg
        assert f"cell '{GEMMA_HCELL}'" in msg
        # the whole gate: a validation FAILURE (rc 4) naming the hashes, never a completed record
        rc = gate.main(["gate", "--manifest", str(world.manifest), "--out-dir", str(tmp_path / "s" / "cli"),
                        "--lora-native-standup", str(standup), "--reference", str(world.ref_path)])
        assert rc == gate.EXIT_FAILED == 4
        marker = json.loads((tmp_path / "s" / "cli" / gate.FAILURE_MARKER).read_text())
        assert DDDD in marker["reason"] and real_hash in marker["reason"]
        assert not (tmp_path / "s" / "cli" / gate.GATE_RECORD).exists()
    finally:
        srv.shutdown()

    def strip_stamps(method, path, doc):
        if method == "POST":
            doc.pop("petri", None)
        return doc
    srv2, _ = _forwarding_stub(world.su.proxy_url, headers, rewrite=strip_stamps)
    try:
        standup2 = _standup_behind(world, tmp_path / "t", srv2.server_address[1])
        with pytest.raises(gate.GateError, match="carries no petri stamp block"):
            _bootstrap(world, tmp_path / "t", standup2, world.ref_path)   # the first base probe refuses
    finally:
        srv2.shutdown()

    def other_identity(method, path, doc):
        if method == "POST" and isinstance(doc.get("petri"), dict) and doc["petri"].get("cell_id") == "none":
            doc["petri"]["serving_identity"] = "proxy_vllm+lens+interleaved"
            doc["petri"]["cell_id"] = "none__lens0"
        return doc
    srv3, _ = _forwarding_stub(world.su.proxy_url, headers, rewrite=other_identity)
    try:
        standup3 = _standup_behind(world, tmp_path / "u", srv3.server_address[1])
        with pytest.raises(gate.GateError, match="does not carry the bound intervention") as ei:
            _bootstrap(world, tmp_path / "u", standup3, world.ref_path)
        assert "petri.cell_id: stamped 'none__lens0', expected 'none'" in str(ei.value)
        assert "petri.serving_identity: stamped 'proxy_vllm+lens+interleaved'" in str(ei.value)
    finally:
        srv3.shutdown()
