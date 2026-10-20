"""Petri Bloom Phase 2 serving-gate LOGIC with FAKE engines — CPU-only.
The real GPU run is ``scripts/sbatch/petri_gate.sbatch`` (launch-time).

Covers: the exact-token-equality comparator (pass = equality, numeric
deltas are diagnostics only — H5); certificate keying + the H5 mode
binding + Tier-2 refusal without a mode-matching certificate
(``check_pairing`` fed from the CertificateStore); the per-kind canary
(PASS on any committed path; every path failing = a HARD gate failure,
no HF fallback); the two SEPARATE no-op certificates (base + scale-0) +
the zero-LoRA one; multi-LoRA mixed-batch selection; the
recorded-unavailable path; the live smoke's four pass criteria; the
8 Phase-0 assertion checks; the worker-side object registry (export /
preload / selector refusals — H3); the vllm serve composition invariants;
the zero-adapter export; and the OpenAI shim's translation over a fake
HF engine.
"""

import importlib.util
import inspect
import io
import json
import sys
from pathlib import Path

import pytest
import torch

from sisteer.capeval import manifest as cm
from sisteer.capeval.petri import defense_cells as dc
from sisteer.capeval.serving import lens_adapters as la
from sisteer.capeval.serving import lens_steer as ls
from sisteer.capeval.serving import lora_export as lx
from sisteer.capeval.serving import openai_shim as osh
from sisteer.capeval.serving import vllm_server as vs
from sisteer.capeval.serving import GenOut

ROOT = Path(__file__).parents[1]
_spec_path = importlib.util.spec_from_file_location(
    "_petri_serving_gate", ROOT / "scripts" / "petri_serving_gate.py")
gate = importlib.util.module_from_spec(_spec_path)
sys.modules["_petri_serving_gate"] = gate
_spec_path.loader.exec_module(gate)

if "_petri_steered_smoke" in sys.modules:      # the gating tests load it too
    smoke = sys.modules["_petri_steered_smoke"]
else:
    _smoke_spec = importlib.util.spec_from_file_location(
        "_petri_steered_smoke", ROOT / "scripts" / "petri_steered_smoke.py")
    smoke = importlib.util.module_from_spec(_smoke_spec)
    sys.modules["_petri_steered_smoke"] = smoke
    _smoke_spec.loader.exec_module(smoke)

REV = "0123456789abcdef0123456789abcdef01234567"
IMPL_COMMIT = "fedcba9876" * 4
SAFETY_PREFIX = b"synthetic safety instruction: be safe (test bytes)"
CONSTITUTION_PREFIX = (b"synthetic constitution instruction: many "
                       b"principles (test bytes)")


BLOOM_OPTIONS = dict(max_turns=40, enable_prefill=False,
                     enable_rollback=False, compaction=False,
                     realism_filter=False, epochs=1, seed=0)


def _spec(**over) -> cm.PetriSpec:
    kwargs = dict(
        audit="petri-twosi-test",
        behaviors=list(cm.PETRI_BEHAVIORS),
        cells=["none", "si_safety__cast"],
        scenario_bundle_hash="ab" * 8,
        auditor_model="anthropic/claude-sonnet-5",
        judge_model="anthropic/claude-opus-5",
        scenarios_model="anthropic/claude-sonnet-5",
        prompt_provenance="package-pinned@petri-bloom==0.2.6",
        target_model="meta-llama/Llama-3.1-8B-Instruct",
        target_revision=REV,
        serving_backend="proxy_vllm",
        serving_mode="exact_replay",
        serving_packages={"vllm": "0.25.1+cu129", "vllm-lens": "1.2.1"},
        serving_launch_flags=["--enable-lora", "--enforce-eager"],
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


PROMPT_FILES = {"auditor": b"# synthetic auditor prompt bytes\n",
                "judge": b"# synthetic judge prompt bytes\n"}

#: the pinned chat-template bytes the vllm_server tests launch with —
#: hashed into the spec so build_server_launch's verification passes.
CHAT_TEMPLATE_BYTES = b"{# synthetic pinned chat template #}\n"
CHAT_TEMPLATE_HASH = cm.sha256_hex16(CHAT_TEMPLATE_BYTES)


def _behavior_files(spec):
    return {k: f"# behavior {k}\n".encode()
            for k in cm.petri_expected_behavior_keys(spec)}


def _chat_template(tmp_path) -> Path:
    tpl = tmp_path / "chat_template.jinja"
    tpl.write_bytes(CHAT_TEMPLATE_BYTES)
    return tpl


def _synthetic_artifact(layer=3, dim=8, seed=0) -> bytes:
    g = torch.Generator().manual_seed(seed)
    buf = io.BytesIO()
    torch.save({"layer": layer,
                "vector": torch.randn(dim, generator=g)}, buf)
    return buf.getvalue()


def _tiny_peft_adapter(adapter_dir, *, r=4, layers=(0, 1), d_in=24, d_out=32,
                       seed=0, lora_alpha=None, target_modules=("o_proj",),
                       module_prefix="base_model.model.model.layers",
                       site="self_attn", dtype=torch.float32,
                       extra_config=None) -> Path:
    """A synthetic PEFT o_proj adapter directory in the shape every
    admitted native export writes — fp32 factors, ``lora_alpha == r``,
    ``target_modules == ["o_proj"]`` — on invented weights (generator-
    seeded; ``d_in != d_out`` so factor orientation is never ambiguous).
    Every knob exists so a test can build the one deviation it refuses."""
    from safetensors.torch import save_file

    adapter_dir = Path(adapter_dir)
    adapter_dir.mkdir(parents=True, exist_ok=True)
    g = torch.Generator().manual_seed(seed)
    tensors = {}
    for layer in layers:
        m = f"{module_prefix}.{layer}.{site}.o_proj"
        tensors[f"{m}.lora_A.weight"] = torch.randn(
            r, d_in, generator=g).to(dtype)
        tensors[f"{m}.lora_B.weight"] = (
            torch.randn(d_out, r, generator=g) * 0.1).to(dtype)
    save_file(tensors, str(adapter_dir / lx.ADAPTER_WEIGHTS))
    config = {
        "peft_type": "LORA",
        "base_model_name_or_path": "meta-llama/Llama-3.1-8B-Instruct",
        "r": r,
        "lora_alpha": float(r if lora_alpha is None else lora_alpha),
        "lora_dropout": 0.0, "bias": "none",
        "target_modules": list(target_modules),
        "task_type": "CAUSAL_LM",
        "petri_genuine_adapter": True,
        "petri_intervention_hash": cm.sha256_hex16(
            f"{adapter_dir.name}-{seed}".encode()),
    }
    if extra_config:
        config.update(extra_config)
    (adapter_dir / lx.ADAPTER_CONFIG).write_text(
        json.dumps(config, indent=2, sort_keys=True))
    return adapter_dir


def _lens_cells(tmp_path, *, with_object=True, sidecar_bytes=None):
    """A minimal enumerated cell set whose one object cell is the
    SYNTHETIC kind (the shipped adapter), for registry tests."""
    cells = ["none"] + (["si_safety__cast"] if with_object else [])
    spec = _spec(cells=cells)
    entries = {"none": dc.RegistryEntry(cell_id="none", kind="none")}
    if with_object:
        artifact = tmp_path / "si_safety__cast.pt"
        artifact.write_bytes(_synthetic_artifact())
        sidecar_path = None
        if sidecar_bytes is not None:
            sidecar = tmp_path / "si_safety__cast.sidecar.json"
            sidecar.write_bytes(sidecar_bytes)
            sidecar_path = str(sidecar)
        entries["si_safety__cast"] = dc.RegistryEntry(
            cell_id="si_safety__cast", kind=la.SYNTHETIC_KIND,
            si_id="safety", object_path=str(artifact),
            sidecar_path=sidecar_path,
            serving_path="lens",
            resolved_spec={"kind": la.SYNTHETIC_KIND, "si_id": "safety",
                           "layer": 3})
    source = dc.MappingCellSource(entries)
    return spec, source, dc.enumerate_cells(
        spec, source, behavior_files=_behavior_files(spec),
        prompt_files=PROMPT_FILES)


# --------------------------------------------------------------------------
# the exact-equality comparator (H5)
# --------------------------------------------------------------------------


def test_exact_token_equality_passes_only_on_equality():
    ok = gate.exact_token_equality([1, 2, 3], [1, 2, 3], prompt_index=0)
    assert ok.passed and ok.first_divergence is None
    div = gate.exact_token_equality([1, 2, 9, 4], [1, 2, 3, 4],
                                    prompt_index=1)
    assert not div.passed and div.first_divergence == 2
    # a length mismatch diverges at the shorter length
    short = gate.exact_token_equality([1, 2], [1, 2, 3], prompt_index=2)
    assert not short.passed and short.first_divergence == 2
    assert short.served_len == 2 and short.reference_len == 3


def test_numeric_diagnostics_never_flip_the_pass_criterion():
    """H5: relative-delta / logit diagnostics ride along; they are never
    consulted for pass/fail — a huge diagnostic on equal tokens still
    passes, a tiny one on unequal tokens still fails."""
    ok = gate.exact_token_equality(
        [5, 6], [5, 6], prompt_index=0,
        diagnostics={"max_abs_logit_delta": 123.0})
    assert ok.passed and ok.diagnostics["max_abs_logit_delta"] == 123.0
    bad = gate.exact_token_equality(
        [5, 7], [5, 6], prompt_index=0,
        diagnostics={"max_abs_logit_delta": 1e-9})
    assert not bad.passed


# --------------------------------------------------------------------------
# certificate keying + mode binding (H5) + check_pairing integration
# --------------------------------------------------------------------------


def test_certificate_key_carries_all_six_fields_and_the_spec_mode():
    cert = gate.certificate_for(_spec(), path="lens")
    assert cert == dc.NoOpCertificate(
        backend="proxy_vllm", path="lens", mode="exact_replay",
        model_revision=REV, serving_impl_commit=IMPL_COMMIT,
        gate_prompt_set_hash=gate.gate_prompt_set_hash())
    # the mode is the SPEC's pinned mode — the gate certifies the mode
    # the audit runs in, and only that mode (H5)
    inter = gate.certificate_for(_spec(serving_mode="interleaved"),
                                 path="lens")
    assert inter.mode == "interleaved"
    # the HF shim is the reference oracle, never a certified serving
    # path: "apply" refuses at identity validation
    with pytest.raises(cm.ManifestError, match="apply"):
        gate.certificate_for(_spec(), path="apply")


def test_certificate_refuses_unfrozen_impl_commit():
    with pytest.raises(gate.GateError, match="UNFROZEN"):
        gate.certificate_for(_spec(serving_impl_commit=cm.PETRI_UNFROZEN),
                             path="lens")


def test_certificate_store_round_trip_and_idempotence(tmp_path):
    store = gate.CertificateStore(tmp_path / "certs.json")
    assert store.certificates() == []
    cert = gate.certificate_for(_spec(), path="lens")
    store.record(cert, evidence={"prompts": 6, "all_exact": True})
    store.record(cert, evidence={"prompts": 6, "all_exact": True})
    # transactional: staged records license NOTHING until publish
    assert store.certificates() == []
    store.publish()
    assert store.certificates() == [cert]  # same key -> one record


def test_certificates_are_transactional_across_runs(tmp_path):
    """G3: a later unavailable/failed run must not leave an earlier
    success's certificates readable — begin_run quarantines the published
    record, and only a full-success publish restores readability."""
    store = gate.CertificateStore(tmp_path / "certs.json")
    cert = gate.certificate_for(_spec(), path="lens")
    store.record(cert, evidence={"prompts": 6, "all_exact": True})
    store.publish()
    assert store.certificates() == [cert]  # run 1 succeeded
    # run 2 starts and then fails/aborts before publishing anything:
    store.begin_run()
    assert store.certificates() == []  # the stale success licenses nothing
    # ... and stays unusable however the run ends, publish never called
    with pytest.raises(gate.GateError, match="nothing staged"):
        store.publish()
    assert store.certificates() == []
    # run 3 completes: its published record is the only readable one
    store.begin_run()
    store.record(cert, evidence={"prompts": 6, "all_exact": True})
    store.publish()
    assert store.certificates() == [cert]


def test_noop_certificates_license_tier2_only_in_their_own_mode(tmp_path):
    """The full loop: gate certificates -> check_pairing. exact_replay
    certificates license exact_replay Tier-2 cross-path references and
    NOTHING in interleaved (H5)."""
    spec = _spec()
    store = gate.CertificateStore(tmp_path / "certs.json")

    def ref(messages):
        return [1, 2, 3]

    store.begin_run()
    for path in ("lens", "lora", "passthrough"):
        gate.run_noop_certificate(spec, store, path=path,
                                  served=ref, reference=ref)
    store.publish()
    certs = store.certificates()
    arm = dc.PairingLeg(cell_id="si_safety__cast",
                        serving_identity="proxy_vllm+lens+exact_replay")
    base = dc.PairingLeg(cell_id="none",
                         serving_identity="proxy_vllm+passthrough+exact_replay")
    ok = dc.check_pairing(
        arm, base, tier=2, certificates=certs, model_revision=REV,
        serving_impl_commit=IMPL_COMMIT,
        gate_prompt_set_hash=gate.gate_prompt_set_hash())
    assert ok.cross_path is True
    arm_i = dc.PairingLeg(cell_id="si_safety__cast",
                          serving_identity="proxy_vllm+lens+interleaved")
    base_i = dc.PairingLeg(
        cell_id="none",
        serving_identity="proxy_vllm+passthrough+interleaved")
    with pytest.raises(dc.PairingError, match="certificate"):
        dc.check_pairing(
            arm_i, base_i, tier=2, certificates=certs, model_revision=REV,
            serving_impl_commit=IMPL_COMMIT,
            gate_prompt_set_hash=gate.gate_prompt_set_hash())
    # and with NO certificates at all, Tier-2 cross-path refuses
    with pytest.raises(dc.PairingError, match="certificate"):
        dc.check_pairing(
            arm, base, tier=2, certificates=(), model_revision=REV,
            serving_impl_commit=IMPL_COMMIT,
            gate_prompt_set_hash=gate.gate_prompt_set_hash())


def test_failed_noop_certificate_records_nothing(tmp_path):
    spec = _spec()
    store = gate.CertificateStore(tmp_path / "certs.json")

    def served(messages):
        return [1, 2, 4]  # diverges

    def reference(messages):
        return [1, 2, 3]

    store.begin_run()
    with pytest.raises(gate.GateError, match="no-op certificate FAILED"):
        gate.run_noop_certificate(spec, store, path="lens",
                                  served=served, reference=reference)
    assert store.certificates() == []  # a failed no-op certifies nothing
    with pytest.raises(gate.GateError, match="nothing staged"):
        store.publish()  # and nothing was staged behind it either


def test_base_and_scale0_are_separate_certificates(tmp_path):
    """H5: base/no-object and scale-0 are two SEPARATE exact checks —
    each records its own certificate under its own path key."""
    spec = _spec()
    store = gate.CertificateStore(tmp_path / "certs.json")

    def ref(messages):
        return [7, 8]

    store.begin_run()
    base_cert = gate.run_noop_certificate(spec, store, path="passthrough",
                                          served=ref, reference=ref)
    scale0_cert = gate.run_noop_certificate(spec, store, path="lens",
                                            served=ref, reference=ref)
    store.publish()
    assert base_cert != scale0_cert
    assert {c.path for c in store.certificates()} == {"passthrough", "lens"}


# --------------------------------------------------------------------------
# the per-kind canary: PASS, or a HARD gate failure when every committed
# path fails (no HF fallback)
# --------------------------------------------------------------------------


def _reference(messages):
    return [len(str(messages)) % 7, 1, 2]


def _served_wrong(messages):
    return [99] + _reference(messages)[1:]


def test_kind_canary_passes_when_every_path_matches():
    outcome = gate.run_kind_canary(
        "obj_cast", served_by_path={"lens": _reference},
        reference=_reference)
    assert outcome.passed
    assert outcome.passed_paths == ("lens",)
    assert len(outcome.path_comparisons["lens"]) == len(gate.GATE_PROMPT_SET)


def test_kind_canary_one_failed_path_is_not_a_gate_failure():
    """G9: a lens mismatch alone must NOT fail the kind when its
    committed LoRA path matches the fp32 reference — the kind PASSES (on
    the LoRA path) and BOTH attempts are persisted."""
    outcome = gate.run_kind_canary(
        "obj_lora_r1",
        served_by_path={"lens": _served_wrong, "lora": _reference},
        reference=_reference)
    assert outcome.passed
    assert outcome.passed_paths == ("lora",)
    assert outcome.failed_paths == ("lens",)
    # both paths' comparisons are on the record
    assert set(outcome.path_comparisons) == {"lens", "lora"}
    assert all(c.first_divergence == 0
               for c in outcome.path_comparisons["lens"])
    assert all(c.passed for c in outcome.path_comparisons["lora"])


def test_kind_canary_every_path_failing_is_not_passed():
    bad = gate.run_kind_canary(
        "obj_lofit",
        served_by_path={"lens": _served_wrong, "lora": _served_wrong},
        reference=_reference)
    # the outcome itself is filed (every per-path attempt persisted);
    # run_gate turns it into the HARD gate failure: an unservable kind
    # blocks the audit, never an HF fallback
    assert not bad.passed
    assert bad.failed_paths == ("lens", "lora")


def test_kind_canary_refuses_an_empty_or_unknown_path_set():
    with pytest.raises(gate.GateError, match="no committed serving paths"):
        gate.run_kind_canary("obj_cast", served_by_path={},
                             reference=_reference)
    with pytest.raises(gate.GateError, match="unknown serving path"):
        gate.run_kind_canary("obj_cast",
                             served_by_path={"apply": _reference},
                             reference=_reference)


def test_gate_prompt_set_is_frozen_by_hash():
    h = gate.gate_prompt_set_hash()
    assert h == gate.gate_prompt_set_hash()  # deterministic
    assert len(h) == cm.HASH_HEX_LEN
    # every prompt is plain chat messages (the gate proves SERVING
    # correctness on harmless inputs)
    for messages in gate.GATE_PROMPT_SET:
        for m in messages:
            assert set(m) == {"role", "content"}


# --------------------------------------------------------------------------
# multi-LoRA mixed-batch selection
# --------------------------------------------------------------------------


def _lora_world():
    table = {"base": [1, 1], "adapter_a": [2, 2], "adapter_b": [3, 3]}

    def solo(name, messages):
        return list(table[name])

    def mixed(batch):
        return [list(table[name]) for name, _ in batch]

    return table, solo, mixed


def test_multi_lora_mixed_batch_selection_passes():
    table, solo, mixed = _lora_world()
    gate.run_multi_lora_selection(
        base="base", adapters=("adapter_a", "adapter_b"),
        mixed_batch=mixed, solo=solo)


def test_multi_lora_contamination_fails_loudly():
    table, solo, mixed = _lora_world()

    def contaminated(batch):
        outs = [list(table[name]) for name, _ in batch]
        # cross-request swap between two DIFFERENT selections' requests
        n = len(gate.GATE_PROMPT_SET)
        outs[0], outs[n] = outs[n], outs[0]  # base p0 <-> adapter_a p0
        return outs

    with pytest.raises(gate.GateError, match="contamination"):
        gate.run_multi_lora_selection(
            base="base", adapters=("adapter_a", "adapter_b"),
            mixed_batch=contaminated, solo=solo)


def test_multi_lora_needs_base_plus_two_distinct_adapters():
    """'base + several adapters' is the check's precondition — one
    adapter, duplicate adapters, base doubling as an adapter, or a
    missing base name all refuse rather than certifying a weaker mix."""
    table, solo, mixed = _lora_world()
    for adapters in (("adapter_a",),                    # only one adapter
                     ("adapter_a", "adapter_a"),        # duplicates
                     ("base", "adapter_a")):            # base is not an adapter
        with pytest.raises(gate.GateError, match="TWO distinct adapters"):
            gate.run_multi_lora_selection(base="base", adapters=adapters,
                                          mixed_batch=mixed, solo=solo)
    with pytest.raises(gate.GateError, match="explicit base"):
        gate.run_multi_lora_selection(base="", adapters=("a", "b"),
                                      mixed_batch=mixed, solo=solo)


def test_multi_lora_refuses_indistinguishable_routing():
    """If two selections produce identical solo outputs on EVERY
    gate prompt, the mixed-batch comparison cannot distinguish correct
    routing from a server sending everything to one of them — refuse to
    certify rather than false-pass on colliding observations."""
    table = {"base": [1, 1], "adapter_a": [1, 1], "adapter_b": [3, 3]}

    def solo(name, messages):
        return list(table[name])

    def mixed(batch):
        return [list(table[name]) for name, _ in batch]

    with pytest.raises(gate.GateError, match="cannot distinguish routing"):
        gate.run_multi_lora_selection(
            base="base", adapters=("adapter_a", "adapter_b"),
            mixed_batch=mixed, solo=solo)


def test_multi_lora_base_only_server_cannot_pass():
    """The concrete false-pass from the review: a server routing every
    request to base. With distinguishable adapters it now fails the
    mixed-vs-solo comparison (and with indistinguishable ones it refuses
    above — either way it can never certify)."""
    table, solo, _ = _lora_world()

    def base_only(batch):
        return [list(table["base"]) for _ in batch]

    with pytest.raises(gate.GateError, match="contamination"):
        gate.run_multi_lora_selection(
            base="base", adapters=("adapter_a", "adapter_b"),
            mixed_batch=base_only, solo=solo)


# --------------------------------------------------------------------------
# recorded-unavailable + the gate CLI on this CPU venv
# --------------------------------------------------------------------------


def test_record_unavailable_writes_the_marker(tmp_path):
    marker = gate.record_unavailable(tmp_path, "vllm bootstrap failed")
    doc = json.loads(marker.read_text())
    assert doc["status"] == "phase-0 unavailable"
    assert doc["reason"] == "vllm bootstrap failed"
    assert doc["gate_prompt_set_hash"] == gate.gate_prompt_set_hash()


def test_gate_cli_records_unavailable_without_vllm(tmp_path):
    """On this CPU venv the gate's bootstrap MUST fail (no vllm) — and
    that failure is a recorded skip (marker on disk) that exits
    EXIT_BOOTSTRAP, never 0: runs in this campaign reported COMPLETED
    0:0 while validating nothing, so 'could not run' must be visible
    off sacct alone. It also quarantines any certificates a previous
    run published (G3): an unavailable run must never leave a stale
    success readable."""
    store = gate.CertificateStore(tmp_path / "noop_certificates.json")
    cert = gate.certificate_for(_spec(), path="lens")
    store.record(cert, evidence={"prompts": 6, "all_exact": True})
    store.publish()
    assert store.certificates() == [cert]  # a previous success on disk
    manifest = ROOT / "configs" / "capeval" / "manifests" / "petri_twosi.yaml"
    rc = gate.main(["gate", "--manifest", str(manifest),
                    "--out-dir", str(tmp_path)])
    assert rc == gate.EXIT_BOOTSTRAP
    assert rc != 0  # a gate that validated NOTHING may never read green
    marker = tmp_path / gate.UNAVAILABLE_MARKER
    assert marker.is_file()
    assert "phase-0 unavailable" in marker.read_text()
    assert store.certificates() == []  # quarantined by the failed run


def test_gate_exit_codes_distinguish_could_not_run_from_disproved():
    """The sacct triage contract (the hardened lora smoke's, mirrored):
    0 = passed, 3 = could not run, 4 = ran and disproved — three
    DISTINCT values, never 0 for a non-success. A regression collapsing
    any pair re-creates the false-green class this pins against."""
    assert gate.EXIT_BOOTSTRAP == 3
    assert gate.EXIT_FAILED == 4
    assert len({0, gate.EXIT_BOOTSTRAP, gate.EXIT_FAILED}) == 3


def test_gate_cli_unavailable_never_returns_zero(tmp_path, monkeypatch):
    """The anti-false-green pin, independent of WHY the bootstrap
    failed: ANY GateBootstrapError path must exit EXIT_BOOTSTRAP —
    never 0 (jobs 232175/232236/233394's false-green class, and job
    243201's six-second crash, must all be triageable off sacct)."""
    def boom(args, spec):
        raise gate.GateBootstrapError("synthetic bootstrap failure")

    monkeypatch.setattr(gate, "_bootstrap_live_stack", boom)
    rc = gate.main(["gate", "--manifest", str(_frozen_manifest(tmp_path)),
                    "--out-dir", str(tmp_path)])
    assert rc == gate.EXIT_BOOTSTRAP
    assert rc != 0
    assert (tmp_path / gate.UNAVAILABLE_MARKER).is_file()
    assert not (tmp_path / gate.FAILURE_MARKER).exists()


def test_gate_cli_bootstrap_time_validation_failure_is_a_failure(
        tmp_path, monkeypatch):
    """A GateError raised DURING the live gathering (an engine-config
    assertion, the production install comparison, a readiness
    validator, the selection record contradicting routing) is 'ran and
    disproved': gate_failed.json + EXIT_FAILED — never the unavailable
    marker, which would mis-triage it as an environment problem to
    resubmit."""
    def disproved(args, spec):
        raise gate.GateError("synthetic: install reports do not match")

    monkeypatch.setattr(gate, "_bootstrap_live_stack", disproved)
    rc = gate.main(["gate", "--manifest", str(_frozen_manifest(tmp_path)),
                    "--out-dir", str(tmp_path)])
    assert rc == gate.EXIT_FAILED
    assert rc != 0
    assert (tmp_path / gate.FAILURE_MARKER).is_file()
    assert not (tmp_path / gate.UNAVAILABLE_MARKER).exists()


def test_gate_cli_missing_registry_is_recorded_unavailable(tmp_path,
                                                           monkeypatch):
    """On a node where vllm/vllm-lens import fine, a missing registry
    export is an environment problem: recorded unavailable +
    EXIT_BOOTSTRAP. (The live wiring is real now — nothing crashes with
    an unrecorded NotImplementedError, job 243201's failure mode.)"""
    from sisteer.capeval.serving import lens_steer as ls_mod
    from sisteer.capeval.serving import vllm_server as vs_mod

    monkeypatch.setattr(vs_mod, "require_vllm", lambda: None)
    monkeypatch.setattr(ls_mod, "require_vllm_lens", lambda: None)
    manifest = ROOT / "configs" / "capeval" / "manifests" / "petri_twosi.yaml"
    rc = gate.main(["gate", "--manifest", str(manifest),
                    "--out-dir", str(tmp_path),
                    "--registry-dir", str(tmp_path / "no_such_registry")])
    assert rc == gate.EXIT_BOOTSTRAP
    assert rc != 0
    assert (tmp_path / gate.UNAVAILABLE_MARKER).is_file()


def _frozen_manifest(tmp_path) -> Path:
    """The repo manifest with serving_impl_commit frozen — run_gate's
    certificate keying refuses UNFROZEN (its own test above), so the
    CLI-level success/failure tests need a freezable world."""
    text = (ROOT / "configs" / "capeval" / "manifests"
            / "petri_twosi.yaml").read_text().replace(
        "serving_impl_commit: UNFROZEN",
        f"serving_impl_commit: {IMPL_COMMIT}")
    manifest = tmp_path / "petri_twosi_frozen.yaml"
    manifest.write_text(text)
    return manifest


def _gate_inputs(*, phase0=None, reference=None, kind_serving=None,
                 noop_serving=None, multi_lora=None, stacked_identity=None,
                 fidelity_evidence=None):
    reference = reference or _reference
    return gate.GateInputs(
        phase0_evidence=phase0 if phase0 is not None else _good_evidence(),
        reference=reference,
        kind_serving=(kind_serving if kind_serving is not None
                      else {"obj_cast": {"lens": reference}}),
        noop_serving=(noop_serving if noop_serving is not None
                      else {"passthrough": reference, "lens": reference}),
        multi_lora=multi_lora,
        stacked_identity=stacked_identity,
        fidelity_evidence=fidelity_evidence)


def test_run_gate_stacked_identity_two_sided(tmp_path):
    """When stacked cells are served, the gate evaluates the
    TWO-SIDED identity evidence — a stack equal to either single dropped
    a leg and fails the whole gate (certificates stay quarantined); a
    genuinely two-sided stack passes and its diagnostics land in the
    record."""
    spec = _spec()
    good = {"cells": {"stacked__safety__psr_kl_all": {
        "stacked_ids": [[1, 2], [3, 4]],
        "object_only_ids": [[1, 9], [3, 4]],
        "prompt_only_ids": [[1, 2], [8, 8]]}}}
    store = gate.CertificateStore(tmp_path / "certs.json")
    result = gate.run_gate(spec, store, _gate_inputs(stacked_identity=good))
    assert result.stacked_identity is not None
    record = json.loads(gate.write_gate_record(tmp_path, result).read_text())
    cell = record["stacked_identity"]["stacked__safety__psr_kl_all"]
    assert cell["object_only"]["n_differing"] == 1
    assert cell["prompt_only"]["n_differing"] == 1

    dropped = {"cells": {"stacked__safety__psr_kl_all": {
        "stacked_ids": [[1, 2], [3, 4]],
        "object_only_ids": [[1, 2], [3, 4]],  # == stack: prompt leg dropped
        "prompt_only_ids": [[9, 9], [8, 8]]}}}
    store2 = gate.CertificateStore(tmp_path / "certs2.json")
    with pytest.raises(gate.GateError, match="stacked two-sided identity"):
        gate.run_gate(spec, store2, _gate_inputs(stacked_identity=dropped))
    assert store2.certificates() == []  # quarantined, nothing published


def test_run_gate_without_stacked_cells_records_none(tmp_path):
    spec = _spec()
    store = gate.CertificateStore(tmp_path / "certs.json")
    result = gate.run_gate(spec, store, _gate_inputs())
    assert result.stacked_identity is None
    record = json.loads(gate.write_gate_record(tmp_path, result).read_text())
    assert record["stacked_identity"] is None


def _fidelity_seqs(n=4, length=6, base=0):
    return [[base + 100 * i + j for j in range(length)] for i in range(n)]


def _good_fidelity_evidence():
    """A minimal passing five-check fidelity evidence set (same-engine)
    — the shape _bootstrap_live_stack gathers through the live proxy."""
    return {
        "engine_determinism": {"run1_ids": _fidelity_seqs(),
                               "run2_ids": _fidelity_seqs()},
        "scale0_noop": {"scale0_ids": _fidelity_seqs(),
                        "unsteered_ids": _fidelity_seqs()},
        "steering_bites": {"arms": {
            "si_safety__psr_kl_all": {"steered_ids": _fidelity_seqs(base=9),
                                      "scale0_ids": _fidelity_seqs()}}},
        "batch_invariance": {
            "steered_reference_ids": _fidelity_seqs(base=9),
            "steered_batched_ids": _fidelity_seqs(base=9),
            "unsteered_reference_ids": _fidelity_seqs(),
            "unsteered_batched_ids": _fidelity_seqs()},
        "selection_honored": {"requests": [
            {"arm": None, "prompt_index": 0,
             "reference_ids": [1, 2, 3], "batched_ids": [1, 2, 3]},
            {"arm": "si_safety__psr_kl_all", "prompt_index": 0,
             "reference_ids": [4, 5, 6], "batched_ids": [4, 5, 6]}]},
    }


def test_run_gate_live_fidelity_gate_catches_a_silent_arm(tmp_path):
    """The live fidelity gate is evaluated by run_gate — a served
    arm that produces NO difference vs its scale-0 twin (hooks installed
    but not biting, the failure the launch attestation cannot see) fails
    the gate; certificates stay quarantined."""
    spec = _spec()
    good = _good_fidelity_evidence()
    store = gate.CertificateStore(tmp_path / "certs.json")
    result = gate.run_gate(spec, store, _gate_inputs(fidelity_evidence=good))
    assert result.fidelity is not None and result.fidelity["passed"]
    record = json.loads(gate.write_gate_record(tmp_path, result).read_text())
    assert record["fidelity"]["passed"] is True

    silent = _good_fidelity_evidence()
    # the arm at serving scale now equals its scale-0 output: not biting
    silent["steering_bites"]["arms"]["si_safety__psr_kl_all"][
        "steered_ids"] = _fidelity_seqs()
    store2 = gate.CertificateStore(tmp_path / "certs2.json")
    with pytest.raises(gate.GateError, match="fidelity gate FAILED"):
        gate.run_gate(spec, store2, _gate_inputs(fidelity_evidence=silent))
    assert store2.certificates() == []


def test_run_gate_full_success_publishes_and_reports(tmp_path):
    """The CPU-testable orchestration (G1): Phase-0 -> per-kind canaries
    -> the two mandatory no-op certificates -> publish. Certificates are
    readable only after the WHOLE gate succeeded."""
    spec = _spec()
    store = gate.CertificateStore(tmp_path / "certs.json")
    result = gate.run_gate(spec, store, _gate_inputs())
    assert result.phase0.passed == ls.PHASE0_ASSERTIONS
    assert result.kind_outcomes["obj_cast"].passed
    assert {c.path for c in result.certificates} == {"passthrough", "lens"}
    assert {c.path for c in store.certificates()} == {"passthrough", "lens"}
    assert result.multi_lora_checked is False
    record = gate.write_gate_record(tmp_path, result)
    doc = json.loads(record.read_text())
    assert doc["status"] == "completed"
    assert doc["kinds"]["obj_cast"]["passed_paths"] == ["lens"]
    # the recorded-exception concept is GONE
    assert "recorded_exception" not in doc["kinds"]["obj_cast"]


def test_run_gate_kind_failing_every_path_is_a_hard_failure(tmp_path):
    """A kind failing on EVERY committed vLLM path is a HARD gate
    failure that blocks the audit — never a recorded exception routing
    the cell to the HF shim. The failure names the kind, and no
    certificate survives (quarantined)."""
    spec = _spec()
    store = gate.CertificateStore(tmp_path / "certs.json")
    old = gate.certificate_for(spec, path="lens")
    store.record(old, evidence={"prompts": 6, "all_exact": True})
    store.publish()
    with pytest.raises(gate.GateError,
                       match="obj_cast.*HARD gate failure"):
        gate.run_gate(spec, store, _gate_inputs(
            kind_serving={"obj_cast": {"lens": _served_wrong,
                                       "lora": _served_wrong}}))
    assert store.certificates() == []  # quarantined, nothing published


def test_gate_cli_unservable_kind_is_nonzero_and_recorded(tmp_path,
                                                          monkeypatch):
    """CLI-level: the unservable-kind failure exits NONZERO with
    gate_failed.json — never exit 0, never an unavailable marker."""
    monkeypatch.setattr(
        gate, "_bootstrap_live_stack",
        lambda args, spec: _gate_inputs(
            kind_serving={"obj_cast": {"lens": _served_wrong,
                                       "lora": _served_wrong}}))
    rc = gate.main(["gate", "--manifest", str(_frozen_manifest(tmp_path)),
                    "--out-dir", str(tmp_path)])
    assert rc == gate.EXIT_FAILED
    assert rc != 0
    marker = tmp_path / gate.FAILURE_MARKER
    assert marker.is_file()
    assert "HARD gate failure" in marker.read_text()
    assert not (tmp_path / gate.UNAVAILABLE_MARKER).exists()


def test_run_gate_phase0_failure_keeps_certificates_quarantined(tmp_path):
    spec = _spec()
    store = gate.CertificateStore(tmp_path / "certs.json")
    old = gate.certificate_for(spec, path="lens")
    store.record(old, evidence={"prompts": 6, "all_exact": True})
    store.publish()
    bad = _good_evidence()
    bad["scale0_noop"]["scale0_tokens"] = [9, 8]
    with pytest.raises(ls.Phase0Failure):
        gate.run_gate(spec, store, _gate_inputs(phase0=bad))
    assert store.certificates() == []  # nothing usable survives a failure


def test_run_gate_requires_the_separate_noop_legs_and_lora_when_live(
        tmp_path):
    spec = _spec()
    store = gate.CertificateStore(tmp_path / "certs.json")
    with pytest.raises(gate.GateError, match="lens"):
        gate.run_gate(spec, store, _gate_inputs(
            noop_serving={"passthrough": _reference}))
    table, solo, mixed = _lora_world()
    probe = gate.MultiLoraProbe(base="base",
                                adapters=("adapter_a", "adapter_b"),
                                mixed_batch=mixed, solo=solo)
    with pytest.raises(gate.GateError, match="lora"):
        gate.run_gate(spec, store, _gate_inputs(multi_lora=probe))
    result = gate.run_gate(spec, store, _gate_inputs(
        multi_lora=probe,
        noop_serving={"passthrough": _reference, "lens": _reference,
                      "lora": _reference}))
    assert result.multi_lora_checked is True
    assert {c.path for c in store.certificates()} == {"passthrough", "lens",
                                                      "lora"}


def test_gate_cli_validation_failure_is_nonzero_and_recorded(tmp_path,
                                                             monkeypatch):
    """G2: a post-bootstrap VALIDATION failure (here: Phase-0) exits
    NONZERO with gate_failed.json — never 'bootstrap unavailable' +
    exit 0."""
    bad = _good_evidence()
    bad["scale0_noop"]["scale0_tokens"] = [9, 8]
    monkeypatch.setattr(gate, "_bootstrap_live_stack",
                        lambda args, spec: _gate_inputs(phase0=bad))
    rc = gate.main(["gate", "--manifest", str(_frozen_manifest(tmp_path)),
                    "--out-dir", str(tmp_path)])
    assert rc == gate.EXIT_FAILED
    assert rc != 0
    marker = tmp_path / gate.FAILURE_MARKER
    assert marker.is_file()
    assert "gate FAILED" in marker.read_text()
    assert not (tmp_path / gate.UNAVAILABLE_MARKER).exists()
    store = gate.CertificateStore(tmp_path / "noop_certificates.json")
    assert store.certificates() == []


def test_gate_cli_success_publishes_and_writes_the_record(tmp_path,
                                                          monkeypatch):
    monkeypatch.setattr(gate, "_bootstrap_live_stack",
                        lambda args, spec: _gate_inputs())
    rc = gate.main(["gate", "--manifest", str(_frozen_manifest(tmp_path)),
                    "--out-dir", str(tmp_path)])
    assert rc == 0
    doc = json.loads((tmp_path / gate.GATE_RECORD).read_text())
    assert doc["status"] == "completed"
    store = gate.CertificateStore(tmp_path / "noop_certificates.json")
    assert {c.path for c in store.certificates()} == {"passthrough", "lens"}


# --------------------------------------------------------------------------
# the live evidence gathering's PURE helpers (_bootstrap_live_stack's
# CPU-testable pieces — the GPU children only feed these)
# --------------------------------------------------------------------------


def test_served_token_ids_extracts_raw_ids_and_refuses_absence():
    resp = {"choices": [{"token_ids": [1, 2, 3],
                         "message": {"content": "x"}}]}
    assert gate.served_token_ids(resp) == [1, 2, 3]
    with pytest.raises(gate.GateError, match="no choices"):
        gate.served_token_ids({"choices": []})
    # a server that ignored return_token_ids can never be
    # exact-compared: text is NEVER re-tokenized (H5)
    with pytest.raises(gate.GateError, match="token_ids"):
        gate.served_token_ids({"choices": [{"message": {"content": "x"}}]})
    with pytest.raises(gate.GateError, match="non-integer"):
        gate.served_token_ids({"choices": [{"token_ids": [1, "2"]}]})


def _zero_entry(cell_id: str) -> dict:
    return {"cell_id": cell_id, "kind": la.ZERO_KIND,
            "resolved_spec": {"kind": la.ZERO_KIND, "hidden_size": 8,
                              "layer": 0},
            "expect_hash": cm.sha256_hex16(cell_id.encode()),
            "artifact": None, "sidecar": None, "artifact_sha16": None,
            "artifact_intervention_hash": None, "forced_scale": None,
            "donor_cell": None, "validated_by": None}


def _write_registry(path: Path, entries: list, header=(1, 2)) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    (path / "registry.json").write_text(json.dumps(
        {"version": 1, "assistant_header_ids": list(header),
         "entries": entries}))
    return path


def test_subset_registry_with_twins_loads_and_selects(tmp_path):
    """The derived children's registry: production entries copied
    verbatim + a donor-backed scale-0 twin per cell, loadable by the
    COMMITTED WorkerObjectRegistry.load (all hash/donor verification)
    with the twin selecting at scale 0.0 and the arm at 1.0."""
    src = _write_registry(tmp_path / "src", [
        _zero_entry("si_a__psr_kl_all"), _zero_entry("si_b__psr_kl_all")])
    out = tmp_path / "derived"
    selectors = gate.subset_registry_with_twins(
        src, ["si_a__psr_kl_all"], out)
    assert set(selectors) == {"si_a__psr_kl_all"}
    reg = ls.WorkerObjectRegistry.load(out)
    assert reg.cell_ids() == ("si_a__psr_kl_all",
                              "si_a__psr_kl_all__scale0")
    sel = selectors["si_a__psr_kl_all"]
    arm = reg.select(**sel["selector"])
    assert arm.effective_scale == 1.0
    twin = reg.select(sel["twin_selector"]["petri_object"],
                      sel["twin_selector"]["expect_hash"], scale=0.0)
    assert twin.effective_scale == 0.0
    # untouched entries do not travel: the derived registry is scoped
    assert "si_b__psr_kl_all" not in reg.cell_ids()
    # the registry-wide header travels (response-mode cells need it)
    assert reg.assistant_header_ids == (1, 2)


def test_subset_registry_copies_artifact_bytes_verbatim(tmp_path):
    blob = b"synthetic-object-bytes"
    spec = {"kind": "vector", "layer": 3}
    e = _zero_entry("si_c__vector_all")
    e.update(kind="vector", resolved_spec=spec,
             artifact="si_c__vector_all/artifact.bin",
             artifact_sha16=cm.sha256_hex16(blob),
             artifact_intervention_hash=cm.petri_intervention_hash(
                 spec, artifact_bytes=blob, sidecar_bytes=b""))
    e["expect_hash"] = e["artifact_intervention_hash"]
    src = _write_registry(tmp_path / "src", [e])
    (src / "si_c__vector_all").mkdir()
    (src / "si_c__vector_all" / "artifact.bin").write_bytes(blob)
    out = tmp_path / "derived"
    selectors = gate.subset_registry_with_twins(src, ["si_c__vector_all"],
                                                out)
    assert (out / "si_c__vector_all" / "artifact.bin").read_bytes() == blob
    twin_hash = cm.sha256_hex16(
        f"{e['expect_hash']}:scale0".encode("utf-8"))
    assert selectors["si_c__vector_all"]["twin_selector"] == {
        "petri_object": "si_c__vector_all__scale0",
        "expect_hash": twin_hash, "scale": 0.0}


def test_subset_registry_refuses_missing_and_control_cells(tmp_path):
    donor = _zero_entry("si_a__psr_kl_all")
    noop = dict(donor, cell_id="none__lens0", forced_scale=0.0,
                donor_cell="si_a__psr_kl_all")
    src = _write_registry(tmp_path / "src", [donor, noop])
    with pytest.raises(gate.GateError, match="not in the worker registry"):
        gate.subset_registry_with_twins(src, ["nope"], tmp_path / "d1")
    with pytest.raises(gate.GateError, match="not a canonical arm"):
        gate.subset_registry_with_twins(src, ["none__lens0"],
                                        tmp_path / "d2")


def test_subset_registry_for_serving_takes_the_donor_closure(tmp_path):
    """Job 243841: launch_server re-adapts EVERY export entry at
    readiness, so one adapterless kind (lofit/token) blocks the whole
    launch. The serving subset is the VERBATIM closure: the cells under
    test + the none__lens0 leg + its donor — unadaptable kinds excluded,
    entries byte-identical to the export's."""
    donor = _zero_entry("alphasteer")
    psr_a, psr_b = _zero_entry("si_a__psr_kl_all"), _zero_entry("si_b__psr_kl_all")
    lofit = dict(_zero_entry("si_a__lofit"), kind="lofit")
    noop = dict(donor, cell_id="none__lens0", forced_scale=0.0,
                donor_cell="alphasteer",
                expect_hash=cm.sha256_hex16(b"none"))
    src = _write_registry(tmp_path / "src", [psr_a, psr_b, lofit, donor,
                                             noop])
    out = tmp_path / "serving"
    info = gate.subset_registry_for_serving(src, ["si_a__psr_kl_all"], out)
    assert info["included"] == ["alphasteer", "none__lens0",
                                "si_a__psr_kl_all"]
    assert "si_a__lofit" in info["excluded"]
    reg = ls.WorkerObjectRegistry.load(out)  # the committed verification
    assert reg.cell_ids() == ("alphasteer", "none__lens0",
                              "si_a__psr_kl_all")
    got = {e["cell_id"]: e for e in json.loads(
        (out / "registry.json").read_text())["entries"]}
    assert got["si_a__psr_kl_all"] == psr_a  # VERBATIM, no synthesis
    assert got["none__lens0"] == noop
    # a leg whose donor is missing from the export refuses loudly
    src2 = _write_registry(tmp_path / "src2", [psr_a, noop])
    with pytest.raises(gate.GateError, match="corrupt registry"):
        gate.subset_registry_for_serving(src2, ["si_a__psr_kl_all"],
                                         tmp_path / "s2")


def test_reference_oracle_dispatches_per_leg_and_refuses_bare_calls():
    msgs = [dict(m) for m in gate.GATE_PROMPT_SET[0]]
    key = gate.ReferenceOracle.key(msgs)
    oracle = gate.ReferenceOracle(
        base={key: [1, 2]},
        per_cell={"si_a__psr_kl_all": {key: [3, 4]}})
    with pytest.raises(gate.GateError, match="before any served leg"):
        oracle(msgs)
    oracle.set_current(None)
    assert oracle(msgs) == [1, 2]
    oracle.set_current("si_a__psr_kl_all")
    assert oracle(msgs) == [3, 4]
    with pytest.raises(gate.GateError, match="no armed fp32 reference"):
        oracle.set_current("si_unknown")
        oracle(msgs)
    oracle.set_current(None)
    with pytest.raises(gate.GateError, match="this prompt"):
        oracle([{"role": "user", "content": "not a gate prompt"}])


def test_reference_oracle_backs_a_real_canary_run():
    """The exact coupling the bootstrap relies on: each SERVED closure
    sets the dispatch pointer before returning, and run_kind_canary /
    run_noop_certificate evaluate served(messages) strictly before
    reference(messages) — so one oracle serves base and armed
    comparisons without cross-talk."""
    keys = [gate.ReferenceOracle.key([dict(m) for m in msgs])
            for msgs in gate.GATE_PROMPT_SET]
    base = {k: [10 + i] for i, k in enumerate(keys)}
    armed = {k: [90 + i] for i, k in enumerate(keys)}
    oracle = gate.ReferenceOracle(base, {"cell_a": armed})

    def served_arm(messages):
        oracle.set_current("cell_a")
        return list(armed[gate.ReferenceOracle.key(messages)])

    def served_base(messages):
        oracle.set_current(None)
        return list(base[gate.ReferenceOracle.key(messages)])

    outcome = gate.run_kind_canary("cell_a",
                                   served_by_path={"lens": served_arm},
                                   reference=oracle)
    assert outcome.passed and outcome.passed_paths == ("lens",)
    spec = _spec()
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        store = gate.CertificateStore(Path(td) / "c.json")
        cert = gate.run_noop_certificate(spec, store, path="lens",
                                         served=served_base,
                                         reference=oracle)
        assert cert.path == "lens"


def test_gate_launch_flags_drops_only_the_bare_lora_restatement():
    """Job 243728: the frozen manifest restates --enable-lora, the gate
    registers no native adapters (every lora-kind cell serves via the
    lens o_proj site), and the composer refuses the restatement without
    adapters. The gate drops EXACTLY that bare token, records it, and
    touches nothing else; with adapters registered the spec passes
    through untouched."""
    spec = _spec()
    spec.serving_launch_flags = ["--enable-lora", "--enforce-eager"]
    adjusted, dropped = gate.gate_launch_flags(spec, lora_enabled=False)
    assert dropped == ["--enable-lora"]
    assert adjusted.serving_launch_flags == ["--enforce-eager"]
    assert spec.serving_launch_flags == ["--enable-lora",
                                         "--enforce-eager"]  # untouched
    same, dropped = gate.gate_launch_flags(spec, lora_enabled=True)
    assert dropped == [] and same is spec
    # any OTHER wrapper-owned flag still reaches the composer's refusal
    spec2 = _spec()
    spec2.serving_launch_flags = ["--no-enable-prefix-caching"]
    kept, dropped = gate.gate_launch_flags(spec2, lora_enabled=False)
    assert dropped == []
    assert kept.serving_launch_flags == ["--no-enable-prefix-caching"]


def _margins_response(ids, margins):
    content = [{"token": "x", "logprob": 0.0,
                "top_logprobs": [{"token": "a", "logprob": 0.0},
                                 {"token": "b", "logprob": -m}]}
               for m in margins]
    return {"choices": [{"token_ids": list(ids),
                         "logprobs": {"content": content}}]}


def test_served_token_margins_extracts_and_refuses_absence():
    resp = _margins_response([1, 2], [0.5, 0.01])
    assert gate.served_token_margins(resp) == pytest.approx([0.5, 0.01])
    with pytest.raises(gate.GateError, match="logprobs.content"):
        gate.served_token_margins({"choices": [{"token_ids": [1]}]})
    bad = _margins_response([1], [0.5])
    bad["choices"][0]["logprobs"]["content"][0]["top_logprobs"] = []
    with pytest.raises(gate.GateError, match="top_logprobs"):
        gate.served_token_margins(bad)


def test_neartie_tolerant_equality_is_the_d39_contract():
    """A cross-engine divergence is tolerated ONLY at a served top-2
    margin <= the stated bound, at the FIRST divergence, with the
    comparison stopping at the fork; a decisive-margin divergence still
    fails, and exactness still passes."""
    kw = dict(prompt_index=0, margin_atol=0.1)
    # exact: identical to the exact comparator's verdict
    r = gate.neartie_tolerant_equality([1, 2, 3], [1, 2, 3],
                                       served_margins=[9, 9, 9], **kw)
    assert r.passed and r.first_divergence is None
    # a near-tie fork: tolerated, suffix (and length) beyond it ignored
    r = gate.neartie_tolerant_equality([1, 2, 9, 9, 9], [1, 2, 7, 7],
                                       served_margins=[9, 9, 0.02, 9, 9],
                                       **kw)
    assert r.passed and r.first_divergence == 2
    assert r.diagnostics["tolerated_neartie"] is True
    assert r.diagnostics["served_top2_margin_nats_at_divergence"] == 0.02
    # a DECISIVE divergence: fails, margin on the record
    r = gate.neartie_tolerant_equality([1, 2, 9], [1, 2, 7],
                                       served_margins=[9, 9, 1.4], **kw)
    assert not r.passed
    assert r.diagnostics["served_top2_margin_nats_at_divergence"] == 1.4
    # margin unavailable at the divergence (served ended early): fails
    r = gate.neartie_tolerant_equality([1, 2], [1, 2, 7],
                                       served_margins=[9, 9], **kw)
    assert not r.passed
    assert r.diagnostics["served_top2_margin_nats_at_divergence"] is None


def test_run_gate_threads_the_cross_engine_comparator(tmp_path):
    """The comparator seam end to end: with cross_engine_comparator=None
    the near-tie world FAILS the no-op certificate (the exact contract,
    untouched for CPU fakes); with the tolerant comparator it passes,
    publishes, and the canary/no-op verdicts carry the same criterion.
    The bound STILL fails a decisive divergence."""
    import functools
    spec = _spec()

    def served_neartie(messages):
        ids = list(_reference(messages))
        ids[0] += 1  # the token-0 flip, job 243998's prompt-5 shape
        return ids

    tolerant = functools.partial(
        gate.neartie_tolerant_equality,
        served_margins=[0.01] * 64, margin_atol=0.1)
    inputs_exact = _gate_inputs(
        noop_serving={"passthrough": _reference, "lens": served_neartie})
    store = gate.CertificateStore(tmp_path / "c1.json")
    with pytest.raises(gate.GateError, match="no-op certificate FAILED"):
        gate.run_gate(spec, store, inputs_exact)
    inputs_tolerant = gate.GateInputs(
        phase0_evidence=_good_evidence(), reference=_reference,
        kind_serving={"obj_cast": {"lens": served_neartie}},
        noop_serving={"passthrough": _reference, "lens": served_neartie},
        cross_engine_comparator=tolerant)
    store2 = gate.CertificateStore(tmp_path / "c2.json")
    result = gate.run_gate(spec, store2, inputs_tolerant)
    assert result.kind_outcomes["obj_cast"].passed
    assert {c.path for c in store2.certificates()} == {"passthrough",
                                                       "lens"}
    # decisive divergence still fails under the tolerant comparator
    decisive = functools.partial(
        gate.neartie_tolerant_equality,
        served_margins=[5.0] * 64, margin_atol=0.1)
    store3 = gate.CertificateStore(tmp_path / "c3.json")
    with pytest.raises(gate.GateError):
        gate.run_gate(spec, store3, gate.GateInputs(
            phase0_evidence=_good_evidence(), reference=_reference,
            kind_serving={"obj_cast": {"lens": served_neartie}},
            noop_serving={"passthrough": _reference,
                          "lens": served_neartie},
            cross_engine_comparator=decisive))
    assert store3.certificates() == []


def test_classify_vllm_server_error_splits_validation_from_bootstrap():
    validation = [
        "steering install reports do not match the registry's expected "
        "state — ...",
        "resolved scheduler state does not match the composed "
        "no-chunking invariant — ...",
    ]
    bootstrap = [
        "vllm serve exited with 1 before becoming ready — see log",
        "something is already listening on 127.0.0.1:8921",
        "'vllm' not on PATH — activate the pinned serving venv",
        "vllm serve did not become ready within 900.0s",
    ]
    for text in validation:
        assert gate.classify_vllm_server_error(
            vs.VllmServerError(text)) == "validation"
    for text in bootstrap:
        assert gate.classify_vllm_server_error(
            vs.VllmServerError(text)) == "bootstrap"


def test_max_concurrency_problem_reads_back_the_engine_line():
    good = ("INFO ... GPU KV cache size: 574,064 tokens\n"
            "INFO ... Maximum concurrency for 16,384 tokens per request: "
            "35.04x\n")
    lines, problem = gate.max_concurrency_problem(good, max_num_seqs=24)
    assert problem is None and len(lines) == 1
    lines, problem = gate.max_concurrency_problem(good, max_num_seqs=48)
    assert problem is not None and "does NOT back" in problem
    lines, problem = gate.max_concurrency_problem("no such line",
                                                  max_num_seqs=24)
    assert problem is not None and "cannot be read back" in problem


def _phase0_children_fixture():
    """Synthetic child outputs shaped exactly like the gatherers'."""
    base_tokens = [5, 6, 7]
    lens_child = {
        "packages": {"vllm": "0.25.1+cu129", "vllm-lens": "1.2.1",
                     "torch": "2.11.0+cu129"},
        "plugin": {"generate_patched": True, "engine_args_patched": True},
        "engine": {"enforce_eager": True, "prefix_caching_enabled": False,
                   "worker_extension_cls": "pkg.SisteerWorkerExtension"},
        "runs": {
            "baseline": {"tokens": base_tokens},
            "steered": {"tokens": [9, 9, 9]},
            "steered_alt_solo": {"tokens": [4, 4, 4]},
            "mixed_unhooked": {"tokens": base_tokens},
            "mixed_hooked": {"tokens": [4, 4, 4]},
        },
        "saved_sentinel_evidence": {"prefill_fires": 1, "decode_fires": 3,
                                    "generated_tokens": 3,
                                    "decode_step_fires": [1, 1, 1]},
        "returned_modification": {"max_abs_delta": 0.01,
                                  "stated_atol": 0.25},
        "phase0_facts": {"registration_count": 1,
                         "per_request_registrations": 0},
        "n_selector_requests": 3,
    }
    disabled_child = {
        "saved_count": 0,
        "runs": {"steer_attempt": {"tokens": base_tokens}},
    }
    proxy_ev = {
        "first_run_tokens": [1, 2, 3],
        "second_run_tokens": [1, 2, 3],
        "lens_salts_unique": True,
        "arm_tokens": [9, 8, 7],
        "twin_cold_tokens": [1, 1, 1],
        "twin_warm_tokens": [1, 1, 1],
        "proxy_base_tokens": [[1, 1, 1], [2, 2, 2]],
        "scale0_tokens": [[1, 1, 1], [2, 2, 2]],
    }
    return lens_child, disabled_child, proxy_ev


def test_assemble_gate_phase0_passes_run_phase0_on_good_children():
    lens_child, disabled_child, proxy_ev = _phase0_children_fixture()
    expected = dict(lens_child["packages"])
    ev = gate.assemble_gate_phase0(
        expected_packages=expected, lens_child=lens_child,
        disabled_child=disabled_child, proxy_ev=proxy_ev,
        prefix_caching_enabled=True,
        worker_extension_cls="pkg.SisteerWorkerExtension")
    report = ls.run_phase0(ev)
    assert report.passed == ls.PHASE0_ASSERTIONS


def test_assemble_gate_phase0_omits_missing_pieces_so_checkers_fail():
    """A gatherer that produced nothing must FAIL BY NAME downstream
    (omitted keys, never defaults) — the run_phase0 failure text names
    every starved assertion."""
    lens_child, disabled_child, proxy_ev = _phase0_children_fixture()
    lens_child = dict(lens_child, runs={}, saved_sentinel_evidence=None,
                      returned_modification=None, phase0_facts={},
                      n_selector_requests=None)
    ev = gate.assemble_gate_phase0(
        expected_packages=lens_child["packages"], lens_child=lens_child,
        disabled_child={}, proxy_ev={},
        prefix_caching_enabled=None,
        worker_extension_cls="pkg.SisteerWorkerExtension")
    with pytest.raises(ls.Phase0Failure) as exc:
        ls.run_phase0(ev)
    text = str(exc.value)
    for name in ("saved_sentinel", "mixed_batch_isolation",
                 "returned_modification", "disable_flag",
                 "single_persistent_hook", "prefix_cache_isolation",
                 "scale0_noop"):
        assert name in text


def test_assemble_gate_phase0_flags_a_leaky_warm_twin():
    """The isolation proof end to end through the assembly: a warm
    twin that moved toward the arm (salt isolation NOT holding) fails
    assertion 7."""
    lens_child, disabled_child, proxy_ev = _phase0_children_fixture()
    proxy_ev = dict(proxy_ev, twin_warm_tokens=proxy_ev["arm_tokens"])
    ev = gate.assemble_gate_phase0(
        expected_packages=lens_child["packages"], lens_child=lens_child,
        disabled_child=disabled_child, proxy_ev=proxy_ev,
        prefix_caching_enabled=True,
        worker_extension_cls="pkg.SisteerWorkerExtension")
    with pytest.raises(ls.Phase0Failure, match="cache_salt isolation is "
                                               "NOT holding"):
        ls.run_phase0(ev)


def test_smoke_cli_refuses_without_the_live_stack(tmp_path, monkeypatch):
    from sisteer.capeval.serving.proxy import PROXY_BASE_URL_ENV

    manifest = ROOT / "configs" / "capeval" / "manifests" / "petri_twosi.yaml"
    monkeypatch.delenv(PROXY_BASE_URL_ENV, raising=False)
    with pytest.raises(gate.GateError, match=PROXY_BASE_URL_ENV):
        gate.main(["smoke", "--manifest", str(manifest)])
    monkeypatch.setenv(PROXY_BASE_URL_ENV, "http://localhost:8814")
    try:
        import inspect_ai  # noqa: F401
        pytest.skip("inspect_ai installed here; the guard is for the CPU "
                    "venv")
    except ImportError:
        pass
    with pytest.raises(gate.GateError, match="petri group"):
        gate.main(["smoke", "--manifest", str(manifest)])


# --------------------------------------------------------------------------
# the live smoke's four pass criteria
# --------------------------------------------------------------------------


def _smoke_events(**drop):
    events = [
        {"type": "assistant", "content": "let me check"},
        {"type": "tool_call", "name": "run_shell",
         "arguments": {"cmd": "ls"}},
        {"type": "tool_result", "content": "a.txt"},
        {"type": "assistant", "content": "there is one file, a.txt"},
        {"type": "judge", "scores": {"self_preservation": 2.0,
                                     "eval_awareness": 3.0,
                                     "scenario_realism": 8.0}},
    ]
    return [e for e in events if e["type"] not in drop]


def test_smoke_transcript_all_four_criteria():
    res = gate.evaluate_smoke_transcript(_smoke_events())
    assert res.passed and all(res.criteria.values())


@pytest.mark.parametrize("drop,failed", [
    ("tool_call", ["tool_call_emitted", "tool_response_consumed",
                   "assistant_resumed"]),
    ("tool_result", ["tool_response_consumed", "assistant_resumed"]),
    ("judge", ["judge_parsed"]),
])
def test_smoke_transcript_failures(drop, failed):
    res = gate.evaluate_smoke_transcript(_smoke_events(**{drop: True}))
    assert not res.passed
    assert [k for k, v in res.criteria.items() if not v] == failed


def test_smoke_malformed_tool_call_or_judge_do_not_count():
    events = [
        {"type": "tool_call", "name": "", "arguments": {}},  # malformed
        {"type": "tool_call", "name": "x", "arguments": "not-a-mapping"},
        {"type": "judge", "scores": {}},  # empty scores parse as failure
        {"type": "judge", "scores": {"dim": "high"}},  # non-numeric
    ]
    res = gate.evaluate_smoke_transcript(events)
    assert not res.passed
    assert not res.criteria["tool_call_emitted"]
    assert not res.criteria["judge_parsed"]


def test_smoke_resume_must_follow_the_tool_result():
    # an assistant turn BEFORE the tool result is not a resumed turn
    events = [
        {"type": "tool_call", "name": "x", "arguments": {}},
        {"type": "assistant", "content": "premature"},
        {"type": "tool_result", "content": "y"},
        {"type": "judge", "scores": {"d": 1}},
    ]
    res = gate.evaluate_smoke_transcript(events)
    assert not res.criteria["assistant_resumed"]


# --------------------------------------------------------------------------
# the 8 Phase-0 assertions (logic; evidence is synthetic)
# --------------------------------------------------------------------------


def _good_evidence():
    return {
        "server_provenance": {
            "expected_packages": {"vllm": "0.25.1+cu129",
                                  "vllm-lens": "1.2.1"},
            "reported_packages": {"vllm": "0.25.1+cu129",
                                  "vllm-lens": "1.2.1"},
            "plugin_registered": True, "enforce_eager": True},
        "saved_sentinel": {"prefill_fires": 1, "decode_fires": 32,
                           "generated_tokens": 32,
                           "decode_step_fires": [1] * 32},
        "mixed_batch_isolation": {
            "unhooked_in_batch_tokens": [1, 2], "unhooked_solo_tokens": [1, 2],
            "hooked_in_batch_tokens": [3, 4], "hooked_solo_tokens": [3, 4]},
        "returned_modification": {"max_abs_delta": 1e-6,
                                  "stated_atol": 1e-4},
        "disable_flag": {"saved_count_with_disable": 0,
                         "tokens_with_disable": [1, 2],
                         "baseline_tokens": [1, 2]},
        "single_persistent_hook": {"registration_count": 1,
                                   "per_request_registrations": 0,
                                   "requests_observed": 12},
        "prefix_cache_isolation": {"prefix_caching_enabled": False,
                                   "first_run_tokens": [5, 6],
                                   "second_run_tokens": [5, 6]},
        "scale0_noop": {"scale0_tokens": [9, 9], "baseline_tokens": [9, 9]},
    }


def test_phase0_all_eight_pass_and_tp_slot_recorded():
    report = ls.run_phase0(_good_evidence())
    assert report.passed == ls.PHASE0_ASSERTIONS
    assert len(report.passed) == 8
    assert report.tp_note == ls.TP_SLOT_NOTE == "TP=1 only"


@pytest.mark.parametrize("name,mutation", [
    ("server_provenance",
     lambda ev: ev["server_provenance"].update(plugin_registered=False)),
    ("server_provenance",
     lambda ev: ev["server_provenance"]["reported_packages"].update(
         {"vllm-lens": "0.19.0"})),
    ("saved_sentinel",
     lambda ev: ev["saved_sentinel"].update(decode_fires=31)),
    ("mixed_batch_isolation",
     lambda ev: ev["mixed_batch_isolation"].update(
         unhooked_in_batch_tokens=[1, 99])),
    ("returned_modification",
     lambda ev: ev["returned_modification"].update(max_abs_delta=1.0)),
    ("returned_modification",
     lambda ev: ev["returned_modification"].update(stated_atol=None)),
    ("disable_flag",
     lambda ev: ev["disable_flag"].update(saved_count_with_disable=3)),
    ("single_persistent_hook",
     lambda ev: ev["single_persistent_hook"].update(
         per_request_registrations=5)),
    ("prefix_cache_isolation",
     lambda ev: ev["prefix_cache_isolation"].update(
         prefix_caching_enabled=True)),
    ("scale0_noop",
     lambda ev: ev["scale0_noop"].update(scale0_tokens=[9, 8])),
])
def test_phase0_each_assertion_fails_loudly(name, mutation):
    ev = _good_evidence()
    mutation(ev)
    with pytest.raises(ls.Phase0Failure, match=name):
        ls.run_phase0(ev)


@pytest.mark.parametrize("name,mutation,why", [
    # G8: vacuous evidence — zero counts / empty outputs prove nothing
    ("saved_sentinel",
     lambda ev: ev["saved_sentinel"].update(
         prefill_fires=1, decode_fires=0, generated_tokens=0,
         decode_step_fires=[]),
     "generated 0 tokens: 0 == 0 decode fires is vacuous"),
    ("saved_sentinel",
     lambda ev: ev["saved_sentinel"].update(
         decode_step_fires=[2] + [1] * 30 + [0]),
     "a double-fired step masking a skipped one — per-step exactly once"),
    ("mixed_batch_isolation",
     lambda ev: ev["mixed_batch_isolation"].update(
         unhooked_in_batch_tokens=[], unhooked_solo_tokens=[]),
     "[] == [] is not isolation evidence"),
    ("single_persistent_hook",
     lambda ev: ev["single_persistent_hook"].update(requests_observed=0),
     "no per-request registrations over ZERO requests is vacuous"),
    ("prefix_cache_isolation",
     lambda ev: ev["prefix_cache_isolation"].update(
         first_run_tokens=[], second_run_tokens=[]),
     "empty repeated runs prove nothing"),
    ("disable_flag",
     lambda ev: ev["disable_flag"].update(tokens_with_disable=[],
                                          baseline_tokens=[]),
     "empty disabled-run output is vacuous"),
    ("scale0_noop",
     lambda ev: ev["scale0_noop"].update(scale0_tokens=[],
                                         baseline_tokens=[]),
     "an empty scale-0 comparison is vacuous"),
    # G8: impossible evidence — a max ABSOLUTE delta is never negative
    ("returned_modification",
     lambda ev: ev["returned_modification"].update(max_abs_delta=-1.0),
     "negative max-abs delta is impossible"),
    ("returned_modification",
     lambda ev: ev["returned_modification"].update(
         max_abs_delta=float("-inf")),
     "-inf sorted below every atol and used to pass"),
])
def test_phase0_vacuous_or_impossible_evidence_fails(name, mutation, why):
    ev = _good_evidence()
    mutation(ev)
    with pytest.raises(ls.Phase0Failure, match=name):
        ls.run_phase0(ev)


def test_phase0_missing_evidence_is_a_failure_and_failures_accumulate():
    ev = _good_evidence()
    del ev["scale0_noop"]
    ev["saved_sentinel"]["decode_fires"] = 0
    with pytest.raises(ls.Phase0Failure, match="2/8"):
        ls.run_phase0(ev)


def test_phase0_malformed_evidence_is_normalized_not_an_abort():
    """A probe emitting the wrong TYPE (int()/list() raising inside
    a checker) becomes that assertion's own failure and every remaining
    assertion is still evaluated — the aggregate report survives."""
    ev = _good_evidence()
    ev["saved_sentinel"]["decode_fires"] = "not-an-int"     # TypeError-ish
    ev["mixed_batch_isolation"]["unhooked_in_batch_tokens"] = None
    ev["scale0_noop"]["scale0_tokens"] = [9, 8]             # a real failure
    with pytest.raises(ls.Phase0Failure) as err:
        ls.run_phase0(ev)
    msg = str(err.value)
    assert "3/8" in msg                       # all three filed, none masked
    assert "saved_sentinel" in msg and "could not be decoded" in msg
    assert "mixed_batch_isolation" in msg
    assert "scale0_noop" in msg


# --------------------------------------------------------------------------
# the worker-side object registry (H3): export / preload / select
# --------------------------------------------------------------------------


def test_registry_export_preload_and_apply(tmp_path):
    spec, source, cells = _lens_cells(tmp_path)
    reg_dir = tmp_path / "registry"
    entries = ls.export_worker_registry(cells, source, reg_dir)
    assert {e.cell_id for e in entries} == {"si_safety__cast",
                                            "none__lens0"}
    noop = next(e for e in entries if e.cell_id == "none__lens0")
    assert noop.forced_scale == 0.0
    assert noop.donor_cell == "si_safety__cast"  # real tensors, scale 0
    by_id = {c.cell_id: c for c in cells.all_served()}
    assert noop.expect_hash == by_id["none__lens0"].intervention_hash

    reg = ls.WorkerObjectRegistry.load(reg_dir)
    assert reg.cell_ids() == ("none__lens0", "si_safety__cast")
    arm = reg.select("si_safety__cast",
                     by_id["si_safety__cast"].intervention_hash)
    assert arm.effective_scale == 1.0 and len(arm.ops) == 1
    # applying at the op's layer steers; at another layer it does not
    hidden = torch.zeros(2, 8)
    steered = la.apply_hook_ops(hidden, arm.ops, arm.effective_scale,
                                layer=3)
    assert not torch.equal(steered, hidden)
    untouched = la.apply_hook_ops(hidden, arm.ops, arm.effective_scale,
                                  layer=5)
    assert torch.equal(untouched, hidden)
    # the scale-0 no-op leg: the SAME tensor comes back (bit-exact no-op)
    noop_arm = reg.select("none__lens0",
                          by_id["none__lens0"].intervention_hash,
                          scale=0.0)
    out = la.apply_hook_ops(hidden, noop_arm.ops,
                            noop_arm.effective_scale, layer=3)
    assert out is hidden


def test_registry_selector_refusals(tmp_path):
    spec, source, cells = _lens_cells(tmp_path)
    reg_dir = tmp_path / "registry"
    ls.export_worker_registry(cells, source, reg_dir)
    reg = ls.WorkerObjectRegistry.load(reg_dir)
    ihash = {c.cell_id: c for c in cells.canonical}[
        "si_safety__cast"].intervention_hash
    with pytest.raises(ls.LensSteerError, match="unknown petri_object"):
        reg.select("si_safety__loreft", ihash)
    with pytest.raises(ls.LensSteerError, match="expect_hash"):
        reg.select("si_safety__cast", "00" * 8)
    with pytest.raises(ls.LensSteerError, match="scale"):
        reg.select("si_safety__cast", ihash, scale=0.0)  # not the no-op leg


def test_registry_refuses_tampered_artifact_at_startup(tmp_path):
    spec, source, cells = _lens_cells(tmp_path)
    reg_dir = tmp_path / "registry"
    ls.export_worker_registry(cells, source, reg_dir)
    blob = reg_dir / "si_safety__cast" / "artifact.bin"
    blob.chmod(0o600)
    blob.write_bytes(b"tampered" + blob.read_bytes())
    with pytest.raises(ls.LensSteerError, match="STARTUP"):
        ls.WorkerObjectRegistry.load(reg_dir)


def test_registry_refuses_a_rebound_artifact_under_the_old_selector(
        tmp_path):
    """G4: an index that REPLACES the artifact and consistently updates
    its own recorded hashes (artifact_sha16 + artifact_intervention_hash)
    while keeping the original expect_hash would serve replacement
    tensors under the old intervention identity — the selector hash must
    bind the canonical artifact's recomputed hash."""
    spec, source, cells = _lens_cells(tmp_path)
    reg_dir = tmp_path / "registry"
    ls.export_worker_registry(cells, source, reg_dir)
    replacement = _synthetic_artifact(seed=99)
    blob = reg_dir / "si_safety__cast" / "artifact.bin"
    blob.chmod(0o600)
    blob.write_bytes(replacement)
    index_path = reg_dir / ls.REGISTRY_INDEX
    index_path.chmod(0o600)
    index = json.loads(index_path.read_text())
    for e in index["entries"]:
        if e["artifact"] is not None:
            # the attacker keeps the index self-consistent...
            e["artifact_sha16"] = cm.sha256_hex16(replacement)
            e["artifact_intervention_hash"] = cm.petri_intervention_hash(
                e["resolved_spec"], artifact_bytes=replacement,
                sidecar_bytes=b"")
            # ...but expect_hash (the immutable request selector) stays
    index_path.write_text(json.dumps(index))
    with pytest.raises(ls.LensSteerError, match="expect_hash"):
        ls.WorkerObjectRegistry.load(reg_dir)


def test_registry_validates_the_donor_relationship_explicitly(tmp_path):
    """G4: the donor-backed no-op leg is the ONE legal expect_hash split —
    a donor entry that is missing, or whose artifact identity the no-op
    entry does not share, refuses at startup."""
    spec, source, cells = _lens_cells(tmp_path)
    reg_dir = tmp_path / "registry"
    ls.export_worker_registry(cells, source, reg_dir)
    index_path = reg_dir / ls.REGISTRY_INDEX
    index_path.chmod(0o600)
    index = json.loads(index_path.read_text())
    for e in index["entries"]:
        if e["cell_id"] == "none__lens0":
            e["donor_cell"] = "si_safety__loreft"  # not in the registry
    index_path.write_text(json.dumps(index))
    with pytest.raises(ls.LensSteerError, match="donor"):
        ls.WorkerObjectRegistry.load(reg_dir)


def test_registry_passes_sidecar_bytes_to_the_adapter(tmp_path):
    """G5: hashed sidecar bytes must REACH the adapter, never be
    discarded after verification — the synthetic kind takes no sidecar,
    so a sidecar-bearing entry refuses INSIDE the adapter (proof the
    bytes arrived), with the refusal naming the sidecar."""
    sidecar = b'{"synthetic": "sidecar metadata"}'
    spec, source, cells = _lens_cells(tmp_path, sidecar_bytes=sidecar)
    reg_dir = tmp_path / "registry"
    ls.export_worker_registry(cells, source, reg_dir)
    with pytest.raises(la.AdapterError, match="sidecar"):
        ls.WorkerObjectRegistry.load(reg_dir)


def test_adapter_contract_takes_sidecar_bytes():
    """The AdapterFn contract is (resolved_spec, artifact_bytes,
    sidecar_bytes): the shipped kinds validate the sidecar explicitly."""
    artifact = _synthetic_artifact()
    ops = la.get_adapter(la.SYNTHETIC_KIND)(
        {"kind": la.SYNTHETIC_KIND, "layer": 3}, artifact, b"")
    assert len(ops) == 1
    with pytest.raises(la.AdapterError, match="sidecar"):
        la.get_adapter(la.SYNTHETIC_KIND)(
            {"kind": la.SYNTHETIC_KIND, "layer": 3}, artifact, b"extra")
    with pytest.raises(la.AdapterError, match="sidecar"):
        la.get_adapter(la.ZERO_KIND)(
            {"kind": la.ZERO_KIND, "hidden_size": 8, "layer": 0},
            b"", b"extra")


def test_materialized_registry_applies_without_per_token_conversion(
        tmp_path):
    """G7: hook ops are materialized to the serving device/dtype ONCE at
    worker startup; the request-time hook refuses a device/dtype mismatch
    rather than quietly converting per generated token."""
    spec, source, cells = _lens_cells(tmp_path)
    reg_dir = tmp_path / "registry"
    ls.export_worker_registry(cells, source, reg_dir)
    reg = ls.WorkerObjectRegistry.load(reg_dir)
    by_id = {c.cell_id: c for c in cells.all_served()}
    ihash = by_id["si_safety__cast"].intervention_hash
    # un-materialized ops are fp32; a non-fp32 hidden state must refuse
    hidden64 = torch.zeros(2, 8, dtype=torch.float64)
    arm = reg.select("si_safety__cast", ihash)
    with pytest.raises(la.AdapterError, match="materialize"):
        la.apply_hook_ops(hidden64, arm.ops, arm.effective_scale, layer=3)
    # materializing once makes the request-time op a pure tensor op
    mat = reg.materialize(device="cpu", dtype=torch.float64)
    arm64 = mat.select("si_safety__cast", ihash)
    assert all(op.vector.dtype == torch.float64 for op in arm64.ops)
    steered = la.apply_hook_ops(hidden64, arm64.ops, arm64.effective_scale,
                                layer=3)
    assert not torch.equal(steered, hidden64)
    # the scale-0 no-op leg stays bit-exact through materialization
    noop = mat.select("none__lens0", by_id["none__lens0"].intervention_hash,
                      scale=0.0)
    out = la.apply_hook_ops(hidden64, noop.ops, noop.effective_scale,
                            layer=3)
    assert out is hidden64


def test_export_refuses_bytes_changed_since_enumeration(tmp_path):
    spec, source, cells = _lens_cells(tmp_path)
    # the artifact changes between enumeration and export -> the
    # recomputed intervention hash no longer matches the cell's
    Path(source.entries["si_safety__cast"].object_path).write_bytes(
        _synthetic_artifact(seed=1))
    with pytest.raises(ls.LensSteerError, match="not the enumerated"):
        ls.export_worker_registry(cells, source, tmp_path / "registry")


def test_export_no_donor_needs_explicit_zero_op(tmp_path):
    spec, source, cells = _lens_cells(tmp_path, with_object=False)
    with pytest.raises(ls.LensSteerError, match="zero_op_hidden_size"):
        ls.export_worker_registry(cells, source, tmp_path / "r1")
    entries = ls.export_worker_registry(cells, source, tmp_path / "r2",
                                        zero_op_hidden_size=8)
    (noop,) = entries
    assert noop.kind == la.ZERO_KIND and noop.forced_scale == 0.0
    reg = ls.WorkerObjectRegistry.load(tmp_path / "r2")
    hidden = torch.ones(2, 8)
    out = reg.apply("none__lens0", noop.expect_hash, hidden, layer=0,
                    scale=0.0)
    assert out is hidden  # armed hook, exact no-op


def test_unknown_kind_refuses_at_preload_not_request_time(tmp_path):
    spec, source, cells = _lens_cells(tmp_path)
    reg_dir = tmp_path / "registry"
    ls.export_worker_registry(cells, source, reg_dir)
    index_path = reg_dir / ls.REGISTRY_INDEX
    index_path.chmod(0o600)
    index = json.loads(index_path.read_text())
    for e in index["entries"]:
        if e["cell_id"] == "si_safety__cast":
            e["kind"] = "obj_cast"  # a real kind with NO adapter yet
            e["resolved_spec"]["kind"] = "obj_cast"
            # keep the integrity hashes consistent with the edited spec
            # (expect_hash included — this test is about the adapter, not
            # the selector-binding refusal, which has its own test above)
            blob = (reg_dir / e["artifact"]).read_bytes()
            rehash = cm.petri_intervention_hash(
                e["resolved_spec"], artifact_bytes=blob, sidecar_bytes=b"")
            e["artifact_intervention_hash"] = rehash
            e["expect_hash"] = rehash
    index_path.write_text(json.dumps(index))
    with pytest.raises(la.AdapterError, match="no lens adapter"):
        ls.WorkerObjectRegistry.load(reg_dir)


def test_selector_parse_allows_ids_only():
    sel = ls.parse_selector({"petri_object": "x", "expect_hash": "ab" * 8,
                             "scale": 0.0})
    assert sel == {"petri_object": "x", "expect_hash": "ab" * 8,
                   "scale": 0.0}
    with pytest.raises(ls.LensSteerError, match="unknown key"):
        ls.parse_selector({"petri_object": "x", "expect_hash": "y",
                           "tensor": [1, 2]})
    with pytest.raises(ls.LensSteerError, match="non-empty string"):
        ls.parse_selector({"petri_object": "", "expect_hash": "y"})


def test_require_vllm_lens_guards_the_cpu_lane():
    if ls.HAVE_VLLM_LENS:
        pytest.skip("vllm_lens installed here; the guard is for the CPU "
                    "venv")
    with pytest.raises(ls.LensSteerError, match="serving venv"):
        ls.require_vllm_lens()


# --------------------------------------------------------------------------
# vllm serve composition invariants
# --------------------------------------------------------------------------


def _tpl_spec(**over):
    """A spec whose chat_template_hash matches CHAT_TEMPLATE_BYTES."""
    over.setdefault("chat_template_hash", CHAT_TEMPLATE_HASH)
    return _spec(**over)


def _flat_registry(tmp_path) -> Path:
    """A real, loadable flat worker registry (the export layout) for
    launch-composition tests — compose-time resolution now refuses a
    path with nothing behind it."""
    spec, source, cells = _lens_cells(tmp_path)
    reg_dir = tmp_path / "reg"
    ls.export_worker_registry(cells, source, reg_dir)
    return reg_dir


def test_build_server_launch_pins_the_invariants(tmp_path):
    tpl = _chat_template(tmp_path)
    launch = vs.build_server_launch(
        _tpl_spec(), port=8801, lens_registry_dir=_flat_registry(tmp_path),
        lora_modules={"none__lora0": tmp_path / "z",
                      "si_safety__lora_r1": tmp_path / "a"},
        exact_replay_strategy="serialize", chat_template_path=tpl)
    cmd = list(launch.cmd)
    assert cmd[:3] == ["vllm", "serve", "meta-llama/Llama-3.1-8B-Instruct"]
    for flag in ("--enforce-eager", "--enable-prefix-caching",
                 "--enable-lora", "--no-enable-auto-tool-choice",
                 "--no-async-scheduling"):
        assert flag in cmd
    # caching ON is the DEFAULT (per-request cache_salt isolation at the
    # proxy; Phase-0 assertion 7 proves it live) — prefix_caching=False
    # composes the legacy OFF launch.
    assert "--no-enable-prefix-caching" not in cmd
    assert launch.prefix_caching is True
    off = vs.build_server_launch(
        _tpl_spec(serving_launch_flags=["--enforce-eager"]), port=8801,
        exact_replay_strategy="serialize",
        chat_template_path=_chat_template(tmp_path), prefix_caching=False)
    assert "--no-enable-prefix-caching" in off.cmd
    assert "--enable-prefix-caching" not in off.cmd
    assert off.prefix_caching is False
    assert cmd[cmd.index("--revision") + 1] == REV
    assert "--tool-call-parser" not in cmd  # generation default: no parser
    assert cmd[cmd.index("--chat-template") + 1] == str(tpl)
    assert f"none__lora0={tmp_path / 'z'}" in cmd
    assert launch.model == "meta-llama/Llama-3.1-8B-Instruct"
    # lens plugin + worker registry ride the env (worker preload, H3)
    assert launch.env["VLLM_PLUGINS"] == "activations"  # entry-point name,
    # NOT the distribution name: vLLM filters general plugins by entry-point
    # name, so "vllm_lens" would load no plugin and steer nothing.
    assert launch.env["VLLM_PLUGINS"] == vs.LENS_PLUGIN_ENTRY_POINT
    assert launch.env[vs.LENS_REGISTRY_DIR_ENV] == str(tmp_path / "reg")
    # serialize strategy -> the PROXY serializes; no batch-invariant env
    assert vs.BATCH_INVARIANT_ENV not in launch.env
    batch = vs.build_server_launch(
        _tpl_spec(serving_launch_flags=["--enforce-eager"]), port=8801,
        exact_replay_strategy="batch_invariant", chat_template_path=tpl)
    assert batch.env[vs.BATCH_INVARIANT_ENV] == "1"
    # THE LORA-NATIVE MODE: the same composer
    # states the OPPOSITE eager choice explicitly, keeps the plugin out via
    # the env, fits the rank cap to the registered adapters, composes the
    # CPU residency, and names the plugin-free extension — while every
    # lens launch above composes exactly as it did before the mode existed.
    bank = tmp_path / "bank"
    native = vs.build_server_launch(
        _tpl_spec(serving_launch_flags=[]), port=8801,
        lora_modules={"none__lora0": _tiny_peft_adapter(bank / "z", r=1),
                      "si_safety__lora_r4": _tiny_peft_adapter(bank / "a",
                                                                r=4)},
        exact_replay_strategy="serialize", chat_template_path=tpl,
        lora_native=True)
    ncmd = list(native.cmd)
    assert "--no-enforce-eager" in ncmd and "--enforce-eager" not in ncmd
    assert ncmd[ncmd.index("--max-loras") + 1] == "2"
    assert ncmd[ncmd.index("--max-lora-rank") + 1] == "8"   # fitted, not 64
    assert ncmd[ncmd.index("--max-cpu-loras") + 1] == "2"
    assert ncmd[ncmd.index("--worker-extension-cls") + 1] == (
        vs.NATIVE_WORKER_EXTENSION_CLS)
    assert native.env["VLLM_PLUGINS"] == ""            # set EMPTY, not omitted
    assert native.env[vs.LENS_DISABLE_ENV] == "1"
    assert native.env[vs.SERVER_DEV_MODE_ENV] == "1"
    assert vs.LENS_REGISTRY_DIR_ENV not in native.env
    assert native.lora_native is True
    assert native.lora_modules == ("none__lora0", "si_safety__lora_r4")
    assert native.max_loras == native.max_cpu_loras == 2
    assert native.max_lora_rank == 8
    assert f"none__lora0={(bank / 'z').resolve()}" in ncmd


def test_no_chunking_geometry_is_composed_and_strategy_sized(tmp_path):
    """The production launch previously composed NO scheduler geometry,
    so vLLM's server default budget (8192 on H200, chunked prefill
    forced on in V1) chunked long/concurrent prefills mid-prompt — which
    the plan lattice refuses, bricking plan-needing arms. The composer
    now always pins max-num-seqs and the product
    max-num-batched-tokens = max_num_seqs * max_model_len (the smoke's
    exact no-chunking argument), sized per exact-replay strategy."""
    tpl = _chat_template(tmp_path)
    ser = vs.build_server_launch(
        _tpl_spec(serving_launch_flags=["--enforce-eager"]), port=8801,
        exact_replay_strategy="serialize", chat_template_path=tpl)
    cmd = list(ser.cmd)
    # serialize: the proxy holds a global lock -> one request in flight
    assert cmd[cmd.index("--max-num-seqs") + 1] == "1"
    assert cmd[cmd.index("--max-num-batched-tokens") + 1] == str(16384)
    assert (ser.max_num_seqs, ser.max_num_batched_tokens,
            ser.max_model_len) == (1, 16384, 16384)
    bi = vs.build_server_launch(
        _tpl_spec(serving_launch_flags=["--enforce-eager"]), port=8801,
        exact_replay_strategy="batch_invariant", chat_template_path=tpl)
    cmd = list(bi.cmd)
    # concurrent default: 8 -> 32 (2026-08-27) for audit concurrency,
    # then 32 -> 24 (2026-08-28) on a MEASURED negative result: 32 x
    # 16384 does not fit one H200 under no-chunking (jobs 229194/229390
    # — full-coverage KV 64.0 GiB + a 14.0 GiB full-budget MLP warmup
    # intermediate + ~15 GiB weights > 139.81 GiB, and utilization is
    # zero-sum between KV and warmup headroom; the constant's own note
    # has the bracketing). 24 x 16384 = 393,216 is the widest that fits
    # with real slack, and it is what the smoke's PETRI-geometry gate
    # certifies with resolved-KV-coverage asserted.
    assert vs.DEFAULT_MAX_NUM_SEQS == 24
    assert cmd[cmd.index("--max-num-seqs") + 1] == "24"
    assert cmd[cmd.index("--max-num-batched-tokens") + 1] == str(24 * 16384)
    # explicit max_num_seqs wins; the product is re-derived, never reused
    wide = vs.build_server_launch(
        _tpl_spec(serving_launch_flags=["--enforce-eager"]), port=8801,
        exact_replay_strategy="batch_invariant", chat_template_path=tpl,
        max_model_len=2048, max_num_seqs=32)
    cmd = list(wide.cmd)
    assert cmd[cmd.index("--max-num-seqs") + 1] == "32"
    assert cmd[cmd.index("--max-num-batched-tokens") + 1] == str(32 * 2048)
    assert wide.max_num_batched_tokens == 65536  # the certified product
    # ... and the sizing is not the whole invariant: the partial-prefill
    # knobs are pinned explicitly on EVERY launch (a non-zero threshold
    # chunks long prompts regardless of the budget; >1 partial prefills
    # silently auto-set it to 4% of max_model_len at the pinned tag).
    for launch in (ser, bi, wide):
        cmd = list(launch.cmd)
        assert cmd[cmd.index("--long-prefill-token-threshold") + 1] == "0"
        assert cmd[cmd.index("--max-num-partial-prefills") + 1] == "1"
        assert cmd[cmd.index("--max-long-partial-prefills") + 1] == "1"
        # and no speculative surface is ever composed:
        assert not any(t.startswith("--spec") for t in cmd)


def test_geometry_validation_refusals(tmp_path):
    tpl = _chat_template(tmp_path)

    def build(**kw):
        return vs.build_server_launch(
            _tpl_spec(serving_launch_flags=["--enforce-eager"]), port=8801,
            exact_replay_strategy="serialize", chat_template_path=tpl, **kw)

    with pytest.raises(vs.VllmServerError, match="max_num_seqs"):
        build(max_num_seqs=0)
    with pytest.raises(vs.VllmServerError, match="max_num_seqs"):
        build(max_num_seqs=True)          # a bool is not a width
    with pytest.raises(vs.VllmServerError, match="max_num_seqs"):
        build(max_num_seqs="8")
    with pytest.raises(vs.VllmServerError, match="max_model_len"):
        build(max_model_len=0)
    # overflow safety: the derived product must never wrap an int32
    # downstream — refused at compose time, loudly.
    with pytest.raises(vs.VllmServerError, match="int32"):
        build(max_num_seqs=2**18, max_model_len=2**14)


def test_manifest_cannot_override_the_scheduler_geometry(tmp_path):
    """Every scheduler-bypass surface the PINNED parser actually accepts
    is wrapper-owned, in every accepted spelling: the sizing options,
    the partial-prefill knobs (a threshold chunks regardless of the
    budget), every 0.25.1 speculative surface (--speculative-config and
    the --spec-* shorthands), the chunked-prefill toggles, a
    replacement scheduler class, and the arbitrary-injection routes
    (--config YAML splices any engine arg; --additional-config). The
    pinned FlexibleArgumentParser takes '_' and '-' interchangeably and
    collapses dotted '--opt.field' forms into '--opt', so those
    spellings must refuse too."""
    tpl = _chat_template(tmp_path)
    for flags in (
            # the sizing, all spellings
            ["--max-num-batched-tokens", "2048"],
            ["--max-num-batched-tokens=2048"],
            ["--max_num_batched_tokens", "2048"],     # underscore alias
            ["--max-num-seqs", "1024"],
            ["--max-num-seqs=1024"],
            ["--max_num_seqs=1024"],
            # the partial-prefill knobs (big budget still chunks)
            ["--long-prefill-token-threshold", "512"],
            ["--long_prefill_token_threshold=512"],
            ["--max-num-partial-prefills", "4"],
            ["--max-long-partial-prefills", "2"],
            ["--enable-chunked-prefill"],
            ["--no-enable-chunked-prefill"],
            # every speculative surface at the pinned tag
            ["--speculative-config", '{"num_speculative_tokens": 3}'],
            ["--speculative_config={\"num_speculative_tokens\": 3}"],
            ["--speculative-config.num-speculative-tokens", "3"],  # dotted
            ["--spec-method", "ngram"],
            ["--spec-model", "some/draft"],
            ["--spec-tokens", "3"],
            # arbitrary-injection and scheduler-replacement routes
            ["--config", "sched.yaml"],
            ["--config=sched.yaml"],
            ["--additional-config", "{}"],
            ["--scheduler-cls", "evil.Scheduler"],
    ):
        with pytest.raises(vs.VllmServerError, match="wrapper-owned"):
            vs.build_server_launch(
                _tpl_spec(serving_launch_flags=flags), port=8801,
                exact_replay_strategy="serialize", chat_template_path=tpl)
    # the short alias -sc dies on the stray-token rule (long-form only)
    with pytest.raises(vs.VllmServerError, match="stray value"):
        vs.build_server_launch(
            _tpl_spec(serving_launch_flags=["-sc",
                                            '{"num_speculative_tokens": 3}']),
            port=8801, exact_replay_strategy="serialize",
            chat_template_path=tpl)
    # underscore restatements of a pin are NOT skipped as duplicates —
    # only the canonical spelling is a documented restatement
    with pytest.raises(vs.VllmServerError, match="wrapper-owned"):
        vs.build_server_launch(
            _tpl_spec(serving_launch_flags=["--enforce_eager"]),
            port=8801, exact_replay_strategy="serialize",
            chat_template_path=tpl)


def test_manifest_cannot_abbreviate_a_wrapper_owned_option(tmp_path):
    """THE ABBREVIATION BYPASS. The pinned parser never sets
    allow_abbrev=False, so argparse accepts any unambiguous PREFIX of a
    real option — '--speculative-conf <json>' sets --speculative-config
    just as surely as the full spelling. Every manifest option whose
    canonical name is a prefix of a wrapper-owned option refuses,
    conservatively (ambiguous prefixes included), in every accepted
    spelling: bare, =-form, underscore, and dotted."""
    tpl = _chat_template(tmp_path)
    for flags in (
            # the reviewer's named cases
            ["--speculative-conf", '{"num_speculative_tokens": 3}'],
            ["--long-prefill-token-thresh", "512"],
            ["--scheduler-c", "evil.Scheduler"],
            # shorter variants, down to ambiguous stubs
            ["--speculative", "{}"],
            ["--spec"],
            ["--long-prefill", "512"],
            ["--sched", "x"],
            ["--max-num-batched", "2048"],
            ["--max-num", "1"],
            ["--conf", "sched.yaml"],
            ["--c", "sched.yaml"],
            # =-form, underscore, and dotted spellings of an abbreviation
            ["--speculative-conf={\"num_speculative_tokens\": 3}"],
            ["--long_prefill_token_thresh", "512"],
            ["--speculative-conf.num-speculative-tokens", "3"],
    ):
        with pytest.raises(vs.VllmServerError,
                           match="ABBREVIATION|wrapper-owned"):
            vs.build_server_launch(
                _tpl_spec(serving_launch_flags=flags), port=8801,
                exact_replay_strategy="serialize", chat_template_path=tpl)
    # ... while unowned passthrough options that merely SHARE TEXT with
    # owned ones (but are not prefixes of any) still travel:
    launch = vs.build_server_launch(
        _tpl_spec(serving_launch_flags=["--enforce-eager",
                                        "--swap-space=8",
                                        "--gpu-memory-utilization", "0.9"]),
        port=8801, exact_replay_strategy="serialize",
        chat_template_path=tpl)
    assert "--swap-space=8" in launch.cmd


def test_tool_parser_is_agent_modality_only(tmp_path):
    """THE GEMMA WAVE BLOCKER (Slurm job 232411, tasks 0-3). The
    composer pinned the manifest's llama3_json parser on EVERY launch,
    including generation-only serving of a gemma-4 target — and the
    pinned vLLM instantiates the parser class per /v1/chat/completions
    request, so the server came up healthy, generated for 28-35
    minutes, and then 500d on the first chat request (Llama3JsonToolParser
    cannot find '<|python_tag|>' in the gemma tokenizer). Tool calls are
    a MODALITY choice now, stated explicitly on argv either way:
    generation launches (the default) pin auto tool choice OFF with no
    parser — those workloads send no `tools` — and only the agent
    modality (tool_calls=True: the petri audit's tool round-trips,
    verified live by the smoke) composes the manifest's per-model
    parser pin."""
    tpl = _chat_template(tmp_path)
    gen = vs.build_server_launch(
        _tpl_spec(serving_launch_flags=["--enforce-eager"]), port=8801,
        exact_replay_strategy="serialize", chat_template_path=tpl)
    assert "--no-enable-auto-tool-choice" in gen.cmd
    assert "--enable-auto-tool-choice" not in gen.cmd
    assert "--tool-call-parser" not in gen.cmd

    agent = vs.build_server_launch(
        _tpl_spec(serving_launch_flags=["--enforce-eager"]), port=8801,
        exact_replay_strategy="serialize", chat_template_path=tpl,
        tool_calls=True)
    cmd = list(agent.cmd)
    assert "--enable-auto-tool-choice" in cmd
    assert cmd[cmd.index("--tool-call-parser") + 1] == "llama3_json"
    assert "--no-enable-auto-tool-choice" not in cmd

    # the toggle is a composed modality choice like prefix caching, so a
    # manifest naming EITHER spelling refuses (it was a tolerated
    # restatement back when the flag was unconditionally pinned).
    for flags in (["--enable-auto-tool-choice"],
                  ["--no-enable-auto-tool-choice"],
                  ["--tool-call-parser", "hermes"]):
        with pytest.raises(vs.VllmServerError, match="wrapper-owned"):
            vs.build_server_launch(
                _tpl_spec(serving_launch_flags=flags), port=8801,
                exact_replay_strategy="serialize", chat_template_path=tpl)


def test_synchronous_scheduling_is_pinned_explicitly(tmp_path):
    """THE LIVE SERVING BLOCKER (Slurm job 194568). The production
    launch composed NO async-scheduling flag, assuming omission meant
    the synchronous Scheduler — but the installed 0.25.1 resolved
    scheduler_cls to vllm.v1.core.sched.async_scheduler.AsyncScheduler
    and readiness (correctly) refused the attestation, blocking the
    launch. Omission is not a pin: the composer now states
    --no-async-scheduling on EVERY launch (the installed CLI generation
    renders bool|None options with argparse.BooleanOptionalAction, so
    the negative form is a real accepted flag), and the runtime
    attestation still expects the pinned synchronous class."""
    tpl = _chat_template(tmp_path)
    ser = vs.build_server_launch(
        _tpl_spec(serving_launch_flags=["--enforce-eager"]), port=8801,
        exact_replay_strategy="serialize", chat_template_path=tpl)
    bi = vs.build_server_launch(
        _tpl_spec(serving_launch_flags=["--enforce-eager"]), port=8801,
        exact_replay_strategy="batch_invariant", chat_template_path=tpl)
    lens = vs.build_server_launch(
        _tpl_spec(), port=8801, lens_registry_dir=_flat_registry(tmp_path),
        lora_modules={"none__lora0": tmp_path / "z"},
        exact_replay_strategy="serialize", chat_template_path=tpl)
    assert vs.SYNC_SCHEDULING_FLAG == "--no-async-scheduling"
    for launch in (ser, bi, lens):
        cmd = list(launch.cmd)
        assert cmd.count(vs.SYNC_SCHEDULING_FLAG) == 1
        assert "--async-scheduling" not in cmd
        # ... and the runtime half is unchanged: every worker's report
        # must still attest the pinned synchronous class the flag
        # selects — the argv pin and the RPC attestation are two halves
        # of the same guarantee, not alternatives.
        exp = vs.expected_scheduler_state(launch)
        assert (exp["scheduler_cls"] == vs.PINNED_DEFAULT_SCHEDULER_CLS
                == "vllm.v1.core.sched.scheduler.Scheduler")


def test_manifest_cannot_toggle_async_scheduling(tmp_path):
    """Both BooleanOptionalAction spellings of the scheduler-class
    toggle are wrapper-owned, in every form the pinned parser accepts:
    positive, negative, underscore, dotted, =-form, split-form, and
    every abbreviation (allow_abbrev is on, so a prefix sets the full
    option). The NEGATIVE spelling refuses too — it is deliberately not
    a tolerated pin restatement: the pair is one owned surface, the pin
    is already composed, and a manifest has no business naming it."""
    tpl = _chat_template(tmp_path)
    for flags in (
            ["--async-scheduling"],                # positive, bare
            ["--async-scheduling=true"],           # =-form
            ["--async-scheduling", "true"],        # split-form value
            ["--async_scheduling"],                # underscore alias
            ["--async_scheduling=1"],
            ["--async-scheduling.enabled", "1"],   # dotted collapses
            ["--no-async-scheduling"],             # negative, bare
            ["--no-async-scheduling=false"],
            ["--no_async_scheduling"],
            ["--no-async-scheduling.x", "0"],
    ):
        with pytest.raises(vs.VllmServerError, match="wrapper-owned"):
            vs.build_server_launch(
                _tpl_spec(serving_launch_flags=flags), port=8801,
                exact_replay_strategy="serialize", chat_template_path=tpl)
    for flags in (
            ["--async-sched"],                     # abbreviated positive
            ["--async"],
            ["--no-async-sched"],                  # abbreviated negative
            ["--no-async"],
            ["--async_sched"],                     # underscore abbreviation
            ["--no_async"],
            ["--async-schedulin=true"],            # one char short, =-form
    ):
        with pytest.raises(vs.VllmServerError,
                           match="ABBREVIATION|wrapper-owned"):
            vs.build_server_launch(
                _tpl_spec(serving_launch_flags=flags), port=8801,
                exact_replay_strategy="serialize", chat_template_path=tpl)


def test_build_server_launch_refusals(tmp_path):
    tpl = _chat_template(tmp_path)
    with pytest.raises(vs.VllmServerError, match="wildcard"):
        vs.build_server_launch(_tpl_spec(), port=8801, host="0.0.0.0",
                               exact_replay_strategy="serialize",
                               chat_template_path=tpl)
    with pytest.raises(vs.VllmServerError, match="stated choice"):
        vs.build_server_launch(_tpl_spec(), port=8801,
                               chat_template_path=tpl)  # no strategy
    with pytest.raises(vs.VllmServerError, match="mislabel"):
        vs.build_server_launch(_tpl_spec(serving_mode="interleaved"),
                               port=8801, exact_replay_strategy="serialize",
                               chat_template_path=tpl)
    # NEITHER prefix-caching spelling may ride the manifest: the
    # setting is a composed CHOICE (prefix_caching=..., default True),
    # wrapper-owned in both spellings — restating even the composed
    # value would let a stale manifest silently pin a superseded state.
    for flag in ("--enable-prefix-caching", "--no-enable-prefix-caching"):
        with pytest.raises(vs.VllmServerError, match="wrapper-owned"):
            vs.build_server_launch(
                _tpl_spec(serving_launch_flags=[flag]),
                port=8801, exact_replay_strategy="serialize",
                chat_template_path=tpl)
    if not vs.HAVE_VLLM:
        with pytest.raises(vs.VllmServerError, match="serving venv"):
            vs.require_vllm()


def test_chat_template_is_required_and_hash_verified(tmp_path):
    """P6: the pinned template is part of the harness identity — its
    absence, a missing file, an UNFROZEN pin, or a byte mismatch against
    spec.chat_template_hash all refuse the launch composition."""
    tpl = _chat_template(tmp_path)
    with pytest.raises(vs.VllmServerError, match="chat_template_path"):
        vs.build_server_launch(_tpl_spec(), port=8801,
                               exact_replay_strategy="serialize")
    with pytest.raises(vs.VllmServerError, match="does not exist"):
        vs.build_server_launch(_tpl_spec(), port=8801,
                               exact_replay_strategy="serialize",
                               chat_template_path=tmp_path / "missing.jinja")
    with pytest.raises(vs.VllmServerError, match="UNFROZEN"):
        vs.build_server_launch(
            _tpl_spec(chat_template_hash=cm.PETRI_UNFROZEN), port=8801,
            exact_replay_strategy="serialize", chat_template_path=tpl)
    stale = tmp_path / "stale.jinja"
    stale.write_bytes(b"{# some other template #}\n")
    with pytest.raises(vs.VllmServerError, match="hashes to"):
        vs.build_server_launch(_tpl_spec(), port=8801,
                               exact_replay_strategy="serialize",
                               chat_template_path=stale)


@pytest.mark.parametrize("flags", [
    ["--dtype=float16"],             # =-form override of the bf16 pin
    ["--dtype", "float16"],          # split-form override
    ["--host=0.0.0.0"],              # bind override behind the validation
    ["--host", "0.0.0.0"],
    ["--revision", "0" * 40],        # model-pin override
    ["--chat-template=/tmp/other"],  # verified-template override
    ["--max-model-len", "999"],      # the old dedup stranded the "999"
    ["--enforce-eager", "true"],     # a pinned flag with a smuggled value
    ["--worker-extension-cls=evil.Ext"],   # the steering-extension slot
    ["--worker-extension-cls", "evil.Ext"],
])
def test_manifest_flags_cannot_override_wrapper_owned_options(tmp_path,
                                                              flags):
    """P5: every wrapper-owned option refuses in BOTH spellings — the old
    composer appended '--dtype=float16' after the pinned '--dtype
    bfloat16', so argparse's last-wins gave the manifest silent control
    of supposedly non-negotiable settings."""
    tpl = _chat_template(tmp_path)
    with pytest.raises(vs.VllmServerError, match="wrapper-owned"):
        vs.build_server_launch(_tpl_spec(serving_launch_flags=flags),
                               port=8801, exact_replay_strategy="serialize",
                               chat_template_path=tpl)


def test_manifest_flag_passthrough_and_pin_restatements(tmp_path):
    tpl = _chat_template(tmp_path)
    launch = vs.build_server_launch(
        _tpl_spec(serving_launch_flags=[
            "--enforce-eager",             # bare restatement of a pin: OK
            "--swap-space=8",              # unowned: passes through
            "--gpu-memory-utilization", "0.9",  # split pair travels WHOLE
        ]),
        port=8801, exact_replay_strategy="serialize",
        chat_template_path=tpl)
    cmd = list(launch.cmd)
    assert cmd.count("--enforce-eager") == 1        # no duplicate token
    assert cmd.count("--enable-prefix-caching") == 1  # the composed default
    assert "--swap-space=8" in cmd
    i = cmd.index("--gpu-memory-utilization")
    assert cmd[i + 1] == "0.9"                      # value not stranded
    # --enable-lora restated is fine WITH adapters, refused without
    with_lora = vs.build_server_launch(
        _tpl_spec(serving_launch_flags=["--enable-lora"]), port=8801,
        lora_modules={"none__lora0": tmp_path / "z"},
        exact_replay_strategy="serialize", chat_template_path=tpl)
    assert list(with_lora.cmd).count("--enable-lora") == 1
    with pytest.raises(vs.VllmServerError, match="wrapper-owned"):
        vs.build_server_launch(
            _tpl_spec(serving_launch_flags=["--enable-lora"]), port=8801,
            exact_replay_strategy="serialize", chat_template_path=tpl)
    # a stray bare value refuses rather than riding as a positional
    with pytest.raises(vs.VllmServerError, match="stray value"):
        vs.build_server_launch(
            _tpl_spec(serving_launch_flags=["float16"]), port=8801,
            exact_replay_strategy="serialize", chat_template_path=tpl)


def test_lens_launch_names_the_sisteer_worker_extension(tmp_path):
    """The production launch must CLAIM vLLM's single worker-extension
    slot: without `--worker-extension-cls` the slot is empty, the
    vllm-lens plugin injects its own stock class, no steering hooks are
    installed, and every arm serves as base — a healthy-looking server
    that steers nothing. The composed flag must name the sisteer
    subclass (which keeps the stock RPC surface via inheritance), and
    the env must make it importable in the spawned server."""
    from sisteer.capeval.serving import lens_worker as lw

    tpl = _chat_template(tmp_path)
    reg = _flat_registry(tmp_path)
    launch = vs.build_server_launch(
        _tpl_spec(serving_launch_flags=["--enforce-eager"]), port=8801,
        lens_registry_dir=reg,
        exact_replay_strategy="serialize", chat_template_path=tpl)
    cmd = list(launch.cmd)
    i = cmd.index("--worker-extension-cls")
    assert cmd[i + 1] == lw.WORKER_EXTENSION_CLS
    assert cmd[i + 1] == ("sisteer.capeval.serving.lens_worker"
                          ".SisteerWorkerExtension")
    # importable in the child: PYTHONPATH points at the dir that holds
    # the live sisteer package
    src = Path(launch.env["PYTHONPATH"])
    assert (src / "sisteer" / "__init__.py").is_file()
    # ... and the external plugin is still registered alongside — the
    # extension does not replace the plugin, it fills the slot the
    # plugin respects (`if not self.worker_extension_cls`).
    assert launch.env["VLLM_PLUGINS"] == vs.LENS_PLUGIN_ENTRY_POINT
    # the install RPC route is exposed (readiness cannot attest without
    # it) and the launch carries the RESOLVED registry for the installer
    assert launch.env[vs.SERVER_DEV_MODE_ENV] == "1"
    assert launch.lens_registry_dir == str(reg)
    # a PLAIN launch (neither lens nor lora-native) claims nothing and
    # sets no env: no extension, no PYTHONPATH, no plugin allow-list, no
    # dev route. (Moved deliberately, native-multilora design §4.9: this
    # used to pin an untested fall-through as "lens-less"; it now pins
    # the plain mode beside the two explicit ones.)
    plain = vs.build_server_launch(
        _tpl_spec(serving_launch_flags=["--enforce-eager"]), port=8801,
        exact_replay_strategy="serialize", chat_template_path=tpl)
    assert "--worker-extension-cls" not in plain.cmd
    assert "PYTHONPATH" not in plain.env
    assert "VLLM_PLUGINS" not in plain.env
    assert vs.SERVER_DEV_MODE_ENV not in plain.env
    assert plain.lens_registry_dir is None
    assert plain.lora_native is False and plain.lora_modules == ()
    # ... the LORA-NATIVE launch is the third, EXPLICIT mode: it claims
    # the slot with the PLUGIN-FREE extension (not the lens subclass,
    # which is unimportable without vllm-lens), keeps the plugin OUT by
    # setting the allow-list EMPTY — set, not omitted: the serving-venv
    # bootstrap `unset`s VLLM_PLUGINS for the lens's benefit and the
    # child must not inherit that — flips the plugin's own disable
    # switch, exposes the dev route its attestation reads through, and
    # carries NO registry env (there is no registry to install).
    native = vs.build_server_launch(
        _tpl_spec(serving_launch_flags=[]), port=8801,
        lora_modules={"si_safety__lora_r4": _tiny_peft_adapter(
            tmp_path / "bank" / "a", r=4)},
        exact_replay_strategy="serialize", chat_template_path=tpl,
        lora_native=True)
    ncmd = list(native.cmd)
    j = ncmd.index("--worker-extension-cls")
    assert ncmd[j + 1] == vs.NATIVE_WORKER_EXTENSION_CLS == (
        "sisteer.capeval.serving.native_worker.NativeWorkerExtension")
    assert ncmd[j + 1] != lw.WORKER_EXTENSION_CLS
    assert ncmd[ncmd.index("--middleware") + 1] == vs.AUTH_MIDDLEWARE_CLS
    assert "VLLM_PLUGINS" in native.env and native.env["VLLM_PLUGINS"] == ""
    assert native.env[vs.LENS_DISABLE_ENV] == "1"
    assert native.env[vs.SERVER_DEV_MODE_ENV] == "1"
    assert vs.LENS_REGISTRY_DIR_ENV not in native.env
    assert (Path(native.env["PYTHONPATH"]) / "sisteer" / "__init__.py"
            ).is_file()
    assert native.lens_registry_dir is None and native.lora_native is True
    # ... and a lens launch against NOTHING refuses at compose time —
    # a registry that resolves to nothing could never be attested
    with pytest.raises(Exception, match="nothing to serve"):
        vs.build_server_launch(
            _tpl_spec(serving_launch_flags=["--enforce-eager"]), port=8801,
            lens_registry_dir=tmp_path / "nowhere",
            exact_replay_strategy="serialize", chat_template_path=tpl)


# The pinned parser, replicated from vllm 0.25.1 (vllm/entrypoints/
# openai/cli_args.py): nargs="+" and an action whose __call__ REPLACES
# the collected list on every flag occurrence (setattr, not append).
import argparse as _argparse


class _PinnedLoRAParserAction(_argparse.Action):
    """Functional stand-in for vLLM 0.25.1's LoRAParserAction: last
    occurrence wins."""

    def __call__(self, parser, namespace, values, option_string=None):
        modules = []
        for item in values:
            name, _, path = item.partition("=")
            modules.append((name, path))
        setattr(namespace, self.dest, modules)  # REPLACES, never appends


def _parse_lora_modules_like_pinned_vllm(cmd):
    parser = _argparse.ArgumentParser()
    parser.add_argument("--lora-modules", nargs="+", type=str,
                        action=_PinnedLoRAParserAction, default=None)
    ns, _unknown = parser.parse_known_args(list(cmd)[2:])  # skip vllm serve
    return dict(ns.lora_modules or [])


def test_all_lora_adapters_register_under_the_pinned_cli_parser(tmp_path):
    """The composer emits ONE --lora-modules flag carrying every
    name=path value. Under the pinned parser (which REPLACES the list on
    every flag occurrence), the composed form registers ALL adapters;
    the old one-flag-per-adapter form silently registered only the LAST,
    and every other arm 404'd at request time."""
    from sisteer.capeval.serving import lens_worker as lw  # noqa: F401

    tpl = _chat_template(tmp_path)
    modules = {"none__lora0": tmp_path / "z",
               "si_safety__lora_r1": tmp_path / "a",
               "si_safety__cb": tmp_path / "b"}
    launch = vs.build_server_launch(
        _tpl_spec(), port=8801, lora_modules=modules,
        exact_replay_strategy="serialize", chat_template_path=tpl)
    cmd = list(launch.cmd)
    assert cmd.count("--lora-modules") == 1  # ONE flag, many values
    i = cmd.index("--lora-modules")
    assert cmd[i + 1:i + 4] == [f"{n}={p}"
                                for n, p in sorted(modules.items())]
    parsed = _parse_lora_modules_like_pinned_vllm(cmd)
    assert set(parsed) == set(modules)  # every adapter registers
    # regression documentation: the repeated-flag form loses all but the
    # last adapter under the very same parser
    repeated = ["vllm", "serve", "m"]
    for n, p in sorted(modules.items()):
        repeated += ["--lora-modules", f"{n}={p}"]
    lost = _parse_lora_modules_like_pinned_vllm(repeated)
    assert len(lost) == 1  # the defect this composition avoids


# --------------------------------------------------------------------------
# per-launch API token: minted, attached, never leaked
# --------------------------------------------------------------------------


def test_every_launch_mints_a_random_token_and_never_leaks_it(tmp_path):
    """Loopback is not tenant isolation on shared Slurm, so every
    composed launch carries a cryptographically random token — and the
    token appears NOWHERE recordable: not on argv (world-readable via
    /proc), not in launch.env (the proxy records it as attestation),
    not in repr (log lines)."""
    tpl = _chat_template(tmp_path)
    a = vs.build_server_launch(
        _tpl_spec(serving_launch_flags=["--enforce-eager"]), port=8801,
        exact_replay_strategy="serialize", chat_template_path=tpl)
    b = vs.build_server_launch(
        _tpl_spec(serving_launch_flags=["--enforce-eager"]), port=8801,
        exact_replay_strategy="serialize", chat_template_path=tpl)
    assert a.api_key and len(a.api_key) >= 32
    assert a.api_key != b.api_key            # per-launch, never shared
    assert a.api_key not in " ".join(a.cmd)  # not on argv
    assert all(a.api_key not in v for v in a.env.values())
    assert a.api_key not in repr(a)          # repr=False holds
    assert a.auth_headers == {"Authorization": f"Bearer {a.api_key}"}


def test_lens_launch_mounts_the_auth_middleware_and_requires_loopback(
        tmp_path):
    """The stock --api-key middleware guards /v1|/v2|/inference ONLY
    (pinned server_utils.GUARDED_PREFIX) — /collective_rpc needs our
    middleware, passed through vLLM's supported --middleware hook. And
    a lens launch (which mounts the dev RPC surface) binds loopback
    only."""
    tpl = _chat_template(tmp_path)
    reg = _flat_registry(tmp_path)
    launch = vs.build_server_launch(
        _tpl_spec(serving_launch_flags=["--enforce-eager"]), port=8801,
        lens_registry_dir=reg, exact_replay_strategy="serialize",
        chat_template_path=tpl)
    cmd = list(launch.cmd)
    assert cmd[cmd.index("--middleware") + 1] == vs.AUTH_MIDDLEWARE_CLS
    # dotted class is importable and reads the SAME env the server does
    from sisteer.capeval.serving import vllm_auth
    assert vs.AUTH_MIDDLEWARE_CLS.endswith(
        vllm_auth.SisteerTokenAuthMiddleware.__name__)
    assert vs.API_KEY_ENV == vllm_auth.TOKEN_ENV == "VLLM_API_KEY"
    # lens-less launches don't mount it (their dev surface is OFF)
    plain = vs.build_server_launch(
        _tpl_spec(serving_launch_flags=["--enforce-eager"]), port=8801,
        exact_replay_strategy="serialize", chat_template_path=tpl)
    assert "--middleware" not in plain.cmd
    # a lens launch off loopback refuses at compose time
    with pytest.raises(vs.VllmServerError, match="loopback"):
        vs.build_server_launch(
            _tpl_spec(serving_launch_flags=["--enforce-eager"]), port=8801,
            host="10.0.0.5", lens_registry_dir=reg,
            exact_replay_strategy="serialize", chat_template_path=tpl)
    # ... and a manifest can override neither auth option
    for flags in (["--api-key", "attacker"], ["--middleware=evil.Mw"]):
        with pytest.raises(vs.VllmServerError, match="wrapper-owned"):
            vs.build_server_launch(
                _tpl_spec(serving_launch_flags=flags), port=8801,
                exact_replay_strategy="serialize", chat_template_path=tpl)


def test_launch_server_refuses_an_unauthenticated_lens_launch(tmp_path):
    reg = _flat_registry(tmp_path)
    launch = _fake_launch(tmp_path, registry=reg, api_key=None)
    with pytest.raises(vs.VllmServerError, match="api_key"):
        vs.launch_server(launch, log_path=tmp_path / "srv.log",
                         ready_timeout_s=5, install_timeout_s=5)


# --------------------------------------------------------------------------
# steering installation + attestation (readiness half 2)
# --------------------------------------------------------------------------


def test_expected_install_state_is_computed_from_the_registry(tmp_path):
    """The launcher's independent expectation: loaded through the
    committed WorkerObjectRegistry path, layers/cells/ledger derived
    with the worker's own helpers — a worker cannot attest a different
    registry than the one the launch composed."""
    reg = _flat_registry(tmp_path)
    exp = vs.expected_install_state(reg)
    assert exp["registration_count"] == 1
    assert exp["cells"] == ["none__lens0", "si_safety__cast"]
    assert exp["layers"] == [3]          # the synthetic object's layer
    assert exp["oproj_layers"] == []     # block-site ops only
    assert exp["plan_layer"] == 3
    assert isinstance(exp["ledger_cells"], list)


def _report_from(expected, **over):
    rep = {k: (list(v) if isinstance(v, list) else v)
           for k, v in expected.items()}
    rep.update(over)
    return rep


def test_validate_install_reports_requires_exact_agreement(tmp_path):
    reg = _flat_registry(tmp_path)
    exp = vs.expected_install_state(reg)
    ok = _report_from(exp)
    # one worker, and several workers, all valid:
    assert vs.validate_install_reports([ok], exp) == [ok]
    assert len(vs.validate_install_reports([ok, dict(ok)], exp)) == 2
    # NO reports = no worker installed anything = unsteered server
    with pytest.raises(vs.VllmServerError, match="PER WORKER"):
        vs.validate_install_reports([], exp)
    with pytest.raises(vs.VllmServerError, match="PER WORKER"):
        vs.validate_install_reports(None, exp)
    # a stringified (non-report) result refuses
    with pytest.raises(vs.VllmServerError, match="non-report"):
        vs.validate_install_reports(["LensWorkerError: ..."], exp)
    # every field gates, ONE bad worker among good ones included
    for key, bad in (("registration_count", 2),
                     ("layers", [3, 5]),
                     ("oproj_layers", [1]),
                     ("plan_layer", 0),
                     ("cells", ["si_safety__cast"]),
                     ("ledger_cells", ["bogus"])):
        with pytest.raises(vs.VllmServerError, match=key):
            vs.validate_install_reports(
                [ok, _report_from(exp, **{key: bad})], exp)


def _oproj_arm(layers=(0, 2)):
    """A real lora_all arm (o_proj-site ops) for registry building —
    the blob schema and header the lora serving tests pin."""
    from test_lora_lens_serving import HEADER, _blob_bytes, _lora_all_blob
    return ({"cell_id": "si_safety__lora_r4", "kind": "lora_all",
             "bytes": _blob_bytes(_lora_all_blob(layers=layers))}, HEADER)


def _smoke_registry(reg_dir, arms, header):
    """A loadable worker registry written by the GPU smokes' own writer
    (petri_steered_smoke.build_registry) — the same dir schema
    WorkerObjectRegistry.load verifies on the production export."""
    smoke.build_registry(Path(reg_dir), arms, header)
    return Path(reg_dir)


def test_expected_install_state_is_site_aware(tmp_path):
    """A pure-LoRA registry expects its layers under ``oproj_layers``,
    with NO block-site hooks — the launcher-side half of the 2026-08-28
    readiness fix (before it, every op layer was lumped into ``layers``
    and a correctly-installed LoRA worker could never attest)."""
    arm, header = _oproj_arm()
    reg = _smoke_registry(tmp_path / "reg", [arm], header)
    exp = vs.expected_install_state(reg)
    assert exp["layers"] == []             # no block-site ops anywhere
    assert exp["oproj_layers"] == [0, 2]   # the o_proj site carries them
    assert exp["plan_layer"] == 0          # first-firing hook overall
    assert exp["cells"] == ["si_safety__lora_r4",
                            "si_safety__lora_r4__scale0"]


def test_worker_install_report_validates_by_site_production_parity(tmp_path):
    """PRODUCTION PARITY — the regression for the scale-12 wave's
    readiness deaths (2026-08-28, tasks 7/8/9 of the 20-condition wave):
    the payload a worker's install ACTUALLY emits over the RPC
    (lens_worker.install_report_payload of a real LensWorkerSteering
    install) must validate against the launcher's independent
    expectation, for EVERY hook-site shape. On the pre-fix code the
    o_proj and mixed cases refused (`worker 0: layers = [], expected
    [0, 2]`) while the LoRA smoke's bespoke install check passed — the
    two paths disagreed and nothing failed until the wave."""
    from test_lora_lens_serving import OprojLayer
    from test_petri_lens_worker import FakeView, _ok_header
    from sisteer.capeval.serving import lens_worker as lw

    lora_arm, header = _oproj_arm()
    block_arm = {"cell_id": "si_safety__cast", "kind": la.SYNTHETIC_KIND,
                 "bytes": _synthetic_artifact()}
    registries = {
        "oproj": _smoke_registry(tmp_path / "oproj", [lora_arm], header),
        "block": _smoke_registry(tmp_path / "block", [block_arm], header),
        "mixed": _smoke_registry(tmp_path / "mixed",
                                 [lora_arm, block_arm], header),
    }
    for name, reg_dir in registries.items():
        expected = vs.expected_install_state(reg_dir)
        registry = ls.WorkerObjectRegistry.load(reg_dir)
        view = FakeView(layers=[OprojLayer() for _ in range(4)])
        view.tensor_parallel_size = lambda: 1
        steering = lw.LensWorkerSteering(view=view, registry=registry)
        steering.preflight(object(), header_verifier=_ok_header)
        payload = lw.install_report_payload(steering.install())
        # the production comparison, verbatim — a refusal here means the
        # worker and the launcher disagree about the same registry
        assert vs.validate_install_reports([payload], expected) == \
            [payload], f"{name}: worker report != launcher expectation"
    # and the shapes really differ by site (the mixed registry splits)
    exp = vs.expected_install_state(registries["mixed"])
    assert exp["layers"] == [3] and exp["oproj_layers"] == [0, 2]
    assert exp["plan_layer"] == 0


def test_an_unhooked_oproj_worker_cannot_attest(tmp_path):
    """The wave's exact failure MODE stays a refusal: a worker that
    installed nothing at the o_proj site (or an older worker whose
    report omits the field — read as []) must not pass readiness
    against an o_proj-serving registry."""
    arm, header = _oproj_arm()
    reg = _smoke_registry(tmp_path / "reg", [arm], header)
    exp = vs.expected_install_state(reg)
    ok = _report_from(exp)
    with pytest.raises(vs.VllmServerError, match="oproj_layers"):
        vs.validate_install_reports([_report_from(exp, oproj_layers=[])],
                                    exp)
    legacy = _report_from(exp)
    del legacy["oproj_layers"]
    with pytest.raises(vs.VllmServerError, match="oproj_layers"):
        vs.validate_install_reports([legacy], exp)


_FAKE_SERVER = r"""
import json, os, signal, sys, threading
from http.server import BaseHTTPRequestHandler, HTTPServer

port, model, conf_path = int(sys.argv[1]), sys.argv[2], sys.argv[3]
conf = json.loads(open(conf_path).read())
threading.Timer(60, lambda: os._exit(0)).start()  # never outlive the test
if conf.get("ignore_sigterm"):
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
# the token arrives the way the real server gets it: the spawn env —
# a fake that enforces it proves both the env injection and that every
# client actually sends the Authorization header.
TOKEN = os.environ.get("VLLM_API_KEY", "")


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _authed(self):
        if not conf.get("require_token", True):
            return True
        return self.headers.get("Authorization") == "Bearer " + TOKEN

    def do_GET(self):
        if self.path == "/v1/models":
            if not self._authed():
                self._json(401, {"error": "Unauthorized"})
                return
            # a lora-native fake lists LoRA cards with their root (the
            # frontend's own name->path binding) when the conf carries
            # them; absent, the listing is the base model alone as before.
            self._json(200, {"data": conf.get("models_data")
                                     or [{"id": model}]})
        else:
            self._json(404, {"detail": "nope"})

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(n)
        if self.path == "/collective_rpc" and conf.get("mount_rpc", True):
            if not self._authed():
                self._json(401, {"error": "Unauthorized"})
                return
            # dispatch by RPC method, like the real route: the launcher
            # now makes TWO calls (install, then scheduler-config).
            try:
                method = json.loads(body.decode() or "{}").get("method")
            except ValueError:
                method = None
            if method == "sisteer_scheduler_config":
                results = conf.get("scheduler_results", [])
            elif method == "sisteer_residual_slot_report":
                results = conf.get("residual_results", [])
            else:
                results = conf["results"]
            self._json(200, {"results": results})
            if conf.get("die_after_rpc"):
                self.wfile.flush()
                os._exit(0)
        else:
            self._json(404, {"detail": "Not Found"})


HTTPServer(("127.0.0.1", port), H).serve_forever()
"""


def _free_port():
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


#: the fake launch's composed geometry (any consistent values work; the
#: fake workers report these back unless a test overrides them).
_FAKE_GEOMETRY = {"max_num_seqs": 8, "max_num_batched_tokens": 8 * 16384,
                  "max_model_len": 16384}
#: ... and the WHOLE expected invariant the launcher derives from it
#: (expected_scheduler_state): geometry + partial-prefill pins +
#: effective ceiling + no speculative config.
_FAKE_EXPECTED = dict(_FAKE_GEOMETRY,
                      long_prefill_token_threshold=0,
                      max_num_partial_prefills=1,
                      max_long_partial_prefills=1,
                      effective_max_num_scheduled_tokens=8 * 16384,
                      speculative=None,
                      scheduler_cls=vs.PINNED_DEFAULT_SCHEDULER_CLS,
                      enable_prefix_caching=True)


def _sched_report(**over):
    """One fake worker's resolved report, in the REAL report shape
    (scheduler-owned fields + model_config-sourced max_model_len +
    effective ceiling + speculative state + the resolved KV cache size,
    here exactly covering the 8 x 16384 = 131,072-token budget)."""
    rep = dict(_FAKE_EXPECTED, enable_chunked_prefill=True,
               kv_num_gpu_blocks=8192, kv_block_size=16,
               kv_num_cache_groups=1)
    rep.update(over)
    return rep


#: the engine / LoRA keys a HEALTHY lora-native worker reports (the
#: lens-shaped report above plus native_worker's additions): graphs on,
#: the plugin never imported, bf16 adapters, no sharding.
_NATIVE_ENGINE_REPORT = {
    "enforce_eager": False,
    "cudagraph_mode": "FULL_AND_PIECEWISE",
    "cudagraph_mode_is_none": False,
    "cudagraph_specialize_lora": True,
    "lens_plugin_imported": False,
    "lora_dtype": "torch.bfloat16",
    "lora_fully_sharded": False,
    # the deployment-time LoRA restriction the launcher COMPOSES to the
    # adapted class (the launcher asserts equality)
    "lora_target_modules": ["o_proj"],
    # WHICH GPU model runner the worker constructed: one of
    # serving_attestation.MODEL_RUNNER_CLASSES, recorded in the
    # attestation, refused by readiness when unpinned
    "model_runner_class": "vllm.v1.worker.gpu.model_runner.GPUModelRunner",
    "model_runner_kind": "v2",
}


def _default_capture_sizes(max_num_seqs):
    """vLLM's DEFAULT CUDA-graph capture surface for a decode width
    (config/vllm.py @0.25.1, no speculative tokens): the initial max is
    ``min(2 * S, 512)``; the sizes are ``[1, 2, 4] + range(8, min(max + 1,
    256), 8) + range(256, max + 1, 16)``, de-duplicated and sorted; vLLM
    then pins ``max_cudagraph_capture_size`` to the largest captured size
    (``post_init_cudagraph_sizes`` asserts the two agree). Returns
    ``(sizes, max)`` in that resolved form."""
    mx = min(2 * int(max_num_seqs), 512)
    sizes = [i for i in (1, 2, 4) if i <= mx]
    if mx >= 8:
        sizes += list(range(8, min(mx + 1, 256), 8))
    if mx >= 256:
        sizes += list(range(256, mx + 1, 16))
    sizes = sorted(set(sizes))
    return sizes, sizes[-1]


def _llama_oproj(canonical_key):
    """The Llama spelling of a canonical ``L<N>.<site>.o_proj`` key — the
    fake worker's raw factor AND execution module name for that layer."""
    return f"model.layers.{canonical_key[1:].split('.')[0]}.self_attn.o_proj"


#: the pinned wheel's eight decoder-layer classes the residual slot patches (native_residual.
#: VLLM_DECODER_LAYER_CLASSES), spelled as the worker's report keys them
_RESIDUAL_PATCHED_CLASSES = (
    "vllm.model_executor.models.llama.LlamaDecoderLayer",
    "vllm.model_executor.models.gemma4.Gemma4DecoderLayer",
    "vllm.model_executor.models.qwen3_next.Qwen3NextDecoderLayer",
    "vllm.model_executor.models.qwen3_5.Qwen3_5DecoderLayer",
    "vllm.model_executor.models.nemotron_h.NemotronHMLPDecoderLayer",
    "vllm.model_executor.models.nemotron_h.NemotronHMoEDecoderLayer",
    "vllm.model_executor.models.nemotron_h.NemotronHMambaDecoderLayer",
    "vllm.model_executor.models.nemotron_h.NemotronHAttentionDecoderLayer")


def _native_residual_report(expected_residual, **over):
    """One fake worker's ``sisteer_residual_slot_report`` (``worker_reports.
    residual_slot_report``'s shape), derived from the LAUNCHER'S OWN
    ``serving_attestation.expected_residual_slot_state`` so the fake agrees by
    construction — a test then perturbs exactly one thing through ``over``.
    Unarmed (no payload registered): ``armed False``, no slots, no bank.
    Armed: the op under its ONE name, every decoder class wrapped, every block
    hooked, one loaded slot per adapter carrying a payload."""
    armed = bool(expected_residual["armed"])
    n_layers = expected_residual.get("n_layers_model")
    slots = {}
    for name, info in expected_residual["adapters"].items():
        if info.get("residual_fingerprint") is None:
            continue
        slots[str(info["lora_int_id"])] = {
            "slot": int(info["lora_int_id"]) - 1, "kind": info["kind"], "op": info["op"],
            "layers": list(info["layers"]), "boundaries": list(info["boundaries"]),
            "mode": info["position_mode"], "header_len": None, "skip_bos": info["kind"] != "psr",
            "coeff": 1.0, "fingerprint": info["residual_fingerprint"]}
    rep = {
        "armed": armed, "op_name": ("vllm.sisteer_residual_slot" if armed else None),
        "installs": 1 if armed else 0,
        "patched_classes": ({c: {"forward": "wrapped", "forward_wrapped": True,
                                 "__init__": "wrapped", "__init___wrapped": True}
                             for c in _RESIDUAL_PATCHED_CLASSES} if armed else {}),
        "machinery": {}, "n_layers_hooked": (int(n_layers) if armed else 0),
        "hooked_boundaries": (list(range(int(n_layers))) if armed else []),
        "op_registered": armed, "stage2_position_classes": False, "error": None,
        "bank": ({"n_slots": len(expected_residual["adapters"]), "n_boundaries": int(n_layers),
                  "hidden_size": int(expected_residual.get("hidden_size") or 0),
                  "slots_loaded": len(slots), "mapping_updates": 1, "op_invocations": int(n_layers),
                  "runner_bound": True, "ledger_size": 0, "ledger_fault": False, "ledger_cap": 1024}
                 if armed else None),
        "slots": slots, "n_layers_model": (int(n_layers) if armed else None),
        "runner_bound": armed, "ledger_size": 0, "ledger_fault": False, "ledger_cap": 1024,
    }
    rep.update(over)
    return rep


def _native_sched_report(expected_lora, *, max_lora_rank=8, **over):
    """One fake lora-native worker's report: the geometry report + the
    engine / LoRA-config keys + one loaded-adapter entry per expected
    adapter, derived from the LAUNCHER'S OWN expectation so the fake
    agrees by construction — a test then perturbs exactly one thing. The
    EXECUTION set (the model modules carrying a LoRA wrapper) defaults to
    the union of every expected adapter's module keys — a healthy engine
    with no `--lora-target-modules` restriction wraps every fitted
    o_proj — and the CUDA-graph capture surface to vLLM's default for the
    report's own ``max_num_seqs``; ``over`` may replace either."""
    n = expected_lora["n_adapters"]
    adapters = []
    for name, info in sorted(expected_lora["adapters"].items(),
                             key=lambda kv: kv[1]["lora_int_id"]):
        adapters.append({
            "id": info["lora_int_id"], "rank": info["r"],
            "n_modules": len(info["module_keys"]),
            "module_keys": list(info["module_keys"]),
            "raw_module_names": [_llama_oproj(k) for k in info["module_keys"]],
            "raw_shapes": {},
            "tensor_dtype": "torch.bfloat16",
            "fingerprint_bf16": info["fingerprint_bf16"],
            # the residual identity beside the carrier's (None for a LoRA)
            "residual_fingerprint": info.get("residual_fingerprint"),
        })
    rep = _sched_report(**_NATIVE_ENGINE_REPORT,
                        lora_max_loras=n, lora_max_lora_rank=max_lora_rank,
                        lora_max_cpu_loras=n, lora_loaded_count=n,
                        lora_adapters=adapters)
    rep.update(over)
    exec_keys = sorted({k for a in adapters for k in a["module_keys"]})
    rep.setdefault("lora_execution_module_keys", exec_keys)
    rep.setdefault("lora_execution_modules_n", len(exec_keys))
    rep.setdefault("lora_execution_modules_other", [])
    # the EXACT execution names: the Llama spelling of the (possibly
    # overridden) canonical execution set, and per adapter the binding
    # vLLM's exact lookup resolves — a factor is bound iff an execution
    # module carries its exact name; a healthy fake binds every factor,
    # a test perturbs one spelling
    final_keys = rep.get("lora_execution_module_keys")
    exec_names = ([_llama_oproj(k) for k in final_keys]
                  if isinstance(final_keys, list)
                  and all(isinstance(k, str) for k in final_keys) else [])
    rep.setdefault("lora_execution_module_names", exec_names)
    names = rep["lora_execution_module_names"]
    if isinstance(rep.get("lora_adapters"), list) and isinstance(names, list):
        for a in rep["lora_adapters"]:
            if isinstance(a, dict) and "execution_bindings" not in a:
                factors = set(a.get("raw_module_names") or [])
                a["execution_bindings"] = {m: (m if m in factors else None) for m in names}
    # the EXECUTING STRIDES: each wrapper's (in, out) off the
    # expected adapters' own factor dims (the healthy fake's wrappers are
    # exactly the adapted o_proj modules), keyed by canonical layer; a
    # wrapper no adapter sizes takes the widest known dims; the maximum rides
    # along — a test perturbs one wrapper to a wider stride or another family
    other = rep.get("lora_execution_modules_other")
    if isinstance(names, list) and isinstance(other, list) \
            and "lora_execution_module_strides" not in rep:
        from sisteer.capeval.serving import serving_attestation as _sa
        from sisteer.capeval.serving import worker_reports as _wr
        by_key = {}
        for info in expected_lora["adapters"].values():
            for module, (d_in, d_out) in _sa.adapter_module_dims(Path(info["dir"])).items():
                k = _wr.canonical_module_key(module)
                prev = by_key.get(k, (0, 0))
                by_key[k] = (max(prev[0], d_in), max(prev[1], d_out))
        widest = max(by_key.values(), key=lambda t: max(t)) if by_key else (1, 1)
        strides = {}
        for m in list(names) + list(other):
            if not isinstance(m, str):
                continue                      # a deliberately malformed name: no stride for it
            k = f"L{m.split('layers.')[-1].split('.')[0]}.self_attn.o_proj" if "layers." in m else None
            d_in, d_out = by_key.get(k, widest)
            strides[m] = {"in": int(d_in), "out": int(d_out)}
        rep["lora_execution_module_strides"] = dict(sorted(strides.items()))
    st = rep.get("lora_execution_module_strides")
    if isinstance(st, dict) and st and "lora_execution_module_stride_max" not in rep:
        rep["lora_execution_module_stride_max"] = max(
            max(int(v["in"]), int(v["out"])) for v in st.values())
    # the RESOLVED capture list of a healthy engine under the composer since
    # 2026-09-16: the list build_server_launch hands every lora-native launch
    # (cudagraph_capture_sizes_for(S) — vLLM's default sizes for S plus S; equal
    # to the bare default whenever S is a default size, e.g. the fake's S = 8),
    # taken verbatim by vLLM with the max pinned to its largest entry. A report
    # of vLLM's BARE default for an S that is not a default size (121, 51) is
    # the incident's surface — a FULL max below S — and readiness now refuses it
    # (a test perturbs the list to reproduce exactly that)
    sizes = list(vs.cudagraph_capture_sizes_for(rep["max_num_seqs"]))
    rep.setdefault("cudagraph_capture_sizes", sizes)
    rep.setdefault("max_cudagraph_capture_size", max(rep["cudagraph_capture_sizes"])
                   if isinstance(rep["cudagraph_capture_sizes"], list) and rep["cudagraph_capture_sizes"]
                   else sizes[-1])
    # THE FULL SURFACE (2026-09-16): the dispatcher's FULL keys are the capture
    # sizes <= S (initialize_cudagraph_keys @0.25.1) — derived from the report's
    # FINAL capture list so a perturbed list perturbs the FULL surface with it;
    # a test overrides either key directly
    final_sizes = rep["cudagraph_capture_sizes"]
    S = rep["max_num_seqs"]
    full = (sorted(x for x in final_sizes
                   if isinstance(x, int) and not isinstance(x, bool) and x <= S)
            if isinstance(final_sizes, list) and isinstance(S, int) else [])
    rep.setdefault("cudagraph_full_capture_sizes", full)
    rep.setdefault("cudagraph_full_capture_max", full[-1] if full else None)
    rep.setdefault("uniform_decode_query_len", 1)
    rep.setdefault("cudagraph_dispatcher_mode", rep.get("cudagraph_mode"))
    # WHICH runner surface the FULL keys came from: the report's own class kind (v2 here, as
    # gemma / llama run on the real engine; a test perturbs the pair to disagree)
    rep.setdefault("model_runner_surface", rep.get("model_runner_kind")
                   if rep.get("model_runner_kind") in ("v1", "v2") else "v2")
    rep.setdefault("cudagraph_lora_capture_cases", [0, 1] if rep["model_runner_surface"] == "v2" else None)
    return rep


def _fake_lora_native_launch(tmp_path, *, adapters, scheduler_results=None,
                             models_data=None, max_lora_rank=8,
                             mount_rpc=True, require_token=True,
                             die_after_rpc=False, api_key=None,
                             model="meta-llama/Llama-3.1-8B-Instruct"):
    """A LORA-NATIVE ServerLaunch whose child is the scripted stand-in
    (`_FAKE_SERVER`): real authenticated /v1/models readiness listing the
    LoRA cards with their roots, real /collective_rpc transport answering
    `sisteer_scheduler_config` with a canned per-worker report. Defaults
    are a HEALTHY server for `adapters` ({name: dir}); pass
    `scheduler_results` / `models_data` to perturb one fact. What it can
    NOT prove (GPU-VERIFY, the gemma canary's job): the real frontend's
    id order, the real worker's LoRA surfaces, real bf16 tensors."""
    import secrets
    import sys

    from sisteer.capeval.serving import serving_attestation as sa

    names = tuple(sorted(adapters))
    dirs = {n: str(Path(p).resolve()) for n, p in adapters.items()}
    if models_data is None:
        models_data = ([{"id": model}]
                       + [{"id": n, "root": dirs[n]} for n in names])
    if scheduler_results is None:
        expected = sa.expected_native_lora_state_for(names, dirs)
        scheduler_results = [_native_sched_report(
            expected, max_lora_rank=max_lora_rank)]
    # the adapted class + the kernel index ceiling's D, exactly as
    # build_server_launch derives them (the readiness check holds the
    # resolved restriction and every executing wrapper's stride to them)
    from sisteer.capeval.serving import worker_reports as _wr
    targets = tuple(sorted({t for d in dirs.values()
                            for t in _wr.read_adapter_config(Path(d))["target_modules"]}))
    max_dim, _ = sa.max_module_dim({n: sa.adapter_module_dims(Path(d)) for n, d in dirs.items()})
    script = tmp_path / "fake_lora_server.py"
    script.write_text(_FAKE_SERVER)
    # THE RESIDUAL SLOT's readiness RPC (native-psr-serving §1.4): the fake worker's report
    # derived from the launcher's own expectation over the same dirs (unarmed for plain LoRAs)
    residual_results = [_native_residual_report(sa.expected_residual_slot_state(names, dirs))]
    conf = tmp_path / "lora_conf.json"
    conf.write_text(json.dumps({"results": [],
                                "scheduler_results": scheduler_results,
                                "residual_results": residual_results,
                                "models_data": models_data,
                                "mount_rpc": mount_rpc,
                                "require_token": require_token,
                                "die_after_rpc": die_after_rpc}))
    port = _free_port()
    return vs.ServerLaunch(
        cmd=(sys.executable, str(script), str(port), model, str(conf)),
        env={}, host="127.0.0.1", port=port, model=model,
        lens_registry_dir=None,
        api_key=(api_key if api_key is not None
                 else secrets.token_urlsafe(24)),
        max_num_seqs=_FAKE_GEOMETRY["max_num_seqs"],
        max_num_batched_tokens=_FAKE_GEOMETRY["max_num_batched_tokens"],
        max_model_len=_FAKE_GEOMETRY["max_model_len"],
        prefix_caching=True,
        lora_native=True, lora_modules=names, lora_adapter_dirs=dirs,
        max_loras=len(names), max_lora_rank=max_lora_rank,
        max_cpu_loras=len(names), revision=REV,
        lora_target_modules=targets, lora_max_module_dim=max_dim,
        lora_kernel_index_ceiling=sa.lora_kernel_index_ceiling(max_dim),
        # the composed capture list (2026-09-16), as build_server_launch derives it: readiness
        # holds every entry to the resolved list (the fake's default list for S=8 is exactly it)
        cudagraph_capture_sizes=vs.cudagraph_capture_sizes_for(_FAKE_GEOMETRY["max_num_seqs"]))


def _fake_launch(tmp_path, *, registry, results=None, mount_rpc=True,
                 require_token=True, die_after_rpc=False,
                 ignore_sigterm=False, api_key="not-set",
                 scheduler_results=None, geometry="composed"):
    """A ServerLaunch whose child is a scripted stand-in for the pinned
    server: real /v1/models readiness, real /collective_rpc transport,
    real token enforcement (read from the spawn env exactly like the
    stock middleware), canned per-worker install reports — the
    strongest CPU-runnable integration of the
    readiness+auth+install+attestation path. What it can NOT prove
    (GPU-VERIFY, the serving gate's job): the real vLLM mounting the
    route and middleware, the workers importing the extension, and
    hooks steering real forwards."""
    import secrets
    import sys

    script = tmp_path / "fake_server.py"
    script.write_text(_FAKE_SERVER)
    if scheduler_results is None:
        # one clean per-worker geometry report per install report, so a
        # multi-worker install test gets a multi-worker geometry too.
        scheduler_results = [_sched_report() for _ in (results or [None])]
    conf = tmp_path / "conf.json"
    conf.write_text(json.dumps({"results": results or [],
                                "scheduler_results": scheduler_results,
                                "mount_rpc": mount_rpc,
                                "require_token": require_token,
                                "die_after_rpc": die_after_rpc,
                                "ignore_sigterm": ignore_sigterm}))
    port = _free_port()
    model = "meta-llama/Llama-3.1-8B-Instruct"
    if api_key == "not-set":
        api_key = secrets.token_urlsafe(24)
    geom = _FAKE_GEOMETRY if geometry == "composed" else {}
    return vs.ServerLaunch(
        cmd=(sys.executable, str(script), str(port), model, str(conf)),
        env={}, host="127.0.0.1", port=port, model=model,
        lens_registry_dir=(str(registry) if registry else None),
        api_key=api_key,
        max_num_seqs=geom.get("max_num_seqs"),
        max_num_batched_tokens=geom.get("max_num_batched_tokens"),
        max_model_len=geom.get("max_model_len"),
        prefix_caching=(True if geometry == "composed" else None))


def test_launch_server_installs_attests_and_only_then_is_ready(tmp_path):
    """The production path end-to-end over real HTTP: readiness =
    AUTHENTICATED /v1/models for THIS model AND validated per-worker
    install reports over the AUTHENTICATED RPC route (the fake enforces
    the Bearer token it reads from its spawn env, exactly like the
    stock middleware — so this also proves the env injection and that
    both clients send the header), then the attestation lands next to
    the log, bound to the live child."""
    import socket as socket_mod

    reg = _flat_registry(tmp_path)
    exp = vs.expected_install_state(reg)
    launch = _fake_launch(tmp_path, registry=reg,
                          results=[_report_from(exp), _report_from(exp)])
    log = tmp_path / "srv.log"
    proc = vs.launch_server(launch, log_path=log, ready_timeout_s=30,
                            install_timeout_s=10)
    try:
        att_text = (tmp_path / vs.STEERING_ATTESTATION_FILE).read_text()
        att = json.loads(att_text)
        assert att["n_workers"] == 2
        assert att["registry_dir"] == str(reg)
        assert att["expected"] == exp
        assert att["rpc_method"] == vs.INSTALL_RPC_METHOD
        assert [r["cells"] for r in att["worker_reports"]] == [
            exp["cells"], exp["cells"]]
        # the attestation binds THIS launch and nothing else:
        assert att["server_pid"] == proc.pid
        assert att["hostname"] == socket_mod.gethostname()
        assert att["port"] == launch.port
        assert att["argv_sha256"] == cm.sha256_hex16(
            "\x00".join(launch.cmd).encode("utf-8"))
        assert att["registry_index_sha16"] == cm.sha256_hex16(
            (reg / "registry.json").read_bytes())
        assert len(att["nonce"]) == 32
        assert "slurm_job_id" in att
        # readiness half 3: the RESOLVED scheduler geometry was read
        # over the same authenticated route and matches the composed
        # no-chunking sizing — runtime evidence in the attestation.
        geom = att["scheduler_geometry"]
        assert geom["rpc_method"] == vs.SCHEDULER_RPC_METHOD
        assert geom["expected"] == _FAKE_EXPECTED
        assert geom["n_workers"] == len(geom["worker_reports"]) >= 1
        for rep in geom["worker_reports"]:
            for k, v in _FAKE_EXPECTED.items():
                assert rep[k] == v
        # ... and NEVER carries the secret
        assert launch.api_key not in att_text
    finally:
        proc.terminate()
        proc.wait(timeout=10)


def test_launch_server_refuses_readiness_on_a_bad_install_report(tmp_path):
    """An HTTP-healthy server whose worker attests the WRONG install
    (registration_count 2 here) is terminated, and launch_server raises
    instead of returning — unsteered-but-healthy can never pass."""
    reg = _flat_registry(tmp_path)
    exp = vs.expected_install_state(reg)
    launch = _fake_launch(
        tmp_path, registry=reg,
        results=[_report_from(exp, registration_count=2)])
    with pytest.raises(vs.VllmServerError, match="registration_count"):
        vs.launch_server(launch, log_path=tmp_path / "srv.log",
                         ready_timeout_s=30, install_timeout_s=10)
    assert not (tmp_path / vs.STEERING_ATTESTATION_FILE).exists()


def test_launch_server_refuses_readiness_without_the_install_route(tmp_path):
    """A server that answers /v1/models but 404s the collective_rpc
    route (dev mode off / wrong vLLM) can never be attested — readiness
    must refuse, not degrade to serving unsteered."""
    reg = _flat_registry(tmp_path)
    launch = _fake_launch(tmp_path, registry=reg, mount_rpc=False)
    with pytest.raises(vs.VllmServerError, match="not mounted"):
        vs.launch_server(launch, log_path=tmp_path / "srv.log",
                         ready_timeout_s=30, install_timeout_s=10)


def test_validate_scheduler_reports_requires_exact_agreement():
    exp = dict(_FAKE_EXPECTED)
    ok = _sched_report()
    assert vs.validate_scheduler_reports([ok], exp) == [ok]
    assert len(vs.validate_scheduler_reports([ok, dict(ok)], exp)) == 2
    with pytest.raises(vs.VllmServerError, match="PER WORKER"):
        vs.validate_scheduler_reports([], exp)
    with pytest.raises(vs.VllmServerError, match="PER WORKER"):
        vs.validate_scheduler_reports(None, exp)
    with pytest.raises(vs.VllmServerError, match="non-report"):
        vs.validate_scheduler_reports(["LensWorkerError: ..."], exp)
    # a silent clamp on ANY worker, any INT field, refuses — this is
    # the smoke's "silent clamp re-enables mid-prompt chunking"
    # assertion, applied to the whole invariant at readiness.
    for key in _FAKE_GEOMETRY:
        with pytest.raises(vs.VllmServerError, match=key):
            vs.validate_scheduler_reports(
                [ok, _sched_report(**{key: 2048})], exp)
    with pytest.raises(vs.VllmServerError, match="unreadable"):
        vs.validate_scheduler_reports(
            [_sched_report(max_num_seqs=None)], exp)
    # enable_chunked_prefill is recorded, never asserted (V1 forces it)
    assert vs.validate_scheduler_reports(
        [_sched_report(enable_chunked_prefill=None)], exp)


def test_validate_scheduler_reports_refuses_kv_under_coverage():
    """THE SILENT-WIDTH-LIE (scale-12 gemma wave, job 232411): the
    server requested max_num_seqs=48 at 16384 and came up cleanly over
    a 146,118-token KV cache — 18.6% of the 786,432-token budget, so
    vLLM scheduled ~9 sequences and preempted the rest, delivering a
    fifth of the requested width with no error anywhere. The smoke's
    per-engine coverage gate has asserted this since the 32x16384 llama
    wall; readiness now holds every lens launch to the same verdict:
    the requested concurrency must exist in MEMORY, not just argv."""
    exp = dict(_FAKE_EXPECTED)
    # exactly covering passes (the base fake: 8192 x 16 == 8 x 16384)
    assert vs.validate_scheduler_reports([_sched_report()], exp)
    # one block short: refuses, naming both numbers
    with pytest.raises(vs.VllmServerError,
                       match="argv but not in memory"):
        vs.validate_scheduler_reports(
            [_sched_report(kv_num_gpu_blocks=8191)], exp)
    # the measured gemma shape: 146,118 tokens vs a 131,072 budget DOES
    # cover here — scale the fake down to gemma's ratio to prove the
    # message carries the numbers (18.6% coverage refuses)
    with pytest.raises(vs.VllmServerError, match="131072-token"):
        vs.validate_scheduler_reports(
            [_sched_report(kv_num_gpu_blocks=1524)], exp)  # 24,384 tokens
    # unreadable/missing facts refuse — an old extension's report has no
    # kv fields, and "cannot verify" must never pass as coverage
    with pytest.raises(vs.VllmServerError, match="could not be read"):
        rep = _sched_report()
        del rep["kv_num_gpu_blocks"], rep["kv_block_size"]
        vs.validate_scheduler_reports([rep], exp)
    with pytest.raises(vs.VllmServerError, match="could not be read"):
        vs.validate_scheduler_reports(
            [_sched_report(kv_block_size=True)], exp)
    # ... and only the under-covered WORKER is named, others pass
    with pytest.raises(vs.VllmServerError, match="worker 1"):
        vs.validate_scheduler_reports(
            [_sched_report(), _sched_report(kv_num_gpu_blocks=100)], exp)


def test_validate_scheduler_reports_catches_non_budget_chunking():
    """THE BIG-BUDGET-STILL-CHUNKS CASES. max_num_batched_tokens is
    exactly the composed 131072, yet each of these settings would still
    split or extend scheduled work: a long-prefill threshold chunks any
    prompt longer than itself; >1 partial prefills silently auto-set
    that threshold; speculative decoding schedules q>1 decode steps and
    shrinks the effective per-step ceiling. Every one must refuse."""
    exp = dict(_FAKE_EXPECTED)
    with pytest.raises(vs.VllmServerError,
                       match="long_prefill_token_threshold"):
        vs.validate_scheduler_reports(
            [_sched_report(long_prefill_token_threshold=655)], exp)
    with pytest.raises(vs.VllmServerError,
                       match="max_num_partial_prefills"):
        vs.validate_scheduler_reports(
            [_sched_report(max_num_partial_prefills=4,
                           max_long_partial_prefills=2,
                           long_prefill_token_threshold=655)], exp)
    with pytest.raises(vs.VllmServerError, match="speculative"):
        vs.validate_scheduler_reports(
            [_sched_report(
                speculative="SpeculativeConfig(num_speculative_tokens=3)",
                effective_max_num_scheduled_tokens=8 * 16384)], exp)
    with pytest.raises(vs.VllmServerError,
                       match="effective_max_num_scheduled_tokens"):
        vs.validate_scheduler_reports(
            [_sched_report(effective_max_num_scheduled_tokens=8192)], exp)
    # a CUSTOM scheduler class interprets every knob above, so even a
    # report whose numbers all match refuses under any other class —
    # the async-switched stock scheduler included:
    with pytest.raises(vs.VllmServerError, match="scheduler_cls"):
        vs.validate_scheduler_reports(
            [_sched_report(scheduler_cls="evil.Scheduler")], exp)
    with pytest.raises(vs.VllmServerError, match="scheduler_cls"):
        vs.validate_scheduler_reports(
            [_sched_report(scheduler_cls=(
                "vllm.v1.core.sched.async_scheduler.AsyncScheduler"))],
            exp)
    with pytest.raises(vs.VllmServerError, match="scheduler_cls"):
        vs.validate_scheduler_reports(
            [_sched_report(scheduler_cls=None)], exp)  # unresolvable


def test_validate_scheduler_reports_refuses_missing_keys_as_drift():
    """A worker whose report LACKS an invariant key (an older extension,
    a config-surface rename) refuses loudly — an invariant that cannot
    be read is not an invariant that holds."""
    exp = dict(_FAKE_EXPECTED)
    for key in ("speculative", "effective_max_num_scheduled_tokens",
                "long_prefill_token_threshold", "max_model_len",
                "scheduler_cls"):
        rep = _sched_report()
        del rep[key]
        with pytest.raises(vs.VllmServerError, match="MISSING"):
            vs.validate_scheduler_reports([rep], exp)


def test_expected_scheduler_state_refuses_a_geometry_less_launch(tmp_path):
    launch = _fake_launch(tmp_path, registry=None, geometry="none")
    with pytest.raises(vs.VllmServerError, match="no composed scheduler"):
        vs.expected_scheduler_state(launch)


def test_launch_server_refuses_readiness_on_a_clamped_geometry(tmp_path):
    """An HTTP-healthy, correctly-installed server whose RESOLVED
    budget came back clamped (8192 — vLLM's own H200 server default)
    must be terminated and refused: argv said 131072, the engine says
    otherwise, and a chunking server bricks plan-needing arms."""
    reg = _flat_registry(tmp_path)
    exp = vs.expected_install_state(reg)
    launch = _fake_launch(
        tmp_path, registry=reg, results=[_report_from(exp)],
        scheduler_results=[_sched_report(max_num_batched_tokens=8192)])
    with pytest.raises(vs.VllmServerError, match="silent clamp"):
        vs.launch_server(launch, log_path=tmp_path / "srv.log",
                         ready_timeout_s=30, install_timeout_s=10)
    assert not (tmp_path / vs.STEERING_ATTESTATION_FILE).exists()


def test_launch_server_refuses_a_chunking_threshold_or_spec_state(tmp_path):
    """End-to-end over real HTTP: a worker whose budget is EXACTLY the
    composed one but whose resolved long-prefill threshold (or
    speculative state) would still chunk/extend scheduled work is
    refused readiness — the invariant is more than the budget."""
    reg = _flat_registry(tmp_path)
    exp = vs.expected_install_state(reg)
    launch = _fake_launch(
        tmp_path, registry=reg, results=[_report_from(exp)],
        scheduler_results=[_sched_report(long_prefill_token_threshold=655)])
    with pytest.raises(vs.VllmServerError,
                       match="long_prefill_token_threshold"):
        vs.launch_server(launch, log_path=tmp_path / "srv.log",
                         ready_timeout_s=30, install_timeout_s=10)
    assert not (tmp_path / vs.STEERING_ATTESTATION_FILE).exists()
    launch = _fake_launch(
        tmp_path, registry=reg, results=[_report_from(exp)],
        scheduler_results=[_sched_report(
            speculative="SpeculativeConfig(num_speculative_tokens=3)")])
    with pytest.raises(vs.VllmServerError, match="speculative"):
        vs.launch_server(launch, log_path=tmp_path / "srv.log",
                         ready_timeout_s=30, install_timeout_s=10)
    assert not (tmp_path / vs.STEERING_ATTESTATION_FILE).exists()
    # ... and a worker running any scheduler class other than the
    # pinned stock one refuses even with every number matching:
    launch = _fake_launch(
        tmp_path, registry=reg, results=[_report_from(exp)],
        scheduler_results=[_sched_report(scheduler_cls="evil.Scheduler")])
    with pytest.raises(vs.VllmServerError, match="scheduler_cls"):
        vs.launch_server(launch, log_path=tmp_path / "srv.log",
                         ready_timeout_s=30, install_timeout_s=10)
    assert not (tmp_path / vs.STEERING_ATTESTATION_FILE).exists()


def test_launch_server_refuses_a_lens_launch_without_composed_geometry(
        tmp_path):
    """Fail-closed: a lens launch that skipped build_server_launch (no
    composed geometry to verify against) must refuse readiness rather
    than skip the verification — "cannot verify" never passes."""
    reg = _flat_registry(tmp_path)
    exp = vs.expected_install_state(reg)
    launch = _fake_launch(tmp_path, registry=reg,
                          results=[_report_from(exp)], geometry="none")
    with pytest.raises(vs.VllmServerError, match="no composed scheduler"):
        vs.launch_server(launch, log_path=tmp_path / "srv.log",
                         ready_timeout_s=30, install_timeout_s=10)
    assert not (tmp_path / vs.STEERING_ATTESTATION_FILE).exists()


def test_launch_server_lensless_launch_needs_no_install(tmp_path):
    """A PLAIN launch (neither lens nor lora-native) keeps the old
    contract: authenticated /v1/models readiness alone, no install RPC,
    no attestation file of EITHER kind — and it CLEARS a stale
    attestation of either kind some earlier launch left, so nothing can
    read yesterday's install (or yesterday's native adapters) as
    today's. (Moved deliberately, native-multilora design §4.9: the
    lens-less fall-through is now the plain mode beside two explicit
    attesting ones.)"""
    stale = tmp_path / vs.STEERING_ATTESTATION_FILE
    stale.write_text(json.dumps({"nonce": "stale", "server_pid": -1}))
    stale_native = tmp_path / vs.SERVING_ATTESTATION_FILE
    stale_native.write_text(json.dumps({"kind": "lora_native",
                                        "nonce": "stale"}))
    launch = _fake_launch(tmp_path, registry=None)
    proc = vs.launch_server(launch, log_path=tmp_path / "srv.log",
                            ready_timeout_s=30, install_timeout_s=10)
    try:
        assert not stale.exists()  # cleared at launch start, none written
        assert not stale_native.exists()
    finally:
        proc.terminate()
        proc.wait(timeout=10)


def test_launch_server_lora_native_writes_serving_attestation_only(tmp_path):
    """A lora-native launch writes `serving_attestation.json` (kind
    lora_native) and NEVER `steering_attestation.json`: the lens file's
    three readers parse that path with lens expectations, so the native
    record is a separate file with a discriminator. A stale lens file
    is cleared at start like on every launch. The full readiness
    contract is exercised in tests/test_lora_native_launch.py."""
    bank = tmp_path / "bank"
    adapters = {"none__lora0": _tiny_peft_adapter(bank / "z", r=1),
                "si_safety__lora_r4": _tiny_peft_adapter(bank / "a", r=4)}
    stale = tmp_path / vs.STEERING_ATTESTATION_FILE
    stale.write_text(json.dumps({"nonce": "stale", "server_pid": -1}))
    launch = _fake_lora_native_launch(tmp_path, adapters=adapters)
    proc = vs.launch_server(launch, log_path=tmp_path / "srv.log",
                            ready_timeout_s=30, install_timeout_s=10)
    try:
        assert not stale.exists()
        att = json.loads((tmp_path / vs.SERVING_ATTESTATION_FILE).read_text())
        assert att["kind"] == "lora_native"
        assert att["server_pid"] == proc.pid
        assert att["name_binding"]["asserted"] is True
    finally:
        proc.terminate()
        proc.wait(timeout=10)


def test_launch_server_clears_stale_attestation_even_on_failure(tmp_path):
    """A failing lens launch must not leave EITHER a stale attestation
    or its own: the stale one is cleared at start, none is written on
    the failure path."""
    reg = _flat_registry(tmp_path)
    exp = vs.expected_install_state(reg)
    stale = tmp_path / vs.STEERING_ATTESTATION_FILE
    stale.write_text(json.dumps({"nonce": "stale"}))
    launch = _fake_launch(
        tmp_path, registry=reg,
        results=[_report_from(exp, registration_count=2)])
    with pytest.raises(vs.VllmServerError):
        vs.launch_server(launch, log_path=tmp_path / "srv.log",
                         ready_timeout_s=30, install_timeout_s=10)
    assert not stale.exists()


def test_attestations_are_per_launch_distinct(tmp_path):
    """Two launches into the same directory produce attestations a
    reader can never confuse: fresh nonce and pid each time (the second
    launch clears and re-publishes)."""
    reg = _flat_registry(tmp_path)
    exp = vs.expected_install_state(reg)
    seen = []
    for _ in range(2):
        launch = _fake_launch(tmp_path, registry=reg,
                              results=[_report_from(exp)])
        proc = vs.launch_server(launch, log_path=tmp_path / "srv.log",
                                ready_timeout_s=30, install_timeout_s=10)
        try:
            att = json.loads(
                (tmp_path / vs.STEERING_ATTESTATION_FILE).read_text())
            seen.append((att["nonce"], att["server_pid"]))
        finally:
            proc.terminate()
            proc.wait(timeout=10)
    assert seen[0][0] != seen[1][0]  # nonce
    assert seen[0][1] != seen[1][1]  # pid


def test_launch_server_dead_child_after_install_invalidates(tmp_path):
    """The child answers the install RPC and then dies: some later
    readiness step (the scheduler-geometry RPC now sits directly after
    the install; the post-publish liveness check backstops it) must
    catch it, leave NO attestation behind, and refuse readiness — a
    truthful-looking attestation for a dead server is the lie this step
    exists to prevent."""
    reg = _flat_registry(tmp_path)
    exp = vs.expected_install_state(reg)
    launch = _fake_launch(tmp_path, registry=reg,
                          results=[_report_from(exp)], die_after_rpc=True)
    with pytest.raises(vs.VllmServerError,
                       match="could not be reached|after the steering "
                             "install"):
        vs.launch_server(launch, log_path=tmp_path / "srv.log",
                         ready_timeout_s=30, install_timeout_s=10)
    assert not (tmp_path / vs.STEERING_ATTESTATION_FILE).exists()


def test_shutdown_escalates_to_kill_on_a_sigterm_ignoring_child(
        tmp_path, monkeypatch):
    """The centralized shutdown is BOUNDED: a child that ignores
    SIGTERM (a wedged engine) is killed after the wait, so a failed
    launch can never leave a zombie holding the port and the GPU."""
    import socket as socket_mod
    import time as time_mod

    monkeypatch.setattr(vs, "CHILD_STOP_TIMEOUT_S", 0.5)
    reg = _flat_registry(tmp_path)
    exp = vs.expected_install_state(reg)
    launch = _fake_launch(
        tmp_path, registry=reg, ignore_sigterm=True,
        results=[_report_from(exp, registration_count=2)])
    with pytest.raises(vs.VllmServerError, match="registration_count"):
        vs.launch_server(launch, log_path=tmp_path / "srv.log",
                         ready_timeout_s=30, install_timeout_s=10)
    # the SIGKILL landed: the port frees despite the ignored SIGTERM
    deadline = time_mod.monotonic() + 10
    while time_mod.monotonic() < deadline:
        try:
            with socket_mod.create_connection(("127.0.0.1", launch.port),
                                              timeout=0.5):
                time_mod.sleep(0.2)
        except OSError:
            break
    else:  # pragma: no cover — the defect this test exists to catch
        raise AssertionError("child survived terminate->kill")


def test_launch_server_refuses_an_already_listening_port(tmp_path):
    """P9: a stale process on the port would return 200 from /v1/models
    while the new child dies on its bind — readiness must be
    instance-specific, starting with refusing an occupied port before
    spawning anything."""
    import http.server
    import threading

    srv = http.server.HTTPServer(
        ("127.0.0.1", 0), http.server.BaseHTTPRequestHandler)
    port = srv.server_address[1]
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    try:
        launch = vs.ServerLaunch(cmd=("sleep", "30"), env={},
                                 host="127.0.0.1", port=port,
                                 model="meta-llama/Llama-3.1-8B-Instruct")
        with pytest.raises(vs.VllmServerError, match="already listening"):
            vs.launch_server(launch, log_path=tmp_path / "srv.log")
    finally:
        srv.shutdown()
        thread.join(timeout=5)


def test_readiness_requires_the_launched_model_in_the_listing():
    model = "meta-llama/Llama-3.1-8B-Instruct"
    assert vs._readiness_lists_model({"data": [{"id": model}]}, model)
    assert not vs._readiness_lists_model({"data": [{"id": "gpt-4o"}]}, model)
    assert not vs._readiness_lists_model({"data": []}, model)
    assert not vs._readiness_lists_model("not-a-mapping", model)


# --------------------------------------------------------------------------
# the zero-adapter export (the none__lora0 leg)
# --------------------------------------------------------------------------


def test_zero_adapter_export_and_verify(tmp_path):
    adapter = lx.export_zero_adapter(
        tmp_path / "none__lora0",
        base_model="meta-llama/Llama-3.1-8B-Instruct", rank=8,
        target_modules={"model.layers.15.self_attn.q_proj": (64, 64),
                        "model.layers.15.self_attn.v_proj": (64, 16)})
    lx.verify_zero_adapter(adapter)  # all-zero, marked, loadable
    from safetensors.torch import load_file, save_file

    tensors = load_file(str(adapter / lx.ADAPTER_WEIGHTS))
    assert len(tensors) == 4  # A+B per module
    assert all(torch.all(t == 0) for t in tensors.values())
    a = tensors["base_model.model.model.layers.15.self_attn.q_proj"
                ".lora_A.weight"]
    assert tuple(a.shape) == (8, 64)
    cfg = json.loads((adapter / lx.ADAPTER_CONFIG).read_text())
    assert cfg["r"] == 8 and cfg["petri_zero_adapter"] is True
    # a NON-zero tensor must fail verification (it would not be a no-op)
    tensors[list(tensors)[0]] = torch.ones_like(tensors[list(tensors)[0]])
    save_file(tensors, str(adapter / lx.ADAPTER_WEIGHTS))
    with pytest.raises(lx.LoraExportError, match="non-zero"):
        lx.verify_zero_adapter(adapter)


def test_genuine_export_refuses_a_kind_it_cannot_convert():
    """The export is implemented now — the conversion itself is covered
    by tests/test_petri_lora_export.py. What this pins is the
    default-deny edge: a kind with no derived conversion refuses loudly
    rather than emitting an approximate adapter."""
    with pytest.raises(lx.LoraExportError):
        lx.export_genuine_adapter({"kind": "not_a_weight_level_kind"},
                                  b"bytes", Path("/nonexistent"),
                                  base_model="TESTMODEL")


# --------------------------------------------------------------------------
# the OpenAI shim over a FAKE HF engine (translation + reference role)
# --------------------------------------------------------------------------


class FakeTokenizer:
    def apply_chat_template(self, messages, tokenize=False,
                            add_generation_prompt=True):
        assert tokenize is False and add_generation_prompt is True
        return "".join(f"<{m['role']}>{m['content']}" for m in messages) \
            + "<assistant>"

    def __call__(self, text):
        return {"input_ids": list(range(len(text.split())))}


class FakeHFEngine:
    """Duck-typed HFEngine: records gen_cfg; emits deterministic tokens."""

    def __init__(self):
        self.tokenizer = FakeTokenizer()
        self.model = None
        self.calls = []

    def generate(self, prompts, arm=None, gen_cfg=None):
        self.calls.append({"prompts": list(prompts), "arm": arm,
                           "gen_cfg": dict(gen_cfg or {})})
        outs = []
        for p in prompts:
            ids = [len(p) % 97, 7, 11]
            outs.append(GenOut(text=f"reply-to:{p[-12:]}",
                               gen_token_logprobs=[-0.1] * len(ids),
                               ended_with_eos=True, gen_token_ids=ids))
        return outs


def _shim():
    eng = FakeHFEngine()
    shim = osh.OpenAIShimEngine(
        "meta-llama/Llama-3.1-8B-Instruct", revision=REV,
        identity="hf_shim+apply+exact_replay", hf_engine=eng)
    shim.load()  # no-op with the injected engine
    return shim, eng


def test_shim_translates_openai_chat_to_greedy_hf(tmp_path):
    shim, eng = _shim()
    out = shim.chat_completions({
        "model": "petri__none__hf",
        "messages": [{"role": "system", "content": "be terse"},
                     {"role": "user", "content": "hi"}],
        "max_tokens": 32, "temperature": 0, "stop": ["\n\n"]})
    call = eng.calls[-1]
    assert call["gen_cfg"] == {"max_new_tokens": 32, "chat": False,
                               "greedy": True, "until": ["\n\n"]}
    assert call["prompts"][0].startswith("<system>be terse<user>hi")
    choice = out["choices"][0]
    assert choice["message"]["role"] == "assistant"
    assert choice["finish_reason"] == "stop"
    assert out["model"] == "petri__none__hf"
    assert out["petri"]["serving_identity"] == "hf_shim+apply+exact_replay"
    assert out["usage"]["completion_tokens"] == 3
    # sampling requests map honestly (the exception path serves what the
    # audit asks; the REFERENCE role uses reference_generate, greedy)
    shim.chat_completions({
        "model": "m", "messages": [{"role": "user", "content": "x"}],
        "max_tokens": 8, "temperature": 0.9, "top_p": 0.95})
    assert eng.calls[-1]["gen_cfg"]["greedy"] is False
    assert eng.calls[-1]["gen_cfg"]["temperature"] == 0.9


def test_shim_refusals_are_loud():
    shim, _ = _shim()
    base = {"model": "m",
            "messages": [{"role": "user", "content": "x"}]}
    with pytest.raises(osh.ShimError, match="max_tokens"):
        shim.chat_completions(base)  # the shim never guesses a budget
    with pytest.raises(osh.ShimError, match="tool"):
        shim.chat_completions({**base, "max_tokens": 8,
                               "tools": [{"type": "function"}]})
    with pytest.raises(osh.ShimError, match="stream"):
        shim.chat_completions({**base, "max_tokens": 8, "stream": True})
    with pytest.raises(osh.ShimError, match="unknown"):
        shim.chat_completions({**base, "max_tokens": 8, "logprobs": True})
    with pytest.raises(osh.ShimError, match="n=1"):
        shim.chat_completions({**base, "max_tokens": 8, "n": 3})
    with pytest.raises(osh.ShimError, match="messages"):
        shim.chat_completions({"model": "m", "messages": [],
                               "max_tokens": 8})
    with pytest.raises(osh.ShimError, match="hf_shim"):
        osh.OpenAIShimEngine("m", revision=None,
                             identity="proxy_vllm+lens+exact_replay")


def test_preloaded_engine_cannot_bypass_the_fp32_reference(monkeypatch):
    """G6: the injection seam is not a dtype bypass — a preloaded engine
    carrying bf16 parameters (or declaring a non-fp32 inference dtype)
    refuses at load(); fp32 tensors pass."""
    class TinyModule(torch.nn.Module):
        def __init__(self, dtype):
            super().__init__()
            self.linear = torch.nn.Linear(2, 2).to(dtype)

    bf16 = FakeHFEngine()
    bf16.model = TinyModule(torch.bfloat16)
    shim = osh.OpenAIShimEngine(
        "meta-llama/Llama-3.1-8B-Instruct", revision=REV,
        identity="hf_shim+apply+exact_replay", hf_engine=bf16)
    with pytest.raises(osh.ShimError, match="fp32"):
        shim.load()

    fp32 = FakeHFEngine()
    fp32.model = TinyModule(torch.float32)
    ok = osh.OpenAIShimEngine(
        "meta-llama/Llama-3.1-8B-Instruct", revision=REV,
        identity="hf_shim+apply+exact_replay", hf_engine=fp32)
    ok.load()  # validated, accepted

    # a DECLARED non-fp32 inference dtype refuses even without tensors
    declared = FakeHFEngine()
    declared.dtype = torch.bfloat16
    shim2 = osh.OpenAIShimEngine(
        "meta-llama/Llama-3.1-8B-Instruct", revision=REV,
        identity="hf_shim+apply+exact_replay", hf_engine=declared)
    with pytest.raises(osh.ShimError, match="dtype"):
        shim2.load()


def test_shim_reference_generate_returns_raw_token_ids():
    shim, eng = _shim()
    ids = shim.reference_generate(
        [{"role": "user", "content": "count to three"}], max_tokens=16)
    assert isinstance(ids, list)
    assert ids == [len(eng.calls[-1]["prompts"][0]) % 97, 7, 11]
    assert eng.calls[-1]["gen_cfg"]["greedy"] is True
    # the gate compares TOKEN IDS: wire it straight into the comparator
    res = gate.exact_token_equality(ids, ids, prompt_index=0)
    assert res.passed


def test_shim_serves_an_arm_under_ctx_discipline():
    shim, eng = _shim()

    class FakeArm:
        name = "si_safety__cast"

        def transform_prompt(self, s):
            return "[armed]" + s

    shim.chat_completions({
        "model": "m", "messages": [{"role": "user", "content": "x"}],
        "max_tokens": 8}, arm=FakeArm())
    call = eng.calls[-1]
    assert call["prompts"][0].startswith("[armed]<user>x")
    assert call["arm"].name == "si_safety__cast"  # generate runs under arm


# --- the mixed-batch check's hard floor (2026-08-30) -------------------------
# The geometry-honest gates proved the old 2048x32 fidelity evidence vacuous
# (27-row plan fit only because 32 >= 27); the floor below pins that the
# subset mechanism can never quietly shrink the check under the accepted
# precedent (11 rows / 5 selections, both hook sites when both serve).


def test_mixed_plan_floor_and_ceiling():
    import importlib.util as ilu
    from pathlib import Path as _P
    spec = ilu.spec_from_file_location(
        "petri_steered_smoke",
        _P(__file__).resolve().parents[1] / "scripts/petri_steered_smoke.py")
    smk = ilu.module_from_spec(spec)
    sys.modules.setdefault("petri_steered_smoke", smk)
    spec.loader.exec_module(smk)
    sites = {"a_psr": ["block"], "b_lora": ["oproj"],
             "c_psr": ["block"], "d_lora": ["oproj"]}
    both = ["a_psr", "b_lora"]
    plan2 = smk.build_mixed_plan(both, 8)          # 3 + 2*4 = 11 rows
    assert len(plan2) == 11
    # the accepted-precedent shape passes at the rerun geometry (12 seqs)
    smk.validate_mixed_plan(plan2, 12, arm_hook_sites=sites,
                            all_arm_ids=list(sites))
    # ceiling: a 27-row plan refuses at 24 (the 45985/45986 refusal)
    plan6 = smk.build_mixed_plan(list(sites) + ["e_psr", "f_lora"], 8)
    assert len(plan6) == 27
    with pytest.raises(RuntimeError, match="split it"):
        smk.validate_mixed_plan(plan6, 24)
    # floor: one arm (7 rows) refuses even where it fits
    plan1 = smk.build_mixed_plan(["a_psr"], 8)
    assert len(plan1) == 7
    with pytest.raises(RuntimeError, match="floor"):
        smk.validate_mixed_plan(plan1, 12, arm_hook_sites=sites,
                                all_arm_ids=list(sites))
    # site coverage: a single-site subset refuses when both sites serve
    plan_same_site = smk.build_mixed_plan(["a_psr", "c_psr"], 8)  # 11 rows
    with pytest.raises(RuntimeError, match="hook sites"):
        smk.validate_mixed_plan(plan_same_site, 12, arm_hook_sites=sites,
                                all_arm_ids=list(sites))


# --- Qwen3.5 hybrid-recurrent serving: thinking-off + structural no-cache ---
# Both land in build_server_launch (steps 1+2 of the qwen wave build,
# 2026-08-29). The thinking pin is keyed on the template carrying the
# switch (the SAME signal models.load_model keys on), and prefix caching
# resolves OFF for the hybrid stack because align mode needs chunked
# prefill and the lens path forbids it (job 243180 phase C32).

#: a synthetic template that carries the reasoning switch, like Qwen3.5's.
THINK_TEMPLATE_BYTES = (b"{# synthetic reasoning template #}"
                        b"{% if enable_thinking %}<think>{% endif %}\n")
THINK_TEMPLATE_HASH = cm.sha256_hex16(THINK_TEMPLATE_BYTES)


def _think_template(tmp_path) -> Path:
    tpl = tmp_path / "qwen_chat_template.jinja"
    tpl.write_bytes(THINK_TEMPLATE_BYTES)
    return tpl


def test_thinking_pin_composed_for_a_reasoning_template(tmp_path):
    """A template carrying the enable_thinking switch composes
    --default-chat-template-kwargs {"enable_thinking": false} — the serve
    twin of the fit's models.load_model pin, so a response-masked fit
    (H=7 on the empty <think></think>) is never served thinking-ON, which
    would slide token 1 into an open <think> the fit never saw."""
    tpl = _think_template(tmp_path)
    launch = vs.build_server_launch(
        _spec(target_model="Qwen/Qwen3.5-9B",
              chat_template_hash=THINK_TEMPLATE_HASH,
              serving_launch_flags=["--enforce-eager"]),
        port=8801, exact_replay_strategy="serialize",
        chat_template_path=tpl)
    cmd = list(launch.cmd)
    assert "--default-chat-template-kwargs" in cmd
    assert (cmd[cmd.index("--default-chat-template-kwargs") + 1]
            == '{"enable_thinking": false}')
    assert (launch.default_chat_template_kwargs
            == vs.THINKING_OFF_TEMPLATE_KWARGS)


def test_thinking_pin_absent_for_a_plain_template(tmp_path):
    """A template with no thinking switch (llama/gemma) composes nothing
    — the flag would be inert noise, and pinning it would drift the
    launch identity from every prior dense launch for no behavior."""
    tpl = _chat_template(tmp_path)  # no 'enable_thinking' in the bytes
    launch = vs.build_server_launch(
        _tpl_spec(serving_launch_flags=["--enforce-eager"]),
        port=8801, exact_replay_strategy="serialize",
        chat_template_path=tpl)
    assert "--default-chat-template-kwargs" not in launch.cmd
    assert launch.default_chat_template_kwargs is None


def test_prefix_caching_defaults_off_for_hybrid_recurrent_target(tmp_path):
    """Qwen3.5 (mamba/GDN hybrid) resolves prefix_caching OFF by default:
    align mode requires chunked prefill, the lens path forbids it, so
    caching is structurally impossible (job 243180 phase C32). Dense
    targets keep the ON default."""
    tpl = _think_template(tmp_path)
    hybrid = vs.build_server_launch(
        _spec(target_model="Qwen/Qwen3.5-9B",
              chat_template_hash=THINK_TEMPLATE_HASH,
              serving_launch_flags=["--enforce-eager"]),
        port=8801, exact_replay_strategy="serialize",
        chat_template_path=tpl)
    assert "--no-enable-prefix-caching" in hybrid.cmd
    assert "--enable-prefix-caching" not in hybrid.cmd
    assert hybrid.prefix_caching is False
    # the resolved-geometry attestation follows the composed value, so a
    # lens readiness check will expect caching OFF for this launch.
    assert vs.expected_scheduler_state(hybrid)["enable_prefix_caching"] is False
    # dense default stays ON
    dense = vs.build_server_launch(
        _tpl_spec(serving_launch_flags=["--enforce-eager"]),
        port=8802, exact_replay_strategy="serialize",
        chat_template_path=_chat_template(tmp_path))
    assert dense.prefix_caching is True


def test_prefix_caching_true_on_hybrid_target_refuses(tmp_path):
    """An EXPLICIT prefix_caching=True on a hybrid target is a config
    error, not a silent downgrade: the engine cannot bring it up honestly
    (align needs chunked prefill), so the composer refuses loudly."""
    tpl = _think_template(tmp_path)
    with pytest.raises(vs.VllmServerError, match="structurally impossible"):
        vs.build_server_launch(
            _spec(target_model="Qwen/Qwen3.5-9B",
                  chat_template_hash=THINK_TEMPLATE_HASH,
                  serving_launch_flags=["--enforce-eager"]),
            port=8801, exact_replay_strategy="serialize",
            chat_template_path=tpl, prefix_caching=True)


def test_default_chat_template_kwargs_is_wrapper_owned(tmp_path):
    """A manifest cannot restate --default-chat-template-kwargs: it is a
    composed pin, wrapper-owned like every other launch-identity flag."""
    tpl = _think_template(tmp_path)
    with pytest.raises(vs.VllmServerError):
        vs.build_server_launch(
            _spec(target_model="Qwen/Qwen3.5-9B",
                  chat_template_hash=THINK_TEMPLATE_HASH,
                  serving_launch_flags=[
                      "--enforce-eager",
                      "--default-chat-template-kwargs", '{"enable_thinking": true}']),
            port=8801, exact_replay_strategy="serialize",
            chat_template_path=tpl)


# --------------------------------------------------------------------------- #
# the gate's --no-output-diagnostics policy knob                              #
# --------------------------------------------------------------------------- #

def test_gate_selection_under_no_output_diagnostics_never_measures_agreement(monkeypatch):
    """The raising-stub test for the GATE: with ``output_diagnostics=False`` the
    lora-native selection check is STRUCTURAL only — ``fidelity.
    multi_lora_output_diagnostics`` (replaced by a stub that raises) is never
    called, the record says ``output_diagnostics: false`` and carries no
    ``numerics``; with the default (True) the same stub IS reached (the
    known-bad), so the knob is what keeps it out."""
    from sisteer.capeval.serving import fidelity
    from test_fidelity_routing_attested import ID_NAME, SEL_ADAPTER, TABLE, _Recorder

    calls = []

    def raiser(**kw):
        calls.append(kw)
        raise AssertionError("multi_lora_output_diagnostics must not run under --no-output-diagnostics")

    monkeypatch.setattr(fidelity, "multi_lora_output_diagnostics", raiser)

    def _routing(rec):
        return gate.LoraRoutingRecorderAccess(arm=rec.arm, observe=rec.observe, disarm=rec.disarm,
                                              mixed_batch=rec.mixed_batch, selection_adapter=SEL_ADAPTER,
                                              id_name_map=ID_NAME)

    rec = _Recorder()
    out = gate.run_multi_lora_selection(base="base", adapters=("genuine", "zero"), mixed_batch=rec.mixed_batch,
                                        solo=lambda n, m: list(TABLE[n]), zero_adapter="zero",
                                        routing=_routing(rec), output_diagnostics=False)
    assert calls == []
    assert out["passed"] is True and out["verdict"] == "routing_attested"
    assert out["output_diagnostics"] is False and "numerics" not in out
    assert out["n_misrouted"] == 0 and out["n_requests"] == out["n_observed"] > 0
    # the known-bad: the default policy reaches the diagnostics (the stub raises)
    rec2 = _Recorder()
    with pytest.raises(AssertionError, match="must not run"):
        gate.run_multi_lora_selection(base="base", adapters=("genuine", "zero"), mixed_batch=rec2.mixed_batch,
                                      solo=lambda n, m: list(TABLE[n]), zero_adapter="zero",
                                      routing=_routing(rec2))
    assert len(calls) == 1
    # the probe carries the policy to the call site (MultiLoraProbe.output_diagnostics)
    probe = gate.MultiLoraProbe(base="base", adapters=("genuine", "zero"), mixed_batch=rec.mixed_batch,
                                solo=lambda n, m: list(TABLE[n]), zero_adapter="zero", routing=_routing(rec),
                                output_diagnostics=False)
    assert probe.output_diagnostics is False
    assert gate.MultiLoraProbe(base="b", adapters=(), mixed_batch=None, solo=None).output_diagnostics is True


def test_gate_parser_accepts_no_output_diagnostics_on_the_lora_native_path_only(tmp_path):
    """``--no-output-diagnostics`` is a lora-native policy: accepted beside
    ``--lora-native-standup`` + ``--reference``, refused on the lens path (which
    measures no such diagnostics to drop)."""
    import contextlib
    import io as _io
    argv = ["gate", "--manifest", "m.yaml", "--out-dir", str(tmp_path), "--lora-native-standup", str(tmp_path),
            "--reference", str(tmp_path / "ref.json"), "--no-output-diagnostics"]
    ns = gate.main.__globals__["argparse"].Namespace  # noqa: F841 — the parser is built inside main
    # main() parses then dispatches; a bogus manifest path makes cmd_gate fail AFTER parsing, which is what
    # we want to observe: parsing succeeded. Capture and swallow that failure.
    with pytest.raises(Exception):
        with contextlib.redirect_stderr(_io.StringIO()):
            gate.main(argv)
    with pytest.raises(SystemExit) as e, contextlib.redirect_stderr(_io.StringIO()):
        gate.main(["gate", "--manifest", "m.yaml", "--out-dir", str(tmp_path), "--no-output-diagnostics"])
    assert e.value.code == 2


def test_gate_output_diagnostics_inherits_the_standups_policy():
    """P2-A's sibling at the gate: the diagnostics run only when the flag did not turn them off AND
    the standup the gate attaches to did not resolve its own policy OFF — omitting
    --no-output-diagnostics can never re-enable them on a residual-bearing standup."""
    assert gate.gate_output_diagnostics(flag_off=False, endpoint={"output_diagnostics": True}) is True
    assert gate.gate_output_diagnostics(flag_off=False, endpoint={}) is True                # a pre-policy standup
    assert gate.gate_output_diagnostics(flag_off=False, endpoint={"output_diagnostics": False}) is False
    assert gate.gate_output_diagnostics(flag_off=True, endpoint={"output_diagnostics": True}) is False
    assert gate.gate_output_diagnostics(flag_off=True, endpoint={"output_diagnostics": False}) is False
    src = inspect.getsource(gate._bootstrap_lora_native_stack)
    assert "gate_output_diagnostics(" in src and 'endpoint=endpoint' in src
