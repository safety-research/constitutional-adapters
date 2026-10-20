"""The REAL defense-cell registry seam: the training side's landed
``sisteer.eval.defense_cell`` registry behind the harness CellSource
protocol (``sisteer.capeval.petri.registry_adapter``).

Covers: the id translation BOTH ways (incl. the loud refusals on
untranslatable ids in either vocabulary); the arm -> object-kind map
pinned against the real object classes; the prompted baselines' REAL SI
bytes and their in-repo prefix-hash pins; absent trained objects ->
ABSENT (the expected state while the training run is still producing
artifacts); a present artifact resolving its spec from its own stored
content (and refusing on kind drift / wrong-SI meta.json); the all-lens
serving state (no genuine LoRA export yet -> no none__lora0 leg); and
the driver wiring (petri_run.load_cell_source builds the real adapter
from configs/two_si.yaml when no test factory is injected).

CPU-offline; the only heavyweight import is the training side's methods
stack (torch), which the adapter itself needs."""

import dataclasses
import json
from pathlib import Path

import pytest
import torch
import yaml

from sisteer.capeval import manifest as cm
from sisteer.capeval.petri import defense_cells as dc
from sisteer.capeval.petri import registry_adapter as ra
from sisteer.capeval.serving import serve_declaration as sd
from sisteer.data.instructions import SAFETY_SI
from sisteer.data.safety_texts import CONSTITUTION_PROMPT

REPO = Path(__file__).parents[1]
# The two-SI model pin — the NousResearch mirror configs/two_si.yaml pins
# since 2026-08-20 (same weights as meta-llama, different chat template);
# the repo-registry tests below cross-check against the REAL config, so
# these must track it.
REV = "d10aef7999a2b5ba950ab3974312feeedbfe0b77"

#: the in-repo SI prefix-hash pins (also asserted against the behavior
#: variant dirs by test_petri_scenarios) — the freeze values.
SAFETY_PREFIX_HASH = "ac3a58f61d53cf24"
CONSTITUTION_PREFIX_HASH = "21e06f482a6559b3"

CANONICAL_22 = (
    ["none", "prompted__safety", "prompted__constitution"]
    # the prompt-SI matrix is over the FIXED confirmatory family
    # (PROMPTABLE_CLASS_A_METHODS), never the serving vocabulary
    # CLASS_A_METHODS, which also admits object-only methods that have no
    # cell under a prompt SI (review of 039a355c, 2026-09-10)
    + [f"si_{si}__{m}" for m in dc.PROMPTABLE_CLASS_A_METHODS
       for si in ("safety", "constitution")]
    + list(dc.CLASS_B_IDS)
)

# only the IN-PLAY variants (safety + constitution — this spec has no
# prompted__irrelevant cell, and an extra behavior_files key refuses)
BEHAVIOR_FILES = {
    **{b: f"# behavior {b}\nsynthetic eval bytes\n".encode()
       for b in cm.PETRI_BEHAVIORS},
    **{f"{b}/{v}": f"# variant {v} of {b}\n".encode()
       for b in cm.PETRI_BEHAVIORS
       for v in (cm.PETRI_SI_VARIANT_DIRS["safety"],
                 cm.PETRI_SI_VARIANT_DIRS["constitution"])},
}
PROMPT_FILES = {
    "auditor": b"synthetic auditor prompt (test bytes)",
    "judge": b"synthetic judge prompt (test bytes)",
}


def _spec(**over) -> cm.PetriSpec:
    kwargs = dict(
        audit="petri-adapter-test",
        behaviors=list(cm.PETRI_BEHAVIORS),
        cells=list(CANONICAL_22),
        scenario_bundle_hash="ab" * 8,
        auditor_model="anthropic/claude-sonnet-5",
        judge_model="anthropic/claude-opus-5",
        scenarios_model="anthropic/claude-sonnet-5",
        prompt_provenance="package-pinned@petri-bloom==0.2.6",
        target_model="NousResearch/Meta-Llama-3.1-8B-Instruct",
        target_revision=REV,
        serving_backend="proxy_vllm",
        serving_mode="exact_replay",
        serving_packages={"vllm": "0.25.1+cu129", "vllm-lens": "1.2.1"},
        serving_launch_flags=["--enable-lora", "--enforce-eager"],
        chat_template_hash="12" * 8,
        tool_parser="llama3_json",
        serving_impl_commit="fedcba9876" * 4,
        # the REAL prefix hashes: the landed SI bytes must match the pins
        prompted_prefix_hashes={"safety": SAFETY_PREFIX_HASH,
                                "constitution": CONSTITUTION_PREFIX_HASH},
        bloom_options=cm.BloomOptions(
            max_turns=15, enable_prefill=False, enable_rollback=False,
            compaction=False, realism_filter=False, epochs=1, seed=0),
    )
    kwargs.update(over)
    return cm.PetriSpec(**kwargs)


def _write_cfg(tmp_path, **over) -> Path:
    """A minimal two-SI config yaml over a tmp defenses_dir."""
    cfg = dict(
        si_ids=["safety_refusal", "si_constitution"],
        defenses_dir=str(tmp_path / "cells"),
        model="NousResearch/Meta-Llama-3.1-8B-Instruct",
        model_revision=REV,
    )
    cfg.update(over)
    path = tmp_path / "two_si.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


def _source(tmp_path, **cfg_over) -> ra.TrainTestCellSource:
    """Build over the tmp store via the EXPLICIT ``defenses_dir`` override
    — the tests' path to the objects dir now that the config's relative
    in-checkout default is never read. (The authoritative resolver has
    landed; the override keeps these tests independent of its resolution
    order — e.g. a TT_TWO_SI_DEFENSES_DIR in the environment.)"""
    return ra.traintest_cell_source(
        _spec(), repo_root=REPO, config_path=_write_cfg(tmp_path, **cfg_over),
        defenses_dir=tmp_path / "cells")


def _enumerate(source, spec=None) -> dc.PetriCellSet:
    return dc.enumerate_cells(spec or _spec(), source,
                              behavior_files=BEHAVIOR_FILES,
                              prompt_files=PROMPT_FILES)


# --------------------------------------------------------------------------
# id translation, both directions
# --------------------------------------------------------------------------


def test_forward_id_translation_pins():
    assert ra.traintest_cell_id("none") == "none"
    assert ra.traintest_cell_id("prompted__safety") == "prompted__safety_refusal"
    assert (ra.traintest_cell_id("prompted__constitution")
            == "prompted__si_constitution")
    assert ra.traintest_cell_id("si_safety__lofit") == "safety_refusal__lofit"
    # the accidental textual coincidence: the training side's constitution
    # SI id IS "si_constitution", so these rows map onto the same string
    assert (ra.traintest_cell_id("si_constitution__cast")
            == "si_constitution__cast")
    for b in dc.CLASS_B_IDS:
        assert ra.traintest_cell_id(b) == b


def test_id_translation_round_trips_all_22():
    for harness_id in CANONICAL_22:
        assert ra.harness_cell_id(ra.traintest_cell_id(harness_id)) == harness_id


def test_forward_translation_refuses_malformed_ids():
    for bad in ("si_paranoia__psr_kl_all", "prompted__both", "not_a_cell",
                "none__lens0"):
        with pytest.raises(dc.DefenseCellError):
            ra.traintest_cell_id(bad)


def test_reverse_translation_refuses_untranslatable_traintest_ids():
    # the alphakl ladder is training-side-only — no harness id exists, and
    # the refusal must name the vocabulary gap rather than silently dropping
    with pytest.raises(ra.RegistryAdapterError, match="alphakl|audited set"):
        ra.harness_cell_id("safety_refusal__alphakl_null")
    with pytest.raises(ra.RegistryAdapterError, match="translation"):
        ra.harness_cell_id("prompted__si_hhh")  # unmapped SI
    with pytest.raises(ra.RegistryAdapterError, match="translation"):
        ra.harness_cell_id("si_hhh__vector_all")  # unmapped SI, Class A
    with pytest.raises(ra.RegistryAdapterError, match="Class B"):
        ra.harness_cell_id("did")  # registered arm outside the matrix
    with pytest.raises(ra.RegistryAdapterError, match="non-empty"):
        ra.harness_cell_id("")


# --------------------------------------------------------------------------
# the arm -> object-kind map is the real methods.py vocabulary
# --------------------------------------------------------------------------


def test_arm_object_kinds_match_the_real_classes():
    from sisteer.extraction.alphasteer import AlphaSteerObject
    from sisteer.extraction.antidote import AntidoteObject
    from sisteer.extraction.caa_static_r import CaaStaticRObject
    from sisteer.extraction.cast import CastObject
    from sisteer.extraction.circuit_breaker import CircuitBreakerObject
    from sisteer.extraction.lofit import LoFiTObject
    from sisteer.extraction.multilayer import (LoRASubtractionObject,
                                                MultiLayerLoRAObject,
                                                MultiLayerVectorObject)
    from sisteer.extraction.objects import RankOneLoRAObject, SoftTokenObject
    from sisteer.extraction.paired_write import PSRPairedWriteObject
    from sisteer.extraction.psr import PSRObject
    from sisteer.extraction.reft import LoReFTObject

    expected = {
        "psr_kl_all": PSRObject.kind,
        "psr_mse_all": PSRObject.kind,
        "vector_all": MultiLayerVectorObject.kind,
        "lora_r1": RankOneLoRAObject.kind,
        "lora_r4": MultiLayerLoRAObject.kind,
        "loreft": LoReFTObject.kind,
        "lofit": LoFiTObject.kind,
        "token_bos": SoftTokenObject.kind,
        # lora_r4 = the canonical spelling of lora_r1_all (2026-08-28):
        # rank-general MultiLayerLoRAObject, stored kind "lora_all"
        "lora_r4": MultiLayerLoRAObject.kind,
        # psr_xfer_* (2026-08-28): cross-model TRANSFERRED PSR units —
        # ordinary PSRObject blobs under transfer-identity arm names
        "psr_xfer_from_e2b": PSRObject.kind,
        "psr_xfer_from_e2b_rw1": PSRObject.kind,
        "psr_xfer_from_e4b": PSRObject.kind,
        "psr_xfer_from_e4b_rw1": PSRObject.kind,
        # sdf-base-steering document-CE arms (2026-09-04): the SAME
        # artifact classes as psr_kl_all / lora_r4
        "psr_ce_all": PSRObject.kind,
        "lora_r4_ce": MultiLayerLoRAObject.kind,
        # sdf-base-steering DERIVED arms (2026-09-06): the corpus fit minus
        # its control fit. The distilled residual is a PSRObject like its
        # parents (the derivation lives in the arm NAME, the psr_xfer_*
        # precedent); the LoRA difference is the one that carries its own
        # class, because it stores BOTH parents' factors.
        "psr_ce_all_minus_ctl_distilled": PSRObject.kind,
        "lora_r4_ce_minus_ctl": LoRASubtractionObject.kind,
        # the UNDISTILLED paired write (relu-differenced-native-qwen, 2026-09-17): both
        # parents bit-exact, served as +u_pos(z) - u_neg(z) on the shared pre-write state
        # through the native residual slot's second gate term — its OWN kind
        "psr_ce_all_minus_ctl": PSRPairedWriteObject.kind,
        "cast": CastObject.kind,
        "antidote": AntidoteObject.kind,
        "alphasteer": AlphaSteerObject.kind,
        # caa_static_r (E45 H2): a kind of its OWN, not the peer's static_r —
        # the two save different blobs and static_r's defaults would read this
        # one as a single-layer last-token edit
        "caa_static_r": CaaStaticRObject.kind,
        # arditi: the Arditi refusal direction, staged as a StaticRObject blob — the "static_r" kind,
        # served through _adapt_static_r's prefill_all/pre deployment. A SERVED-ONLY Class B id: no trainer
        # here mints it, so it is outside CLASS_B_IDS and the map is what lets the serving side resolve and
        # adapt the staged object.
        "arditi": "static_r",
        # arditi_allpos (arditi-qwen-all-positions, 2026-09-16): the SAME fitted direction served in the
        # PAPER'S OWN configuration — the raw difference of means at coefficient c, added at every position
        # of every forward — which is a "vector_all" blob (layers under `vectors` keyed by the SERVED block
        # layer; positions / fit_positions "all"; skip_bos False; no base_scale), served through
        # _adapt_vector_all at PHASE_BOTH or natively on the residual slot's constant path. A SECOND
        # served-only id, never an alias of `arditi`: the two differ in PHASE and in DOSE UNIT at once.
        "arditi_allpos": "vector_all",
    }
    assert ra.ARM_OBJECT_KINDS == expected
    assert expected["lora_r4_ce_minus_ctl"] == "lora_subtraction"
    assert expected["lora_r4_ce_minus_ctl"] != MultiLayerLoRAObject.kind
    assert expected["caa_static_r"] == "caa_static_r"      # its own kind …
    assert CaaStaticRObject.kind != "static_r"             # … not the peer's
    # the map covers exactly the audited matrix arms + the served-only Class B ids
    assert set(ra.ARM_OBJECT_KINDS) == (set(dc.CLASS_A_METHODS)
                                        | set(dc.CLASS_B_IDS)
                                        | set(dc.SERVED_ONLY_CLASS_B_IDS))
    assert expected["arditi"] == "static_r" != CaaStaticRObject.kind
    # the two Arditi ids map to DIFFERENT kinds, which is the whole reason there are two of them
    assert expected["arditi_allpos"] == "vector_all" != expected["arditi"]
    # circuit_breaker is not an ARM any more (E45 H2: it left methods.ARMS in
    # 2026-08, and now the Class B id vocabulary too) — but its KIND survives,
    # because fitted artifacts of it still hydrate and still export
    assert "circuit_breaker" not in ra.ARM_OBJECT_KINDS
    assert "circuit_breaker" in dc.RETIRED_CLASS_B_IDS
    assert CircuitBreakerObject.kind == "circuit_breaker"
    # no object kind collides with the non-object kinds
    for kind in ra.ARM_OBJECT_KINDS.values():
        assert kind not in (cm.NONE_KIND, cm.PETRI_PROMPTED_KIND)


# --------------------------------------------------------------------------
# prompted baselines: the REAL SI bytes, matching the in-repo pins
# --------------------------------------------------------------------------


def test_prompted_entries_carry_the_landed_si_bytes(tmp_path):
    source = _source(tmp_path)
    safety = source.get("prompted__safety")
    constitution = source.get("prompted__constitution")
    assert safety.kind == "prompted" and safety.si_id == "safety"
    assert safety.si_prefix == SAFETY_SI.text.encode("utf-8")
    assert constitution.si_prefix == CONSTITUTION_PROMPT.encode("utf-8")
    # the landed bytes hash to the pinned freeze values
    assert cm.sha256_hex16(safety.si_prefix) == SAFETY_PREFIX_HASH
    assert cm.sha256_hex16(constitution.si_prefix) == CONSTITUTION_PREFIX_HASH


def test_prompted_prefix_hashes_verify_at_enumeration(tmp_path):
    cs = _enumerate(_source(tmp_path))
    by_id = {c.cell_id: c for c in cs.canonical}
    assert by_id["prompted__safety"].si_prefix_hash == SAFETY_PREFIX_HASH
    assert (by_id["prompted__constitution"].si_prefix_hash
            == CONSTITUTION_PREFIX_HASH)
    # a manifest recording a WRONG prefix hash refuses (H8) — the real
    # bytes can never masquerade under a stale pin
    spec = _spec(prompted_prefix_hashes={
        "safety": "00" * 8, "constitution": CONSTITUTION_PREFIX_HASH})
    with pytest.raises(dc.DefenseCellError, match="masquerade|prefix"):
        _enumerate(_source(tmp_path), spec=spec)


# --------------------------------------------------------------------------
# not-yet-trained objects: ABSENT, never a refusal, never a silent drop
# --------------------------------------------------------------------------


def test_untrained_matrix_reports_all_object_cells_absent(tmp_path):
    cs = _enumerate(_source(tmp_path))
    assert [c.cell_id for c in cs.canonical] == [
        "none", "prompted__safety", "prompted__constitution"]
    absent = {a.cell_id: a.reason for a in cs.absent}
    assert set(absent) == set(CANONICAL_22[3:])  # every object cell
    # cast / circuit_breaker left the training-side registry entirely on
    # 2026-08-20 (hydration of fitted artifacts survives in _KIND_LOADERS)
    # — their petri cells stay in the canonical matrix and report ABSENT
    # with the registry reason, never a silent drop.
    _REMOVED = {"si_safety__cast", "si_constitution__cast", "circuit_breaker"}
    # lora_r4 is the CANONICAL arm name (2026-08-28) but the training-side
    # registry still keys its cells by the legacy arm name lora_r1_all.
    # Until that flip lands, the petri cells report ABSENT with the
    # registry reason (never a silent drop, never a here-and-now alias),
    # exactly like the removed arms.
    # RETIRED 2026-08-28: the lora_r1_all -> lora_r4 registry flip landed,
    # so these cells ARE registered now and take the ordinary
    # missing-artifact path. The set stays (empty) rather than the branch
    # being deleted, because _REMOVED below still needs it.
    _PENDING_RENAME: set = set()
    # sdf-base-steering (2026-09-04, widened 2026-09-06): the document-CE
    # arms, the DERIVED arms and the OBJECT-ONLY SIs (constv2_*) reach the
    # registry ONLY through an explicit_cells config — a Cartesian two-SI
    # config never builds them (si_ids never names a corpus
    # pseudo-instruction, and none of those arms is in CLASS_A_ARMS /
    # CLASS_B_ARMS), so their harness cells carry the registry reason here,
    # exactly like the removed arms. DERIVED from the Cartesian arm lists
    # rather than listed, so an arm added to the harness vocabulary without
    # being added to the Cartesian matrix lands in this set by itself.
    from sisteer.eval.defense_cell import CLASS_A_ARMS, CLASS_B_ARMS

    _CARTESIAN_ARMS = set(CLASS_A_ARMS) | set(CLASS_B_ARMS)
    _EXPLICIT_ONLY = {
        c for c in absent
        if (dc.parse_cell_id(c).method not in _CARTESIAN_ARMS
            or dc.parse_cell_id(c).si_id in cm.PETRI_OBJECT_ONLY_SI_IDS)}
    for cell_id, reason in absent.items():
        if cell_id in _REMOVED | _PENDING_RENAME | _EXPLICIT_ONLY:
            assert "not in the defense-cell registry" in reason
            continue
        assert "missing" in reason
        # the reason names the TRAINTEST object path (the real store)
        assert ra.traintest_cell_id(cell_id) in reason
    # no genuine LoRA export exists yet -> everything lens-served,
    # so ONLY the lens no-op control leg derives
    assert [c.cell_id for c in cs.controls] == ["none__lens0"]


def test_missing_artifact_with_no_resolved_spec_is_absent_not_refusal(tmp_path):
    """The reorder this seam needed, pinned: an object cell whose artifact
    does not exist has no authoritative load_object resolution to demand
    (resolved_spec=None) and reports ABSENT — while a PRESENT artifact
    with resolved_spec=None still refuses loudly (H4)."""
    entry = dc.RegistryEntry(
        cell_id="si_safety__loreft", kind="loreft", si_id="safety",
        object_path=str(tmp_path / "nowhere" / "object.pt"),
        serving_path="lens", resolved_spec=None)
    entries = {"si_safety__loreft": entry}
    spec = _spec(cells=["si_safety__loreft"])
    cs = dc.enumerate_cells(spec, dc.MappingCellSource(entries),
                            behavior_files=BEHAVIOR_FILES,
                            prompt_files=PROMPT_FILES)
    assert [a.cell_id for a in cs.absent] == ["si_safety__loreft"]
    assert "missing" in cs.absent[0].reason
    # present artifact + None spec -> the loud H4 refusal stands
    artifact = tmp_path / "object.pt"
    artifact.write_bytes(b"present bytes")
    entries = {"si_safety__loreft": dataclasses.replace(
        entry, object_path=str(artifact))}
    with pytest.raises(dc.DefenseCellError, match="resolved_spec"):
        dc.enumerate_cells(spec, dc.MappingCellSource(entries),
                           behavior_files=BEHAVIOR_FILES,
                           prompt_files=PROMPT_FILES)


# --------------------------------------------------------------------------
# a present artifact: spec resolved from its own stored content
# --------------------------------------------------------------------------


def _write_lora_blob(tmp_path, tt_id="safety_refusal__lora_r1",
                     kind="lora", **fields):
    cell_dir = tmp_path / "cells" / tt_id
    cell_dir.mkdir(parents=True, exist_ok=True)
    blob = {"kind": kind, "A": torch.zeros(4), "B": torch.zeros(4),
            "layer": 15, **fields}
    torch.save(blob, cell_dir / "object.pt")
    return cell_dir


def test_present_torch_blob_resolves_spec_from_stored_fields(tmp_path):
    _write_lora_blob(tmp_path)
    source = _source(tmp_path)
    entry = source.get("si_safety__lora_r1")
    assert entry.kind == "lora"
    assert entry.serving_path == "lens"  # no genuine LoRA export yet
    assert entry.sidecar_path is None  # torch-blob arm, no sidecar written
    # the spec is the blob's own stored non-tensor fields (what load_blob
    # consumes) — tensor payloads are hashed as artifact bytes instead
    assert entry.resolved_spec == {"kind": "lora", "arm": "lora_r1",
                                   "layer": 15, "si_id": "safety"}
    cs = _enumerate(source)
    served = {c.cell_id for c in cs.canonical}
    assert "si_safety__lora_r1" in served
    assert {a.cell_id for a in cs.absent} == set(CANONICAL_22[3:]) - served
    by_id = {c.cell_id: c for c in cs.canonical}
    assert (by_id["si_safety__lora_r1"].serving_identity
            == "proxy_vllm+lens+exact_replay")


def test_caa_static_r_is_wired_end_to_end(tmp_path):
    """E45 H2, the whole chain on a synthetic object: `caa_static_r` used to be
    a real training-side arm the serving side could not name — `harness_cell_id`
    refused it, `ARM_OBJECT_KINDS` did not map it, and no lens adapter was
    registered for its kind — so widening the launcher's parser alone would
    only have moved the failure onto a GPU. Here it goes all the way:
    `harness_cell_id` -> `traintest_cell_source` -> registry resolution (from
    the artifact's OWN stored fields, at `<bank>/caa_static_r/object.pt`) ->
    `export_worker_registry` (which runs the lens adapter and the fit/serve
    guard) -> `WorkerObjectRegistry.load`. The served translation is the
    class's single deployment: every stored layer, ALL prompt positions,
    prefill only, resid_pre — hence the one-block index shift."""
    from sisteer.capeval.serving import lens_adapters as la
    from sisteer.capeval.serving import lens_steer as ls

    layers = (8, 9, 10)
    cell_dir = tmp_path / "cells" / "caa_static_r"
    cell_dir.mkdir(parents=True)
    artifact = cell_dir / "object.pt"        # the bank path object_path_for gives
    torch.save({"kind": "caa_static_r", "base_scale": 0.05,
                "rs": {l: torch.full((4,), 0.1 * (i + 1))
                       for i, l in enumerate(layers)},
                "metadata": {"source": "synthetic"}}, artifact)

    assert ra.harness_cell_id("caa_static_r") == "caa_static_r"
    source = _source(tmp_path)
    entry = source.get("caa_static_r")
    assert entry.kind == "caa_static_r" and entry.serving_path == "lens"
    assert Path(entry.object_path) == artifact and entry.sidecar_path is None
    assert entry.si_id is None                              # SI-agnostic, Class B
    assert entry.resolved_spec == {"kind": "caa_static_r", "arm": "caa_static_r",
                                   "base_scale": 0.05,
                                   "metadata": {"source": "synthetic"}}

    cells = _enumerate(source)
    assert "caa_static_r" in {c.cell_id for c in cells.canonical}

    out = tmp_path / "lens_registry"
    entries = ls.export_worker_registry(cells, source, out)
    assert "caa_static_r" in {e.cell_id for e in entries}
    reg = ls.WorkerObjectRegistry.load(out)
    assert "caa_static_r" in reg.cell_ids()

    ops = la.get_adapter("caa_static_r")(
        entry.resolved_spec, artifact.read_bytes(), b"")
    assert len(ops) == len(layers)
    assert {op.op for op in ops} == {la.OP_ADD_VECTOR}
    assert {op.positions for op in ops} == {la.POSITIONS_ALL}
    assert {op.phase for op in ops} == {la.PHASE_PREFILL}
    assert {op.coeff for op in ops} == {0.05}
    # resid_pre served at a block-output hook site = one block earlier
    assert [op.layer for op in ops] == [l - 1 for l in layers]
    # and a blob claiming another deployment is refused, never followed
    torch.save({"kind": "caa_static_r", "base_scale": 0.05, "site": "post",
                "rs": {8: torch.full((4,), 0.1)}}, artifact)
    with pytest.raises(la.AdapterError, match="one deployment"):
        la.get_adapter("caa_static_r")(
            {"kind": "caa_static_r"}, artifact.read_bytes(), b"")


def test_blob_kind_drift_refuses_naming_both_vocabularies(tmp_path):
    _write_lora_blob(tmp_path, kind="vector_all")
    with pytest.raises(ra.RegistryAdapterError, match="vocabulary drift"):
        _source(tmp_path).get("si_safety__lora_r1")


def test_corrupt_blob_refuses(tmp_path):
    cell_dir = tmp_path / "cells" / "safety_refusal__lora_r1"
    cell_dir.mkdir(parents=True)
    (cell_dir / "object.pt").write_bytes(b"not a torch blob at all")
    with pytest.raises(ra.RegistryAdapterError, match="torch blob"):
        _source(tmp_path).get("si_safety__lora_r1")


def test_wrong_si_meta_json_refuses(tmp_path):
    """An object sitting in one SI's cell dir whose meta.json records the
    OTHER SI is the exact mis-attribution hazard — refused loudly."""
    cell_dir = _write_lora_blob(tmp_path)
    (cell_dir / "meta.json").write_text(json.dumps(
        {"cell_id": "si_constitution__lora_r1", "arm": "lora_r1",
         "si_id": "si_constitution", "class": "A"}))
    with pytest.raises(ra.RegistryAdapterError, match="mis-attribution|wrong"):
        _source(tmp_path).get("si_safety__lora_r1")


def test_consistent_meta_json_passes(tmp_path):
    cell_dir = _write_lora_blob(tmp_path)
    (cell_dir / "meta.json").write_text(json.dumps(
        {"cell_id": "safety_refusal__lora_r1", "arm": "lora_r1",
         "si_id": "safety_refusal", "class": "A"}))
    entry = _source(tmp_path).get("si_safety__lora_r1")
    assert entry.resolved_spec["si_id"] == "safety"


def test_antidote_safetensors_resolves_from_its_sidecar(tmp_path):
    from sisteer.vectors import SteeringVector

    cell_dir = tmp_path / "cells" / "antidote"
    cell_dir.mkdir(parents=True)
    sv = SteeringVector(vectors={15: torch.zeros(8)}, method="antidote",
                        model_name="test-model", instruction_id="pairs",
                        metadata={"base_alpha": 1.25})
    sv.save(cell_dir / "object.safetensors")  # writes object.json too
    source = _source(tmp_path)
    entry = source.get("antidote")
    assert entry.kind == "antidote"
    # the SteeringVector loader REQUIRES its json sidecar -> always declared
    assert entry.sidecar_path == str(cell_dir / "object.json")
    spec = dict(entry.resolved_spec)
    assert spec["kind"] == "antidote" and spec["arm"] == "antidote"
    assert spec["method"] == "antidote"
    assert spec["metadata"] == {"base_alpha": 1.25}
    assert "si_id" not in spec  # Class B stays SI-agnostic
    cs = _enumerate(source)
    assert "antidote" in {c.cell_id for c in cs.canonical}


def test_antidote_missing_sidecar_is_absent(tmp_path):
    from sisteer.vectors import SteeringVector

    cell_dir = tmp_path / "cells" / "antidote"
    cell_dir.mkdir(parents=True)
    sv = SteeringVector(vectors={15: torch.zeros(8)}, method="antidote",
                        model_name="test-model", instruction_id="pairs")
    sv.save(cell_dir / "object.safetensors")
    (cell_dir / "object.json").unlink()
    cs = _enumerate(_source(tmp_path))
    absent = {a.cell_id: a.reason for a in cs.absent}
    assert "sidecar missing" in absent["antidote"]


def test_mis_saved_artifact_format_refuses(tmp_path):
    """An artifact whose CONTENT contradicts its arm's storage format
    would mis-dispatch in load_object — refused, never hashed."""
    from sisteer.vectors import SteeringVector

    # a safetensors container where a torch blob belongs
    cell_dir = tmp_path / "cells" / "safety_refusal__lora_r1"
    cell_dir.mkdir(parents=True)
    sv = SteeringVector(vectors={15: torch.zeros(8)}, method="vector",
                        model_name="m", instruction_id="i")
    sv.save(cell_dir / "object.pt")
    with pytest.raises(ra.RegistryAdapterError, match="mis-dispatch"):
        _source(tmp_path).get("si_safety__lora_r1")


def test_json_safe_extraction_refuses_unknown_representations():
    with pytest.raises(ra.RegistryAdapterError, match="unhandled"):
        ra._json_safe(object(), "blob['x']")
    with pytest.raises(ra.RegistryAdapterError, match="non-finite"):
        ra._json_safe(float("nan"), "blob['x']")
    # tensor-bearing containers are skipped whole (bytes are hashed)
    assert ra._json_safe({"a": torch.zeros(2)}, "b") is ra._TENSOR
    assert ra._json_safe([1, torch.zeros(2)], "b") is ra._TENSOR
    assert ra._json_safe({"layers": [1, 2], "flag": True}, "b") == {
        "layers": [1, 2], "flag": True}


# --------------------------------------------------------------------------
# construction refusals: config vocabulary / model-pin drift
# --------------------------------------------------------------------------


def test_missing_two_si_config_refuses(tmp_path):
    with pytest.raises(ra.RegistryAdapterError, match="missing"):
        ra.traintest_cell_source(_spec(), repo_root=REPO,
                                 config_path=tmp_path / "nope.yaml")


def test_si_vocabulary_drift_refuses(tmp_path):
    with pytest.raises(ra.RegistryAdapterError, match="vocabulary drift"):
        _source(tmp_path, si_ids=["si_hhh", "si_constitution"])


def test_model_pin_drift_refuses(tmp_path):
    with pytest.raises(ra.RegistryAdapterError, match="base model"):
        _source(tmp_path, model="some-org/other-model")
    with pytest.raises(ra.RegistryAdapterError, match="base model"):
        _source(tmp_path, model_revision="ff" * 20)


def test_unregistered_arm_is_absent_from_the_registry(tmp_path):
    """An arm the config names but methods.ARMS lacks contributes no
    registry cell (build_registry filters) -> the harness reports ABSENT
    'not in the defense-cell registry', never a crash."""
    source = _source(tmp_path, class_a_arms=[
        "psr_kl_all", "psr_mse_all", "vector_all", "lora_r1",
        "loreft", "lofit", "token_bos"])  # cast omitted from the matrix
    assert source.get("si_safety__cast") is None
    cs = _enumerate(source)
    absent = {a.cell_id: a.reason for a in cs.absent}
    assert "registry" in absent["si_safety__cast"]


# --------------------------------------------------------------------------
# the compiled-objects dir: explicit override / delegation / loud refusal
# --------------------------------------------------------------------------


def test_explicit_defenses_dir_override_wins_over_the_config(tmp_path):
    """The config keeps its RELATIVE in-checkout default; an explicit
    ``defenses_dir`` governs BOTH the source's dir and every registry
    object_path — the object_path_for cross-check's two sides can never
    disagree about the store."""
    from sisteer.eval import defense_cell as tt

    store = tmp_path / "objects" / "cells"
    cfg = _write_cfg(tmp_path, defenses_dir="out/two_si/defenses/cells")
    source = ra.traintest_cell_source(
        _spec(), repo_root=REPO, config_path=cfg, defenses_dir=store)
    assert source.defenses_dir == store.absolute()
    assert source.defenses_dir.is_absolute()
    cell = source.cells["safety_refusal__lora_r1"]
    assert (Path(cell.object_path)
            == tt.object_path_for(source.defenses_dir,
                                  "safety_refusal__lora_r1"))


def test_delegates_to_traintest_resolve_defenses_dir_when_present(
        tmp_path, monkeypatch):
    """When the training side's ``resolve_defenses_dir`` has landed, the
    adapter delegates to it as the single source of truth — passing the
    cfg and repo_root it expects, never re-implementing its resolution
    order — and an explicit override still wins over it."""
    from sisteer.eval import defense_cell as tt

    landed = tmp_path / "commit-independent" / "cells"
    calls = []

    def fake_resolver(cfg, *, repo_root=None):
        calls.append((cfg["defenses_dir"], repo_root))
        return landed

    monkeypatch.setattr(tt, "resolve_defenses_dir", fake_resolver,
                        raising=False)
    cfg = _write_cfg(tmp_path)
    source = ra.traintest_cell_source(_spec(), repo_root=REPO,
                                      config_path=cfg)
    assert source.defenses_dir == landed.absolute()
    # the resolver saw the config's OWN defenses_dir and the repo root
    assert calls == [(str(tmp_path / "cells"), REPO)]
    # every registry object_path lives under the delegated dir
    cell = source.cells["antidote"]
    assert (Path(cell.object_path)
            == tt.object_path_for(source.defenses_dir, "antidote"))
    # an explicit override still wins: the resolver is not even consulted
    override = tmp_path / "explicit-store"
    source = ra.traintest_cell_source(_spec(), repo_root=REPO,
                                      config_path=cfg,
                                      defenses_dir=override)
    assert source.defenses_dir == override.absolute()
    assert len(calls) == 1


def test_registry_matches_registry_from_cfg_for_absolute_dir(tmp_path):
    """The drift guard for the adapter's direct ``build_registry`` call
    (the fix for the double-resolution bug: the resolver must run
    exactly once, so the adapter mirrors registry_from_cfg's cfg->kwargs
    mapping instead of calling it). For a cfg whose ``defenses_dir`` is
    ABSOLUTE, ``registry_from_cfg`` is GUARANTEED to resolve to that same
    dir (an absolute value is used as-is), so the two builds must produce
    IDENTICAL registries — if the training side later adds a cfg key
    that registry_from_cfg honors and the adapter's mirroring does not,
    this fails loudly instead of the key being silently ignored. Exercises
    the non-default arm-list keys too, and the REAL landed resolver (no
    explicit override), i.e. the live delegation path."""
    from sisteer.eval import defense_cell as tt

    over = dict(
        class_a_arms=["psr_kl_all", "vector_all", "lora_r1", "cast"],
        class_b_arms=["antidote", "circuit_breaker"],
    )
    cfg_path = _write_cfg(tmp_path, **over)
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    assert Path(cfg["defenses_dir"]).is_absolute()  # the rule-1 guarantee
    source = ra.traintest_cell_source(_spec(), repo_root=REPO,
                                      config_path=cfg_path)
    expected = tt.registry_from_cfg(cfg, repo_root=REPO)
    assert list(source.cells.values()) == expected
    assert source.defenses_dir == Path(cfg["defenses_dir"])
    # non-vacuous: the three baselines plus object cells of BOTH classes
    # are being compared (the registered-arm filter applies identically to
    # both builds, so registration state is deliberately NOT pinned here)
    ids = list(source.cells)
    assert ids[:3] == ["none", "prompted__safety_refusal",
                       "prompted__si_constitution"]
    assert {"safety_refusal__psr_kl_all", "si_constitution__psr_kl_all",
            "antidote"} <= set(ids)


def test_refuses_loudly_when_resolver_absent_and_no_override(
        tmp_path, monkeypatch):
    """No explicit dir + no landed resolver = a LOUD refusal naming
    the authoritative resolver and both remedies — never a silent
    fall-back to the config's relative in-checkout default (the bug this
    seam closes: every object cell would report ABSENT forever, even
    after the objects land, indistinguishable from 'not trained yet')."""
    from sisteer.eval import defense_cell as tt

    monkeypatch.delattr(tt, "resolve_defenses_dir", raising=False)
    cfg = _write_cfg(tmp_path, defenses_dir="out/two_si/defenses/cells")
    with pytest.raises(ra.RegistryAdapterError) as ei:
        ra.traintest_cell_source(_spec(), repo_root=REPO, config_path=cfg)
    msg = str(ei.value)
    assert "sisteer.eval.defense_cell.resolve_defenses_dir" in msg
    assert "D58" in msg
    assert "'out/two_si/defenses/cells'" in msg  # names the refused value
    assert "ABSENT forever" in msg  # ... and why falling back is the bug
    assert "pass defenses_dir= explicitly" in msg  # ... and the remedy


def test_absent_reason_names_the_resolved_absolute_artifact_path(tmp_path):
    """A wrong-directory mistake must be visible at a glance in
    print-cells output: every object cell's ABSENT reason carries the
    fully-resolved ABSOLUTE artifact path that was actually checked."""
    from sisteer.eval import defense_cell as tt

    store = tmp_path / "two_si_defenses" / "cells"  # empty store
    source = ra.traintest_cell_source(
        _spec(), repo_root=REPO, config_path=_write_cfg(tmp_path),
        defenses_dir=store)
    cs = _enumerate(source)
    absent = {a.cell_id: a.reason for a in cs.absent}
    assert set(absent) == set(CANONICAL_22[3:])  # every object cell
    # the 2026-08-20 registry-removal cells carry the registry reason, not
    # an artifact path (there is no registry entry to resolve a path FROM);
    # the lora_r4 cells ride the same reason until the training side's
    # lora_r1_all -> lora_r4 registry flip lands (2026-08-28)
    # lora_r4 dropped from this set 2026-08-28: that registry flip landed,
    # so it resolves and reports a missing ARTIFACT rather than an
    # unregistered arm. cast/circuit_breaker remain genuinely unregistered.
    _removed = {"si_safety__cast", "si_constitution__cast", "circuit_breaker"}
    # + the explicit-config-only cells (the CE arms, the DERIVED arms,
    # object-only SIs; see test_absent_and_serving_paths): no registry
    # entry, no path to name. DERIVED from the Cartesian arm lists, so a
    # harness arm outside that matrix lands here without a test edit.
    _cartesian = set(tt.CLASS_A_ARMS) | set(tt.CLASS_B_ARMS)
    _removed |= {
        c for c in absent
        if (dc.parse_cell_id(c).method not in _cartesian
            or dc.parse_cell_id(c).si_id in cm.PETRI_OBJECT_ONLY_SI_IDS)}
    for cell_id, reason in absent.items():
        if cell_id in _removed:
            assert "not in the defense-cell registry" in reason
            continue
        expected = tt.object_path_for(store.absolute(),
                                      ra.traintest_cell_id(cell_id))
        assert expected.is_absolute()
        assert str(expected) in reason


# --------------------------------------------------------------------------
# the driver wiring: petri_run.load_cell_source builds the real adapter
# --------------------------------------------------------------------------


def _petri_run():
    import sys

    sys.path.insert(0, str(REPO / "scripts"))
    import petri_run

    return petri_run


def test_load_cell_source_refuses_without_the_authoritative_resolver(
        monkeypatch):
    """With no injected factory the driver builds the real adapter over
    configs/two_si.yaml — whose defenses_dir is the RELATIVE in-checkout
    default. With the training side's resolve_defenses_dir absent and no
    explicit override, the build must REFUSE loudly: silently enumerating
    the in-checkout dir would report every object cell ABSENT forever
    (indistinguishable from 'not trained yet') now that the writers place
    objects in the absolute commit-independent store. The factory
    override still wins when a test injects one."""
    from sisteer.eval import defense_cell as tt

    pr = _petri_run()
    assert pr.CELL_SOURCE_FACTORY is None  # the injection seam is unset
    monkeypatch.delattr(tt, "resolve_defenses_dir", raising=False)
    with pytest.raises(ra.RegistryAdapterError) as ei:
        pr.load_cell_source(_spec())
    msg = str(ei.value)
    assert "sisteer.eval.defense_cell.resolve_defenses_dir" in msg
    assert "D58" in msg
    # and the factory override still wins when a test injects one
    sentinel = dc.MappingCellSource({})
    pr.CELL_SOURCE_FACTORY = lambda spec: sentinel
    try:
        assert pr.load_cell_source(_spec()) is sentinel
    finally:
        pr.CELL_SOURCE_FACTORY = None


def test_enumerate_petri_over_the_repo_registry_never_drops_a_cell(tmp_path):
    """End-to-end through the driver's own loaders (real two-SI config,
    real behavior files, real SI bytes; synthetic prompt bytes — the
    package-pinned provenance needs the petri venv), with the objects
    store given EXPLICITLY (the launch-time override path). Every
    manifest cell is either served or reported ABSENT — regardless of
    how many objects the still-running training has produced
    so far."""
    pr = _petri_run()
    spec = _spec()
    store = tmp_path / "two_si_defenses" / "cells"  # empty: nothing trained
    source = ra.traintest_cell_source(spec, repo_root=REPO,
                                      defenses_dir=store)
    cs = pr.enumerate_petri(spec, source=source, prompt_files=PROMPT_FILES)
    served = {c.cell_id for c in cs.canonical}
    absent = {a.cell_id for a in cs.absent}
    assert served | absent == set(CANONICAL_22)
    assert not served & absent
    assert {"none", "prompted__safety", "prompted__constitution"} <= served


# --------------------------------------------------------------------------
# serve variants through the seam
# --------------------------------------------------------------------------

TT_PSR = "safety_refusal__psr_kl_all"
H_PSR = "si_safety__psr_kl_all"
OBJECT_22 = tuple(c for c in CANONICAL_22
                  if c != "none" and not c.startswith("prompted__"))


def test_variant_id_translation_both_ways():
    """§2.5.2, derived not spelled: the harness SI for safety_refusal is
    'safety', and the variant rides through both directions unchanged."""
    assert ra.harness_cell_id(f"{TT_PSR}@thinking") == f"{H_PSR}@thinking"
    assert ra.traintest_cell_id(f"{H_PSR}@thinking") == f"{TT_PSR}@thinking"
    for variant in sd.SERVE_VARIANTS:
        for hid in OBJECT_22:
            full = f"{hid}@{variant}"
            tt = ra.traintest_cell_id(full)
            assert tt.endswith(f"@{variant}")
            assert tt == f"{ra.traintest_cell_id(hid)}@{variant}"
            assert ra.harness_cell_id(tt) == full
    # bare Class B ids coincide, suffix included
    assert ra.harness_cell_id("antidote@all_turns") == "antidote@all_turns"
    assert ra.traintest_cell_id("antidote@all_turns") == "antidote@all_turns"


@pytest.mark.parametrize("bad", [
    "si_safety_refusal__psr_kl_all@thinking",  # the training-side spelling
    "none@thinking", "prompted__safety@thinking",
    f"{H_PSR}@bogus", f"{H_PSR}@thinking@thinking", "@thinking",
])
def test_forward_translation_refuses_bad_variant_ids(bad):
    with pytest.raises(dc.DefenseCellError):
        ra.traintest_cell_id(bad)


@pytest.mark.parametrize("bad", [
    "none@thinking",                          # object cells only
    "prompted__safety_refusal@thinking",
    f"{TT_PSR}@bogus",                        # closed vocabulary
    f"{TT_PSR}@thinking@all_turns",
    "@thinking", f"{TT_PSR}@",
    "safety_refusal__alphakl_0@thinking",     # a training-side-only arm
    "si_nope__psr_kl_all@thinking",           # unmapped SI
])
def test_reverse_translation_refuses_bad_variant_ids(bad):
    with pytest.raises(ra.RegistryAdapterError):
        ra.harness_cell_id(bad)


def _write_psr_blob(tmp_path, tt_id=TT_PSR, **fields):
    """A real-shaped response-fit PSR blob (the PSRObject.save schema) in
    the tmp store, at the path the registry resolves for ``tt_id``."""
    cell_dir = tmp_path / "cells" / tt_id
    cell_dir.mkdir(parents=True, exist_ok=True)
    layers = [0, 2]
    blob = {"kind": "psr", "layers": layers, "train_direction": False,
            "positions": "response", "response_header_len": 7,
            "directions": {str(l): torch.zeros(8) for l in layers},
            "gate_w": {str(l): torch.zeros(8) for l in layers},
            "gate_b": {str(l): torch.zeros(1) for l in layers}}
    blob.update(fields)
    torch.save(blob, cell_dir / "object.pt")
    return cell_dir / "object.pt"


def test_variant_cell_resolves_the_base_artifact_with_the_one_overlay(tmp_path):
    """Through the REAL seam: the base spec already carries
    positions='response' (copied from the blob); a variant entry is
    the base entry — same object_path, sidecar, kind, SI, bytes — with the
    overlay applied to the spec: positions REPLACED by the variant's, one
    declaration written, the fit regime read from the bytes untouched."""
    from sisteer.capeval.serving.lens_adapters import recorded_fit_positions

    artifact = _write_psr_blob(tmp_path)
    source = _source(tmp_path)
    base = source.get(H_PSR)
    assert base.resolved_spec["positions"] == "response"
    assert base.object_path == str(artifact)
    base_snapshot = dict(base.resolved_spec)
    raw = artifact.read_bytes()
    sha_before = __import__("hashlib").sha256(raw).hexdigest()

    want_positions = {"thinking": "response", "all_turns": "response_all_turns",
                      "all_positions": "all"}
    for variant in sd.SERVE_VARIANTS:
        full = f"{H_PSR}@{variant}"
        entry = source.get(full)
        assert entry.cell_id == full
        assert (entry.kind, entry.si_id, entry.serving_path) == (
            "psr", "safety", "lens")
        assert entry.object_path == base.object_path
        assert entry.sidecar_path == base.sidecar_path
        assert entry.resolved_spec == sd.apply_variant(base_snapshot, variant)
        assert entry.resolved_spec["positions"] == want_positions[variant]
        assert sd.declarations_of(entry.resolved_spec) == \
            sd.variant_declarations(variant)
        # the base entry's spec object is unmodified by the overlay
        assert base.resolved_spec == base_snapshot
    # the bytes are the same bytes, and the fit regime is still 'response'
    assert __import__("hashlib").sha256(artifact.read_bytes()).hexdigest() \
        == sha_before
    assert recorded_fit_positions("psr", raw, b"") == "response"
    # an untrained base -> the variant is ABSENT the same way (spec None,
    # path resolved), never a fabricated spec
    absent = source.get("si_constitution__psr_kl_all@thinking")
    assert absent is not None and absent.resolved_spec is None
    assert absent.object_path is not None
    # a cell the registry lacks -> None for the variant too
    narrow = ra.traintest_cell_source(
        _spec(), repo_root=REPO,
        config_path=_write_cfg(tmp_path, class_a_arms=["psr_kl_all"]),
        defenses_dir=tmp_path / "cells")
    assert narrow.get("si_safety__lora_r1@thinking") is None
    assert narrow.get(f"{H_PSR}@thinking") is not None


def test_full_id_chain_traintest_to_request_model(tmp_path):
    """§1.4 / §2.5.2: a training-side id -> harness_cell_id -> registry
    entry -> enumerated cell -> routing-table route -> the request's
    `model` field, byte for byte, with the un-translated spelling refused."""
    from sisteer.capeval.serving import proxy as px
    from sisteer.eval.attacks.served_target import route_for

    tt = f"{TT_PSR}@thinking"
    hid = ra.harness_cell_id(tt)
    assert hid == "si_safety__psr_kl_all@thinking"
    _write_psr_blob(tmp_path)
    source = _source(tmp_path)
    spec = _spec(cells=["none", "prompted__safety", "prompted__constitution",
                        H_PSR, hid])
    cs = _enumerate(source, spec)
    assert not cs.absent
    by_id = {c.cell_id: c for c in cs.canonical}
    cell, base = by_id[hid], by_id[H_PSR]
    assert cell.virtual_model_name == "petri__si_safety__psr_kl_all@thinking"
    assert base.virtual_model_name == "petri__si_safety__psr_kl_all"
    # same artifact, DISTINCT served identity (the hashed spec carries the
    # declaration) — the OFF and ON steered arms are two lens identities
    assert cell.object_path == base.object_path
    assert cell.intervention_hash != base.intervention_hash
    assert cell.serving_identity == base.serving_identity
    # exactly one twin, none__lens0, never suffixed
    assert [c.cell_id for c in cs.controls] == ["none__lens0"]
    assert cs.controls[0].virtual_model_name == "petri__none__lens0"
    # the routing table's route and the request `model` field
    routes = px.routes_from_cells(cs)
    route = {r.cell_id: r for r in routes}[hid]
    assert route.virtual_model == "petri__si_safety__psr_kl_all@thinking"
    assert route.intervention_hash == cell.intervention_hash
    assert route_for(tt, routes).virtual_model == \
        "petri__si_safety__psr_kl_all@thinking"
    assert route_for(TT_PSR, routes).virtual_model == \
        "petri__si_safety__psr_kl_all"
    with pytest.raises(dc.DefenseCellError):
        dc.parse_cell_id("si_safety_refusal__psr_kl_all@thinking")


def test_traintest_side_id_splitting(tmp_path):
    """defense_cell.si_of / arm_of / legacy_cell_id / object_path_for strip
    the variant first; build_registry never mints one."""
    from sisteer.eval import defense_cell as tt

    assert tt.si_of(f"{TT_PSR}@thinking") == "safety_refusal"
    assert tt.arm_of(f"{TT_PSR}@thinking") == "psr_kl_all"
    assert tt.si_of("stacked__safety_refusal__psr_kl_all@all_turns") == \
        "safety_refusal"
    assert tt.arm_of("stacked__safety_refusal__psr_kl_all@all_turns") == \
        "psr_kl_all"
    assert tt.arm_of("antidote@all_positions") == "antidote"
    assert tt.si_of("antidote@all_positions") is None
    assert tt.legacy_cell_id("safety_refusal__lora_r4@thinking") == \
        "safety_refusal__lora_r1_all@thinking"
    assert tt.legacy_cell_id(f"{TT_PSR}@thinking") is None
    ddir = tmp_path / "cells"
    assert tt.object_path_for(ddir, f"{TT_PSR}@thinking") == \
        tt.object_path_for(ddir, TT_PSR)
    assert "@" not in str(tt.object_path_for(ddir, f"{TT_PSR}@all_turns"))
    with pytest.raises(ValueError):
        tt.arm_of(f"{TT_PSR}@bogus")
    with pytest.raises(ValueError):
        tt.si_of(f"{TT_PSR}@thinking@thinking")
    assert not any("@" in c.id for c in tt.build_registry(ddir))
