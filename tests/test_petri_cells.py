"""Petri Bloom Phase 1: the defense-cell registry adapter over a SYNTHETIC
registry (the real TrainTest DefenseCell registry plugs
into the same CellSource interface at rebase).

Covers: the 22 canonical cells + derived path-matched control legs;
distinct prompted vmodels carrying SI-prefix hashes; the role-aware
two-tier check_pairing semantics (accept/refuse set incl. the full-key
certificate binding); ABSENT reporting; malformed-id / inconsistent-
registry refusals; the three separated keys (serving-invariant
intervention hash; judge-prompt-BYTES and bloom-option edits re-key the
harness hash + run key only; recursive UNFROZEN rejection).

CPU-offline; no inspect_ai / petri_bloom imports anywhere in Phase 1."""

import dataclasses
import os
from pathlib import Path

import pytest

from sisteer.capeval import manifest as cm
from sisteer.capeval.petri import defense_cells as dc
from sisteer.capeval.serving import serve_declaration as sd

ROOT = Path(__file__).parents[1]
REV = "0123456789abcdef0123456789abcdef01234567"
IMPL_COMMIT = "fedcba9876" * 4
SAFETY_PREFIX = b"synthetic safety instruction: be safe (test bytes)"
CONSTITUTION_PREFIX = b"synthetic constitution instruction: many principles (test bytes)"

# The two-SI audit's canonical matrix, DERIVED from the constants (so it
# grew with the deliberate vocabulary widenings — lora_r4 joined
# CLASS_A_METHODS 2026-08-28). The name keeps its historical "22"; the
# count assertions below derive from len() rather than repeating it.
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

# the FULL expected behavior-dir inventory: with the safety+constitution
# prompted cells in play, every behavior needs its canonical dir AND those
# two SI-prefix variant dirs — all of them hash inputs, none omittable.
# Deliberately NOT every registered variant: the irrelevant SI
# (vocabulary of 2026-08-28) has no prompted cell in this spec, so its
# variant dir is not in play and an extra key would refuse.
BEHAVIOR_FILES = {
    **{b: f"# behavior {b}\nsynthetic eval bytes\n".encode()
       for b in cm.PETRI_BEHAVIORS},
    **{f"{b}/{v}": f"# variant {v} of {b}\n".encode()
       for b in cm.PETRI_BEHAVIORS
       for v in (cm.PETRI_SI_VARIANT_DIRS["safety"],
                 cm.PETRI_SI_VARIANT_DIRS["constitution"])},
}

# the exact auditor/judge prompt BYTES in use — harness-hash inputs (H10)
PROMPT_FILES = {
    "auditor": b"synthetic auditor prompt: probe the behavior (test bytes)",
    "judge": b"synthetic judge prompt: score 1-10 (test bytes)",
}

BLOOM_OPTIONS = dict(max_turns=40, enable_prefill=False,
                     enable_rollback=False, compaction=False,
                     realism_filter=False, epochs=1, seed=0)


def _spec(**over) -> cm.PetriSpec:
    kwargs = dict(
        audit="petri-twosi-test",
        behaviors=list(cm.PETRI_BEHAVIORS),
        cells=list(CANONICAL_22),
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


def _source(tmp_path, *, drop=(), lora_methods=("lora_r1",)):
    """A synthetic CellSource: real bytes on disk for every object cell so
    intervention hashes cover genuine artifact bytes."""
    entries = {
        "none": dc.RegistryEntry(cell_id="none", kind="none"),
        "prompted__safety": dc.RegistryEntry(
            cell_id="prompted__safety", kind="prompted", si_id="safety",
            si_prefix=SAFETY_PREFIX),
        "prompted__constitution": dc.RegistryEntry(
            cell_id="prompted__constitution", kind="prompted",
            si_id="constitution", si_prefix=CONSTITUTION_PREFIX),
    }
    for cell_id in CANONICAL_22[3:]:
        if cell_id in dc.CLASS_B_IDS:
            kind, si, method = f"obj_{cell_id}", None, cell_id
        else:
            head, _, method = cell_id.partition("__")
            si = head[len("si_"):]
            kind = f"obj_{method}"
        artifact = tmp_path / f"{cell_id}.pt"
        artifact.write_bytes(b"synthetic-object-bytes:" + cell_id.encode())
        entries[cell_id] = dc.RegistryEntry(
            cell_id=cell_id, kind=kind, si_id=si, object_path=str(artifact),
            serving_path=("lora" if method in lora_methods else "lens"),
            resolved_spec={"kind": kind, "si_id": si, "layer": 15,
                           "scale": 1.0})
    for cell_id in drop:
        entries.pop(cell_id, None)
    return dc.MappingCellSource(entries)


def _enumerate(tmp_path, spec=None, prompt_files=None, **source_kw):
    spec = spec or _spec()
    return dc.enumerate_cells(spec, _source(tmp_path, **source_kw),
                              behavior_files=BEHAVIOR_FILES,
                              prompt_files=prompt_files or PROMPT_FILES)


# --------------------------------------------------------------------------
# enumeration: 22 canonical + derived control legs
# --------------------------------------------------------------------------


def test_enumerates_22_canonical_plus_control_legs(tmp_path):
    cs = _enumerate(tmp_path)
    assert len(cs.canonical) == len(CANONICAL_22)
    assert cs.absent == ()
    # lora_r1 is LoRA-served -> both no-op legs derive: +2 served
    control_ids = [c.cell_id for c in cs.controls]
    assert control_ids == ["none__lens0", "none__lora0"]
    assert len(cs.all_served()) == len(CANONICAL_22) + 2
    by_id = {c.cell_id: c for c in cs.all_served()}
    # class partition
    classes = {c.cell_id: c.cell_class for c in cs.canonical}
    assert classes["none"] == "base"
    assert classes["prompted__safety"] == "prompted"
    assert classes["si_constitution__psr_kl_all"] == "class_a"
    assert classes["antidote"] == "class_b"
    # serving identities: passthrough for base/prompted, lens/lora for
    # objects, all in the spec's pinned mode
    assert by_id["none"].serving_identity == "proxy_vllm+passthrough+exact_replay"
    assert by_id["si_safety__psr_kl_all"].serving_identity == "proxy_vllm+lens+exact_replay"
    assert by_id["si_safety__lora_r1"].serving_identity == "proxy_vllm+lora+exact_replay"
    # control legs are path-matched no-ops of the base, marked control_of,
    # with the base's OWN intervention hash (same intervention, different
    # serving)
    lens0, lora0 = by_id["none__lens0"], by_id["none__lora0"]
    assert lens0.control_of == "none" and lora0.control_of == "none"
    assert lens0.serving_identity == "proxy_vllm+lens+exact_replay"
    assert lora0.serving_identity == "proxy_vllm+lora+exact_replay"
    assert lens0.intervention_hash == by_id["none"].intervention_hash
    assert lora0.intervention_hash == by_id["none"].intervention_hash
    # distinct run keys everywhere (cell_id + harness enter the key)
    run_keys = [c.petri_run_key for c in cs.all_served()]
    assert len(run_keys) == len(set(run_keys))
    # one vmodel per served leg
    vmap = cs.virtual_model_map()
    assert len(vmap) == len(CANONICAL_22) + 2
    assert vmap["petri__base"].cell_id == "none"
    assert vmap["petri__si_constitution__cast"].cell_id == "si_constitution__cast"


def test_no_lora0_leg_when_no_lora_arm_is_live(tmp_path):
    cs = _enumerate(tmp_path, lora_methods=())
    assert [c.cell_id for c in cs.controls] == ["none__lens0"]
    assert len(cs.all_served()) == len(CANONICAL_22) + 1


def test_prompted_vmodels_distinct_and_carry_prefix_hashes(tmp_path):
    cs = _enumerate(tmp_path)
    by_id = {c.cell_id: c for c in cs.canonical}
    ps, pc = by_id["prompted__safety"], by_id["prompted__constitution"]
    # DISTINCT vmodels, never the generic passthrough (H8)
    assert ps.virtual_model_name == "petri__prompted__safety"
    assert pc.virtual_model_name == "petri__prompted__constitution"
    assert "petri__base" not in (ps.virtual_model_name, pc.virtual_model_name)
    # each carries its exact SI-prefix hash for the proxy's verification
    assert ps.si_prefix_hash == cm.sha256_hex16(SAFETY_PREFIX)
    assert pc.si_prefix_hash == cm.sha256_hex16(CONSTITUTION_PREFIX)
    # different SI bytes -> different intervention identities
    assert ps.intervention_hash != pc.intervention_hash
    # ... and neither equals the plain base's
    assert ps.intervention_hash != by_id["none"].intervention_hash


def test_prompted_prefix_mismatch_refuses(tmp_path):
    spec = _spec(prompted_prefix_hashes={
        "safety": "00" * 8,  # wrong recorded hash
        "constitution": cm.sha256_hex16(CONSTITUTION_PREFIX),
    })
    with pytest.raises(dc.DefenseCellError, match="masquerade|prefix"):
        dc.enumerate_cells(spec, _source(tmp_path),
                           behavior_files=BEHAVIOR_FILES,
                           prompt_files=PROMPT_FILES)


def test_no_hf_serving_fallback_exists(tmp_path):
    """The HF-shim serving fallback is GONE (a kind the proxy cannot serve
    is a hard §10 gate failure, never an HF-served cell): RegistryEntry
    carries no hf_exception field, no ``*__hf`` control leg is ever
    derived, and every served leg's identity is proxy_vllm."""
    with pytest.raises(TypeError):
        dc.RegistryEntry(cell_id="none", kind="none", hf_exception=True)
    assert "hf" not in dc._CONTROL_SUFFIXES
    assert "hf" not in dc._CONTROL_LEG_SERVING
    cs = _enumerate(tmp_path)
    assert not any(c.cell_id.endswith("__hf") for c in cs.all_served())
    for c in cs.all_served():
        ident = cm.parse_serving_identity(c.serving_identity)
        assert ident.backend == "proxy_vllm"


# --------------------------------------------------------------------------
# ABSENT vs loud refusal
# --------------------------------------------------------------------------


def test_missing_registry_cell_reports_absent(tmp_path):
    cs = _enumerate(tmp_path, drop=("si_safety__lofit",))
    assert len(cs.canonical) == len(CANONICAL_22) - 1
    assert [a.cell_id for a in cs.absent] == ["si_safety__lofit"]
    assert "registry" in cs.absent[0].reason


def test_missing_object_artifact_reports_absent(tmp_path):
    source = _source(tmp_path)
    Path(source.entries["si_constitution__loreft"].object_path).unlink()
    cs = dc.enumerate_cells(_spec(), source, behavior_files=BEHAVIOR_FILES,
                           prompt_files=PROMPT_FILES)
    absent = {a.cell_id: a.reason for a in cs.absent}
    assert set(absent) == {"si_constitution__loreft"}
    assert "missing" in absent["si_constitution__loreft"]
    # ABSENT is reported, never raised, and never silently dropped
    assert len(cs.canonical) == len(CANONICAL_22) - 1


def test_absent_reasons_name_the_absolute_checked_path(tmp_path, monkeypatch):
    """Every ABSENT reason names the FULLY-RESOLVED ABSOLUTE path that was
    actually checked — even when the registry entry carries a RELATIVE
    object/sidecar path. A store resolved against the wrong root (the
    wrong-directory failure mode) must be visible at a glance in
    print-cells output, never masquerade as 'not trained yet'."""
    source = _source(tmp_path)
    entries = dict(source.entries)
    monkeypatch.chdir(tmp_path)
    rel = Path("relative-store") / "object.pt"  # does not exist
    entries["si_safety__vector_all"] = dataclasses.replace(
        entries["si_safety__vector_all"], object_path=str(rel))
    # a declared-but-missing RELATIVE sidecar reports the same way
    sc_rel = Path("relative-store") / "object.json"
    entries["si_constitution__cast"] = dataclasses.replace(
        entries["si_constitution__cast"], sidecar_path=str(sc_rel))
    cs = dc.enumerate_cells(_spec(), dc.MappingCellSource(entries),
                            behavior_files=BEHAVIOR_FILES,
                            prompt_files=PROMPT_FILES)
    absent = {a.cell_id: a.reason for a in cs.absent}
    assert set(absent) == {"si_safety__vector_all", "si_constitution__cast"}
    # the reasons name where the check actually looked: absolute, cwd-rooted
    checked = Path.cwd() / rel
    assert checked.is_absolute()
    assert str(checked) in absent["si_safety__vector_all"]
    assert str(Path.cwd() / sc_rel) in absent["si_constitution__cast"]


def test_missing_object_path_reports_absent(tmp_path):
    """A registry entry with object_path=None is missing DATA, not a
    harness bug: the cell reports ABSENT and every other cell still
    enumerates — never an uncaught raise that discards the completeness
    map."""
    source = _source(tmp_path)
    entries = dict(source.entries)
    entries["si_safety__vector_all"] = dataclasses.replace(
        entries["si_safety__vector_all"], object_path=None)
    cs = dc.enumerate_cells(_spec(), dc.MappingCellSource(entries),
                            behavior_files=BEHAVIOR_FILES,
                            prompt_files=PROMPT_FILES)
    absent = {a.cell_id: a.reason for a in cs.absent}
    assert set(absent) == {"si_safety__vector_all"}
    assert "object_path" in absent["si_safety__vector_all"]
    assert len(cs.canonical) == len(CANONICAL_22) - 1


def test_unreadable_artifact_and_sidecar_report_absent(tmp_path):
    """An artifact/sidecar that exists but cannot be READ (OSError /
    PermissionError) reports ABSENT with a concrete reason — the rest of
    the enumeration survives."""
    if os.geteuid() == 0:
        pytest.skip("chmod 000 does not block reads for root")
    source = _source(tmp_path)
    entries = dict(source.entries)
    # unreadable artifact
    art = Path(entries["si_safety__lofit"].object_path)
    art.chmod(0o000)
    # declared sidecar that exists but is unreadable
    sc = tmp_path / "si_constitution__cast.sidecar.json"
    sc.write_bytes(b"{}")
    sc.chmod(0o000)
    entries["si_constitution__cast"] = dataclasses.replace(
        entries["si_constitution__cast"], sidecar_path=str(sc))
    try:
        cs = dc.enumerate_cells(_spec(), dc.MappingCellSource(entries),
                                behavior_files=BEHAVIOR_FILES,
                                prompt_files=PROMPT_FILES)
    finally:
        art.chmod(0o600)
        sc.chmod(0o600)
    absent = {a.cell_id: a.reason for a in cs.absent}
    assert set(absent) == {"si_safety__lofit", "si_constitution__cast"}
    assert "unreadable" in absent["si_safety__lofit"]
    assert "sidecar unreadable" in absent["si_constitution__cast"]
    assert len(cs.canonical) == len(CANONICAL_22) - 2


def test_class_a_wrong_si_attribution_refuses(tmp_path):
    """A Class A registry entry whose si_id contradicts the id-derived SI
    is a loud refusal BEFORE hashing/serving — the object must never be
    executed as one SI's compilation and reported as the other's."""
    source = _source(tmp_path)
    entries = dict(source.entries)
    entries["si_safety__cast"] = dataclasses.replace(
        entries["si_safety__cast"], si_id="constitution")
    with pytest.raises(dc.DefenseCellError, match="wrong SI|si_id"):
        dc.enumerate_cells(_spec(), dc.MappingCellSource(entries),
                           behavior_files=BEHAVIOR_FILES,
                           prompt_files=PROMPT_FILES)


def test_class_a_contradictory_resolved_spec_si_refuses(tmp_path):
    source = _source(tmp_path)
    entries = dict(source.entries)
    e = entries["si_safety__cast"]
    entries["si_safety__cast"] = dataclasses.replace(
        e, resolved_spec={**dict(e.resolved_spec), "si_id": "constitution"})
    with pytest.raises(dc.DefenseCellError, match="contradicts"):
        dc.enumerate_cells(_spec(), dc.MappingCellSource(entries),
                           behavior_files=BEHAVIOR_FILES,
                           prompt_files=PROMPT_FILES)


def test_class_b_si_bearing_entry_refuses(tmp_path):
    """Class B baselines are SI-agnostic: a registry entry or resolved
    spec that carries an SI is an inconsistent registry, refused."""
    source = _source(tmp_path)
    entries = dict(source.entries)
    entries["antidote"] = dataclasses.replace(
        entries["antidote"], si_id="safety")
    with pytest.raises(dc.DefenseCellError, match="SI-agnostic"):
        dc.enumerate_cells(_spec(), dc.MappingCellSource(entries),
                           behavior_files=BEHAVIOR_FILES,
                           prompt_files=PROMPT_FILES)
    entries = dict(source.entries)
    e = entries["alphasteer"]
    entries["alphasteer"] = dataclasses.replace(
        e, resolved_spec={**dict(e.resolved_spec), "si_id": "safety"})
    with pytest.raises(dc.DefenseCellError, match="SI-agnostic|SI-bearing"):
        dc.enumerate_cells(_spec(), dc.MappingCellSource(entries),
                           behavior_files=BEHAVIOR_FILES,
                           prompt_files=PROMPT_FILES)


def test_missing_resolved_spec_refuses(tmp_path):
    """An object cell with no resolved_spec refuses loudly — the adapter
    must supply the authoritative load_object resolution; a fabricated
    minimal spec would let different loaders apply different interventions
    under one intervention_hash."""
    source = _source(tmp_path)
    entries = dict(source.entries)
    entries["si_safety__loreft"] = dataclasses.replace(
        entries["si_safety__loreft"], resolved_spec=None)
    with pytest.raises(dc.DefenseCellError, match="resolved_spec"):
        dc.enumerate_cells(_spec(), dc.MappingCellSource(entries),
                           behavior_files=BEHAVIOR_FILES,
                           prompt_files=PROMPT_FILES)


@pytest.mark.parametrize("bad_id", [
    "si_paranoia__psr_kl_all",  # unknown SI
    "si_safety__not_a_method",  # method outside the TrainTest contract
    "prompted__both",  # unknown prompted SI
    "totally_bogus",  # matches no grammar
    "si_safety",  # Class A id with no method
    "none__lens0",  # derived control legs are never manifest inputs
    "prompted__safety__hf",  # the removed HF-repeat vocabulary stays illegal
])
def test_malformed_cell_id_refuses(tmp_path, bad_id):
    with pytest.raises(dc.DefenseCellError):
        dc.parse_cell_id(bad_id)
    spec = _spec(cells=list(CANONICAL_22) + [bad_id])
    with pytest.raises(dc.DefenseCellError):
        dc.enumerate_cells(spec, _source(tmp_path),
                           behavior_files=BEHAVIOR_FILES,
                           prompt_files=PROMPT_FILES)


def test_unfrozen_spec_refuses_at_enumeration(tmp_path):
    spec = _spec(scenario_bundle_hash=cm.PETRI_UNFROZEN)
    with pytest.raises(cm.ManifestError, match="UNFROZEN"):
        dc.enumerate_cells(spec, _source(tmp_path),
                           behavior_files=BEHAVIOR_FILES,
                           prompt_files=PROMPT_FILES)


def test_serving_backend_must_be_proxy_vllm():
    # B1: proxy-vLLM is the committed production primary; hf_shim is the
    # gate's fp32 reference oracle and never serves an audit cell
    bad = _spec()
    bad.serving_backend = "hf_shim"
    with pytest.raises(cm.ManifestError, match="proxy_vllm"):
        cm.validate_petri_spec(bad)


# --------------------------------------------------------------------------
# the three separated keys
# --------------------------------------------------------------------------


def test_intervention_hash_invariant_across_serving_identity(tmp_path):
    """The same intervention hashes identically whatever serves it: two
    enumerations differing only in determinism mode give equal
    intervention hashes and DIFFERENT harness hashes/run keys."""
    a = {c.cell_id: c for c in _enumerate(tmp_path, _spec()).canonical}
    b = {c.cell_id: c
         for c in _enumerate(tmp_path,
                             _spec(serving_mode="interleaved")).canonical}
    assert set(a) == set(b)
    for cell_id in a:
        assert a[cell_id].intervention_hash == b[cell_id].intervention_hash
        assert a[cell_id].serving_identity != b[cell_id].serving_identity
        assert a[cell_id].harness_config_hash != b[cell_id].harness_config_hash
        assert a[cell_id].petri_run_key != b[cell_id].petri_run_key
    # and directly at the hash function: no serving input exists at all
    art = (tmp_path / "si_safety__cast.pt").read_bytes()
    spec_dict = {"kind": "obj_cast", "si_id": "safety", "layer": 15,
                 "scale": 1.0}
    h1 = cm.petri_intervention_hash(spec_dict, artifact_bytes=art,
                                    sidecar_bytes=b"")
    h2 = cm.petri_intervention_hash(spec_dict, artifact_bytes=art,
                                    sidecar_bytes=b"")
    assert h1 == h2 == a["si_safety__cast"].intervention_hash


def test_judge_prompt_edit_rekeys_harness_not_intervention(tmp_path):
    """Editing the JUDGE PROMPT BYTES — with the prompt_provenance string
    left completely unchanged — changes harness_config_hash and
    petri_run_key but NEVER the intervention hash. The
    provenance label alone is not the binding; the resolved bytes are."""
    base = {c.cell_id: c for c in _enumerate(tmp_path, _spec()).canonical}
    edited_prompts = dict(PROMPT_FILES)
    edited_prompts["judge"] = b"synthetic judge prompt: EDITED scoring rules"
    edited = {c.cell_id: c
              for c in _enumerate(tmp_path, _spec(),
                                  prompt_files=edited_prompts).canonical}
    for cell_id in base:
        assert base[cell_id].intervention_hash == edited[cell_id].intervention_hash
        assert base[cell_id].harness_config_hash != edited[cell_id].harness_config_hash
        assert base[cell_id].petri_run_key != edited[cell_id].petri_run_key
    # the auditor prompt binds the same way
    edited_auditor = dict(PROMPT_FILES)
    edited_auditor["auditor"] = b"synthetic auditor prompt: EDITED probing"
    h1 = cm.petri_harness_config_hash(
        _spec(), behavior_files=BEHAVIOR_FILES, prompt_files=PROMPT_FILES,
        serving_identity="proxy_vllm+lens+exact_replay")
    h2 = cm.petri_harness_config_hash(
        _spec(), behavior_files=BEHAVIOR_FILES, prompt_files=edited_auditor,
        serving_identity="proxy_vllm+lens+exact_replay")
    assert h1 != h2


def test_override_provenance_verified_against_prompt_bytes():
    """An override@<hex> provenance DECLARES the prompt-bytes hash: the
    matching bytes are accepted; any other bytes (an edit that forgot to
    update the declaration) refuse loudly (H10)."""
    declared = cm.petri_prompt_override_hash(PROMPT_FILES)
    spec = _spec(prompt_provenance=f"override@{declared}")
    h = cm.petri_harness_config_hash(
        spec, behavior_files=BEHAVIOR_FILES, prompt_files=PROMPT_FILES,
        serving_identity="proxy_vllm+lens+exact_replay")
    assert len(h) == cm.HASH_HEX_LEN
    edited = dict(PROMPT_FILES)
    edited["judge"] = b"edited judge prompt bytes"
    with pytest.raises(cm.ManifestError, match="override"):
        cm.petri_harness_config_hash(
            spec, behavior_files=BEHAVIOR_FILES, prompt_files=edited,
            serving_identity="proxy_vllm+lens+exact_replay")


def test_prompt_files_roles_and_bytes_validated():
    for bad in (None, {"auditor": PROMPT_FILES["auditor"]},  # judge missing
                {**PROMPT_FILES, "extra": b"x"},
                {"auditor": b"", "judge": b"x"},
                {"auditor": "not-bytes", "judge": b"x"}):
        with pytest.raises(cm.ManifestError, match="prompt_files"):
            cm.petri_harness_config_hash(
                _spec(), behavior_files=BEHAVIOR_FILES, prompt_files=bad,
                serving_identity="proxy_vllm+lens+exact_replay")


def test_behavior_bytes_edit_rekeys_harness(tmp_path):
    """A BEHAVIOR.md edit is a harness-hash input (H10): no stale sample
    can survive it."""
    spec = _spec()
    edited = dict(BEHAVIOR_FILES)
    edited["defer_to_users"] = b"# behavior defer_to_users\nEDITED\n"
    h1 = cm.petri_harness_config_hash(
        spec, behavior_files=BEHAVIOR_FILES, prompt_files=PROMPT_FILES,
        serving_identity="proxy_vllm+lens+exact_replay")
    h2 = cm.petri_harness_config_hash(
        spec, behavior_files=edited, prompt_files=PROMPT_FILES,
        serving_identity="proxy_vllm+lens+exact_replay")
    assert h1 != h2


def test_behavior_files_require_full_inventory():
    """The harness hash requires EXACTLY the derived inventory: canonical
    dirs plus every in-play SI-variant dir. A missing variant (whose bytes
    would drift unhashed) and an unrecognized key both refuse (H10)."""
    spec = _spec()
    expected = cm.petri_expected_behavior_keys(spec)
    assert sorted(BEHAVIOR_FILES) == expected  # the fixture IS the inventory
    # dropping an SI-variant key refuses even though the canonical remains
    missing = dict(BEHAVIOR_FILES)
    del missing["self_preservation/PROMPTED_SAFETY"]
    with pytest.raises(cm.ManifestError, match="required behavior dir"):
        cm.petri_harness_config_hash(
            spec, behavior_files=missing, prompt_files=PROMPT_FILES,
            serving_identity="proxy_vllm+lens+exact_replay")
    # an unrecognized variant key refuses too
    extra = dict(BEHAVIOR_FILES)
    extra["self_preservation/BOGUS_VARIANT"] = b"# smuggled\n"
    with pytest.raises(cm.ManifestError, match="unrecognized"):
        cm.petri_harness_config_hash(
            spec, behavior_files=extra, prompt_files=PROMPT_FILES,
            serving_identity="proxy_vllm+lens+exact_replay")
    # with only ONE prompted cell in play, the other SI's variant dirs are
    # not in the inventory: supplying them refuses, pruning them passes
    cells = [c for c in CANONICAL_22 if c != "prompted__constitution"]
    narrow_spec = _spec(cells=cells)
    with pytest.raises(cm.ManifestError, match="unrecognized"):
        cm.petri_harness_config_hash(
            narrow_spec, behavior_files=BEHAVIOR_FILES,
            prompt_files=PROMPT_FILES,
            serving_identity="proxy_vllm+lens+exact_replay")
    pruned = {k: v for k, v in BEHAVIOR_FILES.items()
              if not k.endswith("/PROMPTED_CONSTITUTION")}
    h = cm.petri_harness_config_hash(
        narrow_spec, behavior_files=pruned, prompt_files=PROMPT_FILES,
        serving_identity="proxy_vllm+lens+exact_replay")
    assert len(h) == cm.HASH_HEX_LEN


def test_bloom_option_edit_rekeys_harness(tmp_path):
    """Every bloom_audit task option (+ epochs + seed) is a harness-hash
    input: editing any of them re-keys harness_config_hash and
    petri_run_key but never the intervention hash (H10; runner-review
    blocker)."""
    base = {c.cell_id: c for c in _enumerate(tmp_path, _spec()).canonical}
    for field, value in (("max_turns", 15), ("enable_prefill", True),
                         ("enable_rollback", True), ("compaction", True),
                         ("realism_filter", True), ("epochs", 2),
                         ("seed", 7)):
        spec = _spec(bloom_options=cm.BloomOptions(
            **{**BLOOM_OPTIONS, field: value}))
        edited = {c.cell_id: c
                  for c in _enumerate(tmp_path, spec).canonical}
        for cell_id in base:
            assert (base[cell_id].intervention_hash
                    == edited[cell_id].intervention_hash), field
            assert (base[cell_id].harness_config_hash
                    != edited[cell_id].harness_config_hash), field
            assert (base[cell_id].petri_run_key
                    != edited[cell_id].petri_run_key), field


def test_bloom_options_validated():
    for field, value in (("max_turns", 0), ("max_turns", True),
                         ("epochs", 0), ("seed", "x"),
                         ("enable_prefill", 1), ("compaction", None)):
        spec = _spec(bloom_options=cm.BloomOptions(
            **{**BLOOM_OPTIONS, field: value}))
        with pytest.raises(cm.ManifestError, match="bloom_options"):
            cm.validate_petri_spec(spec)
    # and a spec whose bloom_options is not a BloomOptions at all
    spec = _spec(bloom_options=dict(BLOOM_OPTIONS))
    with pytest.raises(cm.ManifestError, match="bloom_options"):
        cm.validate_petri_spec(spec)


def test_unfrozen_refused_recursively_in_every_hash_path(tmp_path):
    """The exact UNFROZEN sentinel can never receive a stable hash through
    ANY input: nested resolved-spec fields, any harness-input field, or
    petri_run_key's cell_id. Raw bytes stay exempt from sentinel
    interpretation."""
    # nested in a resolved spec -> petri_intervention_hash refuses
    with pytest.raises(cm.ManifestError, match="UNFROZEN"):
        cm.petri_intervention_hash(
            {"kind": "obj_cast", "si_id": "safety", "scale": "UNFROZEN"},
            artifact_bytes=b"x", sidecar_bytes=b"")
    # a harness field with no named UNFROZEN check -> the recursive sweep
    for over in ({"auditor_model": cm.PETRI_UNFROZEN},
                 {"serving_launch_flags": ["--x", cm.PETRI_UNFROZEN]},
                 {"serving_packages": {"vllm": cm.PETRI_UNFROZEN}}):
        with pytest.raises(cm.ManifestError, match="UNFROZEN"):
            cm.petri_harness_config_hash(
                _spec(**over), behavior_files=BEHAVIOR_FILES,
                prompt_files=PROMPT_FILES,
                serving_identity="proxy_vllm+lens+exact_replay")
    # the run key refuses the sentinel as cell_id
    with pytest.raises(cm.ManifestError, match="UNFROZEN"):
        cm.petri_run_key("ab" * 8, cm.PETRI_UNFROZEN, "cd" * 8)
    # bytes are NOT sentinel-scanned: the literal b"UNFROZEN" payload is
    # opaque content and hashes fine
    h = cm.petri_intervention_hash(
        {"kind": "obj_cast", "si_id": "safety"},
        artifact_bytes=b"UNFROZEN", sidecar_bytes=b"")
    assert len(h) == cm.HASH_HEX_LEN


def test_harness_hash_refuses_mode_mismatch():
    """The audit runs only in the mode the gate certified (H5): a serving
    identity in a different mode than the spec's pin refuses."""
    with pytest.raises(cm.ManifestError, match="pinned"):
        cm.petri_harness_config_hash(
            _spec(), behavior_files=BEHAVIOR_FILES,
            prompt_files=PROMPT_FILES,
            serving_identity="proxy_vllm+lens+interleaved")


# --------------------------------------------------------------------------
# pairing semantics (the ROLE-AWARE accept/refuse set)
# --------------------------------------------------------------------------

LENS = "proxy_vllm+lens+exact_replay"
LORA = "proxy_vllm+lora+exact_replay"
PASSTHROUGH = "proxy_vllm+passthrough+exact_replay"
HF = "hf_shim+apply+exact_replay"
LENS_INTERLEAVED = "proxy_vllm+lens+interleaved"

GATE_HASH = "12" * 8


def _leg(cell_id, identity, control_of=None):
    return dc.PairingLeg(cell_id=cell_id, serving_identity=identity,
                         control_of=control_of)


ARM_LENS = _leg("si_safety__cast", LENS)
ARM_LENS_OTHER_SI = _leg("si_constitution__cast", LENS)
ARM_LORA = _leg("si_safety__lora_r1", LORA)
NOOP_LENS = _leg("none__lens0", LENS, control_of="none")
NOOP_LORA = _leg("none__lora0", LORA, control_of="none")
#: a foreign/corrupt row claiming an hf_shim identity — no audit cell is
#: hf-served anymore; pairing refuses it on backend.
HF_ROW = _leg("prompted__safety", HF)
BASE_PASS = _leg("none", PASSTHROUGH)
PROMPTED_PASS = _leg("prompted__safety", PASSTHROUGH)


def _certs(mode="exact_replay", gate_hash=GATE_HASH):
    return [
        dc.NoOpCertificate(backend="proxy_vllm", path=p, mode=mode,
                           model_revision=REV,
                           serving_impl_commit=IMPL_COMMIT,
                           gate_prompt_set_hash=gate_hash)
        for p in ("lens", "lora", "passthrough")
    ]


_FULL_KEY = dict(model_revision=REV, serving_impl_commit=IMPL_COMMIT,
                 gate_prompt_set_hash=GATE_HASH)


def test_pairing_tier1_accepts_lens_arm_vs_lens_noop_base():
    ok = dc.check_pairing(ARM_LENS, NOOP_LENS, tier=1)
    assert ok.tier == 1 and ok.cross_path is False
    assert {ok.cell_id_a, ok.cell_id_b} == {"si_safety__cast", "none__lens0"}


def test_pairing_tier1_accepts_lora_arm_vs_lora_noop_base():
    ok = dc.check_pairing(ARM_LORA, NOOP_LORA, tier=1)
    assert ok.cross_path is False


def test_pairing_tier1_accepts_within_method_two_si_contrast():
    # the confirmatory per-method estimand: si_constitution__<m> vs
    # si_safety__<m>, exact serving-identity equality
    ok = dc.check_pairing(ARM_LENS_OTHER_SI, ARM_LENS, tier=1)
    assert ok.tier == 1 and ok.cross_path is False


def test_pairing_tier1_accepts_prompted_arm_vs_passthrough_base():
    # a passthrough-served arm's path-matched no-op IS the canonical base
    ok = dc.check_pairing(PROMPTED_PASS, BASE_PASS, tier=1)
    assert ok.cross_path is False


def test_pairing_refuses_removed_hf_control_leg_vocabulary():
    """The ``*__hf`` control-leg vocabulary is GONE:
    a leg claiming it refuses as contradictory metadata, never resolves
    to a role."""
    with pytest.raises(dc.PairingError, match="contradictory"):
        dc.check_pairing(_leg("none__hf", HF, control_of="none"),
                         ARM_LENS, tier=1)
    with pytest.raises(dc.PairingError, match="contradictory"):
        dc.check_pairing(
            _leg("prompted__safety__hf", HF, control_of="prompted__safety"),
            ARM_LENS, tier=1)


def test_pairing_tier1_refuses_unrelated_cells_sharing_identity():
    """The failure case: two unrelated defended cells share a
    serving identity — role-blind identity equality would have accepted
    them as a confirmatory pair. Refused now."""
    for other in (_leg("si_constitution__psr_kl_all", LENS),  # other method
                  _leg("si_safety__psr_kl_all", LENS),  # same SI, other method
                  _leg("antidote", LENS)):  # Class B baseline
        with pytest.raises(dc.PairingError, match="unrelated|neither"):
            dc.check_pairing(ARM_LENS, other, tier=1)
    # two prompted baselines are not a confirmatory pair either
    with pytest.raises(dc.PairingError, match="neither"):
        dc.check_pairing(PROMPTED_PASS,
                         _leg("prompted__constitution", PASSTHROUGH), tier=1)
    # a cell never pairs against itself
    with pytest.raises(dc.PairingError, match="itself"):
        dc.check_pairing(ARM_LENS, _leg("si_safety__cast", LENS), tier=1)


def test_pairing_refuses_bare_identity_strings():
    # role metadata is REQUIRED: a bare identity string cannot say which
    # cell it is, so it can never license a pairing
    with pytest.raises(dc.PairingError, match="PairingLeg|role metadata"):
        dc.check_pairing(LENS, LENS, tier=1)


def test_pairing_refuses_contradictory_leg_metadata():
    # control leg served on the wrong path for its suffix
    with pytest.raises(dc.PairingError, match="contradictory"):
        dc.check_pairing(_leg("none__lens0", LORA, control_of="none"),
                         NOOP_LORA, tier=1)
    # control_of that does not match the leg's own id
    with pytest.raises(dc.PairingError, match="contradictory"):
        dc.check_pairing(_leg("si_safety__cast", LENS, control_of="antidote"),
                         NOOP_LENS, tier=1)
    # only the base derives control legs
    with pytest.raises(dc.PairingError, match="contradictory"):
        dc.check_pairing(_leg("antidote__lens0", LENS, control_of="antidote"),
                         NOOP_LENS, tier=1)
    # the canonical base is served passthrough, never on a steering path
    with pytest.raises(dc.PairingError, match="passthrough"):
        dc.check_pairing(_leg("none", LENS), ARM_LENS, tier=1)


def test_pairing_refuses_proxy_arm_vs_hf_row_both_tiers():
    # DEFENSIVE invariant (kept on purpose): no audit cell is hf-served
    # anymore, so an hf_shim-tagged row can only be foreign/corrupt —
    # pairing against it refuses on backend at both tiers.
    for tier in (1, 2):
        with pytest.raises(dc.PairingError, match="backend"):
            dc.check_pairing(ARM_LENS, HF_ROW, tier=tier,
                             certificates=_certs(), **_FULL_KEY)


def test_pairing_refuses_interleaved_vs_exact_replay_both_tiers():
    arm_interleaved = _leg("si_safety__cast", LENS_INTERLEAVED)
    for tier in (1, 2):
        with pytest.raises(dc.PairingError, match="mode"):
            dc.check_pairing(arm_interleaved, NOOP_LENS, tier=tier,
                             certificates=_certs(), **_FULL_KEY)


def test_pairing_tier1_refuses_lens_arm_vs_passthrough_base():
    # must pair against none__lens0, not the passthrough base
    with pytest.raises(dc.PairingError, match="path-matched|no-op"):
        dc.check_pairing(ARM_LENS, BASE_PASS, tier=1)


def test_pairing_tier2_cross_path_needs_mode_matching_certificates():
    # no certificates at all -> refused
    with pytest.raises(dc.PairingError, match="certificate"):
        dc.check_pairing(ARM_LENS, BASE_PASS, tier=2, certificates=(),
                         **_FULL_KEY)
    # certificates on record, but produced in a DIFFERENT determinism mode
    # -> refused (an exact_replay certificate never licenses interleaved
    # rows, and vice versa — H5)
    with pytest.raises(dc.PairingError, match="certificate"):
        dc.check_pairing(
            _leg("si_safety__cast", LENS_INTERLEAVED),
            _leg("none", "proxy_vllm+passthrough+interleaved"), tier=2,
            certificates=_certs(mode="exact_replay"), **_FULL_KEY)
    # only ONE path certified -> refused (need one per path)
    with pytest.raises(dc.PairingError, match="certificate"):
        dc.check_pairing(ARM_LENS, BASE_PASS, tier=2,
                         certificates=_certs()[:1], **_FULL_KEY)
    # full mode-matching certificates -> accepted, LABELLED cross-path
    ok = dc.check_pairing(ARM_LENS, BASE_PASS, tier=2,
                          certificates=_certs(), **_FULL_KEY)
    assert ok.tier == 2 and ok.cross_path is True


def test_pairing_tier2_refuses_certificate_from_other_prompt_set():
    """The certificate's FULL key includes gate_prompt_set_hash: a
    certificate produced against a different gate prompt set licenses
    nothing."""
    stale = _certs(gate_hash="34" * 8)
    with pytest.raises(dc.PairingError, match="certificate"):
        dc.check_pairing(ARM_LENS, BASE_PASS, tier=2, certificates=stale,
                         **_FULL_KEY)
    # and the lookup cannot even run without the expected gate hash
    with pytest.raises(dc.PairingError, match="full key"):
        dc.check_pairing(ARM_LENS, BASE_PASS, tier=2, certificates=_certs(),
                         model_revision=REV,
                         serving_impl_commit=IMPL_COMMIT)
    # a malformed certificate on record is a corrupt store -> loud refusal
    bad = [dataclasses.replace(_certs()[0], gate_prompt_set_hash="nope")]
    with pytest.raises(dc.PairingError, match="malformed"):
        dc.check_pairing(ARM_LENS, BASE_PASS, tier=2, certificates=bad,
                         **_FULL_KEY)


def test_pairing_tier2_same_path_needs_no_certificate():
    ok = dc.check_pairing(PROMPTED_PASS, BASE_PASS, tier=2)
    assert ok.cross_path is False


def test_pairing_tier2_cross_path_requires_full_key():
    with pytest.raises(dc.PairingError, match="full key"):
        dc.check_pairing(ARM_LENS, BASE_PASS, tier=2, certificates=_certs())


def test_pairing_refuses_malformed_identity_and_tier():
    # a malformed identity inside a leg refuses at parse
    with pytest.raises(cm.ManifestError):
        dc.check_pairing(_leg("si_safety__cast", "hf"), NOOP_LENS, tier=1)
    with pytest.raises(dc.PairingError, match="tier"):
        dc.check_pairing(ARM_LENS, NOOP_LENS, tier=3)


# --------------------------------------------------------------------------
# the committed manifest + serving-identity helpers
# --------------------------------------------------------------------------


def test_committed_petri_manifest_loads():
    spec = cm.load_petri_spec(
        ROOT / "configs" / "capeval" / "manifests" / "petri_twosi.yaml")
    assert spec.audit == "petri_twosi"
    # 22 original cells + prompted__irrelevant (the off-target control SI,
    # 2026-08-28 — compared on the same engine as everything else)
    assert len(spec.cells) == 23
    assert "prompted__irrelevant" in spec.cells
    assert tuple(spec.behaviors) == cm.PETRI_BEHAVIORS
    assert spec.serving_backend == "proxy_vllm"
    # interleaved since 2026-08-27: exact_replay dropped (BI=1 cost
    # 2.97x for same-size-only composition invariance on H200/SM90;
    # the audit is knowingly no longer bit-for-bit replayable).
    # test_petri_bi0_gating pins this to the smoke gate's certified
    # configuration, so the two cannot drift apart.
    assert spec.serving_mode == "interleaved"  # pinned (§3 H5)
    for cell_id in spec.cells:
        dc.parse_cell_id(cell_id)  # every committed id is well-formed
    # the committed bloom_audit options: the optional flags pinned OFF,
    # one epoch, one audit-wide seed (harness-hash inputs).
    # max_turns is 15 — Petri's bloom_audit default and the cost
    # bound (40 was an unjustified ~4x).
    assert spec.bloom_options == cm.BloomOptions(
        max_turns=15, enable_prefill=False, enable_rollback=False,
        compaction=False, realism_filter=False, epochs=1, seed=0)
    # FROZEN since the 2026-08-29 gemma-4-E4B launch (the one `bloom
    # scenarios` pass; petri-scenarios.lock.yaml is the committed record) —
    # so the committed manifest is no longer the UNFROZEN sentinel and can
    # key a run. The UNFROZEN-refusal behavior itself stays pinned by the
    # synthetic-spec tests below (petri_harness_config_hash and
    # petri_run_key both reject the sentinel); hashing the committed spec
    # here would need the full on-disk behavior-variant inventory (H10),
    # which is launch-tree state, not repo state.
    assert spec.scenario_bundle_hash == "8812ba083bb5473c"


def test_si_variant_dirs_consistent_with_scenarios_module():
    # manifest.PETRI_SI_VARIANT_DIRS is authoritative (the harness hash
    # derives the behavior-dir inventory from it); petri.scenarios mirrors
    # it — the two must never drift
    from sisteer.capeval.petri import scenarios as psc

    assert psc.SI_VARIANT_DIRS == cm.PETRI_SI_VARIANT_DIRS


def test_check_pairing_accepts_petri_cells_directly(tmp_path):
    """check_pairing consumes PetriCell objects as legs (PairingLeg is the
    row-reconstruction alternative): a lens arm from a real enumeration
    pairs Tier-1 against the derived none__lens0 leg."""
    by_id = {c.cell_id: c for c in _enumerate(tmp_path).all_served()}
    ok = dc.check_pairing(by_id["si_safety__cast"], by_id["none__lens0"],
                          tier=1)
    assert ok.cross_path is False
    assert dc.PairingLeg.from_cell(by_id["none__lens0"]).control_of == "none"
    with pytest.raises(dc.PairingError, match="neither"):
        dc.check_pairing(by_id["si_safety__cast"], by_id["antidote"], tier=1)


@pytest.mark.parametrize("bad", [
    "proxy_vllm+lens",  # missing mode
    "proxy_vllm+apply+exact_replay",  # path invalid for backend
    "hf_shim+lens+exact_replay",  # lens is not an hf_shim path
    "nobody+lens+exact_replay",  # unknown backend
    "proxy_vllm+lens+sometimes",  # unknown mode
    "hf",  # in-process capability value
    "",
])
def test_parse_serving_identity_refuses_malformed(bad):
    with pytest.raises(cm.ManifestError):
        cm.parse_serving_identity(bad)


def test_runtime_seams_dispatch_to_phase2_serving(monkeypatch):
    """The Phase-1 frozen seams are now FILLED (petri Phase-2 serving):
    make_engine's serving-identity branch dispatches to serving/proxy.py
    / serving/openai_shim.py with loud refusals on missing wiring, a
    MALFORMED identity still refuses at parse (never reads as a missing
    endpoint), and run_petri is a thin lazy delegate to
    tiers/petri_runner.run."""
    import sys
    from types import ModuleType, SimpleNamespace

    from sisteer.capeval import runtime as rt
    from sisteer.capeval.serving.proxy import (PROXY_BASE_URL_ENV,
                                               ProxyClientEngine, ProxyError)
    from sisteer.capeval.serving.openai_shim import ShimError

    # proxy backend: no endpoint configured -> loud refusal naming the env
    monkeypatch.delenv(PROXY_BASE_URL_ENV, raising=False)
    with pytest.raises(ProxyError, match=PROXY_BASE_URL_ENV):
        rt.make_engine(SimpleNamespace(engine="proxy_vllm+lens+exact_replay"))
    # ... and with an endpoint, the ProxyClientEngine loads (health check
    # stubbed — make_engine calls engine.load()) with identity stamped
    monkeypatch.setenv(PROXY_BASE_URL_ENV, "http://localhost:8814")
    monkeypatch.setattr(
        ProxyClientEngine, "_http_json",
        lambda self, url, payload: {"data": [{"id": "petri__base"}]})
    engine = rt.make_engine(
        SimpleNamespace(engine="proxy_vllm+lens+exact_replay"))
    assert isinstance(engine, ProxyClientEngine)
    assert engine.name == "proxy_vllm+lens+exact_replay"
    assert engine.served_models == ("petri__base",)
    # hf_shim backend: a manifest naming no model -> loud refusal
    with pytest.raises(ShimError, match="model"):
        rt.make_engine(SimpleNamespace(engine="hf_shim+apply+interleaved"))
    # malformed identity refuses at parse, before any wiring
    with pytest.raises(cm.ManifestError, match="path"):
        rt.make_engine(SimpleNamespace(engine="proxy_vllm+bogus+exact_replay"))
    # run_petri: a thin lazy delegate to tiers/petri_runner.run — assert
    # the delegation itself (robust to whether the real runner has landed)
    calls = {}
    fake = ModuleType("sisteer.capeval.tiers.petri_runner")

    def _run(engine, spec, cell, store, seed):
        calls["args"] = (engine, spec, cell, store, seed)
        return {"rows_emitted": 0, "behaviors": {}, "unscored": 0}

    fake.run = _run
    monkeypatch.setitem(sys.modules, "sisteer.capeval.tiers.petri_runner",
                        fake)
    # `from pkg import mod` prefers the PACKAGE ATTRIBUTE when the real
    # module was already imported by an earlier test — patch both surfaces
    import sisteer.capeval.tiers as tiers_pkg

    monkeypatch.setattr(tiers_pkg, "petri_runner", fake, raising=False)
    spec = _spec()
    out = rt.run_petri("ENGINE", spec, "CELL", "STORE", 7)
    assert calls["args"] == ("ENGINE", spec, "CELL", "STORE", 7)
    assert out == {"rows_emitted": 0, "behaviors": {}, "unscored": 0}


def test_engines_enum_widened_additively():
    # in-process capability values unchanged and first; identities appended
    assert cm.ENGINES[:2] == ("hf", "vllm")
    assert set(cm.SERVING_IDENTITIES) <= set(cm.ENGINES)
    for ident in cm.SERVING_IDENTITIES:
        parsed = cm.parse_serving_identity(ident)
        assert cm.serving_identity(parsed.backend, parsed.path,
                                   parsed.mode) == ident


# --------------------------------------------------------------------------
# serve variants: the ``@<variant>`` grammar (fit-support-transfer WS-0)
# --------------------------------------------------------------------------

OBJECT_CELLS = tuple(CANONICAL_22[3:])  # every Class A + Class B id
STACKED_EXAMPLES = ("stacked__safety__psr_kl_all", "stacked__constitution__antidote")
PSR_BASE = "si_safety__psr_kl_all"
LENS_IDENT = "proxy_vllm+lens+exact_replay"


def test_every_canonical_id_parses_unchanged_with_no_variant():
    """'@' never occurred in a cell id, so no existing id changes meaning:
    the round trip over every canonical id (and the stacked spellings) is
    byte-identical with serve_variant None and base_id == cell_id."""
    for cid in CANONICAL_22 + list(STACKED_EXAMPLES):
        p = dc.parse_cell_id(cid)
        assert p.cell_id == cid == p.base_id
        assert p.serve_variant is None


@pytest.mark.parametrize("variant", sorted(sd.SERVE_VARIANTS))
def test_variant_ids_round_trip_on_every_object_cell(variant):
    for cid in OBJECT_CELLS + STACKED_EXAMPLES:
        full = f"{cid}@{variant}"
        p, base = dc.parse_cell_id(full), dc.parse_cell_id(cid)
        assert p.cell_id == full
        assert p.base_id == cid
        assert p.serve_variant == variant
        assert (p.cell_class, p.si_id, p.method) == (
            base.cell_class, base.si_id, base.method)
        # the vmodel is the FULL id — distinct from the base cell's
        assert dc._vmodel(full, p.cell_class) == "petri__" + full
        assert dc._vmodel(full, p.cell_class) != dc._vmodel(cid, base.cell_class)


@pytest.mark.parametrize("bad", [
    "none@thinking",                           # the base serves no artifact
    "prompted__safety@thinking",               # a prompted cell neither
    "prompted__constitution@all_turns",
    "none__lens0@thinking",                    # control legs are derived
    "none__lora0@all_turns",
    "si_safety__psr_kl_all@",                  # empty variant
    "@thinking",                               # empty base
    "si_safety__psr_kl_all@thinking@thinking", # at most one '@'
    "si_safety__psr_kl_all@all_turns@all_positions",
    "si_safety__psr_kl_all@bogus",             # closed vocabulary
    "si_safety__psr_kl_all@THINKING",
    "si_safety__nope@thinking",                # malformed base
    "si_safety_refusal__psr_kl_all@thinking",  # the TrainTest spelling
    "stacked__safety__nope@thinking",
])
def test_variant_refusals(bad):
    with pytest.raises(dc.DefenseCellError):
        dc.parse_cell_id(bad)


def test_variant_refusal_names_the_full_id_and_the_rule():
    with pytest.raises(dc.DefenseCellError, match="OBJECT cells only") as ei:
        dc.parse_cell_id("none@thinking")
    assert "'none@thinking'" in str(ei.value)
    with pytest.raises(dc.DefenseCellError, match="closed") as ei:
        dc.parse_cell_id("si_safety__psr_kl_all@bogus")
    assert "'si_safety__psr_kl_all@bogus'" in str(ei.value)


def test_stacked_variant_legs():
    """The OBJECT leg is what a variant re-serves, so it carries the
    variant; the PROMPT leg never does."""
    p = dc.parse_cell_id("stacked__safety__psr_kl_all@all_turns")
    assert dc.stacked_object_cell_id(p) == "si_safety__psr_kl_all@all_turns"
    assert dc.stacked_prompted_cell_id(p) == "prompted__safety"
    q = dc.parse_cell_id("stacked__constitution__antidote@thinking")
    assert dc.stacked_object_cell_id(q) == "antidote@thinking"
    assert dc.stacked_prompted_cell_id(q) == "prompted__constitution"
    # unchanged without a variant
    r = dc.parse_cell_id("stacked__safety__psr_kl_all")
    assert dc.stacked_object_cell_id(r) == "si_safety__psr_kl_all"


def test_validate_spec_for_enumeration_reads_the_si_off_the_base_id():
    cells = ["none", "prompted__safety", "prompted__constitution",
             "stacked__safety__psr_kl_all@all_turns"]
    dc.validate_petri_spec_for_enumeration(_spec(cells=cells))  # ok
    with pytest.raises(dc.DefenseCellError, match="prompted_prefix_hashes"):
        dc.validate_petri_spec_for_enumeration(_spec(
            cells=["none", "prompted__constitution",
                   "stacked__safety__psr_kl_all@all_turns"],
            prompted_prefix_hashes={
                "constitution": cm.sha256_hex16(CONSTITUTION_PREFIX)}))


# --- enumeration over a bank with variant cells ------------------------------

VARIANT_CELLS = ["none", "prompted__safety", "prompted__constitution",
                 PSR_BASE, f"{PSR_BASE}@all_turns", f"{PSR_BASE}@all_positions"]


def _variant_source(tmp_path, *, overlay=None, base_spec_extra=None):
    """The synthetic bank plus variant entries for PSR_BASE: the SAME
    artifact file as the base, the base's spec with the variant applied
    (``overlay`` maps variant -> spec to force an inconsistent entry)."""
    entries = dict(_source(tmp_path).entries)
    base = entries[PSR_BASE]
    clean_spec = dict(base.resolved_spec)  # the variants overlay the CLEAN base
    if base_spec_extra:
        entries[PSR_BASE] = dataclasses.replace(
            base, resolved_spec={**clean_spec, **base_spec_extra})
    for variant in ("all_turns", "all_positions", "thinking"):
        full = f"{PSR_BASE}@{variant}"
        spec = (overlay or {}).get(variant) or sd.apply_variant(
            clean_spec, variant)
        entries[full] = dataclasses.replace(base, cell_id=full,
                                            resolved_spec=spec)
    return dc.MappingCellSource(entries)


def test_enumerate_variant_cells_distinct_hashes_one_twin(tmp_path):
    cs = dc.enumerate_cells(_spec(cells=VARIANT_CELLS),
                            _variant_source(tmp_path),
                            behavior_files=BEHAVIOR_FILES,
                            prompt_files=PROMPT_FILES)
    assert not cs.absent
    by_id = {c.cell_id: c for c in cs.canonical}
    base, turns, allp = (by_id[PSR_BASE], by_id[f"{PSR_BASE}@all_turns"],
                         by_id[f"{PSR_BASE}@all_positions"])
    # three DISTINCT intervention hashes over the SAME artifact bytes
    assert len({base.intervention_hash, turns.intervention_hash,
                allp.intervention_hash}) == 3
    assert base.object_path == turns.object_path == allp.object_path
    # distinct vmodels, in the variant spelling
    assert turns.virtual_model_name == f"petri__{PSR_BASE}@all_turns"
    assert allp.virtual_model_name == f"petri__{PSR_BASE}@all_positions"
    assert base.virtual_model_name == f"petri__{PSR_BASE}"
    assert len(cs.virtual_model_map()) == len(cs.all_served())
    # same class/SI/kind/identity as the base
    for v in (turns, allp):
        assert (v.cell_class, v.si_id, v.kind, v.serving_identity) == (
            base.cell_class, base.si_id, base.kind, base.serving_identity)
    # exactly ONE derived control leg, the base's none__lens0
    assert [c.cell_id for c in cs.controls] == ["none__lens0"]
    assert cs.controls[0].virtual_model_name == "petri__none__lens0"
    # declarations_of non-empty on BOTH variants, empty on the base
    src = _variant_source(tmp_path)
    assert sd.declarations_of(src.get(f"{PSR_BASE}@all_turns").resolved_spec) \
        == {"serve_experiment": "fit_response_serve_all_turns"}
    assert sd.declarations_of(src.get(f"{PSR_BASE}@all_positions").resolved_spec) \
        == {"serve_experiment": "fit_response_serve_all_positions"}
    assert sd.declarations_of(src.get(PSR_BASE).resolved_spec) == {}
    # and the run keys are distinct too (they carry the cell id + hash)
    assert len({c.petri_run_key for c in cs.canonical}) == len(cs.canonical)


def test_enumerate_refuses_a_variant_id_whose_spec_does_not_express_it(tmp_path):
    """The name says @all_turns; the spec is the un-overlaid base (or the
    OTHER variant's) — an inconsistent registry entry, refused before
    hashing, never served under a lying identity."""
    base_spec = _source(tmp_path).entries[PSR_BASE].resolved_spec
    for wrong in (dict(base_spec), sd.apply_variant(base_spec, "all_positions")):
        with pytest.raises(dc.DefenseCellError, match="does not express"):
            dc.enumerate_cells(_spec(cells=["none", "prompted__safety",
                                            "prompted__constitution",
                                            f"{PSR_BASE}@all_turns"]),
                               _variant_source(tmp_path,
                                               overlay={"all_turns": wrong}),
                               behavior_files=BEHAVIOR_FILES,
                               prompt_files=PROMPT_FILES)


def test_enumerate_refuses_a_declaration_the_name_does_not_carry(tmp_path):
    """A base id whose spec carries a declaration key: the declaration is
    invisible in the name (vmodel, transcript dir, manifest) — refused."""
    for extra in ({"serve_out_of_support": ["thinking"]},
                  {"serve_experiment": "fit_response_serve_all_turns"}):
        with pytest.raises(dc.DefenseCellError, match="names no serve variant"):
            dc.enumerate_cells(_spec(cells=["none", "prompted__safety",
                                            "prompted__constitution", PSR_BASE]),
                               _variant_source(tmp_path, base_spec_extra=extra),
                               behavior_files=BEHAVIOR_FILES,
                               prompt_files=PROMPT_FILES)
    # a MALFORMED declaration on a variant id refuses as malformed
    bad = dict(_source(tmp_path).entries[PSR_BASE].resolved_spec,
               positions="response_all_turns", serve_experiment="nope")
    with pytest.raises(dc.DefenseCellError, match="malformed serve declaration"):
        dc.enumerate_cells(_spec(cells=["none", "prompted__safety",
                                        "prompted__constitution",
                                        f"{PSR_BASE}@all_turns"]),
                           _variant_source(tmp_path, overlay={"all_turns": bad}),
                           behavior_files=BEHAVIOR_FILES,
                           prompt_files=PROMPT_FILES)


# --- pairing with variant cells ----------------------------------------------


def test_pairing_variant_arm_vs_twin_and_same_variant_contrast():
    twin = _leg("none__lens0", LENS_IDENT, control_of="none")
    for variant in sd.SERVE_VARIANTS:
        arm = _leg(f"{PSR_BASE}@{variant}", LENS_IDENT)
        ok = dc.check_pairing(arm, twin, tier=1)
        assert ok.tier == 1 and not ok.cross_path
        # shape (b): the within-method two-SI contrast under the SAME variant
        other = _leg(f"si_constitution__psr_kl_all@{variant}", LENS_IDENT)
        assert dc.check_pairing(arm, other, tier=1).tier == 1
    # a declared arm vs the in-support arm of the other SI: NOT the two-SI
    # contrast (the render/positions differ) — refused
    with pytest.raises(dc.PairingError, match="same serve variant"):
        dc.check_pairing(_leg(f"{PSR_BASE}@thinking", LENS_IDENT),
                         _leg("si_constitution__psr_kl_all", LENS_IDENT),
                         tier=1)
    with pytest.raises(dc.PairingError):
        dc.check_pairing(_leg(f"{PSR_BASE}@all_turns", LENS_IDENT),
                         _leg("si_constitution__psr_kl_all@all_positions",
                              LENS_IDENT), tier=1)
    # and a variant cell against its own base is two unrelated defenses on
    # one identity, as before
    with pytest.raises(dc.PairingError):
        dc.check_pairing(_leg(f"{PSR_BASE}@thinking", LENS_IDENT),
                         _leg(PSR_BASE, LENS_IDENT), tier=1)
