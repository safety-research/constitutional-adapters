"""`lens_steer.export_worker_registry` — the export hashes what it WROTE.

R16-A (Evals round-15 E45, H1-adjacent). E40 H1 bound the registry export
to a digest verified OUTSIDE the standup, but `_export_object` wrote
`artifact_bytes` and then returned that same SOURCE buffer, so the
comparison hashed the buffer rather than the copy in the worker registry —
self-consistent by construction, and therefore not the binding to "the
bytes the worker loads" that the docstring claimed. The worker's own
startup re-validation (truncated content hash + intervention hash) meant
nothing was ever served silently, but the OUTSIDE digest is checked only
here, so nothing else could catch it.

The pin below is the one case the old code could not see: a writer that
PERSISTS altered bytes while the function returns the source buffer. It
sits in its own module because the E40 H1 binding test lives in
`tests/test_tinycap_launcher.py`, which a concurrent lane owns.

Worker-side validation is deliberately untouched and untested here.
"""

import hashlib
import io
import json
from pathlib import Path

import pytest

CELL = "si_safety__cast"


def _spec(cm):
    return cm.PetriSpec(
        audit="tinycap-r16a-export-test", behaviors=list(cm.PETRI_BEHAVIORS),
        cells=["none", CELL], scenario_bundle_hash="ab" * 8,
        auditor_model="anthropic/claude-sonnet-5", judge_model="anthropic/claude-opus-5",
        scenarios_model="anthropic/claude-sonnet-5",
        prompt_provenance="package-pinned@petri-bloom==0.2.6",
        target_model="org/synthetic-target",
        target_revision="0123456789abcdef0123456789abcdef01234567",
        serving_backend="proxy_vllm", serving_mode="interleaved",
        serving_packages={"vllm": "0.25.1+cu129", "vllm-lens": "1.2.1"},
        serving_launch_flags=["--enforce-eager"], chat_template_hash="12" * 8,
        tool_parser="llama3_json", serving_impl_commit="fedcba9876" * 4,
        prompted_prefix_hashes={"safety": cm.sha256_hex16(b"synthetic safety prefix"),
                                "constitution": cm.sha256_hex16(b"synthetic constitution")},
        bloom_options=cm.BloomOptions(max_turns=15, enable_prefill=False, enable_rollback=False,
                                      compaction=False, realism_filter=False, epochs=1, seed=0))


def _world(tmp_path):
    """One synthetic lens cell on disk, enumerated the way `bring_up` does."""
    torch = pytest.importorskip("torch")

    from sisteer.capeval import manifest as cm
    from sisteer.capeval.petri import defense_cells as dc
    from sisteer.capeval.serving import lens_adapters as la

    g = torch.Generator().manual_seed(7)
    buf = io.BytesIO()
    torch.save({"layer": 3, "vector": torch.randn(8, generator=g)}, buf)
    artifact = tmp_path / f"{CELL}.pt"
    artifact.write_bytes(buf.getvalue())

    source = dc.MappingCellSource({
        "none": dc.RegistryEntry(cell_id="none", kind="none"),
        CELL: dc.RegistryEntry(
            cell_id=CELL, kind=la.SYNTHETIC_KIND, si_id="safety",
            object_path=str(artifact), serving_path="lens",
            resolved_spec={"kind": la.SYNTHETIC_KIND, "si_id": "safety", "layer": 3})})
    spec = _spec(cm)
    cells = dc.enumerate_cells(
        spec, source,
        behavior_files={k: f"# {k}\n".encode() for k in cm.petri_expected_behavior_keys(spec)},
        prompt_files={"auditor": b"# a\n", "judge": b"# j\n"})
    assert not cells.absent
    return cells, source, artifact


def _tamper_artifact_writes(monkeypatch, suffix=b"\x00tampered"):
    """A writer that PERSISTS altered bytes for every `artifact.bin` while the
    caller keeps the buffer it handed over — the filesystem/short-write/swap
    case, and exactly the state `_export_object` used to return."""
    real = Path.write_bytes
    # the export writes each blob through a crash-safe temp file (`artifact.bin.tmp`,
    # then os.replace — lens_steer.write_immutable), so the injector must tamper the
    # temp file too or the fault never reaches disk
    tampered = ("artifact.bin", "artifact.bin.tmp")

    def writer(self, data):
        return real(self, data + suffix if self.name in tampered else data)

    monkeypatch.setattr(Path, "write_bytes", writer)


def test_the_export_hashes_the_bytes_it_wrote_not_the_buffer_it_read(tmp_path, monkeypatch):
    """A writer that persists altered bytes while returning the source buffer:
    the export REFUSES against the preflight digest, naming both digests, and
    the digest it reports is the one the file on disk actually has (the source
    buffer's digest is what the preflight named and is the one it does NOT
    match). No index is written, so nothing downstream can read a registry the
    export refused."""
    from sisteer.capeval.serving import lens_steer as ls

    cells, source, artifact = _world(tmp_path)
    preflight_digest = hashlib.sha256(artifact.read_bytes()).hexdigest()

    # the control first: an untampered export against that digest is accepted,
    # and what it records is the digest of the file it wrote
    out_ok = tmp_path / "reg_ok"
    entries = ls.export_worker_registry(cells, source, out_ok,
                                        expected_artifact_sha256={CELL: preflight_digest})
    written = (out_ok / f"{CELL}/artifact.bin").read_bytes()
    assert hashlib.sha256(written).hexdigest() == preflight_digest
    entry = next(e for e in entries if e.cell_id == CELL)
    from sisteer.capeval import manifest as cm
    assert entry.artifact_sha16 == cm.sha256_hex16(written)
    assert (out_ok / ls.REGISTRY_INDEX).is_file()

    # now the writer that persists something else
    _tamper_artifact_writes(monkeypatch)
    out_bad = tmp_path / "reg_tampered"
    with pytest.raises(ls.LensSteerError) as e:
        ls.export_worker_registry(cells, source, out_bad,
                                  expected_artifact_sha256={CELL: preflight_digest})
    msg = str(e.value)
    on_disk = hashlib.sha256((out_bad / f"{CELL}/artifact.bin").read_bytes()).hexdigest()
    assert on_disk != preflight_digest
    assert on_disk in msg and preflight_digest in msg, msg
    assert "replaced between the preflight" in msg
    # the pre-R16-A code hashed the SOURCE buffer, so it reported no mismatch
    # at all; the digest now named as "got" is the copy's, not the buffer's
    assert hashlib.sha256(artifact.read_bytes()).hexdigest() == preflight_digest
    assert not (out_bad / ls.REGISTRY_INDEX).exists()


def test_a_tampered_copy_refuses_even_with_no_outside_digest(tmp_path, monkeypatch):
    """Same writer, no `expected_artifact_sha256`: the recomputed intervention
    hash now runs over the bytes on disk too, so the copy that is not the
    enumerated object refuses here rather than at worker startup. (Without an
    outside digest that recomputation is a self-check — it cannot see a swap
    of the SOURCE — but it does see a copy that does not match its source.)"""
    from sisteer.capeval.serving import lens_steer as ls

    cells, source, _ = _world(tmp_path)
    _tamper_artifact_writes(monkeypatch)
    with pytest.raises(ls.LensSteerError, match="recomputed intervention hash"):
        ls.export_worker_registry(cells, source, tmp_path / "reg_unbound")
    assert not (tmp_path / "reg_unbound" / ls.REGISTRY_INDEX).exists()


def test_an_untampered_export_still_loads_and_records_the_written_bytes(tmp_path):
    """The reread is not a behaviour change for an honest writer: the registry
    the export writes loads, and its recorded per-entry digests are those of
    the files in it."""
    from sisteer.capeval import manifest as cm
    from sisteer.capeval.serving import lens_steer as ls

    cells, source, _ = _world(tmp_path)
    out = tmp_path / "reg"
    entries = ls.export_worker_registry(cells, source, out)
    index = json.loads((out / ls.REGISTRY_INDEX).read_text())
    for raw in index["entries"]:
        if raw["artifact"] is None:
            continue
        blob = (out / raw["artifact"]).read_bytes()
        assert raw["artifact_sha16"] == cm.sha256_hex16(blob)
        assert raw["artifact_intervention_hash"] == cm.petri_intervention_hash(
            raw["resolved_spec"], artifact_bytes=blob, sidecar_bytes=b"")
    reg = ls.WorkerObjectRegistry.load(out)
    assert set(reg.cell_ids()) == {CELL, ls.NOOP_LENS_CELL_ID}
    assert {e.cell_id for e in entries} == {CELL, ls.NOOP_LENS_CELL_ID}
