"""objects/ holds every fitted object the paper evaluates that the repository ships, as ORIGINAL bytes keyed by registry
object_id: 149 artifact-bearing objects (sha256(object.pt) == object_id == artifact_sha256) + the 3 VIRTUAL rows of
`v_paper_objects` (two `system_instruction` objects `si:<sha>` and `none`) = 152 listed rows, which with the 2 objects
objects/WITHHELD.json withholds (tests/test_objects_withheld.py) are the 154 in_paper rows. Every
committed file is under 50 MB; the kind and model census matches the registry's; the reader-named sidecars ride beside
each artifact; every file a cell's JSON `COMPLETE` publication marker attests is present and hashes to the sha the marker
records (the trainer's own resume validator applies exactly that check); a CHUNKED artifact is stream-reassembled here
and the COMPUTED sha256 and size compared to object_id / the manifest, never the recorded `original_sha256` alone; no
manifest field names a host or a home directory.

The live-registry cross-check (`v_paper_objects` is a ~2 min query on the 36 GB grading host registry) runs only under
SISTEER_OBJECTS_LIVE_REGISTRY_CHECK=1 when ~/registry/registry.sqlite (or $SISTEER_REGISTRY_DB) exists."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sqlite3
from collections import Counter
from pathlib import Path

import pytest
from front_door_testlib import assert_no_tokens

from sisteer.registry import hooks
from sisteer.registry.identity import object_id as registry_object_id

REPO = Path(__file__).resolve().parents[1]
OBJECTS = REPO / "objects"
MANIFEST = OBJECTS / "MANIFEST.json"
HARD_LIMIT = 50 * 1000 * 1000

#: The paper's object census (registry v_paper_objects, in_paper = 1, 2026-09-29) less the two objects objects/WITHHELD.json
#: withholds (the qwen35_4b_it safety_lora / safety_relu fits, 2026-10-19): the rows MANIFEST.json lists.
EXPECTED_BY_KIND = {"lora": 63, "lora_subtraction": 42, "composed": 19, "relu_steer": 20, "vector_all": 4,
                    "static_r_arditi": 1, "system_instruction": 2, "none": 1}
EXPECTED_BY_MODEL = {"qwen35_4b_base": 68, "gemma4e4b_base": 21, "llama8b_base": 21, "nemotron3nano4b": 17,
                     "qwen35_4b_it": 12, "gemma4e4b_it": 5, "llama8b_it": 5, "None": 3}
#: The in_paper rows objects/WITHHELD.json holds: no directory, no manifest row, counted in the census.
EXPECTED_WITHHELD = 2
#: How many of the 149 carry the trainer's JSON `COMPLETE` publication marker (files: object.pt, train_budget_meta.json,
#: metrics.jsonl); the two legacy plain-text `COMPLETE`s, which attest nothing beyond their presence, were the withheld safety
#: fits', so every shipped `COMPLETE` is the JSON form. Counted over
#: the PRESENT paper subset -- the rows of the paper's models in EXPECTED_BY_MODEL -- so a test-model fit exported later (with its
#: own model_id) extends the manifest without moving this number. Adding a paper object of one of these models is a
#: deliberate census update here, as for the counts above.
EXPECTED_ATTESTED_DIRS = 38
PAPER_MODEL_IDS = frozenset(EXPECTED_BY_MODEL)


def _exporter():
    spec = importlib.util.spec_from_file_location("export_objects_for_census", REPO / "reproduce/tools/export_objects.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


EXPORTER = _exporter()
VIRTUAL_KINDS = set(EXPORTER.VIRTUAL_KINDS)
SIDECAR_NAMES = EXPORTER.SIDECAR_NAMES


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


@pytest.fixture(scope="module")
def manifest():
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


# ------------------------------------------------------------------------------------------------ the per-object check
def stream_reassembled_identity(d: Path, cm: dict) -> tuple[str, int]:
    """sha256 and byte count of the ORIGINAL bytes a chunked directory holds, computed by streaming the ordered chunks
    (decompressed when `compression == zstd`) — independent of the recorded `original_sha256`, which is what a wrong or
    reordered chunk with a matching per-chunk hash would leave untouched. Every chunk's own recorded sha/bytes are checked
    on the way."""
    if cm["compression"] == "zstd":
        try:
            import zstandard
        except ImportError:  # pragma: no cover - the harness venv carries it
            pytest.fail(f"{d}: chunks are zstd-compressed and `zstandard` is not importable — cannot verify the original")
        dec = zstandard.ZstdDecompressor()
    else:
        assert cm["compression"] == "none", f"{d}: unknown chunk compression {cm['compression']!r}"
        dec = None
    h = hashlib.sha256()
    n = 0
    for c in cm["chunks"]:
        p = d / c["name"]
        assert p.is_file(), f"{d}: chunk {c['name']} missing"
        data = p.read_bytes()
        assert len(data) == c["bytes"] and hashlib.sha256(data).hexdigest() == c["sha256"], f"{d}: chunk {c['name']} bytes/sha"
        raw = dec.decompress(data, max_output_size=c.get("raw_bytes", cm["chunk_bytes"])) if dec else data
        h.update(raw)
        n += len(raw)
    return h.hexdigest(), n


def attested_files_of(d: Path) -> dict[str, str] | None:
    """The JSON `COMPLETE` marker's `files` map, or None when the directory carries no marker / the legacy text form."""
    p = d / "COMPLETE"
    if not p.is_file():
        return None
    text = p.read_text(encoding="utf-8", errors="replace")
    if not text.lstrip().startswith("{"):
        return None
    files = json.loads(text).get("files")
    return dict(files) if files is not None else None


def validate_object_dir(objects_root: Path, r: dict, *, sidecar_names=SIDECAR_NAMES) -> Path:
    """One committed artifact row against its directory: identity (whole: sha256(object.pt); chunked: the STREAMED
    original's sha and size) == object_id / bytes; every listed sidecar present and reader-named; every file the JSON
    `COMPLETE` attests present and hashing to the recorded sha (the artifact's recorded sha == object_id, whole or
    chunked); nothing in the directory the manifest row does not account for. Returns the resolved directory."""
    assert r["form"] in ("whole", "chunked"), r
    assert r["object_id"] == r["artifact_sha256"], r["object_id"]
    d = objects_root / r["dir"]
    assert r["dir"] == f"{r['model_id']}/{r['object_id']}" and d.is_dir(), r["dir"]
    chunk_names: set[str] = set()
    if r["form"] == "whole":
        p = d / "object.pt"
        assert p.is_file() and p.stat().st_size == r["bytes"], f"{r['dir']}: object.pt size"
        assert _sha(p) == r["object_id"], f"{r['dir']}: sha(object.pt) != object_id"
        assert not (d / "CHUNKS.json").exists() and r["chunks"] == 0
    else:
        assert (d / "CHUNKS.json").is_file() and not (d / "object.pt").exists(), f"{r['dir']}: chunked form layout"
        cm = json.loads((d / "CHUNKS.json").read_text())
        assert cm["original_name"] == "object.pt" and len(cm["chunks"]) == r["chunks"] > 1, f"{r['dir']}: CHUNKS.json"
        assert cm["original_sha256"] == r["object_id"] and cm["original_bytes"] == r["bytes"], f"{r['dir']}: recorded original"
        got_sha, got_bytes = stream_reassembled_identity(d, cm)
        assert got_sha == r["object_id"], (f"{r['dir']}: the streamed chunks hash to {got_sha[:12]} != object_id "
                                           f"{r['object_id'][:12]} — the recorded original_sha256 is not what the chunks hold")
        assert got_bytes == r["bytes"], f"{r['dir']}: streamed {got_bytes} bytes != manifest {r['bytes']}"
        chunk_names = {c["name"] for c in cm["chunks"]}
    for name in r["sidecars"]:
        assert name in sidecar_names and (d / name).is_file(), f"{r['dir']}: sidecar {name}"
    attested = attested_files_of(d)
    if r["attested"] is None:
        assert attested is None, f"{r['dir']}: COMPLETE attests {sorted(attested)} but the manifest row records none"
    else:
        assert attested is not None and sorted(attested) == sorted(r["attested"]), f"{r['dir']}: attested names"
        for name, sha in attested.items():
            if name == "object.pt":
                assert sha == r["object_id"], f"{r['dir']}: COMPLETE records object.pt sha {sha[:12]} != object_id"
                continue
            p = d / name
            assert p.is_file(), f"{r['dir']}: COMPLETE attests {name} but the file is not committed"
            assert _sha(p) == sha, f"{r['dir']}: {name} does not hash to the sha its COMPLETE records"
    extra = {p.name for p in d.iterdir()} - set(r["sidecars"]) - set(r["attested"] or ()) - {"object.pt", "CHUNKS.json"} - chunk_names
    assert not extra, f"{r['dir']}: unlisted files {extra}"
    return d.resolve()


def assert_attestation_committed(objects_root: Path, r: dict) -> None:
    """Every name the row's JSON `COMPLETE` attests is committed: a PHYSICAL file for each -- except `object.pt` of a CHUNKED row,
    which is LOGICAL: the chunks hold it (no `object.pt` file by design, CHUNKS.json beside them) and its identity is the STREAMED
    check `validate_object_dir` runs over every row (computed sha and size == object_id / manifest), not a file's presence."""
    d = objects_root / r["dir"]
    for name in r["attested"]:
        if name == "object.pt" and r["form"] == "chunked":
            assert (d / "CHUNKS.json").is_file() and not (d / "object.pt").exists(), f"{r['dir']}: chunked layout for the attested object.pt"
            assert json.loads((d / "CHUNKS.json").read_text())["original_sha256"] == r["object_id"], f"{r['dir']}: CHUNKS.json original"
            continue
        assert (d / name).is_file(), f"{r['dir']}: attested {name} not committed"


def paper_rows(rows) -> list[dict]:
    """The manifest's artifact rows of the PRESENT paper subset (the paper's models, EXPECTED_BY_MODEL); a row of another model -- a
    test-model fit -- is outside the census the constants above pin."""
    return [r for r in rows if r["form"] != "virtual" and str(r["model_id"]) in PAPER_MODEL_IDS]


# ------------------------------------------------------------------------------------------------ the committed tree
def test_exporter_sidecar_names_are_the_readers_list_plus_the_store_identity():
    """The stdlib-only exporter spells its sidecar list; it must stay what the registry's readers consume
    (hooks.SIDECAR_NAMES, `read_sidecars`) plus the bank's `_store_identity.json`, or a new sidecar is silently dropped."""
    assert set(SIDECAR_NAMES) == set(hooks.SIDECAR_NAMES) | {"_store_identity.json"}
    assert "fit_selection.json" in SIDECAR_NAMES


def test_census_149_artifacts_plus_3_virtual_plus_2_withheld(manifest):
    rows = manifest["objects"]
    assert len(rows) == 152
    art = [r for r in rows if r["form"] != "virtual"]
    virt = [r for r in rows if r["form"] == "virtual"]
    assert len(art) == 149 and len(virt) == 3
    assert manifest["census"] == {"in_paper": 154, "withheld": EXPECTED_WITHHELD, "artifact": 149, "virtual": 3,
                                  "by_kind": dict(sorted(EXPECTED_BY_KIND.items())),
                                  "by_model": dict(sorted(EXPECTED_BY_MODEL.items()))}
    assert Counter(r["object_kind"] for r in rows) == EXPECTED_BY_KIND
    assert Counter(str(r["model_id"]) for r in rows) == EXPECTED_BY_MODEL
    assert {r["object_kind"] for r in virt} == VIRTUAL_KINDS
    assert len({r["object_id"] for r in rows}) == 152, "duplicate object ids"
    # the withheld list the manifest names holds the rest of the in_paper selection, and none of the listed rows
    assert manifest["withheld_list"] == "WITHHELD.json"
    held = EXPORTER.read_withheld(OBJECTS / manifest["withheld_list"])
    assert len(held) == EXPECTED_WITHHELD and not set(held) & {r["object_id"] for r in rows}


def test_virtual_rows_have_no_directory_and_registry_shaped_ids(manifest):
    for r in manifest["objects"]:
        if r["form"] != "virtual":
            continue
        assert r["dir"] is None and r["artifact_sha256"] is None and r["bytes"] is None
        if r["object_kind"] == "none":
            assert r["object_id"] == registry_object_id("none")
        else:
            assert r["object_id"] == registry_object_id("system_instruction", si_text_sha256=r["si_text_sha256"])
    # the two SI ids are the committed system_instructions/ texts
    si_ids = {r["object_id"] for r in manifest["objects"] if r["object_kind"] == "system_instruction"}
    files = {f"si:{_sha(REPO / 'system_instructions' / n)}" for n in ("safety.md", "constitution.md")}
    assert si_ids == files


def test_every_artifact_sha_equals_its_object_id_and_passes_its_own_attestation(manifest):
    seen_dirs = {validate_object_dir(OBJECTS, r) for r in manifest["objects"] if r["form"] != "virtual"}
    # and nothing under objects/<model>/ that the manifest does not list (objects/_reassembled/ is the gitignored cache)
    on_disk = {p.resolve() for m in OBJECTS.iterdir() if m.is_dir() and not m.name.startswith("_") for p in m.iterdir() if p.is_dir()}
    assert on_disk == seen_dirs


def test_every_json_complete_marker_is_fully_committed(manifest):
    """38 exported directories carried a JSON COMPLETE naming metrics.jsonl and none carried the file — the production
    resume validator rejects exactly that. Every attested name is committed now. The count and the trainer's three-name
    shape are the PAPER subset's (a chunked row's `object.pt` is logical and a test-model row is outside the census;
    every attested row, in the subset or not, is still fully committed)."""
    paper = paper_rows(manifest["objects"])
    attested = [r for r in paper if r["attested"]]
    assert len(attested) == EXPECTED_ATTESTED_DIRS
    for r in attested:
        assert {"object.pt", "train_budget_meta.json", "metrics.jsonl"} <= set(r["attested"]), r["dir"]
    for r in manifest["objects"]:
        if r["form"] != "virtual" and r["attested"]:
            assert_attestation_committed(OBJECTS, r)
    # the legacy text markers are recorded as attesting nothing, and are the only COMPLETEs without a file list
    legacy = [r for r in paper if "COMPLETE" in r["sidecars"] and r["attested"] is None]
    assert len(legacy) == Counter(r["sidecars"].count("COMPLETE") for r in paper)[1] - EXPECTED_ATTESTED_DIRS
    for r in legacy:
        assert not (OBJECTS / r["dir"] / "COMPLETE").read_text(errors="replace").lstrip().startswith("{"), r["dir"]


def test_every_committed_file_is_under_the_size_limit_and_total_is_recorded(manifest):
    total = 0
    for p in OBJECTS.rglob("*"):
        if p.is_file() and "_reassembled" not in p.parts:
            assert p.stat().st_size < HARD_LIMIT, p
    for r in manifest["objects"]:
        total += r["bytes"] or 0
    assert manifest["bytes_total"] == total
    assert 300_000_000 < total < 400_000_000, total  # The paper set is 300-400 MB as-is


def test_manifest_records_sources_by_bank_name_only(manifest):
    text = MANIFEST.read_text(encoding="utf-8")
    assert_no_tokens(text, "MANIFEST.json")
    for tok in ("/home/", "/workspace"):
        assert tok not in text, f"MANIFEST.json carries {tok!r}"
    for r in manifest["objects"]:
        if r["form"] != "virtual":
            bank, sep, rel = r["source"].partition(":")
            assert sep and bank and rel and not rel.startswith("/"), r["source"]


def test_every_artifact_has_at_least_one_sidecar(manifest):
    missing = [r["dir"] for r in manifest["objects"] if r["form"] != "virtual" and not r["sidecars"]]
    assert not missing, missing


def test_reassembly_cache_is_gitignored_and_outside_the_census():
    """A chunked object's default reassembly destination is `objects/_reassembled/
    <object_id>/object.pt` (sisteer.extraction.committed_objects) — under the objects root, never inside objects/<model>/;
    .gitignore covers it and the census walk above skips `_`-prefixed roots, so a reassembled >40 MB object can neither be
    swept into a commit nor fail the manifest census."""
    lines = {ln.strip() for ln in (REPO / ".gitignore").read_text(encoding="utf-8").splitlines()}
    assert "objects/_reassembled/" in lines
    from sisteer.extraction import committed_objects as co
    fake_dir = OBJECTS / "modelx" / ("a" * 64)
    assert co.default_reassemble_dest(fake_dir) == OBJECTS / "_reassembled" / ("a" * 64) / "object.pt"


# ------------------------------------------------------------------------------------------------ the exporter, synthetic
def _synthetic_bank(root: Path, *, blob: bytes, model_id: str, complete: dict | str | None, extra: dict[str, bytes] = {}):
    """One bank cell: object.pt = blob, train_budget_meta.json, the given COMPLETE (a dict is written as the JSON
    publication marker with `files` filled from the cell unless it already carries one; a str is the legacy text form),
    plus `extra` files. Returns (bank dir, census row)."""
    cell = root / model_id / "cells" / "synthetic__cell"
    cell.mkdir(parents=True)
    (cell / "object.pt").write_bytes(blob)
    (cell / "train_budget_meta.json").write_text(json.dumps({"arm": "lora_r4_ce", "seed": 0}))
    for name, data in extra.items():
        (cell / name).write_bytes(data)
    if isinstance(complete, dict):
        marker = dict(complete)
        if "files" not in marker:
            marker["files"] = {n: _sha(cell / n) for n in ("object.pt", "train_budget_meta.json", *extra)}
        (cell / "COMPLETE").write_text(json.dumps(marker, indent=1))
    elif isinstance(complete, str):
        (cell / "COMPLETE").write_text(complete)
    oid = _sha(cell / "object.pt")
    row = {"object_id": oid, "paper_label": "synthetic", "object_kind": "lora", "object_role": "steer", "fit_model_id": model_id,
           "campaign": "synthetic", "artifact_sha256": oid, "artifact_locations_json": "[]", "derivation": "fit"}
    return root, row


def _export(tmp_path: Path, bank: Path, rows: list[dict], *, chunk_mb: int = 40) -> tuple[Path, dict]:
    out = tmp_path / "objects"
    man = EXPORTER.export(rows, EXPORTER.Banks({"bank": bank}), out, chunk_mb, source_note={"census_file": "synthetic.json"})
    return out, man


def test_export_commits_and_verifies_every_attested_file(tmp_path):
    """A cell whose JSON COMPLETE attests metrics.jsonl: the export copies it, verifies its sha, records the attested names,
    and the committed directory passes `validate_object_dir` (the same check the committed tree runs)."""
    metrics = b'{"step": 0, "lora_r4_ce/step0_loss": 1.0}\n' * 2000
    bank, row = _synthetic_bank(tmp_path / "bank", blob=os.urandom(200_000), model_id="modelx",
                                complete={"cell_id": "synthetic__cell", "object_file": "object.pt"}, extra={"metrics.jsonl": metrics})
    out, man = _export(tmp_path, bank, [row])
    (r,) = man["objects"]
    assert r["form"] == "whole" and r["sidecars"] == ["train_budget_meta.json", "COMPLETE"]
    assert r["attested"] == ["metrics.jsonl", "object.pt", "train_budget_meta.json"]
    assert (out / r["dir"] / "metrics.jsonl").read_bytes() == metrics
    validate_object_dir(out, r)
    # the legacy text marker attests nothing: recorded as null, nothing beyond the sidecars is copied
    bank2, row2 = _synthetic_bank(tmp_path / "bank2", blob=os.urandom(1000), model_id="modelx",
                                  complete="promoted from out/hpo/synthetic on 2026-01-01\n", extra={"metrics.jsonl": metrics})
    out2, man2 = _export(tmp_path / "two", bank2, [row2])
    (r2,) = man2["objects"]
    assert r2["attested"] is None and not (out2 / r2["dir"] / "metrics.jsonl").exists()
    validate_object_dir(out2, r2)


@pytest.mark.parametrize("defect", ["absent", "mismatch", "artifact"])
def test_export_refuses_a_cell_that_cannot_pass_its_own_attestation(tmp_path, defect):
    metrics = b'{"step": 0}\n' * 100
    complete = {"cell_id": "synthetic__cell", "object_file": "object.pt"}
    bank, row = _synthetic_bank(tmp_path / "bank", blob=os.urandom(50_000), model_id="modelx", complete=complete,
                                extra={"metrics.jsonl": metrics})
    cell = bank / "modelx" / "cells" / "synthetic__cell"
    marker = json.loads((cell / "COMPLETE").read_text())
    if defect == "absent":
        (cell / "metrics.jsonl").unlink()                        # attested, not present
        expect = "attests metrics.jsonl but the file is absent"
    elif defect == "mismatch":
        (cell / "metrics.jsonl").write_bytes(metrics + b"tampered\n")   # present, other bytes
        expect = "metrics.jsonl hashes"
    else:
        marker["files"]["object.pt"] = "0" * 64                   # the marker attests another artifact
        (cell / "COMPLETE").write_text(json.dumps(marker))
        expect = "COMPLETE records object.pt sha"
    with pytest.raises(SystemExit, match=expect):
        _export(tmp_path, bank, [row])


def test_chunked_export_is_verified_by_streaming_the_chunks_not_the_recorded_sha(tmp_path):
    """A chunked object whose chunks were swapped and whose per-chunk hashes were
    updated to match still records the right `original_sha256`; only streaming the chunks and comparing the COMPUTED sha to
    object_id catches it — and a CHUNKS.json made fully consistent with the wrong bytes is caught by the same comparison."""
    blob = os.urandom(2_500_000)
    bank, row = _synthetic_bank(tmp_path / "bank", blob=blob, model_id="modelx",
                                complete={"cell_id": "synthetic__cell", "object_file": "object.pt"})
    out, man = _export(tmp_path, bank, [row], chunk_mb=1)          # 1 MB chunks -> 3 parts, no whole object.pt
    (r,) = man["objects"]
    assert r["form"] == "chunked" and r["chunks"] == 3 and r["attested"] == ["object.pt", "train_budget_meta.json"]
    d = validate_object_dir(out, r)
    cm = json.loads((d / "CHUNKS.json").read_text())
    # swap two chunks' BYTES and re-record their own hashes: every per-chunk check passes, the original is wrong
    a, b = d / cm["chunks"][0]["name"], d / cm["chunks"][1]["name"]
    a_bytes, b_bytes = a.read_bytes(), b.read_bytes()
    a.write_bytes(b_bytes)
    b.write_bytes(a_bytes)
    cm["chunks"][0]["sha256"], cm["chunks"][1]["sha256"] = _sha(a), _sha(b)
    cm["chunks"][0]["bytes"], cm["chunks"][1]["bytes"] = a.stat().st_size, b.stat().st_size
    (d / "CHUNKS.json").write_text(json.dumps(cm))
    with pytest.raises(AssertionError, match="streamed chunks hash to"):
        validate_object_dir(out, r)
    # ... and a CHUNKS.json whose original_sha256 was ALSO rewritten to the swapped bytes' sha: still not the object_id
    wrong = hashlib.sha256(b_bytes + a_bytes + (d / cm["chunks"][2]["name"]).read_bytes()).hexdigest()
    cm["original_sha256"] = wrong
    (d / "CHUNKS.json").write_text(json.dumps(cm))
    with pytest.raises(AssertionError, match="recorded original"):
        validate_object_dir(out, r)


def test_all_attestations_check_admits_a_chunked_object_and_scopes_the_census_to_the_paper_subset(tmp_path):
    """A valid CHUNKED object attests `object.pt` in its JSON COMPLETE and
    holds no `object.pt` file (the chunks and CHUNKS.json do) -- `assert_attestation_committed` treats that name as LOGICAL, as
    `validate_object_dir` already does with its streamed identity check, and still demands every other attested file; and the
    row is of a model outside the paper census, so it does not count against EXPECTED_ATTESTED_DIRS."""
    metrics = b'{"step": 0}\n' * 100
    bank, row = _synthetic_bank(tmp_path / "bank", blob=os.urandom(2_500_000), model_id="testmodelx",
                                complete={"cell_id": "synthetic__cell", "object_file": "object.pt"}, extra={"metrics.jsonl": metrics})
    out, man = _export(tmp_path, bank, [row], chunk_mb=1)          # 1 MB chunks -> 3 parts, no whole object.pt
    (r,) = man["objects"]
    assert r["form"] == "chunked" and r["attested"] == ["metrics.jsonl", "object.pt", "train_budget_meta.json"]
    assert not (out / r["dir"] / "object.pt").exists() and (out / r["dir"] / "CHUNKS.json").is_file()
    validate_object_dir(out, r)
    assert_attestation_committed(out, r)                           # object.pt is logical for a chunked row
    assert paper_rows(man["objects"]) == [] and "testmodelx" not in PAPER_MODEL_IDS
    # every OTHER attested name is still a physical file: dropping the committed metrics.jsonl fails by name
    (out / r["dir"] / "metrics.jsonl").unlink()
    with pytest.raises(AssertionError, match="attested metrics.jsonl not committed"):
        assert_attestation_committed(out, r)
    # and a WHOLE row's object.pt is still a physical file
    bank2, row2 = _synthetic_bank(tmp_path / "bank2", blob=os.urandom(1000), model_id="modelx",
                                  complete={"cell_id": "synthetic__cell", "object_file": "object.pt"})
    out2, man2 = _export(tmp_path / "two", bank2, [row2])
    (r2,) = man2["objects"]
    assert r2["form"] == "whole" and paper_rows(man2["objects"]) == [] and "modelx" not in PAPER_MODEL_IDS
    assert_attestation_committed(out2, r2)
    (out2 / r2["dir"] / "object.pt").unlink()
    with pytest.raises(AssertionError, match="attested object.pt not committed"):
        assert_attestation_committed(out2, r2)


@pytest.mark.skipif(os.environ.get("SISTEER_OBJECTS_LIVE_REGISTRY_CHECK") != "1",
                    reason="live v_paper_objects cross-check is opt-in (a ~2 min query on the full grading host registry)")
def test_live_registry_in_paper_set_matches_the_manifest(manifest):
    db = Path(os.environ.get("SISTEER_REGISTRY_DB") or Path.home() / "registry/registry.sqlite")
    if not db.is_file():
        pytest.skip(f"no registry at {db}")
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=600)
    try:
        rows = conn.execute("select v.object_id, o.artifact_sha256, v.object_kind, v.fit_model_id from v_paper_objects v "
                            "join object o on o.object_id = v.object_id where v.in_paper = 1").fetchall()
    finally:
        conn.close()
    assert len(rows) == 154
    live = {r[0]: r for r in rows}
    held = EXPORTER.read_withheld(OBJECTS / manifest["withheld_list"])
    assert set(live) == {r["object_id"] for r in manifest["objects"]} | set(held)
    for oid, w in held.items():                                   # a withheld row is the live row it withholds
        assert (live[oid][1], live[oid][2], live[oid][3]) == (w["artifact_sha256"], w["object_kind"], w["model_id"])
    for r in manifest["objects"]:
        if r["form"] != "virtual":
            assert live[r["object_id"]][1] == r["artifact_sha256"]
        assert live[r["object_id"]][2] == r["object_kind"]
        assert (live[r["object_id"]][3] or None) == r["model_id"]
