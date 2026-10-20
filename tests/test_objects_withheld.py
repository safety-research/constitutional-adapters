"""objects/WITHHELD.json: the paper objects this repository does not ship, refused BY NAME, with the recorded reason, by every
reader of objects/ — never a bare FileNotFoundError or a silent miss.

Two qwen35_4b_it fits are withheld (they were trained on prompts that include third-party model output): the
`safety_lora` and `safety_relu` objects. Each has no directory and no MANIFEST.json row; the manifest's census counts them
(`in_paper` = 154 = 152 listed + 2 `withheld`) and names the list (`withheld_list`). The records that name the two ids
(configs, pins, results) are left as they are; what this file holds is that every public path that would LOAD one on a fresh
clone refuses it by name: `sisteer.objects` (resolve, get, the CLI), the committed-object loader, `sisteer.methods.load_object`,
the demo import tool, the stager's `committed:` source, the cosine sidecar script, and the exporter (which skips them, and
refuses a stale list or a leftover directory)."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from sisteer import objects
from sisteer.objects import WithheldObjectError, load_index, read_withheld, resolve, withheld_message

REPO = Path(__file__).resolve().parents[1]
OBJECTS = REPO / "objects"
MANIFEST = OBJECTS / "MANIFEST.json"
WITHHELD = OBJECTS / "WITHHELD.json"

SAFETY_LORA = "0dbf406f94a14ab9f9b960155bb81ddb685d2b1c916f9a04dda2f49eda206341"
SAFETY_RELU = "75625f59809498899db9efcd2b19e9ce8c2f95593e7c7c1a109a03e29fe5b5c8"
REASON = "trained on prompts that include third-party model (xAI Grok) output; withheld from the public release"
EXPECTED = {SAFETY_LORA: ("qwen35_4b_it", "safety_lora", "lora"), SAFETY_RELU: ("qwen35_4b_it", "safety_relu", "relu_steer")}


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _cli(*args: str) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO / "src") + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run([sys.executable, "-m", "sisteer.objects", *args], cwd=REPO, env=env, capture_output=True, text=True,
                          timeout=120)


# ------------------------------------------------------------------------------------------------ the committed list
def test_the_list_names_the_two_safety_fits_with_the_reason_and_the_tree_ships_neither():
    doc = json.loads(WITHHELD.read_text(encoding="utf-8"))
    rows = {r["object_id"]: r for r in doc["objects"]}
    assert set(rows) == set(EXPECTED)
    for oid, (model_id, label, kind) in EXPECTED.items():
        r = rows[oid]
        assert (r["model_id"], r["paper_label"], r["object_kind"], r["object_role"], r["derivation"]) == (model_id, label, kind, "safety", "none")
        assert r["artifact_sha256"] == oid and isinstance(r["bytes"], int) and r["bytes"] > 0
        assert r["reason"] == REASON
        assert not (OBJECTS / model_id / oid).exists(), f"objects/{model_id}/{oid} is withheld but present"
    man = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert not set(rows) & {r["object_id"] for r in man["objects"]}
    assert man["withheld_list"] == WITHHELD.name
    assert man["census"]["withheld"] == 2 and man["census"]["in_paper"] == len(man["objects"]) + 2 == 154
    demo = json.loads((OBJECTS / "DEMO.json").read_text(encoding="utf-8"))
    assert not set(rows) & {o["object_id"] for o in demo["objects"]}
    assert not any(f"/{oid}" in o["dir"] for o in demo["objects"] for oid in rows)
    # the package's reader and the stdlib exporter's reader read the list the same way
    exporter = _load("export_objects_withheld_test", "reproduce/tools/export_objects.py")
    assert read_withheld(WITHHELD) == exporter.read_withheld(WITHHELD) == rows


# ------------------------------------------------------------------------------------------------ sisteer.objects
def test_resolve_and_get_refuse_a_withheld_object_by_name_with_the_reason():
    for oid, (model_id, label, kind) in EXPECTED.items():
        for kw in ({}, {"kind": kind}, {"kind": kind, "derivation": "none"}):
            with pytest.raises(WithheldObjectError) as ei:
                resolve(model_id, label, repo_root=REPO, **kw)
            msg = str(ei.value)
            assert oid in msg and label in msg and REASON in msg and "objects/WITHHELD.json" in msg
        with pytest.raises(WithheldObjectError, match="withheld from this repository") as ei:
            load_index(REPO).get(oid)
        assert oid in str(ei.value) and REASON in str(ei.value)
    assert issubclass(WithheldObjectError, ValueError)                      # the module's refusal type: the CLI catches it
    # an unknown label on the model names the withheld labels beside the shipped ones
    with pytest.raises(ValueError) as ei:
        resolve("qwen35_4b_it", "no_such_label", repo_root=REPO)
    assert "difference_lora" in str(ei.value) and "withheld from this repository (objects/WITHHELD.json): safety_lora, safety_relu" in str(ei.value)
    # find lists shipped objects only; withheld lists the withheld rows
    assert not any(e.object_id in EXPECTED for e in objects.find(repo_root=REPO))
    assert [r["object_id"] for r in objects.withheld(repo_root=REPO, model_id="qwen35_4b_it")] == [SAFETY_LORA, SAFETY_RELU]
    assert [r["object_id"] for r in objects.withheld(repo_root=REPO, label="safety_relu")] == [SAFETY_RELU]
    assert objects.withheld(repo_root=REPO, model_id="qwen35_4b_base") == []
    assert len(load_index(REPO).withheld) == 2


def test_the_cli_refuses_and_names_withheld_objects():
    r = _cli("resolve", "qwen35_4b_it", "safety_lora")
    assert r.returncode == 2 and r.stdout == "" and SAFETY_LORA in r.stderr and REASON in r.stderr
    r = _cli("find", "--model", "qwen35_4b_it", "--label", "safety_relu")
    assert r.returncode == 1 and len(r.stdout.splitlines()) == 1                # no shipped row; the header only
    assert r.stderr.strip() == f"withheld: {withheld_message(read_withheld(WITHHELD)[SAFETY_RELU])}"
    r = _cli("find", "--model", "qwen35_4b_it")
    assert r.returncode == 0 and len(r.stdout.splitlines()) == 1 + 12
    assert [l.split()[:3] for l in r.stderr.splitlines()] == [["withheld:", "object", SAFETY_LORA], ["withheld:", "object", SAFETY_RELU]]
    # a dose or seed filter says nothing of the withheld rows, whose dose and seed are not recorded here
    r = _cli("find", "--model", "qwen35_4b_it", "--dose", "0.5")
    assert r.returncode == 0 and r.stderr == ""


def test_a_manifest_naming_an_absent_or_conflicting_list_is_refused(tmp_path):
    man = json.loads(MANIFEST.read_text(encoding="utf-8"))
    root = tmp_path / "absent"
    (root / "objects").mkdir(parents=True)
    (root / "objects" / "MANIFEST.json").write_text(json.dumps(man))
    with pytest.raises(FileNotFoundError, match="names its withheld list WITHHELD.json, which is not in this checkout"):
        objects.ObjectIndex(root)
    # an id both listed and withheld
    root = tmp_path / "both"
    (root / "objects").mkdir(parents=True)
    (root / "objects" / "MANIFEST.json").write_text(json.dumps(man))
    w = json.loads(WITHHELD.read_text())
    w["objects"].append(dict(w["objects"][0], object_id=man["objects"][0]["object_id"]))
    (root / "objects" / "WITHHELD.json").write_text(json.dumps(w))
    with pytest.raises(ValueError, match="shipped or withheld, never both"):
        objects.ObjectIndex(root)
    # a row without its reason, a list name that is not a file beside the manifest
    w = json.loads(WITHHELD.read_text())
    del w["objects"][0]["reason"]
    (root / "objects" / "WITHHELD.json").write_text(json.dumps(w))
    with pytest.raises(ValueError, match="lacks reason"):
        read_withheld(root / "objects" / "WITHHELD.json")
    (root / "objects" / "MANIFEST.json").write_text(json.dumps(dict(man, withheld_list="../elsewhere.json")))
    with pytest.raises(ValueError, match="not a file name beside the manifest"):
        objects.ObjectIndex(root)


# ------------------------------------------------------------------------------------------------ the loaders
def test_the_committed_object_loader_refuses_a_withheld_object_by_name():
    from sisteer.extraction import committed_objects as co
    for oid, (model_id, label, _kind) in EXPECTED.items():
        d = OBJECTS / model_id / oid
        with pytest.raises(co.WithheldCommittedObjectError) as ei:
            co.committed_object_bytes_path(d)
        assert oid in str(ei.value) and REASON in str(ei.value)
        for target in (d, d / "object.pt", d / "fit_provenance.json"):         # the directory or a file in it, none present
            assert co.withheld_row(target)["paper_label"] == label
        with pytest.raises(co.CommittedObjectError, match="withheld from this repository"):
            co.load_committed_object(d)
        with pytest.raises(WithheldObjectError, match=REASON.replace("(", r"\(").replace(")", r"\)")):
            co.committed_object_bytes_path(d, object_id=oid)
    # a shipped object is unaffected, and an absent id that is not withheld keeps the plain refusal
    assert co.withheld_row(OBJECTS / "qwen35_4b_it" / "5e9b0009eb24eaf66842b4a2a9f647fced95699c7aae748eaab5333b05a4c6ce") is None
    with pytest.raises(co.CommittedObjectError, match="not a directory"):
        co.committed_object_bytes_path(OBJECTS / "qwen35_4b_it" / ("0" * 64))


def test_load_object_names_a_withheld_object_instead_of_a_missing_file():
    from sisteer import methods
    from sisteer.extraction.committed_objects import WithheldCommittedObjectError
    for oid, (model_id, _label, _kind) in EXPECTED.items():
        path = OBJECTS / model_id / oid / "object.pt"
        with pytest.raises(WithheldCommittedObjectError) as ei:
            methods.load_object(path, model=object(), tok=None)
        assert oid in str(ei.value) and REASON in str(ei.value)
        with pytest.raises(WithheldCommittedObjectError):
            methods.load_object_cpu(str(path))
    with pytest.raises(Exception) as ei:                                     # not withheld: whatever the plain absent-file error is
        methods.load_object_cpu(OBJECTS / "qwen35_4b_it" / ("0" * 64) / "object.pt")
    assert not isinstance(ei.value, WithheldObjectError)


def test_the_demo_import_tool_refuses_a_withheld_object_by_name(monkeypatch):
    io = _load("import_objects_withheld_test", "reproduce/tools/import_objects.py")
    plan = io.Plan("qwen35_4b_demo", repo=REPO)
    assert set(plan.withheld) == set(EXPECTED)
    with pytest.raises(SystemExit) as ei:
        plan._manifest_row(SAFETY_LORA, "objects/DEMO.json:safety_lora")
    assert "REFUSED" in str(ei.value) and SAFETY_LORA in str(ei.value) and REASON in str(ei.value)
    # a view cell whose label only a withheld object answers
    monkeypatch.setattr(plan.M, "object_of", lambda m, cell: {"paper_label": "safety_relu"})
    monkeypatch.setattr(plan.M, "committed_objects_of", lambda m, cell=None: "qwen35_4b")
    monkeypatch.setattr(plan.M, "committed_model_id", lambda spelling, table=None: "qwen35_4b_it")
    with pytest.raises(SystemExit) as ei:
        plan._entry("safety_cell")
    assert SAFETY_RELU in str(ei.value) and REASON in str(ei.value)


def test_the_stager_refuses_a_committed_source_naming_a_withheld_object(tmp_path):
    sys.path.insert(0, str(REPO / "scripts" / "sdfbase" / "full22"))
    try:
        so = _load("stage_objects_withheld_test", "scripts/sdfbase/full22/stage_objects.py")
    finally:
        sys.path.remove(str(REPO / "scripts" / "sdfbase" / "full22"))
    for dry in (False, True):
        dest = tmp_path / f"dest_{dry}"
        with pytest.raises(SystemExit) as ei:
            so.pull_source(f"committed:objects/qwen35_4b_it/{SAFETY_RELU}", dest, {}, dry)
        assert "REFUSED" in str(ei.value) and SAFETY_RELU in str(ei.value) and REASON in str(ei.value)
        assert not dest.exists()
    assert so.committed_withheld("objects/qwen35_4b_it/5e9b0009eb24eaf66842b4a2a9f647fced95699c7aae748eaab5333b05a4c6ce") is None


def test_the_cosine_script_refuses_the_withheld_safety_relu_unless_its_variable_names_the_file(tmp_path, monkeypatch):
    cs = _load("relu_arditi_cosines_withheld_test", "paper/notebooks/relu_arditi_cosines.py")
    entry = next(o for o in cs.OBJECTS if o[3] == SAFETY_RELU)
    assert entry[0] == "safety_relu" and entry[4] == cs.SAFETY_RELU_ENV == "SISTEER_COSINE_SAFETY_RELU"
    monkeypatch.delenv(cs.SAFETY_RELU_ENV, raising=False)
    with pytest.raises(SystemExit) as ei:
        cs.committed_object(SAFETY_RELU, cs.SAFETY_RELU_ENV)
    assert SAFETY_RELU in str(ei.value) and REASON in str(ei.value) and "$SISTEER_COSINE_SAFETY_RELU" in str(ei.value)
    # with the variable set the file is read from it and still held to the id it is named by
    other = tmp_path / "object.pt"
    other.write_bytes(b"not the object")
    monkeypatch.setenv(cs.SAFETY_RELU_ENV, str(other))
    with pytest.raises(SystemExit, match="!= the object_id"):
        cs.committed_object(SAFETY_RELU, cs.SAFETY_RELU_ENV)
    # every other ReLU steer it reads is committed (bar the instruct-direct constitution reference, which has its own variable)
    shipped = {r["object_id"] for r in json.loads(MANIFEST.read_text())["objects"]}
    assert {o[3] for o in cs.OBJECTS if o[4] is None} <= shipped and cs.REFERENCE_ID in shipped


# ------------------------------------------------------------------------------------------------ the exporter
def test_the_exporter_skips_withheld_rows_counts_them_and_refuses_a_stale_list(tmp_path):
    from test_objects_committed import EXPORTER, _synthetic_bank, validate_object_dir
    bank, keep = _synthetic_bank(tmp_path / "bank_a", blob=os.urandom(3000), model_id="modelx",
                                 complete={"cell_id": "synthetic__cell", "object_file": "object.pt"})
    bank_b, held_row = _synthetic_bank(tmp_path / "bank_b", blob=os.urandom(2000), model_id="modelx",
                                       complete={"cell_id": "synthetic__cell", "object_file": "object.pt"})
    held = {held_row["object_id"]: {"object_id": held_row["object_id"], "model_id": "modelx", "paper_label": "synthetic",
                                    "object_kind": "lora", "reason": "a synthetic reason"}}
    out = tmp_path / "objects"
    man = EXPORTER.export([keep, held_row], EXPORTER.Banks({"a": bank, "b": bank_b}), out, 40,
                          source_note={"census_file": "synthetic.json"}, withheld=held, withheld_list="WITHHELD.json")
    assert [r["object_id"] for r in man["objects"]] == [keep["object_id"]]
    assert man["census"]["in_paper"] == 2 and man["census"]["withheld"] == 1 and man["census"]["artifact"] == 1
    assert man["withheld_list"] == "WITHHELD.json"
    assert not (out / "modelx" / held_row["object_id"]).exists()
    validate_object_dir(out, man["objects"][0])
    # without a list the manifest names none and counts zero
    man0 = EXPORTER.export([keep], EXPORTER.Banks({"a": bank}), tmp_path / "plain", 40, source_note={"census_file": "s.json"})
    assert man0["census"]["withheld"] == 0 and "withheld_list" not in man0
    # a withheld id the selection lacks is stale; a disagreeing label is refused; a leftover directory is refused
    with pytest.raises(SystemExit, match="not in the selection"):
        EXPORTER.export([keep], EXPORTER.Banks({"a": bank}), tmp_path / "x1", 40, source_note={}, withheld=held, withheld_list="W.json")
    bad = {k: dict(v, paper_label="other") for k, v in held.items()}
    with pytest.raises(SystemExit, match="disagree"):
        EXPORTER.export([keep, held_row], EXPORTER.Banks({"a": bank, "b": bank_b}), tmp_path / "x2", 40, source_note={},
                        withheld=bad, withheld_list="W.json")
    (tmp_path / "x3" / "modelx" / held_row["object_id"]).mkdir(parents=True)
    with pytest.raises(SystemExit, match="has a directory under"):
        EXPORTER.export([keep, held_row], EXPORTER.Banks({"a": bank, "b": bank_b}), tmp_path / "x3", 40, source_note={},
                        withheld=held, withheld_list="W.json")
