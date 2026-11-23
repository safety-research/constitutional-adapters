"""The candidate-SI loader's opt-in SUBSET mode (sisteer.data.candidate_sis, entered by SISTEER_CANDIDATE_SIS_SUBSET_SHA256) and the
script that derives a subset file (scripts/sikl4/make_candidate_sis_subset.py).

The Modal C16 cells resolve C16 through the loader, but the full snapshot (B1 = the whole constitution) must not go on a volume every
member of the workspace can read, so those cells load a sha-pinned file holding only C16. What is pinned here:

* the subset is accepted only under the right pins: the file's own sha in the variable, ``meta.subset_of_sha256`` = the full
  snapshot's pinned sha, ids that carry a pinned text sha (C16 today), entry ids exactly ``meta.ids``, every text hashing to its pin;
  every other shape is refused by name;
* a subset file WITHOUT the variable is refused by the default path's 60-entry check, and with the variable unset (or empty) the
  default path is the one ``tests/test_candidate_si_registry.py`` pins, untouched;
* the C16 Instruction the subset yields equals the full snapshot's field by field, through the loader, through the env-guarded
  registration in a fresh interpreter, and in the TrainTest cell registry of each committed sikl4 rendered config;
* who sets the pair: with every cell of a sikl4 view staged, the registry refuses the WHOLE config without the loader (so every
  container over a config that lists a C16 cell needs the pair, not only the C16 cells' own), and under the subset every row, the
  four non-C16 cells' included, equals the one built under the full snapshot (so setting it on those containers changes nothing);
* the derivation script copies the entry verbatim, is deterministic, verifies its output through the subset mode before putting it in
  place, and refuses by name, an ``--out`` inside any git work tree included;
* the module's pins equal the real-snapshot pins the existing loader tests and the registry seed (configs/registry/si_texts.yaml) carry.

Every fixture is SYNTHETIC (test_candidate_si_registry's invented 60 entries), with the module's two pins pointed at it for the test.
The tests against the REAL snapshot run only when SISTEER_CANDIDATE_SIS names it. The check that a C16 cell's intervention hash
is the same under the subset as under the full snapshot needs the snapshot and the objects, so it is a lane check, not a test here."""
from __future__ import annotations

import dataclasses
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from sisteer.data import candidate_sis as C
from sisteer.data import instructions
from sisteer.data.instructions import Instruction
from test_candidate_si_registry import C16_TEXT_SHA256 as REAL_C16_TEXT_SHA256
from test_candidate_si_registry import SNAPSHOT_SHA256 as REAL_SNAPSHOT_SHA256
from test_candidate_si_registry import _synth_entries, _write

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts/sikl4/make_candidate_sis_subset.py"
_spec = importlib.util.spec_from_file_location("sikl4_make_candidate_sis_subset", SCRIPT)
M = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(M)

LOAD_ENV = M.LOAD_ENV


def _sha(b: bytes | str) -> str:
    return hashlib.sha256(b.encode("utf-8") if isinstance(b, str) else b).hexdigest()


def _make_full(tmp: Path, monkeypatch) -> SimpleNamespace:
    """The synthetic full snapshot in ``tmp``, with the module's subset pins pointed at it (and the subset variable unset). Builds
    wherever ``tmp`` is: only a test that derives needs it outside every git work tree (``_outside_every_work_tree``)."""
    path = Path(_write(tmp, _synth_entries(), name="full.json"))
    raw = path.read_bytes()
    c16 = next(e for e in json.loads(raw)["system_instructions"] if e["id"] == "C16")
    monkeypatch.setattr(C, "CANDIDATE_SIS_FULL_SHA256", _sha(raw))
    monkeypatch.setattr(C, "SUBSET_TEXT_SHA256", {"C16": _sha(c16["text"])})
    monkeypatch.delenv(C.SUBSET_SHA_ENV, raising=False)
    full_c16 = next(i for i in C.load_candidate_instructions(str(path)) if i.id == "C16")
    return SimpleNamespace(path=path, raw=raw, sha=_sha(raw), c16_entry=c16, c16=full_c16, tmp=tmp)


@pytest.fixture()
def full(tmp_path, monkeypatch):
    """The synthetic full snapshot, with the module's subset pins pointed at it (and the subset variable unset)."""
    return _make_full(tmp_path, monkeypatch)


def _outside_every_work_tree(tmp: Path) -> Path:
    """``tmp``, asserted to sit outside every git work tree. The derivation refuses an --out inside one, so a test that derives into
    pytest's tmp dir needs this; with TMPDIR inside a checkout those tests error here, by name, and no other test does."""
    top = M.enclosing_work_tree(tmp / "x")
    assert top is None, (f"pytest's tmp dir {tmp} is inside the git work tree {top}: a test that derives a subset into it needs "
                         "TMPDIR outside every checkout")
    return tmp


@pytest.fixture()
def tmp_outside_work_tree(tmp_path):
    return _outside_every_work_tree(tmp_path)


# for the tests that derive into tmp_path (which is also `full`'s tmp dir: one tmp_path per test)
derives_into_tmp = pytest.mark.usefixtures("tmp_outside_work_tree")


def _subset_doc(full) -> dict:
    """The canonical subset document, as the derivation script writes it."""
    return M.subset_doc(full.raw, ["C16"])


def _put(full, doc, name: str = "subset.json") -> tuple[str, str]:
    """``doc`` written as a subset file -> (path, the file's sha256)."""
    p = full.tmp / name
    p.write_bytes(M.encode_subset(doc))
    return str(p), _sha(p.read_bytes())


def _assert_same_instruction(a: Instruction, b: Instruction) -> None:
    assert a == b
    for f in dataclasses.fields(Instruction):
        assert getattr(a, f.name) == getattr(b, f.name), f.name


# --- the module's pins ------------------------------------------------------------------------------------------------------------

def test_the_pins_are_the_real_snapshots_and_the_registry_seeds():
    assert C.CANDIDATE_SIS_FULL_SHA256 == REAL_SNAPSHOT_SHA256
    assert C.C16_TEXT_SHA256 == REAL_C16_TEXT_SHA256
    assert dict(C.SUBSET_TEXT_SHA256) == {"C16": C.C16_TEXT_SHA256}
    with pytest.raises(TypeError):                      # read-only: a pin is never widened at run time
        C.SUBSET_TEXT_SHA256["C17"] = "0" * 64
    seed = yaml.safe_load((REPO / "configs/registry/si_texts.yaml").read_text())
    [row] = [r for r in seed["candidate_texts"] if r["si_id"] == "C16"]
    assert row["si_text_sha256"] == C.C16_TEXT_SHA256
    assert row["text_source"]["file_sha256"] == C.CANDIDATE_SIS_FULL_SHA256 and row["text_source"]["entry_id"] == "C16"
    assert C.SUBSET_SHA_ENV == "SISTEER_CANDIDATE_SIS_SUBSET_SHA256"


# --- accepted ---------------------------------------------------------------------------------------------------------------------

def test_the_subset_is_accepted_with_the_right_pin_and_yields_the_full_snapshots_c16(full, monkeypatch):
    path, sha = _put(full, _subset_doc(full))
    monkeypatch.setenv(C.SUBSET_SHA_ENV, sha)
    [c16] = C.load_candidate_instructions(path)
    _assert_same_instruction(c16, full.c16)
    assert c16.category == "irrelevant" and c16.checker == "external" and c16.checker_kwargs == {"suite": "align"}
    # the explicit entry point gives the same, with the variable unset
    monkeypatch.delenv(C.SUBSET_SHA_ENV)
    [c16b] = C.load_candidate_subset(path, subset_sha256=sha)
    _assert_same_instruction(c16b, full.c16)


def test_the_subset_path_also_resolves_through_the_default_path_variable(full, monkeypatch):
    path, sha = _put(full, _subset_doc(full))
    monkeypatch.setenv(C.PATH_ENV, path)
    monkeypatch.setenv(C.SUBSET_SHA_ENV, sha)
    assert [i.id for i in C.load_candidate_instructions()] == ["C16"]


def test_expect_sha_still_applies_in_the_subset_mode(full, monkeypatch):
    path, sha = _put(full, _subset_doc(full))
    monkeypatch.setenv(C.SUBSET_SHA_ENV, sha)
    assert len(C.load_candidate_instructions(path, expect_sha=sha)) == 1
    with pytest.raises(ValueError, match="candidate_sis sha mismatch"):
        C.load_candidate_instructions(path, expect_sha="0" * 64)


# --- refused ----------------------------------------------------------------------------------------------------------------------

def _mutate(doc: dict, how: str) -> dict:
    meta, entries = doc["meta"], doc["system_instructions"]
    other = {"id": "C17", "category": "irrelevant", "text": "Synthetic irrelevant instruction C17: always mention c17-marker."}
    if how == "wrong_subset_of":
        meta["subset_of_sha256"] = "f" * 64
    elif how == "text_differs_meta_follows":                 # the text drifted and its meta was rewritten to match
        entries[0]["text"] += " (drifted)"
        meta["text_sha256"]["C16"] = _sha(entries[0]["text"])
    elif how == "text_differs_meta_stale":                   # the text drifted, its meta still claims the pinned sha
        entries[0]["text"] += " (drifted)"
    elif how == "extra_entry":                               # an entry meta.ids does not list
        entries.append(other)
    elif how == "extra_id_listed":                           # an extra id listed in the meta too, with no pinned text sha
        entries.append(other)
        meta["ids"].append("C17")
        meta["text_sha256"]["C17"] = _sha(other["text"])
    elif how == "missing_entry":                             # meta.ids lists C16, the entries do not carry it
        entries.clear()
    elif how == "foreign_id":
        meta["ids"] = ["D1"]
        meta["text_sha256"] = {"D1": "0" * 64}
    elif how == "empty_ids":
        meta["ids"] = []
        meta["text_sha256"] = {}
    elif how == "duplicate_ids":
        meta["ids"] = ["C16", "C16"]
    elif how == "text_sha_keys":
        meta["text_sha256"] = {}
    elif how == "extra_meta_key":
        meta["note"] = "hand-edited"
    elif how == "missing_meta_key":
        del meta["text_sha256"]
    elif how == "extra_top_key":
        doc["extra"] = 1
    elif how == "mislabeled_category":                       # a per-entry check of the full path, applied here too
        entries[0]["category"] = "safety"
    elif how == "duplicate_entry":
        entries.append(dict(entries[0]))
    elif how == "pointer_text":
        entries[0]["text"] = "<<C16_TEXT>>"
    else:
        raise AssertionError(how)
    return doc


@pytest.mark.parametrize("how, needle", [
    ("wrong_subset_of", "meta.subset_of_sha256"),
    ("text_differs_meta_follows", "differ from the pinned full-snapshot text sha256"),
    ("text_differs_meta_stale", "C16's text in the full snapshot"),
    ("extra_entry", "not exactly meta.ids"),
    ("extra_id_listed", "have no pinned text sha256"),
    ("missing_entry", "not exactly meta.ids"),
    ("foreign_id", "are not candidate ids"),
    ("empty_ids", "non-empty list"),
    ("duplicate_ids", "distinct id strings"),
    ("text_sha_keys", "meta.text_sha256 must map exactly meta.ids"),
    ("extra_meta_key", "meta must carry exactly the keys"),
    ("missing_meta_key", "meta must carry exactly the keys"),
    ("extra_top_key", "exactly the keys 'meta' and 'system_instructions'"),
    ("mislabeled_category", "prefix-category binding"),
    ("duplicate_entry", "duplicate id 'C16'"),
    ("pointer_text", "pointer sentinel"),
])
def test_a_malformed_subset_is_refused_by_name(full, monkeypatch, how, needle):
    path, sha = _put(full, _mutate(_subset_doc(full), how))
    monkeypatch.setenv(C.SUBSET_SHA_ENV, sha)                # pinned to its own bytes: only the content can refuse
    with pytest.raises(ValueError, match=re.escape(needle)):
        C.load_candidate_instructions(path)


def test_a_wrong_file_sha_is_refused(full, monkeypatch):
    path, _ = _put(full, _subset_doc(full))
    monkeypatch.setenv(C.SUBSET_SHA_ENV, _sha(b"some other file"))
    with pytest.raises(ValueError, match="candidate_sis subset sha mismatch"):
        C.load_candidate_instructions(path)


@pytest.mark.parametrize("bad", ["abc", "F" * 64, "0" * 63, " " + "0" * 64, "0" * 64 + "\n", "0" * 64 + " "])
def test_a_malformed_pin_is_refused(full, monkeypatch, bad):
    path, _ = _put(full, _subset_doc(full))
    monkeypatch.setenv(C.SUBSET_SHA_ENV, bad)
    with pytest.raises(ValueError, match="64 lowercase hex"):
        C.load_candidate_instructions(path)


def test_the_full_snapshot_is_never_read_as_a_subset(full, monkeypatch):
    monkeypatch.setenv(C.SUBSET_SHA_ENV, full.sha)            # even pinned to its own bytes
    with pytest.raises(ValueError, match="meta must carry exactly the keys"):
        C.load_candidate_instructions(str(full.path))
    bare = full.tmp / "bare.json"                            # the bare-list fixture form carries no meta at all
    bare.write_text(json.dumps(_synth_entries()), encoding="utf-8")
    monkeypatch.setenv(C.SUBSET_SHA_ENV, _sha(bare.read_bytes()))
    with pytest.raises(ValueError, match="exactly the keys 'meta' and 'system_instructions'"):
        C.load_candidate_instructions(str(bare))


def test_a_registry_collision_is_refused_in_the_subset_mode(full, monkeypatch):
    path, sha = _put(full, _subset_doc(full))
    monkeypatch.setenv(C.SUBSET_SHA_ENV, sha)
    squatter = dataclasses.replace(full.c16, text="squatter")
    monkeypatch.setitem(instructions.BY_ID, "C16", squatter)
    with pytest.raises(ValueError, match="already registered"):
        C.load_candidate_instructions(path)
    monkeypatch.setitem(instructions.BY_ID, "C16", full.c16)  # identical content is idempotent, as on the full path
    assert len(C.load_candidate_instructions(path)) == 1


# --- the default path, untouched --------------------------------------------------------------------------------------------------

# the `full` fixture leaves the variable unset; "" sets it empty: the default path either way
subset_var_off = pytest.mark.parametrize("off", [None, ""], ids=["unset", "empty"])


@subset_var_off
def test_a_subset_file_without_the_variable_is_refused_by_the_60_entry_check(full, monkeypatch, off):
    path, _ = _put(full, _subset_doc(full))
    if off is not None:
        monkeypatch.setenv(C.SUBSET_SHA_ENV, off)
    with pytest.raises(ValueError, match="expected exactly 60 entries"):
        C.load_candidate_instructions(path)


@subset_var_off
def test_the_default_path_is_unchanged_with_the_variable_unset_or_empty(full, monkeypatch, off):
    if off is not None:
        monkeypatch.setenv(C.SUBSET_SHA_ENV, off)
    instrs = C.load_candidate_instructions(str(full.path), expect_sha=full.sha)
    assert [i.id for i in instrs] == [e["id"] for e in _synth_entries()]
    assert instrs[0].checker == "external"
    # and a full-path refusal is still the full path's (a 59-entry file names the 60 rule, never the subset mode)
    entries = _synth_entries()[:-1]
    with pytest.raises(ValueError, match="expected exactly 60 entries"):
        C.load_candidate_instructions(_write(full.tmp, entries, name="59.json"))


# --- the env-guarded registration and the cell registry (fresh interpreters, real import paths) ----------------------------------

def _patch_prelude(full) -> str:
    """Python run FIRST in a fresh interpreter: the module's pins pointed at the synthetic snapshot, before instructions.py's
    env-guarded tail runs (candidate_sis imports first by design; see test_candidate_si_registry)."""
    return ("import sisteer.data.candidate_sis as _C\n"
            f"_C.CANDIDATE_SIS_FULL_SHA256 = {full.sha!r}\n"
            f"_C.SUBSET_TEXT_SHA256 = {{'C16': {_sha(full.c16_entry['text'])!r}}}\n")


def _fresh(env_extra: dict, code: str, *args: str, check: bool = True, timeout: int = 120) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    for k in (LOAD_ENV, C.PATH_ENV, C.SUBSET_SHA_ENV):
        env.pop(k, None)
    env["PYTHONPATH"] = str(REPO / "src")
    env.update(env_extra)
    return subprocess.run([sys.executable, "-c", code, *args], env=env, cwd=REPO, capture_output=True, text=True, check=check,
                          timeout=timeout)


# Imports instructions (its env-guarded tail registers the candidates), asserts that C16 is the only candidate id registered, and
# prints the sha256 of the text C16 resolves to.
_C16_ONLY_PROBE = ("import hashlib\n"
                   "from sisteer.data import instructions as I\n"
                   "from sisteer.eval.jailbreak import resolve_instruction\n"
                   "cands = sorted(f'{p}{n}' for p in 'ABC' for n in range(1, 21) if f'{p}{n}' in I.BY_ID)\n"
                   "assert cands == ['C16'], cands\n"
                   "assert len({i.id for i in I.REGISTRY}) == len(I.REGISTRY)\n"
                   "print(hashlib.sha256(resolve_instruction('C16').text.encode()).hexdigest())\n")


def test_the_env_guard_registers_only_c16_under_the_subset(full):
    path, sha = _put(full, _subset_doc(full))
    out = _fresh({LOAD_ENV: path, C.SUBSET_SHA_ENV: sha}, _patch_prelude(full) + _C16_ONLY_PROBE).stdout
    assert out.strip().splitlines()[-1] == _sha(full.c16.text)


def test_the_env_guard_refuses_loudly_on_a_mismatched_pair(full):
    path, sha = _put(full, _subset_doc(full))
    for extra, needle in (({LOAD_ENV: path}, "expected exactly 60 entries"),                           # subset, no variable
                          ({LOAD_ENV: str(full.path), C.SUBSET_SHA_ENV: sha}, "subset sha mismatch"),  # variable, full file
                          ({LOAD_ENV: path, C.SUBSET_SHA_ENV: "0" * 64}, "subset sha mismatch")):     # wrong pin
        r = _fresh(extra, _patch_prelude(full) + "from sisteer.data import instructions\n", check=False)
        assert r.returncode != 0 and needle in r.stderr, (extra, r.stderr[-600:])


_REGISTRY_PROBE = """
import json, os, sys, traceback, yaml
from sisteer.eval import defense_cell as D
out = {}
for view, cfg, view_dir in json.loads(sys.argv[1]):
    os.environ["TT_TWO_SI_DEFENSES_DIR"] = view_dir
    try:
        D.registry_from_cfg(yaml.safe_load(open(cfg)))
        out[view] = {"built": True}
    except ValueError as exc:
        out[view] = {"built": False, "message": str(exc)}
    except Exception:
        out[view] = {"crashed": traceback.format_exc()}
print(json.dumps(out))
"""
SIKL4_VIEWS = ("sikl4s1", "sikl4s2", "sikl4s3")
SIKL4_CELLS = ("safety_refusal__lora_r4", "safety_refusal__psr_kl_all", "si_constitution__lora_r4", "si_constitution__psr_kl_all",
               "C16__lora_r4", "C16__psr_kl_all")


def test_the_c16_cells_resolve_in_each_sikl4_cell_registry_under_the_subset(full):
    """Over each committed sikl4 rendered config and an empty bank view, with the loader in the subset mode: every one of the six
    cells refuses only for its absent artifact, never because C16 does not resolve (tests/test_sikl4_lane.py pins the same over the
    full synthetic snapshot and with the loader off)."""
    path, sha = _put(full, _subset_doc(full))
    views = [(v, str(REPO / f"configs/sdfbase_full22_qwen35_4b_{v}.yaml"), str(full.tmp / "cells_view" / v)) for v in SIKL4_VIEWS]
    r = _fresh({LOAD_ENV: path, C.SUBSET_SHA_ENV: sha}, _patch_prelude(full) + _REGISTRY_PROBE, json.dumps(views),
               check=False, timeout=300)
    assert r.returncode == 0, r.stderr[-3000:]
    probes = json.loads(r.stdout.strip().splitlines()[-1])
    for view, _, view_dir in views:
        probe = probes[view]
        assert "crashed" not in probe and not probe["built"], probe
        for cell in SIKL4_CELLS:
            assert f"{cell!r}: no staged artifact at {Path(view_dir) / cell}" in probe["message"], (view, cell)
            assert f"{cell!r}: SI id 'C16' does not resolve" not in probe["message"], (view, cell)


# Stages every cell of each view (synthetic object + legacy budget sidecar, so the registry's staged-artifact check passes), then
# prints, per view, the registry's rows or its refusal. One interpreter per loader state (the loader reads its switch at import).
_STAGED_PROBE = """
import dataclasses, json, os, sys, yaml
from sisteer.eval import defense_cell as D
out = {}
for view, cfg, view_dir in json.loads(sys.argv[1]):
    os.environ["TT_TWO_SI_DEFENSES_DIR"] = view_dir
    try:
        out[view] = {"built": True, "cells": [dataclasses.asdict(c) for c in D.registry_from_cfg(yaml.safe_load(open(cfg)))]}
    except ValueError as exc:
        out[view] = {"built": False, "message": str(exc)}
print(json.dumps(out, default=str))
"""


def _stage_view(view_dir: Path) -> None:
    from sisteer.eval.defense_cell import sha256_file
    from sisteer.methods import object_artifact_name
    for cell in SIKL4_CELLS:
        d = view_dir / cell
        d.mkdir(parents=True)
        obj = d / object_artifact_name(cell.split("__", 1)[1])
        obj.write_text(f"synthetic object bytes for {view_dir.name}/{cell}")
        (d / "train_budget_meta.json").write_text(json.dumps({"cell_id": cell, "completion_tokens": 512,
                                                              "object_sha256": sha256_file(obj), "max_len": 256}))


def test_every_container_over_a_config_naming_c16_needs_the_pair_and_it_changes_no_other_cell(full):
    """Who sets the pair (plan 7.2 step 5). With all six cells of each sikl4 view staged: without the loader the registry refuses the
    WHOLE config, naming only the two C16 cells, so a container serving a safety or constitution cell over that config refuses at
    standup too; under the subset it builds, and every row, the four non-C16 cells' included, equals the one built under the full
    snapshot. So the pair goes on every container of a row whose config lists a C16 cell, and setting it there changes nothing."""
    path, sha = _put(full, _subset_doc(full))
    views = [(v, str(REPO / f"configs/sdfbase_full22_qwen35_4b_{v}.yaml"), str(full.tmp / "staged" / v)) for v in SIKL4_VIEWS]
    for _, _, view_dir in views:
        _stage_view(Path(view_dir))
    states = {"off": {}, "full": {LOAD_ENV: str(full.path)}, "subset": {LOAD_ENV: path, C.SUBSET_SHA_ENV: sha}}
    probes = {}
    for state, extra in states.items():
        r = _fresh(extra, _patch_prelude(full) + _STAGED_PROBE, json.dumps(views), check=False, timeout=300)
        assert r.returncode == 0, (state, r.stderr[-3000:])
        probes[state] = json.loads(r.stdout.strip().splitlines()[-1])
    for view, _, _ in views:
        off, on_full, on_subset = probes["off"][view], probes["full"][view], probes["subset"][view]
        assert not off["built"], off
        assert off["message"].startswith("'explicit_cells' validation failed for 2 cell id(s)"), off["message"]
        for cell in SIKL4_CELLS:
            named = f"{cell!r}: " in off["message"]
            assert named == cell.startswith("C16__"), (view, cell, off["message"])
        assert on_full["built"] and on_subset["built"], (on_full, on_subset)
        assert [c["id"] for c in on_full["cells"]] == ["none", *SIKL4_CELLS]
        assert on_subset["cells"] == on_full["cells"], view


# --- the derivation script --------------------------------------------------------------------------------------------------------

def _derive(full, out: Path, *, ids=("C16",), full_sha=None) -> dict:
    return M.derive(full.path, full_sha or full.sha, list(ids), out)


@derives_into_tmp
def test_the_script_writes_a_subset_the_subset_mode_accepts(full):
    out = full.tmp / "out" / "c16.subset.json"
    rep = _derive(full, out)
    blob = out.read_bytes()
    assert rep["sha256"] == _sha(blob) and rep["out"] == str(out.resolve()) and rep["bytes"] == len(blob)
    assert rep["env"] == {LOAD_ENV: str(out.resolve()), C.SUBSET_SHA_ENV: _sha(blob)} and rep["unchanged"] is False
    doc = json.loads(blob)
    assert doc == {"meta": {"subset_of_sha256": full.sha, "ids": ["C16"], "text_sha256": {"C16": _sha(full.c16_entry["text"])}},
                   "system_instructions": [full.c16_entry]}          # the entry verbatim, every field
    assert rep["meta"] == doc["meta"]
    [c16] = C.load_candidate_subset(out, subset_sha256=rep["sha256"])
    _assert_same_instruction(c16, full.c16)
    assert not list(out.parent.glob(".c16.subset.json.*"))            # no temp file left behind


@derives_into_tmp
def test_the_script_is_deterministic(full):
    a, b = full.tmp / "a.json", full.tmp / "b.json"
    ra, rb = _derive(full, a), _derive(full, b)
    assert a.read_bytes() == b.read_bytes() and ra["sha256"] == rb["sha256"]
    again = _derive(full, a)                                          # re-deriving onto the same file is a no-op
    assert again["sha256"] == ra["sha256"] and again["unchanged"] is True


@derives_into_tmp
def test_the_script_refuses_by_name(full, monkeypatch):
    out = full.tmp / "s.json"
    with pytest.raises(M.Refusal, match="not the loader's pinned full snapshot"):
        _derive(full, out, full_sha="a" * 64)
    monkeypatch.setattr(C, "CANDIDATE_SIS_FULL_SHA256", "b" * 64)     # the pin and --full-sha agree, the file does not
    with pytest.raises(M.Refusal, match="hashes to"):
        _derive(full, out, full_sha="b" * 64)
    monkeypatch.setattr(C, "CANDIDATE_SIS_FULL_SHA256", full.sha)
    with pytest.raises(M.Refusal, match="inside the repository"):
        _derive(full, REPO / "c16.subset.json")
    assert not (REPO / "c16.subset.json").exists()
    with pytest.raises(M.Refusal, match="names the full snapshot itself"):
        _derive(full, full.path)
    with pytest.raises(M.Refusal, match="have no pinned text sha256"):
        M._parse_ids("C16,C17")
    with pytest.raises(M.Refusal, match="repeats an id"):
        M._parse_ids("C16,C16")
    monkeypatch.setenv(C.SUBSET_SHA_ENV, "c" * 64)
    with pytest.raises(M.Refusal, match=C.SUBSET_SHA_ENV):
        _derive(full, out)
    monkeypatch.delenv(C.SUBSET_SHA_ENV)
    out.write_text("a hand-edited subset\n")
    with pytest.raises(M.Refusal, match="never changes silently"):
        _derive(full, out)
    assert out.read_text() == "a hand-edited subset\n"


@derives_into_tmp
def test_the_script_refuses_an_out_inside_any_git_work_tree(full):
    """Not only this checkout: another repository (a .git directory), a linked worktree such as the base checkout's siblings (a .git
    FILE), at any depth, before any directory is made."""
    other = full.tmp / "other_repo"
    (other / ".git").mkdir(parents=True)
    linked = full.tmp / "linked_worktree"
    linked.mkdir()
    (linked / ".git").write_text("gitdir: /nowhere/elsewhere\n")
    for top, out in ((other, other / "c16.subset.json"), (other, other / "deep" / "new" / "c16.subset.json"),
                     (linked, linked / "c16.subset.json")):
        with pytest.raises(M.Refusal, match=re.escape(f"inside the git work tree {top.resolve()}")):
            _derive(full, out)
        assert not out.exists()
    assert not (other / "deep").exists()
    assert M.enclosing_work_tree(full.tmp / "plain" / "c16.subset.json") is None


@derives_into_tmp
def test_the_script_cli_reports_a_refusal_on_stderr(full, capsys):
    rc = M.main(["--full", str(full.path), "--full-sha", "a" * 64, "--ids", "C16", "--out", str(full.tmp / "x.json")])
    assert rc == 2 and "REFUSED" in capsys.readouterr().err and not (full.tmp / "x.json").exists()
    rc = M.main(["--full", str(full.path), "--full-sha", full.sha.upper(), "--ids", "C16", "--out", str(full.tmp / "x.json")])
    assert rc == 0 and json.loads(capsys.readouterr().out)["sha256"] == _sha((full.tmp / "x.json").read_bytes())


def test_a_tmp_dir_inside_a_work_tree_stops_only_the_derive_tests(tmp_path, monkeypatch):
    """With TMPDIR inside a checkout the synthetic snapshot still builds and the subset mode still loads, so every loader test runs;
    only the guard the derive tests carry refuses, and it names the work tree."""
    checkout = tmp_path / "checkout"
    (checkout / ".git").mkdir(parents=True)
    inside = checkout / "tmp"
    inside.mkdir()
    full = _make_full(inside, monkeypatch)
    path, sha = _put(full, _subset_doc(full))
    monkeypatch.setenv(C.SUBSET_SHA_ENV, sha)
    [c16] = C.load_candidate_instructions(path)
    _assert_same_instruction(c16, full.c16)
    with pytest.raises(AssertionError, match=re.escape(f"inside the git work tree {checkout}:")):
        _outside_every_work_tree(inside)


# --- the REAL snapshot (only when SISTEER_CANDIDATE_SIS names it) ------------------------------------------------------------------

REAL = os.environ.get(C.PATH_ENV, "")
needs_real = pytest.mark.skipif(not (REAL and Path(REAL).is_file()), reason=f"{C.PATH_ENV} does not name the candidate-SI snapshot")


@needs_real
@derives_into_tmp
def test_real_subset_derives_and_yields_the_full_snapshots_c16(tmp_path, monkeypatch):
    monkeypatch.delenv(C.SUBSET_SHA_ENV, raising=False)
    out = tmp_path / "candidate_sis.C16.subset.json"
    rep = M.derive(Path(REAL), C.CANDIDATE_SIS_FULL_SHA256, ["C16"], out)
    assert M.derive(Path(REAL), C.CANDIDATE_SIS_FULL_SHA256, ["C16"], tmp_path / "again.json")["sha256"] == rep["sha256"]
    full_c16 = next(i for i in C.load_candidate_instructions(REAL, expect_sha=C.CANDIDATE_SIS_FULL_SHA256) if i.id == "C16")
    [c16] = C.load_candidate_subset(out, subset_sha256=rep["sha256"])
    _assert_same_instruction(c16, full_c16)
    assert len(c16.text) == 517 and _sha(c16.text) == C.C16_TEXT_SHA256
    doc = json.loads(out.read_bytes())
    assert [e["id"] for e in doc["system_instructions"]] == ["C16"]
    assert doc["meta"] == {"subset_of_sha256": C.CANDIDATE_SIS_FULL_SHA256, "ids": ["C16"], "text_sha256": {"C16": C.C16_TEXT_SHA256}}


@needs_real
@derives_into_tmp
def test_real_subset_resolves_c16_through_the_trainers_resolver(tmp_path):
    out = tmp_path / "candidate_sis.C16.subset.json"
    env = {k: v for k, v in os.environ.items() if k != C.SUBSET_SHA_ENV}
    rep = subprocess.run([sys.executable, str(SCRIPT), "--full", REAL, "--full-sha", C.CANDIDATE_SIS_FULL_SHA256, "--ids", "C16",
                          "--out", str(out)], capture_output=True, text=True, check=True, env=env, timeout=120)
    sha = json.loads(rep.stdout)["sha256"]
    r = _fresh({LOAD_ENV: str(out), C.SUBSET_SHA_ENV: sha}, _C16_ONLY_PROBE)
    assert r.stdout.strip().splitlines()[-1] == C.C16_TEXT_SHA256
