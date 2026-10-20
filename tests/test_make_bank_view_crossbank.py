"""native-psr-serving step 7 — the bank-view builder's CROSS-BANK PIN FORM
(``full22/make_bank_view.py``).

A roster cell may map to a MAPPING that names a PIN in another authority
instead of a staged id of this wave's bank::

    arditi: {pins: <pins file>, pin: gemma4e4b.arditi}

and nothing else — the form has no ``staged_id`` field, so a roster can point
at a pin but can never choose a RUNG (the production Arditi pin is alpha
0.05, and naming ``ar_r0p1__arditi`` beside it is exactly the substitution this
form exists to make impossible).

Built here on a SYNTHETIC PAIR OF BANKS — a full22-shaped wave bank and an
arditi-shaped strength bank under one object root — with a pins fixture shaped
EXACTLY like ``configs/misalign_baseline_pins.json``'s gemma4e4b/arditi entry
(``bank``, ``source``, ``view``, ``staged_id``, ``object_sha256``, ``kind``,
``alpha``, ``served_layer``, ``fit_layer_stored``, ``identity_sidecar``).

Pinned:

- a mixed view BUILDS: the string cell exactly as before, the pin cell as a
  symlink to the SOURCE bank's staged copy, and ``VIEW_MANIFEST.json`` records
  the source bank + the pin path;
- REFUSALS, each a known-bad run against the builder: a sha mismatch; a blob
  whose stored ``kind`` is not the pin's; a SUBSTITUTED rung (the 0.1 rung's
  bytes under the 0.05 rung's staged name); a roster that tries to name
  ``staged_id`` directly; a pin file / entry that does not exist; a source view
  re-aimed at another rung; a ``served_layer`` the blob does not carry; an
  identity sidecar that is not the pin's;
- the STRING form is byte-identical: a string-only view's manifest entries
  carry exactly the keys they carried before this form existed.

CPU-only; synthetic objects; no cluster, no HF cache."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import yaml

REPO = Path(__file__).resolve().parents[1]
for p in (REPO / "src", REPO / "scripts" / "sdfbase" / "full22"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import make_bank_view as mbv  # noqa: E402

TARGET = "gemma4e4b"
PSR_CELL = "safety_refusal__psr_kl_all"
PSR_STAGED = "si12__safety_refusal__psr_kl_all"
ARDITI_CELL = "arditi"
ARDITI_STAGED = "ar_r0p05__arditi"
OTHER_RUNG = "ar_r0p1__arditi"
SERVED_LAYER, FIT_LAYER, HIDDEN = 18, 19, 16


def _sha(p: Path) -> str:
    return mbv.sha256_file(p)


def _static_r_blob(*, layer: int = FIT_LAYER, seed: int = 4, kind: str = "static_r") -> dict:
    g = torch.Generator().manual_seed(seed)
    r = torch.randn(HIDDEN, generator=g)
    return {"kind": kind, "layers": [layer], "rs": {layer: r}, "base_scale": 1.0,
            "steer_positions": "prefill_all", "site": "pre",
            "metadata": {"per_layer_r_norm": {layer: float(r.norm())},
                         "served_layer": layer - 1, "fit_layer": layer, "alpha": 0.05}}


def _stage_arditi(root: Path, staged_id: str, *, blob: dict, view: str) -> Path:
    """One rung of the arditi strength bank: the staged copy (object + its
    ``fit_provenance.json`` + ``staged_from.json``) and the view symlink that
    publishes it, exactly as its builder leaves them."""
    bank = root / "exp/objects/arditi_strength" / TARGET / "strength"
    d = bank / "cells" / staged_id
    d.mkdir(parents=True, exist_ok=True)
    torch.save(blob, d / "object.pt")
    (d / "fit_provenance.json").write_text(json.dumps({
        "resolved_model_id": "google/gemma-4-E4B-it",
        "resolved_revision": "e" * 40, "chat_template_hash": "0a2c8073c878ab1d",
        "writer": "scripts/arditi/fit_arditi.py", "object_sha256": None,
        "positions_at_fit": "all", "alpha": 0.05, "served_layer": SERVED_LAYER,
        "fit_layer": FIT_LAYER}, indent=1))
    (d / "staged_from.json").write_text(json.dumps({
        "staged_id": staged_id, "cell_id": ARDITI_CELL, "view": view,
        "staged_by": "scripts/arditi/make_arditi_objects.py",
        "object_sha256": _sha(d / "object.pt")}, indent=1))
    vd = bank / "cells_view" / view
    vd.mkdir(parents=True, exist_ok=True)
    link = vd / ARDITI_CELL
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(Path(os.path.relpath(d, vd)), target_is_directory=True)
    return d


def _stage_wave_cell(root: Path, staged_id: str, cell_id: str) -> Path:
    """A staged cell of THIS wave's bank, with the training budget sidecar."""
    d = root / "exp/objects/sdfbase/full22" / TARGET / "cells" / staged_id
    d.mkdir(parents=True, exist_ok=True)
    g = torch.Generator().manual_seed(9)
    torch.save({"kind": "psr", "layers": [0], "positions": "response", "response_header_len": 3,
                "directions": {"0": torch.randn(HIDDEN, generator=g)},
                "gate_w": {"0": torch.randn(HIDDEN, generator=g)},
                "gate_b": {"0": torch.randn(1, generator=g)}}, d / "object.pt")
    (d / "staged_from.json").write_text(json.dumps({"source": "synthetic"}))
    (d / "train_budget_meta.json").write_text(json.dumps(
        {"cell_id": cell_id, "object_sha256": _sha(d / "object.pt"), "completion_tokens": 512}))
    return d


def _pins_doc(root: Path, *, staged_id: str = ARDITI_STAGED, view: str = "ar_r0p05",
              sha: str | None = None, kind: str = "static_r",
              served_layer: int = SERVED_LAYER) -> dict:
    """The misalignment pins file's shape, field for field (`:16-38`)."""
    obj = root / "exp/objects/arditi_strength" / TARGET / "strength" / "cells" / staged_id / "object.pt"
    return {"_schema": "misalign_baseline_pins/1",
            "_ruling": "USER RULING (synthetic fixture)",
            "_cells": ["arditi", "alphasteer"],
            TARGET: {
                "model": "google/gemma-4-E4B-it",
                "resolved_revision": "e" * 40,
                "chat_template_hash": "0a2c8073c878ab1d",
                "arditi": {
                    "bank": "arditi_strength",
                    "source": f"arditi_strength/{TARGET}/strength/cells_view/{view}/{ARDITI_CELL}",
                    "view": view,
                    "staged_id": staged_id,
                    "object_sha256": sha if sha is not None else _sha(obj),
                    "kind": kind,
                    "alpha": 0.05,
                    "dose_unit": "residual_norm_fraction",
                    "phase": "prefill",
                    "served_layer": served_layer,
                    "fit_layer_stored": FIT_LAYER,
                    "edit_norm": 4.25,
                    "h_ref": 85.0,
                    "identity_sidecar": "fit_provenance.json",
                    "chat_template_hash": "0a2c8073c878ab1d",
                    "resolved_revision": "e" * 40,
                    "selection_outcome": "ruled",
                    "producer": "scripts/arditi/make_arditi_objects.py"}}}


def _roster(*, cells: dict) -> dict:
    return {"wave": "full22", "bank_root": "exp/objects/sdfbase/full22",
            "lane_flags": {},
            "targets": {TARGET: {"views": {"psr2cert": {
                "config": "configs/x.yaml", "manifest": "configs/y.yaml", "cells": cells}}}}}


@pytest.fixture
def world(tmp_path):
    """A pair of banks + both pin authorities + the roster that names them."""
    root = tmp_path / "shared"
    _stage_wave_cell(root, PSR_STAGED, PSR_CELL)
    _stage_arditi(root, ARDITI_STAGED, blob=_static_r_blob(), view="ar_r0p05")
    _stage_arditi(root, OTHER_RUNG, blob=_static_r_blob(seed=77), view="ar_r0p1")
    mis = tmp_path / "configs" / "misalign_baseline_pins.json"
    mis.parent.mkdir(parents=True, exist_ok=True)
    mis.write_text(json.dumps(_pins_doc(root), indent=1))
    wave_pins = tmp_path / "configs" / "sdfbase_full22_pins.json"
    wave_obj = root / "exp/objects/sdfbase/full22" / TARGET / "cells" / PSR_STAGED / "object.pt"
    wave_pins.write_text(json.dumps({TARGET: {PSR_STAGED: {"object_sha256": _sha(wave_obj)}}}))
    roster_path = tmp_path / "roster.yaml"
    mapping = {"pins": str(mis), "pin": f"{TARGET}.{ARDITI_CELL}"}
    roster_path.write_text(yaml.safe_dump(
        _roster(cells={PSR_CELL: PSR_STAGED, ARDITI_CELL: mapping}), sort_keys=False))
    return {"root": root, "bank": root / "exp/objects/sdfbase/full22" / TARGET,
            "mis": mis, "wave_pins": wave_pins, "roster": roster_path, "tmp": tmp_path}


def _build(world, *, check: bool = False, extra: tuple = ()) -> int:
    argv = ["--target", TARGET, "--view", "psr2cert", "--roster", str(world["roster"]),
            "--pins", str(world["wave_pins"]), "--bank", str(world["bank"]), *extra]
    if check:
        argv.append("--check")
    return mbv.main(argv)


def _view_manifest(world) -> dict:
    return json.loads((world["bank"] / "cells_view" / "psr2cert" / "VIEW_MANIFEST.json").read_text())


def _repoint(world, **over) -> None:
    """Rewrite the misalignment pin file with an override."""
    world["mis"].write_text(json.dumps(_pins_doc(world["root"], **over), indent=1))


# --------------------------------------------------------------------------- the happy path

def test_a_mixed_view_builds_the_string_cell_and_the_cross_bank_pin_cell(world, capsys):
    assert _build(world) == 0, capsys.readouterr().out
    view = world["bank"] / "cells_view" / "psr2cert"
    # the STRING cell: this wave's staged copy, exactly as before
    assert (view / PSR_CELL / "object.pt").is_file()
    # the PIN cell: a symlink into the ARDITI bank's staged copy — never a second copy of
    # the object, and never the pin's view entry (the staged dir is where the identity is)
    link = view / ARDITI_CELL
    assert link.is_symlink()
    staged = world["root"] / "exp/objects/arditi_strength" / TARGET / "strength" / "cells" / ARDITI_STAGED
    assert Path(os.path.realpath(link)) == staged.resolve()
    assert (link / "fit_provenance.json").is_file() and (link / "object.pt").is_file()
    man = _view_manifest(world)
    by_cell = {e["cell_id"]: e for e in man["entries"]}
    pin_rec = by_cell[ARDITI_CELL]
    assert pin_rec["form"] == "pin" and pin_rec["staged_id"] == ARDITI_STAGED
    assert pin_rec["source_bank"] == "arditi_strength" and pin_rec["pin"] == f"{TARGET}.{ARDITI_CELL}"
    assert pin_rec["pins"] == str(world["mis"]) and pin_rec["kind"] == "static_r"
    assert pin_rec["served_layer"] == SERVED_LAYER and pin_rec["identity"] == "fit_provenance"
    assert pin_rec["object_sha256"] == _sha(staged / "object.pt") and pin_rec["mode"] == "symlink"
    assert pin_rec["source"] == str(staged) and pin_rec["source_view"] == "ar_r0p05"
    # the string entry carries NO cross-bank keys: its record is what it always was
    assert not ({"form", "pins", "pin", "source_bank"} & set(by_cell[PSR_CELL]))
    # ... and --check re-verifies the built view
    assert _build(world, check=True) == 0


def test_the_builder_is_idempotent_and_check_passes_twice(world):
    assert _build(world) == 0
    first = _view_manifest(world)["entries"]
    assert _build(world) == 0
    assert [{k: v for k, v in e.items() if k != "written_at"} for e in _view_manifest(world)["entries"]] \
        == [{k: v for k, v in e.items() if k != "written_at"} for e in first]
    assert _build(world, check=True) == 0


# --------------------------------------------------------------------------- the refusals

def test_a_sha_mismatch_refuses(world, capsys):
    _repoint(world, sha="0" * 64)
    assert _build(world) == 2
    out = capsys.readouterr().out
    assert "!= pinned 000000000000" in out and ARDITI_STAGED in out


def test_a_substituted_rung_refuses_on_the_sha(world, capsys):
    """The known-bad case: the 0.1 rung's BYTES staged under the 0.05 rung's name. The pin
    names the staged id, so the substitution is caught by re-hashing that dir."""
    staged = world["root"] / "exp/objects/arditi_strength" / TARGET / "strength" / "cells"
    torch.save(_static_r_blob(seed=77), staged / ARDITI_STAGED / "object.pt")   # the 0.1 rung's
    assert _build(world) == 2
    out = capsys.readouterr().out
    assert "a substituted rung or a drifted object" in out and ARDITI_STAGED in out


def test_a_roster_that_names_the_rung_directly_refuses(world, capsys):
    """The mapping form has no ``staged_id`` (nor ``bank`` / ``object_sha256``): naming one
    is refused by name, so a roster can never select a rung the pin did not."""
    roster = yaml.safe_load(world["roster"].read_text())
    cells = roster["targets"][TARGET]["views"]["psr2cert"]["cells"]
    cells[ARDITI_CELL] = {"pins": str(world["mis"]), "pin": f"{TARGET}.{ARDITI_CELL}",
                          "staged_id": OTHER_RUNG}
    world["roster"].write_text(yaml.safe_dump(roster, sort_keys=False))
    assert _build(world) == 2
    out = capsys.readouterr().out
    assert "takes exactly ['pins', 'pin']" in out and "staged_id" in out


def test_a_blob_of_another_kind_refuses(world, capsys):
    staged = world["root"] / "exp/objects/arditi_strength" / TARGET / "strength" / "cells" / ARDITI_STAGED
    torch.save(_static_r_blob(kind="alphasteer"), staged / "object.pt")
    _repoint(world)                       # re-pin to the new bytes: only the KIND is wrong
    assert _build(world) == 2
    out = capsys.readouterr().out
    assert "the blob stores kind 'alphasteer'" in out


def test_a_pin_whose_kind_is_not_the_arms_refuses(world, capsys):
    _repoint(world, kind="alphasteer")
    assert _build(world) == 2
    assert "ARM_OBJECT_KINDS['arditi'] is 'static_r'" in capsys.readouterr().out


def test_a_served_layer_the_blob_does_not_carry_refuses(world, capsys):
    _repoint(world, served_layer=7)
    assert _build(world) == 2
    assert "do not include the pin's served_layer 7" in capsys.readouterr().out


def test_a_source_view_aimed_at_another_rung_refuses(world, capsys):
    """The pin names a view path as well as a staged id; a view symlink re-aimed at another
    rung is a published substitution and must not pass either."""
    vd = world["root"] / "exp/objects/arditi_strength" / TARGET / "strength" / "cells_view" / "ar_r0p05"
    (vd / ARDITI_CELL).unlink()
    (vd / ARDITI_CELL).symlink_to(Path("../../cells") / OTHER_RUNG, target_is_directory=True)
    assert _build(world) == 2
    assert "not the staged copy" in capsys.readouterr().out


def test_a_missing_pin_file_and_a_missing_pin_entry_refuse(world, capsys):
    roster = yaml.safe_load(world["roster"].read_text())
    cells = roster["targets"][TARGET]["views"]["psr2cert"]["cells"]
    cells[ARDITI_CELL] = {"pins": str(world["tmp"] / "nope.json"), "pin": f"{TARGET}.{ARDITI_CELL}"}
    world["roster"].write_text(yaml.safe_dump(roster, sort_keys=False))
    assert _build(world) == 2
    assert "no pins file at" in capsys.readouterr().out
    cells[ARDITI_CELL] = {"pins": str(world["mis"]), "pin": "qwen35_4b.arditi"}
    world["roster"].write_text(yaml.safe_dump(roster, sort_keys=False))
    assert _build(world) == 2
    assert "has no qwen35_4b.arditi entry" in capsys.readouterr().out
    cells[ARDITI_CELL] = {"pins": str(world["mis"]), "pin": "gemma4e4b"}
    world["roster"].write_text(yaml.safe_dump(roster, sort_keys=False))
    assert _build(world) == 2
    assert "is not `<model key>.<cell>`" in capsys.readouterr().out


def test_a_sidecar_that_is_not_the_pins_refuses(world, capsys):
    staged = world["root"] / "exp/objects/arditi_strength" / TARGET / "strength" / "cells" / ARDITI_STAGED
    (staged / "train_budget_meta.json").write_text(json.dumps({"cell_id": ARDITI_CELL}))
    assert _build(world) == 2
    out = capsys.readouterr().out
    assert "records 'fit_provenance.json'" in out or "more than one identity sidecar" in out


def test_the_object_root_derivation_refuses_a_bank_that_is_not_the_rosters(world, capsys):
    """The cross-bank path is resolved from ``--bank`` (ONE derivation, so it follows the
    cluster's $SHARED or a test root). A ``--bank`` that does not end with the roster's
    ``<bank_root>/<target>`` cannot name an object root and refuses rather than guessing."""
    argv = ["--target", TARGET, "--view", "psr2cert", "--roster", str(world["roster"]),
            "--pins", str(world["wave_pins"]), "--bank", str(world["tmp"] / "elsewhere")]
    rc = mbv.main(argv)
    assert rc == 2
    assert "does not end with the roster's" in capsys.readouterr().out


# --------------------------------------------------------------------------- the string form

def test_a_string_only_view_is_unchanged(world, capsys):
    """The form a production roster uses is byte-identical: the entry record carries exactly
    the keys it carried before the pin form existed."""
    roster = yaml.safe_load(world["roster"].read_text())
    roster["targets"][TARGET]["views"]["psr2cert"]["cells"] = {PSR_CELL: PSR_STAGED}
    world["roster"].write_text(yaml.safe_dump(roster, sort_keys=False))
    assert _build(world) == 0, capsys.readouterr().out
    (entry,) = _view_manifest(world)["entries"]
    assert set(entry) == {"cell_id", "staged_id", "source", "object_sha256", "identity",
                          "mode", "sidecar_cell_id"}
    assert entry["identity"] == "staged" and entry["mode"] == "symlink"


def test_the_production_rosters_are_unchanged_by_the_new_form(capsys):
    """No production roster uses the mapping form — the two committed rosters are read here
    and their cells are all strings (or the prompted ``null``), so every production view's
    manifest is what it was. The certification-only roster's pin-form cells are EXACTLY the
    two named here, keyed by TARGET as well as view: gemma4e4b's psr2cert Arditi object and
    qwen35_4b's pairedcert paired-write object (a read-only store,
    configs/reseed_paired_pins.json). llama8b's pairedcert view names the SAME cell in the
    string form — the bare wave key ``derived__<cell>`` — by design: its object is this wave's
    own derived object, not a cross-bank pin (e2be91ef), so the form is read off the roster
    per target rather than assumed from the view name."""
    paired_cell = "constv2-mix-full__psr_ce_all_minus_ctl"
    pin_form_cells = {("gemma4e4b", "psr2cert", "arditi"),
                      ("qwen35_4b", "pairedcert", paired_cell)}
    seen_pin_form = set()
    for rel in ("configs/sdfbase_full22_roster.yaml", "configs/sdfbase_full22_roster_psrcert.yaml"):
        roster = yaml.safe_load((REPO / rel).read_text())
        for target, t in roster["targets"].items():
            for view, v in t["views"].items():
                for cid, sid in (v.get("cells") or {}).items():
                    mapping = isinstance(sid, dict)
                    if rel.endswith("psrcert.yaml") and (target, view, cid) in pin_form_cells:
                        assert mapping and set(sid) == {"pins", "pin"}, (target, view, cid, sid)
                        seen_pin_form.add((target, view, cid))
                    else:
                        assert not mapping, (rel, target, view, cid)
        if rel.endswith("psrcert.yaml"):
            # the bare wave key, positively: llama8b's pairedcert cell is the string form naming
            # THIS wave's derived object (the same staged-id grammar every string cell uses)
            assert roster["targets"]["llama8b"]["views"]["pairedcert"]["cells"][paired_cell] == f"derived__{paired_cell}"
    assert seen_pin_form == pin_form_cells        # every declared pin-form cell exists, and no other


def test_the_builder_runs_as_a_script_over_the_pin_form(world):
    """The builder is invoked as a SUBPROCESS by common.sh; the pin form works there too
    (the lazy torch / registry imports are inside the function, not at module import)."""
    r = subprocess.run(
        [sys.executable, str(REPO / "scripts/sdfbase/full22/make_bank_view.py"),
         "--target", TARGET, "--view", "psr2cert", "--roster", str(world["roster"]),
         "--pins", str(world["wave_pins"]), "--bank", str(world["bank"])],
        capture_output=True, text=True, cwd=REPO,
        env={**os.environ, "PYTHONPATH": str(REPO / "src")})
    assert r.returncode == 0, (r.stdout, r.stderr)
    assert f"{ARDITI_CELL:<32} <- {ARDITI_STAGED}" in r.stdout


# --------------------------------------------------------------------------- #
# the pin form may serve a DERIVED object under a cell id that is a route name  #
# --------------------------------------------------------------------------- #


def test_the_pin_forms_declared_derived_id_is_what_the_sidecar_is_held_to(tmp_path, monkeypatch):
    """The paired write: the cell `constv2-mix-full__psr_ce_all_minus_ctl` serves a
    derived object whose sidecar says derived_id `constv2-mix-full__psr_ce_all_minus_ctl`. The STRING form keeps
    holding a derived sidecar to the CELL id (a wave cell is the object it is named after); the PIN form holds it to
    the pin's declared `derived_id` when the pin carries one, else the cell id as before — and records which."""
    import json
    seen = []
    monkeypatch.setattr(mbv, "derived_object_problems", None, raising=False)   # not imported at module level
    from sisteer.eval import train_budget as tb
    monkeypatch.setattr(tb, "derived_object_problems", lambda obj, *, derived_id=None, **kw: seen.append(derived_id) or [])
    src = tmp_path / "cells" / "paired_s1__x"; src.mkdir(parents=True)
    obj = src / "object.pt"; obj.write_bytes(b"blob"); sha = _sha(obj)
    (src / "derived_meta.json").write_text(json.dumps({"derived_id": "x_minus_ctl", "kind": "psr_paired_write", "object_sha256": sha}))
    (src / "staged_from.json").write_text("{}")
    # default (the string form's call): held to the CELL id
    rec = {}; mbv.identity_entry_problems("x_minus_ctl_paired", "paired_s1__x", src, obj, sha, rec)
    assert seen[-1] == "x_minus_ctl_paired" and rec["derived_id_expected"] == "x_minus_ctl_paired"
    # the pin form's call with a declared derived_id: held to THAT, and recorded
    rec = {}; mbv.identity_entry_problems("x_minus_ctl_paired", "paired_s1__x", src, obj, sha, rec, derived_id="x_minus_ctl")
    assert seen[-1] == "x_minus_ctl" and rec["derived_id_expected"] == "x_minus_ctl" and rec["derived_id"] == "x_minus_ctl"
    # an explicit None falls back to the cell id (a pin carrying no derived_id changes nothing)
    rec = {}; mbv.identity_entry_problems("x_minus_ctl_paired", "paired_s1__x", src, obj, sha, rec, derived_id=None)
    assert seen[-1] == "x_minus_ctl_paired"
    # the plumbing is real, not decorative: the sidecar's object_sha256 is still held to the bytes
    (src / "derived_meta.json").write_text(json.dumps({"derived_id": "x_minus_ctl", "kind": "psr_paired_write", "object_sha256": "0" * 64}))
    rec = {}; probs = mbv.identity_entry_problems("x_minus_ctl_paired", "paired_s1__x", src, obj, sha, rec, derived_id="x_minus_ctl")
    assert any("object_sha256" in q for q in probs)
    # and the pin-form call site passes the pin's declared derived_id (pinned by text so the wiring cannot drift)
    text = (REPO / "scripts" / "sdfbase" / "full22" / "make_bank_view.py").read_text() if "REPO" in globals() else Path(mbv.__file__).read_text()
    assert 'derived_id=(str(pin["derived_id"]) if pin.get("derived_id") else None)' in text


def test_blob_identity_reads_a_vector_all_blobs_layers_from_its_vectors_keys(tmp_path):
    """The all-positions Arditi rungs the padded steered wave pins (2026-09-23) are `vector_all` blobs:
    extraction/multilayer.MultiLayerVectorObject.save writes `vectors` keyed by the SERVED block layer and NO `layers`
    list, so the pin form's layer gate must read the `vectors` keys (third spelling beside `layers` and `rs`) — a blob
    read as "declares no layers" would refuse every pinned rung by name. Site post: served == stored, no shift."""
    g = torch.Generator().manual_seed(3)
    blob = {"kind": "vector_all", "vectors": {"12": torch.randn(HIDDEN, generator=g)}, "positions": "all",
            "skip_bos": False, "fit_positions": "all", "fit_response_header_len": None}
    obj = tmp_path / "object.pt"; torch.save(blob, obj)
    pin = {"kind": "vector_all", "served_layer": 12}
    assert mbv.blob_identity_problems("ar_c0p3__arditi_allpos", obj, pin) == []
    probs = mbv.blob_identity_problems("ar_c0p3__arditi_allpos", obj, {"kind": "vector_all", "served_layer": 13})
    assert len(probs) == 1 and "do not include the pin's served_layer 13" in probs[0] and "[12]" in probs[0]
    # the two older spellings are untouched: `layers` (psr / paired write) and `rs` at site pre (static_r, shifted)
    torch.save({"kind": "psr_paired_write", "layers": list(range(32)), "site": "post"}, obj)
    assert mbv.blob_identity_problems("paired_s1__x", obj, {"kind": "psr_paired_write", "served_layer": 0}) == []
    torch.save(_static_r_blob(), obj)
    assert mbv.blob_identity_problems(ARDITI_STAGED, obj, {"kind": "static_r", "served_layer": SERVED_LAYER}) == []
    # and a blob with none of the three still refuses by name
    torch.save({"kind": "vector_all", "positions": "all"}, obj)
    assert any("declares no layers" in q for q in mbv.blob_identity_problems("x", obj, pin))
