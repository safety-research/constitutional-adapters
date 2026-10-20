"""E3 (math-eight-arms-extension; decisions-e3 E3-2 / E3-3): the it-doc ADMISSION RECORD and the BANK BINDING both math launchers
run at the bank door — `scripts/sdfbase/full22/itdoc_bind.py` (stdlib only).

The failure this guards: a CELL ID is not unique across lanes (the base lane's `main` object, the superseded batch-8 direct-fit
object, its dose bakes and the batch-64 re-fit all spell `constv2-mix-full__lora_r4_ce`) and a STAGED ID is not unique across
targets (`base__constv2-mix-full__lora_r4_ce` sits under every target) — so a row pinned to the re-fit but resolved through the
batch-8 bank entry would SERVE the old bytes while REPORTING the new pin. Every check here is keyed by the full tuple and refuses
by LANE and TARGET name; a known stale object is named by its label ("stale dose bake", "superseded direct-fit object")."""
from __future__ import annotations

import hashlib
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
for p in (REPO / "scripts" / "sdfbase" / "full22",):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
import itdoc_bind as B  # noqa: E402

LANE = "itdoc_eb64"
KEY, VIEW = "qwen35_4b", "it"
MIX, CTL, DIFF = "constv2-mix-full__lora_r4_ce", "constv2ctl2-full__lora_r4_ce", "constv2-mix-full__lora_r4_ce_minus_ctl"
#: the brief's VERBATIM qualification of the qwen pair (the record must carry exactly this text)
QUALIFICATION = ("the CONSTITUTION cell is BUDGET-LIMITED, NOT CONVERGED — best_epoch 16 of 16 with stopped_early False, so its best held-out CE "
                 "was the last epoch and the budget bound the result (a box-edge argmin, treated as a selection failure elsewhere in this campaign); "
                 "the CONTROL early-stopped at best_epoch 1 of 3; any difference of the pair inherits this asymmetry; nobody is re-fitting for it")
#: the VERBATIM cross-model pattern (on every row of both models)
PATTERN = ("constitution budget-limited, not converged (best_epoch 16/16, stopped_early False, box-edge argmin); control early-stopped "
           "(gemma 4/6; qwen 1/3); any difference object / dose ladder is a trained-to-budget numerator against a converged denominator; "
           "nobody re-fits. State it; do not smooth it.")


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _bank(root: Path, *, cell: str, staged_id: str, blob: bytes, manifest_sha: str | None = None, sidecar: str = "train_budget_meta.json",
          sidecar_sha: str | None = None, parents: dict | None = None, link_target: str | None = None, derived_id: str | None = None) -> Path:
    """A full22-shaped bank view: `<view>/VIEW_MANIFEST.json` + `<view>/<cell>` -> `../../cells/<staged id>/` (a symlink, as the
    real views are) holding object.pt + the sidecar of the kind."""
    view = root / "cells_view" / VIEW
    cells = root / "cells" / (link_target or staged_id)
    cells.mkdir(parents=True, exist_ok=True)
    (cells / "object.pt").write_bytes(blob)
    rec = {"object_sha256": _sha(blob) if sidecar_sha is None else sidecar_sha, "model_id": "synthetic", "model_revision": "0" * 40, "template_hash": "0" * 16}
    if sidecar == "derived_meta.json":
        rec.update({"kind": "lora_subtraction", "derived_id": derived_id or cell})
        for i, (pc, ps) in enumerate(parents.items() if parents else ()):
            rec[f"parent_{'pos' if i == 0 else 'neg'}"] = {"cell_id": pc, "object_sha256": ps}
    (cells / sidecar).write_text(json.dumps(rec))
    view.mkdir(parents=True, exist_ok=True)
    link = view / cell
    if not link.exists():
        os.symlink(Path("..") / ".." / "cells" / (link_target or staged_id), link)
    (view / "VIEW_MANIFEST.json").write_text(json.dumps({"target": KEY, "view": VIEW, "entries": [
        {"cell_id": cell, "staged_id": staged_id, "object_sha256": _sha(blob) if manifest_sha is None else manifest_sha}]}))
    return view


# --------------------------------------------------------------------------- the record
def test_the_committed_record_is_keyed_by_the_full_tuple_carries_the_qualification_verbatim_and_admits_nothing_today():
    adm = B.load_admission()
    assert adm["lane"] == LANE and adm["doc"]["schema"] == B.SCHEMA and adm["qualification"] == QUALIFICATION and adm["doc"]["qualification_pattern"] == PATTERN
    assert set(adm["entries"]) == {(LANE, k, VIEW, c) for k in ("qwen35_4b", "gemma4e4b") for c in (CTL, MIX, DIFF)}
    # 2026-09-13 ~21:50Z: the two qwen batch-64 objects the merged roster + pins at 4e200b68 serve through both clusters'
    # rebuilt `it` views are LAUNCHABLE (bank binding filled from that reading); since 2026-09-14 (ece9e8ce maps gemma's `it`
    # cells on the roster; both clusters' gemma `it` views rebound 01:16Z / 02:15Z) gemma's control + difference too — the hold lifted
    # THROUGH the record from a read of the bank (the GSM8K cells ran from it). The two batch-64 CONSTITUTION
    # entries stay HELD with their reason (the view serves the batch-8 object)
    servable = {(LANE, k, VIEW, c) for k in ("qwen35_4b", "gemma4e4b") for c in (CTL, DIFF)}
    for (lane, key, view, cell), e in adm["entries"].items():
        assert (e["lane"], e["model_key"], e["view"], e["cell"]) == (lane, key, view, cell)
        assert e["staged_id"] == B.STAGED_PREFIX_BY_KIND[e["object_kind"]] + cell and e["qualification_pattern"] == PATTERN
        if (lane, key, view, cell) in servable:
            assert B.entry_launchable(e) and e["hold_reason"] is None and B.bank_complete(e["bank"])
            # the `servable` block names what was read — roster, pin file, both clusters' view manifests; qwen's (21:49Z) closes with a
            # `verdict`, gemma's (08:11Z, the lift from disk) with the four-way `how`. The loader reads neither: the bank door re-checks at launch
            sv = e["servable"]
            assert {"roster", "pin_file", "view_manifest"} <= set(sv) and set(sv["view_manifest"]) == {"primary", "overflow"}, (key, cell)
            assert sv.get("verdict", "").startswith("SERVABLE") or "both clusters" in sv["how"], (key, cell)
            view_rel = f"exp/objects/sdfbase/full22/{key}/cells_view/it"
            assert e["bank"]["bank_view_rel"] == {"primary": view_rel, "overflow": view_rel}
            assert e["bank"]["bank_cell_dir"] == cell and e["bank"]["view"] == VIEW
        else:
            assert not B.entry_launchable(e) and e["hold_reason"] and e["bank"] is None, (key, cell)
        if e["object_sha256"] is not None:
            assert B.is_sha256(e["object_sha256"]) and e["qualification"]
        assert all(B.is_sha256(k) for k in e["stale_objects"]) and e["object_sha256"] not in e["stale_objects"]
    # the MATH-500 twins stay held whatever the bank says (the GSM8K wave alone is authorized) — the record's instrument hold
    assert B.instrument_hold(adm, "math500") and "GSM8K" in B.instrument_hold(adm, "math500") and B.instrument_hold(adm, "gsm8k") is None
    q = adm["entries"][(LANE, "qwen35_4b", VIEW, MIX)]
    assert q["qualification"] == QUALIFICATION and q["object_kind"] == B.OBJECT_KIND_FITTED and q["sidecar"] == "train_budget_meta.json"
    assert q["object_sha256"] == "80b75aae2cefee14b2885fcdf27fd889f49da7fa0cc3f97a501bc3f4d927358f"
    # the batch-64 CONSTITUTION object is served nowhere: the it view keeps the batch-8 object by the rule (2026-09-13; the
    # repoint withdrawn) — the hold names that, never a placeholder binding; the GSM8K wave rows the served batch-8 object instead
    assert q["hold_reason"].startswith("the it view serves the BATCH-8 constitution object") and "served nowhere" in q["hold_reason"]
    assert q["harvest"]["bank_copy"]["same_bytes"] is True and q["harvest"]["bank_copy"]["verified"]["overflow"]["object_sha256"] == q["object_sha256"]
    # link FOUR is per (model, cell): each model's control + difference servable (True — qwen since the 21:49Z reading, gemma since the
    # 2026-09-14 lift; before it `link4["gemma4e4b"]` was a bare False), each batch-64 constitution deliberately not served (the view
    # keeps the batch-8 object)
    link4 = adm["doc"]["link_status"]["links"]["4_servable_view_manifest_pins_roster"]
    for key in ("qwen35_4b", "gemma4e4b"):
        assert link4[key][CTL] is True and link4[key][DIFF] is True and link4[key][MIX].startswith("not served"), key
    assert q["superseded"]["staged_id"] == "itdoc__constv2-mix-full__lora_r4_ce" and q["superseded"]["object_sha256"].startswith("0018b32b")
    assert q["fit_record"]["best_epoch"] == q["fit_record"]["epochs_run"] == 16 and q["fit_record"]["stopped_early"] is False
    c = adm["entries"][(LANE, "qwen35_4b", VIEW, CTL)]
    assert c["object_sha256"].startswith("bace45e2") and c["fit_record"] == {**c["fit_record"], "best_epoch": 1, "epochs_run": 3, "stopped_early": True}
    d = adm["entries"][(LANE, "qwen35_4b", VIEW, DIFF)]
    assert d["object_kind"] == B.OBJECT_KIND_DERIVED and d["sidecar"] == "derived_meta.json" and d["staged_id"] == "derived_itdoc64__" + DIFF
    assert d["parents_expected"] == {MIX: q["object_sha256"], CTL: c["object_sha256"]} and d["object_sha256"].startswith("5e9b0009") and d["hold_reason"] is None
    g = adm["entries"][(LANE, "gemma4e4b", VIEW, MIX)]
    assert g["object_sha256"].startswith("78e84f27") and g["qualification"] == PATTERN and g["fit_record"]["best_epoch"] == 16 and g["fit_record"]["stopped_early"] is False
    gc = adm["entries"][(LANE, "gemma4e4b", VIEW, CTL)]
    assert gc["object_sha256"].startswith("04a90de8") and gc["fit_record"]["best_epoch"] == 4 and gc["fit_record"]["epochs_run"] == 6 and gc["fit_record"]["stopped_early"] is True
    gd = adm["entries"][(LANE, "gemma4e4b", VIEW, DIFF)]                 # built 2026-09-13T20:08Z, hashed on both clusters; held until 2026-09-14 (gemma's view / roster lacked it)
    assert gd["object_sha256"] == "8b7d10301a1fd04fbe4c71f21ae8912bae60ffc56df34acd10fa07a960b2080e" and gd["hold_reason"] is None
    # the lifted hold is kept, not erased: the previous reason rides `hold_history` with the date it lifted
    for e in (gc, gd):
        assert e["hold_history"] and e["hold_history"][0]["until"] == "2026-09-14" and "NOT SERVABLE" in e["hold_history"][0]["hold_reason"], e["cell"]
    assert gd["parents_expected"] == {MIX: g["object_sha256"], CTL: gc["object_sha256"]} and gd["harvest"]["derived_meta"]["parent_pos"]["object_sha256"] == g["object_sha256"]
    # the stale objects are the pin file's shas for the superseded parents and the dose bakes of THAT target (read at full length)
    pins = json.loads((REPO / "configs/sdfbase_full22_pins.json").read_text())
    for key in ("qwen35_4b", "gemma4e4b"):
        stale = adm["entries"][(LANE, key, VIEW, MIX)]["stale_objects"]
        for sid in ("itdoc__constv2-mix-full__lora_r4_ce", "itdoc__constv2-mix-full__psr_ce_all", "d0p1it__constv2-mix-full__lora_r4_ce",
                    "d0p25it__constv2-mix-full__lora_r4_ce", "d1p5it__constv2-mix-full__lora_r4_ce", "dm1it__constv2-mix-full__lora_r4_ce",
                    "halfdoseit__constv2-mix-full__lora_r4_ce"):
            assert pins[key][sid]["object_sha256"] in stale, (key, sid)
            assert (B.SUPERSEDED_OBJECT if sid.startswith("itdoc__") else B.STALE_DOSE_BAKE) in stale[pins[key][sid]["object_sha256"]]
    # the hazards are recorded: the derived-build sbatch, the dose cells' true-positive rule
    hz = json.dumps(adm["doc"]["hazards"])
    assert "sdfbase_derived_build.sbatch" in hz and "REAL finding" in hz and "never hash the grading host staging source" in hz


def test_the_loader_refuses_a_malformed_entry_by_name(tmp_path):
    doc = json.loads(B.ADMISSION_PATH.read_text())

    def load(mutate):
        d = json.loads(json.dumps(doc))
        mutate(d)
        p = tmp_path / "adm.json"
        p.write_text(json.dumps(d))
        return B.load_admission(p)
    q = lambda d: d["entries"]["qwen35_4b"]["it"]  # noqa: E731
    for mutate, why in (
        (lambda d: q(d)[MIX].__setitem__("model_key", "gemma4e4b"), "keyed by \\(lane, target, view, cell\\)"),
        (lambda d: q(d)[MIX].__setitem__("staged_id", "itdoc__" + MIX), "not this lane's spelling"),
        (lambda d: q(d)[MIX].__setitem__("staged_id", "itdoc64__" + CTL), "not this lane's spelling"),
        (lambda d: q(d)[DIFF].__setitem__("staged_id", "derived__" + DIFF), "not this lane's spelling"),
        (lambda d: q(d)[MIX].__setitem__("object_sha256", "deadbeef"), "never a placeholder"),
        (lambda d: q(d)[MIX].__setitem__("hold_reason", ""), "carries no hold_reason"),
        (lambda d: q(d)[MIX].__setitem__("qualification", None), "QUALIFICATION and the cross-model qualification PATTERN"),
        (lambda d: q(d)[MIX].__setitem__("qualification_pattern", ""), "QUALIFICATION and the cross-model qualification PATTERN"),
        (lambda d: q(d)[MIX].__setitem__("sidecar", "derived_meta.json"), "proven through"),
        (lambda d: q(d)[DIFF].__setitem__("parents_expected", {MIX: "0" * 64}), "EXACTLY its two parents"),
        (lambda d: q(d)[DIFF].__setitem__("parents_expected", {MIX: q(d)[MIX]["object_sha256"], CTL: "1" * 64}), "binds to THIS lane's parents"),
        (lambda d: q(d)[CTL].__setitem__("parents_expected", {}), "derived object's field"),
        (lambda d: q(d)[MIX].__setitem__("bank", {"view": "it"}), "neither null .* nor a complete binding"),
        (lambda d: q(d)[MIX].__setitem__("stale_objects", {"abc": "x"}), "64-hex sha: label"),
        (lambda d: q(d)[MIX]["stale_objects"].__setitem__(q(d)[MIX]["object_sha256"], "x"), "itself listed as a stale object"),
        (lambda d: d.__setitem__("schema", "steer_admission/1"), "schema"),
    ):
        with pytest.raises(SystemExit, match=why):
            load(mutate)
    # a launchable entry: pin + ruled staged id + complete bank + no hold reason (+ both qualification texts)
    def staged(d):
        e = q(d)[MIX]
        e["bank"] = {"view": "it", "bank_view_rel": {"primary": "exp/objects/sdfbase/full22/qwen35_4b/cells_view/it", "overflow": None}, "bank_cell_dir": MIX, "cluster_constraint": "any"}
        e["hold_reason"] = None
    adm = load(staged)
    # MIX flips to launchable under the synthetic binding; gemma's CONSTITUTION (bank null + a hold reason in the committed record; gemma's
    # control was the foil until its own hold lifted on 2026-09-14) does not
    assert B.entry_launchable(adm["entries"][(LANE, KEY, VIEW, MIX)]) and not B.entry_launchable(adm["entries"][(LANE, "gemma4e4b", VIEW, MIX)])


def test_the_pin_file_lookup_is_keyed_by_target_and_staged_id_never_by_staged_id_alone():
    """The Strength bug: a whole-file search by staged id returns ANOTHER target's pin first (`base__constv2-mix-full__lora_r4_ce` sits under
    every target) and manufactures a false refusal."""
    pins = json.loads((REPO / "configs/sdfbase_full22_pins.json").read_text())
    sid = "base__constv2-mix-full__lora_r4_ce"
    per_target = {k: B.pins_file_sha(pins, target=k, staged_id=sid) for k in pins}
    assert all(per_target.values()) and len(set(per_target.values())) == len(per_target)      # every target pins the id, all different
    assert per_target["qwen35_4b"] == pins["qwen35_4b"][sid]["object_sha256"] != per_target["gemma4e4b"]
    # 4e200b68 pinned the re-fit under (qwen35_4b, itdoc64__<cell>) — pinned is not served (the view keeps the batch-8 object);
    # the strength frontier's dose ladder later minted and pinned `dm1it64__<cell>`, so that id now READS its pin —
    # an id NOBODY minted (a dose token outside the ladder) still reads None, never a neighbour's pin
    assert B.pins_file_sha(pins, target="qwen35_4b", staged_id="itdoc64__" + MIX) == "80b75aae2cefee14b2885fcdf27fd889f49da7fa0cc3f97a501bc3f4d927358f"
    assert B.pins_file_sha(pins, target="qwen35_4b", staged_id="dm1it64__" + MIX) == pins["qwen35_4b"]["dm1it64__" + MIX]["object_sha256"]
    assert B.pins_file_sha(pins, target="qwen35_4b", staged_id="d0p75it64__" + MIX) is None
    assert B.pins_file_sha(pins, target="nosuch", staged_id=sid) is None
    # a synthetic duplicate under two targets resolves to the RIGHT one; a mismatch is reported with target + lane named
    view_ok = _bank(Path("/tmp/itdoc_bind_dup_ok"), cell=MIX, staged_id="itdoc64__" + MIX, blob=b"good")
    doc = {"qwen35_4b": {"itdoc64__" + MIX: {"object_sha256": _sha(b"good")}}, "gemma4e4b": {"itdoc64__" + MIX: {"object_sha256": _sha(b"other")}}}
    rec = B.bind_problems(view_ok, lane=LANE, model_key="qwen35_4b", view=VIEW, cell=MIX, pin=_sha(b"good"), object_kind="fitted",
                          staged_id="itdoc64__" + MIX, pins_doc=doc)
    assert rec["verdict"] == B.VERDICT_PASS and rec["hashes"]["pins_file"] == _sha(b"good"), rec
    rec = B.bind_problems(view_ok, lane=LANE, model_key="gemma4e4b", view=VIEW, cell=MIX, pin=_sha(b"good"), object_kind="fitted",
                          staged_id="itdoc64__" + MIX, pins_doc=doc)
    assert rec["verdict"] == B.VERDICT_REFUSED and any("target gemma4e4b" in p and "lane itdoc_eb64" in p and "pin file" in p for p in rec["problems"]), rec


# --------------------------------------------------------------------------- the binding
def test_the_binding_passes_on_a_consistent_bank_and_names_the_lane_on_every_refusal(tmp_path):
    good = b"the re-fit"
    pin = _sha(good)
    # PASS: manifest == disk == sidecar == pin, the ruled staged id, the symlink to it
    view = _bank(tmp_path / "ok", cell=MIX, staged_id="itdoc64__" + MIX, blob=good)
    rec = B.bind_problems(view, lane=LANE, model_key=KEY, view=VIEW, cell=MIX, pin=pin, object_kind="fitted", staged_id="itdoc64__" + MIX)
    assert rec["verdict"] == B.VERDICT_PASS and rec["problems"] == [] and rec["hashes"] == {"pin": pin, "manifest": pin, "disk": pin, "sidecar": pin, "pins_file": None}
    line = B.bind_verdict_line(rec)
    assert line.startswith("ITDOC_BIND: PASS (lane itdoc_eb64, target qwen35_4b, view it, cell constv2-mix-full__lora_r4_ce)") and "\n" not in line
    # the staging gap itself: the bank still names the batch-8 object under the batch-8 id — manifest == disk != pin, the OTHER lane named
    old = b"the superseded direct-fit object"
    stale = {_sha(old): f"{B.SUPERSEDED_OBJECT}: itdoc__{MIX}"}
    view = _bank(tmp_path / "gap", cell=MIX, staged_id="itdoc__" + MIX, blob=old)
    rec = B.bind_problems(view, lane=LANE, model_key=KEY, view=VIEW, cell=MIX, pin=pin, object_kind="fitted", staged_id="itdoc64__" + MIX, stale=stale)
    assert rec["verdict"] == B.VERDICT_REFUSED
    text = "\n".join(rec["problems"])
    assert "names staged id 'itdoc__constv2-mix-full__lora_r4_ce', the row binds 'itdoc64__constv2-mix-full__lora_r4_ce'" in text
    assert "that id is the batch-8 direct-fit lane (the superseded parent, D8)" in text and "cell dir resolves to staged id 'itdoc__" in text
    assert "self-consistent on ANOTHER object" in text and B.SUPERSEDED_OBJECT in text and "lane itdoc_eb64, target qwen35_4b" in text
    assert "would SERVE" in text and "while reporting the pin" in text
    # a dose bake of the superseded parent under a `*it__` id: named as a STALE DOSE BAKE (a real finding on a cluster, never paperwork)
    dose = b"a batch-8 dose bake"
    stale = {_sha(dose): f"{B.STALE_DOSE_BAKE}: d0p25it__{MIX} (qwen35_4b)"}
    view = _bank(tmp_path / "dose", cell=MIX, staged_id="d0p25it__" + MIX, blob=dose)
    rec = B.bind_problems(view, lane=LANE, model_key=KEY, view=VIEW, cell=MIX, pin=pin, object_kind="fitted", staged_id="itdoc64__" + MIX, stale=stale)
    text = "\n".join(rec["problems"])
    assert rec["verdict"] == B.VERDICT_REFUSED and B.STALE_DOSE_BAKE in text and "a batch-8 dose bake of the superseded parent" in text
    # a TORN bank: the manifest says the pin, the bytes are something else
    view = _bank(tmp_path / "torn", cell=MIX, staged_id="itdoc64__" + MIX, blob=b"drifted", manifest_sha=pin, sidecar_sha=pin)
    rec = B.bind_problems(view, lane=LANE, model_key=KEY, view=VIEW, cell=MIX, pin=pin, object_kind="fitted", staged_id="itdoc64__" + MIX)
    text = "\n".join(rec["problems"])
    assert rec["verdict"] == B.VERDICT_REFUSED and "TORN bank" in text and "stale object/sidecar pair" in text
    # the base lane's same-named cell under a `base__` id
    view = _bank(tmp_path / "base", cell=MIX, staged_id="base__" + MIX, blob=b"the base lane's object")
    rec = B.bind_problems(view, lane=LANE, model_key=KEY, view=VIEW, cell=MIX, pin=pin, object_kind="fitted", staged_id="itdoc64__" + MIX)
    assert rec["verdict"] == B.VERDICT_REFUSED and "the BASE lane" in "\n".join(rec["problems"])
    # no manifest at all (a raw fits store `cells_all/` is not a serving bank); no entry for the cell; no object
    raw = tmp_path / "raw" / "cells_all"
    (raw / MIX).mkdir(parents=True)
    (raw / MIX / "object.pt").write_bytes(good)
    rec = B.bind_problems(raw, lane=LANE, model_key=KEY, view=VIEW, cell=MIX, pin=pin, object_kind="fitted", staged_id="itdoc64__" + MIX)
    assert rec["verdict"] == B.VERDICT_REFUSED and "not a built bank view" in "\n".join(rec["problems"]) and "no train_budget_meta.json" in "\n".join(rec["problems"])
    view = _bank(tmp_path / "other", cell=CTL, staged_id="itdoc64__" + CTL, blob=good)
    rec = B.bind_problems(view, lane=LANE, model_key=KEY, view=VIEW, cell=MIX, pin=pin, object_kind="fitted", staged_id="itdoc64__" + MIX)
    assert rec["verdict"] == B.VERDICT_REFUSED and "names no entry for cell" in "\n".join(rec["problems"])
    # a held row (no pin) reads HELD without touching the disk
    rec = B.bind_problems(Path("/nonexistent"), lane=LANE, model_key=KEY, view=VIEW, cell=MIX, pin=None, object_kind="fitted")
    assert rec["verdict"] == B.VERDICT_HELD and "HELD" in rec["problems"][0]


def test_the_sidecar_is_read_by_kind_and_a_derived_object_binds_to_exactly_the_lanes_two_parents(tmp_path):
    pos, neg = _sha(b"parent pos"), _sha(b"parent neg")
    blob = b"the subtraction"
    pin = _sha(blob)
    sid = "derived_itdoc64__" + DIFF
    # PASS: derived_meta.json beside the object, derived_id == the cell, parents == the lane's two by sha
    view = _bank(tmp_path / "ok", cell=DIFF, staged_id=sid, blob=blob, sidecar="derived_meta.json", parents={MIX: pos, CTL: neg})
    rec = B.bind_problems(view, lane=LANE, model_key=KEY, view=VIEW, cell=DIFF, pin=pin, object_kind="derived", staged_id=sid, parents_expected={MIX: pos, CTL: neg})
    assert rec["verdict"] == B.VERDICT_PASS and rec["derived_parents"] == {MIX: pos, CTL: neg}, rec
    # the BASE lane's same-named derived object: its parents are the base parents — refused, the other lane named
    bpos, bneg = _sha(b"base pos"), _sha(b"base neg")
    view = _bank(tmp_path / "base", cell=DIFF, staged_id="derived__" + DIFF, blob=blob, sidecar="derived_meta.json", parents={MIX: bpos, CTL: bneg})
    rec = B.bind_problems(view, lane=LANE, model_key=KEY, view=VIEW, cell=DIFF, pin=pin, object_kind="derived", staged_id=sid, parents_expected={MIX: pos, CTL: neg},
                          stale={bpos: "a base-lane parent"})
    text = "\n".join(rec["problems"])
    assert rec["verdict"] == B.VERDICT_REFUSED and "are not exactly the lane's two it-doc parents" in text and "the BASE lane's derived object" in text and "a base-lane parent" in text
    # one parent right, one a stale bake of the superseded parent
    view = _bank(tmp_path / "half", cell=DIFF, staged_id=sid, blob=blob, sidecar="derived_meta.json", parents={MIX: pos, CTL: _sha(b"stale")})
    rec = B.bind_problems(view, lane=LANE, model_key=KEY, view=VIEW, cell=DIFF, pin=pin, object_kind="derived", staged_id=sid, parents_expected={MIX: pos, CTL: neg},
                          stale={_sha(b"stale"): f"{B.STALE_DOSE_BAKE}: x"})
    assert rec["verdict"] == B.VERDICT_REFUSED and B.STALE_DOSE_BAKE in "\n".join(rec["problems"])
    # a derived sidecar bound to another cell id
    view = _bank(tmp_path / "id", cell=DIFF, staged_id=sid, blob=blob, sidecar="derived_meta.json", parents={MIX: pos, CTL: neg}, derived_id=MIX)
    rec = B.bind_problems(view, lane=LANE, model_key=KEY, view=VIEW, cell=DIFF, pin=pin, object_kind="derived", staged_id=sid, parents_expected={MIX: pos, CTL: neg})
    assert rec["verdict"] == B.VERDICT_REFUSED and "records derived_id" in "\n".join(rec["problems"])
    # the WRONG sidecar for the kind: a derived object beside a train_budget_meta.json (and vice versa) refuses — a dir declares ONE identity
    view = _bank(tmp_path / "kind", cell=DIFF, staged_id=sid, blob=blob, sidecar="train_budget_meta.json")
    rec = B.bind_problems(view, lane=LANE, model_key=KEY, view=VIEW, cell=DIFF, pin=pin, object_kind="derived", staged_id=sid, parents_expected={MIX: pos, CTL: neg})
    text = "\n".join(rec["problems"])
    assert rec["verdict"] == B.VERDICT_REFUSED and "no derived_meta.json beside the object" in text and "carries ['train_budget_meta.json']" in text
    view = _bank(tmp_path / "kind2", cell=MIX, staged_id="itdoc64__" + MIX, blob=blob, sidecar="derived_meta.json", parents={MIX: pos, CTL: neg})
    rec = B.bind_problems(view, lane=LANE, model_key=KEY, view=VIEW, cell=MIX, pin=pin, object_kind="fitted", staged_id="itdoc64__" + MIX)
    assert rec["verdict"] == B.VERDICT_REFUSED and "no train_budget_meta.json beside the object" in "\n".join(rec["problems"])
    # a derived row whose record names no parents cannot bind
    view = _bank(tmp_path / "np", cell=DIFF, staged_id=sid, blob=blob, sidecar="derived_meta.json", parents={MIX: pos, CTL: neg})
    rec = B.bind_problems(view, lane=LANE, model_key=KEY, view=VIEW, cell=DIFF, pin=pin, object_kind="derived", staged_id=sid, parents_expected=None)
    assert rec["verdict"] == B.VERDICT_REFUSED and "record carries none" in "\n".join(rec["problems"])


def test_other_lane_of_names_the_batch_8_lanes_and_admits_this_lanes_ids():
    assert B.other_lane_of("itdoc64__" + MIX) is None and B.other_lane_of("derived_itdoc64__" + DIFF) is None
    assert B.other_lane_of("d0p25it64__" + MIX) is None and B.other_lane_of("dm1it64__" + MIX) is None            # the re-bakes are this lane's
    assert "batch-8 direct-fit lane" in B.other_lane_of("itdoc__" + MIX) and "derived object (itdoc__ parents)" in B.other_lane_of("derived_itdoc__" + DIFF)
    assert "stale dose bake" in B.other_lane_of("d0p25it__" + MIX) and "stale dose bake" in B.other_lane_of("halfdoseit__" + MIX)
    assert "BASE lane" in B.other_lane_of("base__" + MIX) and "BASE lane's derived" in B.other_lane_of("derived__" + DIFF)
    assert B.other_lane_of(None) is None and "unrecognised" in B.other_lane_of("candsi__A10__lora_r4")


def test_the_cli_bind_check_and_audit_read_the_record_and_report_pass_or_refusal_per_cell(tmp_path):
    """The launchers call `bind-check` at the bank door (exit 0 PASS / 3 REFUSED, the verdict line on stdout); `audit` runs every entry
    of the record against a shared root — PASS / HELD / REFUSED per cell, explicitly."""
    doc = json.loads(B.ADMISSION_PATH.read_text())
    # the synthetic bank below stages the MIX cell alone: every other entry of the copy is un-bound (bank null + a hold reason) so the
    # audit reads 1 PASS + 5 HELD over it, whatever the committed record binds today
    for key in ("qwen35_4b", "gemma4e4b"):
        for cell, other in doc["entries"][key]["it"].items():
            if (key, cell) != ("qwen35_4b", MIX):
                other["bank"] = None; other["hold_reason"] = other.get("hold_reason") or "synthetic hold: not staged in this test's bank"
    e = doc["entries"]["qwen35_4b"]["it"][MIX]
    blob = b"the staged re-fit"
    e["object_sha256"] = _sha(blob); e["hold_reason"] = None
    e["bank"] = {"view": "it", "bank_view_rel": {"primary": "exp/objects/sdfbase/full22/qwen35_4b", "overflow": None}, "bank_cell_dir": MIX, "cluster_constraint": "any"}
    for k in ("stale_objects",):
        e[k] = {kk: v for kk, v in e[k].items() if kk != _sha(blob)}
    adm_path = tmp_path / "adm.json"
    adm_path.write_text(json.dumps(doc))
    shared = tmp_path / "shared"
    root = shared / "exp/objects/sdfbase/full22/qwen35_4b"
    view = _bank(root, cell=MIX, staged_id="itdoc64__" + MIX, blob=blob)
    # the view dir the row names is `<shared>/<bank_view_rel>` = the model's bank root here, so the manifest must sit THERE
    (root / "VIEW_MANIFEST.json").write_text((view / "VIEW_MANIFEST.json").read_text())
    if not (root / MIX).exists():
        os.symlink(Path("cells") / ("itdoc64__" + MIX), root / MIX)
    pins = tmp_path / "pins.json"
    pins.write_text(json.dumps({"qwen35_4b": {"itdoc64__" + MIX: {"object_sha256": _sha(blob)}}}))
    cli = [sys.executable, str(REPO / "scripts/sdfbase/full22/itdoc_bind.py")]
    r = subprocess.run(cli + ["bind-check", "--admission", str(adm_path), "--lane", LANE, "--key", KEY, "--view", VIEW, "--cell", MIX,
                              "--view-dir", str(root), "--pin", _sha(blob), "--pins", str(pins)], capture_output=True, text=True)
    assert r.returncode == 0 and r.stdout.startswith("ITDOC_BIND: PASS (lane itdoc_eb64, target qwen35_4b, view it, cell constv2-mix-full__lora_r4_ce)"), r.stdout + r.stderr
    # the row's pin must be the record's; an unknown tuple refuses; a bank naming the batch-8 id refuses by lane name (exit 3)
    r = subprocess.run(cli + ["bind-check", "--admission", str(adm_path), "--lane", LANE, "--key", KEY, "--view", VIEW, "--cell", MIX,
                              "--view-dir", str(root), "--pin", "f" * 64], capture_output=True, text=True)
    assert r.returncode == 3 and "the row's pin ffffffffffff… != the record's" in r.stdout
    r = subprocess.run(cli + ["bind-check", "--admission", str(adm_path), "--lane", "full22", "--key", KEY, "--view", VIEW, "--cell", MIX,
                              "--view-dir", str(root), "--pin", _sha(blob)], capture_output=True, text=True)
    assert r.returncode == 3 and "no entry in adm.json" in r.stdout
    bad = _bank(tmp_path / "bad", cell=MIX, staged_id="itdoc__" + MIX, blob=b"old bytes")
    r = subprocess.run(cli + ["bind-check", "--admission", str(adm_path), "--lane", LANE, "--key", KEY, "--view", VIEW, "--cell", MIX,
                              "--view-dir", str(bad), "--pin", _sha(blob)], capture_output=True, text=True)
    assert r.returncode == 3 and "ITDOC_BIND: REFUSED" in r.stdout and "the batch-8 direct-fit lane" in r.stdout and "lane itdoc_eb64, target qwen35_4b" in r.stdout
    # audit: one PASS (the staged row), the rest HELD with their reasons; exit = the worst verdict (1 = held)
    r = subprocess.run(cli + ["audit", "--admission", str(adm_path), "--shared-root", str(shared), "--cluster-role", "primary", "--pins", str(pins)],
                       capture_output=True, text=True)
    lines = [l for l in r.stdout.splitlines() if l.startswith("ITDOC_BIND:")]
    assert r.returncode == 1 and len(lines) == 6 and sum(l.startswith("ITDOC_BIND: PASS") for l in lines) == 1 and sum(l.startswith("ITDOC_BIND: HELD") for l in lines) == 5
    # each HELD line carries the entry's OWN reason: gemma's batch-64 constitution keeps its committed hold (the view serves the batch-8
    # object), the four un-bound control / difference entries the synthetic one — their committed hold_reason is None since the qwen
    # (2026-09-13) and gemma (2026-09-14) lifts; until the gemma lift two of them leaked the record's "NOT SERVABLE" text into this audit
    assert r.stdout.count("\n  - HELD: ") == 5 and r.stdout.count("  - HELD: synthetic hold: not staged in this test's bank") == 4
    assert r.stdout.count("  - HELD: the it view serves the BATCH-8 constitution object") == 1
    # the shared bank door (st_itdoc_bank_check): the python check, then the byte check; a refusal is FATAL with the lane named
    core = REPO / "scripts/sdfbase/full22/served_thinking.sh"

    def door(view_dir_rel: str, shared_root: Path, pin: str, record: Path):
        # ST_ITDOC_PINS: the door's test seam — the synthetic bank's pin file stands in for configs/sdfbase_full22_pins.json (which
        # names no itdoc64__ id until the lane pins the servable objects; production never sets the variable)
        return subprocess.run(["bash", "-c",
                               f'set -u; SHARED={shlex.quote(str(shared_root))}; REPO_DIR={shlex.quote(str(REPO))}; CPY={shlex.quote(sys.executable)}; ST_ITDOC_PINS={shlex.quote(str(pins))}; '
                               'f22_fatal(){ echo "FATAL: $*" >&2; exit 9; }; f22_log(){ echo "LOG $*"; }; '
                               f'source {shlex.quote(str(core))}; st_itdoc_bank_check {shlex.quote(view_dir_rel)} {shlex.quote(MIX)} {shlex.quote(pin)} '
                               f'{shlex.quote(str(record))} {LANE} {KEY} {VIEW} {MIX}; echo DIR=$TT_TWO_SI_DEFENSES_DIR'],
                              capture_output=True, text=True)
    ok = door("exp/objects/sdfbase/full22/qwen35_4b", shared, _sha(blob), adm_path)
    assert ok.returncode == 0 and "LOG ITDOC_BIND: PASS" in ok.stdout and "external bank view OK" in ok.stdout and f"DIR={root}" in ok.stdout, ok.stdout + ok.stderr
    (tmp_path / "shared2/exp/objects/sdfbase/full22").mkdir(parents=True)
    os.symlink(bad.parent.parent, tmp_path / "shared2/exp/objects/sdfbase/full22/qwen35_4b")
    refused = door("exp/objects/sdfbase/full22/qwen35_4b/cells_view/it", tmp_path / "shared2", _sha(blob), adm_path)
    assert refused.returncode == 9 and "it-doc bank binding REFUSED for (lane itdoc_eb64, target qwen35_4b, view it, cell constv2-mix-full__lora_r4_ce)" in refused.stderr
    assert "the batch-8 direct-fit lane" in refused.stderr and "external bank view OK" not in refused.stdout
    # the ONE caller of the byte check inside the it-doc door; both launchers dispatch to the door on object_source itdoc
    src = core.read_text()
    door_body = src[src.index("st_itdoc_bank_check(){"):src.index("st_check_endpoint(){")]
    assert door_body.count("st_external_bank_view_check ") == 1 and "itdoc_bind.py bind-check" in door_body
    for launcher in ("scripts/sdfbase/gsm8k/g8k_common.sh", "scripts/sdfbase/math500/m5t_common.sh"):
        body = (REPO / launcher).read_text()
        assert body.count("st_itdoc_bank_check ") == 1 and "itdoc)" in body
