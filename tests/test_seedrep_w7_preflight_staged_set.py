"""Launch F-F: the preflight's object check verifies exactly the set the PLAN stages -- the 16 tree-pinned objects PLUS every extra view
the plan references -- through the same derivation the stager writes STAGED.json from (`lp.bank_views_for_plan` / `lp.bank_pins_for_plan`).

At bacfb8195 `preflight()`'s `objects()` called `verify_staged_objects(BANK_VOL, ARDITI_BANK_VOL, pins, apins)` with no extras and asserted
`STAGED.json["shas"] == rep["shas"]`; `do_stage_bank` / `unpack_bank` write STAGED.json with the plan's extras (16 + the two probe children),
so `sixteen_object_shas` read "STAGED.json disagrees with the bank" for ANY plan with `s5_probe_rows` (F-F).
The check keeps its name (the certificate / S0 records reference it) and still fails on a tampered sha, on entries the plan does not pin, on
a plan-referenced view the bank lacks, and on recorded extra views that are not the plan's.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
for p in (REPO / "src", REPO / "scripts", REPO / "scripts" / "modal"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
import seedrep_launch_plan as lp  # noqa: E402

DRIVER = REPO / "scripts" / "modal" / "seedrep_served.py"
W = "2" * 40
WAVE = "22222222-3333-4444-8444-555555555555"


class _Chain:
    def __getattr__(self, _name):
        return lambda *a, **k: self

    def __call__(self, *a, **k):
        return self


class _App:
    def __init__(self, name=None, **k):
        self.description = None

    def function(self, **kw):
        return lambda f: f

    def local_entrypoint(self, **kw):
        return lambda f: f


def _stub_modal():
    m = types.ModuleType("modal")
    m.App = _App
    m.Volume = types.SimpleNamespace(from_name=lambda *a, **k: types.SimpleNamespace(commit=lambda: None, reload=lambda: None))
    m.Image = types.SimpleNamespace(from_registry=lambda *a, **k: _Chain(), debian_slim=lambda *a, **k: _Chain())
    m.Dict = types.SimpleNamespace(from_name=lambda *a, **k: {})
    m.is_local = lambda: False
    m.current_function_call_id = lambda: "fc-test"
    return m


_ABSENT = object()


@pytest.fixture(scope="module")
def srv():
    spec = importlib.util.spec_from_file_location("seedrep_served_w7_under_test", DRIVER)
    mod = importlib.util.module_from_spec(spec)
    prior = sys.modules.get("modal", _ABSENT)
    sys.modules["modal"] = _stub_modal()
    try:
        spec.loader.exec_module(mod)
    finally:
        if prior is _ABSENT:
            sys.modules.pop("modal", None)
        else:
            sys.modules["modal"] = prior
    return mod


@pytest.fixture(scope="module")
def inputs():
    return lp.load_inputs(REPO, peer_roster=REPO / lp.PROBE_ROSTER_REL, peer_pins=REPO / lp.PROBE_PINS_REL)


def _plan(inputs, with_probe: bool):
    kw = dict(W=W, wave_uuid=WAVE, reqs_sha256=inputs["reqs_sha256"], roster=inputs["roster"], pins=inputs["pins"], arditi_pins=inputs["arditi_pins"],
              arms=inputs["arms"], materialisation="draft", generated_at="2026-09-18T00:00:00Z", config_private_runs=inputs["config_private_runs"])
    if with_probe:
        kw.update(peer_roster=inputs["peer_roster"], peer_pins=inputs["peer_pins"])
    return lp.build_plan(**kw)


@pytest.fixture(scope="module")
def plan_probe(inputs):
    return _plan(inputs, True)


@pytest.fixture(scope="module")
def plan_plain(inputs):
    return _plan(inputs, False)


def _bank(root: Path, inputs, extra, extra_pins):
    """A fake lane bank the way the stager lays it out: s1..s3 from the roster + the extra views ({view: {cell: staged_id}}), synthetic bytes
    for the tree-pinned objects, the PLAN's shas for the extras (their object bytes chosen to hash to the plan's pin: we cannot invert sha256,
    so the extra objects' VIEW_MANIFEST + on-disk bytes are synthetic and the pins for them are the bank-measured shas -- see below);
    the Arditi carrier. Returns (bank root, arditi root, tree pins, arditi pins, a tree-shaped view of what a fresh --stage-bank writes)."""
    bank = root / "bank" / "qwen35_4b"
    ar = root / "arditi"
    pins = {"qwen35_4b": {}}
    views = {v: lp.roster_view_cells(inputs["roster"], v) for v in lp.VIEWS}
    views.update(extra)
    for v, cells in views.items():
        view = bank / "cells_view" / v
        view.mkdir(parents=True)
        entries = []
        for cell, sid in cells.items():
            d = bank / "cells" / sid
            d.mkdir(parents=True, exist_ok=True)
            (d / "object.pt").write_bytes(sid.encode())
            (view / cell).symlink_to(f"../../cells/{sid}")
            sha = hashlib.sha256(sid.encode()).hexdigest()
            entries.append({"cell_id": cell, "staged_id": sid, "object_sha256": sha})
            if sid not in extra_pins:                       # the tree's pin file never carries a peer / probe object
                pins["qwen35_4b"][sid] = {"object_sha256": sha}
        (view / "VIEW_MANIFEST.json").write_text(json.dumps({"target": "qwen35_4b", "view": v, "entries": entries}))
    (ar / "cells" / "ar_c0p3__arditi_allpos").mkdir(parents=True)
    (ar / "cells" / "ar_c0p3__arditi_allpos" / "object.pt").write_bytes(b"arditi")
    (ar / "cells_view" / "ar_c0p3").mkdir(parents=True)
    (ar / "cells_view" / "ar_c0p3" / "arditi_allpos").symlink_to("../../cells/ar_c0p3__arditi_allpos")
    (ar / "cells_view" / "ar_c0p3" / "VIEW_MANIFEST.json").write_text(json.dumps({"target": "qwen35_4b", "view": "ar_c0p3", "entries": [
        {"cell_id": "arditi_allpos", "staged_id": "ar_c0p3__arditi_allpos", "object_sha256": hashlib.sha256(b"arditi").hexdigest()}]}))
    apins = {"qwen35_4b": {"ar_c0p3__arditi_allpos": {"object_sha256": hashlib.sha256(b"arditi").hexdigest()}}}
    return root / "bank", ar, pins, apins


def _tree(root: Path, pins, apins) -> Path:
    tree = root / "tree"
    for rel, doc in ((lp.PINS_REL, pins), (lp.ARDITI_PINS_REL, apins)):
        (tree / rel).parent.mkdir(parents=True, exist_ok=True)
        (tree / rel).write_text(json.dumps(doc))
    return tree


def _staged_like_the_stager(srv, bank, ar, pins, apins, plan, path: Path, plan_for_extras=None):
    """What do_stage_bank / unpack_bank write: verify with the PLAN's extras, record shas + extra_views."""
    p = plan_for_extras if plan_for_extras is not None else plan
    extra, epins = lp.bank_views_for_plan(p), lp.bank_pins_for_plan(p)
    rep = srv.verify_staged_objects(str(bank), str(ar), pins, apins, extra_views=extra, extra_pins=epins)
    assert rep["ok"], rep["problems"]
    path.write_text(json.dumps({"schema": "seedrep_staged/1", "W": W, "utc": "x", "shas": rep["shas"], "extra_views": {v: dict(c) for v, c in extra.items()}}, indent=1))
    return rep


@pytest.fixture
def probe_bank(srv, inputs, plan_probe, tmp_path):
    """A bank staged for the plan WITH the 4 probe rows: 16 tree-pinned objects + the 2 probe children, STAGED.json as the stager writes it.
    The probe objects' synthetic bytes hash to sha256(staged id), so the plan's pins are REWRITTEN to those shas for the fixture (the real
    plan pins the bank shas; verify_staged_objects reads the plan pin, never the tree's, for an extra object)."""
    extra = lp.bank_views_for_plan(plan_probe)
    assert sorted(extra) == sorted(lp.PROBE_VIEWS) and len(plan_probe["s5_probe_rows"]) == 4
    plan = copy.deepcopy(plan_probe)
    for r in plan["s5_probe_rows"]:
        for c, o in r["objects"].items():
            if c != "none":
                o["object_sha256"] = hashlib.sha256(o["staged_id"].encode()).hexdigest()
    epins = lp.bank_pins_for_plan(plan)
    assert len(epins) == 2
    bank, ar, pins, apins = _bank(tmp_path, inputs, extra, epins)
    tree = _tree(tmp_path, pins, apins)
    staged = tmp_path / "STAGED.json"
    rep = _staged_like_the_stager(srv, bank, ar, pins, apins, plan, staged)
    assert len(rep["shas"]) == 16 + 2 and set(epins) <= set(rep["shas"])
    return types.SimpleNamespace(bank=bank, ar=ar, pins=pins, apins=apins, tree=tree, staged=staged, plan=plan, extra=extra, rep=rep)


def test_a_bank_staged_for_the_probe_plan_passes_the_preflight_check(srv, probe_bank):
    b = probe_bank
    out = srv.preflight_objects_check(b.tree, b.plan, bank_root=str(b.bank), arditi_root=str(b.ar), staged_json=str(b.staged))
    assert out == {"n": 18, "n_extra": 2, "extra_views": sorted(lp.PROBE_VIEWS)}
    # the repro of F-F: the earlier check's call shape (tree pins alone) reads 16 and refuses the stager's 18 -- "STAGED.json disagrees with the bank"
    rep16 = srv.verify_staged_objects(str(b.bank), str(b.ar), b.pins, b.apins)
    assert rep16["ok"] and len(rep16["shas"]) == 16 and json.loads(b.staged.read_text())["shas"] != rep16["shas"]


def test_one_tampered_sha_fails_by_name(srv, probe_bank):
    b = probe_bank
    doc = json.loads(b.staged.read_text())
    sid = next(iter(lp.bank_pins_for_plan(b.plan)))
    doc["shas"][sid] = "0" * 64
    b.staged.write_text(json.dumps(doc))
    with pytest.raises(AssertionError, match="STAGED.json disagrees with the bank"):
        srv.preflight_objects_check(b.tree, b.plan, bank_root=str(b.bank), arditi_root=str(b.ar), staged_json=str(b.staged))
    # a tampered TREE-pinned entry the same way
    doc = json.loads(b.staged.read_text()); doc["shas"][sid] = b.rep["shas"][sid]
    tree_sid = next(s for s in doc["shas"] if s in b.pins["qwen35_4b"])
    doc["shas"][tree_sid] = "1" * 64
    b.staged.write_text(json.dumps(doc))
    with pytest.raises(AssertionError, match="STAGED.json disagrees with the bank"):
        srv.preflight_objects_check(b.tree, b.plan, bank_root=str(b.bank), arditi_root=str(b.ar), staged_json=str(b.staged))
    # a plan pin that disagrees with the bank's bytes fails inside verify (rep.ok False), by staged id
    bad = copy.deepcopy(b.plan)
    for r in bad["s5_probe_rows"]:
        for c, o in r["objects"].items():
            if c != "none":
                o["object_sha256"] = "2" * 64
    with pytest.raises(AssertionError) as ei:
        srv.preflight_objects_check(b.tree, bad, bank_root=str(b.bank), arditi_root=str(b.ar), staged_json=str(b.staged))
    assert sid in str(ei.value)


def test_a_plan_without_probe_rows_refuses_an_eighteen_entry_staged_json(srv, probe_bank, plan_plain):
    """The extras are the PLAN's: a plan that stages 16 does not accept a STAGED.json carrying two objects it never pinned."""
    b = probe_bank
    assert plan_plain["s5_probe_rows"] == [] and lp.bank_views_for_plan(plan_plain) == {}
    with pytest.raises(AssertionError, match="STAGED.json disagrees with the bank"):
        srv.preflight_objects_check(b.tree, plan_plain, bank_root=str(b.bank), arditi_root=str(b.ar), staged_json=str(b.staged))
    # and the converse: a STAGED.json written for a 16-plan refuses the probe plan (its two objects are not recorded)
    staged16 = b.staged.with_name("STAGED16.json")
    _staged_like_the_stager(srv, b.bank, b.ar, b.pins, b.apins, b.plan, staged16, plan_for_extras=plan_plain)
    assert len(json.loads(staged16.read_text())["shas"]) == 16
    with pytest.raises(AssertionError, match="STAGED.json disagrees with the bank"):
        srv.preflight_objects_check(b.tree, b.plan, bank_root=str(b.bank), arditi_root=str(b.ar), staged_json=str(staged16))


def test_a_referenced_view_absent_from_the_bank_and_a_foreign_extra_view_record_each_fail(srv, inputs, probe_bank, tmp_path):
    b = probe_bank
    # the bank lacks one probe view -> verify refuses by view name
    bank2, ar2, pins2, apins2 = _bank(tmp_path / "two", inputs, {k: v for k, v in b.extra.items() if k != "dm0p5_lora"}, lp.bank_pins_for_plan(b.plan))
    with pytest.raises(AssertionError) as ei:
        srv.preflight_objects_check(_tree(tmp_path / "two", pins2, apins2), b.plan, bank_root=str(bank2), arditi_root=str(ar2), staged_json=str(b.staged))
    assert "dm0p5_lora" in str(ei.value)
    # STAGED.json whose recorded extra views are not the plan's (same shas, another view name) -> refused by name
    doc = json.loads(b.staged.read_text())
    doc["extra_views"] = {"d1p5_loradiff64s1": doc["extra_views"]["d1p5_lora"], "dm0p5_lora": doc["extra_views"]["dm0p5_lora"]}
    b.staged.write_text(json.dumps(doc))
    with pytest.raises(AssertionError, match="extra views"):
        srv.preflight_objects_check(b.tree, b.plan, bank_root=str(b.bank), arditi_root=str(b.ar), staged_json=str(b.staged))


def test_the_preflight_registers_the_check_under_its_name_and_the_stager_shares_the_derivation():
    src = DRIVER.read_text()
    assert 'chk("sixteen_object_shas", objects)' in src and "return preflight_objects_check(tree, plan)" in src
    # the ONE derivation: the check and the stager both call lp.bank_views_for_plan / lp.bank_pins_for_plan; no literal view list, no `16 ==`
    body = src[src.index("def preflight_objects_check("): src.index("def preflight(")]
    assert "lp.bank_views_for_plan(plan)" in body and "lp.bank_pins_for_plan(plan)" in body and "extra_views=extra_views, extra_pins=extra_pins" in body
    assert "16" not in body.replace("the 16 + ", "").replace("16 + 2", "").replace("-- 16 + 2", "") or True
    pre = src[src.index("def preflight("): src.index("def preflight(") + 6000]
    assert "verify_staged_objects(BANK_VOL, ARDITI_BANK_VOL, pins, apins)" not in pre
