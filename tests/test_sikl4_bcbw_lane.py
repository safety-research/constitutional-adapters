"""The SI-distilled refit views on the BigCodeBench / BFCL full suites (added 2026-11-15): the three lane rows
``qwen35_4b-f22sikl4s<k>``, their generated configs, the full roster's three steered-only waves, the candidate-SI subset pair every
container of such a row carries, the store suffixes the lane's registry adapter admits, and the generalised lane bank stager.

What is pinned, and why:

- the 18 objects: every row's config cells and pins equal ``configs/sikl4_pins.json``; each row's intervention-hash inventory names
  exactly its six cells; no wave runs (or creates) an unsteered ``none``;
- the configs embed the sikl4 roster and pins of the frozen wave branch (sha256 pinned below), so a roster edit that would put the
  Modal and the cluster runs on different rosters fails here first;
- the subset pair is ROW-keyed: present on exactly the rows whose generated config names a C16 cell, absent on every other row, and
  carried by every registry-building child of such a row (the serving environment and the preflight's identity derivation), because the
  explicit-cells registry refuses the whole config when one listed SI does not resolve;
- the launch tags: the wave suffix and an ``r<N>`` relaunch match the lane adapter's grammar, a ``smoke`` / ``smoke2`` suffix does not,
  and no lane wave name is a paper seed tag;
- the stager's lane table is ``contracts.bcbw_view_sources``'s, and its refusals hold over a SYNTHETIC bank (fake bytes, no Modal).
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from sisteer.capeval.tinycap import contracts as C
from sisteer.capeval.tinycap import widen_roster as W

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_tinycap_launcher as TL  # noqa: E402  (the modal stub and the launcher loader)

VIEWS = C.BCBW_SIKL4_VIEWS
SLUGS = C.BCBW_SIKL4_SLUGS
CELLS6 = ("safety_refusal__lora_r4", "safety_refusal__psr_kl_all", "si_constitution__lora_r4", "si_constitution__psr_kl_all",
          "C16__lora_r4", "C16__psr_kl_all")
#: sha256 of configs/sdfbase_full22_roster_sikl4.yaml and configs/sikl4_pins.json at the frozen wave commit
#: 3ca84bce34e1c4ae58a48a866b85387a5cdf1725: the Modal rows must run on the same roster as the cluster runs of the same objects
M_W_ROSTER_SHA256 = "e7cceb8466bd83382b02c40b1c4120db0ee8ca19154c36c6d4131e7a0113eae9"
M_W_PINS_SHA256 = "a3ad85dc0dd1bdd4bbbfb7a1420dab3878dac42e4e33def3f47954a0453baf36"
#: the lane adapter's store-suffix grammar (the approved design note's lane branch of TaskSpec.suffix_re, D-A6)
LANE_SUFFIX_RE = re.compile(r"^bcbw-(?P<launch_tag>qwen35_4b_sikl4s[123])-(?P<sha7>[0-9a-f]{7})(?P<relaunch>-r\d+)?(?P<task_tag>-(?:bcb|bfcl))?$")
SHA = "0123456789abcdef0123456789abcdef01234567"
PINS = json.loads((REPO / C.BCBW_SIKL4_PINS_REL).read_text())["qwen35_4b"]
FULL = W.load_roster(W.ROSTER_FULL_PATH)


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def L():
    return TL._load_launcher(TL._stub_modal(), app_name=TL.TEST_APP_NAME, name="tinycap_served_sikl4_under_test")


@pytest.fixture(scope="module")
def Wr():
    return _load("bcb_bfcl_widen_launch_sikl4_under_test", "scripts/bcb_bfcl_200_widen_launch.py")


@pytest.fixture(scope="module")
def ST():
    return _load("stage_lane_bank_sikl4_under_test", "scripts/modal/stage_lane_bank_modal.py")


# --------------------------------------------------------------------------------------------------------------------------- #
# the 18 objects                                                                                                                #
# --------------------------------------------------------------------------------------------------------------------------- #

def test_the_three_rows_serve_exactly_the_eighteen_pinned_objects_and_no_none():
    assert len(PINS) == 18 and SLUGS == tuple(f"qwen35_4b-f22{v}" for v in VIEWS)
    for k, (view, slug) in enumerate(zip(VIEWS, SLUGS), start=1):
        row = C.model_row(slug)
        assert row.objects_dir == C.full22_objects_dir("qwen35_4b", view) and C.is_full22_row(row)
        doc = yaml.safe_load((REPO / row.cfg).read_text())
        assert doc["explicit_cells"] == [C.ROUTE_NONE_CELL, C.CELL_PROMPTED_SAFETY, *CELLS6]   # `none` is the registry's anchor, never launched
        assert doc["defenses_dir"] == row.objects_dir
        assert set(C.EXPECTED_INTERVENTION_HASHES[slug]) == set(CELLS6)
        assert all(re.fullmatch(r"[0-9a-f]{16}", h) for h in C.EXPECTED_INTERVENTION_HASHES[slug].values())
        wave = f"qwen35_4b_{view}"
        assert W.wave(FULL, wave)["model_row"] == slug and W.runnable_cells(FULL, wave) == list(CELLS6)
        assert C.ROUTE_NONE_CELL not in {e["cell_id"] for e in W.entries(FULL, wave).values()}          # no unsteered run is created
        for e in W.entries(FULL, wave).values():
            pin = PINS[f"{view}__{e['cell_id']}"]
            assert e["object"]["sha256"] == pin["object_sha256"] and e["object"]["staged_id"] == pin["staged_id"]
            assert e["object"]["pins"] == C.BCBW_SIKL4_PINS_REL
    # 18 distinct objects, and no two rows share a hash (each object is its own intervention)
    hashes = [h for s in SLUGS for h in C.EXPECTED_INTERVENTION_HASHES[s].values()]
    assert len(set(hashes)) == 18 and len({p["object_sha256"] for p in PINS.values()}) == 18


def test_the_generated_configs_embed_the_frozen_wave_branchs_roster_and_pins():
    assert sha256(REPO / C.BCBW_SIKL4_ROSTER_REL) == M_W_ROSTER_SHA256
    assert sha256(REPO / C.BCBW_SIKL4_PINS_REL) == M_W_PINS_SHA256
    for view in VIEWS:
        head = (REPO / C.bcbw_config_path("qwen35_4b", view)).read_text().split("# -- the explicit cell enumeration", 1)[0]
        assert f"#   roster  {C.BCBW_SIKL4_ROSTER_REL}  {M_W_ROSTER_SHA256}\n" in head, view
        assert f"#   pins    {C.BCBW_SIKL4_PINS_REL}  {M_W_PINS_SHA256}\n" in head, view


# --------------------------------------------------------------------------------------------------------------------------- #
# the candidate-SI subset pair: row-keyed                                                                                       #
# --------------------------------------------------------------------------------------------------------------------------- #

def _config_names_a_c16_cell(row) -> bool:
    cfg = REPO / str(row.cfg)
    if not cfg.is_file():
        return False
    doc = yaml.safe_load(cfg.read_text()) or {}
    for cell in doc.get("explicit_cells") or ():
        try:
            parsed = C.parse_launcher_cell(str(cell))
        except Exception:  # noqa: BLE001 — a config cell the launcher grammar does not parse names no SI here
            continue
        if parsed.si == "C16":
            return True
    return False


def test_the_subset_pair_rides_exactly_the_rows_whose_config_names_a_c16_cell():
    from sisteer.data import candidate_sis as CS
    assert C.CANDIDATE_SIS_SUBSET_SHA_ENV == CS.SUBSET_SHA_ENV and C.CANDIDATE_SIS_LOAD_ENV == "SISTEER_LOAD_CANDIDATE_SIS"
    assert re.fullmatch(r"[0-9a-f]{64}", C.CANDIDATE_SIS_SUBSET_C16_SHA256)
    # content-addressed on the volume, OUTSIDE every staged bank view (the stager holds a view's listing exact)
    assert C.CANDIDATE_SIS_SUBSET_C16_REMOTE.startswith(C.CANDIDATE_SIS_SUBSET_ROOT + "/")
    assert C.CANDIDATE_SIS_SUBSET_C16_SHA256[:16] in C.CANDIDATE_SIS_SUBSET_C16_REMOTE
    assert not C.CANDIDATE_SIS_SUBSET_C16_REMOTE.startswith(C.VOL_OBJECTS_FULL22 + "/")
    named = {s for s in C.MODEL_SLUGS if _config_names_a_c16_cell(C.model_row(s))}
    assert named == set(SLUGS) == set(C.CANDIDATE_SIS_SUBSET_ENV_BY_ROW)
    for slug in C.MODEL_SLUGS:
        env = C.candidate_sis_env(slug)
        if slug in named:
            assert env == {"SISTEER_LOAD_CANDIDATE_SIS": C.CANDIDATE_SIS_SUBSET_C16_REMOTE,
                           "SISTEER_CANDIDATE_SIS_SUBSET_SHA256": C.CANDIDATE_SIS_SUBSET_C16_SHA256}, slug
        else:
            assert env == {}, slug
    env = C.candidate_sis_env(SLUGS[0])
    env["SISTEER_LOAD_CANDIDATE_SIS"] = "elsewhere"
    assert C.candidate_sis_env(SLUGS[0])["SISTEER_LOAD_CANDIDATE_SIS"] == C.CANDIDATE_SIS_SUBSET_C16_REMOTE    # a copy, never the table


def test_every_registry_building_child_of_a_sikl4_row_carries_the_pair_and_no_other_row_does(L, monkeypatch):
    monkeypatch.delenv("SISTEER_LOAD_CANDIDATE_SIS", raising=False)
    monkeypatch.delenv("SISTEER_CANDIDATE_SIS_SUBSET_SHA256", raising=False)
    # the serving environment (each standup, driver and verifier child): the row slug is a REQUIRED keyword
    with pytest.raises(TypeError):
        L._serving_env("/staged/tree", C.REDUCTION_MODE_LAUNCHER_DEFAULT)                     # noqa — no caller may drop the row
    for slug in SLUGS:
        env = L._serving_env("/staged/tree", C.REDUCTION_MODE_LAUNCHER_DEFAULT, row_slug=slug)
        assert env["SISTEER_LOAD_CANDIDATE_SIS"] == C.CANDIDATE_SIS_SUBSET_C16_REMOTE
        assert env["SISTEER_CANDIDATE_SIS_SUBSET_SHA256"] == C.CANDIDATE_SIS_SUBSET_C16_SHA256
    for slug in ("qwen35_4b-f22s1", "qwen35_4b-f22ctlstop64s1", "qwen35_4b-f22it", C.MODEL_SLUG):
        env = L._serving_env("/staged/tree", C.REDUCTION_MODE_LAUNCHER_DEFAULT, row_slug=slug)
        assert "SISTEER_LOAD_CANDIDATE_SIS" not in env and "SISTEER_CANDIDATE_SIS_SUBSET_SHA256" not in env, slug
    # ... and the preflight's object-identity derivation (the same explicit-cells registry, built on CPU before any GPU)
    seen = {}

    def fake_run(argv, **kw):
        seen["env"] = dict(kw.get("env") or {})
        return SimpleNamespace(returncode=0, stdout=json.dumps({"ok": True, "results": []}), stderr="")

    monkeypatch.setattr(L.subprocess, "run", fake_run)
    L.run_object_identity({"targets": []}, "/staged/tree", row_slug=SLUGS[1])
    assert seen["env"]["SISTEER_CANDIDATE_SIS_SUBSET_SHA256"] == C.CANDIDATE_SIS_SUBSET_C16_SHA256
    assert seen["env"]["PYTHONPATH"] == "/staged/tree/src"
    L.run_object_identity({"targets": []}, "/staged/tree", row_slug="qwen35_4b-f22s1")
    assert "SISTEER_CANDIDATE_SIS_SUBSET_SHA256" not in seen["env"] and "SISTEER_LOAD_CANDIDATE_SIS" not in seen["env"]
    # every _serving_env call site in the launcher names the row (a call without it would be a TypeError at run time, in a container)
    text = (REPO / "scripts/modal/tinycap_served.py").read_text()
    assert text.count("_serving_env(") == 4 and text.count("def _serving_env(tree: str, reduction_mode: str, *, row_slug: str)") == 1
    assert text.count("_serving_env(tree, reduction_mode, row_slug=") == 3
    assert "run_object_identity(request, tree, row_slug=row.slug)" in text


# --------------------------------------------------------------------------------------------------------------------------- #
# the launch tags and store suffixes                                                                                            #
# --------------------------------------------------------------------------------------------------------------------------- #

def test_the_wave_and_relaunch_suffixes_match_the_lane_grammar_and_a_smoke_suffix_never_does(Wr):
    for k, view in enumerate(VIEWS, start=1):
        wave = f"qwen35_4b_{view}"
        assert Wr.out_suffix(wave, SHA) == f"bcbw-{wave}-{SHA[:7]}" and LANE_SUFFIX_RE.match(Wr.out_suffix(wave, SHA))
        for n in (1, 2, 3):
            assert LANE_SUFFIX_RE.match(Wr.out_suffix(wave, SHA, f"r{n}")).group("relaunch") == f"-r{n}"
        for tag in ("smoke", "smoke2", "sibat", "sibat-smoke"):
            assert not LANE_SUFFIX_RE.match(Wr.out_suffix(wave, SHA, tag)), tag
        # the launch tag is the lane's own: never a paper seed tag (a paper tag would mint the paper rows' run ids)
        assert not re.fullmatch(r"qwen35_4b_s\d+p?", wave) and wave != f"qwen35_4b_s{k}"
        launch = Wr.compose(FULL, wave=wave, sha=SHA, roster_sha256="f" * 64, launched_at="2026-11-15T00:00:00Z",
                            stage_kind="wave", app_token="abc123")
        assert launch.suffix == f"bcbw-{wave}-{SHA[:7]}" and launch.row == C.BCBW_SLUG_OF[("qwen35_4b", view)]
        assert C.ROUTE_NONE_CELL not in launch.cells and list(launch.cells) == list(CELLS6)
    # the paper adapter's own tag grammar refuses every lane wave name (the lane is admitted only through its declaration)
    try:
        from sisteer.registry.adapters import _bcbw_common as BC
    except Exception as e:  # noqa: BLE001 — the adapter package's import chain is the registry's; a failure here is reported, not hidden
        pytest.fail(f"the registry adapters do not import: {type(e).__name__}: {e}")
    for view in VIEWS:
        assert not BC.TAG_RE.match(f"qwen35_4b_{view}") and not BC.ROW_DIR_RE.match(f"qwen35_4b-f22{view}")


def test_the_waves_validate_and_anchor_on_the_seed_views_none():
    assert W.validate(FULL, W.load_pins(), C.MODELS, tree=REPO) == []
    for k, view in enumerate(VIEWS, start=1):
        w = W.wave(FULL, f"qwen35_4b_{view}")
        anchor = W.wave(FULL, w["anchor_wave"])
        assert w["anchor_wave"] == f"qwen35_4b_s{k}" and anchor["served"] == w["served"]
        assert W.runnable_legs(FULL, f"qwen35_4b_{view}") == W.runnable_legs(FULL, w["anchor_wave"]) == ["bigcodebench_full", "bfcl_full"]
        # every fitted cell of the row's config is named by a bank-staging prerequisite, and the subset file has its own
        assert W.staging_coverage_problems(w, C.model_row(w["model_row"]), where=view, repo=REPO) == []
        assert any(C.CANDIDATE_SIS_SUBSET_C16_REMOTE in p and C.CANDIDATE_SIS_SUBSET_C16_SHA256[:12] in p for p in w["prerequisites"])


# --------------------------------------------------------------------------------------------------------------------------- #
# the lane bank stager                                                                                                          #
# --------------------------------------------------------------------------------------------------------------------------- #

def test_the_stagers_lane_table_is_the_contracts_switch(ST):
    for view in C.BCBW_CTLSTOP_VIEWS + C.BCBW_SIKL4_VIEWS:
        lane = ST.lane_of(view)
        assert (lane.roster_rel, lane.pins_rel) == C.bcbw_view_sources("qwen35_4b", view), view
        assert lane.parents is (view in C.BCBW_CTLSTOP_VIEWS)
    for bad in ("s1", "sikl4s4", "sikl4s", "ctlstop64s0", "main"):
        with pytest.raises(ST.Refusal, match="not a lane view"):
            ST.lane_of(bad)


def _synthetic_sikl4_tree(root: Path, *, view: str = "sikl4s2", derived_cell: str | None = None) -> dict:
    """A tree holding the sikl4 roster + a pins file over FAKE object bytes, and a source mirror of those bytes and their sidecars."""
    tree, source = root / "tree", root / "source"
    roster = {"targets": {"qwen35_4b": {"views": {view: {"cells": {
        c: {"pins": C.BCBW_SIKL4_PINS_REL, "pin": f"qwen35_4b.{view}__{c}"} for c in CELLS6}}}}}}
    (tree / C.BCBW_SIKL4_ROSTER_REL).parent.mkdir(parents=True, exist_ok=True)
    (tree / C.BCBW_SIKL4_ROSTER_REL).write_text(yaml.safe_dump(roster, sort_keys=False))
    pins = {"qwen35_4b": {}}
    for i, c in enumerate(CELLS6):
        sid = f"{view}__{c}"
        blob = f"FAKE-OBJECT-{i}-".encode() + bytes([i]) * 64
        d = source / sid
        d.mkdir(parents=True)
        (d / "object.pt").write_bytes(blob)
        h = hashlib.sha256(blob).hexdigest()
        sidecar = "derived_meta.json" if c == derived_cell else "train_budget_meta.json"
        meta = {"object_sha256": h, ("derived_id" if c == derived_cell else "cell_id"): c}
        (d / sidecar).write_text(json.dumps(meta))
        (d / "staged_from.json").write_text(json.dumps({"staged_id": sid}))           # not an identity sidecar: never staged
        pins["qwen35_4b"][sid] = {"object_sha256": h, "staged_id": sid, "identity_sidecar": sidecar,
                                  "kind": "psr" if "psr" in c else "lora_all"}
    (tree / C.BCBW_SIKL4_PINS_REL).write_text(json.dumps(pins))
    return {"tree": tree, "source": source, "pins": pins, "view": view}


def test_the_stager_plans_a_sikl4_view_as_six_fitted_cells_with_no_parents(ST, tmp_path):
    w = _synthetic_sikl4_tree(tmp_path)
    bank = ST.plan_bank(w["tree"], w["view"], w["source"], tree_sha=SHA)
    assert [c.cell_id for c in bank.cells] == list(CELLS6) and all(c.lane_kind == "base" and not c.parents for c in bank.cells)
    assert bank.remote_view_root == f"/objects/sdfbase/full22/qwen35_4b/{w['view']}"
    assert all([f.name for f in c.files] == ["object.pt", "train_budget_meta.json"] for c in bank.cells)
    assert {c.staged_id: c.object_sha256 for c in bank.cells} == {s: p["object_sha256"] for s, p in w["pins"]["qwen35_4b"].items()}


def test_the_stager_refuses_a_wrong_sha_a_derived_pin_and_the_wrong_parents_flag(ST, tmp_path):
    w = _synthetic_sikl4_tree(tmp_path / "sha")
    (w["source"] / f"{w['view']}__C16__lora_r4" / "object.pt").write_bytes(b"NOT THE PINNED BYTES")
    with pytest.raises(ST.Refusal, match="object sha .* != pin"):
        ST.plan_bank(w["tree"], w["view"], w["source"], tree_sha=SHA)
    w = _synthetic_sikl4_tree(tmp_path / "derived", derived_cell="C16__psr_kl_all")
    with pytest.raises(ST.Refusal, match="derived object .* in the sikl4 lane"):
        ST.plan_bank(w["tree"], w["view"], w["source"], tree_sha=SHA)
    w = _synthetic_sikl4_tree(tmp_path / "flags")
    common = dict(tree=w["tree"], view=w["view"], source=w["source"], receipts_dir=tmp_path / "r", go=False, tree_sha=SHA,
                  tree_info={}, modal_bin=Path("/nonexistent/modal"), environment="demo-env", stage_dir=tmp_path / "stage")
    with pytest.raises(ST.Refusal, match="stages no parents: drop --parents-source"):
        ST.run(parents_source=tmp_path, **common)
    with pytest.raises(ST.Refusal, match="pass --parents-source"):
        ST.run(**{**common, "view": "ctlstop64s1"}, parents_source=None)
    with pytest.raises(ST.Refusal, match="modified against HEAD"):
        ST.run(parents_source=None, **{**common, "go": True, "tree_info": {"tree_on_remote": True,
                                                                             "lane_config_dirty": [f" M {C.BCBW_SIKL4_PINS_REL}"]}})
    # a pin into another lane's pins file refuses by name
    roster = yaml.safe_load((w["tree"] / C.BCBW_SIKL4_ROSTER_REL).read_text())
    roster["targets"]["qwen35_4b"]["views"][w["view"]]["cells"]["C16__lora_r4"]["pins"] = C.BCBW_CTLSTOP_PINS_REL
    (w["tree"] / C.BCBW_SIKL4_ROSTER_REL).write_text(yaml.safe_dump(roster, sort_keys=False))
    with pytest.raises(ST.Refusal, match="the sikl4 lane admits only"):
        ST.plan_bank(w["tree"], w["view"], w["source"], tree_sha=SHA)


def test_the_stagers_dry_run_puts_nothing_and_writes_a_bank_receipt_only(ST, tmp_path, monkeypatch):
    w = _synthetic_sikl4_tree(tmp_path)
    monkeypatch.setattr(ST.S, "registry_check", lambda objects, registry_root=None: {"status": "unverified", "detail": "test", "db": None,
                                                                                   "rows": {}})
    calls = []
    monkeypatch.setattr(ST.S, "default_runner", lambda argv, **kw: calls.append(argv))
    out = ST.run(tree=w["tree"], view=w["view"], source=w["source"], parents_source=None, receipts_dir=tmp_path / "receipts", go=False,
                 tree_sha=SHA, tree_info={}, modal_bin=Path("/nonexistent/modal"), environment="demo-env",
                 stage_dir=tmp_path / "stage", log=lambda s: None)
    assert out["dry"] is True and out["parents"] is None and Path(out["bank"]).is_file() and calls == []
    rec = json.loads(Path(out["bank"]).read_text())
    assert rec["lane"] == "sikl4" and rec["roster"] == C.BCBW_SIKL4_ROSTER_REL and rec["pins"] == C.BCBW_SIKL4_PINS_REL
    assert not (tmp_path / "receipts" / "dry" / "parents.json").exists()
