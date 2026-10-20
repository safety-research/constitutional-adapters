"""Geometry certification and adapter parity are SEPARATE facts, and the record says both.

Trigger: a serving probe completed with EVERY pair `reference_mismatch:…` (a reference-harness defect under
investigation elsewhere) and, because the geometry fold was driven by the ALLOWED menu, no geometry_record.json was written — the
unsteered u0 wave, which needs only the geometry, was blocked along with the LoRA menu. After this file's subjects:
* `write_verdict` splits each pair's reasons into SERVING reasons and PARITY reasons; a pair with no serving reason and nothing
  outstanding is SERVING-CERTIFIED and the geometry fold reads it whatever the HF+PEFT reference said; the verdict, the menu and
  the geometry record / overlay carry `adapter_parity` {status verified|mismatch|unverified, basis, per_pair};
* the MENU (`allowed_pairs.json` pairs) still requires verified parity — unchanged;
* consumers: an UNSTEERED reader (rows_for / budget_for / the unsteered padded pin / a u0 sampled view) sizes from a mismatch or
  unverified record; a STEERED one (the padded-with-object pin, the steered preflight, a steered sampled view) refuses it by name
  through geometry.require_verified_adapter_parity;
* `reverdict` re-derives the geometry record from the record with nothing re-run;
* geometry.py's parser requires the block (per set and record-level, the latter the sets' aggregate) and a record serialized
  without one reads `unverified`, stated.
CPU only; the smoke suite's toy dims and fake results; synthetic ids; no cluster path, no host name."""
from __future__ import annotations

import copy
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
import yaml

from sisteer.capeval.serving import geometry as g
from sisteer.capeval.serving import standup_manifest as sm

REPO = Path(__file__).resolve().parents[1]
PAD = REPO / "scripts/sdfbase/full22/static_padding"
for _p in (REPO / "scripts" / "sdfbase", PAD, REPO / "scripts" / "sdfbase" / "full22" / "sampled", REPO / "tests"):
    sys.path.insert(0, str(_p))
import hpo_probe_status as hps  # noqa: E402
import padding_preflight as pf  # noqa: E402
import serving_probe as sp  # noqa: E402
import submit_sampled as ss  # noqa: E402
from test_full22_front_door_targets import front_door_roster  # noqa: E402
from test_serving_probe_revocation import _mint_receipt  # noqa: E402
from test_serving_probe_smoke import REV, TOY, _passed  # noqa: E402
from test_site_geometry_overlay import KEY, QKVO, WO, _pin_program, _record, _set, _table  # noqa: E402

MODEL = KEY
MISMATCH = {"status": "fail", "basis": sp.BASIS_HF_PEFT, "per_adapter": {"constitution__lora": {"status": "fail"}, "control__lora": {"status": "fail"},
                                                                          "difference__lora": {"status": "fail"}}}
#: the front door's synthetic roster block, built ONCE at import from a tmp copy of the real models table in which the demo row RUNS
#: IN PHASES: u0 = its unsteered cells (none + the prompted arms), s2 = its staged objects. The demo itself is single-phase (ONE wave
#: carrying every arm, DECLARED on its row — `wave: battery`, read through the table's one reader, reproduce/lib/models.py — never a
#: view name typed here), and this module's planner tests need an UNSTEERED view beside a STEERED one of the SAME key — the geometry
#: overlay, the preflight and the sbatch pins are all keyed on MODEL — so the twin keeps the demo's key and pins and declares no wave of
#: its own (u0 / s2 are phase views by name); the sampled planner reads only the roster block's cells, never the table.
def _lib_models():
    import importlib.util
    spec = importlib.util.spec_from_file_location("_repro_lib_models_geometry_twin", REPO / "reproduce" / "lib" / "models.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


_TABLE = yaml.safe_load((REPO / "reproduce" / "models.yaml").read_text(encoding="utf-8"))
_ROW = _TABLE["models"][MODEL]
_WAVE = _lib_models().declared_wave_of(_ROW)
assert _WAVE and _WAVE in (_ROW.get("views") or {}), f"{MODEL} declares no wave view in reproduce/models.yaml (wave={_WAVE!r})"
_S1 = _ROW["views"][_WAVE]["cells"]
_ROW["views"] = {"u0": {"cells": [c for c in _S1 if c == "none" or c.startswith("prompted__")]},
                 "s2": {"cells": [c for c in _S1 if c != "none" and not c.startswith("prompted__")]}}
_ROW.pop(_lib_models().WAVE_KEY, None)
_PHASED_TABLE = Path(tempfile.mkdtemp(prefix="sisteer-phased-table-")) / "models.yaml"
_PHASED_TABLE.write_text(yaml.safe_dump(_TABLE, sort_keys=False), encoding="utf-8")
ROSTER = front_door_roster(MODEL, ("u0", "s2"), table=str(_PHASED_TABLE))
S2, U0 = f"{MODEL}_s2", f"{MODEL}_u0"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for v in (g.SITE_GEOMETRY_DIR_ENV, g.SITE_DIR_ENV, g.SHARED_ROOT_ENV, pf.MODELS_TABLE_ENV, sp.VISIBLE_DEVICES_ENV, "SLURM_JOB_ID"):
        monkeypatch.delenv(v, raising=False)
    orig = ss.split_target
    monkeypatch.setattr(ss, "split_target", lambda t: (MODEL, t[len(MODEL) + 1:]) if t.startswith(MODEL + "_") else orig(t))


def _plan():
    """wo r4 (S1cc + S1d + the pair's union) + mlp r4 (S1cc + S1d + S2 + the union) at 4,096 / 49,152 on the toy dims: the split topology
    of the 2026-10-01 ruling plus each pair's union (the served launch class, 2026-10-11), 14 launches (seven manifests, two windows)."""
    plan = sp.build_plan(TOY, model_key=MODEL, model_id="org/toy", revision=REV, sets=("wo", "mlp"), ranks=(4,), windows=(4096, 49152), ladder_caps=())
    assert plan["n_launches"] == 14
    return plan


def _results(plan, reference):
    """Every launch came up and served (engine S 64 at 4,096, 6 at 49,152 — the 49,152 product is the budget); the HF+PEFT reference
    said `reference` on each (None: never recorded)."""
    out = {}
    for rec in plan["launches"]:
        m = sm.Manifest.from_dict(plan["manifests"][rec["manifest"]])
        out[rec["id"]] = _passed(rec, m, engine_S=(64 if rec["id"].endswith("L4096") else 6), reference=reference)
    return out


def _record_dir(tmp_path: Path, reference):
    out = tmp_path / "rec"
    plan = _plan(); results = _results(plan, reference)
    sp.write_json_atomic(out / sp.PLAN_FILE, plan)
    for lid, res in results.items():
        sp.write_json_atomic(sp.launch_result_path(out, lid), res)
    return out, plan, results


# --------------------------------------------------------------------------------------------- the fold


def test_a_fold_with_every_pair_mismatched_still_writes_the_geometry_record_flagged_mismatch_and_the_menu_stays_empty(tmp_path, monkeypatch):
    """The live record's shape: every pair's engines came up and served, every pair failed HF+PEFT parity. Before: `not_applicable`,
    no geometry record, the u0 wave blocked. After: the geometry is folded from the serving-certified pairs and flagged `mismatch` in
    the record, the overlay, the verdict and the menu; the menu's `pairs` stay EMPTY; the status tool says READY; the RESULT line and
    the memo say both facts; an unsteered reader resolves the row."""
    out, plan, results = _record_dir(tmp_path, MISMATCH)
    site = tmp_path / "site"
    v, menu = sp.write_verdict(plan, results, probe_job_id="11111111", out=out, verdict_uri=str(out / sp.VERDICT_FILE), geometry_dir=site,
                               probe_cluster="cluster_a")
    # (2) the MENU is unchanged: every pair excluded on parity alone, nothing allowed, the verdict complete and certifying
    assert v["status"] == "complete" and v["certifying"] is True and menu["pairs"] == [] and len(menu["excluded"]) == 2
    assert all(e["reason"].startswith("reference_mismatch:") and e["decided"] for e in menu["excluded"])
    # (1) … but both pairs' SERVING is certified and the geometry is folded from them, flagged mismatch everywhere it lands
    assert sorted(map(tuple, v["serving_certified_pairs"])) == [(4, "mlp"), (4, "wo")] == sorted(map(tuple, menu["serving_certified_pairs"]))
    fold = v["geometry_fold"]
    assert fold["status"] == "written" and fold[sp.ADAPTER_PARITY_KEY] == "mismatch" and sorted(fold["sets"]) == ["mlp", "wo"], fold
    assert all(s[sp.ADAPTER_PARITY_KEY] == "mismatch" for s in fold["sets"].values())
    assert (out / sp.GEOMETRY_RECORD_FILE).is_file() and (site / f"{MODEL}.json").is_file() and (out / sp.ALLOWED_PAIRS_FILE).is_file()
    doc = json.loads((out / sp.GEOMETRY_RECORD_FILE).read_text())
    ap = doc[g.ADAPTER_PARITY_KEY]
    assert ap["status"] == "mismatch" and ap["basis"] == sp.BASIS_HF_PEFT and ap["rule"] == g.ADAPTER_PARITY_RULE
    assert [(e["rank"], e["set"], e["status"]) for e in ap["per_pair"]] == [(4, "mlp", "mismatch"), (4, "wo", "mismatch")]
    assert ap["per_pair"][0]["mismatched_adapters"] == ["constitution__lora", "control__lora", "difference__lora"]
    for sid in ("wo", "mlp"):
        blk = doc["sets"][sid][g.ADAPTER_PARITY_KEY]
        assert blk["status"] == "mismatch" and blk["per_pair"] == [ap["per_pair"][["mlp", "wo"].index(sid)]] and doc["sets"][sid]["allowed_pairs"] == [[4, sid]]
    assert doc["provenance"]["reference_check_basis"] == sp.BASIS_HF_PEFT and sorted(map(tuple, doc["provenance"]["allowed_pairs"])) == [(4, "mlp"), (4, "wo")]
    for d in (v, menu, json.loads((out / sp.ALLOWED_PAIRS_FILE).read_text()), json.loads((out / sp.VERDICT_FILE).read_text())):
        assert d[sp.ADAPTER_PARITY_KEY]["status"] == "mismatch" and [(e["rank"], e["set"]) for e in d[sp.ADAPTER_PARITY_KEY]["per_pair"]] == [(4, "mlp"), (4, "wo")]
    # every pair record says its own parity beside its serving certification
    for p in v["excluded"]:
        assert p["reason"].startswith("reference_mismatch:")
    # the overlay parses (the block is required and consistent), and an UNSTEERED reader resolves the measured row from it
    monkeypatch.setenv(g.SITE_GEOMETRY_DIR_ENV, str(site))
    sg = g.load_site_geometry(MODEL)
    assert sg.adapter_parity["status"] == "mismatch" and sg.sets["wo"].adapter_parity["per_pair"][0]["status"] == "mismatch"
    rows, rb = g.rows_for(MODEL, "wo"), g.budget_for(MODEL, "wo")
    assert rows.site is not None and rb.measured and rb.batched_token_budget == 6 * 49152 and g.lora_max_module_dim_for(MODEL, "wo") == sg.sets["wo"].lora_max_module_dim
    # the status tool: a complete record with a menu (empty) and a stamped verdict is READY — the front door fetches the geometry
    sp.write_progress(plan, results, out)
    state, detail, info = hps.probe_state_record(out, job_active=False)
    assert state == hps.READY, (state, detail)
    # the RESULT line says both facts; the exit is 0 (the geometry fold is written)
    rc = sp.main(["verdict", "--model-key", MODEL, "--out", str(out), "--probe-job-id", "11111111", "--geometry-dir", str(tmp_path / "site2"),
                  "--probe-cluster", "cluster_a"])
    line = (out / sp.RESULT_FILE).read_text()
    assert rc == 0 and line.startswith("state=VERDICT pairs_allowed=0 pairs_excluded=2") and "parity=mismatch" in line and "geometry=written:mlp(" in line, line
    # the memo banners the parity, tables the per-pair facts and names the rule
    memo = sp.render_memo(json.loads((out / sp.VERDICT_FILE).read_text()))
    assert "ADAPTER PARITY: MISMATCH (D145)" in memo and "| 4 | wo | `mismatch` | hf_peft | constitution__lora, control__lora, difference__lora |" in memo
    assert "| serving-certified pairs | adapter parity |" in memo and "| `wo` |" in memo and "| `mismatch` |" in memo


def test_reference_missing_folds_the_geometry_flagged_unverified_and_still_refuses_certification(tmp_path):
    """No HF+PEFT evidence at all (the `uncertified` verdict): the menu is NOT written and the CLI exits 6 as before — and the geometry of
    the serving-certified pairs IS folded, flagged `unverified`, so an unsteered standup is not held hostage to a reference phase that
    never ran; the RESULT line carries both."""
    out, plan, results = _record_dir(tmp_path, None)
    site = tmp_path / "site"
    v, menu = sp.write_verdict(plan, results, probe_job_id="1", out=out, geometry_dir=site)
    assert v["status"] == "uncertified" and menu["pairs"] == [] and not (out / sp.ALLOWED_PAIRS_FILE).exists()
    assert v["geometry_fold"]["status"] == "written" and v["geometry_fold"][sp.ADAPTER_PARITY_KEY] == "unverified"
    doc = json.loads((site / f"{MODEL}.json").read_text())
    assert doc[g.ADAPTER_PARITY_KEY]["status"] == "unverified" and doc[g.ADAPTER_PARITY_KEY]["basis"] is None
    assert all(e["status"] == "unverified" and e["mismatched_adapters"] == [] for e in doc[g.ADAPTER_PARITY_KEY]["per_pair"])
    assert v[sp.ADAPTER_PARITY_KEY]["status"] == "unverified" and sorted(map(tuple, v["serving_certified_pairs"])) == [(4, "mlp"), (4, "wo")]
    rc = sp.main(["verdict", "--model-key", MODEL, "--out", str(out), "--probe-job-id", "1", "--geometry-dir", str(site)])
    line = (out / sp.RESULT_FILE).read_text()
    assert rc == sp.EXIT_UNCERTIFIED and line.startswith("state=UNCERTIFIED certifying=false reason=reference_missing") and "parity=unverified" in line \
        and "geometry=written:" in line, line
    # a PARTIAL verdict (a launch outstanding) still folds NOTHING
    partial = {k: r for k, r in results.items() if not k.startswith("mlp_r4_S2")}
    v2, _ = sp.write_verdict(plan, partial, probe_job_id="1", out=tmp_path / "p", geometry_dir=tmp_path / "psite")
    assert v2["status"] == "partial" and v2["geometry_fold"]["status"] == "not_folded" and not (tmp_path / "p" / sp.GEOMETRY_RECORD_FILE).exists()


def test_reverdict_rederives_the_geometry_record_from_the_record_with_nothing_rerun(tmp_path):
    """The live record as job 805161 left it under the earlier tree: a complete checkpoint, a `complete` verdict, an empty menu, NO
    geometry record. `reverdict --record <dir> --reopen-token <nonce>` (the receipt minted by the front door) rewrites
    the certification artifacts from the record: the geometry record appears, flagged mismatch, the launch results are byte-identical
    before and after (nothing re-ran), the generation is unchanged and the record reads READY."""
    out, plan, results = _record_dir(tmp_path, MISMATCH)
    site = tmp_path / "site"
    # the earlier state: the verdict + empty menu without a geometry record (the old fold said not_applicable)
    v0, _ = sp.write_verdict(plan, results, probe_job_id="805161", out=out, verdict_uri=str(out / sp.VERDICT_FILE), geometry_dir=tmp_path / "old")
    (out / sp.GEOMETRY_RECORD_FILE).unlink()
    old_v = json.loads((out / sp.VERDICT_FILE).read_text()); old_v["geometry_fold"] = {"status": "not_applicable", "reason": "no allowed pair"}
    old_v.pop(sp.ADAPTER_PARITY_KEY); old_v.pop("serving_certified_pairs")
    sp.write_json_atomic(out / sp.VERDICT_FILE, old_v)
    sp.write_progress(plan, results, out)
    assert json.loads((out / sp.PROGRESS_FILE).read_text())["complete"] is True
    before = {p.name: p.read_bytes() for p in (out / sp.LAUNCHES_DIR).glob("*.json")}
    gen0 = sp.result_generation(out)["generation"]
    nonce = _mint_receipt(out, gen0)
    rc = sp.main(["reverdict", "--record", str(out), "--reopen-token", nonce, "--geometry-dir", str(site), "--probe-cluster", "cluster_a"])
    assert rc == 0
    assert {p.name: p.read_bytes() for p in (out / sp.LAUNCHES_DIR).glob("*.json")} == before               # nothing re-ran
    assert sp.result_generation(out)["generation"] == gen0
    doc = json.loads((out / sp.GEOMETRY_RECORD_FILE).read_text())
    assert doc[g.ADAPTER_PARITY_KEY]["status"] == "mismatch" and doc["provenance"]["probe_job_id"] == "805161" and doc["provenance"][g.RESULT_GENERATION_KEY] == gen0
    assert (site / f"{MODEL}.json").read_bytes() == (out / sp.GEOMETRY_RECORD_FILE).read_bytes()               # one serializer, one content
    v = json.loads((out / sp.VERDICT_FILE).read_text())
    assert v["geometry_fold"]["status"] == "written" and v[sp.ADAPTER_PARITY_KEY]["status"] == "mismatch" and json.loads((out / sp.ALLOWED_PAIRS_FILE).read_text())["pairs"] == []
    assert hps.probe_state_record(out, job_active=False)[0] == hps.READY
    assert "ADAPTER PARITY: MISMATCH (D145)" in (out / sp.MEMO_FILE).read_text()
    assert not (out / sp.RECEIPTS_DIR / f"{gen0}.{nonce}.json").exists()                                     # the receipt was consumed


# --------------------------------------------------------------------------------------------- the consumers


def _mismatch_overlay(tmp_path: Path) -> Path:
    out, plan, results = _record_dir(tmp_path, MISMATCH)
    site = tmp_path / "site"
    v, _ = sp.write_verdict(plan, results, probe_job_id="1", out=out, geometry_dir=site)
    assert v["geometry_fold"]["status"] == "written"
    return site


def test_the_steered_consumers_refuse_a_mismatch_record_by_name_and_the_unsteered_path_accepts_it(tmp_path, monkeypatch):
    """Item (3): the same overlay — folded from a record whose every pair failed parity — is read by every consumer. Unsteered:
    rows_for / budget_for, the preflight's check 5, the unsteered padded pin, the u0 sampled view all size from it. Steered: the
    preflight with --steered, the padded-with-object pin and a steered sampled view refuse BY NAME (the pair, the status, the
    adapters that disagreed, the way back) — never a Traceback, never a silent 0."""
    site = _mismatch_overlay(tmp_path)
    table = _table(tmp_path / "t", WO)                                                  # the promoted pick: o_proj -> wo, rank 4
    monkeypatch.setenv(g.SITE_GEOMETRY_DIR_ENV, str(site)); monkeypatch.setenv(pf.MODELS_TABLE_ENV, str(table))
    L = 49152
    # --- the preflight (check 5): unsteered sizes and REPORTS the parity; steered refuses by name
    geo = pf.geometry_expectation(MODEL, L)
    assert geo["budget_from_site_overlay"] and geo["served_S"] == 6 and geo["steered"] is False
    assert geo[g.ADAPTER_PARITY_KEY]["status"] == "mismatch" and geo[g.ADAPTER_PARITY_KEY]["mismatched_adapters"] == ["constitution__lora", "control__lora", "difference__lora"]
    with pytest.raises(ValueError, match="this steered padded standup serves a trained adapter of \\(rank 4, set 'wo'\\), whose adapter parity .* is 'mismatch'") as ei:
        pf.geometry_expectation(MODEL, L, steered=True)
    assert "control__lora" in str(ei.value) and "reverdict" in str(ei.value) and "unsteered standup may size from it" in str(ei.value)
    with pytest.raises(ValueError, match="\\(rank any, set 'wo'\\)"):
        pf.geometry_expectation(MODEL, L, module_set="wo", steered=True)               # no pick in hand: the set's aggregate decides
    _, problems = pf.check_geometry(MODEL, L, None, steered=True)
    assert len(problems) == 1 and "adapter parity" in problems[0] and "'mismatch'" in problems[0]
    assert pf.check_geometry(MODEL, L, None)[1] == []
    # --- the two padded pins, run from their heredocs: the unsteered launcher passes on the measured geometry, the steered one refuses
    ep = {"engine": "lora_native", "id_name_map": {"1": "none__lora0", "2": "si_x__lora_r4"}, "geometry": {"adjustments": []}}
    ep_path = tmp_path / "endpoint.json"; ep_path.write_text(json.dumps(ep))
    env = {"PYTHONPATH": str(REPO / "src"), "PATH": "/usr/bin:/bin", g.SITE_GEOMETRY_DIR_ENV: str(site), pf.MODELS_TABLE_ENV: str(table)}

    def pin(sbatch, extra):
        r = subprocess.run([sys.executable, "-", str(ep_path), MODEL, str(L), "6"] + extra, input=_pin_program(sbatch), capture_output=True, text=True,
                           cwd=REPO, env=env)
        return r.returncode, r.stdout + r.stderr
    rc, out = pin(PAD / "target_static_padded.sbatch", [])
    assert rc == 0 and "[geometry-pin] OK" in out and "for module set 'wo'" in out, out
    rc, out = pin(PAD / "target_static_padded_obj.sbatch", [""])
    assert rc == 1 and "[geometry-pin] REFUSED" in out and "(rank 4, set 'wo'), whose adapter parity" in out and "'mismatch'" in out \
        and "Traceback" not in out and "[geometry-pin] OK" not in out, out
    # the steered launcher's preflight call passes --steered; the unsteered one does not
    obj = "\n".join(l for l in (PAD / "target_static_padded_obj.sbatch").read_text().splitlines() if not l.lstrip().startswith("#"))
    uns = "\n".join(l for l in (PAD / "target_static_padded.sbatch").read_text().splitlines() if not l.lstrip().startswith("#"))
    assert '--model-key "$KEY" --steered' in obj and "require_verified_adapter_parity" in obj
    assert "--steered" not in uns and "require_verified_adapter_parity" not in uns
    # --- the sampled planner: u0 (base + prompted) sizes from the overlay's row; s2 (staged objects) refuses by name
    p = ss.plan(copy.deepcopy(ROSTER), [U0], {U0: "cluster_b"}, 9)
    served = {r["key"]: r for r in p["served"]}[U0]
    assert served["module_set"] == "wo" and served["budget"] == 6 * 49152
    with pytest.raises(SystemExit, match=f"{S2}: {MODEL}: {S2}: the steered cells \\[.*\\] serves a trained adapter of \\(rank 4, set 'wo'\\), whose adapter parity .* 'mismatch'"):
        ss.plan(copy.deepcopy(ROSTER), [S2], {S2: "cluster_b"}, 9)
    # --- a VERIFIED overlay lets the steered consumers through (the earlier world, stated)
    good = tmp_path / "good"; good.mkdir()
    (good / f"{MODEL}.json").write_text(json.dumps(_record(sets={"wo": _set("wo", WO, 4096, S=6, L=L)})))
    monkeypatch.setenv(g.SITE_GEOMETRY_DIR_ENV, str(good))
    assert pf.geometry_expectation(MODEL, L, steered=True)[g.ADAPTER_PARITY_KEY]["status"] == "verified"
    env[g.SITE_GEOMETRY_DIR_ENV] = str(good)
    rc, out = pin(PAD / "target_static_padded_obj.sbatch", [""])
    assert rc == 0 and "[geometry-pin] adapter parity for (rank 4, set 'wo'): verified" in out and "[geometry-pin] OK" in out, out
    p = ss.plan(copy.deepcopy(ROSTER), [S2], {S2: "cluster_b"}, 9)
    assert {r["key"]: r for r in p["served"]}[S2]["budget"] == 6 * L


# --------------------------------------------------------------------------------------------- the schema


def test_the_record_says_both_facts_and_a_record_written_without_a_block_reads_unverified():
    rec = _record(sets={"wo": _set("wo", WO, 4096), "qkvo": _set("qkvo", QKVO, 9216, S=2, L=98304, parity=g.PARITY_MISMATCH)})
    site = g.parse_site_geometry(rec, model_key=MODEL)
    assert site.adapter_parity["status"] == "mismatch" and site.sets["wo"].adapter_parity["status"] == "verified" and site.sets["qkvo"].adapter_parity["status"] == "mismatch"
    assert [(e["rank"], e["set"], e["status"]) for e in site.adapter_parity["per_pair"]] == [(4, "qkvo", "mismatch"), (4, "wo", "verified")]
    assert g.require_verified_adapter_parity(site.sets["wo"], 4)["status"] == "verified" and g.adapter_parity_for(site.sets["wo"])["status"] == "verified"
    with pytest.raises(g.SiteGeometryError, match="a steered standup serves a trained adapter of \\(rank 4, set 'qkvo'\\), whose adapter parity .* is 'mismatch'.*control__lora"):
        g.require_verified_adapter_parity(site.sets["qkvo"], 4, what="a steered standup")
    with pytest.raises(g.SiteGeometryError, match="\\(rank 16, set 'wo'\\) is not one the serving probe folded"):
        g.require_verified_adapter_parity(site.sets["wo"], 16)
    # the round trip keeps both blocks; the block is CONTENT (the canonical sha moves with it)
    back = g.parse_site_geometry(site.as_record(), model_key=MODEL)
    assert back.adapter_parity == site.adapter_parity and back.sets["qkvo"].adapter_parity == site.sets["qkvo"].adapter_parity
    verified_only = _record(sets={"wo": _set("wo", WO, 4096), "qkvo": _set("qkvo", QKVO, 9216, S=2, L=98304)})
    assert g.canonical_site_geometry_sha256(rec) != g.canonical_site_geometry_sha256(verified_only)
    # a set written WITHOUT the block (an older caller, a fixture): `unverified` for its pairs, stated — never inferred as verified
    legacy = {k: v for k, v in _set("wo", WO, 4096).items() if k != "adapter_parity"}
    rec2 = _record(sets={"wo": legacy})
    assert rec2["sets"]["wo"]["adapter_parity"]["status"] == "unverified" and "no adapter-parity block" in rec2["sets"]["wo"]["adapter_parity"]["note"]
    assert rec2["adapter_parity"]["status"] == "unverified" and rec2["adapter_parity"]["per_pair"] == [
        {"rank": 4, "set": "wo", "status": "unverified", "basis": None, "mismatched_adapters": []}]
    with pytest.raises(g.SiteGeometryError, match="\\(rank 4, set 'wo'\\), whose adapter parity .* is 'unverified'"):
        g.require_verified_adapter_parity(g.parse_site_geometry(rec2, model_key=MODEL).sets["wo"], 4)
    # a caller-given record-level block that is not the sets' aggregate is refused
    with pytest.raises(g.SiteGeometryError, match="disagrees with the sets' blocks"):
        g.site_geometry_record(model_key=MODEL, hf_id="org/toy", revision="0" * 40, written_utc="2026-09-30T00:00:00Z", sets={"wo": _set("wo", WO, 4096)},
                               provenance=rec["provenance"], adapter_parity={"status": "mismatch", "basis": "hf_peft",
                                                                             "per_pair": [{"rank": 4, "set": "wo", "status": "mismatch"}]})
    # the aggregate, and the probe's constants ARE geometry's
    assert g.adapter_parity_aggregate([]) == "unverified" and g.adapter_parity_aggregate(["verified", "unverified"]) == "unverified"
    assert g.adapter_parity_aggregate(["verified", "mismatch", "unverified"]) == "mismatch" and g.adapter_parity_aggregate(["verified", "verified"]) == "verified"
    assert (sp.ADAPTER_PARITY_KEY, sp.PARITY_VERIFIED, sp.PARITY_MISMATCH, sp.PARITY_UNVERIFIED) == (g.ADAPTER_PARITY_KEY, g.PARITY_VERIFIED, g.PARITY_MISMATCH, g.PARITY_UNVERIFIED)
    assert "adapter_parity" in g.SITE_GEOMETRY_SET_REQUIRED
    # the rule is stated where the reader looks: geometry.py's docstring and the memo's constant
    assert "GEOMETRY CERTIFICATION AND ADAPTER PARITY ARE SEPARATE FACTS" in g.__doc__ and "D145" in g.ADAPTER_PARITY_RULE
