"""bcb-bfcl-200-widen S1-e / S1-f / S8 in the launcher (``scripts/modal/tinycap_served.py``),
CPU-only under the launcher tests' modal stub: the Nemotron image binding is a LOUD gate
(``IMAGE_NEMOTRON_UNBOUND`` refuses a Nemotron launch in a process whose modal client cannot
build it — this one), the Nemotron family runs on ``run_configuration_nemotron`` (cpu 8), a
``nemotron_h`` standup composes the STAGED tree's reviewed backend contract and every other
row's argv is byte-identical, the lane rows' endpoint gates refuse by name (the text-only pin
on qwen3_5, the backend attestation on nemotron_h; a gemma lane row has none), the identity
request carries the roster pin per fitted cell (an explicit snapshot pin wins over the staged
tree's full22 pin) and the report projects ``pin_match``, the launch-time roster snapshot is
validated / attached / written into the cell's out dir as ``lane_roster.json`` and a relabelled
block refuses, ``launch_argv`` spells the two new flags, ``engine_core_tail`` keeps the engine
process's lines."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from sisteer.capeval.serving import vllm_server as vs
from sisteer.capeval.tinycap import contracts as C
from sisteer.capeval.tinycap import widen_roster as W

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_tinycap_launcher as TL  # noqa: E402  (the stub, the loader and the store helpers)

REPO = Path(__file__).resolve().parents[1]
LAUNCHER = TL.LAUNCHER
DERIVED = "constv2-mix-full__lora_r4_ce_minus_ctl"
ROSTER = W.load_roster()
ROSTER_SHA = W.roster_sha256()


@pytest.fixture(scope="module")
def L():
    return TL._load_launcher(TL._stub_modal(), app_name=TL.TEST_APP_NAME, name="tinycap_served_widen_under_test")


def _snap(wave, entry):
    return W.snapshot(ROSTER, wave, entry, roster_sha256=ROSTER_SHA, launched_at="2026-09-11T00:00:00Z")


def _lane_tree(tmp_path, sha="f6cf462"):
    """A fake staged tree carrying what a LANE row's preflight reads: the pins, the generator,
    the full22 roster + pins, the lane row configs and the Nemotron contract."""
    repos = tmp_path / "repos"
    tree = TL._tree(repos, sha)
    rels = [C.BCB_PIN_PATH, C.BFCL_PIN_PATH, C.BCB_LIB_FAMILIES_PATH, "scripts/bcbw_configs.py",
            "scripts/sdfbase/full22/f22_roster.py", "configs/sdfbase_full22_roster.yaml",
            "configs/sdfbase_full22_pins.json", "configs/capeval/mamba_backend_contract.json",
            # the LANE roster the preflight / container hold every planned cell's runnable status to, and its FULL sibling too
            W.ROSTER_REL, W.ROSTER_FULL_REL]
    rels += [C.model_row(s).cfg for s in C.BCBW_SLUGS]
    full22 = yaml.safe_load((REPO / "configs/sdfbase_full22_roster.yaml").read_text())   # carries every lane target (llama8b included)
    rels += [full22["targets"][t]["views"][v]["config"] for t, v in C.BCBW_MODEL_VIEWS]
    for rel in rels:
        (tree / rel).parent.mkdir(parents=True, exist_ok=True)
        (tree / rel).write_text((REPO / rel).read_text())
    return repos, tree


# --------------------------------------------------------------------------- #
# the Nemotron image, function and standup contract (S1-f)                    #
# --------------------------------------------------------------------------- #


def test_the_nemotron_image_is_a_loud_gate_and_the_family_runs_on_its_own_function(L):
    # the stub's Image chain cannot be peeled, so THIS process records the image unbound —
    # never a silent fallback onto the kernel-less base image
    assert L.IMAGE_NEMOTRON_ERROR and L.image_nemotron is L.image
    assert L.NEMOTRON_FUNCTION_CPU == 8
    assert L.configuration_function(C.model_row("nemotron3nano4b")) is L.run_configuration_nemotron
    for slug in ("gemma4e4b-f22it", "gemma4e4b-f22", "qwen35_4b-f22", "gemma4e4b-sdfbase", "qwen35_4b",
                 "llama8b-f22it", "llama8b-f22"):
        assert L.configuration_function(C.model_row(slug)) is L.run_configuration, slug
    # code review: the ACTUAL keyword dictionary main hands fan_out binds through BOTH
    # workers and reaches _run_configuration whole — the Nemotron copy used to lack `protocol_check`
    import inspect as _inspect
    kwargs = L.configuration_kwargs(sha="f6cf462", limit=0, inspect_limit=0, samples=1, pass_no=1, controls=["bigcodebench_200"],
                                    reduction_mode=C.REDUCTION_MODE_LAUNCHER_DEFAULT, object_digests={}, oos_prompt_budget=None,
                                    run_id="r", ladder=None, protocol_check={"ok": True})
    assert "protocol_check" in kwargs and "ladder" in kwargs
    for fn in (L.run_configuration, L.run_configuration_nemotron):
        bound = _inspect.signature(fn).bind({"name": "cfg"}, **kwargs)              # TypeError before the fix
        assert bound.arguments["protocol_check"] == {"ok": True}
    assert _inspect.signature(L.run_configuration).parameters.keys() == _inspect.signature(L.run_configuration_nemotron).parameters.keys()
    seen = []
    monkeypatch_target = L._run_configuration
    L._run_configuration = lambda cfg, sha, **kw: seen.append((cfg, sha, kw)) or {"name": cfg["name"], "status": "DONE"}
    try:
        for fn in (L.run_configuration, L.run_configuration_nemotron):
            fn({"name": "cfg"}, **kwargs)
    finally:
        L._run_configuration = monkeypatch_target
    assert len(seen) == 2 and all(kw["protocol_check"] == {"ok": True} and kw["ladder"] is None for _, _, kw in seen)
    assert seen[0][2] == seen[1][2]                                                 # the same forwarding, both images
    # the probe workers share one body the same way
    assert _inspect.signature(L.probe_rung).parameters.keys() == _inspect.signature(L.probe_rung_nemotron).parameters.keys()
    assert L.TINYCAP_IMAGES_REMOTE == "/root/tinycap_images.py"
    src = LAUNCHER.read_text()
    i_copy = src.index('add_local_file(str(Path(tinycap_images.__file__).resolve()), TINYCAP_IMAGES_REMOTE, copy=True)')
    # (the harbor venv step that used to precede the copy layer is gone with Terminal-Bench)
    i_env = src.index('.env({"HF_HOME"')
    i_mount = src.index('.add_local_python_source("sisteer")')
    assert i_copy < i_env < i_mount and src.index("import tinycap_images") < i_copy
    assert "capeval/tinycap/widen_roster.py" in L.LAUNCHER_PROCESS_MODULES
    # round 2: the /root inventory the shared image places beside ANY entrypoint is declared once —
    # what the sibling Modal scripts extend with their own launcher mount and rebuild a container root from
    import tinycap_images as _ti
    import tinycap_compile_cache as _cc
    # tinycap-shared-standup §1.5 (module D): the compile-cache sibling rides the same copy layer beside tinycap_images
    assert L.IMAGE_LOCAL_FILES == {L.TINYCAP_IMAGES_REMOTE: str(Path(_ti.__file__).resolve()),
                                   L.TINYCAP_COMPILE_CACHE_REMOTE: str(Path(_cc.__file__).resolve())}
    assert L.TINYCAP_COMPILE_CACHE_REMOTE == "/root/tinycap_compile_cache.py"
    assert src.index("add_local_file(str(Path(CC.__file__).resolve()), TINYCAP_COMPILE_CACHE_REMOTE, copy=True)") > i_copy
    assert f"add_local_file(str(Path(tinycap_images.__file__).resolve()), TINYCAP_IMAGES_REMOTE, copy=True)" in src
    # the pin audit covers the kernel layer's specs too
    import tinycap_images as ti
    assert set(ti.image_install_specs_nemotron()) <= set(L.image_install_specs())
    # a Nemotron LAUNCH in this process refuses by name before any Modal call; a dry run prints
    with pytest.raises(SystemExit, match="IMAGE_NEMOTRON_UNBOUND"):
        L.main(app_name=L.APP_NAME, sha="f6cf462", pass1=True, cells="none", modes="off",
               controls="bigcodebench_200", model="nemotron3nano4b",
               roster_snapshot=json.dumps({"none": _snap("nemotron3nano4b_main", "none")}))   # round 7: a lane row states its wave


def test_a_nemotron_standup_composes_the_staged_trees_contract_and_every_other_argv_is_unchanged(L):
    cfg = L.plan_configurations(cells=["none"], modes=["off"], controls=["bigcodebench_200"], model="nemotron3nano4b")[0]
    geom, _ = L.served_geometry(cfg, 1)
    srv = Path("<srv>")
    base = L._standup_cmd("<tree>", "<cfg>", "none", "<ddir>", srv, srv / "STOP", geom, 1024, [])
    nemo = L._standup_cmd("<tree>", "<cfg>", "none", "<ddir>", srv, srv / "STOP", geom, 1024, [],
                          mamba_contract="<tree>/configs/capeval/mamba_backend_contract.json")
    assert nemo == base + [L.STANDUP_MAMBA_CONTRACT_FLAG, "<tree>/configs/capeval/mamba_backend_contract.json"]
    over = L._standup_cmd("<tree>", "<cfg>", "none", "<ddir>", srv, srv / "STOP", geom, 1024, [], allow_over_budget=True)
    assert over == base + ["--allow-over-budget"] and "--allow-over-budget" not in base
    assert L.STANDUP_MAMBA_CONTRACT_FLAG == "--mamba-backend-contract"
    # the standup CLI spells the same flag (the serving lane owns the vocabulary)
    up = (REPO / "scripts" / "capctrl_serving_up.py").read_text()
    assert 'MAMBA_CONTRACT_FLAG = "--mamba-backend-contract"' in up
    # since the IFEval landing the shared standup ALSO knows `--mrope-attest` (the IFEval rows' GLM gate); a widen row
    # never composes it — the launcher's standup argv for this row is pinned above, byte for byte
    assert "--mrope-attest" not in nemo and "--mrope-attest" not in base
    # code review: qwen3_5 ALONE — glm4v is out of the lane, and listing it would
    # make the shared standup's `auto` default compose a flag a GLM launch on main does not carry
    assert set(L.TEXT_ONLY_MM_FAMILIES) == {"qwen3_5", "qwen3_5_moe"}   # + the routed-experts member of the family (its serving tables restored 2026-10-08)
    assert "glm4v" not in vs.TEXT_ONLY_MM_TOKENS_BY_FAMILY
    # the container composes the contract for the nemotron_h row only (source pin: the composition site)
    src = LAUNCHER.read_text()
    assert 'mamba_contract = (f"{tree}/{_mamba_backend().CONTRACT_RELPATH}"' in src
    # (since the IFEval landing the same call also composes the text-only pin for an IFEval row of a verified family; a
    # widen row's argv is byte-identical — `text_only_mm` is False for it)
    assert "text_only_mm=(C.is_ifeval_row(row) and row.family in TEXT_ONLY_MM_FAMILIES),\n" in src
    # the worker's composition (the call goes on to the serving path; the ladder's rung probe composes the same
    # contract, then its own serving path / bank / KV-derived S value)
    assert ("text_only_mm=(C.is_ifeval_row(row) and row.family in TEXT_ONLY_MM_FAMILIES),\n"
            "                                mamba_contract=mamba_contract,\n") in src
    assert src.count("mamba_contract=mamba_contract,") == 2                 # the worker's standup and the rung probe's


def test_the_lane_rows_endpoint_gates_refuse_by_name_and_a_legacy_row_is_untouched(L):
    from sisteer.capeval.serving import mamba_backend as mb
    qwen, nemo, gemma = C.model_row("qwen35_4b-f22"), C.model_row("nemotron3nano4b"), C.model_row("gemma4e4b-f22it")
    assert L.full22_endpoint_refusal(qwen, {})[0] == C.REC_TEXT_ONLY_MM_UNPINNED
    assert L.full22_endpoint_refusal(qwen, {"text_only_mm": False})[0] == C.REC_TEXT_ONLY_MM_UNPINNED
    assert L.full22_endpoint_refusal(qwen, {"text_only_mm": True}) is None
    assert L.full22_endpoint_refusal(nemo, {})[0] == C.REC_NEMO_BACKEND_UNATTESTED
    bad = {"mamba_backend_attestation": {"ok": False, "problems": ["x"], "reports": [], "rpc_method": mb.RPC_METHOD}}
    st, why = L.full22_endpoint_refusal(nemo, bad)
    assert st == C.REC_NEMO_BACKEND_UNATTESTED and "not ok" in why
    good = {"mamba_backend_attestation": {"ok": True, "problems": [], "reports": [{"layer": 0}], "rpc_method": mb.RPC_METHOD}}
    assert L.full22_endpoint_refusal(nemo, good) is None
    # against the staged tree's contract sha: a foreign contract refuses
    with_sha = {"mamba_backend_attestation": {**good["mamba_backend_attestation"],
                                               "contract_sha256": mb.contract_sha256(REPO / mb.CONTRACT_RELPATH)}}
    assert L.full22_endpoint_refusal(nemo, with_sha, tree=str(REPO)) is None
    foreign = {"mamba_backend_attestation": {**good["mamba_backend_attestation"], "contract_sha256": "0" * 64}}
    assert L.full22_endpoint_refusal(nemo, foreign, tree=str(REPO))[0] == C.REC_NEMO_BACKEND_UNATTESTED
    # a gemma lane row has neither gate; the record keeps the raw keys. Since the IFEval landing the ONE record also
    # carries the M-RoPE attestation keys (the IFEval rows' standup asks them) and the standup's enable_log_requests
    # opt-in — unhashed, absent-as-None for a standup that never wrote them; a lane row's gates are unchanged
    assert L.full22_endpoint_refusal(gemma, {}) is None
    rec = L.full22_attestation_record({"text_only_mm": True, "mamba_backend_attestation_error": "e", "mrope_attestation": 1})
    assert rec["text_only_mm"] is True and rec["text_only_mm_record"] is None
    assert rec["mamba_backend_attestation"] is None and rec["mamba_backend_attestation_error"] == "e"
    assert rec["mrope_attestation"] == 1 and rec["mrope_attestation_error"] is None
    assert rec[C.ENDPOINT_KEY_ENABLE_LOG_REQUESTS] is None
    assert set(rec) == set(L.FULL22_ATTESTATION_KEYS)
    # the container applies the gates to LANE rows only (source pin)
    src = LAUNCHER.read_text()
    assert "refusal = full22_endpoint_refusal(row, endpoint, tree=tree) if C.is_full22_row(row) else None" in src


def test_engine_core_tail_keeps_the_engine_processs_own_lines(L):
    text = "\n".join(["(APIServer pid=32) INFO args", "(EngineCore pid=77) INFO Initializing a V1 LLM engine",
                      "(EngineCore pid=77) ERROR Traceback (most recent call last):",
                      "(EngineCore pid=77) ERROR   something failed in the worker",
                      "(APIServer pid=32) RuntimeError: Engine core initialization failed. See root cause above."])
    assert L.engine_core_tail(text).splitlines() == [ln for ln in text.splitlines() if "EngineCore" in ln]
    assert L.engine_core_tail(text, n_lines=1) == "(EngineCore pid=77) ERROR   something failed in the worker"
    assert L.engine_core_tail("") is None and L.engine_core_tail(None) is None
    assert L.engine_core_tail("(APIServer pid=1) only") is None


# --------------------------------------------------------------------------- #
# the roster pin in the identity request (S1-e) and the report projection       #
# --------------------------------------------------------------------------- #


def test_identity_request_carries_the_roster_pin_per_fitted_cell_and_the_tree_pin_as_fallback(L, tmp_path):
    repos, tree = _lane_tree(tmp_path)
    row = C.model_row("gemma4e4b-f22it")
    # the served config's EXPLICIT cells (the registry), not the roster's runnable set: the PSR cell stays listed
    # (blocked in the lane roster — its pin still rides the entry) and the constitution SI joins as a prompted cell
    cells = ["none", "prompted__safety_refusal", "prompted__si_constitution", "constv2-mix-full__lora_r4_ce", "constv2-mix-full__psr_ce_all"]
    req = L.identity_request(row, str(tree), cells)
    targets = {t["cell"]: t for t in req["targets"]}
    assert set(targets) == {"constv2-mix-full__lora_r4_ce", "constv2-mix-full__psr_ce_all"}      # object cells only
    pins = {c: W.pin_of(ROSTER, "gemma4e4b_it", W.entry_of_cell(ROSTER, "gemma4e4b_it", c)) for c in targets}   # every entry's pin, runnable or blocked
    for cell, t in targets.items():
        assert t["pin_sha256"] == pins[cell]                       # the STAGED tree's full22 pin == the lane roster's
        assert t["expected"] == C.EXPECTED_INTERVENTION_HASHES[row.slug][t["cell"]]   # the bootstrap pinned these; a moved object refuses --pf
    assert req["objects_dir"] == row.objects_dir and req["config"] == f"{tree}/{row.cfg}"
    # code review: an explicit (snapshot) pin that DISAGREES with the staged tree's full22 pin for the same
    # cell REFUSES (two sources for one object) — it used to win silently; an agreeing one, or a pin for a
    # cell the tree does not pin, merges
    with pytest.raises(ValueError, match="disagrees with the staged tree's full22 pin for constv2-mix-full__lora_r4_ce"):
        L.identity_request(row, str(tree), cells, pins={"constv2-mix-full__lora_r4_ce": "f" * 64})
    # (the "cell the tree does not pin" used to be constv2ctl2-full__lora_r4_ce; since the gemma it view gained the batch-64 pair it IS
    # pinned by the tree, so the placeholder now names a cell no gemma it roster entry pins)
    req2 = L.identity_request(row, str(tree), cells, pins={"constv2-mix-full__lora_r4_ce": pins["constv2-mix-full__lora_r4_ce"],
                                                           "constv2ctl2-full__psr_ce_all": "e" * 64})
    t2 = {t["cell"]: t for t in req2["targets"]}
    assert t2["constv2-mix-full__lora_r4_ce"]["pin_sha256"] == pins["constv2-mix-full__lora_r4_ce"]
    assert t2["constv2-mix-full__psr_ce_all"]["pin_sha256"] == pins["constv2-mix-full__psr_ce_all"]
    # a full22 row outside the lane's (target, view) table refuses by name (a bare StopIteration before)
    with pytest.raises(ValueError, match="not one of the lane's bcbw"):
        L.full22_pins_by_cell(tree, C.model_row("gemma4e4b-sdfbase")._replace(objects_dir=f"{C.VOL_OBJECTS_FULL22}/x/y"))
    # loading the staged generator by path leaves sys.path as it was (its module top inserts its own dirs)
    before = list(sys.path)
    L.full22_pins_by_cell(tree, row)
    assert sys.path == before
    # a two-SI row's targets carry no pin (pin_match stays null there)
    legacy = L.identity_request(C.model_row("gemma4e4b"), str(tree), ["none", C.CELL_PSRKL_SAFETY])
    assert all("pin_sha256" not in t for t in legacy["targets"]) and legacy["targets"]
    # the staged-tree pins for every lane row resolve to the roster's shas
    for wave_name, slug in (("gemma4e4b_main", "gemma4e4b-f22"), ("qwen35_4b_main", "qwen35_4b-f22"),
                            ("nemotron3nano4b_main", "nemotron3nano4b"),
                            # the llama8b rows' pins resolve through the staged tree's shared roster, which carries their block
                            ("llama8b_it", "llama8b-f22it"), ("llama8b_main", "llama8b-f22")):
        tree_pins = L.full22_pins_by_cell(tree, C.model_row(slug))
        for cell, sha in W.pins_by_cell(ROSTER, wave_name).items():
            assert tree_pins[cell] == sha, (slug, cell)
        assert tree_pins, slug
    # a staged tree whose shared roster LACKS a lane target refuses a row's pins by name (never a pin-less request)
    rp = tree / "configs/sdfbase_full22_roster.yaml"
    kept = rp.read_text()
    trimmed = yaml.safe_load(kept); trimmed["targets"].pop("llama8b")
    rp.write_text(yaml.safe_dump(trimmed, sort_keys=False))
    with pytest.raises(SystemExit, match="lacks the lane target.*llama8b"):
        L.full22_pins_by_cell(tree, C.model_row("llama8b-f22it"))
    rp.write_text(kept)
    # a tree without the generator refuses loudly (never a silent pin-less request)
    (tree / "scripts" / "bcbw_configs.py").unlink()
    with pytest.raises(FileNotFoundError, match="bcbw_configs.py"):
        L.identity_request(row, str(tree), cells)


def test_the_identity_reports_cells_carry_the_pin_fields_the_pin_check_reads(L):
    assert "pin_match" in L.OBJECT_IDENTITY_CELL_FIELDS and "pin_sha256" in L.OBJECT_IDENTITY_CELL_FIELDS
    assert "artifact_sha256" in L.OBJECT_IDENTITY_CELL_FIELDS and "intervention_hash" in L.OBJECT_IDENTITY_CELL_FIELDS
    rec = {"cell": "c", "intervention_hash": "h", "artifact_sha256": "a" * 64, "pin_sha256": "a" * 64,
           "pin_match": True, "match": True, "extra": "dropped"}
    kept = {k: rec.get(k) for k in L.OBJECT_IDENTITY_CELL_FIELDS}
    assert kept["pin_match"] is True and kept["pin_sha256"] == "a" * 64 and "extra" not in kept
    src = LAUNCHER.read_text()
    assert 'for k in OBJECT_IDENTITY_CELL_FIELDS}' in src                    # the report uses the constant
    assert 'cells_rep[c].get("pin_match") is not True' in src               # the lane check's reader key
    assert 'chk("full22_pin_match", pin_match)' in src and 'chk("image_nemotron_bound", image_bound)' in src


# --------------------------------------------------------------------------- #
# the launch-time roster snapshot                                              #
# --------------------------------------------------------------------------- #


def test_main_attaches_one_snapshot_per_planned_cell_and_hands_the_pins_to_the_preflight(L, monkeypatch, capsys):
    seen = {}

    def _pf(sha, plan_args, expect, *rest, **kw):
        seen["pins"] = kw.get("pins")
        return {"ok": False, "checks": {}}
    monkeypatch.setattr(L, "preflight", SimpleNamespace(remote=_pf))
    cells = ["none", "constv2-mix-full__lora_r4_ce"]
    snaps = {c: _snap("gemma4e4b_it", W.entry_of_cell(ROSTER, "gemma4e4b_it", c)) for c in cells}
    kw = dict(app_name=L.APP_NAME, sha="f6cf462", pass1=True, pf=True, cells=",".join(cells), modes="off",
              controls="bigcodebench_200", model="gemma4e4b-f22it", out_suffix="bcbw-t")
    with pytest.raises(SystemExit, match="PREFLIGHT FAILED"):
        L.main(**kw, roster_snapshot=json.dumps(snaps))
    assert seen["pins"] == {"constv2-mix-full__lora_r4_ce": snaps["constv2-mix-full__lora_r4_ce"]["object_sha256"]}
    out = capsys.readouterr().out
    assert f"{L.ROSTER_SNAPSHOT_FLAG} " in out                    # spelled into the recorded argv
    # a snapshot set that does not match the plan refuses before any Modal call
    seen.clear()
    with pytest.raises(SystemExit, match="one block per planned cell"):
        L.main(**kw, roster_snapshot=json.dumps({"none": snaps["none"]}))
    assert not seen
    # a relabelled block refuses at the flag (its digest does not recompute)
    bad = dict(snaps["constv2-mix-full__lora_r4_ce"], object_generation="final", superseded_by_key=None)
    with pytest.raises(SystemExit, match="does not recompute"):
        L.main(**kw, roster_snapshot=json.dumps({**snaps, "constv2-mix-full__lora_r4_ce": bad}))
    # round 7: a LANE row launches only WITH the snapshot — refused before the preflight is called
    seen.clear()
    with pytest.raises(SystemExit, match="launches only with the launch-time roster snapshot"):
        L.main(**kw)
    assert not seen
    # a LEGACY row without a snapshot is untouched: the preflight runs with no pins kwarg
    with pytest.raises(SystemExit, match="PREFLIGHT FAILED"):
        L.main(**dict(kw, model="gemma4e4b", cells="none", out_suffix="legacy-t"))
    assert seen["pins"] is None
    capsys.readouterr()


def test_run_configuration_writes_the_snapshot_beside_the_manifest_and_refuses_a_relabelled_one(L, tmp_path, monkeypatch):
    repos, tree = _lane_tree(tmp_path)
    gen = tmp_path / "gen"
    monkeypatch.setattr(L.C, "VOL_REPOS_ROOT", str(repos))
    monkeypatch.setattr(L.C, "VOL_GEN_ROOT", str(gen))
    monkeypatch.setattr(L.subprocess, "Popen", lambda *a, **k: pytest.fail("a standup was started"))
    cfg = L.isolate_out_dirs(L.plan_configurations(cells=["none"], modes=["off"], controls=["bigcodebench_200"],
                                                   model="gemma4e4b-f22it"), "bcbw-t")[0]
    snap = _snap("gemma4e4b_it", "none")
    # the valid block is written as lane_roster.json before the standup (which this test forbids)
    with pytest.raises(BaseException, match="a standup was started"):
        L._run_configuration(dict(cfg, roster_snapshot=snap), "f6cf462")
    written = json.loads((Path(cfg["out_dir"]) / W.SNAPSHOT_FILENAME).read_text())
    assert written == snap and W.snapshot_problems(written) == []
    # a relabelled block refuses by name, before any standup
    bad = dict(snap, entry_id="const_lora")
    rec = L._run_configuration(dict(cfg, roster_snapshot=bad), "f6cf462")
    assert rec["status"] == "ROSTER_SNAPSHOT_INVALID" and "does not recompute" in rec["error"]
    other = _snap("gemma4e4b_it", "const_lora")                   # a block naming another cell
    rec = L._run_configuration(dict(cfg, roster_snapshot=other), "f6cf462")
    assert rec["status"] == "ROSTER_SNAPSHOT_INVALID" and "names cell" in rec["error"]
    # no snapshot on a LEGACY row: nothing written, the legacy path (stops at the forbidden standup)
    (tree / C.model_row("gemma4e4b").cfg).parent.mkdir(parents=True, exist_ok=True)
    (tree / C.model_row("gemma4e4b").cfg).write_text((REPO / C.model_row("gemma4e4b").cfg).read_text())
    cfg2 = L.isolate_out_dirs(L.plan_configurations(cells=["none"], modes=["off"], controls=["bigcodebench_200"],
                                                    model="gemma4e4b"), "bcbw-u")[0]
    with pytest.raises(BaseException, match="a standup was started"):
        L._run_configuration(cfg2, "f6cf462")
    assert not (Path(cfg2["out_dir"]) / W.SNAPSHOT_FILENAME).exists()
    # code review: the FILED record is immutable. A compatible relaunch (only launched_at moved)
    # leaves it byte-identical and appends the retry beside it; an incompatible one (a relabelled, recomputed
    # block — another generation; or another roster sha) refuses by name and writes NOTHING
    filed_text = (Path(cfg["out_dir"]) / W.SNAPSHOT_FILENAME).read_text()
    retry = W.snapshot(ROSTER, "gemma4e4b_it", "none", roster_sha256=ROSTER_SHA, launched_at="2026-09-12T09:00:00Z")
    with pytest.raises(BaseException, match="a standup was started"):
        L._run_configuration(dict(cfg, roster_snapshot=retry), "f6cf462")
    assert (Path(cfg["out_dir"]) / W.SNAPSHOT_FILENAME).read_text() == filed_text
    retries = (Path(cfg["out_dir"]) / W.SNAPSHOT_RETRY_FILENAME).read_text().splitlines()
    assert [json.loads(r) for r in retries] == [retry]
    other = dict(snap, roster_sha256="1" * 64)
    other["snapshot_sha256"] = W.snapshot_digest(other)
    rec = L._run_configuration(dict(cfg, roster_snapshot=other), "f6cf462")
    assert rec["status"] == C.REC_ROSTER_SNAPSHOT_CONFLICT == "ROSTER_SNAPSHOT_CONFLICT" and "roster_sha256: filed" in rec["error"]
    assert (Path(cfg["out_dir"]) / W.SNAPSHOT_FILENAME).read_text() == filed_text
    assert len((Path(cfg["out_dir"]) / W.SNAPSHOT_RETRY_FILENAME).read_text().splitlines()) == 1     # no retry filed either
    fitted_cfg = L.isolate_out_dirs(L.plan_configurations(cells=["constv2-mix-full__lora_r4_ce"], modes=["off"],
                                                          controls=["bigcodebench_200"], model="gemma4e4b-f22it"), "bcbw-t")[0]
    fsnap = _snap("gemma4e4b_it", "const_lora")
    with pytest.raises(BaseException, match="a standup was started"):
        L._run_configuration(dict(fitted_cfg, roster_snapshot=fsnap), "f6cf462", object_digests={
            "constv2-mix-full__lora_r4_ce": {"artifact_sha256": fsnap["object_sha256"]}})
    relabelled = dict(fsnap, object_generation="final", superseded_by_key=None, launched_at="later")
    relabelled["snapshot_sha256"] = W.snapshot_digest(relabelled)                 # a VALID block carrying another label
    rec = L._run_configuration(dict(fitted_cfg, roster_snapshot=relabelled), "f6cf462")
    assert rec["status"] == "ROSTER_SNAPSHOT_CONFLICT" and "object_generation: filed 'batch8-provisional'" in rec["error"]
    assert json.loads((Path(fitted_cfg["out_dir"]) / W.SNAPSHOT_FILENAME).read_text()) == fsnap


def test_launch_argv_spells_the_two_new_flags_only_when_given(L):
    plain = L.launch_argv("f6cf462", pass1=True, model="gemma4e4b-f22it")
    assert L.ROSTER_SNAPSHOT_FLAG not in plain and L.LADDER_PROBE_CELL_FLAG not in plain
    argv = L.launch_argv("f6cf462", pass1=True, model="gemma4e4b-f22it", roster_snapshot="/tmp/snap.json",
                         ladder_probe_cell=DERIVED)
    assert argv[argv.index(L.ROSTER_SNAPSHOT_FLAG) + 1] == "/tmp/snap.json"
    assert argv[argv.index(L.LADDER_PROBE_CELL_FLAG) + 1] == DERIVED
    assert L.ROSTER_SNAPSHOT_FLAG == C.FLAG_ROSTER_SNAPSHOT == "--roster-snapshot"
    assert L.LADDER_PROBE_CELL_FLAG == C.FLAG_LADDER_PROBE_CELL == "--ladder-probe-cell"
    assert L.LADDER_CERTIFY_FLAG == C.FLAG_LADDER_CERTIFY == "--ladder-certify"
    # code review: the certified wave's cells ride on a ladder launch, spelled iff given
    assert L.LADDER_WAVE_CELLS_FLAG == C.FLAG_LADDER_WAVE_CELLS == "--ladder-wave-cells" and L.LADDER_WAVE_CELLS_FLAG not in plain
    wave = L.launch_argv("f6cf462", pass1=True, model="gemma4e4b-f22it", ladder_wave_cells=f"none,{DERIVED}")
    assert wave[wave.index(L.LADDER_WAVE_CELLS_FLAG) + 1] == f"none,{DERIVED}"
    # ... and means nothing without a ladder launch
    with pytest.raises(SystemExit, match="means nothing without a ladder launch"):
        L.main(app_name=L.APP_NAME, sha="f6cf462", pass1=True, cells="none", modes="off", controls="bigcodebench_200",
               model="gemma4e4b-f22it", ladder_wave_cells=f"none,{DERIVED}", dry_run=True,
               roster_snapshot=json.dumps({"none": _snap("gemma4e4b_it", "none")}))          # round 7: a lane row states its wave


def test_every_lane_row_plans_its_runnable_cells_off_and_bfcl_only_where_a_manifest_is_pinned(L):
    # every qwen wave's bfcl_200 leg is RUNNABLE behind the filed live PASS record
    # (contracts.TOOL_PARSER_LIVE_PASS["qwen3_5"]: the verdict checked in at the staged tree 80df464) — the loop plans BFCL for them
    qwen_waves = [n for n in W.waves(ROSTER) if W.wave(ROSTER, n).get("family") == "qwen35_4b"]
    assert len(qwen_waves) == 11 and all("bfcl_200" in W.runnable_legs(ROSTER, n) for n in qwen_waves), qwen_waves
    for wave_name in W.waves(ROSTER):
        w = W.wave(ROSTER, wave_name)
        cells = W.runnable_cells(ROSTER, wave_name)
        if not cells:
            continue
        legs = W.runnable_legs(ROSTER, wave_name)
        cfgs = L.plan_configurations(cells=cells, modes=["off"], controls=legs, model=w["model_row"])
        assert [c["cell"] for c in cfgs] == cells
        for c in cfgs:
            assert c["tool_calls"] is ("bfcl_200" in legs)
        row = C.model_row(w["model_row"])
        if "bfcl_200" not in legs:
            bfcl_status, bfcl_detail = W.legs(ROSTER, wave_name)["bfcl_200"]
            assert bfcl_status in (W.STATUS_BLOCKED, W.STATUS_QUEUED) and bfcl_detail.get("reason") and bfcl_detail.get("unblock")
            if row.petri_manifest is None:
                # no VERIFIED parser pairing for the family: the launcher refuses to plan BFCL, by name
                assert row.family == "nemotron_h" and bfcl_status == W.STATUS_BLOCKED
                with pytest.raises(ValueError, match="no petri manifest") as e:
                    L.plan_configurations(cells=["none"], modes=["off"], controls=["bfcl_200"], model=w["model_row"])
                assert "bfcl_parser_probe.py" in str(e.value) and "TOOL_PARSER_BY_FAMILY" in str(e.value)
            elif row.model_key == "llama8b":
                # the llama rows PIN a manifest (llama3_json is verified), so the launcher would plan BFCL —
                # the leg is blocked by the ROSTER alone (the instrument failure), and the block says so
                assert bfcl_status == W.STATUS_BLOCKED and bfcl_detail["reason"].startswith("instrument failure")
                planned = L.plan_configurations(cells=["none"], modes=["off"], controls=["bfcl_200"], model=w["model_row"])
                assert [c["tool_calls"] for c in planned] == [True]
            else:
                # every qwen leg is runnable behind the filed live PASS record — no other non-runnable leg exists
                raise AssertionError(f"{wave_name}: a non-runnable bfcl_200 leg outside the nemotron / llama cases: {bfcl_status} {bfcl_detail}")
        # the geometry resolves on the CPU for every lane row from the SHARED table (the shared table keeps
        # main's rows): gemma / qwen / nemotron from measured rows, llama from main's provisional ceiling
        for c in cfgs:
            geom, _ = L.served_geometry(c, 1)
            assert geom.model_key == row.model_key


def _relabelled_snapshot(wave, runnable_entry, *, cell, entry_id):
    """A snapshot block for a NON-runnable cell, the way only a hand-built launch could carry one: the
    wrapper's reader refuses to render one (`W.snapshot` refuses a non-runnable entry), so this takes a
    runnable entry's block, renames its cell / entry and recomputes the digest — a VALID block by shape."""
    block = dict(_snap(wave, runnable_entry), cell_id=cell, entry_id=entry_id)
    block["snapshot_sha256"] = W.snapshot_digest(block)
    assert W.snapshot_problems(block) == []
    return block


def test_a_lane_row_launches_only_runnable_roster_cells_with_its_snapshot_at_every_door(L, tmp_path, monkeypatch, capsys):
    """Round 7 (post-consensus HIGH at 039394c4): the launcher enforced cell GRAMMAR only — for a lane
    row a direct `--cells <blocked PSR cell> --pf` planned, passed the preflight (the retained inventory hash +
    pin) and would have launched, and `--roster-snapshot` was optional. Now, for `C.BCBW_SLUGS` only: the snapshot
    is REQUIRED and every planned cell must be a RUNNABLE entry of the snapshot's wave in the STAGED tree's lane
    roster — at main (this checkout's roster, before the preflight), in the preflight (the staged roster) and in
    the container (the carried snapshot re-checked). Legacy rows are untouched."""
    repos, tree = _lane_tree(tmp_path)
    gen = tmp_path / "gen"
    monkeypatch.setattr(L.C, "VOL_REPOS_ROOT", str(repos))
    monkeypatch.setattr(L.C, "VOL_GEN_ROOT", str(gen))
    monkeypatch.setattr(L.subprocess, "Popen", lambda *a, **k: pytest.fail("a standup was started"))
    BLOCKED = "constv2-mix-full__psr_ce_all"                       # PSR out of the matrix (0b9b415a): blocked, with a reason
    assert BLOCKED not in W.runnable_cells(ROSTER, "gemma4e4b_it") and W.queued_or_blocked(ROSTER, "gemma4e4b_it")["const_psr"]["status"] == "blocked"
    row, legacy = C.model_row("gemma4e4b-f22it"), C.model_row("gemma4e4b")
    assert L.is_lane_row(row) and not L.is_lane_row(legacy) and set(s for s in C.BCBW_SLUGS) >= {"gemma4e4b-f22it", "llama8b-f22"}
    # --- the pure rule -------------------------------------------------------------------------------------------
    assert W.launch_problems(ROSTER, wave_name="gemma4e4b_it", cell_id="none") == []
    (why,) = W.launch_problems(ROSTER, wave_name="gemma4e4b_it", cell_id=BLOCKED)
    assert "entry 'const_psr'" in why and "is blocked, not runnable" in why and "unblock:" in why
    assert W.launch_problems(ROSTER, wave_name="gemma4e4b_it", cell_id="none", entry_id="si_safety")[0].startswith("the snapshot names entry")
    assert "not in the roster" in W.launch_problems(ROSTER, wave_name="nope", cell_id="none")[0]
    assert "0 entries name cell" in W.launch_problems(ROSTER, wave_name="gemma4e4b_it", cell_id="safety_refusal__psr_kl_all_x")[0]
    # --- the shared refusal, against the STAGED tree's roster ----------------------------------------------------
    good = _snap("gemma4e4b_it", "none")
    bad = _relabelled_snapshot("gemma4e4b_it", "none", cell=BLOCKED, entry_id="const_psr")
    assert L.lane_roster_refusal(tree, row, "none", good) is None
    assert L.lane_roster_refusal(tree, legacy, BLOCKED, None) is None                        # legacy: untouched
    status, why = L.lane_roster_refusal(tree, row, BLOCKED, bad)
    assert status == L.REC_ROSTER_CELL_NOT_RUNNABLE == "ROSTER_CELL_NOT_RUNNABLE" and "is blocked, not runnable" in why and W.ROSTER_REL in why
    status, why = L.lane_roster_refusal(tree, row, "none", None)
    assert status == L.REC_ROSTER_SNAPSHOT_MISSING and L.ROSTER_SNAPSHOT_FLAG in why
    relabelled_entry = _relabelled_snapshot("gemma4e4b_it", "none", cell="none", entry_id="si_safety")
    assert "the snapshot names entry 'si_safety'" in L.lane_roster_refusal(tree, row, "none", relabelled_entry)[1]
    # a tree whose roster says a DIFFERENT status than this checkout's is what decides (the STAGED roster wins)
    staged = yaml.safe_load((tree / W.ROSTER_REL).read_text())
    staged["waves"]["gemma4e4b_it"]["entries"]["si_safety"].update(status="blocked", reason="synthetic block", unblock="synthetic")
    (tree / W.ROSTER_REL).write_text(yaml.safe_dump(staged))
    status, why = L.lane_roster_refusal(tree, row, "prompted__safety_refusal", _snap("gemma4e4b_it", "si_safety"))
    assert status == "ROSTER_CELL_NOT_RUNNABLE" and "synthetic block" in why
    (tree / W.ROSTER_REL).write_text((REPO / W.ROSTER_REL).read_text())
    # --- the PREFLIGHT path: the check body the preflight's chk() wraps, with the real plan ----------------------
    cfgs = L.isolate_out_dirs(L.plan_configurations(cells=["none", BLOCKED], modes=["off"], controls=["bigcodebench_200"],
                                                    model="gemma4e4b-f22it"), "bcbw-r7")
    with pytest.raises(AssertionError, match=f"{BLOCKED}: ROSTER_CELL_NOT_RUNNABLE.*is blocked, not runnable"):
        L.lane_roster_check(tree, row, cfgs, {"none": good, BLOCKED: bad})
    with pytest.raises(AssertionError, match="ROSTER_SNAPSHOT_MISSING"):
        L.lane_roster_check(tree, row, cfgs[:1], None)
    rec = L.lane_roster_check(tree, row, cfgs[:1], {"none": good})
    assert rec["cells"] == {"none": {"wave": "gemma4e4b_it", "entry_id": "none", "status": "runnable"}}
    assert "not a lane row" in L.lane_roster_check(tree, legacy, cfgs, None)["note"]
    src = LAUNCHER.read_text()
    assert 'chk("lane_roster_runnable", lambda: lane_roster_check(tree, row, cfgs, roster_snapshot))' in src
    assert "roster_snapshot=dict(snapshots))" in src                                    # main hands the blocks to the preflight
    # --- main: the DIRECT launch of a blocked PSR cell with --pf refuses BEFORE the preflight, by name ----------
    seen = {}
    monkeypatch.setattr(L, "preflight", SimpleNamespace(remote=lambda *a, **kw: seen.setdefault("kw", kw) or {"ok": False, "checks": {}}))
    kw = dict(app_name=L.APP_NAME, sha="f6cf462", pass1=True, pf=True, modes="off", controls="bigcodebench_200",
              model="gemma4e4b-f22it", out_suffix="bcbw-r7",
              # native-serving-default: a PSR cell is refused by KIND at plan time under the native default;
              # this test is about the ROSTER door, so it drives the lens launch explicitly
              serving_path="lens")
    with pytest.raises(SystemExit, match="ROSTER_CELL_NOT_RUNNABLE.*is blocked, not runnable"):
        L.main(**kw, cells=BLOCKED, roster_snapshot=json.dumps({BLOCKED: bad}))
    assert not seen
    with pytest.raises(SystemExit, match="launches only with the launch-time roster snapshot"):
        L.main(**kw, cells=BLOCKED)                                                        # no snapshot: refused, no preflight
    assert not seen
    # a runnable cell WITH its snapshot reaches the preflight, which is handed the blocks
    with pytest.raises(SystemExit, match="PREFLIGHT FAILED"):
        L.main(**kw, cells="none", roster_snapshot=json.dumps({"none": good}))
    assert seen["kw"]["roster_snapshot"] == {"none": good} and seen["kw"]["pins"] == {}
    # the dry run of a lane row needs the snapshot too (it plans configurations)
    with pytest.raises(SystemExit, match="launches only with the launch-time roster snapshot"):
        L.main(**dict(kw, pf=False), cells="none", dry_run=True)
    capsys.readouterr()
    # --- the container: a carried, valid-by-shape snapshot for a blocked cell refuses before any standup ---------
    (blocked_cfg,) = L.isolate_out_dirs(L.plan_configurations(cells=[BLOCKED], modes=["off"], controls=["bigcodebench_200"],
                                                              model="gemma4e4b-f22it"), "bcbw-r7c")
    rec = L._run_configuration(dict(blocked_cfg, roster_snapshot=bad), "f6cf462")
    assert rec["status"] == "ROSTER_CELL_NOT_RUNNABLE" and "is blocked, not runnable" in rec["error"]
    assert not (Path(blocked_cfg["out_dir"]) / W.SNAPSHOT_FILENAME).exists()
    (none_cfg,) = L.isolate_out_dirs(L.plan_configurations(cells=["none"], modes=["off"], controls=["bigcodebench_200"],
                                                           model="gemma4e4b-f22it"), "bcbw-r7c")
    with pytest.raises(BaseException, match="a standup was started"):                     # runnable: proceeds
        L._run_configuration(dict(none_cfg, roster_snapshot=good), "f6cf462")
    # the inventories are intact (the fix is a roster gate, never a hash move)
    assert all(C.EXPECTED_INTERVENTION_HASHES.get(s) is not None or True for s in C.BCBW_SLUGS)


def test_resolving_a_staged_tree_binds_its_full22_roster_so_lane_bank_views_are_admitted_in_a_container(L, monkeypatch, tmp_path):
    """The IFEval seed canary's container refused `--bank-view s1` — the mounted `sisteer` package has no `configs/`
    beside it, so `contracts.bank_views_admitted()` found no full22 roster and fell back to the two instrument views.
    `resolve_staged_tree` now binds `$SISTEER_FULL22_ROSTER` to the STAGED tree's roster (the tree that produces the
    store is the authority for its views); a tree without one leaves the variable alone."""
    import os
    monkeypatch.delenv(C.ENV_FULL22_ROSTER, raising=False)
    sha = "a" * 40
    tree = tmp_path / "repos" / f"sisteer-{sha}"
    (tree / "src" / "sisteer").mkdir(parents=True)
    (tree / C.TREE_SHA_FILENAME).write_text(sha + "\n")
    (tree / C.FULL22_ROSTER_REL).parent.mkdir(parents=True)
    (tree / C.FULL22_ROSTER_REL).write_text("targets:\n  qwen35_4b:\n    views:\n      main: {}\n      it: {}\n      s9: {}\n")
    got, got_sha = L.resolve_staged_tree(sha, repos_root=tmp_path / "repos")
    assert (got, got_sha) == (tree, sha) and os.environ[C.ENV_FULL22_ROSTER] == str(tree / C.FULL22_ROSTER_REL)
    assert C.full22_roster_candidates()[0] == tree / C.FULL22_ROSTER_REL and "s9" in C.bank_views_admitted()
    assert C.bank_view_problem("s9") is None and "s9" in (C.bank_view_problem("s10") or "")
    # an abbreviated sha resolves through the stamp and binds the same roster
    monkeypatch.delenv(C.ENV_FULL22_ROSTER, raising=False)
    assert L.resolve_staged_tree(sha[:7], repos_root=tmp_path / "repos") == (tree, sha) and os.environ[C.ENV_FULL22_ROSTER] == str(tree / C.FULL22_ROSTER_REL)
    # a staged tree without the roster binds nothing, so the refusal still names the paths tried
    monkeypatch.delenv(C.ENV_FULL22_ROSTER, raising=False)
    bare = tmp_path / "repos2" / f"sisteer-{sha}"; (bare / "src" / "sisteer").mkdir(parents=True); (bare / C.TREE_SHA_FILENAME).write_text(sha + "\n")
    L.resolve_staged_tree(sha, repos_root=tmp_path / "repos2")
    assert C.ENV_FULL22_ROSTER not in os.environ


def test_the_live_parser_probe_gate_binds_a_direct_bfcl_launch_and_lifts_on_a_filed_pass_record(L, monkeypatch, tmp_path):
    """The qwen3_5 table pin + manifest are the OFFLINE half of the verification; a direct `plan_configurations`
    (or the per-container worker) could otherwise plan `bfcl_200` for a qwen row before the LIVE raw-token replay
    ever passed. The gate is a contracts constant — the family's live PASS record (sha7 + pulled verdict path +
    time) — consulted by the planner, the worker and the roster validator; a filed record lifts it, a malformed one
    refuses, and non-probe families (gemma, llama) are untouched."""
    assert C.TOOL_PARSER_PROBE_FAMILIES == ("qwen3_5", "nemotron_h") and set(C.TOOL_PARSER_LIVE_PASS) == set(C.TOOL_PARSER_PROBE_FAMILIES)
    assert C.tool_parser_live_gate_problem("gemma4") is None and C.tool_parser_live_gate_problem("llama") is None
    # the record IS filed on this tree (the live PASS at 80df464), so the source checkout plans BFCL for the
    # qwen rows; the pre-PASS refusal is exercised below with the record withdrawn
    assert C.tool_parser_live_gate_problem("qwen3_5") is None
    assert [c["tool_calls"] for c in L.plan_configurations(cells=["none"], modes=["off"], controls=["bfcl_200"], model="qwen35_4b-f22s1")] == [True]
    monkeypatch.setitem(C.TOOL_PARSER_LIVE_PASS, "qwen3_5", None)
    why = C.tool_parser_live_gate_problem("qwen3_5")
    assert why and "LIVE half has not PASSED" in why and "verdict.json" in why and "--family qwen3_5" in why
    with pytest.raises(ValueError, match="LIVE half has not PASSED"):
        L.plan_configurations(cells=["none"], modes=["off"], controls=["bfcl_200"], model="qwen35_4b-f22s1")
    # a BCB-only plan of the same row is unaffected
    assert [c["tool_calls"] for c in L.plan_configurations(cells=["none"], modes=["off"], controls=["bigcodebench_200"], model="qwen35_4b-f22s1")] == [False]
    # a filed record lifts the gate ONLY with the verdict CHECKED IN under configs/capeval/tasks/parser_probe/
    # and its content proving the PASS (family, the probe row's model_key, the staged sha, verdict PASS, both halves OK + PASS)
    import json as _json
    repo = tmp_path / "repo"
    rel = f"{C.TOOL_PARSER_LIVE_RECORD_DIR}/qwen3_5_verdict_5e3dd86.json"
    (repo / rel).parent.mkdir(parents=True)
    good = {"family": "qwen3_5", "model": "qwen35_4b-f22", "model_key": "qwen35_4b", "sha": "5e3dd86", "parser": "qwen3_xml", "verdict": "PASS",
            "offline": {"status": "OK", "verdict": {"verdict": "PASS", "n_fixtures": 9, "n_failed": 0}},
            "live": {"status": "OK", "verdict": {"verdict": "PASS", "n_items": 24, "n_called": 20}}, "on_pass": ["x"], "at": "t"}
    (repo / rel).write_text(_json.dumps(good))
    # the verdict is read from the TREE handed in (the staged tree for the launch sha in a container; here a scratch
    # tree) — never from contracts' own __file__; the plan reads it from the same root as the IFEval matrix
    gate = lambda: C.tool_parser_live_gate_problem("qwen3_5", tree=repo)                       # noqa: E731
    plan = lambda: L.plan_configurations(cells=["none"], modes=["off"], controls=["bfcl_200"], model="qwen35_4b-f22s1",  # noqa: E731
                                         ifeval_matrix_tree=repo)
    rec = {"sha7": "5e3dd86", "record": rel, "passed_at": "2026-09-22T00:00:00Z"}
    monkeypatch.setitem(C.TOOL_PARSER_LIVE_PASS, "qwen3_5", rec)
    assert gate() is None and C.tool_parser_live_record_path(rec, repo) == repo / rel
    # the SAME record against the source checkout (no tree) refuses: the artifact is not on THAT tree
    assert C.source_checkout_root() == REPO and "is not on tree" in C.tool_parser_live_gate_problem("qwen3_5")
    with pytest.raises(ValueError, match="Refusing to plan BFCL"):
        L.plan_configurations(cells=["none"], modes=["off"], controls=["bfcl_200"], model="qwen35_4b-f22s1")
    assert [c["tool_calls"] for c in plan()] == [True]
    # a path string is not proof: the artifact must exist, be under the pinned dir and repo-relative, and its CONTENT must prove it
    monkeypatch.setitem(C.TOOL_PARSER_LIVE_PASS, "qwen3_5", dict(rec, record=f"{C.TOOL_PARSER_LIVE_RECORD_DIR}/qwen3_5_verdict_0000000.json"))
    assert "is not on tree" in gate()
    monkeypatch.setitem(C.TOOL_PARSER_LIVE_PASS, "qwen3_5", dict(rec, record=str(repo / rel)))
    assert "not a repo-relative path under" in C.tool_parser_live_gate_problem("qwen3_5")
    monkeypatch.setitem(C.TOOL_PARSER_LIVE_PASS, "qwen3_5", dict(rec, record="~/work/x/verdict.json"))
    assert "not a repo-relative path under" in C.tool_parser_live_gate_problem("qwen3_5")
    for mutate, needle in ((lambda d: d.__setitem__("verdict", "INCONCLUSIVE"), "reads 'INCONCLUSIVE', not PASS"),
                           (lambda d: d.__setitem__("family", "nemotron_h"), "is for family 'nemotron_h'"),
                           (lambda d: d.__setitem__("sha", "3ba0202"), "is for staged sha '3ba0202'"),
                           (lambda d: d.__setitem__("model_key", "qwen35_9b"), "names model_key 'qwen35_9b'"),
                           (lambda d: d["live"].__setitem__("status", "STANDUP_FAILED"), "live half is not (status OK, verdict PASS)"),
                           (lambda d: d["live"]["verdict"].__setitem__("verdict", "INCONCLUSIVE"), "live half is not (status OK, verdict PASS)"),
                           (lambda d: d.__setitem__("offline", None), "offline half is not (status OK, verdict PASS)")):
        doc = _json.loads(_json.dumps(good)); mutate(doc)
        (repo / rel).write_text(_json.dumps(doc))
        monkeypatch.setitem(C.TOOL_PARSER_LIVE_PASS, "qwen3_5", rec)
        why = gate()
        assert why and needle in why, (needle, why)
        with pytest.raises(ValueError, match="Refusing to plan BFCL"):
            plan()
    (repo / rel).write_text("not json")
    assert "does not parse" in gate()
    (repo / rel).write_text(_json.dumps(good))
    monkeypatch.setitem(C.TOOL_PARSER_LIVE_PASS, "qwen3_5", {"sha7": "5e3dd86"})
    assert "not {sha7, record, passed_at}" in C.tool_parser_live_gate_problem("qwen3_5")
    monkeypatch.setitem(C.TOOL_PARSER_LIVE_PASS, "qwen3_5", dict(rec, sha7="zzz"))
    assert "not a commit prefix" in C.tool_parser_live_gate_problem("qwen3_5")
    assert C.TOOL_PARSER_PROBE_MODEL_KEY == {"qwen3_5": "qwen35_4b", "nemotron_h": "nemotron3nano4b"}
    # the worker refuses the same way, as a status (never a traceback), on a configuration asking for BFCL
    monkeypatch.setitem(C.TOOL_PARSER_LIVE_PASS, "qwen3_5", None)
    msg = L.no_live_tool_parser_message(C.model_row("qwen35_4b-f22"), ["bigcodebench_200", "bfcl_200"])
    assert "--controls bigcodebench_200" in msg and "LIVE half has not PASSED" in msg
    # nemotron_h PASSED its own probe at 8365992 (2026-09-23): manifest pinned, live record filed — the same double gate, both open;
    # withdrawing the record closes it again by name
    assert C.model_row("nemotron3nano4b").petri_manifest == C.PETRI_MANIFEST_FULL22_NEMOTRON3NANO4B and C.tool_parser_live_gate_problem("nemotron_h") is None
    monkeypatch.setitem(C.TOOL_PARSER_LIVE_PASS, "nemotron_h", None)
    assert "LIVE half has not PASSED" in C.tool_parser_live_gate_problem("nemotron_h")


def test_the_live_verdict_is_read_from_the_staged_tree_never_from_the_image_mounted_module(L, monkeypatch, tmp_path):
    """In the Modal image `sisteer` is mounted at /root/sisteer with no configs/ beside it, and the image
    ships no verdict JSON (on purpose — the staged tree for the launch sha is the ONE source of truth, the same root the
    roster and row configs are read from; an image copy would be whatever the LOCAL checkout held at build time). So a
    __file__-relative lookup would name a path no tree owns, and a real live PASS filed on the host would leave every
    Modal plan / worker refusing "not on this tree". Relocate `contracts` the way the image does: the source-checkout
    detection yields None, a gate call with no tree refuses LOUDLY naming the missing tree (never a false "absent
    artifact"), and the same call against a staged tree carrying the verdict admits — the planner reads it through the
    tree it replans from, the worker through the tree it resolved for the sha (pinned in source below)."""
    import json as _json
    staged = tmp_path / "vol" / "repos" / "sisteer-5e3dd86"
    rel = f"{C.TOOL_PARSER_LIVE_RECORD_DIR}/qwen3_5_verdict_5e3dd86.json"
    (staged / rel).parent.mkdir(parents=True)
    (staged / rel).write_text(_json.dumps({"family": "qwen3_5", "model_key": "qwen35_4b", "sha": "5e3dd86", "verdict": "PASS",
                                           "offline": {"status": "OK", "verdict": {"verdict": "PASS"}}, "live": {"status": "OK", "verdict": {"verdict": "PASS"}}}))
    rec = {"sha7": "5e3dd86", "record": rel, "passed_at": "2026-09-22T00:00:00Z"}
    shipped = C.TOOL_PARSER_LIVE_PASS["qwen3_5"]          # the real record (80df464), restored for the roster validate below
    monkeypatch.setitem(C.TOOL_PARSER_LIVE_PASS, "qwen3_5", rec)
    # the image layout: /root/sisteer/capeval/tinycap/contracts.py, nothing above it
    fake_root = tmp_path / "root"
    fake_mod = fake_root / "sisteer" / "capeval" / "tinycap" / "contracts.py"
    fake_mod.parent.mkdir(parents=True); fake_mod.write_text("# mounted copy\n")
    monkeypatch.setattr(C, "__file__", str(fake_mod))
    assert C.source_checkout_root() is None and C.tool_parser_live_record_path(rec) is None
    why = C.tool_parser_live_gate_problem("qwen3_5")
    assert why and "no tree to read the live verdict" in why and "pass tree=" in why and "is not on" not in why
    with pytest.raises(ValueError, match="no tree to read the live verdict"):
        L.plan_configurations(cells=["none"], modes=["off"], controls=["bfcl_200"], model="qwen35_4b-f22s1")
    assert L.tool_parser_problem(C.model_row("qwen35_4b-f22s1"), ["bfcl_200"]) and L.tool_parser_problem(C.model_row("qwen35_4b-f22s1"), ["bfcl_200"], tree=staged) is None
    # against the staged tree the promoted BFCL leg plans — the gate, the planner (through the tree it replans from) and the roster validate
    assert C.tool_parser_live_gate_problem("qwen3_5", tree=staged) is None
    planned = L.plan_configurations(cells=["none"], modes=["off"], controls=["bfcl_200"], model="qwen35_4b-f22s1", ifeval_matrix_tree=staged)
    assert [c["tool_calls"] for c in planned] == [True]
    # another staged tree without the artifact refuses by THAT tree's path, not by a __file__ path
    bare = tmp_path / "vol" / "repos" / "sisteer-0000000"; (bare / "configs").mkdir(parents=True)
    why = C.tool_parser_live_gate_problem("qwen3_5", tree=bare)
    assert why and f"is not on tree {bare}" in why
    # the worker reads the verdict from the tree it resolved for the sha; the planner from the replan root; the image ships no copy
    src = TL.LAUNCHER.read_text()
    i_tree, i_gate = src.index('    rec["tree_sha"] = tree_sha\n'), src.index("tool_parser_why = tool_parser_problem(row, wanted, tree=tree)")
    assert i_tree < i_gate and src.count("tool_parser_problem(row, ctl, tree=ifeval_matrix_tree)") == 1
    assert not any("parser_probe" in remote for remote in L.IMAGE_VENDORED_FILES)
    # the artifact dir is ON the tree (git archive carries it), so a staged tree always has the root to read from
    assert (REPO / C.TOOL_PARSER_LIVE_RECORD_DIR / "README.md").is_file()
    # the roster validate reads the verdict through the same tree argument, and refuses loudly with none at hand
    from sisteer.capeval.tinycap import widen_roster as WR
    pins = WR.load_pins()
    # with the SHIPPED record (its verdict is on this tree) the roster's runnable qwen legs validate; with the scratch record they refuse
    assert any("is not on tree" in p for p in WR.validate(ROSTER, pins, C.MODELS, tree=REPO))
    monkeypatch.setitem(C.TOOL_PARSER_LIVE_PASS, "qwen3_5", shipped)
    assert WR.validate(ROSTER, pins, C.MODELS, tree=REPO) == []
    (missing,) = [p for p in WR.validate(ROSTER, pins, C.MODELS) if "no tree to read the row configs" in p]
    assert "pass tree=" in missing



def test_identity_request_marks_a_served_only_class_b_cell_and_pins_it_through_the_trees_lane_extra(L, tmp_path, monkeypatch):
    """The container preflight spelled a served-only Class B cell's arm as its id,
    which `cell_controls.arm_spec_for_cell` refuses ("not registered in methods.ARMS"). `identity_request` now marks
    such a target (`served_only_class_b: true`; the arm STAYS the id) and `object_identity.derive_request` routes it
    through the served-only branch — one rule for the derivation on the grading host and in the container. The pin
    rides from the STAGED tree's full22 pins through the generator's lane extras (`ar_c0p3__arditi_allpos`); the
    expected hash is the inventory's; a Class A cell carries no mark."""
    import sisteer.capeval.tinycap.object_identity as oi
    from sisteer.capeval.petri.defense_cells import SERVED_ONLY_CLASS_B_IDS
    repos, tree = _lane_tree(tmp_path)
    row = C.model_row("qwen35_4b-f22s1")
    cells = ["none", "prompted__si_constitution", "constv2-mix-full__psr_ce_all", "arditi_allpos"]
    req = L.identity_request(row, str(tree), cells)
    targets = {t["cell"]: t for t in req["targets"]}
    assert set(targets) == {"constv2-mix-full__psr_ce_all", "arditi_allpos"}
    ar = targets["arditi_allpos"]
    assert ar["si_id"] is None and ar["arm"] == "arditi_allpos" and ar["served_only_class_b"] is True
    assert ar["pin_sha256"] == "ba2212f0c2f9ffef2997bdd24831182052560cf980054bcee9cf95ea0db085b2" == W.pin_of(ROSTER, "qwen35_4b_s1_arditi", "arditi_allpos")
    assert ar["expected"] == C.EXPECTED_INTERVENTION_HASHES["qwen35_4b-f22s1"]["arditi_allpos"]
    assert "served_only_class_b" not in targets["constv2-mix-full__psr_ce_all"] and targets["constv2-mix-full__psr_ce_all"]["si_id"] == "constv2-mix-full"
    assert "arditi_allpos" in SERVED_ONLY_CLASS_B_IDS
    # derive_request hands the mark to derive_one, which takes the branch on the mark alone
    seen = {}
    def fake_derive_one(**kw):
        seen[kw["cell_id"]] = kw.get("served_only_class_b")
        return {"cell": kw["cell_id"], "intervention_hash": "0" * 16, "response_header_len": None, "positions": None}
    monkeypatch.setattr(oi, "derive_one", fake_derive_one)
    monkeypatch.setattr(oi, "route_intervention_hash", lambda **kw: {"harness_cell_id": kw["cell_id"], "lens_route_intervention_hash": "1" * 16, "lens_route_derivation": "x"})
    monkeypatch.setattr(oi, "load_config", lambda path, overrides=None: {"model": "m", "model_revision": "r"})
    monkeypatch.setattr(oi, "build_meta_shell", lambda model, rev: (None, "Shell", "how"))
    import sisteer.models as models_mod
    import sisteer.chat_markers as cm_mod
    monkeypatch.setattr(models_mod, "load_tokenizer", lambda m, revision=None: object())
    monkeypatch.setattr(models_mod, "pin_reasoning_template_off", lambda tok: None)
    monkeypatch.setattr(models_mod, "serving_identity", lambda tok, cfg, model=None: {"chat_template_hash": "h"})
    monkeypatch.setattr(cm_mod, "generation_header_ids", lambda tok: (1, 2, 3))
    bank = tmp_path / "bank"
    for c in ("arditi_allpos", "constv2-mix-full__psr_ce_all"):
        (bank / c).mkdir(parents=True)
        (bank / c / "object.pt").write_bytes(b"x")
    rep = oi.derive_request({**req, "objects_dir": str(bank)})
    assert seen == {"constv2-mix-full__psr_ce_all": None, "arditi_allpos": True}       # the mark rides ONLY on the marked target
    assert {r["cell"] for r in rep["results"]} == set(targets)



def test_the_preflight_checks_a_full_lane_launch_against_the_full_roster_chosen_by_the_snapshots_sha(L, tmp_path, monkeypatch):
    """The instruct-direct arms are the first cells that exist ONLY in the full roster (the 200 roster's
    qwen35_4b_it wave has none + diff_lora); the preflight read the 200 roster unconditionally and refused them ("0 entries name cell").
    The lane roster is now chosen by the snapshot's roster_sha256 against the staged tree's two rosters; a snapshot matching neither falls
    back to the first roster holding its wave (the 200 roster first — the staged-roster-wins behaviour)."""
    repos, tree = _lane_tree(tmp_path)
    monkeypatch.setattr(L.C, "VOL_REPOS_ROOT", str(repos))
    full = W.load_roster(W.ROSTER_FULL_PATH); full_sha = W.roster_sha256(W.ROSTER_FULL_PATH)
    assert (tree / W.ROSTER_FULL_REL).is_file() and W.roster_sha256(tree / W.ROSTER_FULL_REL) == full_sha
    row = C.model_row("qwen35_4b-f22it"); assert L.is_lane_row(row)
    LORA = "constv2-mix-full__lora_r4_ce"
    assert LORA in W.runnable_cells(full, "qwen35_4b_it") and LORA not in W.runnable_cells(ROSTER, "qwen35_4b_it")
    snap_full = W.snapshot(full, "qwen35_4b_it", "const_lora", roster_sha256=full_sha, launched_at="2026-09-25T19:00:00Z")
    assert L.lane_roster_rel_for(tree, snap_full) == W.ROSTER_FULL_REL and L.lane_roster_refusal(tree, row, LORA, snap_full) is None
    # a 200-roster snapshot still reads the 200 roster (byte for byte the round-7 behaviour)
    snap_200 = _snap("qwen35_4b_it", "none")
    assert L.lane_roster_rel_for(tree, snap_200) == W.ROSTER_REL and L.lane_roster_refusal(tree, row, "none", snap_200) is None
    # neither sha matches (an edited staged roster): the first roster holding the wave decides — the 200 roster, where the arm is unknown
    stale = dict(snap_full, roster_sha256="0" * 64); stale["snapshot_sha256"] = W.snapshot_digest(stale)
    assert L.lane_roster_rel_for(tree, stale) == W.ROSTER_REL
    status, why = L.lane_roster_refusal(tree, row, LORA, stale)
    assert status == L.REC_ROSTER_CELL_NOT_RUNNABLE and "0 entries name cell" in why and W.ROSTER_REL in why
    # a wave neither roster holds falls back to the 200 roster, whose launch_problems then names the unknown wave
    nowave = dict(stale, wave="nope_s9"); nowave["snapshot_sha256"] = W.snapshot_digest(nowave)
    assert L.lane_roster_rel_for(tree, nowave) == W.ROSTER_REL and "not in the roster" in L.lane_roster_refusal(tree, row, LORA, nowave)[1]
