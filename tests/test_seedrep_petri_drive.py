"""scripts/modal/seedrep_petri_drive.py -- the seedrep battery's grading host Petri drive: the six-axis control-key
grammar, the per-view parameters (3 view proxies x 8 arm-runs at width 3, the Arditi proxy x 3 seed bases at
width 4 = 84 <= 100), the schedule that REFUSES an enumeration without the prompted routes, the exact petri_run
argv with the target-decode flags, the HIGHPRIO-only auditor key through the vault mechanism (a lowprio call is
REFUSED; never a literal key), the third auditor leg on a seedrep pool, the completion verdict + idempotent
resume, the 429 throttle-down, and the container->volume path map. Every rule is pinned against a KNOWN-BAD
input. The enumeration fixtures are the REAL ``petri_run.enumerate_petri_native`` over a synthetic registry
(never a hand-written row list): a seed view enumerates 9 rows (8 arms + the derived ``none__lora0`` control),
the Arditi manifest 3."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "scripts/modal/seedrep_petri_drive.py"
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "scripts" / "modal"))

import petri_run as prun  # noqa: E402
import seedrep_launch_plan as lp  # noqa: E402
import test_petri_store_shape as shape  # noqa: E402

from sisteer.capeval import manifest as cm  # noqa: E402
from sisteer.capeval.petri import defense_cells as dc  # noqa: E402
W = "0c1925110000000000000000000000000000abcd"
WAVE = "8f14e45f-ceea-467a-9575-6e8e7c2f5b31"
HEX32 = "0123456789abcdef0123456789abcdef"
W7 = "0c19251"
#: the six-axis key a tunnel record of (WAVE, petri, s1, base, shard 0) must carry as its own `control_key`
TUNNEL_KEY = f"{W7}/{WAVE}/petri/s1/base/0"


def _load():
    """Import the drive. It imports `modal` only lazily (the transport helpers), so a bare stub suffices for the
    load and is REMOVED afterwards -- a stub left in sys.modules would shadow the real/stubbed modal other test
    modules install with setdefault (test-order pollution)."""
    had_modal = "modal" in sys.modules
    if not had_modal:
        sys.modules["modal"] = types.ModuleType("modal")
    spec = importlib.util.spec_from_file_location("seedrep_petri_drive", SRC)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["seedrep_petri_drive"] = mod   # dataclasses resolve string annotations through sys.modules[cls.__module__]
    try:
        spec.loader.exec_module(mod)
    finally:
        if not had_modal:
            sys.modules.pop("modal", None)
    return mod


@pytest.fixture(scope="module")
def d():
    return _load()


#: the lane's 8 view arms in harness spelling: none + the s-view's 5 fitted cells (the real s1 manifest's) + the two prompted routes
VIEW_CELLS = ["none", "si_constv2_mix_full__psr_ce_all", "si_constv2_mix_full__lora_r4_ce", "si_constv2ctl2_full__psr_ce_all",
              "si_constv2ctl2_full__lora_r4_ce", "si_constv2_mix_full__lora_r4_ce_minus_ctl", "prompted__constitution", "prompted__safety"]
ARDITI_CELLS = ["none", "arditi_allpos"]                 # one object, never a second
EXTRA_ARM = "si_constv2ctl2_full__lora_r4_ce_minus_ctl"   # a legal Class A id the lane does not run


def _bf(spec):
    return {k: f"# behavior {k}\nsynthetic eval bytes\n".encode() for k in cm.petri_expected_behavior_keys(spec)}


def _entry(cid: str, bank: Path) -> dc.RegistryEntry:
    """A synthetic registry entry that enumerates NATIVELY for the cell id (kinds as the lane's arms: psr on the
    residual slot, lora_all / lora_subtraction adapters, vector_all for Arditi -- all `lora`-routed, positions all)."""
    if cid == "none":
        return dc.RegistryEntry(cell_id=cid, kind="none")
    if cid.startswith("prompted__"):
        si = cid[len("prompted__"):]
        return dc.RegistryEntry(cell_id=cid, kind="prompted", si_id=si,
                                si_prefix=shape.SAFETY_PREFIX if si == "safety" else shape.CONSTITUTION_PREFIX)
    art = bank / f"{cid}.pt"
    art.write_bytes(f"synthetic-object-{cid}".encode())
    if cid == "arditi_allpos":
        return dc.RegistryEntry(cell_id=cid, kind="vector_all", si_id=None, object_path=str(art), serving_path="lora",
                                resolved_spec={"kind": "vector_all", "positions": "all", "fit_positions": "all"})
    si, method = cid[len("si_"):].split("__", 1)
    if method.startswith("psr"):
        return dc.RegistryEntry(cell_id=cid, kind="psr", si_id=si, object_path=str(art), serving_path="lora",
                                resolved_spec={"kind": "psr", "si_id": si, "layers": [0, 2], "positions": "all"})
    kind = "lora_subtraction" if "minus" in method else "lora_all"
    return dc.RegistryEntry(cell_id=cid, kind=kind, si_id=si, object_path=str(art), serving_path="lora",
                            resolved_spec={"kind": kind, "si_id": si, "layers": [0, 2], "rank": 4, "positions": "all", "response_header_len": None})


def native_cells_json(bank: Path, cells, *, audit: str) -> dict:
    """``cells.json`` EXACTLY as ``petri_run.py --serving-path lora_native print-cells`` freezes it: the REAL
    ``enumerate_petri_native`` over a synthetic registry, then ``petri_run._cells_payload`` -- so the fixture carries
    the derived ``none__lora0`` control and every row field the enumeration writes (cell_class, control_of, ...)."""
    bank.mkdir(parents=True, exist_ok=True)
    spec = shape._spec("ab" * 8, audit=audit, cells=list(cells))
    source = dc.MappingCellSource({c: _entry(c, bank) for c in cells})
    native = prun.enumerate_petri_native(spec, source, behavior_files=_bf(spec), prompt_files=shape.PROMPT_FILES)
    payload = prun._cells_payload(native.cell_set, spec, native=native)
    assert payload["absent"] == [] and payload["cells_skipped"] == []
    return payload


def _served(cells_json) -> list:
    return [r["virtual_model_name"] for r in list(cells_json["canonical"]) + list(cells_json["controls"])]


@pytest.fixture(scope="module")
def view_cells(tmp_path_factory):
    return native_cells_json(tmp_path_factory.mktemp("view-bank"), VIEW_CELLS, audit="seedrep-view-fixture")


@pytest.fixture(scope="module")
def arditi_cells(tmp_path_factory):
    return native_cells_json(tmp_path_factory.mktemp("arditi-bank"), ARDITI_CELLS, audit="seedrep-arditi-fixture")


# --------------------------------------------------------------------------- control-plane grammar

def test_dict_names_and_six_axis_keys_carry_every_axis_and_refuse_a_missing_or_slashed_one(d):
    assert d.w7_of(W) == "0c19251" and d.wave_uuid8(WAVE) == "8f14e45f"
    assert d.dict_name("0c19251", "8f14e45f", "sentinels") == "seedrep-0c19251-8f14e45f-sentinels"
    assert d.dict_name("0c19251", "8f14e45f", "accepted") == "seedrep-0c19251-8f14e45f-accepted"
    k = d.control_key("0c19251", WAVE, "petri", "s2", "base", "0", "modal_tunnel.json")
    assert k == f"0c19251/{WAVE}/petri/s2/base/0/modal_tunnel.json"
    assert k.split("/")[:6] == ["0c19251", WAVE, "petri", "s2", "base", "0"]      # W7, wave_uuid, leg, view, stage, shard
    assert d.control_key("0c19251", WAVE, "petri", "arditi", "petri-rerun", "0") == f"0c19251/{WAVE}/petri/arditi/petri-rerun/0"
    assert d.control_key("0c19251", WAVE, "static_pasttense", "s1", "8192", "3") == f"0c19251/{WAVE}/static_pasttense/s1/8192/3"
    # KNOWN-BAD: an empty axis, a slash inside an axis, a stage off the list, a bad wave uuid, a bad purpose, a slashed name
    for args in (("0c19251", WAVE, "", "s2", "base", "0"), ("0c19251", WAVE, "petri", "s2/x", "base", "0"),
                 ("0c19251", WAVE, "petri", "s2", "4096", "0"), ("0c19", WAVE, "petri", "s2", "base", "0")):
        with pytest.raises(ValueError):
            d.control_key(*args)
    with pytest.raises(ValueError):
        d.control_key("0c19251", WAVE, "petri", "s2", "base", "0", "a/b")
    with pytest.raises(SystemExit):
        d.control_key("0c19251", "not-a-uuid", "petri", "s2", "base", "0")
    with pytest.raises(ValueError):
        d.dict_name("0c19251", "8f14e45f", "tunnels")
    with pytest.raises(SystemExit):
        d.w7_of("0c192511")
    assert d.STAGES == {"": "base", "overflow-rerun": "petri-rerun"} and d.DEFAULT_SHARD == "0"
    assert (d.HEARTBEAT_NAME, d.STOP_NAME, d.TUNNEL_FILE, d.HEARTBEAT_PERIOD_S) == ("driver_heartbeat", "STOP", "modal_tunnel.json", 300)


# --------------------------------------------------------------------------- per-view parameters + widths

def test_view_params_are_the_plan_s_three_view_proxies_plus_arditi_and_the_auditor_width_is_84(d):
    s2 = d.view_params("s2")
    assert (s2.seed_bases, s2.width, s2.expected_cells, s2.expected_runs, s2.is_arditi) == ((2,), 3, 8, 8, False)
    assert s2.cfg_rel == "configs/sdfbase_full22_qwen35_4b_s2.yaml" and s2.manifest_rel == "configs/capeval/manifests/petri_full22_qwen35_4b_s2.yaml"
    assert (REPO / s2.cfg_rel).is_file() and (REPO / s2.manifest_rel).is_file()
    # F-R: the LANE bank's view (what 00_build_views.sh builds and --stage-bank uploads), never the peers' shared full22_objects
    assert s2.bank_view_rel == lp.grading_host_bank_view_rel("s2") and s2.bank_view_rel.endswith("/sdfbase/seedrep_battery/bank/qwen35_4b/cells_view/s2")
    assert s2.bank_root_rel == lp.GRADING_HOST_LANE_BANK_REL and s2.bank_root_rel.endswith("/sdfbase/seedrep_battery/bank")
    ar = d.view_params("arditi")
    assert (ar.seed_bases, ar.width, ar.expected_cells, ar.expected_runs, ar.is_arditi) == ((1, 2, 3), 4, 1, 3, True)
    assert ar.cfg_rel == "configs/seedrep/arditi_allpos_qwen35_4b_ar_c0p3_seedrep.yaml"        # rendered (absent at the base tree)
    assert ar.manifest_rel == "configs/seedrep/arditi_allpos_qwen35_4b_ar_c0p3_seedrep_manifest.yaml"
    assert ar.bank_view_rel == "arditi-stage/objects/arditi_allpos/qwen35_4b/ladder/cells_view/ar_c0p3"
    assert ar.bank_root_rel == "arditi-stage/objects/arditi_allpos/qwen35_4b/ladder" == lp.GRADING_HOST_ARDITI_LADDER_REL
    # 3 views x 8 arm-runs x 3 + 1 Arditi x 3 seed-runs x 4 = 72 + 12 = 84 <= the standing sonnet highprio ceiling 100
    assert d.total_auditor_width() == 84 and d.assert_within_auditor_ceiling() == 84 and d.AUDITOR_CEILING == 100
    with pytest.raises(SystemExit, match="exceed the highprio ceiling"):
        d.assert_within_auditor_ceiling(ceiling=80)
    with pytest.raises(SystemExit, match="--view"):
        d.view_params("s4")
    # out-root tags: a view engine runs at its OWN seed; Arditi has one tag per seed base
    assert d.run_tag("s2", 2) == "qwen35_4b_s2" and d.run_tag("arditi", 3) == "qwen35_4b_arditi_s3"
    with pytest.raises(ValueError):
        d.run_tag("s2", 1)
    with pytest.raises(ValueError):
        d.run_tag("arditi", 4)
    home = Path("/home/testuser")
    # every writable root carries (window, wave_uuid8, attempt_id) -- the petri_seedrep_<window>_<tag> root widened by the wave + attempt axes
    assert d.out_root_for(home, "0c19251", "8f14e45f", "ta-1.uuid", "qwen35_4b_s2") == \
        Path("/home/testuser/.local/state/petri_native/petri_seedrep_0c19251_8f14e45f_ta-1.uuid_qwen35_4b_s2")
    assert d.state_dir_for(home, "0c19251", "8f14e45f", "arditi", "overflow-rerun") == Path("/home/testuser/.local/state/seedrep_battery/petri/0c19251/8f14e45f/arditi_rerun")
    assert d.attempt_state_dir(d.state_dir_for(home, "0c19251", "8f14e45f", "s1", ""), "ta-1.uuid") == \
        Path("/home/testuser/.local/state/seedrep_battery/petri/0c19251/8f14e45f/s1/attempts/ta-1.uuid")
    for bad in (("0c19", "8f14e45f"), ("0c19251", "8f14")):
        with pytest.raises(ValueError):
            d.out_root_for(home, *bad, "ta-1.uuid", "qwen35_4b_s2")
        with pytest.raises(ValueError):
            d.state_dir_for(home, *bad, "s1", "")
    with pytest.raises(SystemExit, match="not a path-safe attempt id"):
        d.out_root_for(home, "0c19251", "8f14e45f", "ta/1", "qwen35_4b_s2")
    # the bundle hash is the s-view manifests'
    for v in ("s1", "s2", "s3"):
        assert f"scenario_bundle_hash: {d.BUNDLE_HASH}" in (REPO / d.view_params(v).manifest_rel).read_text()


def test_fr_view_params_resolve_the_lane_bank_view_never_the_peers_shared_bank(d, tmp_path):
    """The drive derived a seed view's path on the grading host from the shared bank
    (``$STAGE/sdfbase/full22_objects``, which holds only other users' views) and refused every battery view. KNOWN-BAD:
    a fake $HOME with a lane bank carrying ``cells_view/s1`` and an EMPTY shared bank -- the old rel path is not a
    directory (the old derivation refused), the new one resolves the lane view; with the view absent from both banks
    the drive refuses BY NAME (the view, the lane root, the operator step), never falls back to another bank."""
    home = tmp_path
    lane_view = home / lp.GRADING_HOST_LANE_BANK_REL / "qwen35_4b" / "cells_view" / "s1"
    lane_view.mkdir(parents=True)
    (home / "stage/sdfbase/full22_objects/qwen35_4b/cells_view").mkdir(parents=True)    # the peers' bank: no battery view
    vp = d.view_params("s1")
    old_rel = "stage/sdfbase/full22_objects/qwen35_4b/cells_view/s1"
    assert not (home / old_rel).is_dir() and vp.bank_view_rel != old_rel
    assert d.resolve_grading_host_bank_view(vp, home) == lane_view
    assert d.resolve_grading_host_bank_view(vp, home).is_relative_to(home / lp.GRADING_HOST_LANE_BANK_REL)
    lane_view.rmdir()
    with pytest.raises(SystemExit) as ei:
        d.resolve_grading_host_bank_view(vp, home)
    msg = str(ei.value)
    assert "no grading host bank view 's1'" in msg and str(home / lp.GRADING_HOST_LANE_BANK_REL) in msg and "00_build_views.sh" in msg
    assert "full22_objects" not in msg
    # s2 / s3 derive the same way; the derivation is lp's, so the stager and the drive cannot drift apart
    for v in d.SEED_VIEWS:
        assert d.view_params(v).bank_view_rel == lp.grading_host_bank_view_rel(v) == f"{lp.GRADING_HOST_LANE_BANK_REL}/{lp.bank_view_subpath(v)}"
    with pytest.raises(ValueError, match="slash-free"):
        lp.bank_view_subpath("s1/../main")


def test_fr_the_stager_and_the_drive_share_one_grading_host_bank_derivation():
    """The stager (``seedrep_served.py``: the ``bank_local`` default + ``stage_bank_arcnames``) composes the same root and subpath
    through ``lp``; neither file carries its own ``$STAGE/sdfbase`` view literal (the defect was exactly a second literal)."""
    served = (REPO / "scripts/modal/seedrep_served.py").read_text()
    assert "bank_local: str = str(Path.home() / lp.GRADING_HOST_LANE_BANK_REL)" in served
    assert "arditi_local: str = str(Path.home() / lp.GRADING_HOST_ARDITI_LADDER_REL)" in served
    assert 'out.append((f"bank:{lp.bank_view_subpath(v)}", f"sdfbase/full22/{lp.bank_view_subpath(v)}"))' in served
    stage_root = "/".join(lp.GRADING_HOST_LANE_BANK_REL.split("/")[:2])   # the bank's stage root, as lp spells it
    assert f'Path.home() / "{stage_root}' not in served
    drive_src = SRC.read_text()
    code = "\n".join(l for l in drive_src.splitlines() if not l.lstrip().startswith("#"))
    # the only admissible spellings of the shared bank in the drive are prose (docstrings); no f-string / literal path composes it
    assert f'f"{stage_root}' not in code and f'"{stage_root}/full22_objects' not in code
    assert "bank_root_rel=lp.GRADING_HOST_LANE_BANK_REL, bank_view_rel=lp.grading_host_bank_view_rel(view)" in drive_src



# --------------------------------------------------------------------------- the schedule

def test_a_view_schedules_exactly_its_eight_lane_arms_out_of_the_nine_row_native_enumeration(d, view_cells, tmp_path):
    rows = d.leg_rows(view_cells)
    assert len(rows) == 9 and (rows[8]["cell_id"], rows[8]["cell_class"], rows[8]["control_of"]) == ("none__lora0", "control", "none")
    served = _served(view_cells)
    runs = d.petri_schedule(view_cells, served, d.view_params("s3"))
    assert len(runs) == 8
    assert [r.cell_index for r in runs] == list(range(8)) and {r.seed_base for r in runs} == {3} and {r.width for r in runs} == {3}
    assert [r.cell_id for r in runs] == VIEW_CELLS and runs[6].vmodel == "petri__prompted__constitution" and runs[0].vmodel == "petri__base"
    assert {r.tag for r in runs} == {"qwen35_4b_s3"} and runs[0].record_name == "leg_0_s3" and runs[7].run_key == rows[7]["petri_run_key"]
    arms, excluded = d.lane_arm_indices(view_cells, d.view_params("s3"))
    assert arms == list(range(8)) and [(i, c) for i, c, _w in excluded] == [(8, "none__lora0")] and "derived no-op control of 'none'" in excluded[0][2]
    # a requested subset keeps only those indices
    sub = d.petri_schedule(view_cells, served, d.view_params("s3"), requested=[0, 6])
    assert [r.cell_index for r in sub] == [0, 6]
    # KNOWN-BAD 1: the earlier enumeration (none + 5, no prompted routes; 7 rows with its control) is NOT the view -- refused, the ARM count named
    six = native_cells_json(tmp_path / "six", VIEW_CELLS[:6], audit="seedrep-six")
    assert len(d.leg_rows(six)) == 7
    with pytest.raises(SystemExit, match="serves 6 lane arms, the lane expects 8 .*5 fitted \\+ none \\+ 2 prompted"):
        d.petri_schedule(six, _served(six), d.view_params("s3"))
    # KNOWN-BAD 2: a proxy that skipped one arm by name (the standup did not serve it) is not the view either
    with pytest.raises(SystemExit, match="serves 7 lane arms, the lane expects 8"):
        d.petri_schedule(view_cells, [s for s in served if s != "petri__prompted__safety"], d.view_params("s3"))
    # KNOWN-BAD 3: an extra arm (10 rows) -> 9 lane arms; a lens enumeration; a nameless leg; a row without cell_class; the control requested by index
    nine = native_cells_json(tmp_path / "nine", VIEW_CELLS + [EXTRA_ARM], audit="seedrep-nine")
    with pytest.raises(SystemExit, match="serves 9 lane arms"):
        d.petri_schedule(nine, _served(nine), d.view_params("s3"))
    with pytest.raises(SystemExit, match="serving_path='lens'"):
        d.petri_schedule({**view_cells, "serving_path": "lens"}, served, d.view_params("s3"))
    nameless = {**view_cells, "canonical": [dict(r, virtual_model_name="") if i == 2 else r for i, r in enumerate(view_cells["canonical"])]}
    with pytest.raises(SystemExit, match="virtual_model_name"):
        d.petri_schedule(nameless, served, d.view_params("s3"))
    classless = {**view_cells, "controls": [{k: v for k, v in r.items() if k != "cell_class"} for r in view_cells["controls"]]}
    with pytest.raises(SystemExit, match="cannot tell a derived control from a lane arm"):
        d.petri_schedule(classless, served, d.view_params("s3"))
    with pytest.raises(SystemExit, match="requested legs \\[8\\] are not served lane arms"):
        d.petri_schedule(view_cells, served, d.view_params("s3"), requested=[8])


def test_the_arditi_proxy_schedules_arditi_allpos_alone_at_three_seed_bases_into_three_out_roots(d, arditi_cells, tmp_path):
    rows = d.leg_rows(arditi_cells)
    assert [(r["cell_id"], r["cell_class"]) for r in rows] == [("none", "base"), ("arditi_allpos", "class_b"), ("none__lora0", "control")]
    runs = d.petri_schedule(arditi_cells, _served(arditi_cells), d.view_params("arditi"))
    assert [(r.cell_index, r.cell_id, r.seed_base, r.width, r.tag, r.record_name) for r in runs] == [
        (1, "arditi_allpos", 1, 4, "qwen35_4b_arditi_s1", "leg_1_s1"), (1, "arditi_allpos", 2, 4, "qwen35_4b_arditi_s2", "leg_1_s2"),
        (1, "arditi_allpos", 3, 4, "qwen35_4b_arditi_s3", "leg_1_s3")]
    assert len({r.run_key for r in runs}) == 1          # ONE manifest-derived run key -> the seed-runs MUST live in distinct out roots
    assert len({d.out_root_for(Path("/h"), "0c19251", "8f14e45f", "ta-1.uuid", r.tag) for r in runs}) == 3
    arms, excluded = d.lane_arm_indices(arditi_cells, d.view_params("arditi"))
    assert arms == [1] and [(i, c) for i, c, _w in excluded] == [(0, "none"), (2, "none__lora0")]
    # KNOWN-BAD 1: a second OBJECT on the Arditi standup is refused BY NAME, not by count
    two = native_cells_json(tmp_path / "two", ARDITI_CELLS + ["si_constv2_mix_full__lora_r4_ce"], audit="seedrep-arditi-two")
    with pytest.raises(SystemExit, match="SECOND object cell 'si_constv2_mix_full__lora_r4_ce' \\(class_a\\) beside arditi_allpos"):
        d.petri_schedule(two, _served(two), d.view_params("arditi"))
    # KNOWN-BAD 2: the arm itself not served by the proxy
    with pytest.raises(SystemExit, match="serves 0 lane arms, the lane expects 1 .*the one Arditi cell"):
        d.petri_schedule(arditi_cells, ["petri__base", "petri__none__lora0"], d.view_params("arditi"))


def test_leg_argv_is_petri_run_with_ws1b_s_target_decode_flags_after_run(d):
    opts = ["--manifest", "M.yaml", "--serving-path", "lora_native", "--verify-standup", "/s/cells_served.json", "--verify-routing-table", "/s/routing_table.json"]
    run = d.LegRun(view="s2", cell_index=3, cell_id="c", vmodel="petri__c", run_key="rk", seed_base=2, width=3, tag="qwen35_4b_s2")
    argv = d.leg_argv("/venv/bin/python", opts, run, Path("/out/petri_seedrep_0c19251_qwen35_4b_s2"))
    assert argv == ["/venv/bin/python", "scripts/petri_run.py", *opts, "run", "--out-root", "/out/petri_seedrep_0c19251_qwen35_4b_s2",
                    "--cell-index", "3", "--target-decode", "vendor", "--target-seed-base", "2"]
    assert argv.index("run") < argv.index("--target-decode")            # the two target flags are `run` subcommand flags
    rerun = d.leg_argv("/venv/bin/python", opts, d.LegRun(**{**run.__dict__, "petri_pass": "overflow-rerun"}), Path("/out/x"))
    assert rerun[-1] == "--overflow-rerun" and rerun[-3:-1] == ["--target-seed-base", "2"]
    d.assert_lane_leg_argv(argv, run)                          # the drive's own argv passes the lane-profile guard
    d.assert_lane_leg_argv(rerun, d.LegRun(**{**run.__dict__, "petri_pass": "overflow-rerun"}))
    # petri_run.py already has the top-level opts and `run --overflow-rerun`; the two target flags are added on top
    parser_src = (REPO / "scripts/petri_run.py").read_text()
    for flag in ("--serving-path", "--verify-standup", "--verify-routing-table", "--overflow-rerun", "--cell-index"):
        assert flag in parser_src
    assert d.TARGET_DECODE == "vendor"


# --------------------------------------------------------------------------- the auditor key: HIGHPRIO through the vault mechanism

def test_auditor_env_is_highprio_only_through_the_vault_mechanism_and_never_a_literal(d, monkeypatch):
    import sisteer.vault as vault
    calls = []

    def resolver(org):
        calls.append(org)
        return f"resolved-for-{org}-{HEX32}"

    env, stamp = d.auditor_env("highprio", key_resolver=resolver)
    assert calls == ["highprio"]
    assert env["ANTHROPIC_API_KEY"] == f"resolved-for-highprio-{HEX32}"
    assert env["ANTHROPIC_BASE_URL"] == "https://api.anthropic.com" == vault.ANTHROPIC_BASE_URL
    assert env["SISTEER_AUDITOR_POOL"] == "highprio" and env["F22_AUDITOR_ORG"] == "highprio"
    assert stamp == {"pool": "highprio", "key_sha256_12": hashlib.sha256(env["ANTHROPIC_API_KEY"].encode()).hexdigest()[:12]}
    assert env["ANTHROPIC_API_KEY"] not in json.dumps(stamp)
    # KNOWN-BAD 1: a Petri auditor call routed to LOWPRIO is refused BEFORE any key is resolved (never re-routed, never hopped)
    calls.clear()
    with pytest.raises(SystemExit, match="highprio only .*refusing auditor org 'lowprio'"):
        d.auditor_env("lowprio", key_resolver=resolver)
    with pytest.raises(SystemExit, match="refusing auditor org 'batch'"):
        d.auditor_env("batch", key_resolver=resolver)
    assert calls == []
    # KNOWN-BAD 2: a key-shaped "org" (someone passing the credential where the pool name goes)
    for literal in ("sk-ant-api03-" + HEX32, HEX32 + HEX32):
        with pytest.raises(SystemExit, match="POOL NAME"):
            d.auditor_env(literal, key_resolver=resolver)
    with pytest.raises(SystemExit, match="returned nothing"):
        d.auditor_env("highprio", key_resolver=lambda org: "")
    assert calls == []
    # the DEFAULT resolver is sisteer.vault.anthropic_api_key: highprio's OWN env var, else the secret-manager CLI, else PoolUnavailable
    monkeypatch.setenv("ANTHROPIC_API_KEY", "env-highprio-" + HEX32)
    env2, stamp2 = d.auditor_env()
    assert env2["ANTHROPIC_API_KEY"] == "env-highprio-" + HEX32 and stamp2["pool"] == "highprio"
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY_LOWPRIO", "env-lowprio-" + HEX32)       # another pool's key must NEVER satisfy highprio

    def no_secret_cli(*a, **k):
        raise FileNotFoundError("fake-secret-cmd")
    monkeypatch.setattr(vault.subprocess, "check_output", no_secret_cli)
    with pytest.raises(vault.PoolUnavailable, match="pool 'highprio' unavailable"):
        d.auditor_env()
    # the drive's source carries no credential and names the vault mechanism, not a file read
    src = SRC.read_text()
    assert "sk-ant" not in src and "anthropic_api_key" in src and 'AUDITOR_ORG = "highprio"' in src
    assert "read_text()" not in src.split("def auditor_env")[1].split("def qwen_auditor_env")[0]


def test_the_cli_only_offers_highprio_and_the_four_views(d):
    ap = d.build_parser()
    ns = ap.parse_args(["--sha", W, "--wave-json", "w.json", "--view", "s1"])
    assert ns.auditor_org == "highprio" and ns.petri_pass == "" and ns.shard == "0" and ns.retry_sleep_cap_s == 300 and ns.resume_from == ""
    assert ns.width is None and ns.serving_vol == "" and ns.teardown_after_failure is False
    with pytest.raises(SystemExit):
        ap.parse_args(["--sha", W, "--wave-json", "w.json", "--view", "s1", "--auditor-org", "lowprio"])
    with pytest.raises(SystemExit):
        ap.parse_args(["--sha", W, "--wave-json", "w.json", "--view", "s4"])
    with pytest.raises(SystemExit):
        ap.parse_args(["--sha", W, "--wave-json", "w.json", "--view", "s1", "--petri-pass", "second"])


def test_the_third_auditor_leg_rides_on_a_seedrep_pool_under_the_fallback_s_env_names(d):
    from sisteer.capeval.petri import auditor_fallback as af
    rec = {"lane": "seedrep", "model": "huihui-ai/Huihui-Qwen3.6-27B-abliterated", "url": "https://seedrep-attacker-0.modal.run/", "pool_index": 0}
    env = d.qwen_auditor_env(rec, "bearer-" + HEX32)
    assert env == {af.QWEN_BASE_URL_ENV: "https://seedrep-attacker-0.modal.run/v1", af.QWEN_API_KEY_ENV: "bearer-" + HEX32}
    assert af.QWEN_BASE_URL_ENV == "PETRI_QWEN_AUDITOR_BASE_URL" and af.QWEN_API_KEY_ENV == "PETRI_QWEN_AUDITOR_API_KEY"
    # KNOWN-BAD: the campaign pool (not the lane's), another model, a plain-http url, no bearer
    with pytest.raises(SystemExit, match="not a seedrep pool"):
        d.qwen_auditor_env({**rec, "lane": None}, "b")
    with pytest.raises(SystemExit, match="not the abliterated"):
        d.qwen_auditor_env({**rec, "model": "Qwen/Qwen3.5-4B"}, "b")
    with pytest.raises(SystemExit, match="not https"):
        d.qwen_auditor_env({**rec, "url": "http://127.0.0.1:8000"}, "b")
    with pytest.raises(SystemExit, match="no bearer"):
        d.qwen_auditor_env(rec, "")


# --------------------------------------------------------------------------- verdict, idempotent resume, throttle

def test_completion_verdict_and_legs_to_run_skip_complete_legs_and_redrive_the_rest(d, tmp_path, arditi_cells):
    runs = d.petri_schedule(arditi_cells, _served(arditi_cells), d.view_params("arditi"))
    roots = {r.tag: tmp_path / "out" / r.tag for r in runs}
    state = tmp_path / "state"
    state.mkdir()

    def results(run, body='{"audited": 60}'):
        p = roots[run.tag] / "cells" / run.run_key / "results.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(body)
        return p

    reconciled = []

    def bears_out(run, cell_dir):                         # the Petri driver's own store reconciliation, stubbed TRUE here (the real one: review-fixes tests)
        reconciled.append((run.record_name, cell_dir))
        return True, ""

    for r in runs:
        (state / f"{r.record_name}.rc").write_text(d.leg_status_line(0, "D1", results(r)) + "\n")
    v = d.completion_verdict(runs, roots, state, bears_out)
    assert v["counts"] == {"complete": 3, "incomplete": 0, "failed": 0} and v["release"] is True and v["nruns"] == 3
    assert [n for n, _c in reconciled] == [r.record_name for r in runs] and all(c == roots[r.tag] / "cells" / r.run_key for (_n, c), r in zip(reconciled, runs))
    # KNOWN-BAD 0: a results.json the store does NOT bear out (the reconciliation says no) is FAILED `unreconciled`, re-driven, and holds the proxy
    v_bad = d.completion_verdict(runs, roots, state, lambda run, cd: (False, "DriverError: no manifest.json") if run.record_name == "leg_1_s3" else (True, ""))
    assert v_bad["counts"] == {"complete": 2, "incomplete": 0, "failed": 1} and v_bad["release"] is False
    assert v_bad["detail"][2][2] == "unreconciled: DriverError: no manifest.json" and [r.record_name for r in d.legs_to_run(runs, v_bad)] == ["leg_1_s3"]
    v = d.completion_verdict(runs, roots, state, bears_out)
    assert d.legs_to_run(runs, v) == []                                  # IDEMPOTENT: a re-invocation drives nothing
    # KNOWN-BAD 1: a FAILED return with an old results.json present -> held; that leg alone is re-driven
    (state / "leg_1_s2.rc").write_text("1 drive=D1 results=absent\n")
    v = d.completion_verdict(runs, roots, state, bears_out)
    assert v["counts"]["failed"] == 1 and v["release"] is False and v["detail"][1][2] == "failed_rc_1_drive_D1"
    assert [r.record_name for r in d.legs_to_run(runs, v)] == ["leg_1_s2"]
    # KNOWN-BAD 2: a 0 return whose results bytes CHANGED since -> UNVERIFIED (failed), re-driven
    (state / "leg_1_s2.rc").write_text(d.leg_status_line(0, "D1", roots["qwen35_4b_arditi_s2"] / "cells" / runs[1].run_key / "results.json") + "\n")
    results(runs[1], '{"audited": 61}')
    v = d.completion_verdict(runs, roots, state, bears_out)
    assert v["detail"][1][2] == "unverified_results_changed_since_return" and [r.record_name for r in d.legs_to_run(runs, v)] == ["leg_1_s2"]
    # INCOMPLETE on an overflow ledger: releases the proxy, but the leg is still re-driven (on the re-run proxy)
    (roots["qwen35_4b_arditi_s2"] / "cells" / runs[1].run_key / "results.json").unlink()
    (roots["qwen35_4b_arditi_s2"] / "cells" / runs[1].run_key / "petri_incomplete.json").write_text(json.dumps({"unresolved": [{"b": "x"}]}))
    (state / "leg_1_s2.rc").write_text("3 drive=D1 results=absent\n")
    v = d.completion_verdict(runs, roots, state, bears_out)
    assert v["counts"] == {"complete": 2, "incomplete": 1, "failed": 0} and v["release"] is True
    assert [r.record_name for r in d.legs_to_run(runs, v)] == ["leg_1_s2"]
    # a leg with results but NO record (another drive's work) is complete by its cell dir alone -> not re-driven
    (state / "leg_1_s3.rc").unlink()
    v = d.completion_verdict(runs, roots, state, bears_out)
    assert v["detail"][2][2] == "complete_by_cell_dir_alone" and "leg_1_s3" not in [r.record_name for r in d.legs_to_run(runs, v)]
    # no record, no results, no ledger -> FAILED
    (roots["qwen35_4b_arditi_s2"] / "cells" / runs[1].run_key / "petri_incomplete.json").unlink()
    (state / "leg_1_s2.rc").unlink()
    v = d.completion_verdict(runs, roots, state, bears_out)
    assert v["detail"][1][2] == "failed_no_results" and v["release"] is False
    # the return record binds the drive id and the results bytes
    p = results(runs[0])
    line = d.leg_status_line(0, "D9", p)
    assert line == f"0 drive=D9 results={hashlib.sha256(p.read_bytes()).hexdigest()}" and d.parse_leg_status(line) == ("0", "D9", hashlib.sha256(p.read_bytes()).hexdigest())
    assert d.parse_leg_status("3\n") == ("3", "", "")


def test_width_throttles_down_one_step_on_a_429_heavy_attempt_and_never_below_one(d):
    heavy = "\n".join(f"retrying after 429 rate_limit_error ({i})" for i in range(20))
    assert d.width_after_attempt(3, heavy) == 2 and d.width_after_attempt(2, heavy) == 1 and d.width_after_attempt(1, heavy) == 1
    assert d.width_after_attempt(3, "\n".join("429" for _ in range(19))) == 3          # below the threshold: keep
    assert d.width_after_attempt(3, "auditor turn 12 complete\nsample 7 done") == 3
    assert d.width_after_attempt(4, "Rate limit exceeded " * 25) == 3
    with pytest.raises(ValueError):
        d.width_after_attempt(0, heavy)
    assert d.THROTTLE_429_THRESHOLD == 20


def test_container_paths_map_onto_the_attempt_root_and_the_tunnel_record_is_held_to_its_axes(d):
    root = "/vol/seedrep/0c19251/exp/seedrep_battery_modal/petri/s1/0/attempts/ta-1.uuid/exp"
    assert d.container_to_volume_path("/workspace/seedrep/exp/seedrep_battery_modal/petri/s1/serve/routing_table.json", "/workspace/seedrep/exp", root) \
        == f"{root}/seedrep_battery_modal/petri/s1/serve/routing_table.json"
    for bad in ("/workspace/arditi/exp/x.json", "/vol/seedrep/x.json", "/workspace/seedrep/other/x.json"):
        with pytest.raises(ValueError):
            d.container_to_volume_path(bad, "/workspace/seedrep/exp", root)
    assert d.volume_client_path("/vol/seedrep/x") == "/seedrep/x"
    with pytest.raises(ValueError):
        d.volume_client_path("/workspace/x")
    good = {"url": "https://x.modal.host", "function_call_id": "fc-1", "proxy_port": 8984, "view": "s1", "stage": "base", "control_key": TUNNEL_KEY,
            "serving_dir_on_volume": "/vol/seedrep/0c19251/exp/seedrep_battery_modal/petri/s1/0/attempts/ta-1.uuid/exp/serve",
            "attempt_root_on_volume": "/vol/seedrep/0c19251/exp/seedrep_battery_modal/petri/s1/0/attempts/ta-1.uuid", "attempt_id": "ta-1.uuid"}
    assert d.tunnel_record_problems(good, view="s1", stage="base", expected_control_key=TUNNEL_KEY, w7=W7) == []
    assert d.tunnel_record_problems({**good, "view": "s2"}, view="s1", stage="base", expected_control_key=TUNNEL_KEY, w7=W7) == ["view 's2' != 's1'"]
    assert d.tunnel_record_problems({**good, "stage": "petri-rerun"}, view="s1", stage="base", expected_control_key=TUNNEL_KEY, w7=W7) == ["stage 'petri-rerun' != 'base'"]
    assert "url 'http://x' is not https (modal.forward)" in d.tunnel_record_problems({**good, "url": "http://x"}, view="s1", stage="base", expected_control_key=TUNNEL_KEY, w7=W7)
    assert d.tunnel_record_problems({**good, "serving_dir_on_volume": "/vol/arditi_allpos/x"}, view="s1", stage="base", expected_control_key=TUNNEL_KEY, w7=W7) == \
        ["serving_dir_on_volume '/vol/arditi_allpos/x' is not under /vol/seedrep/0c19251/",
         f"serving_dir_on_volume '/vol/arditi_allpos/x' is not under the attempt's exp tree {good['attempt_root_on_volume']}/exp/"]
    assert set(d.tunnel_record_problems({}, view="s1", stage="base", expected_control_key=TUNNEL_KEY, w7=W7)) == {f"missing {k}" for k in d.TUNNEL_REQUIRED_KEYS} and \
        set(d.TUNNEL_REQUIRED_KEYS) == {"url", "function_call_id", "proxy_port", "serving_dir_on_volume", "control_key", "attempt_root_on_volume", "attempt_id", "view", "stage"}
    lp_ = d.local_port_of(8984)                                # F-AR: a free port in range, never 8984 + 10000
    assert d.LOCAL_PORT_MIN <= lp_ <= 65535 and lp_ != 8984 and not d.local_port_held(lp_)
    s = d.sentinel_content("STOP", "D1", "complete=8 incomplete=0 failed=0")
    assert s.startswith(b"STOP ") and b"drive=D1 complete=8" in s and s.endswith(b"\n")
    assert d.parse_legs_arg("ALL", [0, 1]) is None and d.parse_legs_arg("0, 1", [0, 1, 2]) == [0, 1]
    with pytest.raises(SystemExit):
        d.parse_legs_arg("5", [0, 1])


def test_load_wave_reads_the_minted_uuid_and_refuses_a_malformed_file(d, tmp_path):
    p = tmp_path / "wave.json"
    p.write_text(json.dumps({"wave_uuid": WAVE, "launch_plan_sha": "abcd1234abcd1234"}))
    assert d.load_wave(p) == (WAVE, "abcd1234abcd1234")
    p.write_text(json.dumps({"wave_uuid": "0c19251"}))
    with pytest.raises(SystemExit, match="not a uuid4"):
        d.load_wave(p)
    with pytest.raises(SystemExit, match="cannot read"):
        d.load_wave(tmp_path / "absent.json")
