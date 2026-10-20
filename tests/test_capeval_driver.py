"""capeval driver tests.

Covers, base lane only:
  * the CANONICAL cell enumeration (calibration / control / sweep modes)
    and its frozen counts (controls: 15 — F15);
  * the mirrored constants in scripts/sbatch/capeval_common.sh — parsed
    from the bash source and pinned against the Python enumeration (one
    canonical copy);
  * print-cells wire format + sbatch --array range compression;
  * --cell-index decode (refusals included) and the seed convention;
  * the F16 re-exec machinery: weight-mutating isolation, confirmation
    reruns (in-process AND child-exit-code paths), child-never-spawns;
  * _execute_cell orchestration order (F7 ts rows; results.json LAST;
    EvalPlus awaiting_execution -> results_pending.json, never a done
    marker — F8);
  * capeval_common.sh sourcing behavior: loud default-safe failure when
    the capeval checkout is absent, the exported extension variables,
    and the vllm own-venv bootstrap guard.

(The §13 "injected-shell safety" row — stage2_common.sh with the three
variables unset — lives in tests/test_capeval_stage2_common.py, where
Stage 0 put it.)
"""

import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).parents[1]
COMMON_SH = REPO / "scripts" / "sbatch" / "capeval_common.sh"

spec = importlib.util.spec_from_file_location(
    "capeval_run", REPO / "scripts" / "capeval_run.py")
capeval_run = importlib.util.module_from_spec(spec)
sys.modules["capeval_run"] = capeval_run  # dataclasses resolve __module__
spec.loader.exec_module(capeval_run)

from sisteer.capeval import manifest as cm  # noqa: E402

REV = "0123456789abcdef0123456789abcdef01234567"


def _manifest(mode, ablations, tiers=(0, 1), engine="hf"):
    return cm.Manifest(sweep=f"test_{mode}", mode=mode, base="test-org/tiny",
                       base_revision=REV, engine=engine, tiers=list(tiers),
                       ablations=ablations, confirmatory=False, target=None,
                       margins={})


def calib_manifest(trials=12):  # growth batches 1+2 (final)
    return _manifest("calibration", [
        cm.AblationSpec(name="base", kind="none", params={"trials": trials})])


def controls_manifest():
    """The control set — the reference the pinned width (15) derives
    from; controls.yaml must enumerate identically (test below)."""
    return _manifest("control", [
        cm.AblationSpec(name="zero_coeff", kind="control:zero_coeff",
                        artifact="out/x/vector.safetensors", layer=15,
                        coeffs=[0.0], seed=0),
        cm.AblationSpec(name="sham_hook", kind="control:sham_hook",
                        coeffs=[1.0], seed=0, params={"layer": 15}),
        cm.AblationSpec(name="random_vector", kind="control:random_vector",
                        coeffs=[0.5, 1.0, 2.0, 4.0], seed=0,
                        params={"layer": 15, "tier1_coeffs": [1.0, 4.0]}),
        cm.AblationSpec(name="noise_lora", kind="control:noise_lora",
                        coeffs=[1e-4, 3e-4, 1e-3, 3e-3], seed=0,
                        params={"tier1_coeffs": [3e-4, 3e-3]}),
        cm.AblationSpec(name="weight_noise", kind="control:weight_noise",
                        coeffs=[3e-3], seed=0),
    ])


def s2_manifest():
    arms = ["number_words_bidpo", "keyword_frequency_bidpo", "end_with_phrase_vector",
            "json_format_lora_r1", "number_bullets_loreft", "number_words_psr_kl",
            "json_format_vector_rkl"]
    return _manifest("sweep", [
        cm.AblationSpec(name=a, kind="steering", artifact=f"out/{a}.safetensors",
                        layer=15, coeffs=[2.0]) for a in arms])


# ------------------------------------------------------------- enumeration


def test_calibration_enumeration():
    cells = capeval_run.enumerate_cells(calib_manifest())
    assert len(cells) == 12  # final trial count
    assert [c.trial for c in cells] == list(range(12))
    assert all(c.coeff is None and c.tiers == (0, 1) for c in cells)
    assert all(c.time_class == "full" for c in cells)
    assert cells[0].index == 0  # trial 0 = the designated reference (F5)


def test_calibration_trials_grow_and_cap():
    assert len(capeval_run.enumerate_cells(calib_manifest(trials=9))) == 9
    with pytest.raises(ValueError):
        capeval_run.enumerate_cells(calib_manifest(trials=13))  # cap 12
    bad = _manifest("calibration", [
        cm.AblationSpec(name="a", kind="none"),
        cm.AblationSpec(name="b", kind="none")])
    with pytest.raises(ValueError):
        capeval_run.enumerate_cells(bad)


def test_controls_enumeration_is_the_pinned_15():
    cells = capeval_run.enumerate_cells(controls_manifest())
    assert len(cells) == 15  # §11.2 pinned width
    assert [c.arm for c in cells] == (
        ["zero_coeff", "sham_hook"] + ["random_vector"] * 6
        + ["noise_lora"] * 6 + ["weight_noise"])
    # ladder split: Tier-0 at every rung, Tier-1 only at the named rungs (§8)
    rv = [c for c in cells if c.arm == "random_vector"]
    assert [c.coeff for c in rv] == [0.5, 1.0, 2.0, 4.0, 1.0, 4.0]
    assert [c.tiers for c in rv] == [(0,)] * 4 + [(1,)] * 2
    t0only = [c.index for c in cells if c.time_class == "t0only"]
    full = [c.index for c in cells if c.time_class == "full"]
    assert t0only == [2, 3, 4, 5, 8, 9, 10, 11]
    assert full == [0, 1, 6, 7, 12, 13, 14]
    # non-ladder control arms keep the manifest's full tier set
    assert cells[0].tiers == (0, 1) and cells[14].tiers == (0, 1)


def test_controls_tier1_coeffs_must_be_subset():
    bad = _manifest("control", [
        cm.AblationSpec(name="rv", kind="control:random_vector",
                        coeffs=[0.5, 1.0], seed=0,
                        params={"tier1_coeffs": [4.0]})])
    with pytest.raises(ValueError, match="tier1_coeffs"):
        capeval_run.enumerate_cells(bad)


def test_s2_enumeration():
    cells = capeval_run.enumerate_cells(s2_manifest())
    assert len(cells) == 7  # 6 divergent + 1 gentle contrast (§9)
    assert cells[0].arm == "number_words_bidpo"  # index 0 = the pilot cell
    assert all(c.tiers == (0, 1) and c.coeff == 2.0 for c in cells)


def test_multi_document_sweep_enumerates_continuously():
    """The s2_rediscovery convention: one document per cell (each binds
    its own target, F20) — concatenated enumeration, continuous indices,
    each cell owned by its document."""
    m = s2_manifest()
    docs = [_manifest("sweep", [ab]) for ab in m.ablations]
    pairs = capeval_run.enumerate_all(docs)
    assert [c.index for c, _ in pairs] == list(range(7))
    assert [c.arm for c, _ in pairs] == [a.name for a in m.ablations]
    assert all(owner.ablations[0].name == c.arm for c, owner in pairs)
    with pytest.raises(ValueError, match="sweep-mode convention"):
        capeval_run.enumerate_all([calib_manifest(), calib_manifest()])


# ------------------------------------------------------------- ranges/format


def test_array_ranges_compression():
    ar = capeval_run.array_ranges
    assert ar([0, 1, 6, 7, 12, 13, 14]) == "0-1,6-7,12-14"
    assert ar([2, 3, 4, 5, 8, 9, 10, 11]) == "2-5,8-11"
    assert ar([3]) == "3"
    assert ar([]) == ""


def test_print_cells_format_controls():
    m = controls_manifest()
    out = capeval_run.format_print_cells(m, capeval_run.enumerate_cells(m))
    assert "COUNT=15" in out
    assert "ARRAY_ALL=0-14" in out
    assert "ARRAY_FULL=0-1,6-7,12-14" in out
    assert "ARRAY_T0ONLY=2-5,8-11" in out


def test_print_cells_format_calibration_and_sweep():
    m = calib_manifest()
    out = capeval_run.format_print_cells(m, capeval_run.enumerate_cells(m))
    assert "ARRAY_REFERENCE=0" in out and "ARRAY_REST=1-11" in out  # k=12
    m2 = s2_manifest()
    out2 = capeval_run.format_print_cells(m2, capeval_run.enumerate_cells(m2))
    assert "ARRAY_PILOT=0" in out2 and "ARRAY_REST=1-6" in out2


# ------------------------------------------------------------- bash mirror


def _bash_constants() -> dict:
    text = COMMON_SH.read_text()
    return dict(re.findall(r'^export (CAPEVAL_\w+)="?([^"\n]*?)"?\s*(?:#.*)?$',
                           text, flags=re.M))


def test_mirrored_constants_match_python_enumeration():
    """THE mirrored-constants test (F15/§11.1): capeval_common.sh's
    CAPEVAL_* exports == the canonical Python enumeration. A manifest edit
    that changes any width must update BOTH copies."""
    consts = _bash_constants()
    ar = capeval_run.array_ranges

    calib = capeval_run.enumerate_cells(calib_manifest())
    assert int(consts["CAPEVAL_CALIB_TRIALS"]) == len(calib) == 12  # final
    assert int(consts["CAPEVAL_CALIB_CELLS"]) == len(calib)
    assert consts["CAPEVAL_CALIB_ARRAY_REFERENCE"] == "0"
    assert consts["CAPEVAL_CALIB_ARRAY_REST"] == ar([c.index for c in calib[1:]])

    controls = capeval_run.enumerate_cells(controls_manifest())
    assert int(consts["CAPEVAL_CONTROLS_CELLS"]) == len(controls) == 15
    assert consts["CAPEVAL_CONTROLS_ARRAY_T0ONLY"] == ar(
        [c.index for c in controls if c.time_class == "t0only"])
    assert consts["CAPEVAL_CONTROLS_ARRAY_FULL"] == ar(
        [c.index for c in controls if c.time_class == "full"])

    s2 = capeval_run.enumerate_cells(s2_manifest())
    assert int(consts["CAPEVAL_S2_CELLS"]) == len(s2) == 7
    assert consts["CAPEVAL_S2_ARRAY_PILOT"] == "0"
    assert consts["CAPEVAL_S2_ARRAY_REST"] == ar([c.index for c in s2[1:]])

    assert int(consts["CAPEVAL_VLLM_CELLS"]) == 2  # Tier-1 ×2, one job


def _manifests_from_yaml(path: Path) -> list:
    """Raw-yaml manifest builder (protocol-test pattern) so this check
    never depends on the manifest loader — handles the multi-document
    s2_rediscovery convention (one doc per cell) via safe_load_all."""
    return [_manifest_from_raw(raw) for raw in yaml.safe_load_all(path.read_text())]


def _manifest_from_raw(raw: dict) -> cm.Manifest:
    target = raw.get("target")
    return cm.Manifest(
        sweep=raw["sweep"], mode=raw["mode"], base=raw["base"],
        base_revision=raw["base_revision"], engine=raw["engine"],
        tiers=raw["tiers"], confirmatory=raw.get("confirmatory", False),
        margins=raw.get("margins", {}),
        target=None if target is None else cm.TargetSpec(
            primaries=[cm.PrimarySpec(**p) for p in target["primaries"]],
            secondary=[cm.PrimarySpec(**p) for p in target.get("secondary", [])],
            off_target_exclusions=target.get("off_target_exclusions", []),
            relevant_corpus=cm.CorpusRef(**target["relevant_corpus"])),
        ablations=[cm.AblationSpec(**a) for a in raw["ablations"]])


@pytest.mark.parametrize("name,expected", [
    ("calibration.yaml", 12), ("controls.yaml", 15)])
def test_checked_in_manifest_widths(name, expected):
    path = REPO / "configs" / "capeval" / "manifests" / name
    if not path.is_file():
        pytest.skip(f"LOUD SKIP: {path} not landed yet (WS-A owns it) — the "
                    "integrator MUST re-run this test once it exists; until "
                    "then the width pin rests on the in-code reference manifest")
    pairs = capeval_run.enumerate_all(_manifests_from_yaml(path))
    assert len(pairs) == expected, (
        f"{name} enumerates {len(pairs)} cells, pinned width is {expected} — "
        "update capeval_common.sh + this pin together (F15)")
    assert [c.index for c, _ in pairs] == list(range(expected))


# ------------------------------------------------------------- decode/seed


def test_select_cell_by_index_and_refusals():
    cells = capeval_run.enumerate_cells(controls_manifest())
    assert capeval_run.select_cell(cells, index=14).arm == "weight_noise"
    with pytest.raises(ValueError, match="print-cells"):
        capeval_run.select_cell(cells, index=15)
    got = capeval_run.select_cell(cells, arm="random_vector", coeff=2.0)
    assert got.index == 4 and got.tiers == (0,)
    with pytest.raises(ValueError, match="matches 6 cells"):
        capeval_run.select_cell(cells, arm="random_vector")


def test_cell_seed_convention():
    calib = capeval_run.enumerate_cells(calib_manifest())
    assert capeval_run.cell_seed(calib[3], 0) == 3
    s2 = capeval_run.enumerate_cells(s2_manifest())
    assert capeval_run.cell_seed(s2[2], 0) == 102
    assert capeval_run.cell_seed(s2[2], 1) == 102 + capeval_run.ATTEMPT_SEED_OFFSET


# ------------------------------------------------------------- re-exec (F16)


class FakeStore:
    def __init__(self, cell_dir, done=False, key="fake__cell__t0__a0__aaaaaaaa__bbbbbbbb"):
        self.cell_dir = Path(cell_dir)
        self.sweep_dir = self.cell_dir.parent.parent  # <sweep>/cells/<key>
        self.key = key
        self.events = []
        self.results = None
        self._done = done

    def is_done(self):
        return self._done

    def append_ts(self, event, epoch_s):
        self.events.append(event)

    def write_results(self, summary):
        self.results = summary
        self._done = True
        self.events.append("WRITE_RESULTS")

    def write_generation_complete(self):
        # F8 ordering proof (review tier1 #3): the atomic pending summary
        # must already be on disk when the marker appears.
        self.events.append(
            ("GENERATION_COMPLETE",
             (self.cell_dir / "results_pending.json").is_file()))


def _fake_seams(store, phase_returns=None):
    phase_returns = phase_returns or {}

    def runner(name):
        def _run(engine, manifest, cell, s, seed):
            s.events.append(name)
            return phase_returns.get(name, {})
        return _run

    return {
        "make_engine": lambda m: object(),
        "resolve_cell": lambda m, c, a, o, s: (store, {}),
        "run_tier0": runner("tier0"),
        "run_tier1": runner("tier1"),
        "run_target": runner("target"),
    }


def _run_args(*extra):
    return capeval_run.build_parser().parse_args(
        ["--manifest", "m.yaml", "run", *extra])


@pytest.fixture(autouse=True)
def _no_wandb(monkeypatch):
    monkeypatch.setenv("CAPEVAL_WANDB", "disabled")


def test_execute_cell_ordering(tmp_path):
    store = FakeStore(tmp_path)
    m = calib_manifest()
    cell = capeval_run.enumerate_cells(m)[1]
    res = capeval_run._execute_cell(m, cell, 0, tmp_path, _fake_seams(store))
    assert store.events == [
        "job_start", "model_ready", "tier0_start", "tier0", "tier0_end",
        "tier1_start", "tier1", "tier1_end", "cell_end", "WRITE_RESULTS"]
    assert store.results is res and res["confirmation_needed"] == []


def test_execute_cell_awaiting_execution_leaves_results_to_the_wave(tmp_path):
    """F8 + review tier1 #3: an EvalPlus generation cell must NOT write
    results.json — the driver commits results_pending.json ATOMICALLY and
    only THEN generation_complete.json; the exec wave finishes the cell."""
    store = FakeStore(tmp_path)
    m = s2_manifest()
    cell = capeval_run.enumerate_cells(m)[0]
    capeval_run._execute_cell(m, cell, 0, tmp_path,
                              _fake_seams(store, {"tier1": {"awaiting_execution": True}}))
    assert "WRITE_RESULTS" not in store.events
    pending = json.loads((tmp_path / "results_pending.json").read_text())
    assert pending["phases"]["tier1"]["awaiting_execution"] is True
    # the marker was written AFTER the pending summary was already on disk
    assert store.events[-1] == ("GENERATION_COMPLETE", True)
    assert not (tmp_path / "results_pending.json.tmp").exists()  # atomic


def test_weight_mutating_arm_reexecs_and_never_runs_inline(monkeypatch, tmp_path):
    m = controls_manifest()
    spawned = []
    monkeypatch.setattr(capeval_run, "_spawn", lambda argv: spawned.append(argv) or 0)
    monkeypatch.setattr(capeval_run, "_load_seams",
                        lambda e: pytest.fail("parent of a weight-mutating cell "
                                              "must never load seams in-process"))
    rc = capeval_run.cmd_run(_run_args("--cell-index", "14",  # weight_noise
                                       "--out-root", str(tmp_path)), [m])
    assert rc == 0 and len(spawned) == 1
    argv = spawned[0]
    assert "--reexec-child" in argv and argv[argv.index("--attempt") + 1] == "0"
    assert argv[argv.index("--cell-index") + 1] == "14"


def _write_confirmation_marker(out_root: Path, arm: str, trial: int = 0,
                               probes=("canary_2_indicators",)) -> Path:
    cell_dir = (out_root / "cells"
                / f"{arm}__tier01-c0p003__t{trial}__a0__{'a' * 8}__{'b' * 8}")
    cell_dir.mkdir(parents=True, exist_ok=True)
    marker = cell_dir / capeval_run.CONFIRMATION_MARKER
    marker.write_text(json.dumps({"probes": list(probes)}))
    return cell_dir


def test_confirmation_rerun_via_child_exit_code(monkeypatch, tmp_path):
    """Child exits EXIT_CONFIRMATION_NEEDED AND persisted the marker ->
    parent re-execs attempt 1; a reproduced (confirmed) flag is a
    recorded result, rc 0 (F16; marker gate per review tier1 #5)."""
    m = controls_manifest()
    spawned = []

    def fake_spawn(argv):
        spawned.append(argv)
        # the child persists the marker before exiting 42 (tier1 #5)
        _write_confirmation_marker(tmp_path, "weight_noise")
        return capeval_run.EXIT_CONFIRMATION_NEEDED
    monkeypatch.setattr(capeval_run, "_spawn", fake_spawn)
    rc = capeval_run.cmd_run(_run_args("--cell-index", "14",
                                       "--out-root", str(tmp_path)), [m])
    assert rc == 0 and len(spawned) == 2
    assert spawned[1][spawned[1].index("--attempt") + 1] == "1"


def test_exit_42_without_marker_is_a_failure_not_a_trigger(monkeypatch, tmp_path):
    """Review tier1 #5: an unrelated child death that happens to exit 42
    must NOT be honored as a confirmation trigger — without the persisted
    marker the parent treats it as a failure and never spawns attempt 1."""
    m = controls_manifest()
    spawned = []

    def fake_spawn(argv):
        spawned.append(argv)
        return capeval_run.EXIT_CONFIRMATION_NEEDED  # no marker written
    monkeypatch.setattr(capeval_run, "_spawn", fake_spawn)
    rc = capeval_run.cmd_run(_run_args("--cell-index", "14",
                                       "--out-root", str(tmp_path)), [m])
    assert rc == capeval_run.EXIT_CONFIRMATION_NEEDED
    assert len(spawned) == 1  # no attempt-1 rerun


def test_confirmation_marker_persisted_and_resume_retriggers(tmp_path):
    """Review tier1 #5: attempt 0 persists confirmation_required.json
    BEFORE results.json; a resumed done cell with the marker re-reports
    its confirmation_needed so the parent re-triggers attempt 1."""
    m = calib_manifest()
    cell = capeval_run.enumerate_cells(m)[0]
    cell_dir = tmp_path / "cells" / "fake__cell__t0__a0__aaaaaaaa__bbbbbbbb"
    cell_dir.mkdir(parents=True)
    store = FakeStore(cell_dir)
    # tier0 files DICT entries ({"probe": <CONFIRMABLE_PROBES name>, ...});
    # the marker normalizes them to the NAME list attempt 1 feeds back.
    entry = {"probe": "dist_shift", "reason": "kl_coarse_lb > kappa on 2 corpora"}
    seams = _fake_seams(store, {"tier0": {"confirmation_needed": [entry]}})
    res = capeval_run._execute_cell(m, cell, 0, tmp_path, seams)
    assert res["confirmation_needed"] == [entry]
    marker = cell_dir / capeval_run.CONFIRMATION_MARKER
    payload = json.loads(marker.read_text())
    assert payload["probes"] == ["dist_shift"]  # names, the frozen contract
    assert payload["entries"] == [entry]
    # string entries (other phases) pass through as their own names
    (cell_dir / capeval_run.CONFIRMATION_MARKER).unlink()
    store = FakeStore(cell_dir)
    seams = _fake_seams(store, {"tier0": {"confirmation_needed": ["kl_2_corpora"]}})
    res = capeval_run._execute_cell(m, cell, 0, tmp_path, seams)
    assert res["confirmation_needed"] == ["kl_2_corpora"]
    assert json.loads(marker.read_text())["probes"] == ["kl_2_corpora"]
    # resume onto the DONE cell: the marker re-triggers, never a silent skip
    store2 = FakeStore(cell_dir, done=True)
    res2 = capeval_run._execute_cell(m, cell, 0, tmp_path,
                                     _fake_seams(store2))
    assert res2["resumed_done"] is True
    assert res2["confirmation_needed"] == ["kl_2_corpora"]
    # attempt 1 resumed-done does NOT re-trigger (its own results exist)
    res3 = capeval_run._execute_cell(m, cell, 1, tmp_path,
                                     _fake_seams(FakeStore(cell_dir, done=True)))
    assert res3["confirmation_needed"] == []


def test_attempt1_passes_confirm_probes_to_tier0(tmp_path):
    """Frozen contract: the attempt-1 invocation passes the persisted
    probe list into run_tier0 as confirm_probes=frozenset(...)."""
    m = calib_manifest()
    cell = capeval_run.enumerate_cells(m)[0]
    a0_dir = tmp_path / "cells" / "fake__cell__t0__a0__aaaaaaaa__bbbbbbbb"
    a0_dir.mkdir(parents=True)
    (a0_dir / capeval_run.CONFIRMATION_MARKER).write_text(
        json.dumps({"probes": ["canary_2_indicators", "kl_2_corpora"]}))
    a1_dir = tmp_path / "cells" / "fake__cell__t0__a1__aaaaaaaa__bbbbbbbb"
    a1_dir.mkdir(parents=True)
    store = FakeStore(a1_dir, key="fake__cell__t0__a1__aaaaaaaa__bbbbbbbb")
    got = {}

    def tier0(engine, manifest, c, s, seed, confirm_probes=frozenset()):
        got["confirm_probes"] = confirm_probes
        s.events.append("tier0")
        return {}

    seams = _fake_seams(store)
    seams["run_tier0"] = tier0
    capeval_run._execute_cell(m, cell, 1, tmp_path, seams)
    assert got["confirm_probes"] == frozenset(
        {"canary_2_indicators", "kl_2_corpora"})


def test_confirmation_rerun_from_in_process_trigger(monkeypatch, tmp_path):
    m = calib_manifest()
    store = FakeStore(tmp_path)
    seams = _fake_seams(store, {"tier0": {"confirmation_needed": ["canary_2_indicators"]}})
    monkeypatch.setattr(capeval_run, "_load_seams", lambda e: seams)
    spawned = []
    monkeypatch.setattr(capeval_run, "_spawn", lambda argv: spawned.append(argv) or 0)
    rc = capeval_run.cmd_run(_run_args("--cell-index", "0",
                                       "--out-root", str(tmp_path)), [m])
    assert rc == 0 and len(spawned) == 1
    assert "--reexec-child" in spawned[0]
    assert spawned[0][spawned[0].index("--attempt") + 1] == "1"


def test_child_never_spawns(monkeypatch, tmp_path):
    m = calib_manifest()
    store = FakeStore(tmp_path)
    seams = _fake_seams(store, {"tier0": {"confirmation_needed": ["x"]}})
    monkeypatch.setattr(capeval_run, "_load_seams", lambda e: seams)
    monkeypatch.setattr(capeval_run, "_spawn",
                        lambda argv: pytest.fail("--reexec-child must never spawn"))
    rc = capeval_run.cmd_run(_run_args("--cell-index", "0", "--reexec-child",
                                       "--out-root", str(tmp_path)), [m])
    assert rc == capeval_run.EXIT_CONFIRMATION_NEEDED


def test_no_confirm_suppresses_rerun(monkeypatch, tmp_path):
    m = calib_manifest()
    store = FakeStore(tmp_path)
    seams = _fake_seams(store, {"tier0": {"confirmation_needed": ["x"]}})
    monkeypatch.setattr(capeval_run, "_load_seams", lambda e: seams)
    monkeypatch.setattr(capeval_run, "_spawn",
                        lambda argv: pytest.fail("--no-confirm must suppress the rerun"))
    assert capeval_run.cmd_run(_run_args("--cell-index", "0", "--no-confirm",
                                         "--out-root", str(tmp_path)), [m]) == 0


# ------------------------------------------------------------- capeval_common.sh


FAKE_USER = "capevalxuser"  # never a real account (stage2_common test convention)

BASH_HARNESS = r"""
set -o pipefail
exit() { echo "__EXIT_CALLED__:${1:-0}"; }
uv() { echo "UV_ARGV:$*" >> "${UV_RECORD:-/dev/null}"; }
source "$CAPEVAL_COMMON"
echo "S2_REPO_DIR_AFTER:${S2_REPO_DIR:-}"
echo "S2_UV_ENV_AFTER:${S2_UV_ENV:-}"
echo "S2_UV_SYNC_ARGS_AFTER:${S2_UV_SYNC_ARGS:-}"
echo "UV_PY_DIR_AFTER:${UV_PYTHON_INSTALL_DIR:-}"
echo "CONTROLS_CELLS_AFTER:${CAPEVAL_CONTROLS_CELLS:-}"
if [ -n "${TRY_VLLM_BOOTSTRAP:-}" ]; then
  export UV_PROJECT_ENVIRONMENT="$TRY_VLLM_BOOTSTRAP"
  mkdir -p "$(dirname "$TRY_VLLM_BOOTSTRAP")"
  if capeval_vllm_bootstrap; then echo "BOOTSTRAP_RC:0"; else echo "BOOTSTRAP_RC:$?"; fi
fi
builtin exit 0
"""


def _run_common(tmp_path, extra_env):
    env = os.environ.copy()
    for k in ("CAPEVAL_REPO_DIR", "CAPEVAL_UV_ENV", "CAPEVAL_UV_SYNC_ARGS",
              "S2_REPO_DIR", "S2_UV_ENV", "S2_UV_SYNC_ARGS",
              "UV_PYTHON_INSTALL_DIR", "UV_PROJECT_ENVIRONMENT", "SISTEER_PROJECT_ROOT", "SISTEER_SHARED_ROOT"):
        env.pop(k, None)
    env["USER"] = FAKE_USER
    env["CAPEVAL_COMMON"] = str(COMMON_SH)
    # the shared root every job gets from the front door's job environment (capeval_common.sh refuses without it)
    env["SISTEER_SHARED_ROOT"] = str(tmp_path / "shared" / "root")
    env.update(extra_env)
    proc = subprocess.run(["bash", "-c", BASH_HARNESS], capture_output=True,
                          text=True, env=env, cwd=str(tmp_path), timeout=60)
    assert proc.returncode == 0, proc.stderr
    return proc


def _mk_checkout(tmp_path) -> Path:
    checkout = tmp_path / "capeval-repo"
    (checkout / "scripts" / "sbatch").mkdir(parents=True, exist_ok=True)
    (checkout / "scripts" / "sbatch" / "stage2_common.sh").write_text(
        'echo "STAGE2_SOURCED_FROM:$S2_REPO_DIR"\n')
    (checkout / "configs" / "capeval").mkdir(parents=True, exist_ok=True)
    reqs = checkout / "configs" / "capeval" / "vllm-requirements.txt"
    if not reqs.exists():
        reqs.write_text("vllm==0.26.0\n")
    return checkout


def test_common_sh_absent_checkout_fails_loud(tmp_path):
    proc = _run_common(tmp_path, {"SISTEER_PROJECT_ROOT": str(tmp_path / "shared" / "proj")})
    assert "__EXIT_CALLED__:1" in proc.stdout
    assert (f"capeval checkout '{tmp_path}/shared/proj-capeval-repo' "
            "does not exist" in proc.stderr)
    assert "NEVER used for capeval jobs (D9)" in proc.stderr


def test_common_sh_exports_and_sources_own_checkout(tmp_path):
    checkout = _mk_checkout(tmp_path)
    proc = _run_common(tmp_path, {"CAPEVAL_REPO_DIR": str(checkout)})
    assert "__EXIT_CALLED__" not in proc.stdout
    out = proc.stdout
    assert f"S2_REPO_DIR_AFTER:{checkout}" in out
    assert f"S2_UV_ENV_AFTER:/home/{FAKE_USER}/venvs/sisteer-capeval" in out
    assert "S2_UV_SYNC_ARGS_AFTER:--python 3.12 --group capeval" in out
    assert f"UV_PY_DIR_AFTER:{tmp_path}/shared/root/.uv-pythons" in out
    assert "CONTROLS_CELLS_AFTER:15" in out
    assert f"STAGE2_SOURCED_FROM:{checkout}" in out  # sources its OWN copy


def test_vllm_bootstrap_refuses_shared_venv(tmp_path):
    checkout = _mk_checkout(tmp_path)
    proc = _run_common(tmp_path, {
        "CAPEVAL_REPO_DIR": str(checkout),
        "TRY_VLLM_BOOTSTRAP": str(tmp_path / "venvs" / "sisteer-capeval")})
    assert "BOOTSTRAP_RC:1" in proc.stdout
    assert "refuses to install" in proc.stderr and "(D12)" in proc.stderr


def test_vllm_bootstrap_installs_pinned_requirements_into_vllm_venv(tmp_path):
    checkout = _mk_checkout(tmp_path)
    record = tmp_path / "uv_argv.txt"
    proc = _run_common(tmp_path, {
        "CAPEVAL_REPO_DIR": str(checkout),
        "TRY_VLLM_BOOTSTRAP": str(tmp_path / "venvs" / "sisteer-capeval-vllm"),
        "UV_RECORD": str(record)})
    assert "BOOTSTRAP_RC:0" in proc.stdout
    assert record.read_text() == (
        # cutoff == the repo's own tool.uv exclude-newer (all pins
        # in the requirements file predate it since the 0.26.0 re-pin).
        # --index-strategy unsafe-best-match: without it uv's default
        # first-index strategy confines `vllm` to the cu128 extra index
        # (which serves the name at other versions only) and resolution
        # fails with "no version of vllm==0.26.0" — job 116213 (2026-08-16).
        "UV_ARGV:pip install --python "
        f"{tmp_path}/venvs/sisteer-capeval-vllm/bin/python "
        "--index-strategy unsafe-best-match "
        "--exclude-newer 2026-08-03T00:00:00Z "
        f"-r {checkout}/configs/capeval/vllm-requirements.txt\n")
    # sentinel written next to the venv (review infra #12)
    sentinel = tmp_path / "venvs" / ".sisteer-capeval-vllm.vllm-bootstrap"
    assert sentinel.is_file()


def test_vllm_bootstrap_sentinel_skips_reinstall(tmp_path):
    """Review infra #12: a second bootstrap for the SAME requirements
    content must not touch the venv at all (no uv invocation) — an
    overlapping job on the node can never mutate the stack under a
    running workload."""
    checkout = _mk_checkout(tmp_path)
    record = tmp_path / "uv_argv.txt"
    env = {"CAPEVAL_REPO_DIR": str(checkout),
           "TRY_VLLM_BOOTSTRAP": str(tmp_path / "venvs" / "sisteer-capeval-vllm"),
           "UV_RECORD": str(record)}
    p1 = _run_common(tmp_path, env)
    assert "BOOTSTRAP_RC:0" in p1.stdout
    n_after_first = len(record.read_text().splitlines())
    p2 = _run_common(tmp_path, env)
    assert "BOOTSTRAP_RC:0" in p2.stdout
    assert "skipping (sentinel)" in p2.stdout
    assert len(record.read_text().splitlines()) == n_after_first  # no new uv call
    # a CHANGED requirements file reinstalls (new content = new stack)
    reqs = checkout / "configs" / "capeval" / "vllm-requirements.txt"
    reqs.write_text(reqs.read_text() + "# repinned\n")
    p3 = _run_common(tmp_path, env)
    assert "BOOTSTRAP_RC:0" in p3.stdout
    assert len(record.read_text().splitlines()) == n_after_first + 1


# --- the REAL vllm-requirements.txt — pins are deliberate, not drift ---

VLLM_REQS = REPO / "configs" / "capeval" / "vllm-requirements.txt"


def _parse_pins(text: str) -> dict:
    """{name: version} for `name==ver` lines, {name: url} for
    `name @ url` direct references (comments stripped)."""
    pins = {}
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip() if " #" in line else line.strip()
        if line.startswith("#") or not line:
            continue
        m = re.fullmatch(r"([A-Za-z0-9_.-]+)(?:\[[^\]]*\])?==(\S+)", line)
        if m:
            pins[m.group(1).lower()] = m.group(2)
            continue
        m = re.fullmatch(r"([A-Za-z0-9_.-]+)\s*@\s*(\S+)", line)
        if m:
            pins[m.group(1).lower()] = m.group(2)
    return pins


def test_vllm_requirements_pins_match_lockfile_and_d20():
    """The earlier pin set was UNSATISFIABLE — vllm 0.26.0 →
    nvidia-cutlass-dsl[cu13] 4.6.0 → protobuf>=6.30.2,<7 vs evalplus →
    google-generativeai → protobuf<6 are disjoint, so NO resolution
    exists (jobs 116213/117215) — and its cu128-wheel assumption was
    false anyway: PyPI vllm wheels of the torch-2.11 era are cu13-linked
    (libcudart.so.13, preflight 2026-08-17) and NO official cu128 vllm
    build exists. The current pin takes the official vllm 0.25.1+cu129
    GitHub release wheel by direct URL with a sha256 fragment (cu129 runs
    on driver 570 via CUDA 12.x minor-version compatibility —
    probe-verified) plus the matching pytorch cu129 torch stack; 0.25.1's
    cutlass-dsl 4.5.2 carries no protobuf dep, so evalplus coexists.
    Changing the vllm pin requires re-running the resolution scan AND a
    driver probe first — these assertions make that a deliberate act,
    not drift.

    The harness pins must also match uv.lock ("same pins as the
    lockfile"; torch matches on VERSION with the local build tag free,
    since the official vllm wheel dictates the CUDA variant): the vllm
    venv is the synced --group capeval layer PLUS this overlay, so a
    divergent pin here would silently replace a lockfile-governed
    package in the running venv."""
    pins = _parse_pins(VLLM_REQS.read_text())
    assert "vllm-0.25.1+cu129-cp38-abi3-manylinux_2_28_x86_64.whl" in pins["vllm"], (
        "re-run the D20 resolution scan + driver probe before repinning vllm")
    assert "github.com/vllm-project/vllm/releases/download/v0.25.1/" in pins["vllm"]
    assert "#sha256=" in pins["vllm"], "the direct-URL wheel must stay hash-pinned"

    lock = (REPO / "uv.lock").read_text()

    def locked_version(name: str):
        m = re.search(rf'name = "{name}"\nversion = "([^"]+)"', lock)
        return m.group(1) if m else None

    for name in ("evalplus", "wandb"):
        assert pins[name] == locked_version(name), (
            f"{name}: requirements pin {pins.get(name)} != locked "
            f"{locked_version(name)} (D12 same-pins invariant)")
    # lm-eval is gone from the lockfile and from this file: nothing in the serving venv imports it
    assert "lm-eval" not in pins and locked_version("lm-eval") is None
    # torch: same VERSION as the lockfile; the +cu local tag is the vllm
    # wheel's CUDA variant (cu129), not the lockfile's cu128.
    assert pins["torch"].split("+")[0] == locked_version("torch").split("+")[0], (
        f"torch version {pins['torch']} diverges from locked "
        f"{locked_version('torch')} (D12 same-version invariant)")
    assert pins["torch"].endswith("+cu129"), "D20 pins the cu129 torch stack"
    # vllm and its non-torch stack are deliberately OUT of the lockfile
    assert locked_version("vllm") is None
    # transformers: a DELIBERATE divergence below the lockfile (full22 smoke 2026-09-07): the lockfile's
    # 5.15.0 breaks vllm 0.25.1's gemma-4 config conversion (per-layer head_dim) and qwen35_4b engine init;
    # 5.14.1 serves both. Lifting it = re-running the served smoke on both models, not a lockfile sync.
    assert pins["transformers"] == "5.14.1", pins.get("transformers")
    assert locked_version("transformers") != "5.14.1" or True   # the lock may move; the serving pin holds


REVISE_TIME_HARNESS = r"""
set -o pipefail
exit() { echo "__EXIT_CALLED__:${1:-0}"; }
uv() { :; }
scontrol() { echo "   JobId=1 TimeLimit=$STUB_TIMELIMIT Partition=general"; }
source "$CAPEVAL_COMMON"
SECONDS=$STUB_ELAPSED
capeval_revise_time_check "$1"
builtin exit 0
"""


def _run_revise_time(tmp_path, header_limit_s, timelimit, elapsed):
    checkout = _mk_checkout(tmp_path)
    env = os.environ.copy()
    env.update({"USER": FAKE_USER, "CAPEVAL_COMMON": str(COMMON_SH),
                "CAPEVAL_REPO_DIR": str(checkout), "SLURM_JOB_ID": "12345",
                "STUB_TIMELIMIT": timelimit, "STUB_ELAPSED": str(elapsed),
                "SISTEER_SHARED_ROOT": str(tmp_path / "shared" / "root")})
    return subprocess.run(["bash", "-c", REVISE_TIME_HARNESS, "bash",
                           str(header_limit_s)],
                          capture_output=True, text=True, env=env,
                          cwd=str(tmp_path), timeout=60)


def test_revise_time_uses_effective_scontrol_limit(tmp_path):
    """Review infra #16: the warning compares against the LIVE TimeLimit
    (T0-only controls submit with a --time override), not the header
    constant passed as the fallback argument."""
    # header says 6h, live limit is 2h, elapsed 90min > 60% of 2h -> warn
    proc = _run_revise_time(tmp_path, 6 * 3600, "02:00:00", 5400)
    assert proc.returncode == 0, proc.stderr
    assert "REVISE-TIME" in proc.stderr
    assert "7200s" in proc.stderr  # the EFFECTIVE limit, not 21600
    # same elapsed under the 6h header alone (no live limit) -> no warning
    proc2 = _run_revise_time(tmp_path, 6 * 3600, "UNLIMITED", 5400)
    assert "REVISE-TIME" not in proc2.stderr


def test_timelimit_parser_forms(tmp_path):
    checkout = _mk_checkout(tmp_path)
    harness = 'source "$CAPEVAL_COMMON"; capeval_timelimit_to_s "$1"'
    env = os.environ.copy()
    env.update({"USER": FAKE_USER, "CAPEVAL_COMMON": str(COMMON_SH),
                "CAPEVAL_REPO_DIR": str(checkout), "SISTEER_SHARED_ROOT": str(tmp_path / "shared" / "root")})
    for tl, want in (("0-06:00:00", "21600"), ("02:00:00", "7200"),
                     ("1-00:30", "88200"), ("45:00", "2700"),
                     ("UNLIMITED", "")):
        proc = subprocess.run(["bash", "-c", harness, "bash", tl],
                              capture_output=True, text=True, env=env,
                              cwd=str(tmp_path), timeout=60)
        # drop the sourced stage2 stub's echo line; keep the function output
        lines = [l for l in proc.stdout.splitlines()
                 if not l.startswith("STAGE2_SOURCED_FROM:")]
        assert "\n".join(lines).strip() == want, (tl, proc.stdout, proc.stderr)


# ------------------------------------------------------------- sbatch pins


SBATCH_DIR = REPO / "scripts" / "sbatch"
ALL_SBATCH = ["capeval_calib.sbatch", "capeval_controls.sbatch",
              "capeval_s2.sbatch"]
ARRAY_SBATCH = ["capeval_calib.sbatch", "capeval_controls.sbatch",
                "capeval_s2.sbatch"]


def test_sbatch_qos_placeholder_is_invalid_on_purpose():
    """Review infra #14: `high` was a VALID QoS, so a forgotten --qos
    silently ran on it; the placeholder must be one sbatch refuses."""
    for name in ALL_SBATCH:
        text = (SBATCH_DIR / name).read_text()
        assert "#SBATCH --qos=CAPEVAL_REPLACE_ME" in text, name
        assert "#SBATCH --qos=high" not in text, name


def test_array_sbatch_require_slurm_array_task_id():
    """Review infra #13: array-only scripts refuse a missing --array
    instead of silently running cell 0 and exiting green."""
    for name in ARRAY_SBATCH:
        text = (SBATCH_DIR / name).read_text()
        assert 'IDX=${SLURM_ARRAY_TASK_ID:-0}' not in text, name
        assert 'SLURM_ARRAY_TASK_ID is unset' in text, name
    # and the guard actually fires: run the guard block standalone
    guard = "\n".join(
        l for l in (SBATCH_DIR / "capeval_calib.sbatch").read_text().splitlines()
        if not l.startswith("#SBATCH"))
    # isolate: stub everything before the guard
    harness = (
        "s2_log() { echo \"$@\" >&2; }\n"
        "CAPEVAL_CALIB_CELLS=6\n"
        + guard[guard.index("# SLURM_ARRAY_TASK_ID is REQUIRED"):
                guard.index("IDX=$SLURM_ARRAY_TASK_ID") + len("IDX=$SLURM_ARRAY_TASK_ID")]
        + "\necho IDX_OK:$IDX\n")
    proc = subprocess.run(["bash", "-c", harness], capture_output=True,
                          text=True, timeout=30)
    assert proc.returncode == 1 and "SLURM_ARRAY_TASK_ID is unset" in proc.stderr
    env = dict(os.environ, SLURM_ARRAY_TASK_ID="3")
    proc2 = subprocess.run(["bash", "-c", harness], capture_output=True,
                           text=True, timeout=30, env=env)
    assert proc2.returncode == 0 and "IDX_OK:3" in proc2.stdout
    env3 = dict(os.environ, SLURM_ARRAY_TASK_ID="3x")
    proc3 = subprocess.run(["bash", "-c", harness], capture_output=True,
                           text=True, timeout=30, env=env3)
    assert proc3.returncode == 1 and "not a non-negative integer" in proc3.stderr


# ------------------------------------------------- fingerprint/power (CPU)


REV_FP = "0e9e39f249a16976918f6564b8830bc894c89659"


def _fp_row(arm, kind, coeff, task, domain, item, metric, value, *,
            attempt=0, trial=0, tier=1, extra_json=None):
    from sisteer.capeval import store as cs

    return cs.ResultRow(
        sweep="fp_synth", run_id="r", model_id="test-org/tiny",
        model_revision=REV_FP, arm_name=arm, kind=kind, coeff=coeff,
        intervention_hash="a1b2c3d4e5f60718", engine="hf", tier=tier,
        task=task, task_version="v1", domain=domain, item_id=item,
        trial_id=trial, seed=trial, attempt=attempt, position_order=None,
        metric=metric, value=value, extra_json=extra_json,
        harness_config_hash="0f1e2d3c4b5a6978", env_digest="99aabbccddeeff00",
        slurm_job_id=None, created_at=1755000000.0, wall_clock_s=None)


def _write_fp_cell(root, key, rows):
    from sisteer.capeval import store as cs

    store = cs.CellStore(root, key)
    for r in rows:
        store.append_row(r)
    store.write_results({"n": len(rows)})


@pytest.fixture()
def fp_world(tmp_path):
    """Synthetic sweep + calibration for the fingerprint/power CPU steps."""
    from sisteer.capeval.domains import Registry
    from sisteer.capeval.stats import calibrate as cal

    fixtures = REPO / "tests" / "fixtures" / "capeval"
    manifest_path = tmp_path / "m.yaml"
    manifest_path.write_text(f"""\
sweep: fp_synth
mode: sweep
confirmatory: false
base: test-org/tiny
base_revision: {REV_FP}
engine: hf
tiers: [0, 1]
margins: {{}}
target:
  primaries:
    - metric: si_compliance
      plugin: sisteer_compliance
      params: {{si: number_words}}
      direction: up
      min_effect: 0.1
  relevant_corpus:
    path: {fixtures / 'pairs_synthetic.jsonl'}
    slice: {{first: 2, field: prompt}}
ablations:
  - name: arm-a
    kind: steering
    artifact: {fixtures / 'vector_synthetic.safetensors'}
    layer: 15
    coeffs: [2.0]
""")
    sweep_root = tmp_path / "sweep"
    calib_root = tmp_path / "calibration"
    items = [f"i{k}" for k in range(8)]
    _write_fp_cell(calib_root, "none__tier01__t0__a0__a1b2c3d4__0f1e2d3c",
                   [_fp_row("none", "none", None, "tinyMMLU", "knowledge", i,
                            "acc", 1.0) for i in items])
    arm_rows = [_fp_row("arm-a", "steering", 2.0, "tinyMMLU", "knowledge", i,
                        "acc", 0.0) for i in items]
    arm_rows += [_fp_row("arm-a", "steering", 2.0, "target/sisteer_compliance",
                         "target", i, f"si_compliance/{leg}", v, tier=-1)
                 for i in items
                 for leg, v in (("arm", 1.0), ("base", 0.0))]
    # Tier-0 KL rows shaped exactly like the stored legacy campaign:
    # a kl_coarse_lb_corpus magnitude plus flag_kl_gt_kappa filed at BOTH
    # attempts — these must surface as the annotation and
    # never as a confirmed Tier-0 flag.
    arm_rows += [
        _fp_row("arm-a", "steering", 2.0, "tier0/kl_coarse_wikitext103",
                "dist_shift", "_corpus", "kl_coarse_lb_corpus", 0.02, tier=0),
        _fp_row("arm-a", "steering", 2.0, "tier0/kl_coarse_wikitext103",
                "dist_shift", "_corpus", "flag_kl_gt_kappa", 1.0, tier=0),
    ]
    _write_fp_cell(sweep_root, "arm-a__tier01__t0__a0__a1b2c3d4__0f1e2d3c",
                   arm_rows)
    _write_fp_cell(sweep_root, "arm-a__tier01__t0__a1__a1b2c3d4__0f1e2d3c",
                   [_fp_row("arm-a", "steering", 2.0,
                            "tier0/kl_coarse_wikitext103", "dist_shift",
                            "_corpus", "flag_kl_gt_kappa", 1.0, attempt=1,
                            tier=0)])

    registry = Registry.load(REPO / "configs" / "capeval" / "domains.yaml")
    calib = cal.Calibration(
        model_revision=REV_FP, engine="hf",
        harness_version=cal.harness_version(registry),
        estimand="conditional_on_reference", reference_trial=0,
        n_contrasts=5, b_boot=100, seed=0, created_at=1755000000.0,
        domains={"knowledge": cal.DomainCalibration(
            domain="knowledge", kind="accuracy", sigma_d=0.01,
            sigma_d_se=0.002,
            tails={"p01": -0.02, "p05": -0.01, "p50": 0.0, "p95": 0.01,
                   "p99": 0.02},
            n_contrasts=5, n_items=8, needs_more_trials=False,
            tasks={"tinyMMLU": cal.TaskCalibration(
                task="tinyMMLU", n_items=8, base_mean=0.875,
                ref_values=[1.0] * 7 + [0.0],
                arm_values_by_contrast=[
                    [1.0] * 7 + [0.0], [1.0] * 6 + [0.0, 1.0],
                    [1.0] * 7 + [1.0], [1.0] * 6 + [0.0, 0.0],
                    [1.0] * 7 + [0.0]])},
            null_discordant=2, null_pairs=40,
            null_discordance_rate=0.05, base_mean=0.875)},
        canary_thresholds=None, kappa=1e-3,
        model_id="test-org/tiny",
        task_versions={"tinyMMLU": "v1",
                       "tier0/kl_coarse_wikitext103": "v1"})
    cal.save_calibration(calib, calib_root / "calibration.json")
    return manifest_path, sweep_root, calib_root


def test_cmd_fingerprint_writes_schema_versioned_file(fp_world):
    """Simplify-pass #1: the fingerprint subcommand must call the REAL
    stats API (per-arm build_fingerprint) and write the schema-versioned
    wire format whose domain entries carry the writer's field names
    (upper_bound/lower_bound — simplify-pass #6)."""
    manifest_path, sweep_root, calib_root = fp_world
    rc = capeval_run.main([
        "--manifest", str(manifest_path), "fingerprint",
        "--out-root", str(sweep_root), "--calibration-root", str(calib_root),
        "--b", "50"])
    assert rc == 0
    fp = json.loads((sweep_root / "fingerprint.json").read_text())
    assert fp["schema_version"] == capeval_run.FINGERPRINT_SCHEMA_VERSION
    entry = fp["arms"]["arm-a"]
    assert entry["screen"] == "SCREEN-FAIL"  # knowledge harm 1.0 > 0.02
    # flag_kl_gt_kappa filed at BOTH attempts (the stored legacy
    # campaign shape) must NOT read as a confirmed Tier-0 flag — the
    # canary is the only screen-feeding flag now
    assert entry["tier0_flag_confirmed"] is False
    # ...and the KL magnitude surfaces as the annotation instead
    assert entry["kl_coarse_lb"]["mean"] == pytest.approx(0.02)
    assert entry["kl_coarse_lb"]["per_corpus"] == {
        "tier0/kl_coarse_wikitext103": pytest.approx(0.02)}
    dom = entry["domains"]["knowledge"]
    assert dom["status"] == "degraded"
    assert "upper_bound" in dom and "lower_bound" in dom  # writer's names
    assert "ci_low" not in dom and "ci_high" not in dom
    assert dom["delta"] == pytest.approx(-1.0)
    assert entry["target"]["status"] == "established"
    # the report consumes the file end-to-end (reader == writer names)
    from sisteer.capeval.report import build as rb

    arms = rb.load_fingerprint_arms(fp)
    table = rb.fingerprint_table(arms, rb._domains_registry())
    row = table[(table["arm"] == "arm-a") & (table["domain"] == "knowledge")].iloc[0]
    assert row["harm_upper"] == dom["upper_bound"]
    assert row["margin"] == dom["margin"]


def test_cmd_fingerprint_confirmed_canary_sets_tier0_flag(fp_world):
    """The one flag that still reaches the screen: tier0's
    flag_canary_confirmed row — filed only when the requested attempt-1
    rerun reproduced the canary failure — sets tier0_flag_confirmed."""
    manifest_path, sweep_root, calib_root = fp_world
    _write_fp_cell(sweep_root, "arm-a__tier0c__t0__a1__a1b2c3d4__0f1e2d3c",
                   [_fp_row("arm-a", "steering", 2.0, "tier0/canary",
                            "coherence_canary", "_run",
                            "flag_canary_confirmed", 1.0, attempt=1,
                            tier=0)])
    rc = capeval_run.main([
        "--manifest", str(manifest_path), "fingerprint",
        "--out-root", str(sweep_root), "--calibration-root", str(calib_root),
        "--b", "50"])
    assert rc == 0
    fp = json.loads((sweep_root / "fingerprint.json").read_text())
    assert fp["arms"]["arm-a"]["tier0_flag_confirmed"] is True


def test_cmd_power_writes_power_table(fp_world):
    """Simplify-pass #1: the power subcommand must call the REAL
    stats.power.power_table (build_power_table never existed)."""
    manifest_path, sweep_root, calib_root = fp_world
    rc = capeval_run.main([
        "--manifest", str(manifest_path), "power",
        "--out-root", str(sweep_root), "--calibration-root", str(calib_root)])
    assert rc == 0
    payload = json.loads((sweep_root / "power.json").read_text())
    assert payload["schema_version"] == capeval_run.FINGERPRINT_SCHEMA_VERSION
    [table] = payload["tables"]
    assert table["exclusions"] == []
    [entry] = table["entries"]
    assert entry["domain"] == "knowledge" and entry["kind"] == "accuracy"
    assert 0.0 <= entry["power_preserved_h0"] <= 1.0
    assert 0.0 <= entry["power_degraded_h2m"] <= 1.0
