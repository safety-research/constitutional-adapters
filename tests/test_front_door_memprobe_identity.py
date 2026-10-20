"""END-TO-END (memprobe-fix-2): `02 --step memprobe` on the fake cluster, in the state the lane hit live on 2026-10-02 — the pre-fix memprobe
unit `<model>/02_prepare_model/memprobe/<arm>` bound to a COMPLETED probe whose record measured ANOTHER architecture than the committed config
selects (here: a rank-16 MLP probe against the configured-not-run model's rank-4 o_proj config). Expected: the stale record is archived
fail-closed (unique name, sha256 read back, live name absent), the DONE marker is cleared, a NEW unit keyed on the selection fingerprint is
minted through the funnel's first-submit reservation, and the old unit stays bound to its completed job with its archive intact. Negative: a
fresh record for the same selection is neither archived nor re-probed — it reaches the FD 23 verdict (here DOES NOT FIT, exit 5) and mints
nothing; and the POSITIVE verdict folds (`fold ->`, the DONE marker, the two config files rewritten) — driven into a sparse tmp checkout of
HEAD, never this repo."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from front_door_testlib import REPO, REPRO, VENV_PY, job_ids, set_job_state  # noqa: E402
from test_front_door_probe_resume import World  # noqa: E402

MODEL, FIT_KEY = "qwen35_4b_demo", "qwen35_4b_demo_base"                      # the configured-not-run front-door model: 02 runs for it (memprobe + register)
CFG = REPO / "configs" / f"sdfbase_{FIT_KEY}_fits.yaml"
OVERRIDES = REPRO / "overrides" / "paper_defaults.env"              # named explicitly: the stage's overrides are not the shared World's business here
ARM = "lora_r4_ce"
OLD_LWID = f"{MODEL}/02_prepare_model/memprobe/{ARM}"


def _mf():
    spec = importlib.util.spec_from_file_location("memprobe_fold", REPO / "scripts" / "sdfbase" / "memprobe_fold.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def _record(selection: dict, *, within_headroom_rung2: bool, fits: bool = True) -> dict:
    """A memprobe record for THIS config's model + revision (synthetic card numbers); `fits=False` is the FD 23 does-not-fit verdict."""
    cfg = yaml.safe_load(CFG.read_text())
    total = 150_120_759_296
    peak2 = int((0.80 if within_headroom_rung2 else 0.915) * total)
    rungs = [{"micro_batch": 1, "fits": fits, "peak_alloc_bytes": int(0.69 * total) if fits else int(0.98 * total), "s_per_step": 49.2 if fits else None,
              "error": None if fits else "OutOfMemoryError"},
             {"micro_batch": 2, "fits": fits, "peak_alloc_bytes": peak2, "s_per_step": 13.1 if fits else None, "error": None if fits else "OutOfMemoryError"},
             {"micro_batch": 4, "fits": False, "peak_alloc_bytes": int(0.9 * total), "s_per_step": None, "error": "OutOfMemoryError"}]
    verdict = ({"exit_code": 0, "fits_at_min_micro_batch": True, "largest_fitting_micro_batch": 2, "min_micro_batch": 1, "replan_required": False, "s_per_step_at_min": 49.2}
               if fits else {"exit_code": 5, "fits_at_min_micro_batch": False, "largest_fitting_micro_batch": None, "min_micro_batch": 1, "replan_required": True, "s_per_step_at_min": None})
    return {"arm": ARM, "schema": 1, "seq_len": int(cfg["max_len"]), "steps": 3, "gpu_total_bytes": total, "gpu_name": "NVIDIA H200",
            "gradient_checkpointing": True, "dtype": "bfloat16",
            "model": {"model": cfg["model"], "revision": cfg["model_revision"], "model_override": False},
            "object_kwargs": selection, "rungs": rungs, "verdict": verdict}


def _funnel(w: World, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run([VENV_PY, str(REPRO / "lib" / "submit_twin.py"), "--config", str(w.site / "config.env"), "--model", MODEL, *args],
                          capture_output=True, text=True, env=w.env(), cwd=str(REPO), timeout=300)


def _run02(w: World) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", str(REPRO / "02_prepare_model.sh"), "--model", MODEL, "--step", "memprobe", "--arms", ARM,
                           "--config", str(w.site / "config.env"), "--overrides", str(OVERRIDES)], capture_output=True, text=True, env=w.env(), cwd=str(REPO), timeout=600)


def _binding(w: World, lwid: str) -> str:
    r = _funnel(w, "binding", "--lwid", lwid)
    return next((ln for ln in (r.stdout + r.stderr).splitlines() if ln.startswith("binding\t")), r.stdout + r.stderr)


def test_a_completed_stale_probe_is_archived_with_read_back_and_the_new_selection_mints_a_new_unit(tmp_path):
    w = World(tmp_path); mf = _mf(); cfg = yaml.safe_load(CFG.read_text())
    fp = mf.fingerprint(cfg, ARM)
    state = w.shared / "exp" / "reproduce" / MODEL / "02_prepare_model"; state.mkdir(parents=True, exist_ok=True)
    marker_dir = w.tmp / "state" / MODEL / "02_prepare_model"; marker_dir.mkdir(parents=True, exist_ok=True)
    # the funnel reads the fit venv's certification on the target cluster before minting: BUILD-COMPLETE `<iso> <sha16>` + the recipe
    venv = w.shared / "envs" / "sdfbase-fit"; venv.mkdir(parents=True, exist_ok=True)
    (venv / "BUILD-COMPLETE").write_text("2026-09-16T05:09:17+00:00 40b49ecf00359e44\n")
    (w.shared / "envs" / "sdfbase-fit.build").mkdir(parents=True, exist_ok=True)
    (w.shared / "envs" / "sdfbase-fit.build" / "requirements.txt").write_text("torch==2.8.0\n")
    # 1. the pre-fix world: the unkeyed unit bound to a probe job that COMPLETED, its record measuring ANOTHER object (a rank-16 MLP probe), the marker set
    r = _funnel(w, "submit", "--unit", f"memprobe.{ARM}", "--lwid", OLD_LWID, "--script", "scripts/sbatch/memprobe_fit.sbatch",
                "--name", f"memprobe-{MODEL}-{ARM}", "--args", f"{FIT_KEY} {ARM} 1,2,4", "--clusters", "cluster_a",
                "--kind", "fit", "--registers", "none", "--time", "01:30:00")
    assert r.returncode == 0 and "submitted cluster_a" in r.stdout, r.stdout + r.stderr
    (j0,) = job_ids(w.fake); j0 = str(j0); set_job_state(w.fake, j0, "COMPLETED")
    stale = _record({"init_scheme": "zero", "lora_modules": ["down_proj", "gate_proj", "up_proj"], "lora_rank": 16}, within_headroom_rung2=False)
    rec = state / f"memprobe.{ARM}.json"; rec.write_text(json.dumps(stale)); want = hashlib.sha256(rec.read_bytes()).hexdigest()
    (state / f"memprobe_report.{ARM}.md").write_text("# stale report\n")
    (marker_dir / f"DONE.memprobe.{ARM}").write_text("2026-09-30T00:41:00Z folded\n")
    # 2. the selection changed (the committed config): 02 archives, clears, mints the NEW unit
    r = _run02(w); out = r.stdout + r.stderr
    assert r.returncode == 0, out
    assert "STALE" in out and "lora_rank: record probed 16, the config selects 4" in out and "sha256 read back" in out, out
    archives = sorted(state.glob(f"memprobe.{ARM}.json.stale.*")); archives = [a for a in archives if not a.name.endswith(".report.md")]
    assert len(archives) == 1 and hashlib.sha256(archives[0].read_bytes()).hexdigest() == want and not rec.exists(), [a.name for a in state.iterdir()]
    assert (state / (archives[0].name + ".report.md")).read_text() == "# stale report\n"
    assert not (marker_dir / f"DONE.memprobe.{ARM}").exists()
    assert f"submitted cluster_a " in out and f"memprobe.{ARM}.{fp}" in out, out
    jobs = [str(j) for j in job_ids(w.fake)]; assert len(jobs) == 2, jobs
    j1 = [j for j in jobs if j != j0][0]
    calls = {str(c["id"]): c for c in w.sbatch_calls()}          # job_ids() yields ints; the funnel prints them as strings
    assert calls[j1]["name"] == f"memprobe-{MODEL}-{ARM}-{fp}" and calls[j0]["name"] == f"memprobe-{MODEL}-{ARM}"
    assert f"{j0}" in _binding(w, OLD_LWID), "the old unit stays bound to its completed job"
    assert f"{j1}" in _binding(w, f"{OLD_LWID}/{fp}"), "the new unit is bound to the new probe"
    # 3. negative: the probe finished for THIS selection — a fresh record is NOT stale: nothing archived, nothing re-probed, the FD 23 verdict
    #    reaches the caller (here DOES NOT FIT at micro-batch 1, exit 5, so no fold is written into the committed configs — a fold would
    #    regenerate configs/sdfbase_folds.json + the fits config in THIS checkout, a repo write no test performs) and nothing is minted
    set_job_state(w.fake, j1, "COMPLETED")
    fresh = _record(mf.config_selection(cfg, ARM), within_headroom_rung2=False, fits=False)
    rec.write_text(json.dumps(fresh))
    folds_before = (REPO / "configs" / "sdfbase_folds.json").read_bytes(); cfg_before = CFG.read_bytes()
    r = _run02(w); out = r.stdout + r.stderr
    assert r.returncode == 5 and "DOES NOT FIT at micro-batch 1" in out and "FD 23" in out and "submitted cluster_a" not in out and "STALE" not in out, out
    assert "fold ->" not in out and "archived" not in out, out
    assert [str(j) for j in job_ids(w.fake)] == jobs, "no third job"
    assert not (marker_dir / f"DONE.memprobe.{ARM}").exists() and rec.exists()
    assert [a for a in state.glob(f"memprobe.{ARM}.json.stale.*") if not a.name.endswith(".report.md")] == archives, "no second archive"
    assert (REPO / "configs" / "sdfbase_folds.json").read_bytes() == folds_before and CFG.read_bytes() == cfg_before, "nothing folded — the repo stays clean"


def _sparse_checkout(tmp_path: Path) -> Path:
    """A detached sparse worktree of HEAD under tmp carrying only what `02 --step memprobe` reads and writes (reproduce/, scripts/, src/,
    configs/): the fold's repo write lands THERE. Removed by the caller."""
    import shutil
    wt = Path("/tmp") / f"pcr-memprobe-{tmp_path.name[-8:]}-{hashlib.sha256(str(tmp_path).encode()).hexdigest()[:8]}"
    subprocess.run(["git", "worktree", "add", "--detach", "--no-checkout", "-q", str(wt), "HEAD"], cwd=REPO, check=True, capture_output=True, text=True)
    try:
        subprocess.run(["git", "-C", str(wt), "sparse-checkout", "set", "--cone", "reproduce", "scripts", "src", "configs"], check=True, capture_output=True, text=True)
        subprocess.run(["git", "-C", str(wt), "checkout", "-q", "HEAD"], check=True, capture_output=True, text=True)
    except Exception:
        subprocess.run(["git", "worktree", "remove", "--force", str(wt)], cwd=REPO, capture_output=True, text=True); shutil.rmtree(wt, ignore_errors=True)
        raise
    return wt


def _run02_in(w: World, repo: Path) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", str(repo / "reproduce" / "02_prepare_model.sh"), "--model", MODEL, "--step", "memprobe", "--arms", ARM,
                           "--config", str(w.site / "config.env"), "--overrides", str(repo / "reproduce" / "overrides" / "paper_defaults.env")],
                          capture_output=True, text=True, env=w.env(), cwd=str(repo), timeout=900)


def test_a_fresh_fitting_record_folds_into_the_checkout_it_runs_from_and_marks_the_unit_done(tmp_path):
    """The positive FD 23 verdict (memprobe-fold: the record measured THIS selection and micro-batch 1 fits with headroom) folds —
    `make_sdfbase_configs.py --fold-memprobe` rewrites configs/sdfbase_folds.json and the fit config, `repro_mark_done` writes the unit's
    DONE marker, and the stage says `fold ->` — driven into a sparse tmp checkout of HEAD so this repo's configs stay byte-identical."""
    import shutil
    w = World(tmp_path); mf = _mf(); cfg = yaml.safe_load(CFG.read_text())
    state = w.shared / "exp" / "reproduce" / MODEL / "02_prepare_model"; state.mkdir(parents=True, exist_ok=True)
    marker_dir = w.tmp / "state" / MODEL / "02_prepare_model"; marker_dir.mkdir(parents=True, exist_ok=True)
    fresh = _record(mf.config_selection(cfg, ARM), within_headroom_rung2=False)            # fits at micro-batch 1 (rung 2 above headroom)
    (state / f"memprobe.{ARM}.json").write_text(json.dumps(fresh))
    (state / f"memprobe_report.{ARM}.md").write_text("# fresh report\n")
    repo_folds_before = (REPO / "configs" / "sdfbase_folds.json").read_bytes(); repo_cfg_before = CFG.read_bytes()
    wt = _sparse_checkout(tmp_path)
    try:
        wt_folds = wt / "configs" / "sdfbase_folds.json"; wt_cfg = wt / "configs" / f"sdfbase_{FIT_KEY}_fits.yaml"
        folds_before = wt_folds.read_bytes(); cfg_before = wt_cfg.read_bytes()
        assert folds_before == repo_folds_before and cfg_before == repo_cfg_before                    # the checkout starts at HEAD's bytes
        r = _run02_in(w, wt); out = r.stdout + r.stderr
        assert r.returncode == 0 and "fold ->" in out and "FD23" in out and "STALE" not in out and "submitted cluster_a" not in out, out[-3000:]
        assert (marker_dir / f"DONE.memprobe.{ARM}").exists(), sorted(p.name for p in marker_dir.iterdir())
        assert wt_folds.read_bytes() != folds_before and wt_cfg.read_bytes() != cfg_before, "the fold rewrote the checkout's folds store and fit config"
        folded = yaml.safe_load(wt_cfg.read_text())
        assert folded["batch_size_by_arm"][ARM] == 1, folded["batch_size_by_arm"]                   # the MEMPROBE sentinel -> the largest rung that fits with headroom
        assert (REPO / "configs" / "sdfbase_folds.json").read_bytes() == repo_folds_before and CFG.read_bytes() == repo_cfg_before, "this repo stays clean"
    finally:
        subprocess.run(["git", "worktree", "remove", "--force", str(wt)], cwd=REPO, capture_output=True, text=True)
        subprocess.run(["git", "worktree", "prune"], cwd=REPO, capture_output=True, text=True)
        shutil.rmtree(wt, ignore_errors=True)
