"""reproduce/09_dose_ladder.sh on the worked example's COMMITTED dose copies.

The demo row declares `committed_objects_of`, and objects/MANIFEST.json holds the paper's 0.75 dose_scale copies of both differences, so a
dose view whose staged id `d0p75__<cell_id>` resolves to one (reproduce/tools/import_objects.py cells) is IMPORTED, not built: `build`
writes the copy's ledger row in build_strength.py's own schema (source committed, built false, object_sha256 = the registry id, the
sidecar's dose_scaled_from block) and runs no producer; `stage` copies it from this checkout through stage_objects.py and writes the pins
through import_objects.py pins; `views` builds each selected cluster's bank view through the funnel (strength/build_views.sbatch); `guard`
runs launch_guard.py --cluster <label> once per selected cluster (the site block the only source of host and root) and adds the local
imported check under its own name; a dose with no committed copy still takes the build/stage path; `--step views --resubmit` is the
views units' fenced retry (forwarded per unit to the funnel, which skips a live job and re-mints a terminal one on sacct alone, the
build script being in its non-registering allowlist; refused by name with any other step). Every dry run here is against the
NEUTRAL site example (labels cluster_a / cluster_b); the real runs use a one-cluster `local:` seam and a tmp copy of the pins file — no
host, no cluster, no write to the repository's own files."""
from __future__ import annotations

import json
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from front_door_testlib import LIB, REPO, REPRO, VENV_PY, job_ids, run_stage, set_job_state, write_site  # noqa: E402
from test_submit_twin import SHA as FUNNEL_SHA, World  # noqa: E402  (the funnel's CLI over the front-door fake Slurm)
from _rsync_fallback import rsync_on_path  # noqa: E402,F401  (autouse: a fallback rsync on PATH where the host has none)
from test_import_objects import IMPORT_CONTEXT, _fresh_registry, _run as _import_run  # noqa: E402  (a seeds-built registry with the nine committed objects)

sys.path.insert(0, str(REPO / "src"))
from sisteer.registry import harvest as H  # noqa: E402

KEY = "qwen35_4b_demo"
LORA_SID = "d0p75__constv2-mix-full__lora_r4_ce_minus_ctl"
RELU_SID = "d0p75__constv2-mix-full__psr_ce_all_minus_ctl"
DEMO = {e["paper_label"]: e for e in json.loads((REPO / "objects" / "DEMO.json").read_text())["objects"]}
MANIFEST = {r["object_id"]: r for r in json.loads((REPO / "objects" / "MANIFEST.json").read_text())["objects"]}


def _committed_copies() -> dict:
    """{staged id: (object id, objects dir)} of the two 0.75 copies, read off the manifest + the copies' own dose_scaled_from stamps."""
    out = {}
    for label, sid in (("difference_lora", LORA_SID), ("difference_relu", RELU_SID)):
        parent = DEMO[label]["object_id"]
        hits = []
        for r in MANIFEST.values():
            if r.get("derivation") != "dose_scale" or r.get("paper_label") != label or not r.get("dir"):
                continue
            meta = json.loads((REPO / "objects" / r["dir"] / "derived_meta.json").read_text())
            st = meta.get("dose_scaled_from") or {}
            if st.get("parent_object_sha256") == parent and abs(float(st.get("scale", 0)) - 0.75) < 1e-9:
                hits.append((r["object_id"], r["dir"]))
        assert len(hits) == 1, (label, hits)
        out[sid] = hits[0]
    return out


COPIES = _committed_copies()


@pytest.fixture(scope="module")
def site(tmp_path_factory):
    root = tmp_path_factory.mktemp("dose")
    site = root / "site"
    shutil.copytree(REPRO / "site.example", site)
    with open(site / "config.env", "a") as fh:
        fh.write(f"\nSISTEER_STATE_DIR={root / 'state'}\nSISTEER_PYTHON={VENV_PY}\nSISTEER_REGISTRY_ROOT={root / 'registry'}\n")
    return site


def _dry(site, *extra, alpha="0.75"):
    r = run_stage("09_dose_ladder.sh", "--dry-run", "--alpha", alpha, "--model", KEY, *extra, site=site, timeout=900)
    return r, r.stdout + r.stderr


def _lines(out: str, needle: str):
    return [ln for ln in out.splitlines() if needle in ln]


def test_build_imports_the_committed_copies_and_plans_no_producer(site):
    r, out = _dry(site, "--step", "build")
    assert r.returncode == 0, out[-2500:]
    for sid, (oid, d) in COPIES.items():
        assert f"committed copy objects/{d} ({sid}, object {oid}) — imported, not built" in out, (sid, out[-2500:])
        assert any(f"DRY-RUN: ledger row " in ln and f"{KEY}/{sid} (committed copy objects/{d} imported, not built" in ln for ln in out.splitlines()), sid
    assert "build_strength.py" not in out and "no committed copy" not in out and "REFUSED" not in out and "Traceback" not in out


def test_stage_composes_stage_objects_over_the_committed_cells_and_the_importers_pins(site):
    r, out = _dry(site, "--step", "stage")
    assert r.returncode == 0, out[-2500:]
    so = _lines(out, "scripts/sdfbase/full22/stage_objects.py --site-dir")
    assert len(so) == 1 and f" --target {KEY} --cells " in so[0] and so[0].endswith("09_cells.json --clusters cluster_a,cluster_b"), so
    pulls = _lines(out, "DRY-RUN: pull committed:objects/")
    assert sorted(ln.split("committed:objects/", 1)[1].split(" ")[0] for ln in pulls) == sorted(d for _o, d in COPIES.values()), pulls
    pins = _lines(out, "reproduce/tools/import_objects.py pins --model qwen35_4b_demo")
    assert len(pins) == 1 and "--pins-file" in pins[0] and "--manifest" not in pins[0], pins          # no manifest yet on a dry run: the block alone
    assert "stage_strength.py" not in out and "--pins-from" not in out and "REFUSED" not in out and "Traceback" not in out


def test_views_composes_the_bank_view_sbatch_per_selected_cluster_through_the_funnel(site):
    r, out = _dry(site, "--step", "views")
    assert r.returncode == 0, out[-2500:]
    sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()
    lines = out.splitlines()
    for lab in ("cluster_a", "cluster_b"):
        sb = [ln for ln in lines if f"strength/build_views.sbatch {sha} {KEY} d0p75,d0p75_relu" in ln and f"-J llad-views-{KEY}-d0p75-{lab}" in ln]
        assert len(sb) == 1 and "--time=01:15:00" in sb[0] and "--gres" not in sb[0], (lab, sb)
        assert f"submitted {lab} DRY ladder.views.d0p75.{lab}" in out
    mirrors = [i for i, ln in enumerate(lines) if "push -q" in ln and "sisteer-eval-" in ln]
    first_sub = next(i for i, ln in enumerate(lines) if ln.startswith("submitted "))
    assert len(mirrors) == 2 and max(mirrors) < first_sub                                    # the pinned sha mirrored on every cluster first
    assert "--export=" not in out and "REFUSED" not in out
    r, out = _dry(site, "--step", "views", "--clusters", "cluster_b")
    assert r.returncode == 0 and out.count("submitted ") == 1 and "submitted cluster_b DRY ladder.views.d0p75.cluster_b" in out, out[-1500:]


def test_two_rungs_on_one_cluster_are_two_views_units_with_their_own_job_names(site):
    """The funnel is idempotent by logical work id (a unit already submitted is skipped and its earlier job returned), so a
    views unit keyed by the cluster alone handed the second rung the first rung's job with the first rung's views built. Each rung's dose token
    is in the unit, the logical work id and the job name, so sequential rungs on one cluster are their own submissions."""
    sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()
    seen = []
    for alpha, tok in (("0.5", "d0p5"), ("0.75", "d0p75"), ("1.25", "d1p25")):
        r, out = _dry(site, "--step", "views", "--clusters", "cluster_a", alpha=alpha)
        assert r.returncode == 0, (alpha, out[-2000:])
        subs = [ln for ln in out.splitlines() if ln.startswith("submitted ")]
        assert len(subs) == 1 and subs[0].startswith(f"submitted cluster_a DRY ladder.views.{tok}.cluster_a"), (alpha, subs)
        sb = [ln for ln in out.splitlines() if f"strength/build_views.sbatch {sha} {KEY} {tok},{tok}_relu" in ln and f"-J llad-views-{KEY}-{tok}-cluster_a" in ln]
        assert len(sb) == 1, (alpha, sb)
        assert f"llad-views-{KEY}-{tok}-<cluster>" in out and "skipped " not in out
        seen.append((subs[0].split()[3], sb[0]))
    units = [u for u, _s in seen]
    assert len(set(units)) == 3 and len({s for _u, s in seen}) == 3, seen


def test_views_resubmit_is_forwarded_to_the_funnel_per_unit_and_refused_outside_the_views_step(site):
    """A view-build job that died (a node failure, a missing cluster interpreter) could not be retried —
    09 had no --resubmit and a plain `--step views` re-run skipped the bound unit, so `--step guard` never found its view manifest. Now
    `--step views --resubmit` forwards the funnel's fenced retry PER views unit (the dry run names the flag per unit and the `none` contract
    the proof would run under); without the flag nothing is forwarded; the flag with no views step refuses by name."""
    r, out = _dry(site, "--step", "views", "--resubmit")
    assert r.returncode == 0, out[-2500:]
    assert any("09 dose_ladder:" in ln and " resubmit=1 " in ln for ln in out.splitlines()), out[:600]
    for lab in ("cluster_a", "cluster_b"):
        drs = [ln for ln in out.splitlines() if ln.startswith(f"DRY-RUN: --resubmit {KEY}/ladder/views/d0p75/{lab}:")]
        assert len(drs) == 1 and "registration contract none" in drs[0] and "nothing to re-mint" in drs[0], (lab, drs)      # unbound in a fresh world
        assert f"submitted {lab} DRY ladder.views.d0p75.{lab}" in out
    assert "a job that died: --step views --resubmit" in out and "REFUSED" not in out
    r, out = _dry(site, "--step", "views")
    assert r.returncode == 0 and "DRY-RUN: --resubmit" not in out and any(" resubmit=0 " in ln for ln in out.splitlines()), out[-1500:]
    for step in ("guard", "all", "build,stage", "submit"):
        r, out = _dry(site, "--step", step, "--resubmit")
        assert r.returncode == 2 and "--resubmit is the views step's fenced retry" in out and "pass --step views" in out and "submitted " not in out, (step, out[-800:])
    r, out = _dry(site, "--step", "views,guard", "--resubmit", "--clusters", "cluster_a")
    assert r.returncode == 0 and out.count("DRY-RUN: --resubmit ") == 1 and "launch_guard.py" in out, out[-1500:]


def test_a_dead_dose_view_build_is_re_minted_through_the_funnels_fenced_retry_and_a_live_or_merely_bound_one_is_not(tmp_path):
    """The retry 09 forwards, executed for real against the funnel's CLI with the front-door fake Slurm (a test world never mirrors the real
    repo — mirror_repo.sh refuses under SISTEER_TEST_NO_MIRROR — so the real `--step views` cannot run here; the unit below is 09's own
    composition, pinned against the script's text). A first submission binds the unit; a plain re-run SKIPS it (the funnel's idempotence by
    logical work id); `--resubmit` on a LIVE job skips it by name; once sacct says the job is terminal, `--resubmit` re-mints it on sacct
    alone — strength/build_views.sbatch is in the funnel's non-registering allowlist, so no producer spool, registry database or ingest
    watermark is asked for (before: an unknown script derived `registry`, and the proof refused for want of a spool the job never names)."""
    src = (REPRO / "09_dose_ladder.sh").read_text()
    assert '--script scripts/sdfbase/full22/strength/build_views.sbatch --name "llad-views-$K-$TOK-$l"' in src
    assert '--kind other --time 01:15:00 --wave-sha "$SHA" ${extra[@]+"${extra[@]}"}' in src and '[ "$RESUBMIT" = 1 ] && extra+=(--resubmit)' in src
    w = World(tmp_path)
    (w.fake["cluster_a"] / "policy").write_text("start_on_submit\n")
    views = ["submit", "--unit", "ladder.views.d0p75.cluster_a", "--lwid", "m/ladder/views/d0p75/cluster_a", "--script", "scripts/sdfbase/full22/strength/build_views.sbatch",
             "--name", "llad-views-m-d0p75-cluster_a", "--args", f"{FUNNEL_SHA} m d0p75,d0p75_relu", "--clusters", "cluster_a", "--kind", "other", "--time", "01:15:00",
             "--wave-sha", FUNNEL_SHA]
    r1 = w.run(*views)
    assert r1.returncode == 0 and "submitted cluster_a 1001 ladder.views.d0p75.cluster_a" in r1.stdout, r1.stdout + r1.stderr
    attempts = [e for e in w.ledger() if e["event"] == "attempt"]
    assert len(attempts) == 1 and attempts[0]["registers"] == "none" and attempts[0]["spec"]["registers"] == "none"
    assert attempts[0]["spec"]["script"].endswith("strength/build_views.sbatch")
    # a plain re-run: the bound unit is skipped, its job returned, nothing minted
    r2 = w.run(*views)
    assert r2.returncode == 0 and "skipped cluster_a 1001 ladder.views.d0p75.cluster_a (already submitted" in r2.stdout and "pass --resubmit" in r2.stdout, r2.stdout + r2.stderr
    assert job_ids(w.fake["cluster_a"]) == [1001]
    # --resubmit on a LIVE job: skipped by name, nothing minted — a retry is only of a job proven dead
    r3 = w.run(*views, "--resubmit")
    assert r3.returncode == 0 and "skipped cluster_a 1001 ladder.views.d0p75.cluster_a (still RUNNING" in r3.stdout and job_ids(w.fake["cluster_a"]) == [1001], r3.stdout + r3.stderr
    # the job dies (a node failure); no ingest watermark and no producer spool anywhere — the none contract reads neither
    set_job_state(w.fake["cluster_a"], "1001", "NODE_FAIL", end="2026-01-01T00:02:00")
    assert not (w.tmp / "registry" / "state" / "mirror_status.json").exists() and not (w.shared["cluster_a"] / "registry/records").exists()
    r4 = w.run(*views, "--resubmit")
    out = r4.stdout + r4.stderr
    assert r4.returncode == 0 and "submitted cluster_a 1002 ladder.views.d0p75.cluster_a" in r4.stdout, out
    assert "RESUBMIT m/ladder/views/d0p75/cluster_a" in out and "registers=none (kind other)" in out and "check 1 is the whole proof" in out, out
    attempts = [e for e in w.ledger() if e["event"] == "attempt"]
    assert len(attempts) == 2 and all(a["registers"] == "none" for a in attempts) and job_ids(w.fake["cluster_a"]) == [1001, 1002]
    # the unit's `submitted` binding now reads the new attempt (a `none` re-mint supersedes no registered identity: no superseded_attempt event, as the
    # serving probe's re-mint writes none either); a third plain re-run skips THAT job
    binding = json.loads((w.state / "bindings" / "m__ladder__views__d0p75__cluster_a.submitted.json").read_text())
    assert binding["job_id"] == "1002" and binding["attempt_id"] == "m/ladder/views/d0p75/cluster_a@cluster_a@2"
    r5 = w.run(*views)
    assert r5.returncode == 0 and "skipped cluster_a 1002 ladder.views.d0p75.cluster_a (already submitted" in r5.stdout and job_ids(w.fake["cluster_a"]) == [1001, 1002], r5.stdout
    # the contract, read off the funnel: the build script is `none` by allowlist; an unknown script of the same kind still derives `registry`
    sys.path.insert(0, str(LIB))
    import submit_twin as ST  # noqa: E402
    assert ST.script_registers_nothing("scripts/sdfbase/full22/strength/build_views.sbatch", {}) is True
    assert ST.registration_contract("other", None, script="scripts/sdfbase/full22/strength/build_views.sbatch") == "none"
    assert ST.registration_contract("other", None, script="scripts/sdfbase/full22/strength/not_the_build.sbatch") == "registry"


def test_guard_composes_one_cluster_call_per_selected_cluster_and_the_local_imported_check(site):
    r, out = _dry(site, "--step", "guard")
    assert r.returncode == 0, out[-2500:]
    guards = _lines(out, "strength/launch_guard.py")
    assert len(guards) == 2, guards
    for lab in ("cluster_a", "cluster_b"):
        g = [ln for ln in guards if f" --cluster {lab} " in ln]
        assert len(g) == 1 and f" --site-dir {site} " in g[0] and f" --target {KEY} " in g[0] and "d0p75" in g[0] and "d0p75_relu" in g[0], (lab, guards)
        assert f"09_launch_guard.{lab}.json" in g[0] and "--local-root" not in g[0] and "--host" not in g[0]
    checks = _lines(out, "PASS (imported, local)")
    assert len(checks) == 2 and all(ln.startswith("DRY-RUN: imported check ") for ln in checks)
    for sid, (oid, _d) in COPIES.items():
        assert any(f"/cells/{sid}/object.pt" in ln and f"pins[{KEY}][{sid}]" in ln and oid in ln for ln in checks), sid
    assert "GUARD: PASS" not in out                                                           # the local check is its own line, never the guard's pass
    r, out = _dry(site, "--step", "guard", "--clusters", "cluster_b")
    assert r.returncode == 0 and len(_lines(out, "strength/launch_guard.py")) == 1 and " --cluster cluster_b " in out and " --cluster cluster_a " not in out


def test_a_dose_with_no_committed_copy_still_plans_the_builder_and_the_strength_stager(site):
    r, out = _dry(site, "--step", "build,stage", alpha="0.5")
    assert r.returncode == 0, out[-2500:]
    assert "dose view d0p5: no committed copy" in out and "dose view d0p5_relu: no committed copy" in out
    assert any("build_strength.py" in ln and " --doses 0.5 " in ln and f" --targets {KEY} " in ln for ln in out.splitlines()), out[-2500:]
    assert "strength/stage_strength.py" in out and "--pins-from" in out
    assert "imported, not built" not in out and "stage_objects.py" not in out and "import_objects.py pins" not in out


def test_the_step_grammar_takes_a_comma_list_and_refuses_an_unknown_step(site):
    r, out = _dry(site, "--step", "build,guard")
    assert r.returncode == 0 and "imported, not built" in out and "strength/launch_guard.py" in out and "stage_objects.py" not in out, out[-1500:]
    r, out = _dry(site, "--step", "bogus")
    assert r.returncode == 2 and "--step must be build|stage|configs|views|guard|submit|all" in out and "'bogus'" in out


def _runnable_table(tmp_path: Path) -> dict:
    """The models table with the demo row flipped to run_in_this_work: true (a real 09 run refuses the shipped false), under tmp."""
    table = yaml.safe_load((REPRO / "models.yaml").read_text())
    table["models"][KEY]["run_in_this_work"] = True
    tpath = tmp_path / "models.yaml"; tpath.write_text(yaml.safe_dump(table, sort_keys=False))
    return {"SISTEER_MODELS_TABLE": str(tpath)}


def test_a_real_build_writes_the_committed_ledger_rows_in_the_builders_schema_and_runs_no_producer(tmp_path):
    shared = tmp_path / "shared"; shared.mkdir()
    site = write_site(tmp_path, clusters=("cluster_a",), ssh={"cluster_a": f"local:{shared}"}, shared={"cluster_a": str(shared)})
    env = _runnable_table(tmp_path)
    r = run_stage("09_dose_ladder.sh", "--alpha", "0.75", "--model", KEY, "--step", "build", site=site, env=env, timeout=900)
    out = r.stdout + r.stderr
    assert r.returncode == 0, out[-2500:]
    ledger_p = tmp_path / "state" / KEY / "09_build_ledger.json"
    assert ledger_p.is_file() and "build_strength.py" not in out and out.count("ledger: ") == 2, out[-2500:]
    ledger = json.loads(ledger_p.read_text())
    assert ledger["doses"] == [0.75] and ledger["bank"] == str(tmp_path / "state" / "objects") and ledger["absent"] == []
    rows = {c["staged_id"]: c for c in ledger["cells"]}
    assert set(rows) == set(COPIES)
    assert [c["staged_id"] for c in ledger["cells"]] == [c["staged_id"] for c in sorted(ledger["cells"], key=lambda c: (c["target"], c["view"], c["cell_id"]))]
    for sid, (oid, d) in COPIES.items():
        c = rows[sid]
        stamp = json.loads((REPO / "objects" / d / "derived_meta.json").read_text())["dose_scaled_from"]
        assert c["source"] == "committed" and c["built"] is False and c["object_sha256"] == oid and c["committed_dir"] == f"objects/{d}"
        assert c["target"] == KEY and c["staged_id"] == sid and c["cell_id"] == sid.split("__", 1)[1] and c["view"] == ("d0p75_relu" if "psr" in sid else "d0p75")
        assert c["scale"] == 0.75 == stamp["scale"] and c["dose_scaled_from"] == stamp and c["identity_sidecar"] == "derived_meta.json"
        assert c["parent_object_sha256"] == stamp["parent_object_sha256"] == DEMO["difference_relu" if "psr" in sid else "difference_lora"]["object_id"]
        for k in ("parent_staged_id", "producer", "object_type", "scaling_method", "tensors_scaled", "verification", "producer_commit"):
            assert c[k] == stamp[k], (sid, k)                                          # the builder's own fields, from the sidecar's stamp
        assert c["out"] == str(tmp_path / "state" / "objects" / KEY / "cells" / sid)   # the bank dir --step stage fills: stage_strength.py's read
        assert c["built_at"] == stamp["written_at"] and c["family"] == "committed" and c["anchor"] is False
    # idempotent: a second build rewrites the same two rows, never a duplicate
    r = run_stage("09_dose_ladder.sh", "--alpha", "0.75", "--model", KEY, "--step", "build", site=site, env=env, timeout=900)
    assert r.returncode == 0 and len(json.loads(ledger_p.read_text())["cells"]) == 2


def test_a_real_stage_copies_the_committed_copies_through_stage_objects_and_writes_the_demo_pins_block(tmp_path):
    shared = tmp_path / "shared"; shared.mkdir()
    site = write_site(tmp_path, clusters=("cluster_a",), ssh={"cluster_a": f"local:{shared}"}, shared={"cluster_a": str(shared)})
    # the pins are written by the ONE registry-backed writer (import_objects.py pins -> make_full22_configs.pins_from_staging_manifest, scoped to
    # the key): the site's registry (write_site points SISTEER_REGISTRY_ROOT at <root>/registry) is a seeds-built one holding the nine committed
    # objects — `import_objects.py registry` imported them, as the committed sequence's first step does
    root = _fresh_registry(tmp_path / "registry")
    r = _import_run("registry", "--model", KEY, "--registry-root", str(root), "--spool", str(root / "records"), env=IMPORT_CONTEXT)
    assert r.returncode == 0 and r.stdout.count("emitted ") == 9, r.stdout[-2000:] + r.stderr[-2000:]
    rep = H.ingest_once(str(root), workers=0, min_free=0)
    assert rep.refused == 0 and rep.blocked == 0, rep.refusals
    pins_copy = tmp_path / "pins.json"; shutil.copy(REPO / "configs" / "sdfbase_full22_pins.json", pins_copy)
    before = pins_copy.read_text()
    env = {**_runnable_table(tmp_path), "SISTEER_PINS_FILE": str(pins_copy)}
    r = run_stage("09_dose_ladder.sh", "--alpha", "0.75", "--model", KEY, "--step", "stage", "--clusters", "cluster_a", site=site, env=env, timeout=900)
    out = r.stdout + r.stderr
    assert r.returncode == 0, out[-3000:]
    assert "stage_strength.py" not in out and "--pins-from" not in out and "NOT READY" not in out and "REFUSED" not in out, out[-3000:]
    cells = json.loads((tmp_path / "state" / KEY / "09_cells.json").read_text())
    assert sorted(f"{c['lane']}__{c['cell_id']}" for c in cells) == sorted(COPIES) and all(c["source"].startswith("committed:objects/") for c in cells)
    bank = tmp_path / "state" / "objects"
    for sid, (oid, d) in COPIES.items():
        cell_dir = bank / KEY / "cells" / sid
        assert (cell_dir / "object.pt").is_file() and (cell_dir / "derived_meta.json").is_file() and (cell_dir / "staged_from.json").is_file(), sid
        assert (shared / "exp/objects/sdfbase/full22" / KEY / "cells" / sid / "object.pt").is_file()
        stamp = json.loads((cell_dir / "staged_from.json").read_text())
        assert stamp["source"] == f"committed:objects/{d}"
    man = [row for row in json.loads((bank / "staging_manifest.json").read_text()) if row["target"] == KEY]
    assert sorted(row["staged_id"] for row in man) == sorted(COPIES) and all(row["status"] == "staged" and row["source_path"].startswith("committed:") for row in man)
    # the pins: the demo block written by the ONE registry-backed writer — each copy pinned to its committed id, the committed entry's source kept
    # VERBATIM (`objects/MANIFEST.json:<id12> (dose 0.75 copy of objects/DEMO.json:<label>)`: a replay never churns the spelling); the repository's
    # own pins file untouched
    after = json.loads(pins_copy.read_text())
    for sid, (oid, d) in COPIES.items():
        entry = after[KEY][sid]
        assert entry["object_sha256"] == oid and entry["source"].startswith(f"objects/MANIFEST.json:{oid[:12]}") and "objects/DEMO.json:" in entry["source"], entry
    assert json.loads(before)[KEY][LORA_SID]["object_sha256"] == COPIES[LORA_SID][0]            # the committed block already pinned the copies: the write is a no-op by content
    assert json.loads(pins_copy.read_text()) == json.loads(before)
    assert subprocess.run(["git", "status", "--porcelain", "--", "configs/sdfbase_full22_pins.json"], cwd=REPO, capture_output=True, text=True).stdout == ""


def test_the_dose_ladder_builds_in_the_sites_objects_bank_where_05_staged_the_parents(tmp_path):
    """The site-wide bank left the dose ladder on another bank: lib/common.sh's SISTEER_OBJECTS_BANK is the ONE root
    05_stage_roster.sh stages into, 08_grade.sh hands the Petri drive and 09 builds the dose copies in — 09 alone read a singular
    `SISTEER_OBJECT_BANK` nobody documented or set, so a site naming a custom bank had 05 stage the parents there while 09's
    uncommitted dose build pointed build_strength.py's --bank / --parent-root at the old default (missing parents, or an older bank's).
    The REAL 05 stage under a custom bank, then: the real 09 build of the committed 0.75 copies writes the ledger's bank and every row's
    `out` under it; the dry 09 build / stage of an uncommitted dose names the builder's --bank and --parent-root as the directory 05 FILLED
    (every parent of the dose copies is there) and the strength stager's / pins writer's bank the same; the dry guard hashes the imported
    copies under it; the old default is named nowhere and holds nothing; and the stale singular refuses by name in 05, 08 and 09 alike."""
    shared = tmp_path / "shared"; shared.mkdir()
    bank = tmp_path / "site_bank"
    default_bank = tmp_path / "state" / "objects"
    site = write_site(tmp_path, clusters=("cluster_a",), ssh={"cluster_a": f"local:{shared}"}, shared={"cluster_a": str(shared)},
                      extra_config=f"SISTEER_OBJECTS_BANK={bank}")
    env = _runnable_table(tmp_path)
    # 05 stages the demo's committed objects into the SITE's bank — the parents of both dose copies among them
    r = run_stage("05_stage_roster.sh", "--model", KEY, "--step", "stage", "--clusters", "cluster_a", site=site, env=env, timeout=900)
    out = r.stdout + r.stderr
    assert r.returncode == 0, out[-3000:]
    assert f" bank={bank} " in out and f"-> bank {bank} -> clusters cluster_a" in out, out[-3000:]
    cells05 = json.loads((tmp_path / "state" / KEY / "05_cells.json").read_text())
    staged = {f"{c['lane']}__{c['cell_id']}" for c in cells05}
    # the copies' parents BY REGISTRY ID (the committed stamps' parent_object_sha256; their parent_staged_id spells the PAPER's lanes, while the
    # demo table's `derived__` cells carry the same two objects): the two differences, staged by 05 under the site's bank
    parent_ids = {json.loads((REPO / "objects" / d / "derived_meta.json").read_text())["dose_scaled_from"]["parent_object_sha256"] for _, d in COPIES.values()}
    parents = {f"{c['lane']}__{c['cell_id']}" for c in cells05 if c.get("object_sha256_expected") in parent_ids}
    assert len(parent_ids) == 2 == len(parents) and parents <= staged, (parent_ids, parents, staged)
    for sid in staged:
        assert (bank / KEY / "cells" / sid / "object.pt").is_file(), sid
    assert not default_bank.exists()
    # the real 09 build of the committed copies: the ledger's bank and every `out` (the dir --step stage fills, stage_strength.py's read) are the site's bank
    r = run_stage("09_dose_ladder.sh", "--alpha", "0.75", "--model", KEY, "--step", "build", site=site, env=env, timeout=900)
    out = r.stdout + r.stderr
    assert r.returncode == 0, out[-3000:]
    ledger = json.loads((tmp_path / "state" / KEY / "09_build_ledger.json").read_text())
    assert ledger["bank"] == str(bank) and {c["out"] for c in ledger["cells"]} == {str(bank / KEY / "cells" / sid) for sid in COPIES}, ledger
    assert str(default_bank) not in out and not default_bank.exists()
    # an uncommitted dose: the builder's --bank is the site's and its --parent-root IS the directory 05 filled; the stager and the pins writer read the same bank
    r = run_stage("09_dose_ladder.sh", "--dry-run", "--alpha", "0.5", "--model", KEY, "--step", "build,stage", site=site, timeout=900)
    out = r.stdout + r.stderr
    assert r.returncode == 0, out[-3000:]
    (build,) = [ln for ln in out.splitlines() if ln.startswith("DRY-RUN:") and "build_strength.py" in ln]
    argv = shlex.split(build[len("DRY-RUN:"):])
    assert argv[argv.index("--bank") + 1] == str(bank) and argv[argv.index("--targets") + 1] == KEY and argv[argv.index("--doses") + 1] == "0.5", argv
    parent_root = Path(argv[argv.index("--parent-root") + 1])
    assert parent_root == bank / KEY / "cells" and all((parent_root / p / "object.pt").is_file() for p in parents), (parent_root, parents)
    (stager,) = [ln for ln in out.splitlines() if ln.startswith("DRY-RUN:") and "stage_strength.py" in ln]
    sargv = shlex.split(stager[len("DRY-RUN:"):])
    assert sargv[sargv.index("--bank") + 1] == str(bank) and sargv[sargv.index("--ledger") + 1] == str(tmp_path / "state" / KEY / "09_build_ledger.json"), sargv
    assert any(ln.startswith("DRY-RUN:") and "make_full22_configs.py" in ln and f"--pins-from {bank}/staging_manifest.json" in ln for ln in out.splitlines()), out[-3000:]
    assert str(default_bank) not in out
    # the guard's local imported check hashes the copies under the site's bank
    r = run_stage("09_dose_ladder.sh", "--dry-run", "--alpha", "0.75", "--model", KEY, "--step", "guard", site=site, timeout=900)
    out = r.stdout + r.stderr
    assert r.returncode == 0, out[-3000:]
    checks = [ln for ln in out.splitlines() if ln.startswith("DRY-RUN: imported check")]
    assert len(checks) == 2 and all(f"sha256({bank}/{KEY}/cells/{sid}/object.pt)" in "\n".join(checks) for sid in COPIES), checks
    assert str(default_bank) not in out
    # the stale singular is refused by name wherever the site config loads (lib/common.sh), never read as a second bank
    for stage, extra in (("09_dose_ladder.sh", ()), ("05_stage_roster.sh", ("--step", "freeze")), ("08_grade.sh", ("--once",))):
        r = run_stage(stage, "--dry-run", "--model", KEY, *extra, site=site, env={"SISTEER_OBJECT_BANK": str(tmp_path / "elsewhere")}, timeout=300)
        out = r.stdout + r.stderr
        assert r.returncode == 2 and "REFUSED: SISTEER_OBJECT_BANK is not a site key" in out and "SISTEER_OBJECTS_BANK" in out and "09_dose_ladder.sh" in out, (stage, out[-1500:])
        assert str(tmp_path / "elsewhere") not in out
