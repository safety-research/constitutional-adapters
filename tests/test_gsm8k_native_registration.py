"""gsm8k rows register NATIVELY — the registry is written at every step: like every other wave job,
`scripts/sdfbase/gsm8k/gsm8k_thinking.sbatch` calls `f22_registry_register` the moment its overlay
is resolved and the row is launchable — `emit_cli register --from-config` with the pending file written by the job itself — narrowed to
the ONE cell the row serves (`F22_REGISTRY_CELLS`: a job-local copy of the overlay beside the pending file; the committed configs stay
byte-identical), the eval gsm8k, the twin attempt stamped from the environment, and the TERMINAL run_state filed by the job's own
cleanup (`g8k_registry_end`). The registration path runs for real under bash against a RECORDING `sisteer.registry.emit_cli` module in
a synthetic checkout (no registry, no GPU, no real config); the sbatch's call order is pinned from its text. Synthetic keys/cells only."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from front_door_testlib import REPO, VENV_PY  # noqa: E402

COMMON = REPO / "scripts/sdfbase/full22/common.sh"
SBATCH = REPO / "scripts/sdfbase/gsm8k/gsm8k_thinking.sbatch"
SHA = "c" * 40
OVERLAY = {"model": "synthetic/model", "explicit_cells": ["none", "prompted__a", "prompted__b"], "sampled_decode": {"trial": 0}}

RECORDER = '''
import json, os, sys
rec = os.environ["FAKE_EMIT_RECORD"]
args = sys.argv[1:]
with open(rec, "a") as fh:
    fh.write(json.dumps({"argv": args, "env": {k: v for k, v in os.environ.items() if k.startswith("SISTEER_")}}) + "\\n")
if args and args[0] == "register" and "--from-config" in args:
    pend_dir = args[args.index("--pending-dir") + 1]; job = args[args.index("--job-id") + 1]
    cfg = json.load(open(args[args.index("--from-config") + 1])) if args[args.index("--from-config") + 1].endswith(".json") else None
    os.makedirs(pend_dir, exist_ok=True)
    with open(os.path.join(pend_dir, job + ".json"), "w") as fh:
        json.dump({"run_id": "run-synthetic-1", "job_id": job, "runs": [{"run_id": "run-synthetic-1", "route": {"object": "prompted__a", "eval_id": "gsm8k"}, "plan": {"eval_id": "gsm8k"}}]}, fh)
    print(os.path.join(pend_dir, job + ".json"))
sys.exit(0)
'''


def _world(tmp_path: Path):
    """A synthetic checkout (the REAL full22/common.sh; a recording emit_cli under <repo>/src), a shared root and an overlay."""
    repo = tmp_path / "repo"
    (repo / "src/sisteer/registry").mkdir(parents=True)
    (repo / "src/sisteer/__init__.py").write_text(""); (repo / "src/sisteer/registry/__init__.py").write_text("")
    (repo / "src/sisteer/registry/emit_cli.py").write_text(RECORDER)
    (repo / "scripts/sdfbase/full22").mkdir(parents=True)
    (repo / "scripts/sdfbase/full22/common.sh").write_bytes(COMMON.read_bytes())
    cfg_rel = "configs/sdfbase/gsm8k/sdfbase_full22_m_s1.gsm8k.yaml"
    (repo / cfg_rel).parent.mkdir(parents=True)
    (repo / cfg_rel).write_text(yaml.safe_dump(OVERLAY, sort_keys=False))
    (repo / "configs/registry").mkdir(); (repo / "configs/sdfbase_full22_pins.json").write_text("{}\n")   # the checkout's seeds dir + pins
    shared = tmp_path / "shared"; (shared / "registry/records").mkdir(parents=True)        # OUTSIDE the checkout, as on a cluster
    return repo, shared, cfg_rel


def _register(tmp_path: Path, repo: Path, shared: Path, cfg_rel: str, *, cells: str | None, extra: str = "") -> tuple[subprocess.CompletedProcess, list]:
    rec = tmp_path / "emit.jsonl"
    script = textwrap.dedent(f"""
        set -u
        cd {repo}
        export REPO_DIR={repo} SHARED={shared} SISTEER_REGISTRY_SPOOL={shared}/registry/records FAKE_EMIT_RECORD={rec}
        export SLURM_JOB_ID=4242 SLURM_JOB_NAME=g8k-m-s1-prompted__a WAVE_SHA={SHA} CLUSTER=slurm
        export KEY=m VIEW=s1 TAG=m_s1 CFG={cfg_rel} F22_REGISTRY_EVALS=gsm8k F22_ROSTER=configs/roster.yaml F22_SERVING_PATH=lora_native
        export SISTEER_ATTEMPT_ID=m/s1/gsm8k/prompted__a@cluster_a@1 SISTEER_ATTEMPTS_ROOT={shared}/exp/reproduce/m/attempts
        {"export F22_REGISTRY_CELLS=" + cells if cells is not None else "unset F22_REGISTRY_CELLS"}
        {extra}
        source scripts/sdfbase/full22/common.sh
        f22_registry_register; rc=$?
        echo "RC=$rc IDENT=${{SISTEER_REGISTRY_RUN_IDENTITY:-}}"
    """)
    env = dict(os.environ, PATH=f"{Path(VENV_PY).parent}:{os.environ['PATH']}")   # `python3` = the venv's (PyYAML for the narrowing)
    r = subprocess.run(["bash", "-c", script], capture_output=True, text=True, env=env, timeout=120)
    calls = [json.loads(l) for l in rec.read_text().splitlines()] if rec.exists() else []
    return r, calls


def test_f22_registry_register_narrows_the_overlay_to_the_jobs_cell(tmp_path):
    repo, shared, cfg_rel = _world(tmp_path)
    before = (repo / cfg_rel).read_bytes()
    r, calls = _register(tmp_path, repo, shared, cfg_rel, cells="prompted__a")
    assert r.returncode == 0 and "RC=0" in r.stdout, r.stdout + r.stderr
    narrowed = shared / "registry/pending/4242.config.yaml"
    assert narrowed.is_file(), r.stderr
    doc = yaml.safe_load(narrowed.read_text())
    assert doc["explicit_cells"] == ["prompted__a"] and doc["registry_narrowed_from"] == cfg_rel and doc["model"] == "synthetic/model"
    assert (repo / cfg_rel).read_bytes() == before                                   # the committed overlay is byte-identical
    reg = [c for c in calls if c["argv"][:1] == ["register"]]
    assert len(reg) == 1
    a = reg[0]["argv"]
    assert a[a.index("--from-config") + 1] == str(narrowed)
    for flag, val in (("--evals", "gsm8k"), ("--model-key", "m_s1"), ("--roster-selection", "s1"), ("--job-id", "4242"), ("--code-sha", SHA),
                      ("--pending-dir", f"{shared}/registry/pending"), ("--serving-path", "lora_native"), ("--roster", "configs/roster.yaml")):
        assert a[a.index(flag) + 1] == val, (flag, a)
    # the narrowed copy lives under $SHARED/registry/pending, outside the checkout, and jobplan derives the checkout from the CONFIG's
    # path — so the per-cell call NAMES the checkout's seeds and pins (both u0 waves' gsm8k rows refused "no registry seeds at
    # $SHARED/registry/configs/registry" before this)
    assert a[a.index("--seeds") + 1] == f"{repo}/configs/registry" and a[a.index("--pins") + 1] == f"{repo}/configs/sdfbase_full22_pins.json", a
    assert reg[0]["env"]["SISTEER_ATTEMPT_ID"] == "m/s1/gsm8k/prompted__a@cluster_a@1"    # the twin attempt reaches the registration (attempt_stamp)
    assert f"IDENT={shared}/registry/pending/4242.json" in r.stdout                     # the job's own pending file is its run identity
    st = [c for c in calls if c["argv"][:1] == ["state"]]
    assert st and st[0]["argv"][st[0]["argv"].index("--state") + 1] == "running"       # `running` filed for the registered run


def test_a_cell_the_overlay_does_not_enumerate_registers_nothing_and_stops_the_job(tmp_path):
    repo, shared, cfg_rel = _world(tmp_path)
    r, calls = _register(tmp_path, repo, shared, cfg_rel, cells="prompted__zzz")
    # static-reg-fix-2: a refused registration is FATAL — f22_registry_register exits 1 (f22_registry_refused) before "RC=" is reached
    assert r.returncode == 1 and "RC=" not in r.stdout and "FATAL: registry: self-registration REFUSED" in r.stdout + r.stderr, r.stdout + r.stderr
    assert "prompted__zzz" in r.stderr and "not in" in r.stderr and "cannot narrow" in r.stdout + r.stderr
    assert not [c for c in calls if c["argv"][:1] == ["register"]] and not (shared / "registry/pending/4242.json").exists()


def test_without_cells_the_overlay_is_registered_as_is(tmp_path):
    repo, shared, cfg_rel = _world(tmp_path)
    r, calls = _register(tmp_path, repo, shared, cfg_rel, cells=None)
    assert "RC=0" in r.stdout, r.stdout + r.stderr
    reg = [c for c in calls if c["argv"][:1] == ["register"]][0]["argv"]
    assert reg[reg.index("--from-config") + 1] == cfg_rel                              # every other wave job: byte-for-byte what it was
    assert "--seeds" not in reg and "--pins" not in reg                                 # the checkout derives from the config's own path
    assert not (shared / "registry/pending/4242.config.yaml").exists()


def test_gsm8k_sbatch_registers_after_its_row_resolves_and_files_the_terminal_state():
    src = SBATCH.read_text()
    i_cfg, i_rung, i_reg, i_serve, i_standup = (src.index(s) for s in ("g8k_resolve_config; g8k_out_root", "\ng8k_rung_check\n",
                                                                       "f22_registry_register || true", "\nSERVE_DIR=", "f22_standup attacks"))
    assert i_cfg < i_rung < i_reg < i_serve < i_standup                                # after the overlay + the rung check, before anything is served
    assert 'export F22_REGISTRY_EVALS="${F22_REGISTRY_EVALS:-gsm8k}" F22_REGISTRY_CELLS="$CELL"' in src
    cleanup = src[src.index("cleanup(){"):src.index("trap cleanup EXIT")]
    assert cleanup.index("g8k_attempt_end") < cleanup.index("g8k_registry_end") < cleanup.index('f22_result "$rec"')
    fn = _registry_end_fn(src)
    assert "COMPLETE)" in fn and "generated COMPLETED final" in fn and "STOP_REQUESTED)" in fn and "cancelled CANCELLED final" in fn and "failed" in fn
    # defined right after the trap (bash binds a function at call time; cleanup calls it at exit) so the stretch between the source line and
    # the trap stays helper-free (tests/test_sdfbase_gsm8k_sbatch.py pins that), and before the hook / g8k_init
    assert src.index("trap cleanup EXIT") < src.index("g8k_registry_end(){") < src.index("f22_attempt_wait") < i_reg   # the twin hook gates BEFORE the registration


def _registry_end_fn(src: str) -> str:
    i = src.index("g8k_registry_end(){")
    return src[i:src.index("\n}\n", i) + 3]


def test_g8k_registry_end_maps_the_outcome_and_is_inert_before_registration(tmp_path):
    src = SBATCH.read_text()
    fn = _registry_end_fn(src)
    rec = tmp_path / "state.log"
    def run(result, ident):
        prog = f'f22_registry_state() {{ echo "$*" >> {rec}; }}\n{fn}\n' + (f'export SISTEER_REGISTRY_RUN_IDENTITY={ident}\n' if ident else "") + f'G8K_RESULT={result}\ng8k_registry_end\n'
        assert subprocess.run(["bash", "-c", prog], capture_output=True, text=True).returncode == 0
    run("COMPLETE", "/x/pending/1.json"); run("STOP_REQUESTED", "/x/pending/1.json"); run("DRIVER_FAILED", "/x/pending/1.json"); run("COMPLETE", None)
    assert rec.read_text().splitlines() == ["generated COMPLETED final", "cancelled CANCELLED final", "failed DRIVER_FAILED final"]   # nothing for the unregistered call


def test_front_door_plan_declares_gsm8k_per_cell_so_the_unit_binds():
    sys.path.insert(0, str(REPO / "reproduce/lib"))
    import stage as ST
    import models as MT
    rows = [r for r in ST.unit_plan("qwen35_4b_demo", "battery", table=str(REPO / "reproduce/models.yaml"), sha=SHA) if r["family"] == "gsm8k"]
    cells = MT.model("qwen35_4b_demo", str(REPO / "reproduce/models.yaml"))["views"]["battery"]["cells"]      # the single-phase recipe's one DECLARED wave (`wave: battery`, demo-import M3): none, the prompted legs, every arm
    assert [r["unit"] for r in rows] == cells and cells[:3] == ["none", "prompted__safety_refusal", "prompted__si_constitution"] and len(cells) == 10
    m = MT.model("qwen35_4b_demo", str(REPO / "reproduce/models.yaml"))
    for r in rows:
        assert r["evals"] == ["gsm8k"] and r["env"]["F22_REGISTRY_EVALS"] == "gsm8k" and r["args"][2] == MT.roster_cell_id(m, r["unit"])   # the ROSTER cell id rides the 3rd positional (the unit keeps the table's short name)
        assert r["script"] == "scripts/sdfbase/gsm8k/gsm8k_thinking.sbatch"
