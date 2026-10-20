"""`reproduce/03_hpo.sh --phase probe` under the PAPER's protocol submits the serving probe for the ONE declared pair (demo-import M6;
the owners' Q3 ruling (a), corrected): `scripts/sbatch/serving_probe.sbatch <model> <served> <rev> wo 4`, positionals 4/5 = the set id and
rank READ from the fits config's `train_kwargs.lora_r4_ce.fit_selection` (the sorted `lora_modules` mapped through
`lora_module_sets.MODULE_SETS`: `[o_proj]` -> `wo`) and cross-checked against the row's `lora_architecture` — so the carrier's menu is
that pair alone, where `--probe-width 1` on its own would still run the carrier's default 15-pair menu on one GPU. Before this the
branch printed "not applicable" and submitted nothing, while the compute plan (tools/replan_compute.py --probe), the steered-standup
gate (geometry.standup_adapter_parity_gate) and the padded preflight all need the probe's verdict and staged record.

Executed, not string-matched: the probe-resume harness (tests/test_front_door_probe_resume.py's World — fake ssh / rsync / sbatch /
squeue, the real status reader) with the FD 6 fixture swapped for reproduce/overrides/paper_defaults.env at the site. Pinned:
* a fresh NOT_STARTED record -> ONE sbatch through the funnel: `… serving_probe.sbatch qwen35_4b_demo Qwen/Qwen3.5-4B <rev> wo 4`, the
  header's width by default, `--probe-width 1` -> `--gres=gpu:1 --cpus-per-task=16 --mem=128G`; kind other (it registers nothing);
* the dry run composes the same line and submits nothing;
* a resume keeps the RECORD's sets / ranks from probe_plan.json (a one-pair record resumes `wo 4 tp1 …`; a menu record resumes its menu);
* the FD 6 submission is BYTE-IDENTICAL to today's (no positionals 4/5): the paper line is the fd6 line plus the pair, and the source's
  fresh-submit line still spells `"$MODEL $SRV_REPO $SRV_REV"` (tests/test_serving_probe_parallel.py pins it);
* the pair's derivation (`probe_paper_pair`, run under bash from the stage's own text): the config's fit_selection, the MODULE_SETS id,
  the table cross-check; absent / malformed / unmatched / disagreeing refuse by name, and through the stage a disagreeing table refuses
  with nothing submitted;
* `--phase all` under the paper protocol notes that the probe certifies the one declared pair and does not wait on it (the search
  consumes no menu; design still plans) — never "not applicable".
No new flag on 03 names the pair (`--serving-pair` is replan_compute.py's; `--rank` / `--set` record a search's pick). Synthetic
everything (placeholder shas, TESTHOST, the neutral site labels).
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from front_door_testlib import REPO, REPRO, VENV_PY, write_site  # noqa: E402
from test_front_door_probe_resume import LWID, MODEL, NAME, OVERRIDES as FD6_OVERRIDES, REAL_RSYNC, SRV_ID, World, _attempt, _plan, _scan  # noqa: E402

PAPER_OVERRIDES = REPRO / "overrides" / "paper_defaults.env"             # the paper's protocol: searched clip + two warm-up levels -> PROTOCOL=paper
DEMO_CFG = REPO / "configs" / "sdfbase_qwen35_4b_demo_base_fits.yaml"   # the demo's committed fits config: fit_selection rank 4 / [o_proj]
HPO = REPRO / "03_hpo.sh"
PAIR = ["wo", "4"]                                                        # serving_probe.sbatch positionals 4/5 for the demo


class PaperWorld(World):
    """The probe harness with the FD 6 fixture swapped for the paper's defaults: the site's SISTEER_RUN_OVERRIDES names
    reproduce/overrides/paper_defaults.env, so 03 derives PROTOCOL=paper from the site alone (no --overrides on the command line)."""

    def __init__(self, tmp_path: Path):
        super().__init__(tmp_path)
        self.site = write_site(tmp_path, clusters=("cluster_a",), ssh={"cluster_a": f"local:{self.fake}"}, shared={"cluster_a": str(self.shared)},
                               extra_config=f"SISTEER_RUN_OVERRIDES={PAPER_OVERRIDES}\n")


@pytest.fixture
def world(tmp_path):
    if REAL_RSYNC is None:
        pytest.skip("no rsync on this host")
    return PaperWorld(tmp_path)


def _probe_argv(call: dict) -> tuple[list, list]:
    """(the flags before the script, the positionals after it) of one fake-sbatch call."""
    argv = call["argv"]; i = argv.index("scripts/sbatch/serving_probe.sbatch")
    return argv[:i], argv[i + 1:]


def test_the_committed_demo_config_declares_the_pair_the_probe_certifies():
    """The fixture facts, read from the tree: paper_defaults.env is the paper's protocol (searched clip, empty warm-up), the demo
    config's lora_r4_ce fit_selection is rank 4 / [o_proj], MODULE_SETS maps [o_proj] to `wo`, and the row's lora_architecture agrees."""
    text = PAPER_OVERRIDES.read_text()
    assert re.search(r"^OVERRIDE_HPO_CLIP_MODE=searched\b", text, re.M) and re.search(r"^OVERRIDE_HPO_WARMUP_FRAC=''", text, re.M)
    sel = yaml.safe_load(DEMO_CFG.read_text())["train_kwargs"]["lora_r4_ce"]["fit_selection"]
    assert sel["lora_rank"] == 4 and sel["lora_modules"] == ["o_proj"]
    sys.path.insert(0, str(REPO / "src"))
    from sisteer.extraction.lora_module_sets import MODULE_SETS
    assert MODULE_SETS["wo"] == ("o_proj",)
    sys.path.insert(0, str(REPRO / "lib"))
    import models as M
    la = M.model(MODEL, str(REPRO / "models.yaml"))["lora_architecture"]
    assert la["rank"] == 4 and la["modules"] == ["o_proj"]


def test_not_started_under_the_paper_protocol_submits_the_one_declared_pair(world):
    """The production line, executed: a fresh record under PROTOCOL=paper -> ONE sbatch through lib/slurm.sh with positionals 4/5 `wo 4`
    (the fits config's fit_selection, mapped through MODULE_SETS), the header's width by default, kind other, SERVING_PROBE_OUT as the
    funnel's env prefix; the note says the probe certifies the ONE declared pair; the old "not applicable" is gone."""
    r = world.run()
    out = r.stdout + r.stderr
    assert r.returncode == 0, out
    calls = world.sbatch_calls()
    assert len(calls) == 1 and f"submitted cluster_a {calls[0]['id']} hpo.lora.probe" in out, out
    flags, pos = _probe_argv(calls[0])
    assert pos == [MODEL, SRV_ID, pos[2], *PAIR] and len(pos[2]) == 40, pos                      # the served revision from models.yaml
    assert not any(f.startswith("--gres") or f.startswith("--cpus-per-task") or f.startswith("--mem") for f in flags), flags   # the header's width
    assert "--qos=qos_cluster_a" in flags and "--partition=gpu_cluster_a" in flags and flags[flags.index("-J") + 1] == NAME
    assert not any(f.startswith("--export") or f.startswith("--dependency") for f in flags), flags
    assert calls[0]["env"]["SISTEER_RUN_OVERRIDES"].endswith("/reproduce/overrides/paper_defaults.env")
    # the attempt record: kind other (the probe registers nothing), the args carry the pair, the unit is the fresh probe's
    attempts = [json.loads(ln) for ln in (world.tmp / "state" / MODEL / "attempts.jsonl").read_text().splitlines() if ln.strip()]
    att = [a for a in attempts if a.get("event") == "attempt"]
    assert len(att) == 1 and att[0]["logical_work_id"] == LWID and att[0]["kind"] == "other" and att[0]["script"] == "scripts/sbatch/serving_probe.sbatch"
    assert att[0]["args"] == [MODEL, SRV_ID, pos[2], *PAIR] and att[0]["spec"]["env"] == {"SERVING_PROBE_OUT": str(world.record)}
    # the wording
    assert "the probe certifies the ONE declared pair, set wo / rank 4" in out and "positionals 4/5 = 'wo 4'" in out, out
    assert "not applicable" not in out and "no module-set menu to probe" not in out
    assert "lora_r4_ce fit_selection" in out and "sdfbase_qwen35_4b_demo_base_fits.yaml" in out
    # the status reader ran on the "cluster" over the record, on stdin — the same state machine as FD 6
    assert f"python3 - --record '{world.record}' --job-active 0" in world.ssh_log.read_text()
    _scan(world, out)
    # the SAME invocation again, the job now PENDING under the probe's name: nothing submitted, exit 0 (RUNNING)
    r2 = world.run()
    out2 = r2.stdout + r2.stderr
    assert r2.returncode == 0 and len(world.sbatch_calls()) == 1 and "nothing submitted" in out2, out2


def test_probe_width_1_narrows_the_carrier_to_one_gpu_and_keeps_the_pair(world):
    """`--probe-width 1` -> the carrier's documented narrower allocation on the command line (16 CPUs + 128 G per GPU) beside the SAME
    positionals: the ruling's correction — the width narrows the allocation only; the pair is what narrows the menu."""
    r = world.run("--probe-width", "1")
    out = r.stdout + r.stderr
    assert r.returncode == 0, out
    flags, pos = _probe_argv(world.sbatch_calls()[0])
    assert pos[3:] == PAIR and pos[:2] == [MODEL, SRV_ID]
    assert "--gres=gpu:1" in flags and "--cpus-per-task=16" in flags and "--mem=128G" in flags, flags
    assert "--probe-width 1 -> --gres=gpu:1 --cpus-per-task=16 --mem=128G" in out


def test_the_dry_run_composes_the_same_submission_and_submits_nothing(world):
    """The dry run prints the funnel's line with the pair and the width flags, the registration contract (kind other), and touches
    nothing: no sbatch, no record file, no lock."""
    r = world.run("--dry-run", "--probe-width", "1")
    out = r.stdout + r.stderr
    assert r.returncode == 0, out
    lines = [ln for ln in out.splitlines() if "DRY-RUN: ssh" in ln and "serving_probe.sbatch" in ln]
    assert len(lines) == 1, out
    ln = lines[0]
    m = re.search(r"scripts/sbatch/serving_probe\.sbatch (\S+) (\S+) ([0-9a-f]{40}) ([^\s']+) ([^\s']+)'?$", ln)
    assert m and [m.group(1), m.group(2), m.group(4), m.group(5)] == [MODEL, SRV_ID, *PAIR], ln
    assert "--gres=gpu:1 --cpus-per-task=16 --mem=128G" in ln and "--export=" not in ln and "SERVING_PROBE_OUT=" in ln
    assert "registration contract: hpo.lora.probe registers=derived (kind other, scripts/sbatch/serving_probe.sbatch)" in out
    assert "the probe certifies the ONE declared pair, set wo / rank 4" in out
    assert world.sbatch_calls() == [] and not list(world.record.iterdir()) and not (world.tmp / "state" / MODEL / "03_hpo" / ".probe.lora.recover.lock").exists()
    _scan(world, out)


def test_a_resume_keeps_the_records_own_pair_from_probe_plan_json(world):
    """A resumed paper-protocol record keeps its sets / ranks from probe_plan.json (condition 3): a one-pair record resumes as
    `wo 4 tp1 <attempt> <cap> auto` through the attempt-record path, and a record whose plan is the FD 6 menu resumes ITS menu — the
    record is the authority on a resume, never a re-derivation (and never a fresh probe over a partial record)."""
    one = dict(_plan(), sets=["wo"], ranks=[4])
    world.lay(attempt=dict(_attempt(), sets="wo", ranks="4"))
    (world.record / "probe_plan.json").write_text(json.dumps(one))
    r = world.run()
    out = r.stdout + r.stderr
    assert r.returncode == 0, out
    calls = world.sbatch_calls()
    assert len(calls) == 1, out
    flags, pos = _probe_argv(calls[0])
    assert pos == [MODEL, SRV_ID, pos[2], "wo", "4", "tp1", "3", "4", "auto"], pos
    assert "--gres=gpu:4" in flags and f"submitted cluster_a {calls[0]['id']} hpo.lora.probe.a3" in out
    assert "a resume keeps the record's own probe_plan.json sets / ranks" in out
    # a menu record (the FD 6 plan) under the paper protocol resumes with the record's menu, not the declared pair
    w2 = PaperWorld(world.tmp / "w2"); w2.lay()
    r2 = w2.run()
    out2 = r2.stdout + r2.stderr
    assert r2.returncode == 0, out2
    _, pos2 = _probe_argv(w2.sbatch_calls()[0])
    assert pos2[3:] == ["wo,qkvo,attn_mlp,mlp,down", "4,16,64", "tp1", "3", "4", "auto"], pos2
    _scan(world, out + out2)


def test_fd6_submissions_are_byte_identical_and_the_paper_line_is_the_fd6_line_plus_the_pair(world):
    """The frozen surface: under the FD 6 fixture the fresh submission carries NO positionals 4/5 (the carrier's default menu, the rank
    x module-set search's design space) and the same flags as before; the paper submission is exactly that line plus `wo 4`. The
    source's fresh-submit line still spells `"$MODEL $SRV_REPO $SRV_REV"` (tests/test_serving_probe_parallel.py's pin), the pair riding
    `${PROBE_PAIR:+" $PROBE_PAIR"}` on BOTH fresh-submit lines; probe_resume's line is unchanged."""
    fd6 = World(world.tmp / "fd6")
    r = fd6.run()
    assert r.returncode == 0, r.stdout + r.stderr
    fd6_flags, fd6_pos = _probe_argv(fd6.sbatch_calls()[0])
    assert fd6_pos == [MODEL, SRV_ID, fd6_pos[2]] and len(fd6_pos[2]) == 40
    assert "certifies the ONE declared pair" not in r.stdout + r.stderr and "PROBE_PAIR" not in r.stdout + r.stderr   # fd6 prints no pair note
    rp = world.run()
    assert rp.returncode == 0, rp.stdout + rp.stderr
    paper_flags, paper_pos = _probe_argv(world.sbatch_calls()[0])
    assert paper_pos == fd6_pos + PAIR
    strip = lambda fl: [f for f in fl if not f.startswith("--comment=") and not f.startswith("--output=")]   # the comment carries a per-call nonce; --output= the world's shared root
    assert strip(paper_flags) == strip(fd6_flags), (paper_flags, fd6_flags)
    assert "certifies the ONE declared pair" in rp.stdout + rp.stderr
    # the source: the fd6 spelling stands on both fresh-submit lines, each appending the pair the same way
    src = HPO.read_text()
    probe = src[src.index("# ================================================================================================================ probe"):
                src.index("# ================================================================================================================ recon")]
    fresh = [ln for ln in probe.splitlines() if 'submit "$(unit probe)"' in ln]
    assert len(fresh) == 2 and all('"$MODEL $SRV_REPO $SRV_REV"${PROBE_PAIR:+" $PROBE_PAIR"}' in ln for ln in fresh), fresh
    resume = src[src.index("probe_resume(){"):src.index("PROBE_STATUS_LINE=\"\"")]
    assert '"$MODEL $SRV_REPO $SRV_REV $R_SETS $R_RANKS $R_MODE $R_ATTEMPT $R_MAX $R_PARALLEL' in resume and "PROBE_PAIR" not in resume
    reverdict = src[src.index("probe_submit_reverdict(){"):src.index("probe_parity_record_reopen(){")]
    assert '"$MODEL $SRV_REPO $SRV_REV $sets $ranks tp1' in reverdict and "PROBE_PAIR" not in reverdict
    assert "--serving-pair" not in src                                                                        # replan_compute.py's option, never 03's


def _pair_fragment() -> str:
    src = HPO.read_text()
    return src[src.index("probe_paper_pair(){"):src.index("probe_reopen_token(){")]


def _derive(tmp: Path, cfg_text: str, table_text: str | None = None, model: str = MODEL) -> subprocess.CompletedProcess:
    """RUN the stage's own probe_paper_pair under bash over a synthetic fits config (and table): the function's text, the variables it
    reads, nothing else."""
    tmp.mkdir(parents=True, exist_ok=True)
    cfg = tmp / "fits.yaml"; cfg.write_text(cfg_text)
    table = tmp / "models.yaml"
    table.write_text(table_text if table_text is not None else (REPRO / "models.yaml").read_text())
    script = (f'set -u; PY="{VENV_PY}"; REPRO_REPO="{REPO}"; LOCAL_CFG="{cfg}"; REPRO_LIB_DIR="{REPRO / "lib"}"; MODEL="{model}"\n'
              f'repro_models_table(){{ echo "{table}"; }}\n' + _pair_fragment() + "\nprobe_paper_pair\n")
    return subprocess.run(["bash", "-c", script], capture_output=True, text=True, env={"PATH": os.environ["PATH"], "HOME": str(tmp)}, timeout=120)


def _cfg(rank, modules) -> str:
    mods = "".join(f"      - {m}\n" for m in modules)
    return ("model: example-org/example-base\nmodel_revision: 1111111111111111111111111111111111111111\nsi_ids:\n- s\n"
            "train_kwargs:\n  default:\n    heldout_pool: /vol/pool/example/pairs_constv2-heldout.jsonl\n  psr_ce_all:\n    accum_steps: PENDING_MEMPROBE_EFF64_ACCUM\n"
            f"  lora_r4_ce:\n    accum_steps: PENDING_MEMPROBE_EFF64_ACCUM\n    fit_selection:\n      lora_rank: {rank}\n      lora_modules:\n{mods}")


def _table_with(rank, modules) -> str:
    doc = yaml.safe_load((REPRO / "models.yaml").read_text())
    doc["models"][MODEL]["lora_architecture"] = {"rank": rank, "modules": list(modules), "per_expert_rank": None, "selected_by": "paper", "selection_ruling_id": None}
    return yaml.safe_dump(doc, sort_keys=False)


def test_the_pair_is_the_configs_fit_selection_mapped_through_module_sets_and_cross_checked(tmp_path):
    """probe_paper_pair under bash: the committed demo config -> `wo 4`; a config declaring rank 16 on q/k/v/o (with a table that agrees)
    -> `qkvo 16` — the MODULE_SETS mapping, never a constant; the module order in the config does not matter."""
    r = _derive(tmp_path / "demo", DEMO_CFG.read_text())
    assert r.returncode == 0 and r.stdout.strip() == "wo 4", r.stdout + r.stderr
    r = _derive(tmp_path / "qkvo", _cfg(16, ["v_proj", "o_proj", "q_proj", "k_proj"]), _table_with(16, ["q_proj", "k_proj", "v_proj", "o_proj"]))
    assert r.returncode == 0 and r.stdout.strip() == "qkvo 16", r.stdout + r.stderr
    r = _derive(tmp_path / "down", _cfg(4, ["down_proj"]), _table_with(4, ["down_proj"]))
    assert r.returncode == 0 and r.stdout.strip() == "down 4", r.stdout + r.stderr


def test_the_pair_refuses_by_name_when_absent_malformed_unmatched_or_disagreeing(tmp_path):
    """No fit_selection, a non-integer rank, modules matching no MODULE_SETS set, and a table whose lora_architecture disagrees (rank or
    modules) each refuse by name with nothing on stdout — the pair is never guessed."""
    nosel = _cfg(4, ["o_proj"]).split("    fit_selection:\n", 1)[0]
    r = _derive(tmp_path / "a", nosel)
    assert r.returncode != 0 and not r.stdout.strip() and "no train_kwargs.lora_r4_ce.fit_selection" in r.stderr, r.stderr
    r = _derive(tmp_path / "b", _cfg("four", ["o_proj"]))
    assert r.returncode != 0 and not r.stdout.strip() and "lora_rank='four' is not a positive integer" in r.stderr, r.stderr
    r = _derive(tmp_path / "c", _cfg(4, ["o_proj", "gate_proj"]))
    assert r.returncode != 0 and not r.stdout.strip() and "match no module set of lora_module_sets.MODULE_SETS" in r.stderr and "'wo': ['o_proj']" in r.stderr, r.stderr
    r = _derive(tmp_path / "d", _cfg(4, ["o_proj"]), _table_with(16, ["o_proj"]))
    assert r.returncode != 0 and not r.stdout.strip(), r.stderr
    assert "declares rank 4 / ['o_proj'] (set wo)" in r.stderr and "lora_architecture says rank 16 / ['o_proj']" in r.stderr and "two sources that disagree" in r.stderr, r.stderr
    r = _derive(tmp_path / "e", _cfg(4, ["o_proj"]), _table_with(4, ["down_proj"]))
    assert r.returncode != 0 and not r.stdout.strip() and "two sources that disagree" in r.stderr and "['down_proj']" in r.stderr, r.stderr
    r = _derive(tmp_path / "f", _cfg(4, []))                                                   # an empty `lora_modules:` reads as None
    assert r.returncode != 0 and not r.stdout.strip() and "lora_modules=None is not a non-empty list of module names" in r.stderr, r.stderr


def test_through_the_stage_a_disagreeing_table_refuses_before_anything_is_read_or_submitted(world):
    """End to end: a models table whose demo row says rank 16 while the committed config declares rank 4 -> the probe phase refuses by
    name (exit 2), names both sources, submits nothing and reads no record."""
    table = world.tmp / "models_rank16.yaml"
    table.write_text(_table_with(16, ["o_proj"]))
    r = world.run(SISTEER_MODELS_TABLE=str(table))
    out = r.stdout + r.stderr
    assert r.returncode == 2, out
    assert "REFUSED: probe: the paper protocol's one declared pair cannot be derived" in r.stderr and "two sources that disagree" in r.stderr, out
    assert "declares rank 4 / ['o_proj'] (set wo)" in r.stderr and "rank 16" in r.stderr
    assert world.sbatch_calls() == []
    assert not world.ssh_log.exists() or "--record" not in world.ssh_log.read_text()                          # the record was never read
    assert not (world.tmp / "state" / MODEL / "attempts.jsonl").exists()
    _scan(world, out)


def test_phase_all_under_the_paper_protocol_notes_the_pair_and_does_not_wait_on_the_probe(world):
    """`--phase all` under PROTOCOL=paper: the note says the probe certifies the one declared pair and that the search does not wait on
    it (design / trials consume no menu — condition 4: recon / design under paper stay as they are); no probe is submitted, no record is
    read, the design still plans its trials; "not applicable" is gone."""
    r = subprocess.run(["bash", str(HPO), "--model", MODEL, "--object-class", "lora", "--phase", "all", "--dry-run", "--config", str(world.site / "config.env")],
                       capture_output=True, text=True, env=world.env(), cwd=str(REPO), timeout=600)
    out = r.stdout + r.stderr
    assert r.returncode == 0, out
    assert "the search consumes no module-set menu, so --phase all does not wait on the probe" in out and "run --phase probe [--probe-width 1] before 06" in out, out
    assert "probe: not applicable" not in out and "no module-set menu to probe" not in out
    assert "waiting on the probe" not in out and "serving_probe.sbatch" not in "\n".join(ln for ln in out.splitlines() if "DRY-RUN: ssh" in ln)
    assert "hpo_trial.sbatch" in out                                                                          # the design planned its trials
    assert world.sbatch_calls() == []
    assert not world.ssh_log.exists() or "--record" not in world.ssh_log.read_text()                          # no record read
    assert "recon: not applicable under the paper's protocol" in out                                          # recon under paper stays as it is
    _scan(world, out)
