"""The fits-config uniqueness rule of `reproduce/02_prepare_model.sh` / `reproduce/03_hpo.sh` (lib/common.sh `repro_fits_config_for`).

A stage resolves the model's fits config as the ONE `configs/sdfbase_*_fits.yaml` whose `model` / `model_revision` pin the fit
checkpoint. The paper's Qwen3.5-4B checkpoint is pinned by its canonical config AND by three seed replicates
(make_sdfbase_configs.py's `_eb64_s<N>` sibling keys), so the stage used to REFUSE it: "4 fits configs pin … (expected exactly
one)". The rule now: a seed replicate — a name ending `_s<N>_fits.yaml`, or a declared `train_kwargs.default.seed` — is never a
candidate; the canonical config is the one left; two canonical configs pinning the same checkpoint still refuse by name; none
refuses; `--fits-config <path>` names one explicitly (it must pin the checkpoint and lie under the repo; a replicate may be named).

Driven two ways: the REAL shell function sourced from lib/common.sh over a synthetic `configs/` tree (REPRO_REPO re-pointed), and
the REAL stages' `--dry-run` against the committed tree (the paper's model resolves to its canonical config; the demo key to the
lane named by its fit model_id; the explicit flag threads through). Synthetic everything else."""
from __future__ import annotations

import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from front_door_testlib import REPO, REPRO, VENV_PY, run_stage  # noqa: E402

HF = "example-org/example-base"
REV = "1" * 40
OTHER_REV = "2" * 40


def _cfg(model=HF, rev=REV, seed=None, extra="") -> str:
    default = "    heldout_pool: /vol/pool/x/pairs_constv2-heldout.jsonl\n" + (f"    seed: {seed}\n" if seed is not None else "")
    return f"model: {model}\nmodel_revision: {rev}\ndtype: bfloat16\n{extra}train_kwargs:\n  default:\n{default}"


def _repo(root: Path, files: dict[str, str]) -> Path:
    repo = root / "repo"
    (repo / "configs").mkdir(parents=True)
    for name, text in files.items():
        (repo / "configs" / name).write_text(text)
    return repo


def _resolve(repo: Path, hf: str = HF, rev: str = REV, explicit: str = "", fit_id: str = "") -> subprocess.CompletedProcess:
    """The real function, sourced from lib/common.sh, over `repo` (REPRO_REPO re-pointed after the source)."""
    prog = (f"set -u; source {REPO}/reproduce/lib/common.sh; REPRO_REPO={shlex.quote(str(repo))}; "
            f"repro_fits_config_for {shlex.quote(hf)} {shlex.quote(rev)} {shlex.quote(explicit)} {shlex.quote(fit_id)}")
    env = {"PATH": os.environ["PATH"], "HOME": str(repo), "SISTEER_PYTHON": VENV_PY}
    return subprocess.run(["bash", "-c", prog], capture_output=True, text=True, env=env, timeout=120)


# ------------------------------------------------------------------------------------------------------------------
# the rule, over a synthetic configs/ tree
# ------------------------------------------------------------------------------------------------------------------
def test_the_canonical_config_wins_over_its_seed_replicates(tmp_path):
    repo = _repo(tmp_path, {"sdfbase_m_base_fits.yaml": _cfg(),
                            "sdfbase_m_base_eb64_s1_fits.yaml": _cfg(seed=1),
                            "sdfbase_m_base_eb64_s2_fits.yaml": _cfg(seed=2),
                            "sdfbase_m_base_eb64_s3_fits.yaml": _cfg(seed=3),
                            "sdfbase_other_fits.yaml": _cfg(model="example-org/other")})   # another checkpoint: never a hit
    r = _resolve(repo)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "configs/sdfbase_m_base_fits.yaml", r.stdout


def test_a_replicate_marked_only_by_its_declared_seed_is_not_a_candidate(tmp_path):
    # the generator emits `train_kwargs.default.seed` on a replicate ONLY; a replicate under another name is still excluded by it
    repo = _repo(tmp_path, {"sdfbase_m_base_fits.yaml": _cfg(), "sdfbase_m_base_rep_fits.yaml": _cfg(seed=7)})
    r = _resolve(repo)
    assert r.returncode == 0 and r.stdout.strip() == "configs/sdfbase_m_base_fits.yaml", r.stdout + r.stderr


def test_two_canonical_configs_pinning_the_checkpoint_refuse_by_name(tmp_path):
    repo = _repo(tmp_path, {"sdfbase_m_base_fits.yaml": _cfg(), "sdfbase_m_doc_fits.yaml": _cfg(), "sdfbase_m_base_eb64_s1_fits.yaml": _cfg(seed=1)})
    r = _resolve(repo)
    assert r.returncode != 0 and r.stdout.strip() == "", r.stdout
    assert "2 canonical fits configs pin" in r.stderr and "expected exactly one" in r.stderr, r.stderr
    assert "configs/sdfbase_m_base_fits.yaml" in r.stderr and "configs/sdfbase_m_doc_fits.yaml" in r.stderr, r.stderr
    assert "seed replicates, not candidates: ['configs/sdfbase_m_base_eb64_s1_fits.yaml']" in r.stderr, r.stderr
    assert "--fits-config" in r.stderr


def test_no_config_pinning_the_checkpoint_refuses(tmp_path):
    repo = _repo(tmp_path, {"sdfbase_m_base_fits.yaml": _cfg(rev=OTHER_REV), "sdfbase_m_base_eb64_s1_fits.yaml": _cfg(seed=1)})
    r = _resolve(repo)
    assert r.returncode != 0 and r.stdout.strip() == ""
    assert "0 canonical fits configs pin" in r.stderr, r.stderr
    # only replicates pin it: still refused (a replicate is never silently promoted)
    assert "seed replicates, not candidates: ['configs/sdfbase_m_base_eb64_s1_fits.yaml']" in r.stderr, r.stderr
    r = _resolve(_repo(tmp_path / "empty", {}))
    assert r.returncode != 0 and "0 canonical fits configs pin" in r.stderr


def test_an_explicit_fits_config_wins_when_it_pins_the_checkpoint_and_lies_under_the_repo(tmp_path):
    repo = _repo(tmp_path, {"sdfbase_m_base_fits.yaml": _cfg(), "sdfbase_m_doc_fits.yaml": _cfg(), "sdfbase_m_base_eb64_s2_fits.yaml": _cfg(seed=2)})
    # the ambiguous set is resolved by naming one — a replicate included — relative to the repo or absolute, printed repo-relative
    for explicit in ("configs/sdfbase_m_doc_fits.yaml", str(repo / "configs/sdfbase_m_base_eb64_s2_fits.yaml")):
        r = _resolve(repo, explicit=explicit)
        assert r.returncode == 0, (explicit, r.stderr)
        assert r.stdout.strip() == "configs/" + Path(explicit).name, (explicit, r.stdout)
    # a file pinning another checkpoint is refused by name, nothing printed
    (repo / "configs" / "sdfbase_wrong_fits.yaml").write_text(_cfg(rev=OTHER_REV))
    r = _resolve(repo, explicit="configs/sdfbase_wrong_fits.yaml")
    assert r.returncode != 0 and r.stdout.strip() == "" and "not the model's fit checkpoint" in r.stderr and OTHER_REV in r.stderr, r.stderr
    # a missing file
    r = _resolve(repo, explicit="configs/sdfbase_nope_fits.yaml")
    assert r.returncode != 0 and "no such file" in r.stderr, r.stderr
    # a file outside the repo (the job side reads the config from the checkout)
    outside = tmp_path / "elsewhere_fits.yaml"; outside.write_text(_cfg())
    r = _resolve(repo, explicit=str(outside))
    assert r.returncode != 0 and r.stdout.strip() == "" and "lies outside the repo" in r.stderr, r.stderr


def test_a_config_that_does_not_parse_is_said_by_name(tmp_path):
    repo = _repo(tmp_path, {"sdfbase_m_base_fits.yaml": _cfg(), "sdfbase_broken_fits.yaml": "model: [unclosed\n"})
    r = _resolve(repo)
    assert r.returncode != 0 and "cannot read fits config configs/sdfbase_broken_fits.yaml" in r.stderr, r.stderr


def test_the_lane_named_by_the_fit_model_id_wins_over_a_same_checkpoint_sibling(tmp_path):
    """Two table rows may share one checkpoint under their own keys (the worked example qwen35_4b_demo fits the paper's Qwen3.5-4B-Base as
    qwen35_4b_demo_base beside the paper's qwen35_4b_base): the uniqueness scan alone would refuse both, so the lane NAMED by the table's
    fit model_id is the canonical one when it exists and pins the checkpoint; without the id the scan refuses as before."""
    repo = _repo(tmp_path, {"sdfbase_m_base_fits.yaml": _cfg(), "sdfbase_m_demo_base_fits.yaml": _cfg()})
    r = _resolve(repo, fit_id="m_demo_base")
    assert r.returncode == 0 and r.stdout.strip() == "configs/sdfbase_m_demo_base_fits.yaml", r.stdout + r.stderr
    r = _resolve(repo, fit_id="m_base")
    assert r.returncode == 0 and r.stdout.strip() == "configs/sdfbase_m_base_fits.yaml", r.stdout + r.stderr
    r = _resolve(repo)
    assert r.returncode != 0 and "2 canonical fits configs pin" in r.stderr, r.stderr
    # a fit model_id with no config of its own falls to the scan (one canonical config: resolved)
    repo2 = _repo(tmp_path / "two", {"sdfbase_m_base_fits.yaml": _cfg()})
    r = _resolve(repo2, fit_id="m_other")
    assert r.returncode == 0 and r.stdout.strip() == "configs/sdfbase_m_base_fits.yaml", r.stdout + r.stderr


def test_a_named_lane_that_pins_another_checkpoint_refuses_by_name(tmp_path):
    repo = _repo(tmp_path, {"sdfbase_m_base_fits.yaml": _cfg(), "sdfbase_m_demo_base_fits.yaml": _cfg(rev=OTHER_REV)})
    r = _resolve(repo, fit_id="m_demo_base")
    assert r.returncode != 0 and "sdfbase_m_demo_base_fits.yaml" in r.stderr and "not the model's fit checkpoint" in r.stderr, r.stderr


def test_a_seeded_demo_config_is_still_the_lane_named_by_its_fit_model_id(tmp_path):
    """The demo lane declares `train_kwargs.default.seed: 1` (its fits run at effective batch 64 under seed 1, the
    committed objects' recipe), which the SCAN reads as a seed replicate — but the lane NAMED by the table's fit model_id is resolved
    before the scan, seed or no seed, so the demo key still resolves to its own config; the paper's key still resolves to the canonical
    config, and the bare checkpoint scan now sees ONE canonical config beside the seeded demo (a replicate, not a candidate) instead of
    refusing two."""
    repo = _repo(tmp_path, {"sdfbase_m_base_fits.yaml": _cfg(), "sdfbase_m_demo_base_fits.yaml": _cfg(seed=1)})
    r = _resolve(repo, fit_id="m_demo_base")
    assert r.returncode == 0 and r.stdout.strip() == "configs/sdfbase_m_demo_base_fits.yaml", r.stdout + r.stderr
    r = _resolve(repo, fit_id="m_base")
    assert r.returncode == 0 and r.stdout.strip() == "configs/sdfbase_m_base_fits.yaml", r.stdout + r.stderr
    r = _resolve(repo)
    assert r.returncode == 0 and r.stdout.strip() == "configs/sdfbase_m_base_fits.yaml", r.stdout + r.stderr
    # a seeded demo whose config pins another checkpoint is refused by name, never skipped as a replicate
    repo2 = _repo(tmp_path / "two", {"sdfbase_m_base_fits.yaml": _cfg(), "sdfbase_m_demo_base_fits.yaml": _cfg(rev=OTHER_REV, seed=1)})
    r = _resolve(repo2, fit_id="m_demo_base")
    assert r.returncode != 0 and "sdfbase_m_demo_base_fits.yaml" in r.stderr and "not the model's fit checkpoint" in r.stderr, r.stderr


# ------------------------------------------------------------------------------------------------------------------
# the real stages, --dry-run, against the committed tree
# ------------------------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def site(tmp_path_factory):
    root = tmp_path_factory.mktemp("s02fc")
    site = root / "site"
    shutil.copytree(REPRO / "site.example", site)
    with open(site / "config.env", "a") as fh:
        fh.write(f"\nSISTEER_STATE_DIR={root / 'state'}\nSISTEER_PYTHON={VENV_PY}\nSISTEER_REGISTRY_ROOT={root / 'registry'}\n")
    return site


def _committed_fits_configs_pinning(hf: str, rev: str) -> list[str]:
    import yaml
    out = []
    for p in sorted((REPO / "configs").glob("sdfbase_*_fits.yaml")):
        c = yaml.safe_load(p.read_text()) or {}
        if c.get("model") == hf and c.get("model_revision") == rev:
            out.append(p.name)
    return out


def test_the_worked_example_resolves_to_its_canonical_config_in_02_and_03(site):
    sys.path.insert(0, str(REPRO / "lib"))
    import models as M
    m = M.model("qwen35_4b")["fit"]["checkpoint"]
    pinning = _committed_fits_configs_pinning(m["hf_repo"], m["revision"])
    assert "sdfbase_qwen35_4b_base_fits.yaml" in pinning and len(pinning) >= 2, pinning        # the canonical one beside its replicates
    r = run_stage("02_prepare_model.sh", "--dry-run", "--step", "smoke", "--model", "qwen35_4b", site=site, timeout=600)
    out = r.stdout + r.stderr
    assert r.returncode == 0, out[-2500:]
    assert "fits config configs/sdfbase_qwen35_4b_base_fits.yaml, key qwen35_4b_base" in out, out[-2500:]
    # 03 on the paper's model: its canonical config realises effective batch 8 (2 x 4), so the stage's effective-batch cross-check
    # (demo-import M4; tests/test_hpo_effective_batch_cross_check.py) wants an overrides file that says 8 — the paper defaults' 64 is the
    # demo lane's and the seed replicates' batch, and 03 refuses the disagreement by name
    ovr = site / "overrides" / "paper_lane_eb8.env"
    ovr.write_text(re.sub(r"^OVERRIDE_EFFECTIVE_BATCH=.*$", "OVERRIDE_EFFECTIVE_BATCH=8", (REPRO / "overrides" / "paper_defaults.env").read_text(), flags=re.M)
                   .replace("OVERRIDE_ID=paper_defaults", "OVERRIDE_ID=paper_lane_eb8"))
    r = run_stage("03_hpo.sh", "--dry-run", "--object-class", "relu", "--phase", "all", "--model", "qwen35_4b", "--overrides", str(ovr), site=site, timeout=600)
    assert r.returncode == 0, (r.stdout + r.stderr)[-2500:]


def test_the_seeded_demo_config_resolves_by_its_fit_model_id_in_02_and_03(site):
    """The committed demo config carries `train_kwargs.default.seed: 3` (the fit-your-own default training seed, the seed of the committed
    objects since the one-seed objects/DEMO.json; it was 1, a seed replicate to the bare scan, before) and still resolves for the demo key
    through the lane NAMED by its fit model_id, in 02 and in 03 — where its EFF64 sentinel agrees with the paper defaults'
    OVERRIDE_EFFECTIVE_BATCH=64."""
    import yaml
    sys.path.insert(0, str(REPRO / "lib"))
    import models as M
    fit = M.model("qwen35_4b_demo")["fit"]["checkpoint"]
    assert fit["model_id"] == "qwen35_4b_demo_base"
    cfg = yaml.safe_load((REPO / "configs" / "sdfbase_qwen35_4b_demo_base_fits.yaml").read_text())
    assert (cfg["model"], cfg["model_revision"]) == (fit["hf_repo"], fit["revision"]) and cfg["train_kwargs"]["default"]["seed"] == 3
    r = run_stage("02_prepare_model.sh", "--dry-run", "--step", "smoke", "--model", "qwen35_4b_demo", site=site, timeout=600)
    out = r.stdout + r.stderr
    assert r.returncode == 0, out[-2500:]
    assert "fits config configs/sdfbase_qwen35_4b_demo_base_fits.yaml, key qwen35_4b_demo_base" in out, out[-2500:]
    r = run_stage("03_hpo.sh", "--dry-run", "--object-class", "relu", "--phase", "all", "--model", "qwen35_4b_demo", site=site, timeout=600)
    assert r.returncode == 0, (r.stdout + r.stderr)[-2500:]


def test_the_explicit_flag_threads_through_02_and_03(site):
    rep = "configs/sdfbase_qwen35_4b_base_eb64_s1_fits.yaml"
    assert (REPO / rep).is_file()
    r = run_stage("02_prepare_model.sh", "--dry-run", "--step", "smoke", "--model", "qwen35_4b", "--fits-config", rep, site=site, timeout=600)
    out = r.stdout + r.stderr
    assert r.returncode == 0, out[-2500:]
    assert f"fits config {rep}, key qwen35_4b_base_eb64_s1" in out, out[-2500:]
    r = run_stage("03_hpo.sh", "--dry-run", "--object-class", "relu", "--phase", "all", "--model", "qwen35_4b", f"--fits-config={rep}", site=site, timeout=600)
    assert r.returncode == 0, (r.stdout + r.stderr)[-2500:]
    # a flag naming a config of ANOTHER checkpoint refuses by name
    r = run_stage("02_prepare_model.sh", "--dry-run", "--step", "smoke", "--model", "qwen35_4b", "--fits-config", "configs/sdfbase_llama8b_base_fits.yaml", site=site, timeout=600)
    assert r.returncode != 0 and "not the model's fit checkpoint" in r.stderr, r.stdout + r.stderr


def test_the_demo_key_resolves_to_its_own_lane_beside_the_papers(site):
    """The worked example's fit checkpoint (Qwen/Qwen3.5-4B-Base @ 1001bb4d) is pinned by the paper's canonical config, its three seed
    replicates AND the demo lane's own config: 02 resolves qwen35_4b_demo to configs/sdfbase_qwen35_4b_demo_base_fits.yaml (the lane named
    by its fit model_id) and qwen35_4b to the paper's, with no flag."""
    sys.path.insert(0, str(REPRO / "lib"))
    import models as M
    m = M.model("qwen35_4b_demo")["fit"]["checkpoint"]
    assert m["model_id"] == "qwen35_4b_demo_base"
    pinning = _committed_fits_configs_pinning(m["hf_repo"], m["revision"])
    assert {"sdfbase_qwen35_4b_base_fits.yaml", "sdfbase_qwen35_4b_demo_base_fits.yaml"} <= set(pinning), pinning
    r = run_stage("02_prepare_model.sh", "--dry-run", "--step", "smoke", "--model", "qwen35_4b_demo", site=site, timeout=600)
    out = r.stdout + r.stderr
    assert r.returncode == 0, out[-2500:]
    assert "fits config configs/sdfbase_qwen35_4b_demo_base_fits.yaml, key qwen35_4b_demo_base" in out, out[-2500:]
    r = run_stage("03_hpo.sh", "--dry-run", "--object-class", "relu", "--phase", "all", "--model", "qwen35_4b_demo", site=site, timeout=600)
    assert r.returncode == 0, (r.stdout + r.stderr)[-2500:]


def test_nemotron_resolves_to_its_one_canonical_config(site):
    """nemotron3nano4b's checkpoint is pinned by ONE canonical config since the data-scaling ladder lane (`_doc`) was retired (2026-10-08):
    02 resolves it with no flag; the explicit flag still threads through."""
    sys.path.insert(0, str(REPRO / "lib"))
    import models as M
    m = M.model("nemotron3nano4b")["fit"]["checkpoint"]
    pinning = _committed_fits_configs_pinning(m["hf_repo"], m["revision"])
    assert pinning == ["sdfbase_nemotron3nano4b_doc_full_fits.yaml"], pinning
    r = run_stage("02_prepare_model.sh", "--dry-run", "--step", "smoke", "--model", "nemotron3nano4b", site=site, timeout=600)
    out = r.stdout + r.stderr
    assert r.returncode == 0, out[-2500:]
    assert f"fits config configs/{pinning[0]}, key nemotron3nano4b_doc_full" in out, out[-2500:]
    r = run_stage("02_prepare_model.sh", "--dry-run", "--step", "smoke", "--model", "nemotron3nano4b", "--fits-config", "configs/" + pinning[0], site=site, timeout=600)
    out = r.stdout + r.stderr
    assert r.returncode == 0 and f"fits config configs/{pinning[0]}, key nemotron3nano4b_doc_full" in out, out[-2500:]
