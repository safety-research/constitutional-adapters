"""The Petri proxy's walltime is its OWN site key, SISTEER_TIME_PETRI, default a day (petri-completion, 2026-10-13).

06_eval_waves.sh handed SISTEER_TIME_EVAL to every family but gsm8k, and sbatch's command-line --time overrides the proxy's own
`#SBATCH --time=1-00:00:00`; a site that set the eval walltime to 04:00:00 for its other units therefore cut a FULL Petri drive at
521 of 660 scenarios (the proxy TIMEOUT'd at 3 h 47; every paper proxy ran at a day). The plan's petri rows now carry
`time_key: petri`, 06 maps it to SISTEER_TIME_PETRI (never SISTEER_TIME_EVAL), the example site documents the paper's day, and
lib/common.sh defaults it when a site is silent — the proxy frees its GPU when the drive ends, so the day is a ceiling, not a cost.
Executed through the REAL stage's dry run on a copy of reproduce/site.example; synthetic site, neutral labels."""
from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from front_door_testlib import LIB, REPRO, VENV_PY, run_stage  # noqa: E402

sys.path.insert(0, str(LIB))
import stage as ST  # noqa: E402

MODEL, VIEW = "qwen35_4b_demo", "battery"      # the front-door model and its one wave


def _site(root: Path, extra: str = "") -> Path:
    site = root / "site"
    shutil.copytree(REPRO / "site.example", site)
    with open(site / "config.env", "a") as fh:
        fh.write(f"\nSISTEER_STATE_DIR={root / 'state'}\nSISTEER_PYTHON={VENV_PY}\nSISTEER_REGISTRY_ROOT={root / 'registry'}\n{extra}")
    return site


def _sbatch_lines(out: str) -> list[str]:
    return [ln for ln in out.splitlines() if "sbatch" in ln and ".sbatch" in ln]


def _time_of(line: str) -> str:
    m = re.search(r"--time=(\S+)", line)
    assert m, line
    return m.group(1)


def _dry(site: Path, *families: str):
    r = run_stage("06_eval_waves.sh", "--dry-run", "--model", MODEL, "--view", VIEW, "--family", ",".join(families), site=site, timeout=600)
    out = r.stdout + r.stderr
    assert r.returncode == 0, out[-2500:]
    return out


def test_example_site_petri_time_is_the_papers_day(tmp_path):
    assert re.search(r"^SISTEER_TIME_PETRI=1-00:00:00\b", (REPRO / "site.example" / "config.env").read_text(), re.M), \
        "site.example/config.env documents SISTEER_TIME_PETRI at the paper's 1-00:00:00"
    out = _dry(_site(tmp_path), "petri")
    petri = [ln for ln in _sbatch_lines(out) if "petri_proxy.sbatch" in ln]
    assert len(petri) == 1, out[-2500:]
    assert _time_of(petri[0]) == "1-00:00:00", petri[0]


def test_the_petri_walltime_is_its_own_key_not_the_eval_one(tmp_path):
    """The split that bit on 2026-10-13: the site's eval walltime sizes every other eval unit and never reaches the Petri proxy."""
    out = _dry(_site(tmp_path, "SISTEER_TIME_EVAL=04:00:00\nSISTEER_TIME_PETRI=16:00:00\n"), "petri", "static")
    lines = _sbatch_lines(out)
    petri = [ln for ln in lines if "petri_proxy.sbatch" in ln]
    static = [ln for ln in lines if "petri_proxy.sbatch" not in ln]
    assert len(petri) == 1 and static, out[-2500:]
    assert _time_of(petri[0]) == "16:00:00", petri[0]
    assert all(_time_of(ln) == "04:00:00" for ln in static), [ln for ln in static if _time_of(ln) != "04:00:00"]
    assert "04:00:00" not in petri[0]


def test_the_default_applies_when_the_site_is_silent(tmp_path):
    """A site that sets the eval walltime and says nothing about Petri: lib/common.sh's default, the paper's day."""
    out = _dry(_site(tmp_path, "SISTEER_TIME_EVAL=04:00:00\n"), "petri", "static")
    petri = [ln for ln in _sbatch_lines(out) if "petri_proxy.sbatch" in ln]
    assert len(petri) == 1 and _time_of(petri[0]) == "1-00:00:00", petri
    assert all(_time_of(ln) == "04:00:00" for ln in _sbatch_lines(out) if "petri_proxy.sbatch" not in ln)


def test_plan_time_keys():
    """The seam: every petri row carries `time_key: petri` (the first pass and the overflow pass), every gsm8k row `gsm8k`, every other row `eval`."""
    sha = "0" * 40
    rows = ST.unit_plan(MODEL, VIEW, families=["petri", "gsm8k", "static", "controls", "mmlu", "agentic"], sha=sha, clusters=["cluster_a", "cluster_b"])
    keys = {(r["family"], r["time_key"]) for r in rows}
    assert ("petri", ST.PETRI_TIME_KEY) in keys and ("gsm8k", "gsm8k") in keys
    assert all(r["time_key"] == "eval" for r in rows if r["family"] not in ("petri", "gsm8k")), keys
    assert all(r["time_key"] == ST.PETRI_TIME_KEY for r in rows if r["family"] == "petri")
    over = ST.overflow_rows(MODEL, VIEW, ["none", "prompted__si_constitution"], cluster="cluster_a", sha=sha)
    assert len(over) == 1 and over[0]["time_key"] == ST.PETRI_TIME_KEY and over[0]["pass"] == "overflow" and over[0]["cluster"] == "cluster_a"
    assert ST.PETRI_TIME_KEY == "petri" and ST.PETRI_PROXY_SCRIPT == "scripts/sdfbase/full22/petri_proxy.sbatch"
    # lib/common.sh exports the default beside the other four; 06's switch maps the key to it
    common = (LIB / "common.sh").read_text()
    assert 'export SISTEER_TIME_PETRI="${SISTEER_TIME_PETRI:-1-00:00:00}"' in common
    six = (REPRO / "06_eval_waves.sh").read_text()
    assert 'petri) t="$SISTEER_TIME_PETRI";;' in six and 'gsm8k) t="0-01:30:00";;' in six and '*) t="$SISTEER_TIME_EVAL";;' in six
