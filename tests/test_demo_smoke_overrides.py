"""reproduce/overrides/demo_smoke.env — the worked example's documented SMOKE recipe (demo-import M10; decisions F6): the paper's
defaults, knob for knob, plus `OVERRIDE_MODEL` (the planner's reader scopes the smoke knob by the document's model, the 30-fix rule) and
the ONE smoke knob `OVERRIDE_SMOKE_GOALS=0-0/99`. Every registry row a smoke produces is stamped run_overrides.id = demo_smoke, which is
how a smoke row is told from a paper-defaults row; 03_hpo.sh's protocol case derives `paper` from it (searched clip + empty warm-up), so
nothing a smoke fits is spelled under another protocol."""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from front_door_testlib import REPO, REPRO, VENV_PY, run_stage  # noqa: E402

sys.path.insert(0, str(REPRO / "lib"))
import models as M  # noqa: E402
import stage as ST  # noqa: E402

sys.path.insert(0, str(REPO / "src"))
from sisteer.registry import emit as EM  # noqa: E402

DEMO_SMOKE = REPRO / "overrides" / "demo_smoke.env"
PAPER = REPRO / "overrides" / "paper_defaults.env"
MODEL = "qwen35_4b_demo"
SMOKE_ONLY = {ST.SMOKE_GOALS_KEY, ST.HOLD_OUT_MODEL_KEY}       # the two keys the smoke document adds beside its own OVERRIDE_ID


def _shell_values(path: Path) -> dict:
    """The OVERRIDE_* values as a stage reads them: `set -a; source` (lib/common.sh repro_load_overrides), printed one per line."""
    script = f'set -a; source "{path}"; set +a; for k in "${{!OVERRIDE_@}}"; do printf "%s=%s\\n" "$k" "${{!k}}"; done'
    out = subprocess.run(["bash", "-c", script], capture_output=True, text=True, check=True).stdout
    return dict(ln.split("=", 1) for ln in out.splitlines() if ln)


def test_the_shipped_overrides_directory_holds_the_paper_defaults_and_the_smoke_recipe_only():
    assert sorted(p.name for p in (REPRO / "overrides").iterdir() if p.is_file()) == ["demo_smoke.env", "paper_defaults.env"]


def test_every_non_smoke_knob_is_the_papers_verbatim_key_for_key():
    demo, paper = ST.parse_env_file(str(DEMO_SMOKE)), ST.parse_env_file(str(PAPER))
    assert {k: v for k, v in demo.items() if k not in SMOKE_ONLY | {"OVERRIDE_ID"}} == {k: v for k, v in paper.items() if k != "OVERRIDE_ID"}
    assert [k for k in demo if k not in SMOKE_ONLY] == list(paper)                       # the paper's order too, OVERRIDE_ID first
    assert set(demo) - set(paper) == SMOKE_ONLY
    # the stage's own reading (the sourced shell) agrees with the planner's reader on every value, both files
    for p in (DEMO_SMOKE, PAPER):
        assert _shell_values(p) == ST.parse_env_file(str(p)), p.name
    # the knobs 03 derives the paper protocol from, spelled exactly as paper_defaults.env spells them
    assert demo["OVERRIDE_HPO_CLIP_MODE"] == "searched" and demo["OVERRIDE_HPO_WARMUP_FRAC"] == "" and demo["OVERRIDE_HPO_TRIALS"] == "16"
    assert demo["OVERRIDE_EFFECTIVE_BATCH"] == "64" and demo["OVERRIDE_REFUSAL_STEER_ALPHA"] == "0.3" and demo["OVERRIDE_LORA_ARCH_SEARCH"] == "0"


def test_the_id_names_a_smoke_the_model_is_the_worked_example_and_the_knob_is_the_one_value():
    demo = ST.parse_env_file(str(DEMO_SMOKE))
    assert demo["OVERRIDE_ID"] == "demo_smoke" and ST.SMOKE_OVERRIDE_ID_MARK in demo["OVERRIDE_ID"]
    assert demo[ST.HOLD_OUT_MODEL_KEY] == MODEL and MODEL in M.front_door_keys(str(REPRO / "models.yaml"))
    assert demo[ST.SMOKE_GOALS_KEY] == "0-0/99"                                           # N <= 99: the one value multi-turn and padding share
    sg = ST.smoke_goals(str(DEMO_SMOKE), MODEL)
    assert sg["spec"] == "0-0/99" and sg["nshards"] == 99 and sg["shards"] == ["shard00"] and sg["override_id"] == "demo_smoke" and sg["model"] == MODEL
    assert ST.hold_out_evals(str(DEMO_SMOKE), MODEL) == {}                                # the smoke holds nothing out of the battery
    with pytest.raises(ValueError, match="never travels to another model"):
        ST.smoke_goals(str(DEMO_SMOKE), "gemma4e4b")


def test_the_registry_stamp_marks_every_smoke_row_with_the_documents_id_and_the_knob():
    """The job-side stamp (sisteer.registry.emit.run_overrides_stamp over SISTEER_RUN_OVERRIDES) is what marks a registry row demo_smoke. Its
    reader keeps an inline ` # comment` after an unquoted value as part of the value, so the three load-bearing keys stand on comment-free
    lines in the document and reach the stamp verbatim (the file pins that below)."""
    stamp = EM.run_overrides_stamp(str(DEMO_SMOKE))
    assert stamp["id"] == "demo_smoke" and stamp["path"] == "demo_smoke.env" and len(stamp["sha256"]) == 64
    assert stamp["keys"][ST.SMOKE_GOALS_KEY] == "0-0/99" and stamp["keys"][ST.HOLD_OUT_MODEL_KEY] == MODEL and stamp["keys"]["OVERRIDE_ID"] == "demo_smoke"
    assert stamp["sha256"] == ST.smoke_goals(str(DEMO_SMOKE), MODEL)["overrides_sha256"]   # the header's sha256 is the stamp's
    for key in ("OVERRIDE_ID", ST.HOLD_OUT_MODEL_KEY, ST.SMOKE_GOALS_KEY):
        assert re.search(rf"^{key}=[^ #]+$", DEMO_SMOKE.read_text(), re.M), key


def _protocol_case() -> str:
    """03_hpo.sh's protocol derivation, lifted from the stage's own text: the two reads and the `case` that names the design."""
    src = (REPRO / "03_hpo.sh").read_text()
    reads = re.search(r'^WARMUP="\$\{OVERRIDE_HPO_WARMUP_FRAC:-\}"\nCLIP_MODE="\$\{OVERRIDE_HPO_CLIP_MODE:-searched\}"\n', src, re.M)
    case = re.search(r'^case "\$CLIP_MODE:\$\{WARMUP:\+set\}" in\n.*?\nesac\n', src, re.M | re.S)
    assert reads and case, "03_hpo.sh's protocol case moved"
    return reads.group(0) + case.group(0)


def _protocol_of(path: Path) -> str:
    script = f'repro_fatal(){{ echo "FATAL: $*" >&2; exit 3; }}\nset -a; source "{path}"; set +a\n{_protocol_case()}echo "PROTOCOL=$PROTOCOL"'
    r = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else f"rc={r.returncode} {r.stderr.strip()}"


def test_sourcing_it_through_03s_protocol_case_yields_paper(tmp_path):
    assert _protocol_of(DEMO_SMOKE) == "PROTOCOL=paper" and _protocol_of(PAPER) == "PROTOCOL=paper"
    mixed = tmp_path / "mixed.env"                                                        # the harness is live: a set warm-up under the searched clip refuses
    mixed.write_text(DEMO_SMOKE.read_text().replace("OVERRIDE_HPO_WARMUP_FRAC=''", "OVERRIDE_HPO_WARMUP_FRAC=0.15"))
    assert _protocol_of(mixed).startswith("rc=3 FATAL: overrides demo_smoke: OVERRIDE_HPO_CLIP_MODE=searched with OVERRIDE_HPO_WARMUP_FRAC='0.15'")


def test_03_accepts_the_smoke_document_on_the_worked_example_under_the_paper_protocol(tmp_path):
    root = tmp_path / "fd"; site = root / "site"
    shutil.copytree(REPRO / "site.example", site)
    with open(site / "config.env", "a") as fh:
        fh.write(f"\nSISTEER_STATE_DIR={root / 'state'}\nSISTEER_PYTHON={VENV_PY}\nSISTEER_REGISTRY_ROOT={root / 'registry'}\n")
    r = run_stage("03_hpo.sh", "--dry-run", "--object-class", "relu", "--phase", "all", "--model", MODEL, "--overrides", "reproduce/overrides/demo_smoke.env",
                  site=site, env={ST.RUN_OVERRIDES_ENV: ""}, timeout=600)
    out = r.stdout + r.stderr
    assert r.returncode == 0, out[-2500:]
    assert re.search(r"03_hpo: model=qwen35_4b_demo .* overrides=demo_smoke protocol=paper \(warmup 0\.05,0\.15, clip searched", out), out[-2500:]
    assert "mixed settings" not in out and "REFUSED" not in out
