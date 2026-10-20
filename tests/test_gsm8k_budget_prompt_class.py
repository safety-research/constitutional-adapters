"""The GSM8K prompt budget is made for the ROW'S PROMPT CLASS, and the job holds the sidecar to it.

A prompted row (`prompted__<si>`) sends its SI as the system turn of every request, so its budget must fold that SI into the rendered
prompts (`--system-prompt-cell`); a budget of bare prompts is thousands of tokens short for the constitution SI and its G-FIT is false.
Executed against the REAL producer on the worked-example model's PINNED tokenizer (Qwen/Qwen3.5-4B at the campaign's revision, from the
local HF cache, offline — skipped with the reason when it is not there) and the REAL 200 GSM8K prompts (the two public datasets from the
local datasets cache, offline — skipped when absent): the prompted budget is the bare budget plus the SI; both sidecars record their prompt
class; and the JOB's own sidecar check (the python heredoc of g8k_common.sh `g8k_budget_artifact`, executed verbatim) admits each sidecar
for its own row and refuses it for the other. Also: the producer admits an ONBOARDED model key (geometry.ONBOARDED_MODEL_HF_IDS beside
MODEL_HF_IDS — it read MODEL_HF_IDS alone, the same class as the standups' refusal at a front-door wave's first unit), shown offline
through its by-name refusal of a config that serves another checkpoint."""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
PRODUCER = REPO / "scripts" / "tinycap_oos_prompt_budget.py"
COMMON = REPO / "scripts" / "sdfbase" / "gsm8k" / "g8k_common.sh"
CONFIG = "configs/sdfbase_full22_qwen35_4b.yaml"                                 # the campaign's wave config pins the served checkpoint
KEY, HF, REV = "qwen35_4b", "Qwen/Qwen3.5-4B", "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
ONBOARDED_KEY, ONBOARDED_HF = "qwen35_4b_demo", "Qwen/Qwen3.5-4B"              # the front door's onboarded model (reproduce/models.yaml serve.checkpoint)
PROMPTED = "prompted__si_constitution"
SHA = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()   # the pin IS this checkout's HEAD
EVAL, CAP, ARM = "capability:gen_long_thinking", "8192", "on"


def _hf_hub() -> Path:
    return Path(os.environ.get("HF_HUB_CACHE") or (Path(os.environ.get("HF_HOME") or Path.home() / ".cache" / "huggingface") / "hub"))


def _skip_unless_cached():
    snap = _hf_hub() / f"models--{HF.replace('/', '--')}" / "snapshots" / REV / "tokenizer_config.json"
    if not snap.is_file():
        pytest.skip(f"{HF}@{REV[:12]} tokenizer not in the local HF cache (offline; never a download here): {snap}")
    ds = Path(os.environ.get("HF_DATASETS_CACHE") or Path.home() / ".cache" / "huggingface" / "datasets")
    if not (ds / "openai___gsm8k").is_dir() or not (ds / "madrylab___gsm8k-platinum").is_dir():
        pytest.skip("openai/gsm8k + gsm8k-platinum not in the local datasets cache (offline)")


def _client_python() -> str:
    """The interpreter the job runs the producer under is the CLIENT venv (configs/capeval/tinycap-client-requirements.txt: inspect_ai,
    inspect_evals, transformers, datasets); here: $TINYCAP_CLIENT_PYTHON, else a local `~/venvs/tinycap-client-*` venv, else this
    interpreter when it imports the same — else the test is skipped by name (never an install here)."""
    probe = "import inspect_evals, inspect_ai, yaml, transformers, datasets"
    cands = [os.environ.get("TINYCAP_CLIENT_PYTHON") or ""] + sorted(str(p) for p in (Path.home() / "venvs").glob("tinycap-client-*/bin/python")) + [sys.executable]
    for py in cands:
        if py and Path(py).is_file() and subprocess.run([py, "-c", probe], capture_output=True, timeout=300).returncode == 0:
            return py
    pytest.skip("no interpreter with inspect_evals + inspect_ai + transformers + datasets (the client venv): set TINYCAP_CLIENT_PYTHON")


def _produce(tmp: Path, cell: str | None) -> tuple[dict, dict]:
    out = tmp / (cell or "none") / "oos_prompt_budget.json"
    argv = [_client_python(), str(PRODUCER), "--task", "gsm8k_paired200", "--arm", ARM, "--cap", CAP, "--geometry-eval", EVAL,
            "--config", CONFIG, "--model-key", KEY, "--tree-sha", SHA, "--offline", "--out", str(out)]
    if cell:
        argv += ["--system-prompt-cell", cell]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1",
               PYTHONPATH=f"{REPO / 'src'}:{REPO / 'scripts'}", TOKENIZERS_PARALLELISM="false")
    r = subprocess.run(argv, cwd=str(REPO), env=env, capture_output=True, text=True, timeout=900)
    # exit 0 = fits; exit 2 = the G-FIT verdict is `fits: false` — the artifact AND the sidecar are still written (the sidecar records
    # the verdict either way; the job's check is what refuses the row). Anything else is a producer failure.
    assert r.returncode in (0, 2), (cell, r.returncode, r.stdout[-1500:], r.stderr[-2500:])
    side = json.loads((out.parent / (out.name + ".fit.json")).read_text())
    assert (r.returncode == 0) == (side["fits"] is True), (cell, r.returncode, side)
    return json.loads(out.read_text()), side


def _job_check(sidecar: Path, cell: str) -> tuple[int, str]:
    """The job's sidecar check, executed verbatim: the python heredoc of g8k_budget_artifact with the job's seven arguments."""
    body = re.search(r"^g8k_budget_artifact\(\)\{.*?^\}", COMMON.read_text(), re.M | re.S).group(0)
    py = body.split("<<'PY'", 1)[1].split("\n", 1)[1].split("\nPY\n", 1)[0]      # after the shell line's remainder, up to the terminator
    env = dict(os.environ, PYTHONPATH=f"{REPO / 'src'}:{REPO / 'scripts'}")   # the check resolves the SI text from THIS checkout (sisteer.eval.jailbreak) and the currency predicate (tinycap_oos_prompt_budget)
    r = subprocess.run([sys.executable, "-", str(sidecar), CAP, EVAL, SHA, ARM, cell, KEY], input=py, capture_output=True, text=True, timeout=300, cwd=str(REPO), env=env)   # the job's SEVEN arguments (the model key since the currency predicate)
    return r.returncode, r.stdout + r.stderr


def test_a_tree_sha_that_is_not_this_checkouts_head_is_refused_before_anything_is_rendered(tmp_path):
    """`--tree-sha` is the sidecar's pin: the producer refuses a value that is not its own checkout's HEAD (it used to record the
    caller's value unchecked)."""
    out = tmp_path / "x" / "oos_prompt_budget.json"
    argv = [sys.executable, str(PRODUCER), "--task", "gsm8k_paired200", "--arm", ARM, "--cap", CAP, "--geometry-eval", EVAL, "--config", CONFIG,
            "--model-key", KEY, "--tree-sha", "c" * 40, "--offline", "--out", str(out)]
    r = subprocess.run(argv, cwd=str(REPO), env=dict(os.environ, PYTHONPATH=f"{REPO / 'src'}:{REPO / 'scripts'}"), capture_output=True, text=True, timeout=300)
    assert r.returncode != 0 and "names a tree this checkout is not at" in r.stderr and f"HEAD {SHA}" in r.stderr, r.stderr[-800:]
    assert not out.exists()


def test_the_producer_admits_an_onboarded_model_key_through_the_merged_hf_id_view(tmp_path):
    """The producer resolves `--model-key` through geometry.MODEL_HF_IDS AND geometry.ONBOARDED_MODEL_HF_IDS (it read the first alone, and a
    front-door wave's first unit died on it): an onboarded key is KNOWN — a config serving another checkpoint is refused BY NAME, naming the
    key's pinned checkpoint, never as an unknown key. Offline, and before any tokenizer load."""
    cfg = tmp_path / "other.yaml"; cfg.write_text(f"model: org/another-checkpoint\nmodel_revision: {'d' * 40}\n")
    out = tmp_path / "x" / "oos_prompt_budget.json"
    argv = [sys.executable, str(PRODUCER), "--task", "gsm8k_paired200", "--arm", ARM, "--cap", CAP, "--geometry-eval", EVAL, "--config", str(cfg),
            "--model-key", ONBOARDED_KEY, "--tree-sha", SHA, "--offline", "--out", str(out)]
    r = subprocess.run(argv, cwd=str(REPO), env=dict(os.environ, PYTHONPATH=f"{REPO / 'src'}:{REPO / 'scripts'}", HF_HUB_OFFLINE="1"),
                       capture_output=True, text=True, timeout=300)
    assert r.returncode != 0 and f"--model-key '{ONBOARDED_KEY}' names '{ONBOARDED_HF}' but the config serves 'org/another-checkpoint'" in r.stderr, r.stderr[-800:]
    assert f"'{ONBOARDED_KEY}'" in r.stderr.split("known keys:", 1)[1] and f"'{KEY}'" in r.stderr.split("known keys:", 1)[1] and not out.exists()


@pytest.fixture(scope="module")
def budgets(tmp_path_factory):
    _skip_unless_cached()
    tmp = tmp_path_factory.mktemp("g8k_budget")
    bare_doc, bare_side = _produce(tmp, None)
    prom_doc, prom_side = _produce(tmp, PROMPTED)
    return tmp, bare_doc, bare_side, prom_doc, prom_side


def test_the_prompted_budget_folds_the_si_in_and_the_sidecars_record_their_prompt_class(budgets):
    tmp, bare_doc, bare_side, prom_doc, prom_side = budgets
    assert bare_doc["n"] == prom_doc["n"] == 200 and bare_doc["chat_template_hash"] == prom_doc["chat_template_hash"]
    # the constitution SI is thousands of tokens: every prompted prompt is longer by (about) the same amount
    assert prom_doc["max"] - bare_doc["max"] > 1000 and prom_doc["p99"] - bare_doc["p99"] > 1000, (bare_doc["max"], prom_doc["max"])
    assert bare_side["system_prompt_cell"] is None and bare_side["system_prompt_sha16"] is None
    assert prom_side["system_prompt_cell"] == PROMPTED and re.fullmatch(r"[0-9a-f]{16}", prom_side["system_prompt_sha16"])
    for side in (bare_side, prom_side):
        assert side["tree_sha"] == SHA and side["arm"] == ARM and side["cap"] == int(CAP) and side["geometry_eval"] == EVAL
        assert side["model"] == HF and side["revision"] == REV


def test_the_jobs_sidecar_check_admits_each_budget_for_its_own_row_and_refuses_the_other(budgets):
    tmp, *_ = budgets
    bare = tmp / "none" / "oos_prompt_budget.json.fit.json"; prom = tmp / PROMPTED / "oos_prompt_budget.json.fit.json"
    prom_side = json.loads(prom.read_text())
    rc, out = _job_check(bare, "none");            assert rc == 0 and "prompt class none" in out, out
    rc, out = _job_check(prom, PROMPTED)
    if prom_side["fits"] is True:
        assert rc == 0 and f"prompt class {PROMPTED}" in out, out
    else:   # the SI-folded budget does not fit the row's window at this cap: the job refuses the prompted row by name — the true verdict a bare budget hid
        assert rc == 1 and "fits=False" in out and "system_prompt_cell" not in out, (out, prom_side)
    rc, out = _job_check(bare, PROMPTED);          assert rc == 1 and "system_prompt_cell None" in out and PROMPTED in out, out   # bare budget, prompted row: refused
    rc, out = _job_check(prom, "none");            assert rc == 1 and "system_prompt_cell 'prompted__si_constitution'" in out, out
    rc, out = _job_check(bare, "constv2-mix-full__lora_r4_ce"); assert rc == 0, out                                              # an object cell renders like `none`
    # a SAME-CLASS sidecar made from other SI bytes: refused by name (the sha16 is recomputed from this checkout's SI text, never trusted)
    wrong = json.loads(prom.read_text()); wrong["system_prompt_sha16"] = "0" * 16; wrong["fits"] = True
    wrong_path = tmp / "wrong_hash.fit.json"; wrong_path.write_text(json.dumps(wrong))
    rc, out = _job_check(wrong_path, PROMPTED);    assert rc == 1 and "system_prompt_sha16 '0000000000000000' != the pinned checkout's SI text" in out and "21e06f482a6559b3" in out, out
