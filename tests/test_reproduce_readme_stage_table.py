"""reproduce/README.md's "What the stages compute" table, held to the stages it describes.

One row per stage script present (reproduce/NN_*.sh, in order), a method cell of one or two sentences and a compute cell that is a fragment,
and the facts the cells state read from where the stages read them:
  - every walltime knob a compute cell names (`SISTEER_TIME_*` with its value) is the site example's value and lib/common.sh's default, and
    02's memory-probe walltime and GPU are the ones 02 submits it with;
  - 03's paper protocol is the trial count and the two warm-up levels 03's paper branch emits, at the paper's rank and module set; its
    architecture search is the serving probe's default menu (scripts/sbatch/serving_probe.sbatch's ranks over lora_module_sets.MODULE_SETS,
    the grid 03 hands the probe under that protocol), each module set named the same way in the row and in the fit-your-own pick list, the
    probe's width the carrier's header;
  - 04's recipe numbers are the fits configs' sequence length, the paper's effective batch and the demo row's refusal coefficient;
  - the attacker-pool ratio 06's row and Prerequisites state is the planner's, and the multi-turn GPU counts Prerequisites and 06's section
    give are the plan's, at full scale and under the smoke recipe, on one cluster and on two."""
from __future__ import annotations

import collections
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from front_door_testlib import REPO, REPRO, VENV_PY  # noqa: E402

README = REPRO / "README.md"
MODEL = "qwen35_4b_demo"
#: each LoRA module set (lora_module_sets.MODULE_SETS id) as the page names it for a reader of the paper
PAPER_NAMES = {"wo": "W_O", "qkvo": "W_QKVO", "down": "the MLP's down projection", "mlp": "the whole MLP (gate, up and down)",
               "attn_mlp": "all seven projections"}
WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five"}
SENTENCE_END = re.compile(r"[.!?](?=\s+[A-Z`(]|$)")


def _section(heading: str) -> str:
    text = README.read_text(encoding="utf-8")
    start = f"\n## {heading}\n"
    assert start in text, f"the README's {heading!r} heading"
    i = text.index(start)
    j = text.find("\n## ", i + len(start))
    return text[i:j if j > 0 else len(text)]


def _rows() -> list[tuple[str, str, str]]:
    rows = []
    for line in _section("What the stages compute").splitlines():
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if cells[0] == "stage" or set(cells[0]) <= {"-"}:
            continue
        assert len(cells) == 3, f"a row of three cells: {line!r}"
        rows.append((cells[0], cells[1], cells[2]))
    return rows


def _row(stage: str) -> tuple[str, str, str]:
    [row] = [r for r in _rows() if r[0] == stage]
    return row


def _hours(spec: str) -> float:
    """A Slurm walltime, `[D-]HH:MM:SS`, in hours."""
    days, _, hms = spec.rpartition("-")
    h, m, s = (int(x) for x in hms.split(":"))
    return (int(days) if days else 0) * 24 + h + m / 60 + s / 3600


def _module_sets() -> dict[str, tuple[str, ...]]:
    spec = importlib.util.spec_from_file_location("_readme_table_lora_module_sets", REPO / "src/sisteer/extraction/lora_module_sets.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return dict(mod.MODULE_SETS)


def test_the_table_has_one_row_per_stage_script_in_order():
    scripts = sorted(p.name[:2] for p in REPRO.glob("[0-9][0-9]_*.sh"))
    assert scripts, "the stage scripts"
    assert [r[0] for r in _rows()] == scripts, ([r[0] for r in _rows()], scripts)


def test_each_method_cell_is_one_or_two_sentences_and_each_compute_cell_a_fragment():
    for stage, method, compute in _rows():
        n = len(SENTENCE_END.findall(method))
        assert 1 <= n <= 2, (stage, n, method)
        assert compute and not compute.rstrip().endswith((".", ";")), (stage, compute)


def test_every_walltime_knob_a_compute_cell_names_is_the_sites_default():
    cfg = dict(re.findall(r"^(SISTEER_TIME_[A-Z]+)=(\S+)", (REPRO / "site.example" / "config.env").read_text(encoding="utf-8"), flags=re.M))
    common = dict(re.findall(r'(SISTEER_TIME_[A-Z]+)="\$\{\1:-([^}]+)\}"', (REPRO / "lib" / "common.sh").read_text(encoding="utf-8")))
    seen = []
    for stage, _method, compute in _rows():
        for knob, value, unit in re.findall(r"`(SISTEER_TIME_[A-Z]+)` \((\d+(?:\.\d+)?) (h|days?)\)", compute):
            stated = float(value) * (24 if unit.startswith("day") else 1)
            assert knob in cfg and _hours(cfg[knob]) == stated, (stage, knob, value, unit, cfg.get(knob))
            assert knob in common and _hours(common[knob]) == stated, (stage, knob, common.get(knob))
            seen.append(knob)
    assert {"SISTEER_TIME_HPO", "SISTEER_TIME_PROBE", "SISTEER_TIME_FIT", "SISTEER_TIME_EVAL", "SISTEER_TIME_PETRI"} <= set(seen), seen


def test_02s_memory_probe_is_one_gpu_at_the_walltime_02_submits_it_at():
    compute = _row("02")[2]
    m = re.search(r"one GPU per memory probe \((\d+(?:\.\d+)?) h\)", compute)
    assert m, compute
    two = (REPRO / "02_prepare_model.sh").read_text(encoding="utf-8")
    sub = re.search(r"--script scripts/sbatch/memprobe_fit\.sbatch.*?--time (\S+)", two, flags=re.S)
    assert sub and _hours(sub.group(1)) == float(m.group(1)), (sub and sub.group(1), m.group(1))
    assert re.search(r"^#SBATCH --gres=gpu:1$", (REPO / "scripts/sbatch/memprobe_fit.sbatch").read_text(encoding="utf-8"), flags=re.M)


def test_03s_row_states_the_paper_protocol_03_emits_and_the_architecture_searchs_actual_grid():
    method, compute = " ".join(_row("03")[1].split()), _row("03")[2]
    sets = _module_sets()
    assert set(PAPER_NAMES) == set(sets), (sorted(PAPER_NAMES), sorted(sets))
    # each paper name says what its set holds
    attn, mlp = {"q_proj", "k_proj", "v_proj", "o_proj"}, {"gate_proj", "up_proj", "down_proj"}
    assert sets["wo"] == ("o_proj",) and set(sets["qkvo"]) == attn and sets["down"] == ("down_proj",) and set(sets["mlp"]) == mlp
    assert set(sets["attn_mlp"]) == attn | mlp and len(sets["attn_mlp"]) == 7 and PAPER_NAMES["attn_mlp"].startswith("all seven")
    # the architecture search's grid is the carrier's default menu (03 hands the probe empty positionals 4/5 under that protocol)
    carrier = (REPO / "scripts/sbatch/serving_probe.sbatch").read_text(encoding="utf-8")
    default_sets = re.search(r'^SETS="\$\{4:-([^}]+)\}"', carrier, flags=re.M).group(1).split(",")
    ranks = [int(r) for r in re.search(r'^RANKS="\$\{5:-([^}]+)\}"', carrier, flags=re.M).group(1).split(",")]
    assert set(default_sets) == set(sets), default_sets
    spelled = ", ".join(str(r) for r in ranks[:-1]) + f" and {ranks[-1]}"
    assert f"ranks {spelled}" in method, (spelled, method)
    for sid, name in PAPER_NAMES.items():
        assert name in method, (sid, name)
    width = int(re.search(r"^#SBATCH --gres=gpu:(\d+)$", carrier, flags=re.M).group(1))
    assert f"one GPU for one pair, {width} for the search's {len(default_sets) * len(ranks)}" in compute, compute
    # the paper's protocol: the trial count and the two warm-up levels 03's paper branch emits, at the paper's rank and module set
    hpo = (REPRO / "03_hpo.sh").read_text(encoding="utf-8")
    levels = re.search(r'emit-trials --design legacy --out "\$R1".*?--warmup-levels ([\d.,]+) --n "\$N_TRIALS"', hpo, flags=re.S).group(1)
    lo, hi = levels.split(",")
    n = re.search(r'^N_TRIALS="\$\{OVERRIDE_HPO_TRIALS:-(\d+)\}"', hpo, flags=re.M).group(1)
    paper = (REPRO / "overrides" / "paper_defaults.env").read_text(encoding="utf-8")
    assert re.search(rf"^OVERRIDE_HPO_TRIALS={n}$", paper, flags=re.M) and re.search(r"^OVERRIDE_HPO_WARMUP_FRAC=''", paper, flags=re.M)
    assert f"over {n} Sobol trials split between warm-up {lo} and {hi}" in method, method
    assert re.search(r"^OVERRIDE_LORA_RANK=4\b", paper, flags=re.M) and re.search(r"^OVERRIDE_LORA_MODULE_SET=wo\b", paper, flags=re.M)
    assert f"rank 4 on {PAPER_NAMES['wo']} (`{sets['wo'][0]}`)" in method, method


def test_the_fit_your_own_pick_names_the_same_ranks_and_module_sets():
    fit = _section("Fitting your own objects")
    carrier = (REPO / "scripts/sbatch/serving_probe.sbatch").read_text(encoding="utf-8")
    ranks = re.search(r'^RANKS="\$\{5:-([^}]+)\}"', carrier, flags=re.M).group(1).split(",")
    assert f"pick a rank ({', '.join(ranks[:-1])} or {ranks[-1]})" in " ".join(fit.split()), "the pick's ranks"
    for sid, name in PAPER_NAMES.items():
        assert re.search(rf"^  - `{sid}`: .*{re.escape(name)}", fit, flags=re.M), (sid, name)


def test_04s_recipe_numbers_are_the_fits_configs_the_papers_batch_and_the_demo_rows_coefficient():
    method = " ".join(_row("04")[1].split())
    m = re.search(r"\(([\d,]+)-token sequences, effective batch (\d+)\)", method)
    assert m, method
    seq, batch = int(m.group(1).replace(",", "")), int(m.group(2))
    configs = sorted(REPO.glob("configs/sdfbase_*_fits.yaml"))
    assert configs
    for cfg in configs:
        assert re.search(rf"^max_len: {seq}$", cfg.read_text(encoding="utf-8"), flags=re.M), cfg.name
    paper = (REPRO / "overrides" / "paper_defaults.env").read_text(encoding="utf-8")
    assert re.search(rf"^OVERRIDE_EFFECTIVE_BATCH={batch}$", paper, flags=re.M)
    coef = re.search(r"coefficient ([\d.]+)\)", method).group(1)
    row = re.search(rf"^  {MODEL}:\n(.*?)(?=^  \S)", (REPRO / "models.yaml").read_text(encoding="utf-8"), flags=re.S | re.M).group(1)
    assert f"cell_id: ar_c{coef.replace('.', 'p')}__arditi_allpos" in row, coef


def _plan(clusters: str, overrides: str) -> collections.Counter:
    r = subprocess.run([VENV_PY, str(REPRO / "lib" / "stage.py"), "plan", "--model", MODEL, "--view", "battery", "--families", "multiturn",
                        "--nshards", "9", "--sha", "0" * 40, "--clusters", clusters, "--models-table", str(REPRO / "models.yaml"),
                        "--overrides", str(REPRO / "overrides" / overrides), "--json"], cwd=str(REPO), capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-1500:]
    return collections.Counter((row["cluster"], row.get("kind") == "attacker") for row in json.loads(r.stdout))


def test_the_attacker_pool_ratio_and_the_multiturn_gpu_counts_are_the_planners():
    ratio = int(re.search(r"^MT_TARGETS_PER_ATTACKER = (\d+)$", (REPRO / "lib" / "stage.py").read_text(encoding="utf-8"), flags=re.M).group(1))
    assert f"an attacker pool of one GPU per {WORDS[ratio]} multi-turn targets" in _row("06")[2]
    pre = " ".join(_section("Prerequisites").split())
    assert f"an attacker pool of one GPU per {WORDS[ratio]} targets" in pre, pre[:600]
    m = re.search(r"at full scale (\d+) targets and (\d+) attacker GPUs on one cluster, or (\d+) and (\d+) on each of two", pre)
    assert m, pre[:600]
    for clusters, (targets, attackers) in (("cluster_a", m.group(1, 2)), ("cluster_a,cluster_b", m.group(3, 4))):
        per = _plan(clusters, "paper_defaults.env")
        for lab in clusters.split(","):
            assert (per[(lab, False)], per[(lab, True)]) == (int(targets), int(attackers)), (clusters, lab, per)
    # the smoke: one target per multi-turn attack in all, and one attacker GPU on each cluster that carries any
    six = " ".join(_section("The worked example, stage by stage").split())
    sm = re.search(r"The smoke's (\w+) multi-turn targets, one per attack, share an attacker pool of one GPU on each cluster", six)
    assert sm, "06's section states the smoke's multi-turn GPUs"
    for clusters in ("cluster_a", "cluster_a,cluster_b"):
        per = _plan(clusters, "demo_smoke.env")
        assert WORDS[sum(v for (lab, atk), v in per.items() if not atk)] == sm.group(1), (clusters, per)
        assert all(per[(lab, True)] == 1 for lab in clusters.split(",") if per[(lab, False)]), (clusters, per)
