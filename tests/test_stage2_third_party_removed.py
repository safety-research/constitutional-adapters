"""Third-party material no paper result uses is removed, not attributed.

The vendored Google IFEval verifier, the PersonaDrift scorers, AxBench, the
IFEval-at-length (longctx) suite, the IFEval-worded stage-2 system instructions
and the unlicensed HHH prompt were stage-2 material behind no figure or table.
This module pins that they are gone and that every path that used to reach them
now refuses by name instead of falling back to something else.
"""
from __future__ import annotations

import importlib
import importlib.util
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

REMOVED_MODULES = (
    "sisteer.eval.ifeval_official",
    "sisteer.eval.persona_drift",
    "sisteer.eval.persona_drift_scorers",
    "sisteer.eval.axbench",
    "sisteer.eval.longctx",
)
REMOVED_SUITES = ("ifeval_anchor", "drift", "axbench", "longctx")
REMOVED_STAGE2_RUNNERS = ("anchor", "drift", "axbench", "longctx")
REMOVED_PATHS = (
    "src/sisteer/eval/ifeval_official",
    "src/sisteer/eval/data/one_syllable_words.txt",
    "configs/stage2_drift.yaml",
    "configs/stage2_axbench_gemma9b.yaml",
    "configs/stage2_llama8b_rs.yaml",
    "configs/stage2_ladder.yaml",
    "configs/stage2_qwen_spot.yaml",
    "scripts/sbatch/stage2_drift.sbatch",
    "scripts/sbatch/stage2_axbench.sbatch",
    "scripts/sbatch/stage2_anchor.sbatch",
    "scripts/sbatch/stage2_longctx.sbatch",
    "scripts/sbatch/stage2_pairs.sbatch",
    "scripts/sbatch/stage2_pairs_rs.sbatch",
    "scripts/sbatch/stage2_ladder.sbatch",
    "scripts/sbatch/stage2_qwen_spot.sbatch",
    "scripts/sbatch/alphakl_overrefusal.sbatch",
    "tests/fixtures/capeval/evalplus",
    "tests/fixtures/chat_templates/glm46vflash@411bb4d7.jinja",
)
# The IFEval-type ids the stage-2 configs and sbatch arrays used to name
IFEVAL_CONFIG_IDS = ("json_format", "number_words", "response_language_fr",
                     "keyword_frequency", "end_with_phrase", "combined_json_fr")


@pytest.mark.parametrize("name", REMOVED_MODULES)
def test_removed_module_is_not_importable(name):
    assert importlib.util.find_spec(name) is None, f"{name} is still in the tree"


@pytest.mark.parametrize("rel", REMOVED_PATHS)
def test_removed_path_is_absent(rel):
    assert not (REPO / rel).exists(), f"{rel} is still in the tree"


def test_no_stage2_matrix_sbatch_remains():
    assert sorted(p.name for p in (REPO / "scripts" / "sbatch").glob("stage2_matrix*.sbatch")) == []


def test_removed_suites_refuse_by_name():
    from sisteer.eval import suites

    for name in REMOVED_SUITES:
        assert name not in suites.SUITES
        with pytest.raises(KeyError, match=name):
            suites.get_runner(name)


def test_vendored_checker_name_is_an_unknown_checker():
    from sisteer.eval import checkers

    with pytest.raises(KeyError):
        checkers.check("ifeval_official", "some text", inst_type="punctuation:no_comma", kwargs={})
    with pytest.raises(KeyError):
        checkers.check_loose("ifeval_official", "some text", inst_type="punctuation:no_comma", kwargs={})


def test_instruction_registry_carries_no_ifeval_material():
    from sisteer.data import instructions as I

    for attr in ("IFEVAL_SIS", "SI_ALIASES", "LADDER_CELLS", "AXBENCH_SI_TEMPLATE",
                 "persona_instruction", "concept_instruction", "add_to_registry"):
        assert not hasattr(I, attr), attr
    assert not [i.id for i in I.REGISTRY if i.checker == "ifeval_official" or i.category == "ifeval"]
    for cfg_id in IFEVAL_CONFIG_IDS:
        with pytest.raises(KeyError, match=cfg_id):
            I.resolve_si(cfg_id)
    # a kept id still resolves through the same entry point
    assert I.resolve_si_id("concise") == "concise"
    assert I.resolve_si("si_constitution").id == "si_constitution"


def test_hhh_prompt_and_its_si_are_gone():
    from sisteer.data import instructions as I
    from sisteer.data import safety_texts as T

    assert not hasattr(T, "HHH_PROMPT")
    assert "si_hhh" not in I.BY_ID
    with pytest.raises(KeyError, match="si_hhh"):
        I.resolve_si("si_hhh")
    seed = yaml.safe_load((REPO / "configs" / "registry" / "si_texts.yaml").read_text())
    ids = {r["si_id"] for section in seed.values() if isinstance(section, list)
           for r in section if isinstance(r, dict) and "si_id" in r}
    assert "si_hhh" not in ids and "si_caa" in ids


def test_realign_arms_name_no_removed_si():
    from sisteer.realign_arms import REALIGN_ARMS

    assert [a.name for a in REALIGN_ARMS.values() if a.si_id == "si_hhh"] == []
    assert {"prompted_constitution", "distill_constitution", "ft_lora_caa"} <= set(REALIGN_ARMS)


def test_no_arm_carries_a_removed_suite():
    from sisteer.methods import ARMS, arms_for_suite

    for suite in ("axbench", "drift", "longctx"):
        assert arms_for_suite(suite) == []
        assert not [n for n, spec in ARMS.items() if suite in spec.suites]


def test_stage2_driver_drops_the_removed_suites():
    import stage2_run

    for name in REMOVED_STAGE2_RUNNERS:
        assert name not in stage2_run.SUITE_RUNNERS
    assert {"pairs", "ifeval", "composition", "ladder"} <= set(stage2_run.SUITE_RUNNERS)


def _shell_commands(text: str) -> list[str]:
    """A shell file's commands: comment lines dropped, backslash continuations joined."""
    out, cur = [], ""
    for line in text.splitlines():
        if not cur and line.lstrip().startswith("#"):
            continue
        if line.rstrip().endswith("\\"):
            cur += line.rstrip()[:-1] + " "
            continue
        out.append(cur + line)
        cur = ""
    return out + ([cur] if cur else [])


def test_launchers_hand_stage2_run_only_registered_instructions():
    """stage2_run.py resolves --si after the model load, so a launcher naming a
    removed instruction would sync and load the model before it failed."""
    from sisteer.data import instructions as I

    seen, bad = [], []
    scripts = REPO / "scripts"
    for path in sorted([*(scripts / "sbatch").iterdir(), *scripts.glob("*.sh")]):
        if not path.is_file() or path.suffix not in (".sbatch", ".sh"):
            continue
        for cmd in _shell_commands(path.read_text()):
            if "stage2_run.py" not in cmd:
                continue
            for si in re.findall(r"--si[= ]+([^\s\"'$]\S*)", cmd):
                seen.append(path.name)
                try:
                    I.resolve_si(si)
                except KeyError:
                    bad.append(f"{path.name}: --si {si}")
    assert bad == []
    assert "torch_split_proof.sbatch" in seen   # the scan reads the launchers it is meant to


def test_judge_batch_has_no_axbench_mode():
    import judge_batch

    for attr in ("AxBenchRating", "run_axbench_suite", "main_axbench", "AXBENCH_CONFIG"):
        assert not hasattr(judge_batch, attr), attr


def test_fetch_script_provisions_no_removed_item():
    import fetch_stage2_data as F

    for attr in ("verify_axbench_runtime_read", "verify_alpaca_eval_runtime_read",
                 "ensure_wordlist", "REFT_R1_REPO"):
        assert not hasattr(F, attr), attr
    assert set(F.NLTK_PACKAGES) == {"punkt_tab", "punkt"}


def test_stage2_common_has_no_ifeval_arrays():
    sh = (REPO / "scripts" / "sbatch" / "stage2_common.sh").read_text()
    for token in ("S2_IFEVAL_SIS", "S2_RS_SIS", "SISTEER_WORDLIST_DIR"):
        assert token not in sh, token


def test_no_source_imports_a_removed_module():
    """A leftover import anywhere in the package or the scripts would fail only when its path runs."""
    pat = re.compile(r"(?:from|import)\s+(?:sisteer\.eval\.|\.)(?:ifeval_official|persona_drift(?:_scorers)?|axbench|longctx)\b"
                     r"|from\s+sisteer\.eval\s+import\s+[^\n]*\b(?:ifeval_official|persona_drift(?:_scorers)?|axbench|longctx)\b")
    hits = []
    for root in ("src", "scripts", "tests"):
        for path in (REPO / root).rglob("*.py"):
            if path.name == Path(__file__).name:
                continue
            for n, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
                if pat.search(line):
                    hits.append(f"{path.relative_to(REPO)}:{n}: {line.strip()}")
    assert hits == []


# capeval sweep manifests whose cells score removed instructions, each with the
# launcher that refuses it by name before any sync or model load
RETIRED_CAPEVAL_SWEEPS = {"s2_rediscovery.yaml": "capeval_s2.sbatch"}
CAPEVAL_MANIFESTS = REPO / "configs" / "capeval" / "manifests"


def _compliance_sis(path: Path) -> list[str]:
    """The instruction id of every sisteer_compliance primary in a manifest file."""
    sis = []
    for doc in yaml.safe_load_all(path.read_text()):
        for primary in ((doc or {}).get("target") or {}).get("primaries") or []:
            if primary.get("plugin") == "sisteer_compliance":
                sis.append((primary.get("params") or {}).get("si"))
    return sis


def _unresolved(sis: list[str]) -> list[str]:
    from sisteer.data import instructions as I

    bad = []
    for si in sis:
        try:
            I.resolve_si(si)
        except KeyError:
            bad.append(si)
    return bad


def test_capeval_sweeps_score_only_registered_instructions():
    """The sisteer_compliance target resolves its instruction only at scoring
    time, after the model load and Tier 0/1. A manifest naming an instruction
    that does not resolve is allowed only as a retired sweep whose launcher
    refuses up front."""
    unresolved = {path.name: _unresolved(_compliance_sis(path))
                  for path in sorted(CAPEVAL_MANIFESTS.glob("*.yaml"))}
    unresolved = {name: sis for name, sis in unresolved.items() if sis}
    assert set(unresolved) <= set(RETIRED_CAPEVAL_SWEEPS), unresolved


@pytest.mark.parametrize("manifest,launcher", sorted(RETIRED_CAPEVAL_SWEEPS.items()))
def test_retired_capeval_sweep_refuses_before_any_setup(manifest, launcher, tmp_path):
    proc = subprocess.run(
        ["bash", str(REPO / "scripts" / "sbatch" / launcher)], cwd=tmp_path,
        env={"PATH": os.environ["PATH"], "USER": "capevalxuser", "HOME": str(tmp_path)},
        capture_output=True, text=True, timeout=60)
    assert proc.returncode == 1, (proc.stdout, proc.stderr)
    assert proc.stdout == ""
    assert proc.stderr.startswith("FATAL: ") and "is retired" in proc.stderr, proc.stderr
    assert "No such file" not in proc.stderr and "command not found" not in proc.stderr
    path = CAPEVAL_MANIFESTS / manifest
    if path.is_file():
        # every cell is unscoreable, and the refusal names each instruction
        sis = _compliance_sis(path)
        assert sis and _unresolved(sis) == sis
        for si in set(sis):
            assert si in proc.stderr, si
