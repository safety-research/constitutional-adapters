"""Stage-2 driver (scripts/stage2_run.py + scripts/stage2_frontier.py):
SI-id resolution, sbatch/config enumeration consistency,
driver plumbing helpers, the frontier's within-budget selection (--dry
preview, missing-MMLU not eligible), canary namespace quarantine,
resume-provenance manifests, the P1 pairs-race guards,
the ladder/composition driver seams, and the infra fixes pinned into
stage2_common.sh (no wandb sync on the exit trap, nltk data on shared storage, escalation --time carry,
the escalate-first bounded drain, the frozen-or-fail env init, and
the TERM-escalation errexit fixes — the handler must survive a caller's
`set -euo pipefail`, so its tests run under both set modes).
CPU-only; end-to-end paths run on the tiny-llama fixture (the frontier
runs --dry)."""

import json
import os
import re
import shutil
import shlex
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import stage2_frontier  # noqa: E402
import stage2_run  # noqa: E402
from sisteer.data.instructions import (  # noqa: E402
    BY_ID,
    resolve_si,
    resolve_si_id,
)


# ---------------------------------------------------------------- SI ids


def test_core_config_lists_no_removed_si():
    """The IFEval-type SIs the core matrix ran were removed with the vendored
    verifier; the config names none, and every id it does name resolves."""
    cfg = yaml.safe_load((REPO / "configs" / "stage2_llama8b.yaml").read_text())
    assert cfg["instructions"] == []
    for cfg_id in cfg["instructions"]:
        assert resolve_si(cfg_id).id in BY_ID


def test_config_ids_resolve_to_themselves_with_no_alias_layer():
    # with the IFEval aliases gone, "lowercase" is the stage-1 entry itself
    assert resolve_si("lowercase") is BY_ID["lowercase"]
    assert resolve_si("lowercase").checker == "lowercase"
    assert resolve_si_id("json") == "json"


def test_unknown_and_removed_ids_fail_loudly():
    for cfg_id in ("not_a_real_si", "json_format", "number_words", "ifeval_json"):
        with pytest.raises(KeyError, match="unknown instruction id"):
            resolve_si_id(cfg_id)


def test_sbatch_enumerations_match_config_order():
    """stage2_common.sh's S2_ARMS drives array-index -> arm decoding, so its
    ORDER must equal the config's arms list; the removed IFEval-type SI
    arrays are gone."""
    sh = (REPO / "scripts" / "sbatch" / "stage2_common.sh").read_text()

    def bash_array(name: str) -> list[str]:
        m = re.search(rf"{name}=\(([^)]*)\)", sh)
        assert m, f"{name} not found in stage2_common.sh"
        return m.group(1).split()

    cfg = yaml.safe_load((REPO / "configs" / "stage2_llama8b.yaml").read_text())
    assert bash_array("S2_ARMS") == cfg["arms"]
    assert "S2_IFEVAL_SIS" not in sh and "S2_RS_SIS" not in sh


# ---------------------------------------------------------------- driver plumbing


def test_suite_runners_cover_the_sbatch_contract():
    # "lifbench" stays as a key on purpose: it is the removal tombstone, so
    # a stale --suite lifbench errors with the removal reason instead of
    # argparse's generic invalid-choice (tested below).
    assert set(stage2_run.SUITE_RUNNERS) == {
        "pairs", "ifeval", "composition", "lifbench", "ladder",
    }
    # the removed third-party suites are refused by argparse's own choices
    for removed in ("anchor", "drift", "axbench", "longctx"):
        assert removed not in stage2_run.SUITE_RUNNERS


def test_lifbench_suite_removed_fails_loudly():
    """lifbench was removed from the campaign. A stale dispatch must
    raise the removal error BEFORE any config/model work — never silently
    no-op, never run."""
    with pytest.raises(SystemExit, match="D30"):
        stage2_run.SUITE_RUNNERS["lifbench"]({}, SimpleNamespace())


def test_train_kwargs_merge_default_then_arm():
    cfg = {"train_kwargs": {"default": {"epochs": 6, "batch_size": 2},
                            "bidpo": {"epochs": 20, "batch_size": 4}}}
    assert stage2_run._train_kwargs(cfg, "vector") == {"epochs": 6,
                                                       "batch_size": 2}
    assert stage2_run._train_kwargs(cfg, "bidpo") == {"epochs": 20,
                                                      "batch_size": 4}
    assert stage2_run._train_kwargs({}, "vector") == {}


def test_object_path_layout():
    cell = Path("/tmp/cell")
    assert stage2_run._object_path(cell, "vector").name == "vector.safetensors"
    assert stage2_run._object_path(cell, "bidpo").name == "vector.safetensors"
    assert stage2_run._object_path(cell, "psr").name == "object.pt"


def test_qwen_spot_empty_arms_fails_loudly(tmp_path):
    """An `arms: []` placeholder errors instantly, BEFORE any
    model load (run_ifeval validates first)."""
    cfg = {"model": "no-such-model-must-never-load",
           "instructions": ["json"], "arms": [],
           "out_dir": str(tmp_path)}

    class Args:
        all_cells = True
        si = None
        arm = None

    with pytest.raises(SystemExit, match="EMPTY"):
        stage2_run.run_ifeval(cfg, Args())


def test_composition_behaviors_unified_with_data_registry():
    """The composition eval's behavior library is DERIVED from
    data/instructions.BEHAVIORS — same SI text, same checkers — so the
    eval can never score a different instruction than the objects were
    trained on."""
    from sisteer.data.instructions import BEHAVIORS as DATA_BEHAVIORS
    from sisteer.eval.composition import BEHAVIORS as EVAL_BEHAVIORS

    assert set(EVAL_BEHAVIORS) == set(DATA_BEHAVIORS)
    for name, beh in EVAL_BEHAVIORS.items():
        si = DATA_BEHAVIORS[name]
        assert beh.instruction == si.text
        assert beh.checker == si.checker
        assert beh.checker_kwargs == si.checker_kwargs
        assert beh.clause  # joined-instruction clause still owned locally


# ---------------------------------------------------------------- frontier


def _fake_cell(root: Path, instr: str, method: str, *, steered: list[dict],
               base_mmlu: float = 0.6, cap_scales: dict | None = None,
               base_gsm8k: float | None = 0.5,
               gsm8k_scales: dict | None = None):
    """gsm8k defaults to a PASSING panel at every steered scale so tests
    about the MMLU/distinct3 mechanics keep their meaning under the
    GSM8K budget rule; pass gsm8k_scales (or base_gsm8k=None) to exercise
    the GSM8K gate itself."""
    cell = root / instr / method
    cell.mkdir(parents=True)
    arms = [
        {"arm": "baseline", "scale": 0.0, "compliance_strict": 0.1,
         "compliance_loose": 0.1},
        {"arm": "prompted", "scale": 0.0, "compliance_strict": 0.9,
         "compliance_loose": 0.9},
    ] + steered
    records = []
    for a in arms:
        if a["arm"] == "steered" and a.get("degenerate"):
            # repeated 3-grams -> distinct3 far below the 0.35 budget floor
            responses = ["la la la la la la la la"] * 6
        else:
            responses = [f"varied response number {i} with fresh words {i}"
                         for i in range(6)]
        records.append([
            {"prompt": f"p{i}", "response": r, "strict": 1, "loose": 1}
            for i, r in enumerate(responses)
        ])
    cap = {"unsteered": {"mmlu": base_mmlu}}
    for scale, mmlu in (cap_scales or {}).items():
        cap[f"scale_{scale}"] = {"mmlu": mmlu}
    ext: dict = {}
    if base_gsm8k is not None:
        ext["0.0"] = {"gsm8k": base_gsm8k}
        if gsm8k_scales is None:  # default: gsm8k passes everywhere
            gsm8k_scales = {a["scale"]: base_gsm8k for a in steered}
        for scale, acc in gsm8k_scales.items():
            ext.setdefault(str(float(scale)), {})["gsm8k"] = acc
    (cell / "results.json").write_text(json.dumps({
        "instruction": instr, "method": method, "arms": arms,
        "capability": cap, "capability_ext": ext, "records": records,
    }))


def test_frontier_within_budget_selection(tmp_path, monkeypatch, capsys):
    out = tmp_path / "ifeval"
    # vector cell: best strict 0.8@4.0 but 4.0 is over the MMLU budget;
    # 0.75@2.0 is within 0.9x of best AND in budget -> the frontier pick.
    _fake_cell(
        out, "json_format", "vector",
        steered=[
            {"arm": "steered", "scale": 2.0, "compliance_strict": 0.75,
             "compliance_loose": 0.8},
            {"arm": "steered", "scale": 4.0, "compliance_strict": 0.8,
             "compliance_loose": 0.85},
        ],
        cap_scales={2.0: 0.59, 4.0: 0.5},  # 4.0 drops MMLU > 2pt
    )
    # psr cell: best candidate is degenerate (distinct3 floor) -> no frontier
    _fake_cell(
        out, "json_format", "psr",
        steered=[{"arm": "steered", "scale": 1.0, "compliance_strict": 0.7,
                  "compliance_loose": 0.7, "degenerate": True}],
        cap_scales={1.0: 0.6},
    )
    monkeypatch.setattr(sys, "argv", [
        "stage2_frontier.py", "--out-dir", str(out), "--dry",
    ])
    stage2_frontier.main()
    # --dry writes the PREVIEW file only
    frontier = json.loads((out / "frontier.preview.json").read_text())
    assert not (out / "frontier.json").exists()
    vec = frontier["json_format"]["vector"]
    assert vec["frontier"]["scale"] == 2.0
    assert vec["frontier"]["compliance"] == pytest.approx(0.75)
    by_scale = {c["scale"]: c for c in vec["candidates"]}
    assert by_scale[4.0]["in_budget"] is False
    assert by_scale[2.0]["in_budget"] is True
    assert frontier["json_format"]["psr"]["frontier"] is None
    assert "json_format" in capsys.readouterr().out


def test_budget_verdict_d33_rule():
    """GSM8K joins the budget. Definitive failures are False even
    when another metric is unmeasured; required-but-unmeasured -> None;
    legacy drops the GSM8K requirement entirely."""
    v = stage2_frontier.budget_verdict
    ok = dict(rep=0.9, mmlu=0.60, base_mmlu=0.60, gsm8k=0.50, base_gsm8k=0.50)
    assert v(**ok) is True
    assert v(**{**ok, "gsm8k": 0.44}) is False          # drop 6pt > 5pt bound
    assert v(**{**ok, "gsm8k": 0.46}) is True           # drop 4pt <= bound
    assert v(**{**ok, "gsm8k": None}) is None           # required, unmeasured
    assert v(**{**ok, "base_gsm8k": None}) is None
    assert v(**{**ok, "gsm8k": None, "mmlu": 0.50}) is False  # MMLU fail wins
    assert v(**{**ok, "mmlu": None, "gsm8k": 0.30}) is False  # GSM8K fail wins
    assert v(**{**ok, "rep": 0.1, "gsm8k": None}) is False    # rep fail wins
    assert v(**{**ok, "gsm8k": None}, legacy=True) is True    # stage-1 map
    assert v(**{**ok, "gsm8k": 0.30}, legacy=True) is True    # gsm8k ignored
    assert v(**{**ok, "gsm8k": 0.44}, gsm8k_drop=0.02) is False
    assert v(**{**ok, "gsm8k": 0.49}, gsm8k_drop=0.02) is True


def test_frontier_gsm8k_gate_cache_and_fallback(tmp_path, monkeypatch):
    """The frontier falls back past a GSM8K-failing scale; an unmeasured
    scale is not-eligible (None); a cached fill-in (write-through sidecar)
    is honored without a model."""
    out = tmp_path / "ifeval"
    _fake_cell(
        out, "json_format", "vector",
        steered=[
            {"arm": "steered", "scale": 2.0, "compliance_strict": 0.75,
             "compliance_loose": 0.8},
            {"arm": "steered", "scale": 4.0, "compliance_strict": 0.8,
             "compliance_loose": 0.85},
        ],
        cap_scales={2.0: 0.6, 4.0: 0.6},          # both pass MMLU
        gsm8k_scales={2.0: 0.48, 4.0: 0.30},      # 4.0 fails GSM8K (-20pt)
    )
    # bidpo: candidate scale has NO panel gsm8k and no cache -> None
    _fake_cell(out, "json_format", "bidpo",
               steered=[{"arm": "steered", "scale": 1.0,
                         "compliance_strict": 0.7, "compliance_loose": 0.7}],
               cap_scales={1.0: 0.6}, gsm8k_scales={})
    # psr: no panel gsm8k, but the fill-in cache carries a passing value
    _fake_cell(out, "json_format", "psr",
               steered=[{"arm": "steered", "scale": 1.0,
                         "compliance_strict": 0.7, "compliance_loose": 0.7}],
               cap_scales={1.0: 0.6}, gsm8k_scales={})
    (out / "frontier_gsm8k_fillins.json").write_text(
        json.dumps({"json_format|psr|1": 0.5}))
    monkeypatch.setattr(sys, "argv", [
        "stage2_frontier.py", "--out-dir", str(out), "--dry",
    ])
    stage2_frontier.main()
    f = json.loads((out / "frontier.preview.json").read_text())
    vec = {c["scale"]: c for c in f["json_format"]["vector"]["candidates"]}
    assert vec[4.0]["in_budget"] is False           # GSM8K gate
    assert vec[2.0]["in_budget"] is True
    assert f["json_format"]["vector"]["frontier"]["scale"] == 2.0
    bid = f["json_format"]["bidpo"]["candidates"][0]
    assert bid["in_budget"] is None                 # unmeasured = not eligible
    assert f["json_format"]["bidpo"]["frontier"] is None
    psr = f["json_format"]["psr"]["candidates"][0]
    assert psr["gsm8k"] == pytest.approx(0.5)       # cache honored
    assert psr["in_budget"] is True


def test_frontier_legacy_budget_is_mmlu_only(tmp_path, monkeypatch):
    """--legacy-budget reproduces the stage-1-comparable map: a scale that
    fails or lacks GSM8K stays in budget when MMLU/distinct3 pass."""
    out = tmp_path / "ifeval"
    _fake_cell(out, "json_format", "vector",
               steered=[{"arm": "steered", "scale": 1.0,
                         "compliance_strict": 0.7, "compliance_loose": 0.7}],
               cap_scales={1.0: 0.6}, gsm8k_scales={1.0: 0.10})
    monkeypatch.setattr(sys, "argv", [
        "stage2_frontier.py", "--out-dir", str(out), "--dry",
        "--legacy-budget",
    ])
    stage2_frontier.main()
    f = json.loads((out / "frontier.preview.json").read_text())
    assert f["json_format"]["vector"]["frontier"]["scale"] == 1.0


def test_dose1_effect_reuse_and_null_handling(tmp_path, monkeypatch):
    """Fixed-dose panel: the on-target effect is the paired per-prompt
    diff at scale 1.0; off-target metrics reuse panel values, the
    GSM8K fill-in cache, and the dose1 cache; anything unmeasured stays
    null under --dry (never silently zero)."""
    import stage2_dose1

    out = tmp_path / "ifeval"
    _fake_cell(out, "json_format", "vector",
               steered=[{"arm": "steered", "scale": 1.0,
                         "compliance_strict": 0.7, "compliance_loose": 0.7}],
               cap_scales={1.0: 0.59}, gsm8k_scales={1.0: 0.48})
    _fake_cell(out, "json_format", "bidpo",
               steered=[{"arm": "steered", "scale": 1.0,
                         "compliance_strict": 0.7, "compliance_loose": 0.7}],
               cap_scales={}, gsm8k_scales={})
    (out / "frontier_gsm8k_fillins.json").write_text(
        json.dumps({"json_format|bidpo|1": 0.42}))
    (out / "dose1_fillins.json").write_text(
        json.dumps({"json_format|bidpo|mmlu": 0.58}))
    monkeypatch.setattr(sys, "argv", [
        "stage2_dose1.py", "--out-dir", str(out), "--dry"])
    stage2_dose1.main()
    d = json.loads((out / "dose1.json").read_text())
    vec = d["json_format"]["vector"]
    # fixture: steered strict=1 on all 6 prompts, prompted strict=1 too
    assert vec["n_prompts"] == 6 and vec["effect"] == pytest.approx(0.0)
    assert vec["mmlu"] == pytest.approx(0.59)          # panel reuse
    assert vec["mmlu_delta"] == pytest.approx(-0.01)   # vs base 0.6
    assert vec["gsm8k"] == pytest.approx(0.48)
    bid = d["json_format"]["bidpo"]
    assert bid["gsm8k"] == pytest.approx(0.42)         # fill-in reuse
    assert bid["sources"]["gsm8k"] == "fillin-cache"
    assert bid["mmlu"] == pytest.approx(0.58)          # dose1 cache reuse
    assert bid["gpqa_diamond"] is None                 # unmeasured stays null
    assert bid["gpqa_diamond_delta"] is None
    assert d["_bases"]["mmlu"] == pytest.approx(0.6)


def test_frontier_folds_ladder_rungs(tmp_path, monkeypatch):
    out = tmp_path / "ifeval"
    ladder = tmp_path / "ladder"
    _fake_cell(out, "json_format", "vector",
               steered=[{"arm": "steered", "scale": 1.0,
                         "compliance_strict": 0.7, "compliance_loose": 0.7}],
               cap_scales={1.0: 0.6})
    _fake_cell(ladder, "json_format", "rung3",
               steered=[{"arm": "steered", "scale": 1.0,
                         "compliance_strict": 0.85, "compliance_loose": 0.9}],
               cap_scales={1.0: 0.6})
    monkeypatch.setattr(sys, "argv", [
        "stage2_frontier.py", "--out-dir", str(out),
        "--ladder-dir", str(ladder), "--dry",
    ])
    stage2_frontier.main()
    frontier = json.loads((out / "frontier.preview.json").read_text())
    assert frontier["json_format"]["rung3"]["frontier"]["compliance"] == \
        pytest.approx(0.85)


def test_frontier_dry_never_touches_frontier_json(tmp_path, monkeypatch):
    """After a REAL frontier exists, a --dry rerun must leave
    frontier.json byte-identical and write only frontier.preview.json."""
    out = tmp_path / "ifeval"
    _fake_cell(out, "json_format", "vector",
               steered=[{"arm": "steered", "scale": 1.0,
                         "compliance_strict": 0.7, "compliance_loose": 0.7}],
               cap_scales={1.0: 0.6})
    real = json.dumps({"json_format": {"vector": {"frontier": {
        "scale": 1.0, "compliance": 0.7}}}})
    (out / "frontier.json").write_text(real)
    monkeypatch.setattr(sys, "argv", [
        "stage2_frontier.py", "--out-dir", str(out), "--dry",
    ])
    stage2_frontier.main()
    assert (out / "frontier.json").read_text() == real  # untouched
    assert (out / "frontier.preview.json").exists()


def test_frontier_missing_mmlu_is_not_eligible(tmp_path, monkeypatch):
    """A candidate whose MMLU was never measured is UNKNOWN
    (in_budget=null) and can never be picked — under --dry a cell with no
    measured candidate has NO frontier point, never a fake in-budget one."""
    out = tmp_path / "ifeval"
    _fake_cell(out, "json_format", "vector",
               steered=[
                   {"arm": "steered", "scale": 1.0, "compliance_strict": 0.7,
                    "compliance_loose": 0.7},          # no cap entry -> unknown
                   {"arm": "steered", "scale": 2.0, "compliance_strict": 0.72,
                    "compliance_loose": 0.72},         # measured + in budget
               ],
               cap_scales={2.0: 0.6})
    monkeypatch.setattr(sys, "argv", [
        "stage2_frontier.py", "--out-dir", str(out), "--dry",
    ])
    stage2_frontier.main()
    frontier = json.loads((out / "frontier.preview.json").read_text())
    vec = frontier["json_format"]["vector"]
    by_scale = {c["scale"]: c for c in vec["candidates"]}
    assert by_scale[1.0]["in_budget"] is None      # unknown, not eligible
    assert by_scale[2.0]["in_budget"] is True
    assert vec["frontier"]["scale"] == 2.0          # only the measured one


# ---------------------------------------------------------------- end to end


def _tiny_cell_cfg(tmp_path) -> dict:
    return {
        "model": "unused-preloaded",
        "out_dir": str(tmp_path),
        "layer": 1,
        "scales": [1.0],
        "prompts": {"source": "seed", "set": "general"},
        "n_fit_prompts": 2,
        "n_eval_prompts": 2,
        "max_new_tokens": 4,
        "capability": False,          # dataset-backed panel: not a unit test
        "wandb_project": None,
        "train_kwargs": {"default": {"epochs": 1, "batch_size": 2,
                                     "max_len": 64}},
    }


def test_ifeval_cell_end_to_end_on_tiny_model(tiny_llama, tmp_path):
    """The full C1 cell path on the tiny model: P1-shared pairs ->
    registry training -> manifest + artifact save -> baseline/prompted/
    steered/both eval -> results.json in the judge_batch layout ->
    resume skip."""
    model, tok = tiny_llama
    cfg = _tiny_cell_cfg(tmp_path)
    # P1 first: ifeval cells REQUIRE the shared pairs file.
    from sisteer.data.prompts import seed_prompts, split_prompts

    fit, _ = split_prompts(seed_prompts("general"), holdout_frac=0.5)
    stage2_run._ensure_pairs(model, tok, resolve_si("json"), fit[:2],
                             tmp_path / "json" / "pairs.jsonl", 4)
    summary = stage2_run._run_ifeval_cell(model, tok, cfg, "json",
                                          "vector", tmp_path)
    assert summary["instruction"] == "json"
    assert summary["canonical_id"] == "json"
    assert summary["method"] == "vector"
    arms = {a["arm"] for a in summary["arms"]}
    assert arms == {"baseline", "prompted", "steered", "both"}

    cell = tmp_path / "json" / "vector"
    assert (cell / "results.json").exists()
    assert (cell / "vector.safetensors").exists()      # the artifact
    assert (cell / "manifest.json").exists()           # provenance
    manifest = json.loads((cell / "manifest.json").read_text())
    assert manifest["arm"] == "vector"
    assert manifest["si"] == "json"
    assert manifest["config_hash"] == stage2_run._cell_config_hash(
        cfg, "vector")
    assert (tmp_path / "json" / "pairs.jsonl").exists()  # shared pairs
    results = json.loads((cell / "results.json").read_text())
    assert len(results["records"]) == len(results["arms"])

    # resume: an existing results.json short-circuits the whole cell
    again = stage2_run._run_ifeval_cell(model, tok, cfg, "json",
                                        "vector", tmp_path)
    assert again.get("skipped") is True


def test_ifeval_cell_requires_p1_pairs(tiny_llama, tmp_path):
    """With NO shared pairs file, the ifeval cell fails loudly
    naming the P1 stage — never a silent per-cell regeneration (that is
    the race: concurrent arm cells writing one shared file)."""
    model, tok = tiny_llama
    cfg = _tiny_cell_cfg(tmp_path)
    with pytest.raises(SystemExit, match="P1 pairs"):
        stage2_run._run_ifeval_cell(model, tok, cfg, "json",
                                    "vector", tmp_path)


# ---------------------------------------------------------------- pairs race


def test_ensure_pairs_skips_generation_when_file_exists(tmp_path, monkeypatch):
    from sisteer.data.pairs import Pair, save_pairs

    path = tmp_path / "pairs.jsonl"
    save_pairs([Pair(prompt="p", pos="y", neg="n",
                     instruction_id="json")], path)
    monkeypatch.setattr(stage2_run, "generate_pairs",
                        lambda *a, **kw: pytest.fail("must not generate"))
    out = stage2_run._ensure_pairs(None, None, resolve_si("json"),
                                   [], path, 4)
    assert len(out) == 1 and out[0].prompt == "p"


def test_ensure_pairs_rechecks_after_acquiring_the_lock(tmp_path, monkeypatch):
    """Composition-style generation acquires an exclusive flock and
    RECHECKS existence — a peer that won the race means we load its file
    and never generate."""
    from contextlib import contextmanager

    from sisteer.data.pairs import Pair, save_pairs

    path = tmp_path / "pairs.jsonl"

    @contextmanager
    def peer_wins_the_race(pairs_path):
        # as-if a concurrent cell generated + published while we waited
        save_pairs([Pair(prompt="peer", pos="y", neg="n",
                         instruction_id="json")], pairs_path)
        yield

    monkeypatch.setattr(stage2_run, "_pairs_lock", peer_wins_the_race)
    monkeypatch.setattr(stage2_run, "generate_pairs",
                        lambda *a, **kw: pytest.fail("must not generate"))
    out = stage2_run._ensure_pairs(None, None, resolve_si("json"),
                                   [], path, 4)
    assert [p.prompt for p in out] == ["peer"]


def test_ensure_pairs_generates_under_lock_and_publishes(tmp_path, monkeypatch):
    from sisteer.data.pairs import Pair

    path = tmp_path / "sub" / "pairs.jsonl"
    monkeypatch.setattr(
        stage2_run, "generate_pairs",
        lambda *a, **kw: [Pair(prompt="g", pos="y", neg="n",
                               instruction_id="json")])
    out = stage2_run._ensure_pairs(None, None, resolve_si("json"),
                                   ["g"], path, 4)
    assert len(out) == 1
    assert path.exists()
    assert path.with_name(path.name + ".lock").exists()  # the flock file


# ---------------------------------------------------------------- manifests


def test_ensure_object_manifest_resume_and_mismatch(tiny_llama, tmp_path):
    """Lightweight resume provenance: a matching manifest resumes;
    a mismatched one hard-fails NAMING the mismatch; a missing one (torn
    publish / pre-manifest artifact) hard-fails too."""
    from sisteer.data.pairs import Pair

    model, tok = tiny_llama
    cfg = _tiny_cell_cfg(tmp_path) | {"model_revision": "rev-a"}
    si = resolve_si("json")
    cell = tmp_path / "cell"
    pairs = [Pair(prompt="write a thing", pos="a thing", neg="",
                  instruction_id=si.id)] * 2

    obj = stage2_run._ensure_object(model, tok, si, "vector", cfg, cell,
                                    pairs_fn=lambda: pairs)
    assert (cell / "vector.safetensors").exists()
    assert (cell / "manifest.json").exists()

    # matching manifest -> resumes (loads, never retrains)
    again = stage2_run._ensure_object(
        model, tok, si, "vector", cfg, cell,
        pairs_fn=lambda: pytest.fail("resume must not retrain"))
    assert again is not obj  # a fresh load of the same artifact

    # model_revision mismatch -> SystemExit naming the field
    with pytest.raises(SystemExit, match="model_revision"):
        stage2_run._ensure_object(
            model, tok, si, "vector", cfg | {"model_revision": "rev-b"},
            cell, pairs_fn=lambda: pairs)

    # config-subset change (layer) -> config_hash mismatch
    with pytest.raises(SystemExit, match="config_hash"):
        stage2_run._ensure_object(
            model, tok, si, "vector", cfg | {"layer": 2},
            cell, pairs_fn=lambda: pairs)

    # missing manifest next to an existing artifact -> loud failure
    (cell / "manifest.json").unlink()
    with pytest.raises(SystemExit, match="manifest"):
        stage2_run._ensure_object(model, tok, si, "vector", cfg, cell,
                                  pairs_fn=lambda: pairs)


# ---------------------------------------------------------------- canary quarantine


def _no_strays(out_root: Path) -> list[Path]:
    """Every file the canary produced, EXCEPT those under out/canary/ —
    must be empty or the array's artifact-exists resume would silently
    adopt canary-budget artifacts."""
    return [p for p in out_root.rglob("*")
            if p.is_file() and p.relative_to(out_root).parts[0] != "canary"]


def test_composition_canary_quarantines_all_artifacts(
    tiny_llama, tmp_path, monkeypatch
):
    """The composition canary truncates fit prompts and targets (an <and>
    trained on 1 pair, not 8); every artifact it writes — pairs, objects,
    and_token, results — must land under out_dir/canary/, invisible to
    the A4 array's resume."""
    model, tok = tiny_llama
    monkeypatch.setattr(stage2_run, "_load_model",
                        lambda cfg, **kw: (model, tok))
    out_root = tmp_path / "composition"
    cfg = {
        "model": "unused-preloaded",
        "out_dir": str(out_root),
        "layer": 1,
        "n_prompts_per_target": 2,
        "prompts": {"source": "seed", "set": "general"},
        "n_fit_prompts": 2,
        "n_eval_prompts": 2,
        "max_new_tokens": 4,
        "wandb_project": None,
        # joined 'do X and Y' teacher prompts run past 64 tokens; 256
        # keeps _encode_aligned happy on the tiny tokenizer
        "and_token": {"epochs": 1, "batch_size": 2, "max_len": 256},
        "train_kwargs": {
            "default": {"epochs": 1, "batch_size": 2, "max_len": 256},
            # tiny llama has no reserved sentinel token; unk stands in
            "token_faithful": {"sentinel": (tok.unk_token, tok.unk_token_id)},
        },
    }
    stage2_run.run_composition(cfg, SimpleNamespace(canary=True, arm=None))

    assert _no_strays(out_root) == []
    canary = out_root / "canary"
    assert (canary / "objects" / "and_token.pt").is_file()
    assert list((canary / "pairs").glob("*.jsonl"))
    assert list((canary / "objects" / "token_faithful").glob("*/object.pt"))
    assert (canary / "token_faithful" / "results.jsonl").is_file()


# ---------------------------------------------------------------- ladder driver


def _ladder_cfg(tmp_path) -> dict:
    return {
        "model": "unused-preloaded",
        "out_dir": str(tmp_path),
        "layer": 1,
        "scales": [1.0],
        "prompts": {"source": "seed", "set": "general"},
        "n_fit_prompts": 2,
        "n_eval_prompts": 2,
        "max_new_tokens": 4,
        "capability": False,
        "wandb_project": None,
        "rungs": [
            {"id": 2, "trainer": "mc_kl", "n_samples": 2, "temperature": 0.7},
            {"id": 3, "trainer": "gkd", "n_samples": 2,
             "rollout_temperature": 0.9},
        ],
        "train_kwargs": {
            "default": {"epochs": 1, "batch_size": 2, "max_len": 64},
            "gkd": {"epochs": 1, "batch_size": 2, "max_len": 64},
        },
    }


def _fake_ladder_sv(model, si_id, method):
    from sisteer.vectors import SteeringVector

    d = model.config.hidden_size
    return SteeringVector(
        vectors={1: 0.01 * torch.randn(d)}, method=method,
        model_name="tiny", instruction_id=si_id,
        metadata={"n_trajectories_seen": 12, "n_optimizer_steps": 6},
    )


def test_ladder_rung2_uses_train_mc_kl_and_records_budget(
    tiny_llama, tmp_path, monkeypatch
):
    """Driver seam: rung 2 trains through ladder.train_mc_kl (the
    budget-matched sampled-teacher trainer) — never the old
    sample-once + train_distill path — and results.json carries the
    trajectory/step budget from the vector metadata."""
    import sisteer.extraction.ladder as ladder_mod

    model, tok = tiny_llama
    monkeypatch.setattr(stage2_run, "_load_model",
                        lambda cfg, **kw: (model, tok))
    calls = {}

    def fake_mc_kl(m, t, si, fit, layer, **kw):
        calls["kwargs"] = kw
        calls["layer"] = layer
        return _fake_ladder_sv(model, si.id, "mc_kl")

    monkeypatch.setattr(ladder_mod, "train_mc_kl", fake_mc_kl)
    cfg = _ladder_cfg(tmp_path)
    stage2_run.run_ladder(cfg, SimpleNamespace(cell="json", rung=2))

    assert calls["kwargs"]["n_samples"] == 2       # rung config, not default
    assert calls["kwargs"]["temperature"] == 0.7
    assert calls["kwargs"]["epochs"] == 1          # train_kwargs.default
    assert calls["layer"] == 1
    out = tmp_path / "json" / "rung2"
    results = json.loads((out / "results.json").read_text())
    assert results["budget"] == {"n_trajectories_seen": 12,
                                 "n_optimizer_steps": 6}
    assert (out / "vector.safetensors").exists()
    assert (out / "manifest.json").exists()        # provenance


def test_ladder_rung3_passes_n_samples_to_train_gkd(
    tiny_llama, tmp_path, monkeypatch
):
    """Driver seam: rung 3 hands the rung config's n_samples to train_gkd
    so the 2-vs-3 budget match holds (the default n_samples=1
    would silently break it)."""
    import sisteer.extraction.ladder as ladder_mod

    model, tok = tiny_llama
    monkeypatch.setattr(stage2_run, "_load_model",
                        lambda cfg, **kw: (model, tok))
    calls = {}

    def fake_gkd(m, t, si, fit, layer, **kw):
        calls["kwargs"] = kw
        return _fake_ladder_sv(model, si.id, "gkd")

    monkeypatch.setattr(ladder_mod, "train_gkd", fake_gkd)
    cfg = _ladder_cfg(tmp_path)
    stage2_run.run_ladder(cfg, SimpleNamespace(cell="json", rung=3))

    assert calls["kwargs"]["n_samples"] == 2
    assert calls["kwargs"]["rollout_temperature"] == 0.9
    results = json.loads(
        (tmp_path / "json" / "rung3" / "results.json").read_text())
    assert results["budget"]["n_trajectories_seen"] == 12


# ---------------------------------------------------------------- composition driver


def test_composition_monolithic_keyed_by_canonical_order(
    tiny_llama, tmp_path, monkeypatch
):
    """Driver seam: monolithic objects are keyed (and their joint
    instruction rendered) in canonical_order(names) = sorted(names), even
    when a target arrives unsorted — eval_composition refuses
    non-canonical keys, so a driver keying by raw target order would
    KeyError on any unsorted target."""
    from sisteer.eval import composition as comp_mod

    model, tok = tiny_llama
    monkeypatch.setattr(stage2_run, "_load_model",
                        lambda cfg, **kw: (model, tok))
    # an UNSORTED two-behavior target (sorted: lang_fr < lowercase)
    monkeypatch.setattr(comp_mod, "default_targets",
                        lambda seed=0: {"seen": [("lowercase", "lang_fr")]})
    out_root = tmp_path / "composition"
    cfg = {
        "model": "unused-preloaded",
        "out_dir": str(out_root),
        "layer": 1,
        "n_prompts_per_target": 2,
        "prompts": {"source": "seed", "set": "general"},
        "n_fit_prompts": 2,
        "n_eval_prompts": 2,
        "max_new_tokens": 4,
        "wandb_project": None,
        "and_token": {"epochs": 1, "batch_size": 2, "max_len": 256},
        "train_kwargs": {
            "default": {"epochs": 1, "batch_size": 2, "max_len": 256},
            "token_faithful": {"sentinel": (tok.unk_token, tok.unk_token_id)},
        },
    }
    stage2_run.run_composition(cfg, SimpleNamespace(canary=True, arm=None))
    canary = out_root / "canary"
    # the monolithic object dir is named by the CANONICAL (sorted) order
    assert (canary / "objects" / "token_faithful" /
            "monolithic_lang_fr+lowercase").is_dir()
    recs = [json.loads(l) for l in
            (canary / "token_faithful" / "results.jsonl")
            .read_text().splitlines() if l.strip()]
    mono = [r for r in recs if r["arm"] == "monolithic"]
    assert mono and all(r["ordering"] == "canonical" for r in mono)
    assert all(r["object_kind"] == "token_appended" for r in mono)


# ---------------------------------------------------------------- infra pins


def test_stage2_common_pins_the_infra_fixes():
    sh = (REPO / "scripts" / "sbatch" / "stage2_common.sh").read_text()
    # Stale-clone guard — the UNIFIED checkout seam: S2_ROOT is
    # primary (capeval-controls §6; root-derived EXPECTED_SHA/wandb/venv),
    # honoring S2_REPO_ROOT as an alias for the
    # same seam; S2_REPO_DIR derives from it and stays the capeval §3.3
    # override for a checkout not named `repo`. Defaults are byte-identical
    # to the historical shared paths — pinned behaviorally by
    # test_capeval_stage2_common.py and structurally by test_sbatch_lint.py.
    assert 'S2_ROOT="${S2_ROOT:-${S2_REPO_ROOT:-${SISTEER_PROJECT_ROOT:?set SISTEER_PROJECT_ROOT, the cluster project directory (the front door exports it from the site config)}}}"' in sh
    assert 'S2_REPO_ROOT="${S2_REPO_ROOT:-$S2_ROOT}"' in sh
    assert 'S2_REPO_DIR="${S2_REPO_DIR:-$S2_ROOT/repo}"' in sh
    assert 'git -C "$S2_REPO_DIR" rev-parse HEAD' in sh
    # ... and an explicit S2_EXPECTED_SHA_FILE overrides the derived guard
    # location (a sibling checkout not named /repo would otherwise look
    # inside the checkout for its EXPECTED_SHA).
    assert ('local expected_file="${S2_EXPECTED_SHA_FILE:-'
            '${S2_REPO_DIR%/repo}/EXPECTED_SHA}"') in sh
    assert "EXPECTED_SHA" in sh
    # MED-4: no wandb sync EXECUTED on the compute node's exit trap (the
    # login-node post-step may be documented in comments)
    finish = sh.split("s2_finish()", 1)[1].split("}", 1)[0]
    active_lines = [l for l in finish.splitlines()
                    if l.strip() and not l.strip().startswith("#")]
    assert not any("wandb sync" in l for l in active_lines)
    # M13: nltk data on shared storage
    assert 'NLTK_DATA="$SHARED/nltk_data"' in sh
    # Supersedes the earlier loud fallback: `uv sync --frozen` or
    # FAIL — no unfrozen re-resolve anywhere — and the sync runs under a
    # per-node flock so concurrent array tasks can't corrupt the venv.
    assert "re-resolving OUTSIDE the lockfile" not in sh
    assert "uv sync --frozen" in sh
    active = [l for l in sh.splitlines()
              if l.strip() and not l.strip().startswith("#")]
    assert not any("uv sync" in l and "--frozen" not in l for l in active)
    assert "flock -x 9" in sh
    assert "FATAL: 'uv sync --frozen' failed" in sh


def test_stage2_common_torch_split_pins():
    """torch-cpu-split: production path defaults, the cu128-pinned frozen
    sync, the srun-time runtime assert, and the mandatory submit-time SHA
    pin are all load-bearing strings in stage2_common.sh."""
    sh = (REPO / "scripts" / "sbatch" / "stage2_common.sh").read_text()
    # parameterized roots with PRODUCTION DEFAULTS pinned (unset knobs
    # must leave production jobs byte-for-byte unchanged)
    assert ('S2_ROOT="${S2_ROOT:-${S2_REPO_ROOT:-${SISTEER_PROJECT_ROOT:?set SISTEER_PROJECT_ROOT, the cluster project directory (the front door exports it from the site config)}}}"'
            in sh)  # S2_REPO_ROOT honored as the same seam under its
    # torch-cpu-split name
    assert 'S2_REPO_ROOT="${S2_REPO_ROOT:-$S2_ROOT}"' in sh
    # root-NAMED default since the unified-seam merge (historical root ->
    # the same venv byte-identically)
    assert ('S2_UV_ENV="${S2_UV_ENV:-/home/$USER/venvs/'
            '$(basename "$S2_ROOT")}"' in sh)
    assert 'export UV_PROJECT_ENVIRONMENT="$S2_UV_ENV"' in sh
    # the cd is parameterized by S2_REPO_DIR (capeval F18), which
    # defaults to $S2_REPO_ROOT/repo — identical when unset
    assert 'cd "$S2_REPO_DIR" || exit 1' in sh
    # the frozen sync pins the GPU flavor
    assert 'uv sync --frozen --extra cu128 "${uv_extra_flags[@]}"' in sh
    # srun-time cu128 runtime assert, called from s2_init AFTER the sync
    assert "s2_assert_cu128() {" in sh
    body = sh.split("s2_init() {", 1)[1].split("\n}", 1)[0]
    assert re.search(r"^\s*s2_assert_cu128\s*$", body, re.M), (
        "s2_init must CALL s2_assert_cu128")
    assert body.index("uv sync --frozen --extra cu128") < body.index(
        "\n  s2_assert_cu128")
    assert 'torch.__version__ == "2.11.0+cu128"' in sh
    assert "torch.cuda.is_available()" in sh
    assert "torch.cuda.device_count() >= 1" in sh
    # mandatory submit-time SHA pin: trailing-space-safe parse,
    # strict validation, FATAL-when-absent with the fix incantation printed
    assert (r"sed -n 's/^ *Comment=\([^[:space:]]*\)[[:space:]]*$/\1/p'"
            in sh)
    assert "'^sha:[0-9a-f]{40}$'" in sh
    assert "FATAL: submit-time SHA pin missing or ambiguous" in sh
    assert "FATAL: malformed submit-time SHA pin" in sh
    assert "FATAL: cannot read this job's submit-time SHA pin" in sh
    assert sh.count("sbatch --comment=sha:") >= 3, (
        "every pin FATAL must print the fix incantation")
    assert 'SUBMIT_SHA="${pin_lines#sha:}"' in sh
    # HEAD must equal the pin (the check EXPECTED_SHA cannot make when a
    # re-sync moves both comparands together)
    assert "!= submit-time pin" in sh
    # s2_escalate forwards the ORIGINAL pin on preemption resubmits
    esc = sh.split("s2_escalate()", 1)[1].split("\n}", 1)[0]
    assert 'comment_flag=(--comment="sha:$SUBMIT_SHA")' in esc
    assert '"${comment_flag[@]}"' in esc


# The cluster's ACTUAL scontrol rendering (verified on the login host, Slurm
# 23.11.4): a SET comment line carries a TRAILING SPACE; an UNSET comment is
# OMITTED from the output entirely.
_PIN_SHA = "a" * 40
_SCONTROL_WITH_PIN = (
    "JobId=112233 JobName=tsplit-proof\n"
    "   UserId=testuser(1000) GroupId=testuser(1000) MCS_label=N/A\n"
    "   Priority=4294901728 Nice=0 Account=example QOS=high\n"
    f"   Comment=sha:{_PIN_SHA} \n"          # <- trailing space is REAL
    "   Command=/tmp/nonexistent/proof.sbatch\n"
    "   WorkDir=/tmp\n"
)
_SCONTROL_NO_PIN = _SCONTROL_WITH_PIN.replace(
    f"   Comment=sha:{_PIN_SHA} \n", "")

# The array-master rendering (caught in the wild by 139589_20): the
# task that inherits the ARRAY-MASTER JobId gets one record per VISIBLE
# sibling from 'scontrol show job $SLURM_JOB_ID', each carrying the SAME
# inherited submission-level --comment. Three records, one pin.
_SCONTROL_THREE_IDENTICAL = (
    "JobId=139589 ArrayJobId=139589 ArrayTaskId=20 JobName=s2-array\n"
    f"   Comment=sha:{_PIN_SHA} \n"
    "   Command=/tmp/nonexistent/proof.sbatch\n"
    "\n"
    "JobId=140779 ArrayJobId=139589 ArrayTaskId=19 JobName=s2-array\n"
    f"   Comment=sha:{_PIN_SHA} \n"
    "   Command=/tmp/nonexistent/proof.sbatch\n"
    "\n"
    "JobId=140769 ArrayJobId=139589 ArrayTaskId=18 JobName=s2-array\n"
    f"   Comment=sha:{_PIN_SHA} \n"
    "   Command=/tmp/nonexistent/proof.sbatch\n"
)
# Genuinely differing Comment values across records — REAL ambiguity, which
# must stay FATAL (the pin parse is fail-closed).
_SCONTROL_TWO_DIFFERING = (
    "JobId=1000 JobName=s2-array\n"
    f"   Comment=sha:{_PIN_SHA} \n"
    "JobId=1001 JobName=s2-array\n"
    f"   Comment=sha:{'b' * 40} \n"
)


def _pin_sed_expr() -> str:
    """The EXACT sed parse from stage2_common.sh (not a copy that could
    drift)."""
    sh = (REPO / "scripts" / "sbatch" / "stage2_common.sh").read_text()
    m = re.search(r"sed -n '([^']*Comment=[^']*)'", sh)
    assert m, "pin-parse sed expression not found in stage2_common.sh"
    return m.group(1)


def test_submit_sha_pin_parse_fixtures():
    """The parse must extract the pin from the cluster's REAL
    trailing-space rendering, and must extract NOTHING from absent,
    multi-word, or empty comments (fail-closed at the count!=1 check)."""
    expr = _pin_sed_expr()

    def parse(payload: str) -> list[str]:
        res = subprocess.run(["sed", "-n", expr], input=payload,
                             capture_output=True, text=True, check=True)
        return [l for l in res.stdout.splitlines() if l]

    # the real set-case rendering (trailing space) parses to EXACTLY the pin
    assert parse(_SCONTROL_WITH_PIN) == [f"sha:{_PIN_SHA}"]
    assert re.fullmatch(r"sha:[0-9a-f]{40}", parse(_SCONTROL_WITH_PIN)[0])
    # absent comment (omitted line): nothing extracted
    assert parse(_SCONTROL_NO_PIN) == []
    # a multi-word human comment never misparses
    assert parse("   Comment=deploy tomorrow \n") == []
    # an empty comment value yields nothing
    assert parse("   Comment= \n") == []
    # two Comment lines both surface from the raw parse; the count check
    # collapses IDENTICAL values (one inherited submission comment)
    # and FATALs only on genuinely differing ones
    two = _SCONTROL_WITH_PIN + f"   Comment=sha:{'b' * 40} \n"
    assert len(parse(two)) == 2


def _run_s2_init(tmp_path: Path, *, scontrol_payload: str | None,
                 head_sha: str = _PIN_SHA,
                 expected_sha: str | None = None,
                 extra_env: dict[str, str] | None = None,
                 site: bool = True,
                 after: str = ""):
    """Run the REAL s2_init end-to-end with stubbed scontrol/git/uv/srun
    and an isolated S2_REPO_ROOT/S2_UV_ENV (the torch-split
    parameterization). Returns (returncode, combined output). The scontrol
    stub appends each invocation's argv to bin/scontrol.args so tests can
    assert WHICH job id was queried (array-task scoping)."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    args_log = shlex.quote(str(bin_dir / "scontrol.args"))
    if scontrol_payload is None:
        (bin_dir / "scontrol").write_text(
            f'#!/bin/bash\necho "$@" >> {args_log}\nexit 1\n')
    else:
        (bin_dir / "scontrol").write_text(
            f'#!/bin/bash\necho "$@" >> {args_log}\n'
            f"printf '%s' {shlex.quote(scontrol_payload)}\n")
    (bin_dir / "git").write_text(f"#!/bin/bash\necho {head_sha}\n")
    (bin_dir / "uv").write_text("#!/bin/bash\nexit 0\n")
    (bin_dir / "srun").write_text("#!/bin/bash\nexit 0\n")
    # a resubmission's argv, one per line (read by the tests that fire s2_escalate after s2_init)
    (bin_dir / "sbatch").write_text(f'#!/bin/bash\nprintf "%s\\n" "$@" > {shlex.quote(str(bin_dir / "sbatch.args"))}\n')
    for name in ("scontrol", "git", "uv", "srun", "sbatch"):
        (bin_dir / name).chmod(0o755)
    root = tmp_path / "clone"
    (root / "repo").mkdir(parents=True)
    (root / "EXPECTED_SHA").write_text(
        (expected_sha if expected_sha is not None else head_sha) + "\n")
    common = REPO / "scripts" / "sbatch" / "stage2_common.sh"
    harness = f"""
set -uo pipefail
S2_REPO_ROOT={shlex.quote(str(root))}
S2_UV_ENV={shlex.quote(str(tmp_path / 'venvs' / 'tsplit-test'))}
source {shlex.quote(str(common))}
s2_init
echo "S2_INIT_OK SUBMIT_SHA=$SUBMIT_SHA"
{after}
"""
    env = dict(os.environ, PATH=f"{bin_dir}:{os.environ['PATH']}",
               SLURM_JOB_ID="112233", SLURM_JOB_NAME="pin-harness")
    # the job's cluster values (synthetic): the shared root, as the front door passes it, and the QoS a preempted job
    # resubmits at, as its submitter sets it (a neutral name)
    for k in ("SISTEER_SHARED_ROOT", "SISTEER_CLUSTER_QOS", "ESCALATE_QOS"):
        env.pop(k, None)
    if site:
        env.update(SISTEER_SHARED_ROOT=str(tmp_path / "shared" / "u"), ESCALATE_QOS="qos_nonpreempt")
    # Deterministic non-array default even if the test itself runs under a
    # Slurm array allocation; tests opt in via extra_env.
    env.pop("SLURM_ARRAY_JOB_ID", None)
    env.pop("SLURM_ARRAY_TASK_ID", None)
    if extra_env:
        env.update(extra_env)
    res = subprocess.run(["bash", "-c", harness], env=env,
                         capture_output=True, text=True)
    return res.returncode, res.stdout + res.stderr


def test_s2_init_accepts_the_real_trailing_space_pin(tmp_path):
    """The pinned proof submission is the FIRST exercise of the set-case
    parse (P4.3) — with the cluster's actual trailing-space rendering,
    s2_init must pass whole and export the parsed SUBMIT_SHA."""
    rc, out = _run_s2_init(tmp_path, scontrol_payload=_SCONTROL_WITH_PIN)
    assert rc == 0, out
    assert f"S2_INIT_OK SUBMIT_SHA={_PIN_SHA}" in out


def test_s2_init_fatals_without_a_pin_and_prints_the_fix(tmp_path):
    """Fail-closed: an unpinned submission dies at s2_init with the
    exact submit incantation printed."""
    rc, out = _run_s2_init(tmp_path, scontrol_payload=_SCONTROL_NO_PIN)
    assert rc == 1
    assert "FATAL: submit-time SHA pin missing or ambiguous" in out
    assert "sbatch --comment=sha:" in out


def test_s2_init_fatals_on_malformed_or_unreadable_pin(tmp_path):
    """Fail-closed: malformed values and scontrol itself failing are
    both FATAL, never fallback-to-file."""
    bad = _SCONTROL_WITH_PIN.replace(f"sha:{_PIN_SHA}", "sha:short")
    sub = tmp_path / "malformed"
    sub.mkdir()
    rc, out = _run_s2_init(sub, scontrol_payload=bad)
    assert rc == 1
    assert "FATAL: malformed submit-time SHA pin 'sha:short'" in out
    sub = tmp_path / "unreadable"
    sub.mkdir()
    rc, out = _run_s2_init(sub, scontrol_payload=None)
    assert rc == 1
    assert "FATAL: cannot read this job's submit-time SHA pin" in out


def test_s2_init_accepts_identical_multi_record_comments(tmp_path):
    """Caught in the wild by 139589_20: the array task holding the
    array-master JobId gets one scontrol record per visible sibling, each
    with the SAME inherited submission comment. That is ONE pin repeated,
    not ambiguity — s2_init must pass and extract it."""
    rc, out = _run_s2_init(tmp_path,
                           scontrol_payload=_SCONTROL_THREE_IDENTICAL)
    assert rc == 0, out
    assert f"S2_INIT_OK SUBMIT_SHA={_PIN_SHA}" in out


def test_s2_init_fatals_on_genuinely_differing_comments(tmp_path):
    """The pin parse stays fail-closed: multiple Comment= values that
    actually DIFFER are real ambiguity and must still FATAL."""
    rc, out = _run_s2_init(tmp_path,
                           scontrol_payload=_SCONTROL_TWO_DIFFERING)
    assert rc == 1
    assert "FATAL: submit-time SHA pin missing or ambiguous" in out
    assert "(2 Comment= values parsed; D12)" in out


def test_s2_init_scopes_the_pin_query_to_the_array_task(tmp_path):
    """First belt: when SLURM_ARRAY_JOB_ID/SLURM_ARRAY_TASK_ID name
    this task, the pin query asks scontrol for the per-task record
    <array_job>_<task_id> (verified on the login host to return exactly
    one record), never the bare $SLURM_JOB_ID that sweeps in siblings."""
    rc, out = _run_s2_init(
        tmp_path, scontrol_payload=_SCONTROL_WITH_PIN,
        extra_env={"SLURM_ARRAY_JOB_ID": "139589",
                   "SLURM_ARRAY_TASK_ID": "20"})
    assert rc == 0, out
    argv_log = (tmp_path / "bin" / "scontrol.args").read_text()
    assert "show job 139589_20" in argv_log


def test_s2_init_fatals_when_head_moved_past_the_pin(tmp_path):
    """The P7 swap race: a job pinned at sha A that starts after the
    clone moved to sha B must die loudly even though HEAD == EXPECTED_SHA
    (both file comparands moved together)."""
    moved = "b" * 40
    rc, out = _run_s2_init(tmp_path, scontrol_payload=_SCONTROL_WITH_PIN,
                           head_sha=moved, expected_sha=moved)
    assert rc == 1
    assert "!= submit-time pin" in out


def test_escalation_forwards_the_submit_sha_pin(tmp_path):
    """s2_escalate resubmits with --comment=sha:$SUBMIT_SHA — the ORIGINAL
    validated sha — so an escalation crossing a production re-sync FATALs
    at the successor's s2_init instead of silently running new code."""
    argv, _out = _fire_escalation(
        tmp_path,
        "JobId=12345 JobName=s2-drift\n"
        "   RunTime=00:10:00 TimeLimit=1-08:00:00 TimeMin=N/A",
        submit_sha=_PIN_SHA,
    )
    assert f"--comment=sha:{_PIN_SHA}" in argv
    assert argv[-3:] == [str(tmp_path / "job.sh"), "2", "5"]


def test_config_carries_the_appended_arms_at_their_registry_positions():
    """The did/adasteer/alphasteer arms are carried by the
    config arms list + S2_ARMS (config-parity for --all-cells / analysis and
    the S2_ARMS==arms symmetry); since the 2026-08-19 registry-order reorder
    alphasteer sits at [15] and did/adasteer at the tail."""
    cfg = yaml.safe_load((REPO / "configs" / "stage2_llama8b.yaml").read_text())
    assert cfg["arms"][15] == "alphasteer"
    assert cfg["arms"][-2:] == ["did", "adasteer"]


def test_config_carries_the_lofit_arm_and_its_recipe():
    """The lofit arm is carried by the config arms list + S2_ARMS at
    index 16 since the 2026-08-19 registry-order reorder — was 18 — with
    its llama2-7B recipe pinned in the config's train_kwargs."""
    cfg = yaml.safe_load((REPO / "configs" / "stage2_llama8b.yaml").read_text())
    assert cfg["arms"][16] == "lofit"  # the ladder arms follow it
    tk = cfg["train_kwargs"]["lofit"]
    # their llama2-7B recipe, pinned in provenance (extraction/lofit.py)
    assert tk["lr_select"] == 5e-3 and tk["lr_bias"] == 1e-2
    assert tk["l1_lambda"] == 1e-3
    assert tk["n_bias_heads"] == 32  # 32 of 1024 heads (~3%)


# Production job scripts run `set -uo pipefail`; the proof job adds `set -e`
# after s2_init. The TERM/escalation path must survive BOTH (adversarial
# review 2026-08-15: under errexit the handler died at the kill of an
# already-reaped workload and at the scontrol TimeLimit assignment, so a
# preempted job never resubmitted).
_SET_MODES = ("-uo pipefail", "-euo pipefail")


def _fire_escalation(tmp_path: Path, scontrol_payload: str | None,
                     submit_sha: str | None = None,
                     set_flags: str = "-uo pipefail"):
    """Source stage2_common.sh with stubbed sbatch/scontrol on PATH and
    fire s2_escalate exactly as a draining job would. Returns (the argv
    the sbatch stub received — one arg per line, [] if never called —
    and the combined log output)."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    calls = tmp_path / "sbatch_argv"
    (bin_dir / "sbatch").write_text(
        f"#!/bin/bash\nprintf '%s\\n' \"$@\" > {shlex.quote(str(calls))}\n")
    (bin_dir / "scontrol").write_text(
        "#!/bin/bash\n"
        + (f"printf '%s\\n' {shlex.quote(scontrol_payload)}\n"
           if scontrol_payload is not None else "exit 1\n"))
    for name in ("sbatch", "scontrol"):
        (bin_dir / name).chmod(0o755)
    job_script = tmp_path / "job.sh"
    job_script.write_text("#!/bin/bash\n")
    common = REPO / "scripts" / "sbatch" / "stage2_common.sh"
    pin_line = f"SUBMIT_SHA={submit_sha}" if submit_sha else ""
    harness = f"""
set {set_flags}
source {shlex.quote(str(common))}
ATTEMPT=1; MAX_ATTEMPTS=5; ESCALATE_QOS=high; RUN_NAME=testrun
STOP_DIR={shlex.quote(str(tmp_path / 'stop'))}
SCRIPT_PATH={shlex.quote(str(job_script))}
{pin_line}
s2_escalate
"""
    env = dict(os.environ, PATH=f"{bin_dir}:{os.environ['PATH']}",
               SLURM_JOB_ID="12345")
    res = subprocess.run(["bash", "-c", harness], env=env,
                         capture_output=True, text=True)
    assert res.returncode == 0, res.stdout + res.stderr
    argv = calls.read_text().splitlines() if calls.exists() else []
    return argv, res.stdout + res.stderr


def test_escalation_resubmit_carries_the_jobs_time_limit(tmp_path):
    """A submit-time --time override (drift task 20, longctx task 15) is
    not in the script header, so the resubmit must carry the job's ACTUAL
    TimeLimit — silently reverting to the header default on attempts 2+
    was the bug."""
    argv, _out = _fire_escalation(tmp_path, (
        "JobId=12345 JobName=s2-drift\n"
        "   RunTime=00:10:00 TimeLimit=1-08:00:00 TimeMin=N/A"
    ))
    assert "--time=1-08:00:00" in argv
    assert "--qos=high" in argv
    assert argv[-3:] == [str(tmp_path / "job.sh"), "2", "5"]


@pytest.mark.parametrize("set_flags", _SET_MODES)
def test_escalation_falls_back_loudly_when_timelimit_unparseable(tmp_path,
                                                                 set_flags):
    """scontrol failing or reporting a non-duration TimeLimit falls back
    to the header default (no --time) with a LOUD log line — never a
    silently dropped resubmit. Under `set -euo pipefail` the bare
    `time_limit=$(scontrol ...)` assignment used to kill the handler on a
    transient scontrol failure BEFORE the warn-and-resubmit path (finding
    2b, 2026-08-15); the pipeline now lives in an explicit conditional."""
    cases = {"scontrol-fails": None,
             "unlimited": "JobId=12345 TimeLimit=UNLIMITED"}
    for name, payload in cases.items():
        sub = tmp_path / name
        sub.mkdir()
        argv, out = _fire_escalation(sub, payload, set_flags=set_flags)
        assert not any(a.startswith("--time") for a in argv), name
        assert argv[-3:] == [str(sub / "job.sh"), "2", "5"], name  # still resubmits
        assert "WARNING" in out and "TimeLimit" in out, name


def _fire_on_term(tmp_path: Path, *, stop: bool = False,
                  drain_timeout: int = 1,
                  set_flags: str = "-uo pipefail"):
    """Source stage2_common.sh, start a TERM-immune fake workload, and
    fire s2_on_term exactly as a preempted job would. Returns (sbatch
    argv, combined log, exit code, elapsed seconds)."""
    import time as _time

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    calls = tmp_path / "sbatch_argv"
    (bin_dir / "sbatch").write_text(
        f"#!/bin/bash\nprintf '%s\\n' \"$@\" > {shlex.quote(str(calls))}\n")
    (bin_dir / "scontrol").write_text(
        "#!/bin/bash\necho 'JobId=12345 TimeLimit=01:00:00'\n")
    for name in ("sbatch", "scontrol"):
        (bin_dir / name).chmod(0o755)
    job_script = tmp_path / "job.sh"
    job_script.write_text("#!/bin/bash\n")
    stop_dir = tmp_path / "stop"
    stop_dir.mkdir()
    if stop:
        (stop_dir / "STOP").touch()
    common = REPO / "scripts" / "sbatch" / "stage2_common.sh"
    harness = f"""
set {set_flags}
source {shlex.quote(str(common))}
ATTEMPT=1; MAX_ATTEMPTS=5; ESCALATE_QOS=high; RUN_NAME=testrun
STOP_DIR={shlex.quote(str(stop_dir))}
SCRIPT_PATH={shlex.quote(str(job_script))}
_s2_handled=0
S2_DRAIN_TIMEOUT={drain_timeout}
bash -c 'trap "" TERM; touch {shlex.quote(str(stop_dir))}/armed; sleep 20' >/dev/null 2>&1 &
SRUN_PID=$!
# Wait until the child's TERM-immunity trap is INSTALLED before firing:
# without this, a loaded machine can deliver the TERM pre-trap, the child
# dies instantly, the drain never times out, and the "drain timeout" log
# assertion flakes (observed ~1-in-3 under load).
for _ in $(seq 1 100); do [ -e {shlex.quote(str(stop_dir))}/armed ] && break; sleep 0.05; done
s2_on_term
"""
    env = dict(os.environ, PATH=f"{bin_dir}:{os.environ['PATH']}",
               SLURM_JOB_ID="12345")
    t0 = _time.monotonic()
    res = subprocess.run(["bash", "-c", harness], env=env,
                         capture_output=True, text=True)
    elapsed = _time.monotonic() - t0
    argv = calls.read_text().splitlines() if calls.exists() else []
    return argv, res.stdout + res.stderr, res.returncode, elapsed


@pytest.mark.parametrize("set_flags", _SET_MODES)
def test_on_term_escalates_first_then_bounded_drain(tmp_path, set_flags):
    """On SIGTERM the resubmit fires IMMEDIATELY after signalling
    the workload, and the drain wait is BOUNDED — a workload that ignores
    TERM (here: 20s TERM-immune sleep) can no longer eat the whole grace
    window before the requeue exists. Runs under both set modes: the
    normal preemption path must hold under a caller's errexit too."""
    argv, out, rc, elapsed = _fire_on_term(tmp_path, drain_timeout=1,
                                           set_flags=set_flags)
    assert rc == 143
    assert argv[-3:] == [str(tmp_path / "job.sh"), "2", "5"]  # resubmitted
    assert "--qos=high" in argv
    # escalation logged BEFORE the drain timeout: resubmit-first ordering
    assert out.index("resubmitting at qos") < out.index("drain timeout")
    # bounded: nowhere near the workload's 20s lifetime
    assert elapsed < 15


@pytest.mark.parametrize("set_flags", _SET_MODES)
def test_on_term_stop_sentinel_still_blocks_resubmission(tmp_path,
                                                         set_flags):
    """The STOP-sentinel semantics survive the escalate-first reorder: a
    deliberate stop drains without resubmitting."""
    argv, out, rc, elapsed = _fire_on_term(tmp_path, stop=True,
                                           drain_timeout=1,
                                           set_flags=set_flags)
    assert rc == 143
    assert argv == []                      # sbatch never called
    assert "STOP sentinel present" in out
    assert elapsed < 15


def _fire_term_after_completion(tmp_path: Path, set_flags: str):
    """Run a workload to COMPLETION through the real s2_run (stubbed srun
    execs it), then fire s2_on_term — the TERM-between-cells case of the
    multi-run scripts (stage2_drift/composition/longctx/pairs). Returns
    (sbatch argv, combined log, exit code)."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    calls = tmp_path / "sbatch_argv"
    (bin_dir / "sbatch").write_text(
        f"#!/bin/bash\nprintf '%s\\n' \"$@\" > {shlex.quote(str(calls))}\n")
    (bin_dir / "scontrol").write_text(
        "#!/bin/bash\necho 'JobId=12345 TimeLimit=01:00:00'\n")
    (bin_dir / "srun").write_text('#!/bin/bash\nexec "$@"\n')
    for name in ("sbatch", "scontrol", "srun"):
        (bin_dir / name).chmod(0o755)
    job_script = tmp_path / "job.sh"
    job_script.write_text("#!/bin/bash\n")
    stop_dir = tmp_path / "stop"
    stop_dir.mkdir()
    common = REPO / "scripts" / "sbatch" / "stage2_common.sh"
    harness = f"""
set {set_flags}
source {shlex.quote(str(common))}
ATTEMPT=1; MAX_ATTEMPTS=5; ESCALATE_QOS=high; RUN_NAME=testrun
STOP_DIR={shlex.quote(str(stop_dir))}
SCRIPT_PATH={shlex.quote(str(job_script))}
_s2_handled=0
S2_DRAIN_TIMEOUT=1
SRUN_PID=""
s2_run true
s2_on_term
"""
    env = dict(os.environ, PATH=f"{bin_dir}:{os.environ['PATH']}",
               SLURM_JOB_ID="12345")
    res = subprocess.run(["bash", "-c", harness], env=env,
                         capture_output=True, text=True)
    argv = calls.read_text().splitlines() if calls.exists() else []
    return argv, res.stdout + res.stderr, res.returncode


@pytest.mark.parametrize("set_flags", _SET_MODES)
def test_on_term_after_workload_completion_still_escalates(tmp_path,
                                                           set_flags):
    """A TERM that lands AFTER s2_run reaped the workload (between the
    cells of a multi-run script) must still resubmit. s2_run never cleared
    SRUN_PID, so the handler's `kill` hit an already-reaped PID and — as
    the tail of an && list under a caller's `set -e` — aborted the handler
    BEFORE s2_escalate ever ran (finding 2a, 2026-08-15). s2_run now
    clears SRUN_PID once the workload is reaped, and the handler's kill
    tolerates a dead process."""
    argv, out, rc = _fire_term_after_completion(tmp_path, set_flags)
    assert rc == 143, out
    assert argv[-3:] == [str(tmp_path / "job.sh"), "2", "5"], out
    assert "--qos=high" in argv
    assert "workload exited rc=0" in out       # s2_run ran to completion
    assert "resubmitting at qos" in out


def test_attn_implementation_is_backed_by_the_lockfile():
    """flash_attention_2 in a config with no flash-attn wheel in uv.lock
    means transformers ImportErrors at model load — and s2_init's
    `uv sync --frozen` prunes hand-installs, so the whole suite is inert.
    sdpa (the fail-safe default) needs no extra wheel; upgrading to FA2 is
    a C0 decision that FIRST adds the pinned wheel to the lockfile."""
    lock = (REPO / "uv.lock").read_text()
    for cfg_path in sorted((REPO / "configs").glob("*.yaml")):
        impl = (yaml.safe_load(cfg_path.read_text()) or {}).get(
            "attn_implementation")
        if impl == "flash_attention_2":
            assert 'name = "flash-attn"' in lock, (
                f"{cfg_path.name} pins flash_attention_2 but uv.lock has no "
                "flash-attn wheel — the suite would ImportError at model load"
            )


def test_stage2_common_translates_uv_extras_into_extra_flags():
    """The FA2 wheel ships behind the OPTIONAL `longctx` extra (a
    source-built torch-2.11 wheel in the gitignored wheels/ dir — no PyPI
    wheel exists for torch 2.11). A job requests an extra via S2_UV_EXTRAS
    BEFORE s2_init, and stage2_common.sh translates that into `--extra`
    flags on the frozen sync. Core jobs set nothing and sync bare, so they
    keep working with the wheel absent."""
    sh = (REPO / "scripts" / "sbatch" / "stage2_common.sh").read_text()
    assert 'for _extra in ${S2_UV_EXTRAS:-}' in sh
    assert 'uv_extra_flags+=(--extra "$_extra")' in sh
    # torch-cpu-split: the frozen sync always pins the GPU flavor
    assert 'uv sync --frozen --extra cu128 "${uv_extra_flags[@]}"' in sh
    # the extra exists in pyproject and points at the pinned wheel source
    py = (REPO / "pyproject.toml").read_text()
    assert "flash-attn==2.8.3.post1" in py
    assert ('flash-attn = { path = "wheels/flash_attn-2.8.3.post1'
            '-cp312-cp312-linux_x86_64.whl" }') in py
    # no sbatch script requests extras through s2_init: the alphasteer-full22
    # fit derives its extras from the TARGET CONFIG (`env_extras: mamba` on
    # the Mamba-hybrid target only; never hardcoded, so the other targets
    # keep syncing bare — test_fit_class_b pins the derivation);
    # mamba_build.sbatch warms uv's cache with its own direct `--extra mamba`
    # sync and sets no s2_init extras variable
    f22 = (REPO / "scripts" / "sbatch" / "alphasteer_fit_f22.sbatch").read_text()
    assert "as22_env_extras" in f22 and "S2_UV_EXTRAS=mamba" not in f22
    for other in sorted((REPO / "scripts" / "sbatch").glob("*.sbatch")):
        if other.name in ("fa_build.sbatch", "alphasteer_fit_f22.sbatch"):
            continue
        assert "S2_UV_EXTRAS" not in other.read_text(), other.name


def _fake_cuda_toolkit(root, release):
    """A toolkit tree whose bin/nvcc answers `--version` like the real one."""
    (root / "bin").mkdir(parents=True, exist_ok=True)
    nvcc = root / "bin" / "nvcc"
    nvcc.write_text("#!/bin/sh\necho 'nvcc: NVIDIA (R) Cuda compiler driver'\n"
                    f"echo 'Cuda compilation tools, release {release}, V{release}.93'\n")
    nvcc.chmod(0o755)
    return root


def test_mamba_extra_sync_sets_the_kernel_build_env_before_the_frozen_sync(tmp_path):
    """The `mamba` extra is the one extra uv BUILDS FROM SOURCE (pinned PyPI
    sdists compiled against the lockfile's torch; 2026-10-12). s2_init must
    therefore select a VALIDATED CUDA 12 toolkit through the shared
    scripts/sbatch/cuda_toolkit.sh (export CUDA_HOME, its bin first on PATH --
    compute nodes ship nvcc OFF PATH) and bound MAX_JOBS BEFORE the frozen
    sync, only when `mamba` is among the requested extras (every other job's
    sync stays byte-identical), honouring a caller's own values when they
    validate -- and the sync must read the SHARED UV_CACHE_DIR so a build
    cached by mamba_build.sbatch is reused rather than recompiled. The
    2026-10-12 review's finding is pinned: the old unconditional
    `CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}` named cluster B's CUDA 13
    toolkit, which torch refuses against the cu128 torch, so a cold-cache fit
    died inside uv's build; now a node with no CUDA 12 toolkit is FATAL before
    the sync, with the remedy named, and the warm-up's $SHARED/cuda-12.8
    install is a candidate. Run for real under bash: the case block lifted
    from the script, S2_REPO_DIR pointed at a copy of scripts/sbatch whose
    helper has its absolute /usr/local probes repointed at tmp fakes."""
    sh = (REPO / "scripts" / "sbatch" / "stage2_common.sh").read_text()
    code = "\n".join(l for l in sh.splitlines() if l.strip() and not l.lstrip().startswith("#"))
    assert 'export UV_CACHE_DIR="$SHARED/.cache/uv"' in code
    case = re.search(r'  case " \$\{S2_UV_EXTRAS:-\} " in\n.*?\n  esac\n', sh, re.S).group(0)
    assert '*" mamba "*)' in case
    assert 'source "$S2_REPO_DIR/scripts/sbatch/cuda_toolkit.sh"' in case
    assert "if ! s2_cuda12_select; then" in case and "FATAL" in case and "exit 1" in case
    assert "mamba_build.sbatch" in case                        # the remedy is named
    assert "/usr/local/cuda" not in case and "export CUDA_HOME=" not in case
    assert re.search(r'export MAX_JOBS="\$\{MAX_JOBS:-', case)
    sync = 'uv sync --frozen --extra cu128 "${uv_extra_flags[@]}"'
    assert code.index('export UV_CACHE_DIR="$SHARED/.cache/uv"') < code.index(case.strip().splitlines()[0]) < code.index(sync)

    # a repo-shaped copy whose helper probes tmp paths instead of /usr/local
    repo = tmp_path / "repo"
    (repo / "scripts" / "sbatch").mkdir(parents=True)
    helper = (REPO / "scripts" / "sbatch" / "cuda_toolkit.sh").read_text()
    usr_local = tmp_path / "usr" / "local"
    (repo / "scripts" / "sbatch" / "cuda_toolkit.sh").write_text(
        helper.replace("/usr/local/cuda", f"{usr_local}/cuda"))
    shared = tmp_path / "shared"
    shared.mkdir()
    prog = ("s2_log() { echo \"LOG:$*\" >&2; }\n" + case +
            'printf "%s|%s|%s" "${CUDA_HOME:-unset}" "${MAX_JOBS:-unset}" "$PATH"')
    env = {k: v for k, v in os.environ.items() if k not in ("CUDA_HOME", "MAX_JOBS", "SLURM_CPUS_PER_TASK")}
    env.update(S2_REPO_DIR=str(repo), SHARED=str(shared), UV_CACHE_DIR=str(shared / ".cache" / "uv"))

    def run(extras, **extra_env):
        e = dict(env, **extra_env)
        if extras is not None:
            e["S2_UV_EXTRAS"] = extras
        else:
            e.pop("S2_UV_EXTRAS", None)
        r = subprocess.run(["bash", "-c", prog], capture_output=True, text=True, env=e)
        return r.returncode, r.stdout.split("|") if r.returncode == 0 else r.stderr

    # no extras / longctx only: untouched, and no toolkit is even looked for
    assert run(None)[1][:2] == ["unset", "unset"]
    assert run("longctx")[1][:2] == ["unset", "unset"]
    # Cluster A: /usr/local/cuda-12.8 -- selected, nvcc's dir first on PATH,
    # MAX_JOBS bounded by the allocation and capped at 8
    _fake_cuda_toolkit(usr_local / "cuda-12.8", "12.8")
    rc, (cuda, jobs, path) = run("mamba", SLURM_CPUS_PER_TASK="16")
    assert rc == 0 and (cuda, jobs) == (f"{usr_local}/cuda-12.8", "8") and path.startswith(f"{usr_local}/cuda-12.8/bin:")
    assert run("mamba", SLURM_CPUS_PER_TASK="4")[1][1] == "4"
    assert run("longctx mamba")[1][1] == "8"                   # with no SLURM_CPUS_PER_TASK: the cap
    # a caller's own values win when the toolkit validates
    own = _fake_cuda_toolkit(tmp_path / "opt" / "cuda-12.6", "12.6")
    rc, (cuda, jobs, path) = run("mamba", CUDA_HOME=str(own), MAX_JOBS="3")
    assert rc == 0 and (cuda, jobs) == (str(own), "3") and path.startswith(f"{own}/bin:")
    # Cluster B: /usr/local/cuda is 13.0 and nothing else exists -> FATAL
    # before the sync, naming the remedy (the review's scenario)
    shutil.rmtree(usr_local / "cuda-12.8")
    _fake_cuda_toolkit(usr_local / "cuda-13.0", "13.0")
    (usr_local / "cuda").symlink_to(usr_local / "cuda-13.0")
    rc, err = run("mamba")
    assert rc == 1
    assert "LOG:FATAL" in err and "mamba_build.sbatch" in err and "release 13.0" in err
    # ... and once the warm-up has installed $SHARED/cuda-12.8 there, that is selected
    _fake_cuda_toolkit(shared / "cuda-12.8", "12.8")
    rc, (cuda, jobs, path) = run("mamba")
    assert rc == 0 and cuda == str(shared / "cuda-12.8") and path.startswith(f"{shared}/cuda-12.8/bin:")
    # a caller's CUDA_HOME that is NOT CUDA 12 is refused, never silently replaced
    rc, err = run("mamba", CUDA_HOME=str(usr_local / "cuda-13.0"))
    assert rc == 1 and "refusing" in err and "LOG:FATAL" in err


def test_load_model_falls_back_to_sdpa_when_fa2_is_unavailable(monkeypatch,
                                                               capsys):
    """A config that pins flash_attention_2 needs the wheel behind
    the optional `longctx` extra — an env synced WITHOUT it (a core-job
    sync, a lost wheel) makes transformers raise ImportError at model
    load. _load_model must degrade to sdpa with a LOUD warning, never
    kill a multi-hour suite over a perf knob."""
    import sisteer.models as sm

    attempts = []

    def fake_load(name, **kwargs):
        attempts.append(kwargs.get("attn_implementation"))
        if kwargs.get("attn_implementation") == "flash_attention_2":
            raise ImportError(
                "FlashAttention2 has been toggled on, but it cannot be "
                "used because the package flash_attn seems to be not "
                "installed")
        return "MODEL", "TOK"

    monkeypatch.setattr(sm, "load_model_and_tokenizer", fake_load)
    model, tok = stage2_run._load_model(
        {"model": "m", "attn_implementation": "flash_attention_2"})
    assert (model, tok) == ("MODEL", "TOK")
    assert attempts == ["flash_attention_2", "sdpa"]
    out = capsys.readouterr().out
    assert "falling back to sdpa" in out
    assert "--extra longctx" in out       # the warning names the fix


def test_load_model_fallback_is_fa2_specific(monkeypatch):
    """An ImportError under any OTHER attn implementation is real env
    breakage, not a missing optional wheel — it must stay fatal (fail
    loudly, never mask an unrelated broken install behind a retry)."""
    import sisteer.models as sm

    def fake_load(name, **kwargs):
        raise ImportError("libsomething.so: cannot open shared object file")

    monkeypatch.setattr(sm, "load_model_and_tokenizer", fake_load)
    with pytest.raises(ImportError, match="libsomething"):
        stage2_run._load_model({"model": "m", "attn_implementation": "sdpa"})





def test_frontier_sbatch_passes_the_experiment_config():
    """stage2_frontier consumes the experiment config's model + pinned
    model_revision + dtype through the same loader stage2_run uses,
    so the MMLU fill-in runs the exact snapshot/precision the matrix cells
    trained on. The sbatch passes --config; the config carries the pins."""
    sb = (REPO / "scripts" / "sbatch" / "stage2_frontier.sbatch").read_text()
    assert "--config configs/stage2_llama8b.yaml" in sb
    cfg = yaml.safe_load((REPO / "configs" / "stage2_llama8b.yaml").read_text())
    assert cfg["model_revision"] and cfg["dtype"]  # the pins the script consumes


def test_frontier_requires_a_model_source_when_not_dry(tmp_path, monkeypatch):
    """Without --config or --model there is no silent default model: the
    MMLU fill-in must run the matrix's pinned snapshot."""
    _fake_cell(tmp_path, "json_format", "vector",
               steered=[{"arm": "steered", "scale": 1.0,
                         "compliance_strict": 0.7, "compliance_loose": 0.7}],
               cap_scales={1.0: 0.6})
    monkeypatch.setattr(sys, "argv",
                        ["stage2_frontier.py", "--out-dir", str(tmp_path)])
    with pytest.raises(SystemExit, match="--config"):
        stage2_frontier.main()


# ------------------------------------------------------ per-arm factor grids
# The 1/n_layers = 1/32 point on the three _all arms, and the incremental
# factor extension that makes a grid change ADD to completed cells instead
# of skipping the new factor forever (old behavior) or retraining/
# recomputing 30 settled cells.


def test_arm_scales_per_arm_additive_override_decode():
    """_arm_scales: global grid alone passes through untouched (order
    preserved); an arm's extra_scales entry merges ADDITIVELY (dedup,
    ascending); other arms never widen."""
    cfg = {"scales": [0.25, 0.5, 1.0, 2.0, 4.0, 8.0],
           "extra_scales": {"vector_all": [0.03125]}}
    assert stage2_run._arm_scales(cfg, "vector_all") == [
        0.03125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
    assert stage2_run._arm_scales(cfg, "vector") == [
        0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
    # additive semantics: an extra that duplicates a base point is a no-op
    cfg2 = {"scales": [1.0, 2.0], "extra_scales": {"a": [2.0]}}
    assert stage2_run._arm_scales(cfg2, "a") == [1.0, 2.0]
    # no config keys at all -> the [1.0] fallback the driver always used
    assert stage2_run._arm_scales({}, "vector") == [1.0]


def test_load_cfg_expands_site_paths_and_leaves_the_wandb_project_to_the_logger(tmp_path, monkeypatch):
    """A placeholder-form stage-2 config: out_dir, greedy_results_root and padding.filler_cache expand through the site
    module (~ included) and an unset root refuses by name, while wandb_project stays as written, because the W&B logger
    expands it and turns W&B off by name when it has no value. The shipped config loads on a host with no W&B project."""
    from sisteer import site as SITE
    for v in ("SISTEER_SHARED_ROOT", "CLUSTER_SHARED_ROOT", "SISTEER_CONFIG", "SISTEER_CLUSTER_LABEL",
              "WANDB_PROJECT", "SISTEER_WANDB_PROJECT"):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setenv("SISTEER_SITE_DIR", str(tmp_path / "no-site"))
    cfg_path = tmp_path / "s2.yaml"
    cfg_path.write_text(yaml.safe_dump({
        "out_dir": "$SHARED/out/s2", "greedy_results_root": "~/greedy",
        "padding": {"filler_cache": "${SHARED}/data/filler.json"}, "wandb_project": "$WANDB_PROJECT"}))
    with pytest.raises(SITE.PlaceholderError, match=r"\$SHARED in .*\(out_dir\)"):
        stage2_run._load_cfg(str(cfg_path))
    monkeypatch.setenv("SISTEER_SHARED_ROOT", str(tmp_path / "shared"))
    cfg = stage2_run._load_cfg(str(cfg_path))
    assert cfg["out_dir"] == str(tmp_path / "shared" / "out" / "s2")
    assert cfg["greedy_results_root"] == os.path.expanduser("~/greedy")
    assert cfg["padding"]["filler_cache"] == str(tmp_path / "shared" / "data" / "filler.json")
    assert cfg["wandb_project"] == "$WANDB_PROJECT"
    shipped = stage2_run._load_cfg(str(REPO / "configs" / "stage2_llama8b.yaml"))
    assert shipped["out_dir"] == "out/stage2_llama8b/ifeval" and shipped["wandb_project"] == "$WANDB_PROJECT"


def test_config_pins_the_1_over_32_point_on_exactly_the_all_arms():
    """The shipped matrix config: extra_scales names EXACTLY the three
    _all arms, the value is exactly 1/32 = 1/n_layers (Llama-3.1-8B, 32
    layers)."""
    cfg = stage2_run._load_cfg(str(REPO / "configs" / "stage2_llama8b.yaml"))
    assert set(cfg["extra_scales"]) == {"vector_all", "lora_r1_all",
                                        "psr_kl_all"}
    for arm in cfg["extra_scales"]:
        assert cfg["extra_scales"][arm] == [0.03125]
        assert cfg["extra_scales"][arm][0] == 1 / 32   # exact in binary
        assert stage2_run._arm_scales(cfg, arm)[0] == 0.03125
    # single-layer arms keep the 6-point grid exactly
    assert stage2_run._arm_scales(cfg, "vector") == cfg["scales"]


def test_factor_grid_is_outside_the_manifest_config_hash():
    """The manifest identity check must NOT fire on a grid change: scales /
    extra_scales are eval-only knobs, deliberately outside
    _cell_config_hash's subset — that is what makes the incremental
    extension legal against existing manifests, with NO carve-out or
    weakening of the check itself."""
    cfg = _tiny_cell_cfg(Path("/tmp"))
    h0 = stage2_run._cell_config_hash(cfg, "vector")
    h1 = stage2_run._cell_config_hash(
        cfg | {"scales": [0.03125, 1.0],
               "extra_scales": {"vector": [0.03125]}}, "vector")
    assert h0 == h1
    # ... while a REAL identity field still changes it
    assert stage2_run._cell_config_hash(cfg | {"layer": 2}, "vector") != h0


def _completed_tiny_cell(model, tok, tmp_path, scales=None):
    """A completed json/vector cell at the base grid (helper for the
    extension tests): P1 pairs + full cell run."""
    from sisteer.data.prompts import seed_prompts, split_prompts

    cfg = _tiny_cell_cfg(tmp_path)
    if scales is not None:
        cfg["scales"] = scales
    fit, _ = split_prompts(seed_prompts("general"), holdout_frac=0.5)
    stage2_run._ensure_pairs(model, tok, resolve_si("json"), fit[:2],
                             tmp_path / "json" / "pairs.jsonl", 4)
    summary = stage2_run._run_ifeval_cell(model, tok, cfg, "json",
                                          "vector", tmp_path)
    assert "skipped" not in summary
    return cfg


def test_ifeval_cell_incremental_factor_extension(tiny_llama, tmp_path,
                                                  monkeypatch):
    """The resume behavior on a grid change, end to end: a completed cell
    plus a new extra_scales factor (1) NEVER retrains (cost guard: the
    registry train is poisoned and the artifact bytes must not move),
    (2) appends steered+both at the new factor only, (3) preserves every
    existing entry verbatim, records aligned, (4) files anti_steerable for
    the new factor against the RECORDED baseline, and (5) is idempotent —
    the next run skips."""
    model, tok = tiny_llama
    cfg = _completed_tiny_cell(model, tok, tmp_path)
    cell = tmp_path / "json" / "vector"
    artifact = cell / "vector.safetensors"
    before = json.loads((cell / "results.json").read_text())
    art_bytes = artifact.read_bytes()

    # cost guard: any training attempt during the extension is a failure
    monkeypatch.setitem(stage2_run.ARMS, "vector", SimpleNamespace(
        train=lambda *a, **kw: pytest.fail(
            "incremental extension must NEVER retrain")))

    ext_cfg = cfg | {"extra_scales": {"vector": [0.5]}}
    out = stage2_run._run_ifeval_cell(model, tok, ext_cfg, "json",
                                      "vector", tmp_path)
    assert out.get("extended") is True
    assert out["added_scales"] == [0.5]
    assert artifact.read_bytes() == art_bytes          # artifact untouched

    after = json.loads((cell / "results.json").read_text())
    # every pre-existing arm entry and record list preserved verbatim
    n = len(before["arms"])
    assert after["arms"][:n] == before["arms"]
    assert after["records"][:n] == before["records"]
    assert len(after["records"]) == len(after["arms"])
    added = after["arms"][n:]
    assert [(a["arm"], a["scale"]) for a in added] == [
        ("steered", 0.5), ("both", 0.5)]
    for rec_list in after["records"][n:]:
        assert rec_list and {"prompt", "response", "strict"} <= set(
            rec_list[0])
    # anti_steerable: old keys intact, the new factor filed
    assert set(before["anti_steerable"]) <= set(after["anti_steerable"])
    assert "0.5" in after["anti_steerable"]
    # provenance of the extension, loudly recorded in the results file
    (note,) = after["incremental_extensions"]
    assert note["added_scales"] == [0.5]

    # idempotent: the factor is present now, so the next run skips
    again = stage2_run._run_ifeval_cell(model, tok, ext_cfg, "json",
                                        "vector", tmp_path)
    assert again.get("skipped") is True
    assert json.loads((cell / "results.json").read_text()) == after


def test_incremental_extension_measures_capability_at_new_factor(
    tiny_llama, tmp_path, monkeypatch
):
    """Frontier admission needs MMLU (+ the extended panel) AT the new
    factor: the extension measures the panel at the missing factors only
    (include_unsteered=False — the unsteered reference is already in the
    cell) and merges under the existing key conventions
    (capability.scale_<s>.mmlu + capability_ext['<s>']), preserving
    unsteered/best_scale and every existing panel entry."""
    import sisteer.eval.capability_ext as capx_mod

    model, tok = tiny_llama
    cfg = _completed_tiny_cell(model, tok, tmp_path)
    cell = tmp_path / "json" / "vector"
    # seed a pre-existing capability block (the tiny cfg runs capability
    # off; the merge must preserve whatever the real cells stored)
    results = json.loads((cell / "results.json").read_text())
    results["capability"] = {"unsteered": {"mmlu": 0.7}, "best_scale": 1.0,
                             "scale_1.0": {"mmlu": 0.69}}
    results["capability_ext"] = {"0.0": {"mmlu": 0.7}, "1.0": {"mmlu": 0.69}}
    (cell / "results.json").write_text(json.dumps(results))

    calls = {}

    def fake_panel(model_, tok_, obj_, scales, *, include_unsteered=True,
                   **kw):
        calls["scales"] = list(scales)
        calls["include_unsteered"] = include_unsteered
        assert obj_ is not None
        return {s: {"mmlu": 0.65, "gsm8k": 0.5,
                    "gsm8k_extraction_failure_rate": 0.0,
                    "gpqa_diamond": 0.3, "mmlu_pro": 0.4} for s in scales}

    monkeypatch.setattr(capx_mod, "capability_panel", fake_panel)
    ext_cfg = cfg | {"capability": True,
                     "extra_scales": {"vector": [0.03125]}}
    out = stage2_run._run_ifeval_cell(model, tok, ext_cfg, "json",
                                      "vector", tmp_path)
    assert out.get("extended") is True
    assert calls == {"scales": [0.03125], "include_unsteered": False}
    after = json.loads((cell / "results.json").read_text())
    cap = after["capability"]
    assert cap["unsteered"] == {"mmlu": 0.7}          # preserved
    assert cap["best_scale"] == 1.0                   # the ORIGINAL selection
    assert cap["scale_1.0"] == {"mmlu": 0.69}         # preserved
    assert cap["scale_0.03125"] == {"mmlu": 0.65}     # the new factor
    assert after["capability_ext"]["0.03125"]["gsm8k"] == 0.5
    assert after["capability_ext"]["1.0"] == {"mmlu": 0.69}


def test_incremental_extension_missing_artifact_fails_loudly(
    tiny_llama, tmp_path
):
    """results.json without its trained artifact cannot be extended —
    retraining would evaluate a DIFFERENT object than the recorded
    entries. Loud SystemExit, never a silent retrain."""
    model, tok = tiny_llama
    cfg = _completed_tiny_cell(model, tok, tmp_path)
    (tmp_path / "json" / "vector" / "vector.safetensors").unlink()
    ext_cfg = cfg | {"extra_scales": {"vector": [0.5]}}
    with pytest.raises(SystemExit, match="RETRAIN"):
        stage2_run._run_ifeval_cell(model, tok, ext_cfg, "json",
                                    "vector", tmp_path)


def test_incremental_extension_keeps_manifest_identity_checks(
    tiny_llama, tmp_path
):
    """The extension runs the SAME manifest check as a resume: a
    real identity change (layer) still hard-fails — the grid being outside
    the hash is what makes extension legal, not a weakened check."""
    model, tok = tiny_llama
    cfg = _completed_tiny_cell(model, tok, tmp_path)
    ext_cfg = cfg | {"layer": 2, "extra_scales": {"vector": [0.5]}}
    with pytest.raises(SystemExit, match="config_hash"):
        stage2_run._run_ifeval_cell(model, tok, ext_cfg, "json",
                                    "vector", tmp_path)


def test_incremental_extension_refuses_rejection_teacher(
    tiny_llama, tmp_path
):
    """The extension path supports the greedy tree only (RS manifests
    carry teacher config_extra it does not reconstruct); an RS-tree grid
    change fails loudly instead of merging under the wrong provenance."""
    model, tok = tiny_llama
    cfg = _completed_tiny_cell(model, tok, tmp_path)
    ext_cfg = cfg | {"extra_scales": {"vector": [0.5]}}
    with pytest.raises(SystemExit, match="greedy"):
        stage2_run._run_ifeval_cell(model, tok, ext_cfg, "json",
                                    "vector", tmp_path, teacher="rejection")


def test_fresh_cell_stacking_scale_pinned_to_base_grid(tiny_llama, tmp_path):
    """A FRESH cell whose arm carries extras: the steered sweep runs the
    extended grid, but the stacking ('both') point stays the BASE grid's
    median — an added factor must never move the stack point out from
    under the rest of the matrix (the extended median here would be 0.5,
    not the base median 1.0)."""
    from sisteer.data.prompts import seed_prompts, split_prompts

    model, tok = tiny_llama
    cfg = _tiny_cell_cfg(tmp_path)
    cfg["scales"] = [0.5, 1.0]
    cfg["extra_scales"] = {"vector": [0.25]}
    fit, _ = split_prompts(seed_prompts("general"), holdout_frac=0.5)
    stage2_run._ensure_pairs(model, tok, resolve_si("json"), fit[:2],
                             tmp_path / "json" / "pairs.jsonl", 4)
    summary = stage2_run._run_ifeval_cell(model, tok, cfg, "json",
                                          "vector", tmp_path)
    steered = [a["scale"] for a in summary["arms"] if a["arm"] == "steered"]
    assert steered == [0.25, 0.5, 1.0]
    (both,) = [a for a in summary["arms"] if a["arm"] == "both"]
    assert both["scale"] == 1.0


def test_s2_init_reads_the_shared_root_from_the_site_and_the_escalation_qos_from_the_job(tmp_path):
    """s2_init takes the cluster's shared root from the job's site value (SISTEER_SHARED_ROOT, which the front door passes
    from the cluster block) and the QoS a preempted job resubmits at from ESCALATE_QOS, the job's own setting: never from
    which storage paths exist, and never from the QoS the job was submitted at (SISTEER_CLUSTER_QOS). Either one missing
    stops the job by name before any work; ESCALATE_QOS=none turns escalation off."""
    for name in ("none", "no_qos", "site", "off"):
        (tmp_path / name).mkdir()
    rc, out = _run_s2_init(tmp_path / "none", scontrol_payload=_SCONTROL_WITH_PIN, site=False)
    assert rc != 0 and "set SISTEER_SHARED_ROOT" in out and "S2_INIT_OK" not in out, out
    rc, out = _run_s2_init(tmp_path / "no_qos", scontrol_payload=_SCONTROL_WITH_PIN,
                           extra_env={"ESCALATE_QOS": "", "SISTEER_CLUSTER_QOS": "qos_submit"})
    assert rc != 0 and "set ESCALATE_QOS" in out and "S2_INIT_OK" not in out and "qos_submit" not in out, out
    rc, out = _run_s2_init(tmp_path / "site", scontrol_payload=_SCONTROL_WITH_PIN)
    assert rc == 0 and "escalation qos: qos_nonpreempt" in out and "S2_INIT_OK" in out, out
    assert (tmp_path / "site" / "shared" / "u").is_dir()            # the job's own shared root, nothing else
    rc, out = _run_s2_init(tmp_path / "off", scontrol_payload=_SCONTROL_WITH_PIN, extra_env={"ESCALATE_QOS": "none"})
    assert rc == 0 and "escalation qos: none" in out, out


def test_a_job_submitted_at_a_preemptible_qos_never_resubmits_at_it(tmp_path):
    """A job submitted at a preemptible QoS (the site's submission QoS, SISTEER_CLUSTER_QOS=qos_preemptible) and then
    preempted resubmits at its ESCALATE_QOS, the non-preemptible QoS; with no ESCALATE_QOS it stops at s2_init, before
    any work, rather than resubmit at the QoS it was submitted at, which would only invite the next preemption."""
    job = tmp_path / "job.sh"
    job.write_text("#!/bin/bash\n")
    fire = f"SCRIPT_PATH={shlex.quote(str(job))}\ns2_escalate"
    for name in ("unset", "set"):
        (tmp_path / name).mkdir()
    rc, out = _run_s2_init(tmp_path / "unset", scontrol_payload=_SCONTROL_WITH_PIN, after=fire,
                           extra_env={"SISTEER_CLUSTER_QOS": "qos_preemptible", "ESCALATE_QOS": ""})
    assert rc != 0 and "set ESCALATE_QOS" in out and "S2_INIT_OK" not in out, out
    assert not (tmp_path / "unset" / "bin" / "sbatch.args").exists(), out
    rc, out = _run_s2_init(tmp_path / "set", scontrol_payload=_SCONTROL_WITH_PIN, after=fire,
                           extra_env={"SISTEER_CLUSTER_QOS": "qos_preemptible", "ESCALATE_QOS": "qos_nonpreempt"})
    assert rc == 0 and "S2_INIT_OK" in out and "resubmitting at qos=qos_nonpreempt" in out, out
    argv = (tmp_path / "set" / "bin" / "sbatch.args").read_text().splitlines()
    assert "--qos=qos_nonpreempt" in argv and "--qos=qos_preemptible" not in argv, argv
    assert argv[-3:] == [str(job), "2", "5"], argv
