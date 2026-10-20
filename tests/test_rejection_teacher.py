"""The rejection-sampling (compliance-filtered) teacher:
data/pairs.generate_compliant_pairs keeps only checker-passing pos
completions (scored by the SAME checkers.check dispatch the eval uses) with the
neg generated exactly as the greedy teacher's; the stage2_run driver's
--teacher rejection mode selects the sub-threshold SIs from the GREEDY
matrix's prompted strict, writes into the parallel ifeval_rs namespace
(never the greedy tree), and SKIPS low-yield SIs loudly. CPU-only; the
end-to-end cell runs on the tiny-llama fixture."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import stage2_run  # noqa: E402
from sisteer.data import pairs as pairs_mod  # noqa: E402
from sisteer.data.instructions import get as get_instruction  # noqa: E402
from sisteer.data.instructions import resolve_si  # noqa: E402
from sisteer.data.pairs import (  # noqa: E402
    Pair,
    generate_compliant_pairs,
    load_pairs,
    save_pairs,
)


# ------------------------------------------------- generate_compliant_pairs


def _scripted_generate(script: dict[str, list[str]], calls: list[dict]):
    """A generate_batch stand-in: sampling calls (do_sample=True) pop the
    next scripted completion for whichever prompt appears in each rendered
    string; the neg call (do_sample default False, exactly as the greedy
    teacher generates) returns UPPERCASE text — so a filtered neg would be
    caught immediately."""

    def fake(model, tok, rendered, batch_size=8, max_new_tokens=200,
             do_sample=False, temperature=1.0, **kw):
        calls.append({"n": len(rendered), "do_sample": do_sample,
                      "temperature": temperature,
                      "rendered": list(rendered)})
        if do_sample:
            out = []
            for r in rendered:
                key = next(p for p in script if p in r)
                out.append(script[key].pop(0))
            return out
        return ["GREEDY NEG (UNFILTERED)"] * len(rendered)

    return fake


def test_rejection_keeps_only_checker_passing_pos(tiny_llama, monkeypatch):
    _, tok = tiny_llama
    instr = get_instruction("lowercase")  # trivial repo checker
    prompts = ["alpha prompt", "bravo prompt", "charlie prompt"]
    script = {
        "alpha prompt": ["HAS CAPS", "all lower ok"],
        "bravo prompt": ["fine and lower"],
        "charlie prompt": ["FAIL ONE", "FAIL TWO", "FAIL THREE"],
    }
    calls: list[dict] = []
    monkeypatch.setattr(pairs_mod, "generate_batch",
                        _scripted_generate(script, calls))

    pairs, stats = generate_compliant_pairs(
        None, tok, instr, prompts, k=3, temperature=0.7, max_new_tokens=8)

    # kept pos all pass the checker; the unyielded prompt has NO pair
    assert [p.prompt for p in pairs] == ["alpha prompt", "bravo prompt"]
    assert [p.pos for p in pairs] == ["all lower ok", "fine and lower"]
    from sisteer.eval.checkers import check

    assert all(check(instr.checker, p.pos, **instr.checker_kwargs)
               for p in pairs)
    # neg is the greedy teacher's neg, UNFILTERED (uppercase kept), and
    # rendered with the anti-instruction — the pos-filter effect isolated
    assert all(p.neg == "GREEDY NEG (UNFILTERED)" for p in pairs)
    neg_calls = [c for c in calls if not c["do_sample"]]
    assert len(neg_calls) == 1 and neg_calls[0]["n"] == 2  # yielded only
    assert all(instr.anti_text in r for r in neg_calls[0]["rendered"])
    # sampling ran at the requested temperature, one round per pending set
    sample_calls = [c for c in calls if c["do_sample"]]
    assert [c["n"] for c in sample_calls] == [3, 2, 1]
    assert all(c["temperature"] == 0.7 for c in sample_calls)

    # yield_stats
    assert stats["n_prompts"] == 3
    assert stats["n_yielded"] == 2
    assert stats["yield_rate"] == pytest.approx(2 / 3)
    assert stats["mean_samples_used"] == pytest.approx((2 + 1 + 3) / 3)
    assert stats["mean_samples_to_yield"] == pytest.approx(1.5)
    assert stats["pos_compliance"] == pytest.approx(1.0)
    assert stats["unyielded"] == [{"idx": 2, "prompt": "charlie prompt"}]
    assert stats["k"] == 3 and stats["temperature"] == 0.7


def test_rejection_scores_with_the_eval_dispatch(tiny_llama, monkeypatch):
    """Samples are scored by the instruction's registered checker through
    checkers.check — the exact strict path the eval uses."""
    _, tok = tiny_llama
    instr = resolve_si("lowercase")  # the stage-1 SI, repo lowercase checker
    assert instr.checker == "lowercase"
    script = {
        "one prompt": ["This Has Capitals so it fails.",
                       "this response is entirely in english lowercase."],
    }
    monkeypatch.setattr(pairs_mod, "generate_batch",
                        _scripted_generate(script, []))
    pairs, stats = generate_compliant_pairs(
        None, tok, instr, ["one prompt"], k=2)
    assert len(pairs) == 1
    assert pairs[0].pos == "this response is entirely in english lowercase."
    assert stats["mean_samples_used"] == pytest.approx(2.0)
    assert stats["pos_compliance"] == pytest.approx(1.0)


def test_rejection_refuses_external_checkers(tiny_llama):
    _, tok = tiny_llama
    judged = resolve_si("si_constitution")  # checker="external": judged, not programmatic
    assert judged.checker == "external"
    with pytest.raises(ValueError, match="external"):
        generate_compliant_pairs(None, tok, judged, ["p"])


# ------------------------------------------------- config: extends + RS keys


def test_load_cfg_extends_shallow_merge(tmp_path):
    (tmp_path / "base.yaml").write_text(yaml.safe_dump({
        "model": "m", "layer": 15, "arms": ["vector"],
        "out_dir": "/base/out"}))
    (tmp_path / "child.yaml").write_text(yaml.safe_dump({
        "extends": "base.yaml", "out_dir": "/child/out", "teacher":
        "rejection"}))
    cfg = stage2_run._load_cfg(str(tmp_path / "child.yaml"))
    assert cfg["model"] == "m" and cfg["layer"] == 15    # inherited
    assert cfg["out_dir"] == "/child/out"                # overridden
    assert cfg["teacher"] == "rejection"                 # added
    assert "extends" not in cfg


def test_rs_params_refuses_the_greedy_namespace(tmp_path):
    cfg = {"out_dir": str(tmp_path / "ifeval"),
           "greedy_results_root": str(tmp_path / "ifeval"),
           "si_prompted_threshold": 0.9}
    with pytest.raises(SystemExit, match="OWN namespace"):
        stage2_run._rs_params(cfg)


def test_effective_teacher_conflict_fails_loudly():
    args = SimpleNamespace(teacher="greedy")
    with pytest.raises(SystemExit, match="conflicts"):
        stage2_run._effective_teacher({"teacher": "rejection"}, args)
    # agreement and absence both resolve
    assert stage2_run._effective_teacher({"teacher": "rejection"},
                                         SimpleNamespace(teacher=None)) \
        == "rejection"
    assert stage2_run._effective_teacher({}, SimpleNamespace()) == "greedy"


# ------------------------------------------------- sub-threshold SI selector


def _stub_greedy_cell(root: Path, si: str, prompted_strict: float) -> None:
    cell = root / si / "vector"
    cell.mkdir(parents=True)
    (cell / "results.json").write_text(json.dumps({
        "instruction": si,
        "arms": [
            {"arm": "baseline", "scale": 0.0, "compliance_strict": 0.0},
            {"arm": "prompted", "scale": 0.0,
             "compliance_strict": prompted_strict},
            {"arm": "steered", "scale": 1.0, "compliance_strict": 0.5},
        ],
    }))


def _rs_cfg_stub(tmp_path, instructions, **extra) -> dict:
    return {
        "instructions": instructions,
        "si_prompted_threshold": 0.9,
        "greedy_results_root": str(tmp_path / "greedy"),
        "out_dir": str(tmp_path / "rs"),
        **extra,
    }


def test_rs_target_sis_reads_prompted_strict_from_the_greedy_tree(tmp_path):
    greedy = tmp_path / "greedy"
    _stub_greedy_cell(greedy, "json_format", 0.30)
    _stub_greedy_cell(greedy, "lowercase", 0.92)
    _stub_greedy_cell(greedy, "keyword_frequency", 0.17)
    cfg = _rs_cfg_stub(tmp_path,
                       ["json_format", "lowercase", "keyword_frequency"])
    assert stage2_run.rs_target_sis(cfg) == ["json_format",
                                             "keyword_frequency"]
    # an above-threshold SI is refused as a single-cell target
    with pytest.raises(SystemExit, match="not a rejection-teacher target"):
        stage2_run._require_rs_si(cfg, "lowercase")
    stage2_run._require_rs_si(cfg, "json_format")  # sub-threshold: fine


def test_rs_target_sis_fails_loudly_on_missing_greedy_results(tmp_path):
    _stub_greedy_cell(tmp_path / "greedy", "json_format", 0.30)
    cfg = _rs_cfg_stub(tmp_path, ["json_format", "title"])
    with pytest.raises(SystemExit, match="greedy results missing"):
        stage2_run.rs_target_sis(cfg)


# ------------------------------------------------- the RS driver path


def _tiny_rs_cfg(tmp_path) -> dict:
    return {
        "model": "unused-preloaded",
        "teacher": "rejection",
        "k": 2,
        "temperature": 1.0,
        "rs_seed": 0,
        "min_yield": 0.2,
        "si_prompted_threshold": 0.9,
        "greedy_results_root": str(tmp_path / "greedy"),
        "out_dir": str(tmp_path / "rs"),
        "instructions": ["json"],
        "layer": 1,
        "scales": [1.0],
        "prompts": {"source": "seed", "set": "general"},
        "n_fit_prompts": 2,
        "n_eval_prompts": 2,
        "max_new_tokens": 4,
        "capability": False,
        "wandb_project": None,
        "train_kwargs": {"default": {"epochs": 1, "batch_size": 2,
                                     "max_len": 64}},
    }


def _publish_rs_pairs(rs_root: Path, si: str, yield_rate: float,
                      k: int = 2) -> dict:
    """As the P1-RS stage would: yield_stats.json FIRST, then pairs.jsonl."""
    stats = {
        "instruction_id": resolve_si(si).id, "k": k, "temperature": 1.0,
        "seed": 0, "n_prompts": 2, "n_yielded": round(2 * yield_rate),
        "yield_rate": yield_rate, "mean_samples_used": float(k),
        "mean_samples_to_yield": 1.0 if yield_rate else None,
        "pos_compliance": 1.0 if yield_rate else None, "unyielded": [],
    }
    (rs_root / si).mkdir(parents=True, exist_ok=True)
    stage2_run._write_json(rs_root / si / "yield_stats.json", stats)
    save_pairs([Pair(prompt="write a thing", pos="a thing", neg="",
                     instruction_id=resolve_si(si).id)] * 2,
               rs_root / si / "pairs.jsonl")
    return stats


def test_rs_cell_trains_in_ifeval_rs_and_never_touches_greedy_pairs(
    tiny_llama, tmp_path
):
    model, tok = tiny_llama
    cfg = _tiny_rs_cfg(tmp_path)
    greedy = tmp_path / "greedy"
    _stub_greedy_cell(greedy, "json", 0.30)
    # the greedy P1 pairs file the RS run must NEVER overwrite
    greedy_pairs = greedy / "json" / "pairs.jsonl"
    save_pairs([Pair(prompt="greedy prompt", pos="GREEDY POS (unfiltered)",
                     neg="n", instruction_id="json")], greedy_pairs)
    greedy_bytes = greedy_pairs.read_bytes()
    greedy_files = sorted(p for p in greedy.rglob("*") if p.is_file())

    rs_root = Path(cfg["out_dir"])
    stats = _publish_rs_pairs(rs_root, "json", yield_rate=1.0)
    summary = stage2_run._run_ifeval_cell(model, tok, cfg, "json",
                                          "vector", rs_root,
                                          teacher="rejection")

    # trained + evaluated in the RS namespace, identically to a greedy cell
    cell = rs_root / "json" / "vector"
    assert (cell / "vector.safetensors").exists()
    assert (cell / "manifest.json").exists()
    assert (cell / "results.json").exists()
    assert {a["arm"] for a in summary["arms"]} == \
        {"baseline", "prompted", "steered", "both"}
    # teacher provenance in the manifest hash (an RS artifact can never be
    # resumed as a greedy one)
    manifest = json.loads((cell / "manifest.json").read_text())
    assert manifest["config_hash"] == stage2_run._cell_config_hash(
        cfg, "vector", extra={"teacher": "rejection", "k": 2,
                              "temperature": 1.0, "rs_seed": 0})
    # yield stats + the greedy-vs-rs prompted context ride in the results
    assert summary["teacher"]["mode"] == "rejection"
    assert summary["teacher"]["yield_stats"] == stats
    assert summary["teacher"]["greedy_prompted_strict"] == pytest.approx(0.3)
    # the greedy tree is byte-identical: no new files, pairs untouched
    assert greedy_pairs.read_bytes() == greedy_bytes
    assert sorted(p for p in greedy.rglob("*") if p.is_file()) == greedy_files


def test_rs_cell_requires_the_p1_rs_pairs(tiny_llama, tmp_path):
    """A missing RS pairs file names the P1-RS stage — never a silent
    per-cell regeneration."""
    model, tok = tiny_llama
    cfg = _tiny_rs_cfg(tmp_path)
    _stub_greedy_cell(tmp_path / "greedy", "json", 0.30)
    with pytest.raises(SystemExit, match="RS pairs file missing"):
        stage2_run._run_ifeval_cell(model, tok, cfg, "json", "vector",
                                    Path(cfg["out_dir"]),
                                    teacher="rejection")


def test_rs_pairs_without_stats_is_a_torn_publish(tiny_llama, tmp_path):
    model, tok = tiny_llama
    cfg = _tiny_rs_cfg(tmp_path)
    _stub_greedy_cell(tmp_path / "greedy", "json", 0.30)
    rs_root = Path(cfg["out_dir"])
    _publish_rs_pairs(rs_root, "json", yield_rate=1.0)
    (rs_root / "json" / "yield_stats.json").unlink()
    with pytest.raises(SystemExit, match="torn publish"):
        stage2_run._run_ifeval_cell(model, tok, cfg, "json", "vector",
                                    rs_root, teacher="rejection")


def test_rs_cell_skips_low_yield_si_with_the_logged_reason(
    tiny_llama, tmp_path, capsys
):
    """The hard-SI reality, LOUD: an SI yielding <min_yield compliant
    samples at k is skipped — no training, no artifact — with the reason
    on disk and in the log. The skip is the finding."""
    model, tok = tiny_llama
    cfg = _tiny_rs_cfg(tmp_path)
    _stub_greedy_cell(tmp_path / "greedy", "json", 0.30)
    rs_root = Path(cfg["out_dir"])
    _publish_rs_pairs(rs_root, "json", yield_rate=0.1, k=16)

    summary = stage2_run._run_ifeval_cell(model, tok, cfg, "json",
                                          "vector", rs_root,
                                          teacher="rejection")
    assert summary["skipped_low_yield"] is True
    assert "rejection teacher infeasible at k=16" in summary["reason"]
    assert "10% yield" in summary["reason"]
    assert "cannot follow this instruction even under sampling" \
        in summary["reason"]
    out = capsys.readouterr().out
    assert "SKIP json" in out and "infeasible at k=16" in out
    # nothing trained, no results.json (a rerun re-checks the gate)
    cell = rs_root / "json" / "vector"
    assert not (cell / "vector.safetensors").exists()
    assert not (cell / "results.json").exists()
    skip = json.loads(
        (rs_root / "json" / "SKIPPED_LOW_YIELD.json").read_text())
    assert skip["reason"] == summary["reason"]
    assert skip["greedy_prompted_strict"] == pytest.approx(0.3)
    assert skip["yield_stats"]["yield_rate"] == pytest.approx(0.1)


def test_run_pairs_rejection_publishes_pairs_and_stats_then_resumes(
    tmp_path, monkeypatch, capsys
):
    cfg = _tiny_rs_cfg(tmp_path)
    _stub_greedy_cell(tmp_path / "greedy", "json", 0.30)
    monkeypatch.setattr(stage2_run, "_load_model",
                        lambda c, **kw: (None, None))
    fixed_pairs = [Pair(prompt="p", pos="compliant pos", neg="n",
                        instruction_id="json")]
    fixed_stats = {"instruction_id": "json", "k": 2,
                   "temperature": 1.0, "seed": 0, "n_prompts": 1,
                   "n_yielded": 1, "yield_rate": 1.0,
                   "mean_samples_used": 1.0, "mean_samples_to_yield": 1.0,
                   "pos_compliance": 1.0, "unyielded": []}
    monkeypatch.setattr(stage2_run, "generate_compliant_pairs",
                        lambda *a, **kw: (fixed_pairs, fixed_stats))
    args = SimpleNamespace(si="json", teacher="rejection")
    stage2_run.run_pairs(cfg, args)
    si_dir = Path(cfg["out_dir"]) / "json"
    assert load_pairs(si_dir / "pairs.jsonl") == fixed_pairs
    assert json.loads((si_dir / "yield_stats.json").read_text()) == \
        fixed_stats
    assert "RS pairs ready" in capsys.readouterr().out

    # resume: an existing pairs file is loaded, never regenerated
    monkeypatch.setattr(stage2_run, "generate_compliant_pairs",
                        lambda *a, **kw: pytest.fail("must not regenerate"))
    stage2_run.run_pairs(cfg, args)


def test_run_pairs_rejection_warns_on_low_yield(tmp_path, monkeypatch,
                                                capsys):
    cfg = _tiny_rs_cfg(tmp_path)
    _stub_greedy_cell(tmp_path / "greedy", "json", 0.30)
    monkeypatch.setattr(stage2_run, "_load_model",
                        lambda c, **kw: (None, None))
    low_stats = {"instruction_id": "json", "k": 2, "temperature": 1.0,
                 "seed": 0, "n_prompts": 10, "n_yielded": 1,
                 "yield_rate": 0.1, "mean_samples_used": 2.0,
                 "mean_samples_to_yield": 1.0, "pos_compliance": 1.0,
                 "unyielded": []}
    monkeypatch.setattr(stage2_run, "generate_compliant_pairs",
                        lambda *a, **kw: ([], low_stats))
    stage2_run.run_pairs(cfg, SimpleNamespace(si="json",
                                              teacher="rejection"))
    out = capsys.readouterr().out
    assert "WARNING" in out and "SKIP retraining" in out
