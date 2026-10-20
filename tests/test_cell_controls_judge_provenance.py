"""The writer change, the store-key invariance proof and the CLI seam for
``--judge-backend`` (2026-09-02).

* ``controls_summary.json``'s provenance now records ``judge_model`` AND
  ``judge_backend``. Verified before this change: no summary on disk carried
  either, while the campaign mixes claude-sonnet-5 (the Anthropic orgs) with
  x-ai/grok-4.6 (OpenRouter). The summary is an OUTPUT — adding the keys
  changes no harness_config_hash and no store key.
* The default (anthropic, claude-sonnet-5) resolved config and its
  harness_config_hash are byte-identical to the goldens computed on
  origin/main@1aa3a1e with the same synthetic inputs BEFORE this change. A
  non-default backend enters the resolved config only when named (the
  steer_positions / parity_n precedent), so a new openrouter store keys
  apart, while every legacy grok store — generated under
  ``--judge-model x-ai/grok-4.6`` before the key existed — keeps its key and
  derives its backend from the model at judge time.
* The CLI refuses a crossed ``--judge-model`` / ``--judge-backend`` pair,
  and ``judge`` refuses a store generated for another backend.
* Merging a new-provenance pass into a summary written before the keys
  existed refuses (r1 #4: two provenances in one summary is exactly the
  stale fill it forbids) — the deliberate consequence for pre-change
  summaries, stated here so nobody reads it as a bug.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

pytest.importorskip("sisteer.capeval.judging",
                    reason="the judge backend seam needs sisteer.capeval")

import cell_controls_run as run  # noqa: E402
from sisteer.capeval import judging  # noqa: E402
from sisteer.capeval.store import cell_key, harness_config_hash  # noqa: E402
from sisteer.eval import cell_controls as cc  # noqa: E402
from sisteer.eval.cell_controls import (  # noqa: E402
    controls_provenance, load_controls_summaries, store_key_token,
    write_controls_summary)

GROK = "x-ai/grok-4.6"
SONNET = "claude-sonnet-5"

# Synthetic inputs + the goldens they produced on origin/main@1aa3a1e BEFORE
# this change (computed 2026-09-02; store_key_token("none") == "none",
# intervention hash "ab"*8).
CFG = {"model": "synthetic/model-1b", "model_revision": "0123456789ab",
       "scale": 1.0, "completion_tokens": 16}
GOLDEN_DEFAULT_RESOLVED = {
    "cell_id": "none",
    "controls": ["alpaca_eval", "gsm8k", "mmlu", "or_bench_hard_1k", "xstest"],
    "driver": "cell_controls_run_v1",
    "gen": {"do_sample": False, "max_new_tokens": 16},
    "judge_model": SONNET,
    "limit": None,
    "model_id": "synthetic/model-1b",
    "model_revision": "0123456789ab",
    "order_seed": 0,
    "scale": 1.0,
    "subset_seed": None,
    "system_sha": None,
}
GOLDEN_DEFAULT_CHASH = "10d59910b2023622"
GOLDEN_DEFAULT_KEY = "none__controls__t0__a0__abababab__10d59910"
#: prompted__safety_refusal, system "system text", limit 100, parity_n=0
GOLDEN_PROMPTED_CHASH = "fe67beb6490af998"
#: judge_model x-ai/grok-4.6 through the pre-change code: no judge_backend
#: key — the key every existing grok store was generated under.
GOLDEN_LEGACY_GROK_CHASH = "572f2a61069c7ab9"


def _cell(cid="none"):
    return SimpleNamespace(id=cid, class_="baseline", si_id=None, arm=None,
                           object_path=None, stacked=False)


# --- store-key invariance ----------------------------------------------------------


def test_default_resolved_config_and_store_key_are_byte_identical_to_golden():
    r = run._resolved_config(_cell(), CFG, None, 0, SONNET)
    assert r == GOLDEN_DEFAULT_RESOLVED
    assert "judge_backend" not in r
    assert harness_config_hash(r) == GOLDEN_DEFAULT_CHASH
    assert cell_key(store_key_token("none"), "controls", 0, 0, "ab" * 8,
                    harness_config_hash(r)) == GOLDEN_DEFAULT_KEY
    # naming the default explicitly is the SAME spelling, not a sibling store
    assert run._resolved_config(_cell(), CFG, None, 0, SONNET,
                                judge_backend="anthropic") == r
    r2 = run._resolved_config(_cell("prompted__safety_refusal"), CFG,
                              "system text", 100, SONNET, parity_n=0)
    assert harness_config_hash(r2) == GOLDEN_PROMPTED_CHASH
    assert "judge_backend" not in r2


def test_legacy_grok_key_unchanged_and_a_new_openrouter_store_keys_apart():
    legacy = run._resolved_config(_cell(), CFG, None, 0, GROK)
    assert "judge_backend" not in legacy
    assert harness_config_hash(legacy) == GOLDEN_LEGACY_GROK_CHASH
    new = run._resolved_config(_cell(), CFG, None, 0, GROK,
                               judge_backend="openrouter")
    assert new["judge_backend"] == "openrouter"
    assert {k: v for k, v in new.items() if k != "judge_backend"} == legacy
    new_ch = harness_config_hash(new)
    assert new_ch not in (GOLDEN_LEGACY_GROK_CHASH, GOLDEN_DEFAULT_CHASH)
    # and a grok store never collides with a sonnet store either way — the
    # judge model alone separates them
    assert GOLDEN_LEGACY_GROK_CHASH != GOLDEN_DEFAULT_CHASH


# --- the CLI seam --------------------------------------------------------------------


def test_judge_backend_flag_is_validated_against_the_model():
    def ns(backend):
        return SimpleNamespace(judge_backend=backend)

    assert run.JUDGE_BACKENDS == judging.JUDGE_BACKENDS
    assert run.DEFAULT_JUDGE_BACKEND == judging.DEFAULT_JUDGE_BACKEND
    assert run._judge_backend(ns("anthropic"), SONNET) == "anthropic"
    assert run._judge_backend(ns(""), SONNET) == "anthropic"          # default
    assert run._judge_backend(ns("openrouter"), GROK) == "openrouter"
    assert run._judge_backend(ns("openrouter"), "some-other-judge") == "openrouter"
    with pytest.raises(SystemExit, match="requires --judge-backend openrouter"):
        run._judge_backend(ns("anthropic"), GROK)
    with pytest.raises(SystemExit, match="requires --judge-backend openrouter"):
        run._judge_backend(ns(""), GROK)              # the default is no escape
    with pytest.raises(SystemExit, match="requires --judge-backend anthropic"):
        run._judge_backend(ns("openrouter"), SONNET)
    with pytest.raises(SystemExit, match="not one of"):
        run._judge_backend(ns("together"), GROK)


def test_recorded_backend_of_a_store():
    # a legacy grok store (no key anywhere) IS an openrouter store
    assert run._recorded_judge_backend(
        {"resolved_config": {"judge_model": GROK}}) == "openrouter"
    assert run._recorded_judge_backend(
        {"resolved_config": {"judge_model": SONNET}}) == "anthropic"
    # an id naming neither family: the default
    assert run._recorded_judge_backend(
        {"resolved_config": {"judge_model": "stub-judge"}}) == "anthropic"
    assert run._recorded_judge_backend({}) == "anthropic"
    # the recorded key wins over the implication; the resolved config over
    # the top-level stamp
    assert run._recorded_judge_backend(
        {"resolved_config": {"judge_model": GROK,
                             "judge_backend": "openrouter"}}) == "openrouter"
    assert run._recorded_judge_backend(
        {"resolved_config": {"judge_model": "stub-judge"},
         "judge_backend": "openrouter"}) == "openrouter"


def test_cli_exposes_judge_backend_on_generate_and_judge(monkeypatch):
    recorded = {}
    monkeypatch.setattr(run.jailbreak_run, "_load_cfg",
                        lambda path: {"model": "synthetic/model-1b"})
    monkeypatch.setattr(run, "cmd_judge",
                        lambda args, cfg: recorded.update(judge=args))
    monkeypatch.setattr(run, "cmd_generate",
                        lambda args, cfg: recorded.update(generate=args))
    monkeypatch.setattr(sys, "argv", [
        "cell_controls_run.py", "judge", "--cell", "none",
        "--judge-model", GROK, "--judge-backend", "openrouter"])
    run.main()
    assert recorded["judge"].judge_backend == "openrouter"
    assert recorded["judge"].judge_model == GROK
    monkeypatch.setattr(sys, "argv",
                        ["cell_controls_run.py", "generate", "--cell", "none"])
    run.main()
    assert recorded["generate"].judge_backend == "anthropic"     # the default
    monkeypatch.setattr(sys, "argv", [
        "cell_controls_run.py", "judge", "--cell", "none",
        "--judge-backend", "together"])
    with pytest.raises(SystemExit):
        run.main()                                    # argparse choices


def test_judge_phase_cross_checks_the_stores_backend(tmp_path):
    """The store-level refusal: a legacy grok store judged as anthropic
    refuses at the flag; a default-backend store refuses an openrouter
    judge; a matching pair passes the cross-check (and then hits the
    serving_identity refusal the existing test pins, proving the check ran
    in order)."""
    pytest.importorskip("sisteer.capeval.store")
    from test_cell_controls import _cfg, _mini_registry
    from sisteer.eval.defense_cell import by_id

    cfg = _cfg(defenses_dir=str(tmp_path / "cells"),
               class_a_arms=["psr_kl_all"], class_b_arms=["antidote"])
    reg = _mini_registry(tmp_path)
    cell = by_id(reg)["prompted__safety_refusal"]
    token = store_key_token(cell.id)
    out_root = tmp_path / "controls"
    cell_dir = out_root / "cells" / f"{token}__controls__stub"
    cell_dir.mkdir(parents=True)
    rows = [{"task": "s2parity/mmlu_letter", "item_id": f"i{k}",
             "metric": "acc", "value": 1.0} for k in range(3)]
    (cell_dir / "rows.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n")

    def manifest(judge_model, **extra):
        (cell_dir / "manifest.json").write_text(json.dumps({
            "cell_id": cell.id, "intervention_hash": "ab" * 8,
            "harness_config_hash": "cd" * 8,
            "model_revision": cfg["model_revision"], "engine": "hf",
            "row_constants": {"sweep": "two_si_controls"},
            "resolved_config": {"judge_model": judge_model, **extra},
        }))

    def args(**over):
        base = dict(cell=cell.id, out_root=str(out_root), defenses_dir="",
                    config="unused", controls="mmlu", limit=0, concurrency=4,
                    judge_model="", judge_backend="anthropic")
        base.update(over)
        return SimpleNamespace(**base)

    manifest(GROK)                                    # a legacy grok store
    with pytest.raises(SystemExit, match="requires --judge-backend openrouter"):
        run.cmd_judge(args(judge_model=GROK), cfg)
    manifest("stub-judge")                            # a default-backend store
    with pytest.raises(SystemExit,
                       match="generated for judge_backend='anthropic'"):
        run.cmd_judge(args(judge_model="stub-judge", judge_backend="openrouter"),
                      cfg)
    manifest(GROK, judge_backend="openrouter")        # a new openrouter store
    with pytest.raises(SystemExit, match="serving_identity"):
        run.cmd_judge(args(judge_model=GROK, judge_backend="openrouter"), cfg)


# --- the writer ------------------------------------------------------------------------


def test_provenance_records_judge_model_and_backend_and_roundtrips(tmp_path):
    cell = _cell()
    prov = controls_provenance(cell, CFG, rubric_shas={"xstest": "ab12cd34" * 4},
                               judge_model=GROK, judge_backend="openrouter")
    assert prov["judge_model"] == GROK
    assert prov["judge_backend"] == "openrouter"
    out_root = tmp_path / "controls"
    stamp = run._utc_now()
    payload = {"n": 450, "n_graded": 449, "n_abstained": 1,
               "rubric_sha": "ab12cd34" * 4, "judged_at": stamp,
               "judge_api_calls": 3, "judge_cache_hits": 447}
    path = write_controls_summary(out_root, cell, {"xstest": payload}, prov)
    doc = json.loads(path.read_text())
    assert doc["provenance"]["judge_model"] == GROK
    assert doc["provenance"]["judge_backend"] == "openrouter"
    assert doc["controls"]["xstest"]["judged_at"] == stamp
    # r1 #4 validation still passes: the new keys are recorded, not checked
    got = load_controls_summaries(out_root, [cell], CFG)
    assert got["none"]["xstest"]["judge_api_calls"] == 3
    assert got["none"]["xstest"]["judge_cache_hits"] == 447
    # the default records absence, never an invented judge
    d = controls_provenance(cell, CFG)
    assert d["judge_model"] is None and d["judge_backend"] is None
    # the sonnet spelling passes the counts-only rule too
    write_controls_summary(tmp_path / "c2", cell, {"mmlu": {"acc": 0.7, "n": 200}},
                           controls_provenance(cell, CFG, judge_model=SONNET,
                                               judge_backend="anthropic"))


def test_merge_into_a_pre_change_summary_refuses(tmp_path):
    cell = _cell()
    out_root = tmp_path / "controls"
    old = controls_provenance(cell, CFG)
    old.pop("judge_model")
    old.pop("judge_backend")                          # the pre-change shape
    write_controls_summary(out_root, cell, {"mmlu": {"acc": 0.7, "n": 200}}, old)
    new = controls_provenance(cell, CFG, judge_model=GROK,
                              judge_backend="openrouter")
    with pytest.raises(ValueError, match="DIFFERENT provenance"):
        write_controls_summary(out_root, cell,
                               {"xstest": {"n": 1, "rubric_sha": "ab" * 16}},
                               new)


def test_judged_at_stamp_and_judge_ids_pass_the_counts_only_rule():
    stamp = run._utc_now()
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", stamp)
    cc._assert_counts_only({"xstest": {"judged_at": stamp,
                                       "judge_api_calls": 0,
                                       "judge_cache_hits": 12}})
    cc._assert_counts_only({"provenance": {"judge_model": GROK,
                                           "judge_backend": "openrouter"}},
                           path="")
    cc._assert_counts_only({"provenance": {"judge_model": SONNET,
                                           "judge_backend": "anthropic"}},
                           path="")
