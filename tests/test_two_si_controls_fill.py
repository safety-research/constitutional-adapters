"""The aggregate's controls-slot fill.

Offline ``cmd_aggregate`` runs (the test_two_si_run.py harness: injected
grade-fn seams, synthetic transcripts, tmp stores — CPU-only, no network, no
harmful text) against a tmp controls dir. Asserts the fill order and its
guard rails:

- slots fill ONLY from provenance-VALID summaries (a stale component raises
  — the aggregate never publishes over it)
- a slot no summary fills (``gsm8k`` here) stays the explicit sentinel in
  EVERY cell (the mechanism is control-name-generic)
- the aggregate's OWN xstest value wins over a summary's xstest payload (a
  filled slot is never overwritten)
- an absent controls dir leaves every not-yet-run slot at the sentinel
  (aggregate unchanged)
- the sanitizer (allowed_strings + write_sanitized_json) and the grep gate
  pass on the new payloads.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import two_si_run  # noqa: E402
from sisteer.eval.cell_controls import (  # noqa: E402
    controls_provenance,
    write_controls_summary,
)
from sisteer.eval.defense_cell import registry_from_cfg  # noqa: E402
from sisteer.eval.jailbreak_data import Goal  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
TWO_SI_CONFIG = REPO / "configs" / "two_si.yaml"


# -- the test_two_si_run.py offline-aggregate harness (replicated: test
# modules are not importable from each other) --------------------------------


def _goals():
    return [Goal("g0", "benign goal zero text", "illegal", "harmbench", "rev"),
            Goal("g1", "benign goal one text", "illegal", "harmbench", "rev")]


def _xs_goals():
    return [Goal("xs0", "a benign safe prompt zero", "homonyms", "xstest",
                 "rev", split="safe"),
            Goal("xs1", "a benign safe prompt one", "homonyms", "xstest",
                 "rev", split="safe")]


@pytest.fixture()
def cfg(tmp_path):
    c = yaml.safe_load(TWO_SI_CONFIG.read_text())
    c["defenses_dir"] = str(tmp_path / "defenses" / "cells")
    c["out_dir"] = str(tmp_path / "out")
    c["pool_manifest"] = str(tmp_path / "pool_manifest.json")
    c["private_run"] = "two_si_test"
    c["bon"] = {**c["bon"], "n_max": 2, "max_new_tokens": 8}
    # hermetic controls dir (the shipped default <out_dir>/controls would be
    # tmp-rooted here anyway; explicit for clarity)
    c["controls_dir"] = str(tmp_path / "controls")
    Path(c["pool_manifest"]).write_text(json.dumps(
        {"regime": "matched_intersection",
         "pool_sizes": {si: 657 for si in c["si_ids"]}}))
    return c


def _registry(cfg):
    return registry_from_cfg(cfg, repo_root=REPO)


def _store(cfg, root, registry):
    for cell in registry:
        d = Path(root) / cfg["private_run"] / cell.id
        d.mkdir(parents=True, exist_ok=True)
        with (d / "bon_responses.jsonl").open("w") as f:
            for g in _goals():
                for k in range(cfg["bon"]["n_max"]):
                    f.write(json.dumps({
                        "goal_id": g.goal_id, "sample_idx": k, "trial": 0,
                        "variant": f"variant {k}",
                        "response": f"benign response {k} for {g.goal_id}",
                    }) + "\n")
        with (d / "xstest_responses.jsonl").open("w") as f:
            for g in _xs_goals():
                f.write(json.dumps({"goal_id": g.goal_id,
                                    "response": "benign compliance text"}) + "\n")


def _run_aggregate(cfg, root, monkeypatch):
    import sisteer.eval.jailbreak_data as jd

    monkeypatch.setattr(two_si_run, "_eval_goals",
                        lambda cfg, *, confirmatory, attack=None:
                        _goals())
    monkeypatch.setattr(jd, "load_xstest_safe", lambda **kw: _xs_goals())
    args = SimpleNamespace(attack="bon", trial=0, confirmatory=False,
                           private_root=str(root), transport="live",
                           concurrency=1, poll_interval=0, out=None)
    two_si_run.cmd_aggregate(
        args, cfg,
        grade_fn=lambda items: {it: 0.1 for it in items},
        xstest_grade_fn=lambda items: {it: "C" for it in items})
    out = Path(cfg["out_dir"]) / "two_si_bon_aggregate.json"
    return json.loads(out.read_text())


# -- the controls-summary side -------------------------------------------------


def _orb_payload():
    return {"n": 1319, "full_compliance_frac": 0.9,
            "partial_refusal_frac": 0.06, "full_refusal_frac": 0.04,
            "rubric_sha": "ab12cd34" * 4}


def _xstest_payload():
    return {"n": 450,
            "safe": {"n": 250, "full_compliance_frac": 0.99,
                     "partial_refusal_frac": 0.01, "full_refusal_frac": 0.0},
            "unsafe": {"n": 200, "full_compliance_frac": 0.02,
                       "partial_refusal_frac": 0.08,
                       "full_refusal_frac": 0.9},
            "rubric_sha": "ffee0011" * 4}


def _merge_source(cell, **over):
    """One tinycap store's merge source (PW7): every entry the tinycap merge
    writes names the store it came from."""
    return {"cell_key": f"{cell.id.replace('__', '--')}__controls__t0__a0__aaaaaaaa__bbbbbbbb",
            "harness_config_hash": "bb" * 16, "pass": 1, "producer_tree_sha": "c" * 40, **over}


def _write_summary(cfg, cell, payloads, *, sources=None, **prov_over):
    from sisteer.capeval.tinycap import contracts as tc

    prov = controls_provenance(cell, cfg, rubric_shas={
        c: p["rubric_sha"] for c, p in payloads.items() if "rubric_sha" in p})
    prov.update(prov_over)
    if sources is None:
        sources = {k: _merge_source(cell) for k, p in payloads.items()
                   if tc.requires_merge_source(k, p)}
    return write_controls_summary(cfg["controls_dir"], cell, payloads, prov, sources=sources)


def _object_cell_with_artifact(registry):
    cell = next(c for c in registry if c.class_ == "A")
    cell.object_path.parent.mkdir(parents=True, exist_ok=True)
    cell.object_path.write_bytes(b"trained-object-bytes")
    return cell


# --------------------------------------------------------------------------- #
# the fill                                                                     #
# --------------------------------------------------------------------------- #


def test_fill_only_under_valid_provenance_and_unfilled_slots_stay_sentinel(
        cfg, tmp_path, monkeypatch):
    root = tmp_path / "state"
    registry = _registry(cfg)
    _store(cfg, root, registry)
    cell = _object_cell_with_artifact(registry)
    _write_summary(cfg, cell, {"or_bench_hard_1k": _orb_payload(),
                               "mmlu": {"acc": 0.66, "n": 200}})

    agg = _run_aggregate(cfg, root, monkeypatch)
    entry = agg["cells"][cell.id]["controls"]
    # filled from the validated summary, stamped source + provenance digest
    assert entry["or_bench_hard_1k"]["full_compliance_frac"] == 0.9
    assert entry["or_bench_hard_1k"]["source"] == "cell_controls"
    assert entry["mmlu"] == {"acc": 0.66, "n": 200, "source": "cell_controls",
                             "provenance": entry["mmlu"]["provenance"]}
    # not-yet-run slots stay explicitly pending; gsm8k EVERYWHERE (nothing
    # fills it here — the served standard does, through the same mechanism)
    assert entry["gsm8k"] == "pending_control_not_yet_run"
    for c in registry:
        assert agg["cells"][c.id]["controls"]["gsm8k"] == \
            "pending_control_not_yet_run"
    # every OTHER cell's or_bench slot is untouched
    other = next(c for c in registry if c.id != cell.id)
    assert agg["cells"][other.id]["controls"]["or_bench_hard_1k"] == \
        "pending_control_not_yet_run"


def test_traintest_own_xstest_value_wins(cfg, tmp_path, monkeypatch):
    """The aggregate's own 250-safe xstest result fills the slot; a summary's
    xstest payload never overwrites it (a filled slot moves only on new
    information)."""
    root = tmp_path / "state"
    registry = _registry(cfg)
    _store(cfg, root, registry)
    cell = _object_cell_with_artifact(registry)
    _write_summary(cfg, cell, {"xstest": _xstest_payload()})

    agg = _run_aggregate(cfg, root, monkeypatch)
    xs = agg["cells"][cell.id]["controls"]["xstest"]
    # their machinery's shape (n = the 2 synthetic safe prompts, C-graded),
    # not the summary's 450-row payload
    assert xs["n"] == len(_xs_goals())
    assert xs["full_compliance_frac"] == 1.0
    assert "safe" not in xs and "source" not in xs


def test_absent_controls_dir_leaves_all_sentinels(cfg, tmp_path, monkeypatch):
    cfg["controls_dir"] = str(tmp_path / "never-created")
    root = tmp_path / "state"
    registry = _registry(cfg)
    _store(cfg, root, registry)
    agg = _run_aggregate(cfg, root, monkeypatch)
    for c in registry:
        controls = agg["cells"][c.id]["controls"]
        for name, value in controls.items():
            if name != "xstest":
                assert value == "pending_control_not_yet_run"


def test_default_controls_dir_is_under_out_dir(cfg, tmp_path, monkeypatch):
    """No controls_dir key -> <out_dir>/controls (== out/two_si/controls for
    the shipped config, and tmp-rooted/hermetic in tests)."""
    del cfg["controls_dir"]
    root = tmp_path / "state"
    registry = _registry(cfg)
    _store(cfg, root, registry)
    cell = _object_cell_with_artifact(registry)
    cfg_for_write = {**cfg, "controls_dir": str(Path(cfg["out_dir"]) / "controls")}
    _write_summary(cfg_for_write, cell, {"mmlu": {"acc": 0.5, "n": 200}})

    agg = _run_aggregate(cfg, root, monkeypatch)
    assert agg["cells"][cell.id]["controls"]["mmlu"]["acc"] == 0.5


def test_stale_provenance_refuses_the_aggregate(cfg, tmp_path, monkeypatch):
    """r1 #4 at the aggregate: a summary judged under a different model
    revision (or a retrained artifact) raises — the aggregate is never
    published over a stale fill."""
    root = tmp_path / "state"
    registry = _registry(cfg)
    _store(cfg, root, registry)
    cell = _object_cell_with_artifact(registry)
    _write_summary(cfg, cell, {"mmlu": {"acc": 0.5, "n": 200}},
                   model_revision="someone-elses-revision")
    with pytest.raises(ValueError, match="STALE/FOREIGN"):
        _run_aggregate(cfg, root, monkeypatch)

    # fix the provenance, then retrain the object -> refuses again
    (Path(cfg["controls_dir"]) / cell.id / "controls_summary.json").unlink()
    _write_summary(cfg, cell, {"mmlu": {"acc": 0.5, "n": 200}})
    cell.object_path.write_bytes(b"RETRAINED bytes")
    with pytest.raises(ValueError, match="artifact_sha256"):
        _run_aggregate(cfg, root, monkeypatch)


def test_sanitizer_and_grep_gate_pass_on_filled_payloads(cfg, tmp_path,
                                                         monkeypatch):
    """The filled aggregate survives write_sanitized_json's allowlist AND the
    grep gate (cmd_aggregate raises on either) — with hash strings and the
    source marker allowlisted from the validated summaries only."""
    root = tmp_path / "state"
    registry = _registry(cfg)
    _store(cfg, root, registry)
    cell = _object_cell_with_artifact(registry)
    _write_summary(cfg, cell, {"or_bench_hard_1k": _orb_payload(),
                               "xstest": _xstest_payload()})
    agg = _run_aggregate(cfg, root, monkeypatch)   # raises on sanitizer/grep
    out = Path(cfg["out_dir"]) / "two_si_bon_aggregate.json"
    assert out.exists()
    entry = agg["cells"][cell.id]["controls"]
    # (the xstest slot holds this run's own value, which wins over the
    # summary under the fill order, so the payload asserted is or_bench's)
    assert entry["or_bench_hard_1k"]["full_compliance_frac"] == 0.9
    assert entry["or_bench_hard_1k"]["rubric_sha"] == "ab12cd34" * 4


def test_served_standard_gsm8k_slot_fills_and_names_its_instrument(cfg, tmp_path,
                                                                    monkeypatch):
    """PW3 (U2-impl 1/5), the item set flipped by U4(a) / PW17: the two_si
    configs keep the slot NAME `gsm8k`; the served standard (inspect_evals'
    gsm8k on the PAIRED 200-item subset) fills it through the same
    provenance-validated summary, with a payload that NAMES its instrument
    (task inspect/gsm8k_paired200, item_set paired200) — never the retired
    `{acc, n, extraction_failures}` shape — and the sanitizer + grep gate
    pass on it."""
    from sisteer.capeval.tinycap import contracts as tc

    # the sanitizer's fixed key enum carries every slot field
    assert set(tc.GSM8K_SLOT_PAYLOAD_FIELDS) <= set(two_si_run._CONTROL_PAYLOAD_KEYS)
    root = tmp_path / "state"
    registry = _registry(cfg)
    _store(cfg, root, registry)
    cell = _object_cell_with_artifact(registry)
    task_summary = {"acc": 0.81, "n": 200, "n_requests": 200, "n_rejected": 0,
                    "samples_per_item": 1, "task": "inspect/gsm8k_paired200",
                    "item_set": "paired200",
                    "framework": tc.INSPECT_TASK_VERSION, "engine": "vllm",
                    "route": "petri__si_safety__psr_kl_all", "thinking": "off",
                    "render": "native", "wall_clock_s": 83.1, "max_tokens": 512,
                    "temperature": 0.0}
    # what completion.merge_controls_summary writes: the task entry + the slot
    from sisteer.capeval.tinycap import completion
    payloads = {"inspect/gsm8k_paired200": task_summary, **completion.slot_payloads(
        {"inspect/gsm8k_paired200": task_summary})}
    assert set(payloads) == {"inspect/gsm8k_paired200", "gsm8k"}
    _write_summary(cfg, cell, payloads)

    agg = _run_aggregate(cfg, root, monkeypatch)
    slot = agg["cells"][cell.id]["controls"]["gsm8k"]
    assert slot["task"] == "inspect/gsm8k_paired200" and slot["item_set"] == "paired200"
    assert slot["acc"] == 0.81 and slot["n"] == 200
    assert slot["framework"] == tc.INSPECT_TASK_VERSION and slot["thinking"] == "off"
    assert slot["source"] == "cell_controls" and slot["provenance"]
    assert "extraction_failures" not in slot
    # the aggregate carries the six declared slots only — the task-keyed entry
    # stays in the summary file, never in the served aggregate
    assert set(agg["cells"][cell.id]["controls"]) == set(cfg["controls"])
    assert "legacy_controls" not in agg["cells"][cell.id]      # nothing legacy about the standard
    # the other cells' slots stay pending
    other = next(c for c in registry if c.id != cell.id)
    assert agg["cells"][other.id]["controls"]["gsm8k"] == "pending_control_not_yet_run"


def test_legacy_gsm8k_summary_leaves_the_slot_pending_and_carries_the_number_as_legacy(
        cfg, tmp_path, monkeypatch):
    """PW5-1 (re-review F1) at the aggregate: a pre-PW3 controls_summary.json
    carrying the RETIRED instrument's `gsm8k` payload (`{acc, n,
    extraction_failures}`, n≈200 — built here from the old field tuple) is
    provenance-valid, but the `gsm8k` slot now means inspect_evals' stock
    gsm8k, so the slot reads PENDING and the number rides beside the six
    slots under `legacy_controls[gsm8k_legacy_s2parity]` — explicitly legacy,
    never the headline; the sanitizer + grep gate pass on it."""
    from sisteer.capeval.tinycap import contracts as tc

    assert two_si_run._LEGACY_CONTROL_KEYS == (tc.GSM8K_LEGACY_SLOT,)   # the literal twin, pinned
    root = tmp_path / "state"
    registry = _registry(cfg)
    _store(cfg, root, registry)
    cell = _object_cell_with_artifact(registry)
    legacy = {"acc": 0.535, "n": 200, "extraction_failures": 4}
    _write_summary(cfg, cell, {"gsm8k": legacy, "mmlu": {"acc": 0.66, "n": 200}})

    agg = _run_aggregate(cfg, root, monkeypatch)                # raises on sanitizer/grep
    entry = agg["cells"][cell.id]
    assert entry["controls"]["gsm8k"] == "pending_control_not_yet_run"
    assert set(entry["controls"]) == set(cfg["controls"])       # still exactly the six slots
    assert entry["controls"]["mmlu"]["acc"] == 0.66              # the other slots fill as before
    assert set(entry["legacy_controls"]) == {"gsm8k_legacy_s2parity"}
    kept = entry["legacy_controls"]["gsm8k_legacy_s2parity"]
    assert kept["acc"] == 0.535 and kept["n"] == 200 and kept["extraction_failures"] == 4
    assert kept["source"] == "cell_controls" and kept["provenance"] == entry["controls"]["mmlu"]["provenance"]
    # written through the sanitizer: the file on disk carries it too
    out = json.loads((Path(cfg["out_dir"]) / "two_si_bon_aggregate.json").read_text())
    assert out["cells"][cell.id]["legacy_controls"]["gsm8k_legacy_s2parity"]["n"] == 200
    # cells without a legacy number carry no legacy block at all
    for other in registry:
        if other.id != cell.id:
            assert "legacy_controls" not in agg["cells"][other.id]
            assert agg["cells"][other.id]["controls"]["gsm8k"] == "pending_control_not_yet_run"
