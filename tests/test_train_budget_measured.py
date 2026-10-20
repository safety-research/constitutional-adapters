"""The measured GPU budget pass.

CPU-only, no network, no GPU, no Claude. Covers:

- ``measurement_inputs_sha``: the derived ``batch_size_by_*`` overrides,
  ``train_kwargs.*.batch_size``, and unrelated-file edits (a synthetic
  rebase) can NEVER perturb it, while a material trainer-source or
  hyperparameter edit rejects.
- consumption semantics: a golden measured fixture renders a zero-NEEDS_GPU
  table through ``measure_si_budget`` with ZERO caller changes (the
  ``measured_path`` config key / module default); an override above
  ``safe_batch_size`` refuses loudly; the fill comes from the EXACT
  resolved-bs trial row (absent => loud failure); a sha-mismatched table
  raises when explicitly requested and leaves the explicit NEEDS_GPU marks
  (with a warning) under the implicit default.
- an end-to-end subprocess run of the EXISTING plan command
  (``train_two_si_objects.py plan``, unmodified) shows measured values.
- the ladders: fisher hard-capped at 2 (a config 8 clamps), non-fisher
  ladders start at 1.
- exact-match resume: a fingerprint mismatch refuses.
- ``est_seconds`` uses the same formula as ``measure_si_budget``.
- the §5.3 mechanical guarantee on the golden fixture:
  ``resolve_batch_size(si, arm) <= measured-safe`` for ALL 22 (si, arm)
  pairs of the shipped config (via the same check ``table`` runs).
- the synthetic worst-case construction reaches the per-SI ``max_len``
  exactly and refuses when it cannot.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import measure_train_budget as mtb  # noqa: E402
from sisteer.eval.train_budget import (  # noqa: E402
    DEFAULT_BATCH_SIZE,
    FISHER_MAX_BATCH,
    apply_measured,
    format_budget_table,
    load_measured,
    measure_si_budget,
    measurement_inputs_sha,
    resolve_batch_size,
)

REPO = Path(__file__).resolve().parents[1]
TWO_SI_CONFIG = REPO / "configs" / "two_si.yaml"


class WordTokenizer:
    """Whitespace stub (test_train_budget.py convention) — exactly the
    surface the budget/measurement code touches."""

    eos_token_id = 1
    pad_token_id = 0
    bos_token_id = 2
    chat_template = None

    def __call__(self, text, add_special_tokens=True):
        ids = [3 + (i % 1000) for i, _ in enumerate(text.split())]
        if add_special_tokens:
            ids = [self.bos_token_id] + ids
        return {"input_ids": ids}


def _toy_cfg(**over):
    cfg = {
        "model": "toy/model",
        "model_revision": "deadbeef",
        "dtype": "float32",
        "attn_implementation": "sdpa",
        "layer": 1,
        "si_ids": ["sA", "sB"],
        "max_len": 512,
        "max_len_by_si": {"sA": 256, "sB": 512},
        "completion_tokens": 16,
        "class_a_arms": ["vector_all", "alphakl_fisher"],
        "train_kwargs": {"default": {"epochs": 2, "lr": 5e-3}},
    }
    cfg.update(over)
    return cfg


def _trial(si, arm, bs, *, ok=True, oom=False, tps=1000.0, guard=False,
           error=None, calib=False, fingerprint=None):
    return {
        "si_id": si, "arm": arm, "batch_size": bs, "ok": ok, "oom": oom,
        "error": error, "calibration_incoherent_on_synthetic": calib,
        "tokens_per_s_median": tps if ok else None,
        "peak_gb_alloc": 10.0 + bs, "peak_gb_reserved": 12.0 + bs,
        "guard_exceeded": guard, "device_total_gb": 140.0,
        "fingerprint": fingerprint or {"arm_inputs_sha": "x"},
    }


def _measured(cfg, trials, safe, sha=None):
    return {
        "measurement_inputs_sha": sha or measurement_inputs_sha(cfg),
        "seq_construction_version": "two_si_budget_seq_v1",
        "safe_batch_size": safe,
        "trials": trials,
    }


def _write(tmp_path, measured, name="measured.json"):
    p = tmp_path / name
    p.write_text(json.dumps(measured))
    return p


# --------------------------------------------------------------------------- #
# measurement_inputs_sha — r2 #1 / r3 #1 sensitivity                           #
# --------------------------------------------------------------------------- #


def _stub_repo(tmp_path, cfg):
    """A synthetic repo tree holding every source measurement_inputs_sha
    hashes for ``cfg`` — the r3 #1 rebase-survival stage."""
    from sisteer.eval.train_budget import (
        _ARM_TRAINER_SOURCES,
        _MEASUREMENT_SOURCES,
    )

    root = tmp_path / "repo"
    rels = set(_MEASUREMENT_SOURCES)
    for arm in cfg["class_a_arms"]:
        rels.update(_ARM_TRAINER_SOURCES[arm])
    for rel in rels:
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(f"# stub {rel}\n")
    # unrelated files a rebase might touch
    (root / "README.md").write_text("docs\n")
    (root / "docs").mkdir()
    (root / "docs" / "notes.md").write_text("unrelated\n")
    return root


def test_inputs_sha_ignores_derived_batch_overrides_and_rebase(tmp_path):
    """The §5.3 config commit (batch_size_by_*) and an unrelated rebase
    (docs/other-file edits) can NEVER self-invalidate the table (r2 #1)."""
    cfg = _toy_cfg()
    root = _stub_repo(tmp_path, cfg)
    base = measurement_inputs_sha(cfg, repo_root=root)

    # derived batch widths — the §5.3 commit — are excluded
    filled = _toy_cfg(batch_size_by_arm={"vector_all": 8},
                      batch_size_by_si={"sA": 4})
    assert measurement_inputs_sha(filled, repo_root=root) == base
    # train_kwargs batch_size is the same derived knob, same exclusion
    tk = _toy_cfg(train_kwargs={"default": {"epochs": 2, "lr": 5e-3,
                                            "batch_size": 16}})
    assert measurement_inputs_sha(tk, repo_root=root) == base
    # output paths are not hashed at all
    outp = _toy_cfg(measured_path="somewhere/else.json", out_dir="out/x")
    assert measurement_inputs_sha(outp, repo_root=root) == base

    # a synthetic rebase: unrelated files change/appear -> sha unchanged
    (root / "README.md").write_text("docs, rebased\n")
    (root / "docs" / "new_file.py").write_text("print('unrelated')\n")
    assert measurement_inputs_sha(cfg, repo_root=root) == base


def test_inputs_sha_rejects_material_edits(tmp_path):
    """A material trainer-source or hyperparameter/sequence edit perturbs the
    sha — the stale table then rejects (r3 #1)."""
    cfg = _toy_cfg()
    root = _stub_repo(tmp_path, cfg)
    base = measurement_inputs_sha(cfg, repo_root=root)

    # a material trainer edit (the fisher arm's own module)
    p = root / "src/sisteer/extraction/alphakl.py"
    p.write_text(p.read_text() + "\n# material change\n")
    assert measurement_inputs_sha(cfg, repo_root=root) != base

    root2 = _stub_repo(tmp_path / "b", cfg)
    base2 = measurement_inputs_sha(cfg, repo_root=root2)
    # a material hyperparameter edit (config-resolved training spec)
    lr = _toy_cfg(train_kwargs={"default": {"epochs": 2, "lr": 1e-4}})
    assert measurement_inputs_sha(lr, repo_root=root2) != base2
    # the sequence budget is measurement-relevant
    ml = _toy_cfg(max_len_by_si={"sA": 512, "sB": 512})
    assert measurement_inputs_sha(ml, repo_root=root2) != base2
    # so are the model pin and the arm set
    assert measurement_inputs_sha(_toy_cfg(model_revision="cafebabe"),
                                  repo_root=root2) != base2
    assert measurement_inputs_sha(_toy_cfg(class_a_arms=["vector_all"]),
                                  repo_root=root2) != base2


def test_inputs_sha_refuses_absent_source_and_unmapped_arm(tmp_path):
    cfg = _toy_cfg()
    root = _stub_repo(tmp_path, cfg)
    (root / "src/sisteer/extraction/alphakl.py").unlink()
    with pytest.raises(FileNotFoundError, match="measurement source missing"):
        measurement_inputs_sha(cfg, repo_root=root)
    # an arm with no trainer-source mapping is a loud refusal, never an
    # unhashed trainer — use a REGISTERED arm temporarily unmapped
    from sisteer.eval import train_budget as tb

    cfg2 = _toy_cfg()
    root2 = _stub_repo(tmp_path / "c", cfg2)
    removed = tb._ARM_TRAINER_SOURCES.pop("vector_all")
    try:
        with pytest.raises(ValueError, match="_ARM_TRAINER_SOURCES"):
            measurement_inputs_sha(cfg2, repo_root=root2)
    finally:
        tb._ARM_TRAINER_SOURCES["vector_all"] = removed


# --------------------------------------------------------------------------- #
# consumption — golden fixture, refusals, exact-row fill (r3 #2)               #
# --------------------------------------------------------------------------- #


def _rows(cfg, si, arms, measured_path=None):
    tok = WordTokenizer()
    from sisteer.extraction.distill import DistillExample

    examples = [DistillExample(prompt_with_si="si words here prompt",
                               prompt_without="prompt",
                               completion="a b c")]
    return measure_si_budget(cfg, tok, examples, si_id=si, arms=arms,
                             measured_path=measured_path)


def test_golden_fixture_renders_zero_needs_gpu_table(tmp_path):
    cfg = _toy_cfg()
    trials = [_trial("sA", "vector_all", 1), _trial("sA", "vector_all", 2)]
    safe = {"sA": {"vector_all": 2}, "sB": {"vector_all": 2}}
    path = _write(tmp_path, _measured(cfg, trials, safe))
    cfg["measured_path"] = str(path)

    rows = _rows(cfg, "sA", ["vector_all"])
    assert rows[0].batch_size == DEFAULT_BATCH_SIZE == 2
    assert rows[0].peak_memory_gb == 14.0          # reserved peak of the bs=2 trial
    assert rows[0].tokens_per_s == 1000.0
    table = format_budget_table(rows)
    assert "NEEDS_GPU" not in table
    assert "1e+03" in table  # the measured tokens/s rendered


def test_default_path_consumption_without_config_key(tmp_path, monkeypatch):
    """measure_si_budget consumes the MODULE default path with no config key
    and no argument — the zero-caller-changes contract (r1 #13)."""
    from sisteer.eval import train_budget as tb

    cfg = _toy_cfg()
    trials = [_trial("sA", "vector_all", 1), _trial("sA", "vector_all", 2)]
    safe = {"sA": {"vector_all": 2}}
    path = _write(tmp_path, _measured(cfg, trials, safe))
    monkeypatch.setattr(tb, "DEFAULT_MEASURED_PATH", str(path))

    rows = _rows(cfg, "sA", ["vector_all"])
    assert rows[0].tokens_per_s == 1000.0
    assert rows[0].peak_memory_gb == 14.0


def test_absent_default_leaves_needs_gpu_marks(tmp_path, monkeypatch):
    from sisteer.eval import train_budget as tb

    monkeypatch.setattr(tb, "DEFAULT_MEASURED_PATH",
                        str(tmp_path / "nope.json"))
    rows = _rows(_toy_cfg(), "sA", ["vector_all"])
    assert rows[0].peak_memory_gb == "NEEDS_GPU"
    assert rows[0].tokens_per_s == "NEEDS_GPU"


def test_override_above_safe_width_refuses(tmp_path):
    """r3 #2 (b): a later config override above the measured-safe width can
    never silently pass through the plan command."""
    cfg = _toy_cfg(batch_size_by_arm={"vector_all": 8})
    trials = [_trial("sA", "vector_all", b) for b in (1, 2)]
    path = _write(tmp_path, _measured(cfg, trials,
                                      {"sA": {"vector_all": 2}}))
    cfg["measured_path"] = str(path)
    with pytest.raises(ValueError, match="exceeds the measured-safe width"):
        _rows(cfg, "sA", ["vector_all"])


def test_fill_requires_the_exact_resolved_bs_trial(tmp_path):
    """r3 #2 (c): values come from the exact (si, arm, batch_size=resolved)
    trial row — absent is a loud failure, never another trial's numbers."""
    cfg = _toy_cfg(batch_size_by_arm={"vector_all": 3})   # 3 <= safe 4, no trial
    trials = [_trial("sA", "vector_all", b) for b in (1, 2, 4)]
    path = _write(tmp_path, _measured(cfg, trials,
                                      {"sA": {"vector_all": 4}}))
    cfg["measured_path"] = str(path)
    with pytest.raises(ValueError, match="exact resolved batch"):
        _rows(cfg, "sA", ["vector_all"])
    # an OOM row at the resolved width is not a successful trial either
    cfg2 = _toy_cfg()
    trials2 = [_trial("sA", "vector_all", 1),
               _trial("sA", "vector_all", 2, ok=False, oom=True)]
    path2 = _write(tmp_path, _measured(cfg2, trials2,
                                       {"sA": {"vector_all": 2}}),
                   name="m2.json")
    cfg2["measured_path"] = str(path2)
    with pytest.raises(ValueError, match="exact resolved batch"):
        _rows(cfg2, "sA", ["vector_all"])


def test_missing_safe_entry_refuses(tmp_path):
    cfg = _toy_cfg()
    path = _write(tmp_path, _measured(cfg, [_trial("sA", "vector_all", 2)],
                                      {"sA": {}}))
    cfg["measured_path"] = str(path)
    with pytest.raises(ValueError, match="no safe_batch_size"):
        _rows(cfg, "sA", ["vector_all"])


def test_sha_mismatch_explicit_raises_implicit_warns(tmp_path, monkeypatch,
                                                     capsys):
    """The consumption gate is measurement_inputs_sha ONLY: a stale table
    (material edit since the pass) REJECTS — loudly when explicitly
    requested; under the implicit default the rows keep their explicit
    NEEDS_GPU marks and a warning names the mismatch (never silently-wrong
    numbers, never taking unrelated configs hostage)."""
    from sisteer.eval import train_budget as tb

    cfg = _toy_cfg()
    stale = _measured(cfg, [_trial("sA", "vector_all", 2)],
                      {"sA": {"vector_all": 2}}, sha="0" * 64)
    path = _write(tmp_path, stale)

    # explicit (config key) -> raise
    cfg_explicit = _toy_cfg(measured_path=str(path))
    with pytest.raises(ValueError, match="incompatible"):
        _rows(cfg_explicit, "sA", ["vector_all"])
    # explicit (argument) -> raise
    with pytest.raises(ValueError, match="incompatible"):
        _rows(_toy_cfg(), "sA", ["vector_all"], measured_path=str(path))
    # implicit default -> warning + NEEDS_GPU retained
    monkeypatch.setattr(tb, "DEFAULT_MEASURED_PATH", str(path))
    rows = _rows(_toy_cfg(), "sA", ["vector_all"])
    assert rows[0].peak_memory_gb == "NEEDS_GPU"
    assert "incompatible" in capsys.readouterr().err


def test_explicit_measured_path_missing_file_raises(tmp_path):
    cfg = _toy_cfg(measured_path=str(tmp_path / "gone.json"))
    with pytest.raises(FileNotFoundError):
        _rows(cfg, "sA", ["vector_all"])
    with pytest.raises(FileNotFoundError):
        load_measured(tmp_path / "gone.json")


def test_rebase_survival_config_commit_still_consumes(tmp_path):
    """r2 #1 end-to-end: the finalized table + the §5.3 config commit
    (batch_size_by_* populated) cherry-picked together — the table is STILL
    consumed (the derived widths are excluded from the sha), and the filled
    values track the newly-resolved width's own trial row."""
    base_cfg = _toy_cfg()
    trials = [_trial("sA", "vector_all", b, tps=100.0 * b) for b in (1, 2, 4)]
    path = _write(tmp_path, _measured(base_cfg, trials,
                                      {"sA": {"vector_all": 4}}))
    # the §5.3 commit: by_arm width appears AFTER the table was emitted
    cfg = _toy_cfg(batch_size_by_arm={"vector_all": 4},
                   measured_path=str(path))
    rows = _rows(cfg, "sA", ["vector_all"])
    assert rows[0].batch_size == 4
    assert rows[0].tokens_per_s == 400.0   # the bs=4 trial, not any other


def test_est_seconds_formula_matches_measure_si_budget(tmp_path):
    """apply_measured's est_seconds is byte-identical to measure_si_budget's
    own tokens_per_s formatting (same fwd-token convention, same %.3g)."""
    cfg = _toy_cfg()
    tok = WordTokenizer()
    from sisteer.extraction.distill import DistillExample

    examples = [DistillExample(prompt_with_si="si prompt words",
                               prompt_without="prompt", completion="a b c")]
    via_param = measure_si_budget(cfg, tok, examples, si_id="sA",
                                  arms=["vector_all"], tokens_per_s=1000.0)
    trials = [_trial("sA", "vector_all", b) for b in (1, 2)]
    path = _write(tmp_path, _measured(cfg, trials, {"sA": {"vector_all": 2}}))
    via_measured = measure_si_budget(cfg, tok, examples, si_id="sA",
                                     arms=["vector_all"],
                                     measured_path=str(path))
    assert via_param[0].est_seconds == via_measured[0].est_seconds


def test_apply_measured_epochs_default_documented():
    rows_fixture = []
    # apply_measured is callable standalone (load_measured +
    # apply_measured) — empty rows is a no-op, not an error
    assert apply_measured(rows_fixture,
                          {"safe_batch_size": {}, "trials": []}) == []


# --------------------------------------------------------------------------- #
# ladders + resume (r1 #16 / #17)                                              #
# --------------------------------------------------------------------------- #


def test_fisher_ladder_hard_capped_and_others_start_at_1():
    assert mtb.ladder_for("alphakl_fisher") == [1, 2]
    assert mtb.ladder_for("vector_all") == [1, 2, 4, 8, 16]
    assert mtb.ladder_for("vector_all")[0] == 1
    # the config asking for 8 on fisher still resolves to the cap
    cfg = _toy_cfg(batch_size_by_arm={"alphakl_fisher": 8})
    assert resolve_batch_size(cfg, "sA", "alphakl_fisher") == FISHER_MAX_BATCH == 2


def test_resume_skips_exact_match_and_supersedes_any_difference(tmp_path):
    fp = {"model": "m", "shapes": {"max_len": 8}}
    row = {"fingerprint": dict(fp)}
    # no row -> run it
    assert mtb.check_resume(None, fp, ("sA", "a", 1), tmp_path) == "run"
    # identical fingerprint -> skip
    assert mtb.check_resume(row, fp, ("sA", "a", 1), tmp_path) == "skip"
    # ANY difference -> supersede (re-run under the current world; the stale
    # measurement is never read, and other keys' rows are untouched)
    assert mtb.check_resume({"fingerprint": {"model": "m",
                                             "shapes": {"max_len": 9}}},
                            fp, ("sA", "a", 1), tmp_path) == "supersede"


def test_supersede_moves_the_stale_row_to_the_sidecar(tmp_path):
    trials = tmp_path / "trials.jsonl"
    stale = _trial("sA", "vector_all", 2)
    keep = _trial("sA", "alphakl_fisher", 1)
    trials.write_text(json.dumps(stale) + "\n" + json.dumps(keep) + "\n")
    mtb._supersede_row(trials, ("sA", "vector_all", 2), stale)
    live = mtb._read_trials(trials)
    assert set(live) == {("sA", "alphakl_fisher", 1)}   # only the key removed
    sidecar = trials.with_name("trials.jsonl.superseded.jsonl")
    moved = [json.loads(l) for l in sidecar.read_text().splitlines()]
    assert moved[0]["arm"] == "vector_all" and "superseded_at" in moved[0]


def test_trial_kwargs_merge_default_under_arm_block():
    """Production parity: train_kwargs.default merges
    UNDER train_kwargs.<arm> — the alphakl_soft beta reaches its trainer —
    and the measurement-forced knobs always override."""
    cfg = _toy_cfg(train_kwargs={"default": {"epochs": 6, "lr": 5e-3,
                                             "batch_size": 16},
                                 "alphakl_soft": {"beta": 1.0, "lr": 1e-3}})
    kw, extra = mtb._trial_kwargs(cfg, "alphakl_soft", bs=2, max_len=512)
    assert kw["beta"] == 1.0
    assert kw["lr"] == 1e-3                    # arm block wins over default
    assert kw["epochs"] == 1 and kw["batch_size"] == 2   # forced
    assert kw["max_len"] == 512 and kw["seed"] == 0
    assert extra == {}
    # an arm with no block gets the default (minus the forced keys)
    kw2, extra2 = mtb._trial_kwargs(cfg, "vector_all", bs=4, max_len=256)
    assert kw2["lr"] == 5e-3 and "beta" not in kw2
    assert kw2["batch_size"] == 4              # the ladder, never config 16
    assert extra2 == {}
    # (cast's recorded pairs-fallback special case left with its 2026-08-20
    # registry removal — no measurable arm carries trial extras today.)


def test_shipped_config_leaves_alphakl_soft_beta_derived():
    """The shipped config deliberately carries NO stage-2 beta pin
    (2026-08-24, inverting the 2026-08-19 pin this test used to assert):
    beta=1.0 was a crash workaround promoted to a hyperparameter — the
    stage-2 trainer used to raise without a manual beta — and the arm
    diverged under it. train_alphakl now derives beta against the eps
    benign budget when the key is absent, so the trial kwargs must carry
    no beta and the measurement trials exercise the same derived path the
    real fit runs."""
    cfg = _shipped_cfg()
    assert "beta" not in (cfg["train_kwargs"].get("alphakl_soft") or {})
    kw, _ = mtb._trial_kwargs(cfg, "alphakl_soft", bs=1, max_len=8192)
    assert "beta" not in kw


def test_read_trials_refuses_duplicates(tmp_path):
    p = tmp_path / "trials.jsonl"
    row = json.dumps(_trial("sA", "vector_all", 1))
    p.write_text(row + "\n" + row + "\n")
    with pytest.raises(SystemExit, match="duplicate"):
        mtb._read_trials(p)


# --------------------------------------------------------------------------- #
# arm-scoped resume identity + migrate (the 2026-08-19 beta-fix lesson)        #
# --------------------------------------------------------------------------- #


def test_arm_scoped_sha_isolates_one_arm_and_ignores_the_harness(tmp_path):
    from sisteer.eval.train_budget import (
        arm_scoped_inputs_sha,
        measurement_inputs_sha,
    )

    cfg = _toy_cfg()
    root = _stub_repo(tmp_path, cfg)
    base_va = arm_scoped_inputs_sha(cfg, "sA", "vector_all", repo_root=root)
    base_fi = arm_scoped_inputs_sha(cfg, "sA", "alphakl_fisher",
                                    repo_root=root)
    base_global = measurement_inputs_sha(cfg, repo_root=root)

    # a per-arm kwarg (the beta case) changes ONLY that arm's scoped sha
    beta = _toy_cfg(train_kwargs={"default": {"epochs": 2, "lr": 5e-3},
                                  "alphakl_fisher": {"beta": 1.0}})
    assert arm_scoped_inputs_sha(beta, "sA", "vector_all",
                                 repo_root=root) == base_va
    assert arm_scoped_inputs_sha(beta, "sA", "alphakl_fisher",
                                 repo_root=root) != base_fi

    # a trainer-source edit changes ITS arm's scoped sha, not the others'
    p = root / "src/sisteer/extraction/alphakl.py"
    p.write_text(p.read_text() + "# material\n")
    assert arm_scoped_inputs_sha(cfg, "sA", "vector_all",
                                 repo_root=root) == base_va
    assert arm_scoped_inputs_sha(cfg, "sA", "alphakl_fisher",
                                 repo_root=root) != base_fi

    # a HARNESS edit changes NO row's scoped sha (SEQ_CONSTRUCTION_VERSION
    # governs harness semantics) — but still re-keys CONSUMPTION (r3 #1)
    q = root / "scripts/measure_train_budget.py"
    q.write_text(q.read_text() + "# mechanical harness edit\n")
    assert arm_scoped_inputs_sha(cfg, "sA", "vector_all",
                                 repo_root=root) == base_va
    assert measurement_inputs_sha(cfg, repo_root=root) != base_global

    # per-SI scoping: the same arm under the other SI is a different row id
    assert arm_scoped_inputs_sha(cfg, "sB", "vector_all",
                                 repo_root=root) != base_va


def test_annotations_and_acceptance_for_calibration_incoherent_ladders():
    """The 2026-08-19 case: bs=1 completed, a wider rung died with the
    calibration-incoherence error. NOT an acceptance failure (bs=1 rows
    exist); the safe width caps at the largest successful rung and the
    (si, arm) is annotated as a measurement-validity note, never read as a
    memory bound. A bs=1 failure is still the loud acceptance stop."""
    cfg = _toy_cfg(si_ids=["sA"], class_a_arms=["vector_all",
                                                "alphakl_fisher"])
    err = ("RuntimeError: cast calibration found NO coherent scale: the "
           "smallest ladder rung (0.25) already fails the degeneration guard")
    rows = {
        ("sA", "vector_all", 1): _trial("sA", "vector_all", 1),
        ("sA", "vector_all", 2): _trial("sA", "vector_all", 2, ok=False,
                                        error=err, calib=True),
        ("sA", "alphakl_fisher", 1): _trial("sA", "alphakl_fisher", 1),
        ("sA", "alphakl_fisher", 2): _trial("sA", "alphakl_fisher", 2),
    }
    safe, failures, warnings = mtb._safe_batch_sizes(cfg, rows)
    assert not failures                       # bs=1 completed everywhere
    assert safe["sA"]["vector_all"] == 1      # capped at the last OK rung
    assert safe["sA"]["alphakl_fisher"] == 2
    notes, note_warnings = mtb._annotations(cfg, rows)
    assert notes["sA"]["vector_all"] == {
        "error_at_batch_size": 2,
        "calibration_incoherent_on_synthetic": True}
    assert any("not a memory bound" in w for w in note_warnings)
    assert "alphakl_fisher" not in notes.get("sA", {}) or \
        "alphakl_fisher" not in notes["sA"]

    # bs=1 itself failing IS the acceptance stop
    rows_bad = {("sA", "vector_all", 1): _trial("sA", "vector_all", 1,
                                                ok=False, error=err,
                                                calib=True)}
    cfg1 = _toy_cfg(si_ids=["sA"], class_a_arms=["vector_all"])
    _safe, failures, _w = mtb._safe_batch_sizes(cfg1, rows_bad)
    assert failures == ["(sA, vector_all)"]


def test_migrate_keeps_unchanged_arms_and_supersedes_the_changed_one(
        tmp_path):
    """The beta-fix flow end-to-end at the file level: old-schema rows
    (global inputs_sha fingerprints) measured under the old world; the new
    config adds one per-arm kwarg; migrate keeps every other arm's rows
    (re-keyed to the arm-scoped fingerprint) and supersedes ONLY the
    changed arm's."""
    import argparse

    from sisteer.eval.train_budget import (
        SEQ_CONSTRUCTION_VERSION,
        arm_scoped_inputs_sha,
        measurement_inputs_sha,
    )

    old_cfg = _toy_cfg()
    root = _stub_repo(tmp_path, old_cfg)     # one tree = old AND new world
    old_cfg_path = tmp_path / "old_two_si.yaml"
    old_cfg_path.write_text(yaml.safe_dump(old_cfg))
    new_cfg = _toy_cfg(train_kwargs={"default": {"epochs": 2, "lr": 5e-3},
                                     "alphakl_fisher": {"beta": 1.0}})
    old_global = measurement_inputs_sha(old_cfg, repo_root=root)

    def old_fp():
        return {"model": old_cfg["model"],
                "model_revision": old_cfg["model_revision"],
                "config_sha256": "whatever",
                "inputs_sha": old_global,
                "seq_construction_version": SEQ_CONSTRUCTION_VERSION,
                "gpu_name": "NVIDIA H200", "torch": "2.11.0",
                "cuda": "12.8", "shapes": {"max_len": 256}}

    trials = tmp_path / "trials.jsonl"
    with trials.open("w") as f:
        for arm, bs in (("vector_all", 1), ("vector_all", 2),
                        ("alphakl_fisher", 1)):
            f.write(json.dumps(_trial("sA", arm, bs,
                                      fingerprint=old_fp())) + "\n")

    args = argparse.Namespace(trials=str(trials),
                              old_config=str(old_cfg_path),
                              old_repo=str(root), repo_root=str(root))
    mtb.cmd_migrate(args, new_cfg)

    live = mtb._read_trials(trials)
    assert set(live) == {("sA", "vector_all", 1), ("sA", "vector_all", 2)}
    want = arm_scoped_inputs_sha(new_cfg, "sA", "vector_all", repo_root=root)
    for row in live.values():
        assert row["fingerprint"]["arm_inputs_sha"] == want
        assert row["fingerprint"]["gpu_name"] == "NVIDIA H200"  # carried over
    sidecar = trials.with_name("trials.jsonl.superseded.jsonl")
    moved = [json.loads(l) for l in sidecar.read_text().splitlines()]
    assert {m["arm"] for m in moved} == {"alphakl_fisher"}
    assert all(m["superseded_by"] == "migrate" for m in moved)

    # unknown provenance (neither world attests the row) refuses loudly
    trials2 = tmp_path / "trials2.jsonl"
    junk = old_fp()
    junk["inputs_sha"] = "0" * 64
    trials2.write_text(json.dumps(_trial("sA", "vector_all", 1,
                                         fingerprint=junk)) + "\n")
    args2 = argparse.Namespace(trials=str(trials2),
                               old_config=str(old_cfg_path),
                               old_repo=str(root), repo_root=str(root))
    with pytest.raises(SystemExit, match="unknown provenance"):
        mtb.cmd_migrate(args2, new_cfg)


def test_emit_refuses_rows_scoped_to_a_different_world(tmp_path):
    """Emit validates every row's arm-scoped sha against the CURRENT world —
    a stale/old-schema row is refused with a pointer at resume/migrate."""
    cfg = _toy_cfg(si_ids=["sA"], class_a_arms=["vector_all"])
    trials = tmp_path / "trials.jsonl"
    trials.write_text(json.dumps(_trial("sA", "vector_all", 1)) + "\n")
    with pytest.raises(SystemExit, match="DIFFERENT per-\\(si, arm\\)"):
        mtb._emit(cfg, trials, None)


# --------------------------------------------------------------------------- #
# synthetic worst-case construction (r1 #14)                                   #
# --------------------------------------------------------------------------- #


def test_filler_text_reaches_token_floor():
    tok = WordTokenizer()
    text = mtb._filler_text(tok, 100)
    assert len(tok(text, add_special_tokens=False)["input_ids"]) >= 100


def test_expected_shapes_reach_the_per_si_maximum():
    tok = WordTokenizer()
    cfg = {"max_len_by_si": {"safety_refusal": 2048}, "completion_tokens": 16}
    shapes = mtb.expected_shapes(tok, cfg, "safety_refusal", batch_size=2)
    assert shapes["total_seq_tokens"] == 2048       # reaches the maximum
    assert shapes["n_examples"] == 2 * 4            # bs x (1 warmup + 3 timed)
    assert shapes["teacher_prompt_tokens"] >= 1024  # the attack allowance
    assert shapes["student_seq_tokens"] <= 2048


def test_expected_shapes_refuse_a_prompt_exceeding_the_config_maximum():
    """A max_len too small for SI + allowance is production's own preflight
    refusal — loud, before any GPU work."""
    tok = WordTokenizer()
    cfg = {"max_len_by_si": {"safety_refusal": 256}, "completion_tokens": 16}
    with pytest.raises(ValueError, match="max_len"):
        mtb.expected_shapes(tok, cfg, "safety_refusal", batch_size=1)


# --------------------------------------------------------------------------- #
# emit — acceptance + safe widths                                              #
# --------------------------------------------------------------------------- #


def test_safe_batch_sizes_require_bs1_and_respect_the_guard():
    cfg = _toy_cfg(si_ids=["sA"], class_a_arms=["vector_all",
                                                "alphakl_fisher"])
    rows = {
        ("sA", "vector_all", 1): _trial("sA", "vector_all", 1),
        ("sA", "vector_all", 2): _trial("sA", "vector_all", 2),
        ("sA", "vector_all", 4): _trial("sA", "vector_all", 4, guard=True),
        ("sA", "alphakl_fisher", 1): _trial("sA", "alphakl_fisher", 1,
                                            ok=False, oom=True),
    }
    safe, failures, warnings = mtb._safe_batch_sizes(cfg, rows)
    # the guard-exceeded bs=4 never becomes the safe width
    assert safe["sA"]["vector_all"] == 2
    # fisher's bs=1 OOM is a failure
    assert failures == ["(sA, alphakl_fisher)"]
    assert warnings == []


def test_safe_batch_size_bs1_over_guard_is_1_with_warning():
    cfg = _toy_cfg(si_ids=["sA"], class_a_arms=["vector_all"])
    rows = {("sA", "vector_all", 1): _trial("sA", "vector_all", 1,
                                            guard=True)}
    safe, failures, warnings = mtb._safe_batch_sizes(cfg, rows)
    assert safe["sA"]["vector_all"] == 1
    assert not failures
    assert len(warnings) == 1 and "headroom" in warnings[0]


# --------------------------------------------------------------------------- #
# the §5.3 mechanical guarantee on ALL 22 pairs of the shipped config          #
# --------------------------------------------------------------------------- #


def _shipped_cfg():
    return yaml.safe_load(TWO_SI_CONFIG.read_text())


@pytest.mark.xfail(
    strict=True,
    reason="the 2026-08-20 NousResearch re-pin (configs/two_si.yaml) "
    "deliberately invalidates the committed measured table: its widths were "
    "measured under the grandfathered meta-llama pin, and "
    "measurement_inputs_sha hashes the pin BY DESIGN (r3 #1). This branch "
    "(nousresearch-rerun) runs EVALS ONLY — cell_controls never consumes the "
    "table (fixed batch_size 8) — and any future FIT under the new pin is "
    "refused loudly at run time by the same sha gate until a GPU re-measure "
    "pass re-commits the table. Un-xfail when that pass lands.")
def test_resolved_leq_measured_safe_for_all_22_pairs(capsys):
    """resolve_batch_size(si, arm) <= measured-safe(si, arm) for EVERY
    (si, arm) of the SHIPPED config against the COMMITTED measured table
    (2 SIs x 11 registered class-A arms = 22 — derived, never hardcoded),
    via the SAME check the `table` subcommand runs at run time (r2 #2).
    This is the live §5.3 mechanical guarantee, and it also pins that the
    committed table's measurement_inputs_sha matches the current tree — a
    material trainer/sequence edit fails HERE (r3 #1, by design)."""
    import argparse

    cfg = _shipped_cfg()
    sis = list(cfg["si_ids"])
    arms = mtb.registered_class_a_arms(cfg)
    assert len(sis) * len(arms) == 22   # the plan's derived count at d535d8c

    args = argparse.Namespace(measured=None)  # the committed default table
    mtb.cmd_table(args, cfg)
    out = capsys.readouterr().out
    assert "OK: resolve_batch_size(si, arm) <= measured-safe for all 22" in out

    # and the guarantee actually BITES: a violating override fails the check
    cfg_bad = {**cfg,
               "batch_size_by_arm": {**cfg["batch_size_by_arm"],
                                     "vector_all": 64}}
    with pytest.raises(SystemExit, match="MECHANICAL GUARANTEE"):
        mtb.cmd_table(args, cfg_bad)

    # the exact-resolved-bs trial exists for every pair (r3 #2): the shipped
    # by_arm projection resolves to a bs the ladder actually measured
    measured = load_measured(REPO / "configs" / "two_si_budget_measured.json")
    trials = {(t["si_id"], t["arm"], t["batch_size"])
              for t in measured["trials"] if t["ok"]}
    for si in sis:
        for arm in arms:
            bs = resolve_batch_size(cfg, si, arm)
            assert (si, arm, bs) in trials, (si, arm, bs)


def test_table_rejects_a_sha_incompatible_measured_file(tmp_path):
    cfg = _shipped_cfg()
    path = _write(tmp_path, _measured(cfg, [], {}, sha="0" * 64))
    import argparse

    with pytest.raises(SystemExit, match="incompatible"):
        mtb.cmd_table(argparse.Namespace(measured=str(path)), cfg)


# --------------------------------------------------------------------------- #
# end-to-end: the EXISTING plan command, zero changes                          #
# --------------------------------------------------------------------------- #


def test_plan_command_emits_filled_table_with_zero_script_changes(
        tmp_path, tiny_llama):
    """A subprocess run of scripts/train_two_si_objects.py plan (the
    production driver, unmodified) against a config whose measured_path
    points at a compatible table renders the budget table with measured
    values — no NEEDS_GPU."""
    del tiny_llama  # ensures the tiny model is in the local HF cache
    cfg = {
        "model": "hf-internal-testing/tiny-random-LlamaForCausalLM",
        "model_revision": None,
        "dtype": "float32",
        "attn_implementation": "sdpa",
        "layer": 1,
        "si_ids": ["safety_refusal"],
        "max_len": 4096,
        "max_len_by_si": {"safety_refusal": 4096},
        "completion_tokens": 8,
        "class_a_arms": ["vector_all", "alphakl_fisher"],
        "class_b_arms": [],
        "defenses_dir": str(tmp_path / "defenses"),
        "train_kwargs": {"default": {"epochs": 2}},
    }
    trials = [_trial("safety_refusal", a, b, tps=1234.5)
              for a in cfg["class_a_arms"] for b in (1, 2)]
    safe = {"safety_refusal": {a: 2 for a in cfg["class_a_arms"]}}
    measured_path = _write(tmp_path, _measured(cfg, trials, safe))
    cfg["measured_path"] = str(measured_path)

    cfg_path = tmp_path / "toy_two_si.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    pairs_path = tmp_path / "pairs.jsonl"
    pairs_path.write_text(
        json.dumps({"prompt": "hello there", "pos": "hi", "neg": ""}) + "\n")

    env = {**os.environ, "HF_HUB_OFFLINE": "1"}
    proc = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "train_two_si_objects.py"),
         "plan", "--config", str(cfg_path), "--si", "safety_refusal",
         "--pairs", str(pairs_path), "--with-model"],
        capture_output=True, text=True, cwd=REPO, env=env, timeout=600)
    assert proc.returncode == 0, proc.stderr
    assert "NEEDS_GPU" not in proc.stdout
    # the fixture's tokens/s (1234.5 -> "%.3g" -> 1.23e+03) reached the table
    assert "1.23e+03" in proc.stdout


def test_trial_kwargs_pops_driver_policy_keys():
    """Guard bars and optimiser-schedule keys must never reach a trainer here.

    _call_trainer's drop loop only catches TypeError ("unexpected keyword
    argument"), so a key the trainer ACCEPTS but rejects on a given path
    sails past it. train_psr takes lr_schedule/max_grad_norm in its
    signature (its mse loop implements them) and raises ValueError on the
    KL path — so once configs/two_si.yaml pinned psr_kl_all to
    warmup_cosine for its runaway, a psr_kl_all re-measure would abort
    rather than degrade. floor_ratio_max is the same class: a guard bar,
    not a trainer kwarg, which the production driver pops for this reason.

    Dropping them is safe for THIS script specifically: it measures peak
    memory and step time, and none of these changes either beyond noise."""
    cfg = _toy_cfg(train_kwargs={
        "default": {"lr": 5e-3, "floor_ratio_max": 1.0},
        "psr_kl_all": {"lr_schedule": "warmup_cosine", "max_grad_norm": 1.0,
                       "warmup_steps": 50},
    })
    kw, extra = mtb._trial_kwargs(cfg, "psr_kl_all", bs=2, max_len=512)
    for k in ("floor_ratio_max", "lr_schedule", "warmup_steps",
              "max_grad_norm"):
        assert k not in kw, f"{k} would reach the trainer and abort the trial"
    assert kw["lr"] == 5e-3            # ordinary keys still pass through
    assert kw["epochs"] == 1 and kw["batch_size"] == 2
    # ...and the drop is auditable in the row, not silent
    assert extra["policy_keys_dropped"] == {
        "floor_ratio_max": 1.0, "lr_schedule": "warmup_cosine",
        "warmup_steps": 50, "max_grad_norm": 1.0}
    # an arm with no block of its own still has the DEFAULT block's policy
    # key popped — floor_ratio_max is set default-wide in configs/two_si.yaml,
    # so this is the common case, not the exotic one
    kw2, extra2 = mtb._trial_kwargs(cfg, "vector_all", bs=4, max_len=256)
    assert "floor_ratio_max" not in kw2
    assert extra2 == {"policy_keys_dropped": {"floor_ratio_max": 1.0}}
    # and a config carrying no policy keys at all records nothing
    _, extra3 = mtb._trial_kwargs(_toy_cfg(train_kwargs={"default": {"lr": 1}}),
                                  "vector_all", bs=4, max_len=256)
    assert extra3 == {}
