"""The two-SI driver (scripts/two_si_run.py): per-cell generate + aggregate.

CPU-only, no network, no GPU, no harmful text (synthetic prompts only, the
``path=``/seam conventions of the existing driver tests). Covers the glue the
two-SI lane adds on top of the (already-tested) DefenseCell registry and the
single-SI attack machinery:

- ``generate --cell`` resolves the RIGHT cell for both SIs of one arm:
  distinct mode-0700 private dirs keyed on ``cell.id`` (never the arm),
  a ``cell_meta`` sidecar carrying the identity + the target-forward count,
  and a loud refusal of any id outside the registry (including a bare
  single-SI arm name)
- the xstest pass resumes by goal_id
- ``aggregate`` calls ``assert_matrix_complete`` BEFORE grading, so a missing
  cell fails loudly instead of producing a complete-looking aggregate, while
  a refused-with-reason cell is a first-class recorded outcome
- the aggregate emits, per cell, the config's FOUR control slots
  (alpaca_eval and wmdp retired 2026-09-11) and the over-refusal NI
  references (Class B references BOTH prompted arms), plus the
  ``over_refusal_comparisons`` seam — structure only, no statistics
- the pool-regime field is ALWAYS present, in BOTH regimes, with both pool
  sizes — and the aggregate refuses to run without the frozen manifest
- the config carries the sections the driver reads, at the same dataset pins
  as the single-SI configs; the sbatch decode goes through
  ``cell_for_array_index`` and the script observes the cluster hard rules.
"""

from __future__ import annotations

import json
import re
import stat
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import two_si_run  # noqa: E402
from sisteer.eval.defense_cell import (  # noqa: E402
    CLASS_A_ARMS,
    CLASS_B_ARMS,
    by_id,
    cell_for_array_index,
    registry_from_cfg,
)
from sisteer.eval.jailbreak_data import Goal  # noqa: E402
from sisteer.methods import ARMS  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
TWO_SI_CONFIG = REPO / "configs" / "two_si.yaml"
STAGE2_CONFIG = REPO / "configs" / "stage2_jailbreak.yaml"


def _goals():
    # Synthetic, benign, and long enough (>= 12 chars) to act as grep-gate
    # needles in the aggregate.
    return [Goal("g0", "benign goal zero text", "illegal", "harmbench", "rev"),
            Goal("g1", "benign goal one text", "illegal", "harmbench", "rev")]


def _xs_goals():
    return [Goal("xs0", "a benign safe prompt zero", "homonyms", "xstest",
                 "rev", split="safe"),
            Goal("xs1", "a benign safe prompt one", "homonyms", "xstest",
                 "rev", split="safe")]


@pytest.fixture()
def cfg(tmp_path):
    """The REAL two_si.yaml (so the tests exercise the shipped config), with
    tmp-rooted stores and a tiny BoN budget so the fixtures stay small."""
    c = yaml.safe_load(TWO_SI_CONFIG.read_text())
    c["defenses_dir"] = str(tmp_path / "defenses" / "cells")
    c["out_dir"] = str(tmp_path / "out")
    c["pool_manifest"] = str(tmp_path / "pool_manifest.json")
    c["private_run"] = "two_si_test"
    c["bon"] = {**c["bon"], "n_max": 2, "max_new_tokens": 8}
    return c


def _write_manifest(cfg, regime="matched_intersection", sizes=None):
    sizes = sizes or {si: 657 for si in cfg["si_ids"]}
    Path(cfg["pool_manifest"]).write_text(
        json.dumps({"regime": regime, "pool_sizes": sizes}))


def _registry(cfg):
    return registry_from_cfg(cfg, repo_root=REPO)


def _store(cfg, root, registry, *, skip=()):
    """A complete private store: per-cell BoN transcripts (n_max responses per
    goal) + the xstest pass, keyed on cell.id."""
    for cell in registry:
        if cell.id in skip:
            continue
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


def _agg_args(root, **over):
    base = dict(attack="bon", trial=0, confirmatory=False,
                private_root=str(root), transport="live", concurrency=1,
                poll_interval=0, out=None)
    base.update(over)
    return SimpleNamespace(**base)


def _run_aggregate(cfg, root, monkeypatch, **over):
    import sisteer.eval.jailbreak_data as jd

    monkeypatch.setattr(two_si_run, "_eval_goals",
                        lambda cfg, *, confirmatory, attack=None:
                        _goals())
    monkeypatch.setattr(jd, "load_xstest_safe",
                        lambda **kw: _xs_goals())
    two_si_run.cmd_aggregate(
        _agg_args(root, **over), cfg,
        grade_fn=lambda items: {it: 0.1 for it in items},
        xstest_grade_fn=lambda items: {it: "C" for it in items})
    out = Path(cfg["out_dir"]) / "two_si_bon_aggregate.json"
    return json.loads(out.read_text())


# --------------------------------------------------------------------------- #
# generate — the right cell, a cell-id-keyed 0700 dir, and the meta sidecar     #
# --------------------------------------------------------------------------- #


def test_generate_resolves_both_si_cells_of_one_arm(cfg, tmp_path, monkeypatch):
    """The driver read: for BOTH SIs of one arm, generate builds the
    defense from THAT cell, persists under a mode-0700 dir keyed on the CELL
    id (never the bare arm), and writes the identity sidecar with the
    target-forward count from the LLMHFSteered boundary."""
    import sisteer.eval.attacks.bon as bon_mod
    import sisteer.eval.jailbreak as jb
    import sisteer.models as models_mod

    monkeypatch.setattr(models_mod, "load_model_and_tokenizer",
                        lambda *a, **k: (None, None))
    monkeypatch.setattr(two_si_run, "_eval_goals",
                        lambda cfg, *, confirmatory, attack=None:
                        _goals())

    built: list = []

    def fake_build_defense_cell(model, tok, cell, **kw):
        built.append((cell, kw))
        return SimpleNamespace(mode=cell.arm or "none", n_target_forwards=7)

    monkeypatch.setattr(jb, "build_defense_cell", fake_build_defense_cell)

    ran: dict = {}

    def fake_bon_generate(defense, goals, *, out_dir, n_max, sigma, trial,
                          max_new_tokens, batch_size=None):
        ran["out_dir"] = Path(out_dir)
        ran["goals"] = [g.goal_id for g in goals]
        ran["n_max"] = n_max
        return Path(out_dir) / "bon_responses.jsonl"

    monkeypatch.setattr(bon_mod, "bon_generate", fake_bon_generate)

    root = tmp_path / "state"
    arm = next(a for a in cfg["class_a_arms"] if a in ARMS)
    dirs: dict[str, Path] = {}
    for si in cfg["si_ids"]:
        cell_id = f"{si}__{arm}"
        args = SimpleNamespace(cell=cell_id, attack="bon", shard=0, nshards=2,
                               trial=0, confirmatory=False,
                               private_root=str(root), xstest=False)
        two_si_run.cmd_generate(args, cfg)

        cell, kw = built[-1]
        assert cell.id == cell_id and cell.arm == arm and cell.si_id == si
        assert kw["scale"] == cfg["scale"]      # the pre-registered scale
        # the private dir is keyed on the FULL cell id, never the bare arm
        assert ran["out_dir"].name == cell_id != arm
        assert str(ran["out_dir"]).endswith(f"{cfg['private_run']}/{cell_id}")
        assert stat.S_IMODE(ran["out_dir"].stat().st_mode) == 0o700
        assert ran["goals"] == ["g0"]           # shard 0 of 2
        assert ran["n_max"] == cfg["bon"]["n_max"]
        dirs[si] = ran["out_dir"]

        meta = json.loads(
            (ran["out_dir"] / "cell_meta_bon_shard00.json").read_text())
        assert meta["cell_id"] == cell_id
        assert meta["class"] == "A"
        assert meta["si_id"] == si and meta["arm"] == arm
        assert meta["ni_reference_ids"] == [f"prompted__{si}"]
        assert meta["attack"] == "bon"
        assert meta["shard"] == 0 and meta["nshards"] == 2
        assert meta["n_target_forwards"] == 7   # counted at the target boundary
        assert meta["model"] == cfg["model"]
        assert meta["model_revision"] == cfg["model_revision"]

    # the two SIs of ONE arm landed in DISTINCT dirs — the regression
    assert dirs[cfg["si_ids"][0]] != dirs[cfg["si_ids"][1]]


def test_generate_static_family_routes_via_replayable(cfg, tmp_path, monkeypatch):
    """A static/replayable family is dispatched through jailbreak_run's shared
    _run_replayable_family (NO 27B attacker), keyed on cell.id, with a cell_meta
    sidecar recording the attack — the same layout, reused for the seven."""
    import sisteer.eval.jailbreak as jb
    import sisteer.models as models_mod
    import jailbreak_run

    monkeypatch.setattr(models_mod, "load_model_and_tokenizer",
                        lambda *a, **k: (None, None))
    monkeypatch.setattr(two_si_run, "_eval_goals",
                        lambda cfg, *, confirmatory, attack=None:
                        _goals())
    monkeypatch.setattr(jb, "build_defense_cell",
                        lambda model, tok, cell, **kw:
                        SimpleNamespace(mode="none", n_target_forwards=3))
    monkeypatch.setattr(jailbreak_run, "_load_attacker",
                        lambda cfg: (_ for _ in ()).throw(
                            AssertionError("no attacker for a static family")))
    ran: dict = {}

    def fake_family(attack, defense, goals, cfg, out_dir, *, trial=0, resume=True):
        ran.update(attack=attack, out_dir=Path(out_dir),
                   goals=[g.goal_id for g in goals])
        return {"n_goals_run": len(goals), "n_goals_total": len(goals),
                "n_target_forwards_total": 3}

    monkeypatch.setattr(jailbreak_run, "_run_replayable_family", fake_family)

    root = tmp_path / "state"
    args = SimpleNamespace(cell="none", attack="pasttense", shard=0, nshards=1,
                           trial=0, confirmatory=False, private_root=str(root),
                           xstest=False)
    two_si_run.cmd_generate(args, cfg)
    assert ran["attack"] == "pasttense"
    assert str(ran["out_dir"]).endswith(f"{cfg['private_run']}/pasttense/none/shard00")
    meta = json.loads((ran["out_dir"] / "cell_meta.json").read_text())
    assert meta["attack"] == "pasttense" and meta["cell_id"] == "none"
    assert meta["n_target_forwards"] == 3


def test_generate_subenv_family_skips_loudly(cfg, tmp_path, monkeypatch):
    """A sub-env family that skips writes no cell_meta and never loads the
    attacker — the loud-skip path is a clean return, not a crash."""
    import sisteer.eval.jailbreak as jb
    import sisteer.models as models_mod
    import jailbreak_run

    monkeypatch.setattr(models_mod, "load_model_and_tokenizer",
                        lambda *a, **k: (None, None))
    monkeypatch.setattr(two_si_run, "_eval_goals",
                        lambda cfg, *, confirmatory, attack=None:
                        _goals())
    monkeypatch.setattr(jb, "build_defense_cell",
                        lambda model, tok, cell, **kw:
                        SimpleNamespace(mode="none", n_target_forwards=0))
    monkeypatch.setattr(jailbreak_run, "_load_attacker",
                        lambda cfg: (_ for _ in ()).throw(
                            AssertionError("no attacker for a sub-env family")))
    monkeypatch.setattr(
        jailbreak_run, "_run_replayable_family",
        lambda *a, **k: {"skipped": True, "reason": "sub-env not found",
                         "family": "amplegcg_plus", "per_goal": [],
                         "n_goals_run": 0, "n_goals_total": 0})
    root = tmp_path / "state"
    args = SimpleNamespace(cell="none", attack="amplegcg_plus", shard=0, nshards=1,
                           trial=0, confirmatory=False, private_root=str(root),
                           xstest=False)
    two_si_run.cmd_generate(args, cfg)                       # no raise = skip path
    assert not (root / cfg["private_run"] / "amplegcg_plus" / "none" / "shard00"
                / "cell_meta.json").exists()


def test_generate_rejects_non_registry_ids(cfg, tmp_path):
    """The --cell surface accepts registry cell.ids ONLY: an invented id AND a
    bare single-SI arm name (valid in the single-SI driver, ambiguous here)
    both fail loudly before any model load."""
    for bad in ("nonsense__cell", "vector", "prompted"):
        args = SimpleNamespace(cell=bad, attack="bon", shard=0, nshards=1,
                               trial=0, confirmatory=False,
                               private_root=str(tmp_path), xstest=False)
        with pytest.raises(SystemExit, match="unknown --cell"):
            two_si_run.cmd_generate(args, cfg)


def test_check_budget_sidecar_class_a_contract(tmp_path):
    """The generate-side budget guard for Class A: the sidecar is REQUIRED,
    must be complete (exact cell_id, completion_tokens, object_sha256),
    must match the config's budget, and must hash to the actual object
    bytes. Every violation is a loud SystemExit."""
    from sisteer.eval.defense_cell import sha256_file

    obj = tmp_path / "safety_refusal__loreft" / "object.pt"
    obj.parent.mkdir(parents=True)
    obj.write_text("obj")
    cell = SimpleNamespace(id="safety_refusal__loreft", class_="A",
                           object_path=obj)
    cfg = {"completion_tokens": 512}
    meta = obj.parent / "train_budget_meta.json"

    # Class A with NO sidecar: refused — explicit migration, never silent
    with pytest.raises(SystemExit, match="NO budget sidecar"):
        two_si_run._check_budget_sidecar(cell, cfg)

    good = {"cell_id": cell.id, "completion_tokens": 512,
            "object_sha256": sha256_file(obj)}
    meta.write_text(json.dumps(good))
    two_si_run._check_budget_sidecar(cell, cfg)      # complete + matching

    # incomplete (pre-stamp legacy): loud, and the message carries the
    # actual sha so a deliberate hand-migration is one edit
    meta.write_text(json.dumps({"cell_id": cell.id,
                                "completion_tokens": 512}))
    with pytest.raises(SystemExit) as exc:
        two_si_run._check_budget_sidecar(cell, cfg)
    assert "INCOMPLETE" in str(exc.value)
    assert sha256_file(obj) in str(exc.value)

    meta.write_text(json.dumps(good | {"completion_tokens": 256}))
    with pytest.raises(SystemExit, match="completion_tokens=256"):
        two_si_run._check_budget_sidecar(cell, cfg)

    # same budget, DIFFERENT object bytes: the stale pair, refused
    meta.write_text(json.dumps(good))
    obj.write_text("different bytes under the same-budget sidecar")
    with pytest.raises(SystemExit, match="object_sha256 mismatch"):
        two_si_run._check_budget_sidecar(cell, cfg)
    obj.write_text("obj")

    meta.write_text(json.dumps(good | {"cell_id": "other__cell"}))
    with pytest.raises(SystemExit, match="corrupted"):
        two_si_run._check_budget_sidecar(cell, cfg)

    meta.write_text("{not json")
    with pytest.raises(SystemExit, match="unreadable"):
        two_si_run._check_budget_sidecar(cell, cfg)


def test_check_budget_sidecar_class_b_and_baseline_acceptance(tmp_path):
    """Absent sidecars are legitimate ONLY for Class B objects
    (fit_class_b writes no train_budget_meta.json) and for baselines /
    not-yet-trained objects — and a Class B sidecar that IS present must
    still validate."""
    from sisteer.eval.defense_cell import sha256_file

    obj_b = tmp_path / "antidote" / "object.pt"
    obj_b.parent.mkdir(parents=True)
    obj_b.write_text("class-b object")
    cell_b = SimpleNamespace(id="antidote", class_="B", object_path=obj_b)
    cfg = {"completion_tokens": 512}

    two_si_run._check_budget_sidecar(cell_b, cfg)    # absent sidecar: OK for B

    (obj_b.parent / "train_budget_meta.json").write_text(json.dumps(
        {"cell_id": "antidote", "completion_tokens": 256,
         "object_sha256": sha256_file(obj_b)}))
    with pytest.raises(SystemExit, match="completion_tokens=256"):
        two_si_run._check_budget_sidecar(cell_b, cfg)  # present -> validated

    # baselines have no object, hence no budget to validate
    two_si_run._check_budget_sidecar(
        SimpleNamespace(id="none", class_="baseline", object_path=None), cfg)

    # a Class A object NOT YET on disk: the loader's problem, not this
    # check's (there is nothing to mis-pair)
    ghost = SimpleNamespace(id="safety_refusal__loreft", class_="A",
                            object_path=tmp_path / "nope" / "object.pt")
    two_si_run._check_budget_sidecar(ghost, cfg)


def test_generate_refuses_mixed_budget_object_before_model_load(
        cfg, tmp_path, monkeypatch):
    """cmd_generate runs the sidecar check BEFORE loading the model: a
    stale object fails in seconds, never after a GPU allocation."""
    import sisteer.models as models_mod

    from sisteer.eval.defense_cell import sha256_file

    registry = _registry(cfg)
    cell = next(c for c in registry if c.class_ == "A")
    cell.object_path.parent.mkdir(parents=True, exist_ok=True)
    cell.object_path.write_text("obj")
    (cell.object_path.parent / "train_budget_meta.json").write_text(
        json.dumps({"cell_id": cell.id,
                    "completion_tokens": int(cfg["completion_tokens"]) + 1,
                    "object_sha256": sha256_file(cell.object_path)}))

    monkeypatch.setattr(models_mod, "load_model_and_tokenizer",
                        lambda *a, **k: pytest.fail(
                            "the model must not load for a mixed-budget object"))
    args = SimpleNamespace(cell=cell.id, attack="bon", shard=0, nshards=1,
                           trial=0, confirmatory=False,
                           private_root=str(tmp_path / "state"), xstest=False)
    with pytest.raises(SystemExit, match="refusing to mix"):
        two_si_run.cmd_generate(args, cfg)


def test_generate_xstest_resumes_by_goal_id(cfg, tmp_path, monkeypatch):
    """The per-cell xstest pass persists to the cell-id dir and RESUMES by
    goal_id: a second run generates nothing new."""
    import sisteer.eval.jailbreak as jb
    import sisteer.eval.jailbreak_data as jd
    import sisteer.models as models_mod

    monkeypatch.setattr(models_mod, "load_model_and_tokenizer",
                        lambda *a, **k: (None, None))
    monkeypatch.setattr(jd, "load_xstest_safe", lambda **kw: _xs_goals())

    calls = {"n": 0}

    def fake_build_defense_cell(model, tok, cell, **kw):
        def generate(texts, max_new_tokens):
            calls["n"] += len(texts)
            return ["a benign completion"] * len(texts)
        return SimpleNamespace(mode="none", n_target_forwards=0,
                               generate=generate)

    monkeypatch.setattr(jb, "build_defense_cell", fake_build_defense_cell)

    root = tmp_path / "state"
    args = SimpleNamespace(cell="none", attack="bon", shard=0, nshards=1,
                           trial=0, confirmatory=False, private_root=str(root),
                           xstest=True)
    two_si_run.cmd_generate(args, cfg)
    jsonl = root / cfg["private_run"] / "none" / "xstest_responses.jsonl"
    assert len(jsonl.read_text().splitlines()) == len(_xs_goals())
    assert (jsonl.parent / "cell_meta_xstest.json").exists()
    n_first = calls["n"]
    two_si_run.cmd_generate(args, cfg)          # resume: everything covered
    assert calls["n"] == n_first                # no regeneration
    assert len(jsonl.read_text().splitlines()) == len(_xs_goals())


# --------------------------------------------------------------------------- #
# aggregate — completeness first, refusals first-class                          #
# --------------------------------------------------------------------------- #


def test_aggregate_raises_on_incomplete_matrix(cfg, tmp_path, monkeypatch):
    """A cell with no transcripts and no recorded refusal is a silently-dropped
    column — the aggregate raises (naming it) BEFORE any grading."""
    _write_manifest(cfg)
    root = tmp_path / "state"
    registry = _registry(cfg)
    # transcripts for the baselines only — every object cell is missing
    _store(cfg, root, registry,
           skip={c.id for c in registry if c.arm is not None})
    missing = next(c.id for c in registry if c.class_ == "A")
    with pytest.raises(ValueError, match=re.escape(missing)):
        _run_aggregate(cfg, root, monkeypatch)


def test_aggregate_accepts_refused_with_reason(cfg, tmp_path, monkeypatch):
    """A gate-refused cell (recorded in its object-store meta.json) is a
    first-class outcome: the aggregate publishes without it and reports the
    reason, instead of failing or silently dropping the column."""
    _write_manifest(cfg)
    root = tmp_path / "state"
    registry = _registry(cfg)
    refused_cell = next(c for c in registry if c.class_ == "A")
    reason = "gate refusal: positive control moved nothing on held-in"
    meta_dir = Path(refused_cell.object_path).parent
    meta_dir.mkdir(parents=True, exist_ok=True)
    (meta_dir / "meta.json").write_text(json.dumps({"refused": reason}))
    _store(cfg, root, registry, skip={refused_cell.id})

    agg = _run_aggregate(cfg, root, monkeypatch)
    assert agg["refused"] == {refused_cell.id: reason}
    assert refused_cell.id not in agg["cells"]
    # every OTHER cell is present — nothing silently dropped
    assert set(agg["cells"]) == {c.id for c in registry} - {refused_cell.id}


def test_aggregate_refuses_without_pool_manifest(cfg, tmp_path, monkeypatch):
    """No pool manifest, no aggregate — the pool regime is never
    inferred, and the refusal happens before any grading spend."""
    root = tmp_path / "state"
    _store(cfg, root, _registry(cfg))
    with pytest.raises(SystemExit, match="pool manifest"):
        _run_aggregate(cfg, root, monkeypatch)


# --------------------------------------------------------------------------- #
# aggregate — the structure the statistics consume                             #
# --------------------------------------------------------------------------- #


def test_aggregate_emits_controls_ni_refs_and_pool_regime(cfg, tmp_path,
                                                          monkeypatch):
    _write_manifest(cfg)   # matched regime, both pools 657
    root = tmp_path / "state"
    registry = _registry(cfg)
    _store(cfg, root, registry)
    agg = _run_aggregate(cfg, root, monkeypatch)

    # -- pool regime at top level, with BOTH pool sizes -----------------------
    pr = agg["pool_regime"]
    assert pr["regime"] == "matched_intersection"
    assert pr["pool_sizes"] == {si: 657 for si in cfg["si_ids"]}
    assert pr["headline_confounded_by_pool_composition"] is False

    # -- every registry cell present, keyed on cell.id ------------------------
    assert set(agg["cells"]) == {c.id for c in registry}
    assert agg["matrix_counts"]["total"] == len(registry)

    six = cfg["controls"]          # the four slots since 2026-09-11 (alpaca_eval, wmdp retired)
    assert len(six) == 4
    for cell in registry:
        entry = agg["cells"][cell.id]
        assert entry["class"] == cell.class_
        assert entry["si_id"] == cell.si_id and entry["arm"] == cell.arm
        # -- ALL declared control slots, per cell; xstest filled, the
        #    not-yet-run controls explicitly pending (never absent). The
        #    served JSON is written sort_keys=True, so compare as sets. ------
        assert set(entry["controls"]) == set(six)
        assert entry["controls"]["xstest"]["full_compliance_frac"] == 1.0
        for c in six:
            if c != "xstest":
                assert entry["controls"][c] == "pending_control_not_yet_run"
        # -- the NI reference seam ----------------------------------------
        if cell.class_ == "A":
            assert entry["ni_reference_ids"] == [f"prompted__{cell.si_id}"]
            assert entry["training_pool"] == {
                "regime": "matched_intersection", "size": 657}
        elif cell.class_ == "B":
            assert set(entry["ni_reference_ids"]) == {
                f"prompted__{si}" for si in cfg["si_ids"]}
            assert "none" not in entry["ni_reference_ids"]
            assert entry["training_pool"] == {
                "regime": "si_free_harmful_benign_fit_once"}
        else:
            assert entry["ni_reference_ids"] == []
            assert entry["training_pool"] is None
        # attack metrics landed under the cell id
        assert entry["attack_metrics"]["n_goals"] == len(_goals())

    # -- the over_refusal_comparisons seam: (cell, reference, control) ---------
    comps = agg["over_refusal_comparisons"]
    n_a = sum(1 for c in registry if c.class_ == "A")
    n_b = sum(1 for c in registry if c.class_ == "B")
    # Class B compares against EVERY prompted arm — one per config SI
    # (three since stage 3 added si_irrelevant).
    assert len(comps) == (n_a + len(cfg["si_ids"]) * n_b) * len(six)
    b_cell = next(c for c in registry if c.class_ == "B")
    b_refs = {c["reference_id"] for c in comps
              if c["cell_id"] == b_cell.id and c["control"] == "xstest"}
    assert b_refs == {f"prompted__{si}" for si in cfg["si_ids"]}
    assert all(c["cell_id"] not in ("none", "prompted__safety_refusal",
                                    "prompted__si_constitution")
               for c in comps)


def test_aggregate_pool_regime_native_fallback_is_flagged(cfg, tmp_path,
                                                          monkeypatch):
    """The fallback: native per-SI pools publish with the regime stated,
    UNEQUAL sizes carried per cell, and the confounded-headline flag set —
    flagged in this case too, never silently."""
    sizes = dict(zip(cfg["si_ids"], (657, 412, 303)))
    _write_manifest(cfg, regime="native_per_si", sizes=sizes)
    root = tmp_path / "state"
    registry = _registry(cfg)
    _store(cfg, root, registry)
    agg = _run_aggregate(cfg, root, monkeypatch)

    pr = agg["pool_regime"]
    assert pr["regime"] == "native_per_si"
    assert pr["pool_sizes"] == sizes
    assert pr["headline_confounded_by_pool_composition"] is True
    for cell in registry:
        if cell.class_ == "A":
            assert agg["cells"][cell.id]["training_pool"] == {
                "regime": "native_per_si", "size": sizes[cell.si_id]}


def test_load_pool_regime_validates_the_manifest(cfg):
    # unknown regime
    _write_manifest(cfg, regime="subsample_to_common_size")  # rejected
    with pytest.raises(SystemExit, match="regime"):
        two_si_run._load_pool_regime(cfg)
    # pool_sizes must cover exactly the config si_ids
    _write_manifest(cfg, sizes={"safety_refusal": 657})
    with pytest.raises(SystemExit, match="pool_sizes"):
        two_si_run._load_pool_regime(cfg)
    # a matched intersection with UNEQUAL sizes is a contradiction
    _write_manifest(cfg, sizes=dict(zip(cfg["si_ids"], (657, 412, 303))))
    with pytest.raises(SystemExit, match="IDENTICAL"):
        two_si_run._load_pool_regime(cfg)


# --------------------------------------------------------------------------- #
# the config — complete, and pinned to the same data as the single-SI side      #
# --------------------------------------------------------------------------- #


def test_config_carries_every_section_the_driver_reads():
    cfg = yaml.safe_load(TWO_SI_CONFIG.read_text())
    s2 = yaml.safe_load(STAGE2_CONFIG.read_text())
    # this config's own identity keys — three SIs and the FOURTEEN stage-3
    # Class A arms, with the 2026-08-21 amendments: alphakl_fisher pulled
    # from the TRAINED list only — a mid-list removal that shifted
    # diffmean/bidpo's arm-major indices, safe because nothing was in
    # flight — vector + bidpo_lora appended LAST to complete the DPO 2x2,
    # then token_faithful appended LAST after them, keeping the registered
    # name, then the all-layer LoRA arm appended LAST on 2026-08-28 so the
    # scale-12 campaign's staged objects resolve as cells — registered as
    # lora_r1_all and RENAMED to lora_r4 the same day: the staged objects
    # are rank 4, "all" is implicit, and the staged `<si>__lora_r1_all/`
    # dirs keep loading via the artifact-layer alias,
    # methods.LEGACY_ARM_ALIASES. vector_all stays OUT of the matrix but
    # registered; cast left the REGISTRY entirely 2026-08-20 alongside
    # circuit_breaker.
    assert cfg["si_ids"] == ["safety_refusal", "si_constitution",
                             "si_irrelevant"]
    assert cfg["class_a_arms"] == [
        "psr_kl_all", "psr_mse_all", "lora_r1", "loreft", "lofit",
        "token_bos", "alphakl_null", "alphakl_soft",
        "diffmean", "bidpo", "vector", "bidpo_lora", "token_faithful",
        "lora_r4"]
    assert "cast" not in cfg["class_a_arms"]
    assert "vector_all" not in cfg["class_a_arms"]
    assert set(cfg["class_a_arms"]) < set(CLASS_A_ARMS)
    # circuit_breaker left the registry ENTIRELY on 2026-08-20, alongside
    # cast: no ArmSpec, not in the module defaults; its implementation,
    # load_object hydration, and fitted artifact survive. The config's
    # Class B set now equals the module default.
    # caa_static_r (added 2026-08-21): the CAA / diff-in-means floor, the
    # an earlier development branch's static_r fit deployment-matched to alphasteer
    # (extraction/caa_static_r.py) — APPENDED LAST, order load-bearing.
    assert cfg["class_b_arms"] == ["antidote", "alphasteer", "caa_static_r"]
    assert "circuit_breaker" not in cfg["class_b_arms"]
    assert set(cfg["class_b_arms"]) == set(CLASS_B_ARMS)
    assert cfg["defenses_dir"] == "out/two_si/defenses/cells"
    assert cfg["pool_manifest"] == "out/two_si/pool_manifest.json"
    # dataset pins REUSED from the single-SI config (the labelled store the
    # pairs join against was generated at these shas — moving one breaks the
    # same-prompt join silently)
    for ds in ("harmbench", "jbb", "xstest"):
        assert cfg["datasets"][ds]["revision"] == s2["datasets"][ds]["revision"]
        assert cfg["datasets"][ds]["repo"] == s2["datasets"][ds]["repo"]
    # the category holdout matches the store's split (seed 0 / frac 0.33)
    assert cfg["category_holdout"] == s2["category_holdout"]
    # attack budgets: same forward cap; decode budget == the shared
    # completion cap so supervision-vs-decode volume is never a hidden diff
    assert cfg["attacks"]["forward_cap"] == s2["attacks"]["forward_cap"]
    assert cfg["attacks"]["target_max_new_tokens"] == cfg["completion_tokens"]
    assert cfg["bon"]["max_new_tokens"] == cfg["completion_tokens"]
    assert cfg["attacker"]["model"] == s2["attacker"]["model"]
    # prereg: the endpoint threshold sits at the grid midpoint
    assert cfg["prereg"]["threshold_grid"][1] == 0.5
    assert cfg["prereg"]["primary_contrast"] == ["si_constitution",
                                                 "safety_refusal"]
    assert cfg["prereg"]["confirmatory_goals"] == 100
    # outputs: own store, disjoint from the single-SI config's
    assert cfg["out_dir"] == "out/two_si"
    assert cfg["private_run"] == "two_si_v1" != s2["private_run"]
    assert cfg["wandb"] is False
    # preserved shared knobs
    assert cfg["max_len"] == 8192 and cfg["completion_tokens"] == 512
    assert len(cfg["controls"]) == 4          # alpaca_eval and wmdp retired 2026-09-11


def test_registry_from_cfg_honors_config_class_arms(tmp_path):
    """The additive registry_from_cfg extension: config class_a_arms /
    class_b_arms override the module defaults, so configs/two_si.yaml is the
    submission-time record of the matrix; absent keys keep the defaults."""
    cfg = {"si_ids": ["safety_refusal", "si_constitution"],
           "defenses_dir": str(tmp_path),
           "class_a_arms": ["psr_kl_all"], "class_b_arms": ["antidote"]}
    reg = registry_from_cfg(cfg)
    assert {c.id for c in reg} == {
        "none", "prompted__safety_refusal", "prompted__si_constitution",
        "safety_refusal__psr_kl_all", "si_constitution__psr_kl_all",
        "antidote"}
    # absent keys -> the module defaults (backwards compatible)
    default_reg = registry_from_cfg({"si_ids": cfg["si_ids"],
                                     "defenses_dir": str(tmp_path)})
    expected_a = [a for a in CLASS_A_ARMS if a in ARMS]
    assert sum(1 for c in default_reg if c.class_ == "A") == 2 * len(expected_a)


# --------------------------------------------------------------------------- #
# cells — the sbatch array decode goes through cell_for_array_index             #
# --------------------------------------------------------------------------- #


def test_cells_decode_matches_cell_for_array_index(cfg, capsys):
    registry = _registry(cfg)
    nshards = 3
    for task_id in (0, 1, nshards, len(registry) * nshards - 1):
        two_si_run.cmd_cells(
            SimpleNamespace(task_id=task_id, nshards=nshards), cfg)
        out = capsys.readouterr().out.strip()
        cell, shard = cell_for_array_index(registry, task_id, nshards)
        assert out == f"{cell.id} {shard}"
    # a stale --array bound raises (the sbatch guard turns that into FATAL)
    with pytest.raises(IndexError):
        two_si_run.cmd_cells(
            SimpleNamespace(task_id=len(registry) * nshards, nshards=nshards),
            cfg)


def test_cells_listing_derives_counts(cfg, capsys):
    registry = _registry(cfg)
    two_si_run.cmd_cells(SimpleNamespace(task_id=None, nshards=1), cfg)
    out = capsys.readouterr().out
    for i, cell in enumerate(registry):
        assert f"{i:3d}  {cell.id}" in out
    assert f"# {len(registry)} cells" in out


# --------------------------------------------------------------------------- #
# the sbatch script — cluster hard rules + the cell-id decode                   #
# --------------------------------------------------------------------------- #


def test_two_si_sbatch_hygiene_and_decode():
    txt = (REPO / "scripts" / "sbatch" / "two_si.sbatch").read_text()
    # session-unique tt-twosi-* job name (cancel by JOB ID only)
    assert re.search(r"^#SBATCH --job-name=tt-twosi-", txt, re.M)
    assert "#SBATCH --qos=high" in txt
    assert "#SBATCH --gres=gpu:1" in txt
    assert "#SBATCH --mem=128G" in txt
    assert not re.search(r"^#SBATCH --mem=0", txt, re.M)     # hard rule 2
    assert not re.search(r"^#SBATCH --export", txt, re.M)    # hard rule 1
    assert "#SBATCH --open-mode=append" in txt
    assert "#SBATCH --no-requeue" in txt
    assert "#SBATCH --signal=B:TERM@120" in txt
    # mode-0700 private log dir, pre-created before submission
    assert "exp/jailbreak/_logs" in txt
    assert "install -d -m 700" in txt
    # shared scaffolding + JB4 containment
    assert "stage2_common.sh" in txt and "s2_init" in txt
    assert "WANDB_DISABLED=true" in txt and "WANDB_CONSOLE=off" in txt
    # the decode runs through the registry (cell_for_array_index), not a
    # shell copy of the cell list
    assert "two_si_run.py cells" in txt
    assert "--task-id" in txt and "cell_for_array_index" in txt
    assert "two_si_run.py generate" in txt
    assert "--cell \"$CELL\"" in txt
    # the sha submit pin appears in the usage line
    assert "--comment=sha:" in txt
    assert "CUDA_VISIBLE_DEVICES" not in txt          # hard rule 3


def test_pool_gen_sbatch_families_and_hygiene(cfg):
    """The pool-gen array runs one family per index over the static families
    that actually need per-family `none`-condition generation via run_static:
    the committed static_families MINUS bon (already in the labelled store),
    the two sub-env families (skipped-not-attempted, E13), and any non-transform
    CORPUS family (wildjailbreak_harmful is an already-adversarial corpus, not
    a JB_ATTACKS transform, so its none+SI generation goes through the
    pool_sample.py sampling lane, never this per-family run_static array).
    Cluster hard rules hold."""
    from sisteer.eval.attacks import JB_ATTACKS, SUBENV_ATTACKS

    txt = (REPO / "scripts" / "sbatch" / "two_si_pool_gen.sbatch").read_text()
    m = re.search(r"^FAMILIES=\(([^)]*)\)", txt, re.M)
    assert m, "no FAMILIES=(...) array in two_si_pool_gen.sbatch"
    fams = m.group(1).split()
    committed = list(cfg["pool"]["static_families"])
    # only genuine JB_ATTACKS transforms dispatch through run_static; a corpus
    # family (wildjailbreak_harmful) is out — it is not in JB_ATTACKS.
    expected = [f for f in committed
                if f != "bon" and f not in SUBENV_ATTACKS and f in JB_ATTACKS]
    assert fams == expected                                   # in config order
    assert all(f in JB_ATTACKS for f in fams)                 # every one dispatches
    assert not (set(fams) & SUBENV_ATTACKS)                   # never a sub-env family
    # the corpus family is deliberately absent from this per-family array
    assert "wildjailbreak_harmful" not in fams
    # cluster hard rules + JB4 containment + the immutable-checkout guard
    assert not re.search(r"^#SBATCH --mem=0", txt, re.M)      # hard rule 2
    assert not re.search(r"^#SBATCH --export", txt, re.M)     # hard rule 1
    assert "CUDA_VISIBLE_DEVICES" not in txt                  # hard rule 3
    assert "#SBATCH --open-mode=append" in txt
    assert "#SBATCH --no-requeue" in txt
    assert "#SBATCH --mem=128G" in txt
    assert "stage2_common.sh" in txt and "s2_init" in txt     # RESULT sentinel + SHA guard
    assert "WANDB_DISABLED=true" in txt and "WANDB_CONSOLE=off" in txt
    # Cluster B needs an explicit shared private-root (no /home fallback)
    assert "--private-root" in txt
    assert "--cell none" in txt and "two_si_run.py generate" in txt


def test_two_si_sbatch_array_bound_matches_registry(cfg):
    """The documented --array bound derives from the LIVE registry (never a
    literal): total cells x NSHARDS - 1, and every index decodes to a unique
    (cell, shard)."""
    registry = _registry(cfg)
    nshards = 4
    seen = set()
    for idx in range(len(registry) * nshards):
        cell, shard = cell_for_array_index(registry, idx, nshards)
        seen.add((cell.id, shard))
    assert len(seen) == len(registry) * nshards
    # the two SI versions of one arm occupy DISTINCT indices
    a_cells = [c for c in registry if c.class_ == "A"]
    arm = a_cells[0].arm
    idxs = [i for i, c in enumerate(registry) if c.arm == arm and c.class_ == "A"]
    assert len(idxs) == len(set(idxs)) == len(cfg["si_ids"])


# --------------------------------------------------------------------------- #
# bidpo trains on the STRICT (genuine-harmful-compliance) subset only          #
# --------------------------------------------------------------------------- #


def _pair(req):
    """A synthetic Pair carrying a requirement_a label (no real store text)."""
    from sisteer.data.pairs import Pair
    return Pair(prompt=f"SYNTHETIC prompt {req} 11111111",
                pos="SYNTHETIC refusal styled bbbbbbbb",
                neg="SYNTHETIC undefended completion cccccccc",
                instruction_id="safety_refusal", requirement_a=req)


def test_pairs_for_arm_bidpo_keeps_only_strict():
    """bidpo (DPO preference) trains on the STRICT subset alone so its rejected
    side is a genuine harmful compliance, never a relaxed refusal.
    Every other arm gets the full, unfiltered pool."""
    import train_two_si_objects as tt

    pool = [_pair("strict"), _pair("relaxed"), _pair("benign"), _pair("strict")]
    got = tt.pairs_for_arm("bidpo", pool)
    assert [p.requirement_a for p in got] == ["strict", "strict"]
    # a non-preference arm is unfiltered (identity, not a copy)
    assert tt.pairs_for_arm("diffmean", pool) is pool


def test_pairs_for_arm_bidpo_raises_without_labels():
    """A pool with no requirement_a labels (pre-DECISION-1, or the native
    regime) gives bidpo no way to know which negs are real compliances — refuse
    loudly rather than train on a possible refusal."""
    import train_two_si_objects as tt
    from sisteer.data.pairs import Pair

    unlabelled = [Pair(prompt="SYNTHETIC p 22222222", pos="a", neg="b",
                       instruction_id="safety_refusal")]
    with pytest.raises(ValueError, match="requirement_a"):
        tt.pairs_for_arm("bidpo", unlabelled)


def test_pairs_for_arm_bidpo_raises_when_no_strict_pairs():
    """Labels present but ZERO strict: the base model complied with nothing, so
    bidpo has no valid rejected side — refuse and say to raise the yield / K."""
    import train_two_si_objects as tt

    with pytest.raises(ValueError, match="ZERO strict"):
        tt.pairs_for_arm("bidpo", [_pair("relaxed"), _pair("benign")])
