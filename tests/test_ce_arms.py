"""The two document-CE arms (sdf-base-steering §5.2/§5.3) through the
registry and the production driver: registration + lockstep exclusion, the
config-shape guard, the CE-native floor guard's fit_record fields and
refuse-to-publish semantics, the PROTECTED heldout/early_stop seams, the
planted dead-gate refusal with per-layer gate activity recorded, and the
document budget-provenance schema (legacy SI-pair sidecars byte-identical).

CPU-only: tiny model, synthetic doc-shaped pairs — never agent-staged pools.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import train_two_si_objects as t2

from sisteer.data.pairs import Pair, save_pairs
from sisteer.eval.defense_cell import object_path_for
from sisteer.eval.train_budget import (DOCUMENT_TOKENS_KIND,
                                       document_budget_problems,
                                       resolve_budget_kind)
from sisteer.extraction.lockstep import (lockstep_compatible,
                                         lockstep_incompatible_reason)
from sisteer.extraction.multilayer import MultiLayerLoRAObject
from sisteer.extraction.psr import PSRObject
from sisteer.methods import ARMS, ArmSpec, S_TWO_SI

SI = "constv2-mix-1k"
TINY = "hf-internal-testing/tiny-random-LlamaForCausalLM"


def _docs():
    rows = [("Sys scaffold\n\n", "A doc about trees and rivers."),
            ("", "Story body with no masked prefix at all here."),
            ("Second sys\n\n", "Another advice body follows the user turn."),
            ("", "Yet another tiny story body.")]
    return [Pair(prompt=p, pos=pos, neg="", instruction_id=SI)
            for p, pos in rows]


def _held():
    return [Pair(prompt="Held sys\n\n", pos="Held-out document body one.",
                 neg="", instruction_id=SI),
            Pair(prompt="", pos="Held-out story body two, somewhat longer.",
                 neg="", instruction_id=SI)]


def _cfg(tmp_path, arms=("psr_ce_all", "lora_r4_ce"), **over):
    hp = tmp_path / "pairs_constv2-heldout.jsonl"
    if not hp.exists():
        save_pairs(_held(), hp)
    cfg = {
        "layer": 1, "max_len": 256,
        "data_mode": "documents",
        "training_budget_kind": "document_tokens",
        "completion_tokens": None,
        "model": TINY,
        "batch_size_by_arm": {a: 2 for a in arms},
        "train_kwargs": {
            "default": {"epochs": 2, "seed": 0, "holdout_frac": 0.0,
                        "heldout_pool": str(hp), "probe_n": 2,
                        "early_stop_min_epochs": 1,
                        "early_stop_patience": 2},
            "psr_ce_all": {"lr": 1e-3, "gate_reg": 0.1,
                           "floor_ratio_max": None,
                           "ce_floor_ratio_max": 1.1},
            "lora_r4_ce": {"lr": 1e-3, "lora_rank": 4,
                           "floor_ratio_max": None,
                           "ce_floor_ratio_max": 1.1},
        },
        "si_ids": [SI],
    }
    cfg.update(over)
    return cfg


def _manifest_doc(pool: Path, cell: str = SI, **over) -> dict:
    """What the pool builder records beside a pool: the cell and the pool
    FILE's sha256 (the resolver validates both), plus budget fields."""
    import hashlib

    doc = {"cell": cell, "budget": 1000, "seed_salts": ["synthetic"],
           "pool_sha256": hashlib.sha256(pool.read_bytes()).hexdigest()}
    doc.update(over)
    return doc


def _pool_with_manifest(tmp_path):
    pool = tmp_path / f"pairs_{SI}.jsonl"
    save_pairs(_docs(), pool)
    (Path(str(pool) + ".manifest.json")).write_text(
        json.dumps(_manifest_doc(pool)))
    return pool


def test_sample_manifest_resolver_finds_the_staged_naming(tmp_path, capsys):
    """``_sample_manifest_sha_for`` resolves the STAGED pools' real naming —
    ``pairs_<cell>.manifest.json`` beside ``pairs_<cell>.jsonl`` (the
    stem-swapped spelling the pool builder actually ships; verified against
    the staged campaign pools 2026-09-01) — as well as the appended
    ``<pool>.jsonl.manifest.json`` spelling, and records a LOUD null when
    neither is staged. Since 2026-09-10 it is the shared resolver
    (sisteer.data.sample_manifest) and VALIDATES: the manifest's cell is the
    SI and its pool_sha256 is the pool file's bytes; an existing manifest
    that is empty, another cell's or another pool's is a SystemExit, and
    the other spelling is never consulted. Fixture copies the naming
    convention only, never the staged files."""
    import hashlib

    pool = tmp_path / f"pairs_{SI}.jsonl"
    save_pairs(_docs(), pool)
    manifest = tmp_path / f"pairs_{SI}.manifest.json"     # the real convention
    manifest.write_text(json.dumps(_manifest_doc(pool)))
    assert t2._sample_manifest_sha_for(pool, cell=SI) == hashlib.sha256(
        manifest.read_bytes()).hexdigest()

    # the appended spelling also resolves (and wins when both exist)
    appended = Path(str(pool) + ".manifest.json")
    appended.write_text(json.dumps(_manifest_doc(pool, spelling="appended")))
    assert t2._sample_manifest_sha_for(pool, cell=SI) == hashlib.sha256(
        appended.read_bytes()).hexdigest()

    # the preferred spelling EMPTY beside a valid fallback: refused, not
    # skipped (the preflight and the trainer used to disagree here)
    appended.write_text("")
    with pytest.raises(SystemExit, match="is empty"):
        t2._sample_manifest_sha_for(pool, cell=SI)
    appended.unlink()
    # another cell's manifest; a stale one beside a replaced pool
    with pytest.raises(SystemExit, match="is cell"):
        t2._sample_manifest_sha_for(pool, cell="constv2-mix-100k")
    manifest.write_text(json.dumps(_manifest_doc(pool, pool_sha256="0" * 64)))
    with pytest.raises(SystemExit, match="stale manifest"):
        t2._sample_manifest_sha_for(pool, cell=SI)
    # a manifest without the pool's hash cannot be placed on a pool
    manifest.write_text(json.dumps({"cell": SI, "budget": 1000}))
    with pytest.raises(SystemExit, match="no pool_sha256"):
        t2._sample_manifest_sha_for(pool, cell=SI)

    # no manifest staged: null — a stated absence, said out loud
    bare = tmp_path / "pairs_bare.jsonl"
    save_pairs(_docs(), bare)
    capsys.readouterr()
    assert t2._sample_manifest_sha_for(bare, cell=SI) is None
    assert "no sample manifest" in capsys.readouterr().err


# -- registration ---------------------------------------------------------------


def test_registry_entries_appended_last_two_si_only_and_solo():
    order = list(ARMS)
    # a POSITIONAL pin, no longer a TAIL pin: the sdf-base-steering DERIVED
    # arms (2026-09-06) are appended after these two, which is the
    # append-only rule working — the CE pair's own slots are unmoved.
    assert order[36:38] == ["psr_ce_all", "lora_r4_ce"]
    for arm in ("psr_ce_all", "lora_r4_ce"):
        assert ARMS[arm].suites == frozenset({S_TWO_SI})
        # NOT lockstep-compatible (CE on document tokens — no teacher
        # forward at all), and the exclusion reports a reason.
        assert not lockstep_compatible(arm)
        assert lockstep_incompatible_reason(arm)


def test_builders_produce_the_matched_objects(tiny_llama):
    """Object geometry is IDENTICAL to the instruct campaign's arms (the
    spec's object-matched transfer requirement): psr_ce_all builds exactly
    psr_kl_all's object, lora_r4_ce exactly lora_r4's."""
    model, tok = tiny_llama
    ce_psr = ARMS["psr_ce_all"].build(model, tok, 1)
    kl_psr = ARMS["psr_kl_all"].build(model, tok, 1)
    assert isinstance(ce_psr, PSRObject)
    assert ce_psr.layers == kl_psr.layers
    assert ce_psr.kind == kl_psr.kind
    ce_lora = ARMS["lora_r4_ce"].build(model, tok, 1, rank=4)
    r4_lora = ARMS["lora_r4"].build(model, tok, 1, rank=4)
    assert isinstance(ce_lora, MultiLayerLoRAObject)
    assert ce_lora.layers == r4_lora.layers
    assert ce_lora.kind == r4_lora.kind


# -- config-shape guard -----------------------------------------------------------


@pytest.mark.parametrize("mutate, match", [
    (lambda c: c.update({"data_mode": "pairs"}), "data_mode: documents"),
    (lambda c: c["train_kwargs"]["default"].update({"positions": "response"}),
     "positions: all"),
    (lambda c: c["train_kwargs"]["psr_ce_all"].update({"temperature": 2.0}),
     "temperature"),
    (lambda c: c["train_kwargs"]["psr_ce_all"].update(
        {"teacher_prefix_cache": True}), "teacher_prefix_cache"),
    (lambda c: c["train_kwargs"]["psr_ce_all"].update(
        {"floor_ratio_max": 3.0}), "floor_ratio_max"),
])
def test_config_shape_guard_refuses_si_shaped_ce_configs(tmp_path, mutate,
                                                         match):
    cfg = _cfg(tmp_path)
    mutate(cfg)
    with pytest.raises(SystemExit, match=match):
        t2.refuse_ce_misconfiguration(cfg, ["psr_ce_all", "lora_r4_ce"])


def test_guard_refuses_non_ce_arms_on_document_pools(tmp_path):
    cfg = _cfg(tmp_path)
    with pytest.raises(SystemExit, match="not CE arms"):
        t2.refuse_ce_misconfiguration(cfg, ["psr_ce_all", "psr_kl_all"])


def test_guard_refuses_heldout_pool_with_holdout_frac(tmp_path):
    cfg = _cfg(tmp_path)
    cfg["train_kwargs"]["default"]["holdout_frac"] = 0.1
    with pytest.raises(SystemExit, match="mutually exclusive"):
        t2.refuse_ce_misconfiguration(cfg, ["lora_r4_ce"])


def test_guard_passes_the_campaign_shape_and_null_floor_is_fine(tmp_path):
    cfg = _cfg(tmp_path)  # floor_ratio_max: null explicitly — allowed
    t2.refuse_ce_misconfiguration(cfg, ["psr_ce_all", "lora_r4_ce"])
    t2.refuse_ce_misconfiguration({}, ["psr_kl_all"])  # non-CE: untouched


def test_guard_runs_inside_train_si_objects(tiny_llama, tmp_path):
    model, tok = tiny_llama
    cfg = _cfg(tmp_path)
    cfg["data_mode"] = "pairs"
    with pytest.raises(SystemExit, match="data_mode: documents"):
        t2.train_si_objects(model, tok, SI, _docs(), ["lora_r4_ce"],
                            cfg=cfg, save_dir=None)


# -- the full driver path: fit_record fields, audit, publish ----------------------


def test_ce_fields_land_in_fit_record_and_cell_publishes(tiny_llama,
                                                         tmp_path):
    """Both CE arms fit end-to-end through train_si_objects: the CE floor
    verdict fields land in fit_record with floor_ratio_max honestly null
    (audit_cell accepts — the bar field is present), the doc provenance +
    document budget schema land in the sidecar, per-layer gate activity is
    recorded for the PSR fit, and the cells publish COMPLETE."""
    model, tok = tiny_llama
    pool = _pool_with_manifest(tmp_path)
    out = tmp_path / "defenses"
    outcome = t2.train_si_objects(
        model, tok, SI, _docs(), ["psr_ce_all", "lora_r4_ce"],
        cfg=_cfg(tmp_path), save_dir=out, pairs_path=str(pool))
    assert outcome.ok, {a: (r.error_type, r.error)
                        for a, r in outcome.failed.items()}
    for arm in ("psr_ce_all", "lora_r4_ce"):
        cell = out / f"{SI}__{arm}"
        assert (cell / "COMPLETE").exists()
        meta = json.loads((cell / "train_budget_meta.json").read_text())
        fr = meta["fit_record"]
        # the CE floor verdict, refuse-to-publish fields
        assert fr["floor_ratio_max"] is None      # honestly null, recorded
        assert fr["ce_floor_ratio_max"] == 1.1
        assert fr["heldout_ce_noop"] > 0
        assert fr["heldout_ce_active"] > 0
        assert fr["ce_ratio"] == pytest.approx(
            fr["heldout_ce_active"] / max(fr["heldout_ce_noop"], 1e-2))
        assert fr["ce_ratio"] <= 1.1
        # the SI-teacher probe was SKIPPED, not run with a null bar
        assert "final_probe_loss" not in fr
        assert "noop_floor" not in fr
        # early stopping ran (mandatory), lockstep-shaped record
        assert fr["early_stop"]["supported"] is True
        assert isinstance(fr["heldout_curve"], list)
        assert fr["best_epoch"] >= 1
        # doc provenance + document budget schema
        assert meta["data_mode"] == "documents"
        assert meta["chat_template_bypassed"] is True
        assert meta["sample_manifest_sha"]
        assert meta["heldout_pool_sha"]
        assert meta["training_budget_kind"] == DOCUMENT_TOKENS_KIND
        assert meta["sequence_max_len"] == 256
        assert meta["completion_tokens"] is None
        assert meta["positions_at_fit"] == "all"
        assert document_budget_problems(meta) == []
    psr_fr = json.loads((out / f"{SI}__psr_ce_all" /
                         "train_budget_meta.json").read_text())["fit_record"]
    assert set(psr_fr["gate_liveness"]) == {"0", "1"}  # per-layer, every fit
    assert all(v > 0 for v in psr_fr["gate_liveness"].values())


def test_ce_lane_shared_stream_names_displacement_not_heldout_loss(
        tiny_llama, tmp_path):
    """The SHARED metric stream (W&B + the reports tee) of a CE fit names
    the dual-KL probe for what it measures with no teacher: the object's
    displacement from base streams as probe/<arm>/displacement_kl_vs_base
    (mid-fit, via the trainer seam) + final/<arm>/displacement_kl_vs_base
    and is mirrored to the top-level displacement_kl_vs_base panel; the
    teacher-relative names (probe/<arm>/heldout_loss — a false held-out
    loss — residual_kl, both *_ratio keys, the kl_ratio_vs_unsteered
    mirror) never appear. The REAL held-out CE (<arm>/heldout_loss, the
    trainer's) still streams, the CE floor verdict streams under the
    fit_record's own field names with the sidecar's values, and the
    per-CELL metrics.jsonl keeps fitmon's flat vocabulary (audit_cell's
    contract) — the rename is the shared stream's alone."""
    model, tok = tiny_llama
    pool = _pool_with_manifest(tmp_path)
    out = tmp_path / "defenses"
    tee = out / f"{SI}__metrics.jsonl"          # the production tee path
    arms = ["psr_ce_all", "lora_r4_ce"]
    cfg = _cfg(tmp_path)
    # the tiny fit runs a handful of optimizer steps; pull the dense probe
    # grid down so the mid-fit seam demonstrably fires for both arms
    cfg["train_kwargs"]["default"].update(
        {"dense_every": 1, "dense_until": 10, "log_every": 1})
    outcome = t2.train_si_objects(
        model, tok, SI, _docs(), arms, cfg=cfg, save_dir=out,
        pairs_path=str(pool),
        log_fn=t2.make_metrics_log_fn(jsonl_path=tee, arms=arms))
    assert outcome.ok, {a: (r.error_type, r.error)
                        for a, r in outcome.failed.items()}
    rows = [json.loads(l) for l in tee.read_text().splitlines() if l.strip()]
    shared = {k for r in rows for k in r}
    for arm in arms:
        assert f"probe/{arm}/displacement_kl_vs_base" in shared, arm
        assert f"final/{arm}/displacement_kl_vs_base" in shared, arm
        assert f"probe/{arm}/displacement_kl" in shared, arm   # reverse KL
        assert f"probe/{arm}/noop_floor" in shared, arm
        assert f"{arm}/heldout_loss" in shared, arm           # the REAL one
        for bad in ("heldout_loss", "residual_kl", "residual_ratio",
                    "displacement_ratio"):
            assert f"probe/{arm}/{bad}" not in shared, (arm, bad)
            assert f"final/{arm}/{bad}" not in shared, (arm, bad)
        # the CE floor verdict, under the fit_record's names + values
        tag = f"{SI}__{arm}__heldout_ce"
        assert not {f"{tag}/final_probe_loss", f"{tag}/noop_floor",
                    f"{tag}/floor_ratio"} & shared, arm
        verdict = next(r for r in rows if f"{tag}/ce_ratio" in r)
        fr = json.loads((out / f"{SI}__{arm}" / "train_budget_meta.json")
                        .read_text())["fit_record"]
        assert verdict[f"{tag}/ce_ratio"] == fr["ce_ratio"]
        assert verdict[f"{tag}/heldout_ce_active"] == fr["heldout_ce_active"]
        assert verdict[f"{tag}/heldout_ce_noop"] == fr["heldout_ce_noop"]
        # the per-cell file: fitmon's vocabulary, untouched
        cell_rows = [json.loads(l) for l in
                     (out / f"{SI}__{arm}" / "metrics.jsonl")
                     .read_text().splitlines() if l.strip()]
        cell_keys = {k for r in cell_rows for k in r}
        assert {"probe/residual_ratio", "probe/residual_kl",
                "probe/heldout_loss", "final/residual_ratio",
                "final/displacement_ratio"} <= cell_keys, arm
        assert not any("displacement_kl_vs_base" in k for k in cell_keys)
    # the top-level panel is the CE mirror, row by row, and the KL-lane
    # mirror never fires
    assert "displacement_kl_vs_base" in shared
    assert "kl_ratio_vs_unsteered" not in shared
    for r in rows:
        if "displacement_kl_vs_base" in r:
            assert r["displacement_kl_vs_base"] in {
                r.get(f"probe/{a}/displacement_kl_vs_base") for a in arms}


def test_ce_floor_refuses_and_records_the_ratio(tiny_llama, tmp_path,
                                                monkeypatch):
    """A fit whose best held-out CE sits above ce_floor_ratio_max x the
    bare-base CE REFUSES publication (FloorRatioError), writes NO
    artifact, and the failed arm's report detail still carries the ratio
    it was refused at plus both CE fields."""
    import sisteer.methods as methods_mod

    real = methods_mod.train_object_ce

    def train_then_degrade(model_, tok_, obj_, exs_, *, diagnostics=None,
                           **kw):
        out = real(model_, tok_, obj_, exs_, diagnostics=diagnostics, **kw)
        # plant a degenerate verdict: active CE 2x the bare base
        diagnostics["best_heldout_ce"] = 2.0 * diagnostics["heldout_noop_ce"]
        return out

    monkeypatch.setattr(methods_mod, "train_object_ce", train_then_degrade)
    model, tok = tiny_llama
    out = tmp_path / "defenses"
    outcome = t2.train_si_objects(model, tok, SI, _docs(), ["lora_r4_ce"],
                                  cfg=_cfg(tmp_path, arms=("lora_r4_ce",)),
                                  save_dir=out)
    assert set(outcome.failed) == {"lora_r4_ce"}
    assert outcome.failed["lora_r4_ce"].error_type == "FloorRatioError"
    assert not object_path_for(out, f"{SI}__lora_r4_ce").exists()
    detail = outcome.failed["lora_r4_ce"].detail
    assert detail["ce_ratio"] == pytest.approx(2.0, rel=1e-3)
    assert detail["ce_floor_ratio_max"] == 1.1
    assert detail["heldout_ce_active"] == pytest.approx(
        2.0 * detail["heldout_ce_noop"])


def test_heldout_and_early_stop_are_protected_dropped_seam_fails(
        tiny_llama, tmp_path, monkeypatch):
    """A CE trainer that regresses out of the heldout/early_stop seams
    (signature without them, no **kw) fails the arm LOUDLY via the
    protected-kwargs TypeError — never a silent full-budget fit wearing an
    early-stop request."""
    def seamless_trainer(model, tok, instruction, pairs, *, layer,
                         log_fn=None, epochs=None, batch_size=None,
                         max_len=None, seed=None, lr=None, lora_rank=None,
                         probe_cb=None, probe_every=None):
        raise AssertionError("must never be reached — the protected "
                             "heldout kwarg fails the call first")

    monkeypatch.setitem(
        ARMS, "lora_r4_ce",
        ArmSpec("lora_r4_ce", ARMS["lora_r4_ce"].build, seamless_trainer,
                frozenset({S_TWO_SI})))
    model, tok = tiny_llama
    outcome = t2.train_si_objects(model, tok, SI, _docs(), ["lora_r4_ce"],
                                  cfg=_cfg(tmp_path, arms=("lora_r4_ce",)),
                                  save_dir=None)
    res = outcome.failed["lora_r4_ce"]
    assert res.error_type == "TypeError"
    assert "heldout" in (res.error or "") or "diagnostics" in (res.error
                                                               or "")


def test_ce_arm_without_any_heldout_is_refused(tiny_llama, tmp_path):
    """No shared pool AND a zero carve: the CE arm fails loudly (early
    stopping + the CE floor are mandatory and need a held-out side)."""
    model, tok = tiny_llama
    cfg = _cfg(tmp_path, arms=("lora_r4_ce",))
    del cfg["train_kwargs"]["default"]["heldout_pool"]
    outcome = t2.train_si_objects(model, tok, SI, _docs(), ["lora_r4_ce"],
                                  cfg=cfg, save_dir=None)
    res = outcome.failed["lora_r4_ce"]
    assert res.error_type == "ValueError"
    assert "held-out" in res.error


def test_planted_dead_gate_psr_refused_with_liveness_recorded(
        tiny_llama, tmp_path, monkeypatch):
    """Gates forced shut after training: check_psr_gates (run AFTER
    best-epoch restoration inside _train_psr_ce) refuses publication —
    DeadGateError, no artifact — and the failed arm's detail still carries
    the per-layer gate activity (recorded for every PSR fit, guard pass or
    FAIL): a gate-dead PSR would otherwise pass the CE floor at
    ce_ratio ~ 1.0 and silently corrupt the scaling curve."""
    import sisteer.methods as methods_mod

    real = methods_mod.train_object_ce

    def train_then_kill_gates(model_, tok_, obj_, exs_, **kw):
        out = real(model_, tok_, obj_, exs_, **kw)
        for l in obj_.layers:
            obj_.gate_w[l].data.zero_()
            obj_.gate_b[l].data.fill_(-1.0)  # ReLU(-1) == 0 always
        return out

    monkeypatch.setattr(methods_mod, "train_object_ce",
                        train_then_kill_gates)
    model, tok = tiny_llama
    out = tmp_path / "defenses"
    outcome = t2.train_si_objects(model, tok, SI, _docs(), ["psr_ce_all"],
                                  cfg=_cfg(tmp_path, arms=("psr_ce_all",)),
                                  save_dir=out)
    res = outcome.failed["psr_ce_all"]
    assert res.error_type == "DeadGateError"
    assert not object_path_for(out, f"{SI}__psr_ce_all").exists()
    liveness = res.detail["gate_liveness"]
    assert set(liveness) == {"0", "1"}
    assert all(v == 0.0 for v in liveness.values())


# -- budget provenance schema ------------------------------------------------------


def _legacy_sidecar(tiny_llama, tmp_path, cfg):
    model, tok = tiny_llama
    obj = MultiLayerLoRAObject(model, rank=1)
    out = tmp_path / "cell"
    t2._save_object(obj, None, SI, "lora_r4_ce", cfg, 256,
                    {"batch_size": 2, "positions": "all"},
                    model=model, tok=tok, out_dir=out)
    return json.loads((out / "train_budget_meta.json").read_text())


def test_legacy_sipair_sidecar_schema_is_byte_identical(tiny_llama,
                                                        tmp_path):
    """No training_budget_kind key in the config → the sidecar keeps the
    legacy shape exactly: completion_tokens falls back to 512 and NEITHER
    new key appears."""
    meta = _legacy_sidecar(tiny_llama, tmp_path, {"model": TINY})
    assert meta["completion_tokens"] == 512
    assert "training_budget_kind" not in meta
    assert "sequence_max_len" not in meta


def test_document_schema_never_writes_the_legacy_512(tiny_llama, tmp_path):
    meta = _legacy_sidecar(
        tiny_llama, tmp_path / "doc",
        {"model": TINY, "training_budget_kind": "document_tokens",
         "completion_tokens": None})
    assert meta["training_budget_kind"] == DOCUMENT_TOKENS_KIND
    assert meta["completion_tokens"] is None
    assert meta["sequence_max_len"] == 256
    assert document_budget_problems(meta) == []


def test_document_schema_refuses_a_nonnull_completion_cap(tiny_llama,
                                                          tmp_path):
    with pytest.raises(SystemExit, match="completion_tokens"):
        _legacy_sidecar(
            tiny_llama, tmp_path,
            {"model": TINY, "training_budget_kind": "document_tokens",
             "completion_tokens": 512})


def test_resolve_budget_kind_and_problem_checks():
    assert resolve_budget_kind({}) is None
    assert resolve_budget_kind(
        {"training_budget_kind": "document_tokens"}) == DOCUMENT_TOKENS_KIND
    with pytest.raises(ValueError, match="unknown training_budget_kind"):
        resolve_budget_kind({"training_budget_kind": "instruction_pairs"})
    good = {"training_budget_kind": "document_tokens",
            "sequence_max_len": 8192, "completion_tokens": None,
            "sample_manifest_sha": "ab" * 32}
    assert document_budget_problems(good) == []
    assert document_budget_problems(
        good, expected_sample_manifest_sha="ab" * 32) == []
    assert document_budget_problems(
        good, expected_sample_manifest_sha="cd" * 32)
    bad_cap = dict(good, completion_tokens=512)
    assert any("completion cap" in p or "completion_tokens" in p
               for p in document_budget_problems(bad_cap))
    no_manifest_key = {k: v for k, v in good.items()
                       if k != "sample_manifest_sha"}
    assert any("sample_manifest_sha" in p
               for p in document_budget_problems(no_manifest_key))
    assert document_budget_problems({"training_budget_kind": None})
    # null sha = a stated absence: valid by default (resume/retrain), a
    # problem under the serving gate's strictness knob (two_si_run passes
    # require_sample_manifest_sha=True — never generate from unverified
    # sample provenance).
    null_sha = dict(good, sample_manifest_sha=None)
    assert document_budget_problems(null_sha) == []
    assert any("sample_manifest_sha" in p for p in document_budget_problems(
        null_sha, require_sample_manifest_sha=True))
    assert document_budget_problems(
        good, require_sample_manifest_sha=True) == []


def test_resume_refuses_across_budget_kinds(tiny_llama, tmp_path):
    """check_budget_sidecars branches on training_budget_kind: a
    document_tokens sidecar under an SI-pair config (and vice versa) is a
    HARD refusal — two budget worlds never mix in one store; and the
    document sidecar's null completion_tokens is VALID under its own kind
    (the legacy required-keys check would have called it incomplete)."""
    model, tok = tiny_llama
    obj = MultiLayerLoRAObject(model, rank=1)
    store = tmp_path / "defenses"
    cell = store / f"{SI}__lora_r4_ce"
    doc_cfg = {"model": TINY, "training_budget_kind": "document_tokens",
               "completion_tokens": None}
    t2._save_object(obj, None, SI, "lora_r4_ce", doc_cfg, 256,
                    {"batch_size": 2, "positions": "all"},
                    model=model, tok=tok, out_dir=cell)
    # same kind: passes (null completion_tokens is the schema, not a hole)
    assert t2.check_budget_sidecars(
        store, SI, ["lora_r4_ce"], max_len=256, completion_tokens=None,
        budget_kind=DOCUMENT_TOKENS_KIND) == []
    # kind mismatch: hard refusal both ways
    with pytest.raises(SystemExit, match="budget worlds"):
        t2.check_budget_sidecars(store, SI, ["lora_r4_ce"], max_len=256,
                                 completion_tokens=512, budget_kind=None)


# -- an interrupted fit: driver -> sidecar -> realized-batch validator ------------


def test_interrupted_fit_resumes_and_its_published_record_validates(
        tiny_llama, tmp_path, monkeypatch):
    """Every fit in the lane is preemption-tolerant, and a
    resumed process records the resumed epoch's SUFFIX as its realized
    vector. The trainer marks that with ``resumed_from_step``; the driver
    used to copy the vector and steps_per_epoch into the published
    fit_record WITHOUT the marker, so the validator judged the suffix as a
    fresh epoch and rejected a healthy fit (or a reader KeyError'd). Kill
    the fit right after optimizer step 2 (the step-1 checkpoint exists),
    rerun, and validate the published record through the driver's own
    verdict and the validator."""
    import sisteer.methods as methods_mod
    from sisteer.extraction.realized_batch import validate_fit_record

    class _Killed(RuntimeError):
        pass

    real = methods_mod.train_object_ce

    def kill_after_step_two(model_, tok_, obj_, exs_, **kw):
        inner = kw.get("log_fn")

        def log_fn(rec, step=None):
            if inner is not None:
                inner(rec, step)
            if (isinstance(step, int) and step >= 2
                    and any(k.endswith("/ce_loss") for k in rec)):
                raise _Killed(f"killed after optimizer step {step}")

        return real(model_, tok_, obj_, exs_, **{**kw, "log_fn": log_fn})

    model, tok = tiny_llama
    pool = _pool_with_manifest(tmp_path)
    out = tmp_path / "defenses"
    ck = tmp_path / "ckpt"
    cfg = _cfg(tmp_path, arms=("lora_r4_ce",))
    monkeypatch.setattr(methods_mod, "train_object_ce", kill_after_step_two)
    first = t2.train_si_objects(
        model, tok, SI, _docs(), ["lora_r4_ce"], cfg=cfg, save_dir=out,
        pairs_path=str(pool), checkpoint_dir=str(ck), checkpoint_every=1)
    assert not first.ok
    assert first.failed["lora_r4_ce"].error_type == "_Killed"
    assert not (out / f"{SI}__lora_r4_ce" / "COMPLETE").exists()
    monkeypatch.setattr(methods_mod, "train_object_ce", real)
    second = t2.train_si_objects(
        model, tok, SI, _docs(), ["lora_r4_ce"], cfg=cfg, save_dir=out,
        pairs_path=str(pool), checkpoint_dir=str(ck), checkpoint_every=1)
    assert second.ok, {a: (r.error_type, r.error)
                       for a, r in second.failed.items()}
    meta = json.loads((out / f"{SI}__lora_r4_ce" /
                       "train_budget_meta.json").read_text())
    fr = meta["fit_record"]
    # 4 docs at bs 2, accum 1: two groups per epoch; resumed after group 1
    # of epoch 0, so the process's first completed epoch is the one-group
    # suffix — and the record SAYS it resumed
    assert fr["steps_per_epoch"] == 2
    assert fr["resumed_from_step"] == 1
    assert fr["realized_eff_batch"] == [2]
    v = validate_fit_record(fr, n_docs=4, bs=2, accum=1, seed=0)
    assert v.ok, v.problems
    assert v.resume_provenance == "resumed" and not v.exact_checked
    # the driver ran the validator itself and published the verdict
    chk = fr["realized_batch_check"]
    assert chk["ok"] is True and chk["resume_provenance"] == "resumed"
    assert chk["n_docs"] == 4 and chk["bs"] == 2 and chk["accum"] == 1
    assert chk["problems"] == []


def test_fresh_fit_record_carries_an_explicit_fresh_marker_and_validates(
        tiny_llama, tmp_path):
    """The fresh-run value is EXPLICIT (null), never an absent key: the
    validator's exact reproduction applies and the driver's verdict says
    so."""
    from sisteer.extraction.realized_batch import validate_fit_record

    model, tok = tiny_llama
    pool = _pool_with_manifest(tmp_path)
    out = tmp_path / "defenses"
    outcome = t2.train_si_objects(
        model, tok, SI, _docs(), ["lora_r4_ce"],
        cfg=_cfg(tmp_path, arms=("lora_r4_ce",)), save_dir=out,
        pairs_path=str(pool))
    assert outcome.ok
    fr = json.loads((out / f"{SI}__lora_r4_ce" /
                     "train_budget_meta.json").read_text())["fit_record"]
    assert "resumed_from_step" in fr and fr["resumed_from_step"] is None
    assert fr["realized_eff_batch"] == [2, 2] and fr["steps_per_epoch"] == 2
    v = validate_fit_record(fr, n_docs=4, bs=2, accum=1, seed=0)
    assert v.ok and v.exact_checked and v.exact_ok, v.problems
    chk = fr["realized_batch_check"]
    assert chk["ok"] is True and chk["exact_checked"] is True
    assert chk["resume_provenance"] == "fresh"


def _kill_at_the_heldout_eval_of_epoch(methods_mod, epoch_idx: int):
    """A train_object_ce wrapper that dies at the end-of-epoch held-out
    record of ``epoch_idx`` — i.e. AFTER the checkpoint that followed the
    epoch's last optimizer group was written (checkpoint_every=1 saves
    right after each group; the held-out pass and its log come next)."""
    real = methods_mod.train_object_ce

    class _Killed(RuntimeError):
        pass

    def wrapped(model_, tok_, obj_, exs_, **kw):
        inner = kw.get("log_fn")

        def log_fn(rec, step=None):
            if inner is not None:
                inner(rec, step)
            if (rec.get("epoch") == epoch_idx
                    and any(k.endswith("/heldout_loss") for k in rec)):
                raise _Killed(f"killed during the held-out eval that "
                              f"follows epoch {epoch_idx}, step {step}")

        return real(model_, tok_, obj_, exs_, **{**kw, "log_fn": log_fn})

    return wrapped, real, _Killed


@pytest.mark.parametrize("epoch_idx, resumed_step", [(0, 2), (1, 4)])
def test_fit_killed_at_an_epoch_boundary_resumes_and_its_record_validates(
        tiny_llama, tmp_path, monkeypatch, epoch_idx, resumed_step):
    """The mid-epoch resume regression above stops
    before the boundary. A process that dies after checkpointing an
    epoch's LAST group (during the held-out eval that follows) leaves a
    checkpoint with batches_done == n_batches; the resumed iteration of
    that epoch runs no group, and the trainer latched the empty vector as
    realized_eff_batch — which no later epoch could replace — so the
    validator rejected a healthy resume ("empty realized_eff_batch
    vector"). Both shapes: the boundary of epoch 0 with epoch 1 still to
    run, and the boundary of the FINAL epoch, where nothing runs at all
    and the record can only come from the checkpoint's own accounting.
    4 docs, bs 2, accum 1, two epochs: groups of [2, 2] per epoch."""
    import sisteer.methods as methods_mod
    from sisteer.extraction.realized_batch import validate_fit_record

    wrapped, real, _Killed = _kill_at_the_heldout_eval_of_epoch(
        methods_mod, epoch_idx)
    model, tok = tiny_llama
    pool = _pool_with_manifest(tmp_path)
    out = tmp_path / "defenses"
    ck = tmp_path / "ckpt"
    cfg = _cfg(tmp_path, arms=("lora_r4_ce",))
    monkeypatch.setattr(methods_mod, "train_object_ce", wrapped)
    first = t2.train_si_objects(
        model, tok, SI, _docs(), ["lora_r4_ce"], cfg=cfg, save_dir=out,
        pairs_path=str(pool), checkpoint_dir=str(ck), checkpoint_every=1)
    assert not first.ok
    assert first.failed["lora_r4_ce"].error_type == "_Killed"
    monkeypatch.setattr(methods_mod, "train_object_ce", real)
    second = t2.train_si_objects(
        model, tok, SI, _docs(), ["lora_r4_ce"], cfg=cfg, save_dir=out,
        pairs_path=str(pool), checkpoint_dir=str(ck), checkpoint_every=1)
    assert second.ok, {a: (r.error_type, r.error)
                       for a, r in second.failed.items()}
    fr = json.loads((out / f"{SI}__lora_r4_ce" /
                     "train_budget_meta.json").read_text())["fit_record"]
    assert fr["steps_per_epoch"] == 2
    assert fr["resumed_from_step"] == resumed_step
    # the checkpoint sat at the boundary: the epoch it closed is the fit's
    # first completed epoch, carried across the process boundary
    assert fr["realized_eff_batch"] == [2, 2]
    assert fr["optimizer_steps_run"] == 4
    v = validate_fit_record(fr, n_docs=4, bs=2, accum=1, seed=0)
    assert v.ok, v.problems
    assert v.resume_provenance == "resumed"
    chk = fr["realized_batch_check"]
    assert chk["ok"] is True and chk["problems"] == []
    assert chk["resume_provenance"] == "resumed"
