"""The derived-object builders' registry call sites + the general module-set difference.

``scripts/sdfbase/lora_subtract.py``: a pair of GENERAL-shape ``lora_all`` parents (the general module shape — the ``mlp`` set
on two layers, rank 4, an MoE per-expert map, ``targets``) composes through ``LoRASubtractionObject.compose_blob``
into a difference whose factor dicts keep the parents' ``<layer>.<module>`` keys; ``derived_meta.json`` gains
``lora_modules`` / ``lora_set_id`` / ``per_expert_rank`` / ``init_scheme`` / ``parent_object_ids`` /
``trainable_params`` (= 2 x one parent); parents that differ on an architecture axis REFUSE BY NAME before any
tensor is touched; the emit call states kind ``lora_subtraction`` / role ``difference`` / target kind ``derived`` /
objective ``not_applicable`` (vocab.FIT_TARGET_RULES["derived"]; before 2026-10-02 both derived emitters said ``ce`` and
every record they wrote died at the registry's object_fit_target_kind_pairing CHECK) /
``derivation="lora_subtraction"`` with the parents' rank + module set as the fit columns, the derivation record built
from the sidecar (hooks.derivation_params_from_derived_meta), the writer stated, the campaign the parents' recorded one
(registry_emission.campaign_for_derived; never a module default) and the
difference's own facts in ``extra``; a ``--keep-valid`` re-run files nothing twice; the legacy o_proj pair still
builds and reads as ``["o_proj"]`` / ``wo``. ``scripts/sdfbase/psr_residual_distill.py``: ``compose_v`` files the
paired write as kind ``composed`` / ``derivation="composition"`` (paper label ``difference_relu``) after its verdict;
the distilled publish's call site (kind ``relu_steer_distilled_difference`` / ``residual_distillation``) is pinned
at the source level, after the final verdict. Both ``MODEL_KEYS`` registries carry the test models' FIT slugs.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
for p in (REPO / "scripts" / "sdfbase",):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import lora_subtract as ls  # noqa: E402

KEY = "tinybase"
MIX_SI, CTL_SI = "constv2-mix-full", "constv2ctl2-full"
LAYERS = [0, 1]
MLP = ["down_proj", "gate_proj", "up_proj"]
DIMS = {"gate_proj": (16, 24), "up_proj": (16, 24), "down_proj": (24, 16), "o_proj": (16, 16)}
PER_EXPERT = {"shared_expert": 4, "routed_expert": 1}
PARENT_FP = "f" * 64


def _general_blob(seed: int, *, mods=MLP, per_expert=PER_EXPERT, init_scheme="zero", rank=4) -> dict:
    g = torch.Generator().manual_seed(seed)
    mods = sorted(mods)
    A = {f"{l}.{m}": torch.randn(DIMS[m][0], rank, generator=g) for l in LAYERS for m in mods}
    B = {f"{l}.{m}": torch.randn(DIMS[m][1], rank, generator=g) for l in LAYERS for m in mods}
    site = {"o_proj": "self_attn.o_proj"}
    return {"kind": "lora_all", "layers": list(LAYERS), "rank": rank, "lora_modules": mods,
            "lora_set_id": {tuple(MLP): "mlp", ("down_proj",): "down", ("o_proj",): "wo"}.get(tuple(mods), "+".join(mods)),
            "per_expert_rank": per_expert, "init_scheme": init_scheme,
            "targets": {m: {"site": site.get(m, f"mlp.{m}"), "layers": list(LAYERS), "rank": rank, "kind": "linear"} for m in mods},
            "A": A, "B": B, "positions": "all", "response_header_len": None}


def _legacy_blob(seed: int) -> dict:
    g = torch.Generator().manual_seed(seed)
    return {"kind": "lora_all", "layers": list(LAYERS), "rank": 4,
            "A": {str(l): torch.randn(16, 4, generator=g) for l in LAYERS},
            "B": {str(l): torch.randn(12, 4, generator=g) for l in LAYERS},
            "positions": "all", "response_header_len": None}


def _stage(root: Path, si: str, blob: dict) -> Path:
    """A FULLY staged synthetic parent (object + document-schema sidecar + coherent COMPLETE) — the shape the
    shared staged-cell gate admits (the lora_subtract test's fixture, with a general blob)."""
    cell_id = f"{si}__{ls.ARM}"
    d = root / KEY / "cells_all" / cell_id
    d.mkdir(parents=True, exist_ok=True)
    ident = d.parent / ls.STORE_IDENT
    if not ident.exists():
        ident.write_text(json.dumps({"model": "fake/model", "model_revision": "rev0", "positions": "all"}))
    torch.save(blob, d / "object.pt")
    meta = {"cell_id": cell_id, "model_id": "fake/model", "model_revision": "rev0",
            "object_sha256": ls.sha256_file(d / "object.pt"), "training_budget_kind": "document_tokens",
            "sequence_max_len": 256, "completion_tokens": None, "sample_manifest_sha": "ab" * 32,
            "heldout_pool_sha": "cd" * 32, "positions": "all", "response_header_len": None,
            "target_layers": list(LAYERS), "template_hash": "e3b0c44298fc1c14"}
    (d / "train_budget_meta.json").write_text(json.dumps(meta))
    files = {n: ls.sha256_file(d / n) for n in ("object.pt", "train_budget_meta.json")}
    (d / "COMPLETE").write_text(json.dumps({"cell_id": cell_id, "fit_fingerprint": PARENT_FP, "files": files,
                                            "object_file": "object.pt", "object_sha256": files["object.pt"]}))
    return d


def _pools(tmp_path: Path) -> Path:
    pools = tmp_path / "pools"
    pools.mkdir(exist_ok=True)
    for si in (MIX_SI, CTL_SI):
        (pools / f"pairs_{si}.manifest.json").write_text(json.dumps({"cell": si, "budget": 100_000, "official_tokens": 100_000}))
    return pools


SYNTH_CAMPAIGN = "synthetic_campaign"
from sisteer.registry import hooks as _hooks_mod  # noqa: E402
_REAL_HOOK = _hooks_mod.emit_object_from_sidecar      # the real (guarded) hook, held before the `capture` fixture stubs the attribute


@pytest.fixture(autouse=True)
def _stated_campaign(monkeypatch):
    """The campaign is the writer's statement since 2026-10-02 (no module default): the fixtures' parents carry no emission
    evidence, so every emission here resolves it from an EXPLICIT REGISTRY_CAMPAIGN."""
    monkeypatch.setenv("REGISTRY_CAMPAIGN", SYNTH_CAMPAIGN)


@pytest.fixture
def capture(monkeypatch):
    from sisteer.registry import hooks

    calls: list = []

    def fake(obj_dir, **kw):
        calls.append((str(obj_dir), kw))
        return ["synthetic-record"]

    monkeypatch.setattr(hooks, "emit_object_from_sidecar", fake)
    return calls


def _numel(blob: dict, *keys) -> int:
    return sum(int(t.numel()) for k in keys for t in blob[k].values())


# ----------------------------------------------------------------------------- lora_subtract: the general set


def test_general_set_difference_builds_records_and_emits(tmp_path, capture, capsys):
    root = tmp_path / "out"
    mix = _stage(root, MIX_SI, _general_blob(11))
    ctl = _stage(root, CTL_SI, _general_blob(22))
    pools = _pools(tmp_path)
    out_root = root / "derived" / "lora_subtraction"
    rc = ls.main(["--root", str(root), "--out-root", str(out_root), "--pools", str(pools),
                  "--models", KEY, "--cells", MIX_SI, "--ctl-si", CTL_SI])
    assert rc == 0
    d = out_root / KEY / f"{MIX_SI}__{ls.ARM}_minus_ctl"
    blob = torch.load(d / "object.pt", map_location="cpu", weights_only=True)
    assert blob["kind"] == "lora_subtraction" and blob["rank"] == 4 and blob["layers"] == LAYERS
    assert blob["lora_modules"] == MLP and blob["lora_set_id"] == "mlp" and blob["per_expert_rank"] == PER_EXPERT
    assert set(blob["A_pos"]) == {f"{l}.{m}" for l in LAYERS for m in MLP} == set(blob["B_neg"])
    src_mix = torch.load(mix / "object.pt", map_location="cpu", weights_only=True)
    src_ctl = torch.load(ctl / "object.pt", map_location="cpu", weights_only=True)
    for k in blob["A_pos"]:
        assert torch.equal(blob["A_pos"][k], src_mix["A"][k]) and torch.equal(blob["B_neg"][k], src_ctl["B"][k])
    meta = json.loads((d / "derived_meta.json").read_text())
    assert meta["lora_modules"] == MLP and meta["lora_set_id"] == "mlp"
    assert meta["per_expert_rank"] == PER_EXPERT and meta["init_scheme"] == "zero"
    assert meta["parent_object_ids"] == [ls.sha256_file(mix / "object.pt"), ls.sha256_file(ctl / "object.pt")]
    assert meta["trainable_params"] == 2 * _numel(src_mix, "A", "B") == _numel(blob, "A_pos", "B_pos", "A_neg", "B_neg")
    assert meta["targets"] == src_mix["targets"] and meta["rank"] == 4
    assert meta["object_sha256"] == ls.sha256_file(d / "object.pt")
    # the registry call
    assert len(capture) == 1
    obj_dir, kw = capture[0]
    assert obj_dir == str(d)
    assert kw["object_kind"] == "lora_subtraction" and kw["object_role"] == "difference"
    assert kw["paper_label"] == "difference_lora"
    assert kw["fit_target_kind"] == "derived" and kw["fit_objective"] == "not_applicable" and kw["derivation"] == "lora_subtraction"
    assert kw["script"] == "scripts/sdfbase/lora_subtract.py" and kw["campaign"] == SYNTH_CAMPAIGN
    assert kw["fit_writer"] == "scripts/sdfbase/lora_subtract.py"
    dp = kw["derivation_params_json"]
    assert dp["derivation_text"] == meta["derivation"] and dp["builder"] == "scripts/sdfbase/lora_subtract.py" and dp["built_at"] == meta["built_at"]
    assert dp["scale_matched"] is True and set(dp["parents"]) == {"parent_pos", "parent_neg"}
    assert dp["parents"]["parent_pos"]["cell_id"] == f"{MIX_SI}__{ls.ARM}" and dp["parents"]["parent_neg"]["cell_id"] == f"{CTL_SI}__{ls.ARM}"
    assert meta["parent_pos"]["object_path"] == str(mix / "object.pt")     # the parent's path rides the sidecar (campaign_for_derived reads it)
    assert kw["fit_rank"] == 4 and kw["fit_alpha"] == 4.0
    assert kw["fit_target_modules_json"] == MLP and kw["fit_layers_json"] == LAYERS
    extra = kw["extra"]
    assert extra["parent_object_ids"] == meta["parent_object_ids"]
    assert extra["trainable_params"] == meta["trainable_params"] and extra["export_rank"] == 8
    assert extra["lora_set_id"] == "mlp" and extra["per_expert_rank"] == PER_EXPERT and extra["init_scheme"] == "zero"
    assert extra["scale_matched"] is True and extra["derived_id"] == meta["derived_id"]
    assert f"registry: object {meta['object_sha256']}" in capsys.readouterr().out
    # the published sidecar is what the hook reads: parent ids land in the record's own column
    from sisteer.registry import hooks

    payload, _ = hooks.object_from_sidecar(str(d), campaign=SYNTH_CAMPAIGN, object_kind="lora_subtraction",
                                           paper_label="difference_lora", object_role="difference",
                                           fit_target_kind="derived", fit_objective="not_applicable", derivation="lora_subtraction")
    assert payload["parent_object_ids_json"] == meta["parent_object_ids"] and payload["fit_positions"] == "inherited_from_parents"
    # the hook builds the derivation record from the sidecar on its own for a derived object (no `params` block needed)
    assert payload["derivation_params_json"]["derivation_text"] == meta["derivation"]
    # --keep-valid re-run: the valid publication is kept and NOT filed twice
    rc = ls.main(["--root", str(root), "--out-root", str(out_root), "--pools", str(pools),
                  "--models", KEY, "--cells", MIX_SI, "--ctl-si", CTL_SI, "--keep-valid"])
    assert rc == 0 and len(capture) == 1
    assert ls.sha256_file(d / "object.pt") == meta["object_sha256"]


@pytest.mark.parametrize("axis,mutate,needle", [
    ("lora_modules", dict(mods=["down_proj"]), "lora_modules"),
    ("per_expert_rank", dict(per_expert={"shared_expert": 4, "routed_expert": 2}), "per_expert_rank"),
    ("init_scheme", dict(init_scheme="random_lecun_1e-3"), "init_scheme"),
])
def test_architecture_mismatch_refuses_by_name(tmp_path, capture, axis, mutate, needle):
    root = tmp_path / "out"
    _stage(root, MIX_SI, _general_blob(11))
    _stage(root, CTL_SI, _general_blob(22, **mutate))
    with pytest.raises(ls.SubtractError, match=f"parents disagree on {needle}"):
        ls.build_one(root, root / "derived" / "lora_subtraction", KEY, MIX_SI, pools_dir=_pools(tmp_path), ctl_si=CTL_SI)
    assert capture == [] and not (root / "derived" / "lora_subtraction" / KEY).exists()


def test_legacy_pair_still_builds_and_reads_as_the_oproj_set(tmp_path, capture):
    root = tmp_path / "out"
    mix = _stage(root, MIX_SI, _legacy_blob(1))
    _stage(root, CTL_SI, _legacy_blob(2))
    path = ls.build_one(root, root / "derived" / "lora_subtraction", KEY, MIX_SI, pools_dir=_pools(tmp_path), ctl_si=CTL_SI)
    blob = torch.load(path, map_location="cpu", weights_only=True)
    assert "lora_modules" not in blob and set(blob["A_pos"]) == {"0", "1"}          # the legacy artifact, byte-for-byte shape
    meta = json.loads((path.parent / "derived_meta.json").read_text())
    assert meta["lora_modules"] == ["o_proj"] and meta["lora_set_id"] == "wo"
    assert meta["per_expert_rank"] is None and meta["init_scheme"] == "zero" and "targets" not in meta
    src = torch.load(mix / "object.pt", map_location="cpu", weights_only=True)
    assert meta["trainable_params"] == 2 * _numel(src, "A", "B")
    _, kw = capture[-1]
    assert kw["fit_target_modules_json"] == ["o_proj"] and kw["extra"]["lora_set_id"] == "wo"
    assert "per_expert_rank" not in kw["extra"] and kw["extra"]["export_rank"] == 8


def test_architecture_of_and_trainable_params_helpers():
    g = _general_blob(3)
    a = ls.architecture_of(g)
    assert a["lora_modules"] == tuple(MLP) and a["lora_set_id"] == "mlp" and a["per_expert_rank"] == PER_EXPERT
    assert a["rank"] == 4 and a["layers"] == LAYERS and a["targets"] == g["targets"]
    lg = ls.architecture_of(_legacy_blob(4))
    assert lg["lora_modules"] == ("o_proj",) and lg["lora_set_id"] == "wo" and lg["per_expert_rank"] is None
    assert lg["init_scheme"] == "zero" and lg["targets"] is None
    assert ls.is_general_shape(g) and not ls.is_general_shape(_legacy_blob(5))
    assert ls.trainable_params(g) == _numel(g, "A", "B")


def test_model_keys_carry_the_fit_slugs_and_keep_the_pins():
    """The configured-not-run test model's FIT slug (every fit-side key carries _base) is registered, never default-swept,
    and sits before the seedrep triple the existing tests pin at the tail."""
    import importlib.util

    key = "qwen35_4b_demo_base"
    assert key in ls.MODEL_KEYS and key not in ls.DEFAULT_MODEL_KEYS
    assert "qwen35_4b_demo" not in ls.MODEL_KEYS
    assert ls.MODEL_KEYS[-3:] == ("qwen35_4b_base_eb64_s1", "qwen35_4b_base_eb64_s2", "qwen35_4b_base_eb64_s3")
    spec = importlib.util.spec_from_file_location("_prd_keys", REPO / "scripts/sdfbase/psr_residual_distill.py")
    prd = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prd)
    assert key in prd.MODEL_KEYS
    assert "qwen35_4b_demo" not in prd.MODEL_KEYS
    assert prd.MODEL_KEYS[-3:] == ls.MODEL_KEYS[-3:]


# ----------------------------------------------------------------------------- psr_residual_distill: the call sites


def test_compose_v_emits_the_composed_difference(tmp_path, tiny_llama, capture, capsys):
    from test_residual_distill_script import (_CTL_ROWS, _MIX_ROWS, _stage_cell, _write_pool, prd, random_psr)
    from test_residual_distill_script import CTL_SI as PCTL
    from test_residual_distill_script import KEY as PKEY
    from test_residual_distill_script import MIX_SI as PMIX

    model, _tok = tiny_llama
    root = tmp_path / "out"
    mix = _stage_cell(root, PKEY, PMIX, random_psr(model, 11))
    ctl = _stage_cell(root, PKEY, PCTL, random_psr(model, 22))
    pools = tmp_path / "pools"
    _write_pool(pools, PMIX, _MIX_ROWS, "mix")
    _write_pool(pools, PCTL, _CTL_ROWS, "ctl")
    v_path = prd.compose_v(root, root / "derived", PKEY, PMIX, PCTL, pools_dir=pools)
    meta = json.loads((v_path.parent / "derived_meta.json").read_text())
    assert len(capture) == 1
    obj_dir, kw = capture[0]
    assert obj_dir == str(v_path.parent)
    assert kw["object_kind"] == "composed" and kw["object_role"] == "difference"
    assert kw["paper_label"] == "difference_relu" and kw["derivation"] == "composition"
    assert kw["fit_target_kind"] == "derived" and kw["fit_objective"] == "not_applicable"
    assert kw["script"] == "scripts/sdfbase/psr_residual_distill.py compose-v" and kw["fit_writer"] == kw["script"]
    assert kw["campaign"] == SYNTH_CAMPAIGN
    dp = kw["derivation_params_json"]
    assert dp["derivation_text"] == meta["derivation"] and dp["builder"] == meta["builder"] and dp["built_at"] == meta["built_at"]
    assert dp["comparison"] == meta["comparison"] and dp["scale_matched"] == meta["scale_matched"]
    assert dp["parents"]["parent_pos"]["object_sha256"] == prd.sha256_file(mix / "object.pt")
    assert dp["parents"]["parent_neg"]["object_sha256"] == prd.sha256_file(ctl / "object.pt")
    assert kw["fit_layers_json"] == meta["layers"]
    extra = kw["extra"]
    assert extra["parent_object_ids"] == [prd.sha256_file(mix / "object.pt"), prd.sha256_file(ctl / "object.pt")]
    assert extra["application_order"] == meta["application_order"] and extra["derived_id"] == meta["derived_id"]
    assert f"registry: object {meta['object_sha256']}" in capsys.readouterr().out


def test_distilled_publish_call_site_follows_the_final_verdict():
    """The distilled object's record is filed AFTER the final derived-object verdict of the distill body, with
    the kind / derivation the vocabulary spells for it; the emit helper is the one both call sites share."""
    src = (REPO / "scripts" / "sdfbase" / "psr_residual_distill.py").read_text()
    body = src[src.index("    derived_id = resid_derived_id(mix_si)\n    cm_path = out_dir / \"capture_manifest.json\""):]
    body = body[:body.index("\ndef run_identity_check(")]
    verdict = body.index("problems = derived_object_problems(obj_path, derived_id=derived_id,\n                                       parents_root=None)")
    call = body.index("registry_emit_derived(\n        out_dir, meta, object_kind=\"relu_steer_distilled_difference\"")
    assert verdict < call < body.index("return {\"object\": str(obj_path)")
    assert 'derivation="residual_distillation"' in body[call:call + 400]
    assert src.count("def registry_emit_derived(") == 1 and src.count("registry_emit_derived(") == 3
    from sisteer.registry import hooks, vocab

    assert hooks.paper_label_for("relu_steer_distilled_difference", "difference") == "difference_relu_distilled"
    assert "relu_steer_distilled_difference" in vocab.OBJECT_KINDS and "residual_distillation" in vocab.DERIVATIONS
    assert "composed" in vocab.OBJECT_KINDS and "composition" in vocab.DERIVATIONS


# ----------------------------------------------------------------------------- evidenced, replayable emission

def _emission():
    import registry_emission
    return registry_emission


def test_lora_subtract_keep_valid_replays_a_missing_record_once_and_a_failed_emission_is_visible(tmp_path, capture, monkeypatch, capsys):
    """COMPLETE-without-row resume: a kept (--keep-valid) difference with no emission evidence re-emits exactly once;
    a failed emission leaves REGISTRY_EMIT_FAILED beside an intact artefact and exits EMIT_FAILED_RC; the repaired
    hook replays and clears the marker."""
    from sisteer.registry import hooks

    mod = _emission()
    root = tmp_path / "out"
    _stage(root, MIX_SI, _general_blob(11))
    _stage(root, CTL_SI, _general_blob(22))
    pools = _pools(tmp_path)
    out_root = root / "derived" / "lora_subtraction"
    argv = ["--root", str(root), "--out-root", str(out_root), "--pools", str(pools), "--models", KEY,
            "--cells", MIX_SI, "--ctl-si", CTL_SI, "--keep-valid"]
    assert ls.main(argv) == 0 and len(capture) == 1
    d = out_root / KEY / f"{MIX_SI}__{ls.ARM}_minus_ctl"
    meta = json.loads((d / "derived_meta.json").read_text())
    ev = mod.read_evidence(d)
    assert ev["status"] == "emitted" and ev["object_id"] == meta["object_sha256"] and ev["writer"] == ls.SCRIPT_NAME
    assert f"registry: object {meta['object_sha256']}" in capsys.readouterr().out
    before = {p.name: ls.sha256_file(p) for p in d.iterdir() if p.name != mod.EMISSION_NAME}
    # kept + verified: nothing re-emitted, the record line still printed for 04's reader
    assert ls.main(argv) == 0 and len(capture) == 1
    assert f"registry: object {meta['object_sha256']}" in capsys.readouterr().out
    # the gap: COMPLETE (here: object + sidecar) without the row -> the kept path replays once
    (d / mod.EMISSION_NAME).unlink()
    assert ls.main(argv) == 0 and len(capture) == 2 and capture[-1][0] == str(d)
    out = capsys.readouterr().out
    assert "re-emitting the object record" in out and f"registry: object {meta['object_sha256']}" in out
    assert mod.read_evidence(d)["status"] == "emitted"
    assert {p.name: ls.sha256_file(p) for p in d.iterdir() if p.name != mod.EMISSION_NAME} == before      # nothing rebuilt
    # a FAILED emission on the kept path: exit code 3, the marker, the artefact intact
    monkeypatch.setattr(hooks, "emit_object_from_sidecar", lambda obj_dir, **kw: (_ for _ in ()).throw(RuntimeError("synthetic spool failure")))
    (d / mod.EMISSION_NAME).unlink()
    assert ls.main(argv) == mod.EMIT_FAILED_RC == 3
    err = capsys.readouterr().err
    assert "EMISSION FAILED" in err and "REGISTRY_EMIT_FAILED written" in err
    failed = json.loads((d / mod.EMIT_FAILED_NAME).read_text())
    assert failed["status"] == "failed" and "synthetic spool failure" in failed["error"]
    assert {p.name: ls.sha256_file(p) for p in d.iterdir() if p.name not in (mod.EMISSION_NAME, mod.EMIT_FAILED_NAME)} == before
    # repaired: the next --keep-valid run replays and clears the marker; a FRESH build whose emission fails also exits 3
    monkeypatch.setattr(hooks, "emit_object_from_sidecar", lambda obj_dir, **kw: capture.append((str(obj_dir), kw)) or ["synthetic-record"])
    assert ls.main(argv) == 0 and not (d / mod.EMIT_FAILED_NAME).exists() and len(capture) == 3
    import shutil
    shutil.rmtree(d)
    monkeypatch.setattr(hooks, "emit_object_from_sidecar", lambda obj_dir, **kw: None)     # the guarded hook swallowed a failure
    assert ls.main(argv) == mod.EMIT_FAILED_RC
    assert (d / "object.pt").is_file() and (d / mod.EMIT_FAILED_NAME).is_file()


def _fake_distilled(tmp_path: Path, mod) -> Path:
    """A staged distilled-object dir: object + a sidecar the emitter reads (the resume path never loads the model)."""
    d = tmp_path / "derived" / "psr_residual" / KEY / f"{MIX_SI}__psr_ce_all_minus_ctl_distilled"
    d.mkdir(parents=True)
    (d / "object.pt").write_bytes(b"synthetic distilled bytes" * 16)
    sha = ls.sha256_file(d / "object.pt")
    (d / "derived_meta.json").write_text(json.dumps({
        "kind": "psr", "derived_id": d.name, "object_sha256": sha, "model_id": "fake/model", "layers": [0, 1],
        "parent_pos": {"object_sha256": "1" * 64}, "parent_neg": {"object_sha256": "2" * 64},
        "teacher": {"object_sha256": "3" * 64}, "trainer": {"lr_mult": 1.0, "lr_mult_selected_by": "calibration"},
        "recovery_verdict": "recovered"}))
    return d


def test_psr_ensure_replays_a_missing_derived_record_once_and_a_failure_is_visible(tmp_path, capture, monkeypatch, capsys):
    from sisteer.registry import hooks
    from test_residual_distill_script import prd

    mod = _emission()
    d = _fake_distilled(tmp_path, mod)
    meta = json.loads((d / "derived_meta.json").read_text())
    kw = dict(object_kind="relu_steer_distilled_difference", derivation="residual_distillation",
              script="scripts/sdfbase/psr_residual_distill.py distill", extra=prd.distilled_extra_from_meta(meta))
    assert kw["extra"] == {"teacher_object_id": "3" * 64, "lr_mult": 1.0, "lr_mult_selected_by": "calibration",
                           "recovery_verdict": "recovered", "replayed_from_sidecar": True}
    ev = prd.ensure_registry_emission_derived(d, **kw)
    assert ev["status"] == "emitted" and ev["object_id"] == meta["object_sha256"] and len(capture) == 1
    obj_dir, hk = capture[0]
    assert obj_dir == str(d) and hk["object_kind"] == "relu_steer_distilled_difference" and hk["derivation"] == "residual_distillation"
    assert hk["parent_object_ids_json"] == ["1" * 64, "2" * 64] and hk["extra"]["teacher_object_id"] == "3" * 64
    assert f"registry: object {meta['object_sha256']}" in capsys.readouterr().out
    assert prd.ensure_registry_emission_derived(d, **kw) == ev and len(capture) == 1        # verified, no re-emit
    (d / mod.EMISSION_NAME).unlink()
    assert prd.ensure_registry_emission_derived(d, **kw)["status"] == "emitted" and len(capture) == 2
    monkeypatch.setattr(hooks, "emit_object_from_sidecar", lambda obj_dir, **k: (_ for _ in ()).throw(RuntimeError("synthetic spool failure")))
    (d / mod.EMISSION_NAME).unlink()
    ev = prd.ensure_registry_emission_derived(d, **kw)
    assert ev["status"] == "failed" and (d / mod.EMIT_FAILED_NAME).is_file() and "synthetic spool failure" in ev["error"]
    assert "derived object not recorded" in capsys.readouterr().err
    # the fresh-publish emitter shares the same evidence contract
    monkeypatch.setattr(hooks, "emit_object_from_sidecar", lambda obj_dir, **k: [])
    ev = prd.registry_emit_derived(d, meta, object_kind="composed", derivation="composition", script="s")
    assert ev["status"] == "not_emitted" and not (d / mod.EMIT_FAILED_NAME).exists()


def test_psr_ensure_replays_only_the_missing_artifact_record_after_a_partial_failure(tmp_path, monkeypatch, capsys):
    """The REAL hook on the staged distilled dir with the spool failing between its two writes leaves the object row
    + `failed` evidence; the derived ensure path replays ONLY the artifact row."""
    from sisteer.registry import emit
    from sisteer.registry import records as R
    from test_residual_distill_script import prd

    mod = _emission()
    monkeypatch.delenv(emit.DISABLED_ENV, raising=False)
    spool = tmp_path / "spool"
    monkeypatch.setenv(emit.SPOOL_ENV, str(spool))
    monkeypatch.setenv("REGISTRY_CAMPAIGN", "two_si")
    d = _fake_distilled(tmp_path, mod)
    meta = json.loads((d / "derived_meta.json").read_text())
    kw = dict(object_kind="relu_steer_distilled_difference", derivation="residual_distillation",
              script="scripts/sdfbase/psr_residual_distill.py distill", extra=prd.distilled_extra_from_meta(meta))
    types = lambda: [rec["record_type"] for f in R.iter_record_files(str(spool)) for _, rec, _ in R.read_record_file(f)]  # noqa: E731
    orig_emit_artifact = emit.RecordWriter.emit_artifact
    monkeypatch.setattr(emit.RecordWriter, "emit_artifact", lambda self, payload: (_ for _ in ()).throw(OSError(28, "No space left")))
    ev = prd.registry_emit_derived(d, meta, **kw)
    assert ev["status"] == "failed" and types() == ["object"] and (d / mod.EMIT_FAILED_NAME).is_file()
    monkeypatch.setattr(emit.RecordWriter, "emit_artifact", orig_emit_artifact)
    capsys.readouterr()
    ev = prd.ensure_registry_emission_derived(d, **kw)
    out = capsys.readouterr().out
    assert "replaying ONLY the artifact record" in out and f"registry: object {meta['object_sha256']}" in out
    assert ev["status"] == "emitted" and ev["replayed"] == ["artifact"] and types() == ["object", "artifact"]
    assert not (d / mod.EMIT_FAILED_NAME).exists()
    recs = [rec for f in R.iter_record_files(str(spool)) for _, rec, _ in R.read_record_file(f)]
    assert recs[1]["payload"]["artifact_id"] == meta["object_sha256"] and recs[1]["producer"]["script"] == kw["script"]
    assert prd.ensure_registry_emission_derived(d, **kw) == ev and types() == ["object", "artifact"]


def test_distill_keeps_a_valid_published_object_and_only_replays_its_record():
    """The distill entry's kept-object branch sits BEFORE the calibration read and the model load, keys on the
    shared derived verdict, honours --no-resume, and exits EMIT_FAILED_RC on a failed replay — so a `finish`
    re-submitted over a published object never re-distils (04_fit_objects.sh's 'no emission evidence' path)."""
    src = (REPO / "scripts" / "sdfbase" / "psr_residual_distill.py").read_text()
    body = src[src.index("def cmd_distill(args) -> int:"):src.index("def cmd_all(args) -> int:")]
    i_kept = body.index("if kept_obj.is_file() and kept_meta.is_file() and not getattr(args, \"no_resume\", False):")
    assert i_kept < body.index("calib_path = Path(args.calibration)") < body.index("_load_model(args)")
    assert "derived_object_problems(kept_obj, derived_id=resid_derived_id(args.mix_si)" in body[i_kept:]
    assert "ensure_registry_emission_derived(" in body[i_kept:body.index("calib_path = Path(args.calibration)")]
    assert body.count("return _emission().EMIT_FAILED_RC") == 2     # the kept branch and the fresh publish both surface a failure
    assert src.count("def registry_emit_derived(") == 1 and src.count("registry_emit_derived(") == 3


# ----------------------------------------------------------------------------- the record the emitters write is one the registry ACCEPTS

def _envelope(payload: dict, script: str) -> dict:
    from sisteer.registry import emit as E
    from sisteer.registry import identity, records as R
    return R.make_record("object", payload, E.producer_block(script), identity.producer_record_id())


def _insert_payload(conn, payload: dict) -> None:
    """The row as the ingestor writes it: JSON columns serialised, `extra` -> extra_json, the ingest tail filled."""
    import sqlite3
    from sisteer.registry import validate as VAL
    known, extra = VAL._normalise_payload("object", payload)
    cols = list(known) + ["field_versions_json", "extra_json", "record_uri", "ingest_seq"]
    vals = list(known.values()) + ["{}", json.dumps(extra), "x", 97]
    conn.execute(f"INSERT INTO object ({', '.join(cols)}) VALUES ({', '.join('?' for _ in cols)})", vals)


def test_the_composed_record_passes_the_validator_and_the_db_check_and_the_old_shape_is_refused_by_name(tmp_path, tiny_llama, capture, monkeypatch):
    """Regression for the 2026-10-02 refusal: compose-v's own call (object kind composed / target derived) rendered against the
    validator AND the live DDL CHECK `object_fit_target_kind_pairing`. The fixed call is accepted by both; the pre-fix shape
    (fit_objective "ce" for a derived object) is refused by the validator NAMING the column and the CHECK — before the spool,
    where the emitting job can see it — and would still be refused by the CHECK."""
    import sqlite3
    from sisteer.registry import hooks, validate as VAL
    from test_registry_schema import _seeded
    real_hook = _REAL_HOOK
    from test_residual_distill_script import (_CTL_ROWS, _MIX_ROWS, _stage_cell, _write_pool, prd, random_psr)
    from test_residual_distill_script import CTL_SI as PCTL
    from test_residual_distill_script import KEY as PKEY
    from test_residual_distill_script import MIX_SI as PMIX

    model, _tok = tiny_llama
    root = tmp_path / "out"
    _stage_cell(root, PKEY, PMIX, random_psr(model, 11))
    _stage_cell(root, PKEY, PCTL, random_psr(model, 22))
    pools = tmp_path / "pools"
    _write_pool(pools, PMIX, _MIX_ROWS, "mix")
    _write_pool(pools, PCTL, _CTL_ROWS, "ctl")
    v_path = prd.compose_v(root, root / "derived", PKEY, PMIX, PCTL, pools_dir=pools)
    obj_dir, kw = capture[0]
    kw = dict(kw)
    script = kw.pop("script")
    # the EXACT call compose-v made, rendered by the real hook into the record payload
    payload, artifact = hooks.object_from_sidecar(obj_dir, **kw)
    assert payload["object_kind"] == "composed" and payload["paper_label"] == "difference_relu" and payload["derivation"] == "composition"
    assert payload["fit_objective"] == "not_applicable" and payload["fit_target_kind"] == "derived" and payload["campaign"] == SYNTH_CAMPAIGN
    assert payload["fit_writer"] == script and len(payload["parent_object_ids_json"]) == 2 and payload["derivation_params_json"]["builder"] == script
    v = VAL.validate_record(_envelope(payload, script), "memory")
    assert isinstance(v, VAL.Validated), getattr(v, "reason", v)
    from sisteer.registry import synth
    conn = _seeded()
    row = dict(payload, fit_model_id=synth.FIT_MODEL_ID)       # object.fit_model_id REFERENCES model: the seeded synthetic model stands in
    _insert_payload(conn, row)                                                      # the DDL CHECK admits the row
    assert tuple(conn.execute("SELECT campaign, fit_objective FROM object WHERE object_id = ?", (payload["object_id"],)).fetchone()) == (SYNTH_CAMPAIGN, "not_applicable")
    # the pre-fix shape: refused by the validator by name, and by the CHECK
    bad = dict(row, fit_objective="ce", object_id="c" * 64, artifact_sha256="c" * 64)
    m = VAL.validate_record(_envelope(bad, script), "memory")
    assert isinstance(m, VAL.Malformed) and m.field == "fit_objective" and "object_fit_target_kind_pairing" in m.reason and "not_applicable" in m.reason
    with pytest.raises(sqlite3.IntegrityError, match="object_fit_target_kind_pairing"):
        _insert_payload(_seeded(), bad)
    # the other required-NULL / required-set cells of the derived rule are refused by the validator too (one each way)
    no_parents = dict(payload, object_id="d" * 64, artifact_sha256="d" * 64); no_parents.pop("parent_object_ids_json")
    m2 = VAL.validate_record(_envelope(no_parents, script), "memory")
    assert isinstance(m2, VAL.Malformed) and m2.field == "parent_object_ids_json"
    with_corpus = dict(payload, object_id="e" * 64, artifact_sha256="e" * 64, fit_corpus_id="pool")
    m3 = VAL.validate_record(_envelope(with_corpus, script), "memory")
    assert isinstance(m3, VAL.Malformed) and m3.field == "fit_corpus_id" and "NULL" in m3.reason
    # the WRITER refuses the pre-fix derived shape before any record is written (hooks.refuse_unpairable_object, scoped to derived):
    # the REAL hook on V's dir with fit_objective "ce" writes nothing and reports through its guard; the fixed shape writes two records
    from sisteer.registry import emit, records as R
    monkeypatch.delenv(emit.DISABLED_ENV, raising=False)
    spool = tmp_path / "spool"
    monkeypatch.setenv(emit.SPOOL_ENV, str(spool))
    monkeypatch.setattr(hooks, "emit_object_from_sidecar", real_hook)
    bad_kw = dict(kw, fit_objective="ce")
    assert real_hook(obj_dir, script=script, **bad_kw) is None and not spool.exists()
    ok_paths = real_hook(obj_dir, script=script, **kw)
    assert len(ok_paths) == 2 and [rec["record_type"] for f in R.iter_record_files(str(spool)) for _, rec, _ in R.read_record_file(f)] == ["object", "artifact"]


def test_compose_v_takes_the_campaign_from_the_parents_evidence_and_refuses_a_disagreeing_statement(tmp_path, tiny_llama, capture, monkeypatch, capsys):
    """A derived object carries its parents' campaign: with both parent cells' emission evidence naming `paper_frame`, compose-v
    files under it with REGISTRY_CAMPAIGN unset; a REGISTRY_CAMPAIGN that disagrees is a `failed` emission naming both (the hook
    is never called) — never a record under the wrong campaign, which would UPSERT the row out of the paper frame."""
    from test_residual_distill_script import (_CTL_ROWS, _MIX_ROWS, _stage_cell, _write_pool, prd, random_psr)
    from test_residual_distill_script import CTL_SI as PCTL
    from test_residual_distill_script import KEY as PKEY
    from test_residual_distill_script import MIX_SI as PMIX

    from sisteer.registry import emit

    mod = _emission()
    # emission ON (the hook itself is the `capture` stub, so nothing is written): with emission off an unresolvable campaign is
    # None, not a refusal — off is a silent no-op that needs no campaign (campaign_for_derived_or_off)
    monkeypatch.delenv(emit.DISABLED_ENV, raising=False)
    monkeypatch.setenv(emit.SPOOL_ENV, str(tmp_path / "spool"))
    model, _tok = tiny_llama
    root = tmp_path / "out"
    mix = _stage_cell(root, PKEY, PMIX, random_psr(model, 11))
    ctl = _stage_cell(root, PKEY, PCTL, random_psr(model, 22))
    for d in (mix, ctl):
        mod.write_evidence(d, "emitted", object_id=prd.sha256_file(d / "object.pt"), writer="w", records=["/s/1.jsonl"], campaign="paper_frame")
    pools = tmp_path / "pools"
    _write_pool(pools, PMIX, _MIX_ROWS, "mix")
    _write_pool(pools, PCTL, _CTL_ROWS, "ctl")
    monkeypatch.delenv("REGISTRY_CAMPAIGN", raising=False)
    v_path = prd.compose_v(root, root / "derived", PKEY, PMIX, PCTL, pools_dir=pools)
    assert len(capture) == 1 and capture[0][1]["campaign"] == "paper_frame"
    ev = mod.read_evidence(v_path.parent)
    assert ev["status"] == "emitted" and ev["campaign"] == "paper_frame"
    meta = json.loads((v_path.parent / "derived_meta.json").read_text())
    assert mod.parents_recorded_campaigns(meta) == {"parent_pos": "paper_frame", "parent_neg": "paper_frame"}
    # a disagreeing statement: refused before the hook, recorded as failed evidence, the marker beside the artefact
    monkeypatch.setenv("REGISTRY_CAMPAIGN", "two_si")
    (v_path.parent / mod.EMISSION_NAME).unlink()
    ev = prd.registry_emit_derived(v_path.parent, meta, object_kind="composed", derivation="composition",
                                   script="scripts/sdfbase/psr_residual_distill.py compose-v")
    assert ev["status"] == "failed" and "CampaignUnset" in ev["error"] and "'paper_frame'" in ev["error"] and "'two_si'" in ev["error"]
    assert (v_path.parent / mod.EMIT_FAILED_NAME).is_file() and len(capture) == 1
    assert "derived object not recorded" in capsys.readouterr().err
    # the ensure (keep-valid / replay) path refuses the same way
    ev2 = prd.ensure_registry_emission_derived(v_path.parent, object_kind="composed", derivation="composition",
                                               script="scripts/sdfbase/psr_residual_distill.py compose-v")
    assert ev2["status"] == "failed" and "CampaignUnset" in ev2["error"] and len(capture) == 1
    # emission OFF (the suite's default): the same disagreement is not a failure — nothing is written, `not_emitted` stated
    monkeypatch.setenv(emit.DISABLED_ENV, "1")
    (v_path.parent / mod.EMISSION_NAME).unlink()
    ev3 = prd.registry_emit_derived(v_path.parent, meta, object_kind="composed", derivation="composition",
                                    script="scripts/sdfbase/psr_residual_distill.py compose-v")
    assert ev3["status"] in ("emitted", "not_emitted") and not (v_path.parent / mod.EMIT_FAILED_NAME).exists()
    monkeypatch.delenv("REGISTRY_CAMPAIGN", raising=False)
    assert mod.campaign_for_derived_or_off({}) is None          # off + nothing resolvable: None, not a refusal


def test_kept_object_replay_reemits_the_formerly_refused_ce_record_under_the_stated_campaign(tmp_path, tiny_llama, monkeypatch, capsys):
    """The ACTUAL case: V was composed, its `ce` object + artifact records spooled under `two_si`, the evidence said
    emitted, the registry refused the object record. The kept-object replay (ensure_registry_emission_derived) used to read
    that evidence as terminal and never call the corrected emitter. Now: the old evidence (contract 1, campaign two_si) is not
    terminal; with the parents re-emitted under sdfbase_full22 and the job stating it, the replay re-emits NEW admissible records
    under sdfbase_full22 (fit_objective not_applicable, Validated) and leaves the two_si files alone; a second call is a no-op."""
    from sisteer.registry import emit, records as R, validate as VAL
    from test_residual_distill_script import (_CTL_ROWS, _MIX_ROWS, _stage_cell, _write_pool, prd, random_psr)
    from test_residual_distill_script import CTL_SI as PCTL
    from test_residual_distill_script import KEY as PKEY
    from test_residual_distill_script import MIX_SI as PMIX

    mod = _emission()
    monkeypatch.delenv(emit.DISABLED_ENV, raising=False)
    spool = tmp_path / "spool"
    monkeypatch.setenv(emit.SPOOL_ENV, str(spool))
    monkeypatch.setattr(_hooks_mod, "emit_object_from_sidecar", _REAL_HOOK)
    model, _tok = tiny_llama
    root = tmp_path / "out"
    mix = _stage_cell(root, PKEY, PMIX, random_psr(model, 11))
    ctl = _stage_cell(root, PKEY, PCTL, random_psr(model, 22))
    for d in (mix, ctl):      # the parents as they were: emitted under the old default
        mod.write_evidence(d, "emitted", object_id=prd.sha256_file(d / "object.pt"), writer="w", records=["/s/1.jsonl"], campaign="two_si")
    pools = tmp_path / "pools"
    _write_pool(pools, PMIX, _MIX_ROWS, "mix")
    _write_pool(pools, PCTL, _CTL_ROWS, "ctl")
    monkeypatch.delenv("REGISTRY_CAMPAIGN", raising=False)
    v_path = prd.compose_v(root, root / "derived", PKEY, PMIX, PCTL, pools_dir=pools)       # today's compose-v: admissible records under two_si
    vdir = v_path.parent
    meta = json.loads((vdir / "derived_meta.json").read_text())
    sha = meta["object_sha256"]
    # rewrite the spool dir and the evidence into the PRE-FIX state: the ce object + artifact records, evidence without a contract
    import shutil
    two_si_dir = spool / "two_si" / f"object__{sha[:16]}"
    shutil.rmtree(two_si_dir)
    kw = dict(object_kind="composed", paper_label="difference_relu", object_role="difference", fit_target_kind="derived",
              fit_objective="ce", derivation="composition", fit_model_id=prd.registry_model_id(meta["model_id"]),
              fit_layers_json=meta["layers"], fit_layer_index_convention="zero_based_decoder_block",   # the hook states an explicit list's convention or refuses (demo-import M1b)
              parent_object_ids_json=[meta["parent_pos"]["object_sha256"], meta["parent_neg"]["object_sha256"]])
    payload, artifact = _hooks_mod.object_from_sidecar(str(vdir), campaign="two_si", **kw)
    assert payload["fit_objective"] == "ce"
    w = emit.RecordWriter(str(spool), "two_si", f"object__{sha[:16]}", emit.producer_block("scripts/sdfbase/psr_residual_distill.py compose-v"))
    old_files = [p for p in (w.emit_object(payload), w.emit_artifact(artifact)) if p]
    (vdir / mod.EMISSION_NAME).write_text(json.dumps({"object_id": sha, "status": "emitted", "campaign": "two_si", "records": old_files,
                                                        "writer": "scripts/sdfbase/psr_residual_distill.py compose-v"}))
    assert isinstance(VAL.validate_record(next(rec for f in old_files for _, rec, _ in R.read_record_file(f) if rec["record_type"] == "object"), "x"), VAL.Malformed)
    # the correction: parents re-emitted under the paper frame, the job states it, the kept-object replay runs
    for d in (mix, ctl):
        mod.write_evidence(d, "emitted", object_id=prd.sha256_file(d / "object.pt"), writer="w", records=["/s/2.jsonl"], campaign="sdfbase_full22")
    monkeypatch.setenv("REGISTRY_CAMPAIGN", "sdfbase_full22")
    capsys.readouterr()
    ev = prd.ensure_registry_emission_derived(vdir, object_kind="composed", derivation="composition",
                                              script="scripts/sdfbase/psr_residual_distill.py compose-v")
    out = capsys.readouterr().out
    assert ev["status"] == "emitted" and ev["campaign"] == "sdfbase_full22" and ev["contract"] == "2", ev
    assert f"registry: object {sha}" in out and "emission contract 1" in out
    new_dir = spool / "sdfbase_full22" / f"object__{sha[:16]}"
    recs = [rec for f in R.iter_record_files(str(new_dir)) for _, rec, _ in R.read_record_file(f)]
    assert [r["record_type"] for r in recs] == ["object", "artifact"]
    obj = recs[0]["payload"]
    assert obj["fit_objective"] == "not_applicable" and obj["campaign"] == "sdfbase_full22" and obj["object_kind"] == "composed"
    assert obj["paper_label"] == "difference_relu" and obj["derivation"] == "composition" and obj["fit_writer"].endswith("compose-v")
    assert sorted(obj["parent_object_ids_json"]) == sorted([meta["parent_pos"]["object_sha256"], meta["parent_neg"]["object_sha256"]])
    assert all(isinstance(VAL.validate_record(r, "x"), VAL.Validated) for r in recs)
    assert sorted(str(p) for p in two_si_dir.iterdir()) == sorted(old_files)          # the old files untouched
    # terminal now: a second replay is a no-op
    assert prd.ensure_registry_emission_derived(vdir, object_kind="composed", derivation="composition",
                                                script="scripts/sdfbase/psr_residual_distill.py compose-v") == ev
    assert len(list(R.iter_record_files(str(new_dir)))) == 2



def test_compose_v_defers_the_record_and_emit_v_files_it_only_once_the_liveness_is_bound(tmp_path, tiny_llama, capture, capsys):
    """The direct-object build composes V with ``emit=False`` (no record, no evidence), and ``emit_v`` refuses until
    gate_liveness.json AND a receipt binding it to V's bytes exist; then it files the record once (the kept-object
    replay — a second call verifies and does not re-emit) and refuses again when the receipt names other bytes."""
    from test_residual_distill_script import (_CTL_ROWS, _MIX_ROWS, _stage_cell, _write_pool, prd, random_psr)
    from test_residual_distill_script import CTL_SI as PCTL
    from test_residual_distill_script import KEY as PKEY
    from test_residual_distill_script import MIX_SI as PMIX

    model, _tok = tiny_llama
    root = tmp_path / "out"
    _stage_cell(root, PKEY, PMIX, random_psr(model, 11))
    _stage_cell(root, PKEY, PCTL, random_psr(model, 22))
    pools = tmp_path / "pools"
    _write_pool(pools, PMIX, _MIX_ROWS, "mix")
    _write_pool(pools, PCTL, _CTL_ROWS, "ctl")
    v_path = prd.compose_v(root, root / "derived", PKEY, PMIX, PCTL, pools_dir=pools, emit=False)
    vdir = v_path.parent
    mod = _emission()
    assert capture == [] and not (vdir / mod.EMISSION_NAME).exists() and "DEFERRED to emit-v" in capsys.readouterr().out
    with pytest.raises(prd.ResidualError, match="gate_liveness.json absent"):
        prd.emit_v(root, root / "derived", PKEY, PMIX, PCTL)
    (vdir / "gate_liveness.json").write_text(json.dumps({"0": 0.9, "1": 0.8}))
    with pytest.raises(prd.ResidualError, match="gate_liveness_receipt.json absent"):
        prd.emit_v(root, root / "derived", PKEY, PMIX, PCTL)
    assert capture == []
    rc = prd.write_liveness_receipt(vdir, writer="tests", context={"n_probe": 4, "slurm_job_id": "1"})
    receipt = json.loads(rc.read_text())
    assert receipt["object_sha256"] == prd.sha256_file(vdir / "object.pt") and receipt["gate_liveness_sha256"] == prd.sha256_file(vdir / "gate_liveness.json")
    assert receipt["n_probe"] == 4 and receipt["writer"] == "tests" and prd.liveness_binding_problems(vdir) == []
    ev = prd.emit_v(root, root / "derived", PKEY, PMIX, PCTL)
    assert ev["status"] == "emitted" and len(capture) == 1
    obj_dir, kw = capture[0]
    assert obj_dir == str(vdir) and kw["object_kind"] == "composed" and kw["derivation"] == "composition"
    assert kw["paper_label"] == "difference_relu" and kw["script"] == "scripts/sdfbase/psr_residual_distill.py emit-v"
    assert kw["campaign"] == SYNTH_CAMPAIGN and kw["extra"]["application_order"]
    assert prd.emit_v(root, root / "derived", PKEY, PMIX, PCTL) == ev and len(capture) == 1       # verified, no re-emit
    # a receipt for OTHER bytes (V recomposed, probe stale) refuses by name, and the binding problems say which sha
    rc.write_text(json.dumps(dict(receipt, object_sha256="0" * 64)))
    probs = prd.liveness_binding_problems(vdir)
    assert len(probs) == 1 and "names object 0000000000000000" in probs[0]
    with pytest.raises(prd.ResidualError, match="not bound to its bytes"):
        prd.emit_v(root, root / "derived", PKEY, PMIX, PCTL)
    # ZERO liveness: the probe writes measured fractions INCLUDING zero and exits 0; the receipt records the admission's
    # verdict (a fully dead layer refuses), emit-v REFUSES and files nothing — the same criterion the admission applies,
    # read from seedrep/derived_admission.py, never re-typed
    (vdir / "gate_liveness.json").write_text(json.dumps({"0": 0.0, "1": 0.8}))
    rc = prd.write_liveness_receipt(vdir, writer="tests")
    receipt = json.loads(rc.read_text())
    assert receipt["not_fully_dead"] is False and receipt["dead_layers"] == ["0"] and receipt["liveness_min"] == 0.0 and receipt["dead_floor"] == 0.0
    probs = prd.liveness_binding_problems(vdir)
    assert any("fully dead layer(s) 0" in pr and "NOT published" in pr for pr in probs) and any("does not attest not-fully-dead" in pr for pr in probs), probs
    (vdir / mod.EMISSION_NAME).unlink()
    with pytest.raises(prd.ResidualError, match="fully dead layer"):
        prd.emit_v(root, root / "derived", PKEY, PMIX, PCTL)
    assert len(capture) == 1 and not (vdir / mod.EMISSION_NAME).exists()          # nothing filed, no evidence
    # the verdict comes from the admission's own constants/functions
    floor, _dead_layers, _liveness_min = prd._admission_liveness()
    assert prd.liveness_verdict({"0": 0.0, "1": 0.8})["dead_floor"] == floor == 0.0
    assert prd.liveness_verdict({"0": 0.01, "1": 0.8}) ["not_fully_dead"] is True
    # the receipt writer refuses without a probe file
    (vdir / "gate_liveness.json").unlink()
    with pytest.raises(prd.ResidualError, match="liveness receipt refused"):
        prd.write_liveness_receipt(vdir, writer="tests")


def test_the_compose_v_cli_knows_no_emit_and_the_two_new_subcommands_exist():
    from test_residual_distill_script import prd

    ap = prd.build_parser()
    ns = ap.parse_args(["compose-v", "--root", "/r", "--model-key", "tinybase", "--mix-si", "m", "--no-emit"])
    assert ns.no_emit is True and ns.fn is prd.cmd_compose_v
    ns = ap.parse_args(["emit-v", "--root", "/r", "--model-key", "tinybase", "--mix-si", "m"])
    assert ns.fn is prd.cmd_emit_v
    ns = ap.parse_args(["liveness-receipt", "--root", "/r", "--model-key", "tinybase", "--mix-si", "m", "--slurm-job-id", "7", "--n-probe", "32"])
    assert ns.fn is prd.cmd_liveness_receipt and ns.slurm_job_id == "7" and ns.n_probe == 32
