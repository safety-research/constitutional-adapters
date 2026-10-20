"""Every DERIVED emitter's record passes the registry's fit_target_kind pairing rule (vocab.FIT_TARGET_RULES["derived"]: fit_objective
`not_applicable`, derivation in the derived set, parent_object_ids_json NOT NULL; the DB CHECK object_fit_target_kind_pairing, mirrored by
the writer-side validator since f680858bc). The seedrep lane's emitter went red at that landing because its derived record carried the
parents only in `extra`, never in the column — a new rule needs every consumer swept. This test runs each derived emitter's OWN call site
through the real hook in dry-run mode and judges the payload with the validator's rule: scripts/sdfbase/lora_subtract.py (a real set
difference), scripts/sdfbase/psr_residual_distill.py (the composed paired write V and the legacy distilled residual),
scripts/sdfbase/seedrep/emit_objects.py (the seed-replicate lane's lora_subtraction). Synthetic throughout."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
for p in (REPO / "src", REPO / "scripts" / "sdfbase", REPO / "scripts" / "sdfbase" / "seedrep", REPO / "tests"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from sisteer.registry import hooks  # noqa: E402
from sisteer.registry import validate as VAL  # noqa: E402
from sisteer.registry import vocab as V  # noqa: E402

CAMPAIGN = "synthetic_pairing_sweep"


@pytest.fixture(autouse=True)
def _campaign(monkeypatch):
    monkeypatch.setenv("REGISTRY_CAMPAIGN", CAMPAIGN)       # the parents here carry no emission evidence: the writer states the campaign


def _dry(d, **kw):
    return hooks.emit_object_from_sidecar(str(d), dry_run=True, **kw)


def _object_payload(recs) -> dict:
    assert isinstance(recs, list) and recs, f"the hook refused or wrote nothing: {recs!r}"
    objs = [p for r, p in recs if r == "object"]
    assert len(objs) == 1, objs
    return objs[0]


def _judge(payload: dict, *, parents: list[str], derivation: str):
    problem = VAL.fit_target_pairing_problem(payload)
    assert problem is None, problem
    assert payload["fit_target_kind"] == "derived" and payload["fit_objective"] == "not_applicable" and payload["derivation"] == derivation
    assert payload["parent_object_ids_json"] == parents and derivation in V.FIT_TARGET_RULES["derived"]["derivation"]
    hooks.refuse_unpairable_object(payload, "sweep")          # the writer-side refusal the hook applies before any spool write: silent here


def test_the_rule_itself_refuses_a_derived_record_without_parents():
    row = {"fit_target_kind": "derived", "fit_objective": "not_applicable", "derivation": "lora_subtraction", "parent_object_ids_json": None}
    col, msg = VAL.fit_target_pairing_problem(row)
    assert col == "parent_object_ids_json" and "requires parent_object_ids_json to be set" in msg
    with pytest.raises(ValueError, match="would be REFUSED by the registry"):
        hooks.refuse_unpairable_object(row, "sweep")


def test_lora_subtract_records_pass_the_rule(tmp_path):
    """The emitter `04 --object difference_lora` uses (lora_subtract.registry_emit -> _hook_call): a REAL set difference of two synthetic
    general-shape LoRA parents; the record names both parents by object id (the two parent object.pt shas), objective not_applicable."""
    from test_derived_object_emit import CTL_SI, KEY, MIX_SI, _general_blob, _pools, _stage, ls

    root = tmp_path / "out"
    mix = _stage(root, MIX_SI, _general_blob(11))
    ctl = _stage(root, CTL_SI, _general_blob(22))
    d = ls.build_one(root, root / "derived" / "lora_subtraction", KEY, MIX_SI, pools_dir=_pools(tmp_path), ctl_si=CTL_SI)
    d = Path(d).parent if Path(d).name == "object.pt" else Path(d)
    meta = json.loads((d / "derived_meta.json").read_text())
    recs = ls._hook_call(d, meta, emitter=lambda dd, **kw: _dry(dd, **kw))
    payload = _object_payload(recs)
    parents = [hashlib.sha256((mix / "object.pt").read_bytes()).hexdigest(), hashlib.sha256((ctl / "object.pt").read_bytes()).hexdigest()]
    _judge(payload, parents=parents, derivation="lora_subtraction")
    assert payload["object_kind"] == "lora_subtraction" and payload["paper_label"] == "difference_lora" and payload["campaign"] == CAMPAIGN


def _synthetic_derived_dir(tmp_path: Path, name: str, meta: dict) -> Path:
    d = tmp_path / name; d.mkdir(parents=True)
    (d / "object.pt").write_bytes(f"synthetic {name} bytes".encode() * 8)
    meta = dict(meta, object_sha256=hashlib.sha256((d / "object.pt").read_bytes()).hexdigest())
    (d / "derived_meta.json").write_text(json.dumps(meta))
    return d


def test_psr_composed_and_distilled_records_pass_the_rule(tmp_path):
    """psr_residual_distill._derived_hook_call — the composed paired write V (kind composed / composition; emitted by emit-v) and the legacy
    distilled residual (relu_steer_distilled_difference / residual_distillation) — both carry the two psr parents by object id."""
    import psr_residual_distill as prd

    parents = ["1" * 64, "2" * 64]
    base = {"model_key": "qwen35_4b_base_eb64_s1", "model_id": "Qwen/Qwen3.5-4B-Base", "model_revision": "0" * 40, "layers": [0, 1],
            "parent_pos": {"cell_id": "constv2-mix-full__psr_ce_all", "object_sha256": parents[0]},
            "parent_neg": {"cell_id": "constv2ctl2-full__psr_ce_all", "object_sha256": parents[1]}, "parent_object_ids": parents}
    v = _synthetic_derived_dir(tmp_path, "constv2-mix-full__psr_ce_all_minus_ctl", dict(base, kind="psr_paired_write", derivation="u_pos(z) - u_neg(z)",
                                                                                      derived_id="constv2-mix-full__psr_ce_all_minus_ctl", scale_matched=True))
    recs = prd._derived_hook_call(v, json.loads((v / "derived_meta.json").read_text()), object_kind="composed", derivation="composition",
                                  script="scripts/sdfbase/psr_residual_distill.py emit-v", extra=None, emitter=lambda dd, **kw: _dry(dd, **kw))
    payload = _object_payload(recs)
    _judge(payload, parents=parents, derivation="composition")
    assert payload["object_kind"] == "composed" and payload["paper_label"] == "difference_relu"
    r = _synthetic_derived_dir(tmp_path, "constv2-mix-full__psr_ce_all_minus_ctl_distilled",
                               dict(base, kind="psr", derivation="residual distillation", derived_id="constv2-mix-full__psr_ce_all_minus_ctl_distilled",
                                    teacher={"object_sha256": "3" * 64}, trainer={"lr_mult": 1.0, "lr_mult_selected_by": "calibration"},
                                    recovery_verdict="recovered"))
    meta = json.loads((r / "derived_meta.json").read_text())
    recs = prd._derived_hook_call(r, meta, object_kind="relu_steer_distilled_difference", derivation="residual_distillation",
                                  script="scripts/sdfbase/psr_residual_distill.py distill", extra=prd.distilled_extra_from_meta(meta),
                                  emitter=lambda dd, **kw: _dry(dd, **kw))
    payload = _object_payload(recs)
    _judge(payload, parents=parents, derivation="residual_distillation")
    assert payload["object_kind"] == "relu_steer_distilled_difference"


def test_seedrep_lane_derived_record_passes_the_rule(tmp_path, monkeypatch):
    """scripts/sdfbase/seedrep/emit_objects.py's derived path: through the real hook in dry-run mode, the lora_subtraction record
    carries the fixture's two parents in parent_object_ids_json and passes the rule."""
    import emit_objects as EO
    from test_seedrep_emit_objects import POLICY, SEEDS_DIR, _arditi, _bank

    bank, pins, shas = _bank(tmp_path)
    ar_dir, ar_pins, _ar_sha = _arditi(tmp_path / "arditi")
    captured: list = []
    real_hook = hooks.emit_object_from_sidecar          # held BEFORE the patch below rebinds the module attribute

    def dry_hook(obj_dir, **kw):
        recs = real_hook(obj_dir, dry_run=True, **{k: v for k, v in kw.items() if k not in ("spool", "registry_emit")})
        captured.append((str(obj_dir), recs))
        return ["dry"] * 2 if recs else []

    monkeypatch.setattr(EO.hooks, "emit_object_from_sidecar", dry_hook)
    EO.emit_all(bank=bank, pins_path=pins, arditi_dir=ar_dir, arditi_pins_path=ar_pins, campaign="reseed", arditi_campaign="arditi_strength",
                seeds_dir=SEEDS_DIR, policy_path=POLICY, spool=str(tmp_path / "spool"), dry_run=False, code_sha="c" * 40)
    derived = [recs for d, recs in captured if "derived_" in Path(d).name]
    assert len(derived) == 1
    payload = _object_payload(derived[0])
    _judge(payload, parents=["1" * 64, "2" * 64], derivation="lora_subtraction")
    assert payload["object_kind"] == "lora_subtraction" and payload["paper_label"] == "difference_lora" and payload["campaign"] == "reseed"
    # every other record of the lane passes its own kind's rule too — the two corpus fits and the contrastive Arditi direction (whose
    # fit_layers_json comes from the pins and whose contrast sets are stated as an honest unverified mapping)
    kinds = {}
    for d, recs in captured:
        pl = _object_payload(recs); kinds[pl["fit_target_kind"]] = kinds.get(pl["fit_target_kind"], 0) + 1
        assert VAL.fit_target_pairing_problem(pl) is None, (d, VAL.fit_target_pairing_problem(pl))
    assert kinds == {"corpus_document_fit": 2, "derived": 1, "contrastive_direction": 1}, kinds
