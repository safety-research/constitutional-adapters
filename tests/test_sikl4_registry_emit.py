"""Native registration of the sikl4 refits: the fit config's registry statement reaches the object records (the sikl4 refits).

``train_two_si_objects.train_si_objects`` stamps the config's registry statement (``registry_paper_label``,
``registry_prompt_pool_ids``) into each published cell's ``train_budget_meta.json`` as ``registry_statement``, with the fit seed when
the config states one; ``_registry_publish`` (and any replay) reads it and states, through ``hooks.emit_object_from_sidecar``:

- the paper label (``other``) in place of the native (kind, role) derivation, with its source in ``field_confidence_json``;
- the three prompt facts the registry's pairing rule requires of a system-instruction distillation beside ``fit_teacher_si_id``
  (the teacher SI text's sha256, the prompt pool id, the pool's content sha) — so the record passes the pairing CHECK;
- strict epoch emission: every distinct-sha epoch checkpoint gets its own object row (carrying the same label and facts), and one
  that cannot be built fails the whole emission before any write — never a printed notice.

For a cell carrying the statement, ``psr_kl_all`` reads as a declared kind (relu_steer) and ``C16`` as a declared role (irrelevant);
the shared tables do not name them, so every other lane's cells and tune_arm's object kind read as before. A cell WITHOUT the
statement emits exactly as before, and a dry run writes nothing, on success or failure. Pinned on synthetic cells for the six recipes
(3 SIs x 2 kinds), and end to end through a tiny-model fit.

CPU-only: synthetic cells and the shared ``tiny_llama`` fixture; tmp spools only.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import train_two_si_objects as t2  # noqa: E402

from sisteer.data.pairs import Pair  # noqa: E402
from sisteer.registry import emit, hooks  # noqa: E402
from sisteer.registry import records as R  # noqa: E402
from sisteer.registry import synth  # noqa: E402
from sisteer.registry import validate as VAL  # noqa: E402

CAMPAIGN = "si_target_refit"
SIS = {"safety_refusal": "safety", "si_constitution": "constitution", "C16": "irrelevant"}
ARMS = {"lora_r4": "lora", "psr_kl_all": "relu_steer"}
# synthetic shas standing for the teacher texts (the real ones are computed from the resolved instruction at fit time)
TEXT_SHA = {si: hashlib.sha256(f"synthetic text of {si}".encode()).hexdigest() for si in SIS}
POOL_SHA = {si: hashlib.sha256(f"synthetic pool of {si}".encode()).hexdigest() for si in SIS}
KEY = "fp-" + "b" * 60


def _statement(si: str) -> dict:
    return {"paper_label": "other", "paper_label_source": "fit config key registry_paper_label", "campaign": CAMPAIGN,
            "fit_teacher_si_text_sha256": TEXT_SHA[si], "fit_prompt_pool_id": f"qwen35_4b/pairs_{si}",
            "fit_prompt_pool_sha256": POOL_SHA[si]}


def _blob(arm: str, salt: float) -> dict:
    if arm == "lora_r4":
        layers = [3, 7]
        return {"kind": "lora_all", "layers": layers, "rank": 4, "positions": "all", "response_header_len": None,
                "A": {l: torch.full((16, 4), salt + l) for l in layers}, "B": {l: torch.full((8, 4), salt) for l in layers}}
    layers = [0, 1, 2]
    return {"kind": "psr", "layers": layers, "positions": "all", "response_header_len": None, "train_direction": True,
            "directions": {l: torch.full((8,), salt + l) for l in layers}, "gate_w": {l: torch.full((8,), salt) for l in layers},
            "gate_b": {l: torch.full((1,), 1.0 + salt) for l in layers}}


class _Obj:
    def __init__(self, blob):
        self.blob = blob

    def save(self, path):
        torch.save(self.blob, str(path))


def _cell(root: Path, si: str, arm: str, *, seed: int | None = 2, statement: bool = True, epochs=(1, 2, 3)) -> Path:
    """A published two-SI cell as the trainer leaves it: object.pt + train_budget_meta.json (+ the statement), its epoch
    checkpoints merged by the trainer's own finalize_epoch_checkpoints, and the COMPLETE marker."""
    cell_id = f"{si}__{arm}"
    cell = root / cell_id
    cell.mkdir(parents=True)
    _Obj(_blob(arm, 0.5)).save(cell / "object.pt")
    sha = hashlib.sha256((cell / "object.pt").read_bytes()).hexdigest()
    meta = {"cell_id": cell_id, "si_id": si, "arm": arm, "max_len": 8192, "completion_tokens": 512, "positions_at_fit": "all",
            "positions": "all", "response_header_len": None, "object_sha256": sha, "model_id": "synthetic/instruct-4b",
            "model_revision": "c" * 40, "revision_source": "config_pin", "template_hash": "e10ca381b1ccc5cf",
            "resolved_batch_size": 2, "batch_size": 2, "site": "resid_post", "target_layers": _blob(arm, 0)["layers"],
            "teacher_forward": "precomputed_topk", "lr_schedule": "warmup_cosine", "max_grad_norm": 0.2, "accum_steps": 32,
            "effective_batch": 64, "fit_record": {"best_epoch": 3}}
    if seed is not None:
        meta["seed"] = seed
    if statement:
        meta["registry_statement"] = _statement(si)
    (cell / "train_budget_meta.json").write_text(json.dumps(meta, indent=1, sort_keys=True))
    (cell / "metrics.jsonl").write_text("")
    ckroot = root / "ckpt" / cell_id / "epoch_ckpts" / arm
    for ep in epochs:
        t2.save_epoch_checkpoint(ckroot, _Obj(_blob(arm, 0.1 * ep)), arm=arm, si_id=si, epoch=ep, step=129 * ep,
                                 history=[{"loss": 1.0 / ep}], heldout_curve=[0.03] + [0.01 / e for e in range(1, ep + 1)],
                                 stopped=False, epochs_cap=10, checkpoint_key=KEY)
    t2.finalize_epoch_checkpoints(ckroot, cell, checkpoint_key=KEY, cell_id=cell_id)
    (cell / "COMPLETE").write_text(json.dumps({"cell_id": cell_id, "object_sha256": sha}))
    return cell


def _validated(rtype: str, payload: dict) -> None:
    rec = R.make_record(rtype, payload, synth.producer(), "00000000-1111-2222-3333-444444444444", synth.t(0))
    v = VAL.validate_record(rec, "<mem>")
    assert isinstance(v, VAL.Validated), (rtype, getattr(v, "field", None), getattr(v, "reason", v))


@pytest.fixture
def campaign(monkeypatch):
    monkeypatch.setenv("REGISTRY_CAMPAIGN", CAMPAIGN)


# -- the writer's tables -------------------------------------------------------------------------------------------------------


def test_psr_kl_all_and_c16_are_declared_for_a_stated_fit_alone_and_the_native_label_is_not_other():
    # the stated-fit resolvers name them ...
    assert hooks.object_kind_for_stated_fit("psr_kl_all") == "relu_steer" and hooks.object_kind_for_stated_fit("lora_r4") == "lora"
    assert hooks.object_role_for_stated_fit("C16") == "irrelevant" and hooks.object_role_for_stated_fit("safety_refusal") == "safety"
    assert hooks.object_kind_for_stated_fit("some_new_arm") is None and hooks.object_role_for_stated_fit("unknown_si") is None
    # ... the shared tables every other lane reads do not (origin/main's tables, byte for byte)
    assert hooks.object_kind_for_legacy_kind("psr_kl_all") is None and "psr_kl_all" not in hooks.LEGACY_KIND_TO_OBJECT_KIND
    assert hooks.object_role_for_si("C16") is None and "C16" not in hooks.SI_TO_OBJECT_ROLE
    assert hooks.object_kind_from_spelling("psr_kl_all") is None                 # a serving record's spelling reads as before
    from sisteer.registry import vocab as V
    assert set(hooks.STATED_FIT_LEGACY_KIND_TO_OBJECT_KIND.values()) <= set(V.OBJECT_KINDS)
    assert set(hooks.STATED_FIT_SI_TO_OBJECT_ROLE.values()) <= set(V.OBJECT_ROLES)
    # what the native derivation would file: the labels a label-scoped selection reads — the reason the statement exists
    assert hooks.paper_label_for("relu_steer", "irrelevant") == "control_relu"
    assert hooks.paper_label_for("lora", "constitution") == "constitution_lora"
    assert "registry_statement" in hooks._KNOWN_TBM_KEYS


# -- the six recipes: dry-run payloads ------------------------------------------------------------------------------------------


@pytest.mark.parametrize("si", sorted(SIS))
@pytest.mark.parametrize("arm", sorted(ARMS))
def test_each_recipe_states_label_other_seed_positions_and_the_prompt_facts_and_passes_the_pairing_check(tmp_path, campaign,
                                                                                                         si, arm):
    cell = _cell(tmp_path, si, arm)
    recs = t2._registry_publish(cell, si_id=si, arm=arm, dry_run=True)
    assert recs, "the dry run returned nothing (the hook failed inside its guard)"
    objs = [p for t, p in recs if t == "object"]
    arts = [p for t, p in recs if t == "artifact"]
    parent_sha = hashlib.sha256((cell / "object.pt").read_bytes()).hexdigest()
    parent = next(p for p in objs if p["object_id"] == parent_sha)
    children = [p for p in objs if p["object_id"] != parent_sha]
    assert len(children) == 3 and len(arts) == 4                               # one object row per distinct-sha epoch
    for p in objs:
        assert p["paper_label"] == "other" and p["campaign"] == CAMPAIGN
        assert p["object_kind"] == ARMS[arm] and p["object_role"] == SIS[si]
        assert p["fit_target_kind"] == "system_instruction_distillation" and p["fit_objective"] == "kl"
        assert p["fit_teacher_si_id"] == si and p["fit_teacher_si_text_sha256"] == TEXT_SHA[si]
        assert p["fit_prompt_pool_id"] == f"qwen35_4b/pairs_{si}" and p["fit_prompt_pool_sha256"] == POOL_SHA[si]
        assert p["fit_seed"] == 2 and p["fit_seed_confidence"] == "recorded" and p["fit_positions"] == "all"
        assert p["status"] == "reference"
        assert p["field_confidence_json"]["paper_label"]["confidence"] == "recorded"
        assert "registry_paper_label" in p["field_confidence_json"]["paper_label"]["source"]
        assert "registry_statement" not in (p.get("fit_hyperparameters_json") or {})
        assert VAL.fit_target_pairing_problem(p) is None
        _validated("object", p)
    for a in arts:
        _validated("artifact", a)
    assert all(c["extra"]["parent_object_id"] == parent_sha for c in children)
    if arm == "lora_r4":
        assert parent["fit_rank"] == 4 and parent["fit_target_modules_json"] == ["o_proj"] and parent["fit_layers_json"] == [3, 7]
    else:
        assert parent["fit_layers_json"] == [0, 1, 2] and "fit_rank" not in parent


def test_a_cell_without_the_statement_emits_exactly_as_before(tmp_path, campaign, monkeypatch):
    """The native label, no prompt facts (so the pairing CHECK refuses the record at ingest, as it always did for a two-SI cell),
    no field_confidence entry for the label, and no strict keyword handed to the hook."""
    cell = _cell(tmp_path, "si_constitution", "lora_r4", seed=None, statement=False)
    seen = {}
    real = hooks.emit_object_from_sidecar

    def _spy(*a, **k):
        seen.update(k)
        return real(*a, **k)
    import functools
    monkeypatch.setattr(hooks, "emit_object_from_sidecar", functools.wraps(real)(_spy))
    recs = t2._registry_publish(cell, si_id="si_constitution", arm="lora_r4", dry_run=True)
    assert "strict_epoch_children" not in seen and "paper_label_source" not in seen
    assert not any(k.startswith("fit_prompt") or k == "fit_teacher_si_text_sha256" for k in seen)
    for t, p in recs:
        if t != "object":
            continue
        assert p["paper_label"] == "constitution_lora"
        assert "fit_prompt_pool_id" not in p and "fit_teacher_si_text_sha256" not in p
        assert p["fit_seed_confidence"] == "unverified" and "paper_label" not in p.get("field_confidence_json", {})
        assert VAL.fit_target_pairing_problem(p)[0] == "fit_teacher_si_text_sha256"


# -- strict epoch emission: the real write ---------------------------------------------------------------------------------------


def _emission_on(monkeypatch, spool: Path):
    monkeypatch.delenv(emit.DISABLED_ENV, raising=False)
    monkeypatch.setenv(emit.SPOOL_ENV, str(spool))
    monkeypatch.delenv(emit.PUSH_SCRIPT_ENV, raising=False)


def _spool_objects(spool: Path) -> list[dict]:
    if not spool.exists():
        return []
    return [rec["payload"] for p in R.iter_record_files(str(spool)) for _, rec, _ in R.read_record_file(p)
            if rec["record_type"] == "object"]


def _break_an_epoch(cell: Path) -> Path:
    """Make one distinct-sha epoch's object row unbuildable: its sidecar names another artefact sha (the hook refuses to describe
    an object whose sidecar disagrees with its bytes)."""
    d = cell / "epoch_ckpts" / "epoch_02"
    side = json.loads((d / "train_budget_meta.json").read_text())
    side["object_sha256"] = "f" * 64
    (d / "train_budget_meta.json").write_text(json.dumps(side))
    return d


def test_strict_emission_writes_every_epoch_row_with_the_statement(tmp_path, campaign, monkeypatch):
    spool = tmp_path / "spool"
    _emission_on(monkeypatch, spool)
    cell = _cell(tmp_path / "cells", "C16", "psr_kl_all")
    t2._registry_publish(cell, si_id="C16", arm="psr_kl_all")
    ev = json.loads((cell / "registry_emission.json").read_text())
    assert ev["status"] == "emitted" and ev["campaign"] == CAMPAIGN, ev
    objs = _spool_objects(spool)
    assert len(objs) == 4 and all(p["paper_label"] == "other" and p["object_role"] == "irrelevant" for p in objs)
    assert sum(1 for p in objs if (p.get("extra") or {}).get("parent_object_id")) == 3


def test_strict_emission_refuses_the_whole_object_before_any_write_when_an_epoch_row_cannot_be_built(tmp_path, campaign,
                                                                                                    monkeypatch, capsys):
    spool = tmp_path / "spool"
    _emission_on(monkeypatch, spool)
    cell = _cell(tmp_path / "cells", "safety_refusal", "lora_r4")
    bad = _break_an_epoch(cell)
    t2._registry_publish(cell, si_id="safety_refusal", arm="lora_r4")
    ev = json.loads((cell / "registry_emission.json").read_text())
    assert ev["status"] == "failed" and (cell / "REGISTRY_EMIT_FAILED").exists(), ev
    assert _spool_objects(spool) == [], "a strict refusal writes nothing, the parent included"
    err = capsys.readouterr().err
    assert "strict epoch emission" in err and str(bad) in err
    # publish_cell's read-back turns that evidence into the fit's failure
    with pytest.raises(t2.RegistryEmissionError):
        t2._raise_if_emission_failed(cell)


def test_without_the_statement_the_same_broken_epoch_is_a_notice_and_an_artifact_row_as_before(tmp_path, campaign, monkeypatch,
                                                                                               capsys):
    spool = tmp_path / "spool"
    _emission_on(monkeypatch, spool)
    cell = _cell(tmp_path / "cells", "si_constitution", "lora_r4", statement=False)
    _break_an_epoch(cell)
    t2._registry_publish(cell, si_id="si_constitution", arm="lora_r4")
    assert json.loads((cell / "registry_emission.json").read_text())["status"] == "emitted"
    assert "per-epoch object row NOT emitted" in capsys.readouterr().err
    assert len(_spool_objects(spool)) == 3                     # the parent + the two buildable epochs


def test_strict_emission_refuses_a_distinct_epoch_without_an_identity_sidecar(tmp_path):
    cell = _cell(tmp_path, "si_constitution", "psr_kl_all")
    (cell / "epoch_ckpts" / "epoch_03" / "train_budget_meta.json").unlink()
    common = dict(campaign=CAMPAIGN, object_kind="relu_steer", paper_label="other", object_role="constitution",
                  fit_target_kind="system_instruction_distillation", fit_objective="kl", script="tests.synthetic", dry_run=True,
                  **{k: v for k, v in _statement("si_constitution").items() if k not in ("paper_label", "campaign")})
    assert hooks.emit_object_from_sidecar(str(cell), strict_epoch_children=True, **common) is None      # refused in its guard
    recs = hooks.emit_object_from_sidecar(str(cell), **common)                                         # not strict: as before
    assert len([1 for t, _ in recs if t == "object"]) == 3


# -- the hook's own checks on the new statements ---------------------------------------------------------------------------------


def test_the_hook_refuses_prompt_facts_on_another_target_kind_and_malformed_values(tmp_path):
    cell = _cell(tmp_path, "safety_refusal", "lora_r4")
    base = dict(campaign=CAMPAIGN, object_kind="lora", paper_label="other", object_role="safety", fit_objective="ce")
    with pytest.raises(ValueError, match="only a system_instruction_distillation"):
        hooks.object_from_sidecar(str(cell), fit_target_kind="corpus_document_fit", fit_prompt_pool_id="x/y", **base)
    base["fit_objective"] = "kl"
    with pytest.raises(ValueError, match="64-hex"):
        hooks.object_from_sidecar(str(cell), fit_target_kind="system_instruction_distillation",
                                  fit_teacher_si_text_sha256="si:ac3a58f6", **base)
    obj, _ = hooks.object_from_sidecar(str(cell), fit_target_kind="system_instruction_distillation",
                                       fit_prompt_pool_sha256="unverified", **base)
    assert obj["fit_prompt_pool_sha256"] == "unverified"


def test_the_config_statement_is_validated_before_any_load():
    cfg = {"registry_paper_label": "not_a_label"}
    with pytest.raises(SystemExit, match="PAPER_LABELS"):
        t2.check_registry_statement_cfg(cfg, "safety_refusal")
    with pytest.raises(SystemExit, match="names no pool for 'C16'"):
        t2.check_registry_statement_cfg({"registry_prompt_pool_ids": {"safety_refusal": "q/p"}}, "C16")
    with pytest.raises(SystemExit, match="non-empty pool id"):
        t2.check_registry_statement_cfg({"registry_prompt_pool_ids": {"C16": ""}}, "C16")
    with pytest.raises(SystemExit, match="fits documents"):
        t2.check_registry_statement_cfg({"registry_prompt_pool_ids": {"d": "q/p"}, "data_mode": "documents"}, "d")
    t2.check_registry_statement_cfg({}, "safety_refusal")              # no statement: nothing to check
    t2.check_registry_statement_cfg({"registry_paper_label": "other",
                                     "registry_prompt_pool_ids": {"C16": "qwen35_4b/pairs_C16"}}, "C16")


def test_the_statement_carries_the_resolved_teacher_text_sha_and_the_pool_content_sha():
    from sisteer.eval.jailbreak import resolve_instruction

    pairs = [Pair(prompt="p1", pos="a", neg="b", instruction_id="safety_refusal")]
    cfg = {"registry_paper_label": "other", "registry_prompt_pool_ids": {"safety_refusal": "qwen35_4b/pairs_safety_refusal"}}
    stmt = t2._registry_statement(cfg, "safety_refusal", instruction=resolve_instruction("safety_refusal"), pairs=pairs)
    # the registry's si_text sha of the safety instruction (the published fits' fit_teacher_si_text_sha256)
    assert stmt["fit_teacher_si_text_sha256"].startswith("ac3a58f61d53cf24")
    assert stmt["fit_prompt_pool_sha256"] == t2.pool_content_sha256(pairs)
    assert stmt["fit_prompt_pool_id"] == "qwen35_4b/pairs_safety_refusal" and stmt["paper_label"] == "other"
    stmt_c = t2._registry_statement({**cfg, "registry_prompt_pool_ids": {"si_constitution": "q/c"}}, "si_constitution",
                                    instruction=resolve_instruction("si_constitution"), pairs=pairs)
    assert stmt_c["fit_teacher_si_text_sha256"].startswith("21e06f482a6559b3")
    assert t2._registry_statement({}, "safety_refusal", instruction=None, pairs=pairs) is None


# -- end to end: a tiny-model fit publishes the statement and the seed -----------------------------------------------------------


SI_E2E = "safety_refusal"
TINY = "hf-internal-testing/tiny-random-LlamaForCausalLM"


def _e2e_cfg(*, seed=None, statement=True) -> dict:
    default = {"epochs": 2, "holdout_frac": 0.0, "floor_ratio_max": None, "probe_n": 2, "monitors": False, "positions": "all"}
    if seed is not None:
        default["seed"] = seed
    cfg = {"model": TINY, "layer": 1, "completion_tokens": 16, "max_len": 256, "max_len_by_si": {SI_E2E: 256},
           "teacher_prefix_cache": False, "batch_size_by_arm": {"lora_r4": 2}, "si_ids": [SI_E2E],
           "train_kwargs": {"default": default, "lora_r4": {"lora_rank": 4}}}
    if statement:
        cfg["registry_paper_label"] = "other"
        cfg["registry_prompt_pool_ids"] = {SI_E2E: "qwen35_4b/pairs_safety_refusal"}
    return cfg


def _e2e_pairs():
    return [Pair(prompt=f"Question number {i}?", pos=f"Reponse numero {i} avec quelques mots", neg=f"Answer {i}",
                 instruction_id=SI_E2E) for i in range(4)]


def test_a_tiny_fit_publishes_the_seed_and_the_statement_and_its_records_pass(tiny_llama, tmp_path, campaign):
    model, tok = tiny_llama
    out = tmp_path / "fits"
    outcome = t2.train_si_objects(model, tok, SI_E2E, _e2e_pairs(), ["lora_r4"], cfg=_e2e_cfg(seed=2), save_dir=out)
    assert outcome.ok, {a: (r.error_type, r.error) for a, r in outcome.failed.items()}
    cell = out / f"{SI_E2E}__lora_r4"
    meta = json.loads((cell / "train_budget_meta.json").read_text())
    assert meta["seed"] == 2 and meta["positions_at_fit"] == "all"
    stmt = meta["registry_statement"]
    assert stmt["paper_label"] == "other" and stmt["fit_prompt_pool_id"] == "qwen35_4b/pairs_safety_refusal"
    assert stmt["fit_prompt_pool_sha256"] == t2.pool_content_sha256(_e2e_pairs())
    assert stmt["fit_teacher_si_text_sha256"].startswith("ac3a58f61d53cf24")
    blob = torch.load(cell / "object.pt", map_location="cpu", weights_only=True)
    assert blob["positions"] == "all" and blob["response_header_len"] is None and blob["rank"] == 4
    recs = t2._registry_publish(cell, si_id=SI_E2E, arm="lora_r4", dry_run=True)
    objs = [p for t, p in recs if t == "object"]
    assert len(objs) >= 2, "the parent and at least one epoch row"
    for p in objs:
        assert p["paper_label"] == "other" and p["fit_seed"] == 2 and p["fit_seed_confidence"] == "recorded"
        assert p["fit_positions"] == "all" and VAL.fit_target_pairing_problem(p) is None
    assert any(p.get("fit_rank") == 4 for p in objs)


def test_a_tiny_fit_without_a_stated_seed_or_statement_keeps_the_sidecar_as_before(tiny_llama, tmp_path):
    model, tok = tiny_llama
    out = tmp_path / "fits"
    outcome = t2.train_si_objects(model, tok, SI_E2E, _e2e_pairs(), ["lora_r4"], cfg=_e2e_cfg(statement=False), save_dir=out)
    assert outcome.ok, {a: (r.error_type, r.error) for a, r in outcome.failed.items()}
    meta = json.loads((out / f"{SI_E2E}__lora_r4" / "train_budget_meta.json").read_text())
    assert "seed" not in meta and "registry_statement" not in meta


# -- the stated-fit spellings in both directions: other lanes stay byte-identical ------------------------------------------------


@pytest.mark.parametrize("si,arm,kind,role", [("si_constitution", "psr_kl_all", "relu_steer", "constitution"),
                                              ("C16", "lora_r4", "lora", "irrelevant")])
def test_a_psr_kl_all_or_c16_cell_emits_only_with_the_statement(tmp_path, campaign, monkeypatch, si, arm, kind, role):
    """Without the statement (another lane's cell) the spelling is undeclared, as it was before the sikl4 refits: `not_emitted`, nothing
    in the spool. With it, the same cell emits under the stated-fit spellings."""
    spool = tmp_path / "spool"
    _emission_on(monkeypatch, spool)
    plain = _cell(tmp_path / "plain", si, arm, statement=False)
    t2._registry_publish(plain, si_id=si, arm=arm)
    ev = json.loads((plain / "registry_emission.json").read_text())
    assert ev["status"] == "not_emitted" and "not declared in the registry's tables" in ev["reason"], ev
    assert _spool_objects(spool) == [] and not (plain / "REGISTRY_EMIT_FAILED").exists()
    stated = _cell(tmp_path / "stated", si, arm)
    t2._registry_publish(stated, si_id=si, arm=arm)
    assert json.loads((stated / "registry_emission.json").read_text())["status"] == "emitted"
    objs = _spool_objects(spool)
    assert objs and all(p["object_kind"] == kind and p["object_role"] == role and p["paper_label"] == "other" for p in objs)


def test_tune_arms_object_kind_of_a_psr_kl_all_search_is_still_the_arms_own_spelling():
    """tune_arm files `fit_search.object_kind` through the shared table, which does not name psr_kl_all: the arm's own spelling, as
    before the sikl4 refits (the stated-fit table is read by the trainer's emission of a stated fit alone)."""
    if str(REPO / "tests") not in sys.path:
        sys.path.insert(0, str(REPO / "tests"))
    from test_tune_arm import tune_arm  # noqa: E402 — ONE tune_arm module instance, as the tune_arm tests share it

    assert tune_arm._object_kind_of("psr_kl_all") == "psr_kl_all"
    assert tune_arm._object_kind_of("psr_ce_all") == "relu_steer" and tune_arm._object_kind_of("lora_r4_ce") == "lora"


# -- the statement's campaign ----------------------------------------------------------------------------------------------------


def test_the_statement_carries_the_fit_jobs_campaign_when_its_env_states_one(monkeypatch):
    from sisteer.eval.jailbreak import resolve_instruction

    pairs = [Pair(prompt="p1", pos="a", neg="b", instruction_id="safety_refusal")]
    cfg = {"registry_paper_label": "other", "registry_prompt_pool_ids": {"safety_refusal": "qwen35_4b/pairs_safety_refusal"}}
    monkeypatch.setenv("REGISTRY_CAMPAIGN", CAMPAIGN)
    stmt = t2._registry_statement(cfg, "safety_refusal", instruction=resolve_instruction("safety_refusal"), pairs=pairs)
    assert stmt["campaign"] == CAMPAIGN
    monkeypatch.delenv("REGISTRY_CAMPAIGN")
    assert "campaign" not in t2._registry_statement(cfg, "safety_refusal", instruction=resolve_instruction("safety_refusal"), pairs=pairs)
    assert t2._registry_statement({}, "safety_refusal", instruction=None, pairs=pairs) is None    # no statement: no campaign either


def test_a_re_emission_under_another_campaign_than_the_statements_is_refused(tmp_path, monkeypatch):
    """The bank states the statement's campaign for this object, so the producer's record never states another one: a re-emission
    under a different REGISTRY_CAMPAIGN is a failed emission (visible), nothing in the spool."""
    spool = tmp_path / "spool"
    _emission_on(monkeypatch, spool)
    monkeypatch.setenv("REGISTRY_CAMPAIGN", "sdfbase_full22")
    cell = _cell(tmp_path / "cells", "safety_refusal", "lora_r4")
    t2._registry_publish(cell, si_id="safety_refusal", arm="lora_r4")
    ev = json.loads((cell / "registry_emission.json").read_text())
    assert ev["status"] == "failed" and "registry_statement.campaign" in ev["error"] and CAMPAIGN in ev["error"], ev
    assert _spool_objects(spool) == []


# -- a dry run writes nothing, on success or failure -----------------------------------------------------------------------------


def _bytes_beside(cell: Path) -> dict:
    return {p.name: p.read_bytes() for p in sorted(cell.iterdir()) if p.is_file()}


def test_a_dry_run_writes_nothing_on_success_or_failure(tmp_path, campaign, monkeypatch, capsys):
    """The re-emission preflight's view: the records come back on success; a refusal (a label outside PAPER_LABELS, raised before
    the hook), a campaign that disagrees with the statement, and an undeclared arm each return None. In every case the cell's files
    are byte-identical afterwards — no registry_emission.json, no REGISTRY_EMIT_FAILED, an earlier evidence file untouched."""
    spool = tmp_path / "spool"
    _emission_on(monkeypatch, spool)
    cell = _cell(tmp_path / "cells", "safety_refusal", "lora_r4")
    before = _bytes_beside(cell)
    assert t2._registry_publish(cell, si_id="safety_refusal", arm="lora_r4", dry_run=True)
    assert _bytes_beside(cell) == before and not (cell / "registry_emission.json").exists()
    # a refusal raised before the hook
    meta_path = cell / "train_budget_meta.json"
    meta = json.loads(meta_path.read_text())
    meta["registry_statement"]["paper_label"] = "not_a_label"
    meta_path.write_text(json.dumps(meta, indent=1, sort_keys=True))
    before = _bytes_beside(cell)
    assert t2._registry_publish(cell, si_id="safety_refusal", arm="lora_r4", dry_run=True) is None
    assert _bytes_beside(cell) == before
    assert not (cell / "registry_emission.json").exists() and not (cell / "REGISTRY_EMIT_FAILED").exists()
    assert "dry run refused" in capsys.readouterr().err
    # an earlier evidence file is never touched by a failing dry run (here: a campaign the statement disagrees with)
    meta["registry_statement"]["paper_label"] = "other"
    meta_path.write_text(json.dumps(meta, indent=1, sort_keys=True))
    (cell / "registry_emission.json").write_text('{"status": "emitted", "synthetic": true}\n')
    monkeypatch.setenv("REGISTRY_CAMPAIGN", "sdfbase_full22")
    before = _bytes_beside(cell)
    assert t2._registry_publish(cell, si_id="safety_refusal", arm="lora_r4", dry_run=True) is None
    assert _bytes_beside(cell) == before
    # an undeclared arm: not_emitted on a real run, nothing at all on a dry run
    plain = _cell(tmp_path / "plain", "C16", "psr_kl_all", statement=False)
    before = _bytes_beside(plain)
    assert t2._registry_publish(plain, si_id="C16", arm="psr_kl_all", dry_run=True) is None
    assert _bytes_beside(plain) == before and not (plain / "registry_emission.json").exists()
    assert _spool_objects(spool) == []
