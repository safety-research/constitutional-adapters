"""The 20-condition all-served wave vocabulary.

Four changes land together and this file pins them as one contract:

1. ``si_irrelevant`` joins the SERVED vocabulary (harness SI id
   ``irrelevant``): the off-target control SI was deliberately outside the
   audit ("no harness cells"), which blocked 6 of the 20 wave conditions
   from the served path — the control is compared on the SAME engine as
   everything else.
2. ``lora_r4`` is the CANONICAL name of the all-layer low-rank LoRA arm
   (the staged objects are rank 4; "all" is implicit). The legacy spelling
   ``lora_r1_all`` survives at the ARTIFACT layer only
   (``methods.LEGACY_ARM_ALIASES``): staged ``<si>__lora_r1_all/`` dirs
   keep loading, their sidecars' recorded legacy ids are accepted as the
   DECLARED alias (never silently), and the upstream fitting lane can keep
   emitting the old name until it switches.
3. ``configs/two_si_gemma4e4b_reduced.yaml`` declares all three wave SIs
   (previously 7/10 wave cells resolved) and ``train_kwargs.default.
   positions: response`` (the staged objects' own positions_at_fit).
4. The EVAL-side default family set is the five-with-headroom wave set
   (bon/pasttense/grok420 static + codeattack/renellm held-out); the floor
   families (aim 0.0%, encoding 0.4%, pap 2.7% baseline ASR — measured)
   are off the default path but stay selectable. The TRAINING-pool lists
   in configs/two_si.yaml are deliberately untouched (fit-time provenance
   of the staged objects).

The headline assertion: all 20 wave cell ids resolve for both models.
"""

from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]

WAVE_ARMS = ("psr_kl_all", "lora_r4")
WAVE_SIS = ("safety_refusal", "si_constitution", "si_irrelevant")
#: the 10 wave cells per model: 1 undefended + 3 prompted + 3 SIs x 2 arms.
WAVE_CELLS = (
    ["none"]
    + [f"prompted__{si}" for si in WAVE_SIS]
    + [f"{si}__{arm}" for arm in WAVE_ARMS for si in WAVE_SIS]
)
WAVE_CONFIGS = ("configs/two_si.yaml", "configs/two_si_gemma4e4b_reduced.yaml")


def _load_cfg(rel: str) -> dict:
    return yaml.safe_load((REPO / rel).read_text())


# --------------------------------------------------------------------------- #
# 1 + 2 + 3: all 20 wave cell ids resolve, for both models                     #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("config_rel", WAVE_CONFIGS)
def test_all_wave_cells_resolve(config_rel):
    """10/10 wave cells resolve from each shipped config (20 conditions
    across the two models). This is the exact resolution the drivers use
    (registry_from_cfg -> by_id), so a KeyError here is the same KeyError
    that killed scale-12 job 231643."""
    from sisteer.eval.defense_cell import by_id, registry_from_cfg

    cfg = _load_cfg(config_rel)
    cells = by_id(registry_from_cfg(cfg, repo_root=REPO))
    missing = [c for c in WAVE_CELLS if c not in cells]
    assert not missing, (
        f"{config_rel}: wave cells failed to resolve: {missing} "
        f"(resolved: {sorted(cells)})")
    # ... and the object cells resolve with cell-id-keyed object paths.
    for cell_id in WAVE_CELLS:
        cell = cells[cell_id]
        if cell.class_ == "A":
            assert cell.object_path is not None
            assert cell.object_path.name == "object.pt"


def test_gemma_config_declares_the_wave():
    cfg = _load_cfg("configs/two_si_gemma4e4b_reduced.yaml")
    assert cfg["si_ids"] == list(WAVE_SIS)
    assert "lora_r4" in cfg["class_a_arms"]
    assert "lora_r1_all" not in cfg["class_a_arms"]
    # change 4: fit-time positions must describe the staged objects
    # (positions_at_fit: response) — an `all` here would let a blind refit
    # produce all-position objects (serving a response-fit psr object at
    # all positions is a measured model-destroyer).
    assert cfg["train_kwargs"]["default"]["positions"] == "response"


def test_llama_config_names_the_canonical_arm():
    cfg = _load_cfg("configs/two_si.yaml")
    assert "lora_r4" in cfg["class_a_arms"]
    assert "lora_r1_all" not in cfg["class_a_arms"]


# --------------------------------------------------------------------------- #
# 1: the served-path translation for the six previously-blocked conditions     #
# --------------------------------------------------------------------------- #


def test_harness_translation_for_previously_blocked_cells():
    from sisteer.capeval.petri import registry_adapter as ra

    # the six blocked conditions' cells (x 2 models): ir_psrkl, ir_lora,
    # prompted__si_irrelevant — plus round-trips for the lora_r4 cells.
    for tt_id, harness_id in [
        ("si_irrelevant__psr_kl_all", "si_irrelevant__psr_kl_all"),
        ("si_irrelevant__lora_r4", "si_irrelevant__lora_r4"),
        ("prompted__si_irrelevant", "prompted__irrelevant"),
        ("safety_refusal__lora_r4", "si_safety__lora_r4"),
        ("si_constitution__lora_r4", "si_constitution__lora_r4"),
        ("safety_refusal__psr_kl_all", "si_safety__psr_kl_all"),
    ]:
        assert ra.harness_cell_id(tt_id) == harness_id
        assert ra.traintest_cell_id(harness_id) == tt_id


def test_parse_cell_id_accepts_irrelevant_and_lora_r4():
    from sisteer.capeval.petri import defense_cells as dc

    p = dc.parse_cell_id("prompted__irrelevant")
    assert (p.cell_class, p.si_id) == ("prompted", "irrelevant")
    p = dc.parse_cell_id("si_irrelevant__lora_r4")
    assert (p.cell_class, p.si_id, p.method) == ("class_a", "irrelevant",
                                                 "lora_r4")
    p = dc.parse_cell_id("stacked__irrelevant__psr_kl_all")
    assert (p.cell_class, p.si_id, p.method) == ("stacked", "irrelevant",
                                                 "psr_kl_all")
    # the LEGACY arm spelling is an ARTIFACT-layer alias, never a harness
    # id — one method must never wear two harness identities.
    with pytest.raises(dc.DefenseCellError):
        dc.parse_cell_id("si_safety__lora_r1_all")


def test_manifest_records_the_irrelevant_prefix_hash():
    from sisteer.capeval import manifest as cm
    from sisteer.data.instructions import BY_ID

    spec = cm.load_petri_spec(
        REPO / "configs" / "capeval" / "manifests" / "petri_twosi.yaml")
    assert "prompted__irrelevant" in spec.cells
    text = BY_ID["si_irrelevant"].text
    assert spec.prompted_prefix_hashes["irrelevant"] == cm.sha256_hex16(
        text.encode("utf-8"))
    # the SI-variant behavior dirs exist with the EXACT bytes (H8), for
    # every behavior in the audit.
    from sisteer.capeval.petri import scenarios as psc

    assert psc.SI_VARIANT_DIRS == cm.PETRI_SI_VARIANT_DIRS
    for b in spec.behaviors:
        front, body = psc.parse_behavior_md(
            (REPO / "configs" / "capeval" / "petri" / b /
             "PROMPTED_IRRELEVANT" / "BEHAVIOR.md").read_bytes())
        assert front["target_sysprompt_prefix"] == text
        cfront, cbody = psc.parse_behavior_md(
            (REPO / "configs" / "capeval" / "petri" / b /
             "BEHAVIOR.md").read_bytes())
        assert body == cbody
        assert front["modality"] == cfront["modality"]
    # the on-disk inventory petri_run loads matches the spec-implied
    # expected keys exactly (extra/missing keys refuse at hash time).
    import scripts.petri_run as pr

    files = pr.load_behavior_files(spec)
    assert sorted(files) == cm.petri_expected_behavior_keys(spec)


def test_adapter_requires_only_in_play_sis_declared(tmp_path):
    """A config narrower than the FULL mapped vocabulary is fine as long
    as it declares every SI the manifest's cells put IN PLAY — adding
    si_irrelevant to the map must not refuse a deliberate 2-SI serving
    config whose cells never touch it."""
    import dataclasses

    from sisteer.capeval import manifest as cm
    from sisteer.capeval.petri.registry_adapter import (
        RegistryAdapterError, traintest_cell_source)

    spec = cm.load_petri_spec(
        REPO / "configs" / "capeval" / "manifests" / "petri_twosi.yaml")
    two_si = {"defenses_dir": str(tmp_path / "cells"),
              "si_ids": ["safety_refusal", "si_constitution"]}
    cfg_path = tmp_path / "two_si.yaml"
    cfg_path.write_text(yaml.safe_dump(two_si))

    # cells that only touch safety/constitution -> the 2-SI config is fine
    narrow = dataclasses.replace(
        spec, cells=["none", "prompted__safety", "si_constitution__lora_r4"])
    traintest_cell_source(narrow, repo_root=REPO, config_path=cfg_path,
                          defenses_dir=tmp_path / "cells")

    # an in-play irrelevant cell against the same 2-SI config refuses loudly
    in_play = dataclasses.replace(
        narrow, cells=["none", "prompted__irrelevant"])
    with pytest.raises(RegistryAdapterError, match="si_irrelevant"):
        traintest_cell_source(in_play, repo_root=REPO, config_path=cfg_path,
                              defenses_dir=tmp_path / "cells")


# --------------------------------------------------------------------------- #
# 2: the lora_r4 <-> lora_r1_all artifact alias                                 #
# --------------------------------------------------------------------------- #


def test_legacy_cell_id_derivation():
    from sisteer.eval.defense_cell import legacy_cell_id

    assert legacy_cell_id("safety_refusal__lora_r4") == \
        "safety_refusal__lora_r1_all"
    assert legacy_cell_id("si_irrelevant__lora_r4") == \
        "si_irrelevant__lora_r1_all"
    assert legacy_cell_id("lora_r4") == "lora_r1_all"
    assert legacy_cell_id("safety_refusal__psr_kl_all") is None
    assert legacy_cell_id("none") is None
    assert legacy_cell_id("prompted__safety_refusal") is None


def test_object_path_resolves_legacy_dir(tmp_path):
    """Canonical dir wins when its artifact exists; the staged LEGACY dir
    (`<si>__lora_r1_all/`) is resolved when the canonical one holds no
    artifact; with neither present the CANONICAL path is returned so
    ABSENT reporting names the path new fits will write."""
    from sisteer.eval.defense_cell import object_path_for

    cell = "safety_refusal__lora_r4"
    canonical = tmp_path / cell / "object.pt"
    legacy = tmp_path / "safety_refusal__lora_r1_all" / "object.pt"

    # neither exists -> canonical (the forward-looking ABSENT path)
    assert object_path_for(tmp_path, cell) == canonical

    # only the staged legacy artifact exists -> the legacy path
    legacy.parent.mkdir(parents=True)
    legacy.write_bytes(b"staged rank-4 blob")
    assert object_path_for(tmp_path, cell) == legacy

    # canonical lands (a new fit under the new name) -> canonical wins
    canonical.parent.mkdir(parents=True)
    canonical.write_bytes(b"new fit")
    assert object_path_for(tmp_path, cell) == canonical

    # non-aliased arms never look sideways
    other = tmp_path / "safety_refusal__psr_kl_all" / "object.pt"
    assert object_path_for(tmp_path, "safety_refusal__psr_kl_all") == other


def test_registry_resolves_staged_legacy_artifact(tmp_path):
    """registry_from_cfg over a store holding ONLY the staged legacy dirs
    resolves the lora_r4 cells to those artifacts — reading existing
    lora_r1_all artifacts is not broken by the rename."""
    from sisteer.eval.defense_cell import by_id, registry_from_cfg

    for si in WAVE_SIS:
        d = tmp_path / f"{si}__lora_r1_all"
        d.mkdir()
        (d / "object.pt").write_bytes(b"blob")
    cfg = {"defenses_dir": str(tmp_path), "si_ids": list(WAVE_SIS),
           "class_a_arms": ["psr_kl_all", "lora_r4"], "class_b_arms": []}
    cells = by_id(registry_from_cfg(cfg))
    for si in WAVE_SIS:
        cell = cells[f"{si}__lora_r4"]
        assert cell.object_path == tmp_path / f"{si}__lora_r1_all" / "object.pt"
        assert cell.object_path.is_file()


def test_budget_sidecar_accepts_exactly_the_recorded_alias(tmp_path, capsys):
    """The staged dirs' train_budget_meta.json records the LEGACY ids
    (arm=lora_r1_all / cell_id=<si>__lora_r1_all). The provenance check
    accepts EXACTLY that spelling — announced, table-driven — and still
    refuses any other mismatch, so the directory/metadata name mismatch
    can never pass silently."""
    import json

    from sisteer.eval.defense_cell import (by_id, registry_from_cfg,
                                           sha256_file)
    from scripts.two_si_run import _check_budget_sidecar

    d = tmp_path / "safety_refusal__lora_r1_all"
    d.mkdir()
    obj = d / "object.pt"
    obj.write_bytes(b"staged rank-4 blob")
    cfg = {"defenses_dir": str(tmp_path), "si_ids": ["safety_refusal"],
           "class_a_arms": ["lora_r4"], "class_b_arms": [],
           "completion_tokens": 512}
    cell = by_id(registry_from_cfg(cfg))["safety_refusal__lora_r4"]
    assert cell.object_path == obj

    meta = {"cell_id": "safety_refusal__lora_r1_all", "arm": "lora_r1_all",
            "completion_tokens": 512, "object_sha256": sha256_file(obj)}
    (d / "train_budget_meta.json").write_text(json.dumps(meta))
    _check_budget_sidecar(cell, cfg)  # accepted, with an announcement
    assert "LEGACY spelling" in capsys.readouterr().out

    # any OTHER cell_id still refuses — the alias is a single spelling,
    # not a licence.
    meta["cell_id"] = "si_constitution__lora_r1_all"
    (d / "train_budget_meta.json").write_text(json.dumps(meta))
    with pytest.raises(SystemExit, match="corrupted"):
        _check_budget_sidecar(cell, cfg)


def test_lora_r4_registered_everywhere_lora_r1_all_was():
    """The cc4487f registration sites, under the canonical name."""
    from sisteer.eval.decode_capability import DECODE_ONLY_CAPABILITY
    from sisteer.eval.defense_cell import CLASS_A_ARMS
    from sisteer.eval.train_budget import _ARM_TRAINER_SOURCES
    from sisteer.extraction.lockstep import (_LOCKSTEP_COMPATIBLE,
                                             _MASKABLE_ARMS)
    from sisteer.methods import ARMS, LEGACY_ARM_ALIASES, canonical_arm

    assert "lora_r4" in ARMS and "lora_r1_all" in ARMS  # both spellings load
    assert ARMS["lora_r4"].train is ARMS["lora_r1_all"].train
    assert LEGACY_ARM_ALIASES == {"lora_r1_all": "lora_r4"}
    assert canonical_arm("lora_r1_all") == "lora_r4"
    assert canonical_arm("psr_kl_all") == "psr_kl_all"
    assert "lora_r4" in CLASS_A_ARMS and "lora_r1_all" not in CLASS_A_ARMS
    assert DECODE_ONLY_CAPABILITY["lora_r4"][:2] == \
        DECODE_ONLY_CAPABILITY["lora_r1_all"][:2]
    assert (_ARM_TRAINER_SOURCES["lora_r4"] ==
            _ARM_TRAINER_SOURCES["lora_r1_all"])
    assert "lora_r4" in _LOCKSTEP_COMPATIBLE and "lora_r4" in _MASKABLE_ARMS


# --------------------------------------------------------------------------- #
# 4: the eval-side default family set (floor families off the default path)    #
# --------------------------------------------------------------------------- #

FLOOR_FAMILIES = ("pap", "encoding", "aim")  # the three floor families
WAVE_FAMILY_SET = {"bon", "pasttense", "codeattack", "renellm", "grok420"}


def test_eval_defaults_are_the_five_wave_families():
    from sisteer.eval import sweep

    assert (set(sweep.EVAL_DEFAULT_STATIC_FAMILIES)
            | set(sweep.EVAL_DEFAULT_HELD_OUT_FAMILIES)) == WAVE_FAMILY_SET
    # the floor families are OFF every default path ("no headroom" — aim
    # jailbreaks 0/259 undefended; a defense cannot lower a zero)...
    for fam in FLOOR_FAMILIES:
        assert fam not in sweep.EVAL_DEFAULT_STATIC_FAMILIES
        assert fam not in sweep.EVAL_DEFAULT_HELD_OUT_FAMILIES
    # ...but stay SELECTABLE (the universes are unchanged as validation
    # sets — explicit selection is a deliberate act, not a default).
    for fam in FLOOR_FAMILIES:
        assert fam in sweep.STATIC_FAMILIES
    assert "grok420" in sweep.STATIC_FAMILIES  # newly selectable
    # the adaptive (iterative-attacker) leg stays OFF by default — the
    # second tier is reserved, not wired.
    assert sweep.Condition(label="x", cell="none").adaptive == ()


def test_tsv_star_expands_to_the_wave_defaults(tmp_path):
    from sisteer.eval import sweep

    tsv = tmp_path / "conditions.tsv"
    tsv.write_text("full\tnone\t\tall\t*\t-\t-\n"
                   "explicit\tnone\t\tall\tpap,aim\t-\t-\n")
    full, explicit = sweep.load_conditions(tsv)
    assert full.static == sweep.EVAL_DEFAULT_STATIC_FAMILIES
    assert full.held_out == sweep.EVAL_DEFAULT_HELD_OUT_FAMILIES
    # naming a floor family explicitly still works — the default is a
    # default, not a ban.
    assert explicit.static == ("pap", "aim")


def test_training_pool_lists_untouched():
    """The two_si.yaml pool composition is the fit-time PROVENANCE of the
    staged objects and the frozen pool manifest — the eval-default change
    must never rewrite it (the two concerns used to be conflated)."""
    cfg = _load_cfg("configs/two_si.yaml")
    # the staged wave's six-family composition (the training trunk's
    # 2026-08-28 goal-level regime: five headroom transforms + the
    # wildjailbreak corpus) — test_stage3_pool pins the same list
    six = ["bon", "pasttense", "codeattack", "renellm", "grok420",
           "wildjailbreak_harmful"]
    assert cfg["pool"]["static_families"] == six
    assert cfg["pool"]["train_families"] == six


def test_holdout_is_goal_level_not_family_level():
    """Family-level holdout is RETIRED: the five
    wave families are identical between train and eval BY DESIGN, and the
    goal split is the holdout. Shared families must not refuse; a shared-
    family pool with NO goal split must (that is no holdout at all); and
    goal overlap always refuses."""
    from sisteer.eval.matched_pool import (MatchedPoolError, PoolSpec,
                                           build_matched_pool)

    def spec(**kw):
        base = dict(
            si_ids=("safety_refusal", "si_constitution"),
            train_families=("bon", "codeattack"),
            eval_families=("codeattack", "renellm"),
            train_goal_ids=("g1",), eval_goal_ids=("g2",),
            static_families=frozenset({"bon", "codeattack"}))
        base.update(kw)
        return PoolSpec(**base)

    # shared families + a real goal split: legal, and the manifest RECORDS
    # the sharing as design rather than claiming disjointness
    manifest = build_matched_pool([], spec())
    fam = manifest["disjointness"]["family"]
    assert fam["disjoint"] is False
    assert fam["overlap"] == ["codeattack"]
    assert "GOAL split" in fam["note"]

    # shared families + NO eval goals = nothing held out at all -> refuse
    with pytest.raises(MatchedPoolError, match="nothing is held out"):
        build_matched_pool([], spec(eval_goal_ids=()))

    # the goal split IS the holdout: goal overlap always refuses
    with pytest.raises(MatchedPoolError, match="goal"):
        build_matched_pool([], spec(train_goal_ids=("g1", "g2")))

    # the retired flag is gone from PoolSpec (nothing may fail on the
    # abandoned family rule) and from the gemma config
    import dataclasses

    assert "family_disjointness" not in {
        f.name for f in dataclasses.fields(PoolSpec)}
    cfg = _load_cfg("configs/two_si_gemma4e4b_reduced.yaml")
    assert "family_disjointness" not in cfg["pool"]
