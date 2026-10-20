"""The psr_xfer_* transferred-PSR arms: registration + the eval width knob.

Pins the pt829 wave's two changes:

1. FOUR-PLACE ARM REGISTRATION — the cross-model transferred arms
   (psr_xfer_from_{e2b,e4b}{,_rw1}) must be registered coherently in every
   surface that dispatches on an arm name, or a wave fails four different
   ways (unknown arm at cell enumeration, unknown serving semantics at the
   benign leg, unknown kind on the lens path, prompt-only footprint at the
   phase probe). Their object kind is 'psr' — the artifacts are ordinary
   PSRObject blobs, same hook site as psr_kl_all — so the EXISTING psr
   lens adapter must cover them (no new hook site).

2. eval_gen_batch_size — cfg-driven HF generation width for the non-BoN
   JB legs. batch_size is RECORDED serving identity, never a pairing axis;
   the default 8 keeps every config-silent invocation byte-identical.
"""

import importlib
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(_REPO / "scripts"))

XFER_ARMS = (
    "psr_xfer_from_e2b",
    "psr_xfer_from_e2b_rw1",
    "psr_xfer_from_e4b",
    "psr_xfer_from_e4b_rw1",
)


def test_xfer_arms_registered_in_methods():
    from sisteer.methods import ARMS, object_artifact_name

    for arm in XFER_ARMS:
        assert arm in ARMS, f"{arm} missing from methods.ARMS"
        # torch-blob artifact, the psr container convention
        assert object_artifact_name(arm) == "object.pt"


def test_xfer_arms_are_class_a_everywhere():
    from sisteer.capeval.petri.defense_cells import CLASS_A_METHODS
    from sisteer.eval.defense_cell import CLASS_A_ARMS

    for arm in XFER_ARMS:
        assert arm in CLASS_A_ARMS, f"{arm} missing from CLASS_A_ARMS"
        assert arm in CLASS_A_METHODS, f"{arm} missing from CLASS_A_METHODS"


def test_xfer_arms_dispatch_to_the_psr_adapter():
    from sisteer.capeval.petri.registry_adapter import ARM_OBJECT_KINDS
    from sisteer.capeval.serving import lens_adapters

    for arm in XFER_ARMS:
        assert ARM_OBJECT_KINDS.get(arm) == "psr", (
            f"{arm} must map to object kind 'psr' (ordinary PSRObject blob)")
    # the kind's adapter already exists — the whole point of kind 'psr'
    assert "psr" in lens_adapters._ADAPTERS


def test_xfer_trainer_refuses():
    """The trainer must refuse: a transferred artifact is never fit in-repo,
    and a real trainer could silently overwrite a transferred object with a
    natively-fit one under the transfer arm's name."""
    from sisteer.methods import ARMS

    for arm in XFER_ARMS:
        with pytest.raises(RuntimeError, match="TRANSFERRED"):
            ARMS[arm].train(None, None, None, [], layer=0)


def test_xfer_decode_capability_matches_psr():
    """Same mechanism as psr_kl_all => same decode-only verdict; without an
    entry the footprint falls to the prompt-only default and every
    mask=generated serve refuses at the phase probe's subset test."""
    from sisteer.eval.decode_capability import DECODE_ONLY_CAPABILITY

    ref = DECODE_ONLY_CAPABILITY["psr_kl_all"]
    for arm in XFER_ARMS:
        got = DECODE_ONLY_CAPABILITY[arm]
        assert got[:4] == ref[:4], (
            f"{arm} decode-capability verdict {got[:4]} != psr_kl_all's "
            f"{ref[:4]}")


def test_pt829_conditions_tsvs_parse():
    """The wave TSVs must parse through the sweep's own loader, with the
    masks the objects' positions_at_fit=response demands."""
    from sisteer.eval import sweep

    for tsv, xfer_cell in (
        ("configs/swp/conditions_pt829_e4b.tsv",
         "safety_refusal__psr_xfer_from_e2b"),
        ("configs/swp/conditions_pt829_e2b.tsv",
         "safety_refusal__psr_xfer_from_e4b"),
    ):
        conds = {c.label: c for c in sweep.load_conditions(tsv)}
        assert set(conds) == {"none", "p_sr", "psr_gen", "xfer_gen"}
        assert conds["none"].mask == "all"
        assert conds["p_sr"].cell == "prompted__safety_refusal"
        assert conds["psr_gen"].cell == "safety_refusal__psr_kl_all"
        assert conds["psr_gen"].mask == "generated"
        assert conds["xfer_gen"].cell == xfer_cell
        assert conds["xfer_gen"].mask == "generated"
        for c in conds.values():
            assert c.target_engine == "hf"
            assert c.benign == ()  # capabilities leg owns benign
            assert c.adaptive == ()


def test_configs_carry_the_xfer_arms():
    """Registration place 2: the config's class_a_arms (registry_from_cfg
    enumerates cells from THIS list — an arm absent here silently never
    resolves as a cell)."""
    import yaml

    e4b = yaml.safe_load(open("configs/two_si_gemma4e4b_reduced.yaml"))
    e2b = yaml.safe_load(open("configs/two_si_gemma4e2b_eval.yaml"))
    assert "psr_xfer_from_e2b" in e4b["class_a_arms"]
    assert "psr_xfer_from_e4b" in e2b["class_a_arms"]
    # source-direction hygiene: no self-transfers
    assert "psr_xfer_from_e4b" not in e4b["class_a_arms"]
    assert "psr_xfer_from_e2b" not in e2b["class_a_arms"]
    # the eval width knob rides both wave configs
    assert e4b["eval_gen_batch_size"] == 64
    assert e2b["eval_gen_batch_size"] == 64


def test_eval_gen_batch_size_reaches_the_defense():
    """cfg['eval_gen_batch_size'] -> the built defense's own batch_size
    (run_static chunks generation at it); silent config -> the module
    default 8, byte-identical to every existing invocation."""
    from sisteer.eval import sweep

    drv = importlib.import_module("unified_sweep_run")
    cond = sweep.Condition(label="t", cell="none", mask="all")
    d64 = drv._build_condition_defense(None, None, cond,
                                       {"scale": 1.0,
                                        "eval_gen_batch_size": 64})
    assert d64.batch_size == 64
    d8 = drv._build_condition_defense(None, None, cond, {"scale": 1.0})
    assert d8.batch_size == 8
