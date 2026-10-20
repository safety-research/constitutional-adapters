"""Every two_si config's single-layer target is valid for the arms it
advertises, on THAT model's architecture.

Two kinds of single-layer arm read ``cfg["layer"]``:

* block-OUTPUT hooks — bidpo (extraction/bidpo.py: one d_model steering
  vector applied through hooks.steering), vector, loreft, reft_r1 — which
  every decoder block carries whatever its mixer (NemotronH's Mamba2 / MLP
  blocks and Qwen3.5's linear-attention blocks included); the only
  requirement is 0 <= layer < n_layers;
* single-layer o_proj edits — lora_r1 and bidpo_lora (RankOneLoRAObject) —
  which need an ATTENTION block at ``layer``: extraction/objects.py refuses
  any other (models.attention_oproj is None there) and names
  models.oproj_layers().

The review read the Nemotron config's ``bidpo`` at layer 20 (an MLP block)
as the second kind and proposed moving it to an attention block. It is the
first kind — methods.ARMS["bidpo"] builds a VectorObject and
_train_bidpo_arm calls train_bidpo — so the layer stands and the finding
was declined. These tests pin the reason, and would catch the REAL form of
the bug (an o_proj arm advertised at a non-attention layer) in any two_si
config: Qwen3.8-27B's layer 30 is a linear-attention block too, so a
lora_r1 / bidpo_lora arm added to either hybrid's class lists fails here
before it fails after the checkpoint load.

Architecture stubs, never downloads: ATTENTION_LAYOUT records each model's
attention-bearing block indices from its pinned config.json
(``layer_types`` / ``hybrid_override_pattern``, read 2026-09-02), and the
stub reproduces exactly the module shape models.attention_oproj probes —
``self_attn.o_proj`` on attention blocks, a q/k/v/o-less mixer elsewhere.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml
from torch import nn

from sisteer.extraction.objects import RankOneLoRAObject
from sisteer.methods import ARMS, VectorObject
from sisteer.models import get_decoder_layers, oproj_layers

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from derive_model_config import MODEL_SPECS  # noqa: E402

CONFIGS = sorted((REPO / "configs").glob("two_si_*.yaml"))

ALL = "all"

#: model id -> (n_layers, indices of blocks that carry an attention o_proj).
#: Read off each pinned config.json 2026-09-02 (no weights): Llama / gemma-4
#: (sliding + full attention alike) / GLM-4.6V-Flash's text decoder are
#: attention at every block; qwen3_5-family stacks are full attention every
#: ``full_attention_interval`` = 4 blocks from index 3, linear attention
#: elsewhere; NemotronH follows its hybrid_override_pattern
#: "M-M-M-MM-M-M*-M-M*-M-M-M*-M-M-MM*-MMM-M-M-" ('*' = attention).
ATTENTION_LAYOUT = {
    "NousResearch/Meta-Llama-3.1-8B-Instruct": (32, ALL),
    "unsloth/Llama-3.2-1B-Instruct": (16, ALL),
    "unsloth/Llama-3.2-3B-Instruct": (28, ALL),
    "google/gemma-4-E2B-it": (35, ALL),
    "google/gemma-4-E4B-it": (42, ALL),
    "zai-org/GLM-4.6V-Flash": (40, ALL),
    "Qwen/Qwen3.5-4B": (32, tuple(range(3, 32, 4))),
    "Qwen/Qwen3.5-9B": (32, tuple(range(3, 32, 4))),
    "Qwen/Qwen3.8-27B": (64, tuple(range(3, 64, 4))),
    "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16": (42, (12, 17, 24, 32)),
    # paper-companion-repo (2026-09-29; config read from the cohort cache):
    # gemma-4-31B's 60 blocks are 50 sliding + 10 full attention, every one
    # with a self_attn.o_proj
    "google/gemma-4-31B": (60, ALL),
}

#: The arms whose object is a single-layer o_proj edit (RankOneLoRAObject)
#: — the ONLY single-layer arms that need attention at ``layer``. Pinned
#: against the registry below, so a new o_proj arm lands here or fails.
OPROJ_SINGLE_LAYER_ARMS = frozenset({"lora_r1", "bidpo_lora"})

_D = 8


class _Attn(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(_D, _D, bias=False)
        self.k_proj = nn.Linear(_D, _D, bias=False)
        self.v_proj = nn.Linear(_D, _D, bias=False)
        self.o_proj = nn.Linear(_D, _D, bias=False)


class _Block(nn.Module):
    """An attention block carries self_attn.o_proj; any other block a
    mixer without q/k/v/o (Mamba2's in_proj/out_proj shape) — what
    models.attention_oproj distinguishes."""

    def __init__(self, attention: bool):
        super().__init__()
        if attention:
            self.self_attn = _Attn()
        else:
            self.mixer = nn.Module()
            self.mixer.in_proj = nn.Linear(_D, 2 * _D, bias=False)
            self.mixer.out_proj = nn.Linear(_D, _D, bias=False)


def layout_key(model_id: str) -> str:
    """A config may pin a LOCAL staging copy of a hub checkpoint (the
    psr-transfer lane's `<root>/models/<org>--<name>` verified copies, e.g.
    two_si_gemma4e2b_eval.yaml since the 2026-09 truncation fix); the layout
    is the hub checkpoint's, so the table is keyed by hub id and a path whose
    basename is `<org>--<name>` resolves to it. Anything else is looked up
    verbatim (and refuses below if unknown)."""
    if model_id in ATTENTION_LAYOUT:
        return model_id
    base = model_id.rstrip("/").rsplit("/", 1)[-1]
    if "--" in base:
        hub = base.replace("--", "/", 1)
        if hub in ATTENTION_LAYOUT:
            return hub
    return model_id


def stub_for(model_id: str) -> nn.Module:
    n, attn = ATTENTION_LAYOUT[layout_key(model_id)]
    attn_set = set(range(n)) if attn == ALL else set(attn)
    m = nn.Module()
    m.model = nn.Module()
    m.model.layers = nn.ModuleList([_Block(i in attn_set) for i in range(n)])
    m.config = SimpleNamespace(hidden_size=_D)
    return m


def _advertised(cfg: dict) -> set:
    """Class A + class C — the arms that read the shared ``layer``. Class B
    reads its own ``class_b.layer`` and is bounds-checked separately."""
    return set(cfg.get("class_a_arms") or []) | set(cfg.get("class_c_arms") or [])


# ---- the table and the stubs -----------------------------------------------


def test_stubs_reproduce_the_recorded_layouts():
    for model, (n, attn) in ATTENTION_LAYOUT.items():
        stub = stub_for(model)
        assert len(get_decoder_layers(stub)) == n, model
        want = list(range(n)) if attn == ALL else list(attn)
        assert oproj_layers(stub) == want, model


def test_layout_table_agrees_with_derive_model_config():
    """Every ModelSpec whose model the table covers agrees on depth, and
    its depth-fraction layer is a real block index."""
    covered = 0
    for slug, spec in MODEL_SPECS.items():
        if spec.model not in ATTENTION_LAYOUT:
            continue
        covered += 1
        assert ATTENTION_LAYOUT[spec.model][0] == spec.n_layers, slug
        assert 0 <= spec.layer < spec.n_layers, slug
    assert covered >= 4


def test_oproj_single_layer_arms_pinned_against_the_registry():
    """Exactly the registry builders that construct a RankOneLoRAObject
    (read off the lambda's referenced globals), and on a stub each does at
    an attention layer — while bidpo's builder yields a VectorObject."""
    by_code = {name for name, spec in ARMS.items()
               if "RankOneLoRAObject" in spec.build.__code__.co_names}
    assert by_code == OPROJ_SINGLE_LAYER_ARMS
    stub = stub_for("nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16")
    attn = oproj_layers(stub)[0]
    for name in sorted(OPROJ_SINGLE_LAYER_ARMS):
        assert isinstance(ARMS[name].build(stub, None, attn),
                          RankOneLoRAObject), name
    assert isinstance(ARMS["bidpo"].build(stub, None, 20), VectorObject)
    assert "bidpo" not in OPROJ_SINGLE_LAYER_ARMS


# ---- every two_si config ---------------------------------------------------


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_single_layer_arms_target_a_valid_block_of_their_model(path):
    cfg = yaml.safe_load(path.read_text())
    layer = cfg.get("layer")
    if layer is None:
        # The generation-lane configs leave it unset on purpose (their
        # comments: derive per model before any fit); the fit paths refuse
        # them loudly (train_two_si_objects int()s it, tune_arm's bidpo
        # adapter refuses None), so there is no target to validate.
        pytest.skip(f"{path.name}: layer unset by design")
    model = cfg["model"]
    assert layout_key(model) in ATTENTION_LAYOUT, (
        f"{path.name}: {model} is not in ATTENTION_LAYOUT — add it from the "
        "pinned config.json (layer_types / hybrid_override_pattern)")
    stub = stub_for(model)
    n = len(get_decoder_layers(stub))
    layer = int(layer)
    assert 0 <= layer < n, (path.name, layer, n)
    cb_layer = (cfg.get("class_b") or {}).get("layer")
    if cb_layer is not None:
        assert 0 <= int(cb_layer) < n, (path.name, "class_b.layer", cb_layer)
    for arm in sorted(_advertised(cfg) & OPROJ_SINGLE_LAYER_ARMS):
        # the constructor IS the check: it raises, naming oproj_layers(),
        # on a block with no attention o_proj
        try:
            RankOneLoRAObject(stub, layer=layer)
        except ValueError as e:
            pytest.fail(f"{path.name}: {arm} at layer {layer} on {model}: "
                        f"{e}")
        assert layer in oproj_layers(stub), (path.name, arm, layer)


def test_nemotron_bidpo_at_the_mlp_depth_fraction_layer_is_a_block_vector():
    """Why the Nemotron config's layer 20 stands: bidpo is a block-output
    steering vector, every NemotronH block carries that hook, and the
    config advertises no o_proj single-layer arm. The o_proj kind WOULD be
    refused at 20 — the bug the review described, for the arms it applies
    to."""
    cfg = yaml.safe_load((REPO / "configs/two_si_nemotron3nano4b.yaml")
                         .read_text())
    assert cfg["layer"] == 20
    assert "bidpo" in cfg["class_a_arms"] and "bidpo" in cfg["class_c_arms"]
    assert not _advertised(cfg) & OPROJ_SINGLE_LAYER_ARMS
    stub = stub_for(cfg["model"])
    assert oproj_layers(stub) == [12, 17, 24, 32]
    assert 20 not in oproj_layers(stub)
    obj = ARMS["bidpo"].build(stub, None, 20)
    assert isinstance(obj, VectorObject)
    # the trainer behind the arm is train_bidpo (extraction/bidpo.py: "a
    # single d_model vector v at one layer"), returned as a VectorObject
    names = ARMS["bidpo"].train.__code__.co_names
    assert "train_bidpo" in names and "VectorObject" in names
    with pytest.raises(ValueError, match=r"\[12, 17, 24, 32\]"):
        RankOneLoRAObject(stub, layer=20)
    # and the block-output hook the vector rides attaches at 20 —
    # hooks.steering on the stub's MLP-shaped block
    from sisteer.hooks import steering

    with steering(stub, {20: torch.zeros(_D)}) as handle:
        assert handle is not None
        assert get_decoder_layers(stub)[20]._forward_hooks
    assert not get_decoder_layers(stub)[20]._forward_hooks
