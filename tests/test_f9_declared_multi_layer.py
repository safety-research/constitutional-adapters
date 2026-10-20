"""F9 with a DECLARED multi-layer object.

The Nemotron-3-Nano-4B LoRA objects are fitted on the hybrid's four attention layers
([12, 17, 24, 32]); the benign arm spec's single ``layer`` (the config's placeholder, 20) refused
them at F9 and the error text forbade picking a layer. The rule is now ONE function,
``manifest.f9_layer_problem``: a declared set must EQUAL the observed layers; without a declaration
the single-layer membership rule stands. Three cases, at both F9 sites (sidecar / loaded object),
plus the config seam (``arm_spec_for_cell`` reads ``multi_layer_arms``) and the hash invariance."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from sisteer.capeval import manifest as cm
from sisteer.capeval.manifest import AblationSpec
from sisteer.capeval.serving import resolved_spec as rs
from sisteer.eval import cell_controls as cc

OBJ_LAYERS = [12, 17, 24, 32]


class _Blob:
    """A torch-blob-like object: reports its own layers, hydrates nothing."""
    kind = "lora_subtraction"

    def __init__(self, layers):
        self.layers = list(layers)
        self.positions = "all"


def _spec(tmp_path, **kw):
    art = tmp_path / "object.pt"
    art.write_bytes(b"not a real object; the hash covers these bytes")
    base = dict(name="constv2-mix-full__lora_r4_ce", kind="sisteer_object", artifact=str(art), layer=20,
                coeffs=[1.0])
    base.update(kw)
    return AblationSpec(**base)


def test_declared_set_equal_to_the_objects_layers_passes_at_both_sites(tmp_path):
    spec = _spec(tmp_path, layers=[32, 12, 24, 17])                  # order-free
    assert cm.f9_layer_problem(spec, OBJ_LAYERS) is None
    cm.validate_layer_against_sidecar(spec, {"layers": OBJ_LAYERS})   # no raise
    resolved, ih = rs.resolve_and_hash(spec, _Blob(OBJ_LAYERS), 1.0, RuntimeError)
    assert resolved["layers"] == OBJ_LAYERS and len(ih) == 16


def test_a_placeholder_single_layer_against_a_multi_layer_object_still_refuses(tmp_path):
    spec = _spec(tmp_path)                                            # layer 20, no declaration
    assert "manifest layer 20 does not match" in cm.f9_layer_problem(spec, OBJ_LAYERS)
    with pytest.raises(cm.ManifestError, match="manifest layer 20 does not match"):
        cm.validate_layer_against_sidecar(spec, {"layers": OBJ_LAYERS})
    with pytest.raises(RuntimeError, match="NAMED DESIGN GAP") as e:
        rs.resolve_and_hash(spec, _Blob(OBJ_LAYERS), 1.0, RuntimeError)
    assert "multi_layer_arms" in str(e.value)                         # the remedy is named, not a layer
    assert cm.f9_layer_problem(spec, [20, 21]) is None                # membership rule unchanged


@pytest.mark.parametrize("declared", [[12, 17, 24], [12, 17, 24, 32, 40], [20], [12, 17, 24, 33]])
def test_a_declared_set_that_differs_from_the_object_refuses_naming_both(tmp_path, declared):
    spec = _spec(tmp_path, layers=declared)
    problem = cm.f9_layer_problem(spec, OBJ_LAYERS)
    assert "declared layer set" in problem and str(sorted(declared)) in problem and str(OBJ_LAYERS) in problem
    with pytest.raises(cm.ManifestError, match="declared layer set"):
        cm.validate_layer_against_sidecar(spec, {"layers": OBJ_LAYERS})
    with pytest.raises(RuntimeError, match="declared layer set"):
        rs.resolve_and_hash(spec, _Blob(OBJ_LAYERS), 1.0, RuntimeError)


def test_the_declaration_changes_no_hash(tmp_path):
    """The identity already carries the object's own layers; declaring them adds nothing."""
    undeclared = _spec(tmp_path, layer=12)                            # a member: passes the old rule
    declared = _spec(tmp_path, layers=OBJ_LAYERS)
    r1, h1 = rs.resolve_and_hash(undeclared, _Blob(OBJ_LAYERS), 1.0, RuntimeError)
    r2, h2 = rs.resolve_and_hash(declared, _Blob(OBJ_LAYERS), 1.0, RuntimeError)
    assert r1 == r2 and h1 == h2 and r1["layers"] == OBJ_LAYERS and "declared" not in str(r1)


def test_arm_spec_for_cell_reads_the_declaration_from_the_config_and_refuses_a_malformed_one(tmp_path):
    art = tmp_path / "constv2-mix-full__lora_r4_ce" / "object.pt"
    art.parent.mkdir()
    art.write_bytes(b"x")
    cell = SimpleNamespace(id="constv2-mix-full__lora_r4_ce", arm="lora_r4_ce", si_id="constv2-mix-full",
                           class_="A", object_path=str(art), stacked=False)
    plain = cc.arm_spec_for_cell(cell, {"layer": 20, "scale": 1.0})
    assert "layers" not in plain and plain["layer"] == 20
    cfg = {"layer": 20, "scale": 1.0, cc.MULTI_LAYER_ARMS_KEY: {cell.id: OBJ_LAYERS}}
    spec = cc.arm_spec_for_cell(cell, cfg)
    assert spec["layers"] == OBJ_LAYERS and spec["layer"] == 20   # the placeholder stays; the set is declared
    assert AblationSpec(**spec).layers == OBJ_LAYERS
    other = SimpleNamespace(**{**vars(cell), "id": "constv2ctl2-full__lora_r4_ce"})
    assert "layers" not in cc.arm_spec_for_cell(other, cfg)       # only the named cell
    for bad in ([], "12,17", [12, "17"], [True], {"a": 1}):
        with pytest.raises(ValueError, match="non-empty list of layer indices"):
            cc.arm_spec_for_cell(cell, {**cfg, cc.MULTI_LAYER_ARMS_KEY: {cell.id: bad}})


def _petri_free_manifest_yaml(tmp_path, layers_line):
    """A minimal capeval manifest document (the loader's own required keys) with ONE
    sisteer_object arm declaring ``layers`` — what ``configs/tinycap/bcbw_nemotron*.yaml``'s
    multi_layer_arms become at the manifest seam."""
    art = tmp_path / "object.pt"
    art.write_bytes(b"bytes")
    (tmp_path / "object.json").write_text('{"layers": [12, 17, 24, 32], "kind": "lora_subtraction"}')
    return f"""
sweep: s
mode: exploratory
base: m
base_revision: r
engine: served
tiers: [a]
ablations:
  - name: constv2-mix-full__lora_r4_ce
    kind: sisteer_object
    artifact: {art}
    layer: 20
{layers_line}
    coeffs: [1.0]
"""


def test_a_yaml_manifest_carries_the_declared_layers_and_f9_refuses_a_trimmed_set(tmp_path):
    """Code review: ``_build_manifest`` dropped the accepted ``layers`` key, so a
    manifest declaring ``layers: [12, 17, 24, 32]`` loaded with ``layers=None`` and F9 fell back
    to single-layer membership. The YAML round trip must carry the exact set; a trimmed set
    must refuse at F9 naming both sides; a malformed declaration refuses at load."""
    import yaml

    raw = yaml.safe_load(_petri_free_manifest_yaml(tmp_path, "    layers: [12, 17, 24, 32]"))
    m = cm._build_manifest(raw, "m.yaml")
    (arm,) = m.ablations
    assert arm.layers == OBJ_LAYERS and arm.layer == 20
    assert cm.f9_layer_problem(arm, OBJ_LAYERS) is None
    cm.validate_layer_against_sidecar(arm, {"layers": OBJ_LAYERS})
    # a trimmed declaration through the same YAML path refuses at F9 — never trimmed to pass
    trimmed = cm._build_manifest(yaml.safe_load(_petri_free_manifest_yaml(tmp_path, "    layers: [12, 17, 24]")), "m.yaml")
    problem = cm.f9_layer_problem(trimmed.ablations[0], OBJ_LAYERS)
    assert problem and "[12, 17, 24]" in problem and "[12, 17, 24, 32]" in problem and "never trim" in problem
    # without a declaration the single-layer rule stands (the placeholder 20 is not a member: refused)
    plain = cm._build_manifest(yaml.safe_load(_petri_free_manifest_yaml(tmp_path, "")), "m.yaml")
    assert plain.ablations[0].layers is None and "layer 20" in cm.f9_layer_problem(plain.ablations[0], OBJ_LAYERS)
    # the declaration is typed at load: a string, an empty list, a bool member or a repeat refuse
    for bad in ("    layers: '12,17'", "    layers: []", "    layers: [12, true]", "    layers: [12, 12, 17]"):
        with pytest.raises(cm.ManifestError, match="layers"):
            cm._build_manifest(yaml.safe_load(_petri_free_manifest_yaml(tmp_path, bad)), "m.yaml")
